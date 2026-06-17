"""
Build an interactive map of ERCOT-routed wind plants for visual BA-mapping audit.

One marker per plant (eia_id), sized by capacity, colored by BA assignment source:
  blue   = EIA-860 → ERCOT      (high confidence)
  orange = state-fallback → ERCOT  (medium confidence — eyeball these)
  red    = manual override → ERCOT (rare, you wrote it)
  gray   = non-ERCOT (SWPP/MISO/etc) — shown for spatial context

Click a marker for plant details. Layers can be toggled via the layer control.
Texas Panhandle/SPP-county warning fires on any ERCOT-routed plant that sits in
a county that's actually SPP territory.

Usage:
    python map_ercot_wind.py
    python map_ercot_wind.py --states TX OK NM
    python map_ercot_wind.py --output ercot_wind_map.html
    python map_ercot_wind.py --hide-non-ercot          # ERCOT-only view
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import folium
    from folium.plugins import Fullscreen, MeasureControl
except ImportError:
    sys.exit("folium is required. Install with: pip install folium")

# Project modules. Run from scripts/ so these resolve.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from turbine_inventory import load_uswtdb
    from aggregation import attach_ba_iso, load_eia860_ba_map
except ImportError as e:
    sys.exit(
        f"Could not import project modules: {e}\n"
        f"Run this script from the scripts/ directory."
    )


# Counties in Texas that are SPP territory, not ERCOT. Used for the warning
# banner on plant popups. Sourced from ERCOT/SPP boundary maps; the Panhandle
# and a strip of northwest TX counties are SWPP.
TX_SPP_COUNTIES = {
    "Hartley", "Moore", "Hutchinson", "Roberts", "Hemphill",
    "Oldham", "Potter", "Carson", "Gray", "Wheeler",
    "Deaf Smith", "Randall", "Armstrong", "Donley", "Collingsworth",
    "Parmer", "Castro", "Swisher", "Briscoe", "Hall", "Childress",
    "Bailey", "Lamb", "Hale", "Floyd", "Motley", "Cottle", "Hardeman",
}

# Map ba_source values from aggregation.attach_ba_iso to display colors.
SOURCE_COLOR = {
    "eia860":         "blue",
    "state_fallback": "orange",
    "override":       "red",
}


def color_for(row: pd.Series) -> str:
    if row.get("ba_code") != "ERCO":
        return "gray"
    return SOURCE_COLOR.get(row.get("ba_source"), "blue")


def radius_for(mw: float) -> float:
    if pd.isna(mw) or mw <= 0:
        return 4.0
    return max(4.0, min(28.0, 2.0 + (mw ** 0.5) * 0.9))


def popup_html(row: pd.Series) -> str:
    name = row.get("p_name", "Unknown")
    eia_id = row.get("eia_id")
    eia_str = f"{int(eia_id)}" if pd.notna(eia_id) else "—"
    ba = row.get("ba_code", "?") or "?"
    src = row.get("ba_source", "?") or "?"
    state = row.get("t_state", "?")
    county = row.get("t_county", None)
    county_str = f"{county}, {state}" if (county and pd.notna(county)) else state
    mw = row.get("p_cap_mw", 0)
    n_turb = int(row.get("n_turbines", 0))
    p_year = row.get("p_year")
    if pd.notna(p_year):
        try:
            p_year = int(p_year)
        except (TypeError, ValueError):
            pass
    else:
        p_year = "—"

    warning = ""
    if (state == "TX" and county and county in TX_SPP_COUNTIES
            and ba == "ERCO"):
        warning = (
            '<div style="margin-top:8px;padding:6px 8px;background:#fff4e6;'
            'border-left:3px solid #ff9933;font-size:11px;line-height:1.4;">'
            "⚠ Located in a Texas Panhandle/SPP county but currently routed "
            "to ERCOT. Verify and consider an override."
            "</div>"
        )

    return f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;font-size:13px;min-width:240px;">
      <div style="font-weight:600;font-size:14px;margin-bottom:5px;">{name}</div>
      <table style="border-collapse:collapse;width:100%;font-size:12px;">
        <tr><td style="color:#666;padding:1px 6px 1px 0;">EIA ID</td><td>{eia_str}</td></tr>
        <tr><td style="color:#666;padding:1px 6px 1px 0;">Capacity</td><td>{mw:,.1f} MW</td></tr>
        <tr><td style="color:#666;padding:1px 6px 1px 0;">Turbines</td><td>{n_turb}</td></tr>
        <tr><td style="color:#666;padding:1px 6px 1px 0;">Year</td><td>{p_year}</td></tr>
        <tr><td style="color:#666;padding:1px 6px 1px 0;">Location</td><td>{county_str}</td></tr>
        <tr><td style="color:#666;padding:1px 6px 1px 0;">BA</td><td><b>{ba}</b></td></tr>
        <tr><td style="color:#666;padding:1px 6px 1px 0;">Source</td><td>{src}</td></tr>
      </table>
      {warning}
    </div>
    """


def aggregate_to_plants(turbines: pd.DataFrame) -> pd.DataFrame:
    """Collapse turbine rows to one row per plant.

    Turbines with a real eia_id collapse to one marker per plant. Turbines
    with NaN eia_id stay separate (one marker per turbine) — otherwise they
    all get lumped into a single phantom mega-plant labeled with whichever
    name happens to sort first.
    """
    has_county = "t_county" in turbines.columns

    has_eia = turbines["eia_id"].notna()
    keyed = turbines[has_eia].copy()
    orphan = turbines[~has_eia].copy()

    agg_dict = {
        "p_name":     ("p_name", "first"),
        "t_state":    ("t_state", "first"),
        "p_cap_mw":   ("t_cap", lambda s: s.sum() / 1000.0),
        "n_turbines": ("t_cap", "size"),
        "xlong":      ("xlong", "mean"),
        "ylat":       ("ylat", "mean"),
        "ba_code":    ("ba_code", "first"),
        "ba_source":  ("ba_source", "first"),
    }
    if "p_year" in turbines.columns:
        agg_dict["p_year"] = ("p_year", "first")
    if has_county:
        agg_dict["t_county"] = ("t_county", "first")

    if not keyed.empty:
        plants_keyed = keyed.groupby("eia_id", dropna=False).agg(**agg_dict).reset_index()
    else:
        plants_keyed = pd.DataFrame()

    # Each NaN-eia_id turbine becomes its own row.
    if not orphan.empty:
        plants_orphan = orphan.copy()
        plants_orphan["p_cap_mw"] = plants_orphan["t_cap"] / 1000.0
        plants_orphan["n_turbines"] = 1
        keep_cols = ["eia_id", "p_name", "t_state", "p_cap_mw",
                     "n_turbines", "xlong", "ylat", "ba_code", "ba_source"]
        if "p_year" in plants_orphan.columns:
            keep_cols.insert(3, "p_year")
        if has_county:
            keep_cols.append("t_county")
        plants_orphan = plants_orphan[keep_cols]
    else:
        plants_orphan = pd.DataFrame()

    plants = pd.concat([plants_keyed, plants_orphan], ignore_index=True)
    plants = plants.dropna(subset=["xlong", "ylat"]).copy()

    # Sanity guard: drop any aggregated row claiming >800 MW from <5 turbines.
    # No real wind plant has >150 MW per turbine.
    impossible = (plants["p_cap_mw"] > 800) & (plants["n_turbines"] < 5)
    if impossible.any():
        for _, row in plants[impossible].iterrows():
            print(f"  WARNING dropping implausible plant: "
                  f"{row['p_name']} = {row['p_cap_mw']:.0f} MW from "
                  f"{int(row['n_turbines'])} turbine(s)")
        plants = plants[~impossible].copy()

    return plants


def build_map(plants: pd.DataFrame, show_non_ercot: bool) -> folium.Map:
    if not show_non_ercot:
        plants = plants[plants["ba_code"] == "ERCO"].copy()

    if plants.empty:
        center = [31.0, -100.0]
    else:
        w = plants["p_cap_mw"].fillna(0).clip(lower=1)
        center = [(plants["ylat"] * w).sum() / w.sum(),
                  (plants["xlong"] * w).sum() / w.sum()]

    m = folium.Map(location=center, zoom_start=6,
                   tiles="cartodbpositron", control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Streets").add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Satellite", attr="Esri",
    ).add_to(m)

    layers = {
        "blue":   folium.FeatureGroup(name="ERCOT (EIA-860)", show=True),
        "orange": folium.FeatureGroup(name="ERCOT (state fallback) ⚠", show=True),
        "red":    folium.FeatureGroup(name="ERCOT (manual override)", show=True),
        "gray":   folium.FeatureGroup(name="Non-ERCOT", show=show_non_ercot),
    }

    for _, row in plants.iterrows():
        c = color_for(row)
        folium.CircleMarker(
            location=[row["ylat"], row["xlong"]],
            radius=radius_for(row["p_cap_mw"]),
            color=c, fill=True, fillColor=c, fillOpacity=0.55, weight=1.5,
            tooltip=f"{row['p_name']} • {row['p_cap_mw']:,.0f} MW",
            popup=folium.Popup(popup_html(row), max_width=320),
        ).add_to(layers[c])

    for layer in layers.values():
        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    MeasureControl(primary_length_unit="miles").add_to(m)

    ercot = plants[plants["ba_code"] == "ERCO"]
    n_state_fb = (ercot["ba_source"] == "state_fallback").sum()
    n_override = (ercot["ba_source"] == "override").sum()
    legend = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:12px 14px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.15);
                font-family:system-ui,-apple-system,sans-serif;font-size:12px;
                max-width:240px;line-height:1.4;">
      <div style="font-weight:600;margin-bottom:4px;">ERCOT Wind Inventory</div>
      <div style="color:#444;margin-bottom:8px;font-size:11px;">
        {len(ercot)} plants · {ercot['p_cap_mw'].sum():,.0f} MW
      </div>
      <div style="margin:3px 0;"><span style="display:inline-block;width:11px;height:11px;
        background:#3388ff;border-radius:50%;margin-right:7px;vertical-align:middle;"></span>EIA-860 → ERCOT</div>
      <div style="margin:3px 0;"><span style="display:inline-block;width:11px;height:11px;
        background:#ff9933;border-radius:50%;margin-right:7px;vertical-align:middle;"></span>State fallback ({n_state_fb})</div>
      <div style="margin:3px 0;"><span style="display:inline-block;width:11px;height:11px;
        background:#dd3333;border-radius:50%;margin-right:7px;vertical-align:middle;"></span>Override ({n_override})</div>
      <div style="margin:3px 0;"><span style="display:inline-block;width:11px;height:11px;
        background:#999;border-radius:50%;margin-right:7px;vertical-align:middle;"></span>Non-ERCOT</div>
      <div style="margin-top:8px;color:#888;font-size:11px;">
        Marker size ∝ √capacity. Click for details.
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uswtdb", default=str(SCRIPT_DIR / "data" / "uswtdb.csv"),
                   help="Path to USWTDB CSV (default: data/uswtdb.csv)")
    p.add_argument("--states", nargs="+", default=["TX"],
                   help="States to include (default: TX). "
                        "TX OK NM LA covers ERCOT-adjacent footprint.")
    p.add_argument("--eia860", default=str(SCRIPT_DIR / "data" / "2___Plant_Y2024.xlsx"),
                   help="Path to EIA-860 Plant_Y file")
    p.add_argument("--overrides", default=str(SCRIPT_DIR / "plant_overrides.csv"),
                   help="Path to plant_overrides.csv")
    p.add_argument("--output", default="ercot_wind_map.html",
                   help="Output HTML file (default: ercot_wind_map.html)")
    p.add_argument("--hide-non-ercot", action="store_true",
                   help="Hide non-ERCOT plants by default (still toggleable in legend)")
    args = p.parse_args()

    print(f"Loading USWTDB from {args.uswtdb} ...")
    turbines = load_uswtdb(args.uswtdb)
    turbines = turbines[turbines["t_state"].isin(args.states)].copy()
    print(f"  {len(turbines):,} turbines in {args.states}")

    print(f"Attaching BA from EIA-860 ({args.eia860}) ...")
    eia860_map = load_eia860_ba_map(args.eia860)

    # attach_ba_iso signatures have varied across versions of aggregation.py.
    # Inspect available parameters and pass overrides via whichever name fits.
    import inspect
    sig = inspect.signature(attach_ba_iso)
    params = sig.parameters
    kwargs = {}
    for ov_kw in ("overrides_csv", "overrides_path", "overrides"):
        if ov_kw in params:
            kwargs[ov_kw] = args.overrides
            break
    if "eia860_map" in params:
        kwargs["eia860_map"] = eia860_map
        turbines = attach_ba_iso(turbines, **kwargs)
    elif "eia860_path" in params:
        # Older variant takes the path directly
        kwargs["eia860_path"] = args.eia860
        turbines = attach_ba_iso(turbines, **kwargs)
    else:
        # Positional fallback: (inventory, eia860_map)
        turbines = attach_ba_iso(turbines, eia860_map, **kwargs)

    # If no override kwarg was accepted, apply overrides ourselves so the
    # red-marker layer still works.
    if not any(k in kwargs for k in ("overrides_csv", "overrides_path", "overrides")):
        ov_path = Path(args.overrides)
        if ov_path.exists():
            ov = pd.read_csv(ov_path, comment="#")
            if {"eia_id", "ba_code"}.issubset(ov.columns):
                ov_map = dict(zip(
                    pd.to_numeric(ov["eia_id"], errors="coerce").astype("Int64"),
                    ov["ba_code"].astype(str).str.upper(),
                ))
                mask = turbines["eia_id"].isin(ov_map.keys())
                if mask.any():
                    turbines.loc[mask, "ba_code"] = (
                        turbines.loc[mask, "eia_id"].map(ov_map))
                    turbines.loc[mask, "ba_source"] = "override"
                    print(f"  applied {mask.sum():,} override turbine rows "
                          f"({len(ov_map)} plant(s))")

    plants = aggregate_to_plants(turbines)
    print(f"  {len(plants):,} plants aggregated")
    ercot = plants[plants["ba_code"] == "ERCO"]
    n_state_fb = (ercot["ba_source"] == "state_fallback").sum()
    n_override = (ercot["ba_source"] == "override").sum()
    print(f"  {len(ercot):,} routed to ERCOT "
          f"({n_state_fb} state-fallback, {n_override} override)")

    print("Building map ...")
    m = build_map(plants, show_non_ercot=not args.hide_non_ercot)

    out = Path(args.output).resolve()
    m.save(str(out))
    print(f"\n✓ Wrote {out}")
    print(f"  Open in your browser. Orange dots are state-fallback routings — "
          f"those are the ones to scrutinize.")


if __name__ == "__main__":
    main()
