"""
Build an interactive plant-level wind generation map with a time slider.

Reads:
  - forecast_plant_<cycle>.csv  (one row per (eia_id, p_name, valid_time))
  - capacity_plant_<cycle>.csv  (one row per plant with lat/lon, capacity, BA)

Writes:
  - assets/wind_map.html  (Plotly Mapbox map embedded as iframe in charts.html)

Each plant is a circle marker:
  - position: (lat, lon) from the inventory
  - size: scales with sqrt(capacity_MW), so big farms aren't 500x bigger than
    small ones visually
  - color: capacity factor (0%-100%) on a perceptually-uniform colormap
  - hover: plant name, capacity, current MW, current CF%, BA, year, turbines

A slider at the bottom advances through the forecast horizon (typically 48
hourly frames). Default frame is the highest-CF hour so the map opens on
something interesting.

Usage:
    python map_dashboard.py output/forecast_plant_20260504T18Z.csv \
        --capacity output/capacity_plant_20260504T18Z.csv \
        --output assets/wind_map.html
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Plants below this capacity are hidden by default to keep the map readable.
# Most of these are pre-2010 single-turbine sites that don't add information.
MIN_PLANT_MW_DEFAULT = 5.0


def _color_scale() -> list:
    """Capacity factor colormap: blue (calm) → teal → yellow → red (rated).

    Hand-picked to be perceptually monotonic and read well at marker sizes
    of 5-30 pixels. Matches the warmth of the rest of the site palette.
    """
    return [
        [0.00, "#1c4870"],   # deep blue (calm)
        [0.20, "#2c7da0"],   # blue-teal
        [0.40, "#4ea99d"],   # teal-green
        [0.60, "#a3c061"],   # yellow-green
        [0.80, "#e8b34c"],   # warm yellow
        [1.00, "#c75d3a"],   # warm red (saturated)
    ]


def _marker_size(cap_MW: np.ndarray) -> np.ndarray:
    """Marker pixel diameter, scales with sqrt(capacity).

    A 500 MW farm is ~7x the diameter of a 10 MW farm rather than 50x.
    Clamped to [6, 28] so even the smallest farms remain hoverable.
    """
    cap_MW = np.asarray(cap_MW, dtype=float)
    return np.clip(2.5 * np.sqrt(np.maximum(cap_MW, 1.0)), 6.0, 28.0)


def _hover_text(row: pd.Series, mw: float, cf: float) -> str:
    """Build the hover tooltip HTML for one plant at one time step."""
    cap = row.get("capacity_MW", float("nan"))
    n_turbines = int(row.get("n_turbines", 0))
    ba = row.get("ba_code", "?")
    year = row.get("p_year", "")
    if pd.notna(year):
        try:
            year = int(year)
        except (TypeError, ValueError):
            pass
    name = row.get("p_name", "Unknown")
    state = row.get("t_state", "")
    county = row.get("county", "")
    location_bits = [b for b in (county, state) if b and pd.notna(b)]
    location = ", ".join(location_bits)

    parts = [
        f"<b>{name}</b>",
        f"{mw:,.0f} MW  ({cf:.0f}% of {cap:,.0f} MW)",
    ]
    if location:
        parts.append(f"{location}")
    meta = []
    if ba and pd.notna(ba):
        meta.append(f"BA: {ba}")
    if year:
        meta.append(f"Built: {year}")
    if n_turbines:
        meta.append(f"{n_turbines} turbines")
    if meta:
        parts.append("  ·  ".join(meta))
    return "<br>".join(parts)


def build_map(forecast_csv: Path, capacity_csv: Path, output_path: Path,
              min_mw: float = MIN_PLANT_MW_DEFAULT,
              theme: str = "dark") -> None:
    """Build the time-stepped plant-level map and save as HTML."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit("plotly is required. Install with: pip install plotly")

    print(f"Reading forecast from {forecast_csv}")
    fc = pd.read_csv(forecast_csv, parse_dates=["valid_time"])
    print(f"  {len(fc):,} rows, "
          f"{fc['eia_id'].nunique():,} plants, "
          f"{fc['valid_time'].nunique()} timesteps")

    print(f"Reading inventory from {capacity_csv}")
    inv = pd.read_csv(capacity_csv)
    print(f"  {len(inv):,} plants in inventory")

    # Filter to plants above the size threshold AND with valid coordinates.
    inv = inv.dropna(subset=["xlong", "ylat", "capacity_MW"])
    inv = inv[inv["capacity_MW"] >= min_mw].copy()
    print(f"  {len(inv):,} plants above {min_mw} MW threshold with coords")

    # Merge forecast onto inventory. Use eia_id since p_name has duplicates
    # (e.g. multi-phase projects sometimes share a name).
    if "eia_id" not in fc.columns or "eia_id" not in inv.columns:
        sys.exit("Both forecast and capacity CSVs must have eia_id column")

    timesteps = sorted(fc["valid_time"].unique())
    n_frames = len(timesteps)
    print(f"  {n_frames} animation frames")

    # Build a wide table: rows = plants (in inv order), cols = timesteps.
    # This is faster than per-frame merging.
    pivot = (fc.pivot_table(index="eia_id", columns="valid_time",
                            values="MW", aggfunc="sum")
               .reindex(timesteps, axis=1))  # ensure column ordering

    # Align with inventory; plants in inv but not fc get all-zero rows.
    pivot = pivot.reindex(inv["eia_id"].tolist())
    pivot = pivot.fillna(0.0)

    # Pre-compute capacity factor for each (plant, timestep)
    cap_arr = inv["capacity_MW"].to_numpy(dtype=float)
    mw_arr = pivot.to_numpy(dtype=float)
    cf_arr = 100.0 * mw_arr / np.maximum(cap_arr[:, None], 1e-6)
    cf_arr = np.clip(cf_arr, 0.0, 105.0)  # 105 cap allows for rounding

    # Find the most-interesting frame to show first (highest mean CF)
    mean_cf_per_frame = cf_arr.mean(axis=0)
    initial_frame_idx = int(np.argmax(mean_cf_per_frame))

    # Marker sizes (constant across frames)
    sizes = _marker_size(cap_arr)

    # Theme styling
    if theme == "dark":
        mapbox_style = "carto-darkmatter"
        bg_color = "#0f0f0d"
        font_color = "#e6e4dd"
        slider_bg = "rgba(40,40,38,0.85)"
        slider_active = "#c75d3a"
        slider_border = "rgba(255,255,255,0.15)"
    else:
        mapbox_style = "carto-positron"
        bg_color = "#fafaf8"
        font_color = "#1c1c1a"
        slider_bg = "rgba(255,255,255,0.85)"
        slider_active = "#2c4a72"
        slider_border = "rgba(0,0,0,0.15)"

    # Build hover text per (plant, frame). This is N_plants * N_frames
    # strings — for ~700 plants × 48 frames = 33,600 strings. Acceptable
    # for HTML size; would need lazy generation if scale grew.
    print("Building hover text...")
    hover_text_per_frame = []
    inv_records = inv.to_dict("records")
    for f_idx in range(n_frames):
        frame_text = []
        for p_idx, row in enumerate(inv_records):
            mw = float(mw_arr[p_idx, f_idx])
            cf = float(cf_arr[p_idx, f_idx])
            frame_text.append(_hover_text(pd.Series(row), mw, cf))
        hover_text_per_frame.append(frame_text)

    # Map center: weighted by capacity
    weights = cap_arr / cap_arr.sum()
    center_lat = float(np.sum(inv["ylat"].to_numpy() * weights))
    center_lon = float(np.sum(inv["xlong"].to_numpy() * weights))

    print("Assembling Plotly figure...")
    fig = go.Figure()

    # Initial frame: scattermapbox trace
    fig.add_trace(go.Scattermapbox(
        lat=inv["ylat"].tolist(),
        lon=inv["xlong"].tolist(),
        mode="markers",
        marker=dict(
            size=sizes.tolist(),
            color=cf_arr[:, initial_frame_idx].tolist(),
            colorscale=_color_scale(),
            cmin=0, cmax=100,
            colorbar=dict(
                title=dict(text="Capacity Factor (%)", side="right"),
                thickness=12, len=0.6, x=1.0, xanchor="left",
                tickfont=dict(color=font_color, size=10),
            ),
            opacity=0.85,
            sizemode="diameter",
        ),
        text=hover_text_per_frame[initial_frame_idx],
        hovertemplate="%{text}<extra></extra>",
        name="",
    ))

    # Build animation frames
    frames = []
    for f_idx in range(n_frames):
        frames.append(go.Frame(
            data=[go.Scattermapbox(
                lat=inv["ylat"].tolist(),
                lon=inv["xlong"].tolist(),
                mode="markers",
                marker=dict(
                    size=sizes.tolist(),
                    color=cf_arr[:, f_idx].tolist(),
                    colorscale=_color_scale(),
                    cmin=0, cmax=100,
                    opacity=0.85,
                    sizemode="diameter",
                ),
                text=hover_text_per_frame[f_idx],
                hovertemplate="%{text}<extra></extra>",
            )],
            name=str(f_idx),
        ))
    fig.frames = frames

    # Slider steps
    slider_steps = []
    for f_idx, t in enumerate(timesteps):
        ts = pd.Timestamp(t)
        label = ts.strftime("%a %m/%d %HZ")
        slider_steps.append(dict(
            method="animate",
            label=label,
            args=[[str(f_idx)], dict(
                mode="immediate",
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=0),
            )],
        ))

    fig.update_layout(
        mapbox=dict(
            style=mapbox_style,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=3.7,
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=font_color, family="Inter, system-ui, sans-serif"),
        margin=dict(l=0, r=0, t=0, b=70),
        hoverlabel=dict(
            bgcolor=bg_color,
            bordercolor=slider_active,
            font=dict(family="Inter, system-ui, sans-serif", size=12,
                      color=font_color),
        ),
        sliders=[dict(
            active=initial_frame_idx,
            currentvalue=dict(
                prefix="Forecast valid: ",
                font=dict(size=12, color=font_color),
            ),
            pad=dict(t=20, b=10, l=10, r=10),
            len=0.92,
            x=0.04,
            y=0.0,
            yanchor="bottom",
            bgcolor=slider_bg,
            bordercolor=slider_border,
            borderwidth=1,
            ticklen=4,
            tickcolor=font_color,
            font=dict(size=10, color=font_color),
            activebgcolor=slider_active,
            steps=slider_steps,
        )],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.0,
            y=0.0,
            xanchor="left", yanchor="bottom",
            pad=dict(t=20, l=10, b=20),
            bgcolor=slider_bg,
            bordercolor=slider_border,
            font=dict(size=11, color=font_color),
            buttons=[
                dict(label="▶  Play", method="animate", args=[None, dict(
                    mode="immediate",
                    frame=dict(duration=300, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=100),
                )]),
                dict(label="⏸  Pause", method="animate", args=[[None], dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=False),
                    transition=dict(duration=0),
                )]),
            ],
        )],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(displayModeBar=False, responsive=True),
    )
    print(f"\n✓ Wrote {output_path}")
    size_kb = output_path.stat().st_size / 1024
    print(f"  File size: {size_kb:,.0f} KB")
    print(f"  Plants:    {len(inv):,}")
    print(f"  Frames:    {n_frames}")
    print(f"  Initial:   frame {initial_frame_idx} "
          f"({timesteps[initial_frame_idx]}, "
          f"mean CF {mean_cf_per_frame[initial_frame_idx]:.0f}%)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("forecast_csv",
                   help="Path to forecast_plant_<cycle>.csv")
    p.add_argument("--capacity",
                   help="Path to capacity_plant_<cycle>.csv "
                        "(defaults to alongside forecast_csv)")
    p.add_argument("--output", default="assets/wind_map.html",
                   help="Output HTML path (default: assets/wind_map.html)")
    p.add_argument("--min-mw", type=float, default=MIN_PLANT_MW_DEFAULT,
                   help=f"Hide plants smaller than this (default: "
                        f"{MIN_PLANT_MW_DEFAULT} MW)")
    p.add_argument("--theme", choices=("light", "dark"), default="dark",
                   help="Map color scheme (default: dark, matches site)")
    args = p.parse_args()

    forecast_path = Path(args.forecast_csv)
    if args.capacity:
        capacity_path = Path(args.capacity)
    else:
        # Derive: forecast_plant_X.csv → capacity_plant_X.csv in same dir
        cap_name = forecast_path.name.replace("forecast_plant_",
                                              "capacity_plant_")
        capacity_path = forecast_path.parent / cap_name

    if not forecast_path.exists():
        sys.exit(f"Forecast CSV not found: {forecast_path}")
    if not capacity_path.exists():
        sys.exit(f"Capacity CSV not found: {capacity_path}")

    build_map(forecast_path, capacity_path,
              Path(args.output),
              min_mw=args.min_mw, theme=args.theme)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
