"""
Build an interactive plant-level solar generation map with a time slider.

Reads:
  - forecast_plant_<cycle>.csv  (per-plant per-hour MW_AC)
  - capacity_plant_<cycle>.csv  (per-plant lat/lon, capacity, axis, type)

Writes:
  - assets/solar_map.html  (Plotly Mapbox map embedded as iframe in renewables.html)

Each plant is a circle marker:
  - position: (lat, lon) from USPVDB inventory
  - size: scales with sqrt(p_cap_ac) so big plants are emphasized
  - color: capacity factor (0%-100%) on warm yellow-orange colormap
  - hover: plant name, capacity, current MW, current CF%, axis type, state

A slider at the bottom advances through the 48-hour forecast horizon.
Default frame is the highest-mean-CF hour so the map opens on the
peak generation moment.

A click panel below the map shows the per-plant time series when the
user clicks a marker.

Usage:
    python solar_map_dashboard.py \
        assets/solar_forecast_data/forecast_plant_20260523T18Z.csv \
        --capacity assets/solar_forecast_data/capacity_plant_20260523T18Z.csv \
        --output assets/solar_map.html
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Solar plants below this capacity hidden by default. Most US utility-
# scale solar is ≥5 MW; smaller sites are commercial rooftop/canopy
# installations that add visual noise without information.
MIN_PLANT_MW_DEFAULT = 5.0


def _color_scale() -> list:
    """Solar capacity factor colormap: dark teal (low) → warm yellow → orange-red.

    Matches the solar/sunlight visual identity. Yellow at peak feels
    natural for solar; warm orange-red at saturation reinforces the
    "hot midday" signal.
    """
    return [
        [0.00, "#1c4870"],   # deep blue (no irradiance)
        [0.15, "#2c7d8a"],   # teal
        [0.35, "#5dbb63"],   # green
        [0.55, "#e8b34c"],   # warm yellow
        [0.75, "#f6c453"],   # bright yellow
        [1.00, "#c75d3a"],   # saturated orange-red (full nameplate)
    ]


def _marker_size(cap_MW: np.ndarray) -> np.ndarray:
    """Marker pixel diameter, scales with sqrt(capacity).

    A 500 MW plant is ~7x the diameter of a 10 MW plant rather than 50x.
    Clamped to [5, 26] so even small plants remain hoverable but giants
    don't dominate.
    """
    cap_MW = np.asarray(cap_MW, dtype=float)
    return np.clip(2.5 * np.sqrt(np.maximum(cap_MW, 1.0)), 5.0, 26.0)


def _hover_text(row: dict, mw: float, cf: float) -> str:
    """Build the hover tooltip HTML for one plant at one time step."""
    cap = row.get("p_cap_ac", float("nan"))
    name = row.get("p_name", "Unknown")
    state = row.get("p_state", "")
    axis = row.get("p_axis", "")
    sys_type = row.get("p_sys_type", "")
    parts = [
        f"<b>{name}</b>",
        f"{mw:,.1f} MW  ({cf:.0f}% of {cap:,.0f} MW)",
    ]
    if state:
        parts.append(f"{state}")
    meta = []
    if axis and pd.notna(axis):
        meta.append(f"{axis}")
    if sys_type and pd.notna(sys_type):
        meta.append(f"{sys_type}")
    if meta:
        parts.append("  ·  ".join(meta))
    return "<br>".join(parts)


def build_map(forecast_csv: Path, capacity_csv: Path, output_path: Path,
              min_mw: float = MIN_PLANT_MW_DEFAULT,
              theme: str = "dark") -> None:
    """Build the time-stepped plant-level solar map and save as HTML."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        sys.exit("plotly is required. Install with: pip install plotly")

    print(f"Reading forecast from {forecast_csv}")
    fc = pd.read_csv(forecast_csv, parse_dates=["valid_time"])
    print(f"  {len(fc):,} rows, "
          f"{fc['case_id'].nunique():,} plants, "
          f"{fc['valid_time'].nunique()} timesteps")

    print(f"Reading inventory from {capacity_csv}")
    inv = pd.read_csv(capacity_csv)
    print(f"  {len(inv):,} plants in inventory")

    # Filter to plants above the size threshold AND with valid coordinates
    inv = inv.dropna(subset=["xlong", "ylat", "p_cap_ac"])
    inv = inv[inv["p_cap_ac"] >= min_mw].copy()
    print(f"  {len(inv):,} plants above {min_mw} MW threshold with coords")

    # Use case_id as the join key (unique per plant in USPVDB)
    if "case_id" not in fc.columns or "case_id" not in inv.columns:
        sys.exit("Both forecast and capacity CSVs must have case_id column")

    timesteps = sorted(fc["valid_time"].unique())
    n_frames = len(timesteps)
    print(f"  {n_frames} animation frames")

    # Pivot: rows = plants (in inventory order), cols = timesteps, values = MW
    # Forecast rows for hours with zero generation were pruned (to keep
    # CSV small) — reindex+fillna(0) restores them.
    pivot = (fc.pivot_table(index="case_id", columns="valid_time",
                              values="MW_AC", aggfunc="sum")
                .reindex(timesteps, axis=1)
                .reindex(inv["case_id"].values)
                .fillna(0.0))

    # Capacity factor per (plant, timestep)
    cap_arr = inv["p_cap_ac"].to_numpy(dtype=float)
    mw_arr = pivot.to_numpy(dtype=float)
    mw_arr = np.maximum(mw_arr, 0.0)
    mw_arr = np.minimum(mw_arr, cap_arr[:, None])
    cf_arr = 100.0 * mw_arr / np.maximum(cap_arr[:, None], 1e-6)
    cf_arr = np.clip(cf_arr, 0.0, 100.0)

    # Initial frame: highest mean CF (peak generation moment)
    mean_cf_per_frame = cf_arr.mean(axis=0)
    initial_frame_idx = int(np.argmax(mean_cf_per_frame))

    sizes = _marker_size(cap_arr)

    # Theme styling — match wind map
    if theme == "dark":
        mapbox_style = "carto-darkmatter"
        bg_color = "#0f0f0d"
        font_color = "#e6e4dd"
        slider_bg = "rgba(40,40,38,0.85)"
        slider_active = "#f6c453"   # solar yellow
        slider_border = "rgba(255,255,255,0.15)"
    else:
        mapbox_style = "carto-positron"
        bg_color = "#fafaf8"
        font_color = "#1c1c1a"
        slider_bg = "rgba(255,255,255,0.85)"
        slider_active = "#c75d3a"
        slider_border = "rgba(0,0,0,0.15)"

    # Build hover text per (plant, frame). For 6,576 plants × 49 frames
    # = 322k strings, each ~150 bytes = ~48 MB just for hover text.
    # We optimize by reusing static info per plant; only the dynamic
    # parts (current MW, current CF) vary per frame.
    print("Building hover text...")
    inv_records = inv.to_dict("records")
    # Pre-build static prefix per plant (name, state, axis, type)
    static_prefixes = []
    for row in inv_records:
        name = row.get("p_name", "Unknown")
        state = row.get("p_state", "")
        axis = row.get("p_axis", "")
        sys_type = row.get("p_sys_type", "")
        cap = row.get("p_cap_ac", 0)
        parts = [f"<b>{name}</b>"]
        # Insert MW/CF placeholder here; will be substituted per frame
        parts.append("__MWCF__")
        if state:
            parts.append(f"{state}")
        meta = []
        if axis and pd.notna(axis):
            meta.append(f"{axis}")
        if sys_type and pd.notna(sys_type):
            meta.append(f"{sys_type}")
        if meta:
            parts.append("  ·  ".join(meta))
        static_prefixes.append(("<br>".join(parts), float(cap)))

    hover_text_per_frame = []
    for f_idx in range(n_frames):
        frame_text = []
        for p_idx, (prefix, cap) in enumerate(static_prefixes):
            mw = float(mw_arr[p_idx, f_idx])
            cf = float(cf_arr[p_idx, f_idx])
            mwcf = f"{mw:,.1f} MW  ({cf:.0f}% of {cap:,.0f} MW)"
            frame_text.append(prefix.replace("__MWCF__", mwcf))
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

    # Animation frames
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

    # Per-plant time-series data for the click handler
    timeseries_data = []
    timestep_strs = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%MZ")
                     for t in timesteps]
    for p_idx, row in enumerate(inv_records):
        timeseries_data.append({
            "name":     row.get("p_name", "Unknown"),
            "cap_MW":   float(row.get("p_cap_ac", 0)),
            "state":    str(row.get("p_state", "")),
            "axis":     str(row.get("p_axis", "")),
            "sys_type": str(row.get("p_sys_type", "")),
            "mw":       [float(x) for x in mw_arr[p_idx]],
            "cf":       [float(x) for x in cf_arr[p_idx]],
        })

    fig_json = pio.to_json(fig, validate=False)
    html = _build_html(
        fig_json, timeseries_data, timestep_strs,
        theme=theme, bg_color=bg_color, font_color=font_color,
        slider_active=slider_active, slider_bg=slider_bg,
        slider_border=slider_border,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\n✓ Wrote {output_path}")
    size_kb = output_path.stat().st_size / 1024
    print(f"  File size: {size_kb:,.0f} KB")
    print(f"  Plants:    {len(inv):,}")
    print(f"  Frames:    {n_frames}")
    print(f"  Initial:   frame {initial_frame_idx} "
          f"({timesteps[initial_frame_idx]}, "
          f"mean CF {mean_cf_per_frame[initial_frame_idx]:.0f}%)")


def _build_html(fig_json: str, timeseries_data: list, timestep_strs: list,
                *, theme: str, bg_color: str, font_color: str,
                slider_active: str, slider_bg: str,
                slider_border: str) -> str:
    """Wrap the map in HTML with a click-to-timeseries panel below it."""
    hint_color = "rgba(255,255,255,0.4)" if theme == "dark" else "rgba(0,0,0,0.35)"
    plot_line_color = "#f6c453" if theme == "dark" else "#c75d3a"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  html, body {{
    margin: 0; padding: 0;
    background: {bg_color};
    color: {font_color};
    font-family: Inter, system-ui, -apple-system, sans-serif;
    width: 100%; height: 100%;
  }}
  #wrap {{
    display: flex; flex-direction: column;
    width: 100%; height: 100vh;
  }}
  #map-div {{
    flex: 1 1 auto;
    min-height: 60%;
  }}
  #ts-panel {{
    flex: 0 0 auto;
    height: 250px;
    border-top: 1px solid {slider_border};
    padding: 8px 16px;
    box-sizing: border-box;
  }}
  #ts-header {{
    font-size: 13px;
    margin-bottom: 4px;
    color: {font_color};
  }}
  #ts-div {{
    width: 100%;
    height: calc(100% - 24px);
  }}
  .hint {{
    color: {hint_color};
    font-style: italic;
  }}
  @media (max-width: 640px) {{
    #ts-panel {{ height: 190px; }}
    #ts-header {{ font-size: 12px; }}
  }}
</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<div id="wrap">
  <div id="map-div"></div>
  <div id="ts-panel">
    <div id="ts-header"><span class="hint">Click any plant to see its 48-hour forecast</span></div>
    <div id="ts-div"></div>
  </div>
</div>
<script>
  const figData = {fig_json};
  const tsData = {json.dumps(timeseries_data)};
  const timestepStrs = {json.dumps(timestep_strs)};
  const bgColor = "{bg_color}";
  const fontColor = "{font_color}";
  const lineColor = "{plot_line_color}";

  Plotly.newPlot('map-div', figData.data, figData.layout, {{
    responsive: true,
    displayModeBar: false,
    scrollZoom: true,
  }}).then(() => {{
    if (figData.frames) {{
      Plotly.addFrames('map-div', figData.frames);
    }}
  }});

  document.getElementById('map-div').on('plotly_click', function(evt) {{
    if (!evt.points || !evt.points.length) return;
    const idx = evt.points[0].pointIndex;
    const plant = tsData[idx];
    if (!plant) return;
    const header = document.getElementById('ts-header');
    header.innerHTML = '<b>' + plant.name + '</b> · '
      + plant.cap_MW.toLocaleString() + ' MW · '
      + plant.state + ' · ' + plant.axis;
    Plotly.newPlot('ts-div', [{{
      x: timestepStrs,
      y: plant.mw,
      mode: 'lines',
      line: {{ width: 2, color: lineColor }},
      hovertemplate: '%{{x}}<br>%{{y:,.1f}} MW<extra></extra>',
    }}], {{
      paper_bgcolor: bgColor,
      plot_bgcolor: bgColor,
      font: {{ color: fontColor, family: 'Inter, system-ui, sans-serif', size: 11 }},
      margin: {{ l: 50, r: 20, t: 10, b: 40 }},
      yaxis: {{
        title: 'MW',
        gridcolor: 'rgba(150,150,150,0.15)',
        zerolinecolor: 'rgba(150,150,150,0.3)',
        range: [0, plant.cap_MW * 1.05],
      }},
      xaxis: {{
        gridcolor: 'rgba(150,150,150,0.15)',
        zerolinecolor: 'rgba(150,150,150,0.3)',
      }},
    }}, {{
      responsive: true,
      displayModeBar: false,
    }});
  }});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Build interactive solar plant-level map")
    parser.add_argument("forecast_csv", type=Path,
                        help="Per-plant forecast CSV")
    parser.add_argument("--capacity", type=Path, required=True,
                        help="Per-plant capacity CSV")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output HTML path")
    parser.add_argument("--min-mw", type=float, default=MIN_PLANT_MW_DEFAULT,
                        help="Filter plants below this capacity")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark")
    args = parser.parse_args()

    build_map(args.forecast_csv, args.capacity, args.output,
              min_mw=args.min_mw, theme=args.theme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
