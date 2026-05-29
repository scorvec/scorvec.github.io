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

    # Build a wide table: rows = plants, cols = timesteps. Key on the
    # full (eia_id, p_name) tuple — some real plants share an eia_id
    # across phases (e.g. Vineyard Wind, multi-build wind farms in
    # USWTDB), and keying on eia_id alone would collapse them, then
    # reindex would assign all of one row's MW to whichever phase
    # sorted first and zero to the other. Keying on the tuple keeps
    # them separate.
    # Build NaN-safe join keys. Python's NaN != NaN means that
    # (NaN, "Big Sampson Wind") in fc would never match
    # (NaN, "Big Sampson Wind") in inv even though they're "equal" by
    # any reasonable definition. Replace NaN eia_id with a sentinel so
    # tuple comparisons work — this matters for new plants USWTDB has
    # but EIA-860 hasn't yet attributed (Big Sampson, Monte Cristo,
    # Lane City, Revolution Wind, etc.) and for repowers like Brazos
    # Wind Repower and Mountain View Power Partners Repower.
    NAN_SENTINEL = -999999
    fc_eia = fc["eia_id"].fillna(NAN_SENTINEL).astype("int64")
    inv_eia = inv["eia_id"].fillna(NAN_SENTINEL).astype("int64")
    fc_key = list(zip(fc_eia, fc["p_name"]))
    inv_key = list(zip(inv_eia, inv["p_name"]))
    fc = fc.assign(_join_key=fc_key)

    def _pivot(col: str) -> pd.DataFrame:
        return (fc.pivot_table(index="_join_key", columns="valid_time",
                               values=col, aggfunc="sum"
                                       if col in ("MW", "MW_gross") else "mean")
                  .reindex(timesteps, axis=1)
                  .reindex(inv_key)
                  .fillna(0.0))

    pivot = _pivot("MW_gross") if "MW_gross" in fc.columns else _pivot("MW")
    pivot_net = _pivot("MW")
    pivot_ws = _pivot("ws_hh") if "ws_hh" in fc.columns else None
    pivot_rho = _pivot("rho_hh") if "rho_hh" in fc.columns else None

    # Pre-compute capacity factor for each (plant, timestep). Clamp MW at
    # plant capacity to prevent display values exceeding nameplate — the
    # density correction in forecast.py preserves the rated-power flat-top
    # via min(P(ws_hh), P(ws_eq)), but small numerical excursions can
    # still push values fractionally above rated under cold/dense
    # conditions. Hard clamp protects the map's visual integrity.
    cap_arr = inv["capacity_MW"].to_numpy(dtype=float)
    mw_arr = pivot.to_numpy(dtype=float)
    mw_arr = np.minimum(mw_arr, cap_arr[:, None])  # cap at nameplate
    mw_arr = np.maximum(mw_arr, 0.0)               # no negative
    cf_arr = 100.0 * mw_arr / np.maximum(cap_arr[:, None], 1e-6)
    cf_arr = np.clip(cf_arr, 0.0, 100.0)

    # Net MW for the time-series detail (post-loss). Also clamp.
    mw_net_arr = pivot_net.to_numpy(dtype=float)
    mw_net_arr = np.clip(mw_net_arr, 0.0, cap_arr[:, None])

    # Optional weather diagnostics for the click panel
    ws_arr  = pivot_ws.to_numpy(dtype=float)  if pivot_ws  is not None else None
    rho_arr = pivot_rho.to_numpy(dtype=float) if pivot_rho is not None else None

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

    # Pack per-plant time series for the click handler. The browser-side
    # JS will read these to render the time-series chart on click.
    # Format: list aligned with marker indices, each element is a dict with
    # name, capacity, and parallel arrays of timestamps + MW values.
    timeseries_data = []
    timestep_strs = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%MZ")
                     for t in timesteps]
    for p_idx, row in enumerate(inv_records):
        entry = {
            "name":        row.get("p_name", "Unknown"),
            "capacity_MW": float(row.get("capacity_MW", 0)),
            "ba":          str(row.get("ba_code", "?")),
            "state":       str(row.get("t_state", "")),
            "county":      str(row.get("county", "")),
            "year":        (int(row.get("p_year"))
                            if pd.notna(row.get("p_year")) else None),
            "n_turbines":  int(row.get("n_turbines", 0)),
            "mw":      [float(x) for x in mw_arr[p_idx]],       # gross
            "mw_net":  [float(x) for x in mw_net_arr[p_idx]],   # post-loss
            "cf":      [float(x) for x in cf_arr[p_idx]],
        }
        if ws_arr is not None:
            entry["ws_hh"] = [float(x) for x in ws_arr[p_idx]]
        if rho_arr is not None:
            entry["rho_hh"] = [float(x) for x in rho_arr[p_idx]]
        timeseries_data.append(entry)

    # Get the figure as a JSON-serializable dict (compact form)
    import json
    import plotly.io as pio
    fig_json = pio.to_json(fig, validate=False)

    # Custom HTML wrapper: two Plotly divs (map + time-series), a click
    # handler that swaps in per-plant data when a marker is clicked.
    html = _build_html(fig_json, timeseries_data, timestep_strs,
                       theme=theme, bg_color=bg_color, font_color=font_color,
                       slider_active=slider_active,
                       slider_bg=slider_bg,
                       slider_border=slider_border)

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
    """Wrap the map in HTML with a click-to-timeseries panel below it.

    The map is rendered into #map-div, and the time-series chart into
    #ts-div. A Plotly `plotly_click` handler updates the time-series div
    when the user clicks a marker. All per-plant time-series data is
    embedded as a JSON blob in the page so no server round-trips are
    needed.
    """
    import json

    # Hint text shown in the time series panel before any plant is clicked
    hint_color = "rgba(255,255,255,0.4)" if theme == "dark" else "rgba(0,0,0,0.35)"

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
    min-height: 0;       /* lets flex shrink below content size */
  }}
  #ts-panel {{
    flex: 0 0 280px;
    border-top: 1px solid {slider_border};
    background: {bg_color};
    position: relative;
  }}
  #ts-div {{
    width: 100%; height: 100%;
  }}
  #ts-hint {{
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    color: {hint_color};
    font-size: 13px;
    pointer-events: none;
    text-align: center;
    padding: 1rem;
  }}
  #ts-hint.hidden {{ display: none; }}
  @media (max-width: 640px) {{
    #ts-panel {{ flex: 0 0 200px; }}
    #ts-hint {{ font-size: 12px; }}
  }}
</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<div id="wrap">
  <div id="map-div"></div>
  <div id="ts-panel">
    <div id="ts-div"></div>
    <div id="ts-hint">Click a plant on the map to see its 48-hour forecast.</div>
  </div>
</div>

<script>
  const figData = {fig_json};
  const tsData = {json.dumps(timeseries_data)};
  const timestepLabels = {json.dumps(timestep_strs)};

  // Common config for both maps and charts
  const mapConfig = {{
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage'],
    scrollZoom: true,
    responsive: true,
  }};
  const tsConfig = {{
    displayModeBar: false,
    responsive: true,
  }};

  Plotly.newPlot('map-div', figData.data, figData.layout, mapConfig)
    .then(function(gd) {{
      // Click handler: when a marker is clicked, render that plant's time series
      gd.on('plotly_click', function(eventData) {{
        if (!eventData || !eventData.points || !eventData.points.length) return;
        const p = eventData.points[0];
        const idx = p.pointIndex;
        const plant = tsData[idx];
        if (!plant) return;
        renderTimeseries(plant);
      }});
      // Animation frames need to be added explicitly
      if (figData.frames && figData.frames.length) {{
        Plotly.addFrames('map-div', figData.frames);
      }}
    }});

  function renderTimeseries(plant) {{
    document.getElementById('ts-hint').classList.add('hidden');

    const traces = [];

    // Primary: gross MW (solid, filled) — what the turbines are physically producing
    traces.push({{
      x: timestepLabels,
      y: plant.mw,
      type: 'scatter',
      mode: 'lines',
      line: {{ color: '{slider_active}', width: 2.5, shape: 'spline', smoothing: 0.7 }},
      fill: 'tozeroy',
      fillcolor: '{slider_active}33',
      name: 'Gross MW',
      hovertemplate: '<b>%{{y:,.0f}} MW</b> gross<extra></extra>',
      yaxis: 'y',
    }});

    // Net MW (post-loss, ~16% lower) — what the model expects on the grid.
    // Only plot if the data is present and meaningfully different from gross.
    if (plant.mw_net && plant.mw_net.length === plant.mw.length) {{
      traces.push({{
        x: timestepLabels,
        y: plant.mw_net,
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '{slider_active}', width: 1.5, dash: 'dot', shape: 'spline', smoothing: 0.7 }},
        name: 'Net MW (post-loss)',
        hovertemplate: '<b>%{{y:,.0f}} MW</b> net<extra></extra>',
        yaxis: 'y',
      }});
    }}

    // Secondary axis: hub-height wind speed (m/s)
    if (plant.ws_hh && plant.ws_hh.length) {{
      traces.push({{
        x: timestepLabels,
        y: plant.ws_hh,
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#7ab8d6', width: 1.5, shape: 'spline', smoothing: 0.4 }},
        name: 'Hub-height wind (m/s)',
        hovertemplate: '<b>%{{y:.1f}} m/s</b> hub-height<extra></extra>',
        yaxis: 'y2',
      }});
    }}

    // Air density on hover only — same axis as wind speed but very different
    // scale, so plot with extreme transparency or tuck into custom hover.
    // Cleanest: a hidden trace with customdata so it shows on the unified
    // hover but doesn't clutter the chart.
    if (plant.rho_hh && plant.rho_hh.length) {{
      traces.push({{
        x: timestepLabels,
        y: plant.rho_hh.map(function(r) {{ return r * 1; }}),
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#c79b7a', width: 1, dash: 'dash' }},
        name: 'Air density (kg/m³)',
        hovertemplate: '<b>%{{y:.3f}} kg/m³</b> air density<extra></extra>',
        yaxis: 'y3',
        opacity: 0.55,
      }});
    }}

    const meta_bits = [];
    if (plant.county) meta_bits.push(plant.county + ', ' + plant.state);
    else if (plant.state) meta_bits.push(plant.state);
    if (plant.ba && plant.ba !== '?' && plant.ba !== 'nan') meta_bits.push('BA: ' + plant.ba);
    if (plant.year) meta_bits.push('Built: ' + plant.year);
    if (plant.n_turbines) meta_bits.push(plant.n_turbines + ' turbines');
    const subtitle = meta_bits.join(' · ');

    const peakMW = Math.max.apply(null, plant.mw);
    const peakCF = Math.max.apply(null, plant.cf);

    // Wind axis range: cut-in to a reasonable upper bound; expand if data exceeds
    let wsMax = 25;
    if (plant.ws_hh && plant.ws_hh.length) {{
      wsMax = Math.max(25, Math.ceil(Math.max.apply(null, plant.ws_hh) * 1.1));
    }}

    const layout = {{
      title: {{
        text: '<b>' + plant.name + '</b>'
              + '<br><span style="font-size:11px;color:{font_color}99">'
              + subtitle + '  ·  ' + plant.capacity_MW.toLocaleString(undefined, {{maximumFractionDigits: 0}}) + ' MW capacity'
              + '  ·  Peak gross: ' + peakMW.toFixed(0) + ' MW (' + peakCF.toFixed(0) + '%)'
              + '</span>',
        font: {{ size: 13, color: '{font_color}' }},
        x: 0.02, y: 0.96, xanchor: 'left',
      }},
      paper_bgcolor: '{bg_color}',
      plot_bgcolor: '{bg_color}',
      font: {{ color: '{font_color}', family: 'Inter, system-ui, sans-serif' }},
      margin: {{ l: 55, r: 60, t: 55, b: 30 }},
      xaxis: {{
        showgrid: false,
        tickfont: {{ size: 10 }},
        nticks: 8,
        domain: [0, 1],
      }},
      yaxis: {{
        title: {{ text: 'MW', font: {{ size: 11, color: '{slider_active}' }} }},
        showgrid: true,
        gridcolor: '{slider_border}',
        zerolinecolor: '{slider_border}',
        rangemode: 'tozero',
        range: [0, plant.capacity_MW * 1.05],
        tickfont: {{ size: 10, color: '{slider_active}' }},
        side: 'left',
      }},
      yaxis2: {{
        title: {{ text: 'Wind (m/s)', font: {{ size: 11, color: '#7ab8d6' }} }},
        overlaying: 'y',
        side: 'right',
        showgrid: false,
        rangemode: 'tozero',
        range: [0, wsMax],
        tickfont: {{ size: 10, color: '#7ab8d6' }},
      }},
      yaxis3: {{
        // Air density axis — exists so the trace can hover, but hidden visually
        overlaying: 'y',
        side: 'right',
        position: 1,
        showgrid: false,
        showticklabels: false,
        showline: false,
        zeroline: false,
        range: [0.7, 1.4],
      }},
      shapes: [{{
        type: 'line',
        xref: 'paper', x0: 0, x1: 1,
        yref: 'y', y0: plant.capacity_MW, y1: plant.capacity_MW,
        line: {{ color: '{font_color}', width: 1, dash: 'dot' }},
        opacity: 0.4,
      }}],
      annotations: [{{
        xref: 'paper', x: 0.99,
        yref: 'y', y: plant.capacity_MW,
        text: 'Capacity (' + plant.capacity_MW.toFixed(0) + ' MW)',
        showarrow: false, xanchor: 'right', yanchor: 'bottom',
        font: {{ size: 9, color: '{font_color}99' }},
      }}],
      legend: {{
        orientation: 'h', x: 0, y: 1.0, yanchor: 'bottom',
        font: {{ size: 10, color: '{font_color}' }},
        bgcolor: 'rgba(0,0,0,0)',
      }},
      hovermode: 'x unified',
      hoverlabel: {{
        bgcolor: '{bg_color}',
        bordercolor: '{slider_active}',
        font: {{ size: 11, color: '{font_color}' }},
      }},
    }};

    Plotly.react('ts-div', traces, layout, tsConfig);
  }}
</script>
</body>
</html>
"""


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
