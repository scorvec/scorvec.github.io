"""Solar forecast dashboard.

Builds a multi-panel Plotly HTML chart showing 48-hour solar forecast
for each of the dashboard regions (ERCOT, CAISO, MISO, PJM, SPP,
Southeast). Each panel has the forecast generation line and a dashed
nameplate capacity reference.

Output: assets/solar_forecast.html

Usage:
    python solar_dashboard.py                # latest cycle
    python solar_dashboard.py 20260523T18Z   # specific cycle
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from solar_aggregation import (
    DATA_DIR, DASHBOARD_REGIONS, get_latest_cycle,
)


# Output goes alongside wind_forecast.html in /assets/
ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
OUTPUT_HTML = ASSETS_DIR / "solar_forecast.html"


# Dark theme to match wind dashboard
DARK_TEMPLATE = "plotly_dark"
DARK_BG = "#0f0f0d"
DARK_PAPER = "#0f0f0d"
LINE_COLOR = "#f6c453"     # warm amber, evokes sunlight (matches site palette)
CAPACITY_COLOR = "rgba(150,150,150,0.6)"


def build_dashboard(cycle_str: str, theme: str = "dark") -> None:
    """Build the solar forecast dashboard HTML."""
    forecast_path = DATA_DIR / f"forecast_region_{cycle_str}.csv"
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"Region forecast CSV not found: {forecast_path}\n"
            f"Run solar_aggregation.py first."
        )

    df = pd.read_csv(forecast_path, parse_dates=["valid_time"])
    print(f"  Loaded {len(df):,} rows for {df['region'].nunique()} regions")

    # Capacity lookup (use first value per region; constant within cycle)
    capacity = (df.groupby("region")["capacity_MW"]
                  .first().to_dict())

    # Pivot to wide format: rows=valid_time, cols=region, values=MW_AC
    pivot = df.pivot_table(index="valid_time", columns="region",
                            values="MW_AC", aggfunc="sum").sort_index()
    # Fill missing hours with 0 (nighttime gaps from row-pruning in forecast)
    full_index = pd.date_range(pivot.index.min(), pivot.index.max(), freq="h")
    pivot = pivot.reindex(full_index).fillna(0.0)

    # Order regions: by capacity descending, only including DASHBOARD_REGIONS
    available = [r for r in DASHBOARD_REGIONS if r in pivot.columns]
    if not available:
        raise RuntimeError(
            f"No dashboard regions found in data. Available regions: "
            f"{sorted(pivot.columns)}; expected: {DASHBOARD_REGIONS}"
        )
    available = sorted(available, key=lambda r: -capacity.get(r, 0))

    # 6 panels in a 3×2 grid (or 2×3 — wider screens look better with 3 cols)
    n = len(available)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    # Each subplot title includes region name + capacity
    titles = []
    for r in available:
        cap = capacity.get(r, 0)
        titles.append(f"<b>{r}</b> · {cap:,.0f} MW capacity")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        shared_xaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for i, region in enumerate(available):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        cap = capacity.get(region, 0)

        # Forecast line
        y = pivot[region].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=pivot.index, y=y,
                mode="lines",
                name=region,
                line=dict(width=2, color=LINE_COLOR),
                hovertemplate=(
                    f"<b>{region}</b><br>"
                    "%{x|%Y-%m-%d %H:%M} UTC<br>"
                    "%{y:,.0f} MW<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=col,
        )

        # Capacity reference (dashed horizontal)
        if cap > 0:
            fig.add_trace(
                go.Scatter(
                    x=[pivot.index.min(), pivot.index.max()],
                    y=[cap, cap],
                    mode="lines",
                    line=dict(width=1.2, dash="dash", color=CAPACITY_COLOR),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )

        # Y-axis: 0 to capacity (with 5% headroom)
        fig.update_yaxes(
            range=[0, cap * 1.05] if cap > 0 else None,
            row=row, col=col,
            title_text="MW" if col == 1 else None,
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.15)",
        )
        fig.update_xaxes(
            row=row, col=col,
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.15)",
        )

    # Theme & layout
    if theme == "dark":
        plotly_template = DARK_TEMPLATE
        paper_bg = DARK_PAPER
        plot_bg = DARK_BG
        font_color = "#e8e4d8"
    else:
        plotly_template = "plotly_white"
        paper_bg = "#fafaf8"
        plot_bg = "#fafaf8"
        font_color = "#1a1a17"

    # Parse cycle for title
    cycle_dt = pd.to_datetime(cycle_str, format="%Y%m%dT%HZ")
    title_str = (f"<b>HRRR Solar Generation Forecast</b> · "
                 f"Cycle {cycle_dt.strftime('%Y-%m-%d %HZ')} · 48 hours")

    fig.update_layout(
        title=dict(text=title_str, font=dict(size=18), x=0.02, xanchor="left"),
        template=plotly_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color, family="Inter, sans-serif"),
        height=620,
        margin=dict(l=70, r=30, t=80, b=50),
        hovermode="x unified",
    )

    # Subplot title color
    for annotation in fig.layout.annotations:
        annotation.font.color = font_color
        annotation.font.size = 12

    # Write HTML
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUTPUT_HTML,
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(
            displayModeBar=False,
            responsive=True,
        ),
    )
    print(f"  Wrote {OUTPUT_HTML}")
    sz = OUTPUT_HTML.stat().st_size / 1024
    print(f"  File size: {sz:.0f} KB")


def main():
    print("=" * 70)
    print("Solar forecast dashboard")
    print("=" * 70)

    if len(sys.argv) >= 2:
        cycle_str = sys.argv[1]
    else:
        cycle_str = get_latest_cycle()
        if cycle_str is None:
            print(f"ERROR: no forecast files found")
            return 1

    print(f"\nCycle: {cycle_str}")

    # Allow --theme light from CLI
    theme = "dark"
    if "--theme" in sys.argv:
        idx = sys.argv.index("--theme")
        theme = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "dark"
    print(f"Theme: {theme}")

    build_dashboard(cycle_str, theme=theme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
