"""Solar forecast dashboard.

Builds an interactive Plotly chart with a dropdown selector to switch
between the national total, individual regions (ERCOT, CAISO, MISO,
PJM, SPP, Southeast), or overlay views. Mirrors the wind dashboard
UX pattern.

Output: assets/solar_forecast.html

Usage:
    python solar_dashboard.py                # latest cycle
    python solar_dashboard.py 20260523T18Z   # specific cycle
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from solar_aggregation import (
    DATA_DIR, DASHBOARD_REGIONS, get_latest_cycle,
)


# Output goes alongside wind_forecast.html in /assets/
ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
OUTPUT_HTML = ASSETS_DIR / "solar_forecast.html"

# Regions to surface in the dropdown, in the order shown (largest first
# generally, but ordering matches DASHBOARD_REGIONS from solar_aggregation).
# "National" is computed as the sum of all of these.
NATIONAL_LABEL = "National"

# Colors for each region's line — warm/amber-based palette to evoke
# sunlight, with enough variation to distinguish overlaid lines
REGION_COLORS = {
    NATIONAL_LABEL: "#f6c453",   # bright sunlight yellow
    "CAISO":        "#e8893a",   # warm orange (California sun)
    "ERCOT":        "#c75d3a",   # saturated red-orange (Texas sun)
    "Southeast":    "#7fb069",   # warm green (Florida/SE foliage)
    "MISO":         "#6a9aae",   # cool blue-gray (Great Lakes/Plains)
    "PJM":          "#a87fb0",   # muted purple (Mid-Atlantic)
    "SPP":          "#d4b85b",   # warm gold (Plains harvest)
    "ISO-NE":       "#4a8c8c",   # deep teal (New England coast)
    "NYISO":        "#d97658",   # muted terracotta (autumn leaves)
}


def build_dashboard(cycle_str: str, theme: str = "dark") -> None:
    """Build the solar forecast dashboard HTML with dropdown selector."""
    forecast_path = DATA_DIR / f"forecast_region_{cycle_str}.csv"
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"Region forecast CSV not found: {forecast_path}\n"
            f"Run solar_aggregation.py first."
        )

    df = pd.read_csv(forecast_path, parse_dates=["valid_time"])
    print(f"  Loaded {len(df):,} rows for {df['region'].nunique()} regions")

    # Pivot to wide format: index=valid_time, cols=region, values=MW
    pivot = df.pivot_table(index="valid_time", columns="region",
                            values="MW_AC", aggfunc="sum").sort_index()
    full_index = pd.date_range(pivot.index.min(), pivot.index.max(), freq="h")
    pivot = pivot.reindex(full_index).fillna(0.0)

    # Capacity per region
    capacity_lookup = (df.groupby("region")["capacity_MW"]
                         .first().to_dict())

    # Build the National column = sum of all dashboard regions present
    available_regions = [r for r in DASHBOARD_REGIONS if r in pivot.columns]
    if not available_regions:
        raise RuntimeError(
            f"No dashboard regions found. Got: {sorted(pivot.columns)}; "
            f"expected: {DASHBOARD_REGIONS}"
        )
    pivot[NATIONAL_LABEL] = pivot[available_regions].sum(axis=1)
    capacity_lookup[NATIONAL_LABEL] = sum(
        capacity_lookup.get(r, 0) for r in available_regions
    )

    # Ordered list: National first, then individual regions sorted by capacity
    ordered = [NATIONAL_LABEL] + sorted(
        available_regions,
        key=lambda r: -capacity_lookup.get(r, 0),
    )
    print(f"  Series: {ordered}")

    # Build figure with one Scatter per region (forecast) + per region (capacity)
    fig = go.Figure()

    # Forecast traces — start with only National visible
    default_region = NATIONAL_LABEL
    for region in ordered:
        color = REGION_COLORS.get(region, "#888888")
        y = pivot[region].to_numpy()
        fig.add_trace(go.Scatter(
            x=pivot.index, y=y,
            mode="lines",
            name=region,
            line=dict(width=2.5, color=color),
            visible=(region == default_region),
            hovertemplate=(
                f"<b>{region}</b><br>"
                "%{x|%Y-%m-%d %H:%M} UTC<br>"
                "%{y:,.0f} MW<extra></extra>"
            ),
        ))
        # Capacity reference (dashed horizontal at nameplate)
        cap = capacity_lookup.get(region, 0)
        if cap > 0:
            fig.add_trace(go.Scatter(
                x=[pivot.index.min(), pivot.index.max()],
                y=[cap, cap],
                mode="lines",
                line=dict(width=1.2, dash="dash", color=color),
                opacity=0.5,
                name=f"{region} capacity",
                visible=(region == default_region),
                hovertemplate=f"{region} nameplate: {cap:,.0f} MW<extra></extra>",
                showlegend=False,
            ))
        else:
            # Placeholder trace so visibility list stays consistent across regions
            fig.add_trace(go.Scatter(
                x=[pivot.index.min()], y=[None],
                mode="lines", visible=False, showlegend=False,
                hoverinfo="skip", name=f"{region}_no_capacity",
            ))

    def visibility_for(regions_visible: set) -> list:
        """Build per-trace visibility list given a set of visible regions."""
        vis = []
        for region in ordered:
            on = region in regions_visible
            vis.append(on)   # forecast trace
            vis.append(on)   # capacity trace
        return vis

    # Build dropdown buttons
    buttons = []
    for region in ordered:
        cap = capacity_lookup.get(region)
        if cap and cap > 0:
            peak = float(pivot[region].max())
            subtitle = (f"{region} · {cap:,.0f} MW installed · "
                        f"peak forecast {peak:,.0f} MW "
                        f"({100 * peak / cap:.0f}% CF)")
        else:
            subtitle = region
        buttons.append(dict(
            label=region, method="update",
            args=[{"visible": visibility_for({region})},
                  {"title.text": f"__TITLE__<br><sub>{subtitle}</sub>"}],
        ))

    # "All regions overlay" button — everything except National
    buttons.append(dict(
        label="── All regions (overlay) ──", method="update",
        args=[{"visible": visibility_for(set(available_regions))},
              {"title.text": "__TITLE__<br><sub>All regions overlaid</sub>"}],
    ))

    # "Everything overlay" button — National plus all regions
    buttons.append(dict(
        label="── Everything (overlay) ──", method="update",
        args=[{"visible": visibility_for(set(ordered))},
              {"title.text": "__TITLE__<br><sub>National + all regions</sub>"}],
    ))

    # Title and default subtitle
    cycle_dt = pd.to_datetime(cycle_str, format="%Y%m%dT%HZ")
    title = (f"<b>HRRR Solar Generation Forecast</b> · "
             f"Cycle {cycle_dt.strftime('%Y-%m-%d %HZ')} · 48 hours")
    # Substitute placeholder in buttons (so all buttons keep the title)
    for b in buttons:
        if "title.text" in b["args"][1]:
            b["args"][1]["title.text"] = b["args"][1]["title.text"].replace(
                "__TITLE__", title)

    cap_def = capacity_lookup.get(default_region, 0)
    peak_def = float(pivot[default_region].max())
    default_subtitle = (f"{default_region} · {cap_def:,.0f} MW installed · "
                        f"peak forecast {peak_def:,.0f} MW "
                        f"({100 * peak_def / cap_def:.0f}% CF)"
                        if cap_def > 0 else default_region)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Theme
    if theme == "dark":
        plotly_template = "plotly_dark"
        bg_color = "#0f0f0d"
        gridcolor = "rgba(255,255,255,0.08)"
        footer_color = "rgba(255,255,255,0.5)"
        dropdown_bg = "rgba(30,30,28,0.95)"
        dropdown_border = "rgba(255,255,255,0.2)"
    else:
        plotly_template = "plotly_white"
        bg_color = "white"
        gridcolor = "rgba(0,0,0,0.06)"
        footer_color = "rgba(0,0,0,0.5)"
        dropdown_bg = "white"
        dropdown_border = "rgba(0,0,0,0.2)"

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>{default_subtitle}</sub>",
            x=0.02, xanchor="left",
        ),
        xaxis=dict(title="Valid Time (UTC)", showgrid=True,
                   gridcolor=gridcolor),
        yaxis=dict(title="Forecast Generation (MW)", rangemode="tozero",
                   showgrid=True, gridcolor=gridcolor),
        hovermode="x unified",
        template=plotly_template,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=70, r=30, t=120, b=110),
        height=620,
        legend=dict(orientation="h", yanchor="top", y=-0.12,
                    xanchor="center", x=0.5),
        updatemenus=[dict(
            active=ordered.index(default_region),
            buttons=buttons,
            x=1.0, y=1.18, xanchor="right", yanchor="top",
            bgcolor=dropdown_bg,
            bordercolor=dropdown_border,
        )],
        annotations=[
            dict(
                text="<b>Region:</b>", showarrow=False,
                x=1.0, y=1.24, xref="paper", yref="paper",
                xanchor="right", yanchor="bottom",
                font=dict(size=12),
            ),
            dict(
                text=(f"Generated {generated_at} · USPVDB + HRRR · "
                      "pvlib physics (sun position, tracker geometry, "
                      "temperature derate, inverter clipping)"),
                showarrow=False,
                x=0.5, y=-0.22, xref="paper", yref="paper",
                xanchor="center", yanchor="top",
                font=dict(size=11, color=footer_color),
            ),
        ],
    )

    # Write HTML
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUTPUT_HTML,
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(displayModeBar=False, responsive=True),
    )
    print(f"  Wrote {OUTPUT_HTML}")
    sz = OUTPUT_HTML.stat().st_size / 1024
    print(f"  File size: {sz:.0f} KB")
    print(f"  Default view: {default_region}")
    print(f"  Peak {default_region}: {pivot[default_region].max():,.0f} MW")


def main():
    print("=" * 70)
    print("Solar forecast dashboard")
    print("=" * 70)

    if len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
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
