#!/usr/bin/env python3
"""
Build a standalone HTML dashboard from a forecast output CSV.

Reads either the ISO-level or BA-level forecast CSV produced by run.py
and produces a self-contained HTML file with an interactive Plotly
chart and a dropdown menu to switch between regions (or view all of
them at once, or a national aggregate).

Usage
-----
    # Default: read forecast_iso_<cycle>.csv, save dashboard.html alongside
    python dashboard.py output/2026043018/forecast_iso_20260430T18Z.csv

    # Override output path
    python dashboard.py forecast_iso.csv -o my_dashboard.html

    # Use the BA-level file instead of ISO-level (more granular)
    python dashboard.py output/2026043018/forecast_ba_20260430T18Z.csv

    # Pick which region to show by default (defaults to "National")
    python dashboard.py forecast_iso.csv --default ERCOT

The output HTML is fully self-contained (Plotly.js is embedded inline)
so it can be emailed, dropped into a web folder, or opened locally
without any server.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    sys.stderr.write("plotly is required: pip install plotly\n")
    sys.exit(1)


# Region ordering: ISOs first (largest first), then notable non-ISO BAs
ISO_ORDER = ["ERCOT", "MISO", "SPP", "PJM", "CAISO", "NYISO", "ISO-NE"]


def detect_group_col(df: pd.DataFrame) -> str:
    """Find the per-region grouping column in a forecast CSV."""
    for c in ("iso", "ba_code", "t_state"):
        if c in df.columns:
            return c
    raise ValueError(
        f"No grouping column found in CSV. Expected one of "
        f"'iso', 'ba_code', 't_state'. Got: {list(df.columns)}"
    )


def parse_cycle_from_filename(path: Path) -> Optional[str]:
    """Pull a cycle timestamp out of a filename like forecast_iso_20260430T18Z.csv."""
    m = re.search(r"(\d{8}T\d{2}Z)", path.name)
    return m.group(1) if m else None


def order_regions(regions: list[str]) -> list[str]:
    """ISOs first in ISO_ORDER, then everything else alphabetically."""
    in_iso = [r for r in ISO_ORDER if r in regions]
    others = sorted([r for r in regions if r not in ISO_ORDER])
    return in_iso + others


def build_dashboard(csv_path: Path,
                    output_html: Path,
                    default_region: str = "National",
                    title: Optional[str] = None,
                    plotly_js: str = "inline",
                    theme: str = "light") -> None:
    """Read a forecast CSV and write an interactive HTML dashboard.

    plotly_js : {"inline", "cdn", "directory"}
        How Plotly.js is bundled. "inline" embeds the full ~3 MB library
        into the HTML (self-contained, works offline). "cdn" loads it
        from a CDN at view time (file is ~50 KB but needs internet).
        Use "cdn" for web hosting.
    theme : {"light", "dark"}
        Visual theme. "dark" matches a #0f0f0d iframe wrapper.
    """
    df = pd.read_csv(csv_path, parse_dates=["valid_time"])
    group_col = detect_group_col(df)

    # Pivot to wide form: rows = valid_time, cols = region, values = MW
    pivot = (df.pivot_table(index="valid_time",
                            columns=group_col,
                            values="MW",
                            aggfunc="sum")
               .sort_index())

    # Add a National aggregate that sums all regions
    pivot["National"] = pivot.sum(axis=1)

    # Order regions for nice dropdown layout
    region_cols = [c for c in pivot.columns if c != "National"]
    ordered = ["National"] + order_regions(region_cols)

    if default_region not in ordered:
        sys.stderr.write(
            f"WARNING: default region {default_region!r} not in data; "
            f"falling back to 'National'. Available: {ordered}\n"
        )
        default_region = "National"

    # Build a header title
    cycle_tag = parse_cycle_from_filename(csv_path)
    if title is None:
        if cycle_tag:
            title = f"US Wind Generation Forecast — HRRR cycle {cycle_tag}"
        else:
            title = "US Wind Generation Forecast"

    # Capacity context: annotate each trace's max
    fig = go.Figure()
    for region in ordered:
        y = pivot[region].to_numpy()
        fig.add_trace(go.Scatter(
            x=pivot.index, y=y,
            name=region,
            mode="lines",
            line=dict(width=2),
            visible=(region == default_region),
            hovertemplate=(
                "<b>" + region + "</b><br>" +
                "%{x|%Y-%m-%d %H:%M} UTC<br>" +
                "%{y:,.0f} MW<extra></extra>"
            ),
        ))

    # Dropdown: one button per region; "Show all ISOs" makes every ISO
    # visible at once for visual comparison.
    buttons = []
    for region in ordered:
        visibility = [r == region for r in ordered]
        buttons.append(dict(
            label=region, method="update",
            args=[{"visible": visibility},
                  {"title.text": f"{title}<br><sub>{region}</sub>"}],
        ))

    iso_set = set(ISO_ORDER)
    show_all_iso_vis = [r in iso_set for r in ordered]
    if any(show_all_iso_vis):
        buttons.append(dict(
            label="── All ISOs (overlay) ──", method="update",
            args=[{"visible": show_all_iso_vis},
                  {"title.text": f"{title}<br><sub>All ISOs</sub>"}],
        ))

    show_all_vis = [True] * len(ordered)
    buttons.append(dict(
        label="── Everything (overlay) ──", method="update",
        args=[{"visible": show_all_vis},
              {"title.text": f"{title}<br><sub>All regions</sub>"}],
    ))

    from datetime import datetime, timezone
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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
            text=f"{title}<br><sub>{default_region}</sub>",
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
                text=(f"Generated {generated_at} &middot; "
                      "USWTDB + HRRR &middot; "
                      "physics-only (no curtailment correction)"),
                showarrow=False,
                x=0.5, y=-0.22, xref="paper", yref="paper",
                xanchor="center", yanchor="top",
                font=dict(size=11, color=footer_color),
            ),
        ],
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_html),
        include_plotlyjs=plotly_js,
        full_html=True,
    )

    print(f"Wrote {output_html}")
    print(f"  Regions: {len(ordered)} ({', '.join(ordered[:5])}…)")
    print(f"  Time range: {pivot.index.min()} → {pivot.index.max()} UTC")
    print(f"  Peak national MW: {pivot['National'].max():,.0f}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate an interactive Plotly dashboard from a forecast CSV"
    )
    p.add_argument("csv_path", type=Path,
                   help="Path to forecast_iso_*.csv or forecast_ba_*.csv")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output HTML path (default: alongside the CSV)")
    p.add_argument("--default", default="National",
                   help='Default region to show (default: "National")')
    p.add_argument("--title", default=None,
                   help="Override the dashboard title")
    p.add_argument("--plotly-js", default="inline",
                   choices=["inline", "cdn", "directory"],
                   help="How to bundle Plotly.js. 'inline' = self-contained "
                        "(~3 MB), 'cdn' = small file but requires internet "
                        "at view time. Use 'cdn' for web hosting.")
    p.add_argument("--theme", default="light",
                   choices=["light", "dark"],
                   help="Color theme. Use 'dark' to match a dark iframe wrapper.")

    args = p.parse_args(argv)

    if args.output is None:
        args.output = args.csv_path.with_suffix(".html")

    build_dashboard(
        csv_path=args.csv_path,
        output_html=args.output,
        default_region=args.default,
        title=args.title,
        plotly_js=args.plotly_js,
        theme=args.theme,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
