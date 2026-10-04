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

from mobile_html import write_responsive_html, write_select_dashboard_html

try:
    import plotly.graph_objects as go
except ImportError:
    sys.stderr.write("plotly is required: pip install plotly\n")
    sys.exit(1)


# Region ordering: ISOs first (largest first), then notable non-ISO BAs
ISO_ORDER = ["ERCOT", "MISO", "SPP", "PJM", "CAISO", "SunZia (CAISO)",
             "NYISO", "ISO-NE"]


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
                    theme: str = "light",
                    capacity_csv: Optional[Path] = None) -> None:
    """Read a forecast CSV and write an interactive HTML dashboard.

    plotly_js : {"inline", "cdn", "directory"}
        How Plotly.js is bundled. "inline" embeds the full ~3 MB library
        into the HTML (self-contained, works offline). "cdn" loads it
        from a CDN at view time (file is ~50 KB but needs internet).
        Use "cdn" for web hosting.
    theme : {"light", "dark"}
        Visual theme. "dark" matches a #0f0f0d iframe wrapper.
    capacity_csv : Path, optional
        CSV with columns (region, capacity_MW). When provided, draws a
        dashed horizontal reference line for the selected region's
        installed capacity. If not provided, dashboard auto-detects a
        sibling file named like the forecast CSV but with prefix
        `capacity_` instead of `forecast_`.
    """
    df = pd.read_csv(csv_path, parse_dates=["valid_time"])
    group_col = detect_group_col(df)

    # Pivot to wide form: rows = valid_time, cols = region, values = MW
    pivot = (df.pivot_table(index="valid_time",
                            columns=group_col,
                            values="MW",
                            aggfunc="sum")
               .sort_index())

    # Add a National aggregate that sums all regions — EXCLUDING project
    # breakout rows (e.g. "SunZia (CAISO)"), whose generation is already
    # inside their host ISO's total; summing both double-counted ~3 GW.
    try:
        from aggregation import PROJECT_BREAKOUTS
        breakout_labels = set(PROJECT_BREAKOUTS.values())
    except Exception:
        breakout_labels = {c for c in pivot.columns if "(" in str(c)}
    real_cols = [c for c in pivot.columns if c not in breakout_labels]
    pivot["National"] = pivot[real_cols].sum(axis=1)

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

    # Load capacity reference data, if available.
    # Auto-detect: forecast_iso_<tag>.csv → capacity_iso_<tag>.csv
    capacity_lookup: dict = {}
    if capacity_csv is None:
        guess = csv_path.parent / csv_path.name.replace("forecast_", "capacity_", 1)
        if guess.exists():
            capacity_csv = guess
    if capacity_csv is not None and Path(capacity_csv).exists():
        cap_df = pd.read_csv(capacity_csv)
        if {"region", "capacity_MW"}.issubset(cap_df.columns):
            capacity_lookup = dict(zip(cap_df["region"].astype(str),
                                       cap_df["capacity_MW"].astype(float)))
            # Also compute National = sum, since the capacity CSV may not
            # include it explicitly (it does at the iso level via run.py,
            # but be safe)
            if "National" not in capacity_lookup:
                cap_real = cap_df[~cap_df["region"].astype(str).isin(breakout_labels)]
                capacity_lookup["National"] = float(cap_real["capacity_MW"].sum())
            print(f"  Loaded capacity reference for {len(capacity_lookup)} regions")
        else:
            print(f"  Warning: capacity CSV missing required columns: {capacity_csv}")

    # Each region produces TWO traces: forecast (solid line) and capacity
    # (dashed horizontal reference). They share visibility; toggling a
    # region in the dropdown shows/hides both.
    fig = go.Figure()
    n_per_region = 2  # forecast + capacity reference

    for region in ordered:
        y = pivot[region].to_numpy()
        is_default = (region == default_region)

        # Trace 1: forecast
        fig.add_trace(go.Scatter(
            x=pivot.index, y=y,
            name=region,
            mode="lines",
            line=dict(width=2),
            visible=is_default,
            hovertemplate=(
                "<b>" + region + "</b><br>" +
                "%{x|%Y-%m-%d %H:%M} UTC<br>" +
                "%{y:,.0f} MW<extra></extra>"
            ),
        ))

        # Trace 2: installed capacity (horizontal line)
        cap_mw = capacity_lookup.get(region)
        if cap_mw is not None and cap_mw > 0:
            fig.add_trace(go.Scatter(
                x=[pivot.index.min(), pivot.index.max()],
                y=[cap_mw, cap_mw],
                name=f"{region} installed capacity",
                mode="lines",
                line=dict(width=1.2, dash="dash",
                          color="rgba(150,150,150,0.7)"),
                visible=is_default,
                hovertemplate=(
                    f"<b>{region} installed capacity</b><br>" +
                    f"{cap_mw:,.0f} MW<extra></extra>"
                ),
                showlegend=False,
            ))
        else:
            # Add a dummy invisible trace so the trace-index math stays
            # consistent (every region has exactly n_per_region traces).
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
            vis.append(on)               # forecast trace
            vis.append(on)               # capacity trace
        return vis

    # Dropdown: one button per region; "Show all ISOs" makes every ISO
    # visible at once for visual comparison.
    buttons = []
    for region in ordered:
        # Build subtitle with capacity context
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
                  {"title.text": f"{title}<br><sub>{subtitle}</sub>"}],
        ))

    iso_set = set(ISO_ORDER) & set(ordered)
    if iso_set:
        buttons.append(dict(
            label="── All ISOs (overlay) ──", method="update",
            args=[{"visible": visibility_for(iso_set)},
                  {"title.text": f"{title}<br><sub>All ISOs</sub>"}],
        ))

    buttons.append(dict(
        label="── Everything (overlay) ──", method="update",
        args=[{"visible": visibility_for(set(ordered))},
              {"title.text": f"{title}<br><sub>All regions</sub>"}],
    ))

    # Default subtitle (matches the default selection)
    cap_def = capacity_lookup.get(default_region)
    if cap_def and cap_def > 0:
        peak_def = float(pivot[default_region].max())
        default_subtitle = (f"{default_region} · {cap_def:,.0f} MW installed "
                            f"· peak forecast {peak_def:,.0f} MW "
                            f"({100 * peak_def / cap_def:.0f}% CF)")
    else:
        default_subtitle = default_region

    from datetime import datetime, timezone
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if theme == "dark":
        plotly_template = "plotly_dark"
        bg_color = "#0f0f0d"
        gridcolor = "rgba(255,255,255,0.08)"
    else:
        plotly_template = "plotly_white"
        bg_color = "white"
        gridcolor = "rgba(0,0,0,0.06)"

    # Layout WITHOUT updatemenus — region switching is handled by a native
    # HTML <select> in the wrapper template (mobile-friendly). Plotly's
    # own dropdown is unreliable on touch devices.
    fig.update_layout(
        xaxis=dict(title="Valid Time (UTC)", showgrid=True,
                   gridcolor=gridcolor),
        yaxis=dict(title="Forecast Generation (MW)", rangemode="tozero",
                   showgrid=True, gridcolor=gridcolor),
        hovermode="x unified",
        template=plotly_template,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=60, r=20, t=20, b=70),
        legend=dict(orientation="h", yanchor="top", y=-0.12,
                    xanchor="center", x=0.5),
    )

    # Build the option list and the JS-side visibility + subtitle maps.
    # Options mirror the old dropdown: each region, then the two overlays.
    select_options = list(ordered)

    visibility_map: dict = {}
    subtitle_map: dict = {}
    for region in ordered:
        visibility_map[region] = visibility_for({region})
        cap = capacity_lookup.get(region)
        if cap and cap > 0:
            peak = float(pivot[region].max())
            subtitle_map[region] = (f"{region} · {cap:,.0f} MW installed · "
                                     f"peak forecast {peak:,.0f} MW "
                                     f"({100 * peak / cap:.0f}% CF)")
        else:
            subtitle_map[region] = region

    iso_set = set(ISO_ORDER) & set(ordered)
    if iso_set:
        key = "── All ISOs (overlay) ──"
        select_options.append(key)
        visibility_map[key] = visibility_for(iso_set)
        subtitle_map[key] = "All ISOs"

    key_all = "── Everything (overlay) ──"
    select_options.append(key_all)
    visibility_map[key_all] = visibility_for(set(ordered))
    subtitle_map[key_all] = "All regions"

    footer_text = (f"Generated {generated_at} · USWTDB + HRRR · "
                   "physics-only (no curtailment correction)")

    output_html.parent.mkdir(parents=True, exist_ok=True)
    write_select_dashboard_html(
        fig=fig,
        output_html=output_html,
        options=select_options,
        visibility_map=visibility_map,
        subtitle_map=subtitle_map,
        title=title,
        default_key=default_region,
        footer_text=footer_text,
        theme=theme,
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
    p.add_argument("--capacity-csv", type=Path, default=None,
                   help="Path to capacity_<level>_<cycle>.csv with columns "
                        "(region, capacity_MW). If omitted, dashboard "
                        "auto-detects a sibling file alongside the forecast CSV.")

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
        capacity_csv=args.capacity_csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
