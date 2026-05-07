#!/usr/bin/env python3
"""
Build a verification dashboard from verification.csv.

Reads the merged forecast/actual/curtailment dataset produced by the
verification notebook and produces a Plotly HTML with three traces per
region: physics-only forecast (blue), forecast minus reported
curtailment (green dashed), and actual generation (red).

Usage
-----
    python verification_dashboard.py \
        ../assets/wind_forecast_data/verification.csv \
        --plotly-js cdn --theme dark \
        -o ../assets/wind_verification.html

Pairs with dashboard.py (the operational forecast); the chart card on
charts.html embeds both via <iframe> tabs.
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


ISO_ORDER = ["ERCOT", "MISO", "SPP", "PJM", "CAISO", "NYISO", "ISO-NE"]


def order_regions(regions: list[str]) -> list[str]:
    in_iso = [r for r in ISO_ORDER if r in regions]
    others = sorted([r for r in regions if r not in ISO_ORDER])
    return in_iso + others


def build_verification_dashboard(csv_path: Path,
                                 output_html: Path,
                                 default_region: str = "ERCOT",
                                 plotly_js: str = "cdn",
                                 theme: str = "dark",
                                 title: str = None) -> None:
    df = pd.read_csv(csv_path, parse_dates=["valid_time"])
    needed = {"region", "valid_time", "forecast_MW", "actual_MW"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"verification.csv missing columns: {missing}")

    if title is None:
        # Auto-derive a title with the visible date range
        dmin = df["valid_time"].min().strftime("%b %d")
        dmax = df["valid_time"].max().strftime("%b %d")
        title = f"Historical Wind Forecast Performance ({dmin} – {dmax})"

    has_curtailment = "forecast_minus_curtail_MW" in df.columns

    regions = order_regions(sorted(df["region"].unique()))
    if default_region not in regions:
        default_region = regions[0]

    # Theme colors
    if theme == "dark":
        plotly_template = "plotly_dark"
        bg = "#0f0f0d"
        grid = "rgba(255,255,255,0.08)"
        footer_color = "rgba(255,255,255,0.5)"
        dropdown_bg = "rgba(30,30,28,0.95)"
        dropdown_border = "rgba(255,255,255,0.2)"
    else:
        plotly_template = "plotly_white"
        bg = "white"
        grid = "rgba(0,0,0,0.06)"
        footer_color = "rgba(0,0,0,0.5)"
        dropdown_bg = "white"
        dropdown_border = "rgba(0,0,0,0.2)"

    # Trace colors — match the operational dashboard's palette
    C_FORECAST = "#185FA5"   # blue
    C_ADJUSTED = "#1D9E75"   # green
    C_ACTUAL   = "#993556"   # red

    fig = go.Figure()

    # Build three traces per region (forecast, adjusted, actual).
    # We track them so visibility flags can flip together per dropdown.
    traces_per_region = 3 if has_curtailment else 2
    region_trace_map: dict[str, list[int]] = {r: [] for r in regions}

    for region in regions:
        sub = df[df["region"] == region].sort_values("valid_time")
        is_default = (region == default_region)

        # Trace 1: physics forecast
        idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=sub["valid_time"], y=sub["forecast_MW"],
            name=f"{region} — model (no curtailment)",
            mode="lines",
            line=dict(color=C_FORECAST, width=2),
            visible=is_default,
            hovertemplate=(f"<b>{region} model</b><br>"
                           "%{x|%Y-%m-%d %H:%M} UTC<br>"
                           "%{y:,.0f} MW<extra></extra>"),
        ))
        region_trace_map[region].append(idx)

        # Trace 2: curtailment-adjusted forecast (if available)
        if has_curtailment:
            idx = len(fig.data)
            fig.add_trace(go.Scatter(
                x=sub["valid_time"], y=sub["forecast_minus_curtail_MW"],
                name=f"{region} — model after curtailment",
                mode="lines",
                line=dict(color=C_ADJUSTED, width=2, dash="dash"),
                visible=is_default,
                hovertemplate=(f"<b>{region} model − curtailment</b><br>"
                               "%{x|%Y-%m-%d %H:%M} UTC<br>"
                               "%{y:,.0f} MW<extra></extra>"),
            ))
            region_trace_map[region].append(idx)

        # Trace 3: actuals
        idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=sub["valid_time"], y=sub["actual_MW"],
            name=f"{region} — actual generation",
            mode="lines",
            line=dict(color=C_ACTUAL, width=2),
            visible=is_default,
            hovertemplate=(f"<b>{region} actual</b><br>"
                           "%{x|%Y-%m-%d %H:%M} UTC<br>"
                           "%{y:,.0f} MW<extra></extra>"),
        ))
        region_trace_map[region].append(idx)

    n_traces = len(fig.data)

    # Per-region buttons: show only that region's traces
    buttons = []
    for region in regions:
        visibility = [False] * n_traces
        for idx in region_trace_map[region]:
            visibility[idx] = True
        # Region-level metrics for the subtitle
        sub = df[df["region"] == region]
        e = (sub["forecast_MW"] - sub["actual_MW"]).dropna()
        bias = e.mean() if len(e) else 0
        if has_curtailment:
            ea = (sub["forecast_minus_curtail_MW"] - sub["actual_MW"]).dropna()
            bias_after = ea.mean() if len(ea) else None
            sub_text = (f"{region} · physics bias "
                        f"{bias:+,.0f} MW")
            if bias_after is not None:
                sub_text += f" · after curtailment {bias_after:+,.0f} MW"
        else:
            sub_text = f"{region} · physics bias {bias:+,.0f} MW"

        buttons.append(dict(
            label=region, method="update",
            args=[{"visible": visibility},
                  {"title.text": f"{title}<br>"
                                 f"<sub>{sub_text}</sub>"}],
        ))

    # Compute default subtitle
    sub = df[df["region"] == default_region]
    e = (sub["forecast_MW"] - sub["actual_MW"]).dropna()
    default_bias = e.mean() if len(e) else 0
    default_sub = f"{default_region} · physics bias {default_bias:+,.0f} MW"
    if has_curtailment:
        ea = (sub["forecast_minus_curtail_MW"] - sub["actual_MW"]).dropna()
        if len(ea):
            default_sub += f" · after curtailment {ea.mean():+,.0f} MW"

    from datetime import datetime, timezone
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{default_sub}</sub>",
                   x=0.02, xanchor="left"),
        xaxis=dict(title="Valid Time (UTC)", showgrid=True, gridcolor=grid),
        yaxis=dict(title="Generation (MW)", rangemode="tozero",
                   showgrid=True, gridcolor=grid),
        hovermode="x unified",
        template=plotly_template,
        paper_bgcolor=bg, plot_bgcolor=bg,
        margin=dict(l=70, r=30, t=120, b=110),
        height=620,
        showlegend=False,  # too many traces; the title carries the info
        updatemenus=[dict(
            active=regions.index(default_region),
            buttons=buttons,
            x=1.0, y=1.18, xanchor="right", yanchor="top",
            bgcolor=dropdown_bg, bordercolor=dropdown_border,
        )],
        annotations=[
            dict(text="<b>Region:</b>", showarrow=False,
                 x=1.0, y=1.24, xref="paper", yref="paper",
                 xanchor="right", yanchor="bottom", font=dict(size=12)),
            dict(text=("<span style='color:" + C_FORECAST + "'>━ Model (no curtailment)</span>"
                      + ("  <span style='color:" + C_ADJUSTED + "'>┄ Model after curtailment</span>"
                         if has_curtailment else "")
                      + "  <span style='color:" + C_ACTUAL + "'>━ Actual generation</span>"),
                 showarrow=False, x=0.5, y=-0.15, xref="paper", yref="paper",
                 xanchor="center", yanchor="top", font=dict(size=12)),
            dict(text=f"Day-ahead forecast (12–24h lead) · "
                      f"Updated {generated_at}",
                 showarrow=False, x=0.5, y=-0.22, xref="paper", yref="paper",
                 xanchor="center", yanchor="top",
                 font=dict(size=11, color=footer_color)),
        ],
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs=plotly_js, full_html=True)
    print(f"Wrote {output_html}")
    print(f"  Regions: {regions}")
    print(f"  Rows: {len(df):,}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--default", default="ERCOT")
    p.add_argument("--plotly-js", default="cdn", choices=["inline", "cdn", "directory"])
    p.add_argument("--theme", default="dark", choices=["light", "dark"])
    args = p.parse_args(argv)

    if args.output is None:
        args.output = args.csv_path.with_suffix(".html")

    build_verification_dashboard(
        csv_path=args.csv_path, output_html=args.output,
        default_region=args.default,
        plotly_js=args.plotly_js, theme=args.theme,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
