#!/usr/bin/env python3
"""Hadley cell INTENSITY and EDGE LATITUDE time series vs the ERA5 distribution.

Four panels — intensity and poleward-edge latitude for each cell — with the
observed trailing-7-day-mean series (AIFS-ENS 0-h analysis, from the same v-bar
history the Ψ animator uses) over the ERA5 1991-2020 p10-p90 / p25-p75 bands
and median for the matching day of year.

The bands come from hadley_clim.nc, whose spread is taken ACROSS YEARS of each
year's ~weekly mean — the same kind of object as the plotted series. Scoring a
7-day mean against a spread of instantaneous fields would flatter almost any
week into "normal"; scoring it against a 30-year MEAN circulation (the harmonic
Ψ climatology) does the opposite for edge-derived quantities.

    python src/hadley_timeseries.py --out assets/sst/hadley_timeseries.webp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from mmsf import CLIM, HIST, LEVELS, hadley_cells, streamfunction  # noqa: E402
from build_hadley_clim import OUT as CLIMFILE, sample_metrics      # noqa: E402

OBS_C = "#0b3d6b"        # observed series
MED_C = "#8a5a00"        # climatological median
B1 = "#c9d8e6"           # p10-p90
B2 = "#9dbad3"           # p25-p75

PANELS = (("nh_psi",       "Northern cell intensity",  "|Ψ|max  (10¹⁰ kg s⁻¹)", True),
          ("nh_edge_pole", "Northern cell poleward edge", "latitude (°N)",      False),
          ("sh_psi",       "Southern cell intensity",  "|Ψ|max  (10¹⁰ kg s⁻¹)", True),
          ("sh_edge_pole", "Southern cell poleward edge", "latitude (°S)",      False))


def observed(hist: xr.DataArray) -> pd.DataFrame:
    """Trailing 7-day-mean metrics per cycle — identical windowing to mmsf frames."""
    lat = hist.latitude.values.astype(float)
    p_hpa = np.asarray(LEVELS, float)
    t = pd.to_datetime(hist.time.values)
    rows, idx = [], []
    for i in range(len(t)):
        w = (t > t[i] - pd.Timedelta(days=7)) & (t <= t[i])
        if t[i] - t[w][0] < pd.Timedelta(days=6):        # window not yet ~a week
            continue
        psi = streamfunction(hist.values[w].mean(0), p_hpa * 100.0, lat)
        rows.append(sample_metrics(psi, p_hpa, lat))
        idx.append(t[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def clim_at(cl: xr.Dataset, key: str, doys) -> dict:
    """Climatological stats aligned to the observed dates."""
    d = cl[key].sel(doy=xr.DataArray(np.asarray(doys), dims="t"))
    return {s: d.sel(stat=s).values for s in ("p10", "p25", "p50", "p75", "p90", "mean", "sd")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/sst/hadley_timeseries.webp")
    ap.add_argument("--json-out", default="assets/sst/data/hadley_timeseries.json")
    a = ap.parse_args()

    if not CLIMFILE.exists():
        print(f"{CLIMFILE} missing — run build_hadley_clim.py first.", file=sys.stderr)
        return 1
    cl = xr.open_dataset(CLIMFILE)
    obs = observed(xr.open_dataarray(HIST))
    if obs.empty:
        print("no 7-day-mean frames in the v-bar history yet", file=sys.stderr)
        return 1
    # score each trailing-7-day mean at its window MIDPOINT, exactly as the Ψ
    # animator evaluates its climatology — at the end date the seasonal cycle
    # lags by half a window and the two products disagree by a few hundredths
    doys = (obs.index - pd.Timedelta(days=3.5)).dayofyear.values

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.0), sharex=True)
    summary = {}
    for ax, (key, title, ylab, is_psi) in zip(axes.ravel(), PANELS):
        c = clim_at(cl, key, doys)
        o = obs[key].values.astype(float)
        # the SH cell's Ψ is negative and its edge is in the south; plot both
        # cells on a common "bigger = stronger / further poleward" axis
        flip = -1.0 if (key.startswith("sh")) else 1.0
        oo = o * flip
        cc = {k: v * flip for k, v in c.items()}
        lo10, hi90 = np.minimum(cc["p10"], cc["p90"]), np.maximum(cc["p10"], cc["p90"])
        lo25, hi75 = np.minimum(cc["p25"], cc["p75"]), np.maximum(cc["p25"], cc["p75"])
        ax.fill_between(obs.index, lo10, hi90, color=B1, lw=0, label="ERA5 p10–p90")
        ax.fill_between(obs.index, lo25, hi75, color=B2, lw=0, label="p25–p75")
        ax.plot(obs.index, cc["p50"], color=MED_C, lw=1.6, ls=(0, (5, 2)),
                label="ERA5 median 1991–2020")
        ax.plot(obs.index, oo, color=OBS_C, lw=2.1, label="AIFS-ENS analysis (7-d mean)")

        # z in the FLIPPED frame, so +ve always means stronger / further poleward
        # for either cell rather than "more positive Ψ", which reverses in the SH
        sd = abs(c["sd"][-1])
        z = (oo[-1] - cc["mean"][-1]) / sd if np.isfinite(sd) and sd else np.nan
        unit = "" if is_psi else "°"
        ax.set_title(f"{title}   ·   latest {abs(o[-1]):.1f}{unit}"
                     f"   (median {abs(c['p50'][-1]):.1f}{unit}, {z:+.1f}σ)",
                     fontsize=10.5, fontweight="bold", loc="left")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8.5)
        if not is_psi:      # latitude axes read naturally as absolute degrees
            ax.yaxis.set_major_formatter(lambda v, _: f"{abs(v):.0f}°")
        summary[key] = {"latest": round(float(o[-1]), 2),
                        "median": round(float(c["p50"][-1]), 2),
                        "p10": round(float(c["p10"][-1]), 2),
                        "p90": round(float(c["p90"][-1]), 2),
                        "z": None if not np.isfinite(z) else round(float(z), 2),
                        "mean_4wk": round(float(np.nanmean(o[-28:])), 2),
                        "median_4wk": round(float(np.nanmean(c["p50"][-28:])), 2)}

    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, fontsize=8.4, ncol=4, loc="upper left",
               bbox_to_anchor=(0.004, 0.962), frameon=False)
    for ax in axes[1]:
        ax.tick_params(axis="x", rotation=0, labelsize=8.5)
    fig.suptitle("Hadley cell intensity & poleward edge — AIFS-ENS analysis vs ERA5 1991–2020",
                 fontsize=13, fontweight="bold", x=0.005, y=0.988, ha="left")
    fig.text(0.005, 0.005,
             "trailing 7-day means · intensity = peak |Ψ| in the cell · edge = Ψ(300–700 hPa)=0 "
             "crossing on the descending flank · bands = across-year spread of ERA5 weekly means",
             fontsize=8, color="0.35", ha="left")
    fig.tight_layout(rect=(0, 0.018, 1, 0.935))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"wrote {out}")

    jp = Path(a.json_out); jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps({
        "updated": f"{obs.index[-1]:%Y-%m-%dT%H:%MZ}",
        "source": "AIFS-ENS 0-h analysis; climatology ERA5 1991-2020",
        "latest": summary,
        "series": {"date": [f"{d:%Y-%m-%d}" for d in obs.index],
                   **{k: [None if not np.isfinite(v) else round(float(v), 2)
                          for v in obs[k].values] for k, *_ in PANELS}}}))
    print(f"wrote {jp}")
    for k, title, *_ in PANELS:
        s = summary[k]
        print(f"  {title:32s} latest {s['latest']:+7.2f}  median {s['median']:+7.2f}"
              f"  p10/p90 {s['p10']:+6.2f}/{s['p90']:+6.2f}  z {s['z']}"
              f"   4-wk {s['mean_4wk']:+6.2f} vs {s['median_4wk']:+6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
