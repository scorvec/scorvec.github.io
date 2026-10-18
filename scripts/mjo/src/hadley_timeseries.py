#!/usr/bin/env python3
"""Hadley cell INTENSITY and POLEWARD EDGE, analysed and forecast, against the ERA5 1991-2020 distribution.

Four panels - intensity and poleward edge of each cell - with:
  * the analysed 7-day means (overturn_hist.py's archive: daily means of the AIFS analysis, less its mean offset from
    ERA5 at the season), one a day for the last DAYS days;
  * the AIFS-ENS, IFS-ENS and GEFS forecasts to day 15 (overturn_fc.py: every member less its model's drift against
    ERA5, the same trailing 7-day windows, the early ones holding analysed days): the members' median and 10-90 %;
  * the ERA5 p10-p90 / p25-p75 bands and median of the same statistic (build_hadley_clim.py: the metrics of ERA5's
    7-day-mean circulation), at each window's midpoint.
Every value is read on the climatology's 1.5 deg latitude bands (build_hadley_clim.metrics).

    python src/hadley_timeseries.py --out assets/sst/hadley_timeseries.webp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import build_hadley_clim as B
import hadley_enso as E
import overturn as O
import overturn_hist as H
from doycal import noleap_doy

DAYS = 120  # analysed days shown
OBS_C = "#0b3d6b"  # analysed series
MED_C = "#8a5a00"  # climatological median
B1 = "#c9d8e6"  # p10-p90
B2 = "#9dbad3"  # p25-p75
MODEL_C = {"aifs": "#b5482a", "ifs": "#5b3a8c", "gefs": "#2e7d32"}
PANELS = (
    ("nh_psi", "Northern cell intensity", "|Ψ|max  (10¹⁰ kg s⁻¹)", True),
    ("nh_edge_pole", "Northern cell poleward edge", "latitude (°N)", False),
    ("sh_psi", "Southern cell intensity", "|Ψ|max  (10¹⁰ kg s⁻¹)", True),
    ("sh_edge_pole", "Southern cell poleward edge", "latitude (°S)", False),
)


def analysed() -> pd.DataFrame:
    """The metrics of every analysed 7-day mean in the last DAYS days (archive windows, less the analysis offset)."""
    doc, ref = H.load(), O.load_ref()
    days = pd.DatetimeIndex(doc["date"])
    rows, idx = [], []
    for end in days[days > days.max() - pd.Timedelta(days=DAYS)]:
        w = H.week(doc, end, ref)
        if w is None:
            continue
        rows.append(B.metrics(w[0], O.LAT))
        idx.append(end)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def clim_at(cl: xr.Dataset, key: str, ends) -> dict:
    """The climatological stats for windows ending on `ends`, at each window's midpoint (common-year days,
    doycal.noleap_doy; 29 Feb = 59.5 takes the mean of the 28 Feb and 1 March slots)."""
    x = np.asarray(noleap_doy(pd.DatetimeIndex(ends) - pd.Timedelta(days=(B.WIN - 1) / 2)), float)
    d = 0.5 * (
        cl[key].sel(doy=xr.DataArray(np.floor(x).astype(int), dims="t"))
        + cl[key].sel(doy=xr.DataArray(np.ceil(x).astype(int), dims="t"))
    )
    return {s: d.sel(stat=s).values for s in ("p10", "p25", "p50", "p75", "p90", "mean", "sd")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/sst/hadley_timeseries.webp")
    ap.add_argument("--forecast", default="assets/sst/data/overturn_fc.json")
    a = ap.parse_args()
    cl = xr.open_dataset(B.OUT)
    obs = analysed()
    if obs.empty:
        print("no analysed 7-day means in the archive", file=sys.stderr)
        return 1
    fc = json.loads(Path(a.forecast).read_text()) if Path(a.forecast).exists() else None
    if fc is not None and pd.Timestamp(fc["init"][:10]) < obs.index[-1] - pd.Timedelta(days=2):
        fc = None  # a stale forecast would sit behind the analysis
    valid = pd.DatetimeIndex(fc["valid"]) if fc else pd.DatetimeIndex([])
    span = obs.index.union(valid)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.0), sharex=True)
    for ax, (key, title, ylab, is_psi) in zip(axes.ravel(), PANELS):
        flip = -1.0 if key.startswith("sh") else 1.0  # both cells on "bigger = stronger / further poleward"
        c = {k: v * flip for k, v in clim_at(cl, key, span).items()}
        ax.fill_between(
            span, np.minimum(c["p10"], c["p90"]), np.maximum(c["p10"], c["p90"]), color=B1, lw=0, label="ERA5 p10–p90"
        )
        ax.fill_between(
            span, np.minimum(c["p25"], c["p75"]), np.maximum(c["p25"], c["p75"]), color=B2, lw=0, label="p25–p75"
        )
        ax.plot(span, c["p50"], color=MED_C, lw=1.6, ls=(0, (5, 2)), label="ERA5 median 1991–2020")
        o = obs[key].values.astype(float) * flip
        ax.plot(obs.index, o, color=OBS_C, lw=2.1, label="analysis (7-day mean)")
        if fc:
            for m, d in fc["models"].items():
                x = np.array(d[key], float) * flip  # (member, lead)
                ok = np.isfinite(x).sum(0) >= 0.5 * x.shape[0]  # a lead where most members resolve the cell
                q10, q50, q90 = (np.where(ok, np.nanpercentile(x, q, axis=0), np.nan) for q in (10, 50, 90))
                col = MODEL_C.get(m, "0.3")
                ax.fill_between(valid, q10, q90, color=col, alpha=0.13, lw=0)
                ax.plot(valid, q50, color=col, lw=1.9, label=f"{d['label']} median, 10–90 % ({d['members']} members)")
            ax.axvline(valid[0], color="0.55", lw=0.8, ls=":")
        sd = abs(clim_at(cl, key, obs.index[-1:])["sd"][0])
        z = (o[-1] - clim_at(cl, key, obs.index[-1:])["mean"][0] * flip) / sd if np.isfinite(sd) and sd else np.nan
        unit = "" if is_psi else "°"
        ax.set_title(
            f"{title} · latest {abs(o[-1]):.1f}{unit} ({z:+.1f}σ)",
            fontsize=10.5,
            fontweight="bold",
            loc="left",
        )
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8.5)
        if not is_psi:  # latitude axes read naturally as absolute degrees
            ax.yaxis.set_major_formatter(lambda v, _: f"{abs(v):.0f}°")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    run = f" + forecasts from {fc['init'][:10]} 00Z" if fc else ""
    E.finish(
        fig,
        axes,
        f"Hadley cell intensity & poleward edge — analysis{run} vs ERA5 1991–2020",
        "7-day means at 00 UTC · intensity = peak |Ψ| in the cell · edge = Ψ(300–700 hPa)=0 on the descending flank · "
        "bands = ERA5's 7-day means across years\nanalysis less its mean offset from ERA5 · forecasts: members less their "
        "model's drift against ERA5 (median and 10–90 %)",
        Path(a.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
