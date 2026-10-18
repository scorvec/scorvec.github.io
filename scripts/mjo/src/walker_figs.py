#!/usr/bin/env python3
"""The Walker circulation's strength and position, analysed and forecast, against ERA5 - and against the El Niño events.

Two metrics of the 7-day-mean equatorial overturning Ψ_W at 00 UTC: the Pacific cell strength (walker.pac_index) and the
Indo-Pacific ascent longitude (walker.ascent_lon). The analysed weeks come from overturn_hist's archive (less the
analysis offset from ERA5), the forecasts from overturn_fc (every member less its model's drift), the ERA5 bands and the
events from build_walker_metrics.py - the same statistic throughout.

    python src/walker_figs.py series --out assets/sst/walker_timeseries.webp   # last DAYS days + the three ensembles
    python src/walker_figs.py enso --out assets/sst/walker_enso.webp           # this season against every El Niño
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
import build_walker_metrics as W
import hadley_enso as E
import overturn as O
import overturn_hist as H
from doycal import noleap_doy

DAYS = 120  # analysed days in the series
MODEL_C = {"aifs": "#b5482a", "ifs": "#5b3a8c", "gefs": "#2e7d32"}
PANELS = (("pac", "Pacific Walker cell strength", "10¹⁰ kg s⁻¹"), ("asc_lon", "Indo-Pacific ascent longitude", "°E"))
NOTE = (
    "strength = mean Ψ_W over 140°E–160°W, 300–700 hPa (negative = reversed) · ascent = centroid of the rising motion "
    "between 90°E and 120°W, 300–700 hPa"
)


def analysed() -> pd.DataFrame:
    """The metrics of every analysed 7-day mean in the last DAYS days, indexed by window end."""
    doc, ref = H.load(), O.load_ref()
    days = pd.DatetimeIndex(doc["date"])
    rows, idx = [], []
    for end in days[days > days.max() - pd.Timedelta(days=DAYS)]:
        w = H.week(doc, end, ref)
        if w is not None:
            rows.append(W.metrics(w[1]))
            idx.append(end)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def forecast(obs: pd.DataFrame):
    """overturn_fc's forecast if it carries the Walker metrics and does not sit behind the analysis."""
    fc = json.loads(E.FC.read_text()) if E.FC.exists() else None
    if not fc or any("asc_lon" not in d for d in fc["models"].values()):
        return None
    return fc if pd.Timestamp(fc["init"][:10]) >= obs.index[-1] - pd.Timedelta(days=2) else None


def series(out: Path):
    cl = xr.open_dataset(W.CLIM_OUT)
    obs = analysed()
    fc = forecast(obs)
    valid = pd.DatetimeIndex(fc["valid"]) if fc else pd.DatetimeIndex([])
    wk = fc["models"].get("gefs", {}).get("weeks") if fc else None
    wend = pd.DatetimeIndex(wk["end"]) if wk else pd.DatetimeIndex([])
    span = obs.index.union(valid).union(wend)
    mid = noleap_doy(span - pd.Timedelta(days=W.HALF))  # each window's midpoint
    last = mid[span.get_loc(obs.index[-1]) : span.get_loc(obs.index[-1]) + 1]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharex=True)
    for ax, (key, title, ylab) in zip(axes, PANELS):
        q = {s: E.clim_at(cl, key, mid, s) for s in ("p10", "p25", "p50", "p75", "p90")}
        ax.fill_between(span, q["p10"], q["p90"], color="#c9d8e6", lw=0, label="ERA5 p10–p90")
        ax.fill_between(span, q["p25"], q["p75"], color="#9dbad3", lw=0, label="p25–p75")
        ax.plot(span, q["p50"], color="#8a5a00", lw=1.6, ls=(0, (5, 2)), label="ERA5 median 1991–2020")
        ax.plot(obs.index, obs[key], color="#0b3d6b", lw=2.1, label="analysis (7-day mean)")
        if fc:
            for m, d in fc["models"].items():
                p = E.plume(np.array(d[key], float))
                col = MODEL_C.get(m, "0.3")
                ax.fill_between(valid, p[10], p[90], color=col, alpha=0.13, lw=0)
                ax.plot(valid, p[50], color=col, lw=1.9, label=f"{d['label']} median, 10–90 % ({d['members']} members)")
            ax.axvline(valid[0], color="0.55", lw=0.8, ls=":")
        if wk:
            E.weeks_plume(ax, np.array(wk[key], float), wend, MODEL_C["gefs"])
        sd, mu = E.clim_at(cl, key, last, "sd")[0], E.clim_at(cl, key, last, "mean")[0]
        z = (obs[key].iloc[-1] - mu) / sd if np.isfinite(sd) and sd else np.nan
        val = f"{obs[key].iloc[-1]:.0f}°E" if key == "asc_lon" else f"{obs[key].iloc[-1]:+.2f}"
        ax.set_title(f"{title} · latest {val} ({z:+.1f}σ)", fontsize=10.5, fontweight="bold", loc="left")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    run = f" + forecasts from {fc['init'][:10]} 00Z" if fc else ""
    E.finish(
        fig,
        axes,
        f"Walker circulation strength & position — analysis{run} vs ERA5 1991–2020",
        f"7-day means at 00 UTC · {NOTE}\nanalysis less its mean offset from ERA5 · forecasts: members less their "
        f"model's drift against ERA5 (median and 10–90 %){E.WEEKS_NOTE if wk else ''} · bands = ERA5's 7-day means "
        "across years",
        out,
    )


def enso(out: Path):
    ev, cl = xr.open_dataset(W.EVENTS_OUT), xr.open_dataset(W.CLIM_OUT)
    y, obs, fc = E.this_season(pd.Timestamp.now("UTC").tz_localize(None), lambda w: W.metrics(w[1]), W.KEYS)
    x = E.season_days(y)
    doy = noleap_doy(x)
    strong = [e for e in ev.event.values if float(ev.peak.sel(event=e)) >= E.STRONG]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharex=True)
    for ax, (key, title, ylab) in zip(axes, PANELS):
        lo, hi = E.clim_at(cl, key, doy, "p10"), E.clim_at(cl, key, doy, "p90")
        ax.fill_between(x, lo, hi, color="#e3e9ef", lw=0, label="ERA5 1991–2020, 10–90 %")
        ax.plot(x, E.clim_at(cl, key, doy, "p50"), color="#8a5a00", lw=1.3, ls=(0, (5, 2)), label="ERA5 median")
        moderate = [e for e in ev.event.values if e not in strong]
        for e in moderate:
            ax.plot(
                x,
                ev[key].sel(event=e).values,
                color="#9aa3ad",
                lw=0.7,
                alpha=0.45,
                label="moderate El Niño (RONI +1.0 to +2.0)" if e == moderate[0] else None,
            )
        for c, e in zip(E.STRONG_C, strong):
            ax.plot(
                x, ev[key].sel(event=e).values, color=c, lw=1.4, label=f"{e} (RONI {float(ev.peak.sel(event=e)):+.1f})"
            )
        shown = [obs[key].values] if len(obs) else []
        if len(obs):
            ax.plot(obs.index, obs[key], color="#111", lw=2.6, label=f"{y}/{str(y + 1)[2:]} analysis")
        if fc:
            fmid, mem, init = fc
            q = E.plume(mem[key])
            shown += [q[10], q[90]]
            ax.fill_between(fmid, q[10], q[90], color="#111", alpha=0.12, lw=0)
            ax.plot(
                fmid,
                q[50],
                color="#111",
                lw=1.8,
                ls=(0, (3, 1.5)),
                label=f"forecast {init} (3 ensembles, median and 10–90 %)",
            )
        E.limits(ax, ev[key].values, *shown)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    E.finish(
        fig,
        axes,
        "Walker circulation through El Niño: this season against every El Niño since 1963 (ERA5)",
        f"7-day means at 00 UTC, at their midpoint · events: July–March RONI peak ≥ +1.0 (CPC)\n{NOTE}\nthis season: analysis less its mean offset from ERA5; forecasts: members less their model's drift "
        "against ERA5 · each event on its own (no composite) · 1991/92 also carries Pinatubo",
        out,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("figure", choices=("series", "enso"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (series if a.figure == "series" else enso)(Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
