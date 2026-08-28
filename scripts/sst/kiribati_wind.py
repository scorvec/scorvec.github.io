#!/usr/bin/env python3
"""Real-time equatorial westerly-wind-burst (WWB) monitor — Tarawa (Kiribati) METAR.

Tarawa / Bonriki (NGTA, 1.4°N 173°E) sits in the west-central equatorial Pacific where
westerly wind bursts spin up — the easterly trades briefly reverse to westerly, helping
push warm water and convection east and nudge an El Niño along. This pulls the hourly
NGTA wind (and Christmas Island / Kiritimati PLCH, 2°N 157°W, as the eastern-Pacific
contrast) and tracks the zonal wind component (westerly = positive), so a burst shows as
a clear excursion above the easterly-trade baseline.

Outputs (committed hourly by a GitHub Action):
  • assets/sst/kiribati_wind.webp  — recent zonal-wind time series + wind barbs
  • assets/sst/kiribati_wind.json  — latest ob per station, for the live wind-arrow widget

    python scripts/sst/kiribati_wind.py --out assets/sst/kiribati_wind.webp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "scripts") / "lib"))
from webget import get  # noqa: E402

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
CSV = HERE / "metar" / "kiribati_wind.csv"
API = "https://aviationweather.gov/api/data/metar?ids=NGTA,PLCH&format=json&hours={h}"
STN = {"NGTA": dict(key="tarawa", name="Tarawa", lat=1.4, lon=173.0),
       "PLCH": dict(key="christmas", name="Christmas Is.", lat=2.0, lon=-157.5)}
KT2MS = 0.514444
PLOT_DAYS = 21


COLS = [f"{k}_{x}" for k in ("tarawa", "christmas") for x in ("dir", "spd")]


def fetch(hours=72) -> pd.DataFrame:
    """Recent METARs → hourly wind dir/speed (kt) per station. Fails soft: a network error,
    non-JSON response (rate-limit/503), or a station simply not reporting just yields an empty
    (or partial) frame instead of raising — these remote stations report intermittently."""
    try:
        obs = json.loads(get(API.format(h=hours)).decode())
        if not isinstance(obs, list):
            obs = []
    except Exception as e:                                     # noqa: BLE001
        print(f"  METAR fetch failed ({repr(e)[:80]}) — keeping prior data", flush=True)
        obs = []
    rows = []
    for o in obs:
        s = STN.get(o.get("icaoId"))
        if not s:
            continue
        rows.append((pd.Timestamp(o["reportTime"]).tz_localize(None), s["key"],
                     o.get("wdir"), o.get("wspd")))
    df = pd.DataFrame(rows, columns=["time", "stn", "dir", "spd"])
    df["dir"] = pd.to_numeric(df["dir"], errors="coerce")     # 'VRB' → NaN
    df["spd"] = pd.to_numeric(df["spd"], errors="coerce")
    out = {}
    for k in ("tarawa", "christmas"):
        sub = df[df.stn == k]
        if sub.empty:                                          # station not in this batch → empty col
            out[f"{k}_dir"] = pd.Series(dtype=float); out[f"{k}_spd"] = pd.Series(dtype=float)
            continue
        sub = sub.set_index("time").sort_index()
        out[f"{k}_dir"] = sub["dir"].resample("1h").median()
        out[f"{k}_spd"] = sub["spd"].resample("1h").mean()
    return pd.DataFrame(out).reindex(columns=COLS)


def update_history(new: pd.DataFrame) -> pd.DataFrame:
    CSV.parent.mkdir(parents=True, exist_ok=True)
    if CSV.exists():
        # index_col=0: an empty-fetch cycle can write the file with an unnamed
        # index column (combine_first drops the name), which parse_dates=["time"]
        # then chokes on — read positionally and restore the name instead.
        old = pd.read_csv(CSV, index_col=0, parse_dates=[0])
        old.index.name = "time"
        hist = new.combine_first(old); hist.update(new)
    else:
        hist = new
    hist = hist.sort_index().apply(pd.to_numeric, errors="coerce")   # an empty fetch can object-ify cols
    hist.index.name = "time"
    hist.to_csv(CSV)
    return hist


def zonal(dir_deg, spd_kt):
    """Eastward wind component (m/s); westerly (270°) → positive."""
    return -spd_kt * KT2MS * np.sin(np.deg2rad(dir_deg))


def write_json(hist: pd.DataFrame, out: Path):
    latest = {}
    for icao, s in STN.items():
        k = s["key"]
        col_d, col_s = f"{k}_dir", f"{k}_spd"
        sub = hist[[col_d, col_s]].dropna()
        if sub.empty:
            continue
        t = sub.index[-1]; d = float(sub[col_d].iloc[-1]); sp = float(sub[col_s].iloc[-1])
        latest[icao] = dict(name=s["name"], lat=s["lat"], lon=s["lon"],
                            time=t.strftime("%Y-%m-%dT%H:%MZ"), dir=round(d),
                            spd_kt=round(sp), u_ms=round(float(zonal(d, sp)), 1))
    out.write_text(json.dumps({"updated": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%MZ"),
                               "stations": latest}, indent=2))
    return latest


def plot(hist: pd.DataFrame, out: Path):
    t0 = hist.index.max() - pd.Timedelta(days=PLOT_DAYS)
    h = hist[hist.index >= t0]
    uT = zonal(h["tarawa_dir"], h["tarawa_spd"])
    uC = zonal(h["christmas_dir"], h["christmas_spd"])
    if zonal(h["tarawa_dir"], h["tarawa_spd"]).dropna().empty:
        print("  no Tarawa wind in the plot window — skipping render", flush=True); return
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(uT.index, 0, uT.values, where=(uT.values > 0), interpolate=True,
                    color="#d62728", alpha=0.25, lw=0)
    ax.fill_between(uT.index, 0, uT.values, where=(uT.values <= 0), interpolate=True,
                    color="#1f77b4", alpha=0.18, lw=0)
    ax.plot(uC.index, uC.values, color="#888", lw=1.0, alpha=0.8, label="Christmas Is. (east)")
    ax.plot(uT.index, uT.values, color="#222", lw=1.8, label="Tarawa zonal wind")
    last = uT.dropna()
    # y-limits first (we place the barb row relative to the final top), with headroom
    # above the data so the wind-barb row sits INSIDE the axes, below the title.
    both = np.concatenate([uT.values, uC.values])
    dmin, dmax = np.nanmin(both), np.nanmax(both)
    ax.set_ylim(min(-6, dmin - 1), max(8, dmax + 1) + 2.5)
    # wind barbs (kt) along the top, just inside the axes (was overlapping the title), every ~6 h
    yb = ax.get_ylim()[1] - 1.0
    bb = h.iloc[::6].dropna(subset=["tarawa_dir", "tarawa_spd"])
    ub = -bb["tarawa_spd"] * np.sin(np.deg2rad(bb["tarawa_dir"]))
    vb = -bb["tarawa_spd"] * np.cos(np.deg2rad(bb["tarawa_dir"]))
    ax.barbs(mdates.date2num(bb.index), np.full(len(bb), yb), ub.values, vb.values,
             length=5.5, lw=0.5, color="#444", clip_on=True, zorder=5)
    ax.scatter([last.index[-1]], [last.iloc[-1]], s=44, color="#d62728" if last.iloc[-1] > 0 else "#1f77b4", zorder=6)
    ax.axhline(0, color="0.5", lw=0.8)
    # x-axis: start where data actually begins (METAR history is short) to avoid weeks of
    # leading whitespace; still capped at the PLOT_DAYS window.
    valid = pd.concat([uT.dropna(), uC.dropna()])
    x_start = max(valid.index.min(), t0) if not valid.empty else t0
    ax.set_xlim(x_start - pd.Timedelta(hours=12), last.index[-1] + pd.Timedelta(hours=12))
    ax.set_ylabel("zonal wind (m s⁻¹) · westerly +")
    ax.set_title("Equatorial westerly-wind-burst monitor — Tarawa (NGTA) METAR  ·  "
                 "westerly (red) ⇒ WWB / El Niño-favorable", fontsize=11)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9); ax.grid(alpha=0.15)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out} (Tarawa u={last.iloc[-1]:+.1f} m/s {last.index[-1]:%b %-d %HZ})", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/sst/kiribati_wind.webp")
    ap.add_argument("--hours", type=int, default=72)
    args = ap.parse_args(argv)
    hist = update_history(fetch(args.hours))
    if hist.empty or "tarawa_dir" not in hist.columns \
       or zonal(hist["tarawa_dir"], hist["tarawa_spd"]).dropna().empty:
        print("  no usable Tarawa wind data — leaving previous outputs in place", flush=True)
        return 0
    latest = write_json(hist, Path(args.out).with_suffix(".json"))
    plot(hist, Path(args.out))
    print("  latest:", {k: f"{v['dir']:.0f}°/{v['spd_kt']}kt u={v['u_ms']}" for k, v in latest.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
