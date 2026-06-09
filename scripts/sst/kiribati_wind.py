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
import urllib.request
from pathlib import Path

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


def fetch(hours=72) -> pd.DataFrame:
    """Recent METARs → hourly wind dir/speed (kt) per station."""
    obs = json.loads(urllib.request.urlopen(API.format(h=hours), timeout=60).read().decode())
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
        sub = df[df.stn == k].set_index("time").sort_index()
        out[f"{k}_dir"] = sub["dir"].resample("1h").median()
        out[f"{k}_spd"] = sub["spd"].resample("1h").mean()
    return pd.DataFrame(out)


def update_history(new: pd.DataFrame) -> pd.DataFrame:
    CSV.parent.mkdir(parents=True, exist_ok=True)
    if CSV.exists():
        old = pd.read_csv(CSV, parse_dates=["time"]).set_index("time")
        hist = new.combine_first(old); hist.update(new)
    else:
        hist = new
    hist = hist.sort_index()
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
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(uT.index, 0, uT.values, where=(uT.values > 0), interpolate=True,
                    color="#d62728", alpha=0.25, lw=0)
    ax.fill_between(uT.index, 0, uT.values, where=(uT.values <= 0), interpolate=True,
                    color="#1f77b4", alpha=0.18, lw=0)
    ax.plot(uC.index, uC.values, color="#888", lw=1.0, alpha=0.8, label="Christmas Is. (east)")
    ax.plot(uT.index, uT.values, color="#222", lw=1.8, label="Tarawa zonal wind")
    # wind barbs (kt) along the top, Tarawa, every ~6 h
    yb = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 6
    bb = h.iloc[::6].dropna(subset=["tarawa_dir", "tarawa_spd"])
    ub = -bb["tarawa_spd"] * np.sin(np.deg2rad(bb["tarawa_dir"]))
    vb = -bb["tarawa_spd"] * np.cos(np.deg2rad(bb["tarawa_dir"]))
    ax.barbs(mdates.date2num(bb.index), np.full(len(bb), yb * 1.18), ub.values, vb.values,
             length=5.5, lw=0.5, color="#444", clip_on=False)
    last = uT.dropna()
    ax.scatter([last.index[-1]], [last.iloc[-1]], s=44, color="#d62728" if last.iloc[-1] > 0 else "#1f77b4", zorder=6)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylim(min(-12, np.nanmin(uT.values) - 1), max(8, np.nanmax(uT.values) + 1))
    ax.set_xlim(t0, last.index[-1] + pd.Timedelta(hours=12))
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
    latest = write_json(hist, Path(args.out).with_suffix(".json"))
    plot(hist, Path(args.out))
    print("  latest:", {k: f"{v['dir']:.0f}°/{v['spd_kt']}kt u={v['u_ms']}" for k, v in latest.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
