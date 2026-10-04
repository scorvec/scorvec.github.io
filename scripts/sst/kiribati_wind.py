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
WWB_THRESH = 5.0            # m/s daily-mean westerly: the usual burst criterion
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
    """Two stacked panels in the site's figure framing (user 2026-09-07: "the Christmas Island stuff is
    messy"): Tarawa on top, Christmas Island below, each as DAILY-MEAN zonal-wind bars (westerly red,
    easterly blue) with the hourly obs as light dots rather than line fragments, the WWB threshold
    (+5 m/s daily mean) dashed, and a thin barb row per station every 12 h."""
    t0 = hist.index.max() - pd.Timedelta(days=PLOT_DAYS)
    h = hist[hist.index >= t0]
    series = {}
    for key, name in (("tarawa", "Tarawa (NGTA, 1.4°N 173°E) — west-central Pacific"), ("christmas", "Christmas Island (PLCH, 2°N 157°W) — eastern Pacific")):
        u = zonal(h[f"{key}_dir"], h[f"{key}_spd"])
        series[key] = (name, u, u.resample("1D").mean(), u.resample("1D").count())
    if series["tarawa"][1].dropna().empty:
        print("  no Tarawa wind in the plot window — skipping render", flush=True); return
    W, top, bot = 14.0, 1.0, 0.55; ph = 2.6; gap = 0.35; H = top + 2 * ph + gap + bot
    fig = plt.figure(figsize=(W, H))
    valid = pd.concat([series["tarawa"][1].dropna(), series["christmas"][1].dropna()])
    x_start = max(valid.index.min(), t0) if not valid.empty else t0
    x_end = hist.index.max() + pd.Timedelta(hours=12)
    axes = []
    allu = np.concatenate([series[k][1].dropna().values for k in ("tarawa", "christmas")] or [np.array([0.0])])
    lo = min(-6.0, float(allu.min()) - 1) if allu.size else -6.0; hi = max(8.0, float(allu.max()) + 1) if allu.size else 8.0   # one scale for both islands
    for k, key in enumerate(("tarawa", "christmas")):
        name, u, ud, nd = series[key]
        y0 = (bot + (ph + gap) * (1 - k)) / H
        ax = fig.add_axes([0.06, y0, 0.905, ph / H]); axes.append(ax)
        ok = nd >= 4                                                        # a daily mean needs a few obs
        col = np.where(ud.values > 0, "#c0392b", "#3f6f8f")
        ax.bar(ud.index[ok] + pd.Timedelta(hours=12), ud.values[ok], width=0.9, color=col[ok], alpha=0.55, lw=0, label="daily mean", zorder=2)
        ax.scatter(u.index, u.values, s=5, color="#222", alpha=0.45, lw=0, label="hourly METAR", zorder=3)
        ax.axhline(0, color="#666", lw=1.0, zorder=1)
        ax.axhline(WWB_THRESH, color="#c0392b", lw=0.9, ls="--", zorder=1)
        ax.text(x_start - pd.Timedelta(hours=6), WWB_THRESH + 0.25, "WWB threshold", color="#c0392b", fontsize=7.5, va="bottom", ha="left", zorder=4)
        last = u.dropna()
        if len(last):
            ax.scatter([last.index[-1]], [last.iloc[-1]], s=46, color="#c0392b" if last.iloc[-1] > 0 else "#3f6f8f", zorder=6, edgecolor="#fff", lw=1)
            ax.annotate(f"{last.iloc[-1]:+.1f}", (last.index[-1], last.iloc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, va="center", color="#222")
        ax.set_ylim(lo, hi + 2.2)
        bb = h[h.index.minute == 0].iloc[::12].dropna(subset=[f"{key}_dir", f"{key}_spd"])   # barbs every 12 h
        if len(bb):
            ub = -bb[f"{key}_spd"] * np.sin(np.deg2rad(bb[f"{key}_dir"])); vb = -bb[f"{key}_spd"] * np.cos(np.deg2rad(bb[f"{key}_dir"]))
            ax.barbs(mdates.date2num(bb.index), np.full(len(bb), hi + 1.2), ub.values, vb.values, length=5, lw=0.5, color="#444", clip_on=True, zorder=5)
        ax.set_xlim(x_start - pd.Timedelta(hours=12), x_end)
        ax.set_ylabel("zonal wind (m s⁻¹)\nwesterly +", fontsize=8.5)
        ax.set_title(name, fontsize=9.5, loc="left", pad=3)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
        ax.tick_params(labelsize=8); ax.grid(axis="y", color="#e8e6e0", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if k == 0:
            ax.tick_params(labelbottom=False)
            ax.legend(loc="lower left", fontsize=7.5, frameon=False, ncol=2)
    fig.text(0.02, 1 - 0.14 / H, "Equatorial westerly-wind-burst monitor · Tarawa and Christmas Island METAR", fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.50 / H, f"Bars: daily-mean zonal wind (red westerly, blue easterly; days with fewer than 4 obs are left blank); dots: hourly reports; dashed: +{WWB_THRESH:.0f} m/s daily mean, the westerly-burst threshold; barbs: the reported wind every 12 h (kt). A westerly burst over Tarawa pushes warm water east and favours El Niño.",
             fontsize=8.4, color="#444", va="top", wrap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    lastT = series["tarawa"][1].dropna()
    print(f"  saved {out} (Tarawa u={lastT.iloc[-1]:+.1f} m/s {lastT.index[-1]:%b %-d %HZ})", flush=True)


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
