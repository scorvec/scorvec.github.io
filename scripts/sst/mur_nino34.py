#!/usr/bin/env python3
"""Niño-3.4 ABSOLUTE sea-surface temperature from NASA JPL MUR v4.1 (1 km GHRSST L4),
tracked against its own day-of-year climatology and against NOAA OISST v2.1.

Why a second Niño-3.4 tracker
-----------------------------
This is not a duplicate of the OISST index. MUR is an INDEPENDENT analysis with a
different definition: it is a GHRSST **foundation** SST (the temperature below the
diurnal thermocline), built from a different sensor mix at 1 km. OISST v2.1 is a
**bulk** SST on a 0.25° grid, anchored to drifting buoys. The two therefore measure
subtly different quantities and a persistent offset between them is EXPECTED — it is
a property of the definitions, not evidence that either is wrong. What is worth
watching is when the offset MOVES, because that means the two analyses are
disagreeing about the ocean rather than about their own conventions.

Absolute, not anomaly. The seasonal cycle in Niño-3.4 is ~2 °C peak-to-peak and is
the thing an anomaly plot throws away; showing the real temperature against the
climatological envelope says both "how warm is it" and "how unusual is that" at once.

Method notes
------------
* The box mean is area-weighted by cos(latitude). Over 5°S–5°N that is a <0.4%
  correction, but it costs nothing and keeps the definition honest.
* The box is sampled at 0.25° (ERDDAP stride 25) rather than the native 0.01°.
  For a 50°×10° BOX MEAN the sub-sampled mean is an unbiased estimate and the extra
  25× of data buys no accuracy — MUR's value here is that it is a different
  analysis, not that it resolves kilometre structure inside a basin average.
* MUR begins 2002-06-01, so its climatology cannot be the WMO 1991–2020 period. A
  2003–2022 20-year base is used and is labelled as such; it is NOT comparable to
  OISST anomalies computed on 1991–2020.

    python scripts/sst/mur_nino34.py              # incremental update + plot
    python scripts/sst/mur_nino34.py --backfill   # build the full record first
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SITE_ROOT = (Path(os.environ["SST_SITE_ROOT"]).resolve()
             if os.environ.get("SST_SITE_ROOT") else HERE.parent.parent)
ASSETS = SITE_ROOT / "assets" / "sst"
SERIES = HERE / "data" / "mur" / "nino34_series.json"
OUT_PNG = ASSETS / "mur_nino34.webp"
OUT_JSON = ASSETS / "data" / "mur_nino34.json"
OISST_JSON = ASSETS / "data" / "enso_daily.json"

HOSTS = ["https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41",
         "https://polarwatch.noaa.gov/erddap/griddap/jplMURSST41"]
LAT = (-5.0, 5.0)
LON = (-170.0, -120.0)          # Niño-3.4; does not cross the dateline in MUR's -180..180
STRIDE = 25                     # 0.25° sampling — see method note above
MUR_START = "2002-06-01"
CLIM_Y0, CLIM_Y1 = 2003, 2022   # MUR cannot do 1991-2020; 20 whole years
CHUNK_DAYS = 60
UA = {"User-Agent": "Mozilla/5.0 (scorvec.com research monitor)"}


def _load() -> pd.Series:
    if SERIES.exists():
        d = json.loads(SERIES.read_text())
        return pd.Series(d["sst"], index=pd.to_datetime(d["dates"])).sort_index()
    return pd.Series(dtype=float)


def _save(s: pd.Series) -> None:
    SERIES.parent.mkdir(parents=True, exist_ok=True)
    tmp = SERIES.with_suffix(".tmp")
    tmp.write_text(json.dumps({"dates": [f"{d:%Y-%m-%d}" for d in s.index],
                               "sst": [None if not np.isfinite(v) else round(float(v), 4)
                                       for v in s.values]}))
    tmp.replace(SERIES)


def fetch_range(t0: pd.Timestamp, t1: pd.Timestamp) -> pd.Series:
    """Area-weighted Niño-3.4 mean SST (°C) for every MUR day in [t0, t1]."""
    import xarray as xr
    s = f":{STRIDE}:"
    suf = (f".nc?analysed_sst%5B({t0:%Y-%m-%d}T09:00:00Z):({t1:%Y-%m-%d}T09:00:00Z)%5D"
           f"%5B({LAT[0]}){s}({LAT[1]})%5D%5B({LON[0]}){s}({LON[1]})%5D")
    last = None
    for attempt in range(1, 5):
        host = HOSTS[(attempt - 1) % len(HOSTS)]
        try:
            req = urllib.request.Request(host + suf, headers=UA)
            p = Path(tempfile.mktemp(suffix=".nc"))
            with urllib.request.urlopen(req, timeout=420) as r, open(p, "wb") as f:
                f.write(r.read())
            with xr.open_dataset(p) as ds:
                v = ds["analysed_sst"]
                if str(v.attrs.get("units", "")).lower().startswith("k"):
                    v = v - 273.15
                w = np.cos(np.deg2rad(v.latitude))
                m = v.weighted(w).mean(dim=("latitude", "longitude"), skipna=True).load()
                out = pd.Series(m.values, index=pd.to_datetime(m.time.values).normalize())
            p.unlink(missing_ok=True)
            return out[np.isfinite(out.values)]
        except Exception as e:                                   # noqa: BLE001
            last = e
            if "not found" in repr(e).lower() or "no data" in repr(e).lower():
                return pd.Series(dtype=float)
            time.sleep(8 * attempt)
    print(f"  {t0:%Y-%m-%d}..{t1:%Y-%m-%d}: FAILED {repr(last)[:90]}", flush=True)
    return pd.Series(dtype=float)


def update(full: bool) -> pd.Series:
    s = _load()
    # tz_localize(None): utcnow() is tz-aware in pandas >=2 and the series index
    # is naive, so a bare comparison raises
    end = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=2)
    start = pd.Timestamp(MUR_START) if full or s.empty else s.index.max() + pd.Timedelta(days=1)
    if start > end:
        print(f"up to date ({len(s)} days through {s.index.max():%Y-%m-%d})")
        return s
    have = set(s.index)
    cur = start
    while cur <= end:
        stop = min(cur + pd.Timedelta(days=CHUNK_DAYS - 1), end)
        if not all((cur + pd.Timedelta(days=i)) in have
                   for i in range((stop - cur).days + 1)):
            got = fetch_range(cur, stop)
            if len(got):
                s = pd.concat([s[~s.index.isin(got.index)], got]).sort_index()
                _save(s)
                print(f"  {cur:%Y-%m-%d}..{stop:%Y-%m-%d}: +{len(got)} days "
                      f"({len(s)} total)", flush=True)
        cur = stop + pd.Timedelta(days=1)
    return s


def climatology(s: pd.Series):
    """Smoothed day-of-year mean and 10th/90th percentiles over CLIM_Y0..CLIM_Y1."""
    base = s[(s.index.year >= CLIM_Y0) & (s.index.year <= CLIM_Y1)]
    if base.empty:
        return None
    doy = base.index.dayofyear.where(~((base.index.month == 2) & (base.index.day == 29)), 59)
    df = pd.DataFrame({"doy": doy, "v": base.values})
    g = df.groupby("doy")["v"]
    out = pd.DataFrame({"mean": g.mean(), "p10": g.quantile(.10), "p90": g.quantile(.90)})
    out = out.reindex(range(1, 367)).interpolate(limit_direction="both")
    # circular 15-day smooth: day-to-day sampling noise is not climate
    w = 15
    pad = pd.concat([out.iloc[-w:], out, out.iloc[:w]])
    return pad.rolling(w, center=True, min_periods=1).mean().iloc[w:-w]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="walk the whole MUR record from 2002-06 (one-time, ~1 h)")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    s = _load() if a.plot_only else update(a.backfill)
    if s.empty:
        print("no MUR series yet", file=sys.stderr); return 1
    print(f"MUR Niño-3.4: {len(s)} days, {s.index.min():%Y-%m-%d} … {s.index.max():%Y-%m-%d}")
    clim = climatology(s)
    payload = {"updated": f"{s.index.max():%Y-%m-%d}",
               "source": "NASA JPL MUR v4.1 (GHRSST L4 foundation SST) via NOAA CoastWatch ERDDAP",
               "box": {"lat": list(LAT), "lon": list(LON), "sample_deg": STRIDE / 100},
               "clim_base": [CLIM_Y0, CLIM_Y1],
               "latest_c": round(float(s.iloc[-1]), 3),
               "series": {"dates": [f"{d:%Y-%m-%d}" for d in s.index[-400:]],
                          "sst_c": [round(float(v), 3) for v in s.values[-400:]]}}
    if clim is not None:
        d = int(s.index[-1].dayofyear)
        payload["clim_today_c"] = round(float(clim["mean"].loc[d]), 3)
        payload["departure_c"] = round(float(s.iloc[-1] - clim["mean"].loc[d]), 3)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT_JSON.relative_to(SITE_ROOT)}")
    render(s, clim)
    return 0


def render(s: pd.Series, clim) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    oi = None
    if OISST_JSON.exists():
        d = json.loads(OISST_JSON.read_text())["daily"]
        if "nino34_abs" in d:
            oi = pd.Series(d["nino34_abs"], index=pd.to_datetime(d["dates"])).sort_index()

    yrs = 2
    t0 = s.index.max() - pd.Timedelta(days=365 * yrs)
    cur = s[s.index >= t0]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.2, 7.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2.6, 1]})

    if clim is not None:
        doy = cur.index.dayofyear.where(~((cur.index.month == 2) & (cur.index.day == 29)), 59)
        cm = clim["mean"].reindex(doy).values
        c10 = clim["p10"].reindex(doy).values
        c90 = clim["p90"].reindex(doy).values
        ax.fill_between(cur.index, c10, c90, color="#cfe0ee", lw=0,
                        label=f"MUR p10–p90 ({CLIM_Y0}–{CLIM_Y1})")
        ax.plot(cur.index, cm, color="#8a5a00", lw=1.5, ls=(0, (5, 2)),
                label=f"MUR climatology ({CLIM_Y0}–{CLIM_Y1})")
    ax.plot(cur.index, cur.values, color="#0b3d6b", lw=2.0,
            label="MUR v4.1 (1 km, foundation SST)")
    if oi is not None:
        oo = oi[oi.index >= t0]
        ax.plot(oo.index, oo.values, color="#c0392b", lw=1.3, alpha=0.85,
                label="NOAA OISST v2.1 (0.25°, bulk SST)")
    ax.set_ylabel("Niño-3.4 SST (°C)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8.4, loc="upper left", ncol=2, framealpha=0.9)
    lat_txt = f"{abs(LAT[0]):.0f}°S–{LAT[1]:.0f}°N, {abs(LON[0]):.0f}°W–{abs(LON[1]):.0f}°W"
    ax.set_title(f"Niño-3.4 absolute sea-surface temperature — MUR 1 km vs OISST   ·   {lat_txt}",
                 fontsize=13.5, fontweight="bold", loc="left")

    if oi is not None:
        both = pd.concat([cur.rename("mur"), oi.rename("oi")], axis=1).dropna()
        both = both[both.index >= t0]
        diff = both["mur"] - both["oi"]
        bx.axhline(0, color="0.6", lw=0.8)
        bx.plot(both.index, diff.values, color="#5b2c8d", lw=1.4)
        bx.fill_between(both.index, 0, diff.values, color="#5b2c8d", alpha=0.16)
        bx.set_ylabel("MUR − OISST (°C)", fontsize=9.5)
        bx.grid(alpha=0.25, lw=0.6)
        bx.set_title(f"definition offset — mean {diff.mean():+.2f} °C, sd {diff.std():.2f} "
                     f"(foundation vs bulk SST; a steady offset is expected, a MOVING one is news)",
                     fontsize=9.5, loc="left", color="0.35")
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for x in (ax, bx):
        x.tick_params(labelsize=9)
    # two short lines: one long line overran the canvas at this figure width
    fig.text(0.006, 0.026,
             "NASA JPL MUR v4.1 GHRSST L4 via NOAA CoastWatch ERDDAP  ·  box mean area-weighted "
             f"by cos(lat), sampled at {STRIDE/100:.2f}°",
             fontsize=7.8, color="0.42", ha="left")
    fig.text(0.006, 0.006,
             f"MUR begins 2002-06, so its climatology is {CLIM_Y0}–{CLIM_Y1} — NOT the 1991–2020 "
             "base behind OISST anomalies, and a different SST definition besides",
             fontsize=7.8, color="0.42", ha="left")
    fig.tight_layout(rect=(0, 0.042, 1, 1))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(SITE_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
