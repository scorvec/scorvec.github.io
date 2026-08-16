#!/usr/bin/env python3
"""Degree-day surprise tracker — USA / Canada / Brazil / Argentina.

The tradeable quantity in energy weather isn't the forecast, it's the
REVISION: what did today's runs add to or remove from the heating/cooling
demand picture versus the runs people positioned on days ago. Per cycle
this pulls population-weighted country 2-m temperatures from

  - AIFS-ENS ensemble mean + spread ("em"/"es" open-data products — one
    small message per step, 6-hourly to day 15), and
  - ECMWF HRES (oper fc, 6-hourly to day 10),

converts them to daily HDD/CDD (base 18.3 C = 65 F, both models on the
same metro/pop weights), archives every cycle to a committed JSON, and
renders one figure: latest daily bars + the cumulative degree-day
REVISION curve across the archived init cycles. Systematic model or
diurnal-sampling bias cancels in the revisions, which is the point.

Countries via metro population weights (nearest 0.25-deg gridpoint per
metro). Europe/Asia get added as new COUNTRIES entries.

    python scripts/energy/dd_surprise.py                  # latest cycle
    python scripts/energy/dd_surprise.py --date 20260816 --time 00
    python scripts/energy/dd_surprise.py --backfill 6     # seed the archive
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ARCHIVE = HERE / "data" / "dd_archive.json"
OUTPNG = REPO / "assets" / "energy" / "dd_surprise.webp"
BASE_C = 18.3                      # 65 F, common HDD/CDD base
MAX_CYCLES = 40                    # ~3 weeks of 2 cycles/day in the archive

# (name, lat, lon, population millions) — metro areas
COUNTRIES = {
    "USA": [
        ("NYC", 40.7, -74.0, 19.8), ("LA", 34.05, -118.2, 13.0),
        ("Chicago", 41.9, -87.6, 9.4), ("Dallas", 32.8, -96.8, 7.6),
        ("Houston", 29.8, -95.4, 7.1), ("DC", 38.9, -77.0, 6.4),
        ("Philadelphia", 40.0, -75.2, 6.2), ("Atlanta", 33.7, -84.4, 6.1),
        ("Miami", 25.8, -80.2, 6.1), ("Boston", 42.4, -71.1, 4.9),
        ("Phoenix", 33.4, -112.1, 4.8), ("SF", 37.8, -122.4, 4.6),
        ("Riverside", 33.9, -117.4, 4.6), ("Detroit", 42.3, -83.1, 4.3),
        ("Seattle", 47.6, -122.3, 4.0), ("Minneapolis", 45.0, -93.3, 3.7),
        ("San Diego", 32.7, -117.2, 3.3), ("Tampa", 28.0, -82.5, 3.2),
        ("Denver", 39.7, -105.0, 3.0), ("St. Louis", 38.6, -90.2, 2.8),
        ("Baltimore", 39.3, -76.6, 2.8), ("Charlotte", 35.2, -80.8, 2.7),
        ("Orlando", 28.5, -81.4, 2.7), ("San Antonio", 29.4, -98.5, 2.6),
        ("Portland", 45.5, -122.7, 2.5),
    ],
    "Canada": [
        ("Toronto", 43.7, -79.4, 6.4), ("Montreal", 45.5, -73.6, 4.3),
        ("Vancouver", 49.3, -123.1, 2.7), ("Calgary", 51.0, -114.1, 1.6),
        ("Edmonton", 53.5, -113.5, 1.5), ("Ottawa", 45.4, -75.7, 1.5),
        ("Winnipeg", 49.9, -97.1, 0.85), ("Quebec City", 46.8, -71.2, 0.84),
        ("Hamilton", 43.3, -79.9, 0.79), ("London ON", 43.0, -81.2, 0.55),
    ],
    "Brazil": [
        ("Sao Paulo", -23.55, -46.6, 22.0), ("Rio", -22.9, -43.2, 13.6),
        ("Belo Horizonte", -19.9, -43.9, 6.1), ("Brasilia", -15.8, -47.9, 4.7),
        ("Fortaleza", -3.7, -38.5, 4.2), ("Salvador", -13.0, -38.5, 3.9),
        ("Recife", -8.05, -34.9, 4.2), ("Curitiba", -25.4, -49.3, 3.7),
        ("Porto Alegre", -30.0, -51.2, 4.3), ("Manaus", -3.1, -60.0, 2.7),
        ("Belem", -1.45, -48.5, 2.5), ("Goiania", -16.7, -49.3, 2.7),
        ("Campinas", -22.9, -47.1, 3.3), ("Sao Luis", -2.5, -44.3, 1.6),
    ],
    "Argentina": [
        ("Buenos Aires", -34.6, -58.4, 15.4), ("Cordoba", -31.4, -64.2, 1.6),
        ("Rosario", -32.9, -60.7, 1.4), ("Mendoza", -32.9, -68.8, 1.2),
        ("Tucuman", -26.8, -65.2, 0.9), ("La Plata", -34.9, -57.9, 0.9),
        ("Mar del Plata", -38.0, -57.5, 0.65), ("Salta", -24.8, -65.4, 0.65),
    ],
}

AIFS_STEPS = list(range(0, 361, 6))
HRES_STEPS = list(range(0, 241, 6))


def _fetch(model: str, stream: str, typ: str, date: str, hh: str,
           steps: list[int]) -> "object | None":
    """One 2t retrieve -> xarray (step, lat, lon), or None on any failure."""
    import xarray as xr
    from ecmwf.opendata import Client
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    tmp.close()
    try:
        c = Client(source="ecmwf", model=model)
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            c.retrieve(date=date, time=int(hh), stream=stream, type=typ,
                       param="2t", step=steps, target=tmp.name)
        ds = xr.open_dataset(tmp.name, engine="cfgrib",
                             backend_kwargs={"indexpath": ""}).load()
        v = [x for x in ds.data_vars][0]
        return ds[v]
    except Exception as e:                            # noqa: BLE001
        print(f"  {model}/{typ} {date} {hh}Z unavailable ({repr(e)[:60]})")
        return None
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp.name)


def country_series(da, init: pd.Timestamp) -> dict:
    """Pop-weighted country temperature per valid day -> daily HDD/CDD."""
    lat = da.latitude.values
    lon = da.longitude.values % 360
    hours = (da.step / np.timedelta64(1, "h")).values.round().astype(int)
    vals = da.values                                   # (step, lat, lon) K
    out = {}
    for cty, metros in COUNTRIES.items():
        w = np.array([m[3] for m in metros])
        idx = [(int(np.argmin(np.abs(lat - m[1]))),
                int(np.argmin(np.abs(lon - (m[2] % 360))))) for m in metros]
        t = np.array([[vals[k, i, j] for (i, j) in idx]
                      for k in range(len(hours))])     # (step, metro)
        tc = (t * w).sum(axis=1) / w.sum() - 273.15    # country degC per step
        days = pd.DatetimeIndex([init + pd.Timedelta(hours=int(h)) for h in hours])
        ser = pd.Series(tc, index=days)
        cnt = ser.resample("1D").count()
        df = ser.resample("1D").mean().dropna()
        df = df[(df.index > init.normalize()) & (cnt >= 4)]   # only complete days
        # (a day sampled by a single 00Z step — e.g. HRES day 10 — is an
        # instantaneous reading, not a daily mean, and spikes the tail)
        out[cty] = {
            "days": [f"{d:%Y-%m-%d}" for d in df.index],
            "tmean": [round(float(x), 2) for x in df.values],
            "hdd": [round(max(0.0, BASE_C - float(x)), 2) for x in df.values],
            "cdd": [round(max(0.0, float(x) - BASE_C), 2) for x in df.values],
        }
    return out


def run_cycle(date: str, hh: str) -> bool:
    init = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{hh}:00")
    key = f"{date}_{hh}z"
    arch = json.loads(ARCHIVE.read_text()) if ARCHIVE.exists() else {}
    if key in arch and arch[key].get("aifs") and arch[key].get("hres"):
        print(f"{key}: already archived")
        return True
    print(f"fetching {key} …")
    aifs = _fetch("aifs-ens", "enfo", "em", date, hh, AIFS_STEPS)
    hres = _fetch("ifs", "oper", "fc", date, hh, HRES_STEPS)
    if aifs is None and hres is None:
        return False
    rec = arch.get(key, {})
    if aifs is not None:
        rec["aifs"] = country_series(aifs, init)
    if hres is not None:
        rec["hres"] = country_series(hres, init)
    arch[key] = rec
    keep = sorted(arch)[-MAX_CYCLES:]
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(json.dumps({k: arch[k] for k in keep}, separators=(",", ":")))
    print(f"  archived ({len(keep)} cycles held)")
    return True


def render() -> None:
    arch = json.loads(ARCHIVE.read_text())
    cycles = sorted(arch)
    latest = cycles[-1]
    fig, axes = plt.subplots(len(COUNTRIES), 2, figsize=(12.5, 3.1 * len(COUNTRIES)))
    for r, cty in enumerate(COUNTRIES):
        axb, axr = axes[r]
        # ── left: latest-cycle daily degree days, AIFS vs HRES ──
        rec = arch[latest]
        a = rec.get("aifs", {}).get(cty)
        h = rec.get("hres", {}).get(cty)
        if a:
            d = pd.to_datetime(a["days"])
            axb.bar(d, a["cdd"], color="#c62828", alpha=0.8, width=0.55,
                    label="CDD (AIFS-ENS mean)")
            axb.bar(d, [-x for x in a["hdd"]], color="#1f5fb4", alpha=0.8,
                    width=0.55, label="HDD (AIFS-ENS mean)")
        if h:
            dh = pd.to_datetime(h["days"])
            axb.plot(dh, h["cdd"], "o-", color="#7a1d1d", ms=3.5, lw=1.2,
                     label="CDD (HRES)")
            axb.plot(dh, [-x for x in h["hdd"]], "o-", color="#123c78",
                     ms=3.5, lw=1.2, label="HDD (HRES)")
        axb.axhline(0, color="0.4", lw=0.8)
        axb.set_title(f"{cty} — daily degree days, latest cycle "
                      f"({latest.replace('_', ' ').upper()})",
                      fontsize=9.5, fontweight="bold", loc="left")
        axb.tick_params(labelsize=7.5)
        if r == 0:
            axb.legend(fontsize=6.5, ncol=2, loc="upper right")
        # ── right: the SURPRISE — cumulative DD vs init cycle ──
        for src, color in [("aifs", "#c62828"), ("hres", "#1f5fb4")]:
            xs, cdds, hdds = [], [], []
            for c in cycles:
                s = arch[c].get(src, {}).get(cty)
                if not s:
                    continue
                n = min(10, len(s["cdd"]))            # common window: days 1-10
                xs.append(pd.Timestamp(f"{c[:4]}-{c[4:6]}-{c[6:8]}T{c[9:11]}:00"))
                cdds.append(sum(s["cdd"][:n]))
                hdds.append(sum(s["hdd"][:n]))
            lab = "AIFS-ENS mean" if src == "aifs" else "HRES"
            axr.plot(xs, cdds, "o-", color=color, ms=3.5, lw=1.4,
                     label=f"CDD · {lab}")
            axr.plot(xs, hdds, "s--", color=color, ms=3, lw=1.1, alpha=0.75,
                     label=f"HDD · {lab}")
        axr.set_title(f"{cty} — 10-day total vs init cycle (the revision)",
                      fontsize=9.5, fontweight="bold", loc="left")
        axr.tick_params(labelsize=7.5)
        axr.grid(lw=0.25, alpha=0.5)
        if r == 0:
            axr.legend(fontsize=6.5, ncol=2, loc="best")
    fig.suptitle("Degree-day surprise tracker — population-weighted, base 65°F / 18.3°C\n"
                 "left: daily HDD (down) / CDD (up) of the newest run · right: how the "
                 "10-day total has been REVISED run over run",
                 fontsize=11.5, fontweight="bold", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUTPNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPNG, dpi=115)
    plt.close(fig)
    print(f"wrote {OUTPNG.relative_to(REPO)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--time", default="00")
    ap.add_argument("--backfill", type=int, default=0,
                    help="also fetch the N previous 12-hourly cycles")
    args = ap.parse_args()
    if args.date:
        date, hh = args.date, args.time
    else:
        now = datetime.now(timezone.utc) - timedelta(hours=9)   # em lands ~8h after init
        hh = "12" if now.hour >= 12 else "00"
        date = now.strftime("%Y%m%d")
    t0 = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{hh}:00")
    todo = [t0 - pd.Timedelta(hours=12 * k) for k in range(args.backfill + 1)]
    for t in reversed(todo):
        run_cycle(f"{t:%Y%m%d}", f"{t:%H}")
    render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
