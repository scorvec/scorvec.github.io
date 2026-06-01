#!/usr/bin/env python3
"""CanSIPS (GEM-NEMO) Niño-3.4 forecast plume — traditional + relative.

Downloads the GEM-NEMO component of CanSIPS (ensemble members 1–20) sea-surface
temperature forecasts from the ECCC datamart, computes the Niño-3.4 anomaly
plume against the model's own 1991–2020 hindcast climatology (drift-corrected),
and overlays the previous month's forecast to show the month-to-month trend.

  traditional : Niño-3.4 SST anomaly
  relative    : Niño-3.4 anomaly minus the 20°S–20°N tropical-mean anomaly

Datamart:
  forecast  …/100km/forecast/{YYYY}/{MM}/{YYYYMM}_MSC_CanSIPS_WaterTemp_Sfc_LatLon1.0_P{NN}M.grib2
  hindcast  …/100km/hindcast/{Y}/{MM}/{Y}{MM}_MSC_CanSIPS-Hindcast_WaterTemp_Sfc_LatLon1.0_P{NN}M.grib2
  variable  avg_sst (Kelvin), 40 members (1–20 GEM-NEMO, 21–40 CanESM5), 1°, P00M–P11M
"""
from __future__ import annotations

import argparse
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data" / "cansips"
CLIM_PATH = HERE / "cansips_nino34_clim.nc"          # committed cache (per start-month)

BASE = "https://dd.weather.gc.ca/today/model_cansips/100km"
LEADS = list(range(12))                               # P00M … P11M
GEM_NEMO = 20                                         # members 1–20
CLIM_YEARS = list(range(1991, 2021))                  # 1991–2020
N34 = dict(lat=(-5, 5), lon=(190, 240))               # Niño-3.4: 170°W–120°W
TROP = dict(lat=(-20, 20))                            # tropical mean


def _fc_url(ym, nn):
    return f"{BASE}/forecast/{ym[:4]}/{ym[4:]}/{ym}_MSC_CanSIPS_WaterTemp_Sfc_LatLon1.0_P{nn:02d}M.grib2"


def _hc_url(year, mm, nn):
    return (f"{BASE}/hindcast/{year}/{mm:02d}/{year}{mm:02d}"
            f"_MSC_CanSIPS-Hindcast_WaterTemp_Sfc_LatLon1.0_P{nn:02d}M.grib2")


def _fetch(url, dest: Path) -> Path:
    if not (dest.exists() and dest.stat().st_size > 0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def _fetch_many(jobs):                                # jobs: list[(url, dest)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(lambda j: _fetch(*j), jobs))


def _box_means(path: Path):
    """(nino34, tropmean) per member 1–20, in °C, for one GRIB file."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    v = ds["avg_sst"].sel(number=slice(1, GEM_NEMO)) - 273.15      # K → °C
    la = "latitude"; lo = "longitude"
    def _lat(rng):
        return slice(rng[0], rng[1]) if float(v[la][0]) < float(v[la][-1]) else slice(rng[1], rng[0])
    n = v.sel({la: _lat(N34["lat"]), lo: slice(*N34["lon"])})
    n34 = n.weighted(np.cos(np.deg2rad(n[la]))).mean((la, lo))
    t = v.sel({la: _lat(TROP["lat"])})
    trm = t.weighted(np.cos(np.deg2rad(t[la]))).mean((la, lo))
    return n34.values, trm.values                                  # (20,), (20,)


# ── model hindcast climatology (per start-month), cached ──────────────────────
def climatology(mm: int) -> tuple[np.ndarray, np.ndarray]:
    """clim_nino34[lead], clim_tropmean[lead] for start-month mm (member+year mean)."""
    if CLIM_PATH.exists():
        with xr.open_dataset(CLIM_PATH) as cf:
            if mm in [int(x) for x in cf.month.values]:
                row = cf.sel(month=mm).load()
                return row["clim_nino34"].values, row["clim_tropmean"].values
    print(f"  building CanSIPS hindcast climatology for start-month {mm:02d} "
          f"({len(CLIM_YEARS)} yrs × {len(LEADS)} leads) …", flush=True)
    jobs = [(_hc_url(y, mm, nn), DATA / "hindcast" / f"{y}{mm:02d}_P{nn:02d}.grib2")
            for y in CLIM_YEARS for nn in LEADS]
    _fetch_many(jobs)
    n34 = np.full((len(LEADS),), np.nan); trm = np.full((len(LEADS),), np.nan)
    for nn in LEADS:
        ns, ts = [], []
        for y in CLIM_YEARS:
            p = DATA / "hindcast" / f"{y}{mm:02d}_P{nn:02d}.grib2"
            a, b = _box_means(p); ns.append(a.mean()); ts.append(b.mean())
        n34[nn] = np.mean(ns); trm[nn] = np.mean(ts)
    _save_clim(mm, n34, trm)
    return n34, trm


def _save_clim(mm, n34, trm):
    row = xr.Dataset({"clim_nino34": ("lead", n34), "clim_tropmean": ("lead", trm)},
                     coords={"lead": LEADS, "month": mm}).expand_dims("month")
    if CLIM_PATH.exists():
        with xr.open_dataset(CLIM_PATH) as ds:
            old = ds.load()
        if mm in [int(x) for x in old.month.values]:
            old = old.drop_sel(month=mm)
        row = xr.concat([old, row], dim="month").sortby("month")
    tmp = CLIM_PATH.with_suffix(".tmp.nc")
    row.to_netcdf(tmp); os.replace(tmp, CLIM_PATH)    # write-then-replace (avoid open-file lock)
    print(f"  saved climatology for month {mm:02d} → {CLIM_PATH.name}")


def issue_anoms(ym: str):
    """(valid_dates, trad[member,lead], rel[member,lead]) for one issue YYYYMM."""
    mm = int(ym[4:])
    cn34, ctrm = climatology(mm)
    jobs = [(_fc_url(ym, nn), DATA / "forecast" / f"{ym}_P{nn:02d}.grib2") for nn in LEADS]
    _fetch_many(jobs)
    n34 = np.full((GEM_NEMO, len(LEADS)), np.nan)
    trm = np.full((GEM_NEMO, len(LEADS)), np.nan)
    for nn in LEADS:
        a, b = _box_means(DATA / "forecast" / f"{ym}_P{nn:02d}.grib2")
        n34[:, nn] = a; trm[:, nn] = b
    trad = n34 - cn34[None, :]
    rel = (n34 - trm) - (cn34 - ctrm)[None, :]
    start = pd.Timestamp(int(ym[:4]), mm, 1)
    valid = [start + pd.DateOffset(months=nn) for nn in LEADS]
    return pd.to_datetime(valid), trad, rel


def latest_issue() -> str:
    now = pd.Timestamp.utcnow()
    for back in range(0, 3):                          # this month, else fall back
        ym = (now - pd.DateOffset(months=back)).strftime("%Y%m")
        try:
            urllib.request.urlopen(_fc_url(ym, 1), timeout=20).close()
            return ym
        except Exception:
            continue
    raise SystemExit("no recent CanSIPS forecast issue found")


def plot(cur_ym, prev_ym, out: Path):
    cv, ctrad, crel = issue_anoms(cur_ym)
    pv, ptrad, prel = issue_anoms(prev_ym)
    cur_lab = pd.Timestamp(int(cur_ym[:4]), int(cur_ym[4:]), 1)
    prev_lab = pd.Timestamp(int(prev_ym[:4]), int(prev_ym[4:]), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for ax, cdat, pdat, title in ((axes[0], ctrad, ptrad, "Traditional (Niño-3.4 anomaly)"),
                                  (axes[1], crel, prel, "Relative (minus 20°S–20°N mean)")):
        ax.plot(pv, pdat.mean(0), color="#888", lw=1.8, ls="--",
                label=f"{prev_lab:%b %Y} issue (mean)", zorder=3)
        lo, hi = np.percentile(pdat, [10, 90], axis=0)
        ax.fill_between(pv, lo, hi, color="#888", alpha=0.12, zorder=1)
        ax.plot(cv, cdat.T, color="#d62728", lw=0.4, alpha=0.18, zorder=2)
        ax.plot(cv, cdat.mean(0), color="#d62728", lw=2.8,
                label=f"{cur_lab:%b %Y} issue (mean)", zorder=4)
        ax.axhline(0, color="0.5", lw=0.8)
        for g in (0.5, -0.5):
            ax.axhline(g, color="0.7", lw=0.7, ls=":")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2); ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("SST anomaly (°C)")
    fig.suptitle(f"CanSIPS GEM-NEMO Niño-3.4 forecast (members 1–20) — issued {cur_lab:%b %Y}, "
                 f"vs. previous issue", fontsize=12.5, fontweight="bold")
    fig.autofmt_xdate(); fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out} (current {cur_ym} mean@lead0 trad {ctrad.mean(0)[0]:+.2f}°C)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", help="YYYYMM (default: latest available)")
    ap.add_argument("--out", default=str(ASSETS / "cansips_nino34.webp"))
    args = ap.parse_args()
    cur = args.issue or latest_issue()
    prev = (pd.Timestamp(int(cur[:4]), int(cur[4:]), 1) - pd.DateOffset(months=1)).strftime("%Y%m")
    print(f"CanSIPS issue {cur} (prev {prev})")
    plot(cur, prev, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
