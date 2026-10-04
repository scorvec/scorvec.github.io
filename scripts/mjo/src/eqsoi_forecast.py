#!/usr/bin/env python3
"""Equatorial SOI (CPC definition) for the El Niño monitor: observed monthly series from CPC, a
daily tail from the ensembles' day-1 values archived cycle by cycle, and a 15-day AIFS-ENS +
IFS-ENS member forecast.

EQSOI = standardised anomaly of the east (5°N–5°S, 130–80°W) minus west (5°N–5°S, 90–140°E)
box-mean sea-level pressure; negative in El Niño. Standardisation: ERA5 1991–2020 per-month mean
and monthly-mean SD from data/reference/eqsoi_clim.json (build_eqsoi_clim.py), then the linear
calibration to CPC's published series (r 0.944 over 1991–2020), so observed and forecast share
CPC's scale. The user asked for it (2026-09-07) as "a more physically consistent zonal pressure
difference index" than the two-station SOI.

    python src/eqsoi_forecast.py --date 20260907 --time 00 --out ../../assets/sst/eqsoi_forecast.webp
"""
from __future__ import annotations
import argparse, json, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf
from soi_forecast import MODELS, download_msl

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
CLIM = json.loads((REF / "eqsoi_clim.json").read_text())
HIST = REF / "eqsoi_history.json"
TIDE = json.loads((REF / "eqsoi_tide.json").read_text())["offset"] if (REF / "eqsoi_tide.json").exists() else None
CPC = "https://www.cpc.ncep.noaa.gov/data/indices/reqsoi.for"
INK, MUTED, NAVY, RED = "#1a1a1a", "#8a8680", "#1b365d", "#b4453c"


def cpc_series() -> pd.Series:
    txt = urllib.request.urlopen(CPC, timeout=60).read().decode()
    rows = {}
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 13 and p[0].isdigit():
            for i, v in enumerate(p[1:]):
                if float(v) < 900: rows[pd.Timestamp(int(p[0]), i + 1, 15)] = float(v)
    return pd.Series(rows).sort_index()


def box_mean(msl: xr.DataArray, b: dict) -> xr.DataArray:
    lat = msl.latitude
    d = msl.sel(latitude=slice(b["lat"][1], b["lat"][0]) if float(lat[0]) > float(lat[-1]) else slice(b["lat"][0], b["lat"][1]))
    d = d.sel(longitude=slice(b["lon"][0], b["lon"][1]))
    return d.weighted(np.cos(np.deg2rad(d.latitude))).mean(("latitude", "longitude"))


def member_index(paths: dict) -> xr.DataArray:
    """(member, step) EQSOI on CPC's scale."""
    parts = []
    for p in paths.values():
        ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": "msl"}, "indexpath": ""}, chunks={"number": 1})
        msl = ds[[v for v in ds.data_vars][0]]
        if float(msl.longitude.min()) < 0:
            msl = msl.assign_coords(longitude=msl.longitude % 360).sortby("longitude")
        d = (box_mean(msl, CLIM["boxes"]["east"]) - box_mean(msl, CLIM["boxes"]["west"])) / 100.0
        if "number" not in d.dims: d = d.expand_dims("number")
        parts.append(d.compute())
    D = xr.concat(parts, dim="number"); D = D.assign_coords(number=np.arange(D.sizes["number"]))
    return D


def to_index(D_hpa: np.ndarray, months: np.ndarray, hour: int) -> np.ndarray:
    """CPC-scale EQSOI from an instantaneous box difference at a fixed UTC hour: the semidiurnal
    tide's offset at that hour (build_eqsoi_tide.py, ~−2 hPa at 00 UTC) is removed first so the
    value compares with the daily means the climatology and CPC's index are built from."""
    c = CLIM["clim"]; f = CLIM["cpc_fit"]
    mean = np.array([c[str(m)]["mean"] for m in months]); sd = np.array([c[str(m)]["sd_monthly"] for m in months])
    tide = np.array([TIDE[str(hour)][str(m)] for m in months]) if TIDE else 0.0
    return f["a"] + f["b"] * (D_hpa - tide - mean) / sd


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); ap.add_argument("--time", default="00"); ap.add_argument("--out", default="plots/eqsoi_forecast.webp")
    a = ap.parse_args()
    init = pd.Timestamp(f"{a.date}T{a.time}:00")
    cpc = cpc_series()
    parts, included = [], []
    for cfg in MODELS:
        try:
            parts.append(member_index(download_msl(cfg, a.date, a.time))); included.append(cfg["label"])
        except Exception as e:                                       # noqa: BLE001
            print(f"  {cfg['label']}: skipped ({repr(e)[:80]})", flush=True)
    if not parts: raise SystemExit("no ensemble MSL for the EQSOI forecast")
    D = xr.concat(parts, dim="number"); D = D.assign_coords(number=np.arange(D.sizes["number"]))
    steps_h = (D.step / np.timedelta64(1, "h")).values.astype(int)
    valid = pd.to_datetime([init + pd.Timedelta(hours=int(h)) for h in steps_h])
    idx = np.vstack([to_index(D.isel(number=j).values, valid.month.values, int(a.time)) for j in range(D.sizes["number"])])   # (member, day)
    mean = np.nanmean(idx, 0); p10, p90 = np.nanpercentile(idx, 10, 0), np.nanpercentile(idx, 90, 0)
    # daily tail: the day-1 ensemble mean of each cycle, archived
    hist = json.loads(HIST.read_text()) if HIST.exists() else {}
    hist[valid[0].strftime("%Y-%m-%d")] = round(float(mean[0]), 3)
    hist = dict(sorted(hist.items())[-400:]); HIST.write_text(json.dumps(hist, indent=0))
    tail = pd.Series({pd.Timestamp(k): v for k, v in hist.items()}).sort_index()

    # two panels on one y-axis: 30 months of CPC's monthly index, then the 15-day daily forecast
    # at readable width (a single time axis squeezed the forecast into a sliver).
    fig, (axm, axf) = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True, gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.04})
    t0 = init - pd.DateOffset(months=30)
    c = cpc[cpc.index >= t0]
    axm.bar(c.index, c.values, width=26, color=np.where(c.values < 0, "#d98c7a", "#7fa7c9"), alpha=0.9, zorder=2)
    axm.axhline(0, color="#555", lw=0.8); axm.grid(True, alpha=0.2); axm.set_xlim(t0, c.index[-1] + pd.Timedelta(days=20))
    axm.set_title("CPC monthly Equatorial SOI", fontsize=9.5, loc="left", fontweight="bold")
    axm.set_ylabel("Equatorial SOI (standardised, CPC scale)")
    if len(tail) > 1: axf.plot(tail.index, tail.values, color=INK, lw=1.4, label="day-1 ensemble mean, archived per cycle", zorder=3)
    axf.fill_between(valid, p10, p90, color=RED, alpha=0.18, lw=0, label="P10–P90", zorder=3)
    for j in range(idx.shape[0]): axf.plot(valid, idx[j], color=RED, lw=0.4, alpha=0.12, zorder=3)
    axf.plot(valid, mean, color=RED, lw=2.4, label=f"mean, {idx.shape[0]} members", zorder=4)
    axf.axhline(float(c.values[-1]), color="#7a5c4a", lw=1.0, ls="--", label=f"CPC {c.index[-1]:%b %Y}: {c.values[-1]:+.1f}", zorder=2)
    axf.axhline(0, color="#555", lw=0.8); axf.grid(True, alpha=0.2)
    axf.set_xlim(valid[0] - pd.Timedelta(days=(len(tail) if len(tail) > 1 else 1)), valid[-1] + pd.Timedelta(days=1))
    axf.set_title(f"{' + '.join(included)} daily forecast, init {init:%Y-%m-%d %HZ}", fontsize=9.5, loc="left", fontweight="bold")
    axf.tick_params(axis="x", labelsize=8); axf.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    for lab in axf.get_xticklabels(): lab.set_rotation(30); lab.set_ha("right")
    fig.suptitle("Equatorial SOI — east minus west equatorial Pacific pressure, observed and forecast", fontsize=12, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.01, -0.02, "East (5°N–5°S, 130–80°W) minus west (5°N–5°S, 90–140°E) box-mean SLP, tide-corrected for the cycle hour, standardised with ERA5 1991–2020 monthly statistics and calibrated to CPC's series "
             f"(r {CLIM['cpc_fit']['r']:.2f}); negative = El Niño. Daily values scatter more than the monthly index they are drawn against.", fontsize=7.2, color=MUTED, va="top")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
    (out.parent / "data").mkdir(exist_ok=True)
    (out.parent / "data" / "eqsoi_forecast.json").write_text(json.dumps({"init": init.strftime("%Y-%m-%dT%HZ"), "models": included, "n_members": int(idx.shape[0]),
        "valid": [d.strftime("%Y-%m-%d") for d in valid], "mean": np.round(mean, 3).tolist(), "p10": np.round(p10, 3).tolist(), "p90": np.round(p90, 3).tolist(),
        "cpc_latest": {"month": c.index[-1].strftime("%Y-%m"), "value": float(c.values[-1])}, "day1_history": hist}, separators=(",", ":")))
    print(f"  EQSOI: CPC {c.index[-1]:%Y-%m} {c.values[-1]:+.1f}; forecast day 1 {mean[0]:+.2f}, day 15 {mean[-1]:+.2f} ({idx.shape[0]} members) → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
