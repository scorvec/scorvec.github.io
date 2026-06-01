#!/usr/bin/env python3
"""
Equatorial (5°S–5°N) 10 m zonal-wind anomaly Hovmöller forecasts from the
AIFS-ENS (AI) and ECMWF IFS-ENS (physics) ensembles, for the El Niño monitor.

For each model: download daily-step 10u for all members (AIFS = cf+pf, IFS = the
50 pf members), ensemble-mean, 5°S–5°N cosine-weighted average, anomalize vs the
ERA5 climatology (build_eq_wind_clim.py), and contour longitude × forecast day.

    python src/eq_hovmoller.py --date 20260531 --time 12 --out plots/eq_hov.webp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

sys.path.insert(0, str(Path(__file__).parent))
from download_aifs import _retrieve            # aws/azure/ecmwf mirror fallback
from build_eq_wind_clim import eval_clim

DAILY_STEPS = list(range(24, 361, 24))          # forecast days 1..15
LON_GRID = np.arange(0.0, 360.0, 1.0)
LON_VIEW = (40.0, 290.0)                         # Indian Ocean → eastern Pacific
LAT_BAND = 5.0
CLIM_PATH = Path(__file__).resolve().parent.parent / "data" / "reference" / "eq_u10_clim.nc"

MODELS = {
    "aifs": dict(model="aifs-ens", types=["cf", "pf"], label="AIFS-ENS (AI)"),
    "ifs":  dict(model="ifs",      types=["pf"],        label="IFS-ENS (physics)"),
}


def download(model_key: str, date: str, time: str, out_dir: Path) -> dict:
    cfg = MODELS[model_key]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for typ in cfg["types"]:
        p = out_dir / f"u10_{model_key}_{date}_{time}z_{typ}.grib2"
        if not p.exists():
            print(f"  {model_key}/{typ}: downloading 10u (daily steps) …", flush=True)
            _retrieve(dict(model=cfg["model"], date=date, time=int(time),
                           stream="enfo", type=typ, levtype="sfc",
                           param="10u", step=DAILY_STEPS), str(p))
        paths[typ] = p
    return paths


def ensemble_mean_band(paths: dict) -> xr.DataArray:
    """Ensemble-mean, 5°S–5°N weighted-average 10u -> (step, LON_GRID)."""
    tot, cnt = None, 0
    for p in paths.values():
        # chunk per member AT OPEN so cfgrib reads one member at a time
        ds = xr.open_dataset(p, engine="cfgrib",
                             backend_kwargs={"indexpath": ""},
                             chunks={"number": 1})
        u = ds[[v for v in ds.data_vars][0]]            # u10
        u = u.sortby("latitude").sel(latitude=slice(-8, 8))
        if float(u.longitude.min()) < 0:
            u = u.assign_coords(longitude=u.longitude % 360).sortby("longitude")
        lat = u.latitude
        w = np.cos(np.deg2rad(lat)).where(np.abs(lat) <= LAT_BAND, 0.0)
        band = u.weighted(w).mean("latitude")           # (number?, step, lon)
        if "number" in band.dims:
            tot = band.sum("number") if tot is None else tot + band.sum("number")
            cnt += band.sizes["number"]
        else:
            tot = band if tot is None else tot + band
            cnt += 1
    ens = (tot / cnt).interp(longitude=LON_GRID)
    return ens.compute()


def anomalize(ens: xr.DataArray, init: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    coeffs = xr.open_dataarray(CLIM_PATH).values            # (5, nlon)
    steps_h = (ens.step / np.timedelta64(1, "h")).values.astype(int)
    valid = [init + pd.Timedelta(hours=int(h)) for h in steps_h]
    doy = np.array([v.dayofyear for v in valid])
    clim = eval_clim(coeffs, doy)                          # (nstep, nlon)
    anom = ens.values - clim                               # (nstep, nlon)
    return anom, np.array(valid)


def _lon_ticks():
    ticks = [60, 120, 180, 240, 300]
    labs = [f"{t}°E" if t <= 180 else f"{360 - t}°W" for t in ticks]
    return ticks, labs


def plot(anoms: dict, valid: np.ndarray, init: pd.Timestamp, out: Path,
         lim: float = 6.0):
    lead = np.array([(pd.Timestamp(v) - init) / pd.Timedelta(days=1) for v in valid])
    m = (LON_GRID >= LON_VIEW[0]) & (LON_GRID <= LON_VIEW[1])
    lons = LON_GRID[m]
    levels = np.arange(-lim, lim + 0.001, 0.5)
    keys = list(anoms.keys())

    fig, axes = plt.subplots(1, len(keys), figsize=(5.0 * len(keys), 7.2),
                             sharey=True)
    if len(keys) == 1:
        axes = [axes]
    fig.suptitle(f"Equatorial Pacific (5°S–5°N) 10 m zonal-wind anomaly forecast\n"
                 f"Ensemble mean · anomaly vs ERA5 1991–2020 · init "
                 f"{init:%Y-%m-%d %HZ}", fontsize=11, fontweight="bold")
    cf = None
    for ax, k in zip(axes, keys):
        cf = ax.contourf(lons, lead, anoms[k][:, m], levels=levels,
                         cmap="RdBu_r", extend="both",
                         norm=mcolors.TwoSlopeNorm(0, -lim, lim))
        ax.contour(lons, lead, anoms[k][:, m], levels=[0], colors="k", linewidths=0.5, alpha=0.5)
        ax.set_title(MODELS[k]["label"], fontsize=10)
        ax.set_xticks(*_lon_ticks())
        ax.tick_params(labelsize=8)
        ax.axvline(180, color="0.5", lw=0.5, ls=":")
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Forecast lead (days)")
    axes[0].set_ylim(lead.max(), lead.min())               # day 0/1 at top
    fig.colorbar(cf, ax=axes, label="10 m U anomaly (m s⁻¹)",
                 fraction=0.025, pad=0.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/u10")
    ap.add_argument("--out", default="plots/eq_hovmoller.webp")
    ap.add_argument("--models", default="aifs,ifs")
    args = ap.parse_args()

    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    anoms, valid = {}, None
    for k in args.models.split(","):
        print(f"== {k} ==", flush=True)
        try:
            paths = download(k, args.date, args.time, Path(args.data_dir))
            ens = ensemble_mean_band(paths)
            a, valid = anomalize(ens, init)
            anoms[k] = a
        except Exception as e:                      # e.g. that cycle not yet on a model
            print(f"  {k}: skipped ({repr(e)[:90]})", flush=True)
    if not anoms:
        raise SystemExit("no model data available for the Hovmöller")
    plot(anoms, valid, init, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
