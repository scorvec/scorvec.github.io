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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf                                    # shared ECMWF download manager
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


def download(model_key: str, date: str, time: str, out_dir: Path = None) -> dict:
    """Ensure 10u (forecast days) for this model via the shared store; AIFS cf+pf is
    deduped with the torque budget. Returns {typ: cache_path}."""
    cfg = MODELS[model_key]
    cyc = ecmwf.Cycle(date, time)
    return {typ: ecmwf.sfc_path(cyc, cfg["model"], typ, "10u") for typ in cfg["types"]}


def ensemble_mean_band(paths: dict) -> xr.DataArray:
    """Ensemble-mean, 5°S–5°N weighted-average 10u -> (step, LON_GRID)."""
    tot, cnt = None, 0
    for p in paths.values():
        # chunk per member AT OPEN so cfgrib reads one member at a time
        ds = xr.open_dataset(p, engine="cfgrib",
                             backend_kwargs={"filter_by_keys": {"shortName": "10u"}, "indexpath": ""},
                             chunks={"number": 1})       # 10u out of the batched surface file
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
    # Forecast days 1..15 only: drop the 0-h analysis step. The shared store bundles
    # AIFS 10u with the torque budget (STEPS = [0] + days 1..15), so AIFS now carries a
    # step-0 row while IFS does not; without this the two models have different step
    # counts and plot()'s shared time axis fails (y mismatch z). Days-1..15 also matches
    # this plot's intent (DAILY_STEPS).
    ens = ens.isel(step=ens.step.values > np.timedelta64(0))
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


ANOM_LIM = 12.0     # ±6 saturated on real WWBs (the Jul 2026 burst ran past
ABS_LIM = 10.0      # 10 m/s) and 0.5 m/s steps were noise — see levels below


def _ref_map(ax, title):
    """Small tropical-belt basemap (no data) atop one column — same longitude
    span as that column's Hovmöller, stretched to fill the column width so the
    geography lines up with the data below."""
    ax.set_extent([LON_VIEW[0], LON_VIEW[1], -15, 15], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")                          # fill column width (align longitude)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#d9d6cf", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555",
                   linewidth=0.3, zorder=3)
    ax.add_patch(plt.Rectangle((LON_VIEW[0], -5), LON_VIEW[1] - LON_VIEW[0], 10,
                 transform=ccrs.PlateCarree(), facecolor="none",
                 edgecolor="k", lw=0.6, ls="--", zorder=4))
    ax.set_title(title, fontsize=9)


def plot(data: dict, valid: np.ndarray, init: pd.Timestamp, out: Path):
    """data: {model: {'anom': (step,lon), 'abs': (step,lon)}}.
    Columns grouped by model: <model> anomaly, <model> absolute u10. A tropical-
    belt reference map sits above the columns."""
    lead = np.array([(pd.Timestamp(v) - init) / pd.Timedelta(days=1) for v in valid])
    m = (LON_GRID >= LON_VIEW[0]) & (LON_GRID <= LON_VIEW[1])
    lons = LON_GRID[m]
    cols = []
    for k in data:                                  # aifs, ifs (insertion order)
        cols += [(k, "anom"), (k, "abs")]
    ncol = len(cols)

    fig = plt.figure(figsize=(3.35 * ncol, 8.8))
    gs = fig.add_gridspec(2, ncol, height_ratios=[0.85, 6.5], hspace=0.05,
                          wspace=0.10, left=0.06, right=0.9, top=0.9, bottom=0.07)
    fig.suptitle(f"Equatorial Pacific 10 m zonal wind forecast — init {init:%Y-%m-%d %HZ}\n"
                 f"ensemble mean · 5°S–5°N · anomaly vs ERA5 1991–2020",
                 fontsize=12, fontweight="bold")

    imA = imB = None
    for j, (k, kind) in enumerate(cols):
        kind_lab = "anomaly" if kind == "anom" else "absolute u10"
        _ref_map(fig.add_subplot(gs[0, j],
                 projection=ccrs.PlateCarree(central_longitude=180)),
                 f"{MODELS[k]['label']}\n{kind_lab}")
        ax = fig.add_subplot(gs[1, j])
        fld = data[k][kind][:, m]
        if kind == "anom":
            imA = ax.contourf(lons, lead, fld, levels=np.arange(-ANOM_LIM, ANOM_LIM + .01, 1.5),
                              cmap="RdBu_r", extend="both",
                              norm=mcolors.TwoSlopeNorm(0, -ANOM_LIM, ANOM_LIM))
        else:
            imB = ax.contourf(lons, lead, fld, levels=np.arange(-ABS_LIM, ABS_LIM + .01, 1),
                              cmap="RdBu_r", extend="both",
                              norm=mcolors.TwoSlopeNorm(0, -ABS_LIM, ABS_LIM))
        ax.contour(lons, lead, fld, levels=[0], colors="k", linewidths=0.5, alpha=0.5)
        ax.set_ylim(lead.max(), lead.min())
        ax.set_xticks(*_lon_ticks())
        ax.tick_params(labelsize=7.5)
        ax.axvline(180, color="0.5", lw=0.5, ls=":")
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Forecast lead (days)" if j == 0 else "")
        if j > 0:
            ax.set_yticklabels([])
    if imA is not None:
        c = fig.colorbar(imA, cax=fig.add_axes([0.915, 0.52, 0.013, 0.34]), extend="both")
        c.set_label("u anomaly (m s⁻¹)", fontsize=8); c.ax.tick_params(labelsize=7)
    if imB is not None:
        c = fig.colorbar(imB, cax=fig.add_axes([0.915, 0.09, 0.013, 0.34]), extend="both")
        c.set_label("u10 (m s⁻¹)", fontsize=8); c.ax.tick_params(labelsize=7)
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
    data, valid = {}, None
    for k in args.models.split(","):
        print(f"== {k} ==", flush=True)
        try:
            paths = download(k, args.date, args.time, Path(args.data_dir))
            ens = ensemble_mean_band(paths)
            a, valid = anomalize(ens, init)
            data[k] = {"anom": a, "abs": ens.values}
        except Exception as e:                      # e.g. that cycle not yet on a model
            print(f"  {k}: skipped ({repr(e)[:90]})", flush=True)
    if not data:
        raise SystemExit("no model data available for the Hovmöller")
    plot(data, valid, init, Path(args.out))
    # Record which requested models were skipped (e.g. IFS-ENS not yet on the portal) so the
    # pipeline knows this render is incomplete and can re-render once the model lands. One
    # token per line ("aifs"/"ifs") → the run-script greps for an exact "ifs" line.
    missing = [m for m in args.models.split(",") if m not in data]
    flag = Path(str(args.out) + ".missing")
    if missing:
        flag.write_text("\n".join(missing) + "\n")
    else:
        flag.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
