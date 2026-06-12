#!/usr/bin/env python3
"""Equatorial Pacific surface zonal-current Hovmöller (strip) — Copernicus Marine 1/12° model.

A longitude × time strip of the daily surface zonal current averaged 2°S-2°N, 150°E-90°W. Normally
the equatorial surface flows westward (the South Equatorial Current); as El Niño matures the trades
relax and downwelling Kelvin waves drive eastward surface-current surges that propagate into the
east Pacific — they show here as eastward (red) bands tilting down-and-eastward.

Keeps a small rolling store of the band-mean current (committed); each run pulls only the new days
(the first build backfills the window in monthly chunks). Runs in run_local_sst.sh (CMEMS creds).

    python scripts/sst/eq_current_hovmoller.py --out assets/sst/eq_current_hov.webp
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine as cm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cmems"
STORE = HERE / "metar" / "eq_scur_store.nc"      # committed rolling band-mean store
CUR = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
LON0, LON1, LATB = 130, 280, 2.0                 # 130°E-80°W band; display 150°E-90°W
KEEP_DAYS = 300
PLOT_DAYS = 270


def _pull(d0: date, d1: date) -> xr.DataArray | None:
    """Surface (≈0.5 m) zonal current over [d0, d1], 2°S-2°N mean → (time, lon)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "scur_chunk.nc"
    try:
        cm.subset(dataset_id=CUR, variables=["uo"], minimum_longitude=LON0, maximum_longitude=LON1,
                  minimum_latitude=-LATB, maximum_latitude=LATB, minimum_depth=0, maximum_depth=1,
                  start_datetime=str(d0), end_datetime=str(d1),
                  output_filename=f.name, output_directory=str(CACHE), overwrite=True)
        return xr.open_dataset(f)["uo"].isel(depth=0).mean("latitude").load()   # (time, lon)
    except Exception as e:                                       # noqa: BLE001
        print(f"  pull {d0}..{d1} failed: {repr(e)[:90]}", flush=True)
        return None


def update_store() -> xr.Dataset:
    """Append any new days to the rolling band-mean store (monthly-chunked backfill), trimmed."""
    ds = xr.open_dataset(STORE) if STORE.exists() else None
    have = set(pd.to_datetime(ds["time"].values).date) if ds is not None else set()
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=KEEP_DAYS)
    if have:
        start = max(start, max(have) - timedelta(days=2))       # re-pull last couple (latency fills)
    pieces = [ds["uo"]] if ds is not None else []
    d0 = start
    while d0 <= today:
        d1 = min(d0 + timedelta(days=30), today)
        ch = _pull(d0, d1)
        if ch is not None:
            pieces.append(ch)
            print(f"  got {d0}..{d1}: {ch.sizes['time']} d", flush=True)
        d0 = d1 + timedelta(days=1)
    if not pieces:
        return ds if ds is not None else xr.Dataset()
    uo = xr.concat(pieces, dim="time")
    uo = uo.sortby("time")
    uo = uo.isel(time=~pd.Index(uo["time"].values).duplicated(keep="last"))      # dedup
    uo = uo.sel(time=uo["time"] >= np.datetime64(start))                          # trim window
    out = uo.to_dataset(name="uo")
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".nc.tmp"); out.to_netcdf(tmp); tmp.replace(STORE)
    print(f"store: {out.sizes['time']} d × {out.sizes['longitude']} lon "
          f"({str(out['time'].values[0])[:10]}→{str(out['time'].values[-1])[:10]})", flush=True)
    return out


def render(ds: xr.Dataset, out: Path) -> None:
    from scipy.ndimage import gaussian_filter
    uo = ds["uo"]
    lon = uo["longitude"].values; t = uo["time"].values
    sel = t > t[-1] - np.timedelta64(PLOT_DAYS, "D")
    lm = (lon >= 150) & (lon <= 270)
    tt = t[sel]; xx = lon[lm]
    field = gaussian_filter(uo.values[np.ix_(sel, lm)], sigma=(1.0, 2.0))
    fig, ax = plt.subplots(figsize=(8.4, 9.6))
    lim = 0.8
    pm = ax.contourf(xx, tt, field, levels=np.linspace(-lim, lim, 17), cmap="RdBu_r", extend="both")
    ax.contour(xx, tt, field, levels=[0], colors="0.4", linewidths=0.5)
    ax.set_xlim(150, 270); ax.invert_yaxis()
    ax.set_xticks([150, 180, 210, 240, 270])
    ax.set_xticklabels(["150E", "180", "150W", "120W", "90W"])
    ax.axvline(190, color="0.5", lw=0.3, ls=":"); ax.axvline(240, color="0.5", lw=0.3, ls=":")
    ax.yaxis.set_major_formatter(DateFormatter("%d %b"))
    ax.set_xlabel("longitude"); ax.set_ylabel("date")
    ax.set_title("Equatorial Pacific surface zonal current (2°S–2°N)\n"
                 "red = eastward (El Niño-favorable) · blue = westward (trade-driven SEC)", fontsize=10)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.05, aspect=40)
    cb.set_label("surface zonal current (m s⁻¹)   ·   eastward +")
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent.parent / "assets" / "sst" / "eq_current_hov.webp"))
    args = ap.parse_args(argv)
    ds = update_store()
    if ds and ds.sizes.get("time"):
        render(ds, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
