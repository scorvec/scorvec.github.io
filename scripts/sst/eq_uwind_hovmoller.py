#!/usr/bin/env python3
"""Equatorial Pacific surface zonal-wind Hovmöller (strip) — Copernicus Marine ASCAT L4.

A longitude × time strip of the daily-mean surface zonal wind averaged 5°S-5°N, 130°E-80°W, from
the same gap-filled scatterometer (ASCAT) L4 product the equatorial wind map uses. Normally the
equatorial surface wind is easterly (the trades); as El Niño develops, westerly-wind bursts (WWBs)
punch eastward over the warm pool and drive downwelling Kelvin waves — they show here as eastward
(red, westerly) bands, often propagating east, that precede the surface-current surges in the
companion ocean-current strip.

Keeps a rolling band-mean store (local, gitignored — the webp is the committed artifact); each run
pulls only the new days (the first build backfills the window in monthly chunks). The hourly product
is resampled to daily means. Runs locally in run_local_sst.sh (CMEMS creds).

    python scripts/sst/eq_uwind_hovmoller.py --out assets/sst/eq_uwind_hov.webp
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
STORE = CACHE / "eq_uwind_store.nc"              # local rolling band-mean cache (gitignored)
WIND = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"
LON0, LON1, LATB = 130, 280, 5.0                 # 130°E-80°W band, 5°S-5°N; display 150°E-90°W
KEEP_DAYS = 300
PLOT_DAYS = 270


def _pull(d0: date, d1: date) -> xr.DataArray | None:
    """Daily-mean eastward (zonal) wind over [d0, d1], 5°S-5°N mean → (time, lon)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "uwind_chunk.nc"
    try:
        cm.subset(dataset_id=WIND, variables=["eastward_wind"],
                  minimum_longitude=LON0, maximum_longitude=LON1,
                  minimum_latitude=-LATB, maximum_latitude=LATB,
                  start_datetime=str(d0), end_datetime=f"{d1}T23:59:59",
                  output_filename=f.name, output_directory=str(CACHE), overwrite=True)
        da = xr.open_dataset(f)["eastward_wind"]
        da = da.assign_coords(longitude=da["longitude"] % 360).sortby("longitude")
        da = da.sel(longitude=slice(LON0, LON1))
        u = da.mean("latitude").resample(time="1D").mean()        # daily mean of the hourly field
        return u.load()                                            # (time, lon)
    except Exception as e:                                         # noqa: BLE001
        print(f"  pull {d0}..{d1} failed: {repr(e)[:90]}", flush=True)
        return None


def update_store() -> xr.Dataset:
    """Append any new days to the rolling band-mean store (monthly-chunked backfill), trimmed.

    `pull_from` limits the download to the recent tail; the trim uses the full window so the store
    is never collapsed to the last couple of days (the bug the sibling current Hovmöller once had).
    """
    ds = xr.open_dataset(STORE) if STORE.exists() else None
    have = set(pd.to_datetime(ds["time"].values).date) if ds is not None else set()
    today = datetime.now(timezone.utc).date()
    window_start = today - timedelta(days=KEEP_DAYS)            # trim threshold — keep the full window
    pull_from = window_start
    if have:
        pull_from = max(window_start, max(have) - timedelta(days=2))   # only re-download the recent tail
    pieces = [ds["u"]] if ds is not None else []
    d0 = pull_from
    while d0 <= today:
        d1 = min(d0 + timedelta(days=30), today)
        ch = _pull(d0, d1)
        if ch is not None:
            pieces.append(ch)
            print(f"  got {d0}..{d1}: {ch.sizes['time']} d", flush=True)
        d0 = d1 + timedelta(days=1)
    if not pieces:
        return ds if ds is not None else xr.Dataset()
    u = xr.concat(pieces, dim="time")
    u = u.sortby("time")
    u = u.isel(time=~pd.Index(u["time"].values).duplicated(keep="last"))          # dedup
    u = u.sel(time=u["time"] >= np.datetime64(window_start))                       # trim to the full window
    out = u.to_dataset(name="u")
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".nc.tmp"); out.to_netcdf(tmp); tmp.replace(STORE)
    print(f"store: {out.sizes['time']} d × {out.sizes['longitude']} lon "
          f"({str(out['time'].values[0])[:10]}→{str(out['time'].values[-1])[:10]})", flush=True)
    return out


def render(ds: xr.Dataset, out: Path) -> None:
    from scipy.ndimage import gaussian_filter
    u = ds["u"]
    lon = u["longitude"].values; t = u["time"].values
    sel = t > t[-1] - np.timedelta64(PLOT_DAYS, "D")
    lm = (lon >= 150) & (lon <= 270)
    tt = t[sel]; xx = lon[lm]
    field = gaussian_filter(u.values[np.ix_(sel, lm)], sigma=(1.0, 2.0))
    fig, ax = plt.subplots(figsize=(8.4, 9.6))
    lim = 8.0
    pm = ax.contourf(xx, tt, field, levels=np.linspace(-lim, lim, 17), cmap="RdBu_r", extend="both")
    ax.contour(xx, tt, field, levels=[0], colors="0.4", linewidths=0.5)
    ax.set_xlim(150, 270); ax.invert_yaxis()
    ax.set_xticks([150, 180, 210, 240, 270])
    ax.set_xticklabels(["150E", "180", "150W", "120W", "90W"])
    ax.axvline(190, color="0.5", lw=0.3, ls=":"); ax.axvline(240, color="0.5", lw=0.3, ls=":")
    ax.yaxis.set_major_formatter(DateFormatter("%d %b"))
    ax.set_xlabel("longitude"); ax.set_ylabel("date")
    ax.set_title("Equatorial Pacific surface zonal wind (5°S–5°N)\n"
                 "red = westerly (El Niño-favorable / WWB) · blue = easterly (trade winds)", fontsize=10)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.05, aspect=40)
    cb.set_label("surface zonal wind (m s⁻¹)   ·   westerly +")
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent.parent / "assets" / "sst" / "eq_uwind_hov.webp"))
    args = ap.parse_args(argv)
    ds = update_store()
    if ds and ds.sizes.get("time"):
        render(ds, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
