#!/usr/bin/env python3
"""Historical equatorial-OLR record for the wave decomposition (Wheeler-Kiladis).

NOAA's interpolated-OLR product (Liebmann & Smith 1996; daily, 2.5°, global) is a 49-year
homogeneous true-OLR record — but it ends 2022-12-31. Our live GMGSI proxy only reaches
back to 2021. So this builds the long historical backbone the wave filter needs:

  * pulls interp_OLR 1979→2022 over OPeNDAP (server-side lat-subset, yearly chunks),
  * meridional-means the equatorial band over the FULL global longitude circle — Wheeler-
    Kiladis is a zonal-wavenumber FFT, so it needs every longitude, not the 40-300°E display
    window — at two bands (15°S-15°N for filtering, 5°S-5°N to match the display Hovmöller),
  * fits a smooth daily climatology (annual mean + 3 harmonics) for the anomaly baseline.

Output (gitignored cache, ~10 MB): scripts/sst/data/olr_history.nc with olr_15/olr_05
(time × lon), their clim_15/clim_05 (dayofyear × lon), and lon (2.5° global). The GMGSI
real-time tail gets bias-corrected onto this scale over the 2021-22 overlap when the
decomposition itself is built.

    python scripts/sst/build_olr_history.py            # 1979-2022, both bands
    python scripts/sst/build_olr_history.py --y0 2000  # shorter window
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "olr_history.nc"
OPENDAP = "https://psl.noaa.gov/thredds/dodsC/Datasets/interp_OLR/olr.day.mean.nc"
BANDS = {"15": 15.0, "05": 5.0}            # meridional half-widths (°lat)


def _band_mean(ds: xr.Dataset, half: float) -> xr.DataArray:
    """Meridional mean of OLR over ±half° latitude, keeping all longitudes."""
    lat = ds["lat"]
    sub = ds["olr"].sel(lat=lat[(lat >= -half) & (lat <= half)])
    return sub.mean("lat")                  # OLR(time, lon)


def fetch(y0: int, y1: int) -> xr.Dataset:
    """Pull interp_OLR year-by-year over OPeNDAP (each chunk retried), concat in time."""
    rds = xr.open_dataset(OPENDAP, decode_times=True)
    pieces = {b: [] for b in BANDS}
    for yr in range(y0, y1 + 1):
        for attempt in range(1, 5):
            try:
                yds = rds.sel(time=slice(f"{yr}-01-01", f"{yr}-12-31")).load()
                for b, half in BANDS.items():
                    pieces[b].append(_band_mean(yds, half))
                print(f"  {yr}: {yds.sizes['time']} days", flush=True)
                break
            except Exception as e:                          # noqa: BLE001 (OPeNDAP flakiness)
                print(f"  {yr} attempt {attempt} failed ({repr(e)[:70]}); retry in 8s", flush=True)
                time.sleep(8)
        else:
            raise RuntimeError(f"OPeNDAP fetch failed for {yr}")
    out = xr.Dataset()
    for b in BANDS:
        da = xr.concat(pieces[b], dim="time").astype("float32")
        out[f"olr_{b}"] = da
    out = out.assign_coords(lon=rds["lon"].values)
    return out


def daily_clim(da: xr.DataArray, n_harm: int = 3) -> xr.DataArray:
    """Smooth annual cycle: dayofyear mean → keep annual mean + first n harmonics."""
    doy = da["time"].dt.dayofyear
    raw = da.groupby(doy).mean("time")                      # (dayofyear, lon)
    raw = raw.reindex(dayofyear=np.arange(1, 367)).interpolate_na("dayofyear", fill_value="extrapolate")
    x = raw.values                                          # (366, lon)
    n = x.shape[0]
    f = np.fft.rfft(x, axis=0)
    f[n_harm + 1:] = 0                                      # zero everything above harmonic n
    sm = np.fft.irfft(f, n=n, axis=0).astype("float32")
    return xr.DataArray(sm, coords={"dayofyear": np.arange(1, 367), "lon": raw["lon"]},
                        dims=("dayofyear", "lon"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1979, help="start year (1979 skips the 1978 gap)")
    ap.add_argument("--y1", type=int, default=2022, help="end year (interp_OLR ends 2022)")
    args = ap.parse_args(argv)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"interp_OLR {args.y0}-{args.y1}, bands {list(BANDS)} (global longitude circle)…", flush=True)
    ds = fetch(args.y0, args.y1)
    for b in BANDS:
        ds[f"clim_{b}"] = daily_clim(ds[f"olr_{b}"])
    ds.attrs.update(source="NOAA Interpolated OLR (Liebmann & Smith 1996), PSL OPeNDAP",
                    note="equatorial meridional means, full global longitudes; clim = annual mean + 3 harmonics",
                    period=f"{args.y0}-{args.y1}")
    ds.to_netcdf(OUT)
    mb = OUT.stat().st_size / 1e6
    t = ds["time"].values
    print(f"wrote {OUT.name} ({mb:.1f} MB) — {str(t[0])[:10]}→{str(t[-1])[:10]}, "
          f"{ds.sizes['time']} days × {ds.sizes['lon']} lon", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
