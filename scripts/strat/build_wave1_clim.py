#!/usr/bin/env python3
"""Climatological stationary wave-1 in geopotential height, by day of year.

Feeds the superposition index in wave1_maps.py: whether the FORECAST wave-1 is
reinforcing the climatological standing wave or cancelling it.

WHY A COMPLEX COEFFICIENT, NOT AN AMPLITUDE. The whole question is phase. A
forecast wave-1 of 300 gpm sitting on top of the climatological ridge and one
sitting over the climatological trough are the same amplitude and opposite
events - the first drives wave activity up into the vortex, the second cancels
the standing wave and shuts the flux down. So the climatology stored here is
the mean COMPLEX k=1 coefficient per day of year: averaging complex
coefficients keeps the phase, averaging amplitudes would throw away exactly the
information the index needs.

  band mean of z over 55-65 deg (cos-weighted, per hemisphere)
    -> rfft along longitude -> coefficient at k=1  -> complex, per day
    -> mean over years, per day of year
    -> smoothed by keeping the first 3 annual harmonics

Source: WeatherBench2's ERA5 (1.5 deg, 240x121). Coarse resolution is not a
compromise here - k=1 is a planetary wave and 240 longitudes resolves it with
enormous margin - and it is the cheapest ERA5 in the repo. 00Z only: the
stationary wave has no meaningful diurnal cycle at these levels.

    python scripts/strat/build_wave1_clim.py                # 1991-2020
    python scripts/strat/build_wave1_clim.py --y0 1981 --y1 2010
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "reference" / "wave1_clim.nc"

WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
G = 9.80665
LEVELS = (100, 500)
BAND = (55.0, 65.0)          # the wave-driving latitudes, as in wave1_maps
NHARM = 3                    # annual harmonics kept when smoothing over DOY


def band_coeff(z, lat_name="latitude", lon_name="longitude"):
    """Complex k=1 coefficient of the cos-weighted band mean, per time.

    Returns the rfft coefficient scaled so |coeff| is the wave AMPLITUDE in
    gpm (rfft returns a sum, not a mean, so it needs 2/N).
    """
    w = np.cos(np.deg2rad(z[lat_name]))
    prof = (z * w).sum(lat_name) / w.sum()            # time x lon
    spec = np.fft.rfft(prof.values, axis=-1)
    n = prof.sizes[lon_name]
    return spec[:, 1] * 2.0 / n


def smooth_doy(c: np.ndarray, nharm: int = NHARM) -> np.ndarray:
    """Keep the first `nharm` annual harmonics of a 366-long complex series.

    A raw day-of-year mean over 30 years still carries visible sampling noise,
    and the index divides by this climatology - noise in the denominator would
    show up as spurious day-to-day jitter in the published number.
    """
    s = np.fft.fft(c)
    keep = np.zeros_like(s)
    keep[0] = s[0]
    for k in range(1, nharm + 1):
        keep[k] = s[k]
        keep[-k] = s[-k]
    return np.fft.ifft(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1991)
    ap.add_argument("--y1", type=int, default=2020)
    a = ap.parse_args()

    print(f"  opening WeatherBench2 ERA5 ({a.y0}-{a.y1}, 00Z, {LEVELS} hPa)", flush=True)
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    z = ds["geopotential"].sel(time=slice(f"{a.y0}-01-01", f"{a.y1}-12-31"))
    z = z.isel(time=(z.time.dt.hour == 0))
    print(f"  {z.sizes['time']} daily fields", flush=True)

    out = {}
    for lev in LEVELS:
        zl = z.sel(level=lev) / G
        for hemi, lo, hi in (("nh", BAND[0], BAND[1]), ("sh", -BAND[1], -BAND[0])):
            sub = zl.sel(latitude=slice(lo, hi))
            if sub.sizes["latitude"] == 0:                 # descending latitude axis
                sub = zl.sel(latitude=slice(hi, lo))
            print(f"   {lev} hPa {hemi.upper()}: {sub.sizes['latitude']} lat rows"
                  f" ({float(sub.latitude.min()):.1f}..{float(sub.latitude.max()):.1f})",
                  flush=True)
            sub = sub.load()
            c = band_coeff(sub)
            doy = pd.DatetimeIndex(sub.time.values).dayofyear.values
            clim = np.zeros(366, dtype=complex)
            for d in range(1, 367):
                m = doy == d
                clim[d - 1] = c[m].mean() if m.any() else np.nan
            # 29 Feb is sampled ~1 year in 4; fill it from its neighbours before
            # smoothing so the harmonic fit is not pulled by a noisy point.
            if np.isnan(clim[59]):
                clim[59] = 0.5 * (clim[58] + clim[60])
            sm = smooth_doy(clim)
            out[f"{hemi}{lev}_re"] = ("doy", sm.real.astype("float32"))
            out[f"{hemi}{lev}_im"] = ("doy", sm.imag.astype("float32"))
            amp = np.abs(sm)
            print(f"     climatological wave-1 amplitude: "
                  f"min {amp.min():.0f}  max {amp.max():.0f} gpm", flush=True)

    d = xr.Dataset(out, coords={"doy": np.arange(1, 367)})
    d.attrs["source"] = f"WeatherBench2 ERA5 {a.y0}-{a.y1}, 00Z"
    d.attrs["band"] = f"{BAND[0]}-{BAND[1]} deg cos-weighted, both hemispheres"
    d.attrs["note"] = ("mean COMPLEX k=1 coefficient per day of year, smoothed to "
                       f"{NHARM} annual harmonics; |coeff| is amplitude in gpm")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_netcdf(OUT, encoding={k: {"zlib": True, "complevel": 6} for k in out})
    print(f"  wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
