"""Build the tropical-band ERA5 precip climatology for the pseudo-OLR channel.

One-time builder (like setup_reference.py): streams ERA5 daily precip
1991-2020 for the 15S-15N band from the public WeatherBench2 zarr (the
local wb2_1p5_daily store is NH-only so it cannot cover the band), takes
the cos-weighted band mean per longitude, and computes a +/-7-day-window
day-of-year climatology (mean and sigma across the 30 years) on the WH04
EOF 2.5-degree longitude grid.

Output: data/reference/prcp_clim.nc
  clim_prcp  (dayofyear, longitude)  mm/day, band-mean daily precip normal
  sigma_prcp (dayofyear, longitude)  mm/day, interannual+synoptic sigma
The pseudo-OLR channel is -(prcp_anom / scalar sigma(doy)) where the
scalar is the RMS of sigma_prcp over longitude — mirroring WH04's single
std_olr so the spatial variance pattern is preserved.

    python src/build_prcp_clim.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
OUT = Path(__file__).parent.parent / "data" / "reference" / "prcp_clim.nc"
Y0, Y1 = 1991, 2020
WINDOW = 7          # +/- days pooled per day-of-year
GRID_RES = 2.5


def main():
    ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
    pv = ("total_precipitation_6hr" if "total_precipitation_6hr" in ds
          else "total_precipitation")
    p = ds[pv].sel(time=slice(f"{Y0}", f"{Y1}"))
    p = p.sel(latitude=slice(None)).where(
        (p.latitude >= -15) & (p.latitude <= 15), drop=True)
    w = np.cos(np.deg2rad(p.latitude))
    band = p.weighted(w).mean("latitude")            # (time6h, lon240)
    daily = (band.resample(time="1D").sum() * 1000.0)  # mm/day
    print("streaming band precip ...", flush=True)
    daily = daily.compute()
    print(f"loaded {daily.sizes['time']} days x {daily.sizes['longitude']} lons",
          flush=True)

    # interp to the EOF 2.5-degree longitude grid (cyclic)
    lon = daily.longitude.values
    elon = np.arange(0, 360, GRID_RES)
    v = daily.values
    vi = np.concatenate([v, v[:, :1]], axis=1)
    loni = np.concatenate([lon, [lon[0] + 360]])
    ve = np.stack([np.interp(elon, loni, row) for row in vi])

    doy = pd.DatetimeIndex(daily.time.values).dayofyear.values
    doy = np.minimum(doy, 365)                       # fold leap day
    mu = np.zeros((366, len(elon)))
    sd = np.zeros_like(mu)
    for d in range(1, 367):
        dd = min(d, 365)
        dist = np.minimum(np.abs(doy - dd), 365 - np.abs(doy - dd))
        m = dist <= WINDOW
        mu[d - 1] = ve[m].mean(axis=0)
        sd[d - 1] = ve[m].std(axis=0, ddof=1)
        if d % 60 == 0:
            print(f"doy {d}/366", flush=True)

    out = xr.Dataset(
        {"clim_prcp": (("dayofyear", "longitude"), mu),
         "sigma_prcp": (("dayofyear", "longitude"), sd)},
        coords={"dayofyear": np.arange(1, 367), "longitude": elon},
        attrs={"source": WB2, "base_period": f"{Y0}-{Y1}",
               "window_days": WINDOW, "units": "mm/day",
               "note": ("15S-15N cos-weighted band mean; pseudo-OLR = "
                        "-(anom / RMS-over-lon sigma), WH04-style scalar "
                        "standardization")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT)
    print(f"wrote {OUT} (band clim {mu.mean():.2f} mm/day, "
          f"scalar sigma range {np.sqrt((sd**2).mean(1)).min():.2f}-"
          f"{np.sqrt((sd**2).mean(1)).max():.2f})")


if __name__ == "__main__":
    main()
