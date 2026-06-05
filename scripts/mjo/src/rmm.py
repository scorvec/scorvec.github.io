"""
Compute wind-only RMM1/RMM2 for an AIFS-ENS ensemble forecast.

AIFS-ENS does not output OLR, so we project only onto the U850 and U200
components of the Wheeler & Hendon (2004) reference EOFs.  The amplitude
is slightly lower than the full 3-variable RMM, but the MJO phase is valid.

Steps:
  1. Load AIFS-ENS GRIB files for u at 850/200 hPa.
  2. Regrid to 2.5°, average 15°S–15°N retaining longitude.
  3. Subtract W&H (2004) reference climatological annual cycle.
  4. Normalise by reference standard deviations.
  5. Project onto wind portions of reference EOFs → RMM1, RMM2.

Usage:
    python src/rmm.py --aifs-dir data/aifs --date 20240601 --time 00 \
                      --out data/aifs/rmm_20240601_00z.nc
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


LAT_MIN, LAT_MAX = -15, 15
LAT_PAD = 25.0   # raw band kept before interp; wider than 15°S–15°N for interp edges
GRID_RES = 2.5


def load_aifs_uwnd(path: Path) -> tuple[xr.DataArray, xr.DataArray]:
    """Load U850 and U200 from a GRIB2 file as lazy, tropics-only 2.5° fields.

    Memory note: the full ``pf`` array is ~25 GB (50 members × 60 steps × 2
    levels × 721 × 1440).  We keep everything lazy (dask, chunked one member at
    a time) and slice the tropical latitude band *before* interpolating, so the
    global high-resolution field is never materialised in RAM at once.

    Returns (u850, u200) each with dims (step, latitude, longitude) for cf,
    or (number, step, latitude, longitude) for pf.
    """
    # chunks={} → lazy dask single-chunk (no data read on open); indexpath=""
    # avoids littering the data dir with .idx cache files.
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "u"}, "indexpath": ""},
        chunks={},
    )
    if "u" not in ds:
        raise ValueError(f"Variable u not found in {path}")

    da = ds["u"]
    da = da.sel(isobaricInhPa=[850, 200])   # the file now carries all 13 AAM levels; keep 2
    if "number" in da.dims:
        da = da.chunk({"number": 1})   # process one ensemble member at a time

    # Restrict to the tropical band on the raw grid first (≈14× fewer points).
    da = da.sortby("latitude").sel(latitude=slice(-LAT_PAD, LAT_PAD))

    # Normalise longitude to 0–360 monotonic (cheap now that the array is small).
    if float(da.longitude.min()) < 0:
        da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")

    lon_new = np.arange(0, 360, GRID_RES)
    lat_new = np.arange(-20, 20.01, GRID_RES)
    da = da.interp(latitude=lat_new, longitude=lon_new, method="linear")

    u850 = da.sel(isobaricInhPa=850, drop=True)
    u200 = da.sel(isobaricInhPa=200, drop=True)
    return u850, u200


def compute_rmm(
    aifs_dir: Path,
    date: str,
    run_time: str,
    clim: xr.Dataset,
    eofs: xr.Dataset,
    mean120: dict[str, np.ndarray] | None = None,
) -> xr.Dataset:
    """Project the AIFS ensemble onto the wind portions of the reference EOFs.

    ``mean120`` (optional) holds the trailing 120-day-mean U850/U200 anomaly
    maps from recent NCEP analysis (keys "u850"/"u200", each shape (nlon,)).
    Subtracting it applies the Wheeler & Hendon (2004) low-frequency / ENSO
    filter to the forecast, held fixed across lead time.  Without it the
    forecast retains interannual variability and the trajectory drifts.
    """
    stem = f"aifs_{date}_{run_time}z"

    pf_path = aifs_dir / f"{stem}.pf.u.grib2"
    cf_path = aifs_dir / f"{stem}.cf.u.grib2"

    # Precompute wind-only EOF vectors and normalization factors
    e1 = np.concatenate([eofs["eof_u850"].sel(mode=1).values,
                         eofs["eof_u200"].sel(mode=1).values])
    e2 = np.concatenate([eofs["eof_u850"].sel(mode=2).values,
                         eofs["eof_u200"].sel(mode=2).values])
    pc_wind_std1 = float(eofs["pc_wind_std"].sel(mode=1))
    pc_wind_std2 = float(eofs["pc_wind_std"].sel(mode=2))

    rmm1_members = []
    rmm2_members = []
    member_ids = []

    for label, u_path in [("cf", cf_path), ("pf", pf_path)]:
        print(f"  Processing {label} …")

        u850, u200 = load_aifs_uwnd(u_path)

        # 6-hourly → daily means
        def to_daily(da: xr.DataArray) -> xr.DataArray:
            hours = da.step / np.timedelta64(1, "h")
            da = da.assign_coords(step=hours.astype(int))
            day_idx = (da.step.values // 24).astype(int)
            da = da.assign_coords(day=("step", day_idx))
            return da.groupby("day").mean()

        u850_d = to_daily(u850)
        u200_d = to_daily(u200)
        # Force the heavy lazy pipeline (cfgrib read → interp → daily-mean) to run ONCE,
        # vectorized over all members. Without this, `.values` inside the per-member loop
        # below re-triggered the whole dask graph 50× — minutes of scheduling/read overhead.
        u850_d, u200_d = u850_d.load(), u200_d.load()

        def lat_mean_da(da):
            lat = da.latitude if "latitude" in da.coords else da.lat
            mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
            w = np.cos(np.deg2rad(lat)).where(mask, 0.0)
            return da.weighted(w).mean(dim=lat.name)

        n_mem = u850_d.sizes.get("number", 1)
        mem_dim = "number" if "number" in u850_d.dims else None

        init_dt = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{run_time}:00")
        lead_days = u850_d.day.values
        valid_times = [init_dt + pd.Timedelta(days=int(d)) for d in lead_days]
        doy = np.array([pd.Timestamp(t).dayofyear for t in valid_times])

        for m in range(n_mem):
            if mem_dim:
                u850_m = u850_d.isel(number=m)
                u200_m = u200_d.isel(number=m)
                mem_id = f"{label}_{m:03d}"
            else:
                u850_m, u200_m = u850_d, u200_d
                mem_id = label

            u850_lm = lat_mean_da(u850_m)
            u200_lm = lat_mean_da(u200_m)

            a_u850 = u850_lm.values - clim["clim_u850"].sel(dayofyear=doy).values
            a_u200 = u200_lm.values - clim["clim_u200"].sel(dayofyear=doy).values

            # Remove the recent 120-day-mean (low-frequency / ENSO) signal.
            if mean120 is not None:
                a_u850 = a_u850 - mean120["u850"]
                a_u200 = a_u200 - mean120["u200"]

            a_u850 /= clim.attrs["std_u850"]
            a_u200 /= clim.attrs["std_u200"]

            # (nday, 2*nlon) @ (2*nlon,) → (nday,); normalise to unit amplitude
            combined = np.concatenate([a_u850, a_u200], axis=1)
            rmm1_members.append((combined @ e1) / pc_wind_std1)
            rmm2_members.append((combined @ e2) / pc_wind_std2)
            member_ids.append(mem_id)

    rmm1_arr = np.array(rmm1_members)
    rmm2_arr = np.array(rmm2_members)

    return xr.Dataset(
        {
            "rmm1": xr.DataArray(rmm1_arr, dims=["member", "lead_day"],
                                 coords={"member": member_ids, "lead_day": lead_days}),
            "rmm2": xr.DataArray(rmm2_arr, dims=["member", "lead_day"],
                                 coords={"member": member_ids, "lead_day": lead_days}),
        },
        attrs={
            "init_date": date,
            "init_time": run_time,
            "note": "Wind-only RMM (U850+U200); OLR not available from AIFS-ENS",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aifs-dir", default="data/aifs")
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    parser.add_argument("--time", default="00")
    parser.add_argument("--clim", default="data/reference/climatology.nc")
    parser.add_argument("--eofs", default="data/reference/eofs.nc")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    clim = xr.open_dataset(args.clim)
    eofs = xr.open_dataset(args.eofs)

    rmm = compute_rmm(Path(args.aifs_dir), args.date, args.time, clim, eofs)

    out_path = args.out or f"data/aifs/rmm_{args.date}_{args.time}z.nc"
    rmm.to_netcdf(out_path)
    print(f"RMM saved to {out_path}")


if __name__ == "__main__":
    main()
