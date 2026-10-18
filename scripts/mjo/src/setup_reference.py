"""
Download NOAA/CPC reference data and compute Wheeler & Hendon (2004) EOFs.

Downloads (run once, ~1.4 GB total):
  - NOAA interpolated OLR          ~200 MB  (one file, 1974–present)
  - NCEP/NCAR reanalysis u-wind    ~1.2 GB  (one file per year, 1979–2001)

Outputs in data/reference/:
  - eofs.nc          EOF patterns (mode × longitude) for OLR, U850, U200
  - climatology.nc   Smoothed annual cycle + normalisation factors

The observed index computed on this reference is rmm_era5.py's.

Usage:
    python src/setup_reference.py
"""

from pathlib import Path

import requests
import numpy as np
import pandas as pd
import xarray as xr
from eofs.standard import Eof

from doycal import noleap_doy, sel_doy

# ── Paths ─────────────────────────────────────────────────────────────────────
REF_DIR = Path("data/reference")
NCEP_DIR = REF_DIR / "ncep_uwnd"
REF_DIR.mkdir(parents=True, exist_ok=True)
NCEP_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -15, 15
N_HARMONICS = 3
BASE_START = 1979
BASE_END = 2001  # Wheeler & Hendon (2004) base period


# ── Helpers ───────────────────────────────────────────────────────────────────
HEADERS = {"User-Agent": "scorvec.com data pipeline (+https://scorvec.com)"}


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  {dest.name}: already exists, skipping")
        return dest
    print(f"  {dest.name}: downloading …")
    with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  {dest.name}: done ({dest.stat().st_size // 1_000_000} MB)")
    return dest


def lat_mean(da: xr.DataArray) -> xr.DataArray:
    """Latitude-weighted mean over 15°S–15°N, retaining longitude."""
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lat = da[lat_name]
    mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    w = np.cos(np.deg2rad(lat)).where(mask, 0.0)
    return da.weighted(w).mean(dim=lat_name)


def smooth_annual_cycle(da: xr.DataArray) -> xr.DataArray:
    """Fit N_HARMONICS Fourier harmonics; return climatology (dayofyear, lon).

    The phase is the common-year day (doycal.noleap_doy: 1 March = 60 in every year, 29 Feb = 59.5), so leap and
    common years share one annual cycle. Output slots 1..365 are common-year days (slot 366 repeats slot 1); look
    dates up with doycal.sel_doy."""
    doy = noleap_doy(da.time.values)
    t = (doy - 1) / 365.0 * 2 * np.pi
    X = [np.ones(len(t))]
    for k in range(1, N_HARMONICS + 1):
        X.extend([np.cos(k * t), np.sin(k * t)])
    X = np.column_stack(X)
    coeffs, *_ = np.linalg.lstsq(X, da.values, rcond=None)

    doys = np.arange(1, 367)
    t_c = (doys - 1) / 365.0 * 2 * np.pi
    X_c = [np.ones(366)]
    for k in range(1, N_HARMONICS + 1):
        X_c.extend([np.cos(k * t_c), np.sin(k * t_c)])
    clim = np.column_stack(X_c) @ coeffs

    lon_name = "lon" if "lon" in da.coords else "longitude"
    return xr.DataArray(
        clim, coords={"dayofyear": doys, "longitude": da[lon_name].values}, dims=["dayofyear", "longitude"]
    )


def remove_prev120(arr: np.ndarray, times) -> np.ndarray:
    """Subtract the trailing 120-day running mean from a (time, lon) anomaly.

    This is the Wheeler & Hendon (2004) low-frequency / ENSO filter: at each
    day, the mean of the previous 120 days (window ending on that day) is
    removed.  Reindexed to continuous calendar days so the window is 120 *days*
    regardless of gaps; the first ~120 days come back as NaN (drop them).
    """
    idx = pd.to_datetime(times)
    df = pd.DataFrame(arr, index=idx)
    full = df.reindex(pd.date_range(idx.min(), idx.max(), freq="D"))
    roll = full.rolling(window=120, min_periods=120).mean()
    return (full - roll).reindex(idx).values


def align_modes(eofs: np.ndarray, lon: np.ndarray):
    """(order, (sign1, sign2)) that put the two EOFs in the Wheeler & Hendon (2004) convention, read from the OLR
    patterns themselves (an EOF's sign and the order of two near-equal modes are arbitrary). EOF2 is the mode with the
    larger OLR loading over the Indian Ocean (60-100E), made positive there (convection suppressed over the Indian
    Ocean, enhanced over the western Pacific); EOF1's loading over the Maritime Continent (100-150E) is made negative
    (enhanced convection there). Phases 4-5 then hold the convection over the Maritime Continent and 6-7 over the
    western Pacific, as in WH04's phase diagram; the committed eofs.nc satisfies it as it stands."""
    nlon = len(lon)
    olr = eofs[:, :nlon]
    io = (lon >= 60) & (lon <= 100)
    mc = (lon >= 100) & (lon <= 150)
    order = [1, 0] if abs(olr[0, io].mean()) > abs(olr[1, io].mean()) else [0, 1]
    o = olr[order]
    return order, (-1.0 if o[0, mc].mean() > 0 else 1.0, 1.0 if o[1, io].mean() > 0 else -1.0)


# ── 1. NOAA interpolated OLR ─────────────────────────────────────────────────
def download_noaa_olr() -> Path:
    return download(
        "https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc",
        REF_DIR / "olr.day.mean.nc",
    )


# ── 2. NCEP/NCAR reanalysis u-wind ───────────────────────────────────────────
def download_ncep_uwnd() -> list[Path]:
    return [
        download(
            f"https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/uwnd.{year}.nc",
            NCEP_DIR / f"uwnd.{year}.nc",
        )
        for year in range(BASE_START, BASE_END + 1)
    ]


# ── 3. Compute climatology and EOFs ──────────────────────────────────────────
def compute_reference() -> None:
    clim_out = REF_DIR / "climatology.nc"
    eof_out = REF_DIR / "eofs.nc"
    if clim_out.exists() and eof_out.exists():
        print("climatology.nc and eofs.nc already exist, skipping computation")
        return

    print("Loading NOAA OLR …")
    olr_ds = xr.open_dataset(REF_DIR / "olr.day.mean.nc")
    olr = olr_ds["olr"].sel(time=slice(str(BASE_START), str(BASE_END)))

    print("Loading NCEP u-wind …")
    ncep_files = sorted(NCEP_DIR.glob("uwnd.*.nc"))
    uwnd_ds = xr.open_mfdataset(ncep_files, combine="by_coords")
    u850 = uwnd_ds["uwnd"].sel(level=850, drop=True).sel(time=slice(str(BASE_START), str(BASE_END)))
    u200 = uwnd_ds["uwnd"].sel(level=200, drop=True).sel(time=slice(str(BASE_START), str(BASE_END)))

    # Align on common times
    common = np.intersect1d(
        np.intersect1d(olr.time.values, u850.time.values),
        u200.time.values,
    )
    olr = olr.sel(time=common)
    u850 = u850.sel(time=common)
    u200 = u200.sel(time=common)

    # Interpolate OLR to 2.5° to match NCEP grid if needed
    lon_ncep = u850.lon if "lon" in u850.coords else u850.longitude
    lat_ncep = u850.lat if "lat" in u850.coords else u850.latitude
    olr = olr.interp(lon=lon_ncep, lat=lat_ncep, method="linear")

    print("Computing latitude means …")
    olr_lm = lat_mean(olr)
    u850_lm = lat_mean(u850)
    u200_lm = lat_mean(u200)

    print("Fitting smoothed annual cycles …")
    clim_olr = smooth_annual_cycle(olr_lm)
    clim_u850 = smooth_annual_cycle(u850_lm)
    clim_u200 = smooth_annual_cycle(u200_lm)

    tt = olr_lm.time.values
    anom_olr = olr_lm.values - sel_doy(clim_olr, tt)
    anom_u850 = u850_lm.values - sel_doy(clim_u850, tt)
    anom_u200 = u200_lm.values - sel_doy(clim_u200, tt)

    print("Removing previous-120-day mean (low-frequency / ENSO filter) …")
    anom_olr = remove_prev120(anom_olr, common)
    anom_u850 = remove_prev120(anom_u850, common)
    anom_u200 = remove_prev120(anom_u200, common)
    valid = ~(np.isnan(anom_olr).any(1) | np.isnan(anom_u850).any(1) | np.isnan(anom_u200).any(1))
    common = common[valid]
    anom_olr = anom_olr[valid]
    anom_u850 = anom_u850[valid]
    anom_u200 = anom_u200[valid]
    print(f"  {valid.sum()} days retained after dropping incomplete 120-day windows")

    std_olr = float(np.std(anom_olr, ddof=1))
    std_u850 = float(np.std(anom_u850, ddof=1))
    std_u200 = float(np.std(anom_u200, ddof=1))

    xr.Dataset(
        {"clim_olr": clim_olr, "clim_u850": clim_u850, "clim_u200": clim_u200},
        attrs={
            "std_olr": std_olr,
            "std_u850": std_u850,
            "std_u200": std_u200,
            "base_period": f"{BASE_START}–{BASE_END}",
            "n_harmonics": N_HARMONICS,
        },
    ).to_netcdf(clim_out)
    print(f"  climatology.nc saved  (std_olr={std_olr:.3f}, std_u850={std_u850:.3f}, std_u200={std_u200:.3f})")

    print("Computing combined EOFs …")
    anom_olr /= std_olr
    anom_u850 /= std_u850
    anom_u200 /= std_u200
    combined = np.concatenate([anom_olr, anom_u850, anom_u200], axis=1)

    solver = Eof(combined)
    eofs = solver.eofs(neofs=2)
    pcs = solver.pcs(npcs=2)
    varfrac = solver.varianceFraction()[:2]
    nlon = anom_olr.shape[1]

    # EOF sign and 1-vs-2 ordering in the WH04 convention (align_modes).
    lon_vals = (olr_lm.lon if "lon" in olr_lm.coords else olr_lm.longitude).values
    order, (s1, s2) = align_modes(eofs, lon_vals)
    eofs = eofs[order]
    pcs = pcs[:, order]
    varfrac = np.asarray(varfrac)[order]
    eofs[0] *= s1
    pcs[:, 0] *= s1
    eofs[1] *= s2
    pcs[:, 1] *= s2
    print(f"  WH04 convention: order={order} signs=({s1:+.0f},{s2:+.0f})")

    # Normalization factors so that projected values have unit amplitude.
    # Full-variable std (for reference / diagnostics).
    pc_std = np.std(pcs, axis=0, ddof=1)  # shape (2,)

    # Wind-only std: project the training wind anomalies onto the wind portions
    # of the EOFs.  Forecast data (wind-only) is divided by this to get RMM.
    e1_wind = np.concatenate([eofs[0, nlon : 2 * nlon], eofs[0, 2 * nlon :]])
    e2_wind = np.concatenate([eofs[1, nlon : 2 * nlon], eofs[1, 2 * nlon :]])
    wind_train = np.concatenate([anom_u850, anom_u200], axis=1)
    pc_wind_std = np.array(
        [
            float(np.std(wind_train @ e1_wind, ddof=1)),
            float(np.std(wind_train @ e2_wind, ddof=1)),
        ]
    )

    xr.Dataset(
        {
            "eof_olr": xr.DataArray(
                eofs[:, :nlon], dims=["mode", "longitude"], coords={"mode": [1, 2], "longitude": lon_vals}
            ),
            "eof_u850": xr.DataArray(
                eofs[:, nlon : 2 * nlon], dims=["mode", "longitude"], coords={"mode": [1, 2], "longitude": lon_vals}
            ),
            "eof_u200": xr.DataArray(
                eofs[:, 2 * nlon :], dims=["mode", "longitude"], coords={"mode": [1, 2], "longitude": lon_vals}
            ),
            "pc": xr.DataArray(pcs, dims=["time", "mode"], coords={"time": common, "mode": [1, 2]}),
            "variance_fraction": xr.DataArray(varfrac, dims=["mode"], coords={"mode": [1, 2]}),
            "pc_std": xr.DataArray(pc_std, dims=["mode"], coords={"mode": [1, 2]}),
            "pc_wind_std": xr.DataArray(pc_wind_std, dims=["mode"], coords={"mode": [1, 2]}),
        },
        attrs={"description": "Wheeler & Hendon (2004) reference EOFs"},
    ).to_netcdf(eof_out)
    print(f"  eofs.nc saved  (EOF1={varfrac[0] * 100:.1f}%, EOF2={varfrac[1] * 100:.1f}%)")
    print(f"  pc_std={pc_std[0]:.3f},{pc_std[1]:.3f}  pc_wind_std={pc_wind_std[0]:.3f},{pc_wind_std[1]:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== NOAA interpolated OLR ===")
    download_noaa_olr()

    print("\n=== NCEP/NCAR reanalysis u-wind (1979–2001) ===")
    download_ncep_uwnd()

    print("\n=== Computing reference EOFs ===")
    compute_reference()

    print("\nAll done. Reference files in data/reference/")
