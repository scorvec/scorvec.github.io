"""Shared grid / window definitions for the SFS vs GEFS vs GEPS hindcast comparison (scratch, 2026-10-12)."""
import numpy as np

YEARS = list(range(2001, 2020))                      # common to GEPS (2001-2020), GEFS (2000-2019), SFS (1991-2025)
MONTHS = (9, 10)                                     # SFS beta2 reforecast start months on S3
# day d = the UTC day init + (d - 1); windows in days (inclusive)
WINDOWS = {"wk1": (1, 7), "wk2": (8, 14), "wk34": (15, 28), "wk5": (29, 35)}
WNAMES = list(WINDOWS)
ELAT = np.arange(0.0, 90.0 + 1e-6, 1.5)             # ERA5 store wb2_1p5_daily grid: 0..90 N ascending (61)
ELON = np.arange(0.0, 360.0 - 1e-6, 1.5)            # 0..358.5 (240)
OUT = __import__("os").environ.get("SFS_SKILL_WORK", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work") + "/")


def weekly(daily):
    """daily (ndays>=35, ...) with index 0 = day 1 -> (4, ...) window means."""
    return np.stack([np.nanmean(daily[a - 1:b], axis=0) for a, b in WINDOWS.values()])


def regrid(field, lat, lon):
    """Bilinear from a regular lat/lon grid (any orientation, 0..360 lon) onto the ERA5 NH 1.5 deg grid; works on
    (..., lat, lon)."""
    from scipy.interpolate import RegularGridInterpolator
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    if lat[0] > lat[-1]:
        lat = lat[::-1]; field = field[..., ::-1, :]
    # wrap longitude for interpolation across 360
    lon2 = np.concatenate([lon, [lon[0] + 360.0]])
    f2 = np.concatenate([field, field[..., :1]], axis=-1)
    LA, LO = np.meshgrid(ELAT, ELON, indexing="ij")
    lead_shape = f2.shape[:-2]
    flat = f2.reshape(-1, f2.shape[-2], f2.shape[-1])
    out = np.empty((flat.shape[0], len(ELAT), len(ELON)), np.float32)
    for k in range(flat.shape[0]):
        g = RegularGridInterpolator((lat, lon2), flat[k], bounds_error=False, fill_value=None)
        out[k] = g((LA, LO))
    return out.reshape(*lead_shape, len(ELAT), len(ELON))
