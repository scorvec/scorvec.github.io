"""ERA5 3-month means (local ~/era5_store) for Part B: t2m global (wb2_1p5_daily_global, to 2026-06), precipitation
0-90N (wb2_1p5_daily, mm/day, to 2023). Seasons per start: Sep -> OND, NDJ; Oct -> NDJ, DJF (year = start year)."""
import pyproj  # noqa: F401
import sys, os, numpy as np, xarray as xr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2 import OUT

YRS = list(range(1991, 2026))
GLAT = np.arange(-90.0, 90.0 + 1e-6, 1.5); GLON = np.arange(0.0, 360.0 - 1e-6, 1.5)
SPEC = {"t2m": ("wb2_1p5_daily_global", "t2m", GLAT), "pr": ("wb2_1p5_daily", "prcp", GLAT[GLAT >= 0])}
SEAS = {9: ((10, 11, 12), (11, 12, 13)), 10: ((11, 12, 13), (12, 13, 14))}   # month numbers, >12 = next year


def monthly(sub, v, lat):
    out = {}
    for y in range(1991, 2027):
        f = os.path.expanduser("~/era5_store") + f"/{sub}/{v}/{v}_{y}.nc"
        if not os.path.exists(f): continue
        da = xr.open_dataset(f)[v].transpose("time", "latitude", "longitude")
        da = da.reindex(latitude=lat, longitude=GLON, method="nearest", tolerance=0.01)
        g = da.groupby("time.month")
        cnt = da["time"].groupby("time.month").count().to_series()
        mm = g.mean().load()
        for m in mm.month.values:
            full = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}[int(m)]
            if cnt[m] >= full: out[(y, int(m))] = mm.sel(month=m).values.astype(np.float64)
    return out


res = {}
for var, (sub, v, lat) in SPEC.items():
    M = monthly(sub, v, lat)
    A = np.full((2, len(YRS), 2, len(lat), len(GLON)), np.nan, np.float32)
    for mi, m in enumerate((9, 10)):
        for yi, y in enumerate(YRS):
            for k, ms in enumerate(SEAS[m]):
                keys = [(y + (mm - 1) // 12, (mm - 1) % 12 + 1) for mm in ms]
                if all(kk in M for kk in keys):
                    A[mi, yi, k] = np.mean([M[kk] for kk in keys], 0)
    if var == "t2m": assert 270 < np.nanmean(A) < 290, np.nanmean(A)
    else: assert 1 < np.nanmean(A) < 5, np.nanmean(A)
    res[var] = A
    print(var, "seasons with data per (start, season):", np.isfinite(A[:, :, :, 10, 10]).sum(1).tolist(), flush=True)
np.savez(OUT + "era5_seasonal.npz", years=np.array(YRS), **res)
