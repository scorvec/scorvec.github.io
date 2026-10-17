"""Shared grid, coarsening and archive I/O of the month-ahead outlook (scripts/monthly/).

The archive holds ERA5 half-month blocks on a 3° grid, one file per year (scripts/monthly/data/era5_<YYYY>.npz):

    block 0 = days 1-7 of the month, block 1 = days 8 to the month's end

so a calendar-month mean is the count-weighted mean of a month's two blocks, and the predictor window of an outlook
issued on the 15th of month M-1 is block 1 of M-2 plus block 0 of M-1 (the ~30 days ending on the 7th of M-1, the
last week ERA5T reliably holds on the 15th: ARCO publishes ERA5T about six days behind real time).

Fields (int16-packed, PACK gives scale/offset/fill):
    z500  500 hPa geopotential height (m)          NH grid, 0-90N
    t2m   2 m temperature (°C)                     NH grid
    sic   sea-ice concentration (0-1)              NH grid, NaN over land
    sst   sea-surface temperature (°C)             SST grid, 30S-66N, NaN over land
    snow  snow depth (m of water equivalent)       NH grid
    swvl  volumetric soil water, 0-7 cm (m³/m³)    NH grid (ERA5 holds it fixed over the ocean)
Indices (float32): u50eq zonal-mean u at 50 hPa, 5S-5N (QBO); u50v the same at 57-63N (lower polar vortex).
`n` counts the 6-hourly samples in each block (0 = not yet in the archive).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

RES = 3.0
LON = np.arange(0.0, 360.0, RES)  # 120
LAT_NH = np.arange(0.0, 90.0 + RES / 2, RES)  # 31, 0..90
LAT_SST = np.arange(-30.0, 66.0 + RES / 2, RES)  # 33, -30..66
FIELDS = {"z500": LAT_NH, "t2m": LAT_NH, "sic": LAT_NH, "sst": LAT_SST, "snow": LAT_NH, "swvl": LAT_NH}
INDICES = ("u50eq", "u50v")
SPLIT_DAY = 8  # block 1 starts on this day

# scale, offset, fill: value = int16 * scale + offset
PACK = {
    "z500": (0.1, 5500.0),
    "t2m": (0.01, 0.0),
    "sic": (1e-4, 0.0),
    "sst": (0.01, 0.0),
    "snow": (1e-3, 0.0),
    "swvl": (1e-4, 0.0),
}
FILL = np.int16(-32768)


def lat_weights(src_lat: np.ndarray, dst_lat: np.ndarray, half: float) -> np.ndarray:
    """(dst, src) weights of a separable box average: each destination cell spans dst±half; a source point on a
    cell edge counts half. Rows are not normalised (the masked mean divides by the summed weights)."""
    d = np.abs(dst_lat[:, None] - src_lat[None, :])
    w = np.where(d < half - 1e-6, 1.0, 0.0) + np.where(np.abs(d - half) <= 1e-6, 0.5, 0.0)
    return w


def lon_weights(src_lon: np.ndarray, dst_lon: np.ndarray, half: float) -> np.ndarray:
    d = np.abs(((dst_lon[:, None] - src_lon[None, :]) + 180.0) % 360.0 - 180.0)
    return np.where(d < half - 1e-6, 1.0, 0.0) + np.where(np.abs(d - half) <= 1e-6, 0.5, 0.0)


class Coarsener:
    """Box-average a (..., lat, lon) source grid onto the 3° archive grids, ignoring NaNs; a destination cell keeps a
    value when at least `min_frac` of its weight is valid (SST and sea ice: half the cell must be ocean)."""

    def __init__(self, src_lat: np.ndarray, src_lon: np.ndarray):
        self.wlon = lon_weights(src_lon, LON, RES / 2)
        self.wlat = {k: lat_weights(src_lat, v, RES / 2) for k, v in (("nh", LAT_NH), ("sst", LAT_SST))}

    def __call__(self, x: np.ndarray, grid: str, min_frac: float = 0.0) -> np.ndarray:
        wl = self.wlat[grid]
        m = np.isfinite(x)
        x0 = np.where(m, x, 0.0)
        num = np.einsum("ij,...jk,lk->...il", wl, x0, self.wlon, optimize=True)
        den = np.einsum("ij,...jk,lk->...il", wl, m.astype(np.float64), self.wlon, optimize=True)
        full = wl.sum(1)[:, None] * self.wlon.sum(1)[None, :]
        out = num / np.where(den > 0, den, np.nan)
        return np.where(den >= min_frac * full, out, np.nan)


def band_mean(x: np.ndarray, lat: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """cos-weighted mean over lo<=lat<=hi and all longitudes of a (..., lat, lon) array."""
    sel = (lat >= lo - 1e-6) & (lat <= hi + 1e-6)
    w = np.cos(np.deg2rad(lat[sel]))
    return np.einsum("...jk,j->...", x[..., sel, :], w) / (w.sum() * x.shape[-1])


def empty_year() -> dict:
    out = {k: np.full((12, 2, len(v), len(LON)), np.nan, np.float32) for k, v in FIELDS.items()}
    out.update({k: np.full((12, 2), np.nan, np.float32) for k in INDICES})
    out["n"] = np.zeros((12, 2), np.int32)
    return out


def year_path(y: int) -> Path:
    return DATA / f"era5_{y}.npz"


def save_year(y: int, d: dict, path: Path | None = None) -> None:
    arrs = {}
    for k in FIELDS:
        s, o = PACK[k]
        v = d[k]
        q = np.round((v - o) / s)
        arrs[k] = np.where(np.isfinite(q), np.clip(q, -32767, 32767), FILL).astype(np.int16)
    for k in INDICES:
        arrs[k] = d[k].astype(np.float32)
    arrs["n"] = d["n"].astype(np.int32)
    DATA.mkdir(parents=True, exist_ok=True)
    path = path or year_path(y)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    tmp.replace(path)


def load_year(y: int, path: Path | None = None, raw: bool = False) -> dict:
    """One archive year; ARCO-sourced years (>= ARCO_FROM) get the WB2-minus-ARCO offset unless raw."""
    p = path or year_path(y)
    if not p.exists():
        return empty_year()
    with np.load(p) as f:
        d = {}
        for k in FIELDS:
            if k not in f.files:  # a field added to the archive after this year was built
                d[k] = np.full((12, 2, len(FIELDS[k]), len(LON)), np.nan, np.float32)
                continue
            s, o = PACK[k]
            q = f[k]
            d[k] = np.where(q == FILL, np.nan, q.astype(np.float32) * s + o).astype(np.float32)
        for k in INDICES:
            d[k] = f[k].astype(np.float32)
        d["n"] = f["n"].astype(np.int32)
    if not raw and path is None and y >= ARCO_FROM:
        off = arco_offset()
        if off is not None:
            for k in FIELDS:
                d[k] = d[k] - off[k]
    return d


ARCO_FROM = 2023
OVERLAP = DATA / "arco_overlap_2022.npz"
_OFFSET: dict | None = None


def arco_offset() -> dict | None:
    """ARCO-minus-WB2 difference of the two sources' 3° cells (regridding, not weather), from 2022 built both ways:
    per calendar month, the mean over the six blocks of that month and its two neighbours."""
    global _OFFSET  # noqa: PLW0603  computed once per process
    if _OFFSET is None and OVERLAP.exists():
        a, w = load_year(2022, OVERLAP), load_year(2022, raw=True)
        _OFFSET = {}
        for k in FIELDS:
            dif = a[k] - w[k]  # (12, 2, lat, lon)
            per = np.nanmean(dif, axis=1)
            with np.errstate(invalid="ignore"):
                sm = np.stack([np.nanmean(per[[(m - 1) % 12, m, (m + 1) % 12]], 0) for m in range(12)])
            _OFFSET[k] = np.nan_to_num(sm)[:, None].astype(np.float32)
    return _OFFSET


def expected_samples(y: int, m: int, half: int, per_day: int = 4) -> int:
    import calendar

    nd = calendar.monthrange(y, m)[1]
    days = SPLIT_DAY - 1 if half == 0 else nd - SPLIT_DAY + 1
    return days * per_day
