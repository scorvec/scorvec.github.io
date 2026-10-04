"""Compact storage of the lead-dependent GEPS8 model climatology, and its loader.

The climatology (build/build_clim.py) is a (day-of-year centre, lead, lat, lon) mean over the 2001-2020 GEPS8 hindcast
starts within +-15 days of each centre: 74 centres x 39 leads x 1 deg global, nine fields, 2.2 GB as packed netCDF.
It is smooth along BOTH the day-of-year and the lead axis, which ordinary netCDF compression cannot exploit, so the
pack stores it as:

  q   = round(value / STEP)               quantised; STEP is a fixed, stated resolution per field (below)
  d   = q differenced along day of year, then along lead
  pack = lzma(int16 d)

Only leads 0.5 .. 34.5 are kept: the page uses days 1-35, which read L = day - 0.5. Decoding is two cumulative sums;
the result is the original field to within STEP / 2 everywhere (make_reference.py --check reports the measured
maximum error per field). STEP is far below anything the maps, indices or probabilities can resolve - e.g. 0.1 K on a
temperature climatology whose 20-year sampling error alone is ~0.3 K at long leads.
"""
from __future__ import annotations

import functools
import lzma

import numpy as np
import xarray as xr

import paths

# quantisation step per field, in the climatology's own units
# (~1 % of each field's anomaly colour range, and below the climatology's own 20-year sampling noise; halving a step
# costs ~11 MB per field, measured)
STEP = {"t2m": 0.1,        # K
        "zg500": 1.0,      # m
        "mslp": 10.0,      # Pa  (0.1 hPa)
        "olr": 0.25,       # W m-2
        "u850": 0.1, "u200": 0.1, "v850": 0.1, "v200": 0.1,       # m s-1
        "pr": 5e-7}        # kg m-2 s-1  (0.043 mm/day)
LMAX = 35.0                # keep L < LMAX: days 1-35


def encode(da: xr.DataArray, tag: str) -> dict:
    """(doy, L, Y, X) DataArray -> dict of arrays for np.savez."""
    da = da.sel(L=da.L[da.L < LMAX]).transpose("doy", "L", "Y", "X")
    a = da.values.astype("float64")
    if not np.isfinite(a).all():
        raise ValueError(f"{tag}: non-finite values in the climatology")
    step = STEP[tag]
    q = np.round(a / step).astype(np.int64)
    offset = int(np.round(np.median(q)))
    q -= offset
    d = q.copy()
    d[1:] -= q[:-1]                    # along day of year
    d2 = d.copy()
    d2[:, 1:] -= d[:, :-1]             # then along lead
    if d2.min() < -32768 or d2.max() > 32767:
        raise ValueError(f"{tag}: delta range {d2.min()}..{d2.max()} overflows int16 - raise STEP")
    blob = lzma.compress(d2.astype("<i2").tobytes(), preset=6)
    return {"blob": np.frombuffer(blob, dtype=np.uint8), "shape": np.array(d2.shape, dtype=np.int64),
            "step": np.float64(step), "offset": np.int64(offset),
            "doy": da.doy.values.astype(np.int64), "L": da.L.values.astype(np.float32),
            "Y": da.Y.values.astype(np.float32), "X": da.X.values.astype(np.float32),
            "name": np.array(da.name or tag)}


def decode(z) -> xr.DataArray:
    shape = tuple(int(x) for x in z["shape"])
    d = np.frombuffer(lzma.decompress(z["blob"].tobytes()), dtype="<i2").reshape(shape).astype(np.int32)
    np.cumsum(d, axis=1, out=d)        # undo the lead difference
    np.cumsum(d, axis=0, out=d)        # undo the day-of-year difference
    d += np.int32(z["offset"])
    v = d.astype(np.float32)
    del d
    v *= np.float32(z["step"])
    return xr.DataArray(v, dims=("doy", "L", "Y", "X"), name=str(z["name"]),
                        coords={"doy": z["doy"], "L": z["L"], "Y": z["Y"], "X": z["X"]})


def pack_path(tag: str):
    return paths.CLIM / f"geps8_clim_{tag}.npz"


@functools.lru_cache(maxsize=3)
def clim(tag: str) -> xr.DataArray | None:
    """The lead-dependent GEPS8 climatology for one field, (doy, L, Y, X), float32; None if no pack exists.
    Cached: decoding costs a few seconds and ~0.7 GB, and several products read the same field."""
    p = pack_path(tag)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return decode(z)
