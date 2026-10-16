"""Message-by-message reading of the C3S seasonal GRIB files.

cfgrib opens a lagged ensemble as a dense number x start x step hypercube that is almost all
padding: NCEP CFSv2's monthly file (120 members, each with its own start, steps in hours) becomes
120 x 120 x 178 fields, and averaging one global 1 deg tercile period out of it peaks near 25 GB,
more than an Actions runner holds. Read one message at a time, a file costs one field at a time,
and every message names its own member, start, lead and nominal start, so lagged and single-start
systems reduce the same way.

Keys the builders use: `number`, `dataDate`/`dataTime` (the member's own start), `forecastMonth`
(monthly files; lead 1 = the issue month), `indexingDate` (the nominal start the CDS files a lagged
member under: a January hindcast member started on 2 December 1996 belongs to 1997),
`validityDate`/`validityTime` (sub-daily files) and `cfVarName` (t2m, tprate, sst).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A regular lat-lon grid as stored: `lat` (nj,) in file order, `lon` (ni,) in 0..360 deg E."""

    lat: np.ndarray
    lon: np.ndarray


def _grid(h) -> Grid:
    import eccodes

    kind = eccodes.codes_get(h, "gridType")
    if kind != "regular_ll":
        raise ValueError(f"expected a regular_ll grid, got {kind}")
    nj, ni = eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni")
    la1 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
    la2 = eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
    lo1 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
    lo2 = eccodes.codes_get(h, "longitudeOfLastGridPointInDegrees")
    if lo2 < lo1:
        lo2 += 360.0
    return Grid(lat=np.linspace(la1, la2, nj), lon=np.linspace(lo1, lo2, ni) % 360.0)


def messages(path: Path, keys: tuple[str, ...]) -> Iterator[tuple[dict, np.ndarray, Grid]]:
    """(meta, field, grid) for every message of `path`, in file order.

    meta maps each of `keys` to its native value (int or str), or None where the message lacks the
    key; field is the (nj, ni) float64 array with missing points NaN. Every message must share the
    first message's grid (one request, one grid)."""
    import eccodes

    grid = None
    with open(path, "rb") as fh:
        while (h := eccodes.codes_grib_new_from_file(fh)) is not None:
            try:
                g = _grid(h)
                if grid is None:
                    grid = g
                elif g.lat.size != grid.lat.size or g.lon.size != grid.lon.size:
                    raise ValueError(f"{path.name}: messages on different grids")
                meta = {k: eccodes.codes_get(h, k) if eccodes.codes_is_defined(h, k) else None for k in keys}
                v = eccodes.codes_get_values(h).astype("float64")
                v[v == eccodes.codes_get(h, "missingValue")] = np.nan
                field = v.reshape(grid.lat.size, grid.lon.size)
            finally:
                eccodes.codes_release(h)
            yield meta, field, grid
