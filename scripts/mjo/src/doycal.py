"""Day-of-year keys that line up across leap and common years.

Raw day-of-year puts 1 March at 60 in common years and 61 in leap years, so a climatology keyed or phased by it
is one day late for every leap-year date after 28 February.

doy366(dates)       0..365 on a 366-slot calendar: 29 Feb = 59 and 1 March = 60 in every year (the semantics of
                    quantmet.degdays.doy366; kept local because several workflows that import these modules do not
                    install quantmet).
noleap_doy(dates)   day of the 365-day common-year calendar, 1..365, with 29 Feb at 59.5 (midway between 28 Feb = 59
                    and 1 March = 60). This is the phase coordinate of the harmonic climatologies and the key of the
                    reference files with a 'dayofyear' 1..366 coordinate, whose slots 1..365 are common-year days.
sel_doy(da, dates)  a 'dayofyear'-keyed climatology at those dates: slot x for a common-year day, the mean of slots
                    59 and 60 (28 Feb, 1 March) for 29 Feb.
"""

from __future__ import annotations

import numpy as np


def doy366(dates) -> np.ndarray:
    """0..365 on a 366-slot calendar: 29 Feb = 59, 1 March = 60 and 31 Dec = 365 in every year."""
    d = np.asarray(dates, dtype="datetime64[D]")  # Timestamps, datetimes, datetime64, DatetimeIndex
    y = d.astype("datetime64[Y]")
    doy = (d - y.astype("datetime64[D]")).astype(int)  # 0-based
    yr = y.astype(int) + 1970
    leap = (yr % 4 == 0) & ((yr % 100 != 0) | (yr % 400 == 0))
    return np.where(~leap & (doy >= 59), doy + 1, doy)


def noleap_doy(dates) -> np.ndarray:
    """Common-year day of year (float, 1..365); 29 Feb = 59.5."""
    s = doy366(dates)
    return np.where(s < 59, s + 1.0, np.where(s == 59, 59.5, s.astype(float)))


def sel_doy(da, dates, dim: str = "dayofyear") -> np.ndarray:
    """Values of `da` (a climatology with a `dim` coordinate 1..366 keyed by common-year day) at `dates`,
    stacked along a new leading axis in place of `dim`."""
    x = np.atleast_1d(noleap_doy(dates))
    lo = da.sel({dim: np.floor(x).astype(int)}).transpose(dim, ...).values
    hi = da.sel({dim: np.ceil(x).astype(int)}).transpose(dim, ...).values
    return 0.5 * (lo + hi)
