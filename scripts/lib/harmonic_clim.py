"""A smooth daily climatology from twelve monthly means.

The mean and the first NHARM annual harmonics are fitted so that each curve's MONTH AVERAGES match the monthly means
(least squares on the twelve months; one design matrix serves every grid point), so a reanalysis' monthly-mean archive
gives a daily normal without its daily data. Time is the common-year calendar as a continuous coordinate: t = 0 at
1 Jan 00 UTC, 365 at the year's end, 29 Feb on the second half of 28 Feb's day (the common-year calendar of
noleap_doy, whose day d covers t = d - 1 .. d).

    coef = fit(month_means)              # (12, ...) -> (1 + 2 * NHARM, ...)
    normal = evaluate(coef, times_of(t)) # the normal at those times -> (len(t), ...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NHARM = 3
EDGES = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]).astype(float)  # month boundaries, days


def _omega(nh: int) -> np.ndarray:
    return 2.0 * np.pi * np.arange(1, nh + 1) / 365.0


def design(nh: int = NHARM) -> np.ndarray:
    """(12, 1 + 2 nh): each basis function (1, cos k w t, sin k w t) averaged over each month."""
    a, b, w = EDGES[:-1, None], EDGES[1:, None], _omega(nh)
    span = w * (b - a)
    cos_avg = (np.sin(w * b) - np.sin(w * a)) / span
    sin_avg = -(np.cos(w * b) - np.cos(w * a)) / span
    return np.hstack([np.ones((12, 1)), cos_avg, sin_avg])


def fit(month_means, nh: int = NHARM) -> np.ndarray:
    """Coefficients (1 + 2 nh, ...) of the curve whose month averages best match month_means (12, ...); NaN where any
    month is missing."""
    x = np.asarray(month_means, dtype="float64")
    flat = x.reshape(12, -1)
    coef = np.full((1 + 2 * nh, flat.shape[1]), np.nan)
    ok = np.isfinite(flat).all(axis=0)
    coef[:, ok] = np.linalg.lstsq(design(nh), flat[:, ok], rcond=None)[0]
    return coef.reshape((1 + 2 * nh, *x.shape[1:]))


def times_of(times) -> np.ndarray:
    """Common-year time t (days, fractional) of datetimes."""
    ts = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(times)))
    doy = ts.dayofyear.to_numpy().astype(float)
    leap = ts.is_leap_year
    feb29 = leap & (ts.month == 2) & (ts.day == 29)
    doy = np.where(leap & (ts.month > 2), doy - 1, doy)
    frac = (ts.hour.to_numpy() * 3600 + ts.minute.to_numpy() * 60) / 86400.0
    t = doy - 1 + frac
    return np.where(feb29, 58.5 + frac / 2, t)  # 29 Feb: the second half of 28 Feb's slot stretched over the day


def evaluate(coef, t) -> np.ndarray:
    """The curve at common-year times t (days): (len(t), ...)."""
    coef = np.asarray(coef)
    nh = (coef.shape[0] - 1) // 2
    w = _omega(nh)
    t = np.atleast_1d(np.asarray(t, dtype="float64"))
    basis = np.hstack([np.ones((t.size, 1)), np.cos(np.outer(t, w)), np.sin(np.outer(t, w))])
    return np.tensordot(basis, coef, axes=(1, 0))
