"""How unusual the forecast wave driving is: each ensemble's 100 hPa eddy heat flux against every ERA5 value at the same
time of year.

The reference is reference/vt100_era5.nc (build_vt100_clim.py's ERA5 record, 1990-11 to 2026-10): the zonal-mean
[v'T'] at 100 hPa, 45-75 deg, all waves and waves 1 and 2, at 00 and 12 UTC. A forecast value on a date is ranked among
the ERA5 values within WINDOW days of that calendar date in every year (a 366-slot calendar), and set against their
maximum, the record for that time of year:
  daily   the single fields (the forecast's daily steps against ERA5's 00 and 12 UTC values)
  ten     the trailing TEN-day mean of the total flux, the SSW warning meter's measure (the members' paths through the
          analysed days, heatflux100.member_paths, against the ten-day means of ERA5's daily series)
For each, over days 1-15: the ensemble mean's peak, each member's own peak (each ranked at its own date), the share of
members whose peak is at or above the 99th percentile and above the record, and the control run. Descriptive ranks of
a forecast against a 36-year record, not a test. The heat-flux scripts (heatflux100.py, heatflux100_ext.py) add the
block to their JSON ("records"); the stratosphere page draws the table from the four.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from quantmet.degdays import doy366

REF = Path(__file__).resolve().parent / "reference" / "vt100_era5.nc"
WINDOW = 15  # days either side of the calendar date
TEN = 10  # the warning meter's window (heatflux100_warning.json window_days)
KEYS = ("tot", "k1", "k2")


class Era5:
    """The ERA5 samples: the 00/12 UTC values per hemisphere and key, and the ten-day means of the daily total."""

    def __init__(self, path: Path = REF):
        import xarray as xr

        with xr.open_dataset(path) as d:
            t = pd.DatetimeIndex(d.time.values)
            hours = pd.to_timedelta(np.tile(d.hour.values, len(t)), unit="h")
            self.inst, self.ten = {}, {}
            for hemi in ("nh", "sh"):
                for k in KEYS:
                    s = pd.Series(d[f"{hemi}_{k}"].values.ravel(), index=t.repeat(d.hour.size) + hours)
                    self.inst[hemi, k] = s.dropna()
                daily = d[f"{hemi}_tot"].to_pandas().mean(axis=1)  # NaN where both hours are missing
                self.ten[hemi] = daily.rolling(TEN, min_periods=TEN).mean().dropna()
        self.period = f"{t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d}"

    def rank(self, s: pd.Series, value: float, when: pd.Timestamp) -> tuple[float, float, pd.Timestamp]:
        """(percentile of `value` among the samples of s within WINDOW days of `when`'s calendar date, their maximum,
        its date)."""
        d = np.abs(doy366(s.index) - doy366(pd.DatetimeIndex([when]))[0])
        w = s[np.minimum(d, 366 - d) <= WINDOW]
        return 100.0 * float((w.values < value).mean()), float(w.max()), w.idxmax()


def _peaks(x: np.ndarray, days: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Each row's maximum over its columns and the date of it."""
    i = np.nanargmax(x, axis=1)
    return x[np.arange(len(x)), i], days[i]


def _summary(ref: Era5, s: pd.Series, x: np.ndarray, days, ctrl: np.ndarray | None, nd: int = 1) -> dict:
    """The block of one measure: x (member, day) over the forecast days."""
    ens = np.nanmean(x, 0)
    k = int(np.nanargmax(ens))
    pct, rec, rec_day = ref.rank(s, ens[k], days[k])
    pv, pdays = _peaks(x, days)
    ranks = [ref.rank(s, v, d) for v, d in zip(pv, pdays)]
    mp = np.array([r[0] for r in ranks])
    out = {
        "ens_peak": round(float(ens[k]), nd),
        "ens_date": f"{days[k]:%Y-%m-%d}",
        "ens_pct": round(pct, 1),
        "record": round(rec, nd),
        "record_date": f"{rec_day:%Y-%m-%d}",
        "member_median": round(float(np.median(pv)), nd),
        "member_median_pct": round(float(np.median(mp)), 1),
        "member_max": round(float(pv.max()), nd),
        "member_max_pct": round(float(mp[int(np.argmax(pv))]), 1),
        "share_p99": round(float((mp >= 99.0).mean()), 3),
        "share_record": round(float(np.mean([v > r[1] for v, r in zip(pv, ranks)])), 3),
    }
    if ctrl is not None:
        c = int(np.nanargmax(ctrl))
        cp, crec, _ = ref.rank(s, ctrl[c], days[c])
        out["control"] = {"peak": round(float(ctrl[c]), nd), "date": f"{days[c]:%Y-%m-%d}", "pct": round(cp, 1),
                          "above_record": bool(ctrl[c] > crec)}  # fmt: skip
    return out


def block(ref: Era5, hemi: str, valid, mem: dict, ctrl: dict | None, obs: pd.DataFrame | None, paths, init) -> dict:
    """The JSON block of one hemisphere: mem/ctrl {tot, k1, k2: (member, lead) / (lead,)} at the valid times (the
    model's error removed, as on the figure); obs the analysed daily series (heatflux100.history_frame); paths the
    members' daily total through the analysed days (member_paths) for the ten-day mean."""
    valid = pd.DatetimeIndex(valid)
    fc = (valid > pd.Timestamp(init)) & (valid <= pd.Timestamp(init) + pd.Timedelta(days=15))
    days = valid[fc]
    out = {"period": ref.period, "window_days": WINDOW, "measures": {}}
    if obs is not None and len(obs) >= TEN:
        day = obs.index[-1]
        ten = float(obs["tot"].iloc[-TEN:].mean())
        out["analysis"] = {"date": f"{day:%Y-%m-%d}", "tot": round(float(obs["tot"].iloc[-1]), 1),
                           "tot_pct": round(ref.rank(ref.inst[hemi, "tot"], obs["tot"].iloc[-1], day)[0], 1),
                           "ten": round(ten, 1), "ten_pct": round(ref.rank(ref.ten[hemi], ten, day)[0], 1)}  # fmt: skip
    for k in KEYS:
        c = None if ctrl is None else np.asarray(ctrl[k])[fc]
        out["measures"][k] = _summary(ref, ref.inst[hemi, k], np.asarray(mem[k])[:, fc], days, c)
    if paths is not None:
        roll = paths.rolling(TEN, min_periods=TEN).mean()
        roll = roll[(roll.index > pd.Timestamp(init)) & (roll.index <= pd.Timestamp(init) + pd.Timedelta(days=15))]
        if len(roll):
            out["measures"]["ten"] = _summary(ref, ref.ten[hemi], roll.to_numpy().T, roll.index, None)
    return out
