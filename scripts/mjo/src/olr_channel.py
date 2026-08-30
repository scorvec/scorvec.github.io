#!/usr/bin/env python3
"""The third RMM channel: real-time equatorial-band OLR, built in house.

The site's RMM has been wind-only since NOAA's interpolated-OLR product ended
in 2022, because the OLR channel of Wheeler & Hendon (2004) needs a trailing
120-day-mean map to strip interannual variability and no public OLR feed is
current. Two in-house sources close that gap, and they are complementary:

  ERA5 (ARCO)   TRUE top-of-atmosphere OLR, but roughly a week behind real time
                (2026-08-23 when this was written). fetch_arco_olr_band.py.
  GMGSI proxy   scripts/sst/build_olr_realtime.py — GMGSI longwave-IR brightness
                temperature through the Ohring-Gruber flux relation, on the same
                144-point 2.5 deg band grid, current to TODAY.

So: ERA5 wherever it exists, GMGSI for the last few days only, offset-corrected
onto the ERA5 scale per longitude. That splice is measured, not assumed — over
their 220-day overlap the proxy runs -8.3 W/m2 cold in the mean, correlates
r = 0.80 with ERA5 in the anomaly, and carries essentially the same anomaly
spread (17.5 vs 17.0 W/m2). The per-longitude offset is the part worth removing;
the residual scatter is the honest cost of a real-time tail, and it applies to
a handful of days at the end of a 120-day window.

Only the offset is corrected, deliberately. Regressing the proxy onto ERA5
(slope 0.78 in the MSE-optimal direction) would shrink real anomalies as well as
noise, and a damped OLR channel biases RMM amplitude low — the opposite of what
a phase diagram should do.

    python src/olr_channel.py            # report the splice and cache the map
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REF = Path(__file__).resolve().parents[1] / "data" / "reference"
ERA5 = REF / "era5_olr_band.nc"
GMGSI = Path(__file__).resolve().parents[2] / "sst" / "metar" / "olr_rt.nc"
OUT = REF / "olr_map120.nc"
CAL_DAYS = 90        # recent overlap used for the per-longitude offset


def band_series() -> tuple[pd.DatetimeIndex, np.ndarray, pd.Timestamp]:
    """Daily 15S-15N band-mean OLR (W/m2, positive up) on the 2.5 deg grid.

    Returns (times, olr[time, lon], last_era5_day)."""
    e = xr.open_dataset(ERA5)
    et = pd.DatetimeIndex(e.time.values).normalize()
    E = e.olr.values
    if not GMGSI.exists():
        return et, E, et[-1]

    g = xr.open_dataset(GMGSI)
    gt = pd.DatetimeIndex(g.time.values).normalize()
    G = g.olr_15.values

    # per-longitude offset from the recent overlap, applied to the tail only
    both = et.intersection(gt)
    recent = both[-CAL_DAYS:]
    off = (E[et.get_indexer(recent)] - G[gt.get_indexer(recent)]).mean(0)

    tail = gt[gt > et[-1]]
    if len(tail) == 0:
        return et, E, et[-1]
    T = G[gt.get_indexer(tail)] + off
    print(f"  splice: ERA5 to {et[-1]:%Y-%m-%d}, GMGSI {len(tail)} d to {tail[-1]:%Y-%m-%d}; "
          f"offset {off.mean():+.1f} W/m2 (range {off.min():+.1f}..{off.max():+.1f})")
    return et.append(tail), np.vstack([E, T]), et[-1]


def _qc(t: pd.DatetimeIndex, a: np.ndarray, k: float = 5.0) -> np.ndarray:
    """Replace whole days whose anomaly field is an outlier, by interpolation.

    The GMGSI mosaic occasionally publishes a day with missing or duplicated
    hourly passes, and the resulting daily mean is not slightly wrong but wildly
    wrong: 2026-08-27 came through at 35.5 W/m2 RMS (2.3 sigma) between
    neighbours at 16.0 and 18.4, with day-to-day changes of 38 and 41 W/m2
    against a normal 13. Undetected, a bad day at the END of the record is the
    worst case — it becomes the observed RMM's latest point and throws the whole
    analysis-to-forecast junction.

    The gate is on the daily RMS, with a median/MAD threshold so the outliers
    cannot inflate their own criterion."""
    r = np.sqrt((a ** 2).mean(1))
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826
    bad = r > med + k * mad
    if bad.any():
        good = np.where(~bad)[0]
        for i in np.where(bad)[0]:
            j = good[np.argsort(np.abs(good - i))[:2]]
            a[i] = a[j].mean(0)
            print(f"  QC: {t[i]:%Y-%m-%d} rejected (RMS {r[i]:.1f} W/m2 vs "
                  f"threshold {med + k * mad:.1f}), filled from "
                  f"{t[j[0]]:%m-%d}/{t[j[1]]:%m-%d}")
    return a


def anomaly_and_map120(clim: xr.Dataset):
    """OLR anomaly vs the WH04 reference clim, and its trailing 120-day mean map.

    One correction is essential here. clim_olr comes from NOAA interpolated OLR;
    ERA5's top-of-atmosphere longwave sits about 13 W/m2 above it in the mean, so
    a raw difference gives an "anomaly" that is almost entirely source offset —
    the first version of this produced a 120-day map running +0.8 to +27 W/m2,
    all of it bias. Subtracting that into a forecast would inject a spurious
    basin-wide OLR shift. So a per-longitude offset, measured over at least a full
    annual cycle (so it cannot absorb a seasonal mismatch), is removed first.

    Returns (times, anom[time, lon], mean120[lon])."""
    t, olr, _ = band_series()
    doy = np.minimum(t.dayofyear.values, 366)
    a = olr - clim["clim_olr"].sel(dayofyear=doy).values
    span = (t[-1] - t[0]).days
    if span < 365:
        raise SystemExit(f"only {span} d of OLR: need >=1 year to measure the "
                         "ERA5-vs-NOAA offset (widen fetch_arco_olr_band.py)")
    whole = t >= t[-1] - pd.Timedelta(days=365 * (span // 365))   # whole years only
    off = a[whole].mean(0)
    a = a - off
    print(f"  ERA5-vs-NOAA offset removed over {int(whole.sum())} d "
          f"({span // 365} yr): mean {off.mean():+.1f}, "
          f"range {off.min():+.1f}..{off.max():+.1f} W/m2")
    a = _qc(t, a)
    df = pd.DataFrame(a, index=t).reindex(pd.date_range(t[0], t[-1], freq="D"))
    roll = df.rolling(window=120, min_periods=110).mean().reindex(t)
    m = roll.values
    if np.isnan(m[-1]).any():
        raise SystemExit("not enough history for a 120-day OLR mean — widen the sweep")
    return t, a, m[-1]


def main() -> int:
    clim = xr.open_dataset(REF / "climatology.nc")
    t, a, m120 = anomaly_and_map120(clim)
    xr.Dataset({"olr": ("longitude", m120),
                "anom": (("time", "longitude"), a)},
               coords={"time": t, "longitude": clim.longitude.values},
               attrs={"title": "Trailing 120-day-mean equatorial OLR anomaly (WH04 filter map)",
                      "window_end": f"{t[-1]:%Y-%m-%d}",
                      "sources": "ERA5 ARCO (true OLR) spliced with offset-corrected GMGSI proxy",
                      "note": ("Companion to wind_map120.nc. Subtract from an OLR anomaly "
                               "before the RMM projection to remove interannual variability."),
                      "units": "W m-2"}).to_netcdf(OUT)
    print(f"wrote {OUT.name}: window ends {t[-1]:%Y-%m-%d}, "
          f"120-d mean map {m120.min():+.1f}..{m120.max():+.1f} W/m2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
