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
                144-point 2.5 deg band grid, current to yesterday.

So: ERA5 wherever it exists, GMGSI for the last few days only, offset-corrected
onto the ERA5 scale per longitude. That splice is measured, not assumed — over
their 220-day overlap at set-up the proxy ran -8.3 W/m2 cold in the mean, correlated
r = 0.80 with ERA5 in the anomaly, and carried essentially the same anomaly
spread (17.5 vs 17.0 W/m2); the offset applied live is the mean over the last
CAL_DAYS overlapping days. The per-longitude offset is the part worth removing;
the residual scatter is the honest cost of a real-time tail, and it applies to
a handful of days at the end of a 120-day window.

Only the offset is corrected, deliberately. Regressing the proxy onto ERA5
(slope 0.78 in the MSE-optimal direction) would shrink real anomalies as well as
noise, and a damped OLR channel biases RMM amplitude low — the opposite of what
a phase diagram should do.

The anomaly is then taken in the frame of the Wheeler & Hendon reference
climatology, clim_olr in climatology.nc (NOAA interpolated OLR, 1979-2001, three
harmonics), which ERA5 does not share: its top-of-atmosphere longwave runs about
13 W/m2 above NOAA's in the band mean. The shift is fixed: ERA5's own
climatology over the same 1979-2001, in the same band and fitted the same way
(era5_olr_clim.nc, from scripts/sst/build_olr_baseline.py via olr-baseline.yml),
minus clim_olr, by longitude and day of year (frame_offset).

    python src/olr_channel.py            # report the splice, the frame and the 120-day map
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from doycal import sel_doy

REF = Path(__file__).resolve().parents[1] / "data" / "reference"
ERA5 = REF / "era5_olr_band.nc"
FRAME = REF / "era5_olr_clim.nc"  # ERA5 climatology of the band over the reference period (frame_offset)
GMGSI = Path(__file__).resolve().parents[2] / "sst" / "metar" / "olr_rt.nc"
CAL_DAYS = 90  # recent overlap used for the per-longitude offset
NINO4 = (160.0, 210.0)  # 160°E-150°W, for the report


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
    print(
        f"  splice: ERA5 to {et[-1]:%Y-%m-%d}, GMGSI {len(tail)} d to {tail[-1]:%Y-%m-%d}; "
        f"offset {off.mean():+.1f} W/m2 (range {off.min():+.1f}..{off.max():+.1f})"
    )
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
    r = np.sqrt((a**2).mean(1))
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826
    bad = r > med + k * mad
    if bad.any():
        good = np.where(~bad)[0]
        for i in np.where(bad)[0]:
            j = good[np.argsort(np.abs(good - i))[:2]]
            a[i] = a[j].mean(0)
            print(
                f"  QC: {t[i]:%Y-%m-%d} rejected (RMS {r[i]:.1f} W/m2 vs "
                f"threshold {med + k * mad:.1f}), filled from "
                f"{t[j[0]]:%m-%d}/{t[j[1]]:%m-%d}"
            )
    return a


def frame_offset(clim: xr.Dataset) -> xr.DataArray:
    """ERA5 minus NOAA interpolated OLR over the reference's base period, (dayofyear, longitude): the fixed shift that
    puts ERA5 OLR into the frame of clim_olr, looked up like it (doycal.sel_doy).

    Both climatologies are a mean and three annual harmonics fitted to every day of the same years in the same band,
    so their difference is the two sources' systematic difference alone, its seasonal cycle included. It holds no
    weather and no climate anomaly: an El Niño's months of suppressed or enhanced convection stay in the anomaly."""
    if not FRAME.exists():
        raise SystemExit(f"{FRAME.name} missing: build it with olr-baseline.yml (rmm_frame=true)")
    with xr.open_dataset(FRAME) as f:
        era5 = f["clim_olr"].transpose("dayofyear", "longitude").load()
        attrs = dict(f.attrs)
    ref = clim["clim_olr"].transpose("dayofyear", "longitude")
    for k in ("base_period", "n_harmonics"):
        if str(attrs.get(k)) != str(clim.attrs.get(k)):
            raise SystemExit(f"{FRAME.name}: {k} {attrs.get(k)!r}, climatology.nc: {clim.attrs.get(k)!r}")
    if not (np.array_equal(era5["dayofyear"], ref["dayofyear"]) and np.allclose(era5["longitude"], ref["longitude"])):
        raise SystemExit(f"{FRAME.name} is not on climatology.nc's day-of-year and longitude grid")
    return ref.copy(data=era5.values - ref.values)


def anomaly_and_map120(clim: xr.Dataset):
    """OLR anomaly vs the WH04 reference clim, and its trailing 120-day mean map.

    clim_olr is NOAA interpolated OLR, and ERA5's top-of-atmosphere longwave runs
    about 13 W/m2 above it in the band mean, so a raw difference gives an
    "anomaly" that is mostly source offset (the first version of this produced a
    120-day map of +0.8 to +27 W/m2, all of it bias, which subtracted into a
    forecast is a spurious basin-wide OLR shift). frame_offset takes it out. It is
    a climatological offset on purpose, by day of year: an ERA5-minus-NOAA mean
    over the recent record would also take out that record's mean anomaly, a few
    W/m2 at some longitudes over a multi-year El Nino or La Nina, and would miss
    the offset's seasonal cycle (up to 11.6 W/m2 at a longitude). A constant part
    cancels on the observed track, which subtracts the same map from the same
    anomalies, and the GEPS forecast is filtered with GEPS's own OLR
    (scripts/geps/mjo_geps.py forecast_olr_map), where it cancels as well.

    Returns (times, anom[time, lon], mean120[lon])."""
    t, olr, _ = band_series()
    off = frame_offset(clim)
    a = olr - sel_doy(clim["clim_olr"], t) - sel_doy(off, t)  # leap-aware day-of-year lookup
    ann = off.mean("dayofyear")
    n4 = (off.longitude >= NINO4[0]) & (off.longitude <= NINO4[1])
    print(
        f"  ERA5-to-NOAA frame ({clim.attrs['base_period']} climatologies): mean {float(ann.mean()):+.1f}, "
        f"by longitude {float(ann.min()):+.1f}..{float(ann.max()):+.1f}, Nino-4 {float(ann[n4].mean()):+.1f} W/m2"
    )
    a = _qc(t, a)
    df = pd.DataFrame(a, index=t).reindex(pd.date_range(t[0], t[-1], freq="D"))
    roll = df.rolling(window=120, min_periods=110).mean().reindex(t)
    m = roll.values
    if np.isnan(m[-1]).any():
        raise SystemExit("not enough history for a 120-day OLR mean — widen the sweep")
    return t, a, m[-1]


def main() -> int:
    clim = xr.open_dataset(REF / "climatology.nc")
    t, _a, m120 = anomaly_and_map120(clim)
    lon = clim.longitude.values
    n4 = (lon >= NINO4[0]) & (lon <= NINO4[1])
    print(
        f"120-day mean map to {t[-1]:%Y-%m-%d}: {m120.min():+.1f}..{m120.max():+.1f} W/m2, "
        f"Nino-4 {m120[n4].mean():+.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
