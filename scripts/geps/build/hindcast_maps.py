#!/usr/bin/env python3
"""Gridpoint skill and calibration of GEPS weekly-mean anomalies, from the
GEPS8 hindcast — the basis of the skill mask and the calibrated probability
maps.

For each field, every hindcast start of 2001-2020 (ensemble mean, 39 daily
leads) is de-drifted against the lead-matched GEPS8 climatology and re-based
to 1991-2020 exactly as the live maps are, coarsened to 2.5 deg, and averaged
into weeks 1-5. The observed weekly-mean anomaly on the same valid days comes
from an independent analysis on the same grid:

  t2m    NCEP/NCAR R1 2 m air temperature (daily, Gaussian grid, interpolated)
  pr     CPC unified gauge analysis (daily, 0.5 deg, LAND ONLY — over the
         ocean there is no truth, no skill number and no calibration)
  zg500  NCEP/NCAR R1 500 hPa height
  mslp   NCEP/NCAR R1 sea-level pressure

against its own 1991-2020 day-of-year climatology. Pairs are pooled by
calendar month (starts within +-45 days of the 15th) and reduced at every
gridpoint and week to the anomaly correlation r (the skill), the regression
obs = a*fcst + b, its residual spread s, and the observed spread — the same
recipe as the teleconnection indices, gridpoint by gridpoint. The threshold
for "above normal" is the climatological MEDIAN of the weekly-mean anomaly
for that day of year (zero for temperature to within noise, negative for
precipitation, whose weekly means are skewed).

Single pass with running sums, so memory is one field's daily truth (~0.5 GB)
plus one hindcast file. About 3 s per file; run two fields in parallel.

    python hindcast_maps.py --tags t2m,pr
    python hindcast_maps.py --tags zg500,mslp
    -> telecon/map_skill_{tag}.nc
"""

from __future__ import annotations

import argparse
import sys as _sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent

_sys.path[:0] = [str(HERE), str(HERE.parent)]
from telecon_hindcast import CLIM, LAT25, LON25, STORE, TC
from telecon_io import period_shift
from build_telecon_patterns import doy_clim
from refpack import noleap_doy, at_doy  # day-of-year key

NCEP, CPCP = TC / "ncep", TC / "cpc_precip"
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
Y0, Y1 = 2001, 2020
CY0, CY1 = 1991, 2020
WIN = 45
SCALE = {"t2m": 1.0, "pr": 86400.0, "zg500": 1.0, "mslp": 0.01}


def clim_wrapped25(tag):
    """Lead-dependent model climatology on the 2.5 deg grid, day-of-year axis wrapped for interpolation (coarsened once:
    subtracting on the 1 deg grid and coarsening the difference cost 30-40 s per file)."""
    c = xr.open_dataset(CLIM / f"geps8_clim_{tag}.nc")
    cv = c[next(v for v in c.data_vars if v != "n_starts")].load()
    cv = xr.concat([cv, cv.isel(X=0).assign_coords(X=360.0)], dim="X")
    cv = cv.interp(Y=LAT25, X=LON25).rename({"Y": "lat", "X": "lon"})
    return xr.concat(
        [cv.isel(doy=-1).assign_coords(doy=cv.doy[-1] - 366), cv, cv.isel(doy=0).assign_coords(doy=cv.doy[0] + 366)],
        dim="doy",
    )


def to25(a, nearest=False):
    a = a.rename({d: n for d, n in (("latitude", "lat"), ("longitude", "lon")) if d in a.dims})
    a = xr.concat([a, a.isel(lon=0).assign_coords(lon=float(a.lon[0]) + 360.0)], dim="lon")
    if nearest:
        return a.interp(lat=LAT25, lon=LON25, method="nearest")
    return a.interp(lat=LAT25, lon=LON25, kwargs={"fill_value": None})


def truth_daily(tag, years):
    """Daily truth on the 2.5 deg grid, (time, lat, lon) float32."""
    parts = []
    for y in years:
        if tag == "t2m":
            p = NCEP / f"air.2m.gauss.{y}.nc"
            if not p.exists():
                continue
            d = xr.open_dataset(p)
            a = d["air"].load().astype("float32")
            d.close()
            parts.append(to25(a))
        elif tag == "pr":
            p = CPCP / f"precip.{y}.nc"
            if not p.exists():
                continue
            d = xr.open_dataset(p)
            a = d["precip"].load().astype("float32")
            d.close()
            a = a.sortby("lat", ascending=False)
            a = a.coarsen(lat=5, lon=5, boundary="trim").mean()  # 2.5 deg cells, NaN = ocean
            parts.append(to25(a, nearest=True))
        else:
            var = "hgt" if tag == "zg500" else "slp"
            p = NCEP / f"{var}.{y}.nc"
            if not p.exists():
                continue
            d = xr.open_dataset(p)
            a = d[var].sel(level=500) if var == "hgt" else d[var]
            a = a.load().astype("float32")
            d.close()
            if var == "slp" and float(a.max()) > 2000:
                a = a / 100.0
            parts.append(a.assign_coords(lat=a.lat.values, lon=a.lon.values))
    a = xr.concat(parts, dim="time")
    _, ix = np.unique(a.time.values, return_index=True)
    return a.isel(time=np.sort(ix)).transpose("time", "lat", "lon")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="t2m,pr")
    ap.add_argument("--limit", type=int, default=0, help="only the first N hindcast files (smoke test)")
    a = ap.parse_args()
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        t0 = time.time()
        # ---- truth: daily anomaly 1991-2021, weekly running mean, doy median --
        tr = truth_daily(tag, range(CY0, Y1 + 2))
        base = tr.sel(time=slice(f"{CY0}-01-01", f"{CY1}-12-31"))
        clim = doy_clim(base)  # (doy, lat, lon)
        an = (tr.values - at_doy(clim.values, tr.time.values)).astype("float32")
        t_index = pd.DatetimeIndex(tr.time.values)
        pos = {d: i for i, d in enumerate(t_index.values.astype("datetime64[D]"))}
        cs = np.cumsum(np.nan_to_num(an), axis=0, dtype="float64")  # window means (float64: 11k-day sums)
        cnt = np.cumsum(np.isfinite(an), axis=0, dtype="int16")  # <= 11323 days fits int16
        nday = an.shape[0]
        del tr, an

        def week_mean(i0, i1):
            """mean of an[i0:i1] (i1 exclusive), NaN where fewer than 5 days."""
            top = cs[i1 - 1] - (cs[i0 - 1] if i0 > 0 else 0)  # noqa: B023  used in this iteration only; no default arg, so `cs = None` below frees the sums
            nn = cnt[i1 - 1] - (cnt[i0 - 1] if i0 > 0 else 0)  # noqa: B023  used in this iteration only; no default arg, so `cnt = None` below frees the sums
            with np.errstate(invalid="ignore", divide="ignore"):
                m = top / nn
            m[nn < 5] = np.nan
            return m

        # climatological median of the 7-day mean anomaly by day of year
        wk = np.stack([week_mean(i, i + 7) for i in range(nday - 7)])
        wdoy = noleap_doy(t_index[: wk.shape[0]] + pd.Timedelta(days=3))  # window centre, day-of-year key
        wyr = t_index[: wk.shape[0]].year.values
        med = np.full((366, LAT25.size, LON25.size), np.nan, "float32")
        for d in range(1, 366):
            dist = np.abs(((wdoy - d + 182.5) % 365) - 182.5)
            sel = (dist <= 15) & (wyr >= CY0) & (wyr <= CY1)
            med[d - 1] = np.nanmedian(wk[sel], axis=0)
        med[365] = med[0]  # slot 366 = 1 January (the wrap)
        del wk
        print(f"  {tag}: truth ready ({nday} days), {time.time() - t0:.0f} s", flush=True)

        # ---- hindcast pass with running sums per calendar month ---------------
        cw = clim_wrapped25(tag)
        shift = period_shift(tag) if tag != "pr" else None
        nW = len(WEEKS)

        def Z(nW=nW):
            return np.zeros((12, nW, LAT25.size, LON25.size), "float64")

        n_, sf, so, sff, soo, sfo = Z(), Z(), Z(), Z(), Z(), Z()
        mid_doy = np.array([pd.Timestamp(2001, m, 15).dayofyear for m in range(1, 13)])
        files = sorted(STORE.glob(f"geps8_{tag}_*.nc"))
        if a.limit:
            files = files[: a.limit]
        for k, f in enumerate(files, 1):
            ds = xr.open_dataset(f)
            v = ds[next(iter(ds.data_vars))]
            if "P" in v.dims:
                v = v.isel(P=0)
            v = v.load().astype("float32")
            ds.close()
            S = pd.DatetimeIndex(v.S.values)
            L = v.L.values
            day = np.ceil(L).astype(int)
            v25 = to25(v.rename({"Y": "lat", "X": "lon"})).values  # (S, L, 73, 144)
            del v
            for si, s0 in enumerate(S):
                if s0.year < Y0 or s0.year > Y1:
                    continue
                x0 = float(noleap_doy(s0))  # start's day-of-year key
                ref = cw.interp(doy=x0).sel(L=L, method="nearest").values
                anom = v25[si] - ref
                if shift is not None:
                    valid = s0 + pd.to_timedelta(day - 1, unit="D")  # L = 0.5 is the start date itself
                    anom = anom + at_doy(shift, valid)
                anom = anom * SCALE[tag]
                months = np.where(np.abs(((x0 - mid_doy + 182.5) % 365) - 182.5) <= WIN)[0]
                for wi, (w0, w1) in enumerate(WEEKS):
                    sel = (day >= w0) & (day <= w1)
                    fw = anom[sel].mean(0)
                    d0 = (s0 + pd.Timedelta(days=w0 - 1)).to_datetime64().astype("datetime64[D]")
                    i0 = pos.get(d0)
                    if i0 is None or i0 + 7 > nday:
                        continue
                    ow = week_mean(i0, i0 + 7)
                    good = np.isfinite(fw) & np.isfinite(ow)
                    fw = np.where(good, fw, 0.0)
                    ow = np.where(good, ow, 0.0)
                    for m in months:
                        n_[m, wi] += good
                        sf[m, wi] += fw
                        so[m, wi] += ow
                        sff[m, wi] += fw * fw
                        soo[m, wi] += ow * ow
                        sfo[m, wi] += fw * ow
            if k % 40 == 0 or k == len(files):
                print(f"  {tag}: {k}/{len(files)} files, {(time.time() - t0) / 60:.1f} min", flush=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            mf, mo = sf / n_, so / n_
            vf = sff / n_ - mf**2
            vo = soo / n_ - mo**2
            cov = sfo / n_ - mf * mo
            r = cov / np.sqrt(vf * vo)
            slope = cov / vf
            icpt = mo - slope * mf
            resid = np.sqrt(np.clip(vo - slope * cov, 0, None) * n_ / np.clip(n_ - 2, 1, None))
            osd = np.sqrt(vo)
        bad = n_ < 30
        vf = np.where(n_ < 30, np.nan, vf)
        for arr in (r, slope, icpt, resid, osd):
            arr[bad] = np.nan
        out = xr.Dataset(
            {
                "r": (("month", "week", "lat", "lon"), r.astype("float32")),
                "a": (("month", "week", "lat", "lon"), slope.astype("float32")),
                "b": (("month", "week", "lat", "lon"), icpt.astype("float32")),
                "s": (("month", "week", "lat", "lon"), resid.astype("float32")),
                "obs_sd": (("month", "week", "lat", "lon"), osd.astype("float32")),
                # the hindcast ensemble mean's own sd: prob_maps' standardised calibration
                "fsd": (("month", "week", "lat", "lon"), np.sqrt(np.clip(vf, 0, None)).astype("float32")),
                "n": (("month", "week", "lat", "lon"), n_.astype("int32")),
                "median_anom": (("doy", "lat", "lon"), med),
            },
            coords={
                "month": np.arange(1, 13),
                "week": np.arange(1, nW + 1),
                "lat": LAT25,
                "lon": LON25,
                "doy": np.arange(1, 367),
            },
        )
        out.attrs.update(
            tag=tag,
            units={"t2m": "K", "pr": "mm/day", "zg500": "m", "mslp": "hPa"}[tag],
            truth={
                "t2m": "NCEP/NCAR R1 2 m air",
                "pr": "CPC unified gauge (land only)",
                "zg500": "NCEP/NCAR R1 500 hPa height",
                "mslp": "NCEP/NCAR R1 SLP",
            }[tag],
            window_days=WIN,
            base=f"{CY0}-{CY1}",
            hindcast=f"{Y0}-{Y1}",
        )
        out.to_netcdf(
            TC / (f"map_skill_{tag}.nc" if not a.limit else f"map_skill_{tag}_test.nc"),
            encoding={k: {"zlib": True, "complevel": 4} for k in out.data_vars},
        )
        print(
            f"  {tag}: map_skill_{tag}.nc  median r (Sep, 30-70N land/all) by week: "
            + " ".join(f"{np.nanmedian(r[8, w, 8:25]):.2f}" for w in range(nW))
            + f"  ({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )
        cs = cnt = None  # free the running sums before the next field
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
