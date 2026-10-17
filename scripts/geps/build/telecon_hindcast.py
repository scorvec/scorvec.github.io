#!/usr/bin/env python3
"""Score the teleconnection indices on the GEPS8 hindcast, lead by lead, out of sample.

The real-time indices (telecon_geps.py) say what the 21 members forecast. This says how much that is worth: the same
projectors applied to every hindcast start of 2001-2020 (four a week, 39 daily leads, ensemble mean), against the
observed index on the valid date from NCEP/NCAR R1 - the reanalysis the patterns are defined on.

De-drifting is the live product's, [F - M(start doy, L)] - (E_9120 - E_0120)(valid doy), with one change that keeps
the scored start out of its own forecast: LEAVE-ONE-YEAR-OUT. A live forecast is de-drifted with a GEPS8 climatology M
and an ERA5 2001-2020 normal E_0120 that never saw its own year. Scored against the full ones, a hindcast start carries
a twentieth of its own season in both: its model forecasts inside M, and the OBSERVED weather around its valid dates
inside E_0120 (each 5-day centre averages the 15 days around it in every year). E_0120 is the same for every year, but
across the +-45 days of starts a correlation pools, its day-to-day shape holds 1/20 of each year's observed anomalies,
which the forecast then appears to predict: it lifted the weekly correlations of weeks 3-5 by 0.035-0.055 and made
about a quarter of the week 3-4 cells pass the skill gate that do not pass out of sample. So every start is scored
against

  M without its season       every start in M's pools (starts within +-15 days of a 5-day centre) that lies within
                             SEASON days of the scored start: (N M - their sum) / (N - n), at the two centres the live
                             read interpolates between. The pools are mirrored as build_clim.py fills them - the
                             common-year day of year, each start once (build_clim.entries) - and the file must say so;
  E_0120 without its season  the days of the ERA5 normal's windows (+-7 days of a 5-day centre, common-year day) within
                             SEASON days of the valid date, removed the same way. 500 hPa height: the 1.5 deg daily
                             store the normal is built from, every day. Sea-level pressure and 2 m temperature: their
                             normal is the 1 deg ARCO one averaged every fifth day of each year (1, 6, 11 ... January);
                             the same days are taken from the 1.5 deg daily store (a daily mean of the same four
                             synoptic hours). On that 5-day lattice a common-year-day window holds exactly the days the
                             raw-day window held (a leap year's days after February move one key down, which never
                             carries a lattice day across a window edge: lattice distances 0, 5, 10 become 1, 4-6, 9-11
                             against the +-7 cut), so that normal is already on the common-year calendar.

The re-basing target E_9120 is the definition of the anomaly (the observed index is taken against the 1991-2020 NCEP
normal, which holds the scored year too), not a fitted part of the forecast, and stays as it is.

anomalies() is that arithmetic for any field, shared by the gridpoint skill maps (hindcast_maps.py) and the regime hit
rates (scripts/regimes/build_regime_hindcast.py). It needs the reference set being built in GEPS_REF (era5_shift.nc
and telecon/patterns.nc: make_reference.py --only shift,telecon) and the rebuilt climatology files in the store.

The output is the raw paired series, not summary skill: telecon_geps.py pools the starts within +-45 days of the
current init day-of-year and, per index and week, fits obs = a * fcst + b on the weekly means, giving the anomaly
correlation (skill), the amplitude factor a and the residual spread s, and tests that calibration on each year with
the others refitted (its cross-validated reliability).

    python telecon_hindcast.py --workers 4
    -> telecon/hindcast_indices.nc   fcst[index, S, L], obs[index, S, L]
    python telecon_hindcast.py --tags mslp --merge    # only the slp indices, keep the rest

Memory: each worker holds a monthly file (18 x 39 x 181 x 360) and its 2.5 deg copy, ~0.6 GB as float32; the scoring
process holds the 2.5 deg climatology (0.25 GB), the ERA5 leave-out fields (0.3 GB) and three months of starts.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import sys as _sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent

_sys.path[:0] = [str(HERE), str(HERE.parent)]
from paths import LOCAL_STORE as _LS  # offline-build inputs/outputs
from refpack import noleap_doy, at_doy  # the climatologies' read key
from telecon_io import period_shift
import build_clim  # the climatology's pool rules (calendar, each start once)
import build_era5_clim as era5  # the ERA5 normal's store and units

CLIM = _LS / "climatology"
TC = _LS / "telecon"
LAT25 = np.arange(90.0, -90.1, -2.5)
LON25 = np.arange(0.0, 360.0, 2.5)
Y0, Y1 = 2001, 2020
SEASON = 60  # days: a pooled start or day this close to the scored one is its own season (other years are ~365 away)
CLIM_HALF, ERA5_HALF = 15, 7  # pool half-widths of the GEPS8 climatology and the ERA5 normal, days of year
# every n-th day of each year in the ERA5 2001-2020 normal of each re-based field, and the file it is in (docstring)
ERA5_SAMPLE = {"zg500": 1, "mslp": 5, "t2m": 5}
ERA5_NORMAL = {
    "zg500": "era5_clim_2001-2020.nc",
    "mslp": "era5_clim_sfc_2001-2020.nc",
    "t2m": "era5_clim_sfc_2001-2020.nc",
}


@functools.cache
def patterns():
    import telecon_geps as t

    return t.Patterns()


@functools.cache
def ncep_normal(field):
    """The NCEP 1991-2020 day-of-year normal of z500 or slp the patterns were built against (the full pattern file of
    the offline store: the published reference drops these)."""
    return xr.open_dataset(TC / "patterns.nc")[f"clim_doy_{field}"].values


def pooled(doy, centre, half):
    d = np.abs(np.asarray(doy) - centre)
    return np.minimum(d, 365 - d) <= half


def bracket(key, centres):
    """Indices of the two centres around a read key (refpack.noleap_doy) on a 5-day axis and the upper one's weight -
    the linear day-of-year interpolation of the live reads; the axis wraps after its last centre."""
    k0 = int((key - centres[0]) // 5)
    k1 = (k0 + 1) % len(centres)
    return k0, k1, (key - centres[k0]) / 5.0


class ModelClim:
    """The lead-dependent GEPS8 climatology on the 2.5 deg grid with the size of each pool, so that a season can be
    taken out of it: centre k is the mean of the N_k starts whose common-year day is within CLIM_HALF of it."""

    def __init__(self, tag):
        c = xr.open_dataset(CLIM / f"geps8_clim_{tag}.nc")
        if (c.attrs.get("doy_key"), c.attrs.get("starts")) != (build_clim.CALENDAR, build_clim.STARTS):
            raise SystemExit(
                f"geps8_clim_{tag}.nc predates the common-year, each-start-once pools: rerun build_clim.py"
            )
        cv = c[next(v for v in c.data_vars if v != "n_starts")].squeeze(drop=True).load()
        cv = xr.concat([cv, cv.isel(X=0).assign_coords(X=360.0)], dim="X")
        cv = cv.interp(Y=LAT25, X=LON25).transpose("doy", "L", "Y", "X")
        self.C = cv.values.astype("float64")  # (74 centres, L, 73, 144); centre 366 is 1 January again
        self.L = cv.L.values
        self.centres = cv.doy.values.astype("float64")
        self.N = c["n_starts"].values.astype("float64")
        self.seen = np.zeros_like(self.N)  # starts met while scoring: must equal N
        c.close()

    def at(self, s0, mates):
        """(in-sample, leave-one-year-out) climatology for a start: mates = [(day-of-year key, field)] of the starts
        within SEASON days of it."""
        k0, k1, w = bracket(float(noleap_doy(s0)), self.centres)
        full, out = 0.0, 0.0
        for k, wt in ((k0, 1.0 - w), (k1, w)):
            own = [f for d, f in mates if pooled(d, self.centres[k], CLIM_HALF)]
            full = full + wt * self.C[k]
            out = out + wt * (self.N[k] * self.C[k] - sum(own, 0.0)) / (self.N[k] - len(own))
        return full, out


def season_left_out(days, X, keys, dates):
    """A day-of-year normal built as the mean of the daily fields X (n, ...) on `days` within ERA5_HALF of 73 5-day
    centres (each day's pool key in `keys`), read at every one of `dates` the way the live re-basing reads it (linear
    in the noleap key between the two centres around it, the year wrapped) twice - without the pool days within SEASON
    days of the date, and whole - and differenced. -> (that difference (len(dates), ...) float32, the normal)"""
    days = np.asarray(days, dtype="datetime64[D]")
    centres = np.arange(1.0, 366.0, 5.0)
    pools = [np.where(pooled(keys, c, ERA5_HALF))[0] for c in centres]
    E = np.stack([X[p].mean(0) for p in pools])
    out = np.zeros((len(dates), *X.shape[1:]), "float32")
    for i, v in enumerate(np.asarray(dates, dtype="datetime64[D]")):
        k0, k1, w = bracket(float(noleap_doy(v)), centres)
        for k, wt in ((k0, 1.0 - w), (k1, w)):
            own = pools[k][np.abs((days[pools[k]] - v).astype(int)) <= SEASON]
            if own.size:
                out[i] += wt * (own.size * E[k] - X[own].sum(0)) / (pools[k].size - own.size)
    return out, E


def era5_leave_out(tag, dates):
    """E_0120 without the valid date's own season, minus E_0120, on the 2.5 deg grid, for every date in `dates`
    (season_left_out), from the store and days the normal in use was built from: common-year-day windows, 2001-2020;
    checked against that normal."""
    var, mul, _ = era5.VARS[tag]
    step = ERA5_SAMPLE[tag]
    days, fields = [], []
    for y in range(Y0, Y1 + 1):
        d = xr.open_dataset(era5.STORE / var / f"{var}_{y}.nc")
        v = d[var].transpose("time", "latitude", "longitude")  # the store's own order varies: by name
        t = pd.DatetimeIndex(v.time.values)
        keep = ((t - pd.Timestamp(y, 1, 1)).days % step) == 0
        v = v.isel(time=np.where(keep)[0])
        v = xr.concat([v, v.isel(longitude=0).assign_coords(longitude=360.0)], dim="longitude")
        fields.append(v.interp(latitude=LAT25, longitude=LON25).values.astype("float64") * mul)
        days.append(t[keep])
        d.close()
    days = pd.DatetimeIndex(np.concatenate(days))
    out, E = season_left_out(days, np.concatenate(fields), noleap_doy(days), dates)
    ref = xr.open_dataset(CLIM / ERA5_NORMAL[tag])[tag]
    ref = ref.sortby("lat", ascending=False)
    ref = xr.concat([ref, ref.isel(lon=0).assign_coords(lon=360.0)], dim="lon").interp(lat=LAT25, lon=LON25)
    err = E - ref.values
    print(
        f"  ERA5 2001-2020 {tag} normal rebuilt from {len(days)} store days: vs the normal in use "
        f"rms {np.sqrt(np.nanmean(err**2)):.3g}, max |d| {np.nanmax(np.abs(err)):.3g} (file units)",
        flush=True,
    )
    return out


def coarse(job):
    """The starts `keep` of one monthly hindcast file on the 2.5 deg grid: (starts, leads, (S, L, 73, 144) float32 in
    the file's units)."""
    path, keep = job
    ds = xr.open_dataset(path)
    a = ds[next(iter(ds.data_vars))].isel(S=keep)
    if "P" in a.dims:
        a = a.isel(P=0, drop=True)
    a = a.transpose("S", "L", "Y", "X").load().astype("float32")
    ds.close()
    a = xr.concat([a, a.isel(X=0).assign_coords(X=360.0)], dim="X")
    return pd.DatetimeIndex(a.S.values), a.L.values, a.interp(Y=LAT25, X=LON25).values.astype("float32")


def in_order(jobs, workers):
    """coarse() over the files in order, at most 2 x workers in flight (each result is ~30 MB)."""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        it = iter(jobs)
        pend = deque(ex.submit(coarse, j) for j in itertools.islice(it, 2 * workers))
        while pend:
            r = pend.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pend.append(ex.submit(coarse, nxt))
            yield r


def anomalies(tag, workers, rebase=True, limit=0):
    """Every hindcast start of one field once, in date order, de-drifted as the live product does it, on the 2.5 deg
    grid in the file's units: yields (start, leads L, valid dates, in-sample anomaly, leave-one-year-out anomaly), the
    anomalies (L, 73, 144) float64. In sample F - M + shift; leave-one-year-out F - M_out + shift + (E_0120 out -
    E_0120), the ERA5 terms at the valid date and only when `rebase` (the field is re-based to 1991-2020 live).
    limit: the first `limit` monthly files only (a smoke test; the pool sizes then go unchecked)."""
    mc = ModelClim(tag)
    jobs = build_clim.entries(tag)
    if limit:
        jobs = jobs[:limit]
    d0 = pd.Timestamp(Y0, 1, 1)
    shift = dE = None
    if rebase:
        shift = period_shift(tag)
        t0 = time.time()
        dE = era5_leave_out(tag, pd.date_range(d0, pd.Timestamp(Y1 + 1, 3, 1)))
        print(f"  {tag}: ERA5 leave-out fields ({time.time() - t0:.0f} s)", flush=True)
    months = deque(maxlen=3)  # the month being scored and its neighbours: every pool mate is within 20 days

    def month(m):
        S, L, X = months[m]
        mates = [(s, k, x) for S_, _, X_ in months for s, k, x in zip(S_, noleap_doy(S_), X_)]
        for si, s0 in enumerate(S):
            full, out = mc.at(s0, [(k, x) for s, k, x in mates if abs((s - s0).days) <= SEASON])
            valid = s0 + pd.to_timedelta(np.floor(L).astype(int), unit="D")  # L = 0.5 is the start date itself
            a_in, a_out = X[si] - full, X[si] - out
            if rebase:
                sh = at_doy(shift, valid)
                a_in, a_out = a_in + sh, a_out + sh + dE[(valid - d0).days]
            yield s0, L, valid, a_in, a_out

    t0 = time.time()
    for n, (S, L, X) in enumerate(in_order(jobs, workers), 1):
        months.append((S, L, X))
        mc.seen += [pooled(noleap_doy(S), c, CLIM_HALF).sum() for c in mc.centres]
        if len(months) >= 2:
            yield from month(len(months) - 2)
        if n % 20 == 0 or n == len(jobs):
            el = time.time() - t0
            print(
                f"  {tag} {n}/{len(jobs)}  {el / 60:.1f} min  ~{(len(jobs) - n) * el / n / 60:.0f} min left", flush=True
            )
    yield from month(len(months) - 1)
    if not limit and not np.array_equal(mc.seen, mc.N):
        raise SystemExit(f"{tag}: pool sizes {mc.seen} differ from the climatology's n_starts {mc.N}")


def score_tag(tag, workers):
    """{index: {start: (L,) leave-one-year-out index}}, the same in sample, leads."""
    pat = patterns()
    ids = [i for i in pat.ids if pat.meta[i]["field"] == ("z500" if tag == "zg500" else "slp")]
    cv, ins, L_ref = {i: {} for i in ids}, {i: {} for i in ids}, None
    for s0, L, valid, a_in, a_out in anomalies(tag, workers):
        L_ref = L
        pair = np.stack([a_in, a_out])  # (2, L, 73, 144)
        if tag == "mslp":
            pair = pair * 0.01  # Pa -> hPa, the patterns' units
        ok = np.isfinite(pair).all(axis=(0, 2, 3))
        if not ok.any():
            continue
        mo = np.tile(valid.month.values[ok], 2)
        dy = np.tile(noleap_doy(valid)[ok], 2)
        for i in ids:
            v = np.full((2, L.size), np.nan, "float32")
            v[:, ok] = pat.index(i, pair[:, ok].reshape(-1, *pair.shape[2:]), mo, dy).reshape(2, -1)
            ins[i][s0], cv[i][s0] = v
    return cv, ins, L_ref


def observed_series(pat, years):
    """Daily observed indices from NCEP R1 for the hindcast years (+1 for the
    leads that spill into the next year)."""
    import build_telecon_patterns as b

    out = {}
    for var, level, fld in (("hgt", 500, "z500"), ("slp", None, "slp")):
        a = b.daily(var, years, level)
        an = a.values - at_doy(ncep_normal(fld), a.time.values)
        mo = a.time.dt.month.values
        dy = noleap_doy(a.time.values)
        for sid in pat.ids:
            if pat.meta[sid]["field"] != fld:
                continue
            out[sid] = pd.Series(pat.index(sid, an, mo, dy), index=pd.DatetimeIndex(a.time.values))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tags", default="zg500,mslp")
    ap.add_argument(
        "--merge",
        action="store_true",
        help="keep the indices already in telecon/hindcast_indices.nc that this run does "
        "not recompute (e.g. --tags mslp after adding an slp index)",
    )
    a = ap.parse_args()
    pat = patterns()
    t0 = time.time()
    obs = observed_series(pat, list(range(Y0, Y1 + 2)))
    print(f"observed indices: {len(obs)} series, {time.time() - t0:.0f} s", flush=True)
    fc, fc_in, L_ref = {}, {}, None
    for tag in a.tags.split(","):
        cv, ins, L_ref = score_tag(tag, a.workers)
        fc.update(cv)
        fc_in.update(ins)
    ids = list(fc)
    S_union = np.array(sorted({s for i in ids for s in fc[i]}), dtype="datetime64[ns]")
    F = np.full((len(ids), S_union.size, L_ref.size), np.nan, "float32")
    F_in = np.full_like(F, np.nan)
    O = np.full_like(F, np.nan)
    valid_day = np.ceil(L_ref).astype(int)
    for ii, i in enumerate(ids):
        for k, s0 in enumerate(pd.DatetimeIndex(S_union)):
            if s0 in fc[i]:
                F[ii, k], F_in[ii, k] = fc[i][s0], fc_in[i][s0]
            O[ii, k] = obs[i].reindex(s0 + pd.to_timedelta(valid_day - 1, unit="D")).values
    ds = xr.Dataset(
        {"fcst": (("index", "S", "L"), F), "obs": (("index", "S", "L"), O)},
        coords={"index": ids, "S": S_union, "L": L_ref},
    )
    ds.attrs.update(
        units="seasonal",
        source="GEPS8 hindcast ensemble mean vs NCEP/NCAR R1, indices per telecon/patterns.nc",
        note="anomalies de-drifted against the lead-matched GEPS8 climatology and re-based to 1991-2020 as in the "
        "live product, LEAVE-ONE-YEAR-OUT: the climatology and the ERA5 2001-2020 normal of the re-basing without "
        f"the scored start's own season (pool members within {SEASON} days); each start once, common-year day of "
        "year; obs at the valid date of each lead",
        dedrift="leave-one-year-out",
    )
    TC.mkdir(exist_ok=True)
    q = TC / "hindcast_indices.nc"
    if a.merge and q.exists():
        old = xr.open_dataset(q).load()
        old.close()
        keep = [i for i in old.index.values if i not in ids]
        if keep:
            assert np.allclose(old.L.values, L_ref), "lead axis differs from the existing file"
            if old.attrs.get("units", "annual") != ds.attrs["units"]:
                raise SystemExit("existing file is in annual units — rerun without --merge")
            if old.attrs.get("dedrift") != ds.attrs["dedrift"]:
                raise SystemExit("existing file is de-drifted in sample — rerun without --merge")
            S_all = np.union1d(old.S.values, ds.S.values)
            ds = xr.concat([old.sel(index=keep).reindex(S=S_all), ds.reindex(S=S_all)], dim="index")
            # concat keeps the first file's fixed-width string dtype ('eqsoi' became 'eqso')
            ds = ds.assign_coords(index=np.array([str(i) for i in keep] + list(ids), dtype=object))
            print(f"  merged: kept {keep} from the existing file, recomputed {ids}")
    ds.to_netcdf(q)
    print("  daily r, in sample -> leave-one-year-out, days 1/7/14/21/28/35")
    for ii, i in enumerate(ids):
        r = []
        for k in (0, 6, 13, 20, 27, 34):
            for X in (F_in, F):
                ok = np.isfinite(X[ii, :, k]) & np.isfinite(O[ii, :, k])
                r.append(np.corrcoef(X[ii, ok, k], O[ii, ok, k])[0, 1])
        print(f"  {i:5s} " + "  ".join(f"{r[j]:+.3f}->{r[j + 1]:+.3f}" for j in range(0, len(r), 2)), flush=True)
    print(f"done {time.time() - t0:.0f} s -> hindcast_indices.nc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
