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

  M without its season       every file entry of M's pools (starts within +-15 days of a 5-day centre) that lies
                             within SEASON days of the scored start: (N M - their sum) / (N - n), at the two centres
                             the live read interpolates between. The pools are mirrored as the climatology file holds
                             them - raw day of year, and the 154 starts that sit in two consecutive monthly files
                             (e.g. 2001-03-02 in both 200102 and 200103) counted twice, as build_clim.py counted them;
  E_0120 without its season  the days of the ERA5 normal's windows (+-7 days of a 5-day centre) within SEASON days of
                             the valid date, removed the same way. 500 hPa height: the 1.5 deg daily store the normal
                             was built from, every day. Sea-level pressure: its normal is the 1 deg ARCO one averaged
                             every fifth day of each year (1, 6, 11 ... January); the same days are taken from the
                             1.5 deg daily store (a daily mean of the same four synoptic hours).

The re-basing target E_9120 is the definition of the anomaly (the observed index is taken against the 1991-2020 NCEP
normal, which holds the scored year too), not a fitted part of the forecast, and stays as it is.

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
import build_era5_clim as era5  # the ERA5 normal's store and units

STORE = _LS / "geps8_hindcast"
CLIM = _LS / "climatology"
TC = _LS / "telecon"
LAT25 = np.arange(90.0, -90.1, -2.5)
LON25 = np.arange(0.0, 360.0, 2.5)
Y0, Y1 = 2001, 2020
SEASON = 60  # days: a pooled start or day this close to the scored one is its own season (other years are ~365 away)
CLIM_HALF, ERA5_HALF = 15, 7  # pool half-widths of the GEPS8 climatology and the ERA5 normal, days of year
ERA5_SAMPLE = {"zg500": 1, "mslp": 5}  # every n-th day of each year in the ERA5 2001-2020 normal (see the docstring)


@functools.cache
def patterns():
    import telecon_geps as t

    return t.Patterns()


@functools.cache
def ncep_normal(field):
    """The NCEP 1991-2020 day-of-year normal of z500 or slp the patterns were built against (the full pattern file of
    the offline store: the published reference drops these)."""
    return xr.open_dataset(TC / "patterns.nc")[f"clim_doy_{field}"].values


def raw_doy(t):
    """Day of the year as the climatology files were pooled (1 March = 61 in a leap year)."""
    return pd.DatetimeIndex(t).dayofyear.values.astype("float64")


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
    taken out of it: centre k is the mean of the N_k file entries whose raw day of year is within CLIM_HALF of it."""

    def __init__(self, tag):
        c = xr.open_dataset(CLIM / f"geps8_clim_{tag}.nc")
        cv = c[next(v for v in c.data_vars if v != "n_starts")].squeeze(drop=True).load()
        cv = xr.concat([cv, cv.isel(X=0).assign_coords(X=360.0)], dim="X")
        cv = cv.interp(Y=LAT25, X=LON25).transpose("doy", "L", "Y", "X")
        self.C = cv.values.astype("float64")  # (74 centres, L, 73, 144); centre 366 is 1 January again
        self.L = cv.L.values
        self.centres = cv.doy.values.astype("float64")
        self.N = c["n_starts"].values.astype("float64")
        self.seen = np.zeros_like(self.N)  # entries met while scoring: must equal N
        c.close()

    def at(self, s0, mates):
        """(in-sample, leave-one-year-out) climatology for a start: mates = [(raw doy, field)] of the file entries
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
    (season_left_out), from the store and days the normal in use was built from: windows on the raw day of year,
    2001-2020; checked against that normal."""
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
    out, E = season_left_out(days, np.concatenate(fields), raw_doy(days), dates)
    ref = xr.open_dataset(CLIM / ("era5_clim_2001-2020.nc" if tag == "zg500" else "era5_clim_sfc_2001-2020.nc"))[tag]
    ref = ref.sortby("lat", ascending=False)
    ref = xr.concat([ref, ref.isel(lon=0).assign_coords(lon=360.0)], dim="lon").interp(lat=LAT25, lon=LON25)
    err = E - ref.values
    print(
        f"  ERA5 2001-2020 {tag} normal rebuilt from {len(days)} store days: vs the normal in use "
        f"rms {np.sqrt(np.nanmean(err**2)):.3g}, max |d| {np.nanmax(np.abs(err)):.3g} (Pa or m)",
        flush=True,
    )
    return out


@functools.cache
def regimes():
    from regimes_geps import Regimes

    return Regimes()


def regime_labels(an, S, L):
    """Regime of the hindcast ensemble mean at every lead, per sector, using
    the regime set of the START's season (held for the whole forecast, as
    in the live product). an: (L, 73, 144) de-drifted, re-based z500."""
    from regimes_geps import sector_cut, running_partial, season_of, SECTORS

    rg = regimes()
    season = season_of(S.month)
    da = xr.DataArray(an, dims=("L", "latitude", "longitude"), coords={"L": L, "latitude": LAT25, "longitude": LON25})
    out = {}
    for sk in SECTORS:
        sec = sector_cut(da, SECTORS[sk])
        v = running_partial(sec.values, 5, axis=0)
        lab, _ = rg.classify(v, sk, season)
        out[sk] = lab.astype("int8")
    return out


def coarse(path):
    """One monthly hindcast file on the 2.5 deg grid: (starts, leads, (S, L, 73, 144) float32 in the file's units)."""
    ds = xr.open_dataset(path)
    a = ds[next(iter(ds.data_vars))].squeeze(drop=True).transpose("S", "L", "Y", "X").load().astype("float32")
    ds.close()
    a = xr.concat([a, a.isel(X=0).assign_coords(X=360.0)], dim="X")
    return pd.DatetimeIndex(a.S.values), a.L.values, a.interp(Y=LAT25, X=LON25).values.astype("float32")


def in_order(paths, workers):
    """coarse() over the files in order, at most 2 x workers in flight (each result is ~30 MB)."""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        it = iter(paths)
        pend = deque(ex.submit(coarse, p) for p in itertools.islice(it, 2 * workers))
        while pend:
            r = pend.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pend.append(ex.submit(coarse, nxt))
            yield r


def score_tag(tag, workers, want_regimes):
    """{index: {start: (L,) leave-one-year-out index}}, the same in sample, regime labels by start, leads."""
    pat = patterns()
    ids = [i for i in pat.ids if pat.meta[i]["field"] == ("z500" if tag == "zg500" else "slp")]
    mc = ModelClim(tag)
    shift = period_shift(tag)
    files = sorted(STORE.glob(f"geps8_{tag}_*.nc"))
    d0 = pd.Timestamp(Y0, 1, 1)
    vdates = pd.date_range(d0, pd.Timestamp(Y1 + 1, 3, 1))
    t0 = time.time()
    dE = era5_leave_out(tag, vdates)
    print(f"  {tag}: ERA5 leave-out fields for {len(vdates)} valid dates ({time.time() - t0:.0f} s)", flush=True)
    cv, ins, reg, done = {i: {} for i in ids}, {i: {} for i in ids}, {}, set()
    months = deque(maxlen=3)  # the month being scored and its neighbours: every pool mate is within 20 days

    def score(m):
        S, L, X = months[m]
        mates = [(s, d, x) for S_, _, X_ in months for s, d, x in zip(S_, raw_doy(S_), X_)]
        for si, s0 in enumerate(S):
            if s0 in done:  # the second file entry of a start that two monthly files both hold
                continue
            done.add(s0)
            own = [(d, x) for s, d, x in mates if abs((s - s0).days) <= SEASON]
            full, out = mc.at(s0, own)
            valid = s0 + pd.to_timedelta(np.floor(L).astype(int), unit="D")  # L = 0.5 is the start date itself
            sh = at_doy(shift, valid) if shift is not None else 0.0
            pair = np.stack([X[si] - full + sh, X[si] - out + sh + dE[(valid - d0).days]])  # (2, L, 73, 144)
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
            if want_regimes and tag == "zg500" and ok.all():
                reg[s0] = regime_labels(pair[1], s0, L)

    for n, (S, L, X) in enumerate(in_order(files, workers), 1):
        months.append((S, L, X))
        mc.seen += [pooled(raw_doy(S), c, CLIM_HALF).sum() for c in mc.centres]
        if len(months) >= 2:
            score(len(months) - 2)
        if n % 20 == 0 or n == len(files):
            el = time.time() - t0
            print(
                f"  {tag} {n}/{len(files)}  {el / 60:.1f} min  ~{(len(files) - n) * el / n / 60:.0f} min left",
                flush=True,
            )
    score(len(months) - 1)
    if not np.array_equal(mc.seen, mc.N):
        raise SystemExit(f"{tag}: pool sizes {mc.seen} differ from the climatology's n_starts {mc.N}")
    return cv, ins, reg, mc.L


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


def write_regime_hindcast(regs, L):
    """Pair the hindcast-mean regimes with the observed regime on the valid
    day (NCEP daily z500, classified with the START's season set) and the
    observed regime at init, for hit rates by lead."""
    from regimes_geps import Regimes, sector_cut, running_partial, season_of, SECTORS, K
    import build_telecon_patterns as b

    rg = Regimes()
    zd = b.daily("hgt", list(range(Y0, Y1 + 2)), 500)
    t = pd.DatetimeIndex(zd.time.values)
    an = zd.values - at_doy(ncep_normal("z500"), t)
    da = xr.DataArray(
        an,
        dims=("time", "latitude", "longitude"),
        coords={"time": t, "latitude": zd.lat.values, "longitude": zd.lon.values},
    )
    obs = {}
    for sk in SECTORS:
        sec = sector_cut(da, SECTORS[sk])
        v = running_partial(sec.values, 5, axis=0)
        for season in ("cold", "warm"):
            lab, _ = rg.classify(v, sk, season)
            obs[(sk, season)] = pd.Series(lab.astype("int8"), index=t)
    starts = pd.DatetimeIndex(sorted(regs))
    valid_day = np.ceil(L).astype(int)
    data = {}
    for sk in SECTORS:
        F = np.full((len(starts), len(L)), -1, "int8")
        O = F.copy()
        O0 = np.full(len(starts), -1, "int8")
        for i, s0 in enumerate(starts):
            season = season_of(s0.month)
            F[i] = regs[s0][sk]
            o = obs[(sk, season)]
            vd = s0 + pd.to_timedelta(valid_day - 1, unit="D")
            O[i] = o.reindex(vd).fillna(-1).values.astype("int8")
            O0[i] = int(o.get(s0, -1))
        data[f"fc_{sk}"] = (("S", "L"), F)
        data[f"ob_{sk}"] = (("S", "L"), O)
        data[f"ob0_{sk}"] = (("S",), O0)
    ds = xr.Dataset(data, coords={"S": starts.values, "L": L})
    ds.attrs.update(
        note=f"regime index 0..{K - 1} per telecon/regime_names.json, {K} = no regime, -1 = missing; "
        "season set of the start held for the whole forecast; obs = NCEP R1 daily z500; hindcast mean de-drifted "
        "leave-one-year-out"
    )
    ds.to_netcdf(TC / "regime_hindcast.nc")
    for sk in SECTORS:
        F, O = data[f"fc_{sk}"][1], data[f"ob_{sk}"][1]
        ok = (F >= 0) & (O >= 0)
        hit = [np.mean((F[:, k] == O[:, k])[ok[:, k]]) for k in (0, 6, 13, 20, 27)]
        print(f"  regimes {sk}: hit rate day 1/7/14/21/28 " + " ".join(f"{h:.2f}" for h in hit), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tags", default="zg500,mslp")
    ap.add_argument(
        "--regimes",
        action="store_true",
        help="also record the ensemble-mean weather regime per sector and lead, and the "
        "observed regime on the valid day -> telecon/regime_hindcast.nc",
    )
    ap.add_argument("--regimes-only", action="store_true", help="skip the index file (zg500 only)")
    ap.add_argument(
        "--merge",
        action="store_true",
        help="keep the indices already in telecon/hindcast_indices.nc that this run does "
        "not recompute (e.g. --tags mslp after adding an slp index)",
    )
    a = ap.parse_args()
    if a.regimes_only:
        a.regimes, a.tags = True, "zg500"
    pat = patterns()
    t0 = time.time()
    obs = observed_series(pat, list(range(Y0, Y1 + 2)))
    print(f"observed indices: {len(obs)} series, {time.time() - t0:.0f} s", flush=True)
    fc, fc_in, regs, L_ref = {}, {}, {}, None
    for tag in a.tags.split(","):
        cv, ins, reg, L_ref = score_tag(tag, a.workers, a.regimes)
        fc.update(cv)
        fc_in.update(ins)
        regs.update(reg)
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
        f"the scored start's own season (pool members within {SEASON} days); obs at the valid date of each lead",
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
    if not a.regimes_only:
        ds.to_netcdf(q)
    if a.regimes and regs:
        write_regime_hindcast(regs, L_ref)
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
