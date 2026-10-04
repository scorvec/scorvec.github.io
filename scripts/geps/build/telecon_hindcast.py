#!/usr/bin/env python3
"""Score the teleconnection indices on the GEPS8 hindcast, lead by lead.

The real-time indices (telecon_geps.py) say what the 21 members forecast. This
says how much that is worth: the same projectors applied to every hindcast
start of 2001-2020 (four a week, 39 daily leads, ensemble mean), against the
observed index on the valid date from NCEP/NCAR R1 — the reanalysis the
patterns are defined on. The de-drifting is identical to the live product
(forecast minus the lead-matched GEPS8 climatology, re-based to 1991-2020), so
what is scored is exactly what is shown.

The output is the raw paired series, not summary skill: telecon_geps.py pools
the starts within +-45 days of the current init day-of-year and, per index and
week, fits obs = a * fcst + b on the weekly means, giving the anomaly
correlation (skill), the amplitude factor a and the residual spread s. From
those the live ensemble-mean weekly index m becomes a CALIBRATED probability,
P(week >= +0.5) = 1 - Phi((0.5 - a m - b) / s), shown next to the raw member
fraction. Where skill is nil (r ~ 0) the calibration collapses to climatology
— which is the honest statement for a week-5 NAO.

    python telecon_hindcast.py --workers 2
    -> telecon/hindcast_indices.nc   fcst[index, S, L], obs[index, S, L]
    python telecon_hindcast.py --tags mslp --merge    # only the slp indices, keep the rest

Memory: each worker holds a monthly file (18 x 39 x 181 x 360) plus its 2.5 deg
copy; at float64 that was 3.5-5 GB per worker and five workers took 21 GB of
a 36 GB laptop (2026-09-03). Fields are cast to float32 on load and the
default is two workers — about 3 GB total, ~15 min for the 480 files.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path[:0] = [str(HERE), str(HERE.parent)]
from paths import LOCAL_STORE as _LS                             # noqa: E402  laptop-only inputs/outputs
STORE = _LS / "geps8_hindcast"
CLIM = _LS / "climatology"
TC = _LS / "telecon"
NCEP = TC / "ncep"
LAT25 = np.arange(90.0, -90.1, -2.5)
LON25 = np.arange(0.0, 360.0, 2.5)
Y0, Y1 = 2001, 2020

_PAT = None


def patterns():
    global _PAT
    if _PAT is None:
        import telecon_geps as t
        _PAT = t.Patterns()
    return _PAT


def period_shift(tag):
    """ERA5 1991-2020 minus 2001-2020 re-basing on the 2.5 deg grid, by day of
    year (index doy-1), the same term render_maps.reref adds to every map."""
    import render_maps as rm
    dummy = xr.DataArray(np.zeros((366, LAT25.size, LON25.size)), dims=("L", "latitude", "longitude"),
                         coords={"L": np.arange(366), "latitude": LAT25, "longitude": LON25})
    r = rm.reref(tag, dummy, pd.Timestamp("2000-01-01"))     # leap year: L = doy - 1
    return None if r is None else r.values.astype("float32")


_CW25 = {}


def clim_wrapped25(tag):
    """Lead-dependent model climatology on the 2.5 deg grid, day-of-year axis
    wrapped for interpolation. Coarsening the climatology ONCE per worker and
    the forecast once per file is what makes this affordable: subtracting on
    the 1 deg grid and coarsening the difference cost 30-40 s per file."""
    if tag not in _CW25:
        c = xr.open_dataset(CLIM / f"geps8_clim_{tag}.nc")
        cv = c[[v for v in c.data_vars if v != "n_starts"][0]].load()
        cv = xr.concat([cv, cv.isel(X=0).assign_coords(X=360.0)], dim="X")
        cv = cv.interp(Y=LAT25, X=LON25).rename({"Y": "lat", "X": "lon"})
        _CW25[tag] = xr.concat([cv.isel(doy=-1).assign_coords(doy=cv.doy[-1] - 366), cv,
                                cv.isel(doy=0).assign_coords(doy=cv.doy[0] + 366)], dim="doy")
    return _CW25[tag]


_RG = None


def regimes():
    global _RG
    if _RG is None:
        from regimes_geps import Regimes
        _RG = Regimes()
    return _RG


def regime_labels(an, S, L):
    """Regime of the hindcast ensemble mean at every lead, per sector, using
    the regime set of the START's season (held for the whole forecast, as
    in the live product). an: (L, 73, 144) de-drifted, re-based z500."""
    from regimes_geps import sector_cut, running_partial, season_of, SECTORS
    rg = regimes()
    season = season_of(S.month)
    da = xr.DataArray(an, dims=("L", "latitude", "longitude"),
                      coords={"L": L, "latitude": LAT25, "longitude": LON25})
    out = {}
    for sk in SECTORS:
        sec = sector_cut(da, SECTORS[sk])
        v = running_partial(sec.values, 5, axis=0)
        lab, _ = rg.classify(v, sk, season)
        out[sk] = lab.astype("int8")
    return out


def one_file(args):
    """Indices for every (start, lead) in one monthly hindcast file."""
    path, tag, want_regimes = args
    pat = patterns()
    cw = clim_wrapped25(tag)
    shift = period_shift(tag)
    ds = xr.open_dataset(path)
    a = ds[[v for v in ds.data_vars][0]]
    if "P" in a.dims:
        a = a.isel(P=0)
    a = a.load().astype("float32")
    ds.close()
    scale = 0.01 if (tag == "mslp" and float(np.nanmean(a.values)) > 2000) else 1.0
    ids = [i for i in pat.ids if pat.meta[i]["field"] == ("z500" if tag == "zg500" else "slp")]
    S = pd.DatetimeIndex(a.S.values)
    L = a.L.values
    a = xr.concat([a, a.isel(X=0).assign_coords(X=360.0)], dim="X")
    a25 = a.interp(Y=LAT25, X=LON25).values.astype("float32")        # (S, L, 73, 144)
    del a
    out = {i: np.full((S.size, L.size), np.nan, "float32") for i in ids}
    reg = {}
    for si, s0 in enumerate(S):
        ref = cw.interp(doy=s0.dayofyear).sel(L=L, method="nearest").values
        an = a25[si] - ref                                            # (L, 73, 144)
        valid = s0 + pd.to_timedelta(np.floor(L).astype(int), unit="D")      # L = 0.5 is the start date itself
        if shift is not None:
            an = an + shift[valid.dayofyear.values - 1]
        an = an * scale
        ok = np.isfinite(an).all(axis=(1, 2))
        if not ok.any():
            continue
        mo = valid.month.values
        dy = valid.dayofyear.values
        for i in ids:
            out[i][si, ok] = pat.index(i, an[ok], mo[ok], dy[ok])
        if want_regimes and tag == "zg500" and ok.all():
            reg[si] = regime_labels(an, s0, L)
    return tag, S.values, L, out, reg


def observed_series(pat, years):
    """Daily observed indices from NCEP R1 for the hindcast years (+1 for the
    leads that spill into the next year)."""
    import build_telecon_patterns as b
    out = {}
    for var, level, fld in (("hgt", 500, "z500"), ("slp", None, "slp")):
        a = b.daily(var, years, level)
        clim = pat.ds["clim_doy_z500" if var == "hgt" else "clim_doy_slp"]
        an = a.values - clim.values[a.time.dt.dayofyear.values - 1]
        mo = a.time.dt.month.values
        dy = a.time.dt.dayofyear.values
        for sid in pat.ids:
            if pat.meta[sid]["field"] != fld:
                continue
            out[sid] = pd.Series(pat.index(sid, an, mo, dy), index=pd.DatetimeIndex(a.time.values))
    return out


def write_regime_hindcast(pat, regs, L):
    """Pair the hindcast-mean regimes with the observed regime on the valid
    day (NCEP daily z500, classified with the START's season set) and the
    observed regime at init, for hit rates by lead."""
    from regimes_geps import Regimes, sector_cut, running_partial, season_of, SECTORS, K
    import build_telecon_patterns as b
    rg = Regimes()
    zd = b.daily("hgt", list(range(Y0, Y1 + 2)), 500)
    t = pd.DatetimeIndex(zd.time.values)
    an = zd.values - pat.ds["clim_doy_z500"].values[t.dayofyear.values - 1]
    da = xr.DataArray(an, dims=("time", "latitude", "longitude"),
                      coords={"time": t, "latitude": zd.lat.values, "longitude": zd.lon.values})
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
        F = np.full((len(starts), len(L)), -1, "int8"); O = F.copy(); O0 = np.full(len(starts), -1, "int8")
        for i, s0 in enumerate(starts):
            season = season_of(s0.month)
            F[i] = regs[s0][sk]
            o = obs[(sk, season)]
            vd = s0 + pd.to_timedelta(valid_day - 1, unit="D")
            O[i] = o.reindex(vd).fillna(-1).values.astype("int8")
            O0[i] = int(o.get(s0, -1))
        data[f"fc_{sk}"] = (("S", "L"), F); data[f"ob_{sk}"] = (("S", "L"), O); data[f"ob0_{sk}"] = (("S",), O0)
    ds = xr.Dataset(data, coords={"S": starts.values, "L": L})
    ds.attrs.update(note=f"regime index 0..{K-1} per telecon/regime_names.json, {K} = no regime, -1 = missing; "
                         "season set of the start held for the whole forecast; obs = NCEP R1 daily z500")
    ds.to_netcdf(TC / "regime_hindcast.nc")
    for sk in SECTORS:
        F, O = data[f"fc_{sk}"][1], data[f"ob_{sk}"][1]
        ok = (F >= 0) & (O >= 0)
        hit = [np.mean((F[:, k] == O[:, k])[ok[:, k]]) for k in (0, 6, 13, 20, 27)]
        print(f"  regimes {sk}: hit rate day 1/7/14/21/28 " + " ".join(f"{h:.2f}" for h in hit), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tags", default="zg500,mslp")
    ap.add_argument("--regimes", action="store_true",
                    help="also record the ensemble-mean weather regime per sector and lead, and the "
                         "observed regime on the valid day -> telecon/regime_hindcast.nc")
    ap.add_argument("--regimes-only", action="store_true", help="skip the index file (zg500 only)")
    ap.add_argument("--merge", action="store_true",
                    help="keep the indices already in telecon/hindcast_indices.nc that this run does "
                         "not recompute (e.g. --tags mslp after adding an slp index)")
    a = ap.parse_args()
    if a.regimes_only:
        a.regimes, a.tags = True, "zg500"
    pat = patterns()
    t0 = time.time()
    obs = observed_series(pat, list(range(Y0, Y1 + 2)))
    print(f"observed indices: {len(obs)} series, {time.time()-t0:.0f} s", flush=True)
    jobs = []
    for tag in a.tags.split(","):
        jobs += [(p, tag, a.regimes) for p in sorted(STORE.glob(f"geps8_{tag}_*.nc"))]
    print(f"{len(jobs)} hindcast files on {a.workers} processes", flush=True)
    fc, S_all, L_ref = {}, [], None
    regs = {}                                                   # start -> {sector: labels}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, (tag, S, L, out, reg) in enumerate(ex.map(one_file, jobs), 1):
            L_ref = L
            for si, r in reg.items():
                regs[S[si]] = r
            for i, v in out.items():
                fc.setdefault(i, []).append((S, v))
            if n % 20 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el/60:.1f} min  ~{(len(jobs)-n)*el/n/60:.0f} min left", flush=True)
    ids = list(fc)
    S_union = np.unique(np.concatenate([S for i in ids for S, _ in fc[i]]))
    F = np.full((len(ids), S_union.size, L_ref.size), np.nan, "float32")
    O = np.full_like(F, np.nan)
    pos = {s: k for k, s in enumerate(S_union)}
    valid_day = np.ceil(L_ref).astype(int)
    for ii, i in enumerate(ids):
        for S, v in fc[i]:
            for si, s0 in enumerate(S):
                F[ii, pos[s0]] = v[si]
        o = obs[i]
        for k, s0 in enumerate(S_union):
            vd = pd.DatetimeIndex(pd.Timestamp(s0) + pd.to_timedelta(valid_day - 1, unit="D"))
            O[ii, k] = o.reindex(vd).values
    ds = xr.Dataset({"fcst": (("index", "S", "L"), F), "obs": (("index", "S", "L"), O)},
                    coords={"index": ids, "S": S_union, "L": L_ref})
    ds.attrs.update(units="seasonal",
                    source="GEPS8 hindcast ensemble mean vs NCEP/NCAR R1, indices per telecon/patterns.nc",
                    note="anomalies de-drifted against the lead-matched GEPS8 climatology and re-based "
                         "to 1991-2020, as in the live product; obs at the valid date of each lead")
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
            S_all = np.union1d(old.S.values, ds.S.values)
            ds = xr.concat([old.sel(index=keep).reindex(S=S_all), ds.reindex(S=S_all)], dim="index")
            # concat keeps the first file's fixed-width string dtype ('eqsoi' became 'eqso')
            ds = ds.assign_coords(index=np.array([str(i) for i in keep] + list(ids), dtype=object))
            print(f"  merged: kept {keep} from the existing file, recomputed {ids}")
    if not a.regimes_only:
        ds.to_netcdf(q)
    if a.regimes and regs:
        write_regime_hindcast(pat, regs, L_ref)
    for ii, i in enumerate(ids):
        r = [np.corrcoef(*pd.DataFrame({"f": F[ii, :, k], "o": O[ii, :, k]}).dropna().values.T)[0, 1]
             for k in (0, 6, 13, 20, 27)]
        print(f"  {i:5s} r(day 1/7/14/21/28) " + " ".join(f"{x:+.2f}" for x in r))
    print(f"done {time.time()-t0:.0f} s -> hindcast_indices.nc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
