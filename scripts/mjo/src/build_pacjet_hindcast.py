#!/usr/bin/env python3
"""Archived-forecast sample for the North Pacific jet product's drift correction and skill strip (laptop, once;
re-run to extend).

Neither AIFS-ENS nor IFS-ENS has an open reforecast, but the Google Cloud mirror of ECMWF open data keeps the real-time
runs (AIFS-ENS back to 2025-07-01, IFS-ENS to 2024-02-28, measured 2026-09-27). This byte-ranges the 250 hPa zonal
wind of ten members of every other 00Z cycle, days 1-15, from THAT MIRROR ONLY (user rule: Google Cloud, never another
mirror), reduces each field at once to the 1.5 deg jet grid (pacjet_core: conservative 0.25 -> 0.5 -> 1.5 deg), and
keeps the reduced fields under ~/mjo/pacjet_hindcast/ (outside the repo; ~2 MB a cycle). The verifying analysis is the
AIFS-ENS control at step 0 of every 00Z cycle (the ECMWF operational analysis both ensembles start from), one message a
day.

  members  aifs: the control + perturbed 1-9; ifs: perturbed 1-10 (IFS open data has no control on pressure levels)
  cost     ~120 MB per cycle on the wire (0.74 MB a message), none of it kept

    python src/build_pacjet_hindcast.py fetch --model aifs --start 20250701 --end 20260912
    python src/build_pacjet_hindcast.py fetch --model ifs  --start 20250701 --end 20260912
    python src/build_pacjet_hindcast.py analyses --start 20250701 --end 20260928
    python src/build_pacjet_hindcast.py stats          # after build_pacjet_ref.py: drift + skill -> data/reference
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("ECMWF_RPS_GOOGLE", "60")
os.environ.setdefault("ECMWF_RPS", "60")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ecmwf"))
import pacjet_core as PC                                                    # noqa: E402

ROOT = Path(os.environ.get("PACJET_HINDCAST", str(Path.home() / "mjo" / "pacjet_hindcast")))
REF = HERE.parent / "data" / "reference"
LEADS = list(range(1, 16))
SRC = ["google"]


def _decode(blob, entries):
    """Messages in offset order -> list of (entry, (nj, ni) values, lat, lon)."""
    import eccodes
    out, pos = [], 0
    for e in entries:
        gid = eccodes.codes_new_from_message(bytes(blob[pos:pos + e["_length"]]))
        pos += e["_length"]
        try:
            nj, ni = eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")
            la0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
            la1 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
            lo0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
            v = eccodes.codes_get_values(gid).reshape(nj, ni)
        finally:
            eccodes.codes_release(gid)
        lat = np.linspace(la0, la1, nj); lon = (lo0 + 0.25 * np.arange(ni)) % 360
        out.append((e, v, lat, lon))
    if pos != len(blob):
        raise RuntimeError(f"{len(blob)} bytes for {pos} indexed")
    return out


def _grab(date, model, step, kind, numbers, typ=None):
    import rangefetch as rf
    idx = rf.fetch_index(date, "00", model, step, kind, sources=SRC)
    want = sorted(rf.select(idx, param="u", levelist=[PC.LEVEL], numbers=numbers, type=typ), key=lambda e: e["_offset"])
    if not want:
        raise RuntimeError(f"{model} {date} +{step}h: no u{PC.LEVEL} messages")
    blob = rf.fetch_ranges(rf.path_for(date, "00", model, step, kind) + ".grib2", rf.coalesce(want, 512 * 1024),
                           sources=SRC, workers=8)
    return _decode(blob, want)


def fetch_cycle(model, date):
    out = ROOT / model / f"{date}.npz"
    if out.exists():
        return "cached"
    li = lj = None
    U = np.full((10, len(LEADS), PC.EOF_LAT.size, PC.EOF_LON.size), np.nan, "float32")

    def one(L):
        rows = []
        if model == "aifs":
            rows += _grab(date, "aifs-ens", 24 * L, "cf", None)
            rows += _grab(date, "aifs-ens", 24 * L, "pf", list(range(1, 10)))
        else:
            rows += _grab(date, "ifs", 24 * L, "ef", list(range(1, 11)), typ="pf")
        return L, rows

    with ThreadPoolExecutor(4) as ex:
        for L, rows in ex.map(one, LEADS):
            for k, (e, v, lat, lon) in enumerate(rows[:10]):
                if li is None:
                    li, lj = PC.native_slices(lat, lon)
                U[k, L - 1] = PC.to_eof(PC.to_work(v, li, lj))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, u=U.astype("float16"))
    return "ok"


def fetch_analysis(date):
    out = ROOT / "analysis" / f"{date}.npz"
    if out.exists():
        return "cached"
    rows = _grab(date, "aifs-ens", 0, "cf", None)
    e, v, lat, lon = rows[0]
    li, lj = PC.native_slices(lat, lon)
    w = PC.to_work(v, li, lj)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, u=PC.to_eof(w))
    return "ok"


def days(a, b, every=1):
    d0 = dt.datetime.strptime(a, "%Y%m%d"); d1 = dt.datetime.strptime(b, "%Y%m%d")
    out = []
    while d0 <= d1:
        out.append(d0.strftime("%Y%m%d")); d0 += dt.timedelta(days=every)
    return out


def cmd_fetch(a):
    t0 = time.time(); n = 0
    for d in days(a.start, a.end, a.every):
        t1 = time.time()
        try:
            st = fetch_cycle(a.model, d)
        except Exception as e:                                                   # noqa: BLE001
            st = f"FAILED {str(e)[:90]}"
        n += st == "ok"
        print(f"  {a.model} {d}: {st} ({time.time() - t1:.0f}s, total {(time.time() - t0) / 60:.1f} min)", flush=True)


def cmd_analyses(a):
    def one(d):
        try:
            return d, fetch_analysis(d)
        except Exception as e:                                                   # noqa: BLE001
            return d, f"FAILED {str(e)[:80]}"
    with ThreadPoolExecutor(4) as ex:
        for d, st in ex.map(one, days(a.start, a.end)):
            if st != "cached":
                print(f"  analysis {d}: {st}", flush=True)


# ── drift and skill ──────────────────────────────────────────────────────────
BLOCK = 5            # cases per bootstrap / standard-error block: every-other-day cycles -> 10 days, past the indices' memory
MIN_CASES = 30       # below this a drift field is "accruing" and not applied
SMOOTH = 2.0         # gaussian smoothing of the drift field, in 1.5 deg grid points (~3 deg): drift is large-scale, noise is not


def block_boot_ci(stat_fn, n, block=BLOCK, nboot=1000, seed=0):
    """Moving-block bootstrap over the time-ordered cases -> (lo, hi) 90% interval of stat_fn(idx)."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    vals = []
    for _ in range(nboot):
        st = rng.integers(0, max(n - block + 1, 1), nb)
        idx = (st[:, None] + np.arange(block)[None]).ravel()[:n]
        vals.append(stat_fn(idx))
    vals = np.array(vals, float)
    return float(np.nanpercentile(vals, 5)), float(np.nanpercentile(vals, 95))


def drift_estimate(e_case):
    """e_case (case, lead, lat, lon) ensemble-mean-of-member errors, time-ordered -> (applied correction, raw mean, SE).
    SE from block means (autocorrelation), the mean smoothed in space, then a Wiener shrink D^2/(D^2 + SE^2) so a point
    whose drift is within its own noise is left (nearly) alone."""
    from scipy.ndimage import gaussian_filter
    n = e_case.shape[0]
    D = e_case.mean(0)
    nb = n // BLOCK
    bm = e_case[:nb * BLOCK].reshape(nb, BLOCK, *e_case.shape[1:]).mean(1)
    se = bm.std(0, ddof=1) / np.sqrt(nb)
    Ds = gaussian_filter(D, sigma=(0, SMOOTH, SMOOTH), mode="nearest")
    ses = gaussian_filter(se, sigma=(0, SMOOTH, SMOOTH), mode="nearest")
    f = Ds ** 2 / (Ds ** 2 + ses ** 2)
    return (Ds * f).astype("float32"), D.astype("float32"), se.astype("float32")


def cmd_stats(a):
    import xarray as xr
    ref = xr.open_dataset(REF / "pacjet_ref.nc").load()
    an = {p.stem: np.load(p)["u"].astype("float32") for p in sorted((ROOT / "analysis").glob("*.npz"))}
    doc = {"note": "250 hPa zonal wind, 00Z cycles every other day, 10 members; verifying analysis = AIFS-ENS control at "
                   "step 0 (the ECMWF operational analysis). bias = mean member-minus-analysis index (sigma) with 90% "
                   "moving-block bootstrap CI, raw forecasts; r / rmse = ensemble-mean indices vs the analysis indices "
                   "after the drift correction, fitted on the OTHER half of the record (two-fold by time); pers_r = "
                   "persistence of the day-0 analysis index. 'season' = cases whose basis month (init + 7 d) is in the "
                   "phase season.", "leads": LEADS, "models": {}}
    ds = xr.Dataset(coords={"lead": LEADS, "latitude": PC.EOF_LAT, "longitude": PC.EOF_LON})
    for model in ("aifs", "ifs"):
        files = sorted((ROOT / model).glob("*.npz"))
        inits, F, A, A0 = [], [], [], []
        for p in files:
            d0 = dt.datetime.strptime(p.stem, "%Y%m%d")
            vd = [(d0 + dt.timedelta(days=L)).strftime("%Y%m%d") for L in LEADS]
            if p.stem not in an or not all(v in an for v in vd):
                continue
            u = np.load(p)["u"].astype("float32")
            if not np.isfinite(u).all():
                continue
            inits.append(d0); F.append(u); A.append(np.stack([an[v] for v in vd])); A0.append(an[p.stem])
        if len(inits) < MIN_CASES:
            print(f"  {model}: {len(inits)} cases - accruing", flush=True)
            continue
        F = np.stack(F); A = np.stack(A); A0 = np.stack(A0); n = len(inits)
        mon = np.array([d.month for d in inits])
        cold = np.isin(mon, [10, 11, 12, 1, 2, 3, 4])
        e_case = (F - A[:, None]).mean(1)                                        # (case, lead, lat, lon)
        for tag, sel in (("cold", cold), ("warm", ~cold), ("all", np.ones(n, bool))):
            if sel.sum() < MIN_CASES:
                print(f"  {model} {tag}: {int(sel.sum())} cases - accruing, not applied", flush=True)
                continue
            corr, D, se = drift_estimate(e_case[sel])
            if tag == "all":                                                    # the live product uses the all-season field
                ds[f"corr_{tag}_{model}"] = (("lead", "latitude", "longitude"), corr)
                ds[f"corr_{tag}_{model}"].attrs.update(cases=int(sel.sum()))
                ds[f"raw_{tag}_{model}"] = (("lead", "latitude", "longitude"), D)
            sec = PC.sector_mask()[None, :] & (PC.EOF_LAT[:, None] >= 20) & (PC.EOF_LAT[:, None] <= 60)
            print(f"  {model} {tag} ({int(sel.sum())} cases): sector-mean drift d1/d5/d10/d15 "
                  f"{D[0][sec].mean():+.2f}/{D[4][sec].mean():+.2f}/{D[9][sec].mean():+.2f}/{D[14][sec].mean():+.2f} m/s; "
                  f"applied |corr| d15 mean {np.abs(corr[14][sec]).mean():.2f}, max {np.abs(corr[14]).max():.2f}", flush=True)
        # indices on the basis month of each case (init + 7 d), as the live product reads them
        bm = np.array([(d + dt.timedelta(days=7)).month for d in inits])
        vdoy = np.array([[(d + dt.timedelta(days=L)).timetuple().tm_yday for L in LEADS] for d in inits])
        half = np.arange(n) < n // 2
        corr_cv = np.zeros_like(e_case)
        for fold in (half, ~half):
            c, _, _ = drift_estimate(e_case[~fold])
            corr_cv[fold] = c
        pf_raw = np.zeros((n, F.shape[1], len(LEADS), 2)); pf_cor = np.zeros_like(pf_raw)
        pa = np.zeros((n, len(LEADS), 2)); p0 = np.zeros((n, 2))
        for i in range(n):
            proj = ref.proj.sel(month=bm[i]).values
            clim = PC.harm_eval(ref.u_coef.values, vdoy[i])
            pf_raw[i] = PC.project(F[i] - clim[None], proj)
            pf_cor[i] = PC.project(F[i] - corr_cv[i][None] - clim[None], proj)
            pa[i] = PC.project(A[i] - clim, proj)
            c0 = PC.harm_eval(ref.u_coef.values, np.array([inits[i].timetuple().tm_yday]))[0]
            p0[i] = PC.project(A0[i] - c0, proj)
        season = np.array([bool(ref.in_season.sel(month=m)) for m in bm])
        res = {"cases": n, "season_cases": int(season.sum()), "first": inits[0].strftime("%Y-%m-%d"),
               "last": inits[-1].strftime("%Y-%m-%d"), "members": int(F.shape[1])}
        for tag, sel in (("season", season), ("all", np.ones(n, bool))):
            ii = np.flatnonzero(sel)
            if ii.size < MIN_CASES:
                continue
            m = {"n": int(ii.size)}
            for k, name in enumerate(("pc1", "pc2")):
                q = {k2: [] for k2 in ("bias", "bias_lo", "bias_hi", "r", "r_lo", "r_hi", "rmse", "rmse_lo", "rmse_hi",
                                       "rmse_raw", "pers_r", "pers_r_lo", "pers_r_hi")}
                for L in range(len(LEADS)):
                    b = (pf_raw[ii, :, L, k] - pa[ii, None, L, k]).mean(1)
                    x = pf_cor[ii, :, L, k].mean(1); xr_ = pf_raw[ii, :, L, k].mean(1); y = pa[ii, L, k]; y0 = p0[ii, k]
                    f_b = lambda idx: b[idx].mean()
                    f_r = lambda idx: np.corrcoef(x[idx], y[idx])[0, 1]
                    f_m = lambda idx: np.sqrt(np.mean((x[idx] - y[idx]) ** 2))
                    f_p = lambda idx: np.corrcoef(y0[idx], y[idx])[0, 1]
                    al = np.arange(ii.size)
                    for key, fn in (("bias", f_b), ("r", f_r), ("rmse", f_m), ("pers_r", f_p)):
                        q[key].append(fn(al))
                        lo, hi = block_boot_ci(fn, ii.size) if key != "rmse" or True else (np.nan, np.nan)
                        if key + "_lo" in q:
                            q[key + "_lo"].append(lo); q[key + "_hi"].append(hi)
                    q["rmse_raw"].append(float(np.sqrt(np.mean((xr_ - y) ** 2))))
                m[name] = {k2: np.round(np.array(v, float), 3).tolist() for k2, v in q.items()}
                m[name]["sd_obs"] = round(float(pa[ii, :, k].std()), 3)
            res[tag] = m
            for name in ("pc1", "pc2"):
                qq = m[name]
                print(f"    {model} {tag} {name} (n {ii.size}): r d3/5/7/10/14 " + "/".join(f"{qq['r'][L - 1]:.2f}" for L in (3, 5, 7, 10, 14))
                      + f"  [d7 {qq['r_lo'][6]:.2f}-{qq['r_hi'][6]:.2f}; d10 {qq['r_lo'][9]:.2f}-{qq['r_hi'][9]:.2f}]"
                      + f"; persistence d3/7 {qq['pers_r'][2]:.2f}/{qq['pers_r'][6]:.2f}"
                      + f"; bias d1/7/15 {qq['bias'][0]:+.2f}/{qq['bias'][6]:+.2f}/{qq['bias'][14]:+.2f}"
                      + f" [d15 {qq['bias_lo'][14]:+.2f},{qq['bias_hi'][14]:+.2f}]"
                      + f"; rmse d10 raw {qq['rmse_raw'][9]:.2f} -> {qq['rmse'][9]:.2f}", flush=True)
        # Apply the (all-season) field correction only if, cross-validated, it lowers the pooled squared error of both
        # indices over leads 1-15 in the phase season AND over the whole year.
        chg = {}
        for tag in ("season", "all"):
            if tag in res:
                raw = sum(x ** 2 for k in ("pc1", "pc2") for x in res[tag][k]["rmse_raw"])
                cor = sum(x ** 2 for k in ("pc1", "pc2") for x in res[tag][k]["rmse"])
                chg[tag] = round(100 * (cor / raw - 1), 1)
        res["cv_mse_change_pct"] = chg
        res["apply_drift"] = bool(chg) and all(v < 0 for v in chg.values())
        if f"corr_all_{model}" in ds:
            ds[f"corr_all_{model}"].attrs["apply"] = int(res["apply_drift"])
        print(f"  {model}: cross-validated MSE change {chg} -> drift correction {'APPLIED' if res['apply_drift'] else 'not applied'}", flush=True)
        doc["models"][model] = res
    (REF / "pacjet_skill.json").write_text(json.dumps(doc, separators=(",", ":")))
    ds.attrs["note"] = ("250 hPa u drift (member minus analysis, m/s) by lead from build_pacjet_hindcast.py, all 00Z runs: raw_* "
                        "mean, corr_* the correction (smoothed ~3 deg, Wiener-shrunk by D^2/(D^2+SE^2), SE from 10-day block means); "
                        "corr_*.attrs['apply'] = 1 when it lowered the cross-validated index error (Oct-Apr/May-Sep splits printed only)")
    ds.to_netcdf(REF / "pacjet_drift.nc", encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"wrote {REF / 'pacjet_skill.json'} and pacjet_drift.nc", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch"); f.add_argument("--model", choices=["aifs", "ifs"], required=True)
    f.add_argument("--start", default="20250701"); f.add_argument("--end", default="20260912"); f.add_argument("--every", type=int, default=2)
    g = sp.add_parser("analyses"); g.add_argument("--start", default="20250701"); g.add_argument("--end", default="20260928")
    sp.add_parser("stats")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "analyses": cmd_analyses, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
