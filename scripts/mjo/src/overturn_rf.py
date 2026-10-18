#!/usr/bin/env python3
"""GEFS weeks 3-5 of the Hadley and Walker forecasts: the drift of GEFS's 35-day runs against ERA5 there, measured on
the GEFSv12 reforecast, and the skill that decides which weeks are shown.

The 15-day forecasts correct each model by its mean error against ERA5 measured on its own archived runs (overturn.py).
GEFS runs to day 35 at 00 UTC, and NOAA's reforecast of the same model (GEFSv12, noaa-gefs-retrospective, anonymous)
has a 35-day start every Wednesday of 2000-2019, so for its weeks 3-5 the drift comes from twenty years of it:

  REFORECAST  every Wednesday start, gefs_reforecast.MEMBERS (control + 3), the 00 UTC u and v of lead days 15-35 on
              the 13 levels (the Days:10-35 files, 0.5 deg, byte-ranged through their .idx), as WEEKS means of the
              zonal-mean v (overturn.zonal_v, averaged into build_hadley_clim's 1.5 deg bands) and of the 5S-5N
              divergent zonal wind (overturn.eq_ud). Both are linear in the wind, so the week's mean wind gives the
              week's.
  TRUTH       ERA5 at 00 UTC on the same valid days (WeatherBench2, 1.5 deg: the same bands), reduced the same way.
  DRIFT       a week's error, the members' mean less ERA5, regressed per point on the start's common-year day (mean
              and annual harmonic) and shrunk by its signal-to-noise (hadley_slp_ref.fit_drift and eval_drift).
  SKILL       the members less the drift (refitted without the start's year), as the Hadley and Walker
              streamfunctions (mmsf.streamfunction on build_hadley_clim's bands; walker.streamfunction) and as the
              cells' metrics (the series' four Hadley ones, walker's two), against the ERA5 1991-2020 normals the live
              loops and series use; the mean-squared-error skill score of the members' mean, its error taken down by
              (1/len(MEMBERS) - 1/LIVE) times the members' sample variance (the expected error of the live 31-member
              mean), an interval and one-sided p-value from a bootstrap over whole years of starts, Benjamini-Hochberg
              at hadley_slp_ref.FDR_Q over every target and week; a target shows a week when it survives and reaches
              hadley_slp_ref.MIN_SKILL, from week 3 up to the first week that fails.

    python src/overturn_rf.py archive --year 2005 [--limit 2] --out rf_2005.npz   # one year of starts
    python src/overturn_rf.py truth --year 2005 --out era5_2005.npz     # ERA5 for that year's valid weeks
    python src/overturn_rf.py verify parts/*.npz                         # -> DRIFT (release) and SKILL (committed)
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gefs"))
import build_hadley_clim as B
import build_walker_metrics as W
import gefs_reforecast as RF
import hadley_enso as E
import hadley_slp_ref as R
import mmsf
import overturn as O
import walker
from doycal import noleap_doy

REF_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
DRIFT = REF_DIR / "overturn_rf_drift.npz"  # release overturn-ref-v1
SKILL = REF_DIR / "overturn_rf_skill.json"  # committed: the weeks the loops and series show
LEVELS = mmsf.LEVELS
WEEKS = ((15, 21), (22, 28), (29, 35))  # lead days, 00 UTC of each (gefs_reforecast's weeks 3-5)
DAYS = range(WEEKS[0][0], WEEKS[-1][1] + 1)
LIVE = 31  # GEFS's live members: the control and 30
H_KEYS, W_KEYS = E.KEYS, W.KEYS
P_PA, P_HPA = np.asarray(LEVELS, float) * 100.0, np.asarray(LEVELS, float)
HADLEY_LAT = np.abs(B.LAT15) <= 40.0  # the Hadley streamfunction's scored rows


# ── the reforecast ──────────────────────────────────────────────────────────────────────────────────────────────────
def _url(start: str, mem: str, var: str) -> str:
    return f"{RF.S3}/{start[:4]}/{start}00/{mem}/Days:10-35/{var}_pres_{start}00_{mem}.grib2"


def one_member(start: str, mem: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(vbar (week, lev, LAT15), ud (week, lev, 173)) of one member of one start, or None when a message is missing."""
    jobs = []
    for var in ("ugrd", "vgrd"):
        url = _url(start, mem, var)
        idx = {(e[2], e[3]): (e[0], e[1]) for e in RF.index(url)}
        for d in DAYS:
            for k, lev in enumerate(LEVELS):
                hit = idx.get((f"{lev} mb", f"{24 * d} hour fcst"))
                if hit is None:
                    print(f"  {start} {mem}: no {var} {lev} hPa at day {d}", flush=True)
                    return None
                jobs.append((var, d, k, url, hit))
    lat = np.linspace(90.0, -90.0, 361)
    lon = np.arange(720) * 0.5
    wind = {v: np.zeros((len(WEEKS), len(LEVELS), lat.size, lon.size)) for v in ("ugrd", "vgrd")}
    with ThreadPoolExecutor(24) as ex:
        for (var, d, k, *_), b in zip(jobs, ex.map(lambda j: RF.get(j[3], j[4]), jobs), strict=True):
            f = RF.decode(b)
            for w, (a, z) in enumerate(WEEKS):
                if a <= d <= z:
                    wind[var][w, k] += f / (z - a + 1)
    vb = np.stack([[O.zonal_v(wind["vgrd"][w, k], lat) for k in range(len(LEVELS))] for w in range(len(WEEKS))])
    vb = B.band_v(vb, O.LAT)
    ud = np.stack(
        [
            [O.eq_ud(wind["ugrd"][w, k], wind["vgrd"][w, k], lat, lon) for k in range(len(LEVELS))]
            for w in range(len(WEEKS))
        ]
    )
    return vb.astype(np.float32), ud.astype(np.float32)


def archive(year: int, limit: int | None = None) -> dict:
    """Every Wednesday start of the year (the first `limit` of them, for a trial) with all MEMBERS complete:
    (start, member, week, lev, ...) vbar and ud."""
    keep, vbs, uds = [], [], []
    for start in list(RF.wednesdays(year))[:limit]:
        t0 = time.time()
        got = [one_member(start, m) for m in RF.MEMBERS]
        if any(g is None for g in got):
            print(f"  {start}: incomplete, skipped", flush=True)
            continue
        keep.append(pd.Timestamp(start))
        vbs.append(np.stack([g[0] for g in got]))
        uds.append(np.stack([g[1] for g in got]))
        print(f"  {start}: {len(got)} members in {time.time() - t0:.0f} s", flush=True)
    return {"start": np.array(keep, "datetime64[D]"), "vbar": np.array(vbs), "ud": np.array(uds)}


def truth(year: int) -> dict:
    """ERA5's daily 00 UTC vbar and ud on the valid days of the year's starts (weeks 3-5)."""
    starts = [pd.Timestamp(s) for s in RF.wednesdays(year)]
    t0, t1 = starts[0] + pd.Timedelta(days=DAYS[0]), starts[-1] + pd.Timedelta(days=DAYS[-1])
    v, u = B.fetch_wb2(t0, t1), W.fetch(t0, t1, "wb2")
    if not np.array_equal(v["date"], u["date"]):
        raise SystemExit("the zonal-mean v and the divergent u cover different days")
    return {"day": v["date"], "vbar": v["vbar"].astype(np.float32), "ud": u["ud"].astype(np.float32)}


# ── verification ────────────────────────────────────────────────────────────────────────────────────────────────────
def load(paths) -> tuple[dict, dict]:
    """The reforecast {start, vbar, ud} and ERA5 {day, vbar, ud}, merged from the year parts."""
    rf, tr = {"start": [], "vbar": [], "ud": []}, {"day": [], "vbar": [], "ud": []}
    for p in paths:
        with np.load(p) as f:
            dest = rf if "start" in f.files else tr
            for k in dest:
                dest[k].append(f[k])
    rf = {k: np.concatenate(v) for k, v in rf.items()}
    tr = {k: np.concatenate(v) for k, v in tr.items()}
    return rf, tr


def weekly_truth(tr: dict, starts: pd.DatetimeIndex) -> dict:
    """ERA5's (start, week, lev, ...) means on each start's valid weeks, NaN where a day is missing."""
    pos = {pd.Timestamp(d): i for i, d in enumerate(tr["day"])}
    out = {k: np.full((len(starts), len(WEEKS), *tr[k].shape[1:]), np.nan, np.float32) for k in ("vbar", "ud")}
    for i, s in enumerate(starts):
        for w, (a, z) in enumerate(WEEKS):
            rows = [pos.get(s + pd.Timedelta(days=d)) for d in range(a, z + 1)]
            if all(r is not None for r in rows):
                for k in out:
                    out[k][i, w] = tr[k][rows].mean(0)
    return out


def _fits(err: np.ndarray, starts: pd.DatetimeIndex, low: np.ndarray):
    """Per week: the drift fitted on every start and each start's drift refitted without its year."""
    year = starts.year.values
    cv = np.full_like(err, np.nan)
    fits = []
    for w in range(len(WEEKS)):
        ok = np.isfinite(err[:, w, 0])
        fits.append(R.fit_drift(err[ok, w], starts[ok], low))
        for y in np.unique(year[ok]):
            tr, te = ok & (year != y), ok & (year == y)
            cv[te, w] = R.eval_drift(R.fit_drift(err[tr, w], starts[tr], low), R.doy_of(starts[te]))
    return fits, cv


def week_mid(s: pd.Timestamp, w: int) -> float:
    """The common-year day of the middle of week w of the run started on s."""
    a, z = WEEKS[w]
    return float(noleap_doy(pd.DatetimeIndex([s + pd.Timedelta(days=(a + z) / 2)]))[0])


def fields(vb: np.ndarray, ud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Psi (lev, LAT15) and Psi_W (lev, 173) of a week's vbar (on LAT15) and ud."""
    return mmsf.streamfunction(vb, P_PA, B.LAT15), walker.streamfunction(ud, P_PA)


def metrics(psi: np.ndarray, psw: np.ndarray) -> dict:
    """The series' metrics of a week's streamfunctions: the Hadley cells' four, the Walker cell's two."""
    out = B.sample_metrics(psi, P_HPA, B.LAT15)
    return {k: out[k] for k in H_KEYS} | {
        "pac": walker.pac_index(psw, P_HPA, W.LON),
        "asc_lon": walker.ascent_lon(psw, P_HPA, W.LON),
    }


def verify(paths, seed: int = 1) -> dict:
    rf, tr = load(paths)
    starts = pd.DatetimeIndex(rf["start"])
    T = weekly_truth(tr, starts)
    n_m = rf["vbar"].shape[1]
    cm = xr.open_dataarray(mmsf.CLIM)
    cw = xr.open_dataarray(walker.CLIM).interp(longitude=W.LON, kwargs={"fill_value": None})
    cl_h, cl_w = xr.open_dataset(B.OUT), xr.open_dataset(W.CLIM_OUT)
    out = {}
    fits = {}
    for k in ("vbar", "ud"):
        x = rf[k].reshape(len(starts), n_m, len(WEEKS), -1)
        err = x.mean(1) - T[k].reshape(len(starts), len(WEEKS), -1)
        fits[k], out[k] = _fits(err, starts, np.ones(err.shape[-1], bool))
        print(f"  {k}: drift fitted on {np.isfinite(err[:, 0, 0]).sum()} starts", flush=True)
    targets = ["psi", "psi_w", *H_KEYS, *W_KEYS]
    S = {k: np.full((len(starts), len(WEEKS), len(targets)), np.nan) for k in ("fc", "raw", "ref")}
    fair = 1.0 / n_m - 1.0 / LIVE
    for i, s in enumerate(starts):
        for w in range(len(WEEKS)):
            if np.isnan(T["vbar"][i, w]).any() or np.isnan(T["ud"][i, w]).any():
                continue
            mid = week_mid(s, w)
            nm, nw = mmsf.clim_psi(cm, mid), walker.clim_psi(cw, mid)
            o_m, o_w = fields(T["vbar"][i, w], T["ud"][i, w])
            o = metrics(o_m, o_w)
            cv_v = out["vbar"][i, w].reshape(rf["vbar"].shape[3:])
            cv_u = out["ud"][i, w].reshape(rf["ud"].shape[3:])
            mem = [fields(rf["vbar"][i, m, w] - cv_v, rf["ud"][i, m, w] - cv_u) for m in range(n_m)]
            raw = [fields(rf["vbar"][i, m, w], rf["ud"][i, m, w]) for m in range(n_m)]
            for f_list, kind in ((mem, "fc"), (raw, "raw")):
                pm, pw = np.stack([f[0] for f in f_list]), np.stack([f[1] for f in f_list])
                adj = fair if kind == "fc" else 0.0
                S[kind][i, w, 0] = (((pm.mean(0) - o_m) ** 2 - adj * pm.var(0, ddof=1))[:, HADLEY_LAT]).sum()
                S[kind][i, w, 1] = ((pw.mean(0) - o_w) ** 2 - adj * pw.var(0, ddof=1)).sum()
                mets = [metrics(*f) for f in f_list]
                for t, key in enumerate((*H_KEYS, *W_KEYS), start=2):
                    v = np.array([m[key] for m in mets], float)
                    if np.isfinite(v).all() and np.isfinite(o[key]):
                        S[kind][i, w, t] = (v.mean() - o[key]) ** 2 - adj * v.var(ddof=1)
            S["ref"][i, w, 0] = ((o_m - nm) ** 2)[:, HADLEY_LAT].sum()
            S["ref"][i, w, 1] = ((o_w - nw) ** 2).sum()
            for t, key in enumerate(H_KEYS, start=2):
                S["ref"][i, w, t] = (o[key] - E.clim_at(cl_h, key, [mid], "mean")[0]) ** 2
            for t, key in enumerate(W_KEYS, start=2 + len(H_KEYS)):
                S["ref"][i, w, t] = (o[key] - E.clim_at(cl_w, key, [mid], "mean")[0]) ** 2
        if s.month == 1 and s.day <= 7:
            print(f"  scored {s.year}", flush=True)
    years = starts.year.values
    names, block = np.unique(years, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(x, minlength=names.size) for x in rng.integers(0, names.size, (R.BOOT, names.size))])
    res = {}
    for kind in ("fc", "raw"):
        tests = [
            R.skill_test(S[kind][..., t], S["ref"][..., t], block, names.size, counts) for t in range(len(targets))
        ]
        res[kind] = [np.stack(x) for x in zip(*tests, strict=True)]  # msss, lo, hi, p: (target, week)
    passed = R.bh(res["fc"][3].ravel()).reshape(len(targets), len(WEEKS))
    doc = {
        "members": n_m,
        "live_members": LIVE,
        "starts": len(starts),
        "first": f"{starts[0]:%Y-%m-%d}",
        "last": f"{starts[-1]:%Y-%m-%d}",
        "weeks": [list(w) for w in WEEKS],
        "score": "MSSS against the ERA5 1991-2020 normal; the members' mean error made fair for the live ensemble",
        "bootstrap": R.BOOT,
        "blocks": "years of starts",
        "cross_validation": "drift refitted without the start's year",
        "fdr_q": R.FDR_Q,
        "min_skill": R.MIN_SKILL,
        "targets": {},
    }
    for t, name in enumerate(targets):
        ok = passed[t] & (res["fc"][0][t] >= R.MIN_SKILL)
        show = 0
        while show < len(WEEKS) and ok[show]:
            show += 1
        doc["targets"][name] = {
            "show": show,  # weeks 3 .. 3 + show - 1
            "msss": _r3(res["fc"][0][t]),
            "ci": [_r3(res["fc"][1][t]), _r3(res["fc"][2][t])],
            "p": [round(float(x), 4) for x in res["fc"][3][t]],
            "msss_no_drift": _r3(res["raw"][0][t]),
            "scored": [int(x) for x in np.isfinite(S["fc"][..., t]).sum(0)],
        }
    SKILL.write_text(json.dumps(doc, indent=1) + "\n")
    save = {}
    for k, fl in fits.items():
        save[f"{k}_coef"] = np.stack([f["coef"] for f in fl])
        save[f"{k}_cov"] = np.stack([f["cov"] for f in fl])
        save[f"{k}_s2"] = np.stack([f["s2"] for f in fl])
        save[f"{k}_kappa"] = np.array([f["kappa"] for f in fl])
    np.savez_compressed(DRIFT, **{k: v.astype(np.float32) for k, v in save.items()})
    for name, t in doc["targets"].items():
        ms = " ".join("  .  " if v is None else f"{v:+.2f}" for v in t["msss"])
        nd = " ".join("  .  " if v is None else f"{v:+.2f}" for v in t["msss_no_drift"])
        print(f"  {name:13s} weeks 3-5 {ms} | no drift {nd} | p {t['p']} | show {t['show']}")
    return doc


def _r3(x):
    return [round(float(v), 3) if np.isfinite(v) else None for v in np.atleast_1d(x)]


def drift_at(doy: float) -> dict | None:
    """The drift at a start's common-year day: {"vbar": (week, lev, LAT15), "ud": (week, lev, 173)}; None without a
    fit."""
    if not DRIFT.exists():
        return None
    f = dict(np.load(DRIFT))
    shape = {"vbar": (len(LEVELS), B.LAT15.size), "ud": (len(LEVELS), O.NLON_UD)}
    out = {}
    for k, shp in shape.items():
        rows = []
        for w in range(len(WEEKS)):
            fit = {
                "coef": f[f"{k}_coef"][w],
                "cov": f[f"{k}_cov"][w],
                "s2": f[f"{k}_s2"][w],
                "kappa": f[f"{k}_kappa"][w],
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rows.append(R.eval_drift(fit, [doy])[0].reshape(shp))
        out[k] = np.stack(rows).astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("archive", "truth"):
        s = sub.add_parser(name)
        s.add_argument("--year", type=int, required=True)
        s.add_argument("--out", required=True)
        if name == "archive":
            s.add_argument("--limit", type=int)
    s = sub.add_parser("verify")
    s.add_argument("parts", nargs="+")
    a = ap.parse_args()
    if a.cmd == "archive":
        np.savez_compressed(a.out, **archive(a.year, a.limit))
    elif a.cmd == "truth":
        np.savez_compressed(a.out, **truth(a.year))
    else:
        verify(a.parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
