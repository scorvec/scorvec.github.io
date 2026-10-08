#!/usr/bin/env python3
"""Validation of the TC shear product (tcshear.yml mode=verify; Actions only).

A. Against SHIPS. NHC publishes the SHIPS model's input file for every real-time run (ATCF lsdiag,
   ftp.nhc.noaa.gov/atcf/lsdiag/<yymmddhh><BB><nn><yy>_lsdiag.dat; this season only). At t = 0 it carries the storm
   position and, from the GFS, SHRD (200-850 hPa shear of the 200-800 km annulus mean wind), SHDC (the same with SHIPS's
   symmetric vortex removed, 0-500 km), SDDC (its heading), SHRS (850-500 hPa, 200-800 km), SHRG / SHGC (generalized
   shear). The same quantities are computed here from the GFS 0.25 deg fields (AWS S3) at SHIPS's own position.
   Which GFS run SHIPS uses at t = 0 is not stated in the file: it is chosen by SHRD, which involves no vortex removal
   (the run whose SHRD matches better on the first 60 cases: the same-cycle analysis or the previous cycle's 6 h
   forecast), and that choice is applied to everything else. SHRD then measures data agreement, SHDC vs ours measures
   the vortex-removal method (SHIPS: symmetric-vortex subtraction; ours: Galarneau & Davis 2013 inversion, r0 500 km).
B. Against the TDR. Every NOAA tail-Doppler-radar analysis with HRD's 2- and 6-km centres in the tdr-data mirror (the
   2-6 km tilt): the model tilt (850 -> 500 hPa vorticity centroids) in the GFS, IFS HRES and AIFS single analyses at the
   synoptic time nearest the radar analysis (within 3 h; a 6 h forecast when the analysis itself is not published).
Statistics: bias, RMSD, Pearson r with a 95 % bootstrap interval (2,000 resamples of cases), Spearman for tilt.
"""
from __future__ import annotations

import pyproj  # noqa: F401

import argparse
import datetime as dt
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "scripts" / "ecmwf"), str(ROOT / "scripts" / "mjo" / "src"), str(ROOT / "scripts" / "models")]
import sheardata as D                                                 # noqa: E402
import shearlib as S                                                  # noqa: E402

LSDIAG = "https://ftp.nhc.noaa.gov/atcf/lsdiag"
SHIPS_GS_LEVELS = (1000, 850, 700, 500, 400, 300, 250, 200, 150, 100)    # lsdiag.f gshear
RAW = "https://raw.githubusercontent.com/scorvec/scorvec.github.io"


def log(*a):
    print(*a, flush=True)


def lsdiag_list():
    txt = D.http_get(LSDIAG + "/", src="nhc").decode()
    names = sorted(set(re.findall(r'href="(\d{8}(?:AL|EP|CP)\d{4}_lsdiag\.dat)"', txt)))
    return [n for n in names if int(n[10:12]) < 50]                   # 80s = tests, 90s = invests


def parse_lsdiag(txt):
    lines = txt.splitlines()
    head = lines[0].split()
    rows = {}
    times = None
    for ln in lines[1:]:
        tag = ln[115:].split()[0] if len(ln) > 115 and ln[115:].split() else None
        vals = ln[:115].split()
        if tag == "TIME":
            times = [int(x) for x in vals]
            continue
        if tag and times:
            nums = [int(x) for x in vals]
            # rows that start at t=0 are left-padded: align them to the right end of TIME
            off = len(times) - len(nums)
            rows[tag] = {times[off + i]: (None if v == 9999 else v) for i, v in enumerate(nums)}
    t0 = dt.datetime.strptime(head[1] + head[2], "%y%m%d%H")
    lat = rows["LAT"][0] / 10.0; lon = -rows["LON"][0] / 10.0
    def g(k, scale=10.0):
        v = rows.get(k, {}).get(0)
        return None if v is None else v / scale
    return dict(sid=head[7], t0=t0, lat=lat, lon=lon, vmax=int(head[3]), shdc=g("SHDC"), sddc=g("SDDC", 1.0),
                shrd=g("SHRD"), shtd=g("SHTD", 1.0), shrs=g("SHRS"), shrg=g("SHRG"), shgc=g("SHGC"))


def uv_profile(b, la, lo):
    """Environmental (vortex-removed, 0-500 km) and annulus (200-800 km) wind on S.LEVELS from a box."""
    u, v = np.asarray(b["u"], float), np.asarray(b["v"], float)
    z, d = S.vort_div(u, v, b["lat"], b["lon"])
    pl = S.Plane(la, lo)
    U = pl.sample(b["lat"], b["lon"], np.stack([u, v, z, d]))
    return S.environment(U[0], U[1], U[2], U[3], pl)


def ships_quantities(E):
    env, ann, L = E["env"], E["ann"], list(S.LEVELS)
    kt = S.KT
    def lay(prof, top, bot):
        return S.vec(*S.layer_shear(prof, S.LEVELS, top, bot))
    sel = [L.index(p) for p in SHIPS_GS_LEVELS]
    out = dict(shdc=lay(env, 200, 850)[0] * kt, sddc=(lay(env, 200, 850)[1] + 180) % 360,
               shrd=lay(ann, 200, 850)[0] * kt, shtd=(lay(ann, 200, 850)[1] + 180) % 360,
               shrs=lay(ann, 500, 850)[0] * kt,
               shgc=S.generalized_shear(env[sel], SHIPS_GS_LEVELS) * kt,
               shrg=S.generalized_shear(ann[sel], SHIPS_GS_LEVELS) * kt)
    return {k: (round(float(v), 2) if np.isfinite(v) else None) for k, v in out.items()}


def gfs_uv(date, hh, step, cent):
    """u, v boxes only (the verify pass does not need t / q)."""
    url = f"{D.GFS_S3}/gfs.{date}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{int(step):03d}"
    recs = D.idx_records(url)
    if recs is None:
        return None
    lev = {f"{L} mb": L for L in S.LEVELS}
    want = [(s, e) for s, e, var, lv in recs if var in ("UGRD", "VGRD") and lv in lev]
    if len(want) < 2 * len(S.LEVELS):
        return None
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as tp:
        msgs = list(tp.map(lambda w: D.http_get(url, w, src="aws-s3"), want))
    parts = list(D.pool().map(D._work, [(m, {0: cent}, 12.0, None) for m in msgs if m]))
    boxes = D._assemble(parts)
    return boxes


def stats(a, b, n_boot=2000, seed=1, groups=None):
    """Bias, RMSD, Pearson r with 95 % bootstrap intervals. None/NaN pairs dropped. groups: resample whole groups
    (cases that share one model value, e.g. several radar analyses at one synoptic time) instead of single cases."""
    a = np.array([np.nan if x is None else x for x in a], float)
    b = np.array([np.nan if x is None else x for x in b], float)
    g = np.array(groups if groups is not None else np.arange(len(a)))
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, g = a[ok], b[ok], g[ok]
    n = len(a)
    if n < 5:
        return dict(n=int(n))
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    rng = np.random.default_rng(seed)
    d = a - b
    rs, bb = [], []
    for _ in range(n_boot):
        i = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
        bb.append(d[i].mean())
        if np.std(a[i]) > 0 and np.std(b[i]) > 0:
            rs.append(np.corrcoef(a[i], b[i])[0, 1])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    blo, bhi = np.percentile(bb, [2.5, 97.5])
    from scipy.stats import spearmanr
    return dict(n=int(n), n_groups=int(len(ug)), bias=round(float(d.mean()), 2), bias_ci=[round(float(blo), 2), round(float(bhi), 2)],
                rmsd=round(float(np.sqrt((d ** 2).mean())), 2), mad=round(float(np.median(np.abs(d))), 2),
                r=round(float(np.corrcoef(a, b)[0, 1]), 3), r_ci=[round(float(lo), 3), round(float(hi), 3)],
                spearman=round(float(spearmanr(a, b).statistic), 3))


def ang(a, b):
    return ((a - b + 180.0) % 360.0) - 180.0


def part_ships(max_cases=None):
    names = lsdiag_list()
    log(f"SHIPS lsdiag files (named storms, this season): {len(names)}")
    cases = []
    for n in names[:max_cases] if max_cases else names:
        try:
            c = parse_lsdiag(D.http_get(f"{LSDIAG}/{n}", src="nhc").decode())
        except Exception as e:                                         # noqa: BLE001
            log(f"  {n}: {e}"); continue
        if c["shdc"] is None or c["shrd"] is None:
            continue
        c["file"] = n
        cases.append(c)
    # which GFS run: decided on SHRD over the first 60 cases
    def ours(c, variant):
        t = c["t0"] if variant == "anl" else c["t0"] - dt.timedelta(hours=6)
        step = 0 if variant == "anl" else 6
        bx = gfs_uv(f"{t:%Y%m%d}", f"{t:%H}", step, [((0, "c"), c["lat"], c["lon"])])
        if not bx or (0, "c") not in bx:
            return None
        return ships_quantities(uv_profile(bx[(0, "c")], c["lat"], c["lon"]))
    pick = {}
    for v in ("anl", "f06"):
        diffs = []
        for c in cases[:60]:
            q = ours(c, v)
            c.setdefault("ours", {})[v] = q
            if q and q["shrd"] is not None:
                diffs.append(q["shrd"] - c["shrd"])
        pick[v] = float(np.sqrt(np.mean(np.square(diffs)))) if diffs else np.inf
        log(f"  GFS {v}: SHRD rmsd vs SHIPS {pick[v]:.2f} kt over {len(diffs)} cases")
    var = min(pick, key=pick.get)
    log(f"  -> SHIPS t=0 matches the GFS {var}")
    t0 = time.time()
    for i, c in enumerate(cases):
        if var not in c.get("ours", {}):
            c.setdefault("ours", {})[var] = ours(c, var)
        if i % 50 == 0:
            log(f"  {i}/{len(cases)}  {time.time() - t0:.0f} s  {D.LEDGER.report()}")
    rows = []
    for c in cases:
        q = (c.get("ours") or {}).get(var)
        if not q:
            continue
        rows.append(dict(file=c["file"], sid=c["sid"], t0=f"{c['t0']:%Y-%m-%dT%HZ}", lat=c["lat"], lon=c["lon"], vmax=c["vmax"],
                         ships={k: c[k] for k in ("shdc", "sddc", "shrd", "shtd", "shrs", "shrg", "shgc")}, ours=q))
    summ = {}
    for k in ("shrd", "shdc", "shrs", "shrg", "shgc"):
        summ[k] = stats([r["ours"][k] for r in rows], [r["ships"][k] for r in rows])
    # vortex-removal effect: (ours SHDC - ours SHRD) vs (SHIPS SHDC - SHIPS SHRD)
    full = [r for r in rows if None not in (r["ours"]["shdc"], r["ours"]["shrd"], r["ships"]["shdc"], r["ships"]["shrd"])]
    summ["removal_effect"] = stats([r["ours"]["shdc"] - r["ours"]["shrd"] for r in full],
                                   [r["ships"]["shdc"] - r["ships"]["shrd"] for r in full])
    dd = [ang(r["ours"]["sddc"], r["ships"]["sddc"]) for r in rows
          if r["ours"]["sddc"] is not None and r["ships"]["sddc"] is not None and (r["ships"]["shdc"] or 0) >= 10]
    summ["sddc_diff_deg_shear_ge10kt"] = dict(n=len(dd), median_abs=round(float(np.median(np.abs(dd))), 1) if dd else None,
                                              p90_abs=round(float(np.percentile(np.abs(dd), 90)), 1) if dd else None)
    # by intensity (the vortex matters most for strong storms)
    for lab, lo_, hi_ in (("vmax_lt64", 0, 64), ("vmax_ge64", 64, 999)):
        sub = [r for r in rows if lo_ <= r["vmax"] < hi_]
        summ[f"shdc_{lab}"] = stats([r["ours"]["shdc"] for r in sub], [r["ships"]["shdc"] for r in sub])
    return dict(gfs_run=var, gfs_pick_rmsd={k: round(v, 2) for k, v in pick.items()}, n_files=len(names), rows=rows, summary=summ)


def synoptic_source(model, t):
    """(cycle datetime, step) of the newest run that has t as an analysis, else as a 6 h forecast."""
    import cycles
    for back, step in ((0, 0), (6, 6), (12, 12)):
        c = t - dt.timedelta(hours=back)
        names = cycles._list_names(f"{c:%Y%m%d}/{c:%H}z/{model}/0p25/oper/{c:%Y%m%d%H}0000-{step}h-oper-fc.grib2") or []
        if names:
            return c, step
    return None, None


def part_tdr():
    idx = json.loads(D.http_get(f"{RAW}/tdr-data/index.json", src="obs"))
    tdr = []
    for k, m in idx["missions"].items():
        if not m.get("n_ctr"):
            continue
        fl = json.loads(D.http_get(f"{RAW}/tdr-data/{m['file']}", src="obs"))
        for c in fl.get("centres") or []:
            cc = c.get("c") or {}
            if "tilt" in c and "2" in cc and "6" in cc:
                tdr.append(dict(t=int(c["t"]), sid=fl["storm"], flight=fl["id"], tilt=c["tilt"][0], dir=c["tilt"][1] % 360,
                                c2=cc["2"], c6=cc["6"]))
    log(f"TDR analyses with 2- and 6-km centres: {len(tdr)}")
    syn = {}
    for r in tdr:
        t = dt.datetime.fromtimestamp(r["t"], dt.timezone.utc).replace(tzinfo=None)
        ts = (t + dt.timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
        ts = ts.replace(hour=(ts.hour // 6) * 6)
        if abs((t - ts).total_seconds()) > 3 * 3600:
            continue
        r["syn"] = f"{ts:%Y-%m-%dT%HZ}"
        syn.setdefault((ts, r["sid"]), []).append(r)
    out = []
    oi_cache = {}
    for (ts, sid), rs in sorted(syn.items()):
        la, lo = rs[0]["c2"]
        mod = {}
        for model in ("gfs", "ifs", "aifs-single"):
            try:
                if model == "gfs":
                    bx = D.gfs_boxes(f"{ts:%Y%m%d}", f"{ts:%H}", 0, [((0, "c"), la, lo)])
                    src = f"{ts:%Y%m%d %H}Z f000"
                else:
                    c, step = synoptic_source(model, ts)
                    if c is None:
                        continue
                    bx = D.ecmwf_boxes(model, f"{c:%Y%m%d}", f"{c:%H}", step, "fc", "oper", {0: [((0, "c"), la, lo)]})
                    src = f"{c:%Y%m%d %H}Z +{step}h"
                b = bx.get((0, "c")) if bx else None
                if b is None or not D.complete(b):
                    continue
                cen = S.find_centre(b, la, lo, r_search_km=200.0)
                if cen is None:
                    continue
                z = S.vort_div(np.asarray(b["u"], float), np.asarray(b["v"], float), b["lat"], b["lon"])[0]
                L = list(S.LEVELS)
                c850 = S.centroid(b["lat"], b["lon"], z[L.index(850)], cen[0], cen[1])
                c500 = S.centroid(b["lat"], b["lon"], z[L.index(500)], *(c850[:2] if c850 else cen[:2]))
                if not (c850 and c500):
                    continue
                x, y = S.offset_km(c850[0], c850[1], c500[0], c500[1])
                E = uv_profile(b, cen[0], cen[1])
                sh = S.vec(*S.layer_shear(E["env"], S.LEVELS, 200, 850))
                mod[model] = dict(src=src, tilt=round(float(math.hypot(x, y)), 1), dir=round(S.bearing_of(float(x), float(y))),
                                  c850=[round(c850[0], 3), round(c850[1], 3)], deep=[round(sh[0], 1), round(sh[1])],
                                  pmin=round(cen[2], 1))
            except Exception as e:                                     # noqa: BLE001
                log(f"  {ts} {model}: {e}")
        for r in rs:
            out.append(dict(r, models=mod))
        log(f"  {ts:%Y-%m-%d %HZ} {sid}: TDR " + ", ".join(f"{r['tilt']:.0f} km@{r['dir']}" for r in rs) + " | "
            + ", ".join(f"{k} {v['tilt']:.0f} km@{v['dir']}" for k, v in mod.items()))
    summ = {}
    for model in ("gfs", "ifs", "aifs-single"):
        sub = [r for r in out if model in r["models"]]
        summ[model] = dict(magnitude=stats([r["models"][model]["tilt"] for r in sub], [r["tilt"] for r in sub],
                                           groups=[r["syn"] + r["sid"] for r in sub]))
        big = [r for r in sub if r["tilt"] >= 20]
        dd = [ang(r["models"][model]["dir"], r["dir"]) for r in big]
        summ[model]["direction_tdr_ge20km"] = dict(n=len(dd), median_abs=round(float(np.median(np.abs(dd))), 1) if dd else None)
        # can the model tell a tilted vortex from an aligned one? hit rate for TDR tilt >= 30 km vs model >= 30 km
        a = np.array([r["tilt"] >= 30 for r in sub]); b = np.array([r["models"][model]["tilt"] >= 30 for r in sub])
        summ[model]["tilted_ge30km"] = dict(n_tdr=int(a.sum()), n_model=int(b.sum()), both=int((a & b).sum()), n=len(sub))
    return dict(rows=out, summary=summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assets/sst/data/tcshear_validation.json"))
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--skip-ships", action="store_true")
    ap.add_argument("--skip-tdr", action="store_true")
    a = ap.parse_args()
    T0 = time.time()
    doc = dict(v=1, built=f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ}")
    def save():
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False, default=str))
    if not a.skip_tdr:
        doc["tdr"] = part_tdr()
        log(json.dumps(doc["tdr"]["summary"], indent=1))
        save()
    if not a.skip_ships:
        doc["ships"] = part_ships(a.max_cases or None)
        log(json.dumps(doc["ships"]["summary"], indent=1))
    doc["runtime_s"] = round(time.time() - T0)
    doc["bytes"] = D.LEDGER.report()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False, default=str))
    log(f"done in {time.time() - T0:.0f} s {D.LEDGER.report()}")


if __name__ == "__main__":
    main()
