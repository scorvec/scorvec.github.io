#!/usr/bin/env python3
"""Do the C3S seasonal systems have skill for the MONTHLY-MEAN polar vortex wind? (u 60°N, 10 hPa and 100 hPa)

User 2026-10-08: "would be interesting to see if any of these models have any skill for the 10mb winds first" —
"Monthly. Not daily." — and (same day, before anything was computed) "also test 100 hPa". Analysis only: nothing
here goes to the site. Daily data is out of scope (only a monthly signal would justify a daily follow-up).

PRE-REGISTRATION (fixed 2026-10-08 before any model data were fetched; not to be changed after seeing results)
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
DATA. CDS `seasonal-monthly-pressure-levels`, product_type monthly_mean, u_component_of_wind at 10 and 100 hPa,
hindcast years 1993–2016 (24 winters, 1993/94–2016/17), start months Aug–Dec, leadtime_month 1–6, area = the 60°N
ring on a 2° grid (the pattern of seas5_ssw._monthly_u100). Model value = zonal mean of that ring. Every CDS request
is sequential (one key; the workflow shares the `seas5` concurrency group). SEAS5's October-start 100 hPa hindcast is
NOT re-fetched: it is the release table seas5-ref-v1/hc_ssw_u100_10.npz (seas5_ssw.derive_u100, same request).

SYSTEMS (one per centre: the current operational version with a full 1993–2016 hindcast at 10 hPa in the monthly
dataset, CDS constraints checked 2026-10-08):
  ECMWF SEAS5 (51) — the reference; UKMO 610 (hindcast only for start months Apr–Oct: no Nov/Dec starts);
  Météo-France 9; DWD 22; CMCC 35; ECCC CanSIPS = systems 4 (GEM5-NEMO) + 5 (CanESM5) treated as ONE centre's
  two-model ensemble: each component is anomalised against its own hindcast climatology and the centre's anomaly is
  the equal-weight mean of the two (the components are reported separately as description only); JMA 3; BoM 2.
  NCEP CFSv2 has no 10 hPa u in the monthly dataset (constraints) and is excluded. 8 centres.
  Multi-model mean (MMM) = equal-weight mean over the centres of their ensemble-mean anomalies (m/s, not standardised).

TRUTH. 10 hPa: MERRA-2 zonal-mean u at 60°N (scripts/strat/reference/strat_history.nc `u10`, the series
seas5_ssw_skill uses), calendar-month means of the daily values. 100 hPa: no MERRA-2 60°N 100 hPa series is held, so
ERA5 monthly means (CDS reanalysis-era5-pressure-levels-monthly-means, the same 60°N ring, zonal mean). ERA5 at 10 hPa
(same request) is the check on the 10 hPa truth: their agreement, and the family re-scored on ERA5 (descriptive).

STATISTIC. For each (system, start month, target): the ensemble-mean anomaly against the system's own lead-dependent
hindcast climatology, leave-one-year-out (LOYO); the truth anomaly LOYO the same way. DJF = the mean of the Dec, Jan,
Feb monthly means. r = Pearson correlation of the two anomaly series (24 winters). p = ONE-SIDED (r > 0) permutation
test over years (20,000 permutations). 95 % CI = year bootstrap (10,000; each winter is one block). Also MSSS against
the climatology (LOYO: 1 − Σ(f′−o′)² / Σo′²) with a bootstrap CI, and the vortex mean bias = model hindcast
climatology − truth climatology (m/s, 95 % bootstrap CI over years).

FAMILIES (Holm, α = 0.05, within each family, never across):
  F10  — 10 hPa: October start; targets Dec, Jan, Feb, DJF; 8 centres + MMM = 36 tests.
  F100 — 100 hPa: identical design = 36 tests.
POSITIVE CONTROLS (separate families per level, Holm within): November from the October start (leadtime_month 2) and
November from the November start (leadtime_month 1), every centre + MMM (UKMO has no November start).
DESCRIPTIVE (clearly labelled, Benjamini–Hochberg FDR 10 % over each level's grid): every start month Aug–Dec ×
target month Nov–Mar reachable within 6 lead months, per centre + MMM.
POWER: the correlation detectable with 80 % power at n = 24, one-sided α = 0.05 (Fisher z), and at the Holm
threshold of the family's smallest p (α/36).

    python c3s_u10_skill.py fetch [--budget-min 300]   # Actions only: CDS → raw GRIB → table, uploaded per unit
    python c3s_u10_skill.py build                      # statistics, JSON, figure (needs every table)
"""
from __future__ import annotations

import pyproj  # noqa: F401  first, before eccodes/xarray load another libproj
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
DIR = Path(os.environ.get("C3SU10_DIR", HERE / "data" / "c3s_u10"))
RAW = DIR / "raw"
OUT = Path(os.environ.get("C3SU10_OUT", DIR / "out"))
TAG = os.environ.get("C3SU10_TAG", "")                  # release to upload each unit to (Actions)
SEAS5_REF = Path(os.environ.get("SEAS5_REF_DIR", HERE / "data" / "seas5" / "ref"))

YEARS = list(range(1993, 2017))
N = len(YEARS)
LEVELS = (10, 100)
STARTS = (8, 9, 10, 11, 12)
TARGETS = (11, 12, 1, 2, 3)
AREA, GRID = [60, -180, 60, 180], [2.0, 2.0]
NPERM, NBOOT = 20000, 10000
RNG = np.random.default_rng(20261008)

# (key, centre, system, label, start months held in the CDS hindcast)
SYSTEMS = [
    ("ecmwf_51", "ecmwf", "51", "ECMWF SEAS5 (51)", STARTS),
    ("ukmo_610", "ukmo", "610", "UKMO 610", (8, 9, 10)),
    ("mf_9", "meteo_france", "9", "Météo-France 9", STARTS),
    ("dwd_22", "dwd", "22", "DWD 22", STARTS),
    ("cmcc_35", "cmcc", "35", "CMCC 35", STARTS),
    ("eccc_4", "eccc", "4", "ECCC 4 (GEM5-NEMO)", STARTS),
    ("eccc_5", "eccc", "5", "ECCC 5 (CanESM5)", STARTS),
    ("jma_3", "jma", "3", "JMA 3", STARTS),
    ("bom_2", "bom", "2", "BoM 2", STARTS),
]
SYS = {s[0]: s for s in SYSTEMS}
CENTRES = [("ECMWF", ["ecmwf_51"]), ("UKMO", ["ukmo_610"]), ("Météo-France", ["mf_9"]), ("DWD", ["dwd_22"]),
           ("CMCC", ["cmcc_35"]), ("ECCC", ["eccc_4", "eccc_5"]), ("JMA", ["jma_3"]), ("BoM", ["bom_2"])]
MMM = "Multi-model mean"
PRIMARY_TARGETS = ("Dec", "Jan", "Feb", "DJF")
MONTH_NAME = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}


# ── fetch units ──────────────────────────────────────────────────────────────
def units() -> list[dict]:
    """Every CDS request, in fetch order (ERA5 first: small and fast)."""
    out = [dict(unit="era5", dataset="reanalysis-era5-pressure-levels-monthly-means",
                req={"product_type": ["monthly_averaged_reanalysis"], "variable": ["u_component_of_wind"],
                     "pressure_level": ["10", "100"], "year": [str(y) for y in range(1993, 2018)],
                     "month": [f"{m:02d}" for m in range(1, 13)], "time": ["00:00"], "area": AREA, "grid": [1.0, 1.0],
                     "data_format": "grib"})]
    for key, centre, system, _, starts in SYSTEMS:
        groups = [(starts, ["10", "100"])]
        if key == "ecmwf_51":                          # Oct u100 is the seas5-ref-v1 table: fetch Oct at 10 hPa only
            groups = [((8, 9, 11, 12), ["10", "100"]), ((10,), ["10"])]
        for gi, (months, levs) in enumerate(groups):
            out.append(dict(unit=f"{key}" + (f"_g{gi}" if len(groups) > 1 else ""), system=key,
                            dataset="seasonal-monthly-pressure-levels",
                            req={"originating_centre": centre, "system": system, "product_type": ["monthly_mean"],
                                 "variable": ["u_component_of_wind"], "pressure_level": levs,
                                 "year": [str(y) for y in YEARS], "month": [f"{m:02d}" for m in months],
                                 "leadtime_month": [str(i) for i in range(1, 7)], "area": AREA, "grid": GRID,
                                 "data_format": "grib"}))
    return out


def table_path(unit: str) -> Path:
    return DIR / f"u60_{unit}.npz"


def raw_path(unit: str) -> Path:
    return RAW / f"u60_{unit}.grib"


def upload(*paths: Path) -> None:
    if not TAG:
        return
    for _ in range(3):
        r = subprocess.run(["gh", "release", "upload", TAG, "--clobber", *map(str, paths)], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"    uploaded {', '.join(p.name for p in paths)} → {TAG}", flush=True)
            return
        print(f"    upload failed: {r.stderr.strip()[:300]}", flush=True); time.sleep(10)
    raise RuntimeError("release upload failed")


def fetch(budget_min: float) -> int:
    import seas5_lead7 as L7                          # the shared sequential CDS wrapper (guard_fetch: Actions only)
    t0 = time.time(); missing = []
    for u in units():
        if table_path(u["unit"]).exists():
            print(f"  {u['unit']}: table held", flush=True); continue
        if not raw_path(u["unit"]).exists():
            if (time.time() - t0) / 60 > budget_min:
                print(f"  {u['unit']}: budget spent, left for the next run", flush=True); missing.append(u["unit"]); continue
            ok = L7._cds(u["dataset"], u["req"], raw_path(u["unit"]), f"u60 {u['unit']}")
            if not ok:
                print(f"::warning::{u['unit']}: CDS request failed", flush=True); missing.append(u["unit"]); continue
            upload(raw_path(u["unit"]))
        try:
            reduce(u)
        except Exception as e:                            # noqa: BLE001  the raw GRIB is kept in the release
            print(f"::warning::{u['unit']}: decode failed ({e})", flush=True); missing.append(u["unit"]); continue
        upload(table_path(u["unit"]))
    print(f"fetch done in {(time.time() - t0) / 60:.1f} min; missing: {missing or 'none'}", flush=True)
    return 0 if not missing else 2


# ── GRIB → table ─────────────────────────────────────────────────────────────
def _ring_mean(h, eccodes) -> float:
    lat1 = eccodes.codes_get_double(h, "latitudeOfFirstGridPointInDegrees")
    lat2 = eccodes.codes_get_double(h, "latitudeOfLastGridPointInDegrees")
    if abs(lat1 - 60) > 1e-3 or abs(lat2 - 60) > 1e-3:
        raise ValueError(f"ring not at 60N: {lat1}..{lat2}")
    v = eccodes.codes_get_values(h).astype(float)
    if eccodes.codes_get_long(h, "bitmapPresent"):
        v[v == eccodes.codes_get_double(h, "missingValue")] = np.nan
    ni = eccodes.codes_get_long(h, "Ni")
    lo1 = eccodes.codes_get_double(h, "longitudeOfFirstGridPointInDegrees")
    lo2 = eccodes.codes_get_double(h, "longitudeOfLastGridPointInDegrees")
    if v.size != ni:
        raise ValueError(f"expected one row of {ni}, got {v.size}")
    if ni > 1 and abs(((lo2 - lo1) % 360)) < 1e-3:        # −180 and 180 both present: count the dateline once
        v = v[:-1]
    return float(np.nanmean(v))


def _get(h, eccodes, k):
    try:
        return eccodes.codes_get(h, k)
    except Exception:                                     # noqa: BLE001
        return None


def reduce(u: dict) -> None:
    import eccodes
    rows, diag = [], {}
    with open(raw_path(u["unit"]), "rb") as fh:
        while True:
            h = eccodes.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                lev = int(_get(h, eccodes, "level")); dd = int(_get(h, eccodes, "dataDate"))
                if lev not in LEVELS:
                    raise ValueError(f"unexpected level {lev} ({_get(h, eccodes, 'typeOfLevel')})")
                val = _ring_mean(h, eccodes)
                if u["unit"] == "era5":
                    rows.append((dd // 10000, dd // 100 % 100, 0, lev, 0, val)); continue
                fm = _get(h, eccodes, "forecastMonth"); idx = _get(h, eccodes, "indexingDate")
                vm = _get(h, eccodes, "verifyingMonth"); num = int(_get(h, eccodes, "number") or 0)
                dt = int(_get(h, eccodes, "dataTime") or 0)
                if fm is None:
                    raise KeyError("no forecastMonth key")
                fm = int(fm)
                if idx not in (None, 0):
                    ny, nm = int(idx) // 10000, int(idx) // 100 % 100
                elif dd % 100 == 1:
                    ny, nm = dd // 10000, dd // 100 % 100
                else:
                    raise KeyError(f"lagged member (dataDate {dd}) without indexingDate")
                if vm not in (None, 0):                   # cross-check: verifying month = nominal start + (fm − 1)
                    k = ny * 12 + nm - 1 + fm - 1
                    if (k // 12, k % 12 + 1) != (int(vm) // 100, int(vm) % 100):
                        raise ValueError(f"verifyingMonth {vm} ≠ start {ny}-{nm:02d} + {fm - 1} months")
                key = (dd % 100 != 1, idx is not None, vm is not None)
                diag[str(key)] = diag.get(str(key), 0) + 1
                rows.append((ny, nm, fm, lev, (dd % 10000) * 100000 + dt * 100 + num, val))
            finally:
                eccodes.codes_release(h)
    a = np.array(rows, dtype=float)
    print(f"  {u['unit']}: {len(a)} fields; (lagged dataDate, indexingDate, verifyingMonth) key presence {diag}", flush=True)
    DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(table_path(u["unit"]), year=a[:, 0].astype(int), month=a[:, 1].astype(int), lead=a[:, 2].astype(int),
                        level=a[:, 3].astype(int), member=a[:, 4].astype(np.int64), u=a[:, 5])


# ── loading ──────────────────────────────────────────────────────────────────
def _ecmwf_u100_oct() -> dict:
    """seas5-ref-v1 hc_ssw_u100_10.npz: u [600 samples, 6 leads] = number × year folded by cfgrib's stack. The fold
    order is identified, not assumed: in lead month 1 (October, days from the start) members of one winter agree, so
    the right fold puts most of the variance between years."""
    p = SEAS5_REF / "hc_ssw_u100_10.npz"
    z = np.load(p, allow_pickle=True); v = z["u"].astype(float)
    assert v.shape == (600, 6), v.shape
    fa = v.reshape(25, 24, 6)                              # member-major
    fb = v.reshape(24, 25, 6).transpose(1, 0, 2)           # year-major → (member, year, lead)
    share = []
    for f in (fa, fb):
        x = f[:, :, 0]
        share.append(float(np.var(x.mean(0)) / np.var(x)))
    i = int(np.argmax(share))
    print(f"  SEAS5 Oct u100 table: between-year variance share at lead 1, member-major {share[0]:.2f} vs year-major {share[1]:.2f}", flush=True)
    if not (share[i] > 0.3 and share[i] > 2 * share[1 - i]):
        raise ValueError(f"cannot identify the sample order of {p.name}: {share}")
    f = (fa, fb)[i]
    return {(10, L, 100): (f[:, :, L - 1].mean(0), np.full(N, 25)) for L in range(1, 7)}


def system_means(key: str) -> dict:
    """{(start month, lead, level): (ens-mean[24 years], members[24])}."""
    out = {}
    for u in units():
        if u.get("system") != key:
            continue
        z = np.load(table_path(u["unit"]))
        for sm in sorted(set(z["month"])):
            for L in range(1, 7):
                for lev in LEVELS:
                    m = (z["month"] == sm) & (z["lead"] == L) & (z["level"] == lev)
                    if not m.any():
                        continue
                    e = np.full(N, np.nan); n = np.zeros(N, int)
                    for yi, y in enumerate(YEARS):
                        mm = m & (z["year"] == y) & np.isfinite(z["u"])
                        if mm.any():
                            e[yi] = z["u"][mm].mean(); n[yi] = int(mm.sum())
                    out[(int(sm), L, lev)] = (e, n)
    if key == "ecmwf_51":
        out.update(_ecmwf_u100_oct())
    for k, (e, n) in out.items():
        if not np.isfinite(e).all() or n.min() < 2:
            raise ValueError(f"{key} {k}: incomplete hindcast (years with data {int(np.isfinite(e).sum())}, min members {n.min()})")
    return out


def merra2_monthly() -> dict:
    import xarray as xr
    import pandas as pd
    ds = xr.open_dataset(REPO / "scripts" / "strat" / "reference" / "strat_history.nc")
    s = pd.Series(ds["u10"].values.astype(float), index=pd.to_datetime(ds["time"].values)); ds.close()
    g = s.groupby([s.index.year, s.index.month])
    cnt, mean = g.count(), g.mean()
    return {(int(y), int(m)): float(mean[(y, m)]) for (y, m) in mean.index if cnt[(y, m)] >= 28}


def era5_monthly() -> dict:
    z = np.load(table_path("era5"))
    return {lev: {(int(y), int(m)): float(v) for y, m, l, v in zip(z["year"], z["month"], z["level"], z["u"]) if l == lev}
            for lev in LEVELS}


def target_cells(starts) -> list[tuple[int, int, int]]:
    """(start month, target month, lead) reachable in leads 1–6 within the winter."""
    out = []
    for sm in starts:
        for tm in TARGETS:
            L = (tm - sm) % 12 + 1
            if 1 <= L <= 6:
                out.append((sm, tm, L))
    return out


def model_series(means: dict, sm: int, target, lev: int):
    """ens-mean series [24] for a target month (int) or 'DJF' from start sm."""
    if target == "DJF":
        return np.mean([means[(sm, (tm - sm) % 12 + 1, lev)][0] for tm in (12, 1, 2)], axis=0)
    return means[(sm, (target - sm) % 12 + 1, lev)][0]


def truth_series(truth: dict, sm: int, target) -> np.ndarray:
    def one(tm):
        return np.array([truth[(y if tm >= sm else y + 1, tm)] for y in YEARS])
    if target == "DJF":
        return np.mean([one(tm) for tm in (12, 1, 2)], axis=0)
    return one(target)


# ── statistics ───────────────────────────────────────────────────────────────
def loyo(x: np.ndarray) -> np.ndarray:
    return x - (x.sum() - x) / (len(x) - 1)


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


PERM = np.array([RNG.permutation(N) for _ in range(NPERM)])
BOOT = RNG.integers(0, N, size=(NBOOT, N))


def score(fa: np.ndarray, oa: np.ndarray, f_raw=None, o_raw=None) -> dict:
    """fa, oa: LOYO anomalies [24]. f_raw/o_raw (optional) give the bias."""
    r = float(np.corrcoef(fa, oa)[0, 1])
    rp = _corr_rows(np.broadcast_to(fa, (NPERM, N)), oa[PERM])
    p = float((np.sum(rp >= r) + 1) / (NPERM + 1))
    rb = _corr_rows(fa[BOOT], oa[BOOT])
    msss = float(1 - np.sum((fa - oa) ** 2) / np.sum(oa ** 2))
    mb = 1 - np.sum((fa[BOOT] - oa[BOOT]) ** 2, -1) / np.sum(oa[BOOT] ** 2, -1)
    d = dict(r=r, r_ci95=[float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))], p=p,
             msss=msss, msss_ci95=[float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))],
             sd_ratio=float(fa.std() / oa.std()))
    if f_raw is not None:
        diff = f_raw - o_raw
        bb = diff[BOOT].mean(-1)
        d.update(bias=float(diff.mean()), bias_ci95=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
                 model_clim=float(f_raw.mean()), truth_clim=float(o_raw.mean()))
    return d


def holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k]); m = len(keys); out = {}; run = 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k])); out[k] = run
    return out


def bh(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k]); m = len(keys); q = {}
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, p[keys[i]] * m / (i + 1)); q[keys[i]] = prev
    return q


def power_r(n: int, alpha: float, power: float = 0.8) -> float:
    from scipy.stats import norm
    return float(np.tanh((norm.ppf(1 - alpha) + norm.ppf(power)) / np.sqrt(n - 3)))


# ── build ────────────────────────────────────────────────────────────────────
def build() -> dict:
    t0 = time.time()
    need = [u["unit"] for u in units() if not table_path(u["unit"]).exists()]
    if need:
        raise FileNotFoundError(f"tables missing: {need} (run fetch)")
    means = {k: system_means(k) for k in SYS}
    nmem = {k: {f"{sm:02d}": int(np.median(v[1])) for (sm, L, lev), v in means[k].items() if L == 1 and lev == 10} for k in SYS}
    m2 = merra2_monthly(); e5 = era5_monthly()
    truth = {10: m2, 100: e5[100]}
    res = dict(question="Skill of C3S seasonal systems for the monthly-mean zonal wind at 60N, 10 and 100 hPa (northern winter)",
               hindcast=f"{YEARS[0]}/{(YEARS[0] + 1) % 100:02d}–{YEARS[-1]}/{(YEARS[-1] + 1) % 100:02d}", n=N,
               truth={"10": "MERRA-2 (strat_history.nc u10, monthly means of daily)", "100": "ERA5 monthly means (CDS)"},
               systems={k: dict(label=SYS[k][3], centre=SYS[k][1], system=SYS[k][2], members_by_start=nmem[k]) for k in SYS},
               centres={c: ks for c, ks in CENTRES},
               power=dict(r80_alpha05=round(power_r(N, 0.05), 3), r80_holm36=round(power_r(N, 0.05 / 36), 3),
                          note="one-sided Fisher z; the smallest p of a 36-test Holm family must beat 0.05/36"))

    def centre_anoms(sm, target, lev, truth_lev):
        """{centre: (anomaly f', raw ens-mean or None)}, component anomalies for ECCC, and the truth (o', o)."""
        o = truth_series(truth_lev, sm, target); oa = loyo(o)
        cen, comp = {}, {}
        for c, ks in CENTRES:
            parts = []
            for k in ks:
                try:
                    f = model_series(means[k], sm, target, lev)
                except KeyError:
                    continue
                comp[k] = (loyo(f), f); parts.append(k)
            if len(parts) == len(ks):
                fa = np.mean([comp[k][0] for k in parts], axis=0)
                fraw = np.mean([comp[k][1] for k in parts], axis=0)
                cen[c] = (fa, fraw)
        return cen, comp, oa, o

    def cell(sm, target, lev, truth_lev):
        cen, comp, oa, o = centre_anoms(sm, target, lev, truth_lev)
        out = {c: score(fa, oa, fraw, o) for c, (fa, fraw) in cen.items()}
        if len(cen) >= 2:
            out[MMM] = score(np.mean([v[0] for v in cen.values()], axis=0), oa)
            out[MMM]["n_centres"] = len(cen)
        comps = {SYS[k][3]: score(comp[k][0], oa, comp[k][1], o) for k in ("eccc_4", "eccc_5") if k in comp}
        return out, comps

    T = {"Dec": 12, "Jan": 1, "Feb": 2, "DJF": "DJF"}
    res["families"], res["controls"], res["grid"], res["eccc_components"] = {}, {}, {}, {}
    for lev in LEVELS:
        fam, pv = {}, {}
        for tname, tgt in T.items():
            sc, comps = cell(10, tgt, lev, truth[lev])
            fam[tname] = sc
            res["eccc_components"].setdefault(str(lev), {})[tname] = comps
            for c, d in sc.items():
                pv[(tname, c)] = d["p"]
        adj = holm(pv)
        for (tname, c), a in adj.items():
            fam[tname][c]["p_holm"] = a; fam[tname][c]["significant"] = bool(a < 0.05)
        res["families"][str(lev)] = dict(n_tests=len(pv), start="Oct", targets=fam)
        # controls
        ctl, pc = {}, {}
        for name, sm in (("Nov from Oct start (lead month 2)", 10), ("Nov from Nov start (lead month 1)", 11)):
            sc, _ = cell(sm, 11, lev, truth[lev]); ctl[name] = sc
            for c, d in sc.items():
                pc[(name, c)] = d["p"]
        for (name, c), a in holm(pc).items():
            ctl[name][c]["p_holm"] = a; ctl[name][c]["significant"] = bool(a < 0.05)
        res["controls"][str(lev)] = ctl
        # descriptive grid
        grid, pg = {}, {}
        for sm, tm, L in target_cells(STARTS):
            sc, _ = cell(sm, tm, lev, truth[lev])
            key = f"{sm:02d}>{MONTH_NAME[tm]}"
            grid[key] = dict(start=sm, target=MONTH_NAME[tm], lead=L, scores=sc)
            for c, d in sc.items():
                pg[(key, c)] = d["p"]
        for (key, c), q in bh(pg).items():
            grid[key]["scores"][c]["q_bh"] = q; grid[key]["scores"][c]["fdr10"] = bool(q < 0.10)
        res["grid"][str(lev)] = dict(n_tests=len(pg), cells=grid)
    # truth check at 10 hPa: MERRA-2 vs ERA5, and the family on ERA5 (descriptive)
    chk = {}
    for tname, tgt in T.items():
        a, b = truth_series(m2, 10, tgt), truth_series(e5[10], 10, tgt)
        chk[tname] = dict(r=float(np.corrcoef(a, b)[0, 1]), mean_diff_era5_minus_merra2=float((b - a).mean()),
                          rmsd_anom=float(np.sqrt(np.mean((loyo(a) - loyo(b)) ** 2))))
        sc, _ = cell(10, tgt, 10, e5[10])
        chk[tname]["family_on_era5"] = {c: dict(r=d["r"], p=d["p"]) for c, d in sc.items()}
    res["truth_check_10hpa"] = chk
    res["figure"] = render(res)
    res["generated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    OUT.mkdir(parents=True, exist_ok=True)

    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)): return None if not np.isfinite(x) else round(float(x), 4)
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, np.bool_): return bool(x)
        return x
    (OUT / "c3s_u10_skill.json").write_text(json.dumps(clean(res), ensure_ascii=False, indent=1))
    report(res)
    print(f"wrote {OUT / 'c3s_u10_skill.json'} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return res


def report(res: dict) -> None:
    names = [c for c, _ in CENTRES] + [MMM]
    for lev in map(str, LEVELS):
        print(f"\n=== F{lev}: {lev} hPa, October start, Holm over {res['families'][lev]['n_tests']} tests ===")
        print(f"{'centre':<16}{'tgt':<5}{'bias':>7}{'r':>7}{'95% CI':>15}{'p':>8}{'Holm':>8}{'MSSS':>7}{'sdR':>6}")
        for c in names:
            for t in PRIMARY_TARGETS:
                d = res["families"][lev]["targets"][t].get(c)
                if not d:
                    continue
                b = f"{d['bias']:+.1f}" if "bias" in d else "  –"
                print(f"{c:<16}{t:<5}{b:>7}{d['r']:>7.2f}  [{d['r_ci95'][0]:+.2f},{d['r_ci95'][1]:+.2f}]{d['p']:>8.4f}{d['p_holm']:>8.3f}"
                      f"{d['msss']:>7.2f}{d['sd_ratio']:>6.2f}{'  *' if d['significant'] else ''}")
        print(f"--- controls {lev} hPa (Holm within) ---")
        for name, sc in res["controls"][lev].items():
            print("  " + name + ": " + ", ".join(f"{c} r {d['r']:.2f} (p {d['p']:.4f}, Holm {d['p_holm']:.3f})" for c, d in sc.items()))
        g = res["grid"][lev]
        hits = [(k, c, d) for k, v in g["cells"].items() for c, d in v["scores"].items() if d["fdr10"]]
        print(f"--- descriptive grid {lev} hPa: {len(hits)} of {g['n_tests']} cells pass FDR 10 % ---")
        for k, c, d in sorted(hits, key=lambda x: x[2]["q_bh"]):
            print(f"  {k} lead {g['cells'][k]['lead']} {c}: r {d['r']:.2f} q {d['q_bh']:.3f}")
    print("\n--- 10 hPa truth check (MERRA-2 vs ERA5) ---")
    for t, d in res["truth_check_10hpa"].items():
        print(f"  {t}: r {d['r']:.3f}, ERA5−MERRA-2 {d['mean_diff_era5_minus_merra2']:+.2f} m/s; "
              + ", ".join(f"{c} {v['r']:.2f}" for c, v in d["family_on_era5"].items()))
    print(f"\npower: r detectable at 80 %: {res['power']}")


def render(res: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [c for c, _ in CENTRES] + [MMM]
    cols = {"Dec": "#2a78d6", "Jan": "#eb6834", "Feb": "#1baf7a", "DJF": "#eda100"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), gridspec_kw=dict(width_ratios=[2.3, 1]), sharey=True)
    ypos = np.arange(len(names))[::-1]
    for row, lev in enumerate(map(str, LEVELS)):
        ax, axb = axes[row]
        fam = res["families"][lev]["targets"]
        for j, t in enumerate(PRIMARY_TARGETS):
            off = (j - 1.5) * 0.17
            for i, c in enumerate(names):
                d = fam[t].get(c)
                if not d:
                    continue
                y = ypos[i] + off
                ax.plot(d["r_ci95"], [y, y], color=cols[t], lw=1.6, solid_capstyle="round")
                ax.plot([d["r"]], [y], "o", ms=7, mfc=cols[t] if d["significant"] else "white", mec=cols[t], mew=1.6,
                        label=t if i == 0 else None)
        ax.axvline(0, color="#8a8680", lw=0.8)
        ax.axvline(res["power"]["r80_alpha05"], color="#8a8680", lw=0.8, ls=":")
        ax.axvline(res["power"]["r80_holm36"], color="#8a8680", lw=0.8, ls="--")
        ax.set_xlim(-0.8, 1.0)
        ax.set_yticks(ypos); ax.set_yticklabels(names)
        truth = "MERRA-2" if lev == "10" else "ERA5"
        ax.set_title(f"{lev} hPa: correlation of the October-start ensemble-mean anomaly with {truth} (95 % CI)", loc="left", fontsize=10.5)
        ax.set_xlabel("r (filled = Holm-significant within the 36-test family; dotted / dashed = r at 80 % power, raw / Holm α)")
        ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=4, bbox_to_anchor=(1.0, 1.06))
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e6e3dc", lw=0.6)
        for i, c in enumerate(names):
            d = fam["DJF"].get(c)
            if d and "bias" in d:
                axb.plot(d["bias_ci95"], [ypos[i]] * 2, color="#5f5b55", lw=1.6)
                axb.plot([d["bias"]], [ypos[i]], "o", ms=7, color="#5f5b55")
        axb.axvline(0, color="#8a8680", lw=0.8)
        axb.set_title(f"DJF vortex bias at {lev} hPa, model − {truth} (m/s)", loc="left", fontsize=10.5)
        axb.spines[["top", "right"]].set_visible(False); axb.grid(axis="x", color="#e6e3dc", lw=0.6)
    fig.suptitle(f"C3S seasonal systems: monthly-mean zonal wind at 60°N, hindcasts {res['hindcast']} ({res['n']} winters)", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "c3s_u10_skill.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fetch", "build", "units"])
    ap.add_argument("--budget-min", type=float, default=300)
    a = ap.parse_args(argv)
    if a.cmd == "units":
        for u in units():
            print(u["unit"], u["dataset"], json.dumps(u["req"])[:200], "held" if table_path(u["unit"]).exists() else "")
        return 0
    if a.cmd == "fetch":
        return fetch(a.budget_min)
    build(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
