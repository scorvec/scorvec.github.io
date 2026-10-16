#!/usr/bin/env python3
"""December polar vortex outlook from the C3S multi-model mean (monthly u at 60°N, 10 hPa), calibrated on the hindcast.

The pre-registered monthly test (c3s_u10_skill.py, release c3s-u10-v1) found only December significant from the
October start (multi-model mean r 0.67, 95 % CI 0.36-0.86, Holm p 0.003 over the 7 centres since BoM left on
2026-10-16; with BoM r 0.66, p 0.005; UKMO 0.73, DWD 0.61; every other target and all of 100 hPa insignificant),
so the outlook is for December, from the multi-model mean.

WHAT IT DOES, per C3S issue (start month S = Aug ... Dec, December = lead month 12 - S + 1):
  1. FETCH the issue's monthly-mean u at 10 hPa on the 60°N ring (2° grid, the c3s_u10_skill request with the year set to
     the issue) for each centre's CURRENT system, one sequential CDS request each (cds_seq; c3s-vortex.yml, `seas5` group):
     ECMWF 51 (from seas5.yml's cached fc_strat_u when present - never fetched twice), UKMO 610, Météo-France 9, DWD 22,
     CMCC 4, ECCC 4 + 5, JMA 4 (BoM left out since 2026-10-16). CMCC and JMA changed system after the test's hindcast versions (CMCC 35 -> 4 from
     the Aug 2025 start, JMA 3 -> 4 from Mar 2026), so their current systems' Aug-Dec hindcasts (1993-2016, leads 1-6) are
     fetched once (`fetch-hc`) into the same release, same request as the test.
  2. CALIBRATE: each centre's ensemble-mean anomaly against its own hindcast climatology (hindcast leave-one-year-out;
     forecast against all 24 winters; ECCC = mean of its two models); MMM = equal-weight mean over the centres present.
     MERRA-2's December mean is regressed on the hindcast MMM (OLS, 24 winters); the forecast distribution is the
     regression's Student-t predictive distribution (n - 2 dof, scale s·sqrt(1 + 1/n + (x - x̄)²/Sxx)). Terciles: the
     1993-2016 MERRA-2 December means. P(weak) = below the lower tercile, P(strong) = above the upper one.
  3. CHECK: the same calibration leave-one-winter-out over the hindcast (each winter's terciles and fit from the other
     23): ranked probability skill score against 1/3-1/3-1/3 (bootstrap 95 % CI over winters; one-sided permutation p,
     5,000 shuffles of the outcomes through the whole leave-one-out procedure), coverage of the 80 % interval (binomial
     test against 0.8), tercile reliability counts. The composition's own hindcast r (with the current systems) is
     reported next to the pre-registered result.
  The forecast probabilities are published only when at least QUORUM centres have the issue AND the hindcast r of the
  centres present is significant (one-sided permutation p < 0.05); otherwise the card says what it is waiting for.
  Ensemble sizes: forecast ensembles are larger than the hindcast ones, so the forecast MMM is less noisy than the
  predictor the regression was fitted on; the slope is then slightly too small (a conservative forecast).

    python c3s_vortex.py fetch-hc                    # once: CMCC 4 and JMA 4 hindcast tables -> c3s-u10-v1
    python c3s_vortex.py fetch --issue 202610        # this issue's forecasts (missing systems only)
    python c3s_vortex.py build --issue 202610        # assets/sst/data/c3s_vortex.json
"""

from __future__ import annotations

import pyproj  # noqa: F401  first, before eccodes/xarray load another libproj
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
SITE = Path(os.environ.get("SST_SITE_ROOT", REPO))
DIR = Path(os.environ.get("C3SU10_DIR", HERE / "data" / "c3s_u10"))
RAW = DIR / "raw"
TAG = os.environ.get("C3SU10_TAG", "")
OUT_JSON = SITE / "assets" / "sst" / "data" / "c3s_vortex.json"
SEAS5_FC = HERE / "data" / "seas5" / "forecast"  # seas5.yml's cached issue GRIBs (fc_strat_u_{issue}.grib)
YEARS = list(range(1993, 2017))
N = len(YEARS)
AREA, GRID = [60, -180, 60, 180], [2.0, 2.0]
QUORUM = 5
NPERM = 5000
RNG = np.random.default_rng(20261009)

# (key, centre, system, label, start months with a monthly hindcast, the system the pre-registered test used)
CURRENT = [
    ("ecmwf_51", "ecmwf", "51", "ECMWF SEAS5", (8, 9, 10, 11, 12), "ecmwf_51"),
    ("ukmo_610", "ukmo", "610", "UKMO GloSea (610)", (8, 9, 10), "ukmo_610"),
    ("mf_9", "meteo_france", "9", "Météo-France 9", (8, 9, 10, 11, 12), "mf_9"),
    ("dwd_22", "dwd", "22", "DWD GCFS 2.2", (8, 9, 10, 11, 12), "dwd_22"),
    ("cmcc_4", "cmcc", "4", "CMCC SPS4", (8, 9, 10, 11, 12), "cmcc_35"),
    ("eccc_4", "eccc", "4", "ECCC GEM5-NEMO", (8, 9, 10, 11, 12), "eccc_4"),
    ("eccc_5", "eccc", "5", "ECCC CanESM5", (8, 9, 10, 11, 12), "eccc_5"),
    ("jma_4", "jma", "4", "JMA CPS4", (8, 9, 10, 11, 12), "jma_3"),
]
CUR = {c[0]: c for c in CURRENT}
CENTRES = [
    ("ECMWF", ["ecmwf_51"]),
    ("UKMO", ["ukmo_610"]),
    ("Météo-France", ["mf_9"]),
    ("DWD", ["dwd_22"]),
    ("CMCC", ["cmcc_4"]),
    ("ECCC", ["eccc_4", "eccc_5"]),
    ("JMA", ["jma_4"]),
]
NEW_HC = ("cmcc_4", "jma_4")
MMM = "Multi-model mean"
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def hc_table(key: str) -> list[Path]:
    return sorted(DIR.glob(f"u60_{key}.npz")) + sorted(DIR.glob(f"u60_{key}_g*.npz"))


def fc_table(key: str, issue: str) -> Path:
    return DIR / f"u60fc_{key}_{issue}.npz"


def upload(*paths: Path) -> None:
    if not TAG:
        return
    gh = os.environ.get("GH_BIN") or shutil.which("gh") or "gh"
    for _ in range(3):
        r = subprocess.run(
            [
                gh,
                "release",
                "upload",
                TAG,
                "--repo",
                os.environ.get("GITHUB_REPOSITORY", "scorvec/scorvec.github.io"),
                "--clobber",
                *map(str, paths),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            print(f"    uploaded {', '.join(p.name for p in paths)} → {TAG}", flush=True)
            return
        print(f"    upload failed: {r.stderr.strip()[:300]}", flush=True)
        time.sleep(10)
    print("::warning::release upload failed", flush=True)


# ── GRIB → table (the ring, or the 60°N row of a global field) ──────────────
def reduce_monthly(src: Path, dest: Path, level: int = 10) -> int:
    """rows (year, month, lead, level, member-code, u) as c3s_u10_skill.reduce, for any grid that has a 60°N row."""
    import eccodes
    from c3s_u10_skill import _get

    rows = []
    with open(src, "rb") as fh:
        while True:
            h = eccodes.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                lev = int(_get(h, eccodes, "level"))
                if lev != level:
                    continue
                ni, nj = int(_get(h, eccodes, "Ni")), int(_get(h, eccodes, "Nj"))
                la1 = float(_get(h, eccodes, "latitudeOfFirstGridPointInDegrees"))
                la2 = float(_get(h, eccodes, "latitudeOfLastGridPointInDegrees"))
                v = eccodes.codes_get_values(h).astype(float)
                if eccodes.codes_get_long(h, "bitmapPresent"):
                    v[v == eccodes.codes_get_double(h, "missingValue")] = np.nan
                v = v.reshape(nj, ni)
                lats = np.linspace(la1, la2, nj)
                j = int(np.argmin(np.abs(lats - 60)))
                if abs(lats[j] - 60) > 1e-3:
                    raise ValueError(f"no 60N row in {la1}..{la2}")
                lo1 = float(_get(h, eccodes, "longitudeOfFirstGridPointInDegrees"))
                lo2 = float(_get(h, eccodes, "longitudeOfLastGridPointInDegrees"))
                row = v[j]
                if ni > 1 and abs((lo2 - lo1) % 360) < 1e-3:
                    row = row[:-1]
                val = float(np.nanmean(row))
                dd = int(_get(h, eccodes, "dataDate"))
                dtm = int(_get(h, eccodes, "dataTime") or 0)
                fm = int(_get(h, eccodes, "forecastMonth"))
                idx = _get(h, eccodes, "indexingDate")
                vm = _get(h, eccodes, "verifyingMonth")
                num = int(_get(h, eccodes, "number") or 0)
                if idx not in (None, 0):
                    ny, nm = int(idx) // 10000, int(idx) // 100 % 100
                elif dd % 100 == 1:
                    ny, nm = dd // 10000, dd // 100 % 100
                else:
                    raise KeyError(f"lagged member (dataDate {dd}) without indexingDate")
                if vm not in (None, 0):
                    k = ny * 12 + nm - 1 + fm - 1
                    if (k // 12, k % 12 + 1) != (int(vm) // 100, int(vm) % 100):
                        raise ValueError(f"verifyingMonth {vm} ≠ start {ny}-{nm:02d} + {fm - 1}")
                if not np.isfinite(val) or abs(val) > 150:
                    raise ValueError(f"implausible u {val}")
                rows.append((ny, nm, fm, lev, (dd % 10000) * 100000 + dtm * 100 + num, val))
            finally:
                eccodes.codes_release(h)
    if not rows:
        raise ValueError(f"no {level} hPa fields in {src.name}")
    a = np.array(rows, dtype=float)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        year=a[:, 0].astype(int),
        month=a[:, 1].astype(int),
        lead=a[:, 2].astype(int),
        level=a[:, 3].astype(int),
        member=a[:, 4].astype(np.int64),
        u=a[:, 5],
    )
    print(f"  {dest.name}: {len(a)} fields", flush=True)
    return len(a)


def _req(centre, system, years, months):
    return {
        "originating_centre": centre,
        "system": system,
        "product_type": ["monthly_mean"],
        "variable": ["u_component_of_wind"],
        "pressure_level": ["10"],
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in months],
        "leadtime_month": [str(i) for i in range(1, 7)],
        "area": AREA,
        "grid": GRID,
        "data_format": "grib",
    }


def fetch_hc() -> int:
    from cds_seq import retrieve

    bad = 0
    for key in NEW_HC:
        dest = DIR / f"u60_{key}.npz"
        if dest.exists():
            print(f"  {key}: hindcast table held", flush=True)
            continue
        _, centre, system, _, starts, _ = CUR[key]
        raw = RAW / f"u60_{key}.grib"
        if not retrieve(
            "seasonal-monthly-pressure-levels",
            _req(centre, system, YEARS, starts),
            raw,
            f"u60 hindcast {key}",
            max_wait_h=float(os.environ.get("CDS_MAX_WAIT_H", "30")),
        ):
            print(f"::warning::{key}: hindcast request failed", flush=True)
            bad += 1
            continue
        reduce_monthly(raw, dest)
        upload(raw, dest)
    return 0 if not bad else 2


def fetch(issue: str) -> int:
    from cds_seq import available, retrieve

    y, s = int(issue[:4]), int(issue[4:])
    missing = []
    for key, centre, system, _, _, _ in CURRENT:
        dest = fc_table(key, issue)
        if dest.exists():
            continue
        if key == "ecmwf_51":  # seas5.yml's copy of the issue, if the cache gave it to us
            src = SEAS5_FC / f"fc_strat_u_{issue}.grib"
            if src.exists():
                try:
                    reduce_monthly(src, dest)
                    upload(dest)
                    continue
                except Exception as e:  # noqa: BLE001
                    print(f"::warning::SEAS5 cached strat_u unusable ({e}); asking the CDS", flush=True)
        raw = RAW / f"u60fc_{key}_{issue}.grib"
        if available("seasonal-monthly-pressure-levels", _req(centre, system, [y], [s])) is False:
            print(f"  {key}: issue {issue} not in the CDS catalogue yet", flush=True)
            missing.append(key)
            continue
        ok = retrieve(
            "seasonal-monthly-pressure-levels",
            _req(centre, system, [y], [s]),
            raw,
            f"u60 {key} {issue}",
            tries=2,
            max_wait_h=float(os.environ.get("CDS_MAX_WAIT_H", "0.5")),
        )
        if not ok:
            print(f"  {key}: issue {issue} not on the CDS yet (or the request failed)", flush=True)
            missing.append(key)
            continue
        try:
            reduce_monthly(raw, dest)
            upload(dest)
        except Exception as e:  # noqa: BLE001
            print(f"::warning::{key}: decode/QC failed ({e})", flush=True)
            missing.append(key)
    print(f"issue {issue}: missing {missing or 'none'}", flush=True)
    return 0


# ── loading ──────────────────────────────────────────────────────────────────
def _load(paths: list[Path]) -> dict:
    zs = [np.load(p) for p in paths]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("year", "month", "lead", "level", "member", "u")}


def hindcast(key: str, sm: int, lead: int) -> tuple[np.ndarray, np.ndarray]:
    """ens-mean [24] and members [24] of one system's hindcast at start sm, lead month `lead`, 10 hPa."""
    ps = hc_table(key)
    if not ps:
        raise FileNotFoundError(f"no hindcast table for {key}")
    z = _load(ps)
    m = (z["month"] == sm) & (z["lead"] == lead) & (z["level"] == 10) & np.isfinite(z["u"])
    e = np.full(N, np.nan)
    n = np.zeros(N, int)
    for i, y in enumerate(YEARS):
        mm = m & (z["year"] == y)
        if mm.any():
            e[i] = z["u"][mm].mean()
            n[i] = int(mm.sum())
    if not np.isfinite(e).all() or n.min() < 2:
        raise ValueError(
            f"{key} start {sm} lead {lead}: incomplete hindcast ({int(np.isfinite(e).sum())} years, min {n.min()} members)"
        )
    return e, n


def forecast(key: str, issue: str, lead: int) -> np.ndarray | None:
    p = fc_table(key, issue)
    if not p.exists():
        return None
    z = np.load(p)
    y, s = int(issue[:4]), int(issue[4:])
    m = (z["year"] == y) & (z["month"] == s) & (z["lead"] == lead) & (z["level"] == 10) & np.isfinite(z["u"])
    return z["u"][m] if m.sum() >= 2 else None


def merra2_dec() -> dict:
    from c3s_u10_skill import merra2_monthly

    m2 = merra2_monthly()
    return {y: m2[(y, 12)] for y in range(1980, 2030) if (y, 12) in m2}


# ── statistics ───────────────────────────────────────────────────────────────
def loyo(x):
    return x - (x.sum() - x) / (len(x) - 1)


def tcdf(x, df):
    from scipy.stats import t

    return t.cdf(x, df)


def tppf(q, df):
    from scipy.stats import t

    return t.ppf(q, df)


def fit(x, o):
    """OLS o = a + b x → (a, b, s, xbar, sxx)."""
    xb, ob = x.mean(), o.mean()
    sxx = ((x - xb) ** 2).sum()
    b = ((x - xb) * (o - ob)).sum() / sxx
    a = ob - b * xb
    res = o - a - b * x
    s = np.sqrt((res**2).sum() / (len(x) - 2))
    return a, b, s, xb, sxx


def predictive(x0, a, b, s, xb, sxx, n):
    return a + b * x0, s * np.sqrt(1 + 1 / n + (x0 - xb) ** 2 / sxx), n - 2


def probs(mu, sc, df, t1, t2):
    lo = tcdf((t1 - mu) / sc, df)
    hi = 1 - tcdf((t2 - mu) / sc, df)
    return np.array([lo, 1 - lo - hi, hi])


def loo_check(x, o):
    """Leave-one-winter-out calibration: tercile probabilities, category, PIT, 80 % interval hit, per winter."""
    n = len(x)
    P = np.zeros((n, 3))
    cat = np.zeros(n, int)
    pit = np.zeros(n)
    hit = np.zeros(n, bool)
    mu_ = np.zeros(n)
    for i in range(n):
        k = np.arange(n) != i
        a, b, s, xb, sxx = fit(x[k], o[k])
        mu, sc, df = predictive(x[i], a, b, s, xb, sxx, n - 1)
        t1, t2 = np.quantile(o[k], [1 / 3, 2 / 3])
        P[i] = probs(mu, sc, df, t1, t2)
        cat[i] = 0 if o[i] < t1 else (2 if o[i] > t2 else 1)
        pit[i] = tcdf((o[i] - mu) / sc, df)
        lo, hi = mu + sc * tppf(0.1, df), mu + sc * tppf(0.9, df)
        hit[i] = lo <= o[i] <= hi
        mu_[i] = mu
    return P, cat, pit, hit, mu_


def rps(P, cat):
    O = np.zeros_like(P)
    O[np.arange(len(cat)), cat] = 1
    return ((np.cumsum(P, 1) - np.cumsum(O, 1)) ** 2).sum(1)


def rpss(P, cat):
    clim = np.full_like(P, 1 / 3)
    return 1 - rps(P, cat).sum() / rps(clim, cat).sum()


def reliability(P, cat, col):
    bins = [(0.0, 0.2), (0.2, 0.45), (0.45, 1.01)]
    out = []
    for lo, hi in bins:
        m = (P[:, col] >= lo) & (P[:, col] < hi)
        out.append(
            dict(
                bin=[lo, min(hi, 1.0)],
                n=int(m.sum()),
                mean_p=float(P[m, col].mean()) if m.any() else None,
                obs_freq=float((cat[m] == col).mean()) if m.any() else None,
            )
        )
    return out


def calibration_block(x, o, years) -> dict:
    """Hindcast skill + leave-one-out reliability of the calibrated forecast for predictor x (MMM anomalies)."""
    from c3s_u10_skill import score
    from scipy.stats import binomtest

    sk = score(x, loyo(o))
    P, cat, pit, hit, mu = loo_check(x, o)
    rs = rpss(P, cat)
    B = RNG.integers(0, len(x), size=(10000, len(x)))
    rps_f, rps_c = rps(P, cat), rps(np.full_like(P, 1 / 3), cat)
    rb = 1 - rps_f[B].sum(1) / rps_c[B].sum(1)
    perm = np.empty(NPERM)
    for k in range(NPERM):
        oo = o[RNG.permutation(len(o))]
        Pp, cp, *_ = loo_check(x, oo)
        perm[k] = rpss(Pp, cp)
    p_rpss = float((np.sum(perm >= rs) + 1) / (NPERM + 1))
    cov = int(hit.sum())
    a, b, s, xb, sxx = fit(x, o)
    return dict(
        r=sk["r"],
        r_ci95=sk["r_ci95"],
        p=sk["p"],
        msss=sk["msss"],
        msss_ci95=sk["msss_ci95"],
        sd_ratio=sk["sd_ratio"],
        significant=bool(sk["p"] < 0.05),
        regression=dict(a=a, b=b, s=s, xbar=xb, sxx=sxx, n=len(x)),
        loo=dict(
            rpss=float(rs),
            rpss_ci95=[float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
            rpss_p=p_rpss,
            rpss_significant=bool(p_rpss < 0.05),
            cover80=cov,
            cover80_n=len(x),
            cover80_p=float(binomtest(cov, len(x), 0.8).pvalue),
            pit=[float(v) for v in pit],
            rel_weak=reliability(P, cat, 0),
            rel_strong=reliability(P, cat, 2),
            hits=dict(
                weak_forecast_weak=int(((P.argmax(1) == 0) & (cat == 0)).sum()),
                most_likely_correct=int((P.argmax(1) == cat).sum()),
                n=len(x),
            ),
        ),
        winters=[
            dict(
                year=int(yv),
                x=float(xv),
                obs=float(ov),
                loo_mean=float(m),
                p_weak=float(pp[0]),
                p_strong=float(pp[2]),
                cat=int(c),
            )
            for yv, xv, ov, m, pp, c in zip(years, x, o, mu, P, cat)
        ],
    )


# ── build ────────────────────────────────────────────────────────────────────
def prereg(sm: int) -> dict:
    """The pre-registered result (start 10) or the descriptive FDR grid cell (other starts) from c3s_u10_skill.json."""
    p = DIR / "c3s_u10_skill.json"
    if not p.exists():
        return {}
    J = json.loads(p.read_text())
    if sm == 10:
        cell = J["families"]["10"]["targets"]["Dec"]
        out = {
            c: dict(
                r=d["r"],
                r_ci95=d["r_ci95"],
                p=d["p"],
                p_holm=d["p_holm"],
                significant=d["significant"],
                msss=d["msss"],
                msss_ci95=d["msss_ci95"],
            )
            for c, d in cell.items()
        }
        return dict(
            kind="pre-registered",
            family=f"Holm over {J['families']['10']['n_tests']} tests ({len(CENTRES)} centres + MMM × Dec/Jan/Feb/DJF)",
            scores=out,
            generated=J.get("generated"),
            power=J.get("power"),
        )
    key = f"{sm:02d}>Dec"
    c = J["grid"]["10"]["cells"].get(key)
    if not c:
        return {}
    out = {
        k: dict(r=d["r"], r_ci95=d["r_ci95"], p=d["p"], q_bh=d["q_bh"], significant=d["fdr10"])
        for k, d in c["scores"].items()
    }
    return dict(
        kind="descriptive grid",
        family=f"Benjamini-Hochberg FDR 10 % over {J['grid']['10']['n_tests']} start × target cells",
        scores=out,
        generated=J.get("generated"),
    )


def build(issue: str) -> dict:
    y, s = int(issue[:4]), int(issue[4:])
    key = f"{issue[:4]}-{issue[4:]}"
    t0 = time.time()
    if s not in (8, 9, 10, 11, 12):
        print(f"issue {key}: no December target (start months Aug-Dec only); the card keeps its last build", flush=True)
        return {}
    lead = 12 - s + 1
    dec = merra2_dec()
    o = np.array([dec[yy] for yy in YEARS])
    terc = [float(v) for v in np.quantile(o, [1 / 3, 2 / 3])]
    expected = [c for c, ks in CENTRES if all(s in CUR[k][4] for k in ks)]
    rows, hc_an, fc_an, present = [], {}, {}, []
    for c, ks in CENTRES:
        comp_h, comp_f, comp_rows, ok_h, ok_f = [], [], [], True, True
        for k in ks:
            if s not in CUR[k][4]:
                ok_h = False
                continue
            try:
                h, nh = hindcast(k, s, lead)
            except (FileNotFoundError, ValueError) as e:
                print(f"  {k}: {e}", flush=True)
                ok_h = False
                continue
            f = forecast(k, issue, lead)
            comp_h.append(loyo(h))
            row = dict(key=k, label=CUR[k][3], hc_members=int(np.median(nh)), normal=float(h.mean()))
            if f is None:
                ok_f = False
            else:
                comp_f.append(f.mean() - h.mean())
                an = f - h.mean()
                row.update(
                    members=len(f),
                    mean=float(f.mean()),
                    anom=float(f.mean() - h.mean()),
                    anom_p10=float(np.percentile(an, 10)),
                    anom_p90=float(np.percentile(an, 90)),
                    members_anom=[round(float(v), 2) for v in an],
                )
            comp_rows.append(row)
        if not ok_h:
            continue
        hc_an[c] = np.mean(comp_h, axis=0)
        r = dict(centre=c, components=comp_rows, has_issue=ok_f)
        if ok_f:
            fc_an[c] = float(np.mean(comp_f))
            present.append(c)
            r["anom"] = fc_an[c]
        rows.append(r)
    # the predictor: the centres present in this issue (or, while waiting, every expected centre)
    use = present if len(present) >= QUORUM else expected
    use = [c for c in use if c in hc_an]
    x = np.mean([hc_an[c] for c in use], axis=0)
    cal = calibration_block(x, o, YEARS)
    pre = prereg(s)
    # the current composition at this start month vs the tested one
    for r in rows:
        ps = pre.get("scores", {}).get(r["centre"])
        changed = [(CUR[k["key"]][3], CUR[k["key"]][5]) for k in r["components"] if CUR[k["key"]][5] != k["key"]]
        r["tested"] = (
            dict(ps or {}, system_changed=[f"{a} (tested: {b.replace('_', ' ')})" for a, b in changed]) if ps else None
        )
        try:
            from c3s_u10_skill import score

            r["current_hindcast"] = {
                k: v for k, v in score(hc_an[r["centre"]], loyo(o)).items() if k in ("r", "r_ci95", "p")
            }
        except Exception:  # noqa: BLE001
            pass
    state = "waiting"
    fc = None
    if len(present) >= QUORUM and cal["significant"]:
        state = "complete" if set(present) >= set(expected) else "provisional"
        a, b, sres, xb, sxx = (cal["regression"][k] for k in ("a", "b", "s", "xbar", "sxx"))
        xf = float(np.mean([fc_an[c] for c in use]))
        mu, sc, df = predictive(xf, a, b, sres, xb, sxx, N)
        pr = probs(mu, sc, df, *terc)
        q = {f"p{int(100 * qq)}": float(mu + sc * tppf(qq, df)) for qq in (0.1, 0.25, 0.5, 0.75, 0.9)}
        fc = dict(
            mmm_anom=xf,
            mean=float(mu),
            scale=float(sc),
            df=int(df),
            quantiles=q,
            p_weak=float(pr[0]),
            p_normal=float(pr[1]),
            p_strong=float(pr[2]),
            centres=use,
        )
    elif len(present) >= QUORUM:
        state = "no_skill"
    out = dict(
        generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        issue=key,
        start_month=s,
        target="December",
        target_year=y,
        lead_month=lead,
        state=state,
        centres_expected=expected,
        centres_present=present,
        quorum=QUORUM,
        centres_used=use,
        truth=dict(
            source="MERRA-2 (NASA GMAO), December mean of daily u 60°N 10 hPa",
            years=[YEARS[0], YEARS[-1]],
            mean=float(o.mean()),
            sd=float(o.std(ddof=1)),
            terciles=terc,
            values=[dict(year=yy, u=float(v)) for yy, v in zip(YEARS, o)],
        ),
        forecast=fc,
        skill_current=cal,
        prereg=pre,
        systems=rows,
        note=(
            "Forecast ensembles are larger than the hindcast ones: the regression slope fitted on the noisier hindcast "
            "predictor is slightly too small for the forecast (a conservative forecast)."
        ),
    )

    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(w) for k, w in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(w) for w in v]
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else round(float(v), 4)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(clean(out), ensure_ascii=False, separators=(",", ":")))
    print(
        f"wrote {OUT_JSON} ({state}; centres present {present or 'none'}; predictor {use}) in {time.time() - t0:.0f} s",
        flush=True,
    )
    print(
        f"  hindcast r {cal['r']:.2f} {cal['r_ci95']}, p {cal['p']:.4f}; LOO RPSS {cal['loo']['rpss']:.2f} "
        f"{cal['loo']['rpss_ci95']} p {cal['loo']['rpss_p']:.4f}; 80 % cover {cal['loo']['cover80']}/{N}",
        flush=True,
    )
    if fc:
        print(
            f"  forecast: MMM anomaly {fc['mmm_anom']:+.2f} m/s → {fc['mean']:.1f} m/s; P weak {fc['p_weak']:.2f}, "
            f"normal {fc['p_normal']:.2f}, strong {fc['p_strong']:.2f}",
            flush=True,
        )
    return out


def done(issue: str) -> bool:
    """c3s_status: the card holds this issue with every expected centre, or the issue has no December target."""
    s = int(issue[4:])
    if s not in (8, 9, 10, 11, 12):
        return True
    try:
        J = json.loads(OUT_JSON.read_text())
    except Exception:  # noqa: BLE001
        return False
    return J.get("issue") == f"{issue[:4]}-{issue[4:]}" and set(J.get("centres_present", [])) >= set(
        J.get("centres_expected", ["?"])
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch-hc", "fetch", "build", "done"])
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    a = ap.parse_args(argv)
    if a.cmd == "fetch-hc":
        return fetch_hc()
    if a.cmd == "fetch":
        return fetch(a.issue)
    if a.cmd == "done":
        print("1" if done(a.issue) else "0")
        return 0
    build(a.issue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
