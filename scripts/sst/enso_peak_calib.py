#!/usr/bin/env python3
"""Can the ENSO forecasts page's peak-strength odds be calibrated? A test on the C3S hindcasts.

THE ODDS. enso-forecasts.html gives the chance that the event's peak reaches +1.5, +2.0, +2.5 °C (and higher) on
the ONI and on RONI as the share of the C3S members whose own peak gets there, each model weighted equally. A
member's peak is the highest of its four 3-month means lying wholly inside the six-month forecast (centred on leads
2-5), on each index exactly as the plume builds it (c3s_nino34.py). Raw ensemble shares tend to be overconfident and
biased, so the page labels them raw.

THE HINDCAST. For each start month: every kept system's 1993-2016 reforecasts, the plume's own two files (the 1°
Niño-3.4 box and the 5° 20°S-20°N band, as c3s_nino34.climatology() fetches them), read message by message
(c3s_grib) and filed under the year of the NOMINAL start (indexingDate: a lagged member started the month before
belongs to the start month's year). Each member becomes an anomaly the way the forecast does, against its system's
hindcast mean per lead, except that the verified year is left out of that mean (the live forecast's own year is
outside its climatology too); the relative index is rescaled per valid month by σ(ONI)/σ(relative)
(roni_sigma.json); ECCC's two components are merged into CanSIPS. Verification: CPC's official ONI and RONI
(cpc_official.json) over the same four seasons.

THE CALIBRATION. One shift-and-scale of the multi-model peak distribution, three numbers per index and start month:
each member's peak x becomes a + b·m + c·(x − m), where m is the model-weighted mean peak; b rescales the signal and
c the spread. Fitted by minimum continuous ranked probability score (CRPS) over the hindcast years.

THE TEST. Leave one year out: fit on 23 years, forecast the 24th, for every year. Score: the ranked probability score
(RPS) over the cumulative levels +0.5 ... +2.5 °C (weak, moderate, strong, very strong, and the page's +2.5). Skill
of the calibrated odds against the raw ones, RPSS = 1 − ΣRPS_cal / ΣRPS_raw over the years, with a 95 % moving-block
bootstrap interval (blocks of three years) and a one-sided permutation p (raw and calibrated swapped at random within
each year). Rule: the calibrated odds replace the raw ones for an index only if RPSS > 0, the interval lies above zero
and p < 0.05/24 -- Bonferroni over the 24 tests (12 start months, two indices): at 0.05 each, a chance pass somewhere
would be more likely than not. Otherwise the page keeps the raw shares, labelled raw, and quotes the test. A passed
calibration is applied only to an issue built from the same systems it was tested on.

Outputs
  scripts/sst/reference/enso_peak_calib_MM.json  member and observed peaks per hindcast year, the test, and the
                                                 calibration fitted on all years (committed; one per start month)
  assets/sst/data/enso_peak_calib.json           the issue's start month: the verdict for the page

Runs in c3s.yml after the plume (with --fetch: a start month without a reference is built there, its missing
hindcast files fetched one at a time) and in sst.yml (the page file only, from the committed references).

    python scripts/sst/enso_peak_calib.py                 # the page file for the issue in enso_forecast.json
    python scripts/sst/enso_peak_calib.py --fetch         # ... building a missing reference (Actions)
    python scripts/sst/enso_peak_calib.py --build 9 10    # (re)build references from the cached hindcast files
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c3s_nino34 import (  # one model registry, one reader, one climatology layout
    AREA_N34,
    AREA_TROP,
    CLIM_YEARS,
    DATA,
    GRID_N34,
    GRID_TROP,
    MAXLEAD,
    MERGE,
    MODELS as _ALL_MODELS,
    N34_LAT,
    N34_LON,
    TROP_LAT,
    _box_series,
    _clim_rows,
    _load_clim,
    _month_of_lead,
    _retrieve,
    _roni_scale_table,
)
from enso_models import EXCLUDE_SYSTEMS

_EXCL_PARTS = {p for m, parts in MERGE.items() if m in EXCLUDE_SYSTEMS for p in parts}
MODELS = [m for m in _ALL_MODELS if m[2] not in EXCLUDE_SYSTEMS and m[2] not in _EXCL_PARTS]

SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE.parents[1]
FC_JSON = SITE_ROOT / "assets" / "sst" / "data" / "enso_forecast.json"
CPC_JSON = SITE_ROOT / "assets" / "sst" / "data" / "cpc_official.json"
OUT = SITE_ROOT / "assets" / "sst" / "data" / "enso_peak_calib.json"
REF = HERE / "reference"

LEVELS = (0.5, 1.0, 1.5, 2.0, 2.5)  # the RPS's cumulative levels: weak, moderate, strong, very strong, the page's +2.5
PAGE_LEVELS = (1.5, 2.0, 2.5)  # the page's own table
CENTRES = range(2, MAXLEAD)  # leads at the centre of a 3-month mean wholly inside the forecast (2..5)
INDICES = {"oni": "ONI", "roni": "RONI"}
N_BOOT, N_PERM, BLOCK = 10000, 20000, 3
P_MAX = 0.05 / 24  # Bonferroni: 12 start months x two indices
SEED = 1993
# reliability bins of the forecast probability: near-certain no, lean no, toss-up, lean yes, near-certain yes
BINS = (0.0, 0.05, 0.35, 0.65, 0.95, 1.0000001)


def ref_path(month: int) -> Path:
    return REF / f"enso_peak_calib_{month:02d}.json"


# ── hindcast member peaks ─────────────────────────────────────────────────────
def _peak(a: np.ndarray) -> np.ndarray:
    """(member, lead 1..MAXLEAD) -> each member's highest 3-month mean centred on CENTRES."""
    return np.max([a[:, L - 2 : L + 1].mean(1) for L in CENTRES], axis=0)


def _system_peaks(centre: str, system: str, month: int, fetch: bool) -> dict | None:
    """{year: {"oni": member peaks, "roni": member peaks}} for one system's hindcast, None if a file is missing.

    Built as c3s_nino34.model_members builds a forecast, with the climatology per lead taken over every member of
    the OTHER years. The full-period climatology must reproduce the plume's committed one (c3s_nino34_clim.csv)."""
    files = []
    for suffix, area, grid in (("_n34", AREA_N34, GRID_N34), ("", AREA_TROP, GRID_TROP)):
        p = DATA / "hindcast" / f"{centre}_{system}_{month:02d}{suffix}.grib"
        if not p.exists() and not (fetch and _retrieve(centre, system, CLIM_YEARS, f"{month:02d}", p, area, grid)):
            print(f"  {centre}_{system} {month:02d}: no hindcast file {p.name}", flush=True)
            return None
        files.append(p)
    keys = ("number", "dataDate", "dataTime", "forecastMonth", "indexingDate")
    n34 = _box_series(files[0], N34_LAT, N34_LON, keys)
    trop = _box_series(files[1], TROP_LAT, None, keys)
    if n34.keys() != trop.keys():
        raise ValueError(f"{centre}_{system} {month:02d}: the 1° and 5° hindcast files hold different messages")
    by_member: dict = {}
    for (num, dd, dt, L, idx), v in n34.items():
        by_member.setdefault((num, dd, dt, idx), {})[L] = (v, trop[(num, dd, dt, L, idx)])
    rows, years = [], []
    for k, leads in sorted(by_member.items()):
        vals = [leads.get(L, (np.nan, np.nan)) for L in range(1, MAXLEAD + 1)]
        if np.isfinite(vals).all():
            rows.append(vals)
            years.append((k[3] or k[1]) // 10000)  # the nominal start's year
    x = np.array(rows)  # (member, lead, [n34, trop]) in °C
    years = np.array(years)
    uy = sorted(set(years.tolist()))
    if uy != [int(y) for y in CLIM_YEARS]:
        raise ValueError(f"{centre}_{system} {month:02d}: hindcast years {uy[0]}-{uy[-1]} ({len(uy)}), not 1993-2016")
    counts = {int(y): int((years == y).sum()) for y in uy}
    if len(set(counts.values())) > 1:
        print(f"  {centre}_{system} {month:02d}: members per year vary {min(counts.values())}-{max(counts.values())}")
    # the full-period climatology against the plume's own (rounded to 4 decimals there)
    full = x.mean(0)
    cached = _clim_rows(_load_clim(), f"{centre}_{system}", month)
    if cached is not None:
        dev = max(np.abs(full[:, 0] - cached[0]).max(), np.abs(full[:, 1] - cached[1]).max())
        if dev > 0.002:
            raise ValueError(
                f"{centre}_{system} {month:02d}: hindcast mean is {dev:.3f} °C off the plume's climatology"
            )
    scale = _roni_scale_table()
    sc = np.array([scale.get(_month_of_lead(month, L), 1.0) for L in range(1, MAXLEAD + 1)])
    tot, n = x.sum(0), len(x)
    out = {}
    for y in uy:
        sel = years == y
        clim = (tot - x[sel].sum(0)) / (n - sel.sum())  # leave the verified year out
        a_n34 = x[sel, :, 0] - clim[None, :, 0]
        a_rel = ((x[sel, :, 0] - x[sel, :, 1]) - (clim[None, :, 0] - clim[None, :, 1])) * sc[None]
        out[y] = {"oni": _peak(a_n34), "roni": _peak(a_rel)}
    return out


def hindcast_peaks(month: int, fetch: bool = False) -> dict | None:
    """{label: {year: {"oni", "roni"}}} for every kept system (ECCC merged), None unless all are complete."""
    res = {}
    for centre, system, label, _colour in MODELS:
        r = _system_peaks(centre, system, month, fetch)
        if r is None:
            return None
        res[label] = r
    for merged, parts in MERGE.items():
        if merged in EXCLUDE_SYSTEMS or not all(p in res for p in parts):
            continue
        res[merged] = {y: {k: np.concatenate([res[p][y][k] for p in parts]) for k in INDICES} for y in res[parts[0]]}
        for p in parts:
            del res[p]
    return res


def observed_peaks(month: int, years: list[int]) -> dict:
    """CPC's official ONI and RONI peak over the four seasons the forecast's peak is taken from, per year."""
    cpc = json.loads(CPC_JSON.read_text())
    out = {}
    for k in INDICES:
        vals = []
        for y in years:
            ms = [f"{y + (month - 1 + L - 1) // 12}-{_month_of_lead(month, L):02d}" for L in CENTRES]
            v = [cpc[k].get(m) for m in ms]
            if any(q is None for q in v):
                raise ValueError(f"CPC {k} has no value for one of {ms}")
            vals.append(max(v))
        out[k] = vals
    return out


# ── probabilities, scores, the calibration ────────────────────────────────────
def _ensemble(models: dict, k: str, y: int) -> tuple[np.ndarray, np.ndarray]:
    """(member peaks, weights): every model weighs 1/K, shared among its members, as on the page."""
    xs, ws = [], []
    for m in models.values():
        v = np.asarray(m[y][k], float)
        xs.append(v)
        ws.append(np.full(v.size, 1.0 / (v.size * len(models))))
    return np.concatenate(xs), np.concatenate(ws)


def exceed(x: np.ndarray, w: np.ndarray, levels=LEVELS) -> np.ndarray:
    return np.array([w[x >= t].sum() for t in levels])


def _mean_abs_diff(x: np.ndarray, w: np.ndarray) -> float:
    """Σ_ij w_i w_j |x_i − x_j| in O(n log n)."""
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    cw, cs = np.cumsum(ws) - ws, np.cumsum(ws * xs) - ws * xs  # sums over the members below each one
    return float(2.0 * np.sum(ws * (xs * cw - cs)))


def transform(x: np.ndarray, w: np.ndarray, p) -> np.ndarray:
    a, b, c = p
    m = float(np.dot(w, x))
    return a + b * m + c * (x - m)


def fit(ens: list, obs: np.ndarray) -> tuple[float, float, float]:
    """(a, b, c) minimising the mean CRPS of the transformed ensembles over the years given."""
    from scipy.optimize import minimize

    D = np.array([_mean_abs_diff(x, w) for x, w in ens])

    def loss(q):
        a, b, lc = q
        c = np.exp(lc)
        return np.mean(
            [np.dot(w, np.abs(transform(x, w, (a, b, c)) - y)) - 0.5 * c * d for (x, w), y, d in zip(ens, obs, D)]
        )

    r = minimize(loss, x0=(0.0, 1.0, 0.0), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-7, maxiter=4000))
    a, b, lc = r.x
    return float(a), float(b), float(np.exp(lc))


def rps(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Per-year RPS over the cumulative levels: Σ_t (P(peak ≥ t) − 1[observed ≥ t])²."""
    return ((p - o) ** 2).sum(-1)


def _skill(num: np.ndarray, den: np.ndarray) -> float:
    return float(1.0 - num.sum() / den.sum())


def _blocks(n: int, rng) -> np.ndarray:
    """Moving-block (circular) bootstrap indices, blocks of BLOCK years."""
    starts = rng.integers(0, n, size=(N_BOOT, -(-n // BLOCK)))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None]) % n
    return idx.reshape(N_BOOT, -1)[:, :n]


def reliability(p: np.ndarray, o: np.ndarray) -> list[dict]:
    """Pooled over years and levels: per forecast-probability bin, the count, mean forecast and observed share."""
    p, o = p.ravel(), o.ravel()
    out = []
    for lo, hi in pairwise(BINS):
        s = (p >= lo) & (p < hi)
        out.append(
            dict(
                bin=[lo, round(min(hi, 1.0), 2)],
                n=int(s.sum()),
                forecast=round(float(p[s].mean()), 3) if s.any() else None,
                observed=round(float(o[s].mean()), 3) if s.any() else None,
            )
        )
    return out


def _rel_term(p: np.ndarray, o: np.ndarray) -> float:
    """The reliability term of the Brier decomposition over the BINS (0 = perfectly reliable)."""
    p, o = p.ravel(), o.ravel()
    t = 0.0
    for lo, hi in pairwise(BINS):
        s = (p >= lo) & (p < hi)
        if s.any():
            t += s.sum() * (p[s].mean() - o[s].mean()) ** 2
    return float(t / p.size)


def test_index(models: dict, years: list[int], obs: list[float], k: str) -> dict:
    """Leave-one-year-out: raw, calibrated and climatological odds per year, their scores, the verdict."""
    n = len(years)
    ens = [_ensemble(models, k, y) for y in years]
    ob = np.asarray(obs, float)
    o = (ob[:, None] >= np.array(LEVELS)[None]).astype(float)
    p_raw = np.array([exceed(x, w) for x, w in ens])
    p_cal = np.zeros_like(p_raw)
    p_clim = np.zeros_like(p_raw)
    params = []
    for i in range(n):
        tr = np.arange(n) != i
        q = fit([ens[j] for j in np.where(tr)[0]], ob[tr])
        params.append(q)
        x, w = ens[i]
        p_cal[i] = exceed(transform(x, w, q), w)
        p_clim[i] = o[tr].mean(0)
    r_raw, r_cal, r_clim = rps(p_raw, o), rps(p_cal, o), rps(p_clim, o)
    rng = np.random.default_rng(SEED)
    bi = _blocks(n, rng)
    boot = 1.0 - r_cal[bi].sum(1) / r_raw[bi].sum(1)
    boot_raw = 1.0 - r_raw[bi].sum(1) / r_clim[bi].sum(1)
    boot_cal = 1.0 - r_cal[bi].sum(1) / r_clim[bi].sum(1)
    swap = rng.random((N_PERM, n)) < 0.5
    num = np.where(swap, r_raw[None], r_cal[None]).sum(1)
    den = np.where(swap, r_cal[None], r_raw[None]).sum(1)
    obs_skill = _skill(r_cal, r_raw)
    p_perm = float((1 + np.sum(1.0 - num / den >= obs_skill)) / (N_PERM + 1))

    def ci(b):
        lo, hi = np.percentile(b, [2.5, 97.5])
        return [round(float(lo), 3), round(float(hi), 3)]

    ci_cal = ci(boot)
    passed = bool(obs_skill > 0 and p_perm < P_MAX and ci_cal[0] > 0)
    a, b, c = fit(ens, ob)  # the calibration the page would apply: every year
    brier = {}
    for t in PAGE_LEVELS:
        j = LEVELS.index(t)
        brier[f"{t:.1f}"] = dict(
            raw=round(float(((p_raw[:, j] - o[:, j]) ** 2).mean()), 4),
            cal=round(float(((p_cal[:, j] - o[:, j]) ** 2).mean()), 4),
            events=int(o[:, j].sum()),
        )
    return dict(
        rpss_cal_vs_raw=round(obs_skill, 3),
        rpss_cal_vs_raw_ci=ci_cal,
        p_perm=float(f"{p_perm:.2g}"),
        rpss_raw_vs_clim=round(_skill(r_raw, r_clim), 3),
        rpss_raw_vs_clim_ci=ci(boot_raw),
        rpss_cal_vs_clim=round(_skill(r_cal, r_clim), 3),
        rpss_cal_vs_clim_ci=ci(boot_cal),
        rps_mean=dict(
            raw=round(float(r_raw.mean()), 4), cal=round(float(r_cal.mean()), 4), clim=round(float(r_clim.mean()), 4)
        ),
        cal_better_years=int((r_cal < r_raw).sum()),
        raw_better_years=int((r_raw < r_cal).sum()),
        brier_page_levels=brier,
        reliability=dict(raw=reliability(p_raw, o), cal=reliability(p_cal, o)),
        reliability_term=dict(raw=round(_rel_term(p_raw, o), 4), cal=round(_rel_term(p_cal, o), 4)),
        loyo_params=dict(
            a=[round(min(q[0] for q in params), 3), round(max(q[0] for q in params), 3)],
            b=[round(min(q[1] for q in params), 3), round(max(q[1] for q in params), 3)],
            c=[round(min(q[2] for q in params), 3), round(max(q[2] for q in params), 3)],
        ),
        calibration=dict(a=round(a, 4), b=round(b, 4), c=round(c, 4)),
        passed=passed,
        by_year={
            str(y): dict(
                observed=round(float(ob[i]), 2),
                mean_peak=round(float(np.dot(ens[i][1], ens[i][0])), 2),
                raw=[round(float(v), 3) for v in p_raw[i]],
                cal=[round(float(v), 3) for v in p_cal[i]],
                rps_raw=round(float(r_raw[i]), 4),
                rps_cal=round(float(r_cal[i]), 4),
            )
            for i, y in enumerate(years)
        },
    )


# ── reference per start month ─────────────────────────────────────────────────
def build_reference(month: int, fetch: bool = False) -> dict | None:
    """Member peaks, observed peaks and the test for one start month -> reference/enso_peak_calib_MM.json."""
    print(f"peak-odds calibration, start month {month:02d}", flush=True)
    models = hindcast_peaks(month, fetch)
    if models is None:
        return None
    years = sorted(set.intersection(*[set(m) for m in models.values()]))
    obs = observed_peaks(month, years)
    ref = dict(
        start_month=month,
        years=years,
        levels=list(LEVELS),
        models={
            label: dict(
                n_members=int(np.median([len(m[y]["oni"]) for y in years])),
                **{k: {str(y): [round(float(v), 3) for v in m[y][k]] for y in years} for k in INDICES},
            )
            for label, m in models.items()
        },
        observed={k: [round(float(v), 2) for v in obs[k]] for k in INDICES},
        verification="CPC official ONI and RONI (cpc_official.json), highest of the four seasons centred on leads 2-5",
        test={k: test_index(models, years, obs[k], k) for k in INDICES},
    )
    REF.mkdir(parents=True, exist_ok=True)
    ref_path(month).write_text(json.dumps(ref, separators=(",", ":")))
    for k, t in ref["test"].items():
        print(
            f"  {INDICES[k]:4s} RPSS cal vs raw {t['rpss_cal_vs_raw']:+.3f} {t['rpss_cal_vs_raw_ci']} p {t['p_perm']}"
            f" | raw vs clim {t['rpss_raw_vs_clim']:+.3f}, cal vs clim {t['rpss_cal_vs_clim']:+.3f}"
            f" | calibration {t['calibration']} -> {'PASSED' if t['passed'] else 'raw kept'}",
            flush=True,
        )
    return ref


# ── the page file ─────────────────────────────────────────────────────────────
FC_KEY = {"oni": "n34", "roni": "rnino"}  # enso_forecast.json's member arrays behind each index
ON = {"oni": "the ONI", "roni": "RONI"}  # as the page words them
BIN_WORDS = ("under 5%", "5–35%", "35–65%", "65–95%", "95% or more")


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.2g}"


def _sgn(v: float, d: int = 2) -> str:
    return f"{v:+.{d}f}".replace("-", "−")  # a true minus sign, as on the page


def _skill(t: dict, key: str, first: bool = True, p: bool = False) -> str:
    """'+0.03 (95% interval −0.62 to +0.44, p = 0.46)': a skill score, its interval and (p) the permutation p."""
    lo, hi = t[f"{key}_ci"]
    s = f"{_sgn(t[key])} ({'95% interval ' if first else ''}{_sgn(lo)} to {_sgn(hi)}"
    return s + (f", p {_fmt_p(t['p_perm'])})" if p else ")")


def _vs_clim(t: dict, name: str, first: bool) -> str:
    s = f"{_skill(t, 'rpss_raw_vs_clim', first)} on {name}"
    return s if t["rpss_raw_vs_clim_ci"][0] > 0 else s + ", no better than climatology at 95%"


def _issue_mean_peak(fc: dict, k: str) -> float:
    """The issue's model-weighted mean member peak on one index, as the hindcast's `mean_peak`."""
    labs = [m for m in fc["models"] if m not in EXCLUDE_SYSTEMS]
    return float(np.mean([_peak(np.array(fc["models"][m][FC_KEY[k]]).T).mean() for m in labs]))


def _brief_text(ref: dict, idx: dict, mname: str, fc: dict) -> str:
    """The note under the odds table."""
    t = ref["test"]
    used = [k for k in INDICES if idx[k]["applied"]]
    held = [k for k in INDICES if idx[k]["passed"] and not idx[k]["applied"]]
    failed = [k for k in INDICES if not idx[k]["passed"]]
    where = f"the {mname} hindcasts, {ref['years'][0]}–{ref['years'][-1]}"
    beat = " and ".join(f"{_skill(t[k], 'rpss_cal_vs_raw', i == 0, p=True)} on {ON[k]}" for i, k in enumerate(failed))
    if not used and not held:
        out = [
            f"These are the raw shares of members. A calibration fitted on {where} (a shift and scale of every "
            f"member's peak, each year left out in turn) did not beat them: its ranked probability skill against them "
            f"is {beat}, not significant."
        ]
    else:
        out = [
            f"The {INDICES[k]} column is calibrated on {where}: out of sample it beats the raw shares, with a ranked "
            f"probability skill of {_skill(t[k], 'rpss_cal_vs_raw', p=True)}, significant allowing for 24 such tests."
            for k in used
        ]
        out += [
            f"The {INDICES[k]} calibration passed its test on {', '.join(ref['models'])}, but this issue is built from "
            f"other systems, so that column stays raw."
            for k in held
        ]
        if failed:
            out.append(f"The {INDICES[failed[0]]} column stays raw: calibration did not beat it, scoring {beat}.")
        for k in used:
            now = _issue_mean_peak(fc, k)
            top = max(v["mean_peak"] for v in t[k]["by_year"].values())
            if now > top:
                out.append(
                    f"This year's average {INDICES[k]} peak, {_sgn(now, 1)} °C, lies beyond every hindcast year's "
                    f"(highest {_sgn(top, 1)}), so its calibration is extrapolated."
                )
    top = {k: max(ref["observed"][k]) for k in INDICES}
    out.append(
        f"Against climatology the raw odds score {_vs_clim(t['oni'], 'the ONI', True)} and "
        f"{_vs_clim(t['roni'], 'RONI', False)}. The highest peak in those years was {_sgn(top['oni'], 1)} °C on the "
        f"ONI and {_sgn(top['roni'], 1)} on RONI, so the odds of higher levels are untested."
    )
    return " ".join(out)


def _method_text(ref: dict, mname: str) -> str:
    """The numbers in the method section: both tests, and how often the raw odds came true."""
    t = ref["test"]
    n = len(ref["years"])

    def rel(k):
        return "; ".join(
            f"{w}, {round(100 * b['observed'])}% ({b['n']})"
            for w, b in zip(BIN_WORDS, t[k]["reliability"]["raw"])
            if b["n"]
        )

    tested = sorted(int(p.stem[-2:]) for p in REF.glob("enso_peak_calib_??.json"))
    passed = [m for m in tested if any(json.loads(ref_path(m).read_text())["test"][k]["passed"] for k in INDICES)]

    def names(ms):
        return ", ".join(pd.Timestamp(2000, m, 1).strftime("%B") for m in ms)

    return (
        f"For {mname} starts ({n} hindcast years, {ref['years'][0]}–{ref['years'][-1]}; {', '.join(ref['models'])}), "
        f"the skill of the calibrated odds against the raw ones, out of sample, is "
        f"{_skill(t['oni'], 'rpss_cal_vs_raw', p=True)} on the ONI and "
        f"{_skill(t['roni'], 'rpss_cal_vs_raw', False, p=True)} on RONI; the raw odds score "
        f"{_vs_clim(t['oni'], 'the ONI', True)} and {_vs_clim(t['roni'], 'RONI', False)} against climatology. "
        f"How often the raw odds came true, pooled over the five levels and {n} years (forecasts in brackets): ONI odds "
        f"{rel('oni')}; RONI odds {rel('roni')}. Start months tested so far: {names(tested)}; the calibration passed in "
        + (f"{names(passed)}." if passed else "none of them.")
    )


def page_file(fetch: bool) -> dict:
    """The verdict for the issue's start month -> assets/sst/data/enso_peak_calib.json."""
    fc = json.loads(FC_JSON.read_text())
    issue, month = fc["issue"], int(fc["issue"][5:7])
    mname = pd.Timestamp(2000, month, 1).strftime("%B")
    p = ref_path(month)
    ref = json.loads(p.read_text()) if p.exists() else build_reference(month, fetch) if fetch else None
    out = dict(
        generated=pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        issue=issue,
        start_month=month,
        tested=ref is not None,
    )
    if ref is None:
        out["text"] = (
            f"These are the raw shares of members; their calibration has not been tested on the {mname} hindcasts yet."
        )
    else:
        same = sorted(ref["models"]) == sorted(m for m in fc["models"] if m not in EXCLUDE_SYSTEMS)
        idx = {}
        for k, t in ref["test"].items():
            idx[k] = {q: t[q] for q in ("rpss_cal_vs_raw", "rpss_cal_vs_raw_ci", "p_perm", "passed")}
            idx[k]["applied"] = bool(t["passed"] and same)  # tested on the systems this issue is built from
            if idx[k]["applied"]:
                idx[k]["calibration"] = t["calibration"]
        out.update(
            years=[ref["years"][0], ref["years"][-1]],
            models=list(ref["models"]),
            indices=idx,
            text=_brief_text(ref, idx, mname, fc),
            method=_method_text(ref, mname),
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(
        f"saved {OUT.name}: issue {issue}, "
        + (out["text"] if ref is None else str({k: v["applied"] for k, v in idx.items()}))
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="+", type=int, metavar="MONTH", help="(re)build these start months' references")
    ap.add_argument("--fetch", action="store_true", help="fetch missing hindcast files (Actions)")
    args = ap.parse_args()
    if args.build:
        bad = [m for m in args.build if build_reference(m, args.fetch) is None]
        return 1 if bad else 0
    page_file(args.fetch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
