#!/usr/bin/env python3
"""Wave reflection and downward propagation on a large CMIP6 sample: events in every run's daily series
(cmip6_coupling_extract.py), what followed at the surface, and whether it holds across models.

ANOMALIES. Each run against itself, on its own calendar (365-day, 360-day or Gregorian years): four annual harmonics and
a linear trend for the mean; for the standardised series a three-harmonic fit to the squared residual for the variance.
  ssi        the SSW imprint index (ssw_imprint.py): the projected sea-level pressure less its fit; the pattern made
             without the model for the nine models that made it
  cap10, cap100   standardised polar-cap height at 10 and 100 hPa (high = weak vortex)
  zsib, zcan      standardised regional v*T* at 100 hPa; ri = zsib - zcan (Messori et al. 2022)
  tna, teu        2 m temperature anomaly, North America and northern Eurasia (K)

EVENTS (onsets; November-March unless said):
  ssw          u(60N, 10 hPa) easterly (Charlton & Polvani 2007: a return to 10 westerly days before 30 April, 20
               westerly days between events), split two ways:
                 absorbing / reflecting   the 100 hPa [v'T'] over days +5..+15 above / below the model's median
                 (Kodera et al. 2016's recovery-phase wave driving)
                 reflect_k1               the wave-1 [v'T'] at 50 hPa (5-day mean) negative on some day +1..+20
  refl_zm      wave-1 [v'T'] at 50 hPa, 5-day mean, below zero (net downward wave-1 activity; Shaw & Perlwitz 2013,
               Dunn-Sigouin & Shaw 2015), December-March, 20 days apart
  refl_reg     ri > 1 for 10 days or more, December-March, with upward flux over Siberia and downward over Canada on
               average over the event (Messori et al. 2022; their third criterion, a reflective upper stratosphere,
               needs winds above 10 hPa, which the day table does not have)
  cap100_td / cap100_bu   the 100 hPa cap first reaching +1 sd, 30 days apart, with the 10 hPa cap at +1 sd in the
               3-20 days before (top-down) or not (bottom-up)

TESTS. Per model, each outcome's event mean less its mean over random days of the same calendar months in the same
runs (NULL draws): the model's effect. Across models (models weighted equally, those with >= MIN_EVENTS events): the
mean effect, a one-sample t-test, the share of models agreeing on its sign; robust = Benjamini-Hochberg FDR 10 % over
every test and >= 80 % sign agreement, the site's CMIP6 rule.

    python cmip6_coupling.py --runs extracts/ --out scripts/strat/reference/cmip6_coupling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WINTER = (11, 12, 1, 2, 3)
DJFM = (12, 1, 2, 3)
MIN_EVENTS = 5
NULL = 400
WINDOWS = {"d-5_0": (-5, 0), "d1_10": (1, 10), "d11_20": (11, 20), "d1_30": (1, 30), "d31_60": (31, 60)}
OUTCOMES = ("ssi", "tna", "teu", "cap10", "cap100")


# ---------------------------------------------------------------------------------------------------- calendar ---
def year_fraction(y, m, d, calendar: str):
    """(day of year - 1) / days in that year, in the run's own calendar."""
    y, m, d = (np.asarray(x, int) for x in (y, m, d))
    if calendar == "360_day":
        return ((m - 1) * 30 + d - 1) / 360.0, np.full(len(y), 360)
    if calendar in ("noleap", "365_day"):
        leap = np.zeros(len(y), bool)
    elif calendar in ("all_leap", "366_day"):
        leap = np.ones(len(y), bool)
    else:  # standard, gregorian, proleptic_gregorian, julian
        leap = (y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0)) if calendar != "julian" else y % 4 == 0
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    doy = cum[m - 1] + d + (leap & (m > 2))
    n = np.where(leap, 366, 365)
    return (doy - 1) / n, n


def design(frac, years, nh: int, trend: bool):
    x = 2 * np.pi * frac
    cols = [np.ones(len(frac))] + [f(k * x) for k in range(1, nh + 1) for f in (np.cos, np.sin)]
    if trend:
        t = years + frac
        cols.append((t - t.mean()) / 100.0)
    return np.column_stack(cols)


def anomaly(y, X):
    ok = np.isfinite(y)
    b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    return y - X @ b


def standardise(y, X, Xs):
    r = anomaly(y, X)
    ok = np.isfinite(r)
    r2 = np.minimum(r**2, np.nanpercentile(r**2, 99.9))
    v = Xs @ np.linalg.lstsq(Xs[ok], r2[ok], rcond=None)[0]
    return r / np.sqrt(np.maximum(v, np.nanmax(v) / 9.0))


# ------------------------------------------------------------------------------------------------------ series ---
def load_run(path: Path) -> pd.DataFrame:
    z = np.load(path)
    cal = str(z["calendar"])
    y, m, d = z["year"].astype(int), z["month"].astype(int), z["day"].astype(int)
    frac, _ = year_fraction(y, m, d, cal)
    X4, X3 = design(frac, y, 4, True), design(frac, y, 3, False)
    model = path.stem.split("_")[0]
    pattern = "slp_loo"  # the leave-one-out projection (= slp_p for a model outside the composite)
    f = pd.DataFrame({"year": y, "month": m, "day": d})
    f["u10"] = z["u10"]
    f["ssi"] = anomaly(z[pattern].astype(float), X4)
    f["cap10"] = standardise(z["z10"].astype(float), X4, X3)
    f["cap100"] = standardise(z["z100"].astype(float), X4, X3)
    f["zsib"] = standardise(z["hf100_sib"].astype(float), design(frac, y, 4, False), X3)
    f["zcan"] = standardise(z["hf100_can"].astype(float), design(frac, y, 4, False), X3)
    f["ri"] = f.zsib - f.zcan
    f["hf100_tot"] = z["hf100_tot"]
    f["hf50_k1"] = z["hf50_k1"]
    f["hf50_k1_5"] = f.hf50_k1.rolling(5, center=True).mean()
    f["sib"], f["can"] = z["hf100_sib"], z["hf100_can"]
    f["tna"] = anomaly(z["tas_na"].astype(float), X4)
    f["teu"] = anomaly(z["tas_eu"].astype(float), X4)
    f.attrs = {"model": model, "run": path.stem, "calendar": cal}
    return f


# ------------------------------------------------------------------------------------------------------ events ---
def ssw_onsets(f: pd.DataFrame) -> list[int]:
    """Charlton & Polvani (2007) on positions: easterly u(60N, 10 hPa) November-March, a return to 10 consecutive
    westerly days by 30 April (else a final warming, and nothing later that winter counts), 20 westerly days between."""
    u, mo, yr = f.u10.values, f.month.values, f.year.values
    winter_of = np.where(mo >= 7, yr, yr - 1)
    out, next_ok = [], 0
    for w in np.unique(winter_of):
        idx = np.flatnonzero((winter_of == w) & np.isin(mo, WINTER))
        spring = np.flatnonzero((winter_of == w) & np.isin(mo, (*WINTER, 4)))
        if len(idx) < 100:
            continue
        for i in idx:
            if u[i] >= 0 or i < next_ok:
                continue
            after = u[i : spring[-1] + 1] > 0
            run = np.convolve(after.astype(int), np.ones(10, int), "valid")
            if not (run == 10).any():
                break
            out.append(int(i))
            tail = u[i:] > 0
            r20 = np.flatnonzero(np.convolve(tail.astype(int), np.ones(20, int), "valid") == 20)
            next_ok = i + int(r20[0]) + 20 if len(r20) else len(u)
    return out


def spell_onsets(mask: np.ndarray, months: np.ndarray, allowed, gap: int, min_len: int = 1) -> list[tuple[int, int]]:
    """(start, length) of spells where mask holds for >= min_len days, starting in the allowed months, >= gap apart."""
    out, last, i, n = [], -(10**9), 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len and months[i] in allowed and i - last >= gap:
                out.append((i, j - i))
                last = i
            i = j
        else:
            i += 1
    return out


def events(f: pd.DataFrame) -> list[dict]:
    ev = []
    mo = f.month.values
    ons = ssw_onsets(f)
    rec = {i: np.nanmean(f.hf100_tot.values[i + 5 : i + 16]) for i in ons}
    med = np.nanmedian(list(rec.values())) if rec else np.nan
    for i in ons:
        k1neg = bool((f.hf50_k1_5.values[i + 1 : i + 21] < 0).any())
        ev.append({"type": "ssw", "i": i, "recovery_hf100": rec[i], "absorbing": rec[i] > med, "reflect_k1": k1neg})
    for i, _ in spell_onsets(f.hf50_k1_5.values < 0, mo, DJFM, 20):
        ev.append({"type": "refl_zm", "i": i})
    for i, n in spell_onsets(f.ri.values > 1.0, mo, DJFM, 1, 10):
        sl = slice(i, i + n)
        if (
            np.nanmean(f.sib.values[sl]) > 0
            and np.nanmean(f.can.values[sl]) < 0
            and mo[min(i + n - 1, len(mo) - 1)] in DJFM
        ):
            ev.append({"type": "refl_reg", "i": i, "length": n})
    c10, c100 = f.cap10.values, f.cap100.values
    for i, _ in spell_onsets(c100 >= 1.0, mo, WINTER, 30):
        top = bool((c10[max(0, i - 20) : max(0, i - 2)] >= 1.0).any())
        ev.append({"type": "cap100_td" if top else "cap100_bu", "i": i})
    return ev


def window_mean(x: np.ndarray, i: int, a0: int, a1: int) -> float:
    lo, hi = i + a0, i + a1 + 1
    if lo < 0 or hi > len(x):
        return np.nan
    return float(np.nanmean(x[lo:hi]))


def outcomes(f: pd.DataFrame, i: int) -> dict:
    return {f"{o}_{w}": window_mean(f[o].values, i, *ab) for o in OUTCOMES for w, ab in WINDOWS.items()}


# ------------------------------------------------------------------------------------------------------- tests ---
def null_means(runs: list[pd.DataFrame], months: list[int], rng) -> dict:
    """For one model: the mean of every outcome over NULL draws of random days matched to the events' months."""
    pool = {}
    for r, f in enumerate(runs):
        for m in set(months):
            pool.setdefault(m, []).extend((r, int(i)) for i in np.flatnonzero(f.month.values == m)[5:-65])
    keys = [f"{o}_{w}" for o in OUTCOMES for w in WINDOWS]
    acc = {k: [] for k in keys}
    for _ in range(NULL):
        draw = [pool[m][rng.integers(len(pool[m]))] for m in months]
        vals = [outcomes(runs[r], i) for r, i in draw]
        for k in keys:
            acc[k].append(np.nanmean([v[k] for v in vals]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def fdr(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(p.shape, bool)
    ps = np.sort(p[ok])
    m = len(ps)
    below = np.flatnonzero(ps <= q * np.arange(1, m + 1) / m)
    if len(below):
        out[ok] = p[ok] <= ps[below[-1]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory of the extract npz files")
    ap.add_argument("--out", required=True, help="output stem: <out>_events.csv.gz and <out>_tests.json")
    a = ap.parse_args()
    rng = np.random.default_rng(2026)
    by_model: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(Path(a.runs).glob("*.npz")):
        if ".part" in p.name:
            continue
        f = load_run(p)
        by_model.setdefault(f.attrs["model"], []).append(f)
    rows, effects, years = [], {}, {}
    for model, runs in by_model.items():
        years[model] = int(sum(f.year.nunique() for f in runs))
        evs = []
        for r, f in enumerate(runs):
            for e in events(f):
                e.update(outcomes(f, e["i"]), model=model, run=f.attrs["run"], r=r)
                e["date"] = f"{f.year.values[e['i']]:04d}-{f.month.values[e['i']]:02d}-{f.day.values[e['i']]:02d}"
                e["month"] = int(f.month.values[e["i"]])
                evs.append(e)
        E = pd.DataFrame(evs)
        rows.append(E)
        groups = {t: E[E.type == t] for t in E.type.unique()}
        if "ssw" in groups:
            s = groups["ssw"]
            groups.update(ssw_absorbing=s[s.absorbing], ssw_reflecting=s[~s.absorbing], ssw_reflect_k1=s[s.reflect_k1])
        for g, G in groups.items():
            if len(G) < MIN_EVENTS:
                continue
            null = null_means(runs, list(G.month), rng)
            effects.setdefault(g, {})[model] = {"n": len(G), **{k: float(G[k].mean() - null[k]) for k in null}}
        print(f"{model}: {years[model]} years, " + ", ".join(f"{g} {len(G)}" for g, G in groups.items()), flush=True)
    E = pd.concat(rows, ignore_index=True)
    E.drop(columns=["i", "r"]).to_csv(f"{a.out}_events.csv.gz", index=False, float_format="%.3f")

    tests, ps = [], []
    for g, per in effects.items():
        for k in (k for k in next(iter(per.values())) if k != "n"):
            x = np.array([v[k] for v in per.values()])
            x = x[np.isfinite(x)]
            if len(x) < 5:
                continue
            t = stats.ttest_1samp(x, 0.0)
            same = max((x > 0).mean(), (x < 0).mean())
            tests.append(
                {
                    "group": g,
                    "outcome": k,
                    "models": len(x),
                    "events": int(sum(v["n"] for v in per.values())),
                    "mean": round(float(x.mean()), 3),
                    "p": float(f"{t.pvalue:.3g}"),
                    "agree": round(float(same), 2),
                }
            )
            ps.append(t.pvalue)
    keep = fdr(np.array(ps))
    for t, k in zip(tests, keep):
        t["robust"] = bool(k and t["agree"] >= 0.8)
    doc = {"model_years": years, "tests": tests, "per_model": effects}
    Path(f"{a.out}_tests.json").write_text(json.dumps(doc, indent=1))
    for t in tests:
        if t["outcome"] in ("ssi_d1_30", "ssi_d31_60", "tna_d1_10", "tna_d11_20") or t["robust"]:
            print(
                f"  {t['group']:16s} {t['outcome']:12s} {t['mean']:+.2f}  p {t['p']:.2g}  agree {t['agree']:.0%}  "
                f"{t['models']} models {t['events']} events{'  ROBUST' if t['robust'] else ''}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
