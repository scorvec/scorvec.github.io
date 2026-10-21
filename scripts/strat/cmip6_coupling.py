#!/usr/bin/env python3
"""Wave reflection and downward propagation on a large CMIP6 sample: events in every run's daily series
(cmip6_coupling_extract.py), what followed at the surface, and whether it holds across models.

ANOMALIES. Each run against itself, on its own calendar (365-day, 360-day or Gregorian years): four annual harmonics and
a linear trend for the mean; for the standardised series a three-harmonic fit to the squared residual for the variance.
  ssi          the SSW imprint index (ssw_imprint.py): the projected sea-level pressure less its fit; the pattern made
               without the model for the nine models that made it
  wk10, wk100  minus the standardised u(60N) at 10 and 100 hPa: positive = a weak vortex (the CMIP6 drips' measure;
               zonal-mean height is too rarely published daily)
  zsib, zcan   standardised regional v*T* at 100 hPa; ri = zsib - zcan (Messori et al. 2022), where gridded v and T exist
  tna, teu     2 m temperature anomaly, North America and northern Eurasia (K)

EVENTS (onsets; November-March unless said); the flux events only in runs with the EP flux, the regional ones only in
runs with gridded v and T, the shear ones only where u reaches 2 hPa:
  ssw          u(60N, 10 hPa) easterly (Charlton & Polvani 2007: a return to 10 westerly days before 30 April, 20
               westerly days between events), split two ways:
                 ssw_absorbing / ssw_reflecting   the 100 hPa EP flux Fz over days +5..+15 above / below the model's
                 median (Kodera et al. 2016's recovery-phase wave driving)
                 ssw_reflect_fz   Fz at 50 hPa (5-day mean) negative on some day +1..+20: waves coming back down
  refl_zm      Fz at 50 hPa, 5-day mean, below zero (net downward wave activity; the zonal-mean reflection of Shaw &
               Perlwitz 2013 and Dunn-Sigouin & Shaw 2015, all waves here), December-March, 20 days apart
  refl_low     the same below the run's December-March 10th percentile
  refl_reg     ri > 1 for 10 days or more, December-March, with upward flux over Siberia and downward over Canada on
               average over the event (Messori et al. 2022)
  shear_neg    u(60N) at 2 hPa minus 10 hPa, 5-day mean, below its January-March 10th percentile: a reflective upper
               stratosphere (Perlwitz & Harnik 2003)
  wk100_td / wk100_bu   the 100 hPa weakness first reaching +1 sd, 30 days apart, with 10 hPa at +1 sd in the 3-20 days
               before (top-down) or not (bottom-up)

TESTS. Per model, each outcome's event mean less its expected value on a random day of the events' calendar months
in the same runs (every day of those months, weighted as the events are): the model's effect. Across models (models weighted equally, those with >= MIN_EVENTS events): the
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
WINDOWS = {"d-5_0": (-5, 0), "d1_10": (1, 10), "d11_20": (11, 20), "d1_30": (1, 30), "d31_60": (31, 60)}
OUTCOMES = ("ssi", "tna", "teu", "wk10", "wk100")


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
    X4n = design(frac, y, 4, False)
    f = pd.DataFrame({"year": y, "month": m, "day": d})
    f["u10"] = z["u10"]
    f["ssi"] = anomaly(z[pattern].astype(float), X4)
    f["wk10"] = -standardise(z["u10"].astype(float), X4, X3)
    f["wk100"] = -standardise(z["u100"].astype(float), X4, X3)
    f["tna"] = anomaly(z["tas_na"].astype(float), X4)
    f["teu"] = anomaly(z["tas_eu"].astype(float), X4)
    has = {k: bool(np.isfinite(z[k]).mean() > 0.9) for k in ("fz100", "fz50", "hf100_sib", "u2")}
    if has["fz100"] and has["fz50"]:
        f["fz100"] = z["fz100"]
        f["fz50_5"] = pd.Series(z["fz50"]).rolling(5, center=True).mean()
    if has["hf100_sib"]:
        f["zsib"] = standardise(z["hf100_sib"].astype(float), X4n, X3)
        f["zcan"] = standardise(z["hf100_can"].astype(float), X4n, X3)
        f["ri"] = f.zsib - f.zcan
        f["sib"], f["can"] = z["hf100_sib"], z["hf100_can"]
    if has["u2"]:
        f["shear5"] = pd.Series(z["u2"] - z["u10"]).rolling(5, center=True).mean()
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
    flux = "fz100" in f
    for i in ssw_onsets(f):
        e = {"type": "ssw", "i": i}
        if flux:
            e["recovery_fz100"] = float(np.nanmean(f.fz100.values[i + 5 : i + 16]))
            e["reflect_fz"] = bool((f.fz50_5.values[i + 1 : i + 21] < 0).any())
        ev.append(e)
    if flux:
        for i, _ in spell_onsets(f.fz50_5.values < 0, mo, DJFM, 20):
            ev.append({"type": "refl_zm", "i": i})
        q = np.nanpercentile(f.fz50_5.values[np.isin(mo, DJFM)], 10)
        for i, _ in spell_onsets(f.fz50_5.values < q, mo, DJFM, 20):
            ev.append({"type": "refl_low", "i": i})
    if "ri" in f:
        for i, n in spell_onsets(f.ri.values > 1.0, mo, DJFM, 1, 10):
            sl = slice(i, i + n)
            end_ok = mo[min(i + n - 1, len(mo) - 1)] in DJFM
            if end_ok and np.nanmean(f.sib.values[sl]) > 0 and np.nanmean(f.can.values[sl]) < 0:
                ev.append({"type": "refl_reg", "i": i, "length": n})
    if "shear5" in f:
        q = np.nanpercentile(f.shear5.values[np.isin(mo, (1, 2, 3))], 10)
        for i, _ in spell_onsets(f.shear5.values < q, mo, (1, 2, 3), 20):
            ev.append({"type": "shear_neg", "i": i})
    w10, w100 = f.wk10.values, f.wk100.values
    for i, _ in spell_onsets(w100 >= 1.0, mo, WINTER, 30):
        top = bool((w10[max(0, i - 20) : max(0, i - 2)] >= 1.0).any())
        ev.append({"type": "wk100_td" if top else "wk100_bu", "i": i})
    return ev


def window_table(f: pd.DataFrame) -> dict:
    """{outcome_window: (day,) float32}: each outcome's mean over each window after every day (NaN where the window
    leaves the run), from cumulative sums, so an event or a random day is a lookup."""
    n = len(f)
    i = np.arange(n)
    out = {}
    for o in OUTCOMES:
        x = f[o].values.astype(float)
        ok = np.isfinite(x)
        cs = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
        cn = np.concatenate([[0], np.cumsum(ok)])
        for w, (a0, a1) in WINDOWS.items():
            lo, hi = np.clip(i + a0, 0, n), np.clip(i + a1 + 1, 0, n)
            cnt = cn[hi] - cn[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                v = (cs[hi] - cs[lo]) / cnt
            inside = (i + a0 >= 0) & (i + a1 + 1 <= n) & (cnt > 0)
            out[f"{o}_{w}"] = np.where(inside, v, np.nan).astype("float32")
    return out


def outcomes(table: dict, i: int) -> dict:
    return {k: float(v[i]) for k, v in table.items()}


# ------------------------------------------------------------------------------------------------------- tests ---
def null_means(runs: list[pd.DataFrame], tables: list[dict], months: list[int]) -> dict:
    """For one model: each outcome's expected value on a random day of the events' calendar months (in their
    proportions), over every day of those months in its runs: the exact mean of month-matched random draws."""
    share = pd.Series(months).value_counts(normalize=True)
    out = {}
    for k in tables[0]:
        tot = 0.0
        for m, wgt in share.items():
            vals = np.concatenate([t[k][f.month.values == m] for f, t in zip(runs, tables)])
            tot += wgt * float(np.nanmean(vals))
        out[k] = tot
    return out


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
    paths: dict[str, list[Path]] = {}
    for p in sorted(Path(a.runs).glob("*.npz")):
        if ".part" not in p.name:
            paths.setdefault(p.stem.split("_")[0], []).append(p)
    rows, effects, years = [], {}, {}
    for model, ps in paths.items():  # one model in memory at a time
        runs = [load_run(p) for p in ps]
        tables = [window_table(f) for f in runs]
        years[model] = int(sum(f.year.nunique() for f in runs))
        evs = []
        for r, (f, t) in enumerate(zip(runs, tables)):
            for e in events(f):
                e.update(outcomes(t, e["i"]), model=model, run=f.attrs["run"], r=r)
                e["date"] = f"{f.year.values[e['i']]:04d}-{f.month.values[e['i']]:02d}-{f.day.values[e['i']]:02d}"
                e["month"] = int(f.month.values[e["i"]])
                evs.append(e)
        E = pd.DataFrame(evs)
        rows.append(E)
        groups = {t: E[E.type == t] for t in E.type.unique()}
        if "ssw" in groups and "recovery_fz100" in E:
            sw = groups["ssw"].dropna(subset=["recovery_fz100"])
            med = sw.recovery_fz100.median()
            groups.update(
                ssw_absorbing=sw[sw.recovery_fz100 > med],
                ssw_reflecting=sw[sw.recovery_fz100 <= med],
                ssw_reflect_fz=sw[sw.reflect_fz.astype(bool)],
            )
        for g, G in groups.items():
            if len(G) < MIN_EVENTS:
                continue
            null = null_means(runs, tables, list(G.month))
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
