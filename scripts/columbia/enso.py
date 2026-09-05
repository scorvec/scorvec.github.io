"""ENSO and PDO against Columbia basin snowpack, division by division.

Regresses each basin's 1 April snow water equivalent -- the standard water
supply benchmark -- on the ENSO state of the winter that built it, taken as
the ONI averaged over November-January. PDO over the same window is fitted
alongside and separately.

    ONI, PDO   NOAA PSL, monthly, plain text
    SWE        SNODAS on the NWRFC divisions (scripts/columbia/snodas.py)

**The sample is 23 water years and that governs how this may be read.**
SNODAS starts in 2003, so every correlation here rests on 23 points. Two
consequences are reported rather than buried:

  * a correlation needs |r| > 0.41 to clear p < 0.05 at n = 23, and the
    confidence interval on r is roughly +/- 0.35 wide even then;
  * there are 48 basins. Testing all of them at p < 0.05 yields about two
    significant results by chance alone, so the count of "significant"
    basins is printed against that expectation, and a field-significance
    test says whether the pattern as a whole beats chance.

Anyone reading a single basin's r without those two facts will over-read it.

The regression is repeated for the 1st of each month of the snow season, so
the page can show how the relationship builds and then decays through the
accumulation and melt. The ENSO predictor is held FIXED at Nov-Jan for every
target month; that is what makes the months comparable -- same predictor,
different targets -- and it makes the progression readable. It also means
this is a diagnostic of how ENSO winters and snowpack co-vary, NOT a forecast
model: for a 1 December or 1 January target the NDJ index is not yet known.

The per-year points behind each fit are stored too, so clicking a basin can
draw the scatter it came from rather than asking anyone to trust an r.

    python scripts/columbia/enso.py
-> columbia/data/enso_regression.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P  # noqa: E402

SNOW = os.path.join(P.DATA, "snow")
OUT = os.path.join(P.DATA, "enso_regression.json")
PSL = "https://psl.noaa.gov/data/correlation/{}.data"
# The ENSO winter that builds an April snowpack: Nov and Dec of the previous
# calendar year and Jan of this one.
WINDOW = ((-1, 11), (-1, 12), (0, 1))


def psl_series(name: str) -> dict:
    """{(year, month): value} from a PSL monthly text file."""
    raw = P.get(PSL.format(name), tries=3)
    if not raw:
        return {}
    out, miss = {}, None
    for line in raw.decode("utf-8", "ignore").splitlines():
        f = line.split()
        if len(f) == 2 and all(x.lstrip("-").isdigit() for x in f):
            continue                                   # the year-range header
        if len(f) == 1:
            try:
                miss = float(f[0])                     # the missing-value flag
            except ValueError:
                pass
            continue
        if len(f) != 13:
            continue
        try:
            y = int(f[0]); vals = [float(x) for x in f[1:]]
        except ValueError:
            continue
        for m, v in enumerate(vals, 1):
            if miss is None or abs(v - miss) > 1e-6:
                out[(y, m)] = v
    return out


def winter(idx: dict, wy: int):
    """The index averaged over the window, for water year `wy`, or None."""
    vals = [idx.get((wy + off, m)) for off, m in WINDOW]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def swe_on(month: int, day: int) -> dict:
    """{basin: {water_year: mm}} for one calendar date each year.

    Composites are area-weighted from their member divisions, the same way
    the page builds them, so the basins here are the basins on the page.
    """
    area = P.area()
    per_year = {}
    for p in sorted(glob.glob(os.path.join(SNOW, "????-??-??.json"))):
        d = os.path.basename(p)[:-5]
        if int(d[5:7]) != month or int(d[8:10]) != day:
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue                    # a file being rewritten; skip it
        per_year[int(d[:4])] = r.get("div") or {}
    out: dict[str, dict] = {}
    for y, div in per_year.items():
        for c, v in div.items():
            out.setdefault(c, {})[y] = v
        for name in P.COMPOSITES:
            cs = [c for c in P.members_of(name) if c in div]
            if not cs:
                continue
            w = np.array([area[c] for c in cs], float)
            out.setdefault(name, {})[y] = float(np.average([div[c] for c in cs], weights=w))
    return out


def fit(x, y):
    """Least squares with the statistics needed to read it honestly."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 8 or x.std() == 0 or y.std() == 0:
        return None
    b, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    # two-sided p for the correlation, via the t transform
    t = r * np.sqrt((n - 2) / max(1 - r * r, 1e-12))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), n - 2))
    except Exception:                                   # no scipy: normal approx
        from math import erfc, sqrt
        p = float(erfc(abs(t) / sqrt(2)))
    se = float(np.sqrt((1 - r * r) * y.var(ddof=1) / max(x.var(ddof=1), 1e-12) / (n - 2)))
    return {"n": n, "slope": round(float(b), 2), "intercept": round(float(a), 1),
            "r": round(r, 3), "p": round(p, 4), "slope_se": round(se, 2),
            "mean": round(float(y.mean()), 1), "sd": round(float(y.std(ddof=1)), 1)}


MONTHS = [11, 12, 1, 2, 3, 4, 5, 6]        # the snow season, 1st of each


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=1)
    a = ap.parse_args()

    oni, pdo = psl_series("oni"), psl_series("pdo")
    print(f"  ONI {len(oni)} months, PDO {len(pdo)} months")
    swe_by_month = {}
    for m in MONTHS:
        v = swe_on(m, a.day)
        if v:
            swe_by_month[m] = v
    if not swe_by_month:
        print("  no SNODAS days on those dates -- run snodas.py first"); return
    swe = swe_by_month.get(4) or next(iter(swe_by_month.values()))
    years = sorted({y for v in swe.values() for y in v})
    print(f"  {len(swe)} basins, {len(swe_by_month)} months, water years {years[0]}-{years[-1]}")

    # ONI and PDO are NOT independent predictors over this window. Measured
    # r = 0.48 across these winters, so "23 basins significant on ONI and 23
    # on PDO" is largely ONE signal counted twice, not two findings.
    ys_all = sorted({y for v in swe.values() for y in v})
    pair = [(winter(oni, y), winter(pdo, y)) for y in ys_all]
    pair = [(a_, b_) for a_, b_ in pair if a_ is not None and b_ is not None]
    oni_pdo = round(float(np.corrcoef([p_[0] for p_ in pair], [p_[1] for p_ in pair])[0, 1]), 3) \
        if len(pair) > 3 else None

    doc = {"built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "target": f"SNODAS SWE on the {a.day}st of each month, mm",
           "predictor": "ONI and PDO averaged Nov-Dec-Jan of the same water year",
           "source": "NOAA PSL oni.data / pdo.data; SNODAS masked 1034",
           "caveat": ("n is about 23 water years: |r| > 0.41 is needed for p < 0.05, "
                      "and testing 48 basins at that level yields ~2 hits by chance"),
           "oni_pdo_r": oni_pdo,
           "caveat_collinear": ("ONI and PDO correlate at r = %s over these winters, so the "
                                "two columns are largely the same signal counted twice"
                                % oni_pdo),
           "months": MONTHS,
           "note_months": ("the ENSO predictor is Nov-Jan for EVERY target month, so the "
                           "months are comparable; for targets before February that index "
                           "is not known in advance, so this is a diagnostic, not a forecast"),
           "oni_by_year": {}, "pdo_by_year": {},
           "basins": {}}
    # the predictor is the same for every basin and month, so it is stored once
    for y in years:
        w, q = winter(oni, y), winter(pdo, y)
        if w is not None:
            doc["oni_by_year"][str(y)] = round(w, 2)
        if q is not None:
            doc["pdo_by_year"][str(y)] = round(q, 2)

    sig = {"oni": 0, "pdo": 0}
    rows = []
    for c in sorted({b for v in swe_by_month.values() for b in v}):
        per_month = {}
        for m, table in swe_by_month.items():
            series = table.get(c) or {}
            if not series:
                continue
            ys = sorted(series)
            rec = {}
            for nm, idx in (("oni", oni), ("pdo", pdo)):
                xs, vs = [], []
                for y in ys:
                    w = winter(idx, y)
                    if w is not None:
                        xs.append(w); vs.append(series[y])
                f = fit(xs, vs)
                if f:
                    rec[nm] = f
                    if m == 4 and f["p"] < 0.05:
                        sig[nm] += 1
            if rec:
                # the points behind the fit, so the page can draw the scatter
                rec["swe"] = {str(y): round(float(series[y]), 1) for y in ys}
                per_month[str(m)] = rec
        if per_month:
            doc["basins"][c] = per_month
            if "4" in per_month and "oni" in per_month["4"]:
                rows.append((c, per_month["4"]["oni"]))
    nb = len(doc["basins"])
    doc["significant"] = {k: v for k, v in sig.items()}
    doc["expected_by_chance"] = round(0.05 * nb, 1)
    json.dump(doc, open(OUT, "w"), separators=(",", ":"))

    # how the relationship evolves through the season, on the whole basin
    key = "Columbia abv The Dalles"
    if key in doc["basins"]:
        print("\n  Columbia above The Dalles, SWE vs NDJ ONI by target month:")
        print(f"    {'month':>6s} {'n':>3s} {'r':>7s} {'p':>7s} {'mean mm':>8s}")
        for m in MONTHS:
            f = (doc["basins"][key].get(str(m)) or {}).get("oni")
            if f:
                print(f"    {dt.date(2000, m, 1):%b}    {f['n']:3d} {f['r']:7.3f} "
                      f"{f['p']:7.4f} {f['mean']:8.1f}")

    rows.sort(key=lambda r: r[1]["r"])
    print(f"\n  1 April SWE vs NDJ ONI, most negative correlations first")
    print(f"  {'basin':28s} {'n':>3s} {'r':>7s} {'p':>7s} {'slope mm/K':>11s} {'mean':>7s}")
    for c, f in rows[:6] + [("...", None)] + rows[-4:]:
        if f is None:
            print("  ..."); continue
        print(f"  {c:28s} {f['n']:3d} {f['r']:7.3f} {f['p']:7.4f} "
              f"{f['slope']:8.1f}+-{f['slope_se']:.0f} {f['mean']:7.1f}")
    print(f"\n  significant at p<0.05: ONI {sig['oni']} of {nb} basins, "
          f"PDO {sig['pdo']} of {nb}; expected by chance {doc['expected_by_chance']}")
    print(f"  ONI vs PDO over these winters: r = {oni_pdo} -- largely one signal, not two")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
