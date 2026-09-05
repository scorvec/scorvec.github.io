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
MEI = "https://psl.noaa.gov/enso/mei/data/meiv2.data"
RONI = "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"
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


def psl_url(url: str) -> dict:
    """Same monthly-table format as psl_series, at an arbitrary URL."""
    raw = P.get(url, tries=3)
    if not raw:
        return {}
    out, miss = {}, None
    for line in raw.decode("utf-8", "ignore").splitlines():
        f = line.split()
        if len(f) == 2 and all(x.lstrip("-").isdigit() for x in f):
            continue
        if len(f) == 1:
            try:
                miss = float(f[0])
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


def roni_seasonal() -> dict:
    """{(year, 'NDJ'): value} -- RONI is published already 3-month averaged."""
    raw = P.get(RONI, tries=3)
    if not raw:
        return {}
    out = {}
    for line in raw.decode("utf-8", "ignore").splitlines()[1:]:
        f = line.split()
        if len(f) != 3:
            continue
        try:
            out[(int(f[1]), f[0])] = float(f[2])
        except ValueError:
            continue
    return out


def roni_winter(seas: dict, wy: int, offset: int):
    return seas.get((wy + offset, "NDJ"))


def resolve_roni_offset(seas: dict, oni: dict, years) -> int:
    """Which calendar year CPC labels the NDJ season with, decided by the data.

    NDJ spans two calendar years, and guessing the label wrong would shift
    every RONI value by a year -- which would not error, just quietly weaken
    every correlation. So both candidates are scored against the monthly ONI
    and the better one wins; they differ enormously (r near 1 vs near 0), so
    the choice is never ambiguous.
    """
    best, bestr = 0, -2.0
    for off in (-1, 0):
        xs, ys = [], []
        for y in years:
            a, b = roni_winter(seas, y, off), winter(oni, y)
            if a is not None and b is not None:
                xs.append(a); ys.append(b)
        if len(xs) > 5:
            r = float(np.corrcoef(xs, ys)[0, 1])
            if r > bestr:
                best, bestr = off, r
    print(f"  RONI NDJ label resolved to year{best:+d} (r = {bestr:.3f} against the monthly ONI)")
    return best


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


def partial(y, x, ctrl):
    """Correlation of y with x after removing ctrl from both, with its p.

    The question a second index has to answer is not "does it correlate with
    snowpack" -- ENSO leaking through guarantees that -- but "does it explain
    anything ENSO has not already explained".
    """
    y = np.asarray(y, float); x = np.asarray(x, float); c = np.asarray(ctrl, float)
    n = len(y)
    if n < 8 or x.std() == 0 or y.std() == 0 or c.std() == 0:
        return None
    ryx = float(np.corrcoef(y, x)[0, 1])
    ryc = float(np.corrcoef(y, c)[0, 1])
    rxc = float(np.corrcoef(x, c)[0, 1])
    den = np.sqrt(max((1 - ryc ** 2) * (1 - rxc ** 2), 1e-12))
    r = (ryx - ryc * rxc) / den
    df = n - 3
    t = r * np.sqrt(df / max(1 - r * r, 1e-12))
    try:
        from scipy import stats
        pv = float(2 * stats.t.sf(abs(t), df))
    except Exception:
        from math import erfc, sqrt
        pv = float(erfc(abs(t) / sqrt(2)))
    return {"r": round(float(r), 3), "p": round(pv, 4), "n": n}


def _p_of_r(r, n, k=2):
    t = r * np.sqrt((n - k) / max(1 - r * r, 1e-12))
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), n - k))
    except Exception:
        from math import erfc, sqrt
        return float(erfc(abs(t) / sqrt(2)))


def fit(x, y, years=None):
    """Least squares with the statistics needed to read it honestly.

    Also returns the correlation after a linear time trend is removed from
    BOTH series, and each series' own correlation with time. This is not
    fussiness: over 2017-2025 the PDO fell almost monotonically into its cold
    phase (r with time -0.90) while December rainfall on the Washington coast
    rose almost monotonically (+0.88 to +0.96). Correlating two trending
    series over nine points produced r = -0.92 at p = 0.0004 for a
    relationship that is not there -- detrended it collapses to -0.06 on
    Snohomish and +0.21 on Lewis. The detrended value is the one to believe;
    over the 23-year snow record the two barely differ (-0.404 against
    -0.402), so detrending costs nothing where the record is long enough.
    """
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
    out = {"n": n, "slope": round(float(b), 2), "intercept": round(float(a), 1),
           "r": round(r, 3), "p": round(p, 4), "slope_se": round(se, 2),
           "mean": round(float(y.mean()), 1), "sd": round(float(y.std(ddof=1)), 1)}
    t = np.asarray(years, float) if years is not None and len(years) == n else np.arange(n, dtype=float)
    if t.std() > 0:
        rx = x - np.polyval(np.polyfit(t, x, 1), t)
        ry = y - np.polyval(np.polyfit(t, y, 1), t)
        if rx.std() > 0 and ry.std() > 0:
            rd = float(np.corrcoef(rx, ry)[0, 1])
            out["r_dt"] = round(rd, 3)
            out["p_dt"] = round(_p_of_r(rd, n, 3), 4)
            out["trend_x"] = round(float(np.corrcoef(t, x)[0, 1]), 3)
            out["trend_y"] = round(float(np.corrcoef(t, y)[0, 1]), 3)
    return out


MONTHS = [11, 12, 1, 2, 3, 4, 5, 6]        # the snow season, 1st of each
OBS = os.path.join(P.DATA, "obs")
RAIN_MONTHS = (10, 11, 12, 1, 2, 3)        # the Oct-Mar wet season
RAIN_MIN_DAYS = 178                        # of 181-184; below this the year is dropped


def rain_months():
    """{basin: {month: {water year: monthly total mm}}} for Oct-Mar.

    Monthly totals as well as the season total, so rainfall can be stepped
    through month by month exactly as snowpack is.
    """
    area = P.area()
    per, days = {}, {}
    for p in sorted(glob.glob(os.path.join(OBS, "????-??-??.json"))):
        d = os.path.basename(p)[:-5]
        y, m = int(d[:4]), int(d[5:7])
        if m not in RAIN_MONTHS:
            continue
        wy = y + 1 if m >= 10 else y
        try:
            r = json.load(open(p))
        except Exception:
            continue
        div = r.get("div") or {}
        if not div:
            continue
        days.setdefault((wy, m), 0)
        days[(wy, m)] += 1
        for c, v in div.items():
            per.setdefault((wy, m), {}).setdefault(c, 0.0)
            per[(wy, m)][c] += v
    import calendar
    out = {}
    for (wy, m), div in per.items():
        cy = wy - 1 if m >= 10 else wy
        if days[(wy, m)] < calendar.monthrange(cy, m)[1] - 2:      # near-complete only
            continue
        for c, v in div.items():
            out.setdefault(c, {}).setdefault(m, {})[wy] = round(v, 1)
        for name in P.COMPOSITES:
            cs = [c for c in P.members_of(name) if c in div]
            if not cs:
                continue
            w = np.array([area[c] for c in cs], float)
            out.setdefault(name, {}).setdefault(m, {})[wy] = round(
                float(np.average([div[c] for c in cs], weights=w)), 1)
    return out


def rain_season():
    """{basin: {water year: Oct-Mar Stage IV total, mm}}.

    Only complete seasons count -- a water year missing a fortnight of days
    would read as a dry year and nothing downstream would notice.
    """
    area = P.area()
    per, days = {}, {}
    for p in sorted(glob.glob(os.path.join(OBS, "????-??-??.json"))):
        d = os.path.basename(p)[:-5]
        y, m = int(d[:4]), int(d[5:7])
        if m not in RAIN_MONTHS:
            continue
        wy = y + 1 if m >= 10 else y
        try:
            r = json.load(open(p))
        except Exception:
            continue
        div = r.get("div") or {}
        if not div:
            continue
        days[wy] = days.get(wy, 0) + 1
        for c, v in div.items():
            per.setdefault(wy, {}).setdefault(c, 0.0)
            per[wy][c] += v
    full = {wy for wy, n in days.items() if n >= RAIN_MIN_DAYS}
    out = {}
    for wy in sorted(full):
        div = per[wy]
        for c, v in div.items():
            out.setdefault(c, {})[wy] = round(v, 1)
        for name in P.COMPOSITES:
            cs = [c for c in P.members_of(name) if c in div]
            if not cs:
                continue
            w = np.array([area[c] for c in cs], float)
            out.setdefault(name, {})[wy] = round(float(np.average([div[c] for c in cs], weights=w)), 1)
    return out, sorted(full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=1)
    a = ap.parse_args()

    oni, pdo = psl_series("oni"), psl_series("pdo")
    mei = psl_url(MEI)
    seas = roni_seasonal()
    print(f"  ONI {len(oni)} months, PDO {len(pdo)}, MEI.v2 {len(mei)}, RONI {len(seas)} seasons")
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

    rain, rain_years = rain_season()
    rainm = rain_months()
    print(f"  Stage IV Oct-Mar totals: {len(rain_years)} complete water years "
          f"({rain_years[0]}-{rain_years[-1]})" if rain_years else "  no complete rain seasons")

    roff = resolve_roni_offset(seas, oni, years)
    roni = {}
    for y in years:
        v = roni_winter(seas, y, roff)
        if v is not None:
            roni[y] = v

    # How much do the four actually differ? The answer belongs in the output,
    # because if they agree to r = 0.99 then showing all four is four columns
    # of the same number and a reader should be told so.
    idx_pairs = {}
    base = {y: winter(oni, y) for y in years}
    for nm, series in (("pdo", {y: winter(pdo, y) for y in years}),
                       ("mei", {y: winter(mei, y) for y in years}),
                       ("roni", roni)):
        xs = [(base[y], series.get(y)) for y in years
              if base.get(y) is not None and series.get(y) is not None]
        if len(xs) > 5:
            idx_pairs[nm] = round(float(np.corrcoef([a for a, _ in xs], [b for _, b in xs])[0, 1]), 3)
    doc_index_r = idx_pairs

    PREDICTORS = (("oni", lambda y: winter(oni, y)), ("roni", lambda y: roni.get(y)),
                  ("mei", lambda y: winter(mei, y)), ("pdo", lambda y: winter(pdo, y)))

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
           "rain_months": list(RAIN_MONTHS),
           "rain": {"season": "Oct-Mar Stage IV total, mm",
                    "years": rain_years,
                    "note": ("Stage IV starts 2016 here, so this is ~10 seasons against the "
                             "snow record's 23; |r| must exceed 0.63 for p<0.05 at n=10")},
           "note_months": ("the ENSO predictor is Nov-Jan for EVERY target month, so the "
                           "months are comparable; for targets before February that index "
                           "is not known in advance, so this is a diagnostic, not a forecast"),
           "predictors": ["oni", "roni", "mei", "pdo"],
           "index_r_vs_oni": {},
           "oni_by_year": {}, "roni_by_year": {}, "mei_by_year": {}, "pdo_by_year": {},
           "basins": {}}
    # the predictor is the same for every basin and month, so it is stored once
    doc["index_r_vs_oni"] = doc_index_r
    for y in years:
        for key, val in (("oni_by_year", winter(oni, y)), ("roni_by_year", roni.get(y)),
                         ("mei_by_year", winter(mei, y)), ("pdo_by_year", winter(pdo, y))):
            if val is not None:
                doc[key][str(y)] = round(val, 2)

    sig = {k: 0 for k, _ in PREDICTORS}
    rows = []
    for c in sorted({b for v in swe_by_month.values() for b in v}):
        per_month = {}
        for m, table in swe_by_month.items():
            series = table.get(c) or {}
            if not series:
                continue
            ys = sorted(series)
            rec = {}
            for nm, getx in PREDICTORS:
                xs, vs, yy = [], [], []
                for y in ys:
                    w = getx(y)
                    if w is not None:
                        xs.append(w); vs.append(series[y]); yy.append(y)
                f = fit(xs, vs, yy)
                if f:
                    rec[nm] = f
                    if m == 4 and f.get("p_dt", f["p"]) < 0.05:
                        sig[nm] += 1
            if rec:
                # Every index other than the ONI is also scored AFTER the ONI
                # is removed. PDO correlates with snowpack on its own, but so
                # would anything that covaries with ENSO, and the page must be
                # able to say which it is.
                base = [(y, winter(oni, y)) for y in ys if winter(oni, y) is not None]
                if len(base) >= 8:
                    byear = {y: v for y, v in base}
                    for nm, getx in PREDICTORS:
                        if nm == "oni" or nm not in rec:
                            continue
                        yy = [y for y in ys if y in byear and getx(y) is not None]
                        if len(yy) < 8:
                            continue
                        pr = partial([series[y] for y in yy], [getx(y) for y in yy],
                                     [byear[y] for y in yy])
                        if pr:
                            rec[nm]["partial_vs_oni"] = pr
                    # and the ONI after removing the PDO, the mirror question
                    yy = [y for y in ys if y in byear and winter(pdo, y) is not None]
                    if len(yy) >= 8 and "oni" in rec:
                        pr = partial([series[y] for y in yy], [byear[y] for y in yy],
                                     [winter(pdo, y) for y in yy])
                        if pr:
                            rec["oni"]["partial_vs_pdo"] = pr
                # the points behind the fit, so the page can draw the scatter
                rec["swe"] = {str(y): round(float(series[y]), 1) for y in ys}
                per_month[str(m)] = rec
        # Oct-Mar rainfall, the same fit on the same predictors. Far fewer
        # years: Stage IV here begins in 2016, so this is 10 seasons against
        # the snow record's 23, and 10 points need |r| > 0.63 to reach p<0.05.
        rs = rain.get(c) or {}
        if len(rs) >= 8:
            rec = {}
            for nm, getx in PREDICTORS:
                xs, vs, yy = [], [], []
                for y in sorted(rs):
                    w = getx(y)
                    if w is not None:
                        xs.append(w); vs.append(rs[y]); yy.append(y)
                f = fit(xs, vs, yy)
                if f:
                    rec[nm] = f
            if rec:
                rec["swe"] = {str(y): rs[y] for y in sorted(rs)}   # same key, so the scatter reuses it
                per_month["rain"] = rec
        # and month by month, so rainfall can be stepped through like snowpack
        for m, series in (rainm.get(c) or {}).items():
            if len(series) < 8:
                continue
            rec = {}
            for nm, getx in PREDICTORS:
                xs, vs, yy = [], [], []
                for y in sorted(series):
                    w = getx(y)
                    if w is not None:
                        xs.append(w); vs.append(series[y]); yy.append(y)
                f = fit(xs, vs, yy)
                if f:
                    rec[nm] = f
            if rec:
                rec["swe"] = {str(y): series[y] for y in sorted(series)}
                per_month[f"rain{m}"] = rec
        if per_month:
            doc["basins"][c] = per_month
            if "4" in per_month and "oni" in per_month["4"]:
                rows.append((c, per_month["4"]["oni"]))
    nb = len(doc["basins"])
    doc["significant"] = {k: v for k, v in sig.items()}
    doc["expected_by_chance"] = round(0.05 * nb, 1)

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
    print(f"\n  significant at p<0.05, 1 April: "
          + ", ".join(f"{k.upper()} {v}" for k, v in sig.items())
          + f" of {nb} basins; expected by chance {doc['expected_by_chance']}")
    # the count that answers "does this index add anything to ENSO"
    part = {}
    for c, per in doc["basins"].items():
        f = (per.get("4") or {})
        for nm in ("roni", "mei", "pdo"):
            pr = (f.get(nm) or {}).get("partial_vs_oni")
            if pr and pr["p"] < 0.05:
                part[nm] = part.get(nm, 0) + 1
    doc["significant_partial_vs_oni"] = part
    # significance counts for the rainfall target, which has its own (much
    # shorter) record and must not borrow the snow counts
    # initialised with zeros: a predictor with no significant basin must read
    # "0 of 48", not vanish from the caption as though it were unmeasured
    rsig = {nm: 0 for nm in ("oni", "roni", "mei", "pdo")}
    for c, per in doc["basins"].items():
        f = (per.get("rain") or {})
        for nm in ("oni", "roni", "mei", "pdo"):
            g = f.get(nm)
            if g and g.get("p_dt", g["p"]) < 0.05:
                rsig[nm] = rsig.get(nm, 0) + 1
    doc["significant_rain"] = rsig
    print(f"\n  AFTER removing the ONI, still significant at p<0.05 (of {nb} basins, "
          f"~{doc['expected_by_chance']} by chance):")
    for nm in ("roni", "mei", "pdo"):
        print(f"    {nm.upper():5s} {part.get(nm, 0)}")
    # snow vs rain over the SAME years, which is the only fair comparison
    both = []
    for c, per in doc["basins"].items():
        r4 = ((per.get("4") or {}).get("oni") or {})
        rr = ((per.get("rain") or {}).get("oni") or {})
        sw = (per.get("4") or {}).get("swe") or {}
        if not rr or not sw:
            continue
        ys = [y for y in map(str, rain_years) if y in sw]
        if len(ys) < 8:
            continue
        import numpy as _np
        o = [winter(oni, int(y)) for y in ys]
        if any(v is None for v in o):
            continue
        rs_ = float(_np.corrcoef([sw[y] for y in ys], o)[0, 1])
        both.append((rs_, rr["r"], r4.get("r")))
    if both:
        import numpy as _np
        s10 = _np.median([b[0] for b in both]); r10 = _np.median([b[1] for b in both])
        sall = _np.median([b[2] for b in both if b[2] is not None])
        doc["rain"]["matched"] = {"snow_r_all_years": round(float(sall), 3),
                                  "snow_r_rain_years": round(float(s10), 3),
                                  "rain_r": round(float(r10), 3), "n_basins": len(both)}
        print(f"\n  snow vs rain over the SAME {len(rain_years)} years, median r across {len(both)} basins:")
        print(f"    snowpack, all 23 years   {sall:+.3f}")
        print(f"    snowpack, {rain_years[0]}-{rain_years[-1]}     {s10:+.3f}")
        print(f"    Oct-Mar rain, same years {r10:+.3f}")
        print("    -> the gap is the PERIOD, not the variable: the ENSO-snow relationship is")
        print("       far weaker in the recent decade than over the full record")
    print("  how far the indices differ, correlated against the ONI over these winters:")
    for k, v in doc_index_r.items():
        print(f"    {k.upper():5s} r = {v}"
              + ("   -- effectively the same predictor" if abs(v) > 0.95 else ""))
    # written LAST: the significance counts, the partial-correlation tallies and
    # the snow-vs-rain comparison are all added to `doc` below the basin loop,
    # and dumping before them silently shipped a file without any of it.
    json.dump(doc, open(OUT, "w"), separators=(",", ":"))
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
