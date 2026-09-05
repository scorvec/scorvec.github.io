"""The ENSO regressions again, on Daymet's 46 water years instead of 23.

Same machinery as enso.py -- the index parsers, the detrended fit, the
partial correlations are imported from it rather than reimplemented -- but
the targets come from Daymet, which changes what the answers are worth:

    SNODAS   2003-2026, 23 water years, snow only, observation-based
    Daymet   1980-2025, 46 water years, snow AND rain AND temperature,
             one grid, modelled snow

Doubling the record is the point. The December-rainfall correlations that
produced a spurious r = -0.92 rested on nine winters; here they rest on
forty-six, where a decade-long trend in one index can no longer masquerade
as a relationship.

Three targets, each by month:

    snow   monthly mean SWE            -- MODELLED, not the SNODAS analysis
    rain   monthly precipitation, and the Oct-Mar total
    temp   monthly mean of (tmax+tmin)/2

The SNODAS and Daymet snow records are deliberately kept in separate files.
They are different quantities -- an analysis and a model -- and concatenating
them would put a step change into the middle of any series that spanned 2003.

    python scripts/columbia/enso_daymet.py
-> columbia/data/enso_daymet.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P  # noqa: E402
import enso as E  # noqa: E402

SRC = os.path.join(P.DATA, "daymet_monthly.json")
OUT = os.path.join(P.DATA, "enso_daymet.json")
SNOW_MONTHS = (11, 12, 1, 2, 3, 4, 5, 6)
RAIN_MONTHS = (10, 11, 12, 1, 2, 3)
TEMP_MONTHS = (10, 11, 12, 1, 2, 3, 4)


def water_year(y: int, m: int) -> int:
    return y + 1 if m >= 10 else y


def series_by_basin(doc):
    """{basin: {(field, month): {water year: value}}}, composites included."""
    area = P.area()
    raw = {}
    for c, rec in doc["div"].items():
        idx = rec["index"]
        for i, (y, m) in enumerate(idx):
            wy = water_year(y, m)
            for field, key in (("snow", "swe"), ("rain", "prcp")):
                raw.setdefault(c, {}).setdefault((field, m), {})[wy] = rec[key][i]
            raw.setdefault(c, {}).setdefault(("temp", m), {})[wy] = \
                round((rec["tmax"][i] + rec["tmin"][i]) / 2.0, 2)
    # composites, area weighted like everything else on this page
    for name in P.COMPOSITES:
        cs = [c for c in P.members_of(name) if c in raw]
        if not cs:
            continue
        w = np.array([area[c] for c in cs], float); w /= w.sum()
        keys = set().union(*[set(raw[c]) for c in cs])
        for k in keys:
            years = set().union(*[set(raw[c].get(k, {})) for c in cs])
            for y in years:
                vals = [raw[c].get(k, {}).get(y) for c in cs]
                if any(v is None for v in vals):
                    continue
                raw.setdefault(name, {}).setdefault(k, {})[y] = float(np.dot(w, vals))
    # the Oct-Mar rainfall total
    for c, per in list(raw.items()):
        tot = {}
        for m in RAIN_MONTHS:
            for y, v in (per.get(("rain", m)) or {}).items():
                tot[y] = tot.get(y, 0.0) + v
        # only water years with all six months present
        full = {y: v for y, v in tot.items()
                if all(y in (per.get(("rain", m)) or {}) for m in RAIN_MONTHS)}
        if full:
            per[("rain", "season")] = full
    return raw


def main():
    if not os.path.exists(SRC):
        print(f"  no {SRC} -- run daymet.py first"); return
    doc_in = json.load(open(SRC))
    raw = series_by_basin(doc_in)
    oni, pdo = E.psl_series("oni"), E.psl_series("pdo")
    mei = E.psl_url(E.MEI)
    seas = E.roni_seasonal()
    years = sorted({y for per in raw.values() for s in per.values() for y in s})
    roff = E.resolve_roni_offset(seas, oni, years)
    roni = {y: v for y in years if (v := E.roni_winter(seas, y, roff)) is not None}
    PRED = (("oni", lambda y: E.winter(oni, y)), ("roni", lambda y: roni.get(y)),
            ("mei", lambda y: E.winter(mei, y)), ("pdo", lambda y: E.winter(pdo, y)))
    print(f"  {len(raw)} basins, water years {years[0]}-{years[-1]}")

    doc = {"built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": "Daymet V4 R1 on the NWRFC divisions (scripts/columbia/daymet.py)",
           "predictor": "ONI/RONI/MEI/PDO averaged Nov-Dec-Jan of the same water year",
           "period": doc_in.get("period"),
           "fields": {"snow": "monthly mean SWE, kg/m2 — MODELLED, not the SNODAS analysis",
                      "rain": "monthly precipitation, mm; `season` is the Oct-Mar total",
                      "temp": "monthly mean of (tmax+tmin)/2, degC"},
           "months": {"snow": list(SNOW_MONTHS), "rain": list(RAIN_MONTHS) + ["season"],
                      "temp": list(TEMP_MONTHS)},
           "predictors": [k for k, _ in PRED],
           "note": ("46 water years against SNODAS's 23; correlations are reported "
                    "detrended, and Daymet snow is a model field so it is kept in a "
                    "separate file from the SNODAS record rather than concatenated"),
           "oni_by_year": {}, "roni_by_year": {}, "mei_by_year": {}, "pdo_by_year": {},
           "basins": {}}
    for y in years:
        for key, val in (("oni_by_year", E.winter(oni, y)), ("roni_by_year", roni.get(y)),
                         ("mei_by_year", E.winter(mei, y)), ("pdo_by_year", E.winter(pdo, y))):
            if val is not None:
                doc[key][str(y)] = round(val, 2)

    sig = {}
    for c, per in raw.items():
        out = {}
        for (field, m), s in per.items():
            if len(s) < 12:
                continue
            key = f"{field}{m}"
            rec = {}
            for nm, getx in PRED:
                xs, vs, yy = [], [], []
                for y in sorted(s):
                    w = getx(y)
                    if w is not None:
                        xs.append(w); vs.append(s[y]); yy.append(y)
                f = E.fit(xs, vs, yy)
                if f:
                    rec[nm] = f
                    if field == "snow" and m == 4 and f.get("p_dt", f["p"]) < 0.05:
                        sig[nm] = sig.get(nm, 0) + 1
            if rec:
                # Over 46 years the time trend is a RESULT, not just a nuisance
                # to be removed. At n=9 a trend was pure contamination -- two
                # trending series faking r = -0.92. At n=46 a declining
                # snowpack or a warming winter is a real signal in its own
                # right, and burying it inside the detrending would throw away
                # the more interesting half of the record. So the correlation
                # against ENSO is reported detrended, AND the trend is reported
                # separately with its own slope and significance.
                yy2 = sorted(s)
                tv = np.array([float(y) for y in yy2])
                vv = np.array([float(s[y]) for y in yy2])
                if len(tv) >= 12 and tv.std() > 0 and vv.std() > 0:
                    b, _ = np.polyfit(tv, vv, 1)
                    rt = float(np.corrcoef(tv, vv)[0, 1])
                    rec["trend"] = {"per_decade": round(float(b) * 10, 3),
                                    "r": round(rt, 3),
                                    "p": round(E._p_of_r(rt, len(tv)), 4),
                                    "n": len(tv),
                                    # A percentage change is meaningless on a
                                    # Celsius mean -- the zero is arbitrary, so
                                    # "-15% per decade" for a winter warming of
                                    # +0.57 C is nonsense. Only for the
                                    # non-negative quantities.
                                    "pct_per_decade": (round(100 * float(b) * 10 / vv.mean(), 2)
                                                       if field in ("snow", "rain")
                                                       and vv.mean() > 1e-6 else None)}
                rec["swe"] = {str(y): round(float(s[y]), 1) for y in sorted(s)}
                out[key] = rec
        if out:
            doc["basins"][c] = out
    doc["significant"] = sig
    doc["expected_by_chance"] = round(0.05 * len(doc["basins"]), 1)
    json.dump(doc, open(OUT, "w"), separators=(",", ":"))

    key = "Columbia abv The Dalles"
    if key in doc["basins"]:
        print(f"\n  {key}, detrended r against the NDJ ONI:")
        print(f"    {'target':10s} {'n':>3s} {'r':>7s} {'p':>7s}")
        for lab, k in (("Apr snow", "snow4"), ("Mar snow", "snow3"), ("Jan snow", "snow1"),
                       ("Oct-Mar rain", "rainseason"), ("Dec rain", "rain12"),
                       ("Nov rain", "rain11"), ("DJF temp", "temp1")):
            f = (doc["basins"][key].get(k) or {}).get("oni")
            if f:
                print(f"    {lab:10s} {f['n']:3d} {f.get('r_dt', f['r']):+7.3f} "
                      f"{f.get('p_dt', f['p']):7.4f}")
    # the trends, which 46 years can actually speak to
    tr = {}
    for c, per in doc["basins"].items():
        for k, rec in per.items():
            t = rec.get("trend")
            if t and t["p"] < 0.05:
                tr[k] = tr.get(k, 0) + 1
    doc["significant_trend"] = tr
    if key in doc["basins"]:
        print(f"\n  {key}, trend over {doc['period'][0]}-{doc['period'][1]}:")
        print(f"    {'target':10s} {'per decade':>11s} {'%/decade':>9s} {'p':>7s}")
        for lab, k in (("Apr snow", "snow4"), ("Mar snow", "snow3"),
                       ("Oct-Mar rain", "rainseason"), ("DJF temp", "temp1")):
            t = (doc["basins"][key].get(k) or {}).get("trend")
            if t:
                pc = t.get("pct_per_decade")
                print(f"    {lab:10s} {t['per_decade']:+11.2f} "
                      + (f"{pc:+9.1f} " if pc is not None else f"{'—':>9s} ")
                      + f"{t['p']:7.4f}")
    print(f"\n  1 Apr snow significant at p<0.05: {sig} of {len(doc['basins'])} basins, "
          f"~{doc['expected_by_chance']} by chance")
    print(f"  wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
