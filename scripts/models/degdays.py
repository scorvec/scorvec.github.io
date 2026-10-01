#!/usr/bin/env python3
"""Degree days from the short-range models (models.py -> assets/models/data/<model>/<cycle>/dd.json; 2026-10-01, user:
"HDD and CDD by forecast day out to each model's range, plus hourly temperature", PJM / NYISO / ISO-NE / ERCOT / MISO).

Conventions are the private ddtracker's, frozen into scripts/models/dd_regions.json by build_dd_regions.py:
  * points: Census 2020 county centres of population, 2 m temperature sampled BILINEARLY from the model grid in the
    worker while the field is in memory (no extra download);
  * local day per county: UTC + round(lon / 15) hours (standard time), base 65 F;
  * KINK FIRST, WEIGHT AFTER: degree days are taken per county, then weighted - gas-heating households for HDD (and
    population, which the power desks quote), population for CDD;
  * two bases, both labelled, never swapped: "mid" = the traded convention, 65 - (Tmax + Tmin) / 2 of the day's hourly
    model temperatures; "int" = the hourly integral, the day's mean of max(65 - T, 0) (degree hours / 24). The two agree
    in deep winter and differ by 3-5x for shoulder-season HDD (a warm afternoon does not undo a cool dawn);
  * which days count (ddtracker's day rule, degdays.day_masks): the run's FIRST local day needs >= 60 % of its hours and
    hours that bracket the extremes (a dawn hour 02-09 and an afternoon hour 12-19 local); the LAST local day only when
    complete; days in between are complete by construction. The integral needs >= 22 of 24 hours (an integral over a
    partial day is not a daily number) and is null otherwise. A region's day needs every one of its counties.
  * RRFS past 36 h is 3-hourly: the series is interpolated LINEARLY to hourly before the integral (the midpoint is then
    the sampled extremes); the day says `step_h: 3`. ddtracker's 4-sample diurnal profile fit is not applied.
  * normals: ddtracker's n30 (1991-2020) EXPECTED degree days on the same two bases, per region and weighting.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REGIONS_JSON = HERE / "dd_regions.json"
BASE_F = 65.0
MIN_HOURS = (2, 9)            # local hours that can carry the daily minimum / maximum (ddtracker degdays.py)
MAX_HOURS = (12, 19)
OUT = {"hdd": ("gas", "pop"), "cdd": ("pop",)}     # weighting(s) published for each quantity
_DOC = None


def regions_doc():
    global _DOC
    if _DOC is None:
        _DOC = json.loads(REGIONS_JSON.read_text())
    return _DOC


def county_points():
    """-> (lat, lon, offset) arrays for every county the regions use."""
    c = np.array(regions_doc()["counties"], float)
    return c[:, 0], c[:, 1], c[:, 2].astype(int)


def _hourly(hours, T):
    """(ncounty, nh) at forecast `hours` -> (h0..hN hourly, (ncounty, n) linear in time, step of the source sample)."""
    hours = np.asarray(hours, int)
    full = np.arange(hours[0], hours[-1] + 1)
    out = np.full((T.shape[0], full.size), np.nan)
    for c in range(T.shape[0]):
        ok = np.isfinite(T[c])
        if ok.sum() >= 2:
            out[c] = np.interp(full, hours[ok], T[c, ok], left=np.nan, right=np.nan)
    step = np.ones(full.size)
    for k in range(1, len(hours)):                  # the source spacing each hourly value was interpolated across
        step[(full > hours[k - 1]) & (full <= hours[k])] = hours[k] - hours[k - 1]
    return full, out, step


def compute(model, cycle, hours, T):
    """T: (ncounty, nhours) 2 m temperature in F at the forecast `hours` of run `cycle` (YYYYMMDDHH)."""
    doc = regions_doc()
    lat, lon, off = county_points()
    init = dt.datetime.strptime(cycle, "%Y%m%d%H")
    full, Th, step = _hourly(hours, np.asarray(T, float))
    valid = [init + dt.timedelta(hours=int(h)) for h in full]
    # per county: local date -> (mid hdd, mid cdd, int hdd, int cdd, n hours, step)
    offs = sorted(set(off.tolist()))
    days = {}                                       # date -> dict(offset -> arrays over the counties of that offset)
    for o in offs:
        cs = np.flatnonzero(off == o)
        loc = np.array([(v + dt.timedelta(hours=o)) for v in valid])
        ldate = np.array([x.date() for x in loc])
        lhour = np.array([x.hour for x in loc])
        uniq = sorted(set(ldate.tolist()))
        for k, d in enumerate(uniq):
            m = ldate == d
            n = int(m.sum())
            first, last = k == 0, k == len(uniq) - 1
            hrs = lhour[m]
            covered = ((hrs >= MIN_HOURS[0]) & (hrs <= MIN_HOURS[1])).any() and \
                      ((hrs >= MAX_HOURS[0]) & (hrs <= MAX_HOURS[1])).any()
            if n < 24 and (last or not first or n < 15 or not covered):
                continue
            x = Th[np.ix_(cs, np.flatnonzero(m))]
            ok = np.isfinite(x).all(axis=1)
            tmid = np.where(ok, (np.nanmax(x, axis=1) + np.nanmin(x, axis=1)) / 2.0, np.nan) if x.size else np.nan
            ih = np.maximum(BASE_F - x, 0).mean(axis=1) if n >= 22 else np.full(len(cs), np.nan)
            ic = np.maximum(x - BASE_F, 0).mean(axis=1) if n >= 22 else np.full(len(cs), np.nan)
            ih = np.where(ok, ih, np.nan)
            ic = np.where(ok, ic, np.nan)
            days.setdefault(d, {})[o] = (cs, tmid, ih, ic, n, float(step[m].max()))
    # region aggregation: kink first (above), weight after
    regions = {}
    norm = doc.get("normals", {})
    for r in doc["weights"]:
        W = {k: (np.array([i for i, _ in v], int), np.array([w for _, w in v])) for k, v in doc["weights"][r].items()}
        need = np.unique(np.concatenate([W[k][0] for k in W]))
        dates, rows = [], []
        for d in sorted(days):
            per = days[d]
            tm = np.full(len(lat), np.nan)
            ih = np.full(len(lat), np.nan)
            ic = np.full(len(lat), np.nan)
            nmin, smax = 24, 1.0
            for o, (cs, a, b, c, n, s) in per.items():
                tm[cs], ih[cs], ic[cs] = a, b, c
                nmin, smax = min(nmin, n), max(smax, s)
            if not np.isfinite(tm[need]).all():
                continue                             # some county of this region lacks the day (time zones, grid edge)
            hd, cd = np.maximum(BASE_F - tm, 0.0), np.maximum(tm - BASE_F, 0.0)
            row = dict(n_hours=nmin, step_h=smax)
            for q, kinds in OUT.items():
                for k in kinds:
                    idx, w = W[k]
                    mid = float(np.dot(w, (hd if q == "hdd" else cd)[idx]))
                    integ = (ih if q == "hdd" else ic)[idx]
                    row[f"{q}_{k}_mid"] = round(mid, 2)
                    row[f"{q}_{k}_int"] = round(float(np.dot(w, integ)), 2) if np.isfinite(integ).all() else None
            idx, w = W["pop"]
            row["tmean_pop"] = round(float(np.dot(w, tm[idx])), 2)
            dates.append(d)
            rows.append(row)
        # hourly regional temperature at the run's own forecast hours (weighted mean of county temperatures)
        hourly = {}
        for k in ("pop", "gas"):
            idx, w = W[k]
            x = np.asarray(T, float)[idx]
            ok = np.isfinite(x).all(axis=0)
            hourly[f"t_{k}"] = [round(float(v), 1) if g else None for v, g in zip(w @ np.nan_to_num(x), ok)]
        out = dict(dates=[d.isoformat() for d in dates], hourly=hourly)
        if rows:
            for key in rows[0]:
                out[key] = [rw[key] for rw in rows]
        nr = norm.get(r, {})
        if nr and dates:
            ix = [d.timetuple().tm_yday - 1 for d in dates]
            nm = {}
            for q, kinds in OUT.items():
                for k in kinds:
                    if k in nr:
                        nm[f"{q}_{k}_mid"] = [nr[k][q][i] for i in ix]
                        if f"{q}_hourly" in nr[k]:
                            nm[f"{q}_{k}_int"] = [nr[k][f"{q}_hourly"][i] for i in ix]
            if "pop" in nr:
                nm["tmean_pop"] = [nr["pop"]["tmean"][i] for i in ix]
            out["normal"] = nm
        regions[r] = out
    return dict(v=1, model=model, cycle=cycle, init=init.strftime("%Y-%m-%dT%H:00Z"), base_f=BASE_F,
                hours=list(map(int, hours)),
                regions=regions, states=doc["regions"], weights={k: doc["kinds"][k] for k in ("pop", "gas")},
                normals_period=doc.get("normals_period"),
                conventions=dict(
                    mid="65 F minus (Tmax + Tmin) / 2 of the local day's hourly model 2 m temperatures - the traded convention",
                    int="the local day's mean of max(65 - T, 0) over its hours (degree hours / 24); null unless >= 22 hours",
                    weighting="degree days per county (Census 2020 centre of population, bilinear 2 m T), then weighted: "
                              "hdd_gas by utility-gas heating households, hdd_pop and cdd_pop by population",
                    day="local standard day per county, UTC + round(lon / 15) h; first day kept with >= 60 % of hours "
                        "bracketing dawn and afternoon, last day only when complete; a region needs all its counties",
                    hourly="t_pop / t_gas: weighted mean county 2 m temperature (F) at each forecast hour in `hours`",
                    normal="ddtracker n30 expected degree days (PRISM 1991-2020), same bases and weights"))
