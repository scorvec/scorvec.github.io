#!/usr/bin/env python3
"""Weather-regime probabilities from the 21 GEPS members, days 1-35, Euro-Atlantic and North American sectors.

Revamped 2026-09-28 (user review): the classification, the figure and the hindcast skill now come from the site
repo's shared regime code (~/scorvec.github.io/scripts/regimes: regime_core, regime_figure), the same code the GEFS
page and the multi-model view on subseasonal.html use.

  members     each member's de-drifted, re-based 500 hPa anomaly (live/geps_members_zg500_<cycle>.nc, the maps' field,
              2.5 deg), 5-day running mean along the lead, assigned to the nearest centroid of the set of EACH VALID
              DAY'S season (Oct-Mar cold, Apr-Sep warm - no longer the init month's set held throughout), or "no
              regime" below pattern correlation 0.25
  observed    GEPS day-0 analyses (telecon/analysis) on the same projection, each day in its own season's set
  previous    the previous extended cycle, same classification, drawn as a thin outline
  skill       the GEPS8 hindcast (scripts/regimes/data/regime_hindcast_geps.npz, season-aware): hit rate of the
              ensemble-mean regime by lead with a 95% year-block bootstrap interval, against persistence and
              climatology, starts within +-45 d of this date
  outputs     figs/ + assets/geps: geps_regimes_{ea,na}_fc.webp; assets/geps: regimes.json (summary),
              regimes_members.json (per-member labels - the multi-model view's input, regime_core.SCHEMA)

    python regimes_geps.py --cycle 20260928
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths                                                                # noqa: E402
sys.path.insert(0, str(paths.REPO / "scripts" / "regimes"))
import regime_core as RC                                                    # noqa: E402
import regime_figure as RF                                                  # noqa: E402
import telecon_geps as tg                                                   # noqa: E402
from telecon_io import TC, period_shift                                     # noqa: E402

FIGS = paths.FIGS
SITE = paths.SITE_ASSETS


def member_labels(ref, cycle):
    """-> (valid dates of days 1..35, {sk: labels (35, 21)}) or None."""
    a = tg.members_anom("zg500", cycle, pd.Timestamp(cycle))                 # (L, number, lat, lon)
    if a is None:
        return None
    assert np.allclose(a.latitude.values, RC.LAT25) and np.allclose(a.longitude.values, RC.LON25)
    base = RC.parse_date(cycle)
    L = a.L.values.astype(int)
    days = [base + pd.Timedelta(days=int(x) - 1) for x in L]                 # day L = the UTC day base + L - 1
    days = [d.date() if hasattr(d, "date") else d for d in days]
    out = {}
    for sk, spec in RC.SECTORS.items():
        sec, _, _ = RC.sector_cut(a.values, spec)
        v = RC.running_partial(sec, RC.SMOOTH, axis=0)
        n, m = v.shape[:2]
        lab, _ = ref.classify_dated(v.reshape(n * m, *v.shape[2:]), np.repeat(days, m), sk)
        out[sk] = lab.reshape(n, m)
    return days, out


def analysis_labels(ref, base, days=45):
    """Observed tail: GEPS day-0 analyses anomalised as telecon_geps does, 5-day mean, each day in its own set."""
    from telecon_geps import model_clim25, analysis_field
    cw = model_clim25("zg500"); shift = period_shift("zg500")
    ts, vs = [], []
    for k in range(days, -1, -1):
        d = base - pd.Timedelta(days=k)
        v = analysis_field("zg500", d)
        if v is None:
            continue
        an = v - cw.interp(doy=d.dayofyear).values
        if shift is not None:
            an = an + shift[d.dayofyear - 1]
        ts.append(d.date()); vs.append(an)
    if not ts:
        return None
    out = {}
    for sk, spec in RC.SECTORS.items():
        sec, _, _ = RC.sector_cut(np.stack(vs), spec)
        lab, _ = ref.classify_dated(RC.running_partial(sec, RC.SMOOTH, axis=0), ts, sk)
        out[sk] = (ts, lab)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--prev")
    a = ap.parse_args()
    ref = RC.Ref()
    base = RC.parse_date(a.cycle)
    got = member_labels(ref, a.cycle)
    if got is None:
        print("  no zg500 members for this cycle"); return 1
    days, labs = got
    prev_cycle = a.prev or tg.previous_cycle(a.cycle)
    prev = member_labels(ref, prev_cycle) if prev_cycle else None
    obs = analysis_labels(ref, pd.Timestamp(a.cycle))
    skill = RC.hindcast_skill("geps", base)
    js = {"cycle": a.cycle, "prev_cycle": prev_cycle, "r_min": RC.R_MIN, "classification": "each valid day in its own "
          "season's set (Oct-Mar cold, Apr-Sep warm)", "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%MZ"),
          "sectors": {}}
    for sk, spec in RC.SECTORS.items():
        P = RC.family_probs(ref, sk, labs[sk], days)
        pv = None
        if prev is not None:
            pv = (prev[0], RC.family_probs(ref, sk, prev[1][sk], prev[0]))
        ob = None
        if obs is not None:
            od, ol = obs[sk]
            ob = (od, RC.family_of_labels(ref, sk, ol[:, None], od)[:, 0])
        foot = ("k-means regimes (k = 4) on 5-day-mean 500 hPa anomalies, NCEP/NCAR R1 1991–2020, one set for Oct–Mar and "
                "one for Apr–Sep; every day is classified with the set of its own season, and regimes whose cold and warm "
                "patterns correlate ≥ 0.6 share a colour. A member-day\nwith pattern correlation below 0.25 to its nearest "
                "centroid is 'no regime'. Members de-drifted against the lead-matched GEPS8 hindcast climatology "
                "(2001–2020) and re-based to 1991–2020. Hatched weeks: the hindcast hit rate is not significantly above "
                "the better of persistence and climatology (year-block bootstrap, 95%).")
        for d in (FIGS, SITE):
            RF.render(d / f"geps_regimes_{sk}_fc.webp", ref, sk, base, P,
                      title=f"{spec['label']} weather regimes — GEPS, 21 members, init {base:%Y-%m-%d} 00Z",
                      subtitle="Share of members in each regime by day, days 1–35; the observed regime (GEPS analyses) "
                               "runs into the forecast from the left" + (f"; outline: previous run "
                               f"{RC.parse_date(prev_cycle):%b %-d}" if pv is not None else ""),
                      obs=ob, obs_label="observed (GEPS analyses)", prev=pv,
                      prev_label=f"previous run ({RC.parse_date(prev_cycle):%b %-d})" if pv is not None else None,
                      skills=[skill] if skill else None, footer=foot, n_label="% of 21 members")
        print(f"  geps_regimes_{sk}_fc.webp")
        F = len(RC.FAMILIES[sk])
        names = [ref.family_name(sk, f) for f in range(F)] + ["no regime"]
        rec = {"label": spec["label"], "families": [f["key"] for f in RC.FAMILIES[sk]] + ["none"], "names": names,
               "daily": {"t": [d.isoformat() for d in days], "set": [RC.season_of_date(d) for d in days],
                         "p": np.round(P, 3).tolist()},
               "weeks": [dict(w, p=[round(x, 3) for x in w["p"]]) for w in RC.weekly(P)]}
        if skill:
            rec["skill"] = skill["sectors"][sk]["weeks"]
        js["sectors"][sk] = rec
        print(f"  {sk}: " + "  ".join(f"wk{w['w0'] // 7 + 1} " + "/".join(f"{p * 100:.0f}" for p in w["p"])
                                      for w in rec["weeks"]))
    (TC / "regimes.json").write_text(json.dumps(js))
    (SITE / "regimes.json").write_text(json.dumps(js))
    mem = RC.members_record("geps", "GEPS", base, 21, days, labs,
                            {sk: obs[sk] for sk in RC.SECTORS} if obs else None,
                            notes="ECCC GEPS extended ensemble, 00Z, control + 20; anomalies vs the GEPS8 hindcast "
                                  "climatology re-based to 1991-2020", max_age_days=5)
    (SITE / "regimes_members.json").write_text(json.dumps(mem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
