#!/usr/bin/env python3
"""Weather-regime probabilities from every GEFS member, days 1-35, Euro-Atlantic and North American sectors - the GEPS
page's regime product on GEFS (ported 2026-09-26, revamped 2026-09-28 with the shared scripts/regimes code).

  members     each member's de-drifted, re-based daily 500 hPa anomaly (gefs_patterns.member_anoms, 2.5 deg), 5-day
              running mean along the lead, assigned to the nearest centroid of the set of EACH VALID DAY'S season
              (Oct-Mar cold, Apr-Sep warm; regime_core), or "no regime" below pattern correlation 0.25
  observed    GEFS's own analyses (12Z + next 00Z pair per day, gefs_patterns.observed_anoms), each day in its own set
  previous    the most recent earlier run in the archive (normally yesterday's), a thin outline on the common days
  skill       the GEFSv12 reforecast 2000-2019 (scripts/regimes/data/regime_hindcast_gefs.npz, season-aware): hit rate
              of the 4-member-mean regime by lead with a 95% year-block bootstrap interval, against persistence and
              climatology, starts within +-45 d of this date

    python gefs_regimes.py --npz /tmp/gefs_work/gefs2_20260925.npz --date 20260925 --site . --cache /tmp/gefs_work
    -> assets/gefs/gefs_regimes_{ea,na}_fc.webp, assets/gefs/data/gefs_regimes.json (summary),
       assets/gefs/data/gefs_regimes_members.json (per-member labels, the multi-model view's input);
       run archive gefs_regimes_<date>.npz (labels)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "regimes"))
import gefs_patterns as GP                                           # noqa: E402
import regime_core as RC                                             # noqa: E402
import regime_figure as RF                                           # noqa: E402


def member_labels(ref, z500, days):
    """z500 (35, M, 73, 144) -> {sk: labels (35, M)}, each day in its own season's set."""
    out = {}
    for sk, spec in RC.SECTORS.items():
        sec, _, _ = RC.sector_cut(z500, spec)
        v = RC.running_partial(sec, RC.SMOOTH, axis=0)
        n, m = v.shape[:2]
        lab, _ = ref.classify_dated(v.reshape(n * m, *v.shape[2:]), np.repeat(days, m), sk)
        out[sk] = lab.reshape(n, m)
    return out


def previous(ref, runs, date):
    """Previous run's family probabilities per sector on its own valid days, from the archive (labels since
    2026-09-28; before that both held-set probabilities, re-read day by day)."""
    pdate, pz = GP.previous_run(runs, "regimes", date)
    if pz is None:
        return None, {}
    pdays = GP.valid_days(GP.parse_date(pdate))
    out = {}
    for sk in RC.SECTORS:
        if f"lab_{sk}" in pz.files:
            out[sk] = (pdays, RC.family_probs(ref, sk, pz[f"lab_{sk}"], pdays))
        elif all(f"{sk}_{s}" in pz.files for s in RC.SEASONS):
            out[sk] = (pdays, RC.season_probs_to_family(ref, sk, {s: pz[f"{sk}_{s}"].astype("float64")
                                                                  for s in RC.SEASONS}, pdays))
    return pdate, out


def main() -> int:
    ap = GP.common_args(argparse.ArgumentParser())
    a = ap.parse_args()
    base, out, data, runs, cache, gref, clim, z = GP.setup(a)
    if clim is None:
        print("no daily climatology (gefs-clim-v2): regimes skipped"); return 0
    ref = RC.Ref()
    n_members = int(len(z["members"]))
    days = GP.valid_days(base)
    z500 = GP.member_anoms(z, base, gref, {"zg500": clim["zg500"]})["zg500"]
    labs = member_labels(ref, z500, days)
    prev_date, prev = previous(ref, runs, a.date)
    obs_fields, ts = GP.observed_anoms(base, gref, runs)
    GP.prune_tail(runs, base)
    obs = {}
    if ts and "zg500" in obs_fields:
        for sk, spec in RC.SECTORS.items():
            sec, _, _ = RC.sector_cut(obs_fields["zg500"], spec)
            obs[sk] = (list(ts), ref.classify_dated(RC.running_partial(sec, RC.SMOOTH, axis=0), ts, sk)[0])
    skill = RC.hindcast_skill("gefs", base)
    js = {"cycle": a.date, "prev_cycle": prev_date, "r_min": RC.R_MIN, "n_members": n_members,
          "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
          "classification": "each valid day in its own season's set (Oct-Mar cold, Apr-Sep warm)",
          "valid_convention": "day d valid on init + (d - 1)", "test": bool(GP.TEST_NOTE), "sectors": {}}
    foot = ("k-means regimes (k = 4) on 5-day-mean 500 hPa anomalies, NCEP/NCAR R1 1991–2020, one set for Oct–Mar and one "
            "for Apr–Sep; every day is classified with the set of its own season, and regimes whose cold and warm patterns "
            "correlate ≥ 0.6 share a colour. A member-day\nwith pattern correlation below 0.25 to its nearest centroid is "
            "'no regime'. Members de-drifted against the lead-matched GEFSv12 reforecast climatology (2000–2019) with the "
            "operational drift removed, re-based to 1991–2020. Hatched weeks: the reforecast hit rate is not significantly "
            "above the better of persistence and climatology (year-block bootstrap, 95%).")
    for sk, spec in RC.SECTORS.items():
        P = RC.family_probs(ref, sk, labs[sk], days)
        ob = None
        if sk in obs:
            od, ol = obs[sk]
            ob = (od, RC.family_of_labels(ref, sk, np.asarray(ol)[:, None], od)[:, 0])
        pv = prev.get(sk)
        RF.render(out / f"gefs_regimes_{sk}_fc.webp", ref, sk, base, P,
                  title=f"{spec['label']} weather regimes — GEFS, {n_members} members, init {base:%Y-%m-%d} 00Z"
                        + GP.TEST_NOTE,
                  subtitle="Share of members in each regime by day, days 1–35; the observed regime (GEFS analyses) runs "
                           "into the forecast from the left" + (f"; outline: previous run {GP.parse_date(prev_date):%b %-d}"
                                                               if pv is not None else ""),
                  obs=ob, obs_label="observed (GEFS analyses)", prev=pv,
                  prev_label=f"previous run ({GP.parse_date(prev_date):%b %-d})" if pv is not None else None,
                  skills=[skill] if skill else None, footer=foot, n_label=f"% of {n_members} members")
        print(f"  gefs_regimes_{sk}_fc.webp", flush=True)
        F = len(RC.FAMILIES[sk])
        rec = {"label": spec["label"], "families": [f["key"] for f in RC.FAMILIES[sk]] + ["none"],
               "names": [ref.family_name(sk, f) for f in range(F)] + ["no regime"],
               "daily": {"t": [d.isoformat() for d in days], "set": [RC.season_of_date(d) for d in days],
                         "p": np.round(P, 3).tolist()},
               "weeks": [dict(w, p=[round(x, 3) for x in w["p"]]) for w in RC.weekly(P)]}
        if skill:
            rec["skill"] = skill["sectors"][sk]["weeks"]
        js["sectors"][sk] = rec
        print(f"  {sk}: " + "  ".join(f"wk{w['w0'] // 7 + 1} " + "/".join(f"{p * 100:.0f}" for p in w["p"])
                                      for w in rec["weeks"]), flush=True)
    (data / "gefs_regimes.json").write_text(json.dumps(js, allow_nan=False))
    mem = RC.members_record("gefs", "GEFS", base, n_members, days, labs, obs or None,
                            notes="NOAA GEFS 00Z, control + 30; anomalies vs the GEFSv12 reforecast climatology with the "
                                  "operational drift removed, re-based to 1991-2020", max_age_days=3)
    (data / "gefs_regimes_members.json").write_text(json.dumps(mem))
    np.savez_compressed(runs / f"gefs_regimes_{a.date}.npz", **{f"lab_{sk}": v.astype("int8") for sk, v in labs.items()})
    GP.prune_runs(runs, "regimes", base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
