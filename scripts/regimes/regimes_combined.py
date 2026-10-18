#!/usr/bin/env python3
"""The multi-model weather-regime view for subseasonal.html (2026-09-28): GEPS (21 members, days 1-35), GEFS (31, days
1-35), AIFS-ENS (26, days 1-15) and IFS-ENS (50, days 1-15), from the per-model member files every producer writes in
one format (regime_core.members_record):

  geps  assets/geps/regimes_members.json                 geps.yml (run_geps.sh), Mon/Thu
  gefs  assets/gefs/data/gefs_regimes_members.json       gefs.yml, daily
  aifs  assets/regimes/data/regimes_members_aifs.json    strat.yml (00Z), daily
  ifs   assets/regimes/data/regimes_members_ifs.json     strat-ifs.yml (00Z), daily

COMBINING. Each model's newest run is used if it is no older than its `max_age_days` (GEPS 5, GEFS 3, ECMWF 2) -
otherwise it is left out and the figure says so. The axis starts at the newest init; a model whose run is older
contributes the same valid DATES (at its own, longer leads). The combined probability of a regime on a date is the
EQUAL-WEIGHT mean of the models that cover that date: skill weighting would need hindcasts of all four models, and
AIFS-ENS and IFS-ENS have none in open data; GEPS and GEFS, which do, are within each other's 95% intervals at nearly
every lead (overlapping 95% intervals in 827 of 840 month x sector x lead combinations at mid-month inits; GEPS is
higher at the exceptions at days 1-4, GEFS at those after day 13). Days 16-35 are GEPS and GEFS only; the axis ends at the last date two models cover.

Observed tail: GEFS analyses (daily; GEPS's when the GEFS file is missing). Previous issue: the combined view of the
previous day, kept in regimes_combined_history.json (10 issues).

    python scripts/regimes/regimes_combined.py [--date YYYYMMDD] [--force]
    -> assets/regimes/regimes_{ea,na}_combined.webp, assets/regimes/data/regimes_combined.json (+ _history.json)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import regime_core as RC
import regime_figure as RF

SOURCES = [
    ("geps", REPO / "assets/geps/regimes_members.json"),
    ("gefs", REPO / "assets/gefs/data/gefs_regimes_members.json"),
    ("aifs", REPO / "assets/regimes/data/regimes_members_aifs.json"),
    ("ifs", REPO / "assets/regimes/data/regimes_members_ifs.json"),
]
MAX_AGE = {"geps": 5, "gefs": 3, "aifs": 2, "ifs": 2}
OUT = REPO / "assets" / "regimes"
MAXDAYS = 35
KEEP_HISTORY = 10


def boot_combined(fls, F1, n_boot: int = 1000, seed: int = 17):
    """fls: per model, family slots (N axis days, members; -1 = no member that day). The combined probability with
    each model's members resampled with replacement and the models equal-weighted as in P: (n_boot, F1, N)."""
    rng = np.random.default_rng(seed)
    N = fls[0].shape[0]
    S = np.zeros((n_boot, F1, N))
    n = np.zeros((n_boot, N))
    for fl in fls:
        X = fl[:, rng.integers(0, fl.shape[1], (n_boot, fl.shape[1]))]  # (N, B, M)
        k = (X >= 0).sum(2)
        has = k > 0
        for f in range(F1):
            S[:, f, :] += np.where(has, (X == f).sum(2) / np.maximum(k, 1), 0.0).T  # noqa: SIM300  variable X (maths notation), not a constant
        n += has.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return S / n[:, None, :]


def weekly_vs_clim(P, C, CB, PB):
    """Per forecast week: the normal (R1 share over the week's valid days), the forecast minus normal, and its 95%
    interval from the two bootstraps (members within models; R1 years). sig = the interval excludes zero."""
    out = []
    for w in RC.weekly(P):
        sl = np.arange(P.shape[1])[w["w0"] - 1 : w["w1"]]
        days = sl[np.isfinite(P[:, sl]).all(0)]
        c = C[:, days].mean(1)
        d = np.nanmean(PB[:, :, days], 2) - CB[:, :, days].mean(2)
        lo, hi = np.nanpercentile(d, 2.5, 0), np.nanpercentile(d, 97.5, 0)
        out.append(
            dict(
                w0=w["w0"],
                clim=np.round(c, 3).tolist(),
                dev=np.round(np.array(w["p"]) - c, 3).tolist(),
                lo=np.round(lo, 3).tolist(),
                hi=np.round(hi, 3).tolist(),
                sig=[bool(x) for x in (lo > 0) | (hi < 0)],
            )
        )
    return out


def load(issue: dt.date):
    used, skipped, raw = [], [], b""
    for key, p in SOURCES:
        try:
            b = p.read_bytes()
            j = json.loads(b)
        except Exception as e:  # noqa: BLE001
            skipped.append((key, "no run available" if isinstance(e, FileNotFoundError) else "unreadable"))
            continue
        if j.get("schema") != RC.SCHEMA:
            skipped.append((key, "old format"))
            continue
        init = RC.parse_date(j["init"])
        age = (issue - init).days
        lim = j.get("max_age_days", MAX_AGE[key])
        if age > lim or age < 0:
            skipped.append((key, f"run {init:%b %-d} is {age} days old (limit {lim})"))
            continue
        raw += b
        used.append((key, j))
    return used, skipped, hashlib.sha1(raw).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="issue date (default: today UTC)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    issue = RC.parse_date(a.date) if a.date else dt.datetime.now(dt.UTC).date()
    used, skipped, digest = load(issue)
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    old = {}
    with contextlib.suppress(Exception):  # noqa: BLE001
        old = json.loads((data / "regimes_combined.json").read_text())
    if not a.force and old.get("inputs") == digest and old.get("issue") == issue.isoformat():
        print(f"  inputs unchanged since {old.get('generated')} - nothing to do")
        return 0
    if len(used) < 2:
        print(f"::warning::only {len(used)} regime model(s) current ({skipped}) - the multi-model view is not redrawn")
        return 0
    ref = RC.Ref()
    d0 = max(RC.parse_date(j["init"]) for _, j in used)
    cover = {}
    for _key, j in used:
        for t in j["valid"]:
            cover[t] = cover.get(t, 0) + 1
    last = max(RC.parse_date(t) for t, c in cover.items() if c >= 2)
    N = min(MAXDAYS, (last - d0).days + 1)
    dates = [d0 + dt.timedelta(days=i) for i in range(N)]
    iso = [d.isoformat() for d in dates]
    order = ["geps", "gefs", "aifs", "ifs"]
    used.sort(key=lambda kj: order.index(kj[0]))
    skills = [s for s in (RC.hindcast_skill("geps", d0), RC.hindcast_skill("gefs", d0)) if s]
    hist_p = data / "regimes_combined_history.json"
    try:
        hist = json.loads(hist_p.read_text())
    except Exception:  # noqa: BLE001
        hist = {}
    prev_issue = max([k for k in hist if k < issue.isoformat()], default=None)
    js = {
        "issue": issue.isoformat(),
        "start": d0.isoformat(),
        "inputs": digest,
        "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "weights": "equal per model, over the models covering each date",
        "models": [
            {"model": k, "label": j["label"], "init": j["init"], "n_members": j["n_members"], "days": len(j["valid"])}
            for k, j in used
        ],
        "skipped": [{"model": k, "why": w} for k, w in skipped],
        "prev_issue": prev_issue,
        "sectors": {},
    }
    hist.setdefault(issue.isoformat(), {})
    for sk, spec in RC.SECTORS.items():
        F = len(RC.FAMILIES[sk])
        per, fls = [], []
        for key, j in used:
            vd = j["valid"]
            lab = np.array(j["sectors"][sk]["lab"])
            Pm = RC.family_probs(ref, sk, lab, vd)
            fm = RC.family_of_labels(ref, sk, lab, vd)
            A = np.full((F + 1, N), np.nan)
            FL = np.full((N, fm.shape[1]), -1, "int16")
            for i, t in enumerate(iso):
                if t in vd:
                    A[:, i] = Pm[:, vd.index(t)]
                    FL[i] = fm[vd.index(t)]
            per.append((key, j, A))
            fls.append(FL)
        stack = np.stack([A for _, _, A in per])
        with np.errstate(invalid="ignore"):
            P = np.nanmean(stack, 0)
        nmod = np.isfinite(stack[:, 0]).sum(0)
        P[:, nmod == 0] = np.nan
        # observed tail: GEFS analyses, days GEFS lacks (its newest run is a day older) filled from GEPS analyses
        obs, src = {}, []
        for key in ("gefs", "geps"):
            j = next((j for k, j in used if k == key), None)
            if not j or "observed" not in j["sectors"][sk]:
                continue
            o = j["sectors"][sk]["observed"]
            od = [RC.parse_date(t) for t in o["t"]]
            fam = RC.family_of_labels(ref, sk, np.array(o["lab"])[:, None], od)[:, 0]
            n0 = len(obs)
            for d, f in zip(od, fam):
                if d < d0 and d not in obs and f >= 0:
                    obs[d] = int(f)
            if len(obs) > n0:
                src.append({"gefs": "GEFS", "geps": "GEPS"}[key])
        obs = (sorted(obs), [obs[d] for d in sorted(obs)], " / ".join(src)) if obs else None
        prev = None
        if prev_issue and sk in hist.get(prev_issue, {}):
            h = hist[prev_issue][sk]
            pd0 = RC.parse_date(h["start"])
            pp = np.array(h["p"], dtype="float64")
            prev = ([pd0 + dt.timedelta(days=i) for i in range(pp.shape[1])], pp)
        hist[issue.isoformat()][sk] = {"start": d0.isoformat(), "p": np.round(np.nan_to_num(P, nan=0.0), 3).tolist()}
        models = [(f"{j['label']} · {RC.parse_date(j['init']):%b %-d}", A) for _, j, A in per]
        C, CB = RC.climatology(ref, sk, dates)
        vs = weekly_vs_clim(P, C, CB, boot_combined(fls, F + 1))
        mlabel = " + ".join(f"{j['label']} {j['n_members']}" for _, j, _ in per)
        sk_note = (
            (
                "  Left out: "
                + "; ".join(
                    f"{dict(geps='GEPS', gefs='GEFS', aifs='AIFS-ENS', ifs='IFS-ENS')[k]} ({w})" for k, w in skipped
                )
            )
            if skipped
            else ""
        )
        foot = (
            "Combined = equal-weight mean of the models covering each date (skill weighting would need hindcasts of "
            "every model; AIFS-ENS and IFS-ENS have none, and the GEPS and GEFS hindcast hit rates have overlapping 95% "
            "intervals at 98% of month × lead × sector combinations). "
            "Each model: members' 5-day-mean 500 hPa anomalies\nassigned to k-means regimes (NCEP/NCAR R1 1991–2020; "
            "cold set Oct–Mar, warm Apr–Sep, each day in its own season's set; 'no regime' below pattern correlation "
            "0.25). GEPS/GEFS de-drifted with their hindcasts; AIFS-ENS/IFS-ENS against ERA5 1991–2020 "
            "(drift correction accruing).\nHatched weeks: neither hindcast beats the better of persistence and "
            "climatology significantly (year-block bootstrap, 95%)" + sk_note + ".\n"
            "Second small number: the forecast minus the normal (NCEP R1 1991–2020 share of days in that regime within "
            "±15 days of each date, each day in its own season's set), in percentage points, printed where its 95% "
            "interval excludes zero (members resampled within each model, R1 years resampled), else 'n.s.'; in a "
            "hatched week a significant difference is the models' agreement, not a skilful forecast."
        )
        RF.render(
            OUT / f"regimes_{sk}_combined.webp",
            ref,
            sk,
            d0,
            P,
            title=f"{spec['label']} weather regimes — multi-model, from {d0:%b %-d}",
            subtitle=f"{mlabel} members, each model weighted equally; days 16–35 from GEPS and GEFS only"
            + (f"; outline: the {RC.parse_date(prev_issue):%b %-d} issue" if prev else ""),
            obs=(obs[0], obs[1]) if obs else None,
            obs_label=f"observed ({obs[2]} analyses)" if obs else "observed",
            prev=prev,
            prev_label=f"previous issue, {RC.parse_date(prev_issue):%b %-d} (outline)" if prev else None,
            models=models,
            skills=skills,
            footer=foot,
            n_label="probability (%)",
            vs_clim=vs,
        )
        print(
            f"  regimes_{sk}_combined.webp: "
            + ", ".join(k for k, _, _ in per)
            + "  "
            + "  ".join(f"wk{w['w0'] // 7 + 1} " + "/".join(f"{p * 100:.0f}" for p in w["p"]) for w in RC.weekly(P)),
            flush=True,
        )
        js["sectors"][sk] = {
            "label": spec["label"],
            "families": [f["key"] for f in RC.FAMILIES[sk]] + ["none"],
            "names": [ref.family_name(sk, f) for f in range(F)] + ["no regime"],
            "daily": {
                "t": iso,
                "set": [RC.season_of_date(d) for d in dates],
                "n_models": nmod.tolist(),
                "p": np.round(np.nan_to_num(P, nan=0.0), 3).tolist(),
            },
            "weeks": [dict(w, p=[round(x, 3) for x in w["p"]]) for w in RC.weekly(P)],
            "models": {k: [dict(w, p=[round(x, 3) for x in w["p"]]) for w in RC.weekly(A)] for k, _, A in per},
            "skill": {s["model"]: s["sectors"][sk]["weeks"] for s in skills},
            "vs_normal": {
                "base": "NCEP R1 1991-2020, +-15 days of each valid date, each day in its own season's set",
                "weeks": vs,
            },
        }
    (data / "regimes_combined.json").write_text(json.dumps(js, allow_nan=False))
    hist = {k: hist[k] for k in sorted(hist)[-KEEP_HISTORY:]}
    hist_p.write_text(json.dumps(hist, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
