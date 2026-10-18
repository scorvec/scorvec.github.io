#!/usr/bin/env python3
"""Recurvature watch for live storms (diagnostic only).

Rule: a Northern-Hemisphere storm is on WATCH when at least WATCH_P of the members of EVERY ensemble that tracks it
(AIFS-ENS and IFS-ENS, from data/tcens.json) recurve, with the median recurvature by day WATCH_DAY. North Pacific and
North Atlantic only: a North Indian storm (id letter A/B, or the control west of 100E) is listed, never on watch.
The watch says which storms are likely to recurve, in how many members, and when - nothing about what a recurving storm
does to the flow.

    python src/tc_alert.py --out-dir ../../assets/sst [--date 20261015 --time 12]
Reads data/tcens.json (and, for the storms' basins, data/tcjet.json of the same cycle) and writes data/tc_alert.json only;
the circulation page renders the watch line from it. With --date/--time, a tcens.json from any other cycle means this
cycle's ensemble tracks are missing: the file then records status "missing" instead of a stale watch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tc_jet as TC

WATCH_P = 0.5
WATCH_DAY = 10.0
BASIN_OF = {"W": "WP", "E": "EP", "C": "EP", "L": "NA", "A": "NI", "B": "NI"}
BASIN_LAB = {
    "WP": "western North Pacific",
    "EP": "eastern North Pacific",
    "NA": "North Atlantic",
    "NI": "North Indian Ocean",
}
NI_LON = 100.0  # west of 100E the control's position is in the North Indian Ocean (Arabian Sea / Bay of Bengal)
MLAB = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS", "aifs-ens": "AIFS-ENS", "ifs-ens": "IFS-ENS"}


def write(out: Path, js: dict) -> None:
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / "tc_alert.json").write_text(json.dumps(js))
    print("  " + " | ".join(js["lines"]), flush=True)


def missing(out: Path, init: str | None, why: str) -> int:
    """tc_alert.json for a cycle whose ensemble tracks are missing: no watch is given, and the file says why."""
    TC.report(f"recurvature watch: {why}", "warning")
    write(
        out,
        dict(
            init=init,
            status="missing",
            level="missing",
            rule=dict(p_recurve_min=WATCH_P, by_day=WATCH_DAY, every_ensemble=True),
            watch=[],
            not_on_watch=[],
            missing_models=["aifs", "ifs"],
            lines=[f"Recurvature watch: {TC.UNAVAILABLE.lower()}."],
        ),
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../../assets/sst")
    ap.add_argument("--date", default=None, help="the cycle (YYYYMMDD) this run is for; with --time")
    ap.add_argument("--time", default=None, help="the cycle hour, 00 or 12")
    a = ap.parse_args()
    out = Path(a.out_dir)
    want = f"{pd.Timestamp(f'{a.date}T{a.time}:00'):%Y-%m-%dT%H:%MZ}" if a.date and a.time else None
    ens_f, jet_f = out / "data" / "tcens.json", out / "data" / "tcjet.json"
    if not ens_f.exists():
        return missing(out, want, "no tcens.json")
    ens = json.loads(ens_f.read_text())
    if want and ens.get("init") != want:
        return missing(out, want, f"tcens.json is from {ens.get('init')}, not this cycle ({want})")
    if ens.get("status") == "missing":
        return missing(out, ens["init"], "both ensembles' tracker files are missing")
    gone = [m for m, q in (ens.get("models") or {}).items() if q.get("status") == "missing"]
    jet = json.loads(jet_f.read_text()) if jet_f.exists() else {"storms": []}
    init = pd.Timestamp(ens["init"].rstrip("Z"))
    # the control's tracks only from the SAME cycle: a stale file must never be combined with a new one
    ctl = {s["id"]: s for s in jet.get("storms", [])} if jet.get("init") == ens["init"] else {}
    alerts, others = [], []
    for s in ens["storms"]:
        sid = str(s.get("id") or "")
        if not s["models"]:
            continue
        ps = {m: q["p_recurve"] for m, q in s["models"].items()}
        med = [
            q["recurve_day_p10_p50_p90"][1]
            for q in s["models"].values()
            if q.get("recurve_day_p10_p50_p90") and q["recurve_day_p10_p50_p90"][1] is not None
        ]
        mday = float(np.median(med)) if med else None
        on = all(p >= WATCH_P for p in ps.values()) and mday is not None and mday <= WATCH_DAY
        when = init + pd.Timedelta(days=mday or 0)
        # basin = where the CONTROL has the storm nearest the median recurvature time (Nolo 2026 is 15E but recurves near
        # 168E: its downstream is the western Pacific's); the id letter only when the control does not carry the storm
        core = sid[4:] if sid.startswith("ifs-") else sid  # an IFS-ENS-only system is "ifs-71W" (tc_ens)
        basin = BASIN_OF.get(core.split("-")[0][-1:].upper())  # tc_ens names a 2nd group under one id "72E-2"
        # the control's track through the cluster's own control ids (tc_ens "control_ids"): the control's tracker id can
        # differ from the cluster's (e.g. control 72L in cluster 73L). A tcens.json without the field: the id itself
        cids = s["control_ids"] if "control_ids" in s else ([core] if not sid.startswith("ifs-") else [])
        tr = next((ctl[c]["track"] for c in cids if c in ctl and ctl[c].get("track")), [])
        if tr:
            k = int(np.argmin([abs(t["h"] - 24 * (mday if mday is not None else 0)) for t in tr]))
            lon360 = tr[k]["lon"] % 360
            basin = "NI" if lon360 < NI_LON else "WP" if lon360 < 180 else ("EP" if lon360 < 260 else "NA")
        if basin is None:
            continue
        if basin == "NI":
            # handled explicitly: a North Indian storm that "recurves" turns into the subcontinent or Myanmar, not into
            # the mid-latitude westerlies the watch is about; it is listed, never put on watch
            others.append(
                dict(
                    id=sid,
                    name=s["name"],
                    basin=basin,
                    watch=False,
                    p_recurve=ps,
                    median_recurve_day=mday,
                    recurve_date=when.strftime("%Y-%m-%d") if mday is not None else None,
                    note="North Indian Ocean: not part of the recurvature watch",
                )
            )
            continue
        e = dict(
            id=sid,
            name=s["name"],
            basin=basin,
            watch=bool(on),
            p_recurve=ps,
            median_recurve_day=mday,
            recurve_date=when.strftime("%Y-%m-%d") if mday is not None else None,
        )
        (alerts if on else others).append(e)
    level = "watch" if alerts else "none"
    lines = []
    if alerts:
        # each share labelled with its ensemble: a storm tracked by only one ensemble has only one share
        lines.append(
            "Recurvature watch (share of members recurving, median date): "
            + "; ".join(
                f"{x['name']} "
                + ", ".join(f"{MLAB.get(m, m)} {100 * p:.0f}%" for m, p in x["p_recurve"].items())
                + f", ~{pd.Timestamp(x['recurve_date']):%d %b}"
                for x in alerts
            )
        )
    else:
        lines.append(
            f"No recurvature watch: no North Pacific or North Atlantic storm has ≥ {100 * WATCH_P:.0f}% of members recurving in every "
            f"ensemble by day {WATCH_DAY:.0f}."
        )
    if gone:
        lines.append(
            f"{', '.join(MLAB.get(m, m) for m in gone)}: {TC.UNAVAILABLE.lower()}; the watch uses "
            f"{', '.join(MLAB.get(m, m) for m in (ens.get('models') or {}) if m not in gone)} alone."
        )
    write(
        out,
        dict(
            init=ens["init"],
            status="ok",
            level=level,
            rule=dict(p_recurve_min=WATCH_P, by_day=WATCH_DAY, every_ensemble=True),
            watch=alerts,
            not_on_watch=others,
            missing_models=gone,
            lines=lines,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
