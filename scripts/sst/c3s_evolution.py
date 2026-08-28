#!/usr/bin/env python3
"""Evolution of the C3S multi-model ENSO forecast across issue months.

Maintains assets/sst/data/c3s_evolution.json — for every C3S issue month, the
multi-model-mean (model-weighted, like the plume) Niño-3.4 and RONI-scaled
relative anomaly per lead. The forecasts page draws one line per issue, so the
reader can watch successive monthly outlooks converge (or not) on the event.

Reuses c3s_nino34's whole retrieval/anomalization stack: forecast GRIBs and the
hindcast climatology CSV are cached, so re-running over an already-fetched
issue costs no CDS traffic. Issues already in the store are FROZEN (never
re-collected — a missing model would otherwise be re-probed on the CDS every
month, forever), except the newest one, which is recomputed each run so late
model arrivals within the release window are picked up.

    python c3s_evolution.py                # refresh newest cached issue only
    python c3s_evolution.py 202601 202602  # backfill specific issues
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c3s_nino34 as c3s

STORE = c3s.ASSETS / "data" / "c3s_evolution.json"

# Centres bump their system number over time; older issues live under the older
# number (e.g. UKMO was 604 through the 2026-01 issue, 610 after). Backfills try
# the current number first, then these. Each system anomalizes against its OWN
# hindcast (climatology is keyed centre_system), so mixing numbers across issues
# stays drift-clean. ECCC deliberately absent: its 4 and 5 are different models.
ALT_SYSTEMS = {"ukmo": ["605", "604", "603"], "dwd": ["21"], "meteo_france": ["8"]}

_orig_model_members = c3s.model_members


def _model_members_with_fallback(centre, system, ym):
    r = _orig_model_members(centre, system, ym)
    if r is not None:
        return r
    for alt in ALT_SYSTEMS.get(centre, []):
        if alt == system:
            continue
        r = _orig_model_members(centre, alt, ym)
        if r is not None:
            print(f"    ({centre}: fell back to system {alt} for {ym})")
            return r
    return None


c3s.model_members = _model_members_with_fallback


def load_store() -> dict:
    if STORE.exists():
        return json.loads(STORE.read_text())
    return {"issues": []}


def cached_issues() -> list[str]:
    """Issue months (YYYYMM) with at least one forecast GRIB on disk."""
    yms = set()
    for f in (c3s.DATA / "forecast").glob("*_n34.grib"):
        m = re.search(r"_(\d{6})_n34\.grib$", f.name)
        if m:
            yms.add(m.group(1))
    return sorted(yms)


def summarize(ym: str, results: dict) -> dict:
    """One store entry: model-weighted multi-model mean per lead plus the
    member-pooled P10/P90 envelope (same pooling as the plume's fan), both
    indices."""
    labels = sorted(results)
    out = {}
    for key in ("n34", "rnino"):
        mean, p10, p90 = [], [], []
        for L in range(1, c3s.MAXLEAD + 1):
            per_model = [results[l][0][key][L] for l in labels]
            mean.append(round(float(np.mean([a.mean() for a in per_model])), 3))
            pool = np.concatenate(per_model)
            p10.append(round(float(np.percentile(pool, 10)), 3))
            p90.append(round(float(np.percentile(pool, 90)), 3))
        out[key] = mean
        out[key + "_p10"] = p10
        out[key + "_p90"] = p90
    start = pd.Timestamp(int(ym[:4]), int(ym[4:]), 1)
    return {
        "issue": f"{ym[:4]}-{ym[4:]}",
        "valid_months": [(start + pd.DateOffset(months=L)).strftime("%Y-%m")
                         for L in range(c3s.MAXLEAD)],
        "models": labels,
        "nmem": int(sum(results[l][0]["n34"][1].size for l in labels)),
        **out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("issues", nargs="*", help="YYYYMM to (re)compute; default: newest cached")
    args = ap.parse_args()

    store = load_store()
    have = {e["issue"].replace("-", ""): e for e in store["issues"]}

    if args.issues:
        targets = args.issues
    else:
        cached = cached_issues()
        if not cached:
            print("no cached C3S issues found — run c3s_nino34.py first", file=sys.stderr)
            return 1
        targets = [cached[-1]]                      # newest only; older are frozen

    changed = False
    for ym in targets:
        print(f"{ym}: collecting …", flush=True)
        results = c3s.collect(ym)
        if not results:
            print(f"{ym}: no models — skipped")
            continue
        entry = summarize(ym, results)
        if have.get(ym) != entry:
            have[ym] = entry
            changed = True
            print(f"{ym}: {len(entry['models'])} models, {entry['nmem']} members · "
                  f"Niño-3.4 {entry['n34'][0]:+.2f}→{entry['n34'][-1]:+.2f}")
        else:
            print(f"{ym}: unchanged")

    if changed or not STORE.exists():
        store = {
            "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "hindcast": "1993-2016 (each model vs its own)",
            "issues": [have[k] for k in sorted(have)],
        }
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(store, separators=(",", ":")))
        print(f"saved {STORE.name} ({len(store['issues'])} issues)")
    else:
        print("store unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
