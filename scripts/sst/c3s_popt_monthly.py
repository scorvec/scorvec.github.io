#!/usr/bin/env python3
"""Population-weighted MONTHLY temperature, every member of every C3S system.

The SEAS5 card next to this one pools 6-hourly members into daily values, which resolves
a cold snap but costs ~120 MB per country per issue and exists for ECMWF alone. This is the
monthly-mean cousin, and it is free: the monthly members for all eight systems are already
on disk from the tercile build (scripts/sst/data/c3s/terc), as is each system's matching
1993-2016 hindcast. So the same question -- how warm a month, for the people who feel it --
gets asked of every system, with the spread across members rather than across member-days.

Read it as a monthly distribution, not a daily one: the width here is the spread of
monthly MEANS, which is narrower than the daily spread and says nothing about cold snaps.

Anomalies are each member minus that system's own hindcast mean at the same lead, so the
systems are comparable; the raw levels carry each model's bias and are kept for the issue-
to-issue read within one system.

Population: geonames cities15000 through seas5_popT.pop_grid, the same places and the same
1 deg cells the SEAS5 card uses.

    SST_SITE_ROOT=. python scripts/sst/c3s_popt_monthly.py --issue 202609
-> assets/sst/data/c3s_popt.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c3s_nino34 import MODELS
from c3s_terciles import path_for
from c3s_terciles_render import period_means
from seas5_popT import REGIONS, pop_grid

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
OUT = SITE / "assets" / "sst" / "data" / "c3s_popt.json"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LEADS = {f"m{i + 1}": [i] for i in range(6)}          # one lead month per column
QS = [10, 25, 50, 75, 90]


def weights_for(region: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Population per cell on THIS system's grid, zero outside the country box."""
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    w = pop_grid(region, lat, lon180)
    return w


def series(region: str, centre: str, system: str, issue: str):
    """(fc [member, lead], hc_mean [lead], months) population-weighted, in K."""
    fp, hp = path_for("fc", centre, system, "t2m", issue), path_for("hc", centre, system, "t2m", issue)
    if not (fp.exists() and hp.exists()):
        return None
    fcm, lat, lon, _ = period_means(fp, "t2m", 1.0, LEADS)
    hcm, lat_h, lon_h, _ = period_means(hp, "t2m", 1.0, LEADS)
    if not fcm or not hcm:
        return None
    w = weights_for(region, lat, lon)
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    keys = [k for k in LEADS if k in fcm and k in hcm]
    fc = np.stack([(fcm[k] * w[None]).sum(axis=(1, 2)) for k in keys], axis=1)      # [member, lead]
    hc = np.array([float((hcm[k] * w[None]).sum(axis=(1, 2)).mean()) for k in keys])
    return fc, hc, keys


def to_unit(v, unit):
    return (v - 273.15) * 9 / 5 + 32 if unit == "F" else v - 273.15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--regions", default="us,br")
    a = ap.parse_args()
    issue = a.issue
    y, mo = int(issue[:4]), int(issue[4:6])
    regions = [r for r in a.regions.split(",") if r in REGIONS]

    doc = {"issue": f"{y}-{mo:02d}", "issue_label": f"{MONTHS[mo - 1]} {y}",
           "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "regions": {}, "models": [],
           "note": ("monthly means, population-weighted; spread is across MEMBERS of that system, "
                    "not across days — a cold snap does not show here")}
    seen = []
    for region in regions:
        label, cc, area, unit = REGIONS[region]
        months, per_model, pooled = None, {}, []
        for entry in MODELS:
            centre, system, mlabel = entry[0], str(entry[1]), entry[2]
            colour = entry[3] if len(entry) > 3 else None
            got = series(region, centre, system, issue)
            if got is None:
                continue
            fc, hc, keys = got
            if months is None:
                months = [f"{MONTHS[(mo - 1 + int(k[1:]) - 1) % 12]} {y + (mo - 1 + int(k[1:]) - 1) // 12}" for k in keys]
            mid = f"{centre}_{system}"
            anom = fc - hc[None]
            per_model[mid] = {
                "raw": {f"p{q}": [round(float(x), 2) for x in to_unit(np.percentile(fc, q, axis=0), unit)] for q in QS},
                "anom": {f"p{q}": [round(float(x * (9 / 5 if unit == "F" else 1.0)), 2)
                                   for x in np.percentile(anom, q, axis=0)] for q in QS},
                "mean_raw": [round(float(x), 2) for x in to_unit(fc.mean(axis=0), unit)],
                "mean_anom": [round(float(x * (9 / 5 if unit == "F" else 1.0)), 2) for x in anom.mean(axis=0)],
                "members": int(fc.shape[0]),
            }
            # equal weight per SYSTEM in the pool, whatever its ensemble size
            pooled.append((anom, fc, 1.0 / len(MODELS)))
            if mid not in [m["id"] for m in seen]:
                seen.append({"id": mid, "label": mlabel, "color": colour})
            print(f"  {region} {mlabel}: {fc.shape[0]} members, anomaly {np.round(anom.mean(axis=0), 2)}", flush=True)
        if not per_model:
            continue
        # multi-model: every system's members pooled with equal system weight
        wts = np.concatenate([np.full(a_.shape[0], w) for a_, _, w in pooled])
        A = np.concatenate([a_ for a_, _, _ in pooled], axis=0)
        F = np.concatenate([f_ for _, f_, _ in pooled], axis=0)
        def wq(x, q):
            o = np.argsort(x, axis=0)
            xs = np.take_along_axis(x, o, axis=0)
            ws = wts[o]
            c = np.cumsum(ws, axis=0) / ws.sum(axis=0)
            return np.array([np.interp(q / 100.0, c[:, j], xs[:, j]) for j in range(x.shape[1])])
        per_model["mmm"] = {
            "raw": {f"p{q}": [round(float(x), 2) for x in to_unit(wq(F, q), unit)] for q in QS},
            "anom": {f"p{q}": [round(float(x * (9 / 5 if unit == "F" else 1.0)), 2) for x in wq(A, q)] for q in QS},
            "mean_raw": [round(float(x), 2) for x in to_unit((F * wts[:, None]).sum(0) / wts.sum(), unit)],
            "mean_anom": [round(float(v * (9 / 5 if unit == "F" else 1.0)), 2)
                          for v in (A * wts[:, None]).sum(0) / wts.sum()],
            "members": int(A.shape[0]),
        }
        doc["regions"][region] = {"label": label, "unit": "°F" if unit == "F" else "°C",
                                  "months": months, "values": per_model}
    doc["models"] = [{"id": "mmm", "label": "Multi-model", "color": "#111111"}] + seen
    if not doc["regions"]:
        raise SystemExit("no members on disk — run c3s_terciles.py first")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT.name}: {len(doc['regions'])} region(s), {len(doc['models'])} model(s) incl. the pool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
