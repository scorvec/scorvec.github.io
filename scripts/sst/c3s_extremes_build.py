#!/usr/bin/env python3
"""Threshold days for every C3S system: cold days in the US, hot days in Brazil.

The SEAS5 card this generalises asks how many days a month fall past a threshold, and it
cannot be answered from a model's raw output: a seasonal model's diurnal range is too
small, so its Tmin bias is larger than its mean bias and a raw count of freezing days is
wrong by more than the signal. Every member is therefore QUANTILE-MAPPED per grid cell
from that system's own 1993-2016 hindcast extremes onto the CPC daily record, and only
then counted -- the same correction, applied per system, which is why each one needed its
own 14 GB of hindcast.

Seven systems, not eight: NCEP publishes no daily maximum/minimum temperature to this
collection at all (MarsNoDataError on every request), so it cannot be corrected or counted.

The multi-model pools members with equal weight per SYSTEM, as every other card here does.

Output is the national, population-weighted series -- expected days a month, with the
member spread and the CPC 1991-2020 normal beside it. The per-cell maps the SEAS5 card
draws are not rendered for seven systems; the JSON carries what a chart needs.

    SST_SITE_ROOT=. python scripts/sst/c3s_extremes_build.py --issue 202609
-> assets/sst/data/c3s_xdays.json
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c3s_nino34 import MODELS
from c3s_sixh import hc_path, path_for as fc_path
from seas5_extremes import REGIONS, THRESH
from seas5_extremes_build import cpc_daily, cpc_normal, quantile_map
from seas5_popT import pop_grid
from seas5_build import valid_months

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
OUT = SITE / "assets" / "sst" / "data" / "c3s_xdays.json"
NMONTH = 6
MIN_SYSTEMS = 3


def _read(paths, stat: str, want=None):
    """[sample, day, lat, lon] °C for ONE calendar month, folding member × start.

    `want` is (year, month) and it matters: a lagged system's members start on different
    days, so the same `step` is a different calendar day for each start. Flattening step as
    if it were "day" mixes late August into September for those systems -- which is exactly
    how UKMO first came out with more than twice ECMWF's count of cool days. Days are
    therefore selected on VALID TIME, as everywhere else on this page, and every start
    contributes the same calendar days.
    """
    import xarray as xr
    import pandas as pd
    out, lat, lon = [], None, None
    for p in paths:
        ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
        pick = [v for v in ds.data_vars if v.startswith("mn" if stat == "tn" else "mx")]
        if not pick:
            ds.close(); continue
        da = ds[pick[0]]
        for d in ("number", "time"):
            if d not in da.dims:
                da = da.expand_dims(d)
        da = da.transpose("number", "time", "step", "latitude", "longitude")
        a = da.values.astype(np.float32)
        lat, lon = da.latitude.values, da.longitude.values
        t0 = np.atleast_1d(da["time"].values)[:, None]
        valid = pd.to_datetime((t0 + da["step"].values[None, :]).ravel()).normalize()
        valid = np.asarray(valid).reshape(len(t0), -1)
        ds.close()
        if want is None:
            out.append(a.reshape(-1, *a.shape[2:]))
            continue
        yy, mm = want
        per_start = []
        for si in range(a.shape[1]):
            sel = np.array([str(d)[:7] == f"{yy}-{mm:02d}" for d in valid[si]])
            if sel.sum() < 25:                       # this start does not cover the month
                continue
            per_start.append(a[:, si, sel][:, :sel.sum()])
        if not per_start:
            continue
        n = min(x.shape[1] for x in per_start)
        out.append(np.concatenate([x[:, :n] for x in per_start], axis=0))
    if not out:
        return None, None, None
    n = min(x.shape[1] for x in out)
    a = np.concatenate([x[:, :n] for x in out], axis=0)
    keep = np.isfinite(a).any(axis=(1, 2, 3))          # padded combinations of a lagged cube
    return a[keep] - 273.15, lat, lon


def forecast_files(centre, system, region, issue, k):
    p = fc_path(centre, system, region, issue, k, "x")
    if p.exists():
        return [p]
    return sorted(p.parent.glob(p.stem + "_p*" + p.suffix))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--regions", default="us,br")
    a = ap.parse_args()
    issue = a.issue
    regions = [r for r in a.regions.split(",") if r in REGIONS]
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": issue,
           "issue_label": f"{calendar.month_abbr[int(issue[4:6])]} {issue[:4]}",
           "note": ("members quantile-mapped per cell from each system's own 1993–2016 hindcast "
                    "extremes onto the CPC daily record, then counted; NCEP is absent because it "
                    "publishes no daily extremes to this collection"),
           "models": [], "regions": {}}
    seen = []

    for region in regions:
        label, cc, area, unit = REGIONS[region]
        rec = {"label": label, "sets": {}, "models": {}}
        w = None
        for set_key, stat, thr_label, thrs, op in THRESH[region]:
            rec["sets"][set_key] = {"kind": thr_label, "stat": stat, "thresholds": thrs,
                                    "months": [], "op": op}
        for entry in MODELS:
            centre, system, mlabel = entry[0], str(entry[1]), entry[2]
            colour = entry[3] if len(entry) > 3 else None
            mid = f"{centre}_{system}"
            per_set = {}
            for set_key, stat, thr_label, thrs, op in THRESH[region]:
                months_out = {}
                for k in range(1, NMONTH + 1):
                    fps = forecast_files(centre, system, region, issue, k)
                    hp = hc_path(centre, system, region, issue[4:6], k)
                    if not fps or not hp.exists():
                        continue
                    vm_k = valid_months(issue)[k - 1]
                    fc, lat, lon = _read(fps, stat, want=(int(vm_k[:4]), int(vm_k[5:])))
                    hc, _, _ = _read([hp], stat)      # one start per hindcast year: step IS the day
                    if fc is None or hc is None:
                        continue
                    if w is None:
                        w = pop_grid(region, lat, lon)
                        w = w / w.sum()
                    vm = vm_k
                    mo = int(vm[5:])
                    ov, om, oyrs = cpc_daily(stat, region, range(1993, 2017), lat, lon)
                    if ov is None or len(oyrs) < 20:
                        continue
                    field = quantile_map(fc, hc, ov[om == mo])
                    del hc
                    normal, used = cpc_normal(region, stat, lat, lon)
                    row = {}
                    for t in thrs:
                        hit = (field <= t) if op == "le" else (field >= t)
                        cnt = hit.sum(1).astype(np.float32)                  # [member, lat, lon]
                        days = np.tensordot(cnt, w, axes=([1, 2], [0, 1]))   # [member]
                        nrm = normal[t][mo - 1] if used else None
                        pn = (float(np.nansum(np.where(np.isfinite(nrm), nrm, 0) * w)
                                    / max(np.sum(w[np.isfinite(nrm)]), 1e-9)) if used else None)
                        row[str(t)] = {"days": round(float(days.mean()), 2),
                                       "p10": round(float(np.percentile(days, 10)), 2),
                                       "p90": round(float(np.percentile(days, 90)), 2),
                                       "normal": round(pn, 2) if pn is not None else None,
                                       "pct": round(100 * float(days.mean()) / pn, 1) if pn else None,
                                       "members": int(days.size)}
                    months_out[vm] = row
                    del field, fc
                if months_out:
                    per_set[set_key] = months_out
                    if not rec["sets"][set_key]["months"]:
                        rec["sets"][set_key]["months"] = list(months_out)
            if per_set:
                rec["models"][mid] = per_set
                if mid not in [m["id"] for m in seen]:
                    seen.append({"id": mid, "label": mlabel, "color": colour})
                first = next(iter(per_set.values()))
                mon = next(iter(first))
                print(f"  {region} {mlabel}: " + ", ".join(
                    f"{t}°C {v['days']:.1f} d ({v['pct']}% of normal)" for t, v in first[mon].items()), flush=True)

        # multi-model: the mean of the systems' expected days, equal weight per system, and
        # the spread taken across systems' own p10/p90 so the band still means something
        mm = {}
        for set_key in rec["sets"]:
            months = rec["sets"][set_key]["months"]
            out = {}
            for vm in months:
                row = {}
                for t in [str(x) for x in rec["sets"][set_key]["thresholds"]]:
                    vals = [rec["models"][m][set_key][vm][t] for m in rec["models"]
                            if set_key in rec["models"][m] and vm in rec["models"][m][set_key]
                            and t in rec["models"][m][set_key][vm]]
                    if len(vals) < MIN_SYSTEMS:
                        continue
                    nrm = [v["normal"] for v in vals if v["normal"] is not None]
                    # a threshold nothing reaches in this month has a normal of exactly zero
                    # (the US never sees -20 C in September), and "% of normal" is then not a
                    # number — the guard has to be on the VALUE, not on the list being non-empty
                    nmean = float(np.mean(nrm)) if nrm else None
                    row[t] = {"days": round(float(np.mean([v["days"] for v in vals])), 2),
                              "p10": round(float(np.mean([v["p10"] for v in vals])), 2),
                              "p90": round(float(np.mean([v["p90"] for v in vals])), 2),
                              "normal": round(nmean, 2) if nmean is not None else None,
                              "pct": (round(100 * float(np.mean([v["days"] for v in vals])) / nmean, 1)
                                      if nmean else None),
                              "systems": len(vals)}
                if row:
                    out[vm] = row
            if out:
                mm[set_key] = out
        if mm:
            rec["models"]["mmm"] = mm
        doc["regions"][region] = rec

    if not doc["regions"]:
        raise SystemExit("no extremes on disk — run c3s_sixh.py --var x and --hindcast first")
    doc["models"] = [{"id": "mmm", "label": "Multi-model", "color": "#111111"}] + seen
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT.name}: {len(doc['regions'])} region(s), {len(doc['models'])} model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
