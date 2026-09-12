#!/usr/bin/env python3
"""Daily population-weighted temperature distributions, every C3S system.

The monthly card (c3s_popt_monthly.py) shows the spread of monthly MEANS. This one pools
member-DAYS: every member, every day of the month, population-weighted into one national
number, which is the distribution a cold snap actually lives in. It needs the 6-hourly
members c3s_sixh.py fetches, and it is written to run on whatever has landed -- a system
whose chunks are missing is skipped, not faked.

Alignment is on VALID time, never on lead index. A lagged system (UKMO, NCEP, BoM) carries
members started across a month, so the same lead hour means a different calendar day for
each start; the chunk is fetched with a widened lead window and the days are selected here.

Anomalies use each system's own monthly hindcast mean at that lead -- the same reference
the monthly card uses -- so the systems are comparable at their different biases. A daily
hindcast would let the tails be calibrated too; that pull has not been made.

    SST_SITE_ROOT=. python scripts/sst/c3s_popt_daily.py --issue 202609
-> assets/sst/data/c3s_popt_daily.json
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
from c3s_sixh import path_for as sixh_path
from c3s_terciles import path_for as monthly_path
from c3s_terciles_render import period_means
from seas5_popT import REGIONS, pop_grid

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
OUT = SITE / "assets" / "sst" / "data" / "c3s_popt_daily.json"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
QS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
NMONTH = 6


def to_unit(v, unit):
    return (v - 273.15) * 9 / 5 + 32 if unit == "F" else v - 273.15


def daily_members(path: Path, region: str):
    """(values [sample, day], days) population-weighted daily means, K, for one chunk.

    `sample` folds member × start date: a lagged system's starts are separate ensembles of
    the same forecast, and for a distribution of days they pool.
    """
    import xarray as xr
    import pandas as pd
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]]
    lat, lon = da.latitude.values, da.longitude.values
    w = pop_grid(region, lat, np.where(lon > 180.0, lon - 360.0, lon))
    if w.sum() <= 0:
        ds.close(); return None, None
    w = w / w.sum()
    # valid time for every (start, step) pair. Both `number` and `time` are optional in a
    # grib: a single-start system has no time dimension, and a system whose chunk holds one
    # member per start has no number dimension. Give the array both, then transpose once.
    for d in ("number", "time"):
        if d not in da.dims:
            da = da.expand_dims(d)
    da = da.transpose("number", "time", "step", "latitude", "longitude")
    t0 = np.atleast_1d(da["time"].values)[:, None]
    valid = t0 + da["step"].values[None, :]
    v = (da.values * w[None, None, None]).sum(axis=(3, 4))            # [member, start, step]
    ds.close()
    days = pd.to_datetime(valid.ravel()).normalize().values.reshape(valid.shape)
    uniq = np.unique(days)
    out = np.full((v.shape[0] * v.shape[1], uniq.size), np.nan, dtype="float32")
    for j, d in enumerate(uniq):
        m = days == d                                                 # [start, step]
        if m.sum(axis=1).min() < 3:                                    # a partial day is not a daily mean
            continue
        for s in range(v.shape[1]):
            sel = m[s]
            if sel.sum() >= 3:
                out[s * v.shape[0]:(s + 1) * v.shape[0], j] = v[:, s, sel].mean(axis=1)
    keep = np.isfinite(out).any(axis=1)
    return out[keep], uniq


def hindcast_monthly(centre: str, system: str, region: str, issue: str, nmonth: int):
    """Monthly hindcast mean per lead, population-weighted, K — the anomaly reference."""
    hp = monthly_path("hc", centre, system, "t2m", issue)
    if not hp.exists():
        return None
    leads = {f"m{i + 1}": [i] for i in range(nmonth)}
    hcm, lat, lon, _ = period_means(hp, "t2m", 1.0, leads)
    if not hcm:
        return None
    w = pop_grid(region, lat, np.where(lon > 180.0, lon - 360.0, lon))
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    return {k: float((hcm[k] * w[None]).sum(axis=(1, 2)).mean()) for k in hcm}


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
           "note": ("member-DAYS pooled per month: every member, every day, population-weighted. "
                    "Anomalies use each system's monthly hindcast mean at that lead; the daily "
                    "hindcast that would calibrate the tails has not been pulled.")}
    seen = []
    for region in regions:
        label, cc, area, unit = REGIONS[region]
        months, per_model, pool = None, {}, []
        for entry in MODELS:
            centre, system, mlabel = entry[0], str(entry[1]), entry[2]
            colour = entry[3] if len(entry) > 3 else None
            hc = hindcast_monthly(centre, system, region, issue, NMONTH)
            cols, labels = [], []
            for k in range(1, NMONTH + 1):
                p = sixh_path(centre, system, region, issue, k)
                if not p.exists():
                    continue
                vals, days = daily_members(p, region)
                if vals is None or not len(days):
                    continue
                mm = (mo - 1 + k - 1) % 12 + 1
                yy = y + (mo - 1 + k - 1) // 12
                want = np.array([str(d)[:7] == f"{yy}-{mm:02d}" for d in days])
                if want.sum() < 20:                                    # not a month's worth
                    continue
                x = vals[:, want]
                x = x[np.isfinite(x).all(axis=1)]
                if not x.size:
                    continue
                ref = (hc or {}).get(f"m{k}")
                cols.append((x, ref))
                labels.append(f"{MONTHS[mm - 1]} {yy}")
            if not cols:
                continue
            mid = f"{centre}_{system}"
            if months is None:
                months = labels
            per_model[mid] = {
                "raw": {f"p{q}": [round(float(to_unit(np.percentile(x, q), unit)), 2) for x, _ in cols] for q in QS},
                "anom": {f"p{q}": [round(float((np.percentile(x, q) - r) * (9 / 5 if unit == "F" else 1.0)), 2)
                                   if r is not None else None for x, r in cols] for q in QS},
                "mean_raw": [round(float(to_unit(x.mean(), unit)), 2) for x, _ in cols],
                "member_days": [int(x.size) for x, _ in cols],
                "members": int(cols[0][0].shape[0]),
            }
            pool.append((mid, cols))
            if mid not in [m["id"] for m in seen]:
                seen.append({"id": mid, "label": mlabel, "color": colour})
            print(f"  {region} {mlabel}: {cols[0][0].shape[0]} members × {len(cols)} month(s), "
                  f"{per_model[mid]['member_days'][0]} member-days in month 1", flush=True)
        if not per_model:
            continue
        # multi-model: every system weighted equally, whatever its member count
        nmon = min(len(c) for _, c in pool)
        mmm = {"raw": {}, "anom": {}, "mean_raw": [], "member_days": [], "members": 0}
        for q in QS:
            mmm["raw"][f"p{q}"] = [round(float(np.mean([to_unit(np.percentile(c[j][0], q), unit)
                                                        for _, c in pool])), 2) for j in range(nmon)]
            mmm["anom"][f"p{q}"] = [round(float(np.mean([(np.percentile(c[j][0], q) - c[j][1]) * (9 / 5 if unit == "F" else 1.0)
                                                         for _, c in pool if c[j][1] is not None])), 2)
                                    if any(c[j][1] is not None for _, c in pool) else None for j in range(nmon)]
        mmm["mean_raw"] = [round(float(np.mean([to_unit(c[j][0].mean(), unit) for _, c in pool])), 2) for j in range(nmon)]
        mmm["member_days"] = [int(sum(c[j][0].size for _, c in pool)) for j in range(nmon)]
        mmm["members"] = int(sum(c[0][0].shape[0] for _, c in pool))
        per_model["mmm"] = mmm
        doc["regions"][region] = {"label": label, "unit": "°F" if unit == "F" else "°C",
                                  "months": months[:nmon], "values": per_model}
    if not doc["regions"]:
        raise SystemExit("no 6-hourly chunks on disk yet — run c3s_sixh.py")
    doc["models"] = [{"id": "mmm", "label": "Multi-model", "color": "#111111"}] + seen
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT.name}: {len(doc['regions'])} region(s), {len(doc['models'])} model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
