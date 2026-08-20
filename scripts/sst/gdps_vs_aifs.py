#!/usr/bin/env python3
"""Does ECCC GDPS beat AIFS-ENS on Colombian basin rain, 1-10 days?

Both are scored against the same truth the operational stack calibrates
on - gauge-blended, gauge-corrected IMERG on energy-weighted basins - over
whatever days BOTH models have in the archive, so no model is helped by an
easier sample.

Three questions, in order of what would actually change the pipeline:

  1. Is GDPS more accurate day by day (MAE, correlation)?
  2. Is it less biased? Every model tested here runs wet; AIFS is 1.09 on
     ANTIOQUIA and IFS 1.61.
  3. Does a simple GDPS+AIFS mean beat either alone? A deterministic run
     and an ensemble mean fail differently, so a blend can win even when
     neither member does.

GDPS is deterministic, so it is compared against the AIFS ENSEMBLE MEAN -
the like-for-like central estimate. It cannot supply spread, and that
limit is reported rather than papered over.

    python scripts/sst/gdps_vs_aifs.py [--maxlead 10]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PRIV = Path.home() / "colombia_hydro"
ARCH = PRIV / "raw" / "fcst_rain"
TRUTH = PRIV / "raw" / "imerg_basin_daily.json"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]


def load_truth():
    """{region: {YYYY-MM-DD: mm}} — the cache stores dates as YYYYMMDD while
    the forecast archives use YYYY-MM-DD, so normalise here rather than let
    every lookup miss silently and report 'no overlapping days'."""
    t = json.loads(TRUTH.read_text())
    dates = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" if "-" not in d else d
             for d in t["dates"]]
    return {r: dict(zip(dates, t[r])) for r in ORDER if r in t}


def load_forecasts(model):
    """{(region, valid): (lead, value)} using the ensemble MEAN."""
    out = defaultdict(dict)
    for f in sorted(ARCH.glob(f"{model}_*.json.gz")):
        try:
            d = json.load(gzip.open(f, "rt"))
        except Exception:                              # noqa: BLE001
            continue
        be = d.get("basins_energy") or {}
        valid = d.get("valid") or []
        n = int(d.get("n_members") or 1)
        init = d["init_date"]
        for r, arr in be.items():
            a = np.asarray(arr, float)
            if a.size % n == 0 and a.size // n == len(valid):
                a = a.reshape(n, -1).mean(0)
            elif a.size == len(valid):
                a = a.ravel()
            else:
                continue
            for li, v in enumerate(valid):
                lead = li + 1
                # keep the FRESHEST forecast for each (region, valid, lead)
                out[(r, v, lead)][init] = float(a[li])
    return {k: v[max(v)] for k, v in out.items() if v}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxlead", type=int, default=10)
    a = ap.parse_args(argv)
    truth = load_truth()
    G = load_forecasts("gdps")
    A = load_forecasts("aifs")
    keys = [k for k in set(G) & set(A)
            if k[2] <= a.maxlead and k[1] in truth.get(k[0], {})]
    if not keys:
        print("no overlapping verifiable days yet")
        return 0
    days = sorted({k[1] for k in keys})
    print(f"{len(keys)} matched forecast-days over {len(days)} valid days "
          f"({days[0]} .. {days[-1]}), leads 1-{a.maxlead}\n")

    def score(sel, get):
        f = np.array([get(k) for k in sel])
        o = np.array([truth[k[0]][k[1]] for k in sel])
        m = np.isfinite(f) & np.isfinite(o)
        f, o = f[m], o[m]
        return (len(f), float(np.mean(np.abs(f - o))),
                float(f.mean() / o.mean()) if o.mean() > 0 else np.nan,
                float(np.corrcoef(f, o)[0, 1]) if len(f) > 5 else np.nan)

    print(f"{'basin':11}{'model':8}{'n':>6}{'MAE':>8}{'bias':>7}{'r':>7}")
    print("-" * 47)
    tot = defaultdict(list)
    for r in ORDER:
        sel = [k for k in keys if k[0] == r]
        if len(sel) < 20:
            continue
        for lab, get in (("GDPS", lambda k: G[k]),
                         ("AIFS", lambda k: A[k]),
                         ("blend", lambda k: 0.5 * (G[k] + A[k]))):
            n, mae, bias, rr = score(sel, get)
            tot[lab].append((mae, bias, rr, n))
            print(f"{r if lab=='GDPS' else '':11}{lab:8}{n:6}{mae:8.2f}"
                  f"{bias:7.2f}{rr:7.3f}")
        print()
    print("=" * 47)
    print(f"{'ALL':11}{'':8}{'':6}{'MAE':>8}{'bias':>7}{'r':>7}")
    for lab in ("GDPS", "AIFS", "blend"):
        v = tot[lab]
        if v:
            print(f"{'':11}{lab:8}{'':6}{np.mean([x[0] for x in v]):8.2f}"
                  f"{np.mean([x[1] for x in v]):7.2f}"
                  f"{np.mean([x[2] for x in v]):7.3f}")
    print("\nby lead (all basins pooled):")
    print(f"  {'lead':>5}{'GDPS MAE':>10}{'AIFS MAE':>10}{'blend':>9}{'winner':>9}")
    for L in range(1, a.maxlead + 1):
        sel = [k for k in keys if k[2] == L]
        if len(sel) < 20:
            continue
        g = score(sel, lambda k: G[k])[1]
        s = score(sel, lambda k: A[k])[1]
        b = score(sel, lambda k: 0.5 * (G[k] + A[k]))[1]
        best = min((g, "GDPS"), (s, "AIFS"), (b, "blend"))[1]
        print(f"  {L:5}{g:10.2f}{s:10.2f}{b:9.2f}{best:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
