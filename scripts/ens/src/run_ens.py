#!/usr/bin/env python3
"""Build the ensemble-anomaly products for a cycle: fetch each ensemble's median,
then render each (variable, product) as a Day 0–15 lead-slider animation.

    python src/run_ens.py --date 20260603 --run 00 \
        --vars z500 --ensembles gefs --products anom30 \
        --out-root /path/to/assets/ens/anim
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fetch as F
import render as R
import archive as A
from common import VARS, ENSEMBLES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--run", default="00")
    ap.add_argument("--vars", default="z500", help="comma list of z500,t2m")
    ap.add_argument("--ensembles", default="gefs", help="comma list of gefs,ifs,aifs,geps")
    ap.add_argument("--products", default="anom30", help="comma list of anom30,dprev,d48")
    ap.add_argument("--out-root", default="../../assets/ens/anim")
    ap.add_argument("--from-archive", action="store_true",
                    help="reuse the archived median for this init instead of re-fetching (style iteration)")
    a = ap.parse_args()
    init = pd.Timestamp(f"{a.date}T{a.run}:00")
    out = Path(a.out_root)
    vars_ = [v for v in a.vars.split(",") if v in VARS]
    enss = [e for e in a.ensembles.split(",") if e in ENSEMBLES and e in F.FETCHERS]
    prods = [p for p in a.products.split(",") if p in R.PRODUCTS or p in R.CHANGE]

    for var in vars_:
        meds = {}
        if a.from_archive:
            print(f"== {var}: loading {enss} medians from archive ==", flush=True)
            for e in enss:
                m = A.load_median(e, var, init)
                if m is not None:
                    meds[e] = m
        else:
            print(f"== {var}: fetching {enss} ==", flush=True)
            for e in enss:
                try:
                    meds[e] = F.fetch(e, a.date, a.run, var)
                except Exception as ex:                      # noqa: BLE001
                    print(f"  {e} {var}: FAILED ({repr(ex)[:80]})", flush=True)
        if not meds:
            print(f"  no ensembles for {var}; skipping"); continue

        # Archive each run's median FIRST so change products can look back to it later,
        # and re-runs of the same init replace cleanly.
        for e in meds:
            A.archive_run(e, var, init, meds[e])

        for prod in prods:
            adir = out / f"{var}_{prod}"; mani = out / f"{var}_{prod}_manifest.json"
            if prod in R.PRODUCTS:
                R.render_product(meds, var, prod, init, adir, mani)
            else:                                            # change product
                hours_back = R.CHANGE[prod][0]
                fields = {e: ch for e in meds
                          if (ch := A.change_fields(e, var, init, meds[e], hours_back)) is not None}
                if fields:
                    R.render_change(fields, meds, var, prod, init, adir, mani)
                else:
                    print(f"  {prod}: no run {hours_back}h earlier archived yet — skipping", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
