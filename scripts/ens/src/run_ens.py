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
from common import VARS, ENSEMBLES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--run", default="00")
    ap.add_argument("--vars", default="z500", help="comma list of z500,t2m")
    ap.add_argument("--ensembles", default="gefs", help="comma list of gefs,ifs,aifs,geps")
    ap.add_argument("--products", default="anom30", help="comma list of anom30,anom10")
    ap.add_argument("--out-root", default="../../assets/ens/anim")
    a = ap.parse_args()
    init = pd.Timestamp(f"{a.date}T{a.run}:00")
    out = Path(a.out_root)
    vars_ = [v for v in a.vars.split(",") if v in VARS]
    enss = [e for e in a.ensembles.split(",") if e in ENSEMBLES and e in F.FETCHERS]
    prods = [p for p in a.products.split(",") if p in R.PRODUCTS]

    for var in vars_:
        print(f"== {var}: fetching {enss} ==", flush=True)
        meds = {}
        for e in enss:
            try:
                meds[e] = F.fetch(e, a.date, a.run, var)
            except Exception as ex:                          # noqa: BLE001
                print(f"  {e} {var}: FAILED ({repr(ex)[:80]})", flush=True)
        if not meds:
            print(f"  no ensembles fetched for {var}; skipping"); continue
        for prod in prods:
            R.render_product(meds, var, prod, init,
                             out / f"{var}_{prod}", out / f"{var}_{prod}_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
