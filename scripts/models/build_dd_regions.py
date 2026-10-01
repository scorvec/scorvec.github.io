#!/usr/bin/env python3
"""County weights and normals for the short-range degree days (models.py -> dd.json; 2026-10-01, user: "degree days
from the short-range models ... PJM, NYISO, ISO-NE, ERCOT (+ MISO if cheap) ... use the existing ddtracker weighting
conventions").

Reproduces the PRIVATE ~/ddtracker's definitions exactly (read only, laptop) and freezes them into a static file the
Actions job reads (scripts/models/dd_regions.json):
  - regions = ddtracker's state lists (ddt/regions.py POWER): PJM PA NJ MD DE DC VA WV OH KY IN MI IL NC; MISO MN IA WI
    IL IN MI MO AR LA MS ND SD; ERCOT TX; NYISO NY; ISONE MA CT RI VT NH ME. Whole states, so the footprints overlap
    (IL/IN/MI are in both PJM and MISO) exactly as they do in the tracker;
  - points = the Census 2020 county CENTRES OF POPULATION, local day = UTC + round(lon / 15) h per county (standard
    time, ddt/regions.county_frame);
  - weights per county (ddt/regions.county_weight): pop = 2020 population; gas = ACS households heating with utility
    gas; elec = ACS households heating with electricity. ddtracker's rule: gas-weighted for (gas) HDD, population for
    CDD (cooling is nearly all electric); pop HDD is kept too, as the power desks quote it;
  - aggregation = KINK FIRST, WEIGHT AFTER (ddtracker CLAUDE.md, fixed 2026-09-13): degree days per county, then the
    weighted mean;
  - normals = ddtracker's ddt/climo.region() n30 (1991-2020, PRISM): EXPECTED degree days per day of year on both the
    midpoint and the hourly-integral basis, per region and weighting (366 rows; Feb 29 folded onto Feb 28).

    python scripts/models/build_dd_regions.py --ddtracker ~/ddtracker --out scripts/models/dd_regions.json
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REGIONS = ["PJM", "NYISO", "ISONE", "ERCOT", "MISO"]
KINDS = ["pop", "gas", "elec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddtracker", default=str(Path.home() / "ddtracker"))
    ap.add_argument("--out", default="scripts/models/dd_regions.json")
    a = ap.parse_args()
    sys.path.insert(0, a.ddtracker)
    from ddt import climo as CL, regions as RG                  # noqa: E402
    cf = RG.county_frame()
    states = {r: RG.REGIONS[r] for r in REGIONS}
    sel = cf.st.isin(sorted({s for v in states.values() for s in v})).values
    idx = np.flatnonzero(sel)
    pos = {int(k): n for n, k in enumerate(idx)}
    counties = [[round(float(cf.lat.values[k]), 4), round(float(cf.lon.values[k]), 4), int(cf.offset.values[k])]
                for k in idx]
    weights = {}
    for kind in KINDS:
        w = RG.county_weight(kind)
        assert (w.fips.values == cf.fips.values).all(), "county order changed"
        vals = np.nan_to_num(np.asarray(w["w"], float))
        for r in REGIONS:
            m = w.st.isin(states[r]).values & (vals > 0)
            v = vals[m] / vals[m].sum()
            weights.setdefault(r, {})[kind] = [[pos[int(k)], round(float(x), 7)] for k, x in zip(np.flatnonzero(m), v)]
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(366)]
    dates = [d.isoformat() for d in days]
    normals = {}
    for r in REGIONS:
        for kind in KINDS:
            n = CL.region(kind, r, dates, "n30")
            if n is None:
                continue
            normals.setdefault(r, {})[kind] = {k: [round(float(x), 2) for x in n[k]]
                                               for k in ("tmean", "hdd", "cdd", "hdd_hourly", "cdd_hourly") if k in n}
    doc = dict(source="ddtracker conventions (ddt/regions.py, ddt/climo.py), frozen " + dt.date.today().isoformat(),
               regions={r: states[r] for r in REGIONS}, kinds={k: RG.KINDS[k] for k in KINDS},
               county_fields=["lat", "lon", "utc_offset_h"], counties=counties, weights=weights,
               normals_period=CL.PERIODS["n30"], normals_doy="index = day of year - 1 of the date itself (366 rows, as ddt.climo.doy)",
               normals=normals)
    Path(a.out).write_text(json.dumps(doc, separators=(",", ":")))
    print(f"{len(counties)} counties in {REGIONS}; normals for {sorted(normals)} -> {a.out} "
          f"({Path(a.out).stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
