#!/usr/bin/env python3
"""What GEFS's jet of weeks 3 and 4 implies for North America: build_pacjet_implied.py's regression maps with weekly
jet indices as the predictor, for the weeks pacjet_ext.py gives.

Predictor  the two jet indices (pacjet_ref.nc PCs, in sigma) as 7-day means, as GEFS's weekly means are.
Target     the 7-day mean anomaly of 2 m temperature, precipitation and 500 hPa height over North America: of the SAME
           week from that week's two indices ("wk3": GEFS's week 3, where both indices passed on the reforecast), and
           of the week AFTER from those two and the next week's extension index ("wk4": GEFS's week 4, where only the
           extension index passed). ERA5 1991-2020, November to March, anomalies and the out-of-sample test as in
           build_pacjet_implied.py (seasons left out, Benjamini-Hochberg at its ALPHA, r_cv at least its R_MIN).

The Himalayan mountain torque is not a predictor: its 7-day mean added to the indices lowers no target's out-of-sample
error over North America (2 m temperature +0.2 %, precipitation 0.0 %, of the same week).

    python src/build_pacjet_implied_wk.py   -> scripts/mjo/data/reference/pacjet_implied_wk.{npz,json}
    render(...) is called by pacjet_ext.py with each run's members' mean indices.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.stats import benjamini_hochberg
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "lib")]
import build_pacjet_impacts as BI
import build_pacjet_implied as BM

NPZ = BI.REF / "pacjet_implied_wk.npz"
OUT = BI.REF / "pacjet_implied_wk.json"
PANELS = (("t2m", "wk3"), ("prcp", "wk3"), ("t2m", "wk4"))  # drawn; 500 hPa height is contoured on the t2m panels


def main() -> int:
    with xr.open_dataset(BI.REF / "pacjet_ref.nc") as ds:
        ref = ds.load()
    rt = pd.DatetimeIndex(ref.time.values)
    n = len(rt)
    x7 = uniform_filter1d(ref.pc.values.astype(float), 7, axis=0, mode="nearest")
    cold = np.isin(rt.month, BM.SEASON)
    season = np.where(rt.month >= 7, rt.year, rt.year - 1)
    rng = np.random.default_rng(0)
    i = np.arange(n)
    out = {}
    doc = {"period": [BI.Y0, BI.Y1], "months": list(BM.SEASON), "alpha_fdr": BM.ALPHA, "r_min": BM.R_MIN, "skill": {}}
    for var in ("t2m", "prcp", "z500"):
        t, lat, lon, v = BI.load(var)
        if not (t == rt).all():
            raise SystemExit(f"{var}: the days differ from the jet reference's")
        w7 = uniform_filter1d(BI.anomalies(t, v, cold), 7, axis=0, mode="nearest")
        out[f"lat_{var}"], out[f"lon_{var}"] = lat, lon
        for wk, off in (("wk3", 0), ("wk4", 7)):
            idx = np.flatnonzero(cold & (i >= 3) & (i + off + 3 < n))
            idx = idx[np.isin(rt[idx + off].month, BM.SEASON)]
            x = x7[idx] if wk == "wk3" else np.c_[x7[idx], x7[idx + off, 0]]
            b, r, p = BM.skill(x, w7[idx + off].reshape(len(idx), -1).astype(float), season[idx], rng)
            keep = benjamini_hochberg(p, BM.ALPHA) & (r >= BM.R_MIN)
            doc["skill"][f"{var}_{wk}"] = {
                "days": len(idx),
                "r_cv_median": round(float(np.median(r)), 2),
                "r_cv_max": round(float(r.max()), 2),
                "share_kept": round(float(keep.mean()), 3),
            }
            print(f"  {var:5s} {wk}: {doc['skill'][f'{var}_{wk}']}", flush=True)
            if var == "z500" or (var, wk) in PANELS:
                out[f"{var}_{wk}"] = np.where(keep, b, np.nan).reshape(b.shape[0], *v.shape[1:]).astype("float32")
    np.savez_compressed(NPZ, **out)
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"wrote {NPZ} ({NPZ.stat().st_size / 1e3:.0f} kB) and {OUT.name}", flush=True)
    return 0


def render(init: pd.Timestamp, members: int, wk3: dict, wk4: dict, out: Path) -> bool:
    """Draw what the members' mean jet indices of weeks 3 and 4 imply. wk3, wk4: pacjet_ext's rows of those weeks (both
    indices in week 3, the extension index in week 4). False, and nothing drawn, outside SEASON."""
    if wk3["month"] not in BM.SEASON or wk4["month"] not in BM.SEASON:
        return False
    a = [wk3["targets"][k]["mean"] for k in ("pc1", "pc2")]
    xs = {"wk3": np.array(a), "wk4": np.array([*a, wk4["targets"]["pc1"]["mean"]])}
    weeks = {k: (pd.Timestamp(w["from"]), pd.Timestamp(w["to"])) for k, w in (("wk3", wk3), ("wk4", wk4))}
    sub = (
        f"Run of {init:%-d %b %Y} 00 UTC, mean of {members} members. Week 3: extension index {a[0]:+.1f} σ, poleward shift "
        f"index {a[1]:+.1f} σ. Week 4: extension index {xs['wk4'][2]:+.1f} σ"
    ).replace("-", "−")
    note = (
        f"From ERA5 {BI.Y0}–{BI.Y1}, November to March: the week's anomaly that went with a jet like this one (week 4 "
        f"from week 3's indices and its own extension index). Colour only where that link held out of sample (seasons "
        f"left out, false-discovery rate {BM.ALPHA:.0%}, correlation at least {BM.R_MIN}). Lines: 500 hPa height "
        "(every 20 m, dashed = lower)."
    )
    BM.draw(np.load(NPZ), xs, weeks, PANELS, "What the GEFS jet of weeks 3 and 4 implies", sub, note, out)
    return True


if __name__ == "__main__":
    sys.exit(main())
