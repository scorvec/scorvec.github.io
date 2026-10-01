#!/usr/bin/env python3
"""Robustness of Q1/Q2 on a second reanalysis and a longer record: NCEP R1 u(60N,10hPa), CP07 central dates on R1 for
winters 1958/59-2025/26 (R1 ends 2026-03-17), ERA5 tropical layer temperature from 1959. Same nulls (N0 original,
N1 eligible) and the same window claims as s1_composites, in base and anomaly modes.
Output: ~/research/ssw_structure/data/s1b_r1.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import xarray as xr

import sswr_common as C
from s1_composites import CLAIMS, LAGS, comp_test, window_p

sys.path.insert(0, str(C.HOME / "research" / "ssw_tropical_precursors"))
from common import cp07  # noqa: E402  (the 2026-09-27 study's CP07 implementation)


def main():
    d = xr.open_dataset(C.TD / "u60n10_r1.nc")
    u = d.uwnd.to_series().astype(float)
    u.index = pd.DatetimeIndex(u.index).normalize()
    u = u[~u.index.duplicated()].asfreq("D").interpolate(limit=3)
    ev = [e for e in cp07(u) if pd.Timestamp("1958-11-01") <= e <= pd.Timestamp("2026-01-31")]
    E = pd.DataFrame({"date": ev}); E["winter"] = [C.winter_of(x) for x in E.date]
    print(f"R1 CP07 events: {len(E)}")
    tt = C.ttr()
    Z = {"u10": C.std_anom(u), "Ttr": C.std_anom(tt).reindex(u.index), "dTtr": C.std_anom(tt.diff(10).shift(-5)).reindex(u.index)}
    run = C.westerly_run(u)
    refr = C.refractory(u, ev)
    days = u.index
    wd = np.array([(x - pd.Timestamp(C.winter_of(x), 7, 1)).days for x in days])
    win = np.array([C.winter_of(x) for x in days])
    okm = np.isin(days.month, C.WIN_MONTHS)
    rng_ok = (days >= "1958-08-01") & (days <= pd.Timestamp("2026-03-17") - pd.Timedelta(days=41)) & np.isfinite(u.values)
    cs = []
    for e in E.itertuples():
        i = days.get_indexer([e.date])[0]
        L = int(min(run.iloc[i], 120))
        base0 = (win != e.winter) & rng_ok
        cs.append({"N0": days[base0 & (np.abs(wd - wd[i]) <= 10)],
                   "N1": days[base0 & okm & (np.abs(wd - wd[i]) <= 15) & (run.values >= L) & ~refr.values]})
    out = {"n_events": len(E), "events": [str(x.date()) for x in E.date], "claims": {}}
    for null in ("N0", "N1"):
        for mode in ("base", "anom"):
            rows = {}
            for name, (k, w, sg) in CLAIMS.items():
                if k not in Z:
                    continue
                r = comp_test(Z[k], E.date, [c[null] for c in cs], base=(mode == "base"))
                v, p, same = window_p(r, w, sg)
                rows[name] = {"value": round(v, 3), "p": round(p, 4), "same_sign": same}
                if mode == "anom" and k in ("u10", "Ttr"):
                    v0, p0, _ = window_p(r, (-90, -61), sg)
                    rows[f"{k} -90..-61"] = {"value": round(v0, 3), "p": round(p0, 4)}
            sig = C.fdr_bh([r["p"] for r in rows.values()], 0.10)
            for r, s in zip(rows.values(), sig):
                r["fdr10"] = bool(s)
            out["claims"][f"{null}|{mode}"] = rows
            print(null, mode, {k: (v["value"], v["p"]) for k, v in rows.items()}, flush=True)
    (C.DATA / "s1b_r1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
