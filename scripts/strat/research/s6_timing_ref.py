#!/usr/bin/env python3
"""The "timing" block of scripts/strat/reference/ssw_precursors.json, rebuilt 2026-09-30 on the approved test
(ssw_precursors_2026-09-30.pdf): composites of the 28 MERRA-2 major warmings, each event measured from its own lag
-90..-61 mean, against ELIGIBLE comparison dates (same calendar window +-15 d in another winter, Nov-Mar, outside a
CP07 refractory period, westerly on every one of the event's own L pre-onset days), with the gap-filled u(60N,10hPa)
and Benjamini-Hochberg FDR 10 % over the 131 lags -90..+40 as the colour mask. Called by build_ssw_precursors.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import sswr_common as C
from s1_composites import B, LAGS, candidates, comp_test

SPECS = (("vT100", "100 hPa eddy heat flux, 45–75°N", "MERRA-2", +1),
         ("u10", "Zonal wind at 60°N, 10 hPa", "MERRA-2", -1),
         ("Tcap10", "Polar-cap temperature, 10 hPa (65–90°N)", "MERRA-2", +1),
         ("dTtr_e5", "Tropical cooling rate, 100–50 hPa layer, 0–15°N", "ERA5", -1),
         ("Ttr_e5", "Tropical temperature, 100–50 hPa layer, 0–15°N", "ERA5", -1))


def onset(c, sig, sign):
    """First lag of the first run of >= 5 consecutive FDR-significant days in the expected direction, before +10."""
    ok = sig & (np.sign(c) == sign)
    run = 0
    for i, L in enumerate(LAGS):
        if L > 10:
            break
        run = run + 1 if ok[i] else 0
        if run >= 5:
            return int(LAGS[i - 4])
    return None


def timing_block(keep_lags=(-60, 30)):
    u = C.u10()
    tt = C.ttr()
    M = pd.read_pickle(C.DATA / "metrics_daily.pkl") if (C.DATA / "metrics_daily.pkl").exists() else None
    if M is None:
        import s0_metrics
        s0_metrics.main()
        M = pd.read_pickle(C.DATA / "metrics_daily.pkl")
    t10 = xr.open_dataset(C.SD / "reference" / "strat_history.nc").t10.to_series().astype(float).asfreq("D")
    Z = {"vT100": C.std_anom(M.vT100), "u10": C.std_anom(u), "Tcap10": C.std_anom(t10),
         "dTtr_e5": C.std_anom(tt.diff(10).shift(-5)), "Ttr_e5": C.std_anom(tt)}
    E = C.events()
    run = C.westerly_run(u)
    rev = C.all_reversals()
    refr = C.refractory(u, rev)
    cs = candidates(u, run, refr, E, rev)
    keep = (LAGS >= keep_lags[0]) & (LAGS <= keep_lags[1])
    win = (LAGS >= -30) & (LAGS <= 30)
    series = {}
    for k, lab, src, sign in SPECS:
        r = comp_test(Z[k], E.date, [c["N1"] for c in cs], base=True, rng=np.random.default_rng(20260930))
        sig = C.fdr_bh(r["p"], 0.10)
        c = r["comp"]
        series[k] = {"label": lab, "source": src, "comp": [round(float(v), 3) for v in c[keep]],
                     "p": [round(float(v), 4) for v in r["p"][keep]], "sig": [bool(v) for v in sig[keep]],
                     "onset": onset(c, sig, sign), "peak": int(LAGS[win][np.argmax(sign * c[win])])}
        print(f"{k:8s} onset {series[k]['onset']}, peak {series[k]['peak']}, FDR runs {C.runs_of(sig, LAGS, 1)}")
    return {"lags": LAGS[keep].tolist(), "n_events": int(len(E)), "series": series,
            "events": [str(d.date()) for d in E.date],
            "test": ("composite minus ELIGIBLE comparison dates: for each event, 3,000 Monte-Carlo draws of a date within +-15 d of "
                     "the same calendar day in another MERRA-2 winter, Nov-Mar, outside a CP07 refractory period and westerly at "
                     "60N/10 hPa on every one of the event's own pre-onset westerly days (capped at 120); every date measured from "
                     "its own day -90..-61 mean; u(60N,10hPa) gap-filled from NCEP R1; drawn in colour where significant after "
                     "Benjamini-Hochberg FDR 10 % over lags -90..+40; onset = first lag of a run of >= 5 such days"),
            "study": "~/strat_reports/ssw_precursors_2026-09-30.pdf",
            "forecast": ("leave-one-winter-out logistic models on all 5,898 eligible Nov-Mar days 1980/81-2025/26: no predictor "
                         "(vortex strength, tropical T or its trailing 10-day change, 100 hPa heat flux or its 40-day mean, QBO, "
                         "ENSO, 16 vortex-structure metrics) beats the seasonal climatology for an onset 21-42 days ahead "
                         "(best BSS +0.03, n.s.); for 1-14 days a weak vortex with a strong heat flux gives BSS +0.19, AUC 0.85 "
                         "(winter-block bootstrap, FDR 10 %)")}


if __name__ == "__main__":
    import json
    print(json.dumps({k: v for k, v in timing_block().items() if k != "series"}, indent=1)[:600])
