#!/usr/bin/env python3
"""Per-basin / per-season rain->ENA model selection from the benchmark.

Reads out/rain_inflow_benchmark.json (nse7 per family per fold) and picks,
per basin, the family for the DRY season (May-Oct, judged on F1_dry2026)
and the WET season (Nov-Apr, judged on F2_wet2025_26). Rules:
  - eligible = nse7 > 0.10 in the fold; else fall back to the family with
    the best r (shape) among {kernel_v3, cascade, hybrid} — level is then
    anchored to observations by the fan/anchor logic anyway;
  - PARANA (routed mainstem) is forced to 'kernel_v3' with a note (no rain
    model reproduces it; anchor + persistence do the work).
Also refits the winning statistical families on the FULL record and stores
their coefficients so deck_emulator can evaluate them (SMAP params already
live in smap_params.json).

Output: ~/brazil_hydro/out/ena_selector.json
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brazil_model import MAJORS                                     # noqa: E402
from rain_inflow_model import ema                                   # noqa: E402
from smap_ons import smap_run, thornthwaite_pet, basin_centroid, basin_tmean_monthly  # noqa: E402
from rain_inflow_benchmark import ear_anom, nnls_fit, CAS, TAUS      # noqa: E402

PRIV = Path.home() / "brazil_hydro"
BENCH = PRIV / "out" / "rain_inflow_benchmark.json"
TRUTH = PRIV / "raw" / "imerg_basin_daily.json"
ENA = PRIV / "raw" / "ena_bacia_daily.json.gz"
SMAP = PRIV / "out" / "smap_params.json"
OUT = PRIV / "out" / "ena_selector.json"
STAT = ["kernel_v3", "cascade", "hybrid"]
FOLD = {"dry": "F1_dry2026", "wet": "F2_wet2025_26"}


def pick(R, season):
    if season not in FOLD:
        return "kernel_v3", "default"
    F = R[FOLD[season]]
    cands = {m: F[m]["nse7"] for m in ("kernel_v3", "cascade", "smap_nse", "smap_kge", "hybrid")}
    best = max(cands, key=cands.get)
    if cands[best] > 0.10:
        return best, f"nse7={cands[best]:+.2f}"
    # shape fallback
    best = max(STAT, key=lambda m: F[m]["r"] if np.isfinite(F[m]["r"]) else -1)
    return best, f"fallback-by-r={F[best]['r']:.2f}"


def main() -> int:
    bench = json.loads(BENCH.read_text())["basins"]
    tc = json.loads(TRUTH.read_text())
    dates = [datetime.strptime(d, "%Y%m%d") for d in tc["dates"]]
    n = len(dates)
    with gzip.open(ENA, "rt") as f:
        ena_all = json.load(f)
    smap = json.loads(SMAP.read_text())["params"]
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in dates])
    th = 2 * np.pi * doy / 365
    season = np.column_stack([np.sin(th), np.cos(th)])
    out = {"generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
           "season_rule": "dry=May-Oct (F1), wet=Nov-Apr (F2); eligible nse7>0.10 else best-r statistical",
           "basins": {}}
    for b in MAJORS:
        if b not in bench or b not in tc:
            continue
        sel = {}
        for s in ("dry", "wet"):
            m, why = pick(bench[b], s)
            if b == "PARANA":
                m, why = "kernel_v3", "forced: routed mainstem"
            sel[s] = {"model": m, "why": why}
        # refit statistical families on full record (coefficients for the emulator)
        P = np.nan_to_num(np.array(tc[b], float))
        y = np.array([ena_all[b].get(d.strftime("%Y-%m-%d"), [np.nan])[0] for d in dates], float)
        S = ear_anom(b, dates)
        m_ok = np.isfinite(y) & (np.arange(n) >= 60)
        fits = {}
        # kernel_v3
        best = None
        Pm = P - P.mean()
        for tau in TAUS:
            k = ema(Pm, tau); ks = ema(Pm, 180)
            for lag in range(0, 6):
                kl = np.roll(k, lag); kl[:lag] = np.nan
                ksl = np.roll(ks, lag); ksl[:lag] = np.nan
                X = np.column_stack([kl, ksl, S, season])
                m = m_ok & np.all(np.isfinite(X), axis=1)
                A = np.column_stack([np.ones(m.sum()), X[m]])
                beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
                sc = 1 - np.sum((A @ beta - y[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)
                if best is None or sc > best[0]:
                    best = (sc, tau, lag, beta)
        fits["kernel_v3"] = {"tau": best[1], "lag": best[2], "coefs": [float(v) for v in best[3]],
                             "rain_mean": float(P.mean()), "nse_in": round(float(best[0]), 3)}
        # cascade
        E = np.column_stack([ema(P, t) for t in CAS])
        Xc = np.column_stack([E, season, -season, S, -S])
        coef = nnls_fit(Xc, y, m_ok)
        fits["cascade"] = {"taus": CAS, "coefs": [float(v) for v in coef]}
        # hybrid needs smap runoff
        if b in smap:
            p = smap[b]
            lat, lon = basin_centroid(b)
            Ep = thornthwaite_pet(dates, basin_tmean_monthly(lat, lon), lat)
            prm = (p["Str"], p["K2t"], p["Crec"], p["Ai"], p["Capc"], p["Kkt"], p["gain_MW_per_mmday"])
            Q, _ = smap_run(P, Ep, prm)
            Xh = np.column_stack([Q, E, season, -season, S, -S])
            coef = nnls_fit(Xh, y, m_ok)
            fits["hybrid"] = {"taus": CAS, "coefs": [float(v) for v in coef]}
        out["basins"][b] = {"select": sel, "fits": fits,
                            "ear_anom_now": float(S[-1]), "rain_mean": float(P.mean())}
        print(f"  {b:16} dry→{sel['dry']['model']:10} ({sel['dry']['why']})   "
              f"wet→{sel['wet']['model']:10} ({sel['wet']['why']})")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
