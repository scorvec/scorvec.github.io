#!/usr/bin/env python3
"""SMAP (Soil Moisture Accounting Procedure) rainfall-runoff — daily form
(Lopes, Braga & Conejo 1981; ONS uses an ONS-modified variant SMAP/ONS),
calibrated per SIN basin against ONS ENA (energy-weighted natural flow).

Daily SMAP, three linear reservoirs (soil Rsolo, surface Rsup, ground Rsub):
  Tu   = Rsolo/Str                              (soil wetness fraction)
  Es   = (P - Ai)^2 / (P - Ai + Str - Rsolo)     if P > Ai else 0   (surface runoff)
  Er   = Ep                if (P - Es) > Ep else (P - Es) + (Ep - (P - Es)) * Tu
  Rec  = Crec * Tu * (Rsolo - Capc*Str)         if Rsolo > Capc*Str else 0
  Rsolo' = Rsolo + P - Es - Er - Rec
  Rsup'  = Rsup + Es - Ed,   Ed  = Rsup * (1 - 0.5^(1/K2t))
  Rsub'  = Rsub + Rec - Eb,  Eb  = Rsub * (1 - 0.5^(1/Kkt))
  Q [mm/d] = Ed + Eb   ->  ENA [MWmed] = Q * gain   (gain absorbs area x productivity)
Parameters per basin: Str (mm), K2t (d), Crec (%), Ai (mm), Capc (%), Kkt (d),
gain (MW per mm/d), plus initial states via a 365-d spin-up. Calibration:
differential evolution on NSE of daily ENA over the corrected-IMERG record
(2024-07 -> ), holding out the last 120 days for validation. Ep: Thornthwaite
monthly PET from the local ERA5 t2m layer, per basin centroid.

Outputs: ~/brazil_hydro/out/smap_params.json (+ validation NSE/bias per basin)

    python scripts/sst/smap_ons.py [--basins GRANDE PARANAIBA ...]
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

PRIV = Path.home() / "brazil_hydro"
TRUTH = PRIV / "raw" / "imerg_basin_daily.json"                    # corrected rain
ENA = PRIV / "raw" / "ena_bacia_daily.json.gz"
BASINS_GJ = PRIV / "out" / "brazil_basins.geojson"
OUT = PRIV / "out" / "smap_params.json"
ERA5 = Path.home() / "era5_store" / "wb2_1p5_daily_global" / "t2m"


def smap_run(P, Ep, prm, states=None):
    """Vectorized-in-time daily SMAP. prm = (Str, K2t, Crec, Ai, Capc, Kkt, gain)."""
    Str, K2t, Crec, Ai, Capc, Kkt, gain = prm
    Rsolo, Rsup, Rsub = states if states else (0.5 * Str, 0.0, 0.0)
    kk2 = 1 - 0.5 ** (1.0 / max(K2t, 0.5))
    kkb = 1 - 0.5 ** (1.0 / max(Kkt, 1.0))
    Q = np.empty(len(P))
    for t in range(len(P)):
        p, ep = P[t], Ep[t]
        Tu = Rsolo / Str
        Es = (p - Ai) ** 2 / (p - Ai + Str - Rsolo) if p > Ai else 0.0
        pe = p - Es
        Er = ep if pe > ep else pe + (ep - pe) * Tu
        Rec = (Crec / 100.0) * Tu * (Rsolo - (Capc / 100.0) * Str) \
            if Rsolo > (Capc / 100.0) * Str else 0.0
        Rsolo = min(max(Rsolo + p - Es - Er - Rec, 0.0), Str)
        Ed = Rsup * kk2
        Rsup = Rsup + Es - Ed
        Eb = Rsub * kkb
        Rsub = Rsub + Rec - Eb
        Q[t] = (Ed + Eb) * gain
    return Q, (Rsolo, Rsup, Rsub)


def thornthwaite_pet(dates, tmean_monthly, lat_deg):
    """Daily PET (mm/d) from monthly mean T via Thornthwaite, day-length corrected."""
    T = np.maximum(tmean_monthly, 0.0)                    # 12 values, degC
    I = np.sum((T / 5.0) ** 1.514)
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
    out = np.empty(len(dates))
    lat = np.deg2rad(lat_deg)
    for i, d in enumerate(dates):
        m = d.month - 1
        pet_m = 16.0 * (10.0 * T[m] / max(I, 1e-6)) ** a          # mm/month, 12h days
        doy = d.timetuple().tm_yday
        decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
        ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1))
        N = 24 * ws / np.pi
        out[i] = pet_m / 30.0 * (N / 12.0)
    return out


def basin_centroid(b):
    gj = json.loads(BASINS_GJ.read_text())
    for ft in gj["features"]:
        if ft["properties"]["basin"] == b:
            pts = np.vstack([np.array(p[0]) for p in ft["geometry"]["coordinates"]])
            return float(pts[:, 1].mean()), float(pts[:, 0].mean())
    return -20.0, -48.0


def basin_tmean_monthly(lat, lon):
    import xarray as xr
    files = sorted(ERA5.glob("t2m_*.nc"))[-6:]
    vals = []
    for f in files:
        ds = xr.open_dataset(f)
        v = ds["t2m"].sel(latitude=lat, longitude=lon % 360, method="nearest") - 273.15
        vals.append(v)
    s = xr.concat(vals, dim="time")
    s = s[np.isfinite(s.values)]
    return s.groupby("time.month").mean().values


def nse(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    return 1 - np.sum((sim[m] - obs[m]) ** 2) / np.sum((obs[m] - obs[m].mean()) ** 2)


def calibrate(b, P, Ep, ena, split):
    """DE over the 6 hydrological params; gain solved in closed form (LS)."""
    from scipy.optimize import differential_evolution
    warm = 365
    o = ena[warm:split]
    om = np.isfinite(o)
    def fit_gain(Qmm):
        s = Qmm[warm:split][om]
        return float(np.dot(s, o[om]) / max(np.dot(s, s), 1e-9))
    def loss(prm6):
        Qmm, _ = smap_run(P, Ep, tuple(prm6) + (1.0,))
        g = fit_gain(Qmm)
        return -nse(Qmm[warm:split] * g, o)
    bounds = [(200, 4000), (0.5, 15), (0.1, 60), (0.0, 20), (10, 70), (30, 400)]
    res = differential_evolution(loss, bounds, seed=1, maxiter=80, popsize=14,
                                 tol=1e-5, polish=True, workers=1)
    Qmm, _ = smap_run(P, Ep, tuple(res.x) + (1.0,))
    g = fit_gain(Qmm)
    prm = np.concatenate([res.x, [g]])
    Q, states = smap_run(P, Ep, prm)
    return prm, Q, states


def main() -> int:
    a = sys.argv[1:]
    basins = a[a.index("--basins") + 1:] if "--basins" in a else MAJORS
    tc = json.loads(TRUTH.read_text())
    dates = [datetime.strptime(d, "%Y%m%d") for d in tc["dates"]]
    with gzip.open(ENA, "rt") as f:
        ena_all = json.load(f)
    out = json.loads(OUT.read_text()) if OUT.exists() else {"params": {}}
    for b in basins:
        if b not in tc or b not in ena_all:
            print(f"  {b}: no data"); continue
        P = np.array(tc[b], float)
        P = np.where(np.isfinite(P), P, 0.0)
        ena = np.array([ena_all[b].get(d.strftime("%Y-%m-%d"), [np.nan])[0]
                        for d in dates], float)
        lat, lon = basin_centroid(b)
        Ep = thornthwaite_pet(dates, basin_tmean_monthly(lat, lon), lat)
        split = len(dates) - 120
        # gain prior from mean ratio ENA / (P - Ep) — bounds handled in DE
        prm, Q, states = calibrate(b, P, Ep, ena, split)
        n_cal = nse(Q[365:split], ena[365:split])
        n_val = nse(Q[split:], ena[split:])
        bias_val = float(np.nanmean(Q[split:]) / np.nanmean(ena[split:]) - 1)
        out["params"][b] = {
            "Str": round(prm[0], 1), "K2t": round(prm[1], 2), "Crec": round(prm[2], 2),
            "Ai": round(prm[3], 2), "Capc": round(prm[4], 1), "Kkt": round(prm[5], 1),
            "gain_MW_per_mmday": round(prm[6], 3),
            "states_end": [round(float(s), 2) for s in states],
            "state_date": dates[-1].strftime("%Y-%m-%d"),
            "pet_monthly_mmday": np.round([thornthwaite_pet([datetime(2025, m, 15)],
                                             basin_tmean_monthly(lat, lon), lat)[0]
                                           for m in range(1, 13)], 2).tolist(),
            "nse_cal": round(float(n_cal), 3), "nse_val120": round(float(n_val), 3),
            "bias_val120": round(bias_val, 3),
            "cal_window": f"{dates[365].date()}..{dates[split].date()}"}
        print(f"  {b:16} NSE cal {n_cal:.2f} val {n_val:.2f} bias {bias_val:+.2f} "
              f"| Str {prm[0]:.0f} K2t {prm[1]:.1f} Crec {prm[2]:.1f} Ai {prm[3]:.1f} "
              f"Capc {prm[4]:.0f} Kkt {prm[5]:.0f} gain {prm[6]:.2f}", flush=True)
        OUT.write_text(json.dumps(out, indent=1))
    out["generated"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    out["note"] = ("daily SMAP (LBC81) at basin scale on gauge-corrected IMERG; "
                   "target = ONS ENA MWmed; Thornthwaite PET from ERA5 t2m; DE on NSE "
                   "with 365-d spin-up; last 120 d held out")
    OUT.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
