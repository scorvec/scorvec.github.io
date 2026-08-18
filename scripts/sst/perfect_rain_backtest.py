#!/usr/bin/env python3
"""Potential-skill study: what if the 15-day rain forecast were near-perfect?

Separates the two error sources in the daily inflow fan.  The delta model
is driven with OBSERVED IMERG rain as though it had been forecast
perfectly, which is the ceiling the inflow model could ever reach, and
then again with observed rain DEGRADED to the error statistics the live
AIFS-ENS tracker actually measures.  The gap between those two curves is
what better rain forecasts could buy; the gap between the degraded curve
and persistence is what is achievable today.

Rain here is corrected IMERG on the energy-weighted basin masks, built
straight from the backfilled daily cache.  Pre-2024 days carry no daily
gauge blend (the IDEAM station archive starts in 2024), so the historical
arm is corrected-satellite — a real seam, reported rather than hidden.

ENSO enters as monthly ONI held daily; the daily RONI series only starts
in 2025 and cannot reach these periods.

Every period is scored with the model FIT OUTSIDE IT (plus a 90-day
embargo), so the event is never in its own training set.

    python scripts/sst/perfect_rain_backtest.py [--periods 2009-06:2010-06,...]

Outputs:
  ~/colombia_hydro/out/perfect_rain_backtest.json
  ~/colombia_hydro/site/perfect_rain_backtest.webp
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                          # noqa: E402
from hydro_region_rain import (region_weights_energy, gauge_correction,  # noqa: E402
                               gauge_blend_field)
from build_imerg_clim import OUT as CLIM_NC, eval_clim             # noqa: E402
import inflow_delta_model as M                                     # noqa: E402

REPO = HERE.parent.parent
PRIV = Path.home() / "colombia_hydro"
INFLOW_JSON = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
NINO_JSON = REPO / "assets" / "sst" / "data" / "nino_history.json"
CACHE = PRIV / "raw" / "basin_rain_long.npz"
OUT_JSON = PRIV / "out" / "perfect_rain_backtest.json"
OUT_PNG = PRIV / "site" / "perfect_rain_backtest.webp"
ORDER = M.ORDER
MAX_LEAD = 15
EMBARGO = 90
NMEM = 30                     # perturbed rain realisations, degraded arm
# measured AIFS-ENS corrected error, mm/day, from aifs_tracker (walk-forward)
AIFS_MAE = {(1, 3): 1.67, (4, 7): 1.69, (8, 15): 2.37}


def build_rain_long():
    """Basin rain + climatology over every cached IMERG day."""
    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if int(z["nfiles"]) == len(files):
            return (z["dates"], z["rain"].item(), z["clim"].item())
    import xarray as xr
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    F = gauge_correction(lons, lats)
    W = region_weights_energy(lons, lats, ORDER)
    coef = xr.open_dataset(CLIM_NC)["coef"].values
    climcache = {}
    dates, rain, clim = [], {r: [] for r in ORDER}, {r: [] for r in ORDER}
    for n, f in enumerate(files):
        g = gauge_blend_field(np.load(f) * F, f.stem, lons, lats)
        doy = min(datetime.strptime(f.stem, "%Y%m%d").timetuple().tm_yday, 365)
        if doy not in climcache:
            climcache[doy] = eval_clim(coef, doy) * F
        c = climcache[doy]
        dates.append(f"{f.stem[:4]}-{f.stem[4:6]}-{f.stem[6:8]}")
        for r in ORDER:
            rain[r].append(float((g * W[r]).sum()))
            clim[r].append(float((c * W[r]).sum()))
        if n % 500 == 0:
            print(f"  rain {n}/{len(files)}", flush=True)
    dates = np.array(dates, dtype="datetime64[D]")
    rain = {r: np.asarray(v) for r, v in rain.items()}
    clim = {r: np.asarray(v) for r, v in clim.items()}
    np.savez_compressed(CACHE, dates=dates, rain=np.array(rain, dtype=object),
                        clim=np.array(clim, dtype=object), nfiles=len(files))
    return dates, rain, clim


def load_all():
    rd, rain, clim = build_rain_long()
    inf = json.loads(INFLOW_JSON.read_text())["full_pct_of_norm"]
    idl = np.array(inf["dates"], dtype="datetime64[D]")
    common, ri, ii = np.intersect1d(rd, idl, return_indices=True)
    anom = {r: (rain[r] - clim[r])[ri] for r in ORDER}
    absr = {r: rain[r][ri] for r in ORDER}
    climr = {r: clim[r][ri] for r in ORDER}
    y = {}
    for r in ORDER:
        v = np.asarray(inf[r], float)[ii]
        v[v == 0] = np.nan
        y[r] = v
    nh = json.loads(NINO_JSON.read_text())
    om = dict(zip(nh["months"], nh["series"]["oni"]["anom"]))
    oni = np.array([om.get(str(x)[:7], np.nan) for x in common], float)
    for i in range(1, len(oni)):
        if not np.isfinite(oni[i]):
            oni[i] = oni[i - 1]
    for i in range(len(oni) - 2, -1, -1):
        if not np.isfinite(oni[i]):
            oni[i] = oni[i + 1]
    from xm_storage import pct_anomaly_series
    sd, sa = pct_anomaly_series()
    sd = np.array([str(x) for x in sd], dtype="datetime64[D]")
    stor = {}
    for r in ORDER:
        v = np.full(len(common), np.nan)
        _, c2, s2 = np.intersect1d(common, sd, return_indices=True)
        v[c2] = np.asarray(sa[r], float)[s2]
        for i in range(1, len(v)):
            if not np.isfinite(v[i]):
                v[i] = v[i - 1]
        stor[r] = np.nan_to_num(v)
    return {"dates": common, "rain": anom, "rain_abs": absr, "clim": climr,
            "y": y, "roni": oni, "stor": stor}


def band_mae(lead):
    for (a, b), v in AIFS_MAE.items():
        if a <= lead <= b:
            return v
    return AIFS_MAE[(8, 15)]


def degrade(x_true, clim_mean, lead_of, rng):
    """Observed rain perturbed to the AIFS-ENS measured error level.

    Multiplicative lognormal noise, sigma solved per lead band so the
    expected |error| matches the tracker's corrected MAE for that band.
    Rain error is multiplicative in character — wet days carry most of the
    absolute error — so a flat additive perturbation would be wrong.
    """
    out = x_true.copy()
    lvl = np.maximum(x_true + clim_mean, 0.1)      # back to a rain-like level
    for i, lead in lead_of.items():
        if lead is None:
            continue
        target = band_mae(lead)
        sig = min(2.0, target / max(0.7979 * lvl[i], 0.05))
        z = rng.normal()
        fac = np.exp(sig * z - 0.5 * sig * sig)
        out[i] = (lvl[i] * fac) - clim_mean
    return out


def fwd(beta, off, sh, kf_h, ks_h, rain_h, roni, stor, tau, lag, i0, y0,
        xf=None):
    """Step the recursion forward 15 days from the EMA state at i0.

    simulate() rebuilds both EMAs over the entire series on every call,
    which is fine for one path but hopeless for 30 perturbed realisations
    per start day across 26 years.  History is identical across
    realisations, so the kernels are precomputed once and only the
    forecast window is stepped here.  `xf` is the forecast-window rain
    anomaly (length MAX_LEAD); None means use the observed values, i.e.
    the perfect arm.
    """
    a_f, a_s = 1.0 / tau, 1.0 / M.TAU_SLOW
    n = len(rain_h)
    kf = dict(); ks = dict(); rl = dict()
    for j in range(MAX_LEAD):
        i = i0 + 1 + j
        if i >= n:
            break
        x = rain_h[i] if xf is None else xf[j]
        kf[i] = (1 - a_f) * (kf.get(i - 1, kf_h[i - 1])) + a_f * x
        ks[i] = (1 - a_s) * (ks.get(i - 1, ks_h[i - 1])) + a_s * x
        rl[i] = x
    getk = lambda D, H, i: D[i] if i in D else (H[i] if 0 <= i < n else np.nan)
    out = np.full(MAX_LEAD, np.nan)
    yp = y0
    for j in range(MAX_LEAD):
        i = i0 + 1 + j
        if i >= n:
            break
        il = i - lag
        f = [yp - 100.0, getk(rl, rain_h, il), getk(kf, kf_h, il),
             getk(ks, ks_h, il), roni[i], stor[i]]
        if not np.all(np.isfinite(f)):
            break
        yp = float(np.clip(yp + beta[0] + np.dot(beta[1:], f), *M.CLIP))
        out[j] = yp + off[j] + sh[j] * 0.0        # calibration applied by caller
        out[j] = yp
    return out


def run_period(d, basin, a0, a1, rng, arm="perfect"):
    """Fit outside the window (+embargo), forecast every start day in it."""
    dates = d["dates"]
    y, rain = d["y"][basin], d["rain"][basin]
    roni, stor = d["roni"], d["stor"][basin]
    n = len(y)
    ym = np.array([str(x)[:7] for x in dates])
    inper = (ym >= a0) & (ym <= a1)
    if inper.sum() < 30:
        return None
    hi = np.where(inper)[0]
    tr = ~inper
    tr[max(0, hi[0] - EMBARGO):min(n, hi[-1] + EMBARGO + 1)] = False
    if tr.sum() < 400:
        return None
    tau, lag = M.select_hyper(rain, y, roni, stor, tr, False)
    X, dy = M.design(rain, y, roni, stor, tau, lag)
    m = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
    beta = M.fit(X, dy, m)
    sh, off = M.shrinkage(beta, rain, y, roni, stor, tau, lag, tr)
    kf_h = M.ema(rain, tau)
    ks_h = M.ema(rain, M.TAU_SLOW)
    rabs, rclim = d["rain_abs"][basin], d["clim"][basin]
    nrl = NMEM if arm == "degraded" else 1
    mae_check = []

    def perturb(true_abs, scale=None):
        """Observed rain -> a forecast with the tracker's measured error.

        Multiplicative lognormal for the wet-day error, which scales with
        the amount, plus a calibration rescale of the residual so the
        realised MAE actually hits the target per band.  Pure
        multiplicative noise cannot: on a near-dry day it can only produce
        a near-zero absolute error, whereas a real forecast's error there
        is a false alarm, i.e. additive.  Measured undershoot before this
        was ~25% (1.1-1.4 against a 1.67 target at d1-3).
        """
        lvl = np.maximum(true_abs, 0.15)
        sig = np.array([min(2.5, band_mae(j + 1) / max(0.7979 * lvl[j], 0.05))
                        for j in range(MAX_LEAD)])
        z = rng.normal(size=MAX_LEAD)
        err = lvl * (np.exp(sig * z - 0.5 * sig ** 2) - 1.0)
        if scale is not None:
            err = err * scale
        return np.maximum(true_abs + err, 0.0)

    scale = None
    if arm == "degraded":                       # calibrate on a sample first
        acc = np.zeros(MAX_LEAD); cnt = 0
        for i0 in hi[::5]:
            if i0 + MAX_LEAD >= n:
                continue
            ta = np.maximum(rabs[i0 + 1: i0 + 1 + MAX_LEAD], 0.0)
            for _ in range(4):
                acc += np.abs(perturb(ta) - ta); cnt += 1
        if cnt:
            realized = acc / cnt
            tgt = np.array([band_mae(j + 1) for j in range(MAX_LEAD)])
            scale = np.clip(tgt / np.maximum(realized, 1e-3), 0.5, 6.0)

    P = {h: [] for h in range(1, MAX_LEAD + 1)}
    O = {h: [] for h in range(1, MAX_LEAD + 1)}
    Q = {h: [] for h in range(1, MAX_LEAD + 1)}
    for i0 in hi:
        if not np.isfinite(y[i0]) or i0 + MAX_LEAD >= n:
            continue
        sl = slice(i0 + 1, i0 + 1 + MAX_LEAD)
        true_abs = np.maximum(rabs[sl], 0.0)
        cl = rclim[sl]
        paths = []
        for k in range(nrl):
            if arm == "perfect":
                xf = None
            elif arm == "climo":
                xf = np.zeros(MAX_LEAD)          # future rain = climatology
            else:
                fabs = perturb(true_abs, scale)
                mae_check.append(np.abs(fabs - true_abs))
                xf = fabs - cl
            sim = fwd(beta, off, sh, kf_h, ks_h, rain, roni, stor,
                      tau, lag, i0, y[i0], xf)
            paths.append([y[i0] + off[j] + sh[j] * (v - y[i0])
                          if np.isfinite(v) else np.nan
                          for j, v in enumerate(sim)])
        paths = np.asarray(paths, float)
        for j in range(MAX_LEAD):
            i = i0 + 1 + j
            col = paths[:, j]
            col = col[np.isfinite(col)]
            if not len(col) or not np.isfinite(y[i]):
                continue
            P[j + 1].append(float(np.mean(col)))
            O[j + 1].append(float(y[i]))
            Q[j + 1].append(float(y[i0]))

    res = {"tau_days": int(tau), "lag_days": int(lag), "leads": {}}
    if mae_check:
        mc = np.asarray(mae_check)
        res["implied_mae_by_band"] = {
            f"d{a_}-{b_}": round(float(np.mean(mc[:, a_ - 1:b_])), 2)
            for (a_, b_) in AIFS_MAE}
    for h in range(1, MAX_LEAD + 1):
        if len(P[h]) < 15:
            continue
        p, o, q = map(np.asarray, (P[h], O[h], Q[h]))
        rm = float(np.sqrt(np.mean((p - o) ** 2)))
        rp = float(np.sqrt(np.mean((q - o) ** 2)))
        rc = float(np.sqrt(np.mean((o - np.mean(o)) ** 2)))
        res["leads"][h] = {
            "n": int(len(p)),
            "r": round(float(np.corrcoef(p, o)[0, 1]), 3),
            "rmse": round(rm, 2), "rmse_persistence": round(rp, 2),
            "skill_vs_persistence": round(1 - rm / rp, 3),
            "skill_vs_climatology": round(1 - rm / rc, 3)}
    return res


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, INK = "#13273d", "#1a2733"
    pers = [p for p in out["periods"] if out["periods"][p].get("perfect")]
    if not pers:
        return
    fig = plt.figure(figsize=(5.2 * len(pers) + 1.0, 8.6))
    hd = fig.add_axes([0, 0.935, 1, 0.065]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.012, 0.62, "IS THE RAIN FORECAST THE BINDING CONSTRAINT?",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    hd.text(0.012, 0.2, "the delta model driven by perfect rain, by rain "
            "degraded to the AIFS-ENS measured error, and by climatology",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")
    arms = (("perfect", "#1f4e8c", "perfect rain (ceiling)"),
            ("degraded", "#e08214", "AIFS-quality rain"),
            ("climo", "#c62828", "climatology (no forecast)"))
    for i, per in enumerate(pers):
        ax = fig.add_axes([0.06 + i * (0.92 / len(pers)), 0.565,
                           0.92 / len(pers) - 0.075, 0.33])
        for arm, c, lab in arms:
            nat = out["periods"][per].get(arm, {}).get("national_energy_weighted")
            if not nat:
                continue
            hs = sorted(int(h) for h in nat)
            ax.plot(hs, [nat[str(h)]["skill_vs_persistence"] if str(h) in nat
                         else nat[h]["skill_vs_persistence"] for h in hs],
                    color=c, lw=2.0, marker="o", ms=3.5, label=lab)
        ax.axhline(0, color="0.45", lw=0.9)
        ax.set_xlabel("lead, days", fontsize=9)
        if i == 0:
            ax.set_ylabel("national RMSE skill vs persistence", fontsize=9)
            ax.legend(fontsize=8, loc="lower right")
        ax.set_title(per, fontsize=10.5, fontweight="bold", loc="left", color=INK)
        ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8)

    ax2 = fig.add_axes([0.06, 0.09, 0.88, 0.36])
    labs, gap_f, gap_c = [], [], []
    for per in pers:
        P = out["periods"][per].get("perfect", {}).get("national_energy_weighted", {})
        D = out["periods"][per].get("degraded", {}).get("national_energy_weighted", {})
        C = out["periods"][per].get("climo", {}).get("national_energy_weighted", {})
        g = lambda D_, h: (D_.get(str(h), D_.get(h, {})) or {}).get("skill_vs_persistence")
        for h in (1, 3, 7, 15):
            if g(P, h) is None or g(C, h) is None:
                continue
            labs.append(f"{per.split(':')[0]}\nh{h}")
            gap_c.append(g(D, h) - g(C, h) if g(D, h) is not None else np.nan)
            gap_f.append(g(P, h) - g(D, h) if g(D, h) is not None else np.nan)
    x = np.arange(len(labs))
    ax2.bar(x, gap_c, 0.62, color="#e08214",
            label="value of having a rain forecast at all (AIFS − climatology)")
    ax2.bar(x, gap_f, 0.62, bottom=gap_c, color="#1f4e8c",
            label="extra from making that forecast PERFECT")
    ax2.set_xticks(x); ax2.set_xticklabels(labs, fontsize=7.4)
    ax2.set_ylabel("skill contribution", fontsize=9)
    ax2.set_title("Where the skill comes from — the blue slivers are what "
                  "better rain forecasts could still buy",
                  fontsize=11, fontweight="bold", loc="left", color=INK)
    ax2.legend(fontsize=8.5); ax2.grid(lw=0.25, alpha=0.5, axis="y")
    ax2.tick_params(labelsize=8)
    fig.text(0.06, 0.022, "model fitted OUTSIDE each window with a 90-day "
             "embargo · degraded arm calibrated so its realised MAE matches the "
             "tracker's measured corrected AIFS error per lead band · pre-2024 "
             "rain is corrected-satellite (no daily gauge blend)",
             fontsize=7.6, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=118); plt.close(fig)
    print(f"wrote {OUT_PNG}")


NATIONAL_W = {"ANTIOQUIA": 0.490, "CENTRO": 0.209, "ORIENTE": 0.166,
              "VALLE": 0.071, "CALDAS": 0.042, "CARIBE": 0.022}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", default="2009-06:2010-06,2007-06:2008-06,"
                                         "2002-06:2003-06")
    ap.add_argument("--arms", default="perfect,degraded,climo")
    a = ap.parse_args()
    rng = np.random.default_rng(20260818)
    d = load_all()
    print(f"rain+inflow overlap {d['dates'][0]}..{d['dates'][-1]} "
          f"({len(d['dates'])} d)", flush=True)

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "window": f"{d['dates'][0]}..{d['dates'][-1]}",
           "note": "perfect arm = observed IMERG used as the 15-day forecast "
                   "(ceiling); degraded arm = observed rain perturbed to the "
                   "AIFS-ENS measured corrected MAE by lead band",
           "aifs_mae_used": {f"d{a_}-{b_}": v for (a_, b_), v in AIFS_MAE.items()},
           "gauge_seam": "pre-2024 days are corrected-satellite (no daily "
                         "IDEAM blend); post-2024-07 days are gauge-blended",
           "periods": {}}
    for per in a.periods.split(","):
        a0, a1 = per.split(":")
        ym = np.array([str(x)[:7] for x in d["dates"]])
        if not ((ym >= a0) & (ym <= a1)).any():
            print(f"{per}: no data yet — skipped", flush=True)
            continue
        out["periods"][per] = {}
        for arm in a.arms.split(","):
            per_basin = {}
            for b in ORDER:
                r = run_period(d, b, a0, a1, rng, arm)
                if r:
                    per_basin[b] = r
            if not per_basin:
                continue
            nat = {}
            for h in range(1, MAX_LEAD + 1):
                ss = [(NATIONAL_W[b], per_basin[b]["leads"][h])
                      for b in per_basin if h in per_basin[b]["leads"]]
                if len(ss) < 4:
                    continue
                w = sum(x[0] for x in ss)
                nat[h] = {k: round(sum(x[0] * x[1][k] for x in ss) / w, 3)
                          for k in ("skill_vs_persistence",
                                    "skill_vs_climatology", "r")}
            out["periods"][per][arm] = {"basins": per_basin,
                                        "national_energy_weighted": nat}
            imp = {b: per_basin[b].get("implied_mae_by_band")
                   for b in per_basin if per_basin[b].get("implied_mae_by_band")}
            if imp:
                out["periods"][per][arm]["implied_mae_by_band"] = imp
            sk = " ".join(f"h{h}:{nat[h]['skill_vs_persistence']:+.2f}"
                          for h in (1, 3, 7, 10, 15) if h in nat)
            print(f"{per} {arm:9} national skill vs persistence  {sk}", flush=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    try:
        figure(out)
    except Exception as e:                       # noqa: BLE001
        print(f"figure failed: {repr(e)[:140]}")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


