#!/usr/bin/env python3
"""SSW Risk Index — operational scorer.

Reads the rolling obs cache (data/strat_obs.nc), reconstructs the 19 model
features exactly as fitted (climatology parameters from
scripts/telecon/data/sri_climatology.json), applies the frozen ridge-logistic
models (sri_model_op.json), and renders:

  assets/strat/sri_history.webp   90-day P(SSW in 1-15 d / 16-30 d)
  assets/strat/sri_current.json   latest values for the page

The model is trained on NDJFM only; outside that season scoring is
extrapolation and the chart is not rendered (use --force for dry-runs).

    python scripts/strat/sri_score.py [--force]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TELECON = REPO / "scripts" / "telecon" / "data"
ASSETS = REPO / "assets" / "strat"
A = 6.371e6
KAP = 0.2854


def harm_eval(coefs, doy, nh):
    w = 2 * np.pi * doy / 365.25
    B = np.column_stack([np.ones_like(w)] + [f(n * w) for n in range(1, nh + 1)
                                             for f in (np.cos, np.sin)])
    return B @ np.array(coefs)


def build_features(obs, clim):
    t = pd.DatetimeIndex(obs.time.values)
    doy = t.dayofyear.values
    lat = obs.lat.values
    F = {}

    def deseas(vals, key, nh=2, hkey="harm"):
        c = clim[key]
        return (vals - harm_eval(c[hkey], doy, nh)) / c["std"]

    j60 = np.argmin(np.abs(lat - 60.0))
    u60 = obs.u10.values[:, j60, :].mean(1)
    F["u10_weak"] = pd.Series(-deseas(u60, "u10_60n"), index=t)
    d10 = pd.Series(u60, index=t).diff(10)
    F["u10_decel10"] = -d10 / clim["u10_diff10_std"]

    capm = lat >= 65
    wgt = np.cos(np.deg2rad(lat[capm]))
    z10p = (obs.z10.values[:, capm, :].mean(2) * wgt).sum(1) / wgt.sum()
    z100p = (obs.z100.values[:, capm, :].mean(2) * wgt).sum(1) / wgt.sum()
    F["z10_warm"] = pd.Series(deseas(z10p, "z10_pcap"), index=t).rolling(5, min_periods=1).mean()
    F["z100_pre"] = pd.Series(deseas(z100p, "z100_pcap"), index=t).rolling(5, min_periods=1).mean()

    ri = (lat >= 55) & (lat <= 65)
    ring = obs.z10.values[:, ri, :].mean(1)
    cf = np.fft.rfft(ring, axis=1) / ring.shape[1]
    for k in (1, 2):
        amp = 2 * np.abs(cf[:, k])
        c = clim[f"w{k}_10"]
        F[f"w{k}_10"] = pd.Series((amp - c["mean"]) / c["std"], index=t) \
                          .rolling(5, min_periods=1).mean()

    bandm = (lat >= 45) & (lat <= 75)
    wgtb = np.cos(np.deg2rad(lat[bandm]))
    vt100 = (obs.vt.values[:, -1, :][:, bandm] * wgtb).sum(1) / wgtb.sum()
    vt100a = pd.Series(deseas(vt100, "vt100"), index=t)
    F["vt100_5d"] = vt100a.rolling(5, min_periods=3).mean()
    F["vt100_20d"] = vt100a.rolling(20, min_periods=10).mean()
    vk1 = (obs.vt_k1_100.values[:, bandm] * wgtb).sum(1) / wgtb.sum()
    F["vt_k1_20d"] = pd.Series(deseas(vk1, "vt_k1_100"), index=t) \
                       .rolling(20, min_periods=10).mean()
    vk2 = (obs.vt_k2_100.values[:, bandm] * wgtb).sum(1) / wgtb.sum()
    F["vtk2_5d"] = pd.Series(deseas(vk2, "vt_k2_100"), index=t) \
                     .rolling(5, min_periods=3).mean()
    F["vtk2_burst"] = np.maximum(F["vtk2_5d"] - 1.5, 0)

    # TEM descent @30 hPa (NH cap)
    p = obs.lev.values * 100.0
    fac = (1e5 / p)[None, :, None] ** KAP
    dthdp = np.gradient(obs.Tbar.values * fac, p, axis=1)
    dthdp = np.where(np.abs(dthdp) < 2e-4, -2e-4, dthdp)
    vstar = obs.vbar.values - np.gradient(obs.vt.values * fac / dthdp, p, axis=1)
    cosb = np.cos(np.deg2rad(60)); area = 2 * np.pi * A**2 * (1 - np.sin(np.deg2rad(60)))
    om30 = -(2 * np.pi * A * cosb / area) * np.cumsum(
        vstar[:, :, j60] * np.gradient(p)[None, :], axis=1)[:, 2] * 864.0
    F["descent30_5d"] = pd.Series(deseas(om30, "om30"), index=t) \
                          .rolling(5, min_periods=3).mean()

    # tropospheric z500 boxes + wave-1 interference
    for nm in ("ural", "labrador", "pac_shield"):
        c = clim[nm]
        a0, a1, o0, o1 = c["box"]
        lon = obs.lon.values
        m = np.ix_((lat >= a0) & (lat <= a1), (lon >= o0) & (lon <= o1))
        raw = obs.z500.values[:, m[0][:, 0], :][:, :, m[1][0]].mean(axis=(1, 2))
        an = raw - harm_eval(c["harm3"], doy, 3)
        F[nm] = pd.Series(an / c["std"], index=t).rolling(5, min_periods=1).mean()
    ring5 = obs.z500.values[:, ri, :].mean(1)
    cf5 = np.fft.rfft(ring5, axis=1) / ring5.shape[1]
    re1, im1 = cf5[:, 1].real, cf5[:, 1].imag
    reC = harm_eval(clim["lin1"]["harm3_re"], doy, 3)
    imC = harm_eval(clim["lin1"]["harm3_im"], doy, 3)
    proj = ((re1 - reC) * reC + (im1 - imC) * imC) / np.hypot(reC, imC)
    F["lin1"] = pd.Series(proj / clim["lin1"]["std"], index=t) \
                  .rolling(5, min_periods=1).mean()

    F["w2_burst"] = np.maximum(F["w2_10"] - 1.5, 0)
    F["dual_block"] = np.minimum(F["ural"], F["pac_shield"]).clip(lower=0)
    F["qbo_E"] = pd.Series(-obs.u50_eq.values / clim["qbo_u50_std"], index=t)
    oni = {}
    seas = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7,
            "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    for l in open(TELECON / "oni.txt").read().splitlines()[1:]:
        pr = l.split()
        if len(pr) == 4:
            oni[(int(pr[1]), seas[pr[0]])] = float(pr[3])
    last = max(oni)
    F["oni"] = pd.Series([oni.get((d.year, d.month), oni[last]) for d in t], index=t)
    F["oni_modNino"] = ((F["oni"] >= 0.5) & (F["oni"] < 1.5)).astype(float)
    F["doy_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=t)
    F["doy_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=t)
    return pd.DataFrame(F)


def score(Xf, model):
    out = {}
    for win, m in model["windows"].items():
        z = np.full(len(Xf), m["intercept"])
        for nm in model["features_order"]:
            s = model["standardize"][nm]
            z += m["coef"][nm] * (Xf[nm].values - s["mean"]) / s["sd"]
        out[win] = pd.Series(1 / (1 + np.exp(-z)), index=Xf.index)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="render even outside NDJFM (dry-run)")
    a = ap.parse_args()
    month = pd.Timestamp.utcnow().month
    in_season = month in (11, 12, 1, 2, 3)
    try:                                        # keep ONI current (CPC, monthly)
        import urllib.request
        txt = urllib.request.urlopen(
            "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
            timeout=30).read().decode()
        if "SEAS" in txt.splitlines()[0]:
            (TELECON / "oni.txt").write_text(txt)
    except Exception:
        pass                                    # stale ONI is tolerable for weeks
    obs = xr.open_dataset(HERE / "data" / "strat_obs.nc")
    clim = json.load(open(TELECON / "sri_climatology.json"))
    model = json.load(open(TELECON / "sri_model_op.json"))
    Xf = build_features(obs, clim).dropna()
    P = score(Xf, model)
    cur = {("ssw_" + w): float(s.iloc[-1]) for w, s in P.items()}
    # Gate state + Gate Risk Index (the impact-relevant products)
    gate_now = float(Xf["z100_pre"].iloc[-1])
    gate_open = bool((Xf["z100_pre"].iloc[-7:] >= 1).sum() >= 5)
    cur["gate_z100_sigma"] = gate_now
    cur["gate_open"] = gate_open
    gri = json.load(open(TELECON / "gri_model_op.json"))
    G = score(Xf, gri)
    for w, s in G.items():
        cur["gri_" + w] = float(s.iloc[-1])
    cur["date"] = str(Xf.index[-1].date())
    cur["in_season"] = in_season
    ASSETS.mkdir(parents=True, exist_ok=True)
    json.dump(cur, open(ASSETS / "sri_current.json", "w"), indent=1)
    print("  SRI current:", {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in cur.items()}, flush=True)
    if not (in_season or a.force):
        print("  out of season — chart not rendered", flush=True)
        return
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.plot(P["1-15d"].index, 100 * P["1-15d"].values, color="#b3475f", lw=2.2,
            label="P(major SSW in 1–15 d)")
    ax.plot(P["16-30d"].index, 100 * P["16-30d"].values, color="#e07b39", lw=2.0,
            label="P(major SSW in 16–30 d)")
    for w, c in (("1-15d", "#b3475f"), ("16-30d", "#e07b39")):
        ax.axhline(100 * model["windows"][w]["base_rate"], color=c, lw=0.9, ls=":")
    ax.set_ylabel("probability (%)"); ax.set_ylim(0, None)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.set_title("SSW Risk Index — LOWO-validated (dotted: climatological base rates)",
                 fontsize=11.5, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(ASSETS / "sri_history.webp", dpi=115, bbox_inches="tight")
    print("  wrote sri_history.webp", flush=True)


if __name__ == "__main__":
    main()