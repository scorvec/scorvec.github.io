#!/usr/bin/env python3
"""Lag-regression + event-composite of ERA5 jet (u250) and teleconnection (z500)
on AAM surface-torque indices — quantifies what statistically follows a torque event.

Reads the caches from build_torque_teleconn_data.py and, per predictor × season
(NDJFM, JJA) × field (u250, z500) × lag τ∈[-3..+12 d]:
  • REGRESSION  — response per +1σ torque (slope of field_anom(t+τ) on the
    standardized index(t)); significance from an autocorrelation-adjusted
    effective sample size (Bretherton et al. 1999) → stipple p<0.05.
  • COMPOSITE   — mean field_anom(t+τ) after +1σ events minus after −1σ events;
    effective-N Welch significance → stipple.
Plus a Himalaya↔Rockies "mountain-torque relay" cross-correlation diagnostic.

Renders cartopy panels (rows = lags, cols = u250 | z500) to assets/sst/aam_teleconn/.
Loads in seconds from cache; freely re-runnable.

    python src/torque_teleconnection.py --y0 1979 --y1 2023
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

CACHE = Path.home() / "mjo" / "era5_cache"
OUT = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "sst" / "aam_teleconn"

INDICES = ["mtn_himalaya", "mtn_rockies", "mtn_andes", "mtn_global", "fric_global"]
LABELS = {"mtn_himalaya": "Himalaya/Tibet mtn torque", "mtn_rockies": "Rockies mtn torque",
          "mtn_andes": "Andes mtn torque", "mtn_global": "Global mtn torque",
          "fric_global": "Global friction torque"}
# Physical-chain captions (torque → AAM → jet → teleconnection) shown on each figure.
DESCRIPTIONS = {
    "mtn_himalaya": "+ve Himalaya/Tibet mountain torque injects westerly momentum over Asia (↑ AAM) → "
                    "accelerates the East-Asian / N-Pacific jet (lag 0–5 d) → downstream Rossby-wave "
                    "train → PNA-like height response over N. America (lag ~7–14 d)",
    "mtn_rockies": "+ve Rockies mountain torque adds westerly momentum over W. N. America (↑ AAM) → "
                   "modulates the N-Pacific jet exit → downstream N. American height pattern "
                   "(often a relay ~a week after a Himalayan event)",
    "mtn_andes": "+ve Andes mountain torque (the dominant global term) adds westerly momentum in the "
                 "SH subtropics (↑ AAM) → SH subtropical-jet response (plotted domain is 20°S–87°N)",
    "mtn_global": "+ve global mountain torque = net westerly torque on the atmosphere → rise in global "
                  "AAM → stronger / poleward-shifted subtropical jets (strongest at lag 0)",
    "fric_global": "+ve global friction torque = westerly surface-stress torque → the slower, "
                   "ENSO/MJO-modulated branch of the AAM budget → broad subtropical-jet response",
    "aam_tendency": "Standardized anomaly of the global AAM tendency dM/dt (= global mountain + "
                    "friction torque; GWD omitted). +ve = global AAM rising faster than normal → "
                    "stronger / poleward-shifted NH subtropical jets; composite shows the lagged "
                    "hemispheric jet & height response.",
    "glaam_nh": "Standardized anomaly of NH relative AAM (cos²φ-weighted 0–87°N integral of the "
                "250-hPa zonal-wind anomaly — a jet-level proxy for the AAM state, not its tendency). "
                "±2σ = anomalously strong / weak NH westerlies; composite shows the jet & height "
                "pattern of a high vs low AAM state.",
}
LABELS["aam_tendency"] = "Global AAM tendency dM/dt"
LABELS["glaam_nh"] = "NH relative AAM (250-hPa proxy)"
SEASONS = {"NDJFM": [11, 12, 1, 2, 3], "JJA": [6, 7, 8]}
LAGS = list(range(-3, 13))           # days; predictor leads for τ>0
LAGS_SHOW = [-2, 0, 2, 4, 6, 8, 10]  # rows in the static panels (t−2 → t+10)
ANIM_LAGS = list(range(-2, 11))      # daily frames for the composite animation (t−2 → t+10)
FIELDS = {"u250": "250-hPa zonal wind", "z500": "500-hPa height"}
FUNITS = {"u250": "m s⁻¹", "z500": "m"}


def harmonic_deseason(arr: np.ndarray, doy: np.ndarray, n_harm: int = 4) -> np.ndarray:
    """Remove mean + n_harm annual harmonics. arr: (ntime, ...) → anomalies, same shape."""
    w = 2 * np.pi * doy / 365.25
    cols = [np.ones_like(w)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    X = np.column_stack(cols)                                   # (ntime, 1+2H)
    shp = arr.shape
    flat = arr.reshape(shp[0], -1).astype("float64")
    ok = np.isfinite(flat).all(axis=0)                          # fit only all-finite cols
    coef = np.full((X.shape[1], flat.shape[1]), np.nan)
    if ok.any():
        coef[:, ok], *_ = np.linalg.lstsq(X, flat[:, ok], rcond=None)
    clim = X @ coef
    return (flat - clim).reshape(shp)


def lag1(x: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation along axis 0 (NaN-aware, per column)."""
    a, b = x[:-1], x[1:]
    a = a - np.nanmean(a, axis=0); b = b - np.nanmean(b, axis=0)
    num = np.nansum(a * b, axis=0)
    den = np.sqrt(np.nansum(a * a, axis=0) * np.nansum(b * b, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(num / den, -0.99, 0.99)


def regress(x: np.ndarray, Y: np.ndarray, a1x: float):
    """Regress columns of Y on scalar predictor x (already +σ-standardized).
    Returns (slope per +1σ, p-value) using effective-N from lag-1 autocorr."""
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=0)
    sxx = (xc * xc).sum()
    b = (xc[:, None] * Yc).sum(axis=0) / sxx                    # response per +1σ
    syy = (Yc * Yc).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xc[:, None] * Yc).sum(axis=0) / np.sqrt(sxx * syy)
    a2 = lag1(Y)
    neff = len(x) * (1 - a1x * a2) / (1 + a1x * a2)
    neff = np.clip(neff, 3, len(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt((neff - 2) / (1 - r ** 2))
    p = 2 * stats.t.sf(np.abs(t), neff - 2)
    return b, p


def composite(Yhi: np.ndarray, Ylo: np.ndarray):
    """Difference of composites (hi − lo) with effective-N Welch significance."""
    def eff(Z):
        n = Z.shape[0]
        a = lag1(Z)
        return np.clip(n * (1 - a) / (1 + a), 2, n)
    mhi, mlo = Yhi.mean(0), Ylo.mean(0)
    vhi, vlo = Yhi.var(0, ddof=1), Ylo.var(0, ddof=1)
    nhi, nlo = eff(Yhi), eff(Ylo)
    se = np.sqrt(vhi / nhi + vlo / nlo)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (mhi - mlo) / se
    dof = (vhi / nhi + vlo / nlo) ** 2 / (
        (vhi / nhi) ** 2 / (nhi - 1) + (vlo / nlo) ** 2 / (nlo - 1))
    p = 2 * stats.t.sf(np.abs(t), np.clip(dof, 1, None))
    return mhi - mlo, p


def _slug(index: str) -> str:
    """File slug: strip a leading mtn_/aam_ prefix only (NOT a substring — glaam_nh
    contains 'aam_' but must stay 'glaam_nh')."""
    for pre in ("mtn_", "aam_"):
        if index.startswith(pre):
            return index[len(pre):]
    return index


def composite_one(Y):
    """One-sample composite: mean(Y) and 2-sided p vs 0, effective-N (autocorr-adjusted).
    For showing the +σ and −σ event composites SEPARATELY (test of antisymmetry)."""
    n = Y.shape[0]
    a = lag1(Y)
    neff = np.clip(n * (1 - a) / (1 + a), 2, n)
    m = Y.mean(0)
    se = Y.std(0, ddof=1) / np.sqrt(neff)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = m / se
    p = 2 * stats.t.sf(np.abs(t), np.clip(neff - 1, 1, None))
    return m, p


def _titleblock(fig, head, sub, index, note):
    """Stacked, wrapped figure header so the explainer never runs off one long line:
    bold title · method line · wrapped torque→AAM→jet chain · small note."""
    fig.suptitle(head, fontsize=13, fontweight="bold", y=0.998)
    fig.text(0.5, 0.966, sub, ha="center", va="top", fontsize=9, color="0.25")
    fig.text(0.5, 0.946, textwrap.fill(DESCRIPTIONS[index], width=85),
             ha="center", va="top", fontsize=9, style="italic")
    fig.text(0.5, 0.906, note, ha="center", va="top", fontsize=8, color="0.45")


def render(maps: dict, lats, lons, index: str, season: str, method: str,
           normalized: bool = False):
    """maps[(field, lag)] = (response_2d, p_2d). Rows=LAGS_SHOW, cols=fields.

    normalized=True → response already divided by the local field σ (dimensionless;
    for regression this is the correlation r). Physical maps cap the colour scale on
    the equatorward band (poleward z500 variance otherwise swamps the signal);
    standardized maps are comparable across latitude, so use the full domain."""
    proj = ccrs.PlateCarree(central_longitude=180)
    nrow, ncol = len(LAGS_SHOW), len(FIELDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6 * ncol, 1.85 * nrow + 0.5),
                             subplot_kw={"projection": proj}, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.0, hspace=0.0, wspace=0.0)
    axes = np.atleast_2d(axes)
    band = np.ones(lats.shape, bool) if normalized else (np.abs(lats) <= 70.0)
    lim = {}
    for f in FIELDS:
        vals = np.concatenate([np.abs(m[0][band][np.isfinite(m[0][band])])
                               for lg in LAGS_SHOW for m in [maps[(f, lg)]]])
        lim[f] = (float(np.nanpercentile(vals, 99.5)) or 1.0) * 1.3   # wide range → weak noise fades
    cunit = ("correlation r" if method == "regression" else "σ units") if normalized else None
    col_cf = {}
    for i, lg in enumerate(LAGS_SHOW):
        for j, f in enumerate(FIELDS):
            ax = axes[i, j]
            resp, p = maps[(f, lg)]
            levels = np.linspace(-lim[f], lim[f], 21)
            col_cf[j] = ax.contourf(lons, lats, resp, levels=levels, cmap="RdBu_r",
                                    extend="both", transform=ccrs.PlateCarree())
            ax.contourf(lons, lats, np.where(p < 0.05, 1.0, np.nan), levels=[0.5, 1.5],
                        colors="none", hatches=["..."], transform=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.4, color="0.3")
            ax.set_extent([-179.9, 179.9, float(lats.min()), float(lats.max())],
                          crs=ccrs.PlateCarree())
            if j == 0:
                ax.text(-0.018, 0.5, ("lag 0 d" if lg == 0 else f"lag {lg:+d} d"), transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=9.5, fontweight="bold", clip_on=False)
            if i == 0:
                ax.set_title(FIELDS[f], fontsize=11)
    for j, f in enumerate(FIELDS):                              # one colorbar per column
        cb = fig.colorbar(col_cf[j], ax=list(axes[:, j]), location="bottom",
                          shrink=0.55, aspect=32, pad=0.008)
        cb.set_label(cunit or FUNITS[f], fontsize=9); cb.ax.tick_params(labelsize=7)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = method + ("_std" if normalized else "")
    fig.savefig(OUT / f"torque_teleconn_{index}_{season}_{tag}.webp", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved torque_teleconn_{index}_{season}_{tag}.webp", flush=True)


def render_abs(maps: dict, lats, lons, index: str, season: str):
    """maps[(field, lag)] = absolute_field_2d. Mean ABSOLUTE u250/z500 composited over
    +1σ torque days at each lag — shows how the actual jet/height config evolves."""
    proj = ccrs.PlateCarree(central_longitude=180)
    nrow, ncol = len(LAGS_SHOW), len(FIELDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 2.35 * nrow + 0.5),
                             subplot_kw={"projection": proj}, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.0, hspace=0.0, wspace=0.0)
    axes = np.atleast_2d(axes)
    EXTENT = [120, 260, 8, 74]                                  # N-Pacific / W-N-America focus
    # u250 = ABSOLUTE jet (sequential; only strong winds coloured, white below the floor so
    # the jet pops; season-aware since the JJA jet is much weaker). z500 = composite height
    # ANOMALY (diverging, symmetric robust auto-scale) — the absolute height is climatology-
    # dominated and flat, the anomaly is what carries the teleconnection.
    u_lev, u_cs, u_lab = ((np.arange(15, 51, 3), [25, 40], "m s⁻¹ (jet ≥15)") if season == "JJA"
                          else (np.arange(30, 86, 5), [50, 70], "m s⁻¹ (jet ≥30)"))
    zv = np.concatenate([np.abs(maps[("z500", lg)][np.isfinite(maps[("z500", lg)])])
                         for lg in LAGS_SHOW])
    zlim = (float(np.nanpercentile(zv, 99)) or 10.0)
    SPEC = {"u250": dict(levels=u_lev, cmap="YlOrRd", extend="max",
                         cs=u_cs, coast="0.25", lab=u_lab, title="250-hPa zonal wind (abs)"),
            "z500": dict(levels=np.linspace(-zlim, zlim, 21), cmap="RdBu_r", extend="both",
                         cs=None, coast="0.35", lab="m (anomaly)", title="500-hPa height anomaly")}
    col_cf = {}
    for i, lg in enumerate(LAGS_SHOW):
        for j, f in enumerate(FIELDS):
            ax = axes[i, j]; field = maps[(f, lg)]; sp = SPEC[f]
            col_cf[j] = ax.contourf(lons, lats, field, levels=sp["levels"], cmap=sp["cmap"],
                                    extend=sp["extend"], transform=ccrs.PlateCarree())
            if sp["cs"]:
                ax.contour(lons, lats, field, levels=sp["cs"], colors="k", linewidths=0.4,
                           alpha=0.5, transform=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.5, color=sp["coast"])
            ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
            if j == 0:
                ax.text(-0.018, 0.5, ("lag 0 d" if lg == 0 else f"lag {lg:+d} d"), transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=9.5, fontweight="bold", clip_on=False)
            if i == 0:
                ax.set_title(sp["title"], fontsize=11)
    for j, f in enumerate(FIELDS):
        cb = fig.colorbar(col_cf[j], ax=list(axes[:, j]), location="bottom",
                          shrink=0.55, aspect=32, pad=0.008)
        cb.set_label(SPEC[f]["lab"], fontsize=9); cb.ax.tick_params(labelsize=7)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"torque_teleconn_{index}_{season}_composite_abs.webp", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved torque_teleconn_{index}_{season}_composite_abs.webp", flush=True)


def render_jet_hovmoller(hov, lons, lags, index: str, season: str):
    """Lag–longitude Hovmöller of the standardized u250 response in the 25–50°N jet band.
    An eastward (rightward) tilt with increasing lag = downstream jet extension — the
    clearest single view of Himalaya → N-Pacific → N-America propagation."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    lim = float(np.nanpercentile(np.abs(hov), 98)) or 0.1
    cf = ax.contourf(lons, lags, hov, levels=np.linspace(-lim, lim, 21),
                     cmap="RdBu_r", extend="both")
    ax.invert_yaxis()                                          # lag 0 at top, increasing down
    for lon0, lab in [(90, "Himalaya"), (180, "dateline"), (235, "W N.Am")]:
        ax.axvline(lon0, color="0.35", lw=0.9, ls=":")
        ax.text(lon0, lags[0] - 0.7, lab, fontsize=8, ha="center", color="0.35")
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 60))
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("lag (days; torque leads →)")
    fig.colorbar(cf, ax=ax, label="std u250 response (≈ r) · 25–50°N cos-mean")
    fig.suptitle(f"{LABELS[index]} → 250-hPa jet — lag–longitude Hovmöller — {season}",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, "eastward tilt with lag ⇒ downstream jet extension "
             "(Himalaya → N. Pacific → N. America)", ha="center", fontsize=9,
             style="italic", color="0.3")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"torque_teleconn_{index}_{season}_jet_hovmoller.webp"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}", flush=True)


def render_composite_anim(Fabs, Fanom, istd, smask, full, pos, pres, lats, lons,
                          index, season, sigma):
    """Daily composite frames (one per lag, t−2…t+10): absolute 250-hPa jet | 500-hPa
    height anomaly over the N Pacific, on +sigma-torque days. Fixed scales across frames
    so the animation is comparable; writes a manifest the page player reads."""
    slug = _slug(index)
    proj = ccrs.PlateCarree(central_longitude=180); pc = ccrs.PlateCarree()
    if "tendency" in index or "glaam" in index:   # hemispheric AAM state / tendency
        EXTENT_U = [-179.9, 179.9, 0, 87]; EXTENT_Z = [-179.9, 179.9, 0, 87]
    elif "rockies" in index:                # Rockies source → downstream N America / Atlantic
        EXTENT_U = [180, 320, 12, 74]; EXTENT_Z = [180, 360, 10, 82]
    else:                                   # Himalaya (default): N Pacific → North America
        EXTENT_U = [120, 260, 8, 74]; EXTENT_Z = [130, 308, 8, 82]
    u_lev, u_cs = ((np.arange(15, 51, 3), [25, 40]) if season == "JJA"
                   else (np.arange(30, 86, 5), [50, 70]))
    u_lab = f"m s⁻¹ (jet ≥{15 if season == 'JJA' else 30})"
    sig_z = Fanom["z500"][smask].std(0).reshape(lats.size, lons.size)   # local σ for standardizing
    frames = []
    for lg in ANIM_LAGS:
        t = pos[smask]; t = t[(t + lg >= 0) & (t + lg < len(full))]; t = t[pres[t + lg]]
        x = istd[t]; hi, lo = t[x > sigma], t[x < -sigma]
        u = Fabs["u250"][hi + lg].mean(0).reshape(lats.size, lons.size)
        # standardized composite (+σ)−(−σ) with significance — identical to the composite_std map
        d, p = composite(Fanom["z500"][hi + lg], Fanom["z500"][lo + lg])
        z = d.reshape(lats.size, lons.size) / sig_z
        frames.append((lg, u, z, p.reshape(lats.size, lons.size)))
    # match the static composite_std map's scale (same 99.5th-pct ×1.3 over the LAGS_SHOW lags)
    zsub = np.abs(np.array([f[2] for f in frames if f[0] in LAGS_SHOW]))
    zlim = (float(np.nanpercentile(zsub, 99.5)) or 0.3) * 1.3
    z_lev = np.linspace(-zlim, zlim, 21)
    adir = OUT / "anim"; adir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, (lg, u, z, zp) in enumerate(frames):
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.4), subplot_kw={"projection": proj},
                               layout="constrained")
        cu = ax[0].contourf(lons, lats, u, levels=u_lev, cmap="YlOrRd", extend="max", transform=pc)
        ax[0].contour(lons, lats, u, levels=u_cs, colors="k", linewidths=0.4, alpha=0.5, transform=pc)
        ax[0].coastlines(linewidth=0.5, color="0.25"); ax[0].set_extent(EXTENT_U, crs=pc)
        ax[0].set_title("250-hPa jet (abs)", fontsize=10)
        fig.colorbar(cu, ax=ax[0], location="bottom", shrink=0.8, aspect=35, pad=0.02, label=u_lab)
        cz = ax[1].contourf(lons, lats, z, levels=z_lev, cmap="RdBu_r", extend="both", transform=pc)
        ax[1].contourf(lons, lats, np.where(zp < 0.05, 1.0, np.nan), levels=[0.5, 1.5],
                       colors="none", hatches=["..."], transform=pc)
        ax[1].coastlines(linewidth=0.5, color="0.35"); ax[1].set_extent(EXTENT_Z, crs=pc)
        ax[1].set_title("500-hPa height anomaly (standardized)", fontsize=10)
        fig.colorbar(cz, ax=ax[1], location="bottom", shrink=0.8, aspect=35, pad=0.02,
                     label="height anomaly ÷ local σ")
        fig.suptitle(f"{LABELS[index]} composite — {season} — lag {('%+d' % lg) if lg else '0'} d",
                     fontsize=13, fontweight="bold")
        nm = f"{slug}_anim_{season}_f{i:02d}.webp"
        fig.savefig(adir / nm, dpi=100, bbox_inches="tight"); plt.close(fig)
        names.append(nm)
    (adir / f"{slug}_anim_{season}.json").write_text(json.dumps(
        {"lags": ANIM_LAGS, "frames": names, "season": season}))
    print(f"  anim: {len(names)} daily frames, ~{int((istd[smask] > sigma).sum())} "
          f"events averaged/frame ({slug}/{season})", flush=True)


def render_glaam_state(Fanom, sig_f, istd, smask, lats, lons, index, season):
    """Time-mean (composite) standardized circulation anomaly for ±1σ / ±2σ states — ALL
    qualifying days grouped, NO lag. Rows = thresholds, cols = u250 | z500; shared colour
    scale per field so the ±2σ state reads as a stronger version of ±1σ. Stipple p<0.05."""
    proj = ccrs.PlateCarree(central_longitude=180)
    THR = [("+2σ", istd > 2), ("+1σ", istd > 1), ("−1σ", istd < -1), ("−2σ", istd < -2)]
    nrow, ncol = len(THR), len(FIELDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6 * ncol, 1.85 * nrow + 0.5),
                             subplot_kw={"projection": proj}, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.0, hspace=0.0, wspace=0.0)
    axes = np.atleast_2d(axes)
    cells, lim = {}, {f: [] for f in FIELDS}
    for ti, (lab, cond) in enumerate(THR):
        days = np.where(smask & cond)[0]
        for f in FIELDS:
            m, p = composite_one(Fanom[f][days])
            z = m.reshape(lats.size, lons.size) / sig_f[f]
            cells[(ti, f)] = (z, p.reshape(lats.size, lons.size), len(days))
            lim[f].append(np.abs(z[np.isfinite(z)]))
    lim = {f: (float(np.nanpercentile(np.concatenate(lim[f]), 99.5)) or 0.3) * 1.3 for f in FIELDS}
    col_cf = {}
    for ti, (lab, cond) in enumerate(THR):
        for j, f in enumerate(FIELDS):
            ax = axes[ti, j]; z, p, n = cells[(ti, f)]
            col_cf[j] = ax.contourf(lons, lats, z, levels=np.linspace(-lim[f], lim[f], 21),
                                    cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree())
            ax.contourf(lons, lats, np.where(p < 0.05, 1.0, np.nan), levels=[0.5, 1.5],
                        colors="none", hatches=["..."], transform=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.4, color="0.3")
            ax.set_extent([-179.9, 179.9, float(lats.min()), float(lats.max())], crs=ccrs.PlateCarree())
            if j == 0:
                ax.text(-0.02, 0.5, f"GLAAM {lab}\n(n={n})", transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9, fontweight="bold", clip_on=False)
            if ti == 0:
                ax.set_title(FIELDS[f], fontsize=11)
    for j, f in enumerate(FIELDS):
        cb = fig.colorbar(col_cf[j], ax=list(axes[:, j]), location="bottom",
                          shrink=0.55, aspect=32, pad=0.008)
        cb.set_label("anomaly ÷ local σ", fontsize=9); cb.ax.tick_params(labelsize=7)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"torque_teleconn_{index}_{season}_state.webp", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved torque_teleconn_{index}_{season}_state.webp", flush=True)


def relay_diagnostic(tor: xr.Dataset, doy: np.ndarray):
    """Himalaya↔Rockies lead/lag cross-correlation + index autocorrelations."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    h = harmonic_deseason(tor["mtn_himalaya"].values, doy)
    r = harmonic_deseason(tor["mtn_rockies"].values, doy)
    h = (h - h.mean()) / h.std(); r = (r - r.mean()) / r.std()
    lags = np.arange(-20, 21)
    cc = [np.corrcoef(h[max(0, -k):len(h) - max(0, k)],
                      r[max(0, k):len(r) - max(0, -k)])[0, 1] for k in lags]
    a1.plot(lags, cc, lw=2)
    a1.axvline(0, color="0.6", lw=0.8); a1.axhline(0, color="0.6", lw=0.8)
    a1.set_xlabel("lag (days; +) Himalaya leads Rockies"); a1.set_ylabel("cross-correlation")
    a1.set_title("Mountain-torque relay: Himalaya ↔ Rockies")
    for name in INDICES:
        s = harmonic_deseason(tor[name].values, doy)
        s = (s - s.mean()) / s.std()
        al = np.arange(0, 21)
        ac = [1.0] + [np.corrcoef(s[:-k], s[k:])[0, 1] for k in al[1:]]
        a2.plot(al, ac, label=LABELS[name], lw=1.6)
    a2.axhline(1 / np.e, color="0.6", ls="--", lw=0.8, label="1/e")
    a2.set_xlabel("lag (days)"); a2.set_ylabel("autocorrelation")
    a2.set_title("Torque-index autocorrelation"); a2.legend(fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "torque_teleconn_relay_diagnostic.webp"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}", flush=True)


def _load_nino34() -> dict:
    """Monthly Niño-3.4 anomaly {(year, month): value} from the cached NOAA file (or {})."""
    p = Path(__file__).resolve().parent.parent / "data" / "reference" / "nina34.anom.data"
    if not p.exists():
        return {}
    out = {}
    for ln in p.read_text().splitlines()[1:]:
        q = ln.split()
        if len(q) == 13:
            for m in range(12):
                v = float(q[m + 1])
                if v > -90:
                    out[(int(q[0]), m + 1)] = v
    return out


def write_events(index, season, istd, smask, full, sigma, nino):
    """Write the +sigma event list for an index/season → distinct events (peak day per
    consecutive run, with Niño-3.4) + a daily CSV, into the figure folder."""
    idx = np.where(smask & (istd > sigma))[0]
    if len(idx) == 0:
        return
    groups, cur = [], [idx[0]]
    for k in idx[1:]:
        if k - cur[-1] <= 2:
            cur.append(k)
        else:
            groups.append(cur); cur = [k]
    groups.append(cur)

    def n34(i):
        v = nino.get((full[i].year, full[i].month))
        return round(v, 2) if v is not None else ""

    slug = _slug(index)                                     # mtn_himalaya → himalaya
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted((full[g[int(np.argmax(istd[g]))]].strftime("%Y-%m-%d"),
                   round(float(istd[g[int(np.argmax(istd[g]))]]), 2), len(g),
                   n34(g[int(np.argmax(istd[g]))])) for g in groups)
    with open(OUT / f"{slug}_torque_events_{season}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["peak_date", "peak_sigma", "duration_d", "nino34"]); w.writerows(rows)
    with open(OUT / f"{slug}_torque_events_{season}_daily.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "sigma", "nino34"])
        w.writerows((full[i].strftime("%Y-%m-%d"), round(float(istd[i]), 2), n34(i)) for i in idx)
    print(f"  events: {len(idx)} +{sigma:g}σ days → {len(groups)} events ({slug}/{season})", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=1979)
    ap.add_argument("--y1", type=int, default=2023)
    ap.add_argument("--indices", default=",".join(INDICES))
    ap.add_argument("--sigma", type=float, default=2.0, help="event threshold for composites + list")
    args = ap.parse_args(argv)
    indices = args.indices.split(",")
    nino = _load_nino34()

    tor = xr.open_dataset(CACHE / f"torque_indices_{args.y0}_{args.y1}.nc")
    fld = xr.open_dataset(CACHE / f"uz_fields_{args.y0}_{args.y1}.nc")

    # align on a complete daily calendar (NaN-fill gaps so positional lagging is exact)
    common = np.intersect1d(tor.time.values, fld.time.values)
    full = pd.date_range(pd.Timestamp(common.min()).normalize() + pd.Timedelta(hours=12),
                         pd.Timestamp(common.max()).normalize() + pd.Timedelta(hours=12), freq="D")
    tor = tor.reindex(time=full); fld = fld.reindex(time=full)
    doy = full.dayofyear.values
    month = full.month.values
    pres = np.isfinite(tor["mtn_global"].values)               # day present in both
    lats = fld.latitude.values; lons = fld.longitude.values
    npts = lats.size * lons.size

    print(f"loaded {pres.sum()} present days, {args.y0}-{args.y1}; fields "
          f"{lats.size}×{lons.size}", flush=True)

    # deseasonalized field anomalies (ntime, npts) per field. Transpose explicitly:
    # WB2 caches dims as (time, lon, lat), so without this the reshape(nlat, nlon)
    # below would scramble every map (lon-major flatten read back as lat-major).
    Fanom = {f: harmonic_deseason(
                 fld[f].transpose("time", "latitude", "longitude").values.reshape(len(full), -1),
                 doy)
             for f in FIELDS}
    # raw absolute fields (no deseasonalization) — for the absolute-composite panels
    Fabs = {f: fld[f].transpose("time", "latitude", "longitude").values.reshape(len(full), -1)
            for f in FIELDS}
    pos = np.arange(len(full))

    relay_diagnostic(tor, doy)

    for index in indices:
        if index == "glaam_nh":
            # NH relative-AAM state proxy: cos²φ-weighted NH integral of the 250-hPa u anomaly
            w2 = (np.cos(np.deg2rad(lats)) ** 2) * (lats > 0)
            uA = Fanom["u250"].reshape(len(full), lats.size, lons.size)
            ia_raw = np.nansum(uA * w2[None, :, None], axis=(1, 2))
        elif index == "aam_tendency":
            # dM/dt = total torque on the atmosphere (AAM budget) ≈ global mountain + friction
            ia_raw = tor["mtn_global"].values + tor["fric_global"].values
        else:
            ia_raw = tor[index].values
        ia = harmonic_deseason(ia_raw, doy)                    # (ntime,)
        a1x = float(lag1(ia[pres][:, None])[0])                # predictor autocorr
        for season, months in SEASONS.items():
            smask = np.isin(month, months) & pres
            mu, sd = np.nanmean(ia[smask]), np.nanstd(ia[smask])
            istd = (ia - mu) / sd                              # +σ units
            if index == "glaam_nh":
                # absolute STATE: group all qualifying days (±1σ, ±2σ), no lag → mean state
                sig_f = {f: Fanom[f][smask].std(0).reshape(lats.size, lons.size) for f in FIELDS}
                render_glaam_state(Fanom, sig_f, istd, smask, lats, lons, index, season)
                write_events(index, season, istd, smask, full, args.sigma, nino)
                n_hi = int((istd[smask] > args.sigma).sum()); n_lo = int((istd[smask] < -args.sigma).sum())
                print(f"{index} / {season}: state composites · {n_hi} +{args.sigma:g}σ, {n_lo} −{args.sigma:g}σ days", flush=True)
                continue
            reg_maps = {f: {} for f in FIELDS}
            cmp_maps = {f: {} for f in FIELDS}
            pos_maps = {f: {} for f in FIELDS}
            neg_maps = {f: {} for f in FIELDS}
            abs_maps = {}
            for lg in LAGS_SHOW:
                t = pos[smask]
                t = t[(t + lg >= 0) & (t + lg < len(full))]
                t = t[pres[t + lg]]
                x = istd[t]
                hi, lo = t[x > args.sigma], t[x < -args.sigma]
                for f in FIELDS:
                    Y = Fanom[f][t + lg]
                    b, p = regress(x, Y, a1x)
                    reg_maps[f][lg] = (b.reshape(lats.size, lons.size),
                                       p.reshape(lats.size, lons.size))
                    d, pc = composite(Fanom[f][hi + lg], Fanom[f][lo + lg])
                    cmp_maps[f][lg] = (d.reshape(lats.size, lons.size),
                                       pc.reshape(lats.size, lons.size))
                    mp_, pp_ = composite_one(Fanom[f][hi + lg])          # +σ composite alone
                    pos_maps[f][lg] = (mp_.reshape(lats.size, lons.size), pp_.reshape(lats.size, lons.size))
                    mn_, pn_ = composite_one(Fanom[f][lo + lg])          # −σ composite alone
                    neg_maps[f][lg] = (mn_.reshape(lats.size, lons.size), pn_.reshape(lats.size, lons.size))
                    src_arr = Fabs[f] if f == "u250" else Fanom[f]   # abs jet, but z500 ANOMALY
                    abs_maps[(f, lg)] = src_arr[hi + lg].mean(0).reshape(lats.size, lons.size)
            # standardized response only (÷ local σ; regression ⇒ correlation r)
            sig_f = {f: Fanom[f][smask].std(0).reshape(lats.size, lons.size) for f in FIELDS}
            for m, mp in (("regression", reg_maps), ("composite", cmp_maps),
                          ("composite_pos", pos_maps), ("composite_neg", neg_maps)):
                stdz = {(f, lg): (mp[f][lg][0] / sig_f[f], mp[f][lg][1])
                        for f in FIELDS for lg in LAGS_SHOW}
                render(stdz, lats, lons, index, season, m, normalized=True)
            render_abs(abs_maps, lats, lons, index, season)
            # jet lag–longitude Hovmöller: standardized u250 response in the 25–50°N band
            jetlat = (lats >= 25) & (lats <= 50)
            wj = np.cos(np.deg2rad(lats[jetlat]))
            hov_lags = list(range(-2, 11))                 # t−2 → t+10
            hov = np.full((len(hov_lags), lons.size), np.nan)
            for k, lg in enumerate(hov_lags):
                t = pos[smask]
                t = t[(t + lg >= 0) & (t + lg < len(full))]
                t = t[pres[t + lg]]
                b, _ = regress(istd[t], Fanom["u250"][t + lg], a1x)
                r = b.reshape(lats.size, lons.size) / sig_f["u250"]
                hov[k] = np.average(r[jetlat], axis=0, weights=wj)
            render_jet_hovmoller(hov, lons, hov_lags, index, season)
            render_composite_anim(Fabs, Fanom, istd, smask, full, pos, pres,
                                  lats, lons, index, season, args.sigma)
            write_events(index, season, istd, smask, full, args.sigma, nino)
            n_hi = int((istd[smask] > args.sigma).sum()); n_lo = int((istd[smask] < -args.sigma).sum())
            print(f"{index} / {season}: {n_hi} +{args.sigma:g}σ, {n_lo} −{args.sigma:g}σ days", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
