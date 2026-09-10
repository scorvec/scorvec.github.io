#!/usr/bin/env python3
"""Atmospheric river monitor from the AIFS-ENS open data — North Pacific and the West Coast.

Integrated vapour transport (IVT, kg m⁻¹ s⁻¹) is the AR variable (Ralph et al. 2004; Rutz et
al. 2014; the Ralph et al. 2019 AR scale). Exact IVT needs humidity and wind through the column,
which for 51 members × 31 steps is ~12 GB per cycle — so:
  • the CONTROL member gets exact IVT: (1/g) ∫ q·V dp over 1000–300 hPa, 12-hourly to day 15;
  • every member gets a PROXY, IVT ≈ k · TCW · V(850 hPa), with k fitted each cycle against the
    control's exact IVT (through the origin, where IVT > 100) and the fit quality printed on the
    figure. Total column water is in the open data; the 850 hPa wind is already pulled.
Products (assets/ar/):
  anim/ar_ivt, ar_prob, ar_ctrl + ar_manifest.json   12-hourly loops to day 15 (ensemble-mean IVT with vectors;
                                                      P(IVT ≥ 250); member 0's exact IVT) for the site viewer
  ar_coast.webp   West-coast landfall tool: coast latitude × forecast time, ensemble-mean IVT and P(≥250/500)
  ar_monitor.json AR-scale probabilities at the named coastal points, the calibration, the coast arrays
    python scripts/ar/ar_monitor.py --date 20260906 --time 00
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "ecmwf"))
import store as ecmwf                                                  # noqa: E402

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
ASSETS = SITE / "assets" / "ar"
COAST = json.loads((HERE / "data" / "coast_transect.json").read_text())
S12 = tuple(range(0, 361, 12))
LEVELS = (1000, 925, 850, 700, 500, 300)
DOM = dict(lat=(10.0, 70.0), lon=(120.0, 270.0))                       # 120°E–90°W
G0 = 9.80665
THR = (250, 500, 750, 1000, 1250)
INK, MUTED, NAVY = "#1a1a1a", "#8a8680", "#1b365d"


# ── data ─────────────────────────────────────────────────────────────────────
def _open(path, var, levels=None):
    da = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}, chunks={"number": 1} if "pf_" in path.name else None)[var]
    if levels is not None and "isobaricInhPa" in da.dims:
        da = da.sel(isobaricInhPa=list(levels))
    if "number" not in da.dims:
        da = da.expand_dims("number")
    lon = da.longitude.values
    da = da.assign_coords(longitude=np.where(lon < 0, lon + 360, lon)).sortby("longitude").sortby("latitude")
    da = da.sel(latitude=slice(DOM["lat"][0], DOM["lat"][1]), longitude=slice(DOM["lon"][0], DOM["lon"][1]))
    return da


def members(cyc, param, levtype, levels=()):
    parts = []
    for typ in ("cf", "pf"):
        p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, param, levtype, tuple(levels), S12))
        da = _open(p, param if param != "tcw" else "tcw", levels or None)
        if levels and "isobaricInhPa" in da.dims:
            da = da.squeeze("isobaricInhPa", drop=True)
        parts.append(da)
    da = xr.concat(parts, dim="number").assign_coords(number=np.arange(sum(p.sizes["number"] for p in parts)))
    return da.transpose("number", "step", "latitude", "longitude").astype("float32").load()


def control_ivt(cyc):
    """Exact IVT (magnitude, east, north) for member 0, (step, lat, lon)."""
    q = _open(ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "q", "pl", LEVELS, S12)), "q", LEVELS).load()
    u = _open(ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "u", "pl", LEVELS, S12)), "u", LEVELS).load()
    v = _open(ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl", LEVELS, S12)), "v", LEVELS).load()
    p = np.array(sorted(LEVELS, reverse=True), float) * 100.0           # 1000 → 300 Pa
    q = q.sel(isobaricInhPa=p / 100).squeeze("number", drop=True); u = u.sel(isobaricInhPa=p / 100).squeeze("number", drop=True); v = v.sel(isobaricInhPa=p / 100).squeeze("number", drop=True)
    qu = (q * u).values; qv = (q * v).values                             # (step, lev, lat, lon)
    dp = -np.diff(p)                                                     # positive, Pa
    ivte = np.sum(0.5 * (qu[:, 1:] + qu[:, :-1]) * dp[None, :, None, None], axis=1) / G0
    ivtn = np.sum(0.5 * (qv[:, 1:] + qv[:, :-1]) * dp[None, :, None, None], axis=1) / G0
    return np.hypot(ivte, ivtn), ivte, ivtn, q.latitude.values, q.longitude.values, q.step.values


# ── AR scale ─────────────────────────────────────────────────────────────────
def ar_rank(series: np.ndarray, dt_h: float = 12.0) -> tuple[int, float, float]:
    """Ralph et al. (2019) category from a point IVT time series: preliminary rank from the peak IVT
    (250/500/750/1000/1250 → 1..5), −1 if the ≥250 spell is shorter than 24 h, +1 if 48 h or longer
    (capped at 5). Returns (rank, max IVT, duration h) for the strongest spell; rank 0 = no AR."""
    best = (0, 0.0, 0.0)
    i = 0; n = len(series)
    while i < n:
        if series[i] >= 250:
            j = i
            while j + 1 < n and series[j + 1] >= 250:
                j += 1
            peak = float(series[i:j + 1].max()); dur = dt_h * (j - i + 1)
            r = int(np.searchsorted(THR, peak, side="right"))
            if dur < 24: r -= 1
            elif dur >= 48: r += 1
            r = int(np.clip(r, 0, 5))
            if (r, peak) > (best[0], best[1]):
                best = (r, peak, dur)
            i = j + 1
        else:
            i += 1
    return best


# ── figures ──────────────────────────────────────────────────────────────────
def _frame_axes(fig):
    import cartopy.crs as ccrs
    ax = fig.add_axes([0.01, 0.11, 0.98, 0.82], projection=ccrs.PlateCarree(central_longitude=200))
    ax.set_extent([130, 250, 15, 65], crs=ccrs.PlateCarree())
    return ax


def _coast(ax):
    import cartopy.feature as cfeature
    ax.coastlines(resolution="50m", lw=1.0, color="#000", zorder=6); ax.add_feature(cfeature.BORDERS, lw=0.5, edgecolor="#222", zorder=6); ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="#555", zorder=6)


def loops(ivt, ivte_m, ivtn_m, ivt_c, ivte, ivtn, lat, lon, valid, init, k_fit, anim: Path, manifest: Path) -> None:
    """Three 12-hourly loops to day 15 on the viewer's manifest contract:
    ar_ivt  ensemble-mean IVT magnitude and vector, P(≥250) contours
    ar_prob probability of AR conditions (IVT ≥ 250) with the ensemble-mean 250/500 contours
    ar_ctrl member 0's exact IVT with vectors (what the proxy is calibrated against)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    dirs = {k: anim / k for k in ("ar_ivt", "ar_prob", "ar_ctrl")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("F*.webp"):
            old.unlink()
    entries = {k: [] for k in dirs}
    lev_ivt = [250, 300, 400, 500, 600, 750, 1000, 1250, 1500]; cm_ivt = plt.get_cmap("YlGnBu", len(lev_ivt))
    lev_p = np.arange(0.1, 1.01, 0.1); cm_p = plt.get_cmap("YlOrRd", len(lev_p) - 1)
    sk = 14
    mean_mag = ivt.mean(0)                                             # (step, lat, lon)
    prob = (ivt >= 250).mean(0)
    for k, t in enumerate(valid):
        lab = ("analysis" if k == 0 else f"+{12 * k} h") + f" · {pd.Timestamp(t):%a %d %b %HZ}"
        # 1. ensemble-mean IVT
        fig = plt.figure(figsize=(12, 6.3)); ax = _frame_axes(fig)
        cf = ax.pcolormesh(lon, lat, np.where(mean_mag[k] >= 250, mean_mag[k], np.nan), cmap=cm_ivt, norm=BoundaryNorm(lev_ivt, cm_ivt.N, extend="max"), transform=pc, shading="nearest", rasterized=True)
        ue, un = ivte_m[k], ivtn_m[k]; m = mean_mag[k] >= 200
        ax.quiver(lon[::sk], lat[::sk], np.where(m, ue, np.nan)[::sk, ::sk], np.where(m, un, np.nan)[::sk, ::sk], transform=pc, scale=12000, width=0.0022, color="#111", alpha=0.85)
        ax.contour(lon, lat, prob[k], levels=[0.5, 0.8], colors=["#b4453c", "#6a0d0d"], linewidths=[0.8, 1.1], transform=pc)
        _coast(ax); ax.set_title(f"Ensemble-mean IVT — AIFS-ENS 51 members, init {init:%d %b %HZ} · {lab}", fontsize=10.5, loc="left", fontweight="bold")
        cax = fig.add_axes([0.25, 0.065, 0.5, 0.02]); cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=8)
        cb.set_label("IVT (kg m⁻¹ s⁻¹), proxy calibrated on member 0 (r = %.2f) · arrows where ≥ 200 · red: P(IVT ≥ 250) = 0.5 and 0.8" % k_fit["r"], fontsize=8)
        fp = dirs["ar_ivt"] / f"F{k:02d}.webp"; fig.savefig(fp, dpi=100, facecolor="white", pil_kwargs={"quality": 82, "method": 6}); plt.close(fig)
        entries["ar_ivt"].append({"idx": k, "file": fp.name, "date": pd.Timestamp(t).strftime("%Y-%m-%d"), "label": lab})
        # 2. probability
        fig = plt.figure(figsize=(12, 6.3)); ax = _frame_axes(fig)
        cf = ax.pcolormesh(lon, lat, np.where(prob[k] >= 0.1, prob[k], np.nan), cmap=cm_p, norm=BoundaryNorm(lev_p, cm_p.N), transform=pc, shading="nearest", rasterized=True)
        ax.contour(lon, lat, mean_mag[k], levels=[250, 500], colors=["#1b365d", "#08102a"], linewidths=[0.8, 1.1], transform=pc)
        _coast(ax); ax.set_title(f"Probability of AR conditions, IVT ≥ 250 — AIFS-ENS 51 members, init {init:%d %b %HZ} · {lab}", fontsize=10.5, loc="left", fontweight="bold")
        cax = fig.add_axes([0.25, 0.065, 0.5, 0.02]); cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=8)
        cb.set_label("fraction of members with IVT ≥ 250 kg m⁻¹ s⁻¹ · navy contours: ensemble-mean IVT 250 and 500", fontsize=8)
        fp = dirs["ar_prob"] / f"F{k:02d}.webp"; fig.savefig(fp, dpi=100, facecolor="white", pil_kwargs={"quality": 82, "method": 6}); plt.close(fig)
        entries["ar_prob"].append({"idx": k, "file": fp.name, "date": pd.Timestamp(t).strftime("%Y-%m-%d"), "label": lab})
        # 3. member 0, exact
        fig = plt.figure(figsize=(12, 6.3)); ax = _frame_axes(fig)
        cf = ax.pcolormesh(lon, lat, np.where(ivt_c[k] >= 250, ivt_c[k], np.nan), cmap=cm_ivt, norm=BoundaryNorm(lev_ivt, cm_ivt.N, extend="max"), transform=pc, shading="nearest", rasterized=True)
        m = ivt_c[k] >= 200
        ax.quiver(lon[::sk], lat[::sk], np.where(m, ivte[k], np.nan)[::sk, ::sk], np.where(m, ivtn[k], np.nan)[::sk, ::sk], transform=pc, scale=12000, width=0.0022, color="#111", alpha=0.85)
        _coast(ax); ax.set_title(f"Control member, exact IVT (1/g)∫q·V dp, 1000–300 hPa — init {init:%d %b %HZ} · {lab}", fontsize=10.5, loc="left", fontweight="bold")
        cax = fig.add_axes([0.25, 0.065, 0.5, 0.02]); cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=8)
        cb.set_label("IVT (kg m⁻¹ s⁻¹) · arrows: IVT vector where ≥ 200", fontsize=8)
        fp = dirs["ar_ctrl"] / f"F{k:02d}.webp"; fig.savefig(fp, dpi=100, facecolor="white", pil_kwargs={"quality": 82, "method": 6}); plt.close(fig)
        entries["ar_ctrl"].append({"idx": k, "file": fp.name, "date": pd.Timestamp(t).strftime("%Y-%m-%d"), "label": lab})
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "default": "ar_ivt",
            "regions": {"ar_ivt": {"label": "Ensemble-mean IVT", "frames": entries["ar_ivt"]},
                        "ar_prob": {"label": "P(IVT ≥ 250)", "frames": entries["ar_prob"]},
                        "ar_ctrl": {"label": "Control, exact IVT", "frames": entries["ar_ctrl"]}}}
    manifest.parent.mkdir(parents=True, exist_ok=True); manifest.write_text(json.dumps(mani))
    print(f"  wrote {len(valid)} frames × 3 loops → {manifest}", flush=True)


def coast_tool(ivt, ivt_c, lat, lon, valid, init, out: Path) -> dict:
    """Landfall Hovmöller on the coastal transect. Returns the arrays for the JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    tr = COAST["transect"]
    tl = np.array([r["lat"] for r in tr]); tlon = np.array([r["point_lon"] % 360 for r in tr])
    ii = np.array([int(np.argmin(np.abs(lat - la))) for la in tl]); jj = np.array([int(np.argmin(np.abs(lon - lo))) for lo in tlon])
    series = ivt[:, :, ii, jj]                                         # (member, step, coastpoint)
    ctrl = ivt_c[:, ii, jj]
    mean = series.mean(0); p250 = (series >= 250).mean(0); p500 = (series >= 500).mean(0); p750 = (series >= 750).mean(0)
    t = pd.DatetimeIndex(valid)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2), sharey=True)
    ax = axes[0]
    cf = ax.contourf(t, tl, mean.T, levels=[100, 150, 200, 250, 300, 400, 500, 600, 750, 1000, 1250], cmap="YlGnBu", extend="max")
    ax.contour(t, tl, ctrl.T, levels=[250, 500], colors=["#b4453c", "#6a0d0d"], linewidths=[0.9, 1.1])
    ax.set_title("Ensemble-mean IVT at the coast; red: member 0 250 / 500", fontsize=9.5, loc="left", fontweight="bold")
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05); cb.ax.tick_params(labelsize=7.5)
    ax = axes[1]
    cf = ax.contourf(t, tl, p250.T, levels=np.arange(0.1, 1.01, 0.1), cmap="YlOrRd", extend="max")
    ax.contour(t, tl, p500.T, levels=[0.25, 0.5], colors="#333", linewidths=[0.8, 1.2], linestyles=["--", "-"])
    ax.set_title("P(IVT ≥ 250); contours P(IVT ≥ 500) = 0.25, 0.5", fontsize=9.5, loc="left", fontweight="bold")
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05); cb.ax.tick_params(labelsize=7.5)
    ax = axes[2]
    q90 = np.percentile(series, 90, axis=0)
    cf = ax.contourf(t, tl, q90.T, levels=[250, 300, 400, 500, 600, 750, 1000, 1250, 1500], cmap="PuRd", extend="max")
    ax.set_title("90th-percentile member IVT — the strong-AR tail", fontsize=9.5, loc="left", fontweight="bold")
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05); cb.ax.tick_params(labelsize=7.5)
    named = [nm for nm in COAST["named"] if tl.min() <= nm["lat"] <= tl.max()]
    ticks, labels = [], []
    for nm in named:                                                   # names as the y axis, thinned so none collide
        if not ticks or nm["lat"] - ticks[-1] >= 1.5:
            ticks.append(nm["lat"]); labels.append(f'{nm["name"]}  {nm["lat"]:.0f}°N')
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        for la in ticks:
            ax.axhline(la, color="#000", lw=0.3, alpha=0.35)
    axes[0].set_yticks(ticks); axes[0].set_yticklabels(labels, fontsize=7.5)
    axes[2].yaxis.tick_right(); axes[2].yaxis.set_label_position("right"); axes[2].set_yticks(range(25, 60, 5)); axes[2].set_yticklabels([f"{v}°N" for v in range(25, 60, 5)], fontsize=7.5)
    fig.suptitle(f"West-coast atmospheric-river landfall tool — AIFS-ENS 51 members, init {init:%Y-%m-%d %HZ}, 12-hourly to day 15", fontsize=13, fontweight="bold", x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.945, "Read at the first ocean point west of the coast on each 0.25° latitude (Baja California to southeast Alaska). IVT from the calibrated proxy for the members, exact for member 0.", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.115, right=0.965, top=0.88, bottom=0.1, wspace=0.06)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    print(f"saved {out}", flush=True)
    return {"lat": tl.tolist(), "mean": np.round(mean, 0).tolist(), "p250": np.round(p250, 2).tolist(), "p500": np.round(p500, 2).tolist(), "p750": np.round(p750, 2).tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    a = ap.parse_args()
    t0 = time.time()
    cyc = ecmwf.Cycle(a.date, a.time); init = pd.Timestamp(f"{a.date}T{a.time}:00")
    ASSETS.mkdir(parents=True, exist_ok=True)
    tcw = members(cyc, "tcw", "sfc"); print(f"  tcw {dict(tcw.sizes)} ({time.time() - t0:.0f}s)", flush=True)
    u8 = members(cyc, "u", "pl", (850,)); v8 = members(cyc, "v", "pl", (850,)); print(f"  850 hPa wind loaded ({time.time() - t0:.0f}s)", flush=True)
    lat, lon = tcw.latitude.values, tcw.longitude.values
    steps = (tcw.step / np.timedelta64(1, "h")).values.astype(int)
    valid = [init + pd.Timedelta(hours=int(h)) for h in steps]
    ivt_c, ivte, ivtn, clat, clon, csteps = control_ivt(cyc); print(f"  member 0 IVT ({time.time() - t0:.0f}s)", flush=True)
    assert clat.shape == lat.shape and clon.shape == lon.shape
    # proxy calibration on member 0: IVT ≈ k · tcw · |V850|
    spd = np.hypot(u8.values, v8.values)
    x = (tcw.values[0] * spd[0]).ravel(); y = ivt_c.ravel(); ok = np.isfinite(x) & np.isfinite(y) & (y > 100)
    k = float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum()); r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    k_fit = {"k": k, "r": r, "n": int(ok.sum()), "rmse": float(np.sqrt(np.mean((k * x[ok] - y[ok]) ** 2)))}
    print(f"  proxy: k = {k:.3f}, r = {r:.3f}, rmse {k_fit['rmse']:.0f} kg/m/s over {ok.sum():,} member-0 points", flush=True)
    ivt = (k * tcw.values * spd).astype("float32")                      # (member, step, lat, lon)
    ivt[0] = ivt_c                                                       # the control keeps its exact field
    ivte_m = (k * tcw.values * u8.values).mean(0); ivtn_m = (k * tcw.values * v8.values).mean(0)   # ensemble-mean vector
    ivte_m[:] = np.where(np.isfinite(ivte_m), ivte_m, 0); ivtn_m[:] = np.where(np.isfinite(ivtn_m), ivtn_m, 0)
    del spd
    coast = coast_tool(ivt, ivt_c, lat, lon, valid, init, ASSETS / "ar_coast.webp")
    loops(ivt, ivte_m, ivtn_m, ivt_c, ivte, ivtn, lat, lon, valid, init, k_fit, ASSETS / "anim", ASSETS / "anim" / "ar_manifest.json")
    # AR scale at the named points
    table = []
    for nm in COAST["named"]:
        i = int(np.argmin(np.abs(lat - nm["lat"]))); j = int(np.argmin(np.abs(lon - nm["lon"] % 360)))
        ranks, peaks, onsets = [], [], []
        for m in range(ivt.shape[0]):
            rk, pk, du = ar_rank(ivt[m, :, i, j]); ranks.append(rk); peaks.append(pk)
            hit = np.where(ivt[m, :, i, j] >= 250)[0]; onsets.append(int(steps[hit[0]]) if hit.size else None)
        ranks = np.array(ranks)
        table.append({"name": nm["name"], "lat": nm["lat"], "lon": nm["lon"], "p_ar": round(float((ranks >= 1).mean()), 2),
                      "p_ge": {str(c): round(float((ranks >= c).mean()), 2) for c in range(1, 6)},
                      "most_likely": int(np.bincount(ranks, minlength=6).argmax()), "median_peak": round(float(np.median(peaks)), 0),
                      "onset_h_median": (None if all(o is None for o in onsets) else float(np.median([o for o in onsets if o is not None]))),
                      "control_rank": int(ranks[0])})
    doc = {"init": init.strftime("%Y-%m-%dT%HZ"), "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "members": int(ivt.shape[0]),
           "steps_h": steps.tolist(), "proxy": k_fit, "coast": coast, "named": table,
           "scale": "Ralph et al. 2019: rank from peak IVT (250/500/750/1000/1250), -1 if the spell is under 24 h, +1 if 48 h or longer; 12-hourly sampling"}
    (ASSETS / "ar_monitor.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote ar_monitor.json in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
