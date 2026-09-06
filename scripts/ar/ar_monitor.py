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
  ar_prob.webp    P(IVT ≥ 250) for days 1–10 (daily max of the two 12-h steps), ensemble-mean 250 contour
  ar_coast.webp   West-coast landfall tool: coast latitude × forecast time, ensemble-mean IVT and P(≥250/500)
  ar_now.webp     control-member exact IVT at the analysis and days 1–3 (vectors + magnitude)
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
    """Exact IVT (magnitude, east, north) for the control member, (step, lat, lon)."""
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
def maps_prob(ivt, lat, lon, valid, init, k_fit, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    pc = ccrs.PlateCarree(central_longitude=200)
    days = pd.DatetimeIndex(valid).normalize()
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 4.9), subplot_kw={"projection": pc})
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for d in range(1, 11):
        ax = axes[(d - 1) // 5, (d - 1) % 5]
        day = init.normalize() + pd.Timedelta(days=d)
        sel = np.where(days == day)[0]
        if sel.size == 0:
            ax.axis("off"); continue
        daymax = ivt[:, sel].max(axis=1)                               # (member, lat, lon)
        prob = (daymax >= 250).mean(0)
        mean = daymax.mean(0)
        cf = ax.contourf(lon, lat, prob, levels=levels, cmap="YlGnBu", extend="max", transform=ccrs.PlateCarree())
        ax.contour(lon, lat, mean, levels=[250, 500], colors=["#b4453c", "#6a0d0d"], linewidths=[0.9, 1.1], transform=ccrs.PlateCarree())
        ax.coastlines(lw=0.6, color="#222"); ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#666"); ax.add_feature(cfeature.STATES, lw=0.2, edgecolor="#888")
        ax.set_extent([130, 250, 20, 65], crs=ccrs.PlateCarree())
        ax.set_title(f"Day {d} · {day:%a %d %b}", fontsize=9.5, loc="left", fontweight="bold")
    cax = fig.add_axes([0.30, 0.085, 0.40, 0.022]); cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=8)
    cb.set_label("probability that IVT reaches 250 kg m⁻¹ s⁻¹ at some point in the day (51 members) · red contours: ensemble-mean daily-max IVT 250 and 500", fontsize=8.5)
    fig.suptitle(f"Atmospheric rivers: probability of AR conditions by day — AIFS-ENS 51 members, init {init:%Y-%m-%d %HZ}", fontsize=13, fontweight="bold", x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.925, f"IVT for the members is the proxy k·TCW·|V₈₅₀| with k = {k_fit['k']:.3f} fitted to the control member's exact IVT this cycle (r = {k_fit['r']:.2f} over {k_fit['n']:,} points with IVT > 100).", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.84, bottom=0.17, wspace=0.05, hspace=0.3)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    print(f"saved {out}", flush=True)


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
    ax.set_title("Ensemble-mean IVT at the coast; red: control 250 / 500", fontsize=9.5, loc="left", fontweight="bold")
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
    fig.text(0.02, 0.945, "Read at the first ocean point west of the coast on each 0.25° latitude (Baja California to southeast Alaska). IVT from the calibrated proxy for the members, exact for the control.", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.115, right=0.965, top=0.88, bottom=0.1, wspace=0.06)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    print(f"saved {out}", flush=True)
    return {"lat": tl.tolist(), "mean": np.round(mean, 0).tolist(), "p250": np.round(p250, 2).tolist(), "p500": np.round(p500, 2).tolist(), "p750": np.round(p750, 2).tolist()}


def now_maps(ivt_c, ivte, ivtn, lat, lon, valid, init, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    pc = ccrs.PlateCarree(central_longitude=200)
    picks = [0, 2, 4, 6]                                                # analysis, +24, +48, +72 h (12-h steps)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), subplot_kw={"projection": pc})
    for k, s in enumerate(picks):
        ax = axes[k // 2, k % 2]
        cf = ax.contourf(lon, lat, ivt_c[s], levels=[250, 300, 400, 500, 600, 750, 1000, 1250, 1500], cmap="YlGnBu", extend="max", transform=ccrs.PlateCarree())
        sk = 8
        m = ivt_c[s] >= 200
        qe = np.where(m, ivte[s], np.nan)[::sk, ::sk]; qn = np.where(m, ivtn[s], np.nan)[::sk, ::sk]
        ax.quiver(lon[::sk], lat[::sk], qe, qn, transform=ccrs.PlateCarree(), scale=14000, width=0.002, color="#111", alpha=0.75)
        ax.contour(lon, lat, ivt_c[s], levels=[250], colors="#6a0d0d", linewidths=0.6, transform=ccrs.PlateCarree())
        ax.coastlines(lw=0.6, color="#222"); ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#666"); ax.add_feature(cfeature.STATES, lw=0.2, edgecolor="#888")
        ax.set_extent([130, 250, 15, 65], crs=ccrs.PlateCarree())
        ax.set_title(("0-h analysis" if s == 0 else f"+{12 * s} h") + f" · {pd.Timestamp(valid[s]):%a %d %b %HZ}", fontsize=10, loc="left", fontweight="bold")
    cax = fig.add_axes([0.30, 0.06, 0.40, 0.018]); cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=8)
    cb.set_label("IVT (kg m⁻¹ s⁻¹), control member, exact: (1/g)∫ q·V dp over 1000–300 hPa · arrows: IVT vector where IVT ≥ 200", fontsize=8.5)
    fig.suptitle(f"Atmospheric rivers now and the next three days — AIFS-ENS control, init {init:%Y-%m-%d %HZ}", fontsize=13, fontweight="bold", x=0.02, ha="left", y=0.985)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.93, bottom=0.13, wspace=0.04, hspace=0.16)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    print(f"saved {out}", flush=True)


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
    ivt_c, ivte, ivtn, clat, clon, csteps = control_ivt(cyc); print(f"  control IVT ({time.time() - t0:.0f}s)", flush=True)
    assert clat.shape == lat.shape and clon.shape == lon.shape
    # proxy calibration on the control: IVT ≈ k · tcw · |V850|
    spd = np.hypot(u8.values, v8.values)
    x = (tcw.values[0] * spd[0]).ravel(); y = ivt_c.ravel(); ok = np.isfinite(x) & np.isfinite(y) & (y > 100)
    k = float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum()); r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    k_fit = {"k": k, "r": r, "n": int(ok.sum()), "rmse": float(np.sqrt(np.mean((k * x[ok] - y[ok]) ** 2)))}
    print(f"  proxy: k = {k:.3f}, r = {r:.3f}, rmse {k_fit['rmse']:.0f} kg/m/s over {ok.sum():,} control points", flush=True)
    ivt = (k * tcw.values * spd).astype("float32")                      # (member, step, lat, lon)
    ivt[0] = ivt_c                                                       # the control keeps its exact field
    maps_prob(ivt, lat, lon, valid, init, k_fit, ASSETS / "ar_prob.webp")
    coast = coast_tool(ivt, ivt_c, lat, lon, valid, init, ASSETS / "ar_coast.webp")
    now_maps(ivt_c, ivte, ivtn, lat, lon, valid, init, ASSETS / "ar_now.webp")
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
