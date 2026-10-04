#!/usr/bin/env python3
"""What each weather regime brings at the surface, and the regime patterns themselves - static reference figures
(laptop, once; ERA5 from the local store ~/era5_store, regime days from NCEP/NCAR R1 1991-2020).

IMPACTS. For every regime of both seasonal sets in both sectors: the composite 2 m temperature and precipitation
anomaly over the days the regime was observed (R1, 1991-2020, each day in its own season's set, data/r1_labels.npz),
from ERA5 daily means at 1.5 deg (t2m: wb2_1p5_daily_global; precipitation: wb2_1p5_daily, mm/day - the store's
attributes say m, the README and a value check say mm). Anomaly = the day minus the 1991-2020 day-of-year mean
(31-day circular smoothing) minus a linear trend fitted per grid point and season over the 30 years, so a regime that
happened to be more common late in the period does not inherit the warming.

SIGNIFICANCE (site rule: nothing insignificant is shown as a finding). Regime days come in episodes (mean length ~7-8
days) and neighbouring days are not independent, so the test is on EPISODES: every run of consecutive days in one
regime is averaged into one value, and a two-sided one-sample t-test across episodes asks whether the composite
differs from zero at each grid point. The field of p-values is then controlled for multiplicity with the false
discovery rate (Benjamini-Hochberg, alpha_FDR = 0.10, Wilks 2016) over the grid points of the map. Colour is shown
only where the FDR test passes; elsewhere the map is left white.

PATTERNS. The eight centroids per sector with named regimes (shared family colours), a fixed 20 m colour ladder, and
the share of days in each regime by calendar month (1991-2020).

    python scripts/regimes/build_regime_impacts.py
    -> ~/data_archive/geps_subx/figs/ and assets/geps/: regimes_impacts_{ea,na}_{cold,warm}.webp,
       geps_regimes_{ea,na}.webp (patterns + frequency by month), and scripts/regimes/data/regime_impacts.json
       (share of the map passing the test, per regime and variable)
The figures go out with the GEPS page's own publish (run_geps.sh copies figs/*.webp into assets/geps).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import regime_core as RC                                                  # noqa: E402

STORE = Path.home() / "era5_store"
FIGS = Path.home() / "data_archive" / "geps_subx" / "figs"
SITE = REPO / "assets" / "geps"
import os
OUTS = [Path(x) for x in os.environ["REGIME_FIG_OUT"].split(",")] if os.environ.get("REGIME_FIG_OUT") else [FIGS, SITE]
Y0, Y1 = 1991, 2020
ALPHA_FDR = 0.10
DOMAIN = {"na": dict(lat=(15, 75), lon=(-170, -50), label="North America"),
          "ea": dict(lat=(30, 75), lon=(-30, 50), label="Europe and the North Atlantic")}
T_LEVELS = np.arange(-3.5, 3.51, 0.5)
P_LEVELS = np.array([-3, -2, -1.5, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.5, 2, 3])
Z_LEVELS = np.arange(-140, 141, 20)


def load(var, sub):
    """ERA5 daily 1991-2020 over a lat/lon box -> (time, lat, lon) float32, lat descending, lon -180..180."""
    root = STORE / ("wb2_1p5_daily_global" if var == "t2m" else "wb2_1p5_daily") / var
    parts, t = [], []
    for y in range(Y0, Y1 + 1):
        d = xr.open_dataset(root / f"{var}_{y}.nc")
        a = d[var].transpose("time", "latitude", "longitude")
        a = a.assign_coords(longitude=((a.longitude + 180) % 360) - 180).sortby("longitude").sortby("latitude",
                                                                                                  ascending=False)
        a = a.sel(latitude=slice(sub["lat"][1], sub["lat"][0]), longitude=slice(sub["lon"][0], sub["lon"][1]))
        parts.append(a.values.astype("float32")); t += list(pd.DatetimeIndex(a.time.values))
        lat, lon = a.latitude.values, a.longitude.values
        d.close()
    v = np.concatenate(parts, 0)
    if var == "t2m":
        assert 200 < np.nanmean(v) < 320, "t2m should be K"
        v = v - 273.15
    else:
        assert 0.3 < np.nanmean(v) < 10, "precipitation should be mm/day (store attribute says m - README says mm)"
    return pd.DatetimeIndex(t), lat, lon, v


def anomalies(t, v):
    """Minus the day-of-year mean (31-day circular smoothing), minus a linear trend per grid point and season."""
    from scipy.ndimage import uniform_filter1d
    dy = np.minimum(t.dayofyear.values, 365) - 1
    c = np.zeros((365,) + v.shape[1:]); n = np.zeros(365)
    np.add.at(c, dy, v); np.add.at(n, dy, 1)
    c = uniform_filter1d(c / n[:, None, None], 31, axis=0, mode="wrap")
    a = v - c[dy]
    cold = np.isin(t.month, RC.SEASONS["cold"])
    yr = (t.year + (t.month >= 10)).values.astype("float64")                 # a cold season is one winter
    for m in (cold, ~cold):
        x = yr[m] - yr[m].mean()
        b = np.tensordot(x, a[m], 1) / (x ** 2).sum()
        a[m] -= (x[:, None, None] * b[None]).astype("float32")
    return a


def episodes(lab_ok):
    """Boolean day mask -> list of index arrays, one per run of consecutive True days."""
    idx = np.flatnonzero(lab_ok)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) != 1) + 1
    return np.split(idx, cut)


def fdr(p, alpha=ALPHA_FDR):
    """Benjamini-Hochberg over the finite p-values -> boolean mask of discoveries (Wilks 2016)."""
    q = p[np.isfinite(p)]
    if q.size == 0:
        return np.zeros_like(p, dtype=bool)
    s = np.sort(q)
    k = np.flatnonzero(s <= alpha * np.arange(1, s.size + 1) / s.size)
    thr = s[k.max()] if k.size else -1.0
    return np.isfinite(p) & (p <= thr)


def fdr_threshold_pass(p, p_in):
    """Grid points outside the tested domain (the map's frame corners) are coloured by the same p threshold."""
    q = p_in[np.isfinite(p_in)]
    if q.size == 0:
        return np.zeros_like(p, dtype=bool)
    s = np.sort(q)
    k = np.flatnonzero(s <= ALPHA_FDR * np.arange(1, s.size + 1) / s.size)
    return np.isfinite(p) & (p <= (s[k.max()] if k.size else -1.0))


def _extent_xy(proj, ccrs, dom):
    """The projected bounding box of the lat/lon domain (what set_extent will show)."""
    lo = np.linspace(dom["lon"][0], dom["lon"][1], 60); la = np.linspace(dom["lat"][0], dom["lat"][1], 60)
    LO, LA = np.meshgrid(lo, la)
    xy = proj.transform_points(ccrs.PlateCarree(), LO.ravel(), LA.ravel())
    return xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()


def composite(t, a, lab, k, months):
    m = (lab == k) & np.isin(t.month, months)
    eps = episodes(m)
    E = np.stack([a[e].mean(0) for e in eps])                                 # (episodes, lat, lon)
    tt, p = stats.ttest_1samp(E, 0.0, axis=0)
    mean = a[m].mean(0)
    return mean, p, len(eps), int(m.sum())


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans"})
    ref = RC.Ref()
    r1 = np.load(RC.DATA / "r1_labels.npz")
    rt = pd.DatetimeIndex([pd.Timestamp(str(x)) for x in r1["t"]])
    stats_out = {}
    for sk in RC.SECTORS:
        dom = DOMAIN[sk]
        data = {}
        for var in ("t2m", "prcp"):
            t, lat, lon, v = load(var, dict(lat=(dom["lat"][0] - 8, min(dom["lat"][1] + 8, 88)),
                                              lon=(dom["lon"][0] - 20, dom["lon"][1] + 20)))
            a = anomalies(t, v)
            lab = pd.Series(r1[f"own_{sk}"], index=rt).reindex(t).fillna(-1).values.astype(int)
            data[var] = (t, lat, lon, a, lab)
            print(f"  {sk} {var}: {v.shape}, lat {lat[0]}..{lat[-1]}, lon {lon[0]}..{lon[-1]}", flush=True)
        proj = ccrs.LambertConformal(central_longitude=float(np.mean(dom["lon"])), standard_parallels=(35, 60))
        cent_all = {s: ref.centroids(sk, s) for s in RC.SEASONS}
        for season, months in RC.SEASONS.items():
            st = ref.meta["stats"][f"{sk}_{season}"]
            fams = [(fi, f) for fi, f in enumerate(RC.FAMILIES[sk]) if f[season] is not None]
            fams.sort(key=lambda x: x[1][season])
            ncol = len(fams)
            # geometry from the projected extent, so the maps have no letterbox
            px0, px1, py0, py1 = _extent_xy(proj, ccrs, dom)
            mw = 3.3
            mh = mw * (py1 - py0) / (px1 - px0)
            gap, left, right, top, bot = 0.12, 0.55, 1.25, 1.35, 0.55
            W = left + ncol * mw + (ncol - 1) * gap + right
            H = top + 2 * mh + 0.25 + bot
            fig = plt.figure(figsize=(W, H))
            rows = [("t2m", "2 m temperature (°C)", T_LEVELS, "RdBu_r"),
                    ("prcp", "precipitation (mm/day)", P_LEVELS, "BrBG")]
            for c, (fi, f) in enumerate(fams):
                k = f[season]
                name = ref.name(sk, season, k)
                cent, clat, clon = cent_all[season]
                for r, (var, vlab, lev, cmap) in enumerate(rows):
                    t, lat, lon, a, lab = data[var]
                    mean, p, n_ep, n_d = composite(t, a, lab, k, months)
                    inside = ((lat >= dom["lat"][0]) & (lat <= dom["lat"][1]))[:, None] & \
                             ((lon >= dom["lon"][0]) & (lon <= dom["lon"][1]))[None, :]
                    p_in = np.where(inside, p, np.nan)                       # the test's family = the map domain
                    ok = fdr(p_in) | (~inside & fdr_threshold_pass(p, p_in))
                    stats_out.setdefault(f"{sk}_{season}", {}).setdefault(name, {})[var] = dict(
                        episodes=n_ep, days=n_d, frac_significant=round(float(fdr(p_in)[inside].mean()), 3),
                        max_abs_significant=round(float(np.abs(mean[fdr(p_in)]).max()), 2) if fdr(p_in).any() else None)
                    x0 = (left + c * (mw + gap)) / W
                    y0 = (bot + (1 - r) * (mh + 0.25)) / H
                    ax = fig.add_axes([x0, y0, mw / W, mh / H], projection=proj)
                    ax.set_extent([dom["lon"][0], dom["lon"][1], dom["lat"][0], dom["lat"][1]], ccrs.PlateCarree())
                    shown = np.where(ok, mean, np.nan)
                    cf = ax.contourf(lon, lat, shown, levels=lev, cmap=cmap, extend="both", transform=ccrs.PlateCarree())
                    ax.contour(clon, clat, cent[k], levels=[x for x in Z_LEVELS if x != 0], colors="#222",
                               linewidths=0.6, transform=ccrs.PlateCarree())
                    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, edgecolor="#333")
                    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.35, edgecolor="#555")
                    if sk == "na":
                        ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="#777")
                    for sp in ax.spines.values():
                        sp.set_edgecolor(f["color"]); sp.set_linewidth(2.4)
                    if r == 0:
                        ax.set_title(f"{name}\n{st['freq'][k] * 100:.0f}% of days · {n_ep} episodes", fontsize=11.5,
                                     fontweight="bold", color=RC.INK)
                    if c == 0:
                        fig.text((left - 0.15) / W, y0 + mh / H / 2, "temperature" if var == "t2m" else "precipitation",
                                 rotation=90, ha="right", va="center", fontsize=11.5, fontweight="bold", color=RC.INK)
                    if not fdr(p_in).any():
                        ax.text(0.5, 0.5, "no significant signal", transform=ax.transAxes, ha="center", va="center",
                                fontsize=10, color=RC.MUTED, bbox=dict(facecolor="white", lw=0, alpha=0.8))
                    if c == ncol - 1:
                        cax = fig.add_axes([(left + ncol * (mw + gap)) / W, y0 + 0.06 * mh / H, 0.14 / W, 0.88 * mh / H])
                        cb = fig.colorbar(cf, cax=cax, ticks=lev[::2] if var == "t2m" else lev)
                        cb.set_label(vlab, fontsize=10)
                        cb.ax.tick_params(labelsize=9)
            import textwrap
            wrapc = int(W * 13.5)
            fig.text(0.2 / W, 1 - 0.15 / H, f"{RC.SECTORS[sk]['label']} regimes, {season} season "
                     f"({'Oct–Mar' if season == 'cold' else 'Apr–Sep'}): what each brings at the surface",
                     fontsize=15, fontweight="bold", va="top", color=RC.INK)
            fig.text(0.2 / W, 1 - 0.5 / H, textwrap.fill(
                "ERA5 1991–2020 composite over the days NCEP/NCAR R1 was in each regime; anomalies against the "
                "day-of-year normal with each season's linear trend removed. Contours: the regime's 500 hPa pattern "
                "(every 20 m, negative dashed).", wrapc), fontsize=10, va="top", color=RC.MUTED)
            fig.text(0.2 / W, 0.08 / H, textwrap.fill(
                "Colour only where significant: two-sided t-test on regime-episode means (one value per run of "
                "consecutive regime days), false discovery rate held at 10% over the map (Benjamini–Hochberg; "
                "Wilks 2016). White = no significant difference from normal.", wrapc), fontsize=9.5, va="bottom",
                color=RC.MUTED)
            name = f"regimes_impacts_{sk}_{season}.webp"
            for d in OUTS:
                fig.savefig(d / name, dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
            plt.close(fig)
            print(f"  {name}", flush=True)
        patterns_figure(ref, sk, plt, ccrs, cfeature)
    (RC.DATA / "regime_impacts.json").write_text(json.dumps(
        dict(test="t-test on episode means, BH FDR 0.10 over the map", base="ERA5 1991-2020, R1 regime days",
             regimes=stats_out), indent=1))
    return 0


def patterns_figure(ref, sk, plt, ccrs, cfeature):
    """The eight centroids (named, family colours, a fixed 20 m ladder) and the share of days by calendar month."""
    import textwrap
    spec = RC.SECTORS[sk]
    proj = ccrs.LambertConformal(central_longitude=float(np.mean(spec["lon"])), standard_parallels=(35, 55))
    dom = dict(lat=(spec["lat"][1], spec["lat"][0]), lon=spec["lon"])
    px0, px1, py0, py1 = _extent_xy(proj, ccrs, dom)
    mw, gap, left, barw = 3.3, 0.12, 0.2, 2.9
    mh = mw * (py1 - py0) / (px1 - px0)
    top, row_t, cbar = 1.05, 0.85, 0.95
    W = left + RC.K * mw + (RC.K - 1) * gap + 0.55 + barw + 0.25
    H = top + 2 * (row_t + mh) + cbar
    fig = plt.figure(figsize=(W, H))
    for si, (season, months) in enumerate(RC.SEASONS.items()):
        st = ref.meta["stats"][f"{sk}_{season}"]
        cent, lat, lon = ref.centroids(sk, season)
        y0 = (H - top - (si + 1) * (row_t + mh) + 0.0) / H
        for k in range(RC.K):
            fi = ref.family_index(sk, season, k)
            col = RC.FAMILIES[sk][fi]["color"]
            ax = fig.add_axes([(left + k * (mw + gap)) / W, y0, mw / W, mh / H], projection=proj)
            ax.set_extent([spec["lon"][0], spec["lon"][1], spec["lat"][1], spec["lat"][0]], ccrs.PlateCarree())
            cf = ax.contourf(lon, lat, cent[k], levels=Z_LEVELS, cmap="RdBu_r", extend="both",
                             transform=ccrs.PlateCarree())
            ax.contour(lon, lat, cent[k], levels=[x for x in Z_LEVELS if x != 0], colors="k", linewidths=0.35,
                       transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.6, edgecolor="#333")
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.6)
            ax.set_title(f"{ref.name(sk, season, k)}\n{st['freq'][k] * 100:.0f}% of {season}-season days",
                         fontsize=11.5, fontweight="bold", color=RC.INK)
        axb = fig.add_axes([(W - 0.25 - barw) / W, y0, barw / W, mh / H])
        mlist = list(months)
        bottom = np.zeros(len(mlist))
        for k in [RC.NONE] + list(range(RC.K)):
            v = np.array([st["by_month"][str(m)][k] for m in mlist]) * 100
            col = RC.NONE_COLOR if k == RC.NONE else RC.FAMILIES[sk][ref.family_index(sk, season, k)]["color"]
            axb.bar(range(len(mlist)), v, bottom=bottom, color=col, width=0.82, edgecolor="white", lw=0.8)
            bottom += v
        axb.set_xticks(range(len(mlist)))
        axb.set_xticklabels([dt.date(2000, m, 1).strftime("%b") for m in mlist], fontsize=10)
        axb.set_ylim(0, 100); axb.set_yticks([0, 25, 50, 75, 100])
        axb.tick_params(labelsize=9.5)
        for s_ in ("top", "right"):
            axb.spines[s_].set_visible(False)
        axb.set_title(f"% of days by month, {season} set\n(grey: no regime, {st['freq'][RC.NONE] * 100:.0f}% overall)",
                      fontsize=11, color=RC.INK)
    cax = fig.add_axes([(left + 0.6) / W, 0.55 / H, (2 * mw) / W, 0.16 / H])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=Z_LEVELS[::2])
    cb.set_label("500 hPa height anomaly (m), 20 m steps", fontsize=10.5)
    fig.text(left / W, 1 - 0.15 / H, f"{spec['label']} weather regimes — the patterns and when they occur", fontsize=15,
             fontweight="bold", va="top", color=RC.INK)
    fig.text(left / W, 1 - 0.5 / H, textwrap.fill(
        "k-means (k = 4) on 5-day-mean 500 hPa anomalies, NCEP/NCAR R1 1991–2020, one set for Oct–Mar (top) and one "
        "for Apr–Sep (bottom), named by inspection. Frame colours pair cold and warm regimes whose patterns correlate "
        "≥ 0.6 - the same colours as the forecast figures. Bars: share of days in each regime by month, 1991–2020.",
        int(W * 13.5)), fontsize=10, va="top", color=RC.MUTED)
    name = f"geps_regimes_{sk}.webp"
    for d in OUTS:
        fig.savefig(d / name, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"  {name}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
