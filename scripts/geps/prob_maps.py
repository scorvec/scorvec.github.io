#!/usr/bin/env python3
"""Weekly TERCILE probability maps for 2 m temperature and precipitation from
the 21 GEPS members, raw and hindcast-calibrated, with a skill mask.

Categories  below / near / above normal, the terciles of the 7-day-mean
            anomaly for that week of the year over the RECENT DECADE
            (2016-2025, recent_median.py). Against the 1991-2020 base the
            2026 maps read "above" nearly everywhere before the forecast
            says anything — warming, not weather — so the recent decade is
            the normal; the anomalies themselves stay on 1991-2020.
Raw         fraction of the 21 members in each tercile.
Calibrated  standardised form (since 2026-09-26): x = (m - mean) / fsd, the
            ensemble-mean departure from the recent-decade MEAN in units of
            the hindcast ensemble mean's sd (hindcast_maps.py, fitted on
            2001-2020 hindcasts for this calendar month and week); the
            observed category follows N(r x, 1 - r^2) against the standard
            normal's terciles +-0.4307. No skill (r <= 0) is exactly 1/3 each.
            The old form, median + a(m - median) against q33/q67, gave
            "below" ~40 % at nil skill for precipitation (its terciles sit
            unevenly about the median). The intercept stays dropped: it is
            the 2001-2020 mean state. A tie that includes near goes to near.
Display     CPC-style: hue for the most likely tercile (blue below, grey
            near, red above), intensity for its probability from 33% up.
Mask        hatched where the hindcast anomaly correlation for this month
            and week is below SKILL_MIN; cross-hatched where there is no
            truth (precipitation over the ocean — the CPC gauge analysis is
            land only) or where the climatological week is dry (< DRY_MM
            mm/day: the terciles collapse onto zero and mean nothing, the
            reason CPC masks dry regions in its precipitation outlooks).

Members are on the grid forecast.py kept them at (1 deg for these two
fields); the skill, threshold and re-basing fields are interpolated to it.

    python prob_maps.py --cycle 20260903
    -> figs/ and assets/geps/  geps_{dom}_{tag}_prob_{raw|cal}.webp
"""
from __future__ import annotations

import argparse
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 0.5
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths                                                    # noqa: E402
import render_maps as rm                                        # noqa: E402

TC = paths.TC_REF               # map_skill_<tag>.nc, recent_median_<tag>.nc
LIVE = paths.LIVE
FIGS = paths.FIGS
SITE = paths.SITE_ASSETS
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
SKILL_MIN = 0.25
DRY_MM = 0.5                    # climatological weekly mean below this: terciles undefined
P_LO, P_HI = 0.33, 0.70         # colour ramp of the most likely tercile
Z3 = 0.4307272992954576         # the standard normal's upper tercile, Phi(Z3) = 2/3


def most_likely(P, tol=1e-6):
    """The most likely category (0 below, 1 near, 2 above); a tie that includes near goes to near - a plain argmax
    painted every exact-one-third (no skill) cell "below"."""
    Q = np.where(np.isfinite(P), P, -1.0)
    tied = Q >= Q.max(0)[None] - tol
    return np.where(tied[1], 1, np.where(tied[0], 0, 2))
TAGS = {"t2m": ("2 m temperature", 1.0), "pr": ("Precipitation", 86400.0)}
CATS = [("below", "Blues"), ("near", "Greys"), ("above", "Reds")]
# precipitation reads dry/wet, not cold/warm: brown below, green above
CATS_BY_TAG = {"t2m": CATS, "pr": [("below", "YlOrBr"), ("near", "Greys"), ("above", "Greens")]}
# the tercile maps are read for the populated part of the continent: a tighter
# North America window than the anomaly maps use (which keep the Arctic and
# the Pacific for the circulation)
DOMAIN_OVERRIDE = {"na": dict(label="North America", extent=(-132, -58, 22, 62), central=-97,
                              proj="lambert")}
phi = np.vectorize(lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0))))


def members(tag, cycle, base):
    p = LIVE / f"geps_members_{tag}_{cycle}.nc"
    if not p.exists():
        return None
    a = xr.open_dataset(p)[f"{tag}_anom"].load()               # (L, number, lat, lon)
    if tag != "pr":
        dummy = xr.DataArray(np.zeros((366, a.latitude.size, a.longitude.size)),
                             dims=("L", "latitude", "longitude"),
                             coords={"L": np.arange(366), "latitude": a.latitude.values,
                                     "longitude": a.longitude.values})
        r = rm.reref(tag, dummy, pd.Timestamp("2000-01-01"))    # ERA5 1991-2020 re-basing, own grid
        if r is not None:
            valid = base + pd.to_timedelta(a.L.values - 1, unit="D")    # day L = the UTC day base + L - 1
            a = a + r.values.astype("float32")[valid.dayofyear.values - 1][:, None]
    return a * TAGS[tag][1]


def on_grid(da, lat, lon):
    """2.5 deg field -> member grid, seam wrapped; NaN stays where the source
    has no truth (nearest fills the bilinear edge, not the NaN interior)."""
    da = xr.concat([da, da.isel(lon=0).assign_coords(lon=360.0)], dim="lon")
    lin = da.interp(lat=lat, lon=lon)
    near = da.interp(lat=lat, lon=lon, method="nearest")
    return xr.where(np.isfinite(lin), lin, near).values


def weekly_probs(tag, cycle, base, skill, recent):
    a = members(tag, cycle, base)
    if a is None:
        return None
    day = a.L.values.astype(int)
    lat, lon = a.latitude.values, a.longitude.values
    out = []
    for wi, (w0, w1) in enumerate(WEEKS):
        sel = (day >= w0) & (day <= w1)
        wk = a.values[sel].mean(0)                                # (number, lat, lon)
        centre = base + pd.Timedelta(days=(w0 + w1) // 2 - 1)
        d = centre.dayofyear - 1
        mean_now, q33, q67, cw = (on_grid(recent[k].isel(doy=d), lat, lon)
                                  for k in ("mean_recent", "q33_recent", "q67_recent", "clim_weekly"))
        ok = np.isfinite(q33) & np.isfinite(q67)
        raw = np.stack([np.where(ok, (wk < q33[None]).mean(0), np.nan),
                        np.where(ok, ((wk >= q33[None]) & (wk <= q67[None])).mean(0), np.nan),
                        np.where(ok, (wk > q67[None]).mean(0), np.nan)])
        m = wk.mean(0)
        sk = skill.sel(month=base.month, week=wi + 1)
        R, FSD = (on_grid(sk[k], lat, lon) for k in ("r", "fsd"))
        # Standardised form (2026-09-26, found while porting to GEFS): the old median + a(m - median) against the
        # observed q33/q67 put ~40 % in "below" and ~20 % in "near" wherever skill is nil, because weekly
        # precipitation's terciles sit unevenly about its median - so weeks 3-5 read "below" almost everywhere,
        # against this script's own "no skill -> one third each". Now the ensemble-mean departure from the
        # recent-decade MEAN, in units of the hindcast mean's sd, predicts N(r x, 1 - r^2) against the standard
        # normal's terciles: r <= 0 is exactly a third each. For a Gaussian variable (t2m) it is the same forecast.
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = np.clip(R, 0.0, 0.99)
            x = (m - mean_now) / FSD
            den = np.sqrt(1.0 - rr ** 2)
            pb = phi((-Z3 - rr * x) / den)
            pa = 1.0 - phi((Z3 - rr * x) / den)
        cal = np.stack([pb, 1.0 - pb - pa, pa])
        cal = np.where(np.isfinite(cal) & ok[None], cal, np.nan)
        dry = (cw < DRY_MM) if tag == "pr" else np.zeros_like(cw, bool)
        out.append(dict(w0=w0, w1=w1, raw=raw, cal=cal, r=R, dry=dry, lat=lat, lon=lon,
                        ndays=int(sel.sum())))
    return out


def draw(tag, cycle, base, weeks, kind):
    title, _ = TAGS[tag]
    lat, lon = weeks[0]["lat"], weeks[0]["lon"]
    lon_w = np.r_[lon, lon[0] + 360.0]
    wrap = lambda f: np.concatenate([f, f[..., :1]], axis=-1)
    # the upper part of each ramp is dropped: a 70% cell in full-strength
    # Reds/Blues/Greys is nearly black and swallows the coastlines
    from matplotlib.colors import LinearSegmentedColormap
    cats = CATS_BY_TAG.get(tag, CATS)
    cmaps = {c: LinearSegmentedColormap.from_list(cm + "_soft", plt.get_cmap(cm)(np.linspace(0.10, 0.62, 256)))
             for c, cm in cats}
    vmin, vmax = P_LO * 100 - 10, P_HI * 100 + 6
    for key, dom0 in rm.DOMAINS.items():
        dom = DOMAIN_OVERRIDE.get(key, dom0)
        akey = key + ("_prob" if key in DOMAIN_OVERRIDE else "")
        asp = rm.aspect(akey, dom)
        proj = rm.projection(akey, dom)
        stack = dom.get("stack", False)
        fig, rects, cax_rect, note_y, title_y = rm._grid(stack, asp, len(weeks))
        for i, (wk, rect) in enumerate(zip(weeks, rects)):
            ax = fig.add_axes(rect, projection=proj)
            if dom.get("proj") == "merc":
                ax.set_global()
                e = dom["extent"]
                y0 = proj.transform_point(0.0, e[2], ccrs.PlateCarree())[1]
                y1 = proj.transform_point(0.0, e[3], ccrs.PlateCarree())[1]
                ax.set_ylim(y0, y1)
            else:
                ax.set_extent(dom["extent"], crs=ccrs.PlateCarree())
            P = wrap(wk[kind])                                    # (3, lat, lon+1)
            fin = np.isfinite(P).all(0)
            best = most_likely(P)
            pmax = np.max(np.where(np.isfinite(P), P, -1), axis=0)
            for ci, (cat, _) in enumerate(cats):
                layer = np.where((best == ci) & fin, pmax * 100.0, np.nan)
                ax.pcolormesh(lon_w, lat, np.ma.masked_invalid(layer), cmap=cmaps[cat],
                              vmin=vmin, vmax=vmax, shading="nearest",
                              transform=ccrs.PlateCarree(), rasterized=True)
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", alpha=0.45, zorder=2)
            r = wrap(wk["r"])
            noskill = (np.isfinite(r) & (r < SKILL_MIN)).astype(float)
            notruth = (~np.isfinite(r) | wrap(wk["dry"])).astype(float)
            for fld, hatch, col in ((noskill, "////", "#2f2c28"), (notruth, "xx", "#a6a29a")):
                if fld.any():
                    cs = ax.contourf(lon_w, lat, fld, levels=[0.5, 1.5], colors="none", hatches=[hatch],
                                     transform=ccrs.PlateCarree(), zorder=3)
                    for c in getattr(cs, "collections", [cs]):
                        c.set_edgecolor(col); c.set_linewidth(0.0)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.7, edgecolor="#1f1f1f", zorder=4)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, edgecolor="#3a3a3a", zorder=4)
            ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.25, edgecolor="#5a5a5a", zorder=4)
            d0 = base + pd.Timedelta(days=wk["w0"] - 1); d1 = base + pd.Timedelta(days=wk["w1"] - 1)
            ax.set_title((f"Week {i+1}   {d0:%b %d} – {d1:%b %d}" if stack
                          else f"Week {i+1}\n{d0:%b %d} – {d1:%b %d}"),
                         fontsize=10.5, fontweight="bold", pad=4, loc="left" if stack else "center")
        # three colour ramps, 33 -> 70 %, side by side where the single colorbar sat
        x0, y0, w, h = cax_rect
        gap = 0.02
        cw3 = (w + 0.12 - 2 * gap) / 3
        for ci, (cat, _) in enumerate(cats):
            cax = fig.add_axes([x0 - 0.06 + ci * (cw3 + gap), y0 + (0.006 if not stack else 0.004), cw3, h])
            sm = plt.cm.ScalarMappable(cmap=cmaps[cat], norm=plt.Normalize(vmin, vmax))
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[40, 50, 60, 70])
            cb.ax.set_xlim(P_LO * 100, P_HI * 100)
            cb.ax.xaxis.set_ticks_position("top")           # ticks above the bar, name below:
            cb.ax.xaxis.set_label_position("bottom")        # nothing then collides with the map
            cb.set_label(f"{cat} normal (%)", fontsize=9.2, labelpad=2)
            cb.ax.tick_params(labelsize=7.8, pad=1.2)
        fig.suptitle(f"{dom['label']} — {title}: most likely tercile by week · GEPS init {base:%Y-%m-%d} · "
                     f"{'raw, 21 members' if kind == 'raw' else 'calibrated, 2001–2020 hindcast'}",
                     fontsize=12.5, fontweight="bold", y=title_y, va="top")
        note = (f"2016–2025 terciles · colour = most likely tercile, intensity = its probability "
                f"(33% = no signal) · hatched = hindcast r < {SKILL_MIN} (no skill)"
                + (" · cross-hatched = no gauge truth / dry week" if tag == "pr" else "")
                + (" · raw: 5% steps" if kind == "raw" else " · calibrated: no skill → ⅓ each"))
        fig.text(0.5, 0.004, note, ha="center", va="bottom", fontsize=7.0, color="#5a5650")
        for d in (FIGS, SITE):
            fig.savefig(d / f"geps_{key}_{tag}_prob_{kind}.webp", dpi=105, facecolor="white",
                        pil_kwargs={"quality": 88, "method": 6})
        plt.close(fig)
        print(f"  geps_{key}_{tag}_prob_{kind}.webp")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    a = ap.parse_args()
    base = pd.Timestamp(a.cycle)
    for tag in TAGS:
        q = TC / f"map_skill_{tag}.nc"
        qr = TC / f"recent_median_{tag}.nc"
        if not q.exists() or not qr.exists():
            print(f"  {tag}: need map_skill and recent_median files"); continue
        skill, recent = xr.open_dataset(q), xr.open_dataset(qr)
        weeks = weekly_probs(tag, a.cycle, base, skill, recent)
        if weeks is None:
            print(f"  {tag}: no member file for {a.cycle}"); continue
        for wk in weeks:
            r = wk["r"]
            fin = np.isfinite(wk["cal"]).all(0)
            best = most_likely(wk["cal"])
            share = {c: float(np.mean(best[fin] == i)) for i, (c, _) in enumerate(CATS)}
            print(f"  {tag} week {wk['w0']}-{wk['w1']}: skill r median {np.nanmedian(r):.2f}, "
                  f"share r>={SKILL_MIN}: {np.nanmean(r >= SKILL_MIN)*100:.0f}%, most likely (cal): "
                  + " ".join(f"{c} {v*100:.0f}%" for c, v in share.items()))
        for kind in ("raw", "cal"):
            draw(tag, a.cycle, base, weeks, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
