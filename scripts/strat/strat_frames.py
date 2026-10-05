#!/usr/bin/env python3
"""Daily 10/100 hPa anomaly frames for the site's animator, out to day 35.

The summary board in strat_maps.py samples four leads; this walks every forecast
day so the vortex can be watched evolving, which is the whole point during a
warming event — the useful signal is a wave-1 ridge marching over the pole and
the zero-wind line collapsing inward, and four snapshots miss it.

One frame is a three-panel row (temperature, zonal wind, height anomaly) for one
hemisphere and level, so a single loop shows all three fields together rather
than forcing three separate animations that have to be watched side by side.
Region ids are `<hemi><level>` — nh10, nh100, sh10, sh100 — flat, because the
site's player only routes a base ending in /anim to the frames branch.

Frames share ONE colour scale across all leads (computed over the whole run),
so a loop shows the anomaly actually growing and decaying instead of every frame
being renormalised to look equally dramatic.

    python strat_frames.py                  # both hemispheres, both levels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyproj  # noqa: F401,E402  before xarray/eccodes: eccodes preloads its own libproj (CLAUDE.md gotcha)
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strat_maps as sm                                        # noqa: E402
from strat_maps import demean_height                           # noqa: E402

OUT = sm.OUT / "anim"
LEADS = list(range(0, 841, 24))
MAPBIAS = sm.DATA / "geps_s2s"


def map_bias(hemi, base, halfwin=8):
    """The GEPS drift of the anomaly maps, from the ECMWF S2S archive's GEPS reforecasts (geps_s2s_hindcast.py mapbias;
    2026-09-27, user: "Now that we have the 10mb hindcasts for the geps, can we update the stratosphere maps to look
    like the [GEFS] version with height and temp anomaly plots?").

    The reforecasts DO carry 10 hPa, which the GEPS8 hindcast behind the rest of the page does not - the reason 10 hPa
    was drawn absolute. Per start date: zonal-mean (model - ERA5 reference) by latitude and lead, 20 years x 4 members.
    Start dates within +/-halfwin days of the init's day of year are pooled with triangular weights and the result is
    run-meaned over 3 days of lead. Returns f(key, L, clat) -> the zonal-mean bias on the climatology latitudes (key
    t10/t100/z10/z100, L in hours; the analysis uses day 1, the first reforecast lead), or None when no start date is
    close enough - then 10 hPa falls back to the absolute view."""
    f = MAPBIAS / f"mapbias_{hemi}.json"
    if not f.exists():
        return None
    doy = base.dayofyear
    picks = []
    for d, e in json.loads(f.read_text())["starts"].items():
        gap = abs(pd.Timestamp(d).dayofyear - doy); gap = min(gap, 365 - gap)
        if gap <= halfwin:
            picks.append((1.0 - gap / (halfwin + 1.0), e))
    if not picks:
        return None
    steps = np.asarray(picks[0][1]["step_h"], float); lat = np.asarray(picks[0][1]["lat"], float)
    wsum = sum(w for w, _ in picks)
    tab = {}
    for k in ("t10", "t100", "z10", "z100"):
        b = sum(w * np.asarray(e[k], float) for w, e in picks) / wsum            # [step, lat]
        pad = np.concatenate([b[:1], b, b[-1:]])
        tab[k] = (pad[:-2] + pad[1:-1] + pad[2:]) / 3.0
    o = np.argsort(lat)

    def get(key, L, clat):
        b = tab[key]
        row = np.array([np.interp(max(L, steps[0]), steps, b[:, j]) for j in range(b.shape[1])])
        return np.interp(clat, lat[o], row[o])
    get.n_starts = len(picks)
    get.starts = sorted(d for d, e in json.loads(f.read_text())["starts"].items()
                        if min(abs(pd.Timestamp(d).dayofyear - doy), 365 - abs(pd.Timestamp(d).dayofyear - doy)) <= halfwin)
    return get


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--cyc", default="00")
    ap.add_argument("--clim", default=str(sm.CLIM))
    a = ap.parse_args()
    c = xr.open_dataset(a.clim)
    tr = xr.open_dataset(sm.TREND) if sm.TREND.exists() else None
    clat, clon = c.lat.values, c.lon.values

    date = a.date or sm.latest_extended(a.cyc)
    base = pd.Timestamp(f"{date} {a.cyc}:00")
    print(f"GEPS extended {base:%Y-%m-%d %H}Z", flush=True)
    regions = {}

    for hemi in ("nh", "sh"):
        north = hemi == "nh"
        proj = (ccrs.NorthPolarStereo() if north else ccrs.SouthPolarStereo())
        lat0, lat1 = (20, 90) if north else (-90, -20)
        phi = 60 if north else -60
        sub = (clat >= lat0) & (clat <= lat1)
        capm = (clat >= 65) if north else (clat <= -65)
        wc = np.cos(np.deg2rad(clat[capm]))
        mb = map_bias(hemi, base)
        anom_levels = set(sm.ANOM_LEVELS) | ({10, 100} if mb else set())
        print(f"  {hemi}: " + (f"reforecast drift removed ({mb.n_starts} start dates: {', '.join(mb.starts)})" if mb
                                else "no reforecast start date within 8 days - 10 hPa absolute, 100 hPa uncorrected"), flush=True)
        for lev in (10, 100):
            # pass 1: build every lead, so one colour scale can cover the run
            flds = sm.FIELDS if lev in anom_levels else sm.fields_for(lev)
            per = {}
            for key, var, ckey, label, unit, cmap, mode0 in flds:
                mode = mode0 if (mode0 == "speed" or lev in anom_levels) else "abs"
                seq, lim, vmin = [], 0.0, np.inf
                for L in LEADS:
                    fz = sm.sm_fetch_mean(date, a.cyc, "HGT", lev, L)
                    if fz is None:
                        seq.append(None); continue
                    valid = base + pd.Timedelta(hours=L)
                    gz = sm.to_clim_grid(fz, clat, clon).values
                    if mode == "speed":
                        fu = sm.sm_fetch_mean(date, a.cyc, "UGRD", lev, L)
                        fv = sm.sm_fetch_mean(date, a.cyc, "VGRD", lev, L)
                        if fu is None or fv is None:
                            seq.append(None); continue
                        u = sm.to_clim_grid(fu, clat, clon).values
                        v = sm.to_clim_grid(fv, clat, clon).values
                        fld = np.hypot(u, v) * sm.MS2KT
                        lim = max(lim, float(np.nanpercentile(fld[sub], 99)))
                        seq.append((fld, valid, gz, None,
                                    float(np.interp(phi, clat[::-1], u.mean(1)[::-1])), u))
                    else:
                        src = fz if key == "Z" else sm.sm_fetch_mean(date, a.cyc, var, lev, L)
                        if src is None:
                            seq.append(None); continue
                        fc = sm.to_clim_grid(src, clat, clon)
                        if mode == "abs":
                            an = fc.values
                            vmin = min(vmin, float(np.nanpercentile(an[sub], 1)))
                            lim = max(lim, float(np.nanpercentile(an[sub], 99)))
                        else:
                            ref, _ = sm.clim_at(c, ckey, lev, valid, tr)
                            an = fc.values - ref.values
                            if mb:
                                an = an - mb(f"{ckey.lower()}{lev}", L, clat)[:, None]
                            an = sm.demean_height(key, an, clat, north)
                            lim = max(lim, float(np.nanpercentile(np.abs(an[sub]), 99)))
                        cav = float((an[capm].mean(1) * wc).sum() / wc.sum())
                        seq.append((an, valid, gz, cav, None, None))
                per[key] = (seq, lim, label, unit, cmap, mode, vmin)
            rid = f"{hemi}{lev}"
            d = OUT / rid
            d.mkdir(parents=True, exist_ok=True)
            frames = []
            for i, L in enumerate(LEADS):
                if any(per[f[0]][0][i] is None for f in flds):
                    continue
                nc = len(flds)
                fig = plt.figure(figsize=(3.9 * nc + 0.2, 5.55))
                span = 1.0 / nc
                for col, (key, var, ckey, label, unit, cmap, mode0) in enumerate(flds):
                    seq, lim, lab, unit, cmap, mode, vmin = per[key]
                    fld, valid, gz, cav, u60, uu = seq[i]
                    ax = fig.add_axes([0.014 + col * span, 0.145, span * 0.945, 0.755],
                                      projection=proj)
                    ax.set_extent([-180, 180, lat0, lat1], crs=ccrs.PlateCarree())
                    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4,
                                   edgecolor="#555")
                    ax.gridlines(lw=0.3, color="#bbb", ylocs=[phi], draw_labels=False)
                    fc_, lo_ = sm.cyc(fld, clon)
                    gz_, _ = sm.cyc(gz, clon)
                    if mode == "speed":
                        uu_, _ = sm.cyc(uu, clon)
                        msh = ax.contourf(lo_, clat, fc_, levels=sm.WLEV, cmap=sm.WCMAP,
                                          norm=sm.WNORM, extend="both",
                                          transform=ccrs.PlateCarree())
                        ax.contour(lo_, clat, gz_, levels=10, colors="#1a1a1a",
                                   linewidths=0.55, transform=ccrs.PlateCarree())
                        ax.contourf(lo_, clat, uu_, levels=[-1e9, 0.0], colors="none",
                                    hatches=["//"], transform=ccrs.PlateCarree())
                        ax.contour(lo_, clat, uu_, levels=[0], colors="#c2410c",
                                   linewidths=1.1, transform=ccrs.PlateCarree())
                    elif key == "Z" and mode == "abs":
                        # Absolute height as CONTOUR LINES only (2026-09-25, user: "just use contour lines for the gph
                        # not contour fill"): a fixed interval for the whole loop so lines are comparable frame to
                        # frame, every other line heavier and labelled.
                        from matplotlib.ticker import MaxNLocator
                        lv = MaxNLocator(nbins=14, steps=[1, 2, 2.5, 4, 5, 8, 10]).tick_values(vmin, lim)
                        ci = float(lv[1] - lv[0]) if len(lv) > 1 else 0.0
                        heavy = lv[::2]; light = [v for v in lv if v not in set(heavy)]
                        if light:
                            ax.contour(lo_, clat, fc_, levels=light, colors="#3a3a3a", linewidths=0.6,
                                       transform=ccrs.PlateCarree())
                        cs = ax.contour(lo_, clat, fc_, levels=heavy, colors="#111", linewidths=1.25,
                                        transform=ccrs.PlateCarree())
                        ax.clabel(cs, fmt="%d", fontsize=6.5, inline=True, inline_spacing=2)
                        msh = None
                    else:
                        from matplotlib.ticker import MaxNLocator
                        lv = (MaxNLocator(nbins=20).tick_values(vmin, lim)
                              if mode == "abs" else sm.nice_levels(lim))
                        msh = ax.contourf(lo_, clat, fc_, levels=lv,
                                          cmap=("viridis" if mode == "abs" else cmap),
                                          extend="both", transform=ccrs.PlateCarree())
                        if key == "Z":
                            ax.contour(lo_, clat, gz_, levels=8, colors="#333",
                                       linewidths=0.45, transform=ccrs.PlateCarree())
                    ax.set_title(lab + ("" if mode != "anom" else " anomaly"),
                                 fontsize=11, fontweight="bold", pad=4)
                    note = (f"u(60{'N' if north else 'S'}) {u60 * sm.MS2KT:+.0f} kt" if u60 is not None
                            else (f"cap {cav:.0f}" if mode == "abs" else f"cap {cav:+.1f}")
                            + (" m" if key == "Z" else " K"))
                    ax.text(0.5, -0.055, note, transform=ax.transAxes, ha="center",
                            va="top", fontsize=8.5, color="#2c4a72", fontweight="bold")
                    if msh is None:                                  # contour-line panel: an interval note, no colour bar
                        fig.text(0.014 + col * span + span * 0.475, 0.082, f"contours every {ci:.0f} {unit}",
                                 ha="center", va="center", fontsize=8, color="#333")
                    else:
                        cax = fig.add_axes([0.014 + col * span + span * 0.16, 0.072,
                                            span * 0.63, 0.020])
                        cb = fig.colorbar(msh, cax=cax, orientation="horizontal", extend="both",
                                          ticks=sm.WLEV[::2] if mode == "speed"
                                          else msh.levels[::4])
                        cb.set_label(unit, fontsize=8, labelpad=1)
                        cb.ax.tick_params(labelsize=7, pad=1)
                valid = base + pd.Timedelta(hours=L)
                fig.suptitle(f"GEPS · {'Northern' if north else 'Southern'} polar vortex — "
                             f"{lev} hPa · "
                             + ("analysis" if LEADS[i] == 0 else f"day {LEADS[i]//24}")
                             + f", valid {valid:%a %d %b}"
                             + (" · reforecast drift removed" if mb and lev in anom_levels else ""),
                             fontsize=12.5, fontweight="bold", y=0.988, va="top")
                fp = d / f"F{i:02d}.webp"
                # loop frames: WebP q65 method 6 (2026-10-05 encode study: ~30-35 % fewer bytes than q82, text/contours/colour bars unchanged at 1x)
                fig.savefig(fp, dpi=100, facecolor="white",
                            pil_kwargs={"quality": 65, "method": 6})
                plt.close(fig)
                frames.append({"idx": i, "file": fp.name,
                               "date": valid.strftime("%Y-%m-%d"),
                               "label": ("analysis" if L == 0 else f"day {L//24}")
                                        + f" · {valid:%b %d}"})
            regions[rid] = {"label": f"{'NH' if north else 'SH'} {lev} hPa",
                            "frames": frames}
            print(f"  {rid}: {len(frames)} frames", flush=True)

    order = ["nh10", "nh100", "sh10", "sh100"]
    man = {"ver": int(pd.Timestamp.now().timestamp()), "selectorLabel": "Level",
           "regions": {k: regions[k] for k in order if k in regions},
           "default": "nh10"}
    (OUT / "geps_strat_manifest.json").write_text(json.dumps(man))
    print(f"  geps_strat_manifest.json: {len(man['regions'])} regions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
