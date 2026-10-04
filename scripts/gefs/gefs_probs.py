#!/usr/bin/env python3
"""Weekly TERCILE probability maps for 2 m temperature and precipitation from the GEFS extended members, raw and
hindcast-calibrated, with the skill mask - the GEFS port of the GEPS page's prob_maps.py (2026-09-26; user: "the
GEFS pages should have all the same maps/charts the GEPS one does").

Anomaly     every member's weekly mean minus the GEFSv12 reforecast climatology at the INIT day of year and the same
            week of lead (release gefs-clim-v1, read as gefs_live.clim_at reads it), re-based to 1991-2020 by
            subtracting the ERA5 1991-2020 minus 2001-2020 normal difference averaged over the week's valid dates
            (era5_shift.npz; 2001-2020 stands in for the reforecast's 2000-2019) - the weekly maps' convention.
Categories  below / near / above normal: the terciles of the observed 7-day-mean anomaly for that time of year over
            the RECENT DECADE (2016-2025, build_gefs_mapskill.py recent), so the warming since 1991-2020 is not
            counted as a forecast; the anomalies themselves stay on 1991-2020. Truth: NCEP/NCAR R1 2 m air, CPC
            unified gauge precipitation (land only), both on the GEFS 1.5 deg grid.
Raw         the fraction of the members in each tercile.
Calibrated  GEPS's gridpoint regression obs = a*m + b, fitted on the 2000-2019 reforecast for this calendar month of
            the init and week (build_gefs_mapskill.py skill), intercept dropped, Gaussian residual - written in
            STANDARDISED form: x = (m - c) / sd_f, the ensemble-mean anomaly m as a departure from c, the recent-decade
            mean anomaly, in units of the hindcast ensemble mean's sd; the observed category then follows
            N(r x, 1 - r^2) against the standard normal's terciles +-0.4307. For a Gaussian variable this is GEPS's
            median + a(m - median) against q33/q67 (a = r sd_o / sd_f, s = sd_o sqrt(1 - r^2)); for weekly
            precipitation, whose terciles sit asymmetrically about the median, GEPS's form gives "below" ~40 %
            wherever there is no skill (its maps' weeks 3-5 read below normal almost everywhere) and centres the
            ensemble MEAN on the observed MEDIAN; here no skill (r <= 0) is exactly a third each, as GEPS's own footnote
            promises, and the neutral point is the mean. The intercept stays dropped (GEPS's reasoning: it is the
            hindcast-period mean state). The fit is on a 4-MEMBER hindcast mean and applied to the 31-member live mean,
            as GEPS applies its 4-member fit to 21: that errs toward climatology, never toward overconfidence.
            Tested out of sample on the reforecast (build_gefs_mapskill.py assess; numbers in gefs_probs.json):
            t2m RPSS +0.46/+0.24/+0.13/+0.10/+0.08 by week and reliable; pr +0.18/+0.04/+0.01/0/0 where GEPS's form
            scores -0.00/-0.04/-0.04/-0.05 from week 2.
Mask        hatched where the hindcast anomaly correlation for this month and week is below 0.25; cross-hatched where
            there is no gauge truth (precipitation over the ocean) or the climatological week is dry (< 0.5 mm/day:
            the terciles collapse onto zero and mean nothing).
Display     CPC-style: hue for the most likely tercile, intensity for its probability from 33 % up; precipitation
            brown below / green above.

    python gefs_probs.py --npz /tmp/gefs_work/gefs2_YYYYMMDD.npz --date YYYYMMDD --site . --cache /tmp/gefs_work/clim \
        --ref /tmp/gefs_work/ref
    -> SITE/assets/gefs/gefs_{na,sa,glb}_{t2m,pr}_prob_{raw,cal}.webp, SITE/assets/gefs/data/gefs_probs.json

Needs (--ref): gefs_mapskill_{t2m,pr}.npz, gefs_terciles_{t2m,pr}.npz, era5_shift.npz; (--cache) gefs_clim_DDD.npz,
downloaded from the release when missing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_skill as SK                                               # noqa: E402

CLIM_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-clim-v1/gefs_clim_{c:03d}.npz"
LAT, LON = SK.LAT, SK.LON
SKILL_MIN = SK.SKILL_MIN
DRY_MM = 0.5                        # climatological weekly mean below this: terciles undefined
P_LO, P_HI = 0.33, 0.70             # colour ramp of the most likely tercile
TAGS = {"t2m": "2 m temperature", "pr": "Precipitation"}
CATS = [("below", "Blues"), ("near", "Greys"), ("above", "Reds")]
CATS_BY_TAG = {"t2m": CATS, "pr": [("below", "YlOrBr"), ("near", "Greys"), ("above", "Greens")]}
# GEPS's domains (render_maps.DOMAINS) with its tighter North America window for the tercile maps
DOMAINS = {"na": dict(label="North America", extent=(-132, -58, 22, 62), central=-97, proj="lambert"),
           "sa": dict(label="South America", extent=(-95, -30, -58, 15), central=-62, proj="plate"),
           "glb": dict(label="Global", extent=(-180, 180, -60, 70), central=180, proj="merc", stack=True)}
Z3 = 0.4307272992954576            # the standard normal's upper tercile, Phi(Z3) = 2/3


def phi(x):
    """Standard normal CDF; NaN stays NaN."""
    from scipy.special import ndtr
    return ndtr(np.asarray(x, float))


# ── data ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def clim_at(init: dt.date, cache: Path, tags):
    """gefs-clim-v1 weekly climatology at the init day of year, linear between the two 5-day centres (gefs_live.clim_at's
    arithmetic; the files are shared with it through the same cache directory)."""
    i, wi, j, wj = SK.bracket(init.timetuple().tm_yday)
    out = {}
    for ci, wt in ((i, wi), (j, wj)):
        c = int(SK.CENTRES[ci])
        p = cache / f"gefs_clim_{c:03d}.npz"
        if not p.exists():
            cache.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(CLIM_URL.format(c=c), headers={"User-Agent": "scorvec-gefs"})
            with urllib.request.urlopen(req, timeout=120) as r:
                p.write_bytes(r.read())
        z = np.load(p)
        for t in tags:
            out[t] = out.get(t, 0) + wt * z[f"w_{t}"].astype("float64")
    return out


def weekly_probs(tag, z, base: dt.date, cache: Path, ref: Path):
    """Per week: raw and calibrated tercile probabilities (3, lat, lon), skill r, the dry mask."""
    F = z[f"w_{tag}"].astype("float64")                                # (M, 5, lat, lon)
    C = clim_at(base, cache, [tag])[tag]
    sh = SK.weekly_shift(np.load(ref / "era5_shift.npz")[f"s15_{tag}"].astype("float64"), base)
    A = F - C[None] - sh[None]                                         # member anomalies vs 1991-2020
    import gefs_drift
    dr = gefs_drift.from_env()                                         # operational drift the reforecast lacks
    if dr is not None:
        A = A - dr.weekly(tag)[None, :, :, None]
    terc = np.load(ref / f"gefs_terciles_{tag}.npz")
    sk = SK.load(tag, ref)
    if sk is None:
        raise SystemExit(f"need {ref / f'gefs_mapskill_{tag}.npz'}")
    mi = base.month - 1
    out = []
    for k in range(1, 6):
        wk = A[:, k - 1]
        centre = base + dt.timedelta(days=7 * k - 4)                   # middle of init + 7k-7 ... init + 7k-1
        doy = centre.timetuple().tm_yday
        q33, q67 = (SK.at_doy(terc[n].astype("float64"), doy) for n in ("q33", "q67"))
        ok = np.isfinite(q33) & np.isfinite(q67)
        raw = np.stack([np.where(ok, (wk < q33[None]).mean(0), np.nan),
                        np.where(ok, ((wk >= q33[None]) & (wk <= q67[None])).mean(0), np.nan),
                        np.where(ok, (wk > q67[None]).mean(0), np.nan)])
        m = wk.mean(0)
        r, fsd = (sk[n][mi, k - 1].astype("float64") for n in ("r", "fsd"))
        centre_now = SK.at_doy(terc["mean"].astype("float64"), doy)       # the recent-decade mean anomaly
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = np.clip(r, 0.0, 0.99)                                     # r <= 0: no skill, climatology
            x = (m - centre_now) / fsd                                     # standardised ensemble-mean departure
            den = np.sqrt(1.0 - rr ** 2)
            pb = phi((-Z3 - rr * x) / den)
            pa = 1.0 - phi((Z3 - rr * x) / den)
        cal = np.stack([pb, 1.0 - pb - pa, pa])
        cal = np.where(np.isfinite(cal) & ok[None], cal, np.nan)
        if tag == "pr":
            with np.errstate(invalid="ignore"):
                dry = SK.at_doy(terc["clim_weekly"].astype("float64"), doy) < DRY_MM
        else:
            dry = np.zeros(r.shape, bool)
        out.append(dict(k=k, raw=raw, cal=cal, r=r, dry=dry,
                        d0=base + dt.timedelta(days=7 * k - 7), d1=base + dt.timedelta(days=7 * k - 1)))
    return out


def most_likely(P, tol=1e-6):
    """Index of the most likely category (0 below, 1 near, 2 above) of P (3, ...). Categories within `tol` of the
    maximum count as tied, and a tie that includes near normal goes to near: the calibration's exact one-third where
    r <= 0 differs only by rounding, and a plain argmax painted every such cell "below"."""
    Q = np.where(np.isfinite(P), P, -1.0)
    top = Q.max(0)
    tied = Q >= top[None] - tol
    return np.where(tied[1], 1, np.where(tied[0], 0, 2))


# ── figures (prob_maps.draw, geometry from render_maps) ──────────────────────────────────────────────────────────
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["hatch.linewidth"] = 0.5
    import matplotlib.pyplot as plt
    return plt


def projection(dom):
    import cartopy.crs as ccrs
    k = dom.get("proj", "plate")
    if k == "lambert":
        return ccrs.LambertConformal(central_longitude=dom["central"], standard_parallels=(33, 45))
    if k == "merc":
        return ccrs.Mercator(central_longitude=dom["central"], min_latitude=dom["extent"][2],
                             max_latitude=dom["extent"][3])
    return ccrs.PlateCarree(central_longitude=dom["central"])


_ASP: dict = {}


def aspect(key, dom):
    """Height/width of the domain in the projection (render_maps.aspect; Mercator analytic)."""
    import cartopy.crs as ccrs
    plt = _plt()
    if key not in _ASP:
        if dom.get("proj") == "merc":
            f_ = lambda d: np.log(np.tan(np.pi / 4 + np.deg2rad(d) / 2))
            e = dom["extent"]
            _ASP[key] = (f_(e[3]) - f_(e[2])) / (2 * np.pi)
        else:
            f = plt.figure(figsize=(4, 4))
            a = f.add_subplot(1, 1, 1, projection=projection(dom))
            a.set_extent(dom["extent"], crs=ccrs.PlateCarree())
            x0, x1, y0, y1 = a.get_extent(projection(dom))
            _ASP[key] = abs((y1 - y0) / (x1 - x0))
            plt.close(f)
    return _ASP[key]


def _grid(stack, asp, n=5):
    """render_maps._grid: (fig, rects, cax_rect, note_y, title_y) for n weekly panels."""
    plt = _plt()
    if stack:
        panel_w, top_in, bot_in = 11.0, 0.72, 1.15
        map_h = panel_w * asp
        fig_w = panel_w + 0.35
        fig_h = n * (map_h + 0.30) + top_in + bot_in
        fig = plt.figure(figsize=(fig_w, fig_h))
        rects = [[0.015, 1.0 - (top_in + (i + 1) * (map_h + 0.30)) / fig_h, 0.968, map_h / fig_h] for i in range(n)]
        return fig, rects, [0.36, 0.62 / fig_h, 0.28, 0.13 / fig_h], 0.12 / fig_h, 1.0 - 0.18 / fig_h
    panel_w, top_in, bot_in, gap_y = 5.8, 0.92, 0.98, 0.44
    map_h = panel_w * asp
    rows = (n + 1) // 2
    fig_w = 2 * panel_w + 0.30
    fig_h = rows * map_h + (rows - 1) * gap_y + top_in + bot_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    w_frac = panel_w / fig_w
    rects = []
    for i in range(n):
        row, col = divmod(i, 2)
        y = 1.0 - (top_in + (row + 1) * map_h + row * gap_y) / fig_h
        x = (1.0 - w_frac) / 2 if (i == n - 1 and n % 2 == 1) else 0.010 + col * (w_frac + 0.006)
        rects.append([x, y, w_frac * 0.98, map_h / fig_h])
    return fig, rects, [0.36, 0.44 / fig_h, 0.28, 0.15 / fig_h], 0.06 / fig_h, 1.0 - 0.18 / fig_h


def draw(tag, base, weeks, kind, n_members, out: Path):
    plt = _plt()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import LinearSegmentedColormap
    title = TAGS[tag]
    lon_w = np.r_[LON, LON[0] + 360.0]
    wrap = lambda f: np.concatenate([f, f[..., :1]], axis=-1)
    cats = CATS_BY_TAG.get(tag, CATS)
    # the upper part of each ramp is dropped: a 70 % cell in full-strength Reds/Blues/Greys is nearly black
    cmaps = {c: LinearSegmentedColormap.from_list(cm + "_soft", plt.get_cmap(cm)(np.linspace(0.10, 0.62, 256)))
             for c, cm in cats}
    vmin, vmax = P_LO * 100 - 10, P_HI * 100 + 6
    names = []
    for key, dom in DOMAINS.items():
        proj = projection(dom)
        stack = dom.get("stack", False)
        fig, rects, cax_rect, note_y, title_y = _grid(stack, aspect(key + "_prob", dom), len(weeks))
        for i, (wk, rect) in enumerate(zip(weeks, rects)):
            ax = fig.add_axes(rect, projection=proj)
            if dom.get("proj") == "merc":
                ax.set_global()
                e = dom["extent"]
                ax.set_ylim(proj.transform_point(0.0, e[2], ccrs.PlateCarree())[1],
                            proj.transform_point(0.0, e[3], ccrs.PlateCarree())[1])
            else:
                ax.set_extent(dom["extent"], crs=ccrs.PlateCarree())
            P = wrap(wk[kind])                                           # (3, lat, lon+1)
            fin = np.isfinite(P).all(0)
            # most likely category; a tie that includes near normal goes to near - an exact three-way tie (the
            # calibration's one third each where r <= 0) would otherwise paint every no-skill cell "below"
            best = most_likely(P)
            pmax = np.max(np.where(np.isfinite(P), P, -1), axis=0)
            for ci, (cat, _) in enumerate(cats):
                layer = np.where((best == ci) & fin, pmax * 100.0, np.nan)
                ax.pcolormesh(lon_w, LAT, np.ma.masked_invalid(layer), cmap=cmaps[cat], vmin=vmin, vmax=vmax,
                              shading="nearest", transform=ccrs.PlateCarree(), rasterized=True)
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", alpha=0.45, zorder=2)
            r = wrap(wk["r"])
            noskill = (np.isfinite(r) & (r < SKILL_MIN)).astype(float)
            notruth = (~np.isfinite(r) | wrap(wk["dry"])).astype(float)
            for fld, hatch, col in ((noskill, "////", "#2f2c28"), (notruth, "xx", "#a6a29a")):
                if fld.any():
                    cs = ax.contourf(lon_w, LAT, fld, levels=[0.5, 1.5], colors="none", hatches=[hatch],
                                     transform=ccrs.PlateCarree(), zorder=3)
                    cs.set_edgecolor(col)
                    cs.set_linewidth(0.0)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.7, edgecolor="#1f1f1f", zorder=4)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, edgecolor="#3a3a3a", zorder=4)
            ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.25, edgecolor="#5a5a5a", zorder=4)
            ax.set_title((f"Week {wk['k']}   {wk['d0']:%b %d} – {wk['d1']:%b %d}" if stack
                          else f"Week {wk['k']}\n{wk['d0']:%b %d} – {wk['d1']:%b %d}"),
                         fontsize=10.5, fontweight="bold", pad=4, loc="left" if stack else "center")
        # three colour ramps, 33 -> 70 %, side by side where the single colorbar sits on the anomaly maps
        x0, y0, w, h = cax_rect
        gap = 0.02
        cw3 = (w + 0.12 - 2 * gap) / 3
        for ci, (cat, _) in enumerate(cats):
            cax = fig.add_axes([x0 - 0.06 + ci * (cw3 + gap), y0 + (0.006 if not stack else 0.004), cw3, h])
            sm = plt.cm.ScalarMappable(cmap=cmaps[cat], norm=plt.Normalize(vmin, vmax))
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[40, 50, 60, 70])
            cb.ax.set_xlim(P_LO * 100, P_HI * 100)
            cb.ax.xaxis.set_ticks_position("top")
            cb.ax.xaxis.set_label_position("bottom")
            cb.set_label(f"{cat} normal (%)", fontsize=9.2, labelpad=2)
            cb.ax.tick_params(labelsize=7.8, pad=1.2)
        fig.suptitle(f"{dom['label']} — {title}: most likely tercile by week · GEFS init {base:%Y-%m-%d} · "
                     f"{f'raw, {n_members} members' if kind == 'raw' else 'calibrated, 2000–2019 hindcast'}",
                     fontsize=12.5, fontweight="bold", y=title_y, va="top")
        note = (f"2016–2025 terciles · colour = most likely tercile, intensity = its probability "
                f"(33% = no signal) · hatched = hindcast r < {SKILL_MIN} (no skill)"
                + (" · cross-hatched = no gauge truth / dry week" if tag == "pr" else "")
                + (f" · raw: {100 / n_members:.0f}% steps" if kind == "raw" else " · calibrated: no skill → ⅓ each"))
        fig.text(0.5, 0.004, note, ha="center", va="bottom", fontsize=7.0, color="#5a5650")
        p = out / f"gefs_{key}_{tag}_prob_{kind}.webp"
        fig.savefig(p, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
        plt.close(fig)
        names.append(p.name)
        print(f"  {p.name}", flush=True)
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="gefs2_<date>.npz from gefs_live.fetch")
    ap.add_argument("--date", required=True, help="init YYYYMMDD (00Z)")
    ap.add_argument("--site", required=True)
    ap.add_argument("--cache", required=True, help="gefs-clim-v1 cache (shared with gefs_live)")
    ap.add_argument("--ref", default=None, help="reference files (default: CACHE/../ref)")
    a = ap.parse_args()
    base = SK.as_date(a.date)
    cache = Path(a.cache)
    ref = Path(a.ref) if a.ref else cache.parent / "ref"
    out = Path(a.site) / "assets" / "gefs"
    (out / "data").mkdir(parents=True, exist_ok=True)
    z = np.load(a.npz)
    M = int(len(z["members"]))
    summary = {"cycle": a.date, "init": f"{base:%Y-%m-%d} 00Z", "members": M, "skill_min": SKILL_MIN,
               "terciles": "2016-2025 observed weekly anomaly", "calibration": "GEFSv12 reforecast 2000-2019, 4 members",
               "fields": {}, "figures": []}
    for tag in TAGS:
        weeks = weekly_probs(tag, z, base, cache, ref)
        meta = json.loads(str(np.load(ref / f"gefs_mapskill_{tag}.npz")["meta"]))
        summary.setdefault("calibration_test", {})[tag] = meta.get("calibration_test")
        rows = []
        for wk in weeks:
            r = wk["r"]
            fin = np.isfinite(wk["cal"]).all(0) & ~wk["dry"]
            best = most_likely(wk["cal"])
            share = {c: round(float(np.mean(best[fin] == i)), 3) for i, (c, _) in enumerate(CATS)}
            rows.append({"week": wk["k"], "valid": f"{wk['d0']:%Y-%m-%d}/{wk['d1']:%Y-%m-%d}",
                         "skill_r_median": round(float(np.nanmedian(r)), 3),
                         "share_skilful": round(float(np.mean(r[np.isfinite(r)] >= SKILL_MIN)), 3),
                         "most_likely_cal": share})
            print(f"  {tag} week {wk['k']} {wk['d0']:%b %d}-{wk['d1']:%b %d}: skill r median {np.nanmedian(r):.2f}, "
                  f"share r>={SKILL_MIN}: {np.mean(r[np.isfinite(r)] >= SKILL_MIN) * 100:.0f}%, most likely (cal): "
                  + " ".join(f"{c} {v * 100:.0f}%" for c, v in share.items()), flush=True)
        summary["fields"][tag] = rows
        for kind in ("raw", "cal"):
            summary["figures"] += draw(tag, base, weeks, kind, M, out)
    (out / "data" / "gefs_probs.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
