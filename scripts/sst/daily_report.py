#!/usr/bin/env python3
"""Colombia hydro — daily 00Z PDF briefing.

Pages 1-6: gauge-corrected IMERG rainfall over the hydro basins, one
accumulation window per page (yesterday, 7 d, 14 d, 30 d, 90 d, since
Jan 1) — total on the left, % of normal on the right, big uncramped
maps, gauge ground truth overlaid on every total map.

Output:
  colombia_hydro/report/colombia_hydro_daily.pdf       (latest, public)
  ~/colombia_hydro/reports/colombia_hydro_YYYYMMDD.pdf (dated archive)

    python scripts/sst/daily_report.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                   # noqa: E402
import colombia_rain_map as CRM                             # noqa: E402
from build_imerg_clim import OUT as CLIM_NC, eval_clim      # noqa: E402
from hydro_region_rain import gauge_correction              # noqa: E402

REPO = HERE.parent.parent
PRIV = Path.home() / "colombia_hydro"
OUT_PDF = REPO / "colombia_hydro" / "report" / "colombia_hydro_daily.pdf"
ARCHIVE = Path.home() / "colombia_hydro" / "reports"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]

NAVY = "#13273d"
INK = "#1a2733"
MUTE = "#5a6b7a"

# rainfall ramp — quiet low end, saturated top, no muddy midtones
RAIN_COLS = ["#f6f7f5", "#d9edcf", "#a5d99b", "#57b86b", "#1f9e89",
             "#2380b9", "#20539c", "#5b3f9e", "#93357f", "#c2185b"]
RAIN_CMAP = ListedColormap(RAIN_COLS)
RAIN_CMAP.set_over("#7a1240")
# % of normal — browns dry, near-white 90-110, greens to blue-purple wet
PCT_COLS = ["#7a4a12", "#a8702a", "#cd9d57", "#e8cf9e", "#f6f5f0",
            "#c9e7c2", "#7cc87c", "#2e9e4f", "#1d6fb8", "#6a3d9a"]
PCT_CMAP = ListedColormap(PCT_COLS)
PCT_CMAP.set_over("#3f1f66")
LEV_PCT = [0, 25, 50, 75, 90, 110, 125, 150, 200, 300, 1000]
# 1-day anomaly in mm (daily %-of-normal is near-binary and unreadable)
ANOM_COLS = ["#6a3b0f", "#a8702a", "#d3aa66", "#eed9ac", "#f6f5f0",
             "#cfe6f0", "#8fc4de", "#4292c6", "#1c5fa8", "#5b3f9e"]
ANOM_CMAP = ListedColormap(ANOM_COLS)
ANOM_CMAP.set_over("#3f1f66")
ANOM_CMAP.set_under("#452508")
LEV_ANOM_1D = [-100, -50, -25, -10, -3, 3, 10, 25, 50, 100, 200]

LEVS = {1:   [0, 1, 2, 5, 10, 20, 35, 50, 75, 100, 150],
        7:   [0, 5, 10, 25, 50, 75, 100, 150, 200, 300, 450],
        14:  [0, 10, 25, 50, 100, 150, 200, 300, 400, 500, 700],
        30:  [0, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1200],
        90:  [0, 50, 100, 200, 300, 450, 600, 800, 1100, 1500, 2200],
        "ytd": [0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000]}


def draw_map(ax, lons, lats, field, cmap, norm, rp, gauges=None,
             mask_ocean=False, decim=0.14):
    pm = ax.pcolormesh(lons, lats, field, cmap=cmap, norm=norm,
                       shading="nearest", transform=ccrs.PlateCarree(),
                       rasterized=True)
    if mask_ocean:
        ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                       facecolor="#e6ebf0", zorder=3)
    ax.coastlines(resolution="50m", lw=0.9, color="#2b2b2b", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.7,
                   edgecolor="#4a4a4a", zorder=4)
    for r, rings in rp.items():
        if r not in ORDER:
            continue
        col = CRM.RCOL[r]
        for ring in rings:
            ax.plot(ring[:, 0] % 360, ring[:, 1], color=col, lw=2.0,
                    transform=ccrs.PlateCarree(), zorder=5,
                    path_effects=[pe.withStroke(linewidth=3.2,
                                                foreground="white")])
        big = max(rings, key=len)
        cx, cy = (big[:, 0] % 360).mean(), big[:, 1].mean()
        ax.annotate(r, (cx - 360, cy), ha="center", fontsize=8.6,
                    fontweight="bold", color=col, zorder=7,
                    xycoords=ax.transData,
                    path_effects=[pe.withStroke(linewidth=2.6,
                                                foreground="white")])
    n_in = 0
    if gauges:
        # display decimation: one gauge per ~0.14 deg cell so dense city
        # clusters never merge into a blob that hides the field; sensor
        # sanity: drop gauges wildly above the satellite at their own cell
        # (stuck/cumulative counters)
        seen = set()
        for g in sorted(gauges.values(), key=lambda x: -x["mm"]):
            lo, la, mm = g["lo"] % 360, g["la"], g["mm"]
            if not (CRM.LON0 <= lo <= CRM.LON1 and CRM.LAT0 <= la <= CRM.LAT1):
                continue
            sat = field[int(np.argmin(np.abs(lats - la))),
                        int(np.argmin(np.abs(lons - lo)))]
            if np.isfinite(sat) and mm > 4.0 * (sat + 25.0):
                continue
            key = (round(lo / decim), round(la / decim))
            if key in seen:
                continue
            seen.add(key)
            n_in += 1
            ax.scatter([lo], [la], c=[mm], cmap=cmap, norm=norm, s=26,
                       edgecolors="black", linewidths=0.55, zorder=5,
                       transform=ccrs.PlateCarree())
            ax.annotate(f"{mm:.0f}", (lo - 360, la), xytext=(0, 3.8),
                        textcoords="offset points", ha="center", fontsize=6.0,
                        fontweight="bold", color="white", zorder=6,
                        xycoords=ax.transData,
                        path_effects=[pe.withStroke(linewidth=1.6,
                                                    foreground="black")])
    ax.set_extent([CRM.LON0 - 360, CRM.LON1 - 360, CRM.LAT0, CRM.LAT1],
                  ccrs.PlateCarree())
    for sp in ax.spines.values():
        sp.set_edgecolor("#9aa7b2")
        sp.set_linewidth(0.8)
    return pm, n_in


_PAGE = [0]                      # running page number, stamped by header()


def header(fig, when, subtitle, right2):
    """`right2` is the units/context line; the page number is appended
    automatically. Hardcoding it meant every reordering silently left stale
    labels - the national page shipped as "page 4" while sitting tenth."""
    _PAGE[0] += 1
    right2 = f"{right2} \u00b7 page {_PAGE[0]}"
    hd = fig.add_axes([0, 0.925, 1, 0.075])
    hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes,
                               facecolor=NAVY, edgecolor="none"))
    hd.text(0.035, 0.58, "COLOMBIA HYDRO — DAILY BRIEFING",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    hd.text(0.035, 0.16, subtitle, transform=hd.transAxes, color="#b9c6d4",
            fontsize=7.8, va="center")
    hd.text(0.965, 0.58, f"{when:%A, %B %d, %Y}", transform=hd.transAxes,
            color="white", fontsize=11.5, fontweight="bold",
            va="center", ha="right")
    hd.text(0.965, 0.16, right2, transform=hd.transAxes, color="#b9c6d4",
            fontsize=7.8, va="center", ha="right")


def footer(fig, note=""):
    fig.text(0.035, 0.012, "scorvec.com/colombia_hydro", fontsize=7, color=MUTE)
    if note:
        fig.text(0.5, 0.012, note, fontsize=7, color=MUTE, ha="center")
    fig.text(0.965, 0.012,
             f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
             fontsize=7, color=MUTE, ha="right")


def rain_pages(pdf, when: datetime) -> str:
    # crop to basin bbox exactly as the site maps do
    rp = CRM.region_paths()
    xs = np.concatenate([r[:, 0] % 360 for reg, rr in rp.items()
                         if reg in ORDER for r in rr])
    ys = np.concatenate([r[:, 1] for reg, rr in rp.items()
                         if reg in ORDER for r in rr])
    # natural portrait aspect (no squaring) — two tall maps fill the page
    CRM.LAT0, CRM.LAT1 = ys.min() - 0.45, ys.max() + 0.45
    CRM.LON0, CRM.LON1 = xs.min() - 0.45, xs.max() + 0.45

    end = when
    jan1 = datetime(when.year, 1, 1)
    nback = (end - jan1).days + 8
    days = [end - timedelta(days=k) for k in range(nback)][::-1]
    IP.ensure_daily(set(days[-35:]))
    fields, lons, lats = CRM.imerg_stack(days)
    have = [i for i, f in enumerate(fields) if f is not None]
    latest_i = have[-1]
    through = days[latest_i]
    F = gauge_correction(lons, lats)     # embeds the cached field on any crop

    import xarray as xr
    coef = xr.open_dataset(CLIM_NC)["coef"].values
    # crop indices matching imerg_stack's subset (clim fields are full-grid)
    ml, mt = IP._grid_axes()
    _lo = np.sort(IP._LON[ml] % 360)
    _la = np.sort(IP._LAT[mt])
    _li = (_lo >= CRM.LON0) & (_lo <= CRM.LON1)
    _ti = (_la >= CRM.LAT0) & (_la <= CRM.LAT1)
    clim_by_doy = {}

    def clim_day(d):
        dy = min(d.timetuple().tm_yday, 365)
        if dy not in clim_by_doy:
            clim_by_doy[dy] = eval_clim(coef, dy)[np.ix_(_ti, _li)]
        return clim_by_doy[dy]

    print("fetching gauges …", flush=True)
    gauges = CRM.fetch_range(end, nback)

    ytd_start = next(i for i, d in enumerate(days) if d >= jan1)

    windows = [
        ("Yesterday", 1), ("Last 7 days", 7), ("Last 14 days", 14),
        ("Last 30 days", 30), ("Last 90 days", 90), ("Since January 1", "ytd"),
    ]
    for pageno, (label, wkey) in enumerate(windows, start=1):
        if wkey == "ytd":
            idx = [i for i in have if i >= ytd_start and i <= latest_i]
        else:
            idx = [i for i in have if i <= latest_i][-wkey:]
        tot = np.nansum([fields[i] for i in idx], axis=0) * F
        cl = np.sum([clim_day(days[i]) for i in idx], axis=0) * F
        if wkey == 1:
            pct = tot - cl                       # mm anomaly — % is binary daily
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(cl > 1.0, 100.0 * tot / cl, np.nan)
        d0 = days[idx[0]]

        # gauge sums over the same days (paired; coverage threshold relaxes
        # for the long windows so slightly gappy stations still qualify)
        win_days = [days[i] for i in idx if gauges.get(days[i])]
        gg = None
        if win_days:
            need = max(1, int((0.85 if len(idx) <= 30 else 0.7) * len(win_days)))
            acc, cnt, meta = {}, {}, {}
            for d in win_days:
                for code, g in gauges[d].items():
                    acc[code] = acc.get(code, 0.0) + g["mm"]
                    cnt[code] = cnt.get(code, 0) + 1
                    meta[code] = (g["la"], g["lo"])
            gg = {c: {"la": meta[c][0], "lo": meta[c][1], "mm": acc[c]}
                  for c in acc if cnt[c] >= need}

        lev = LEVS[wkey]
        fig = plt.figure(figsize=(11.69, 8.27))
        header(fig, when,
               "Basin rainfall · GPM IMERG corrected by ~500 IDEAM gauges "
               "(43% station error reduction, held-out validation)",
               f"satellite through {through:%b %d} · page {pageno}")
        span = (f"{d0:%b %d} – {through:%b %d}"
                if len(idx) > 1 else f"{through:%b %d}")
        if wkey == 1:
            right = (pct, ANOM_CMAP, LEV_ANOM_1D, "Anomaly (mm vs climatology)")
        else:
            right = (pct, PCT_CMAP, LEV_PCT, "Percent of normal")
        # size axes to the map aspect exactly and center the pair
        H = 0.775
        aspect = (CRM.LON1 - CRM.LON0) / (CRM.LAT1 - CRM.LAT0)
        Wp = H * 8.27 / 11.69 * aspect
        gapx = 0.045
        xleft = (1.0 - (2 * Wp + gapx)) / 2
        for k, (field, cmap, levk, ttl, g_) in enumerate([
                (tot, RAIN_CMAP, lev, f"{label} — rainfall (mm)", gg),
                (*right, None)]):
            x0 = xleft + k * (Wp + gapx)
            norm = BoundaryNorm(levk, cmap.N)
            ax = fig.add_axes([x0, 0.105, Wp, H],
                              projection=ccrs.PlateCarree())
            dec = (0.14 if wkey in (1, 7) else
                   0.18 if wkey == 14 else
                   0.24 if wkey == 30 else
                   0.34 if wkey == 90 else 0.44)
            pm, n_in = draw_map(ax, lons, lats, field, cmap, norm, rp, g_,
                                mask_ocean=(k == 1), decim=dec)
            ttl_full = (f"{ttl}   ·   {span}   ·   {n_in} gauges" if g_
                        else ttl)
            ax.set_title(ttl_full, fontsize=9.8, fontweight="bold",
                         loc="left", color=INK, pad=5)
            cax = fig.add_axes([x0 + 0.02, 0.055, Wp - 0.04, 0.016])
            ext = ("both" if cmap is ANOM_CMAP else
                   "max" if cmap in (RAIN_CMAP, PCT_CMAP) else "neither")
            cb = fig.colorbar(pm, cax=cax, orientation="horizontal",
                              spacing="uniform", extend=ext)
            cb.set_ticks(levk[1:-1] if cmap in (PCT_CMAP, ANOM_CMAP)
                         else levk[:-1])
            cb.ax.tick_params(labelsize=7.4, length=2.5, pad=2, color=MUTE,
                              labelcolor=MUTE)
            cb.outline.set_edgecolor("#9aa7b2")
            cb.outline.set_linewidth(0.5)
            if cmap is PCT_CMAP:
                cax.set_title("dry  ←  % of the day-of-year climatology  →  wet",
                              fontsize=6.6, color=MUTE, pad=2)
            elif cmap is ANOM_CMAP:
                cax.set_title("drier than normal  ←  mm  →  wetter than normal",
                              fontsize=6.6, color=MUTE, pad=2)
        footer(fig, note=(f"{len(idx)} satellite days in window · gauge dots "
                          "share the map color scale — a dot that vanishes "
                          "means satellite and gauge agree"))
        pdf.savefig(fig, dpi=200)
        plt.close(fig)
        print(f"  page {pageno}: {label}", flush=True)
    return f"{through:%Y-%m-%d}"








def forecast_map_page(pdf, when: datetime) -> None:
    """Page 7: combined skill-corrected most-likely 15-day rainfall —
    blended AIFS-ENS + IFS-ENS ensemble means, per-basin/lead bias
    factors applied, absolute total and % of normal."""
    import glob
    import json
    import re
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    grib_dir = REPO / "scripts" / "mjo" / "data" / "aifs"
    latest = {}
    for f in sorted(glob.glob(str(grib_dir / "*_*z.pf.tp.grib2"))):
        m = re.match(r"(aifs|ifs)_(\d{8})_(\d{2})z", Path(f).name)
        if m:
            latest[m.group(1)] = (m.group(2), m.group(3))
    if not latest:
        return
    verif = json.loads((Path.home() / "colombia_hydro" / "out" /
                        "fcst_verif.json").read_text())
    rp = CRM.region_paths()

    def daily_fields(model, date, hh):
        """(valid_days, daily[nday, ny, nx] ens mean, lons, lats)."""
        parts = []
        for typ in (("cf", "pf") if model == "aifs" else ("pf",)):
            pth = grib_dir / f"{model}_{date}_{hh}z.{typ}.tp.grib2"
            if not pth.exists():
                continue
            ds = xr.open_dataset(pth, engine="cfgrib", chunks={},
                                 backend_kwargs={"filter_by_keys":
                                                 {"shortName": "tp"},
                                                 "indexpath": ""})
            da = ds["tp"]
            if da.attrs.get("units", "").strip() in ("m", "metre", "metres"):
                da = da * 1000.0
            lons_ = da.longitude.values
            if lons_.max() > 180:
                da = da.assign_coords(longitude=(da.longitude + 180) % 360
                                      - 180)
            da = da.sortby("longitude").sortby("latitude")
            da = da.sel(longitude=slice(CRM.LON0 - 360, CRM.LON1 - 360),
                        latitude=slice(CRM.LAT0, CRM.LAT1))
            if "number" not in da.dims:
                da = da.expand_dims("number")
            parts.append(da.compute())
        da = parts[0] if len(parts) == 1 else xr.concat(parts, dim="number")
        steps_h = (da.step.values / np.timedelta64(1, "h")).astype(int)
        order = np.argsort(steps_h)
        steps_h = steps_h[order]
        v = da.isel(step=order).mean("number")\
              .transpose("step", "latitude", "longitude").values
        init = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{hh}:00")
        bh = np.concatenate([[0], steps_h])
        bv = np.concatenate([np.zeros((1,) + v.shape[1:]), v], axis=0)
        days_, buckets = [], []
        for k in range(len(bh) - 1):
            t0_ = init + np.timedelta64(int(bh[k]), "h")
            if bh[k + 1] - bh[k] == 24 and t0_ == t0_.astype("datetime64[D]"):
                days_.append(t0_.astype("datetime64[D]"))
                buckets.append(np.clip(bv[k + 1] - bv[k], 0, None))
        return (days_, np.array(buckets),
                da.longitude.values % 360, da.latitude.values)

    fields = {m: daily_fields(m, d, h) for m, (d, h) in latest.items()}
    # common valid-day window across models
    sets = [set(f[0]) for f in fields.values()]
    common = sorted(set.intersection(*sets)) if len(sets) > 1 else sorted(sets[0])
    if len(common) < 5:
        return
    lons, lats = list(fields.values())[0][2], list(fields.values())[0][3]

    # per-region bias-factor rasters by lead band, per model; smooth edges
    LO, LA = np.meshgrid(lons, lats)
    pts = np.column_stack([LO.ravel() % 360, LA.ravel()])
    reg_mask = {}
    from matplotlib.path import Path as MplPath
    for r, rings in rp.items():
        if r not in ORDER:
            continue
        inside = np.zeros(LO.shape, bool)
        for ring in rings:
            inside |= MplPath(np.column_stack([ring[:, 0] % 360, ring[:, 1]])
                              ).contains_points(pts).reshape(LO.shape)
        reg_mask[r] = inside
    bands = verif["bands_lead_days"]

    def band_of(lead):
        for bi, (a, b) in enumerate(bands):
            if a <= lead <= b:
                return str(bi)
        return str(len(bands) - 1)

    init0 = min(np.datetime64(f"{d}"[:4] + "-" + f"{d}"[4:6] + "-"
                              + f"{d}"[6:8]) for d, _ in latest.values())
    blended = np.zeros(LO.shape)
    w_ai = float(np.mean([verif["weight_aifs"][r][b] for r in ORDER
                          for b in verif["weight_aifs"][r]]))
    for model, (days_, buckets, _, _) in fields.items():
        w = w_ai if model == "aifs" else 1 - w_ai
        acc = np.zeros(LO.shape)
        for dd, bucket in zip(days_, buckets):
            if dd not in common:
                continue
            lead = max(int((dd - init0).astype(int)), 1)
            Fr = np.full(LO.shape, np.nan)
            for r in ORDER:
                Fr[reg_mask[r]] = verif["bias_factors"][r][band_of(lead)][model]
            Fr = np.where(np.isfinite(Fr), Fr, np.nanmean(
                [verif["bias_factors"][r][band_of(lead)][model]
                 for r in ORDER]))
            acc += bucket * gaussian_filter(Fr, 1.5)
        blended += w * acc

    # climatology over the same days (corrected)
    ml, mt = IP._grid_axes()
    _lo = np.sort(IP._LON[ml] % 360)
    _la = np.sort(IP._LAT[mt])
    _li = (_lo >= CRM.LON0) & (_lo <= CRM.LON1)
    _ti = (_la >= CRM.LAT0) & (_la <= CRM.LAT1)
    coef = xr.open_dataset(CLIM_NC)["coef"].values
    Fc = gauge_correction(np.sort(IP._LON[ml])[_li], _la[_ti])
    cl = np.sum([eval_clim(coef, min(dd.item().timetuple().tm_yday, 365)
                           )[np.ix_(_ti, _li)] for dd in common], axis=0) * Fc
    # regrid clim (0.1 deg) -> model grid (0.25) by nearest sampling
    ii = np.array([int(np.argmin(np.abs(_la[_ti] - la))) for la in lats])
    jj = np.array([int(np.argmin(np.abs(np.sort(IP._LON[ml])[_li] - lo)))
                   for lo in lons % 360])
    clm = cl[np.ix_(ii, jj)]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(clm > 2.0, 100.0 * blended / clm, np.nan)

    span = (f"{common[0].astype(object):%b %d} – "
            f"{common[-1].astype(object):%b %d}")
    fig = plt.figure(figsize=(11.69, 8.27))
    inits_s = " · ".join(f"{m.upper()} {d} {h}Z" for m, (d, h) in latest.items())
    header(fig, when,
           f"Most-likely rainfall, next {len(common)} days · blended "
           f"ensemble means, per-basin/lead bias factors + skill weights "
           f"from the live verification archive",
           f"{inits_s}")
    H = 0.775
    aspect = (CRM.LON1 - CRM.LON0) / (CRM.LAT1 - CRM.LAT0)
    Wp = H * 8.27 / 11.69 * aspect
    gapx = 0.045
    xleft = (1.0 - (2 * Wp + gapx)) / 2
    lev15 = [0, 10, 25, 50, 100, 150, 200, 300, 400, 500, 700]
    for k, (field, cmap, levk, ttl) in enumerate([
            (blended, RAIN_CMAP, lev15,
             f"Skill-corrected total (mm)   ·   {span}"),
            (pct, PCT_CMAP, LEV_PCT, "Percent of normal")]):
        x0 = xleft + k * (Wp + gapx)
        norm = BoundaryNorm(levk, cmap.N)
        ax = fig.add_axes([x0, 0.105, Wp, H], projection=ccrs.PlateCarree())
        pm, _ = draw_map(ax, lons, lats, field, cmap, norm, rp,
                         mask_ocean=(k == 1))
        ax.set_title(ttl, fontsize=10.5, fontweight="bold", loc="left",
                     color=INK, pad=5)
        cax = fig.add_axes([x0 + 0.02, 0.055, Wp - 0.04, 0.016])
        ext = "max"
        cb = fig.colorbar(pm, cax=cax, orientation="horizontal",
                          spacing="uniform", extend=ext)
        cb.set_ticks(levk[1:-1] if cmap is PCT_CMAP else levk[:-1])
        cb.ax.tick_params(labelsize=7.4, length=2.5, pad=2, color=MUTE,
                          labelcolor=MUTE)
        cb.outline.set_edgecolor("#9aa7b2")
        cb.outline.set_linewidth(0.5)
    footer(fig, note="blended by verified inverse-error weights · bias "
                     "factors mature against corrected IMERG twice daily")
    pdf.savefig(fig, dpi=200)
    plt.close(fig)
    print(f"  page {_PAGE[0]}: most-likely 15-day rainfall", flush=True)


def rain_fan_page(pdf, when: datetime) -> None:
    """Page 7: per-basin corrected-rainfall time series vs seasonal norm,
    with the bias-corrected ensemble rain fan."""
    import json
    import matplotlib.dates as mdates
    tc = json.loads((Path.home() / "colombia_hydro" / "raw" /
                     "imerg_basin_daily.json").read_text())
    rdates = [datetime.strptime(d, "%Y%m%d") for d in tc["dates"]]
    fan = None
    fj = REPO / "colombia_hydro" / "data" / "inflow_forecast.json"
    if fj.exists():
        f_ = json.loads(fj.read_text())
        if "rain" in f_ and (np.datetime64(when.strftime("%Y-%m-%d"))
                - np.datetime64(f_["dates"][0])).astype(int) <= 2:
            fan = f_

    t0 = when - timedelta(days=100)
    t1 = when + timedelta(days=21)
    fig = plt.figure(figsize=(11.69, 8.27))
    header(fig, when,
           "Basin rainfall vs seasonal norm · gauge-corrected IMERG observed "
           "· bias-corrected AIFS-ENS + IFS-ENS ensemble forecast",
           "mm/day")
    for k, r in enumerate(ORDER):
        row, col = divmod(k, 3)
        ax = fig.add_axes([0.052 + col * 0.325, 0.53 - row * 0.435,
                           0.29, 0.345])
        obs = np.array(tc[r], float)
        cl = np.array(tc[r + "_clim"], float)
        m = [i for i, d in enumerate(rdates) if d >= t0]
        td = [rdates[i] for i in m]
        ax.bar(td, obs[m], width=1.0, color="#a8c6e2", lw=0,
               label="daily (corrected IMERG)")
        k7 = np.convolve(np.where(np.isfinite(obs), obs, 0),
                         np.ones(7) / 7, "full")[:len(obs)]
        ax.plot(td, k7[m], color="#c62828", lw=1.5, label="7-day mean")
        ax.plot(td, cl[m], color="#1f4e8c", lw=1.5, label="seasonal norm")
        if fan is not None:
            fd = [datetime.strptime(x, "%Y-%m-%d") for x in fan["dates"]]
            q = fan["rain"][r]
            ok = [j for j, v in enumerate(q["p50"]) if v is not None]
            fdo = [fd[j] for j in ok]
            ax.fill_between(fdo, [q["p10"][j] for j in ok],
                            [q["p90"][j] for j in ok], color="#e08214",
                            alpha=0.22, lw=0)
            ax.fill_between(fdo, [q["p25"][j] for j in ok],
                            [q["p75"][j] for j in ok], color="#e08214",
                            alpha=0.35, lw=0)
            ax.plot(fdo, [q["p50"][j] for j in ok], color="#b35806",
                    lw=1.6, ls="--", label="ensemble forecast")
            # norm continues under the fan for direct comparison
            fdoy = np.array([min(d.timetuple().tm_yday, 365)
                             for d in fdo])
            doy_hist = np.array([min(d.timetuple().tm_yday, 365)
                                 for d in rdates])
            clv = []
            for dy in fdoy:
                mm2 = doy_hist == dy
                clv.append(float(np.nanmean(cl[mm2])) if mm2.any() else np.nan)
            ax.plot(fdo, clv, color="#1f4e8c", lw=1.5, ls=(0, (2, 2)))
        wk = np.nanmean(obs[m][-7:]) if len(m) >= 7 else np.nan
        wkc = np.nanmean(cl[m][-7:]) if len(m) >= 7 else np.nan
        ttl = r
        if np.isfinite(wk) and wkc > 0:
            ttl += f"   ·   last 7 d: {100 * wk / wkc:.0f}% of norm"
        ax.set_title(ttl, fontsize=10.5, fontweight="bold", loc="left",
                     color=INK, pad=4)
        ax.set_xlim(t0, t1)
        ax.set_ylim(bottom=0)
        ax.axvline(when, color="0.55", lw=0.7, ls=":")
        ax.grid(lw=0.25, alpha=0.5)
        ax.tick_params(labelsize=7.5)
        ax.set_ylabel("mm/day", fontsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if k == 0:
            ax.legend(fontsize=6.6, loc="upper left", framealpha=0.9)
    footer(fig, note="fan: ~101 members, per-basin/lead bias factors from the "
                     "verification archive · norm = harmonic climatology of the "
                     "corrected satellite record · dotted line = today")
    pdf.savefig(fig, dpi=200)
    plt.close(fig)
    print(f"  page {_PAGE[0]}: basin rain fans", flush=True)


def inflow_page(pdf, when: datetime) -> None:
    """Page 7: inflows vs seasonal norms, zoomed to ~3 months + the fan.
    Traditional units (GWh/day)."""
    import json
    ic = json.loads((REPO / "colombia_hydro" / "data" /
                     "inflow_clim.json").read_text())
    fan = None
    fj = REPO / "colombia_hydro" / "data" / "inflow_forecast.json"
    if fj.exists():
        f_ = json.loads(fj.read_text())
        if (np.datetime64(when.strftime("%Y-%m-%d"))
                - np.datetime64(f_["dates"][0])).astype(int) <= 2:
            fan = f_

    t0 = when - timedelta(days=100)
    t1 = when + timedelta(days=21)
    axis = [t0 + timedelta(days=k) for k in range((t1 - t0).days + 1)]
    doyx = np.array([min(d.timetuple().tm_yday, 365) for d in axis]) - 1

    rdates = [datetime.strptime(d, "%Y-%m-%d") for d in ic["recent"]["dates"]]
    fig = plt.figure(figsize=(11.69, 8.27))
    header(fig, when,
           "Inflows vs seasonal norms · fleet-corrected per-river "
           "climatologies (2000–2026) · AIFS-ENS + IFS-ENS ensemble forecast",
           "GWh/day")
    import matplotlib.dates as mdates
    for k, r in enumerate(ORDER):
        row, col = divmod(k, 3)
        ax = fig.add_axes([0.052 + col * 0.325, 0.53 - row * 0.435,
                           0.29, 0.345])
        pct = np.array(ic["clim"][r]["pct"], float)         # (365, 5) GWh
        ax.fill_between(axis, pct[doyx, 0], pct[doyx, 4], color="#9db8d8",
                        alpha=0.32, lw=0, label="p10–p90 norm")
        ax.fill_between(axis, pct[doyx, 1], pct[doyx, 3], color="#5b87c0",
                        alpha=0.32, lw=0, label="p25–p75")
        ax.plot(axis, pct[doyx, 2], color="#1f4e8c", lw=1.5, label="median")
        obs = np.array(ic["recent"][r], float)
        m = [i for i, d in enumerate(rdates) if d >= t0]
        ax.plot([rdates[i] for i in m], obs[m], color="#c62828", lw=1.4,
                label="observed")
        last_pct = None
        v = np.array(ic["recent"]["pct_of_norm"][r], float)
        v = v[v > 0]
        if len(v):
            last_pct = v[-5:].mean()
        if fan is not None:
            fd = [datetime.strptime(x, "%Y-%m-%d") for x in fan["dates"]]
            fdoy = np.array([min(d.timetuple().tm_yday, 365) for d in fd]) - 1
            nrm = np.array(ic["clim"][r]["mean"], float)[fdoy] / 100.0
            q = fan["basins"][r]["q"]
            ax.fill_between(fd, np.array(q["p10"]) * nrm,
                            np.array(q["p90"]) * nrm, color="#e08214",
                            alpha=0.22, lw=0)
            ax.fill_between(fd, np.array(q["p25"]) * nrm,
                            np.array(q["p75"]) * nrm, color="#e08214",
                            alpha=0.35, lw=0)
            ax.plot(fd, np.array(q["p50"]) * nrm, color="#b35806", lw=1.6,
                    ls="--", label="ensemble forecast")
        ttl = f"{r}"
        if last_pct is not None:
            ttl += f"   ·   now {last_pct:.0f}% of norm"
        ax.set_title(ttl, fontsize=10.5, fontweight="bold", loc="left",
                     color=INK, pad=4)
        ax.set_xlim(t0, t1)
        ax.axvline(when, color="0.55", lw=0.7, ls=":")
        ax.grid(lw=0.25, alpha=0.5)
        ax.tick_params(labelsize=7.5)
        ax.set_ylabel("GWh/day", fontsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.DayLocator([10, 20]))
        if k == 0:
            ax.legend(fontsize=6.6, loc="upper left", framealpha=0.9)
    footer(fig, note="orange fan: ~101 bias-corrected AIFS-ENS + IFS-ENS members "
                     "through each basin's fitted memory-kernel model, anchored "
                     "to observations · dotted line = today")
    pdf.savefig(fig, dpi=200)
    plt.close(fig)
    print(f"  page {_PAGE[0]}: inflows vs norms + fans", flush=True)




def storage_page(pdf, when: datetime) -> None:
    """Page 10: reservoir storage vs seasonal norms + member storage fans."""
    import json
    import matplotlib.dates as mdates
    st = json.loads((REPO / "colombia_hydro" / "data" /
                     "storage.json").read_text())
    fan = None
    fj = REPO / "colombia_hydro" / "data" / "inflow_forecast.json"
    if fj.exists():
        f_ = json.loads(fj.read_text())
        if "storage" in f_ and (np.datetime64(when.strftime("%Y-%m-%d"))
                - np.datetime64(f_["dates"][0])).astype(int) <= 2:
            fan = f_

    regs = [r for r in ORDER if r in st.get("pct_doy", {})]
    sdates = [datetime.strptime(d, "%Y-%m-%d") for d in st["recent"]["dates"]]
    t0 = when - timedelta(days=365)          # storage moves slowly — show the year
    t1 = when + timedelta(days=21)
    axis = [t0 + timedelta(days=k) for k in range((t1 - t0).days + 1)]
    doyx = np.array([min(d.timetuple().tm_yday, 365) for d in axis]) - 1

    fig = plt.figure(figsize=(11.69, 8.27))
    header(fig, when,
           "Reservoir storage · useful volume as % of useful capacity · "
           "fans integrate the member inflow forecasts through the water "
           "balance", "% of capacity")

    def panel(ax, r, label_fs=9.8):
        env = np.array(st["pct_doy"][r], float)
        ax.fill_between(axis, env[doyx, 0], env[doyx, 4], color="#9db8d8",
                        alpha=0.32, lw=0, label="p10–p90 norm")
        ax.fill_between(axis, env[doyx, 1], env[doyx, 3], color="#5b87c0",
                        alpha=0.32, lw=0, label="p25–p75")
        ax.plot(axis, env[doyx, 2], color="#1f4e8c", lw=1.4, label="median")
        obs = np.array(st["recent"]["pct_full"][r], float)
        obs[obs == 0] = np.nan
        m = [i for i, d in enumerate(sdates) if d >= t0]
        ax.plot([sdates[i] for i in m], obs[m], color="#c62828", lw=1.5,
                label="observed")
        if fan is not None and r in fan["storage"]["basins"]:
            fd = [datetime.strptime(x, "%Y-%m-%d") for x in fan["dates"]]
            q = fan["storage"]["basins"][r]
            ax.fill_between(fd, q["p10"], q["p90"], color="#e08214",
                            alpha=0.22, lw=0)
            ax.fill_between(fd, q["p25"], q["p75"], color="#e08214",
                            alpha=0.35, lw=0)
            ax.plot(fd, q["p50"], color="#b35806", lw=1.6, ls="--",
                    label="forecast")
        now = obs[np.isfinite(obs)][-1] if np.isfinite(obs).any() else None
        ttl = r + (f"   ·   {now:.1f}% full" if now is not None else "")
        ax.set_title(ttl, fontsize=label_fs, fontweight="bold", loc="left",
                     color=INK, pad=3)
        ax.set_xlim(t0, t1)
        ax.axvline(when, color="0.55", lw=0.7, ls=":")
        ax.grid(lw=0.25, alpha=0.5)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("% full", fontsize=7.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    for k, r in enumerate(regs[:6]):
        row, col = divmod(k, 3)
        ax = fig.add_axes([0.052 + col * 0.325, 0.635 - row * 0.29,
                           0.29, 0.215])
        panel(ax, r, label_fs=9.2)
        if k == 0:
            ax.legend(fontsize=5.8, loc="lower left", framealpha=0.85)
    axn = fig.add_axes([0.052, 0.055, 0.915, 0.24])
    panel(axn, "NATIONAL", label_fs=10.5)
    footer(fig, note="storage fan: every inflow member integrated through "
                     "S' = S + inflow − outflow (outflow = recent reality "
                     "blended with the ENSO-adjusted seasonal norm)")
    pdf.savefig(fig, dpi=200)
    plt.close(fig)
    print(f"  page {_PAGE[0]}: reservoir storage", flush=True)


def generation_page(pdf, when: datetime) -> None:
    """Page 8: national hydro generation outlook — the final piece."""
    import json
    import matplotlib.dates as mdates
    g = json.loads((REPO / "colombia_hydro" / "data" /
                    "generation.json").read_text())
    gm = json.loads((REPO / "colombia_hydro" / "data" /
                     "gen_model.json").read_text())
    st = json.loads((REPO / "colombia_hydro" / "data" /
                     "storage.json").read_text())
    fan = None
    fj = REPO / "colombia_hydro" / "data" / "inflow_forecast.json"
    if fj.exists():
        f_ = json.loads(fj.read_text())
        if "generation" in f_ and (np.datetime64(when.strftime("%Y-%m-%d"))
                - np.datetime64(f_["dates"][0])).astype(int) <= 2:
            fan = f_

    gdates = [datetime.strptime(d, "%Y-%m-%d") for d in g["recent"]["dates"]]
    hyd = np.array(g["recent"]["hydro"], float)
    tot = np.array(g["recent"]["total"], float)
    env = np.array(g["env_doy"]["hydro"], float)            # (365,5) GWh/d

    t0 = when - timedelta(days=120)
    t1 = when + timedelta(days=18)
    axis = [t0 + timedelta(days=k) for k in range((t1 - t0).days + 1)]
    doyx = np.array([min(d.timetuple().tm_yday, 365) for d in axis]) - 1

    fig = plt.figure(figsize=(11.69, 8.27))
    header(fig, when,
           "National hydro generation outlook · persistence + rain/storage "
           "state model driven by the same ensemble",
           "GW (avg power)")
    ax = fig.add_axes([0.055, 0.40, 0.62, 0.475])
    ax.fill_between(axis, env[doyx, 0] / 24, env[doyx, 4] / 24,
                    color="#9db8d8", alpha=0.30, lw=0,
                    label="p10–p90, 2000–2026")
    ax.fill_between(axis, env[doyx, 1] / 24, env[doyx, 3] / 24,
                    color="#5b87c0", alpha=0.30, lw=0, label="p25–p75")
    ax.plot(axis, env[doyx, 2] / 24, color="#1f4e8c", lw=1.4,
            label="median (full record)")
    m = [i for i, d in enumerate(gdates) if d >= t0]
    ax.plot([gdates[i] for i in m], hyd[m] / 24, color="#c62828", lw=1.5,
            label="observed hydro generation")
    if fan is not None:
        fd = [datetime.strptime(x, "%Y-%m-%d") for x in fan["dates"]]
        q = fan["generation"]
        ax.fill_between(fd, np.array(q["p10"]) / 24, np.array(q["p90"]) / 24,
                        color="#e08214", alpha=0.22, lw=0)
        ax.fill_between(fd, np.array(q["p25"]) / 24, np.array(q["p75"]) / 24,
                        color="#e08214", alpha=0.35, lw=0)
        ax.plot(fd, np.array(q["p50"]) / 24, color="#b35806", lw=1.8, ls="--",
                label="15-day forecast (weekly dispatch cycle resolved)")
    ax.axvline(when, color="0.55", lw=0.7, ls=":")
    ax.set_xlim(t0, t1)
    ax.set_title("National hydro generation — last 120 days, forecast fan, "
                 "and the 26-year day-of-year envelope",
                 fontsize=11, fontweight="bold", loc="left", color=INK)
    ax.set_ylabel("GW (average power)", fontsize=9)
    ax.grid(lw=0.25, alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(fontsize=7.4, loc="lower left", framealpha=0.9)
    ax.tick_params(labelbottom=False)

    # hydro share of total generation — the thermal displacement, one year
    ax2 = fig.add_axes([0.055, 0.10, 0.62, 0.255])
    t0y = when - timedelta(days=365)
    my = [i for i, d in enumerate(gdates) if d >= t0y]
    shr = 100 * hyd / np.where(tot > 0, tot, np.nan)
    ax2.plot([gdates[i] for i in my], shr[my], color="#d8b89d", lw=0.7,
             alpha=0.8)
    k30 = np.ones(30) / 30
    sm = np.convolve(np.where(np.isfinite(shr), shr, np.nanmean(shr)),
                     k30, "full")[:len(shr)]
    ax2.plot([gdates[i] for i in my], sm[my], color="#b35806", lw=1.8,
             label="30-day mean")
    # NORM LINE. Total demand trends +2.6%/yr (p<0.0001), so an absolute
    # GWh norm must be detrended - but the SHARE does not: hydro capacity
    # has grown in step with demand, giving +0.13 pts/yr, p=0.69, and a
    # full-record mean of 75.7% against a last-5-year 75.6%. A flat recent
    # mean is therefore the correct reference here, and a trend line fitted
    # to noise would be the error. Use the last 5 years, not the full
    # record, so any slow drift that does exist cannot bias it.
    yr5 = [i for i, d in enumerate(gdates) if d >= when - timedelta(days=5*365)]
    norm_share = float(np.nanmean(shr[yr5])) if yr5 else float(np.nanmean(shr))
    ax2.axhline(norm_share, color="#444", lw=1.1, ls="--",
                label=f"5-yr norm {norm_share:.0f}%")
    # forward view from the seasonal outlook, if present
    try:
        import json as _json
        _o = _json.loads((PRIV / "out" / "outlook_2027.json").read_text())
        _m = _o.get("months") or _o.get("monthly") or []
        _x, _p50, _lo, _hi = [], [], [], []
        for _r in _m:
            _k = _r.get("month")
            if not _k:
                continue
            _d = datetime.strptime(_k + "-15", "%Y-%m-%d")
            _h = _r.get("hydro_share_pct") or {}
            _s = _h.get("p50")
            if _s is None:
                continue
            _x.append(_d); _p50.append(_s)
            _lo.append(_h.get("p10", _s)); _hi.append(_h.get("p90", _s))
        if _x:
            ax2.fill_between(_x, _lo, _hi, color="#6b2d7d", alpha=0.16, lw=0)
            ax2.plot(_x, _p50, "o--", color="#6b2d7d", lw=1.6, ms=4,
                     label="outlook p50")
            ax2.set_xlim(t0y, max(_x) + timedelta(days=20))
    except Exception:                              # noqa: BLE001 - optional
        pass
    ax2.set_ylabel("hydro %", fontsize=8)
    if not ax2.get_xlim()[1] > mdates.date2num(t1):
        ax2.set_xlim(t0y, t1)
    ax2.set_title("Hydro share of total generation \u2014 observed against a "
                  "5-year norm, with the seasonal outlook ahead",
                  fontsize=9.5, fontweight="bold", loc="left", color=INK)
    ax2.grid(lw=0.25, alpha=0.5)
    ax2.tick_params(labelsize=7.5)
    ax2.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # sidebar stat cards
    n30 = min(30, len(hyd))
    share = 100 * np.nansum(hyd[-n30:]) / np.nansum(tot[-n30:])
    natst = st["regions"]["NATIONAL"]
    cards = [
        ("HYDRO, 30-DAY MEAN", f"{np.nanmean(hyd[-n30:])/24:.2f} GW",
         f"{share:.0f}% of total generation"),
        ("RESERVOIRS", f"{natst['pct_full']:.1f}% full",
         f"useful storage {natst['vol_kwh']/1e9:.1f} of "
         f"{natst['cap_kwh']/1e9:.1f} TWh"),
        ("ENSO", f"Niño-3.4  {gm['nino_now']:+.1f} °C",
         "El Niño conditions" if gm["nino_now"] > 0.5 else
         "La Niña conditions" if gm["nino_now"] < -0.5 else "neutral"),
        ("MODEL SKILL", f"r = {gm['r_loyo']:.2f}",
         "state model, 23-yr out-of-sample"),
    ]
    for i, (t_, big, sub) in enumerate(cards):
        y0 = 0.705 - i * 0.155
        cx = fig.add_axes([0.705, y0, 0.26, 0.135])
        cx.set_axis_off()
        cx.add_patch(plt.Rectangle((0, 0), 1, 1, transform=cx.transAxes,
                                   facecolor="#f2f5f8", edgecolor="#d5dde4",
                                   lw=0.8))
        cx.text(0.07, 0.76, t_, transform=cx.transAxes, fontsize=7.2,
                color=MUTE, fontweight="bold")
        cx.text(0.07, 0.40, big, transform=cx.transAxes, fontsize=15.5,
                color=INK, fontweight="bold")
        cx.text(0.07, 0.13, sub, transform=cx.transAxes, fontsize=7.6,
                color=MUTE)
    fig.text(0.705, 0.115,
             "Forecast = fitted persistence (dispatch is storage-buffered)\n"
             "blended with the inflow/storage/ENSO state model, driven\n"
             "member-by-member by the same AIFS+IFS rain ensemble as\n"
             "the basin pages. Bands widen by the blend's measured\n"
             "residual uncertainty at each lead.",
             fontsize=7.6, color=MUTE, va="top")
    footer(fig)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)
    print(f"  page {_PAGE[0]}: national generation outlook", flush=True)




def load_page(pdf, when: datetime) -> None:
    """Page 12: national load evolution + temperature link."""
    import json
    import matplotlib.dates as mdates
    lj = REPO / "colombia_hydro" / "data" / "load.json"
    if not lj.exists():
        return
    ld = json.loads(lj.read_text())

    fig = plt.figure(figsize=(11.69, 8.27))
    header(fig, when,
           "National electricity demand · XM DemaReal, daily average power",
           "GW")
    # 26-year evolution
    fd = [datetime.strptime(d, "%Y-%m-%d") for d in ld["full"]["dates"]]
    fg = np.array(ld["full"]["gw"], float)
    ax = fig.add_axes([0.055, 0.53, 0.62, 0.345])
    ax.plot(fd, fg, color="#9db8d8", lw=0.6, alpha=0.85)
    yrs = sorted(ld["annual_mean_gw"])
    ax.plot([datetime(int(y), 7, 1) for y in yrs],
            [ld["annual_mean_gw"][y] for y in yrs], color="#1f4e8c", lw=2.0,
            marker="o", ms=3, label="annual mean")
    ax.set_title("Demand, 2000–2026 — weekly samples + annual means",
                 fontsize=10.5, fontweight="bold", loc="left", color=INK)
    ax.grid(lw=0.25, alpha=0.5)
    ax.tick_params(labelsize=7.5)
    ax.set_ylabel("GW", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # recent two years
    rd = [datetime.strptime(d, "%Y-%m-%d") for d in ld["recent"]["dates"]]
    rg = np.array(ld["recent"]["gw"], float)
    t0 = when - timedelta(days=730)
    m = [i for i, d in enumerate(rd) if d >= t0]
    ax2 = fig.add_axes([0.055, 0.10, 0.62, 0.345])
    ax2.plot([rd[i] for i in m], rg[m], color="#a8c6e2", lw=0.7)
    k30 = np.ones(30) / 30
    sm = np.convolve(np.where(np.isfinite(rg), rg, np.nanmean(rg)), k30,
                     "full")[:len(rg)]
    ax2.plot([rd[i] for i in m], sm[m], color="#c62828", lw=1.8,
             label="30-day mean")
    ax2.set_title("Last two years — daily (weekly dispatch cycle visible) "
                  "and 30-day mean", fontsize=10.5, fontweight="bold",
                  loc="left", color=INK)
    ax2.grid(lw=0.25, alpha=0.5)
    ax2.tick_params(labelsize=7.5)
    ax2.set_ylabel("GW", fontsize=8.5)
    ax2.legend(fontsize=7.5, loc="lower right")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    yrs_i = [int(y) for y in yrs]
    g_now = float(np.nanmean(rg[-30:]))
    yoy = None
    if len(yrs_i) >= 2 and (yrs_i[-1] - 1) in ld["annual_mean_gw"]:
        pass
    prev = {int(k): v for k, v in ld["annual_mean_gw"].items()}
    if yrs_i[-1] - 1 in prev and yrs_i[-1] in prev:
        yoy = 100 * (prev[yrs_i[-1]] / prev[yrs_i[-1] - 1] - 1)
    tl = ld.get("temp_link", {})
    cards = [("DEMAND, 30-DAY MEAN", f"{g_now:.2f} GW",
              "national daily average power"),
             ("GROWTH", f"{yoy:+.1f}% YoY" if yoy is not None else "—",
              f"annual mean {yrs_i[-1]} vs {yrs_i[-1]-1}"),
             ("TEMPERATURE LINK", f"{tl.get('pct_per_degC_monthly', '—')} %/°C",
              f"monthly r = {tl.get('r_monthly', '—')} · 2000–2023, "
              "4-city pop-weighted ERA5"),
             ("DAILY SENSITIVITY", f"{tl.get('pct_per_degC_daily', '—')} %/°C",
              f"daily r = {tl.get('r_daily', '—')} (weekday-adjusted, "
              "detrended)")]
    for i, (t_, big, sub) in enumerate(cards):
        y0 = 0.705 - i * 0.155
        cx = fig.add_axes([0.705, y0, 0.26, 0.135])
        cx.set_axis_off()
        cx.add_patch(plt.Rectangle((0, 0), 1, 1, transform=cx.transAxes,
                                   facecolor="#f2f5f8", edgecolor="#d5dde4",
                                   lw=0.8))
        cx.text(0.07, 0.76, t_, transform=cx.transAxes, fontsize=7.2,
                color=MUTE, fontweight="bold")
        cx.text(0.07, 0.40, str(big), transform=cx.transAxes, fontsize=14.5,
                color=INK, fontweight="bold")
        cx.text(0.07, 0.13, sub, transform=cx.transAxes, fontsize=7.2,
                color=MUTE)
    fig.text(0.705, 0.115,
             "The honest result: temperature barely moves Colombian\n"
             "demand — a weak daily bump (~0.8%/°C, r≈0.1) and nothing\n"
             "at monthly scale. Near-equatorial, mild Andean cities have\n"
             "no heating/cooling season; growth drives the curve. El\n"
             "Niño's risk here is supply (hydro), not demand.",
             fontsize=7.6, color=MUTE, va="top")
    footer(fig)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)
    print(f"  page {_PAGE[0]}: national load", flush=True)



def basin_map_page(pdf, when: datetime) -> None:
    """Page 1: the physical setting — terrain, basins, plants.

    Front-loaded deliberately. Everything downstream is an anomaly against
    a norm, and an anomaly is meaningless until you know which piece of
    ground it belongs to."""
    import matplotlib.image as mpimg
    img = REPO / "colombia_hydro" / "xm_regions_topo.webp"
    fig = plt.figure(figsize=(11.69, 8.27))
    header(fig, when,
           "The physical setting \u00b7 XM hydrological regions over terrain, "
           "with the generating fleet",
           "orientation")
    if img.exists():
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.80])
        ax.imshow(mpimg.imread(str(img)))
        ax.set_axis_off()
    else:
        ax = fig.add_axes([0.1, 0.3, 0.8, 0.4]); ax.set_axis_off()
        ax.text(0.5, 0.5, "terrain map unavailable\n"
                "(run scripts/sst/render_hydro_maps.py)",
                ha="center", va="center", fontsize=11, color="0.4")
    footer(fig, "Dams sit where rivers drop off the cordilleras, which is why "
                "a basin's energy value depends on the drop below it as much "
                "as the water in it.")
    pdf.savefig(fig); plt.close(fig)
    print(f"  page {_PAGE[0]}: basins over terrain", flush=True)


def national_page(pdf, when: datetime) -> None:
    """THE headline page: national inflow forecast and its uncertainty.

    Placed ahead of the per-basin detail because it is the number the
    report exists to deliver; the basins are the decomposition behind it."""
    import json
    import matplotlib.dates as mdates
    nat_p = PRIV / "out" / "national_inflow.json"
    fig = plt.figure(figsize=(11.69, 8.27))
    header(fig, when,
           "NATIONAL INFLOW FORECAST \u00b7 balance-of-month daily, then "
           "monthly to +6 \u00b7 the headline deliverable",
           "% of norm and GWh/day")
    if not nat_p.exists():
        ax = fig.add_axes([0.1, 0.4, 0.8, 0.2]); ax.set_axis_off()
        ax.text(0.5, 0.5, "national_inflow.json not found", ha="center")
        pdf.savefig(fig); plt.close(fig); return
    nat = json.loads(nat_p.read_text())

    # -- top: daily balance-of-month fan
    ax = fig.add_axes([0.07, 0.53, 0.88, 0.33])
    dbm = nat.get("daily_balance_of_month", [])
    if dbm:
        xs = [datetime.strptime(r["date"], "%Y-%m-%d") for r in dbm]
        p50 = [r["gwh_p50"] for r in dbm]
        lo = [r.get("gwh_p5", r["gwh_p50"]) for r in dbm]
        hi = [r.get("gwh_p95", r["gwh_p50"]) for r in dbm]
        nrm = [r.get("norm_gwh") for r in dbm]
        ax.fill_between(xs, lo, hi, color="#4d8fe8", alpha=0.22, lw=0,
                        label="p5\u2013p95")
        ax.plot(xs, p50, color="#1f4e9c", lw=2.2, label="median")
        if all(n is not None for n in nrm):
            ax.plot(xs, nrm, color="#888", lw=1.4, ls="--",
                    label="seasonal norm (current fleet)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    ax.set_ylabel("GWh/day", fontsize=9)
    ax.set_title("Balance of month \u2014 daily national inflow energy",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.25)

    # -- bottom: monthly outlook to +6
    ax2 = fig.add_axes([0.07, 0.10, 0.88, 0.31])
    mf = nat.get("monthly_forecast", {}).get("months", [])
    if mf:
        lab = [r["month"] for r in mf]
        x = np.arange(len(mf))
        p50 = np.array([r["pct_p50"] for r in mf])
        lo = np.array([r["pct_p10"] for r in mf])
        hi = np.array([r["pct_p90"] for r in mf])
        ax2.fill_between(x, lo, hi, color="#e8833a", alpha=0.22, lw=0,
                         label="p10\u2013p90")
        ax2.plot(x, p50, "o-", color="#b8551a", lw=2.2, ms=6, label="median")
        ax2.axhline(100, color="#555", lw=1.2, ls="--", label="norm")
        ax2.set_xticks(x); ax2.set_xticklabels(lab, fontsize=8.5)
        for xi, v, h in zip(x, p50, hi):
            ax2.annotate(f"{v:.0f}%", (xi, h), textcoords="offset points",
                         xytext=(0, 5), ha="center", fontsize=8.5,
                         fontweight="bold")
        ax2.legend(fontsize=8, frameon=False, ncol=3)
    ax2.set_ylabel("% of seasonal norm", fontsize=9)
    ax2.set_title("Monthly outlook to +6 \u2014 % of norm with 80% interval",
                  fontsize=10.5, fontweight="bold")
    ax2.grid(alpha=0.25)
    footer(fig, "Intervals are out-of-sample: the spread comes from blocked "
                "cross-validation, not from the fit's own residuals.")
    pdf.savefig(fig); plt.close(fig)
    print(f"  page {_PAGE[0]}: NATIONAL inflow forecast", flush=True)


def price_page(pdf, when: datetime) -> None:
    """Spot-price outlook — the money translation of the water view."""
    import json
    pj = PRIV / "out" / "price_outlook.json"
    fig = plt.figure(figsize=(11.69, 8.27))
    header(fig, when,
           "Spot price (bolsa) outlook \u00b7 driven by the inflow forecast "
           "\u00b7 indicative, wide intervals",
           "COP/kWh \u00b7 final page")
    if not pj.exists():
        ax = fig.add_axes([0.1, 0.4, 0.8, 0.2]); ax.set_axis_off()
        ax.text(0.5, 0.5, "price_outlook.json not found \u2014 run "
                "scripts/sst/price_outlook.py", ha="center")
        pdf.savefig(fig); plt.close(fig); return
    pr = json.loads(pj.read_text())
    m = pr["months"]
    x = np.arange(len(m))
    p50 = np.array([r["price_p50"] for r in m])
    lo = np.array([r["price_p10"] for r in m])
    hi = np.array([r["price_p90"] for r in m])
    ax = fig.add_axes([0.08, 0.30, 0.87, 0.55])
    ax.fill_between(x, lo, hi, color="#a05fb4", alpha=0.20, lw=0,
                    label="p10\u2013p90")
    ax.plot(x, p50, "o-", color="#6b2d7d", lw=2.4, ms=7, label="median")
    ax.set_xticks(x); ax.set_xticklabels([r["month"] for r in m], fontsize=9)
    for xi, v in zip(x, p50):
        ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("COP/kWh", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_title("Monthly mean spot price \u2014 80% interval",
                 fontsize=11, fontweight="bold")
    txt = (f"Log-space fit on 2015\u20132026 monthly means: inflow anomaly, "
           f"storage, ONI, seasonal harmonics and the scarcity price "
           f"(the fuel-indexed ceiling the market runs toward under stress). "
           f"In-sample R\u00b2 {pr['r2_in_sample']:.2f}; blocked "
           f"leave-one-year-out residual spread \u00d7"
           f"{pr['one_sigma_multiplier']:.2f} one-sigma.\n"
           f"The structural fit currently runs LOW \u2014 actual/model has a "
           f"median of {pr.get('regime_anchor', 1):.2f} over the last three "
           f"months, so a regime adjustment is applied in full at lead 1 and "
           f"decayed to 1.0 by lead 6. Direction is well supported; the LEVEL "
           f"is indicative only.")
    fig.text(0.08, 0.19, txt, fontsize=8.6, va="top", wrap=True, color=INK)
    footer(fig, "Not investment advice \u2014 a physical-dispatch model, not "
                "a market model.")
    pdf.savefig(fig); plt.close(fig)
    print(f"  page {_PAGE[0]}: spot price outlook", flush=True)


def main() -> int:
    when = datetime.now(timezone.utc).replace(tzinfo=None)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        # Order follows the decision path: where -> what fell -> what is
        # coming -> WHAT IT MEANS NATIONALLY (the deliverable) -> the basin
        # decomposition behind it -> state -> generation -> price.
        basin_map_page(pdf, when)
        through = rain_pages(pdf, when)
        forecast_map_page(pdf, when)
        rain_fan_page(pdf, when)
        national_page(pdf, when)
        inflow_page(pdf, when)
        storage_page(pdf, when)
        generation_page(pdf, when)
        load_page(pdf, when)
        price_page(pdf, when)
        d = pdf.infodict()
        d["Title"] = f"Colombia Hydro Daily Briefing — {when:%Y-%m-%d}"
        d["Author"] = "scorvec.com"
    dated = ARCHIVE / f"colombia_hydro_{when:%Y%m%d}.pdf"
    dated.write_bytes(OUT_PDF.read_bytes())
    print(f"wrote {OUT_PDF.relative_to(REPO)} (satellite through {through}) "
          f"+ archive {dated.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
