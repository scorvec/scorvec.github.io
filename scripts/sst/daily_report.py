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

REPO = HERE.parent.parent
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


def header(fig, when, subtitle, right2):
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
    CRM.LAT0, CRM.LAT1 = ys.min() - 0.5, ys.max() + 0.5
    pad = max(0.6, ((CRM.LAT1 - CRM.LAT0) - (xs.max() - xs.min())) / 2)
    CRM.LON0, CRM.LON1 = xs.min() - pad, xs.max() + pad

    end = when
    jan1 = datetime(when.year, 1, 1)
    nback = (end - jan1).days + 8
    days = [end - timedelta(days=k) for k in range(nback)][::-1]
    IP.ensure_daily(set(days[-35:]))
    fields, lons, lats = CRM.imerg_stack(days)
    have = [i for i, f in enumerate(fields) if f is not None]
    latest_i = have[-1]
    through = days[latest_i]
    rdates = np.array([f"{d:%Y-%m-%d}" for d in days])
    F = CRM.build_correction(days, fields, rdates, lons, lats)

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
        for k, (field, cmap, levk, ttl, g_) in enumerate([
                (tot, RAIN_CMAP, lev, f"{label} — rainfall (mm)", gg),
                (*right, None)]):
            x0 = 0.045 + k * 0.475
            norm = BoundaryNorm(levk, cmap.N)
            ax = fig.add_axes([x0, 0.115, 0.44, 0.755],
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
            cax = fig.add_axes([x0 + 0.03, 0.068, 0.38, 0.016])
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


def main() -> int:
    when = datetime.now(timezone.utc).replace(tzinfo=None)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        through = rain_pages(pdf, when)
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
