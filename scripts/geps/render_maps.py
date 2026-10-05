#!/usr/bin/env python3
"""Render the GEPS subseasonal anomaly maps for North and South America.

Three views of the same de-drifted anomalies:
  daily   one frame per lead day 1-35, for the site's frame player
  weekly  week 1-5 means as a single panel figure
  change  weekly means of this cycle minus the previous extended cycle, on
          the valid days both forecast (see change())

Anomalies come from forecast.py: forecast minus the GEPS8 hindcast climatology
at the SAME lead, so the model's own drift is removed rather than being read as
signal.

One deliberate asymmetry between fields. Height carries a real climate offset
between the 2001-2020 hindcast epoch and today — a flat +16 to +17 m at every
lead, measured — which would sit over every map as a uniform ridge. For z500 the
per-lead global mean is therefore removed so the CIRCULATION shows. For t2m and
precipitation the raw anomaly is kept, because there the departure from the
2001-2020 climate is part of what you want to see. Both choices are stated on
the figures.

(That offset used to be measured as +17 m at day 1 GROWING to +34 m by day 35.
The growth was not climate: it was forecast.py reading the start-indexed
climatology at the valid day-of-year, so long leads were differenced against
runs started weeks later. Fixed 2026-08-29; the residual is flat, as a climate
offset should be.)

    python render_maps.py --cycle 20260827
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths                                                    # noqa: E402
import archive                                                  # noqa: E402

LIVE = paths.LIVE
FIGS = paths.FIGS

DOMAINS = {
    "na": dict(label="North America", extent=(-168, -52, 12, 72), central=-110,
               proj="lambert"),
    "sa": dict(label="South America", extent=(-95, -30, -58, 15), central=-62,
               proj="plate"),
    # Full longitude circle on Mercator, both hemispheres, centred on the Pacific:
    # the view for Rossby wave trains, where what matters is the zonal chain of
    # alternating highs and lows and how far downstream it reaches. A polar
    # projection hides exactly that by wrapping the chain around the pole. Cut at
    # 70N/60S because Mercator's stretch is 1/cos(lat) and the map becomes
    # unreadable beyond that.
    "glb": dict(label="Global", extent=(-180, 180, -60, 70), central=180,
                proj="merc", stack=True),
}
# tag -> (title, units, colourmap, remove per-lead global mean, scale factor)
# The global mean is removed only for height, where the 2001-2020-to-now climate
# trend would otherwise sit over every map as a uniform ridge. Everywhere else the
# raw anomaly is the quantity of interest.
# OLR uses BrBG_r so NEGATIVE anomaly (enhanced convection) reads green, the
# convention used on MJO diagnostics.
FIELDS = {
    "t2m":   ("2 m temperature", "°C", "RdBu_r", False, 1.0),
    "pr":    ("Precipitation", "mm/day", "BrBG", False, 86400.0),
    "zg500": ("500 hPa height", "m", "RdBu_r", True, 1.0),
    # demean: the live operational GEPS radiates ~7.6 W/m2 differently from the
    # GEPS8 reforecast the climatology was built on — flat across all 35 leads, so
    # it is a model-version offset, not signal, and it tinted every map green.
    "olr":   ("Outgoing longwave", "W m⁻²", "BrBG_r", True, 1.0),
    "u850":  ("850 hPa zonal wind", "m/s", "PuOr_r", False, 1.0),
    "u200":  ("200 hPa zonal wind", "m/s", "PuOr_r", False, 1.0),
    "v850":  ("850 hPa meridional wind", "m/s", "PuOr_r", False, 1.0),
    "v200":  ("200 hPa meridional wind", "m/s", "PuOr_r", False, 1.0),
    "mslp":  ("Mean sea-level pressure", "hPa", "RdBu_r", False, 0.01),
}
# Re-basing 2001-2020 -> 1991-2020.
#
# The hindcast only spans 2001-2020, so the model's 1991-2020 climatology does not
# exist and has to be ESTIMATED. It is estimated by shifting the model climatology
# by the observed change between the two periods:
#
#     M_1991-2020  ~=  M_2001-2020 + [ ERA5_1991-2020 - ERA5_2001-2020 ]
#
# so the anomaly becomes  [F(L) - M(L)] - [ERA5_9120 - ERA5_0120].
#
# Note what this deliberately does NOT do. Differencing straight against ERA5
# would also be "vs 1991-2020", but it puts the model's mean BIAS back into the
# anomaly — measured here at +0.55 mm/day for precipitation and +9.7 W/m2 for
# outgoing longwave, which would swamp the signal. Re-basing keeps the bias and
# drift removal that the hindcast provides and changes only the reference period,
# which is a small, smooth, physical correction: two decades of climate change.
# Only the DIFFERENCE of the two ERA5 normals is ever used, so the reference ships it precomputed
# (make_reference.py -> era5_shift.nc: 1 deg lat/lon for the surface fields, 1.5 deg lat_u/lon_u for the rest).
ERA5_SHIFT = "era5_shift.nc"
SFC_TAGS = ("t2m", "mslp", "pr", "olr")
BASE_FROM, BASE_TO = (2001, 2020), (1991, 2020)
# the animator loops these, per domain, in this order
ANIM_ORDER = ["t2m", "pr", "mslp", "zg500", "olr", "u850", "u200", "v850", "v200"]
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]


def reref(tag: str, a, base: pd.Timestamp):
    """Shift an anomaly from the 2001-2020 model base onto the 1991-2020 normal.

    The hindcast spans 2001-2020 only, so the model's 1991-2020 climatology does
    not exist. It is estimated as M_9120 ~= M_0120 + (ERA5_9120 - ERA5_0120), which
    makes the anomaly

        [F(L) - M_0120(L)]  -  (ERA5_9120 - ERA5_0120)

    — the existing drift- and bias-corrected anomaly, minus the observed change
    between the two reference periods. Only the period moves; the model's own bias
    stays removed, which is the whole value of having a hindcast. Differencing
    straight against ERA5 would also be "vs 1991-2020" but would hand the bias back
    (+0.55 mm/day on precipitation, +9.7 W/m2 on outgoing longwave)."""
    q = paths.CLIM / ERA5_SHIFT
    if not q.exists():
        raise SystemExit(f"need {q} — the geps-ref reference set (make_reference.py)")
    es = xr.open_dataset(q)
    if tag not in es:
        es.close()
        return None
    d = es[tag].load()                            # observed period change, ERA5 BASE_TO minus BASE_FROM
    if "lat_u" in d.dims:
        d = d.rename({"lat_u": "lat", "lon_u": "lon"})
    dw = xr.concat([d.isel(doy=-1).assign_coords(doy=d.doy[-1] - 365), d,
                    d.isel(doy=0).assign_coords(doy=d.doy[0] + 365)], dim="doy")
    step = float(d.lon[1] - d.lon[0])
    dw = xr.concat([dw, dw.isel(lon=0).assign_coords(
        lon=float(dw.lon[-1]) + step)], dim="lon")   # close the seam
    lat = "latitude" if "latitude" in a.dims else "lat"
    lon = "longitude" if "longitude" in a.dims else "lon"
    tgt_lat, tgt_lon = a[lat].values, np.sort(a[lon].values % 360)
    out = []
    for L in a.L.values:
        v = base + pd.Timedelta(days=int(L) - 1)        # day L (the 24 h ending at L) is the UTC day base + L - 1
        o = (dw.interp(doy=v.dayofyear).rename({"lat": lat, "lon": lon})
               .interp({lat: tgt_lat, lon: tgt_lon}))
        out.append(-o.values)                     # subtract the period change
    es.close()
    r = xr.DataArray(np.stack(out), dims=a.dims, coords=a.coords)
    if bool(np.isnan(r.values).any()):
        raise SystemExit(f"{tag}: NaN in the re-basing — check the seam wrap")
    return r


def load_fc(tag: str, cycle: str):
    """The ensemble-mean field itself, for contouring over the anomaly."""
    p = LIVE / f"geps_live_{tag}_{cycle}.nc"
    if not p.exists() or tag not in CONTOUR:
        return None
    d = xr.open_dataset(p)
    return d[f"{tag}_fc"] * FIELDS[tag][4]


def load(tag: str, cycle: str, base_kind: str = "model", base=None):
    p = LIVE / f"geps_live_{tag}_{cycle}.nc"
    if p.exists():
        a = xr.open_dataset(p)[f"{tag}_anom"]
    else:
        # an earlier cycle (the change panels): its compact 1 deg archive, rebuilt onto the 0.5 deg grid exactly as
        # forecast.anomaly did (archive.py)
        a = archive.load_anom(tag, cycle)
        if a is None:
            return None
    _, _, _, demean, scale = FIELDS[tag]
    if base_kind == "era5":
        r = reref(tag, a, base)
        if r is None:
            print(f"    {tag}: no ERA5 normal — staying on the model base")
        else:
            a = a + r
    if demean and "global_mean" in a.coords:
        a = a - xr.DataArray(a.global_mean.values, dims="L", coords={"L": a.L})
    return a * scale


def _panel(ax, a, dom, lim, cmap, over=None, clev=None, clw=0.5):
    if dom.get("proj") == "merc":
        # set_extent(-180, 180) is degenerate on a Mercator centred on 180 —
        # those bounds ARE the seam, so the x range collapses and the panel
        # renders as a bare vertical line. Take the full globe and clip in
        # projection coordinates instead.
        ax.set_global()
        pr = ax.projection
        e = dom["extent"]
        y0 = pr.transform_point(0.0, e[2], ccrs.PlateCarree())[1]
        y1 = pr.transform_point(0.0, e[3], ccrs.PlateCarree())[1]
        ax.set_ylim(y0, y1)
    else:
        ax.set_extent(dom["extent"], crs=ccrs.PlateCarree())
    lon = a.longitude.values.copy()
    v = a.values
    lon = np.where(lon > 180, lon - 360, lon)
    o = np.argsort(lon)
    m = ax.pcolormesh(lon[o], a.latitude.values, v[:, o], cmap=cmap,
                      vmin=-lim, vmax=lim, shading="auto",
                      transform=ccrs.PlateCarree(), rasterized=True)
    if over is not None and clev is not None:
        lo_c, ov = _wrap(lon[o], np.asarray(over)[:, o])
        cs = ax.contour(lo_c, a.latitude.values, ov, levels=clev, colors="#1f1f1f",
                        linewidths=clw, transform=ccrs.PlateCarree())
        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=6, fmt="%.0f")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#3a3a3a")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#6a6a6a")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.18, edgecolor="#9a9a9a")
    return m


_ASPECT: dict[str, float] = {}


def projection(key, dom):
    k = dom.get("proj", "plate")
    if k == "lambert":
        return ccrs.LambertConformal(central_longitude=dom["central"],
                                     standard_parallels=(33, 45))
    if k == "merc":
        return ccrs.Mercator(central_longitude=dom["central"],
                             min_latitude=dom["extent"][2],
                             max_latitude=dom["extent"][3])
    return ccrs.PlateCarree(central_longitude=dom["central"])


def aspect(key, dom) -> float:
    """Height/width of the domain IN THE PROJECTION, measured once.

    Sizing a figure from the lon/lat extent is wrong for Lambert Conformal: the
    projected box is a different shape, so cartopy pads the axes to preserve the
    equal aspect and the panel ends up with a band of dead space above the map.
    Measuring the projected extent removes the guesswork — and the whitespace."""
    if key not in _ASPECT and dom.get("proj") == "merc":
        # Analytic, not measured: cartopy's get_extent on a latitude-clipped
        # Mercator returns a degenerate y range and the figure height came out
        # as 7e17 pixels. Mercator y is log(tan(45 + lat/2)) and x spans 2*pi
        # for the full circle, so the ratio is exact.
        f_ = lambda d: np.log(np.tan(np.pi / 4 + np.deg2rad(d) / 2))
        e = dom["extent"]
        _ASPECT[key] = (f_(e[3]) - f_(e[2])) / (2 * np.pi)
    if key not in _ASPECT:
        f = plt.figure(figsize=(4, 4))
        a = f.add_subplot(1, 1, 1, projection=projection(key, dom))
        a.set_extent(dom["extent"], crs=ccrs.PlateCarree())
        x0, x1, y0, y1 = a.get_extent(projection(key, dom))
        _ASPECT[key] = abs((y1 - y0) / (x1 - x0))
        plt.close(f)
    return _ASPECT[key]


# Per-field percentile for the colour limit. 95 is right for fields whose
# extremes are small-scale noise, but temperature anomalies have a long, MEANINGFUL
# tail — the deep cold pool and the ridge are the story — and clipping at the 95th
# left large areas flat at the end colour. Higher percentile, wider scale.
SCALE_PCT = {"t2m": 99.3, "zg500": 98.0, "mslp": 99.0}

# Fields that get the ensemble-mean FIELD contoured over the anomaly fill. For
# pressure and height the anomaly says how unusual, and the contours say what the
# pattern actually is — where the ridge and trough sit, whether a low is closed.
# Standard intervals: 4 hPa and 60 m.
CONTOUR = {"mslp": (np.arange(880.0, 1084.0, 4.0), 0.6),
           "zg500": (np.arange(4680.0, 6121.0, 60.0), 0.5)}


def _wrap(lon, *fields):
    """Append the 0 deg column at 360 so contours close across the seam."""
    if lon[-1] - lon[0] < 350:                 # regional domain, nothing to wrap
        return (lon,) + fields
    lo = np.r_[lon, lon[0] + 360.0]
    return (lo,) + tuple(np.concatenate([f, f[:, :1]], axis=1) for f in fields)


def scale_for(a, pct=95):
    """Robust symmetric limit: the pct-th percentile of |anomaly|.

    (The first version took a percentile OF a percentile, which inflated the
    range to +-250 m and washed every map out.)"""
    v = np.abs(a.values[np.isfinite(a.values)])
    return float(np.nanpercentile(v, pct)) if v.size else 1.0


def _grid(stack, asp, n=5):
    """Figure + per-panel axes rectangles for n weekly panels.

    Returns (fig, rects, cax_rect, note_y, title_y). stack: five wide strips one
    above the other (global Mercator); otherwise a 2 x 2 x 1 grid with the odd
    panel centred. Explicit geometry: subplots_adjust and a colorbar attached to
    `axes` fight each other, which left the maps squashed into the lower half."""
    if stack:
        panel_w, top_in, bot_in = 11.0, 0.72, 1.15
        map_h = panel_w * asp
        fig_w = panel_w + 0.35
        fig_h = n * (map_h + 0.30) + top_in + bot_in
        fig = plt.figure(figsize=(fig_w, fig_h))
        rects = [[0.015, 1.0 - (top_in + (i + 1) * (map_h + 0.30)) / fig_h,
                  0.968, map_h / fig_h] for i in range(n)]
        cax = [0.36, 0.62 / fig_h, 0.28, 0.13 / fig_h]
        note_y = 0.12 / fig_h
    else:
        panel_w, top_in, bot_in = 5.8, 0.92, 0.98
        map_h = panel_w * asp
        gap_y = 0.44
        rows = (n + 1) // 2
        fig_w = 2 * panel_w + 0.30
        fig_h = rows * map_h + (rows - 1) * gap_y + top_in + bot_in
        fig = plt.figure(figsize=(fig_w, fig_h))
        w_frac = panel_w / fig_w
        rects = []
        for i in range(n):
            row, col = divmod(i, 2)
            y = 1.0 - (top_in + (row + 1) * map_h + row * gap_y) / fig_h
            last_alone = (i == n - 1 and n % 2 == 1)
            x = (1.0 - w_frac) / 2 if last_alone else 0.010 + col * (w_frac + 0.006)
            rects.append([x, y, w_frac * 0.98, map_h / fig_h])
        cax = [0.36, 0.44 / fig_h, 0.28, 0.15 / fig_h]
        note_y = 0.06 / fig_h
    return fig, rects, cax, note_y, 1.0 - 0.18 / fig_h


def previous_cycle(tag: str, cycle: str) -> str | None:
    """The newest archived cycle older than `cycle` for this field."""
    have = sorted({q.name.split("_")[-1][:8] for q in LIVE.glob(f"geps_live_{tag}_*.nc")}
                  | set(archive.cycles(f"geps_anom1_{tag}")))
    older = [c for c in have if c < cycle]
    return older[-1] if older else None


def change(tag, cycle, base, base_kind="model", only=None, prev=None):
    """Weekly means of (this cycle - previous cycle) on the COMMON valid days.

    The previous extended cycle started 3 or 4 days earlier (Mon/Thu), so its
    forecast for a given calendar day sits at a longer lead. Both anomalies are
    aligned on the VALID date and differenced where both exist: with a 3-day
    gap that is this cycle's days 1-32, so week 5 is a 4-day mean and says so
    in its title. Nothing is drawn where only one cycle has a forecast — a
    partial week padded with zeros would read as "no change" where the truth
    is "no comparison". Same colour map as the field, symmetric scale from the
    differences themselves."""
    prev = prev or previous_cycle(tag, cycle)
    if prev is None:
        print(f"  {tag}: no earlier cycle archived — no change panel"); return []
    a = load(tag, cycle, base_kind, base)
    pbase = pd.Timestamp(prev)
    b = load(tag, prev, base_kind, pbase)
    if a is None or b is None:
        return []
    gap = int((base - pbase).days)
    # previous cycle's lead for this cycle's day L is L + gap
    common = [int(L) for L in a.L.values if (int(L) + gap) in set(int(x) for x in b.L.values)]
    if not common:
        print(f"  {tag}: no common valid days with {prev}"); return []
    bb = b.sel(L=[L + gap for L in common]).assign_coords(L=common)
    d = a.sel(L=common) - bb
    title, unit, cmap, _, _ = FIELDS[tag]
    out = []
    for key, dom in DOMAINS.items():
        if only and key != only:
            continue
        sub = d.sel(latitude=slice(dom["extent"][3], dom["extent"][2])) \
              if d.latitude[0] > d.latitude[-1] else \
              d.sel(latitude=slice(dom["extent"][2], dom["extent"][3]))
        weeks = []
        for w0, w1 in WEEKS:
            days = [L for L in common if w0 <= L <= w1]
            if not days:
                continue
            weeks.append((w0, days[-1], len(days), sub.sel(L=days).mean("L")))
        lim = max(scale_for(m, SCALE_PCT.get(tag, 95)) for *_, m in weeks)
        asp = aspect(key, dom)
        proj = projection(key, dom)
        stack = dom.get("stack", False)
        fig, rects, cax_rect, note_y, title_y = _grid(stack, asp, len(weeks))
        for i, ((w0, w1, nd, mn), rect) in enumerate(zip(weeks, rects)):
            ax = fig.add_axes(rect, projection=proj)
            mesh = _panel(ax, mn, dom, lim, cmap)
            d0 = base + pd.Timedelta(days=w0 - 1)
            d1 = base + pd.Timedelta(days=w1 - 1)
            part = f" ({nd} common days)" if nd < 7 else ""
            ax.set_title((f"Week {i+1}   {d0:%b %d} – {d1:%b %d}{part}" if stack
                          else f"Week {i+1}{part}\n{d0:%b %d} – {d1:%b %d}"),
                         fontsize=10.5, fontweight="bold", pad=4,
                         loc="left" if stack else "center")
        cax = fig.add_axes(cax_rect)
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
        cb.ax.xaxis.set_label_position("top")
        cb.set_label(f"{title}: this run minus previous run ({unit})",
                     fontsize=10, labelpad=4)
        cb.ax.tick_params(labelsize=8.5, pad=1.5)
        fig.suptitle(f"{dom['label']} — {title}, change vs previous run · "
                     f"GEPS init {base:%Y-%m-%d} minus {pbase:%Y-%m-%d}",
                     fontsize=13.5, fontweight="bold", y=title_y, va="top")
        fig.text(0.5, note_y, f"weekly means over the valid days both cycles forecast "
                 f"(days 1–{common[-1]} of this run); red = this run more positive",
                 ha="center", va="bottom", fontsize=8.5, color="#8a8680")
        FIGS.mkdir(parents=True, exist_ok=True)
        p = FIGS / f"geps_{key}_{tag}_change.webp"
        fig.savefig(p, dpi=105, facecolor="white",
                    pil_kwargs={"quality": 88, "method": 6})
        plt.close(fig)
        out.append(p.name)
        print(f"  {p.name}")
    return out


SKILL_MIN = 0.25
TC = paths.TC_REF


def skill_mask(tag, base):
    """(lat, lon, [week] -> (noskill, notruth) boolean fields) from
    hindcast_maps.py, for this init month; None if the field was never
    scored."""
    q = TC / f"map_skill_{tag}.nc"
    if not q.exists():
        return None
    ds = xr.open_dataset(q)
    r = ds["r"].sel(month=base.month).values                    # (week, lat, lon)
    return ds.lat.values, ds.lon.values, r


def hatch_skill(ax, mask, i):
    """Hatch a weekly panel where the hindcast says the week has no usable
    skill (////) or no verification data at all (xxxx)."""
    lat, lon, r = mask
    rw = np.concatenate([r[i], r[i][:, :1]], axis=1)
    lon_w = np.r_[lon, 360.0]
    for fld, hatch in ((np.isfinite(rw) & (rw < SKILL_MIN), "////"), (~np.isfinite(rw), "xxxx")):
        if fld.any():
            ax.contourf(lon_w, lat, fld.astype(float), levels=[0.5, 1.5], colors="none",
                        hatches=[hatch], transform=ccrs.PlateCarree())


def weekly(tag, cycle, base, base_kind="model", only=None, masked=False):
    a = load(tag, cycle, base_kind, base)
    if a is None:
        return []
    mask = skill_mask(tag, base) if masked else None
    if masked and mask is None:
        return []
    fc = load_fc(tag, cycle)
    title, unit, cmap, demean, _ = FIELDS[tag]
    out = []
    for key, dom in DOMAINS.items():
        if only and key != only:
            continue
        sub = a.sel(latitude=slice(dom["extent"][3], dom["extent"][2])) \
              if a.latitude[0] > a.latitude[-1] else \
              a.sel(latitude=slice(dom["extent"][2], dom["extent"][3]))
        means = [sub.sel(L=slice(w0, w1)).mean("L") for w0, w1 in WEEKS]
        fcm = None
        if fc is not None:
            fsub = fc.sel(latitude=sub.latitude)
            fcm = [fsub.sel(L=slice(w0, w1)).mean("L").values for w0, w1 in WEEKS]
        pct = SCALE_PCT.get(tag, 95)
        lim = max(scale_for(m, pct) for m in means)
        asp = aspect(key, dom)
        proj = projection(key, dom)
        stack = dom.get("stack", False)
        fig, rects, cax_rect, note_y, title_y = _grid(stack, asp, len(WEEKS))
        for i, ((w0, w1), mn) in enumerate(zip(WEEKS, means)):
            ax = fig.add_axes(rects[i], projection=proj)
            clev, clw = CONTOUR.get(tag, (None, 0.5))
            mesh = _panel(ax, mn, dom, lim, cmap,
                          over=(fcm[i] if fcm else None), clev=clev, clw=clw)
            if mask is not None:
                hatch_skill(ax, mask, i)
            d0 = base + pd.Timedelta(days=w0 - 1)
            d1 = base + pd.Timedelta(days=w1 - 1)
            ax.set_title(f"Week {i+1}   {d0:%b %d} – {d1:%b %d}" if stack
                         else f"Week {i+1}\n{d0:%b %d} – {d1:%b %d}",
                         fontsize=10.5, fontweight="bold", pad=4,
                         loc="left" if stack else "center")
        cax = fig.add_axes(cax_rect)
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
        cb.ax.xaxis.set_label_position("top")
        cb.set_label(f"{title} anomaly vs 1991\u20132020 ({unit})"
                     + ("   ·   contours: ensemble mean" if tag in CONTOUR else ""),
                     fontsize=10, labelpad=4)
        cb.ax.tick_params(labelsize=8.5, pad=1.5)
        # Only note the thing a reader cannot infer from the colourbar. The
        # colourbar already says "anomaly"; what the anomaly is measured against
        # belongs on the page, not repeated on every figure. The global-mean
        # removal DOES need saying, because it changes what the numbers mean.
        note = "global mean removed per lead" if demean else ""
        if mask is not None:
            note = (note + "   ·   " if note else "") + \
                f"hatched: hindcast anomaly correlation below {SKILL_MIN} for this week and season (no usable skill)" + \
                ("; cross-hatched: no gauge truth (ocean)" if tag == "pr" else "")
        fig.suptitle(f"{dom['label']} — {title}, weekly means · GEPS init {base:%Y-%m-%d}",
                     fontsize=15, fontweight="bold", y=title_y, va="top")
        if note:
            fig.text(0.5, note_y, note, ha="center",
                     va="bottom", fontsize=8.5, color="#8a8680")
        FIGS.mkdir(parents=True, exist_ok=True)
        p = FIGS / f"geps_{key}_{tag}_weekly{'_masked' if mask is not None else ''}.webp"
        fig.savefig(p, dpi=105, facecolor="white",
                    pil_kwargs={"quality": 88, "method": 6})
        plt.close(fig)
        out.append(p.name)
        print(f"  {p.name}")
    return out


def daily(tag, cycle, base, base_kind="model", only=None):
    a = load(tag, cycle, base_kind, base)
    if a is None:
        return {}
    fc = load_fc(tag, cycle)
    title, unit, cmap, demean, _ = FIELDS[tag]
    man = {}
    for key, dom in DOMAINS.items():
        if only and key != only:
            continue
        sub = a.sel(latitude=slice(dom["extent"][3], dom["extent"][2])) \
              if a.latitude[0] > a.latitude[-1] else \
              a.sel(latitude=slice(dom["extent"][2], dom["extent"][3]))
        fsub = fc.sel(latitude=sub.latitude) if fc is not None else None
        clev, clw = CONTOUR.get(tag, (None, 0.5))
        lim = scale_for(sub, SCALE_PCT.get(tag, 95))
        # Fixed geometry, NOT bbox_inches="tight": tight crops to the drawn
        # content, so every frame comes out a slightly different pixel size and
        # the loop jitters. Explicit add_axes gives identical frames, and the
        # margins below are sized in INCHES so the title and colorbar always fit.
        asp = aspect(key, dom)
        proj = projection(key, dom)
        fw, top_in, bot_in = 7.6, 0.62, 0.66
        map_h = fw * asp
        fig_h = map_h + top_in + bot_in
        # flat "<domain>_<field>" directories, not nested: the site's frame player
        # only routes a base ending in /anim to the frames branch, so the region id
        # has to carry the domain (see sst_anim.html ON_FRAMES_BRANCH)
        d = FIGS / "anim" / f"{key}_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, L in enumerate(sub.L.values):
            fig = plt.figure(figsize=(fw, fig_h))
            ax = fig.add_axes([0.0, bot_in / fig_h, 1.0, map_h / fig_h], projection=proj)
            mesh = _panel(ax, sub.sel(L=L), dom, lim, cmap,
                          over=(fsub.sel(L=L).values if fsub is not None else None),
                          clev=clev, clw=clw)
            valid = base + pd.Timedelta(days=int(L) - 1)
            ax.set_title(f"{dom['label']} — {title} anomaly\n"
                         f"GEPS init {base:%b %d} · day {int(L)}, valid {valid:%a %d %b}",
                         fontsize=11, fontweight="bold", pad=6)
            cax = fig.add_axes([0.30, 0.34 / fig_h, 0.40, 0.14 / fig_h])
            cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
            cb.ax.xaxis.set_label_position("top")
            cb.set_label(f"{title} anomaly vs 1991\u20132020 ({unit})"
                         + ("   ·   contours: ensemble mean" if tag in CONTOUR else ""),
                         fontsize=9, labelpad=3)
            cb.ax.tick_params(labelsize=8, pad=1.5)
            f = d / f"F{i:02d}.webp"
            # loop frames: WebP q65 method 6 (2026-10-05 encode study: ~30-35 % fewer bytes than q82, text/contours/colour bars unchanged at 1x)
            fig.savefig(f, dpi=100, facecolor="white",
                        pil_kwargs={"quality": 65, "method": 6})
            plt.close(fig)
            frames.append({"idx": i, "file": f.name,
                           "date": valid.strftime("%Y-%m-%d"),
                           "label": f"day {int(L)} · {valid:%b %d}"})
        man[f"{key}_{tag}"] = {"label": title, "frames": frames}
        print(f"  {key}_{tag}: {len(frames)} frames")
    return man


def _job(spec):
    """One (field, domain) unit of work, run in its own process.

    The unit is a whole field/domain rather than a single frame: each worker
    opens the one netCDF it needs and renders all 35 leads, so the only thing
    crossing the process boundary is the small manifest that comes back. Frame-
    level tasks would pickle the field array 35 times for no gain.

    Rendering is the only part of the daily cycle worth parallelising — the
    downloads are network-bound and adding processes there just contends.
    """
    tag, key, cycle, base, base_kind, skip_daily, prev = spec
    import matplotlib
    matplotlib.use("Agg")
    weekly(tag, cycle, base, base_kind, only=key)
    weekly(tag, cycle, base, base_kind, only=key, masked=True)   # no-op without a skill file
    change(tag, cycle, base, base_kind, only=key, prev=prev)
    if skip_daily:
        return {}
    return daily(tag, cycle, base, base_kind, only=key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--tags", default=",".join(FIELDS))
    ap.add_argument("--skip-daily", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="render processes; frames are independent so this scales")
    ap.add_argument("--domains", default=",".join(DOMAINS),
                    help="restrict to some domains, e.g. --domains glb")
    ap.add_argument("--prev", help="cycle to difference against for the change "
                    "panels (default: the newest archived cycle before --cycle)")
    ap.add_argument("--base", default="era5", choices=("model", "era5"),
                    help="anomaly reference: the ERA5 1991-2020 WMO normal "
                         "(default) or the raw GEPS hindcast base")
    a = ap.parse_args()
    base = pd.Timestamp(a.cycle)
    want = [d.strip() for d in a.domains.split(",") if d.strip()]
    for k in list(DOMAINS):
        if k not in want:
            DOMAINS.pop(k)
    FIGS.mkdir(parents=True, exist_ok=True)
    regions = {}
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    tags = [t for t in tags if (LIVE / f"geps_live_{t}_{a.cycle}.nc").exists()]
    specs = [(t, k, a.cycle, base, a.base, a.skip_daily, a.prev)
             for t in tags for k in DOMAINS]
    n = min(a.workers, len(specs)) or 1
    print(f"rendering {len(specs)} field/domain units on {n} processes", flush=True)
    t0 = time.time()
    if n == 1:
        for sp in specs:
            regions.update(_job(sp))
    else:
        with ProcessPoolExecutor(max_workers=n) as ex:
            for i, r in enumerate(ex.map(_job, specs), 1):
                regions.update(r)
                if i % 5 == 0 or i == len(specs):
                    el = time.time() - t0
                    print(f"  {i}/{len(specs)}  {el/60:.1f} min elapsed  "
                          f"~{(len(specs)-i)*el/i/60:.1f} min left", flush=True)
    if regions:
        ver = int(pd.Timestamp.now().timestamp())
        for key, dom in DOMAINS.items():
            sub = {k: v for k, v in regions.items() if k.startswith(key + "_")}
            if not sub:
                continue
            order = [f"{key}_{t}" for t in ANIM_ORDER if f"{key}_{t}" in sub]
            man = {"ver": ver, "selectorLabel": "Field",
                   "regions": {t: sub[t] for t in order},
                   "default": order[0]}
            out = FIGS / "anim" / f"geps_{key}_manifest.json"
            out.write_text(json.dumps(man))
            print(f"  {out.name}: {len(order)} fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
