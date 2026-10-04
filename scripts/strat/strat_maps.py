#!/usr/bin/env python3
"""10 and 100 hPa temperature, wind and height ANOMALY maps — NH polar vortex.

One figure per level: three rows (temperature, zonal wind, geopotential height)
by four columns (analysis, day 5, day 10, day 15), all as anomalies against the
ERA5 1991-2020 day-of-year climatology from fetch_strat_clim.py.

Why anomalies and not raw fields. In early winter the whole polar cap is cooling
fast, so a raw 10 hPa temperature map looks equally cold every year and says
nothing; a raw height map is dominated by the seasonal collapse of the vortex.
The anomaly is the part that carries information, and until now the strat
products had no gridded temperature climatology at all to form one.

What each row is for:

  T   the warming that IS the event. A sudden stratospheric warming shows here
      first and most clearly, and the sign convention is the opposite of the
      surface intuition: warm cap = weak vortex = the pattern that couples down.
  U   zonal wind, with the ZERO LINE drawn heavy. The WMO definition of a major
      SSW is a reversal of the zonal-mean zonal wind at 10 hPa, 60N, so a zero
      contour crossing the 60N circle is the diagnostic, visible directly.
  Z   height anomaly with the FULL height field contoured over it. The contours
      show whether the vortex is displaced (one centre, off the pole) or split
      (two centres) — a distinction the anomaly shading alone cannot make.

The ensemble mean is weighted by MEMBER COUNT: a GEPS `allmbrs` file holds 20
perturbed members plus one control, and averaging the perturbed mean and the
control as two equal things gives the control 1/2 the weight instead of 1/21,
which leaks a single member's noise into every panel.

    python strat_maps.py                    # latest GEPS cycle, both levels
    python strat_maps.py --hemi sh
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyproj  # noqa: F401,E402  before xarray/eccodes: eccodes preloads its own libproj (CLAUDE.md gotcha)
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
from cartopy.util import add_cyclic_point

HERE = Path(__file__).resolve().parent
# every Datamart request goes through the GEPS pipeline's fetcher (honest User-Agent, retries, per-run ledger)
sys.path.insert(0, str(HERE.parent / "geps"))
import datamart                                                  # noqa: E402
import paths as _gp                                              # noqa: E402
DATA = HERE / "data"
DL = Path(os.environ.get("STRAT_DL", DATA / "vortex_dl"))        # GRIB cache, shared with forecast.py (500 hPa)
CLIM = DATA / "strat_clim_1991-2020.nc"
TREND = DATA / "strat_trend.nc"
# these boards live on the PUBLIC subseasonal page (same GEPS chain, same
# climatology discipline), not with the private strat research under assets/strat
OUT = _gp.SITE_ASSETS
UA = _gp.UA
GEPS = ("https://dd.weather.gc.ca/{d}/WXO-DD/ensemble/geps/grib2/raw/{c}/{L:03d}/"
        "CMC_geps-raw_{v}_ISBL_{lev:04d}_latlon0p5x0p5_{d}{c}_P{L:03d}_allmbrs.grib2")
# The EXTENDED (Mon/Thu) cycle publishes 10 and 100 hPa all the way to day 35,
# the same cycle the rest of the subseasonal page runs on — so the vortex board
# shares an init with everything beside it instead of drifting off on the daily
# run, and reaches day 35 rather than day 16.
LEADS = [0, 240, 480, 840]
FRAME_LEADS = list(range(0, 841, 24))
# Rows. "anom" panels are departures on a diverging scale; the wind row is the
# TOTAL wind speed on an absolute sequential scale with geopotential height
# contoured over it — the classic vortex view, where the contours show the shape
# (one centre displaced, or two centres split) and the fill shows the jet that
# encircles it. A zonal-wind anomaly cannot show either.
# field -> (key, GEPS var, clim key, label, units, colormap, mode)
# 10 hPa is shown ABSOLUTE, not as an anomaly. The GEPS8 hindcast carries no
# 10 hPa level at all (ua/va stop at 100, zg at 200/500, and there is no
# pressure-level temperature), so a 10 hPa anomaly can only be taken against
# ERA5 — which removes none of the model's own drift, and the upper stratosphere
# is exactly where that drift is largest. Showing the absolute state is honest;
# calling the difference an anomaly would not be. 100 hPa keeps its anomalies.
# (2026-09-27: the S2S-archive GEPS reforecasts DO carry 10 hPa T and height; strat_frames.map_bias removes their
# measured drift and turns 10 hPa anomalies on whenever a reforecast start date lies within 8 days of the init's day
# of year. This constant is the fallback when none does. The 10 hPa cap T drift is +3 K at day 15, +7 K at day 35.)
ANOM_LEVELS = (100,)
# Temperature is dropped entirely at 10 hPa. With no hindcast level there it can
# only be drawn absolute, and an absolute 10 hPa temperature map is dominated by
# the seasonal march of the polar night — it looks the same every year and says
# nothing a reader can act on. Wind and height still carry the vortex.
DROP = {10: ("T",)}


def fields_for(lev):
    return [f for f in FIELDS if f[0] not in DROP.get(lev, ())]
FIELDS = [("T", "TMP",  "T", "Temperature",  "K",  "RdBu_r", "anom"),
          ("W", "UGRD", "U", "Total wind",   "kt", None,     "speed"),
          ("Z", "HGT",  "Z", "Geopotential height", "m", "RdBu_r", "anom")]

MS2KT = 1.94384
mpl.rcParams["hatch.linewidth"] = 0.4
mpl.rcParams["hatch.color"] = "#5a5a5a"
# The same isotach ladder the site's GDPS 150 hPa charts use
# (scripts/sst/gdps_eq_charts.py, W150_LEV / W150_COLS), so upper-level wind
# reads identically wherever it appears here. White below 25 kt keeps the
# quiescent summer hemisphere blank instead of tinting it, and magenta above
# 150 kt still resolves the SH polar-night jet, which runs past 170 kt.
# Two bands added above the GDPS ladder (170, 190 kt): the SH polar-night jet at
# 10 hPa runs past 170 kt, so stopping at 150 turned the whole vortex core into
# one flat magenta blob and threw away exactly the structure the panel is for.
WLEV = [25, 35, 45, 55, 65, 80, 95, 110, 130, 150, 170, 190]       # knots
WCOLS = ["#dbeef8", "#a9d4ec", "#6fb4de", "#3f8fc7", "#61b26b",
         "#b5d24a", "#f2d03a", "#f29b2c", "#e0562c", "#b0289b", "#7b1d78"]
WCMAP = ListedColormap(WCOLS)
WCMAP.set_under("#ffffff"); WCMAP.set_over("#4a1050")
WNORM = BoundaryNorm(WLEV, WCMAP.N)


def fetch(date, cyc, var, lev, L):
    dest = DL / f"geps_{date}{cyc}_{var}{lev}_{L:03d}.grib2"
    got = datamart.download(GEPS.format(d=date, c=cyc, v=var, lev=lev, L=L), dest, timeout=300)
    if got is None:
        print(f"    miss {var}{lev} +{L}h", flush=True)
    return got


def ens_mean(path):
    """Member-count-weighted mean of the perturbed file and the control."""
    parts = []
    for dt in ("pf", "cf"):
        try:
            ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
                filter_by_keys={"dataType": dt}, indexpath=""))
            a = ds[list(ds.data_vars)[0]]
            n = a.sizes["number"] if "number" in a.dims else 1
            parts.append(((a.mean("number") if "number" in a.dims else a), n))
        except Exception:                                        # noqa: BLE001
            pass
    if not parts:
        return None
    tot = sum(n for _, n in parts)
    return (sum(a * n for a, n in parts) / tot).load()


def to_clim_grid(a, clat, clon):
    """0.5 deg GEPS -> the 1 deg climatology grid, band-limited first.

    Same reasoning as the subseasonal maps: differencing a finer forecast against
    a coarser climatology leaves the unresolved terrain/land-sea structure in the
    anomaly. Here the levels are stratospheric so the effect is small, but the
    difference still belongs on the reference's grid."""
    lat = "latitude" if "latitude" in a.dims else "lat"
    lon = "longitude" if "longitude" in a.dims else "lon"
    a = a.assign_coords({lon: a[lon].values % 360}).sortby(lon).sortby(lat)
    for dim, per in ((lat, False), (lon, True)):
        lo, hi = a.shift({dim: 1}), a.shift({dim: -1})
        if per:
            lo = xr.where(lo.isnull(), a.roll({dim: 1}, roll_coords=False), lo)
            hi = xr.where(hi.isnull(), a.roll({dim: -1}, roll_coords=False), hi)
        else:
            lo, hi = lo.fillna(a), hi.fillna(a)
        a = 0.25 * lo + 0.5 * a + 0.25 * hi
    out = a.interp({lat: clat, lon: clon % 360})
    # a pole is a single point: the latitude filter above mixes the 89.5 ring into
    # it, which then renders as a starburst. Force each pole to its zonal mean.
    for i in (0, -1):
        if abs(abs(float(out[lat][i])) - 90.0) < 1e-6:
            out[{lat: i}] = out.isel({lat: i}).mean(lon)
    return out


def demean_height(key, an, clat, north):
    """Remove the hemispheric mean from HEIGHT anomalies.

    Geopotential height anomalies at these levels carry a large domain-wide
    offset that is not circulation. Two things drive it and neither belongs on a
    vortex map: the slow rise of the whole surface as the troposphere warms, and
    — far larger at 10 hPa — the QBO, which moves tropical and subtropical
    stratospheric heights by well over 100 m from year to year. In late August
    2026 the tropical 10 hPa height sits +216 m above the 1991-2020 mean, the
    largest in the 36-year record, and the northern extratropics with it at
    +106 m. Left in, that paints the entire height panel one colour and buries
    the vortex, which is the only thing the panel is for.

    What matters for the vortex — and for the AO/NAO response it couples into —
    is the polar cap RELATIVE to its surroundings, so the area-weighted mean over
    the plotted hemisphere is removed. Temperature and wind are left alone: their
    domain-mean offsets are small against their own variability.
    """
    if key != "Z":
        return an
    m = (clat >= 20) if north else (clat <= -20)
    w = np.cos(np.deg2rad(clat[m]))
    return an - float((an[m].mean(1) * w).sum() / w.sum())


def clim_at(c, key, lev, valid, tr=None):
    """Climatology for one field/level at a valid date, referenced to its EPOCH.

    The day-of-year interpolation is the easy half. The important half is the
    epoch shift: the climatology is centred on ~2005 and the forecast is 2026, so
    for geopotential height — which is rising steadily as the troposphere warms —
    an uncorrected reference makes an ordinary polar cap read tens of metres high
    and manufactures a weak vortex out of climate drift.

    The correction is applied only where the trend is BOTH significant (|b| > 2
    standard errors) and material (the shift exceeds 10% of the field's own
    residual sd). Winds fail that gate everywhere, so they are left alone."""
    cv = c[key].sel(level=lev)
    cw = xr.concat([cv.isel(doy=-1).assign_coords(doy=cv.doy[-1] - 365), cv,
                    cv.isel(doy=0).assign_coords(doy=cv.doy[0] + 365)], dim="doy")
    ref = cw.interp(doy=valid.dayofyear)
    if tr is None:
        return ref, 0.0
    dt = (valid.year + (valid.dayofyear - 1) / 365.25) - float(tr.attrs["epoch"])
    b = tr[key].sel(level=lev).values
    se = tr[f"{key}_se"].sel(level=lev).values
    sd = tr[f"{key}_sd"].sel(level=lev).values
    adj = b * dt
    keep = (np.abs(b) > 2 * se) & (np.abs(adj) > 0.10 * sd)
    adj = np.where(keep, adj, 0.0)
    return ref + adj, float(np.abs(adj).max())


def latest_extended(cyc="00"):
    """Newest GEPS cycle that reaches day 35 — Monday and Thursday only."""
    day = pd.Timestamp.utcnow().tz_localize(None).normalize()
    for back in range(8):
        d = (day - pd.Timedelta(days=back)).strftime("%Y%m%d")
        if datamart.exists(GEPS.format(d=d, c=cyc, v="TMP", lev=10, L=840)):
            return d
    return None


def cyc(field, lon):
    """Append the 0 deg column at 360 so contourf closes the longitude circle.

    Without it every filled panel carries a thin unfilled wedge along the prime
    meridian — contourf, unlike pcolormesh, does not know the axis is periodic."""
    out, lo = add_cyclic_point(field, coord=lon)
    return out, lo


def nice_levels(lim, n=24):
    """Symmetric contour levels on round numbers, so the colourbar reads
    -100/-50/0/50/100 rather than -4.870/-3.652/..."""
    lv = MaxNLocator(nbins=n, symmetric=True).tick_values(-lim, lim)
    return lv[lv != 0] if len(lv) > 40 else lv


def sm_fetch_mean(date, cyc_, var, lev, L):
    f = fetch(date, cyc_, var, lev, L)
    return ens_mean(f) if f else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--cyc", default="00")
    ap.add_argument("--hemi", default="nh", choices=("nh", "sh"))
    ap.add_argument("--levels", default="10,100")
    ap.add_argument("--clim", default=str(CLIM))
    a = ap.parse_args()
    cp = Path(a.clim)
    if not cp.exists():
        raise SystemExit(f"need {cp.name} — run fetch_strat_clim.py first")
    c = xr.open_dataset(cp)
    tr = xr.open_dataset(TREND) if TREND.exists() else None
    if tr is None:
        print("  (no strat_trend.nc — climatology NOT epoch-referenced)", flush=True)
    clat, clon = c.lat.values, c.lon.values

    date = a.date
    if date is None:                       # newest cycle that has the day-15 step
        day = pd.Timestamp.utcnow().tz_localize(None).normalize()
        for back in range(8):
            d = (day - pd.Timedelta(days=back)).strftime("%Y%m%d")
            if datamart.exists(GEPS.format(d=d, c=a.cyc, v="TMP", lev=10, L=LEADS[-1])):
                date = d; break
        if date is None:
            raise SystemExit("no GEPS cycle found")
    base = pd.Timestamp(f"{date} {a.cyc}:00")
    print(f"GEPS {base:%Y-%m-%d %H}Z, {a.hemi.upper()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    north = a.hemi == "nh"
    proj = ccrs.NorthPolarStereo(central_longitude=0) if north else \
        ccrs.SouthPolarStereo(central_longitude=0)
    lat0, lat1 = (20, 90) if north else (-90, -20)
    phi = 60 if north else -60
    sub = (clat >= lat0) & (clat <= lat1)          # colour-scale domain
    capmask = (clat >= 65) if north else (clat <= -65)

    leads = LEADS
    for lev in [int(x) for x in a.levels.split(",")]:
        flds = fields_for(lev)
        nrow, ncol = len(flds), len(leads)
        fig = plt.figure(figsize=(3.25 * ncol + 1.6, 3.35 * nrow + 1.42))
        # one VERTICAL colourbar per row, at the right. Three horizontal bars
        # stacked under the panels overlapped each other's tick labels and ate
        # the height the maps needed.
        left, bot, top = 0.052, 0.028, 0.862
        w = (0.885 - left) / ncol
        h = (top - bot) / nrow
        meshes = {}
        for r, (key, var, ckey, label, unit, cmap, mode0) in enumerate(flds):
            mode = mode0 if (mode0 == "speed" or lev in ANOM_LEVELS) else "abs"
            fields, lim, vmin = [], 0.0, np.inf
            for L in leads:
                valid = base + pd.Timedelta(hours=L)
                # every row needs the height field: the wind row contours it,
                # the height row is its anomaly
                fz = sm_fetch_mean(date, a.cyc, "HGT", lev, L)
                if fz is None:
                    fields.append(None); continue
                gz = to_clim_grid(fz, clat, clon).values
                if mode == "speed":
                    fu = sm_fetch_mean(date, a.cyc, "UGRD", lev, L)
                    fv = sm_fetch_mean(date, a.cyc, "VGRD", lev, L)
                    if fu is None or fv is None:
                        fields.append(None); continue
                    u = to_clim_grid(fu, clat, clon).values
                    v = to_clim_grid(fv, clat, clon).values
                    fld = np.hypot(u, v) * MS2KT
                    lim = max(lim, float(np.nanpercentile(fld[sub], 99)))
                    cav, u60 = None, float(np.interp(phi, clat[::-1], u.mean(1)[::-1]))
                    fields.append((fld, valid, gz, cav, u60, u))
                else:
                    src = fz if key == "Z" else sm_fetch_mean(date, a.cyc, var, lev, L)
                    if src is None:
                        fields.append(None); continue
                    fc = to_clim_grid(src, clat, clon)
                    wc = np.cos(np.deg2rad(clat[capmask]))
                    if mode == "abs":
                        an = fc.values
                        vmin = min(vmin, float(np.nanpercentile(an[sub], 1)))
                        lim = max(lim, float(np.nanpercentile(an[sub], 99)))
                    else:
                        ref, _ = clim_at(c, ckey, lev, valid, tr)
                        an = demean_height(key, fc.values - ref.values, clat, north)
                        lim = max(lim, float(np.nanpercentile(np.abs(an[sub]), 98)))
                    cav = float((an[capmask].mean(1) * wc).sum() / wc.sum())
                    fields.append((an, valid, gz, cav, None, None))
            for col, item in enumerate(fields):
                ax = fig.add_axes([left + col * w, top - (r + 1) * h + 0.012,
                                   w * 0.97, h * 0.88], projection=proj)
                ax.set_extent([-180, 180, lat0, lat1], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4,
                               edgecolor="#555")
                ax.gridlines(lw=0.3, color="#bbb", ylocs=[phi], draw_labels=False)
                if item is None:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
                    continue
                fld, valid, gz, cav, u60, uu = item
                if mode == "speed":
                    fc_, lo_ = cyc(fld, clon)
                    gz_, _ = cyc(gz, clon)
                    uu_, _ = cyc(uu, clon)
                    msh = ax.contourf(lo_, clat, fc_, levels=WLEV, cmap=WCMAP,
                                      norm=WNORM, extend="both",
                                      transform=ccrs.PlateCarree())
                    ax.contour(lo_, clat, gz_, levels=10, colors="#1a1a1a",
                               linewidths=0.6, transform=ccrs.PlateCarree())
                    # Total wind is a MAGNITUDE and cannot be negative, so the sign
                    # of the flow has to be drawn separately: hatching marks where
                    # the zonal wind is EASTERLY, and its boundary (u = 0) is the
                    # WMO sudden-warming criterion.
                    ax.contourf(lo_, clat, uu_, levels=[-1e9, 0.0], colors="none",
                                hatches=["//"], transform=ccrs.PlateCarree())
                    ax.contour(lo_, clat, uu_, levels=[0], colors="#c2410c",
                               linewidths=1.1, transform=ccrs.PlateCarree())
                else:
                    # contourf, not pcolormesh: on a polar projection pcolormesh
                    # draws 360 cells converging on one point and their edges show
                    # as a starburst at the pole.
                    fc_, lo_ = cyc(fld, clon)
                    lv = (MaxNLocator(nbins=22).tick_values(vmin, lim) if mode == "abs"
                          else nice_levels(lim))
                    msh = ax.contourf(lo_, clat, fc_, levels=lv,
                                      cmap=("viridis" if mode == "abs" else cmap),
                                      extend="both", transform=ccrs.PlateCarree())
                    if key == "Z":
                        gz_, _ = cyc(gz, clon)
                        ax.contour(lo_, clat, gz_, levels=8, colors="#333",
                                   linewidths=0.5, transform=ccrs.PlateCarree())
                meshes[r] = (msh, label, unit, lim, mode)
                note = (f"u(60{'N' if north else 'S'}) {u60 * MS2KT:+.0f} kt" if u60 is not None
                        else (f"cap {cav:.0f}" if mode == "abs" else f"cap {cav:+.1f}")
                        + (" m" if key == "Z" else " K"))
                ax.text(0.5, -0.045, note, transform=ax.transAxes, ha="center",
                        va="top", fontsize=8.5, color="#2c4a72", fontweight="bold")
                if r == 0:
                    ax.set_title(("analysis" if leads[col] == 0 else f"day {leads[col]//24}")
                                 + f"\n{valid:%a %d %b}", fontsize=10, fontweight="bold",
                                 pad=4)
            fig.text(0.012, top - (r + 0.5) * h, label, rotation=90, va="center",
                     ha="center", fontsize=10.5, fontweight="bold")
        for r, (msh, label, unit, lim, mode) in meshes.items():
            y0 = top - (r + 1) * h + 0.035
            cax = fig.add_axes([0.905, y0, 0.013, h * 0.72])
            tk = WLEV if mode == "speed" else msh.levels[::3]
            cb = fig.colorbar(msh, cax=cax, orientation="vertical",
                              extend="both", ticks=tk)
            cb.set_label(f"{label}" + (" anom" if mode == "anom" else "")
                         + f" ({unit})", fontsize=9, labelpad=3)
            cb.ax.tick_params(labelsize=7.5, pad=1.5)
        fig.suptitle(f"{'Northern' if north else 'Southern'} polar vortex — {lev} hPa\n"
                     f"GEPS ensemble mean · init {base:%Y-%m-%d %H}Z",
                     fontsize=13, fontweight="bold", y=0.99, va="top")
        p = OUT / f"strat_{a.hemi}_{lev}hPa_maps.webp"
        fig.savefig(p, dpi=110, facecolor="white", pil_kwargs={"quality": 90, "method": 6})
        plt.close(fig)
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
