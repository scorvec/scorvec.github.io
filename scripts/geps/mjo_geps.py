#!/usr/bin/env python3
"""MJO diagnostics from the live GEPS extended cycle.

Two products, both built from the de-drifted anomalies forecast.py writes
(forecast minus the GEPS8 2001-2020 hindcast climatology at the SAME lead):

  phase      Wheeler & Hendon RMM phase diagram — observed last 40 days plus
             the GEPS ensemble-mean track, days 1-35.
  hovmoller  15S-15N OLR anomaly, lead x longitude, with u850 contours: the
             direct view of convective propagation.

This is the FULL three-field Wheeler & Hendon projection (OLR + U850 + U200),
not the wind-only version the site's AIFS product is limited to. AIFS does not
output OLR at all; GEPS does, and olr_channel.py supplies the observed OLR side
in real time (ERA5 spliced with the offset-corrected GMGSI proxy), so the
120-day filter map the third channel needs now exists. Both the forecast track
and the observed track here are three-field, so they are like-for-like.

Reference frames, which are easy to get wrong and were got wrong here first.
Anomalising the forecast against the GEPS model climatology removes drift, but
it also puts the forecast in the MODEL's frame while the observed track sits in
the observed frame; the two differ by the model bias, which for the band-mean
channels is 0.18-0.37 sigma in pattern. That showed up exactly where you would
expect — a jump across the analysis/forecast junction (amplitude 0.86 to 1.53
in one day) on the first version of this plot.

The fix keeps both properties instead of trading one for the other. Write M for
the model climatology and O for the observed one, both at the valid date:

    A(L) = [F(L) - M(L)] + [M(L=0) - O]
         = [F(L) - O] - [M(L) - M(L=0)]

which is the observed-frame anomaly minus the model's drift RELATIVE TO ITS OWN
day-0 state. At L=0 it reduces to F - O and joins the analysis track continuously;
at long lead it still removes the lead-dependent drift the hindcast archive exists
to measure. What it deliberately does not remove is the model's stationary bias,
because that bias is common to the analysis and would otherwise be double-counted.

The 120-day filter maps then come from analysis in the observed frame, which is
now the same frame the forecast is in.

One property to read honestly: this projects the ENSEMBLE MEAN, so the
amplitude damps with lead by construction as members decohere. A shrinking
radius is loss of ensemble agreement, not a forecast that the MJO decays.

    python mjo_geps.py --cycle 20260827
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths                                                   # noqa: E402
import refpack                                                 # noqa: E402
import datamart                                                # noqa: E402
# the site's mjo modules import each other flat (recent_analysis -> setup_reference),
# so src/ has to be on the path as well as the package root
_MJO = paths.REPO / "scripts" / "mjo"
sys.path[:0] = [str(_MJO), str(_MJO / "src")]
import olr_channel                                             # noqa: E402
import plot                                                    # noqa: E402
import recent_analysis as ra                                   # noqa: E402

HERE = Path(__file__).resolve().parent
LIVE = paths.LIVE
FIGS = paths.FIGS
REF = _MJO / "data" / "reference"
ANL_CACHE = paths.STATE / "mjo_analysis"      # band-mean U850/U200 of each day's GEPS lead-0 analysis

LAT_MIN, LAT_MAX = -15.0, 15.0
PHASE_LABEL = ["", "Western\nhemisphere,\nAfrica", "Indian\nOcean", "Indian\nOcean",
               "Maritime\ncontinent", "Maritime\ncontinent", "Western\nPacific",
               "Western\nPacific", "Western\nhemisphere,\nAfrica"]


def band_mean(a: xr.DataArray, lon_target: np.ndarray) -> np.ndarray:
    """cos-weighted 15S-15N mean, interpolated to the 2.5 deg EOF longitudes.

    Returns (lead, nlon)."""
    lat = a.latitude
    m = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    w = np.cos(np.deg2rad(lat)).where(m, 0.0)
    b = a.weighted(w).mean(dim="latitude")
    b = b.assign_coords(longitude=b.longitude % 360).sortby("longitude")
    # wrap so 357.5->0 interpolates across the dateline seam
    b = xr.concat([b, b.isel(longitude=0).assign_coords(longitude=360.0)], dim="longitude")
    return b.interp(longitude=lon_target).transpose("L", "longitude").values


def load(tag: str, cycle: str) -> xr.DataArray | None:
    p = LIVE / f"geps_live_{tag}_{cycle}.nc"
    return xr.open_dataset(p)[f"{tag}_anom"] if p.exists() else None


GEPS_URL = datamart.URL.replace("{c}", "00")
MVARS = {"u850": ("UGRD_ISBL_0850", False), "u200": ("UGRD_ISBL_0200", False),
         "olr": ("OLR_NTAT_0", True)}
SUBSTEPS = (-18, -12, -6, 0)


def _grib_members(date, var, L):
    """All 21 members of one allmbrs GRIB as (member, lat, lon), or None.

    forecast.py stores only the ensemble mean, which is all the mean track needs;
    a plume needs the members themselves, so this goes back to the GRIBs. The
    perturbed and control files are stacked rather than averaged."""
    import os, tempfile
    # mkstemp hands back an OPEN fd as well as a path. Keeping only the
    # path leaks that fd on every call: unlink() drops the name, not the
    # descriptor. At 35 days x 4 substeps = 140 grabs per field it blew
    # through launchd's 256-fd limit partway into the second field and the
    # 2026-08-31 cycle died as "Too many open files" (plus DNS failures,
    # which were sockets that could no longer be opened either).
    _fd, _name = tempfile.mkstemp(suffix=".grib2")
    os.close(_fd)
    tmp = Path(_name)
    try:
        body = datamart.get(GEPS_URL.format(d=date, v=var, L=L))
        if body is None:
            return None, None
        tmp.write_bytes(body)
        out = []
        for dt in ("pf", "cf"):
            try:
                d = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs=dict(
                    filter_by_keys={"dataType": dt}, indexpath=""))
                a = d[list(d.data_vars)[0]].load()
                out.append(a.values[None] if "number" not in a.dims else a.values)
            except Exception:                                  # noqa: BLE001
                pass
        if not out:
            return None, None
        ref = a
        return np.concatenate(out, axis=0), ref
    except Exception:                                          # noqa: BLE001
        return None, None
    finally:
        tmp.unlink(missing_ok=True)


def _band_members(v, ref, lon):
    """(member, lat, lon) -> cos-weighted 15S-15N mean on the 2.5 deg EOF grid."""
    lat = ref["latitude"].values if "latitude" in ref.coords else ref["lat"].values
    lo = ref["longitude"].values if "longitude" in ref.coords else ref["lon"].values
    w = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX), np.cos(np.deg2rad(lat)), 0.0)
    b = (v * w[None, :, None]).sum(1) / w.sum()                 # (member, nlon)
    o = np.argsort(lo % 360)
    lo_s = (lo % 360)[o]
    b = b[:, o]
    lo_w = np.r_[lo_s, 360.0]
    b_w = np.concatenate([b, b[:, :1]], axis=1)
    return np.stack([np.interp(lon, lo_w, row) for row in b_w])


def member_bands(cycle, base, lon, workers=8):
    """Per-member band means for the three RMM channels, cached to disk. forecast.py writes the cache from the GRIBs
    it already downloads, so the fetch below is only a fallback (a cycle built before that, or a partial run)."""
    cache = LIVE / f"geps_rmm_members_{cycle}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in MVARS}, z["L"]
    from concurrent.futures import ThreadPoolExecutor
    days = list(range(1, 36))
    out = {}
    for tag, (var, accum) in MVARS.items():
        want = sorted({24 * d for d in days}) if accum else \
            sorted({24 * d + s for d in days for s in SUBSTEPS})
        with ThreadPoolExecutor(max_workers=workers) as ex:
            got = dict(zip(want, ex.map(lambda L: _grib_members(cycle, var, L), want)))
        cur = {}
        for L, (v, ref) in got.items():
            if v is not None:
                cur[L] = _band_members(v, ref, lon)
        rows = []
        for d in days:
            if accum:
                end = cur.get(24 * d); start = cur.get(24 * (d - 1))
                if end is None:
                    rows.append(None); continue
                rows.append(((end - start) if start is not None else end) / 86400.0)
            else:
                have = [cur[24 * d + s] for s in SUBSTEPS if (24 * d + s) in cur]
                rows.append(np.mean(have, axis=0) if have else None)
        keep = [i for i, r in enumerate(rows) if r is not None]
        out[tag] = np.stack([rows[i] for i in keep])            # (lead, member, nlon)
        print(f"    members {tag}: {out[tag].shape[0]} leads x {out[tag].shape[1]}",
              flush=True)
    L = np.array([days[i] for i in keep])
    np.savez_compressed(cache, L=L, **out)
    return out, L


def _project(folr, f850, f200, eofs):
    """Three-field WH04 projection -> (rmm1, rmm2), unit-variance normalised."""
    e1 = np.concatenate([eofs["eof_u850"].sel(mode=1).values, eofs["eof_u200"].sel(mode=1).values])
    e2 = np.concatenate([eofs["eof_u850"].sel(mode=2).values, eofs["eof_u200"].sel(mode=2).values])
    o1, o2 = eofs["eof_olr"].sel(mode=1).values, eofs["eof_olr"].sel(mode=2).values
    comb = np.concatenate([f850, f200], axis=1)
    return ((folr @ o1 + comb @ e1) / float(eofs["pc_std"].sel(mode=1)),
            (folr @ o2 + comb @ e2) / float(eofs["pc_std"].sel(mode=2)))


ANL = ("https://dd.weather.gc.ca/{d}/WXO-DD/ensemble/geps/grib2/raw/00/000/"
       "CMC_geps-raw_{v}_latlon0p5x0p5_{d}00_P000_allmbrs.grib2")


def geps_analysis_winds(base: pd.Timestamp, days: int, lon):
    """Band-mean U850/U200 from GEPS's OWN lead-0 analysis, one point per day.

    This follows the methodology already established for the AIFS product
    (archive_truth.py): grow the observed track from the forecast model's own
    zero-lag analysis, so there is no reanalysis-to-model handoff jump at the
    start of the forecast. It also solves a latency problem that has no other
    cheap fix — ERA5 (and ARCO) run six days behind and NCEP R1 five months, so
    a reanalysis track would stop days before the GEPS init and leave a visible
    gap on the phase diagram. Datamart keeps roughly 30 days of cycles, which
    sets the length of the track.
    """
    import forecast as fx
    out = {}
    dates = [base - pd.Timedelta(days=k) for k in range(days)]

    def one(d):
        # one small npz per analysis day, carried between runs, so each day's two lead-0 files are read once ever
        cf = ANL_CACHE / f"uv_{d:%Y%m%d}.npz"
        if cf.exists():
            z = np.load(cf)
            if np.allclose(z["lon"], lon):
                return {"u850": z["u850"], "u200": z["u200"]}
        got = {}
        for tag, var in (("u850", "UGRD_ISBL_0850"), ("u200", "UGRD_ISBL_0200")):
            a = fx._grab(d.strftime("%Y%m%d"), var, 0)
            if a is None:
                return None
            lat = "latitude" if "latitude" in a.dims else "lat"
            w = np.cos(np.deg2rad(a[lat])).where(
                (a[lat] >= LAT_MIN) & (a[lat] <= LAT_MAX), 0.0)
            b = a.weighted(w).mean(dim=lat)
            b = b.assign_coords(longitude=b.longitude % 360).sortby("longitude")
            b = xr.concat([b, b.isel(longitude=0).assign_coords(longitude=360.0)],
                          dim="longitude")
            got[tag] = b.interp(longitude=lon).values
        ANL_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cf, lon=lon, **got)
        return got

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=datamart.WORKERS) as ex:
        for d, g in zip(dates, ex.map(one, dates)):
            if g is not None:
                out[d.normalize()] = g
    if not out:
        raise SystemExit("no GEPS analysis cycles retrievable")
    t = pd.DatetimeIndex(sorted(out))
    return (t, np.stack([out[d]["u850"] for d in t]),
            np.stack([out[d]["u200"] for d in t]))


def observed(base: pd.Timestamp, clim, eofs, days: int = 30):
    """Three-field observed RMM track, and the fixed 120-day filter maps.

    Winds: GEPS lead-0 analysis (see above). OLR: olr_channel.py. Filter maps:
    the site's wind_map120.nc plus the new olr_map120.nc, both from ERA5-grade
    analysis — a 120-day mean tolerates a few days of latency, unlike the track."""
    lon = eofs.longitude.values.astype(float)
    t, lm850, lm200 = geps_analysis_winds(base, days, lon)
    doy = xr.DataArray(t.dayofyear, dims="time")
    a850 = lm850 - clim["clim_u850"].sel(dayofyear=doy).values
    a200 = lm200 - clim["clim_u200"].sel(dayofyear=doy).values

    # sanity-check the model analysis against ERA5 where they overlap, rather
    # than assuming the two agree
    ep = REF / "era5_eq_uv_recent.nc"
    if ep.exists():
        e = xr.open_dataset(ep)
        et = pd.DatetimeIndex(e.time.values).normalize()
        both = t.intersection(et)
        if len(both) >= 5:
            i, j = t.get_indexer(both), et.get_indexer(both)
            d8 = lm850[i] - e.u.sel(level=850).values[j]
            d2 = lm200[i] - e.u.sel(level=200).values[j]
            print(f"  GEPS analysis vs ERA5 over {len(both)} d: "
                  f"U850 bias {d8.mean():+.2f} rms {d8.std():.2f} m/s "
                  f"(std_u850 {clim.attrs['std_u850']:.2f}); "
                  f"U200 bias {d2.mean():+.2f} rms {d2.std():.2f}")

    m = xr.open_dataset(REF / "wind_map120.nc")
    to, ao, mo = olr_channel.anomaly_and_map120(clim)
    idx = pd.Index(pd.DatetimeIndex(to).normalize()).get_indexer(t)
    ok = idx >= 0
    aolr = np.full_like(a850, np.nan)
    aolr[ok] = ao[idx[ok]]

    maps = {"u850": m["u850"].values, "u200": m["u200"].values, "olr": mo}
    fo = (aolr - maps["olr"]) / clim.attrs["std_olr"]
    f8 = (a850 - maps["u850"]) / clim.attrs["std_u850"]
    f2 = (a200 - maps["u200"]) / clim.attrs["std_u200"]
    r1, r2 = _project(fo, f8, f2, eofs)
    good = np.isfinite(r1)
    sel = np.where(good)[0][-days:]
    print(f"  observed 3-field: {t[sel][0]:%Y-%m-%d} -> {t[sel][-1]:%Y-%m-%d} "
          f"({len(sel)} d), amp now {np.hypot(r1[sel][-1], r2[sel][-1]):.2f}")
    return t[sel], r1[sel], r2[sel], maps


def _reref(tag: str, lon, valid: pd.DatetimeIndex, clim, cvar: str) -> np.ndarray:
    """M(L=0) - O at each valid date: the shift from the model frame to the observed one.

    Evaluated at the SAME day-of-year as the forecast, so the seasonal cycle is
    consistent on both sides of the difference."""
    cv = refpack.clim(tag).squeeze(drop=True)
    out = []
    for d in valid:
        m = (cv.sel(L=0.5, method="nearest").interp(doy=d.dayofyear)
               .rename({"Y": "latitude", "X": "longitude"}))
        lat = m.latitude
        w = np.cos(np.deg2rad(lat)).where((lat >= LAT_MIN) & (lat <= LAT_MAX), 0.0)
        b = m.weighted(w).mean(dim="latitude")
        b = b.assign_coords(longitude=b.longitude % 360).sortby("longitude")
        b = xr.concat([b, b.isel(longitude=0).assign_coords(longitude=360.0)], dim="longitude")
        out.append(b.interp(longitude=lon).values
                   - clim[cvar].sel(dayofyear=min(d.dayofyear, 366)).values)
    return np.stack(out)


def rmm(cycle: str, base: pd.Timestamp, clim, eofs, maps):
    """Three-field RMM for the GEPS ensemble-mean forecast, days 1-35."""
    lon = eofs.longitude.values.astype(float)
    u850, u200, olr = load("u850", cycle), load("u200", cycle), load("olr", cycle)
    if u850 is None or u200 is None or olr is None:
        return None
    L = u850.L.values.astype(int)
    valid = pd.DatetimeIndex([base + pd.Timedelta(days=int(x) - 1) for x in L])   # day L = the UTC day base + L - 1
    ch = {}
    for tag, a, cvar in (("u850", u850, "clim_u850"), ("u200", u200, "clim_u200"),
                         ("olr", olr, "clim_olr")):
        ch[tag] = (band_mean(a, lon) + _reref(tag, lon, valid, clim, cvar)
                   - maps[tag]) / clim.attrs[f"std_{tag}"]
    r1, r2 = _project(ch["olr"], ch["u850"], ch["u200"], eofs)
    return L, r1, r2


def _model_clim_band(tag, lon, L, init_doy):
    """Band mean of the GEPS model climatology M(L) at the forecast's INIT
    day-of-year — the same reference forecast.py subtracts to form its anomaly.

    The member arrays come straight off the GRIBs as RAW fields, whereas the mean
    track is built from forecast.py's anomalies. Feeding raw fields into the same
    projection silently omits this term, which left the plume in a different frame
    from the mean track — the mean did not sit inside its own members."""
    cv = refpack.clim(tag).squeeze(drop=True)
    out = []
    for x in L:
        m = (cv.sel(L=float(x) - 0.5, method="nearest").interp(doy=init_doy)
               .rename({"Y": "latitude", "X": "longitude"}))
        lat = m.latitude
        w = np.cos(np.deg2rad(lat)).where((lat >= LAT_MIN) & (lat <= LAT_MAX), 0.0)
        b = m.weighted(w).mean(dim="latitude")
        b = b.assign_coords(longitude=b.longitude % 360).sortby("longitude")
        b = xr.concat([b, b.isel(longitude=0).assign_coords(longitude=360.0)],
                      dim="longitude")
        out.append(b.interp(longitude=lon).values)
    return np.stack(out)


def rmm_members(cycle, base, clim, eofs, maps):
    """Per-member three-field RMM, days 1-35, in the same frame as the mean track."""
    lon = eofs.longitude.values.astype(float)
    bands, L = member_bands(cycle, base, lon)
    valid = pd.DatetimeIndex([base + pd.Timedelta(days=int(x) - 1) for x in L])   # day L = the UTC day base + L - 1
    ch = {}
    for tag, ckey in (("u850", "clim_u850"), ("u200", "clim_u200"), ("olr", "clim_olr")):
        rr = _reref(tag, lon, valid, clim, ckey)                # (lead, nlon)
        mc = _model_clim_band(tag, lon, L, base.dayofyear)      # (lead, nlon)
        ch[tag] = ((bands[tag] - mc[:, None, :] + rr[:, None, :] - maps[tag])
                   / clim.attrs[f"std_{tag}"])
    n = ch["u850"].shape[1]
    r1 = np.empty((len(L), n)); r2 = np.empty_like(r1)
    for m in range(n):
        a, b = _project(ch["olr"][:, m, :], ch["u850"][:, m, :], ch["u200"][:, m, :], eofs)
        r1[:, m], r2[:, m] = a, b
    return L, r1, r2


def plot_phase(cycle: str, base: pd.Timestamp):
    clim = xr.open_dataset(REF / "climatology.nc")
    eofs = xr.open_dataset(REF / "eofs.nc")
    ot, or1, or2, maps = observed(base, clim, eofs)
    got = rmm(cycle, base, clim, eofs, maps)
    if got is None:
        print("  phase: missing a channel"); return None
    L, r1, r2 = got

    # Same wheel the AIFS-ENS product draws (src/plot.py), so the two phase
    # diagrams on the site share one convention: phase 1 in the left sector,
    # numbering counterclockwise, West. Hem./Africa left and Indian Ocean bottom.
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    fig.subplots_adjust(left=0.082, right=0.978, top=0.938, bottom=0.145)
    plot.draw_phase_wheel(ax)

    # observed track: faint connector with points by month, as in plot.py
    ax.plot(or1, or2, color="0.55", lw=1.0, alpha=0.75, zorder=5)
    months = ot.to_period("M")
    mcmap = plt.get_cmap("tab10")
    for k, mo in enumerate(dict.fromkeys(months)):
        sel = months == mo
        ax.scatter(or1[sel], or2[sel], s=22, zorder=6, color=mcmap(k % 10),
                   edgecolor="0.3", linewidth=0.3, label=f"Obs {mo.strftime('%b %Y')}")
    ax.scatter(or1[-1], or2[-1], marker="*", s=210, zorder=8, color="#111111",
               label=f"analysis {ot[-1]:%b %d}")

    # ensemble plume: every member, faint, under the mean
    try:
        Lm, m1, m2 = rmm_members(cycle, base, clim, eofs, maps)
        for k in range(m1.shape[1]):
            ax.plot(np.r_[or1[-1], m1[:, k]], np.r_[or2[-1], m2[:, k]],
                    color="#4a7bb5", alpha=0.42, lw=0.9, zorder=3)
        amp = np.hypot(m1, m2)
        print(f"  plume: {m1.shape[1]} members, day35 amplitude "
              f"{np.percentile(amp[-1], 10):.2f}-{np.percentile(amp[-1], 90):.2f} "
              f"(mean track {np.hypot(r1[-1], r2[-1]):.2f})")
    except Exception as e:                                      # noqa: BLE001
        print(f"  plume unavailable: {str(e)[:70]}")

    # forecast track coloured by lead, anchored on the analysis point
    cmap, norm = plt.get_cmap("plasma_r"), mcolors.Normalize(0, len(L) - 1)
    x = np.r_[or1[-1], r1]
    y = np.r_[or2[-1], r2]
    for i in range(len(x) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], color=cmap(norm(max(i - 1, 0))), lw=2.4, zorder=7)
    for k in range(4, len(L), 5):
        ax.scatter(r1[k], r2[k], s=42, color=cmap(norm(k)), edgecolor="0.25",
                   linewidth=0.4, zorder=8)
    # a correctly damped ensemble-mean track is compact, so the labels stack on
    # each other and on the line: fan them out and back them with white
    for k, (dx, dy) in zip((9, 19, 34), ((11, 9), (11, -13), (-13, 11))):
        if k < len(L):
            ax.annotate(f"day {L[k]}", (r1[k], r2[k]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=9, fontweight="bold", zorder=10,
                        color=cmap(norm(k)),
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.82))
    ax.scatter(r1[-1], r2[-1], marker="s", s=70, color=cmap(norm(len(L) - 1)),
               edgecolor="0.25", linewidth=0.5, zorder=9,
               label=f"GEPS day {L[-1]}")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
    ax.set_title(f"GEPS Ensemble RMM  —  Init: {base:%Y-%m-%d} 00Z",
                 fontsize=13, fontweight="bold")

    # explicit line breaks, not wrap=True: matplotlib's wrapper measures against
    # the FIGURE width and happily ran this caption off both edges
    fig.text(0.5, 0.022, "Three-field RMM (OLR + U850 + U200) · 21 members",
             ha="center",
             va="bottom", fontsize=9, color="#8a8680")
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / "geps_mjo_phase.webp"
    fig.savefig(p, dpi=110, facecolor="white", pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    amp = np.hypot(r1, r2)
    print(f"  {p.name}  amp day1 {amp[0]:.2f} -> day35 {amp[-1]:.2f}")
    return p.name


def _hov_panel(ax, lon, L, F, cmap, lim, base, dates_on_right):
    m = ax.pcolormesh(lon, L, F, cmap=cmap, vmin=-lim, vmax=lim, shading="auto")
    ax.invert_yaxis()
    ax.set_xlim(0, 357.5)
    ax.set_xticks([0, 60, 120, 180, 240, 300])
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W"], fontsize=9)
    for x in (100, 150):                       # Indian | Maritime | Pacific
        ax.axvline(x, color="#00000033", lw=0.8, ls=(0, (4, 3)))
    for x, t in ((62, "Indian Ocean"), (125, "Maritime"), (215, "Pacific"), (310, "Atlantic")):
        ax.text(x, 1.4, t, ha="center", va="top", fontsize=8.5, color="#4a453f",
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78))
    ax.tick_params(labelsize=9)
    if dates_on_right:
        ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim())
        ticks = [1, 7, 14, 21, 28, 35]
        ax2.set_yticks(ticks)
        ax2.set_yticklabels([f"{base + pd.Timedelta(days=t - 1):%b %d}" for t in ticks], fontsize=8.5)   # lead day t = UTC day base + t - 1
    return m


def plot_hovmoller(cycle: str, base: pd.Timestamp):
    """Two panels: convection (OLR) and the 850 hPa zonal wind that goes with it.

    These were one panel with the wind overlaid as contours. Separating them and
    filling the wind reads far better: the MJO's signature is the PHASE RELATION
    between the two — westerly anomalies trailing the convective envelope — and
    two filled panels side by side show that directly, where contours over a
    filled field mostly just clutter it. Wind is in knots."""
    olr, u850 = load("olr", cycle), load("u850", cycle)
    if olr is None or u850 is None:
        print("  hovmoller: missing olr or u850"); return None
    lon = np.arange(0, 360, 2.5)
    # Same offset removal the OLR maps use: the live operational GEPS radiates
    # ~7.6 W/m2 differently from the GEPS8 reforecast at EVERY lead, so without
    # this the whole panel is tinted green. (It is irrelevant to the RMM — a
    # uniform offset moves the projection by <0.05 — but it dominates a plot.)
    olr = olr - xr.DataArray(olr.global_mean.values, dims="L", coords={"L": olr.L})
    O = band_mean(olr, lon)
    U = band_mean(u850, lon) * 1.94384                       # m/s -> knots
    L = olr.L.values.astype(int)

    fig = plt.figure(figsize=(11.4, 9.2))
    left, right, bot, top = 0.062, 0.918, 0.108, 0.885
    gap = 0.055
    w = (right - left - gap) / 2
    axO = fig.add_axes([left, bot, w, top - bot])
    axU = fig.add_axes([left + w + gap, bot, w, top - bot])

    limO = float(np.nanpercentile(np.abs(O), 98))
    limU = float(np.nanpercentile(np.abs(U), 98))
    mO = _hov_panel(axO, lon, L, O, "BrBG_r", limO, base, False)
    mU = _hov_panel(axU, lon, L, U, "RdBu_r", limU, base, True)
    axO.set_ylabel("forecast lead (days)", fontsize=10.5)
    axU.set_yticklabels([])
    axO.set_title("Convection — OLR anomaly", fontsize=11.5, fontweight="bold", pad=6)
    axU.set_title("850 hPa zonal wind anomaly", fontsize=11.5, fontweight="bold", pad=6)

    for ax, m, lab in ((axO, mO, "OLR anomaly (W m⁻²)"),
                       (axU, mU, "850 hPa u anomaly (knots)")):
        bb = ax.get_position()
        cax = fig.add_axes([bb.x0 + 0.10 * bb.width, 0.042, 0.80 * bb.width, 0.015])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(lab, fontsize=9, labelpad=2)
        cb.ax.tick_params(labelsize=8, pad=1.5)

    fig.suptitle("Tropical convection and low-level wind, 15°S–15°N\n"
                 f"GEPS extended · init {base:%Y-%m-%d}",
                 fontsize=13, fontweight="bold", y=0.972, va="top")
    p = FIGS / "geps_mjo_hovmoller.webp"
    fig.savefig(p, dpi=110, facecolor="white", pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    print(f"  {p.name}  OLR ±{limO:.0f} W/m², u850 ±{limU:.0f} kt")
    return p.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    a = ap.parse_args()
    base = pd.Timestamp(a.cycle)
    FIGS.mkdir(parents=True, exist_ok=True)
    plot_phase(a.cycle, base)
    plot_hovmoller(a.cycle, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
