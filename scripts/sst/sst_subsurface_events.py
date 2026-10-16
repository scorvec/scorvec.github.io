#!/usr/bin/env python3
"""Subsurface El Niño analog comparison (TAO/TRITON equatorial Pacific).

Compares the current event's equatorial subsurface temperature against the
1997-98, 2015-16 and 2023-24 El Niños, reusing the TAO DISDEL fetch
(tao_subsurface) and the 1991-2020 harmonic climatology (sst_subsurface).

Products (assets/sst/):
  subsurface_events_xsec.webp  matching-phase depth×lon temperature-anomaly
                               cross-sections (2×2: current vs analogs)
  subsurface_events_hc.webp    equatorial 0–300 m temperature-anomaly index
                               through each event's development year + next;
                               1982-83 (before TAO) from two monthly ocean
                               reanalyses, ORAS5 and GODAS (reanalysis_hc.py)

TAO mooring coverage varies by year (1997 and 2023 are gappier), so some
longitudes / periods are missing — shown as gaps.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import tao_subsurface as tao
import sst_subsurface as ss
import tao_hc as H

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data"
EVDIR = DATA / "subsurface_events"

ANALOGS = [1997, 2015, 2023]
ANALOG_COLORS = {1997: "#1f77b4", 2015: "#2ca02c", 2023: "#9467bd"}
XSEC_WIN = 14  # days averaged for the matching-phase cross-section
HC_SMOOTH = 15  # days, heat-content running mean


def _anomaly(grid: xr.DataArray) -> xr.DataArray:
    coeffs = ss.load_or_build_coeffs(grid.longitude.values)
    doy = ss.harmonic_day(grid.time.values)
    return xr.DataArray(grid.values - ss.eval_climatology(coeffs, doy), dims=grid.dims, coords=grid.coords)


def event_year(latest: pd.Timestamp) -> int:
    """Development year (Yr0) of the CURRENT event. Through June of Yr1 the chart stays on the event
    that started the year before (the old code switched to the new calendar year on 1 January and
    would have restarted the red line at "Jan Yr0" with a few days of data)."""
    return latest.year if latest.month >= 7 else latest.year - 1


def fetch_event(year: int, current: bool) -> xr.Dataset:
    """DISDEL delivery Jan (Yr0-1) .. Dec Yr1 (or today), parsed WITH quality/source codes.
    The year before Yr0 is there so carry-forward and the reanalysis offset have history."""

    EVDIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    if current:
        p = EVDIR / f"tao_{year - 1}_cur.ascii"
        tao.deliver(datetime(year - 1, 1, 1), now, p)  # always fresh: PMEL posts and backfills late
    else:
        p = EVDIR / f"tao_{year - 1}_{year + 1}.ascii"
        if not (p.exists() and p.stat().st_size > 0):
            tao.deliver(datetime(year - 1, 1, 1), datetime(year + 1, 12, 31), p)
    return H.parse_qc(p)


def event_anom(ds: xr.Dataset) -> xr.DataArray:
    """Regridded temperature ANOMALY (time, depth, lon) from the screened temperatures."""
    scr = xr.Dataset({"temp": H.screened_temp(ds)})
    return _anomaly(ss.to_depth_grid(scr))


def _same_day(yr: int, match: pd.Timestamp) -> pd.Timestamp:
    """The same calendar date in year yr; 29 Feb becomes 28 Feb only in a common year."""
    leap = yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)
    return pd.Timestamp(yr, match.month, min(match.day, 29 if leap else 28) if match.month == 2 else match.day)


# ── 1. matching-phase cross-sections ──────────────────────────────────────────
def xsec(events: dict, cur_year: int, match: pd.Timestamp, out: Path, detr: bool = False, yoff: int = 0):
    years = [cur_year, *ANALOGS]
    labels = [f"{cur_year} (current)"] + [f"{y}–{str(y + 1)[2:]}" for y in ANALOGS]
    # Set the shared anomaly scale from THESE panels' data.
    #
    # ss.ANOM_LEVELS is a module default until sst_subsurface.main() replaces it
    # at runtime - and that never happens here, because this is a separate
    # entry point. So this figure quietly kept the fixed +/-12 while the other
    # product had moved to a data-driven scale, and its 2026 panel was clipping
    # exactly the way the animation used to. One scale across all four panels,
    # chosen from all four, or the comparison between events is meaningless.
    allA = []
    for yr in years:
        a = events[yr]
        end = _same_day(yr + yoff, match)
        w = a.sel(time=slice(end - pd.Timedelta(days=XSEC_WIN - 1), end)).mean("time")
        allA.append(np.asarray(w.values, float))
    ss.set_anom_scale(np.concatenate([x.ravel() for x in allA]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey=True)
    im = None
    for ax, yr, lab in zip(axes.ravel(), years, labels):
        a = events[yr]
        end = _same_day(yr + yoff, match)
        win = a.sel(time=slice(end - pd.Timedelta(days=XSEC_WIN - 1), end)).mean("time")
        lons = win.longitude.values
        if len(lons) >= 2:
            Ag = ss.interp_lon(win.values, lons)
            im = ax.contourf(
                ss.LON_GRID,
                ss.DEPTH_GRID,
                Ag,
                levels=ss.ANOM_LEVELS,
                cmap="RdBu_r",
                extend="both",
                norm=mcolors.TwoSlopeNorm(0, -ss.ANOM_LIM, ss.ANOM_LIM),
            )
            ax.contour(ss.LON_GRID, ss.DEPTH_GRID, Ag, levels=[0], colors="k", linewidths=0.6)
            # The strongest cores had no line on them at all - contours stopped
            # at zero here while the field reaches past +10.
            lv = [v for v in ss.ANOM_CONTOURS if np.nanmin(Ag) <= v <= np.nanmax(Ag)]
            if lv:
                cc = ax.contour(ss.LON_GRID, ss.DEPTH_GRID, Ag, levels=lv, colors="k", linewidths=0.7)
                ax.clabel(cc, fmt="%+d", fontsize=6)
        ax.scatter(lons, np.full(len(lons), 6), marker="v", s=18, color="k", zorder=5)
        ax.set_ylim(300, 0)
        ax.set_title(lab, fontsize=10)
        ax.set_xticks([165, 180, 200, 220, 240, 260])
        ax.set_xticklabels(["165°E", "180°", "160°W", "140°W", "120°W", "100°W"], fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Depth (m)")
    extra = "\ndetrended with data from 1991–2020" if detr else ""
    fig.suptitle(
        f"Equatorial Pacific subsurface temperature anomaly — week of ~{match:%b %d} of each event"
        f"{extra}\n(triangles = moorings reporting; TAO coverage varies by year)",
        fontsize=11.5,
        fontweight="bold",
    )
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, extend="both")
    cb.set_label("Temperature anomaly (°C)", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (matching ~{match:%b %d})")


# ── 2. heat-content (0–300 m T anomaly) evolution overlay ─────────────────────
# The index itself lives in tao_hc.py (screened profiles, fixed mooring set, carry-forward
# <= tao_hc.PERSIST_DAYS counted from the last REAL report, NaN below tao_hc.MIN_MOORINGS).
# Validation against NOAA CPC's GODAS index and GLORYS12 is tao_hc_audit.py (tao-hc-audit.yml);
# its numbers are frozen in reference/tao_hc_validation.json, which also switches the
# reanalysis fill on or off.
REF = HERE / "reference"
VALIDATION = REF / "tao_hc_validation.json"
FILL_CSV = REF / "glorys_eq_moorings_anom.csv.gz"  # GLORYS12 0-300 m anomaly at the 8 moorings
FILL_COEFFS = REF / "glorys_eq_clim_coeffs.npz"  # its 1993-2020 harmonic climatology
CPC_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ocean/index/heat_content_index.txt"
ANFC = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
CAT_COLORS = ["#ffffff", "#c8c8c8", "#e3a33b", "#4a4a4a", "#c9b3a6"]
CAT_LABELS = [
    "all 8 TAO moorings reporting",
    "6–7 reporting; silent ones from GLORYS12 (or carried ≤10 d)",
    "fewer than 6 reporting: rests on GLORYS12 (dashed)",
    "no estimate",
    "before TAO: monthly reanalyses only (ORAS5, GODAS)",
]

# Pre-TAO events from two MONTHLY ocean reanalyses, the index rebuilt on the same 8 points (reanalysis_hc.py,
# run offline; the small monthly table and its validation are committed, nothing is fetched here).
RA_CSV = REF / "reanalysis_hc_monthly.csv"
RA_VALIDATION = REF / "reanalysis_hc_validation.json"
RA_EVENTS = [1982]
RA_COLOR = "#8c564b"


def _meta() -> dict:
    try:
        return json.loads(VALIDATION.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _anfc_recent(start: pd.Timestamp, meta: dict) -> pd.DataFrame | None:
    """GLORYS-family ANALYSIS (anfc) anomaly at the 8 moorings from `start` to yesterday, put on the
    reanalysis' footing with the per-mooring anfc-vs-reanalysis offset measured in the audit.
    Kept in data/cmems_hc/anfc_store.csv (actions/cache in sst-events.yml); each run re-pulls only
    the last 10 days (the analysis is revised as newer runs arrive) and anything newer."""
    store = DATA / "cmems_hc" / "anfc_store.csv"
    have = None
    if store.exists():
        try:
            have = pd.read_csv(store, index_col=0, parse_dates=True)
            have.columns = [float(c) for c in have.columns]
        except Exception:  # noqa: BLE001
            have = None
    if not os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") or not FILL_COEFFS.exists():
        print("reanalysis fill: no Copernicus Marine credentials or coefficients; reanalysis record only")
        return have
    import copernicusmarine as cm

    z = np.load(FILL_COEFFS)
    coeffs = {float(l): c for l, c in zip(z["lons"], z["coeffs"])}
    off = meta.get("fill", {}).get("anfc_offset", {})
    yday = pd.Timestamp(datetime.now(UTC).date()) - pd.Timedelta(days=1)
    t0 = start if have is None or have.empty else max(start, have.index.max() - pd.Timedelta(days=10))
    if t0 > yday:
        return have
    cache = DATA / "cmems_hc"
    cache.mkdir(parents=True, exist_ok=True)
    cols = {}
    for lo in H.LONS:
        glon = lo if lo < 180 else lo - 360
        fn = f"anfc_{int(lo)}.nc"
        try:
            cm.subset(
                dataset_id=ANFC,
                variables=["thetao"],
                minimum_longitude=glon,
                maximum_longitude=glon,
                minimum_latitude=-0.25,
                maximum_latitude=0.25,
                minimum_depth=0,
                maximum_depth=330,
                start_datetime=t0.strftime("%Y-%m-%d"),
                end_datetime=yday.strftime("%Y-%m-%d"),
                coordinates_selection_method="nearest",
                output_filename=fn,
                output_directory=str(cache),
                overwrite=True,
                disable_progress_bar=True,
            )
            d = xr.open_dataset(cache / fn)["thetao"].load()
        except Exception as e:  # noqa: BLE001
            print(f"::warning::reanalysis fill: anfc {H.LON_LABEL[lo]} failed ({e}); that mooring has no new fill")
            continue
        d = d.mean([c for c in ("latitude", "longitude") if c in d.dims]).transpose("time", "depth")
        dep = d.depth.values.astype(float)
        V = d.values
        t = pd.to_datetime(d.time.values).normalize()
        w = 2 * np.pi * ss.harmonic_day(t) / 365.25  # as tao_hc_audit.harmonic_fit
        X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
        clim = X @ coeffs[lo]
        a = np.full(len(t), np.nan)
        for i in range(len(t)):
            m = np.isfinite(V[i])
            if m.sum() >= 3 and dep[m].max() >= 300:
                a[i] = (np.interp(ss.DEPTH_GRID, dep[m], V[i][m]) - clim[i]).mean()
        cols[lo] = pd.Series(a, index=t) + float(off.get(H.LON_LABEL[lo], 0.0))
    new = pd.DataFrame(cols).reindex(columns=H.LONS) if cols else None
    if new is None:
        return have
    out = new if have is None else new.combine_first(have)
    out = out[out.index <= yday].sort_index()
    out.to_csv(store, float_format="%.4f")
    print(
        f"reanalysis fill: anfc {t0:%Y-%m-%d} -> {yday:%Y-%m-%d} fetched; store {out.index.min():%Y-%m-%d} -> {out.index.max():%Y-%m-%d}"
    )
    return out


def reanalysis_fill(meta: dict) -> pd.DataFrame | None:
    """Daily GLORYS12 0-300 m anomaly at the 8 moorings (reanalysis record + recent analysis), or
    None when the audit did not validate the fill."""
    if not meta.get("fill", {}).get("use") or not FILL_CSV.exists():
        return None
    g = pd.read_csv(FILL_CSV, index_col=0, parse_dates=True)
    g.columns = [float(c) for c in g.columns]
    rec = _anfc_recent(g.index.max() - pd.Timedelta(days=5), meta)
    if rec is not None:
        g = pd.concat([g, rec[rec.index > g.index.max()]])
    return g.reindex(columns=H.LONS)


def hc_frame(ds: xr.Dataset, g: pd.DataFrame | None, detr: bool) -> pd.DataFrame:
    df, _ = H.mooring_anomaly(ds, strict=True, detrend=detr)
    ix = H.index(df, fill_df=H.fill_from(df, g))
    ix["best"] = ix["hc"]
    ix["smooth"] = H.smooth(ix["hc"], HC_SMOOTH)
    return ix


def _cpc_latest(cur: pd.DataFrame, meta: dict) -> str:
    """One line: this event's newest complete month on our index vs CPC's GODAS 180-100W value
    mapped onto our scale with the frozen audit regression."""
    fit = meta.get("validation", {}).get("cpc_fit")
    if not fit:
        return ""
    try:
        import urllib.request

        req = urllib.request.Request(CPC_URL, headers={"User-Agent": "scorvec.com data pipeline"})
        txt = urllib.request.urlopen(req, timeout=60).read().decode("latin-1")
        rows = {}
        for ln in txt.splitlines():
            p = ln.split()
            if len(p) == 5 and p[0].isdigit():
                rows[pd.Timestamp(int(p[0]), int(p[1]), 1)] = float(p[3 if fit.get("column") == "160E-80W" else 4])
        b = cur["best"].where(cur["cat"] <= 1)
        g = b.groupby(b.index.to_period("M"))
        mon = g.mean().where(g.count() >= 20)
        mon.index = mon.index.to_timestamp()
        both = [m for m in mon.dropna().index if m in rows]
        if not both:
            return ""
        m = both[-1]
        est = fit["slope"] * rows[m] + fit["intercept"]
        dom = "160°E–80°W" if fit.get("column") == "160E-80W" else "180°–100°W"
        return (
            f"Latest month, {m:%b %Y}: this index {mon[m]:+.2f} °C; CPC GODAS {dom} {rows[m]:+.2f} °C, "
            f"{est:+.2f} °C on this index's scale (difference {mon[m] - est:+.2f}, fit RMSE "
            f"{fit.get('rmse_after_fit', float('nan')):.2f})."
        )
    except Exception as e:  # noqa: BLE001
        print(f"CPC check skipped: {e}")
        return ""


def reanalysis_events() -> tuple[pd.DataFrame | None, dict]:
    """Monthly ORAS5 / GODAS versions of the index (deg C, own 1991-2020 climatologies) + validation."""
    try:
        ra = pd.read_csv(RA_CSV, index_col=0, parse_dates=True)[["oras5", "godas"]]
        rv = json.loads(RA_VALIDATION.read_text()).get("validation", {})
        return ra, rv
    except Exception as e:  # noqa: BLE001
        print(f"::warning::reanalysis events unavailable ({e}); chart drawn without 1982-83")
        return None, {}


def _ra_detrend(ra: pd.DataFrame) -> pd.DataFrame:
    """The detrended chart's own trend (TAO 1991-2020 slope, per mooring and depth, sst_subsurface.detrend)
    applied at index level: the 8-point, 0-300 m mean slope x (decimal year - pivot). Before 1991 this
    extends the 1991-2020 trend backwards."""
    sl = ss.load_or_build_trend().reindex(longitude=H.LONS, depth=ss.DEPTH_GRID).fillna(0.0)
    k = float(sl.mean(["depth", "longitude"]))
    mid = ra.index + pd.Timedelta(days=14)
    return ra.sub(k * (ss.decimal_year(mid) - ss.TREND_REF), axis=0)


def _ra_note(rv: dict, years: list, detr: bool) -> str:
    """Footnote for the reanalysis lines: what they are, how certain, and how they score against TAO."""
    ev = ", ".join(f"{y}–{str(y + 1)[2:]}" for y in years)
    txt = (
        f"{ev} (brown, before the TAO array): the same index rebuilt from two MONTHLY ocean reanalyses, ECMWF ORAS5 "
        "and NOAA NCEP GODAS – same 8 equatorial points (±0.5°), 0–300 m mean potential temperature, anomaly vs "
        "each reanalysis' own 1991–2020 monthly climatology, plotted mid-month; shading = the spread between the two. "
        "Before TAO (1985–94) the reanalyses rest mostly on XBTs; 1982/83 is less certain."
    )
    o, g = rv.get("oras5_vs_tao6_1993on"), rv.get("godas_vs_tao6_1993on")
    if o and g:
        txt += (
            f" Check against this TAO index, monthly 1993–2026 (≥6 moorings live): ORAS5 r {o['r']:.2f}, "
            f"RMS {o['rms']:.2f} °C (n {o['n']}); GODAS r {g['r']:.2f}, RMS {g['rms']:.2f} °C."
        )
    sp, mx = rv.get("oras5_vs_godas_1982_83"), rv.get("oras5_minus_godas_1982_83_maxabs")
    if sp and mx and 1982 in years:
        m = pd.Timestamp(mx["month"] + "-01")
        txt += (
            f" In 1982–83 the two differ by up to {abs(mx['value']):.1f} °C ({m:%b %Y}; RMS {sp['rms']:.2f} °C, "
            f"against {rv.get('oras5_vs_godas_1993on', {}).get('rms', float('nan')):.2f} °C since 1993)."
        )
    e = {k: rv.get(f"events_{k}", {}) for k in ("oras5", "godas", "tao6")}
    try:
        if detr:
            raise KeyError("peaks are quoted on the raw chart only")
        txt += (
            f" Same basis, peak monthly value 1982–83: ORAS5 {e['oras5']['1982']['max']:+.2f}, GODAS "
            f"{e['godas']['1982']['max']:+.2f} °C; 1997–98: ORAS5 {e['oras5']['1997']['max']:+.2f}, GODAS "
            f"{e['godas']['1997']['max']:+.2f}, TAO {e['tao6']['1997']['max']:+.2f} °C."
        )
    except (KeyError, TypeError):
        pass
    if detr:
        txt += " Detrended with the TAO 1991–2020 trend (8-point, 0–300 m mean), extended back before 1991."
    txt += (
        " ORAS5: generated using Copernicus Climate Change Service information "
        f"{datetime.utcnow():%Y}; GODAS: NOAA NCEP via NOAA PSL."
    )
    return txt


def heat_content(
    frames: dict,
    cur_year: int,
    out: Path,
    meta: dict,
    detr: bool = False,
    ra: pd.DataFrame | None = None,
    rv: dict | None = None,
):
    fig = plt.figure(figsize=(9.5, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.05], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axc = fig.add_subplot(gs[1], sharex=ax)

    def devdays(idx, yr):
        # x on a two-year axis of common years (1 Jan Yr0 = 0, 1 Jan Yr1 = 365), so a calendar date sits at the same
        # x in every event whether or not its years are leap; 29 Feb falls midway between 28 Feb and 1 March
        return 365 * (idx.year.values - yr) + ss.harmonic_day(idx) - 1

    def window(ix, yr):
        return ix[(ix.index >= pd.Timestamp(yr, 1, 1)) & (ix.index <= pd.Timestamp(yr + 1, 12, 31))]

    order = [cur_year, *ANALOGS]
    for yr in order:
        ix = window(frames[yr], yr)
        col = "#d62728" if yr == cur_year else ANALOG_COLORS[yr]
        lw = 3.0 if yr == cur_year else 1.8
        lab = f"{yr}–{str(yr + 1)[2:]}" + (" (current)" if yr == cur_year else "")
        x = devdays(ix.index, yr)
        solid = ix["smooth"].where(ix["cat"] <= 1)
        dashed = ix["smooth"].where(ix["cat"] == 2)
        ax.plot(x, solid.values, color=col, lw=lw, label=lab, zorder=5 if yr == cur_year else 3)
        if dashed.notna().any():
            # bridge one day either side so the dashed piece meets the solid line
            d2 = ix["smooth"].where((ix["cat"] == 2) | (ix["cat"].shift(1) == 2) | (ix["cat"].shift(-1) == 2))
            ax.plot(x, d2.values, color=col, lw=lw * 0.6, ls=(0, (3, 2)), zorder=4 if yr == cur_year else 2)
        if yr == cur_year:
            pi = ix.index[(ix["n_carried"] > 0).values & solid.notna().values]
            if len(pi):
                ax.plot(
                    devdays(pi, yr),
                    ix.loc[pi, "smooth"].values,
                    "o",
                    ms=4.5,
                    mfc="white",
                    mec="#d62728",
                    mew=1.4,
                    zorder=6,
                    label=f"{yr}: a silent mooring carried (≤{H.PERSIST_DAYS} d)",
                )
    ra_years = [y for y in RA_EVENTS if ra is not None and not ra.loc[f"{y}-01-01" : f"{y + 1}-12-01"].dropna().empty]
    for yr in ra_years:
        w = ra.loc[f"{yr}-01-01" : f"{yr + 1}-12-01"]
        x = devdays(w.index + pd.Timedelta(days=14), yr)  # mid-month
        lab = f"{yr}–{str(yr + 1)[2:]}"
        ax.fill_between(x, w.min(axis=1).values, w.max(axis=1).values, color=RA_COLOR, alpha=0.18, lw=0, zorder=1)
        ax.plot(
            x,
            w["oras5"].values,
            color=RA_COLOR,
            lw=1.5,
            marker="o",
            ms=3.2,
            zorder=3,
            label=f"{lab} ECMWF ORAS5 (monthly reanalysis)",
        )
        ax.plot(
            x,
            w["godas"].values,
            color=RA_COLOR,
            lw=1.2,
            ls=(0, (1.5, 1.5)),
            marker="s",
            ms=2.8,
            mfc="white",
            zorder=3,
            label=f"{lab} NOAA GODAS (monthly reanalysis)",
        )
    if any((window(frames[y], y)["cat"] == 2).any() for y in order):
        ax.plot([], [], color="0.3", lw=1.2, ls=(0, (3, 2)), label="dashed: TAO + GLORYS12 fill")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlim(0, 730)
    ax.set_ylabel("Equatorial 0–300 m temperature anomaly (°C)")
    if detr:
        title = "Subsurface heat content (eq. Pacific 0–300 m T anomaly)\ndetrended with data from 1991–2020"
        fs = 10.5
    else:
        title = "Subsurface heat content (eq. Pacific 0–300 m T anomaly): current vs. " + ", ".join(
            str(y) for y in sorted(ra_years + ANALOGS)
        )
        fs = 11.5
    ax.set_title(title, fontsize=fs, fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.setp(ax.get_xticklabels(), visible=False)

    # coverage strip: one row per event, coloured by what the day's value rests on
    cmap = mcolors.ListedColormap(CAT_COLORS)
    rows = order + ra_years
    M = np.full((len(rows), 731), np.nan)
    for r, yr in enumerate(order):
        ix = window(frames[yr], yr)
        d = devdays(ix.index, yr)
        ok = (d >= 0) & (d <= 730)
        M[r, np.floor(d[ok]).astype(int)] = ix["cat"].values[ok]
    for r, _yr in enumerate(ra_years, start=len(order)):
        M[r, :] = 4
    order = rows
    axc.imshow(
        np.ma.masked_invalid(M),
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(CAT_COLORS) - 0.5,
        interpolation="nearest",
        extent=(-0.5, 730.5, len(order) - 0.5, -0.5),
    )
    axc.set_yticks(range(len(order)))
    axc.set_yticklabels([f"{y}" for y in order], fontsize=7.5)
    for r in range(len(order) + 1):
        axc.axhline(r - 0.5, color="0.75", lw=0.5)
    ticks = [pd.Timestamp(2001, m, 1).dayofyear - 1 for m in (1, 4, 7, 10)]
    ticks = ticks + [t + 365 for t in ticks]
    axc.set_xticks(ticks)
    axc.set_xticklabels(
        [f"{m}\nYr0" for m in ("Jan", "Apr", "Jul", "Oct")] + [f"{m}\nYr1" for m in ("Jan", "Apr", "Jul", "Oct")],
        fontsize=8,
    )
    axc.set_xlim(0, 730)
    from matplotlib.patches import Patch

    used = sorted({int(v) for v in np.unique(M[np.isfinite(M)])})
    axc.legend(
        handles=[Patch(facecolor=CAT_COLORS[k], edgecolor="0.5", label=CAT_LABELS[k]) for k in used],
        fontsize=7,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        frameon=False,
    )
    axc.set_ylabel("coverage", fontsize=7.5)

    v = meta.get("validation", {})
    lines = []
    if ra_years:
        lines.append(_ra_note(rv or {}, ra_years, detr))
    if v.get("summary_line"):
        lines.append(v["summary_line"])
    cl = "" if detr else _cpc_latest(frames[cur_year], meta)  # the check is on the raw index
    if cl:
        lines.append(cl)
    if lines:
        import textwrap

        txt = "\n".join(textwrap.fill(l, 150) for l in lines)
        fig.text(0.01, -0.035, txt, fontsize=7.6, color="#4a4744", ha="left", va="top")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    cur = frames[cur_year]
    last = cur["smooth"].last_valid_index()
    rl = cur["best"].last_valid_index()
    print(
        f"wrote {out} (current: last day {rl:%Y-%m-%d} raw {cur.at[rl, 'best']:+.2f} °C "
        f"[{int(cur.at[rl, 'n_live'])} live, {int(cur.at[rl, 'n_carried'])} carried, cat {int(cur.at[rl, 'cat'])}], "
        f"15-d {cur.at[last, 'smooth']:+.2f} °C on {last:%Y-%m-%d})"
    )
    return cur


def write_json(frames: dict, cur_year: int, meta: dict) -> None:
    """Small companion JSON (latest value, coverage of the last 60 days) for the page and for checks."""
    cur = frames[cur_year]
    tail = cur[cur.index >= cur.index.max() - pd.Timedelta(days=59)]
    rl = cur["best"].last_valid_index()
    out = {
        "generated": datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "event_year": cur_year,
        "last_day": rl.strftime("%Y-%m-%d"),
        "value_raw": round(float(cur.at[rl, "best"]), 3),
        "value_15d": round(float(cur["smooth"].dropna().iloc[-1]), 3),
        "rule": {
            "persist_days": H.PERSIST_DAYS,
            "min_moorings": H.MIN_MOORINGS,
            "moorings": [H.LON_LABEL[l] for l in H.LONS],
            "quality_codes": list(H.QSET),
            "profile": f"screened sensors spanning <= {H.PROFILE_TOP:.0f} m to >= {H.PROFILE_BOTTOM:.0f} m, "
            f"gaps <= {H.PROFILE_MAX_GAP:.0f} m",
        },
        "validation": meta.get("validation", {}),
        "fill_used": bool(meta.get("fill", {}).get("use")),
        "recent": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "n_live": int(r["n_live"]),
                "n_fill": int(r["n_fill"]),
                "n_carried": int(r["n_carried"]),
                "cat": int(r["cat"]),
                "raw": None if pd.isna(r["best"]) else round(float(r["best"]), 3),
            }
            for d, r in tail.iterrows()
        ],
    }
    p = ASSETS / "data" / "subsurface_hc.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")


def main() -> int:

    meta = _meta()
    now = pd.Timestamp(datetime.now(UTC).date())
    cur_year = event_year(now)
    dss = {cur_year: fetch_event(cur_year, current=True)}
    for y in ANALOGS:
        dss[y] = fetch_event(y, current=False)
    match = pd.to_datetime(dss[cur_year].time.values[-1])
    events = {y: event_anom(d) for y, d in dss.items()}
    # the cross-section sits at the latest DATA day's position in the event (Yr0 or Yr1)
    yoff = int(match.year - cur_year)
    xsec(events, cur_year, match, ASSETS / "subsurface_events_xsec.webp", yoff=yoff)
    g = reanalysis_fill(meta)
    frames = {y: hc_frame(d, g, False) for y, d in dss.items()}
    ra, rv = reanalysis_events()
    heat_content(frames, cur_year, ASSETS / "subsurface_events_hc.webp", meta, ra=ra, rv=rv)
    write_json(frames, cur_year, meta)

    # de-trended companions: remove the 1991–2020 secular trend so the analog
    # comparison isolates the ENSO signal from the background ocean warming/cooling.
    events_dt = {y: ss.detrend(a) for y, a in events.items()}
    xsec(events_dt, cur_year, match, ASSETS / "subsurface_events_xsec_detrended.webp", detr=True, yoff=yoff)
    frames_dt = {y: hc_frame(d, g, True) for y, d in dss.items()}
    heat_content(
        frames_dt,
        cur_year,
        ASSETS / "subsurface_events_hc_detrended.webp",
        meta,
        detr=True,
        ra=None if ra is None else _ra_detrend(ra),
        rv=rv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
