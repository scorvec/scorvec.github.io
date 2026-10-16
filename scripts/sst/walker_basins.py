#!/usr/bin/env python3
"""Walker circulation by basin: what the Indian and Atlantic Oceans are doing to the tropical
zonal overturning right now, with ENSO taken out — two independent estimates side by side.

  Observed   the 5°S–5°N divergent zonal wind at 200 minus 850 hPa (W, m/s) from the AIFS-ENS 0-h
             analyses kept by walker.py, as an anomaly against the ERA5 1991–2020 harmonic
             climatology (7-day and 30-day trailing means).
  SST        OISST v2.1 (PSL) last 30 days, anomaly vs the 1991–2020 base, with the 1991–2020
             linear trend removed per calendar month (from ERSST) so a warming ocean is not
             read as a forcing anomaly. Basin indices in units of their training σ.
  Leg A      the regression built by build_walker_basins_clim.py (ERA5 + ERSST 1991–2020):
             W regressed on Niño-3.4 (and its 3-month lag), then on the Indian (IOD, basin mean)
             and Atlantic (ATL3, tropical North Atlantic) indices as residuals after ENSO.
  Leg B      the Gill (1980) linear response to the SST anomaly of each basin, with the ENSO
             regression pattern removed from the Indian and Atlantic fields; one amplitude,
             calibrated on ENSO in the builder, applies to every basin.
  Tests      every regression part is drawn only where it is significant: a Freedman-Lane
             permutation of whole training years (perm_pvalues), Benjamini-Hochberg q = FDR_Q
             over each figure element's tests (the profile's 4 groups x 180 longitudes, the
             table's 12 sector cells, the R² panel's 48 bars).

Outputs: assets/sst/walker_basins.webp, assets/sst/data/walker_basins.json.
    SST_SITE_ROOT=/path/to/site python scripts/sst/walker_basins.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.degdays import doy366
from quantmet.stats import benjamini_hochberg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oisst9120
from gill_model import LAT2, LON2, band_u, basin_mask, gill_response, heating_from_ssta

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
ASSETS = SITE / "assets" / "sst"
REF = HERE.parents[1] / "scripts" / "mjo" / "data" / "reference"
CLIM = REF / "walker_basins_clim.nc"
HIST = REF / "walker_ud_history.nc"
ERSST = HERE / "data" / "ersst_v6_mnmean.nc"  # fetched by sst.yml before the Walker steps
PAC_BOX = (140.0, 200.0)  # the walker.py Pacific-cell box, in longitude
INK, MUTED, NAVY, RED, GREEN, ORANGE = "#1a1a1a", "#6f6b64", "#1b365d", "#b4453c", "#2b7a3d", "#d0801a"
GROUPS = {"enso": ["n34", "n34_lag3"], "indian": ["dmi", "iob"], "atlantic": ["atl3", "tna"]}
BASIN_PREDS = ("dmi", "iob", "atl3", "tna")  # entered as residuals after ENSO, in this order
# columns of the training design [Niño-3.4, its lag, dmi, iob, atl3, tna residuals] each part is tested on
PART_COLS = {"enso": [0, 1], "indian": [2, 3], "atlantic": [4, 5], "total": [0, 1, 2, 3, 4, 5]}
FDR_Q = 0.10  # Benjamini-Hochberg false discovery rate over each figure element's tests
NPERM = 999  # whole-year permutations per test
SEED = 20261016
SECTORS = [
    ("Indian Ocean, 50–100°E", 50, 100),
    ("Maritime Continent, 100–150°E", 100, 150),
    ("Pacific Walker box, 140°E–160°W", 140, 200),
    ("East Pacific, 160–90°W", 200, 270),
]
LABEL = {
    "n34": "Niño-3.4",
    "n34_lag3": "Niño-3.4, 3 months ago",
    "dmi": "IOD (DMI)",
    "iob": "Indian Ocean basin",
    "atl3": "Atlantic Niño (ATL3)",
    "tna": "Tropical N. Atlantic",
}


def harmonic_day(t) -> np.ndarray:
    """Common-year day of the year (1..365, 29 Feb = 59.5), the phase coordinate of the harmonic climatologies: 1 March
    is day 60 in every year, so a leap-year date sits at the same point of the annual cycle as in a common year."""
    s = doy366(t)
    return np.where(s < 59, s + 1.0, np.where(s == 59, 59.5, s.astype(float)))


# ── SST ──────────────────────────────────────────────────────────────────────
def current_ssta(ref: xr.Dataset, days: int = 30):
    """(detrended SSTA on the Gill grid, its 30-day window, raw anomaly on the same grid)."""
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    year = now.year
    p = oisst9120.ensure_mean(year)
    ds = xr.open_dataset(p)
    sst = ds["sst"].sel(lat=slice(-31, 31))
    t = pd.DatetimeIndex(sst.time.values)
    if len(t) < days:  # early January: reach into last year
        prev = xr.open_dataset(oisst9120.ensure_mean(year - 1))["sst"].sel(lat=slice(-31, 31))
        sst = xr.concat([prev.isel(time=slice(-days, None)), sst], dim="time")
        t = pd.DatetimeIndex(sst.time.values)
    win = sst.isel(time=slice(-days, None)).load()
    an = oisst9120.anom(win).mean("time")
    an2 = an.coarsen(lat=8, lon=8, boundary="trim").mean()
    an2 = an2.interp(lat=LAT2, lon=LON2, kwargs={"fill_value": None})
    mid = pd.DatetimeIndex(win.time.values)[len(win.time) // 2]
    m = mid.month
    slope = ref.sst_slope.sel(month=m).interp(lat=LAT2, lon_e=LON2, kwargs={"fill_value": None}).values
    yr = mid.year + (mid.dayofyear - 1) / 365.25
    det = an2.values - slope * (yr - 2005.5)  # remove the 1991–2020 trend, extrapolated
    return det, (pd.Timestamp(win.time.values[0]), pd.Timestamp(win.time.values[-1])), an2.values, m


def box_mean(field, lat, lon, box):
    (la0, la1), (lo0, lo1) = box
    msk = ((lat >= la0) & (lat <= la1))[:, None] & basin_mask(lon, lo0, lo1)[None]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)) * msk * np.isfinite(field)
    return float((np.nan_to_num(field) * w).sum() / w.sum())


def indices_from(field, ref, month):
    boxes = json.loads(ref.attrs["boxes"])
    out = {k: box_mean(field, LAT2, LON2, boxes[k]) for k in ("n34", "iob", "atl3", "tna")}
    out["dmi"] = box_mean(field, LAT2, LON2, boxes["dmi_w"]) - box_mean(field, LAT2, LON2, boxes["dmi_e"])
    return out


def n34_lag3(ref: xr.Dataset, end) -> float:
    """Niño-3.4 three months before `end`, from ERSST (the training source), detrended the same way.

    Raises when ERSST does not have that month: the regression's lag term has no stand-in, and the current value is
    not one (with Niño-3.4 still rising it overstates the lagged ENSO forcing)."""
    tm = pd.Timestamp(end) - pd.DateOffset(months=3)
    with xr.open_dataset(ERSST) as ds:
        e = ds["sst"].sel(time=f"{tm.year}-{tm.month:02d}")
        if e.time.size == 0:
            raise LookupError(f"{ERSST.name} has no {tm:%Y-%m}")
        fld = e.isel(time=0).values.astype(float)
        lat, lon = ds.lat.values, ds.lon.values
    an3 = (
        fld
        - ref.sst_clim.sel(month=tm.month).values
        - ref.sst_slope.sel(month=tm.month).values * (tm.year + (tm.month - 0.5) / 12 - 2005.5)
    )
    (la0, la1), (lo0, lo1) = json.loads(ref.attrs["boxes"])["n34"]
    msk = ((lat >= la0) & (lat <= la1))[:, None] & ((lon >= lo0) & (lon <= lo1))[None]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)) * msk * np.isfinite(an3)
    if w.sum() == 0:
        raise ValueError(f"no ERSST values in the Niño-3.4 box for {tm:%Y-%m}")
    return float((np.nan_to_num(an3) * w).sum() / w.sum())


# ── significance of the regression ───────────────────────────────────────────
def training_design(ref: xr.Dataset, month: int):
    """The builder's regression for `month`, rebuilt from the reference's training series (idx_train, w_train): the
    design [Niño-3.4, Niño-3.4 three months ago, then the four basin indices as residuals after those two], centred,
    the centred W anomalies (rows x longitudes), and each row's calendar year. Rows = the ±1-month window, 1991-2020."""
    t = pd.DatetimeIndex(ref.time.values)
    preds = list(ref.pred.values)
    X = ref.idx_train.values.astype(float)
    win = np.isin(t.month, [(month - 2) % 12 + 1, month, month % 12 + 1]) & np.isfinite(X).all(1)
    X = X[win]
    E = X[:, [preds.index("n34"), preds.index("n34_lag3")]]
    Ec = E - E.mean(0)
    cols = [Ec]
    for k in BASIN_PREDS:
        xk = X[:, preds.index(k)] - X[:, preds.index(k)].mean()
        g, *_ = np.linalg.lstsq(Ec, xk, rcond=None)
        cols.append((xk - Ec @ g)[:, None])
    Y = ref.w_train.values.astype(float)[win]
    return np.hstack(cols), Y - Y.mean(0), t.year.values[win]


def _extra_ss(X, Y, cols):
    """(extra sum of squares of columns `cols` of X over the rest, the full model's residual sum of squares), per
    column of Y; their ratio is a pseudo-F, monotone in F for fixed degrees of freedom."""
    keep = [c for c in range(X.shape[1]) if c not in cols]

    def rss(A):
        if A.shape[1] == 0:
            return (Y**2).sum(0)
        b, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return ((Y - A @ b) ** 2).sum(0)

    full = rss(X)
    return rss(X[:, keep]) - full, full


def perm_pvalues(X, Y, years, cols, nperm=NPERM, rng=None):
    """Freedman-Lane permutation p, per column of Y and pooled over all columns, for adding columns `cols` of X: the
    reduced model's residuals are permuted between whole years - the three months of a year share most of their ENSO,
    basin and Walker anomalies, so they must not be counted as independent samples - and the extra sum of squares
    recomputed. In synthetic nulls built on the training indices (W within-year correlated up to 0.8) this held
    4-6 % at the 5 % level; an F-test counting the 30 years instead held under 1 %, a year-clustered Wald test 10-24 %."""
    rng = np.random.default_rng(SEED) if rng is None else rng
    keep = [c for c in range(X.shape[1]) if c not in cols]
    if keep:
        b, *_ = np.linalg.lstsq(X[:, keep], Y, rcond=None)
        fit = X[:, keep] @ b
    else:
        fit = np.zeros_like(Y)
    res = Y - fit
    _, inv = np.unique(years, return_inverse=True)
    blocks = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    if len({len(b_) for b_ in blocks}) != 1:
        raise ValueError("training years with unequal month counts cannot be permuted whole")
    blocks = np.array(blocks)

    def stat(Yp):
        extra, full = _extra_ss(X, Yp, cols)
        return extra / full, float(extra.sum() / full.sum())

    s0, p0 = stat(Y)
    cnt = np.zeros(Y.shape[1])
    cntp = 0
    idx = np.empty(len(years), int)
    for _ in range(nperm):
        idx[blocks.ravel()] = blocks[rng.permutation(len(blocks))].ravel()
        s, p = stat(fit + res[idx])
        cnt += s >= s0
        cntp += p >= p0
    return (cnt + 1) / (nperm + 1), (cntp + 1) / (nperm + 1)


def sector_mean(a, lo, hi):
    m = (LON2 >= lo) & (LON2 <= hi)  # noqa: SIM300  coordinate array LON2, not a constant
    return np.nanmean(np.asarray(a)[..., m], axis=-1)


def regression_tests(ref: xr.Dataset, month: int) -> dict:
    """Which regression parts the figure may show: for `month`, each part (ENSO, Indian, Atlantic, total) per longitude
    and the three basin parts per sector, each set under BH q = FDR_Q; and, for every month, the R² panel's ENSO-only
    R² and its gains from adding the Indian, the Atlantic and all four basin indices (permutation of the pooled extra
    sum of squares, all 48 under one BH)."""
    rng = np.random.default_rng(SEED)
    X, Y, years = training_design(ref, month)
    p_lon = {g: perm_pvalues(X, Y, years, c, rng=rng)[0] for g, c in PART_COLS.items()}
    flat = benjamini_hochberg(np.concatenate(list(p_lon.values())), FDR_Q)
    sig_lon = dict(zip(p_lon, np.split(flat, len(p_lon))))
    Ys = np.stack([sector_mean(Y, lo, hi) for _, lo, hi in SECTORS], 1)
    p_sec = {g: perm_pvalues(X, Ys, years, PART_COLS[g], rng=rng)[0] for g in ("enso", "indian", "atlantic")}
    flat = benjamini_hochberg(np.concatenate(list(p_sec.values())), FDR_Q)
    sig_sec = dict(zip(p_sec, np.split(flat, len(p_sec))))
    # R² panel: ENSO alone, + Indian, + Atlantic, + all basins (the stages of ref.r2), each against what it adds
    stages = (([0, 1], [0, 1]), ([0, 1, 2, 3], [2, 3]), ([0, 1, 4, 5], [2, 3]), ([0, 1, 2, 3, 4, 5], [2, 3, 4, 5]))
    p_r2 = np.ones((12, 4))
    for m in range(1, 13):
        Xm, Ym, ym = (X, Y, years) if m == month else training_design(ref, m)
        for s, (use, cols) in enumerate(stages):
            p_r2[m - 1, s] = perm_pvalues(Xm[:, use], Ym, ym, cols, rng=rng)[1]
    sig_r2 = benjamini_hochberg(p_r2, FDR_Q)
    return {"p_lon": p_lon, "sig_lon": sig_lon, "p_sec": p_sec, "sig_sec": sig_sec, "p_r2": p_r2, "sig_r2": sig_r2}


# ── observed Walker ──────────────────────────────────────────────────────────
def observed_w(ref: xr.Dataset):
    """W anomaly (7-day and 30-day trailing means) on LON2 from the AIFS analyses, and the dates."""
    h = xr.open_dataset(HIST)["ud"]
    t = pd.DatetimeIndex(h.time.values)
    lon = h.longitude.values

    def band(lev):
        a = h.sel(level=lev, method="nearest").values  # (time, lon)
        return np.stack(
            [np.interp(LON2, np.concatenate([lon, [lon[0] + 360]]), np.concatenate([row, [row[0]]])) for row in a]
        )

    w = band(200) - band(850)
    ang = 2 * np.pi * harmonic_day(t) / 365.25
    B = np.stack([np.ones_like(ang), np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)], -1)
    clim = B @ ref.ud200_coef.values - B @ ref.ud850_coef.values
    # the same detrending the regression saw: subtract the extrapolated 1991–2020 trend of W for the month
    yr = t.year.values + (t.dayofyear.values - 1) / 365.25
    trend = np.stack([ref.w_slope.sel(month=m).values * (y - 2005.5) for m, y in zip(t.month.values, yr)])
    wa = w - clim - trend

    def trailing(n):
        sel = t > t[-1] - pd.Timedelta(days=n)
        return np.nanmean(wa[sel], axis=0), int(sel.sum())

    w7, n7 = trailing(7)
    w30, n30 = trailing(30)
    return w7, w30, n7, n30, t[-1]


# ── figure ───────────────────────────────────────────────────────────────────
def render(
    out: Path, ref, ssta, ssta_res, window, month, idx, idx_sig, w7, w30, n7, n30, tlast, reg, gill, gmaps, r2, tests
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import cartopy.crs as ccrs
    import textwrap

    shown = {g: np.where(tests["sig_lon"][g], reg[g], np.nan) for g in PART_COLS}  # parts only where significant

    fig = plt.figure(figsize=(13.4, 11.8))
    gs = GridSpec(
        6,
        2,
        height_ratios=[0.44, 0.02, 1.2, 1.0, 0.44, 0.8],
        hspace=0.5,
        wspace=0.1,
        left=0.055,
        right=0.985,
        top=0.932,
        bottom=0.02,
    )
    pc = ccrs.PlateCarree(central_longitude=180)
    boxes = json.loads(ref.attrs["boxes"])
    basins = json.loads(ref.attrs["basin_lon"])
    import mapstyle as MS  # the site's shared SST palette (scripts/sst/mapstyle.py)

    sty = MS.sst_anom_style(3.0)  # white band +-0.2, 0.2 degC classes to +-3

    def sst_panel(k, field, title):
        ax = fig.add_subplot(gs[0, k], projection=pc)
        cf = ax.contourf(
            LON2,
            LAT2,
            field,
            levels=sty["levels"],
            cmap=sty["cmap"],
            norm=sty["norm"],
            extend="both",
            transform=ccrs.PlateCarree(),
        )
        MS.sst_features(ax, scale="110m", coast_lw=0.45)
        for name, ((la0, la1), (lo0, lo1)) in boxes.items():
            colr = {"n34": "#141414", "dmi_w": GREEN, "dmi_e": GREEN, "iob": GREEN, "atl3": ORANGE, "tna": ORANGE}[name]
            ax.add_patch(
                plt.Rectangle(
                    (lo0, la0),
                    lo1 - lo0,
                    la1 - la0,
                    fill=False,
                    ec=colr,
                    lw=1.1,
                    ls=("--" if name in ("iob",) else "-"),
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
            )
        for lo0, _lo1 in basins.values():
            ax.plot([lo0, lo0], [-30, 30], color="#333", lw=0.6, ls=":", transform=ccrs.PlateCarree(), zorder=5)
        ax.set_extent([0, 359.99, -30, 30], crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=9.5, loc="left", fontweight="bold")
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="#bbb", x_inline=False, y_inline=False)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = k == 0
        gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
        return cf

    cf = sst_panel(0, ssta, f"SST anomaly, OISST {window[0]:%d %b}–{window[1]:%d %b} (1991–2020 base, trend removed)")
    sst_panel(1, ssta_res, "The same field with the ENSO regression pattern removed")
    fig.canvas.draw()  # cartopy fixes the map aspect at draw time
    b0 = min((a.get_position() for a in fig.axes if getattr(a, "projection", None) is not None), key=lambda b: b.y0)
    cax = fig.add_axes([0.40, b0.y0 - 0.022, 0.20, 0.006])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", spacing="uniform")
    MS.sst_colorbar(
        cb,
        sty,
        "°C   ·   boxes: black Niño-3.4, green IOD poles and basin (dashed), orange ATL3 and TNA",
        size=7,
        tick_size=6.5,
    )

    # W profile
    ax = fig.add_subplot(gs[2, :])
    for b, (lo0, lo1) in basins.items():
        ax.axvspan(lo0, min(lo1, 360), color={"indian": GREEN, "pacific": RED, "atlantic": ORANGE}[b], alpha=0.04, lw=0)
        if lo1 > 360:
            ax.axvspan(0, lo1 - 360, color=ORANGE, alpha=0.04, lw=0)
    ax.axhline(0, color="#555", lw=0.8)
    ax.plot(LON2, w7, color="#222", lw=1.1, alpha=0.8, label=f"observed, last {n7} analyses (7 d)")
    ax.plot(LON2, w30, color="#000", lw=2.4, label=f"observed, last {n30} analyses (30 d)")
    ax.plot(LON2, shown["total"], color=NAVY, lw=2.0, ls="--", label="regression total")
    ax.plot(LON2, shown["enso"], color=RED, lw=1.8, label="ENSO part (Niño-3.4 now and 3 months ago)")
    ax.plot(LON2, shown["indian"], color=GREEN, lw=1.8, label="Indian Ocean part, ENSO removed")
    ax.plot(LON2, shown["atlantic"], color=ORANGE, lw=1.8, label="Atlantic part, ENSO removed")
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 30))
    ax.set_xticklabels([f"{x}°E" if x <= 180 else f"{360 - x}°W" for x in range(0, 361, 30)], fontsize=7.5)
    ax.set_ylabel("W = u_D(200) − u_D(850), m/s", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.2)
    ax.set_title(
        "Zonal overturning anomaly along the equator: observed vs the ENSO-first regression, parts drawn where significant "
        "(+ = upper-level eastward flow)",
        fontsize=9.2,
        loc="left",
        fontweight="bold",
    )
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in ax.get_yticks() if lo <= t <= hi])
    ax.set_ylim(lo - 0.42 * (hi - lo), hi)  # blank band under the data for the legend
    ax.legend(fontsize=7.4, ncol=3, frameon=False, loc="lower center")

    # Gill profiles + skill
    ax = fig.add_subplot(gs[3, 0])
    ax.axhline(0, color="#555", lw=0.8)
    for key, colr, lab in (
        ("pacific", RED, "Pacific SST (full)"),
        ("indian", GREEN, "Indian Ocean, ENSO removed"),
        ("atlantic", ORANGE, "Atlantic, ENSO removed"),
    ):
        ax.plot(LON2, gill[key], color=colr, lw=1.9, label=f"Gill: {lab}")
    ax.plot(LON2, gill["total"], color=NAVY, lw=1.4, ls="--", label="Gill: all three")
    ax.plot(LON2, shown["indian"], color=GREEN, lw=1.0, ls=":", label="regression Indian (for comparison)")
    ax.plot(LON2, shown["atlantic"], color=ORANGE, lw=1.0, ls=":", label="regression Atlantic")
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 60))
    ax.set_xticklabels([f"{x}°E" if x <= 180 else f"{360 - x}°W" for x in range(0, 361, 60)], fontsize=7.5)
    ax.set_ylabel("W-equivalent, m/s", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.2)
    ax.set_title(
        "Gill (1980) response to each basin's SST anomaly, amplitude calibrated on ENSO",
        fontsize=9.2,
        loc="left",
        fontweight="bold",
    )
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in ax.get_yticks() if lo <= t <= hi])
    ax.set_ylim(lo - 0.38 * (hi - lo), hi)
    ax.legend(fontsize=6.8, ncol=2, frameon=False, loc="lower left")
    ax = fig.add_subplot(gs[3, 1])
    mm = np.arange(1, 13)
    wdt = 0.2
    for k, (col, lab, c) in enumerate(
        ((0, "ENSO only", RED), (1, "+ Indian", GREEN), (2, "+ Atlantic", ORANGE), (3, "all", NAVY))
    ):
        ok = tests["sig_r2"][:, col]  # a gain over ENSO alone is drawn only where it is significant
        ax.bar(mm[ok] + (k - 1.5) * wdt, r2[ok, col], width=wdt, color=c, alpha=0.85, label=lab)
    ax.axvspan(month - 0.5, month + 0.5, color="#000", alpha=0.06, lw=0)
    ax.set_xticks(mm)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax.set_ylabel("R² of monthly W, all longitudes", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.99,
        0.80,
        "a gain over ENSO alone is drawn only where significant",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.8,
        color=MUTED,
    )
    ax.set_title(
        "Variance of monthly W explained, ERA5 1991–2020 (this month shaded)",
        fontsize=9.2,
        loc="left",
        fontweight="bold",
    )
    ax.legend(fontsize=7.2, ncol=4, frameon=False, loc="upper left")

    # Gill maps
    for k, (key, title) in enumerate(
        (
            ("indian", "Gill response to the Indian Ocean anomaly (ENSO removed): low-level wind and pressure"),
            ("atlantic", "Gill response to the Atlantic anomaly (ENSO removed)"),
        )
    ):
        ax = fig.add_subplot(gs[4, k], projection=pc)
        u, v, p = gmaps[key]
        pm = np.nanmax(np.abs(p)) or 1.0
        ax.contourf(
            LON2, LAT2, p, levels=np.linspace(-pm, pm, 12), cmap="PuOr", extend="both", transform=ccrs.PlateCarree()
        )
        sk = 2
        ax.quiver(
            LON2[::sk],
            LAT2[::sk],
            u[::sk, ::sk],
            v[::sk, ::sk],
            transform=ccrs.PlateCarree(),
            scale=np.nanmax(np.hypot(u, v)) * 25 + 1e-9,
            width=0.0022,
            color="#222",
        )
        ax.coastlines(lw=0.5, color="#555")
        ax.set_extent([0, 359.99, -30, 30], crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=8.8, loc="left", fontweight="bold")
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="#bbb", x_inline=False, y_inline=False)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = k == 0
        gl.xlabel_style = gl.ylabel_style = {"size": 6.5}

    # summary: sector table (what each basin does where) + the index line
    ax = fig.add_subplot(gs[5, :])
    ax.axis("off")
    cols = [
        "Sector",
        "Observed (30 d)",
        "ENSO part",
        "Indian, ENSO removed\nregression / Gill",
        "Atlantic, ENSO removed\nregression / Gill",
        "Residual",
    ]

    def part(g, i, v):
        return f"{v:+.1f}" if tests["sig_sec"][g][i] else "n.s."

    rows, tint = [], {}
    for i, (name, lo, hi) in enumerate(SECTORS):
        o = float(sector_mean(w30, lo, hi))
        e, i_r, a_r = (float(sector_mean(reg[g], lo, hi)) for g in ("enso", "indian", "atlantic"))
        i_g, a_g = (float(sector_mean(gill[g], lo, hi)) for g in ("indian", "atlantic"))
        sig = {g: bool(tests["sig_sec"][g][i]) for g in ("enso", "indian", "atlantic")}
        resid = o - sum(v for g, v in (("enso", e), ("indian", i_r), ("atlantic", a_r)) if sig[g])
        rows.append(
            [
                name,
                f"{o:+.1f}",
                part("enso", i, e),
                f"{part('indian', i, i_r)} / {i_g:+.1f}",
                f"{part('atlantic', i, a_r)} / {a_g:+.1f}",
                f"{resid:+.1f}",
            ]
        )
        # basin cells: same sign as a significant ENSO part = reinforcing (red), opposite = damping (green)
        for c, g, v in ((3, "indian", i_r), (4, "atlantic", a_r)):
            if sig[g] and sig["enso"] and abs(v) >= 0.05:  # a significant link with a part that rounds to 0 is untinted
                tint[(i + 1, c)] = "#fbe9e7" if np.sign(v) == np.sign(e) else "#e8f3ea"
    tb = ax.table(
        cellText=rows,
        colLabels=cols,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.18, 0.76, 0.82],
        colWidths=[0.26, 0.13, 0.11, 0.18, 0.18, 0.10],
    )
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.6)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#d9d6d0")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#f1efe9")
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_height(cell.get_height() * 1.5)
        elif c == 0:
            cell.set_text_props(ha="left", color=INK)
            cell._loc = "left"
            cell.PAD = 0.03
        if (r, c) in tint:
            cell.set_facecolor(tint[(r, c)])
    ax.text(
        0.775,
        0.99,
        "\n".join(
            textwrap.wrap(
                "Units m/s of W; negative = overturning weakened. A regression part (ERA5 + ERSSTv6 1991–2020, ENSO "
                "first, the basins as residuals) is shown only where its link to W is significant: whole-year "
                "permutation, false discovery rate 10 % over the table; n.s. otherwise. Tint: a significant basin part "
                "with the sign of the ENSO part (red, reinforcing) or against it (green, damping). Residual = observed "
                "minus the significant parts. Gill is a model run, not a test, and is shown in full.",
                54,
            )
        ),
        transform=ax.transAxes,
        fontsize=6.9,
        va="top",
        color=MUTED,
        linespacing=1.3,
    )
    sig_r2 = tests["sig_r2"][month - 1]
    ax.text(
        0.0,
        0.08,
        "Indices, last 30 days, detrended, σ of the month:  "
        + "   ".join(f"{LABEL[k]} {idx_sig[k]:+.1f}σ" for k in ("n34", "dmi", "iob", "atl3", "tna"))
        + "      R² this month: "
        + (f"ENSO alone {r2[month - 1, 0]:.2f}" if sig_r2[0] else "ENSO alone not significant")
        + (f", all basins {r2[month - 1, 3]:.2f}" if sig_r2[3] else ", the other basins add nothing significant"),
        transform=ax.transAxes,
        fontsize=7.8,
        va="top",
        color=INK,
    )
    fig.suptitle(
        "Walker circulation by ocean basin — the Indian and Atlantic contributions with ENSO removed",
        fontsize=13.5,
        fontweight="bold",
        x=0.055,
        ha="left",
        y=0.988,
    )
    fig.text(
        0.055,
        0.966,
        f"Observed: AIFS-ENS 0-h analyses, divergent wind at 200 minus 850 hPa over 5°S–5°N, anomaly against ERA5 1991–2020 · latest analysis {tlast:%Y-%m-%d %HZ} · SST: OISST last 30 days, detrended",
        fontsize=8,
        color=MUTED,
        va="top",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # 140 dpi: the 13.4 in figure is ~1880 px, sharp at the width the deck shows it on a retina screen
    fig.savefig(out, dpi=140, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"saved {out}", flush=True)


def main() -> int:
    t0 = time.time()
    if not CLIM.exists():
        raise SystemExit("reference missing: run scripts/mjo/src/build_walker_basins_clim.py")
    ref = xr.open_dataset(CLIM).load()
    ssta, window, _ssta_raw, month = current_ssta(ref)
    idx = indices_from(ssta, ref, month)
    sd = {k: float(ref.pred_sd.sel(month=month, pred=k)) for k in ref.pred.values}
    idx_sig = {k: idx[k] / sd[k] for k in idx}
    try:
        idx["n34_lag3"] = n34_lag3(ref, window[1])
    except Exception as ex:  # noqa: BLE001
        raise SystemExit(f"lagged Niño-3.4 unavailable ({ex}); nothing written") from ex
    idx_sig["n34_lag3"] = idx["n34_lag3"] / sd["n34_lag3"]

    # leg A: regression contributions (basin indices as residuals after ENSO, as trained)
    ensov = np.array([idx["n34"], idx["n34_lag3"]])
    ops = ref.resid_ops.sel(month=month).values  # (basin_pred, enso_pred)
    resid = {k: idx[k] - float(ops[i] @ ensov) for i, k in enumerate(("dmi", "iob", "atl3", "tna"))}
    coef = ref.coef.sel(month=month)
    part = {}
    for k in ref.pred.values:
        val = ensov[0] if k == "n34" else ensov[1] if k == "n34_lag3" else resid[k]
        part[k] = coef.sel(pred=k).values * val
    reg = {g: sum(part[k] for k in ks) for g, ks in GROUPS.items()}
    reg["total"] = reg["enso"] + reg["indian"] + reg["atlantic"]

    # leg B: Gill, per basin, with the ENSO pattern removed from the Indian and Atlantic fields
    pat = ref.enso_pattern.sel(month=month).interp(lat=LAT2, lon_e=LON2, kwargs={"fill_value": None}).values
    ssta_res = ssta - pat * idx["n34"]
    clim2 = ref.sst_clim.sel(month=month).interp(lat=LAT2, lon_e=LON2, kwargs={"fill_value": None}).values
    alpha = float(ref.attrs["gill_alpha"])
    basins = json.loads(ref.attrs["basin_lon"])
    gill, gmaps = {}, {}
    for b in ("pacific", "indian", "atlantic"):
        src = ssta if b == "pacific" else ssta_res
        Q = heating_from_ssta(np.where(basin_mask(LON2, *basins[b])[None], src, 0.0), clim2)
        u, v, p = gill_response(Q)
        gill[b] = -2.0 * alpha * band_u(u)
        gmaps[b] = (u, v, p)
    gill["total"] = gill["pacific"] + gill["indian"] + gill["atlantic"]

    w7, w30, n7, n30, tlast = observed_w(ref)
    r2 = ref.r2.values
    tests = regression_tests(ref, month)
    render(
        ASSETS / "walker_basins.webp",
        ref,
        ssta,
        ssta_res,
        window,
        month,
        idx,
        idx_sig,
        w7,
        w30,
        n7,
        n30,
        tlast,
        reg,
        gill,
        gmaps,
        r2,
        tests,
    )
    pac = (LON2 >= PAC_BOX[0]) & (LON2 <= PAC_BOX[1])  # noqa: SIM300  coordinate array LON2, not a constant
    ipac = next(i for i, s in enumerate(SECTORS) if (s[1], s[2]) == PAC_BOX)
    box = {g: float(np.nanmean(reg[g][pac])) for g in ("enso", "indian", "atlantic")}
    box_sig = {g: bool(tests["sig_sec"][g][ipac]) for g in box}
    sig_r2 = tests["sig_r2"][month - 1]
    doc = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "sst_window": [window[0].strftime("%Y-%m-%d"), window[1].strftime("%Y-%m-%d")],
        "month": int(month),
        "last_analysis": tlast.strftime("%Y-%m-%dT%HZ"),
        "indices_degC": {k: round(v, 3) for k, v in idx.items()},
        "indices_sigma": {k: round(v, 2) for k, v in idx_sig.items()},
        "lon": LON2.tolist(),
        "observed_w7": np.round(w7, 3).tolist(),
        "observed_w30": np.round(w30, 3).tolist(),
        # null where the part is not significant (whole-year permutation, BH over the profile)
        "regression": {
            k: [round(float(v), 3) if s else None for v, s in zip(reg[k], tests["sig_lon"][k])] for k in PART_COLS
        },
        "regression_p": {k: [float(f"{p:.3g}") for p in v] for k, v in tests["p_lon"].items()},
        "gill": {k: np.round(v, 3).tolist() for k, v in gill.items()},
        "pacific_box": {
            "observed_30d": round(float(np.nanmean(w30[pac])), 3),
            "observed_7d": round(float(np.nanmean(w7[pac])), 3),
            **{f"reg_{k}": round(v, 3) if box_sig[k] else None for k, v in box.items()},
            "reg_significant_sum": round(sum(v for k, v in box.items() if box_sig[k]), 3),
            **{f"gill_{k}": round(float(np.nanmean(v[pac])), 3) for k, v in gill.items()},
        },
        "r2_this_month": {
            "enso": round(float(r2[month - 1, 0]), 3) if sig_r2[0] else None,
            "full": round(float(r2[month - 1, 3]), 3) if sig_r2[3] else None,
        },
        "tests": {
            "method": "Freedman-Lane permutation of whole training years (1991-2020, ±1-month window), "
            f"{NPERM} permutations; Benjamini-Hochberg q {FDR_Q} over the profile (4 parts x {LON2.size} longitudes), "
            "the sector table (12 cells) and the R² panel (12 months x 4 bars)",
            "r2_p": np.round(tests["p_r2"], 4).tolist(),
            "r2_significant": tests["sig_r2"].tolist(),
        },
        "gill_alpha": alpha,
    }
    (ASSETS / "data").mkdir(parents=True, exist_ok=True)
    (ASSETS / "data" / "walker_basins.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote walker_basins.json in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
