#!/usr/bin/env python3
"""Weekly departure of the tropical overturning from Pacific forcing — the "is it the IOD?" board.

For each of the last four weeks (7-day means of the AIFS-ENS 0-h analyses) and the 28-day mean:
  observed      χ200 anomaly (velocity potential, spherical harmonics, vs the ERA5 1991–2020 harmonic
                climatology, 1991–2020 trend removed per calendar month) — negative = upper-level
                divergence = enhanced deep convection;
  Pacific part  the ENSO regression of χ200 on Niño-3.4 (now and 3 months ago), ERA5+ERSST 1991–2020,
                fed with the week's detrended OISST Niño-3.4;
  departure     observed − Pacific part: what Pacific SST does not account for;
  Indian+Atl.   the regression's ENSO-independent Indian and Atlantic part (IOD, basin mean, ATL3,
                TNA as residuals after ENSO) — the most those SST anomalies are expected to do;
and for the 28-day SST the Gill (1980) responses (Pacific only; Indian + Atlantic with the ENSO
pattern removed) converted to an upper-level velocity potential. Each departure panel says whether
the Indian+Atlantic SST part matches it (15°S–15°N), and how much of its variance that part explains,
so a "but the IOD is negative" claim can be sized on the spot.
Tests: a regression part is kept only where its predictors are significant at that grid point
(r2_pvalues, BH over the strip) and is 0 (hatched) elsewhere; a pattern match is quoted only when it
beats every circular longitude shift (shift_p, one-sided, BH over the figure's seven comparisons).

Outputs: assets/sst/walker_chi_weekly.webp, assets/sst/data/walker_chi_weekly.json.
    SST_SITE_ROOT=/path/to/site python scripts/sst/walker_chi_weekly.py
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
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "mjo" / "src"))
import oisst9120
from gill_model import LAT2, LON2, A_EARTH, basin_mask, gill_response, heating_from_ssta
from walker_basins import n34_lag3
import walker_chi as wc
from matplotlib.colors import LinearSegmentedColormap

# same palette and levels as wind200_vpot.py (not imported: that module needs pyshtools, which the sst runner lacks)
VP_CMAP = LinearSegmentedColormap.from_list(
    "vpot", ["#1b5e20", "#43a047", "#86c98a", "#cfe8cf", "#ffffff", "#fbe2bd", "#f0a64b", "#df6a1e", "#a8330f"]
)
VP_LEVELS = [-16, -12, -8, -5, -3, -1.5, 1.5, 3, 5, 8, 12, 16]

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
ASSETS = SITE / "assets" / "sst"
REF = HERE.parents[1] / "scripts" / "mjo" / "data" / "reference"
CLIM = REF / "walker_chi_clim.nc"
WREF = REF / "walker_basins_clim.nc"
HIST = ASSETS / "anim" / "walker" / wc.HIST_NAME
NWEEKS = 4
STAT_LAT = 15.0
INK, MUTED, NAVY, RED, GREEN, ORANGE = "#1a1a1a", "#6f6b64", "#1b365d", "#b4453c", "#2b7a3d", "#d0801a"
GROUPS = {"enso": ["n34", "n34_lag3"], "indian": ["dmi", "iob"], "atlantic": ["atl3", "tna"]}
FDR_Q = 0.10  # Benjamini-Hochberg false discovery rate: over the strip's grid points, and over the panel matches
N_EFF = 62  # independent samples among the 90 training months, for the R²-based F-test (r2_pvalues)


def harmonic_day(t) -> np.ndarray:
    """Common-year day of the year (1..365, 29 Feb = 59.5), the phase coordinate of the harmonic climatologies: 1 March
    is day 60 in every year, so a leap-year date sits at the same point of the annual cycle as in a common year."""
    s = doy366(t)
    return np.where(s < 59, s + 1.0, np.where(s == 59, 59.5, s.astype(float)))


# ── SST for an arbitrary window ───────────────────────────────────────────────
def oisst_days(start: pd.Timestamp, end: pd.Timestamp) -> xr.DataArray:
    parts = []
    for y in range(start.year, end.year + 1):
        ds = xr.open_dataset(oisst9120.ensure_mean(y))
        parts.append(ds["sst"].sel(lat=slice(-31, 31)))
    sst = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    return sst.sel(time=slice(start, end)).load()


def ssta_window(wref: xr.Dataset, start: pd.Timestamp, end: pd.Timestamp):
    """Detrended OISST anomaly on the Gill grid, mean over [start, end]; None if OISST has no days there."""
    win = oisst_days(start, end)
    if win.time.size == 0:
        return None, None
    an = oisst9120.anom(win).mean("time")
    an2 = an.coarsen(lat=8, lon=8, boundary="trim").mean().interp(lat=LAT2, lon=LON2, kwargs={"fill_value": None})
    mid = pd.DatetimeIndex(win.time.values)[len(win.time) // 2]
    slope = wref.sst_slope.sel(month=mid.month).interp(lat=LAT2, lon_e=LON2, kwargs={"fill_value": None}).values
    yr = mid.year + (mid.dayofyear - 1) / 365.25
    return an2.values - slope * (yr - 2005.5), mid


def box_mean(field, lat, lon, box):
    (la0, la1), (lo0, lo1) = box
    msk = ((lat >= la0) & (lat <= la1))[:, None] & basin_mask(lon, lo0, lo1)[None]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)) * msk * np.isfinite(field)
    return float((np.nan_to_num(field) * w).sum() / w.sum())


def indices_from(field, wref):
    boxes = json.loads(wref.attrs["boxes"])
    out = {k: box_mean(field, LAT2, LON2, boxes[k]) for k in ("n34", "iob", "atl3", "tna")}
    out["dmi"] = box_mean(field, LAT2, LON2, boxes["dmi_w"]) - box_mean(field, LAT2, LON2, boxes["dmi_e"])
    return out


# ── observed χ ────────────────────────────────────────────────────────────────
def chi_anomalies(h: xr.DataArray, clim: xr.Dataset) -> xr.DataArray:
    """χ200 anomaly per analysis: minus the harmonic climatology and the extrapolated 1991–2020 trend."""
    t = pd.DatetimeIndex(h.time.values)
    ang = 2 * np.pi * harmonic_day(t) / 365.25
    B = np.stack([np.ones_like(ang), np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)], -1)  # (t, 5)
    c = clim.chi_coef.sel(level=200).values  # (5, lat, lon)
    cl = np.einsum("th,hij->tij", B, c)
    yr = t.year.values + (t.dayofyear.values - 1) / 365.25
    sl = clim.chi_slope.sel(level=200).values[t.month.values - 1]  # (t, lat, lon)
    a = h.sel(level=200).values - cl - sl * (yr - 2005.5)[:, None, None]
    return xr.DataArray(
        a,
        dims=("time", "latitude", "longitude"),
        coords={"time": h.time, "latitude": h.latitude, "longitude": h.longitude},
    )


# ── Gill → upper-level velocity potential ─────────────────────────────────────
def poisson_chi(div: np.ndarray, lat=LAT2, lon=LON2) -> np.ndarray:
    """χ with ∇²χ = div on the (lat, lon) strip: FFT in longitude (periodic), second-order finite
    differences in latitude with χ = 0 at the strip walls; metric terms of the sphere kept."""
    latr = np.deg2rad(lat)
    dphi = latr[1] - latr[0]
    ny, nx = div.shape
    cosp = np.cos(latr)
    k = np.fft.rfftfreq(nx, d=1.0 / nx)  # integer zonal wavenumbers
    dk = np.fft.rfft(div, axis=1)  # (ny, nk)
    out = np.zeros_like(dk)
    # (1/cosφ) d/dφ (cosφ dχ/dφ) − m²/cos²φ χ = div·a²
    cosh = np.cos(0.5 * (latr[1:] + latr[:-1]))  # half levels
    for j, m in enumerate(k):
        A = np.zeros((ny, ny), dtype=complex)
        b = dk[:, j] * A_EARTH**2
        for i in range(ny):
            if i in (0, ny - 1):
                A[i, i] = 1.0
                b[i] = 0.0
                continue
            lo, hi = cosh[i - 1] / (cosp[i] * dphi**2), cosh[i] / (cosp[i] * dphi**2)
            A[i, i - 1] = lo
            A[i, i + 1] = hi
            A[i, i] = -(lo + hi) - (m / cosp[i]) ** 2
        out[:, j] = np.linalg.solve(A, b)
    return np.fft.irfft(out, n=nx, axis=1).real


def gill_chi200(Q: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Upper-level χ implied by the Gill response: upper wind = −(lower wind) (first baroclinic mode),
    divergence on the sphere, then the Poisson inversion above. Nondimensional until scaled: the W
    amplitude α was calibrated on the band-mean TOTAL Gill wind, most of which is rotational, so the
    χ leg gets its own single amplitude — Gill χ for the ENSO SST pattern matched, in RMS over the
    statistics strip, to the ENSO regression χ pattern of the same month (see gill_chi_scale)."""
    u, v, _ = gill_response(Q)
    uu, vv = -alpha * u, -alpha * v
    latr = np.deg2rad(LAT2)
    lonr = np.deg2rad(LON2)
    cosp = np.cos(latr)[:, None]
    du = (np.roll(uu, -1, 1) - np.roll(uu, 1, 1)) / (2 * (lonr[1] - lonr[0]))
    dv = np.gradient(vv * cosp, latr, axis=0)
    return poisson_chi((du + dv) / (A_EARTH * cosp))


def gill_chi_scale(wref: xr.Dataset, clim: xr.Dataset, month: int, lat, lon, basins: dict) -> float:
    """m² s⁻¹ per nondimensional Gill χ unit: RMS(regression χ200 per 1σ Niño-3.4) / RMS(Gill χ for the
    1σ ENSO SST pattern over the Pacific), both over |lat| ≤ STAT_LAT."""
    sd = float(wref.pred_sd.sel(month=month, pred="n34"))
    pat = wref.enso_pattern.sel(month=month).interp(lat=LAT2, lon_e=LON2, kwargs={"fill_value": None}).values
    clim2 = wref.sst_clim.sel(month=month).interp(lat=LAT2, lon_e=LON2, kwargs={"fill_value": None}).values
    g = gill_chi200(heating_from_ssta(np.where(basin_mask(LON2, *basins["pacific"])[None], pat * sd, 0.0), clim2))
    reg = clim.coef.sel(month=month, pred="n34").values * sd
    mg = np.abs(LAT2) <= STAT_LAT
    mr = np.abs(lat) <= STAT_LAT
    wg = np.cos(np.deg2rad(LAT2[mg]))[:, None]
    wr = np.cos(np.deg2rad(lat[mr]))[:, None]
    rms_g = np.sqrt((wg * g[mg] ** 2).sum() / (wg * np.ones_like(g[mg])).sum())
    rms_r = np.sqrt((wr * reg[mr] ** 2).sum() / (wr * np.ones_like(reg[mr])).sum())
    return float(rms_r / max(rms_g, 1e-30))


# ── stats ─────────────────────────────────────────────────────────────────────
def strip_stats(dep: np.ndarray, part: np.ndarray, lat: np.ndarray) -> dict:
    m = np.abs(lat) <= STAT_LAT
    w = np.cos(np.deg2rad(lat[m]))[:, None] * np.ones((1, dep.shape[1]))
    d = dep[m]
    p = part[m]
    w = w / w.sum()

    def rms(x):
        return float(np.sqrt((w * x**2).sum()))

    dm, pm = (w * d).sum(), (w * p).sum()
    cov = (w * (d - dm) * (p - pm)).sum()
    r = float(cov / np.sqrt((w * (d - dm) ** 2).sum() * (w * (p - pm) ** 2).sum() + 1e-30))
    share = 1.0 - rms(d - p) ** 2 / (rms(d) ** 2 + 1e-30)
    return {"rms_departure": rms(d), "rms_part": rms(p), "r": r, "share": float(share)}


def shift_p(dep: np.ndarray, part: np.ndarray, lat: np.ndarray) -> float:
    """One-sided p that `part` matches `dep` (pattern r over the statistics band) better than chance: its r against
    `dep` circularly shifted by every other longitude step. When the two fields are unrelated every shift is
    exchangeable with the true alignment, and each field keeps its own spatial autocorrelation; synthetic nulls of
    smooth random strip fields held 4-5 % at the 5 % level."""
    r0 = strip_stats(dep, part, lat)["r"]
    rn = np.array([strip_stats(np.roll(dep, k, axis=1), part, lat)["r"] for k in range(1, dep.shape[1])])
    return float((1 + (rn >= r0).sum()) / (1 + rn.size))


def r2_pvalues(clim: xr.Dataset, month: int):
    """Masks (ENSO part, Indian+Atlantic part) of the grid points where those predictors are significant in the
    month's regression: nested F-tests from the reference's in-sample R² (ENSO only, full), Benjamini-Hochberg over
    both maps. The 90 training months (±1 month x 30 years) are not independent - the months of a year share their
    anomalies - so the residual degrees of freedom are N_EFF - 7: N_EFF = 62 is the smallest effective sample, over
    months and predictor groups, for which this test reproduces the whole-year permutation test of the band-mean W
    (walker_basins.perm_pvalues), with which it then agrees at 89-99 % of longitudes. The χ reference keeps no
    training anomalies that could be permuted directly."""
    r2e = clim.r2.sel(month=month, stage="enso").values.astype(float)
    r2f = clim.r2.sel(month=month, stage="full").values.astype(float)
    df2 = N_EFF - len(clim.pred) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        den = (1 - r2f) / df2
        p = np.stack([stats.f.sf(r2e / 2 / den, 2, df2), stats.f.sf((r2f - r2e) / 4 / den, 4, df2)])
    sig = benjamini_hochberg(p, FDR_Q)
    return sig[0], sig[1]


# ── figure ────────────────────────────────────────────────────────────────────
def render(out: Path, weeks: list, month_sst: dict, gill: dict, lat, lon, sig_share) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.gridspec import GridSpec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    nrow = len(weeks) + 1
    fig = plt.figure(figsize=(13.4, 0.98 * nrow + 2.15))
    gs = GridSpec(
        nrow,
        4,
        hspace=0.95,
        wspace=0.05,
        left=0.03,
        right=0.99,
        top=1 - 1.15 / fig.get_figheight(),
        bottom=0.72 / fig.get_figheight(),
    )
    pc = ccrs.PlateCarree(central_longitude=180)
    lev = np.array(VP_LEVELS, float)
    norm = BoundaryNorm(lev, VP_CMAP.N, extend="both")
    cols = [
        "Observed χ200 anomaly",
        "Pacific part (ENSO regression)",
        "Departure = observed − Pacific part",
        "Indian + Atlantic SST part (ENSO removed)",
    ]  # the longer title ran off the right edge

    def panel(
        r,
        c,
        field,
        title=None,
        note=None,
        glat=lat,
        glon=lon,
        rowlab=None,
        cmap=VP_CMAP,
        levels=lev,
        nrm=norm,
        scale=1e6,
        ns=None,
    ):
        ax = fig.add_subplot(gs[r, c], projection=pc)
        f = np.concatenate([field, field[:, :1]], 1)
        gl_ = np.concatenate([glon, [glon[0] + 360]])
        cf = ax.contourf(
            gl_, glat, f / scale, levels=levels, cmap=cmap, norm=nrm, extend="both", transform=ccrs.PlateCarree()
        )
        if ns is not None and ns.any():  # regression not significant there: the part is 0, hatched
            ax.contourf(
                gl_,
                glat,
                np.concatenate([ns, ns[:, :1]], 1).astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["////"],
                transform=ccrs.PlateCarree(),
            )
        ax.coastlines(lw=0.45, color="#444")
        ax.add_feature(cfeature.LAND, facecolor="#eeeae2", zorder=0)
        ax.set_extent([0, 359.99, -30, 30], crs=ccrs.PlateCarree())
        if title:
            ax.set_title(title, fontsize=7.6, loc="left", fontweight="bold", pad=2.5)
        gl = ax.gridlines(
            draw_labels=(r == nrow - 1),
            lw=0.25,
            color="#bbb",
            x_inline=False,
            y_inline=False,
            xlocs=range(0, 360, 60),
            ylocs=[-20, 0, 20],
        )
        gl.top_labels = gl.right_labels = gl.left_labels = False
        gl.xlabel_style = {"size": 6}
        if note and r == nrow - 1:  # the last row carries the longitude labels: note goes inside
            ax.text(
                0.006,
                0.06,
                note,
                transform=ax.transAxes,
                fontsize=6.4,
                va="bottom",
                color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2),
                zorder=6,
            )
        elif note:
            ax.text(0.0, -0.07, note, transform=ax.transAxes, fontsize=6.6, va="top", color=INK)
        if rowlab:
            ax.text(
                -0.012,
                0.5,
                rowlab,
                transform=ax.transAxes,
                rotation=90,
                ha="right",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=INK,
            )
        for y in (-STAT_LAT, STAT_LAT):
            ax.plot([0, 359.99], [y, y], color="#666", lw=0.4, ls=":", transform=ccrs.PlateCarree())
        return cf

    plt.rcParams["hatch.color"] = "#a7a299"
    plt.rcParams["hatch.linewidth"] = 0.4

    def match_note(s):
        if s["sig"]:
            return f"pattern r {s['r']:+.2f} (p {s['p']:.3f}), explains {100 * s['share']:.0f}%"
        return f"no significant match (p {s['p']:.2f})"

    for r, wk in enumerate(weeks):
        lab = f"{wk['start']:%d %b}–{wk['end']:%d %b}" + ("\n(28-day mean)" if wk["days"] > 8 else "")
        s = wk["stats_reg"]
        ix, sg = wk["idx"], wk["idx_sig"]
        cf = panel(r, 0, wk["obs"], cols[0] if r == 0 else None, note=f"{wk['n']} analyses", rowlab=lab)
        panel(
            r,
            1,
            wk["pac"],
            cols[1] if r == 0 else None,
            note=f"Niño-3.4 {ix['n34']:+.2f} °C ({sg['n34']:+.1f}σ), 3 months ago {ix['n34_lag3']:+.2f} °C",
            ns=wk["ns_pac"],
        )
        panel(
            r,
            2,
            wk["dep"],
            cols[2] if r == 0 else None,
            note=f"RMS {s['rms_departure'] / 1e6:.1f}  ·  vs the Indian+Atlantic part: {match_note(s)}",
        )
        panel(
            r,
            3,
            wk["ind_atl"],
            cols[3] if r == 0 else None,
            note=f"IOD {sg['dmi']:+.1f}σ · IO basin {sg['iob']:+.1f}σ · ATL3 {sg['atl3']:+.1f}σ · TNA {sg['tna']:+.1f}σ  ·  RMS {s['rms_part'] / 1e6:.1f}",
            ns=wk["ns_ind_atl"],
        )
    # Gill row for the 28-day SST
    r = nrow - 1
    g = gill
    sg = g["stats"]
    sp = g["pac_stats"]
    cfs = panel(
        r,
        0,
        month_sst["ssta"],
        f"SST anomaly, OISST {month_sst['start']:%d %b}–{month_sst['end']:%d %b}, detrended (°C)",
        glat=LAT2,
        glon=LON2,
        rowlab="Gill (1980)\nidealised",
        cmap="RdBu_r",
        levels=np.arange(-2.4, 2.41, 0.3),
        nrm=None,
        scale=1.0,
    )
    panel(
        r,
        1,
        g["pacific"],
        "Gill, Pacific SST only → upper-level χ",
        note=f"vs observed 28-d χ200: {match_note(sp)}",
        glat=LAT2,
        glon=LON2,
    )
    panel(
        r,
        2,
        g["ind_atl"],
        "Gill, Indian + Atlantic SST (ENSO removed) → χ",
        note=f"vs the 28-d departure: {match_note(sg)}",
        glat=LAT2,
        glon=LON2,
    )
    panel(
        r,
        3,
        weeks[-1]["remainder"],
        "28-d remainder: departure − Indian/Atlantic part",
        note="what neither regression part accounts for",
    )
    # colour bars + text
    H = fig.get_figheight()
    yb = 0.36 / H
    cax = fig.add_axes([0.30, yb, 0.36, 0.09 / H])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=lev)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("χ200 anomaly, 10⁶ m² s⁻¹  (green = upper-level divergence, enhanced deep convection)", fontsize=7)
    cax2 = fig.add_axes([0.05, yb, 0.16, 0.09 / H])
    cb2 = fig.colorbar(cfs, cax=cax2, orientation="horizontal")
    cb2.ax.tick_params(labelsize=6.5)
    cb2.set_label("SST anomaly, °C", fontsize=7)
    fig.text(
        0.99,
        0.05 / H,
        "AIFS-ENS 0-h analyses · ERA5 · ERSSTv6 · OISST v2.1 · Gill (1980)",
        fontsize=6.5,
        color=MUTED,
        ha="right",
    )
    fig.suptitle(
        "Tropical overturning: weekly departure from Pacific forcing — and how much of it the Indian and Atlantic Oceans can claim",
        fontsize=13,
        fontweight="bold",
        x=0.03,
        ha="left",
        y=1 - 0.12 / H,
    )
    import textwrap

    blurb = (
        "7-day means of the AIFS-ENS 0-h analyses against the ERA5 1991–2020 harmonic climatology, 1991–2020 trend removed. Pacific part: the ERA5 + ERSSTv6 1991–2020 regression of χ200 on Niño-3.4 "
        "(now and 3 months ago), fed with each week's detrended OISST index; Indian + Atlantic part: its ENSO-independent terms (IOD, Indian Ocean basin, ATL3, tropical North Atlantic as residuals after ENSO). "
        f"Each part is 0 (hatched) where its predictors are not significant (F-test, false discovery rate 10 %): this month ENSO is significant over {100 * sig_share[0]:.0f}% and Indian+Atlantic over {100 * sig_share[1]:.0f}% of "
        f"{STAT_LAT:.0f}°S–{STAT_LAT:.0f}°N (dotted). A pattern r over that band is quoted only when it beats the field shifted to every other longitude (one-sided, false discovery rate 10 % over the seven comparisons); "
        "share = 1 − RMS(departure − part)² / RMS(departure)². A linear regression under-predicts a strong El Niño: the Gill row (total-SST convective mask) shows how much of the departure that can be."
    )
    fig.text(
        0.03, 1 - 0.32 / H, "\n".join(textwrap.wrap(blurb, 236)), fontsize=7.2, color=MUTED, va="top", linespacing=1.35
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105, facecolor="white")
    plt.close(fig)
    print(f"saved {out}", flush=True)


def main() -> int:
    t0 = time.time()
    for p in (CLIM, WREF):
        if not p.exists():
            raise SystemExit(f"reference missing: {p}")
    clim = xr.open_dataset(CLIM).load()
    wref = xr.open_dataset(WREF).load()
    h = wc.load_history(HIST)
    if h is None:
        raise SystemExit("no χ history (frames branch assets/sst/anim/walker/walker_chi_history.nc or the seed)")
    h = h.sel(latitude=clim.latitude, longitude=clim.longitude, method="nearest")
    lat, lon = clim.latitude.values, clim.longitude.values
    an = chi_anomalies(h, clim)
    t = pd.DatetimeIndex(an.time.values)
    tlast = t[-1]
    print(f"  χ history {h.time.size} analyses to {tlast:%Y-%m-%d %HZ}", flush=True)
    weeks, masks = [], {}
    spans = [(tlast - pd.Timedelta(days=7 * (k + 1)), tlast - pd.Timedelta(days=7 * k)) for k in range(NWEEKS)][::-1]
    spans.append((tlast - pd.Timedelta(days=7 * NWEEKS), tlast))
    for a, b in spans:
        sel = (t > a) & (t <= b)
        if sel.sum() < 6:
            print(f"  window {a:%m-%d}–{b:%m-%d}: only {int(sel.sum())} analyses, skipped", flush=True)
            continue
        start, end = (a + pd.Timedelta(days=1)).normalize(), b.normalize()
        ssta, mid = ssta_window(wref, start, end)
        if ssta is None:
            print(f"  window {a:%m-%d}–{b:%m-%d}: no OISST, skipped", flush=True)
            continue
        month = int(mid.month)
        idx = indices_from(ssta, wref)
        try:
            idx["n34_lag3"] = n34_lag3(wref, end)
        except Exception as ex:  # noqa: BLE001
            raise SystemExit(f"lagged Niño-3.4 unavailable ({ex}); nothing written") from ex
        sd = {k: float(wref.pred_sd.sel(month=month, pred=k)) for k in wref.pred.values}
        idx_sig = {k: idx[k] / sd[k] for k in idx}
        ensov = np.array([idx["n34"], idx["n34_lag3"]])
        ops = wref.resid_ops.sel(month=month).values
        resid = {k: idx[k] - float(ops[i] @ ensov) for i, k in enumerate(("dmi", "iob", "atl3", "tna"))}
        coef = clim.coef.sel(month=month)
        part = {
            k: coef.sel(pred=k).values * (ensov[0] if k == "n34" else ensov[1] if k == "n34_lag3" else resid[k])
            for k in clim.pred.values
        }
        if month not in masks:
            masks[month] = r2_pvalues(clim, month)
        sig_e, sig_b = masks[month]  # a part is 0 where its predictors are not significant
        pac = np.where(sig_e, part["n34"] + part["n34_lag3"], 0.0)
        ind_atl = np.where(sig_b, part["dmi"] + part["iob"] + part["atl3"] + part["tna"], 0.0)
        obs = an.sel(time=sel).mean("time").values
        dep = obs - pac
        weeks.append(
            {
                "start": start,
                "end": end,
                "days": int((end - start).days) + 1,
                "n": int(sel.sum()),
                "month": month,
                "idx": idx,
                "idx_sig": idx_sig,
                "ssta": ssta,
                "obs": obs,
                "pac": pac,
                "dep": dep,
                "ind_atl": ind_atl,
                "ns_pac": ~sig_e,
                "ns_ind_atl": ~sig_b,
                "remainder": dep - ind_atl,
                "stats_reg": strip_stats(dep, ind_atl, lat),
            }
        )
        s = weeks[-1]["stats_reg"]
        print(
            f"  {start:%m-%d}–{end:%m-%d} n={int(sel.sum())} Niño3.4 {idx['n34']:+.2f} IOD {idx_sig['dmi']:+.1f}σ  departure RMS {s['rms_departure'] / 1e6:.2f}  Ind+Atl share {100 * s['share']:.0f}% r {s['r']:+.2f}",
            flush=True,
        )
    if not weeks:
        raise SystemExit("no complete weeks in the χ history")
    # Gill on the 28-day SST
    last = weeks[-1]
    month = last["month"]
    pat = wref.enso_pattern.sel(month=month).interp(lat=LAT2, lon_e=LON2, kwargs={"fill_value": None}).values
    clim2 = wref.sst_clim.sel(month=month).interp(lat=LAT2, lon_e=LON2, kwargs={"fill_value": None}).values
    basins = json.loads(wref.attrs["basin_lon"])
    scale = gill_chi_scale(wref, clim, month, lat, lon, basins)
    print(f"  Gill χ amplitude (ENSO-anchored, month {month}): {scale:.3g} m² s⁻¹ per unit", flush=True)
    ssta = last["ssta"]
    ssta_res = ssta - pat * last["idx"]["n34"]
    gpac = gill_chi200(heating_from_ssta(np.where(basin_mask(LON2, *basins["pacific"])[None], ssta, 0.0), clim2), scale)
    gia = gill_chi200(
        heating_from_ssta(
            np.where(
                basin_mask(LON2, *basins["indian"])[None] | basin_mask(LON2, *basins["atlantic"])[None], ssta_res, 0.0
            ),
            clim2,
        ),
        scale,
    )
    # compare Gill (Gill grid) with the 28-d departure (χ grid) on the Gill grid
    dep_g = (
        xr.DataArray(last["dep"], coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
        .interp(lat=LAT2, lon=LON2, kwargs={"fill_value": None})
        .values
    )
    obs_g = np.nan_to_num(
        dep_g
        + xr.DataArray(last["pac"], coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
        .interp(lat=LAT2, lon=LON2, kwargs={"fill_value": None})
        .values
    )
    dep_g = np.nan_to_num(dep_g)
    gill = {
        "pacific": gpac,
        "ind_atl": gia,
        "stats": strip_stats(dep_g, gia, LAT2),
        "pac_stats": strip_stats(obs_g, gpac, LAT2),
    }
    # one family: each week's basin part vs its departure, and the two Gill comparisons
    pv = [shift_p(w["dep"], w["ind_atl"], lat) for w in weeks] + [
        shift_p(dep_g, gia, LAT2),
        shift_p(obs_g, gpac, LAT2),
    ]
    sig = benjamini_hochberg(np.array(pv), FDR_Q) & (np.array(pv) < 0.05)
    for s, p, ok in zip([w["stats_reg"] for w in weeks] + [gill["stats"], gill["pac_stats"]], pv, sig):
        s["p"], s["sig"] = p, bool(ok)
    print(
        f"  Gill: Pacific-only vs the observed 28-d χ200 r {gill['pac_stats']['r']:+.2f} (p {gill['pac_stats']['p']:.3f}); "
        f"Indian+Atlantic vs the departure r {gill['stats']['r']:+.2f} (p {gill['stats']['p']:.3f})",
        flush=True,
    )
    m15 = np.abs(lat) <= STAT_LAT
    wt = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    sig_e, sig_b = masks[month]
    sig_share = tuple(float((wt * s_)[m15].sum() / wt[m15].sum()) for s_ in (sig_e, sig_b))
    render(
        ASSETS / "walker_chi_weekly.webp",
        weeks,
        {"ssta": ssta, "start": last["start"], "end": last["end"]},
        gill,
        lat,
        lon,
        sig_share,
    )

    def match(s):  # a match statistic is published only when the match is significant
        return {"p": round(s["p"], 4), "significant": s["sig"]} | (
            {"r": round(s["r"], 3), "share": round(s["share"], 3)} if s["sig"] else {}
        )

    doc = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "last_analysis": tlast.strftime("%Y-%m-%dT%HZ"),
        "stat_band_deg": STAT_LAT,
        "weeks": [
            {
                "start": w["start"].strftime("%Y-%m-%d"),
                "end": w["end"].strftime("%Y-%m-%d"),
                "n_analyses": w["n"],
                "month": w["month"],
                "indices_degC": {k: round(v, 3) for k, v in w["idx"].items()},
                "indices_sigma": {k: round(v, 2) for k, v in w["idx_sig"].items()},
                "rms_departure_1e6": round(w["stats_reg"]["rms_departure"] / 1e6, 3),
                "rms_indian_atlantic_1e6": round(w["stats_reg"]["rms_part"] / 1e6, 3),
                "indian_atlantic_match": match(w["stats_reg"]),
            }
            for w in weeks
        ],
        "gill_28d": {
            "pacific_vs_observed": match(gill["pac_stats"]),
            "indian_atlantic_vs_departure": match(gill["stats"]),
        },
        "significant_share_strip": {"enso": round(sig_share[0], 3), "indian_atlantic": round(sig_share[1], 3)},
        "tests": {
            "regression": f"nested F-test from the reference R², {N_EFF} effective samples, BH q {FDR_Q} over the strip",
            "match": f"one-sided circular longitude-shift test of the pattern r, BH q {FDR_Q} and p < 0.05 over the "
            f"{len(pv)} comparisons",
        },
        "gill_chi_scale": scale,
    }
    (ASSETS / "data").mkdir(parents=True, exist_ok=True)
    (ASSETS / "data" / "walker_chi_weekly.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote walker_chi_weekly.json in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
