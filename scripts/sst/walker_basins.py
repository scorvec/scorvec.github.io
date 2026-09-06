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
             and Atlantic (ATL3, tropical North Atlantic) indices as RESIDUALS after ENSO.
  Leg B      the Gill (1980) linear response to the SST anomaly of each basin, with the ENSO
             regression pattern removed from the Indian and Atlantic fields; one amplitude,
             calibrated on ENSO in the builder, applies to every basin.

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oisst9120                                                     # noqa: E402
from gill_model import LAT2, LON2, BAND, band_u, basin_mask, gill_response, heating_from_ssta   # noqa: E402

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
ASSETS = SITE / "assets" / "sst"
REF = HERE.parents[1] / "scripts" / "mjo" / "data" / "reference"
CLIM = REF / "walker_basins_clim.nc"
HIST = REF / "walker_ud_history.nc"
PAC_BOX = (140.0, 200.0)                                            # the walker.py Pacific-cell box, in longitude
INK, MUTED, NAVY, RED, GREEN, ORANGE = "#1a1a1a", "#8a8680", "#1b365d", "#b4453c", "#2b7a3d", "#d0801a"
GROUPS = {"enso": ["n34", "n34_lag3"], "indian": ["dmi", "iob"], "atlantic": ["atl3", "tna"]}
LABEL = {"n34": "Niño-3.4", "n34_lag3": "Niño-3.4, 3 months ago", "dmi": "IOD (DMI)", "iob": "Indian Ocean basin", "atl3": "Atlantic Niño (ATL3)", "tna": "Tropical N. Atlantic"}


# ── SST ──────────────────────────────────────────────────────────────────────
def current_ssta(ref: xr.Dataset, days: int = 30):
    """(detrended SSTA on the Gill grid, its 30-day window, raw anomaly on the same grid)."""
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    year = now.year
    p = oisst9120.ensure_mean(year)
    ds = xr.open_dataset(p)
    sst = ds["sst"].sel(lat=slice(-31, 31))
    t = pd.DatetimeIndex(sst.time.values)
    if len(t) < days:                                                # early January: reach into last year
        prev = xr.open_dataset(oisst9120.ensure_mean(year - 1))["sst"].sel(lat=slice(-31, 31))
        sst = xr.concat([prev.isel(time=slice(-days, None)), sst], dim="time"); t = pd.DatetimeIndex(sst.time.values)
    win = sst.isel(time=slice(-days, None)).load()
    an = oisst9120.anom(win).mean("time")
    an2 = an.coarsen(lat=8, lon=8, boundary="trim").mean()
    an2 = an2.interp(lat=LAT2, lon=LON2, kwargs={"fill_value": None})
    mid = pd.DatetimeIndex(win.time.values)[len(win.time) // 2]
    m = mid.month
    slope = ref.sst_slope.sel(month=m).interp(lat=LAT2, lon_e=LON2, kwargs={"fill_value": None}).values
    yr = mid.year + (mid.dayofyear - 1) / 365.25
    det = an2.values - slope * (yr - 2005.5)                        # remove the 1991–2020 trend, extrapolated
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


# ── observed Walker ──────────────────────────────────────────────────────────
def observed_w(ref: xr.Dataset):
    """W anomaly (7-day and 30-day trailing means) on LON2 from the AIFS analyses, and the dates."""
    h = xr.open_dataset(HIST)["ud"]
    t = pd.DatetimeIndex(h.time.values)
    lon = h.longitude.values
    def band(lev):
        a = h.sel(level=lev, method="nearest").values           # (time, lon)
        return np.stack([np.interp(LON2, np.concatenate([lon, [lon[0] + 360]]), np.concatenate([row, [row[0]]])) for row in a])
    w = band(200) - band(850)
    doy = t.dayofyear.values; ang = 2 * np.pi * doy / 365.25
    B = np.stack([np.ones_like(ang), np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)], -1)
    clim = B @ ref.ud200_coef.values - B @ ref.ud850_coef.values
    # the same detrending the regression saw: subtract the extrapolated 1991–2020 trend of W for the month
    yr = t.year.values + (doy - 1) / 365.25
    trend = np.stack([ref.w_slope.sel(month=m).values * (y - 2005.5) for m, y in zip(t.month.values, yr)])
    wa = w - clim - trend
    def trailing(n):
        sel = t > t[-1] - pd.Timedelta(days=n)
        return np.nanmean(wa[sel], axis=0), int(sel.sum())
    w7, n7 = trailing(7); w30, n30 = trailing(30)
    return w7, w30, n7, n30, t[-1]


# ── figure ───────────────────────────────────────────────────────────────────
def render(out: Path, ref, ssta, ssta_res, window, month, idx, idx_sig, w7, w30, n7, n30, tlast, reg, gill, gmaps, r2) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import textwrap
    fig = plt.figure(figsize=(13.4, 12.4))
    gs = GridSpec(6, 2, height_ratios=[0.62, 0.14, 1.2, 1.0, 0.62, 0.55], hspace=0.5, wspace=0.1, left=0.055, right=0.985, top=0.925, bottom=0.03)
    pc = ccrs.PlateCarree(central_longitude=180)
    boxes = json.loads(ref.attrs["boxes"]); basins = json.loads(ref.attrs["basin_lon"])
    lev = np.arange(-2.4, 2.41, 0.3)

    def sst_panel(k, field, title):
        ax = fig.add_subplot(gs[0, k], projection=pc)
        cf = ax.contourf(LON2, LAT2, field, levels=lev, cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree())
        ax.coastlines(lw=0.5, color="#555"); ax.add_feature(cfeature.LAND, facecolor="#ece9e1", zorder=2)
        for name, ((la0, la1), (lo0, lo1)) in boxes.items():
            colr = {"n34": RED, "dmi_w": GREEN, "dmi_e": GREEN, "iob": GREEN, "atl3": ORANGE, "tna": ORANGE}[name]
            ax.add_patch(plt.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False, ec=colr, lw=1.1, ls=("--" if name in ("iob",) else "-"), transform=ccrs.PlateCarree(), zorder=5))
        for b, (lo0, lo1) in basins.items():
            ax.plot([lo0, lo0], [-30, 30], color="#333", lw=0.6, ls=":", transform=ccrs.PlateCarree(), zorder=5)
        ax.set_extent([0, 359.99, -30, 30], crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=9.5, loc="left", fontweight="bold")
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="#bbb", x_inline=False, y_inline=False); gl.top_labels = gl.right_labels = False; gl.left_labels = (k == 0)
        gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
        return cf
    cf = sst_panel(0, ssta, f"SST anomaly, OISST {window[0]:%d %b}–{window[1]:%d %b} (1991–2020 base, trend removed)")
    sst_panel(1, ssta_res, "The same field with the ENSO regression pattern removed")
    b0 = gs[1, 0].get_position(fig)
    cax = fig.add_axes([0.40, b0.y0 + 0.55 * b0.height, 0.20, 0.006]); cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=6.5); cb.set_label("°C   ·   boxes: red Niño-3.4, green IOD poles and basin (dashed), orange ATL3 and TNA", fontsize=7)

    # W profile
    ax = fig.add_subplot(gs[2, :])
    for b, (lo0, lo1) in basins.items():
        ax.axvspan(lo0, min(lo1, 360), color={"indian": GREEN, "pacific": RED, "atlantic": ORANGE}[b], alpha=0.04, lw=0)
        if lo1 > 360:
            ax.axvspan(0, lo1 - 360, color=ORANGE, alpha=0.04, lw=0)
    ax.axhline(0, color="#555", lw=0.8)
    ax.plot(LON2, w7, color="#222", lw=1.1, alpha=0.8, label=f"observed, last {n7} analyses (7 d)")
    ax.plot(LON2, w30, color="#000", lw=2.4, label=f"observed, last {n30} analyses (30 d)")
    ax.plot(LON2, reg["total"], color=NAVY, lw=2.0, ls="--", label="regression total")
    ax.plot(LON2, reg["enso"], color=RED, lw=1.8, label="ENSO part (Niño-3.4 now and 3 months ago)")
    ax.plot(LON2, reg["indian"], color=GREEN, lw=1.8, label="Indian Ocean part, ENSO removed")
    ax.plot(LON2, reg["atlantic"], color=ORANGE, lw=1.8, label="Atlantic part, ENSO removed")
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 30)); ax.set_xticklabels([f"{x}°E" if x <= 180 else f"{360 - x}°W" for x in range(0, 361, 30)], fontsize=7.5)
    ax.set_ylabel("W = u_D(200) − u_D(850), m/s", fontsize=8); ax.tick_params(labelsize=7.5); ax.grid(True, alpha=0.2)
    ax.set_title("Zonal overturning anomaly along the equator: observed vs the ENSO-first regression (+ = upper-level eastward, lower-level westward divergent flow)", fontsize=9.2, loc="left", fontweight="bold")
    ax.legend(fontsize=7.2, ncol=3, frameon=False, loc="upper left")

    # Gill profiles + skill
    ax = fig.add_subplot(gs[3, 0])
    ax.axhline(0, color="#555", lw=0.8)
    for key, colr, lab in (("pacific", RED, "Pacific SST (full)"), ("indian", GREEN, "Indian Ocean, ENSO removed"), ("atlantic", ORANGE, "Atlantic, ENSO removed")):
        ax.plot(LON2, gill[key], color=colr, lw=1.9, label=f"Gill: {lab}")
    ax.plot(LON2, gill["total"], color=NAVY, lw=1.4, ls="--", label="Gill: all three")
    ax.plot(LON2, reg["indian"], color=GREEN, lw=1.0, ls=":", label="regression Indian (for comparison)")
    ax.plot(LON2, reg["atlantic"], color=ORANGE, lw=1.0, ls=":", label="regression Atlantic")
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 60)); ax.set_xticklabels([f"{x}°E" if x <= 180 else f"{360 - x}°W" for x in range(0, 361, 60)], fontsize=7.5)
    ax.set_ylabel("W-equivalent, m/s", fontsize=8); ax.tick_params(labelsize=7.5); ax.grid(True, alpha=0.2)
    ax.set_title("Gill (1980) response to each basin's SST anomaly, amplitude calibrated on ENSO", fontsize=9.2, loc="left", fontweight="bold")
    ax.legend(fontsize=6.8, ncol=2, frameon=False)
    ax = fig.add_subplot(gs[3, 1])
    mm = np.arange(1, 13); wdt = 0.2
    for k, (col, lab, c) in enumerate(((0, "ENSO only", RED), (1, "+ Indian", GREEN), (2, "+ Atlantic", ORANGE), (3, "all", NAVY))):
        ax.bar(mm + (k - 1.5) * wdt, r2[:, col], width=wdt, color=c, alpha=0.85, label=lab)
    ax.axvspan(month - 0.5, month + 0.5, color="#000", alpha=0.06, lw=0)
    ax.set_xticks(mm); ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=7.5); ax.tick_params(labelsize=7.5)
    ax.set_ylabel("R² of monthly W, all longitudes", fontsize=8); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.2)
    ax.set_title("Variance of monthly W explained, ERA5 1991–2020 (this month shaded)", fontsize=9.2, loc="left", fontweight="bold")
    ax.legend(fontsize=7.2, ncol=4, frameon=False, loc="upper left")

    # Gill maps
    for k, (key, title) in enumerate((("indian", "Gill response to the Indian Ocean anomaly (ENSO removed): low-level wind and pressure"), ("atlantic", "Gill response to the Atlantic anomaly (ENSO removed)"))):
        ax = fig.add_subplot(gs[4, k], projection=pc)
        u, v, p = gmaps[key]
        pm = np.nanmax(np.abs(p)) or 1.0
        cf2 = ax.contourf(LON2, LAT2, p, levels=np.linspace(-pm, pm, 12), cmap="PuOr", extend="both", transform=ccrs.PlateCarree())
        sk = 2
        ax.quiver(LON2[::sk], LAT2[::sk], u[::sk, ::sk], v[::sk, ::sk], transform=ccrs.PlateCarree(), scale=np.nanmax(np.hypot(u, v)) * 25 + 1e-9, width=0.0022, color="#222")
        ax.coastlines(lw=0.5, color="#555"); ax.set_extent([0, 359.99, -30, 30], crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=8.8, loc="left", fontweight="bold")
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="#bbb", x_inline=False, y_inline=False); gl.top_labels = gl.right_labels = False; gl.left_labels = (k == 0); gl.xlabel_style = gl.ylabel_style = {"size": 6.5}

    # summary text
    ax = fig.add_subplot(gs[5, :]); ax.axis("off")
    pac = (LON2 >= PAC_BOX[0]) & (LON2 <= PAC_BOX[1])
    def pm_(a): return float(np.nanmean(np.asarray(a)[pac]))
    lines = [f"Indices from the last 30 days, detrended, in σ of the month:  " + "   ".join(f"{LABEL[k]} {idx_sig[k]:+.1f}σ ({idx[k]:+.2f} °C)" for k in ("n34", "dmi", "iob", "atl3", "tna")),
             f"Pacific Walker box (140°E–160°W) W anomaly, m/s:  observed {pm_(w30):+.2f} (30 d) / {pm_(w7):+.2f} (7 d)   ·   regression total {pm_(reg['total']):+.2f} = ENSO {pm_(reg['enso']):+.2f} + Indian {pm_(reg['indian']):+.2f} + Atlantic {pm_(reg['atlantic']):+.2f}",
             f"Gill model over the same box:  Pacific {pm_(gill['pacific']):+.2f}   Indian (ENSO removed) {pm_(gill['indian']):+.2f}   Atlantic (ENSO removed) {pm_(gill['atlantic']):+.2f}   ·   negative = Pacific cell weakened",
             f"Read across the two legs: where the regression and Gill Indian/Atlantic curves agree in sign and size, the attribution is robust; where they differ, the statistical link is not a linear SST-forced one. R² this month: ENSO alone {r2[month - 1, 0]:.2f}, all basins {r2[month - 1, 3]:.2f}."]
    wrapped = []
    for ln in lines:
        wrapped += textwrap.wrap(ln, 175, subsequent_indent="      ")
    ax.text(0, 1, "\n".join(wrapped), transform=ax.transAxes, fontsize=8.1, va="top", color=INK, linespacing=1.5)
    fig.suptitle("Walker circulation by ocean basin — the Indian and Atlantic contributions with ENSO removed", fontsize=13.5, fontweight="bold", x=0.055, ha="left", y=0.985)
    fig.text(0.055, 0.967, "\n".join(textwrap.wrap("Observed: AIFS-ENS 0-h analyses (divergent wind at 200 minus 850 hPa, 5°S–5°N) as an anomaly against ERA5 1991–2020, latest analysis "
             f"{tlast:%Y-%m-%d %HZ}. Regression: ERA5 + ERSST 1991–2020, Niño-3.4 (now and 3 months ago) first, then the Indian and Atlantic indices as residuals after ENSO — so their parts are ENSO-independent by construction. "
             "Gill: linear equatorial β-plane response to heating ∝ SST anomaly over water warmer than 27 °C; the ENSO regression pattern is subtracted from the Indian and Atlantic SST before forcing; one amplitude, calibrated on ENSO, for all basins.", 205)),
             fontsize=8, color=MUTED, va="top", linespacing=1.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    print(f"saved {out}", flush=True)


def main() -> int:
    t0 = time.time()
    if not CLIM.exists():
        raise SystemExit("reference missing: run scripts/mjo/src/build_walker_basins_clim.py")
    ref = xr.open_dataset(CLIM).load()
    ssta, window, ssta_raw, month = current_ssta(ref)
    idx = indices_from(ssta, ref, month)
    sd = {k: float(ref.pred_sd.sel(month=month, pred=k)) for k in ref.pred.values}
    idx_sig = {k: idx[k] / sd[k] for k in idx}
    # the 3-month-lag Niño-3.4 from ERSST (the training source), detrended the same way
    n34_lag = None
    try:
        e = xr.open_dataset(HERE / "data" / "ersst_v5_mnmean.nc")["sst"]
        tm = pd.Timestamp(window[1]) - pd.DateOffset(months=3)
        fld = e.sel(time=f"{tm.year}-{tm.month:02d}").isel(time=0).values.astype(float)
        m3 = tm.month
        an3 = fld - ref.sst_clim.sel(month=m3).values - ref.sst_slope.sel(month=m3).values * (tm.year + (tm.month - 0.5) / 12 - 2005.5)
        boxes = json.loads(ref.attrs["boxes"]); (la0, la1), (lo0, lo1) = boxes["n34"]
        lat, lon = e.lat.values, e.lon.values
        msk = ((lat >= la0) & (lat <= la1))[:, None] & ((lon >= lo0) & (lon <= lo1))[None]
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)) * msk * np.isfinite(an3)
        n34_lag = float((np.nan_to_num(an3) * w).sum() / w.sum())
    except Exception as ex:                                              # noqa: BLE001
        print(f"  lagged Niño-3.4 unavailable ({str(ex)[:60]}); using the current value", flush=True)
    if n34_lag is None:
        n34_lag = idx["n34"]
    idx["n34_lag3"] = n34_lag; idx_sig["n34_lag3"] = n34_lag / sd["n34_lag3"]

    # leg A: regression contributions (basin indices as residuals after ENSO, as trained)
    ensov = np.array([idx["n34"], idx["n34_lag3"]])
    ops = ref.resid_ops.sel(month=month).values                         # (basin_pred, enso_pred)
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
        gill[b] = -2.0 * alpha * band_u(u); gmaps[b] = (u, v, p)
    gill["total"] = gill["pacific"] + gill["indian"] + gill["atlantic"]

    w7, w30, n7, n30, tlast = observed_w(ref)
    r2 = ref.r2.values
    render(ASSETS / "walker_basins.webp", ref, ssta, ssta_res, window, month, idx, idx_sig, w7, w30, n7, n30, tlast, reg, gill, gmaps, r2)
    pac = (LON2 >= PAC_BOX[0]) & (LON2 <= PAC_BOX[1])
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "sst_window": [window[0].strftime("%Y-%m-%d"), window[1].strftime("%Y-%m-%d")], "month": int(month),
           "last_analysis": tlast.strftime("%Y-%m-%dT%HZ"), "indices_degC": {k: round(v, 3) for k, v in idx.items()}, "indices_sigma": {k: round(v, 2) for k, v in idx_sig.items()},
           "lon": LON2.tolist(), "observed_w7": np.round(w7, 3).tolist(), "observed_w30": np.round(w30, 3).tolist(),
           "regression": {k: np.round(v, 3).tolist() for k, v in reg.items()}, "gill": {k: np.round(v, 3).tolist() for k, v in gill.items()},
           "pacific_box": {"observed_30d": round(float(np.nanmean(w30[pac])), 3), "observed_7d": round(float(np.nanmean(w7[pac])), 3),
                            **{f"reg_{k}": round(float(np.nanmean(v[pac])), 3) for k, v in reg.items()}, **{f"gill_{k}": round(float(np.nanmean(v[pac])), 3) for k, v in gill.items()}},
           "r2_this_month": {"enso": round(float(r2[month - 1, 0]), 3), "full": round(float(r2[month - 1, 3]), 3)}, "gill_alpha": alpha}
    (ASSETS / "data").mkdir(parents=True, exist_ok=True)
    (ASSETS / "data" / "walker_basins.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote walker_basins.json in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
