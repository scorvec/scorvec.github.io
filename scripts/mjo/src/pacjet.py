#!/usr/bin/env python3
"""North Pacific jet monitor: extension / retraction / shift, member by member, with the
Himalayan mountain torque that may be driving it.

Data: the 200 hPa AIFS-ENS zonal wind already pulled for the RMM/AAM products (cf + pf, daily
steps to day 15) over 10–70°N, 100°E–120°W, interpolated to the ERA5 1.5° reference grid.
Reference: scripts/mjo/data/reference/pacjet_clim.nc from build_pacjet_clim.py (ERA5 1991–2020:
harmonic day-of-year climatology, σ(doy), Nov–Mar jet-regime EOFs, index climatologies) and
pacjet_lag.json (ERA5 lead–lag statistics of the jet indices after Himalayan torque events).

Indices, all per member and forecast day:
  extension  PC1 of the Nov–Mar 200 hPa anomaly EOFs (σ): + = jet extended east across the Pacific
  shift      PC2 (σ): + = jet displaced poleward
  exit       mean anomaly over the exit region 30–40°N 170°E–150°W, in σ of the day of year
  terminus   easternmost longitude of the ≥ 30 m/s core walking east from 130°E (NaN in summer)
Observed tail: ERA5 (ARCO, ~6-day lag) through era5_store, plus this and earlier cycles' 0-h
analyses — both kept in pacjet_history.nc so the tail survives between runs.

    python src/pacjet.py --date 20260906 --time 00 --out ../../assets/sst/pacjet.webp \
        --json ../../assets/sst/data/pacjet.json --torque ../../assets/sst/data/torque_ranges.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "era5"))
import store as ecmwf                                    # noqa: E402
from aam import DAILY_STEPS                              # noqa: E402
from build_pacjet_clim import (EXIT, HIM, SECTOR, ZDOM, ALASKA, GOA, CORE_MS, COLD, COMP_LAGS, G0, harm_eval, terminus, sel_box, to_lon360)  # noqa: E402

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
CLIM, LAG, HIST, COMP = REF / "pacjet_clim.nc", REF / "pacjet_lag.json", REF / "pacjet_history.nc", REF / "pacjet_composites.nc"
TAIL_DAYS = 45
INK, MUTED, NAVY, GOLD, BROWN = "#1a1a1a", "#8a8680", "#1b365d", "#b8860b", "#b4541f"


# ── model field ──────────────────────────────────────────────────────────────
def load_u200(date: str, time_: str, ref: xr.Dataset) -> xr.DataArray:
    """(number, day, latitude, longitude) 200 hPa u on the reference 1.5° sector grid."""
    cyc = ecmwf.Cycle(date, time_); steps = tuple(DAILY_STEPS)
    parts = []
    for typ in ("cf", "pf"):
        p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "u", "pl", ecmwf.LEVELS_RMM, steps))
        u = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}, chunks={"number": 1})["u"]
        if "isobaricInhPa" in u.dims:
            u = u.sel(isobaricInhPa=200)
        if "number" not in u.dims:
            u = u.expand_dims("number")
        u = to_lon360(u).sortby("latitude")
        u = u.sel(latitude=slice(SECTOR["lat"][0] - 2, SECTOR["lat"][1] + 2), longitude=slice(SECTOR["lon"][0] - 2, SECTOR["lon"][1] + 2))
        parts.append(u)
    u = xr.concat(parts, dim="number").assign_coords(number=np.arange(sum(p.sizes["number"] for p in parts)))
    hrs = (u.step / np.timedelta64(1, "h")).values.astype(int)
    u = u.isel(step=np.isin(hrs, DAILY_STEPS)).assign_coords(step=(hrs[np.isin(hrs, DAILY_STEPS)] // 24)).rename(step="day")
    u = u.interp(latitude=ref.latitude.values, longitude=ref.longitude.values, method="linear").load()
    return u.transpose("number", "day", "latitude", "longitude")


def load_z500(date: str, time_: str, ref: xr.Dataset) -> xr.DataArray:
    """(number, day, zlat, zlon) 500 hPa height (m) on the reference grid, or None if the field is not on disk."""
    cyc = ecmwf.Cycle(date, time_); steps = tuple(DAILY_STEPS)
    parts = []
    for typ in ("cf", "pf"):
        try:
            p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "z", "pl", (500,), steps))
        except Exception as e:                                                  # noqa: BLE001
            print(f"  z500 {typ} unavailable ({str(e)[:60]})"); return None
        z = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}, chunks={"number": 1})["z"]
        if "isobaricInhPa" in z.dims:
            z = z.sel(isobaricInhPa=500)
        if "number" not in z.dims:
            z = z.expand_dims("number")
        z = to_lon360(z).sortby("latitude").sel(latitude=slice(ZDOM["lat"][0] - 2, ZDOM["lat"][1] + 2), longitude=slice(ZDOM["lon"][0] - 2, ZDOM["lon"][1] + 2))
        parts.append(z)
    z = xr.concat(parts, dim="number").assign_coords(number=np.arange(sum(p.sizes["number"] for p in parts)))
    hrs = (z.step / np.timedelta64(1, "h")).values.astype(int)
    z = z.isel(step=np.isin(hrs, DAILY_STEPS)).assign_coords(step=(hrs[np.isin(hrs, DAILY_STEPS)] // 24)).rename(step="day")
    z = z.interp(latitude=ref.zlat.values, longitude=ref.zlon.values, method="linear").load() / G0
    return z.transpose("number", "day", "latitude", "longitude")


def z_indices(z: xr.DataArray, valid: pd.DatetimeIndex, ref: xr.Dataset) -> dict:
    """z (..., day, lat, lon) height (m) → anomaly (m) and the Alaska / GoA box indices (σ of the day of year)."""
    doy = valid.dayofyear.values
    anom = z.values - harm_eval(ref.z_coef.values, doy)
    lat, lon = ref.zlat.values, ref.zlon.values
    out = {"zanom": anom}
    for name, box in (("alaska", ALASKA), ("goa", GOA)):
        bm = ((lat >= box["lat"][0]) & (lat <= box["lat"][1]))[:, None] & ((lon >= box["lon"][0]) & (lon <= box["lon"][1]))[None]
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)) * bm
        v = np.einsum("...tij,ij->...t", np.nan_to_num(anom), w) / w.sum()
        out[name] = v / ref[f"{name}_sd"].values[doy - 1]
    return out


# ── indices ──────────────────────────────────────────────────────────────────
def indices(u: xr.DataArray, valid: pd.DatetimeIndex, ref: xr.Dataset) -> dict:
    """u (..., day, lat, lon) absolute → dict of arrays (..., day) in the reference units."""
    doy = valid.dayofyear.values
    clim = harm_eval(ref.u_coef.values, doy)                                     # (day, lat, lon)
    anom = u.values - clim
    proj = ref.proj.values                                                        # (mode, lat, lon)
    pcs = np.einsum("...tij,kij->...tk", np.nan_to_num(anom), proj)
    lat, lon = ref.latitude.values, ref.longitude.values
    em = ((lat >= EXIT["lat"][0]) & (lat <= EXIT["lat"][1]))[:, None] & ((lon >= EXIT["lon"][0]) & (lon <= EXIT["lon"][1]))[None]
    w = (np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))) * em
    exit_ms = np.einsum("...tij,ij->...t", np.nan_to_num(anom), w) / w.sum()
    exit_sd = ref.exit_sd.values[doy - 1]
    flat = u.values.reshape(-1, u.shape[-3], u.shape[-2], u.shape[-1]) if u.ndim == 4 else u.values[None]
    term = np.stack([terminus(xr.DataArray(f, coords={"time": valid, "latitude": lat, "longitude": lon}, dims=("time", "latitude", "longitude"))) for f in flat])
    term = term.reshape(*u.shape[:-2]) if u.ndim == 4 else term[0]
    term_anom = (term - harm_eval(ref.term_coef.values, doy)) / ref.term_sd.values[doy - 1]
    return {"extension": pcs[..., 0], "shift": pcs[..., 1], "exit": exit_ms / exit_sd, "exit_ms": exit_ms,
            "terminus": term, "terminus_anom": term_anom, "anom": anom}


# ── observed tail ────────────────────────────────────────────────────────────
COLS = ["extension", "shift", "exit", "terminus", "terminus_anom", "alaska", "goa"]


def observed_tail(init: pd.Timestamp, ref: xr.Dataset, analysis: dict | None) -> pd.DataFrame:
    """ERA5 daily (00Z+12Z mean) indices for the last TAIL_DAYS via era5_store (ARCO on miss),
    merged with archived 0-h analyses; persisted in pacjet_history.nc."""
    cols = COLS
    hist = pd.DataFrame(columns=cols + ["source"])
    if HIST.exists():
        h = xr.open_dataset(HIST).to_dataframe(); h.index = pd.to_datetime(h.index)
        for c in cols:
            if c not in h.columns:
                h[c] = np.nan
        hist = h
    if analysis is not None:
        row = {k: float(analysis.get(k, np.nan)) for k in cols}; row["source"] = "aifs_an"
        hist.loc[init.normalize()] = row
    try:
        import era5_store
        want = pd.date_range(init.normalize() - pd.Timedelta(days=TAIL_DAYS), init.normalize() - pd.Timedelta(days=5), freq="D")
        have = set(hist.index[(hist["source"] == "era5") & np.isfinite(hist["alaska"].astype(float))])
        todo = [d for d in want if d not in have]
        if todo:
            print(f"  ERA5 tail: {len(todo)} day(s) to compute …", flush=True)
        for d in todo:
            fields = []
            for hh in (0, 12):
                try:
                    f = era5_store.get_u(d + pd.Timedelta(hours=hh), [200]).sel(level=200)
                except Exception as e:                                          # noqa: BLE001
                    print(f"    {d:%Y-%m-%d} {hh:02d}Z unavailable ({str(e)[:50]})"); f = None; break
                if bool(np.isnan(f.values).all()):
                    f = None; break
                fields.append(to_lon360(f).sortby("latitude"))
            if not fields:
                continue
            u = xr.concat(fields, dim="t").mean("t")
            u = u.interp(latitude=ref.latitude.values, longitude=ref.longitude.values).expand_dims(day=[0])
            ix = indices(u, pd.DatetimeIndex([d]), ref)
            row = {k: float(np.ravel(ix[k])[0]) for k in cols if k in ix}
            zf = []
            for hh in (0, 12):
                try:
                    g = era5_store.get_z(d + pd.Timedelta(hours=hh), [500]).sel(level=500) / G0
                    if not bool(np.isnan(g.values).all()):
                        zf.append(to_lon360(g).sortby("latitude"))
                except Exception as e:                                          # noqa: BLE001
                    print(f"    z500 {d:%Y-%m-%d} {hh:02d}Z unavailable ({str(e)[:50]})")
            if zf:
                z = xr.concat(zf, dim="t").mean("t").interp(latitude=ref.zlat.values, longitude=ref.zlon.values).expand_dims(day=[0])
                zi = z_indices(z, pd.DatetimeIndex([d]), ref)
                row["alaska"] = float(np.ravel(zi["alaska"])[0]); row["goa"] = float(np.ravel(zi["goa"])[0])
            hist.loc[d] = {**{k: row.get(k, np.nan) for k in cols}, "source": "era5"}
    except Exception as e:                                                      # noqa: BLE001
        print(f"  ERA5 tail skipped ({str(e)[:80]})", flush=True)
    hist = hist.sort_index()
    hist = hist[~hist.index.duplicated(keep="last")]
    hist = hist[hist.index >= init.normalize() - pd.Timedelta(days=400)]
    HIST.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({k: ("time", hist[k].astype(float).values) for k in cols} | {"source": ("time", hist["source"].astype(str).values)},
                    coords={"time": hist.index.values})
    ds.to_netcdf(HIST)
    return hist


def torque_peak(torque: dict | None):
    """(peak date, value, σ) of the Himalayan torque series in this cycle, or None."""
    if not torque or "Himalaya/Tibet" not in torque.get("ranges", {}):
        return None
    tv = pd.to_datetime(torque["valid"]); tq = np.array(torque["ranges"]["Himalaya/Tibet"], float)
    s1 = torque.get("sd", {}).get("Himalaya/Tibet")
    k = int(np.argmax(tq))
    return tv[k], float(tq[k]), (float(tq[k] / s1) if s1 else None)


def season_for(init: pd.Timestamp) -> str:
    return "NDJFM" if init.month in COLD else "SON"


def composite_path(lag: dict | None, season: str, key: str):
    """(lags, mean, lo, hi) of the ERA5 composite for `key` after strong torque days, or None."""
    if not lag or season not in lag.get("seasons", {}) or key not in lag["seasons"][season]["composite"]:
        return None
    c = lag["seasons"][season]["composite"][key]
    return np.array(lag["lags"]), np.array([np.nan if v is None else v for v in c["mean"]], float), np.array(c["null_p05"], float), np.array(c["null_p95"], float)


def overlay_composite(ax, peak, lag, season, key, colr="#8b1a1a"):
    """Draw what ERA5 says usually follows a strong torque day, anchored on this cycle's torque peak."""
    cp = composite_path(lag, season, key)
    if peak is None or cp is None:
        return
    lags, m, lo, hi = cp
    sel = lags >= 0
    x = [peak[0] + pd.Timedelta(days=int(L)) for L in lags[sel]]
    ax.fill_between(x, lo[sel], hi[sel], color=colr, alpha=0.08, lw=0)
    ax.plot(x, m[sel], color=colr, lw=1.6, ls="--", label=f"ERA5 composite after ≥+1.5σ torque ({season})")


# ── figure ───────────────────────────────────────────────────────────────────
def render(init, valid, ix, u_mean_day0, ref, tail, torque, lag, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    nmem = ix["extension"].shape[0]
    fig = plt.figure(figsize=(13.4, 14.4))
    gs = GridSpec(6, 2, height_ratios=[1.35, 0.16, 0.55, 1, 1, 0.95], hspace=0.55, wspace=0.16, left=0.055, right=0.985, top=0.925, bottom=0.05)
    pc = ccrs.PlateCarree(central_longitude=180)
    lat, lon = ref.latitude.values, ref.longitude.values
    doy0 = valid[0].dayofyear
    clim0 = harm_eval(ref.u_coef.values, np.array([doy0]))[0]
    lev = np.arange(-30, 31, 5)
    # maps: analysis (left) and day 5–10 mean (right)
    for k, (title, field_abs) in enumerate((("0-h analysis", u_mean_day0), ("days 5–10, ensemble mean", None))):
        ax = fig.add_subplot(gs[0, k], projection=pc)
        if field_abs is None:
            sel = slice(5, 11)
            an = ix["anom"][:, sel].mean((0, 1)); ab = an + harm_eval(ref.u_coef.values, valid[sel].dayofyear.values).mean(0)
        else:
            an = field_abs - clim0; ab = field_abs
        cf = ax.contourf(lon, lat, an, levels=lev, cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree())
        cs = ax.contour(lon, lat, ab, levels=[30, 40, 50, 60, 70], colors="k", linewidths=[0.8, 1.0, 1.2, 1.4, 1.6], transform=ccrs.PlateCarree())
        ax.clabel(cs, fmt="%d", fontsize=7)
        ax.contour(lon, lat, ref.eof.sel(mode="extension").values, levels=[-2, -1, 1, 2], colors="#2b7a3d", linewidths=0.7, linestyles=["--", "--", "-", "-"], transform=ccrs.PlateCarree())
        ax.add_patch(plt.Rectangle((EXIT["lon"][0], EXIT["lat"][0]), EXIT["lon"][1] - EXIT["lon"][0], EXIT["lat"][1] - EXIT["lat"][0], fill=False, ec=GOLD, lw=1.6, transform=ccrs.PlateCarree(), zorder=6))
        ax.coastlines(lw=0.5, color="#555"); ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#777")
        ax.set_extent([SECTOR["lon"][0], SECTOR["lon"][1], SECTOR["lat"][0], SECTOR["lat"][1]], crs=ccrs.PlateCarree())
        ax.set_title(f"200 hPa wind — {title}", fontsize=10.5, loc="left", fontweight="bold")
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="#bbb", x_inline=False, y_inline=False)
        gl.top_labels = gl.right_labels = False; gl.left_labels = (k == 0); gl.xlabel_style = gl.ylabel_style = {"size": 7}
    bs = gs[1, 0].get_position(fig)
    cax = fig.add_axes([0.35, bs.y0 + 0.55 * bs.height, 0.30, 0.007])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=7); cb.set_label("u anomaly vs ERA5 day-of-year normal (m/s) · black: u (m/s) · green: extension pattern (EOF1, m/s per σ) · gold: exit region", fontsize=7.5)

    # torque strip
    axq = fig.add_subplot(gs[2, :])
    tvalid = None
    if torque and "Himalaya/Tibet" in torque.get("ranges", {}):
        tvalid = pd.to_datetime(torque["valid"]); tq = np.array(torque["ranges"]["Himalaya/Tibet"], float)
        s1 = torque.get("sd", {}).get("Himalaya/Tibet")
        if s1:
            axq.fill_between(tvalid, -2 * s1, 2 * s1, color=BROWN, alpha=0.08, lw=0); axq.fill_between(tvalid, -s1, s1, color=BROWN, alpha=0.15, lw=0)
        axq.plot(tvalid, tq, color=BROWN, lw=2.2); axq.axhline(0, color="0.5", lw=0.7)
        pk = int(np.argmax(np.abs(tq)))
        axq.set_title(f"Himalaya/Tibet mountain-torque anomaly, AIFS-ENS ensemble mean (init {pd.Timestamp(torque['init']).strftime('%d %b %HZ')}) — peak {tq[pk]:+.0f} Hadley"
                      + (f" ({tq[pk] / s1:+.1f}σ) on {tvalid[pk]:%d %b}" if s1 else ""), fontsize=9.5, loc="left", fontweight="bold")
        axq.set_ylabel("Hadley", fontsize=8)
    else:
        axq.text(0.5, 0.5, "torque series not available for this cycle (torque_map_anim.py runs first)", ha="center", va="center", fontsize=9, color=MUTED, transform=axq.transAxes)
    axq.tick_params(labelsize=7.5); axq.grid(True, alpha=0.2)

    # plumes
    x_fc = valid
    panels = [("extension", "Jet extension index (EOF1, σ)  + = extended east", "σ"), ("exit", "Exit-region 200 hPa wind anomaly, 30–40°N 170°E–150°W (σ of the day of year)", "σ"),
              ("shift", "Jet shift index (EOF2, σ)  + = poleward", "σ"), ("terminus", "Jet terminus: easternmost longitude of the ≥30 m/s core", "°E")]
    summ = {}
    for k, (key, title, unit) in enumerate(panels):
        ax = fig.add_subplot(gs[3 + k // 2, k % 2])
        M = ix[key]
        if key == "terminus":
            climv = harm_eval(ref.term_coef.values, valid.dayofyear.values); sdv = ref.term_sd.values[valid.dayofyear.values - 1]
            ax.fill_between(x_fc, climv - sdv, climv + sdv, color="#000", alpha=0.05, lw=0); ax.plot(x_fc, climv, color="#888", lw=0.9, ls="--")
            frac = ref.term_defined_frac.values[valid.dayofyear.values - 1]
        else:
            ax.axhspan(-1, 1, color="#000", alpha=0.05); ax.axhline(0, color="#555", lw=0.8)
        for m in range(nmem):
            ax.plot(x_fc, M[m], color=NAVY, lw=0.5, alpha=0.18)
        with np.errstate(all="ignore"):
            q10, q50, q90 = np.nanpercentile(M, [10, 50, 90], axis=0); mean = np.nanmean(M, axis=0)
        ax.fill_between(x_fc, q10, q90, color=NAVY, alpha=0.15, lw=0)
        ax.plot(x_fc, mean, color=NAVY, lw=2.4, marker="o", ms=3)
        if tail is not None and len(tail):
            for src, colr, lab in (("era5", "#222", "ERA5"), ("aifs_an", "#777", "AIFS 0-h analysis")):
                t = tail[tail["source"] == src]
                if len(t):
                    ax.plot(t.index, t[key].values, color=colr, lw=1.3, marker="o", ms=2.6, ls=("-" if src == "era5" else "none"))
        ax.axvline(init, color="#999", lw=0.8, ls=":")
        if key in ("extension", "exit"):
            overlay_composite(ax, torque_peak(torque), lag, season_for(init), key)
            if k == 0 and torque_peak(torque) is not None:
                ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.set_title(title, fontsize=9.5, loc="left", fontweight="bold"); ax.set_ylabel(unit, fontsize=8)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(labelsize=7.5); ax.grid(True, alpha=0.2)
        ax.set_xlim(init - pd.Timedelta(days=TAIL_DAYS), valid[-1] + pd.Timedelta(days=1))
        if key == "terminus":
            und = np.mean(~np.isfinite(M), axis=0)
            if und.max() > 0.1:
                ax.text(0.01, 0.03, f"core < {CORE_MS:.0f} m/s in {und.max():.0%} of members on some days (no terminus)", transform=ax.transAxes, fontsize=7, color=MUTED)
        def stat(a):
            a = np.asarray(a, float); return None if not np.isfinite(a).any() else round(float(np.nanmean(a)), 2)
        summ[key] = {"day0": stat(M[:, 0]), "d1_5": stat(M[:, 1:6]), "d6_10": stat(M[:, 6:11]), "d11_15": stat(M[:, 11:16]),
                     "mean": [stat(M[:, d]) for d in range(M.shape[1])], "p10": [None if not np.isfinite(v) else round(float(v), 2) for v in q10],
                     "p90": [None if not np.isfinite(v) else round(float(v), 2) for v in q90]}
        if key != "terminus":
            with np.errstate(invalid="ignore"):
                summ[key]["p_above_1"] = [round(float(np.nanmean(M[:, d] >= 1)), 2) for d in range(M.shape[1])]
                summ[key]["p_below_1"] = [round(float(np.nanmean(M[:, d] <= -1)), 2) for d in range(M.shape[1])]

    # ERA5 lag statistics
    axl = fig.add_subplot(gs[5, 0]); axc = fig.add_subplot(gs[5, 1])
    season = season_for(init)
    if lag and season in lag.get("seasons", {}):
        L = lag["seasons"][season]; lags = lag["lags"]
        for key, colr in (("extension", NAVY), ("exit", GOLD), ("shift", "#2b7a3d"), ("terminus", "#8b1a1a")):
            c = [np.nan if v is None else v for v in L["corr"][key]]
            axl.plot(lags, c, color=colr, lw=1.8, label=key)
        axl.axhline(0, color="0.5", lw=0.7); axl.axvline(0, color="0.6", lw=0.7, ls=":")
        axl.set_title(f"ERA5 {season}: correlation, jet index n days after the Himalayan torque", fontsize=9, loc="left", fontweight="bold")
        axl.set_xlabel("lag (days; + = jet after torque)", fontsize=8); axl.set_ylabel("r", fontsize=8); axl.legend(fontsize=7.5, ncol=4, frameon=False); axl.tick_params(labelsize=7.5); axl.grid(True, alpha=0.2)
        comp = L["composite"]["extension"]
        cm = np.array([np.nan if v is None else v for v in comp["mean"]]); lo = np.array(comp["null_p05"], float); hi = np.array(comp["null_p95"], float)
        axc.fill_between(lags, lo, hi, color="#000", alpha=0.07, lw=0, label="random-date 5–95%")
        axc.plot(lags, cm, color=NAVY, lw=2.2, marker="o", ms=3, label="after torque ≥ +1.5σ")
        ce = L["composite"]["exit"]; axc.plot(lags, [np.nan if v is None else v for v in ce["mean"]], color=GOLD, lw=1.6, label="exit index")
        axc.axhline(0, color="0.5", lw=0.7); axc.axvline(0, color="0.6", lw=0.7, ls=":")
        axc.set_title(f"Composite after {L['n_events']} torque days ≥ +1.5σ ({season}, ERA5)", fontsize=9, loc="left", fontweight="bold")
        axc.set_xlabel("days after the torque peak", fontsize=8); axc.set_ylabel("σ", fontsize=8); axc.legend(fontsize=7.5, frameon=False); axc.tick_params(labelsize=7.5); axc.grid(True, alpha=0.2)
    else:
        for a in (axl, axc):
            a.text(0.5, 0.5, "ERA5 lag statistics not built", ha="center", va="center", color=MUTED, transform=a.transAxes)
    import textwrap
    fig.suptitle(f"North Pacific jet: extension, shift and terminus — AIFS-ENS {nmem} members, init {init:%Y-%m-%d %HZ}", fontsize=13.5, fontweight="bold", x=0.055, ha="left", y=0.99)
    fig.text(0.055, 0.972, "\n".join(textwrap.wrap("Indices on the 200 hPa zonal wind over 10–70°N 100°E–120°W against ERA5 1991–2020. Extension/shift are the leading Nov–Mar EOFs (Jaffe et al. 2011; Winters et al. 2019 use 250 hPa) "
             "in σ of their cold-season spread — meaningful all year as pattern projections, calibrated for winter. Black: ERA5 (~6-day lag); grey dots: AIFS 0-h analyses; navy: members, mean and p10–p90; "
             "dashed red: the ERA5 composite path after strong Himalayan torque days, anchored on this cycle's torque peak.", 200)), fontsize=8, color=MUTED, va="top", linespacing=1.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    print(f"saved {out}", flush=True)
    return summ


def render_z500(init, valid, zi, ref, tail, torque, lag, out: Path) -> dict:
    """The downstream test: ERA5 composite 500 hPa anomalies after strong Himalayan torque days (top),
    this cycle's AIFS-ENS height anomalies (middle), the Alaska and Gulf of Alaska box indices (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import textwrap
    season = season_for(init)
    comp = xr.open_dataset(COMP) if COMP.exists() else None
    lat, lon = ref.zlat.values, ref.zlon.values
    nmem = zi["alaska"].shape[0]
    fig = plt.figure(figsize=(13.4, 8.6))
    gs = GridSpec(3, 4, height_ratios=[0.62, 0.62, 1.3], hspace=0.3, wspace=0.06, left=0.04, right=0.945, top=0.9, bottom=0.065)
    pc = ccrs.PlateCarree(central_longitude=200)
    lev = np.arange(-60, 61, 10); levm = np.arange(-150, 151, 25)

    def boxes(ax):
        for box, colr in ((ALASKA, "#c2185b"), (GOA, GOLD)):
            ax.add_patch(plt.Rectangle((box["lon"][0], box["lat"][0]), box["lon"][1] - box["lon"][0], box["lat"][1] - box["lat"][0], fill=False, ec=colr, lw=1.5, transform=ccrs.PlateCarree(), zorder=6))

    def frame(ax, title, first=False):
        ax.coastlines(lw=0.5, color="#555"); ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#777")
        ax.set_extent([ZDOM["lon"][0], ZDOM["lon"][1], ZDOM["lat"][0], ZDOM["lat"][1]], crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="#bbb", x_inline=False, y_inline=False)
        gl.top_labels = gl.right_labels = False; gl.left_labels = first; gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
        boxes(ax)

    cfc = None
    if comp is not None and season in comp.season.values:
        for k, L in enumerate((3, 6, 9, 12)):
            ax = fig.add_subplot(gs[0, k], projection=pc)
            c = comp.z500_comp.sel(season=season, lag=L).values; pv = comp.z500_p.sel(season=season, lag=L).values
            cfc = ax.contourf(lon, lat, c, levels=lev, cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree())
            sig = pv < 0.05
            yy, xx = np.meshgrid(lat, lon, indexing="ij")
            ax.scatter(xx[sig][::2], yy[sig][::2], s=1.2, color="k", alpha=0.55, transform=ccrs.PlateCarree(), zorder=5)
            frame(ax, (f"ERA5 composite +{L} d (n = {int(comp.n_events.sel(season=season))}, {season})" if k == 0 else f"+{L} d after the torque peak"), first=(k == 0))
    else:
        ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.text(0.5, 0.5, "no composite for this season", ha="center", va="center", color=MUTED)
    cfm = None
    row2 = []
    for k, (title, sl) in enumerate((("AIFS 0-h analysis", slice(0, 1)), ("AIFS days 1–5", slice(1, 6)), ("AIFS days 6–10", slice(6, 11)), ("AIFS days 11–15", slice(11, 16)))):
        ax = fig.add_subplot(gs[1, k], projection=pc); row2.append(ax)
        a = zi["zanom"][:, sl].mean((0, 1))
        cfm = ax.contourf(lon, lat, a, levels=levm, cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree())
        frame(ax, title, first=(k == 0))
    # vertical colour bars in the right margin, spanning each map row
    def vbar(mappable, row, label):
        b0 = gs[row, 0].get_position(fig); cax = fig.add_axes([0.952, b0.y0 + 0.2 * b0.height, 0.008, 0.6 * b0.height])
        cb = fig.colorbar(mappable, cax=cax, orientation="vertical"); cb.ax.tick_params(labelsize=6.5); cb.set_label(label, fontsize=6.8)
    if cfc is not None:
        vbar(cfc, 0, "composite anomaly (m)")
    vbar(cfm, 1, "AIFS anomaly (m)")
    summ = {}
    peak = torque_peak(torque)
    for k, (key, title, colr) in enumerate((("alaska", "Alaska ridge index: 500 hPa anomaly 55–70°N 165–125°W (σ of the day of year)", "#c2185b"),
                                            ("goa", "Gulf of Alaska / West Coast ridge index: 40–60°N 145–120°W (σ)", GOLD))):
        ax = fig.add_subplot(gs[2, 2 * k:2 * k + 2])
        M = zi[key]
        ax.axhspan(-1, 1, color="#000", alpha=0.05); ax.axhline(0, color="#555", lw=0.8)
        for m in range(nmem):
            ax.plot(valid, M[m], color=NAVY, lw=0.5, alpha=0.18)
        q10, q90 = np.nanpercentile(M, [10, 90], axis=0); mean = np.nanmean(M, axis=0)
        ax.fill_between(valid, q10, q90, color=NAVY, alpha=0.15, lw=0); ax.plot(valid, mean, color=NAVY, lw=2.4, marker="o", ms=3)
        if tail is not None and len(tail) and key in tail.columns:
            for src, c2, ls in (("era5", "#222", "-"), ("aifs_an", "#777", "none")):
                t = tail[(tail["source"] == src) & np.isfinite(tail[key].astype(float))]
                if len(t):
                    ax.plot(t.index, t[key].astype(float).values, color=c2, lw=1.3, marker="o", ms=2.6, ls=ls)
        overlay_composite(ax, peak, lag, season, key)
        if peak is not None:
            ax.axvline(peak[0], color=BROWN, lw=1.2, ls="--"); ax.text(peak[0], 0.97, " torque peak", transform=ax.get_xaxis_transform(), fontsize=7, color=BROWN, va="top")
        ax.axvline(init, color="#999", lw=0.8, ls=":")
        ax.set_title(title, fontsize=9.2, loc="left", fontweight="bold", color=INK)
        if k == 0:
            ax.set_ylabel("σ", fontsize=8)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(labelsize=7.5); ax.grid(True, alpha=0.2)
        ax.set_xlim(init - pd.Timedelta(days=TAIL_DAYS), valid[-1] + pd.Timedelta(days=1))
        if k == 0 and peak is not None:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
        stat = lambda a: round(float(np.nanmean(a)), 2)
        with np.errstate(invalid="ignore"):
            summ[key] = {"day0": stat(M[:, 0]), "d1_5": stat(M[:, 1:6]), "d6_10": stat(M[:, 6:11]), "d11_15": stat(M[:, 11:16]),
                         "mean": [stat(M[:, d]) for d in range(M.shape[1])], "p10": q10.round(2).tolist(), "p90": q90.round(2).tolist(),
                         "p_above_1": [round(float(np.nanmean(M[:, d] >= 1)), 2) for d in range(M.shape[1])]}
    fig.suptitle(f"Downstream of the torque: 500 hPa ridging over Alaska and the Gulf of Alaska — AIFS-ENS {nmem} members, init {init:%Y-%m-%d %HZ}", fontsize=13, fontweight="bold", x=0.04, ha="left", y=0.985)
    fig.text(0.04, 0.962, "\n".join(textwrap.wrap("Top: what ERA5 1991–2020 says usually follows a Himalayan mountain-torque day ≥ +1.5σ in this season — the composite 500 hPa height anomaly 3, 6, 9 and 12 days later, stippled where fewer than 5% of "
             "random same-season date sets are as extreme. Middle: this cycle's AIFS-ENS ensemble-mean height anomaly. Bottom: the Alaska (magenta box) and Gulf of Alaska / West Coast (gold box) ridge indices, members and mean, "
             "with the ERA5 record (black), the AIFS analyses (grey) and the composite expectation anchored on this cycle's torque peak (dashed red, with its random-date band).", 205)), fontsize=8, color=MUTED, va="top", linespacing=1.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    print(f"saved {out}", flush=True)
    return summ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--out", default="assets/sst/pacjet.webp"); ap.add_argument("--json", default="assets/sst/data/pacjet.json")
    ap.add_argument("--torque", default="assets/sst/data/torque_ranges.json")
    ap.add_argument("--out-z", default=None, help="second figure (default: <out> with _z500 suffix)")
    a = ap.parse_args()
    t0 = time.time()
    if not CLIM.exists():
        raise SystemExit("reference missing: run build_pacjet_clim.py")
    ref = xr.open_dataset(CLIM).load()
    init = pd.Timestamp(f"{a.date}T{a.time}:00")
    u = load_u200(a.date, a.time, ref)
    valid = pd.DatetimeIndex([init.normalize() + pd.Timedelta(days=int(d)) for d in u.day.values])
    print(f"  u200 {dict(u.sizes)} loaded ({time.time() - t0:.0f}s)", flush=True)
    ix = indices(u, valid, ref)
    an0 = {k: float(np.nanmean(ix[k][:, 0])) for k in ("extension", "shift", "exit", "terminus", "terminus_anom")}
    zf = load_z500(a.date, a.time, ref)
    zi = None
    if zf is not None:
        zi = z_indices(zf, valid, ref); an0["alaska"] = float(np.nanmean(zi["alaska"][:, 0])); an0["goa"] = float(np.nanmean(zi["goa"][:, 0]))
        print(f"  z500 {dict(zf.sizes)} loaded ({time.time() - t0:.0f}s)", flush=True)
    tail = observed_tail(init, ref, an0)
    torque = json.loads(Path(a.torque).read_text()) if Path(a.torque).exists() else None
    lag = json.loads(LAG.read_text()) if LAG.exists() else None
    summ = render(init, valid, ix, u.isel(day=0).mean("number").values, ref, tail, torque, lag, Path(a.out))
    if zi is not None:
        outz = Path(a.out_z) if a.out_z else Path(a.out).with_name(Path(a.out).stem + "_z500" + Path(a.out).suffix)
        summ.update(render_z500(init, valid, zi, ref, tail, torque, lag, outz))
    doc = {"init": init.strftime("%Y-%m-%dT%HZ"), "valid": [v.strftime("%Y-%m-%d") for v in valid], "members": int(u.sizes["number"]),
           "indices": summ, "analysis": {k: round(v, 2) for k, v in an0.items()},
           "observed_tail": [{"date": d.strftime("%Y-%m-%d"), "source": r["source"], **{k: (None if not np.isfinite(float(r[k])) else round(float(r[k]), 2)) for k in ("extension", "shift", "exit", "terminus", "alaska", "goa")}} for d, r in tail.iterrows()] if tail is not None else [],
           "torque": ({"init": torque["init"], "himalaya": torque["ranges"].get("Himalaya/Tibet"), "sd": torque.get("sd", {}).get("Himalaya/Tibet"), "valid": torque["valid"],
                       "peak": (lambda pk: {"date": pk[0].strftime("%Y-%m-%d"), "hadley": round(pk[1], 1), "sigma": (None if pk[2] is None else round(pk[2], 2))})(torque_peak(torque))} if torque else None),
           "season": season_for(init), "lag_summary": ({s_: {"n_events": lag["seasons"][s_]["n_events"], "corr": lag["seasons"][s_]["corr"], "corr_extension_to_alaska": lag["seasons"][s_].get("corr_extension_to_alaska"),
                                                              "corr_extension_to_goa": lag["seasons"][s_].get("corr_extension_to_goa")} for s_ in lag["seasons"]} if lag else None),
           "ridge_track": (lag or {}).get("ridge_track"),
           "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {a.json} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
