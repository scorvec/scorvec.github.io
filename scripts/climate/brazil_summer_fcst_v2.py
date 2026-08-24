#!/usr/bin/env python3
"""
Brazil 2026/27 summer monthly temperature forecast, v2 — adds to the v1
trend + EP-fingerprint construction:

  · Atlantic teleconnections: per-month partial-regression maps of 20CR t2m
    on detrended TNA and TSA (controlling RONI, east-lean and PDO), times
    lag-damped forecasts of each index from its June 2026 value
  · PDO: same treatment (NCEI ERSST PDO index)
  · C3S dynamical MME: 10-system Aug-init member forecasts over South
    America (cached SA box), anomalies vs each system's own 1993–2016
    hindcast climatology — available for Nov/Dec/Jan valid months

Final field per month = 0.5 · statistical + 0.5 · dynamical (Nov–Jan);
statistical-only for Feb/Mar. Statistical = ERA5 hinge trend projection
+ EP fingerprint (v1 blend) + TNA/TSA/PDO contributions.

Outputs: ~/brazil_summer_fcst_v2_vs{30,10,5}yr.png + components figure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.path.expanduser("~/c3s/scripts"))
import brazil_flavor_composites as bfc                       # noqa: E402
import brazil_summer_fcst_maps as v1                         # noqa: E402

ERSST = HERE.parent / "sst" / "data" / "ersst_v5_mnmean.nc"
PDO_DAT = HERE.parent / "sst" / "data" / "ersst.v5.pdo.dat"
MONTHS = v1.MONTHS
MNAME = v1.MNAME
EXT = v1.EXT
DYN_MONTHS = {11: 3, 12: 4, 1: 5}          # calendar month → f4 step index
TELE = ["TNA", "TSA", "PDO"]

BOXES = {
    "TNA": ((5.5, 23.5), [(302.5, 345)]),
    "TSA": ((-20, 0), [(330, 360), (0, 10)]),
}


def wm(da):
    w = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(w).mean(("lat", "lon"), skipna=True)


def hinge(vals: np.ndarray, yrs: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones_like(yrs), yrs - 2000,
                         np.clip(yrs - 1970, 0, None)])
    m = np.isfinite(vals)
    c, *_ = np.linalg.lstsq(A[m], vals[m], rcond=None)
    return vals - A @ c


def monthly_index_table() -> dict[str, pd.DataFrame]:
    """Each index as a (event_year × calendar_month) table of hinge-detrended
    monthly anomalies. Event year y spans Aug(y)…Jul(y+1)."""
    ds = xr.open_dataset(ERSST)
    sst = ds["sst"].sortby("lat")
    clim = (sst.sel(time=slice("1991-01-01", "2020-12-31"))
            .groupby("time.month").mean("time"))
    anom = sst.groupby("time.month") - clim
    trop = wm(anom.sel(lat=slice(-20, 20))).to_series()

    def box(lat, lons):
        parts = [anom.sel(lat=slice(*lat), lon=slice(*lo)) for lo in lons]
        return wm(xr.concat(parts, dim="lon")).to_series()

    raw = {k: box(*v) for k, v in BOXES.items()}
    n34 = box((-5, 5), [(190, 240)]) - trop
    n12 = box((-10, 0), [(270, 280)]) - trop
    n4 = box((-5, 5), [(160, 210)]) - trop
    raw["RONI"] = n34
    raw["elean"] = n12 - n4

    pdo_rows = {}
    for ln in PDO_DAT.read_text().splitlines():
        p = ln.split()
        if len(p) == 13 and p[0].isdigit():
            pdo_rows[int(p[0])] = [float(x) if float(x) < 90 else np.nan
                                   for x in p[1:]]
    pdo = pd.Series({pd.Timestamp(y, m + 1, 1): v[m]
                     for y, v in pdo_rows.items() for m in range(12)}).sort_index()
    raw["PDO"] = pdo

    out = {}
    for k, s in raw.items():
        tab = {}
        for y in range(1870, 2026):
            row = {}
            for m in range(1, 13):
                cy = y if m >= 8 else y + 1
                row[m] = s.get(pd.Timestamp(cy, m, 1), np.nan)
            tab[y] = row
        df = pd.DataFrame(tab).T
        yrs = df.index.to_numpy(float)
        for m in df.columns:
            df[m] = hinge(df[m].to_numpy(float), yrs)
        out[k] = df
    return out


def tna_plume_fc() -> dict[int, float]:
    """Per-calendar-month detrended TNA forecast from the f19 C3S plume:
    MME mean of member anomalies (vs own hindcast clim) minus the ERSST
    century hinge-trend value at that date. Missing late months persist
    the last available value."""
    import f19_tna_plume as f19
    import common as C

    c3s = f19.c3s_plumes()
    er = xr.open_dataset(f"{C.DATA}/analog/ersstv5_global_f1.nc")["sst"]
    clim = er.sel(time=slice("1991", "2020")).groupby("time.month").mean("time")
    tna_m = f19.box_mean(er.groupby("time.month") - clim).to_series()
    x = np.array([t.year + t.month / 12 for t in tna_m.index])
    A = np.column_stack([np.ones_like(x), x - 2000,
                         np.clip(x - 1970, 0, None)])
    hcoef, *_ = np.linalg.lstsq(A, tna_m.values, rcond=None)

    def trend_val(t):
        xx = t.year + t.month / 12
        return float(hcoef[0] + hcoef[1] * (xx - 2000)
                     + hcoef[2] * max(0.0, xx - 1970))

    pool = {}
    for ser in c3s.values():
        for t, vals in ser.items():
            pool.setdefault(t, []).append(float(np.mean(vals)))
    out, last = {}, None
    for m in MONTHS:
        cy = v1.TARGET_YEAR if m >= 8 else v1.TARGET_YEAR + 1
        t = pd.Timestamp(cy, m, 1)
        if t in pool:
            out[m] = float(np.mean(pool[t])) - trend_val(t)
            last = out[m]
        elif last is not None:
            out[m] = last                     # persist beyond model range
    print("TNA plume (detrended MME):",
          {MNAME[m]: round(v, 2) for m, v in out.items()})
    return out


def tele_betas_and_fc(idx):
    """Per target month: 20CR partial-regression beta maps for TNA/TSA/PDO
    (controlling RONI + east-lean) and lag-damped index forecasts."""
    ds = xr.open_dataset(bfc.DATA / "air.2m.mon.mean.nc")
    air = ds["air"].sortby("lat")
    sa = air.sel(lat=slice(EXT[2] - 4, EXT[3] + 4),
                 lon=slice((EXT[0] - 4) % 360, (EXT[1] + 4) % 360)).load()
    clim = sa.groupby("time.month").mean("time")
    an = sa.groupby("time.month") - clim
    t = pd.DatetimeIndex(an["time"].values)
    lo = sa["lon"].values
    lat, lon = sa["lat"].values, np.where(lo > 180, lo - 360, lo)

    preds = ["RONI", "elean", "TNA", "TSA", "PDO"]
    betas, fc_val = {}, {}
    for m in MONTHS:
        sel = an.isel(time=(t.month == m))
        yrs = (pd.DatetimeIndex(sel["time"].values).year
               - (1 if m <= 3 else 0)).to_numpy(float)
        keep = (yrs >= 1870) & (yrs <= 2014)
        sel, yrs = sel.isel(time=keep), yrs[keep]
        v = sel.values.reshape(len(yrs), -1)
        yc = yrs - yrs.mean()
        sl = (yc @ (v - v.mean(0))) / (yc @ yc)
        dv = v - v.mean(0) - np.outer(yc, sl)

        X = np.column_stack([idx[p].loc[yrs.astype(int), m].to_numpy(float)
                             for p in preds])
        ok = np.isfinite(X).all(1)
        A = np.column_stack([np.ones(ok.sum()), X[ok]])
        c, *_ = np.linalg.lstsq(A, dv[ok], rcond=None)
        betas[m] = {p: xr.DataArray(
            c[1 + i].reshape(sel.shape[1:]),
            coords=dict(lat=lat, lon=lon), dims=("lat", "lon"))
            for i, p in enumerate(preds) if p in TELE}

        for p in TELE:
            tab = idx[p]
            tgt = tab[m].to_numpy(float)
            jun = tab[6].to_numpy(float)          # June of event year y+? —
            # month 6 in event-year row y is Jun(y+1); we want Jun BEFORE the
            # summer, i.e. Jun of calendar 2026 = month 6 of event-year 2025.
            yrs_all = tab.index.to_numpy(float)
            jun_prev = np.full_like(tgt, np.nan)
            jun_prev[1:] = jun[:-1]                # Jun(y) sits in row y-1
            mm = np.isfinite(tgt) & np.isfinite(jun_prev) & \
                (yrs_all >= 1870) & (yrs_all <= 2024)
            b = np.polyfit(jun_prev[mm], tgt[mm], 1)
            cur = float(tab.loc[2025, 6])          # Jun 2026 detrended
            fc_val.setdefault(m, {})[p] = float(np.polyval(b, cur))

    # TNA: the June-ERSST lag regression misses the sharp Jul–Aug warming;
    # use the f19 C3S plume (bias-corrected vs own hindcasts, detrended by
    # the same century hinge) instead.
    tna_fc = tna_plume_fc()
    for m in MONTHS:
        if m in tna_fc:
            fc_val[m]["TNA"] = tna_fc[m]
    print("index forecasts (detrended):")
    for m in MONTHS:
        print(f"  {MNAME[m]}: " + "  ".join(
            f"{p}={fc_val[m][p]:+.2f}" for p in TELE))
    return betas, fc_val


def dynamical_anoms(lat_t, lon_t):
    """C3S MME t2m anomaly (vs own 1993–2016 hindcast clim) per month, on the
    target grid. Returns dict month → 2-D array, plus system count."""
    import f4_lib as F
    out = {m: [] for m in DYN_MONTHS}
    systems = F.models_present("sa_fc")
    for mdl in systems:
        ds = F.load_sa(mdl)
        for m, k in DYN_MONTHS.items():
            a = (ds["fc_t2m"].isel(step=k).mean("number")
                 - ds["hc_t2m"].isel(step=k).mean("sample"))
            a = a.rename(latitude="lat", longitude="lon").sortby("lat")
            out[m].append(a.interp(lat=lat_t, lon=lon_t).values)
    print(f"C3S systems: {len(systems)} ({', '.join(systems)})")
    return {m: np.nanmean(v, axis=0) for m, v in out.items()}, len(systems)


def draw_panel(fig, pos, lon, lat, fld, title, vmax=3.0, cmap="RdBu_r",
               unit="°C"):
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    ax.set_extent(EXT, crs=ccrs.PlateCarree())
    lv = np.linspace(-vmax, vmax, 25)
    cf = ax.contourf(lon, lat, fld, levels=lv, cmap=cmap, extend="both",
                     transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6,
                   edgecolor="#333")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.25,
                   edgecolor="#888")
    ax.coastlines("50m", lw=0.6, color="#333")
    cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(unit, fontsize=8)
    ax.set_title(title, fontsize=10.5, loc="left")
    return ax


def compute() -> dict:
    """Build all forecast fields; returns everything the figures (and
    downstream diagnostics) need."""
    # ── ERA5 base (as v1) + 1993–2016 clim for dynamical anchoring ──
    ds = xr.open_dataset(v1.ERA5)
    t2 = ds["t2m"].sortby("lat")
    t2 = t2.assign_coords(lon=(((t2["lon"] + 180) % 360) - 180)).sortby("lon")
    t2 = t2.sel(lat=slice(EXT[2] - 4, EXT[3] + 4),
                lon=slice(EXT[0] - 4, EXT[1] + 4)).load()
    t = pd.DatetimeIndex(t2["time"].values)
    lat_e, lon_e = t2["lat"].values, t2["lon"].values

    proj, normals, clim9316 = {}, {30: {}, 10: {}, 5: {}}, {}
    for m in MONTHS:
        sel = t2.isel(time=(t.month == m))
        yrs = (pd.DatetimeIndex(sel["time"].values).year
               - (1 if m <= 3 else 0)).to_numpy(float)
        A = np.column_stack([np.ones_like(yrs), yrs - 2000,
                             np.clip(yrs - 1970, 0, None)])
        v = sel.values.reshape(len(yrs), -1)
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        xt = np.array([1.0, v1.TARGET_YEAR - 2000, v1.TARGET_YEAR - 1970])
        proj[m] = (xt @ c).reshape(sel.shape[1:])
        for nb in (30, 10, 5):
            normals[nb][m] = sel.isel(time=slice(-nb, None)).mean("time").values
        hcw = (yrs >= 1993) & (yrs <= 2016)
        clim9316[m] = sel.isel(time=hcw).mean("time").values

    # ── EP fingerprint (v1 blend) ──
    amp_obs = v1.obs_amplitude()
    obs_fp = v1.obs_monthly_fingerprint()
    mdl_fp = v1.model_monthly_fingerprint(lat_e, lon_e)
    fp = {m: 0.5 * obs_fp[m].interp(lat=lat_e, lon=lon_e).values
          * (v1.RONI_FC / amp_obs) + 0.5 * mdl_fp[m] for m in MONTHS}

    # ── teleconnections ──
    idx = monthly_index_table()
    betas, fc_val = tele_betas_and_fc(idx)
    tele = {m: {p: betas[m][p].interp(lat=lat_e, lon=lon_e).values
                * fc_val[m][p] for p in TELE} for m in MONTHS}

    # ── dynamical MME ──
    mme, nsys = dynamical_anoms(lat_e, lon_e)

    # ── assemble ──
    stat_abs, final_abs = {}, {}
    for m in MONTHS:
        stat_abs[m] = proj[m] + fp[m] + sum(tele[m][p] for p in TELE)
        if m in mme:
            dyn_abs = clim9316[m] + mme[m]
            final_abs[m] = 0.5 * stat_abs[m] + 0.5 * dyn_abs
        else:
            final_abs[m] = stat_abs[m]
    return dict(lat=lat_e, lon=lon_e, t2=t2, proj=proj, normals=normals,
                clim9316=clim9316, fp=fp, tele=tele, mme=mme, nsys=nsys,
                stat_abs=stat_abs, final_abs=final_abs)


def main() -> int:
    d = compute()
    lat_e, lon_e = d["lat"], d["lon"]
    proj, normals, clim9316 = d["proj"], d["normals"], d["clim9316"]
    fp, tele, mme, nsys = d["fp"], d["tele"], d["mme"], d["nsys"]
    stat_abs, final_abs = d["stat_abs"], d["final_abs"]

    # ── monthly figures vs normals ──
    for nb in (30, 10, 5):
        fig = plt.figure(figsize=(15.5, 11.5), dpi=150)
        fields = {m: final_abs[m] - normals[nb][m] for m in MONTHS}
        fields["NDJFM"] = np.mean([fields[m] for m in MONTHS], axis=0)
        for i, key in enumerate(MONTHS + ["NDJFM"]):
            tag = "" if key == "NDJFM" else \
                (" · stat+C3S" if key in mme else " · stat only")
            draw_panel(fig, (2, 3, i + 1), lon_e, lat_e, fields[key],
                       MNAME.get(key, "NDJFM mean") + tag)
        fig.suptitle(
            f"Brazil 2026/27 summer temperature forecast v2 — anomaly vs "
            f"trailing {nb}-year normal\n"
            "statistical (ERA5 trend + EP fingerprint + TNA/TSA/PDO terms) "
            f"blended 50/50 with C3S {nsys}-system MME (Nov–Jan)",
            fontsize=13, x=0.03, ha="left")
        fig.text(0.012, 0.008,
                 "EP fingerprint: 50% 20CR obs (n=5) + 50% CMIP6 passers "
                 "(n=203), scaled to RONI +2.75. Teleconnection betas: 20CR "
                 "1870–2014 partial regressions; index forecasts lag-damped "
                 "from Jun 2026. C3S anomalies vs own 1993–2016 hindcasts, "
                 "anchored to ERA5 1993–2016.", fontsize=7.5, color="#555")
        out = Path.home() / f"brazil_summer_fcst_v2_vs{nb}yr.png"
        fig.savefig(out, facecolor="white", bbox_inches="tight",
                    pad_inches=0.15)
        plt.close(fig)
        print(f"wrote {out}")

    # ── components figure (NDJFM means; C3S over its Nov–Jan window) ──
    fig = plt.figure(figsize=(19, 10.5), dpi=150)
    comp = [
        ("Climate trend (vs 30-yr normal)",
         np.mean([proj[m] - normals[30][m] for m in MONTHS], axis=0), 3.0),
        ("EP El Niño fingerprint", np.mean([fp[m] for m in MONTHS], axis=0),
         3.0),
        ("TNA contribution", np.mean([tele[m]["TNA"] for m in MONTHS],
                                     axis=0), 1.0),
        ("TSA contribution", np.mean([tele[m]["TSA"] for m in MONTHS],
                                     axis=0), 1.0),
        ("PDO contribution", np.mean([tele[m]["PDO"] for m in MONTHS],
                                     axis=0), 1.0),
        (f"C3S {nsys}-system MME anomaly (Nov–Jan)",
         np.mean([mme[m] for m in mme], axis=0), 3.0),
        ("Statistical total (vs 30-yr)",
         np.mean([stat_abs[m] - normals[30][m] for m in MONTHS], axis=0),
         3.0),
        ("FINAL blend (vs 30-yr)",
         np.mean([final_abs[m] - normals[30][m] for m in MONTHS], axis=0),
         3.0),
    ]
    for i, (ttl, fld, vmax) in enumerate(comp):
        draw_panel(fig, (2, 4, i + 1), lon_e, lat_e, fld, ttl, vmax=vmax)
    fig.suptitle("Forecast components — NDJFM 2026/27 means",
                 fontsize=13.5, x=0.03, ha="left")
    fig.text(0.012, 0.008,
             "Note narrower ±1 °C scale on the teleconnection panels. "
             "C3S MME panel is the raw dynamical anomaly vs 1993–2016.",
             fontsize=8, color="#555")
    out = Path.home() / "brazil_summer_fcst_v2_components.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
