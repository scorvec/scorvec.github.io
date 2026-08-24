#!/usr/bin/env python3
"""
Does El Niño FLAVOR (east- vs central-based) matter for Brazil summer heat,
beyond amplitude (RONI)?

East-lean index: RONI-style relative anomalies (box minus 20S–20N tropical
mean, 3-mo smoothed, ERSST v5) of Niño-1+2 minus Niño-4, NDJFM-mean per
summer. 1997/98 = +2.8 (canonical EP), 2009/10 = −0.7 (Modoki). The current
event is running ~+3 (30-day OISST), i.e. maximally east-based.

Tests:
  1. National: detrended load-weighted NDJFM CDD ~ RONI (+ east-lean):
     coefficient, p, ΔR²; same on the El Niño-only subset.
  2. Per city: DJF temp anomaly ~ hinge trend + RONI + east-lean → map of
     the east-lean coefficient (90 % significance filled).

Output: ~/brazil_flavor_elnino.png + printed stats.
"""
from __future__ import annotations

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
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from roni_summer_brazil import (city_summers, roni_history, CLIM0, CLIM1,   # noqa: E402
                                HINGE_YEAR, P_SIG, ERSST)
from brazil_cdd_ndjfm import (roni_monthly, month_year, MONTHS,             # noqa: E402
                              load_weights, _ensure_ghcnm, BASE, NDAYS,
                              Y0 as CY0, Y1 as CY1, FC_CSV)

HOME = Path.home()


def elean_series() -> tuple[pd.Series, pd.Series]:
    """(NDJFM east-lean per summer, DJF east-lean per summer)."""
    ds = xr.open_dataset(ERSST)
    sst = ds["sst"].sortby("lat")
    clim = (sst.sel(time=slice(f"{CLIM0}-01-01", f"{CLIM1}-12-31"))
            .groupby("time.month").mean("time"))
    anom = sst.groupby("time.month") - clim

    def wm(da):
        w = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(w).mean(("lat", "lon"), skipna=True)

    trop = wm(anom.sel(lat=slice(-20, 20)))
    n12 = (wm(anom.sel(lat=slice(-10, 0), lon=slice(270, 280))) - trop)
    n4 = (wm(anom.sel(lat=slice(-5, 5), lon=slice(160, 210))) - trop)
    el = (n12 - n4).rolling(time=3, center=True, min_periods=2).mean().to_series()
    out_n, out_d = {}, {}
    for y in range(1900, CY1 + 1):
        mn = [month_year(y, m) for m in MONTHS]
        md = [month_year(y, m) for m in (12, 1, 2)]
        vn = [el.get(x, np.nan) for x in mn]
        vd = [el.get(x, np.nan) for x in md]
        if np.isfinite(vn).sum() >= 4:
            out_n[y] = float(np.nanmean(vn))
        if np.isfinite(vd).sum() >= 2:
            out_d[y] = float(np.nanmean(vd))
    return pd.Series(out_n), pd.Series(out_d)


def national_cdd() -> pd.Series:
    """Rebuild the load-weighted NDJFM CDD total (same as brazil_cdd_ndjfm)."""
    fc = pd.read_csv(FC_CSV)
    cities = list(fc.city)
    sid_by_city = dict(zip(fc.city, fc.station))
    w = load_weights(cities)
    dat_path, _ = _ensure_ghcnm()
    wanted = set(sid_by_city.values())
    raw: dict[str, dict] = {sid: {} for sid in wanted}
    with open(dat_path) as f:
        for ln in f:
            sid = ln[:11]
            if sid not in raw:
                continue
            year = int(ln[11:15])
            if not (CY0 - 1 <= year <= CY1 + 2):
                continue
            for m in range(12):
                v = int(ln[19 + m * 8: 24 + m * 8])
                if v != -9999:
                    raw[sid][(year, m + 1)] = v / 100.0
    nat_m = {}
    for m in MONTHS:
        norm = {}
        for c in cities:
            s = raw[sid_by_city[c]]
            base = [s.get((month_year(yy, m).year, m))
                    for yy in range(CLIM0, CLIM1 + 1)]
            base = [v for v in base if v is not None]
            norm[c] = (max(0.0, np.mean(base) - BASE) * NDAYS[m]
                       if len(base) >= 12 else np.nan)
        nc = pd.Series(norm)
        nn = float((nc[nc.notna()] * w[nc.notna()]).sum() / w[nc.notna()].sum())
        ser = {}
        for y in range(CY0, CY1 + 1):
            row = {}
            for c in cities:
                t = raw[sid_by_city[c]].get((month_year(y, m).year, m))
                if t is not None:
                    row[c] = max(0.0, t - BASE) * NDAYS[m]
            row = pd.Series(row).reindex(cities)
            mm = row.notna() & nc.notna()
            if w[mm].sum() < 0.6:
                continue
            ser[y] = float(((row - nc)[mm] * w[mm]).sum() / w[mm].sum()) + nn
        nat_m[m] = pd.Series(ser)
    return pd.concat(nat_m, axis=1).dropna().sum(axis=1)


def ols(y, X):
    A = np.column_stack([np.ones(len(y))] + list(X))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = len(y) - A.shape[1]
    cov = (resid @ resid) / dof * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    p = 2 * stats.t.sf(np.abs(coef / se), dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return coef, p, r2, resid


def main() -> int:
    from roni_summer_brazil import roni_history as roni_djf_hist
    el_n, el_d = elean_series()
    total = national_cdd()
    roni_m = roni_monthly()
    roni_n = pd.Series({y: np.nanmean([roni_m.get(month_year(y, m), np.nan)
                                       for m in MONTHS]) for y in total.index})

    yr = total.index.to_numpy(float)
    Xt = [yr - 2000, np.clip(yr - HINGE_YEAR, 0, None)]
    _, _, _, det = ols(total.values, Xt)                  # detrended residual
    detr = pd.Series(det, index=total.index)

    both = pd.concat([detr.rename("cdd"), roni_n.rename("roni"),
                      el_n.rename("elean")], axis=1).dropna()
    c1, p1, r2_1, _ = ols(both.cdd.values, [both.roni.values])
    c2, p2, r2_2, _ = ols(both.cdd.values, [both.roni.values, both.elean.values])
    print(f"national detrended NDJFM CDD (n={len(both)}):")
    print(f"  RONI only:      {c1[1]:+.1f} CDD/°C (p={p1[1]:.3f})  R²={r2_1:.3f}")
    print(f"  RONI + eastlean: RONI {c2[1]:+.1f} (p={p2[1]:.3f})  "
          f"east-lean {c2[2]:+.1f} CDD/°C (p={p2[2]:.3f})  R²={r2_2:.3f}  "
          f"ΔR²={r2_2-r2_1:+.3f}")
    nino = both[both.roni > 0.5]
    if len(nino) > 8:
        c3, p3, r2_3, _ = ols(nino.cdd.values, [nino.elean.values])
        print(f"  El Niño summers only (n={len(nino)}): east-lean "
              f"{c3[1]:+.1f} CDD/°C (p={p3[1]:.3f})")

    # per-city DJF: trend + RONI + east-lean
    anom, meta = city_summers()
    roni_djf = roni_djf_hist()
    rows = []
    for _, mrow in meta.iterrows():
        c = mrow["city"]
        ser = anom[c].dropna()
        idx = ser.index.intersection(roni_djf.index).intersection(el_d.index)
        y = ser.loc[idx].to_numpy()
        yrs = idx.to_numpy(float)
        X = [yrs - 2000, np.clip(yrs - HINGE_YEAR, 0, None),
             roni_djf.loc[idx].to_numpy(), el_d.loc[idx].to_numpy()]
        coef, p, _, _ = ols(y, X)
        rows.append(dict(city=c, lat=mrow["lat"], lon=mrow["lon"],
                         elean_coef=coef[4], elean_p=p[4],
                         roni_coef=coef[3], roni_p=p[3]))
    cf = pd.DataFrame(rows)
    ns = int((cf.elean_p < P_SIG).sum())
    print(f"per-city: east-lean significant at {ns}/{len(cf)} cities; "
          f"coef range {cf.elean_coef.min():+.2f}…{cf.elean_coef.max():+.2f} °C per °C")

    # ── figure: partial scatter + coefficient map ──
    fig = plt.figure(figsize=(14.5, 6.6), dpi=150)
    ax1 = fig.add_subplot(1, 2, 1)
    _, _, _, res_r = ols(both.cdd.values, [both.roni.values])
    sc = ax1.scatter(both.elean, res_r, c=both.roni, cmap="coolwarm",
                     vmin=-2, vmax=2, s=46, edgecolor="#333", lw=0.4)
    cb = fig.colorbar(sc, ax=ax1, fraction=0.045, pad=0.02)
    cb.set_label("NDJFM RONI (°C)", fontsize=9)
    bfit = np.polyfit(both.elean, res_r, 1)
    xs = np.linspace(both.elean.min() - 0.2, 3.1, 40)
    ax1.plot(xs, np.polyval(bfit, xs), ls="--", color="#888",
             label=f"{bfit[0]:+.1f} CDD per east-lean °C")
    ax1.axvline(2.5, color="#7a0018", lw=1.4, ls=":",
                label="2026/27 if event stays east-based (~+2.5)")
    for y in (1997, 1982, 2009, 2015, 2023):
        if y in both.index:
            ax1.annotate(str(y), (both.elean[y], res_r[both.index.get_loc(y)]),
                         textcoords="offset points", xytext=(4, 3), fontsize=7,
                         color="#555")
    ax1.axhline(0, color="#333", lw=0.7)
    ax1.set_xlabel("NDJFM east-lean (rel. N1+2 − rel. N4, °C)", fontsize=10)
    ax1.set_ylabel("national CDD residual (trend & RONI removed)", fontsize=10)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(alpha=0.25)
    ax1.set_title("Does flavor add heat beyond amplitude?", fontsize=11, loc="left")

    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_extent([-74.5, -34, -34.5, 5.8], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5, edgecolor="#555")
    ax2.add_feature(cfeature.STATES.with_scale("50m"), lw=0.3, edgecolor="#999")
    ax2.coastlines("50m", lw=0.5, color="#555")
    sig = cf.elean_p < P_SIG
    vm = float(np.abs(cf.elean_coef).max())
    sc2 = ax2.scatter(cf.lon[sig], cf.lat[sig], c=cf.elean_coef[sig],
                      cmap="RdBu_r", vmin=-vm, vmax=vm, s=70,
                      edgecolor="#222", lw=0.6, zorder=5,
                      transform=ccrs.PlateCarree())
    ax2.scatter(cf.lon[~sig], cf.lat[~sig], c=cf.elean_coef[~sig],
                cmap="RdBu_r", vmin=-vm, vmax=vm, s=28, alpha=0.45,
                edgecolor="none", zorder=4, transform=ccrs.PlateCarree())
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label("DJF °C per east-lean °C (RONI held fixed)", fontsize=9)
    ax2.set_title(f"East-lean coefficient by city (big = sig. 90 %, {ns}/{len(cf)})",
                  fontsize=11, loc="left")

    fig.suptitle("Brazil summer heat and El Niño flavor — east-based vs central "
                 "events (current event: east-lean ≈ +3, 1997-class)",
                 fontsize=13, x=0.02, ha="left")
    fig.text(0.012, 0.012,
             "East-lean = relative Niño-1+2 minus relative Niño-4 (ERSST, 3-mo "
             "smoothed). Left: national load-weighted NDJFM CDD residuals after "
             "removing hinge trend and RONI. Right: per-city DJF regression with "
             "trend + RONI + east-lean.", fontsize=7.5, color="#666")
    out = HOME / "brazil_flavor_elnino.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
