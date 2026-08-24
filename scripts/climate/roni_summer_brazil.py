#!/usr/bin/env python3
"""
DJF (austral summer) temperature outlook for ~70 Brazilian cities from
RONI + climate trend — the Brazil-summer sibling of roni_winter_forecast.py.

Method identical: per-city OLS on GHCN-M v4 adjusted station observations,
  anomaly = a + b1·year + b2·max(0, year−1970) + c·RONI
with the RONI term kept only where significant at 90%. Summer Y = Dec Y
through Feb Y+1; anomalies vs the city's 1991–2020 DJF normal, in °C.
Brazilian GHCN records are patchier than US ones, so station selection
accepts records ending ≥2010 and 2-of-3 DJF months (the regression fits on each station's span;
the forecast extrapolates the trend), still preferring starts ≤1910.

    python roni_summer_brazil.py [--roni 2.75] [--out-dir plots]
"""
from __future__ import annotations

import argparse
import json
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
REPO = HERE.parent.parent
ERSST = REPO / "scripts" / "sst" / "data" / "ersst_v5_mnmean.nc"
MANIFEST = REPO / "assets" / "sst" / "manifest.json"

Y0, Y1 = 1900, 2025            # summers 1900/01 … 2025/26 (year = Dec year)
CLIM0, CLIM1 = 1991, 2020
TARGET = 2026                  # forecast summer 2026/27 (Dec 2026–Feb 2027)
P_SIG = 0.10
HINGE_YEAR = 1970
MIN_END = 2010                 # station record must reach at least this summer
MIN_WINTERS = 40

# ~70 cities across Brazil (name, lat, lon)
CITIES = [
    ("Boa Vista RR", 2.82, -60.67), ("Macapá AP", 0.03, -51.07),
    ("Belém PA", -1.46, -48.49), ("Santarém PA", -2.44, -54.70),
    ("Marabá PA", -5.35, -49.12), ("Altamira PA", -3.20, -52.21),
    ("Manaus AM", -3.12, -60.02), ("Tefé AM", -3.35, -64.71),
    ("Cruzeiro do Sul AC", -7.63, -72.67), ("Rio Branco AC", -9.97, -67.81),
    ("Porto Velho RO", -8.76, -63.90), ("Vilhena RO", -12.74, -60.15),
    ("São Luís MA", -2.53, -44.30), ("Imperatriz MA", -5.53, -47.48),
    ("Caxias MA", -4.86, -43.36), ("Teresina PI", -5.09, -42.80),
    ("Floriano PI", -6.77, -43.02), ("Fortaleza CE", -3.72, -38.54),
    ("Quixeramobim CE", -5.20, -39.29), ("Natal RN", -5.79, -35.21),
    ("João Pessoa PB", -7.12, -34.86), ("Campina Grande PB", -7.22, -35.88),
    ("Recife PE", -8.05, -34.88), ("Petrolina PE", -9.39, -40.50),
    ("Maceió AL", -9.67, -35.74), ("Aracaju SE", -10.91, -37.07),
    ("Salvador BA", -12.97, -38.51), ("Barreiras BA", -12.15, -44.99),
    ("Vitória da Conquista BA", -14.86, -40.84), ("Caravelas BA", -17.71, -39.25),
    ("Bom Jesus da Lapa BA", -13.26, -43.42), ("Remanso BA", -9.62, -42.08),
    ("Palmas TO", -10.24, -48.36), ("Porto Nacional TO", -10.71, -48.42),
    ("Araguaína TO", -7.19, -48.21), ("Cuiabá MT", -15.60, -56.10),
    ("Cáceres MT", -16.07, -57.68), ("Sinop MT", -11.86, -55.51),
    ("Campo Grande MS", -20.44, -54.65), ("Corumbá MS", -19.01, -57.65),
    ("Dourados MS", -22.22, -54.81), ("Goiânia GO", -16.69, -49.26),
    ("Rio Verde GO", -17.80, -50.93), ("Formosa GO", -15.54, -47.33),
    ("Brasília DF", -15.79, -47.88), ("Montes Claros MG", -16.73, -43.86),
    ("Belo Horizonte MG", -19.92, -43.94), ("Uberaba MG", -19.75, -47.93),
    ("Juiz de Fora MG", -21.76, -43.35), ("Diamantina MG", -18.24, -43.60),
    ("Vitória ES", -20.32, -40.34), ("Rio de Janeiro RJ", -22.91, -43.17),
    ("Campos RJ", -21.75, -41.33), ("Resende RJ", -22.47, -44.45),
    ("São Paulo SP", -23.55, -46.63), ("Campinas SP", -22.91, -47.06),
    ("Presidente Prudente SP", -22.13, -51.39), ("Franca SP", -20.54, -47.40),
    ("Santos SP", -23.96, -46.33), ("Curitiba PR", -25.43, -49.27),
    ("Londrina PR", -23.30, -51.17), ("Foz do Iguaçu PR", -25.55, -54.59),
    ("Florianópolis SC", -27.60, -48.55), ("Chapecó SC", -27.10, -52.62),
    ("Porto Alegre RS", -30.03, -51.23), ("Santa Maria RS", -29.68, -53.81),
    ("Pelotas RS", -31.77, -52.34), ("Uruguaiana RS", -29.76, -57.09),
    ("Bagé RS", -31.33, -54.11), ("Passo Fundo RS", -28.26, -52.41),
]

sys.path.insert(0, str(HERE))
from roni_winter_forecast import (_ensure_ghcnm, GHCN_URL)   # noqa: E402,F401


def roni_history() -> pd.Series:
    """DJF-mean RONI per summer start year (Dec), from ERSST v5."""
    ds = xr.open_dataset(ERSST)
    sst = ds["sst"].sortby("lat")
    clim = (sst.sel(time=slice(f"{CLIM0}-01-01", f"{CLIM1}-12-31"))
            .groupby("time.month").mean("time"))
    anom = sst.groupby("time.month") - clim

    def wmean(da):
        w = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(w).mean(("lat", "lon"), skipna=True)

    n34 = wmean(anom.sel(lat=slice(-5, 5), lon=slice(190, 240)))
    trop = wmean(anom.sel(lat=slice(-20, 20)))
    roni_m = (n34 - trop).rolling(time=3, center=True, min_periods=2).mean()
    s = roni_m.to_series()
    out = {}
    for y in range(Y0, Y1 + 1):
        months = [pd.Timestamp(y, 12, 1),
                  pd.Timestamp(y + 1, 1, 1), pd.Timestamp(y + 1, 2, 1)]
        vals = [s.get(m, np.nan) for m in months]
        if np.isfinite(vals).sum() >= 2:
            out[y] = float(np.nanmean(vals))
    return pd.Series(out)


def _pick_stations(inv_path: Path) -> list[dict]:
    stns = []
    for ln in inv_path.read_text().splitlines():
        sid = ln[:11]
        if not sid.startswith("BR"):
            continue
        stns.append((sid, float(ln[12:20]), float(ln[21:30]), ln[38:68].strip()))
    sarr = np.array([[s[1], s[2]] for s in stns])
    picks = []
    for name, clat, clon in CITIES:
        d2 = (sarr[:, 0] - clat) ** 2 + ((sarr[:, 1] - clon)
                                         * np.cos(np.deg2rad(clat))) ** 2
        order = np.argsort(d2)
        picks.append(dict(city=name, lat=clat, lon=clon,
                          cand=[stns[i] for i in order[:15]],
                          cand_d=np.sqrt(d2[order[:15]])))
    return picks


def city_summers() -> tuple[pd.DataFrame, pd.DataFrame]:
    dat_path, inv_path = _ensure_ghcnm()
    picks = _pick_stations(inv_path)

    wanted = {c for p in picks for c, *_ in p["cand"]}
    series: dict[str, dict] = {sid: {} for sid in wanted}
    with open(dat_path) as f:
        for ln in f:
            sid = ln[:11]
            if sid not in series:
                continue
            year = int(ln[11:15])
            if not (Y0 - 1 <= year <= Y1 + 1):
                continue
            for m in range(12):
                v = int(ln[19 + m * 8: 24 + m * 8])
                if v != -9999:
                    series[sid][(year, m + 1)] = v / 100.0

    def summer_vals(sid):
        s = series.get(sid, {})
        out = {}
        for y in range(Y0, Y1 + 1):
            vals = [s.get((y, 12)), s.get((y + 1, 1)), s.get((y + 1, 2))]
            good = [v for v in vals if v is not None]
            if len(good) >= 2:                      # >=2 of 3 DJF months (BR gaps)
                out[y] = float(np.mean(good))
        return pd.Series(out, dtype=float)

    cols, meta = {}, []
    for p in picks:
        best, best_ser, best_d = None, None, None
        for (sid, slat, slon, sname), d in zip(p["cand"], p["cand_d"]):
            ser = summer_vals(sid)
            if len(ser) < MIN_WINTERS or ser.index.max() < MIN_END:
                continue
            if len(ser.loc[CLIM0:CLIM1]) < 12:
                continue
            starts_early = ser.index.min() <= 1910
            if best is None or (starts_early and not best[4]) or \
               (starts_early == best[4] and len(ser) > len(best_ser)):
                best, best_ser, best_d = (sid, slat, slon, sname, starts_early), ser, d
                if starts_early and d <= 0.8:
                    break
        if best is None:
            print(f"  !! no usable station for {p['city']} — dropped")
            continue
        sid, slat, slon, sname, early = best
        cols[p["city"]] = best_ser - best_ser.loc[CLIM0:CLIM1].mean()   # °C
        meta.append(dict(city=p["city"], lat=p["lat"], lon=p["lon"],
                         station=sid, station_name=sname,
                         rec_start=int(best_ser.index.min()),
                         rec_end=int(best_ser.index.max()),
                         n_summers=len(best_ser), dist_deg=round(float(best_d), 2)))
    return pd.DataFrame(cols), pd.DataFrame(meta)


def fit_and_forecast(anom, meta, roni, roni_now):
    out = []
    for _, row in meta.iterrows():
        name = row["city"]
        ser = anom[name].dropna()
        years = ser.index.intersection(roni.index)
        y = ser.loc[years].to_numpy()
        X_year = (years - 2000).to_numpy(dtype=float)
        X_hinge = np.clip(years.to_numpy(dtype=float) - HINGE_YEAR, 0, None)
        X_roni = roni.loc[years].to_numpy()
        A = np.column_stack([np.ones_like(X_year), X_year, X_hinge, X_roni])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        dof = len(y) - 4
        s2 = (resid @ resid) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        p_roni = 2 * stats.t.sf(abs(coef[3] / np.sqrt(cov[3, 3])), dof)
        r = np.corrcoef(A @ coef, y)[0, 1]
        c_used = coef[3] if p_roni < P_SIG else 0.0
        trend_part = (coef[0] + coef[1] * (TARGET - 2000)
                      + coef[2] * (TARGET - HINGE_YEAR))
        fc = trend_part + c_used * roni_now
        out.append(dict(city=name, lat=row["lat"], lon=row["lon"],
                        forecast_C=fc, trend_C=trend_part,
                        roni_C=c_used * roni_now, roni_coef=coef[3],
                        roni_p=p_roni,
                        trend_per_decade_modern=(coef[1] + coef[2]) * 10,
                        fit_r=r, station=row["station"],
                        station_name=row["station_name"],
                        rec_start=row["rec_start"], rec_end=row["rec_end"],
                        n_summers=len(y)))
    return pd.DataFrame(out)


def draw_map(fc: pd.DataFrame, roni_now: float, out_path: Path):
    proj = ccrs.AlbersEqualArea(central_longitude=-54, central_latitude=-14,
                                standard_parallels=(-5, -25))
    fig = plt.figure(figsize=(10.5, 10.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-74.5, -34, -34.5, 5.8], crs=ccrs.PlateCarree())

    from scipy.interpolate import RBFInterpolator
    gx, gy = np.meshgrid(np.arange(-75, -33.9, 0.25), np.arange(-35, 6.1, 0.25))
    scale = np.cos(np.deg2rad(-14.0))
    pts = fc[["lon", "lat"]].to_numpy() * [scale, 1.0]
    rbf = RBFInterpolator(pts, fc["forecast_C"].to_numpy(),
                          kernel="thin_plate_spline", smoothing=2.0)
    gz = rbf(np.column_stack([gx.ravel() * scale, gy.ravel()])).reshape(gx.shape)

    vmax = max(1.0, np.ceil(np.abs(fc["forecast_C"]).max() * 4) / 4)
    levels = np.arange(-vmax, vmax + 0.125, 0.25)
    cf = ax.contourf(gx, gy, gz, levels=levels, cmap="RdBu_r", extend="both",
                     transform=ccrs.PlateCarree(), alpha=0.85)
    cl = ax.contour(gx, gy, gz, levels=levels[::2], colors="#444",
                    linewidths=0.5, transform=ccrs.PlateCarree())
    ax.clabel(cl, fmt=lambda v: f"{v:+.1f}", fontsize=7)

    import cartopy.io.shapereader as shpreader
    from matplotlib.path import Path as MplPath
    from cartopy.mpl.patch import geos_to_path
    shp = shpreader.natural_earth(resolution="110m", category="cultural",
                                  name="admin_0_countries")
    bra = next(r.geometry for r in shpreader.Reader(shp).records()
               if r.attributes["ADM0_A3"] == "BRA")
    clip = MplPath.make_compound_path(*geos_to_path(bra))
    tr = ccrs.PlateCarree()._as_mpl_transform(ax)
    for cs in (cf, cl):
        for art in (getattr(cs, "collections", None) or [cs]):
            art.set_clip_path(clip, transform=tr)

    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.35, edgecolor="#777")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.6, edgecolor="#333")
    ax.coastlines("50m", lw=0.6, color="#333")

    sig = fc["roni_C"] != 0.0
    ax.scatter(fc.lon[sig], fc.lat[sig], s=11, c="k",
               transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(fc.lon[~sig], fc.lat[~sig], s=13, facecolors="none",
               edgecolors="k", lw=0.6, transform=ccrs.PlateCarree(), zorder=5)

    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.045,
                      pad=0.03, aspect=42, shrink=0.75)
    cb.set_label("DJF 2-m temperature anomaly vs 1991–2020 (°C)", fontsize=10)
    ax.set_title(f"Summer 2026–27 (DJF) outlook — RONI + climate trend "
                 f"(hinge {HINGE_YEAR}) — assumed RONI {roni_now:+.1f} °C",
                 fontsize=12, loc="left", pad=8)
    fig.text(0.015, 0.012,
             "Per-city OLS on GHCN-M v4 adjusted station observations (DJF means; "
             "winters through 2025/26 where reported):\n"
             "anomaly = a + b·year + b₂·max(0, year−1970) + c·RONI (ERSST-derived, "
             "from 1900). RONI kept only where significant at 90% (filled dots; "
             "open = trend only).\nStatistical outlook, not a dynamical forecast.",
             fontsize=7.5, color="#555")
    fig.savefig(out_path, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roni", type=float, default=None)
    ap.add_argument("--out-dir", default=str(HERE / "plots"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roni_now = args.roni
    if roni_now is None:
        roni_now = float(json.loads(MANIFEST.read_text())["daily_roni"])
        print(f"assumed DJF RONI from site manifest: {roni_now:+.2f}")

    roni = roni_history()
    print(f"RONI summers: {roni.index.min()}–{roni.index.max()}")
    anom, meta = city_summers()
    early = int((meta.rec_start <= 1910).sum())
    print(f"stations: {len(meta)} cities matched; {early} with records from "
          f"≤1910 (earliest {meta.rec_start.min()}); "
          f"median record {int(meta.n_summers.median())} summers")

    fc = fit_and_forecast(anom, meta, roni, roni_now)
    nsig = int((fc.roni_C != 0).sum())
    print(f"RONI term significant at {nsig}/{len(fc)} cities; "
          f"range {fc.forecast_C.min():+.2f} … {fc.forecast_C.max():+.2f} °C")

    fc.round(3).to_csv(out_dir / "roni_summer_brazil_2026.csv", index=False)
    draw_map(fc, roni_now, out_dir / "roni_summer_brazil_2026.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
