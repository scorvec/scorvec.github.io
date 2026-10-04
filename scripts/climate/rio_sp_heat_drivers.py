#!/usr/bin/env python3
"""
What actually drives Rio / São Paulo summer heatwaves? — driver attribution.

Data: ERA5-Land daily Tmax (c3s project cache, 1959–2026), DJF summers
1959/60–2025/26. Heat metrics per city-summer:
  hw_days    days with Tmax above the city's full-record DJF 90th percentile
  hw3        hottest 3-day-mean Tmax of the summer (peak heatwave)
  tmax_mean  DJF mean Tmax

Candidate drivers (ERSST DJF means, all RELATIVE indices where conventional):
  TNA, TSA, SWATL (local offshore box), ATL3, SASD, RONI, east-lean, DMI.

Everything is hinge-1970 detrended before analysis, so drivers compete on
variability, not on shared warming. Statistics: pairwise r (+p), partial r
controlling all other drivers, LMG variance shares for the top-4, leave-one-
year-out single-driver skill, warm/cold composites, and the same suite for
LAGGED (ASO) drivers — the forecast-relevant version.

Output: ~/rio_sp_heat_drivers.png + printed tables.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
ERSST = HERE.parent / "sst" / "data" / "ersst_v5_mnmean.nc"
CSV = Path.home() / "c3s" / "data" / "f16" / "city_era5land_daily.csv"

Y0, Y1 = 1959, 2025
CITIES = ["Rio", "Sao Paulo"]

BOXES = {
    "TNA": ((5.5, 23.5), [(302.5, 345)], False),
    "TSA": ((-20, 0), [(330, 360), (0, 10)], False),
    "SWATL": ((-35, -25), [(310, 330)], False),      # offshore Rio/Santos
    "ATL3": ((-3, 3), [(340, 360)], False),
    "RONI": ((-5, 5), [(190, 240)], True),           # relative (minus tropics)
    "N12rel": ((-10, 0), [(270, 280)], True),
    "N4rel": ((-5, 5), [(160, 210)], True),
    "DMI_W": ((-10, 10), [(50, 70)], False),
    "DMI_E": ((-10, 0), [(90, 110)], False),
    "SASD_SW": ((-40, -30), [(330, 350)], False),
    "SASD_NE": ((-25, -15), [(340, 360)], False),
}


def monthly_indices() -> pd.DataFrame:
    ds = xr.open_dataset(ERSST)
    sst = ds["sst"].sortby("lat")
    clim = (sst.sel(time=slice("1991-01-01", "2020-12-31"))
            .groupby("time.month").mean("time"))
    anom = sst.groupby("time.month") - clim

    def wm(da):
        w = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(w).mean(("lat", "lon"), skipna=True)

    trop = wm(anom.sel(lat=slice(-20, 20))).to_series()
    out = {}
    for name, (lat, lons, relative) in BOXES.items():
        parts = [anom.sel(lat=slice(*lat), lon=slice(*lo)) for lo in lons]
        s = wm(xr.concat(parts, dim="lon")).to_series()
        out[name] = s - trop if relative else s
    df = pd.DataFrame(out)
    df["elean"] = df["N12rel"] - df["N4rel"]
    df["DMI"] = df["DMI_W"] - df["DMI_E"]
    df["SASD"] = df["SASD_SW"] - df["SASD_NE"]
    return df.drop(columns=["N12rel", "N4rel", "DMI_W", "DMI_E",
                            "SASD_SW", "SASD_NE"])


def season_mean(s: pd.Series, y: int, months) -> float:
    vals = [s.get(pd.Timestamp(y if m >= 9 else y + 1, m, 1), np.nan)
            for m in months]
    return float(np.nanmean(vals))


def hinge_detrend(s: pd.Series) -> pd.Series:
    x = s.index.to_numpy(float)
    A = np.column_stack([np.ones_like(x), x - 2000, np.clip(x - 1970, 0, None)])
    m = np.isfinite(s.values)
    c, *_ = np.linalg.lstsq(A[m], s.values[m], rcond=None)
    return pd.Series(s.values - A @ c, index=s.index)


def partial_r(y, x, Z):
    """r(y, x | Z) via residuals."""
    def resid(v):
        A = np.column_stack([np.ones(len(v)), Z])
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ c
    return float(np.corrcoef(resid(y), resid(x))[0, 1])


def lmg(y, X: pd.DataFrame):
    """Average ΔR² over all orderings (LMG) for each column of X."""
    cols = list(X.columns)
    shares = {c: [] for c in cols}
    for perm in itertools.permutations(cols):
        r2_prev = 0.0
        for i in range(1, len(perm) + 1):
            A = np.column_stack([np.ones(len(y)), X[list(perm[:i])].values])
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = y - A @ c
            r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
            shares[perm[i - 1]].append(r2 - r2_prev)
            r2_prev = r2
    return {c: float(np.mean(v)) for c, v in shares.items()}


def loyo_r(y: pd.Series, x: pd.Series) -> float:
    idx = y.index
    preds = []
    for yy in idx:
        tr = idx[idx != yy]
        b = np.polyfit(x[tr], y[tr], 1)
        preds.append(np.polyval(b, x[yy]))
    return float(np.corrcoef(preds, y.values)[0, 1])


def main() -> int:
    d = pd.read_csv(CSV, index_col=0, parse_dates=True)
    mon = monthly_indices()

    # per-summer heat metrics
    metrics = {}
    for city in CITIES:
        tmax = d[f"{city}_tmax"]
        djf_all = tmax[tmax.index.month.isin([12, 1, 2])]
        p90 = float(djf_all.quantile(0.90))
        rows = {}
        for y in range(Y0, Y1 + 1):
            sel = tmax[(tmax.index >= f"{y}-12-01") & (tmax.index <= f"{y+1}-02-28")]
            if len(sel) < 80:
                continue
            r3 = sel.rolling(3).mean()
            rows[y] = dict(hw_days=int((sel > p90).sum()),
                           hw3=float(r3.max()), tmax_mean=float(sel.mean()))
        metrics[city] = pd.DataFrame(rows).T
        print(f"{city}: {len(rows)} summers, DJF p90 = {p90:.1f} °C")

    # driver seasonal means: concurrent DJF and lagged ASO
    DRIVERS = ["TNA", "TSA", "SWATL", "ATL3", "SASD", "RONI", "elean", "DMI"]
    idx_djf = pd.DataFrame({k: {y: season_mean(mon[k], y, (12, 1, 2))
                                for y in range(Y0, Y1 + 1)} for k in DRIVERS})
    idx_aso = pd.DataFrame({k: {y: season_mean(mon[k], y, (9, 10, 11))
                                for y in range(Y0, Y1 + 1)} for k in DRIVERS})
    # (months 9,10,11 of the init year = SON just before the summer)

    results = {}
    for city in CITIES:
        M = metrics[city]
        res = {}
        for tag, idx in (("DJF", idx_djf), ("SON", idx_aso)):
            common = M.index.intersection(idx.dropna().index)
            Xd = idx.loc[common].apply(hinge_detrend)
            for met in ("hw_days", "hw3", "tmax_mean"):
                yv = hinge_detrend(M.loc[common, met].astype(float))
                row = {}
                for k in DRIVERS:
                    r, p = stats.pearsonr(Xd[k], yv)
                    row[k] = dict(r=round(r, 3), p=round(p, 3))
                # partials controlling all other drivers
                for k in DRIVERS:
                    Z = Xd[[c for c in DRIVERS if c != k]].values
                    row[k]["pr"] = round(partial_r(yv.values, Xd[k].values, Z), 3)
                # LOYO single-driver skill
                for k in DRIVERS:
                    row[k]["loyo"] = round(loyo_r(yv, Xd[k]), 3)
                res[(tag, met)] = row
        results[city] = res

        # dominance for the top-4 (by |r|) on hw_days, both timings
        for tag, idx in (("DJF", idx_djf), ("SON", idx_aso)):
            common = M.index.intersection(idx.dropna().index)
            Xd = idx.loc[common].apply(hinge_detrend)
            yv = hinge_detrend(M.loc[common, "hw_days"].astype(float)).values
            top4 = sorted(DRIVERS, key=lambda k: -abs(res[(tag, "hw_days")][k]["r"]))[:4]
            sh = lmg(yv, Xd[top4])
            tot = 1 - np.var(yv - np.column_stack(
                [np.ones(len(yv)), Xd[top4].values]) @ np.linalg.lstsq(
                np.column_stack([np.ones(len(yv)), Xd[top4].values]), yv,
                rcond=None)[0]) / np.var(yv)
            print(f"\n{city} hw_days [{tag}] top-4 LMG shares "
                  f"(model R²={tot:.2f}): " +
                  "  ".join(f"{k}={sh[k]:.3f}" for k in top4))

    # ── print main tables ──
    for city in CITIES:
        for tag in ("DJF", "SON"):
            print(f"\n=== {city} — drivers ({tag}) — detrended ===")
            print(f"{'driver':7s} " + " | ".join(
                f"{m:>22s}" for m in ("hw_days", "hw3", "tmax_mean")))
            for k in DRIVERS:
                cells = []
                for met in ("hw_days", "hw3", "tmax_mean"):
                    v = results[city][(tag, met)][k]
                    cells.append(f"r{v['r']:+.2f} pr{v['pr']:+.2f} cv{v['loyo']:+.2f}")
                print(f"{k:7s} " + " | ".join(f"{c:>22s}" for c in cells))

    # ── figure: detrended r bars per city (DJF concurrent + SON lagged) ──
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), dpi=150, sharey=True)
    for i, city in enumerate(CITIES):
        for j, tag in enumerate(("DJF", "SON")):
            ax = axes[i][j]
            w = 0.26
            xs = np.arange(len(DRIVERS))
            for k3, met in enumerate(("hw_days", "hw3", "tmax_mean")):
                vals = [results[city][(tag, met)][k]["r"] for k in DRIVERS]
                sig = [results[city][(tag, met)][k]["p"] < 0.05 for k in DRIVERS]
                bars = ax.bar(xs + (k3 - 1) * w, vals, w,
                              label=met if (i == 0 and j == 0) else None,
                              color=["#d9402a", "#e8890c", "#2b6fd6"][k3],
                              alpha=0.85)
                for b, s in zip(bars, sig):
                    if s:
                        ax.annotate("*", (b.get_x() + b.get_width() / 2,
                                          b.get_height()),
                                    ha="center", fontsize=11,
                                    va="bottom" if b.get_height() >= 0 else "top")
            ax.set_xticks(xs)
            ax.set_xticklabels(DRIVERS, fontsize=8.5, rotation=30)
            ax.axhline(0, color="#333", lw=0.8)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(f"{city} — {'concurrent DJF' if tag=='DJF' else 'preceding SON (lagged)'}",
                         fontsize=11, loc="left")
            if j == 0:
                ax.set_ylabel("detrended Pearson r", fontsize=10)
    axes[0][0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("What drives Rio / São Paulo summer heat? — detrended driver "
                 "correlations, 1959/60–2025/26 (* p<0.05)",
                 fontsize=13, x=0.03, ha="left")
    fig.text(0.012, 0.008,
             "ERA5-Land daily Tmax; hinge-1970 detrending on metrics and drivers; "
             "SWATL = local offshore box 25–35°S, 50–30°W.",
             fontsize=7.5, color="#666")
    out = Path.home() / "rio_sp_heat_drivers.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
