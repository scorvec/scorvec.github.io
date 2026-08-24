#!/usr/bin/env python3
"""
Brazil load-weighted NDJFM CDDs — season total and per-month breakdown.

Same machinery as brazil_cdd_chart.py, widened to Nov–Mar and refit PER
CALENDAR MONTH: for each city and month, T-anomaly = a + b·year +
b₂·max(0, year−1970) + c·RONI(month) — so the ENSO sensitivity is allowed
to differ between November and March (it does). Forecast temp anomalies
convert to CDDs through each city's monthly normal (base 18 °C).

Outputs (per request, to the home folder):
    ~/brazil_cdd_ndjfm.png       season-total bars, 1950s→2025 + 2026/27
    ~/brazil_cdd_by_month.png    five panels, Nov…Mar

    python brazil_cdd_ndjfm.py [--roni 2.75]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from roni_summer_brazil import _ensure_ghcnm, CLIM0, CLIM1     # noqa: E402
from brazil_cdd_chart import load_weights, CITY_LOAD           # noqa: E402
from scipy import stats                                        # noqa: E402

BASE = 18.0
Y0, Y1 = 1950, 2025            # summer label = Nov year
TARGET = 2026
MONTHS = [11, 12, 1, 2, 3]
MNAME = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}
NDAYS = {11: 30, 12: 31, 1: 31, 2: 28.25, 3: 31}
P_SIG = 0.10
HINGE = 1970
FC_CSV = HERE / "plots" / "roni_summer_brazil_2026.csv"
ERSST = HERE.parent / "sst" / "data" / "ersst_v5_mnmean.nc"


def roni_monthly() -> pd.Series:
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
    return (n34 - trop).rolling(time=3, center=True, min_periods=2).mean().to_series()


def month_year(y_label: int, m: int) -> pd.Timestamp:
    yy = y_label if m >= 11 else y_label + 1
    return pd.Timestamp(yy, m, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roni", type=float, default=2.75)
    args = ap.parse_args()

    fc = pd.read_csv(FC_CSV)
    cities = list(fc.city)
    sid_by_city = dict(zip(fc.city, fc.station))
    w = load_weights(cities)
    roni_m = roni_monthly()

    dat_path, _ = _ensure_ghcnm()
    wanted = set(sid_by_city.values())
    raw: dict[str, dict] = {sid: {} for sid in wanted}
    with open(dat_path) as f:
        for ln in f:
            sid = ln[:11]
            if sid not in raw:
                continue
            year = int(ln[11:15])
            if not (Y0 - 1 <= year <= Y1 + 2):
                continue
            for m in range(12):
                v = int(ln[19 + m * 8: 24 + m * 8])
                if v != -9999:
                    raw[sid][(year, m + 1)] = v / 100.0

    years = np.arange(Y0, Y1 + 1)
    # observed monthly CDD panels + per-city/month fits → forecast
    obs = {m: {} for m in MONTHS}          # month -> year -> weighted CDD
    fc_cdd = {m: {} for m in MONTHS}       # month -> city -> forecast CDD
    norm_cdd = {m: {} for m in MONTHS}
    nsig = 0
    for c in cities:
        s = raw[sid_by_city[c]]
        for m in MONTHS:
            t = pd.Series({y: s.get((month_year(y, m).year, m)) for y in years},
                          dtype=float).dropna()
            base_t = t.loc[CLIM0:CLIM1]
            if len(base_t) < 12:
                norm_cdd[m][c] = np.nan
                fc_cdd[m][c] = np.nan
                continue
            norm = base_t.mean()
            norm_cdd[m][c] = max(0.0, norm - BASE) * NDAYS[m]
            a = t - norm
            rn = pd.Series({y: roni_m.get(month_year(y, m), np.nan)
                            for y in a.index})
            mm = np.isfinite(a.values) & np.isfinite(rn.values)
            yv, yr = a.values[mm], a.index.to_numpy()[mm].astype(float)
            X = np.column_stack([np.ones_like(yr), yr - 2000,
                                 np.clip(yr - HINGE, 0, None), rn.values[mm]])
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            resid = yv - X @ coef
            dof = len(yv) - 4
            cov = (resid @ resid) / dof * np.linalg.inv(X.T @ X)
            p = 2 * stats.t.sf(abs(coef[3] / np.sqrt(cov[3, 3])), dof)
            c_used = coef[3] if p < P_SIG else 0.0
            nsig += int(c_used != 0)
            anom_fc = (coef[0] + coef[1] * (TARGET - 2000)
                       + coef[2] * (TARGET - HINGE) + c_used * args.roni)
            fc_cdd[m][c] = max(0.0, norm + anom_fc - BASE) * NDAYS[m]
            # observed CDDs for the history panels
            for y, tv in t.items():
                obs[m].setdefault(y, {})[c] = max(0.0, tv - BASE) * NDAYS[m]

    # weighted national series per month (anomaly-renormalised like the DJF chart)
    nat_m, fc_m, norm_m = {}, {}, {}
    for m in MONTHS:
        nc = pd.Series(norm_cdd[m])
        ok = nc.notna()
        norm_m[m] = float((nc[ok] * w[ok]).sum() / w[ok].sum())
        ser = {}
        for y, d in obs[m].items():
            row = pd.Series(d).reindex(cities)
            mm2 = row.notna() & nc.notna()
            if w[mm2].sum() < 0.6:
                continue
            ser[y] = float(((row - nc)[mm2] * w[mm2]).sum() / w[mm2].sum()) + norm_m[m]
        nat_m[m] = pd.Series(ser).sort_index()
        fr = pd.Series(fc_cdd[m])
        mm3 = fr.notna()
        fc_m[m] = float(((fr - nc)[mm3] * w[mm3]).sum() / w[mm3].sum()) + norm_m[m]

    total = pd.concat(nat_m, axis=1).dropna().sum(axis=1)
    fc_total = sum(fc_m.values())
    panel_norm = sum(norm_m.values())
    rank = int((total > fc_total).sum()) + 1
    print(f"NDJFM normal {panel_norm:.0f}; forecast {fc_total:.0f} "
          f"(rank {rank} of {len(total)+1}); sig month-fits {nsig}")
    for m in MONTHS:
        r = int((nat_m[m] > fc_m[m]).sum()) + 1
        print(f"  {MNAME[m]}: normal {norm_m[m]:.0f}  forecast {fc_m[m]:.0f} "
              f"(rank {r}/{len(nat_m[m])+1})")

    home = Path.home()

    # ── season-total chart ──
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=150)
    colors = np.where(total.values >= panel_norm, "#d9402a", "#2b6fd6")
    ax.bar(total.index, total.values - panel_norm, bottom=panel_norm,
           color=colors, alpha=0.75, width=0.8)
    ax.bar([TARGET], [fc_total - panel_norm], bottom=panel_norm,
           color="#7a0018", width=0.8, hatch="//", edgecolor="white")
    ax.axhline(panel_norm, color="#333", lw=0.9)
    for y, v in total.nlargest(4).items():
        ax.annotate(str(y), (y, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7, color="#555")
    ax.annotate(f"2026–27 forecast: {fc_total:.0f}  (#{rank} of {len(total)+1})",
                (TARGET - 1, fc_total), ha="right", fontsize=9.5,
                color="#7a0018", fontweight="bold")
    ax.set_ylabel("NDJFM cooling degree days (base 18 °C)", fontsize=10)
    ax.set_xlim(int(total.index.min()) - 1, TARGET + 2)
    ax.set_ylim(total.min() - 40, max(total.max(), fc_total) + 40)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Brazil load-weighted NDJFM CDDs — observed and the RONI+trend "
                 f"2026/27 forecast (RONI {args.roni:+.1f})",
                 fontsize=12, loc="left", pad=8)
    fig.text(0.012, 0.012,
             "GHCN-M v4 stations · CDD base 18 °C · metro-population weights scaled "
             "to ONS subsystem load shares · per-city, PER-MONTH RONI+hinge fits · "
             "years with ≥60 % load-weight coverage in every month.",
             fontsize=7, color="#666")
    p1 = home / "brazil_cdd_ndjfm.png"
    fig.savefig(p1, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {p1}")

    # ── monthly breakdown ──
    fig, axes = plt.subplots(5, 1, figsize=(12.5, 13.5), dpi=150, sharex=True)
    for ax, m in zip(axes, MONTHS):
        s, fv, nv = nat_m[m], fc_m[m], norm_m[m]
        colors = np.where(s.values >= nv, "#d9402a", "#2b6fd6")
        ax.bar(s.index, s.values - nv, bottom=nv, color=colors, alpha=0.75,
               width=0.8)
        ax.bar([TARGET], [fv - nv], bottom=nv, color="#7a0018", width=0.8,
               hatch="//", edgecolor="white")
        ax.axhline(nv, color="#333", lw=0.8)
        r = int((s > fv).sum()) + 1
        ax.set_ylabel(f"{MNAME[m]} CDD", fontsize=9.5)
        ax.text(0.008, 0.90, f"{MNAME[m]}: normal {nv:.0f} · forecast {fv:.0f} "
                f"(#{r} of {len(s)+1})", transform=ax.transAxes, fontsize=9,
                color="#7a0018", fontweight="bold", va="top")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(int(total.index.min()) - 1, TARGET + 2)
    axes[0].set_title("Brazil load-weighted CDDs by month — observed and the "
                      f"2026/27 forecast (RONI {args.roni:+.1f})",
                      fontsize=12, loc="left", pad=8)
    fig.text(0.012, 0.005,
             "Same weighting and per-month fits as the season chart; month belongs "
             "to the summer labelled by its November.", fontsize=7, color="#666")
    p2 = home / "brazil_cdd_by_month.png"
    fig.savefig(p2, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {p2}")

    # ── scatter: NDJFM CDD vs NDJFM RONI, forecast starred ──
    roni_s = pd.Series({y: np.nanmean([roni_m.get(month_year(y, m), np.nan)
                                       for m in MONTHS]) for y in total.index})
    both = pd.concat([total.rename("cdd"), roni_s.rename("roni")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=150)
    sc = ax.scatter(both.roni, both.cdd, c=both.index, cmap="viridis", s=42,
                    edgecolor="#333", lw=0.4, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("summer (Nov year)", fontsize=9)
    b, a = np.polyfit(both.roni, both.cdd, 1)
    xs = np.linspace(both.roni.min() - 0.3, max(both.roni.max(), args.roni) + 0.3, 50)
    ax.plot(xs, a + b * xs, color="#888", lw=1.2, ls="--", zorder=2,
            label=f"OLS: {b:+.0f} CDD per RONI °C (raw, trend not removed)")
    ax.scatter([args.roni], [fc_total], marker="*", s=560, color="#7a0018",
               edgecolor="white", lw=1.2, zorder=5)
    ax.annotate(f"2026–27 expectation\nRONI {args.roni:+.1f}, {fc_total:.0f} CDD",
                (args.roni, fc_total), textcoords="offset points",
                xytext=(-150, -12), fontsize=10, color="#7a0018",
                fontweight="bold")
    for y in both.index:
        if both.cdd[y] >= both.cdd.nlargest(4).min() or both.roni[y] >= both.roni.nlargest(3).min():
            ax.annotate(str(y), (both.roni[y], both.cdd[y]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7,
                        color="#555")
    ax.set_xlabel("NDJFM-mean RONI (°C)", fontsize=10)
    ax.set_ylabel("NDJFM load-weighted CDD (base 18 °C)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Brazil load-weighted NDJFM CDDs vs RONI — observed summers "
                 "and the 2026/27 expectation", fontsize=12, loc="left", pad=8)
    fig.text(0.012, 0.008,
             "Colour = year: the upward drift at every RONI is the climate trend, "
             "which the forecast model carries separately from the RONI term.",
             fontsize=7.5, color="#666")
    p3 = home / "brazil_cdd_vs_roni.png"
    fig.savefig(p3, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {p3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
