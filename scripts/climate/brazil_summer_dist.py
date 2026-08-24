#!/usr/bin/env python3
"""
Distributions from the v2 Brazil summer forecast: national population-
weighted temperature and key cities.

Fig 1 (~/brazil_summer_dist_national.png):
  · histogram of ERA5 NDJFM pop-weighted temperature, 1950/51–2025/26,
    with the v2 forecast, trailing normals and analog summers marked
  · forecast anomaly vs 30/10/5-yr normals by month (bars)
  · pooled C3S member distribution (10 systems, NDJ pop-weighted anomaly,
    anchored to ERA5 1993–2016) vs the statistical and blended values

Fig 2 (~/brazil_summer_dist_cities.png): per-city NDJFM histograms
  (nearest ERA5 2° cell) with the v2 forecast line and its rank.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.path.expanduser("~/c3s/scripts"))
import brazil_summer_fcst_v2 as v2                           # noqa: E402
import common as C                                           # noqa: E402
import f4_lib as F                                           # noqa: E402

MONTHS = v2.MONTHS
MNAME = v2.MNAME
CITIES = {
    "São Paulo": (-23.55, -46.63), "Rio de Janeiro": (-22.91, -43.17),
    "Belo Horizonte": (-19.92, -43.94), "Brasília": (-15.79, -47.88),
    "Salvador": (-12.97, -38.50), "Recife": (-8.05, -34.88),
    "Manaus": (-3.10, -60.02), "Porto Alegre": (-30.03, -51.23),
}
ANALOGS = {1982: "82/83", 1997: "97/98", 2015: "15/16", 2023: "23/24"}


def series_ndjfm(t2, t, w):
    """Weighted NDJFM mean per event year from an ERA5 monthly cube."""
    flat = t2.values.reshape(len(t), -1) @ w.ravel()
    s = pd.Series(flat, index=t)
    out = {}
    for y in range(1950, 2026):
        stamps = [pd.Timestamp(y, 11, 1), pd.Timestamp(y, 12, 1)] + \
                 [pd.Timestamp(y + 1, m, 1) for m in (1, 2, 3)]
        v = [s.get(x, np.nan) for x in stamps]
        if np.isfinite(v).sum() == 5:
            out[y] = float(np.mean(v))
    return pd.Series(out)


def main() -> int:
    d = v2.compute()
    lat_e, lon_e, t2 = d["lat"], d["lon"], d["t2"]
    t = pd.DatetimeIndex(t2["time"].values)
    w = C.pop_grid(lat_e, lon_e, country="BR")
    w = w / w.sum()

    def wavg(field):
        m = np.isfinite(field) & (w > 0)
        return float((field[m] * w[m]).sum() / w[m].sum())

    fc_nat = {m: wavg(d["final_abs"][m]) for m in MONTHS}
    stat_nat = {m: wavg(d["stat_abs"][m]) for m in MONTHS}
    norm_nat = {nb: {m: wavg(d["normals"][nb][m]) for m in MONTHS}
                for nb in (30, 10, 5)}
    clim_nat = {m: wavg(d["clim9316"][m]) for m in MONTHS}
    hist = series_ndjfm(t2, t, w)
    fc_ndjfm = float(np.mean([fc_nat[m] for m in MONTHS]))
    print(f"national NDJFM forecast {fc_ndjfm:.2f} °C; history "
          f"{hist.min():.2f}–{hist.max():.2f}, warmest {hist.idxmax()}")

    # C3S member NDJ pop-weighted anomalies, pooled across systems
    ndj = [11, 12, 1]
    mem_pool = []
    for mdl in F.models_present("sa_fc"):
        ds = F.load_sa(mdl)
        wlat, wlon = ds["latitude"].values, ds["longitude"].values
        wm_ = C.pop_grid(wlat, wlon, country="BR")
        wm_ = wm_ / wm_.sum()
        steps = [v2.DYN_MONTHS[m] for m in ndj]
        fcv = ds["fc_t2m"].isel(step=steps).values      # (num, 3, lat, lon)
        hcv = ds["hc_t2m"].isel(step=steps).values.mean(0)
        an = (fcv - hcv[None]).mean(1)                  # (num, lat, lon)
        mem_pool.extend((an.reshape(an.shape[0], -1) @ wm_.ravel()).tolist())
    mem_pool = np.array(mem_pool)
    ndj_clim = float(np.mean([clim_nat[m] for m in ndj]))
    ndj_norm10 = float(np.mean([norm_nat[10][m] for m in ndj]))
    mem_vs10 = mem_pool + ndj_clim - ndj_norm10
    stat_vs10 = float(np.mean([stat_nat[m] - norm_nat[10][m] for m in ndj]))
    blend_vs10 = float(np.mean([fc_nat[m] - norm_nat[10][m] for m in ndj]))
    print(f"C3S members pooled: {len(mem_pool)}; NDJ vs 10-yr: "
          f"mean {mem_vs10.mean():+.2f}, stat {stat_vs10:+.2f}, "
          f"blend {blend_vs10:+.2f}")

    # ── Fig 1: national ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), dpi=150)
    ax = axes[0]
    ax.hist(hist.values, bins=24, color="#b8cbe0", edgecolor="#5b7ea6",
            lw=0.6)
    for nb, cc in ((30, "#7f7f7f"), (10, "#5a5a5a"), (5, "#333333")):
        vv = float(np.mean([norm_nat[nb][m] for m in MONTHS]))
        ax.axvline(vv, color=cc, lw=1.1, ls="--")
        ax.text(vv, ax.get_ylim()[1] * 0.97, f"{nb}y", ha="center",
                fontsize=8, color=cc)
    for y, lb in ANALOGS.items():
        if y in hist.index:
            ax.axvline(hist[y], color="#e8890c", lw=1.0, alpha=0.8)
            ax.text(hist[y], ax.get_ylim()[1] * 0.80, lb, rotation=90,
                    fontsize=7.5, color="#a35f05", ha="right")
    ax.axvline(fc_ndjfm, color="#d9402a", lw=2.5)
    rank = int((hist.values >= fc_ndjfm).sum()) + 1
    ax.text(fc_ndjfm - 0.05, ax.get_ylim()[1] * 0.55,
            f"2026/27 forecast\n{fc_ndjfm:.2f} °C (#{rank} of "
            f"{len(hist) + 1})  ", color="#d9402a", fontsize=9,
            fontweight="bold", ha="right")
    ax.set_xlabel("NDJFM pop-weighted mean temperature (°C)")
    ax.set_ylabel("summers (1950/51–2025/26)")
    ax.set_title("Brazil national, population-weighted", loc="left",
                 fontsize=11)

    ax = axes[1]
    xs = np.arange(len(MONTHS))
    for i, (nb, cc) in enumerate(((30, "#c86b52"), (10, "#a63a22"),
                                  (5, "#701f0f"))):
        vals = [fc_nat[m] - norm_nat[nb][m] for m in MONTHS]
        ax.bar(xs + (i - 1) * 0.27, vals, 0.27, color=cc,
               label=f"vs {nb}-yr normal")
    ax.set_xticks(xs)
    ax.set_xticklabels([MNAME[m].split()[0] for m in MONTHS])
    ax.axhline(0, color="#333", lw=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5)
    ax.set_ylabel("forecast anomaly (°C)")
    ax.set_title("Monthly forecast vs trailing normals", loc="left",
                 fontsize=11)

    ax = axes[2]
    ax.hist(mem_vs10, bins=28, color="#c9b8e0", edgecolor="#7a5ba6", lw=0.6,
            density=True)
    ax.axvline(float(mem_vs10.mean()), color="#5b3e8f", lw=1.8,
               label=f"C3S MME mean {mem_vs10.mean():+.2f}")
    ax.axvline(stat_vs10, color="#2b6fd6", lw=1.8, ls="--",
               label=f"statistical {stat_vs10:+.2f}")
    ax.axvline(blend_vs10, color="#d9402a", lw=2.5,
               label=f"final blend {blend_vs10:+.2f}")
    ax.axvline(0, color="#333", lw=0.8)
    ax.legend(fontsize=8.5)
    ax.set_xlabel("NDJ anomaly vs 10-yr normal (°C)")
    ax.set_ylabel("member density")
    ax.set_title(f"C3S member spread (n={len(mem_pool)}, 10 systems)",
                 loc="left", fontsize=11)
    fig.suptitle("Brazil 2026/27 summer — national temperature outlook",
                 fontsize=13, x=0.02, ha="left")
    fig.text(0.01, -0.02,
             "ERA5 2° grid, GeoNames city populations ≥15k as weights. "
             "Forecast = v2 blend (stat + C3S for Nov–Jan).",
             fontsize=8, color="#555")
    out1 = Path.home() / "brazil_summer_dist_national.png"
    fig.savefig(out1, facecolor="white", bbox_inches="tight",
                pad_inches=0.2)
    plt.close(fig)
    print(f"wrote {out1}")

    # ── Fig 2: cities ──
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), dpi=150)
    for i, (city, (la, lo)) in enumerate(CITIES.items()):
        ax = axes[i // 4][i % 4]
        ii = int(np.abs(lat_e - la).argmin())
        jj = int(np.abs(lon_e - lo).argmin())
        cw = np.zeros((len(lat_e), len(lon_e)))
        cw[ii, jj] = 1.0
        h = series_ndjfm(t2, t, cw)
        fcv = float(np.mean([d["final_abs"][m][ii, jj] for m in MONTHS]))
        n5 = float(np.mean([d["normals"][5][m][ii, jj] for m in MONTHS]))
        ax.hist(h.values, bins=20, color="#b8cbe0", edgecolor="#5b7ea6",
                lw=0.5)
        ax.axvline(n5, color="#555", lw=1.0, ls="--")
        ax.axvline(fcv, color="#d9402a", lw=2.2)
        rank = int((h.values >= fcv).sum()) + 1
        ax.set_title(f"{city}", loc="left", fontsize=10.5)
        ax.text(0.02, 0.95,
                f"fcst {fcv:.1f} °C (#{rank}/{len(h) + 1})\n"
                f"vs 5-yr normal {fcv - n5:+.1f}",
                transform=ax.transAxes, va="top", fontsize=8.5,
                color="#a32315")
        ax.set_yticks([])
    fig.suptitle("City NDJFM temperature — history (1950/51–2025/26) vs "
                 "2026/27 forecast (red; dashed = 5-yr normal)",
                 fontsize=13, x=0.02, ha="left")
    fig.text(0.01, -0.015,
             "Nearest ERA5 2° cell per city — regional means, not station "
             "values.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = Path.home() / "brazil_summer_dist_cities.png"
    fig.savefig(out2, facecolor="white", bbox_inches="tight",
                pad_inches=0.2)
    plt.close(fig)
    print(f"wrote {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
