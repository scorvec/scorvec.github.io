#!/usr/bin/env python3
"""
State-based population weighting for the national summer temperature —
comparison against the city-cell weighting.

Method A (current): GeoNames city populations dropped on their nearest
0.25° cell. Method B (this test): area-mean temperature over each of the
27 UFs (Natural Earth admin-1 polygons), states weighted by IBGE 2022
census population.

Outputs ~/brazil_summer_state_weight.png:
  1. city-vs-state national NDJFM series scatter (1950/51–2025/26)
  2. both historical distributions + 2026/27 forecast under each method
  3. per-state forecast anomaly vs 5-yr normal, ranked, pop-labelled
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
from brazil_summer_dist import series_ndjfm                  # noqa: E402

MONTHS = v2.MONTHS

POP = {  # IBGE 2022 census, millions
    "SP": 44.41, "MG": 20.54, "RJ": 16.06, "BA": 14.14, "PR": 11.44,
    "RS": 10.88, "PE": 9.06, "CE": 8.79, "PA": 8.12, "SC": 7.61,
    "GO": 7.06, "MA": 6.78, "PB": 3.97, "AM": 3.94, "ES": 3.83,
    "MT": 3.66, "RN": 3.30, "PI": 3.27, "AL": 3.13, "DF": 2.82,
    "MS": 2.76, "SE": 2.21, "RO": 1.58, "TO": 1.51, "AC": 0.83,
    "AP": 0.73, "RR": 0.64,
}


def state_masks(lat, lon):
    """UF code → boolean mask on the (lat, lon) grid."""
    import cartopy.io.shapereader as shpreader
    import shapely
    shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                  name="admin_1_states_provinces")
    lon2, lat2 = np.meshgrid(lon, lat)
    masks = {}
    for rec in shpreader.Reader(shp).records():
        if rec.attributes.get("adm0_a3") != "BRA":
            continue
        uf = (rec.attributes.get("postal")
              or rec.attributes.get("iso_3166_2", "").split("-")[-1])
        if uf not in POP:
            continue
        m = shapely.contains_xy(shapely.geometry.shape(rec.geometry),
                                lon2.ravel(), lat2.ravel())
        masks[uf] = m.reshape(lat2.shape)
    missing = set(POP) - set(masks)
    if missing:
        print(f"WARNING: no polygon for {sorted(missing)}")
    return masks


def main() -> int:
    d = v2.compute()
    lat_e, lon_e, t2 = d["lat"], d["lon"], d["t2"]
    t = pd.DatetimeIndex(t2["time"].values)
    coslat = np.cos(np.deg2rad(lat_e))[:, None]

    # method A: city cells
    wA = C.pop_grid(lat_e, lon_e, country="BR")
    wA = wA / wA.sum()

    # method B: state means × state population
    masks = state_masks(lat_e, lon_e)
    wB = np.zeros_like(wA)
    for uf, m in masks.items():
        area = (coslat * m).sum()
        if area > 0:
            wB += POP[uf] * (coslat * m) / area
    wB = wB / wB.sum()
    print(f"states with polygons: {len(masks)}; "
          f"cells carrying weight — city: {(wA > 0).sum()}, "
          f"state: {(wB > 0).sum()}")

    def natfc(w):
        def wavg(f):
            mm = np.isfinite(f) & (w > 0)
            return float((f[mm] * w[mm]).sum() / w[mm].sum())
        fc = np.mean([wavg(d["final_abs"][m]) for m in MONTHS])
        n5 = np.mean([wavg(d["normals"][5][m]) for m in MONTHS])
        n30 = np.mean([wavg(d["normals"][30][m]) for m in MONTHS])
        return fc, fc - n5, fc - n30

    histA = series_ndjfm(t2, t, wA)
    histB = series_ndjfm(t2, t, wB)
    fcA, dA5, dA30 = natfc(wA)
    fcB, dB5, dB30 = natfc(wB)
    r = float(np.corrcoef(histA.values, histB.values)[0, 1])
    ra = float(np.corrcoef(histA.values - histA.rolling(10).mean().values,
                           histB.values - histB.rolling(10).mean().values
                           )[0, 1]) if len(histA) > 10 else np.nan
    print(f"city:  fc {fcA:.2f}  vs5 {dA5:+.2f}  vs30 {dA30:+.2f}")
    print(f"state: fc {fcB:.2f}  vs5 {dB5:+.2f}  vs30 {dB30:+.2f}")
    print(f"historical series r={r:.3f}")

    # per-state forecast anomalies
    rows = {}
    for uf, m in masks.items():
        w = (coslat * m) / max((coslat * m).sum(), 1e-9)
        fcs, n5s = [], []
        for mo in MONTHS:
            f = d["final_abs"][mo]
            fin = np.isfinite(f) & m
            if fin.sum() == 0:
                break
            ww = (coslat * fin) / (coslat * fin).sum()
            fcs.append(float((f * ww)[fin].sum()))
            n5s.append(float((d["normals"][5][mo] * ww)[fin].sum()))
        if fcs:
            rows[uf] = dict(anom5=np.mean(fcs) - np.mean(n5s), pop=POP[uf])
    tab = pd.DataFrame(rows).T.sort_values("anom5", ascending=False)

    # ── figure ──
    fig = plt.figure(figsize=(17, 5.6), dpi=150)
    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(histA.values, histB.values, s=14, color="#2b6fd6", alpha=0.7)
    lims = [min(histA.min(), histB.min()) - 0.2,
            max(histA.max(), histB.max()) + 0.2]
    ax.plot(lims, lims, color="#999", lw=0.8)
    ax.plot([fcA], [fcB], "*", ms=18, color="#d9402a")
    ax.text(fcA, fcB, "  2026/27", color="#d9402a", fontsize=9,
            fontweight="bold", va="center")
    ax.set_xlabel("city-cell weighting (°C)")
    ax.set_ylabel("state-aggregate weighting (°C)")
    ax.set_title(f"National NDJFM, two weightings (r={r:.3f})",
                 loc="left", fontsize=11)

    ax = fig.add_subplot(1, 3, 2)
    ax.hist(histA.values, bins=22, alpha=0.55, color="#2b6fd6",
            label="city-cell")
    ax.hist(histB.values, bins=22, alpha=0.55, color="#5ba67a",
            label="state-aggregate")
    ax.axvline(fcA, color="#2b6fd6", lw=2.2)
    ax.axvline(fcB, color="#1e7a4b", lw=2.2, ls="--")
    ax.text(fcA, ax.get_ylim()[1] * 0.9,
            f" city fc {fcA:.2f} ({dA5:+.2f} vs 5y)", color="#2b6fd6",
            fontsize=8.5, ha="right", rotation=90, va="top")
    ax.text(fcB, ax.get_ylim()[1] * 0.9,
            f" state fc {fcB:.2f} ({dB5:+.2f} vs 5y)", color="#1e7a4b",
            fontsize=8.5, rotation=90, va="top")
    ax.legend(fontsize=8.5)
    ax.set_xlabel("NDJFM pop-weighted temperature (°C)")
    ax.set_title("Distributions + 2026/27 forecast", loc="left", fontsize=11)

    ax = fig.add_subplot(1, 3, 3)
    cols = plt.cm.YlOrRd(0.25 + 0.75 * (tab["anom5"] - tab["anom5"].min())
                         / (tab["anom5"].max() - tab["anom5"].min()))
    ax.barh(np.arange(len(tab))[::-1], tab["anom5"], color=cols,
            edgecolor="#888", lw=0.4)
    ax.set_yticks(np.arange(len(tab))[::-1])
    ax.set_yticklabels([f"{uf} ({p:.1f}M)" for uf, p in
                        zip(tab.index, tab["pop"])], fontsize=6.6)
    ax.set_xlabel("forecast NDJFM anomaly vs 5-yr normal (°C)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("By state (area mean)", loc="left", fontsize=11)

    fig.suptitle("Brazil national temperature — city-cell vs state-aggregate "
                 "population weighting", fontsize=13, x=0.02, ha="left")
    fig.text(0.01, -0.03,
             "State means over Natural Earth admin-1 polygons on the ERA5 "
             "0.25° grid; IBGE 2022 census populations.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path.home() / "brazil_summer_state_weight.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    print(f"wrote {out}")
    print(tab.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
