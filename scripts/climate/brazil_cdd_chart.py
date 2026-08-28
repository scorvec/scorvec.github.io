#!/usr/bin/env python3
"""
Brazil DJF cooling-degree-days, load-weighted: history + the RONI/trend forecast.

CDD base 18 °C on monthly means (fine in Brazil where summer means sit far
above base almost everywhere): DJF CDD = Σ_month max(0, T_mon − 18) · n_days.

Weights reflect actual load, not station geography: metro population per city,
rescaled so each ONS subsystem's total weight matches its share of national
load (SE/CW 58 %, NE 18 %, S 17 %, N 7 %). Each year's national CDD is the
weight-renormalised mean over cities reporting that year (as an anomaly vs the
city's 1991–2020 normal, re-expressed in CDD by adding the weighted normal).

The 2026/27 bar applies each city's forecast DJF anomaly (from
roni_summer_brazil — RONI + hinge trend) to its monthly normals.

    python brazil_cdd_chart.py [--roni 2.75]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from roni_summer_brazil import (_ensure_ghcnm, _pick_stations, CITIES,   # noqa: E402
                                CLIM0, CLIM1, MIN_END, MIN_WINTERS)

BASE = 18.0
Y0, Y1 = 1950, 2025
TARGET = 2026
NDAYS = {12: 31, 1: 31, 2: 28.25}
FC_CSV = HERE / "plots" / "roni_summer_brazil_2026.csv"
OUT_PNG = HERE / "plots" / "brazil_cdd_djf.png"

# (metro population, ONS subsystem) per city; pop in millions, approximate —
# it only needs to be right in the large and to rank cities sensibly.
SUBSYS_SHARE = {"SE": 0.58, "S": 0.17, "NE": 0.18, "N": 0.07}
CITY_LOAD = {
    "Boa Vista RR": (0.4, "N"), "Macapá AP": (0.5, "N"), "Belém PA": (2.2, "N"),
    "Santarém PA": (0.3, "N"), "Marabá PA": (0.3, "N"), "Altamira PA": (0.1, "N"),
    "Manaus AM": (2.7, "N"), "Tefé AM": (0.06, "N"),
    "Cruzeiro do Sul AC": (0.09, "N"), "Rio Branco AC": (0.4, "N"),
    "Porto Velho RO": (0.5, "N"), "Vilhena RO": (0.1, "N"),
    "São Luís MA": (1.6, "NE"), "Imperatriz MA": (0.3, "NE"),
    "Caxias MA": (0.16, "NE"), "Teresina PI": (1.2, "NE"),
    "Floriano PI": (0.06, "NE"), "Fortaleza CE": (4.2, "NE"),
    "Quixeramobim CE": (0.08, "NE"), "Natal RN": (1.6, "NE"),
    "João Pessoa PB": (1.3, "NE"), "Campina Grande PB": (0.4, "NE"),
    "Recife PE": (4.2, "NE"), "Petrolina PE": (0.4, "NE"),
    "Maceió AL": (1.4, "NE"), "Aracaju SE": (1.0, "NE"),
    "Salvador BA": (3.9, "NE"), "Barreiras BA": (0.16, "NE"),
    "Vitória da Conquista BA": (0.34, "NE"), "Caravelas BA": (0.02, "NE"),
    "Bom Jesus da Lapa BA": (0.07, "NE"), "Remanso BA": (0.04, "NE"),
    "Palmas TO": (0.3, "N"), "Porto Nacional TO": (0.05, "N"),
    "Araguaína TO": (0.18, "N"), "Cuiabá MT": (1.0, "SE"),
    "Cáceres MT": (0.09, "SE"), "Sinop MT": (0.15, "SE"),
    "Campo Grande MS": (0.9, "SE"), "Corumbá MS": (0.1, "SE"),
    "Dourados MS": (0.23, "SE"), "Goiânia GO": (2.7, "SE"),
    "Rio Verde GO": (0.24, "SE"), "Formosa GO": (0.12, "SE"),
    "Brasília DF": (3.9, "SE"), "Montes Claros MG": (0.42, "SE"),
    "Belo Horizonte MG": (6.0, "SE"), "Uberaba MG": (0.34, "SE"),
    "Juiz de Fora MG": (0.57, "SE"), "Diamantina MG": (0.05, "SE"),
    "Vitória ES": (2.0, "SE"), "Rio de Janeiro RJ": (12.6, "SE"),
    "Campos RJ": (0.51, "SE"), "Resende RJ": (0.13, "SE"),
    "São Paulo SP": (21.7, "SE"), "Campinas SP": (3.3, "SE"),
    "Presidente Prudente SP": (0.23, "SE"), "Franca SP": (0.36, "SE"),
    "Santos SP": (1.8, "SE"), "Curitiba PR": (3.7, "S"),
    "Londrina PR": (0.58, "S"), "Foz do Iguaçu PR": (0.26, "S"),
    "Florianópolis SC": (1.2, "S"), "Chapecó SC": (0.22, "S"),
    "Porto Alegre RS": (4.3, "S"), "Santa Maria RS": (0.28, "S"),
    "Pelotas RS": (0.34, "S"), "Uruguaiana RS": (0.13, "S"),
    "Bagé RS": (0.12, "S"), "Passo Fundo RS": (0.2, "S"),
}


def load_weights(cities: list[str]) -> pd.Series:
    pop = pd.Series({c: CITY_LOAD[c][0] for c in cities})
    sub = pd.Series({c: CITY_LOAD[c][1] for c in cities})
    w = pop.copy()
    for s, share in SUBSYS_SHARE.items():
        m = sub == s
        if m.any():
            w[m] = pop[m] / pop[m].sum() * share
    return w / w.sum()


def monthly_series() -> dict[str, dict]:
    """{city: {(year, month): T}} for the same stations the forecast used."""
    dat_path, inv_path = _ensure_ghcnm()
    fc = pd.read_csv(FC_CSV)
    sid_by_city = dict(zip(fc.city, fc.station))
    wanted = set(sid_by_city.values())
    raw: dict[str, dict] = {sid: {} for sid in wanted}
    with open(dat_path) as f:
        for ln in f:
            sid = ln[:11]
            if sid not in raw:
                continue
            year = int(ln[11:15])
            if not (Y0 - 1 <= year <= Y1 + 1):
                continue
            for m in range(12):
                v = int(ln[19 + m * 8: 24 + m * 8])
                if v != -9999:
                    raw[sid][(year, m + 1)] = v / 100.0
    return {c: raw[s] for c, s in sid_by_city.items()}, fc


def djf_cdd(s: dict, y: int) -> float | None:
    tot = 0.0
    for (yy, mm) in [(y, 12), (y + 1, 1), (y + 1, 2)]:
        t = s.get((yy, mm))
        if t is None:
            return None
        tot += max(0.0, t - BASE) * NDAYS[mm]
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roni", type=float, default=2.75)
    args = ap.parse_args()

    series, fc = monthly_series()
    cities = list(fc.city)
    w = load_weights(cities)

    # per-city CDD history + 1991–2020 normals
    cdd = pd.DataFrame({c: {y: djf_cdd(series[c], y) for y in range(Y0, Y1 + 1)}
                        for c in cities})
    normals = cdd.loc[CLIM0:CLIM1].mean()
    panel_normal = float((normals * w).sum())

    # weighted national series: anomaly over reporting cities, renormalised
    nat = {}
    for y in cdd.index:
        row = cdd.loc[y]
        m = row.notna() & normals.notna()
        if w[m].sum() < 0.6:                      # require 60 % of load weight
            continue
        nat[y] = float(((row - normals)[m] * w[m]).sum() / w[m].sum()) + panel_normal
    nat = pd.Series(nat).sort_index()

    # forecast: monthly normals per city shifted by the forecast DJF anomaly
    def _mn(c, mm):
        vals = [series[c].get((yy + (1 if mm != 12 else 0), mm))
                for yy in range(CLIM0, CLIM1 + 1)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else np.nan
    mon_norm = {c: {mm: _mn(c, mm) for mm in (12, 1, 2)} for c in cities}
    fc_anom = dict(zip(fc.city, fc.forecast_C))
    fc_cdd = {}
    for c in cities:
        tot = 0.0
        for mm in (12, 1, 2):
            t = mon_norm[c][mm] + fc_anom[c]
            tot += max(0.0, t - BASE) * NDAYS[mm]
        fc_cdd[c] = tot
    fc_nat = float((pd.Series(fc_cdd) * w).sum())

    rank = int((nat > fc_nat).sum()) + 1
    print(f"panel normal {panel_normal:.0f} CDD; forecast {fc_nat:.0f} "
          f"(rank {rank} vs {len(nat)} observed summers)")

    # ── chart ──
    fig, ax = plt.subplots(figsize=(13, 5.4), dpi=150)
    colors = np.where(nat.values >= panel_normal, "#d9402a", "#2b6fd6")
    ax.bar(nat.index, nat.values - panel_normal, bottom=panel_normal,
           color=colors, alpha=0.75, width=0.8)
    ax.bar([TARGET], [fc_nat - panel_normal], bottom=panel_normal,
           color="#7a0018", width=0.8, hatch="//", edgecolor="white")
    ax.axhline(panel_normal, color="#333", lw=0.9)
    top5 = nat.nlargest(5)
    for y, v in top5.items():
        ax.annotate(str(y), (y, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7, color="#555")
    ax.annotate(f"2026–27\nforecast: {fc_nat:.0f}\n(#{rank} of {len(nat)+1})",
                (TARGET, panel_normal + (fc_nat - panel_normal) * 0.45),
                textcoords="offset points", xytext=(-118, -10), fontsize=9,
                color="#7a0018", fontweight="bold", ha="left")
    ax.set_ylabel("DJF cooling degree days (base 18 °C)", fontsize=10)
    ax.set_xlim(int(nat.index.min()) - 1, TARGET + 2)
    lo, hi = nat.min(), max(nat.max(), fc_nat)
    ax.set_ylim(lo - 25, hi + 30)
    ax.set_title("Brazil load-weighted summer CDDs — observed 1950/51–2025/26 "
                 f"and the RONI+trend 2026/27 forecast (RONI {args.roni:+.1f})",
                 fontsize=12, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.012, 0.012,
             "GHCN-M v4 station DJF means → CDD base 18 °C · city weights = metro "
             "population scaled to ONS subsystem load shares (SE/CW 58 %, NE 18 %, "
             "S 17 %, N 7 %) · years shown only when ≥60 % of load weight reports · "
             "forecast = per-city RONI+hinge-trend anomaly applied to monthly normals.",
             fontsize=7, color="#666")
    fig.savefig(OUT_PNG, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
