#!/usr/bin/env python3
"""Global AAM in El Nino and La Nina from GFZ's effective angular momentum functions (the explainer's opening figure).

GFZ ESMGFZ AAM v1.0 (Dobslaw et al., 2010): 3-hourly chi3 motion term from ECMWF analyses, independent of the ERA5
pipeline. Wind-term AAM M_r = (chi3 + GFZ_CHI3_MEAN) * Omega C / alpha3 (aam.py's constants: files issued 2023+ print
a wrong subtracted-mean header, so the constant is always added). Sources, in order:
    ~/mjo/aam_validation/esmgfz_Mr_daily.csv   daily M_r 1976-2025 (aam.py's validation cache)
    ~/mjo/aam_validation/esmgfz_{year}.asc     the yearly 3-hourly files (aam.py's cache; --fetch reads them from IERS)
ENSO phase from CPC's official ONI (assets/sst/data/cpc_official.json); RONI as a sensitivity test.

Method: monthly means (complete months only), anomalies from the 1991-2020 monthly means, the 1979-2025 linear trend
removed (as for the ERA5 series in torque_enso_study.py), then a centred 3-month mean to match the ONI's seasons.
1976-78 are dropped: GFZ sits ~0.45e25 above ERA5 then, against +/-0.15 after 1979 (which record is wrong is unknown).
Tests: 24-month moving-block bootstrap of the paired series (2,000 draws), as for the study's monthly regressions;
the correlation's p also from a t-test with Bretherton et al. (1999) effective n.

Writes scripts/mjo/data/reference/torque_enso_study/gfz_enso_series.csv and gfz_enso.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import torque_enso_study as S                                           # noqa: E402

CACHE = Path.home() / "mjo" / "aam_validation"
URL = "https://datacenter.iers.org/products/geofluids/atmosphere/eamf/GGFC2010/GFZ/ESMGFZ_AAM_v1.0_03h_{year}.asc"
CHI3_MEAN = 2.734436408977633e-8                                         # aam.py GFZ_CHI3_MEAN
TO_KGM2S = 5.2092e33                                                     # C_m * Omega
OUT = S.REF / "torque_enso_study"
Y0, TREND = 1979, ("1979-01", "2025-12")
NB, BLOCK, SEED = 2000, 24, 20261013


def read_asc(text: str) -> pd.Series:
    rows = []
    for line in text.splitlines():
        c = line.split()
        if len(c) >= 11 and c[0].isdigit() and len(c[0]) == 4:
            rows.append((pd.Timestamp(int(c[0]), int(c[1]), int(c[2])), float(c[-1])))
    d = pd.DataFrame(rows, columns=["date", "chi3"])
    n = d.groupby("date").size()
    daily = d.groupby("date").chi3.mean()[n == 8]                        # whole days of eight 3-hourly values
    return (daily + CHI3_MEAN) * TO_KGM2S / 1e25


def load_gfz(fetch: bool) -> pd.Series:
    """Daily wind-term AAM (1e25 kg m2 s-1), 1976 to the latest whole day."""
    years = range(1976, pd.Timestamp.utcnow().year + 1)
    parts = []
    if fetch:
        import requests
        for y in years:
            r = requests.get(URL.format(year=y), timeout=60, headers={"User-Agent": "scorvec.com research (torque study)"})
            r.raise_for_status()
            parts.append(read_asc(r.text))
    else:
        csv = pd.read_csv(CACHE / "esmgfz_Mr_daily.csv", parse_dates=["date"]).set_index("date")["Mr_kg_m2_s"] / 1e25
        parts.append(csv)
        for y in years:
            f = CACHE / f"esmgfz_{y}.asc"
            if f.exists() and y > csv.index.max().year:
                parts.append(read_asc(f.read_text()))
    g = pd.concat(parts).sort_index()
    return g[~g.index.duplicated(keep="last")]


def monthly_complete(d: pd.Series) -> pd.Series:
    m = d.resample("MS").mean()
    n = d.resample("MS").count()
    return m[n == m.index.days_in_month]


def official(key: str) -> pd.Series:
    j = json.loads((S.ROOT / "assets/sst/data/cpc_official.json").read_text())[key]
    return pd.Series({pd.Timestamp(k + "-01"): v for k, v in j.items()}).sort_index()


def anomalies(m: pd.Series, detrend: bool = True) -> pd.Series:
    base = m.loc["1991":"2020"]
    a = m - m.index.month.map(base.groupby(base.index.month).mean()).values
    if detrend:
        t = (a.index.year + (a.index.month - 0.5) / 12).values
        w = (a.index >= TREND[0]) & (a.index <= TREND[1] + "-28")
        b = np.polyfit(t[w], a.values[w], 1)
        a = a - np.polyval(b, t)
    return a


def r1(x):
    x = np.asarray(x) - np.mean(x)
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


def stats_of(aam3: pd.Series, idx: pd.Series) -> dict:
    df = pd.concat([aam3.rename("aam"), idx.rename("oni")], axis=1).dropna()
    df = df.loc[f"{Y0}-01":]
    x, y = df.oni.values, df.aam.values
    n = len(df)
    ph = np.where(x >= 0.5, "EN", np.where(x <= -0.5, "LN", "NEU"))

    def summ(xx, yy):
        p_ = np.where(xx >= 0.5, "EN", np.where(xx <= -0.5, "LN", "NEU"))
        out = {k: float(yy[p_ == k].mean()) for k in ("EN", "NEU", "LN")}
        sl = np.polyfit(xx, yy, 1)[0]
        return np.array([out["EN"], out["NEU"], out["LN"], out["EN"] - out["LN"], sl, np.corrcoef(xx, yy)[0, 1]])
    est = summ(x, y)
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(n / BLOCK))
    bs = np.empty((NB, est.size))
    for i in range(NB):
        st = rng.integers(0, n - BLOCK + 1, nb)
        ii = (st[:, None] + np.arange(BLOCK)).ravel()[:n]
        bs[i] = summ(x[ii], y[ii])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    pboot = [float(min(1.0, 2 * min((bs[:, k] <= 0).mean(), (bs[:, k] >= 0).mean()) + 1 / NB)) for k in range(est.size)]
    r = est[5]
    a1, a2 = r1(x), r1(y)
    neff = n * (1 - a1 * a2) / (1 + a1 * a2)
    tt = r * np.sqrt((neff - 2) / (1 - r * r))
    keys = ["EN", "NEU", "LN", "EN_minus_LN", "slope_per_K", "r"]
    res = {k: dict(est=float(est[j]), lo=float(lo[j]), hi=float(hi[j]), p=pboot[j]) for j, k in enumerate(keys)}
    res["r"]["p_bretherton"] = float(2 * stats.t.sf(abs(tt), neff - 2))
    res["r"]["n_eff"] = float(neff)
    res["n_months"] = int(n)
    res["n_by_phase"] = {k: int((ph == k).sum()) for k in ("EN", "NEU", "LN")}
    res["share_EN_above"] = float((y[ph == "EN"] > 0).mean())
    res["share_LN_below"] = float((y[ph == "LN"] < 0).mean())
    res["span"] = [df.index[0].strftime("%Y-%m"), df.index[-1].strftime("%Y-%m")]
    # lagged correlation: positive lag = AAM after the ONI
    res["lag_r"] = {int(L): float(df.oni.corr(aam3.shift(-L).reindex(df.index))) for L in range(-6, 7)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="read the yearly files from IERS (Actions) instead of the cache")
    a = ap.parse_args()
    g = load_gfz(a.fetch)
    m = monthly_complete(g)
    e = S.load_era5()["aam_glob"]
    em = monthly_complete(e)
    both = pd.concat([m.rename("gfz"), em.rename("era5")], axis=1).dropna()
    off = (both.gfz - both.era5).resample("YS").mean()
    an = anomalies(m)
    aam3 = an.rolling(3, center=True).mean()
    aam3[an.rolling(3, center=True).count() < 3] = np.nan
    oni, roni = official("oni"), official("roni")
    res = dict(source="GFZ ESMGFZ AAM v1.0, chi3 motion term (ECMWF analyses), daily means of 3-hourly values",
               gfz_last_day=g.index.max().strftime("%Y-%m-%d"), last_complete_month=m.index.max().strftime("%Y-%m"),
               oni=stats_of(aam3, oni), roni=stats_of(aam3, roni),
               oni_no_detrend=stats_of(anomalies(m, False).rolling(3, center=True).mean(), oni),
               gfz_minus_era5_annual={str(k.year): round(float(v), 3) for k, v in off.items()},
               corr_monthly_anom_with_era5=float(anomalies(both.gfz).corr(anomalies(both.era5))),
               lod_ms_per_1e25=1e25 / TO_KGM2S * 86400 * 1e3)
    ser = pd.DataFrame({"aam_anom": an, "aam3": aam3}).join(oni.rename("oni")).join(roni.rename("roni"))
    ser = ser.loc[f"{Y0}-01":]
    ser.index.name = "month"
    OUT.mkdir(parents=True, exist_ok=True)
    ser.round(4).to_csv(OUT / "gfz_enso_series.csv")
    (OUT / "gfz_enso.json").write_text(json.dumps(res, indent=1))
    o = res["oni"]
    print(json.dumps({k: o[k] for k in ("EN", "NEU", "LN", "EN_minus_LN", "slope_per_K", "r", "n_by_phase",
                                         "share_EN_above", "share_LN_below", "span")}, indent=1))
    print("lag r", {k: round(v, 3) for k, v in o["lag_r"].items()})
    print("roni slope", res["roni"]["slope_per_K"], "no-detrend slope", res["oni_no_detrend"]["slope_per_K"])
    print("corr with ERA5", res["corr_monthly_anom_with_era5"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
