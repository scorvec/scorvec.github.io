#!/usr/bin/env python3
"""Does a forecast model's eddy feedback, and its stratosphere-to-surface coupling, match ERA5 on the same days? The
GEFSv12 reforecast's 35-day runs (gefs_eddy_extract.py; 11 members, Wednesdays October-April, 2000-2019) against ERA5
(eddy_extract.py).

ANOMALIES. GEFS: each lead day against the reforecast's own climate at that lead (three annual harmonics in the valid
date, fitted over every init and member), so the model's drift drops out. ERA5: three harmonics over 1999-2020.

MEASURES, pooled over members, inits and lead days; z and m are the jet index and its eddy forcing on ERA5's leading
EOF of daily DJF u anomalies at 250 + 500 hPa, 20-87.5N, 1979-2022 (eddy_feedback.jet); cap is the 65-90N sea-level
pressure anomaly (hPa, + = negative NAM); u100 is u(60N, 100 hPa).
  b      Lorenz & Hartmann (2001): cov(z(t), m(t + L)) / var(z(t)) over L = 7-14 days, t = lead days 5-20 valid in
         December-February (also t = 5-12 and 13-20 apart), per day
  down   the mean cap over days t+1..t+15 regressed on u100(t), with cap(t) as a second predictor (persistence),
         t = lead days 1-19 valid in December-March, hPa per m/s; down_sd in standard deviations
  snr    weeks 3-4 (lead days 15-28) mean cap and z, inits whose window centres on December-February: the ensemble
         mean's correlation r with ERA5, the ratio of predictable components RPC = r / (sd of the ensemble mean / sd of
         single members) (Eade et al. 2014; RPC > 1 = the model's signal too weak for its noise), and the slope of ERA5
         on the ensemble mean (> 1 likewise); every init, and those starting with a weak (strong) lower stratosphere:
         ERA5's u100 anomaly on the init day below -1 (above +1) standard deviation
ERA5 "matched": the same measure from ERA5 on exactly the forecast days, so both sample the same weather.
Uncertainty: 1000 bootstrap draws of whole winters (an init's winter is that of its October); one-sided p for the
difference (GEFS weaker), and for RPC > 1.

    python eddy_feedback_gefs.py --gefs gefs/ --era5 era5/ --out reference/eddy_feedback_gefs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cmip6_coupling as CC
import eddy_feedback as EF

TROP = (250, 500)
B_T, B_LAGS = (5, 20), np.arange(7, 15)
DOWN_T, DOWN_NEXT = (1, 19), 15
WEEKS34 = (15, 28)
ERA5_CLIM = (1999, 2020)
EOF_YEARS = (1979, 2022)
BOOT = 1000
RNG = np.random.default_rng(20261108)


def harmonics(dates: pd.DatetimeIndex) -> np.ndarray:
    frac, _ = CC.year_fraction(dates.year, dates.month, dates.day, "proleptic_gregorian")
    return CC.design(frac, dates.year.values.astype(float), 3, False)


def fit_remove(y: np.ndarray, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """y (n, ...) less its least-squares fit on X (n, k), fitted over `rows` (finite columns only)."""
    Y = y.reshape(len(y), -1)
    out = np.full(Y.shape, np.nan)
    ok = rows[:, None] & np.isfinite(Y)
    for c in range(Y.shape[1]):
        r = ok[:, c]
        if r.sum() > X.shape[1] * 5:
            out[:, c] = Y[:, c] - X @ np.linalg.lstsq(X[r], Y[r, c], rcond=None)[0]
    return out.reshape(y.shape)


# --------------------------------------------------------------------------------------------------------- loading ---
def load_gefs(d: Path) -> dict:
    zs = [np.load(p) for p in sorted(d.glob("GEFSv12_*.npz")) if "test" not in p.name and ".part" not in p.name]
    g = {k: np.concatenate([z[k] for z in zs]) for k in ("init", "u", "uv", "slp")}
    g["init"] = pd.DatetimeIndex(g["init"])
    o = np.argsort(g["init"].values)
    g = {k: v[o] for k, v in g.items()}
    z0 = zs[0]
    g.update(days=z0["days"].astype(int), u_levels=[int(x) for x in z0["u_levels"]],
             uv_levels=[int(x) for x in z0["uv_levels"]], lat=z0["lat"].astype(float), members=list(z0["members"]))  # fmt: skip
    return g


def derived(u, uv, slp, ulev, uvlev, lat) -> dict:
    """The daily series every measure uses, from either source (trailing axes level, lat)."""
    ku = [ulev.index(L) for L in TROP]
    kv = [uvlev.index(L) for L in TROP]
    j60 = int(np.argmin(abs(lat - 60)))
    return {
        "ut": u[..., ku, :].mean(-2),
        "f": EF.accel(uv[..., kv, :].mean(-2).astype(float), lat),
        "cap": EF.area_mean(slp.astype(float), lat, EF.CAP),
        "u100": u[..., ulev.index(100), j60].astype(float),
    }


# --------------------------------------------------------------------------------------------------------- measures ---
def b_parts(z, m, valid, winter_of, t_range) -> tuple[np.ndarray, list]:
    """Per winter and lag: [pairs, sum z, sum m, sum z m, sum z^2] for z on lead days t_range valid in DJF and m L days
    later valid in DJFM. z, m: (inits, ens, days); valid: (inits, days) months. Returns (winters, lags, 5), winters."""
    ws = sorted(set(winter_of))
    out = np.zeros((len(ws), len(B_LAGS), 5))
    days = z.shape[-1]
    for wi, w in enumerate(ws):
        ii = np.flatnonzero(winter_of == w)
        for li, L in enumerate(B_LAGS):
            for t in range(t_range[0] - 1, min(t_range[1], days - L)):
                ok = np.isin(valid[ii, t], (12, 1, 2)) & np.isin(valid[ii, t + L], (12, 1, 2, 3))
                if not ok.any():
                    continue
                zz, mm = z[ii[ok], :, t], m[ii[ok], :, t + L]
                good = np.isfinite(zz) & np.isfinite(mm)
                zz, mm = zz[good], mm[good]
                out[wi, li] += (len(zz), zz.sum(), mm.sum(), (zz * mm).sum(), (zz * zz).sum())
    return out, ws


def b_value(parts: np.ndarray) -> np.ndarray:
    """b from summed parts (..., winters, lags, 5)."""
    s = parts.sum(-3)
    n, sz, sm, szm, szz = (s[..., k] for k in range(5))
    return ((szm / n - sz / n * sm / n) / (szz / n - (sz / n) ** 2)).mean(-1)


def down_parts(cap, u100, valid, winter_of) -> np.ndarray:
    """Per winter, cross-products of [1, u100(t), cap(t), mean cap(t+1..t+15)] over lead days DOWN_T valid in DJFM:
    cap, u100 (inits, ens, days). (winters, 4, 4)."""
    ws = sorted(set(winter_of))
    out = np.zeros((len(ws), 4, 4))
    for wi, w in enumerate(ws):
        ii = np.flatnonzero(winter_of == w)
        for t in range(DOWN_T[0] - 1, DOWN_T[1]):
            ok = np.isin(valid[ii, t], (12, 1, 2, 3))
            if not ok.any():
                continue
            c, u = cap[ii[ok]], u100[ii[ok]]
            V = np.stack(
                [np.ones_like(c[..., t]), u[..., t], c[..., t], c[..., t + 1 : t + 1 + DOWN_NEXT].mean(-1)], -1
            )
            V = V.reshape(-1, 4)
            V = V[np.isfinite(V).all(1)]
            out[wi] += V.T @ V
    return out


def snr(f: np.ndarray, o: np.ndarray, winter_of: np.ndarray, rows=None) -> dict:
    """f (inits, ens) forecasts and o (inits,) ERA5 for one target; rows (draws, winters) resamples winters."""
    ws = np.array(sorted(set(winter_of)))
    if rows is None:
        rows = np.arange(len(ws))[None]
    out = {"r": [], "rpc": [], "slope": []}
    for rr in rows:
        ii = np.concatenate([np.flatnonzero(winter_of == ws[k]) for k in rr])
        fm, ff, oo = np.nanmean(f[ii], 1), f[ii], o[ii]
        ok = np.isfinite(fm) & np.isfinite(oo)
        fm, ff, oo = fm[ok], ff[ok], oo[ok]
        r = np.corrcoef(fm, oo)[0, 1]
        out["r"].append(r)
        out["rpc"].append(r / (fm.std() / np.sqrt(np.nanmean((ff - fm.mean()) ** 2))))
        out["slope"].append(np.cov(fm, oo)[0, 1] / fm.var(ddof=1))
    return {k: np.array(v) for k, v in out.items()}


def ci(x) -> list:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [EF.rnd(np.percentile(x, 5)), EF.rnd(np.percentile(x, 95))] if len(x) else [None, None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gefs", required=True, help="directory of gefs_eddy_extract.py's npz files")
    ap.add_argument("--era5", required=True, help="directory of eddy_extract.py's ERA5 npz files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = load_gefs(Path(a.gefs))
    era5 = EF.load(sorted(p for p in Path(a.era5).glob("ERA5_*.npz") if ".part" not in p.name))
    lat = g["lat"]
    ni, ne, nd = g["u"].shape[:3]
    valid_dt = g["init"].values[:, None] + (g["days"][None] * np.timedelta64(1, "D"))
    valid = pd.DatetimeIndex(valid_dt.ravel())
    vmonth = valid.month.values.reshape(ni, nd)
    winter_of = EF.EX.winter(g["init"].year.values, g["init"].month.values)

    # GEFS anomalies against its own lead-dependent climate
    G = derived(g["u"], g["uv"], g["slp"], g["u_levels"], g["uv_levels"], lat)
    for k, x in G.items():
        y = np.full(x.shape, np.nan)
        for d in range(nd):
            X = np.repeat(harmonics(pd.DatetimeIndex(valid_dt[:, d])), ne, 0)
            col = x[:, :, d].reshape(ni * ne, *x.shape[3:])
            y[:, :, d] = fit_remove(col, X, np.ones(len(col), bool)).reshape(ni, ne, *x.shape[3:])
        G[k] = y
    # ERA5 anomalies (1999-2020 climate) on its own days, then picked out on the forecast days
    edates = pd.to_datetime(pd.DataFrame({"year": era5["year"], "month": era5["month"], "day": era5["day"]}))
    edates = pd.DatetimeIndex(edates)
    E = derived(era5["u"], era5["uv"], era5["slp"], era5["levels"], era5["levels"], lat)
    clim_rows = (edates.year >= ERA5_CLIM[0]) & (edates.year <= ERA5_CLIM[1])
    Xe = harmonics(edates)
    E = {k: fit_remove(x, Xe, np.asarray(clim_rows)) for k, x in E.items()}
    # the jet pattern from ERA5, applied to both
    eof_rows = np.asarray((edates.year >= EOF_YEARS[0]) & (edates.year <= EOF_YEARS[1]))
    proj = EF.jet_eof(E["ut"], lat, eof_rows & np.isin(era5["month"], EF.DJF))
    for X in (G, E):
        X["z"], X["m"] = proj(X["ut"]), proj(X["f"])
    pos = pd.Series(np.arange(len(edates)), index=edates)
    vi = pos.reindex(valid).values.reshape(ni, nd)  # ERA5 position of each forecast day (NaN beyond ERA5)
    have = np.isfinite(vi)
    vi = np.where(have, vi, 0).astype(int)
    M = {k: np.where(have, E[k][vi], np.nan)[:, None] for k in ("z", "m", "cap", "u100")}  # (inits, 1, days)
    init_pos = pos.reindex(g["init"]).values
    u100_0 = np.where(np.isfinite(init_pos), E["u100"][np.nan_to_num(init_pos).astype(int)], np.nan)
    djfm = np.isin(edates.month, EF.DJFM) & clim_rows
    sd100 = np.nanstd(E["u100"][djfm])

    ws = sorted(set(winter_of))
    boot = RNG.integers(0, len(ws), (BOOT, len(ws)))
    doc = {"inits": int(ni), "members": int(ne), "winters": len(ws), "first": str(g["init"][0].date()),
           "last": str(g["init"][-1].date()), "boot": BOOT}  # fmt: skip

    # b, overall and by lead
    doc["b"] = {}
    for name, tr in (("days5_20", B_T), ("days5_12", (5, 12)), ("days13_20", (13, 20))):
        pg, _ = b_parts(G["z"], G["m"], vmonth, winter_of, tr)
        pe, _ = b_parts(M["z"], M["m"], vmonth, winter_of, tr)
        bg, be = b_value(pg), b_value(pe)
        dg, de = b_value(pg[boot]), b_value(pe[boot])
        doc["b"][name] = {"gefs": EF.rnd(bg), "era5": EF.rnd(be), "ratio": EF.rnd(bg / be, 3),
                          "gefs_ci90": ci(dg), "era5_ci90": ci(de), "ratio_ci90": ci(dg / de),
                          "p_gefs_weaker": EF.rnd((1 + (dg - de >= 0).sum()) / (BOOT + 1))}  # fmt: skip
    # down
    pg, pe = down_parts(G["cap"], G["u100"], vmonth, winter_of), down_parts(M["cap"], M["u100"], vmonth, winter_of)
    (dg, dgs), (de, des) = EF.down_coef(pg), EF.down_coef(pe)
    (bg, bgs), (be, bes) = EF.down_coef(pg[boot]), EF.down_coef(pe[boot])
    doc["down"] = {"gefs": EF.rnd(dg), "era5": EF.rnd(de), "gefs_ci90": ci(bg), "era5_ci90": ci(be),
                   "gefs_sd": EF.rnd(dgs), "era5_sd": EF.rnd(des), "gefs_sd_ci90": ci(bgs), "era5_sd_ci90": ci(bes),
                   "p_gefs_weaker": EF.rnd((1 + (bg - be <= 0).sum()) / (BOOT + 1))}  # fmt: skip
    # signal to noise in weeks 3-4
    d0, d1 = WEEKS34[0] - 1, WEEKS34[1]
    centre = pd.DatetimeIndex(g["init"] + pd.Timedelta(days=int(np.mean(WEEKS34))))
    in_djf = np.isin(centre.month, EF.DJF)
    doc["snr"] = {}
    for var in ("cap", "z"):
        f = np.nanmean(G[var][:, :, d0:d1], -1)
        o = np.nanmean(M[var][:, 0, d0:d1], -1)
        for subset, keep in (("all", in_djf), ("weak_100", in_djf & (u100_0 < -sd100)),
                             ("strong_100", in_djf & (u100_0 > sd100))):  # fmt: skip
            if keep.sum() < 20:
                continue
            ii = np.flatnonzero(keep)
            nw = len(set(winter_of[ii]))
            s0 = snr(f[ii], o[ii], winter_of[ii])
            sb = snr(f[ii], o[ii], winter_of[ii], rows=RNG.integers(0, nw, (BOOT, nw)))
            doc["snr"][f"{var}_{subset}"] = {
                "inits": len(ii), **{k: EF.rnd(v[0], 3) for k, v in s0.items()},
                **{f"{k}_ci90": ci(v) for k, v in sb.items()},
                "p_rpc_above_1": EF.rnd((1 + (sb["rpc"] <= 1).sum()) / (BOOT + 1)),
            }  # fmt: skip
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
