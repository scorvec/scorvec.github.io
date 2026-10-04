#!/usr/bin/env python3
""""What usually follows" from the CMIP6 sample: surface temperature, sea-level pressure, the regional cold odds, a daily
annular-mode proxy and a monthly NAO after thousands of model sudden warmings and strong-vortex events (2026-09-27; user:
"Will you also update 'what usually follows' with CMIP6?" and "Would also be worth looking at the NAO index as well").

LAPTOP-ONLY, run once. Inputs:
  scripts/strat/data/cmip6/<model>_<exp>_<member>.nc        daily u(60N) (cmip6_strat_extract.py)      -> the events
  scripts/strat/data/cmip6/follow/<same>.npz                monthly tas, psl, 2.5 deg, 20-90N (cmip6_follow_extract.py)
  ~/era5_store/wb2_1p5_daily_global/slp                     ERA5 daily SLP 1991-2020 (hPa, whatever the label says)
  scripts/telecon/data/cpc_nao_monthly.txt                  CPC monthly NAO, for the sign/pattern cross-check only
Output: scripts/strat/reference/cmip6_follow.nc (per-model composites; strat_history.py tests and draws them in Actions).

EVENTS: exactly the u(60N) drips' events (build_cmip6_drip.py, whose functions this imports): the same 9 models, every
member and experiment with a daily extract, CP07 SSWs Dec-Mar on u(60N, 10 hPa), deep/shallow at each model's median of
the days +1..+30 lower-stratospheric (100 + 150 hPa) wind anomaly, strong-vortex events where standardised u'(60N, 10 hPa)
first crosses +1.5 (Nov-Mar, 60 days apart). The event counts are checked against reference/cmip6_drip.nc.

SURFACE FIELDS are MONTHLY (Amon; there is no daily 2 m temperature or pressure in this sample's archive): every member's
monthly anomaly from its own running climatology (a 31-year centred running mean of each calendar month, so control drift
and forced trends drop out, as in the drips), and an event's "days 1-30" / "days 31-60" value is the overlap-weighted mean
of the monthly anomalies - each day of the window carries its calendar month's anomaly. That smooths the windows by up to a
month; the ERA5 version averages daily anomalies. Sea-level pressure stands in for ERA5's 500 hPa height (Amon zg is ~1.5 TB
to stream).

INDICES
  u'(60N, 700 hPa)  the drips' standardised daily wind anomaly: a daily annular-mode proxy with the AO's sign.
  SLP NAM           monthly: (35-55N minus 65-90N) zonal-mean SLP anomaly, standardised per model and calendar month.
  NAO               monthly: each member's SLP anomaly over 20-80N, 90W-40E projected (cos-lat weights) on the ERA5 NAO
                    pattern of that calendar month - the leading EOF of ERA5 monthly SLP anomalies (1991-2020, the month
                    and its two neighbours, sqrt-cos weighted, on the same 2.5-degree grid), signed so a low over Iceland is
                    positive - then standardised per model and calendar month. ERA5 projected the same way is correlated
                    with CPC's monthly NAO as the check (stored as attrs).
REGIONS: build_strat_history.REGIONS, weights = land fraction (Natural Earth 110 m, 0.25-degree subsample) x cos(lat);
"cold" = the window's mean anomaly below zero; each model's normal chance of a cold window from every Dec-Mar start day.

    python scripts/strat/build_cmip6_follow.py
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import build_cmip6_drip as B                                       # noqa: E402
import build_strat_history as BH                                   # noqa: E402
import cmip6_strat_analysis as A                                   # noqa: E402
import cmip6_follow_extract as FX                                  # noqa: E402

OUT = HERE / "reference" / "cmip6_follow.nc"
FOLLOW = B.DATA / "follow"
SETS = B.SETS
WINS = {"d1_30": (1, 30), "d31_60": (31, 60)}
LAGS = B.LAGS
LAT, LON = FX.LAT_C, FX.LON_C
REGIONS = BH.REGIONS
NAO_BOX = (20.0, 80.0, 270.0, 40.0)
ERA5_SLP = Path.home() / "era5_store" / "wb2_1p5_daily_global" / "slp"
CPC_NAO_M = REPO / "scripts" / "telecon" / "data" / "cpc_nao_monthly.txt"
CPC_NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"
DJFM = (12, 1, 2, 3)


# ------------------------------------------------------------------ grids, masks, patterns
def box_mask():
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    return (la >= NAO_BOX[0]) & (la <= NAO_BOX[1]) & ((lo >= NAO_BOX[2]) | (lo <= NAO_BOX[3]))


def region_weights():
    """(6, lat, lon) land-fraction x cos(lat) weights, normalised."""
    import shapely
    import cartopy.io.shapereader as shp
    land = shapely.union_all([g for g in shp.Reader(shp.natural_earth("110m", "physical", "land")).geometries()])
    shapely.prepare(land)
    sub = (np.arange(10) + 0.5) * 0.25
    frac = np.zeros((len(LAT), len(LON)))
    for i, la in enumerate(LAT):
        ys = la - 1.25 + sub
        for j, lo in enumerate(LON):
            xs = ((lo - 1.25 + sub + 180) % 360) - 180
            X, Y = np.meshgrid(xs, ys)
            frac[i, j] = shapely.contains_xy(land, X.ravel(), Y.ravel()).mean()
    W = []
    for rn, (la0, la1, lo0, lo1) in REGIONS.items():
        inlat = (LAT >= la0) & (LAT <= la1)
        inlon = ((LON >= lo0) & (LON <= lo1)) if lo0 < lo1 else ((LON >= lo0) | (LON <= lo1))
        w = frac * np.cos(np.deg2rad(LAT))[:, None] * (inlat[:, None] & inlon[None, :])
        W.append(w / w.sum())
    return np.stack(W), frac


def era5_monthly_slp():
    """ERA5 monthly SLP (hPa) on the 2.5-degree grid, 1991-2020."""
    fields, ym, M = [], [], None
    for f in sorted(ERA5_SLP.glob("slp_*.nc")):
        d = xr.open_dataset(f)
        v = d[list(d.data_vars)[0]].transpose("time", "latitude", "longitude")
        if M is None:
            M, empty = FX.coarsen_matrix(v.latitude.values, v.longitude.values)
            assert not empty
        mm = v.resample(time="1MS").mean()
        vals = mm.values.reshape(mm.sizes["time"], -1).astype("float64")
        fields.append((M @ vals.T).T.reshape(-1, len(LAT), len(LON)))
        ym += [(t.year, t.month) for t in pd.DatetimeIndex(mm.time.values)]
    X = np.concatenate(fields)
    assert 900 < np.nanmean(X) < 1100, f"ERA5 slp not in hPa? mean {np.nanmean(X):.1f}"
    return X, np.array(ym)


def nao_patterns():
    """12 calendar-month NAO patterns (hPa per standard deviation) on the box cells, and the ERA5 check against CPC."""
    X, ym = era5_monthly_slp()
    bm = box_mask()
    w = np.sqrt(np.cos(np.deg2rad(LAT)))[:, None] * np.ones((1, len(LON)))
    yrs, mon = ym[:, 0], ym[:, 1]
    A_ = np.full_like(X, np.nan)
    for m in range(1, 13):                                   # anomalies: minus each calendar month's mean and trend
        k = mon == m
        t = yrs[k] - yrs[k].mean()
        G = X[k].reshape(k.sum(), -1)
        b = (t[:, None] * (G - G.mean(0))).sum(0) / (t ** 2).sum()
        A_[k] = (G - G.mean(0) - t[:, None] * b).reshape(-1, len(LAT), len(LON))
    pats = np.zeros((12, bm.sum()))
    box_idx = np.flatnonzero(bm.ravel())
    la2, lo2 = np.meshgrid(LAT, LON, indexing="ij")
    dist = (la2 - 65.0) ** 2 + (((lo2 - 340.0 + 180.0) % 360.0) - 180.0) ** 2        # Iceland, 65N 20W
    ice_pos = int(np.argmin(dist.ravel()[box_idx]))
    var1 = {}
    for m in range(1, 13):
        k = np.isin(mon, [(m - 2) % 12 + 1, m, m % 12 + 1])
        Z = A_[k].reshape(k.sum(), -1)[:, box_idx]
        Zw = Z * w.ravel()[box_idx]
        U, S, Vt = np.linalg.svd(Zw - Zw.mean(0), full_matrices=False)
        pc = (Zw - Zw.mean(0)) @ Vt[0]
        pc = (pc - pc.mean()) / pc.std()
        pat = Z.T @ pc / len(pc)                             # regression map, hPa per sigma
        if pat[ice_pos] > 0:
            pat = -pat
        pats[m - 1] = pat
        var1[m] = float(S[0] ** 2 / (S ** 2).sum())
    # check: ERA5 projected like the models, standardised per calendar month, against CPC's monthly NAO
    cw = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, len(LON)))
    cwb = cw.ravel()[box_idx]
    raw = np.array([(A_[i].ravel()[box_idx] * cwb * pats[mon[i] - 1]).sum() / (cwb * pats[mon[i] - 1] ** 2).sum()
                    for i in range(len(mon))])
    std = np.full_like(raw, np.nan)
    for m in range(1, 13):
        k = mon == m
        std[k] = (raw[k] - raw[k].mean()) / raw[k].std()
    if not CPC_NAO_M.exists():
        import urllib.request
        urllib.request.urlretrieve(CPC_NAO_URL, CPC_NAO_M)
    cpc = {}
    for line in open(CPC_NAO_M):
        p = line.split()
        if len(p) == 3 and float(p[2]) > -90:
            cpc[(int(p[0]), int(p[1]))] = float(p[2])
    c = np.array([cpc.get((y, m), np.nan) for y, m in ym])
    ok = np.isfinite(c)
    r_all = float(np.corrcoef(std[ok], c[ok])[0, 1])
    dj = ok & np.isin(mon, DJFM)
    r_djfm = float(np.corrcoef(std[dj], c[dj])[0, 1])
    print(f"  NAO patterns: EOF1 variance {min(var1.values()):.0%}-{max(var1.values()):.0%}; ERA5 projection vs CPC monthly NAO "
          f"1991-2020 r = {r_all:.2f} (all months), {r_djfm:.2f} (Dec-Mar)", flush=True)
    return pats, box_idx, {"r_cpc_all": round(r_all, 3), "r_cpc_djfm": round(r_djfm, 3),
                           "eof1_var": {str(k): round(v, 3) for k, v in var1.items()}}


# ------------------------------------------------------------------ per member / per model
def model_events(model):
    """The drips' events, per member (build_cmip6_drip.model_composites, returning the event indices)."""
    mem, ss, cnt, p = [], None, None, None
    for f, exp, member in B.files_for(model):
        try:
            Y, M, dix, nd, u, p = B.read_member(f)
        except Exception as e:                                      # noqa: BLE001
            print(f"  {model} {exp} {member}: skipped ({str(e)[:80]})", flush=True); continue
        if len(np.unique(Y)) < 20:
            continue
        a = B.running_anom(u, Y, dix, nd)
        if ss is None:
            ss, cnt = np.zeros((nd, u.shape[1])), np.zeros((nd, u.shape[1]))
        ok = np.isfinite(a)
        np.add.at(ss, dix, np.where(ok, a * a, 0.0)); np.add.at(cnt, dix, ok)
        mem.append([f, exp, member, Y, M, dix, nd, u, a])
    var = ss / np.maximum(cnt, 1)
    pad = np.concatenate([var[-15:], var, var[:15]])
    sd = np.sqrt(pd.DataFrame(pad).rolling(31, center=True, min_periods=16).mean().values[15:-15])
    sd = np.maximum(sd, np.nanmax(sd, 0, keepdims=True) / 3)
    k10, k100, k150, kst = (int(np.argmin(np.abs(p - x))) for x in (10.0, 100.0, 150.0, B.STRIP_P))
    out, depth_all = [], []
    for f, exp, member, Y, M, dix, nd, u, a in mem:
        z = a / sd[dix]
        qw = -z
        ev = A.cp07_djfm(u[:, k10], Y, M)
        ev = [i for i in ev if i + 30 < len(u)]
        sv = B.strong_vortex(z[:, k10], M)
        depth = [float(np.nanmean(qw[i + 1:i + 31][:, [k100, k150]])) for i in ev]
        depth_all += depth
        out.append({"exp": exp, "member": member, "Y": Y, "M": M, "dix": dix, "nd": nd, "ssw": ev, "depth": depth, "sv": sv,
                    "strip": z[:, kst].astype("float32")})
    mem.clear()
    med = float(np.median(depth_all))
    for o in out:
        deep = [d >= med for d in o["depth"]]
        o["sets"] = {"ssw_all": o["ssw"], "ssw_deep": [i for i, d in zip(o["ssw"], deep) if d],
                     "ssw_shallow": [i for i, d in zip(o["ssw"], deep) if not d], "sv": o["sv"]}
    return out


def running_month_anom(year, month, arr):
    """Monthly anomalies from a 31-year centred running mean of each calendar month. Returns G (ny, 12, ...) and y0."""
    y0 = int(year.min()); ny = int(year.max()) - y0 + 1
    G = np.full((ny, 12) + arr.shape[1:], np.nan, "float32")
    G[year - y0, month - 1] = arr
    flat = G.reshape(ny, -1)
    C = pd.DataFrame(flat).rolling(31, center=True, min_periods=10).mean().values
    return (flat - C).reshape(G.shape).astype("float32"), y0


def window_sum(s, a, b):
    """For a daily series s (T, k): mean over days c+a..c+b for every start day c (NaN where the window leaves the record
    or has no finite day)."""
    T = s.shape[0]
    fin = np.isfinite(s)
    cs = np.vstack([np.zeros((1,) + s.shape[1:]), np.cumsum(np.where(fin, s, 0.0), 0)])
    cn = np.vstack([np.zeros((1,) + s.shape[1:]), np.cumsum(fin, 0)])
    c = np.arange(T)
    lo, hi = c + a, c + b + 1
    ok = hi <= T
    out = np.full(s.shape, np.nan)
    n = cn[hi[ok]] - cn[lo[ok]]
    out[ok] = np.where(n > 0, (cs[hi[ok]] - cs[lo[ok]]) / np.maximum(n, 1), np.nan)
    return out


def model_follow(args):
    model, pats, box_idx, RW = args
    t0 = time.time()
    members = model_events(model)
    cw = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, len(LON)))
    cwb = cw.ravel()[box_idx]
    capm = LAT >= 65.0; midm = (LAT >= 35.0) & (LAT <= 55.0)
    wlat = np.cos(np.deg2rad(LAT))
    acc = {"tas": {(s, w): [] for s in SETS for w in WINS}, "psl": {(s, w): [] for s in SETS for w in WINS}}
    ev_vals = {(s, w): [] for s in SETS for w in WINS}                   # per event: 6 regions, nao, nam, strip
    base_vals = {w: [] for w in WINS}                                    # every Dec-Mar start day
    paths = {s: [] for s in SETS}; pbase = {s: [] for s in SETS}
    P0num = P0den = None
    raw_idx = []                                                         # (member slot, index series) for standardising
    per_member = []
    n_surface = {s: 0 for s in SETS}
    for o in members:
        fz = FOLLOW / f"{model}_{o['exp']}_{o['member']}.npz"
        sm = pd.Series(o["strip"]).rolling(7, center=True, min_periods=4).mean().values
        nd = o["nd"]
        if P0num is None:
            P0num, P0den = np.zeros(nd), np.zeros(nd)
        fin = np.isfinite(sm)
        np.add.at(P0num, o["dix"][fin], (sm[fin] < 0)); np.add.at(P0den, o["dix"][fin], 1)
        per_member.append({"o": o, "sm": sm, "fz": fz if fz.exists() else None})
    P0 = P0num / np.maximum(P0den, 1)
    # monthly indices: raw NAO projection and SLP NAM per member, standardised per model x calendar month afterwards
    mon_ix = []
    for pm in per_member:
        if pm["fz"] is None:
            continue
        z = np.load(pm["fz"])
        yr, mo = z["year"].astype(int), z["month"].astype(int)
        psl = np.where(z["psl"] == -32768, np.nan, (z["psl"] + 101300.0) / 100.0).astype("float32")      # hPa
        Gp, y0 = running_month_anom(yr, mo, psl)
        ny = Gp.shape[0]
        flat = Gp.reshape(ny, 12, -1)[..., box_idx]                      # (ny, 12, box)
        nao = np.einsum("ymb,mb->ym", flat * cwb, pats) / (cwb * pats ** 2).sum(1)[None, :]
        zm = np.nanmean(Gp, 3)                                           # (ny, 12, lat) zonal mean
        nam = (np.nansum(zm[..., midm] * wlat[midm], -1) / wlat[midm].sum()
               - np.nansum(zm[..., capm] * wlat[capm], -1) / wlat[capm].sum())
        pm.update(y0=y0, nao=nao, nam=nam)
        mon_ix.append((nao, nam))
        del Gp, flat, zm
    # per-model standardisation of the monthly indices by calendar month (all members pooled)
    na = np.concatenate([x[0] for x in mon_ix]); nm_ = np.concatenate([x[1] for x in mon_ix])
    mu_n, sd_n = np.nanmean(na, 0), np.nanstd(na, 0); mu_m, sd_m = np.nanmean(nm_, 0), np.nanstd(nm_, 0)
    for pm in per_member:
        o = pm["o"]
        if pm["fz"] is None:
            continue
        Y, M = o["Y"], o["M"]
        z = np.load(pm["fz"])
        yr, mo = z["year"].astype(int), z["month"].astype(int)
        pm["Gt"], _ = running_month_anom(yr, mo, np.where(z["tas"] == -32768, np.nan, z["tas"] / 100.0 + 260.0).astype("float32"))
        pm["Gp"], _ = running_month_anom(yr, mo, np.where(z["psl"] == -32768, np.nan, (z["psl"] + 101300.0) / 100.0).astype("float32"))
        reg = np.einsum("ymij,rij->ymr", np.nan_to_num(pm["Gt"]), RW)     # (ny, 12, 6)
        reg[~np.isfinite(pm["Gt"]).all((2, 3))] = np.nan
        pm["reg"] = reg
        yy = Y - pm["y0"]; mm = M - 1
        inside = (yy >= 0) & (yy < pm["Gt"].shape[0])
        yyc = np.clip(yy, 0, pm["Gt"].shape[0] - 1)
        naod = np.where(inside, ((pm["nao"] - mu_n) / sd_n)[yyc, mm], np.nan)
        namd = np.where(inside, ((pm["nam"] - mu_m) / sd_m)[yyc, mm], np.nan)
        regd = np.where(inside[:, None], pm["reg"][yyc, mm], np.nan)
        daily = np.column_stack([regd, naod, namd, o["strip"]])          # (T, 9): 6 regions, NAO, NAM, u'700
        startok = np.isin(M, DJFM)
        for w, (a, b) in WINS.items():
            S = window_sum(daily, a, b)
            base_vals[w].append(S[startok & np.isfinite(S[:, :8]).all(1)])
            for s in SETS:
                for i in o["sets"][s]:
                    ev_vals[(s, w)].append(S[i])
                    j = np.arange(i + a, min(i + b, len(Y) - 1) + 1)
                    if not len(j) or not inside[j].all():
                        continue
                    keys, cnt = np.unique(np.column_stack([yy[j], mm[j]]), axis=0, return_counts=True)
                    wts = cnt / cnt.sum()
                    acc["tas"][(s, w)].append(np.tensordot(wts, pm["Gt"][keys[:, 0], keys[:, 1]], 1))
                    acc["psl"][(s, w)].append(np.tensordot(wts, pm["Gp"][keys[:, 0], keys[:, 1]], 1))
        for s in SETS:
            idx = o["sets"][s]
            if idx:
                paths[s].append(B.windows(pm["sm"][:, None], idx)[..., 0])
                j = np.clip(np.array(idx)[:, None] + LAGS[None, :], 0, len(Y) - 1)
                valid = (np.array(idx)[:, None] + LAGS[None, :] >= 0) & (np.array(idx)[:, None] + LAGS[None, :] < len(Y))
                pbase[s].append(np.where(valid, P0[o["dix"][j]], np.nan))
                n_surface[s] += len(idx)
        for k in ("Gt", "Gp"):
            pm.pop(k, None)
    res = {"n": {s: int(sum(len(pm["o"]["sets"][s]) for pm in per_member)) for s in SETS}, "n_surface": n_surface}
    for v in ("tas", "psl"):
        res[v] = {k: (np.nanmean(np.stack(x), 0) if x else np.full((len(LAT), len(LON)), np.nan)) for k, x in acc[v].items()}
    E = {k: (np.stack(x) if x else np.full((0, 9), np.nan)) for k, x in ev_vals.items()}
    Bv = {w: np.concatenate(x) for w, x in base_vals.items()}
    res["ev_mean"] = {k: np.nanmean(e, 0) for k, e in E.items()}                         # (9,)
    res["ev_pneg"] = {k: np.nanmean(np.where(np.isfinite(e), e < 0, np.nan), 0) for k, e in E.items()}
    res["base_pneg"] = {w: np.nanmean(b < 0, 0) for w, b in Bv.items()}                 # (9,)
    res["path"] = {s: np.nanmean(np.concatenate(paths[s]), 0) for s in SETS}
    res["path_pneg"] = {s: np.nanmean(np.where(np.isfinite(np.concatenate(paths[s])), np.concatenate(paths[s]) < 0, np.nan), 0)
                        for s in SETS}
    res["path_base"] = {s: np.nanmean(np.concatenate(pbase[s]), 0) for s in SETS}
    res["years"] = int(sum(len(np.unique(pm["o"]["Y"])) for pm in per_member))
    res["members_surface"] = int(sum(pm["fz"] is not None for pm in per_member))
    res["members"] = len(per_member)
    print(f"  {model:16s} {res['members']} runs ({res['members_surface']} with Amon) {res['years']} yrs  events {res['n']}  "
          f"{time.time() - t0:.0f} s", flush=True)
    return model, res


def main():
    t0 = time.time()
    RW, frac = region_weights()
    pats, box_idx, check = nao_patterns()
    with ProcessPoolExecutor(3) as ex:
        res = dict(ex.map(model_follow, [(m, pats, box_idx, RW) for m in B.GOOD]))
    models = [m for m in B.GOOD if m in res]
    R0 = xr.open_dataset(B.OUT)
    chk = {m: {s: (res[m]["n"][s], int(R0.cmip6_n.sel(set=s, model=m))) for s in SETS} for m in models if m in R0.model.values}
    bad = {m: v for m, v in chk.items() if any(a != b for a, b in v.values())}
    print("  event counts vs cmip6_drip.nc:", "identical" if not bad else f"DIFFER {bad}")
    IX = ["reg_" + r for r in REGIONS] + ["nao", "nam", "u700"]
    ds = xr.Dataset(
        {"tas": (("set", "window", "model", "lat", "lon"),
                 np.stack([[[res[m]["tas"][(s, w)] for m in models] for w in WINS] for s in SETS]).astype("float32")),
         "psl": (("set", "window", "model", "lat", "lon"),
                 np.stack([[[res[m]["psl"][(s, w)] for m in models] for w in WINS] for s in SETS]).astype("float32")),
         "ix_mean": (("set", "window", "model", "index"),
                     np.stack([[[res[m]["ev_mean"][(s, w)] for m in models] for w in WINS] for s in SETS]).astype("float32")),
         "ix_pneg": (("set", "window", "model", "index"),
                     np.stack([[[res[m]["ev_pneg"][(s, w)] for m in models] for w in WINS] for s in SETS]).astype("float32")),
         "ix_base_pneg": (("window", "model", "index"),
                          np.stack([[res[m]["base_pneg"][w] for m in models] for w in WINS]).astype("float32")),
         "u700_path": (("set", "model", "lag"), np.stack([[res[m]["path"][s] for m in models] for s in SETS]).astype("float32")),
         "u700_pneg": (("set", "model", "lag"), np.stack([[res[m]["path_pneg"][s] for m in models] for s in SETS]).astype("float32")),
         "u700_base_pneg": (("set", "model", "lag"), np.stack([[res[m]["path_base"][s] for m in models] for s in SETS]).astype("float32")),
         "n": (("set", "model"), np.array([[res[m]["n"][s] for m in models] for s in SETS], "int32")),
         "n_surface": (("set", "model"), np.array([[res[m]["n_surface"][s] for m in models] for s in SETS], "int32")),
         "years": ("model", np.array([res[m]["years"] for m in models], "int32")),
         "members": ("model", np.array([res[m]["members"] for m in models], "int32")),
         "region_w": (("region", "lat", "lon"), RW.astype("float32")),
         "land_frac": (("lat", "lon"), frac.astype("float32")),
         "nao_pattern": (("calmonth", "box"), pats.astype("float32")),
         "nao_box_cell": ("box", box_idx.astype("int32"))},
        coords={"set": SETS, "window": list(WINS), "model": models, "lat": LAT, "lon": LON, "index": IX,
                "region": list(REGIONS), "lag": LAGS, "calmonth": np.arange(1, 13)},
        attrs={"title": "CMIP6 'What usually follows': per-model composites after the u(60N) drips' events",
               "built": pd.Timestamp.utcnow().strftime("%Y-%m-%d"), "builder": "scripts/strat/build_cmip6_follow.py",
               "units": "tas K, psl hPa (monthly anomalies); indices: regions K, nao/nam standardised (monthly), u700 standardised (daily)",
               "windows": "overlap-weighted monthly anomalies: each day of the window carries its calendar month's anomaly "
                          "(maps, regions, NAO, NAM); u700 = mean of the daily standardised anomaly",
               "z500_substitute": "sea-level pressure (Amon psl) in place of ERA5's 500 hPa height",
               "nao": "projection on the ERA5 1991-2020 calendar-month EOF1 of monthly SLP, 20-80N 90W-40E; " + json.dumps(check),
               "regions": json.dumps({k: list(v) for k, v in REGIONS.items()}),
               "experiments": R0.attrs.get("experiments", "")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT, encoding={v: {"zlib": True, "complevel": 5} for v in ds.data_vars})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), {len(models)} models, {int(ds.years.sum())} model-years, "
          f"events {dict(zip(SETS, ds.n.sum('model').values.tolist()))}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
