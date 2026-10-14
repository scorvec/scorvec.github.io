#!/usr/bin/env python3
"""The rain-truth blend for the model-rain bias field (2026-10-13, user: "try to come up with the 'ultimate' rain truth blend").

Inputs on the 1-degree Americas grid (releases precip-bias-arch-v1 and rain-truth-arch-v1, unpacked in one directory):
  IMERG Late scaled by its seasonal ratio to IMERG Final (build_bias.truth_terms) - complete, oceans included
  MERGE (INPE, IMERG Late + South American gauges), CPC unified (gauges), GPCC First Guess Daily (gauges), CHIRPS (IR + gauges)
  IDEAM gauges (Colombia): mostly NOT exchanged on the GTS, so mostly independent of the gauge grids - the validation set
Day definitions are aligned to UTC days as far as they are known (MERGE / CPC 12Z-12Z ending on the labelled day: UTC day D =
half of day D + half of day D+1; GPCC 06Z-06Z from the labelled day: 1/4 of D-1 + 3/4 of D). Misalignment adds day-to-day noise
but does not bias a long-run mean, which is what the bias field uses.

Method:
  0. Gauge screen (screen_gauges): physical daily cap, then each station-month against the median of its neighbouring stations
     (gauges only, so no product is favoured); stuck, accumulating and dead sensors go.
  1. Station boxes: IDEAM stations averaged per 1-degree box and UTC day; a box-month needs >= 25 days.
  2. Each product against the station box-months, by terrain class of the box (lowland < 500 m, hills 500-1500, mountains >= 1500
     m mean ERA5 height): mean ratio (product / gauges) and mean-squared error of the monthly mean rate.
  3. Blend over land = sum_k w_k P_k / sum_k w_k over the products present that day, w_k = 1 / MSE_k of the box's terrain class
     Over the ocean: the adjusted IMERG alone.
  4. Scores by 5-fold cross-validation over the boxes (each box scored with weights fitted on the other folds), every product and
     the blend on the SAME box-months (all present); blend-minus-product RMSE differences with 95 % intervals from resampling
     boxes. The published blend uses weights fitted on all boxes. Truth uncertainty per box and season = the spread (standard
     deviation) of the products' seasonal means.

    python scripts/precip_bias/blend.py --data DIR --out DIR --json assets/precip_bias/truth_blend.json
-> DIR/blend_{YYYYMM}.npz (dates, pr (day, lat, lon) mm/day, nprod), DIR/blend_spread.npz, the summary JSON
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid as G                                          # noqa: E402
import build_bias as B                                    # noqa: E402

PRODUCTS = ("imerg", "merge", "cpc", "gpcc", "chirps")
SHIFT = {"merge": (0.5, 0.5), "cpc": (0.5, 0.5), "gpcc": None, "chirps": None}      # (weight of day D, of day D+1)
NFOLD, NBOOT = 5, 4000
CLASS_EDGES = ((0, 500, "lowland (<500 m)"), (500, 1500, "hills (500-1500 m)"), (1500, 9999, "mountains (>=1500 m)"))


def _d(s):
    return datetime.strptime(str(s), "%Y%m%d").date()


def load_grid(data: Path, src):
    files = sorted(data.glob(f"{src}_[0-9]*.npz"))
    if not files:
        return {}, None
    dates, arr = [], []
    for f in files:
        z = np.load(f)
        dates += [_d(s) for s in z["dates"]]; arr.append(z["pr"].astype("float32"))
    a = np.concatenate(arr); o = np.argsort(np.array(dates, dtype="datetime64[D]"))
    dates = [dates[i] for i in o]; a = a[o]
    return {d: i for i, d in enumerate(dates)}, a


def utc_series(src, idx, arr, days):
    """(day, NY, NX) on UTC days for the requested days, from a product's labelled days."""
    out = np.full((len(days), G.NY, G.NX), np.nan, dtype="float32")
    for k, d in enumerate(days):
        if src in ("merge", "cpc"):
            a, b = idx.get(d), idx.get(d + timedelta(days=1))
            if a is not None and b is not None:
                out[k] = 0.5 * arr[a] + 0.5 * arr[b]
        elif src == "gpcc":
            a, b = idx.get(d - timedelta(days=1)), idx.get(d)
            if a is not None and b is not None:
                out[k] = 0.25 * arr[a] + 0.75 * arr[b]
        else:
            a = idx.get(d)
            if a is not None:
                out[k] = arr[a]
    return out


def terrain_class(elev):
    c = np.full(elev.shape, -1, int)
    for k, (lo, hi, _) in enumerate(CLASS_EDGES):
        c[(elev >= lo) & (elev < hi)] = k
    return c


# gauge screening (2026-10-14: the archive holds stuck and accumulating sensors - a La Guajira station averaging 1,782 mm/day,
# Choco stations at 200-500 mm/day - and dead ones reporting 0.0 for months while their neighbours rain). Gauges only: no
# product enters the screen, so it cannot favour one.
DAY_MAX = 500.0                      # mm/day; above Colombia's daily records
NB_KM, NB_KM_WIDE, NB_MAX, NB_MIN = 75.0, 150.0, 8, 3
FACTOR = 4.0                         # station-month vs the median of its neighbours' same-month means
MONTH_MIN_DAYS = 20
STATION_BAD_FRAC = 0.5               # a station with more flagged months than this is dropped whole


def screen_gauges(d, st):
    """d: station-days (id, t, mm); st: stations (index id, lat, lon). -> (screened d, QC summary)."""
    q = {"station_days_in": int(len(d)), "stations_in": int(d.id.nunique())}
    d = d[(d.mm >= 0) & (d.mm <= DAY_MAX)].copy()
    q["days_over_max"] = q["station_days_in"] - int(len(d))
    d["ym"] = [(t.year, t.month) for t in d.t]
    g = d.groupby(["id", "ym"]).mm.agg(["mean", "size"]).reset_index()
    M = g[g["size"] >= MONTH_MIN_DAYS].pivot(index="id", columns="ym", values="mean")
    ids = M.index.values; lat = st.loc[ids, "lat"].values; lon = st.loc[ids, "lon"].values
    km = 6371.0 * np.hypot(np.radians(lat[:, None] - lat[None]), np.radians(lon[:, None] - lon[None]) * np.cos(np.radians(lat))[:, None])
    np.fill_diagonal(km, np.inf)
    X = M.values; flag = np.zeros(X.shape, bool); checked = np.zeros(X.shape, bool); no_nb = 0
    for i in range(len(ids)):
        nb = np.argsort(km[i])
        near = [j for j in nb[:NB_MAX] if km[i, j] <= NB_KM]
        if len(near) < NB_MIN:
            near = [j for j in nb[:NB_MAX] if km[i, j] <= NB_KM_WIDE]
        if len(near) < NB_MIN:
            no_nb += 1; continue
        ref = np.nanmedian(np.where(np.isfinite(X[near]).sum(0) >= NB_MIN, X[near], np.nan), 0) \
            if np.isfinite(X[near]).any() else np.full(X.shape[1], np.nan)
        x = X[i]; ok = np.isfinite(x) & np.isfinite(ref)
        checked[i] = ok
        with np.errstate(invalid="ignore"):
            over = x > FACTOR * np.maximum(ref, 1.0)                 # dry months: 4 mm/day above a near-zero reference is still wrong
            under = (ref >= 2.0) & (x < ref / FACTOR)                # dead or clogged gauge while the neighbours rain
        flag[i] = ok & (over | under)
    nchk = checked.sum(1); nbad = flag.sum(1)
    drop_station = (nchk == 0) | (nbad > STATION_BAD_FRAC * np.maximum(nchk, 1))
    keep = set(ids[~drop_station])
    bad_months = {(ids[i], M.columns[k]) for i, k in zip(*np.nonzero(flag)) if not drop_station[i]}
    d = d[d.id.isin(keep)]
    d = d[[(a, b) not in bad_months for a, b in zip(d.id, d.ym)]]
    q.update({"stations_without_neighbours": int(no_nb), "stations_dropped": int(drop_station.sum()),
              "station_months_checked": int(checked.sum()), "station_months_flagged": int(flag.sum()),
              "stations_out": int(d.id.nunique()), "station_days_out": int(len(d)),
              "rules": f"days > {DAY_MAX:g} mm out; each station-month (>= {MONTH_MIN_DAYS} days) against the median of up to {NB_MAX} "
                       f"neighbours within {NB_KM:g} km ({NB_KM_WIDE:g} km where fewer than {NB_MIN}): out if above {FACTOR:g} x max(ref, 1 mm/day) "
                       f"or below ref / {FACTOR:g} with ref >= 2 mm/day; a station with over {STATION_BAD_FRAC:.0%} of its months out, "
                       f"or none checkable, is dropped whole"})
    print("IDEAM screen: " + ", ".join(f"{k} {v}" for k, v in q.items() if k != "rules"), flush=True)
    return d.drop(columns="ym"), q


def station_boxes(data: Path, days):
    """IDEAM station-day totals averaged per box and UTC day: (day, NY, NX) with NaN where no station, and stations per box."""
    st = pd.read_csv(data / "ideam_stations.csv", dtype={"id": str})
    st = st.dropna(subset=["lat", "lon"])
    st["i"] = np.floor(st.lat - G.LAT[0] + 0.5).astype(int); st["j"] = np.floor(st.lon - G.LON[0] + 0.5).astype(int)
    st = st[(st.i >= 0) & (st.i < G.NY) & (st.j >= 0) & (st.j < G.NX)].set_index("id")
    frames = [pd.read_csv(f, dtype={"id": str, "date": str}) for f in sorted(data.glob("ideam_2*.csv.gz"))]
    d = pd.concat(frames); d = d[d.id.isin(st.index)]
    d["i"] = d.id.map(st.i); d["j"] = d.id.map(st.j); d["t"] = pd.to_datetime(d.date, format="%Y%m%d").dt.date
    pos = {x: k for k, x in enumerate(days)}
    d = d[d.t.isin(pos)]
    d, qc = screen_gauges(d, st)
    st = st[st.index.isin(set(d.id))]
    g = d.groupby(["t", "i", "j"]).mm.mean().reset_index()
    out = np.full((len(days), G.NY, G.NX), np.nan, dtype="float32")
    out[g.t.map(pos).values, g.i.values, g.j.values] = g.mm.values
    nst = np.zeros((G.NY, G.NX), int)
    for (i, j), k in st.groupby(["i", "j"]).size().items():
        nst[i, j] = k
    return out, nst, int(d.id.nunique()), qc


def monthly(a, days, min_days=25):
    """(day, NY, NX) -> {(y, m): (NY, NX) mean rate over the month's valid days, NaN with fewer than min_days}."""
    out = {}
    for ym in sorted({(d.year, d.month) for d in days}):
        k = [i for i, d in enumerate(days) if (d.year, d.month) == ym]
        x = a[k]; n = np.isfinite(x).sum(0)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(x, 0)
        out[ym] = np.where(n >= min_days, m, np.nan)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--json", required=True)
    ap.add_argument("--orog", default=str(Path(__file__).resolve().parents[1] / "mjo/data/reference/era5_orography.nc"))
    a = ap.parse_args(); data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1013)
    # ---- inputs
    li, lp = B.load_obs(data, "late"); fi, fp = B.load_obs(data, "final")
    days = sorted(li)
    D = B.truth_terms(li, lp, fi, fp, rng)
    mids = np.array([d.month for d in days])
    L = np.stack([lp[li[d]] for d in days])
    Ladj, _ = B.adjust_late(L, mids, D)
    P = {"imerg": Ladj if Ladj is not None else L}
    for src in PRODUCTS[1:]:
        idx, arr = load_grid(data, src)
        if idx:
            P[src] = utc_series(src, idx, arr, days)
            print(f"{src}: {len(idx)} days, valid {100 * np.isfinite(P[src]).mean():.0f} % of box-days", flush=True)
    # land: CPC or GPCC has data on most days
    land = np.zeros((G.NY, G.NX), bool)
    for src in ("cpc", "gpcc"):
        if src in P:
            land |= np.isfinite(P[src]).mean(0) > 0.8
    if "merge" in P:                                           # MERGE over the sea is IMERG Late again: keep it on land only
        P["merge"] = np.where(land[None], P["merge"], np.nan)
    import xarray as xr
    o = xr.open_dataset(a.orog)["geopotential_at_surface"].values                    # metres despite the name
    elev = G.ecmwf_to_1deg(np.roll(o[::-1], 720, axis=1)); cls = terrain_class(elev)
    # ---- station boxes; weights fitted and products scored by 5-fold cross-validation over the boxes (every box is scored by
    # weights fitted without it), and the blend-minus-product RMSE differences tested by resampling boxes
    S, nst, nstations, qc = station_boxes(data, days)
    Sm = monthly(S, days); Pm = {k: monthly(v, days) for k, v in P.items()}
    boxes = np.argwhere(np.isfinite(np.stack(list(Sm.values()))).any(0) & land)
    rng.shuffle(boxes); folds = np.array_split(np.arange(len(boxes)), NFOLD)
    print(f"IDEAM: {nstations} stations in {len(boxes)} land boxes, {NFOLD}-fold cross-validation", flush=True)

    def fit_weights(bx):
        stats = {}
        for k, pm in Pm.items():
            rows = [(pm[ym][i, j], sv[i, j], cls[i, j]) for ym, sv in Sm.items() if ym in pm for i, j in bx
                    if np.isfinite(sv[i, j]) and np.isfinite(pm[ym][i, j])]
            x = np.array(rows) if rows else np.zeros((0, 3)); stats[k] = {}
            for c, (_, _, lab) in enumerate(CLASS_EDGES):
                sel = x[x[:, 2] == c]
                if len(sel) >= 20:
                    stats[k][lab] = {"n": int(len(sel)), "ratio": float(sel[:, 0].sum() / sel[:, 1].sum()),
                                     "mse": float(np.mean((sel[:, 0] - sel[:, 1]) ** 2)), "r": float(np.corrcoef(sel[:, 0], sel[:, 1])[0, 1])}
        W = {lab: {k: 1.0 / stats[k][lab]["mse"] for k in stats if lab in stats[k]} for _, _, lab in CLASS_EDGES}
        for lab in W:
            tot = sum(W[lab].values()) or 1.0
            W[lab] = {k: v / tot for k, v in W[lab].items()}
        return stats, W

    def weight_maps(W):
        wmap = {k: np.zeros((G.NY, G.NX)) for k in P}
        for c, (_, _, lab) in enumerate(CLASS_EDGES):
            for k, v in W.get(lab, {}).items():
                wmap[k][(cls == c) & land] = v
        return wmap

    def blend_of(wmap):
        num = np.zeros((len(days), G.NY, G.NX), "float32"); den = np.zeros_like(num); npr = np.zeros(num.shape, "int8")
        for k, x in P.items():
            ok = np.isfinite(x) & (wmap[k][None] > 0)
            num += np.where(ok, x, 0) * wmap[k][None]; den += ok * wmap[k][None]; npr += ok
        with np.errstate(invalid="ignore", divide="ignore"):
            bl = np.where(den > 0, num / den, np.nan)
        bl = np.where(land[None], bl, P["imerg"])                                      # ocean: the adjusted IMERG
        return np.where(np.isfinite(bl), bl, P["imerg"]), npr                         # land days with no weighted product

    # out-of-fold rows: (box, class, gauge, product..., blend) on box-months where every product and the blend are present
    names = list(P) + ["blend"]; rows = []
    for f in folds:
        test = boxes[f]; fitb = boxes[np.setdiff1d(np.arange(len(boxes)), f)]
        _, Wf = fit_weights(fitb)
        Bf = monthly(blend_of(weight_maps(Wf))[0], days)
        for ym, sv in Sm.items():
            for q, (i, j) in zip(f, test):
                if not np.isfinite(sv[i, j]):
                    continue
                vals = [Pm[k].get(ym, np.full((G.NY, G.NX), np.nan))[i, j] for k in P] + [Bf[ym][i, j]]
                if np.all(np.isfinite(vals)):
                    rows.append([q, cls[i, j], sv[i, j]] + vals)
    R = np.array(rows) if rows else np.zeros((0, 3 + len(names)))
    score, vs = {k: {} for k in names}, {k: {} for k in P}
    for c, (_, _, lab) in enumerate(CLASS_EDGES):
        sel = R[R[:, 1] == c]
        if len(sel) < 20:
            continue
        ub = np.unique(sel[:, 0]); y = sel[:, 2]
        for q, k in enumerate(names):
            x = sel[:, 3 + q]
            score[k][lab] = {"n": int(len(sel)), "boxes": int(len(ub)), "ratio": round(float(x.sum() / y.sum()), 3),
                             "rmse": round(float(np.sqrt(np.mean((x - y) ** 2))), 3), "r": round(float(np.corrcoef(x, y)[0, 1]), 3)}
        # resample boxes (box-months within a box are not independent)
        se = {k: np.array([np.sum((sel[sel[:, 0] == b, 3 + q] - sel[sel[:, 0] == b, 2]) ** 2) for b in ub]) for q, k in enumerate(names)}
        nb = np.array([np.sum(sel[:, 0] == b) for b in ub])
        draw = rng.integers(0, len(ub), (NBOOT, len(ub)))
        rm = {k: np.sqrt(se[k][draw].sum(1) / nb[draw].sum(1)) for k in names}
        for k in P:
            dd = rm["blend"] - rm[k]; lo, hi = np.quantile(dd, [0.025, 0.975])
            vs[k][lab] = {"diff": round(float(score["blend"][lab]["rmse"] - score[k][lab]["rmse"]), 3), "lo": round(float(lo), 3),
                          "hi": round(float(hi), 3), "sig": bool(hi < 0 or lo > 0)}
    for name, sc in score.items():
        print(f"  cv {name:7s} " + "  ".join(f"{lab.split(' ')[0]}: ratio {v['ratio']:.2f} rmse {v['rmse']:.2f} r {v['r']:.2f} (n {v['n']}, {v['boxes']} boxes)"
                                            for lab, v in sc.items()), flush=True)
    for k, v in vs.items():
        print(f"  blend - {k:7s} rmse " + "  ".join(f"{lab.split(' ')[0]}: {x['diff']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]{' *' if x['sig'] else ''}"
                                                   for lab, x in v.items()), flush=True)
    # ---- the published blend: weights fitted on every box
    stats, W = fit_weights(boxes)
    print("weights:", json.dumps({lab: {k: round(v, 2) for k, v in w.items()} for lab, w in W.items()}), flush=True)
    wmap = weight_maps(W)
    blend, npr = blend_of(wmap)
    # ---- truth spread by season: sd of the products' seasonal means, each product on the SAME days (those on which every product
    # covering the box has data), so a product with a shorter record does not pass off a different period as disagreement
    spread = {}
    stack = np.stack([P[k] for k in P])                                              # (product, day, NY, NX)
    for sk, months in B.SEASONS.items():
        kd = [i for i, d in enumerate(days) if d.month in months]
        x = stack[:, kd]; fin = np.isfinite(x)
        avail = fin.mean(1) >= 0.5                                                   # products covering the box this season
        okd = np.where(avail[:, None], fin, True).all(0) & (avail.sum(0) >= 2)[None]  # shared days
        n = okd.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(avail & (n >= 20)[None], np.where(okd[None], np.nan_to_num(x), 0).sum(1) / n[None], np.nan)
            sd = np.nanstd(means, 0)
        spread[sk] = np.where(land & np.isfinite(sd) & (np.isfinite(means).sum(0) >= 2), sd, 0.0).astype("float32")
    np.savez_compressed(out / "blend_spread.npz", lat=G.LAT, lon=G.LON, land=land, elev=elev, nst=nst, **spread,
                        **{f"w_{k}": v for k, v in wmap.items()})
    for ym in sorted({(d.year, d.month) for d in days}):
        k = [i for i, d in enumerate(days) if (d.year, d.month) == ym]
        np.savez_compressed(out / f"blend_{ym[0]}{ym[1]:02d}.npz", dates=np.array([days[i].strftime("%Y%m%d") for i in k]),
                            pr=blend[k].astype("float32"), nprod=npr[k], lat=G.LAT, lon=G.LON)
    summ = {"generated": datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "days": [str(days[0]), str(days[-1])], "products": list(P),
            "ideam_stations": nstations, "ideam_qc": qc, "station_boxes": int(len(boxes)), "classes": [c[2] for c in CLASS_EDGES],
            "folds": NFOLD, "fit": stats, "weights": W, "test": score, "blend_vs": vs}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True); Path(a.json).write_text(json.dumps(summ, indent=1))
    print(f"blend {len(days)} days -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
