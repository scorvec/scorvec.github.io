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
  1. Station boxes: IDEAM stations averaged per 1-degree box and UTC day; a box-month needs >= 25 days.
  2. Each product against the station box-months, by terrain class of the box (lowland < 500 m, hills 500-1500, mountains >= 1500
     m mean ERA5 height): mean ratio (product / gauges) and mean-squared error of the monthly mean rate. Boxes are split at
     random into a fitting 70 % and a test 30 %.
  3. Blend over land = sum_k w_k P_k / sum_k w_k over the products present that day, w_k = 1 / MSE_k of the box's terrain class
     (fitted boxes). Over the ocean: the adjusted IMERG alone.
  4. The blend and every product are scored on the test boxes, on the SAME box-months (all present); truth uncertainty per box
     and season = the spread (standard deviation) of the products' seasonal means.

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
    g = d.groupby(["t", "i", "j"]).mm.mean().reset_index()
    out = np.full((len(days), G.NY, G.NX), np.nan, dtype="float32")
    out[g.t.map(pos).values, g.i.values, g.j.values] = g.mm.values
    nst = np.zeros((G.NY, G.NX), int)
    for (i, j), k in st.groupby(["i", "j"]).size().items():
        nst[i, j] = k
    return out, nst, int(d.id.nunique())


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
    # ---- station boxes and validation
    S, nst, nstations = station_boxes(data, days)
    Sm = monthly(S, days); Pm = {k: monthly(v, days) for k, v in P.items()}
    boxes = np.argwhere(np.isfinite(np.stack(list(Sm.values()))).any(0) & land)
    rng.shuffle(boxes); nfit = int(0.7 * len(boxes)); fit, test = boxes[:nfit], boxes[nfit:]
    print(f"IDEAM: {nstations} stations in {len(boxes)} land boxes ({nfit} fit, {len(boxes) - nfit} test)", flush=True)

    def pairs(prod_m, bx):
        """All (product, station, class) box-month pairs on the boxes bx."""
        rows = []
        for ym, s in Sm.items():
            p = prod_m.get(ym)
            if p is None:
                continue
            for i, j in bx:
                if np.isfinite(s[i, j]) and np.isfinite(p[i, j]):
                    rows.append((p[i, j], s[i, j], cls[i, j]))
        return np.array(rows) if rows else np.zeros((0, 3))

    stats = {}
    for k, pm in Pm.items():
        x = pairs(pm, fit)
        stats[k] = {}
        for c, (_, _, lab) in enumerate(CLASS_EDGES):
            sel = x[x[:, 2] == c] if len(x) else x
            if len(sel) < 20:
                continue
            stats[k][lab] = {"n": int(len(sel)), "ratio": float(sel[:, 0].sum() / sel[:, 1].sum()),
                             "mse": float(np.mean((sel[:, 0] - sel[:, 1]) ** 2)), "r": float(np.corrcoef(sel[:, 0], sel[:, 1])[0, 1])}
    W = {lab: {k: 1.0 / stats[k][lab]["mse"] for k in stats if lab in stats[k]} for _, _, lab in CLASS_EDGES}
    for lab in W:
        tot = sum(W[lab].values()) or 1.0
        W[lab] = {k: v / tot for k, v in W[lab].items()}
    print("weights:", json.dumps({lab: {k: round(v, 2) for k, v in w.items()} for lab, w in W.items()}), flush=True)
    # ---- the blend
    wmap = {k: np.zeros((G.NY, G.NX)) for k in P}
    for c, (_, _, lab) in enumerate(CLASS_EDGES):
        for k, v in W.get(lab, {}).items():
            wmap[k][(cls == c) & land] = v
    num = np.zeros((len(days), G.NY, G.NX), "float32"); den = np.zeros_like(num); npr = np.zeros(num.shape, "int8")
    for k, x in P.items():
        ok = np.isfinite(x) & (wmap[k][None] > 0)
        num += np.where(ok, x, 0) * wmap[k][None]; den += ok * wmap[k][None]; npr += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        blend = np.where(den > 0, num / den, np.nan)
    blend = np.where(land[None], blend, P["imerg"])                                   # ocean: the adjusted IMERG
    blend = np.where(np.isfinite(blend), blend, P["imerg"])                           # land days with no weighted product
    # ---- test-box scores on the SAME box-months for every product and the blend (all of them present)
    Bm = monthly(blend, days)
    allm = dict(Pm); allm["blend"] = Bm
    rows = []
    for ym, sv in Sm.items():
        for i, j in test:
            if not np.isfinite(sv[i, j]):
                continue
            vals = [allm[k].get(ym, np.full((G.NY, G.NX), np.nan))[i, j] for k in allm]
            if np.all(np.isfinite(vals)):
                rows.append([sv[i, j], cls[i, j]] + vals)
    R = np.array(rows) if rows else np.zeros((0, 2 + len(allm)))
    score = {k: {} for k in allm}
    for c, (_, _, lab) in enumerate(CLASS_EDGES):
        sel = R[R[:, 1] == c] if len(R) else R
        if len(sel) < 10:
            continue
        for q, k in enumerate(allm):
            x = sel[:, 2 + q]; y = sel[:, 0]
            score[k][lab] = {"n": int(len(sel)), "ratio": round(float(x.sum() / y.sum()), 3),
                             "rmse": round(float(np.sqrt(np.mean((x - y) ** 2))), 3), "r": round(float(np.corrcoef(x, y)[0, 1]), 3)}
    for name, sc in score.items():
        print(f"  test {name:7s} " + "  ".join(f"{lab.split(' ')[0]}: ratio {v['ratio']:.2f} rmse {v['rmse']:.2f} r {v['r']:.2f} (n {v['n']})" for lab, v in sc.items()), flush=True)
    # ---- truth spread by season (sd of the products' seasonal means, land; 0 over the sea)
    spread = {}
    for sk, months in B.SEASONS.items():
        k = [i for i, d in enumerate(days) if d.month in months]
        means = np.stack([np.nanmean(x[k], 0) for x in P.values()])
        with np.errstate(invalid="ignore"):
            sd = np.nanstd(means, 0)
        spread[sk] = np.where(land & (np.isfinite(means).sum(0) >= 2), sd, 0.0).astype("float32")
    np.savez_compressed(out / "blend_spread.npz", lat=G.LAT, lon=G.LON, land=land, elev=elev, nst=nst, **spread,
                        **{f"w_{k}": v for k, v in wmap.items()})
    for ym in sorted({(d.year, d.month) for d in days}):
        k = [i for i, d in enumerate(days) if (d.year, d.month) == ym]
        np.savez_compressed(out / f"blend_{ym[0]}{ym[1]:02d}.npz", dates=np.array([days[i].strftime("%Y%m%d") for i in k]),
                            pr=blend[k].astype("float32"), nprod=npr[k], lat=G.LAT, lon=G.LON)
    summ = {"generated": datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "days": [str(days[0]), str(days[-1])], "products": list(P),
            "ideam_stations": nstations, "station_boxes": int(len(boxes)), "classes": [c[2] for c in CLASS_EDGES],
            "fit": stats, "weights": W, "test": score}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True); Path(a.json).write_text(json.dumps(summ, indent=1))
    print(f"blend {len(days)} days -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
