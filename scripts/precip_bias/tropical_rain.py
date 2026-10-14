#!/usr/bin/env python3
"""Bias-adjusted ensemble rainfall forecast, weeks 1 and 2, global band and a flat tropics strip (2026-10-14, user: "global
bias-adjusted rainfall/rainfall anomaly plots from the aifs/ifs ensembles and put that somewhere on the site in the tropics
section? (do a flat tropics zoom)" - "use your best judgement"). Shown on circulation.html#rainfc.

The latest 00Z IFS-ENS and AIFS-ENS run with day 14 on the Google Cloud mirror (the only ECMWF host): all 50 perturbed
members, accumulated rain at 168 h and 336 h, as 1° box means (grid.py, global band 60°S-60°N).
  correction  every member's week total times the model's significant rain-bias ratio for that week (global_bias.py: current
              model cycle vs IMERG adjusted to Final, season-weighted, 2° boxes, bilinear to 1°);
  normal      GPCP V3.2 daily 2001-2020 (gpcp_clim.py): the mean and terciles of 7-day totals starting near the week's first
              day of year (IMERG Final-based, so the same yardstick as the corrected forecast);
  sources     both ensembles pooled (100 members, each ensemble half the weight), or each alone.
Maps (weeks 1 and 2 stacked): corrected rain (mm/day); anomaly against the normal (mm/day) and the favoured tercile with its
probability - both only where the members' tercile counts differ from the climatological 1/3 (one-sided binomial test per
tercile, the two directions Bonferroni-combined, BH-FDR q 0.10 over the map); and the correction applied per ensemble.
Cells whose normal week has an upper tercile under 2 mm (dry season / desert) get no tercile or anomaly test.

    python scripts/precip_bias/tropical_rain.py --ref DIR --out assets/sst [--date YYYYMMDD]
-> assets/sst/rainfc_{both,ifs,aifs}_{rain,anom,terc}_{tropics,global}.webp, rainfc_{ifs,aifs}_bias_{tropics,global}.webp,
   assets/sst/data/rainfc.json
"""
from __future__ import annotations

import pyproj  # noqa: F401  - before eccodes (repo gotcha)

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecmwf")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "sst"))
import grid as G                                          # noqa: E402

MODELS = {"ifs": "IFS-ENS", "aifs-ens": "AIFS-ENS"}
SHORT = {"ifs": "ifs", "aifs-ens": "aifs"}
MEMBERS = list(range(1, 51))
STEPS = (168, 336)
DRY_Q67 = 2.0                                             # mm per week
DOMAINS = {"tropics": (-30, 30), "global": (-60, 60)}
INK, MUTED = "#1a1a1a", "#6f6b64"


# ── data ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def latest_run(today: date) -> date:
    import rangefetch as R
    for k in range(3):
        d = today - timedelta(days=k)
        if all(R.cycle_complete(f"{d:%Y%m%d}", "00", m, last_step=STEPS[-1]) for m in MODELS):
            return d
    raise SystemExit("no 00Z run with day 14 complete for both ensembles in the last 3 days")


def members(model: str, d: date) -> np.ndarray:
    """(member, step, NY, NX) accumulated rain, mm."""
    import rangefetch as R
    import ecmwf_tp as E
    out = np.full((len(MEMBERS), len(STEPS), G.NY, G.NX), np.nan, "float32")
    for si, step in enumerate(STEPS):
        for k in range(4):
            try:
                f = E.decode(R.fetch(f"{d:%Y%m%d}", "00", model, step, param="tp", numbers=MEMBERS, kind="pf"))
                miss = [m for m in MEMBERS if m not in f]
                if len(miss) > 5:
                    raise ValueError(f"{len(miss)} members missing")
                for mi, m in enumerate(MEMBERS):
                    if m in f:
                        out[mi, si] = G.ecmwf_to_1deg(f.pop(m))
                break
            except Exception as ex:                                              # noqa: BLE001
                if k == 3:
                    raise
                print(f"  {model} +{step} h: {repr(ex)[:90]}; retrying", flush=True); time.sleep(10 * (k + 1))
    ok = np.isfinite(out).all(axis=(1, 2, 3))
    return out[ok]


def upsample(r2: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """2° box field -> 1° grid, bilinear in log, longitude periodic, latitude clamped at the edges."""
    from scipy.interpolate import RegularGridInterpolator
    lr = np.log(np.where(np.isfinite(r2) & (r2 > 0), r2, 1.0))
    lonp = np.concatenate([[lon2[-1] - 360], lon2, [lon2[0] + 360]]); lrp = np.concatenate([lr[:, -1:], lr, lr[:, :1]], axis=1)
    f = RegularGridInterpolator((lat2, lonp), lrp, bounds_error=False, fill_value=None)
    la = np.clip(G.LAT, lat2[0], lat2[-1])
    pts = np.stack(np.meshgrid(la, G.LON, indexing="ij"), -1).reshape(-1, 2)
    return np.exp(f(pts)).reshape(G.NY, G.NX).astype("float32")


def clim_at(C, start: date) -> dict:
    """Normal 7-day totals for the week starting on `start` (linear between pentads)."""
    doy = start.timetuple().tm_yday
    if start.year % 4 == 0 and start.month > 2:
        doy -= 1
    doy = min(doy, 365)
    s = C["start_doy"]; p0 = int((doy - 1) // 5); p1 = (p0 + 1) % len(s)
    span = (s[p1] - s[p0]) % 365 or 5; fr = ((doy - s[p0]) % 365) / span
    return {k: (1 - fr) * C[k][p0] + fr * C[k][p1] for k in ("mean", "q33", "q50", "q67")}


def tercile_test(X: np.ndarray, cl: dict, weights: np.ndarray):
    """X (member, NY, NX) week totals; weights per member (sum 1). -> P(below), P(above), favoured sign (+1/-1/0 where not
    significant), the testable mask. One-sided binomial tests against 1/3 on the effective member count, the two
    directions combined (x2), BH-FDR q 0.10 over the testable cells."""
    from scipy.stats import binom
    import build_bias as BB
    n_eff = 1.0 / (weights ** 2).sum()
    pb = np.tensordot(weights, (X < cl["q33"][None]).astype("float32"), axes=1)
    pa = np.tensordot(weights, (X > cl["q67"][None]).astype("float32"), axes=1)
    test = np.isfinite(cl["q67"]) & (cl["q67"] >= DRY_Q67) & np.isfinite(X).all(0)
    n = int(round(n_eff))
    p_up = binom.sf(np.round(pa * n) - 1, n, 1 / 3); p_dn = binom.sf(np.round(pb * n) - 1, n, 1 / 3)
    p = np.where(test, np.minimum(1.0, 2 * np.minimum(p_up, p_dn)), np.nan)
    sig = BB.bh_fdr(p, q=0.10) & test
    fav = np.where(sig, np.where(pa >= pb, 1, -1), 0)
    return pb, pa, fav, test


# ── drawing ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def palettes():
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import matplotlib.cm as cm
    rain_lev = [0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30]
    rain_col = ["#e9f3e1", "#c7e4b6", "#94cf8f", "#5cb37c", "#2f9475", "#227a8e", "#2b5fa3", "#4a3f9e", "#6d2b8e", "#98207a", "#c2186a"]
    anom_lev = [-12, -8, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 8, 12]
    a = cm.get_cmap("BrBG", 15)(np.linspace(0, 1, 15)); a[7] = (1, 1, 1, 1)
    terc_lev = [-100, -90, -80, -70, -60, -50, -40, 40, 50, 60, 70, 80, 90, 100]
    t = cm.get_cmap("BrBG", 15)(np.linspace(0.02, 0.98, 15)); t = np.delete(t, 7, axis=0); t = np.insert(t, 6, (1, 1, 1, 1), axis=0)
    bias_lev = [1 / 3, 0.5, 0.67, 0.8, 0.9, 1 / 0.9, 1 / 0.8, 1 / 0.67, 2, 3]
    b = cm.get_cmap("BrBG", 11)(np.linspace(0.05, 0.95, 11)); b[5] = (1, 1, 1, 1)
    def mk(lev, cols, ext):
        cmap = ListedColormap(cols[1:-1] if ext == "both" else cols)
        if ext == "both":
            cmap.set_under(cols[0]); cmap.set_over(cols[-1])
        return lev, cmap, BoundaryNorm(lev, cmap.N)
    rl = rain_lev; rc = ListedColormap(rain_col[1:]); rc.set_under("#ffffff"); rc.set_over(rain_col[-1])
    return {"rain": (rl, rc, BoundaryNorm(rl, rc.N), "max"),
            "anom": mk(anom_lev, a, "both") + ("both",),
            "terc": (terc_lev, ListedColormap(t[1:-1]), BoundaryNorm(terc_lev, 13), "neither"),
            "bias": mk(bias_lev, b, "both") + ("both",)}


def draw(panels, kind: str, dom: str, title: str, sub: str, cbar: str, out: Path, lat=None, lon=None, grey=None):
    """panels: [(label, field (NY, NX))] stacked; grey: optional [(NY, NX) bool] per panel = no test possible (dry season)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import mapstyle as MS
    lat = G.LAT if lat is None else lat; lon = G.LON if lon is None else lon
    lev, cmap, norm, ext = palettes()[kind]
    s, n = DOMAINS[dom]; W = 14.0; mh = W * 0.94 * (n - s) / 360.0; gap = 0.32; top, bot = 0.95, 0.85
    H = top + bot + len(panels) * mh + (len(panels) - 1) * gap
    fig = plt.figure(figsize=(W, H)); pc = ccrs.PlateCarree(); proj = ccrs.PlateCarree(central_longitude=180)
    # cell edges for pcolormesh (1° or 2° boxes)
    dl = float(lat[1] - lat[0]); dn = float(lon[1] - lon[0])
    le = np.r_[lat - dl / 2, lat[-1] + dl / 2]; oe = np.r_[lon - dn / 2, lon[-1] + dn / 2]
    rows = (lat >= s - dl) & (lat <= n + dl); re = np.r_[le[:-1][rows], le[1:][rows][-1]]
    for i, (lab, fld) in enumerate(panels):
        y0 = (bot + (len(panels) - 1 - i) * (mh + gap)) / H
        ax = fig.add_axes([0.03, y0, 0.94, mh / H], projection=proj)
        ax.set_extent([-180, 180, s, n], crs=proj)
        m = ax.pcolormesh(oe, re, np.ma.masked_invalid(fld[rows]), cmap=cmap, norm=norm, transform=pc, zorder=1, shading="flat")
        if grey is not None and grey[i] is not None:
            ax.pcolormesh(oe, re, np.ma.masked_where(~grey[i][rows], np.ones(int(rows.sum()) * len(lon)).reshape(-1, len(lon))),
                          cmap=matplotlib.colors.ListedColormap(["#e4e2dc"]), transform=pc, zorder=1, shading="flat")
        MS.features(ax, states=False)
        ax.text(0.006, 0.965, lab, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="top", color=INK, zorder=6,
                bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
    MS.heading(fig, H, title, sub)
    cb = MS.colorbar(fig, H, m, cbar, extend=ext)
    if kind == "terc":
        cb.set_ticks(lev); cb.set_ticklabels([f"{abs(x)}" for x in lev])
    if kind == "bias":
        cb.set_ticks(lev); cb.set_ticklabels(["⅓", "½", "⅔", "0.8", "0.9", "1.1", "1.25", "1.5", "2", "3"])
    MS.save(fig, out)


# ── main ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="dir with bias_ratio_global.npz and gpcp_clim_week_global.npz")
    ap.add_argument("--out", required=True); ap.add_argument("--date", help="init YYYYMMDD (default: the latest complete 00Z)")
    ap.add_argument("--skip-if-current", action="store_true", help="exit 0 without work when that run is already published")
    a = ap.parse_args()
    G.use_domain("global")
    ref, out = Path(a.ref), Path(a.out); (out / "data").mkdir(parents=True, exist_ok=True)
    B = np.load(ref / "bias_ratio_global.npz"); bmeta = json.loads(str(B["meta"]))
    C = np.load(ref / "gpcp_clim_week_global.npz")
    init = datetime.strptime(a.date, "%Y%m%d").date() if a.date else latest_run(datetime.now(timezone.utc).date())
    print(f"init {init} 00Z", flush=True)
    pub = out / "data" / "rainfc.json"
    if a.skip_if_current and pub.exists() and json.loads(pub.read_text()).get("init") == f"{init}":
        print("already published - nothing to do", flush=True); return 0
    weeks = {"wk1": init, "wk2": init + timedelta(days=7)}
    X, ratios, info = {}, {}, {"init": f"{init}", "members": {}}
    for model in MODELS:
        acc = members(model, init); info["members"][model] = int(acc.shape[0])
        X[model] = {"wk1": np.maximum(acc[:, 0], 0), "wk2": np.maximum(acc[:, 1] - acc[:, 0], 0)}
        for wk in weeks:
            r = upsample(B[f"{model}_{wk}_ratio"], B["lat"], B["lon"]); ratios[(model, wk)] = r
            X[model][wk] = X[model][wk] * r[None]
    cl = {wk: clim_at(C, d) for wk, d in weeks.items()}
    srcs = {"both": list(MODELS), "ifs": ["ifs"], "aifs": ["aifs-ens"]}
    label = {"both": "IFS-ENS + AIFS-ENS", "ifs": "IFS-ENS", "aifs": "AIFS-ENS"}
    fmt = lambda d: f"{d:%-d %b}"
    rng_ = lambda d, e: f"{d.day}–{e.day} {e:%b}" if d.month == e.month else f"{fmt(d)}–{fmt(e)}"
    wl = {wk: f"Week {wk[-1]} · {rng_(d, d + timedelta(days=6))}" for wk, d in weeks.items()}
    stats = {}
    for sk, ms in srcs.items():
        res = {}
        for wk in weeks:
            Xs = np.concatenate([X[m][wk] for m in ms]); wts = np.concatenate([np.full(X[m][wk].shape[0], 1.0 / (len(ms) * X[m][wk].shape[0])) for m in ms])
            mean = np.tensordot(wts, Xs, axes=1)
            pb, pa, fav, test = tercile_test(Xs, cl[wk], wts)
            anom = np.where(fav != 0, (mean - cl[wk]["mean"]) / 7.0, np.nan)
            terc = np.where(fav > 0, 100 * pa, np.where(fav < 0, -100 * pb, np.nan))
            res[wk] = {"rain": mean / 7.0, "anom": anom, "terc": terc, "dry": ~test}
            band = (np.abs(G.LAT) <= 30)[:, None] & test
            stats[f"{sk}_{wk}"] = {"tested_cells": int(test.sum()), "sig_frac_tropics": round(float((fav != 0)[band].mean()), 3),
                                   "above_frac_tropics": round(float((fav > 0)[band].mean()), 3), "below_frac_tropics": round(float((fav < 0)[band].mean()), 3)}
        n_mem = sum(info["members"][m] for m in ms)
        for dom in DOMAINS:
            head = f"{label[sk]} ({n_mem} members) · 00Z {fmt(init)} {init.year}"
            draw([(wl[w], res[w]["rain"]) for w in weeks], "rain", dom, f"{head} · bias-adjusted rain",
                 "Week mean rain rate after each ensemble's significant rain bias against IMERG (adjusted to Final) is removed, member by member.",
                 "mm per day", out / f"rainfc_{sk}_rain_{dom}.webp")
            draw([(wl[w], res[w]["anom"]) for w in weeks], "anom", dom, f"{head} · rain anomaly",
                 "Bias-adjusted ensemble mean minus the GPCP 2001–2020 normal for the same week; shown only where the members' tercile split differs significantly from a normal week (FDR 10 %). Grey: too dry in a normal week to test.",
                 "mm per day vs normal", out / f"rainfc_{sk}_anom_{dom}.webp", grey=[res[w]["dry"] for w in weeks])
            draw([(wl[w], res[w]["terc"]) for w in weeks], "terc", dom, f"{head} · favoured tercile",
                 "Chance (%) of the favoured tercile against the GPCP 2001–2020 normal week: green = above normal, brown = below normal; shown only where significant (FDR 10 %). Grey: too dry to test.",
                 "probability of the below-normal (brown) or above-normal (green) tercile, %", out / f"rainfc_{sk}_terc_{dom}.webp",
                 grey=[res[w]["dry"] for w in weeks])
    for model in MODELS:
        for dom in DOMAINS:
            panels = []
            for wk in weeks:
                m = bmeta[f"{model}_{wk}"]; r2 = B[f"{model}_{wk}_ratio"]
                panels.append((f"Week {wk[-1]} · {m['runs']} runs {m['first']} to {m['last']}", np.where(r2 != 1.0, 1.0 / r2, np.nan)))
            draw(panels, "bias", dom, f"{MODELS[model]} {bmeta[f'{model}_wk1']['cycle']} · the rain bias removed",
                 "Model week rain ÷ IMERG (adjusted to Final), season-weighted runs of this model cycle; only where significant (FDR 10 %), shrunk by its uncertainty. Green: model too wet, brown: too dry.",
                 "model ÷ observed rain", out / f"rainfc_{SHORT[model]}_bias_{dom}.webp", lat=B["lat"], lon=B["lon"])
    info.update(weeks={wk: [f"{d}", f"{d + timedelta(days=6)}"] for wk, d in weeks.items()}, stats=stats, bias=bmeta,
                normal="GPCP V3.2 daily 2001-2020, 7-day totals", made=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"))
    (out / "data" / "rainfc.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
