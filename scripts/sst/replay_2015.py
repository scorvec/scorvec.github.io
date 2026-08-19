#!/usr/bin/env python3
"""The current model vs the catchment-decomposed one, replayed on 2015-16.

Everything validated since the basin-mean baseline, applied to the
hardest event in the record:

  * rain decomposed into the 24 DISTINCT catchment shapes rather than six
    basin means (identical polygons collapsed, energy summed) — worth
    +0.027 overall but +23% on top-decile rises, because averaging across
    disconnected catchments dilutes exactly the localised storms that
    cause spikes;
  * drift-robust recession baseline (365-day trailing mean) — Colombia's
    basins drift, VALLE +15.8%/decade;
  * amplitude calibration and tau/lag re-selected inside every training
    fold.

The window and a 90-day embargo are withheld from the fit, and a fresh
15-day forecast is launched from every day inside it, so nothing here
saw 2015-16.

Quantile mapping is NOT applied: it corrects NWP rain bias, and this
replay is driven by observed rain (no 2015 forecast archive exists).

    python scripts/sst/replay_2015.py [--window 2015-08:2016-03]

Outputs: ~/colombia_hydro/out/replay_2015.json
         ~/colombia_hydro/site/replay_2015.webp
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perfect_rain_backtest as PR                                  # noqa: E402
import delta_backtest_long as DB                                    # noqa: E402
import inflow_delta_model as M                                      # noqa: E402
import national_inflow as NI                                        # noqa: E402

PRIV = Path.home() / "colombia_hydro"
CATCH = PRIV / "raw" / "catchment_rain.npz"
OUT_JSON = PRIV / "out" / "replay_2015.json"
OUT_PNG = PRIV / "site" / "replay_2015.webp"
MAXLEAD = 15
EMB = 90


def load():
    d = DB.add_national(PR.load_all())
    dates = np.array([str(x) for x in d["dates"]])
    z = np.load(CATCH, allow_pickle=True)
    keys = list(z["keys"]); R = z["rain"]; C = z["clim"]; meta = z["meta"].item()
    _, i1, i2 = np.intersect1d(dates, z["dates"], return_indices=True)
    # collapse identical masks; several rivers share one polygon
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(hash(R[:, i].tobytes()), []).append(i)
    grp = sorted([(g[0], sum(meta[keys[j]]["energy_gwh"] for j in g))
                  for g in groups.values()], key=lambda x: -x[1])
    anom = {i: (R[:, i] - C[:, i])[i2] for i, _ in grp}
    out = {k: (v[i1] if isinstance(v, np.ndarray) else v)
           for k, v in (("dates", dates), ("roni", d["roni"]))}
    out["y"] = d["y"]["NATIONAL"][i1]
    out["stor"] = d["stor"]["NATIONAL"][i1]
    out["pooled"] = d["rain"]["NATIONAL"][i1]
    out["anom"] = anom
    out["grp"] = grp
    return out


def replay(D, a0, a1, mode):
    y, pooled, roni, stor = D["y"], D["pooled"], D["roni"], D["stor"]
    dates = D["dates"]; n = len(y)
    ym = np.array([s[:7] for s in dates])
    inper = (ym >= a0) & (ym <= a1)
    hi = np.where(inper)[0]
    tr = ~inper
    tr[max(0, hi[0] - EMB):min(n, hi[-1] + EMB + 1)] = False
    lg, em = M.lagged, M.ema

    def feats(tau):
        cols = [lg(y, 1) - M.baseline_series(y, 365)]
        if mode == "pooled":
            cols += [lg(pooled, 0), lg(em(pooled, tau), 0)]
        else:
            for i, e in D["grp"]:
                if e < 1.0:
                    continue
                cols += [lg(D["anom"][i], 0), lg(em(D["anom"][i], tau), 0)]
        cols += [lg(em(pooled, 90), 0), roni, stor]
        return np.column_stack(cols)

    best = None
    dy = np.full(n, np.nan); dy[1:] = y[1:] - y[:-1]
    for tau in (2, 4, 7, 12, 20, 30):
        X = feats(tau)
        m = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        if m.sum() < 400:
            continue
        A = np.column_stack([np.ones(m.sum()), X[m]])
        b, *_ = np.linalg.lstsq(A, dy[m], rcond=None)
        sc = np.corrcoef(A @ b, dy[m])[0, 1]
        if best is None or sc > best[0]:
            best = (sc, tau, b, X)
    _, tau, beta, X = best

    # amplitude calibration by inner CV inside the training set only
    idx = np.where(tr)[0]
    inner = np.linspace(0, len(idx), 11).astype(int)
    P, O = {h: [] for h in range(1, MAXLEAD + 1)}, {h: [] for h in range(1, MAXLEAD + 1)}
    for k in range(10):
        te = np.zeros(n, bool); te[idx[inner[k]:inner[k + 1]]] = True
        tr2 = tr & ~te
        m = tr2 & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        if m.sum() < 300:
            continue
        A = np.column_stack([np.ones(m.sum()), X[m]])
        b2, *_ = np.linalg.lstsq(A, dy[m], rcond=None)
        for i0 in np.where(te)[0][::3]:
            if not np.isfinite(y[i0]):
                continue
            yp = y[i0]
            for j in range(MAXLEAD):
                i = i0 + 1 + j
                if i >= n or not np.all(np.isfinite(X[i])):
                    break
                xf = X[i].copy(); xf[0] = yp - 100.0
                yp = float(np.clip(yp + b2[0] + xf @ b2[1:], 0, 400))
                if np.isfinite(y[i]):
                    P[j + 1].append(yp - y[i0]); O[j + 1].append(y[i] - y[i0])
    sh = np.ones(MAXLEAD); off = np.zeros(MAXLEAD)
    for j in range(MAXLEAD):
        p, o = np.asarray(P[j + 1]), np.asarray(O[j + 1])
        if len(p) > 60 and np.std(p) > 1e-6:
            A = np.column_stack([np.ones(len(p)), p])
            c, *_ = np.linalg.lstsq(A, o, rcond=None)
            off[j], sh[j] = float(c[0]), float(np.clip(c[1], 0, 1.5))

    rows = {h: {"p": [], "o": [], "q": []} for h in range(1, MAXLEAD + 1)}
    traj = []
    for i0 in hi:
        if not np.isfinite(y[i0]):
            continue
        yp = y[i0]; path = []
        for j in range(MAXLEAD):
            i = i0 + 1 + j
            if i >= n or not np.all(np.isfinite(X[i])):
                break
            xf = X[i].copy(); xf[0] = yp - 100.0
            yp = float(np.clip(yp + beta[0] + xf @ beta[1:], 0, 400))
            cal = y[i0] + off[j] + sh[j] * (yp - y[i0])
            path.append(cal)
            if np.isfinite(y[i]):
                rows[j + 1]["p"].append(cal); rows[j + 1]["o"].append(y[i])
                rows[j + 1]["q"].append(y[i0])
        traj.append({"init": dates[i0], "y0": float(y[i0]),
                     "path": [round(float(v), 1) for v in path]})
    res = {"tau": int(tau), "mode": mode, "leads": {}, "traj": traj,
           "n_features": int(X.shape[1])}
    for h in range(1, MAXLEAD + 1):
        p, o, q = (np.asarray(rows[h][k]) for k in ("p", "o", "q"))
        if len(p) < 15:
            continue
        rm = float(np.sqrt(np.mean((p - o) ** 2)))
        rp = float(np.sqrt(np.mean((q - o) ** 2)))
        dch, och = p - q, o - q
        big = och >= np.percentile(och, 90)
        res["leads"][h] = {
            "n": int(len(p)), "rmse": round(rm, 2), "rmse_pers": round(rp, 2),
            "skill": round(1 - rm / rp, 3),
            "change_r": round(float(np.corrcoef(dch, och)[0, 1]), 3),
            "rise_r": round(float(np.corrcoef(dch[big], och[big])[0, 1]), 3)
            if big.sum() > 10 else None,
            "rise_amp": round(float(np.mean(dch[big]) / max(np.mean(och[big]), 1e-6)), 3)
            if big.sum() > 10 else None}
    return res


def figure(D, res, a0, a1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    NAVY, INK = "#13273d", "#1a2733"
    dates = D["dates"]; y = D["y"]
    ym = np.array([s[:7] for s in dates])
    k = np.where((ym >= a0) & (ym <= a1))[0]
    t = [datetime.strptime(dates[i], "%Y-%m-%d") for i in k]

    fig = plt.figure(figsize=(15.2, 9.4))
    hd = fig.add_axes([0, 0.935, 1, 0.065]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.013, 0.62, "2015-16 SUPER EL NIÑO — CURRENT MODEL vs "
            "CATCHMENT-DECOMPOSED", transform=hd.transAxes, color="white",
            fontsize=14.5, fontweight="bold", va="center")
    hd.text(0.013, 0.2, f"window {a0}..{a1} and a 90-day embargo withheld from "
            "the fit · a fresh 15-day forecast launched from every day inside it",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")

    ax = fig.add_axes([0.055, 0.53, 0.915, 0.36])
    for mode, c, lab in (("pooled", "#c62828", "current (6 basin means)"),
                         ("catch", "#1f7a4d", "catchment-decomposed")):
        tr = res[mode]["traj"]
        for n_, tt in enumerate(tr):
            if n_ % 6:
                continue
            ti = datetime.strptime(tt["init"], "%Y-%m-%d")
            xs = [ti + __import__("datetime").timedelta(days=j + 1)
                  for j in range(len(tt["path"]))]
            ax.plot(xs, tt["path"], color=c, lw=0.7, alpha=0.45, zorder=2)
        ax.plot([], [], color=c, lw=1.6, label=lab)
    ax.plot(t, [y[i] for i in k], color="#111", lw=2.4, zorder=6, label="observed")
    ax.axhline(100, color="0.55", lw=0.8, ls=":")
    ax.set_ylabel("national inflow, % of norm", fontsize=9.5)
    ax.set_title("Every thin line is a 15-day forecast issued that morning",
                 fontsize=11, fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=8.5, ncol=3); ax.grid(lw=0.25, alpha=0.5)
    ax.tick_params(labelsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    for i, (key, ttl, fmt) in enumerate((
            ("skill", "RMSE skill vs persistence", "{:+.2f}"),
            ("rise_r", "r on the top-decile RISES", "{:.2f}"),
            ("rise_amp", "amplitude captured on rises", "{:.0%}"))):
        axx = fig.add_axes([0.055 + i * 0.32, 0.10, 0.26, 0.33])
        for mode, c, lab in (("pooled", "#c62828", "current"),
                             ("catch", "#1f7a4d", "decomposed")):
            L = sorted(int(h) for h in res[mode]["leads"])
            v = [res[mode]["leads"][h].get(key) for h in L]
            v = [np.nan if x is None else x for x in v]
            axx.plot(L, v, color=c, lw=2.0, marker="o", ms=4, label=lab)
        axx.axhline(0, color="0.45", lw=0.9)
        axx.set_xlabel("lead, days", fontsize=8.8)
        axx.set_title(ttl, fontsize=10, fontweight="bold", loc="left", color=INK)
        axx.grid(lw=0.25, alpha=0.5); axx.tick_params(labelsize=8)
        if i == 0:
            axx.legend(fontsize=8.5)
    fig.text(0.055, 0.02, f"decomposed model uses {res['catch']['n_features']} "
             f"features against {res['pooled']['n_features']}; identical "
             "polygons are collapsed and their energy summed, so this is 24 "
             "real catchments not 37 nominal ones. Quantile mapping is not "
             "applied — it corrects NWP rain bias and this replay uses "
             "observed rain.", fontsize=7.8, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=118); plt.close(fig)
    print(f"wrote {OUT_PNG}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="2015-08:2016-03")
    a = ap.parse_args()
    a0, a1 = a.window.split(":")
    D = load()
    print(f"{len(D['grp'])} distinct catchments; window {a0}..{a1}\n")
    res = {}
    print(f"{'model':16}{'lead':>5}{'skill':>8}{'rise r':>9}{'rise amp':>10}")
    for mode, lab in (("pooled", "6 basin means"), ("catch", "catchments")):
        res[mode] = replay(D, a0, a1, mode)
        for h in (1, 3, 7, 15):
            v = res[mode]["leads"].get(h)
            if v:
                print(f"{lab:16}{h:5d}{v['skill']:+8.3f}"
                      f"{(v['rise_r'] if v['rise_r'] is not None else float('nan')):9.3f}"
                      f"{(v['rise_amp'] if v['rise_amp'] is not None else float('nan')):10.2f}")
        print()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
         "window": a.window,
         "models": {k: {kk: vv for kk, vv in v.items() if kk != "traj"}
                    for k, v in res.items()}}, indent=1))
    figure(D, res, a0, a1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
