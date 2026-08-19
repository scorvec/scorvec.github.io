#!/usr/bin/env python3
"""Full-record delta backtest (2000->) plus a daily forecast replay of an event.

The delta model was originally validated on 757 days of gauge-blended
record.  With the IMERG archive backfilled to 2000 there are ~9,300 days,
so the same nested blocked CV can run over 26 years — twelve contiguous
folds, 90-day embargo, tau/lag and the amplitude calibration re-selected
inside every training fold.  That is the answer to "does this survive on
more data".

It also replays an event day by day: fit the model with the whole window
plus a 90-day embargo removed, then launch a fresh 15-day forecast from
EVERY day in the window and score it by lead.  Nothing in the window is
ever in the model's training set, so the replay is what the model would
have produced live.

Rain before 2024-07 is corrected-satellite: the daily IDEAM gauge blend
only starts then.  Reported, not hidden.

    python scripts/sst/delta_backtest_long.py [--event 2015-08:2016-01]

Outputs:
  ~/colombia_hydro/out/delta_backtest_long.json
  ~/colombia_hydro/site/delta_backtest_long.webp
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perfect_rain_backtest as PR                                  # noqa: E402
import inflow_delta_model as M                                      # noqa: E402

PRIV = Path.home() / "colombia_hydro"
OUT_JSON = PRIV / "out" / "delta_backtest_long.json"
OUT_PNG = PRIV / "site" / "delta_backtest_long.webp"
W = {"ANTIOQUIA": 0.490, "CENTRO": 0.209, "ORIENTE": 0.166,
     "VALLE": 0.071, "CALDAS": 0.042, "CARIBE": 0.022}
ORDER = list(W)
MAX_LEAD = 15
EMBARGO = 90


def add_national(d):
    """Energy-weighted national series, treated as a seventh basin."""
    d["rain"]["NATIONAL"] = sum(np.nan_to_num(d["rain"][b]) * W[b] for b in ORDER)
    d["rain_abs"]["NATIONAL"] = sum(np.nan_to_num(d["rain_abs"][b]) * W[b] for b in ORDER)
    d["clim"]["NATIONAL"] = sum(np.nan_to_num(d["clim"][b]) * W[b] for b in ORDER)
    y = np.zeros(len(d["dates"])); wsum = np.zeros(len(d["dates"]))
    for b in ORDER:
        v = np.asarray(d["y"][b], float)
        ok = np.isfinite(v)
        y[ok] += v[ok] * W[b]; wsum[ok] += W[b]
    d["y"]["NATIONAL"] = np.where(wsum > 0.5, y / np.maximum(wsum, 1e-9), np.nan)
    d["stor"]["NATIONAL"] = sum(np.nan_to_num(d["stor"][b]) * W[b] for b in ORDER)
    return d


def replay(d, basin, a0, a1):
    """Fit outside the window (+embargo); forecast from EVERY day inside it."""
    dates = d["dates"]
    y, rain = d["y"][basin], d["rain"][basin]
    roni, stor = d["roni"], d["stor"][basin]
    n = len(y)
    ym = np.array([str(x)[:7] for x in dates])
    inper = (ym >= a0) & (ym <= a1)
    hi = np.where(inper)[0]
    if len(hi) < 30:
        return None
    tr = ~inper
    tr[max(0, hi[0] - EMBARGO):min(n, hi[-1] + EMBARGO + 1)] = False
    tau, lag = M.select_hyper(rain, y, roni, stor, tr, False)
    X, dy = M.design(rain, y, roni, stor, tau, lag)
    m = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
    beta = M.fit(X, dy, m)
    sh, off = M.shrinkage(beta, rain, y, roni, stor, tau, lag, tr)
    kf_h, ks_h = M.ema(rain, tau), M.ema(rain, M.TAU_SLOW)

    P = {h: [] for h in range(1, MAX_LEAD + 1)}
    O = {h: [] for h in range(1, MAX_LEAD + 1)}
    Q = {h: [] for h in range(1, MAX_LEAD + 1)}
    traj = []
    for i0 in hi:
        if not np.isfinite(y[i0]) or i0 + MAX_LEAD >= n:
            continue
        sim = PR.fwd(beta, off, sh, kf_h, ks_h, rain, roni, stor,
                     tau, lag, i0, y[i0], None)
        path = [y[i0] + off[j] + sh[j] * (v - y[i0]) if np.isfinite(v) else np.nan
                for j, v in enumerate(sim)]
        traj.append({"init": str(dates[i0]), "y0": round(float(y[i0]), 1),
                     "path": [round(float(v), 1) if np.isfinite(v) else None
                              for v in path]})
        for j, v in enumerate(path):
            i = i0 + 1 + j
            if i >= n or not np.isfinite(v) or not np.isfinite(y[i]):
                continue
            P[j + 1].append(v); O[j + 1].append(y[i]); Q[j + 1].append(y[i0])
    out = {"tau_days": int(tau), "lag_days": int(lag), "leads": {}, "traj": traj}
    for h in range(1, MAX_LEAD + 1):
        if len(P[h]) < 10:
            continue
        p, o, q = map(np.asarray, (P[h], O[h], Q[h]))
        rm = float(np.sqrt(np.mean((p - o) ** 2)))
        rp = float(np.sqrt(np.mean((q - o) ** 2)))
        rc = float(np.sqrt(np.mean((o - o.mean()) ** 2)))
        out["leads"][h] = {
            "n": int(len(p)), "r": round(float(np.corrcoef(p, o)[0, 1]), 3),
            "rmse": round(rm, 2), "rmse_persistence": round(rp, 2),
            "mae": round(float(np.mean(np.abs(p - o))), 2),
            "bias": round(float(np.mean(p - o)), 2),
            "skill_vs_persistence": round(1 - rm / rp, 3),
            "skill_vs_climatology": round(1 - rm / rc, 3)}
    return out


def figure(out, ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    NAVY, INK = "#13273d", "#1a2733"
    fig = plt.figure(figsize=(15.4, 10.2))
    hd = fig.add_axes([0, 0.943, 1, 0.057]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.013, 0.62, "NATIONAL INFLOW — DAILY FORECAST REPLAY, "
            f"{ev.upper()}", transform=hd.transAxes, color="white",
            fontsize=15, fontweight="bold", va="center")
    hd.text(0.013, 0.2, "a fresh 15-day forecast launched every day, with the "
            "whole window and a 90-day embargo removed from training",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")
    hd.text(0.987, 0.5, f"full record {out['window']}", transform=hd.transAxes,
            color="#b9c6d4", fontsize=9, va="center", ha="right")

    R = out["event"]["NATIONAL"]
    tr = R["traj"]
    ax = fig.add_axes([0.055, 0.545, 0.92, 0.34])
    d0 = [datetime.strptime(t["init"], "%Y-%m-%d") for t in tr]
    obs_x, obs_y = [], []
    for t in tr:
        obs_x.append(datetime.strptime(t["init"], "%Y-%m-%d"))
        obs_y.append(t["y0"])
    for k, t in enumerate(tr):
        if k % 5:
            continue
        ti = datetime.strptime(t["init"], "%Y-%m-%d")
        xs = [ti + np.timedelta64(j + 1, "D").astype("timedelta64[D]").astype(object)
              for j in range(len(t["path"]))]
        ys = t["path"]
        ok = [(x, v) for x, v in zip(xs, ys) if v is not None]
        if ok:
            ax.plot([a for a, _ in ok], [b for _, b in ok], color="#c62828",
                    lw=0.9, alpha=0.55, zorder=2)
    ax.plot(obs_x, obs_y, color="#111", lw=2.4, zorder=5, label="observed")
    ax.plot([], [], color="#c62828", lw=1.2, label="15-day forecasts (every 5th shown)")
    ax.axhline(100, color="0.55", lw=0.9, ls=":")
    ax.set_ylabel("national inflow, % of norm", fontsize=9.5)
    ax.set_title("Every red trace is a forecast the model would have issued that "
                 "morning, having never seen this window",
                 fontsize=11, fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=8.5); ax.grid(lw=0.25, alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax2 = fig.add_axes([0.055, 0.075, 0.42, 0.36])
    L = sorted(int(k) for k in R["leads"])
    ax2.plot(L, [R["leads"][str(h)]["rmse"] if str(h) in R["leads"]
                 else R["leads"][h]["rmse"] for h in L],
             color="#1f4e8c", lw=2.0, marker="o", ms=4, label="model")
    ax2.plot(L, [R["leads"][str(h)]["rmse_persistence"] if str(h) in R["leads"]
                 else R["leads"][h]["rmse_persistence"] for h in L],
             color="#c62828", lw=1.6, ls="--", marker="s", ms=3.5,
             label="persistence")
    ax2.set_xlabel("lead, days", fontsize=9)
    ax2.set_ylabel("RMSE, % of norm", fontsize=9)
    ax2.set_title(f"Error by lead in {ev}", fontsize=11, fontweight="bold",
                  loc="left", color=INK)
    ax2.legend(fontsize=8.5); ax2.grid(lw=0.25, alpha=0.5); ax2.tick_params(labelsize=8)

    ax3 = fig.add_axes([0.555, 0.075, 0.42, 0.36])
    for b, c, lw in (("NATIONAL", "#111", 2.4), ("ANTIOQUIA", "#1f4e8c", 1.3),
                     ("CENTRO", "#e08214", 1.3), ("ORIENTE", "#2e7d32", 1.3)):
        fr = out["full_record"].get(b)
        if not fr:
            continue
        hs = sorted(int(k) for k in fr["leads"])
        ax3.plot(hs, [fr["leads"][str(h)]["rmse_skill_vs_persistence"]
                      if str(h) in fr["leads"]
                      else fr["leads"][h]["rmse_skill_vs_persistence"] for h in hs],
                 color=c, lw=lw, marker="o", ms=3, label=b)
    ax3.axhline(0, color="0.45", lw=0.9)
    ax3.set_xlabel("lead, days", fontsize=9)
    ax3.set_ylabel("RMSE skill vs persistence", fontsize=9)
    ax3.set_title(f"Full record {out['window']} — out-of-sample",
                  fontsize=11, fontweight="bold", loc="left", color=INK)
    ax3.legend(fontsize=8); ax3.grid(lw=0.25, alpha=0.5); ax3.tick_params(labelsize=8)
    fig.text(0.055, 0.018, "rain before 2024-07 is corrected-satellite (the daily "
             "IDEAM gauge blend starts then) · national series modelled DIRECTLY, "
             "not aggregated from six basin models — aggregating first averages "
             "away much of XM's daily reporting noise",
             fontsize=7.8, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=118); plt.close(fig)
    print(f"wrote {OUT_PNG}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="2015-08:2016-01")
    ap.add_argument("--events", default="2015-08:2016-01,2009-08:2010-03,"
                                        "2010-09:2011-04,2023-08:2024-03",
                    help="comma-separated windows, all replayed")
    ap.add_argument("--baseline", type=int, default=365)
    ap.add_argument("--folds", type=int, default=12)
    a = ap.parse_args()
    M.BASELINE_WIN = a.baseline
    M.N_OUTER = a.folds
    M.EMBARGO = EMBARGO
    d = add_national(PR.load_all())
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "window": f"{d['dates'][0]}..{d['dates'][-1]}",
           "days": int(len(d["dates"])),
           "design": {"outer_folds": a.folds, "embargo_days": EMBARGO,
                      "baseline_window_days": a.baseline,
                      "note": "tau/lag and amplitude calibration re-selected "
                              "inside every training fold"},
           "full_record": {}, "event": {}}
    print(f"full record {out['window']} ({out['days']} d), {a.folds} folds, "
          f"embargo {EMBARGO} d, baseline {a.baseline} d\n", flush=True)
    print(f"{'series':11}{'dr@1':>8}{'blind':>8}" +
          "".join(f"{'h'+str(h):>8}" for h in (1, 3, 7, 15)))
    for b in ["NATIONAL"] + ORDER:
        r = M.backtest(d, b)
        r.pop("residuals_lead1", None)
        for h in list(r["leads"]):
            r["leads"][h].pop("resid", None); r["leads"][h].pop("fold", None)
        out["full_record"][b] = r
        g = lambda h: r["leads"].get(h, {}).get("rmse_skill_vs_persistence")
        print(f"{b:11}{r['delta_lead1']['model']['r']:8.3f}"
              f"{r['delta_lead1']['rain_blind']['r']:8.3f}" +
              "".join(f"{g(h):+8.3f}" if g(h) is not None else f"{'--':>8}"
                      for h in (1, 3, 7, 15)), flush=True)

    out["events"] = {}
    for ev in a.events.split(","):
        a0, a1 = ev.split(":")
        print(f"\nEVENT REPLAY {ev} — window + {EMBARGO} d embargo held out")
        print(f"{'series':11}{'tau':>5}{'lag':>5}" +
              "".join(f"{'h'+str(h):>9}" for h in (1, 3, 5, 7, 10, 15)))
        out["events"][ev] = {}
        for b in ["NATIONAL"] + ORDER:
            R = replay(d, b, a0, a1)
            if not R:
                continue
            if ev != a.event:
                R.pop("traj", None)          # keep only the headline trajectory
            out["events"][ev][b] = R
            if ev == a.event:
                out["event"][b] = R
            g = lambda h: R["leads"].get(h, {}).get("skill_vs_persistence")
            print(f"{b:11}{R['tau_days']:5d}{R['lag_days']:5d}" +
                  "".join(f"{g(h):+9.3f}" if g(h) is not None else f"{'--':>9}"
                          for h in (1, 3, 5, 7, 10, 15)), flush=True)
    out["event_window"] = a.event
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    try:
        figure(out, a.event)
    except Exception as e:                        # noqa: BLE001
        print(f"figure failed: {repr(e)[:150]}")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
