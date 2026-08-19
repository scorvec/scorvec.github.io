#!/usr/bin/env python3
"""Re-issue the daily national fan as of a past date, and see what it showed.

Answers "what would we have been looking at the morning before the
event". Everything is restricted to what existed on the init date:

  * inflow, storage and ENSO are truncated at the init;
  * rain is observed up to the init and comes from the ARCHIVED AIFS/IFS
    cycle of that morning thereafter, bias-corrected with the operational
    factors;
  * the model itself is fitted with the whole event month PLUS a 90-day
    embargo removed, so the fitted coefficients never saw the spike.

Observed inflow is drawn over the top, so the forecast can be read
against what actually happened.

    python scripts/sst/hindcast_fan.py --inits 2026-08-10,2026-08-14

Output: ~/colombia_hydro/site/hindcast_fan.webp
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perfect_rain_backtest as PR                                  # noqa: E402
import delta_backtest_long as DB                                    # noqa: E402
import inflow_delta_model as M                                      # noqa: E402
import national_inflow as NI                                        # noqa: E402
from colombia_forecast import _bas, VERIF_JSON, BANDS, ORDER        # noqa: E402

PRIV = Path.home() / "colombia_hydro"
ARCH = PRIV / "raw" / "fcst_rain"
OUT_PNG = PRIV / "site" / "hindcast_fan.webp"
MAXLEAD = 15


def fit_outside(d, hold_month):
    """Fit the national delta model with hold_month (+90 d) withheld."""
    dates = np.array([str(x) for x in d["dates"]])
    ym = np.array([s[:7] for s in dates])
    hold = ym == hold_month
    hi = np.where(hold)[0]
    n = len(dates)
    tr = ~hold
    tr[max(0, hi[0] - 90):min(n, hi[-1] + 91)] = False
    rain, y = d["rain"]["NATIONAL"], d["y"]["NATIONAL"]
    roni, stor = d["roni"], d["stor"]["NATIONAL"]
    tau, lag = M.select_hyper(rain, y, roni, stor, tr, False)
    X, dy = M.design(rain, y, roni, stor, tau, lag)
    m = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
    beta = M.fit(X, dy, m)
    sh, off = M.shrinkage(beta, rain, y, roni, stor, tau, lag, tr)
    return beta, sh, off, tau, lag, int(m.sum())


def resid_by_lead(d, hold_month):
    """Out-of-sample residuals by lead and the predicted level with each,
    from blocked CV that also excludes the event month."""
    dates = np.array([str(x) for x in d["dates"]])
    ym = np.array([s[:7] for s in dates])
    hold = ym == hold_month
    rain, y = d["rain"]["NATIONAL"], d["y"]["NATIONAL"]
    roni, stor = d["roni"], d["stor"]["NATIONAL"]
    n = len(y)
    edges = np.linspace(0, n, 13).astype(int)
    res = {h: [] for h in range(1, MAXLEAD + 1)}
    lev = {h: [] for h in range(1, MAXLEAD + 1)}
    for k in range(12):
        a, b = edges[k], edges[k + 1]
        te = np.zeros(n, bool); te[a:b] = True
        tr = ~te & ~hold
        tr[max(0, a - 90):min(n, b + 90)] = False
        if tr.sum() < 400:
            continue
        tau, lag = M.select_hyper(rain, y, roni, stor, tr, False)
        X, dy = M.design(rain, y, roni, stor, tau, lag)
        m = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        beta = M.fit(X, dy, m)
        sh, off = M.shrinkage(beta, rain, y, roni, stor, tau, lag, tr)
        kf, ks = M.ema(rain, tau), M.ema(rain, M.TAU_SLOW)
        for i0 in range(a, b):
            if not np.isfinite(y[i0]) or hold[i0]:
                continue
            sim = PR.fwd(beta, off, sh, kf, ks, rain, roni, stor,
                         tau, lag, i0, y[i0], None)
            for j, v in enumerate(sim):
                i = i0 + 1 + j
                if i >= n or not np.isfinite(v) or not np.isfinite(y[i]) or hold[i]:
                    continue
                p = y[i0] + off[j] + sh[j] * (v - y[i0])
                res[j + 1].append(y[i] - p); lev[j + 1].append(p)
    return ({h: np.asarray(v) for h, v in res.items()},
            {h: np.asarray(v) for h, v in lev.items()})


def cycle_for(init):
    """Archived AIFS/IFS cycles initialised on `init` (00Z preferred)."""
    out = {}
    for f in sorted(glob.glob(str(ARCH / "*.json.gz"))):
        rec = json.load(gzip.open(f, "rt"))
        dd = f"{rec['init_date'][:4]}-{rec['init_date'][4:6]}-{rec['init_date'][6:8]}"
        if dd != init:
            continue
        if rec["model"] not in out or rec["init_hh"] == "00":
            out[rec["model"]] = rec
    return out


def run_init(d, init, beta, sh, off, tau, lag, resid, resid_lev, factors, W):
    dates = np.array([str(x) for x in d["dates"]])
    dmap = {s: i for i, s in enumerate(dates)}
    i0 = dmap.get(init)
    cyc = cycle_for(init)
    if i0 is None or not cyc:
        return None
    rain = d["rain"]["NATIONAL"]
    clim = d["clim"]["NATIONAL"]
    y, roni, stor = d["y"]["NATIONAL"], d["roni"], d["stor"]["NATIONAL"]
    band_of = lambda L: next((i for i, (a, b) in enumerate(BANDS)
                              if a <= L <= b), None)
    paths = []
    for mdl, rec in cyc.items():
        nmem = None
        for b in ORDER:
            arr = np.array(_bas(rec, b), float)
            Fb = []
            for v in rec["valid"]:
                L = int((np.datetime64(v) - np.datetime64(init)).astype(int)) + 1
                bd = band_of(L)
                Fb.append(factors.get(b, {}).get(str(bd), {}).get(mdl, 1.0)
                          if bd is not None else 1.0)
            contrib = arr * np.array(Fb)[None, :] * W[b]
            nmem = contrib if nmem is None else nmem + contrib
        for k in range(nmem.shape[0]):
            x = rain.copy()
            for li, v in enumerate(rec["valid"]):
                j = dmap.get(v)
                if j is not None:
                    x[j] = nmem[k, li] - clim[j]
            sim = PR.fwd(beta, off, sh, M.ema(x, tau), M.ema(x, M.TAU_SLOW),
                         x, roni, stor, tau, lag, i0, y[i0], None)
            paths.append([y[i0] + off[j] + sh[j] * (v - y[i0])
                          if np.isfinite(v) else np.nan
                          for j, v in enumerate(sim)])
    P = np.asarray(paths, float)
    rows = []
    for j in range(MAXLEAD):
        i = i0 + 1 + j
        if i >= len(dates):
            break
        col = P[:, j]; col = col[np.isfinite(col)]
        if not len(col):
            continue
        R = resid.get(j + 1, np.zeros(0))
        Rp = resid_lev.get(j + 1, np.zeros(0))
        if len(R) > 200 and len(Rp) == len(R):
            L = float(np.median(col))
            band = ((Rp < 70) if L < 70 else (Rp >= 110) if L >= 110
                    else ((Rp >= 70) & (Rp < 110)))
            if band.sum() > 100:
                R = R[band]
        full = np.clip((col[:, None] + R[None, :]).ravel(), 0, None) \
            if len(R) > 30 else col
        rows.append({"date": dates[i], "lead": j + 1,
                     "p10": float(np.percentile(full, 10)),
                     "p50": float(np.percentile(full, 50)),
                     "p90": float(np.percentile(full, 90))})
    # forecast rain per MODEL, keyed by valid date.  Pooling AIFS and IFS
    # members into one median is misleading here: on 17 Aug AIFS carried
    # 18.7 mm/day against 18.7 observed while IFS carried 8-11, so the
    # pooled median understates what the better model actually said.
    rainf = {}
    for mdl, rec in cyc.items():
        nm = None
        for b in ORDER:
            arr = np.array(_bas(rec, b), float)
            Fb = []
            for v in rec["valid"]:
                L = int((np.datetime64(v) - np.datetime64(init)).astype(int)) + 1
                bd = band_of(L)
                Fb.append(factors.get(b, {}).get(str(bd), {}).get(mdl, 1.0)
                          if bd is not None else 1.0)
            c = arr * np.array(Fb)[None, :] * W[b]
            nm = c if nm is None else nm + c
        rainf[mdl] = {v: float(np.mean(nm[:, li]))
                      for li, v in enumerate(rec["valid"])}
    return {"init": init, "y0": float(y[i0]), "rows": rows, "rainf": rainf,
            "models": {m: f"{r['init_date']} {r['init_hh']}Z" for m, r in cyc.items()}}


def figure(d, runs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    NAVY, INK = "#13273d", "#1a2733"
    dates = np.array([str(x) for x in d["dates"]])
    y = d["y"]["NATIONAL"]
    rain = d["rain_abs"]["NATIONAL"] if "NATIONAL" in d["rain_abs"] else None
    lo, hi = "2026-08-01", "2026-09-02"
    k = [i for i, s in enumerate(dates) if lo <= s <= hi]
    to = [datetime.strptime(dates[i], "%Y-%m-%d") for i in k]

    fig = plt.figure(figsize=(14.6, 8.6))
    hd = fig.add_axes([0, 0.935, 1, 0.065]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.014, 0.62, "WHAT THE MODEL SHOWED BEFORE THE 17 AUG SPIKE",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    hd.text(0.014, 0.2, "each fan uses only the data and the AIFS/IFS cycle "
            "available that morning; the model is fitted with all of Aug-2026 "
            "plus a 90-day embargo withheld",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")

    ax = fig.add_axes([0.055, 0.40, 0.915, 0.47])
    cols = ["#1f4e8c", "#1f7a4d", "#7b1fa2"]
    for c, run in zip(cols, runs):
        R = run["rows"]
        t = [datetime.strptime(r["date"], "%Y-%m-%d") for r in R]
        ti = datetime.strptime(run["init"], "%Y-%m-%d")
        ax.fill_between(t, [r["p10"] for r in R], [r["p90"] for r in R],
                        color=c, alpha=0.15, lw=0)
        ax.plot([ti] + t, [run["y0"]] + [r["p50"] for r in R], color=c, lw=2.2,
                marker="o", ms=3.5,
                label=f"issued {run['init']} (10–90%)")
        ax.axvline(ti, color=c, lw=0.9, ls=":", alpha=0.8)
    ax.plot(to, [y[i] for i in k], color="#111", lw=2.6, marker="o", ms=4,
            label="observed", zorder=6)
    ax.axhline(100, color="0.55", lw=0.9, ls=":")
    sp = datetime.strptime("2026-08-17", "%Y-%m-%d")
    ax.annotate("the spike\n160.6% of norm", xy=(sp, 160.6),
                xytext=(sp - timedelta(days=7), 178), fontsize=9,
                color="#c62828", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.3))
    ax.set_ylabel("national inflow, % of norm", fontsize=9.5)
    ax.set_title("Forecast fans issued before the event, against what happened",
                 fontsize=11, fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(lw=0.25, alpha=0.5)
    ax.tick_params(labelsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax2 = fig.add_axes([0.055, 0.20, 0.915, 0.15])
    if rain is not None:
        ax2.bar(to, [rain[i] for i in k], width=0.8, color="#9db8d8",
                label="observed basin rain")
    for c, run in zip(cols, runs):
        for mdl, ls, mk in (("aifs", "--", "s"), ("ifs", ":", "^")):
            rf = (run.get("rainf") or {}).get(mdl)
            if not rf:
                continue
            it = sorted(rf)
            t = [datetime.strptime(v, "%Y-%m-%d") for v in it]
            ax2.plot(t, [rf[v] for v in it], color=c, lw=1.4, ls=ls,
                     marker=mk, ms=3,
                     label=f"{mdl.upper()} {run['init'][5:]}")
    ax2.set_ylabel("mm/day", fontsize=9)
    ax2.set_title("The rain that drove it — AIFS (dashed) and IFS (dotted) "
                  "vs observed",
                  fontsize=10, fontweight="bold", loc="left", color=INK)
    ax2.legend(fontsize=7.6, ncol=4); ax2.grid(lw=0.25, alpha=0.5)
    ax2.tick_params(labelsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax3 = fig.add_axes([0.055, 0.03, 0.915, 0.12]); ax3.set_axis_off()
    lines = []
    for run in runs:
        hit = [r for r in run["rows"] if r["date"] == "2026-08-17"]
        if hit:
            r = hit[0]
            lines.append(f"issued {run['init']} (lead {r['lead']}):  median "
                         f"{r['p50']:.0f}%,  10-90% {r['p10']:.0f}-{r['p90']:.0f}%"
                         f"   vs observed 160.6%   (persistence would have said "
                         f"{run['y0']:.0f}%)")
    ax3.text(0, 0.95, "For 17 Aug specifically", fontsize=10.4,
             fontweight="bold", color="#1f7a4d", va="top")
    for i, ln in enumerate(lines):
        ax3.text(0.005, 0.68 - i * 0.24, "• " + ln, fontsize=9, color=INK, va="top")
    fig.text(0.055, 0.008, "AIFS had 18.7 mm/day for 17 Aug from the 10 Aug "
             "cycle, exactly what fell; IFS carried 8-11. So the rain signal "
             "was there eight days out — the shortfall is the inflow "
             "amplitude, which is the cross-validated shrinkage behaving as "
             "fitted, plus the blend being pulled down by the drier model.",
             fontsize=7.8, color="#5a6b7a")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=118); plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", default="2026-08-10,2026-08-14")
    ap.add_argument("--hold", default="2026-08")
    a = ap.parse_args()
    d = DB.add_national(PR.load_all())
    W = NI.basin_energy_weights()
    factors = json.loads(VERIF_JSON.read_text())["bias_factors"]
    beta, sh, off, tau, lag, nfit = fit_outside(d, a.hold)
    print(f"model fitted on {nfit} days outside {a.hold} (+90 d embargo); "
          f"tau={tau} lag={lag}", flush=True)
    resid, resid_lev = resid_by_lead(d, a.hold)
    runs = []
    for init in a.inits.split(","):
        r = run_init(d, init.strip(), beta, sh, off, tau, lag,
                     resid, resid_lev, factors, W)
        if r:
            runs.append(r)
            hit = [x for x in r["rows"] if x["date"] == "2026-08-17"]
            msg = (f"median {hit[0]['p50']:.0f}% "
                   f"[{hit[0]['p10']:.0f}-{hit[0]['p90']:.0f}]" if hit else "—")
            print(f"  issued {r['init']}: anchor {r['y0']:.0f}%  "
                  f"-> 17 Aug {msg}", flush=True)
    if not runs:
        print("no archived cycles for those inits"); return 1
    figure(d, runs, OUT_PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
