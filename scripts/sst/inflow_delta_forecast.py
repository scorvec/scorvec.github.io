#!/usr/bin/env python3
"""Daily inflow forecast DISTRIBUTIONS per basin, from the delta model.

The operational fan (colombia_forecast.py) propagates the v3 LEVEL kernel
and reports quantiles of the rain ensemble.  This runs the validated
delta model instead, and carries two independent sources of uncertainty
through to every forecast day:

  1. rain uncertainty — each of the ~101 AIFS+IFS members is bias
     corrected and pushed through its own recursion, so members that
     disagree about the rain produce different inflow trajectories;
  2. model uncertainty — the residual the delta model itself leaves,
     taken per lead from the blocked-CV residual pool, which never saw
     the day being forecast.

Ignoring (2) is the usual way ensemble fans end up over-confident: at
lead 7 the rain spread explains well under half the total error.  The
published distribution is the convolution of the two.

Every parameter (tau, lag, coefficients, amplitude calibration) comes
from the same nested-CV machinery as the backtest — nothing is fitted on
the days being forecast.

    python scripts/sst/inflow_delta_forecast.py

Outputs:
  colombia_hydro/data/inflow_delta_forecast.json   (per basin, per day)
  colombia_hydro/inflow_delta_forecast.webp
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inflow_delta_model as M                                   # noqa: E402
from colombia_forecast import (ARCH, ORDER, BANDS, MAX_LEAD, VERIF_JSON,  # noqa: E402
                               _bas)

REPO = HERE.parent.parent
PRIV = Path.home() / "colombia_hydro"
OUT_JSON = REPO / "colombia_hydro" / "data" / "inflow_delta_forecast.json"
OUT_PNG = REPO / "colombia_hydro" / "inflow_delta_forecast.webp"
QS = [5, 10, 25, 50, 75, 90, 95]


def latest_cycles():
    """Newest archived cycle per model, restricted to CO_MODELS.

    The archive keeps every model ever pulled, so without this filter an
    AIFS-only run still blends whatever IFS cycle happens to be on disk -
    and silently, since the only trace is the `init` stamp in the output.
    Reads the same CO_MODELS variable colombia_forecast.py uses so the two
    stages cannot disagree about which ensembles are in play.
    """
    import os
    models = tuple(m.strip() for m in
                   os.environ.get("CO_MODELS", "aifs,ifs").split(",") if m.strip())
    best = {}
    for f in sorted(glob.glob(str(ARCH / "*.json.gz"))):
        with gzip.open(f, "rt") as fh:
            rec = json.load(fh)
        k = rec["model"]
        if k not in models:
            continue
        stamp = f"{rec['init_date']}{rec['init_hh']}"
        if k not in best or stamp > best[k][0]:
            best[k] = (stamp, rec)
    return {k: v[1] for k, v in best.items()}


def band_of(lead):
    return next((i for i, (a, b) in enumerate(BANDS) if a <= lead <= b), None)


def member_rain(basin, cycles, factors, dates, rain_hist):
    """[member][lead] bias-corrected rain ANOMALY spliced onto history.

    Returns a list of full-length anomaly series, one per member, each
    identical to the observed record up to the init and carrying that
    member's corrected forecast thereafter.
    """
    clim = CLIM[basin]
    dmap = {str(d): i for i, d in enumerate(dates)}
    out, meta = [], None
    for mdl, rec in cycles.items():
        d0 = np.datetime64(f"{rec['init_date'][:4]}-{rec['init_date'][4:6]}-"
                           f"{rec['init_date'][6:8]}")
        mem = _bas(rec, basin)
        for m in mem:
            x = rain_hist.copy()
            for li, vd in enumerate(rec["valid"]):
                lead = int((np.datetime64(vd) - d0).astype(int)) + 1
                b = band_of(lead)
                if b is None:
                    continue
                i = dmap.get(vd)
                if i is None:
                    continue
                F = factors.get(basin, {}).get(str(b), {}).get(mdl, 1.0)
                x[i] = float(m[li]) * F - clim[i]
            out.append(x)
        meta = meta or {}
        meta[mdl] = f"{rec['init_date']} {rec['init_hh']}Z"
    return out, meta


def main() -> int:
    d = M.load()
    dates, n = d["dates"], len(d["dates"])
    global CLIM
    tc = json.loads((PRIV / "raw" / "imerg_basin_daily.json").read_text())
    keep = {x: i for i, x in enumerate(tc["dates"])}
    CLIM = {}
    for r in ORDER:
        c = np.asarray(tc[r + "_clim"], float)
        CLIM[r] = np.array([c[keep[str(x).replace("-", "")]] for x in dates])

    cycles = latest_cycles()
    if not cycles:
        print("no archived cycles"); return 1
    factors = json.loads(VERIF_JSON.read_text())["bias_factors"]

    # extend the calendar past the last observation to cover the leads
    horizon = max(
        int((np.datetime64(v) - np.datetime64(str(dates[-1]))).astype(int))
        for rec in cycles.values() for v in rec["valid"])
    ext = np.array([dates[-1] + np.timedelta64(k, "D")
                    for k in range(1, max(horizon, MAX_LEAD) + 1)])
    all_dates = np.concatenate([dates, ext])

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "model": "delta (linear reservoir), nested-CV calibrated",
           "note": "distribution = rain-ensemble spread convolved with the "
                   "per-lead model residual from blocked CV",
           "quantiles": QS, "basins": {}}
    print(f"{'basin':10}{'tau':>5}{'lag':>5}{'members':>9}  median path (% of norm)")
    for b in ORDER:
        rain, y = d["rain"][b], d["y"][b]
        roni, stor = d["roni"], d["stor"][b]
        # final model: hyperparameters + calibration from CV, all data
        tr = np.ones(n, bool)
        tau, lag = M.select_hyper(rain, y, roni, stor, tr, False)
        X, dy = M.design(rain, y, roni, stor, tau, lag)
        m = np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        beta = M.fit(X, dy, m)
        sh, off = M.shrinkage(beta, rain, y, roni, stor, tau, lag, tr)
        cv = M.backtest(d, b)                      # residual pools per lead
        resid = {h: np.asarray(cv["leads"].get(h, {}).get("resid", []), float)
                 for h in range(1, MAX_LEAD + 1)}

        # pad the driver series over the forecast horizon
        pad = len(all_dates) - n
        rh = np.concatenate([rain, np.zeros(pad)])
        rn = np.concatenate([roni, np.full(pad, roni[-1])])
        st = np.concatenate([stor, np.full(pad, stor[-1])])
        CLIM[b] = np.concatenate([CLIM[b], CLIM[b][-365:][:pad]
                                  if pad <= 365 else np.zeros(pad)])
        mems, meta = member_rain(b, cycles, factors, all_dates, rh)
        i0 = n - 1
        y0 = float(y[i0]) if np.isfinite(y[i0]) else float(np.nanmean(y[-5:]))

        paths = []
        for x in mems:
            sim = M.simulate(beta, x, rn, st, tau, lag, i0, y0, MAX_LEAD)
            paths.append([y0 + off[j] + sh[j] * (v - y0) if np.isfinite(v)
                          else np.nan for j, v in enumerate(sim)])
        paths = np.asarray(paths, float)

        rows = []
        for j in range(MAX_LEAD):
            col = paths[:, j]
            col = col[np.isfinite(col)]
            R = resid.get(j + 1, np.zeros(0))
            R = R[np.isfinite(R)]
            if not len(col):
                continue
            full = (col[:, None] + R[None, :]).ravel() if len(R) > 30 else col
            full = np.clip(full, 0.0, None)     # % of norm cannot go negative
            rows.append({
                "lead": j + 1, "date": str(all_dates[i0 + 1 + j]),
                "p": {f"p{q}": round(float(np.percentile(full, q)), 1) for q in QS},
                "rain_only_p5_p95": [round(float(max(np.percentile(col, 5), 0)), 1),
                                     round(float(np.percentile(col, 95)), 1)],
                "sd_rain": round(float(np.std(col)), 1),
                "sd_model": round(float(np.std(R)), 1) if len(R) > 30 else None})
        pct = float(np.nanmean(y <= y0) * 100)
        out["basins"][b] = {"tau_days": int(tau), "lag_days": int(lag),
                            "members": int(len(paths)), "init": meta,
                            "last_obs": round(y0, 1),
                            "last_obs_percentile": round(pct, 1),
                            "anchor_is_extreme": bool(pct > 97 or pct < 3),
                            "days": rows}
        med = " ".join(f"{r['p']['p50']:.0f}" for r in rows[:8])
        print(f"{b:10}{tau:5d}{lag:5d}{len(paths):9d}  {med}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    try:
        figure(out)
    except Exception as e:                            # noqa: BLE001
        print(f"figure failed: {repr(e)[:140]}")
    return 0


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    NAVY, INK = "#13273d", "#1a2733"
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.8))
    hd = fig.add_axes([0, 0.945, 1, 0.055]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.014, 0.62, "COLOMBIA — DAILY INFLOW DISTRIBUTIONS BY BASIN",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    init = " · ".join(f"{k.upper()} {v}" for k, v in
                      (list(out["basins"].values())[0]["init"] or {}).items())
    hd.text(0.014, 0.2, f"delta model · rain ensemble + model residual · {init}",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")
    for ax, b in zip(axes.ravel(), ORDER):
        v = out["basins"][b]
        rows = v["days"]
        t = [datetime.strptime(r["date"], "%Y-%m-%d") for r in rows]
        g = lambda k: [r["p"][k] for r in rows]
        ax.fill_between(t, g("p5"), g("p95"), color="#1f4e8c", alpha=0.14, lw=0,
                        label="5–95% (rain + model)")
        ax.fill_between(t, g("p25"), g("p75"), color="#1f4e8c", alpha=0.30, lw=0,
                        label="25–75%")
        ax.fill_between(t, [r["rain_only_p5_p95"][0] for r in rows],
                        [r["rain_only_p5_p95"][1] for r in rows],
                        facecolor="none", edgecolor="#c62828", lw=1.0, ls="--",
                        label="5–95% rain spread only")
        ax.plot(t, g("p50"), color="#1f4e8c", lw=2.0, label="median")
        ax.axhline(100, color="0.55", lw=0.8, ls=":")
        ax.axhline(v["last_obs"], color="#2e7d32", lw=1.0, ls="-.",
                   label=f"last obs {v['last_obs']:.0f}")
        ax.set_title(f"{b} — τ={v['tau_days']} d, lag {v['lag_days']} d",
                     fontsize=10.5, fontweight="bold", loc="left", color=INK)
        ax.set_ylabel("inflow, % of norm", fontsize=8.5)
        ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        for lb in ax.get_xticklabels():
            lb.set_rotation(30); lb.set_ha("right")
        if v.get("anchor_is_extreme"):
            ax.text(0.98, 0.94, f"anchor at {v['last_obs_percentile']:.0f}th pct",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    color="#b35806", fontweight="bold")
        if b == ORDER[0]:
            ax.legend(fontsize=7, loc="upper left")
    fig.text(0.014, 0.015, "the dashed red envelope is what a rain-ensemble-only "
             "fan would show — the gap to the filled band is the model error "
             "such a fan omits", fontsize=8, color="#5a6b7a")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=118); plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")


if __name__ == "__main__":
    raise SystemExit(main())
