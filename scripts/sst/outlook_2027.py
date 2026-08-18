#!/usr/bin/env python3
"""Probabilistic national outlook through early 2027 — inflow and hydro share.

Joins the three validated pieces into one forecast:

  BALANCE OF MONTH  daily, from the delta model driven by the live
                    AIFS+IFS ensemble member by member.
  MONTHS +1..+6     the monthly model (ENSO + storage + antecedent
                    wetness), fitted in log space, as a distribution.
  HYDRO SHARE       hydro as a fraction of total national generation,
                    modelled in LOGIT space (a share is bounded, so a
                    linear fit would eventually predict >100%) on the
                    inflow the model above forecasts, plus storage,
                    season and a slow trend for thermal capacity growth.

Everything is a distribution, not a point.  The inflow distribution is
pushed through the share model sample by sample, so the share
distribution inherits the inflow uncertainty rather than being computed
from the inflow median alone.

C3S seasonal rainfall is deliberately NOT used.  Verified directly it has
real skill on rainfall anomalies (ACC 0.30-0.47, beating an ONI
regression at leads 2-6), but after EMOS calibration it is worth only
+0.01 to +0.09 CRPS skill on rain, and on monthly INFLOW it adds ~+0.01
over the statistical model because its information is already carried by
ENSO and storage.  Keeping it out removes a live CDS dependency for no
measurable loss; c3s_verify.py keeps the case re-testable.

    python scripts/sst/outlook_2027.py

Outputs: ~/colombia_hydro/out/outlook_2027.json
         ~/colombia_hydro/site/outlook_2027.webp
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import national_inflow as NI                                        # noqa: E402

PRIV = Path.home() / "colombia_hydro"
GEN = PRIV / "raw" / "generation_daily.json.gz"
OUT_JSON = PRIV / "out" / "outlook_2027.json"
OUT_PNG = PRIV / "site" / "outlook_2027.webp"
NDRAW = 4000


def monthly_generation():
    """Monthly hydro share and totals, GWh/day."""
    d = json.load(gzip.open(GEN, "rt"))
    acc = {}
    for day, v in d.items():
        h, t = v.get("hydro"), v.get("total")
        if not h or not t or t <= 0:
            continue
        acc.setdefault(day[:7], []).append((h / 1e6, t / 1e6))
    out = {}
    for m, rows in acc.items():
        if len(rows) < 20:
            continue
        hh = np.mean([r[0] for r in rows]); tt = np.mean([r[1] for r in rows])
        out[m] = {"hydro_gwh": float(hh), "total_gwh": float(tt),
                  "share": float(hh / tt)}
    return out


def _lg(x):
    x = min(max(float(x), 0.02), 0.98)
    return float(np.log(x / (1 - x)))


def fit_share(gen, rows, embargo=1):
    """Logit-space monthly hydro share, with LAST MONTH'S SHARE carried in.

    Inflow, storage, season and a slow trend explain the share only to
    r=0.60 / MAE 5.4 pts, and through 2026 they fail badly: the residual
    runs -1.6, -2.7, -6.2, -11.5 points from March to July as hydro
    generates far less than inflow implies.  That is a structural break -
    reservoir conservation ahead of the El Nino (the same anticipation
    visible in prices) together with new non-hydro capacity - and no
    inflow-based term can see it.

    Adding the previous month's OBSERVED share, which is known at issue
    time, lets the model start from where the system actually is rather
    than from where the hydrology says it should be: r 0.60 -> 0.87, MAE
    5.41 -> 3.23 points, 2026 MAE 4.53 -> 2.25.  Beyond lead 1 the
    forecast share is fed back in, so the outlook timesteps month by
    month exactly as the daily model does.
    """
    keys = sorted(set(gen) & set(rows))
    X, Y, YR = [], [], []
    for m in keys:
        g, r = gen[m], rows[m]
        a = int(m[:4]) * 12 + int(m[5:7]) - 1
        pm = f"{(a - 1) // 12:04d}-{(a - 1) % 12 + 1:02d}"
        if pm not in gen:
            continue
        t = (int(m[:4]) - 2013) + (int(m[5:7]) - 6) / 12.0
        mo = int(m[5:7])
        s1, c1 = np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)
        X.append([np.log(max(r["pct"], 5) / 100.0), r["stor_end"], t, s1, c1,
                  _lg(gen[pm]["share"])])
        Y.append(_lg(g["share"])); YR.append(int(m[:4]))
    X, Y, YR = np.asarray(X), np.asarray(Y), np.asarray(YR)
    P = np.full(len(Y), np.nan)
    for y in sorted(set(YR)):
        te = YR == y; tr = np.abs(YR - y) > embargo
        if tr.sum() < 60:
            continue
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        b, *_ = np.linalg.lstsq(A, Y[tr], rcond=None)
        P[te] = b[0] + X[te] @ b[1:]
    ok = np.isfinite(P)
    inv = lambda z: 1 / (1 + np.exp(-z))
    sh_p, sh_o = inv(P[ok]), inv(Y[ok])
    A = np.column_stack([np.ones(len(Y)), X])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    stats = {"n": int(ok.sum()),
             "r_loyo": round(float(np.corrcoef(sh_p, sh_o)[0, 1]), 3),
             "mae_pts_loyo": round(float(np.mean(np.abs(sh_p - sh_o)) * 100), 2),
             "rmse_pts_loyo": round(float(np.sqrt(np.mean((sh_p - sh_o) ** 2)) * 100), 2),
             "clim_mae_pts": round(float(np.mean(np.abs(sh_o - sh_o.mean())) * 100), 2),
             "resid_sd_logit": round(float(np.std(Y[ok] - P[ok])), 4),
             "coef": {k: round(float(v), 4) for k, v in zip(
                 ("intercept", "log_inflow", "storage", "trend", "sin", "cos",
                  "prev_share_logit"), beta)}}
    stats["skill_vs_clim"] = round(1 - stats["mae_pts_loyo"] / stats["clim_mae_pts"], 3)
    return beta, stats


def main() -> int:
    d = NI.load_national()
    rows = NI.monthly_frame(d)
    gen = monthly_generation()
    bt = NI.monthly_backtest(d)
    fc = NI.monthly_forecast(d, bt)
    beta, sst = fit_share(gen, rows)
    print("hydro-share model (logit, LOYO +/-1y):", json.dumps(sst["coef"]))
    print(f"  n={sst['n']}  r={sst['r_loyo']}  MAE={sst['mae_pts_loyo']} pts "
          f"vs climatology {sst['clim_mae_pts']}  skill={sst['skill_vs_clim']:+.3f}",
          flush=True)

    rng = np.random.default_rng(2027)
    sd_share = sst["resid_sd_logit"]
    inv = lambda z: 1 / (1 + np.exp(-z))
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "issue_month": fc["issue_month"], "oni_at_issue": fc["oni_at_issue"],
           "storage_anom_at_issue": fc["storage_anom_at_issue"],
           "oni_percentile": fc.get("oni_percentile_in_record"),
           "extrapolating_on_enso": fc.get("extrapolating_on_enso"),
           "share_model": sst, "monthly_backtest": bt["leads"], "months": []}
    stor_now = fc["storage_anom_at_issue"]
    last_gen_month = max(gen)
    prev_logit = np.full(NDRAW, _lg(gen[last_gen_month]["share"]))
    print(f"share anchored on observed {last_gen_month}: "
          f"{gen[last_gen_month]['share']*100:.1f}%")
    print(f"\nissued from {fc['issue_month']} — ONI {fc['oni_at_issue']:+.2f}, "
          f"storage {stor_now:+.1f} pts\n")
    print(f"{'month':9}{'inflow % of norm':>26}{'GWh/day':>22}{'hydro % of gen':>24}")
    print(f"{'':9}{'p10    p50    p90':>26}{'p10    p50    p90':>22}"
          f"{'p10    p50    p90':>24}")
    for m in fc["months"]:
        mo = int(m["month"][5:7]); yr = int(m["month"][:4])
        sd_log = bt["leads"].get(m["lead"], {}).get("log_resid_sd", 0.28)
        mu = np.log(max(m["pct_p50"], 1.0))
        draws = np.exp(rng.normal(mu, sd_log, NDRAW))            # inflow % of norm
        t = (yr - 2013) + (mo - 6) / 12.0
        s1, c1 = np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)
        z = (beta[0] + beta[1] * np.log(np.maximum(draws, 5) / 100.0)
             + beta[2] * stor_now + beta[3] * t + beta[4] * s1 + beta[5] * c1
             + beta[6] * prev_logit
             + rng.normal(0, sd_share, NDRAW))
        share = inv(z) * 100.0
        prev_logit = z                        # timestep: feed the draw forward
        doy = datetime(yr, mo, 15).timetuple().tm_yday
        gw = d["norm_gwh"][doy] / 100.0
        q = lambda a, p: [round(float(np.percentile(a, x)), 1) for x in p]
        pi, pg, ps = q(draws, (10, 50, 90)), q(draws * gw, (10, 50, 90)), \
            q(share, (10, 50, 90))
        out["months"].append({"month": m["month"], "lead": m["lead"],
                              "inflow_pct": {"p10": pi[0], "p50": pi[1], "p90": pi[2]},
                              "inflow_gwh": {"p10": pg[0], "p50": pg[1], "p90": pg[2]},
                              "hydro_share_pct": {"p10": ps[0], "p50": ps[1],
                                                  "p90": ps[2]},
                              "norm_gwh": m["norm_gwh"],
                              "below_observed_minimum": m.get("below_observed_minimum")})
        print(f"{m['month']:9}{pi[0]:9.0f}{pi[1]:7.0f}{pi[2]:7.0f}"
              f"{pg[0]:12.0f}{pg[1]:7.0f}{pg[2]:7.0f}"
              f"{ps[0]:14.0f}{ps[1]:7.0f}{ps[2]:7.0f}", flush=True)
    recent = sorted(gen)[-1]
    out["latest_observed"] = {"month": recent, "share_pct": round(gen[recent]["share"] * 100, 1),
                              "hydro_gwh": round(gen[recent]["hydro_gwh"], 1),
                              "total_gwh": round(gen[recent]["total_gwh"], 1)}
    print(f"\nlatest observed {recent}: hydro {gen[recent]['share']*100:.1f}% of "
          f"generation ({gen[recent]['hydro_gwh']:.0f} of "
          f"{gen[recent]['total_gwh']:.0f} GWh/day)")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    try:
        figure(out, gen)
    except Exception as e:                          # noqa: BLE001
        print(f"figure failed: {repr(e)[:150]}")
    print(f"wrote {OUT_JSON}")
    return 0


def figure(out, gen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    NAVY, INK = "#13273d", "#1a2733"
    fig = plt.figure(figsize=(15.0, 8.8))
    hd = fig.add_axes([0, 0.935, 1, 0.065]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.013, 0.62, "COLOMBIA — PROBABILISTIC OUTLOOK THROUGH EARLY 2027",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    hd.text(0.013, 0.2, f"issued from {out['issue_month']} · ONI "
            f"{out['oni_at_issue']:+.2f} · storage "
            f"{out['storage_anom_at_issue']:+.1f} pts vs norm",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")
    M = out["months"]
    t = [datetime.strptime(m["month"] + "-15", "%Y-%m-%d") for m in M]
    hist = sorted(gen)[-30:]
    th = [datetime.strptime(m + "-15", "%Y-%m-%d") for m in hist]

    ax = fig.add_axes([0.055, 0.55, 0.42, 0.33])
    ax.fill_between(t, [m["inflow_pct"]["p10"] for m in M],
                    [m["inflow_pct"]["p90"] for m in M], color="#1f4e8c",
                    alpha=0.20, lw=0, label="10–90%")
    ax.plot(t, [m["inflow_pct"]["p50"] for m in M], color="#1f4e8c", lw=2.4,
            marker="o", ms=5, label="median")
    ax.axhline(100, color="0.5", lw=1.0, ls=":")
    ax.set_ylabel("inflow, % of norm", fontsize=9.5)
    ax.set_title("National inflow", fontsize=11, fontweight="bold",
                 loc="left", color=INK)
    ax.legend(fontsize=8.5); ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    ax2 = fig.add_axes([0.555, 0.55, 0.42, 0.33])
    ax2.fill_between(t, [m["inflow_gwh"]["p10"] for m in M],
                     [m["inflow_gwh"]["p90"] for m in M], color="#1f7a4d",
                     alpha=0.20, lw=0, label="10–90%")
    ax2.plot(t, [m["inflow_gwh"]["p50"] for m in M], color="#1f7a4d", lw=2.4,
             marker="o", ms=5, label="median")
    ax2.plot(t, [m["norm_gwh"] for m in M], color="0.45", lw=1.3, ls=":",
             label="seasonal norm, current fleet")
    ax2.set_ylabel("inflow energy, GWh/day", fontsize=9.5)
    ax2.set_title("Hydro generation potential", fontsize=11, fontweight="bold",
                  loc="left", color=INK)
    ax2.legend(fontsize=8.5); ax2.grid(lw=0.25, alpha=0.5); ax2.tick_params(labelsize=8.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    ax3 = fig.add_axes([0.055, 0.085, 0.92, 0.34])
    ax3.plot(th, [gen[m]["share"] * 100 for m in hist], color="#111", lw=2.2,
             marker="o", ms=3.5, label="observed")
    ax3.fill_between(t, [m["hydro_share_pct"]["p10"] for m in M],
                     [m["hydro_share_pct"]["p90"] for m in M], color="#b35806",
                     alpha=0.22, lw=0, label="forecast 10–90%")
    ax3.plot(t, [m["hydro_share_pct"]["p50"] for m in M], color="#b35806",
             lw=2.4, marker="s", ms=6, label="forecast median")
    ax3.axvline(th[-1], color="0.6", lw=1.0, ls="--")
    ax3.set_ylabel("hydro, % of national generation", fontsize=9.5)
    ax3.set_title(f"Hydro share of generation — LOYO r={out['share_model']['r_loyo']}, "
                  f"MAE {out['share_model']['mae_pts_loyo']} pts vs "
                  f"{out['share_model']['clim_mae_pts']} for climatology",
                  fontsize=11, fontweight="bold", loc="left", color=INK)
    ax3.legend(fontsize=8.5); ax3.grid(lw=0.25, alpha=0.5); ax3.tick_params(labelsize=8.5)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.text(0.055, 0.018, "inflow distribution pushed through the share model sample "
             "by sample, so the share band inherits inflow uncertainty · C3S seasonal "
             "rainfall deliberately excluded (verified: real ACC but ~+0.01 on inflow)",
             fontsize=7.8, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=118); plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    raise SystemExit(main())
