#!/usr/bin/env python3
"""Every out-of-sample forecast in the record, at one lead, on one chart.

The event scatters answer "how did it do in 2015-16".  This answers the
harder question: how does it do across ALL 26 years, with no window
chosen by me.

Honesty rules, all enforced in code rather than asserted:
  * every point comes from a blocked-CV fold - the model that made it was
    fitted on the other ~11/12 of the record with a 90-day embargo either
    side, so it never saw that day or its neighbours;
  * tau, lag AND the amplitude calibration are re-selected inside each
    training fold, so no hyperparameter has seen the test day either;
  * nothing is filtered.  Dry-season days where nothing happens are
    plotted alongside flood onsets, which is why the cloud is dense near
    the origin - that density is the truth about the problem, and hiding
    it by keeping only "interesting" days would flatter the model badly;
  * the display is a density hexbin, not overplotted dots, because with
    ~9,500 points a scatter's visual weight is dominated by whichever
    points were drawn last.

Zero on the forecast axis is the persistence forecast, so the chart reads
directly as: how often, and how far, does the model step away from
"no change", and does it get paid for it.

    python scripts/sst/scatter_galaxy.py [--lead 10]

Outputs: ~/colombia_hydro/out/scatter_galaxy_h{lead}.json
         ~/colombia_hydro/site/scatter_galaxy_h{lead}.webp
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
import inflow_delta_model as M                                      # noqa: E402
import delta_backtest_long as DB                                    # noqa: E402

PRIV = Path.home() / "colombia_hydro"
EMBARGO = 90
FOLDS = 12


def all_pairs(d, series, lead):
    """Out-of-sample (pred delta, obs delta) for EVERY day, via blocked CV."""
    y, rain = d["y"][series], d["rain"][series]
    roni, stor = d["roni"], d["stor"][series]
    n = len(y)
    edges = np.linspace(0, n, FOLDS + 1).astype(int)
    P, O, Y0, DT, EN = [], [], [], [], []
    for k in range(FOLDS):
        a, b = edges[k], edges[k + 1]
        te = np.zeros(n, bool); te[a:b] = True
        tr = ~te
        tr[max(0, a - EMBARGO):min(n, b + EMBARGO)] = False
        if tr.sum() < 400:
            continue
        tau, lag = M.select_hyper(rain, y, roni, stor, tr, False)
        X, dy = M.design(rain, y, roni, stor, tau, lag)
        m = tr & np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        if m.sum() < 200:
            continue
        beta = M.fit(X, dy, m)
        sh, off = M.shrinkage(beta, rain, y, roni, stor, tau, lag, tr)
        kf_h, ks_h = M.ema(rain, tau), M.ema(rain, M.TAU_SLOW)
        for i0 in range(a, b):
            j = i0 + lead
            if j >= n or not np.isfinite(y[i0]) or not np.isfinite(y[j]):
                continue
            sim = PR.fwd(beta, off, sh, kf_h, ks_h, rain, roni, stor,
                         tau, lag, i0, y[i0], None)
            v = sim[lead - 1]
            if not np.isfinite(v):
                continue
            P.append(off[lead - 1] + sh[lead - 1] * (v - y[i0]))
            O.append(y[j] - y[i0])
            Y0.append(y[i0]); DT.append(str(d["dates"][i0]))
            EN.append(float(roni[i0]))
        print(f"  fold {k+1}/{FOLDS}: tau={tau} lag={lag}  n={len(P)}", flush=True)
    return (np.asarray(P), np.asarray(O), np.asarray(Y0),
            np.asarray(DT), np.asarray(EN))


def stats(P, O):
    rm = float(np.sqrt(np.mean((P - O) ** 2)))
    rp = float(np.sqrt(np.mean(O ** 2)))
    return {"n": int(len(P)),
            "r": round(float(np.corrcoef(P, O)[0, 1]), 3),
            "rmse": round(rm, 2), "rmse_persistence": round(rp, 2),
            "skill_vs_persistence": round(1 - rm / rp, 3),
            "mae": round(float(np.mean(np.abs(P - O))), 2),
            "bias": round(float(np.mean(P - O)), 2),
            "sign_hit_rate": round(float(np.mean(np.sign(P) == np.sign(O))), 3),
            "reliability_slope": round(float(np.dot(P, O) / max(np.dot(P, P), 1e-9)), 3)}


def figure(P, O, Y0, EN, lead, series, st, dec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    NAVY, INK = "#13273d", "#1a2733"
    fig = plt.figure(figsize=(15.2, 9.6))
    hd = fig.add_axes([0, 0.935, 1, 0.065]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.012, 0.62, f"EVERY OUT-OF-SAMPLE FORECAST, {lead}-DAY LEAD — {series}",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    hd.text(0.012, 0.2, f"{st['n']:,} forecasts, 2000-2026, twelve blocked CV folds "
            "with a 90-day embargo — nothing filtered, no window chosen",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")

    lim = float(np.ceil(np.percentile(np.abs(np.concatenate([P, O])), 99.5) / 10) * 10)
    ax = fig.add_axes([0.055, 0.085, 0.46, 0.80])
    hb = ax.hexbin(O, P, gridsize=58, extent=(-lim, lim, -lim, lim),
                   cmap="magma_r", mincnt=1, norm=LogNorm(), linewidths=0)
    ax.plot([-lim, lim], [-lim, lim], color="#111", lw=1.4, ls="--",
            label="perfect (1:1)", zorder=4)
    xs = np.array([-lim, lim])
    b = np.polyfit(O, P, 1)
    ax.plot(xs, b[0] * xs + b[1], color="#1f7a4d", lw=2.0, zorder=5,
            label=f"forecast on observed, slope {b[0]:.2f}")
    ax.axhline(0, color="0.45", lw=1.0, zorder=3)
    ax.axvline(0, color="0.45", lw=1.0, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("observed change over the next "
                  f"{lead} days, pts of norm", fontsize=10)
    ax.set_ylabel(f"forecast change, pts of norm", fontsize=10)
    ax.set_title("Density, not dots — log colour scale", fontsize=11,
                 fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(lw=0.25, alpha=0.4)
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("forecasts per cell", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    ax.text(0.985, 0.03,
            f"r = {st['r']:.3f}\nskill vs persistence = {st['skill_vs_persistence']:+.3f}\n"
            f"sign correct = {st['sign_hit_rate']*100:.0f}%\n"
            f"reliability slope = {st['reliability_slope']:.2f}\n"
            f"RMSE {st['rmse']:.1f} vs {st['rmse_persistence']:.1f} persistence",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            family="monospace", bbox=dict(facecolor="white", alpha=0.85,
                                          edgecolor="#c9d2dc"))

    ax2 = fig.add_axes([0.575, 0.53, 0.39, 0.355])
    c = np.array([x["pred_mid"] for x in dec])
    om = np.array([x["obs_mean"] for x in dec])
    lo = np.array([x["obs_p10"] for x in dec]); hi = np.array([x["obs_p90"] for x in dec])
    ax2.fill_between(c, lo, hi, color="#1f4e8c", alpha=0.18, lw=0,
                     label="observed 10–90%")
    ax2.plot(c, om, color="#1f4e8c", lw=2.2, marker="o", ms=5,
             label="mean observed")
    ax2.plot([c.min(), c.max()], [c.min(), c.max()], color="#111", lw=1.3,
             ls="--", label="perfect calibration")
    ax2.axhline(0, color="0.5", lw=0.8); ax2.axvline(0, color="0.5", lw=0.8)
    ax2.set_xlabel("forecast change (binned by decile)", fontsize=9)
    ax2.set_ylabel("observed change", fontsize=9)
    ax2.set_title("Calibration — is a forecast of +X followed by +X?",
                  fontsize=11, fontweight="bold", loc="left", color=INK)
    ax2.legend(fontsize=8); ax2.grid(lw=0.25, alpha=0.5); ax2.tick_params(labelsize=8)

    ax3 = fig.add_axes([0.575, 0.085, 0.39, 0.335])
    # binned by the size of the PREDICTED move, not the observed one.
    # Binning by |observed| conditions on the outcome: informative, but not
    # actionable, since the size of the move is exactly what is unknown at
    # issue time.  |predicted| IS known that morning, so this version can
    # be traded on: it says how much to trust the model when it calls a
    # big move.
    qs = np.percentile(np.abs(P), np.arange(0, 101, 10))
    mags, sk = [], []
    for i in range(len(qs) - 1):
        k = (np.abs(P) >= qs[i]) & (np.abs(P) < qs[i + 1])
        if k.sum() < 30:
            continue
        rm = np.sqrt(np.mean((P[k] - O[k]) ** 2))
        rp = np.sqrt(np.mean(O[k] ** 2))
        mags.append(0.5 * (qs[i] + qs[i + 1])); sk.append(1 - rm / rp)
    ax3.bar(range(len(sk)), sk, color=["#c62828" if v < 0 else "#1f7a4d" for v in sk],
            alpha=0.9)
    ax3.set_xticks(range(len(sk)))
    ax3.set_xticklabels([f"{m:.0f}" for m in mags], fontsize=7.5)
    ax3.axhline(0, color="0.4", lw=1.0)
    ax3.set_xlabel("size of the FORECAST move, |predicted change| — known at issue",
                   fontsize=9)
    ax3.set_ylabel("skill vs persistence", fontsize=9)
    ax3.set_title("When can you trust it? — by size of the call",
                  fontsize=11, fontweight="bold", loc="left", color=INK)
    ax3.grid(lw=0.25, alpha=0.5, axis="y"); ax3.tick_params(labelsize=8)
    fig.text(0.055, 0.018, "Zero on the vertical axis IS persistence. The dense core "
             "near the origin is real: most days the basin is not doing much, and "
             "removing those quiet days would flatter the model. "
             "Rain before 2024-07 is corrected-satellite.",
             fontsize=8, color="#5a6b7a")
    png = PRIV / "site" / f"scatter_galaxy_h{lead}.webp"
    fig.savefig(png, dpi=118); plt.close(fig)
    print(f"wrote {png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=10)
    ap.add_argument("--series", default="NATIONAL")
    ap.add_argument("--baseline", type=int, default=365)
    a = ap.parse_args()
    M.BASELINE_WIN = a.baseline
    d = DB.add_national(PR.load_all())
    print(f"{a.series}, lead +{a.lead} d, {FOLDS} folds, embargo {EMBARGO} d")
    P, O, Y0, DT, EN = all_pairs(d, a.series, a.lead)
    st = stats(P, O)
    print("\n" + json.dumps(st, indent=1))

    q = np.percentile(P, np.arange(0, 101, 10))
    dec = []
    for i in range(10):
        k = (P >= q[i]) & (P < q[i + 1] if i < 9 else P <= q[i + 1])
        if k.sum() < 20:
            continue
        dec.append({"pred_mid": round(float(np.mean(P[k])), 2),
                    "obs_mean": round(float(np.mean(O[k])), 2),
                    "obs_p10": round(float(np.percentile(O[k], 10)), 2),
                    "obs_p90": round(float(np.percentile(O[k], 90)), 2),
                    "n": int(k.sum())})
    # regime splits, honest slices rather than hand-picked windows
    reg = {}
    for name, k in (("El Nino (ONI>+0.5)", EN > 0.5),
                    ("neutral", np.abs(EN) <= 0.5),
                    ("La Nina (ONI<-0.5)", EN < -0.5),
                    ("dry basin (<80% norm)", Y0 < 80),
                    ("wet basin (>120% norm)", Y0 > 120)):
        if k.sum() > 50:
            reg[name] = stats(P[k], O[k])
    def by_mag(vals, label):
        qq = np.percentile(np.abs(vals), np.arange(0, 101, 10))
        rows = []
        for i in range(10):
            k = (np.abs(vals) >= qq[i]) & (np.abs(vals) < qq[i + 1])
            if k.sum() < 30:
                continue
            rm = float(np.sqrt(np.mean((P[k] - O[k]) ** 2)))
            rp = float(np.sqrt(np.mean(O[k] ** 2)))
            rows.append({"bin_mid": round(0.5 * (qq[i] + qq[i + 1]), 1),
                         "n": int(k.sum()),
                         "skill": round(1 - rm / rp, 3),
                         "sign_hit": round(float(np.mean(np.sign(P[k]) ==
                                                         np.sign(O[k]))), 3)})
        return {label: rows}
    mag = {}
    mag.update(by_mag(P, "by_predicted_magnitude"))
    mag.update(by_mag(O, "by_observed_magnitude"))
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "series": a.series, "lead_days": a.lead, "folds": FOLDS,
           "embargo_days": EMBARGO, "window": f"{DT[0]}..{DT[-1]}",
           "overall": st, "calibration_deciles": dec, "by_regime": reg,
           "skill_by_magnitude": mag,
           "note": "every point out-of-sample; tau/lag and amplitude calibration "
                   "re-selected inside each training fold; nothing filtered"}
    (PRIV / "out" / f"scatter_galaxy_h{a.lead}.json").write_text(json.dumps(out, indent=1))
    print(f"\n{'regime':26}{'n':>7}{'r':>7}{'skill':>8}{'sign%':>7}{'slope':>7}")
    for k, v in reg.items():
        print(f"{k:26}{v['n']:7d}{v['r']:7.3f}{v['skill_vs_persistence']:+8.3f}"
              f"{v['sign_hit_rate']*100:7.0f}{v['reliability_slope']:7.2f}")
    try:
        figure(P, O, Y0, EN, a.lead, a.series, st, dec)
    except Exception as e:                          # noqa: BLE001
        print(f"figure failed: {repr(e)[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
