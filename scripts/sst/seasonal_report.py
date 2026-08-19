#!/usr/bin/env python3
"""Compact PDF: model methodology, backtesting, and the t+1..t+6 outlook.

Six A4-landscape pages, written to be read by someone who has to decide
whether to trust the numbers:

  1  outlook            what the model says, t+1 to t+6, as distributions
  2  methodology        every equation, written out
  3  daily backtest     26 years out-of-sample, and when to trust it
  4  event replays      2015-16 and three other events, cold-started
  5  seasonal backtest  CRPS against climatology, and what does NOT work
  6  assumptions        the honest limits, in one place

    python scripts/sst/seasonal_report.py

Output: ~/colombia_hydro/reports/colombia_inflow_outlook.pdf (+ dated copy)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PRIV = Path.home() / "colombia_hydro"
OUT = PRIV / "reports"
NAVY, INK, MUTED = "#13273d", "#1a2733", "#5a6b7a"
HY, DRY = "#1f7a4d", "#b35806"
W, H = 11.69, 8.27


def load(name):
    p = PRIV / "out" / name
    return json.loads(p.read_text()) if p.exists() else None


def head(fig, title, sub, right=""):
    ax = fig.add_axes([0, 0.935, 1, 0.065]); ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=NAVY))
    ax.text(0.015, 0.62, title, transform=ax.transAxes, color="white",
            fontsize=14.5, fontweight="bold", va="center")
    ax.text(0.015, 0.20, sub, transform=ax.transAxes, color="#b9c6d4",
            fontsize=8.6, va="center")
    if right:
        ax.text(0.985, 0.5, right, transform=ax.transAxes, color="#b9c6d4",
                fontsize=8.4, va="center", ha="right")


def para(ax, x, y, txt, width=150, fs=8.5, color=INK, dy=0.0235, weight=None):
    """Wrap to a character width and return the new y — matplotlib's wrap=True
    does not respect axes width reliably, so lines are broken explicitly."""
    import textwrap
    for line in textwrap.wrap(txt, width):
        ax.text(x, y, line, fontsize=fs, color=color, va="top",
                transform=ax.transAxes, fontweight=weight)
        y -= dy
    return y


def foot(fig, txt):
    fig.text(0.015, 0.015, txt, fontsize=7.0, color=MUTED)


def table(ax, rows, cols, widths, fs=8.0, hl=None, SCALE=1.55):
    ax.set_axis_off()
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                 loc="upper center", colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, SCALE)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#c9d2dc")
        if r == 0:
            cell.set_facecolor(NAVY); cell.set_text_props(color="white",
                                                          fontweight="bold")
        elif hl and (r - 1) in hl:
            cell.set_facecolor("#fdf0e3")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f6f9")
    return t


# ── page 1: the outlook ─────────────────────────────────────────────────────
def page_outlook(pdf, uo, ol):
    fig = plt.figure(figsize=(W, H))
    M = uo["variants"]["ONI_ISSUE"]
    head(fig, "COLOMBIA — NATIONAL INFLOW OUTLOOK, t+1 TO t+6",
         f"issued from {uo['issue_month']} · ONI {uo['oni_at_issue']:+.2f} · "
         f"storage {uo['storage_anom']:+.1f} pts vs norm · every figure is a "
         "distribution, not a point",
         uo["generated"])
    t = [datetime.strptime(m["month"] + "-15", "%Y-%m-%d") for m in M]

    ax = fig.add_axes([0.055, 0.50, 0.42, 0.36])
    ax.fill_between(t, [m["inflow_pct"]["p10"] for m in M],
                    [m["inflow_pct"]["p90"] for m in M], color="#1f4e8c",
                    alpha=0.20, lw=0, label="10–90%")
    ax.plot(t, [m["inflow_pct"]["p50"] for m in M], color="#1f4e8c", lw=2.4,
            marker="o", ms=5, label="median")
    ax.axhline(100, color="0.5", lw=1.0, ls=":", label="normal")
    ax.axhline(43, color="#c62828", lw=1.0, ls="--",
               label="26-yr record low (43%)")
    ax.set_ylabel("inflow, % of norm", fontsize=9)
    ax.set_title("National inflow", fontsize=10.5, fontweight="bold",
                 loc="left", color=INK)
    ax.legend(fontsize=7.4); ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    ax2 = fig.add_axes([0.555, 0.50, 0.42, 0.36])
    ax2.fill_between(t, [m["hydro_share_pct"]["p10"] for m in M],
                     [m["hydro_share_pct"]["p90"] for m in M], color=DRY,
                     alpha=0.22, lw=0, label="10–90%")
    ax2.plot(t, [m["hydro_share_pct"]["p50"] for m in M], color=DRY, lw=2.4,
             marker="s", ms=5, label="median")
    lo = ol.get("latest_observed", {}) if ol else {}
    if lo:
        ax2.axhline(lo["share_pct"], color="#111", lw=1.2, ls="-.",
                    label=f"latest observed {lo['month']} ({lo['share_pct']:.0f}%)")
    ax2.axhline(45.3, color="#c62828", lw=1.0, ls="--",
                label="record low (45.3%)")
    ax2.set_ylabel("hydro, % of national generation", fontsize=9)
    ax2.set_title("Hydro share of generation", fontsize=10.5, fontweight="bold",
                  loc="left", color=INK)
    ax2.legend(fontsize=7.2); ax2.grid(lw=0.25, alpha=0.5); ax2.tick_params(labelsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    rows = [[m["month"],
             f"{m['inflow_pct']['p10']:.0f} – {m['inflow_pct']['p50']:.0f} – "
             f"{m['inflow_pct']['p90']:.0f}",
             f"{m['inflow_gwh']['p10']:.0f} – {m['inflow_gwh']['p50']:.0f} – "
             f"{m['inflow_gwh']['p90']:.0f}",
             f"{m['norm_gwh']:.0f}",
             f"{m['hydro_share_pct']['p10']:.0f} – {m['hydro_share_pct']['p50']:.0f}"
             f" – {m['hydro_share_pct']['p90']:.0f}"] for m in M]
    ax3 = fig.add_axes([0.055, 0.06, 0.915, 0.40])
    table(ax3, rows, ["month", "inflow, % of norm\n(p10 – p50 – p90)",
                      "inflow energy, GWh/day\n(p10 – p50 – p90)",
                      "seasonal norm\nGWh/day",
                      "hydro % of generation\n(p10 – p50 – p90)"],
          [.10, .24, .26, .14, .26], fs=9.4, SCALE=2.3)
    foot(fig, "GWh/day applies TODAY'S fleet norm to a fleet-corrected % of "
              "norm — national inflow energy roughly doubled 2005-2025 on fleet "
              "growth alone, so the two must not be conflated.")
    pdf.savefig(fig); plt.close(fig)


# ── page 2: methodology ─────────────────────────────────────────────────────
def page_method(pdf):
    fig = plt.figure(figsize=(W, H))
    head(fig, "METHODOLOGY — EVERY EQUATION",
         "simple models, chosen so they can be validated honestly")
    ax = fig.add_axes([0, 0, 1, 0.92]); ax.set_axis_off()
    L = [
        ("1 · Exponential memory", None),
        (r"A basin integrates rain. The lightest object with that property is a "
         r"causal exponential moving average with time constant $\tau$:", "n"),
        (r"$K^{\tau}_t = (1-1/\tau)\,K^{\tau}_{t-1} + (1/\tau)\,x_t$"
         r"$\qquad x_t = P_t - \bar{P}(\mathrm{doy})$", "e"),
        (r"Two kernels: fast ($\tau$ fitted per basin, 2–60 d) and slow "
         r"($\tau$=90 d, antecedent wetness the fast kernel has forgotten).", "n"),
        ("2 · The daily model — a linear reservoir on the CHANGE", None),
        (r"$\Delta y_t = a + b_{rec}(y_{t-1}-\bar{y}) + b_r x_{t-\ell} + "
         r"b_f K^{\tau}_{t-\ell} + b_s K^{90}_{t-\ell} + b_e N_t + b_{st} S_{t-1}$",
         "e"),
        (r"$b_{rec}<0$ is recession — the discrete form of $dQ/dt=-kQ$. The target "
         r"is the RAW daily change: the previously used 5-day mean has lag-1 "
         r"autocorrelation 0.978, so persistence scores r=0.98 on it and any r "
         r"quoted there measures the smoothing. Raw $\Delta y$ has -0.09.", "n"),
        ("3 · It timesteps, like an NWP model", None),
        (r"state $z_t=(y_t,K^{\tau}_t,K^{90}_t,S_t,N_t)$  →  force with $x_{t+1}$  "
         r"→  step  →  repeat.  A 10-day forecast is the same one-day equation "
         r"applied ten times with its own output fed back.", "n"),
        ("4 · Amplitude calibration — and the trap in it", None),
        (r"$\hat{y}_{t+h} = y_t + \alpha_h + s_h(\hat{y}^{raw}_{t+h}-y_t)$"
         r"$\qquad s_h = \langle pred,obs\rangle/\langle pred,pred\rangle$", "e"),
        (r"Estimating $s_h$ on the days the coefficients came from returns "
         r"$s\approx0.9$; the honest out-of-sample optimum is $\approx0.5$. "
         r"Applying 0.9 converts +0.10 RMSE skill into -0.02. So $s_h$ is itself "
         r"cross-validated.", "n"),
        ("5 · The monthly model — log space", None),
        (r"$\log y_{t+L} = c_0 + c_1 N_t + c_2 N_t \sin\theta + c_3 N_t\cos\theta"
         r" + c_4 S_t + c_5 A^{30}_t + c_6 A^{90}_t + c_7\sin\theta + c_8\cos\theta$",
         "e"),
        (r"The $N\!\cdot\!\sin\theta$ terms matter: Colombia's ENSO response is "
         r"seasonal. Logs because a linear fit extrapolates without bound — at "
         r"ONI +2.12 it returned 13% of norm with a NEGATIVE p10, against a "
         r"26-year observed minimum of 43%. Logs also verify better at every lead.",
         "n"),
        ("6 · Hydro share of generation", None),
        (r"$\mathrm{logit}(share) = d_0 + d_1\log(y/100) + d_2 S + d_3 t + "
         r"d_4\sin\theta + d_5\cos\theta + d_6\,\mathrm{logit}(share_{t-1})$", "e"),
        (r"Logit because a share is bounded. The previous month's OBSERVED share "
         r"is essential: without it r=0.60 and the fit breaks down through 2026 "
         r"(residuals to -11.5 pts as hydro generates far less than inflow "
         r"implies). With it, r=0.87, MAE 3.2 pts vs 6.7 for climatology.", "n"),
    ]
    y = 0.975
    for txt, kind in L:
        if kind is None:
            y -= 0.008
            ax.text(0.040, y, txt, fontsize=10.6, fontweight="bold", color=HY,
                    transform=ax.transAxes, va="top")
            y -= 0.036
        elif kind == "e":
            ax.text(0.060, y, txt, fontsize=10.0, color=INK,
                    transform=ax.transAxes, va="top")
            y -= 0.048
        else:
            y = para(ax, 0.040, y, txt, width=152, fs=8.3, dy=0.0215)
            y -= 0.006
    foot(fig, "Every hyperparameter is either fitted inside a training fold or "
              "fixed a priori; none is chosen on test performance.")
    pdf.savefig(fig); plt.close(fig)


# ── page 3: daily backtest ──────────────────────────────────────────────────
def _box(ax, x, y, w, h, txt, fc, ec, fs=8.0, bold=False, tc="#1a2733"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                linewidth=1.1, facecolor=fc, edgecolor=ec,
                                transform=ax.transAxes, zorder=2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=tc, transform=ax.transAxes, zorder=3,
            fontweight="bold" if bold else None, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, style="-|>", lw=1.2, color="#5a6b7a"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords=ax.transAxes,
                textcoords=ax.transAxes, zorder=1,
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                shrinkA=1, shrinkB=1))


def page_flow(pdf):
    """How data becomes a forecast — one picture."""
    fig = plt.figure(figsize=(W, H))
    head(fig, "HOW A FORECAST IS BUILT",
         "inputs on the left, the two model chains in the middle, deliverables "
         "on the right")
    ax = fig.add_axes([0, 0, 1, 0.925]); ax.set_axis_off()
    IN, MID, OUT_, ACC = "#eaf1f8", "#e8f3ec", "#fdf0e3", "#f2f4f7"
    EI, EM, EO = "#1f4e8c", "#1f7a4d", "#b35806"

    ax.text(0.055, 0.955, "OBSERVED INPUTS", fontsize=8.6, fontweight="bold",
            color=EI, transform=ax.transAxes)
    obs = [("GPM IMERG\n0.1°, 2000→ (9,574 d)", 0.86),
           ("IDEAM gauges\n549 stations, 2024→", 0.755),
           ("XM AporEner\nper-river inflow energy", 0.65),
           ("XM generation\nhydro & total, 2000→", 0.545),
           ("Reservoir storage\n% full vs doy norm", 0.44),
           ("ONI / RONI\nENSO state", 0.335)]
    for t, y in obs:
        _box(ax, 0.045, y, 0.185, 0.075, t, IN, EI, fs=7.6)

    _box(ax, 0.275, 0.79, 0.175, 0.075,
         "Gauge correction\n$F$ field (log-IDW)", ACC, EI, fs=7.8)
    _box(ax, 0.275, 0.66, 0.175, 0.085,
         "Energy-weighted\nbasin rain  $x_t$\n(anomaly vs clim)", ACC, EI,
         fs=7.8, bold=True)
    _box(ax, 0.275, 0.545, 0.175, 0.075,
         "EMA kernels\n$K^{\\tau}_t,\\;K^{90}_t$", ACC, EI, fs=8.4)
    for y0 in (0.895, 0.79):
        _arrow(ax, 0.232, y0, 0.273, 0.83)
    _arrow(ax, 0.362, 0.788, 0.362, 0.747)
    _arrow(ax, 0.362, 0.658, 0.362, 0.621)

    _box(ax, 0.50, 0.60, 0.215, 0.145,
         "DAILY CHAIN\n\nlinear reservoir on\nthe CHANGE, timestepped\n"
         "one day at a time\n\n$\\Delta y_t = f(z_t,\\,x_{t+1})$",
         MID, EM, fs=8.2, bold=True)
    _box(ax, 0.50, 0.33, 0.215, 0.145,
         "SEASONAL CHAIN\n\nlog-space monthly model\non ENSO, storage and\n"
         "antecedent wetness\n\n$\\log y_{t+L} = g(N_t,S_t,A_t)$",
         MID, EM, fs=8.2, bold=True)
    _box(ax, 0.50, 0.13, 0.215, 0.115,
         "SHARE MODEL\n\nlogit share on inflow\nand last month's\nOBSERVED share",
         MID, EM, fs=8.2, bold=True)
    _arrow(ax, 0.452, 0.583, 0.497, 0.672)
    _arrow(ax, 0.232, 0.478, 0.497, 0.44)
    _arrow(ax, 0.232, 0.372, 0.497, 0.41)
    _arrow(ax, 0.232, 0.583, 0.497, 0.20)
    _arrow(ax, 0.607, 0.325, 0.607, 0.248)

    _box(ax, 0.765, 0.735, 0.19, 0.10,
         "AIFS + IFS ensemble\n~101 members\nbias-corrected", IN, EI, fs=7.8)
    _arrow(ax, 0.762, 0.785, 0.718, 0.71)

    _box(ax, 0.765, 0.575, 0.19, 0.105,
         "Balance of month\nDAILY distribution\n(p10/p50/p90)", OUT_, EO,
         fs=8.0, bold=True)
    _box(ax, 0.765, 0.375, 0.19, 0.105,
         "Months +1..+6\nmonthly distributions\n% of norm & GWh/day", OUT_, EO,
         fs=8.0, bold=True)
    _box(ax, 0.765, 0.16, 0.19, 0.10,
         "Hydro share of\nnational generation", OUT_, EO, fs=8.0, bold=True)
    _arrow(ax, 0.718, 0.66, 0.762, 0.627)
    _arrow(ax, 0.718, 0.40, 0.762, 0.427)
    _arrow(ax, 0.718, 0.19, 0.762, 0.21)
    _arrow(ax, 0.86, 0.573, 0.86, 0.482, style="-|>", color="#b35806")
    ax.text(0.868, 0.527, "hands over at ~15 d", fontsize=7.2, color="#b35806",
            transform=ax.transAxes, va="center")

    ax.text(0.055, 0.075, "Every arrow into a model box is data known AT ISSUE "
            "TIME. The only forecast input is the rain ensemble; the seasonal "
            "chain uses none at all.", fontsize=8.4, color=INK,
            transform=ax.transAxes)
    ax.text(0.055, 0.040, "Validation mirrors the flow: the daily chain is scored "
            "against persistence, the seasonal chain against a climatological "
            "distribution, and the share model against climatology.",
            fontsize=8.4, color=INK, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def page_steps(pdf):
    """Step by step: what goes in, what comes out."""
    fig = plt.figure(figsize=(W, H))
    head(fig, "STEP BY STEP — INPUTS, EQUATION, OUTPUT",
         "each row is one transformation; symbols are defined on the "
         "methodology page")
    rows = [
        ["1", "Correct the satellite",
         "IMERG $P^{raw}$; 549 IDEAM gauges",
         "$P_t = P^{raw}_t \\cdot F(\\mathbf{s})$",
         "gauge-corrected daily rain"],
        ["2", "Reduce to basins",
         "corrected grid; river catchments; AporEner",
         "$P^b_t=\\sum_{\\mathbf{s}} w_b(\\mathbf{s})P_t(\\mathbf{s})$",
         "basin rain, energy-weighted"],
        ["3", "Take the anomaly",
         "basin rain; harmonic climatology",
         "$x_t = P^b_t - \\bar{P}^b(\\mathrm{doy})$",
         "rain anomaly, mm/day"],
        ["4", "Build memory",
         "rain anomaly; fitted $\\tau$",
         "$K^{\\tau}_t=(1-\\frac{1}{\\tau})K^{\\tau}_{t-1}"
         "+\\frac{1}{\\tau}x_t$",
         "fast + slow kernel states"],
        ["5", "Predict the daily CHANGE",
         "$y_{t-1}$, $x$, $K^{\\tau}$, $K^{90}$, $N_t$, $S_{t-1}$",
         "$\\Delta y_t=a+b_{rec}(y_{t-1}-\\bar{y})+\\ldots$",
         "$\\Delta$ inflow, % of norm"],
        ["6", "Timestep forward",
         "state $z_t$; next day's rain (ensemble member)",
         "$y_{t+1}=y_t+\\Delta y(z_t,x_{t+1})$",
         "15-day path, per member"],
        ["7", "Calibrate amplitude",
         "raw path; $s_h,\\alpha_h$ from inner CV",
         "$\\hat{y}_{t+h}=y_t+\\alpha_h+s_h(\\hat{y}^{raw}_{t+h}-y_t)$",
         "calibrated path"],
        ["8", "Make it a distribution",
         "member paths; CV residuals by lead",
         "$F_h=\\{\\hat{y}^{(m)}_{t+h}+\\epsilon^{(j)}_h\\}$",
         "daily p10/p50/p90"],
        ["9", "Beyond the weather",
         "$N_t$, $S_t$, $A^{30}_t$, $A^{90}_t$",
         "$\\log y_{t+L}=c_0+c_1N_t+\\ldots$",
         "monthly distributions, +1..+6"],
        ["10", "Aggregate to national",
         "basin series; energy shares $w_b$",
         "$y^{nat}=\\sum_b w_b\\,y^b$",
         "national % of norm"],
        ["11", "Convert to energy",
         "% of norm; CURRENT fleet doy norm",
         "$\\mathrm{GWh/d}=\\frac{y^{nat}}{100}\\cdot\\mathrm{Norm(doy)}$",
         "inflow energy, GWh/day"],
        ["12", "Hydro share",
         "inflow draw; $S$; last month's observed share",
         "$\\mathrm{logit}(sh)=d_0+d_1\\log\\frac{y}{100}+\\ldots$",
         "hydro % of generation"],
    ]
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.85])
    t = table(ax, rows, ["#", "step", "inputs", "equation", "output"],
              [.035, .20, .265, .285, .215], fs=8.2, SCALE=2.05)
    for (r, c), cell in t.get_celld().items():
        if c == 3 and r > 0:
            cell.set_text_props(fontsize=9.2)
        if c in (0, 1) and r > 0:
            cell.set_text_props(fontweight="bold" if c == 0 else None)
    foot(fig, "Steps 1-8 are the daily chain, 9 the seasonal chain, 10-12 shared. "
              "Only step 6 consumes a forecast; everything else is observed at "
              "issue time.")
    pdf.savefig(fig); plt.close(fig)


def page_daily(pdf, gal, gal1, db):
    fig = plt.figure(figsize=(W, H))
    o, o1 = gal["overall"], (gal1 or {}).get("overall", {})
    head(fig, "BACKTEST — 26 YEARS, EVERY FORECAST OUT-OF-SAMPLE",
         f"{o['n']:,} forecasts at a {gal['lead_days']}-day lead · twelve blocked "
         f"CV folds, 90-day embargo · nothing filtered, no window chosen",
         gal["window"])
    ax = fig.add_axes([0.055, 0.50, 0.40, 0.36])
    dec = gal["calibration_deciles"]
    c = np.array([x["pred_mid"] for x in dec]); om = np.array([x["obs_mean"] for x in dec])
    lo = np.array([x["obs_p10"] for x in dec]); hi = np.array([x["obs_p90"] for x in dec])
    ax.fill_between(c, lo, hi, color="#1f4e8c", alpha=0.16, lw=0, label="observed 10–90%")
    ax.plot(c, om, color="#1f4e8c", lw=2.2, marker="o", ms=5, label="mean observed")
    ax.plot([c.min(), c.max()], [c.min(), c.max()], color="#111", lw=1.3, ls="--",
            label="perfect calibration")
    ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("forecast change (decile bins)", fontsize=8.6)
    ax.set_ylabel("observed change, pts of norm", fontsize=8.6)
    ax.set_title("Is a forecast of +X followed by +X?", fontsize=10.5,
                 fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=7.4); ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8)

    ax2 = fig.add_axes([0.545, 0.50, 0.42, 0.36])
    mg = gal["skill_by_magnitude"]["by_predicted_magnitude"]
    xs = np.arange(len(mg))
    ax2.bar(xs, [m["skill"] for m in mg],
            color=["#c62828" if m["skill"] < 0.02 else HY for m in mg], alpha=0.9)
    for i, m in enumerate(mg):
        ax2.text(i, m["skill"] + 0.008, f"{m['sign_hit']*100:.0f}%", ha="center",
                 fontsize=6.8, color=INK)
    ax2.set_xticks(xs); ax2.set_xticklabels([f"{m['bin_mid']:.0f}" for m in mg],
                                            fontsize=7.4)
    ax2.axhline(0, color="0.4", lw=1.0)
    ax2.set_xlabel("size of the FORECAST move, pts (known at issue)", fontsize=8.6)
    ax2.set_ylabel("skill vs persistence", fontsize=8.6)
    ax2.set_title("When to trust it — % = direction correct", fontsize=10.5,
                  fontweight="bold", loc="left", color=INK)
    ax2.grid(lw=0.25, alpha=0.5, axis="y"); ax2.tick_params(labelsize=8)

    rows = [["+1 day (needs only tomorrow's rain forecast)",
             f"{o1.get('n', 0):,}", f"{o1.get('r', float('nan')):.3f}",
             f"{o1.get('skill_vs_persistence', float('nan')):+.3f}",
             f"{o1.get('sign_hit_rate', 0)*100:.0f}%",
             f"{o1.get('reliability_slope', float('nan')):.2f}"],
            [f"+{gal['lead_days']} days (assumes perfect rain forecast)",
             f"{o['n']:,}", f"{o['r']:.3f}", f"{o['skill_vs_persistence']:+.3f}",
             f"{o['sign_hit_rate']*100:.0f}%", f"{o['reliability_slope']:.2f}"]]
    ax3 = fig.add_axes([0.055, 0.30, 0.915, 0.13])
    table(ax3, rows, ["horizon", "n", "r", "RMSE skill vs\npersistence",
                      "direction\ncorrect", "reliability\nslope"],
          [.40, .10, .10, .16, .12, .12], fs=8.6)
    ax4 = fig.add_axes([0.055, 0.05, 0.915, 0.22]); ax4.set_axis_off()
    ax4.text(0, 0.99, "What these mean", fontsize=10.4, fontweight="bold",
             color=HY, va="top", transform=ax4.transAxes)
    y = 0.80
    for it in [
        "Reliability slope ~ 1.0 is calibration: a call of \u201c+30 points over ten "
        "days\u201d was followed on average by about +30.",
        "Skill rises with lead (+0.24 to +0.33) because persistence decays while a "
        "rain-driven model holds - the model is worth most where the decision "
        "horizon is longest.",
        "Skill is NOT uniform. Below ~5 points the model is a coin flip and should "
        "be ignored; above 25 points it calls direction correctly 87-96% of the "
        "time. Its own signal strength is the confidence filter.",
        "The perfect-rain assumption is cheap, not convenient: degrading rain to the "
        "AIFS-ENS measured error moved skill by <0.01 at every lead.",
    ]:
        y = para(ax4, 0.004, y, "\u2022 " + it, width=152, fs=8.2, dy=0.095)
        y -= 0.02
    foot(fig, "Rain before 2024-07 is corrected-satellite; the daily IDEAM gauge "
              "blend begins then.")
    pdf.savefig(fig); plt.close(fig)


# ── page 4: event replays ───────────────────────────────────────────────────
def page_events(pdf, db, sc):
    fig = plt.figure(figsize=(W, H))
    head(fig, "EVENT REPLAYS — COLD-STARTED",
         "each window and a 90-day embargo removed from training, then a fresh "
         "15-day forecast launched from every day inside it")
    R = db["event"]["NATIONAL"]
    tr = R.get("traj", [])
    ax = fig.add_axes([0.055, 0.47, 0.915, 0.40])
    ox = [datetime.strptime(t["init"], "%Y-%m-%d") for t in tr]
    oy = [t["y0"] for t in tr]
    for k, t0 in enumerate(tr):
        if k % 5:
            continue
        ti = datetime.strptime(t0["init"], "%Y-%m-%d")
        xs = [ti + __import__("datetime").timedelta(days=j + 1)
              for j in range(len(t0["path"]))]
        ok = [(x, v) for x, v in zip(xs, t0["path"]) if v is not None]
        if ok:
            ax.plot([a for a, _ in ok], [b for _, b in ok], color="#c62828",
                    lw=0.85, alpha=0.55)
    ax.plot(ox, oy, color="#111", lw=2.3, zorder=5, label="observed")
    ax.plot([], [], color="#c62828", lw=1.2, label="15-day forecasts (every 5th)")
    ax.axhline(100, color="0.55", lw=0.8, ls=":")
    ax.set_ylabel("national inflow, % of norm", fontsize=9)
    ax.set_title("2015–16 super El Niño — every red trace is a forecast the model "
                 "would have issued that morning", fontsize=10.5,
                 fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=8); ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    rows = []
    for ev, R2 in db.get("events", {}).items():
        n = R2.get("NATIONAL")
        if not n:
            continue
        g = lambda h: (n["leads"].get(str(h), n["leads"].get(h, {}))
                       or {}).get("skill_vs_persistence")
        rows.append([ev.replace(":", " → "),
                     *[f"{g(h):+.3f}" if g(h) is not None else "--"
                       for h in (1, 3, 5, 7, 10, 15)]])
    ax2 = fig.add_axes([0.055, 0.24, 0.60, 0.18])
    table(ax2, rows, ["event window", "h1", "h3", "h5", "h7", "h10", "h15"],
          [.34, .11, .11, .11, .11, .11, .11], fs=8.2, hl={0})
    ax3 = fig.add_axes([0.68, 0.24, 0.29, 0.18])
    if sc:
        srows = [[k.split(":")[0], f"{v['r']:.2f}", f"{v['skill']:+.2f}",
                  f"{v['hit_rate_sign']*100:.0f}%"]
                 for k, v in sc["events"].items()]
        table(ax3, srows, ["event", "r", "skill", "sign"],
              [.34, .22, .22, .22], fs=8.0)
    ax4 = fig.add_axes([0.055, 0.05, 0.915, 0.16]); ax4.set_axis_off()
    ax4.text(0, 0.95, "Reading the replays", fontsize=10.5, fontweight="bold",
             color=HY, va="top")
    ax4.text(0, 0.68,
             "• National skill is positive at every lead in all four events, "
             "strongest through 2015-16 (+0.27 at h1 to +0.36 at h15) — the hardest "
             "and most valuable case.\n"
             "• During that window model RMSE runs 13.7–18.6 % of norm against "
             "persistence at 18.8–28.9.\n"
             "• CALDAS is the one failure (−0.32 to −0.47 in 2015-16, selecting "
             "τ=30–60 d where everything else picks τ=2). It is 4.2% of national "
             "energy, so national is unaffected — but it is not usable standalone.",
             fontsize=8.6, color=INK, va="top", linespacing=1.7)
    pdf.savefig(fig); plt.close(fig)


# ── page 5: seasonal backtest ───────────────────────────────────────────────
def page_seasonal(pdf, si, cv):
    fig = plt.figure(figsize=(W, H))
    head(fig, "SEASONAL BACKTEST — MONTHLY MEAN NATIONAL INFLOW",
         f"probabilistic, CRPS against a climatological distribution · "
         f"{si['validation']} · {si['period']}")
    cols = {"STAT": "#1f4e8c", "C3S": DRY, "BOTH": HY}
    lbl = {"STAT": "ENSO + storage + antecedent", "C3S": "C3S rainfall alone",
           "BOTH": "both"}
    ax = fig.add_axes([0.055, 0.50, 0.40, 0.36])
    for m, R in si["models"].items():
        L = sorted(int(k) for k in R["by_lead"])
        ax.plot(L, [R["by_lead"][str(k)]["crps_skill"] for k in L],
                color=cols[m], lw=2.1, marker="o", ms=5, label=lbl[m])
    ax.axhline(0, color="0.4", lw=1.1)
    ax.set_xlabel("lead, months", fontsize=8.8)
    ax.set_ylabel("CRPS skill vs climatology", fontsize=8.8)
    ax.set_title("Probabilistic skill", fontsize=10.5, fontweight="bold",
                 loc="left", color=INK)
    ax.legend(fontsize=7.8); ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8)

    ax2 = fig.add_axes([0.545, 0.50, 0.42, 0.36])
    if cv:
        L = sorted(int(k) for k in cv["by_lead"])
        ax2.plot(L, [cv["by_lead"][str(k)]["acc"] for k in L], color=HY,
                 lw=2.2, marker="o", ms=5, label="C3S rainfall ACC")
        ax2.plot(L, [cv["by_lead"][str(k)]["acc_oni_regression"] or np.nan
                     for k in L], color="#c62828", lw=1.7, ls="--", marker="s",
                 ms=4, label="ONI regression")
        em = [cv["by_lead"][str(k)].get("emos_calibrated", {}).get("crps_skill")
              for k in L]
        ax2.plot(L, em, color="#7b1fa2", lw=1.7, marker="^", ms=4,
                 label="C3S CRPS skill after EMOS")
        ax2.axhline(0, color="0.4", lw=1.0)
        ax2.set_xlabel("lead, months", fontsize=8.8)
        ax2.set_title("The C3S rain forecast, verified directly",
                      fontsize=10.5, fontweight="bold", loc="left", color=INK)
        ax2.legend(fontsize=7.6); ax2.grid(lw=0.25, alpha=0.5)
        ax2.tick_params(labelsize=8)

    rows = []
    for L in range(1, 7):
        a = si["models"]["STAT"]["by_lead"].get(str(L))
        b = si["models"]["C3S"]["by_lead"].get(str(L))
        c = si["models"]["BOTH"]["by_lead"].get(str(L))
        if not a:
            continue
        rows.append([str(L), f"{a['n']}", f"{a['r']:.2f}", f"{a['crps_skill']:+.3f}",
                     f"{b['r']:.2f}", f"{b['crps_skill']:+.3f}",
                     f"{c['crps_skill']:+.3f}", f"{c['pit_mean']:.2f}",
                     f"{c['coverage_10_90']*100:.0f}%"])
    ax3 = fig.add_axes([0.055, 0.28, 0.915, 0.17])
    table(ax3, rows, ["lead\n(months)", "n", "STAT\nr", "STAT\nCRPS skill",
                      "C3S\nr", "C3S\nCRPS skill", "BOTH\nCRPS skill",
                      "PIT\nmean", "10–90%\ncoverage"],
          [.09, .08, .09, .14, .09, .14, .14, .10, .13], fs=8.4)
    ax4 = fig.add_axes([0.055, 0.03, 0.915, 0.20]); ax4.set_axis_off()
    ax4.text(0, 0.99, "What works, and what does not", fontsize=10.4,
             fontweight="bold", color=HY, va="top", transform=ax4.transAxes)
    y = 0.80
    for it in [
        "There IS seasonal skill and it is calibrated: CRPS skill +0.24 at one "
        "month to +0.11 at six, PIT means 0.489-0.501, 76-86% of observations "
        "inside a nominal 80% band.",
        "It comes from ENSO state, storage and antecedent wetness - all known at "
        "issue time. No seasonal rainfall forecast is required.",
        "The C3S rain forecast is NOT garbage: verified directly it has real "
        "anomaly correlation (0.30-0.47) and beats an ONI regression at leads 2-6. "
        "But it is +4.2 mm/day too wet, its raw ensemble has NEGATIVE CRPS skill, "
        "and EMOS shows the spread carries no information (d<0 at every lead) - its "
        "value is entirely in the mean.",
        "Calibrated it still adds only ~+0.01 to monthly inflow, because its "
        "information is already in ENSO and storage. It is therefore excluded from "
        "the operational path - removing a dependency, not adding one.",
    ]:
        y = para(ax4, 0.004, y, "\u2022 " + it, width=155, fs=8.2, dy=0.105)
        y -= 0.02
    pdf.savefig(fig); plt.close(fig)


# ── page 6: assumptions and limits ──────────────────────────────────────────
def page_limits(pdf, uo, gal):
    fig = plt.figure(figsize=(W, H))
    head(fig, "ASSUMPTIONS, LIMITS AND KNOWN FAILURES",
         "everything that could make the front page wrong, in one place")
    ax = fig.add_axes([0, 0, 1, 0.92]); ax.set_axis_off()
    ef = uo.get("enso_forecast_n34", {})
    et = " · ".join(f"{m[-2:]}/{m[2:4]}: {v[1]:+.1f}" for m, v in list(ef.items())[:6])
    blocks = [
        ("The ENSO forcing is beyond the training set", [
            f"Multi-model Niño-3.4 forecast ({uo.get('enso_forecast_issue','')}): {et}.",
            "That exceeds 2015-16 (+2.65) and 1997-98 (~+2.4). The monthly model was "
            "fitted on a record whose maximum ONI is +2.65, so it is extrapolating.",
            "Running it with the forecast ENSO trajectory instead of ONI-at-issue "
            "changes the answer very little (Jan-2027 median 40% vs 39% of norm): the "
            "fitted response saturates rather than scaling linearly. That is "
            "reassuring for robustness and a warning against reading precision into "
            "the depth."]),
        ("The model overstates depth at extreme ENSO", [
            "Cold-started on 2015-16 it called the collapse six months ahead but put "
            "Jan–Mar 2016 at 34–40% of norm where 53–64% verified.",
            "The same signature is present now. Direction: high confidence. "
            "Magnitude: treat the median as too deep, and note that the p90 is closer "
            "to what 2015-16 actually delivered."]),
        ("Two forecast months fall below anything observed", [
            "Dec-2026 and Jan-2027 medians sit below the 26-year observed monthly "
            "minimum of 43% of norm, and the hydro-share median reaches the all-time "
            "record low of 45.3%. Read 'severe', not the point value."]),
        ("The hydro share is being driven by behaviour, not only hydrology", [
            "Through 2026 hydro generated far less than inflow implied — residuals "
            "of −1.6, −2.7, −6.2, −11.5 points from March to July. That is reservoir "
            "conservation ahead of the event plus new non-hydro capacity.",
            "The model absorbs it only by anchoring on last month's OBSERVED share. "
            "If that behaviour changes abruptly the forecast will lag it by a month."]),
        ("Structural caveats", [
            "Rain before 2024-07 is corrected-satellite; the daily IDEAM gauge blend "
            "starts then, so the historical backtest sits across that seam.",
            "XM's daily inflow contains reporting noise (BLANCO has lag-1 "
            "autocorrelation 0.083; CHUZA more than doubles on 1,396 days — these are "
            "diversion accounts). National aggregation averages much of it away, "
            "which is why national is modelled directly rather than summed from "
            "basins.",
            "CALDAS fails standalone in 2015-16 and should not be used alone.",
            "Total generation was ~180 GWh/day in 2015-16 and is ~248 now: a given "
            "hydro share implies less system stress than the same share did then."]),
    ]
    y = 0.975
    for title, items in blocks:
        ax.text(0.040, y, title, fontsize=10.6, fontweight="bold", color=DRY,
                transform=ax.transAxes, va="top")
        y -= 0.036
        for it in items:
            y = para(ax, 0.052, y, "\u2022 " + it, width=150, fs=8.3, dy=0.0215)
            y -= 0.005
        y -= 0.012
    foot(fig, "Full working, code and dated ledger: colombia-hydro-research.")
    pdf.savefig(fig); plt.close(fig)


def page_balmonth(pdf, uo):
    """Colombia daily inflow through the balance of the month."""
    fig = plt.figure(figsize=(W, H))
    D = uo.get("daily") or []
    if not D:
        return
    head(fig, "COLOMBIA — BALANCE OF MONTH, DAILY",
         "national inflow from the live AIFS+IFS ensemble, member by member, "
         "convolved with the model's own per-lead residual",
         f"issued {uo['generated']}")
    t = [datetime.strptime(r["date"], "%Y-%m-%d") for r in D]
    ax = fig.add_axes([0.055, 0.50, 0.915, 0.36])
    O = uo.get("observed_recent") or []
    if O:
        to = [datetime.strptime(r["date"], "%Y-%m-%d") for r in O]
        ax.plot(to, [r["pct"] for r in O], color="#111", lw=1.9, marker="o",
                ms=2.6, label="observed", zorder=5)
        # join the last observation to the first forecast so the eye follows
        ax.plot([to[-1], t[0]], [O[-1]["pct"], D[0]["p50"]], color="#1f4e8c",
                lw=1.4, ls=":", zorder=4)
        ax.axvline(to[-1], color="0.55", lw=1.0, ls="--", zorder=1)
        ax.text(to[-1], ax.get_ylim()[1], "  forecast from here", fontsize=7.4,
                color="#5a6b7a", va="top", zorder=6)
    ax.fill_between(t, [r["p10"] for r in D], [r["p90"] for r in D],
                    color="#1f4e8c", alpha=0.20, lw=0, label="10–90%")
    ax.plot(t, [r["p50"] for r in D], color="#1f4e8c", lw=2.4, marker="o",
            ms=4, label="median")
    ax.axhline(100, color="0.5", lw=1.0, ls=":", label="normal")
    ax.set_ylabel("national inflow, % of norm", fontsize=9)
    ax.set_title("Day by day — observed history and the horizon the rain "
                 "forecast actually covers", fontsize=10.5, fontweight="bold",
                 loc="left", color=INK)
    ax.legend(fontsize=8); ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ex = lambda r, k: (f"{r['exceed'][k]*100:.0f}%" if r.get("exceed") else "--")
    rows = [[r["date"], f"+{r['lead']}", f"{r['p10']:.0f}", f"{r['p50']:.0f}",
             f"{r['p90']:.0f}", ex(r, "gt125"), ex(r, "gt150")] for r in D]
    half = (len(rows) + 1) // 2
    for i, chunk in enumerate((rows[:half], rows[half:])):
        if not chunk:
            continue
        axt = fig.add_axes([0.055 + i * 0.475, 0.06, 0.44, 0.37])
        table(axt, chunk, ["date", "lead", "p10", "p50", "p90",
                           "P(>125%)", "P(>150%)"],
              [.23, .10, .12, .12, .12, .15, .15], fs=8.0, SCALE=1.35)
    # reconcile the two halves: month-to-date observed + this forecast for the
    # balance gives an implied monthly mean, which can be compared with what the
    # monthly model said from last month's state.
    rec = uo.get("reconcile")
    if rec:
        import textwrap
        axr = fig.add_axes([0.055, 0.425, 0.915, 0.065]); axr.set_axis_off()
        axr.text(0.005, 0.72, "\n".join(textwrap.wrap(rec, 128)), fontsize=8.6,
                 color=INK, va="top", transform=axr.transAxes, linespacing=1.5,
                 bbox=dict(facecolor="#fdf0e3", edgecolor="#c9d2dc", pad=5))
    foot(fig, "Zero rain-forecast assumption here: these are the days the "
              "ensemble genuinely covers. Beyond ~15 days the outlook hands over "
              "to the monthly model on page 1.")
    pdf.savefig(fig); plt.close(fig)


def page_brazil(pdf, bs):
    """Brazil DJF heat — the demand-side counterpart to the inflow story."""
    import textwrap
    fig = plt.figure(figsize=(W, H))
    head(fig, "BRAZIL — CHANCE OF THE WARMEST SUMMER ON RECORD",
         f"DJF 2-m temperature weighted by metro population "
         f"({bs['metros']} metros, {bs['pop_millions']:.0f} M) · C3S "
         f"bias-corrected against its own 1993-2016 hindcast",
         bs["observed_record_period"])
    obs = {int(k): v for k, v in bs["observed"].items()}
    yrs = sorted(obs)
    rec = bs["record_value_c"]
    p = bs["forecast_p10_p50_p90"]
    ax = fig.add_axes([0.055, 0.45, 0.62, 0.41])
    v = [obs[y] for y in yrs]
    ax.bar(yrs, v, color=["#c62828" if abs(x - rec) < 1e-9 else "#9db8d8"
                          for x in v])
    ax.axhline(rec, color="#c62828", lw=1.2, ls="--",
               label=f"record {rec:.2f} °C ({bs['record_summer']})")
    ny = yrs[-1] + 1
    ax.errorbar([ny], [p[1]], yerr=[[p[1] - p[0]], [p[2] - p[1]]], fmt="o",
                color=DRY, ms=9, lw=2.2, capsize=5,
                label=f"forecast {ny}/{str(ny+1)[2:]} (10–90%)")
    ax.set_ylim(min(v) - 0.5, max(max(v), p[2]) + 0.4)
    ax.set_ylabel("DJF mean, °C (population-weighted)", fontsize=9)
    ax.set_xlabel("DJF start year", fontsize=9)
    ax.set_title("Observed summers and the coming one", fontsize=10.5,
                 fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=8); ax.grid(lw=0.25, alpha=0.5, axis="y")
    ax.tick_params(labelsize=8)

    ax2 = fig.add_axes([0.72, 0.45, 0.25, 0.41]); ax2.set_axis_off()
    ax2.text(0.5, 0.92, f"{bs['p_warmest_on_record']*100:.0f}%", ha="center",
             fontsize=44, fontweight="bold", color="#c62828",
             transform=ax2.transAxes)
    ax2.text(0.5, 0.70, "chance of the warmest\npopulation-weighted\nsummer on record",
             ha="center", fontsize=10, color=INK, transform=ax2.transAxes)
    ax2.text(0.5, 0.46, f"{bs['p_top3']*100:.0f}%", ha="center", fontsize=26,
             fontweight="bold", color=DRY, transform=ax2.transAxes)
    ax2.text(0.5, 0.36, "chance of a top-three summer", ha="center", fontsize=9,
             color=INK, transform=ax2.transAxes)
    ax2.text(0.5, 0.20, f"median {p[1]:.2f} °C\n({bs['anomaly_vs_last10_c']:+.2f} °C "
             "vs the last ten)", ha="center", fontsize=9.5, color=INK,
             transform=ax2.transAxes)

    ax3 = fig.add_axes([0.055, 0.05, 0.915, 0.33]); ax3.set_axis_off()
    ax3.text(0, 0.99, "How to read this", fontsize=10.4, fontweight="bold",
             color=HY, va="top", transform=ax3.transAxes)
    y = 0.84
    for t in [
        f"\u201cOn record\u201d means within the ERA5 store held here — "
        f"{bs['observed_record_period']}, {len(yrs)} summers — not all time. The "
        f"standing record is {rec:.2f} °C in {bs['record_summer']}.",
        "Population weighting uses the 14 metros of the degree-day tracker, so a hot "
        "Amazon does not outvote São Paulo. It is metro-weighted rather than gridded "
        "population, so smaller cities are unrepresented.",
        "Members are bias-corrected as anomalies against the system's own 1993-2016 "
        "hindcast for the same init month and lead, then added to the ERA5 "
        "climatology — raw seasonal temperature is offset and drifts with lead.",
        "Caveats: a single system (ECMWF SEAS5), and seasonal ensembles are usually "
        "under-dispersed, which inflates a probability this far into a tail. The "
        "same EMOS diagnosis that applied to C3S rainfall — over-confident mean, "
        "uninformative spread — has not been repeated for temperature here, so treat "
        "92% as indicative rather than calibrated.",
        "It is conditioned on an El Niño already forecast to be the strongest on "
        "record, which is exactly the regime that drives Brazilian summer heat — and "
        "the same forcing behind the Colombian inflow outlook on page 1.",
    ]:
        y = para(ax3, 0.004, y, "\u2022 " + t, width=152, fs=8.4, dy=0.062)
        y -= 0.014
    pdf.savefig(fig); plt.close(fig)


def page_brazil_map(pdf, bs):
    """The gridded anomaly, so the pop-weighted number has a picture behind it."""
    png = PRIV / "site" / "brazil_summer_map.webp"
    if not png.exists():
        return
    import matplotlib.image as mpimg
    fig = plt.figure(figsize=(W, H))
    head(fig, f"BRAZIL — {bs.get('season_label','')} TEMPERATURE ANOMALY",
         "C3S SEAS5 ensemble mean against its own 1993-2016 hindcast · marker "
         "size = metro population")
    ax = fig.add_axes([0.13, 0.02, 0.74, 0.90]); ax.set_axis_off()
    ax.imshow(mpimg.imread(str(png)))
    foot(fig, "The population-weighted number on the previous page is this field "
              "sampled at the metros — the Amazon warms most, but few people live "
              "there.")
    pdf.savefig(fig); plt.close(fig)


def main() -> int:
    uo = load("unified_outlook.json"); ol = load("outlook_2027.json")
    gal = load("scatter_galaxy_h10.json"); gal1 = load("scatter_galaxy_h1.json")
    db = load("delta_backtest_long.json"); sc = load("scatter_delta_h10.json")
    si = load("seasonal_inflow.json"); cv = load("c3s_verify.json")
    bs = load("brazil_summer.json")
    if not (uo and gal and db and si):
        print("missing inputs — run the upstream scripts first"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "colombia_inflow_outlook.pdf"
    with PdfPages(pdf_path) as pdf:
        page_outlook(pdf, uo, ol)
        page_method(pdf)
        page_flow(pdf)
        page_steps(pdf)
        page_daily(pdf, gal, gal1, db)
        page_events(pdf, db, sc)
        page_seasonal(pdf, si, cv)
        page_limits(pdf, uo, gal)
        if uo.get("daily"):
            page_balmonth(pdf, uo)
        if bs:
            page_brazil(pdf, bs)
            page_brazil_map(pdf, bs)
        d = pdf.infodict()
        d["Title"] = "Colombia national inflow — outlook and validation"
        d["Author"] = "scorvec"
        d["CreationDate"] = datetime.now(timezone.utc)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    (OUT / f"colombia_inflow_outlook_{stamp}.pdf").write_bytes(pdf_path.read_bytes())
    print(f"wrote {pdf_path}  ({pdf_path.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
