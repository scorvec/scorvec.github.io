#!/usr/bin/env python3
"""Render the warming-study items of the "Stratosphere history" group on stratosphere.html from the committed reference
scripts/strat/reference/ssw_precursors.json (built on the laptop by build_ssw_precursors.py). Runs in Actions
(ssw-precursors.yml): the static figures when the reference or this script change, the seasonal-odds card also on a
light schedule, because its inputs change monthly.

Outputs (assets/sst/):
  ssw_timing.webp     who moves first around a sudden warming: heat flux, vortex, polar cap, tropical upwelling/cooling
  ssw_qbo.webp        the QBO and the vortex (observed) and the QBO and sudden warmings (CMIP6 QBO models)
  ssw_enso.webp       El Nino and sudden warmings in CMIP6: frequency by Nino-3.4 and the month of the event
  ssw_odds.webp       seasonal-odds card: this winter's QBO and ENSO on the tested relations (model-based)
  data/ssw_odds.json  the card's inputs and numbers

Card inputs: QBO = KIT/FU Berlin monthly equatorial wind (https://www.atmohub.kit.edu/data/qbo.dat), falling back to the
site's IGRA Singapore series (qbo/qbo.json); ENSO = CPC's official ONI and RONI (assets/sst/data/cpc_official.json) and the
site's daily OISST Nino-3.4 (assets/sst/data/enso_daily.json).

    python scripts/strat/ssw_precursors.py [--out-root .] [--card-only]
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REF = HERE / "reference" / "ssw_precursors.json"
INK, MUTED, GRID, NS = "#1c2430", "#6f6b64", "#d9dde2", "#c4c7cc"
C = {"hf": "#eb6834", "u": "#2a78d6", "cap": "#e34948", "cool": "#4a3aa7", "tt": "#1baf7a",
     "e": "#eb6834", "w": "#2a78d6", "all": "#4a3aa7", "strong": "#b0402a", "en": "#eb6834", "neutral": "#9a978f", "ln": "#2a78d6"}
QBO_URL = "https://www.atmohub.kit.edu/data/qbo.dat"


def fig_header(fig, title, sub, x=0.045, y=None):
    W, H = fig.get_figwidth(), fig.get_figheight()
    y = 1 - 0.28 / H if y is None else y
    fig.text(x, y, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    lines = textwrap.wrap(sub, width=int((1 - 2 * x) * W * 12.6))
    fig.text(x, y - 0.36 / H, "\n".join(lines), ha="left", va="top", fontsize=9.6, color=MUTED, linespacing=1.35)
    return y - (0.36 + 0.19 * len(lines)) / H


def footer(fig, text):
    fig.text(0.045, 0.012, "\n".join(textwrap.wrap(text, width=int(0.91 * fig.get_figwidth() * 14.2))), ha="left", va="bottom",
             fontsize=8.2, color=MUTED, linespacing=1.3)


def style(ax):
    ax.tick_params(colors=INK, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa3ad")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="webp", pil_kwargs={"quality": 88})
    plt.close(fig)
    print(f"  {out.name}  {out.stat().st_size / 1e3:.0f} KB")


def pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p {p:.3f}" if p < 0.01 else f"p {p:.2f}"


# ---------------------------------------------------------------- 1. who moves first
def fig_timing(R, out):
    T = R["timing"]; lags = np.array(T["lags"])
    order = ["vT100", "u10", "Tcap10", "dTtr_e5", "Ttr_e5"]
    cols = [C["hf"], C["u"], C["cap"], C["cool"], C["tt"]]
    fig = plt.figure(figsize=(13.4, 9.2), dpi=125)
    top = fig_header(fig, "Who moves first around a sudden warming",
                     f"Composite of the {T['n_events']} major warmings on MERRA-2, 1981–2026, in standard deviations, each event measured from its own "
                     "level 60–90 days before. Colour where the composite differs from season-matched random dates (p < 0.05), grey where it does not; "
                     "the dashed line is the first day of a significant run of five or more. The planetary waves come first, the vortex breaks, and the "
                     "tropical lower stratosphere cools (upwelling speeds up) in step with the wave pulse, not before it.")
    gs = fig.add_gridspec(len(order), 1, left=0.07, right=0.975, top=top - 0.02, bottom=0.095, hspace=0.42)
    axs = []
    for i, (k, col) in enumerate(zip(order, cols)):
        s = T["series"][k]; c = np.array(s["comp"], float); p = np.array(s["p"], float)
        ax = fig.add_subplot(gs[i], sharex=axs[0] if axs else None); axs.append(ax)
        ax.axhline(0, color="#9aa3ad", lw=0.8); ax.axvline(0, color="#9aa3ad", lw=0.9, ls=":")
        ax.plot(lags, c, color=NS, lw=1.6)
        ax.plot(lags, np.where(p < 0.05, c, np.nan), color=col, lw=2.6)
        if s["onset"] is not None and s["onset"] >= lags[0]:
            ax.axvline(s["onset"], color=col, lw=1.1, ls="--")
            ax.annotate(f"onset day {s['onset']:+d}", (s["onset"], 0), xytext=(-5, -3), textcoords="offset points",
                        ha="right", va="top", fontsize=8.5, color=INK)
        ax.set_title(f"{s['label']} ({s['source']})", loc="left", fontsize=10, color=INK, pad=3)
        ax.set_ylabel("sd", fontsize=9, color=MUTED)
        style(ax)
        if i < len(order) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    axs[-1].set_xlabel("days from the central date (first easterly day at 60°N, 10 hPa)", fontsize=9.5, color=INK)
    axs[-1].set_xlim(lags[0], lags[-1])
    footer(fig, "Cooling rate = 10-day change of the 100–50 hPa layer temperature (negative = cooling = faster upwelling). Heat flux, wind and "
                "polar cap: MERRA-2 daily means (NASA GMAO); tropics: ERA5 heights (C3S). Static reference, built from the 2026 warming study.")
    save(fig, out)


# ---------------------------------------------------------------- 2. the QBO
def fig_qbo(R, out):
    Q = R["qbo"]
    fig = plt.figure(figsize=(13.4, 7.4), dpi=125)
    top = fig_header(fig, "The QBO and the polar vortex",
                     "Left: an easterly QBO in October–November goes with a weaker vortex that winter (observed, 1958–2026). Middle: the same "
                     "relation at 50 hPa, one dot per winter. Right: in five CMIP6 models with a realistic QBO, winters with an easterly "
                     "QBO at 30–70 hPa have more sudden warmings. Grey = not significant.")
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.985, top=top - 0.06, bottom=0.16, wspace=0.32, width_ratios=[1, 1.05, 1.25])
    ax = fig.add_subplot(gs[0])
    lv = Q["levels"]; y = np.arange(len(lv))
    for off, key, lab, col in ((-0.19, "nd", "Nov–Dec vortex", C["u"]), (0.19, "jf", "Jan–Feb vortex", C["all"])):
        v = [r[f"r_{key}"] for r in lv]; sig = [r[f"sig_{key}"] for r in lv]
        ax.barh(y + off, v, height=0.36, color=[col if s else NS for s in sig], label=lab)
        for yy, vv, s, r in zip(y + off, v, sig, lv):
            ax.text(vv + (0.015 if vv >= 0 else -0.015), yy, f"{vv:+.2f}" + ("" if s else " n.s."), va="center",
                    ha="left" if vv >= 0 else "right", fontsize=8, color=INK if s else MUTED)
    ax.set_yticks(y, [f"{r['lev']} hPa" for r in lv]); ax.invert_yaxis(); ax.set_xlim(-0.45, 0.72)
    ax.axvline(0, color="#9aa3ad", lw=0.8)
    ax.set_xlabel("correlation of the Oct–Nov equatorial wind\nwith u(60°N, 10 hPa)", fontsize=9)
    ax.set_title("Observed, 68 winters", loc="left", fontsize=10.5, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["u"], label="Nov–Dec vortex"), Patch(color=C["all"], label="Jan–Feb vortex"), Patch(color=NS, label="not significant")],
              frameon=False, fontsize=8.5, loc="lower right"); style(ax)
    ax = fig.add_subplot(gs[1])
    pts = Q["winters"]; x = np.array([p["q50"] for p in pts]); yv = np.array([p["u_nd"] for p in pts])
    ss = np.array([p["ssw"] for p in pts])
    ax.scatter(x[ss == 0], yv[ss == 0], s=22, color="#9aa3ad", label="no warming that winter", zorder=3)
    ax.scatter(x[ss == 1], yv[ss == 1], s=26, color=C["cap"], label="a warming that winter", zorder=3)
    rg = Q["regression_q50"]["u_nd"]; xx = np.linspace(-35, 20, 50)
    ax.plot(xx, rg["a"] + rg["b"] * xx, color=INK, lw=1.6)
    ax.axvline(0, color="#9aa3ad", lw=0.8)
    ax.set_xlabel("Oct–Nov equatorial wind at 50 hPa (m/s)\n← easterly QBO   westerly QBO →", fontsize=9)
    ax.set_ylabel("Nov–Dec mean u(60°N, 10 hPa), m/s", fontsize=9)
    ax.set_title(f"r = {rg['r']:+.2f} ({pfmt(rg['p'])}): +{rg['b']:.2f} m/s per m/s", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); style(ax)
    ax = fig.add_subplot(gs[2])
    ht = Q["cmip6"]; y = np.arange(len(ht))
    for off, key, lab, col in ((0.19, "e", "QBO easterly", C["e"]), (-0.19, "w", "QBO westerly", C["w"])):
        ax.barh(y + off, [h[key] for h in ht], height=0.36, color=[col if h["sig"] else NS for h in ht], label=lab)
    for i, h in enumerate(ht):
        ax.text(0.985, i, f"odds ratio {h['or']:.2f}{'' if h['sig'] else ' n.s.'}   {h['agree']}/{h['models']} models",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [f"{h['lev']} hPa" for h in ht]); ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of winters with a Dec–Mar sudden warming", fontsize=9)
    ax.set_title(f"CMIP6, QBO models, {Q['cmip6_winters']:,} winters", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, 1.0), ncol=2); style(ax)
    ax.set_ylim(len(ht) - 0.4, -1.1)
    footer(fig, "Observed: FU Berlin/KIT QBO; vortex = MERRA-2 from 1980, NCEP R1 before; FDR 10 % over the levels. The share of winters with a warming "
                "by QBO phase is not significant in the observations after that correction and is not drawn. CMIP6: UKESM1-0-LL, MRI-ESM2-0, "
                "CESM2-WACCM(-FV2), HadGEM3-GC31-LL; Cochran–Mantel–Haenszel test, FDR 10 %.")
    save(fig, out)


# ---------------------------------------------------------------- 3. ENSO
def fig_enso(R, out):
    N = R["enso"]; bins = N["bins"]; x = np.arange(len(bins)); T = N["tests"]
    fig = plt.figure(figsize=(13.4, 6.6), dpi=125)
    top = fig_header(fig, "El Niño and sudden warmings in climate models",
                     f"{N['n_winters']:,} winters and {N['n_events']:,} December–March warmings in 9 stratosphere-resolving CMIP6 models. Stronger "
                     f"El Niños bring more warmings ({T['strong_vs_neutral']['a']:.0%} of strong El Niño winters against {T['strong_vs_neutral']['b']:.0%} "
                     f"of neutral ones, {pfmt(T['strong_vs_neutral']['p'])}, all {T['strong_vs_neutral']['models']} models agree) and push them later "
                     f"into the winter. The observed winters since 1957 are too few to test this: their points are drawn for reference and are not significant.")
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.985, top=top - 0.09, bottom=0.2, wspace=0.2, width_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0])
    for name, lab, col, off in (("all", "all 9 models", C["all"], -0.15), ("qbo_w", "QBO models, westerly QBO", C["w"], 0.0),
                                ("qbo_e", "QBO models, easterly QBO", C["e"], 0.15)):
        rows = N["dose"][name]
        v = np.array([r["share"] if r["share"] is not None else np.nan for r in rows], float)
        lo = np.array([r["ci"][0] if r["ci"] else np.nan for r in rows], float); hi = np.array([r["ci"][1] if r["ci"] else np.nan for r in rows], float)
        ax.errorbar(x + off, v, yerr=[v - lo, hi - v], fmt="o-", color=col, ms=4.5, lw=1.5, capsize=0, label=lab)
    ob = N["obs_bins"]
    ov = np.array([r["share"] if r["share"] is not None else np.nan for r in ob], float)
    ax.scatter(x, ov, marker="s", s=[max(r["n"], 1) * 7 for r in ob], facecolor="none", edgecolor=MUTED, lw=1.1, zorder=4,
               label="observed 1957–2025 (not significant; size = winters)")
    ax.set_xticks(x, bins, fontsize=8.5); ax.set_ylim(0, 0.95)
    ax.set_xlabel("December–February Niño-3.4 (K; model values rescaled to the observed spread)", fontsize=9)
    ax.set_ylabel("share of winters with a sudden warming", fontsize=9)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left"); style(ax)
    ax = fig.add_subplot(gs[1])
    for i, (cls, lab) in enumerate((("ln", "La Niña"), ("neutral", "neutral"), ("en", "El Niño"), ("strong", "strong El Niño"))):
        v = [N["months"][cls][m] for m in ("12", "1", "2", "3")]
        ax.bar(np.arange(4) + (i - 1.5) * 0.2, v, width=0.19, color=C[cls], label=f"{lab} ({N['months_n'][cls]:,} events)")
    ax.set_xticks(range(4), ["Dec", "Jan", "Feb", "Mar"]); ax.set_ylabel("share of the class's events", fontsize=9)
    lt = T["late_en_vs_neutral"]
    ax.set_title(f"Month of the warming: Feb–Mar share El Niño {lt['a']:.0%}\nvs neutral {lt['b']:.0%} ({pfmt(lt['p'])}, {lt['agree']} of 9 models)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); style(ax)
    footer(fig, "CMIP6 piControl, historical, SSP and AMIP runs; each run's Niño-3.4 against its own 31-year running climatology. Test: "
                "Cochran–Mantel–Haenszel across model × experiment strata. Bars: 95 % bootstrap intervals. Observed: CPC ONI; Fisher p 0.14–0.25.")
    save(fig, out)


# ---------------------------------------------------------------- 4. seasonal-odds card
def qbo_current(repo: Path):
    """(value m/s at 50 hPa, label, source). Oct-Nov mean of the current winter if both months exist, else the latest month."""
    rows = {}
    try:
        txt = urllib.request.urlopen(QBO_URL, timeout=40).read().decode("latin-1")
        for ln in txt.splitlines():
            m = re.match(r"^\s*(\d{5})\s+(\d{2})(\d{2})\s+(.*)$", ln)
            if not m:
                continue
            yy, mm = int(m.group(2)), int(m.group(3)); year = 1900 + yy if yy >= 53 else 2000 + yy
            toks = [t for t in m.group(4).split() if not re.fullmatch(r"\d", t)]
            if len(toks) >= 2:
                rows[pd.Timestamp(year, mm, 1)] = float(toks[1]) / 10.0            # 70, 50, ... -> 50 hPa
        src = "KIT/FU Berlin"
    except Exception as e:                                                        # noqa: BLE001
        print("KIT QBO unavailable:", str(e)[:80])
    if not rows:
        j = json.loads((repo / "qbo" / "qbo.json").read_text())
        k = j["levels"].index(50)
        st = list(j["stations"]).index("48698") if isinstance(j["u"][0][0], list) else None
        for i, mth in enumerate(j["months"]):
            try:
                v = j["u"][k][i] if st is None else j["u"][st][k][i]
            except Exception:                                                     # noqa: BLE001
                v = None
            if v is not None:
                rows[pd.Timestamp(mth + "-01")] = float(v)
        src = "IGRA Singapore (site QBO tracker)"
    s = pd.Series(rows).sort_index()
    last = s.index[-1]
    wy = last.year if last.month >= 7 else last.year - 1
    on = s[(s.index >= f"{wy}-10-01") & (s.index <= f"{wy}-11-30")]
    if len(on) == 2:
        return float(on.mean()), f"Oct–Nov {wy} mean", src, wy
    return float(s.iloc[-1]), f"{last:%B %Y}", src, (last.year if last.month >= 7 else last.year - 1)


def enso_current(repo: Path):
    j = json.loads((repo / "assets" / "sst" / "data" / "cpc_official.json").read_text())
    oni = {k: v for k, v in j["oni"].items() if v is not None}; roni = {k: v for k, v in j["roni"].items() if v is not None}
    km = max(oni); kr = max(roni)
    seas = "DJF JFM FMA MAM AMJ MJJ JJA JAS ASO SON OND NDJ".split()
    lab = lambda k: f"{seas[int(k[5:7]) - 1]} {k[:4]}"
    d = {}
    try:
        e = json.loads((repo / "assets" / "sst" / "data" / "enso_daily.json").read_text())["latest"]
        d = {"nino34": e.get("nino34"), "date": e.get("date")}
    except Exception:                                                             # noqa: BLE001
        pass
    return {"oni": float(oni[km]), "oni_season": lab(km), "roni": float(roni[kr]), "roni_season": lab(kr), **d}


def card(R, out, out_json, repo):
    O = R["odds"]; Q = R["qbo"]
    q, qlab, qsrc, wy = qbo_current(repo)
    en = enso_current(repo)
    ph = "W" if q >= 0 else "E"
    edges = O["edges"]; k = int(np.searchsorted(edges, en["oni"], side="right") - 1); k = min(max(k, 0), len(O["bins"]) - 1)
    cell = O["lookup"][ph][k]; base = O["base"]
    rows = [("All winters (models)", base["share"], None, base["n"], base["p_decjan"], base["p_febmar"], C["neutral"])]
    rows.append((f"QBO {'westerly' if ph == 'W' else 'easterly'}, Niño-3.4 {O['bins'][k]} K", cell["share"], cell["ci"], cell["n"],
                 cell["p_decjan"], cell["p_febmar"], C["strong"]))
    if k + 1 < len(O["bins"]):
        c2 = O["lookup"][ph][k + 1]
        rows.append((f"… if El Niño peaks one class stronger ({O['bins'][k + 1]} K)" if en["oni"] > 0.5 else
                     f"… one class warmer ({O['bins'][k + 1]} K)", c2["share"], c2["ci"], c2["n"], c2["p_decjan"], c2["p_febmar"], "#d98a7c"))
    rg = Q["regression_q50"]
    pred = {}
    for key in ("u_nd", "u_jf"):
        g = rg[key]; yhat = g["a"] + g["b"] * q
        se = g["resid_sd"] * np.sqrt(1 + 1 / g["n"] + (q - g["x_mean"]) ** 2 / g["sxx"])
        pred[key] = {"pred": round(yhat, 1), "lo": round(yhat - 1.67 * se, 1), "hi": round(yhat + 1.67 * se, 1), "clim": g["clim"], "r": g["r"], "p": g["p"]}
    fig = plt.figure(figsize=(13.4, 7.8), dpi=125)
    top = fig_header(fig, f"Seasonal odds for winter {wy}/{str(wy + 1)[2:]}: what the QBO and ENSO say",
                     f"Model-based. The QBO at 50 hPa is {q:+.1f} m/s ({qlab}, {qsrc}) and ENSO's latest ONI is {en['oni']:+.1f} °C ({en['oni_season']}; "
                     f"RONI {en['roni']:+.1f}). Left: the share of CMIP6 winters with a sudden warming in that state, and when in the winter. "
                     "Middle: the vortex the observed QBO relation expects. Right: the observed winters that looked like this, for reference, not a test.")
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.985, top=top - 0.07, bottom=0.14, wspace=0.16, width_ratios=[1.25, 0.85, 1.1])
    ax = fig.add_subplot(gs[0])
    for i, (lab, v, ci, n, dj, fm, col) in enumerate(rows):
        yb = i * 1.5
        ax.text(0.0, yb - 0.5, lab, fontsize=9.2, color=INK, va="center", fontweight="bold" if i == 1 else "normal")
        ax.barh(yb, v, color=col, height=0.5)
        if ci:
            ax.plot(ci, [yb, yb], color=INK, lw=1.3)
        ax.text((ci[1] if ci else v) + 0.015, yb, f"{v:.0%}", va="center", fontsize=10, color=INK, fontweight="bold")
        ax.text(0.0, yb + 0.48, f"n {n:,} winters · a warming in Dec–Jan {dj:.0%}, in Feb–Mar {fm:.0%}", fontsize=8.2, color=MUTED, va="center")
    ax.axvline(O["obs_base"], color=INK, ls=":", lw=1.1)
    ax.text(O["obs_base"] + 0.01, (len(rows) - 1) * 1.5 + 0.95, f"observed, all winters {O['obs_base']:.0%}", fontsize=8, color=INK, va="center")
    ax.set_yticks([]); ax.set_ylim((len(rows) - 1) * 1.5 + 1.2, -1.0); ax.set_xlim(0, 1)
    ax.set_xlabel("share of winters with a Dec–Mar sudden warming (95 % interval)", fontsize=9)
    ax.set_title("CMIP6, 5 models with a realistic QBO", loc="left", fontsize=10.5, color=INK); style(ax); ax.spines["left"].set_visible(False)
    ax = fig.add_subplot(gs[1])
    labs = ["Nov–Dec", "Jan–Feb"]
    for i, key in enumerate(("u_nd", "u_jf")):
        p = pred[key]
        ax.plot([p["lo"], p["hi"]], [i, i], color=C["u"], lw=7, alpha=0.35, solid_capstyle="butt")
        ax.plot(p["pred"], i, "o", color=C["u"], ms=8)
        ax.plot(p["clim"], i, "|", color=INK, ms=20, mew=2)
        ax.text(p["pred"], i - 0.3, f"{p['pred']:.0f} m/s (normal {p['clim']:.0f}; r {p['r']:+.2f})", ha="center", fontsize=8.5, color=INK)
    ax.set_yticks([0, 1], labs); ax.set_ylim(1.6, -0.7); ax.set_xlim(0, 60)
    ax.set_xlabel("u(60°N, 10 hPa) from the QBO alone (m/s)\nblue: 90 % range; black tick: normal", fontsize=9)
    ax.set_title("Observed QBO → vortex", loc="left", fontsize=10.5, color=INK); style(ax)
    ax = fig.add_subplot(gs[2]); ax.axis("off")
    ax.set_title("Observed strong El Niño winters (ONI ≥ 1.5)", loc="left", fontsize=10.5, color=INK)
    yy = 0.95
    mono = dict(fontsize=8.4, family="DejaVu Sans Mono", transform=ax.transAxes)
    ax.text(0, yy, "winter   ONI   QBO50  QBO30  warming", color=MUTED, **mono)
    for a in O["analogs"]:
        yy -= 0.075
        ssw = ", ".join(pd.Timestamp(d).strftime("%-d %b") for d in a["ssw"]) if a["ssw"] else "none"
        like = (a["q50"] >= 0) == (q >= 0)
        ax.text(0, yy, f"{a['winter']:8s} {a['oni']:+4.1f}  {a['q50']:+5.1f}  {a['q30']:+5.1f}  {ssw}", color=INK if like else MUTED, **mono)
    n_like = sum(1 for a in O["analogs"] if (a["q50"] >= 0) == (q >= 0)); k_like = sum(1 for a in O["analogs"] if (a["q50"] >= 0) == (q >= 0) and a["ssw"])
    ax.text(0, yy - 0.1, textwrap.fill(f"Dark rows share this winter's QBO phase at 50 hPa (Oct–Nov): {k_like} of {n_like} had a warming. "
                                       "A handful of winters cannot test a rate; the model shares at left are the tested numbers.", 55),
            fontsize=8.5, color=MUTED, transform=ax.transAxes, va="top")
    footer(fig, "Models: UKESM1-0-LL, MRI-ESM2-0, CESM2-WACCM(-FV2), HadGEM3-GC31-LL; QBO = Oct–Nov wind at 50 hPa; Niño-3.4 = DJF, rescaled to the observed "
                "spread. The El Niño and QBO effects are significant in the models (p < 1e-10); the observed record alone is too short to confirm the El Niño one.")
    save(fig, out)
    d = {"winter": f"{wy}/{str(wy + 1)[2:]}",
         "qbo50": round(q, 1), "qbo_label": qlab, "qbo_source": qsrc, "enso": en, "enso_bin": O["bins"][k], "qbo_phase": ph,
         "cmip6_share": cell["share"], "cmip6_ci": cell["ci"], "cmip6_n": cell["n"], "cmip6_decjan": cell["p_decjan"], "cmip6_febmar": cell["p_febmar"],
         "cmip6_base": base["share"], "obs_base": O["obs_base"], "vortex_from_qbo": pred}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(d, indent=1))
    print("  ssw_odds.json", d["cmip6_share"], d["enso_bin"], d["qbo_phase"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(REPO)); ap.add_argument("--card-only", action="store_true")
    a = ap.parse_args()
    root = Path(a.out_root); R = json.loads(REF.read_text())
    A = root / "assets" / "sst"
    if not a.card_only:
        fig_timing(R, A / "ssw_timing.webp")
        fig_qbo(R, A / "ssw_qbo.webp")
        fig_enso(R, A / "ssw_enso.webp")
    card(R, A / "ssw_odds.webp", A / "data" / "ssw_odds.json", REPO)


if __name__ == "__main__":
    main()
