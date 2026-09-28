#!/usr/bin/env python3
"""The weather-regime forecast figure - one renderer for the single-model views (GEPS, GEFS) and the multi-model view.

Layout (2026-09-28 revamp, user review of the North America figure):
  top      stacked daily probabilities by regime family, "no regime" at the bottom in grey; the last 14 observed days
           run straight into the forecast as full-height columns of the observed regime; the previous run is a thin
           step outline of its cumulative boundaries; the season boundary, where the other set takes over, is a
           labelled line (every day is classified with its own season's set - regime_core)
  middle   (multi-model only) model agreement: each model's most likely regime per day, shaded by its probability
  bottom   left, the weekly heat-grid (regimes x weeks, full names, % in the cells; weeks whose hindcast skill is not
           significantly above the better of persistence and climatology are hatched and say so); right, the hindcast
           hit rate by lead with its 95% year-block bootstrap interval, persistence and climatology
Every panel is daily-resolution (step columns), nothing is interpolated between days.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

import regime_core as RC

TAIL = 14
SKILL_COLORS = {"geps": "#16335c", "gefs": "#0f7c7a"}
FONT = 10.5


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": FONT, "font.family": "DejaVu Sans"})
    return plt


def _ink_on(hexcol, alpha):
    """Dark or white text on a colour blended over white at `alpha`."""
    c = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    c = alpha * c + (1 - alpha) * 1.0
    lin = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "white" if 1.05 / (lum + 0.05) > (lum + 0.05) / 0.075 else RC.INK     # whichever contrasts more


def render(out: Path, ref: RC.Ref, sk: str, init, P, *, title: str, subtitle: str, obs=None, obs_label="observed",
           prev=None, prev_label=None, models=None, skills=None, footer: str = "", n_label: str = "% of members",
           dpi: int = 120):
    """P (F+1, N) family probabilities for forecast days 1..N (day d valid init + d - 1), NaN where missing.
    obs: (dates, family slots) observed tail; prev: (dates, P') previous run; models: [(label, Pm (F+1, N'))] for the
    agreement strip; skills: [hindcast_skill() dicts] (their sector entry is used)."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    init = RC.parse_date(init)
    F = len(RC.FAMILIES[sk])
    cols = ref.family_colors(sk)
    N = P.shape[1]
    days = [init + dt.timedelta(days=i) for i in range(N)]
    seasons_present = []
    for d in days:
        s = RC.season_of_date(d)
        if s not in seasons_present:
            seasons_present.append(s)
    boundary = RC.boundary_in(days)
    # names: the set that covers most forecast days; the other set's names are listed under the legend
    n_s = {s: sum(RC.season_of_date(d) == s for d in days) for s in seasons_present}
    main_s = max(n_s, key=n_s.get)
    minor_s = [s for s in seasons_present if s != main_s]
    names = [ref.family_name(sk, f, (main_s,)) if RC.FAMILIES[sk][f][main_s] is not None
             else ref.family_name(sk, f, seasons_present) for f in range(F)] + ["no regime"]
    minor_note = ""
    if minor_s:
        ms = minor_s[0]
        md = [d for d in days if RC.season_of_date(d) == ms]
        span = f"{md[0]:%b %-d}" + (f"–{md[-1]:%-d}" if len(md) > 1 and md[0].month == md[-1].month
                                   else (f"–{md[-1]:%b %-d}" if len(md) > 1 else ""))
        pairs = []
        for f in range(F):
            k = RC.FAMILIES[sk][f][ms]
            if k is None:
                continue
            n_minor = ref.name(sk, ms, k)
            if n_minor != names[f]:
                pairs.append(f"{names[f]} = {n_minor}" if RC.FAMILIES[sk][f][main_s] is not None else n_minor)
        if pairs:
            minor_note = f"{span}: {ms}-season set, where " + "; ".join(pairs)
    # a family absent from every set in play (e.g. the warm-only Hudson Bay ridge in a cold-season forecast) is left
    # out of the legend and the grid
    present = [f for f in range(F) if any(RC.FAMILIES[sk][f][s] is not None for s in seasons_present)] + [F]

    n_mod = len(models) if models else 0
    W = 12.0
    items = [f for f in present if f != F] + [F]
    per_row = 4 if max(len(names[f]) for f in items) <= 24 else 3
    n_leg_rows = int(np.ceil((len(items) + (1 if prev is not None and prev_label else 0)) / per_row))
    import textwrap
    subtitle = textwrap.fill(subtitle, 128)
    h_title, h_leg, h_main = 0.8 + 0.2 * subtitle.count("\n") + 0.15, 0.3 * n_leg_rows + (0.3 if minor_note else 0.0) + 0.3, 3.35
    h_agree = (0.34 + 0.27 * n_mod) if n_mod else 0.0
    h_bottom = 0.62 + 0.42 * len(present)
    h_foot = (0.25 + 0.155 * (len(footer) // 170 + 1)) if footer else 0.15
    H = h_title + h_leg + h_main + h_agree + (0.3 if n_mod else 0) + 0.5 + h_bottom + h_foot
    fig = plt.figure(figsize=(W, H))
    L_, R_ = 1.95 / W, 1 - 0.3 / W

    def yfrac(top_in, h_in):                                            # axes rect from inches below the top
        return 1 - (top_in + h_in) / H, h_in / H

    fig.text(0.5 / W, 1 - 0.18 / H, title, fontsize=15, fontweight="bold", va="top", ha="left", color=RC.INK)
    fig.text(0.5 / W, 1 - 0.58 / H, subtitle, fontsize=FONT, va="top", ha="left", color=RC.MUTED)

    # ── main panel ────────────────────────────────────────────────────────────────────────────────────────────────
    y0, hh = yfrac(h_title + h_leg, h_main)
    ax = fig.add_axes([L_, y0, R_ - L_, hh])
    x0 = -TAIL - 0.5 if obs is not None else -0.5
    ax.set_xlim(x0, N - 0.5)
    ax.set_ylim(0, 100)
    order = [F] + [f for f in range(F)]                                  # 'no regime' at the bottom
    edges = np.arange(N + 1) - 0.5
    cum = np.zeros(N)
    for f in order:
        v = np.nan_to_num(P[f]) * 100
        if f in present or v.any():
            ax.fill_between(edges, np.r_[cum, cum[-1]], np.r_[cum + v, (cum + v)[-1]], step="post", color=cols[f],
                            lw=0.0, alpha=0.95 if f != F else 1.0)
        cum = cum + v
    # 2 px surface gap between the stacked fills
    cum = np.zeros(N)
    for f in order[:-1]:
        cum = cum + np.nan_to_num(P[f]) * 100
        ax.step(edges, np.r_[cum, cum[-1]], where="post", color="white", lw=0.9)
    if obs is not None:
        od, ofam = obs
        for d, fam in zip(od, ofam):
            i = (RC.parse_date(d) - init).days
            if -TAIL <= i < 0 and fam >= 0:
                ax.add_patch(Rectangle((i - 0.5, 0), 1, 100, facecolor=cols[fam], edgecolor="white", lw=0.6))
        ax.text(-TAIL / 2 - 0.5, 102, obs_label, ha="center", va="bottom", fontsize=FONT - 0.5, color=RC.INK,
                fontweight="bold")
        ax.axvline(-0.5, color=RC.INK, lw=1.4)
    bi = (boundary - init).days - 0.5 if boundary is not None else None
    near = bi is not None and bi < 9
    ax.text(-0.2, 102, "forecast →" + (f"   {RC.season_of_date(boundary)}-season regimes from {boundary:%b %-d}"
                                       if near else ""), ha="left", va="bottom", fontsize=FONT - 0.5, color=RC.INK,
            fontweight="bold")
    if prev is not None:
        pdts, PP = prev
        pidx = np.array([(RC.parse_date(d) - init).days for d in pdts])
        keep = (pidx >= 0) & (pidx < N)
        if keep.any():
            pe = np.r_[pidx[keep] - 0.5, pidx[keep][-1] + 0.5]
            c2 = np.zeros(keep.sum())
            for j, f in enumerate(order[:-1]):
                c2 = c2 + np.nan_to_num(PP[f][keep]) * 100
                ax.step(pe, np.r_[c2, c2[-1]], where="post", color="#111", lw=0.9, alpha=0.85,
                        label=prev_label if j == 0 else None)
    for w0, _ in RC.WEEKS[1:]:
        if w0 - 1 < N:
            ax.axvline(w0 - 1.5, color="white", lw=2.2, zorder=3)
    for i, (w0, w1) in enumerate(RC.WEEKS):
        if w0 - 1 < N:
            ax.text((w0 - 1 + min(w1, N) - 1) / 2, 1.5, f"wk {i + 1}", ha="center", va="bottom", fontsize=FONT - 1.5,
                    color="#222", zorder=4, bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=1.2))
    if boundary is not None:
        bi = (boundary - init).days - 0.5
        ax.axvline(bi, color="#111", lw=1.3, ls=(0, (4, 2)), zorder=5)
        s_after = RC.season_of_date(boundary)
        if not near:
            ax.plot([bi, bi], [100, 106], color="#111", lw=1.3, ls=(0, (4, 2)), clip_on=False)
            ax.text(bi + 0.25, 102, f"{s_after}-season regimes from {boundary:%b %-d} →", ha="left", va="bottom",
                    fontsize=FONT - 0.5, color="#111", zorder=6, bbox=dict(facecolor="white", lw=0, pad=1.0))
    ticks = [i for i in range(int(np.ceil(x0)), N) if (init + dt.timedelta(days=i)).weekday() == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{init + dt.timedelta(days=i):%b %-d}" for i in ticks], fontsize=FONT - 0.5)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(n_label, fontsize=FONT)
    ax.tick_params(length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # legend of the families (and the previous-run outline), above the panel
    lx = L_
    ly = 1 - (h_title + 0.05) / H
    colw = (R_ - L_ - 0.02) / per_row
    if prev is not None and prev_label:
        j = len(items)
        r, c = divmod(j, per_row)
        xx, yy = lx + c * colw, ly - r * 0.3 / H
        fig.lines.append(plt.Line2D([xx, xx + 0.22 / W], [yy - 0.05 / H] * 2, transform=fig.transFigure, color="#111",
                                    lw=1.0))
        fig.text(xx + 0.3 / W, yy - 0.05 / H, prev_label + " (outline)", fontsize=FONT, va="center", color=RC.INK)
    for j, f in enumerate(items):
        r, c = divmod(j, per_row)
        xx, yy = lx + c * colw, ly - r * 0.3 / H
        fig.patches.append(Rectangle((xx, yy - 0.13 / H), 0.22 / W, 0.16 / H, transform=fig.transFigure,
                                     facecolor=cols[f], edgecolor="none"))
        fig.text(xx + 0.3 / W, yy - 0.05 / H, names[f], fontsize=FONT, va="center", color=RC.INK)
    if minor_note:
        fig.text(lx, ly - n_leg_rows * 0.3 / H - 0.03 / H, minor_note, fontsize=FONT - 1, va="center", color=RC.MUTED)

    # ── model agreement ───────────────────────────────────────────────────────────────────────────────────────────
    top = h_title + h_leg + h_main + 0.42
    if n_mod:
        y0, hh = yfrac(top, h_agree)
        axa = fig.add_axes([L_, y0, R_ - L_, hh])
        axa.set_xlim(x0, N - 0.5)
        axa.set_ylim(n_mod, -0.9)
        axa.text(x0, -0.55, "Model agreement: each model's most likely regime, shaded by its probability",
                 fontsize=FONT - 0.5, va="center", ha="left", color=RC.INK, fontweight="bold")
        for r, (lab, Pm) in enumerate(models):
            for i in range(min(N, Pm.shape[1])):
                col = Pm[:, i]
                if not np.isfinite(col).all():
                    continue
                f = int(np.argmax(col))
                a = float(np.clip(0.15 + 0.85 * (col[f] - 0.2) / 0.6, 0.15, 1.0))
                axa.add_patch(Rectangle((i - 0.5, r + 0.08), 1, 0.84, facecolor=cols[f], alpha=a, lw=0))
            axa.text(x0 - 0.3, r + 0.5, lab, ha="right", va="center", fontsize=FONT - 0.5, color=RC.INK)
        for w0, _ in RC.WEEKS[1:]:
            axa.axvline(w0 - 1.5, color="white", lw=2.2)
        if boundary is not None:
            axa.axvline((boundary - init).days - 0.5, color="#111", lw=1.1, ls=(0, (4, 2)))
        axa.axis("off")
        top += h_agree + 0.3

    # ── weekly heat-grid ──────────────────────────────────────────────────────────────────────────────────────────
    top += 0.1
    y0, hh = yfrac(top, h_bottom)
    axg = fig.add_axes([L_ + 0.35 / W, y0, 0.5 - L_ - 0.25 / W, hh])
    wk = RC.weekly(P)
    wk_models = [RC.weekly(Pm) for _, Pm in (models or [])]
    rows = [f for f in present if f != F] + [F]
    nW = len(wk)
    axg.set_xlim(0, nW)
    axg.set_ylim(len(rows) + 1.25, 0)
    axg.axis("off")
    sig = {}
    for s in skills or []:
        for w in s["sectors"][sk]["weeks"]:
            sig.setdefault(w["w0"], []).append(w["skilful"])
    for c, w in enumerate(wk):
        d0 = init + dt.timedelta(days=w["w0"] - 1)
        d1 = init + dt.timedelta(days=w["w0"] - 2 + w["ndays"])
        axg.text(c + 0.5, 0.42, f"Week {c + 1}", ha="center", va="bottom", fontsize=FONT, fontweight="bold",
                 color=RC.INK)
        axg.text(c + 0.5, 0.46, f"{d0:%b %-d}–{d1:%-d}" if d0.month == d1.month else f"{d0:%b %-d}–{d1:%b %-d}",
                 ha="center", va="top", fontsize=FONT - 1.5, color=RC.MUTED)
        ok = sig.get(w["w0"])
        skilful = ok is not None and any(ok)
        best = int(np.argmax(w["p"]))
        for r, f in enumerate(rows):
            p = w["p"][f]
            a = 0.06 + 0.9 * min(p / 0.6, 1.0)
            axg.add_patch(Rectangle((c + 0.03, r + 1.03), 0.94, 0.94, facecolor=cols[f], alpha=a, lw=0))
            if f == best:
                axg.add_patch(Rectangle((c + 0.03, r + 1.03), 0.94, 0.94, facecolor="none", edgecolor="#111", lw=1.6))
            if ok is not None and not skilful:
                axg.add_patch(Rectangle((c + 0.03, r + 1.03), 0.94, 0.94, facecolor="none", edgecolor="#8a8680",
                                        hatch="////", lw=0))
            ink = _ink_on(cols[f], a)
            rng = ""
            if wk_models:
                vals = [m[c]["p"][f] for m in wk_models if c < len(m) and m[c]["w0"] == w["w0"]]
                if len(vals) > 1:
                    rng = f"{min(vals) * 100:.0f}–{max(vals) * 100:.0f}"
            axg.text(c + 0.5, r + 1.5 - (0.12 if rng else 0), f"{p * 100:.0f}%", ha="center", va="center",
                     fontsize=FONT + 1, fontweight="bold", color=ink)
            if rng:
                axg.text(c + 0.5, r + 1.78, rng, ha="center", va="center", fontsize=FONT - 2.5, color=ink)
        foot = ("skill ✓" if skilful else "n.s.") if ok is not None else "no hindcast"
        if wk_models:
            tops = [int(np.argmax(m[c]["p"])) for m in wk_models if c < len(m) and m[c]["w0"] == w["w0"]]
            foot += f" · {sum(t == best for t in tops)}/{len(tops)} agree"
        axg.text(c + 0.5, len(rows) + 1.12, foot, ha="center", va="center", fontsize=FONT - 2,
                 color=RC.INK if skilful else RC.MUTED)
    for r, f in enumerate(rows):
        axg.text(-0.08, r + 1.5, names[f], ha="right", va="center", fontsize=FONT - 0.5, color=RC.INK, wrap=True)
    axg.text(-0.08, 0.5, "Weekly regime\nprobability" + ("\n(small: model range)" if wk_models else ""),
             ha="right", va="center", fontsize=FONT - 0.5, color=RC.INK, fontweight="bold")

    # ── hindcast skill strip ──────────────────────────────────────────────────────────────────────────────────────
    axs = fig.add_axes([0.605, y0 + 0.32 / H, R_ - 0.605, hh - 0.62 / H])
    if skills:
        base = None
        for s in skills:
            e = s["sectors"][sk]
            d_ = np.array(e["day"])
            col = SKILL_COLORS.get(s["model"], "#16335c")
            axs.fill_between(d_, np.array(e["hit_lo"]) * 100, np.array(e["hit_hi"]) * 100, color=col, alpha=0.16, lw=0)
            axs.plot(d_, np.array(e["hit"]) * 100, color=col, lw=2.0,
                     label=f"{s['label'].split(' ')[0]} (n = {s['n_starts']} starts)")
            if base is None:
                base = e
        d_ = np.array(base["day"])
        axs.plot(d_, np.array(base["persistence"]) * 100, color="#6f6b64", lw=1.4, ls="--", label="persistence")
        axs.plot(d_, np.array(base["clim"]) * 100, color="#b4453c", lw=1.3, ls=":", label="climatology (most common)")
        axs.set_xlim(1, max(len(base["day"]), 15))
        axs.set_ylim(0, 100)
        axs.set_xlabel("forecast day", fontsize=FONT - 0.5)
        axs.set_ylabel("regime correct (%)", fontsize=FONT - 0.5)
        axs.grid(alpha=0.25, lw=0.5)
        axs.legend(fontsize=FONT - 2.5, loc="upper right", frameon=True, framealpha=0.9)
        axs.set_title("Hindcast skill, starts within ±45 days of this date\nshaded: 95% interval (bootstrap by year)",
                      fontsize=FONT - 1, loc="left", color=RC.INK)
        for sp in ("top", "right"):
            axs.spines[sp].set_visible(False)
        axs.tick_params(labelsize=FONT - 1.5)
    else:
        axs.axis("off")
        axs.text(0, 0.5, "no hindcast for this model", fontsize=FONT, color=RC.MUTED)
    if footer:
        import textwrap
        body = "\n".join(textwrap.fill(par, 178) for par in footer.replace("\n", " ").split("  "))
        fig.text(0.5 / W, 0.08 / H, body, fontsize=FONT - 2.5, color=RC.MUTED, va="bottom", ha="left")
    fig.savefig(out, dpi=dpi, facecolor="white", pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    return out
