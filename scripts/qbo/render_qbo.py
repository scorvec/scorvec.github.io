#!/usr/bin/env python3
"""Static QBO figures from qbo/qbo.json (IGRA radiosonde tropical mean u).

Replaces the page's interactive Plotly charts. Four figures, each answering a
question a line chart cannot:

  qbo_section.webp   The canonical time-height section. The QBO IS a descending
                     pattern, so the only honest primary view is height against
                     time with the zero-wind line drawn - you read the descent
                     rate straight off the slope.

  qbo_phase.webp     Where in the cycle we are. The standard QBO phase space:
                     the two leading EOFs of the level profile, which together
                     hold most of the variance, traced over the last three years
                     with the current month marked. A cycle is a loop; a
                     DISRUPTION is a loop that fails to close, and that is
                     visible here and almost nowhere else.

  qbo_descent.webp   How fast each shear zone is coming down, in km/month, with
                     the long-run average for scale. Stalls are what precede the
                     famous disruptions.

  qbo_cycles.webp    Every cycle since the record begins, aligned on the 30 hPa
                     westerly onset, with the current one drawn over the top -
                     so "unusually long" or "stalled" is something you can see
                     rather than something the text asserts.

    python scripts/qbo/render_qbo.py --out-dir assets/qbo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import NullFormatter, NullLocator

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "qbo" / "qbo.json"

# Easterly blue, westerly red - the convention in every QBO figure worth
# copying, and the opposite of a temperature colourmap, so it is worth being
# explicit rather than reaching for RdBu_r out of habit.
CMAP = "RdBu_r"
SECTION_YEARS = 14

# The QBO page is a dark design (--bg #0b0b12, --panel #11111c). Figures on a
# white ground would sit on it like pasted-in screenshots, so they are rendered
# in the page's own palette instead.
BG, PANEL, INK, MUTED, RULE, ACCENT = (
    "#0b0b12", "#11111c", "#e8e8f0", "#8b8ba3", "#23233a", "#64d2ff")
plt.rcParams.update({
    "figure.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.facecolor": BG, "axes.edgecolor": RULE, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": RULE, "legend.facecolor": PANEL, "legend.edgecolor": RULE,
    "font.size": 10,
})


def load():
    d = json.loads(SRC.read_text())
    lv = np.array(d["levels"], float)
    mo = pd.PeriodIndex(d["months"], freq="M")
    U = np.array([[np.nan if x is None else x for x in row] for row in d["u"]], float)
    order = np.argsort(-lv)                     # 100 hPa first (bottom of plot)
    return lv[order], mo, U[order], d


def fill_short_gaps(U, max_gap=3):
    """Interpolate gaps of at most `max_gap` months, per level.

    The radiosonde record has scattered holes - 10 hPa is missing a quarter of
    all months - and leaving them blank shreds the section into stripes that
    read as structure. Anything longer than a season stays NaN: that is a real
    hole in the observations and should look like one.
    """
    out = U.copy()
    for j in range(U.shape[0]):
        s = pd.Series(U[j])
        out[j] = s.interpolate(limit=max_gap, limit_area="inside").to_numpy()
    return out


def section(lv, mo, U, out_path):
    n = SECTION_YEARS * 12
    sl = slice(max(0, U.shape[1] - n), U.shape[1])
    Us, mos = U[:, sl], mo[sl]
    x = np.arange(Us.shape[1])
    fig, ax = plt.subplots(figsize=(13.2, 5.9), dpi=125)
    # Explicit margins, never tight_layout: the figure-level captions below are
    # not axes artists, so tight_layout lays out as if they were not there and
    # the title lands on top of the subtitle.
    fig.subplots_adjust(left=0.052, right=0.935, top=0.82, bottom=0.115)
    lim = np.nanpercentile(np.abs(Us), 99)
    lim = 5 * np.ceil(lim / 5)
    levels = np.arange(-lim, lim + 0.01, 2.5)
    cf = ax.contourf(x, lv, Us, levels=levels, cmap=CMAP, extend="both",
                     norm=TwoSlopeNorm(0, -lim, lim))
    # The zero line IS the QBO: its slope is the descent rate. Drawn BLACK, not
    # in the page's ink colour: u = 0 falls where RdBu_r is at its palest, so a
    # near-white line there is invisible however dark the rest of the figure is.
    ax.contour(x, lv, Us, levels=[0], colors="#000000", linewidths=1.7)
    # Missing months must not inherit the near-black axes colour - on a diverging
    # scale that reads as an extreme value rather than as absent data.
    ax.set_facecolor("#39394d")
    ax.set_yscale("log")
    ax.set_ylim(lv.max(), lv.min())
    ax.set_yticks(lv); ax.set_yticklabels([f"{int(v)}" for v in lv])
    # A log axis adds its own minor ticks - "4 x 10^1" printed over the 30 and
    # 50 hPa labels we actually want.
    ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("pressure (hPa)")
    ticks = [i for i, m in enumerate(mos) if m.month == 1]
    ax.set_xticks(ticks); ax.set_xticklabels([str(mos[i].year) for i in ticks])
    ax.set_xlim(0, len(mos) - 1)
    ax.grid(axis="x", color=RULE, alpha=0.6, lw=0.6)
    cb = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.028, extend="both")
    cb.set_label("zonal wind (m s$^{-1}$)   —  westerly +, easterly −", fontsize=9)
    fig.suptitle("Quasi-Biennial Oscillation — tropical mean zonal wind",
                 fontsize=14, fontweight="bold", x=0.052, ha="left", y=0.965)
    fig.text(0.052, 0.895, f"{mos[0]} to {mos[-1]} · IGRA radiosondes · black contour "
             f"u = 0, and its downward slope IS the QBO descent — a flat stretch is a "
             f"stalled shear zone", fontsize=9, color=MUTED, ha="left")
    fig.text(0.052, 0.028, "Flat grey is missing data — months with no usable "
             "sounding at that level. Only holes of up to three months are "
             "interpolated; anything longer stays a hole.",
             fontsize=8, color=MUTED, ha="left")
    fig.savefig(out_path, pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    return out_path


def eof_phase(lv, U):
    """Two leading EOFs of the standardised level profile, and the projections."""
    ok = np.isfinite(U).all(axis=0)
    X = U[:, ok]
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    Z = (X - mu) / sd
    C = np.cov(Z)
    w, v = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:2]
    E = v[:, idx]                                  # levels x 2
    pcs = E.T @ Z                                  # 2 x time
    var = w[idx] / w.sum()
    # Sign convention: PC1 positive when 30 hPa is westerly, so the diagram
    # reads the same way every time it is rebuilt.
    j30 = int(np.argmin(np.abs(lv - 30)))
    if E[j30, 0] < 0:
        E[:, 0] *= -1; pcs[0] *= -1
    if np.mean(np.diff(np.unwrap(np.arctan2(pcs[1], pcs[0])))) < 0:
        E[:, 1] *= -1; pcs[1] *= -1                # make the loop run anticlockwise
    return ok, pcs, var


def phase(lv, mo, U, out_path):
    ok, pcs, var = eof_phase(lv, U)
    mo_ok = mo[ok]
    fig, ax = plt.subplots(figsize=(8.4, 8.6), dpi=125)
    fig.subplots_adjust(left=0.095, right=0.855, top=0.845, bottom=0.135)
    ax.plot(pcs[0], pcs[1], color="#33334d", lw=0.5, zorder=1)
    n = 36
    seg = slice(len(mo_ok) - n, len(mo_ok))
    x, y = pcs[0][seg], pcs[1][seg]
    pts = ax.scatter(x, y, c=np.arange(n), cmap="viridis", s=26, zorder=3,
                     edgecolor=PANEL, linewidth=0.4)
    ax.plot(x, y, color="#7a7a96", lw=1.0, alpha=0.8, zorder=2)
    ax.scatter([x[-1]], [y[-1]], s=190, facecolor="#ff5a5f", edgecolor=INK,
               linewidth=1.4, zorder=5)
    ax.annotate(f"{mo_ok[-1]}", (x[-1], y[-1]), textcoords="offset points",
                xytext=(12, 8), fontsize=11, fontweight="bold", color="#ff5a5f")
    r = float(np.nanmax(np.hypot(pcs[0], pcs[1]))) * 1.05
    ax.add_artist(plt.Circle((0, 0), r * 0.55, fill=False, color=RULE, ls=":", lw=1.1))
    ax.axhline(0, color=RULE, lw=1.0); ax.axvline(0, color=RULE, lw=1.0)
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal")
    ax.set_xlabel(f"EOF 1  ({100*var[0]:.0f}% of variance)")
    ax.set_ylabel(f"EOF 2  ({100*var[1]:.0f}%)")
    fig.suptitle("QBO phase space — where the cycle is now",
                 fontsize=14, fontweight="bold", x=0.095, ha="left", y=0.965)
    fig.text(0.095, 0.905, f"grey: every month since {mo_ok[0]} · coloured: the last "
             f"{n} months, dark to bright · red: {mo_ok[-1]}",
             fontsize=9, color=MUTED, ha="left")
    fig.text(0.095, 0.062, "A full QBO cycle is one loop around the origin, and the "
             "radius is the strength of the oscillation.", fontsize=8.5, color=MUTED)
    fig.text(0.095, 0.032, "A loop that shrinks toward the centre without closing is a "
             "DISRUPTION — the QBO stalling mid-descent, as in 2016 and 2020.",
             fontsize=8.5, color=MUTED)
    cb = fig.colorbar(pts, ax=ax, fraction=0.035, pad=0.02)
    cb.set_ticks([0, n - 1]); cb.set_ticklabels([str(mo_ok[seg][0]), str(mo_ok[-1])])
    cb.ax.tick_params(labelsize=8)
    fig.savefig(out_path, pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    return out_path


def onset_paths(lv, U, kind="W"):
    """Descent paths of the wind-regime onsets, level by level.

    A FIRST attempt tracked "the" u = 0 height per month. That is not a
    well-defined quantity: the QBO carries two or three zero crossings at once,
    so scanning for one picks a different crossing from month to month and the
    result is a sawtooth between 10 and 100 hPa - noise presented as a descent
    rate. What IS well defined is when each LEVEL changes regime, and the QBO's
    defining behaviour is that the same onset arrives later at each level below.

    So: find every easterly->westerly (or westerly->easterly) crossing per
    level, then chain a crossing at one level to the next crossing BELOW it
    that follows within a plausible window. Each chain is one descending
    regime; its slope in km per month is the descent rate.
    """
    cross = []
    for j in range(len(lv)):
        u = U[j]
        ms = []
        for i in range(1, len(u)):
            a, b = u[i - 1], u[i]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if (kind == "W" and a < 0 <= b) or (kind == "E" and a > 0 >= b):
                ms.append(i)
        cross.append(ms)
    # lv is ordered 100 -> 10 hPa, so "below" is a LOWER index.
    paths = []
    used = set()
    for jt in range(len(lv) - 1, -1, -1):              # start at the top
        for m in cross[jt]:
            if (jt, m) in used:
                continue
            path = [(jt, m)]; used.add((jt, m)); jc, mc = jt, m
            while jc > 0:
                nxt = [x for x in cross[jc - 1] if 0 < x - mc <= 14 and (jc - 1, x) not in used]
                if not nxt:
                    break
                x = min(nxt)
                path.append((jc - 1, x)); used.add((jc - 1, x)); jc, mc = jc - 1, x
            if len(path) >= 3:
                paths.append(path)
    return paths


def descent(lv, mo, U, out_path):
    n = SECTION_YEARS * 12
    sl = slice(max(0, U.shape[1] - n), U.shape[1])
    Us, mos = U[:, sl], mo[sl]
    H = 7.0
    km = -H * np.log(lv / 1000.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.2, 7.0), dpi=125,
                                   gridspec_kw=dict(height_ratios=[2, 1], hspace=0.30))
    fig.subplots_adjust(left=0.062, right=0.975, top=0.845, bottom=0.135)
    rates = {"W": [], "E": []}
    for kind, col, lab in (("W", "#ff5a5f", "westerly onset"),
                           ("E", ACCENT, "easterly onset")):
        first = True
        for path in onset_paths(lv, Us, kind):
            xs = [p[1] for p in path]; ys = [lv[p[0]] for p in path]
            ax1.plot(xs, ys, color=col, lw=2.0, marker="o", ms=3.4,
                     label=lab if first else None, alpha=0.85)
            first = False
            # slope of a straight fit through the path, km per month
            dz = np.polyfit(xs, [km[p[0]] for p in path], 1)[0]
            rates[kind].append(-dz)
    ax1.set_yscale("log"); ax1.set_ylim(lv.max(), lv.min())
    ax1.set_yticks(lv); ax1.set_yticklabels([f"{int(v)}" for v in lv])
    ax1.yaxis.set_minor_locator(NullLocator()); ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_ylabel("pressure (hPa)")
    ticks = [i for i, m in enumerate(mos) if m.month == 1]
    ax1.set_xticks(ticks); ax1.set_xticklabels([str(mos[i].year) for i in ticks])
    ax1.set_xlim(0, len(mos) - 1); ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.95, ncol=2, labelcolor=INK)
    allr = [r for v in rates.values() for r in v if np.isfinite(r)]
    ax2.hist(allr, bins=np.arange(0, 2.01, 0.1), color="#3f6d8f", edgecolor=ACCENT)
    if allr:
        med = float(np.median(allr))
        ax2.axvline(med, color="#ff5a5f", lw=1.8)
        ax2.text(med, ax2.get_ylim()[1] * 0.92, f"  median {med:.2f} km/month",
                 color="#ff5a5f", fontsize=9, va="top")
    ax2.set_xlabel("descent rate of a single onset, km per month")
    ax2.set_ylabel("count")
    ax2.grid(alpha=0.25)
    fig.suptitle("How the QBO descends — every regime onset, tracked level by level",
                 fontsize=14, fontweight="bold", x=0.062, ha="left", y=0.965)
    fig.text(0.062, 0.895, "each line follows ONE onset as it arrives at each level "
             "below; the slope is the descent rate",
             fontsize=9, color=MUTED, ha="left")
    fig.text(0.062, 0.035, "A shallow line is a fast descent, a steep one is a slow "
             "one, and a line that stops before reaching 50-70 hPa is an onset that "
             "never made it down — the signature of a disrupted cycle.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.savefig(out_path, pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    return out_path


def cycles(lv, mo, U, out_path):
    """Every cycle aligned on the 30 hPa westerly onset, current one on top."""
    j = int(np.argmin(np.abs(lv - 30)))
    u30 = pd.Series(U[j]).interpolate(limit=3, limit_area="inside").to_numpy()
    onsets = [i for i in range(1, len(u30))
              if np.isfinite(u30[i]) and np.isfinite(u30[i - 1])
              and u30[i - 1] < 0 <= u30[i]]
    fig, ax = plt.subplots(figsize=(11.4, 6.1), dpi=125)
    fig.subplots_adjust(left=0.068, right=0.975, top=0.825, bottom=0.125)
    lens = []
    for a, b in zip(onsets, onsets[1:]):
        seg = u30[a:b]
        if not (18 <= len(seg) <= 42):
            continue
        lens.append(len(seg))
        ax.plot(np.arange(len(seg)), seg, color="#3d3d59", lw=1.0, zorder=1)
    cur = u30[onsets[-1]:]
    ax.plot(np.arange(len(cur)), cur, color="#ff5a5f", lw=2.6, zorder=4,
            label=f"current cycle — {mo[onsets[-1]]} onward, {len(cur)} months so far")
    ax.axhline(0, color=MUTED, lw=1.0)
    if lens:
        ax.axvline(np.median(lens), color=ACCENT, ls="--", lw=1.4,
                   label=f"median cycle length {np.median(lens):.0f} months "
                         f"({len(lens)} cycles)")
    ax.set_xlabel("months since the 30 hPa westerly onset")
    ax.set_ylabel("30 hPa zonal wind (m s$^{-1}$)")
    fig.suptitle("This cycle against every cycle in the record",
                 fontsize=14, fontweight="bold", x=0.068, ha="left", y=0.965)
    fig.text(0.068, 0.885, "each grey line is one QBO cycle, aligned on the month the "
             "30 hPa wind turned westerly", fontsize=9, color=MUTED, ha="left")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95, labelcolor=INK)
    ax.grid(alpha=0.25)
    fig.savefig(out_path, pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="assets/qbo")
    a = ap.parse_args()
    lv, mo, U, raw = load()
    Uf = fill_short_gaps(U)
    out = Path(a.out_dir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    for fn, f in (("qbo_section.webp", section), ("qbo_phase.webp", phase),
                  ("qbo_descent.webp", descent), ("qbo_cycles.webp", cycles)):
        p = f(lv, mo, Uf, out / fn)
        try: rel = p.relative_to(REPO)
        except ValueError: rel = p
        print(f"  wrote {rel}")
    print(f"  data through {mo[-1]}, updated {raw['updated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
