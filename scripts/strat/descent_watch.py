#!/usr/bin/env python3
"""Descent watch: is the stratosphere's weak-vortex anomaly reaching the troposphere in the days 1-15 ensembles? Three
measures, each from fields another product already holds (nothing is fetched):

  reach  per member and day, how far down the 65-90N cap height stands at THRESHOLD standard deviations or more above
         its normal at every level of CHAIN from 100 hPa, on that day: one unbroken column. The share of members whose
         column reaches each level is set against how often MERRA-2 shows such a column on that calendar day
         (reference/descent_base.json, build_descent_base.py): a one-sided binomial test of the member count, the days
         and levels of a run held to a false discovery rate of ALPHA. forecast_drip.py calls reach() with its
         drift-corrected, standardised heights; a level counts only when the model's drift is measured at it and at
         every level above it, so a model that drifts in the troposphere is not read as a descent. November to April.
  ring   whether the cap's anomaly at 500 hPa and at sea level is the whole ring or one sector: the cap split into
         SECTORS, the ensemble mean coloured where enough members share its sign (polarcap.agreed), and the shift of
         the westerlies that goes with a descent, the geostrophic zonal wind anomaly over 55-65N minus 35-45N: at 500
         hPa from the zonal-mean height, at sea level from the pressure over low ground. polarcap.py calls ring() with
         each member's anomalies (raw model output against ERA5 1991-2020).
  link   inside one ensemble, do the members with the weaker lower-stratospheric vortex on days 6-10 have the higher
         Arctic heights and pressure on days 11-15? The partial correlation across members of the days 11-15 cap (500
         hPa height, sea-level pressure) with the days 6-10 cap height at 100 hPa, the member's own days 6-10 cap at
         500 hPa and at sea level regressed out of both, so what its troposphere already held is not counted. A
         Freedman-Lane permutation test (quantmet.stats.partial_corr_test), the figure's panels held to a false
         discovery rate of ALPHA; a panel that fails says so and shows no line and no number. ring() files each
         member's five numbers, combined() tests and draws every ensemble's newest run and the ensembles pooled (each
         one's own means removed).

Outputs, model m:
  assets/sst/dwatch_reach_{m}.webp              the share of members by day and level against the base rate
  assets/sst/dwatch_ring_{m}.webp               the sectors and the westerlies' shift
  assets/sst/data/dwatch_{m}.json               the days 11-15 sectors and the members' link values
  assets/sst/data/runs/{reach,ring}_{m}.json    the run records (run_record.py draws the run-to-run figures)
  --combined: assets/sst/dwatch_link.webp, assets/sst/data/dwatch.json (strat-combine.yml)

    python descent_watch.py --combined
"""

from __future__ import annotations
import sys

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from quantmet.stats import benjamini_hochberg, partial_corr_test, residualize

import run_record as RR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

HERE = Path(__file__).resolve().parent
OUT = RR.REPO / "assets" / "sst"
BASE = HERE / "reference" / "descent_base.json"
ENSEMBLES = ("aifs", "ifs", "gefs", "geps")
CHAIN = (100.0, 200.0, 250.0, 300.0, 500.0, 700.0, 850.0)  # hPa, from the lower stratosphere down
DEPTHS = (300.0, 500.0, 850.0)  # the levels of CHAIN the record and the lines follow
THRESHOLD = 0.5  # sd of the day's normal
ALPHA = 0.05
EARLY, LATE = (6, 10), (11, 15)  # days: the stratosphere's window and the troposphere's
SECTORS = (  # east longitudes from, to
    (0.0, 60.0, "Barents–Kara, 0–60°E"),
    (60.0, 120.0, "Taymyr–Laptev, 60–120°E"),
    (120.0, 180.0, "East Siberian, 120–180°E"),
    (180.0, 240.0, "Chukchi–Beaufort, 180–120°W"),
    (240.0, 300.0, "Canadian Arctic, 120–60°W"),
    (300.0, 360.0, "Greenland–Norwegian, 60–0°W"),
)
JET = ((55.0, 65.0), (35.0, 45.0))  # deg N: the westerlies' index is the first band minus the second
G, OMEGA, RE = 9.80665, 7.2921e-5, 6.371e6
RHO = 1.25  # kg m-3: air at sea level
DEPTH_COLOUR = {300.0: "#3f6f8f", 500.0: "#b4541f", 850.0: "#1c1c1a"}
NEUTRAL, MUTED, FAINT = "#d9d6d0", "#6f6b64", "#b9b5ad"
RING_PANELS = (("z500", "500 hPa height", "m", 160.0, "{:+.0f}"), ("msl", "sea-level pressure", "hPa", 16.0, "{:+.1f}"))
WIND_PANELS = (
    ("u500", "500 hPa, from the zonal-mean height"),
    ("usfc", "sea level, from the pressure over low ground"),
)
LINK_TARGETS = (("y500", "500 hPa height", "m"), ("ymsl", "sea-level pressure", "hPa"))
LINK_CONTROLS = ("c500", "cmsl")


def _minus(s: str) -> str:
    return s.replace("-", "−")


def _days(valid: pd.DatetimeIndex, init: pd.Timestamp, window: tuple) -> np.ndarray:
    day = (valid - init) / pd.Timedelta(days=1)
    return (day >= window[0]) & (day <= window[1])


# ── reach: the unbroken column, member by member ────────────────────────────────────────────────────────────────
def connected(z: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """z (..., level) down CHAIN -> True at a level when every level from the first to it is at the threshold or more."""
    return np.logical_and.accumulate(z >= threshold, axis=-1)


def base_rate(valid: pd.DatetimeIndex) -> np.ndarray:
    """MERRA-2's rate of a column down to each level of CHAIN on each valid day, (day, level); NaN outside November to
    April."""
    d = json.loads(BASE.read_text())
    if tuple(d["chain_hpa"]) != CHAIN or d["threshold_sd"] != THRESHOLD:
        raise SystemExit(f"{BASE.name} was built for another chain or threshold: rebuild it (build_descent_base.py)")
    return np.array([d["rate"].get(f"{t:%m-%d}", [np.nan] * len(CHAIN)) for t in valid], float)


def reach(model: str, init: pd.Timestamp, levels, Z: np.ndarray, valid, corrected, credit: str, out: Path) -> None:
    """Record and draw the run's reach from Z (member, day, level), the cap heights in standard deviations at `levels`
    (hPa) and the valid times; `corrected` holds the levels whose drift is measured."""
    from scipy.stats import binom

    lab = RR.MODEL_LABEL[model]
    levels = [float(L) for L in levels]
    deep = next((k for k, L in enumerate(CHAIN) if L not in levels or L not in corrected), len(CHAIN))
    depths = [L for L in DEPTHS if CHAIN.index(L) < deep]
    if not depths:
        print(f"  {lab}: no measured drift at every level from 100 to {DEPTHS[0]:.0f} hPa: no reach", flush=True)
        return
    valid = pd.DatetimeIndex(valid)
    base = base_rate(valid)[:, :deep]
    season = np.isfinite(base[:, 0])
    z = Z[:, season][:, :, [levels.index(L) for L in CHAIN[:deep]]]
    z = z[np.isfinite(z).all((1, 2))]
    if not season.any() or z.shape[0] < 2:
        print(f"  {lab}: no reach (outside November to April, or fewer than two whole members)", flush=True)
        return
    valid, base, n = valid[season], base[season], z.shape[0]
    hit = connected(z)  # (member, day, level)
    count = hit.sum(0)
    more = benjamini_hochberg(binom.sf(count - 1, n, base), ALPHA)  # P(this many members or more | the base rate)
    share = 100.0 * count / n
    key = {L: f"r{L:.0f}" for L in depths}
    col = {L: CHAIN.index(L) for L in depths}
    scalars = {
        "base": {key[L]: np.round(100.0 * base[:, col[L]], 1).tolist() for L in depths},
        "more": {key[L]: more[:, col[L]].astype(int).tolist() for L in depths},
    }
    late = _days(valid, init, LATE)
    if 500.0 in depths and late.any():
        scalars["late"] = round(float(hit[:, late, col[500.0]].any(1).mean()), 3)
    frames = {key[L]: pd.DataFrame(100.0 * hit[:, :, col[L]].T, index=valid) for L in depths}
    RR.record("reach", model, init, RR.entry("reach", frames, scalars))
    RR.render("reach", model)
    print(f"  {lab} reach, {n} members: " + "; ".join(
        f"{L:.0f} hPa days {LATE[0]}-{LATE[1]} {share[late, col[L]].mean():.0f} % (usual "
        f"{100 * base[late, col[L]].mean():.0f} %, more than usual on {more[late, col[L]].sum()} of {late.sum()} days)"
        for L in depths if late.any()), flush=True)  # fmt: skip
    _reach_figure(model, init, valid, share, more, base, n, depths, credit, out)


def _reach_figure(model, init, valid, share, more, base, n, depths, credit, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import house

    house.apply()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    nd, deep = share.shape
    x = np.arange(nd)
    dates = [f"{t:%-d %b}" for t in valid]
    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=(1.25, 1.0), hspace=0.3)
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("reach", ["#f6f3ee", "#ecc3a4", "#cf7a45", "#8f3d12", "#5a2408"])
    ax.imshow(share.T, aspect="auto", cmap=cmap, vmin=0, vmax=100, interpolation="nearest",
              extent=(-0.5, nd - 0.5, deep - 0.5, -0.5))  # fmt: skip
    for i in range(nd):
        for j in range(deep):
            ax.text(i, j, f"{share[i, j]:.0f}", ha="center", va="center", fontsize=9.5 if more[i, j] else 8.5,
                    fontweight="bold" if more[i, j] else "normal",
                    color="white" if share[i, j] >= 55 else (house.INK if more[i, j] else MUTED))  # fmt: skip
    for v in np.arange(-0.5, nd, 1):
        ax.axvline(v, color="white", lw=0.8)
    for v in np.arange(-0.5, deep, 1):
        ax.axhline(v, color="white", lw=0.8)
    ax.set_yticks(range(deep))
    ax.set_yticklabels([f"{L:.0f} hPa · usual {100 * base[:, j].mean():.0f} %" for j, L in enumerate(CHAIN[:deep])],
                       fontsize=9)  # fmt: skip
    ax.set_xticks(x)
    ax.set_xticklabels(dates, fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"% of the {n} members whose anomaly runs unbroken from 100 hPa down to each level, day by day",
                 fontsize=11, fontweight="bold", loc="left")  # fmt: skip
    ax2 = fig.add_subplot(gs[1])
    at, last = {}, -np.inf  # the lines' end labels, kept a line of text apart
    for L in sorted(depths, key=lambda L: share[-1, CHAIN.index(L)]):
        at[L] = last = max(share[-1, CHAIN.index(L)], last + 6.5)
    for L in depths:
        j, c = CHAIN.index(L), DEPTH_COLOUR[L]
        ax2.plot(x, 100 * base[:, j], color=c, lw=1.1, ls=(0, (4, 3)), zorder=3)
        ax2.plot(x, share[:, j], color=c, lw=2.2, zorder=5)
        ax2.scatter(x, share[:, j], s=44, zorder=6, edgecolors=c, linewidths=1.6,
                    facecolors=[c if m else "white" for m in more[:, j]])  # fmt: skip
        ax2.annotate(f"{L:.0f} hPa", (x[-1], at[L]), xytext=(9, 0), textcoords="offset points", va="center",
                     fontsize=9.5, fontweight="bold", color=c, annotation_clip=False)  # fmt: skip
    ax2.set_xlim(-0.5, nd - 0.5)
    ax2.set_ylim(-3, 103)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dates, fontsize=8.5)
    ax2.set_ylabel("% of members", fontsize=9.5)
    ax2.grid(alpha=0.22)
    ax2.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.legend(
        handles=[
            Line2D([], [], color=house.INK2, lw=2.2, marker="o", ms=6.5, mfc=house.INK2, label="more members than usual"),
            Line2D([], [], color=house.INK2, lw=2.2, marker="o", ms=6.5, mfc="white", label="no more than usual"),
            Line2D([], [], color=house.INK2, lw=1.1, ls=(0, (4, 3)), label="usual for the date (MERRA-2)"),
        ],
        fontsize=8.5, loc="upper left", ncol=3, framealpha=0.9,
    )  # fmt: skip
    ax2.set_title("Members connected from 100 hPa down to 300, 500 and 850 hPa", fontsize=11, fontweight="bold",
                  loc="left")  # fmt: skip
    fig.suptitle(f"{RR.MODEL_LABEL[model]} · descent watch: how far down the weak-vortex anomaly reaches · run "
                 f"{init:%Y-%m-%d %HZ}", fontsize=13.5, fontweight="bold", x=0.07, ha="left", y=0.985)  # fmt: skip
    fig.text(0.07, 0.945, f"a member counts when its 65–90°N cap height is {THRESHOLD:+.1f} sd or more above normal at "
             "every level from 100 hPa down; bold numbers, filled markers: more members than usual", fontsize=9.5,
             color="0.25")  # fmt: skip
    fig.text(0.07, 0.008, textwrap.fill(
        "Cap heights drift-corrected and standardised as on the dripping-paint chart. Usual: how often MERRA-2 shows "
        "such a column within 15 days of the calendar date. More than usual: a one-sided binomial test of the member "
        f"count against that rate, the run's days and levels held to a {100 * ALPHA:.0f} % false discovery rate. "
        f"{credit}; "
        "MERRA-2 (NASA GMAO).", 190), fontsize=8, color=MUTED, linespacing=1.4)  # fmt: skip
    fig.subplots_adjust(left=0.15, right=0.93, top=0.89, bottom=0.085)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


# ── ring: sectors, the westerlies' shift, and each member's link values ─────────────────────────────────────────
def split(x: np.ndarray, lon: np.ndarray, cap, keep: np.ndarray | None = None) -> np.ndarray:
    """(member, lead, 1 + len(SECTORS)): the cap mean of x (member, lead, lat, lon) over the cells `keep`, then each
    sector's; `cap` is polarcap.cap."""
    every = np.ones(x.shape[-2:], bool) if keep is None else keep
    parts = [every] + [every & ((lon >= a) & (lon < b))[None, :] for a, b, _ in SECTORS]
    return np.stack([cap(x, k) for k in parts], axis=-1)


def westerlies(z500: np.ndarray, msl: np.ndarray, lat: np.ndarray, low: np.ndarray) -> dict:
    """{u500, usfc: (member, lead)} in m/s: the geostrophic zonal wind anomaly over JET's first band minus its second,
    from the zonal-mean 500 hPa height anomaly (m) and from the sea-level pressure anomaly (hPa) over the cells `low`
    whose neighbours to the north and south are low too. Below zero = the westerlies shifted towards the equator."""
    phi = np.deg2rad(lat)
    f = 2 * OMEGA * np.sin(phi)
    w = np.cos(phi)
    u5 = -G / (f * RE) * np.gradient(z500.mean(-1), phi, axis=-1)
    us = -np.gradient(100.0 * msl, phi, axis=-2) / (RHO * RE * f[:, None])
    flat = low.copy()
    flat[1:-1] &= low[:-2] & low[2:]
    flat[[0, -1]] = False

    def band(u, south, north, keep=None):
        rows = (lat >= south) & (lat <= north)
        if keep is None:
            return (u[..., rows] * w[rows]).sum(-1) / w[rows].sum()
        ww = w[rows, None] * keep[rows]
        return (u[..., rows, :] * ww).sum((-2, -1)) / ww.sum()

    (s1, n1), (s2, n2) = JET
    return {"u500": band(u5, s1, n1) - band(u5, s2, n2), "usfc": band(us, s1, n1, flat) - band(us, s2, n2, flat)}


def ring(model, init, valid, A: dict, lat, lon, low, cap, agreed, credit: str, figures: bool = True) -> None:
    """Record the run's westerlies; with `figures`, also file the days 11-15 sectors and the members' link values and
    draw the ring figure. A {field: (member, lead, lat, lon)} anomalies (polarcap.anomalies), `low` the cells of the
    sea-level pressure cap, `cap` and `agreed` polarcap's cap mean and sign test."""
    lab = RR.MODEL_LABEL[model]
    wind = westerlies(A["z500"], A["msl"], lat, low)
    RR.record("ring", model, init, RR.entry("ring", {k: pd.DataFrame(v.T, index=valid) for k, v in wind.items()}))
    RR.render("ring", model)
    if not figures:
        return
    sec = {"z500": split(A["z500"], lon, cap), "msl": split(A["msl"], lon, cap, low)}
    early, late = _days(valid, init, EARLY), _days(valid, init, LATE)
    summary = {}
    for key, _, _, _, fmt in RING_PANELS:
        mean, ok = agreed(sec[key][:, late].mean(1))  # (1 + sectors,)
        summary[key] = {
            "cap": float(fmt.format(mean[0])),
            "agreed": bool(ok[0]),
            "same": int((ok[1:] & (np.sign(mean[1:]) == np.sign(mean[0]))).sum()),
            "of": len(SECTORS),
        }
    doc = {"model": model, "init": f"{init:%Y-%m-%dT%H}", "members": int(A["z500"].shape[0]), "late": summary}
    if "z100" in A:
        caps = {"x100": cap(A["z100"])[:, early], "c500": sec["z500"][:, early, 0], "cmsl": sec["msl"][:, early, 0],
                "y500": sec["z500"][:, late, 0], "ymsl": sec["msl"][:, late, 0]}  # fmt: skip
        doc["link"] = {k: np.round(v.mean(1), 2).tolist() for k, v in caps.items()}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data" / f"dwatch_{model}.json").write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    print(f"  {lab} ring, days {LATE[0]}-{LATE[1]}: " + "; ".join(
        f"{name} cap {s['cap']:+g} {unit}, " + (f"{s['same']} of {s['of']} sectors with it" if s["agreed"] else
                                               "members split on its sign")
        for (key, name, unit, _, _), s in zip(RING_PANELS, summary.values())), flush=True)  # fmt: skip
    _ring_figure(model, init, valid, sec, wind, summary, agreed, credit, OUT / f"dwatch_ring_{model}.webp")


def _ring_figure(model, init, valid, sec, wind, summary, agreed, credit, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import house

    house.apply()

    col, lab = RR.MODEL_COLOUR[model], RR.MODEL_LABEL[model]
    nl, rows = len(valid), 1 + len(SECTORS)
    late = _days(valid, init, LATE)
    fig = plt.figure(figsize=(12.6, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.34)
    for j, (key, name, unit, vmax, fmt) in enumerate(RING_PANELS):
        ax = fig.add_subplot(gs[0, j])
        mean, ok = agreed(sec[key])  # (lead, 1 + sectors)
        ax.set_facecolor(NEUTRAL)
        im = ax.imshow(np.ma.masked_invalid(np.where(ok, mean, np.nan).T), aspect="auto", cmap="RdBu_r", vmin=-vmax,
                       vmax=vmax, interpolation="nearest", extent=(-0.5, nl - 0.5, rows - 0.5, -0.5))  # fmt: skip
        for v in np.arange(-0.5, nl, 1):
            ax.axvline(v, color="white", lw=0.6)
        for v in np.arange(-0.5, rows, 1):
            ax.axhline(v, color="white", lw=2.6 if v == 0.5 else 0.6)
        ax.set_yticks(range(rows))
        ax.set_yticklabels(["whole cap, 65–90°N", *(s[2] for s in SECTORS)] if j == 0 else [], fontsize=8.8)
        ax.set_xticks(range(0, nl, 2))
        ax.set_xticklabels([f"{t:%-d %b}" for t in valid[::2]], fontsize=8.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045, extend="both")
        cb.set_label(f"{unit}, where the members agree on the sign", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
        s = summary[key]
        ax.set_title(f"{name} anomaly, sector by sector", fontsize=11, fontweight="bold", loc="left", pad=16)
        ax.text(0, 1.02, _minus(
            f"days {LATE[0]}–{LATE[1]}: whole cap {fmt.format(s['cap'])} {unit}, {s['same']} of {s['of']} sectors share "
            "its sign" if s["agreed"] else f"days {LATE[0]}–{LATE[1]}: the members do not agree on the cap's sign"),
            transform=ax.transAxes, fontsize=9, color="0.25")  # fmt: skip
    for j, (key, name) in enumerate(WIND_PANELS):
        ax = fig.add_subplot(gs[1, j])
        u = wind[key]
        for row in u:
            ax.plot(valid, row, color=col, lw=0.8, alpha=0.35, zorder=3)
        ax.plot(valid, u.mean(0), color=col, lw=2.8, zorder=5, label=f"{lab} mean, {u.shape[0]} members")
        lm = u[:, late].mean(1)
        ax.axhline(0, color="#8a8680", lw=1.0, zorder=2)
        ax.set_title(f"Westerlies' shift: {name}", fontsize=11, fontweight="bold", loc="left", pad=16)
        ax.text(0, 1.02, _minus(f"days {LATE[0]}–{LATE[1]}: members' mean {lm.mean():+.1f} m s⁻¹, "
                                f"{100 * (lm < 0).mean():.0f} % of the members below zero"), transform=ax.transAxes,
                fontsize=9, color="0.25")  # fmt: skip
        ax.set_ylabel("m s⁻¹, 55–65°N minus 35–45°N", fontsize=9.5)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    fig.suptitle(f"{lab} · descent watch: the whole ring or one sector · run {init:%Y-%m-%d %HZ}", fontsize=13.5,
                 fontweight="bold", x=0.07, ha="left", y=0.985)  # fmt: skip
    fig.text(0.07, 0.945, "a descent from the stratosphere lifts heights and pressure around the whole Arctic and "
             "shifts the westerlies towards the equator (below zero); a block lifts one or two sectors", fontsize=9.5,
             color="0.25")  # fmt: skip
    fig.text(0.07, 0.008, textwrap.fill(
        "Ensemble-mean anomalies against ERA5 1991–2020, raw model output, a sector coloured only where enough members "
        "share its sign for a two-sided sign test at 5 %; sea-level pressure over ground below 500 m. Winds are "
        "geostrophic, from the anomalies' gradients, not the models' own winds: the zonal-mean westerly anomaly over "
        f"55–65°N minus 35–45°N. {credit}.", 185), fontsize=8, color=MUTED, linespacing=1.4)  # fmt: skip
    fig.subplots_adjust(left=0.17, right=0.97, top=0.875, bottom=0.1)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


# ── link: every ensemble's newest run ───────────────────────────────────────────────────────────────────────────
def _link(x, y, covs) -> dict:
    """The partial correlation of x with y given covs, and both residuals (what the figure plots)."""
    t = partial_corr_test(x, y, covs, rng=0)
    return {"r": t["r"], "ci": t["ci"], "p": t["p_perm"], "n": t["n"], "rx": residualize(x, covs),
            "ry": residualize(y, covs)}  # fmt: skip


def combined(out: Path, max_age_days: float = 2.0) -> Path | None:
    """Every ensemble's newest run from data/dwatch_{m}.json, and the ensembles pooled: tested, filed (data/dwatch.json)
    and drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import house

    house.apply()

    docs = {m: json.loads(f.read_text()) for m in ENSEMBLES if (f := OUT / "data" / f"dwatch_{m}.json").exists()}
    docs = {m: d for m, d in docs.items() if d.get("link")}
    if not docs:
        return None
    t1 = max(pd.Timestamp(d["init"]) for d in docs.values())
    docs = {m: d for m, d in docs.items() if t1 - pd.Timestamp(d["init"]) <= pd.Timedelta(days=max_age_days)}
    val = {m: {k: np.asarray(v, float) for k, v in d["link"].items()} for m, d in docs.items()}
    who = np.concatenate([np.full(v["x100"].size, m) for m, v in val.items()])
    pool = {k: np.concatenate([v[k] for v in val.values()]) for k in ("x100", *LINK_CONTROLS, "y500", "ymsl")}
    own = [(who == m).astype(float) for m in list(val)[1:]]  # each ensemble's own mean, beside the intercept
    tests = {}
    for target, _, _ in LINK_TARGETS:
        for m, v in val.items():
            tests[target, m] = _link(v["x100"], v[target], [v[c] for c in LINK_CONTROLS])
        if len(val) > 1:
            tests[target, "all"] = _link(pool["x100"], pool[target], [*(pool[c] for c in LINK_CONTROLS), *own])
    passed = benjamini_hochberg(np.array([t["p"] for t in tests.values()]), ALPHA)
    for t, ok in zip(tests.values(), passed):
        t["pass"] = bool(ok)
    cols = [*val, *(["all"] if len(val) > 1 else [])]
    doc = {
        "runs": {m: d["init"] for m, d in docs.items()},
        "alpha": ALPHA,
        "link": {
            target: {
                m: {"n": t["n"], "pass": t["pass"],
                    **({"r": round(t["r"], 2), "ci": [round(c, 2) for c in t["ci"]]} if t["pass"] else {})}
                for m in cols
                for t in [tests[target, m]]
            }
            for target, _, _ in LINK_TARGETS
        },
    }  # fmt: skip
    (OUT / "data" / "dwatch.json").write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    fig, axes = plt.subplots(len(LINK_TARGETS), len(cols), figsize=(13.4, 7.4), squeeze=False)
    for i, (target, name, unit) in enumerate(LINK_TARGETS):
        for j, m in enumerate(cols):
            ax, t = axes[i, j], tests[target, m]
            colour = [RR.MODEL_COLOUR[w] for w in who] if m == "all" else RR.MODEL_COLOUR[m]
            ax.scatter(t["rx"], t["ry"], s=16, color=colour if t["pass"] else FAINT, alpha=0.8, linewidths=0, zorder=4)
            if t["pass"]:
                xs = np.array([t["rx"].min(), t["rx"].max()])
                ax.plot(xs, np.polyval(np.polyfit(t["rx"], t["ry"], 1), xs), color=house.INK, lw=1.8, zorder=5)
            ax.text(0.03, 0.96, _minus(f"r = {t['r']:+.2f} ({t['ci'][0]:+.2f} to {t['ci'][1]:+.2f})\n{t['n']} members")
                    if t["pass"] else f"no significant link\n{t['n']} members", transform=ax.transAxes, va="top",
                    fontsize=9.5, fontweight="bold" if t["pass"] else "normal", zorder=6,
                    color=house.INK if t["pass"] else MUTED,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})  # fmt: skip
            ax.axhline(0, color="#8a8680", lw=0.8, zorder=2)
            ax.axvline(0, color="#8a8680", lw=0.8, zorder=2)
            ax.grid(alpha=0.22)
            ax.tick_params(labelsize=8.5)
            if i == 0:
                ax.set_title("All pooled" if m == "all" else
                             f"{RR.MODEL_LABEL[m]} {pd.Timestamp(docs[m]['init']):%-d %b %HZ}", fontsize=10.5,
                             fontweight="bold", loc="left")  # fmt: skip
            if j == 0:
                ax.set_ylabel(f"days {LATE[0]}–{LATE[1]} cap: {name}, {unit}", fontsize=9.5)
            if i == len(LINK_TARGETS) - 1:
                ax.set_xlabel(f"days {EARLY[0]}–{EARLY[1]} cap height at 100 hPa, m", fontsize=9.5)
    fig.suptitle("Every ensemble's newest run · descent watch: does a member's stratosphere explain its Arctic surface?",
                 fontsize=13.5, fontweight="bold", x=0.06, ha="left", y=0.985)  # fmt: skip
    fig.text(0.06, 0.94, f"one dot per member: its days {LATE[0]}–{LATE[1]} polar cap against its days "
             f"{EARLY[0]}–{EARLY[1]} cap height at 100 hPa, after what its own days {EARLY[0]}–{EARLY[1]} troposphere "
             "explains is taken out of both; a line and r only where the link passes", fontsize=9.5, color="0.25")  # fmt: skip
    fig.text(0.06, 0.008, textwrap.fill(
        "65–90°N cap means of raw model anomalies against ERA5 1991–2020 (sea-level pressure over ground below 500 m). "
        f"Both axes are residuals after the member's days {EARLY[0]}–{EARLY[1]} cap at 500 hPa and at sea level (pooled: "
        "and each ensemble's own mean) are regressed out. r is the partial correlation with its 95 % bootstrap range; "
        f"Freedman–Lane permutation test, the panels held to a {100 * ALPHA:.0f} % false discovery rate. Above zero: the "
        "members with the weaker lower-stratospheric vortex go on to the higher Arctic heights or pressure.", 205),
        fontsize=8, color=MUTED, linespacing=1.4)  # fmt: skip
    fig.subplots_adjust(left=0.07, right=0.985, top=0.875, bottom=0.135, hspace=0.2, wspace=0.24)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"wrote {out}: " + "; ".join(f"{target} {m} " + (f"r {t['r']:+.2f}" if t["pass"] else "no link")
                                         for (target, m), t in tests.items()), flush=True)  # fmt: skip
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--combined", action="store_true", help="the link figure from every ensemble's newest run")
    a = ap.parse_args()
    if not a.combined:
        ap.error("only --combined runs on its own (forecast_drip.py and polarcap.py feed each model's run)")
    combined(OUT / "dwatch_link.webp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
