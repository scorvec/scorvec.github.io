#!/usr/bin/env python3
"""The shape of the northern polar vortex at 10 hPa, member by member: split, displaced or centred.

The same geometry as the history catalogue's split/displacement typing (build_strat_history.py), applied to each
member's 10 hPa height on polarcap.py's 1.5-deg cells, each day to day 15:
  edge       the vortex is the region poleward of 30N below EDGE, the DJFM-mean 10 hPa height at 60N (Seviour et al.
             2013). Each dataset against its own climate: the catalogue's edge is NCEP R1's (30 221 m), and R1 runs
             76 m above MERRA-2 there on the same 5541 winter days, so the forecasts, whose analyses are modern
             assimilations, take MERRA-2's (1980-2026).
  split      two separate low centres below the edge less SPLIT_OFFSET, the smaller at least FRAC of the larger's area
             (the catalogue's test: 20 of its 20 typed warmings, the aspect ratio's 2.4 threshold 17 of 20)
  displaced  not split, and the centroid of the region below the edge south of CENTRE_LAT (Seviour et al. 2013)
  none       no part of the cap below the edge: no vortex of winter strength (it happens early in the season, when the
             vortex is still small, or after a warming)
  centred    the rest
The edge is a winter value: in late October the climatological vortex reaches it only north of about 67N, so a young
vortex is smaller here than a midwinter one, and the shares are a statement about these thresholds, not a forecast
of a sudden warming (the vortex record's easterly share is that).

    python vortex_shape.py --combined     # every model's newest run from the records (strat-combine.yml)
polarcap.py calls shape() and members() for each model's run and records it (run_record.py, product vshape).
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import run_record as RR

EDGE = 30145.0  # m: MERRA-2's DJFM-mean 10 hPa height at 60N, 1980-2026
SPLIT_OFFSET = 150.0  # m below the edge: two low centres at this contour, the smaller >= FRAC of the larger
FRAC = 0.25
CENTRE_LAT = 66.0  # deg N: the centroid south of this is a displaced vortex
STATES = ("centred", "displaced", "split", "none")
STATE_COLOUR = {"centred": "#d9d6d0", "displaced": "#e39b3c", "split": "#a3262a", "none": "#4b3b66"}
LINE_COLOUR = {**STATE_COLOUR, "centred": "#7d786f"}  # a member's vortex contour on the maps (polarcap.py)
STATE_LABEL = {
    "centred": "centred",
    "displaced": "displaced (centre south of 66°N)",
    "split": "split (two centres)",
    "none": "no vortex below the edge",
}
OUT = RR.REPO / "assets" / "sst"


def geometry(f: np.ndarray, lat: np.ndarray, lon: np.ndarray, edge: float) -> np.ndarray:
    """(n, 2): the centroid latitude and aspect ratio of the region below `edge` of each field of f (n, lat, lon),
    moments on an equal-area polar grid; NaN where nothing is below it."""
    la, lo = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    r = 2 * np.sin((np.pi / 2 - la) / 2)
    x, y = r * np.cos(lo), r * np.sin(lo)
    da = np.cos(la)
    keep = la >= np.deg2rad(30)
    out = []
    for g in f:
        q = np.where(keep, np.clip(edge - g, 0, None), 0) * da
        m0 = q.sum()
        if m0 <= 0:
            out.append((np.nan, np.nan))
            continue
        xc, yc = (q * x).sum() / m0, (q * y).sum() / m0
        j11 = (q * (x - xc) ** 2).sum() / m0
        j22 = (q * (y - yc) ** 2).sum() / m0
        j12 = (q * (x - xc) * (y - yc)).sum() / m0
        s = np.sqrt((j11 - j22) ** 2 + 4 * j12**2)
        ar = np.sqrt((j11 + j22 + s) / max(j11 + j22 - s, 1e-12))
        clat = 90 - np.rad2deg(2 * np.arcsin(min(np.hypot(xc, yc) / 2, 1)))
        out.append((clat, ar))
    return np.array(out).reshape(-1, 2)


def vortex_count(f: np.ndarray, lat: np.ndarray, level: float, frac: float = FRAC) -> tuple[int, float]:
    """Separate low centres below `level` poleward of 30N in f (lat, lon) (longitude wraps, the pole row is one
    point): the number whose area is at least `frac` of the largest's, and the second-to-first area ratio."""
    from scipy import ndimage

    lab, n = ndimage.label((f < level) & (lat[:, None] >= 30))
    par = list(range(n + 1))

    def find(a):
        while par[a] != a:
            a = par[a]
        return a

    for i in range(f.shape[0]):
        if lab[i, 0] and lab[i, -1]:
            par[find(lab[i, 0])] = find(lab[i, -1])
    pole = [x for x in np.unique(lab[np.argmax(lat)]) if x]
    for x in pole[1:]:
        par[find(x)] = find(pole[0])
    root = np.vectorize(lambda x: find(x) if x else 0)(lab)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones(f.shape[1])[None, :]
    a = sorted([area[root == r].sum() for r in np.unique(root) if r], reverse=True)
    if not a:
        return 0, 0.0
    return sum(1 for v in a if v >= frac * a[0]), (a[1] / a[0] if len(a) > 1 else 0.0)


def shape(z: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict:
    """{clat, ar, state: (member, lead)} for the 10 hPa heights z (member, lead, lat, lon), state an index of
    STATES."""
    n, nl = z.shape[:2]
    flat = z.reshape(n * nl, *z.shape[2:])
    g = geometry(flat, lat, lon, EDGE)
    count = np.array([vortex_count(f, lat, EDGE - SPLIT_OFFSET)[0] for f in flat])
    state = np.where(count >= 2, 2, np.where(g[:, 0] < CENTRE_LAT, 1, 0))
    state = np.where(np.isnan(g[:, 0]), 3, state)
    return {"clat": g[:, 0].reshape(n, nl), "ar": g[:, 1].reshape(n, nl), "state": state.reshape(n, nl)}


def series(s: dict) -> dict:
    """The run record's member series (member, lead): 100 where a member's vortex is split (displaced), else 0, and
    the centroid latitude."""
    return {"split": 100.0 * (s["state"] == 2), "displaced": 100.0 * (s["state"] == 1), "clat": s["clat"]}


def any_split(s: dict) -> float:
    """The share of members whose vortex is split on any day 1-15."""
    return float((s["state"][:, 1:] == 2).any(1).mean())


def members(model: str, init: pd.Timestamp, s: dict, ctrl: dict | None, credit: str, out: Path) -> Path:
    """Every member's vortex each day (a row per member, the members ordered by the day their vortex first splits,
    then is displaced), the control on its own row above, and the day's shares beneath."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    st = s["state"]
    n, nl = st.shape

    def first(row, k):
        hit = np.flatnonzero(row == k)
        return hit[0] if hit.size else nl

    order = sorted(range(n), key=lambda i: (first(st[i], 2), first(st[i], 1), i))
    cmap = ListedColormap([STATE_COLOUR[k] for k in STATES])
    rows = n + (2 if ctrl is not None else 0)
    fig = plt.figure(figsize=(11.5, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=(2.4, 1.0), hspace=0.2)
    ax = fig.add_subplot(gs[0])
    grid = np.full((rows, nl), np.nan)
    if ctrl is not None:
        grid[0] = ctrl["state"][0]
    grid[rows - n :] = st[order]
    ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(STATES) - 0.5,
              interpolation="nearest", extent=(-0.5, nl - 0.5, rows - 0.5, -0.5))  # fmt: skip
    ax.set_yticks([0] if ctrl is not None else [])
    ax.set_yticklabels([RR.MODEL_LABEL[ctrl["model"]]] if ctrl is not None else [], fontsize=8.5)
    ax.set_ylabel(f"{n} members, sorted", fontsize=9)
    ax.set_xticks(range(nl))
    ax.set_xticklabels([])
    ax.tick_params(length=0)
    for x in np.arange(-0.5, nl, 1):
        ax.axvline(x, color="white", lw=0.6)
    ax.set_title(f"Every member's vortex, each day · {n} members, sorted by when theirs first splits · split on some "
                 f"day 1–15: {any_split(s):.0%}", fontsize=11, fontweight="bold", loc="left")  # fmt: skip
    ax2 = fig.add_subplot(gs[1])
    days = np.arange(nl)
    bottom = np.zeros(nl)
    for k in ("split", "displaced", "none"):
        share = 100.0 * (st == STATES.index(k)).mean(0)
        ax2.bar(days, share, bottom=bottom, width=0.82, color=STATE_COLOUR[k], label=STATE_LABEL[k])
        bottom += share
    ax2.set_ylim(0, 100)
    ax2.set_xlim(-0.5, nl - 0.5)
    ax2.set_ylabel("% of members", fontsize=9)
    valid = [init + pd.Timedelta(days=int(d)) for d in days]
    ax2.set_xticks(days)
    ax2.set_xticklabels([f"{v:%-d %b}" if d % 2 == 0 else "" for d, v in zip(days, valid)], fontsize=8.5)
    ax2.grid(axis="y", alpha=0.22)
    ax2.legend(fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False)
    ax2.set_title("Share of members", fontsize=11, fontweight="bold", loc="left")
    lab = RR.MODEL_LABEL[model]
    fig.suptitle(f"{lab} · the polar vortex at 10 hPa, member by member · run {init:%Y-%m-%d %HZ}", fontsize=13.5,
                 fontweight="bold", x=0.07, ha="left", y=0.985)  # fmt: skip
    fig.text(0.07, 0.945, "split: two low centres; displaced: one, its centre south of 66°N; no vortex: nothing below "
             "the winter vortex edge (the DJFM-mean height at 60°N)", fontsize=9.5, color="0.25")  # fmt: skip
    fig.text(0.07, 0.008, textwrap.fill(f"Each member's 10 hPa height on 1.5° cells, the vortex below {EDGE:,.0f} m and "
             f"a split two centres below {EDGE - SPLIT_OFFSET:,.0f} m, the smaller at least a quarter the size of the "
             f"larger (the site's warming catalogue's test). Raw model output. {credit}.", 190), fontsize=8,
             color="#6f6b64", linespacing=1.4)  # fmt: skip
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.085)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out


def combined(out: Path, max_age_days: float = 2.0) -> Path | None:
    """Every model's newest run from the records: the share of members split and displaced each day."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    models = ("aifs", "ifs", "gefs", "geps", "aifsc", "hres")
    newest = {m: rs[-1] for m in models if (rs := RR.runs("vshape", m))}
    if not newest:
        return None
    t1 = max(pd.Timestamp(r["init"]) for r in newest.values())
    newest = {m: r for m, r in newest.items() if t1 - pd.Timestamp(r["init"]) <= pd.Timedelta(days=max_age_days)}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True)
    for ax, key in zip(axes, ("split", "displaced")):
        for m, r in newest.items():
            y = RR.series(r, key)
            lab = f"{RR.MODEL_LABEL[m]} {pd.Timestamp(r['init']):%-d %b %HZ}"
            if r["members"] == 1:
                ax.step(y.index, y.values, where="mid", color="#1f1f1f" if m == "hres" else RR.MODEL_COLOUR[m],
                        lw=1.6, ls="--", zorder=6, label=lab)  # fmt: skip
            else:
                ax.plot(y.index, y.values, color=RR.MODEL_COLOUR[m], lw=2.4, zorder=5,
                        label=f"{lab}, {r['members']} members")  # fmt: skip
        ax.set_ylim(-3, 103)
        ax.set_ylabel("% of members", fontsize=9.5)
        ax.set_title(f"Members with a {STATE_LABEL[key]} vortex", fontsize=11, fontweight="bold", loc="left")
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
        ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.9)
    fig.suptitle(f"Every ensemble's newest run · the polar vortex at 10 hPa: split or displaced (newest {t1:%Y-%m-%d %HZ})",
                 fontsize=13.5, fontweight="bold", x=0.07, ha="left", y=0.985)  # fmt: skip
    fig.text(0.07, 0.945, "the share of each model's members whose vortex is split or displaced that day; a single run "
             "(dashed) is 0 or 100", fontsize=9.5, color="0.25")  # fmt: skip
    fig.text(0.07, 0.008, textwrap.fill(RR.PRODUCTS["vshape"].note + " Each model is labelled with its run.", 190),
             fontsize=8, color="#6f6b64", linespacing=1.4)  # fmt: skip
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.085, hspace=0.25)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--combined", action="store_true", help="every model's newest run from the records")
    a = ap.parse_args()
    if not a.combined:
        ap.error("only --combined runs on its own (polarcap.py draws each model's run)")
    combined(OUT / "vshape_models.webp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
