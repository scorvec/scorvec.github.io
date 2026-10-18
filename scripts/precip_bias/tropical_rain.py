#!/usr/bin/env python3
"""Bias-adjusted ensemble rainfall forecast, weeks 1 and 2, global band and a flat tropics strip (circulation.html#rainfc).

The latest 00Z run of four ensembles, every member, as 1° box means (grid.py, global band 60°S-60°N), read by ens_tp.py:
ECMWF IFS-ENS and AIFS-ENS (50 members each, Google Cloud mirror only), NOAA GEFS (31, AWS Open Data) and ECCC GEPS (21,
MSC Datamart). The run is the newest with both ECMWF ensembles complete to day 15 (they land last); GEFS or GEPS missing
for that day are left out, the figures say so, and the next run of this script completes the day.
  correction  every member's week total times its model's significant rain-bias ratio for that week (global_bias.py: the
              current model cycle against IMERG adjusted to Final, season-weighted, 2° boxes, refined to 1°); a model with
              no significant bias anywhere (GEPS until its archive is long enough) is used uncorrected;
  normal      GPCP V3.2 daily 2001-2020 (gpcp_clim.py): the mean and terciles of 7-day totals starting near the week's first
              day of year (IMERG Final-based, so the same yardstick as the corrected forecast);
  sources     the blend - every model's members together, each model sharing its weight (blend.py: equal weights unless
              weights fitted out of sample beat them significantly) - or each model alone.
Maps (weeks 1 and 2 stacked): corrected rain (mm/day); anomaly against the normal (mm/day) and the favoured tercile with its
probability - both only where the members' tercile counts differ from the climatological 1/3 (one-sided binomial test per
tercile, the two directions Bonferroni-combined, BH-FDR q 0.10 over the map); the correction applied per ensemble; and the
change since the previous day's runs over the same calendar weeks (that run's days 2-8 and 9-15), both runs with this week's
correction: shown only where Welch's t-test on the two runs' members is significant (BH-FDR q 0.10 over the map).
Cells whose normal week has an upper tercile under 2 mm (dry season / desert) get no tercile or anomaly test.

Every run's member windows are kept in --state (ens_tp.py live files, carried between runs in release rainfc-state-v1):
read from there when present, fetched and written there otherwise (STATE/new.txt lists what this run wrote). The
previous day's runs come from there too and are fetched again only when missing.

    python scripts/precip_bias/tropical_rain.py --ref DIR --state DIR --out assets/sst [--date YYYYMMDD]
-> assets/sst/rainfc_{blend,ifs,aifs,gefs,geps}_{rain,anom,terc,dod}_{tropics,global}.webp,
   rainfc_{ifs,aifs,gefs,geps}_bias_{tropics,global}.webp, assets/sst/data/rainfc.json
"""

from __future__ import annotations

import pyproj  # noqa: F401  side effect: load before eccodes/xarray, which preload eckit's own libproj

import argparse
import json
import sys
import textwrap
from datetime import date, datetime, timedelta, UTC
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sst"))
import grid as G
import build_bias as BB
import ens_tp as ET

MODELS = {"ifs": "IFS-ENS", "aifs-ens": "AIFS-ENS", "gefs": "GEFS", "geps": "GEPS"}
SHORT = {"ifs": "ifs", "aifs-ens": "aifs", "gefs": "gefs", "geps": "geps"}
LEAD = ("ifs", "aifs-ens")  # the run date follows these two: they land last, and the page needs at least them
DOMAINS = {"tropics": (-30, 30), "global": (-60, 60)}
INK, MUTED = "#1a1a1a", "#6f6b64"
WRAP = 205  # subtitle characters per line


# ── data ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def latest_run(today: date) -> date:
    for k in range(3):
        d = today - timedelta(days=k)
        if all(ET.ready(m, d) for m in LEAD):
            return d
    raise SystemExit("no 00Z run with day 15 complete for both ECMWF ensembles in the last 3 days")


def windows(model: str, d: date, state: Path, new: list) -> tuple[dict, int] | None:
    """The member windows of a model's 00Z run of day d: from the state directory, else fetched and written there; None
    when the run cannot be had (it is then left out, said on the page, and retried by the next run)."""
    p = ET.live_path(state, model, d)
    if not p.exists():
        try:
            acc, members = ET.run(model, d, ET.LIVE_STEPS, ET.LIVE_MEMBERS[model])
        except ET.NotPublished as ex:
            print(f"  {model} {d}: {ex}", flush=True)
            return None
        except Exception as ex:  # noqa: BLE001  one failing system leaves the blend, it does not stop the forecast
            print(f"::warning::{model} {d}: {repr(ex)[:150]}", flush=True)
            return None
        ET.save_live(state, model, d, acc, members)
        new.append(p.name)
    return ET.load_live(p)


def ratio(B, bmeta: dict, model: str, wk: str) -> tuple[np.ndarray, str | None]:
    """A model's 1° correction for one lead week and, when there is none, why."""
    key = f"{model}_{wk}_ratio"
    if key not in B.files:
        return np.ones((G.NY, G.NX), "float32"), "no bias estimate yet"
    m = bmeta[f"{model}_{wk}"]
    return G.refine(B[key], B["lat"], B["lon"]), None if m.get("corrected", True) else m.get("note")


def blend_weights(W: dict | None, wk: str, have: list[str]) -> dict:
    """{model: weight} of the blend for one lead week over the models in this run (blend.py's weights, else equal),
    renormalised when one is missing."""
    fitted = ((W or {}).get("weeks", {}).get(wk) or {}).get("weights", {})
    w = {m: float(fitted.get(m, 1.0)) for m in have}
    s = sum(w.values())
    return {m: v / s for m, v in w.items()}


def tercile_test(X: np.ndarray, cl: dict, weights: np.ndarray):
    """X (member, NY, NX) week totals; weights per member (sum 1). -> P(below), P(above), favoured sign (+1/-1/0 where not
    significant), the testable mask. One-sided binomial tests against 1/3 on the effective member count, the two
    directions combined (x2), BH-FDR q 0.10 over the testable cells."""
    from scipy.stats import binom

    n_eff = 1.0 / (weights**2).sum()
    pb = np.tensordot(weights, (X < cl["q33"][None]).astype("float32"), axes=1)  # noqa: SIM300  variable X (maths notation), not a constant
    pa = np.tensordot(weights, (X > cl["q67"][None]).astype("float32"), axes=1)  # noqa: SIM300  variable X (maths notation), not a constant
    test = np.isfinite(cl["q67"]) & (cl["q67"] >= BB.DRY_Q67) & np.isfinite(X).all(0)
    n = int(np.rint(n_eff))
    p_up = binom.sf(np.round(pa * n) - 1, n, 1 / 3)
    p_dn = binom.sf(np.round(pb * n) - 1, n, 1 / 3)
    p = np.where(test, np.minimum(1.0, 2 * np.minimum(p_up, p_dn)), np.nan)
    sig = BB.bh_fdr(p, q=0.10) & test
    fav = np.where(sig, np.where(pa >= pb, 1, -1), 0)
    return pb, pa, fav, test


def welch(today: dict, prev: dict, weights: dict):
    """The change of a weighted ensemble mean between two runs over the same dates and its two-sided p-value per cell.
    today, prev: {model: (member, NY, NX) week totals}; weights {model: w}. Welch's t for the linear combination of
    independent member means sum_m w_m (mean_today - mean_prev), with Satterthwaite's degrees of freedom; p is NaN where
    no member varies. -> change (NY, NX), p (NY, NX)."""
    from scipy.stats import t as student

    d = v = vv = 0.0
    for m, w in weights.items():
        for X, sign in ((today[m], 1.0), (prev[m], -1.0)):
            n = X.shape[0]
            vi = w * w * X.var(axis=0, ddof=1) / n
            d = d + sign * w * X.mean(axis=0)
            v = v + vi
            vv = vv + vi * vi / (n - 1)
    tested = np.isfinite(d) & (v > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(tested, d / np.sqrt(v), 0.0)
        df = np.where(tested, v * v / vv, 1.0)
    return d, np.where(tested, 2 * student.sf(np.abs(t), df), np.nan)


def change_test(today: dict, prev: dict, weights: dict, q: float = 0.10):
    """welch() and where the change is significant: BH-FDR q over the tested cells. -> change, significant, tested."""
    d, p = welch(today, prev, weights)
    return d, BB.bh_fdr(p, q=q), np.isfinite(p)


# ── drawing ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def palettes():
    import matplotlib
    from matplotlib.colors import BoundaryNorm, ListedColormap

    def brbg(n, x):
        return matplotlib.colormaps["BrBG"].resampled(n)(x)  # (cm.get_cmap is gone in matplotlib >= 3.9)

    rain_lev = [0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30]
    rain_col = [
        "#e9f3e1",
        "#c7e4b6",
        "#94cf8f",
        "#5cb37c",
        "#2f9475",
        "#227a8e",
        "#2b5fa3",
        "#4a3f9e",
        "#6d2b8e",
        "#98207a",
        "#c2186a",
    ]
    anom_lev = [-12, -8, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 8, 12]
    dod_lev = [-10, -6, -4, -2, -1, -0.5, -0.2, 0.2, 0.5, 1, 2, 4, 6, 10]
    a = brbg(15, np.linspace(0, 1, 15))  # 13 bins between 14 levels + the two extensions; white on the middle bin
    a[7] = (1, 1, 1, 1)
    # favoured tercile = signed chance (- below normal, + above normal), 7 bins a side from the climatological 33.3 % to 100 %,
    # white only on the (practically empty) middle bin. White on a populated bin (e.g. 40-50 % below normal) would hide its
    # signal, and a tint on the middle bin, which can hold significant 33-40 % cells of BOTH signs, would colour weak wet
    # signals brown.
    terc_lev = [-100, -90, -80, -70, -60, -50, -40, -100 / 3, 100 / 3, 40, 50, 60, 70, 80, 90, 100]
    cm = matplotlib.colormaps["BrBG"]
    t = np.vstack([cm(np.linspace(0.0, 0.40, 7)), [(1, 1, 1, 1)], cm(np.linspace(0.60, 1.0, 7))])  # 15 bins, 15 colours
    bias_lev = [1 / 3, 0.5, 0.67, 0.8, 0.9, 1 / 0.9, 1 / 0.8, 1 / 0.67, 2, 3]
    b = brbg(11, np.linspace(0.05, 0.95, 11))
    b[5] = (1, 1, 1, 1)

    def mk(lev, cols, ext):
        cmap = ListedColormap(cols[1:-1] if ext == "both" else cols)
        if ext == "both":
            cmap.set_under(cols[0])
            cmap.set_over(cols[-1])
        return lev, cmap, BoundaryNorm(lev, cmap.N)

    # 9 bins + the "over 30" colour, all distinct (10 colours for 9 bins would make BoundaryNorm skip one)
    rl = rain_lev
    rc = ListedColormap(rain_col[1:10])
    rc.set_under("#ffffff")
    rc.set_over(rain_col[10])
    return {
        "rain": (rl, rc, BoundaryNorm(rl, rc.N), "max"),
        "anom": (*mk(anom_lev, a, "both"), "both"),
        "dod": (*mk(dod_lev, a, "both"), "both"),
        "terc": (terc_lev, ListedColormap(t), BoundaryNorm(terc_lev, 15), "neither"),
        "bias": (*mk(bias_lev, b, "both"), "both"),
    }


def draw(panels, kind: str, dom: str, title: str, sub: str, cbar: str, out: Path, lat=None, lon=None, grey=None):
    """panels: [(label, field (NY, NX))] stacked; grey: optional [(NY, NX) bool] per panel = no test possible (dry season)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import mapstyle as MS

    lat = G.LAT if lat is None else lat
    lon = G.LON if lon is None else lon
    lev, cmap, norm, ext = palettes()[kind]
    s, n = DOMAINS[dom]
    W = 14.0
    mh = W * 0.94 * (n - s) / 360.0
    gap = 0.32
    top = 0.95 + 0.15 * max(0, len(textwrap.wrap(sub, WRAP)) - 2)  # room for a third subtitle line
    bot = 0.85
    H = top + bot + len(panels) * mh + (len(panels) - 1) * gap
    fig = plt.figure(figsize=(W, H))
    pc = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180)
    # cell edges for pcolormesh (1° or 2° boxes)
    dl = float(lat[1] - lat[0])
    dn = float(lon[1] - lon[0])
    le = np.r_[lat - dl / 2, lat[-1] + dl / 2]
    oe = np.r_[lon - dn / 2, lon[-1] + dn / 2]
    rows = (lat >= s - dl) & (lat <= n + dl)
    re = np.r_[le[:-1][rows], le[1:][rows][-1]]
    for i, (lab, fld) in enumerate(panels):
        y0 = (bot + (len(panels) - 1 - i) * (mh + gap)) / H
        ax = fig.add_axes([0.03, y0, 0.94, mh / H], projection=proj)
        ax.set_extent([-180, 180, s, n], crs=proj)
        m = ax.pcolormesh(
            oe, re, np.ma.masked_invalid(fld[rows]), cmap=cmap, norm=norm, transform=pc, zorder=1, shading="flat"
        )
        if grey is not None and grey[i] is not None:
            ax.pcolormesh(
                oe,
                re,
                np.ma.masked_where(~grey[i][rows], np.ones(int(rows.sum()) * len(lon)).reshape(-1, len(lon))),
                cmap=matplotlib.colors.ListedColormap(["#e4e2dc"]),
                transform=pc,
                zorder=1,
                shading="flat",
            )
        MS.features(ax, states=False)
        ax.text(
            0.006,
            0.965,
            lab,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            va="top",
            color=INK,
            zorder=6,
            bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
        )
    MS.heading(fig, H, title, sub, wrap=WRAP)  # long subtitles wrap to a second (and third) line
    cb = MS.colorbar(fig, H, m, cbar, extend=ext)
    if kind == "terc":
        cb.set_ticks(lev)
        cb.set_ticklabels([f"{abs(x):.0f}" for x in lev])
    if kind in ("anom", "rain", "dod"):  # every boundary labelled (the default skipped every other one)
        cb.set_ticks(lev)
        cb.set_ticklabels([f"{x:g}".replace("-", "\u2212") for x in lev])
    if kind == "bias":
        cb.set_ticks(lev)
        cb.set_ticklabels(["⅓", "½", "⅔", "0.8", "0.9", "1.1", "1.25", "1.5", "2", "3"])
    MS.save(fig, out)


def pct(x: float) -> str:
    return f"{100 * x:+.1f} %".replace("-", "\u2212")


def blend_note(W: dict | None, wts: dict) -> str:
    """The blend's weights and how they were chosen (blend.py), for the blend maps' subtitle."""
    inb = [m for m in wts["wk1"] if any(w[m] > 0 for w in wts.values())]
    names = " · ".join(f"{MODELS[m]} {'/'.join(str(round(100 * w[m])) for w in wts.values())} %" for m in inb)
    if not W:
        return f"Weights (week 1/2): {names}, equal: the out-of-sample weight test has not run yet."
    opt = [wk[-1] for wk, w in W["weeks"].items() if w.get("optimised")]
    how = (
        "fitted"
        if len(opt) == len(W["weeks"])
        else f"fitted for week {opt[0]}, equal for the other"
        if opt
        else "equal"
    )
    tests = []
    for wk, w in W["weeks"].items():
        if "cv_skill" in w:
            s, lo, hi = (pct(v) for v in w["cv_skill"])
            tests.append(f"week {wk[-1]} {s} (90 % interval {lo} to {hi})")
        else:
            tests.append(f"week {wk[-1]} not fitted: {w.get('note', 'too little shared archive')}")
    fits = [w for w in W["weeks"].values() if "cv_skill" in w]
    unfit = [MODELS[m] for m in MODELS if fits and any(m not in w["fitted"] for w in fits)]
    tail = f" Not in the blend until its archive is long enough to verify: {', '.join(unfit)}." if unfit else ""
    return (
        f"Weights (week 1/2): {names}, {how}. RPS skill of fitted over equal weights in cross-validation: "
        f"{'; '.join(tests)}; fitted weights are used only where the whole interval is above zero.{tail}"
    )


def uncorrected(ms: list[str], notes: dict, name: str | None = None) -> str:
    """Subtitle sentence on the models of a map that go uncorrected, and why (one model: give its `name`)."""
    if name:
        why = notes[ms[0]]
        if why["wk1"] and why["wk1"] == why["wk2"]:
            return f" {name} uncorrected: {why['wk1']}."
        return "".join(f" {name} uncorrected in week {wk[-1]}: {y}." for wk, y in why.items() if y)
    off = [MODELS[m] for m in ms if all(notes[m].values())]
    return f" Uncorrected (no significant bias estimate yet): {', '.join(off)}." if off else ""


# ── main ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref",
        required=True,
        help="dir with bias_ratio_global.npz, gpcp_clim_week_global.npz [, blend_weights_global.json]",
    )
    ap.add_argument("--state", required=True, help="dir of ens_tp live files (this and the previous day's runs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", help="init YYYYMMDD (default: the latest complete 00Z)")
    ap.add_argument(
        "--skip-if-current",
        action="store_true",
        help="exit 0 without work when that run is already published with every model",
    )
    a = ap.parse_args()
    G.use_domain("global")
    ref, out, state = Path(a.ref), Path(a.out), Path(a.state)
    (out / "data").mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    B = np.load(ref / "bias_ratio_global.npz")
    bmeta = json.loads(str(B["meta"]))
    C = np.load(ref / "gpcp_clim_week_global.npz")
    wf = ref / "blend_weights_global.json"
    W = json.loads(wf.read_text()) if wf.exists() else None
    init = datetime.strptime(a.date, "%Y%m%d").date() if a.date else latest_run(datetime.now(UTC).date())
    prev = init - timedelta(days=1)
    print(f"init {init} 00Z", flush=True)
    pub = out / "data" / "rainfc.json"
    if a.skip_if_current and pub.exists():
        p = json.loads(pub.read_text())
        if p.get("init") == f"{init}" and set(p.get("members", {})) == set(MODELS):
            print("already published with every model - nothing to do", flush=True)
            return 0
    weeks = {"wk1": init, "wk2": init + timedelta(days=7)}
    X, P, info, notes, new = {}, {}, {"init": f"{init}", "members": {}}, {}, []
    for model in MODELS:
        got = windows(model, init, state, new)
        if got is None:
            continue
        win, info["members"][model] = got
        r = {wk: ratio(B, bmeta, model, wk) for wk in weeks}
        notes[model] = {wk: why for wk, (_, why) in r.items()}
        X[model] = {wk: win[wk] * r[wk][0][None] for wk in weeks}
        before = windows(model, prev, state, new)
        if before is not None:  # the same calendar weeks in the previous day's run, with this week's correction
            P[model] = {"wk1": before[0]["next1"] * r["wk1"][0][None], "wk2": before[0]["next2"] * r["wk2"][0][None]}
    (state / "new.txt").write_text("".join(f"{n}\n" for n in new))
    if not all(m in X for m in LEAD):
        raise SystemExit(f"the ECMWF ensembles of {init} could not be read")
    missing = [m for m in MODELS if m not in X]
    cl = {wk: BB.clim_at(C, d) for wk, d in weeks.items()}
    wts = {wk: blend_weights(W, wk, list(X)) for wk in weeks}

    def fmt(d):
        return f"{d:%-d %b}"

    def span(d, e):
        return f"{d.day}–{e.day} {e:%b}" if d.month == e.month else f"{fmt(d)}–{fmt(e)}"

    wl = {wk: f"Week {wk[-1]} · {span(d, d + timedelta(days=6))}" for wk, d in weeks.items()}
    gone = f" Not in this run: {', '.join(MODELS[m] for m in missing)} (not available yet)." if missing else ""
    # the blend holds the models with weight in it (one not yet verified is shown only on its own)
    inblend = [m for m in X if any(wts[wk].get(m, 0.0) > 0 for wk in weeks)]
    srcs = {"blend": inblend, **{SHORT[m]: [m] if m in X else [] for m in MODELS}}
    label = {"blend": "Blend", **{SHORT[m]: MODELS[m] for m in MODELS}}
    stats = {}
    for sk, ms in srcs.items():
        if not ms:  # a model missing from this run: its figures say so instead of showing an older run
            for kind in ("rain", "anom", "terc", "dod"):
                for dom in DOMAINS:
                    draw(
                        [(wl[w], np.full((G.NY, G.NX), np.nan)) for w in weeks],
                        kind,
                        dom,
                        f"{label[sk]} · 00Z {fmt(init)} {init.year} · not available",
                        f"The {label[sk]} run of 00Z {fmt(init)} is not on its source yet; this figure fills in at the next update.",
                        "",
                        out / f"rainfc_{sk}_{kind}_{dom}.webp",
                    )
            continue
        res = {}
        for wk in weeks:
            w = wts[wk] if sk == "blend" else {ms[0]: 1.0}
            Xs = np.concatenate([X[m][wk] for m in ms])
            mw = np.concatenate([np.full(X[m][wk].shape[0], w[m] / X[m][wk].shape[0]) for m in ms])
            mean = np.tensordot(mw, Xs, axes=1)
            pb, pa, fav, test = tercile_test(Xs, cl[wk], mw)
            anom = np.where(fav != 0, (mean - cl[wk]["mean"]) / 7.0, np.nan)
            terc = np.where(fav > 0, 100 * pa, np.where(fav < 0, -100 * pb, np.nan))
            res[wk] = {"rain": mean / 7.0, "anom": anom, "terc": terc, "dry": ~test}
            both = [m for m in ms if m in P]
            if both:
                wb = {m: w[m] for m in both}
                wb = {m: v / sum(wb.values()) for m, v in wb.items()}
                dlt, sig, tested = change_test({m: X[m][wk] for m in both}, {m: P[m][wk] for m in both}, wb)
                res[wk]["dod"] = np.where(sig, dlt / 7.0, np.nan)
                res[wk]["dod_models"] = both
            band = (np.abs(G.LAT) <= 30)[:, None] & test
            stats[f"{sk}_{wk}"] = {
                "tested_cells": int(test.sum()),
                "sig_frac_tropics": round(float((fav != 0)[band].mean()), 3),
                "above_frac_tropics": round(float((fav > 0)[band].mean()), 3),
                "below_frac_tropics": round(float((fav < 0)[band].mean()), 3),
            }
            if "dod" in res[wk]:
                stats[f"{sk}_{wk}"]["dod_sig_frac"] = round(float(sig[tested].mean()), 3) if tested.any() else 0.0
        n_mem = sum(info["members"][m] for m in ms)
        head = (
            f"{label[sk]} ({len(ms)} ensembles, {n_mem} members)" if sk == "blend" else f"{label[sk]} ({n_mem} members)"
        )
        head += f" · 00Z {fmt(init)} {init.year}"
        extra = f" {blend_note(W, wts)}{gone}" if sk == "blend" else ""
        unc = uncorrected(ms, notes) if sk == "blend" else uncorrected(ms, notes, label[sk])
        for dom in DOMAINS:
            draw(
                [(wl[w], res[w]["rain"]) for w in weeks],
                "rain",
                dom,
                f"{head} · bias-adjusted rain",
                "Week mean rain rate after each ensemble's significant rain bias against IMERG (adjusted to Final) is removed, member by member."
                + unc
                + extra,
                "mm per day",
                out / f"rainfc_{sk}_rain_{dom}.webp",
            )
            draw(
                [(wl[w], res[w]["anom"]) for w in weeks],
                "anom",
                dom,
                f"{head} · rain anomaly",
                "Bias-adjusted ensemble mean minus the GPCP 2001–2020 normal for the same week; shown only where the members' tercile split differs significantly from a normal week (FDR 10 %). Grey: too dry in a normal week to test."
                + unc
                + extra,
                "mm per day vs normal",
                out / f"rainfc_{sk}_anom_{dom}.webp",
                grey=[res[w]["dry"] for w in weeks],
            )
            draw(
                [(wl[w], res[w]["terc"]) for w in weeks],
                "terc",
                dom,
                f"{head} · favoured tercile",
                "Chance (%) of the favoured tercile against the GPCP 2001–2020 normal week: green = above normal, brown = below normal; shown only where significant (FDR 10 %). Grey: too dry to test."
                + unc
                + extra,
                "probability of the below-normal (brown) or above-normal (green) tercile, %",
                out / f"rainfc_{sk}_terc_{dom}.webp",
                grey=[res[w]["dry"] for w in weeks],
            )
            if "dod" in res["wk1"]:
                both = res["wk1"]["dod_models"]
                who = (
                    f" Blend of {', '.join(MODELS[m] for m in both)}, the models in both runs."
                    if sk == "blend" and len(both) < len(ms)
                    else ""
                )
                draw(
                    [
                        (f"{wl[w]} · previous run's days {int(w[-1]) * 7 - 5}–{int(w[-1]) * 7 + 1}", res[w]["dod"])
                        for w in weeks
                    ],
                    "dod",
                    dom,
                    f"{label[sk]} · change since the previous run · 00Z {fmt(init)} minus 00Z {fmt(prev)}",
                    "Bias-adjusted ensemble-mean rain of this run minus the previous day's run over the same dates, both with this week's correction; "
                    "shown only where the two runs' members differ significantly (Welch t-test, FDR 10 %). Green: wetter than the previous run, brown: drier."
                    + who,
                    "change since the previous run, mm per day",
                    out / f"rainfc_{sk}_dod_{dom}.webp",
                )
            else:
                draw(
                    [(wl[w], np.full((G.NY, G.NX), np.nan)) for w in weeks],
                    "dod",
                    dom,
                    f"{label[sk]} · change since the previous run · not available",
                    f"The previous day's run (00Z {fmt(prev)}) of {label[sk]} is not available, so there is nothing to compare this run with.",
                    "change since the previous run, mm per day",
                    out / f"rainfc_{sk}_dod_{dom}.webp",
                )
    for model in MODELS:
        m1 = bmeta.get(f"{model}_wk1")
        for dom in DOMAINS:
            if m1 is None:  # a reference built before this model joined
                draw(
                    [(wl[w], np.full((G.NY, G.NX), np.nan)) for w in weeks],
                    "bias",
                    dom,
                    f"{MODELS[model]} · the rain bias removed",
                    f"No bias estimate for {MODELS[model]} yet: it is used uncorrected until the next reference build.",
                    "model ÷ observed rain",
                    out / f"rainfc_{SHORT[model]}_bias_{dom}.webp",
                )
                continue
            panels, band = [], []
            for wk in weeks:
                m = bmeta[f"{model}_{wk}"]
                r2 = B[f"{model}_{wk}_ratio"]
                if not m.get("corrected", True):
                    lab = f"Week {wk[-1]} · no correction: {m['note']}"
                else:
                    lab = f"Week {wk[-1]} · {m['runs']} runs {m['first']} to {m['last']}"
                panels.append((lab, np.where(r2 != 1.0, 1.0 / r2, np.nan)))
                if "tropics_fc_over_obs" in m:  # the tested 30°S-30°N ratio (90 % interval) per week
                    r, lo, hi = m["tropics_fc_over_obs"]
                    band.append(
                        f"week {wk[-1]} {r:.2f} ({lo:.2f}–{hi:.2f})"
                        if (lo > 1 or hi < 1)
                        else f"week {wk[-1]} no significant bias"
                    )
            sub = (
                "Model week rain ÷ IMERG (adjusted to Final), season-weighted runs of this model cycle; only where significant (FDR 10 %), shrunk by its uncertainty. "
                "Green: model too wet, brown: too dry."
            )
            if band:
                sub += f" All of 30°S–30°N: {', '.join(band)} (90 % interval)."
            draw(
                panels,
                "bias",
                dom,
                f"{MODELS[model]} {m1['cycle']} · the rain bias removed",
                sub,
                "model ÷ observed rain",
                out / f"rainfc_{SHORT[model]}_bias_{dom}.webp",
                lat=B["lat"],
                lon=B["lon"],
            )
    info.update(
        missing=missing,
        weeks={wk: [f"{d}", f"{d + timedelta(days=6)}"] for wk, d in weeks.items()},
        stats=stats,
        bias=bmeta,
        blend={"weights": wts, "fit": W},
        dod={"previous": f"{prev}", "models": sorted(P)},
        normal="GPCP V3.2 daily 2001-2020, 7-day totals",
        made=datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
    )
    pub.write_text(json.dumps(info, indent=1))
    print(json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
