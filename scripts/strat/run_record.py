#!/usr/bin/env python3
"""How the stratosphere forecasts move from run to run: one record per product and model.

The job that renders a model's run hands it to record(): per key, a series against valid time (the members' mean and
10th/90th percentiles; one member, a deterministic run, is its own mean), and per scalar one number. The last
KEEP_DAYS days of runs stay in assets/sst/data/runs/<product>_<model>.json. One file per product and model, written only
by the job that renders that model, so workflows that finish close together never overwrite each other's runs; the
figures that put the models together (nh_vortex.py --from-records, ssw_summary.py, polarcap.py --combined,
vortex_shape.py --combined) read the newest run of each.

render() draws one model's record: the last SHOW runs over one another (the oldest faint, the newest bold with its band),
each key's change over days 6-15 against the run 24 h earlier, and the scalar run by run
(assets/sst/runs/<product>_<model>.webp). The heat flux and the dripping paint are taken from the JSON their scripts
publish (capture); the vortex, waves, polar-cap (with the vortex shape) and upper-stratosphere scripts record their own
members.

    python run_record.py capture heatflux --model ifs      # this run, from assets/sst/data/ifs_heatflux100.json
    python run_record.py render vortex --model ifs
    python run_record.py combined vortex --out assets/sst/nh_vortex_runs.webp   # the ensembles side by side
    python run_record.py merge /tmp/backfill/runs       # a backfill's records laid over main's
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "assets" / "sst" / "data"
RUNS = DATA / "runs"
FIGS = REPO / "assets" / "sst" / "runs"
KEEP_DAYS = 30
SHOW = 8  # four days of 00/12Z runs (two of AIFS-ENS, which runs at 00, 06, 12 and 18Z)
CHANGE = (6, 15)  # days of the newest run whose mean change against the run 24 h earlier is printed
MODEL_LABEL = {"aifs": "AIFS-ENS", "aifsc": "AIFS-ENS control", "ifs": "IFS-ENS", "hres": "IFS control", "gefs": "GEFS", "geps": "GEPS",
               "gdps": "GDPS", "geosfp": "GEOS FP", "gfs": "GFS", "geos": "GEOS FP"}  # fmt: skip
MODEL_COLOUR = {"aifs": "#b4541f", "aifsc": "#b4541f", "ifs": "#1f4b6e", "hres": "#1f4b6e", "gefs": "#3b88c4", "geps": "#2f7d4f",
                "gdps": "#6b4c9a", "geosfp": "#b0306a", "gfs": "#5a6f2b", "geos": "#b0306a"}  # fmt: skip


@dataclass
class Key:
    title: str  # panel title
    axis: str  # y-axis label
    dec: int  # decimals kept
    zero: str | None = "#8a8680"  # the colour of the reference line (None: none)
    unit: str = ""  # of the 24 h change
    short: str = ""  # the title in a row of several models (the title when empty)
    ref: float = 0.0  # where the reference line is
    share: bool = False  # a share of members, 0-100 (each member 0 or 100): no members' band, the axis 0-100


@dataclass
class Product:
    title: str  # the figure's title, before the model
    keys: dict  # key -> Key, recorded and drawn
    scalar: tuple | None = None  # (name, panel title, axis label): one number per run, x 100 when a share
    share: bool = True  # the scalar is a share (drawn as a percentage)
    band: str = "members' 10–90 %"  # what the newest run's band is
    note: str = ""  # the footer's source and method line
    extra: tuple = field(default_factory=tuple)  # keys recorded but not drawn here (nh_vortex's main figure reads them)


PRODUCTS = {
    "vortex": Product(
        "Northern polar vortex",
        {
            "u10": Key("u 10 hPa, 60°N", "m s⁻¹", 1, "#a33", "m s⁻¹", "u 10 hPa"),
            "zcap": Key("100 hPa 65–90°N cap height anomaly", "m", 0, unit="m", short="cap height"),
        },
        ("ssw", "Chance of a sudden warming by day 15: members easterly at 10 hPa, 60°N on any day", "% of members"),
        note="Height anomaly against the MERRA-2 1980– median detrended to the current year, as on the main vortex figure. "
        "AIFS-ENS 25 members + control, IFS-ENS 50 perturbed members (ECMWF open data, Google Cloud mirror), GEFS 30 "
        "members + control (NOAA), GEPS 20 members + control (ECCC).",
        extra=("u100",),
    ),
    "heatflux": Product(
        "100 hPa eddy heat flux, 45–75°N",
        {
            "tot": Key("[v′T′], all waves", "K m s⁻¹", 1, unit="K m s⁻¹"),
            "k1": Key("wave 1", "K m s⁻¹", 1, unit="K m s⁻¹"),
            "k2": Key("wave 2", "K m s⁻¹", 1, unit="K m s⁻¹"),
        },
        (
            "warn",
            "Members whose 10-day mean flux crosses the SSW warning level (27.5 K m s⁻¹) by day 15",
            "% of members",
        ),
        note="Daily-mean zonal-mean eddy heat flux, the model error removed as on the main figure.",
    ),
    "drip": Product(
        "Polar-cap height, 65–90°N",
        {
            "cap10": Key("10 hPa", "sd of the day's normal", 2, unit="sd"),
            "cap100": Key("100 hPa", "sd of the day's normal", 2, unit="sd"),
        },
        ("frac", "Members with the 100 hPa cap at +1 sd or more on the last day", "% of members"),
        band="± one member sd",
        note="Ensemble mean, drift-corrected as on the dripping-paint chart; positive = a weaker, warmer polar vortex.",
    ),
    "waves": Product(
        "Planetary waves 1 and 2, 55–65°N",
        {
            "w1_100": Key("wave 1, 100 hPa", "× normal", 2, unit="× normal", ref=1.0),
            "w2_100": Key("wave 2, 100 hPa", "× normal", 2, unit="× normal", ref=1.0),
            "w1_10": Key("wave 1, 10 hPa", "× normal", 2, unit="× normal", ref=1.0),
            "w2_10": Key("wave 2, 10 hPa", "× normal", 2, unit="× normal", ref=1.0),
        },
        note="Each member's wave amplitude over the 55–65°N band as a multiple of the day's normal standing wave (ERA5 "
        "1991–2020, wave_clim.nc); 1 = normal. A day the normal has no standing wave at that level is left blank.",
    ),
    "polarcap": Product(
        "Polar cap, 65–90°N: sea level and 500 hPa",
        {
            "msl": Key(
                "sea-level pressure anomaly (cells below 500 m)", "hPa", 1, unit="hPa", short="sea-level pressure"
            ),
            "z500": Key("500 hPa height anomaly", "m", 0, unit="m", short="500 hPa height"),
        },
        note="Anomalies against ERA5 1991–2020 at each day and hour, raw model output (polarcap.py); above zero = high "
        "pressure over the Arctic, the negative Arctic Oscillation.",
    ),
    "telecon": Product(
        "Arctic Oscillation and North Atlantic Oscillation",
        {
            "ao": Key("Arctic Oscillation", "sd of the daily index", 2, unit="sd", short="AO"),
            "nao": Key("North Atlantic Oscillation", "sd of the daily index", 2, unit="sd", short="NAO"),
        },
        note="Each member's daily index on the site's CPC-style patterns, raw model output against ERA5 1991–2020 "
        "(polarcap.py); negative = high pressure over the Arctic and Greenland.",
    ),
    "vshape": Product(
        "Polar vortex shape, 10 hPa",
        {
            "split": Key(
                "members with a split vortex (two centres)", "% of members", 0, None, "pts", "split", share=True
            ),
            "displaced": Key(
                "members with a displaced vortex (centre south of 66°N)",
                "% of members",
                0,
                None,
                "pts",
                "displaced",
                share=True,
            ),
            "clat": Key("vortex centre latitude", "°N", 1, unit="°", short="centre latitude", ref=66.0),
        },
        ("anysplit", "Members whose vortex splits on any day 1–15", "% of members"),
        band="members' 10–90 % (centre latitude)",
        note="Each member's 10 hPa height: split = two low centres 150 m inside the winter vortex edge, displaced = "
        "its centre south of 66°N (Seviour et al. 2013). Raw model output.",
    ),
    "upperstrat": Product(
        "Upper stratosphere, northern vortex",
        {
            "u1": Key("u 1 hPa, 60°N", "m s⁻¹", 1, "#a33", "m s⁻¹", "u 1 hPa"),
            "u10": Key("u 10 hPa, 60°N", "m s⁻¹", 1, "#a33", "m s⁻¹", "u 10 hPa"),
            "t1": Key("polar-cap temperature, 1 hPa", "K", 1, None, "K", "cap T 1 hPa"),
        },
        note="Raw model output, nothing drift-corrected (as on the upper-stratosphere charts). GEFS 30 members + control "
        "(NOAA); GFS, GDPS and GEOS FP single runs.",
    ),
}


# ── the record ──────────────────────────────────────────────────────────────────────────────────────────────────
def path(product: str, model: str) -> Path:
    return RUNS / f"{product}_{model}.json"


def summary(df: pd.DataFrame, dec: int) -> dict:
    """The members' mean and 10th/90th percentiles at each valid time (frame: valid time x member); null where no
    member has a value."""
    v = df.to_numpy(dtype=float)
    have = np.isfinite(v).any(axis=1)

    def stat(fn) -> list:
        s = np.full(len(v), np.nan)
        s[have] = fn(v[have])
        return [round(float(x), dec) if np.isfinite(x) else None for x in s]

    return {
        "mean": stat(lambda a: np.nanmean(a, axis=1)),
        "p10": stat(lambda a: np.nanpercentile(a, 10, axis=1)),
        "p90": stat(lambda a: np.nanpercentile(a, 90, axis=1)),
    }


def entry(product: str, frames: dict, scalars: dict | None = None) -> dict | None:
    """A run's record from member frames {key: DataFrame (valid time x member)} on one valid-time axis; None when the
    first key has nothing."""
    P = PRODUCTS[product]
    keys = [k for k in (*P.keys, *P.extra) if frames.get(k) is not None]
    if not keys:
        return None
    lead = frames[keys[0]].dropna(axis=1, how="all").dropna(axis=0, how="all")
    if lead.empty:
        return None
    rec = {"members": int(lead.shape[1]), "valid": [f"{t:%Y-%m-%dT%H}" for t in lead.index]}
    dec = {**{k: v.dec for k, v in P.keys.items()}, **dict.fromkeys(P.extra, 1)}
    for k in keys:
        rec[k] = summary(frames[k].reindex(lead.index), dec[k])
    rec.update(scalars or {})
    return rec


def record(product: str, model: str, base: pd.Timestamp, rec: dict | None) -> None:
    """Add (or replace) run `base` of the model to its record; runs older than KEEP_DAYS before it are dropped."""
    if rec is None:
        return
    p = path(product, model)
    doc = json.loads(p.read_text()) if p.exists() else {"product": product, "model": model, "runs": []}
    init = f"{base:%Y-%m-%dT%H}"
    cut = base - pd.Timedelta(days=KEEP_DAYS)
    runs = [r for r in doc["runs"] if r["init"] != init and pd.Timestamp(r["init"]) >= cut] + [{"init": init, **rec}]
    doc["runs"] = sorted(runs, key=lambda r: r["init"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    print(f"  run record {p.name}: {len(doc['runs'])} runs, newest {init}", flush=True)


def merge(src: Path) -> list[Path]:
    """Every record file in src merged into RUNS by start. Where both hold a run, the one with more fields wins (a
    record of the current kind over an older, thinner one), else the one RUNS holds: how a backfill or an old record
    moved over joins the record main holds at commit time."""
    done = []
    for f in sorted(src.glob("*.json")):
        new = json.loads(f.read_text())
        p = RUNS / f.name
        old = json.loads(p.read_text()) if p.exists() else {**new, "runs": []}
        by = {r["init"]: r for r in old["runs"]}
        for r in new["runs"]:
            if r["init"] not in by or len(r) > len(by[r["init"]]):
                by[r["init"]] = r
        newest = max(pd.Timestamp(i) for i in by)
        keep = [by[i] for i in sorted(by) if pd.Timestamp(i) >= newest - pd.Timedelta(days=KEEP_DAYS)]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**old, "runs": keep}, separators=(",", ":"), allow_nan=False))
        done.append(p)
    return done


def runs(product: str, model: str) -> list:
    p = path(product, model)
    return json.loads(p.read_text())["runs"] if p.exists() else []


def series(rec: dict, key: str, stat: str = "mean") -> pd.Series:
    return pd.Series(
        [np.nan if v is None else v for v in rec[key][stat]], index=pd.DatetimeIndex(rec["valid"]), dtype=float
    )


def day_change(rs: list, key: str, unit: str) -> str | None:
    """The newest run's mean over days CHANGE against the same valid times in the run 24 h earlier."""
    new = rs[-1]
    t0 = pd.Timestamp(new["init"])
    old = [r for r in rs if pd.Timestamp(r["init"]) == t0 - pd.Timedelta(days=1) and key in r]
    if not old or key not in new:
        return None
    a, b = series(new, key), series(old[0], key)
    win = a.index[(a.index >= t0 + pd.Timedelta(days=CHANGE[0])) & (a.index <= t0 + pd.Timedelta(days=CHANGE[1]))]
    win = win.intersection(b.index)
    if len(win) < 3:
        return None
    d = float((a[win] - b[win]).mean())
    return f"24 h change, days {CHANGE[0]}–{CHANGE[1]}: {d:+.{2 if unit in ('sd', '× normal') else 1}f} {unit}".replace(
        "-", "−"
    )


# ── capture from a product's published JSON ────────────────────────────────────────────────────────────────────
def capture_heatflux(model: str) -> tuple[pd.Timestamp, dict] | None:
    f = DATA / ("heatflux100.json" if model == "aifs" else f"{model}_heatflux100.json")
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    nh = d["hemispheres"]["nh"]
    rec = {"members": int(d["members"]), "valid": [f"{pd.Timestamp(t):%Y-%m-%dT%H}" for t in d["valid"]]}
    for k in PRODUCTS["heatflux"].keys:
        rec[k] = {s: [None if x is None else round(float(x), 1) for x in nh[k][s]] for s in ("mean", "p10", "p90")}
    w = nh.get("ssw_warning") or {}
    rec["warn"] = None if w.get("members_crossing") is None else round(float(w["members_crossing"]), 3)
    return pd.Timestamp(d["init"]), rec


def capture_drip(model: str) -> tuple[pd.Timestamp, dict] | None:
    f = DATA / f"{model}_drip.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    init = pd.Timestamp(d["init"].replace("Z", "").replace(" ", "T") + ":00")
    lev = [float(x) for x in d["levels"]]
    mean = np.asarray(d["forecast_mean_sd"], float)  # (lead, level)
    sd = np.asarray(d["member_sd"], float) if d.get("member_sd") is not None else np.zeros_like(mean)
    valid = [f"{init + pd.Timedelta(hours=int(h)):%Y-%m-%dT%H}" for h in d["lead_h"]]
    rec = {"members": int(d["members"]), "valid": valid}
    for k, L in (("cap10", 10.0), ("cap100", 100.0)):
        j = lev.index(L)

        def r2(x):
            return [None if not np.isfinite(v) else round(float(v), 2) for v in x]

        rec[k] = {"mean": r2(mean[:, j]), "p10": r2(mean[:, j] - sd[:, j]), "p90": r2(mean[:, j] + sd[:, j])}
    fr = d.get("frac100_ge1")
    rec["frac"] = None if not fr or fr[-1] is None else round(float(fr[-1]), 3)
    # the whole time-height field, which the next day's run is drawn against (forecast_drip.change)
    rec["field"] = {"lead_h": d["lead_h"], "levels": lev, "mean": d["forecast_mean_sd"]}
    return init, rec


CAPTURE = {"heatflux": capture_heatflux, "drip": capture_drip}


# ── the figure ──────────────────────────────────────────────────────────────────────────────────────────────────
def render(product: str, model: str, out: Path | None = None) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    P = PRODUCTS[product]
    rs = runs(product, model)
    if not rs:
        return None
    col, lab = MODEL_COLOUR[model], MODEL_LABEL[model]
    mine = rs[-SHOW:]
    nk = len(P.keys)
    fig = plt.figure(figsize=(11.5, 2.7 * nk + (2.4 if P.scalar else 0) + 0.9))
    gs = fig.add_gridspec(nk + (1 if P.scalar else 0), 1, hspace=0.62)
    for i, (key, K) in enumerate(P.keys.items()):
        ax = fig.add_subplot(gs[i])
        shown = [r for r in mine if key in r]
        for k, r in enumerate(shown):
            new = k == len(shown) - 1
            m = series(r, key)
            if new and r["members"] > 1 and not K.share:
                ax.fill_between(m.index, series(r, key, "p10"), series(r, key, "p90"), color=col, alpha=0.16, lw=0)
            age = (k + 1) / len(shown)
            ax.plot(
                m.index,
                m.values,
                color=col if new else "0.35",
                alpha=1.0 if new else 0.15 + 0.6 * age,
                lw=2.4 if new else 0.9 + 0.6 * age,
                zorder=5 if new else 3,
            )
        if K.zero:
            ax.axhline(K.ref, color=K.zero, lw=1.2, zorder=4)
        if K.share:
            ax.set_ylim(-3, 103)
        ax.set_title(K.title, fontsize=10.5, fontweight="bold", loc="left", pad=14)
        ax.text(0, 1.02, day_change(mine, key, K.unit) or "", transform=ax.transAxes, fontsize=8.5, color="0.25")
        ax.set_ylabel(K.axis, fontsize=9)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=8.5)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    if P.scalar:
        name, title, axis = P.scalar
        ax = fig.add_subplot(gs[nk])
        have = [r for r in rs if r.get(name) is not None]
        t = pd.DatetimeIndex([r["init"] for r in have])
        y = [(100 if P.share else 1) * r[name] for r in have]
        ax.plot(t, y, color=col, lw=1.8, marker="o", ms=3.5)
        ax.set_ylim(bottom=0, top=max([5.0, *y]) * 1.15)
        ax.set_ylabel(axis, fontsize=9)
        ax.set_title(title + ", run by run", fontsize=10.5, fontweight="bold", loc="left")
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=8.5)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b %HZ"))
    newest = pd.Timestamp(rs[-1]["init"])
    fig.suptitle(
        f"{lab} · {P.title}: the last runs (newest {newest:%-d %b %HZ})",
        fontsize=13,
        fontweight="bold",
        x=0.07,
        ha="left",
        y=0.995,
    )
    fig.text(
        0.07,
        0.962 if nk > 2 else 0.952,
        f"the last {len(mine)} runs: older runs grey and fainter, the newest bold with its {P.band}",
        fontsize=9,
        color="0.25",
    )
    fig.text(
        0.07,
        0.008,
        f"Each line is one run's ensemble mean. The 24 h change is the newest run's mean minus the run 24 h before it, "
        f"over its days {CHANGE[0]}–{CHANGE[1]}. {P.note}",
        fontsize=8,
        color="#6f6b64",
        wrap=True,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.915 if nk > 2 else 0.9, bottom=0.07)
    out = out or FIGS / f"{product}_{model}.webp"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def render_combined(product: str, models: tuple, out: Path) -> Path | None:
    """The models' records side by side: a column per model (its last SHOW runs, as render()) and every model's scalar
    run by run beneath (the vortex: nh_vortex_runs.webp)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    P = PRODUCTS[product]
    have = {m: runs(product, m) for m in models}
    have = {m: rs for m, rs in have.items() if rs}
    if not have:
        return None
    nk = len(P.keys)
    ns = 1 if P.scalar else 0
    fig = plt.figure(figsize=(13.6, 3.4 * nk + 3.6 * ns + 1.0))
    gs = fig.add_gridspec(nk + ns, len(have), height_ratios=(*(1,) * nk, *(0.8,) * ns), hspace=0.5, wspace=0.16)
    for j, (m, rs) in enumerate(have.items()):
        col, lab = MODEL_COLOUR[m], MODEL_LABEL[m]
        mine = rs[-SHOW:]
        for i, (key, K) in enumerate(P.keys.items()):
            ax = fig.add_subplot(gs[i, j])
            shown = [r for r in mine if key in r]
            for k, r in enumerate(shown):
                new = k == len(shown) - 1
                mean = series(r, key)
                if new and r["members"] > 1 and not K.share:
                    ax.fill_between(
                        mean.index, series(r, key, "p10"), series(r, key, "p90"), color=col, alpha=0.16, lw=0
                    )
                age = (k + 1) / len(shown)
                ax.plot(
                    mean.index,
                    mean.values,
                    color=col if new else "0.35",
                    alpha=1.0 if new else 0.15 + 0.6 * age,
                    lw=2.4 if new else 0.9 + 0.6 * age,
                    zorder=5 if new else 3,
                )
            if K.zero:
                ax.axhline(K.ref, color=K.zero, lw=1.2 if K.zero == "#a33" else 0.9, zorder=4)
            if K.share:
                ax.set_ylim(-3, 103)
            # the 24 h change on its own line under the title: beside it, four panels a row run the two together
            ax.set_title(f"{lab} · {K.short or K.title}", fontsize=10, fontweight="bold", loc="left", pad=14)
            ax.text(0, 1.015, day_change(mine, key, K.unit) or "", transform=ax.transAxes, fontsize=8, color="0.25")
            if j == 0:
                ax.set_ylabel(K.axis, fontsize=8.6)
            ax.grid(alpha=0.22)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    if P.scalar:
        name, title, axis = P.scalar
        ax = fig.add_subplot(gs[nk, :])
        for m, rs in have.items():
            got = [r for r in rs if r.get(name) is not None]
            t = pd.DatetimeIndex([r["init"] for r in got])
            ax.plot(t, [(100 if P.share else 1) * r[name] for r in got], color=MODEL_COLOUR[m], lw=1.8, marker="o",
                    ms=3.5, label=MODEL_LABEL[m])  # fmt: skip
        ax.set_ylim(bottom=0)
        ax.set_ylabel(axis, fontsize=8.6)
        ax.set_title(title + ", run by run", fontsize=10, fontweight="bold", loc="left")
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper left", ncol=4, framealpha=0.9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b %HZ"))
    newest = max(pd.Timestamp(rs[-1]["init"]) for rs in have.values())
    fig.suptitle(
        f"{P.title}: how the forecast has moved, run by run (newest {newest:%Y-%m-%d %HZ})",
        fontsize=13.5,
        fontweight="bold",
        x=0.06,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.06,
        0.952,
        f"the last {SHOW} runs of each model: older runs grey and fainter, the newest bold in the model's colour (an "
        f"ensemble's with its {P.band} band)",
        fontsize=9,
        color="0.25",
    )
    fig.text(
        0.06,
        0.012,
        "\n".join(
            textwrap.wrap(
                "Each line is one run's ensemble mean. The 24 h change is the newest run's mean minus the run 24 h "
                f"before it, over the newest run's days {CHANGE[0]}–{CHANGE[1]}. {P.note}",
                200,
            )
        ),
        fontsize=8,
        color="#6f6b64",
        linespacing=1.45,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9 if nk + ns < 4 else 0.915, bottom=0.1 if nk + ns < 4 else 0.08)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("capture", "render", "combined", "merge"))
    ap.add_argument("product", help="a product, or for merge the directory of record files to merge in")
    ap.add_argument("--model", choices=sorted(MODEL_LABEL), help="capture, render: the model")
    ap.add_argument("--models", default="aifs,ifs,gefs,geps", help="combined: the columns")
    ap.add_argument("--out", type=Path, help="combined: the figure")
    a = ap.parse_args()
    if a.cmd == "merge":
        for p in merge(Path(a.product)):
            print(f"  merged {p.name}", flush=True)
        return 0
    if a.product not in PRODUCTS:
        ap.error(f"unknown product {a.product} (one of {', '.join(PRODUCTS)})")
    if a.cmd == "combined":
        render_combined(a.product, tuple(a.models.split(",")), a.out or FIGS / f"{a.product}_all.webp")
        return 0
    if not a.model:
        ap.error(f"{a.cmd} needs --model")
    if a.cmd == "capture":
        got = CAPTURE[a.product](a.model)
        if got is None:
            print(f"  {a.product} {a.model}: nothing published to record", flush=True)
            return 0
        record(a.product, a.model, *got)
    render(a.product, a.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
