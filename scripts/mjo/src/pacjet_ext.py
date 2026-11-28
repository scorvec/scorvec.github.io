#!/usr/bin/env python3
"""North Pacific jet and Himalayan mountain torque in weeks 3-5, member by member, from the GEFS 00Z run.

Every member's weekly-mean 250 hPa zonal wind on the jet's grid and its weekly-mean Himalayan torque (pacjet_rf.reduce
and torque: what the reforecast was reduced to), less GEFS's drift against ERA5 at that week and time of year
(pacjet_rf_drift.npz, release pacjet-ref-v1). The wind anomaly from pacjet_ref.nc's climatology is projected on the
jet-phase patterns of the month of the week's middle day (PC1 extension/retraction, PC2 poleward/equatorward shift, in
the daily indices' standard deviations); the torque is its departure from ERA5's climatology, in Hadley.

Only what passed on the reforecast is given: a target's weeks from week 3 up to pacjet_rf.SKILL's `show`, and the jet
indices only in the weeks whose month is in the jet's season. A week's members are then sign-tested against no
departure (two-sided binomial, Benjamini-Hochberg at FDR_Q over the weeks given): the figure colours a week's mean
where that passes and leaves it grey where the members do not say which way.

NOAA publishes a 00Z run's days 16-35 about a day later, so the run is the previous day's unless --date names one.

    python src/pacjet_ext.py --drift pacjet_rf_drift.npz --out-dir assets/sst [--date 20261127]
        -> assets/sst/data/pacjet_ext.json and assets/sst/pacjet_ext.webp (no figure when nothing passed)

A week in which both jet indices passed also gets the phase diagram (assets/sst/pacjet_gefs_phase.webp): its members
in the plane of the two indices, the share of them in each phase (pacjet_core.phase_of of the weekly means) against
ERA5's share of 7-day means in that phase in the week's month and its neighbours (binomial test of the members' count,
Benjamini-Hochberg at FDR_Q over the phases), and the AIFS-ENS mean's days 0-15 leading in where it is read on the
same month's patterns.
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import datetime as dt
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter1d
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gefs"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import gefs_live as GL
import gefs_reforecast as RF
import hadley_slp_ref as R
import pacjet_core as PC
import pacjet_rf as J
from doycal import noleap_doy

MEMBERS = ("gec00", *(f"gep{i:02d}" for i in range(1, J.LIVE)))
FDR_Q = 0.10
# the phases' colours of pacjet.py's diagram
PH_COL = {"extension": "#c0392b", "poleward": "#2e86c1", "retraction": "#7d5ba6", "equatorward": "#d68910"}
NAMES = {
    "pc1": ("Jet extension (+) or retraction (−)", "PC1, standard deviations"),
    "pc2": ("Jet shifted poleward (+) or equatorward (−)", "PC2, standard deviations"),
    "torque": ("Himalayan mountain torque", "departure from normal, Hadley"),
}


@functools.cache
def _index(url: str) -> dict:
    return {(e[4], e[2]): (e[0], e[1]) for e in RF.index(url)}


def _get(job: tuple) -> bytes:
    url, var, level = job
    return RF.get(url, _index(url)[var, level])


def member(date: str, mem: str, weeks: int) -> tuple[np.ndarray, np.ndarray]:
    """(u (week, 47, 133), ps (week, lat, lon)) of one member of the run, the first `weeks` of pacjet_rf.WEEKS filled."""
    last = J.WEEKS[weeks - 1][1]
    jobs = [
        ("u", 24 * d, (GL.step_url(date, mem, 24 * d), "UGRD", f"{PC.LEVEL} mb")) for d in range(J.DAYS[0], last + 1)
    ]
    jobs += [("ps", h, (GL.step_url(date, mem, h), "PRES", "surface")) for h in J.ps_hours(J.DAYS[0], last)]
    return J.reduce(jobs, _get)


def drift_at(f: dict, doy: float) -> tuple[np.ndarray, np.ndarray]:
    """The drift of the wind (week, 47, 133) and of the torque (week,) at a start's common-year day."""
    out = []
    for k in ("u", "torque"):
        rows = []
        for w in range(len(J.WEEKS)):
            fit = {
                "coef": f[f"{k}_coef"][w],
                "cov": f[f"{k}_cov"][w],
                "s2": f[f"{k}_s2"][w],
                "kappa": f[f"{k}_kappa"][w],
            }
            with np.errstate(invalid="ignore", divide="ignore"):
                rows.append(R.eval_drift(fit, [doy])[0])
        out.append(np.stack(rows))
    return out[0].reshape(len(J.WEEKS), PC.EOF_LAT.size, PC.EOF_LON.size), out[1][:, 0]


def summary(x: np.ndarray) -> dict:
    """The members' values of one target in one week: mean, 10-90 % range, the share above zero and its sign test."""
    pos = int((x > 0).sum())
    return {
        "mean": round(float(x.mean()), 3),
        "p10": round(float(np.percentile(x, 10)), 3),
        "p90": round(float(np.percentile(x, 90)), 3),
        "above": round(pos / x.size, 3),
        "p": round(float(binomtest(pos, x.size, 0.5).pvalue), 4),
        "members": [round(float(v), 2) for v in x],
    }


def phase_odds(ref: xr.Dataset, pc1: list, pc2: list, month: int) -> dict:
    """Per phase (and neutral): the members' share, ERA5's share of 7-day means in the month and its neighbours, and
    whether the members' count differs from it."""
    t = pd.DatetimeIndex(ref.time.values)
    wk = uniform_filter1d(ref.pc.values.astype(float), 7, axis=0, mode="nearest")
    near = np.isin(t.month, [(month - 2) % 12 + 1, month, month % 12 + 1])
    clim = PC.phase_of(wk[near, 0], wk[near, 1])
    mem = PC.phase_of(pc1, pc2)
    out = {}
    for k, name in (*enumerate(PC.PHASES), (-1, "neutral")):
        c = float((clim == k).mean())
        n = int((mem == k).sum())
        out[name] = {
            "share": round(n / mem.size, 3),
            "normal": round(c, 3),
            "p": float(binomtest(n, mem.size, c).pvalue),
        }
    for o, ok in zip(out.values(), R.bh([o["p"] for o in out.values()], FDR_Q), strict=True):
        o["sig"], o["p"] = bool(ok), round(o["p"], 4)
    return out


def build(run: pd.Timestamp, drift: dict, skill: dict) -> dict:
    show = {k: v["show"] for k, v in skill["targets"].items()}
    doc = {
        "model": "gefs",
        "init": f"{run:%Y-%m-%dT00:00Z}",
        "show": show,
        "skill": {k: {"msss": v["msss"], "correlation": v["correlation"]} for k, v in skill["targets"].items()},
        "weeks": [],
    }
    nw = max(show.values())
    if not nw:
        return doc
    got = [member(f"{run:%Y%m%d}", m, nw) for m in MEMBERS]
    u = np.stack([g[0] for g in got])
    tq = np.stack([J.torque(drift["h"], g[1][:nw]) for g in got])
    du, dt_ = drift_at(drift, float(noleap_doy(pd.DatetimeIndex([run]))[0]))
    ref = xr.open_dataset(J.REF)
    in_season = ref.in_season.values.astype(bool)
    doc["members"] = len(got)
    for w in range(nw):
        days = J.week_days(run, w)
        month = J.week_month(run, w)
        row = {
            "week": w + 3,
            "from": f"{days[0]:%Y-%m-%d}",
            "to": f"{days[-1]:%Y-%m-%d}",
            "month": month,
            "targets": {},
        }
        if in_season[month - 1] and (show["pc1"] > w or show["pc2"] > w):
            pcs = PC.project(u[:, w] - du[w] - J.week_clim(ref, run, w), ref.proj.sel(month=month).values)
            for k, name in enumerate(("pc1", "pc2")):
                if show[name] > w:
                    row["targets"][name] = summary(pcs[:, k])
            if min(show["pc1"], show["pc2"]) > w:
                row["phase"] = phase_odds(ref, pcs[:, 0], pcs[:, 1], month)
        if show["torque"] > w:
            clim = float(PC.harm_eval(drift["torque_clim"], noleap_doy(days)).mean())
            row["targets"]["torque"] = summary(tq[:, w] - dt_[w] - clim) | {
                "normal": round(clim, 2),
                "sd": skill["targets"]["torque"]["obs_sd"][w],
            }
        doc["weeks"].append(row)
    cells = [t for row in doc["weeks"] for t in row["targets"].values()]
    for t, ok in zip(cells, R.bh([t["p"] for t in cells], FDR_Q), strict=True):
        t["sig"] = bool(ok)
    return doc


def render(doc: dict, skill: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import house
    import matplotlib.pyplot as plt

    house.apply()
    names = [k for k in J.TARGETS if any(k in row["targets"] for row in doc["weeks"])]
    run = pd.Timestamp(doc["init"][:10])
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names) + 0.6, 5.6), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, name in zip(axes[0], names, strict=True):
        rows = [row for row in doc["weeks"] if name in row["targets"]]
        sd = skill["targets"][name]["obs_sd"]
        for i, row in enumerate(rows):
            t = row["targets"][name]
            w = row["week"] - 3
            ax.fill_between([i - 0.42, i + 0.42], -sd[w], sd[w], color=house.GRID, lw=0, zorder=1)
            ax.scatter(
                i + rng.uniform(-0.22, 0.22, len(t["members"])), t["members"], s=13, color=house.EDGE, lw=0, zorder=2
            )
            col = (house.WARM if t["mean"] > 0 else house.COOL) if t["sig"] else house.MUTED
            ax.plot([i, i], [t["p10"], t["p90"]], color=col, lw=2.2, zorder=3, solid_capstyle="round")
            ax.plot([i - 0.3, i + 0.3], [t["mean"]] * 2, color=col, lw=3.2, zorder=4, solid_capstyle="round")
            lean = f"{t['above']:.0%} above normal" if t["mean"] > 0 else f"{1 - t['above']:.0%} below normal"
            note = lean if t["sig"] else "no clear lean"
            ax.annotate(
                note, (i, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -4), textcoords="offset points",
                ha="center", va="top", fontsize=9.5, color=col,
            )  # fmt: skip
        ax.axhline(0, color=house.INK, lw=0.8, zorder=1.5)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(
            [f"Week {r['week']}\n{pd.Timestamp(r['from']):%-d %b} – {pd.Timestamp(r['to']):%-d %b}" for r in rows]
        )
        ax.set_xlim(-0.6, len(J.WEEKS) - 0.4)  # every panel keeps the three weeks' room
        lim = 1.25 * max([*sd, *(abs(v) for r in rows for v in r["targets"][name]["members"])])
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(NAMES[name][1])
        ms = skill["targets"][name]["msss"]
        ax.set_title(
            f"{NAMES[name][0]}\nskill on the reforecast: " + ", ".join(f"{ms[r['week'] - 3]:.2f}" for r in rows),
            loc="left", fontsize=11,
        )  # fmt: skip
        ax.grid(axis="y", color=house.GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(
        f"North Pacific jet in weeks 3–5: GEFS, {doc['members']} members, run of {run:%-d %b %Y} 00 UTC",
        x=0.02, ha="left", fontsize=14, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.02, 0.012,
        "Dots: members, less the model's drift. Bar: their mean and 10–90 % range, coloured where the members lean one "
        "way (sign test, 10 % false discovery rate). Grey band: ±1 standard deviation of ERA5's weekly values.",
        fontsize=9, color=house.MUTED,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    house.clear(fig, 110)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


def render_phase(doc: dict, lead_in: Path, out: Path) -> None:
    """The phase diagram of the weeks that carry both indices (the first of them drawn)."""
    import matplotlib

    matplotlib.use("Agg")
    import house
    import matplotlib.pyplot as plt

    house.apply()
    row = next(r for r in doc["weeks"] if "phase" in r)
    x, y = (np.array(row["targets"][k]["members"]) for k in ("pc1", "pc2"))
    run = pd.Timestamp(doc["init"][:10])
    span = f"{pd.Timestamp(row['from']):%-d %b} – {pd.Timestamp(row['to']):%-d %b}"
    fig = plt.figure(figsize=(12.6, 7.0))
    ax = fig.add_axes([0.06, 0.12, 0.72 * 7.0 / 12.6, 0.72])
    lim = float(np.clip(np.abs(np.r_[x, y]).max() + 0.5, 2.5, 4.5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(np.cos(th), np.sin(th), color=house.GRID, zorder=0)
    ax.plot(np.cos(th), np.sin(th), color=house.EDGE, lw=1)
    for a in (45, 135):
        r = np.array([1, lim * 1.5])
        for sgn in (1, -1):
            ax.plot(sgn * np.cos(np.radians(a)) * r, sgn * np.sin(np.radians(a)) * r, color=house.EDGE, lw=0.8)
    spots = {
        "extension": (lim * 0.97, 0, "right", "center"),
        "poleward": (0, lim * 0.97, "center", "top"),
        "retraction": (-lim * 0.97, 0, "left", "center"),
        "equatorward": (0, -lim * 0.97, "center", "bottom"),
    }
    for name, (px, py, ha, va) in spots.items():
        ax.text(
            px, py, PC.PHASE_LABEL[name], ha=ha, va=va, fontsize=11, fontweight="bold", color=PH_COL[name], zorder=8
        )
    lead = json.loads(lead_in.read_text()) if lead_in.exists() else {}
    if lead.get("basis_month") == row["month"] and pd.Timestamp(lead["init"][:10]) >= run:
        tr = np.array(lead["pc_mean"])
        ax.plot(tr[:, 0], tr[:, 1], color=house.MUTED, lw=1.4, zorder=3, label="AIFS-ENS mean, days 0–15 (daily)")
        ax.scatter(tr[0, 0], tr[0, 1], s=26, color="white", edgecolor=house.MUTED, zorder=4)
        ax.scatter(tr[-1, 0], tr[-1, 1], s=26, color=house.MUTED, zorder=4)
    ax.scatter(
        x,
        y,
        s=30,
        color=house.ACCENT,
        edgecolor="white",
        lw=0.5,
        zorder=5,
        label=f"GEFS members, week {row['week']} mean",
    )
    ax.scatter(
        x.mean(), y.mean(), s=190, marker="*", color=house.INK, edgecolor="white", lw=0.8, zorder=6, label="their mean"
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("PC1: retraction (−) to extension (+), standard deviations")
    ax.set_ylabel("PC2: equatorward (−) to poleward (+), standard deviations")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    bx = fig.add_axes([0.60, 0.20, 0.36, 0.56])
    names = [*PC.PHASES, "neutral"]
    odds = row["phase"]
    yy = np.arange(len(names))[::-1]
    for yv, name in zip(yy, names, strict=True):
        o = odds[name]
        col = PH_COL.get(name, house.MUTED)
        bx.barh(yv, o["share"], height=0.5, color=col if o["sig"] else house.EDGE, zorder=2)
        bx.plot([o["normal"]] * 2, [yv - 0.36, yv + 0.36], color=house.INK, lw=1.6, zorder=3)
        bx.text(
            max(o["share"], o["normal"]) + 0.02, yv, f"{o['share']:.0%}", va="center", fontsize=10.5, color=house.INK
        )
    bx.set_yticks(yy)
    bx.set_yticklabels([PC.PHASE_LABEL[n] for n in names])
    bx.set_xlim(0, 1)
    bx.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    bx.set_title(f"Share of members in each phase, {span}", loc="left", fontsize=11)
    bx.grid(axis="x", color=house.GRID, lw=0.6)
    bx.set_axisbelow(True)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.tick_params(axis="y", length=0)
    fig.text(
        0.60, 0.085,
        "Black line: the normal share (ERA5 7-day means, this time of year).\nColoured where the members' share differs "
        "from it (10 % false discovery rate).",
        fontsize=9, color=house.MUTED, va="top",
    )  # fmt: skip
    fig.suptitle(
        f"North Pacific jet phase, week {row['week']} ({span}): GEFS, {doc['members']} members, run of {run:%-d %b %Y} 00 UTC",
        x=0.03, ha="left", fontsize=14, fontweight="bold",
    )  # fmt: skip
    house.clear(fig, 110)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="the 00Z run, YYYYMMDD (default: yesterday's)")
    ap.add_argument("--drift", default=str(J.DRIFT))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    run = pd.Timestamp(a.date) if a.date else pd.Timestamp(dt.datetime.now(dt.UTC).date()) - pd.Timedelta(days=1)
    skill = json.loads(J.SKILL.read_text())
    doc = build(run, dict(np.load(a.drift)), skill)
    doc["generated"] = f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%MZ}"
    out = Path(a.out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    fig = out / "pacjet_ext.webp"
    phase = out / "pacjet_gefs_phase.webp"
    if any("phase" in row for row in doc["weeks"]):
        render_phase(doc, out / "data" / "pacjet_aifs.json", phase)
    else:
        phase.unlink(missing_ok=True)
    if any(row["targets"] for row in doc["weeks"]):
        render(doc, skill, fig)
    else:
        fig.unlink(missing_ok=True)
        print("  nothing passed on the reforecast (or out of season): no figure", flush=True)
    (out / "data" / "pacjet_ext.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    for row in doc["weeks"]:
        print(
            f"  week {row['week']} ({row['from']}): "
            + ", ".join(f"{k} {t['mean']:+.2f} (p {t['p']})" for k, t in row["targets"].items())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
