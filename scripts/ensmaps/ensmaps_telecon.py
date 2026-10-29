#!/usr/bin/env python3
"""Teleconnections, days 1-15, for circulation.html#telecon15: the PNA, EPO and WPO indices of every member of the
newest IFS-ENS, AIFS-ENS, GEFS and GEPS runs, each model's ensemble mean calibrated against its own archived runs, and
the chance of each index's positive and negative phase at the leads where that model has demonstrated skill. No
download: each ensmaps job leaves its members' fields in the Actions cache (Capture, from ensmaps.py), and `run` reads
the newest run of each model there.

  fields       each member's instantaneous 500 hPa height at 00Z of each day of a 00Z run (12Z of a 12Z run), days
               1-15, remapped conservatively from the model's grid onto the patterns' 2.5 deg grid 20-90N
               (latlon.remap) in whole metres, as the verification database keeps its members (z500_acc.run_fields):
               not the maps' daily means, since the hindcast is read at one hour
  indices      telecon15.Ref: the GEFS page's projectors on the anomaly from ERA5 1991-2020 at that hour, in seasonal
               sd of the daily index; the ensemble mean is the mean of the members' indices (the index is linear),
               at a lead where at least MIN_SHARE of the members are complete
  calibration  per model, index and lead, from the hindcast table (telecon15.py hindcast: every 00Z run the 500 hPa
               database holds from z500_acc.SINCE, against ERA5 at 00Z), the runs started within WINDOW days of this
               run's day of the year in any year: the regression obs = a m + b of ERA5's index on the ensemble-mean
               index m, its residual sd s, and the correlation r with its one-sided p on the effective sample
               n (1 - r1 r2) / (1 + r1 r2) (Bretherton et al. 1999; r1, r2 the lag-1 autocorrelations of the forecast
               and observed series over consecutive daily starts), from MIN_RUNS runs - the GEFS page's fit
               (gefs_telecon.py, telecon_geps.fit) on daily starts and single days. A 12Z run takes the 00Z runs' fit
               at the same lead in hours.
  skill        a lead is skilful only when r >= SKILL_MIN, p <= P_MAX and it passes Benjamini-Hochberg at FDR_Q over
               every model x index x lead of the update; elsewhere the figures hatch it, no demonstrated skill, and
               give no probability
  probability  P(index >= +THR) and P(index <= -THR) from the regression's predictive Student t (cal_cdf: n_eff - 2
               degrees of freedom, s widened for the fit's own uncertainty), at the skilful leads
  observed     ERA5's index at 00Z of the last TAIL_DAYS days, from the same table (ERA5T, about a week behind)
  outputs      per index the members and means (<id>_plume.webp), the phase probabilities (<id>_prob.webp) and the
               hindcast skill with each cell's sample (<id>_skill.webp) in FIG (the frames store); the numbers in JSON

    python scripts/ensmaps/ensmaps_telecon.py run --fields f/*.npz --ref telecon15_ref.npz --table tbl --site .
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[0] / "lib")]
import telecon15 as T
from latlon import G25N, remap, slot366

MODELS = {"ifs": "IFS-ENS", "aifs": "AIFS-ENS", "gefs": "GEFS", "geps": "GEPS"}  # ensmaps keys, the database's names
COLORS = {"ifs": "#b3372f", "aifs": "#7b4fb0", "gefs": "#2c5f8a", "geps": "#2e7d4f"}  # verification.html's centres
WINDOW = 45  # days of the year either side of the run's: the hindcast starts its calibration pools
MIN_RUNS = 30
SKILL_MIN = 0.25  # hindcast correlation below this = no demonstrated skill
P_MAX = 0.05  # ... and so is a correlation not significant at this level (one-sided, on the effective sample)
FDR_Q = 0.10  # ... or one that fails Benjamini-Hochberg at this rate over every model x index x lead of the update
THR = 0.5  # +-0.5 seasonal sd: the positive and negative phases
MIN_SHARE = 0.8  # the members complete at a lead for its ensemble mean
TAIL_DAYS = 21
FIG = "assets/telecon15/fig"
JSON_OUT = "assets/sst/data/telecon15.json"
NEG, POS, NEUTRAL = "#3f6f8f", "#b4541f", "#e4e2dc"  # the site's cool and warm tokens, its rule grey
NOT_SIG = "#dcd9d1"  # tested, not skilful (verify_figs.NOT_SIG)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ── the fields, from ensmaps.py ───────────────────────────────────────────────────────────────────────────────────────
class Capture:
    """Each member's 500 hPa height at the steps of telecon15.LEADS, on the patterns' grid. A failure is logged and ends
    the capture, never the maps: put and save catch everything."""

    def __init__(self, n: int):
        self.z = np.full((n, len(T.LEADS), *G25N.shape), np.nan, np.float32)
        self.row = {s: k for k, s in enumerate(T.LEADS)}
        self.error = ""

    def put(self, i: int, step: int, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> None:
        k = self.row.get(step)
        if k is None or self.error:
            return
        try:
            self.z[i, k] = remap(v, lat, lon, G25N)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            log(f"::warning::teleconnection fields: {self.error}; none saved for this run")

    def save(self, path: Path, model: str, cycle: str, init: pd.Timestamp) -> bool:
        try:
            if self.error:
                return False
            ok = np.isfinite(self.z).all((2, 3))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                model=model,
                cycle=cycle,
                init=f"{init:%Y-%m-%dT%H}:00Z",
                lead_h=np.array(T.LEADS, np.int16),
                z=np.where(ok[..., None, None], np.round(np.nan_to_num(self.z)), 0).astype(np.int16),
                ok=ok,
            )
            log(f"  teleconnection fields: {path} ({path.stat().st_size / 1e6:.1f} MB, {int(ok.any(0).sum())} leads)")
            return True
        except Exception as e:  # noqa: BLE001
            log(f"::warning::teleconnection fields not saved: {type(e).__name__}: {e}")
            return False


def load(paths: list[Path]) -> list[dict]:
    """The newest run of each model among the files, in MODELS order."""
    runs = {}
    for p in paths:
        with np.load(p) as z:
            r = {k: z[k] for k in z.files}
        key = str(r["model"])
        r["key"], r["cycle"] = key, str(r["cycle"])
        r["init"] = pd.Timestamp(str(r["init"])).tz_localize(None)
        if key in MODELS and (key not in runs or r["cycle"] > runs[key]["cycle"]):
            runs[key] = r
    return [runs[k] for k in MODELS if k in runs]


def member_indices(ref: T.Ref, r: dict) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """(valid times, members' indices (member, lead, index; NaN where a member is incomplete), the ensemble mean (lead,
    index; NaN at a lead with fewer than MIN_SHARE of the members))."""
    t = pd.DatetimeIndex(r["init"] + pd.to_timedelta(r["lead_h"].astype(int), unit="h"))
    n, nl = r["ok"].shape
    z = r["z"].reshape(n * nl, *G25N.shape)
    tt = pd.DatetimeIndex(np.tile(t.values, n))
    v = ref.index(ref.anomaly(z, tt), tt).reshape(n, nl, -1)
    v[~r["ok"]] = np.nan
    enough = r["ok"].sum(0) >= np.ceil(MIN_SHARE * n)
    mean = np.full(v.shape[1:], np.nan)
    mean[enough] = np.nanmean(v[:, enough], 0)
    return t, v, mean


# ── calibration (the GEFS and GEPS pages' fit, on daily starts) ───────────────────────────────────────────────────────
def lag1(x: np.ndarray, nxt: np.ndarray) -> float:
    """Correlation of x with its value at the next start (pairs where one exists), 0 when too few
    (telecon_geps.lag1)."""
    ok = nxt >= 0
    if ok.sum() < 10:
        return 0.0
    a, b = x[ok], x[nxt[ok]]
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 10 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
        return 0.0
    return float(np.corrcoef(a[good], b[good])[0, 1])


def fit(f: np.ndarray, o: np.ndarray, nxt: np.ndarray) -> dict | None:
    """The regression o = a f + b of paired values (NaN = no pair; nxt as lag1 takes it), its residual sd s, the
    correlation r and its one-sided p on the effective sample; None below MIN_RUNS pairs (telecon_geps.fit)."""
    from scipy.stats import t as tdist

    ok = np.isfinite(f) & np.isfinite(o)
    if ok.sum() < MIN_RUNS:
        return None
    r1, r2 = lag1(np.where(ok, f, np.nan), nxt), lag1(np.where(ok, o, np.nan), nxt)
    f, o = f[ok], o[ok]
    r = float(np.corrcoef(f, o)[0, 1])
    a, b = np.polyfit(f, o, 1)
    n = int(ok.sum())
    neff = float(np.clip(n * (1 - r1 * r2) / (1 + r1 * r2), 3, n))
    tt = r * np.sqrt(max(neff - 2, 1) / max(1 - r * r, 1e-9))
    return dict(
        n=n,
        n_eff=neff,
        p=float(tdist.sf(tt, max(neff - 2, 1))),
        r=r,
        a=float(a),
        b=float(b),
        s=float(np.std(o - (a * f + b), ddof=2)),
        xbar=float(f.mean()),
        sxx=float(((f - f.mean()) ** 2).sum()),
    )


def cal_cdf(x, c: dict, m):
    """P(observed index <= x) for the ensemble mean m under the regression c: Student t on n_eff - 2 degrees of freedom
    centred on a m + b, scale s sqrt(1 + 1/n_eff + (m - xbar)^2 / (sxx n_eff / n)) - the fit's own uncertainty, counted
    on the effective sample (telecon_geps.cal_cdf)."""
    from scipy.stats import t as tdist

    m = np.asarray(m, np.float64)
    w = np.sqrt(1.0 + 1.0 / c["n_eff"] + (m - c["xbar"]) ** 2 / (c["sxx"] * c["n_eff"] / c["n"]))
    return tdist.cdf((np.asarray(x) - (c["a"] * m + c["b"])) / (c["s"] * w), max(c["n_eff"] - 2.0, 1.0))


def pairs(fc: pd.DataFrame, obs: pd.DataFrame, model: str) -> pd.DataFrame:
    """The model's hindcast rows with ERA5's index on each valid day (obs_<id>), where ERA5 has it."""
    f = fc[fc["model"] == model].copy()
    f["valid"] = f["init"] + pd.to_timedelta(f["lead_h"].astype(int), unit="h")
    o = obs.set_index("valid")[list(T.IDS)].add_prefix("obs_")
    return f.join(o, on="valid", how="inner")


def calibrate(fc: pd.DataFrame | None, obs: pd.DataFrame | None, model: str, t0: pd.Timestamp) -> dict:
    """{(index, lead_h): fit or None} of one model from the runs started within WINDOW days of t0's day of the year."""
    out = dict.fromkeys(((s, lead) for s in T.IDS for lead in T.LEADS), None)
    if fc is None or obs is None:
        return out
    p = pairs(fc, obs, model)
    if p.empty:
        return out
    d = (slot366(pd.DatetimeIndex(p["init"])) - slot366(pd.DatetimeIndex([t0]))[0] + 183) % 366 - 183
    p = p[np.abs(d) <= WINDOW]
    for lead, g in p.sort_values("init").groupby("lead_h"):
        pos = {t: k for k, t in enumerate(g["init"])}
        nxt = np.array([pos.get(t + pd.Timedelta(days=1), -1) for t in g["init"]])
        for s in T.IDS:
            out[s, int(lead)] = fit(g[s].to_numpy(np.float64), g[f"obs_{s}"].to_numpy(np.float64), nxt)
    return out


def gate(cal: dict) -> None:
    """Mark every fitted cell of the update (model x index x lead) skilful or not: r >= SKILL_MIN, p <= P_MAX and
    Benjamini-Hochberg at FDR_Q over all of them."""
    from quantmet.stats import benjamini_hochberg

    cells = [c for per in cal.values() for c in per.values() if c]
    if not cells:
        return
    for c, k in zip(cells, benjamini_hochberg(np.array([c["p"] for c in cells]), FDR_Q), strict=True):
        c["fdr_ok"] = bool(k)
        c["skilful"] = bool(k and c["r"] >= SKILL_MIN and c["p"] <= P_MAX)


def skilful(c: dict | None) -> bool:
    return bool(c and c.get("skilful"))


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _plt():
    sys.path.insert(0, str(HERE.parents[0] / "verify"))
    import matplotlib.pyplot as plt
    import verify_figs as VF

    VF.style()
    return plt, VF


def ink_on(face, ink: str) -> str:
    """White or the ink, whichever contrasts more with the cell's colour (WCAG relative luminance)."""
    from matplotlib.colors import to_rgb

    def lum(c) -> float:
        v = np.array(to_rgb(c))
        v = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
        return float(v @ [0.2126, 0.7152, 0.0722])

    f = lum(face)
    return "white" if 1.05 / (f + 0.05) > (f + 0.05) / (lum(ink) + 0.05) else ink


def run_label(r: dict) -> str:
    return f"{MODELS[r['key']]} · {r['ok'].shape[0]} members · {r['init']:%-d %b %HZ} run"


def spans(t: pd.DatetimeIndex, bad: np.ndarray) -> list[tuple]:
    """The runs of consecutive leads marked bad, as (start, end) half a day either side."""
    out, k = [], 0
    while k < len(t):
        if bad[k]:
            j = k
            while j + 1 < len(t) and bad[j + 1]:
                j += 1
            out.append((t[k] - pd.Timedelta(hours=12), t[j] + pd.Timedelta(hours=12)))
            k = j + 1
        else:
            k += 1
    return out


def plume(sid: str, k: int, runs: list[dict], ind: dict, cal: dict, obs: pd.DataFrame | None, name: str, path: Path):
    """Every member's index (thin) and the ensemble mean, one panel per model, after ERA5's last weeks; the leads
    without demonstrated skill hatched."""
    plt, VF = _plt()
    import house
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    t_last = max(ind[r["key"]][0][-1] for r in runs)
    t0 = min(r["init"] for r in runs).normalize()
    vals = [ind[r["key"]][1][..., k] for r in runs]
    tail = pd.Series(dtype=float)
    if obs is not None:
        tail = obs[obs["valid"] >= t0 - pd.Timedelta(days=TAIL_DAYS)].set_index("valid")[sid].dropna()
        vals.append(tail.to_numpy())
    t_first = t0 - pd.Timedelta(days=TAIL_DAYS) if len(tail) else t0
    top = max(2.5, 1.08 * float(np.nanmax(np.abs(np.concatenate([v.ravel() for v in vals])))))
    nr, nc = (1, len(runs)) if len(runs) <= 2 else (2, 2)
    fig, axes = plt.subplots(nr, nc, figsize=house.size(15, 0.36 if nr == 1 else 0.62), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)  # fmt: skip
    for ax, r in zip(axes.ravel(), runs, strict=False):
        t, v, mean = ind[r["key"]]
        v, mean = v[..., k], mean[:, k]
        col = COLORS[r["key"]]
        ax.axhspan(-THR, THR, color=NEUTRAL, alpha=0.55, lw=0, zorder=0)
        ax.axhline(0, color=house.EDGE, lw=1.0, zorder=1)
        bad = np.array([not skilful(cal[r["key"]][sid, int(h)]) for h in r["lead_h"]])
        for a0, a1 in spans(t, bad):
            ax.axvspan(a0, a1, facecolor="none", edgecolor=house.EDGE, hatch="//", lw=0, zorder=0)
        ax.plot(t, v.T, color=col, lw=0.7, alpha=0.38, zorder=2)
        ax.plot(t, mean, color=col, lw=3.0, zorder=4)
        if len(tail):
            ax.plot(tail.index, tail.to_numpy(), color=house.INK, lw=2.0, marker="o", ms=3.5, zorder=5)
        ax.axvline(r["init"], color=house.MUTED, lw=1.0, ls=":", zorder=3)
        ax.set_title(run_label(r), color=house.INK)
        ax.grid(True, color=house.GRID, lw=0.8)
        ax.set_ylim(-top, top)
    for j in range(len(runs), nr * nc):  # three models: the panel above the empty one carries the dates
        axes.flat[j].set_axis_off()
        axes.flat[j - nc].tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Seasonal sd")
    # every other day from the first: a calendar locator restarts at each month, putting 31 Oct beside 1 Nov
    axes[0, 0].set_xticks(pd.date_range(pd.Timestamp(t_first).normalize(), t_last, freq="2D"))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    axes[0, 0].set_xlim(t_first, t_last + pd.Timedelta(hours=12))
    handles = [
        plt.Line2D([], [], color=house.INK2, lw=0.9, alpha=0.6, label="each member"),
        plt.Line2D([], [], color=house.INK2, lw=3.0, label="ensemble mean"),
        *[plt.Line2D([], [], color=house.INK, lw=2.0, marker="o", ms=3.5, label="ERA5, 00Z")] * bool(len(tail)),
        Patch(facecolor=NEUTRAL, alpha=0.55, label=f"neutral, within ±{THR:g} sd"),
        Patch(facecolor="white", edgecolor=house.EDGE, hatch="//", label="no demonstrated skill"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)
    fig.suptitle(f"{name}: each member's index every day and the ensemble mean, days 1–15", x=0.01, ha="left")
    return VF.save(fig, path)


def prob(sid: str, k: int, runs: list[dict], ind: dict, probs: dict, cal: dict, name: str, path: Path):
    """The calibrated chance of each phase by lead, one row per model, at the skilful leads only (a lead the run does
    not reach, a 12Z ECMWF run's day 15, left blank)."""
    plt, VF = _plt()
    import house
    from matplotlib.patches import Patch

    days = np.arange(1, len(T.LEADS) + 1)
    fig, axes = plt.subplots(len(runs), 1, figsize=house.size(15, 0.13 + 0.115 * len(runs)), sharex=True,
                             constrained_layout=True, squeeze=False)  # fmt: skip
    for ax, r in zip(axes[:, 0], runs, strict=True):
        mean = ind[r["key"]][2][:, k]
        for d, h in zip(days, T.LEADS, strict=True):
            c, pp = cal[r["key"]][sid, h], probs[r["key"]].get((sid, h))
            if not np.isfinite(mean[d - 1]):
                continue
            if not skilful(c) or pp is None:
                ax.bar(d, 1.0, width=0.86, facecolor="none", edgecolor=house.EDGE, hatch="//", lw=0)
                continue
            pn, pq = pp
            for y0, y1, col in ((0.0, pn, NEG), (pn, 1.0 - pq, NEUTRAL), (1.0 - pq, 1.0, POS)):
                ax.bar(d, y1 - y0, bottom=y0, width=0.86, color=col, edgecolor="white", lw=1.5)
            for y, p in ((pn / 2, pn), (1.0 - pq / 2, pq)):
                if p >= 0.12:
                    ax.text(d, y, f"{100 * p:.0f}", ha="center", va="center", color="white", fontweight="semibold")
        ax.set(ylim=(0, 1), yticks=[0, 0.5, 1], yticklabels=["0", "50", "100 %"], xlim=(0.4, len(days) + 0.6))
        ax.set_ylabel(f"{MODELS[r['key']]}\n{r['init']:%-d %b %HZ}", rotation=0, ha="right", va="center", labelpad=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xticks(days)
    axes[-1, 0].set_xlabel("Lead (days)")
    handles = [
        Patch(color=POS, label=f"positive, ≥ +{THR:g} sd"),
        Patch(color=NEUTRAL, label="neutral"),
        Patch(color=NEG, label=f"negative, ≤ −{THR:g} sd"),
        Patch(facecolor="white", edgecolor=house.EDGE, hatch="//", label="no demonstrated skill: no probability"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False)
    fig.suptitle(f"{name}: calibrated chance of each phase by lead (%)", x=0.01, ha="left")
    return VF.save(fig, path)


def skill(sid: str, runs: list[dict], cal: dict, name: str, t0: pd.Timestamp, path: Path):
    """The hindcast correlation of each model's ensemble-mean index with ERA5's by lead, and each cell's sample (runs):
    coloured where skilful, grey where tested and not, blank below MIN_RUNS runs."""
    plt, VF = _plt()
    import house
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    cmap = LinearSegmentedColormap.from_list("r", ["#e3eaf3", house.ACCENT])
    days = np.arange(1, len(T.LEADS) + 1)
    fig, ax = plt.subplots(figsize=house.size(15, 0.1 + 0.075 * len(runs)), constrained_layout=True)
    for i, r in enumerate(runs):
        for d, h in zip(days, T.LEADS, strict=True):
            c = cal[r["key"]][sid, h]
            if c is None:
                face, text, ink = "white", "–", house.MUTED
            else:
                x = min(max((c["r"] - SKILL_MIN) / (1 - SKILL_MIN), 0.0), 1.0)
                face = cmap(x) if skilful(c) else NOT_SIG
                ink = ink_on(face, house.INK)
                text = f"{c['r']:.2f}\nn {c['n']}"
            ax.add_patch(plt.Rectangle((d - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", lw=2))
            ax.text(d, i, text, ha="center", va="center", color=ink, fontsize=12.5, linespacing=1.25)
    ax.set(xlim=(0.5, len(days) + 0.5), ylim=(len(runs) - 0.5, -0.5), xticks=days, xlabel="Lead (days)")
    ax.set_yticks(range(len(runs)), [MODELS[r["key"]] for r in runs])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    handles = [
        Patch(color=cmap(0.6), label=f"skilful: r ≥ {SKILL_MIN:g}, significant at {P_MAX:g} and in the FDR test"),
        Patch(color=NOT_SIG, label="tested: no demonstrated skill"),
        Patch(facecolor="white", edgecolor=house.EDGE, label=f"–  fewer than {MIN_RUNS} runs"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle(
        f"{name}: hindcast correlation with ERA5 and runs (n) started within {WINDOW} days of {t0:%-d %B}",
        x=0.01,
        ha="left",
    )
    return VF.save(fig, path)


# ── the update ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def summary(ref: T.Ref, runs, ind, cal, probs, fc, obs, figs) -> dict:
    sample = {}
    for key, model in MODELS.items():
        f = fc[fc["model"] == model] if fc is not None else None
        if f is not None and len(f):
            sample[key] = {"runs": int(f["init"].nunique()), "first": f"{f['init'].min():%Y-%m-%d}",
                           "last": f"{f['init'].max():%Y-%m-%d}"}  # fmt: skip
    out = {
        "made": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "window_days": WINDOW,
        "gate": {"r_min": SKILL_MIN, "p_max": P_MAX, "fdr_q": FDR_Q, "min_runs": MIN_RUNS, "threshold_sd": THR},
        "era5_last": None if obs is None or obs.empty else f"{obs['valid'].max():%Y-%m-%d}",
        "hindcast": sample,
        "runs": {},
        "figures": figs,
    }
    for r in runs:
        _, v, mean = ind[r["key"]]
        e = {"cycle": r["cycle"], "init": f"{r['init']:%Y-%m-%dT%H}:00Z", "members": int(r["ok"].shape[0]),
             "lead_h": [int(h) for h in r["lead_h"]], "indices": {}}  # fmt: skip
        for k, s in enumerate(ref.ids):
            rows = []
            for j, h in enumerate(r["lead_h"]):
                c, pp = cal[r["key"]][s, int(h)], probs[r["key"]].get((s, int(h)))
                m = v[:, j, k][np.isfinite(v[:, j, k])]
                row = {"mean": _r(mean[j, k], 2)} | ({"p10": _r(np.percentile(m, 10), 2),
                                                     "p90": _r(np.percentile(m, 90), 2)} if m.size else {})  # fmt: skip
                if c:
                    row |= {"n": c["n"], "n_eff": _r(c["n_eff"], 1), "r": _r(c["r"], 2), "p": _r(c["p"], 4),
                            "fdr_ok": c["fdr_ok"], "skilful": c["skilful"]}  # fmt: skip
                if skilful(c) and pp is not None:
                    row |= {"p_neg": _r(pp[0]), "p_pos": _r(pp[1]), "cal_mean": _r(c["a"] * mean[j, k] + c["b"], 2)}
                rows.append(row)
            e["indices"][s] = rows
        out["runs"][r["key"]] = e
    return out


def cmd_run(a) -> int:
    runs = load(a.fields)
    if not runs:
        log("no teleconnection fields: nothing to draw")
        return 0
    ref = T.Ref(a.ref)
    fc = pd.read_parquet(a.table / T.FC) if a.table and (a.table / T.FC).exists() else None
    obs = pd.read_parquet(a.table / T.OBS) if a.table and (a.table / T.OBS).exists() else None
    if fc is None or obs is None:
        log("::notice::no hindcast table yet: every lead is drawn without demonstrated skill")
    ind = {r["key"]: member_indices(ref, r) for r in runs}
    cal = {r["key"]: calibrate(fc, obs, MODELS[r["key"]], r["init"]) for r in runs}
    gate(cal)
    probs = {}
    for r in runs:
        mean = ind[r["key"]][2]
        probs[r["key"]] = {
            (s, int(h)): (float(cal_cdf(-THR, c, mean[j, k])), float(1.0 - cal_cdf(THR, c, mean[j, k])))
            for k, s in enumerate(ref.ids)
            for j, h in enumerate(r["lead_h"])
            if skilful(c := cal[r["key"]][s, int(h)]) and np.isfinite(mean[j, k])
        }
    out = a.site / FIG
    out.mkdir(parents=True, exist_ok=True)
    t0 = max(r["init"] for r in runs)
    figs = []
    for k, s in enumerate(ref.ids):
        name = f"{s.upper()} ({ref.meta['patterns'][s]['name']})"
        figs.append(plume(s, k, runs, ind, cal, obs, name, out / f"{s}_plume.webp"))
        figs.append(prob(s, k, runs, ind, probs, cal, name, out / f"{s}_prob.webp"))
        figs.append(skill(s, runs, cal, name, t0, out / f"{s}_skill.webp"))
    js = a.site / JSON_OUT
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(summary(ref, runs, ind, cal, probs, fc, obs, figs), separators=(",", ":")) + "\n")
    passing = {
        f"{r['key']} {s}": [int(h) // 24 for h in r["lead_h"] if skilful(cal[r["key"]][s, int(h)])]
        for r in runs
        for s in ref.ids
    }
    log(f"teleconnections: {', '.join(r['key'] + ' ' + r['cycle'] for r in runs)}; skilful leads (days) {passing}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fields", type=Path, nargs="+", required=True, help="the models' telecon/<model>.npz (Capture)")
    r.add_argument("--ref", type=Path, required=True, help=f"{T.REF_NAME} (release {T.REF_TAG})")
    r.add_argument("--table", type=Path, help="the hindcast table's directory (telecon15.py hindcast)")
    r.add_argument("--site", type=Path, default=HERE.parents[1])
    a = ap.parse_args()
    return cmd_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
