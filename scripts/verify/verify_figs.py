#!/usr/bin/env python3
"""The figures and chart numbers of verification.html, built each night from the verification database, every score
with its uncertainty. The page draws its line charts itself, at the reader's own width, from the numbers; the
scorecards and the rain maps are static WebP (the page's controls pick one). figs.json lists what was written, for the
page, and what the database holds (database: each model's runs).

  acc   The 500 hPa height anomaly correlation against ERA5 (z500_acc.py), for each region and window of runs
        (acc_<region>_<window>.json): a time series for each of six leads, as NCEP EMC draws it, a running mean of the
        runs, and the window's mean by lead with its 95 % interval and each single run's paired difference from the
        IFS control and each ensemble mean's from the IFS-ENS mean.
  rain  Rain against Stage IV cell by cell (mverify_scores.rain_map): the ratio of the totals and the frequency
        biases at 1, 10 and 25 mm for each lead group, significant cells only (rain_<map>_<group>.webp), and what the
        analysis gave over the same days (rain_<map>_obs.webp); and for each window of runs and 24 h window, the
        equitable threat score by threshold and the fractions skill score by neighbourhood on the common grid
        (rainskill_<window>.json).
  scores  For each window of runs of the database: the scorecards (card_<window>_<region>_<model>_<reference>.webp),
        the RMSE and bias by lead of each variable in each region (lead_<window>_<variable>.json) and the error against
        pressure at a lead (profile_<window>_<quantity>.json).
  ensembles  The ensembles' scorecards (ecard_), their CRPS, spread and rank histograms by lead
        (ens_<window>_<variable>.json) and their reliability (rel_<window>_<variable>_<threshold>.json).

Thin samples are left out, not charted: a score by lead or by pressure needs MIN_PAIRS pairs of forecast and
observation from every model over the window, a reliability curve MIN_BIN events and MIN_BIN non-events, each of its
bins MIN_BIN cases.

Uncertainty. A window's runs are a sample: the same models over other weeks would score differently, and runs a day
apart share their weather, so they are not independent. Every interval is a 95 % Student t interval on the runs' own
scores, its variance inflated for lag-1 autocorrelation (the n runs count as n (1 - rho) / (1 + rho), t_crit), the
anomaly correlation averaged through Fisher's z = artanh(r) and turned back, and a score that cannot go below zero
(an RMSE, a CRPS, a ratio) taken on its logarithm. rho is one value per lead for a window's figures (pooled_rho): estimated from
a single series of ten runs it swings by +-0.3, and a high draw would cut the sample to about one run and widen an
interval tenfold. A difference from the reference is the
mean of the runs' paired differences (Geer 2016: unpaired intervals hide most of what separates two models) and is
significant when its paired test passes Benjamini-Hochberg at a 10 % false-discovery rate over every model and lead of
the figure.

    python verify_figs.py --derived db --clim clim_z500_1p5.npz --out fig
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import house
import mverify_scores as D
import z500_acc as Z

DPI = 220
COLORS = {"GFS": "#2c5f8a", "IFS": "#b3372f", "AIFS": "#7b4fb0", "GDPS": "#2e7d4f"}  # verification.html's COLORS
CENTRE = {"GEFS": "GFS", "IFS-ENS": "IFS", "AIFS-ENS": "AIFS", "GEPS": "GDPS"}  # an ensemble is drawn in its centre's
REF = "IFS"  # the reference of the differences: ECMWF's unperturbed physics run
WINDOWS = (30, 90, 365, None)  # newest runs of a window (None: the whole database), as the page's other scores
SERIES_DAYS = (1, 3, 5, 7, 10, 15)
SMOOTH_DAYS = 7  # the time series' running mean: a single run's score is mostly that day's weather
USEFUL = 0.6  # the anomaly correlation conventionally taken as the end of useful synoptic skill
FDR_Q = 0.10
RAIN_EXTENT = (-125.0, -66.5, 24.0, 50.0)
RAIN_EDGES = [0.4, 0.5, 1 / 1.5, 0.8, 1 / 1.1, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5]  # symmetric in log ratio, none white
RAIN_TITLES = {
    "ratio": "Total rain, forecast ÷ Stage IV",
    "fb1": "Days with 1 mm or more, forecast ÷ Stage IV",
    "fb10": "Days with 10 mm or more, forecast ÷ Stage IV",
    "fb25": "Days with 25 mm or more, forecast ÷ Stage IV",
}
RAIN_OBS = {
    "ratio": "mean 24 h rain",
    "fb1": "share of days with 1 mm or more",
    "fb10": "share of days with 10 mm or more",
    "fb25": "share of days with 25 mm or more",
}
NOT_SIG = "#dcd9d1"  # tested, not significant


def style() -> None:
    """The house style, a size up: the figures are read at the page's width, scorecards and maps most of all; the
    legends a size above the tick labels, since a tall figure is shown shrunk to fit the window."""
    house.apply()
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 17,
            "legend.title_fontsize": 17,
            "figure.titlesize": 19,
        }
    )


def save(fig, path: Path) -> str:
    fig.savefig(path, dpi=DPI, format="webp", pil_kwargs={"quality": 92, "method": 4})
    plt.close(fig)
    return path.name


def name_of(w) -> str:
    return str(w or "all")


def lag1(x: np.ndarray) -> float:
    """The lag-1 autocorrelation of one series (in run order); NaN for a constant one."""
    a = x - x.mean()
    den = (a * a).sum()
    return float((a[1:] * a[:-1]).sum() / den) if den > 0 else float("nan")


def pooled_rho(series) -> float:
    """The lag-1 autocorrelation that many run series share (each in run order, four runs or more): the mean of their
    estimates with Kendall's small-sample correction, E[r] = rho - (1 + 3 rho) / n, clipped to [0, 0.9]."""
    rs, ns = [], []
    for one in series:
        x = np.asarray(one, float)
        x = x[np.isfinite(x)]
        if len(x) >= 4 and np.isfinite(r := lag1(x)):
            rs.append(r)
            ns.append(len(x))
    if not rs:
        return 0.0
    n = float(np.mean(ns))
    return float(np.clip((np.mean(rs) + 1 / n) / (1 - 3 / n), 0.0, 0.9)) if n > 3 else 0.0


def n_eff(n: int, rho: float) -> float:
    """Runs left after allowing for a lag-1 autocorrelation rho."""
    return n * (1 - rho) / (1 + rho)


def t_crit(n: int, level: float) -> float:
    """The Student t quantile of an interval on n runs. The autocorrelation inflates the variance (n_eff) but not this
    quantile: on n_eff - 1 degrees of freedom, a window of ten runs whose leads are strongly autocorrelated (n_eff near
    two) would take the quantile of one degree of freedom, 12.7 for 95 % against 2.3."""
    return float(stats.t.ppf(0.5 + level / 2, n - 1))


def ci(x: np.ndarray, rho: float, level: float = 0.95) -> tuple[float, float, float]:
    """The mean of a run series and its interval, the variance inflated for the lag-1 autocorrelation rho
    (pooled_rho)."""
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    h = t_crit(len(x), level) * x.std(ddof=1) / np.sqrt(n_eff(len(x), rho))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def ratio_resid(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Each run's residual num - R den from a pooled score R = sum(num) / sum(den) (the runs both have)."""
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    num, den = num[ok], den[ok]
    return num - num.sum() / den.sum() * den if den.sum() > 0 else num[:0]


def ratio_ci(
    num: np.ndarray, den: np.ndarray, rho: float, level: float = 0.95, log: bool = False
) -> tuple[float, float, float]:
    """A pooled score sum(num) / sum(den) over runs (in run order) and its interval: the ratio estimator's variance
    from each run's residual num - R den (a run is a cluster of cases that share its weather), inflated for the lag-1
    autocorrelation rho (pooled_rho). With ``log`` the interval is taken on log R (delta method), for a score that
    cannot go below zero (a mean squared error, a CRPS, a ratio of two): it stays above zero and is wider above."""
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    num, den = num[ok], den[ok]
    if len(num) < 3 or den.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    r = num.sum() / den.sum()
    d = num - r * den
    n = len(d)
    se = np.sqrt(n / (n - 1) * (d**2).sum()) / den.sum() * np.sqrt(n / n_eff(n, rho))
    h = t_crit(n, level) * se
    if log and r > 0:
        return float(r), float(r * np.exp(-h / r)), float(r * np.exp(h / r))
    return float(r), float(r - h), float(r + h)


def run_rho(s: pd.DataFrame, num: str, den: str, by: list[str]) -> dict:
    """Per lead, the lag-1 autocorrelation (pooled_rho) of the runs' residuals num - R den over every series ``by``
    identifies in the window's per-run sums ``s`` (in run order)."""
    return {
        lead: pooled_rho(
            ratio_resid(d[num].to_numpy(float), d[den].to_numpy(float)) for _, d in g.groupby(by, observed=True)
        )
        for lead, g in s.groupby("lead_h")
    }


def num(x, digits: int = 4) -> float | None:
    """A value as the page's JSON holds it: ``digits`` significant figures, None where there is none."""
    return float(f"{x:.{digits}g}") if np.isfinite(x) else None


def nums(xs, digits: int = 4) -> list:
    return [num(x, digits) for x in xs]


def dump(data: dict, path: Path) -> str:
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    return path.name


def against(w: pd.DataFrame, m: str, ref: str, rho: float, root: bool) -> tuple[float, float, float]:
    """``m``'s score against ``ref``'s in % with its interval, on the runs both have (``w``: runs x models, each run's
    summed score, in run order): the ratio of the two pooled scores with each run a cluster (``ratio_ci`` on its
    logarithm), its square root for squared errors (``root``: an RMSE from the sums of squares)."""
    if m not in w or ref not in w or len(w := w[[m, ref]].dropna()) < 3:
        return float("nan"), float("nan"), float("nan")
    ratio = np.clip(ratio_ci(w[m].to_numpy(float), w[ref].to_numpy(float), rho, log=True), 0, None)
    return tuple(100 * ((np.sqrt(ratio) if root else ratio) - 1))


# --------------------------------------------------------------------------- anomaly correlation
def acc_window(t: pd.DataFrame, region: str, runs: int | None) -> tuple[pd.DataFrame, list, dict]:
    """The window's runs (the newest ``runs`` with every lead scored), the models with WINDOW_SHARE of them, and the
    scores of those models on the runs they all have, as (init x (model, lead)) anomaly correlations."""
    r = t[t["region"] == region]
    last = r["model"].map(lambda m: max(Z.DET_LEADS) if m in Z.DET else max(Z.E.LEADS))
    complete = r[r["lead_h"] == last][["init", "model"]]  # a run's model whose last lead is scored has them all
    inits = sorted(complete["init"].unique())
    inits = inits[-runs:] if runs else inits
    complete = complete[complete["init"].isin(inits)]
    models, used, left_out = D.window_runs(complete)
    wide = r[r["init"].isin(used) & r["model"].isin(models)].pivot_table(
        index="init", columns=["model", "lead_h"], values="acc"
    )
    return wide, models, {"left_out": left_out} | (D.run_span(used) if used else {"runs": 0})


def series_data(t: pd.DataFrame, region: str, day: int, start: pd.Timestamp) -> dict | None:
    """The anomaly correlation at one lead by the date it verifies, as a running mean of the runs (each run's own score
    swings with the day's weather far more than the models differ), and each model's mean over the period (through
    Fisher's z): over the days every model with WINDOW_SHARE of them has, so those compare on one sample; a model
    with fewer (GDPS and GEPS, kept by HPFX for their last weeks only) over its own days, which it counts."""
    r = t[(t["region"] == region) & (t["valid_time"] >= start) & (t["lead_h"] == 24 * day)]
    if r.empty:
        return None
    days = pd.date_range(r["valid_time"].min().normalize(), r["valid_time"].max().normalize(), freq="D")
    daily = {}
    for m in [*Z.DET, *Z.E.ENSEMBLES]:
        x = r[r["model"] == m].set_index("valid_time")["acc"].sort_index()
        if not x.empty:
            daily[m] = x.groupby(x.index.normalize()).mean().reindex(days)  # a missing run leaves a gap
    most = max(x.notna().sum() for x in daily.values())
    full = [m for m, x in daily.items() if x.notna().sum() >= D.WINDOW_SHARE * most]
    common = pd.concat([daily[m] for m in full], axis=1).notna().all(axis=1)
    models = {}
    for m, x in daily.items():
        on = x[common] if m in full else x.dropna()
        smooth = x.rolling(SMOOTH_DAYS, center=True, min_periods=SMOOTH_DAYS // 2 + 1).mean()
        models[m] = {
            "acc": nums(smooth.to_numpy()),
            "mean": num(np.tanh(np.arctanh(np.clip(on.dropna().to_numpy(), -0.9999, 0.9999)).mean())),
            "days": None if m in full else int(on.notna().sum()),
        }
    return {"first": f"{days[0]:%Y-%m-%d}", "shared": int(common.sum()), "models": models}


def lead_data(wide: pd.DataFrame, models: list[str]) -> dict:
    """The window's mean anomaly correlation by lead with its interval (through Fisher's z), and each model's paired
    difference from the IFS control (the single runs) or the IFS-ENS mean (the ensemble means), with its interval and
    whether its paired test passes Benjamini-Hochberg at FDR_Q over every model and lead of the two."""
    from mverify.significance import fdr, paired_test

    z = np.arctanh(wide.clip(-0.9999, 0.9999))
    rho = {lead: pooled_rho(z[c].to_numpy() for c in z.columns if c[1] == lead) for lead in {c[1] for c in z.columns}}
    leads = {m: sorted(lead for mm, lead in wide.columns if mm == m) for m in models}
    mean = {}
    for m in models:
        c, lo, hi = np.tanh([ci(z[(m, lead)].to_numpy(), rho[lead]) for lead in leads[m]]).T
        mean[m] = {"days": [int(lead) // 24 for lead in leads[m]], "acc": nums(c), "lo": nums(lo), "hi": nums(hi)}
    kinds = {"single": (REF, [m for m in models if m in Z.DET and m != REF]),
             "means": ("IFS-ENS", [m for m in models if m in CENTRE and m != "IFS-ENS"])}  # fmt: skip
    rows = []
    for kind, (ref, ms) in kinds.items():
        for m in ms if ref in models else []:
            for lead in leads[m]:
                if (ref, lead) in wide.columns:
                    d = (wide[(m, lead)] - wide[(ref, lead)]).to_numpy()
                    rows.append((kind, m, lead, *ci(d, rho[lead]), paired_test(d[np.isfinite(d)])))
    sig = fdr(np.array([row[-1] for row in rows]), FDR_Q) if rows else []
    diff = {kind: {"ref": ref, "models": {}} for kind, (ref, _) in kinds.items()}
    for (kind, m, lead, c, lo, hi, _p), s in zip(rows, sig, strict=True):
        one = diff[kind]["models"].setdefault(m, {"days": [], "d": [], "lo": [], "hi": [], "sig": []})
        for k, v in (("days", int(lead) // 24), ("d", num(c)), ("lo", num(lo)), ("hi", num(hi)), ("sig", int(s))):
            one[k].append(v)
    return {"mean": mean, "diff": diff}


def acc_figs(derived: Path, clim: Path, out: Path) -> dict:
    t = Z.table(derived, np.load(clim)["z500"])
    t["init"], t["valid_time"] = pd.to_datetime(t["init"]), pd.to_datetime(t["valid_time"])
    t = t[t["init"] >= pd.to_datetime(Z.START, format="%Y%m%d%H")]  # the verification database's runs, as the page's
    manifest = {}
    for region in Z.REGIONS:
        for w in WINDOWS:
            wide, models, span = acc_window(t, region, w)
            if len(wide) < 10 or (w and len(wide) < w * D.WINDOW_SHARE):
                continue
            key = f"{region}_{name_of(w)}"
            start = wide.index.min() + pd.Timedelta(days=1)
            data = {
                "window": span,
                "useful": USEFUL,
                "smooth_days": SMOOTH_DAYS,
                "series": {str(day): d for day in SERIES_DAYS if (d := series_data(t, region, day, start))},
                "lead": lead_data(wide, models),
            }
            manifest[key] = {"window": span, "data": dump(data, out / f"acc_{key}.json")}
            print(f"acc {key}: {span['runs']} runs, {', '.join(models)}", flush=True)
    return manifest


# --------------------------------------------------------------------------- rain
def rain_figs(derived: Path, out: Path) -> dict:
    """The rain maps: per map and lead group the four models side by side, significant cells only."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from mverify.precip.maps import LEAD_GROUPS

    path = derived / "precip_maps.npz"
    summary_path = D.OUT_DIR / "precip_maps.json"
    if not path.exists() or not summary_path.exists():
        return {}
    z, summary = np.load(path), json.loads(summary_path.read_text())
    models = [m for m in COLORS if m in summary["models"]]
    lat, lon = z["lat"], z["lon"]
    ye, xe = np.r_[lat - D.GRID / 2, lat[-1] + D.GRID / 2], np.r_[lon - D.GRID / 2, lon[-1] + D.GRID / 2]
    proj = ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45))
    pc = ccrs.PlateCarree()
    states = cfeature.STATES.with_scale("50m")
    cmap = ListedColormap(plt.get_cmap("BrBG")(np.linspace(0.04, 0.96, len(RAIN_EDGES) + 1)))
    cmap.set_extremes(under=cmap(0), over=cmap(cmap.N - 1))
    norm = BoundaryNorm(RAIN_EDGES, cmap.N, extend="both")
    grey = ListedColormap([NOT_SIG])
    w = summary["window"]

    def base(ax):
        ax.set_extent(RAIN_EXTENT, pc)
        ax.add_feature(states, edgecolor="#8c887f", facecolor="none", lw=0.45)
        ax.coastlines("50m", color="#5d5a54", lw=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#5d5a54", lw=0.6)
        ax.set_frame_on(False)

    manifest = {}
    for metric, title in RAIN_TITLES.items():
        for g, (glabel, _, _) in enumerate(LEAD_GROUPS):
            fig, axes = plt.subplots(2, 2, figsize=house.size(15, 0.7), subplot_kw={"projection": proj})
            fig.subplots_adjust(left=0.01, right=0.99, top=0.865, bottom=0.13, wspace=0.02, hspace=0.12)
            for ax, m in zip(axes.flat, models, strict=False):
                v, s = z[f"{m}/{metric}"][g], z[f"{m}/{metric}_sig"][g]
                base(ax)
                tested = np.isfinite(v)
                ax.pcolormesh(xe, ye, np.where(tested & ~s, 1.0, np.nan), cmap=grey, transform=pc, rasterized=True)
                ax.pcolormesh(xe, ye, np.where(s, v, np.nan), cmap=cmap, norm=norm, transform=pc, rasterized=True)
                k = summary["models"][m]["maps"][metric][g]
                share = k["significant"] / k["tested"] if k["tested"] else 0
                ax.set_title(f"{m}: {summary['models'][m]['runs']} runs, {share:.0%} of {k['tested']:,} cells significant",
                             loc="left", fontsize=14)  # fmt: skip
            for ax in axes.flat[len(models) :]:
                ax.set_visible(False)
            cax = fig.add_axes((0.2, 0.065, 0.6, 0.024))
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                              extend="both", ticks=RAIN_EDGES)  # fmt: skip
            cb.ax.set_xticklabels([f"{e:.2g}" for e in RAIN_EDGES])
            cb.set_label("forecast ÷ Stage IV  (brown: too little, green: too much; grey: tested, within the noise)")
            fig.suptitle(f"{title}, {glabel}", x=0.01, y=0.985, ha="left")
            first, last = (pd.to_datetime(w[k], format="%Y%m%d%H") for k in ("first", "last"))
            fig.text(0.01, 0.93, f"00Z runs {first:%-d %b} to {last:%-d %b %Y} on the 0.5° common grid · cells "
                     "significant at a 10 % false-discovery rate", color=house.INK2, fontsize=15)  # fmt: skip
            manifest[f"{metric}_{g}"] = save(fig, out / f"rain_{metric}_{g}.webp")
        # what the analysis gave over the days of the maps (the reference model's days, every lead group alike)
        ref = REF if REF in models else models[0]
        obs = z[f"{ref}/{metric}_obs"][0]
        fig, ax = plt.subplots(figsize=house.size(10, 0.62), subplot_kw={"projection": proj})
        fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.14)
        base(ax)
        im = ax.pcolormesh(xe, ye, obs, cmap="YlGnBu", vmin=0, vmax=np.nanpercentile(obs, 98), transform=pc,
                           rasterized=True)  # fmt: skip
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, aspect=40)
        cb.set_label("mm a day" if metric == "ratio" else "share of days")
        fig.suptitle(f"Stage IV over the same days: {RAIN_OBS[metric]}", x=0.01, ha="left")
        manifest[f"{metric}_obs"] = save(fig, out / f"rain_{metric}_obs.webp")
        print(f"rain {metric}: maps drawn", flush=True)
    return {"maps": manifest, "groups": [g[0] for g in LEAD_GROUPS], "window": w, "models": models}


RAIN_SKILL_THRESHOLDS = (1.0, 10.0, 25.0, 50.0)  # mm in 24 h, as the nightly's Stage IV scores
RAIN_SKILL_WINDOWS = (10, 30, 90, 365)  # the newest runs of the database; its rain files reach back a year (RAIN_RUNS)
RAIN_SKILL_MIN = 100 * np.cos(np.deg2rad(40))  # cells over a threshold in the analysis (summed over the runs, each
# weighted by the cosine of its latitude: 100 cells at 40N) before a score is drawn


def jack_ci(tables: np.ndarray, score, rho: float, level: float = 0.95) -> tuple[float, float, float]:
    """A score of runs' tables summed (``tables``: runs x counts, in run order) and its interval: the leave-one-run-out
    jackknife's variance (a run is a cluster of cells that share its weather, as in ``ratio_ci``), the sample cut for
    the lag-1 autocorrelation rho (pooled_rho)."""
    n = len(tables)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    total = tables.sum(0)
    full = score(total)
    pseudo = n * full - (n - 1) * np.array([score(total - t) for t in tables])
    h = t_crit(n, level) * pseudo.std(ddof=1) / np.sqrt(n_eff(n, rho))
    return float(full), float(full - h), float(full + h)


def ets(t: np.ndarray) -> float:
    """The equitable threat score of summed (hits, misses, false alarms, cells) (mverify.precip.scores.finish)."""
    h, mi, fa, w = t
    rh = (h + mi) * (h + fa) / w
    with np.errstate(invalid="ignore", divide="ignore"):
        return (h - rh) / (h + mi + fa - rh)


def rain_sums(paths: list[Path]) -> pd.DataFrame:
    """Each run's Stage IV sums on the common grid (mverify.precip.scores.case_sums, cosine-of-latitude weights):
    per model, 24 h window and threshold the hits, misses, false alarms and the fractions skill score's sums, over the
    land cells (as the rain maps') that every model of the run has."""
    from mverify.precip.regrid import EARTH_KM_PER_DEG
    from mverify.precip.scores import FSS_KM, case_sums

    edge = pd.read_parquet(paths[-1], columns=["lat", "lon"])
    lat0, lon0 = (float(edge[c].min()) - D.RAIN_MARGIN for c in ("lat", "lon"))
    frame = tuple(
        int(np.rint((edge[c].max() + D.RAIN_MARGIN - o) / D.GRID)) + 1 for c, o in (("lat", lat0), ("lon", lon0))
    )
    mask = np.load(Path(__file__).resolve().parent / "reference" / "land_0p5.npz")
    lats = lat0 + D.GRID * np.arange(frame[0])
    rows = np.rint((lats + 90) / D.GRID).astype(int)
    cols = np.rint(((lon0 + D.GRID * np.arange(frame[1]) + 180) % 360) / D.GRID).astype(int)
    land = mask["land"][np.ix_(rows, cols)] >= D.RAIN_LAND
    weights = np.repeat(np.cos(np.deg2rad(lats))[:, None], frame[1], 1)
    out = []
    for f in paths:
        r = pd.read_parquet(f, columns=["model", "init", "lead_h", "lat", "lon", "fcst", "obs"])
        r["j"], r["i"] = (np.rint((r[c] - o) / D.GRID).astype(int) for c, o in (("lat", lat0), ("lon", lon0)))
        r = r[r["j"].between(0, frame[0] - 1) & r["i"].between(0, frame[1] - 1)]
        models = sorted(r["model"].unique())
        for lead, g in r.groupby("lead_h"):
            obs = np.full(frame, np.nan)
            obs[g["j"], g["i"]] = g["obs"]
            fc = {m: np.full(frame, np.nan) for m in models}
            for m, gm in g.groupby("model", observed=True):
                fc[m][gm["j"], gm["i"]] = gm["fcst"]
            ok = land & np.isfinite(obs) & np.logical_and.reduce([np.isfinite(x) for x in fc.values()])
            for m, x in fc.items():
                sums = case_sums(np.where(ok, x, np.nan), np.where(ok, obs, np.nan), ok, weights,
                                 D.GRID * EARTH_KM_PER_DEG, False, RAIN_SKILL_THRESHOLDS, FSS_KM)  # fmt: skip
                out.extend({"init": g["init"].iloc[0], "model": m, "lead_h": int(lead), "threshold": t, **sums[t]}
                           for t in RAIN_SKILL_THRESHOLDS)  # fmt: skip
    return pd.DataFrame(out)


def rain_skill_figs(derived: Path, out: Path) -> dict:
    """For each window of the database's newest runs and each 24 h window ending 12Z: the equitable threat score by
    threshold and the fractions skill score by neighbourhood width at each threshold, every model on the same runs
    and cells, each with its 95 % interval (``jack_ci``, ``ratio_ci``)."""
    from mverify.precip.scores import FSS_KM

    wet = sorted(derived.glob("rain_*.parquet"))
    if len(wet) < min(RAIN_SKILL_WINDOWS):
        return {}
    sums = rain_sums(wet[-max(RAIN_SKILL_WINDOWS) :]).sort_values("init")
    manifest = {}
    for runs in RAIN_SKILL_WINDOWS:
        inits = sorted(sums["init"].unique())
        if len(inits) < runs:
            continue
        models, used, left_out = D.window_runs(sums[sums["init"].isin(inits[-runs:])])
        s = sums[sums["model"].isin(models) & sums["init"].isin(used)]
        span = D.run_span(used) | {"left_out": left_out}
        data = {}
        for lead, g in s.groupby("lead_h"):
            # one autocorrelation for the lead's intervals: of the runs' hit rates, over every model and threshold
            rho = pooled_rho(ratio_resid(d["hits"].to_numpy(float), (d["hits"] + d["misses"]).to_numpy(float))
                             for _, d in g.groupby(["model", "threshold"]))  # fmt: skip
            score = {}  # (model, threshold) -> ETS and its interval, and per width the FSS and its interval
            for (m, t), d in g.groupby(["model", "threshold"]):
                if (d["hits"] + d["misses"]).sum() < RAIN_SKILL_MIN:
                    continue
                fss = [1 - np.array(ratio_ci(d[f"fss_num_{km}"].to_numpy(float), d[f"fss_den_{km}"].to_numpy(float),
                                             rho))[[0, 2, 1]] for km in FSS_KM]  # fmt: skip
                score[m, t] = (jack_ci(d[["hits", "misses", "false_alarms", "w"]].to_numpy(float), ets, rho), fss)
            data[str(int(lead))] = {
                m: {
                    f"{t:g}": {
                        "ets": nums(np.clip(score[m, t][0], -1 / 3, 1)),
                        "fss": [nums(f) for f in score[m, t][1]],
                    }
                    for t in RAIN_SKILL_THRESHOLDS
                    if (m, t) in score
                }
                for m in models
            }
        manifest[str(runs)] = {"window": span, "models": models, "leads": sorted(int(x) for x in s["lead_h"].unique()),
                               "thresholds": [f"{t:g}" for t in RAIN_SKILL_THRESHOLDS],
                               "data": dump({"km": list(FSS_KM), "leads": data}, out / f"rainskill_{runs}.json")}  # fmt: skip
        print(f"rain skill {runs}: {span['runs']} runs, {len(data)} leads", flush=True)
    return manifest


def database(derived: Path, have: Path | None) -> dict:
    """What the verification database holds, for the page's pointer to it: each model's runs (from the runs' sums,
    which every model of a run is in), each ensemble's (from the names of its station cases, ens-sfc-<m>_<init>: a
    run's sums wait for every ensemble that can still come) and the 500 hPa anomaly correlations'."""
    out, names = {}, have.read_text().split() if have and have.exists() else []
    if files := sorted(derived.glob("stats_*.parquet")):
        t = pd.concat([pd.read_parquet(f, columns=["model", "init"]).drop_duplicates() for f in files])
        out["models"] = {str(m): D.run_span(sorted(g["init"].unique())) for m, g in t.groupby("model", observed=True)}
    ens = {}
    for name, code in Z.E.ENSEMBLES.items():
        if inits := sorted(n[len(f"ens-sfc-{code}_") : -8] for n in names if n.startswith(f"ens-sfc-{code}_")):
            ens[name] = D.run_span([pd.to_datetime(i, format="%Y%m%d%H") for i in inits])
    if ens:
        out["ensembles"] = ens
    if acc := sorted(f.stem.split("_")[1] for f in derived.glob("acc_*.parquet")):
        out["z500"] = D.run_span([pd.to_datetime(i, format="%Y%m%d%H") for i in acc])
    return out


# --------------------------------------------------------------------------- the deterministic models
CARD_ROWS = ["t2m", "td2m", "msl", "wind10", "z850", "z500", "z250", "t850", "t500", "t250", "wind850", "wind500",
             "wind250", "q850", "q700"]  # fmt: skip  # the page's main rows (verification.html MAIN)
LEAD_VARS = [*CARD_ROWS, "pr24"]
CARD_REGIONS = ("NH 20-90N", "SH 20-90S", "Tropics 20S-20N", "Europe", "North America", "CONUS")
DET_WINDOWS = (10, 30, 90, 365, None)  # the page's run switch (its nightly ten are the database's newest ten here)
SURFACE = {"t2m", "td2m", "msl", "wind10", "pr24"}  # scored at the GHCNh stations, the rest against radiosondes
BETTER, WORSE = (44, 110, 158), (180, 50, 31)  # verification.html's scorecard colours
SIG_P = 0.05  # the scorecard's level, as on the page
QTY = {"z": "Geopotential height", "t": "Temperature", "wind": "Vector wind", "q": "Specific humidity"}
PROFILE_DAYS = (1, 3, 5, 7, 10)
UNITS = {"degC": "°C", "m s-1": "m/s", "g kg-1": "g/kg"}  # the scores' units as the page writes them


def slug(text: str) -> str:
    return "".join(c for c in text.split(" 20")[0] if c.isalnum())


def short(v: str) -> str:
    named = {
        "t2m": "2 m T",
        "td2m": "2 m Td",
        "msl": "MSLP",
        "wind10": "10 m wind",
        "ws10": "10 m speed",
        "pr24": "24 h rain",
        "tp": "24 h rain (Stage IV)",
    }
    if v in named:
        return named[v]
    q = next(q for q in ("wind", "z", "t", "q") if v.startswith(q))
    return f"{dict(wind='Wind', z='Z', t='T', q='Q')[q]} {v[len(q) :]}"


def region_labels() -> dict[str, str]:
    """The sums' region codes (nh, north_america ...) as the page names them (NH 20-90N, North America ...), from the
    scorecards' own keys (network:variable:code)."""
    out = {}
    for f in sorted(D.OUT_DIR.glob("*scores*.json")):
        for r in json.loads(f.read_text()).get("rows", []):
            out.setdefault(r["key"].split(":")[-1], r["region"])
    return out


def det_window(derived: Path, runs: int | None) -> tuple[pd.DataFrame, list, dict] | None:
    """The per-run sums of the window's runs that every model it scores has (``mverify_scores.window_card``'s)."""
    inits = sorted(f.stem.split("_")[1] for f in derived.glob("stats_*.parquet"))
    use = inits[-runs:] if runs else inits
    if not use or len(use) < (runs or 366):  # a window once the database holds it (mverify_scores.windows)
        return None
    stats = pd.read_parquet(
        [derived / f"stats_{i}.parquet" for i in use], columns=[*D.STAT_BY, "n", "se", "sse", "init"]
    )
    stats = stats.astype(dict.fromkeys(D.STAT_BY[:4], str))
    stats = stats[stats["network"] == np.where(stats["variable"].isin(SURFACE), "GHCNH", "RAOB")]
    labels = region_labels()
    stats["region"] = stats["region"].map(lambda c: labels.get(c, c))
    models, used, left_out = D.window_runs(stats)
    if not used:
        return None
    s = stats[stats["model"].isin(models) & stats["init"].isin(used)].sort_values("init")  # run order: ratio_ci
    return s.assign(mse=s["sse"] / s["n"], bias=s["se"] / s["n"]), models, D.run_span(used) | {"left_out": left_out}


def pooled(d: pd.DataFrame, rho: tuple[float, float]) -> tuple:
    """RMSE and bias of one model's per-run sums (in run order), each with its interval (``ratio_ci``; rho, the lead's
    autocorrelations of the squared and the plain errors, det_rho)."""
    mse = ratio_ci(d["sse"].to_numpy(float), d["n"].to_numpy(float), rho[0], log=True)
    return (*np.sqrt(np.clip(mse, 0, None)), *ratio_ci(d["se"].to_numpy(float), d["n"].to_numpy(float), rho[1]))


def det_rho(s: pd.DataFrame) -> dict:
    """Per lead, the window's autocorrelations of the runs' squared and plain errors (run_rho over every model,
    variable and region): one pair for every interval of the window's figures at that lead."""
    by = ["model", "variable", "region"]
    sq, plain = run_rho(s, "sse", "n", by), run_rho(s, "se", "n", by)
    return {lead: (sq[lead], plain[lead]) for lead in sq}


def draw_card(s: pd.DataFrame, region: str, a: str, b: str, span: dict, path: Path) -> str | None:
    """``a`` against ``b`` in one region: the RMSE difference in %, in the cells where the paired test on the runs'
    mean squared errors passes at SIG_P; a dot where it does not."""
    g = s[(s["region"] == region) & s["variable"].isin(CARD_ROWS) & (s["lead_h"] % 24 == 0) & (s["lead_h"] > 0)]
    if g.empty:
        return None
    pooled = g.groupby(["variable", "lead_h", "model"])[["n", "sse"]].sum()
    rmse = np.sqrt(pooled["sse"] / pooled["n"]).unstack("model")
    per = g.pivot_table(index=["variable", "lead_h", "init"], columns="model", values="mse")
    title = f"{a} against {b}, {region}: RMSE difference, %"
    return card_figure(rmse, per, CARD_ROWS, a, b, title, span, path)


def card_figure(score: pd.DataFrame, per: pd.DataFrame, order: list[str], a: str, b: str, title: str, span: dict,
                path: Path) -> str:  # fmt: skip
    """A scorecard: rows of variables, columns of days, the difference of ``a``'s score from ``b``'s in % where the
    paired test on the runs' own scores (``per``) passes at SIG_P, a dot where it does not."""
    from mverify.significance import paired_test

    rows = [v for v in order if v in score.index.get_level_values(0)]
    days = sorted({int(h) // 24 for h in score.index.get_level_values(1)})
    groups = [
        "sfc" if v in SURFACE | ENS_SURFACE else next(q for q in ("wind", "z", "t", "q") if v.startswith(q))
        for v in rows
    ]
    gaps = np.cumsum([0] + [0.35 * (g1 != g0) for g0, g1 in itertools.pairwise(groups)])
    fig = plt.figure(figsize=(1.6 + 0.78 * len(days), 1.55 + 0.48 * len(rows) + gaps[-1] * 0.48))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-1.75, len(days) + 0.1)
    ax.set_ylim(len(rows) + gaps[-1] + 0.15, -2.6)
    ax.axis("off")
    for j, d in enumerate(days):
        ax.text(j + 0.5, -0.35, str(d), ha="center", va="bottom", fontsize=14, color=house.INK2)
    ax.text(-0.15, -0.35, "Day", ha="right", va="bottom", fontsize=14, color=house.INK2)
    for i, (v, gap) in enumerate(zip(rows, gaps, strict=True)):
        y = i + gap
        ax.text(-0.15, y + 0.5, short(v), ha="right", va="center", fontsize=14.5, color=house.INK)
        for j, d in enumerate(days):
            key = (v, 24 * d)
            if key not in score.index or not np.isfinite(score.loc[key, [a, b]]).all():
                continue
            pct = 100 * (score.loc[key, a] - score.loc[key, b]) / score.loc[key, b]
            diff = (per.loc[key, a] - per.loc[key, b]).dropna().to_numpy()
            sig = paired_test(diff) < SIG_P
            if sig:
                k = 0.35 + 0.65 * min(1.0, abs(pct) / 20)
                rgb = [(k * c + (1 - k) * 255) / 255 for c in (BETTER if pct < 0 else WORSE)]
                ink = "white" if k > 0.55 else house.INK
                label = f"{pct:+.0f}" if abs(pct) >= 1 else f"{pct:+.1f}"
                ax.add_patch(plt.Rectangle((j + 0.04, y + 0.05), 0.92, 0.9, color=rgb, lw=0))
                ax.text(j + 0.5, y + 0.52, label.replace("-", "−"), ha="center", va="center", fontsize=13.5,
                        color=ink, weight="semibold")  # fmt: skip
            else:
                ax.add_patch(plt.Rectangle((j + 0.04, y + 0.05), 0.92, 0.9, color="#f3f2ee", lw=0))
                ax.plot(j + 0.5, y + 0.5, "o", ms=2.5, color=house.MUTED)
    ax.text(-1.7, -2.15, title, fontsize=17, weight="semibold",
            va="center", color=house.INK)  # fmt: skip
    ax.text(-1.7, -1.45, f"blue: {a} smaller error · red: larger · dot: no significant difference · {span['runs']} runs",
            fontsize=15, va="center", color=house.INK2)  # fmt: skip
    return save(fig, path)


SCORE_KEYS = ("rmse", "rlo", "rhi", "bias", "blo", "bhi")
MIN_PAIRS = 100  # pairs of forecast and observation a model needs over the window before a score of it is published


def sampled(g: pd.DataFrame, by: str) -> pd.DataFrame:
    """``g`` at the values of the column ``by`` (leads, levels) where every model has MIN_PAIRS pairs over the window:
    a region with a handful of reports (24 h gauge totals over the US) would otherwise chart one station's errors."""
    n = g.groupby([by, "model"], observed=True)["n"].sum().unstack("model")
    return g[g[by].isin(n.index[n.min(axis=1) >= MIN_PAIRS])]


def along(g: pd.DataFrame, by: str, at: list, models: list[str], rho) -> dict:
    """Each model's RMSE and bias with their 95 % intervals (``pooled``) at each value ``at`` of the column ``by`` (a
    variable's leads, a profile's levels), from one variable or quantity and region's per-run sums ``g``; and for
    every model but the IFS control its RMSE against the control's in % (``against``). ``rho(x)`` gives the pair of
    autocorrelations at x (det_rho)."""
    sse = {x: d.pivot_table(index="init", columns="model", values="sse") for x, d in g.groupby(by)}
    blank = (float("nan"),) * len(SCORE_KEYS)
    out = {}
    for m in models:
        rows = {x: pooled(d, rho(x)) for x, d in g[g["model"] == m].groupby(by)}
        if not rows:
            continue
        out[m] = dict(zip(SCORE_KEYS, map(nums, zip(*(rows.get(x, blank) for x in at), strict=True)), strict=True))
        if REF in models and m != REF:
            pct = [against(sse[x], m, REF, rho(x)[0], root=True) if x in sse else blank[:3] for x in at]
            out[m] |= dict(zip(("pct", "plo", "phi"), map(nums, zip(*pct, strict=True)), strict=True))
    return out


def det_data(s: pd.DataFrame, models: list[str], meta: dict, rho: dict, name: str, out: Path) -> dict:
    """The window's numbers for the page's charts: per variable the error by lead in each region (the charts of the
    error by lead and of the headline variables), per quantity the error by pressure at each of PROFILE_DAYS."""
    lead, profile = {}, {}
    for v in LEAD_VARS:
        label, unit = meta.get(v, (short(v), ""))
        regions = {}
        for region, g in s[s["variable"] == v].groupby("region"):
            if (g := sampled(g, "lead_h")).empty:
                continue
            hours = sorted(int(h) for h in g["lead_h"].unique())
            regions[region] = {"leads": hours, "models": along(g, "lead_h", hours, models, rho.get)}
        if regions:
            lead[v] = dump({"label": label, "unit": UNITS.get(unit, unit), "regions": regions},
                           out / f"lead_{name}_{v}.json")  # fmt: skip
    upper = s[s["network"] == "RAOB"]
    for qty in QTY:
        unit = next((u for v, (_, u) in meta.items() if v.removeprefix(qty).isdigit()), "")
        g = upper[upper["variable"].str.fullmatch(rf"{qty}\d+") & upper["lead_h"].isin([24 * d for d in PROFILE_DAYS])]
        g = g.assign(p=g["variable"].str[len(qty) :].astype(int))
        regions = {}
        for (region, hour), d in g.groupby(["region", "lead_h"]):
            if (d := sampled(d, "p")).empty:
                continue
            levels = sorted((int(p) for p in d["p"].unique()), reverse=True)
            regions.setdefault(region, {})[str(int(hour) // 24)] = {
                "levels": levels,
                "models": along(d, "p", levels, models, lambda _p, pair=rho[hour]: pair),
            }
        if regions:
            profile[qty] = dump({"unit": UNITS.get(unit, unit), "regions": regions}, out / f"profile_{name}_{qty}.json")
    return {"lead": lead, "profile": profile}


def det_figs(derived: Path, out: Path) -> dict:
    """The scorecards of each window of runs of the database, and the numbers of its error by lead and by pressure."""
    meta = {}
    for f in sorted(D.OUT_DIR.glob("scores*.json")):
        for r in json.loads(f.read_text())["rows"]:
            meta.setdefault(r["var"], (r["label"], r["unit"]))
    manifest = {}
    for w in DET_WINDOWS:
        if (found := det_window(derived, w)) is None:
            continue
        s, models, span = found
        rho = det_rho(s)
        name, cards = name_of(w), {}
        for region in CARD_REGIONS:
            for a in models:
                for b in models:
                    if a != b and (
                        f := draw_card(s, region, a, b, span, out / f"card_{name}_{slug(region)}_{a}_{b}.webp")
                    ):
                        cards.setdefault(region, {})[f"{a}|{b}"] = f
        data = det_data(s, models, meta, rho, name, out)
        manifest[name] = {"window": span, "models": models, "card": cards, "data": data}
        print(f"scores {name}: {span['runs']} runs, {len(cards)} scorecard regions, {len(data['lead'])} variables, "
              f"{len(data['profile'])} quantities; autocorrelation of the squared errors by day: "
              f"{' '.join(f'{rho[h][0]:.2f}' for h in sorted(rho) if h % 24 == 0)}", flush=True)  # fmt: skip
    return manifest


# --------------------------------------------------------------------------- the ensembles
ENS_ROWS = ["t2m", "td2m", "msl", "ws10", "wind10", "pr24", "z500", "t850", "wind850", "wind250"]  # the page's ORDER
ENS_SURFACE = {"t2m", "td2m", "msl", "ws10", "wind10", "pr24"}
ENS_REGIONS = ("Global", *CARD_REGIONS)
ENS_WINDOWS = (30, 90, 365, None)
MIN_BIN = 30  # cases a reliability bin needs before it is published; and the events, and the non-events, a curve needs


def ens_window(derived: Path, runs: int | None) -> tuple[pd.DataFrame, pd.DataFrame, list, dict] | None:
    """The window's per-run sums and reliability counts on the runs that every ensemble it scores has (``ens_verify
    .ens_card``'s), with the wind at each level from its components."""
    inits = sorted(f.stem.split("_")[1] for f in derived.glob("ens-stats_*.parquet"))
    use = inits[-runs:] if runs else inits
    if not use or len(use) < (runs or 366):  # a window once the database holds it (ens_verify.windows)
        return None
    stats = pd.read_parquet([derived / f"ens-stats_{i}.parquet" for i in use])
    rel = pd.read_parquet([derived / f"ens-rel_{i}.parquet" for i in use])
    models, used, left_out = D.window_runs(stats)
    if not used:
        return None
    s = Z.E.with_wind(stats[stats["model"].isin(models) & stats["init"].isin(used)])
    s = s[s["network"] == np.where(s["variable"].isin(ENS_SURFACE), "GHCNH", "RAOB")]
    r = rel[rel["model"].isin(models) & rel["init"].isin(used)].astype(dict.fromkeys(D.STAT_BY[:4], str))
    labels = region_labels()
    s["region"], r["region"] = (x["region"].map(lambda c: labels.get(c, c)) for x in (s, r))
    return s.sort_values("init"), r.sort_values("init"), models, D.run_span(used) | {"left_out": left_out}


def ens_card(s: pd.DataFrame, region: str, a: str, b: str, span: dict, path: Path) -> str | None:
    g = s[(s["region"] == region) & s["variable"].isin(ENS_ROWS) & (s["lead_h"] % 24 == 0) & (s["lead_h"] > 0)]
    if g.empty:
        return None
    pooled = g.groupby(["variable", "lead_h", "model"])[["n", "crps"]].sum()
    crps = (pooled["crps"] / pooled["n"]).unstack("model")
    per = g.assign(c=g["crps"] / g["n"]).pivot_table(index=["variable", "lead_h", "init"], columns="model", values="c")
    title = f"{a} against {b}, {region}: CRPS difference, %"
    return card_figure(crps, per, ENS_ROWS, a, b, title, span, path)


def group_rho(rho: dict, a: int, b: int) -> float:
    """A lead group's autocorrelation: the mean of its leads' (run_rho)."""
    vals = [r for lead, r in rho.items() if a <= lead <= b]
    return float(np.mean(vals)) if vals else 0.0


def ens_lead_data(g: pd.DataFrame, models: list, rho: dict) -> dict:
    """One variable and region: each ensemble's CRPS and the spread of its members over the error of its mean by lead,
    with their 95 % intervals, and for every ensemble but the IFS-ENS its CRPS against the IFS-ENS's in %
    (``against``)."""
    hours = sorted(int(h) for h in g["lead_h"].unique())
    crps = {h: d.pivot_table(index="init", columns="model", values="crps") for h, d in g.groupby("lead_h")}
    out = {"leads": hours, "models": {}}
    for m in models:
        rows = {}
        for lead, d in g[g["model"] == m].groupby("lead_h"):
            var = ratio_ci((d["svar"] + d["sov"]).to_numpy(float), d["sse"].to_numpy(float), rho[lead], log=True)
            rows[lead] = (*ratio_ci(d["crps"].to_numpy(float), d["n"].to_numpy(float), rho[lead], log=True),
                          *np.sqrt(np.clip(var, 0, None)))  # fmt: skip
        if not rows:
            continue
        cols = zip(*(rows.get(h, (float("nan"),) * 6) for h in hours), strict=True)
        out["models"][m] = dict(zip(("crps", "clo", "chi", "ratio", "rlo", "rhi"), map(nums, cols), strict=True))
        if "IFS-ENS" in models and m != "IFS-ENS":
            pct = [against(crps[h], m, "IFS-ENS", rho[h], root=False) for h in hours]
            out["models"][m] |= dict(zip(("pct", "plo", "phi"), map(nums, zip(*pct, strict=True)), strict=True))
    return out


def ens_rank_data(g: pd.DataFrame, models: list, rho: dict) -> dict:
    """One variable and region's rank histograms for each lead group: the share of the observations in each tenth of
    the ensemble's ranks, with its 95 % interval; a tenth each is calibrated."""
    ranks = [c for c in g.columns if c.startswith("rank_")]
    g = g.dropna(subset=ranks, how="all")
    out = {}
    for label, a, b in Z.E.LEAD_GROUPS:
        gg = g[g["lead_h"].between(a, b)]
        for m in models:
            per = gg[gg["model"] == m].groupby("init")[ranks].sum()
            if per.empty:
                continue
            tot = per.sum(axis=1).to_numpy(float)
            f = np.array([ratio_ci(per[c].to_numpy(float), tot, group_rho(rho, a, b)) for c in ranks])
            out.setdefault(label, {})[m] = {"f": nums(f[:, 0]), "lo": nums(f[:, 1]), "hi": nums(f[:, 2])}
    return out


def ens_rel_data(g: pd.DataFrame, models: list, rho: dict) -> dict:
    """One variable, threshold and region's reliability for each lead group: the observed frequency of the event in
    each forecast probability bin of MIN_BIN cases or more, with its 95 % interval, against the bin's mean forecast
    probability, and the Brier skill score against the sample's base rate. A model's curve needs MIN_BIN events and
    MIN_BIN non-events in the group: from fewer, the frequencies and the skill score are a few stations' luck. A bin
    where the event never came, or always did, has no spread between runs to take an interval from, so it has none."""
    out = {}
    for label, a, b in Z.E.LEAD_GROUPS:
        gg = g[g["lead_h"].between(a, b)]
        for m in models:
            gm = gg[gg["model"] == m]
            if gm.empty:
                continue
            n, o = gm["n"].sum(), gm["o"].sum()
            if min(o, n - o) < MIN_BIN:
                continue
            bss = 1 - gm["bs"].sum() / n / (o / n * (1 - o / n))
            pts = []
            for _, d in gm.groupby("bin"):
                if d["n"].sum() >= MIN_BIN:
                    per = d.groupby("init")[["n", "p", "o"]].sum()
                    f, lo, hi = ratio_ci(per["o"].to_numpy(float), per["n"].to_numpy(float), group_rho(rho, a, b))
                    if 0 < per["o"].sum() < per["n"].sum():
                        lo, hi = max(lo, 0.0), min(hi, 1.0)
                    else:
                        lo = hi = float("nan")
                    pts.append((per["p"].sum() / per["n"].sum(), f, lo, hi))
            if pts:
                p, f, lo, hi = zip(*pts, strict=True)
                out.setdefault(label, {})[m] = {"bss": num(bss, 3), "p": nums(p), "o": nums(f), "lo": nums(lo),
                                                "hi": nums(hi)}  # fmt: skip
    return out


def ens_data(s: pd.DataFrame, r: pd.DataFrame, models: list, rho: dict, name: str, out: Path) -> dict:
    """The window's numbers for the page's charts: per variable the scores by lead and the rank histograms of each
    region, per variable and threshold the reliability."""
    lead, rel = {}, {}
    for v in ENS_ROWS:
        regions = {}
        for region, g in s[s["variable"] == v].groupby("region"):
            if region in ENS_REGIONS and not (g := sampled(g, "lead_h")).empty:
                regions[region] = ens_lead_data(g, models, rho) | {"rank": ens_rank_data(g, models, rho)}
        if regions:
            lead[v] = dump({"regions": regions}, out / f"ens_{name}_{v}.json")
    for (v, thr), g in r[r["variable"].isin([*ENS_ROWS, "tp"]) & r["region"].isin(ENS_REGIONS)].groupby(
        ["variable", "threshold"]
    ):
        regions = {region: d for region, gr in g.groupby("region") if (d := ens_rel_data(gr, models, rho))}
        if regions:
            rel[f"{v}|{thr:g}"] = dump({"regions": regions}, out / f"rel_{name}_{v}_{thr:g}.json")
    return {"lead": lead, "rel": rel}


def ens_figs(derived: Path, out: Path) -> dict:
    """The ensembles' scorecards for each window of runs, and the numbers of their scores by lead, rank histograms
    and reliability."""
    manifest = {}
    for w in ENS_WINDOWS:
        if (found := ens_window(derived, w)) is None:
            continue
        s, r, models, span = found
        rho = run_rho(s, "crps", "n", ["model", "variable", "region"])  # per lead, for every interval of the window
        name, cards = name_of(w), {}
        for region in ENS_REGIONS:
            for a in models:
                for b in models:
                    if a != b and (
                        f := ens_card(s, region, a, b, span, out / f"ecard_{name}_{slug(region)}_{a}_{b}.webp")
                    ):
                        cards.setdefault(region, {})[f"{a}|{b}"] = f
        data = ens_data(s, r, models, rho, name, out)
        manifest[name] = {"window": span, "models": models, "card": cards, "data": data}
        print(f"ensembles {name}: {span['runs']} runs, {len(cards)} scorecard regions", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", type=Path, required=True, help="the database files the scores are drawn from")
    ap.add_argument("--clim", type=Path, help="the 500 hPa normal (release verify-clim-z500-v1)")
    ap.add_argument("--out", type=Path, required=True, help="the figures' directory")
    ap.add_argument("--have", type=Path, help="names of the files the database holds, one per line (verifydb.sh list)")
    a = ap.parse_args()
    style()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")}
    if a.clim and a.clim.exists():
        manifest["acc"] = acc_figs(a.derived, a.clim, a.out)
    manifest["rain"] = rain_figs(a.derived, a.out)
    manifest["rain_skill"] = rain_skill_figs(a.derived, a.out)
    manifest["scores"] = det_figs(a.derived, a.out)
    manifest["ensembles"] = ens_figs(a.derived, a.out)
    manifest["database"] = database(a.derived, a.have)
    (D.OUT_DIR / "figs.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
