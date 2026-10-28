#!/usr/bin/env python3
"""The figures of verification.html, drawn each night from the verification database as static WebP (the page's
controls pick one), every score with its uncertainty.

  acc   The 500 hPa height anomaly correlation against ERA5 (z500_acc.py), for each region and window of runs: a time
        series for each of six leads, as NCEP EMC draws it, a running mean of the runs
        (acc_series_<region>_<window>_d<day>.webp), and the window's mean by lead with its 95 % interval over each
        single run's paired difference from the IFS control and each ensemble mean's from the IFS-ENS mean
        (acc_lead_<region>_<window>.webp).
  rain  Rain against Stage IV cell by cell (mverify_scores.rain_map): the ratio of the totals and the frequency
        biases at 1, 10 and 25 mm for each lead group, significant cells only (rain_<map>_<group>.webp), and what the
        analysis gave over the same days (rain_<map>_obs.webp).
figs.json lists what was drawn, for the page.

Uncertainty. A window's runs are a sample: the same models over other weeks would score differently, and runs a day
apart share their weather, so they are not independent. Every interval is a 95 % Student t interval on the runs' own
scores with the sample cut for lag-1 autocorrelation (mverify.significance.effective_runs, the scorecards' rule), the
anomaly correlation averaged through Fisher's z = artanh(r) and turned back. A difference from the reference is the
mean of the runs' paired differences (Geer 2016: unpaired intervals hide most of what separates two models) and is
significant when its paired test passes Benjamini-Hochberg at a 10 % false-discovery rate over every model and lead of
the figure.

    python verify_figs.py --derived db --clim clim_z500_1p5.npz --out fig
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
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
LABEL = {"IFS": "IFS control", **{m: f"{m} mean" for m in CENTRE}}
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
    """The house style, a size up: the figures are read at the page's width, scorecards and maps most of all."""
    house.apply()
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 19,
        }
    )


def save(fig, path: Path) -> str:
    fig.savefig(path, dpi=DPI, format="webp", pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)
    return path.name


def name_of(w) -> str:
    return str(w or "all")


def ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """The mean of a run series and its interval, the sample cut for lag-1 autocorrelation."""
    from mverify.significance import effective_runs

    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    n = effective_runs(x)
    h = stats.t.ppf(0.5 + level / 2, max(n - 1, 1.0)) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


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


def area_of(region: str) -> str:
    return "20–80°N" if region == "NH" else "20–80°S"


def draw_series(t: pd.DataFrame, region: str, day: int, start: pd.Timestamp, path: Path) -> str | None:
    """The anomaly correlation at one lead against the date it verifies, as EMC's charts, as a running mean of the runs
    (each run's own score swings with the day's weather far more than the models differ)."""
    r = t[(t["region"] == region) & (t["valid_time"] >= start) & (t["lead_h"] == 24 * day)]
    if r.empty:
        return None
    days = pd.date_range(r["valid_time"].min().normalize(), r["valid_time"].max().normalize(), freq="D")
    fig, ax = plt.subplots(figsize=house.size(15, 0.45), constrained_layout=True)
    seen = []
    for m in [*Z.DET, *Z.E.ENSEMBLES]:
        s = r[r["model"] == m].set_index("valid_time")["acc"].sort_index()
        if s.empty:
            continue
        s = s.groupby(s.index.normalize()).mean().reindex(days)  # a missing run leaves a gap
        s = s.rolling(SMOOTH_DAYS, center=True, min_periods=SMOOTH_DAYS // 2 + 1).mean()
        ax.plot(s.index, s.to_numpy(), color=COLORS[CENTRE.get(m, m)], lw=2.4, ls=(0, (4, 2)) if m in CENTRE else "-",
                label=LABEL.get(m, m))  # fmt: skip
        seen.append(s.to_numpy())
    lo, hi = np.nanmin(np.concatenate(seen)), np.nanmax(np.concatenate(seen))
    pad = 0.08 * max(hi - lo, 0.01)
    ax.set(xlim=(days[0], days[-1]), ylim=(lo - pad, min(1.0, hi + pad)), ylabel="Anomaly correlation")
    if lo <= USEFUL:
        ax.axhline(USEFUL, color=house.MUTED, lw=1.2, ls=":")
    loc = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set(major_locator=loc, major_formatter=mdates.ConciseDateFormatter(loc))
    ax.grid(True, color=house.GRID, lw=0.8)
    fig.legend(loc="outside right upper", frameon=False, handlelength=2.6)
    fig.suptitle(
        f"Day {day}: 500 hPa height anomaly correlation against ERA5, {area_of(region)}, {SMOOTH_DAYS}-day running mean",
        x=0.01,
        ha="left",
    )
    return save(fig, path)


def differences(ax, wide: pd.DataFrame, models: list[str], ref: str) -> list[tuple]:
    """Each model's paired difference from ``ref`` by lead, with its interval, on ``ax``; returns the rows drawn."""
    from mverify.significance import paired_test

    rows = []
    for m in models:
        for lead in sorted(lead for mm, lead in wide.columns if mm == m):
            if (ref, lead) in wide.columns:
                d = (wide[(m, lead)] - wide[(ref, lead)]).to_numpy()
                rows.append((m, lead, *ci(d), paired_test(d[np.isfinite(d)])))
    ax.axhline(0, color=COLORS[CENTRE.get(ref, ref)], lw=2, ls=(0, (4, 2)) if ref in CENTRE else "-")
    ax.set(xlabel="Lead (days)", title=f"Minus the {LABEL.get(ref, ref)}")
    ax.set_xticks(range(1, 16))
    ax.grid(True, color=house.GRID, lw=0.8)
    return rows


def draw_lead(wide: pd.DataFrame, models: list[str], region: str, span: dict, path: Path) -> tuple[str, dict]:
    """The window's mean anomaly correlation by lead with its interval, and each model's paired difference from the
    IFS control (the single runs) or the IFS-ENS mean (the ensemble means)."""
    from mverify.significance import fdr

    fig = plt.figure(figsize=house.size(15, 0.78), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=(1.15, 1))
    a1 = top.subplots()
    a2, a3 = bottom.subplots(1, 2)
    table = {}
    for m in models:
        leads = sorted(lead for mm, lead in wide.columns if mm == m)
        # each lead's mean and interval through Fisher's z
        mean, lo, hi = np.tanh(
            [ci(np.arctanh(np.clip(wide[(m, lead)].to_numpy(), -0.9999, 0.9999))) for lead in leads]
        ).T
        x, color = np.array(leads) / 24, COLORS[CENTRE.get(m, m)]
        a1.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
        a1.plot(x, mean, color=color, lw=2.4, ls=(0, (4, 2)) if m in CENTRE else "-", marker="o", ms=5,
                label=LABEL.get(m, m))  # fmt: skip
        table[m] = {
            f"{int(lead) // 24}": [round(v, 4) for v in (a, b, c)] for lead, a, b, c in zip(leads, mean, lo, hi)
        }
    a1.axhline(USEFUL, color=house.MUTED, lw=1.2, ls=":")
    a1.set(xlabel="Lead (days)", ylabel="Anomaly correlation", title="Mean over the window, with its 95 % interval")
    a1.set_xticks(range(1, 16))
    a1.set_xlim(0.6, 15.4)
    a1.grid(True, color=house.GRID, lw=0.8)
    a1.legend(loc="lower left", ncols=2, frameon=False, handlelength=2.6)
    single = [m for m in models if m in Z.DET and m != REF]
    ens_ref = "IFS-ENS"
    means = [m for m in models if m in CENTRE and m != ens_ref]
    panels = [(a2, single, REF), (a3, means, ens_ref)]
    rows = {id(ax): differences(ax, wide, ms, ref) if ref in models else [] for ax, ms, ref in panels}
    every = [row for r in rows.values() for row in r]
    sig = iter(fdr(np.array([row[-1] for row in every]), FDR_Q)) if every else iter(())
    for ax, ms, _ in panels:
        for m, lead, c, lo, hi, _p in rows[id(ax)]:
            s, color = next(sig), COLORS[CENTRE.get(m, m)]
            x = lead / 24 + (ms.index(m) - (len(ms) - 1) / 2) * 0.16
            ax.plot([x, x], [lo, hi], color=color, lw=2, alpha=1 if s else 0.45)
            ax.plot(x, c, marker="o", ms=7, color=color, mfc=color if s else "white", mew=2, alpha=1 if s else 0.7)
        for m in ms:
            ax.plot([], [], color=COLORS[CENTRE.get(m, m)], marker="o", ls="none", ms=7, label=LABEL.get(m, m))
        ax.set_xlim(0.5, (max(Z.DET_LEADS) if ax is a2 else max(Z.E.LEADS)) / 24 + 0.5)
    a2.set_ylabel("Difference in anomaly correlation")
    handles, labels = a2.get_legend_handles_labels()
    more = a3.get_legend_handles_labels()
    handles, labels = handles + more[0], labels + more[1]
    bottom.legend(handles, labels, loc="outside lower center", ncols=len(labels), frameon=False)
    bottom.suptitle("Paired differences: filled where significant, hollow where within the noise", x=0.01, ha="left",
                    fontsize=plt.rcParams["axes.titlesize"])  # fmt: skip
    fig.suptitle(f"500 hPa height anomaly correlation against ERA5, {area_of(region)}: {span['runs']} runs", x=0.01,
                 ha="left")  # fmt: skip
    return save(fig, path), table


def acc_figs(derived: Path, clim: Path, out: Path) -> dict:
    t = Z.table(derived, np.load(clim)["z500"])
    t["init"], t["valid_time"] = pd.to_datetime(t["init"]), pd.to_datetime(t["valid_time"])
    manifest = {}
    for region in Z.REGIONS:
        for w in WINDOWS:
            wide, models, span = acc_window(t, region, w)
            if len(wide) < 10 or (w and len(wide) < w * D.WINDOW_SHARE):
                continue
            key = f"{region}_{name_of(w)}"
            start = wide.index.min() + pd.Timedelta(days=1)
            series = {
                str(day): name
                for day in SERIES_DAYS
                if (name := draw_series(t, region, day, start, out / f"acc_series_{key}_d{day}.webp"))
            }
            lead, table = draw_lead(wide, models, region, span, out / f"acc_lead_{key}.webp")
            manifest[key] = {"series": series, "lead": lead, "window": span, "mean": table}
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
                     "significant at a 10 % false-discovery rate", color=house.INK2, fontsize=12.5)  # fmt: skip
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", type=Path, required=True, help="the database files the scores are drawn from")
    ap.add_argument("--clim", type=Path, help="the 500 hPa normal (release verify-clim-z500-v1)")
    ap.add_argument("--out", type=Path, required=True, help="the figures' directory")
    a = ap.parse_args()
    style()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")}
    if a.clim and a.clim.exists():
        manifest["acc"] = acc_figs(a.derived, a.clim, a.out)
    manifest["rain"] = rain_figs(a.derived, a.out)
    (D.OUT_DIR / "figs.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
