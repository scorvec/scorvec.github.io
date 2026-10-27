#!/usr/bin/env python3
"""How the recent stratosphere forecasts verified: each model's past runs against what happened.

From the run records (run_record.py) and analyses the site already publishes; nothing is downloaded:
  u10   zonal-mean u at 60N, 10 hPa (the vortex record, raw model output) against GEOS FP's daily-mean analysis
        (upperstrat_tail.json)
  flux  100 hPa [v'T'], 45-75N, all waves (the heat-flux record, each model's error removed as on its figure) against
        the analysed daily means on the heat-flux figure (heatflux100.json "observed": the AIFS-ENS analyses, their
        mean error against ERA5 removed)
  ao    the Arctic Oscillation (the telecon record) against the IFS control's analysis, step 0 of its runs
For each model, the 00Z runs from 3, 6, 9, 12 and 15 days before its newest are drawn over the analysis, each labelled
with its mean absolute error over the days that have verified so far. Descriptive: single runs are not a sample to
rank the models on.

    python strat_verify.py      # -> assets/sst/runs/verify_strat.webp (strat-combine.yml)
"""

from __future__ import annotations
import sys

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import run_record as RR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

MODELS = ("aifs", "ifs", "hres", "gefs", "geps")
BACK = (15, 12, 9, 6, 3)  # days before the newest run
OUT = RR.FIGS / "verify_strat.webp"
ROWS = (
    ("vortex", "u10", "u at 60°N, 10 hPa", "m s⁻¹", "GEOS FP daily-mean analysis"),
    ("heatflux", "tot", "100 hPa eddy heat flux, 45–75°N", "K m s⁻¹", "analysed daily mean"),
    ("telecon", "ao", "Arctic Oscillation", "sd", "IFS control analysis"),
)


def truth(product: str) -> pd.Series:
    """The analysis each row is verified against, by valid time (empty when none is published)."""
    if product == "vortex":
        d = json.loads((RR.DATA / "upperstrat_tail.json").read_text())
        j = d["levels"].index(10.0)
        s = pd.Series({pd.Timestamp(day): v["nh"]["u60"][j] for day, v in d["days"].items()}, dtype=float)
    elif product == "heatflux":
        ob = json.loads((RR.DATA / "heatflux100.json").read_text())["hemispheres"]["nh"]["observed"]
        s = pd.Series(ob["tot"], index=pd.DatetimeIndex(ob["dates"]), dtype=float)
    else:  # the IFS control's analysis: step 0 of each of its runs
        s = pd.Series({pd.Timestamp(r["valid"][0]): r["ao"]["mean"][0] for r in RR.runs(product, "hres")}, dtype=float)
    s.index = pd.DatetimeIndex(s.index)
    return s.dropna().sort_index()


def daily(t: pd.Series) -> pd.Series:
    """The analysis as daily means (the forecasts are matched by valid date)."""
    return t.groupby(t.index.normalize()).mean()


def chosen(rs: list) -> list:
    """(index in BACK, start, run) of the 00Z runs BACK days before the newest one, oldest first (those recorded)."""
    if not rs:
        return []
    by = {pd.Timestamp(r["init"]): r for r in rs}
    t1 = max(by).normalize()
    return [(i, t, by[t]) for i, k in enumerate(BACK) if (t := t1 - pd.Timedelta(days=k)) in by]


def render(out: Path = OUT) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import house

    house.apply()

    truths = {p: daily(truth(p)) for p, *_ in ROWS}
    cols = [m for m in MODELS if any(RR.runs(p, m) for p, *_ in ROWS)]
    if not cols or all(t.empty for t in truths.values()):
        return None
    shades = plt.cm.viridis(np.linspace(0.85, 0.1, len(BACK)))  # oldest light, newest dark
    fig, axes = plt.subplots(len(ROWS), len(cols), figsize=(3.3 * len(cols) + 1.2, 3.1 * len(ROWS) + 1.6),
                             sharex=True, squeeze=False)  # fmt: skip
    t0 = None
    for i, (product, key, name, unit, source) in enumerate(ROWS):
        tr = truths[product]
        for j, m in enumerate(cols):
            ax = axes[i, j]
            runs = chosen(RR.runs(product, m))
            for k, init, r in runs:
                f = RR.series(r, key)
                f.index = f.index.normalize()
                both = f.index.intersection(tr.index[tr.index > init])
                err = f" · {np.mean(np.abs(f[both] - tr[both])):.1f} {unit}, {len(both)} d" if len(both) else ""
                ax.plot(f.index, f.values, color=shades[k], lw=1.5, label=f"{init:%-d %b}{err}")
                t0 = init if t0 is None else min(t0, init)
            if not tr.empty:
                ax.plot(tr.index, tr.values, color="#111", lw=2.4, zorder=6, label=source if j == 0 else None)
            if key != "tot":
                ax.axhline(0, color="#8a8680", lw=0.9, zorder=2)
            if i == 0:
                ax.set_title(RR.MODEL_LABEL[m], fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{name}\n{unit}", fontsize=9)
            ax.grid(alpha=0.22)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
            if runs or (j == 0 and not tr.empty):
                ax.legend(fontsize=7, loc="best", framealpha=0.85, handlelength=1.4)
            else:
                ax.text(0.5, 0.5, "no runs recorded yet", transform=ax.transAxes, ha="center", fontsize=8.5,
                        color="#6f6b64")  # fmt: skip
    if t0 is not None:
        axes[0, 0].set_xlim(left=t0)
    fig.suptitle("How the recent runs verified: each model's 00Z runs from 3 to 15 days ago against the analysis",
                 fontsize=13.5, fontweight="bold", x=0.05, ha="left", y=0.995)  # fmt: skip
    fig.text(0.05, 0.955, "each run's ensemble mean (the IFS control: the run), oldest light, newest dark, labelled with "
             "its mean absolute error over the days verified so far · black: the analysis", fontsize=9.5,
             color="0.25")  # fmt: skip
    fig.text(0.05, 0.008, "\n".join(textwrap.wrap(
        "u at 60°N: raw model output against GEOS FP's daily mean (NASA GMAO). Heat flux: each model's mean error "
        "against ERA5 removed, as on the heat-flux figure, against the analysed daily means there. AO: against the IFS "
        "control's own analysis on the site's CPC-style pattern. A forecast is matched to the analysis by valid date. "
        "Single runs: descriptive, not a ranking of the models.", 210)), fontsize=8, color="#6f6b64",
        linespacing=1.4)  # fmt: skip
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.09, hspace=0.28, wspace=0.22)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    render(ap.parse_args().out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
