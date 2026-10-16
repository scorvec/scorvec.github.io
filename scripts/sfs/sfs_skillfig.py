#!/usr/bin/env python3
"""The SFS subseasonal page's hindcast-skill figure: SFS beta2 against GEFSv12 and GEPS8, weekly ensemble-mean anomaly
correlation with ERA5 for Sep 1 / Oct 1 starts 2001-2019 (study tables in scripts/sfs/skill/).

Every system as a 4-member mean against its own leave-one-year-out hindcast climatology; bars are year-block bootstrap
95% intervals. A difference is marked only where its own interval excludes zero (the site's significance rule); the
week-5 "SFS ahead" CRPS cases of the study are not drawn - with ~100 tests a few such hits are expected by chance.

    python scripts/sfs/sfs_skillfig.py [--site .]   -> assets/sfs/sub/sfs_hindcast_skill.webp
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "skill" / "hindcast_skill_sepoct_2001_2019.csv"
SYS = [("sfs", "SFS beta2", "#b0562a"), ("gefs", "GEFSv12", "#1f5fbf"), ("geps", "GEPS8", "#2f7d4f")]
WIN = [("wk1", "week 1"), ("wk2", "week 2"), ("wk34", "weeks 3–4"), ("wk5", "week 5")]
VARS = [("t2m", "2 m temperature"), ("z500", "500 hPa height"), ("pr", "Precipitation")]
REG = [("NH 20-90N", "Northern Hemisphere 20–90°N"), ("CONUS land", "Contiguous US, land")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(HERE.parent.parent))
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(CSV)
    d = d[d.metric == "acc"]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6), sharey=True)
    x = np.arange(len(WIN))
    for ri, (reg, rlab) in enumerate(REG):
        for ci, (var, vlab) in enumerate(VARS):
            ax = axes[ri, ci]
            sub = d[(d["var"] == var) & (d.region == reg)]
            drawn = 0
            for si, (s, slab, col) in enumerate(SYS):
                rows = [sub[(sub.system == s) & (sub.window == w)] for w, _ in WIN]
                if not all(len(r) for r in rows):
                    continue
                v = np.array([float(r.value.iloc[0]) for r in rows])
                lo = np.array([float(r.lo.iloc[0]) for r in rows])
                hi = np.array([float(r.hi.iloc[0]) for r in rows])
                off = (si - 1) * 0.22
                ax.errorbar(
                    x + off,
                    v,
                    yerr=[v - lo, hi - v],
                    fmt="o",
                    ms=5.5,
                    color=col,
                    ecolor=col,
                    elinewidth=1.4,
                    capsize=3,
                    label=slab if (ri == 0 and ci == 0) else None,
                )
                drawn += 1
                if s != "sfs":
                    # the SFS-minus-this-system difference, marked only where its interval excludes zero
                    for k, (w, _) in enumerate(WIN):
                        dd = sub[(sub.system == f"sfs-{s}") & (sub.window == w)]
                        if len(dd) and str(dd.sig.iloc[0]) == "yes":
                            ax.annotate(
                                "▲" if float(dd.value.iloc[0]) < 0 else "▼",
                                (x[k] + off, hi[k]),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                color=col,
                            )
            if not drawn:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="#6f6b64")
            ax.axhline(0, color="#8a8680", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([w for _, w in WIN], fontsize=9.5)
            ax.grid(axis="y", alpha=0.25)
            if ri == 0:
                ax.set_title(vlab, fontsize=11.5, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{rlab}\nanomaly correlation", fontsize=10)
    axes[0, 0].legend(loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(
        "SFS beta2 against GEFS and GEPS: weekly skill in hindcasts, Sep 1 and Oct 1 starts 2001–2019, verified against ERA5",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "Ensemble-mean anomaly correlation, every system as a 4-member mean against its own leave-one-year-out "
        "hindcast climatology; bars: 95% year-block bootstrap. ▲ over GEFS or GEPS: that system beats SFS, the difference's own "
        "interval excluding zero (▼: SFS ahead); no marker = no significant difference. "
        "GEFS: the Wednesday start within 3 days of the 1st, to day 35; no GEFS precipitation in this test. Only Sep/Oct "
        "starts exist for SFS beta2, so winter skill can differ.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#5c6b73",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    out = Path(a.site) / "assets" / "sfs" / "sub"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "sfs_hindcast_skill.webp"
    fig.savefig(p, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
