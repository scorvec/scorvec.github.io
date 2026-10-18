#!/usr/bin/env python3
"""SFS beta2 against CFSv2 in hindcasts: the figures behind the SFS pages' CFS comparison. Reads the study tables in
scripts/sfs/skill/cfs/ (offline study, see its README); draws only.

    python scripts/sfs/sfs_cfsfig.py [--site .]
      -> assets/sfs/sub/sfs_cfs_weekly_skill.webp    (sfs-subseasonal.html, Hindcast skill, "vs CFSv2")
         assets/sfs/sfs_cfs_nino34_skill.webp        (seasonal.html, NOAA SFS group, SFS against CFSv2)
         assets/sfs/sfs_cfs_seasonal_skill.webp      (the same product, 3-month temperature and precipitation)

Every system's skill is drawn with its 95% year-block bootstrap interval; a DIFFERENCE is marked only where it is
significant over the figure's comparisons (sfs_verdicts: its own interval excludes zero and it passes a false discovery
rate of 10 %), and an unmarked point means no significant difference.
"""

import argparse
from pathlib import Path

import numpy as np

from sfs_verdicts import table

HERE = Path(__file__).resolve().parent
SFS_C, CFS_C = "#b0562a", "#5b4a9e"
NOTE_C = "#5c6b73"
MON = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]


def _row(d, **k):
    for c, v in k.items():
        d = d[d[c] == v]
    return d.iloc[0] if len(d) else None


def _save(fig, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    print(f"wrote {p}")


def weekly(plt, site):
    """Weekly anomaly correlation, the layout of sfs_skillfig.py with CFSv2 in place of GEFS and GEPS."""
    d = table("cfs_weekly")
    d = d[(d.metric == "acc") & (d.years == "all")]
    WIN = [("wk1", "week 1"), ("wk2", "week 2"), ("wk34", "weeks 3–4"), ("wk5", "week 5")]
    VARS = [("t2m", "2 m temperature"), ("z500", "500 hPa height"), ("pr", "Precipitation")]
    REG = [("NH 20-90N", "Northern Hemisphere 20–90°N"), ("CONUS land", "Contiguous US, land")]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6), sharey=True)
    x = np.arange(len(WIN))
    for ri, (reg, rlab) in enumerate(REG):
        for ci, (var, vlab) in enumerate(VARS):
            ax = axes[ri, ci]
            sub = d[(d["var"] == var) & (d.region == reg)]
            for si, (s, slab, col) in enumerate((("sfs", "SFS beta2", SFS_C), ("cfs", "CFSv2", CFS_C))):
                rows = [_row(sub, system=s, window=w) for w, _ in WIN]
                v = np.array([r.value for r in rows])
                lo = np.array([r.lo for r in rows])
                hi = np.array([r.hi for r in rows])
                off = (si - 0.5) * 0.24
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
                if s == "cfs":
                    for k, (w, _) in enumerate(WIN):
                        dd = _row(sub, system="sfs-cfs", window=w)
                        if dd is not None and bool(dd.fdr_sig):
                            ax.annotate(
                                "▲" if dd.value < 0 else "▼",
                                (x[k] + off, hi[k]),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                color=col,
                            )
            ax.axhline(0, color="#8a8680", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([w for _, w in WIN], fontsize=9.5)
            ax.grid(axis="y", alpha=0.25)
            if ri == 0:
                n = int(_row(sub, system="sfs", window="wk1").n_years)
                ax.set_title(f"{vlab} ({n} years)", fontsize=11.5, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{rlab}\nanomaly correlation", fontsize=10)
    axes[0, 0].set_ylim(-0.23, 1.07)
    axes[0, 0].legend(loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(
        "SFS beta2 against CFSv2: weekly skill in hindcasts, Sep 1 and Oct 1 starts 1999–2025, verified against ERA5",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "Ensemble-mean anomaly correlation, each system against its own leave-one-year-out climatology; bars: 95% "
        "year-block bootstrap. ▲ over CFSv2: CFSv2 beats SFS, the difference's own interval excluding zero and passing a "
        "false discovery rate of 10% over the figure (▼: SFS ahead); no marker = no significant difference. SFS: four-member means of its reforecast. CFSv2: the 00, 06, 12 and 18Z runs of the "
        "1st, one member each (IRI SubX hindcast 1999–2016 and real-time 2018–2025, no 2017; 500 hPa height 2019–2025 from NOAA's "
        "operational CFSv2 on AWS). ERA5 precipitation ends Jan 2023. Only Sep/Oct starts exist for SFS beta2.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=NOTE_C,
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.985))
    _save(fig, Path(site) / "assets" / "sfs" / "sub" / "sfs_cfs_weekly_skill.webp")
    plt.close(fig)


def _winner_marks(ax, xs, diffs, better_low):
    """Squares along the top edge where the SFS-minus-CFSv2 difference is significant, in the better system's colour."""
    for x, dd in zip(xs, diffs):
        if dd is None or not bool(dd.fdr_sig):
            continue
        sfs_better = (dd.value < 0) if better_low else (dd.value > 0)
        ax.plot(
            [x],
            [0.955],
            marker="s",
            ms=7.5,
            color=SFS_C if sfs_better else CFS_C,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )


def nino34(plt, site):
    d = table("cfs_seasonal")
    d = d[d.part == "nino34"]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 7.8), sharex="col")
    for ci, start in enumerate(("Sep", "Oct")):
        m0 = MON.index(start)
        for ri, (metric, ylab, better_low) in enumerate(
            (("acc", "anomaly correlation", False), ("rmse", "RMSE (°C)", True))
        ):
            ax = axes[ri, ci]
            sub = d[(d.start == start) & (d.metric == metric)]
            leads = sorted(int(p[4:]) for p in sub.period.unique())
            x = np.array(leads)
            for s, slab, col in (
                ("sfs", "SFS beta2 (11 members)", SFS_C),
                ("cfs_split", "CFSv2 (11 of its 24 NMME members)", CFS_C),
            ):
                rows = [_row(sub, system=s, period=f"lead{L}") for L in leads]
                v = np.array([r.value for r in rows])
                lo = np.array([r.lo for r in rows])
                hi = np.array([r.hi for r in rows])
                ax.fill_between(x, lo, hi, color=col, alpha=0.16, lw=0)
                ax.plot(x, v, "-o", color=col, lw=2, ms=4.5, label=slab)
            _winner_marks(ax, x, [_row(sub, system="sfs-cfs_split", period=f"lead{L}") for L in leads], better_low)
            ax.grid(alpha=0.25)
            ax.set_xticks(x)
            ax.set_xticklabels([MON[(m0 + L) % 12] if m0 + L < len(MON) else "" for L in leads], fontsize=9.5)
            if ri == 0:
                ax.set_title(
                    f"Starts on 1 {'September' if start == 'Sep' else 'October'}, 1991–2025",
                    fontsize=11.5,
                    fontweight="bold",
                )
                ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.13)
            else:
                ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
                ax.set_xlabel("forecast month (monthly mean)", fontsize=9.5)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [
        plt.Line2D([], [], ls="", marker="s", ms=7.5, color=SFS_C),
        plt.Line2D([], [], ls="", marker="s", ms=7.5, color=CFS_C),
    ]
    l += ["SFS significantly better", "CFSv2 significantly better"]
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.062))
    fig.suptitle(
        "SFS beta2 against CFSv2: Niño-3.4 in hindcasts, monthly means against OISST v2.1",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "SFS: 11 reforecast members started on the 1st. CFSv2: the NMME archive (24 members from lagged starts "
        "before the 1st, hindcast 1991–2010 and real-time 2011–2025), scored as random 11-member draws with separate "
        "climatologies before and after 1999 as CPC uses for its hindcast. Each system against its own leave-one-year-out "
        "climatology; shading: 95% year-block bootstrap. A square: the SFS-minus-CFSv2 difference's own interval excludes "
        "zero and it passes a false discovery rate of 10% over the figure, in the better system's colour; no square = no "
        "significant difference.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=NOTE_C,
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.985))
    _save(fig, Path(site) / "assets" / "sfs" / "sfs_cfs_nino34_skill.webp")
    plt.close(fig)


def seasonal(plt, site):
    d = table("cfs_seasonal")
    d = d[(d.part == "seasonal") & (d.metric == "acc")]
    COMBO = [("Sep", "OND"), ("Sep", "NDJ"), ("Oct", "NDJ"), ("Oct", "DJF")]
    COLS = [
        ("t2m", "Global", "2 m temperature, global"),
        ("t2m", "NH 20-90N", "2 m temperature, 20–90°N"),
        ("t2m", "CONUS land", "2 m temperature, US land"),
        ("pr", "NH 20-90N", "Precipitation, 20–90°N"),
        ("pr", "CONUS land", "Precipitation, US land"),
    ]
    ROWS = [("loo", "as forecast"), ("loo_detrended", "trend removed")]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.6), sharey=True)
    x = np.arange(len(COMBO))
    for ri, (anom, rlab) in enumerate(ROWS):
        for ci, (var, reg, clab) in enumerate(COLS):
            ax = axes[ri, ci]
            sub = d[(d["var"] == var) & (d.region == reg) & (d.anom == anom)]
            for si, (s, slab, col) in enumerate((("sfs", "SFS beta2", SFS_C), ("cfs", "CFSv2", CFS_C))):
                rows = [_row(sub, system=s, start=a, period=b) for a, b in COMBO]
                v = np.array([r.value for r in rows])
                lo = np.array([r.lo for r in rows])
                hi = np.array([r.hi for r in rows])
                ax.errorbar(
                    x + (si - 0.5) * 0.24,
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
            _winner_marks(ax, x, [_row(sub, system="sfs-cfs", start=a, period=b) for a, b in COMBO], False)
            ax.axhline(0, color="#8a8680", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{a} → {b}" for a, b in COMBO], fontsize=8.8)
            ax.grid(axis="y", alpha=0.25)
            if ri == 0:
                ax.set_title(clab, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{rlab}\nanomaly correlation", fontsize=10)
    axes[0, 0].set_ylim(top=max(ax.get_ylim()[1] for ax in axes.flat) * 1.1)
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [
        plt.Line2D([], [], ls="", marker="s", ms=7.5, color=SFS_C),
        plt.Line2D([], [], ls="", marker="s", ms=7.5, color=CFS_C),
    ]
    l += ["SFS significantly better", "CFSv2 significantly better"]
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.075))
    fig.suptitle(
        "SFS beta2 against CFSv2: 3-month means in hindcasts 1991–2025, verified against ERA5",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "Ensemble-mean anomaly correlation of the 3-month mean from each start (Sep → OND: started 1 September, "
        "scored over October–December). SFS: 11 reforecast members started on the 1st;\nCFSv2: the NMME archive, random "
        "11-member draws of its 24 lagged members. Each system and ERA5 against its own leave-one-year-out climatology; "
        "bottom row: each series' linear trend removed first\n(fitted leaving the year out), so skill from the warming trend "
        "alone drops out. Bars: 95% year-block bootstrap; a square: the difference's own interval excludes zero and it "
        "passes a false discovery rate of 10% over the figure, in the better system's colour;\nno square = no significant "
        "difference. ERA5 precipitation: Northern Hemisphere only, to Jan 2023.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=NOTE_C,
        linespacing=1.35,
    )
    fig.tight_layout(rect=(0, 0.115, 1, 0.985))
    _save(fig, Path(site) / "assets" / "sfs" / "sfs_cfs_seasonal_skill.webp")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(HERE.parent.parent))
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weekly(plt, a.site)
    nino34(plt, a.site)
    seasonal(plt, a.site)


if __name__ == "__main__":
    main()
