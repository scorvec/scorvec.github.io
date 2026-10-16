#!/usr/bin/env python3
"""Which SFS hindcast differences are significant, and the page text that reports them - both from the study tables.

The study tables (scripts/sfs/skill/, offline study) give every SFS-minus-other skill difference with its 95 % year-block
bootstrap interval. A difference's two-sided p is read from that interval (normal approximation, the standard error
from the half of the interval on the side of zero), and it counts as significant only when p < ALPHA AND it passes
Benjamini-Hochberg at FDR_Q over the family of differences its figure shows (FAMILIES) - each figure tests dozens of
cells, and a bare 95 % interval passes one in twenty by chance. The figures (sfs_skillfig.py, sfs_cfsfig.py) mark exactly these cells, and the captions of the two
pages are generated here and written between /*sfs-verdict:<key>*/ and /*end*/ markers:

    python scripts/sfs/sfs_verdicts.py [--site .]      rerun whenever a study table changes
"""

from __future__ import annotations

import argparse
import json
import re
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from quantmet.stats import benjamini_hochberg

HERE = Path(__file__).resolve().parent
TABLES = {
    "gefs_geps": HERE / "skill" / "hindcast_skill_sepoct_2001_2019.csv",
    "cfs_weekly": HERE / "skill" / "cfs" / "partA_sfs_vs_cfs_weekly.csv",
    "cfs_seasonal": HERE / "skill" / "cfs" / "partB_sfs_vs_cfs_seasonal.csv",
}
ALPHA, FDR_Q = 0.05, 0.10
# family -> (table, the rows its figure tests)
FAMILIES = {
    "gefs_geps": ("gefs_geps", lambda d: (d.metric == "acc") & d.system.isin(["sfs-gefs", "sfs-geps"])),
    "cfs_weekly": ("cfs_weekly", lambda d: (d.metric == "acc") & (d.years == "all") & (d.system == "sfs-cfs")),
    "cfs_nino34": ("cfs_seasonal", lambda d: (d.part == "nino34") & (d.system == "sfs-cfs_split")),
    "cfs_seasonal": ("cfs_seasonal", lambda d: (d.part == "seasonal") & (d.metric == "acc") & (d.system == "sfs-cfs")),
}
PAGES = {"gefs_geps": "sfs-subseasonal.html", "cfs_weekly": "sfs-subseasonal.html"}
PAGES |= {"cfs_nino34": "seasonal.html", "cfs_seasonal": "seasonal.html"}
WIN = {"wk1": "week 1", "wk2": "week 2", "wk34": "weeks 3–4", "wk5": "week 5"}
MON = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
SEAS = {"t2m": "temperature", "pr": "precipitation"}
REG = {"Global": "globally", "NH 20-90N": "over 20–90°N", "CONUS land": "over US land"}


def table(name: str) -> pd.DataFrame:
    """A study table with p and fdr_sig (the significance the figures and captions use) for its family rows; every
    other row has fdr_sig False."""
    d = pd.read_csv(TABLES[name])
    # the half of the interval on the side of zero: correlations near 1 have skewed intervals, and this keeps
    # p < 0.05 exactly where the interval excludes zero
    se = np.where(d.value > 0, d.value - d.lo, d.hi - d.value) / 1.959964
    d["p"] = [erfc(abs(v) / s / sqrt(2)) if s > 0 else np.nan for v, s in zip(d.value, se)]
    d["fdr_sig"] = False
    for tab, rows in FAMILIES.values():
        if tab == name:
            m = rows(d).to_numpy()
            p = d.p.to_numpy()[m]
            d.loc[m, "fdr_sig"] = benjamini_hochberg(p, FDR_Q) & (p < ALPHA)
    return d


def family(key: str) -> pd.DataFrame:
    tab, rows = FAMILIES[key]
    d = table(tab)
    return d[rows(d)]


def _list(xs) -> str:
    xs = list(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else (xs[0] if xs else "")


def _rule(n: int) -> str:
    return (
        f"significant means p < {ALPHA:g} and a false discovery rate of {FDR_Q:.0%} over the figure's {n} comparisons"
    )


def _weekly(d: pd.DataFrame, other: str) -> str:
    """Per window: how many of its differences are significant, each way."""
    lo, hi = [], []
    for w, lab in WIN.items():
        g = d[d.window == w]
        nb, na = int((g.fdr_sig & (g.value < 0)).sum()), int((g.fdr_sig & (g.value > 0)).sum())
        lo.append(f"{nb} of {len(g)} in {lab}")
        if na:
            hi.append(f"{na} in {lab}")
    s = f"SFS's anomaly correlation is significantly below {other}'s in " + _list(lo)
    return s + ("; above it in " + _list(hi) if hi else "; it is significantly above in none") + f" ({_rule(len(d))})."


def verdicts() -> dict[str, str]:
    out = {}
    d = family("gefs_geps")
    out["gefs_geps"] = (
        "SFS beta2, GEFSv12 and GEPS hindcasts on 38 starts (1 September and 1 October, 2001–2019; for GEFSv12 the "
        "Wednesday start within 3 days of the 1st), each a four-member mean against its own hindcast climatology, "
        "verified against ERA5. Against each of the other two, for three fields in two regions: "
        + _weekly(d, "the other system")
        + " ▲ marks the other system ahead, ▼ SFS ahead."
    )
    d = family("cfs_weekly")
    out["cfs_weekly"] = (
        "SFS beta2 against CFSv2, the system it is meant to replace, on 1 September and 1 October starts 1999–2025 "
        "(no 2017), each against its own hindcast climatology, verified against ERA5. Over three fields in two regions, "
        + _weekly(d, "CFSv2")
        + " ▲ marks CFSv2 ahead, ▼ SFS ahead. The seasonal comparison (Niño-3.4, 3-month temperature and precipitation) "
        "is on the seasonal page, NOAA SFS group."
    )
    d = family("cfs_nino34")
    parts = []
    for start, slab in (("Oct", "1 October"), ("Sep", "1 September")):
        m0 = MON.index(start)
        bits = []
        for who, sign_acc in (("SFS", 1), ("CFSv2", -1)):
            got = []
            for metric, mlab, better in (("acc", "correlation", sign_acc), ("rmse", "RMSE", -sign_acc)):
                g = d[(d.start == start) & (d.metric == metric) & d.fdr_sig & (np.sign(d.value) == better)]
                months = [MON[m0 + int(p[4:])] for p in g.period if m0 + int(p[4:]) < len(MON)]
                if months:
                    got.append(f"{_list(months)} ({mlab})")
            if got:
                bits.append(f"{who} is significantly better in " + _list(got))
        parts.append(f"From {slab} starts " + ("; ".join(bits) if bits else "no difference is significant") + ".")
    out["cfs_nino34"] = (
        "SFS beta2 against CFSv2 for Niño-3.4, monthly means 1991–2025 against OISST, CFSv2 with CPC’s separate "
        "climatologies before and after 1999. " + " ".join(parts) + f" ({_rule(len(d)).capitalize()}.)"
    )
    d = family("cfs_seasonal")
    rows = []
    for anom, alab in (("loo", "As forecast"), ("loo_detrended", "With each series' trend removed")):
        g = d[(d.anom == anom) & d.fdr_sig]
        bits = []
        for who, sign in (("SFS", 1), ("CFSv2", -1)):
            cells = [
                f"{SEAS[r['var']]} {REG[r.region]} ({r.start} → {r.period})"
                for _, r in g[np.sign(g.value) == sign].iterrows()
            ]
            if cells:
                bits.append(f"{who}'s anomaly correlation is significantly higher for " + _list(cells))
        rows.append(f"{alab}, " + ("; ".join(bits) if bits else "no difference is significant") + ".")
    out["cfs_seasonal"] = (
        "SFS beta2 against CFSv2, the system it is meant to replace, for the 3-month mean after each start, 1991–2025, "
        "verified against ERA5 (Sep → OND: started 1 September, scored over October–December). "
        + " ".join(rows)
        + f" ({_rule(len(d)).capitalize()}.) Only September and October starts exist for SFS beta2."
    )
    return out


MARK = re.compile(r"/\*sfs-verdict:(\w+)\*/(.*?)/\*end\*/", re.S)


def stamp(site: Path) -> None:
    text = verdicts()
    for page in sorted(set(PAGES.values())):
        p = site / page
        s = p.read_text()
        keys = {m.group(1) for m in MARK.finditer(s)}
        want = {k for k, v in PAGES.items() if v == page}
        if keys != want:
            raise SystemExit(f"{page}: verdict markers {sorted(keys)}, expected {sorted(want)}")
        p.write_text(MARK.sub(lambda m: f"/*sfs-verdict:{m.group(1)}*/{json.dumps(text[m.group(1)])}/*end*/", s))
        print(f"stamped {page}: {', '.join(sorted(want))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(HERE.parents[1]))
    stamp(Path(ap.parse_args().site))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
