#!/usr/bin/env python3
"""Static scorecard + figures for aifs-verify.html (2026-10-04, user: "the interactive charts are clunky and annoying,
let's switch to slick scorecards and static images").

Reads assets/verify/aifs_scores_v2.json (scripts/verify/aifs_station_verify.py) and writes
  assets/verify/aifs_scorecard.json        the page's scorecard + headline numbers (a few kB; the page no longer loads the
                                           11 MB scores file)
  assets/verify/aifs_lead_<truth>_<var>_<region>.webp   RMSE / ACC / bias / activity vs lead, the four series
  assets/verify/aifs_spectra.webp          each series' power as a ratio to the single, days 1, 5, 10

Scores by lead average over the SAME runs for every series (the runs every series has at that lead). Differences and
their 95 % intervals come from the file's paired moving-block bootstrap ("diffs"); nothing is re-tested here.
    python scripts/verify/aifs_verify_figs.py [--scores assets/verify/aifs_scores_v2.json] [--out assets/verify]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullFormatter
import pandas as pd

ROWS = [("raob", "z500", "nh", "500 hPa height", "radiosondes", "NH 20–80°N"),
        ("raob", "z500", "glb", "500 hPa height", "radiosondes", "global"),
        ("raob", "t850", "nh", "850 hPa temperature", "radiosondes", "NH 20–80°N"),
        ("raob", "t850", "glb", "850 hPa temperature", "radiosondes", "global"),
        ("era5", "z500", "nh", "500 hPa height", "ERA5", "NH 20–80°N"),
        ("era5", "t850", "nh", "850 hPa temperature", "ERA5", "NH 20–80°N"),
        ("era5", "msl", "nh", "Sea-level pressure", "ERA5", "NH 20–80°N"),
        ("era5", "t2m", "nh", "2 m temperature", "ERA5", "NH 20–80°N")]
FIGS = [r for r in ROWS if r[2] == "nh"]                 # one lead figure per NH row (global only in the scorecard)
UNIT = {"z500": "m", "t850": "°C", "msl": "hPa", "t2m": "°C"}
MODELS = ["single", "matched", "control", "ensmean"]
NAME = {"single": "AIFS single", "matched": "member 0, matched", "control": "member 0, as published", "ensmean": "ensemble mean"}
COL = {"single": "#b3372f", "matched": "#2c5f8a", "control": "#8fb0cc", "ensmean": "#5c5a55"}
STY = {"single": "-", "matched": "-", "control": "--", "ensmean": ":"}


def paired_means(R, truth, var, region):
    """{model: DataFrame(lead -> mean rmse/acc/bias/act, n)} over the runs every model has at that lead."""
    S = R[(R.truth == truth) & (R["var"] == var) & (R.region == region)]
    out = {}
    for lead, g in S.groupby("lead"):
        common = None
        for m in MODELS:
            inits = set(g.loc[g.model == m, "init"])
            common = inits if common is None else common & inits
        if not common:
            continue
        for m in MODELS:
            x = g[(g.model == m) & g.init.isin(common)]
            out.setdefault(m, []).append(dict(lead=int(lead), n=len(common), **{k: float(x[k].mean()) for k in ("rmse", "acc", "bias", "act")}))
    return {m: pd.DataFrame(v).set_index("lead") for m, v in out.items()}


def scorecard(D, R):
    leads = [int(x) for x in D["leads"]]
    rows = []
    for truth, var, region, vlab, tlab, rlab in ROWS:
        key = f"{truth}:{var}:{region}"
        P = paired_means(R, truth, var, region)
        if "single" not in P:
            continue
        sgl = P["single"]
        row = dict(key=key, var=var, label=vlab, truth=tlab, region=rlab, unit=UNIT[var],
                   single={m: [None if l not in sgl.index else round(float(sgl.at[l, m]), 4) for l in leads] for m in ("rmse", "acc")},
                   n=[None if l not in sgl.index else int(sgl.at[l, "n"]) for l in leads], cmp={})
        dd = (D.get("diffs") or {}).get(key, {})
        for m in ("matched", "control", "ensmean"):
            row["cmp"][m] = {}
            for metric in ("rmse", "acc"):
                cells = []
                byl = {int(x["lead"]): x for x in (dd.get(m) or {}).get(metric, [])}
                for l in leads:
                    x = byl.get(l)
                    if not x or l not in sgl.index:
                        cells.append(None); continue
                    if metric == "rmse":                      # % change in RMSE (negative = better)
                        base = sgl.at[l, "rmse"]
                        cells.append(dict(v=round(100 * x["mean"] / base, 1), lo=round(100 * x["lo"] / base, 1),
                                          hi=round(100 * x["hi"] / base, 1), sig=bool(x["sig"]), n=int(x["n"]),
                                          d=round(x["mean"], 3)))
                    else:                                     # change in ACC, hundredths (positive = better)
                        cells.append(dict(v=round(100 * x["mean"], 1), lo=round(100 * x["lo"], 1),
                                          hi=round(100 * x["hi"], 1), sig=bool(x["sig"]), n=int(x["n"]), d=round(x["mean"], 4)))
                row["cmp"][m][metric] = cells
        rows.append(row)
    inits = sorted(set(R.init))
    k = {}
    P = paired_means(R, "raob", "z500", "nh")
    if "single" in P and 120 in P["single"].index:
        k = {m: round(float(P[m].at[120, "rmse"]), 1) for m in MODELS if m in P and 120 in P[m].index}
    d5 = next((x for x in ((D.get("diffs") or {}).get("raob:z500:nh", {}).get("matched", {}).get("rmse", [])) if int(x["lead"]) == 120), None)
    return dict(generated=D.get("generated"), leads=leads, rows=rows, runs=len(inits), first=inits[0] if inits else None,
                last=inits[-1] if inits else None, day5_z500_rmse=k,
                day5_diff=None if not d5 else {k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in d5.items()},
                lmax_t=D.get("lmax_t"), diff_test=D.get("diff_test", {}).get("method"))


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8780", "axes.labelcolor": "#3d3b37", "xtick.color": "#3d3b37",
                         "ytick.color": "#3d3b37", "axes.grid": True, "grid.color": "#ebe9e3", "grid.linewidth": 0.8})
    return plt


def lead_fig(R, row, out: Path, D):
    plt = style()
    truth, var, region, vlab, tlab, rlab = row
    P = paired_means(R, truth, var, region)
    if "single" not in P:
        return False
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.4))
    u = UNIT[var]
    for ax, (k, lab) in zip(axs.flat, (("rmse", f"RMSE ({u})"), ("acc", "anomaly correlation"), ("bias", f"bias ({u})"),
                                      ("act", "activity ratio (forecast / observed anomaly sd)"))):
        for m in MODELS:
            if m not in P:
                continue
            x = P[m].index.values / 24.0
            ax.plot(x, P[m][k].values, STY[m], color=COL[m], lw=2.4 if m in ("single", "matched") else 1.7,
                    marker="o", ms=3.5, label=NAME[m])
        if k in ("bias",):
            ax.axhline(0, color="#8a8780", lw=0.9)
        if k == "act":
            ax.axhline(1, color="#8a8780", lw=0.9)
        ax.set_title(lab, loc="left", fontsize=11, fontweight="bold", color="#1c1c1a")
        ax.set_xticks(range(1, 11)); ax.set_xlim(0.7, 10.3); ax.set_xlabel("lead (days)", fontsize=9.5)
    n = int(P["single"]["n"].median())
    fig.suptitle(f"{vlab} vs {tlab}, {rlab} — mean over the same {n} runs at each lead (median)", x=0.01, ha="left",
                 fontsize=13.5, fontweight="bold", y=0.995)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.995, 0.955))
    fig.text(0.01, 0.005, f"AIFS single and AIFS-ENS, 00Z + 12Z runs {D['_first']}–{D['_last']}; all series truncated at "
             f"T{D.get('lmax_t', 120)} before scoring. Whether two lines differ significantly: see the scorecard.",
             fontsize=8.6, color="#6f6b64")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6}); plt.close(fig)
    return True


def spectra_fig(D, out: Path):
    plt = style()
    S = D.get("spectra") or {}
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    LC = {24: "#9ecae1", 120: "#4292c6", 240: "#08306b"}
    ok = False
    for ax, var in zip(axs.flat, ("z500", "t850", "msl", "t2m")):
        for lead, c in LC.items():
            base = S.get(f"single_{var}_{lead}")
            if not base:
                continue
            b = np.asarray(base["mean"], float)
            l = np.arange(len(b))
            for m, ls, lw in (("control", "-", 2.2), ("matched", "--", 1.5)):
                x = S.get(f"{m}_{var}_{lead}")
                if not x:
                    continue
                r = np.asarray(x["mean"], float) / np.where(b > 0, b, np.nan)
                ax.plot(l[1:], r[1:], ls, color=c, lw=lw, label=f"day {lead // 24}, {NAME[m]}"); ok = True
        ax.axhline(1, color="#b3372f", lw=1.2)
        ax.axvline(D.get("lmax_t", 120), color="#8a8780", lw=1, ls=":")
        ax.set_xscale("log"); ax.set_xlim(2, 360); ax.set_yscale("log"); ax.set_ylim(0.6, 3.6)
        ax.set_yticks([0.7, 1, 1.5, 2, 3]); ax.set_yticklabels(["0.7", "1", "1.5", "2", "3"])
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
        ax.yaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title({"z500": "500 hPa height", "t850": "850 hPa temperature", "msl": "Sea-level pressure", "t2m": "2 m temperature"}[var],
                     loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("total wavenumber l (log)", fontsize=9.5); ax.set_ylabel("power / single's power", fontsize=9.5)
    if not ok:
        plt.close(fig); return False
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=3, frameon=False, fontsize=9.3, bbox_to_anchor=(0.995, 0.955))
    fig.suptitle("Power spectra: member 0's power relative to the single (red line = the single)", x=0.01, ha="left",
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.text(0.01, 0.005, "Solid: member 0 as published - above 1 where the single has smoothed small scales away, more so at "
             "longer leads.\nDashed: member 0 after spectral matching, which should sit on 1. Dotted vertical: the T"
             f"{D.get('lmax_t', 120)} truncation every score uses.", fontsize=8.6, color="#6f6b64")
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6}); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="assets/verify/aifs_scores_v2.json")
    ap.add_argument("--out", default="assets/verify")
    a = ap.parse_args()
    D = json.loads(Path(a.scores).read_text())
    R = pd.DataFrame(D["records"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sc = scorecard(D, R)
    D["_first"], D["_last"] = (sc["first"] or "")[:8], (sc["last"] or "")[:8]
    sc["figures"] = {}
    for row in FIGS:
        f = f"aifs_lead_{row[0]}_{row[1]}_{row[2]}.webp"
        if lead_fig(R, row, out / f, D):
            sc["figures"][f"{row[0]}:{row[1]}:{row[2]}"] = f
    sc["spectra"] = "aifs_spectra.webp" if spectra_fig(D, out / "aifs_spectra.webp") else None
    (out / "aifs_scorecard.json").write_text(json.dumps(sc, separators=(",", ":")))
    print(f"scorecard: {len(sc['rows'])} rows, {sc['runs']} runs; {len(sc['figures'])} lead figures; spectra {bool(sc['spectra'])}")


if __name__ == "__main__":
    main()
