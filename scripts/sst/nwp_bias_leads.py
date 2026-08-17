#!/usr/bin/env python3
"""AIFS vs IFS rainfall bias by lead time, per basin — both countries.

From the forecast archives + gauge-corrected truth caches: at each lead
1-15 d, the ensemble-mean bias ratio (sum fcst / sum obs) and MAE per
model per basin. The per-lead curves show HOW the bias evolves with
lead — the engine's banded factors are the piecewise version of these.

Outputs: colombia_hydro/bias_by_lead.webp, brazil_hydro/bias_by_lead.webp

    python scripts/sst/nwp_bias_leads.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
CFG = {
    "colombia": dict(
        arch=Path.home() / "colombia_hydro" / "raw" / "fcst_rain",
        truth=Path.home() / "colombia_hydro" / "raw" / "imerg_basin_daily.json",
        basins=["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"],
        grid=(2, 3), fig=(13.5, 8.0),
        out=REPO / "colombia_hydro" / "bias_by_lead.webp",
        title="Colombia — NWP rainfall bias by lead time (vs gauge-corrected IMERG)"),
    "brazil": dict(
        arch=Path.home() / "brazil_hydro" / "raw" / "fcst_rain",
        truth=Path.home() / "brazil_hydro" / "raw" / "imerg_basin_daily.json",
        basins=["GRANDE", "PARANAIBA", "TIETE", "PARANAPANEMA", "PARANA",
                "IGUACU", "URUGUAI", "JACUI", "SAO FRANCISCO", "TOCANTINS",
                "AMAZONAS", "PARAIBA DO SUL"],
        grid=(4, 3), fig=(13.5, 13.5),
        out=REPO / "brazil_hydro" / "bias_by_lead.webp",
        title="Brazil — NWP rainfall bias by lead time (vs gauge-corrected IMERG)"),
}
MAXLEAD = 15


def run(name, cfg):
    tc = json.loads(cfg["truth"].read_text())
    dmap = {f"{d[:4]}-{d[4:6]}-{d[6:8]}": i for i, d in enumerate(tc["dates"])}
    acc = {m: {b: {ld: [0.0, 0.0, 0.0, 0] for ld in range(1, MAXLEAD + 1)}
               for b in cfg["basins"]} for m in ("aifs", "ifs")}
    for f in sorted(cfg["arch"].glob("*.json.gz")):
        rec = json.loads(gzip.open(f, "rt").read())
        mdl = rec["model"]
        d0 = np.datetime64(f"{rec['init_date'][:4]}-{rec['init_date'][4:6]}-"
                           f"{rec['init_date'][6:8]}")
        for li, vd in enumerate(rec["valid"]):
            i = dmap.get(vd)
            if i is None:
                continue
            lead = int((np.datetime64(vd) - d0).astype(int)) + 1
            if not (1 <= lead <= MAXLEAD):
                continue
            for b in cfg["basins"]:
                if b not in rec["basins"] or b not in tc:
                    continue
                obs = tc[b][i]
                fc = float(np.mean([m_[li] for m_ in rec["basins"][b]]))
                a = acc[mdl][b][lead]
                a[0] += fc
                a[1] += obs
                a[2] += abs(fc - obs)
                a[3] += 1
    rows, cols = cfg["grid"]
    fig, axes = plt.subplots(rows, cols, figsize=cfg["fig"], sharex=True)
    leads = np.arange(1, MAXLEAD + 1)
    for ax, b in zip(np.ravel(axes), cfg["basins"]):
        allv = []
        for mdl, col in (("aifs", "#1f4e8c"), ("ifs", "#c62828")):
            ratio = [acc[mdl][b][ld][0] / acc[mdl][b][ld][1]
                     if acc[mdl][b][ld][1] > 0 and acc[mdl][b][ld][3] >= 8
                     else np.nan for ld in leads]
            allv += [v for v in ratio if np.isfinite(v)]
            n1 = acc[mdl][b][1][3]
            ax.plot(leads, ratio, color=col, lw=1.8, marker="o", ms=3.5,
                    label=f"{mdl.upper()} (n={n1}/lead)")
        ax.axhline(1.0, color="0.4", lw=0.9, ls="--")
        ax.set_yscale("log")
        lo = min(0.25, (min(allv) / 1.25) if allv else 0.25)
        hi = max(8.0, (max(allv) * 1.25) if allv else 8.0)
        ax.set_ylim(lo, hi)
        tick_all = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
        tk = [t for t in tick_all if lo <= t <= hi]
        ax.set_yticks(tk)
        ax.set_yticklabels([f"{t:g}" for t in tk])
        ax.set_title(b, fontsize=10, fontweight="bold", loc="left")
        ax.grid(lw=0.25, alpha=0.5)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("fcst / obs", fontsize=8)
        if b == cfg["basins"][0]:
            ax.legend(fontsize=7.5, loc="upper left")
        ax.set_xticks([1, 3, 5, 7, 10, 13, 15])
    for ax in np.ravel(axes)[len(cfg["basins"]):]:
        ax.set_axis_off()
    for ax in np.ravel(axes)[-cols:]:
        ax.set_xlabel("lead (days)", fontsize=8.5)
    fig.suptitle(cfg["title"] + "\nfcst/obs ratio per lead · >1 = too wet "
                 "· log scale · dry-season ratios inflate on small denominators",
                 fontsize=11.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(cfg["out"], dpi=115)
    plt.close(fig)
    print(f"wrote {cfg['out'].relative_to(REPO)}")


def main() -> int:
    for name, cfg in CFG.items():
        run(name, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
