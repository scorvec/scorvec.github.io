#!/usr/bin/env python3
"""Rolling verification of the bias-corrected AIFS-ENS basin rain forecast.

Requested 2026-08-18: start tracking how the corrected AIFS-ENS actually
performs, and keep accumulating.  Two things make this different from the
scorecard already on the page:

  1. WALK-FORWARD CORRECTION.  The operational bias factors in
     fcst_verif.json are fitted on every matured pair, this one included,
     so scoring against them is in-sample and flatters the model.  Here
     each valid date is corrected using only pairs that matured STRICTLY
     BEFORE it — the factor you would genuinely have had that morning.
     Raw and in-sample-corrected are reported alongside, so the value the
     correction adds is visible rather than assumed.

  2. THE WHOLE ENSEMBLE, NOT THE MEAN.  Tracks CRPS and the rank
     histogram, so under- or over-dispersion shows up.  A model whose
     mean verifies well but whose spread is wrong is still unusable for
     the probability distributions the inflow fans are built on.

Pairs are persisted to a permanent append-only archive keyed by
(model, init, basin, lead); the GRIBs behind them are pruned at +7 d, so
this file is the record.

    python scripts/sst/aifs_tracker.py

Outputs:
  ~/colombia_hydro/raw/aifs_verif_pairs.json.gz   (permanent)
  ~/colombia_hydro/out/aifs_tracker.json
  ~/colombia_hydro/site/aifs_tracker.webp
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from colombia_forecast import (ARCH, ORDER, BANDS, K_PRIOR, PRIOR_RATIO,  # noqa: E402
                               TRUTH_CACHE, _bas)

PRIV = Path.home() / "colombia_hydro"
PAIRS = PRIV / "raw" / "aifs_verif_pairs.json.gz"
OUT_JSON = PRIV / "out" / "aifs_tracker.json"
OUT_PNG = PRIV / "site" / "aifs_tracker.webp"
MAX_LEAD = 15


def band_of(lead):
    return next((i for i, (a, b) in enumerate(BANDS) if a <= lead <= b), None)


def build_pairs():
    """Every (model, init, basin, lead) pair with a matured truth value."""
    tc = json.loads(TRUTH_CACHE.read_text())
    truth = {r: dict(zip(tc["dates"], tc[r])) for r in ORDER}
    tclim = {r: float(np.nanmean(np.asarray(tc[r + "_clim"], float))) for r in ORDER}
    rows = []
    for f in sorted(glob.glob(str(ARCH / "*.json.gz"))):
        with gzip.open(f, "rt") as fh:
            rec = json.load(fh)
        init = f"{rec['init_date']}{rec['init_hh']}"
        d0 = np.datetime64(f"{rec['init_date'][:4]}-{rec['init_date'][4:6]}-"
                           f"{rec['init_date'][6:8]}")
        for li, vd in enumerate(rec["valid"]):
            lead = int((np.datetime64(vd) - d0).astype(int)) + 1
            if not 1 <= lead <= MAX_LEAD:
                continue
            key = vd.replace("-", "")
            for r in ORDER:
                obs = truth[r].get(key)
                if obs is None or not np.isfinite(obs):
                    continue
                mem = np.asarray([m[li] for m in _bas(rec, r)], float)
                if not mem.size or not np.isfinite(mem).all():
                    continue
                rows.append({"model": rec["model"], "init": init, "valid": vd,
                             "lead": lead, "basin": r,
                             "mean": round(float(mem.mean()), 3),
                             "sd": round(float(mem.std()), 3),
                             "obs": round(float(obs), 3),
                             "mem": [round(float(v), 2) for v in mem]})
    return rows, tclim


def walk_forward(rows, tclim):
    """Correct each pair with factors from strictly earlier valid dates."""
    rows = sorted(rows, key=lambda r: (r["valid"], r["init"]))
    acc = defaultdict(lambda: [0.0, 0.0])          # (model,basin,band) -> sums
    seen_upto = None
    pend = []
    for r in rows:
        if r["valid"] != seen_upto:                # fold in everything older
            for p in pend:
                k = (p["model"], p["basin"], band_of(p["lead"]))
                acc[k][0] += p["mean"]
                acc[k][1] += p["obs"]
            pend = []
            seen_upto = r["valid"]
        b = band_of(r["lead"])
        sf, so = acc[(r["model"], r["basin"], b)]
        P = K_PRIOR * tclim[r["basin"]]
        F = (so + P * PRIOR_RATIO[r["model"]]) / max(sf + P, 1e-6)
        r["F_wf"] = round(float(F), 4)
        r["corr"] = round(float(r["mean"] * F), 3)
        r["mem_corr"] = [round(v * F, 2) for v in r["mem"]]
        pend.append(r)
    return rows


def crps_ens(mem, obs):
    m = np.sort(np.asarray(mem, float))
    n = len(m)
    t1 = np.mean(np.abs(m - obs))
    i = np.arange(n)
    t2 = 2.0 * np.sum((2 * i - n + 1) * m) / (n * n)
    return float(t1 - 0.5 * t2)


def score(rows, sel, key="corr"):
    p = np.array([r[key] for r in rows if sel(r)])
    o = np.array([r["obs"] for r in rows if sel(r)])
    if len(p) < 8:
        return None
    return {"n": int(len(p)),
            "r": round(float(np.corrcoef(p, o)[0, 1]), 3) if np.std(p) > 1e-9 else None,
            "mae": round(float(np.mean(np.abs(p - o))), 2),
            "bias": round(float(np.mean(p - o)), 2),
            "rmse": round(float(np.sqrt(np.mean((p - o) ** 2))), 2)}


def main() -> int:
    rows, tclim = build_pairs()
    if not rows:
        print("no matured pairs yet")
        return 0
    rows = walk_forward(rows, tclim)

    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(PAIRS, "wt") as f:                # permanent record
        json.dump({"generated": datetime.now(timezone.utc)
                   .strftime("%Y-%m-%d %H:%M UTC"),
                   "rows": [{k: v for k, v in r.items() if k != "mem"}
                            for r in rows]}, f, separators=(",", ":"))

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "pairs": len(rows),
           "window": f"{rows[0]['valid']}..{rows[-1]['valid']}",
           "note": "corr = walk-forward bias correction (factors from pairs "
                   "with strictly earlier valid dates only)",
           "by_model_band": {}, "by_basin": {}, "crps": {}, "rank_hist": {}}

    for mdl in ("aifs", "ifs"):
        out["by_model_band"][mdl] = {}
        for bi, (a, b) in enumerate(BANDS):
            sel = lambda r, m=mdl, x=a, y=b: (r["model"] == m and x <= r["lead"] <= y)
            out["by_model_band"][mdl][f"d{a}-{b}"] = {
                "raw": score(rows, sel, "mean"),
                "corrected_walkforward": score(rows, sel, "corr")}
    for r_ in ORDER:
        sel = lambda r, x=r_: r["model"] == "aifs" and r["basin"] == x
        out["by_basin"][r_] = {"raw": score(rows, sel, "mean"),
                               "corrected_walkforward": score(rows, sel, "corr")}

    # ensemble calibration, AIFS only
    for bi, (a, b) in enumerate(BANDS):
        sub = [r for r in rows if r["model"] == "aifs" and a <= r["lead"] <= b]
        if len(sub) < 20:
            continue
        out["crps"][f"d{a}-{b}"] = {
            "crps_corrected": round(float(np.mean(
                [crps_ens(r["mem_corr"], r["obs"]) for r in sub])), 3),
            "crps_raw": round(float(np.mean(
                [crps_ens(r["mem"], r["obs"]) for r in sub])), 3),
            "n": len(sub)}
        ranks = [int(np.sum(np.asarray(r["mem_corr"]) < r["obs"])) /
                 max(len(r["mem_corr"]), 1) for r in sub]
        out["rank_hist"][f"d{a}-{b}"] = np.histogram(
            ranks, bins=10, range=(0, 1))[0].tolist()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"pairs: {len(rows)}  {out['window']}")
    print(f"{'band':8}{'AIFS raw mae':>14}{'AIFS corr mae':>15}"
          f"{'corr r':>8}{'CRPS raw':>10}{'CRPS corr':>11}")
    for bi, (a, b) in enumerate(BANDS):
        k = f"d{a}-{b}"
        m = out["by_model_band"]["aifs"][k]
        c = out["crps"].get(k, {})
        if m["raw"]:
            print(f"{k:8}{m['raw']['mae']:14.2f}"
                  f"{m['corrected_walkforward']['mae']:15.2f}"
                  f"{m['corrected_walkforward']['r']:8.3f}"
                  f"{c.get('crps_raw', float('nan')):10.3f}"
                  f"{c.get('crps_corrected', float('nan')):11.3f}")
    try:
        figure(out, rows)
    except Exception as e:                              # noqa: BLE001
        print(f"figure failed: {repr(e)[:120]}")
    print(f"wrote {OUT_JSON}")
    return 0


def figure(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    NAVY, INK = "#13273d", "#1a2733"
    fig = plt.figure(figsize=(14.5, 9.6))
    hd = fig.add_axes([0, 0.94, 1, 0.06]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.015, 0.62, "AIFS-ENS BASIN RAIN — LIVE VERIFICATION",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    hd.text(0.015, 0.2, "bias correction applied WALK-FORWARD: each day uses "
            "only factors earned from earlier days", transform=hd.transAxes,
            color="#b9c6d4", fontsize=9, va="center")
    hd.text(0.985, 0.5, f"{out['window']} · {out['pairs']} pairs",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9,
            va="center", ha="right")

    ax = fig.add_axes([0.055, 0.55, 0.40, 0.33])
    ks = list(out["by_model_band"]["aifs"])
    x = np.arange(len(ks))
    for off, key, c, lab in ((-0.2, "raw", "#9db8d8", "raw"),
                             (0.2, "corrected_walkforward", "#1f4e8c",
                              "corrected (walk-forward)")):
        v = [out["by_model_band"]["aifs"][k][key]["mae"]
             if out["by_model_band"]["aifs"][k][key] else np.nan for k in ks]
        ax.bar(x + off, v, 0.38, color=c, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(ks)
    ax.set_ylabel("MAE, mm/day", fontsize=9)
    ax.set_title("Does the correction actually help?", fontsize=10.5,
                 fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=8); ax.grid(lw=0.25, alpha=0.5, axis="y")
    ax.tick_params(labelsize=8)

    ax2 = fig.add_axes([0.56, 0.55, 0.40, 0.33])
    for r_ in ORDER:
        v = out["by_basin"][r_]["corrected_walkforward"]
        if v:
            ax2.barh(r_, v["mae"], color="#1f4e8c", alpha=0.85)
    ax2.set_xlabel("MAE, mm/day (AIFS corrected, all leads)", fontsize=9)
    ax2.set_title("Where the error lives", fontsize=10.5, fontweight="bold",
                  loc="left", color=INK)
    ax2.grid(lw=0.25, alpha=0.5, axis="x"); ax2.tick_params(labelsize=8)

    ax3 = fig.add_axes([0.055, 0.09, 0.40, 0.33])
    ds = sorted({r["valid"] for r in rows})
    dt = [datetime.strptime(x, "%Y-%m-%d") for x in ds]
    for key, c, lab in (("mean", "#9db8d8", "raw"), ("corr", "#1f4e8c", "corrected")):
        v = []
        for x in ds:
            s = [r for r in rows if r["valid"] == x and r["model"] == "aifs"]
            v.append(np.mean([abs(r[key] - r["obs"]) for r in s]) if s else np.nan)
        ax3.plot(dt, v, lw=1.4, color=c, marker="o", ms=2.5, label=lab)
    ax3.set_ylabel("daily MAE, mm/day", fontsize=9)
    ax3.set_title("Day by day — this is the series that accumulates",
                  fontsize=10.5, fontweight="bold", loc="left", color=INK)
    ax3.legend(fontsize=8); ax3.grid(lw=0.25, alpha=0.5)
    ax3.tick_params(labelsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax4 = fig.add_axes([0.56, 0.09, 0.40, 0.33])
    for k, c in zip(out["rank_hist"], ("#1f4e8c", "#e08214", "#2e7d32")):
        h = np.asarray(out["rank_hist"][k], float)
        ax4.plot(np.linspace(0.05, 0.95, len(h)), h / h.sum(), lw=1.6,
                 marker="o", ms=3, color=c, label=k)
    ax4.axhline(0.1, color="0.45", lw=1.0, ls="--", label="calibrated")
    ax4.set_xlabel("ensemble rank of the observation", fontsize=9)
    ax4.set_ylabel("frequency", fontsize=9)
    ax4.set_title("Rank histogram — U-shape means under-dispersed",
                  fontsize=10.5, fontweight="bold", loc="left", color=INK)
    ax4.legend(fontsize=8); ax4.grid(lw=0.25, alpha=0.5)
    ax4.tick_params(labelsize=8)
    fig.text(0.055, 0.02, "truth = gauge-blended corrected IMERG on the "
             "energy-weighted basin masks · pairs archived permanently in "
             "raw/aifs_verif_pairs.json.gz", fontsize=7.5, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=118); plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    raise SystemExit(main())
