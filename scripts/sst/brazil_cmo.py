#!/usr/bin/env python3
"""Brazil: rainfall -> prices. CMO (weekly marginal operating cost, the
anchor of the PLD settlement price) vs storage (EAR) and inflows (ENA).

The chain: rain integrates into ENA (energy-space rainfall), ENA
integrates into EAR (storage), and the hydrothermal optimization prices
water from EAR + expected ENA. So the rainfall-price relation should
peak at LONG memory and act through storage. This script measures that
on 21 years of weekly CMO by subsystem.

Outputs:
  ~/brazil_hydro/raw/cmo_weekly.json.gz
  ~/brazil_hydro/out/cmo_analysis.json
  brazil_hydro/cmo_price.webp

    python scripts/sst/brazil_cmo.py [--backfill]
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO = Path(__file__).resolve().parent.parent.parent
PRIV = Path.home() / "brazil_hydro"
CACHE = PRIV / "raw" / "cmo_weekly.json.gz"
ENA_S = PRIV / "raw" / "ena_subsistema_daily.json.gz"
EAR_S = PRIV / "raw" / "ear_subsistema_daily.json.gz"
OUT_JSON = PRIV / "out" / "cmo_analysis.json"
OUT_PNG = REPO / "brazil_hydro" / "cmo_price.webp"
BASE = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/cmo_se"
SUBS = {"SUDESTE/CENTRO-OESTE": "SE/CO", "SUL": "SUL",
        "NORDESTE": "NORDESTE", "NORTE": "NORTE"}
FLOOR = 40.0                                   # R$/MWh, log floor


def fetch(backfill: bool) -> dict:
    data = {}
    if CACHE.exists():
        with gzip.open(CACHE, "rt") as f:
            data = json.load(f)
    thisyear = datetime.now().year
    years = (range(2005, thisyear + 1) if backfill or not data
             else [thisyear])
    for y in years:
        try:
            with urllib.request.urlopen(
                    f"{BASE}/CMO_SEMANAL_{y}.csv", timeout=120) as r:
                rows = list(csv.DictReader(
                    io.StringIO(r.read().decode("utf-8")), delimiter=";"))
        except Exception as e:                  # noqa: BLE001
            print(f"  {y}: {repr(e)[:60]}")
            continue
        n = 0
        for row in rows:
            sub = row["id_subsistema"].strip().upper()
            try:
                v = float(row["val_cmomediasemanal"])
            except (ValueError, KeyError):
                continue
            data.setdefault(sub, {})[row["din_instante"][:10]] = round(v, 2)
            n += 1
        print(f"  CMO {y}: {n} rows", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE, "wt") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def main() -> int:
    backfill = "--backfill" in sys.argv[1:]
    cmo = fetch(backfill)
    with gzip.open(ENA_S, "rt") as f:
        ena = json.load(f)
    with gzip.open(EAR_S, "rt") as f:
        ear = json.load(f)
    SUBNAME = "SUDESTE"

    sub = "SE"                                  # the price-setting subsystem
    wdays = sorted(cmo[sub])
    wd = np.array(wdays, dtype="datetime64[D]")
    p = np.array([cmo[sub][d] for d in wdays], float)
    logp = np.log(np.maximum(p, FLOOR))

    edays = sorted(ena[SUBNAME])
    edt = np.array(edays, dtype="datetime64[D]")
    epct = np.array([ena[SUBNAME][d][1] for d in edays], float)
    sdays = sorted(ear[SUBNAME])
    sdt = np.array(sdays, dtype="datetime64[D]")
    spct = np.array([ear[SUBNAME][d] for d in sdays], float)

    # ENA smoothed over various memories, sampled at CMO weeks
    def at_weeks(series_dt, series_v, smooth_n):
        k = np.ones(smooth_n) / smooth_n
        sm = np.convolve(np.where(np.isfinite(series_v), series_v,
                                  np.nanmean(series_v)), k, "full")[:len(series_v)]
        sm[:smooth_n - 1] = np.nan
        idx = np.searchsorted(series_dt, wd)
        idx = np.clip(idx, 0, len(series_v) - 1)
        out = sm[idx]
        out[(wd < series_dt[0]) | (wd > series_dt[-1])] = np.nan
        return out

    mems = [7, 14, 30, 60, 90, 180, 365]
    memcorr = []
    for n in mems:
        x = at_weeks(edt, epct, n)
        m = np.isfinite(x) & np.isfinite(logp)
        memcorr.append(round(float(np.corrcoef(x[m], logp[m])[0, 1]), 3))
    earw = at_weeks(sdt, spct, 7)
    m = np.isfinite(earw) & np.isfinite(logp)
    r_ear = float(np.corrcoef(earw[m], logp[m])[0, 1])
    ena90 = at_weeks(edt, epct, 90)
    m2 = m & np.isfinite(ena90)
    X = np.column_stack([np.ones(m2.sum()), earw[m2], ena90[m2]])
    beta, *_ = np.linalg.lstsq(X, logp[m2], rcond=None)
    fit = X @ beta
    r_both = float(np.corrcoef(fit, logp[m2])[0, 1])

    stats = {"generated": datetime.now(timezone.utc)
             .strftime("%Y-%m-%d %H:%M UTC"),
             "subsystem": sub, "weeks": int(m.sum()),
             "corr_logCMO_vs_ENA_by_memory_days": dict(zip(map(str, mems),
                                                           memcorr)),
             "corr_logCMO_vs_EAR": round(r_ear, 3),
             "joint_fit_r": round(r_both, 3),
             "joint_coefs": {"intercept": round(float(beta[0]), 3),
                             "per_EAR_pct": round(float(beta[1]), 4),
                             "per_ENA90_pct": round(float(beta[2]), 4)},
             "note": ("log CMO floored at R$40; ENA %MLT smoothed over "
                      "N days; storage (EAR) is the dominant channel — "
                      "rain reaches prices through the reservoir integral")}
    OUT_JSON.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)

    # ── figure ──────────────────────────────────────────────────────────────
    t = wd.astype("datetime64[s]").astype(datetime)
    fig = plt.figure(figsize=(13.5, 9.5))
    ax = fig.add_axes([0.06, 0.55, 0.9, 0.38])
    ax.plot(t, np.maximum(p, FLOOR), color="#c62828", lw=1.0, label="CMO SE/CO (R$/MWh, floored)")
    ax.set_yscale("log")
    ax.set_ylabel("CMO, R$/MWh (log)", fontsize=9, color="#c62828")
    ax.tick_params(axis="y", labelcolor="#c62828", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax2 = ax.twinx()
    si = np.searchsorted(sdt, wd).clip(0, len(spct) - 1)
    ax2.plot(t, spct[si], color="#1f4e8c", lw=1.3,
             label="EAR SE/CO (% of max)")
    ax2.set_ylabel("EAR, % of max", fontsize=9, color="#1f4e8c")
    ax2.tick_params(axis="y", labelcolor="#1f4e8c", labelsize=8)
    ax2.invert_yaxis()
    ax.set_title("Price vs storage, 2005–2026 — CMO (log, red) against "
                 "EAR (blue, inverted): empty reservoirs = expensive power",
                 fontsize=11, fontweight="bold", loc="left")
    ax.grid(lw=0.25, alpha=0.5)

    ax3 = fig.add_axes([0.06, 0.07, 0.40, 0.36])
    sc = ax3.scatter(earw[m2], np.exp(logp[m2]), c=ena90[m2], cmap="BrBG",
                     vmin=40, vmax=160, s=10, alpha=0.75)
    ax3.set_yscale("log")
    ax3.set_xlabel("EAR SE/CO, % of max", fontsize=9)
    ax3.set_ylabel("CMO, R$/MWh (log)", fontsize=9)
    ax3.set_title(f"Every week since 2005 · r(log CMO, EAR) = {r_ear:.2f}",
                  fontsize=10, fontweight="bold", loc="left")
    ax3.grid(lw=0.25, alpha=0.5)
    ax3.tick_params(labelsize=8)
    cb = fig.colorbar(sc, ax=ax3, pad=0.02)
    cb.set_label("ENA %MLT, 90-day mean", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax4 = fig.add_axes([0.58, 0.07, 0.38, 0.36])
    ax4.plot(mems, [-c for c in memcorr], color="#13273d", lw=2.0,
             marker="o")
    ax4.set_xscale("log")
    ax4.set_xticks(mems)
    ax4.set_xticklabels([str(m_) for m_ in mems], fontsize=8)
    ax4.set_xlabel("ENA smoothing memory (days)", fontsize=9)
    ax4.set_ylabel("−corr(log CMO, ENA %MLT)", fontsize=9)
    ax4.set_title("How much rain memory do prices carry?\n"
                  "correlation strengthens with accumulation window",
                  fontsize=10, fontweight="bold", loc="left")
    ax4.grid(lw=0.25, alpha=0.5)
    ax4.tick_params(labelsize=8)
    fig.suptitle("Brazil — rainfall to prices: the reservoir is the "
                 "transmission mechanism", fontsize=13, fontweight="bold",
                 y=0.985)
    fig.savefig(OUT_PNG, dpi=115)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
