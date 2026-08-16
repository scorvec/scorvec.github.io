#!/usr/bin/env python3
"""Full XM inflow (AporEner) history + per-region day-of-year climatology.

Pulls the complete per-river daily inflow-energy record from XM's public
API (chunked <=30-day requests, incrementally cached — a rerun only
fetches days newer than the cache), aggregates to the six hydro regions
using ALL rivers ever listed for a region (inactive rivers simply stop
reporting; filtering to ACTIVO would delete their history), and reduces
to the seasonal-norm products the page needs:

  - day-of-year climatology per region: mean + p10/p25/p50/p75/p90,
    +/-10-day window, over the full record (2000-);
  - percent-of-normal daily series;
  - 90-day (seasonal) and 365-day (annual) running means, full history.

Outputs:
  ~/colombia_hydro/raw/aporener_daily.json.gz     (cache, {date:{river:kWh}})
  colombia_hydro/data/inflow_clim.json            (climatology + recent series)
  colombia_hydro/inflow_norms.webp                (6 regions vs norms + running means)

    python scripts/sst/xm_inflow_history.py [--start 2000-01-01]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RAW = Path.home() / "colombia_hydro" / "raw"
CACHE = RAW / "aporener_daily.json.gz"
RIVERS_JSON = RAW / "xm_listado_rios.json"
OUT_JSON = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
OUT_PNG = REPO / "colombia_hydro" / "inflow_norms.webp"
API = "https://servapibi.xm.com.co/daily"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
PCTS = [10, 25, 50, 75, 90]
WIN = 10                        # +/- days pooled per day-of-year


# Historical names that predate the current ListadoRios (verified 2026-08-16:
# succession seams are exact, zero overlap with their successors):
#   MAGDALENA BETANIA (..2015-10-31) -> BETANIA CP     : CENTRO
#   NARE              (..2023-08-31) -> NARE CP        : ANTIOQUIA
#   PORCE II          (..2018-04-30) -> PORCE2 CP      : ANTIOQUIA
#   FLORIDA II        (..2025-07-30, retired)          : VALLE (Rio Frio RoR)
# Together these carried 18% of all-record inflow energy — dropping them
# gutted the pre-2016 climatology. OTROS RIOS (ESTIMADOS) is a regionless
# system-wide estimate (~4 GWh/d to 2022) and is excluded deliberately.
ALIASES = {"MAGDALENA BETANIA": "CENTRO", "NARE": "ANTIOQUIA",
           "PORCE II": "ANTIOQUIA", "FLORIDA II": "VALLE"}
EXCLUDE = {"OTROS RIOS (ESTIMADOS)"}


def river_region() -> dict[str, str]:
    d = json.load(open(RIVERS_JSON))
    out = dict(ALIASES)
    for it in d["Items"]:
        for e in it["ListEntities"]:
            v = e["Values"]
            if v.get("HydroRegion") in ORDER:
                out[v["Name"].strip().upper()] = v["HydroRegion"]
    return out


def fetch_range(d0: datetime, d1: datetime) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    cur = d0
    while cur <= d1:
        end = min(cur + timedelta(days=29), d1)
        for attempt in range(3):
            try:
                r = requests.post(API, json={"MetricId": "AporEner",
                                             "StartDate": f"{cur:%Y-%m-%d}",
                                             "EndDate": f"{end:%Y-%m-%d}",
                                             "Entity": "Rio"}, timeout=90)
                r.raise_for_status()
                break
            except Exception as e:                    # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"  retry {cur:%Y-%m} ({repr(e)[:40]})", flush=True)
        for it in r.json().get("Items", []):
            day = it["Date"]
            for e in it.get("DailyEntities", []):
                out.setdefault(day, {})[e["Name"].strip().upper()] = float(e["Value"])
        print(f"  {cur:%Y-%m-%d}..{end:%Y-%m-%d}: {len(out)} days total", flush=True)
        cur = end + timedelta(days=1)
    return out


def load_cache() -> dict:
    if CACHE.exists():
        with gzip.open(CACHE, "rt") as f:
            return json.load(f)
    return {}


def save_cache(d: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE, "wt") as f:
        json.dump(d, f, separators=(",", ":"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2000-01-01")
    a = ap.parse_args()
    start = datetime.strptime(a.start, "%Y-%m-%d")
    end = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)

    data = load_cache()
    have = sorted(data)
    # backfill anything older than the cache head, and refresh the tail
    # (XM revises the last few days)
    if not have or have[0] > f"{start:%Y-%m-%d}":
        first = datetime.strptime(have[0], "%Y-%m-%d") - timedelta(days=1) if have else end
        if first >= start:
            print(f"backfilling {start:%Y-%m-%d}..{first:%Y-%m-%d}", flush=True)
            data.update(fetch_range(start, first))
    tail0 = (datetime.strptime(have[-1], "%Y-%m-%d") - timedelta(days=5)) if have else start
    if tail0 <= end:
        print(f"updating {tail0:%Y-%m-%d}..{end:%Y-%m-%d}", flush=True)
        data.update(fetch_range(tail0, end))
    save_cache(data)
    print(f"cache: {len(data)} days ({min(data)}..{max(data)})", flush=True)

    # ── aggregate to regions (GWh/day) ──────────────────────────────────────
    r2r = river_region()
    days = sorted(data)
    dates = np.array(days, dtype="datetime64[D]")
    reg = {r: np.zeros(len(days)) for r in ORDER}
    unmatched = set()
    for i, d in enumerate(days):
        for riv, kwh in data[d].items():
            if riv in EXCLUDE:
                continue
            r = r2r.get(riv)
            if r is None:
                unmatched.add(riv)
                continue
            reg[r][i] += kwh / 1e6
    if unmatched:
        print(f"note: {len(unmatched)} river names not in ListadoRios "
              f"(unmetered/aliases): {sorted(unmatched)[:6]} …", flush=True)

    doy = np.array([(datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday)
                    for d in days])
    doy = np.minimum(doy, 365)

    clim = {}
    for r in ORDER:
        v = reg[r]
        mu = np.zeros(365)
        pct = np.zeros((365, len(PCTS)))
        for dd in range(1, 366):
            dist = np.minimum(np.abs(doy - dd), 365 - np.abs(doy - dd))
            m = (dist <= WIN) & (v > 0)
            mu[dd - 1] = v[m].mean() if m.any() else np.nan
            pct[dd - 1] = np.percentile(v[m], PCTS) if m.sum() > 20 else np.nan
        clim[r] = {"mean": mu, "pct": pct}

    # ── figure: norms + running means ───────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 13.5))
    x365 = np.arange(1, 366)
    this_year = dates >= (dates[-1] - np.timedelta64(365, "D"))
    for ax, r in zip(axes.flat[:6], ORDER):
        c = clim[r]
        ax.fill_between(x365, c["pct"][:, 0], c["pct"][:, 4], color="#9db8d8",
                        alpha=0.35, label="p10–p90")
        ax.fill_between(x365, c["pct"][:, 1], c["pct"][:, 3], color="#5b87c0",
                        alpha=0.35, label="p25–p75")
        ax.plot(x365, c["pct"][:, 2], color="#1f4e8c", lw=1.4, label="median")
        recent = dates > dates[-1] - np.timedelta64(200, "D")
        dd = np.minimum(np.array([(datetime.strptime(str(x), "%Y-%m-%d")
                                   .timetuple().tm_yday) for x in dates[recent]]), 365)
        ax.plot(dd, reg[r][recent], color="#c62828", lw=1.3,
                label="last 200 days")
        ax.set_title(r, fontsize=10, fontweight="bold", loc="left")
        ax.set_xlim(1, 365)
        ax.set_xticks([1, 91, 182, 274, 365])
        ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Jan"], fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(lw=0.25, alpha=0.5)
        if r == ORDER[0]:
            ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel("GWh/day", fontsize=8)

    # bottom row: 90-day and 365-day running means of the NATIONAL total
    total = sum(reg[r] for r in ORDER)
    t = dates.astype("datetime64[s]").astype(datetime)
    for ax, (win, lab) in zip(axes.flat[6:], [(90, "90-day (seasonal) mean"),
                                              (365, "365-day (annual) mean")]):
        k = np.ones(win) / win
        rm = np.convolve(total, k, mode="valid")
        ax.plot(t[win - 1:], rm, color="#1f4e8c", lw=1.2)
        # climatological norm run through the SAME window: the daily norm
        # oscillates seasonally; its 365-day mean is nearly flat
        daily_norm = np.array([sum(clim[r]["mean"][min(d, 365) - 1] for r in ORDER)
                               for d in doy])
        norm_rm = np.convolve(daily_norm, k, mode="valid")
        ax.plot(t[win - 1:], norm_rm, color="0.5", lw=0.9, ls="--", label="norm")
        ax.set_title(f"All regions — {lab}", fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(lw=0.25, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_ylabel("GWh/day", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("XM hydro inflows vs seasonal norms — full-record climatology "
                 f"({min(data)[:4]}–{max(data)[:4]}, ±{WIN}-day windows)",
                 fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=115)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")

    # ── JSON feed ───────────────────────────────────────────────────────────
    keep = dates > dates[-1] - np.timedelta64(3 * 365, "D")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "base_period": f"{min(data)[:4]}-{max(data)[:4]}",
        "pcts": PCTS,
        "clim": {r: {"mean": np.round(clim[r]["mean"], 2).tolist(),
                     "pct": np.round(clim[r]["pct"], 2).tolist()} for r in ORDER},
        "recent": {"dates": [str(x) for x in dates[keep]],
                   **{r: np.round(reg[r][keep], 2).tolist() for r in ORDER}},
    }, separators=(",", ":")))
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
