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
import os
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
# CO_INFLOW_METRIC selects the target. AporEner (default) is kWh and
# therefore volume x the head available downstream, so it moves when the
# FLEET changes: CAUCA SALVAJINA's kWh-per-m3/s stepped 20,278 -> 63,190
# at Hidroituango commissioning, and because riv_mu is a whole-record
# climatology the same hydrology then reads 222% of norm after the step
# and 81% before. AporCaudal is m3/s - pure water, no turbines - and is
# what rain actually drives.
METRIC = os.environ.get("CO_INFLOW_METRIC", "AporEner")
_VOL = METRIC == "AporCaudal"
# kWh -> GWh for energy; m3/s needs no conversion
SCALE = 1.0 if _VOL else 1e6
_SFX = "_vol" if _VOL else ""
CACHE = RAW / (f"aporcaudal_daily.json.gz" if _VOL else "aporener_daily.json.gz")
RIVERS_JSON = RAW / "xm_listado_rios.json"
OUT_JSON = REPO / "colombia_hydro" / "data" / f"inflow_clim{_SFX}.json"
OUT_PNG = REPO / "colombia_hydro" / f"inflow_norms{_SFX}.webp"
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
# Renames: successor series continue the SAME river (seams verified exact) —
# merged into one series for per-river climatologies
SUCCESSOR = {"MAGDALENA BETANIA": "BETANIA CP", "NARE": "NARE CP",
             "PORCE II": "PORCE2 CP"}


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
                r = requests.post(API, json={"MetricId": METRIC,
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

    # ── per-river series (successors merged), regions, current fleet ──────
    # The 2000-2026 as-metered regional totals double as the fleet grows
    # (18 -> 41 rivers; Ituango alone is ~1/3 of Antioquia today), so norms
    # built on raw totals read recent years "wet" for free. Fix: each
    # river gets a day-of-year climatology over ITS OWN record, the
    # regional norm is the SUM over the CURRENT fleet, and the envelope
    # comes from the fleet-normalized ratio r(t) = sum(obs, reporting
    # subset) / sum(norms, same subset) — numerator and denominator always
    # share the same rivers, so partial fleets can't bias it.
    r2r = river_region()
    days = sorted(data)
    dates = np.array(days, dtype="datetime64[D]")
    nd = len(days)
    doy = np.minimum(np.array([(datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday)
                               for d in days]), 365)

    canon = {old: new for old, new in SUCCESSOR.items()}
    riv_series: dict[str, np.ndarray] = {}
    for i, d in enumerate(days):
        for riv, kwh in data[d].items():
            if riv in EXCLUDE:
                continue
            name = canon.get(riv, riv)
            if name not in riv_series:
                riv_series[name] = np.full(nd, np.nan)
            v = riv_series[name][i]
            riv_series[name][i] = (0.0 if np.isnan(v) else v) + kwh / SCALE
    unmatched = sorted(r for r in riv_series if r2r.get(r) is None)
    if unmatched:
        print(f"note: unmapped rivers dropped from norms: {unmatched}", flush=True)
        for r in unmatched:
            riv_series.pop(r)

    recent = dates > dates[-1] - np.timedelta64(30, "D")
    fleet = {r for r, v in riv_series.items() if np.nansum(v[recent]) > 0}
    print(f"current fleet: {len(fleet)} rivers "
          f"({len(riv_series)} with history)", flush=True)

    # per-river doy mean (±WIN window over its own record)
    riv_mu = {}
    for r, v in riv_series.items():
        ok = np.isfinite(v) & (v >= 0)
        mu = np.full(365, np.nan)
        for dd in range(1, 366):
            dist = np.minimum(np.abs(doy - dd), 365 - np.abs(doy - dd))
            m = ok & (dist <= WIN)
            if m.sum() >= 30:
                mu[dd - 1] = v[m].mean()
        riv_mu[r] = mu

    by_region = {reg: [r for r in riv_series if r2r.get(r) == reg] for reg in ORDER}
    reg = {}          # as-metered observed regional series (current-fleet rivers)
    norm_reg = {}     # fleet-corrected regional norm by doy (current fleet)
    ratio = {}        # fleet-normalized % of norm series
    nums, dens = {}, {}   # matched numerator/denominator (same reporting subset)
    for r_ in ORDER:
        cur = [x for x in by_region[r_] if x in fleet]
        obs = np.nansum([riv_series[x] for x in cur], axis=0)
        reg[r_] = obs
        norm_reg[r_] = np.nansum([riv_mu[x] for x in cur], axis=0)
        num = np.zeros(nd); den = np.zeros(nd)
        for x in cur:
            v = riv_series[x]
            m = np.isfinite(v)
            mu_at = riv_mu[x][doy - 1]
            use = m & np.isfinite(mu_at)
            num[use] += v[use]
            den[use] += mu_at[use]
        with np.errstate(invalid="ignore"):
            ratio[r_] = np.where(den > 0, num / den, np.nan)
        nums[r_], dens[r_] = num, den

    # regional envelope: norm × doy-windowed percentiles of the ratio
    clim = {}
    for r_ in ORDER:
        rt = ratio[r_]
        pct = np.zeros((365, len(PCTS)))
        for dd in range(1, 366):
            dist = np.minimum(np.abs(doy - dd), 365 - np.abs(doy - dd))
            m = (dist <= WIN) & np.isfinite(rt)
            pct[dd - 1] = (np.percentile(rt[m], PCTS) if m.sum() > 20 else np.nan)
        clim[r_] = {"mean": norm_reg[r_],
                    "pct": pct * norm_reg[r_][:, None]}

    # national fleet-normalized % of norm — matched subsets on BOTH sides:
    # each day divides the reporting rivers' inflow by the SAME rivers' norms
    # (dividing by the full-fleet norm made early low-fleet years read ~50%
    # spuriously; manual check 2003-06-15 = 1.11 with matched subsets)
    nat_num = np.sum([nums[r_] for r_ in ORDER], axis=0)
    nat_den = np.sum([dens[r_] for r_ in ORDER], axis=0)
    with np.errstate(invalid="ignore"):
        nat_ratio = np.where(nat_den > 0, nat_num / nat_den, np.nan)

    # ── forecast fan (colombia_forecast.py), drawn if <3 days old ───────────
    fan = None
    fan_path = REPO / "colombia_hydro" / "data" / "inflow_forecast.json"
    if fan_path.exists():
        f_ = json.loads(fan_path.read_text())
        age = (np.datetime64(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
               - np.datetime64(f_["dates"][0])).astype(int)
        if age <= 2:
            fan = f_

    # ── figure: norms + % of normal running means ───────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 13.5))
    x365 = np.arange(1, 366)
    for ax, r_ in zip(axes.flat[:6], ORDER):
        c = clim[r_]
        ax.fill_between(x365, c["pct"][:, 0] / 24, c["pct"][:, 4] / 24,
                        color="#9db8d8", alpha=0.35, label="p10–p90")
        ax.fill_between(x365, c["pct"][:, 1] / 24, c["pct"][:, 3] / 24,
                        color="#5b87c0", alpha=0.35, label="p25–p75")
        ax.plot(x365, c["pct"][:, 2] / 24, color="#1f4e8c", lw=1.4, label="median")
        rec = dates > dates[-1] - np.timedelta64(200, "D")
        dd = np.minimum(np.array([(datetime.strptime(str(x), "%Y-%m-%d")
                                   .timetuple().tm_yday) for x in dates[rec]]), 365)
        ax.plot(dd, reg[r_][rec] / 24, color="#c62828", lw=1.3, label="last 200 days")
        if fan is not None and r_ in fan["basins"]:
            # % of norm -> GWh/day via the same fleet-corrected doy norm;
            # clipped at the year wrap (fan is only ~2 weeks long)
            fdoy = np.array([min(datetime.strptime(x, "%Y-%m-%d")
                                 .timetuple().tm_yday, 365) for x in fan["dates"]])
            wrap = np.where(np.diff(fdoy) < 0)[0]
            stop = int(wrap[0]) + 1 if len(wrap) else len(fdoy)
            fd = fdoy[:stop]
            nrm = norm_reg[r_][fd - 1] / 100.0 / 24.0
            q = fan["basins"][r_]["q"]
            ax.fill_between(fd, np.array(q["p10"][:stop]) * nrm,
                            np.array(q["p90"][:stop]) * nrm,
                            color="#e08214", alpha=0.25, lw=0)
            ax.fill_between(fd, np.array(q["p25"][:stop]) * nrm,
                            np.array(q["p75"][:stop]) * nrm,
                            color="#e08214", alpha=0.35, lw=0)
            ax.plot(fd, np.array(q["p50"][:stop]) * nrm, color="#b35806",
                    lw=1.4, ls="--", label="AIFS+IFS ens forecast")
        ax.set_title(f"{r_} — current fleet, fleet-corrected norms",
                     fontsize=10, fontweight="bold", loc="left")
        ax.set_xlim(1, 365)
        ax.set_xticks([1, 91, 182, 274, 365])
        ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Jan"], fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(lw=0.25, alpha=0.5)
        if r_ == ORDER[0]:
            ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel("GW (avg power)", fontsize=8)

    t = dates.astype("datetime64[s]").astype(datetime)
    for ax, (win, lab) in zip(axes.flat[6:], [(90, "90-day"), (365, "365-day")]):
        k = np.ones(win) / win
        rr = nat_ratio.copy()
        rr[~np.isfinite(rr)] = 1.0
        rm = np.convolve(100 * rr, k, mode="valid")
        ax.plot(t[win - 1:], rm, color="#1f4e8c", lw=1.2)
        ax.axhline(100, color="0.45", lw=0.9, ls="--")
        ax.set_title(f"National inflow, % of norm — {lab} mean",
                     fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("% of normal", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(lw=0.25, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("XM hydro inflows vs seasonal norms — per-river climatologies "
                 f"aggregated to TODAY'S fleet ({min(data)[:4]}–{max(data)[:4]} record, "
                 f"±{WIN}-day windows)", fontsize=12.5, fontweight="bold", y=0.995)
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
        "method": "per-river doy climatologies over each river's own record "
                  "(successor names merged), aggregated to the current fleet; "
                  "envelope from fleet-normalized ratio percentiles",
        "pcts": PCTS,
        "clim": {r: {"mean": np.round(np.nan_to_num(clim[r]["mean"]), 2).tolist(),
                     "pct": np.round(np.nan_to_num(clim[r]["pct"]), 2).tolist()} for r in ORDER},
        "full_pct_of_norm": {"dates": [str(x) for x in dates],
                             **{r: np.round(np.nan_to_num(100 * ratio[r]), 1).tolist()
                                for r in ORDER}},
        "recent": {"dates": [str(x) for x in dates[keep]],
                   **{r: np.round(np.nan_to_num(reg[r][keep]), 2).tolist() for r in ORDER},
                   "pct_of_norm": {r: np.round(np.nan_to_num(100 * ratio[r][keep]), 1).tolist()
                                   for r in ORDER}},
    }, separators=(",", ":")))
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
