#!/usr/bin/env python3
"""XM reservoir storage: useful volume vs capacity, seasonal norms, and the
implied-outflow climatology that closes the water balance for the forecast.

Metrics (servapibi.xm.com.co, POST /daily, chunks <=30 d, Entity=Embalse):
  VoluUtilDiarEner — useful stored volume, kWh (energy-equivalent)
  CapaUtilDiarEner — useful capacity, kWh
Reservoir -> region from POST /lists MetricId=ListadoEmbalses.

Per region and nationally:
  pct full  = sum(volume) / sum(capacity) over SAME-DAY reporting reservoirs
              (capacity grows with the fleet, so the ratio is seam-free)
  delta S   = sum over reservoirs of day-to-day volume change, computed PER
              RESERVOIR where both days report (fleet-seam-proof)
  implied outflow = inflow(GWh, from the AporEner cache) - delta S
              — generation + spill + evaporation, no generation feed needed.

Outputs:
  ~/colombia_hydro/raw/storage_daily.json.gz    (cache, incremental)
  colombia_hydro/storage_norms.webp             (norms chart, + fan if fresh)
  colombia_hydro/data/storage.json              (state + norms for the engine)

    python scripts/sst/xm_storage.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(HERE))
REPO = HERE.parent.parent
API = "https://servapibi.xm.com.co/daily"
CACHE = Path.home() / "colombia_hydro" / "raw" / "storage_daily.json.gz"
APOR_CACHE = Path.home() / "colombia_hydro" / "raw" / "aporener_daily.json.gz"
RIVERS_JSON = Path.home() / "colombia_hydro" / "raw" / "xm_listado_rios.json"
FAN_JSON = REPO / "colombia_hydro" / "data" / "inflow_forecast.json"
NINO_JSON = REPO / "assets" / "sst" / "data" / "nino_history.json"
OUT_PNG = REPO / "colombia_hydro" / "storage_norms.webp"
OUT_JSON = REPO / "colombia_hydro" / "data" / "storage.json"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
START = "2000-01-01"
WIN = 10                       # +/- days for doy windows
PCTS = [10, 25, 50, 75, 90]

from xm_inflow_history import river_region, ALIASES  # noqa: E402,F401


def reservoir_region() -> dict[str, str]:
    r = requests.post("https://servapibi.xm.com.co/lists",
                      json={"MetricId": "ListadoEmbalses", "Entity": "Sistema"},
                      timeout=90)
    out = {}
    for it in r.json()["Items"]:
        for e in it["ListEntities"]:
            v = e["Values"]
            if v.get("HydroRegion") in ORDER:
                out[v["Name"].strip().upper()] = v["HydroRegion"]
    return out


def fetch_metric(mid: str, d0: datetime, d1: datetime) -> dict:
    out: dict[str, dict[str, float]] = {}
    cur = d0
    while cur <= d1:
        end = min(cur + timedelta(days=29), d1)
        for attempt in range(3):
            try:
                r = requests.post(API, json={"MetricId": mid,
                                             "StartDate": f"{cur:%Y-%m-%d}",
                                             "EndDate": f"{end:%Y-%m-%d}",
                                             "Entity": "Embalse"}, timeout=90)
                r.raise_for_status()
                break
            except Exception as e:                    # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"  retry {mid} {cur:%Y-%m} ({repr(e)[:40]})", flush=True)
        for it in r.json().get("Items", []):
            for e in it.get("DailyEntities", []):
                out.setdefault(it["Date"], {})[e["Name"].strip().upper()] = float(e["Value"])
        cur = end + timedelta(days=1)
    return out


def load_storage() -> dict:
    """{'vol': {day: {res: kWh}}, 'cap': {...}} — incremental cache."""
    data = {"vol": {}, "cap": {}}
    if CACHE.exists():
        with gzip.open(CACHE, "rt") as f:
            data = json.load(f)
    have = sorted(data["vol"])
    d0 = (datetime.strptime(have[-1], "%Y-%m-%d") - timedelta(days=3)
          if have else datetime.strptime(START, "%Y-%m-%d"))
    d1 = datetime.now()
    if (d1 - d0).days >= 1:
        print(f"fetching storage {d0:%Y-%m-%d}..{d1:%Y-%m-%d}", flush=True)
        for key, mid in [("vol", "VoluUtilDiarEner"), ("cap", "CapaUtilDiarEner")]:
            new = fetch_metric(mid, d0, d1)
            for day, v in new.items():
                data[key][day] = v
            print(f"  {mid}: cache {len(data[key])} days", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(CACHE, "wt") as f:
            json.dump(data, f, separators=(",", ":"))
    return data


def pct_anomaly_series() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(dates, anom[r]) — regional %-full anomaly vs its doy median norm,
    computed from the local cache only (no fetch). For model v3 fitting."""
    res_reg = reservoir_region()
    with gzip.open(CACHE, "rt") as f:
        data = json.load(f)
    days = sorted(set(data["vol"]) & set(data["cap"]))
    dates = np.array(days, dtype="datetime64[D]")
    doy = np.array([min(datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday, 365)
                    for d in days])
    out = {}
    for r in ORDER:
        v = np.full(len(days), np.nan)
        for i, day in enumerate(days):
            sv = sc = 0.0
            for res, x in data["vol"][day].items():
                if res_reg.get(res) == r and data["cap"][day].get(res, 0) > 0:
                    sv += x
                    sc += data["cap"][day][res]
            if sc > 0:
                v[i] = 100 * sv / sc
        an = np.full(len(days), np.nan)
        for d_ in range(1, 366):
            dist = np.minimum(np.abs(doy - d_), 365 - np.abs(doy - d_))
            m = (dist <= WIN) & np.isfinite(v)
            if m.sum() > 20:
                an[doy == d_] = v[doy == d_] - np.median(v[m])
        out[r] = an
    return dates, out


def _recent_outflow(x: np.ndarray, rec14: np.ndarray) -> float:
    v = np.where(np.abs(x[rec14]) < 5e8, x[rec14], np.nan)
    return float(np.round(np.nanmean(v), 0)) if np.isfinite(v).any() else 0.0


def main() -> int:
    res_reg = reservoir_region()
    data = load_storage()
    days = sorted(set(data["vol"]) & set(data["cap"]))
    dates = np.array(days, dtype="datetime64[D]")
    nd = len(days)
    print(f"storage: {days[0]}..{days[-1]} ({nd} days, "
          f"{len(res_reg)} mapped reservoirs)", flush=True)

    regions = ORDER + ["NATIONAL"]
    vol = {r: np.full(nd, np.nan) for r in regions}
    cap = {r: np.full(nd, np.nan) for r in regions}
    dS = {r: np.zeros(nd) for r in regions}
    prev: dict[str, float] = {}
    prev_day_i = -1
    for i, day in enumerate(days):
        vs, cs = data["vol"][day], data["cap"][day]
        sums_v = {r: 0.0 for r in regions}
        sums_c = {r: 0.0 for r in regions}
        for res, v in vs.items():
            rg = res_reg.get(res)
            c = cs.get(res)
            if rg is None or c is None or c <= 0:
                continue
            for k in (rg, "NATIONAL"):
                sums_v[k] += v
                sums_c[k] += c
            # per-reservoir day-to-day change (seam-proof across fleet growth)
            if i - prev_day_i == 1 and res in prev:
                d_ = v - prev[res]
                dS[rg][i] += d_
                dS["NATIONAL"][i] += d_
        for r in regions:
            if sums_c[r] > 0:
                vol[r][i] = sums_v[r]
                cap[r][i] = sums_c[r]
        prev = {res: v for res, v in vs.items() if res in res_reg}
        prev_day_i = i

    pct = {r: 100.0 * vol[r] / cap[r] for r in regions}
    has_storage = [r for r in ORDER if np.isfinite(pct[r]).sum() > 365]

    # ── inflow GWh per region (full AporEner cache) for implied outflow ─────
    with gzip.open(APOR_CACHE, "rt") as f:
        apor = json.load(f)
    r2reg = river_region()
    infl = {r: np.full(nd, np.nan) for r in regions}
    for i, day in enumerate(days):
        dd = apor.get(day)
        if not dd:
            continue
        s = {r: 0.0 for r in regions}
        for riv, v in dd.items():
            rg = r2reg.get(riv)
            if rg:
                s[rg] += v
                s["NATIONAL"] += v
        for r in regions:
            infl[r][i] = s[r]
    # implied outflow (kWh/day): generation + spill + evap
    outfl = {r: infl[r] - dS[r] for r in regions}

    doy = np.array([min(datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday, 365)
                    for d in days])
    # monthly Nino-3.4 anomaly on the daily axis (ENSO channel: reservoir
    # evaporation + ENSO-conditioned dispatch both load on this regressor)
    nh = json.loads(NINO_JSON.read_text())
    nmon = np.array(nh["months"], dtype="datetime64[M]")
    nanom = np.array(nh["series"]["nino34"]["anom"], float)
    mon_of = dates.astype("datetime64[M]")
    nino = np.full(nd, np.nan)
    idx = np.searchsorted(nmon, mon_of)
    ok = (idx < len(nmon)) & (nmon[np.minimum(idx, len(nmon) - 1)] == mon_of)
    nino[ok] = nanom[idx[ok]]
    nino_now = float(nanom[np.isfinite(nanom)][-1])
    # doy norms: % full percentile envelope + median implied outflow
    norms = {}
    for r in has_storage + ["NATIONAL"]:
        p_ = np.full((365, len(PCTS)), np.nan)
        of = np.full(365, np.nan)
        for d_ in range(1, 366):
            dist = np.minimum(np.abs(doy - d_), 365 - np.abs(doy - d_))
            m = (dist <= WIN) & np.isfinite(pct[r])
            if m.sum() > 20:
                p_[d_ - 1] = np.percentile(pct[r][m], PCTS)
            mo = (dist <= WIN) & np.isfinite(outfl[r]) & (np.abs(outfl[r]) < 5e8)
            if mo.sum() > 20:
                of[d_ - 1] = np.median(outfl[r][mo])
        # ENSO coefficient on outflow anomaly (kWh/day per degC Nino-3.4):
        # captures evaporation loss + dispatch response together
        oa = outfl[r] - of[doy - 1]
        mb = np.isfinite(oa) & np.isfinite(nino) & (np.abs(oa) < 3e8)
        beta = (float(np.polyfit(nino[mb], oa[mb], 1)[0]) if mb.sum() > 500 else 0.0)
        norms[r] = {"pct": p_, "outflow_kwh": of, "beta_enso": beta}

    # ── fan (from colombia_forecast.py), if fresh ───────────────────────────
    fan = None
    if FAN_JSON.exists():
        f_ = json.loads(FAN_JSON.read_text())
        if "storage" in f_ and (np.datetime64(datetime.now(timezone.utc)
                                .strftime("%Y-%m-%d"))
                                - np.datetime64(f_["dates"][0])).astype(int) <= 2:
            fan = f_

    # ── figure ──────────────────────────────────────────────────────────────
    panels = has_storage + ["NATIONAL"]
    nrow = (len(panels) + 2) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(13.5, 3.2 * nrow))
    x365 = np.arange(1, 366)
    for ax, r in zip(axes.flat, panels):
        n = norms[r]
        ax.fill_between(x365, n["pct"][:, 0], n["pct"][:, 4], color="#9db8d8",
                        alpha=0.35, label="p10–p90")
        ax.fill_between(x365, n["pct"][:, 1], n["pct"][:, 3], color="#5b87c0",
                        alpha=0.35, label="p25–p75")
        ax.plot(x365, n["pct"][:, 2], color="#1f4e8c", lw=1.4, label="median")
        rec = dates > dates[-1] - np.timedelta64(200, "D")
        ax.plot(doy[rec], pct[r][rec], color="#c62828", lw=1.3, label="last 200 days")
        if fan is not None and r in fan["storage"]["basins"]:
            q = fan["storage"]["basins"][r]
            fdoy = np.array([min(datetime.strptime(x, "%Y-%m-%d").timetuple().tm_yday,
                                 365) for x in fan["dates"]])
            wrap = np.where(np.diff(fdoy) < 0)[0]
            stop = int(wrap[0]) + 1 if len(wrap) else len(fdoy)
            fd = fdoy[:stop]
            ax.fill_between(fd, q["p10"][:stop], q["p90"][:stop],
                            color="#e08214", alpha=0.25, lw=0)
            ax.fill_between(fd, q["p25"][:stop], q["p75"][:stop],
                            color="#e08214", alpha=0.35, lw=0)
            ax.plot(fd, q["p50"][:stop], color="#b35806", lw=1.4, ls="--",
                    label="AIFS+IFS ens forecast")
        ax.set_title(f"{r} — useful storage, % of capacity",
                     fontsize=10, fontweight="bold", loc="left")
        ax.set_xlim(1, 365)
        ax.set_ylim(0, 105)
        ax.set_xticks([1, 91, 182, 274, 365])
        ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Jan"], fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(lw=0.25, alpha=0.5)
        if r == panels[0]:
            ax.legend(fontsize=7, loc="lower left")
        ax.set_ylabel("% full", fontsize=8)
    # long-term national line in the last slot if free
    if len(panels) < nrow * 2:
        ax = axes.flat[-1]
        t = dates.astype("datetime64[s]").astype(datetime)
        ax.plot(t, pct["NATIONAL"], color="#1f4e8c", lw=0.8)
        ax.set_title("NATIONAL — % full, full record", fontsize=10,
                     fontweight="bold", loc="left")
        ax.set_ylim(0, 105)
        ax.tick_params(labelsize=8)
        ax.grid(lw=0.25, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("XM reservoir storage vs seasonal norms — useful volume as % of "
                 f"useful capacity ({days[0][:4]}–{days[-1][:4]}, ±{WIN}-day windows)",
                 fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PNG, dpi=115)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")

    # ── state JSON for the forecast engine ──────────────────────────────────
    rec14 = dates > dates[-1] - np.timedelta64(14, "D")
    rec400 = dates > dates[-1] - np.timedelta64(400, "D")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "last_day": days[-1],
        "units": "kWh (energy-equivalent storage)",
        "nino34_now": round(nino_now, 2),
        "regions": {r: {
            "vol_kwh": float(np.round(vol[r][np.isfinite(vol[r])][-1], 0)),
            "cap_kwh": float(np.round(cap[r][np.isfinite(cap[r])][-1], 0)),
            "pct_full": float(np.round(pct[r][np.isfinite(pct[r])][-1], 2)),
            "pct_anom_latest": float(np.round(
                (pct[r] - norms[r]["pct"][doy - 1, 2])[np.isfinite(pct[r])][-1], 2)),
            "outflow_doy_kwh": np.round(np.nan_to_num(
                norms[r]["outflow_kwh"]), 0).tolist(),
            "outflow_recent_kwh": _recent_outflow(outfl[r], rec14),
            "beta_enso_kwh_per_degc": float(np.round(norms[r]["beta_enso"], 0)),
            # what the engine integrates: half recent reality, half the
            # ENSO-adjusted seasonal norm (evap + dispatch in beta_enso)
            "outflow_fcst_kwh": np.round(
                0.5 * _recent_outflow(outfl[r], rec14)
                + 0.5 * (np.nan_to_num(norms[r]["outflow_kwh"])
                         + norms[r]["beta_enso"] * nino_now), 0).tolist(),
        } for r in has_storage + ["NATIONAL"]},
        "recent": {"dates": [str(x) for x in dates[rec400]],
                   "pct_full": {r: np.round(np.nan_to_num(pct[r][rec400]), 2).tolist()
                                for r in has_storage + ["NATIONAL"]}},
        "pct_doy": {r: np.round(np.nan_to_num(norms[r]["pct"]), 2).tolist()
                    for r in has_storage + ["NATIONAL"]},
    }, separators=(",", ":")))
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    for r in has_storage + ["NATIONAL"]:
        v = json.loads(OUT_JSON.read_text())["regions"][r]
        print(f"  {r}: {v['pct_full']}% full, outflow recent "
              f"{v['outflow_recent_kwh']/1e6:.1f} GWh/d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
