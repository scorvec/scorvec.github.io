#!/usr/bin/env python3
"""Argentina: temperature -> electricity demand, from CAMMESA + our NWP temps.

CAMMESA's open API serves 5-minute national demand with temperature
(retention: current calendar year). Daily aggregates are cached and a
V-curve is fitted: demand vs daily mean temp with heating and cooling
branches (base temps scanned), plus a weekend offset. The parked
degree-day tracker archives population-weighted AIFS/HRES daily tmean
forecasts for Argentina every cycle — pushed through the fitted curve
they become a 14-day demand forecast.

Outputs:
  ~/argentina_energy/raw/cammesa_daily.json.gz
  ~/argentina_energy/out/demand_model.json
  argentina_energy/demand.webp (site/, private repo)

    python scripts/sst/argentina_demand.py [--backfill]
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO = Path(__file__).resolve().parent.parent.parent
PRIV = Path.home() / "argentina_energy"
CACHE = PRIV / "raw" / "cammesa_daily.json.gz"
OUT_JSON = PRIV / "out" / "demand_model.json"
SITE = PRIV / "site"
OUT_PNG = SITE / "demand.webp"
DD_ARCHIVE = REPO / "scripts" / "energy" / "data" / "dd_archive.json"
API = ("https://api.cammesa.com/demanda-svc/demanda/"
       "ObtieneDemandaYTemperaturaRegionByFecha?id_region=1002&fecha={d}")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
Y0 = "2026-01-01"
NAVY = "#13273d"
INK = "#1a2733"


def fetch_day(d: str):
    req = urllib.request.Request(API.format(d=d), headers=UA)
    with urllib.request.urlopen(req, timeout=45) as r:
        rows = json.loads(r.read().decode())
    dem = [x["dem"] for x in rows if x.get("dem") is not None]
    tmp = [x["temp"] for x in rows if x.get("temp") is not None]
    if len(dem) < 200:
        return None
    return {"dem_mean": round(float(np.mean(dem)), 0),
            "dem_max": round(float(np.max(dem)), 0),
            "temp_mean": round(float(np.mean(tmp)), 2) if tmp else None,
            "temp_min": round(float(np.min(tmp)), 1) if tmp else None,
            "temp_max": round(float(np.max(tmp)), 1) if tmp else None}


def load_cache(backfill: bool) -> dict:
    data = {}
    if CACHE.exists():
        with gzip.open(CACHE, "rt") as f:
            data = json.load(f)
    end = datetime.now() - timedelta(days=1)
    start = datetime.strptime(Y0, "%Y-%m-%d") if backfill or not data else \
        datetime.strptime(max(data), "%Y-%m-%d") - timedelta(days=2)
    d = start
    n = 0
    while d <= end:
        key = f"{d:%Y-%m-%d}"
        if key not in data:
            try:
                v = fetch_day(key)
                if v:
                    data[key] = v
                    n += 1
            except Exception as e:                  # noqa: BLE001
                print(f"  {key}: {repr(e)[:60]}", flush=True)
            time.sleep(0.15)
        d += timedelta(days=1)
    if n:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(CACHE, "wt") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"cache: +{n} days -> {len(data)}", flush=True)
    return data


def main() -> int:
    backfill = "--backfill" in sys.argv[1:]
    data = load_cache(backfill)
    days = sorted(k for k, v in data.items() if v.get("temp_mean") is not None)
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in days]
    dem = np.array([data[d]["dem_mean"] for d in days]) / 1000  # GW
    T = np.array([data[d]["temp_mean"] for d in days], float)
    wknd = np.array([dt.weekday() >= 5 for dt in dates], float)

    # V-curve fit with base-temp scan
    best = None
    for th in np.arange(12, 19.5, 0.5):
        for tc in np.arange(18, 25.5, 0.5):
            if tc <= th:
                continue
            H = np.maximum(th - T, 0)
            C = np.maximum(T - tc, 0)
            X = np.column_stack([np.ones_like(T), wknd, H, C])
            beta, *_ = np.linalg.lstsq(X, dem, rcond=None)
            r = np.corrcoef(X @ beta, dem)[0, 1]
            if best is None or r > best[0]:
                best = (r, th, tc, beta)
    r, th, tc, beta = best
    H = np.maximum(th - T, 0)
    C = np.maximum(T - tc, 0)
    fit = beta[0] + beta[1] * wknd + beta[2] * H + beta[3] * C
    stats = {"generated": datetime.now(timezone.utc)
             .strftime("%Y-%m-%d %H:%M UTC"),
             "days": len(days), "window": f"{days[0]}..{days[-1]}",
             "r": round(float(r), 3),
             "base_heat_C": float(th), "base_cool_C": float(tc),
             "heating_MW_per_degC": round(float(-beta[2] * 1000), 0),
             "cooling_MW_per_degC": round(float(beta[3] * 1000), 0),
             "weekend_MW": round(float(beta[1] * 1000), 0),
             "base_GW": round(float(beta[0]), 2)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)

    # forecast: latest DD-archive cycle tmean -> demand
    fc = {}
    if DD_ARCHIVE.exists():
        arch = json.load(open(DD_ARCHIVE))
        latest = sorted(arch)[-1]
        for mdl in ("aifs", "hres"):
            ar = arch[latest].get(mdl, {}).get("Argentina")
            if not ar:
                continue
            fd = [datetime.strptime(x, "%Y-%m-%d") for x in ar["days"]]
            tm = np.array(ar["tmean"], float)
            w = np.array([d.weekday() >= 5 for d in fd], float)
            dfc = (beta[0] + beta[1] * w + beta[2] * np.maximum(th - tm, 0)
                   + beta[3] * np.maximum(tm - tc, 0))
            fc[mdl] = (fd, dfc, tm)
        fc["cycle"] = latest

    # ── figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 9.2))
    hd = fig.add_axes([0, 0.93, 1, 0.07])
    hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes,
                               facecolor=NAVY))
    hd.text(0.03, 0.5, "ARGENTINA — TEMPERATURE → ELECTRICITY DEMAND",
            transform=hd.transAxes, color="white", fontsize=14,
            fontweight="bold", va="center")
    hd.text(0.97, 0.5, f"CAMMESA 5-min national demand · {days[0]} – {days[-1]}",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9,
            va="center", ha="right")

    ax = fig.add_axes([0.06, 0.56, 0.42, 0.32])
    sc = ax.scatter(T[wknd == 0], dem[wknd == 0], s=14, c="#1f4e8c",
                    alpha=0.65, label="weekday")
    ax.scatter(T[wknd == 1], dem[wknd == 1], s=14, c="#9db8d8",
               alpha=0.75, label="weekend")
    ts = np.linspace(T.min() - 1, T.max() + 1, 200)
    vf = (beta[0] + beta[2] * np.maximum(th - ts, 0)
          + beta[3] * np.maximum(ts - tc, 0))
    ax.plot(ts, vf, color="#c62828", lw=2.2,
            label=f"V-fit (r={r:.2f})")
    ax.axvline(th, color="0.6", lw=0.7, ls=":")
    ax.axvline(tc, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("daily mean temperature, °C (CAMMESA/GBA)", fontsize=9)
    ax.set_ylabel("daily mean demand, GW", fontsize=9)
    ax.set_title(f"The V: heating {stats['heating_MW_per_degC']:.0f} MW/°C "
                 f"below {th:.0f}° · cooling {stats['cooling_MW_per_degC']:.0f} "
                 f"MW/°C above {tc:.0f}°", fontsize=10.5, fontweight="bold",
                 loc="left", color=INK)
    ax.grid(lw=0.25, alpha=0.5)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    ax2 = fig.add_axes([0.56, 0.56, 0.40, 0.32])
    ax2.plot(dates, dem, color="#1f4e8c", lw=1.0, label="observed")
    ax2.plot(dates, fit, color="#c62828", lw=1.0, alpha=0.85,
             label="temp + weekday model")
    ax2.set_title("2026 to date — observed vs temperature-driven model",
                  fontsize=10.5, fontweight="bold", loc="left", color=INK)
    ax2.set_ylabel("GW", fontsize=9)
    ax2.grid(lw=0.25, alpha=0.5)
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax3 = fig.add_axes([0.06, 0.07, 0.90, 0.40])
    rec = [i for i, dt in enumerate(dates) if dt >= dates[-1] - timedelta(days=30)]
    ax3.plot([dates[i] for i in rec], dem[rec], color="#1f4e8c", lw=1.6,
             marker="o", ms=3, label="observed")
    ax3.plot([dates[i] for i in rec], fit[rec], color="#c62828", lw=1.1,
             alpha=0.8, label="temp model (obs T)")
    if "aifs" in fc:
        fd, dfc, tm = fc["aifs"]
        ax3.plot(fd, dfc, color="#b35806", lw=1.9, ls="--", marker="o",
                 ms=3.5, label=f"forecast (AIFS tmean, {fc['cycle']})")
    if "hres" in fc:
        fd, dfc, tm = fc["hres"]
        ax3.plot(fd, dfc, color="#7b1fa2", lw=1.3, ls=":", marker="s",
                 ms=3, label="forecast (HRES tmean)")
    ax3.axvline(dates[-1], color="0.55", lw=0.7, ls=":")
    ax3.set_title("Last 30 days + 14-day demand forecast — our NWP "
                  "pop-weighted temps through the fitted V-curve",
                  fontsize=10.5, fontweight="bold", loc="left", color=INK)
    ax3.set_ylabel("GW", fontsize=9)
    ax3.grid(lw=0.25, alpha=0.5)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.tick_params(labelsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.text(0.03, 0.012, "cammesa.com open API · forecast temps: "
             "population-weighted AIFS-ENS / HRES (degree-day tracker "
             "archive)", fontsize=7.5, color="#5a6b7a")
    SITE.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
