#!/usr/bin/env python3
"""Argentina regional + distributor demand: V-curves per region with LOCAL
temperature, and the GBA distributor block (Edenor+Edesur+Edelap) as the
residential/commercial proxy.

Why the distributors: large industrials buy on the wholesale market as
GUMA/GUME agents, bypassing distributors — so distributor demand is
overwhelmingly homes + shops. Its temperature slope vs the SADI total's
is the cleanest available split of "how much of the sensitivity is
residential/commercial".

Regions (CAMMESA id): SADI 1002, GBA 426, ProvBA 425, Litoral 417,
Centro 422, Cuyo 429, Comahue 420, NOA 419, NEA 418, Patagonia 111;
distributors Edenor 1077, Edesur 1078, Edelap 1943 (temp = GBA's).

Outputs:
  ~/argentina_energy/raw/cammesa_regions_daily.json.gz
  ~/argentina_energy/out/regional_models.json
  ~/argentina_energy/site/regions.webp

    python scripts/sst/argentina_regions.py [--backfill]
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

PRIV = Path.home() / "argentina_energy"
CACHE = PRIV / "raw" / "cammesa_regions_daily.json.gz"
OUT_JSON = PRIV / "out" / "regional_models.json"
OUT_PNG = PRIV / "site" / "regions.webp"
API = ("https://api.cammesa.com/demanda-svc/demanda/"
       "ObtieneDemandaYTemperaturaRegionByFecha?id_region={r}&fecha={d}")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
Y0 = "2026-01-01"
REGIONS = {"SADI": 1002, "GBA": 426, "PROV_BA": 425, "LITORAL": 417,
           "CENTRO": 422, "CUYO": 429, "COMAHUE": 420, "NOA": 419,
           "NEA": 418, "PATAGONIA": 111,
           "EDENOR": 1077, "EDESUR": 1078, "EDELAP": 1943}
DISTRIB = ["EDENOR", "EDESUR", "EDELAP"]
PANELS = ["SADI", "DISTRIB", "GBA", "PROV_BA", "LITORAL", "CENTRO", "CUYO",
          "COMAHUE", "NOA", "NEA", "PATAGONIA"]
NAVY = "#13273d"
INK = "#1a2733"


def fetch_day(rid: int, d: str):
    req = urllib.request.Request(API.format(r=rid, d=d), headers=UA)
    with urllib.request.urlopen(req, timeout=45) as r:
        rows = json.loads(r.read().decode())
    dem = [x["dem"] for x in rows if x.get("dem") is not None]
    tmp = [x["temp"] for x in rows if x.get("temp") is not None]
    if len(dem) < 200:
        return None
    return {"dem": round(float(np.mean(dem)), 0),
            "temp": round(float(np.mean(tmp)), 2) if tmp else None}


def load_cache(backfill: bool) -> dict:
    """Parallel per-region fetch (thread pool), cache saved after every
    region so a killed run loses at most one region's progress."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    data = {}
    if CACHE.exists():
        with gzip.open(CACHE, "rt") as f:
            data = json.load(f)
    end = datetime.now() - timedelta(days=1)

    def days_needed(name):
        reg = data.get(name, {})
        start = (datetime.strptime(Y0, "%Y-%m-%d") if backfill or not reg
                 else datetime.strptime(max(reg), "%Y-%m-%d") - timedelta(days=2))
        out, d = [], start
        while d <= end:
            k = f"{d:%Y-%m-%d}"
            if k not in reg:
                out.append(k)
            d += timedelta(days=1)
        return out

    def fetch_region(name):
        rid = REGIONS[name]
        got = {}
        for k in days_needed(name):
            try:
                v = fetch_day(rid, k)
                if v:
                    got[k] = v
            except Exception as e:                  # noqa: BLE001
                print(f"  {name} {k}: {repr(e)[:40]}", flush=True)
            time.sleep(0.05)
        return name, got

    def save():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(CACHE, "wt") as f:
            json.dump(data, f, separators=(",", ":"))

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch_region, n) for n in REGIONS]
        for fu in as_completed(futs):
            name, got = fu.result()
            data.setdefault(name, {}).update(got)
            save()
            print(f"  {name}: +{len(got)} -> {len(data[name])} days", flush=True)
    return data


def vfit(T, dem, wknd):
    best = None
    for th in np.arange(10, 20.5, 0.5):
        for tc in np.arange(17, 26.5, 0.5):
            if tc <= th:
                continue
            H = np.maximum(th - T, 0)
            C = np.maximum(T - tc, 0)
            X = np.column_stack([np.ones_like(T), wknd, H, C])
            beta, *_ = np.linalg.lstsq(X, dem, rcond=None)
            r = np.corrcoef(X @ beta, dem)[0, 1]
            if best is None or r > best[0]:
                best = (r, th, tc, beta)
    return best


def main() -> int:
    backfill = "--backfill" in sys.argv[1:]
    data = load_cache(backfill)

    # series per panel: distributor block = sum, temp from GBA
    def series(name):
        if name == "DISTRIB":
            days = sorted(set.intersection(*[set(data[d]) for d in DISTRIB])
                          & set(data["GBA"]))
            dem = np.array([sum(data[d][k]["dem"] for d in DISTRIB)
                            for k in days]) / 1000
            T = np.array([data["GBA"][k]["temp"] for k in days], float)
        else:
            days = sorted(k for k, v in data[name].items()
                          if v.get("temp") is not None)
            dem = np.array([data[name][k]["dem"] for k in days]) / 1000
            T = np.array([data[name][k]["temp"] for k in days], float)
        ok = np.isfinite(T)
        days = [d for d, o in zip(days, ok) if o]
        wk = np.array([datetime.strptime(d, "%Y-%m-%d").weekday() >= 5
                       for d in days], float)
        return days, dem[ok], T[ok], wk

    models = {}
    fig, axes = plt.subplots(3, 4, figsize=(16, 11.5))
    for ax, name in zip(axes.flat, PANELS):
        days, dem, T, wk = series(name)
        if len(dem) < 60:
            ax.set_axis_off()
            continue
        r, th, tc, beta = vfit(T, dem, wk)
        mean_gw = float(dem.mean())
        models[name] = {
            "days": len(days), "r": round(float(r), 3),
            "mean_GW": round(mean_gw, 2),
            "base_heat_C": float(th), "base_cool_C": float(tc),
            "heating_MW_per_degC": round(float(-beta[2] * 1000), 0),
            "cooling_MW_per_degC": round(float(beta[3] * 1000), 0),
            "heating_pct_per_degC": round(float(-beta[2] / mean_gw * 100), 2),
            "cooling_pct_per_degC": round(float(beta[3] / mean_gw * 100), 2),
            "weekend_MW": round(float(beta[1] * 1000), 0)}
        ax.scatter(T[wk == 0], dem[wk == 0], s=9, c="#1f4e8c", alpha=0.6)
        ax.scatter(T[wk == 1], dem[wk == 1], s=9, c="#9db8d8", alpha=0.75)
        ts = np.linspace(T.min() - 1, T.max() + 1, 150)
        ax.plot(ts, beta[0] + beta[2] * np.maximum(th - ts, 0)
                + beta[3] * np.maximum(ts - tc, 0), color="#c62828", lw=2)
        lab = ("GBA distributors\n(Edenor+Edesur+Edelap ≈ res/com)"
               if name == "DISTRIB" else name)
        ax.set_title(f"{lab}  ·  {mean_gw:.1f} GW  ·  r={r:.2f}\n"
                     f"heat {models[name]['heating_pct_per_degC']:.1f}%/°C  "
                     f"cool {models[name]['cooling_pct_per_degC']:.1f}%/°C  "
                     f"[{th:.0f}°,{tc:.0f}°]",
                     fontsize=8.6, fontweight="bold", loc="left", color=INK)
        ax.grid(lw=0.25, alpha=0.5)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("daily mean temp °C (local)", fontsize=7.5)
        ax.set_ylabel("GW", fontsize=7.5)
    for ax in axes.flat[len(PANELS):]:
        ax.set_axis_off()
    fig.suptitle("Argentina — temperature sensitivity by region (local temps) "
                 "and the residential/commercial proxy · CAMMESA 2026",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=115)
    plt.close(fig)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "note": "V-curve per region with its own CAMMESA temp; DISTRIB = "
                "Edenor+Edesur+Edelap on GBA temp (residential/commercial proxy)",
        "models": models}, indent=1))
    print(f"{'region':10} {'GW':>5} {'r':>5} {'heat%/°C':>9} {'cool%/°C':>9} "
          f"{'bases':>9}")
    for k, m in models.items():
        print(f"{k:10} {m['mean_GW']:5.1f} {m['r']:5.2f} "
              f"{m['heating_pct_per_degC']:9.2f} {m['cooling_pct_per_degC']:9.2f}"
              f"  {m['base_heat_C']:.0f}/{m['base_cool_C']:.0f}")
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
