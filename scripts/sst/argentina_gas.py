#!/usr/bin/env python3
"""Argentina residential/commercial GAS demand vs population-weighted
temperature — the heating-fuel model that complements the power V-curve.

Gas: SSPM/datos.gob.ar monthly production + consumption by sector
(residencial, comercial, industria, centrales electricas, ...) and by
distributor, 1996 -> present, MMm3/month.
Temperature: WB2 ERA5 t2m at the DD-tracker's 8 Argentine metros with
the same population weights (2000-2023), monthly means; extended to
present with CAMMESA GBA daily temps scaled to the pop-weighted series
via their overlap regression.

Model: monthly res+com gas per day = a + b*HDD_month + trend, HDD base
scanned; also fitted per capita-of-connection isn't available, so a
linear trend absorbs connection growth. Outputs a per-degree gas
sensitivity in MMm3/day per degC and a monthly forecast from our NWP
tmean archive (AIFS/HRES pop-weighted) for the current + next month.

Outputs:
  ~/argentina_energy/raw/gas_monthly.json
  ~/argentina_energy/raw/popw_t2m_monthly.json  (cache)
  ~/argentina_energy/out/gas_model.json
  ~/argentina_energy/site/gas.webp

    python scripts/sst/argentina_gas.py
"""
from __future__ import annotations

import csv
import gzip
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
PRIV = Path.home() / "argentina_energy"
GAS_CSV = PRIV / "raw" / "gas_364_3.csv"
GAS_URL = ("https://infra.datos.gob.ar/catalog/sspm/dataset/364/"
           "distribution/364.3/download/actividad-gas.csv")
T2M_CACHE = PRIV / "raw" / "popw_t2m_monthly.json"
CAMMESA = PRIV / "raw" / "cammesa_daily.json.gz"
DD_ARCHIVE = REPO / "scripts" / "energy" / "data" / "dd_archive.json"
OUT_JSON = PRIV / "out" / "gas_model.json"
OUT_PNG = PRIV / "site" / "gas.webp"
WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
METROS = [("Buenos Aires", -34.6, -58.4, 15.4), ("Cordoba", -31.4, -64.2, 1.6),
          ("Rosario", -32.9, -60.7, 1.4), ("Mendoza", -32.9, -68.8, 1.2),
          ("Tucuman", -26.8, -65.2, 0.9), ("La Plata", -34.9, -57.9, 0.9),
          ("Mar del Plata", -38.0, -57.5, 0.65), ("Salta", -24.8, -65.4, 0.65)]
NAVY = "#13273d"
INK = "#1a2733"
DIM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def gas_monthly() -> dict:
    import time
    if not GAS_CSV.exists() or time.time() - GAS_CSV.stat().st_mtime > 7 * 86400:
        try:
            req = urllib.request.Request(GAS_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                GAS_CSV.write_bytes(r.read())
        except Exception as e:                      # noqa: BLE001
            print(f"gas csv refresh failed: {repr(e)[:60]}")
    rows = list(csv.DictReader(GAS_CSV.open()))
    out = {}
    for r in rows:
        try:
            out[r["indice_tiempo"][:7]] = {
                k: float(r[k]) for k in ("residencial", "comercial", "industria",
                                          "centrales_electricas", "total")
                if r.get(k) not in (None, "")}
        except ValueError:
            continue
    return out


ERA5_GLOBAL = Path.home() / "era5_store" / "wb2_1p5_daily_global" / "t2m"


def popw_t2m_monthly() -> dict:
    """8-metro population-weighted monthly mean t2m from the LOCAL global
    daily ERA5 layer (WB2 1959-2023 + ARCO tail, streamed once by
    scripts/era5/wb2_daily_store.py --global). Rebuilt when the layer's
    newest file changes; no remote reads."""
    import xarray as xr
    files = sorted(ERA5_GLOBAL.glob("t2m_*.nc"))
    if not files:
        raise SystemExit("global ERA5 t2m layer missing — run "
                         "scripts/era5/wb2_daily_store.py --global --vars t2m")
    sig = f"{files[-1].name}:{int(files[-1].stat().st_mtime)}"
    if T2M_CACHE.exists():
        c = json.loads(T2M_CACHE.read_text())
        if c.get("_sig") == sig:
            return {k: v for k, v in c.items() if not k.startswith("_")}
    wsum = sum(m[3] for m in METROS)
    out = {}
    for f in files:
        ds = xr.open_dataset(f)
        acc = None
        for name, la, lo, w in METROS:
            v = ds["t2m"].sel(latitude=la, longitude=lo % 360, method="nearest")
            acc = (w / wsum) * v if acc is None else acc + (w / wsum) * v
        mon = (acc.resample(time="1MS").mean() - 273.15)
        for x, val in zip(mon.time.values, mon.values):
            if np.isfinite(val):
                out[f"{str(x)[:7]}"] = round(float(val), 2)
    T2M_CACHE.write_text(json.dumps({**out, "_sig": sig}))
    print(f"pop-weighted t2m: {min(out)}..{max(out)} ({len(out)} months)", flush=True)
    return out


def extend_with_cammesa(tm: dict) -> dict:
    """Extend pop-weighted monthly T past 2023 with CAMMESA GBA daily temps,
    scaled through the overlap regression (GBA is 70% of the weight)."""
    if not CAMMESA.exists():
        return tm
    with gzip.open(CAMMESA, "rt") as f:
        cd = json.load(f)
    gba = {}
    for d, v in cd.items():
        if v.get("temp_mean") is not None:
            gba.setdefault(d[:7], []).append(v["temp_mean"])
    gba_m = {k: float(np.mean(v)) for k, v in gba.items() if len(v) >= 20}
    common = sorted(set(gba_m) & set(tm))
    if len(common) >= 6:
        x = np.array([gba_m[k] for k in common])
        y = np.array([tm[k] for k in common])
        b = np.polyfit(x, y, 1)
    else:
        b = np.array([1.0, -1.0])            # GBA is ~1 degC warmer than pop-wtd
    out = dict(tm)
    for k, v in gba_m.items():
        if k not in out:
            out[k] = round(float(b[0] * v + b[1]), 2)
    return out


def main() -> int:
    gas = gas_monthly()
    tm = extend_with_cammesa(popw_t2m_monthly())
    months = sorted(k for k in gas if k in tm and k >= "2004-01")
    T = np.array([tm[k] for k in months])
    yr = np.array([int(k[:4]) for k in months])
    mo = np.array([int(k[5:]) for k in months])
    dim = np.array([DIM[m - 1] + (1 if m == 2 and y % 4 == 0 else 0)
                    for y, m in zip(yr, mo)])
    rc = np.array([gas[k]["residencial"] + gas[k]["comercial"]
                   for k in months]) / dim                # MMm3/day
    res = np.array([gas[k]["residencial"] for k in months]) / dim
    trend = (yr - 2004) + (mo - 1) / 12.0

    best = None
    for tb in np.arange(14, 22.5, 0.5):
        H = np.maximum(tb - T, 0)
        X = np.column_stack([np.ones_like(T), H, trend])
        beta, *_ = np.linalg.lstsq(X, rc, rcond=None)
        r = np.corrcoef(X @ beta, rc)[0, 1]
        if best is None or r > best[0]:
            best = (r, tb, beta)
    r, tb, beta = best
    H = np.maximum(tb - T, 0)
    fit = beta[0] + beta[1] * H + beta[2] * trend
    # summer floor (non-heating) and winter peak sensitivity
    summer = rc[np.isin(mo, [12, 1, 2])].mean()
    stats = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
             "months": len(months), "window": f"{months[0]}..{months[-1]}",
             "r": round(float(r), 3), "hdd_base_C": float(tb),
             "MMm3_per_day_per_degC_below_base": round(float(beta[1]), 2),
             "pct_of_summer_floor_per_degC": round(float(100 * beta[1] / summer), 1),
             "trend_MMm3_per_day_per_year": round(float(beta[2]), 3),
             "summer_floor_MMm3_per_day": round(float(summer), 1),
             "latest": {"month": months[-1], "rescom_MMm3_per_day": round(float(rc[-1]), 1),
                        "temp_C": float(T[-1])}}
    # July peak vs power: gas res+com in energy terms (1 MMm3 ~ 10.7 GWh thermal)
    jul = rc[mo == 7].mean()
    stats["july_mean_rescom_GWh_thermal_per_day"] = round(float(jul * 10.7), 0)
    stats["note"] = ("monthly res+com gas per day vs pop-weighted (8-metro) "
                     "monthly mean t2m (local ERA5 daily store, WB2+ARCO, 2000-present); "
                     "linear trend absorbs connection growth")

    # forecast: current + next month from NWP tmean archive
    fc = None
    if DD_ARCHIVE.exists():
        arch = json.load(open(DD_ARCHIVE))
        latest = sorted(arch)[-1]
        ar = arch[latest].get("aifs", {}).get("Argentina")
        if ar:
            t14 = float(np.mean(ar["tmean"]))
            H14 = max(tb - t14, 0)
            now_m = datetime.now(timezone.utc)
            tr_now = (now_m.year - 2004) + (now_m.month - 1) / 12.0
            fc = {"cycle": latest, "tmean_14d": round(t14, 2),
                  "rescom_MMm3_per_day": round(float(beta[0] + beta[1] * H14
                                                     + beta[2] * tr_now), 1)}
    stats["forecast_next14d"] = fc
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    # ── figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 9.0))
    hd = fig.add_axes([0, 0.93, 1, 0.07])
    hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.03, 0.5, "ARGENTINA — TEMPERATURE → RESIDENTIAL/COMMERCIAL GAS",
            transform=hd.transAxes, color="white", fontsize=14, fontweight="bold",
            va="center")
    hd.text(0.97, 0.5, f"SSPM monthly gas by sector · {months[0]} – {months[-1]}",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center", ha="right")
    ax = fig.add_axes([0.06, 0.55, 0.42, 0.33])
    sc = ax.scatter(T, rc, c=yr, cmap="viridis", s=18, alpha=0.8)
    ts = np.linspace(T.min() - 1, T.max() + 1, 100)
    ax.plot(ts, beta[0] + beta[1] * np.maximum(tb - ts, 0) + beta[2] * trend[-1],
            color="#c62828", lw=2.2, label=f"fit at {yr[-1]} trend (r={r:.2f})")
    ax.axvline(tb, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("pop-weighted monthly mean temp, °C", fontsize=9)
    ax.set_ylabel("res+com gas, MMm³/day", fontsize=9)
    ax.set_title(f"Heating hockey-stick: {beta[1]:.2f} MMm³/day per °C below "
                 f"{tb:.0f}°C  (≈{stats['pct_of_summer_floor_per_degC']:.0f}% of "
                 "the summer floor)", fontsize=10, fontweight="bold", loc="left",
                 color=INK)
    ax.grid(lw=0.25, alpha=0.5)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("year", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax2 = fig.add_axes([0.56, 0.55, 0.40, 0.33])
    md = [datetime.strptime(k, "%Y-%m") for k in months]
    ax2.plot(md, rc, color="#1f4e8c", lw=1.2, label="observed res+com")
    ax2.plot(md, fit, color="#c62828", lw=1.0, alpha=0.85, label="temp + trend model")
    ax2.plot(md, res, color="#9db8d8", lw=0.9, label="residential only")
    ax2.set_ylabel("MMm³/day", fontsize=9)
    ax2.set_title("Two decades: the model tracks the winter peaks", fontsize=10,
                  fontweight="bold", loc="left", color=INK)
    ax2.grid(lw=0.25, alpha=0.5)
    ax2.legend(fontsize=7.5)
    ax2.tick_params(labelsize=8)

    ax3 = fig.add_axes([0.06, 0.07, 0.90, 0.38])
    rec = [i for i, k in enumerate(months) if k >= "2022-01"]
    ax3.bar([md[i] for i in rec], rc[rec], width=25, color="#a8c6e2", label="observed")
    ax3.plot([md[i] for i in rec], fit[rec], color="#c62828", lw=1.8, marker="o",
             ms=4, label="temp + trend model")
    if fc:
        ax3.scatter([datetime.now()], [fc["rescom_MMm3_per_day"]], color="#b35806",
                    s=90, zorder=5, marker="D",
                    label=f"next 14 d @ AIFS tmean {fc['tmean_14d']:.1f}°C "
                          f"({fc['cycle']})")
    ax3.set_ylabel("MMm³/day", fontsize=9)
    ax3.set_title("Recent months + the NWP-driven estimate for the coming two "
                  "weeks", fontsize=10, fontweight="bold", loc="left", color=INK)
    ax3.grid(lw=0.25, alpha=0.5)
    ax3.legend(fontsize=8)
    ax3.tick_params(labelsize=8)
    fig.text(0.03, 0.012, "gas: datos.gob.ar/SSPM 'produccion y consumo de gas "
             "natural' · temp: ERA5 8-metro pop-weighted (2000-23) extended with "
             "CAMMESA GBA · 1 MMm³ ≈ 10.7 GWh thermal", fontsize=7.5, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
