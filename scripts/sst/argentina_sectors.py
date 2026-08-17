#!/usr/bin/env python3
"""Argentina OFFICIAL demand-by-sector from CAMMESA's Informe Mensual base.

Source: WordPress download 'base_informe_mensual_YYYY-MM' ZIP on
cammesaweb (the documents API doesn't carry it). Two workbooks:
  Demanda Mensual.xlsx  -> monthly MWh: Residencial, No Residencial
                           (seasonalized distribution demand), GUDI/GUME/
                           GUMA (large users), MATE, DEMANDA LOCAL — 2023->
  Demanda Horaria por Tipo.xlsx -> hourly Gran Usuario MEM vs Distribuidor
Fitted: residential monthly demand per day vs the 8-metro pop-weighted
ERA5 temp (local store) with a V (scanned bases) + trend.

Outputs:
  ~/argentina_energy/raw/cammesa_sectors_monthly.json
  ~/argentina_energy/out/sector_model.json
  ~/argentina_energy/site/sectors.webp

    python scripts/sst/argentina_sectors.py [--refresh]
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIV = Path.home() / "argentina_energy"
RAWD = PRIV / "raw" / "cammesa_informe"
OUT_JSON = PRIV / "out" / "sector_model.json"
SEC_JSON = PRIV / "raw" / "cammesa_sectors_monthly.json"
OUT_PNG = PRIV / "site" / "sectors.webp"
T2M_CACHE = PRIV / "raw" / "popw_t2m_monthly.json"
PAGE = "https://cammesaweb.cammesa.com/informe-sintesis-mensual/"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
NAVY = "#13273d"
INK = "#1a2733"
DIM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
SECTORS = ["Demanda Residencial", "Demanda No Residencial", "GUDI", "GUME",
           "GUMA", "DEMANDA LOCAL"]


def latest_base_zip(refresh: bool) -> Path:
    RAWD.mkdir(parents=True, exist_ok=True)
    have = sorted(RAWD.glob("base_*.zip"))
    if have and not refresh:
        return have[-1]
    html = urllib.request.urlopen(urllib.request.Request(PAGE, headers=UA),
                                  timeout=60).read().decode("utf-8", "replace")
    m = re.findall(r'https://cammesaweb\.cammesa\.com/download/base_informe_mensual_'
                   r'(\d{4}-\d{2})/\?wpdmdl=\d+[^"]*', html)
    urls = re.findall(r'(https://cammesaweb\.cammesa\.com/download/base_informe_mensual_'
                      r'\d{4}-\d{2}/\?wpdmdl=\d+[^"]*)', html)
    if not urls:
        if have:
            return have[-1]
        raise SystemExit("no base_informe_mensual link found")
    ym = m[0]
    dst = RAWD / f"base_{ym}.zip"
    if not dst.exists():
        data = urllib.request.urlopen(urllib.request.Request(urls[0], headers=UA),
                                      timeout=600).read()
        dst.write_bytes(data)
        print(f"downloaded {dst.name} ({len(data)/1e6:.0f} MB)", flush=True)
    return dst


def read_sectors(zp: Path) -> dict:
    import openpyxl
    zf = zipfile.ZipFile(zp)
    name = [n for n in zf.namelist() if n.endswith("Demanda Mensual.xlsx")][0]
    tmp = RAWD / "_dm.xlsx"
    tmp.write_bytes(zf.read(name))
    ws = openpyxl.load_workbook(tmp, read_only=True, data_only=True)["DEMANDA"]
    out = {}
    months = None
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 4:
            months = [str(c)[:7] for c in row[1:] if c is not None
                      and str(c)[:4].isdigit()]
            continue
        if months is None or row[0] is None:
            continue
        lab = str(row[0]).strip()
        if lab in SECTORS:
            vals = [float(v) if v is not None else np.nan
                    for v in row[1:1 + len(months)]]
            out[lab] = dict(zip(months, vals))
        if lab == "DETALLE DEMANDA":
            break
    tmp.unlink(missing_ok=True)
    return out


def main() -> int:
    refresh = "--refresh" in sys.argv[1:]
    zp = latest_base_zip(refresh)
    sec = read_sectors(zp)
    SEC_JSON.write_text(json.dumps(sec, indent=0))
    months = sorted(sec["Demanda Residencial"])
    tm = {k: v for k, v in json.loads(T2M_CACHE.read_text()).items()
          if not k.startswith("_")}
    months = [m for m in months if m in tm
              and np.isfinite(sec["Demanda Residencial"][m])]
    T = np.array([tm[m] for m in months])
    yr = np.array([int(m[:4]) for m in months])
    mo = np.array([int(m[5:]) for m in months])
    dim = np.array([DIM[m - 1] + (1 if m == 2 and y % 4 == 0 else 0)
                    for y, m in zip(yr, mo)])
    trend = (yr - 2023) + (mo - 1) / 12.0

    def per_day(lab):
        return np.array([sec[lab][m] for m in months]) / dim / 1e3 / 24   # GW avg

    res = per_day("Demanda Residencial")
    nonres = per_day("Demanda No Residencial")
    gu = (per_day("GUDI") + per_day("GUME") + per_day("GUMA"))
    tot = per_day("DEMANDA LOCAL")

    def vfit(y):
        best = None
        for th in np.arange(12, 20.5, 0.5):
            for tc in np.arange(17, 25.5, 0.5):
                if tc <= th:
                    continue
                X = np.column_stack([np.ones_like(T), np.maximum(th - T, 0),
                                     np.maximum(T - tc, 0), trend])
                b, *_ = np.linalg.lstsq(X, y, rcond=None)
                r = np.corrcoef(X @ b, y)[0, 1]
                if best is None or r > best[0]:
                    best = (r, th, tc, b)
        return best

    models = {}
    for lab, y in (("residential", res), ("non_residential", nonres),
                   ("large_users", gu), ("total", tot)):
        r, th, tc, b = vfit(y)
        models[lab] = {"mean_GW": round(float(y.mean()), 2), "r": round(float(r), 3),
                       "base_heat_C": float(th), "base_cool_C": float(tc),
                       "heating_MW_per_degC": round(float(-b[1] * 1000), 0),
                       "cooling_MW_per_degC": round(float(b[2] * 1000), 0),
                       "heating_pct_per_degC": round(float(-b[1] / y.mean() * 100), 2),
                       "cooling_pct_per_degC": round(float(b[2] / y.mean() * 100), 2),
                       "trend_MW_per_year": round(float(b[3] * 1000), 0)}
    stats = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
             "source": zp.name, "months": len(months),
             "window": f"{months[0]}..{months[-1]}",
             "shares_latest": {"residential": round(float(res[-1] / tot[-1]), 3),
                               "non_residential": round(float(nonres[-1] / tot[-1]), 3),
                               "large_users": round(float(gu[-1] / tot[-1]), 3)},
             "models": models}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    # ── figure ──────────────────────────────────────────────────────────────
    md = [datetime.strptime(m, "%Y-%m") for m in months]
    fig = plt.figure(figsize=(13.5, 8.6))
    hd = fig.add_axes([0, 0.93, 1, 0.07])
    hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.03, 0.5, "ARGENTINA — OFFICIAL DEMAND BY SECTOR (CAMMESA INFORME MENSUAL)",
            transform=hd.transAxes, color="white", fontsize=13.5, fontweight="bold",
            va="center")
    hd.text(0.97, 0.5, f"{months[0]} – {months[-1]} · monthly MWh → average GW",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center", ha="right")
    ax = fig.add_axes([0.06, 0.55, 0.55, 0.33])
    ax.stackplot(md, res, nonres, gu, labels=["Residencial", "No residencial "
                 "(commercial/small)", "Grandes usuarios (GUDI+GUME+GUMA)"],
                 colors=["#c62828", "#e08214", "#1f4e8c"], alpha=0.85)
    ax.plot(md, tot, color="k", lw=1.0)
    ax.set_ylabel("average GW", fontsize=9)
    ax.set_title("Who uses the power: residential is the seasonal one",
                 fontsize=10.5, fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(lw=0.25, alpha=0.5)
    ax.tick_params(labelsize=8)
    ax2 = fig.add_axes([0.68, 0.55, 0.29, 0.33])
    for lab, y, col in (("residential", res, "#c62828"),
                        ("large_users", gu, "#1f4e8c")):
        m = models[lab]
        ax2.scatter(T, y, s=16, c=col, alpha=0.7, label=lab.replace("_", " "))
        ts = np.linspace(T.min() - 1, T.max() + 1, 100)
        b = None
        # re-evaluate the fit at latest trend for the curve
        X = np.column_stack([np.ones_like(T), np.maximum(m["base_heat_C"] - T, 0),
                             np.maximum(T - m["base_cool_C"], 0), trend])
        bb, *_ = np.linalg.lstsq(X, y, rcond=None)
        ax2.plot(ts, bb[0] + bb[1] * np.maximum(m["base_heat_C"] - ts, 0)
                 + bb[2] * np.maximum(ts - m["base_cool_C"], 0) + bb[3] * trend[-1],
                 color=col, lw=2)
    ax2.set_xlabel("pop-weighted monthly temp °C", fontsize=8.5)
    ax2.set_ylabel("GW", fontsize=8.5)
    ax2.set_title(f"Residential V: {models['residential']['heating_pct_per_degC']:.1f}%/°C "
                  f"heat, +{models['residential']['cooling_pct_per_degC']:.1f}%/°C cool\n"
                  f"large users: {models['large_users']['heating_pct_per_degC']:.1f} / "
                  f"+{models['large_users']['cooling_pct_per_degC']:.1f}%/°C — flat",
                  fontsize=9, fontweight="bold", loc="left", color=INK)
    ax2.grid(lw=0.25, alpha=0.5)
    ax2.legend(fontsize=7.5)
    ax2.tick_params(labelsize=8)
    ax3 = fig.add_axes([0.06, 0.07, 0.91, 0.38])
    labs = ["residential", "non_residential", "large_users", "total"]
    xs = np.arange(len(labs))
    h = [-models[k]["heating_MW_per_degC"] for k in labs]
    c = [models[k]["cooling_MW_per_degC"] for k in labs]
    ax3.bar(xs - 0.2, h, 0.38, color="#1d6fb8", label="heating MW/°C (below base)")
    ax3.bar(xs + 0.2, c, 0.38, color="#c62828", label="cooling MW/°C (above base)")
    for x, k in zip(xs, labs):
        ax3.text(x, max(h[x], c[x]) + 15,
                 f"{models[k]['mean_GW']:.1f} GW · r={models[k]['r']:.2f}",
                 ha="center", fontsize=8, color="#555")
    ax3.set_xticks(xs)
    ax3.set_xticklabels(["Residencial", "No residencial", "Grandes usuarios",
                         "Total (demanda local)"], fontsize=9)
    ax3.set_ylabel("MW per °C", fontsize=9)
    ax3.set_title("Temperature sensitivity by sector — the official split confirms "
                  "the distributor-proxy result", fontsize=10.5, fontweight="bold",
                  loc="left", color=INK)
    ax3.grid(lw=0.25, alpha=0.5, axis="y")
    ax3.legend(fontsize=8)
    ax3.tick_params(labelsize=8)
    fig.text(0.03, 0.012, f"CAMMESA Informe Mensual base ({zp.name}) · Demanda Mensual.xlsx "
             "· temps: 8-metro pop-weighted ERA5, local daily store", fontsize=7.5,
             color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
