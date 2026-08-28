#!/usr/bin/env python3
"""Argentina gas demand outlook — next 14 days vs normal.

Puts the pieces together:
  res/com gas  = a + b*HDD(base 19.5) + trend        (argentina_gas.py fit,
                 monthly; applied to DAILY pop-weighted temps)
  power-sector gas = thermal generation gas burn implied by weather-driven
                 demand: the national V-curve gives demand(T); above the
                 weather-neutral demand, the marginal MW is served by gas
                 CCGT/OCGT at ~0.20 MMm3 per GWh_e (Argentina thermal fleet
                 mostly gas, ~48% CCGT efficiency; heat rate 7.5 MJ/kWh ->
                 0.21 Mm3/GWh at 36 MJ/m3). Marginal only, not the whole
                 thermal burn — hydro/nuclear/renewables run first.
Normal = the same models at each calendar day's climatological pop-
weighted temperature (25-yr daily ERA5 clim from the local store).
Forecast = AIFS + HRES daily tmean from the DD-tracker archive.
Anomaly = forecast − normal, MMm3/day and %.

Output: ~/argentina_energy/site/gas_outlook.webp + out/gas_outlook.json
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO = Path(__file__).resolve().parent.parent.parent
PRIV = Path.home() / "argentina_energy"
GAS_MODEL = PRIV / "out" / "gas_model.json"
DEM_MODEL = PRIV / "out" / "demand_model.json"
DD_ARCHIVE = REPO / "scripts" / "energy" / "data" / "dd_archive.json"
ERA5_T2M = Path.home() / "era5_store" / "wb2_1p5_daily_global" / "t2m"
CLIM_CACHE = PRIV / "raw" / "popw_t2m_dailyclim.json"
CAMMESA = PRIV / "raw" / "cammesa_daily.json.gz"
OUT_PNG = PRIV / "site" / "gas_outlook.webp"
OUT_JSON = PRIV / "out" / "gas_outlook.json"
METROS = [("Buenos Aires", -34.6, -58.4, 15.4), ("Cordoba", -31.4, -64.2, 1.6),
          ("Rosario", -32.9, -60.7, 1.4), ("Mendoza", -32.9, -68.8, 1.2),
          ("Tucuman", -26.8, -65.2, 0.9), ("La Plata", -34.9, -57.9, 0.9),
          ("Mar del Plata", -38.0, -57.5, 0.65), ("Salta", -24.8, -65.4, 0.65)]
MM3_PER_GWH_E = 0.21               # marginal gas per GWh electric at the margin
NAVY = "#13273d"
INK = "#1a2733"


def daily_clim() -> np.ndarray:
    """365-day pop-weighted t2m climatology (2000-2024, +/-7 d smoothing)."""
    import xarray as xr
    if CLIM_CACHE.exists():
        return np.array(json.loads(CLIM_CACHE.read_text()))
    wsum = sum(m[3] for m in METROS)
    acc, doys = [], []
    for f in sorted(ERA5_T2M.glob("t2m_*.nc")):
        y = int(f.stem[-4:])
        if y < 2000 or y > 2024:
            continue
        ds = xr.open_dataset(f)
        s = None
        for _, la, lo, w in METROS:
            v = ds["t2m"].sel(latitude=la, longitude=lo % 360, method="nearest")
            s = (w / wsum) * v if s is None else s + (w / wsum) * v
        acc.append(s.values - 273.15)
        doys.append(np.minimum(s.time.dt.dayofyear.values, 365))
    T = np.concatenate(acc)
    D = np.concatenate(doys)
    clim = np.array([np.nanmean(T[(np.minimum(np.abs(D - d), 365 - np.abs(D - d)) <= 7)])
                     for d in range(1, 366)])
    CLIM_CACHE.write_text(json.dumps(np.round(clim, 3).tolist()))
    return clim


def main() -> int:
    gm = json.loads(GAS_MODEL.read_text())
    dm = json.loads(DEM_MODEL.read_text())
    clim = daily_clim()
    arch = json.load(open(DD_ARCHIVE))
    cyc = sorted(arch)[-1]
    fc = {}
    for mdl in ("aifs", "hres"):
        ar = arch[cyc].get(mdl, {}).get("Argentina")
        if ar:
            fc[mdl] = ([datetime.strptime(x, "%Y-%m-%d") for x in ar["days"]],
                       np.array(ar["tmean"], float))
    days, T_aifs = fc["aifs"]
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in days])
    Tn = clim[doy - 1]

    # models
    tb, b_gas = gm["hdd_base_C"], gm["MMm3_per_day_per_degC_below_base"]
    now = datetime.now(timezone.utc)
    trend_now = (now.year - 2004) + (now.month - 1) / 12.0
    a_gas = gm["summer_floor_MMm3_per_day"]           # floor at current trend
    # recover intercept+trend at now from stats: floor is at zero HDD
    def rescom(T):
        return a_gas + b_gas * np.maximum(tb - T, 0)
    th, tc = dm["base_heat_C"], dm["base_cool_C"]
    hm, cm = abs(dm["heating_MW_per_degC"]), dm["cooling_MW_per_degC"]
    def power_gas(T):
        # weather-driven electric increment (GW) -> GWh/day -> MMm3/day
        inc_gw = (hm * np.maximum(th - T, 0) + cm * np.maximum(T - tc, 0)) / 1000
        return inc_gw * 24 * MM3_PER_GWH_E

    rows = {}
    for mdl, (dd, T) in fc.items():
        n = len(dd)
        rc_f, rc_n = rescom(T), rescom(Tn[:n])
        pg_f, pg_n = power_gas(T), power_gas(Tn[:n])
        rows[mdl] = {"days": [d.strftime("%Y-%m-%d") for d in dd],
                     "tmean": T.round(2).tolist(), "tclim": Tn[:n].round(2).tolist(),
                     "rescom_fcst": rc_f.round(2).tolist(), "rescom_norm": rc_n.round(2).tolist(),
                     "power_fcst": pg_f.round(2).tolist(), "power_norm": pg_n.round(2).tolist()}
    a = rows["aifs"]
    tot_f = np.array(a["rescom_fcst"]) + np.array(a["power_fcst"])
    tot_n = np.array(a["rescom_norm"]) + np.array(a["power_norm"])
    summary = {"cycle": cyc, "days": len(days),
               "tmean_fcst_avg": round(float(T_aifs.mean()), 2),
               "tmean_clim_avg": round(float(Tn.mean()), 2),
               "rescom_fcst_avg": round(float(np.mean(a["rescom_fcst"])), 1),
               "rescom_norm_avg": round(float(np.mean(a["rescom_norm"])), 1),
               "power_gas_fcst_avg": round(float(np.mean(a["power_fcst"])), 1),
               "power_gas_norm_avg": round(float(np.mean(a["power_norm"])), 1),
               "total_weather_gas_anom_MMm3_per_day": round(float((tot_f - tot_n).mean()), 1),
               "total_weather_gas_anom_pct": round(float(100 * ((tot_f - tot_n).mean()
                                                            / tot_n.mean())), 1),
               "rescom_anom_pct": round(float(100 * (np.mean(a["rescom_fcst"])
                                                   / np.mean(a["rescom_norm"]) - 1)), 1)}
    OUT_JSON.write_text(json.dumps({"generated": now.strftime("%Y-%m-%d %H:%M UTC"),
                                    "summary": summary, "series": rows,
                                    "assumptions": {"mm3_per_gwh_e_marginal": MM3_PER_GWH_E,
                                                    "gas_model": gm["note"],
                                                    "clim": "2000-2024 pop-wtd ERA5 daily, ±7d"}},
                                   indent=1))
    print(json.dumps(summary, indent=1))

    # ── figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 9.0))
    hd = fig.add_axes([0, 0.93, 1, 0.07])
    hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.03, 0.5, "ARGENTINA — GAS DEMAND OUTLOOK, NEXT 14 DAYS VS NORMAL",
            transform=hd.transAxes, color="white", fontsize=14, fontweight="bold",
            va="center")
    hd.text(0.97, 0.5, f"AIFS-ENS + HRES pop-weighted temps · cycle {cyc}",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center", ha="right")
    ax = fig.add_axes([0.06, 0.66, 0.90, 0.23])
    ax.plot(days, Tn, color="0.45", lw=1.5, ls="--", label="climatology (2000-24)")
    ax.plot(days, T_aifs, color="#b35806", lw=2.0, marker="o", ms=4, label="AIFS-ENS tmean")
    if "hres" in fc:
        ax.plot(fc["hres"][0], fc["hres"][1], color="#7b1fa2", lw=1.3, ls=":",
                marker="s", ms=3, label="HRES tmean")
    ax.fill_between(days, Tn, T_aifs, where=T_aifs < Tn, color="#1d6fb8", alpha=0.18)
    ax.fill_between(days, Tn, T_aifs, where=T_aifs >= Tn, color="#c62828", alpha=0.18)
    ax.set_ylabel("°C", fontsize=9)
    ax.set_title(f"Population-weighted temperature: forecast {summary['tmean_fcst_avg']:.1f}°C "
                 f"vs normal {summary['tmean_clim_avg']:.1f}°C (blue = colder than normal)",
                 fontsize=10.5, fontweight="bold", loc="left", color=INK)
    ax.grid(lw=0.25, alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax2 = fig.add_axes([0.06, 0.36, 0.90, 0.23])
    rc_f = np.array(a["rescom_fcst"]); rc_n = np.array(a["rescom_norm"])
    ax2.plot(days, rc_n, color="0.45", lw=1.5, ls="--", label="normal (climatological T)")
    ax2.plot(days, rc_f, color="#c62828", lw=2.0, marker="o", ms=4, label="forecast")
    ax2.fill_between(days, rc_n, rc_f, color="#c62828", alpha=0.15)
    ax2.set_ylabel("MMm³/day", fontsize=9)
    ax2.set_title(f"Residential + commercial gas: {summary['rescom_fcst_avg']:.1f} vs "
                  f"{summary['rescom_norm_avg']:.1f} MMm³/day normal "
                  f"({summary['rescom_anom_pct']:+.0f}%)", fontsize=10.5,
                  fontweight="bold", loc="left", color=INK)
    ax2.grid(lw=0.25, alpha=0.5)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.tick_params(labelsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax3 = fig.add_axes([0.06, 0.06, 0.90, 0.23])
    an_rc = rc_f - rc_n
    an_pg = np.array(a["power_fcst"]) - np.array(a["power_norm"])
    ax3.bar(days, an_rc, width=0.8, color="#c62828", label="res/com gas anomaly")
    ax3.bar(days, an_pg, width=0.8, bottom=an_rc, color="#1f4e8c",
            label="power-sector gas anomaly (marginal thermal)")
    ax3.axhline(0, color="0.4", lw=0.8)
    ax3.set_ylabel("MMm³/day vs normal", fontsize=9)
    ax3.set_title(f"Weather-driven gas anomaly: {summary['total_weather_gas_anom_MMm3_per_day']:+.1f} "
                  f"MMm³/day ({summary['total_weather_gas_anom_pct']:+.0f}%) over the window",
                  fontsize=10.5, fontweight="bold", loc="left", color=INK)
    ax3.grid(lw=0.25, alpha=0.5, axis="y")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.tick_params(labelsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.text(0.03, 0.008, "res/com: SSPM gas model (4.36 MMm³/day/°C below 19.5°C) · power: "
             "national demand V-curve × 0.21 MMm³/GWh marginal gas · normal = model at "
             "climatological pop-weighted temp", fontsize=7.5, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
