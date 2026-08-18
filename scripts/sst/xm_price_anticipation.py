#!/usr/bin/env python3
"""Colombia: is the current price / thermal ramp about ACTUAL water, or
ANTICIPATION of El Nino?

Data: XM bolsa price (PrecBolsNaci hourly -> daily mean, 2015->, cached),
hydro & total generation (2000->), reservoir storage % full (2000->),
inflow % of norm, ONI/Nino-3.4.

Test: for each day, compare the realized price with the price history has
paid AT THE SAME STORAGE LEVEL (+/-3 pts, same calendar season). The
percentile / ratio is the "anticipation premium": high premium = the market
is pricing scarcity that the reservoirs do not yet show.

Outputs: ~/colombia_hydro/out/price_anticipation.json
         ~/colombia_hydro/site/price_anticipation.webp
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xm_storage import reservoir_region                              # noqa: E402

PRIV = Path.home() / "colombia_hydro"
BOLSA = PRIV / "raw" / "bolsa_daily.json.gz"
ESCA = PRIV / "raw" / "escasez_daily.json.gz"      # precio de escasez (indexed benchmark)
GEN = PRIV / "raw" / "generation_daily.json.gz"
STO = PRIV / "raw" / "storage_daily.json.gz"
REPO = HERE.parent.parent
ICLIM = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
NINO = REPO / "assets" / "sst" / "data" / "nino_history.json"
OUT_JSON = PRIV / "out" / "price_anticipation.json"
OUT_PNG = PRIV / "site" / "price_anticipation.webp"
NAVY = "#13273d"; INK = "#1a2733"; MUTE = "#5a6b7a"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]


def fetch_escasez():
    """Daily precio de escasez de activacion (COP/kWh) — the regulator's
    fuel+inflation indexed benchmark; used to deflate the bolsa price."""
    import requests, time
    from datetime import datetime as _dt
    data = json.load(gzip.open(ESCA, "rt")) if ESCA.exists() else {}
    if len(data) > 3000:
        return data
    cur = _dt(2015, 1, 1); end = _dt.now() - timedelta(days=1)
    while cur <= end:
        e2 = min(cur + timedelta(days=29), end)
        for att in range(3):
            try:
                r = requests.post("https://servapibi.xm.com.co/daily",
                                  json={"MetricId": "PrecEscaAct", "StartDate": f"{cur:%Y-%m-%d}",
                                        "EndDate": f"{e2:%Y-%m-%d}", "Entity": "Sistema"}, timeout=90)
                r.raise_for_status(); break
            except Exception:
                if att == 2: raise
                time.sleep(3)
        for it in r.json().get("Items", []):
            for e in it.get("DailyEntities", []):
                data[it["Date"]] = float(e["Value"])
        cur = e2 + timedelta(days=1)
    ESCA.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, gzip.open(ESCA, "wt"), separators=(",", ":"))
    print(f"escasez: {len(data)} days", flush=True)
    return data


def main() -> int:
    price = json.load(gzip.open(BOLSA, "rt"))
    esca = fetch_escasez()
    gen = json.load(gzip.open(GEN, "rt"))
    S = json.load(gzip.open(STO, "rt"))
    res_reg = reservoir_region()
    sto = {}
    for d in sorted(set(S["vol"]) & set(S["cap"])):
        sv = sc = 0.0
        for r_, v in S["vol"][d].items():
            if r_ in res_reg and S["cap"][d].get(r_, 0) > 0:
                sv += v; sc += S["cap"][d][r_]
        if sc > 0:
            sto[d] = 100 * sv / sc
    ic = json.load(open(ICLIM))["full_pct_of_norm"]
    inf = dict(zip(ic["dates"], np.mean([np.array(ic[r], float) for r in ORDER], axis=0)))
    nh = json.load(open(NINO))
    nmon = list(nh["months"])
    _n = nh["series"]["nino34"]
    nval = _n["anom"] if isinstance(_n, dict) and "anom" in _n else _n
    nino = {}
    for m_, v in zip(nmon, nval):
        k = str(m_)
        k = k[:7] if len(k) >= 7 and k[4] in "-/" else k
        nino[k] = v

    # scarcity price is a monthly-indexed regulatory value: forward-fill to daily
    all_days = sorted(set(price) & set(sto) & set(gen))
    esca_ff, lastv = {}, None
    for d in all_days:
        if d in esca:
            lastv = esca[d]
        if lastv is not None:
            esca_ff[d] = lastv
    esca = esca_ff
    # drop partial generation days (incomplete last day / feed hiccups)
    tot_med = np.median([gen[d]["total"] for d in all_days[-400:]])
    days = [d for d in all_days if d in esca and gen[d]["total"] > 0.5 * tot_med]
    dt = [datetime.strptime(d, "%Y-%m-%d") for d in days]
    P_nom = np.array([price[d] for d in days], float)
    SC = np.array([esca[d] for d in days], float)
    P = 100.0 * P_nom / SC                       # bolsa as % of the scarcity price
    E = np.array([sto[d] for d in days], float)
    HY = np.array([gen[d]["hydro"] / max(gen[d]["total"], 1) * 100 for d in days], float)
    IN = np.array([inf.get(d, np.nan) for d in days], float)
    NI = np.array([nino.get(d[:7], np.nan) for d in days], float)
    for i in range(1, len(NI)):                    # ENSO lags ~1 month — carry forward
        if not np.isfinite(NI[i]):
            NI[i] = NI[i - 1]
    doy = np.array([min(x.timetuple().tm_yday, 365) for x in dt])
    yrs = np.array([x.year for x in dt])

    # storage anomaly vs its own doy norm (is water actually scarce?)
    snorm = np.full(365, np.nan)
    for d_ in range(1, 366):
        m = np.minimum(np.abs(doy - d_), 365 - np.abs(doy - d_)) <= 10
        if m.sum() > 30:
            snorm[d_ - 1] = np.median(E[m])
    Eanom = E - snorm[doy - 1]

    # anticipation premium: price vs the price history paid at this storage
    prem, pct_rank = np.full(len(days), np.nan), np.full(len(days), np.nan)
    for i in range(len(days)):
        sel = (np.abs(E - E[i]) <= 3.0) & (np.abs(doy - doy[i]) % 365 <= 45) & (yrs != yrs[i])
        if sel.sum() >= 30:
            ref = P[sel]
            prem[i] = P[i] / np.median(ref)
            pct_rank[i] = 100 * float((ref < P[i]).mean())

    last = -1
    res = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "as_of": days[last], "price_COP_kWh": round(float(P_nom[last]), 1),
           "scarcity_price_COP_kWh": round(float(SC[last]), 1),
           "bolsa_pct_of_scarcity": round(float(P[last]), 1),
           "storage_pct_full": round(float(E[last]), 1),
           "storage_anom_vs_doy_norm_pts": round(float(Eanom[last]), 1),
           "hydro_share_pct": round(float(HY[last]), 1),
           "hydro_share_30d": round(float(np.nanmean(HY[-30:])), 1),
           "inflow_pct_of_norm_30d": round(float(np.nanmean(IN[-30:])), 1),
           "nino34": round(float(NI[last]), 2),
           "anticipation_premium_x": round(float(prem[last]), 2),
           "price_percentile_at_this_storage": round(float(pct_rank[last]), 0),
           "premium_30d_mean_x": round(float(np.nanmean(prem[-30:])), 2)}
    # 2015-16 analog at the same time of year
    an = [i for i in range(len(days)) if days[i][:7] in ("2023-10", "2023-11", "2023-12")]
    if an:
        res["analog_2023_elnino_onset"] = {
            "price": round(float(np.mean(P[an])), 1),
            "storage_pct": round(float(np.mean(E[an])), 1),
            "storage_anom": round(float(np.mean(Eanom[an])), 1),
            "hydro_share": round(float(np.mean(HY[an])), 1),
            "premium_x": round(float(np.nanmean(prem[an])), 2)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 10.5))
    hd = fig.add_axes([0, 0.945, 1, 0.055]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.03, 0.5, "COLOMBIA — PRICE, THERMAL RAMP AND WATER: ANTICIPATION OR SCARCITY?",
            transform=hd.transAxes, color="white", fontsize=13, fontweight="bold", va="center")
    hd.text(0.97, 0.5, f"XM bolsa · through {days[last]}", transform=hd.transAxes,
            color="#b9c6d4", fontsize=9, va="center", ha="right")
    t0 = dt[-1] - timedelta(days=1100)
    m = [i for i, x in enumerate(dt) if x >= t0]
    md = [dt[i] for i in m]
    ax = fig.add_axes([0.07, 0.70, 0.88, 0.20])
    ax.plot(md, P[m], color="#c62828", lw=1.0)
    def roll30(x):
        y = np.convolve(np.asarray(x, float), np.ones(30) / 30, "full")[:len(x)]
        y[:29] = np.nan
        return y
    ax.plot(md, roll30(P)[m], color="#7a1240", lw=2.0, label="30-day mean")
    ax.set_yscale("log"); ax.set_ylabel("bolsa, % of scarcity price", fontsize=8.5, color="#c62828")
    ax.tick_params(labelsize=8, labelcolor="#c62828", axis="y")
    ax.set_title("Spot price as % of the scarcity price (log) — inflation/fuel-cost neutral",
                 fontsize=10, fontweight="bold",
                 loc="left", color=INK)
    ax.grid(lw=0.25, alpha=0.5); ax.legend(fontsize=7.5, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax2 = fig.add_axes([0.07, 0.475, 0.88, 0.185])
    ax2.plot(md, roll30(HY)[m], color="#b35806", lw=2.0, label="hydro share of generation, 30-d")
    ax2.set_ylabel("hydro %", fontsize=8.5, color="#b35806"); ax2.tick_params(labelsize=8, labelcolor="#b35806", axis="y")
    ax2b = ax2.twinx()
    ax2b.plot(md, E[m], color="#1f4e8c", lw=2.0, label="reservoirs % full")
    ax2b.plot(md, snorm[doy[m] - 1], color="#1f4e8c", lw=1.0, ls="--", alpha=0.7, label="seasonal norm")
    ax2b.set_ylabel("% full", fontsize=8.5, color="#1f4e8c"); ax2b.tick_params(labelsize=8, labelcolor="#1f4e8c", axis="y")
    ax2.set_title("Thermal takes over while reservoirs stay AT or ABOVE their seasonal norm",
                  fontsize=10, fontweight="bold", loc="left", color=INK)
    ax2.grid(lw=0.25, alpha=0.5); ax2.tick_params(labelsize=8)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    # premium
    ax3 = fig.add_axes([0.07, 0.27, 0.88, 0.155])
    ax3.plot(md, prem[m], color="#2e7d32", lw=1.6)
    ax3.axhline(1, color="0.4", lw=0.9, ls="--")
    ax3.fill_between(md, 1, prem[m], where=np.array(prem[m]) > 1, color="#2e7d32", alpha=0.18)
    ax3.set_ylabel("× median price\nat same storage", fontsize=8)
    ax3.set_title("ANTICIPATION PREMIUM — price paid vs what history paid at the same reservoir level "
                  "(same season, other years)", fontsize=10, fontweight="bold", loc="left", color=INK)
    ax3.grid(lw=0.25, alpha=0.5); ax3.tick_params(labelsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    # scatter: price vs storage, colour by ENSO, today marked
    ax4 = fig.add_axes([0.07, 0.05, 0.40, 0.16])
    sc = ax4.scatter(E, P, c=NI, cmap="RdBu_r", vmin=-2, vmax=2.5, s=5, alpha=0.55)
    ax4.scatter([E[last]], [P[last]], s=140, marker="*", color="k", zorder=5, label="today")
    if an:
        ax4.scatter([np.mean(E[an])], [np.mean(P[an])], s=90, marker="D", facecolor="none",
                    edgecolor="k", lw=1.5, zorder=5, label="Oct–Dec 2023")
    ax4.set_yscale("log"); ax4.set_xlabel("reservoirs % full", fontsize=8.5)
    ax4.set_ylabel("bolsa % of scarcity", fontsize=8.5)
    ax4.set_title(f"Every day {days[0][:4]}–{days[-1][:4]}, coloured by Niño-3.4", fontsize=9.5, fontweight="bold",
                  loc="left", color=INK)
    ax4.grid(lw=0.25, alpha=0.5); ax4.tick_params(labelsize=7.5); ax4.legend(fontsize=7)
    cb = fig.colorbar(sc, ax=ax4, pad=0.02); cb.set_label("Niño-3.4 °C", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    # verdict box
    axv = fig.add_axes([0.52, 0.05, 0.43, 0.16]); axv.set_axis_off()
    axv.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axv.transAxes, facecolor="#f2f5f8",
                                edgecolor="#d5dde4", lw=0.8))
    a15 = res.get("analog_2023_elnino_onset", {})
    txt = (f"TODAY ({res['as_of']})\n"
           f"  bolsa {res['price_COP_kWh']:.0f} COP/kWh = {res['bolsa_pct_of_scarcity']:.0f}% of scarcity"
           f" ({res['scarcity_price_COP_kWh']:.0f})\n"
           f"  hydro share {res['hydro_share_30d']:.0f}% (30-d)   ·   inflows {res['inflow_pct_of_norm_30d']:.0f}% of norm\n"
           f"  reservoirs {res['storage_pct_full']:.1f}% full ({res['storage_anom_vs_doy_norm_pts']:+.1f} pts vs norm)"
           f"   ·   Niño-3.4 {res['nino34']:+.1f}°C\n"
           f"  PREMIUM ×{res['anticipation_premium_x']:.2f}"
           f"  ({res['price_percentile_at_this_storage']:.0f}th pct at this storage)\n")
    if a15:
        txt += (f"\nOCT–DEC 2023  (last El Niño onset)\n"
                f"  bolsa {a15['price']:.0f}% of scarcity   ·   hydro {a15['hydro_share']:.0f}%\n"
                f"  reservoirs {a15['storage_pct']:.1f}% ({a15['storage_anom']:+.1f} vs norm)"
                f"   ·   premium ×{a15['premium_x']:.2f}")
    axv.text(0.035, 0.94, txt, transform=axv.transAxes, fontsize=7.8, va="top", color=INK,
             family="monospace")
    fig.text(0.03, 0.008, "premium = price ÷ median price on days within ±3 storage points and ±45 doy in OTHER years; "
             ">1 means the market is paying more than the reservoirs alone have historically justified",
             fontsize=7, color=MUTE)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120); plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
