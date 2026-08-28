#!/usr/bin/env python3
"""Backtest of the deck emulator + report of the 10 scenarios.

Layer 1 (21 yr): the ENA/EAR -> CMO proxy on REALIZED data, leave-one-year-out:
  (a) log CMO ~ EAR + ENA90 (what the emulator uses)
  (b) + next-week realized ENA %MLT (perfect-foresight ceiling of the ENA step)
Layer 2 (archive, ~3 wk): emulator run as-of each day; week-1 emulated SE ENA vs
  realized; expected CMO for the next operative week vs realized weekly CMO.
Report: ~/brazil_hydro/site/report/deck_scenarios.pdf (2 pages)
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deck_emulator as D                                            # noqa: E402

PRIV = Path.home() / "brazil_hydro"
CMO_W = PRIV / "raw" / "cmo_weekly.json.gz"
ENA_S = PRIV / "raw" / "ena_subsistema_daily.json.gz"
EAR_S = PRIV / "raw" / "ear_subsistema_daily.json.gz"
ENA_B = PRIV / "raw" / "ena_bacia_daily.json.gz"
OUT_PDF = PRIV / "site" / "report" / "deck_scenarios.pdf"
OUT_JSON = PRIV / "out" / "deck_backtest.json"
NAVY = "#13273d"; INK = "#1a2733"; MUTE = "#5a6b7a"
SE = ["GRANDE", "PARANAIBA", "TIETE", "PARANAPANEMA", "PARANA", "PARAIBA DO SUL"]


def layer1():
    cmo = json.load(gzip.open(CMO_W, "rt"))["SE"]
    ena = json.load(gzip.open(ENA_S, "rt"))["SUDESTE"]
    ear = json.load(gzip.open(EAR_S, "rt"))["SUDESTE"]
    wk = sorted(cmo)
    wd = np.array(wk, dtype="datetime64[D]")
    p = np.log(np.maximum(np.array([cmo[k] for k in wk]), 40.0))
    edays = sorted(ena); edt = np.array(edays, dtype="datetime64[D]")
    epct = np.array([ena[d][1] for d in edays], float)
    sdays = sorted(ear); sdt = np.array(sdays, dtype="datetime64[D]")
    spct = np.array([ear[d] for d in sdays], float)
    def at(dt_arr, v, d, back, fwd=0):
        i = np.searchsorted(dt_arr, d)
        lo, hi = max(0, i - back), min(len(v), i + fwd)
        seg = v[lo:hi]
        return float(np.nanmean(seg)) if len(seg) else np.nan
    ear_w = np.array([at(sdt, spct, d, 7) for d in wd])
    ena90 = np.array([at(edt, epct, d, 90) for d in wd])
    ena_next = np.array([at(edt, epct, d, 0, 7) for d in wd])   # realized next-week ENA
    yrs = wd.astype("datetime64[Y]").astype(int) + 1970
    ok = np.isfinite(ear_w) & np.isfinite(ena90) & np.isfinite(ena_next)
    def loyo(X):
        yh = np.full(len(p), np.nan)
        for y in np.unique(yrs[ok]):
            tr = ok & (yrs != y); te = ok & (yrs == y)
            A = np.column_stack([np.ones(tr.sum()), X[tr]])
            b, *_ = np.linalg.lstsq(A, p[tr], rcond=None)
            yh[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
        m = np.isfinite(yh)
        r = float(np.corrcoef(yh[m], p[m])[0, 1])
        mae = float(np.mean(np.abs(yh[m] - p[m])))
        # direction skill: sign of week-over-week change
        dp = np.diff(p[m]); dh = np.diff(yh[m])
        hit = float(np.mean(np.sign(dp) == np.sign(dh)))
        return {"r_loyo": round(r, 3), "mae_log": round(mae, 3),
                "mae_factor": round(float(np.exp(mae)), 2), "direction_hit": round(hit, 3),
                "n": int(m.sum())}, yh
    r_a, yh_a = loyo(np.column_stack([ear_w, ena90]))
    r_b, yh_b = loyo(np.column_stack([ear_w, ena90, ena_next]))
    return {"proxy_EAR_ENA90": r_a, "plus_realized_next_week_ENA": r_b,
            "series": {"weeks": wk, "logcmo": p.tolist(), "fit_a": yh_a.tolist(), "fit_b": yh_b.tolist()}}


def layer2():
    """Run the emulator as-of each archived day; verify week-1 ENA and week-1 CMO."""
    ena_b = json.load(gzip.open(ENA_B, "rt"))
    cmo = json.load(gzip.open(CMO_W, "rt"))["SE"]
    cmo_w = {np.datetime64(k): v for k, v in cmo.items()}
    cmo_keys = sorted(cmo_w)
    inits = sorted({f.name.split("_")[1] for f in (PRIV / "raw" / "fcst_rain").glob("ifs_*_00z.json.gz")})
    rows = []
    for d8 in inits:
        asof = f"{d8[:4]}-{d8[4:6]}-{d8[6:8]}"
        try:
            r = D.run(asof, make_fig=False)
        except Exception as e:                       # noqa: BLE001
            print("  asof", asof, "failed:", str(e)[:60]); continue
        days = [np.datetime64(x) for x in r["days"]]
        # emulated week-1 SE ENA (MWmed): sum of deterministic per-basin
        em = np.sum([np.array(r["ena_det_mwmed"][b]) for b in SE if b in r["ena_det_mwmed"]], axis=0)
        # realized
        real = []
        for d in days[:7]:
            k = str(d)
            v = sum(ena_b[b][k][0] for b in SE if k in ena_b.get(b, {}))
            real.append(v if v > 0 else np.nan)
        real = np.array(real)
        # CMO of the next operative week (first Saturday after asof → key at/after)
        nxt = [k for k in cmo_keys if k > np.datetime64(asof)]
        cmo_next = cmo_w[nxt[0]] if nxt else np.nan
        exp = r["price_se"]["expected"]
        rows.append({"asof": asof, "ena_em_w1": float(np.nanmean(em[:7])),
                     "ena_real_w1": float(np.nanmean(real)),
                     "cmo_exp_w1": float(np.mean(exp[:7])),
                     "cmo_now": r["price_se"]["cmo_now_weekly"],
                     "cmo_next_real": float(cmo_next) if np.isfinite(cmo_next) else None,
                     "cmo_p10_w1": float(np.mean(r["price_se"]["p10"][:7])),
                     "cmo_p90_w1": float(np.mean(r["price_se"]["p90"][:7]))})
        print(f"  {asof}: ENA em {rows[-1]['ena_em_w1']:.0f} real {rows[-1]['ena_real_w1']:.0f} | "
              f"CMO exp {rows[-1]['cmo_exp_w1']:.0f} now {rows[-1]['cmo_now']} next {cmo_next}", flush=True)
    return rows


def main() -> int:
    L1 = layer1()
    print("Layer 1:", json.dumps({k: v for k, v in L1.items() if k != "series"}))
    L2 = layer2()
    live = D.run(None, make_fig=False)
    OUT_JSON.write_text(json.dumps({"layer1": {k: v for k, v in L1.items() if k != "series"},
                                    "layer2": L2}, indent=1))
    # ── report ───────────────────────────────────────────────────────────────
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        # page 1: scenarios
        fd = [datetime.strptime(x, "%Y-%m-%d") for x in live["days"]]
        cl = live["clusters"]
        fig = plt.figure(figsize=(11.69, 8.27))
        hd = fig.add_axes([0, 0.925, 1, 0.075]); hd.set_axis_off()
        hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
        hd.text(0.03, 0.58, "BRAZIL — DECK EMULATOR: 10 SCENARIOS, IMPLIED SE/CO PRICES",
                transform=hd.transAxes, color="white", fontsize=12.5, fontweight="bold", va="center")
        hd.text(0.03, 0.18, "previsão conjunta (ECMWF ENS + AIFS-ENS, bias-removed & weighted) → "
                "per-basin selected rain→ENA models → k-means(10) scenarios → CMO proxy",
                transform=hd.transAxes, color="#b9c6d4", fontsize=7.6, va="center")
        hd.text(0.97, 0.58, " · ".join(f"{m.upper()} {v.split()[0][4:]}" for m, v in live["inits"].items()),
                transform=hd.transAxes, color="white", fontsize=9, fontweight="bold", va="center", ha="right")
        hd.text(0.97, 0.18, f"generated {live['generated']} · season: {live['season']}",
                transform=hd.transAxes, color="#b9c6d4", fontsize=7.6, va="center", ha="right")
        # left: SE ENA scenarios
        ax = fig.add_axes([0.06, 0.50, 0.55, 0.38])
        se_mlt = None
        for i, c in enumerate(cl):
            ena_se = np.sum([np.array(c["ena"][b]) for b in SE if b in c["ena"]], axis=0)
            ax.plot(fd, ena_se, color=plt.cm.tab10(i % 10), lw=0.9 + 3.5 * c["weight"], alpha=0.85,
                    label=f"S{i+1}  w={c['weight']:.2f}  ({c['ena_se_mean_pct_mlt']:.0f}% MLT)")
        det = np.sum([np.array(live["ena_det_mwmed"][b]) for b in SE if b in live["ena_det_mwmed"]], axis=0)
        ax.plot(fd, det, color="k", lw=1.6, ls="--", label="deterministic conjunta")
        ax.set_title("SE/CO natural inflow energy (MWmed) — the 10 scenarios (line width ∝ probability)",
                     fontsize=9.6, fontweight="bold", loc="left", color=INK)
        ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=6.4, ncol=2, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        # right: table
        axt = fig.add_axes([0.64, 0.50, 0.33, 0.38]); axt.set_axis_off()
        rows = [["Sc", "prob", "ENA %MLT", "CMO wk1", "CMO wk2", "CMO d18"]]
        for i, c in enumerate(cl):
            pr = c["cmo_se"]
            rows.append([f"S{i+1}", f"{c['weight']:.2f}", f"{c['ena_se_mean_pct_mlt']:.0f}",
                         f"{np.mean(pr[:7]):.0f}", f"{np.mean(pr[7:14]):.0f}", f"{pr[-1]:.0f}"])
        ex = live["price_se"]["expected"]
        rows.append(["E[·]", "1.00", "", f"{np.mean(ex[:7]):.0f}", f"{np.mean(ex[7:14]):.0f}", f"{ex[-1]:.0f}"])
        tbl = axt.table(cellText=rows[1:], colLabels=rows[0], loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.4); tbl.scale(1, 1.25)
        for (r_, c_), cell in tbl.get_celld().items():
            cell.set_edgecolor("#d5dde4")
            if r_ == 0: cell.set_facecolor("#e8eef4"); cell.set_text_props(fontweight="bold")
            if r_ == len(rows) - 1: cell.set_facecolor("#fbe9d7"); cell.set_text_props(fontweight="bold")
        axt.set_title("Implied SE/CO CMO (R$/MWh) per scenario", fontsize=9.6, fontweight="bold",
                      loc="left", color=INK)
        # bottom: prices
        ax2 = fig.add_axes([0.06, 0.07, 0.91, 0.36])
        for i, c in enumerate(cl):
            ax2.plot(fd, c["cmo_se"], color=plt.cm.tab10(i % 10), lw=0.9 + 3.5 * c["weight"], alpha=0.8)
        ax2.plot(fd, ex, color="#c62828", lw=2.4, label="probability-weighted expected CMO")
        ax2.fill_between(fd, live["price_se"]["p10"], live["price_se"]["p90"], color="#c62828", alpha=0.13,
                         label="P10–P90 across scenarios")
        if live["price_se"]["cmo_now_weekly"]:
            ax2.axhline(live["price_se"]["cmo_now_weekly"], color="0.35", lw=1.1, ls=":",
                        label=f"latest published weekly CMO {live['price_se']['cmo_now_weekly']:.0f}")
        ax2.set_yscale("log"); ax2.set_ylabel("R$/MWh (log)", fontsize=8.5)
        from matplotlib.ticker import ScalarFormatter, NullFormatter
        ax2.yaxis.set_major_formatter(ScalarFormatter()); ax2.yaxis.set_minor_formatter(NullFormatter())
        lo_, hi_ = min(min(c["cmo_se"]) for c in cl), max(max(c["cmo_se"]) for c in cl)
        if live["price_se"]["cmo_now_weekly"]:
            hi_ = max(hi_, live["price_se"]["cmo_now_weekly"] * 1.05)
        ax2.set_ylim(lo_ * 0.9, hi_ * 1.1)
        ax2.set_title(f"CMO paths per scenario · EAR SE/CO now {live['price_se']['ear_now_pct']:.1f}% · "
                      f"ENA90 {live['price_se']['ena90_now_pct_mlt']:.0f}% MLT · proxy: log CMO ~ EAR + ENA90",
                      fontsize=9.6, fontweight="bold", loc="left", color=INK)
        ax2.grid(lw=0.25, alpha=0.5); ax2.tick_params(labelsize=7.5); ax2.legend(fontsize=7.2, loc="upper left")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.text(0.03, 0.038, "READ WITH THE BACKTEST (p.2): the ENA scenarios are informative — week-1 SE ENA "
                 "r=0.78 vs realized (−18% biased).", fontsize=7.2, color="#8a4a00")
        fig.text(0.03, 0.025, "The CMO mapping is a REGIME indicator, NOT a weekly price forecast: 21-yr levels "
                 "r=0.67 but week-over-week direction ~47% (coin-flip) — DECOMP steps with CVaR/VMinOp "
                 "thresholds this proxy cannot see.", fontsize=7.2, color="#8a4a00")
        fig.text(0.03, 0.012, "v0.1 · clusters are a provisional k-means(10) stand-in for ONS's EC46 51→10 "
                 "aggregation · CMO is a statistical proxy (no CVaR/VMinOp) · not investment advice",
                 fontsize=7, color=MUTE)
        pdf.savefig(fig, dpi=200); plt.close(fig)

        # page 2: backtest
        fig = plt.figure(figsize=(11.69, 8.27))
        hd = fig.add_axes([0, 0.925, 1, 0.075]); hd.set_axis_off()
        hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
        hd.text(0.03, 0.5, "BACKTEST — how much of this is real?", transform=hd.transAxes,
                color="white", fontsize=13, fontweight="bold", va="center")
        s1 = L1["series"]; wk = [datetime.strptime(k, "%Y-%m-%d") for k in s1["weeks"]]
        ax = fig.add_axes([0.06, 0.56, 0.60, 0.33])
        ax.plot(wk, np.exp(s1["logcmo"]), color="#1f4e8c", lw=0.9, label="realized weekly CMO SE/CO")
        fa = np.array(s1["fit_a"], float)
        ax.plot(wk, np.exp(fa), color="#c62828", lw=0.9, alpha=0.9, label="proxy (EAR + ENA90), leave-one-year-out")
        ax.set_yscale("log"); ax.set_ylabel("R$/MWh", fontsize=8.5)
        a = L1["proxy_EAR_ENA90"]; b_ = L1["plus_realized_next_week_ENA"]
        ax.set_title(f"Layer 1 — the price proxy on realized hydrology, 21 yr LOYO: r={a['r_loyo']:.2f}, "
                     f"median error ×{a['mae_factor']:.2f}, week-over-week direction {100*a['direction_hit']:.0f}%\n"
                     f"with perfect next-week ENA foresight: r={b_['r_loyo']:.2f}, ×{b_['mae_factor']:.2f}, "
                     f"direction {100*b_['direction_hit']:.0f}%  ← ceiling of the ENA step",
                     fontsize=8.8, fontweight="bold", loc="left", color=INK)
        ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=7.5); ax.legend(fontsize=7)
        # scatter proxy vs realized
        axs = fig.add_axes([0.71, 0.56, 0.26, 0.33])
        m = np.isfinite(fa)
        axs.scatter(np.exp(np.array(s1["logcmo"])[m]), np.exp(fa[m]), s=6, alpha=0.5, color="#1f4e8c")
        lim = [40, 3000]; axs.plot(lim, lim, color="0.5", lw=0.8, ls="--")
        axs.set_xscale("log"); axs.set_yscale("log"); axs.set_xlim(lim); axs.set_ylim(lim)
        axs.set_xlabel("realized CMO", fontsize=8); axs.set_ylabel("proxy", fontsize=8)
        axs.set_title("weekly, 2005–2026", fontsize=8.5, fontweight="bold", loc="left")
        axs.grid(lw=0.25, alpha=0.5); axs.tick_params(labelsize=7)
        # layer 2
        if L2:
            ad = [datetime.strptime(r["asof"], "%Y-%m-%d") for r in L2]
            ax3 = fig.add_axes([0.06, 0.08, 0.42, 0.38])
            ax3.plot(ad, [r["ena_em_w1"] for r in L2], color="#b35806", lw=1.8, marker="o", ms=3.5,
                     label="emulated week-1 SE ENA (as-of)")
            ax3.plot(ad, [r["ena_real_w1"] for r in L2], color="#1f4e8c", lw=1.8, marker="s", ms=3.5,
                     label="realized week-1 SE ENA")
            em = np.array([r["ena_em_w1"] for r in L2]); re_ = np.array([r["ena_real_w1"] for r in L2])
            mm = np.isfinite(em) & np.isfinite(re_)
            rr = np.corrcoef(em[mm], re_[mm])[0, 1] if mm.sum() > 3 else np.nan
            pb = 100 * (em[mm].mean() / re_[mm].mean() - 1)
            ax3.set_title(f"Layer 2 — emulator vs realized week-1 ENA, {mm.sum()} as-of days: r={rr:.2f}, "
                          f"bias {pb:+.0f}%", fontsize=8.8, fontweight="bold", loc="left", color=INK)
            ax3.set_ylabel("MWmed", fontsize=8); ax3.grid(lw=0.25, alpha=0.5)
            ax3.tick_params(labelsize=7.5); ax3.legend(fontsize=7)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax4 = fig.add_axes([0.55, 0.08, 0.42, 0.38])
            ax4.plot(ad, [r["cmo_exp_w1"] for r in L2], color="#c62828", lw=1.8, marker="o", ms=3.5,
                     label="emulator expected CMO, next week")
            ax4.fill_between(ad, [r["cmo_p10_w1"] for r in L2], [r["cmo_p90_w1"] for r in L2],
                             color="#c62828", alpha=0.12)
            ax4.plot(ad, [r["cmo_next_real"] if r["cmo_next_real"] else np.nan for r in L2], color="#1f4e8c",
                     lw=1.8, marker="s", ms=3.5, label="realized CMO, next operative week")
            ax4.plot(ad, [r["cmo_now"] for r in L2], color="0.5", lw=1.0, ls=":", label="latest published CMO at as-of")
            ax4.set_title("Layer 2 — expected next-week CMO vs realized (proxy level, small sample)",
                          fontsize=8.8, fontweight="bold", loc="left", color=INK)
            ax4.set_ylabel("R$/MWh", fontsize=8); ax4.grid(lw=0.25, alpha=0.5)
            ax4.tick_params(labelsize=7.5); ax4.legend(fontsize=7)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.text(0.03, 0.012, "Layer 1 uses REALIZED ENA/EAR (tests the price proxy only). Layer 2 uses only "
                 "information available at each as-of date (archive from late July 2026 — weeks, not years).",
                 fontsize=7, color=MUTE)
        pdf.savefig(fig, dpi=200); plt.close(fig)
    print(f"wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
