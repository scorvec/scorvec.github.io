#!/usr/bin/env python3
"""ONS deck emulator v0 — "what will Thursday's DECOMP see", and price per
scenario. Documented in ~/brazil_hydro/docs/ONS_DECOMP_RECIPE.md.

Stage A — previsão conjunta (NT-ONS 0075/2020 logic, v0 approximation):
  inputs: ECMWF ENS mean (from our fcst_rain archive, 'ifs') + GEFS mean
  ('gefs', if archived) [+ ETA: TODO CPTEC feed] per basin per day.
  reference: CPTEC MERGE basin means (v0: gauge-corrected IMERG truth —
  MERGE ingest queued; both are 0.1° gauge-merged satellite products).
  bias removal: per basin/model, ratio sum(obs)/sum(fcst) over the last 120
  matured days at each lead band; combination weights per basin/model ∝
  1/MAE over the same 120 d (proxy for ONS's optimization), summing to 1;
  caps: daily <= 99th pct of the reference record for the basin.
  output: conjunta[b][d+1..d+14] mm/d, extended to d+18 with the 14-d mean.

Stage B — flows: conjunta -> SMAP (calibrated per basin, states carried
  from the truth record) -> ENA MWmed weeks 1-2 (deterministic, as ONS).

Stage C — scenarios weeks 3-4: ONS clusters ECMWF extended-range 51->10.
  v0: cluster the available ensemble members' days 8-15 (ECMWF ENS to 15 d;
  GEFS to 35 d when archived) by k-means(10) on the basin-rain vector; each
  representative member's daily rain -> SMAP -> ENA path; weight = cluster
  size / N. FLAGGED PROVISIONAL (EC46 not public; algorithm pending CT report).

Stage D — price per scenario: SE/CO EAR trajectory per scenario via the
  water balance (S' = S + ENA - outflow_norm), then the fitted CMO relation
  log CMO = a + b*EAR + c*ENA90 (brazil_cmo.py, r=0.71) -> CMO path per
  scenario + probability-weighted mean and P10/P90. This is a STATISTICAL
  PROXY for DECOMP (no CVaR/VMinOp mechanics beyond what the fit absorbs).

Outputs: ~/brazil_hydro/out/deck_emulator.json, site/deck_emulator.webp
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

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from smap_ons import smap_run                                        # noqa: E402
from rain_inflow_model import ema                                    # noqa: E402

PRIV = Path.home() / "brazil_hydro"
ARCH = PRIV / "raw" / "fcst_rain"
TRUTH = PRIV / "raw" / "imerg_basin_daily.json"
SMAP = PRIV / "out" / "smap_params.json"
SELECTOR = PRIV / "out" / "ena_selector.json"
ENA = PRIV / "raw" / "ena_bacia_daily.json.gz"
EAR = PRIV / "raw" / "ear_subsistema_daily.json.gz"
CMO_AN = PRIV / "out" / "cmo_analysis.json"
CMO_W = PRIV / "raw" / "cmo_weekly.json.gz"
OUT_JSON = PRIV / "out" / "deck_emulator.json"
OUT_PNG = PRIV / "site" / "deck_emulator.webp"
SE_BASINS = ["GRANDE", "PARANAIBA", "TIETE", "PARANAPANEMA", "PARANA", "PARAIBA DO SUL"]
NCLUST = 10
NAVY = "#13273d"; INK = "#1a2733"


def load_latest(model):
    fs = sorted(ARCH.glob(f"{model}_*.json.gz"))
    return json.loads(gzip.open(fs[-1], "rt").read()) if fs else None


def matured_pairs(model, truth, tdates, days_back=120):
    """per basin: list of (lead, fcst_mean, obs) over the last N days."""
    dmap = {str(d): i for i, d in enumerate(tdates)}
    cutoff = tdates[-1] - np.timedelta64(days_back, "D")
    pairs = {}
    for f in sorted(ARCH.glob(f"{model}_*.json.gz")):
        rec = json.loads(gzip.open(f, "rt").read())
        d0 = np.datetime64(f"{rec['init_date'][:4]}-{rec['init_date'][4:6]}-{rec['init_date'][6:8]}")
        if d0 < cutoff - np.timedelta64(16, "D"):
            continue
        for li, vd in enumerate(rec["valid"]):
            i = dmap.get(vd)
            if i is None or np.datetime64(vd) < cutoff:
                continue
            lead = int((np.datetime64(vd) - d0).astype(int)) + 1
            for b, mem in rec["basins"].items():
                if b not in truth:
                    continue
                pairs.setdefault(b, []).append((lead, float(np.mean([m[li] for m in mem])),
                                                float(truth[b][i])))
    return pairs


def main() -> int:
    tc = json.loads(TRUTH.read_text())
    tdates = np.array([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in tc["dates"]], dtype="datetime64[D]")
    smap = json.loads(SMAP.read_text())["params"]
    basins = [b for b in smap if b in tc]
    models = [m for m in ("ifs", "aifs", "gefs") if load_latest(m)]
    latest = {m: load_latest(m) for m in models}
    print("models available:", models, "| basins with SMAP:", len(basins))

    # ── Stage A: bias factors + weights per basin/model (120-d), by lead band
    bands = [(1, 3), (4, 7), (8, 15)]
    def band(l): return next(i for i, (a, b_) in enumerate(bands) if a <= l <= b_) if l <= 15 else 2
    F, Wt = {}, {}
    for m in models:
        pr = matured_pairs(m, tc, tdates)
        for b in basins:
            P = pr.get(b, [])
            for bi in range(3):
                sel = [(f_, o) for (l, f_, o) in P if band(l) == bi]
                if len(sel) >= 10:
                    sf = sum(x[0] for x in sel); so = sum(x[1] for x in sel)
                    F[(m, b, bi)] = (so + 5) / (sf + 5)
                    mae = np.mean([abs(f_ * F[(m, b, bi)] - o) for f_, o in sel])
                    Wt[(m, b, bi)] = 1.0 / max(mae, 0.05)
                else:
                    F[(m, b, bi)] = 1.0; Wt[(m, b, bi)] = 1.0
    # caps: 99th pct daily rain per basin
    cap = {b: float(np.nanpercentile(np.array(tc[b], float), 99)) for b in basins}

    # forecast day axis: d+1..d+18 from the newest init among models
    init0 = max(np.datetime64(f"{r['init_date'][:4]}-{r['init_date'][4:6]}-{r['init_date'][6:8]}")
                for r in latest.values())
    fdays = [init0 + np.timedelta64(k, "D") for k in range(1, 19)]
    conj = {b: np.full(18, np.nan) for b in basins}
    members_pool = {b: [] for b in basins}                # (weight, [18-day rain])
    for b in basins:
        for k, d in enumerate(fdays):
            lead = k + 1
            num = den = 0.0
            for m in models:
                rec = latest[m]
                vmap = {np.datetime64(v): i for i, v in enumerate(rec["valid"])}
                if d not in vmap or b not in rec["basins"]:
                    continue
                li = vmap[d]
                mean_f = float(np.mean([mm[li] for mm in rec["basins"][b]]))
                f = F[(m, b, band(lead))]; w = Wt[(m, b, band(lead))]
                num += w * min(mean_f * f, cap[b]); den += w
            if den > 0:
                conj[b][k] = num / den
        # ONS extension: days beyond the 14-day horizon = mean of the 14
        core = conj[b][:14]
        if np.isfinite(core).sum() >= 10:
            fill = float(np.nanmean(core))
            conj[b] = np.where(np.isfinite(conj[b]), conj[b], fill)
        # member pool for clustering (bias-corrected members, weeks 3-4 proxy)
        for m in models:
            rec = latest[m]
            vmap = {np.datetime64(v): i for i, v in enumerate(rec["valid"])}
            for mm in rec["basins"].get(b, []):
                row = []
                for k, d in enumerate(fdays):
                    li = vmap.get(d)
                    row.append(min(mm[li] * F[(m, b, band(k + 1))], cap[b]) if li is not None else np.nan)
                members_pool[b].append((1.0 / rec["n_members"], row))

    # ── Stage C: k-means(10) on the pooled member vectors (all basins jointly)
    #    over days 8-18 (the part beyond the deterministic fortnight in DECOMP terms)
    from scipy.cluster.vq import kmeans2
    nmem = min(len(members_pool[b]) for b in basins)
    X = np.array([[np.nan_to_num(members_pool[b][j][1][7:18],
                                 nan=np.nanmean(conj[b])) for b in basins]
                  for j in range(nmem)]).reshape(nmem, -1)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    np.random.seed(0)
    cent, lab = kmeans2(Xs, NCLUST, minit="++", seed=0)
    clusters = []
    for c in range(NCLUST):
        idx = np.where(lab == c)[0]
        if len(idx) == 0:
            continue
        # representative = member closest to centroid
        rep = idx[np.argmin(((Xs[idx] - cent[c]) ** 2).sum(1))]
        clusters.append({"id": c, "n": int(len(idx)), "weight": len(idx) / nmem,
                         "rain": {b: [round(float(v), 2) if np.isfinite(v) else None
                                      for v in members_pool[b][rep][1]] for b in basins}})
    clusters.sort(key=lambda c: -c["weight"])

    # ── Stage B: rain -> ENA via the per-basin/season SELECTED model
    #    (ena_selector.json: kernel_v3 / cascade / hybrid / smap_*), evaluated
    #    on observed history + the forecast days so kernels carry true state.
    selector = json.loads(SELECTOR.read_text())["basins"] if SELECTOR.exists() else {}
    month_now = fdays[0].astype(object).month
    season_now = "dry" if 5 <= month_now <= 10 else "wet"
    hist_rain = {b: np.nan_to_num(np.array(tc[b], float)) for b in basins}
    hdates = [d.astype(object) for d in tdates]
    used_model = {}

    def ena_path(rain_by_basin):
        out = {}
        for b in basins:
            p = smap[b]
            prm = (p["Str"], p["K2t"], p["Crec"], p["Ai"], p["Capc"], p["Kkt"],
                   p["gain_MW_per_mmday"])
            r = np.array([np.nan_to_num(v, nan=float(np.nanmean(conj[b])))
                          for v in rain_by_basin[b]], float)
            m0 = [fdays[k].astype(object).month - 1 for k in range(len(fdays))]
            Ep = np.array([p["pet_monthly_mmday"][mm] for mm in m0])
            Q_smap, _ = smap_run(r, Ep, prm, states=tuple(p["states_end"]))
            sel = selector.get(b, {})
            mdl = sel.get("select", {}).get(season_now, {}).get("model", "smap_nse")
            used_model[b] = mdl
            if mdl.startswith("smap") or "fits" not in sel:
                out[b] = Q_smap
                continue
            fits = sel["fits"]
            full = np.concatenate([hist_rain[b], r])
            nh = len(hist_rain[b])
            alld = hdates + [d.astype(object) for d in fdays]
            doy = np.array([min(d.timetuple().tm_yday, 365) for d in alld])
            th = 2 * np.pi * doy / 365
            season = np.column_stack([np.sin(th), np.cos(th)])
            S = np.full(len(full), sel["ear_anom_now"])
            if mdl == "kernel_v3" and "kernel_v3" in fits:
                f = fits["kernel_v3"]
                Pm = full - f["rain_mean"]
                k = np.roll(ema(Pm, f["tau"]), f["lag"]); ks = np.roll(ema(Pm, 180), f["lag"])
                X = np.column_stack([k, ks, S, season])
                c = np.array(f["coefs"])
                out[b] = (c[0] + X @ c[1:])[nh:]
            elif mdl in ("cascade", "hybrid") and mdl in fits:
                f = fits[mdl]
                E = np.column_stack([ema(full, t) for t in f["taus"]])
                cols = [E, season, -season, S[:, None], -S[:, None]]
                if mdl == "hybrid":
                    # smap runoff over the whole history+forecast (mm->MW via gain)
                    Ep_full = np.array([p["pet_monthly_mmday"][d.month - 1] for d in alld])
                    Qh, _ = smap_run(full, Ep_full, prm)
                    cols = [Qh[:, None]] + cols
                X = np.column_stack(cols)
                c = np.array(f["coefs"])
                out[b] = (c[0] + X @ c[1:])[nh:]
            else:
                out[b] = Q_smap
        return out
    det = ena_path({b: conj[b] for b in basins})
    for c in clusters:
        # weeks 1-2 deterministic (as ONS), cluster rain from day 15 on... v0: from day 8
        rain = {b: list(conj[b][:7]) + [v if v is not None else conj[b][k + 7]
                                         for k, v in enumerate(c["rain"][b][7:])]
                for b in basins}
        c["ena"] = {b: np.round(v, 1).tolist() for b, v in ena_path(rain).items()}

    # ── Stage D: price per scenario (SE/CO)
    cmo = json.loads(CMO_AN.read_text())
    a0, bE, cN = cmo["joint_coefs"]["intercept"], cmo["joint_coefs"]["per_EAR_pct"], cmo["joint_coefs"]["per_ENA90_pct"]
    with gzip.open(EAR, "rt") as f:
        ear = json.load(f)["SUDESTE"]
    with gzip.open(ENA, "rt") as f:
        ena_hist = json.load(f)
    ear_now = float(ear[max(ear)])
    # SE ENA %MLT recent 90-d mean and MLT scale (sum over SE basins of MW/pct)
    se_hist = {}
    for b in SE_BASINS:
        d = ena_hist.get(b, {})
        for k, (mw, pc) in d.items():
            se_hist.setdefault(k, [0.0, 0.0])
            se_hist[k][0] += mw; se_hist[k][1] += mw / max(pc, 1e-3) * 100
    hk = sorted(se_hist)[-90:]
    mlt_se = np.mean([se_hist[k][1] for k in hk])                  # MW at 100%
    ena90_hist = np.mean([se_hist[k][0] for k in hk]) / mlt_se * 100
    # storage capacity SE (MWmes) from EAR pct + ENA: use CMO regression directly on
    # EAR path: dEAR/dt ≈ (ENA - outflow)/EARmax; EARmax_SE ≈ 200,000 MWmes (ONS)
    EARMAX = 200000.0
    outflow = mlt_se * 0.95                                          # dry-season SE outflow ≈ ~MLT
    def price_path(ena_se):
        e = ear_now; e90 = list(np.full(90, ena90_hist))
        path = []
        for k, v in enumerate(ena_se):
            e = np.clip(e + (v - outflow) * (1 / 30.4) / EARMAX * 100, 0, 100)
            e90 = e90[1:] + [v / mlt_se * 100]
            path.append(float(np.exp(a0 + bE * e + cN * np.mean(e90))))
        return path
    ena_se_det = np.sum([det[b] for b in SE_BASINS if b in det], axis=0)
    p_det = price_path(ena_se_det)
    for c in clusters:
        ena_se = np.sum([np.array(c["ena"][b]) for b in SE_BASINS if b in c["ena"]], axis=0)
        c["cmo_se"] = np.round(price_path(ena_se), 1).tolist()
        c["ena_se_mean_pct_mlt"] = round(float(ena_se.mean() / mlt_se * 100), 1)
    w = np.array([c["weight"] for c in clusters])
    P = np.array([c["cmo_se"] for c in clusters])
    exp_price = (w[:, None] * P).sum(0) / w.sum()
    order = np.argsort(P, axis=0)
    def wq(q):
        out = []
        for k in range(P.shape[1]):
            col = P[:, k]; idx = np.argsort(col); cw = np.cumsum(w[idx]) / w.sum()
            out.append(float(col[idx][np.searchsorted(cw, q)]))
        return out
    res = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "inits": {m: f"{latest[m]['init_date']} {latest[m]['init_hh']}Z" for m in models},
           "days": [str(d) for d in fdays],
           "conjunta_mmday": {b: np.round(conj[b], 2).tolist() for b in basins},
           "ena_det_mwmed": {b: np.round(det[b], 1).tolist() for b in basins},
           "clusters": clusters,
           "price_se": {"deterministic_conjunta": np.round(p_det, 1).tolist(),
                        "expected": np.round(exp_price, 1).tolist(),
                        "p10": np.round(wq(0.10), 1).tolist(),
                        "p90": np.round(wq(0.90), 1).tolist(),
                        "ear_now_pct": ear_now, "ena90_now_pct_mlt": round(ena90_hist, 1),
                        "cmo_now_weekly": None},
           "rain_to_ena_model_used": used_model, "season": season_now,
           "notes": ["v0: MERGE reference not yet wired (corrected IMERG used)",
                     "v0: ETA not included; GEFS only if archived",
                     "clusters: k-means(10) on days 8-18 of the available members — "
                     "PROVISIONAL stand-in for ONS's EC46 51->10 clustering",
                     "price: statistical CMO proxy (log CMO ~ EAR + ENA90), not DECOMP"]}
    try:
        cw = json.load(gzip.open(CMO_W, "rt"))["SE"]
        res["price_se"]["cmo_now_weekly"] = cw[max(cw)]
    except Exception:
        pass
    OUT_JSON.write_text(json.dumps(res, indent=1))

    # ── figure
    fd = [d.astype(object) for d in fdays]
    fig = plt.figure(figsize=(13.5, 9.5))
    hd = fig.add_axes([0, 0.93, 1, 0.07]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.03, 0.5, "BRAZIL — ONS DECK EMULATOR v0.1: what Thursday's DECOMP will see",
            transform=hd.transAxes, color="white", fontsize=13.5, fontweight="bold", va="center")
    hd.text(0.97, 0.5, " · ".join(f"{m.upper()} {latest[m]['init_date']} {latest[m]['init_hh']}Z" for m in models),
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center", ha="right")
    ax = fig.add_axes([0.06, 0.55, 0.42, 0.33])
    for b, col in zip(["GRANDE", "PARANAIBA", "IGUACU", "URUGUAI", "SAO FRANCISCO", "TOCANTINS"],
                      ["#c62828", "#e08214", "#1f4e8c", "#5b87c0", "#2e9e4f", "#6a3d9a"]):
        if b in conj:
            ax.plot(fd, conj[b], color=col, lw=1.6, marker="o", ms=3, label=b)
    ax.set_title("Previsão conjunta (bias-removed, weighted ECMWF+GEFS), mm/day",
                 fontsize=10, fontweight="bold", loc="left", color=INK)
    ax.grid(lw=0.25, alpha=0.5); ax.legend(fontsize=7); ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2 = fig.add_axes([0.56, 0.55, 0.40, 0.33])
    for c in clusters:
        ena_se = np.sum([np.array(c["ena"][b]) for b in SE_BASINS if b in c["ena"]], axis=0)
        ax2.plot(fd, ena_se / mlt_se * 100, color="#b35806", lw=0.8 + 3 * c["weight"], alpha=0.7)
    ax2.plot(fd, ena_se_det / mlt_se * 100, color="k", lw=2, ls="--", label="deterministic conjunta")
    ax2.axhline(100, color="0.6", lw=0.7, ls=":")
    ax2.set_title(f"SE/CO ENA scenarios via selected models ({season_now}) — {len(clusters)} clusters",
                  fontsize=10, fontweight="bold", loc="left", color=INK)
    ax2.set_ylabel("% of MLT", fontsize=8.5); ax2.grid(lw=0.25, alpha=0.5)
    ax2.legend(fontsize=7.5); ax2.tick_params(labelsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax3 = fig.add_axes([0.06, 0.08, 0.90, 0.38])
    for c in clusters:
        ax3.plot(fd, c["cmo_se"], color="#9db8d8", lw=0.8 + 3 * c["weight"], alpha=0.8)
    ax3.plot(fd, exp_price, color="#c62828", lw=2.4, label="probability-weighted expected CMO")
    ax3.fill_between(fd, wq(0.10), wq(0.90), color="#c62828", alpha=0.15, label="P10–P90 across scenarios")
    ax3.plot(fd, p_det, color="k", lw=1.6, ls="--", label="deterministic conjunta path")
    if res["price_se"]["cmo_now_weekly"]:
        ax3.axhline(res["price_se"]["cmo_now_weekly"], color="0.4", lw=1, ls=":",
                    label=f"latest weekly CMO {res['price_se']['cmo_now_weekly']:.0f}")
    ax3.set_yscale("log")
    ax3.set_title(f"SE/CO CMO per scenario (statistical proxy) · EAR now {ear_now:.1f}% · "
                  f"ENA90 {ena90_hist:.0f}% MLT", fontsize=10, fontweight="bold", loc="left", color=INK)
    ax3.set_ylabel("R$/MWh (log)", fontsize=8.5); ax3.grid(lw=0.25, alpha=0.5)
    ax3.legend(fontsize=7.5, loc="upper left"); ax3.tick_params(labelsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.text(0.03, 0.012, "v0 caveats: IMERG-corrected as MERGE proxy · no ETA · clusters = k-means(10) "
             "on member days 8-18 (EC46 not public) · price = log CMO ~ EAR + ENA90 fit, not DECOMP",
             fontsize=7.5, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120); plt.close(fig)
    print(f"clusters: {[(c['n'], round(c['weight'],2), c['ena_se_mean_pct_mlt']) for c in clusters]}")
    print(f"expected CMO d1..d18: {np.round(exp_price[[0,6,13,17]],0)}  det: {np.round(np.array(p_det)[[0,6,13,17]],0)}")
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
