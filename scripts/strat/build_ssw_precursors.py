#!/usr/bin/env python3
"""Reference for the "Stratosphere history" warming-study items (2026-09-27): who moves first around a sudden warming,
the QBO and the vortex, El Nino and sudden warmings in CMIP6, and the seasonal-odds card. LAPTOP, run once.

It packs the results of the tropical-precursor study (~/research/ssw_tropical_precursors: MERRA-2, ERA5, NCEP R1,
FU Berlin/KIT QBO, CPC ONI, and 25,300 winters of 9 stratosphere-resolving CMIP6 models) into one small committed
file, scripts/strat/reference/ssw_precursors.json. ssw_precursors.py draws it in Actions (ssw-precursors.yml), and
re-draws the seasonal card when the QBO or ENSO inputs change. Only tested results go in; each block carries its test.

    python scripts/strat/build_ssw_precursors.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
STUDY = Path.home() / "research" / "ssw_tropical_precursors" / "data"
OUT = HERE / "reference" / "ssw_precursors.json"
RNG = np.random.default_rng(20260927)
QBO_MODELS = ["UKESM1-0-LL", "MRI-ESM2-0", "CESM2-WACCM-FV2", "CESM2-WACCM", "HadGEM3-GC31-LL"]
ENSO_EDGES = [-9, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 9]
ENSO_LABELS = ["< −1.5", "−1.5 to −1", "−1 to −0.5", "neutral", "0.5 to 1", "1 to 1.5", "1.5 to 2", "> 2"]


def J(name):
    return json.loads((STUDY / name).read_text())


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)


def fdr(p, q=0.10):
    p = np.asarray(p, float); o = np.sort(p); n = len(o)
    ok = o[o <= q * np.arange(1, n + 1) / n]
    return p <= (ok.max() if ok.size else -1)


def share_ci(x, B=2000):
    x = np.asarray(x, float)
    if len(x) < 5:
        return None
    bs = RNG.choice(x, (B, len(x))).mean(1)
    return [r3(np.percentile(bs, 2.5)), r3(np.percentile(bs, 97.5))]


def main():
    out = {"built": pd.Timestamp.now().strftime("%Y-%m-%d"),
           "source": "~/research/ssw_tropical_precursors (report 2026-09-27)"}
    # ---------------- 1. who moves first (MERRA-2 / ERA5, 28 events) ----------------
    t = J("p8_bdc_timing.json")["M2"]
    lags = np.array(t["lags"]); keep = (lags >= -60) & (lags <= 30)
    series = {}
    for key, lab, src in (("vT100|base", "100 hPa eddy heat flux, 45–75°N", "MERRA-2"),
                          ("u10|base", "Zonal wind at 60°N, 10 hPa", "MERRA-2"),
                          ("Tcap10|base", "Polar-cap temperature, 10 hPa (65–90°N)", "MERRA-2"),
                          ("dTtr_e5|base", "Tropical cooling rate, 100–50 hPa layer, 0–15°N", "ERA5"),
                          ("Ttr_e5|base", "Tropical temperature, 100–50 hPa layer, 0–15°N", "ERA5")):
        s = t["series"][key]
        series[key.split("|")[0]] = {"label": lab, "source": src, "comp": [r3(v) for v in np.array(s["comp"])[keep]],
                                     "p": [r3(v) for v in np.array(s["p"])[keep]], "onset": s["onset"], "peak": s["peak"]}
    ev = json.loads((HERE / "reference" / "strat_history_events.json").read_text())
    out["timing"] = {"lags": lags[keep].tolist(), "n_events": t["n_events"], "series": series,
                     "events": [e["date"] for e in ev["ssw"] if not e["marginal"]],
                     "test": "composite minus season-matched random dates (same calendar day +-10 d, another winter, 3,000 draws); "
                             "anomalies standardised and taken relative to each event's own day -90..-61 mean; drawn in colour "
                             "where p < 0.05; onset = first lag of a run of >= 5 days with p < 0.05",
                     "granger": "trailing 10-day tropical cooling rate given vortex, vortex change, 10- and 30-day heat flux and QBO: "
                                "odds ratio 0.99 (p 0.94) for an onset within 1-30 d, 1.00 (0.99) 10-30 d, 1.10 (0.53) 10-40 d, "
                                "1.17 (0.25) 20-50 d; winter-permutation test"}
    # ---------------- 2. the QBO and the vortex ----------------
    W = pd.read_csv(STUDY / "winters_qbo.csv")
    Wv = pd.read_csv(STUDY / "winters_enso.csv")[["winter", "oni"]]
    W = W.merge(Wv, on="winter")
    import sys
    sys.path.insert(0, str(STUDY.parent))
    from common import m2_u60, r1_u60                                       # noqa: E402
    u = pd.concat([r1_u60()[:"1979-12-31"], m2_u60()["1980-01-01":]])
    W["u_nd"] = [u[f"{y}-11-01":f"{y}-12-31"].mean() for y in W.winter]
    W["u_jf"] = [u[f"{y + 1}-01-01":f"{y + 1}-02-28"].mean() for y in W.winter]
    W = W.dropna(subset=["u_nd", "u_jf"])
    lev_rows = []
    for L in (70, 50, 40, 30, 20, 15, 10):
        rn = stats.pearsonr(W[f"u{L}"], W.u_nd); rj = stats.pearsonr(W[f"u{L}"], W.u_jf)
        lev_rows.append({"lev": L, "r_nd": r3(rn[0]), "p_nd": float(f"{rn[1]:.3g}"), "r_jf": r3(rj[0]), "p_jf": float(f"{rj[1]:.3g}")})
    for key in ("nd", "jf"):
        sig = fdr([r[f"p_{key}"] for r in lev_rows])
        for r, s in zip(lev_rows, sig):
            r[f"sig_{key}"] = bool(s)
    reg = {}
    for key in ("u_nd", "u_jf"):
        X = np.column_stack([np.ones(len(W)), W.u50]); b = np.linalg.lstsq(X, W[key], rcond=None)[0]
        res = W[key] - X @ b; s = float(res.std(ddof=2))
        reg[key] = {"a": r3(b[0]), "b": r3(b[1]), "resid_sd": r3(s), "clim": r3(W[key].mean()), "n": int(len(W)),
                    "r": r3(stats.pearsonr(W.u50, W[key])[0]), "p": float(f"{stats.pearsonr(W.u50, W[key])[1]:.3g}"),
                    "x_mean": r3(W.u50.mean()), "sxx": r3(((W.u50 - W.u50.mean()) ** 2).sum())}
    obs_pts = [{"winter": int(r.winter), "q50": r3(r.u50), "u_nd": r3(r.u_nd), "u_jf": r3(r.u_jf), "ssw": int(r.n_ssw > 0),
                "oni": r3(r.oni)} for r in W.itertuples()]
    c6 = J("p6_cmip6.json")
    ht = [{"lev": h["lev"], "e": r3(h["rate_a"]), "w": r3(h["rate_b"]), "n_e": h["n_a"], "n_w": h["n_b"], "or": r3(h["or"]),
           "p": h["p"], "agree": h["agree"], "models": h["models"], "sig": h["fdr10"]}
          for h in c6["holton_tan"]["qbo_models"] if h["lev"] != 40]
    obs_ht = J("p2_qbo_obs.json")["holton_tan_all"]
    out["qbo"] = {"levels": lev_rows, "regression_q50": reg, "winters": obs_pts,
                  "obs_share_note": {f"{h['lev']}": {"e": r3(h["E_rate"]), "w": r3(h["W_rate"]), "p": r3(h["p_fisher"])} for h in obs_ht},
                  "cmip6": ht, "cmip6_models": QBO_MODELS, "cmip6_winters": int(sum(h["n_a"] + h["n_b"] for h in c6["holton_tan"]["qbo_models"][:1])),
                  "test_obs": "Pearson r over 1958/59-2025/26 winters (Oct-Nov FU Berlin/KIT QBO; vortex = MERRA-2 u(60N,10hPa) from 1980, "
                              "NCEP R1 before); FDR 10 % over the 7 levels. The share of winters with a warming, easterly vs westerly, "
                              "is not significant after FDR (best: 40 hPa, Fisher p 0.016) and is not drawn.",
                  "test_cmip6": "Cochran-Mantel-Haenszel across model x experiment strata; FDR 10 % over the levels; models agreeing in sign"}
    # ---------------- 3. El Nino and sudden warmings, CMIP6 ----------------
    Wc = pd.read_pickle(STUDY / "cmip6_winters.pkl")
    Wc["n34z"] = Wc.groupby("model").n34.transform(lambda s: s / s.std()) * 0.95
    Wc = Wc.dropna(subset=["n34z"])
    Ec = pd.read_pickle(STUDY / "cmip6_events.pkl").merge(Wc[["model", "exp", "member", "winter", "n34z"]],
                                                         on=["model", "exp", "member", "winter"])
    cat = pd.cut(Wc.n34z, ENSO_EDGES, labels=range(8))
    dose = {}
    for name, sel in (("all", np.ones(len(Wc), bool)), ("qbo_w", Wc.model.isin(QBO_MODELS) & (Wc.q50 >= 0)),
                      ("qbo_e", Wc.model.isin(QBO_MODELS) & (Wc.q50 < 0))):
        rows = []
        for k in range(8):
            x = (Wc[sel & (cat == k)].n_ssw > 0).values.astype(float)
            rows.append({"share": r3(x.mean()) if len(x) else None, "n": int(len(x)), "ci": share_ci(x)})
        dose[name] = rows
    obs = pd.read_csv(STUDY / "winters_enso.csv")
    oc = pd.cut(obs.oni, ENSO_EDGES, labels=range(8))
    obs_bins = [{"share": r3((obs[oc == k].n_ssw > 0).mean()) if (oc == k).any() else None, "n": int((oc == k).sum())} for k in range(8)]
    Ec["cls"] = np.where(Ec.n34z >= 1.5, "strong", np.where(Ec.n34z >= 0.5, "en", np.where(Ec.n34z <= -0.5, "ln", "neutral")))
    months = {c: {str(m): r3(v) for m, v in Ec[Ec.cls == c].month.value_counts(normalize=True).reindex([12, 1, 2, 3]).items()}
              for c in ("ln", "neutral", "en", "strong")}
    months_n = {c: int((Ec.cls == c).sum()) for c in ("ln", "neutral", "en", "strong")}
    en = c6["enso"]
    oe = J("p4_enso_obs.json")
    out["enso"] = {"bins": ENSO_LABELS, "dose": dose, "obs_bins": obs_bins, "months": months, "months_n": months_n,
                   "tests": {"strong_vs_neutral": {"a": r3(en["strongEN_vs_N"]["rate_a"]), "b": r3(en["strongEN_vs_N"]["rate_b"]),
                                                   "or": r3(en["strongEN_vs_N"]["or"]), "p": en["strongEN_vs_N"]["p"],
                                                   "agree": en["strongEN_vs_N"]["agree"], "models": en["strongEN_vs_N"]["models"]},
                             "en_vs_neutral": {"a": r3(en["EN_vs_N"]["rate_a"]), "b": r3(en["EN_vs_N"]["rate_b"]), "or": r3(en["EN_vs_N"]["or"]),
                                               "p": en["EN_vs_N"]["p"], "agree": en["EN_vs_N"]["agree"], "models": en["EN_vs_N"]["models"]},
                             "late_en_vs_neutral": {"a": r3(en["timing"]["EN_vs_N"]["late_a"]), "b": r3(en["timing"]["EN_vs_N"]["late_b"]),
                                                    "p": en["timing"]["EN_vs_N"]["p"], "agree": en["timing"]["EN_vs_N"]["agree"]},
                             "late_en_vs_ln": {"a": r3(en["timing"]["EN_vs_LN"]["late_a"]), "b": r3(en["timing"]["EN_vs_LN"]["late_b"]),
                                               "p": en["timing"]["EN_vs_LN"]["p"], "agree": en["timing"]["EN_vs_LN"]["agree"]},
                             "obs_en_vs_neutral_p": r3(oe["freq_EN_vs_N"]["p_fisher"]), "obs_ln_vs_neutral_p": r3(oe["freq_LN_vs_N"]["p_fisher"])},
                   "n_winters": int(len(Wc)), "n_events": int(len(Ec))}
    # ---------------- 4. seasonal odds lookup (CMIP6 QBO models) ----------------
    Wq = Wc[Wc.model.isin(QBO_MODELS)].copy()
    cq = pd.cut(Wq.n34z, ENSO_EDGES, labels=range(8))
    Eq = Ec[Ec.model.isin(QBO_MODELS)]
    evm = Eq.groupby(["model", "exp", "member", "winter"]).month.apply(list)
    look = {}
    for ph, sel in (("W", Wq.q50 >= 0), ("E", Wq.q50 < 0)):
        rows = []
        for k in range(8):
            g = Wq[sel & (cq == k)]
            x = (g.n_ssw > 0).values.astype(float)
            keys = list(zip(g.model, g.exp, g.member, g.winter))
            dj = np.mean([any(m in (12, 1) for m in evm.get(kk, [])) for kk in keys]) if keys else np.nan
            fm = np.mean([any(m in (2, 3) for m in evm.get(kk, [])) for kk in keys]) if keys else np.nan
            rows.append({"share": r3(x.mean()) if len(x) else None, "n": int(len(x)), "ci": share_ci(x), "p_decjan": r3(dj), "p_febmar": r3(fm)})
        look[ph] = rows
    allx = (Wq.n_ssw > 0).values.astype(float)
    keys = list(zip(Wq.model, Wq.exp, Wq.member, Wq.winter))
    O = J("p11_outlook.json")
    out["odds"] = {"bins": ENSO_LABELS, "edges": ENSO_EDGES, "lookup": look,
                   "base": {"share": r3(allx.mean()), "n": int(len(allx)),
                            "p_decjan": r3(np.mean([any(m in (12, 1) for m in evm.get(k, [])) for k in keys])),
                            "p_febmar": r3(np.mean([any(m in (2, 3) for m in evm.get(k, [])) for k in keys]))},
                   "obs_base": r3(O["obs_base_rate"]), "obs_winters": 69,
                   "analogs": [{"winter": a["winter"], "oni": r3(a["oni_djf"]), "q50": a["q50_ON"], "q30": a["q30_ON"], "ssw": a["ssw"]}
                               for a in O["obs_strong_el_nino"]],
                   "models": QBO_MODELS,
                   "note": "model Nino-3.4 (DJF, 31-year running climatology removed) rescaled to the observed DJF sd of 0.95 K; "
                           "QBO = Oct-Nov mean equatorial wind at 50 hPa; Dec-Mar Charlton-Polvani warmings"}
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
