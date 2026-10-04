#!/usr/bin/env python3
"""Q1/Q2 of the 2026-09-30 SSW precursor study: is the strong vortex (and the warm tropical lower stratosphere) before
the 28 MERRA-2 major warmings real, or an artefact of comparing with dates that need not have had westerlies?

Composites of standardised daily anomalies at lags -90..+40 around the 28 central dates, "base" mode = each event (and
each null date) measured from its own lag -90..-61 mean, exactly as ssw_timing.webp. Four null families:
  N0  original: same calendar day +-10 d in another winter, no conditions (reproduces the site figure)
  N1  ELIGIBLE: same calendar day +-15 d in another MERRA-2 winter, Nov-Mar, and u(60N,10hPa) westerly on every one of
      the L_i days before it, L_i = the event's own westerly run (capped at 120 d) -> the same westerly history and,
      because L_i >= 20 after an earlier event, the same CP07 separation rule
  N2  eligible + quiet: N1 and no reversal (marginal ones included) within [d-90, d+40]
  N3  eligible + vortex-matched: N1 and the candidate's mean u10 anomaly over lags -50..-20 within 0.3 sd of the event's
      (at least the 10 nearest) -> what the tropics do at the same vortex state
3,000 Monte-Carlo composites per null; two-sided p per lag; Benjamini-Hochberg FDR 10 % over the 131 lags of each
series; and pre-registered WINDOW tests for the figure's claims (FDR over those windows).
Output: ~/research/ssw_structure/data/s1_composites.{json,npz}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import sswr_common as C

LAGS = np.arange(-90, 41)
B = 3000
CLAIMS = {  # (series, lag window, sign the figure shows)
    "u10 strong -48..-44": ("u10", (-48, -44), +1),
    "u10 strong -31..-23": ("u10", (-31, -23), +1),
    "u10o strong -48..-44": ("u10o", (-48, -44), +1),
    "u10o strong -31..-23": ("u10o", (-31, -23), +1),
    "Ttr warm -30..-17": ("Ttr", (-30, -17), +1),
    "dTtr warming -43..-39": ("dTtr", (-43, -39), +1),
    "dTtr cooling -14..-1": ("dTtr", (-14, -1), -1),
    "vT100 -10..-1": ("vT100", (-10, -1), +1),
}


def series():
    u = C.u10()
    Z = {"u10": C.std_anom(u)}
    import xarray as xr                                          # the site figure's own series: MERRA-2 u10_60n, gaps NOT filled
    uo = xr.open_dataset(C.TD / "strat_series_m2.nc").u10_60n.to_series().astype(float)
    Z["u10o"] = C.std_anom(uo[~uo.index.duplicated()])
    tt = C.ttr()
    Z["Ttr"] = C.std_anom(tt)
    Z["dTtr"] = C.std_anom(tt.diff(10).shift(-5))               # centred 10-day tendency, as on the site figure
    M = pd.read_pickle(C.DATA / "metrics_daily.pkl")
    Z["vT100"] = C.std_anom(M.vT100)
    for k in ("jet_lat", "jet_width", "depth_ratio", "top_shear", "qy_edge", "qy_pole", "bt_neg_upper", "qg_neg_upper",
              "bt_neg_flank", "n2k1", "n2k2", "n2k2_pos", "cen_lat", "aspect", "area", "u60_100", "u60_50", "u60_1"):
        s = M[k].copy()
        if k == "aspect":
            s = np.log(s.where((s >= 1) & (s < 20)))
        Z[k] = C.std_anom(s)
    return u, Z


def candidates(u, run, refr, E, rev):
    """Per event: arrays of candidate dates for N0, N1, N2."""
    days = u.index
    wd = np.array([(d - pd.Timestamp(C.winter_of(d), 7, 1)).days for d in days])     # day of the winter (Jul 1 = 0)
    win = np.array([C.winter_of(d) for d in days])
    okmonth = np.isin(days.month, C.WIN_MONTHS)
    in_m2 = (days >= "1980-08-01") & (days <= pd.Timestamp("2026-06-26") - pd.Timedelta(days=41))
    revv = np.array(rev, dtype="datetime64[ns]")
    out = []
    for e in E.itertuples():
        i = days.get_indexer([e.date])[0]
        L = int(min(run.iloc[i], 120))
        w0, wy = wd[i], e.winter
        base0 = (win != wy) & in_m2 & (win >= 1980)
        n0 = base0 & (np.abs(wd - w0) <= 10)
        n1 = base0 & okmonth & (np.abs(wd - w0) <= 15) & (run.values >= L) & ~refr.values
        dd = days.values[n1]
        quiet = np.array([not np.any((revv >= x - np.timedelta64(90, "D")) & (revv <= x + np.timedelta64(40, "D"))) for x in dd])
        n2 = np.zeros(len(days), bool); n2[np.where(n1)[0][quiet]] = True
        out.append({"date": e.date, "L": L, "N0": days[n0], "N1": days[n1], "N2": days[n2]})
    return out


def comp_test(s, dates, cand_sets, base=True, rng=C.RNG):
    """Composite (event mean minus null-mean) and per-lag p against Monte-Carlo composites drawn one candidate per
    event from that event's own candidate set."""
    M = C.lag_matrix(s, dates, LAGS)
    if base:
        M = M - np.nanmean(C.lag_matrix(s, dates, np.arange(-90, -60)), 1)[:, None]
    comp = np.nanmean(M, 0)
    nulls = np.zeros((B, len(LAGS)))
    cnt = np.zeros((B, len(LAGS)))
    for j, cs in enumerate(cand_sets):
        Mc = C.lag_matrix(s, cs, LAGS)
        if base:
            Mc = Mc - np.nanmean(C.lag_matrix(s, cs, np.arange(-90, -60)), 1)[:, None]
        pick = rng.integers(0, len(cs), B)
        v = Mc[pick]
        ok = np.isfinite(v)
        nulls += np.where(ok, v, 0); cnt += ok
    nm = nulls / np.maximum(cnt, 1)
    mu = nm.mean(0)
    p = 2 * np.minimum((nm >= comp).mean(0), (nm <= comp).mean(0))
    p = np.maximum(p, 1 / B)
    return {"comp": comp - mu, "raw": comp, "null_mean": mu, "lo": np.percentile(nm, 2.5, 0) - mu,
            "hi": np.percentile(nm, 97.5, 0) - mu, "p": p, "nm": nm - mu}


def window_p(res, w, sign):
    m = (LAGS >= w[0]) & (LAGS <= w[1])
    v = res["comp"][m].mean()
    nv = res["nm"][:, m].mean(1)
    p = 2 * min((nv >= v).mean(), (nv <= v).mean()); p = max(p, 1 / B)
    return float(v), float(p), bool(np.sign(v) == sign)


def main():
    u, Z = series()
    E = C.events()
    run = C.westerly_run(u)
    rev = C.all_reversals()
    refr = C.refractory(u, rev)
    cs = candidates(u, run, refr, E, rev)
    E["L"] = [c["L"] for c in cs]
    print("events: westerly run before onset (days):", E.L.tolist())
    print("candidates per event N0/N1/N2 (median):", int(np.median([len(c["N0"]) for c in cs])),
          int(np.median([len(c["N1"]) for c in cs])), int(np.median([len(c["N2"]) for c in cs])))
    # N3: vortex-matched within N1
    uw = lambda d: np.nanmean(C.lag_matrix(Z["u10"], d, np.arange(-50, -19)) -
                              np.nanmean(C.lag_matrix(Z["u10"], d, np.arange(-90, -60)), 1)[:, None], 1)
    ev_u = uw(E.date)
    for c, eu in zip(cs, ev_u):
        cu = uw(c["N1"])
        ok = np.abs(cu - eu) <= 0.3
        if ok.sum() < 10:
            ok = np.zeros(len(cu), bool); ok[np.argsort(np.abs(cu - eu))[:10]] = True
        c["N3"] = c["N1"][ok]
    print("N3 candidates (median):", int(np.median([len(c["N3"]) for c in cs])))
    res, js = {}, {"lags": LAGS.tolist(), "n_events": len(E), "L": E.L.tolist(), "claims": {}, "fdr_runs": {}, "p05_runs": {}}
    npz = {}
    keys = list(Z)
    for null in ("N0", "N1", "N2", "N3"):
        for k in keys:
            if null in ("N2", "N3") and k not in ("u10", "u10o", "Ttr", "dTtr", "vT100"):
                continue
            for base in (True, False):
                if not base and k not in ("u10", "u10o", "Ttr", "dTtr", "vT100"):
                    continue
                tag = f"{k}|{null}|{'base' if base else 'anom'}"
                r = comp_test(Z[k], E.date, [c[null] for c in cs], base=base)
                res[tag] = r
                sig = C.fdr_bh(r["p"], 0.10)
                js["fdr_runs"][tag] = C.runs_of(sig, LAGS, 1)
                js["p05_runs"][tag] = C.runs_of(r["p"] < 0.05, LAGS, 3)
                npz[tag + "|comp"] = r["comp"]; npz[tag + "|p"] = r["p"]; npz[tag + "|lo"] = r["lo"]; npz[tag + "|hi"] = r["hi"]
                npz[tag + "|fdr"] = sig
        print(f"{null} done", flush=True)
    # window claims
    for null in ("N0", "N1", "N2", "N3"):
        rows = {}
        for name, (k, w, sg) in CLAIMS.items():
            tag = f"{k}|{null}|base"
            v, p, same = window_p(res[tag], w, sg)
            rows[name] = {"value": round(v, 3), "p": round(p, 4), "same_sign": same}
        sig = C.fdr_bh([r["p"] for r in rows.values()], 0.10)
        for (name, r), s in zip(rows.items(), sig):
            r["fdr10"] = bool(s)
        js["claims"][null] = rows
        print(null, {k: (v["value"], v["p"], v["fdr10"]) for k, v in rows.items()})
    # Q2: tropical T residual after the vortex state (all Nov-Apr days, 1980-2026)
    d = pd.DataFrame({"T": Z["Ttr"], "u": Z["u10"], "hf": Z["vT100"]})
    for L in (10, 20, 30, 45, 60):
        d[f"u_m{L}"] = Z["u10"].rolling(L, min_periods=int(0.8 * L)).mean()
    d["hf40"] = Z["vT100"].rolling(40, min_periods=30).mean()
    d["hf20"] = Z["vT100"].rolling(20, min_periods=15).mean()
    fitm = d.index.month.isin([10, 11, 12, 1, 2, 3, 4]) & (d.index.year >= 1980)
    cols = ["u", "u_m10", "u_m20", "u_m30", "u_m45", "u_m60", "hf20", "hf40"]
    dd = d[fitm].dropna()
    X = np.column_stack([np.ones(len(dd))] + [dd[c] for c in cols])
    b = np.linalg.lstsq(X, dd["T"], rcond=None)[0]
    pred = pd.Series(np.column_stack([np.ones(len(d))] + [d[c] for c in cols]) @ b, index=d.index)
    r2 = 1 - np.var(dd["T"] - X @ b) / np.var(dd["T"])
    js["ttr_regression"] = {"cols": cols, "coef": np.round(b, 3).tolist(), "r2": round(float(r2), 3), "n_days": int(len(dd))}
    print(f"Ttr ~ vortex history + heat flux, Oct-Apr: R2 {r2:.3f}; coef {np.round(b, 2)}")
    Z["Ttr_resid"] = (Z["Ttr"] - pred).where(pred.notna())
    for null in ("N1", "N3"):
        tag = f"Ttr_resid|{null}|base"
        r = comp_test(Z["Ttr_resid"], E.date, [c[null] for c in cs], base=True)
        res[tag] = r
        for kk in ("comp", "p", "lo", "hi"):
            npz[f"{tag}|{kk}"] = r[kk]
        npz[f"{tag}|fdr"] = C.fdr_bh(r["p"], 0.10)
        js["fdr_runs"][tag] = C.runs_of(npz[f"{tag}|fdr"], LAGS, 1)
        js["p05_runs"][tag] = C.runs_of(r["p"] < 0.05, LAGS, 3)
        v, p, _ = window_p(r, (-30, -17), +1)
        js["claims"][f"Ttr_resid {null} -30..-17"] = {"value": round(v, 3), "p": round(p, 4)}
        print(f"Ttr residual {null} -30..-17: {v:+.3f} p {p:.3f}")
    # lagged all-days correlation Ttr(t) vs u10(t-L), Nov-Mar
    lc = []
    for L in range(-60, 61, 2):
        x = pd.concat([Z["u10"].shift(L), Z["Ttr"]], axis=1).dropna()
        x = x[x.index.month.isin(C.WIN_MONTHS) & (x.index.year >= 1980)]
        lc.append((L, float(x.corr().iloc[0, 1])))
    js["lagcorr_u_leads_T"] = lc
    # the background of SSW winters: QBO, RONI, and the tropical T with QBO + RONI regressed out (anomaly mode)
    q, rn = C.qbo50(), C.roni()
    Z["qbo50"] = ((q - q.mean()) / q.std()).reindex(u.index)
    Z["roni"] = rn.reindex(u.index)
    dq = pd.DataFrame({"T": Z["Ttr"], "q": Z["qbo50"], "q2": C.qbo50().shift(-90).reindex(u.index), "r": Z["roni"]})
    dq["q2"] = (dq.q2 - dq.q2.mean()) / dq.q2.std()
    fm = dq.index.month.isin([10, 11, 12, 1, 2, 3, 4]) & (dq.index.year >= 1980)
    dd = dq[fm].dropna()
    X = np.column_stack([np.ones(len(dd)), dd.q, dd.r])
    bq = np.linalg.lstsq(X, dd["T"], rcond=None)[0]
    r2q = 1 - np.var(dd["T"] - X @ bq) / np.var(dd["T"])
    Z["Ttr_noqe"] = Z["Ttr"] - (bq[0] + bq[1] * Z["qbo50"] + bq[2] * Z["roni"])
    js["ttr_qbo_enso"] = {"coef": np.round(bq, 3).tolist(), "r2": round(float(r2q), 3)}
    print(f"Ttr ~ QBO50 + RONI, Oct-Apr: R2 {r2q:.3f}, coef {np.round(bq, 2)}")
    for k in ("qbo50", "roni", "Ttr_noqe"):
        for base in (False, True):
            tag = f"{k}|N1|{'base' if base else 'anom'}"
            r = comp_test(Z[k], E.date, [c["N1"] for c in cs], base=base)
            for kk in ("comp", "p", "lo", "hi"):
                npz[f"{tag}|{kk}"] = r[kk]
            npz[f"{tag}|fdr"] = C.fdr_bh(r["p"], 0.10)
            js["fdr_runs"][tag] = C.runs_of(npz[f"{tag}|fdr"], LAGS, 1)
            js["p05_runs"][tag] = C.runs_of(r["p"] < 0.05, LAGS, 3)
            for w in ((-90, -61), (-48, -23), (-30, -17)):
                v, p, _ = window_p(r, w, +1)
                js["claims"][f"{tag} {w[0]}..{w[1]}"] = {"value": round(v, 3), "p": round(p, 4)}
            print(f"{tag:20s} -90..-61 / -48..-23 / -30..-17:",
                  [js['claims'][f'{tag} {a}..{b}'] for a, b in ((-90, -61), (-48, -23), (-30, -17))])
    # anomaly-mode window values for the main series (absolute state vs eligible dates)
    for k in ("u10", "u10o", "Ttr", "vT100"):
        for null in ("N0", "N1"):
            r = res[f"{k}|{null}|anom"]
            for w in ((-90, -61), (-48, -44), (-31, -23), (-30, -17)):
                v, p, _ = window_p(r, w, +1)
                js["claims"][f"{k}|{null}|anom {w[0]}..{w[1]}"] = {"value": round(v, 3), "p": round(p, 4)}
    for k in ("u10", "Ttr"):
        print(k, "anom N1 windows:", {kk.split(" ")[-1]: v for kk, v in js["claims"].items() if kk.startswith(f"{k}|N1|anom")})
    print("corr(u10(t-L), Ttr(t)) at L=0/20/40:", [round(r, 2) for L, r in lc if L in (0, 20, 40)])
    js["events"] = [{"date": str(e.date.date()), "type": e.type, "depth": e.depth, "L": int(e.L)} for e in E.itertuples()]
    js["n_candidates"] = {n: [int(len(c[n])) for c in cs] for n in ("N0", "N1", "N2", "N3")}
    (C.DATA / "s1_composites.json").write_text(json.dumps(js, indent=1, default=float))
    np.savez_compressed(C.DATA / "s1_composites.npz", **npz)
    for tag in ("u10o|N0|base", "u10o|N1|base", "u10o|N3|base", "u10|N0|base", "u10|N1|base", "u10|N2|base", "Ttr|N0|base", "Ttr|N1|base", "Ttr|N3|base", "dTtr|N0|base",
                "dTtr|N1|base", "u10|N1|anom", "Ttr|N1|anom", "Ttr_resid|N1|base"):
        print(f"{tag:22s} p<.05 runs {js['p05_runs'][tag]}  FDR runs {js['fdr_runs'][tag]}")


if __name__ == "__main__":
    main()
