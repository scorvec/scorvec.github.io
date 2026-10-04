#!/usr/bin/env python3
"""Q4 of the 2026-09-30 SSW precursor study: is a fast-spinning vortex FRAGILE because of its structure?

(a) Structure-metric composites around the 28 central dates, absolute anomalies, against the ELIGIBLE null (N1 of
    s1_composites: same calendar window +-15 d, other winters, the event's own westerly history). FDR 10 % over lags.
(b) Splits vs displacements: metric means over days -42..-21 and -20..-1, label-permutation test (10,000), FDR over
    metrics x windows.
(c) Strong-vortex spells: every run of >= 10 consecutive Nov-Feb days with u(60N,10hPa) >= +1 sd. A spell is
    "followed" if a confirmed central date falls within 63 days of its last day. Spell-mean structure anomalies,
    followed vs not, permutation test over spells, FDR across metrics. Named cases (2010/11, 2019/20 and the
    followed spells) tabulated as percentiles of all strong-vortex days.
Output: ~/research/ssw_structure/data/s3_structure.{json,npz}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import sswr_common as C
from s1_composites import LAGS, candidates, comp_test, series
from s2_forward import STRUCT

NPERM = 10000


def main():
    rng = np.random.default_rng(20260930)
    u, Z = series()
    E = C.events()
    run = C.westerly_run(u)
    rev = C.all_reversals()
    refr = C.refractory(u, rev)
    cs = candidates(u, run, refr, E, rev)
    out, npz = {"lags": LAGS.tolist(), "composites": {}, "split_disp": {}, "spells": {}}, {}
    keys = ["u10", "vT100"] + [k for k in STRUCT if k in Z] + ["u60_50", "u60_1"]
    for k in keys:
        r = comp_test(Z[k], E.date, [c["N1"] for c in cs], base=False, rng=rng)
        sig = C.fdr_bh(r["p"], 0.10)
        for kk in ("comp", "p", "lo", "hi"):
            npz[f"{k}|{kk}"] = r[kk]
        npz[f"{k}|fdr"] = sig
        out["composites"][k] = {"fdr_runs": C.runs_of(sig, LAGS, 1), "p05_runs": C.runs_of(r["p"] < 0.05, LAGS, 3),
                                "mean_-60..-21": float(r["comp"][(LAGS >= -60) & (LAGS <= -21)].mean())}
        print(f"{k:13s} FDR runs {out['composites'][k]['fdr_runs']}", flush=True)
    # ---------------- (b) splits vs displacements
    isplit = (E.type == "split").values
    rows = {}
    for k in keys:
        for (a, b) in ((-42, -21), (-20, -1), (0, 10)):
            M = C.lag_matrix(Z[k], E.date, np.arange(a, b + 1))
            v = np.nanmean(M, 1)
            ok = np.isfinite(v)
            dlt = v[ok & isplit].mean() - v[ok & ~isplit].mean()
            lab = isplit[ok]; vv = v[ok]
            null = np.empty(NPERM)
            for i in range(NPERM):
                pl = rng.permutation(lab)
                null[i] = vv[pl].mean() - vv[~pl].mean()
            p = max((np.abs(null) >= abs(dlt)).mean(), 1 / NPERM)
            rows[f"{k}|{a}..{b}"] = {"split": float(v[ok & isplit].mean()), "disp": float(v[ok & ~isplit].mean()),
                                     "diff": float(dlt), "p": float(p), "n_split": int((ok & isplit).sum()),
                                     "n_disp": int((ok & ~isplit).sum())}
    sig = C.fdr_bh([r["p"] for r in rows.values()], 0.10)
    for (kk, r), s in zip(rows.items(), sig):
        r["fdr10"] = bool(s)
    out["split_disp"] = rows
    print("split vs displacement, p < 0.05:", {k: (round(r["diff"], 2), round(r["p"], 3), r["fdr10"]) for k, r in rows.items() if r["p"] < 0.05})
    # ---------------- (c) strong-vortex spells
    U = Z["u10"]
    ok = U.index.month.isin([11, 12, 1, 2]) & (U.index >= "1980-11-01") & (U.index <= "2026-02-28")
    strong = (U >= 1.0) & ok
    spells, start = [], None
    idx = U.index
    for i in range(len(idx)):
        if strong.iloc[i] and start is None:
            start = i
        if (not strong.iloc[i] or i == len(idx) - 1) and start is not None:
            end = i - 1 if not strong.iloc[i] else i
            if end - start + 1 >= 10:
                spells.append((idx[start], idx[end]))
            start = None
    ev = pd.DatetimeIndex(E.date)
    S = []
    for a, b in spells:
        nxt = ev[ev > b]
        lead = (nxt[0] - b).days if len(nxt) else 9999
        row = {"start": a, "end": b, "len": (b - a).days + 1, "winter": C.winter_of(a), "lead_to_ssw": lead,
               "followed": lead <= 63, "next_ssw": str(nxt[0].date()) if len(nxt) and lead <= 63 else "",
               "type": E.set_index("date").type.get(nxt[0], "") if len(nxt) and lead <= 63 else "",
               "U_mean": float(U[a:b].mean()), "U_max": float(U[a:b].max())}
        for k in keys:
            row[k] = float(Z[k][a:b].mean())
        S.append(row)
    S = pd.DataFrame(S)
    # a spell that ends within 63 days of the data end with no SSW is censored only in 2025/26 (has an SSW): none dropped
    print(f"{len(S)} strong spells ({int(S.followed.sum())} followed by an SSW within 63 d) in {S.winter.nunique()} winters")
    res = {}
    for k in keys:
        x = S[k].values; f = S.followed.values; okk = np.isfinite(x)
        dlt = x[okk & f].mean() - x[okk & ~f].mean()
        null = np.empty(NPERM)
        lab = f[okk]; xv = x[okk]
        for i in range(NPERM):
            pl = rng.permutation(lab)
            null[i] = xv[pl].mean() - xv[~pl].mean()
        p = max((np.abs(null) >= abs(dlt)).mean(), 1 / NPERM)
        res[k] = {"followed": float(x[okk & f].mean()), "not": float(x[okk & ~f].mean()), "diff": float(dlt), "p": float(p),
                  "auc": float(C.auc(lab, xv))}
    sig = C.fdr_bh([r["p"] for r in res.values()], 0.10)
    for (k, r), s in zip(res.items(), sig):
        r["fdr10"] = bool(s)
    out["spells"]["test"] = res
    print("spells followed vs not:", {k: (round(r["diff"], 2), round(r["p"], 3), r["fdr10"]) for k, r in res.items()})
    S2 = S.copy(); S2["start"] = S2.start.dt.strftime("%Y-%m-%d"); S2["end"] = S2.end.dt.strftime("%Y-%m-%d")
    out["spells"]["table"] = S2.round(3).to_dict(orient="records")
    # named case percentiles vs all strong-vortex days (Nov-Feb, U >= 1)
    sd = pd.DataFrame({k: Z[k] for k in keys})[strong]
    cases = {"2010/11 (no SSW)": ("2010-12-15", "2011-02-28"), "2019/20 (no SSW)": ("2019-12-15", "2020-02-29")}
    for r in S[S.followed].itertuples():
        cases[f"before {r.next_ssw} ({r.type})"] = (str(r.start.date()), str(r.end.date()))
    ct = {}
    for name, (a, b) in cases.items():
        seg = sd[a:b]
        if not len(seg):
            continue
        ct[name] = {"days": int(len(seg)), **{k: float((sd[k] < seg[k].mean()).mean()) for k in keys}}
    out["spells"]["cases_pct"] = ct
    (C.DATA / "s3_structure.json").write_text(json.dumps(out, indent=1, default=str))
    np.savez_compressed(C.DATA / "s3_structure.npz", **npz)
    S.to_pickle(C.DATA / "s3_spells.pkl")


if __name__ == "__main__":
    main()
