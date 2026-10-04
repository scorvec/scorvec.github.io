#!/usr/bin/env python3
"""
US credit conditions index (2026-10-04, user: "a financial conditions index that mainly focuses on how tight credit is in
the US economy right now ... credit spreads, mortgage rates, 2s10s, fed funds ... let's also look at the corporate borrowing
sector"; decisions: public page, FIRST PRINCIPAL COMPONENT weights, free data only).

Weekly (Friday) index from FRED (fredgraph.csv, no key). Every input is oriented so that HIGHER = TIGHTER, standardised
over the fit window, and the index is the first principal component of the standardised inputs, signed so the Baa spread
loads positively and scaled to unit standard deviation (0 = average conditions 2001-present, +1 = one sd tighter).

Real-time discipline: each input enters a week only once it would have been published (RELEASE_LAG_D); a quarterly or
monthly value is carried forward until the next release. Daily series are averaged over the week.

The ICE BofA option-adjusted spreads (IG / BBB / HY) are on FRED only from Oct 2023 (licensing): too short to estimate
weights, so they are shown beside the index ("today's market spreads"), not inside it. Moody's Baa/Aaa minus the 10-year
carry the corporate-spread signal back to the 1980s.

Fed funds and the mortgage rate enter in REAL terms (minus trailing core PCE inflation) - in nominal levels their secular
decline since the 1980s would become the first component by itself.

  python scripts/fci/fci_build.py [--out assets/fci] [--cache scripts/fci/data]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FIT_START = "2001-01-05"        # every input exists from here (CP outstanding starts 2001-01)
SHOW_START = "1990-01-05"       # the index is shown from here, on the inputs available each week

# input key -> (label, group, description); all oriented HIGHER = TIGHTER
INPUTS = {
    "baa":        ("Baa corporate spread", "Corporate bonds", "Moody's Baa yield minus the 10-year Treasury (BAA10Y)"),
    "aaa":        ("Aaa corporate spread", "Corporate bonds", "Moody's Aaa yield minus the 10-year Treasury (AAA10Y)"),
    "cp_nonfin":  ("Commercial paper spread, nonfinancial", "Corporate short-term funding", "3-month AA nonfinancial CP minus the 3-month T-bill"),
    "cp_fin":     ("Commercial paper spread, financial", "Corporate short-term funding", "3-month AA financial CP minus the 3-month T-bill"),
    "cp_out":     ("Commercial paper outstanding (slower growth = tighter)", "Corporate short-term funding", "minus the year-on-year % change in CP outstanding (COMPOUT)"),
    "ci_loans":   ("Bank business loans (slower growth = tighter)", "Bank lending", "minus the year-on-year % change in commercial & industrial loans (TOTCI)"),
    "sloos_lg":   ("Banks tightening business-loan standards, large & mid firms", "Bank lending", "Senior Loan Officer Survey, net % tightening (DRTSCILM)"),
    "sloos_sm":   ("Banks tightening business-loan standards, small firms", "Bank lending", "Senior Loan Officer Survey, net % tightening (DRTSCIS)"),
    "sloos_dem":  ("Weaker business-loan demand", "Bank lending", "minus the Senior Loan Officer Survey net % reporting stronger demand (DRSDCILM)"),
    "corp_debt":  ("Nonfinancial corporate debt (slower growth = tighter)", "Corporate bonds", "minus the year-on-year % change in nonfinancial corporate debt securities and loans (Fed Z.1, BCNSDODNS)"),
    "real_ff":    ("Real fed funds rate", "Policy and rates", "effective fed funds minus core PCE inflation over the past year"),
    "curve":      ("Flatter yield curve (2s10s, inverted)", "Policy and rates", "minus the 10-year minus 2-year Treasury spread (T10Y2Y)"),
    "real_mort":  ("Real 30-year mortgage rate", "Mortgages", "Freddie Mac 30-year fixed rate minus core PCE inflation over the past year"),
    "mort_sprd":  ("Mortgage rate spread to the 10-year", "Mortgages", "Freddie Mac 30-year fixed rate minus the 10-year Treasury"),
}
# days from a value's date stamp to its publication (conservative)
RELEASE_LAG_D = {"TOTCI": 9, "COMPOUT": 2, "PCEPILFE": 58, "DRTSCILM": 50, "DRTSCIS": 50, "DRSDCILM": 50, "BCNSDODNS": 160,
                 "MORTGAGE30US": 0}
CARRY_W = {"sloos_lg": 20, "sloos_sm": 20, "sloos_dem": 20, "corp_debt": 30, "ci_loans": 4, "cp_out": 3, "real_mort": 2}
SERIES = ["DFF", "T10Y2Y", "DGS10", "DTB3", "MORTGAGE30US", "BAA10Y", "AAA10Y", "DCPN3M", "DCPF3M", "COMPOUT", "TOTCI",
          "DRTSCILM", "DRTSCIS", "DRSDCILM", "BCNSDODNS", "PCEPILFE", "NFCI", "NFCICREDIT",
          "BAMLC0A0CM", "BAMLC0A4CBBB", "BAMLH0A0HYM2", "USREC"]
# sub-indices (2026-10-04, user chose "PCA + rate sub-indices"): the PCA headline is the corporate credit cycle (real fed
# funds, loan growth and the curve barely load, the curve with a recession-steepening sign), so three transparent
# equal-weight averages of the oriented z-scores sit beside it and show where the tightness is
SUBS = {"corporate": ("Corporate credit", ["baa", "aaa", "cp_nonfin", "cp_fin", "cp_out", "ci_loans", "sloos_lg", "sloos_sm",
                                           "sloos_dem", "corp_debt"]),
        "mortgage": ("Mortgages", ["real_mort", "mort_sprd"]),
        "policy": ("Policy rate and yield curve", ["real_ff", "curve"])}
SHOW_ONLY = {"BAMLC0A0CM": "Investment-grade spread (ICE BofA OAS)", "BAMLC0A4CBBB": "BBB spread (ICE BofA OAS)",
             "BAMLH0A0HYM2": "High-yield spread (ICE BofA OAS)"}


def fetch(sid, cache: Path, max_age_h=6.0):
    """One FRED series as a float Series indexed by date (cached; a failed refresh falls back to the cache)."""
    fp = cache / f"{sid}.csv"
    if not fp.exists() or time.time() - fp.stat().st_mtime > max_age_h * 3600:
        for attempt in range(4):
            try:
                req = urllib.request.Request(FRED.format(sid), headers={"User-Agent": "scorvec-fci/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    txt = r.read().decode()
                if not txt.startswith("observation_date"):
                    raise ValueError("not a FRED csv")
                fp.write_text(txt)
                break
            except Exception as e:                                   # noqa: BLE001
                if attempt == 3:
                    if not fp.exists():
                        raise
                    print(f"  {sid}: refresh failed ({e}); using the cache", flush=True)
                time.sleep(2 ** attempt)
    d = pd.read_csv(fp)
    s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(d.iloc[:, 0])
    return s.dropna()


def weekly(s, how="mean", lag_d=0):
    """To the Friday grid. Daily: the mean of the week; sparser series: the latest value PUBLISHED by that Friday."""
    if lag_d:
        s = s.copy(); s.index = s.index + pd.Timedelta(days=lag_d)
    if how == "mean":
        return s.resample("W-FRI").mean()
    return s.resample("W-FRI").last().ffill()


def yoy(s):
    """Year-on-year % change of a level series at its own frequency."""
    s = s.sort_index()
    prev = s.reindex(s.index - pd.DateOffset(years=1), method="nearest", tolerance=pd.Timedelta(days=10))
    return pd.Series(100.0 * (s.values / prev.values - 1.0), index=s.index)


def build_inputs(raw):
    W = {}
    core = yoy(raw["PCEPILFE"])
    core_w = weekly(core, "last", RELEASE_LAG_D["PCEPILFE"])
    W["baa"] = weekly(raw["BAA10Y"])
    W["aaa"] = weekly(raw["AAA10Y"])
    W["cp_nonfin"] = weekly(raw["DCPN3M"] - raw["DTB3"].reindex(raw["DCPN3M"].index))
    W["cp_fin"] = weekly(raw["DCPF3M"] - raw["DTB3"].reindex(raw["DCPF3M"].index))
    W["cp_out"] = -weekly(yoy(raw["COMPOUT"]), "last", RELEASE_LAG_D["COMPOUT"])
    W["ci_loans"] = -weekly(yoy(raw["TOTCI"]), "last", RELEASE_LAG_D["TOTCI"])
    W["sloos_lg"] = weekly(raw["DRTSCILM"], "last", RELEASE_LAG_D["DRTSCILM"])
    W["sloos_sm"] = weekly(raw["DRTSCIS"], "last", RELEASE_LAG_D["DRTSCIS"])
    W["sloos_dem"] = -weekly(raw["DRSDCILM"], "last", RELEASE_LAG_D["DRSDCILM"])
    W["corp_debt"] = -weekly(yoy(raw["BCNSDODNS"]), "last", RELEASE_LAG_D["BCNSDODNS"])
    ff = weekly(raw["DFF"])
    W["real_ff"] = ff - core_w.reindex(ff.index).ffill()
    W["curve"] = -weekly(raw["T10Y2Y"])
    mort = weekly(raw["MORTGAGE30US"], "last", 0)
    W["real_mort"] = mort - core_w.reindex(mort.index).ffill()
    W["mort_sprd"] = mort - weekly(raw["DGS10"]).reindex(mort.index)
    X = pd.DataFrame(W)
    # every input carries forward to the latest Friday until its next release, but never further than its own cadence
    # allows (a discontinued series must not freeze into the index): weeks
    grid = pd.date_range(X.index.min(), max(s.index.max() for s in W.values()), freq="W-FRI")
    X = X.reindex(grid)
    for k in X.columns:
        X[k] = X[k].ffill(limit=CARRY_W.get(k, 2))
    return X[X.index >= pd.Timestamp(SHOW_START)]


def fit_pca(X, start=FIT_START):
    """Standardise over the fit window, first principal component of the correlation matrix; returns
    (mu, sd, loadings, explained share, scale) with the sign set so the Baa spread loads positively."""
    F = X[X.index >= pd.Timestamp(start)].dropna()
    mu, sd = F.mean(), F.std(ddof=0)
    Z = (F - mu) / sd
    C = np.corrcoef(Z.values.T)
    ev, evec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]; ev, evec = ev[o], evec[:, o]
    w = pd.Series(evec[:, 0], index=X.columns)
    if w["baa"] < 0:
        w = -w
    pc = Z.values @ w.values
    scale = pc.std(ddof=0)
    return mu, sd, w, ev / ev.sum(), scale, len(F)


def index_from(X, mu, sd, w, scale):
    """Index each week from the inputs available that week: the missing loadings' share is removed and the score
    rescaled by the loadings present (so a week with fewer inputs is not pulled toward zero)."""
    Z = (X - mu) / sd
    avail = Z.notna()
    num = (Z.fillna(0.0) * w).sum(axis=1)
    norm = np.sqrt((avail * w ** 2).sum(axis=1))
    full = np.sqrt((w ** 2).sum())
    idx = num * (full / norm.replace(0, np.nan)) / scale
    cover = (avail * w.abs()).sum(axis=1) / w.abs().sum()
    contrib = (Z * w).div(scale).mul(full / norm.replace(0, np.nan), axis=0)
    return idx, cover, contrib, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/fci")
    ap.add_argument("--cache", default="scripts/fci/data")
    a = ap.parse_args()
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw = {s: fetch(s, cache) for s in SERIES}
    X = build_inputs(raw)
    mu, sd, w, share, scale, nfit = fit_pca(X)
    idx, cover, contrib, Z = index_from(X, mu, sd, w, scale)
    keep = cover >= 0.5
    idx = idx[keep]; cover = cover[keep]; contrib = contrib[keep]; Z = Z[keep]; X = X[keep]
    last = idx.dropna().index[-1]
    # benchmarks: the Chicago Fed's NFCI and its credit subindex, weekly (Friday-dated), same orientation (+ = tighter)
    bench = {k: weekly(raw[k], "last").reindex(idx.index) for k in ("NFCI", "NFCICREDIT")}
    corr = {k: round(float(pd.concat([idx, v], axis=1).dropna().corr().iloc[0, 1]), 3) for k, v in bench.items()}
    pct = float((idx.dropna() <= idx[last]).mean())
    subs = {}
    for sk, (slab, keys) in SUBS.items():
        v = Z[keys].mean(axis=1, skipna=True)
        v = v.where(Z[keys].notna().sum(axis=1) >= max(1, len(keys) // 2))
        vv = v.dropna()
        subs[sk] = dict(label=slab, inputs=keys, latest=round(float(vv.iloc[-1]), 3), pct_rank=round(float((vv <= vv.iloc[-1]).mean()), 3),
                        chg_13w=round(float(vv.iloc[-1] - vv.iloc[-14]), 3), series=[[str(d.date()), round(float(x), 3)] for d, x in vv.items()])
    rec = raw["USREC"]; rec = rec[rec.index >= pd.Timestamp(SHOW_START) - pd.DateOffset(months=1)]
    spans, st_ = [], None                                         # NBER recessions as [start, end] month spans
    for d_, v_ in rec.items():
        if v_ >= 1 and st_ is None: st_ = d_
        if v_ < 1 and st_ is not None: spans.append([str(st_.date()), str(d_.date())]); st_ = None
    if st_ is not None: spans.append([str(st_.date()), str(rec.index[-1].date())])
    # group contributions today
    groups = {}
    for k, (lab, g, desc) in INPUTS.items():
        groups.setdefault(g, 0.0)
        v = contrib.loc[last, k]
        if np.isfinite(v):
            groups[g] += float(v)
    inputs = []
    for k, (lab, g, desc) in INPUTS.items():
        s = X[k].dropna()
        lv = s.iloc[-1] if len(s) else np.nan
        z = Z.loc[last, k]
        prev = Z[k].dropna()
        z13 = prev.iloc[-14] if len(prev) > 14 else np.nan
        inputs.append(dict(key=k, label=lab, group=g, desc=desc, loading=round(float(w[k]), 3),
                           value=None if not np.isfinite(lv) else round(float(lv), 3), asof=str(s.index[-1].date()) if len(s) else None,
                           z=None if not np.isfinite(z) else round(float(z), 2),
                           z_13w_ago=None if not np.isfinite(z13) else round(float(z13), 2),
                           contribution=None if not np.isfinite(contrib.loc[last, k]) else round(float(contrib.loc[last, k]), 3),
                           pct_rank=None if not len(s) else round(float((Z[k].dropna() <= z).mean()), 3) if np.isfinite(z) else None))
    show = {}
    for sid, lab in SHOW_ONLY.items():
        s = raw[sid]
        show[sid] = dict(label=lab, last=round(float(s.iloc[-1]), 2), asof=str(s.index[-1].date()),
                         min_3y=round(float(s.min()), 2), max_3y=round(float(s.max()), 2),
                         pct_3y=round(float((s <= s.iloc[-1]).mean()), 3),
                         series=[[str(d.date()), round(float(v), 3)] for d, v in s.resample("W-FRI").last().dropna().items()])
    ser = lambda s, nd=3: [[str(d.date()), None if not np.isfinite(v) else round(float(v), nd)] for d, v in s.items()]
    js = dict(made=pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"), asof=str(last.date()),
              latest=round(float(idx[last]), 3), pct_rank=round(pct, 3),
              chg_4w=round(float(idx[last] - idx.dropna().iloc[-5]), 3), chg_13w=round(float(idx[last] - idx.dropna().iloc[-14]), 3),
              chg_52w=round(float(idx[last] - idx.dropna().iloc[-53]), 3),
              fit=dict(start=FIT_START, weeks=int(nfit), pc1_share=round(float(share[0]), 3), pc2_share=round(float(share[1]), 3)),
              subs=subs, recessions=spans, corr=corr, groups={g: round(v, 3) for g, v in groups.items()}, inputs=inputs, show_only=show,
              index=ser(idx), coverage=ser(cover, 2), bench={k: ser(v) for k, v in bench.items()},
              contrib={g: ser(contrib[[k for k, x in INPUTS.items() if x[1] == g]].sum(axis=1, min_count=1)) for g in groups})
    (out / "fci.json").write_text(json.dumps(js, separators=(",", ":")))
    print(f"FCI {last.date()}: {idx[last]:+.2f} sd (pct {pct:.0%}); PC1 {share[0]:.0%} of variance over {nfit} weeks; "
          f"corr NFCI {corr['NFCI']}, NFCI credit {corr['NFCICREDIT']}")
    print("loadings: " + ", ".join(f"{k} {w[k]:+.2f}" for k in w.sort_values(ascending=False).index))
    print("groups now: " + ", ".join(f"{g} {v:+.2f}" for g, v in groups.items()))
    print("sub-indices: " + ", ".join(f"{v['label']} {v['latest']:+.2f} (pct {v['pct_rank']:.0%}, 13w {v['chg_13w']:+.2f})" for v in subs.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
