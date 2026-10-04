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
    # 2026-10-04 (user: "What about the stock market, the TW-USD and things like the VIX?" -> "In the headline"): with them
    # the index catches market-led tightenings the credit inputs see late (peaks 2018Q4 -0.31 -> +0.17, 2022 -0.05 -> +0.70,
    # Apr 2025 -0.11 -> +0.31), PC1 28 -> 31 %, corr NFCI 0.75 -> 0.79, NFCI credit 0.75 -> 0.78; today unchanged
    "vix":        ("Stock-market volatility (VIX)", "Markets", "CBOE VIX, weekly average (VIXCLS)"),
    "usd":        ("Stronger dollar", "Markets", "year-on-year % change in the Fed's broad trade-weighted dollar (DTWEXBGS, spliced to TWEXB before 2006)"),
}
# days from a value's date stamp to its publication (conservative)
RELEASE_LAG_D = {"TOTCI": 9, "COMPOUT": 2, "PCEPILFE": 58, "DRTSCILM": 50, "DRTSCIS": 50, "DRSDCILM": 50, "BCNSDODNS": 160,
                 "MORTGAGE30US": 0}
CARRY_W = {"usd": 3, "sloos_lg": 20, "sloos_sm": 20, "sloos_dem": 20, "corp_debt": 30, "ci_loans": 4, "cp_out": 3, "real_mort": 2}
SERIES = ["DFF", "T10Y2Y", "DGS10", "DTB3", "MORTGAGE30US", "BAA10Y", "AAA10Y", "DCPN3M", "DCPF3M", "COMPOUT", "TOTCI",
          "DRTSCILM", "DRTSCIS", "DRSDCILM", "BCNSDODNS", "PCEPILFE", "NFCI", "NFCICREDIT",
          "USREC", "VIXCLS", "DTWEXBGS", "TWEXB", "LNFACBW027SBOG"]
# DATA TERMS (2026-10-04, user: "Always make sure we are sourcing and attributing all data sources correctly and collecting
# the data in an appropriate manner"): FRED marks the ICE BofA OAS series (BAMLC0A0CM, BAMLC0A4CBBB, BAMLH0A0HYM2), the
# S&P 500 and the Nasdaq Composite "Copyrighted: Pre-Approval Required" ("Reproduction of this data in any form is prohibited
# except with the prior written permission of ICE Data Indices") - they are NOT used. Yahoo Finance has no public API (the
# chart endpoint is unofficial; its terms do not allow automated retrieval or redistribution) - the S&P 500 drawdown, the
# HYG/IEI high-yield nowcast and the BDC prices are gone with it. Everything left is public domain or "Copyrighted:
# Citation Required" on FRED (Moody's, Freddie Mac, Cboe VIX, Chicago Fed NFCI, the curve, NBER) and is credited on fci.html.
# Private credit (2026-10-04, user: "What about private credit? This has been a huge funder of the AI CAPEX boom" ->
# "Sub-index + panel"): bank loans to nonbank financial institutions as the funding read (the listed-BDC market read used
# Yahoo prices and was removed the same day - see DATA TERMS). The history (2015 on) is too short for the 2001 weights, so
# it is a separate sub-index standardised over its own history and stays out of the headline.
PRIVATE = {
    "pc_ndfi": ("Bank lending to nonbank lenders (slower growth = tighter)", "minus the year-on-year % change in commercial banks' loans to nondepository financial institutions (H.8, LNFACBW027SBOG; one-off reclassification jumps chain-linked out)"),
}
# sub-indices (2026-10-04, user chose "PCA + rate sub-indices"): the PCA headline is the corporate credit cycle (real fed
# funds, loan growth and the curve barely load, the curve with a recession-steepening sign), so three transparent
# equal-weight averages of the oriented z-scores sit beside it and show where the tightness is
SUBS = {"corporate": ("Corporate credit", ["baa", "aaa", "cp_nonfin", "cp_fin", "cp_out", "ci_loans", "sloos_lg", "sloos_sm",
                                           "sloos_dem", "corp_debt"]),
        "mortgage": ("Mortgages", ["real_mort", "mort_sprd"]),
        "policy": ("Policy rate and yield curve", ["real_ff", "curve"]),
        "markets": ("Markets", ["vix", "usd"])}


def fetch(sid, cache: Path, max_age_h=6.0):
    """One FRED series as a float Series indexed by date (cached; a failed refresh falls back to the cache)."""
    fp = cache / f"{sid}.csv"
    if not fp.exists() or time.time() - fp.stat().st_mtime > max_age_h * 3600:
        for attempt in range(6):
            try:
                # an honest user agent (2026-10-04, data-terms rule; FRED answers it). A browser UA used to hang (oil board notes).
                req = urllib.request.Request(FRED.format(sid), headers={"User-Agent": "scorvec.com credit conditions index (+https://scorvec.com/fci.html)", "Accept": "*/*"})
                with urllib.request.urlopen(req, timeout=25) as r:
                    txt = r.read().decode()
                if not txt.startswith("observation_date"):
                    raise ValueError("not a FRED csv")
                fp.write_text(txt)
                break
            except Exception as e:                                   # noqa: BLE001
                if attempt == 5:
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
    W["vix"] = weekly(raw["VIXCLS"])
    old, new = raw["TWEXB"], raw["DTWEXBGS"]
    k = (new / old.reindex(new.index)).dropna(); k = float(k[k.index <= "2019-12-31"].median())
    usd = pd.concat([old[old.index < new.index[0]] * k, new]).sort_index()
    W["usd"] = weekly(yoy(usd), "last")
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


def daily_index(raw, X, mu, sd, w, scale, days=400):
    """The headline day by day (2026-10-04, user: 'a zoomed-in "real time" conditions chart'): the same weights and
    standardisation; inputs that move daily (spreads, CP rates, fed funds, the curve, VIX, the S&P 500, the dollar) take each
    business day's value, the slow ones (loan officer survey, bank loans, CP outstanding, Z.1, the weekly mortgage rate)
    their latest published value - the weekly frame forward-filled. Daily levels are noisier than the weekly averages."""
    end = max(raw[k].index.max() for k in ("BAA10Y", "DFF", "VIXCLS"))
    bd = pd.bdate_range(end - pd.Timedelta(days=days + 400), end)
    dl = lambda s_: s_.reindex(s_.index.union(bd)).ffill(limit=5).reindex(bd)
    core = yoy(raw["PCEPILFE"]); core.index = core.index + pd.Timedelta(days=RELEASE_LAG_D["PCEPILFE"]); core = dl(core.reindex(core.index.union(bd)).ffill())
    Xd = X.reindex(X.index.union(bd)).ffill(limit=10).reindex(bd)          # slow inputs: latest published value
    Xd["baa"] = dl(raw["BAA10Y"]); Xd["aaa"] = dl(raw["AAA10Y"])
    tb = raw["DTB3"]
    Xd["cp_nonfin"] = dl(raw["DCPN3M"] - tb.reindex(raw["DCPN3M"].index)); Xd["cp_fin"] = dl(raw["DCPF3M"] - tb.reindex(raw["DCPF3M"].index))
    Xd["real_ff"] = dl(raw["DFF"]) - core; Xd["curve"] = -dl(raw["T10Y2Y"])
    mort = dl(raw["MORTGAGE30US"].reindex(raw["MORTGAGE30US"].index.union(bd)).ffill(limit=7))
    Xd["real_mort"] = mort - core; Xd["mort_sprd"] = mort - dl(raw["DGS10"])
    Xd["vix"] = dl(raw["VIXCLS"])
    usd = raw["DTWEXBGS"]; ud = dl(usd)
    Xd["usd"] = 100.0 * (ud / dl(usd.reindex(usd.index.union(bd - pd.DateOffset(years=1))).ffill().reindex(bd - pd.DateOffset(years=1))).values - 1.0)
    Xd = Xd[list(X.columns)]
    idx, cover, contrib, Z = index_from(Xd, mu, sd, w, scale)
    keep = (idx.index >= end - pd.Timedelta(days=days)) & (cover >= 0.8)
    idx, contrib = idx[keep].dropna(), contrib[keep]
    groups = {}
    for k, (lab, g, desc) in INPUTS.items():
        groups.setdefault(g, []).append(k)
    return dict(index=[[str(d.date()), round(float(v), 3)] for d, v in idx.items()],
                contrib={g: [[str(d.date()), round(float(x), 3)] for d, x in contrib[ks].sum(axis=1, min_count=1).reindex(idx.index).items()] for g, ks in groups.items()},
                asof=str(idx.index[-1].date()), latest=round(float(idx.iloc[-1]), 3),
                chg_5d=round(float(idx.iloc[-1] - idx.iloc[-6]), 3) if len(idx) > 6 else None,
                chg_21d=round(float(idx.iloc[-1] - idx.iloc[-22]), 3) if len(idx) > 22 else None)


def private_credit(raw, cache):
    """Weekly private-credit inputs (higher = tighter), their own-history z-scores and the panel's series; None without data."""
    nd = raw["LNFACBW027SBOG"].sort_index()
    if len(nd) < 120: return None
    g = nd.pct_change()
    g[g.abs() > 0.10] = 0.0                                          # a >10 % week is a reclassification (Jan 2025: +20 %)
    ndc = (1 + g.fillna(0)).cumprod() * nd.iloc[0]
    I = pd.DataFrame({"pc_ndfi": -weekly(yoy(ndc), "last", RELEASE_LAG_D["TOTCI"])})
    I = I[I.index >= pd.Timestamp("2006-01-01")]
    I["pc_ndfi"] = I["pc_ndfi"].ffill(limit=4)
    Z = (I - I.mean()) / I.std(ddof=0)
    sub = Z.mean(axis=1, skipna=True).dropna()
    nd_w = nd.resample("W-FRI").last()
    return dict(I=I, Z=Z, sub=sub, ndfi_level=nd_w, ndfi_yoy=-I["pc_ndfi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/fci")
    ap.add_argument("--cache", default="scripts/fci/data")
    a = ap.parse_args()
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw = {}
    for s in SERIES:                                             # one at a time, gently: FRED throttles bursts
        raw[s] = fetch(s, cache); time.sleep(0.5)
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
    pc = private_credit(raw, cache)
    if pc is not None:
        v = pc["sub"]
        subs["private"] = dict(label="Private credit", inputs=list(PRIVATE), latest=round(float(v.iloc[-1]), 3),
                               pct_rank=round(float((v <= v.iloc[-1]).mean()), 3), chg_13w=round(float(v.iloc[-1] - v.iloc[-14]), 3),
                               since=str(v.index[0].date()), series=[[str(d.date()), round(float(x), 3)] for d, x in v.items()])
        for k, (lab, desc) in PRIVATE.items():
            s_ = pc["I"][k].dropna(); zz = pc["Z"][k].dropna()
            inputs.append(dict(key=k, label=lab, group="Private credit (not in the headline)", desc=desc + f" · since {s_.index[0].year}",
                               loading=None, value=round(float(s_.iloc[-1]), 3), asof=str(s_.index[-1].date()),
                               z=round(float(zz.iloc[-1]), 2), z_13w_ago=round(float(zz.iloc[-14]), 2) if len(zz) > 14 else None,
                               contribution=None, pct_rank=round(float((zz <= zz.iloc[-1]).mean()), 3)))
        t0 = pd.Timestamp(last) - pd.DateOffset(years=6)
        privp = dict(ndfi=[[str(d.date()), round(float(x), 1), None if not np.isfinite(y_) else round(float(y_), 2)]
                           for (d, x), y_ in zip(pc["ndfi_level"].dropna().items(), pc["ndfi_yoy"].reindex(pc["ndfi_level"].dropna().index).values)])
    else:
        privp = None
    ser = lambda s, nd=3: [[str(d.date()), None if not np.isfinite(v) else round(float(v), nd)] for d, v in s.items()]
    js = dict(made=pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"), asof=str(last.date()),
              latest=round(float(idx[last]), 3), pct_rank=round(pct, 3),
              chg_4w=round(float(idx[last] - idx.dropna().iloc[-5]), 3), chg_13w=round(float(idx[last] - idx.dropna().iloc[-14]), 3),
              chg_52w=round(float(idx[last] - idx.dropna().iloc[-53]), 3),
              fit=dict(start=FIT_START, weeks=int(nfit), pc1_share=round(float(share[0]), 3), pc2_share=round(float(share[1]), 3)),
              private=privp, daily=daily_index(raw, X, mu, sd, w, scale), subs=subs, recessions=spans, corr=corr, groups={g: round(v, 3) for g, v in groups.items()}, inputs=inputs,
              index=ser(idx), coverage=ser(cover, 2), bench={k: ser(v) for k, v in bench.items()},
              contrib={g: ser(contrib[[k for k, x in INPUTS.items() if x[1] == g]].sum(axis=1, min_count=1)) for g in groups})
    (out / "fci.json").write_text(json.dumps(js, separators=(",", ":")))
    print(f"FCI {last.date()}: {idx[last]:+.2f} sd (pct {pct:.0%}); PC1 {share[0]:.0%} of variance over {nfit} weeks; "
          f"corr NFCI {corr['NFCI']}, NFCI credit {corr['NFCICREDIT']}")
    print("loadings: " + ", ".join(f"{k} {w[k]:+.2f}" for k in w.sort_values(ascending=False).index))
    print("groups now: " + ", ".join(f"{g} {v:+.2f}" for g, v in groups.items()))
    print("sub-indices: " + ", ".join(f"{v['label']} {v['latest']:+.2f} (pct {v['pct_rank']:.0%}, 13w {v['chg_13w']:+.2f})" for v in subs.values()))
    if pc is not None:
        print("private credit inputs now: " + ", ".join(f"{k} {pc['I'][k].dropna().iloc[-1]:+.2f} (z {pc['Z'][k].dropna().iloc[-1]:+.2f})" for k in PRIVATE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
