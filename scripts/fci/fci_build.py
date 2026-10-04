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
    "eq_dd":      ("S&P 500 below its 52-week high", "Markets", "% drawdown of the S&P 500 from its highest weekly close of the past year"),
    "usd":        ("Stronger dollar", "Markets", "year-on-year % change in the Fed's broad trade-weighted dollar (DTWEXBGS, spliced to TWEXB before 2006)"),
}
# days from a value's date stamp to its publication (conservative)
RELEASE_LAG_D = {"TOTCI": 9, "COMPOUT": 2, "PCEPILFE": 58, "DRTSCILM": 50, "DRTSCIS": 50, "DRSDCILM": 50, "BCNSDODNS": 160,
                 "MORTGAGE30US": 0}
CARRY_W = {"usd": 3, "sloos_lg": 20, "sloos_sm": 20, "sloos_dem": 20, "corp_debt": 30, "ci_loans": 4, "cp_out": 3, "real_mort": 2}
SERIES = ["DFF", "T10Y2Y", "DGS10", "DTB3", "MORTGAGE30US", "BAA10Y", "AAA10Y", "DCPN3M", "DCPF3M", "COMPOUT", "TOTCI",
          "DRTSCILM", "DRTSCIS", "DRSDCILM", "BCNSDODNS", "PCEPILFE", "NFCI", "NFCICREDIT",
          "BAMLC0A0CM", "BAMLC0A4CBBB", "BAMLH0A0HYM2", "USREC",
          "VIXCLS", "DTWEXBGS", "TWEXB", "NASDAQCOM"]
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval=1d&events=div"
# sub-indices (2026-10-04, user chose "PCA + rate sub-indices"): the PCA headline is the corporate credit cycle (real fed
# funds, loan growth and the curve barely load, the curve with a recession-steepening sign), so three transparent
# equal-weight averages of the oriented z-scores sit beside it and show where the tightness is
SUBS = {"corporate": ("Corporate credit", ["baa", "aaa", "cp_nonfin", "cp_fin", "cp_out", "ci_loans", "sloos_lg", "sloos_sm",
                                           "sloos_dem", "corp_debt"]),
        "mortgage": ("Mortgages", ["real_mort", "mort_sprd"]),
        "policy": ("Policy rate and yield curve", ["real_ff", "curve"]),
        "markets": ("Markets", ["vix", "eq_dd", "usd"])}
SHOW_ONLY = {"BAMLC0A0CM": "Investment-grade spread (ICE BofA OAS)", "BAMLC0A4CBBB": "BBB spread (ICE BofA OAS)",
             "BAMLH0A0HYM2": "High-yield spread (ICE BofA OAS)"}


def fetch(sid, cache: Path, max_age_h=6.0):
    """One FRED series as a float Series indexed by date (cached; a failed refresh falls back to the cache)."""
    fp = cache / f"{sid}.csv"
    if not fp.exists() or time.time() - fp.stat().st_mtime > max_age_h * 3600:
        for attempt in range(6):
            try:
                # FRED hangs some clients (oil board notes: a browser UA never answers); curl's UA is the one that works
                req = urllib.request.Request(FRED.format(sid), headers={"User-Agent": "curl/8.4.0", "Accept": "*/*"})
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


def fetch_yahoo(ticker, cache: Path, start="1990-01-01", max_age_h=6.0, field="adjclose"):
    """Daily close (dividend-adjusted by default) from Yahoo's chart API, cached. Yahoo answers a bare "Mozilla/5.0" UA and
    throttles bursts (429 'Too Many Requests'): one request at a time, retried with back-off; None if it never answers."""
    fp = cache / f"yahoo_{ticker.replace('^', '_')}.json"
    if not fp.exists() or time.time() - fp.stat().st_mtime > max_age_h * 3600:
        url = YAHOO.format(urllib.request.quote(ticker), int(pd.Timestamp(start).timestamp()), int(time.time()))
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    txt = r.read().decode()
                json.loads(txt)["chart"]["result"][0]["timestamp"]
                fp.write_text(txt)
                break
            except Exception as e:                                   # noqa: BLE001
                if attempt == 4:
                    print(f"  yahoo {ticker}: {e}", flush=True)
                time.sleep(3 * (attempt + 1))
        time.sleep(1.2)
    if not fp.exists():
        return None
    d = json.loads(fp.read_text())["chart"]["result"][0]
    vals = d["indicators"]["adjclose"][0]["adjclose"] if field == "adjclose" else d["indicators"]["quote"][0][field]
    idx = pd.to_datetime(d["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    return pd.Series(vals, index=idx, dtype=float).dropna()


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
    eq = raw.get("SPX")
    if eq is None or len(eq) < 2000:                                  # Yahoo unreachable: the Nasdaq from FRED stands in
        eq = raw["NASDAQCOM"]; print("  equity: S&P 500 unavailable, using the Nasdaq Composite (FRED)", flush=True)
    pw = eq.resample("W-FRI").last()
    W["eq_dd"] = 100.0 * (1.0 - pw / pw.rolling(52, min_periods=26).max())
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


def hy_nowcast(raw, cache):
    """Real-time high-yield spread (2026-10-04, user: HYG/IEI "sounds like a plan"): the official ICE BofA HY OAS posts on
    FRED a day late; between postings, today's spread = the last official value + the change implied by HYG's daily return
    against IEI's (3-7 y Treasuries: TLT's 16-year duration swamped the credit signal, daily r 0.33 vs 0.76). The regression
    is refitted every run on the trailing 500 days; the track record is the one-day-ahead error over the last 250 days
    (each day estimated from the previous official value, out of sample)."""
    hyg, iei = fetch_yahoo("HYG", cache, start="2021-01-01"), fetch_yahoo("IEI", cache, start="2021-01-01")
    if hyg is None or iei is None:
        return None
    R = pd.concat([np.log(hyg).diff().rename("hyg") * 100, np.log(iei).diff().rename("iei") * 100], axis=1).dropna()
    oas = raw["BAMLH0A0HYM2"] * 100.0                                   # bp
    D = pd.concat([R, oas.diff().rename("d")], axis=1).dropna()
    def coefs(F):
        X = np.c_[np.ones(len(F)), F["hyg"], F["iei"]]
        return np.linalg.lstsq(X, F["d"].values, rcond=None)[0]
    pred, act = [], []
    for t in D.index[-250:]:                                          # out of sample: fitted on the 500 days before t
        F = D[D.index < t].iloc[-500:]
        if len(F) < 200:
            continue
        b = coefs(F); pred.append(b[0] + b[1] * D.at[t, "hyg"] + b[2] * D.at[t, "iei"]); act.append(D.at[t, "d"])
    pred, act = np.array(pred), np.array(act)
    b = coefs(D.iloc[-500:])
    last = oas.index[-1]; after = R[R.index > last]                   # ETF days the official series has not reached yet
    est = float(oas.iloc[-1] + (b[0] + after["hyg"] * b[1] + after["iei"] * b[2]).sum()) if len(after) else None
    track = []
    for t in D.index[-120:]:                                          # yesterday's official + today's ETF-implied change
        prev = oas[oas.index < t]
        if len(prev):
            track.append([str(t.date()), round(float(oas.at[t]) / 100, 3),
                          round(float(prev.iloc[-1] + b[0] + b[1] * D.at[t, "hyg"] + b[2] * D.at[t, "iei"]) / 100, 3)])
    return dict(official=dict(date=str(last.date()), value=round(float(oas.iloc[-1]) / 100, 3)),
                estimate=None if est is None else dict(date=str(after.index[-1].date()), value=round(est / 100, 3), days=len(after)),
                bp_per_pct=dict(hyg=round(float(b[1]), 1), iei=round(float(b[2]), 1)),
                oos=dict(days=int(len(act)), r=round(float(np.corrcoef(pred, act)[0, 1]), 2),
                         rmse_bp=round(float(np.sqrt(np.mean((pred - act) ** 2))), 1),
                         no_change_bp=round(float(np.sqrt(np.mean(act ** 2))), 1)),
                track=track, etf_last=str(R.index[-1].date()))


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
    raw["SPX"] = fetch_yahoo("^GSPC", cache, field="close")
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
              hy_nowcast=hy_nowcast(raw, cache), subs=subs, recessions=spans, corr=corr, groups={g: round(v, 3) for g, v in groups.items()}, inputs=inputs, show_only=show,
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
