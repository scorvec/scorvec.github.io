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

Fed funds and the mortgage rate enter in REAL terms (minus trailing core PCE inflation) - in nominal levels their secular
decline since the 1980s would become the first component by itself.

2026-10-07 extension (user approved a gap list: high-yield spreads, an equity risk premium and drawdown, the TIPS yield and
term premium, a liquidity block, CRE / card standards and delinquencies as context, the Call Report breakdown of bank loans to
nonbank lenders). Every new input is a CANDIDATE: it joins its group's equal-weight sub-index, but enters the PC1 headline
only if the ADMISSION TEST (admission()) says it improves or preserves the fit to past credit events and to the Chicago Fed
NFCI credit subindex. ADMIT holds the inputs admitted on that evidence; the build recomputes the test every run, publishes
the evidence on the page and warns if it no longer supports ADMIT.

  python scripts/fci/fci_build.py [--out assets/fci] [--cache scripts/fci/data]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import callreport  # noqa: E402
import housing  # noqa: E402

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
EBP_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"
SHILLER_PAGE = "https://shillerdata.com/"
UA = "scorvec.com credit conditions index (+https://scorvec.com/fci.html)"
FIT_START = "2001-01-05"        # every headline input exists from here (CP outstanding starts 2001-01)
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
BASE = list(INPUTS)

# CANDIDATES (2026-10-07). DATA TERMS checked 2026-10-07 on each FRED series page: the ICE BofA high-yield OAS (BAMLH0A0HYM2)
# and CCC & lower OAS (BAMLH0A3HYC) are "Copyrighted: Pre-Approval Required" AND FRED now holds only their last 3 years
# (from 2023-10-09) - they cannot be used. The free stand-in for a high-yield-inclusive spread is the Federal Reserve
# Board's Gilchrist-Zakrajsek spread and excess bond premium (FEDS Note, Favara et al. 2016; monthly from 1973, posted on
# the 4th business day of the next month; a staff research product, revised as the bond sample updates - no vintages).
# The S&P 500 on FRED (SP500) is also Pre-Approval Required, so equities come from Robert Shiller's monthly ie_data.xls
# (shillerdata.com; free download, no licence stated, credited on the page). SOFR and RRPONTSYD (NY Fed) are "Copyrighted:
# Citation Required"; DFII10, THREEFYTP10, WRESBAL, IORB/IOER, TLAACBW027SBOG, the SLOOS and delinquency series are public
# domain (citation requested).
CAND = {
    "gz":        ("Corporate bond spread incl. high yield (GZ)", "Corporate bonds", "Gilchrist-Zakrajšek spread: average spread over matched Treasuries of senior unsecured bonds of US nonfinancial firms, investment grade and high yield (Federal Reserve Board, monthly; entered 40 days after the month starts)"),
    "ebp":       ("Excess bond premium", "Corporate bonds", "the GZ spread net of expected default risk: the price of bearing corporate credit risk (Federal Reserve Board, monthly; entered 40 days after the month starts)"),
    "erp":       ("Equity risk premium (cheaper stocks = tighter)", "Markets", "S&P 500 cyclically adjusted earnings yield (10-year average real earnings, entered 6 months late, over the monthly average price, entered a month late; Shiller) minus the 10-year real Treasury yield"),
    "eq_dd":     ("S&P 500 below its 12-month high", "Markets", "% the S&P 500 monthly average price sits below its highest monthly average of the past 12 months (Shiller; entered a month late)"),
    "real10":    ("Real 10-year Treasury yield", "Policy and rates", "10-year TIPS yield (DFII10) from 2003; before that, a stated proxy: the 10-year Treasury minus core PCE inflation over the past year, shifted by the 2003–07 average gap to TIPS"),
    "tp":        ("10-year term premium", "Policy and rates", "Kim-Wright 10-year term premium (Federal Reserve Board, THREEFYTP10), weekly average"),
    "sloos_cre": ("Banks tightening commercial real estate standards", "Bank lending", "Senior Loan Officer Survey, net % tightening CRE loans (DRTSCREL to 2013Q3; then the average of construction, nonfarm nonresidential and multifamily: SUBLPDRCSC/N/M)"),
    "sloos_cc":  ("Banks tightening credit card standards", "Bank lending", "Senior Loan Officer Survey, net % tightening credit card loans (DRTSCLCC)"),
    "del_ci":    ("Business loan delinquency rate", "Bank lending", "delinquency rate on business loans, all commercial banks (DRBLACBS), seasonally adjusted; entered 150 days after the quarter starts"),
    "del_cc":    ("Credit card delinquency rate", "Bank lending", "delinquency rate on credit card loans, all commercial banks (DRCCLACBS); entered 150 days after the quarter starts"),
    "del_cre":   ("Commercial real estate delinquency rate", "Bank lending", "delinquency rate on CRE loans excluding farmland, all commercial banks (DRCRELEXFACBS); entered 150 days after the quarter starts"),
}
CONTEXT = ["sloos_cre", "sloos_cc", "del_ci", "del_cc", "del_cre"]          # shown as context unless admitted
# ADMITTED on the evidence of the 2026-10-07 admission test (see admission(); run 37687348415): against the 16-input base
# (r NFCI credit 0.770, mean episode-peak percentile 0.855) the greedy path added the credit-card standards (r +0.016,
# peaks +0.2 pp) and then the S&P 500 drawdown (r +0.008, peaks +1.0 pp). Every other candidate failed the gate: the GZ
# spread, ERP, term premium and the three delinquency rates lower r and the episode peaks; CRE standards raise r (+0.044)
# but drop the 1998 peak 2.3 pp; the TIPS yield drops 2018 and 2022; the EBP passes alone but not after the two above.
# The build re-checks this every run and warns when the evidence moves.
ADMIT: list[str] = ["sloos_cc", "eq_dd"]
# liquidity (2026-10-07): histories start 2009 (ample reserves), 2013 (ON RRP facility) and 2018 (SOFR) - far too short for the
# 2001 weights, so a separate sub-index standardised over its own history, like private credit; not a headline candidate
LIQ = {
    "liq_res":  ("Bank reserves (smaller share = tighter)", "minus reserve balances with Federal Reserve Banks (H.4.1, WRESBAL) as % of commercial banks' total assets (H.8, TLAACBW027SBOG); from 2009, the ample-reserves regime"),
    "liq_rrp":  ("Overnight reverse repo cushion (smaller = tighter)", "minus the Fed's overnight reverse repo take-up (RRPONTSYD) as % of commercial banks' total assets; from the facility's start in Sep 2013. Money funds' cash parked at the Fed is the buffer that balance-sheet runoff drains before it drains bank reserves, so a smaller cushion reads as tighter"),
    "liq_repo": ("Repo rate over the Fed's floor (SOFR minus IORB)", "SOFR minus the interest rate on reserve balances (IORB; IOER before 29 Jul 2021), weekly average, bp; from April 2018. No free general-collateral repo rate exists before SOFR"),
}
PRIVATE = {
    "pc_ndfi": ("Bank lending to nonbank lenders (slower growth = tighter)", "minus the year-on-year % change in commercial banks' loans to nondepository financial institutions (H.8, LNFACBW027SBOG; one-off reclassification jumps chain-linked out)"),
}
# HOUSEHOLD HOUSING FINANCE (2026-10-07, user: "What about including something to do with refinancing/HELOCs?"). DATA TERMS
# checked 2026-10-07: the FRED series below are Federal Reserve Board data with no copyright notice (public domain, citation
# requested); FHFA's National Mortgage Database aggregates are a federal agency's public data; the New York Fed's Household
# Debt and Credit workbook may be used and redistributed with its notice (housing.py). NOT USED: the MBA refinance index
# (proprietary), Freddie Mac's cash-out refinance report (last issued for 2018Q4; its successor, the Refinance Transition
# Report, covers Freddie Mac loans from 2020 only), ICE/Black Knight outstanding-rate data (proprietary). HMDA refinance counts
# are annual and duplicate NMDB, so they are not used.
# Stage 2 of the admission test: each of these is added to the CURRENT headline (BASE + ADMIT); only those with data from
# before the 2001 fit start can compete, the rest form the housing sub-index only (like the liquidity block).
HCAND = {
    "heloc_g":    ("Bank home-equity lines (slower growth = tighter)", "Household housing finance", "minus the year-on-year % change in revolving home-equity loans at commercial banks (H.8, RHEACBW027SBOG; one-off reclassification jumps chain-linked out)"),
    "sloos_mort": ("Banks tightening residential mortgage standards", "Household housing finance", "Senior Loan Officer Survey, net % tightening: all mortgages (H0SUBLPDHMSNQ) to 2007Q1, prime (DRTSPM) 2007Q2–2014Q4, GSE-eligible (SUBLPDHMSENQ) from 2015 - three question wordings spliced end to end"),
    "sloos_mdem": ("Weaker residential mortgage demand", "Household housing finance", "minus the Senior Loan Officer Survey net % reporting stronger mortgage demand: all mortgages (H0SUBLPDHMDNQ) to 2007Q1, prime (DRSDPM) 2007Q2–2014Q4, GSE-eligible (SUBLPDHMDENQ) from 2015, spliced"),
    "refi_cash":  ("Cash-out refinancing (less = tighter)", "Household housing finance", "minus cash-out refinance loan volume (the whole new loans, not only the cash taken out; FHFA National Mortgage Database, quarterly, annualised) as % of households' one-to-four-family mortgage debt (Fed Z.1, HHMSDODNS); entered 275 days after the quarter ends"),
    "del_mort":   ("Residential mortgage delinquency rate", "Household housing finance", "delinquency rate on single-family residential mortgages, all commercial banks (DRSFRMACBS), seasonally adjusted; entered 150 days after the quarter starts"),
    "sloos_heloc": ("Banks tightening home-equity line standards", "Household housing finance", "Senior Loan Officer Survey, net % tightening standards for HELOCs (SUBLPDCLHSNQ); asked since 2008"),
    "heloc_dem":  ("Weaker home-equity line demand", "Household housing finance", "minus the Senior Loan Officer Survey net % reporting stronger HELOC demand (SUBLPDCLHDNQ); since 2008"),
    "heloc_lim":  ("Home-equity credit limits (slower growth = tighter)", "Household housing finance", "minus the year-on-year % change in home-equity revolving credit limits, all lenders (New York Fed Consumer Credit Panel / Equifax, quarterly from 2003; entered 50 days after the quarter ends)"),
    "lockin":     ("Mortgage lock-in gap", "Household housing finance", "Freddie Mac 30-year fixed rate minus the average contract rate on all outstanding first-lien mortgages (FHFA National Mortgage Database, balance-weighted, quarterly from 2013, entered 95 days after the quarter ends). Positive = most owners would refinance into a higher rate: little rate-refinancing or cash-out capacity"),
}
HCONTEXT = ["del_mort", "del_heloc"]                     # context unless admitted; del_heloc (2003) is chart and table only
HSUB = ["heloc_g", "heloc_lim", "sloos_heloc", "heloc_dem", "sloos_mort", "sloos_mdem", "refi_cash", "lockin"]
# ADMITTED on the evidence of stage 2 of the admission test (2026-10-07, run 37692314610): against the current 18-input headline
# (r NFCI credit 0.794, mean episode-peak percentile 0.867) only the cash-out refinance ratio clears the gate with a non-negative
# gain, and only formally: loading +0.02, r and the peaks unchanged to three decimals (d_r 95 % CI -0.002..+0.002). The bank
# HELOC growth loads the wrong way (-0.11; banks' home-equity balances rise when borrowers draw lines in a crunch) and fails;
# mortgage standards raise r (+0.023, CI -0.007..+0.049) but cut the 2018 peak 3.7 pp and fail; mortgage demand and bank
# mortgage delinquency pass the gate alone but lose the greedy step. The HELOC survey questions (2008), HE credit limits (2003)
# and the lock-in gap (2013) start too late to compete and feed the housing sub-index only.
ADMIT_H: list[str] = ["refi_cash"]
ALL = dict(INPUTS); ALL.update(CAND); ALL.update(HCAND)
ALL["del_heloc"] = ("Home-equity line delinquency rate", "Household housing finance", "% of HELOC balances 90+ days delinquent, all lenders (New York Fed Consumer Credit Panel / Equifax, quarterly from 2003; entered 50 days after the quarter ends)")

# days from a value's date stamp to its publication (conservative)
RELEASE_LAG_D = {"TOTCI": 9, "COMPOUT": 2, "PCEPILFE": 58, "DRTSCILM": 50, "DRTSCIS": 50, "DRSDCILM": 50, "BCNSDODNS": 160,
                 "MORTGAGE30US": 0, "EBP": 40, "SHILLER_P": 32, "SLOOS": 50, "DELINQ": 150, "THREEFYTP10": 5,
                 "WRESBAL": 1, "TLAACBW027SBOG": 9, "SOFR": 1,
                 "RHEACBW027SBOG": 9, "NMDB_NEW": 275, "NMDB_OUT": 95, "HHDC": 50}     # NMDB/HHDC: days after the quarter END
CARRY_W = {"usd": 3, "sloos_lg": 20, "sloos_sm": 20, "sloos_dem": 20, "corp_debt": 30, "ci_loans": 4, "cp_out": 3, "real_mort": 2,
           "gz": 9, "ebp": 9, "erp": 9, "eq_dd": 9, "real10": 2, "tp": 3, "sloos_cre": 20, "sloos_cc": 20, "del_ci": 20, "del_cc": 20,
           "del_cre": 20, "liq_res": 3, "liq_rrp": 2, "liq_repo": 2,
           "heloc_g": 4, "sloos_mort": 20, "sloos_mdem": 20, "refi_cash": 20, "del_mort": 20, "sloos_heloc": 20, "heloc_dem": 20,
           "heloc_lim": 20, "lockin": 2, "del_heloc": 20}
SERIES = ["DFF", "T10Y2Y", "DGS10", "DTB3", "MORTGAGE30US", "BAA10Y", "AAA10Y", "DCPN3M", "DCPF3M", "COMPOUT", "TOTCI",
          "DRTSCILM", "DRTSCIS", "DRSDCILM", "BCNSDODNS", "PCEPILFE", "NFCI", "NFCICREDIT",
          "USREC", "VIXCLS", "DTWEXBGS", "TWEXB", "LNFACBW027SBOG",
          "DFII10", "THREEFYTP10", "WRESBAL", "TLAACBW027SBOG", "RRPONTSYD", "SOFR", "IORB", "IOER",
          "DRTSCREL", "SUBLPDRCSC", "SUBLPDRCSN", "SUBLPDRCSM", "DRTSCLCC", "DRBLACBS", "DRCCLACBS", "DRCRELEXFACBS",
          "RHEACBW027SBOG", "H0SUBLPDHMSNQ", "DRTSPM", "SUBLPDHMSENQ", "H0SUBLPDHMDNQ", "DRSDPM", "SUBLPDHMDENQ",
          "SUBLPDCLHSNQ", "SUBLPDCLHDNQ", "DRSFRMACBS", "HHMSDODNS"]
# private credit (2026-10-04, user: "What about private credit? ..." -> "Sub-index + panel"): bank loans to nonbank financial
# institutions as the funding read; the H.8 history (2015 on) is too short for the 2001 weights, so it is a separate sub-index
# standardised over its own history and stays out of the headline. 2026-10-07: the Call Report breakdown by borrower type
# (callreport.py) is charted beside it - quarterly from Dec 2024 only, so descriptive.
# sub-indices (2026-10-04, user chose "PCA + rate sub-indices"): transparent equal-weight averages of the oriented z-scores
SUBS = {"corporate": ("Corporate credit", ["baa", "aaa", "gz", "ebp", "cp_nonfin", "cp_fin", "cp_out", "ci_loans", "sloos_lg", "sloos_sm",
                                           "sloos_dem", "corp_debt"]),
        "mortgage": ("Mortgages", ["real_mort", "mort_sprd"]),
        "policy": ("Policy rate and yield curve", ["real_ff", "curve", "real10", "tp"]),
        "markets": ("Markets", ["vix", "usd", "erp", "eq_dd"]),
        "housing": ("Home equity and refinancing", HSUB)}
# credit events and market-led tightenings the headline must register (admission test): code, name, window start, end
EPISODES = [("1998", "LTCM and Russia", "1998-08-01", "1998-12-31"), ("2001", "2001 recession", "2000-10-01", "2002-10-31"),
            ("2008", "Global financial crisis", "2008-09-01", "2009-06-30"), ("2020", "Covid shock", "2020-02-15", "2020-06-30"),
            ("2023", "Regional bank stress", "2023-03-01", "2023-06-30"), ("2018", "Late-2018 selloff", "2018-10-01", "2019-01-31"),
            ("2022", "2022 rate shock", "2022-01-01", "2022-12-31"), ("2025", "April 2025 tariff shock", "2025-03-15", "2025-05-31")]
TOL_R, TOL_PEAK, TOL_EP = 0.005, 0.005, 0.02     # admission: "preserves" = r down <= 0.005, mean peak pct <= 0.5 pp, no episode > 2 pp


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def cached(name, cache: Path, getter, check, max_age_h=6.0):
    """A cached download: refresh when older than max_age_h; a failed refresh falls back to the cache."""
    fp = cache / name
    if not fp.exists() or time.time() - fp.stat().st_mtime > max_age_h * 3600:
        for attempt in range(6):
            try:
                b = getter()
                check(b)
                fp.write_bytes(b)
                break
            except Exception as e:                                   # noqa: BLE001
                if attempt == 5:
                    if not fp.exists():
                        raise
                    print(f"  {name}: refresh failed ({e}); using the cache", flush=True)
                time.sleep(2 ** attempt)
    return fp


def fetch(sid, cache: Path, max_age_h=6.0):
    """One FRED series as a float Series indexed by date (cached; a failed refresh falls back to the cache)."""
    def chk(b):
        if not b.decode().startswith("observation_date"):
            raise ValueError("not a FRED csv")
    # an honest user agent (2026-10-04, data-terms rule; FRED answers it). A browser UA used to hang (oil board notes).
    fp = cached(f"{sid}.csv", cache, lambda: _get(FRED.format(sid), 25), chk, max_age_h)
    d = pd.read_csv(fp)
    s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(d.iloc[:, 0])
    s = s.dropna()
    if s.index.duplicated().any():
        raise ValueError(f"{sid}: duplicated dates")
    return s


def fetch_ebp(cache):
    """The Fed Board's GZ spread and excess bond premium, monthly (first-of-month stamps), screened."""
    def chk(b):
        if not b.decode().startswith("date,gz_spread,ebp"):
            raise ValueError("unexpected EBP csv header")
    d = pd.read_csv(cached("ebp_csv.csv", cache, lambda: _get(EBP_URL), chk), parse_dates=["date"]).set_index("date")
    d = d[["gz_spread", "ebp"]].astype(float).dropna()
    assert d.index.is_monotonic_increasing and not d.index.duplicated().any(), "EBP dates not monotone/unique"
    assert (d.index.day == 1).all(), "EBP dates are not first-of-month"
    bad = ~d["gz_spread"].between(0.0, 15.0) | ~d["ebp"].between(-3.0, 10.0)
    assert not bad.any(), f"EBP values out of range: {d[bad].head()}"
    if (pd.Timestamp.now() - d.index[-1]).days > 75:
        print(f"  ::warning::EBP csv is stale (latest month {d.index[-1].date()})", flush=True)
    return d


def fetch_shiller(cache):
    """Robert Shiller's monthly S&P composite price, earnings and CPI (ie_data.xls, sheet Data), screened."""
    def get_xls():
        page = _get(SHILLER_PAGE).decode("utf-8", "replace")
        links = re.findall(r'href="([^"]*ie_data\.xls[^"]*)"', page)
        if not links:
            raise ValueError("no ie_data.xls link on shillerdata.com")
        u = links[0]
        return _get("https:" + u if u.startswith("//") else u, 120)

    def chk(b):
        if not b[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":                 # OLE2 (.xls) signature
            raise ValueError("not an .xls file")
    fp = cached("ie_data.xls", cache, get_xls, chk, max_age_h=24.0)
    raw = pd.read_excel(fp, sheet_name="Data", header=None, engine="xlrd")
    hdr = [i for i in range(20) if str(raw.iloc[i, 0]).strip() == "Date"]
    assert hdr and str(raw.iloc[hdr[0], 1]).strip() == "P" and str(raw.iloc[hdr[0], 3]).strip() == "E" \
        and str(raw.iloc[hdr[0], 4]).strip() == "CPI", "Shiller layout changed"
    rows = []
    for _, r in raw.iloc[hdr[0] + 1:].iterrows():
        try:
            dt = float(r[0])
        except (TypeError, ValueError):
            break                                                     # the notes under the table
        if not np.isfinite(dt):
            break
        y, m = f"{dt:.2f}".split(".")                                 # 2025.1 = October 2025
        rows.append((pd.Timestamp(int(y), int(m), 1), r[1], r[3], r[4]))
    d = pd.DataFrame(rows, columns=["date", "P", "E", "CPI"]).set_index("date").apply(pd.to_numeric, errors="coerce")
    assert d.index.is_monotonic_increasing and not d.index.duplicated().any(), "Shiller months not monotone/unique"
    assert (d["P"].dropna() > 0).all() and (d["CPI"].dropna() > 0).all(), "Shiller values out of range"
    gaps = d.index.to_series().diff().dt.days.dropna()
    assert gaps.between(28, 31).all(), "Shiller months not contiguous"
    if (pd.Timestamp.now() - d["P"].dropna().index[-1]).days > 70:
        print(f"  ::warning::Shiller price is stale (latest {d['P'].dropna().index[-1].date()})", flush=True)
    return d


def weekly(s, how="mean", lag_d=0):
    """To the Friday grid. Daily: the mean of the week; sparser series: the latest value PUBLISHED by that Friday."""
    if lag_d:
        s = s.copy(); s.index = s.index + pd.Timedelta(days=lag_d)
    s = s[s.index <= pd.Timestamp.now().normalize()]                  # not yet published = not yet known
    if how == "mean":
        return s.resample("W-FRI").mean()
    return s.resample("W-FRI").last().ffill()


def yoy(s):
    """Year-on-year % change of a level series at its own frequency."""
    s = s.sort_index()
    prev = s.reindex(s.index - pd.DateOffset(years=1), method="nearest", tolerance=pd.Timedelta(days=10))
    return pd.Series(100.0 * (s.values / prev.values - 1.0), index=s.index)


def shiller_monthly(shl):
    """Cyclically adjusted earnings yield (%) and drawdown from the 12-month high (%), by observation month.
    Real time: the price of month m is known once the month ends; earnings come with a six-month lag (S&P reports quarters
    6-8 weeks after they end and Shiller interpolates months inside a quarter), so the 10-year average of real earnings for
    month m runs over m-125 .. m-6, deflated by CPI and re-inflated to month m-1 prices (the latest CPI then out)."""
    er = shl["E"] / shl["CPI"]
    avg_e = er.rolling(120, min_periods=120).mean().shift(6)
    ey = 100.0 * avg_e * shl["CPI"].shift(1) / shl["P"]
    dd = 100.0 * (1.0 - shl["P"] / shl["P"].rolling(12, min_periods=12).max())
    return ey.dropna(), dd.dropna()


def splice(*parts):
    """End-to-end splice of successive question wordings: each part supplies the dates after the previous part's last date."""
    out = []
    for s in parts:
        s = s.dropna()
        if out:
            s = s[s.index > out[-1].index[-1]]
        if len(s):
            out.append(s)
    return pd.concat(out).sort_index()


def link_breaks(s, thr, name):
    """Chain-link one-off reclassification jumps (a weekly change beyond thr) out of a level series; logs them."""
    s = s.sort_index()
    g = s.pct_change()
    jumps = g[g.abs() > thr]
    if len(jumps):
        print(f"  {name}: {len(jumps)} weekly jumps beyond {thr:.0%} linked out: " + ", ".join(f"{d.date()} {100 * v:+.1f}%" for d, v in jumps.items()), flush=True)
    g[g.abs() > thr] = 0.0
    return (1 + g.fillna(0)).cumprod() * s.iloc[0]


def _qseries(d):
    """{quarter_end: value} -> float Series."""
    return pd.Series({pd.Timestamp(k): float(v) for k, v in (d or {}).items()}, dtype=float).sort_index()


def housing_inputs(raw, hs, mort):
    """Household housing-finance inputs (higher = tighter) as weekly series, plus the descriptive series for the panel."""
    W, P = {}, {}
    he = link_breaks(raw["RHEACBW027SBOG"], 0.03, "RHEACBW027SBOG")
    W["heloc_g"] = -weekly(yoy(he), "last", RELEASE_LAG_D["RHEACBW027SBOG"])
    mstd = splice(raw["H0SUBLPDHMSNQ"], raw["DRTSPM"], raw["SUBLPDHMSENQ"])
    mdem = splice(raw["H0SUBLPDHMDNQ"], raw["DRSDPM"], raw["SUBLPDHMDENQ"])
    W["sloos_mort"] = weekly(mstd, "last", RELEASE_LAG_D["SLOOS"])
    W["sloos_mdem"] = -weekly(mdem, "last", RELEASE_LAG_D["SLOOS"])
    W["sloos_heloc"] = weekly(raw["SUBLPDCLHSNQ"], "last", RELEASE_LAG_D["SLOOS"])
    W["heloc_dem"] = -weekly(raw["SUBLPDCLHDNQ"], "last", RELEASE_LAG_D["SLOOS"])
    W["del_mort"] = weekly(raw["DRSFRMACBS"], "last", RELEASE_LAG_D["DELINQ"])
    new = (hs.get("nmdb_new") or {}).get("data", {})
    out = (hs.get("nmdb_out") or {}).get("data", {})
    hh = (hs.get("hhdc") or {}).get("data", {})
    empty = pd.Series(dtype=float)
    # cash-out refinancing: refinance-market volume ($ million) x the dollar-weighted cash-out share, annualised, over home
    # mortgage debt at the end of the same quarter (Z.1 stamps the quarter's first day)
    if new:
        vol = _qseries(new.get("All Mortgages (Refinance)|TOT_LOANAMT|1")) * _qseries(new.get("All Mortgages (Refinance)|PCT_CASHOUT|2")) / 100.0
        debt = raw["HHMSDODNS"].copy() / 1000.0; debt.index = debt.index + pd.offsets.QuarterEnd(0)    # $ million -> $ billion
        assert 1000.0 < debt.iloc[-1] < 50000.0, f"HHMSDODNS units changed ({debt.iloc[-1]:.0f} $ billion?)"
        ratio = (400.0 * vol / 1000.0 / debt.reindex(vol.index)).dropna()                                  # annualised, %
        assert ratio.between(0.1, 30.0).all(), "cash-out refinance ratio out of range"
        W["refi_cash"] = -weekly(ratio, "last", RELEASE_LAG_D["NMDB_NEW"])
        P["refi_share"] = _qseries(new.get("All Mortgages|PCT_REFI|1"))
        P["cash_share"] = _qseries(new.get("All Mortgages (Refinance)|PCT_CASHOUT|1"))
        P["cash_ratio"] = ratio
        P["orig"] = _qseries(new.get("All Mortgages|TOT_ORIG|1"))
    else:
        W["refi_cash"] = empty
    if out:
        orate = _qseries(out.get("All Mortgages|AVE_INTRATE|2"))
        ow = weekly(orate, "last", RELEASE_LAG_D["NMDB_OUT"])
        ow = ow.reindex(ow.index.union(mort.index)).ffill(limit=CARRY_W["refi_cash"]).reindex(mort.index)
        W["lockin"] = (mort - ow).dropna()
        P["out_rate"] = orate
        P["lt3"] = _qseries(out.get("All Mortgages|PCT_INTRATE_LT_3|2"))
        P["ge6"] = _qseries(out.get("All Mortgages|PCT_INTRATE_GE_6|2"))
    else:
        W["lockin"] = empty
    if hh:
        lim = _qseries(hh.get("he_lim"))
        W["heloc_lim"] = -weekly(yoy(lim), "last", RELEASE_LAG_D["HHDC"])
        W["del_heloc"] = weekly(_qseries(hh.get("heloc_90")), "last", RELEASE_LAG_D["HHDC"])
        P["he_lim"], P["he_bal"] = lim, _qseries(hh.get("he_bal"))
        P["hhdc_report"] = hh.get("report")
    else:
        W["heloc_lim"] = W["del_heloc"] = empty
    P["heloc_h8"] = he
    P["mort30"] = mort
    return W, P


def build_inputs(raw, ebp, shl, hs=None):
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
    # --- candidates (2026-10-07) ---
    W["gz"] = weekly(ebp["gz_spread"], "last", RELEASE_LAG_D["EBP"])
    W["ebp"] = weekly(ebp["ebp"], "last", RELEASE_LAG_D["EBP"])
    tips = weekly(raw["DFII10"]).dropna()
    prox = weekly(raw["DGS10"]); prox = (prox - core_w.reindex(prox.index).ffill()).dropna()
    off = float((tips - prox.reindex(tips.index)).loc["2003-01-01":"2007-12-31"].dropna().mean())
    t0 = tips.index[0]
    W["real10"] = pd.concat([prox[prox.index < t0] + off, tips]).sort_index()
    ey, dd = shiller_monthly(shl)
    ey_w = weekly(ey, "last", RELEASE_LAG_D["SHILLER_P"])
    ey_w = ey_w.reindex(ey_w.index.union(W["real10"].index)).ffill(limit=CARRY_W["erp"]).reindex(W["real10"].index)
    W["erp"] = ey_w - W["real10"]
    W["eq_dd"] = weekly(dd, "last", RELEASE_LAG_D["SHILLER_P"])
    W["tp"] = weekly(raw["THREEFYTP10"], "mean", RELEASE_LAG_D["THREEFYTP10"])
    sub = pd.concat([raw[k] for k in ("SUBLPDRCSC", "SUBLPDRCSN", "SUBLPDRCSM")], axis=1).mean(axis=1, skipna=False).dropna()
    cre = pd.concat([raw["DRTSCREL"][raw["DRTSCREL"].index < sub.index[0]], sub]).sort_index()
    W["sloos_cre"] = weekly(cre, "last", RELEASE_LAG_D["SLOOS"])
    W["sloos_cc"] = weekly(raw["DRTSCLCC"], "last", RELEASE_LAG_D["SLOOS"])
    W["del_ci"] = weekly(raw["DRBLACBS"], "last", RELEASE_LAG_D["DELINQ"])
    W["del_cc"] = weekly(raw["DRCCLACBS"], "last", RELEASE_LAG_D["DELINQ"])
    W["del_cre"] = weekly(raw["DRCRELEXFACBS"], "last", RELEASE_LAG_D["DELINQ"])
    HW, HP = housing_inputs(raw, hs or {}, mort)
    W.update(HW)
    X = pd.DataFrame(W)
    # every input carries forward to the latest Friday until its next release, but never further than its own cadence
    # allows (a discontinued series must not freeze into the index): weeks
    grid = pd.date_range(X.index.min(), max(s.dropna().index.max() for s in W.values() if len(s.dropna())), freq="W-FRI")
    X = X.reindex(grid)
    for k in X.columns:
        X[k] = X[k].ffill(limit=CARRY_W.get(k, 2))
    meta = dict(real10_offset=round(off, 3), real10_tips_start=str(t0.date()), housing=HP)
    return X[X.index >= pd.Timestamp(SHOW_START)], meta


def liquidity(raw, grid):
    """Liquidity inputs (higher = tighter) on the weekly grid, with screens."""
    ta = weekly(raw["TLAACBW027SBOG"], "last", RELEASE_LAG_D["TLAACBW027SBOG"])                 # $ billion
    res = weekly(raw["WRESBAL"], "last", RELEASE_LAG_D["WRESBAL"]) / 1000.0                     # $ million -> $ billion
    res_sh = 100.0 * res / ta.reindex(res.index).ffill()
    res_sh = res_sh[res_sh.index >= "2009-01-01"]
    rrp = raw["RRPONTSYD"]; rrp = rrp[rrp.index >= "2013-09-23"]
    bd = raw["DGS10"].index[(raw["DGS10"].index >= rrp.index[0]) & (raw["DGS10"].index <= rrp.index[-1])]
    miss = bd.difference(rrp.index)
    print(f"  RRPONTSYD: {len(rrp)} days since 2013-09-23; {len(miss)} of {len(bd)} Treasury market days missing", flush=True)
    if len(miss) > 0.05 * len(bd):
        print(f"  ::warning::RRPONTSYD misses {len(miss)} of {len(bd)} market days", flush=True)
    rrp_w = weekly(rrp, "mean", RELEASE_LAG_D["SOFR"])
    rrp_sh = 100.0 * rrp_w / ta.reindex(rrp_w.index).ffill()
    floor = pd.concat([raw["IOER"][raw["IOER"].index < raw["IORB"].index[0]], raw["IORB"]]).sort_index()
    sofr = raw["SOFR"]
    spread = (100.0 * (sofr - floor.reindex(floor.index.union(sofr.index)).ffill().reindex(sofr.index))).dropna()
    bad = spread.abs() > 1000                                              # a units error, not stress (17 Sep 2019 was +315 bp, kept)
    if bad.any():
        print(f"  ::warning::SOFR-IORB beyond 1000 bp on {int(bad.sum())} days (dropped)", flush=True)
        spread = spread[~bad]
    rep = weekly(spread, "mean", RELEASE_LAG_D["SOFR"])
    assert res_sh.dropna().between(0, 60).all() and rrp_sh.dropna().between(0, 60).all(), "liquidity shares out of range"
    I = pd.DataFrame({"liq_res": -res_sh, "liq_rrp": -rrp_sh, "liq_repo": rep})
    I = I.reindex(I.index.union(grid))
    for k in I.columns:
        I[k] = I[k].ffill(limit=CARRY_W[k])
    I = I.reindex(grid)
    I = I[I.index >= pd.Timestamp("2009-01-01")]
    Z = (I - I.mean()) / I.std(ddof=0)
    sub = Z.mean(axis=1, skipna=True).where(Z.notna().sum(axis=1) >= 2).dropna()
    return dict(I=I, Z=Z, sub=sub)


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


def evaluate(X, keys, bench):
    """Fit PC1 on `keys` and score it: correlation with the NFCI credit subindex and the NFCI (weekly, 1990 on), and the
    percentile (among all weeks since 1990) of the index's peak inside each episode window."""
    mu, sd, w, share, scale, n = fit_pca(X[keys])
    idx, cover, _, _ = index_from(X[keys], mu, sd, w, scale)
    idx = idx[cover >= 0.5].dropna()
    out = dict(keys=list(keys), w=w, share=float(share[0]), idx=idx, weeks=int(n))
    for b, v in bench.items():
        j = pd.concat([idx, v], axis=1, sort=True).dropna()
        out["r_" + b] = float(j.corr().iloc[0, 1])
    pk = {}
    for code, _, a, z in EPISODES:
        seg = idx[(idx.index >= a) & (idx.index <= z)]
        pk[code] = float((idx <= seg.max()).mean()) if len(seg) else float("nan")
    out["peaks"] = pk
    out["mean_peak"] = float(np.nanmean(list(pk.values())))
    return out


def _gate(m, ref):
    d_r = m["r_NFCICREDIT"] - ref["r_NFCICREDIT"]
    d_p = m["mean_peak"] - ref["mean_peak"]
    worst = min(m["peaks"][e] - ref["peaks"][e] for e in ref["peaks"])
    ok = d_r >= -TOL_R and d_p >= -TOL_PEAK and worst >= -TOL_EP
    return ok, d_r, d_p, worst


def block_boot_dr(a, b, y, block=52, reps=2000, seed=7):
    """95 % moving-block bootstrap interval for corr(a, y) - corr(b, y) on aligned weekly series (52-week blocks)."""
    j = pd.concat([a, b, y], axis=1, sort=True).dropna().values
    n = len(j); rng = np.random.default_rng(seed); nb = int(np.ceil(n / block)); out = []
    for _ in range(reps):
        st = rng.integers(0, n - block + 1, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        s = j[ix]
        out.append(np.corrcoef(s[:, 0], s[:, 2])[0, 1] - np.corrcoef(s[:, 1], s[:, 2])[0, 1])
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def admission(X, bench, base=None, cands=None):
    """Greedy forward selection over the candidates with history from before the fit start: at each step the candidate whose
    addition raises (corr with NFCI credit + mean episode-peak percentile) the most, among those that pass the gate (r falls
    by no more than TOL_R, mean peak percentile by no more than TOL_PEAK, no single episode by more than TOL_EP); stop when
    none passes with a non-negative gain. In-sample evidence (the weights are fitted on the same weeks)."""
    base = list(BASE if base is None else base)
    cands = CAND if cands is None else cands
    eligible = [c for c in cands if len(X[c].dropna()) and X[c].dropna().index.min() <= pd.Timestamp(FIT_START)]
    m0 = evaluate(X, base, bench)
    single = {}
    for c in cands:
        if c not in eligible:
            single[c] = None; continue
        m = evaluate(X, base + [c], bench)
        ok, d_r, d_p, worst = _gate(m, m0)
        single[c] = dict(ok=bool(ok), d_r=round(d_r, 3), d_r_nfci=round(m["r_NFCI"] - m0["r_NFCI"], 3), d_peak=round(d_p, 3),
                         worst=round(worst, 3), loading=round(float(m["w"][c]), 3), peaks={e: round(v, 3) for e, v in m["peaks"].items()})
    cur, mc, path, left = list(base), m0, [], list(eligible)
    while left:
        best = None
        for c in left:
            m = evaluate(X, cur + [c], bench)
            ok, d_r, d_p, worst = _gate(m, mc)
            if ok and d_r + d_p >= 0 and (best is None or d_r + d_p > best[1]):
                best = (c, d_r + d_p, m, d_r, d_p)
        if best is None:
            break
        cur.append(best[0]); left.remove(best[0]); mc = best[2]
        path.append(dict(add=best[0], d_r=round(best[3], 3), d_peak=round(best[4], 3), r=round(mc["r_NFCICREDIT"], 3), mean_peak=round(mc["mean_peak"], 3)))
    return m0, single, path, cur


def daily_index(raw, X, mu, sd, w, scale, days=400):
    """The headline day by day (2026-10-04, user: 'a zoomed-in "real time" conditions chart'): the same weights and
    standardisation; inputs that move daily take each business day's value, the slow ones (loan officer survey, bank loans,
    CP outstanding, Z.1, the weekly mortgage rate, the monthly Fed Board and Shiller series) their latest published value -
    the weekly frame forward-filled. Daily levels are noisier than the weekly averages."""
    keys = list(w.index)
    end = max(raw[k].index.max() for k in ("BAA10Y", "DFF", "VIXCLS"))
    bd = pd.bdate_range(end - pd.Timedelta(days=days + 400), end)
    dl = lambda s_: s_.reindex(s_.index.union(bd)).ffill(limit=5).reindex(bd)
    core = yoy(raw["PCEPILFE"]); core.index = core.index + pd.Timedelta(days=RELEASE_LAG_D["PCEPILFE"]); core = dl(core.reindex(core.index.union(bd)).ffill())
    Xd = X[keys].reindex(X.index.union(bd)).ffill(limit=10).reindex(bd)    # slow inputs: latest published value
    tb = raw["DTB3"]
    mort = lambda: dl(raw["MORTGAGE30US"].reindex(raw["MORTGAGE30US"].index.union(bd)).ffill(limit=7))
    usd = raw["DTWEXBGS"]
    fast = {"baa": lambda: dl(raw["BAA10Y"]), "aaa": lambda: dl(raw["AAA10Y"]),
            "cp_nonfin": lambda: dl(raw["DCPN3M"] - tb.reindex(raw["DCPN3M"].index)),
            "cp_fin": lambda: dl(raw["DCPF3M"] - tb.reindex(raw["DCPF3M"].index)),
            "real_ff": lambda: dl(raw["DFF"]) - core, "curve": lambda: -dl(raw["T10Y2Y"]),
            "real_mort": lambda: mort() - core, "mort_sprd": lambda: mort() - dl(raw["DGS10"]),
            "vix": lambda: dl(raw["VIXCLS"]), "real10": lambda: dl(raw["DFII10"]),
            "tp": lambda: dl(raw["THREEFYTP10"]),
            "usd": lambda: 100.0 * (dl(usd) / dl(usd.reindex(usd.index.union(bd - pd.DateOffset(years=1))).ffill().reindex(bd - pd.DateOffset(years=1))).values - 1.0)}
    for k in keys:
        if k in fast:
            Xd[k] = fast[k]()
    if "erp" in keys:                                                       # monthly earnings yield minus the daily TIPS yield
        ey = (X["erp"] + X["real10"]).reindex(X.index.union(bd)).ffill(limit=10).reindex(bd)
        Xd["erp"] = ey - dl(raw["DFII10"])
    idx, cover, contrib, Z = index_from(Xd[keys], mu, sd, w, scale)
    keep = (idx.index >= end - pd.Timedelta(days=days)) & (cover >= 0.8)
    idx, contrib = idx[keep].dropna(), contrib[keep]
    groups = {}
    for k in keys:
        groups.setdefault(ALL[k][1], []).append(k)
    return dict(index=[[str(d.date()), round(float(v), 3)] for d, v in idx.items()],
                contrib={g: [[str(d.date()), round(float(x), 3)] for d, x in contrib[ks].sum(axis=1, min_count=1).reindex(idx.index).items()] for g, ks in groups.items()},
                asof=str(idx.index[-1].date()), latest=round(float(idx.iloc[-1]), 3),
                chg_5d=round(float(idx.iloc[-1] - idx.iloc[-6]), 3) if len(idx) > 6 else None,
                chg_21d=round(float(idx.iloc[-1] - idx.iloc[-22]), 3) if len(idx) > 22 else None)


def private_credit(raw):
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


def housing_panel(X, HP):
    """Series for the home equity and refinancing card. Index inputs as entered (dated when public, natural sign); the NMDB
    and HHDC descriptive series by the quarter they describe."""
    q = lambda s, nd=2: ser(s.dropna(), nd) if s is not None and len(s.dropna()) else []
    o = dict(heloc=q(-X["heloc_g"]), he_lim=steps(-X["heloc_lim"], 2),
             sloos=dict(heloc=steps(X["sloos_heloc"], 1), heloc_dem=steps(-X["heloc_dem"], 1), mort=steps(X["sloos_mort"], 1),
                        mort_dem=steps(-X["sloos_mdem"], 1), splices=["2007-04-01", "2015-01-01"]),
             refi=dict(share=q(HP.get("refi_share"), 1), cash_share=q(HP.get("cash_share"), 1), cash_ratio=q(HP.get("cash_ratio"), 2)),
             lockin=dict(gap=q(X["lockin"]), out_rate=q(HP.get("out_rate"), 1), lt3=q(HP.get("lt3"), 1), ge6=q(HP.get("ge6"), 1)),
             hhdc_report=HP.get("hhdc_report"))
    lk = X["lockin"].dropna()
    if len(lk):
        m = HP["mort30"].dropna()
        o["lockin"]["mort"] = q(m[m.index >= lk.index[0] - pd.DateOffset(months=6)])
    return o


def ser(s, nd=3):
    return [[str(d.date()), None if not np.isfinite(v) else round(float(v), nd)] for d, v in s.items()]


def steps(s, nd=3):
    """A carried-forward (monthly/quarterly) weekly series as its change points only: plotted as a step line."""
    s = s.dropna().round(nd)
    s = s[s.ne(s.shift())]
    return ser(s, nd)


def sub_entry(label, keys, v):
    vv = v.dropna()
    return dict(label=label, inputs=keys, latest=round(float(vv.iloc[-1]), 3), pct_rank=round(float((vv <= vv.iloc[-1]).mean()), 3),
                chg_13w=round(float(vv.iloc[-1] - vv.iloc[-14]), 3), since=str(vv.index[0].date()), series=ser(vv))


def input_row(k, label, group, desc, s, zser, loading=None, contribution=None):
    s = s.dropna(); zz = zser.dropna()
    ch = s[s.ne(s.shift())]                                           # the week the current value entered (carried values stay put)
    z = zz.iloc[-1] if len(zz) else np.nan
    return dict(key=k, label=label, group=group, desc=desc, loading=loading,
                value=None if not len(s) else round(float(s.iloc[-1]), 3), asof=str(ch.index[-1].date()) if len(s) else None,
                z=None if not np.isfinite(z) else round(float(z), 2),
                z_13w_ago=round(float(zz.iloc[-14]), 2) if len(zz) > 14 else None,
                contribution=contribution, pct_rank=round(float((zz <= z).mean()), 3) if np.isfinite(z) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/fci")
    ap.add_argument("--cache", default="scripts/fci/data")
    ap.add_argument("--no-callreport", action="store_true")
    a = ap.parse_args()
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw = {}
    for s in SERIES:                                             # one at a time, gently: FRED throttles bursts
        raw[s] = fetch(s, cache); time.sleep(0.5)
    ebp = fetch_ebp(cache)
    shl = fetch_shiller(cache)
    try:
        hs = housing.update(out / "housing_src.json")
    except Exception as e:                                                   # noqa: BLE001
        print(f"  ::warning::housing sources failed ({e})", flush=True); hs = {}
    X, meta = build_inputs(raw, ebp, shl, hs)
    HP = meta.pop("housing")
    benchw = {k: weekly(raw[k], "last") for k in ("NFCI", "NFCICREDIT")}

    # --- admission test (2026-10-07) ---
    m0, single, path, greedy = admission(X, benchw)
    head1 = BASE + [k for k in ADMIT if k in CAND]
    if sorted(greedy) != sorted(head1):
        print(f"::warning::admission test now selects {[k for k in greedy if k not in BASE]} but ADMIT = {ADMIT} - review", flush=True)
    m1 = evaluate(X, head1, benchw)
    # stage 2 (2026-10-07, household housing finance): each candidate added to the current headline
    h0, hsingle, hpath, hgreedy = admission(X, benchw, head1, HCAND)
    head = head1 + [k for k in ADMIT_H if k in HCAND]
    if sorted(hgreedy) != sorted(head):
        print(f"::warning::housing admission now selects {[k for k in hgreedy if k not in head1]} but ADMIT_H = {ADMIT_H} - review", flush=True)
    h1 = evaluate(X, head, benchw)
    X_full = X
    Xh = X[head]
    mu, sd, w, share, scale, nfit = fit_pca(Xh)
    idx, cover, contrib, Zh = index_from(Xh, mu, sd, w, scale)
    keep = cover >= 0.5
    idx = idx[keep]; cover = cover[keep]; contrib = contrib[keep]; Zh = Zh[keep]; X = X[keep]
    last = idx.dropna().index[-1]
    # every non-headline input's tightness score over its own 2001-present values
    Fw = X[X.index >= pd.Timestamp(FIT_START)]
    Zall = (X - Fw.mean()) / Fw.std(ddof=0)
    for k in head:
        Zall[k] = Zh[k]
    bench = {k: v.reindex(idx.index) for k, v in benchw.items()}
    corr = {k: round(float(pd.concat([idx, v], axis=1, sort=True).dropna().corr().iloc[0, 1]), 3) for k, v in bench.items()}
    pct = float((idx.dropna() <= idx[last]).mean())
    subs = {}
    for sk, (slab, keys) in SUBS.items():
        v = Zall[keys].mean(axis=1, skipna=True)
        v = v.where(Zall[keys].notna().sum(axis=1) >= max(1, len(keys) // 2))
        subs[sk] = sub_entry(slab, keys, v)
    liq = liquidity(raw, X.index)
    subs["liquidity"] = sub_entry("Liquidity", list(LIQ), liq["sub"])
    rec = raw["USREC"]; rec = rec[rec.index >= pd.Timestamp(SHOW_START) - pd.DateOffset(months=1)]
    spans, st_ = [], None                                         # NBER recessions as [start, end] month spans
    for d_, v_ in rec.items():
        if v_ >= 1 and st_ is None: st_ = d_
        if v_ < 1 and st_ is not None: spans.append([str(st_.date()), str(d_.date())]); st_ = None
    if st_ is not None: spans.append([str(st_.date()), str(rec.index[-1].date())])
    groups = {}
    for k in head:
        g = ALL[k][1]
        groups.setdefault(g, 0.0)
        v = contrib.loc[last, k]
        if np.isfinite(v):
            groups[g] += float(v)
    inputs = []
    for k in head:
        lab, g, desc = ALL[k]
        cv = contrib.loc[last, k]
        inputs.append(input_row(k, lab, g, desc, X[k], Zall[k], round(float(w[k]), 3), None if not np.isfinite(cv) else round(float(cv), 3)))
    for k in CAND:
        if k in head:
            continue
        lab, g, desc = ALL[k]
        grp = "Context: real estate, consumer credit, loan quality (not in the headline)" if k in CONTEXT else g + " (sub-index only, not the headline)"
        inputs.append(input_row(k, lab, grp, desc, X[k], Zall[k]))
    for k in list(HCAND) + ["del_heloc"]:
        if k in head:
            continue
        lab, g, desc = ALL[k]
        grp = "Context: housing loan quality (not in the headline)" if k in HCONTEXT else "Home equity and refinancing (sub-index only, not the headline)"
        inputs.append(input_row(k, lab, grp, desc, X[k], Zall[k]))
    for k, (lab, desc) in LIQ.items():
        inputs.append(input_row(k, lab, "Liquidity (own sub-index, not the headline)", desc, liq["I"][k], liq["Z"][k]))
    pc = private_credit(raw)
    privp = None
    if pc is not None:
        subs["private"] = sub_entry("Private credit", list(PRIVATE), pc["sub"])
        for k, (lab, desc) in PRIVATE.items():
            inputs.append(input_row(k, lab, "Private credit (not in the headline)", desc + f" · since {pc['I'][k].dropna().index[0].year}", pc["I"][k], pc["Z"][k]))
        privp = dict(ndfi=[[str(d.date()), round(float(x), 1), None if not np.isfinite(y_) else round(float(y_), 2)]
                           for (d, x), y_ in zip(pc["ndfi_level"].dropna().items(), pc["ndfi_yoy"].reindex(pc["ndfi_level"].dropna().index).values)])
    # Call Report breakdown (quarterly, descriptive)
    cr = None
    crp = out / "callreport_ndfi.json"
    if not a.no_callreport:
        try:
            cr = list(callreport.update(crp)["quarters"].values())
        except Exception as e:                                               # noqa: BLE001
            print(f"  ::warning::Call Report update failed ({e})", flush=True)
            cr = list(json.loads(crp.read_text())["quarters"].values()) if crp.exists() else None
    if privp is not None and cr:
        privp["callreport"] = cr
    # panels for the new charts (values as entered in the index: dated when they became public)
    panels = dict(
        spreads=dict(baa=ser(X["baa"].dropna(), 3), gz=steps(X["gz"]), ebp=steps(X["ebp"])),
        equity=dict(erp=ser(X["erp"].dropna(), 2), ey=steps(X["erp"] + X["real10"], 2), dd=steps(X["eq_dd"], 2)),
        rates=dict(real10=ser(X["real10"].dropna(), 2), tp=ser(X["tp"].dropna(), 2), tips_start=meta["real10_tips_start"]),
        liquidity=dict(res=ser(-liq["I"]["liq_res"].dropna(), 2), rrp=ser(-liq["I"]["liq_rrp"].dropna(), 2), repo=ser(liq["I"]["liq_repo"].dropna(), 1)),
        context={k: ser(X[k].dropna().resample("QS").last().dropna(), 2) for k in CONTEXT + HCONTEXT if len(X[k].dropna())},
        housing=housing_panel(X, HP),
    )

    def ev(m):
        return dict(r_credit=round(m["r_NFCICREDIT"], 3), r_nfci=round(m["r_NFCI"], 3), mean_peak=round(m["mean_peak"], 3),
                    peaks={e: round(v, 3) for e, v in m["peaks"].items()}, pc1_share=round(m["share"], 3), weeks=m["weeks"],
                    loadings={k: round(float(v), 3) for k, v in m["w"].items()})
    j = pd.concat([m1["idx"], m0["idx"]], axis=1, sort=True).dropna()
    j2 = pd.concat([h1["idx"], h0["idx"]], axis=1, sort=True).dropna()
    adm_h = dict(before=ev(h0), after=ev(h1), admitted=[k for k in head if k not in head1], greedy=[k for k in hgreedy if k not in head1],
                 path=hpath, single=hsingle, labels={k: ALL[k][0] for k in HCAND}, ineligible=[k for k, v in hsingle.items() if v is None],
                 context=HCONTEXT, r_old_new=round(float(j2.corr().iloc[0, 1]), 3),
                 dr_ci=block_boot_dr(h1["idx"], h0["idx"], benchw["NFCICREDIT"]) if head != head1 else None,
                 single_ci={k: block_boot_dr(evaluate(X_full, head1 + [k], benchw)["idx"], h0["idx"], benchw["NFCICREDIT"])
                            for k, v in hsingle.items() if v is not None})
    adm = dict(rule=dict(tol_r=TOL_R, tol_peak=TOL_PEAK, tol_episode=TOL_EP),
               episodes=[dict(code=c, name=n, start=s_, end=e_) for c, n, s_, e_ in EPISODES],
               before=ev(m0), after=ev(m1), admitted=[k for k in head1 if k not in BASE], greedy=[k for k in greedy if k not in BASE],
               path=path, single=single, labels={k: ALL[k][0] for k in CAND},
               ineligible=[k for k, v in single.items() if v is None],
               r_old_new=round(float(j.corr().iloc[0, 1]), 3),
               dr_ci=block_boot_dr(m1["idx"], m0["idx"], benchw["NFCICREDIT"]) if head1 != BASE else None, housing=adm_h)
    js = dict(made=pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%MZ"), asof=str(last.date()),
              latest=round(float(idx[last]), 3), pct_rank=round(pct, 3),
              chg_4w=round(float(idx[last] - idx.dropna().iloc[-5]), 3), chg_13w=round(float(idx[last] - idx.dropna().iloc[-14]), 3),
              chg_52w=round(float(idx[last] - idx.dropna().iloc[-53]), 3),
              fit=dict(start=FIT_START, weeks=int(nfit), pc1_share=round(float(share[0]), 3), pc2_share=round(float(share[1]), 3), n_inputs=len(head)),
              private=privp, daily=daily_index(raw, X, mu, sd, w, scale), subs=subs, recessions=spans, corr=corr,
              groups={g: round(v, 3) for g, v in groups.items()}, inputs=inputs, admission=adm, panels=panels, meta=meta,
              index=ser(idx), coverage=ser(cover, 2), bench={k: ser(v) for k, v in bench.items()},
              contrib={g: ser(contrib[[k for k in head if ALL[k][1] == g]].sum(axis=1, min_count=1)) for g in groups})
    txt = json.dumps(js, separators=(",", ":"), allow_nan=False)
    (out / "fci.json").write_text(txt)
    print(f"FCI {last.date()}: {idx[last]:+.2f} sd (pct {pct:.0%}); PC1 {share[0]:.0%} of variance over {nfit} weeks; "
          f"corr NFCI {corr['NFCI']}, NFCI credit {corr['NFCICREDIT']}; json {len(txt) / 1e3:.0f} kB")
    print("loadings: " + ", ".join(f"{k} {w[k]:+.2f}" for k in w.sort_values(ascending=False).index))
    print("groups now: " + ", ".join(f"{g} {v:+.2f}" for g, v in groups.items()))
    print("sub-indices: " + ", ".join(f"{v['label']} {v['latest']:+.2f} (pct {v['pct_rank']:.0%}, 13w {v['chg_13w']:+.2f}, since {v['since'][:4]})" for v in subs.values()))
    print("ADMISSION baseline: r credit %.3f, r NFCI %.3f, mean peak pct %.3f, PC1 %.3f, peaks %s" % (m0["r_NFCICREDIT"], m0["r_NFCI"], m0["mean_peak"], m0["share"], {e: round(v, 3) for e, v in m0["peaks"].items()}))
    for k, v in single.items():
        print(f"  + {k:9s}: " + ("ineligible (starts after the fit start)" if v is None else
              f"{'PASS' if v['ok'] else 'fail'} d_r {v['d_r']:+.3f} d_r_nfci {v['d_r_nfci']:+.3f} d_peak {v['d_peak']:+.3f} worst ep {v['worst']:+.3f} loading {v['loading']:+.2f} peaks {v['peaks']}"))
    print("  greedy path: " + json.dumps(path))
    print("  ADMIT in code: %s; after: r credit %.3f, r NFCI %.3f, mean peak %.3f, PC1 %.3f; r(old, new index) %.3f; d_r 95%% CI %s" %
          (ADMIT, m1["r_NFCICREDIT"], m1["r_NFCI"], m1["mean_peak"], m1["share"], adm["r_old_new"], adm["dr_ci"]))
    print("HOUSING ADMISSION (stage 2) baseline = current headline: r credit %.3f, r NFCI %.3f, mean peak %.3f, PC1 %.3f, peaks %s" % (h0["r_NFCICREDIT"], h0["r_NFCI"], h0["mean_peak"], h0["share"], {e: round(v, 3) for e, v in h0["peaks"].items()}))
    for k, v in hsingle.items():
        print(f"  + {k:11s}: " + ("ineligible (starts after the fit start)" if v is None else
              f"{'PASS' if v['ok'] else 'fail'} d_r {v['d_r']:+.3f} d_r_nfci {v['d_r_nfci']:+.3f} d_peak {v['d_peak']:+.3f} worst ep {v['worst']:+.3f} loading {v['loading']:+.2f} peaks {v['peaks']} d_r CI {adm_h['single_ci'][k]}"))
    print("  housing greedy path: " + json.dumps(hpath))
    print("  ADMIT_H in code: %s; after: r credit %.3f, r NFCI %.3f, mean peak %.3f, PC1 %.3f; r(old, new) %.3f; d_r 95%% CI %s" %
          (ADMIT_H, h1["r_NFCICREDIT"], h1["r_NFCI"], h1["mean_peak"], h1["share"], adm_h["r_old_new"], adm_h["dr_ci"]))
    for k in list(HCAND) + ["del_heloc"]:
        s_ = X[k].dropna()
        print(f"  housing input {k}: " + ("EMPTY" if not len(s_) else f"{s_.index[0].date()}..{s_.index[-1].date()} n={len(s_)} latest {s_.iloc[-1]:+.3f} (z {Zall[k].dropna().iloc[-1]:+.2f})"))
    print("now: " + ", ".join(f"{i['key']} {i['value']} (z {i['z']}, {i['asof']})" for i in inputs if i['key'] in list(CAND) + list(LIQ) + list(HCAND)))
    if cr:
        print(f"call report {cr[-1]['quarter']}: {json.dumps(cr[-1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
