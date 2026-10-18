#!/usr/bin/env python3
"""
Household housing-finance sources (refinancing and home-equity lines) for the credit conditions index. Two non-FRED sources,
both free to republish with attribution; the FRED series (H.8 revolving home-equity loans, the Senior Loan Officer Survey
HELOC and residential-mortgage questions, bank mortgage delinquency) are fetched by fci_build.py like every other FRED input.

1. FHFA National Mortgage Database (NMDB) aggregate statistics (fhfa.gov/data/national-mortgage-database-aggregate-statistics;
   a federal agency's public data, credited on the page). A nationally representative 5 % sample of closed-end first-lien
   mortgages from a credit bureau, scaled to the market.
     - New residential mortgages, quarterly from 1998Q1, national, "All Mortgages" market: originations (1,000s), volume
       ($ million), refinance share and cash-out share of originations; and the same shares inside the refinance market.
       Published about nine months after the quarter (2025Q4 appeared on 2026-09-24).
     - Outstanding residential mortgages, quarterly from 2013Q1: the average contract rate on every outstanding first-lien
       mortgage (VALUE2 = weighted by unpaid balance). Published about three months after the quarter (2026Q2 on 2026-09-24).
   NMDB revises history as late loans reach the bureau; there are no archived vintages, so its history is the latest estimate.
2. Federal Reserve Bank of New York, Quarterly Report on Household Debt and Credit (HHDC; source New York Fed Consumer Credit
   Panel / Equifax). The workbook hhd_c_report_YYYYqN.xlsx: home-equity revolving balance (page 3), home-equity revolving
   credit limit (page 10) and the share of HELOC balances 90+ days delinquent (page 12), quarterly from 2003Q1, published
   about five weeks after the quarter. NY Fed terms of use: free to use and distribute with the notice
   "(c) <year> Federal Reserve Bank of New York. Content from the New York Fed subject to the Terms of Use at newyorkfed.org"
   and the attribution "New York Fed Consumer Credit Panel / Equifax"; derived series must be labelled as modified.

The national aggregates are kept in assets/fci/housing_src.json (small); a run re-downloads a source when its stored copy is
older than refresh_days, and keeps the stored copy when the download or the screen fails.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import re
import time
import urllib.request
import zipfile
from datetime import datetime, UTC
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (compatible; scorvec.com credit conditions index; +https://scorvec.com/fci.html)"}
NMDB = "https://www.fhfa.gov/document/d/nmdb/nmdb-{}-statistics-national-census-areas-quarterly.zip"
HHDC = "https://www.newyorkfed.org/medialibrary/interactives/householdcredit/data/xls/hhd_c_report_{}q{}.xlsx"
NEW_KEEP = {
    ("All Mortgages", "TOT_ORIG"),
    ("All Mortgages", "TOT_LOANAMT"),
    ("All Mortgages", "PCT_REFI"),
    ("All Mortgages", "PCT_CASHOUT"),
    ("All Mortgages (Refinance)", "PCT_CASHOUT"),
    ("All Mortgages (Refinance)", "TOT_LOANAMT"),
    ("All Mortgages", "AVE_INTRATE"),
}
OUT_KEEP = {
    ("All Mortgages", "AVE_INTRATE"),
    ("All Mortgages", "TOT_UPB"),
    ("All Mortgages", "PCT_INTRATE_LT_3"),
    ("All Mortgages", "PCT_INTRATE_GE_6"),
}


def _get(url, timeout=300):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _qend(period):
    """'2025Q4' -> '2025-12-31'."""
    y, q = int(period[:4]), int(period[-1])
    return {1: f"{y}-03-31", 2: f"{y}-06-30", 3: f"{y}-09-30", 4: f"{y}-12-31"}[q]


def _contiguous(qs):
    """True when the quarter-end dates step one quarter at a time."""
    ix = [(int(q[:4]) * 4 + (int(q[5:7]) - 1) // 3) for q in qs]
    return all(b - a == 1 for a, b in itertools.pairwise(ix))


def nmdb(kind):
    """National series from one NMDB quarterly file: {'<market>|<series>|v1/v2': {quarter_end: value}} (screened)."""
    keep = NEW_KEEP if kind == "new-mortgage" else OUT_KEEP
    z = zipfile.ZipFile(io.BytesIO(_get(NMDB.format(kind))))
    name = [n for n in z.namelist() if n.endswith(".csv")]
    if len(name) != 1:
        raise ValueError(f"NMDB {kind}: expected one csv, got {z.namelist()}")
    out, n_sup = {}, 0
    with z.open(name[0]) as f:
        rd = csv.reader(io.TextIOWrapper(f, encoding="utf-8-sig"))
        head = next(rd)
        c = {h: i for i, h in enumerate(head)}
        for need in ("GEOLEVEL", "MARKET", "PERIOD", "SUPPRESSED", "SERIESID", "VALUE1", "VALUE2"):
            if need not in c:
                raise ValueError(f"NMDB {kind}: column {need} missing (layout changed)")
        for r in rd:
            if r[c["GEOLEVEL"]] != "National" or (r[c["MARKET"]], r[c["SERIESID"]]) not in keep:
                continue
            if r[c["SUPPRESSED"]] not in ("0", ""):
                n_sup += 1
                continue
            q = _qend(r[c["PERIOD"]])
            for v in ("VALUE1", "VALUE2"):
                x = r[c[v]].strip()
                if x == "":
                    continue
                key = f"{r[c['MARKET']]}|{r[c['SERIESID']]}|{v[-1]}"
                if q in out.setdefault(key, {}):
                    raise ValueError(f"NMDB {kind}: duplicate {key} {q}")
                out[key][q] = float(x)
    # screen: every kept series present, quarters contiguous, values in range
    for m, s in keep:
        if f"{m}|{s}|1" not in out:
            raise ValueError(f"NMDB {kind}: {m} {s} missing")
    for key, ser in out.items():
        qs = sorted(ser)
        if not _contiguous(qs):
            raise ValueError(f"NMDB {kind}: {key} quarters not contiguous")
        lo, hi = (
            (0.5, 20.0) if "INTRATE|" in key and "PCT_" not in key else (0.0, 100.0) if "|PCT_" in key else (0.0, 1e9)
        )
        bad = [q for q, x in ser.items() if not lo <= x <= hi]
        if bad:
            raise ValueError(f"NMDB {kind}: {key} out of range at {bad[:3]}")
    if n_sup:
        print(f"  NMDB {kind}: {n_sup} suppressed national cells skipped", flush=True)
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def _hhdc_latest():
    """The newest HHDC workbook: walk back from the current quarter."""
    now = datetime.now(UTC)
    y, q = now.year, (now.month - 1) // 3 + 1
    for _ in range(6):
        try:
            b = _get(HHDC.format(y, q), 120)
            if b[:2] == b"PK":
                return f"{y}Q{q}", b
        except Exception:  # noqa: BLE001
            pass
        y, q = (y, q - 1) if q > 1 else (y - 1, 4)
        time.sleep(1)
    raise ValueError("no HHDC workbook found for the last six quarters")


def _qlab(s):
    m = re.fullmatch(r"\s*(\d\d):Q([1-4])\s*", str(s))
    return None if not m else _qend(f"{2000 + int(m.group(1))}Q{m.group(2)}")


def hhdc():
    """HE revolving balance and limit ($ trillion) and HELOC 90+ day delinquency (% of balance), by quarter end (screened)."""
    import pandas as pd

    label, b = _hhdc_latest()
    x = pd.ExcelFile(io.BytesIO(b))

    def sheet(name, cols):
        d = pd.read_excel(x, name, header=None)
        hr = next(i for i in range(10) if any(str(v).strip() in cols for v in d.iloc[i].values))
        idx = {str(v).strip(): j for j, v in enumerate(d.iloc[hr].values)}
        for col in cols:
            if col not in idx:
                raise ValueError(f"HHDC {name}: column {col!r} missing (layout changed)")
        out, cur = {c: {} for c in cols}, None
        for i in range(hr + 1, len(d)):
            q = _qlab(d.iloc[i, 0])
            if q:
                cur = q
            if cur is None:
                continue
            for col in cols:
                v = d.iloc[i, idx[col]]
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v == v:  # not NaN
                    out[col][cur] = v
        return out

    p3 = sheet("Page 3 Data", ["HE Revolving"])
    p10 = sheet("Page 10 Data", ["HE Revolving Balance", "HE Revolving Limit"])
    p12 = sheet("Page 12 Data", ["HELOC"])
    res = dict(
        report=label,
        he_bal=p3["HE Revolving"],
        he_lim=p10["HE Revolving Limit"],
        he_bal10=p10["HE Revolving Balance"],
        heloc_90=p12["HELOC"],
    )
    for k in ("he_bal", "he_lim", "heloc_90"):
        qs = sorted(res[k])
        if len(qs) < 80 or not _contiguous(qs):
            raise ValueError(f"HHDC {k}: {len(qs)} quarters, contiguous={_contiguous(qs)}")
    if not all(0.05 < v < 2.0 for v in res["he_bal"].values()) or not all(
        0.1 < v < 3.0 for v in res["he_lim"].values()
    ):
        raise ValueError("HHDC home-equity balances or limits out of range ($ trillion)")
    if not all(0.0 <= v < 20.0 for v in res["heloc_90"].values()):
        raise ValueError("HHDC HELOC delinquency out of range")
    over = [q for q in res["he_lim"] if res["he_bal10"].get(q, 0) > res["he_lim"][q] + 1e-9]
    if over:
        raise ValueError(f"HHDC: home-equity balance above its limit in {over[:3]}")
    del res["he_bal10"]
    return {k: (dict(sorted(v.items())) if isinstance(v, dict) else v) for k, v in res.items()}


def update(path: Path, refresh_days=7.0, log=print):
    """Bring the stored NMDB and HHDC aggregates up to date; returns the stored dict."""
    st = json.loads(path.read_text()) if path.exists() else {}
    now = datetime.now(UTC)
    for key, fn in (
        ("nmdb_new", lambda: nmdb("new-mortgage")),
        ("nmdb_out", lambda: nmdb("outstanding-mortgage")),
        ("hhdc", hhdc),
    ):
        old = st.get(key)
        if old and (now - datetime.fromisoformat(old["fetched"])).total_seconds() < refresh_days * 86400:
            continue
        for attempt in range(3):
            try:
                data = fn()
                if old:  # a new copy must not lose history
                    for k, v in old["data"].items():
                        if isinstance(v, dict) and k in data and len(data[k]) < len(v) - 1:
                            raise ValueError(f"{key} {k}: {len(data[k])} quarters, stored copy has {len(v)}")
                st[key] = dict(fetched=now.isoformat(), data=data)
                last = {k: (max(v) if isinstance(v, dict) and v else v) for k, v in data.items()}
                log(f"  {key}: refreshed; latest {json.dumps(last)}")
                break
            except Exception as e:  # noqa: BLE001
                log(f"  {key}: attempt {attempt + 1} failed ({e})")
                time.sleep(10 * (attempt + 1))
        else:
            if old:
                log(f"  ::warning::{key} refresh failed; using the stored copy from {old['fetched'][:10]}")
            else:
                log(f"  ::warning::{key} unavailable and nothing stored")
    path.write_text(json.dumps(st, separators=(",", ":")))
    return st
