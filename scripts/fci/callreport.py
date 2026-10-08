#!/usr/bin/env python3
"""
Banks' loans to nondepository financial institutions by borrower type, from the FFIEC Call Report (2026-10-07, user
approved: "the FFIEC Call Report breakdown of loans to nondepository financial institutions ... aggregate across all banks,
quarterly").

Source: FFIEC Central Data Repository, Public Data Distribution, bulk download "Call Reports -- Single Period" (tab-delimited,
public data). Schedule RC-C Part I, item 9.a, all commercial banks and savings institutions that file a Call Report:

  J454  loans to nondepository financial institutions (all filers)
  PV05  ... to mortgage credit intermediaries          PV06  ... to business credit intermediaries (private credit, BDCs)
  PV07  ... to private equity funds                    PV08  ... to consumer credit intermediaries
  PV09  other loans to nondepository financial institutions
  2122  total loans and leases (for scale)

Each MDRM item comes as RCFD (consolidated, FFIEC 031 filers) and RCON (domestic offices; 041/051 filers). A bank's value is
its RCFD figure when it reports one, else RCON - never the sum. The PV items are reported only by the larger banks; the gap
J454 - (PV05..PV09) is shown as "not broken out". Thousands of dollars in the file, $ billion here.

The FFIEC page is an ASP.NET form: GET it, post back the series choice, then post the Download button with the quarter.
Every quarter's aggregate is kept in assets/fci/callreport_ndfi.json; a run fetches only quarters it does not hold, and
re-pulls the newest two once a week (banks amend their reports).
"""
from __future__ import annotations

import csv
import http.cookiejar
import io
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

URL = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"
UA = {"User-Agent": "Mozilla/5.0 (compatible; scorvec.com credit conditions index; +https://scorvec.com/fci.html)"}
ITEMS = {"J454": "all", "PV05": "mortgage", "PV06": "business", "PV07": "pe", "PV08": "consumer", "PV09": "other", "2122": "loans"}
FIRST_QUARTER = "2024-03-31"            # the breakdown items are looked for from here on (absent quarters are recorded as such)


def _post(op, data, timeout=300):
    req = urllib.request.Request(URL, data=urllib.parse.urlencode(data).encode(),
                                 headers=dict(UA, **{"Content-Type": "application/x-www-form-urlencoded", "Referer": URL}))
    with op.open(req, timeout=timeout) as r:
        return r.read(), r.headers


def _hidden(t):
    return dict(re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', t))


def _session():
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    with op.open(urllib.request.Request(URL, headers=UA), timeout=60) as r:
        t = r.read().decode("utf-8", "replace")
    lb = re.search(r'<select[^>]*name="([^"]*ListBox1)"', t).group(1)
    form = _hidden(t); form.update({lb: "ReportingSeriesSinglePeriod", "__EVENTTARGET": lb, "__EVENTARGUMENT": ""})
    b, _ = _post(op, form, 60)
    t2 = b.decode("utf-8", "replace")
    sel = re.search(r'<select[^>]*name="([^"]*DatesDropDownList)"[^>]*>(.*?)</select>', t2, re.S)
    dates = [(v, datetime.strptime(lab.strip(), "%m/%d/%Y").strftime("%Y-%m-%d"))
             for v, lab in re.findall(r'<option[^>]*value="([^"]*)"[^>]*>([^<]*)<', sel.group(2))]
    return op, t2, lb, sel.group(1), dates


def download(quarter):
    """The bulk zip (all schedules, TSV) for one quarter end 'YYYY-MM-DD'."""
    op, t2, lb, dname, dates = _session()
    val = dict((d, v) for v, d in dates).get(quarter)
    if val is None:
        raise KeyError(f"{quarter} not offered")
    form = _hidden(t2)
    form.update({lb: "ReportingSeriesSinglePeriod", dname: val, "ctl00$MainContentHolder$FormatType": "TSVRadioButton",
                 "ctl00$MainContentHolder$TabStrip1$Download_0": "Download", "__EVENTTARGET": "", "__EVENTARGUMENT": ""})
    b, h = _post(op, form)
    if not b.startswith(b"PK"):
        raise ValueError(f"{quarter}: the FFIEC answered {h.get('Content-Type')}, not a zip")
    return b


def offered():
    return [d for _, d in _session()[4]]


def aggregate(zb, quarter):
    """Sum RC-C item 9.a and its breakdown over every filer (RCFD if reported, else RCON) with a consistency screen."""
    z = zipfile.ZipFile(io.BytesIO(zb))
    parts = [n for n in z.namelist() if re.search(r"Schedule RCCI(\s|\(| \d)", n) and "RCCII" not in n]
    if not parts:
        raise ValueError(f"{quarter}: no RC-C Part I file in the zip")
    per_bank = {}
    have = set()
    for n in parts:
        with z.open(n) as f:
            rows = csv.reader(io.TextIOWrapper(f, encoding="latin-1"), delimiter="\t")
            head = next(rows); next(rows)                                  # MDRM codes, then descriptions
            head = [h.strip().strip('"') for h in head]
            col = {h: i for i, h in enumerate(head)}
            idc = col.get("IDRSSD")
            for r in rows:
                if not r or idc is None or len(r) <= idc:
                    continue
                bank = per_bank.setdefault(r[idc].strip('"'), {})
                for it in ITEMS:
                    for pre in ("RCFD", "RCON"):
                        j = col.get(pre + it)
                        if j is None or j >= len(r):
                            continue
                        have.add(it)
                        v = r[j].strip().strip('"')
                        if v == "":
                            continue
                        try:
                            x = float(v)
                        except ValueError:
                            continue
                        if pre == "RCFD" or it not in bank:                # RCFD wins; RCON only when no RCFD
                            bank[it] = x
    tot = {k: 0.0 for k in ITEMS}
    n_break, n_bad, worst = 0, 0, 0.0
    for b in per_bank.values():
        for it in ITEMS:
            tot[it] += b.get(it, 0.0)
        parts_ = [b[k] for k in ("PV05", "PV06", "PV07", "PV08", "PV09") if k in b]
        if parts_:
            n_break += 1
            gap = abs(sum(parts_) - b.get("J454", 0.0))
            if gap > max(1000.0, 0.005 * b.get("J454", 0.0)):                # > $1 million and > 0.5 %: breakdown != total
                n_bad += 1; worst = max(worst, gap)
    out = dict(quarter=quarter, banks=len(per_bank), banks_breakdown=n_break, breakdown_reported=all(k in have for k in ("PV05", "PV06", "PV07", "PV08", "PV09")),
               screen=dict(breakdown_ne_total=n_bad, worst_gap_bn=round(worst / 1e6, 3)))
    out.update({ITEMS[k]: round(v / 1e6, 3) for k, v in tot.items()})               # $ thousand -> $ billion
    if out["breakdown_reported"]:
        out["not_broken_out"] = round(out["all"] - sum(out[k] for k in ("mortgage", "business", "pe", "consumer", "other")), 3)
    return out


def update(path: Path, max_new=12, refresh_days=7.0, log=print):
    """Bring the stored quarterly aggregates up to date; returns the stored dict (quarters sorted)."""
    st = json.loads(path.read_text()) if path.exists() else {"quarters": {}, "refreshed": None}
    q = st["quarters"]
    try:
        avail = sorted(d for d in offered() if d >= FIRST_QUARTER)
    except Exception as e:                                                   # noqa: BLE001
        log(f"  FFIEC: date list failed ({e}); keeping {len(q)} stored quarters")
        return st
    todo = [d for d in avail if d not in q][:max_new]
    now = datetime.now(timezone.utc)
    stale = st.get("refreshed") is None or (now - datetime.fromisoformat(st["refreshed"])).total_seconds() > refresh_days * 86400
    if stale:
        todo += [d for d in avail[-2:] if d not in todo]
    for d in todo:
        for attempt in range(3):
            try:
                agg = aggregate(download(d), d)
                agg["fetched"] = now.strftime("%Y-%m-%dT%H:%MZ")
                q[d] = agg
                log(f"  FFIEC {d}: {agg['banks']} filers, breakdown from {agg['banks_breakdown']}; NDFI ${agg['all']:.0f}B"
                    + (f" (business credit intermediaries ${agg['business']:.0f}B, PE funds ${agg['pe']:.0f}B; screen {agg['screen']})" if agg["breakdown_reported"] else " (no breakdown items this quarter)"))
                break
            except Exception as e:                                           # noqa: BLE001
                log(f"  FFIEC {d}: attempt {attempt + 1} failed ({e})")
                time.sleep(10 * (attempt + 1))
        time.sleep(3)                                                        # gently: one quarter at a time
    if stale and todo:
        st["refreshed"] = now.isoformat()
    st["quarters"] = dict(sorted(q.items()))
    path.write_text(json.dumps(st, separators=(",", ":")))
    return st
