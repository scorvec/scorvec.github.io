#!/usr/bin/env python3
"""Early-vote tracker for scorvec.com/midterms/earlyvote.html (2026-09-27).

User: "This could be useful to keep track of early voting", "use the Michael McDonald version", then "We can rely on the
official early voting data directly from the states wherever possible". So OFFICIAL state election-office data is the
primary source, and the UF Election Lab tracker (Michael McDonald) covers the states without a machine-readable official
feed. This is turnout tracking only: nothing here feeds the forecast model.

Official sources (aggregates only - no voter-level record is ever stored or published):
  NC  NC State Board of Elections, dl.ncsbe.gov ENRS/2026_11_03/absentee_counts_state_20261103.csv - a daily AGGREGATE
      file (ballots by request type x party x demographics), refreshed ~08:40 UTC with counts through the prior day.
      ONE-STOP / EARLY VOTING = in-person early; CIVILIAN / MILITARY / OVERSEAS = mail. Counts are accepted ballots
      (accepted + accepted-exception + accepted-cured, checked against the voter file on 2026-09-27).
  PA  PA Department of State, data.pa.gov "2026 General Election Mail Ballot Requests" (nbwd-pfn4), public domain.
      Queried with SoQL aggregates (count(*) GROUP BY party) so individual rows never leave the server. PA has no
      in-person early voting; returned = ballotreturneddate present.
  TX  SOS earlyvoting.texas-election.com - the 2026 general is not listed yet (early voting starts 19 Oct); the script
      reports when it appears. Until then TX falls back to UF.
Blocked or CAPTCHA-gated official sources (skipped, never worked around), so these fall back to UF: GA (sos.ga.gov 403),
FL (county VBM/EV public stats behind reCAPTCHA), NV (nvsos.gov Incapsula), MI (mvic Cloudflare challenge), WI
(elections.wi.gov 403), AZ (azsos.gov 403).

UF Election Lab: election.lab.ufl.edu/data-downloads/earlyvote/2026/US.csv, CC BY-NC-ND 4.0 ("If you use these statistics
you MUST provide reasonable attribution and cannot use these data for commercial purposes"). NoDerivatives: UF figures are
shown VERBATIM (counts as published, no shares, ratios or sums computed from them), attributed, with a link to each state's
UF page. Official-state numbers are ours to compute on.

2022 baseline at the same number of days before Election Day (8 Nov 2022): built once with `baseline`
  NC  the 2022 voter-level absentee file, reduced IN MEMORY to accepted ballots per return date x method x party
  PA  data.pa.gov "2022 General Election Mail Ballot Requests" (uhfm-zhus) via SoQL, per issue / return date x party
and stored as midterms/data/earlyvote_baseline_2022.json (aggregates only).

    python scripts/election2026/earlyvote.py baseline      # one-off (NC 2022 file is ~100 MB, streamed, not kept)
    python scripts/election2026/earlyvote.py update        # daily (Actions: .github/workflows/earlyvote.yml)
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "midterms" / "data" / "earlyvote.json"
BASE = REPO / "midterms" / "data" / "earlyvote_baseline_2022.json"
UA = "scorvec.com early-vote tracker (github.com/scorvec/scorvec.github.io)"
E26, E22 = dt.date(2026, 11, 3), dt.date(2022, 11, 8)

NC_S3 = "https://s3.amazonaws.com/dl.ncsbe.gov"
NC_26 = "ENRS/2026_11_03/absentee_counts_state_20261103.csv"
NC_22_ZIP = "ENRS/2022_11_08/absentee_20221108.zip"
PA_26, PA_22 = "nbwd-pfn4", "uhfm-zhus"
UF_CSV = "https://election.lab.ufl.edu/data-downloads/earlyvote/2026/US.csv"
UF_PAGE = "https://election.lab.ufl.edu/early-vote/2026-early-voting/"
TX_PAGE = "https://earlyvoting.texas-election.com/Elections/getElectionDetails.do"
OFFICIAL = {"NC", "PA"}
NAMES = {"NC": "North Carolina", "PA": "Pennsylvania", "TX": "Texas"}


def get(url: str, tries: int = 3, timeout: int = 120) -> bytes:
    """One polite GET: identifying user agent, sequential, backoff on failure; raises after the last try."""
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                                     # noqa: BLE001
            last = e
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def party3(p: str) -> str:
    p = (p or "").strip().upper()
    return "dem" if p in ("DEM", "D") else "rep" if p in ("REP", "R") else "oth"


# ---------------------------------------------------------------------------------------------------------------- NC
def nc_last_modified(key: str) -> dt.datetime:
    s = get(f"{NC_S3}?prefix={urllib.parse.quote(key)}").decode()
    m = re.search(r"<LastModified>([^<]+)</LastModified>", s)
    return dt.datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))


def nc_now() -> dict:
    lm = nc_last_modified(NC_26)
    rows = csv.DictReader(io.StringIO(get(f"{NC_S3}/{NC_26}").decode("latin-1")))
    cnt_col = None
    mail = inp = 0
    party = Counter()
    for r in rows:
        if cnt_col is None:                                       # the count column's header is not stable ("0" in 2026)
            cnt_col = [k for k in r if k not in ("abs_request_type", "party_cd", "gender", "race", "ethnicity")][0]
        n = int(float(r[cnt_col] or 0))
        if not n:
            continue
        t = r["abs_request_type"].strip().upper()
        if t in ("ONE-STOP", "EARLY VOTING"):
            inp += n
        else:
            mail += n
        party[party3(r["party_cd"])] += n
    asof = (lm - dt.timedelta(hours=12)).date()                   # generated overnight with counts through the prior day
    return dict(state="NC", name=NAMES["NC"], source="official", source_name="NC State Board of Elections",
                source_url=f"{NC_S3}/{NC_26}", file_time=lm.strftime("%Y-%m-%d %H:%MZ"), asof=asof.isoformat(),
                mail_returned=mail, in_person=inp, total=mail + inp, requested=None, party=dict(party),
                party_basis="registration of accepted ballots (UNA and minor parties = oth)")


def nc_baseline() -> dict:
    """2022: accepted ballots per return date x method x party, streamed from the voter-level zip and reduced in memory."""
    raw = get(f"{NC_S3}/{NC_22_ZIP}", timeout=600)
    z = zipfile.ZipFile(io.BytesIO(raw))
    daily = defaultdict(Counter)
    with z.open(z.namelist()[0]) as f:
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="latin-1")):
            if not (r.get("ballot_rtn_status") or "").strip().upper().startswith("ACCEPTED"):
                continue
            d = (r.get("ballot_rtn_dt") or "").strip()
            try:
                day = dt.datetime.strptime(d, "%m/%d/%Y").date()
            except ValueError:
                continue
            meth = "in_person" if (r.get("ballot_req_type") or "").strip().upper() in ("ONE-STOP", "EARLY VOTING") else "mail"
            daily[day.isoformat()][f"{meth}|{party3(r.get('voter_party_code'))}"] += 1
    del raw, z
    return {k: dict(v) for k, v in sorted(daily.items())}


# ---------------------------------------------------------------------------------------------------------------- PA
def soql(ds: str, query: str) -> list:
    return json.loads(get(f"https://data.pa.gov/resource/{ds}.json?" + urllib.parse.urlencode({"$query": query})))


def pa_now() -> dict:
    meta = json.loads(get(f"https://data.pa.gov/api/views/{PA_26}.json"))
    upd = dt.datetime.fromtimestamp(meta["rowsUpdatedAt"], dt.timezone.utc)
    req = Counter(); ret = Counter()
    for r in soql(PA_26, "SELECT party, count(*) AS n GROUP BY party"):
        req[party3(r.get("party"))] += int(r["n"])
    for r in soql(PA_26, "SELECT party, count(*) AS n WHERE ballotreturneddate IS NOT NULL GROUP BY party"):
        ret[party3(r.get("party"))] += int(r["n"])
    last = soql(PA_26, f"SELECT max(ballotreturneddate) AS d WHERE ballotreturneddate <= '{dt.date.today().isoformat()}'")
    asof = (last[0].get("d") or upd.date().isoformat())[:10]
    return dict(state="PA", name=NAMES["PA"], source="official", source_name="PA Department of State (data.pa.gov)",
                source_url=f"https://data.pa.gov/d/{PA_26}", file_time=upd.strftime("%Y-%m-%d %H:%MZ"), asof=asof,
                mail_returned=sum(ret.values()), in_person=0, total=sum(ret.values()), requested=sum(req.values()),
                party=dict(ret), party_requested=dict(req), party_basis="registration of returned mail ballots")


def pa_baseline() -> dict:
    """2022: requests per issue date and returns per return date, x party (SoQL aggregates)."""
    out = defaultdict(Counter)
    for r in soql(PA_22, "SELECT date_trunc_ymd(appissuedate) AS d, party, count(*) AS n GROUP BY d, party LIMIT 50000"):
        if r.get("d"):
            out[r["d"][:10]][f"requested|{party3(r.get('party'))}"] += int(r["n"])
    for r in soql(PA_22, "SELECT ballotreturneddate AS d, party, count(*) AS n WHERE ballotreturneddate IS NOT NULL "
                         "GROUP BY ballotreturneddate, party LIMIT 50000"):
        if r.get("d"):
            out[r["d"][:10]][f"mail|{party3(r.get('party'))}"] += int(r["n"])
    return {k: dict(v) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------------------------------------------- 2022
def same_point(daily: dict, asof: str) -> dict | None:
    """Cumulative 2022 counts up to the same number of days before Election Day as `asof` is before 3 Nov 2026."""
    days_before = (E26 - dt.date.fromisoformat(asof)).days
    cut = (E22 - dt.timedelta(days=days_before)).isoformat()
    acc = Counter()
    for d, c in daily.items():
        if d <= cut:
            acc.update(c)
    fin = Counter()
    for c in daily.values():
        fin.update(c)
    def pick(C, pre):
        return {p: sum(v for k, v in C.items() if k.startswith(pre + "|") and k.endswith("|" + p)) for p in ("dem", "rep", "oth")}
    mail, inp, req = pick(acc, "mail"), pick(acc, "in_person"), pick(acc, "requested")
    return dict(days_before=days_before, date_2022=cut, mail_returned=sum(mail.values()), in_person=sum(inp.values()),
                total=sum(mail.values()) + sum(inp.values()), requested=sum(req.values()) or None,
                party={p: mail[p] + inp[p] for p in mail},
                final_total=sum(v for k, v in fin.items() if not k.startswith("requested|")))


# ---------------------------------------------------------------------------------------------------------------- UF
UF_KEEP = ["request_all", "accept_all", "inperson_all", "voted_all", "voted_dem", "voted_rep", "voted_none", "turn_2022"]


def uf_rows() -> list:
    rows = []
    for r in csv.DictReader(io.StringIO(get(UF_CSV).decode("utf-8", "replace"))):
        vals = {k: (int(float(r[k])) if (r.get(k) or "NA") not in ("", "NA") else None) for k in UF_KEEP}
        slug = re.sub(r"(^-+|-+$)", "", re.sub(r"[^a-z0-9]+", "-", r["state"].lower()))
        rows.append(dict(state=r["state_abbv"], name=r["state"], last_update=r.get("last_update"),
                         data_source=r.get("data_source") if r.get("data_source") not in ("NA", "TBD") else None,
                         uf_url=f"{UF_PAGE}2026-general-election-early-vote-{slug}/", **vals))
    return rows


def tx_listed() -> bool:
    try:
        s = get(TX_PAGE).decode("latin-1")
        return bool(re.search(r"2026 NOVEMBER 3RD GENERAL", s, re.I))
    except Exception:                                             # noqa: BLE001
        return False


# ---------------------------------------------------------------------------------------------------------------- main
def update() -> int:
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev_off = {s["state"]: s for s in prev.get("official", [])}
    base = json.loads(BASE.read_text()) if BASE.exists() else {}
    official, errors = [], []
    for code, fn in (("NC", nc_now), ("PA", pa_now)):             # sequential, one pass a day
        try:
            s = fn()
        except Exception as e:                                    # noqa: BLE001  keep yesterday's numbers, marked stale
            errors.append(f"{code}: {str(e)[:160]}")
            s = dict(prev_off.get(code, {}), stale=True) if code in prev_off else None
        if s:
            if base.get(code):
                s["baseline_2022"] = same_point(base[code], s["asof"])
            official.append(s)
    uf, uf_err = [], None
    try:
        uf = uf_rows()
    except Exception as e:                                        # noqa: BLE001
        uf_err = str(e)[:160]
        uf = prev.get("uf", [])
    cross = {r["state"]: {k: r[k] for k in ("voted_all", "accept_all", "inperson_all", "last_update")} for r in uf if r["state"] in OFFICIAL}
    for s in official:
        s["uf_crosscheck"] = cross.get(s["state"])
    uf_states = [r for r in uf if r["state"] not in OFFICIAL]
    reporting = [r for r in uf_states if (r.get("voted_all") or 0) > 0 or (r.get("request_all") or 0) > 0]
    doc = dict(generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), election=E26.isoformat(),
               baseline_election=E22.isoformat(), days_to_election=(E26 - dt.date.today()).days,
               official=official, official_pending={"TX": "listed" if tx_listed() else "not yet listed by the SOS"},
               uf=uf_states, uf_reporting=[r["state"] for r in reporting], uf_error=uf_err, errors=errors,
               licence_uf="CC BY-NC-ND 4.0 - shown verbatim, attributed; no derived statistics",
               blocked_official={"GA": "sos.ga.gov 403", "FL": "public stats behind reCAPTCHA", "NV": "nvsos.gov bot wall",
                                 "MI": "mvic bot challenge", "WI": "elections.wi.gov 403", "AZ": "azsos.gov 403"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    for s in official:
        b = s.get("baseline_2022") or {}
        print(f"{s['state']} as of {s['asof']}: total {s['total']:,} (mail {s['mail_returned']:,}, in person {s['in_person']:,})"
              + (f"; 2022 same point ({b['date_2022']}) {b['total']:,}" if b else ""))
    print(f"UF: {len(reporting)} other states reporting; errors: {errors or 'none'}")
    return 0


def baseline() -> int:
    base = json.loads(BASE.read_text()) if BASE.exists() else {}
    print("PA 2022 ...", flush=True); base["PA"] = pa_baseline()
    print("NC 2022 (100 MB voter file, reduced in memory) ...", flush=True); base["NC"] = nc_baseline()
    base["_meta"] = dict(built=dt.date.today().isoformat(), note="2022 general: cumulative counts by date x method x party; "
                         "NC accepted absentee/one-stop ballots from the SBE voter file, PA mail requests/returns from data.pa.gov")
    BASE.write_text(json.dumps(base, separators=(",", ":")))
    print(f"wrote {BASE} ({BASE.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit({"update": update, "baseline": baseline}[sys.argv[1] if len(sys.argv) > 1 else "update"]())
