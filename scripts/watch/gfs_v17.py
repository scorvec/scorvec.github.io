#!/usr/bin/env python3
"""Watch for news of the GFS v17 upgrade; gfs-v17-watch.yml runs it daily and opens an issue when something is new.

Sources, each a single small page or listing a run:
  nws     weather.gov/notification: a notice (SCN, TIN, PNS) on the GFS or GDAS; MOS and LAMP guidance notices are left out
  nco     the NCO implementation schedule: a GFS or GDAS row (a new row, or an edited one)
  emc     the EMC GFSv17 evaluation page: its timeline table (evaluation, code handoff, IT test, implementation date)
          and each dated announcement
  s3      the newest 00Z GFS cycle in the AWS Open Data bucket noaa-gfs-bdp-pds (NOAA models are read from S3, never
          NOMADS): a sub-directory other than atmos/ and wave/ (the coupled v17 adds ocean and ice output)
Every signal has a stable id. Ids in the baseline file (the state when the watch began) or already named in an issue
labelled LABEL are not reported again; the rest go into ONE issue per run that mentions WATCH_MENTION, so GitHub
notifies that account.

    python scripts/watch/gfs_v17.py              # print the new signals as JSON lines
    python scripts/watch/gfs_v17.py --issues     # ... and open the issue (gh CLI; GH_TOKEN, GH_REPO)
    python scripts/watch/gfs_v17.py --baseline   # record every current id as known (refused if a source fails)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / "gfs_v17_baseline.json"
LABEL = "gfs-v17"
UA = {"User-Agent": "scorvec-gfs-watch/1.0 (+https://scorvec.com)"}
NWS = "https://www.weather.gov/notification/"
NCO = "https://www.nco.ncep.noaa.gov/pmb/changes/"
EMC = "https://www.emc.ncep.noaa.gov/users/meg/gfsv17/"
S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GFS = re.compile(r"GFS|GDAS|Global Forecast System|Global Data Assimilation", re.I)
GUIDANCE = re.compile(r"\b(MOS|LAMP)\b")  # statistical guidance built on the GFS, not the model itself
MARK = re.compile(r"<!-- watch-id: (.+?) -->")


def get(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def plain(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def digest(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def signal(sid: str, title: str, url: str) -> dict:
    return dict(id=sid, title=title, url=url)


def nws() -> list[dict]:
    out = []
    for href, inner in re.findall(r'(?is)<a\s[^>]*href="([^"]+\.pdf)"[^>]*>(.*?)</a>', get(NWS)):
        name, title = href.rsplit("/", 1)[-1], plain(inner)
        if GFS.search(f"{title} {name}") and not GUIDANCE.search(title):
            out.append(signal(f"nws:{name}", title, urllib.parse.urljoin(NWS, href)))
    return out


def nco() -> list[dict]:
    out = []
    for row in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", get(NCO)):
        t = plain(row)
        if GFS.search(t) and not GUIDANCE.search(t):
            out.append(signal(f"nco:{digest(t)}", f"NCO implementation schedule: {t[:200]}", NCO))
    return out


def emc() -> list[dict]:
    page = get(EMC)
    out = [
        signal(f"emc:timeline:{digest(plain(t))}", f"EMC GFSv17 timeline: {plain(t)[:300]}", EMC)
        for t in re.findall(r"(?is)<table.*?</table>", page)
        if "implementation" in t.lower()
    ]
    for day, rest in re.findall(r"(?is)<b>\s*(\d{1,2}/\d{1,2}/\d{2})\s*</b>\s*-\s*(.*?)(?:<br|</li|</p)", page):
        t = plain(rest)
        out.append(signal(f"emc:news:{day}:{digest(t)}", f"EMC GFSv17 page, {day}: {t[:200]}", EMC))
    return out


def s3() -> list[dict]:
    today = dt.datetime.now(dt.UTC).date()
    for back in range(3):
        day = f"{today - dt.timedelta(days=back):%Y%m%d}"
        subs = re.findall(
            rf"<Prefix>gfs\.{day}/00/([^/<]+)/</Prefix>", get(f"{S3}/?list-type=2&delimiter=/&prefix=gfs.{day}/00/")
        )
        if subs:
            return [
                signal(
                    f"s3:{s}", f"New output directory in the GFS cycle on AWS: {s}/", f"{S3}/index.html#gfs.{day}/00/"
                )
                for s in subs
                if s not in ("atmos", "wave")
            ]
    raise RuntimeError("no 00Z GFS cycle listed in the last three days")


SOURCES = (nws, nco, emc, s3)


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def issue_ids() -> set[str]:
    bodies = json.loads(gh("issue", "list", "--label", LABEL, "--state", "all", "--limit", "1000", "--json", "body"))
    return {m for b in bodies for m in MARK.findall(b["body"] or "")}


def open_issue(new: list[dict]) -> None:
    gh("label", "create", LABEL, "--force", "--color", "1d76db", "--description", "GFS v17 upgrade watch")
    lines = [f"- {s['title']}\n  {s['url']}" for s in new]
    mention = os.environ.get("WATCH_MENTION", "")
    body = "\n".join(
        [
            "New signals about the GFS v17 upgrade:",
            "",
            *lines,
            "",
            (f"@{mention} " if mention else "") + "(found by scripts/watch/gfs_v17.py)",
            "",
            *(f"<!-- watch-id: {s['id']} -->" for s in new),
        ]
    )
    title = f"GFS v17 watch: {new[0]['title'][:150]}" + (f" (+{len(new) - 1} more)" if len(new) > 1 else "")
    print(gh("issue", "create", "--title", title, "--body", body, "--label", LABEL).strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--issues", action="store_true", help="open an issue for the new signals")
    ap.add_argument("--baseline", action="store_true", help="record every current signal as known")
    a = ap.parse_args()
    found, failed = [], []
    for source in SOURCES:
        try:
            found += source()
        except Exception as e:  # noqa: BLE001  one unreachable source must not hide the others
            failed.append(source.__name__)
            print(f"::warning::{source.__name__}: {e}", file=sys.stderr)
    if a.baseline:
        if failed:
            print(f"baseline not written: {', '.join(failed)} could not be read", file=sys.stderr)
            return 1
        BASELINE.write_text(json.dumps(sorted({s["id"] for s in found}), indent=1) + "\n")
        print(f"{len(found)} signals recorded in {BASELINE.name}")
        return 0
    known = set(json.loads(BASELINE.read_text()))
    if a.issues:
        known |= issue_ids()
    new = list({s["id"]: s for s in found if s["id"] not in known}.values())
    for s in new:
        print(json.dumps(s))
    if new and a.issues:
        open_issue(new)
    print(f"{len(found)} signals, {len(new)} new, sources failed: {', '.join(failed) or 'none'}", file=sys.stderr)
    return 1 if len(failed) == len(SOURCES) else 0


if __name__ == "__main__":
    raise SystemExit(main())
