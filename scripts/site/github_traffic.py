#!/usr/bin/env python3
"""GitHub traffic of the public repositories, kept beyond GitHub's 14-day window, for stats.html.

GitHub serves clones and page views per day for the last 14 days only (Insights -> Traffic). Each run merges those days
into assets/site_stats/github_traffic.json, so the history accumulates; the newest day is partial and the next run
overwrites it. Clones count every clone and fetch, GitHub Actions checkouts and `pip install git+...` included, so they
mostly measure automation; page views, stars and forks are the human signal.

Runs daily from .github/workflows/site-stats.yml. Needs GH_TRAFFIC_TOKEN, a fine-grained token with "Administration:
read" on the repositories (the workflow's own token cannot read traffic); without it the script says so and exits 0.

    GH_TRAFFIC_TOKEN=... python scripts/site/github_traffic.py
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "assets" / "site_stats" / "github_traffic.json"
OWNER = "scorvec"
REPOS = ("scorvec.github.io", "quantmet-tools", "midterms-2026")
API = "https://api.github.com"


def get(path: str, token: str) -> dict:
    req = urllib.request.Request(
        API + path,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "scorvec-site-stats",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def update(doc: dict, repo: str, token: str) -> None:
    """Merge one repository's 14 daily rows of clones and views, and its current stars / forks / watchers, into doc."""
    base = f"/repos/{OWNER}/{repo}"
    meta = get(base, token)
    clones = get(f"{base}/traffic/clones?per=day", token)
    views = get(f"{base}/traffic/views?per=day", token)
    r = doc["repos"].setdefault(repo, {"days": {}})
    for kind, rows in (("clones", clones["clones"]), ("views", views["views"])):
        for row in rows:
            day = r["days"].setdefault(row["timestamp"][:10], {})
            day[kind], day[f"{kind}_unique"] = row["count"], row["uniques"]
    r["days"] = dict(sorted(r["days"].items()))
    r["since"] = next(iter(r["days"]), None)
    r["total"] = {k: sum(d.get(k, 0) for d in r["days"].values()) for k in ("clones", "views")}
    r["last_14d"] = {
        "clones": clones["count"],
        "clones_unique": clones["uniques"],
        "views": views["count"],
        "views_unique": views["uniques"],
    }
    r["stars"], r["forks"], r["watchers"] = meta["stargazers_count"], meta["forks_count"], meta["subscribers_count"]


def main() -> int:
    token = os.environ.get("GH_TRAFFIC_TOKEN", "")
    if not token:
        print("GH_TRAFFIC_TOKEN not set - GitHub traffic skipped")
        return 0
    doc = json.loads(OUT.read_text()) if OUT.exists() else {"repos": {}}
    for repo in REPOS:
        update(doc, repo, token)
    doc["updated"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%MZ")
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(
        ", ".join(
            f"{k}: {v['last_14d']['views']} views / {v['last_14d']['clones']} clones" for k, v in doc["repos"].items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
