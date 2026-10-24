#!/usr/bin/env python3
"""GitHub traffic of the public repositories, kept beyond GitHub's 14-day window, for stats.html.

GitHub serves page views per day for the last 14 days only (Insights -> Traffic). Each run merges those days into
assets/site_stats/github_traffic.json, so the history accumulates; the newest day is partial and the next run
overwrites it. Clones are not kept: GitHub counts every clone and fetch, so they measure this site's own GitHub Actions
jobs (hundreds a day, several fetches each) and the `pip install git+...` of its workflows, not people.

Every public, non-fork, unarchived repository of OWNER is read, so a new one is picked up by itself; one the token
cannot read is skipped with a warning and keeps its history. Runs daily from .github/workflows/site-stats.yml. Needs
GH_TRAFFIC_TOKEN, a fine-grained token with "Administration: read" on the repositories (the workflow's own token cannot
read traffic); without it the script says so and exits 0.

    GH_TRAFFIC_TOKEN=... python scripts/site/github_traffic.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "assets" / "site_stats" / "github_traffic.json"
OWNER = "scorvec"
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


def public_repos(token: str) -> list[str]:
    """OWNER's public repositories, forks and archived ones left out."""
    rows = get(f"/users/{OWNER}/repos?type=owner&per_page=100", token)
    return sorted(r["name"] for r in rows if not (r["private"] or r["fork"] or r["archived"]))


def update(doc: dict, repo: str, token: str) -> None:
    """Merge one repository's 14 daily rows of page views, and its current stars and forks, into doc."""
    base = f"/repos/{OWNER}/{repo}"
    meta = get(base, token)
    views = get(f"{base}/traffic/views?per=day", token)
    r = doc["repos"].setdefault(repo, {"days": {}})
    for row in views["views"]:
        r["days"][row["timestamp"][:10]] = {"views": row["count"], "views_unique": row["uniques"]}
    r["days"] = dict(sorted(r["days"].items()))
    r["since"] = next(iter(r["days"]), None)
    r["total"] = {"views": sum(d["views"] for d in r["days"].values())}
    r["last_14d"] = {"views": views["count"], "views_unique": views["uniques"]}
    r["stars"], r["forks"] = meta["stargazers_count"], meta["forks_count"]


def main() -> int:
    token = os.environ.get("GH_TRAFFIC_TOKEN", "")
    if not token:
        print("GH_TRAFFIC_TOKEN not set - GitHub traffic skipped")
        return 0
    doc = json.loads(OUT.read_text()) if OUT.exists() else {"repos": {}}
    for repo in public_repos(token):
        try:
            update(doc, repo, token)
        except urllib.error.HTTPError as e:
            print(f"::warning::{repo}: HTTP {e.code} - not readable with this token; its history is kept")
    if not doc["repos"]:
        print("::warning::no repository's traffic was readable - nothing written")
        return 0
    doc["updated"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%MZ")
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(", ".join(f"{k}: {v['last_14d']['views']} views in 14 days" for k, v in doc["repos"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
