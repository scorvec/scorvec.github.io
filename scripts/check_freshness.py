#!/usr/bin/env python3
"""Daily sentinel: is every data product still updating?

For each configured path, ask the GitHub API for the last commit touching
it on its branch. Anything older than its max age is reported; the workflow
turns the report into a single maintained issue. Paths that don't exist yet
are skipped with a warning (config can lead the pipeline).
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

REPO = os.environ.get("GITHUB_REPOSITORY", "scorvec/scorvec.github.io")
TOK = os.environ.get("GITHUB_TOKEN", "")


def last_commit_age_h(path: str, branch: str) -> float | None:
    url = (f"https://api.github.com/repos/{REPO}/commits"
           f"?path={path}&sha={branch}&per_page=1")
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        **({"Authorization": f"Bearer {TOK}"} if TOK else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    if not data:
        return None
    when = datetime.fromisoformat(
        data[0]["commit"]["committer"]["date"].replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - when).total_seconds() / 3600


def main() -> int:
    cfg = json.load(open(os.path.join(os.path.dirname(__file__),
                                      "freshness_config.json")))
    stale, ok = [], 0
    for path, spec in cfg["checks"].items():
        try:
            age = last_commit_age_h(path, spec["branch"])
        except Exception as e:  # noqa: BLE001
            stale.append(f"- `{path}` ({spec['branch']}): check FAILED — {e}")
            continue
        if age is None:
            print(f"  warn: {path} has no commits on {spec['branch']} — skipped")
            continue
        if age > spec["max_age_hours"]:
            stale.append(f"- `{path}` ({spec['branch']}): last updated "
                         f"{age:.0f} h ago (limit {spec['max_age_hours']} h)")
        else:
            ok += 1
            print(f"  ok: {path} — {age:.1f} h")
    print(f"{ok} fresh, {len(stale)} stale")
    if stale:
        with open("stale_report.md", "w") as f:
            f.write("The freshness sentinel found data products that have "
                    "stopped updating:\n\n" + "\n".join(stale) +
                    "\n\nCheck the corresponding workflow's recent runs.\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
