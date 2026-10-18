#!/usr/bin/env python3
"""Keep refs/frames-pins/<sha> for every frames commit a manifest on main names.

    python scripts/lib/frames_pins.py            # report only
    python scripts/lib/frames_pins.py --apply    # create the missing pins, delete the stale ones

WHY. Manifests pin their frames to a commit ("frames_sha"; the models' JSON carry a "sha" per cycle and field) and the
viewers ask the CDN and jsDelivr for <sha>/<path>. The frames branch is force-pushed as one parentless commit, so each
earlier commit is unreachable the moment the next publish lands, and GitHub garbage-collects unreachable commits: a
product that has not republished since (CanSIPS, monthly) is left pointing at frames that no longer exist. A ref under
refs/frames-pins/ keeps each named commit reachable. publish_frames_ci.sh pushes the pin with the commit; this adds the
ones that are missing (commits written by other publishers) and deletes the ones no manifest names any more.

A pin is deleted only once its commit is older than GRACE_H: the publisher pushes it minutes before its workflow
commits the manifest that names it. The branch itself is never touched here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta

NS = "refs/frames-pins/"
GRACE_H = 48
SHA = re.compile(r'"(frames_sha|sha)": ?"([0-9a-f]{40})"')


def sh(*a: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(a, capture_output=True, text=True, check=check)


def referenced() -> dict[str, int]:
    """sha -> number of manifests on origin/main naming it."""
    sh("git", "fetch", "-q", "origin", "main")
    out = sh("git", "grep", "-c", "-E", '"(frames_sha|sha)": ?"[0-9a-f]{40}"', "origin/main", "--", "*.json").stdout
    refs: dict[str, int] = {}
    for line in out.splitlines():
        path = line.split(":", 2)[1]
        text = sh("git", "show", f"origin/main:{path}").stdout
        # "sha" is the models' key (assets/models/*.json); elsewhere only "frames_sha" names a frames commit
        for key, sha in SHA.findall(text):
            if key == "frames_sha" or path.startswith("assets/models/"):
                refs[sha] = refs.get(sha, 0) + 1
    return refs


def pins() -> set[str]:
    out = sh("git", "ls-remote", "origin", NS + "*")
    if out.returncode:
        sys.exit(f"cannot list the pins: {out.stderr.strip()}")
    return {ln.split("\t")[1][len(NS) :] for ln in out.stdout.splitlines() if "\t" in ln}


def api(*a: str) -> subprocess.CompletedProcess:
    return sh("gh", "api", *a)


def commit_time(repo: str, sha: str) -> datetime | None:
    r = api(f"repos/{repo}/git/commits/{sha}", "-q", ".committer.date")
    if r.returncode:
        return None
    return datetime.fromisoformat(r.stdout.strip().replace("Z", "+00:00"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="create missing pins and delete stale ones")
    a = ap.parse_args()
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        sys.exit("GITHUB_REPOSITORY is required")

    want, have = referenced(), pins()
    missing = sorted(s for s in want if s not in have)
    unused = sorted(s for s in have if s not in want)
    print(
        f"  manifests name {len(want)} frames commit(s); {len(have)} pinned; {len(missing)} to pin; {len(unused)} unnamed"
    )

    created, lost = 0, []
    for s in missing:
        if not a.apply:
            print(f"    would pin {s[:10]} ({want[s]} reference(s))")
            continue
        r = api("-X", "POST", f"repos/{repo}/git/refs", "-f", f"ref={NS}{s}", "-f", f"sha={s}")
        if r.returncode == 0:
            created += 1
            print(f"    pinned {s[:10]} ({want[s]} reference(s))")
        elif commit_time(repo, s) is None:
            lost.append(s)
        else:
            print(f"::warning::could not create {NS}{s[:10]}: {(r.stderr or r.stdout).strip()[:200]}")
    for s in lost:
        print(
            f"::warning::frames commit {s[:10]} is gone from GitHub: {want[s]} reference(s) name it; republish those products"
        )

    cutoff = datetime.now(UTC) - timedelta(hours=GRACE_H)
    stale = []
    for s in unused:
        t = commit_time(repo, s)
        if t is not None and t < cutoff:
            stale.append(s)
    print(f"  {len(stale)} unnamed pin(s) older than {GRACE_H} h")
    if a.apply and stale:
        for i in range(0, len(stale), 100):
            batch = [NS + s for s in stale[i : i + 100]]
            r = sh("git", "push", "-q", "origin", "--delete", *batch)
            if r.returncode:
                print(f"::warning::could not delete {len(batch)} stale pin(s): {r.stderr.strip()[:200]}")
            else:
                print(f"    deleted {len(batch)} stale pin(s)")
    print(
        json.dumps(
            {"named": len(want), "pinned": len(have), "created": created, "lost": len(lost), "stale": len(stale)}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
