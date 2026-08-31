#!/usr/bin/env python3
"""Garbage-collect the orphan `frames` branch.

    python scripts/lib/gc_frames.py              # report only, deletes nothing
    python scripts/lib/gc_frames.py --prune      # delete what RETIRED lists

WHAT IT DOES. Every animation manifest on main names the frame directories its
viewer will request. Anything on the frames branch that no manifest references
is a candidate for deletion - a product that was renamed, replaced or retired
and left its frames behind. On 2026-08-30 that was 1225 frames across 21
directories, including wave1_maps and vortex_winds which had been replaced by
per-hemisphere and per-level loops hours earlier.

WHY IT DOES NOT JUST DELETE THEM. Several products are PRIVATE: their pages and
manifests live only on the laptop and are deliberately not tracked, so from a
runner's point of view their frames look exactly like orphans. Measured on the
same branch, 23 directories were "unreferenced" purely because their manifest
is not on main - cptec, brazil, sfs, gatun. Deleting on inference alone would
have destroyed all of them.

So deletion is opt-in per path: only directories listed in retired_frames.txt
are removed, PROTECT_PREFIXES are never touched whatever else is true, and
everything else is reported so a human can decide. The report is the product
here; the pruning is the small safe part.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RETIRED = HERE / "retired_frames.txt"
BRANCH = os.environ.get("FRAMES_BRANCH", "frames")

# Never pruned, whatever the manifests say. These are the private/local-only
# products whose manifests are intentionally absent from main - the exact set
# that a reference-only rule would have eaten.
PROTECT_PREFIXES = (
    "assets/cptec/anim",
    "assets/brazil/anim",
    "assets/sfs/anim",
    "assets/sst/anim/gatun",
)


def sh(*a: str) -> str:
    return subprocess.run(a, capture_output=True, text=True, cwd=REPO).stdout


def referenced() -> set[str]:
    """Frame directories named by any manifest tracked on main.

    Manifests are enumerated from origin/main's TREE, not from `git ls-files`.
    The index lists what the local checkout knows about, so a working copy that
    has not pulled misses manifests that exist on the remote - and every
    directory they reference then shows up as an orphan. Seen on 2026-08-31: a
    local run reported the newest ECAPE cycle's three directories as
    unreferenced because ecape_2026083100.json was on main but not in this
    index. Nothing was deleted (the retired list gates that), but a report that
    names live data as garbage is one edit away from being acted on.
    """
    out: set[str] = set()
    tree = sh("git", "ls-tree", "-r", "--name-only", "origin/main").split()
    for m in (p for p in tree
              if re.match(r"assets/[^/]+/anim/.*\.json$", p)):
        try:
            d = json.loads(sh("git", "show", f"origin/main:{m}") or "{}")
        except json.JSONDecodeError:
            continue
        root = os.path.dirname(m)
        for rid in (d.get("regions") or {}):
            out.add(f"{root}/{rid}")
    return out


def on_branch() -> dict[str, int]:
    """Frame directories on the branch, with their frame counts."""
    counts: dict[str, int] = {}
    for p in sh("git", "ls-tree", "-r", "--name-only", f"origin/{BRANCH}").split():
        if p.endswith(".webp"):
            counts[os.path.dirname(p)] = counts.get(os.path.dirname(p), 0) + 1
    return counts


def retired_list() -> list[str]:
    if not RETIRED.exists():
        return []
    return [ln.strip() for ln in RETIRED.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", action="store_true",
                    help="actually delete the retired paths (default: report only)")
    a = ap.parse_args()

    sh("git", "fetch", "-q", "origin", BRANCH)
    sh("git", "fetch", "-q", "origin", "main")
    ref, br = referenced(), on_branch()
    retired = retired_list()

    protected = [d for d in br if d.startswith(PROTECT_PREFIXES)]
    orphans = sorted(d for d in br
                     if d not in ref and not d.startswith(PROTECT_PREFIXES))
    to_prune = [d for d in orphans if d in retired]
    review = [d for d in orphans if d not in retired]

    print(f"  branch dirs {len(br)} | referenced {len(ref)} | protected {len(protected)}")
    if to_prune:
        n = sum(br[d] for d in to_prune)
        print(f"\n  RETIRED, will be pruned ({n} frames):")
        for d in to_prune:
            print(f"    {d:46s} {br[d]:5d}")
    if review:
        n = sum(br[d] for d in review)
        print(f"\n  UNREFERENCED, needs a decision ({n} frames) — add to "
              f"{RETIRED.relative_to(REPO)} to have them pruned:")
        for d in sorted(review, key=lambda x: -br[x]):
            print(f"    {d:46s} {br[d]:5d}")
    missing = sorted(d for d in ref if d not in br)
    if missing:
        print(f"\n  REFERENCED BUT ABSENT — a manifest points at frames that do "
              f"not exist ({len(missing)}):")
        for d in missing:
            print(f"    {d}")

    if not a.prune:
        print("\n  report only; nothing deleted (pass --prune to act on the retired list)")
        return 0
    if not to_prune:
        print("\n  nothing on the retired list to prune")
        return 0
    # The caller does the branch surgery: this script never force-pushes, so a
    # bug here cannot take the branch with it.
    print("\nPRUNE_PATHS=" + " ".join(to_prune))
    return 0


if __name__ == "__main__":
    sys.exit(main())
