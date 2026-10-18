"""Restore missing RMM loop frames from git history (maintenance; run by mjo-restore-frames.yml).

The loop on mjo.html shows the newest KEEP cycles of assets/mjo/rmm_YYYYMMDD_HHz.webp: one file per
cycle, PNG before 2026-10-16, the WebP where a cycle has both. Until 2026-10-03 mjo.yml pruned with `ls -t`, whose
name tie-break on equal checkout mtimes deleted the NEWEST cycles, so the loop kept August runs and lost most of late
September (and re-plotting a past cycle dropped the latest frame). This puts back,
from the last commit that added or modified each file, every frame among the newest KEEP cycle names the history has
ever held, then prunes by name to KEEP. Needs the history of assets/mjo (a blobless fetch is enough: `git show` pulls
the few blobs it needs). Writes nothing else; generate_page.py rebuilds the manifest and page afterwards.

    python scripts/mjo/restore_frames.py [--keep 60] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "mjo"
PREFIXES = ("rmm",)


def git(*args, binary=False):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=not binary, check=True)
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for pre in PREFIXES:
        pat = re.compile(rf"^assets/mjo/{pre}_(\d{{8}}_(?:00|12)z)\.(webp|png)$")
        ever = {}  # cycle -> the path to restore it from, the WebP where the history has both
        for ln in git(
            "log",
            "--diff-filter=AM",
            "--name-only",
            "--format=",
            "--",
            f"assets/mjo/{pre}_*z.webp",
            f"assets/mjo/{pre}_*z.png",
        ).splitlines():
            m = pat.match(ln.strip())
            if m and (m[1] not in ever or m[2] == "webp"):
                ever[m[1]] = ln.strip()
        want = [ever[c] for c in sorted(ever, reverse=True)[: a.keep]]
        restored = 0
        for path in want:
            p = ROOT / path
            if any((ASSETS / f"{pre}_{pat.match(path)[1]}.{x}").exists() for x in ("webp", "png")):  # either format
                continue
            sha = git("log", "-n1", "--format=%H", "--diff-filter=AM", "--", path).strip()
            if not sha:
                continue
            if not a.dry_run:
                p.write_bytes(git("show", f"{sha}:{path}", binary=True))
            restored += 1
            print(f"  {pre}: restored {p.name} from {sha[:9]}")
        have = {}  # cycle -> files on disk
        for q in ASSETS.glob(f"{pre}_*z.*"):
            m = pat.match(f"assets/mjo/{q.name}")
            if m:
                have.setdefault(m[1], []).append(q)
        cycles = sorted(have, reverse=True)
        drop = [q for c in cycles[a.keep :] for q in have[c]]
        drop += [q for c in cycles[: a.keep] if len(have[c]) > 1 for q in have[c] if q.suffix == ".png"]
        for q in drop:
            if not a.dry_run:
                q.unlink()
        print(
            f"{pre}: {len(ever)} cycles in history, {restored} restored, {len(drop)} older than the newest "
            f"{a.keep} pruned, {min(len(cycles), a.keep)} kept ({'dry run' if a.dry_run else 'written'})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
