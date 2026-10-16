#!/usr/bin/env python3
"""Remove SEAS5 figures that no published SEAS5 data file names any more.

Each issue writes its figures under issue-stamped names (e.g. ``seas5_map_t2m_anom_2026_10.webp``) and lists them in
``assets/sst/data/seas5_*.json``; the previous issue's figures are then orphans. A figure is removed only when no
``seas5_*.json`` mentions its file name, so a product that was not rebuilt in a partial run keeps its figures.
The regime maps are named by the page itself and are always kept.

    python scripts/sst/seas5_prune.py            # list, then ``git rm`` the orphans
    python scripts/sst/seas5_prune.py --dry-run  # list only
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEEP = re.compile(r"^seas5_regimes_[a-z]+\.webp$")  # built from the sector name in the page


def orphans() -> list[str]:
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "assets/sst/seas5_*.webp"], capture_output=True, text=True, check=True
    ).stdout.split()
    named: set[str] = set()
    for f in (ROOT / "assets" / "sst" / "data").glob("seas5_*.json"):
        named |= set(re.findall(r"seas5_[A-Za-z0-9_\-]+\.webp", f.read_text()))
    return [f for f in tracked if Path(f).name not in named and not KEEP.match(Path(f).name)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    gone = orphans()
    for f in gone:
        print(f"  orphan: {f}")
    print(f"{len(gone)} SEAS5 figure(s) no longer named by any seas5_*.json")
    if gone and not a.dry_run:
        subprocess.run(["git", "-C", str(ROOT), "rm", "-q", "--", *gone], check=True)


if __name__ == "__main__":
    main()
