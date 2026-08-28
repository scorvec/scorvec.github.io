#!/usr/bin/env python3
"""Merge freshly-built wind-speed climo keys into the published skewt-climo
branch — strictly additive, so ECAPE/SHIP (which need the native SHARPlib
helper this laptop doesn't have built) and every other existing key stay
byte-identical.

Flow:
  1. python scripts/skewt/build_climo.py FRESH_DIR --all     # hours; cache-resumable
  2. python scripts/skewt/add_wind_climo.py FRESH_DIR        # this script
       - checks out skewt-climo into a temp worktree
       - for every fresh {gid}.json, injects idx["850spd"]/["250spd"]
         into the published JSON (doy grids are identical by construction)
       - commits + force-pushes the branch (single parentless commit,
         matching the mirror convention)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WIND_KEYS = ("850spd", "250spd")


def main() -> int:
    fresh = Path(sys.argv[1])
    files = sorted(fresh.glob("*.json"))
    if not files:
        print(f"no fresh JSONs in {fresh}")
        return 1
    wt = Path(tempfile.mkdtemp(prefix="skewt_climo_wt_"))
    subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin", "skewt-climo"], check=True)
    subprocess.run(["git", "-C", str(REPO), "worktree", "add", "-q", "--detach",
                    str(wt), "origin/skewt-climo"], check=True)
    try:
        merged = skipped = 0
        for f in files:
            pub = wt / "climo" / f.name
            if not pub.exists():
                skipped += 1
                continue
            new = json.loads(f.read_text())
            if not all(k in new.get("idx", {}) for k in WIND_KEYS):
                skipped += 1
                continue
            cur = json.loads(pub.read_text())
            for k in WIND_KEYS:
                cur["idx"][k] = new["idx"][k]
            pub.write_text(json.dumps(cur, separators=(",", ":")))
            merged += 1
        print(f"merged wind keys into {merged} station JSONs ({skipped} skipped)")
        if not merged:
            return 1
        subprocess.run(["git", "-C", str(wt), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(wt), "-c", "user.name=Shawn Corvec",
                        "-c", "user.email=scorvec@outlook.com", "commit", "-q",
                        "-m", f"Climatology: add 850/250 hPa wind speed ({merged} stations)"],
                       check=True)
        subprocess.run(["git", "-C", str(wt), "push", "-f", "origin",
                        "HEAD:skewt-climo"], check=True)
        print("pushed skewt-climo")
    finally:
        subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(wt)],
                       check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
