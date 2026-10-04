#!/usr/bin/env python3
"""Bundle complete UTC days of mirrored soundings into the permanent archive.

Reads per-launch files ({id}_{YYYYMMDDHH}.csv) from the mirror output and
writes one zip per COMPLETE day (day-YYYYMMDD.zip) into the skewt-archive
branch checkout — append-only, committed once, never rewritten, so pushes
stay incremental and the browser can fetch any day via raw.githubusercontent
(CORS-open) and unzip with fflate.

    python scripts/skewt/archive_days.py SOUNDINGS_DIR ARCHIVE_DIR
"""
from __future__ import annotations

import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

LAG_DAYS = 2


def main() -> int:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    # Bundles hold the mirror's IEM (real-time) launches only. The mirror re-asks IEM
    # about each hour for ~30 h (late stations), so a day is final LAG_DAYS later -
    # still inside the 96 h retention, so every launch of the day is on disk.
    last = (datetime.now(timezone.utc) - timedelta(days=LAG_DAYS)).strftime("%Y%m%d")

    by_day: dict = {}
    for f in src.glob("*_*.csv"):
        day = f.stem.split("_")[1][:8]
        if day <= last:                                   # only complete days
            by_day.setdefault(day, []).append(f)

    made = 0
    for day, files in sorted(by_day.items()):
        out = dst / f"day-{day}.zip"
        if out.exists():
            continue
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(files):
                z.write(f, f.name)
        print(f"  archived {out.name}: {len(files)} soundings, "
              f"{out.stat().st_size // 1024} KB", flush=True)
        made += 1
    print(f"  {made} new day bundle(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
