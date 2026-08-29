#!/usr/bin/env python3
"""
Assemble the animator manifest from whatever ECAPE frames actually exist.

Deliberately glob-driven rather than driven by the list of forecast hours the
job *intended* to render: a 48 h run is ~29 frames per field and any one of them
can lose its fetch. Building the manifest from the frames on disk means a
partial cycle still animates over the hours it got, instead of the viewer
requesting files that were never written.

Frames live at  <root>/<field>/F<fxx>.webp  and are served from the orphan
`frames` branch (see scripts/lib/publish_frames.sh); the manifest itself stays
on main so the animator fetches it same-origin.

Usage:
  python build_manifest.py assets/ecape/anim --cycle 2026082812
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Order here is the order of the picker buttons in the viewer.
FIELDS = [
    ("ratio_mu", "ECAPE / CAPE — most-unstable"),
    ("ratio_ml", "ECAPE / CAPE — mixed-layer"),
    ("ecape_mu", "ECAPE — most-unstable"),
    ("ecape_ml", "ECAPE — mixed-layer"),
]
FRAME_RE = re.compile(r"^F(\d{2,3})\.webp$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="animation root, e.g. assets/ecape/anim")
    ap.add_argument("--cycle", required=True, help="YYYYMMDDHH of the model run")
    ap.add_argument("--out", default=None,
                    help="manifest path (default: <root>/ecape_manifest.json)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    init = datetime.strptime(a.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)

    regions = {}
    for fid, label in FIELDS:
        d = root / fid
        if not d.is_dir():
            continue
        hours = sorted(int(m.group(1))
                       for m in (FRAME_RE.match(p.name) for p in d.iterdir())
                       if m)
        if not hours:
            continue
        frames = []
        for i, fxx in enumerate(hours):
            valid = init + timedelta(hours=fxx)
            frames.append({
                "idx": i,
                "file": f"F{fxx:02d}.webp",
                "date": valid.strftime("%Y-%m-%d %H:%MZ"),
                # What the scrubber shows: both the lead and the valid time, so
                # a reader can tell "F27" from "Thu 15Z" without arithmetic.
                "label": f"F{fxx:02d} · valid {valid:%a %d %b %HZ}",
            })
        regions[fid] = {"label": label, "n_frames": len(frames), "frames": frames}

    if not regions:
        print(f"no frames under {root}", file=sys.stderr)
        return 1

    manifest = {
        "init": init.strftime("%Y-%m-%d %HZ"),
        # Cache-buster. Frames are force-pushed to identical URLs every cycle
        # (F00.webp is always F00.webp), so without a per-cycle version the
        # browser happily serves yesterday's run from cache. The viewer appends
        # this to every frame request.
        "ver": a.cycle,
        "selectorLabel": "Field",
        "default": FIELDS[0][0],
        "regions": regions,
    }
    out = Path(a.out) if a.out else root / "ecape_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    span = {k: v["n_frames"] for k, v in regions.items()}
    print(f"  wrote {out} — init {manifest['init']}, frames per field: {span}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
