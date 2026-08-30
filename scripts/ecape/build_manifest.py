#!/usr/bin/env python3
"""
Assemble the animator manifest from whatever ECAPE frames actually exist.

Deliberately glob-driven rather than driven by the list of forecast hours the
job *intended* to render: a 48 h run is ~29 frames per field and any one of them
can lose its fetch. Building the manifest from the frames on disk means a
partial cycle still animates over the hours it got, instead of the viewer
requesting files that were never written.

Frames live at  <root>/<cycle>/<field>/F<fxx>.webp  and are served from the
orphan `frames` branch; the manifest itself stays on main so the animator
fetches it same-origin.

CYCLE-SCOPED since 2026-08-30, so several runs can be kept side by side and the
page can offer a run picker. Region ids are "<cycle>/<field>" because the viewer
builds a frame URL as <base>/<region>/<file> - putting the cycle in the region
id makes the archive work with no viewer change at all.

Also maintains <root>/index.json, the list of archived cycles newest-first, which
is what the run picker reads.

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
    cdir = root / a.cycle

    regions = {}
    for fid, label in FIELDS:
        d = cdir / fid
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
        regions[f"{a.cycle}/{fid}"] = {"label": label,
                                       "n_frames": len(frames), "frames": frames}

    if not regions:
        print(f"no frames under {cdir}", file=sys.stderr)
        return 1

    manifest = {
        "init": init.strftime("%Y-%m-%d %HZ"),
        # Cache-buster. Frames are force-pushed to identical URLs every cycle
        # (F00.webp is always F00.webp), so without a per-cycle version the
        # browser happily serves yesterday's run from cache. The viewer appends
        # this to every frame request.
        "ver": a.cycle,
        "selectorLabel": "Field",
        "default": f"{a.cycle}/{FIELDS[0][0]}",
        "regions": regions,
    }
    out = Path(a.out) if a.out else root / f"ecape_{a.cycle}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    span = {k.split("/")[-1]: v["n_frames"] for k, v in regions.items()}
    print(f"  wrote {out} — init {manifest['init']}, frames per field: {span}")

    # index.json: every cycle that still has frames on disk, newest first. The
    # page's run picker reads this, so a cycle pruned for retention disappears
    # from the picker in the same step that removes its frames.
    cycles = sorted((d.name for d in root.iterdir()
                     if d.is_dir() and re.fullmatch(r"\d{10}", d.name)
                     and any(d.glob("*/F*.webp"))), reverse=True)
    idx = {"cycles": [{"cycle": c,
                       "init": datetime.strptime(c, "%Y%m%d%H")
                               .replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %HZ"),
                       "manifest": f"ecape_{c}.json"} for c in cycles],
           "latest": cycles[0] if cycles else None}
    (root / "index.json").write_text(json.dumps(idx, indent=2))
    print(f"  index.json: {len(cycles)} cycle(s) archived — {', '.join(cycles) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
