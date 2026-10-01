#!/usr/bin/env python3
"""Merge the two writers of assets/models/hrrr.json (2026-09-30, ECAPE on models.html).

Two GitHub jobs publish the SAME HRRR cycle at different times and in either order:
  models.yml (models-hrrr)   the surface fields          -> manifest["cycles"]
  ecape.yml                  ECAPE / CAPE from wrfnat    -> manifest["ecape"][<cycle>]
Each job owns one part of the manifest and must keep the other's part as it is on main RIGHT NOW (re-read on every
push attempt), so neither overwrites the other whichever runs second. The page joins them by cycle.

ECAPE entries are kept for every cycle at or after the oldest retained model cycle: one for the retained cycle, and one
for a NEWER cycle whose ECAPE arrived before the surface fields (it waits there until the models job publishes that
cycle). Anything older is dropped here, and its files go with the whole cycle directory when the models job prunes it
(publish_data.sh PRUNE / PRUNE_OLDER).

    python scripts/models/manifest_merge.py --part cycles --ours /tmp/ours.json --base main.json --out assets/models/hrrr.json
    python scripts/models/manifest_merge.py --part ecape  --ours /tmp/ecape.json --base main.json --out assets/models/hrrr.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def floor_cycle(man):
    """Oldest retained model cycle ('' when the manifest has none yet)."""
    cyc = [c["cycle"] for c in man.get("cycles", []) if c.get("cycle")]
    return min(cyc) if cyc else ""


def prune_ecape(man, max_pending=2):
    """Drop ECAPE entries older than the oldest retained model cycle; keep at most `max_pending` newer ones."""
    ec = man.get("ecape") or {}
    fl = floor_cycle(man)
    have = {c["cycle"] for c in man.get("cycles", [])}
    keep = {k: v for k, v in ec.items() if k >= fl}
    pending = sorted((k for k in keep if k not in have), reverse=True)
    for k in pending[max_pending:]:
        keep.pop(k)
    man["ecape"] = dict(sorted(keep.items(), reverse=True))
    return [k for k in ec if k not in man["ecape"]]


def merge(part, ours, base):
    base = base or {}
    if part == "cycles":            # the models job: its cycles and header, the ECAPE section as it is on main
        out = dict(ours)
        out["ecape"] = {**(ours.get("ecape") or {}), **(base.get("ecape") or {})}
    elif part == "ecape":           # the ECAPE job: everything as it is on main, plus its own ECAPE entries
        out = dict(base) if base else dict(model="hrrr", label="HRRR", cycles=[])
        out["ecape"] = {**(base.get("ecape") or {}), **(ours.get("ecape") or {})}
    else:
        raise ValueError(part)
    dropped = prune_ecape(out)
    return out, dropped


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:                                           # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["cycles", "ecape"])
    ap.add_argument("--ours", required=True)
    ap.add_argument("--base", required=True, help="the manifest as it is on main now (may be missing)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ours = _load(a.ours)
    if ours is None:
        raise SystemExit(f"cannot read {a.ours}")
    out, dropped = merge(a.part, ours, _load(a.base))
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"merged ({a.part}): cycles {[c['cycle'] for c in out.get('cycles', [])]}, ecape {list(out['ecape'])}"
          + (f", dropped ecape {dropped}" if dropped else ""))


if __name__ == "__main__":
    main()
