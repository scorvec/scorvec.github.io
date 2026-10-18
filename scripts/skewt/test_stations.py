#!/usr/bin/env python3
"""Offline checks on skewt/stations.json: positions of active stations, and reused WMO indexes.

python scripts/skewt/test_stations.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mirror_soundings as M

STATIONS = HERE.parents[1] / "skewt" / "stations.json"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    st = json.loads(STATIONS.read_text())["stations"]
    active = [s for s in st if s.get("y1", 0) >= 2020]

    # A latitude of exactly 0 or a longitude within 0.05 deg of 0 has so far always been a dropped
    # coordinate in the source list, never a real station; the mirror and the AIFS verification
    # would place the ascent in the wrong ocean.
    bad = [(s["id"], s["n"], s["la"], s["lo"]) for s in active if s["la"] == 0 or abs(s["lo"]) < 0.05]
    check(not bad, f"active stations with a dropped coordinate: {bad}")
    check(all(-90 <= s["la"] <= 90 and -180 <= s["lo"] <= 180 for s in st), "position out of range")

    by_wmo, _, _ = M.load_stations()
    for sid in {s["id"] for s in st if s.get("id")}:
        latest = max((s for s in st if s.get("id") == sid), key=lambda s: s.get("y1", 0))
        check(by_wmo[sid]["gid"] == latest["gid"], f"WMO {sid} resolves to an older station")
    check(by_wmo["94775"]["n"] == "Tocal Aws", "94775 should be the active Tocal AWS, not Point Judith (1944-46)")
    print(f"ok: {len(st)} stations, {len(active)} active")


if __name__ == "__main__":
    main()
