#!/usr/bin/env python3
"""The Siberian High and the East Asian trough in the newest run of each ensemble, for the North Pacific jet's
extension watch (circulation.html#pacjet): the two box indices whose state changed the odds of a jet extension in ERA5
(scripts/mjo/src/build_pacjet_drivers.py, which writes their boxes, day-of-year normals and sigma to
pacjet_drivers.json). No download: the fields are the ones the teleconnection job reads (ensmaps_telecon.py), each
member's 500 hPa height and sea-level pressure on the 2.5 degree grid north of 20N at days 1-15.

An index is the box mean (cos-latitude weights; the reference's ERA5 fields were averaged onto the same grid first),
for the trough less the mean north of 20N, less the normal of the valid day, over the November-March sigma. The
ensemble mean is given at a lead where at least MIN_SHARE of the members are complete.

    python scripts/ensmaps/jet_upstream.py --fields f/*.npz --boxes pacjet_drivers.json --out pacjet_upstream.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[0] / "lib")]
from ensmaps_telecon import MIN_SHARE, SAVED, load
from latlon import G25N, box_mean, slot366


def indices(run: dict, boxes: dict) -> dict:
    """One run's ensemble-mean indices by lead, in sigma (None at a lead without enough complete members)."""
    t = pd.DatetimeIndex(run["init"] + pd.to_timedelta(run["lead_h"].astype(int), unit="h"))
    out = {"init": f"{run['init']:%Y-%m-%dT%HZ}", "valid": [f"{v:%Y-%m-%d}" for v in t]}
    for k, b in boxes.items():
        key, ok_key, scale = SAVED[b["field"]]
        if key not in run:
            continue
        x = run[key] * scale
        v = box_mean(x, G25N, b["lat"], b["lon"]) - (box_mean(x, G25N, b["less"]) if b["less"] else 0.0)
        z = (v - np.asarray(b["normal"])[slot366(t)]) / b["sd"]
        ok = run[ok_key]
        enough = ok.sum(0) >= np.ceil(MIN_SHARE * len(ok))
        mean = np.where(ok, z, 0.0).sum(0) / np.maximum(ok.sum(0), 1)
        out[k] = [round(float(m), 2) if e else None for m, e in zip(mean, enough)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", nargs="+", type=Path, required=True)
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    boxes = json.loads(a.boxes.read_text())["boxes"]
    doc = {
        "generated": f"{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M UTC}",
        "models": {r["key"]: indices(r, boxes) for r in load(a.fields)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, separators=(",", ":")))
    for k, m in doc["models"].items():
        print(f"  {k} {m['init']}: " + ", ".join(f"{b} {m[b][0]}" for b in boxes if b in m), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
