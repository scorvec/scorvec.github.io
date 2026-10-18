#!/usr/bin/env python3
"""How complete is a C3S issue? Writes assets/sst/data/c3s_status.json and prints key=value lines
for $GITHUB_OUTPUT (c3s.yml gates every part on it, so a run after the issue is complete costs a
checkout and this script, nothing else).

Parts and what "done" means:
  plume   enso_forecast.json holds this issue with every system the page keeps (5)
  maps    c3s_maps.json holds this issue with every system x field (7 x 4); a field is listed once it is
          tested, i.e. once its start month's hindcast spread (c3s_ref hc_anom) is in
  terc    every system x variable has its issue table (c3s_ref is_terc, 7 x 2)
  sixh    every system x region x forecast month has its daily table (c3s_ref is_sixh, 7 x 2 x 6)

    python scripts/sst/c3s_status.py --issue 202610 [--write]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
DATA = SITE / "assets" / "sst" / "data"
OUT = DATA / "c3s_status.json"

# the registry, read without importing the modules (they pull cdsapi/xarray)
from c3s_nino34 import MERGE, MODELS  # (light: numpy/pandas/xarray only)
from enso_models import EXCLUDE_SYSTEMS

REGIONS = ("us", "br")
NMONTH = 6


def expected_plume() -> int:
    parts = {p for ps in MERGE.values() for p in ps}
    labels = [m[2] for m in MODELS if m[2] not in EXCLUDE_SYSTEMS]
    n = len([lab for lab in labels if lab not in parts])
    n += sum(1 for merged, ps in MERGE.items() if merged not in EXCLUDE_SYSTEMS and any(p in labels for p in ps))
    return n


def _json(name: str) -> dict:
    try:
        return json.loads((DATA / name).read_text())
    except Exception:  # noqa: BLE001
        return {}


def status(issue: str) -> dict:
    import c3s_ref as CR

    key = f"{issue[:4]}-{issue[4:6]}"
    F = _json("enso_forecast.json")
    plume = len(F.get("models", {})) if F.get("issue") == key else 0
    flav = _json("enso_flavour.json").get("issue") == key
    M = _json("c3s_maps.json")
    maps = (
        sum(len(m.get("fields", [])) for m in M.get("models", []) if m.get("id") != "mmm")
        if M.get("issue") == key
        else 0
    )
    systems = [(m[0], str(m[1])) for m in MODELS]
    terc = sum(CR.is_terc(c, s, v, issue).exists() for c, s in systems for v in ("t2m", "tp"))
    sixh = sum(
        CR.is_sixh(c, s, r, issue, k).exists() for c, s in systems for r in REGIONS for k in range(1, NMONTH + 1)
    )
    want = {
        "plume": expected_plume(),
        "maps": len(systems) * 4,
        "terc": len(systems) * 2,
        "sixh": len(systems) * len(REGIONS) * NMONTH,
    }
    have = {"plume": plume, "maps": maps, "terc": terc, "sixh": sixh}
    done = {k: have[k] >= want[k] for k in want}
    done["plume"] = done["plume"] and flav
    return {"issue": key, "have": have, "want": want, "done": done, "complete": all(done.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True)
    ap.add_argument("--write", action="store_true", help="write assets/sst/data/c3s_status.json")
    a = ap.parse_args()
    st = status(a.issue)
    for k, v in st["done"].items():
        print(f"{k}_done={'1' if v else '0'}")
    print(f"complete={'1' if st['complete'] else '0'}")
    print(f"summary={json.dumps(st['have'])} of {json.dumps(st['want'])}", file=sys.stderr)
    if a.write:
        st["updated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(st, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
