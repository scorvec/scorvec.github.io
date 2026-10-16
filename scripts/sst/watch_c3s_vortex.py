#!/usr/bin/env python3
"""Dispatcher (run outside Actions) for c3s-vortex.yml during a C3S release window (Actions cron is throttled to ~1/3 of firings).

Every 30 min: which centres have the issue in the CDS catalogue (cds_seq.available, no key used), against the centres
the published card already holds (assets/sst/data/c3s_vortex.json on main via raw GitHub API). When the catalogue has
more, and no seas5-group run is queued or running, dispatch c3s-vortex.yml once. For the October issue it also waits for
the daily product (c3s_vortex_daily.json) to reach "complete". Exits when the card is complete, or after the 20th once the daily product is complete (October), at most 75 days.

    nohup python scripts/sst/watch_c3s_vortex.py --issue 202610 >> watch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cds_seq as C

GH = C.GH
DAILY = HERE / "data" / "c3s_vortex_daily"
REPO = C.GH_REPO


def published(path: str) -> dict:
    r = subprocess.run(
        [GH, "api", f"repos/{REPO}/contents/{path}?ref=main"], capture_output=True, text=True, timeout=60, check=False
    )
    if r.returncode != 0:
        return {}
    try:
        return json.loads(base64.b64decode(json.loads(r.stdout)["content"]))
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True)
    ap.add_argument("--every-min", type=float, default=30)
    a = ap.parse_args()
    import c3s_vortex as V
    import c3s_vortex_daily as VD

    y, s = int(a.issue[:4]), int(a.issue[4:])
    key = f"{a.issue[:4]}-{a.issue[4:]}"
    last_dispatch = 0.0
    t_start = time.time()
    while True:
        now = time.gmtime()
        in_window = now.tm_mday <= 20 and (now.tm_year, now.tm_mon) == (y, s)
        if time.time() - t_start > 75 * 86400:
            print(time.strftime("%FT%TZ", now), "75 days; exiting", flush=True)
            return 0
        C._CONSTRAINTS.clear()
        cat = [
            c
            for c, ks in V.CENTRES
            if all(
                C.available("seasonal-monthly-pressure-levels", V._req(V.CUR[k][1], V.CUR[k][2], [y], [s])) for k in ks
            )
        ]
        J = published("assets/sst/data/c3s_vortex.json")
        have = J.get("centres_present", []) if J.get("issue") == key else []
        daily_done = True
        if s == 10:
            D = published("assets/sst/data/c3s_vortex_daily.json")
            daily_done = D.get("issue") == key and D.get("state") == "complete"
        msg = f"{time.strftime('%FT%TZ', now)} catalogue {cat}; card {have}; daily done {daily_done}"
        if (set(have) >= {c for c, _ in V.CENTRES} or not in_window) and daily_done:
            print(msg, "- complete" if in_window else "- window over, daily done", "; exiting", flush=True)
            return 0
        busy = C.group_busy()
        nd = sum(
            (DAILY / f"ud_{u['unit']}.npz").exists() for u in VD.units()
        )  # the daily hindcast tables on disk (240 when whole)
        # after the 20th only the daily product is waited for (the offline hindcast fetch may run past the window)
        want = (in_window and set(cat) - set(have)) or (
            s == 10 and not daily_done and nd >= len(VD.units()) and time.time() - last_dispatch > 6 * 3600
        )
        if want and not busy and time.time() - last_dispatch > 3600:
            r = subprocess.run(
                [GH, "workflow", "run", "c3s-vortex.yml", "--repo", REPO, "-f", f"issue={a.issue}"],
                capture_output=True,
                text=True,
                check=False,
            )
            print(msg, "- dispatched" if r.returncode == 0 else f"- dispatch failed {r.stderr[:120]}", flush=True)
            last_dispatch = time.time()
        else:
            print(msg, "- busy " + "; ".join(busy)[:120] if busy else "- nothing new", flush=True)
        time.sleep(a.every_min * 60)


if __name__ == "__main__":
    raise SystemExit(main())
