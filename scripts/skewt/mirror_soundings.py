#!/usr/bin/env python3
"""Mirror the latest University of Wyoming soundings for the Skew-T Explorer.

weather.uwyo.edu serves global BUFR/GTS soundings but sends no CORS headers,
so a browser can't fetch it directly. This job (GitHub Actions, ~4x/day)
pulls UW's own per-hour station manifest (/wsgi/sounding_json) plus each
station's TEXT:CSV profile, thins it, and publishes to the `skewt-data`
branch — raw.githubusercontent.com serves it with `Access-Control-Allow-
Origin: *`, which makes the viewer purely client-side.

Retention: a station keeps its most recent profile for 36 h, so 00Z-only
sites stay clickable between launches.

    python scripts/skewt/mirror_soundings.py OUTDIR
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

UW = "https://weather.uwyo.edu/wsgi"
RAW_PREV = ("https://raw.githubusercontent.com/scorvec/scorvec.github.io/"
            "skewt-data/manifest.json")
MAX_LEVELS = 260
RETAIN_H = 36
WORKERS = 6


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent":
        "scorvec.com skew-t mirror (contact: site owner; ~4 fetches/day/station)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def synoptic_hours(n: int = 2):
    now = datetime.now(timezone.utc) - timedelta(hours=2)
    h0 = now.replace(hour=12 if now.hour >= 12 else 0, minute=0, second=0, microsecond=0)
    return [h0 - timedelta(hours=12 * k) for k in range(n)]


def manifest_for(dt: datetime) -> list:
    q = urllib.parse.quote(f"{dt:%Y-%m-%d %H:%M:%S}")
    try:
        d = json.loads(get(f"{UW}/sounding_json?datetime={q}"))
        out = []
        for s in d.get("stations", []):
            name = str(s.get("name", "")).split("\n")[0].strip()
            if "dtype" in name.lower() or "name:" in name.lower():
                name = ""                       # UW sometimes leaks a pandas repr here
            out.append({"id": str(s["stationid"]), "n": name,
                        "la": round(float(s["lat"]), 3), "lo": round(float(s["lon"]), 3),
                        "src": s.get("src", "BUFR"), "dt": f"{dt:%Y-%m-%d %H:00}"})
        return out
    except Exception as e:                                   # noqa: BLE001
        print(f"  manifest {dt:%Y-%m-%d %HZ} failed: {repr(e)[:70]}", flush=True)
        return []


def thin_csv(text: str) -> str | None:
    lines = text.strip().split("\n")
    if len(lines) < 12 or not lines[0].startswith("time"):
        return None
    body = lines[1:]
    if len(body) > MAX_LEVELS:
        k = -(-len(body) // MAX_LEVELS)                      # ceil
        body = [b for i, b in enumerate(body) if i % k == 0 or i == len(body) - 1]
    return "\n".join([lines[0]] + body) + "\n"


def fetch_station(entry: dict, outdir: Path) -> bool:
    q = urllib.parse.quote(f"{entry['dt']}:00")
    url = (f"{UW}/sounding?datetime={q}&id={entry['id']}"
           f"&type=TEXT:CSV&src={entry['src']}")
    try:
        text = get(url).decode("utf-8", errors="ignore")
        thinned = thin_csv(text)
        if not thinned:
            return False
        (outdir / "soundings" / f"{entry['id']}.csv").write_text(thinned)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def main() -> int:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "skewt-data-out")
    (outdir / "soundings").mkdir(parents=True, exist_ok=True)

    # newest-first union of the last two synoptic manifests
    entries: dict = {}
    for dt in synoptic_hours(2):
        for e in manifest_for(dt):
            entries.setdefault(e["id"], e)
    print(f"  UW manifest union: {len(entries)} stations", flush=True)

    # retention: carry forward recent previous-cycle stations we still lack
    try:
        prev = json.loads(get(RAW_PREV).decode())
        cutoff = datetime.now(timezone.utc) - timedelta(hours=RETAIN_H)
        kept = 0
        for sid, e in prev.get("entries", {}).items():
            if sid in entries:
                continue
            if datetime.strptime(e["dt"], "%Y-%m-%d %H:%M").replace(
                    tzinfo=timezone.utc) >= cutoff:
                e["carry"] = True
                entries[sid] = e
                kept += 1
        print(f"  carried forward {kept} station(s) from previous cycle", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"  no previous manifest ({repr(e)[:50]})", flush=True)

    ok, carried = {}, 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for sid, e in entries.items():
            if e.get("carry"):
                # re-download carried stations too (cheap; keeps files present)
                futs[ex.submit(fetch_station, e, outdir)] = sid
            else:
                futs[ex.submit(fetch_station, e, outdir)] = sid
        for f in cf.as_completed(futs):
            sid = futs[f]
            if f.result():
                ok[sid] = {k: v for k, v in entries[sid].items() if k != "carry"}
            time.sleep(0.02)

    (outdir / "manifest.json").write_text(json.dumps(
        {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
         "entries": ok}))
    print(f"  mirrored {len(ok)}/{len(entries)} stations → {outdir}", flush=True)
    return 0 if len(ok) > 100 else 1


if __name__ == "__main__":
    sys.exit(main())
