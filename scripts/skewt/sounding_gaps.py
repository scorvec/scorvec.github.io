#!/usr/bin/env python3
"""Track US stations missing their routine 00Z/12Z soundings.

For every active US radiosonde site, read the IGRA year-to-date file
(headers only) and grade each 00Z/12Z slot since START:
  1 = reported on time (nominal 00/12Z launch)
  2 = off-hour launch instead (nearest slot had no on-time release)
  0 = missed entirely
Writes skewt/gaps.json: per-station day strings (2 chars/day) plus
rollups for the last 14/60 days.

    python scripts/skewt/sounding_gaps.py [START]   # default 2025-01-01
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
IGRA = ("https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/"
        "access/data-y2d/")
START = date(2026, 1, 1)


def station_days(gid: str, y0: int):
    """{date: {"on": set(hours 0/12), "off": [hours]}} from y2d headers."""
    raw = None
    for yy in (y0 + 1, y0, y0 - 1):        # beg-year varies with por freshness
        try:
            raw = urlopen(IGRA + f"{gid}-data-beg{yy}.txt.zip", timeout=300).read()
            break
        except Exception:
            continue
    if raw is None:
        raise RuntimeError("no y2d file")
    out: dict = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            for line in f:
                if line[:1] != b"#":
                    continue
                s = line.decode("ascii", "replace")
                try:
                    d = date(int(s[13:17]), int(s[18:20]), int(s[21:23]))
                    h = int(s[24:26])
                except ValueError:
                    continue
                slot = out.setdefault(d, {"on": set(), "off": []})
                if h in (0, 12):
                    slot["on"].add(h)
                elif 0 <= h <= 23:
                    slot["off"].append(h)
    return out


def grade(days: dict, start: date, end: date) -> str:
    """two chars per day (00Z slot, 12Z slot)."""
    chars = []
    d = start
    while d <= end:
        rec = days.get(d, {"on": set(), "off": []})
        for slot in (0, 12):
            if slot in rec["on"]:
                chars.append("1")
            else:
                # any off-hour launch nearer this slot than the other one?
                near = [h for h in rec["off"]
                        if abs(h - slot) <= 6 or abs(h - slot) >= 18]
                chars.append("2" if near else "0")
        d += timedelta(days=1)
    return "".join(chars)


def main() -> int:
    start = (date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else START)
    # y2d files lag ~2 days
    end = datetime.now(timezone.utc).date() - timedelta(days=2)
    stns = json.loads((ROOT / "skewt" / "stations.json").read_text())["stations"]
    us = [s for s in stns if s["gid"].startswith("USM") and s.get("y1", 0) >= 2025]
    print(f"{len(us)} active US stations, {start} .. {end}", flush=True)
    out = {"start": str(start), "end": str(end),
           "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "stations": {}}
    for s in us:
        try:
            days = station_days(s["gid"], start.year)
        except Exception as e:
            print(f"  {s['id']} {s['n'][:24]}: FAILED {e}", flush=True)
            continue
        g = grade(days, start, end)
        n = len(g)
        miss60 = g[-120:].count("0")
        off60 = g[-120:].count("2")
        miss14 = g[-28:].count("0")
        out["stations"][s["id"]] = {
            "n": s["n"].split(";")[0].split("/")[0].strip(),
            "la": s["la"], "lo": s["lo"], "g": g,
            "miss60": miss60, "off60": off60, "miss14": miss14,
            # research/range sites (ARM Lamont, proving grounds) never ran the
            # routine schedule — flag so the page separates them
            "routine": 1 if g.count("1") >= 0.4 * len(g) else 0,
        }
        if miss60 or off60:
            print(f"  {s['id']} {s['n'][:28]:30s} last60d: {miss60:3d} missed, "
                  f"{off60:2d} off-hour", flush=True)
    (ROOT / "skewt" / "gaps.json").write_text(json.dumps(out, separators=(",", ":")))
    total = sum(v["miss60"] for v in out["stations"].values())
    print(f"wrote gaps.json · {len(out['stations'])} stations · "
          f"{total} slots missed in last 60 days", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
