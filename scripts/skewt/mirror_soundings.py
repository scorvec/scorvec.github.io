#!/usr/bin/env python3
"""Mirror the latest University of Wyoming soundings for the Skew-T Explorer.

weather.uwyo.edu serves global BUFR/GTS soundings but sends no CORS headers,
so a browser can't fetch it directly. This job (GitHub Actions, ~4x/day)
pulls UW's own per-hour station manifest (/wsgi/sounding_json) plus each
station's TEXT:CSV profile, thins it, and publishes to the `skewt-data`
branch — raw.githubusercontent.com serves it with `Access-Control-Allow-
Origin: *`, which makes the viewer purely client-side.

Retention: per-launch files ({id}_{YYYYMMDDHH}.csv) are kept for RETAIN_H
hours (~4 days), carried forward from the previous branch state (PREVDIR), so
Archive mode can serve recent dates at full BUFR fidelity before falling back
to NOAA IGRA. The newest launch per station is also written as {id}.csv for
client compatibility.

    python scripts/skewt/mirror_soundings.py OUTDIR [PREVDIR]
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

UW = "https://weather.uwyo.edu/wsgi"
MAX_LEVELS = 260
RETAIN_H = 96
WORKERS = 6


def get(url: str, timeout: int = 60, tries: int = 1, backoff: int = 5) -> bytes:
    """Fetch a URL, optionally retrying transient failures.

    UW's read times are erratic: a single 60 s timeout on the station manifest
    silently dropped a whole synoptic slot (12Z vanished on 2026-07-12 and -13),
    which greyed out every "live" dot on the map. Callers that cannot afford to
    lose a slot ask for retries; per-station fetches stay single-shot so one
    slow station can't blow the job's time budget.
    """
    req = urllib.request.Request(url, headers={"User-Agent":
        "scorvec.com skew-t mirror (contact: site owner; ~4 fetches/day/station)"})
    for k in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError:      # the server answered — retrying won't help
            raise
        except Exception as e:              # noqa: BLE001 — timeout / reset / DNS
            if k == tries:
                raise
            print(f"    retry {k}/{tries - 1} after {repr(e)[:60]}", flush=True)
            time.sleep(backoff * k)
    raise RuntimeError("unreachable")


def synoptic_hours(n: int = 5):
    """Most recent synoptic slots, newest first, stepping every SIX hours.

    NO 2-HOUR LOOKBACK. It used to subtract 2 h before flooring, which does not
    merely wait for data - it skips a whole slot: a run at 19:46Z floored to 12Z
    and never asked UW about 18Z at all. That is why the 24 extra North American
    launches at 18Z on 2026-08-31 were absent from the map. An empty manifest for
    a slot whose data has not landed yet costs one cheap request; missing the slot
    costs six hours of coverage.
    """
    h0 = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    h0 = h0.replace(hour=(h0.hour // 6) * 6)
    return [h0 - timedelta(hours=6 * k) for k in range(n)]


def offhour_hours(n: int = 24):
    """Every NON-synoptic hour in the last n hours, newest first.

    True off-hour releases exist and were invisible to 6-hourly stepping: on
    2026-08-31 five stations reported at 15Z and nowhere else. They are cheap to
    look for - the manifest call is small and, crucially, a station already
    claimed by a synoptic slot is never re-fetched.

    WHY THESE ARE MERGED SECOND, NOT BY RECENCY. UW's manifest smears a synoptic
    launch across neighbouring hours: on 2026-08-31 the 12Z launches also came
    back for 10Z, 11Z, 13Z and 14Z (311, 312, 311, 275 stations). The manifest
    carries no per-station launch time, and both the saved filename and the
    profile URL use the QUERY hour - so accepting the smear would have written
    the same 12Z ascent three times as phantom 11Z/13Z launches. Taking off-hours
    only for stations no synoptic slot claimed keeps the genuine 15Z releases and
    drops the duplicates. A real off-hour launch appears at its own hour alone.
    """
    h0 = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return [h for h in (h0 - timedelta(hours=k) for k in range(n)) if h.hour % 6]


def manifest_for(dt: datetime) -> list:
    q = urllib.parse.quote(f"{dt:%Y-%m-%d %H:%M:%S}")
    try:
        d = json.loads(get(f"{UW}/sounding_json?datetime={q}",
                           timeout=75, tries=3, backoff=6))
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


def _tag(dt: str) -> str:
    return dt.replace("-", "").replace(" ", "").replace(":", "")[:10]


def fetch_station(entry: dict, outdir: Path) -> bool:
    dest = outdir / "soundings" / f"{entry['id']}_{_tag(entry['dt'])}.csv"
    if dest.exists():                                        # carried forward
        return True
    q = urllib.parse.quote(f"{entry['dt']}:00")
    url = (f"{UW}/sounding?datetime={q}&id={entry['id']}"
           f"&type=TEXT:CSV&src={entry['src']}")
    try:
        text = get(url).decode("utf-8", errors="ignore")
        thinned = thin_csv(text)
        if not thinned:
            return False
        dest.write_text(thinned)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def main() -> int:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "skewt-data-out")
    prevdir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    (outdir / "soundings").mkdir(parents=True, exist_ok=True)

    # carry forward previous per-launch files still inside the retention window
    cutoff = datetime.now(timezone.utc) - timedelta(hours=RETAIN_H)
    prev_manifest = {}
    if prevdir and (prevdir / "manifest.json").exists():
        prev_manifest = json.loads((prevdir / "manifest.json").read_text())
        carried = 0
        for f in (prevdir / "soundings").glob("*_*.csv"):
            try:
                ts = datetime.strptime(f.stem.split("_")[1], "%Y%m%d%H").replace(
                    tzinfo=timezone.utc)
            except (IndexError, ValueError):
                continue
            if ts >= cutoff:
                (outdir / "soundings" / f.name).write_text(f.read_text())
                carried += 1
        print(f"  carried forward {carried} launch file(s)", flush=True)

    # newest-first union of the last few synoptic manifests (6-hourly)
    entries: dict = {}
    per_slot: dict = {}
    for dt in synoptic_hours(5):
        got = manifest_for(dt)
        per_slot[dt] = len(got)
        for e in got:
            entries.setdefault(e["id"], e)
    n_syn = len(entries)
    # Second pass ONLY - see offhour_hours() for why recency must not reorder it.
    off = 0
    for dt in offhour_hours(24):
        for e in manifest_for(dt):
            if e["id"] not in entries:
                entries[e["id"]] = e
                off += 1
    print(f"  UW manifest union: {len(entries)} stations "
          f"({', '.join(f'{d:%H}Z:{n}' for d, n in per_slot.items())})"
          f" + {off} off-hour", flush=True)

    # The map's live (blue) dots are exactly the stations that reported at the
    # newest regular slot. Lose that slot and every dot greys out even though the
    # balloons flew — so fail loudly instead of publishing a map that looks broken.
    regular = [d for d in per_slot if d.hour in (0, 12)]
    if regular and not per_slot[max(regular)]:
        print(f"::warning::UW manifest for {max(regular):%Y-%m-%d %H}Z came back "
              "empty — live dots will be missing from the map this cycle", flush=True)

    # stations from the previous manifest whose launches are still in window
    for sid, e in (prev_manifest.get("entries") or {}).items():
        if sid in entries:
            continue
        try:
            if datetime.strptime(e["dt"], "%Y-%m-%d %H:%M").replace(
                    tzinfo=timezone.utc) >= cutoff:
                entries[sid] = e
        except (KeyError, ValueError):
            continue

    ok = {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_station, e, outdir): sid
                for sid, e in entries.items()}
        for f in cf.as_completed(futs):
            sid = futs[f]
            if f.result():
                ok[sid] = entries[sid]
            time.sleep(0.02)

    # per-station available launch hours (within retention) + legacy latest copy
    hours: dict = {}
    for f in (outdir / "soundings").glob("*_*.csv"):
        sid, tag = f.stem.split("_")
        dtstr = f"{tag[:4]}-{tag[4:6]}-{tag[6:8]} {tag[8:10]}:00"
        hours.setdefault(sid, []).append(dtstr)
    for sid, hs in hours.items():
        hs.sort()
        if sid in ok:
            ok[sid]["hours"] = hs
            ok[sid]["dt"] = hs[-1]
            latest = outdir / "soundings" / f"{sid}_{_tag(hs[-1])}.csv"
            (outdir / "soundings" / f"{sid}.csv").write_text(latest.read_text())

    (outdir / "manifest.json").write_text(json.dumps(
        {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
         "entries": ok}))
    print(f"  mirrored {len(ok)}/{len(entries)} stations "
          f"({sum(len(h) for h in hours.values())} launch files) → {outdir}", flush=True)
    return 0 if len(ok) > 100 else 1


if __name__ == "__main__":
    sys.exit(main())
