"""Coalesced byte-range fetcher for ECMWF open data (fetch v2, task #22).

Instead of per-message retrieves through ecmwf-opendata (thousands of tiny
requests under a 500-connection portal cap), this module:

  1. downloads the small `.index` (JSON-lines: one entry per GRIB message
     with `_offset`/`_length` + metadata),
  2. selects the wanted messages (param / levelist / member),
  3. sorts by offset and COALESCES entries whose gaps are < `max_gap`
     into a handful of large ranges,
  4. fetches those ranges in parallel with hard timeouts, preferring the
     S3 mirror (no connection-count regime), rotating mirrors per-range
     on failure,
  5. reassembles the exact message bytes into one GRIB2 payload.

A 700-message AAM slice becomes ~a dozen fat range-GETs instead of 700
polite ones.

    python rangefetch.py --bench          # live benchmark, latest cycle
"""
from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

try:
    import budget
except ImportError:                                        # ad-hoc callers off-path
    from pathlib import Path as _P
    import sys as _sys; _sys.path.insert(0, str(_P(__file__).resolve().parent))
    import budget

MIRRORS = {
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
    "azure": "https://ai4edataeuwest.blob.core.windows.net/ecmwf",
    "ecmwf": "https://data.ecmwf.int/forecasts",
    "google": "https://storage.googleapis.com/ecmwf-open-data",
}
TIMEOUT = (10, 90)          # (connect, read) — nothing hangs for hours, ever
MAX_GAP = 3 * 1024 * 1024   # merge ranges separated by < 3 MB
WORKERS = 16                # empirical single-IP ceiling ~32; leave headroom
                            # for OTHER processes sharing this IP (pipeline,
                            # benchmarks). Adaptive: throttle responses shrink
                            # effective concurrency for the rest of the batch.
_throttle_events = 0


def path_for(date: str, hh: str, model: str, step: int, kind: str = "pf",
             stream: str = "enfo", resol: str = "0p25") -> str:
    return (f"{date}/{hh}z/{model}/{resol}/{stream}/"
            f"{date}{hh}0000-{step}h-{stream}-{kind}")


def fetch_index(date: str, hh: str, model: str, step: int, kind: str = "pf",
                stream: str = "enfo",
                sources: list[str] | None = None) -> list[dict]:
    """Parse the .index file: list of message dicts with _offset/_length."""
    rel = path_for(date, hh, model, step, kind, stream=stream) + ".index"
    last_err = None
    for attempt in range(3):
        for src in sources or ["google", "aws", "azure", "ecmwf"]:
            try:
                budget.acquire(src)
                r = requests.get(f"{MIRRORS[src]}/{rel}", timeout=TIMEOUT)
                if r.status_code == 200:
                    return [json.loads(line) for line in r.text.splitlines() if line.strip()]
                last_err = RuntimeError(f"{src}: HTTP {r.status_code}")
                if r.status_code in (429, 503):
                    budget.penalize(src)
                    time.sleep(1.5 + 3.0 * attempt)
            except Exception as e:                             # noqa: BLE001
                last_err = e
    raise RuntimeError(f"index unavailable for {rel}: {last_err}")


def select(entries: list[dict], param: str | list | None = None,
           levelist: list | None = None, numbers: list | None = None) -> list[dict]:
    params = {param} if isinstance(param, str) else (set(param) if param else None)
    levs = {str(l) for l in levelist} if levelist else None
    nums = {str(n) for n in numbers} if numbers else None
    out = []
    for e in entries:
        if params and e.get("param") not in params:
            continue
        if levs and str(e.get("levelist")) not in levs:
            continue
        if nums and str(e.get("number")) not in nums:
            continue
        out.append(e)
    return out


def coalesce(entries: list[dict], max_gap: int = MAX_GAP) -> list[tuple[int, int, list[dict]]]:
    """[(range_start, range_end_exclusive, member_entries), ...] sorted."""
    es = sorted(entries, key=lambda e: e["_offset"])
    ranges = []
    for e in es:
        s, l = e["_offset"], e["_length"]
        if ranges and s - ranges[-1][1] <= max_gap:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], s + l), ranges[-1][2] + [e])
        else:
            ranges.append((s, s + l, [e]))
    return ranges


def fetch_ranges(rel_grib: str, ranges, sources: list[str] | None = None,
                 workers: int = WORKERS, stats: dict | None = None) -> bytes:
    """Parallel ranged GETs; returns wanted message bytes concatenated in
    offset order. Rotates mirrors per-range on failure. If `stats` is given,
    bytes served are accumulated per mirror (for speed attribution)."""
    sources = sources or ["google", "aws", "azure", "ecmwf"]
    import threading
    gate = threading.Semaphore(workers)          # shrinks on throttle signals
    shrink_lock = threading.Lock()
    state = {"permits": workers}
    stats_lock = threading.Lock()

    def _shrink():
        with shrink_lock:
            if state["permits"] > 4:             # never below 4
                gate.acquire(blocking=False)     # permanently retire a permit
                state["permits"] -= 1

    def one(rng):
        start, end, members = rng
        last = None
        for attempt in range(3):
            for src in sources:
                try:
                    budget.acquire(src)
                    r = requests.get(f"{MIRRORS[src]}/{rel_grib}",
                                     headers={"Range": f"bytes={start}-{end - 1}"},
                                     timeout=TIMEOUT)
                    if r.status_code in (200, 206):
                        blob = r.content
                        if stats is not None:
                            with stats_lock:
                                stats[src] = stats.get(src, 0) + len(blob)
                        return b"".join(
                            blob[e["_offset"] - start: e["_offset"] - start + e["_length"]]
                            for e in members)
                    last = RuntimeError(f"{src}: HTTP {r.status_code}")
                    if r.status_code in (429, 503):            # throttled: back off AND
                        budget.penalize(src)
                        _shrink()                              # shed concurrency
                        time.sleep(1.0 + 2.0 * attempt)
                except Exception as e:                         # noqa: BLE001
                    last = e
        raise RuntimeError(f"range {start}-{end} failed on all mirrors: {last}")

    def gated(rng):
        with gate:
            return one(rng)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(gated, ranges))
    return b"".join(parts)


def fetch(date: str, hh: str, model: str, step: int, *, param=None,
          levelist=None, numbers=None, kind: str = "pf",
          sources: list[str] | None = None, out: Path | None = None) -> bytes:
    """Top-level: index → select → coalesce → parallel ranges → GRIB bytes."""
    idx = fetch_index(date, hh, model, step, kind, sources)
    want = select(idx, param=param, levelist=levelist, numbers=numbers)
    if not want:
        raise RuntimeError(f"no messages matched (param={param} lev={levelist})")
    ranges = coalesce(want)
    blob = fetch_ranges(path_for(date, hh, model, step, kind) + ".grib2",
                        ranges, sources)
    if out:
        Path(out).write_bytes(blob)
    return blob


def cycle_complete(date: str, hh: str, model: str = "aifs-ens",
                   last_step: int = 360, kind: str = "pf") -> bool:
    """Publication sentinel: the LAST step's .index exists on some mirror →
    the cycle is fully disseminated; safe to start bulk fetching."""
    rel = path_for(date, hh, model, last_step, kind) + ".index"
    for src in ("aws", "azure", "ecmwf"):
        try:
            budget.acquire(src)
            r = requests.head(f"{MIRRORS[src]}/{rel}", timeout=(5, 15))
            if r.status_code == 200:
                return True
        except Exception:                                  # noqa: BLE001
            continue
    return False


def _bench():
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    cand = now - timedelta(hours=9)
    date, hh = cand.strftime("%Y%m%d"), "12" if cand.hour >= 12 else "00"
    print(f"benchmark cycle: {date} {hh}z (aifs-ens)")
    for label, kw in (
        ("RMM slice  (u @ 200+850, all members)", dict(param="u", levelist=[200, 850])),
        ("AAM slice  (u @ 13 levels, all members)",
         dict(param="u", levelist=[50, 100, 150, 200, 250, 300, 400, 500,
                                   600, 700, 850, 925, 1000])),
    ):
        t0 = time.time()
        idx = fetch_index(date, hh, "aifs-ens", 0)
        want = select(idx, **kw)
        ranges = coalesce(want)
        t1 = time.time()
        blob = fetch_ranges(path_for(date, hh, "aifs-ens", 0) + ".grib2", ranges)
        dt = time.time() - t1
        mb = len(blob) / 1e6
        print(f"  {label}: {len(want)} msgs -> {len(ranges)} ranges | "
              f"index+plan {t1-t0:.1f}s | fetch {mb:.0f} MB in {dt:.1f}s "
              f"({mb/max(dt,0.01):.0f} MB/s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    if a.bench:
        _bench()
