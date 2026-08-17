#!/usr/bin/env python3
"""INMET automatic-station daily rainfall -> per-day gauge cache.

Parses the INMET bulk yearly ZIPs (portal.inmet.gov.br historical
uploads, ~600 automatic stations, hourly, latin-1, comma decimals,
UTC) into the same day-file format as the Colombia IDEAM cache:

  ~/brazil_hydro/raw/gauges/YYYYMMDD.json = {code: {la, lo, mm}}

Daily total = sum of hourly precip over the UTC day; days with <20
valid hours are dropped for that station. Each ZIP is parsed once
(marker file keyed to zip size); re-download the current-year ZIP to
extend (INMET refreshes it monthly).

    python scripts/sst/brazil_gauges.py
"""
from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

RAW = Path.home() / "brazil_hydro" / "raw"
ZIPS = RAW / "inmet"
GCACHE = RAW / "gauges"
MAX_MM_DAY = 500.0


def parse_zip(zp: Path) -> dict:
    """{YYYYMMDD: {code: {la, lo, mm}}} from one yearly ZIP."""
    out: dict[str, dict] = defaultdict(dict)
    zf = zipfile.ZipFile(zp)
    for name in zf.namelist():
        if not name.upper().endswith(".CSV"):
            continue
        try:
            txt = zf.read(name).decode("latin-1")
        except Exception:                       # noqa: BLE001
            continue
        lines = txt.splitlines()
        meta = {}
        for ln in lines[:8]:
            if ":;" in ln:
                k, v = ln.split(":;", 1)
                meta[k.strip().upper()] = v.strip()
        try:
            code = meta.get("CODIGO (WMO)", "").strip()
            la = float(meta["LATITUDE"].replace(",", "."))
            lo = float(meta["LONGITUDE"].replace(",", "."))
        except (KeyError, ValueError):
            continue
        if not code:
            continue
        # hourly rows: Data;Hora UTC;precip;...
        daysum: dict[str, list] = defaultdict(lambda: [0.0, 0])
        for ln in lines[9:]:
            parts = ln.split(";")
            if len(parts) < 3 or len(parts[0]) < 8:
                continue
            d = parts[0].replace("/", "").replace("-", "")[:8]
            v = parts[2].strip().replace(",", ".")
            if v in ("", "-9999"):
                continue
            try:
                mm = float(v)
            except ValueError:
                continue
            if mm < 0 or mm > 250:              # hourly sanity
                continue
            acc = daysum[d]
            acc[0] += mm
            acc[1] += 1
        for d, (mm, nh) in daysum.items():
            if nh >= 20 and mm <= MAX_MM_DAY:
                out[d][code] = {"la": round(la, 4), "lo": round(lo, 4),
                                "mm": round(mm, 1)}
    return out


def refresh_current_year() -> None:
    """INMET refreshes the current-year ZIP monthly — re-download when ours
    is >30 days old."""
    import time
    import urllib.request
    from datetime import datetime
    y = datetime.now().year
    zp = ZIPS / f"{y}.zip"
    if zp.exists() and time.time() - zp.stat().st_mtime < 30 * 86400:
        return
    url = f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{y}.zip"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        if len(data) > 1e6:
            zp.write_bytes(data)
            (ZIPS / f".{y}.parsed").unlink(missing_ok=True)
            print(f"refreshed {y}.zip ({len(data)/1e6:.0f} MB)", flush=True)
    except Exception as e:                      # noqa: BLE001
        print(f"zip refresh failed: {repr(e)[:80]}", flush=True)


def main() -> int:
    GCACHE.mkdir(parents=True, exist_ok=True)
    refresh_current_year()
    for zp in sorted(ZIPS.glob("*.zip")):
        marker = ZIPS / f".{zp.stem}.parsed"
        sig = str(zp.stat().st_size)
        if marker.exists() and marker.read_text() == sig:
            continue
        print(f"parsing {zp.name} …", flush=True)
        data = parse_zip(zp)
        n = 0
        for d, stations in data.items():
            f = GCACHE / f"{d}.json"
            if f.exists():                       # merge across zips
                cur = json.loads(f.read_text())
                cur.update(stations)
                stations = cur
            f.write_text(json.dumps(stations, separators=(",", ":")))
            n += 1
        marker.write_text(sig)
        print(f"  {zp.stem}: {n} day files "
              f"(~{np_len(data)} station-days)", flush=True)
    days = sorted(GCACHE.glob("*.json"))
    print(f"gauge cache: {len(days)} days "
          f"({days[0].stem}..{days[-1].stem})" if days else "empty")
    return 0


def np_len(data: dict) -> int:
    return sum(len(v) for v in data.values())


if __name__ == "__main__":
    raise SystemExit(main())
