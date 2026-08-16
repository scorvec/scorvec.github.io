#!/usr/bin/env python3
"""Recharge-oscillator orbit feed: PMEL warm water volume vs Niño-3.4.

Fetches the official WWV index (volume of water warmer than 20 °C,
5°S–5°N / 120°E–80°W, monthly since 1980, McPhaden/PMEL) and joins it
with the CPC ERSSTv5 Niño-3.4 anomaly already cached in
nino_history.json. The pair traces the classic recharge–discharge orbit
(Jin 1997): WWV leads Niño-3.4 by ~2–3 seasons, so the vertical position
now hints at where SST is headed.

Output: assets/sst/data/wwv_orbit.json
  {months, nino34, wwv, events, latest_wwv_month}
WWV anomalies in 1e14 m^3. Non-fatal: any fetch/parse failure keeps the
previous JSON (monthly-updating data; staleness is benign).

    python scripts/sst/wwv_orbit.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(__import__("os").environ.get("SST_SITE_ROOT", HERE.parent.parent))
OUT = SITE_ROOT / "assets" / "sst" / "data" / "wwv_orbit.json"
NINO = SITE_ROOT / "assets" / "sst" / "data" / "nino_history.json"
WWV_URL = "https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat"


def fetch_wwv() -> dict[str, float]:
    req = urllib.request.Request(WWV_URL, headers={
        "User-Agent": "Mozilla/5.0 (research; SST/RONI El Nino monitor)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode()
    out = {}
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 6:
            ym = f"{parts[0][:4]}-{parts[0][4:6]}"
            out[ym] = float(parts[2]) / 1e14          # anomaly, 1e14 m^3
    if len(out) < 400:
        raise ValueError(f"suspiciously short WWV record ({len(out)} rows)")
    return out


def main() -> int:
    try:
        wwv = fetch_wwv()
    except Exception as e:                            # noqa: BLE001
        print(f"WWV fetch failed ({e}); keeping previous wwv_orbit.json")
        return 0
    nh = json.loads(NINO.read_text())
    n34 = dict(zip(nh["months"], nh["series"]["nino34"]["anom"]))
    months = sorted(set(wwv) & {m for m, v in n34.items() if v is not None})
    feed = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "PMEL WWV (20C-isotherm volume, 5S-5N/120E-80W) x CPC ERSSTv5 Nino-3.4",
        "latest_wwv_month": months[-1],
        "months": months,
        "nino34": [round(n34[m], 2) for m in months],
        "wwv": [round(wwv[m], 2) for m in months],
        "events": nh.get("events", []),
    }
    OUT.write_text(json.dumps(feed, separators=(",", ":")))
    print(f"wrote {OUT.name}: {months[0]}..{months[-1]} "
          f"(latest WWV {feed['wwv'][-1]:+.2f}e14 m3, Nino-3.4 {feed['nino34'][-1]:+.2f} C)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
