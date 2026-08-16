#!/usr/bin/env python3
"""Historical IDEAM gauge pull for the multi-year IMERG backtest.

Per-station DAILY totals from the Socrata feed, one half-month per
request (stations x days stays under the row cap), cached monthly:
raw/gauges_hist/YYYYMM.json = {station: {la, lo, days: {D: mm}}}.
The feed reaches back to 2003-01-20; station density grows over time
and there are multi-week ingest holes — the backtest analysis handles
both by pairing on available days.

    python scripts/sst/ideam_gauge_history.py --start 2014-01 --end 2026-08
"""
from __future__ import annotations

import argparse
import calendar
import json
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.datos.gov.co/resource/s54a-sgyg.json"
CACHE = Path.home() / "colombia_hydro" / "raw" / "gauges_hist"
MAX_MM_DAY = 450.0


def _q(where: str) -> list:
    q = {"$select": ("codigoestacion,latitud,longitud,"
                     "date_trunc_ymd(fechaobservacion) AS d,"
                     "sum(valorobservado) AS mm"),
         "$where": where,
         "$group": "codigoestacion,latitud,longitud,d",
         "$limit": "50000"}
    url = BASE + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "scorvec-hydro/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def fetch_month(ym: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{ym}.json"
    if f.exists():
        return json.loads(f.read_text())
    y, m = int(ym[:4]), int(ym[4:6])
    last = calendar.monthrange(y, m)[1]
    halves = [(f"{y}-{m:02d}-01", f"{y}-{m:02d}-16"),
              (f"{y}-{m:02d}-16", f"{y}-{m:02d}-{last:02d}T23:59:59")]
    out: dict = {}
    for a, b in halves:
        rows = _q(f"fechaobservacion >= '{a}T00:00:00' AND fechaobservacion < '{b}'"
                  if "T" not in b else
                  f"fechaobservacion >= '{a}T00:00:00' AND fechaobservacion <= '{b}'")
        for row in rows:
            try:
                la, lo = float(row["latitud"]), float(row["longitud"])
                mm = float(row["mm"])
                day = row["d"][:10]
            except (KeyError, ValueError):
                continue
            if not (0 <= mm <= MAX_MM_DAY):
                continue
            st = out.setdefault(row["codigoestacion"],
                                {"la": round(la, 5), "lo": round(lo, 5), "days": {}})
            st["days"][day] = round(st["days"].get(day, 0.0) + mm, 2)
    f.write_text(json.dumps(out, separators=(",", ":")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01")
    ap.add_argument("--end", default="2026-08")
    a = ap.parse_args()
    y0, m0 = int(a.start[:4]), int(a.start[5:7])
    y1, m1 = int(a.end[:4]), int(a.end[5:7])
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        ym = f"{y}{m:02d}"
        try:
            d = fetch_month(ym)
            print(f"  {ym}: {len(d)} stations", flush=True)
        except Exception as e:                        # noqa: BLE001
            print(f"  {ym} FAILED ({repr(e)[:60]})", flush=True)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
