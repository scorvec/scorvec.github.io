#!/usr/bin/env python3
"""Airport markers for models.html (user 2026-09-30: "add some airport stations with markers on the map (US and Canada)").

OurAirports (public domain, davidmegginson/ourairports-data): large and medium airports in the US and Canada with
scheduled service and a 4-letter ICAO identifier. Tier 1 = large_airport (drawn at national scale), 2 = medium.

    python scripts/models/build_airports.py --out data/models_airports.json
"""
import argparse
import csv
import io
import json
import re
import urllib.request

URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/models_airports.json")
    a = ap.parse_args()
    raw = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "scorvec-models/1.0"}), timeout=120).read()
    out = []
    for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        if r["iso_country"] not in ("US", "CA") or r["type"] not in ("large_airport", "medium_airport"):
            continue
        if r["scheduled_service"] != "yes":
            continue
        icao = (r.get("icao_code") or r.get("gps_code") or r["ident"]).strip().upper()
        if not re.fullmatch(r"[KCP][A-Z0-9]{3}", icao):
            continue
        name = re.sub(r"\s+(International|Regional|Municipal)?\s*Airport$", "", r["name"]).strip()
        out.append([round(float(r["longitude_deg"]), 4), round(float(r["latitude_deg"]), 4), icao, name,
                    1 if r["type"] == "large_airport" else 2])
    out.sort(key=lambda x: (x[4], x[2]))
    json.dump(dict(source="OurAirports (public domain), ourairports.com", fields=["lon", "lat", "icao", "name", "tier"],
                   airports=out), open(a.out, "w"), separators=(",", ":"), ensure_ascii=False)
    print(f"{len(out)} airports ({sum(1 for x in out if x[4] == 1)} large) -> {a.out}")


if __name__ == "__main__":
    main()
