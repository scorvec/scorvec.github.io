#!/usr/bin/env python3
"""Airport markers for models.html (US and Canada) and samerica-models.html (South America).

OurAirports (public domain, davidmegginson/ourairports-data): large and medium airports of the region with scheduled
service and a 4-letter ICAO identifier. Tier 1 = large_airport (drawn at national scale), 2 = medium.
The North American list is refreshed by hand (no workflow runs it); models-sa.yml builds the South American one once,
when main does not have it yet.

    python scripts/models/build_airports.py                  # -> data/models_airports.json
    python scripts/models/build_airports.py --region sa      # -> data/models_airports_sa.json
"""

import argparse
import csv
import io
import json
import re
import urllib.request

URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
# region -> (which rows, the ICAO identifiers its airports use, the default output)
REGIONS = {
    "na": (lambda r: r["iso_country"] in ("US", "CA"), r"[KCP][A-Z0-9]{3}", "data/models_airports.json"),
    "sa": (lambda r: r["continent"] == "SA", r"S[A-Z0-9]{3}", "data/models_airports_sa.json"),
}


def airports(rows, region):
    """OurAirports rows -> [[lon, lat, icao, name, tier], ...], large airports first."""
    keep, icao_re, _ = REGIONS[region]
    out = []
    for r in rows:
        if not keep(r) or r["type"] not in ("large_airport", "medium_airport"):
            continue
        if r["scheduled_service"] != "yes":
            continue
        icao = (r.get("icao_code") or r.get("gps_code") or r["ident"]).strip().upper()
        if not re.fullmatch(icao_re, icao):
            continue
        name = re.sub(r"\s+(International|Regional|Municipal)?\s*Airport$", "", r["name"]).strip()
        # Spanish and Portuguese names lead with the word instead ("Aeropuerto Internacional Jorge Chávez")
        name = re.sub(r"^Aero(puerto|porto)( Internacional| Regional)?( de| do| da| del)?\s+", "", name)
        out.append(
            [
                round(float(r["longitude_deg"]), 4),
                round(float(r["latitude_deg"]), 4),
                icao,
                name,
                1 if r["type"] == "large_airport" else 2,
            ]
        )
    out.sort(key=lambda x: (x[4], x[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=sorted(REGIONS), default="na")
    ap.add_argument("--out", default="", help="default: the region's file under data/")
    a = ap.parse_args()
    raw = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"User-Agent": "scorvec-models/1.0"}), timeout=120
    ).read()
    out = airports(csv.DictReader(io.StringIO(raw.decode("utf-8"))), a.region)
    path = a.out or REGIONS[a.region][2]
    with open(path, "w") as f:
        json.dump(
            dict(
                source="OurAirports (public domain), ourairports.com",
                fields=["lon", "lat", "icao", "name", "tier"],
                airports=out,
            ),
            f,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    print(f"{len(out)} airports ({sum(1 for x in out if x[4] == 1)} large) -> {path}")


if __name__ == "__main__":
    main()
