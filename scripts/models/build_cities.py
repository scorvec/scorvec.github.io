#!/usr/bin/env python3
"""The degree-day tracker's temperature cities as meteogram points for models.html (2026-10-01, user: "also at a
city list: the main temperature cities used by the degree-day tracker").

Reads the PRIVATE ~/ddtracker (read only, laptop): the morning brief's 31 major cities (`src_morning_brief.CITIES`) and
every station in the weighted-region station sets (`ddt/stations.SV`, the vendor's 27 regions) - 85 stations. All but
four are already airports in data/models_airports.json; the four that are not (Central Park, downtown Los Angeles/USC,
March ARB, Sacramento Executive) are written here with the tracker's own coordinates. `tracker` lists all 85 so the
page can mark them. Nothing else from the tracker (weights, forecasts) goes into this public file.

    python scripts/models/build_cities.py --ddtracker ~/ddtracker --out data/models_cities.json
"""
import argparse
import json
import sys
from pathlib import Path

NAMES = {"KNYC": "New York Central Park", "KFHM": "Los Angeles Downtown (USC)", "KRIV": "March ARB (Riverside)",
         "KSAC": "Sacramento Executive"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddtracker", default=str(Path.home() / "ddtracker"))
    ap.add_argument("--airports", default="data/models_airports.json")
    ap.add_argument("--out", default="data/models_cities.json")
    a = ap.parse_args()
    sys.path.insert(0, a.ddtracker)
    from ddt import stations as SV                              # noqa: E402
    import src_morning_brief as B                               # noqa: E402
    brief = list(B.CITIES)
    ids = brief + sorted({s for v in SV.SV.values() for s, _, _ in v} - set(brief))
    have = {r[2] for r in json.loads(Path(a.airports).read_text())["airports"]}
    t = SV.unique_stations().set_index("icao")
    rows = []
    for s in ids:
        if s in have:
            continue
        r = t.loc[s]
        rows.append([round(float(r.lon), 4), round(float(r.lat), 4), s, NAMES.get(s, s), 1])
    doc = dict(source="degree-day tracker city set (morning-brief cities + weighted-region stations)",
               fields=["lon", "lat", "icao", "name", "tier"], cities=rows, tracker=ids, brief=brief)
    Path(a.out).write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    print(f"{len(ids)} tracker cities, {len(rows)} not airports -> {a.out}")


if __name__ == "__main__":
    main()
