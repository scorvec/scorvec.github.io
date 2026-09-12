#!/usr/bin/env python3
"""CPC's OFFICIAL ONI and RONI, straight from CPC — the reference the site's
OISST estimates are estimates OF.

Everything else on the El Nino monitor is computed here from NOAA OISST v2.1,
which is the point of the page: a daily estimate available now rather than
after the month closes. But the card that sets ONI beside RONI had been drawing
both OISST series with the gold one labelled "ONI (official)", which it is not.
On JJA 2026 that read +2.10 where CPC had published +1.80.

Two sources, because CPC publishes them differently:

  ONI   data/indices/oni.ascii.txt  — SEAS YR TOTAL ANOM, one row per centred
        season. This is the OFFICIAL ONI, on the 30-year base periods that
        shift every five years — NOT the fixed 1991-2020 base that
        build_nino_history.py reads from ersst5.nino.mth.91-20.ascii. The two
        differ: MJJ 2026 is +1.39 official against +1.17 on the fixed base.

  RONI  the HTML table on the RONI product page. There is no ascii endpoint.
        Rows are year x season and the CURRENT year is SHORT — parsing that
        assumed 12 columns and silently dropped 2026 entirely.

CPC adopted RONI as its official ENSO index in February 2026, so the RONI row
is the one that now carries the official designation; ONI is kept for the
comparison the card exists to make. RONI is computed on ERSSTv6.

Updated by the 5th of each month, so the newest season here lags the OISST
estimate by up to ~6 weeks. That gap is exactly what the estimate is for.

    python cpc_official_enso.py
-> assets/sst/data/cpc_official.json
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "scripts") / "lib"))
from webget import get_text  # noqa: E402

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
OUT = SITE_ROOT / "assets" / "sst" / "data" / "cpc_official.json"

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
RONI_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/"
SEAS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
# the month a centred season is centred on (DJF -> Jan, JJA -> Jul, NDJ -> Dec)
CENTRE = {s: i + 1 for i, s in enumerate(SEAS)}


def fetch_oni() -> dict:
    """{'YYYY-MM': anom} keyed by the season's CENTRE month."""
    out = {}
    for ln in get_text(ONI_URL).splitlines()[1:]:
        f = ln.split()
        if len(f) != 4 or f[0] not in CENTRE:
            continue
        yr, seas = int(f[1]), f[0]
        # CPC labels a season by the year of its LATER months: "DJF 1950" is
        # Dec 1949 + Jan/Feb 1950, centred January 1950. So the centre month's
        # year is the row's year for every season, DJF included.
        try:
            out[f"{yr}-{CENTRE[seas]:02d}"] = float(f[3])
        except ValueError:
            pass
    return out


def fetch_roni() -> dict:
    """Same keying, parsed from the product page's year x season table.

    The current year's row is SHORT (only the seasons published so far), so a
    fixed column count must not be required — that is what dropped 2026.
    """
    h = get_text(RONI_URL)
    out = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)]
        if not cells or not re.fullmatch(r"(19|20)\d\d", cells[0] or ""):
            continue
        yr = int(cells[0])
        for seas, v in zip(SEAS, cells[1:]):
            if v in ("", "-", "\xa0"):
                continue
            try:
                val = float(v)
            except ValueError:
                continue
            out[f"{yr}-{CENTRE[seas]:02d}"] = val
    return out


def main():
    oni, roni = fetch_oni(), fetch_roni()
    if not oni or not roni:
        raise SystemExit(f"empty parse: oni={len(oni)} roni={len(roni)}")
    doc = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "note": ("CPC's OFFICIAL published values. Keys are the centred month of "
                    "each 3-month season (JJA -> 07). ONI is on CPC's shifting 30-year "
                    "base periods; RONI is ERSSTv6 on 1991-2020 and has been CPC's "
                    "official ENSO index since February 2026."),
           "sources": {"oni": ONI_URL, "roni": RONI_URL},
           "oni": oni, "roni": roni,
           "latest": {"oni": max(oni), "roni": max(roni)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"  ONI  {len(oni)} seasons, latest {max(oni)} = {oni[max(oni)]:+.2f}")
    print(f"  RONI {len(roni)} seasons, latest {max(roni)} = {roni[max(roni)]:+.2f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
