#!/usr/bin/env python3
"""Cheap gate for tcshear.yml (stdlib only, runs before any install): is any tropical cyclone active?

NHC's CurrentStorms.json (Atlantic, eastern and central Pacific) and JTWC's public RSS feed (warnings for the other
basins). Fails OPEN: if either source cannot be read, the full job runs (and finds the storms itself from ECMWF's
tracker). With no storm anywhere it writes the empty manifest (once) so the page says so, and prints any=false.
"""

import datetime as dt
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "assets" / "sst" / "anim" / "tcshear_manifest.json"
UA = {"User-Agent": "scorvec.com TC shear gate (+https://scorvec.com)"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
        return r.read().decode("utf-8", "replace")


def main():
    n_nhc = n_jtwc = None
    try:
        n_nhc = len(json.loads(get("https://www.nhc.noaa.gov/CurrentStorms.json")).get("activeStorms", []))
    except Exception as e:  # noqa: BLE001
        print(f"NHC unreadable: {e}")
    try:
        rss = get("https://www.metoc.navy.mil/jtwc/rss/jtwc.rss")
        n_jtwc = len(
            re.findall(
                r"(?:super typhoon|typhoon|tropical storm|tropical depression|tropical cyclone)\s+\d{2}[A-Z]"
                r"[^<]{0,60}warning",
                rss,
                re.I,
            )
        )
    except Exception as e:  # noqa: BLE001
        print(f"JTWC unreadable: {e}")
    any_ = n_nhc is None or n_jtwc is None or n_nhc > 0 or n_jtwc > 0
    print(f"NHC active: {n_nhc}, JTWC warnings: {n_jtwc} -> any={'true' if any_ else 'false'}")
    if not any_:
        old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        if old.get("storms") != [] or "built" not in old:
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(
                json.dumps(
                    dict(
                        v=1,
                        dir="tcshear",
                        init=None,
                        storms=[],
                        complete=True,
                        built=f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%MZ}",
                    ),
                    separators=(",", ":"),
                )
            )
            print("wrote the empty manifest")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"any={'true' if any_ else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
