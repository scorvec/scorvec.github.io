#!/usr/bin/env python3
"""Product freshness for the catalogue: when each product last published, next to how often it is meant to update.

Runs in pages.yml on every deploy and writes status.json into the site artifact (never committed). Each product is
dated by the stamp it carries itself, read from the deploy's checkout:
  stamp   a timestamp field inside a product file on main: `generated` (build time), a manifest's `ver` (build time
          in epoch seconds), or the forecast `init` where a product records nothing else. `stamps` lists several
          producers of one page, which is dated by the newest of them.
  branch  the head commit of a data branch that is force-pushed as a single commit per update (skewt-data)
  json    a timestamp field inside a file on the frames branch (the climate trends metadata)
Commit times on main are not used while a stamp can be read: the repository history is squashed now and then, which
makes every file look as new as the squash. A product whose stamp cannot be read falls back to the last commit
touching that file (GitHub API, the workflow's own token).

`every_h` is the product's normal interval and `late_h` the age past which the catalogue calls it late:
2 x every_h + 6 h up to daily products (Actions schedules slip by hours, so anything tighter would cry wolf),
one day past the interval for the twice-weekly ones and a week past it for the monthly ones, whose issue dates
wander by a few days.

    python scripts/site/status.py --out status.json          # GITHUB_TOKEN in the environment, else anonymous (60/h)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "scorvec/scorvec.github.io"
API = "https://api.github.com/repos/" + REPO
RAW_FRAMES = "https://raw.githubusercontent.com/" + REPO + "/frames/"
SITE = Path(__file__).resolve().parents[2]

# homepage href -> (normal interval in hours, source); a stamp is (file on main, field)
PRODUCTS = {
    "/enso.html": (24, {"stamp": ("assets/sst/data/enso_daily.json", "generated")}),
    "/skewt/": (6, {"branch": "skewt-data"}),
    "/climate.html": (744, {"json": "assets/climate/anim/data/meta.json", "field": "generated"}),
    "/ar.html": (12, {"stamp": ("assets/ar/anim/ar_det_manifest.json", "generated")}),
    "/mjo.html": (12, {"stamp": ("assets/mjo/rmm_manifest.json", "ver")}),
    "/circulation.html": (24, {"stamp": ("assets/sst/anim/aam_zonal_manifest.json", "ver")}),
    # AIFS-ENS (strat.yml) and IFS-ENS (strat-ifs.yml) products: neither records more than its cycle
    "/stratosphere.html": (
        12,
        {"stamps": [("assets/sst/data/heatflux100.json", "init"), ("assets/sst/data/ifs_heatflux100.json", "init")]},
    ),
    "/subseasonal.html": (96, {"stamp": ("assets/geps/telecon.json", "generated")}),
    "/gefs.html": (24, {"stamp": ("assets/gefs/data/gefs.json", "generated")}),
    "/enso-forecasts.html": (744, {"stamp": ("assets/sst/data/enso_forecast.json", "generated")}),
    # seasonal.html holds the C3S/SEAS5 and NOAA SFS groups: dated by the newer of the two
    "/seasonal.html": (
        744,
        {
            "stamps": [
                ("assets/sst/data/c3s_indices.json", "generated"),
                ("assets/sfs/data/sfs_indices.json", "generated"),
            ]
        },
    ),
    "/aifs-verify.html": (12, {"stamp": ("assets/verify/aifs_scores_v2.json", "generated")}),
}


def late_after(every_h: float) -> float:
    if every_h <= 24:
        return 2 * every_h + 6
    if every_h <= 168:
        return every_h + 24
    return every_h + 168


def get(url: str, api: bool = True):
    headers = {"User-Agent": "scorvec-status"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if api:
        headers["Accept"] = "application/vnd.github+json"
        if tok:
            headers["Authorization"] = "Bearer " + tok
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return json.loads(r.read())


def iso(s) -> str:
    """Any of the timestamp shapes the products use -> UTC ISO 8601 with a Z. Numbers are epoch seconds."""
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return dt.datetime.fromtimestamp(s, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    s = str(s).strip().replace(" UTC", "")
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%MZ",
        "%Y-%m-%dT%HZ",
        "%Y-%m-%d %HZ",
    ):
        try:
            t = dt.datetime.strptime(s, fmt)
            t = t.replace(tzinfo=dt.UTC) if t.tzinfo is None else t.astimezone(dt.UTC)
            return t.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp {s!r}")


def committed(path: str) -> str:
    c = get(f"{API}/commits?sha=main&per_page=1&path={path}")
    return iso(c[0]["commit"]["committer"]["date"])


def stamped(path: str, field: str) -> tuple[str, str]:
    """(time, basis) from the product's own field; the file's last commit when the field cannot be read."""
    try:
        return iso(json.loads((SITE / path).read_text())[field]), f"{path}:{field}"
    except Exception as e:  # noqa: BLE001  a missing or reshaped file falls back to its commit time
        print(f"  {path}:{field} unreadable ({str(e)[:80]}); dating it by its last commit", file=sys.stderr)
        return committed(path), f"{path}:commit"


def updated(src: dict) -> tuple[str, str]:
    if "stamps" in src:  # a page with several producers: its newest publish
        got = []
        for path, field in src["stamps"]:
            try:
                got.append(stamped(path, field))
            except Exception as e:  # noqa: BLE001  one missing file must not hide the other
                print(f"  {path}: {str(e)[:120]}", file=sys.stderr)
        if not got:
            raise ValueError(f"no source dated: {src['stamps']}")
        return max(got)
    if "stamp" in src:
        return stamped(*src["stamp"])
    if "branch" in src:
        return iso(get(f"{API}/commits/{src['branch']}")["commit"]["committer"]["date"]), f"{src['branch']}:commit"
    if "json" in src:
        return iso(get(RAW_FRAMES + src["json"], api=False)[src["field"]]), f"frames:{src['json']}:{src['field']}"
    raise ValueError(src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="status.json")
    a = ap.parse_args()
    out = {"generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "products": {}}

    def one(item):
        href, (every, src) = item
        try:
            when, basis = updated(src)
            return href, {"updated": when, "every_h": every, "late_h": late_after(every), "basis": basis}
        except Exception as e:  # noqa: BLE001  one bad source must not sink the rest
            print(f"  {href}: {str(e)[:120]}", file=sys.stderr)
            return href, None

    with ThreadPoolExecutor(6) as ex:
        for href, rec in ex.map(one, PRODUCTS.items()):
            if rec:
                out["products"][href] = rec
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"status: {len(out['products'])}/{len(PRODUCTS)} products dated -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
