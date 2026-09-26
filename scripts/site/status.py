#!/usr/bin/env python3
"""Product freshness for the homepage: when each product last published (2026-09-26; user: "Get to work on the
'last updated' part" — the homepage said how OFTEN each product updates, never WHETHER it had).

Runs in pages.yml on every deploy and writes status.json into the site artifact (never committed). Each product is
dated from what it actually publishes:
  path    the last commit on main touching the product's data file (GitHub API, the workflow's own token)
  branch  the head commit of a data branch that is force-pushed as a single commit per update (skewt-data)
  json    a timestamp field inside a file on the frames branch (the climate trends metadata)
The loop frames themselves cannot be dated this way: the frames branch is one parentless commit shared by every loop,
so its date says when ANY loop last published. Products are dated by their main-branch manifest or data file instead.

`every_h` is the product's normal interval; the homepage calls a product late past 2 x every_h + 6 h (Actions
schedules slip by hours, so anything tighter would cry wolf).

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

REPO = "scorvec/scorvec.github.io"
API = "https://api.github.com/repos/" + REPO
RAW_FRAMES = "https://raw.githubusercontent.com/" + REPO + "/frames/"

# homepage href -> (normal interval in hours, source)
PRODUCTS = {
    "/enso.html":            (24,  {"path": "assets/sst/data/enso_daily.json"}),
    "/skewt/":               (6,   {"branch": "skewt-data"}),
    "/qbo/":                 (168, {"path": "qbo/qbo.json"}),
    "/climate.html":         (744, {"json": "assets/climate/anim/data/meta.json", "field": "generated"}),
    "/ar.html":              (12,  {"path": "assets/ar/ar_monitor.json"}),
    "/cities/":              (6,   {"path": "cities/data/cities.json"}),
    "/ecape.html":           (6,   {"path": "assets/ecape/anim/index.json"}),
    "/mjo.html":             (12,  {"path": "assets/mjo/rmm_manifest.json"}),
    "/circulation.html":     (24,  {"path": "assets/sst/aam.webp"}),
    "/stratosphere.html":    (12,  {"path": "assets/sst/nh_vortex.webp"}),
    "/subseasonal.html":     (96,  {"path": "assets/geps/telecon.json"}),
    "/gefs.html":            (24,  {"path": "assets/gefs/data/gefs.json"}),
    "/enso-forecasts.html":  (744, {"path": "assets/sst/data/enso_forecast.json"}),
    "/seasonal.html":        (744, {"path": "assets/sst/data/c3s_indices.json"}),
    "/sfs.html":             (744, {"path": "assets/sfs/data/sfs_indices.json"}),
    "/cities/verify.html":   (6,   {"path": "cities/data/verify.json"}),
    "/aifs-verify.html":     (12,  {"path": "assets/verify/aifs_scores_v2.json"}),
}


def get(url: str, api: bool = True):
    headers = {"User-Agent": "scorvec-status"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if api:
        headers["Accept"] = "application/vnd.github+json"
        if tok:
            headers["Authorization"] = "Bearer " + tok
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return json.loads(r.read())


def iso(s: str) -> str:
    """Any of the timestamp shapes the sources use -> UTC ISO 8601 with a Z."""
    s = s.strip().replace(" UTC", "")
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            t = dt.datetime.strptime(s, fmt)
            t = t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t.astimezone(dt.timezone.utc)
            return t.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp {s!r}")


def updated(src: dict) -> str:
    if "path" in src:
        c = get(f"{API}/commits?sha=main&per_page=1&path={src['path']}")
        return iso(c[0]["commit"]["committer"]["date"])
    if "branch" in src:
        return iso(get(f"{API}/commits/{src['branch']}")["commit"]["committer"]["date"])
    if "json" in src:
        return iso(get(RAW_FRAMES + src["json"], api=False)[src["field"]])
    raise ValueError(src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="status.json")
    a = ap.parse_args()
    out = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "products": {}}

    def one(item):
        href, (every, src) = item
        try:
            return href, {"updated": updated(src), "every_h": every}
        except Exception as e:                                   # noqa: BLE001  one bad source must not sink the rest
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
