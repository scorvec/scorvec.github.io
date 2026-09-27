#!/usr/bin/env python3
"""Every plot on the site, as one index: assets/site/catalog.json (2026-09-27, the topic redesign; user: "reorganize
things and change the interface for finding certain plots").

Feeds the header finder (Ctrl/Cmd-K, assets/site.js) and the catalogue page (catalog.html). Built from the pages
themselves so it never goes stale:
  stage pages (window.STAGE) and the GEPS/GEFS pages (var P / GROUPS)  ->  scripts/site/catalog_eval.js runs their
      inline scripts in node with the DOM stubbed out and calls the spec's own functions, so option lists, image
      URLs, loop manifests and captions are the real ones, not regex guesses;
  rail pages (<main data-rail>)  ->  every .card with its data-label / data-group / id (rail.js makes the same ids);
  single-figure pages  ->  one entry each, described from the chrome's PRODUCTS table.
Each product gets exactly ONE topic (the page's, with group-level overrides such as the ENSO page's CMIP6 impacts
under Research), tags for the filters (model, range, region, variable), search keywords from its About text, deep
links for its option combinations (#product/a/b/c, the stage viewer's own hash format) and, for the product and for
EVERY option combination, the figure that combination shows: a key into the `thumbs` table, whose entries name the
source (a file on main, or the first frame of a loop on the frames branch) and, once scripts/site/build_thumbs.py has
made it, a ~360 px WebP thumbnail on the frames branch (assets/site/thumbs/).

Default order (the catalogue's "Latest first"): `tier` 0 = live forecasts updated daily or more, 1 = research,
history, verification and tools, 2 = monthly seasonal outlooks and anything out of season (the snow bands outside
November to mid-April); `prio` orders pages inside a tier. The page adds freshness from /status.json.

Privacy: only the pages listed below are read, and any keyword text naming the private boards is dropped;
the build fails if a private name reaches a label.

    python scripts/site/build_catalog.py            # -> assets/site/catalog.json
    python scripts/site/build_catalog.py --check    # build, print a summary, write nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import html as H
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_chrome as A                                                       # noqa: E402

OUT = REPO / "assets" / "site" / "catalog.json"
EVAL = Path(__file__).resolve().parent / "catalog_eval.js"
THUMB_DIR = "assets/site/thumbs"                       # on the frames branch
THUMB_INDEX_URL = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/frames/" + THUMB_DIR + "/index.json"

# ids stay stable (they are in catalogue URLs, ?topic=...); labels are the menu group titles in apply_chrome.PRODUCTS
TOPICS = [
    ("weather", "Short and medium range", "Days 1 to 15: storms, hazards and the temperature forecast"),
    ("outlooks", "Subseasonal and seasonal", "From weeks two to five out to the coming seasons"),
    ("drivers", "Climate drivers", "The modes: ENSO, the MJO and the QBO, as they stand now"),
    ("circulation", "Atmospheric circulation", "The stratosphere and the troposphere's jets, waves and overturning cells"),
    ("research", "Studies and verification", "What the record and the models say, and how the forecasts score"),
    ("tools", "Tools", "Explore the data yourself"),
]
HORIZONS = ["Observed", "Days 1–15", "Weeks 1–5", "Seasons", "Climate record"]
TIERS = ["Live forecasts and monitors", "Research, history and tools", "Seasonal and out of season"]


def snow_season(today: dt.date | None = None) -> bool:
    """The snow-band products run November to mid-April (snowband.yml); outside that they are case studies."""
    d = today or dt.date.today()
    return d.month in (11, 12, 1, 2, 3) or (d.month == 4 and d.day <= 15)


# page -> kind, topic, default tags, tier/prio; `groups` overrides them for one rail group on that page.
# `region` is the page's default region, used only when a product's own label, options and caption name none.
PAGES = {
    "/enso.html": dict(kind="stage", file="enso.html", topic="drivers", title="El Niño monitor", tier=0, prio=0,
                       models=["OISST"], horizon="Observed", region=["Tropical Pacific"],
                       groups={"Impacts on the Americas": dict(topic="research", horizon="Climate record", models=["CMIP6", "ERA5"],
                                                               region=["North America", "South America"], tier=1, prio=20)}),
    "/stratosphere.html": dict(kind="stage", file="stratosphere.html", topic="circulation", title="Stratosphere and polar vortex",
                               tier=0, prio=1, models=["AIFS-ENS"], horizon="Days 1–15", region=["Northern Hemisphere"],
                               groups={"Stratosphere history": dict(topic="research", horizon="Climate record", models=["MERRA-2"],
                                                                    tier=1, prio=21)}),
    "/circulation.html": dict(kind="stage", file="circulation.html", topic="circulation", title="Jets, Walker and Hadley cells",
                              tier=0, prio=4, models=["AIFS-ENS"], horizon="Days 1–15", region=["Global"]),
    "/snowbands.html": dict(kind="stage", file="snowbands.html", topic="weather", title="Snow-band diagnostics",
                            tier=0 if snow_season() else 2, prio=5 if snow_season() else 40,
                            models=["HRRR", "RRFS", "RDPS"], horizon="Days 1–15", region=["North America"]),
    "/subseasonal.html": dict(kind="geps", file="subseasonal.html", topic="outlooks", title="GEPS weeks 1–5", tier=0, prio=3,
                              models=["GEPS"], horizon="Weeks 1–5", region=["North America"], own_models=True),
    "/gefs.html": dict(kind="geps", file="gefs.html", topic="outlooks", title="GEFS weeks 1–5", tier=0, prio=3,
                       models=["GEFS"], horizon="Weeks 1–5", region=["North America"], own_models=True),
    "/seasonal.html": dict(kind="rail", file="seasonal.html", topic="outlooks", title="Eight C3S seasonal models", tier=2, prio=30,
                           models=["C3S", "SEAS5"], horizon="Seasons", region=["Global"]),
    "/sfs.html": dict(kind="rail", file="sfs.html", topic="outlooks", title="NOAA SFS seasonal", tier=2, prio=31,
                      models=["SFS"], horizon="Seasons", region=["Global"]),
    # single-figure / app pages: one entry each (label and description from the chrome's PRODUCTS table)
    "/mjo.html": dict(kind="page", topic="drivers", tier=0, prio=2, models=["AIFS-ENS"], horizon="Days 1–15", region=["Tropics"],
                      thumb_manifest=("assets/mjo", "rmm_manifest.json", "mjo", "last"), variable=["Convection"]),
    "/ar.html": dict(kind="page", topic="weather", tier=0, prio=6, models=["AIFS-ENS"], horizon="Days 1–15",
                     region=["North America", "Pacific"], thumb="assets/ar/ar_now.webp", variable=["Moisture transport"]),
    "/ecape.html": dict(kind="page", topic="weather", tier=0, prio=7, models=["HRRR"], horizon="Days 1–15", region=["North America"],
                        thumb_ecape=True, variable=["Instability"]),
    # "/cities/" paused 2026-09-27 (city forecasts turned off); restore from git history
    "/qbo/": dict(kind="page", topic="drivers", tier=0, prio=9, models=["Radiosondes"], horizon="Observed", region=["Tropics"],
                  thumb="assets/qbo/qbo_section.webp", variable=["Stratosphere", "Wind"]),
    "/enso-forecasts.html": dict(kind="page", topic="outlooks", tier=2, prio=29, models=["Multi-model"], horizon="Seasons",
                                 region=["Tropical Pacific"], variable=["SST"]),
    # "/cities/verify.html" paused 2026-09-27
    "/aifs-verify.html": dict(kind="page", topic="research", tier=1, prio=23, models=["AIFS", "AIFS-ENS", "ERA5"], horizon="Days 1–15",
                              region=["Northern Hemisphere"], thumb="assets/verify/tt_compare_d05.webp", variable=["Height", "Temperature"]),
    "/topics/": dict(kind="page", topic="research", tier=1, prio=26, models=[], horizon=None, region=[]),
    "/research.html": dict(kind="page", topic="research", tier=1, prio=27, models=[], horizon=None, region=[]),
    "/catalog.html": dict(kind="page", topic="tools", tier=1, prio=28, models=[], horizon=None, region=[]),
    "/skewt/": dict(kind="page", topic="tools", tier=1, prio=24, models=["Radiosondes"], horizon="Observed", region=["Global"],
                    thumb="skewt/og-card.png", variable=["Soundings"]),
    "/asos5.html": dict(kind="page", topic="tools", tier=1, prio=25, models=["ASOS"], horizon="Observed", region=["North America"],
                        variable=["Temperature"]),
    "/climate.html": dict(kind="page", topic="tools", tier=1, prio=25, models=["nClimDiv", "PRISM"], horizon="Climate record",
                          region=["North America"], variable=["Temperature", "Precipitation"]),
}

# tag vocabularies. Models and regions are read from what a product SHOWS - its label, group, option labels and
# caption - never from the About prose, which names other models and regions to compare or explain ("unlike GEPS's",
# "the Niño-3.4 box"); read from the prose, "Tropical Pacific" had landed on 55 of 125 plots and "Global" on 50.
MODEL_RX = [
    ("AIFS-ENS", r"\bAIFS[- ]ENS\b|\bAIFS ensemble\b"), ("AIFS", r"\bAIFS single\b|\bAIFS\b(?![- ](?:ENS|[Ee]nsemble))"),
    ("IFS", r"\bIFS\b"), ("GEPS", r"\bGEPS\b"), ("GEFS", r"\bGEFS\b"), ("GDPS", r"\bGDPS\b"), ("GFS", r"\bGFS\b"),
    ("HRRR", r"\bHRRR\b"), ("RRFS", r"\bRRFS\b"), ("RDPS", r"\bRDPS\b"), ("GEOS FP", r"\bGEOS[ -]FP\b"),
    ("MERRA-2", r"\bMERRA-?2\b"), ("ERA5", r"\bERA5\b"), ("CMIP6", r"\bCMIP6\b"), ("SEAS5", r"\bSEAS5\b"),
    ("C3S", r"\bC3S\b"), ("SFS", r"\bSFS\b"), ("OISST", r"\bOISST\b"), ("ERSST", r"\bERSST"), ("NCEP R1", r"\bR1\b|Reanalysis\s*1"),
    ("GOES", r"\bGOES\b|\bGMGSI\b"), ("IMERG", r"\bIMERG\b"), ("TAO", r"\bTAO\b"), ("CPC", r"\bCPC\b"),
]
REGION_RX = [
    ("North America", r"North America|\bUS\b|United States|CONUS|Canada|West Coast|Alaska|Great Lakes|Northeast|Mid-Atlantic|Midwest|Rockies"),
    ("South America", r"South America|Amazon|Andes|\bBrazil"), ("Europe", r"\bEurope|Euro-Atlantic"),
    ("Tropics", r"\btropics\b|\btropical belt|\bequator|\bMJO\b"),
    ("Tropical Pacific", r"Ni[nñ]o[- ]?[1-4]|equatorial Pacific|Tropical Pacific|Kiribati|Tarawa|\bTAO\b|warm pool|cold tongue"),
    ("Pacific", r"North Pacific|\bPacific jet|Pacific basin"), ("Atlantic", r"\bAtlantic\b"),
    ("Northern Hemisphere", r"Northern Hemisphere|\bNH\b|\bArctic|polar cap|\bNorthern\b|polar vortex"),
    ("Southern Hemisphere", r"Southern Hemisphere|\bSH\b|Antarctic|\bSouthern\b"),
    ("Global", r"\bglobal map|\bGlobal\b(?! mean)|whole globe|\bworld"),
]
VAR_RX = [
    ("Temperature", r"temperature|\bt2m\b|2 m\b|\bT850\b|warm|cold"), ("Precipitation", r"precipitation|rainfall|\brain\b|\bpr\b"),
    ("Height", r"\bz500\b|500 hPa|geopotential|height"), ("Wind", r"\bwind|\bjet\b|u850|u200|zonal-mean wind"),
    ("SST", r"\bSST\b|sea surface"), ("Convection", r"\bOLR\b|convection|infrared|\bMJO\b|χ|velocity potential"),
    ("Stratosphere", r"stratospher|vortex|10 hPa|heat flux|E[–-]P flux|QBO|Brewer"), ("Snow", r"\bsnow"),
    ("Pressure", r"\bMSLP\b|sea-level pressure|\bSOI\b"), ("Teleconnections", r"teleconnection|\bNAO\b|\bAO\b|\bPNA\b|regime"),
]
# thumbnails for figures whose images are injected by script (the seasonal pages' Plotly-free cards): a stable file,
# a glob resolved at build time (dated names rotate monthly; the catalogue rebuilds daily), or a loop manifest
THUMBS = {
    "/seasonal.html#c3sMapCard": "assets/sst/c3s/c3s_mmm_t2m_s1.webp",
    "/seasonal.html#c3sTercCard": "assets/sst/c3s/c3s_terc_mmm_t2m_s1.webp|assets/sst/c3s/c3s_terc_*_t2m_s1.webp",
    "/seasonal.html#changeCard": "assets/sst/seas5_map_t2m_anom_*.webp",
    "/seasonal.html#calibrated-terciles": "assets/sst/seas5_terc_t2m_*.webp",
    "/seasonal.html#normCard": "assets/sst/seas5_norm_z500_trend_terc_*.webp",
    "/seasonal.html#xdaysCard": "assets/sst/seas5_xdays_us_tn_0_*.webp",
    "/seasonal.html#snowCard": "assets/sst/seas5_snowtot_pct_*.webp",
    "/seasonal.html#windCard": "assets/sst/seas5_wind_vanom_*.webp",
    "/sfs.html#monthly-anomaly-maps": ("manifest", "assets/sfs/anim", "sfs_t2m_manifest.json"),
}
# forecast systems: on a page that is one system's own product (own_models=True) the captions name the others only to
# compare ("unlike GEPS's"), so they are not tagged from the text there
SYSTEMS = {"AIFS-ENS", "AIFS", "IFS", "GEPS", "GEFS", "GDPS", "GFS", "HRRR", "RRFS", "RDPS", "SEAS5", "C3S", "SFS"}
# private boards and products that must never reach the public index (labels are checked; keyword text is filtered)
PRIVATE = re.compile(r"gat[uú]n|colombia|brazil|brasil|hydro board|/hydro/|\bXM\b|\bONS\b|pjm|nyiso|\boil\b|refin|early.?vote", re.I)


def run_eval(path: Path) -> dict:
    r = subprocess.run(["node", str(EVAL), str(path)], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise SystemExit(f"catalog_eval failed on {path.name}: {r.stderr[:400]}")
    return json.loads(r.stdout)


def text_of(fragment: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", fragment, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", H.unescape(t)).strip()


def templates(html: str) -> dict:
    return {m.group(1): text_of(m.group(2)) for m in re.finditer(r'<template id="([^"]+)">(.*?)</template>', html, re.S)}


def public(t: str) -> str:
    """Drop sentences that name a private board (keyword text only; labels are asserted clean separately)."""
    return " ".join(s for s in re.split(r"(?<=[.;!?])\s+", t) if not PRIVATE.search(s))


def tags(rx, text, base=(), own=False):
    out = list(base)
    for name, pat in rx:
        if own and name in SYSTEMS:
            continue
        if name not in out and re.search(pat, text):
            out.append(name)
    return out


def regions_of(text: str, default) -> list:
    """The regions a product's own label, options and caption name; the page default only when they name none."""
    found = tags(REGION_RX, text)
    return found or list(default)


_TRACKED = None


def tracked() -> set:
    """Every file git tracks under assets/ and skewt/: the catalogue workflow checks out only the pages and the JSON
    (sparse), so a figure's existence is read from the index, not the disk."""
    global _TRACKED
    if _TRACKED is None:
        try:
            out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "assets", "skewt"],
                                 capture_output=True, text=True, check=True).stdout
            _TRACKED = set(out.split("\0")) - {""}
        except (OSError, subprocess.CalledProcessError):
            _TRACKED = set()
    return _TRACKED


def exists(rel: str) -> bool:
    return rel in tracked() or (REPO / rel).exists()


# ── the figure table: one entry per distinct source, keyed by a short hash of it ─────────────────────────────────
THUMB_TABLE: dict = {}


def thumb_key(rel: str, frames: bool) -> str | None:
    """Register a figure (a repo path on main, or a frame on the frames branch) and return its key."""
    if not rel:
        return None
    rel = rel.split("?")[0].split("#")[0].lstrip("/")
    if not re.search(r"\.(webp|png|jpe?g|gif)$", rel, re.I):
        return None
    if not frames and not exists(rel):
        return None
    k = hashlib.sha1(("f:" if frames else "m:").encode() + rel.encode()).hexdigest()[:10]
    THUMB_TABLE.setdefault(k, {"s": rel, **({"f": 1} if frames else {})})
    return k


def thumb_name(rel: str) -> str:
    """The thumbnail's id: stable per source path. It lives at assets/site/thumbs/<id>.webp on the frames branch."""
    return hashlib.sha1(rel.encode()).hexdigest()[:12]


def frame_source(url: str, page_dir: Path):
    """sst_anim embed URL -> (rel, on_frames_branch) for the first frame of that loop."""
    u = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    base = q.get("base", "assets/sst/anim")
    man = q.get("manifest", "manifest.json")
    try:
        M = json.loads((page_dir / base / man).read_text())
    except Exception:
        return None
    region = q.get("region") or M.get("default") or next(iter(M.get("regions", {})), None)
    R = M.get("regions", {}).get(region)
    if not R or not R.get("frames"):
        return None
    # a forecast loop (F00, F01 ...) opens on its first frame; an observed loop (dated frames) on its newest, which is
    # also the one frame certain to still be on the branch (older ones are pruned as the loop rolls)
    fr = R["frames"]
    f = fr[0]["file"] if re.match(r"F\d+\.", fr[0]["file"]) else fr[-1]["file"]
    rel = f"{base}/{region}/{f}"
    return rel, (base.endswith("/anim") and "//" not in base)


def fig_key(fig: dict | None, page_dir: Path) -> str | None:
    """None: the view is drawn in the browser (no figure); "": it names a figure that is not on the site yet (a new
    product before its first run); else the figure's key."""
    if not fig:
        return None
    if fig.get("img"):
        return thumb_key(fig["img"], False) or ""
    if fig.get("frame"):
        src = frame_source(fig["frame"], page_dir)
        return (thumb_key(*src) if src else None) or ""
    return None


def stage_items(href: str, cfg: dict) -> list:
    path = REPO / cfg["file"]
    html = path.read_text()
    spec = run_eval(path)
    tpl = templates(html)
    js = lambda x: "null" if x is None else str(x)                       # JS object-key stringification
    items = []
    for g in spec["groups"]:
        gcfg = cfg.get("groups", {}).get(g["label"], {})
        for pid, _lbl, *_ in g["items"]:
            p = spec["products"].get(pid)
            if not p:
                continue
            about = tpl.get(p.get("about") or "", "")
            figs = p.get("figBy", {})
            # option combinations -> deep links (the stage viewer's #product/a/b/c) and each one's own figure
            variants = []
            A_ = p.get("a")
            a_items = A_["items"] if A_ else [[None, None]]
            for av, al in a_items:
                B = p["bBy"].get(js(av))
                b_items = B["items"] if B else [[None, None]]
                for bv, bl in b_items:
                    C = p["cBy"].get(f"{js(av)}|{js(bv)}")
                    c_items = C["items"] if C else [[None, None]]
                    for cv, cl in c_items:
                        parts = [x for x in (av, bv, cv) if x is not None]
                        labs = [x for x in (al, bl, cl) if x]
                        if labs:
                            k = fig_key(figs.get(f"{js(av)}|{js(bv)}|{js(cv)}"), path.parent)
                            variants.append([" · ".join(text_of(str(x)) for x in labs),
                                             "#" + "/".join([pid] + [str(x) for x in parts]), k])
            if len(variants) > 80:                       # huge option grids: keep the first axis and the defaults
                seen, keep = set(), []
                for v in variants:
                    first = v[0].split(" · ")[0]
                    if first not in seen:
                        seen.add(first); keep.append(v)
                variants = keep[:80]
            d = [js(x) for x in p.get("def", [None, None, None])]
            thumb = fig_key(figs.get("|".join(d)), path.parent)
            if thumb is None and p["kind"] == "img" and p.get("img"):
                thumb = thumb_key(p["img"], False) or ""
            opts_text = " ".join(v[0] for v in variants)
            shows = " ".join([p["label"], g["label"], p.get("sub", ""), p.get("cap", ""), opts_text])
            blob = shows + " " + about[:1500]
            items.append(dict(
                id=f"{href}#{pid}", page=href, page_title=cfg["title"], group=text_of(g["label"]),
                label=text_of(p["label"]), sub=text_of(p.get("sub", "")), url=f"{href}#{pid}",
                topic=gcfg.get("topic", cfg["topic"]), horizon=gcfg.get("horizon", cfg["horizon"]),
                models=tags(MODEL_RX, shows, gcfg.get("models", cfg["models"]), cfg.get("own_models", False)),
                regions=regions_of(shows, gcfg.get("region", cfg["region"])),
                variables=tags(VAR_RX, blob), thumb=thumb, cap=text_of(p.get("cap", ""))[:240],
                kw=public(about)[:600], variants=variants, live=True,
                tier=gcfg.get("tier", cfg["tier"]), prio=gcfg.get("prio", cfg["prio"]),
                kind="chart" if p["kind"] == "dom" else "fig"))
    return items


def rail_items(href: str, cfg: dict) -> list:
    """rail.js pages: every element the page's data-rail selector names (".card" on both rail pages)."""
    path = REPO / cfg["file"]
    html = path.read_text()
    main = re.search(r"<main[^>]*data-rail[^>]*>(.*)</main>", html, re.S)
    body = main.group(1) if main else html
    starts = [m for m in re.finditer(r'<(div|section|article)\s+class="card\b[^"]*"([^>]*)>', body)]
    used, items = set(), []
    for i, m in enumerate(starts):
        chunk = body[m.start(): starts[i + 1].start() if i + 1 < len(starts) else len(body)]
        attrs = m.group(0)
        def attr(n):
            a = re.search(rf'{n}="([^"]*)"', attrs)
            return H.unescape(a.group(1)) if a else ""
        label = attr("data-label") or text_of((re.search(r"<h[23][^>]*>(.*?)</h[23]>", chunk, re.S) or [None, ""])[1]) or "Figure"
        if label == "About this page":
            continue
        pid = attr("id") or re.sub(r"^-|-$", "", re.sub(r"[^a-z0-9]+", "-", re.sub(r"&[a-z]+;", "", label.lower())))[:60]
        while pid in used:
            pid += "-2"
        used.add(pid)
        img = re.search(r'<img[^>]+(?:data-src|src)="([^"]+\.(?:webp|png|jpg))[^"]*"', chunk)
        thumb = thumb_key(img.group(1), False) if img and not img.group(1).startswith("data:") else None
        about = text_of(chunk)
        head = text_of((re.search(r"<h[23][^>]*>(.*?)</h[23]>", chunk, re.S) or [None, ""])[1])
        shows = " ".join([label, attr("data-group"), head])
        items.append(dict(
            id=f"{href}#{pid}", page=href, page_title=cfg["title"], group=attr("data-group") or "Figures",
            label=label, sub="", url=f"{href}#{pid}", topic=cfg["topic"], horizon=cfg["horizon"],
            models=tags(MODEL_RX, shows, cfg["models"]), regions=regions_of(shows, cfg["region"]),
            variables=tags(VAR_RX, label + " " + about[:1500]), thumb=thumb, cap="", kw=public(about)[:600], variants=[],
            live=True, tier=cfg["tier"], prio=cfg["prio"], kind="fig" if thumb else "chart"))
    return items


def ecape_thumb():
    """ECAPE: a mid-afternoon frame of the newest cycle's default field (its frames are on the frames branch)."""
    try:
        idx = json.loads((REPO / "assets/ecape/anim/index.json").read_text())
        M = json.loads((REPO / "assets/ecape/anim" / idx["cycles"][0]["manifest"]).read_text())
        region = M.get("default") or next(iter(M["regions"]))
        fr = M["regions"][region]["frames"]
        return thumb_key(f"assets/ecape/anim/{region}/{fr[min(9, len(fr) - 1)]['file']}", True)
    except Exception:
        return None


def mjo_items() -> list:
    """mjo.html's two sections below the forecast (2026-09-27; the user could not find the impacts): the ENSO-removed
    index, and the impacts composites with a deep link and a figure for every selector combination (field x data x
    month x lag x view, the page's own #mi/f/d/mm/l/v hash), the current month first as on the page."""
    path = REPO / "mjo.html"
    try:
        html = path.read_text()
    except OSError:
        return []
    items = []
    sec = re.search(r'<section id="mjo-clean".*?</section>', html, re.S)
    if sec:
        body = sec.group(0)
        head = text_of((re.search(r"<h2>(.*?)</h2>", body, re.S) or [None, "MJO with the El Niño signal removed"])[1])
        lead = text_of((re.search(r"</h2>\s*<p[^>]*>(.*?)</p>", body, re.S) or [None, ""])[1])
        thumb = None
        try:
            M = json.loads((REPO / "assets/mjo/rmmclean_manifest.json").read_text())
            thumb = thumb_key(f"assets/mjo/{M['regions']['mjo']['frames'][-1]['file']}", False) or ""
        except Exception:
            pass
        shows = " ".join([head, lead])
        items.append(dict(
            id="/mjo.html#mjo-clean", page="/mjo.html", page_title="MJO forecast", group="ENSO-removed index", label=head,
            sub="ENSO-removed RMM, wind-only", url="/mjo.html#mjo-clean", topic="drivers", horizon="Days 1–15",
            models=tags(MODEL_RX, shows, ["AIFS-ENS"]), regions=["Tropics"], variables=["Convection", "Wind"], thumb=thumb,
            cap=lead[:240], kw=public(text_of(body))[:600], variants=[], live=True, tier=0, prio=2, kind="fig"))
    sec = re.search(r'<section class="mi" id="mjo-impacts">.*?</section>', html, re.S)
    if sec:
        body = sec.group(0)
        def opts(sid):
            m = re.search(rf'<select id="{sid}">(.*?)</select>', body, re.S)
            return [(v, text_of(l)) for _q, v, l in re.findall(r'<option value=(["\'])([^"\']*)\1>(.*?)</option>', m.group(1))] if m else []
        # page order: the first option of each selector is the page's own default (DJF / CMIP6 / strip since v2)
        F, D, Mo, L, V = (opts(x) for x in ("mi-f", "mi-d", "mi-m", "mi-l", "mi-v"))
        lag_l = {"10": "days +8 to +12", "0": "days \u22122 to +2"}
        view_l = {"loop": "phase by phase", "strip": "all eight phases", "compare": "CMIP6 above observed"}
        try:
            MI = json.loads((REPO / "assets/mjo/impacts/anim/mjo_impacts_manifest.json").read_text())["regions"]
        except Exception:
            MI = {}
        def key_of(rid):
            R = MI.get(rid)
            if not R or not R.get("frames"):
                return ""
            return thumb_key(f"assets/mjo/impacts/anim/{rid}/{R['frames'][0]['file']}", True) or ""
        # the page's own region ids (mjo.html go()): _s = the eight-phase strip; "compare" stacks the CMIP6 strip over
        # the observed one and ignores the Data choice, so it is listed once, figured by its CMIP6 half
        variants = []
        for f, fl in F:
            for mo, ml in Mo:
                for l, _ll in L:
                    # region ids as mjo.html's go() builds them: "_djf_l10", "_m01_l10" (v2); plain month numbers were v1
                    tail = (f"_m{mo}" if mo.isdigit() else f"_{mo}") + f"_l{int(l):02d}"
                    for d, dl in D:
                        dshort = dl.split(" (")[0]
                        for v, _vl in V:
                            if v == "compare":
                                continue
                            variants.append([f"{fl} · {dshort} · {ml} · {lag_l.get(l, l)} · {view_l.get(v, v)}",
                                             f"#mi/{f}/{d}/{mo}/{l}/{v}", key_of(f"mi_{f}_{d}{tail}" + ("_s" if v == "strip" else ""))])
                    if any(v == "compare" for v, _ in V):
                        d0 = D[0][0] if D else "cmip6"
                        variants.append([f"{fl} · {ml} · {lag_l.get(l, l)} · {view_l['compare']}",
                                         f"#mi/{f}/{d0}/{mo}/{l}/compare", key_of(f"mi_{f}_cmip6{tail}_s")])
        head = text_of((re.search(r"<h2>(.*?)</h2>", body, re.S) or [None, "MJO impacts by phase and month"])[1]).split(":")[0].strip()
        lead = text_of((re.search(r'<p class="lede[^"]*">(.*?)</p>', body, re.S) or [None, ""])[1])
        default = variants[0] if variants else None
        shows = " ".join([head, lead, " ".join(fl for _, fl in F), " ".join(dl for _, dl in D)])
        items.append(dict(
            id="/mjo.html#mjo-impacts", page="/mjo.html", page_title="MJO forecast",
            group=re.sub(r"^MJO i", "I", head) if head.startswith("MJO impacts") else "Impacts",
            label=head, sub="Composites by MJO phase, season or month and lag: CMIP6 and observed",
            url="/mjo.html" + (default[1] if default else "#mjo-impacts"), topic="research", horizon="Climate record",
            models=tags(MODEL_RX, shows, ["CMIP6", "ERA5"]), regions=regions_of(shows, ["North America"]),
            variables=tags(VAR_RX, shows + " convection MJO"), thumb=default[2] if default else "",
            cap=lead[:240], kw=public("MJO composites by phase. " + text_of(body))[:600], variants=variants, live=True,
            tier=1, prio=19, kind="fig"))
    return items


def page_items() -> list:
    """Single-figure / app pages, described from the chrome's PRODUCTS table."""
    items = []
    for title, _blurb, grp, *_ in A.PRODUCTS:
        for href, label, what, _when in A.group_items(grp):
            cfg = PAGES.get(href)
            if not cfg or cfg["kind"] != "page":
                continue
            thumb = None
            if cfg.get("thumb"):
                thumb = thumb_key(cfg["thumb"], False)
            elif cfg.get("thumb_ecape"):
                thumb = ecape_thumb()
            elif cfg.get("thumb_manifest"):
                base, man, region, which = cfg["thumb_manifest"]
                try:
                    M = json.loads((REPO / base / man).read_text())
                    fr = M["regions"][region]["frames"]
                    thumb = thumb_key(f"{base}/{fr[-1 if which == 'last' else 0]['file']}", False)
                except Exception:
                    thumb = None
            blob = " ".join([label, what])
            items.append(dict(
                id=href, page=href, page_title=label, group="", label=label, sub="", url=href, topic=cfg["topic"],
                horizon=cfg.get("horizon"), models=tags(MODEL_RX, blob, cfg["models"]), regions=list(cfg["region"]),
                variables=tags(VAR_RX, blob, cfg.get("variable", [])), thumb=thumb, cap=what, kw=public(what), variants=[],
                live=False, tier=cfg["tier"], prio=cfg["prio"], kind="page"))
    return items


def resolve_thumb(spec):
    if isinstance(spec, tuple) and spec[0] == "manifest":
        _, base, man = spec
        src = frame_source(f"sst_anim.html?base={base}&manifest={man}", REPO)
        return thumb_key(*src) if src else None
    for pat in spec.split("|"):
        if "*" in pat:
            hits = sorted({str(p.relative_to(REPO)) for p in REPO.glob(pat)} | {f for f in tracked() if fnmatch.fnmatchcase(f, pat)})
        else:
            hits = [pat] if exists(pat) else []
        if hits:
            return thumb_key(hits[0], False)
    return None


def thumb_index(path: str | None) -> dict:
    """What build_thumbs.py has made: {source key string: {t, sig}}, from a local file or the frames branch."""
    try:
        if path:
            return json.loads(Path(path).read_text())
        with urllib.request.urlopen(THUMB_INDEX_URL + "?t=" + dt.datetime.now().strftime("%Y%m%d%H%M"), timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        # the frames branch unreachable (or raw.githubusercontent still serving a cached 404): keep the thumbnails the
        # committed index already points at, rather than a push-triggered rebuild dropping every one of them
        try:
            old = json.loads(OUT.read_text()).get("thumbs", {})
            prev = {("f:" if r.get("f") else "m:") + r["s"]: {"t": r["t"]} for r in old.values() if r.get("t")}
        except (OSError, ValueError):
            prev = {}
        print(f"  no thumbnail index ({e.__class__.__name__}); keeping the {len(prev)} thumbnails the current index names")
        return prev


def attach_thumbs(index: dict) -> int:
    n = 0
    for k, rec in THUMB_TABLE.items():
        hit = index.get(("f:" if rec.get("f") else "m:") + rec["s"])
        if hit and hit.get("t"):
            rec["t"] = hit["t"]
            n += 1
    return n


def build(index_path: str | None = None) -> dict:
    THUMB_TABLE.clear()
    items = []
    for href, cfg in PAGES.items():
        if cfg["kind"] == "stage":
            items += stage_items(href, cfg)
        elif cfg["kind"] == "geps":
            got = stage_items(href, cfg)
            for it in got:
                it["live"] = False                          # GEPS/GEFS pages read the hash only on load
            items += got
        elif cfg["kind"] == "rail":
            items += rail_items(href, cfg)
    items += page_items()
    items += mjo_items()
    for it in items:
        if not it["thumb"] and it["id"] in THUMBS:
            it["thumb"] = resolve_thumb(THUMBS[it["id"]])
            if it["thumb"] and it["kind"] == "chart":
                it["kind"] = "fig"
    # every product exactly once
    seen = set()
    uniq = []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"]); uniq.append(it)
    for it in uniq:
        for f in ("label", "group", "page_title", "sub"):
            if PRIVATE.search(it.get(f) or ""):
                raise SystemExit(f"private name in a public catalogue label: {it['id']} {f}={it[f]!r}")
        for v in it["variants"]:
            if PRIVATE.search(v[0]):
                raise SystemExit(f"private name in a variant: {it['id']} {v[0]!r}")
    # only figures something still points at
    used = {it["thumb"] for it in uniq} | {v[2] for it in uniq for v in it["variants"]}
    for it in uniq:                                  # a product whose default figure is not out yet: kind says so
        if it["thumb"] == "" and it["kind"] == "fig":
            it["kind"] = "pending"
    for k in list(THUMB_TABLE):
        if k not in used:
            del THUMB_TABLE[k]
    n_t = attach_thumbs(thumb_index(index_path))
    counts = {t: sum(1 for i in uniq if i["topic"] == t) for t, _, _ in TOPICS}
    # the menus' pages in menu order, for the finder's opening list (a site map in the palette)
    tid = {l: t for t, l, _ in TOPICS}
    pages = [dict(href=href, label=label, what=what, when=when, topic=tid[title])
             for title, _b, grp, *_ in A.PRODUCTS for href, label, what, when in A.group_items(grp)]
    for pg in pages:
        if PRIVATE.search(pg["label"] + " " + pg["what"]):
            raise SystemExit(f"private name in a menu page: {pg['href']}")
    print(f"  {len(THUMB_TABLE)} distinct figures, {n_t} with a thumbnail on the frames branch")
    return {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "topics": [dict(id=t, label=l, blurb=b, n=counts[t]) for t, l, b in TOPICS],
        "horizons": HORIZONS,
        "tiers": TIERS,
        "pages": pages,
        "thumbs": THUMB_TABLE,
        "items": uniq,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--thumb-index", help="a local index.json from build_thumbs.py (default: the frames branch copy)")
    a = ap.parse_args()
    cat = build(a.thumb_index)
    n = len(cat["items"]); nv = sum(len(i["variants"]) for i in cat["items"]); nt = sum(1 for i in cat["items"] if i["thumb"])
    pend = [i["id"] for i in cat["items"] if i["thumb"] == ""] + [f"{i['id']}{v[1]}" for i in cat["items"] for v in i["variants"] if v[2] == ""]
    if pend:
        print(f"  {len(pend)} view(s) name a figure not on the site yet: " + ", ".join(pend[:6]) + (" ..." if len(pend) > 6 else ""))
    print(f"{n} products, {nv} option views, {nt} with a figure; by topic " +
          ", ".join(f"{t['label']} {t['n']}" for t in cat["topics"]))
    if a.check:
        return 0
    # keep the old stamp when nothing else changed, so the workflow's "no changes" test holds on a quiet day
    try:
        old = json.loads(OUT.read_text())
        if {**old, "generated": None} == {**cat, "generated": None}:
            print("catalogue unchanged")
            return 0
    except (OSError, ValueError):
        pass
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cat, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
