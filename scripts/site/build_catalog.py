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
under Research), tags for the filters (model, range, region, variable), search keywords from its About text, a
thumbnail (a figure URL, or the first frame of a loop on the frames branch), and deep links for its option
combinations (#product/a/b/c, the stage viewer's own hash format).

Privacy: only the pages listed below are read, and any keyword text naming the private boards is dropped;
the build fails if a private name reaches a label.

    python scripts/site/build_catalog.py            # -> assets/site/catalog.json
    python scripts/site/build_catalog.py --check    # build, print a summary, write nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import html as H
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_chrome as A                                                       # noqa: E402

OUT = REPO / "assets" / "site" / "catalog.json"
EVAL = Path(__file__).resolve().parent / "catalog_eval.js"

TOPICS = [
    ("weather", "Weather", "The next two weeks, from single storms to the cities"),
    ("drivers", "Climate drivers", "ENSO, the MJO, the QBO, the stratosphere and the jets"),
    ("outlooks", "Outlooks", "From weeks two to five out to the coming seasons"),
    ("research", "Research", "What the record and the models say, and how the forecasts score"),
    ("tools", "Tools", "Explore the data yourself"),
]
HORIZONS = ["Observed", "Days 1–15", "Weeks 1–5", "Seasons", "Climate record"]

# page -> kind, topic, default tags; `groups` overrides the topic/range for a rail group on that page
PAGES = {
    "/enso.html": dict(kind="stage", file="enso.html", topic="drivers", title="El Niño monitor",
                       models=["OISST"], horizon="Observed", region=["Tropical Pacific"],
                       groups={"Impacts on the Americas": dict(topic="research", horizon="Climate record", models=["CMIP6", "ERA5"])}),
    "/circulation.html": dict(kind="stage", file="circulation.html", topic="drivers", title="Jets, Walker and Hadley cells",
                              models=["AIFS-ENS"], horizon="Days 1–15", region=["Global"]),
    "/stratosphere.html": dict(kind="stage", file="stratosphere.html", topic="drivers", title="Stratosphere and polar vortex",
                               models=["AIFS-ENS"], horizon="Days 1–15", region=["Northern Hemisphere"],
                               groups={"Stratosphere history": dict(topic="research", horizon="Climate record", models=["MERRA-2"])}),
    "/snowbands.html": dict(kind="stage", file="snowbands.html", topic="weather", title="Snow-band diagnostics",
                            models=["HRRR", "RRFS", "RDPS"], horizon="Days 1–15", region=["North America"]),
    "/subseasonal.html": dict(kind="geps", file="subseasonal.html", topic="outlooks", title="GEPS weeks 1–5",
                              models=["GEPS"], horizon="Weeks 1–5", region=["North America"], own_models=True),
    "/gefs.html": dict(kind="geps", file="gefs.html", topic="outlooks", title="GEFS weeks 1–5",
                       models=["GEFS"], horizon="Weeks 1–5", region=["North America"], own_models=True),
    "/seasonal.html": dict(kind="rail", file="seasonal.html", topic="outlooks", title="Eight C3S seasonal models",
                           models=["C3S", "SEAS5"], horizon="Seasons", region=["Global"]),
    "/sfs.html": dict(kind="rail", file="sfs.html", topic="outlooks", title="NOAA SFS seasonal",
                      models=["SFS"], horizon="Seasons", region=["Global"]),
    # single-figure / app pages: one entry each (label and description from the chrome's PRODUCTS table)
    "/ar.html": dict(kind="page", topic="weather", models=["AIFS-ENS"], horizon="Days 1–15", region=["North America", "Pacific"],
                     thumb="assets/ar/ar_now.webp", variable=["Moisture transport"]),
    "/ecape.html": dict(kind="page", topic="weather", models=["HRRR"], horizon="Days 1–15", region=["North America"], variable=["Instability"]),
    "/cities/": dict(kind="page", topic="weather", models=["Consensus", "NBM", "AIFS", "GEFS", "GEPS"], horizon="Days 1–15",
                     region=["North America"], variable=["Temperature"]),
    "/enso-forecasts.html": dict(kind="page", topic="drivers", models=["Multi-model"], horizon="Seasons", region=["Tropical Pacific"], variable=["SST"]),
    "/mjo.html": dict(kind="page", topic="drivers", models=["AIFS-ENS"], horizon="Days 1–15", region=["Tropics"],
                      thumb_manifest=("assets/mjo", "rmm_manifest.json", "mjo", "last"), variable=["Convection"]),
    "/qbo/": dict(kind="page", topic="drivers", models=["Radiosondes"], horizon="Observed", region=["Tropics"],
                  thumb="assets/qbo/qbo_section.webp", variable=["Stratosphere", "Wind"]),
    "/cities/verify.html": dict(kind="page", topic="research", models=["Consensus", "NBM", "AIFS"], horizon="Days 1–15",
                                region=["North America"], variable=["Temperature"]),
    "/aifs-verify.html": dict(kind="page", topic="research", models=["AIFS", "AIFS-ENS", "ERA5"], horizon="Days 1–15",
                              region=["Northern Hemisphere"], thumb="assets/verify/tt_compare_d05.webp", variable=["Height", "Temperature"]),
    "/topics/": dict(kind="page", topic="research", models=[], horizon=None, region=[]),
    "/research.html": dict(kind="page", topic="research", models=[], horizon=None, region=[]),
    "/catalog.html": dict(kind="page", topic="tools", models=[], horizon=None, region=[]),
    "/skewt/": dict(kind="page", topic="tools", models=["Radiosondes"], horizon="Observed", region=["Global"], variable=["Soundings"]),
    "/asos5.html": dict(kind="page", topic="tools", models=["ASOS"], horizon="Observed", region=["North America"], variable=["Temperature"]),
    "/climate.html": dict(kind="page", topic="tools", models=["nClimDiv", "PRISM"], horizon="Climate record", region=["North America"],
                          variable=["Temperature", "Precipitation"]),
}

# tag vocabularies, matched against a product's label, group, caption, option labels and About text
MODEL_RX = [
    ("AIFS-ENS", r"\bAIFS[- ]ENS\b|\bAIFS ensemble\b"), ("AIFS", r"\bAIFS single\b|\bAIFS\b(?![- ]ENS)"),
    ("IFS", r"\bIFS\b"), ("GEPS", r"\bGEPS\b"), ("GEFS", r"\bGEFS\b"), ("GDPS", r"\bGDPS\b"), ("GFS", r"\bGFS\b"),
    ("HRRR", r"\bHRRR\b"), ("RRFS", r"\bRRFS\b"), ("RDPS", r"\bRDPS\b"), ("GEOS FP", r"\bGEOS[ -]FP\b"),
    ("MERRA-2", r"\bMERRA-?2\b"), ("ERA5", r"\bERA5\b"), ("CMIP6", r"\bCMIP6\b"), ("SEAS5", r"\bSEAS5\b"),
    ("C3S", r"\bC3S\b"), ("SFS", r"\bSFS\b"), ("OISST", r"\bOISST\b"), ("ERSST", r"\bERSST"), ("NCEP R1", r"\bR1\b|Reanalysis\s*1"),
    ("GOES", r"\bGOES\b|\bGMGSI\b"), ("IMERG", r"\bIMERG\b"), ("TAO", r"\bTAO\b"), ("CPC", r"\bCPC\b"),
]
REGION_RX = [
    ("North America", r"North America|\bUS\b|United States|CONUS|Canada|West Coast|Alaska|Great Lakes"),
    ("South America", r"South America|Amazon|Andes"), ("Europe", r"\bEurope"), ("Tropics", r"\btropic|equator"),
    ("Tropical Pacific", r"Ni[nñ]o|equatorial Pacific|Tropical Pacific|Kiribati|Tarawa|TAO"),
    ("Pacific", r"\bPacific\b"), ("Atlantic", r"\bAtlantic\b"),
    ("Northern Hemisphere", r"Northern Hemisphere|\bNH\b|Arctic|polar cap|60 ?°N"),
    ("Southern Hemisphere", r"Southern Hemisphere|\bSH\b|Antarctic"), ("Global", r"\bglobal\b|\bGlobal\b|whole globe"),
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
PRIVATE = re.compile(r"gat[uú]n|colombia|brazil|brasil|hydro board|/hydro/|\bXM\b|\bONS\b|pjm|nyiso|\boil\b|refin", re.I)


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


def thumb_from_frame(url: str, page_dir: Path):
    """sst_anim embed URL -> the first frame of that loop: {frame: rel} on the frames branch, {src: rel} elsewhere."""
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
    f = R["frames"][0]["file"]
    rel = f"{base}/{region}/{f}"
    return {"frame": rel} if (base.endswith("/anim") and "//" not in base) else {"src": rel}


def stage_items(href: str, cfg: dict) -> list:
    path = REPO / cfg["file"]
    html = path.read_text()
    spec = run_eval(path)
    tpl = templates(html)
    items = []
    for g in spec["groups"]:
        gcfg = cfg.get("groups", {}).get(g["label"], {})
        for pid, _lbl, *_ in g["items"]:
            p = spec["products"].get(pid)
            if not p:
                continue
            about = tpl.get(p.get("about") or "", "")
            # option combinations -> deep links (the stage viewer's #product/a/b/c)
            variants = []
            A_ = p.get("a")
            a_items = A_["items"] if A_ else [[None, None]]
            js = lambda x: "null" if x is None else str(x)          # JS object-key stringification
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
                            variants.append([" · ".join(text_of(str(x)) for x in labs), "#" + "/".join([pid] + [str(x) for x in parts])])
            if len(variants) > 80:                       # huge option grids: keep the first axis and the defaults
                seen, keep = set(), []
                for v in variants:
                    k = v[0].split(" · ")[0]
                    if k not in seen:
                        seen.add(k); keep.append(v)
                variants = keep[:80]
            thumb = None
            if p["kind"] == "img" and p.get("img"):
                thumb = {"src": p["img"].lstrip("/")}
            elif p["kind"] == "frame" and p.get("frame"):
                thumb = thumb_from_frame(p["frame"], path.parent)
            opts_text = " ".join(v[0] for v in variants)
            blob = " ".join([p["label"], g["label"], p.get("sub", ""), p.get("cap", ""), opts_text, about[:1500]])
            items.append(dict(
                id=f"{href}#{pid}", page=href, page_title=cfg["title"], group=text_of(g["label"]),
                label=text_of(p["label"]), sub=text_of(p.get("sub", "")), url=f"{href}#{pid}",
                topic=gcfg.get("topic", cfg["topic"]), horizon=gcfg.get("horizon", cfg["horizon"]),
                models=tags(MODEL_RX, blob, gcfg.get("models", cfg["models"]), cfg.get("own_models", False)),
                regions=tags(REGION_RX, blob, cfg["region"]) if not gcfg.get("region") else gcfg["region"],
                variables=tags(VAR_RX, blob), thumb=thumb, cap=text_of(p.get("cap", ""))[:240],
                kw=public(about)[:600], variants=variants, live=True))
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
        thumb = {"src": img.group(1).lstrip("/")} if img and not img.group(1).startswith("data:") else None
        about = text_of(chunk)
        blob = " ".join([label, attr("data-group"), about[:1500]])
        items.append(dict(
            id=f"{href}#{pid}", page=href, page_title=cfg["title"], group=attr("data-group") or "Figures",
            label=label, sub="", url=f"{href}#{pid}", topic=cfg["topic"], horizon=cfg["horizon"],
            models=tags(MODEL_RX, blob, cfg["models"]), regions=tags(REGION_RX, blob, cfg["region"]),
            variables=tags(VAR_RX, blob), thumb=thumb, cap="", kw=public(about)[:600], variants=[], live=True))
    return items


def page_items() -> list:
    """Single-figure / app pages, described from the chrome's PRODUCTS table."""
    items = []
    for title, _blurb, grp in A.PRODUCTS:
        for href, label, what, _when in A.group_items(grp):
            cfg = PAGES.get(href)
            if not cfg or cfg["kind"] != "page":
                continue
            thumb = None
            if cfg.get("thumb"):
                thumb = {"src": cfg["thumb"]}
            elif cfg.get("thumb_manifest"):
                base, man, region, which = cfg["thumb_manifest"]
                try:
                    M = json.loads((REPO / base / man).read_text())
                    fr = M["regions"][region]["frames"]
                    thumb = {"src": f"{base}/{fr[-1 if which == 'last' else 0]['file']}"}
                except Exception:
                    thumb = None
            blob = " ".join([label, what])
            items.append(dict(
                id=href, page=href, page_title=label, group="", label=label, sub="", url=href, topic=cfg["topic"],
                horizon=cfg.get("horizon"), models=tags(MODEL_RX, blob, cfg["models"]), regions=tags(REGION_RX, blob, cfg["region"]),
                variables=tags(VAR_RX, blob, cfg.get("variable", [])), thumb=thumb, cap=what, kw=public(what), variants=[], live=False))
    return items


_TRACKED = None


def tracked() -> set:
    """Every file git tracks under assets/: the catalogue workflow checks out only the pages and the JSON (sparse), so a
    figure's existence is read from the index, not the disk."""
    global _TRACKED
    if _TRACKED is None:
        try:
            out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "assets"], capture_output=True, text=True, check=True).stdout
            _TRACKED = set(out.split("\0")) - {""}
        except (OSError, subprocess.CalledProcessError):
            _TRACKED = set()
    return _TRACKED


def resolve_thumb(spec):
    if isinstance(spec, tuple) and spec[0] == "manifest":
        _, base, man = spec
        return thumb_from_frame(f"sst_anim.html?base={base}&manifest={man}", REPO)
    for pat in spec.split("|"):
        if "*" in pat:
            hits = sorted({str(p.relative_to(REPO)) for p in REPO.glob(pat)} | {f for f in tracked() if fnmatch.fnmatchcase(f, pat)})
        else:
            hits = [pat] if (pat in tracked() or (REPO / pat).exists()) else []
        if hits:
            return {"src": hits[0]}
    return None


def build() -> dict:
    items = []
    for href, cfg in PAGES.items():
        if cfg["kind"] == "stage":
            items += stage_items(href, cfg)
        elif cfg["kind"] == "geps":
            items += stage_items(href, cfg)
            for it in items:
                if it["page"] == href:
                    it["live"] = False                  # GEPS/GEFS pages read the hash only on load
        elif cfg["kind"] == "rail":
            items += rail_items(href, cfg)
    items += page_items()
    for it in items:
        if not it["thumb"] and it["id"] in THUMBS:
            it["thumb"] = resolve_thumb(THUMBS[it["id"]])
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
    counts = {t: sum(1 for i in uniq if i["topic"] == t) for t, _, _ in TOPICS}
    # the menus' pages in menu order, for the finder's opening list (a site map in the palette)
    tid = {l: t for t, l, _ in TOPICS}
    pages = [dict(href=href, label=label, what=what, when=when, topic=tid[title])
             for title, _b, grp in A.PRODUCTS for href, label, what, when in A.group_items(grp)]
    for pg in pages:
        if PRIVATE.search(pg["label"] + " " + pg["what"]):
            raise SystemExit(f"private name in a menu page: {pg['href']}")
    return {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "topics": [dict(id=t, label=l, blurb=b, n=counts[t]) for t, l, b in TOPICS],
        "horizons": HORIZONS,
        "pages": pages,
        "items": uniq,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cat = build()
    n = len(cat["items"]); nv = sum(len(i["variants"]) for i in cat["items"]); nt = sum(1 for i in cat["items"] if i["thumb"])
    print(f"{n} products, {nv} option views, {nt} with thumbnails; by topic " +
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
