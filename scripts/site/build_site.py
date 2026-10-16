#!/usr/bin/env python3
"""Assemble the GitHub Pages artifact: the files the site serves, each dated by its last commit, and a dated sitemap.

Runs in pages.yml after scripts/site/status.py and writes the directory that is uploaded:
  - only the deployable part of the checkout: the pages, the section folders and assets (DIRS, ROOT_EXT, ROOT_FILES),
    never scripts/, workflows, notes or pipeline state (EXCLUDE);
  - every file's modification time set to the time of the last commit that changed it, from one `git log` pass over
    the first-parent history (a merge counts as the commit that changed what it brought in). An unchanged file then
    keeps its Last-Modified and ETag from one deploy to the next and a returning browser gets a 304; a fresh checkout
    would date every file to the moment of the deploy;
  - sitemap.xml with a lastmod for every indexable page (no noindex, canonical to itself): a product page by the
    newer of its own data stamp (status.json, the catalogue's freshness) and its last commit, any other page by its
    last commit. The pages listed in the committed sitemap.xml keep their order; an indexable page it misses is added.

--check then reads every page, script and stylesheet in the artifact for the same-origin paths it names (src, href,
srcset, url(), quoted asset paths) and fails if one exists in the checkout but not in the artifact: a gap in the
lists above. A path missing from both is a file another branch or a later build provides, and is only counted.

    python scripts/site/build_site.py --out _site --status status.json --check
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[2]
SITE = "https://scorvec.com"

DIRS = ("assets", "board", "data", "midterms", "qbo", "skewt", "studies", "topics")
ROOT_EXT = (".html", ".webp", ".jpg", ".png", ".svg", ".ico")
ROOT_FILES = ("CNAME", "robots.txt", "sitemap.xml")
EXCLUDE = {
    "assets/verify/aifs_scores_v2.json",  # 12 MB of station scores: the input of scripts/verify, read by no page
}
GENERATED = ("status.json",)  # written by this deploy (scripts/site/status.py), never committed


def deployable(path: str) -> bool:
    if path in EXCLUDE or path.endswith(".md"):
        return False
    top, _, rest = path.partition("/")
    if rest:
        return top in DIRS
    return path in ROOT_FILES or Path(path).suffix in ROOT_EXT


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def commit_times() -> dict[str, int]:
    """Path -> epoch seconds of the newest first-parent commit that changed it (merges diffed against their first
    parent), in one pass."""
    if git("rev-parse", "--is-shallow-repository").strip() == "true":
        print("::warning::shallow checkout: every file is dated by the oldest fetched commit", file=sys.stderr)
    last: dict[str, int] = {}
    t = 0
    for ln in git("log", "--first-parent", "-m", "--no-renames", "--format=@%ct", "--name-only", "HEAD").splitlines():
        if ln.startswith("@") and ln[1:].isdigit():
            t = int(ln[1:])
        elif ln and ln not in last:
            last[ln] = t
    return last


def assemble(out: Path, times: dict[str, int]) -> tuple[int, int]:
    if out.exists():
        shutil.rmtree(out)
    now = time.time()
    n = size = 0
    files = [p for p in git("ls-files", "-z").split("\0") if p and deployable(p)]
    for rel in files + [g for g in GENERATED if (REPO / g).is_file()]:
        src, dst = REPO / rel, out / rel
        if not src.is_file():  # a tracked file this checkout does not have (sparse) or a submodule
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        t = times.get(rel, now)
        os.utime(dst, (t, t))
        n += 1
        size += dst.stat().st_size
    return n, size


# ── sitemap ──────────────────────────────────────────────────────────────────────────────────────────────────────
def page_url(rel: str) -> str:
    return f"{SITE}/{rel[: -len('index.html')] if rel.endswith('index.html') else rel}"


def indexable(out: Path) -> dict[str, str]:
    """URL -> artifact path of every page a search engine should list: no noindex, and a canonical (if any) to itself."""
    pages = {}
    for p in sorted(out.rglob("*.html")):
        rel = p.relative_to(out).as_posix()
        h = p.read_text(errors="replace")
        head = h[: h.find("</head>")] if "</head>" in h else h[:20000]
        robots = re.search(r"<meta\s+name=[\"']robots[\"']\s+content=[\"']([^\"']*)", head, re.I)
        if robots and "noindex" in robots.group(1).lower():
            continue
        canon = re.search(r"<link\s+rel=[\"']canonical[\"']\s+href=[\"']([^\"']+)", head, re.I)
        if canon and canon.group(1) != page_url(rel):
            continue
        pages[page_url(rel)] = rel
    return pages


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC).timestamp()


def sitemap(out: Path, times: dict[str, int], status: dict) -> int:
    listed = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", (REPO / "sitemap.xml").read_text())
    pages = indexable(out)
    for u in listed:
        if u not in pages:
            print(
                f"::warning::sitemap.xml lists {u}, which is not an indexable page of the site; left out",
                file=sys.stderr,
            )
    added = sorted(u for u in pages if u not in listed)
    for u in added:
        print(f"  sitemap: added {u} (indexable, not in sitemap.xml)")
    products = status.get("products", {})
    rows = []
    for u in [u for u in listed if u in pages] + added:
        t = times.get(pages[u], time.time())
        prod = products.get(u[len(SITE) :])
        if prod and prod.get("updated"):
            t = max(t, stamp(prod["updated"]))
        rows.append(f"  <url>\n    <loc>{escape(u)}</loc>\n    <lastmod>{iso(t)}</lastmod>\n  </url>\n")
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    (out / "sitemap.xml").write_text(xml + "".join(rows) + "</urlset>\n")
    return len(rows)


# ── the reference check ──────────────────────────────────────────────────────────────────────────────────────────
_ATTR = re.compile(r"""\s(?:src|href|data-src|poster)\s*=\s*(["'])(.*?)\1""", re.S | re.I)
_SRCSET = re.compile(r"""\s(?:srcset|imagesrcset)\s*=\s*(["'])(.*?)\1""", re.S | re.I)
_URL = re.compile(r"""url\(\s*(["']?)([^"')]+)\1\s*\)""")
# a complete quoted path to a site file inside a script: no concatenation, no template
_LIT = re.compile(
    r"""(["'`])((?:/|\.\./)?(?:""" + "|".join(DIRS) + r""")/[\w./-]+\.[A-Za-z0-9]{2,5})(?:\?[^"'`]*)?\1"""
)


def refs(path: Path, text: str) -> set[str]:
    out = set()
    if path.suffix == ".html":
        out |= {m.group(2) for m in _ATTR.finditer(text)}
        out |= {c.strip().split()[0] for m in _SRCSET.finditer(text) for c in m.group(2).split(",") if c.strip()}
    out |= {m.group(2) for m in _URL.finditer(text)}
    out |= {m.group(2) for m in _LIT.finditer(text)}
    return out


def resolve(ref: str, page: str) -> str | None:
    """The artifact path a same-origin reference from `page` points at, or None when it is not a plain file path."""
    ref = ref.strip()
    if ref.startswith(SITE + "/"):
        ref = ref[len(SITE) :]
    if not ref or re.match(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", ref, re.I) or re.search(r"[{}$<>+\s\\]|' |\" ", ref):
        return None
    ref = re.split(r"[?#]", ref, maxsplit=1)[0]
    if not ref:
        return None
    base = "" if ref.startswith("/") else page.rsplit("/", 1)[0] + "/" if "/" in page else ""
    parts = []
    for seg in (base + ref.lstrip("/")).split("/"):
        if seg == "..":
            if parts:
                parts.pop()
        elif seg not in ("", "."):
            parts.append(seg)
    return "/".join(parts) + ("/" if ref.endswith("/") else "")


def present(root: Path, rel: str) -> bool:
    p = root / rel
    return p.is_file() or (p / "index.html").is_file() or Path(str(p) + ".html").is_file() or rel in ("", "/")


def check(out: Path) -> int:
    gaps, elsewhere, seen = {}, set(), 0
    for p in sorted(out.rglob("*")):
        if p.suffix not in (".html", ".js", ".css") or not p.is_file():
            continue
        page = p.relative_to(out).as_posix()
        for ref in refs(p, p.read_text(errors="replace")):
            rel = resolve(ref, "" if p.suffix == ".js" else page)  # a script's paths are the page's: the site root
            if rel is None:
                continue
            seen += 1
            if present(out, rel):
                continue
            if present(REPO, rel):
                gaps.setdefault(rel, set()).add(page)
            else:
                elsewhere.add(rel)
    for rel, by in sorted(gaps.items()):
        print(
            f"::error::{rel} is referenced by {', '.join(sorted(by)[:3])} but left out of the artifact", file=sys.stderr
        )
    print(
        f"check: {seen} same-origin references, {len(gaps)} left out of the artifact, {len(elsewhere)} not in this checkout"
    )
    return 1 if gaps else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site")
    ap.add_argument("--status", default="status.json", help="product freshness from scripts/site/status.py")
    ap.add_argument("--check", action="store_true", help="fail if a page names a checkout file the artifact lacks")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    times = commit_times()
    n, size = assemble(out, times)
    try:
        status = json.loads(Path(a.status).read_text())
    except (OSError, ValueError):
        status = {}
        print("::warning::no status.json: product pages are dated by their last commit only", file=sys.stderr)
    urls = sitemap(out, times, status)
    print(f"artifact: {n} files, {size / 1e6:.0f} MB -> {out}; sitemap: {urls} pages")
    return check(out) if a.check else 0


if __name__ == "__main__":
    sys.exit(main())
