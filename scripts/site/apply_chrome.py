#!/usr/bin/env python3
"""Stamp the shared site chrome (header, section tabs, footer) into every page.

One source of truth for the navigation: PRODUCTS below feeds the header's
Products menu on every page, and the same groups are what the homepage lists.
Re-run after adding a page or a product; the stamp is idempotent (it replaces
whatever sits between the <!-- sh:start --> / <!-- sf:start --> markers).

    python scripts/site/apply_chrome.py            # stamp every page in PAGES
    python scripts/site/apply_chrome.py --check    # exit 1 if any page is out of date

The look lives in assets/site.css; the menu behaviour in assets/site.js.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ── the navigation, as data ──────────────────────────────────────────────────
PRODUCTS = [
    ("Tropics and El Niño", [
        ("/sst.html", "El Niño monitor"),
        ("/enso-subsurface.html", "Subsurface temperature"),
        ("/enso-forecasts.html", "ENSO forecasts"),
        ("/enso-atmosphere.html", "Atmospheric response"),
        ("/mjo.html", "MJO forecast"),
        ("/subseasonal.html", "Subseasonal outlook"),
        ("/qbo/", "QBO tracker"),
    ]),
    ("Convection and observations", [
        ("/ecape.html", "Entraining CAPE"),
        ("/skewt/", "Sounding explorer"),
        ("/asos5.html", "Five-minute airport observations"),
        ("/columbia/", "Columbia River basin precipitation"),
    ]),
    ("Seasonal", [
        ("/seas5.html", "ECMWF SEAS5 outlook"),
        ("/sfs.html", "NOAA SFS outlook"),
    ]),
    ("Model verification", [
        ("/aifs-verify.html", "AIFS single versus ensemble control"),
        ("/spectra.html", "HRRR and RRFS kinetic-energy spectra"),
    ]),
]
PRIMARY = [("/research.html", "Research"), ("/resume.html", "Resume")]

# Tab rows for page families. Keys are referenced from PAGES.
TABS = {
    "enso": [
        ("/sst.html", "Overview"),
        ("/enso-subsurface.html", "Subsurface"),
        ("/enso-forecasts.html", "Forecasts"),
        ("/seas5.html", "SEAS5"),
        ("/enso-atmosphere.html", "Atmospheric response"),
    ],
    "skewt": [
        ("/skewt/", "Explorer"),
        ("/skewt/methodology.html", "How it works"),
        ("/skewt/gaps.html", "US gap report"),
    ],
}

# ── pages ────────────────────────────────────────────────────────────────────
# mode:
#   nav          replace the first <nav>…</nav>
#   site-header  replace the first <header class="site-header">…</header> (nav + sub-nav inside)
#   after-body   no nav on the page: insert right after <body…>
#   before       insert before the literal `anchor`; optionally delete `drop` (a regex) first
# fixes: literal (old, new) substrings — the top padding that used to clear a fixed nav.
PAGES = [
    dict(path="index.html", mode="nav", skin="overlay", footer=False),
    dict(path="sst.html", mode="site-header", tabs="enso", fixes=[
        ("padding: 6.4rem 2.2rem 2.5rem;", "padding: 2rem 2.2rem 2.5rem;"),
        ("main { padding: 6rem 1rem 2rem; max-width: 100%; }", "main { padding: 1.5rem 1rem 2rem; max-width: 100%; }"),
        ("main { padding: 9.5rem 1rem 3rem; }", "main { padding: 1.5rem 1rem 3rem; }"),
        ("scroll-margin-top: 5.5rem;", "scroll-margin-top: 4.5rem;"),
    ]),
    dict(path="enso-subsurface.html", mode="site-header", tabs="enso", fixes=[
        ("padding: 6.4rem 2.2rem 2.5rem;", "padding: 2rem 2.2rem 2.5rem;"),
        ("main { padding: 6rem 1rem 2rem; max-width: 100%; }", "main { padding: 1.5rem 1rem 2rem; max-width: 100%; }"),
        ("main { padding: 9.5rem 1rem 3rem; }", "main { padding: 1.5rem 1rem 3rem; }"),
    ]),
    dict(path="enso-forecasts.html", mode="site-header", tabs="enso", fixes=[
        ("padding: 6.4rem 2.2rem 2.5rem;", "padding: 2rem 2.2rem 2.5rem;"),
        ("main { padding: 6rem 1rem 2rem; max-width: 100%; }", "main { padding: 1.5rem 1rem 2rem; max-width: 100%; }"),
        ("main { padding: 9.5rem 1rem 3rem; }", "main { padding: 1.5rem 1rem 3rem; }"),
    ]),
    dict(path="enso-atmosphere.html", mode="site-header", tabs="enso", fixes=[
        ("padding: 6.4rem 2.2rem 2.5rem;", "padding: 2rem 2.2rem 2.5rem;"),
        ("main { padding: 6rem 1rem 2rem; max-width: 100%; }", "main { padding: 1.5rem 1rem 2rem; max-width: 100%; }"),
        ("main { padding: 9.5rem 1rem 3rem; }", "main { padding: 1.5rem 1rem 3rem; }"),
    ]),
    dict(path="seas5.html", mode="after-body", tabs="enso"),
    dict(path="subseasonal.html", mode="site-header", fixes=[
        ("padding: 6.2rem 2.5rem 2rem; width: 100%;", "padding: 2rem 2.5rem 2rem; width: 100%;"),
        ("main { padding: 9rem 1.5rem 3.5rem; }", "main { padding: 1.5rem 1.5rem 3.5rem; }"),
        ("main { padding: 9.5rem 1rem 3rem; }", "main { padding: 1.5rem 1rem 3rem; }"),
    ]),
    dict(path="mjo.html", mode="nav"),
    dict(path="ecape.html", mode="nav", fixes=[
        ("padding: 7.5rem 2.5rem 5rem;", "padding: 2.5rem 2.5rem 5rem;"),
        ("main { padding: 6.5rem 1.2rem 3rem; }", "main { padding: 1.5rem 1.2rem 3rem; }"),
    ]),
    dict(path="aifs-verify.html", mode="nav", fixes=[
        ("padding: 7.2rem 2.5rem 5rem;", "padding: 2.5rem 2.5rem 5rem;"),
        ("main { padding: 6.3rem 1rem 3rem; max-width: 100%; }", "main { padding: 1.5rem 1rem 3rem; max-width: 100%; }"),
    ]),
    dict(path="spectra.html", mode="nav", fixes=[
        ("padding: 7.5rem 2.5rem 5rem;", "padding: 2.5rem 2.5rem 5rem;"),
        ("main { padding: 6.5rem 1.2rem 3rem; }", "main { padding: 1.5rem 1.2rem 3rem; }"),
    ]),
    dict(path="asos5.html", mode="site-header", fixes=[
        ("padding: 6.2rem 2rem 3rem;", "padding: 2rem 2rem 3rem;"),
        ("main { padding: 5.4rem 0.8rem 2rem; }", "main { padding: 1.2rem 0.8rem 2rem; }"),
    ]),
    dict(path="sfs.html", mode="nav"),
    dict(path="research.html", mode="nav", fixes=[
        ("padding: 7.5rem 2rem 5rem;", "padding: 2.5rem 2rem 5rem;"),
        ("main { padding: 6rem 1.5rem 4rem; }", "main { padding: 1.5rem 1.5rem 4rem; }"),
        ('class="scholar-link"', 'class="button button--secondary"'),   # the standard outlined button
    ]),
    dict(path="resume.html", mode="nav", fixes=[
        ("    min-height: 100vh;\n    padding-top: 57px;", "    min-height: calc(100vh - 57px);"),
        ("    position: sticky;\n    top: 57px;", "    position: sticky;\n    top: 64px;"),
        ("    position: sticky;\n    top: 56px;", "    position: sticky;\n    top: 64px;"),
    ]),
    dict(path="stats.html", mode="nav", fixes=[
        ("padding: 7.5rem 2.5rem 5rem;", "padding: 2.5rem 2.5rem 5rem;"),
        ("main { padding: 6.5rem 1.5rem 3.5rem; }", "main { padding: 1.5rem 1.5rem 3.5rem; }"),
        ("main { padding: 5.5rem 1rem 3rem; }", "main { padding: 1rem 1rem 3rem; }"),
        ('style="margin:4.5rem 0 0.5rem;"', 'style="margin:0.5rem 0 0.5rem;"'),
    ]),
    dict(path="skewt/index.html", mode="before", anchor="<nav>", tabs="skewt", footer=False,
         drop=[r'\s*<a href="\.\./index\.html" class="nav-name">Shawn Corvec</a>',
               r'\s*<a href="\.\./index\.html" class="back">← HOME</a>']),
    dict(path="skewt/methodology.html", mode="nav", tabs="skewt"),
    dict(path="skewt/gaps.html", mode="nav", tabs="skewt"),
    dict(path="qbo/index.html", mode="nav"),
    dict(path="columbia/index.html", mode="before", anchor='<header class="top">',
         drop=r'<p class="pagelinks"><a href="\.\./index\.html">&larr; scorvec\.com</a>\s*&nbsp;·&nbsp; '),
    # the 404 body is a centring flexbox: stack it so the header spans the top and the message centres below
    dict(path="404.html", mode="after-body", footer=False, fixes=[
        ("display:flex;\nmin-height:100vh;align-items:center;justify-content:center;margin:0;",
         "display:flex;flex-direction:column;\nmin-height:100vh;margin:0;"),
        ("p{color:#8b8ba3}a{color:#64d2ff}", "p{color:#8b8ba3}a{color:#64d2ff}body>div{margin:auto;padding:2rem}"),
    ]),
]

HEAD_SNIPPET = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400'
    '&family=Schibsted+Grotesk:wght@400;500&display=swap" rel="stylesheet">\n'
    '<link rel="stylesheet" href="/assets/site.css">\n'
    '<script src="/assets/site.js" defer></script>\n'
)


def _current(href: str, page: str) -> str:
    return ' aria-current="page"' if href == page else ""


def header_html(page: str, skin: str) -> str:
    cls = "sh" + (f" sh--{skin}" if skin else "")
    in_products = any(h == page for _, items in PRODUCTS for h, _ in items)
    out = [f'<!-- sh:start -->\n<header class="{cls}" id="site-header">\n  <div class="sh-in">',
           '    <a class="sh-brand" href="/">Shawn Corvec</a>',
           '    <button class="sh-toggle" type="button" aria-expanded="false" aria-controls="sh-menu">Menu</button>',
           '    <nav class="sh-nav" id="sh-menu" aria-label="Site">\n      <ul class="sh-list">',
           '        <li class="sh-item sh-has-menu">',
           f'          <button class="sh-link sh-menubtn" type="button" aria-expanded="false" aria-controls="sh-products"'
           f'{" aria-current=page" if in_products else ""}>Products</button>',
           '          <div class="sh-menu" id="sh-products">']
    for title, items in PRODUCTS:
        out.append(f'            <div>\n              <h3>{title}</h3>\n              <ul>')
        for href, label in items:
            out.append(f'                <li><a href="{href}"{_current(href, page)}>{label}</a></li>')
        out.append('              </ul>\n            </div>')
    out.append('            <p class="sh-all">Every product, with what it shows and how often it updates, is listed on the <a href="/#products">home page</a>.</p>')
    out.append('          </div>\n        </li>')
    for href, label in PRIMARY:
        out.append(f'        <li class="sh-item"><a class="sh-link" href="{href}"{_current(href, page)}>{label}</a></li>')
    out.append('      </ul>\n    </nav>\n  </div>\n</header>')
    return "\n".join(out).replace(' aria-current=page', ' aria-current="page"')


def tabs_html(key: str, page: str, dark: bool) -> str:
    cls = "st" + (" st--dark" if dark else "")
    links = "\n".join(f'    <a href="{h}"{_current(h, page)}>{l}</a>' for h, l in TABS[key])
    return f'<nav class="{cls}" aria-label="Section">\n  <div class="st-in">\n{links}\n  </div>\n</nav>'


def footer_html(dark: bool) -> str:
    cls = "sf" + (" sf--dark" if dark else "")
    return (
        f'<!-- sf:start -->\n<footer class="{cls}">\n  <div class="sf-in">\n'
        '    <ul>\n      <li><a href="/">Home</a></li>\n      <li><a href="/#products">Products</a></li>\n'
        '      <li><a href="/research.html">Research</a></li>\n      <li><a href="/resume.html">Resume</a></li>\n'
        '      <li><a href="https://github.com/scorvec" rel="noopener">GitHub</a></li>\n'
        '      <li><a href="https://www.linkedin.com/in/shawn-corvec-35895b231/" rel="noopener">LinkedIn</a></li>\n'
        '      <li><a href="https://scholar.google.com/citations?user=EYLRCJIAAAAJ&amp;hl=en" rel="noopener">Google Scholar</a></li>\n'
        '      <li><a href="/stats.html">Visitor stats</a></li>\n    </ul>\n'
        '    <p>Shawn Corvec. Built from open data; sources are credited on each page.</p>\n'
        '  </div>\n</footer>\n<!-- sf:end -->'
    )


def is_dark(html: str) -> bool:
    """A page whose --bg is darker than mid-grey gets the dark header skin."""
    m = re.search(r"--bg:\s*#([0-9a-fA-F]{6})\b", html)
    if not m:
        m = re.search(r"body\s*\{[^}]*background:\s*#([0-9a-fA-F]{6})", html)
    if not m:
        return False
    r, g, b = (int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 100


def page_key(path: str) -> str:
    return "/" + path.replace("index.html", "")


def stamp(cfg: dict) -> tuple[str, str]:
    p = REPO / cfg["path"]
    html = p.read_text()
    orig = html
    page = page_key(cfg["path"])
    dark = cfg.get("skin") == "dark" or (is_dark(html) and cfg.get("skin") != "overlay")
    skin = cfg.get("skin") or ("dark" if dark else "")

    block = header_html(page, skin)
    if cfg.get("tabs"):
        block += "\n" + tabs_html(cfg["tabs"], page, dark)
    block += "\n<!-- sh:end -->"

    for pat in ([cfg["drop"]] if isinstance(cfg.get("drop"), str) else cfg.get("drop", [])):
        html = re.sub(pat, "", html, count=1)
    if "<!-- sh:start -->" in html:                       # re-stamp
        html = re.sub(r"<!-- sh:start -->.*?<!-- sh:end -->", lambda _: block, html, count=1, flags=re.S)
    else:
        mode = cfg["mode"]
        if mode == "nav":
            html, n = re.subn(r"<nav>.*?</nav>", lambda _: block, html, count=1, flags=re.S)
        elif mode == "site-header":
            html, n = re.subn(r'<header class="site-header">.*?</header>', lambda _: block, html, count=1, flags=re.S)
        elif mode == "after-body":
            html, n = re.subn(r"(<body[^>]*>)", lambda m: m.group(1) + "\n" + block, html, count=1)
        elif mode == "before":
            n = html.count(cfg["anchor"])
            html = html.replace(cfg["anchor"], block + "\n" + cfg["anchor"], 1)
        else:
            raise SystemExit(f"{cfg['path']}: unknown mode {mode}")
        if n != 1:
            raise SystemExit(f"{cfg['path']}: anchor for mode {mode} not found")
    for old, new in cfg.get("fixes", []):
        if old not in html and new not in html:
            print(f"  warning: {cfg['path']}: fix not found: {old[:50]!r}", file=sys.stderr)
        html = html.replace(old, new)

    if "/assets/site.css" not in html:
        snippet = HEAD_SNIPPET
        if "family=Newsreader" in html:                    # the page already loads the fonts
            snippet = "\n".join(l for l in snippet.splitlines() if "fonts.g" not in l) + "\n"
        if "</head>" in html:
            html = html.replace("</head>", snippet + "</head>", 1)
        else:                                              # bare HTML5 document without <head> tags
            html = html.replace("<!-- sh:start -->", snippet + "<!-- sh:start -->", 1)

    if cfg.get("footer", True):
        foot = footer_html(dark)
        if "<!-- sf:start -->" in html:
            html = re.sub(r"<!-- sf:start -->.*?<!-- sf:end -->", lambda _: foot, html, count=1, flags=re.S)
        else:
            html = html.replace("</body>", foot + "\n</body>", 1)
    return orig, html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report pages that would change; write nothing")
    args = ap.parse_args()
    changed = 0
    for cfg in PAGES:
        orig, new = stamp(cfg)
        if orig != new:
            changed += 1
            if args.check:
                print(f"out of date: {cfg['path']}")
            else:
                (REPO / cfg["path"]).write_text(new)
                print(f"stamped {cfg['path']}")
    if args.check:
        return 1 if changed else 0
    print(f"{changed} page(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
