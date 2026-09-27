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
# BY TOPIC since 2026-09-27 (user: "another site navigation redesign as it is getting quite large ... reorganize
# things and change the interface for finding certain plots"; chose topic menus + a ⌘K finder + a catalogue).
# Renamed the same day (user: "separate my personal info and resume from the rest of the site ... 'weather' is not
# very descriptive, short/medium range, subseasonal/seasonal forecast groups would make more sense"): the product
# menus hold products only; the person (about, resume, publications, profiles) lives in PERSONAL, apart.
# A page lives under its topic whatever model it comes from. Items are (href, label, what, when): the menus use
# the first two, the homepage rows (between <!-- hp:start --> and <!-- hp:end --> in index.html) all four.
# A group's items may be a flat list or a list of (subtitle, items) columns. Group titles are also the catalogue's
# topic labels (scripts/site/build_catalog.py TOPICS) - rename both together.
PRODUCTS = [
    ("Short and medium range", "Days 1 to 15: storms, hazards and the temperature forecast", [
        ("/snowbands.html", "Snow-band diagnostics",
         "Where HRRR, RRFS and RDPS set up mesoscale snow bands: 700 hPa frontogenesis and deformation, EPV* and slantwise instability, lift through the dendritic growth zone, the model's own bands and where the ingredients overlap. November to mid-April, with case studies against the radar.",
         "Every run in season"),
        ("/ar.html", "Atmospheric rivers",
         "Integrated vapour transport from the 51-member AIFS ensemble: probability of AR conditions at each 12-hourly step, a West Coast landfall tool and Ralph-scale category odds at named locations.",
         "Twice daily"),
        ("/ecape.html", "Entraining CAPE",
         "Gridded ECAPE over the continental US from the 3 km HRRR, hourly to 18 hours and 3-hourly to 48.",
         "Four times daily"),
        # /cities/ (city temperature forecasts) PAUSED 2026-09-27 (user: "Turn off the city forecast and verification
        # page for now"); restore this row and the verification row from git history.
    ], "Short/medium range"),
    ("Subseasonal and seasonal", "From weeks two to five out to the coming seasons", [
        ("Weeks 2–5", [
            ("/subseasonal.html", "GEPS extended ensemble",
             "Environment Canada's extended ensemble to day 35 for the Americas, the tropics and the polar vortex: weekly anomalies against its own reforecast, terciles, teleconnections, regimes and change since the previous run.",
             "Mondays and Thursdays"),
            ("/gefs.html", "GEFS extended ensemble",
             "NOAA's GEFS extended ensemble to day 35 in the same layout: weekly anomalies against its own reforecast, the change since a week earlier, and a drift-corrected polar vortex.",
             "Daily"),
        ]),
        ("Seasons", [
            ("/seasonal.html", "Eight C3S seasonal models",
             "Eight seasonal systems from seven centres: anomaly maps and tercile probabilities for every model and their mean, plus the member-level ECMWF SEAS5 outlook: indices, teleconnections, the stratosphere and impacts.",
             "Monthly"),
            ("/sfs.html", "NOAA SFS seasonal",
             "Maps from NOAA's new Seasonal Forecast System, 31 members, for the months ahead.",
             "Monthly"),
            ("/enso-forecasts.html", "ENSO forecasts",
             "Every member of seven centres' seasonal models for Niño-3.4, with percentile fans and a record of how past forecasts did.",
             "Monthly"),
        ]),
    ], "Subseasonal/seasonal"),
    ("Climate drivers", "The modes: ENSO, the MJO and the QBO, as they stand now", [
        ("/enso.html", "El Niño monitor",
         "Daily ONI and RONI, Niño-region sea surface temperatures, the subsurface, winds and convection, with SST anomaly maps and animation.",
         "Daily"),
        ("/mjo.html", "MJO forecast",
         "Real-time multivariate MJO index from the ECMWF AIFS ensemble, with the observed phase-space track, an ENSO-removed version, and what each phase has meant for temperature and rain.",
         "Twice daily"),
        ("/qbo/", "QBO tracker",
         "Equatorial stratospheric winds at 10 to 100 hPa from radiosondes, 1950 to the present.",
         "Weekly"),
    ]),
    # (user, same day: "The jets, aam and stratosphere stuff should not really be under 'climate drivers' either")
    # A "#" item under its own page is a sub-item: indented in the menu, left off the homepage rows.
    ("Atmospheric circulation", "The stratosphere and the troposphere's jets, waves and overturning cells", [
        ("Stratosphere", [
            ("/stratosphere.html", "Stratosphere and polar vortex",
             "Will the vortex weaken? AIFS, GEPS and GEFS 60°N winds, heat flux and wave driving, potential vorticity, the Brewer–Dobson circulation, and every sudden warming since 1980 with what followed.",
             "Twice daily"),
            ("/stratosphere.html#heatflux", "Eddy heat flux",
             "100 hPa poleward heat flux, the wave activity going up into the vortex, from AIFS-ENS, GEPS and GEFS.",
             "Twice daily"),
            ("/stratosphere.html#pv", "Potential vorticity",
             "GEOS FP Ertel PV on the 475 and 850 K surfaces.",
             "Daily"),
            ("/stratosphere.html#bdc", "Brewer–Dobson circulation",
             "Tropical upwelling and the residual circulation from GEOS FP and MERRA-2.",
             "Daily"),
        ]),
        ("Troposphere", [
            ("/circulation.html", "Jets, Walker and Hadley cells",
             "Wave activity flux, the dynamic tropopause, angular momentum and mountain torques, and the Hadley and Walker cells from the AIFS ensemble.",
             "Daily"),
            ("/circulation.html#waf", "Wave activity flux",
             "Takaya–Nakamura wave activity flux at 250 hPa, the ensemble's quasi-stationary wave trains.",
             "Daily"),
            ("/circulation.html#aam", "Angular momentum and torques",
             "Global atmospheric angular momentum and the mountain and friction torques that change it.",
             "Daily"),
            ("/circulation.html#dt", "Dynamic tropopause",
             "The 2-PVU surface and PV on the 330 and 350 K isentropes from the AIFS ensemble.",
             "Daily"),
        ]),
    ], "Circulation"),
    ("Studies and verification", "What the record and the models say, and how the forecasts score", [
        ("Studies", [
            ("/enso.html#imp_reg", "ENSO impacts on the Americas",
             "Thousands of El Niño and La Niña events in 16 CMIP6 models against the observed record: regressions, composites, super and east-based El Niños, 500 hPa wave trains, and why the PDO mostly follows the atmosphere rather than driving it. Significant results only.",
             "Static"),
            ("/stratosphere.html#shwinters", "Stratosphere history",
             "Every northern winter and sudden warming since 1980 in MERRA-2, with 15,000 more from nine CMIP6 models: the dripping-paint composites and what usually follows at the surface.",
             "Static"),
            ("/topics/", "Explainers",
             "How to read the products: the equations and the physics behind each diagnostic.",
             "As written"),
        ]),
        ("Verification", [
            # /cities/verify.html paused 2026-09-27 with the city forecasts (see above)
            ("/aifs-verify.html", "AIFS single versus member 0",
             "The two ECMWF AIFS configurations compared as deterministic models against radiosondes and ERA5, with the ensemble member spectrally matched to the single model and the ensemble mean for reference.",
             "Every 00Z and 12Z run"),
        ]),
    ], "Studies/verification"),
    ("Tools", "Explore the data yourself", [
        ("/catalog.html", "Every plot on the site",
         "One searchable sheet of every figure and loop, filterable by topic, model, forecast range and region, each linked to its exact view.",
         "Rebuilt with the pages"),
        ("/skewt/", "Sounding explorer",
         "Real-time and archived radiosondes worldwide, drawn and analysed in the browser, with launch history and record rings.",
         "Hourly mirror"),
        ("/asos5.html", "Five-minute airport observations",
         "Temperature at major US airports every five minutes from ASOS, with running daily highs and lows.",
         "Live"),
        ("/climate.html", "US climate trends",
         "Every US county and calendar month since 1895: temperature, precipitation and degree-day trends per decade with significance, the 1991–2020 normal and what normal is now.",
         "Monthly"),
    ]),
]

# The person, kept apart from the products (user, 2026-09-27): one "About" item at the far right of the header,
# behind a divider, and its own footer group. External profiles open in the same tab like every other link.
PROFILES = [
    ("https://github.com/scorvec", "GitHub"),
    ("https://scholar.google.com/citations?user=EYLRCJIAAAAJ&amp;hl=en", "Google Scholar"),
    ("https://www.linkedin.com/in/shawn-corvec-35895b231/", "LinkedIn"),
]
PERSONAL = [("/about.html", "About me"), ("/resume.html", "Resume"), ("/research.html", "Publications")] + PROFILES


def group_items(items):
    """Flatten a group's items whether it is a flat list or (subtitle, items) columns."""
    if items and isinstance(items[0][1], list):
        return [it for _, sub in items for it in sub]
    return list(items)

# Plain header links between the product menus and the special button (none since the About menu took Resume).
PRIMARY: list = []
# A highlighted button, apart from the menus, on every page (2026-09-23, user: "put this on my site as a special button").
SPECIAL = ("/midterms/", "2026 Midterms")

# Tab rows for page families. Keys are referenced from PAGES.
TABS = {
    "enso": [
        ("/enso.html", "Overview"),
        ("/enso-forecasts.html", "Forecasts"),
        ("/seasonal.html", "Seasonal models"),
        ("/circulation.html", "Atmospheric response"),
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
    dict(path="enso-forecasts.html", mode="site-header", tabs="enso", fixes=[
        ("padding: 6.4rem 2.2rem 2.5rem;", "padding: 2rem 2.2rem 2.5rem;"),
        ("main { padding: 6rem 1rem 2rem; max-width: 100%; }", "main { padding: 1.5rem 1rem 2rem; max-width: 100%; }"),
        ("main { padding: 9.5rem 1rem 3rem; }", "main { padding: 1.5rem 1rem 3rem; }"),
    ]),
    dict(path="seasonal.html", mode="after-body", tabs="enso"),   # the merged C3S page (was seas5.html)
    dict(path="enso.html", mode="after-body", tabs="enso"),
    dict(path="circulation.html", mode="after-body", tabs="enso"),
    dict(path="stratosphere.html", mode="after-body", tabs="enso"),   # split out of circulation.html 2026-09-26
    dict(path="subseasonal.html", mode="site-header", fixes=[]),   # page restyled 2026-09-07 on /assets/outlook.css; no padding patches needed
    dict(path="gefs.html", mode="site-header"),                    # GEFS extended page (scripts/gefs, gefs.yml), GEPS layout
    dict(path="subseasonal-method.html", mode="site-header"),      # methodology note split out of subseasonal.html; linked from there only, not in PRODUCTS
    dict(path="mjo.html", mode="nav"),
    dict(path="ecape.html", mode="nav", fixes=[
        ("padding: 7.5rem 2.5rem 5rem;", "padding: 2.5rem 2.5rem 5rem;"),
        ("main { padding: 6.5rem 1.2rem 3rem; }", "main { padding: 1.5rem 1.2rem 3rem; }"),
    ]),
    dict(path="aifs-verify.html", mode="nav", fixes=[
        ("padding: 7.2rem 2.5rem 5rem;", "padding: 2.5rem 2.5rem 5rem;"),
        ("main { padding: 6.3rem 1rem 3rem; max-width: 100%; }", "main { padding: 1.5rem 1rem 3rem; max-width: 100%; }"),
    ]),
    dict(path="asos5.html", mode="site-header", fixes=[
        ("padding: 6.2rem 2rem 3rem;", "padding: 2rem 2rem 3rem;"),
        ("main { padding: 5.4rem 0.8rem 2rem; }", "main { padding: 1.2rem 0.8rem 2rem; }"),
    ]),
    dict(path="sfs.html", mode="nav"),
    dict(path="ar.html", mode="after-body"),
    dict(path="snowbands.html", mode="after-body"),                # snow-band diagnostics (scripts/snowband, snowband.yml), stage viewer
    dict(path="catalog.html", mode="after-body"),
    dict(path="about.html", mode="after-body"),                    # the person: bio, resume, publications, profiles                  # every plot on the site (scripts/site/build_catalog.py)
    dict(path="climate.html", mode="after-body"),
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
    dict(path="midterms/index.html", mode="after-body", footer=True),   # 2026 midterm forecast (published from ~/midterms, weekly)
    dict(path="midterms/about.html", mode="after-body", footer=True),   # how the midterm forecast works + sources
    dict(path="midterms/polls.html", mode="after-body", footer=True),   # every poll in the midterm forecast and its weight
    dict(path="cities/index.html", mode="nav"),
    dict(path="cities/verify.html", mode="nav"),
    # the 404 body is a centring flexbox: stack it so the header spans the top and the message centres below
    dict(path="404.html", mode="after-body", footer=False, fixes=[
        ("display:flex;\nmin-height:100vh;align-items:center;justify-content:center;margin:0;",
         "display:flex;flex-direction:column;\nmin-height:100vh;margin:0;"),
        ("p{color:#8b8ba3}a{color:#64d2ff}", "p{color:#8b8ba3}a{color:#64d2ff}body>div{margin:auto;padding:2rem}"),
    ]),
]
# topic explainers (written by scripts/site/topics.py, 2026-09-26): every page in topics/ gets the header and footer
PAGES += [dict(path=f"topics/{p.name}", mode="after-body", footer=True) for p in sorted((REPO / "topics").glob("*.html"))]

HEAD_SNIPPET = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&family=Source+Sans+3:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">\n'
    '<link rel="stylesheet" href="/assets/site.css">\n'
    '<script src="/assets/site.js" defer></script>\n'
    '<script src="/assets/rail.js" defer></script>\n'
)


def _current(href: str, page: str) -> str:
    return ' aria-current="page"' if href == page else ""


def header_html(page: str, skin: str) -> str:
    cls = "sh" + (f" sh--{skin}" if skin else "")
    out = [f'<!-- sh:start -->\n<header class="{cls}" id="site-header">\n  <div class="sh-in">',
           '    <a class="sh-brand" href="/">Shawn Corvec</a>',
           # the finder (2026-09-27): a search palette over every plot on the site, opened here or with Ctrl/Cmd-K;
           # the behaviour is in assets/site.js, the index in assets/site/catalog.json (scripts/site/build_catalog.py)
           '    <button class="sh-find" type="button" aria-haspopup="dialog" aria-keyshortcuts="Control+K Meta+K" '
           # the icon carries its own size and drawing attributes (inline style on the shapes outranks page CSS): a page
           # whose stylesheet styles bare svg/circle, or a stale cached site.css, drew it as a large black disc
           'title="Find a plot (Ctrl K)"><svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" '
           'stroke-width="1.8" stroke-linecap="round" aria-hidden="true" focusable="false">'
           '<circle cx="8.5" cy="8.5" r="5.5" style="fill:none;stroke:currentColor"/>'
           '<path d="M12.6 12.6 17 17" style="fill:none;stroke:currentColor"/></svg>'
           '<span class="sh-find-t">Find a plot</span><kbd>Ctrl K</kbd></button>',
           '    <button class="sh-toggle" type="button" aria-expanded="false" aria-controls="sh-menu">Menu</button>',
           '    <nav class="sh-nav" id="sh-menu" aria-label="Site">\n      <ul class="sh-list">']
    for n, (title, _blurb, items, *short) in enumerate(PRODUCTS):
        # the header's short label on the one-line desktop row, the full title in the phone / compact menu
        label = (f'<span class="sh-lf">{title}</span><span class="sh-ls">{short[0]}</span>') if short else title
        in_group = any(it[0] == page for it in group_items(items))
        mid = f"sh-g{n}"
        columns = bool(items) and isinstance(items[0][1], list)
        out.append('        <li class="sh-item sh-has-menu">')
        out.append(f'          <button class="sh-link sh-menubtn" type="button" aria-expanded="false" aria-controls="{mid}"'
                   f'{" aria-current=page" if in_group else ""}>{label}</button>')
        if columns:
            out.append(f'          <div class="sh-menu sh-menu--cols" id="{mid}">')
            for sub, its in items:
                out.append(f'            <div>\n              <h3>{sub}</h3>\n              <ul>')
                pages_here = {h for h, *_ in its if "#" not in h}
                for href, text, *_ in its:
                    cls = ' class="sh-subitem"' if "#" in href and href.split("#")[0] in pages_here else ""
                    out.append(f'                <li{cls}><a href="{href}"{_current(href, page)}>{text}</a></li>')
                out.append('              </ul>\n            </div>')
            out.append('          </div>\n        </li>')
        else:
            out.append(f'          <div class="sh-menu sh-menu--list" id="{mid}">\n            <ul>')
            for href, text, *_ in items:
                out.append(f'              <li><a href="{href}"{_current(href, page)}>{text}</a></li>')
            out.append('            </ul>\n          </div>\n        </li>')
    for href, label in PRIMARY:
        out.append(f'        <li class="sh-item"><a class="sh-link" href="{href}"{_current(href, page)}>{label}</a></li>')
    if SPECIAL:
        href, label = SPECIAL; cur = ' aria-current="page"' if href.strip("/") + "/" in "/" + page.lstrip("/") else ""
        out.append(f'        <li class="sh-item sh-item--special"><a class="sh-special" href="{href}"{cur}>{label}</a></li>')
    # the person, after a divider: a small quiet menu that does not read as one of the products
    mine = any(h == page for h, _ in PERSONAL)
    out.append('        <li class="sh-item sh-has-menu sh-item--me">')
    out.append(f'          <button class="sh-link sh-menubtn sh-me" type="button" aria-expanded="false" aria-controls="sh-me"'
               f'{" aria-current=page" if mine else ""}>About</button>')
    out.append('          <div class="sh-menu sh-menu--list sh-menu--me" id="sh-me">\n            <ul>')
    for i, (href, label) in enumerate(PERSONAL):
        sep = ' class="sh-me-ext"' if i == len(PERSONAL) - len(PROFILES) else ""
        out.append(f'              <li{sep}><a href="{href}"{_current(href, page)}>{label}</a></li>')
    out.append('            </ul>\n          </div>\n        </li>')
    out.append('      </ul>\n    </nav>\n  </div>\n</header>')
    return "\n".join(out).replace(' aria-current=page', ' aria-current="page"')


def homepage_rows() -> str:
    """The homepage product list (index.html, between <!-- hp:start --> and <!-- hp:end -->), from PRODUCTS: the
    same classes as the hand-written rows it replaced (.group / .rows / .sub / .row / .what / .when), so the
    homepage CSS and its freshness script (status.json keyed by href) work unchanged."""
    out = ["<!-- hp:start -->"]
    for title, blurb, items, *_ in PRODUCTS:
        out.append(f'    <div class="group">\n      <h3>{title}<small>{blurb}</small></h3>\n      <ul class="rows">')
        cols = bool(items) and isinstance(items[0][1], list)
        # a menu sub-item (a "#" link under its own page) is covered by its page's row
        colrows = [(sub, [it for it in its if not ("#" in it[0] and it[0].split("#")[0] in {h for h, *_ in its if "#" not in h})])
                   for sub, its in (items if cols else [(None, items)])]
        if cols and all(len(r) <= 1 for _, r in colrows):
            colrows = [(None, [it for _, r in colrows for it in r])]   # one row per column: no subheadings, one grid
        for sub, its in colrows:
            if sub:
                out.append(f'        <li class="sub">{sub}</li>')
            for href, label, what, when in its:
                out.append(f'        <li class="row"><a href="{href}">{label}</a><span class="what">{what}</span>'
                           f'<span class="when">{when}</span></li>')
        out.append("      </ul>\n    </div>\n")
    out.append("    <!-- hp:end -->")
    return "\n".join(out)


def stamp_homepage(check: bool = False) -> int:
    p = REPO / "index.html"
    html = p.read_text()
    if "<!-- hp:start -->" not in html:
        # first run: replace the hand-written groups between the index head and the end of the section
        m = re.search(r'(<section class="index" id="products">.*?</div>\s*\n)(\s*<div class="group">.*?)(\s*</section>)', html, re.S)
        if not m:
            raise SystemExit("index.html: product groups not found")
        html2 = html[:m.start(2)] + "\n    " + homepage_rows() + html[m.end(2):]
    else:
        html2 = re.sub(r"<!-- hp:start -->.*?<!-- hp:end -->", lambda _: homepage_rows(), html, count=1, flags=re.S)
    if html2 == html:
        return 0
    if check:
        print("out of date: index.html (product rows)")
    else:
        p.write_text(html2); print("stamped index.html product rows")
    return 1


SECTION_TABS = False


def tabs_html(key: str, page: str, dark: bool) -> str:
    cls = "st" + (" st--dark" if dark else "")
    links = "\n".join(f'    <a href="{h}"{_current(h, page)}>{l}</a>' for h, l in TABS[key])
    return f'<nav class="{cls}" aria-label="Section">\n  <div class="st-in">\n{links}\n  </div>\n</nav>'


def footer_html(dark: bool) -> str:
    cls = "sf" + (" sf--dark" if dark else "")
    site = [("/", "Home"), ("/catalog.html", "Every plot"), ("/topics/", "Explainers"), ("/stats.html", "Visitor stats")]
    me = [("/about.html", "About"), ("/resume.html", "Resume"), ("/research.html", "Publications"),
          ("https://github.com/scorvec", "GitHub"), ("https://www.linkedin.com/in/shawn-corvec-35895b231/", "LinkedIn"),
          ("https://scholar.google.com/citations?user=EYLRCJIAAAAJ&amp;hl=en", "Google Scholar")]
    ext = ' rel="noopener"'
    li = lambda xs: "\n".join(f'        <li><a href="{h}"{ext if h.startswith("http") else ""}>{l}</a></li>' for h, l in xs)
    return (
        f'<!-- sf:start -->\n<footer class="{cls}">\n  <div class="sf-in">\n'
        '    <div class="sf-group">\n      <p class="sf-h" role="heading" aria-level="2">This site</p>\n      <ul>\n' + li(site) + '\n      </ul>\n'
        '      <p class="sf-disclaimer">This site is under constant development. Nothing here is checked before publication and there is no expectation of accuracy, completeness or availability; do not rely on it for decisions. Built from open data; sources are credited on each page.</p>\n'
        '    </div>\n'
        '    <div class="sf-group sf-group--me">\n      <p class="sf-h" role="heading" aria-level="2">Shawn Corvec</p>\n      <ul>\n' + li(me) + '\n      </ul>\n    </div>\n'
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
    # Section tab strips (the second banner under the header) were retired 2026-09-07 (user: "two
    # top menus is confusing, keep the top one"); the dropdown menu already reaches every page.
    # The `tabs` keys stay in PAGES so the strips can come back with one flag.
    if cfg.get("tabs") and SECTION_TABS:
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
        if new in html:                                   # already applied: never re-apply
            continue
        if old not in html:
            print(f"  warning: {cfg['path']}: fix not found: {old[:50]!r}", file=sys.stderr)
        html = html.replace(old, new)

    if "/assets/site.css" not in html:
        snippet = HEAD_SNIPPET
        if "family=Source+Serif+4" in html:                # the page already loads the fonts
            snippet = "\n".join(l for l in snippet.splitlines() if "fonts.g" not in l) + "\n"
        if "</head>" in html:
            html = html.replace("</head>", snippet + "</head>", 1)
        else:                                              # bare HTML5 document without <head> tags
            html = html.replace("<!-- sh:start -->", snippet + "<!-- sh:start -->", 1)

    # rail.js (one-figure-at-a-time viewer) rides with the chrome; it does nothing unless the page
    # carries data-rail, so every page gets the include and pages opt in by markup alone.
    if "/assets/rail.js" not in html and '<script src="/assets/site.js" defer></script>' in html:
        html = html.replace('<script src="/assets/site.js" defer></script>',
                            '<script src="/assets/site.js" defer></script>\n<script src="/assets/rail.js" defer></script>', 1)

    if cfg.get("footer", True):
        foot = footer_html(dark)
        if "<!-- sf:start -->" in html:
            html = re.sub(r"<!-- sf:start -->.*?<!-- sf:end -->", lambda _: foot, html, count=1, flags=re.S)
        elif "</body>" in html:
            html = html.replace("</body>", foot + "\n</body>", 1)
        else:                                              # bare document (no <body>): footer goes last
            html = html.rstrip() + "\n" + foot + "\n"
    return orig, html


def stamp_all(check: bool = False, only: set | None = None, quiet: bool = False) -> int:
    """Stamp every page in PAGES (or the subset in `only`, repo-relative paths). Idempotent.
    Called by enso_site.render_all after it regenerates the ENSO pages from their partials —
    those partials carry no header, so without this every SST run shipped the old raw nav."""
    changed = 0
    for cfg in PAGES:
        if only is not None and cfg["path"] not in only:
            continue
        if not (REPO / cfg["path"]).exists():
            continue
        orig, new = stamp(cfg)
        if orig != new:
            changed += 1
            if check:
                print(f"out of date: {cfg['path']}")
            else:
                (REPO / cfg["path"]).write_text(new)
                if not quiet:
                    print(f"stamped {cfg['path']}")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report pages that would change; write nothing")
    args = ap.parse_args()
    changed = stamp_all(check=args.check) + stamp_homepage(check=args.check)
    if args.check:
        return 1 if changed else 0
    print(f"{changed} page(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
