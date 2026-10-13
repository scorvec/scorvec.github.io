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
    # 2026-10-03, user: "reorganize the order and categories of the option on the dropdown menu at the top of the site";
    # chose the THEME-FIRST layout (over a range-first one): Days 1-15 / Tropics / Stratosphere / Weeks to seasons /
    # Studies and verification / Tools. Tropical cyclones and the days 1-15 upper-air loops got their own entries (they had
    # been reachable only inside circulation.html; diagnostic only - the 1991-2020 "what follows a recurving storm" record
    # was withdrawn the same evening, user: "keep the TC-jet stuff as diagnostic as possible"); the QBO moved to
    # Stratosphere; "Climate drivers" and "Atmospheric
    # circulation" are gone. The 2026-09-27 rules stand: products only (the person lives in PERSONAL) and never a group
    # called "Weather" or "Research". Group titles are the catalogue's topic labels (build_catalog.TOPICS) - rename both.
    # 2026-10-03, user: "Days 1-15" -> "Mid-latitude weather", and Pacific pressure and wind moved under the El Niño monitor.
    ("Mid-latitude weather", "The next two weeks: storms, hazards and the flow aloft", [
        ("Surface and hazards", [
            ("/models.html", "Short-range model maps",
             "Zoomable maps of the newest HRRR, RRFS and RDPS runs for aviation, severe and winter weather and renewables: radar with precipitation type, accumulation, snow-band diagnostics (frontogenesis, EPV*, dendritic-zone lift), ceiling, visibility, temperature, dewpoint, wind, HRRR entraining CAPE (ECAPE), smoke and sunshine, with contours and hover values.",
             "Four times daily"),
            # 2026-10-09, user: a real-time national NEXRAD composite that runs entirely in the browser ("something that
            # wouldnt rely on actions to generate the data"): IEM's N0Q composite fetched and recoloured client-side
            ("/radar.html", "US Radar Explorer",
             "The national NEXRAD base-reflectivity mosaic from NOAA MRMS with GOES lightning flashes, live every two minutes and decoded in the browser: zoom from the whole country to a city, loop the last hours at your own speed, read the dBZ anywhere, on the traditional NWS colour scale.",
             "Every two minutes"),
            # snowbands.html retired 2026-10-09 (user: "fold in the snowband diagnostics into the short-term model viewer",
            # then "remove the old snowbands page"): its maps are the "Winter: snow bands" group of /models.html
            ("/ar.html", "Atmospheric rivers",
             "Integrated vapour transport maps to day 7 from AIFS, IFS, GFS and GDPS; West Coast landfall and AR-scale ranks; AR probabilities to day 15 from four ensembles and their blend; GEFS weeks 3–5 AR-frequency anomalies where significant.",
             "Twice daily"),
        ]),
        ("Upper air", [
            ("/circulation.html#dt", "Dynamic tropopause",
             "The 2-PVU surface and PV on the 330 and 350 K isentropes from the AIFS ensemble: the hemisphere and four regions, with tropical storms and their outflow marked.",
             "Daily"),
            ("/circulation.html#waf", "Wave activity flux",
             "Takaya–Nakamura wave activity flux at 250 hPa, the ensemble's quasi-stationary wave trains.",
             "Daily"),
        ]),
    ]),
    # 2026-10-12, user: "On the homepage I would change tropics to 'Global circulation and tropics' and throw anything
    # related to momentum and jets in there": the North Pacific jet (from Mid-latitude weather) and angular momentum
    # (from Stratosphere) moved here with the subtropical jets; the menu keeps a short label
    ("Global circulation and tropics", "ENSO, the MJO, tropical cyclones, the jets and the atmosphere's angular momentum", [
        ("ENSO and the MJO", [
            ("/enso.html", "El Niño monitor",
             "Daily ONI and RONI, Niño-region sea surface temperatures, the subsurface, winds and convection, with SST anomaly maps and animation.",
             "Daily"),
            ("/circulation.html#mslp", "Pacific pressure and wind",
             "Sea-level pressure and 10 m wind over the Pacific from the AIFS and IFS ensembles, separately and blended, to day 15.",
             "Twice daily"),
            ("/mjo.html", "MJO forecast",
             "Real-time multivariate MJO index from the ECMWF AIFS ensemble, with the observed phase-space track, an ENSO-removed version, and what each phase has meant for temperature and rain.",
             "Twice daily"),
            ("/tropical-waves.html", "Tropical waves",
             "Where the equatorial waves are now: Kelvin and Rossby wave trackers along the equator with the forecast, the OLR wave decomposition and ocean Kelvin waves.",
             "Daily"),
        ]),
        ("Tropical cyclones", [
            ("/circulation.html#tcjet", "Tropical cyclones and the jet",
             "How strongly each storm's outflow meets the jet stream, the wave packets along the jet with the storms' positions, and a watch when a recurvature looks likely.",
             "Daily"),
            ("/circulation.html#tcens", "Tropical cyclone ensembles",
             "Every AIFS and IFS ensemble member's track, the odds and timing of recurvature, and how deep the storm gets once it becomes extratropical.",
             "Daily"),
            ("/circulation.html#tcphase", "Cyclone phase space",
             "Hart's diagrams: whether each storm is still a warm-core tropical cyclone or turning into a cold-core extratropical low, and when.",
             "Daily"),
            ("/circulation.html#tcshear", "Shear and environment",
             "Each storm's full environmental wind profile with the storm removed: layer and generalized shear, vortex tilt against the shear and the radar, the ventilation index and the trough causing the shear, across 51 ensemble members and four models.",
             "Every 6 hours with a storm"),
            ("/recon.html", "Hurricane reconnaissance",
             "Atlantic hurricane hunter flights decoded and mapped: the track by SFMR or flight-level wind, dropsonde skew-Ts, vortex fixes, best track and forecast, with the aircraft's quality flags respected.",
             "Live during flights"),
        ]),
        ("Jets and momentum", [
            ("/circulation.html#pacjet", "North Pacific jet",
             "250 hPa jet maps, the jet axis from every member, and the jet's extension and shift phases with what they usually bring downstream.",
             "Twice daily"),
            ("/circulation.html#jets", "Subtropical jets",
             "Speed and latitude of each hemisphere's subtropical jet core in the AIFS ensemble against the normal, and the zonal-mean wind cross-section through the 15-day forecast.",
             "Daily"),
            ("/circulation.html#aam", "Angular momentum and torques",
             "Global atmospheric angular momentum and the mountain and friction torques that change it.",
             "Daily"),
        ]),
        ("Tropical overturning", [
            ("/circulation.html#walker", "Walker and Hadley cells",
             "The tropical overturning circulation from the AIFS ensemble: the Walker circulation by ocean basin and the Hadley cells.",
             "Daily"),
            ("/circulation.html#vpot", "Velocity potential",
             "200 hPa velocity potential and the irrotational wind: where the tropics are venting and where they are subsiding.",
             "Twice daily"),
            ("/circulation.html#gillrain", "Rain-forced circulation (Gill model)",
             "How much of the tropical upper-level circulation the rain itself explains: the Gill (1980) model forced by IMERG and AIFS-ENS rainfall against the observed and forecast velocity potential.",
             "Twice daily"),
        ]),
    ], "Circulation/tropics"),
    # one hub since 2026-10-08 (user: "put all the strat stuff in one place and consolidate"): stratosphere.html holds
    # every stratosphere product by timescale (the GEPS/GEFS/SEAS5 vortex products and the QBO tracker moved in), and
    # this menu lists its sections
    ("Stratosphere", "The polar vortex and the waves going up into it, from today's analysis to the season, and the QBO", [
        ("Polar vortex and QBO", [
            ("/stratosphere.html", "Stratosphere and polar vortex",
             "Every stratosphere product in one place, by timescale: analyses and the QBO now; AIFS-ENS, IFS-ENS and GEFS vortex, heat flux and wave driving for the next two weeks; GEPS and GEFS to day 35; SEAS5's vortex and sudden-warming risk for the season; and every sudden warming since 1980 with what followed.",
             "Twice daily"),
            ("/stratosphere.html#pv", "Now: analyses and observations",
             "GEOS FP potential vorticity, the momentum budget and the Brewer–Dobson circulation, the water-vapour tape recorder, and the QBO from radiosondes.",
             "Daily"),
            ("/stratosphere.html#vortex", "Next two weeks",
             "AIFS-ENS, IFS-ENS and GEFS: vortex winds, 100 hPa heat flux, E–P flux, wave-1 and the upper stratosphere.",
             "Twice daily"),
            ("/stratosphere.html#vortex35", "Weeks 3–5",
             "GEPS and GEFS to day 35, drift-corrected: vortex diagnostics, loops and the dripping paint.",
             "Daily"),
            ("/stratosphere.html#s5vortex", "Season",
             "SEAS5's vortex, QBO and annular mode, sudden-warming risk against its own normal, and the polar-cap heights.",
             "Monthly"),
            ("/stratosphere.html#shwinters", "History and studies",
             "Every winter and sudden warming since 1980 in MERRA-2 and 15,000 more in CMIP6, with what followed at the surface.",
             "Static"),
            ("/stratosphere.html#qbo", "QBO tracker",
             "Equatorial stratospheric winds at 10 to 100 hPa from radiosondes, 1950 to the present.",
             "Weekly"),
        ]),
    ]),
    ("Weeks to seasons", "From weeks two to five out to the coming seasons", [
        ("Weeks 2–5", [
            ("/subseasonal.html", "GEPS extended ensemble",
             "Environment Canada's extended ensemble to day 35 for the Americas and the tropics: weekly anomalies against its own reforecast, terciles, teleconnections, regimes and change since the previous run (its polar vortex is on the stratosphere page).",
             "Mondays and Thursdays"),
            ("/gefs.html", "GEFS extended ensemble",
             "NOAA's GEFS extended ensemble to day 35 in the same layout: weekly anomalies against its own reforecast and the change since a week earlier (its drift-corrected polar vortex is on the stratosphere page).",
             "Daily"),
            # 2026-10-12, user: "make an SFS-beta 2 page like what we have for the GEPS/GEFS"; its daily products are gated on a
            # data check (NOAA's beta2 daily forecast store was defective in its first two issues)
            ("/sfs-subseasonal.html", "SFS extended (experimental)",
             "NOAA's experimental SFS beta2, 31 members from the first of each month, up to six weeks in the same layout against its own 1991–2020 reforecast, with its hindcast skill against GEFS, GEPS and CFSv2.",
             "Monthly"),
        ]),
        ("Seasons", [
            # renamed from "Eight C3S seasonal models" and the NOAA SFS page folded in (2026-10-06, user)
            ("/seasonal.html", "Seasonal model forecasts",
             "Eight C3S systems from seven centres plus NOAA's SFS beta: anomaly maps and tercile probabilities for every C3S model and their mean, the member-level ECMWF SEAS5 outlook (indices, teleconnections, impacts; its stratosphere is on the stratosphere page) and the 31-member SFS plumes and monthly maps.",
             "Monthly"),
            ("/enso-forecasts.html", "ENSO forecasts",
             "Every member of seven centres' seasonal models for Niño-3.4, with percentile fans and a record of how past forecasts did.",
             "Monthly"),
        ]),
    ]),
    ("Studies and verification", "What the record and the models say, and how the forecasts score", [
        ("Studies", [
            ("/enso.html#imp_reg", "ENSO impacts on the Americas",
             "Thousands of El Niño and La Niña events in 16 CMIP6 models against the observed record: regressions, composites, super and east-based El Niños, 500 hPa wave trains, and why the PDO mostly follows the atmosphere rather than driving it. Significant results only.",
             "Static"),
            ("/mjo.html#mi", "MJO impacts",
             "What each MJO phase has meant for temperature and rain, season by season, in CMIP6 models and the observed record. Significant results only.",
             "Static"),
            ("/stratosphere.html#shwinters", "Stratosphere history",
             "Every northern winter and sudden warming since 1980 in MERRA-2, with 15,000 more from nine CMIP6 models: the dripping-paint composites and what usually follows at the surface.",
             "Static"),
            ("/aifs-aam-budget.html", "AINWP and conservation",
             "Do AI weather models obey the conservation laws? ECMWF's AI ensemble (AIFS-ENS), the single AIFS and the physics ensemble against angular momentum and its torques, dry-air mass, water and energy. Winter 2025–26, days 1–15.",
             "Static"),
            ("/torque-enso.html", "El Niño and mountain torque",
             "Does El Niño raise the atmosphere's angular momentum through more mountain-torque events east of the Andes, Rockies and Himalaya, or through a standing shift? ERA5 1959–2026, 41 CMIP6 models, the 2026 budget hour by hour and ERA5's analysis increments. Significant results only.",
             "Static"),
            ("/topics/", "Explainers",
             "How to read the products: the equations and the physics behind each diagnostic.",
             "As written"),
        ]),
        ("Verification", [
            ("/aifs-verify.html", "AIFS single versus member\u00a00",   # no-break space: no lone "0" on its own line
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
        ("/fci.html", "US credit conditions",
         "How tight credit is in the US, weekly since 1990: corporate bond spreads, commercial paper, bank lending and the loan officer survey, mortgage rates, the fed funds rate, the yield curve, stocks and the dollar, with sub-indices for companies, mortgages, rates, markets, liquidity and private credit.",
         "Daily"),
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
    dict(path="seasonal.html", mode="after-body", tabs="enso"),   # seasonal model forecasts: C3S + SEAS5 (was seas5.html) + NOAA SFS (was sfs.html)
    dict(path="enso.html", mode="after-body", tabs="enso"),
    dict(path="circulation.html", mode="after-body", tabs="enso"),
    dict(path="stratosphere.html", mode="after-body", tabs="enso"),   # split out of circulation.html 2026-09-26
    dict(path="tropical-waves.html", mode="after-body", tabs="enso"),   # every tropical-wave product, 2026-10-03
    dict(path="subseasonal.html", mode="site-header", fixes=[]),   # page restyled 2026-09-07 on /assets/outlook.css; no padding patches needed
    dict(path="gefs.html", mode="site-header"),
    dict(path="sfs-subseasonal.html", mode="site-header"),        # SFS beta2 subseasonal page (scripts/sfs/sfs_sub.py, sfs.yml), GEFS layout                    # GEFS extended page (scripts/gefs, gefs.yml), GEPS layout
    dict(path="subseasonal-method.html", mode="site-header"),      # methodology note split out of subseasonal.html; linked from there only, not in PRODUCTS
    dict(path="mjo.html", mode="nav"),
    # ecape.html: retired 2026-09-30, now a redirect stub to models.html#hrrr/ecape (no chrome)
    dict(path="aifs-verify.html", mode="nav", fixes=[
        ("padding: 7.2rem 2.5rem 5rem;", "padding: 2.5rem 2.5rem 5rem;"),
        ("main { padding: 6.3rem 1rem 3rem; max-width: 100%; }", "main { padding: 1.5rem 1rem 3rem; max-width: 100%; }"),
    ]),
    dict(path="asos5.html", mode="site-header"),        # v3 (2026-10-02) has no fixed-header padding to undo
    dict(path="recon.html", mode="after-body"),         # Atlantic recon plotter (scripts/recon, recon.yml -> recon-data branch), 2026-10-07
    # sfs.html: folded into seasonal.html 2026-10-06, now a redirect stub (no chrome)
    dict(path="ar.html", mode="after-body"),
    dict(path="models.html", mode="after-body"),
    dict(path="radar.html", mode="after-body"),                    # US Radar Explorer (2026-10-09): NOAA MRMS base reflectivity (IEM N0Q fallback), client-side only                   # short-range model viewer (scripts/models, models*.yml), WebGL data page
    # snowbands.html: retired 2026-10-09, now a noindex redirect stub to models.html#hrrr/sbing (no chrome)
    dict(path="catalog.html", mode="after-body"),
    dict(path="aifs-aam-budget.html", mode="after-body"),         # static study (2026-10-01), built from ~/aam_budget/scripts/build_page.py
    dict(path="torque-enso.html", mode="after-body"),             # static study (2026-10-13), scripts/mjo/src/torque_enso_page.py
    dict(path="about.html", mode="after-body"),                    # the person: bio, resume, publications, profiles                  # every plot on the site (scripts/site/build_catalog.py)
    dict(path="climate.html", mode="after-body"),
    dict(path="fci.html", mode="after-body"),
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
    # qbo/index.html: folded into stratosphere.html 2026-10-08 ("QBO from radiosondes"), now a redirect stub (no chrome)
    dict(path="midterms/index.html", mode="after-body", footer=True),   # 2026 midterm forecast (synced daily from github.com/scorvec/midterms-2026 by midterms-sync.yml)
    dict(path="midterms/about.html", mode="after-body", footer=True),   # how the midterm forecast works + sources
    dict(path="midterms/polls.html", mode="after-body", footer=True),   # every poll in the midterm forecast and its weight
    dict(path="midterms/whatif.html", mode="after-body", footer=True),  # "What if?": the forecast re-simulated with the reader's assumptions
    dict(path="midterms/legislatures.html", mode="after-body", footer=True),  # state legislative chambers (experimental, 2026-10-05)
    dict(path="cities/index.html", mode="nav"),
    dict(path="cities/verify.html", mode="nav"),
    # the 404 body is a centring flexbox: stack it so the header spans the top and the message centres below
    dict(path="404.html", mode="after-body", footer=False, fixes=[
        ("display:flex;\nmin-height:100vh;align-items:center;justify-content:center;margin:0;",
         "display:flex;flex-direction:column;\nmin-height:100vh;margin:0;"),
        ("p{color:#8b8ba3}a{color:#64d2ff}", "p{color:#8b8ba3}a{color:#64d2ff}body>div:not(.fp){margin:auto;padding:2rem}"),
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


# The homepage shows the menus' six themes as tiles (user 2026-10-12: "a litany of different cards ... remove all these and
# come up with some 'super groups' ... there's a lot of clutter"): per theme a one-line purpose, its key pages as plain
# links (the long descriptions live in the menus' pages and the catalogue) and a link to the theme's plots in the
# catalogue. HOME: group title -> (catalogue topic id, short purpose, featured hrefs in order). A group missing here
# features its first six pages.
HOME = {
    "Mid-latitude weather": ("weather", "Storms, hazards and the flow aloft over the next two weeks",
                             ["/models.html", "/radar.html", "/ar.html", "/circulation.html#dt", "/circulation.html#waf"]),
    "Global circulation and tropics": ("drivers", "El Niño, the MJO, cyclones, the jets and the atmosphere's momentum",
                                       ["/enso.html", "/mjo.html", "/tropical-waves.html", "/circulation.html#tcens",
                                        "/circulation.html#pacjet", "/circulation.html#jets", "/circulation.html#aam",
                                        "/circulation.html#walker"]),
    "Stratosphere": ("circulation", "The polar vortex from today's analysis to the season, and the QBO",
                     ["/stratosphere.html", "/stratosphere.html#vortex", "/stratosphere.html#vortex35", "/stratosphere.html#s5vortex",
                      "/stratosphere.html#qbo", "/stratosphere.html#pv"]),
    "Weeks to seasons": ("outlooks", "Extended ensembles to day 35 and the seasonal models",
                         ["/subseasonal.html", "/gefs.html", "/seasonal.html", "/enso-forecasts.html"]),
    "Studies and verification": ("research", "What the record and the models say, and how forecasts score",
                                 ["/enso.html#imp_reg", "/mjo.html#mi", "/stratosphere.html#shwinters", "/aifs-aam-budget.html",
                                  "/torque-enso.html",
                                  "/aifs-verify.html", "/topics/"]),
    "Tools": ("tools", "Explore the data yourself",
              ["/catalog.html", "/skewt/", "/asos5.html", "/climate.html", "/fci.html"]),
}


def homepage_rows() -> str:
    """The homepage theme tiles (index.html, between <!-- hp:start --> and <!-- hp:end -->), from PRODUCTS + HOME. Each
    link carries its page's one-line description as a tooltip (the per-link "3 h ago" freshness labels were removed
    2026-10-12, user: "not really important")."""
    esc = lambda t: t.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    out = ["<!-- hp:start -->", '    <div class="sgs">']
    for title, blurb, items, *_ in PRODUCTS:
        allit = group_items(items)
        by = {h: (lab, what) for h, lab, what, _ in allit}
        topic, short, feat = HOME.get(title, (None, blurb, [h for h, *_ in allit if "#" not in h][:6]))
        feat = [h for h in feat if h in by]
        rest = [h for h, *_ in allit if h not in feat]
        li = lambda h: f'          <li data-href="{h}"><a href="{h}" title="{esc(by[h][1])}">{by[h][0]}</a></li>'
        out.append(f'      <section class="sg" data-topic="{topic or ""}">')
        out.append(f'        <h3>{title}</h3>\n        <p>{short}</p>\n        <ul>')
        out += [li(h) for h in feat]
        out.append("        </ul>")
        if rest:   # the theme's other pages, one tap away instead of on show
            out.append(f'        <details class="sg-more"><summary>{len(rest)} more</summary>\n          <ul>')
            out += ["  " + li(h) for h in rest]
            out.append("          </ul>\n        </details>")
        if topic and topic != "tools":
            out.append(f'        <a class="sg-all" href="/catalog.html?topic={topic}">All plots in this theme</a>')
        out.append("      </section>")
    out.append("    </div>")
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
