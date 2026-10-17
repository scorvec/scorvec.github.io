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
import functools
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ── the navigation, as data ──────────────────────────────────────────────────
# Menus are BY THEME, with a ⌘K finder and a catalogue for individual plots. The product menus hold products only;
# the person (about, resume, publications, profiles) lives in PERSONAL, apart.
# A page lives under its topic whatever model it comes from. Items are (href, label, what, when): the menus use
# the first two, the homepage rows (between <!-- hp:start --> and <!-- hp:end --> in index.html) all four.
# A group's items may be a flat list or a list of (subtitle, items) columns. Group titles are also the catalogue's
# topic labels (scripts/site/build_catalog.py TOPICS) - rename both together.
PRODUCTS = [
    # THEME-FIRST: Mid-latitude weather / Global circulation and tropics / Stratosphere / Weeks to seasons / Studies and
    # verification / Tools. Tropical cyclones and the days 1-15 upper-air loops have their own entries; the tropical-
    # cyclone products are diagnostic only. Products only (the person lives in PERSONAL) and never a group called
    # "Weather" or "Research".
    (
        "Mid-latitude weather",
        "The next two weeks: storms, hazards and the flow aloft",
        [
            (
                "Surface and hazards",
                [
                    (
                        "/models.html",
                        "Short-range model maps",
                        "Zoomable maps of the newest HRRR, RRFS and RDPS runs for aviation, severe and winter weather and renewables: radar with precipitation type, accumulation, snow-band diagnostics (frontogenesis, EPV*, dendritic-zone lift), ceiling, visibility, temperature, dewpoint, wind, HRRR entraining CAPE (ECAPE), smoke and sunshine, with contours and hover values.",
                        "Four times daily",
                    ),
                    # snowbands.html is retired: its maps are the "Winter: snow bands" group of /models.html
                    (
                        "/ar.html",
                        "Atmospheric rivers",
                        "Integrated vapour transport maps to day 7 from AIFS, IFS, GFS and GDPS; West Coast landfall and AR-scale ranks; AR probabilities to day 15 from four ensembles and their blend; GEFS weeks 3–5 AR-frequency anomalies where significant.",
                        "Twice daily",
                    ),
                ],
            ),
            (
                "Upper air",
                [
                    (
                        "/circulation.html#dt",
                        "Dynamic tropopause",
                        "The 2-PVU surface and PV on the 330 and 350 K isentropes from the AIFS-ENS control run: the hemisphere and five regions.",
                        "Daily",
                    ),
                    (
                        "/circulation.html#pacjet",
                        "North Pacific jet",
                        "250 hPa jet maps, the jet axis from every member, and the jet's extension and shift phases with what they usually bring downstream.",
                        "Twice daily",
                    ),
                    (
                        "/circulation.html#waf",
                        "Wave activity flux",
                        "Takaya–Nakamura wave activity flux at 250 hPa, the ensemble's quasi-stationary wave trains.",
                        "Daily",
                    ),
                ],
            ),
        ],
    ),
    # the subtropical jets and the angular-momentum products sit here with the tropics (the North Pacific jet is
    # mid-latitude weather); the columns follow circulation.html's groups, and the menu keeps a short label
    (
        "Global circulation and tropics",
        "ENSO, the MJO, tropical cyclones, the Walker and Hadley cells, the jets and the atmosphere's angular momentum",
        [
            (
                "ENSO and the MJO",
                [
                    (
                        "/enso.html",
                        "El Niño monitor",
                        "Daily ONI and RONI, Niño-region sea surface temperatures, the subsurface, winds and convection, with SST anomaly maps and animation.",
                        "Daily",
                    ),
                    (
                        "/circulation.html#mslp",
                        "Pacific pressure and wind",
                        "Sea-level pressure and 10 m wind over the tropical and subtropical Pacific to day 15: the AIFS and IFS ensembles, separately and blended, and the AIFS, IFS and GDPS single runs.",
                        "Twice daily",
                    ),
                    (
                        "/circulation.html#gdps_ir",
                        "Simulated satellite (GDPS)",
                        "Environment Canada's GDPS forecast drawn as an infrared satellite loop over the tropical Pacific, 3-hourly to day 10: the cold cloud tops are deep convection.",
                        "Daily",
                    ),
                    (
                        "/mjo.html",
                        "MJO forecast",
                        "Real-time multivariate MJO index from the AIFS-ENS, IFS-ENS, GEFS and GEPS ensembles on one phase diagram, standard or ENSO-removed, with the observed track and what each phase has meant for temperature and rain.",
                        "Twice daily",
                    ),
                    (
                        "/tropical-waves.html",
                        "Tropical waves",
                        "Where the equatorial waves are now: Kelvin and Rossby wave trackers along the equator with the forecast, the OLR wave decomposition and ocean Kelvin waves.",
                        "Daily",
                    ),
                ],
            ),
            (
                "Tropical cyclones",
                [
                    (
                        "/circulation.html#tcshear",
                        "Shear and environment",
                        "Each storm's full environmental wind profile with the storm removed: layer and generalized shear, vortex tilt against the shear and the radar, the ventilation index and the trough causing the shear, across 51 ensemble members and four models.",
                        "Every 6 hours with a storm",
                    ),
                    (
                        "/recon.html",
                        "Hurricane reconnaissance",
                        "Atlantic hurricane hunter flights decoded and mapped: the track by SFMR or flight-level wind, dropsonde skew-Ts, vortex fixes, best track and forecast, with the aircraft's quality flags respected.",
                        "Live during flights",
                    ),
                ],
            ),
            (
                "Jets and momentum",
                [
                    (
                        "/circulation.html#jets",
                        "Subtropical jets",
                        "Speed and latitude of each hemisphere's subtropical jet core in the AIFS ensemble against the normal, and the zonal-mean wind cross-section through the 15-day forecast.",
                        "Daily",
                    ),
                    (
                        "/circulation.html#aam",
                        "Angular momentum and torques",
                        "Global atmospheric angular momentum and the mountain and friction torques that change it.",
                        "Daily",
                    ),
                ],
            ),
            (
                "Tropical overturning",
                [
                    (
                        "/circulation.html#walker",
                        "Walker circulation",
                        "The equatorial Walker circulation: every analysed week since July 2025 and the AIFS-ENS, IFS-ENS and GEFS forecasts to day 15 against ERA5, the circulation by ocean basin, and its weekly departure from Pacific forcing.",
                        "Daily",
                    ),
                    (
                        "/circulation.html#hadley",
                        "Hadley cells",
                        "The meridional overturning: every analysed week since July 2025 and three ensembles' forecasts to day 15 against ERA5, each cell's strength and poleward edge, and this season against every El Niño since 1963.",
                        "Daily",
                    ),
                    (
                        "/circulation.html#vpot",
                        "Velocity potential",
                        "200 hPa velocity potential and the irrotational wind: where the tropics are venting and where they are subsiding.",
                        "Twice daily",
                    ),
                    (
                        "/circulation.html#gdps150",
                        "Upper-level outflow (GDPS)",
                        "Environment Canada's GDPS 150 hPa wind and height to day 10: the outflow layer of the Hadley and Walker cells and the subtropical jets.",
                        "Daily",
                    ),
                    (
                        "/circulation.html#gillrain",
                        "Rain-forced circulation (Gill model)",
                        "How much of the tropical upper-level circulation the rain itself explains: the Gill (1980) model forced by IMERG and AIFS-ENS rainfall against the observed and forecast velocity potential.",
                        "Twice daily",
                    ),
                    (
                        "/circulation.html#rainfc",
                        "Rain forecast, weeks 1–2",
                        "Bias-adjusted rain for weeks 1 and 2 from four ensembles (IFS-ENS, AIFS-ENS, GEFS, GEPS) and their verified blend over the tropics and the globe: corrected rain, its anomaly, the favoured tercile against the GPCP normal and the change since the previous day's runs, where significant.",
                        "Daily",
                    ),
                ],
            ),
        ],
        "Circulation/tropics",
    ),
    # one hub: stratosphere.html holds every stratosphere product by timescale (including the GEPS/GEFS/SEAS5 vortex
    # products and the QBO tracker), and this menu lists its sections; the one group heading must not repeat the
    # menu button or the hub link
    (
        "Stratosphere",
        "The polar vortex and the waves going up into it, from today's analysis to the season, and the QBO",
        [
            (
                "By timescale",
                [
                    (
                        "/stratosphere.html",
                        "Every stratosphere product",
                        "Every stratosphere product in one place, by timescale: analyses and the QBO now; AIFS-ENS, IFS-ENS, GEFS and the Canadian models' vortex, heat flux and wave driving for the next two weeks; GEPS and GEFS to day 35; the C3S models and SEAS5 for the season; and every sudden warming since 1980 with what followed.",
                        "Twice daily",
                    ),
                    (
                        "/stratosphere.html#pv",
                        "Now: analyses and observations",
                        "GEOS FP potential vorticity, the momentum budget and the Brewer–Dobson circulation, the water-vapour tape recorder and the QBO zero-wind line.",
                        "Daily",
                    ),
                    (
                        "/stratosphere.html#vortex",
                        "Next two weeks",
                        "AIFS-ENS, IFS-ENS, GEFS and the Canadian models to day 15: vortex winds and polar-cap heights, 100 hPa heat flux, E–P flux, wave-1, the upper stratosphere and the dripping paint.",
                        "Twice daily",
                    ),
                    (
                        "/stratosphere.html#vortex35",
                        "Weeks 3–5",
                        "GEPS and GEFS to day 35: drift-corrected vortex diagnostics, loops and dripping paint, and the 100 hPa heat flux.",
                        "Daily",
                    ),
                    (
                        "/stratosphere.html#s5vortex",
                        "Season",
                        "The C3S multi-model December vortex, calibrated on its hindcasts; SEAS5's vortex, QBO and annular mode, sudden-warming risk and polar-cap heights; and this winter's odds from the QBO and ENSO.",
                        "Monthly",
                    ),
                    (
                        "/stratosphere.html#shwinters",
                        "History and studies",
                        "Every winter and sudden warming since 1980 in MERRA-2 and 15,000 more in CMIP6, with what followed at the surface.",
                        "Static",
                    ),
                    (
                        "/stratosphere.html#qbo",
                        "QBO tracker",
                        "Equatorial stratospheric winds at 10 to 100 hPa from radiosondes, from the 1950s to the present.",
                        "Weekly",
                    ),
                ],
            ),
        ],
    ),
    (
        "Weeks to seasons",
        "From weeks two to five out to the coming seasons",
        [
            (
                "Extended range",
                [
                    (
                        "/subseasonal.html",
                        "GEPS extended ensemble",
                        "Environment Canada's extended ensemble to day 35 for the Americas and the tropics: weekly anomalies against its own reforecast, terciles, teleconnections, regimes and change since the previous run (its polar vortex is on the stratosphere page).",
                        "Mondays and Thursdays",
                    ),
                    (
                        "/gefs.html",
                        "GEFS extended ensemble",
                        "NOAA's GEFS extended ensemble to day 35 in the same layout: weekly anomalies against its own reforecast and the change since a week earlier (its drift-corrected polar vortex is on the stratosphere page).",
                        "Daily",
                    ),
                    # the GEPS/GEFS layout for NOAA's SFS beta2; its daily products are gated on a data check (NOAA's beta2 daily
                    # forecast store was defective in its first two issues)
                    (
                        "/sfs-subseasonal.html",
                        "SFS extended (experimental)",
                        "NOAA's experimental SFS beta2, 31 members from the first of each month, up to six weeks in the same layout against its own 1991–2020 reforecast, with its hindcast skill against GEFS, GEPS and CFSv2.",
                        "Monthly",
                    ),
                    # ERA5 statistics blended with the 35-day GEFS (scripts/monthly, monthly-outlook.yml)
                    (
                        "/monthly-outlook.html",
                        "Claude's month-ahead forecast",
                        "A forecast designed and run by Claude: next month's 2 m temperature, precipitation and 500 hPa height over the Northern Hemisphere, issued on the 15th as anomalies and tercile probabilities. A statistical forecast from ERA5 (trend, RONI, QBO and vortex regression, tropical constructed analogues, persistence, analogue years) is blended with the 35-day GEFS run of the 15th where its reforecast shows it helps, with skill masks and a running record against ERA5 and the C3S models.",
                        "Monthly, on the 15th",
                    ),
                ],
            ),
            (
                "Seasons",
                [
                    # the C3S systems with the NOAA SFS page folded in
                    (
                        "/seasonal.html",
                        "Seasonal model forecasts",
                        "Eight C3S systems from seven centres plus NOAA's SFS beta: anomaly maps and tercile probabilities for every C3S model and their mean, the member-level ECMWF SEAS5 outlook (indices, teleconnections, impacts; its stratosphere is on the stratosphere page) and the 31-member SFS plumes and monthly maps.",
                        "Monthly",
                    ),
                    (
                        "/enso-forecasts.html",
                        "ENSO forecasts",
                        "Every member of five centres' seasonal models for Niño-3.4, with percentile fans and a record of how past forecasts did.",
                        "Monthly",
                    ),
                ],
            ),
        ],
    ),
    (
        "Studies and verification",
        "What the record and the models say, and how the forecasts score",
        [
            (
                "Studies",
                [
                    (
                        "/enso.html#imp_reg",
                        "ENSO impacts on the Americas",
                        "Thousands of El Niño and La Niña events in 16 CMIP6 models against the observed record: regressions, composites, super and east-based El Niños, 500 hPa wave trains, and why the PDO mostly follows the atmosphere rather than driving it. Significant results only.",
                        "Static",
                    ),
                    (
                        "/mjo.html#mi",
                        "MJO impacts",
                        "What each MJO phase has meant for temperature and rain, season by season, in CMIP6 models and the observed record. Significant results only.",
                        "Static",
                    ),
                    (
                        "/aifs-aam-budget.html",
                        "AI models and conservation laws",
                        "Do AI weather models obey the conservation laws? ECMWF's AI ensemble (AIFS-ENS), the single AIFS and the physics ensemble against angular momentum and its torques, dry-air mass, water and energy. Winter 2025–26, days 1–15.",
                        "Static",
                    ),
                    (
                        "/torque-enso.html",
                        "El Niño and angular momentum",
                        "Why the atmosphere spins up in El Niño: a standing shift in the Andes mountain torque supplies the extra angular momentum, and ERA5's model loses westerly momentum that its analysis puts back, mostly over the oceans. ERA5 1959–2026, 41 CMIP6 models, the 2026 budget hour by hour. Significant results only.",
                        "Static",
                    ),
                    (
                        "/topics/",
                        "Explainers",
                        "How to read the products: the equations and the physics behind each diagnostic.",
                        "As written",
                    ),
                ],
            ),
            (
                "Verification",
                [
                    (
                        "/aifs-verify.html",
                        "AIFS single versus ensemble control",
                        "The two ECMWF AIFS configurations compared as deterministic models against radiosondes and ERA5, with the ensemble member spectrally matched to the single model and the ensemble mean for reference.",
                        "Every 00Z and 12Z run",
                    ),
                ],
            ),
        ],
        "Studies/verification",
    ),
    (
        "Tools",
        "Explore the data yourself",
        [
            (
                "/catalog.html",
                "Every plot on the site",
                "One searchable sheet of every figure and loop, filterable by topic, model, forecast range and region, each linked to its exact view.",
                "Rebuilt with the pages",
            ),
            (
                "/skewt/",
                "Sounding explorer",
                "Real-time and archived radiosondes worldwide, drawn and analysed in the browser, with launch history and record rings.",
                "Hourly mirror",
            ),
            (
                "/asos5.html",
                "Five-minute airport observations",
                "Five-minute ASOS observations at 22 major US airports: temperature, dewpoint, wind, visibility, ceiling and weather, with running daily highs and lows.",
                "Live",
            ),
            (
                "/climate.html",
                "US climate trends",
                "Every US county and calendar month since 1895: temperature, precipitation and degree-day trends per decade with significance, the 1991–2020 normal and what normal is now.",
                "Monthly",
            ),
            (
                "/fci.html",
                "US credit conditions",
                "How tight credit is in the US, weekly since 1990: corporate bond spreads, commercial paper, bank lending and the loan officer survey, mortgage rates, the fed funds rate, the yield curve, stocks and the dollar, with sub-indices for companies, mortgages, rates, markets, liquidity and private credit.",
                "Daily",
            ),
        ],
    ),
]

# The person, kept apart from the products: one "About" item at the far right of the header,
# behind a divider, and its own footer group. External profiles open in the same tab like every other link.
PROFILES = [
    ("https://github.com/scorvec", "GitHub"),
    ("https://scholar.google.com/citations?user=EYLRCJIAAAAJ&amp;hl=en", "Google Scholar"),
    ("https://www.linkedin.com/in/shawn-corvec-35895b231/", "LinkedIn"),
]
PERSONAL = [("/about.html", "About me"), ("/resume.html", "Resume"), ("/research.html", "Publications"), *PROFILES]


def _split_columns(items):
    """A columned menu's (subtitle, items) groups in two stacks, in order, split where the stacks' heights (a heading
    plus its links) come closest: each column then runs straight down, with no gap beside a shorter neighbour."""
    if len(items) < 2:
        return [items]
    h = [1 + len(its) for _, its in items]
    k = min(range(1, len(items)), key=lambda k: abs(sum(h[:k]) - sum(h[k:])))
    return [items[:k], items[k:]]


def group_items(items):
    """Flatten a group's items whether it is a flat list or (subtitle, items) columns."""
    if items and isinstance(items[0][1], list):
        return [it for _, sub in items for it in sub]
    return list(items)


# Plain header links between the product menus and the special button (none since the About menu took Resume).
PRIMARY: list = []
# A highlighted button, apart from the menus, on every page.
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
# synced: the page's <head> belongs to the repository it is synced from (the midterms pages): header and footer
#   only, no analytics tag, integrity attributes, font links or asset versions stamped (see head_tags).
PAGES = [
    dict(path="index.html", mode="nav", skin="overlay", footer=False),
    dict(
        path="enso-forecasts.html",
        mode="site-header",
        tabs="enso",
        fixes=[
            ("padding: 6.4rem 2.2rem 2.5rem;", "padding: 2rem 2.2rem 2.5rem;"),
            (
                "main { padding: 6rem 1rem 2rem; max-width: 100%; }",
                "main { padding: 1.5rem 1rem 2rem; max-width: 100%; }",
            ),
            ("main { padding: 9.5rem 1rem 3rem; }", "main { padding: 1.5rem 1rem 3rem; }"),
        ],
    ),
    dict(
        path="seasonal.html", mode="after-body", tabs="enso"
    ),  # seasonal model forecasts: C3S + SEAS5 (was seas5.html) + NOAA SFS (was sfs.html)
    dict(path="enso.html", mode="after-body", tabs="enso"),
    dict(path="circulation.html", mode="after-body", tabs="enso"),
    dict(path="stratosphere.html", mode="after-body", tabs="enso"),  # every stratosphere product
    dict(path="tropical-waves.html", mode="after-body", tabs="enso"),  # every tropical-wave product
    dict(
        path="subseasonal.html", mode="site-header", fixes=[]
    ),  # styled by /assets/outlook.css; no padding patches needed
    dict(path="gefs.html", mode="site-header"),  # GEFS extended page (scripts/gefs, gefs.yml), GEPS layout
    dict(path="sfs-subseasonal.html", mode="site-header"),  # SFS beta2 page (scripts/sfs/sfs_sub.py), GEFS layout
    dict(path="monthly-outlook.html", mode="site-header"),  # statistical month-ahead outlook (scripts/monthly)
    dict(
        path="subseasonal-method.html", mode="site-header"
    ),  # methodology note split out of subseasonal.html; linked from there only, not in PRODUCTS
    dict(path="mjo.html", mode="nav"),
    # ecape.html: retired, now a redirect stub to models.html#hrrr/ecape (no chrome)
    dict(
        path="aifs-verify.html",
        mode="nav",
        fixes=[
            ("padding: 7.2rem 2.5rem 5rem;", "padding: 2.5rem 2.5rem 5rem;"),
            (
                "main { padding: 6.3rem 1rem 3rem; max-width: 100%; }",
                "main { padding: 1.5rem 1rem 3rem; max-width: 100%; }",
            ),
        ],
    ),
    dict(path="asos5.html", mode="site-header"),  # no fixed-header padding to undo
    dict(
        path="recon.html", mode="after-body"
    ),  # Atlantic recon plotter (scripts/recon, recon.yml -> recon-data branch)
    # sfs.html: folded into seasonal.html, now a redirect stub (no chrome)
    dict(path="ar.html", mode="after-body"),
    dict(path="models.html", mode="after-body"),  # short-range model viewer (scripts/models), WebGL
    # snowbands.html: retired, now a noindex redirect stub to models.html#hrrr/sbing (no chrome)
    dict(path="catalog.html", mode="after-body"),  # every plot on the site (scripts/site/build_catalog.py)
    dict(path="aifs-aam-budget.html", mode="after-body"),  # static study, built offline
    dict(path="torque-enso.html", mode="after-body"),  # static study, scripts/mjo/src/torque_enso_page.py
    dict(path="about.html", mode="after-body"),  # the person: bio, resume, publications, profiles
    dict(path="climate.html", mode="after-body"),
    dict(path="fci.html", mode="after-body"),
    dict(
        path="research.html",
        mode="nav",
        fixes=[
            ("padding: 7.5rem 2rem 5rem;", "padding: 2.5rem 2rem 5rem;"),
            ("main { padding: 6rem 1.5rem 4rem; }", "main { padding: 1.5rem 1.5rem 4rem; }"),
            ('class="scholar-link"', 'class="button button--secondary"'),  # the standard outlined button
        ],
    ),
    dict(
        path="resume.html",
        mode="nav",
        fixes=[
            ("    min-height: 100vh;\n    padding-top: 57px;", "    min-height: calc(100vh - 57px);"),
            ("    position: sticky;\n    top: 57px;", "    position: sticky;\n    top: 64px;"),
            ("    position: sticky;\n    top: 56px;", "    position: sticky;\n    top: 64px;"),
        ],
    ),
    dict(
        path="stats.html",
        mode="nav",
        fixes=[
            ("padding: 7.5rem 2.5rem 5rem;", "padding: 2.5rem 2.5rem 5rem;"),
            ("main { padding: 6.5rem 1.5rem 3.5rem; }", "main { padding: 1.5rem 1.5rem 3.5rem; }"),
            ("main { padding: 5.5rem 1rem 3rem; }", "main { padding: 1rem 1rem 3rem; }"),
            ('style="margin:4.5rem 0 0.5rem;"', 'style="margin:0.5rem 0 0.5rem;"'),
        ],
    ),
    dict(
        path="skewt/index.html",
        mode="before",
        anchor="<nav>",
        tabs="skewt",
        footer=False,
        local_assets=("sharplib.js", "app.js"),  # the explorer's own scripts, versioned like the shared ones
        drop=[
            r'\s*<a href="\.\./index\.html" class="nav-name">Shawn Corvec</a>',
            r'\s*<a href="\.\./index\.html" class="back">← HOME</a>',
        ],
    ),
    dict(path="skewt/methodology.html", mode="nav", tabs="skewt"),
    dict(path="skewt/gaps.html", mode="nav", tabs="skewt"),
    # qbo/index.html: folded into stratosphere.html ("QBO from radiosondes"), now a redirect stub (no chrome)
    dict(
        path="midterms/index.html", mode="after-body", footer=True, synced=True
    ),  # 2026 midterm forecast (synced daily from github.com/scorvec/midterms-2026 by midterms-sync.yml)
    dict(
        path="midterms/about.html", mode="after-body", footer=True, synced=True
    ),  # how the midterm forecast works + sources
    dict(
        path="midterms/polls.html", mode="after-body", footer=True, synced=True
    ),  # every poll in the midterm forecast and its weight
    dict(
        path="midterms/whatif.html", mode="after-body", footer=True, synced=True
    ),  # "What if?": the forecast re-simulated with the reader's assumptions
    dict(
        path="midterms/legislatures.html", mode="after-body", footer=True, synced=True
    ),  # state legislative chambers (experimental)
    # the 404 body is a centring flexbox: stack it so the header spans the top and the message centres below
    dict(
        path="404.html",
        mode="after-body",
        footer=False,
        fixes=[
            (
                "display:flex;\nmin-height:100vh;align-items:center;justify-content:center;margin:0;",
                "display:flex;flex-direction:column;\nmin-height:100vh;margin:0;",
            ),
            (
                "p{color:#8b8ba3}a{color:#64d2ff}",
                "p{color:#8b8ba3}a{color:#64d2ff}body>div:not(.fp){margin:auto;padding:2rem}",
            ),
        ],
    ),
]
# topic explainers (written by scripts/site/topics.py): every page in topics/ gets the header and footer
PAGES += [
    dict(path=f"topics/{p.name}", mode="after-body", footer=True) for p in sorted((REPO / "topics").glob("*.html"))
]

HEAD_SNIPPET = '<link rel="stylesheet" href="/assets/site.css">\n<script src="/assets/site.js" defer></script>\n'

# ── web fonts: one Google Fonts request per page, for exactly the families it draws ─────────────────────────────
# site.css sets the type on every page (Source Serif 4 for titles, Source Sans 3 for text), and its html-prefixed
# rules outrank the families a page names itself, so those two are all most pages render. A page that draws another
# family (a monospace column, canvas or chart text) lists it in FONT_EXTRA. Every other Google Fonts tag in the
# head is dropped: the families a page still names but never renders cost a render-blocking request each.
FONT_BASE = (
    "Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400",
    "Source+Sans+3:ital,wght@0,400;0,500;0,600;1,400",
)
FONT_EXTRA = {
    "resume.html": ("DM+Mono:wght@400",),  # the dates and tags column (--mono)
    "skewt/index.html": ("Inter:wght@400;600;700", "Lora:wght@600"),  # the sounding canvas and its export (app.js)
}
_FONT_TAG = re.compile(r'[ \t]*<link\b[^>]*\bhref="https://fonts\.(?:googleapis|gstatic)\.com[^"]*"[^>]*>[ \t]*\n?')


def font_links(path: str) -> str:
    fams = "&".join("family=" + f for f in FONT_BASE + FONT_EXTRA.get(path, ()))
    return (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        f'<link href="https://fonts.googleapis.com/css2?{fams}&display=swap" rel="stylesheet">\n'
    )


def with_fonts(html: str, path: str) -> str:
    """The page's head with its one font block where its first font tag was (else right before the site
    stylesheet); every other font tag removed."""
    end = html.find("</head>")
    head, rest = (html[:end], html[end:]) if end >= 0 else (html, "")
    first = _FONT_TAG.search(head)
    css = re.search(r'[ \t]*<link rel="stylesheet" href="/assets/site\.css', head)
    at = first.start() if first else css.start() if css else len(head)  # no font tag before `at`
    return head[:at] + font_links(path) + _FONT_TAG.sub("", head[at:]) + rest


# ── cache-busting: the shared assets carry their content hash, so a new page never runs with yesterday's script ─
# The query is ignored by the server; it only makes a changed file a new URL for browsers and the CDN. Re-stamped on
# every run, so a page picks up a new hash whenever any workflow or edit re-applies the chrome.
VERSIONED = (
    "site.css",
    "site.js",
    "stage.css",
    "stage.js",
    "outlook.css",
    "outlook-charts.js",
    "frames_root.js",
    "lightbox.css",
    "lightbox.js",
    "seas5-cards.js",
    "c3s-vortex-card.js",
)
_ASSET_REF = re.compile(r"""(["'])(/?assets/(""" + "|".join(map(re.escape, VERSIONED)) + r"""))(?:\?v=\w*)?\1""")


@functools.cache
def asset_hash(name: str) -> str:
    return hashlib.sha256((REPO / "assets" / name).read_bytes()).hexdigest()[:10]


def version_assets(html: str) -> str:
    return _ASSET_REF.sub(lambda m: f"{m[1]}{m[2]}?v={asset_hash(m[3])}{m[1]}", html)


def version_local(html: str, page: str, names: tuple) -> str:
    """Stamp a page's own scripts (`src="name"` beside the page, not under /assets/) with their content hash."""
    for name in names:
        digest = hashlib.sha256((REPO / page).parent.joinpath(name).read_bytes()).hexdigest()[:10]
        html = re.sub(r'src="' + re.escape(name) + r'(?:\?v=\w*)?"', f'src="{name}?v={digest}"', html)
    return html


# the loop embeds (in an iframe on most pages): no chrome, but their shared scripts carry the same versions
EMBEDS = [dict(path="sst_anim.html", mode="embed"), dict(path="loop.html", mode="embed")]


# ── analytics and third-party integrity, stamped into every page's <head> ─────────────────────────────────────
# ONE GoatCounter tag for the whole site: the versioned script, pinned by its hash. Every other GoatCounter script
# tag on a stamped page (a generator's own, the old unversioned count.js) is replaced by it. Embeds and redirect
# stubs are not in PAGES and never count.
GOATCOUNTER = (
    '<script data-goatcounter="https://scorvec.goatcounter.com/count" async src="https://gc.zgo.at/count.v4.js" '
    'integrity="sha384-nRw6qfbWyJha9LhsOtSb2YJDyZdKvvCFh0fJYlkquSFjUxp9FVNugbfy8q1jdxI+" crossorigin="anonymous"></script>'
)
_GC_TAG = re.compile(r"[ \t]*<script\b[^>]*gc\.zgo\.at/count[^>]*>\s*</script>[ \t]*\n?")
# Subresource Integrity (sha384 of the exact file) for the pinned CDN files the pages load as tags: a <script> or
# <link> with one of these URLs and no integrity attribute gets one, plus crossorigin, so the browser refuses a
# changed copy. Code that loads one of them itself sets the same hash (assets/seas5-cards.js,
# assets/c3s-vortex-card.js, and the Plotly loaders of ar, climate, enso, enso-forecasts, fci, seasonal and
# stratosphere). A new library version is a new URL: add its hash here or it simply loads without one.
SRI = {
    "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js": "sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg",
    "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js": "sha384-43gviWU0YVjaDtb/GhzOouOXtZMP/7XUzwPTstBeZFe/+rCMvRwr4yROQP43s0Xk",
    "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css": "sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+",
    "https://cdn.jsdelivr.net/npm/plotly.js-cartesian-dist-min@2.35.2/plotly-cartesian.min.js": "sha384-gKF7EaMDoWW1zLWkouldmVWF5jqEKrAswAciW73iNzqlomabVVPO60XkJvL5q/Ie",
    "https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js": "sha384-CjloA8y00+1SDAUkjs099PVfnY2KmDC2BZnws9kh8D/lX1s46w6EPhpXdqMfjK6i",
    "https://cdnjs.cloudflare.com/ajax/libs/topojson/3.0.2/topojson.min.js": "sha384-9dCJK6nh7skY14HrcvlLYlFga9/MehJjL9ONWRflmiXNRuf8p2jiF4Y5PR881PTq",
}
_ATTRS = r"""(?:[^>"']|"[^"]*"|'[^']*')*?"""  # attributes, quoted values may hold ">"
_CDN_TAG = re.compile(rf'<(?:script|link)\b{_ATTRS}\s(?:src|href)="({"|".join(map(re.escape, SRI))})"{_ATTRS}(/?)>')


def _with_sri(m: re.Match) -> str:
    tag = m.group(0)
    if re.search(r"\sintegrity=", tag):
        return tag
    extra = f' integrity="{SRI[m.group(1)]}"'
    if not re.search(r"\scrossorigin\b", tag):
        extra += ' crossorigin="anonymous"'
    end = len(tag) - 1 - len(m.group(2))
    return tag[:end].rstrip() + extra + (" />" if m.group(2) else ">")


def head_tags(html: str) -> str:
    """The site's one analytics tag at the end of <head>, and integrity attributes on the pinned CDN tags."""
    html = _GC_TAG.sub("", html)
    if "</head>" in html:
        html = html.replace("</head>", GOATCOUNTER + "\n</head>", 1)
    else:  # bare HTML5 document without <head> tags
        html = html.replace("<!-- sh:start -->", GOATCOUNTER + "\n<!-- sh:start -->", 1)
    return _CDN_TAG.sub(_with_sri, html)


def _current(href: str, page: str) -> str:
    return ' aria-current="page"' if href == page else ""


def header_html(page: str, skin: str, skip: bool = False) -> str:
    cls = "sh" + (f" sh--{skin}" if skin else "")
    # the first tab stop: past the header menus to the page's main region (only where the page has one)
    skip_link = '<a class="skip-link" href="#main">Skip to main content</a>\n' if skip else ""
    out = [
        f'<!-- sh:start -->\n{skip_link}<header class="{cls}" id="site-header">\n  <div class="sh-in">',
        '    <a class="sh-brand" href="/">Shawn Corvec</a>',
        # the finder: a search palette over every plot on the site, opened here or with Ctrl/Cmd-K;
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
        '    <nav class="sh-nav" id="sh-menu" aria-label="Site">\n      <ul class="sh-list">',
    ]
    for n, (title, _blurb, items, *short) in enumerate(PRODUCTS):
        # the header's short label on the one-line desktop row, the full title in the phone / compact menu
        label = (f'<span class="sh-lf">{title}</span><span class="sh-ls">{short[0]}</span>') if short else title
        in_group = any(it[0] == page for it in group_items(items))
        mid = f"sh-g{n}"
        columns = bool(items) and isinstance(items[0][1], list)
        out.append('        <li class="sh-item sh-has-menu">')
        out.append(
            f'          <button class="sh-link sh-menubtn" type="button" aria-expanded="false" aria-controls="{mid}"'
            f"{' aria-current=page' if in_group else ''}>{label}</button>"
        )
        if columns:
            cols = _split_columns(items)
            one = " sh-menu--one" if len(cols) == 1 else ""  # a group with one column: a narrow panel, no empty half
            out.append(f'          <div class="sh-menu sh-menu--cols{one}" id="{mid}">')
            for col in cols:
                out.append('            <div class="sh-col">')
                for sub, its in col:
                    out.append(f"              <div>\n                <h3>{sub}</h3>\n                <ul>")
                    pages_here = {h for h, *_ in its if "#" not in h}
                    for href, text, *_ in its:
                        cls = ' class="sh-subitem"' if "#" in href and href.split("#")[0] in pages_here else ""
                        out.append(f'                  <li{cls}><a href="{href}"{_current(href, page)}>{text}</a></li>')
                    out.append("                </ul>\n              </div>")
                out.append("            </div>")
            out.append("          </div>\n        </li>")
        else:
            out.append(f'          <div class="sh-menu sh-menu--list" id="{mid}">\n            <ul>')
            for href, text, *_ in items:
                out.append(f'              <li><a href="{href}"{_current(href, page)}>{text}</a></li>')
            out.append("            </ul>\n          </div>\n        </li>")
    for href, label in PRIMARY:
        out.append(
            f'        <li class="sh-item"><a class="sh-link" href="{href}"{_current(href, page)}>{label}</a></li>'
        )
    if SPECIAL:
        href, label = SPECIAL
        cur = ' aria-current="page"' if href.strip("/") + "/" in "/" + page.lstrip("/") else ""
        out.append(
            f'        <li class="sh-item sh-item--special"><a class="sh-special" href="{href}"{cur}>{label}</a></li>'
        )
    # the person, after a divider: a small quiet menu that does not read as one of the products
    mine = any(h == page for h, _ in PERSONAL)
    out.append('        <li class="sh-item sh-has-menu sh-item--me">')
    out.append(
        f'          <button class="sh-link sh-menubtn sh-me" type="button" aria-expanded="false" aria-controls="sh-me"'
        f"{' aria-current=page' if mine else ''}>About</button>"
    )
    out.append('          <div class="sh-menu sh-menu--list sh-menu--me" id="sh-me">\n            <ul>')
    for i, (href, label) in enumerate(PERSONAL):
        sep = ' class="sh-me-ext"' if i == len(PERSONAL) - len(PROFILES) else ""
        out.append(f'              <li{sep}><a href="{href}"{_current(href, page)}>{label}</a></li>')
    out.append("            </ul>\n          </div>\n        </li>")
    out.append("      </ul>\n    </nav>\n  </div>\n</header>")
    return "\n".join(out).replace(" aria-current=page", ' aria-current="page"')


# The homepage shows the menus' six themes as tiles: per theme a one-line purpose, its key pages as plain
# links (the long descriptions live in the menus' pages and the catalogue) and a link to the theme's plots in the
# catalogue. HOME: group title -> (catalogue topic id, short purpose, featured hrefs in order). A group missing here
# features its first six pages.
HOME = {
    "Mid-latitude weather": (
        "weather",
        "Storms, hazards and the flow aloft over the next two weeks",
        ["/models.html", "/ar.html", "/circulation.html#dt", "/circulation.html#pacjet", "/circulation.html#waf"],
    ),
    "Global circulation and tropics": (
        "drivers",
        "El Niño, the MJO, cyclones, the overturning cells, the jets and the atmosphere's momentum",
        [
            "/enso.html",
            "/mjo.html",
            "/tropical-waves.html",
            "/circulation.html#tcshear",
            "/circulation.html#walker",
            "/circulation.html#hadley",
            "/circulation.html#jets",
            "/circulation.html#aam",
        ],
    ),
    "Stratosphere": (
        "circulation",
        "The polar vortex from today's analysis to the season, and the QBO",
        [
            "/stratosphere.html",
            "/stratosphere.html#vortex",
            "/stratosphere.html#vortex35",
            "/stratosphere.html#s5vortex",
            "/stratosphere.html#qbo",
            "/stratosphere.html#pv",
        ],
    ),
    "Weeks to seasons": (
        "outlooks",
        "Extended ensembles to day 35 and the seasonal models",
        ["/subseasonal.html", "/gefs.html", "/seasonal.html", "/enso-forecasts.html"],
    ),
    "Studies and verification": (
        "research",
        "What the record and the models say, and how forecasts score",
        [
            "/enso.html#imp_reg",
            "/mjo.html#mi",
            "/aifs-aam-budget.html",
            "/torque-enso.html",
            "/aifs-verify.html",
            "/topics/",
        ],
    ),
    "Tools": (
        "tools",
        "Explore the data yourself",
        ["/catalog.html", "/skewt/", "/asos5.html", "/climate.html", "/fci.html"],
    ),
}


def homepage_rows() -> str:
    """The homepage theme tiles (index.html, between <!-- hp:start --> and <!-- hp:end -->), from PRODUCTS + HOME. Each
    link carries its page's one-line description as a tooltip."""

    def esc(t):
        return t.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")

    out = ["<!-- hp:start -->", '    <div class="sgs">']
    for title, blurb, items, *_ in PRODUCTS:
        allit = group_items(items)
        by = {h: (lab, what) for h, lab, what, _ in allit}
        topic, short, feat = HOME.get(title, (None, blurb, [h for h, *_ in allit if "#" not in h][:6]))
        feat = [h for h in feat if h in by]
        rest = [h for h, *_ in allit if h not in feat]

        def li(h, by=by):
            return f'          <li><a href="{h}" title="{esc(by[h][1])}">{by[h][0]}</a></li>'

        out.append(f'      <section class="sg" data-topic="{topic or ""}">')
        out.append(f"        <h3>{title}</h3>\n        <p>{short}</p>\n        <ul>")
        out += [li(h) for h in feat]
        out.append("        </ul>")
        if rest:  # the theme's other pages, one tap away instead of on show
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
        m = re.search(
            r'(<section class="index" id="products">.*?</div>\s*\n)(\s*<div class="group">.*?)(\s*</section>)',
            html,
            re.S,
        )
        if not m:
            raise SystemExit("index.html: product groups not found")
        html2 = html[: m.start(2)] + "\n    " + homepage_rows() + html[m.end(2) :]
    else:
        html2 = re.sub(r"<!-- hp:start -->.*?<!-- hp:end -->", lambda _: homepage_rows(), html, count=1, flags=re.S)
    if html2 == html:
        return 0
    if check:
        print("out of date: index.html (product rows)")
    else:
        p.write_text(html2)
        print("stamped index.html product rows")
    return 1


SECTION_TABS = False


def tabs_html(key: str, page: str, dark: bool) -> str:
    cls = "st" + (" st--dark" if dark else "")
    links = "\n".join(f'    <a href="{h}"{_current(h, page)}>{l}</a>' for h, l in TABS[key])
    return f'<nav class="{cls}" aria-label="Section">\n  <div class="st-in">\n{links}\n  </div>\n</nav>'


def footer_html(dark: bool) -> str:
    cls = "sf" + (" sf--dark" if dark else "")
    site = [
        ("/", "Home"),
        ("/catalog.html", "Every plot"),
        ("/topics/", "Explainers"),
        ("/stats.html", "Visitor stats"),
    ]
    me = [
        ("/about.html", "About"),
        ("/resume.html", "Resume"),
        ("/research.html", "Publications"),
        ("https://github.com/scorvec", "GitHub"),
        ("https://www.linkedin.com/in/shawn-corvec-35895b231/", "LinkedIn"),
        ("https://scholar.google.com/citations?user=EYLRCJIAAAAJ&amp;hl=en", "Google Scholar"),
    ]
    ext = ' rel="noopener"'

    def li(xs):
        return "\n".join(f'        <li><a href="{h}"{ext if h.startswith("http") else ""}>{l}</a></li>' for h, l in xs)

    return (
        f'<!-- sf:start -->\n<footer class="{cls}">\n  <div class="sf-in">\n'
        '    <div class="sf-group">\n      <p class="sf-h" role="heading" aria-level="2">This site</p>\n      <ul>\n'
        + li(site)
        + "\n      </ul>\n"
        '      <p class="sf-disclaimer">This site is under constant development. Its products are generated automatically and published without review by a person; each page describes its automated checks and significance tests. There is no expectation of accuracy, completeness or availability; do not rely on it for decisions. Built from open data; sources are credited on each page.</p>\n'
        "    </div>\n"
        '    <div class="sf-group sf-group--me">\n      <p class="sf-h" role="heading" aria-level="2">Shawn Corvec</p>\n      <ul>\n'
        + li(me)
        + "\n      </ul>\n    </div>\n"
        "  </div>\n</footer>\n<!-- sf:end -->"
    )


def is_dark(html: str) -> bool:
    """A page whose --bg is darker than mid-grey gets the dark header skin."""
    m = re.search(r"--bg:\s*#([0-9a-fA-F]{6})\b", html)
    if not m:
        m = re.search(r"body\s*\{[^}]*background:\s*#([0-9a-fA-F]{6})", html)
    if not m:
        return False
    r, g, b = (int(m.group(1)[i : i + 2], 16) for i in (0, 2, 4))
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 100


def page_key(path: str) -> str:
    return "/" + path.replace("index.html", "")


def stamp(cfg: dict) -> tuple[str, str]:
    p = REPO / cfg["path"]
    html = p.read_text()
    orig = html
    if cfg["mode"] == "embed":
        return orig, version_assets(html)
    page = page_key(cfg["path"])
    dark = cfg.get("skin") == "dark" or (is_dark(html) and cfg.get("skin") != "overlay")
    skin = cfg.get("skin") or ("dark" if dark else "")

    # the skip link's target is the page's <main>; a synced page brings its own markup, so it only gets the
    # link once its <main> carries id="main" upstream
    if not cfg.get("synced"):
        html = re.sub(r"<main\b(?![^>]*\sid=)", '<main id="main"', html, count=1)
    block = header_html(page, skin, skip=re.search(r'<main\b[^>]*\sid="main"', html) is not None)
    # Section tab strips (a second banner under the header) are off: two top menus confuse, and the
    # dropdown menu already reaches every page.
    # The `tabs` keys stay in PAGES so the strips can come back with one flag.
    if cfg.get("tabs") and SECTION_TABS:
        block += "\n" + tabs_html(cfg["tabs"], page, dark)
    block += "\n<!-- sh:end -->"

    for pat in [cfg["drop"]] if isinstance(cfg.get("drop"), str) else cfg.get("drop", []):
        html = re.sub(pat, "", html, count=1)
    if "<!-- sh:start -->" in html:  # re-stamp
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
        if new in html:  # already applied: never re-apply
            continue
        if old not in html:
            print(f"  warning: {cfg['path']}: fix not found: {old[:50]!r}", file=sys.stderr)
        html = html.replace(old, new)

    if "/assets/site.css" not in html:
        if "</head>" in html:
            html = html.replace("</head>", HEAD_SNIPPET + "</head>", 1)
        else:  # bare HTML5 document without <head> tags
            html = html.replace("<!-- sh:start -->", HEAD_SNIPPET + "<!-- sh:start -->", 1)

    if not cfg.get("synced"):
        html = head_tags(html)
        html = version_assets(with_fonts(html, cfg["path"]))
        if cfg.get("local_assets"):
            html = version_local(html, cfg["path"], cfg["local_assets"])

    if cfg.get("footer", True):
        foot = footer_html(dark)
        if "<!-- sf:start -->" in html:
            html = re.sub(r"<!-- sf:start -->.*?<!-- sf:end -->", lambda _: foot, html, count=1, flags=re.S)
        elif "</body>" in html:
            html = html.replace("</body>", foot + "\n</body>", 1)
        else:  # bare document (no <body>): footer goes last
            html = html.rstrip() + "\n" + foot + "\n"
    return orig, html


def stamp_all(check: bool = False, only: set | None = None, quiet: bool = False) -> int:
    """Stamp every page in PAGES (or the subset in `only`, repo-relative paths). Idempotent.
    Called by enso_site.render_all after it regenerates the ENSO pages from their partials —
    those partials carry no header, so without this every SST run shipped the old raw nav."""
    changed = 0
    for cfg in PAGES + EMBEDS:
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
