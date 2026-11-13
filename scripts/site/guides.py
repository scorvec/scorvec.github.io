#!/usr/bin/env python3
"""Page guides: what every figure of a product page shows, how it is made and what it is built from, on a page of its
own under /guides/, so that the product pages carry numbers and figures and a link.

A guide's text is the fragment scripts/site/guides/<slug>.html: an optional <section id="about"> on the page as a
whole, then <h2> group headings and one <section id="KEY"><h3>name</h3>...</section> per figure. KEY is the figure's
about id on its stage page without the "about-" (STAGE.products[...].about), which is where the figure's "About this
figure" link lands; a section's data-fig gives the fragment that opens the figure where that is not the KEY. The
contents list and each section's link back to its figure are built here. Text that belongs to a seasonal product sits
between <!-- season:name --> markers and is left out while the product is (enso_site.OFF_SEASON).

    python scripts/site/guides.py            # writes guides/*.html, stamps the site chrome, updates sitemap.xml
"""

from __future__ import annotations

import datetime as dt
import html
import re
import sys

import topics as T

sys.path.insert(0, str(T.REPO / "scripts" / "sst"))
import enso_site

SRC = T.REPO / "scripts" / "site" / "guides"
OUT = T.REPO / "guides"

# slug (the fragment's and the page's file name), the product page, its name, and the guide's description
GUIDES = [
    dict(
        slug="enso",
        page="/enso.html",
        name="El Niño monitor",
        desc="What each figure of the El Niño monitor shows and how it is made: the daily ONI and RONI estimates and "
        "their grading, the satellite views, the winds and the Southern Oscillation, the subsurface ocean and the "
        "impacts on the Americas.",
    ),
    dict(
        slug="circulation",
        page="/circulation.html",
        name="circulation page",
        desc="What each figure of the circulation page shows and how it is made: the dynamic tropopause, the North "
        "Pacific jet and its branches, wave activity flux and teleconnections, the Walker and Hadley circulations, "
        "tropical rain, the subtropical jets, atmospheric angular momentum and the surface torques.",
    ),
    dict(
        slug="stratosphere",
        page="/stratosphere.html",
        name="stratosphere page",
        desc="What each figure of the stratosphere page shows and how it is made: the polar vortex and the chance of a "
        "sudden warming, the waves that drive it, the descent to the surface, weeks 3 to 5, the season ahead, the "
        "QBO and the record of past winters.",
    ),
    dict(
        slug="tropical-waves",
        page="/tropical-waves.html",
        name="tropical waves page",
        desc="What each figure of the tropical waves page shows and how it is made: the filtered Kelvin, equatorial "
        "Rossby and MJO signals along the equator, the wave trackers, the Kelvin-wave forecast and the ocean's "
        "Kelvin waves.",
    ),
    dict(
        slug="mjo",
        page="/mjo.html",
        name="MJO forecast page",
        desc="What the figures of the MJO forecast page show and how they are made: the time-longitude chart of the "
        "MJO with the El Niño pattern taken out and its phase reading, and the successive ECMWF RMM forecasts.",
    ),
    dict(
        slug="ar",
        page="/ar.html",
        name="atmospheric rivers page",
        desc="What each figure of the atmospheric rivers page shows and how it is made: integrated vapour transport "
        "in four models, the West Coast landfall views, the ensembles' AR odds, weeks 3 to 5 and precipitable water.",
    ),
    dict(
        slug="aifs-verify",
        page="/aifs-verify.html",
        name="AIFS verification page",
        desc="What each figure of the AIFS verification page shows and how it is made: the question the page asks, the "
        "spectral matching of member 0, the radiosonde and ERA5 truths, the scorecard's significance tests, the "
        "scores by lead and the power spectra.",
    ),
    dict(
        slug="seasonal",
        page="/seasonal.html",
        name="seasonal forecasts page",
        desc="What each figure of the seasonal forecasts page shows and how it is made: the seven C3S systems' maps, "
        "terciles and indices, SEAS5's ENSO and ocean indices, maps, teleconnections, regimes and impact products, "
        "NOAA's SFS beta and CanSIPS, and the significance tests behind the shading.",
    ),
]

STYLE = """<style>
  .guide .lede { margin: 0 0 1.4rem; }
  .guide .toc { border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); padding: 0.9rem 1.1rem; margin: 0 0 1rem; font-size: 0.95em; }
  .guide .toc dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.5rem 1.3rem; margin: 0; }
  .guide .toc dt { font-weight: 600; }
  .guide .toc dd { margin: 0; display: flex; flex-wrap: wrap; gap: 0.2rem 1.1rem; line-height: 1.5; }
  @media (max-width: 640px) {
    .guide .toc dl { grid-template-columns: 1fr; gap: 0.15rem; }
    .guide .toc dd { margin-bottom: 0.7rem; }
  }
  .guide section { scroll-margin-top: 1rem; }
  .guide section:target { border-left: 3px solid var(--accent); padding-left: 0.9rem; margin-left: calc(-0.9rem - 3px); }   /* the figure the reader came from */
  .guide h3 { margin: 1.9rem 0 0.15rem; text-align: left; }
  .guide .open { font-size: 0.92em; margin: 0 0 0.6rem; }
</style>"""

SECTION = re.compile(r'<section id="([^"]+)"(?: data-fig="([^"]+)")?>\s*<h3>(.*?)</h3>', re.S)
BYLINE = (
    '<p class="byline">Written by Claude, Anthropic\'s AI model, at the direction of <a href="/">Shawn Corvec</a>, '
    "meteorologist.</p>"
)


def cap(name: str) -> str:
    """A page's name as it opens a line."""
    return name[0].upper() + name[1:]


def contents(body: str) -> str:
    """The contents list: each <h2> group of the fragment with links to its figures' sections."""
    rows = []
    for group in re.split(r"(?=<h2>)", body):
        figs = SECTION.findall(group)
        name = re.match(r"<h2>(.*?)</h2>", group, re.S)
        if name and figs:
            links = "".join(f'<a href="#{key}">{label}</a>' for key, _, label in figs)
            rows.append(f"<dt>{name.group(1)}</dt><dd>{links}</dd>")
    return f'<nav class="toc" aria-label="Figures"><dl>{"".join(rows)}</dl></nav>'


def guide_page(g: dict) -> str:
    url = f"{T.SITE}/guides/{g['slug']}.html"
    title = f"Guide to the {g['name']}"
    body = enso_site.in_season((SRC / f"{g['slug']}.html").read_text()).strip()
    toc = contents(body)
    # a link to "#figure" was the page switching figures: here it opens that figure, unless the guide has the section
    here = {"about"} | {key for key, _, _ in SECTION.findall(body)}
    body = re.sub(r'href="#([^"]+)"', lambda m: m[0] if m[1] in here else f'href="{g["page"]}#{m[1]}"', body)
    body = SECTION.sub(
        lambda m: (
            f'{m.group(0)}\n<p class="open"><a href="{g["page"]}#{m.group(2) or m.group(1)}">Open this figure</a></p>'
        ),
        body,
    )
    ld = {
        "@context": "https://schema.org",
        "@type": "TechArticle",
        "headline": title,
        "description": g["desc"],
        "url": url,
        "inLanguage": "en",
        "author": T.PERSON,
        "publisher": T.PERSON,
        "isPartOf": T.SITE + g["page"],
    }
    e = html.escape
    return (
        T.head(title, g["desc"], url, ld, math="$" in body, image=T.share_image(None), extra=STYLE)
        + f"""<main>
<article class="topic guide">
  <p class="crumbs"><a href="/">Home</a> › <a href="/guides/">Page guides</a> › {e(cap(g["name"]))}</p>
  <h1>{e(title)}</h1>
  <p class="lede">What each figure shows, how it is made and what it is built from. <a href="{g["page"]}">Open the {e(g["name"])}</a></p>
{toc}
{body}
  {BYLINE}
</article>
</main>
</body>
</html>
"""
    )


def hub_page() -> str:
    url = f"{T.SITE}/guides/"
    title = "Page guides: what each figure shows"
    desc = "A guide to each product page: what every figure shows, how it is made and what it is built from."
    ld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "url": url,
        "name": title,
        "description": desc,
        "author": T.PERSON,
        "mainEntity": {
            "@type": "ItemList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": i + 1,
                    "url": f"{T.SITE}/guides/{g['slug']}.html",
                    "name": f"Guide to the {g['name']}",
                }
                for i, g in enumerate(GUIDES)
            ],
        },
    }
    items = "".join(
        f'<li><a href="/guides/{g["slug"]}.html">{html.escape(cap(g["name"]))}</a><p>{html.escape(g["desc"])}</p></li>'
        for g in GUIDES
    )
    return (
        T.head(title, desc, url, ld, math=False, image=T.share_image(None), extra=STYLE)
        + f"""<main>
<article class="topic guide">
  <p class="crumbs"><a href="/">Home</a> › Page guides</p>
  <h1>Page guides</h1>
  <p class="lede">The product pages show numbers and figures; what each figure shows, how it is made and what it is
  built from is written up here, one guide a page. The physics behind the diagnostics is in the
  <a href="/topics/">explainers</a>.</p>
  <ul class="hub-list">{items}</ul>
  {BYLINE}
</article>
</main>
</body>
</html>
"""
    )


def main() -> int:
    OUT.mkdir(exist_ok=True)
    pages = {f"guides/{g['slug']}.html": guide_page(g) for g in GUIDES} | {"guides/index.html": hub_page()}
    before = {p: (T.REPO / p).read_text() if (T.REPO / p).exists() else "" for p in pages}
    for p, text in pages.items():
        (T.REPO / p).write_text(text)
    import apply_chrome as A  # (PAGES picks up guides/*.html)

    A.stamp_all(only=set(pages), quiet=True)
    changed = [p for p in pages if (T.REPO / p).read_text() != before[p]]
    T.update_sitemap([p.removesuffix("index.html") for p in changed], dt.date.today().isoformat())
    print(f"wrote {len(pages)} pages to {OUT}, {len(changed)} changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
