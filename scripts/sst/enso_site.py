#!/usr/bin/env python3
"""Static assembler for the El Niño Monitor pages.

The monitor (enso.html) and its sibling pages in PAGES share chrome
(head / nav / sub-nav / footer) from partials/ and take per-page card
fragments from pages/. Both HTML-writing scripts use this module so a single
edit to the chrome (or the page list) propagates everywhere:

  - sst-roni.py  stamp_html()  -> render_all()  (stamps __CACHE__/__SST_DAY__/__RONI_MONTH__)
  - sst_subsurface.py          -> stamp_tao()   (fills __TAO_DAY__, known later in the run)

Run order matters: sst-roni renders all pages first; sst_subsurface fills the
TAO date afterwards (it isn't known at stamp time). Both leave the other data
tokens already resolved.

Each figure's explanation lives in the page's methods page (/guides/, scripts/site/guides.py), not on the page.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTIALS_DIR = HERE / "partials"
PAGES_DIR = HERE / "pages"

# sub-nav active-class placeholders, one per page (see partials/nav.html)
_SUBNAV_KEYS = ["A_OVERVIEW", "A_SUBSURFACE", "A_FORECASTS", "A_ATMOSPHERE"]

# Seasonal products a page leaves out for now. A page fragment marks what belongs to one with
# <!-- season:name --> ... <!-- /season:name --> (/* season:name */ ... /* /season:name */ inside a script).
OFF_SEASON = {"tc"}  # tropical cyclones, until the next hurricane season
_SEASON = re.compile(
    r"(?:^[ \t]*)?(?:<!--|/\*) season:(\w+) (?:-->|\*/).*?(?:<!--|/\*) /season:\1 (?:-->|\*/)\n?", re.S | re.M
)

PAGES = [
    dict(
        slug="enso",
        out="enso.html",
        active="A_OVERVIEW",
        layout="stage",
        title="El Ni&ntilde;o Monitor &mdash; Daily ONI, RONI &amp; Ni&ntilde;o Indices &middot; Shawn Corvec",
        desc="Daily ONI and RONI estimates, interactive Niño-index charts, OISST and 1 km satellite SST maps, the SOI, "
        "the MEI, and equatorial convection and waves.",
        canonical="https://scorvec.com/enso.html",
        image="https://scorvec.com/assets/sst/roni.webp",
    ),
    dict(
        slug="forecasts",
        out="enso-forecasts.html",
        active="A_FORECASTS",
        title="ENSO Forecasts &mdash; How Strong, When, How East-Based &middot; Shawn Corvec",
        desc="The El Niño outlook of five C3S seasonal systems in plain language: every member, the odds of each peak "
        "strength on ONI and RONI, and how east-based it will be.",
        canonical="https://scorvec.com/enso-forecasts.html",
        image="https://scorvec.com/assets/sst/c3s_nino34.webp",
    ),
    dict(
        slug="circulation",
        out="circulation.html",
        active="A_ATMOSPHERE",
        layout="stage",
        title="Global Circulation and Jets &mdash; AAM, Torque, Walker &middot; Shawn Corvec",
        desc="Daily global circulation from ECMWF AIFS-ENS: wave activity flux, dynamic tropopause, angular momentum "
        "and torques, Hadley and Walker cells, and the jets.",
        canonical="https://scorvec.com/circulation.html",
        image="https://scorvec.com/assets/sst/pacjet_aifs_wk1.webp",
    ),
    # every stratosphere product on one page, by timescale
    dict(
        slug="stratosphere",
        out="stratosphere.html",
        active="A_ATMOSPHERE",
        layout="stage",
        title="Stratosphere and the Polar Vortex &mdash; SSW Forecasts &middot; Shawn Corvec",
        desc="Every stratosphere product by timescale: analyses and the radiosonde QBO, vortex, heat flux and E–P flux "
        "forecasts to day 35, SEAS5, and every SSW since 1980.",
        canonical="https://scorvec.com/stratosphere.html",
        image="https://scorvec.com/assets/sst/nh_vortex.webp",
    ),
    # every tropical-wave product on one page (Tropics menu)
    dict(
        slug="waves",
        out="tropical-waves.html",
        active="A_ATMOSPHERE",
        layout="stage",
        title="Tropical Waves &mdash; Kelvin and Rossby Wave Trackers &middot; Shawn Corvec",
        desc="Convectively coupled equatorial waves: Kelvin, Rossby and MJO filtered OLR Hovmöllers, wave trackers "
        "with the AIFS-ENS forecast, and ocean Kelvin waves.",
        canonical="https://scorvec.com/tropical-waves.html",
        image="https://scorvec.com/assets/sst/wave_tracker.webp",
    ),
]


def in_season(text: str) -> str:
    """The text without the blocks of the products that are out of season."""
    return _SEASON.sub(lambda m: "" if m[1] in OFF_SEASON else m[0], text)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def assemble(page: dict) -> str:
    """Full HTML for one page (still containing the __…__ data tokens).

    Stage pages take head_stage.html / footer_stage.html; every other page takes
    the one partials/head.html. A `chrome` key on a page entry is ignored.
    """
    stage = page.get("layout") == "stage"  # rail + stage pages: outlook.css look, no legacy nav
    head = (
        _read(PARTIALS_DIR / ("head_stage.html" if stage else "head.html"))
        .replace("{{TITLE}}", page["title"])
        .replace("{{DESC}}", page["desc"])
        .replace("{{CANONICAL}}", page["canonical"])
        .replace("{{IMAGE}}", page["image"])
    )
    nav = "" if stage else _read(PARTIALS_DIR / "nav.html")
    for key in _SUBNAV_KEYS:
        nav = nav.replace("{{" + key + "}}", "active" if key == page["active"] else "")
    body = in_season(_read(PAGES_DIR / f"{page['slug']}.html"))
    foot = _read(PARTIALS_DIR / ("footer_stage.html" if stage else "footer.html"))
    return head + nav + body + foot


def enso_exclude_json() -> str:
    """The systems the ENSO forecasts page leaves out (enso_models.py, the one list), as a JSON array."""
    sys.path.insert(0, str(HERE))
    from enso_models import EXCLUDE_SYSTEMS

    return json.dumps(sorted(EXCLUDE_SYSTEMS), ensure_ascii=False)


def render_all(tokens: dict, site_root, only: list[str] | None = None) -> list[Path]:
    """Write every page into site_root, stamping the non-TAO data tokens.
    `tokens` = {cache, sst_day, roni_month} (pre-formatted strings).

    `only` restricts the write to those page slugs. A page not in the list keeps
    whatever is on disk: the ENSO pages carry data tokens (the SST day, the RONI
    month) that only the workflow that just fetched them knows, so a local render
    of one page must not rewrite the others with stale ones.
    Missing tokens are left as they are, for the same reason."""
    site_root = Path(site_root)
    written = []
    for page in PAGES:
        if only is not None and page["slug"] not in only:
            continue
        html = assemble(page)
        for tok, key in (("__CACHE__", "cache"), ("__SST_DAY__", "sst_day"), ("__RONI_MONTH__", "roni_month")):
            if key in tokens:
                html = html.replace(tok, tokens[key])
        html = html.replace("__ENSO_EXCLUDE__", enso_exclude_json())
        out = site_root / page["out"]
        out.write_text(html, encoding="utf-8")
        written.append(out)
    # The partials carry no site header/footer: re-apply the shared chrome to what was just
    # written, or an SST run silently reverts these pages to the raw partial nav.
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "apply_chrome", Path(__file__).resolve().parents[1] / "site" / "apply_chrome.py"
        )
        chrome = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chrome)
        rel = {str(p.relative_to(site_root)) for p in written}
        n = chrome.stamp_all(only=rel, quiet=True)
        print(f"  site chrome re-applied to {n} regenerated page(s)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: site chrome not applied ({e!r}) — run scripts/site/apply_chrome.py", flush=True)
    return written


def stamp_tao(site_root, tao_text: str) -> None:
    """Fill __TAO_DAY__ (e.g. 'TAO 2026-06-03') across the pages that show it."""
    site_root = Path(site_root)
    for page in PAGES:
        out = site_root / page["out"]
        if out.exists():
            txt = out.read_text(encoding="utf-8")
            if "__TAO_DAY__" in txt:
                out.write_text(txt.replace("__TAO_DAY__", tao_text), encoding="utf-8")
