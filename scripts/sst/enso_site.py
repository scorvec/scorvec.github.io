#!/usr/bin/env python3
"""Static assembler for the El Niño Monitor pages.

The monitor is one Overview homepage (sst.html) plus four themed subpages, all
sharing chrome (head / nav / sub-nav / footer) from partials/ and per-page card
fragments from pages/. Both HTML-writing scripts use this module so a single
edit to the chrome (or the page list) propagates everywhere:

  - sst-roni.py  stamp_html()  -> render_all()  (stamps __CACHE__/__SST_DAY__/__RONI_MONTH__)
  - sst_subsurface.py          -> stamp_tao()   (fills __TAO_DAY__, known later in the run)

Run order matters: sst-roni renders all pages first; sst_subsurface fills the
TAO date afterwards (it isn't known at stamp time). Both leave the OTHER data
tokens already resolved.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTIALS_DIR = HERE / "partials"
PAGES_DIR = HERE / "pages"

# sub-nav active-class placeholders, one per page (see partials/nav.html)
_SUBNAV_KEYS = ["A_OVERVIEW", "A_SUBSURFACE", "A_ANALOGS", "A_FORECASTS", "A_ATMOSPHERE"]

PAGES = [
    dict(slug="overview", out="sst.html", active="A_OVERVIEW", chrome="dark",
         title="El Ni&ntilde;o Monitor &mdash; Daily ONI, RONI &amp; Ni&ntilde;o Indices &middot; Shawn Corvec",
         desc="Daily estimates of ONI and RONI with interactive Niño-region SST index "
              "charts, high-resolution global and tropical Pacific anomaly maps from "
              "NOAA OISST v2.1, and ENSO subsurface, analog, forecast and atmospheric "
              "diagnostics.",
         canonical="https://scorvec.com/sst.html"),
    dict(slug="subsurface", out="enso-subsurface.html", active="A_SUBSURFACE",
         title="Subsurface Temperature &middot; El Ni&ntilde;o Monitor",
         desc="Equatorial Pacific depth–longitude subsurface temperature cross-sections "
              "(NOAA/PMEL TAO/TRITON), raw and with the 1991–2020 climate trend removed.",
         canonical="https://scorvec.com/enso-subsurface.html"),
    dict(slug="analogs", out="enso-analogs.html", active="A_ANALOGS",
         title="El Ni&ntilde;o Analog Comparisons &middot; El Ni&ntilde;o Monitor",
         desc="The current event vs the 1997, 2015 and 2023 El Niños at matching phase — "
              "surface, subsurface and 850 hPa low-level wind.",
         canonical="https://scorvec.com/enso-analogs.html"),
    dict(slug="forecasts", out="enso-forecasts.html", active="A_FORECASTS",
         title="ENSO Forecasts &middot; El Ni&ntilde;o Monitor",
         desc="Multi-model C3S seasonal Niño-3.4 outlook — the Copernicus multi-system "
              "seasonal ensemble (ECMWF, UK Met Office, Météo-France, DWD, NCEP, "
              "ECCC/CanSIPS, BOM), shown as both traditional ONI and RONI.",
         canonical="https://scorvec.com/enso-forecasts.html"),
    dict(slug="atmosphere", out="enso-atmosphere.html", active="A_ATMOSPHERE",
         title="Atmospheric Response &middot; El Ni&ntilde;o Monitor",
         desc="Observed and forecast equatorial winds, atmospheric angular momentum and "
              "its torque budget, and the meridional overturning circulation.",
         canonical="https://scorvec.com/enso-atmosphere.html"),
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def assemble(page: dict) -> str:
    """Full HTML for one page (still containing the __…__ data tokens).

    chrome="dark" swaps in the Gatun-style dark head (the overview); the
    themed subpages keep the light chrome. The nav partial is shared — its
    classes are styled by whichever head is in play.
    """
    head_file = "head_dark.html" if page.get("chrome") == "dark" else "head.html"
    head = (_read(PARTIALS_DIR / head_file)
            .replace("{{TITLE}}", page["title"])
            .replace("{{DESC}}", page["desc"])
            .replace("{{CANONICAL}}", page["canonical"]))
    nav = _read(PARTIALS_DIR / "nav.html")
    for key in _SUBNAV_KEYS:
        nav = nav.replace("{{%s}}" % key, "active" if key == page["active"] else "")
    body = _read(PAGES_DIR / f"{page['slug']}.html")
    foot = _read(PARTIALS_DIR / "footer.html")
    return head + nav + body + foot


def render_all(tokens: dict, site_root) -> list[Path]:
    """Write every page into site_root, stamping the non-TAO data tokens.
    `tokens` = {cache, sst_day, roni_month} (pre-formatted strings)."""
    site_root = Path(site_root)
    written = []
    for page in PAGES:
        html = (assemble(page)
                .replace("__CACHE__", tokens["cache"])
                .replace("__SST_DAY__", tokens["sst_day"])
                .replace("__RONI_MONTH__", tokens["roni_month"]))
        out = site_root / page["out"]
        out.write_text(html, encoding="utf-8")
        written.append(out)
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
