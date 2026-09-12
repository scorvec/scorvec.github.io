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
_SUBNAV_KEYS = ["A_OVERVIEW", "A_SUBSURFACE", "A_FORECASTS", "A_ATMOSPHERE"]

PAGES = [
    dict(slug="enso", out="enso.html", active="A_OVERVIEW", layout="stage",
         title="El Ni&ntilde;o Monitor &mdash; Daily ONI, RONI &amp; Ni&ntilde;o Indices &middot; Shawn Corvec",
         desc="Daily estimates of ONI and RONI with interactive Niño-region SST index "
              "charts, high-resolution global and tropical Pacific anomaly maps from "
              "NOAA OISST v2.1, MUR 1 km SST, the SOI and MEI, and equatorial convection — "
              "one figure at a time.",
         canonical="https://scorvec.com/enso.html"),
    dict(slug="forecasts", out="enso-forecasts.html", active="A_FORECASTS",
         title="ENSO Forecasts &mdash; Interactive Multi-Model Outlook &middot; El Ni&ntilde;o Monitor",
         desc="Interactive C3S multi-model Niño-3.4 outlook: every ensemble member from "
              "seven centres, percentile fans, ONI vs RONI, and the forecast measured "
              "against every ENSO event since 1970.",
         canonical="https://scorvec.com/enso-forecasts.html"),
    dict(slug="circulation", out="circulation.html", active="A_ATMOSPHERE", layout="stage",
         title="Global Circulation and Jets &mdash; Wave Activity Flux, Angular Momentum, Walker Cell "
               "and the Stratosphere &middot; Shawn Corvec",
         desc="Daily global circulation diagnostics from ECMWF AIFS-ENS: Takaya-Nakamura wave "
              "activity flux, dynamic tropopause, atmospheric angular momentum and its torque "
              "budget, Hadley and Walker cells, subtropical and North Pacific jets, polar vortex "
              "and E-P flux, plus equatorial winds and the Southern Oscillation.",
         canonical="https://scorvec.com/circulation.html"),
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def assemble(page: dict) -> str:
    """Full HTML for one page (still containing the __…__ data tokens).

    One head for every page since 2026-08-28: the navy/cyan dark chrome that
    Overview and Forecasts used to ship was retired at the user's request, and its
    components were folded into partials/head.html. A `chrome` key on a page entry
    is now ignored.
    """
    stage = page.get("layout") == "stage"          # rail + stage pages (2026-09-07): outlook.css look, no legacy nav
    head = (_read(PARTIALS_DIR / ("head_stage.html" if stage else "head.html"))
            .replace("{{TITLE}}", page["title"])
            .replace("{{DESC}}", page["desc"])
            .replace("{{CANONICAL}}", page["canonical"]))
    nav = "" if stage else _read(PARTIALS_DIR / "nav.html")
    for key in _SUBNAV_KEYS:
        nav = nav.replace("{{%s}}" % key, "active" if key == page["active"] else "")
    body = _read(PAGES_DIR / f"{page['slug']}.html")
    foot = _read(PARTIALS_DIR / ("footer_stage.html" if stage else "footer.html"))
    return head + nav + body + foot


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
        out = site_root / page["out"]
        out.write_text(html, encoding="utf-8")
        written.append(out)
    # The partials carry no site header/footer: re-apply the shared chrome to what was just
    # written, or an SST run silently reverts these pages to the pre-2026-09-06 raw nav.
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("apply_chrome", Path(__file__).resolve().parents[1] / "site" / "apply_chrome.py")
        chrome = importlib.util.module_from_spec(spec); spec.loader.exec_module(chrome)
        rel = {str(p.relative_to(site_root)) for p in written}
        n = chrome.stamp_all(only=rel, quiet=True)
        print(f"  site chrome re-applied to {n} regenerated page(s)", flush=True)
    except Exception as e:                                             # noqa: BLE001
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
