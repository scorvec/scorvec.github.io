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

Stage pages flagged `index` also get an "Every plot on this page" block (plot_index).
A stage page edited by hand rather than built here refreshes its own block with

    python scripts/sst/enso_site.py --index ar.html
"""

from __future__ import annotations

import html as htmllib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTIALS_DIR = HERE / "partials"
PAGES_DIR = HERE / "pages"

# sub-nav active-class placeholders, one per page (see partials/nav.html)
_SUBNAV_KEYS = ["A_OVERVIEW", "A_SUBSURFACE", "A_FORECASTS", "A_ATMOSPHERE"]

PAGES = [
    dict(
        slug="enso",
        out="enso.html",
        active="A_OVERVIEW",
        layout="stage",
        index=True,
        title="El Ni&ntilde;o Monitor &mdash; Daily ONI, RONI &amp; Ni&ntilde;o Indices &middot; Shawn Corvec",
        desc="Daily ONI and RONI estimates, interactive Niño-index charts, OISST and MUR SST anomaly maps, the SOI, "
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
        index=True,
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
        index=True,
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
        index=True,
        title="Tropical Waves &mdash; Kelvin and Rossby Wave Trackers &middot; Shawn Corvec",
        desc="Convectively coupled equatorial waves: Kelvin, Rossby and MJO filtered OLR Hovmöllers, wave trackers "
        "with the AIFS-ENS forecast, and ocean Kelvin waves.",
        canonical="https://scorvec.com/tropical-waves.html",
        image="https://scorvec.com/assets/sst/wave_tracker.webp",
    ),
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ── "Every plot on this page" ───────────────────────────────────────────────────────────────────────────────────────
# A stage page keeps each product's explanation in an inert <template> and addresses its products by #fragment, so a
# search engine or a reader without JavaScript sees almost none of the page's text. plot_index() writes the stage's
# own product list into the page: a heading per rail group, each product linked to its fragment with the first
# paragraph of its About text (or its caption when it has none), in a collapsed <details> under the stage.
INDEX_START, INDEX_END = "<!-- plots:start -->", "<!-- plots:end -->"
_EXPLAINER = re.compile(r'\s*<a href="(/topics/[^"]+)">The explainer</a>')
_JS_U = re.compile(r"\\u\{([0-9a-fA-F]+)\}")  # ES2015 code-point escapes, which JSON does not have


def _from_js(src: str):
    """A JavaScript string or array-of-strings literal, read as JSON."""
    return json.loads(_JS_U.sub(lambda m: chr(int(m.group(1), 16)), src))


def _close(text: str, i: int) -> int:
    """Index just past the bracket that closes the one at text[i], skipping string literals and comments."""
    pairs, stack, q = {"[": "]", "{": "}"}, [], None
    while i < len(text):
        c = text[i]
        if not q and text.startswith("//", i):
            i = text.find("\n", i)
            if i < 0:
                break
        elif not q and text.startswith("/*", i):
            i = text.index("*/", i) + 1
        elif q:
            if c == "\\":
                i += 1
            elif c == q:
                q = None
        elif c in "\"'`":
            q = c
        elif c in pairs:
            stack.append(pairs[c])
        elif stack and c == stack[-1]:
            stack.pop()
            if not stack:
                return i + 1
        i += 1
    raise ValueError("unbalanced brackets in the stage spec")


def _stage_spec(html_text: str) -> tuple[list, dict]:
    """(groups, products) from a page's window.STAGE: groups = [(label, [[id, name, sub], ...]), ...] and
    products = {id: (about template ids, caption)} as far as they are literals."""
    s = html_text.index("window.STAGE = {")
    spec = html_text[s : _close(html_text, html_text.index("{", s))]
    g0 = spec.index("[", spec.index("groups:"))
    groups_src = spec[g0 : _close(spec, g0)]
    groups = []
    for m in re.finditer(r'label:\s*"((?:[^"\\]|\\.)*)"', groups_src):
        it = groups_src.index("[", groups_src.index("items:", m.end()))
        groups.append((_from_js(f'"{m.group(1)}"'), _from_js(groups_src[it : _close(groups_src, it)])))
    p0 = spec.index("products:")
    ids = [p[0] for _, items in groups for p in items]
    starts = sorted(
        (m.start(), pid) for pid in ids if (m := re.compile(rf"\b{re.escape(pid)}\s*:\s*\{{").search(spec, p0))
    )
    products = {}
    for k, (pos, pid) in enumerate(starts):
        chunk = spec[pos : starts[k + 1][0] if k + 1 < len(starts) else len(spec)]
        abouts = list(dict.fromkeys(re.findall(r'"(about-[\w-]+)"', chunk)))
        cap = re.search(r'cap:\s*function\s*\([^)]*\)\s*\{\s*return\s+"((?:[^"\\]|\\.)*)"\s*;?\s*\}', chunk)
        products[pid] = (abouts, _from_js(f'"{cap.group(1)}"') if cap else "")
    return groups, products


def _first_paragraph(tpl: str) -> tuple[str, str]:
    """(first explanatory paragraph's inner HTML, explainer href) of an About template."""
    explainer = ""
    for m in re.finditer(r"<p\b([^>]*)>(.*?)</p>", tpl, re.S):
        attrs, inner = m.group(1), m.group(2).strip()
        if "chart-sources" in attrs:
            continue
        x = _EXPLAINER.match(inner)
        if x:
            explainer = x.group(1)
            continue
        inner = re.sub(r'\s+id="[^"]*"', "", re.sub(r"\s+", " ", inner))
        if inner.endswith(":"):  # the sentence leads into an equation or a list that is not carried over
            k = inner.rfind(". ")
            inner = inner[: k + 1] if k > 0 else inner[:-1] + "."
        return inner, explainer
    return "", explainer


def plot_index(html_text: str) -> str:
    """The "Every plot on this page" block for a stage page (its STAGE spec and About templates)."""
    groups, products = _stage_spec(html_text)
    tpls = dict(re.findall(r'<template id="([^"]+)">(.*?)</template>', html_text, re.S))

    def esc(t: str) -> str:
        return htmllib.escape(htmllib.unescape(t), quote=False)

    seen, out, n = set(), [], 0
    for label, items in groups:
        out.append(f"<h2>{esc(label)}</h2>")
        for item in items:
            pid, name, sub = [*item, "", ""][:3]
            abouts, cap = products.get(pid, ([], ""))
            paras, explainer = [], ""
            for a in abouts:
                if a in seen or a not in tpls:
                    continue
                seen.add(a)
                text, ex = _first_paragraph(tpls[a])
                explainer = explainer or ex
                if text:
                    paras.append(text)
            if not paras and cap:
                paras.append(esc(cap))
            head = f'<a href="#{pid}"><b>{esc(name)}</b></a>' + (f" &middot; {esc(sub)}" if sub else "")
            if explainer:
                head += f' &middot; <a href="{explainer}">explainer</a>'
            out.append(f"<p>{head}" + (f"<br>\n{paras[0]}" if paras else "") + "</p>")
            out += [f"<p>{p}</p>" for p in paras[1:]]
            n += 1
    return (
        f"{INDEX_START}\n"
        f'<details class="about" id="every-plot">\n<summary>Every plot on this page ({n})</summary>\n'
        + "\n".join(out)
        + "\n</details>\n"
        # a link in the list picks that plot on the stage: bring the stage into view
        + '<script>document.getElementById("every-plot").addEventListener("click", function (e) {\n'
        + '  var a = e.target.closest(\'a[href^="#"]\'), st = document.querySelector(".ss-layout");\n'
        + "  if (a && st) setTimeout(function () {\n"
        + '    st.style.scrollMarginTop = "5rem"; st.scrollIntoView({ block: "start" });\n'
        + "  }, 0);\n"
        + "});</script>\n"
        f"{INDEX_END}"
    )


def with_plot_index(html_text: str) -> str:
    """The page with its "Every plot" block (re)written under the stage's hint line."""
    html_text = re.sub(rf"\n?{re.escape(INDEX_START)}.*?{re.escape(INDEX_END)}", "", html_text, flags=re.S)
    block = plot_index(html_text)
    m = re.search(r'<p class="hint">.*?</p>', html_text, re.S)
    if not m:
        raise ValueError("no stage hint line to put the plot index under")
    return html_text[: m.end()] + "\n" + block + html_text[m.end() :]


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
    body = _read(PAGES_DIR / f"{page['slug']}.html")
    if page.get("index"):
        body = with_plot_index(body)
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


def main(argv: list[str]) -> int:
    """`--index PAGE...`: rewrite the "Every plot" block of stage pages that are not built from these partials."""
    if len(argv) < 2 or argv[0] != "--index":
        print("usage: enso_site.py --index PAGE.html [...]", file=sys.stderr)
        return 2
    for name in argv[1:]:
        p = Path(name)
        txt = p.read_text(encoding="utf-8")
        new = with_plot_index(txt)
        if new != txt:
            p.write_text(new, encoding="utf-8")
            print(f"  plot index written: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
