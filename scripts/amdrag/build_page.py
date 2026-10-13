#!/usr/bin/env python3
"""Write scorvec.com/model-drag-intercomparison.html from scripts/amdrag/data/results.json (study.py) and the figures
in studies/model-drag-intercomparison/ (figures.py). Every number on the page is read from results.json; the wording
of each finding follows its significance (a result that is not significant after the false-discovery-rate control is
reported as "n.s." or left out of the summary). Re-run daily as the sample grows:

    python scripts/amdrag/study.py && python scripts/amdrag/figures.py --site . && python scripts/amdrag/build_page.py --site .
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                     # noqa: E402

SLUG = "model-drag-intercomparison"
# UNLINKED (user, 2026-10-13): published for review only - noindex, and no menu / homepage / sitemap / catalogue entry or
# cross-link anywhere. To make it public: (1) apply_chrome.py: a PRODUCTS row under "Studies" and the href in HOME's
# "Studies and verification" list, then run apply_chrome.py (menus on every page + the homepage tile); (2) sitemap.xml
# entry; (3) one cross-link sentence each in torque-enso.html (+ scripts/mjo/src/torque_enso_page.py) and
# aifs-aam-budget.html; (4) delete the robots meta in wrap() below and rebuild; (5) build_catalog.py PAGES entry
# (thumb studies/model-drag-intercomparison/aam_drift_own.webp).
IMG = f"/studies/{SLUG}"
URL = f"https://scorvec.com/{SLUG}.html"
TITLE = "Model drag intercomparison"
H1 = "Conservation and drag in the US, ECMWF and Canadian global models"
DESC = ("Eight global forecast systems through one angular-momentum budget: do they conserve angular momentum and mass, "
        "how much drag do they apply, and where do their low-level winds drift? GFS, GEFS, AIGFS, IFS-ENS, AIFS, AIFS-ENS, "
        "GDPS and GEPS against ERA5 and the 2012 WGNE Drag Project. Significant results only; the sample grows daily.")
ORDER = ["gfs", "gefs", "aigfs", "ifs-ens", "aifs-single", "aifs-ens", "gdps", "geps"]
REG = {"ORO": "rough terrain and 500 km downstream", "OCX": "ocean 30–65°", "OCO": "other ocean", "LND": "other land"}
RANGE = {"HIM": "Himalaya/Tibet", "ROC": "Rockies", "AND": "Andes", "GRL": "Greenland", "ANT": "Antarctica",
         "ALP": "Alps/Europe"}


def num(x, d=2, sign=True):
    if x is None:
        return "–"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


def E(b, d=2, unit=""):
    """'+0.12 [+0.05, +0.19]' when significant, '+0.12 (n.s.)' otherwise, '–' when absent."""
    if not b or b.get("mean") is None:
        return "–"
    u = f" {unit}" if unit else ""
    if b.get("lo") is None:
        return f"{num(b['mean'], d)}{u} (n = {b.get('n', 0)}, too few to test)"
    if b.get("sig"):
        return f"{num(b['mean'], d)}{u} [{num(b['lo'], d)}, {num(b['hi'], d)}]"
    return f"<span class='ns'>{num(b['mean'], d)}{u} n.s.</span>"


def td(b, d=2):
    if not b or b.get("mean") is None:
        return "<td></td>"
    if b.get("lo") is None:
        return f"<td class='ns'>{num(b['mean'], d)} (n={b.get('n', 0)})</td>"
    if b.get("sig"):
        return f"<td>{num(b['mean'], d)} <span class='sig'>[{num(b['lo'], d)}, {num(b['hi'], d)}]</span></td>"
    return f"<td class='ns'>{num(b['mean'], d)} n.s.</td>"


def sig(b):
    return bool(b and b.get("sig") and b.get("mean") is not None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=".")
    a = ap.parse_args()
    site = Path(a.site)
    R = json.loads((C.DATA / "results.json").read_text())
    Mo = R["models"]
    keys = [k for k in ORDER if k in Mo]
    L = {k: Mo[k]["label"] for k in keys}
    today = dt.date.today()
    made = R.get("made", "")[:10]

    FIGS = ["budget", "mass", "drift", "drifte", "levels", "torques", "wgne", "winds", "mountain", "oro"]
    FN = {k: i + 1 for i, k in enumerate(FIGS)}

    def fig(key, f, alt, cap):
        n = FN[key]
        return (f"<figure id='fig{n}'><a href='{IMG}/{f}' class='zoom'><img src='{IMG}/{f}' alt=\"{alt}\" loading='lazy'>"
                f"</a><figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>")

    # ---------------- sample table
    rows = []
    GRID = {"gfs": "C768 (≈13 km); torque fields on its 3072×1536 Gaussian physics grid, winds 0.25°",
            "gefs": "C384 (≈25 km); 0.5° output", "aigfs": "0.25° (an AI model initialised from GDAS)",
            "ifs-ens": "TCo1279 (≈9 km); 0.25° output", "aifs-single": "N320 (≈31 km); 0.25° output",
            "aifs-ens": "N320 (≈31 km); 0.25° output", "gdps": "Yin-Yang 15 km; 0.15° output",
            "geps": "Yin-Yang 25 km; 0.5° output"}
    LEV = {"gfs": "41 (0.01–1000)", "gefs": "31 (1–1000)", "aigfs": "13 (50–1000)", "ifs-ens": "14 (10–1000)",
           "aifs-single": "14 (10–1000)", "aifs-ens": "14 (10–1000)", "gdps": "33 (1–1015)", "geps": "12 (10–1000)"}
    STRESS = {"gfs": "turbulent (UFLX) + orographic GWD and blocking (U-GWD)", "gefs": "turbulent + GWD/blocking",
              "ifs-ens": "turbulent incl. orographic form drag (ewss); no GWD/blocking stress",
              "aigfs": "none (AI)", "aifs-single": "none (AI)", "aifs-ens": "none (AI)", "gdps": "none published",
              "geps": "none published"}
    MEM = {"gfs": "1", "gefs": "control + 4", "aigfs": "1", "ifs-ens": "control (the 9 km deterministic run, to day 10) + 4",
           "aifs-single": "1", "aifs-ens": "control + 4", "gdps": "1", "geps": "control + 4"}
    for k in keys:
        M = Mo[k]
        rows.append(f"<tr><th scope='row'>{L[k]}</th><td>{M['centre']}</td><td style='white-space:normal'>{GRID[k]}</td>"
                    f"<td>{LEV[k]}</td><td>{MEM[k]}</td><td>{M['n_inits']}</td><td>{M['first'][5:]} – {M['last'][5:]}</td>"
                    f"<td style='white-space:normal'>{STRESS[k]}</td></tr>")
    t_models = ("<div class='tw'><table class='res'><thead><tr><th>model</th><th>centre</th><th>resolution, grid used</th>"
                "<th>levels (hPa)</th><th>members</th><th>00Z inits</th><th>first – last</th><th>own surface stress</th>"
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    body = PAGE_BODY(R, Mo, keys, L, fig, t_models, made)
    page = wrap(site, body)
    out = site / f"{SLUG}.html"
    old = out.read_text() if out.exists() else ""
    if "<!-- sh:start -->" in old:
        page = page.replace("<body>\n", "<body>\n" + old.split("<body>\n", 1)[1].split("<!-- sh:end -->", 1)[0]
                            + "<!-- sh:end -->\n", 1)
    out.write_text(page)
    print("wrote", out)
    sys.path.insert(0, str(HERE.parents[0] / "site"))
    import apply_chrome                                                  # noqa: E402  (header, menus, footer)
    if site.resolve() == apply_chrome.REPO.resolve():
        apply_chrome.stamp_all(only={f"{SLUG}.html"})
    else:
        print("site is not the repo: chrome not stamped (preview build)")
    return 0


def wrap(site: Path, body: str) -> str:
    ld = {"@context": "https://schema.org", "@type": "ScholarlyArticle", "headline": H1, "name": TITLE, "description": DESC,
          "url": URL, "inLanguage": "en", "datePublished": "2026-10-13", "dateModified": dt.date.today().isoformat(),
          "author": {"@id": "https://scorvec.com/#person"}, "image": [f"https://scorvec.com{IMG}/aam_drift_own.webp"],
          "keywords": "atmospheric angular momentum, surface drag, mountain torque, friction torque, gravity-wave drag, "
                      "conservation, mass conservation, GFS, GEFS, AIGFS, IFS, AIFS, GDPS, GEPS, GEM, ERA5, WGNE drag project"}
    aam = site / "aifs-aam-budget.html"
    style = aam.read_text().split("<style>", 1)[1].split("</style>", 1)[0]
    style += "\n  .study code { font-size: 0.9em; }\n  table.res td.l { text-align: left; white-space: normal; }\n"
    lb = aam.read_text().split("</main>", 1)[1].split("<!-- sf:start -->", 1)[0].strip()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{TITLE} · Shawn Corvec</title>
<meta name="robots" content="noindex">
<meta name="description" content="{DESC}">
<link rel="canonical" href="{URL}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<meta name="author" content="Shawn Corvec">
<meta property="og:type" content="article">
<meta property="og:title" content="{H1}">
<meta property="og:description" content="{DESC}">
<meta property="og:url" content="{URL}">
<meta property="og:image" content="https://scorvec.com{IMG}/aam_drift_own.webp">
<meta property="og:site_name" content="Shawn Corvec">
<meta name="twitter:card" content="summary_large_image">
<script type="application/ld+json">
{json.dumps(ld, indent=1, ensure_ascii=False)}
</script>
<script data-goatcounter="https://scorvec.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
<link rel="stylesheet" href="/assets/site.css">
<script src="/assets/site.js" defer></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],throwOnError:false}});"></script>
<style>{style}</style>
</head>
<body>
<main>
{body}
</main>
{lb}
</body>
</html>
"""


def PAGE_BODY(R, Mo, keys, L, fig, t_models, made):
    import page_text                                                     # noqa: E402  (imports helpers from here)
    return page_text.body(R, Mo, keys, L, fig, t_models, made)


if __name__ == "__main__":
    sys.exit(main())
