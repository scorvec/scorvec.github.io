#!/usr/bin/env python3
"""rain-bias.html - the Americas model-rain bias field (build_bias.py + render_bias.py) as one page (2026-10-13).

UNLINKED until the user has reviewed it: noindex, and no menu / homepage / sitemap / catalogue entry or cross-link. To make
it public: a PRODUCTS row in scripts/site/apply_chrome.py and the homepage list, sitemap.xml, build_catalog.py, and drop
the robots meta below.

    python scripts/precip_bias/build_page.py --json assets/precip_bias/bias_summary.json --site .
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bias import LEADS, MODELS, REGIONS          # noqa: E402

SLUG = "rain-bias"
TITLE = "Model rain bias, Americas"
H1 = "Where the ECMWF ensembles rain too much or too little: the Americas"
DESC = ("IFS-ENS and AIFS-ENS rainfall against gauge-adjusted IMERG over North and South America, by lead time (day 1 to week 2) "
        "and season, from the operational archive. Significant cells only; updated as the archive grows.")
SEAS = [("all", "All seasons"), ("DJF", "Dec-Feb"), ("MAM", "Mar-May"), ("JJA", "Jun-Aug"), ("SON", "Sep-Nov")]
LEAD_ORDER = ["day1", "d2_3", "d4_7", "d8_10", "d11_14", "wk1", "wk2"]


def f2(x, d=2):
    return f"{x:+.{d}f}".replace("-", "−")


def region_table(S, comp, lk):
    """Rows of significant regional biases; 'n.s.' where the interval spans zero."""
    C = S["comparisons"].get(comp, {})
    ms = [m for m in MODELS if m in C]
    if not ms:
        return ""
    head = "".join(f"<th>{MODELS[m]}</th>" for m in ms)
    rows = []
    for reg in REGIONS:
        cells = []
        for m in ms:
            r = C[m]["leads"].get(lk, {}).get("all", {}).get("regions", {}).get(reg)
            if not r:
                cells.append("<td>-</td>"); continue
            if r["sig"]:
                rel = f" ({100 * r['rel']:+.0f} %)".replace("-", "−") if r.get("rel") is not None else ""
                cells.append(f"<td>{f2(r['bias'])}{rel}</td>")
            else:
                cells.append('<td class="ns">n.s.</td>')
        obs = C[ms[0]]["leads"].get(lk, {}).get("all", {}).get("regions", {}).get(reg, {}).get("obs")
        rows.append(f"<tr><td class=\"l\">{reg}</td><td>{obs:.1f}</td>{''.join(cells)}</tr>" if obs is not None else "")
    return (f'<div class="tw"><table class="res"><thead><tr><th class="l">Region</th><th>Observed (mm/day)</th>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def body(S):
    full = S["comparisons"].get("ifs_full", {}).get("ifs")
    com = S["comparisons"].get("common", {})
    stab = S.get("ifs_week1_year_to_year_pattern_r", {})
    stab_txt = ", ".join(f"{k}: r = {v:.2f}" for k, v in stab.items())
    figs = set(S.get("figures", []))
    opts = lambda pairs, sel: "".join(f'<option value="{v}"{" selected" if v == sel else ""}>{html.escape(t)}</option>' for v, t in pairs)
    lead_opts = [(k, LEADS[k][2]) for k in LEAD_ORDER]
    comp_opts = [("ifs_full", f"IFS-ENS, whole record ({full['first']} to {full['last']})" if full else "IFS-ENS")]
    if com:
        c0 = com[next(iter(com))]
        comp_opts.append(("common", f"IFS-ENS vs AIFS-ENS, the same runs ({c0['first']} to {c0['last']})"))
    wk1 = full["leads"].get("wk1", {}).get("all", {}) if full else {}
    first_fig = "ifs_full_wk1_all_abs.webp"
    tf = S.get("imerg_final") or ["-", "-"]
    return f"""
<article class="study">
<p class="kicker">Verification</p>
<h1>{H1}</h1>
<p class="lede">How much more or less rain the ECMWF ensembles forecast than fell, cell by cell over North and South America, by lead time
and season. ECMWF publishes no reforecast with its open data, so the models' own rain climate is built here from the operational
archive: every 00Z run, six members, since March 2024 for IFS-ENS and July 2025 for AIFS-ENS.</p>

<section>
<h2>Maps</h2>
<div class="pick">
<label>Comparison <select id="rb-comp">{opts(comp_opts, "ifs_full")}</select></label>
<label>Lead <select id="rb-lead">{opts(lead_opts, "wk1")}</select></label>
<label>Season <select id="rb-season">{opts(SEAS, "all")}</select></label>
<label>Units <select id="rb-kind">{opts([("abs", "mm/day"), ("rel", "percent of observed")], "abs")}</select></label>
</div>
<figure><img id="rb-img" src="/assets/precip_bias/{first_fig}" alt="Model minus observed rain map" loading="lazy" width="1680" height="1600">
<figcaption id="rb-cap">Brown: the model is too dry; green: too wet. Grey: not significant. Percent maps leave out cells drier than
0.5 mm/day.</figcaption></figure>
</section>

<section>
<h2>By lead time</h2>
<figure><img src="/assets/precip_bias/leads_regions.webp" alt="Regional rain bias by lead for IFS-ENS and AIFS-ENS" loading="lazy">
<figcaption>Area means in the dashed boxes on the maps, both models on the same runs.</figcaption></figure>
<p>Week 1 (days 1-7), all seasons. Values are mm/day (percent of observed); "n.s." = the 95 % interval includes zero.</p>
<h3>IFS-ENS, whole record</h3>
{region_table(S, "ifs_full", "wk1")}
<h3>IFS-ENS and AIFS-ENS, the same runs</h3>
{region_table(S, "common", "wk1")}
</section>

<section>
<h2>How it is measured</h2>
<ul>
<li><strong>Forecast</strong>: the mean of members 1-6 of each 00Z run, accumulated over the lead band, as 1° box means.</li>
<li><strong>Truth</strong>: IMERG V07 Late, the satellite record that keeps up with the forecasts, corrected by its own seasonal
difference from IMERG Final, which carries the monthly gauge adjustment over land ({tf[0]} to {tf[1]}). The size of that
correction is on the <a href="#truth">last map</a>.</li>
<li><strong>Significance</strong>: runs a few days apart share their weather, so each mean's uncertainty comes from resampling
whole 21-day blocks of start dates; a cell is shaded only if it passes a false-discovery-rate test at q = 0.10 across the map.
{f"For IFS-ENS week 1, {100 * wk1['frac_sig']:.0f} % of cells pass." if wk1 else ""}</li>
<li><strong>Stability</strong>: the IFS-ENS week-1 bias pattern from March-October of one year against the same months of another:
{stab_txt or "not yet two years"}. Model upgrades inside the record (IFS cycle 49r1, November 2024) would show up as a drop here.</li>
<li><strong>Domain</strong>: 60°S-60°N, 170°W-30°W. IMERG is unreliable over snow and ice and poleward of 60°, so Alaska and
northern Canada are left out.</li>
</ul>
</section>

<section id="truth">
<h2>The truth adjustment</h2>
<figure><img src="/assets/precip_bias/truth_JJA.webp" id="rb-truth" alt="IMERG Late minus Final" loading="lazy">
<figcaption><label>Season <select id="rb-tseason">{opts([s for s in SEAS if s[0] != "all"], "JJA")}</select></label>
IMERG Late minus Final. Where it is large, the gauge adjustment matters as much as the model.</figcaption></figure>
</section>
<p class="byline">Updated {S['generated']} UTC.</p>
</article>
<script>
(function () {{
  var figs = new Set({json.dumps(sorted(figs))});
  function $(id) {{ return document.getElementById(id); }}
  function upd() {{
    var f = $("rb-comp").value + "_" + $("rb-lead").value + "_" + $("rb-season").value + "_" + $("rb-kind").value + ".webp";
    var img = $("rb-img");
    if (figs.has(f)) {{ img.src = "/assets/precip_bias/" + f; img.style.opacity = 1; $("rb-cap").dataset.missing = ""; }}
    else {{ img.style.opacity = 0.25; $("rb-cap").dataset.missing = "1"; }}
  }}
  ["rb-comp", "rb-lead", "rb-season", "rb-kind"].forEach(function (k) {{ $(k).addEventListener("change", upd); }});
  $("rb-tseason").addEventListener("change", function () {{ $("rb-truth").src = "/assets/precip_bias/truth_" + this.value + ".webp"; }});
}})();
</script>
"""


def wrap(site: Path, inner: str) -> str:
    aam = site / "aifs-aam-budget.html"
    style = aam.read_text().split("<style>", 1)[1].split("</style>", 1)[0]
    style += """
  .study .pick { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; margin: 0 0 0.6rem; }
  .study .pick label, .study figcaption label { font-size: 0.9em; }
  .study .pick select, .study figcaption select { font: inherit; font-size: 0.9em; margin-left: 0.3rem; }
  .study figure img#rb-img, .study figure img { width: 100%; }
  #rb-cap[data-missing="1"]::before { content: "Not enough runs for this combination yet. "; font-weight: 600; }
  table.res td.l, table.res th.l { text-align: left; white-space: normal; }
  table.res td.ns { color: var(--c-muted, #6f6b64); }
"""
    lb = aam.read_text().split("</main>", 1)[1].split("<!-- sf:start -->", 1)[0].strip()
    url = f"https://scorvec.com/{SLUG}.html"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{TITLE} · Shawn Corvec</title>
<meta name="robots" content="noindex">
<meta name="description" content="{DESC}">
<link rel="canonical" href="{url}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<meta name="author" content="Shawn Corvec">
<script data-goatcounter="https://scorvec.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
<link rel="stylesheet" href="/assets/site.css">
<script src="/assets/site.js" defer></script>
<style>{style}</style>
</head>
<body>
<main>
{inner}
</main>
{lb}
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True); ap.add_argument("--site", default=".")
    a = ap.parse_args()
    site = Path(a.site); S = json.loads(Path(a.json).read_text())
    page = wrap(site, body(S))
    out = site / f"{SLUG}.html"
    old = out.read_text() if out.exists() else ""
    if "<!-- sh:start -->" in old:                                   # keep the stamped header until apply_chrome re-stamps
        page = page.replace("<body>\n", "<body>\n" + old.split("<body>\n", 1)[1].split("<!-- sh:end -->", 1)[0] + "<!-- sh:end -->\n", 1)
    out.write_text(page)
    print("wrote", out)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "site"))
    import apply_chrome                                              # noqa: E402
    if site.resolve() == apply_chrome.REPO.resolve():
        apply_chrome.stamp_all(only={f"{SLUG}.html"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
