"""
Regenerate /mjo.html from the archived RMM plots in assets/mjo/.

Run from the repo root (the GitHub Action does this after each forecast):
    python scripts/mjo/generate_page.py

Shows the most recent plot as the hero, the latest 00Z and 12Z, and a dated
archive grid. Styled to match the rest of scorvec.com.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_section import clean_section   # noqa: E402  the ENSO-removed RMM block (2026-09-27)

ASSETS = Path("assets/mjo")
OUT = Path("mjo.html")
MANIFEST = ASSETS / "rmm_manifest.json"
RE = re.compile(r"rmm_(\d{8})_(\d{2})z\.png$")

NAV = """<nav>
  <a href="index.html" class="nav-name">Shawn Corvec</a>
  <ul class="nav-links">
    <li><a href="index.html">Home</a></li>
    <li><a href="resume.html">Resume</a></li>
    <li><a href="research.html">Research</a></li>
    <li><a href="mjo.html" class="active">MJO</a></li>
  </ul>
</nav>"""


def discover():
    items = []
    for p in ASSETS.glob("rmm_*z.png"):
        m = RE.search(p.name)
        if not m:
            continue
        date, hh = m.group(1), m.group(2)
        items.append((date, hh, p))
    items.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return items


def label(date, hh):
    return f"{date[:4]}-{date[4:6]}-{date[6:8]} {hh}Z"


def write_manifest(items):
    """Animator manifest for the sst_anim viewer: successive RMM runs, oldest→newest
    (the slider then defaults to the latest). Frames live in assets/mjo/ → base=assets,
    region=mjo, so the viewer loads assets/mjo/<file>."""
    frames = [{"idx": i, "file": p.name,
               "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}", "label": label(d, h)}
              for i, (d, h, p) in enumerate(reversed(items))]
    manifest = {"ver": int(datetime.now(timezone.utc).timestamp()), "days": len(frames),
                "regions": {"mjo": {"label": "AIFS-ENS RMM — successive forecast runs",
                                    "n_frames": len(frames), "frames": frames}}}
    MANIFEST.write_text(json.dumps(manifest))


IMPACTS_REF = Path("scripts/mjo/data/reference/mjo_impacts_site.json")


def impacts_section():
    """'MJO impacts by phase and season' (2026-09-27): CMIP6 and observed composites drawn by
    scripts/mjo/src/mjo_impacts_render.py (mjo-impacts.yml) from the committed reference. Empty until it exists.
    CMIP6 leads (user, 2026-09-27: "the MJO composites should come from the CMIP6 runs"); the observed record is the
    comparison, and the "compare" view stacks the two all-phase strips of the same field / month / lag."""
    if not IMPACTS_REF.exists():
        return ""
    m = json.loads(IMPACTS_REF.read_text())
    scr = m.get("screen", [])
    passed = [r for r in scr if r.get("pass") and r["model"] in m.get("models", [])]
    failed = [r for r in scr if not r.get("pass") and not r["model"].startswith("OBS")]
    fam = [r for r in scr if r.get("pass") and r["model"] not in m.get("models", []) and not r["model"].startswith("OBS")]
    obs = next((r for r in scr if r["model"].startswith("OBS")), {})
    def row(r):
        return f"{r['model']} (E/W {r['ew']:.1f}, r {r['prop_r']:.2f})"
    v = m.get("validation", {})
    val = ""
    if v:
        e, u = v.get("era5_windonly_vs_bom", {}), v.get("u250_for_u200", {})
        val = (f" Checks: the same machinery on ERA5 winds reproduces BoM&rsquo;s RMM1/RMM2 at r&nbsp;{e.get('r_rmm1', 0):.2f}/"
               f"{e.get('r_rmm2', 0):.2f} (wind-only, {e.get('years', '')}); substituting 250 for 200&nbsp;hPa keeps the "
               f"phase on {100 * u.get('same_phase', 0):.0f}&nbsp;% of active days (r&nbsp;{u.get('r_rmm1', 0):.2f}/{u.get('r_rmm2', 0):.2f}).")
    ens = ""
    ec = m.get("ensocheck", {}).get("obs", {}).get("all", {})
    if ec.get("strong El Nino") and ec.get("neutral"):
        se, ne = ec["strong El Nino"], ec["neutral"]
        pv = m.get("ensocheck", {}).get("obs_mwu_strong_vs_neutral_fixed_p")
        ens = (f" Check, DJF share of active days after the filter: strong El Ni&ntilde;o winters {se['active_fixed']:.2f} "
               f"(n&nbsp;{se['n']}; {se['active_raw']:.2f} before), neutral {ne['active_fixed']:.2f} (n&nbsp;{ne['n']})"
               + (f"; the difference is not significant (Mann&ndash;Whitney p&nbsp;{pv:.2f})." if pv is not None and pv >= 0.05
                  else f" (Mann&ndash;Whitney p&nbsp;{pv:.2f})." if pv is not None else "."))
    opts_f = "".join(f'<option value="{k}">{lab}</option>' for k, lab in [
        ("tas_na", "2 m temperature · North America"), ("tas_sa", "2 m temperature · South America"),
        ("pr_na", "Precipitation (mm/day) · North America"), ("pr_sa", "Precipitation (mm/day) · South America"),
        ("prpct_na", "Precipitation (% of normal) · North America"), ("prpct_sa", "Precipitation (% of normal) · South America"),
        ("z500_nh", "500 hPa height · Northern Hemisphere")])
    n_mod = len(m.get("models", []))
    n_mem = sum(len(v) for v in m.get("members", {}).values())
    per = m.get("periods", [])
    ismon = lambda k: k[:1] == "m" and k[1:].isdigit()                     # noqa: E731
    seas = [(k, "Nov&ndash;Mar (NDJFM)" if k == "ndjfm" else lab) for k, lab in per if not ismon(k)]
    mons = [(k, lab) for k, lab in per if ismon(k)]
    opts_m = (f'<optgroup label="Seasons">{"".join(f"<option value={k!r}>{lab}</option>" for k, lab in seas)}</optgroup>'
              + (f'<optgroup label="Single months (a third of the sample)">'
                 f'{"".join(f"<option value={k!r}>{lab}</option>" for k, lab in mons)}</optgroup>' if mons else ""))
    lw = m.get("lag_windows", {"0": [-2, 2], "10": [8, 12]})
    sg = lambda v: f"+{v}" if v > 0 else (f"&minus;{-v}" if v < 0 else "0")   # noqa: E731
    ag = m.get("tests", {}).get("agree", 0.8)
    agree = "two-thirds" if abs(ag - 2 / 3) < 0.01 else f"{100 * ag:.0f}&nbsp;%"
    ff = m.get("field_filter", {}); kn = ff.get("knots", {})
    dchk = ""
    dc = m.get("detrendcheck", {})
    if dc:
        v1 = [r["v1"]["var_frac_gt120d"] for r in dc.values()]; fin = [r["final"]["var_frac_gt120d"] for r in dc.values()]
        z = [abs(r["final"].get(f"{s_}_trend_per_decade", 0.0)) / max(r["final"].get(f"{s_}_trend_se", 1e-9), 1e-9)
             for r in dc.values() for s_ in ("DJF", "JJA") if f"{s_}_trend_se" in r["final"]]
        dchk = (f" Check on {len(dc)} test series (central US and Alaska temperature, Colombia and south-east Brazil "
                f"rain, Arctic 500&nbsp;hPa height; observed and three models): the share of variance at periods over "
                f"120&nbsp;days falls from {100 * min(v1):.0f}&ndash;{100 * max(v1):.0f}&nbsp;% to at most "
                f"{100 * max(fin):.1f}&nbsp;%, and the trends of the DJF and JJA seasonal means are within about two "
                f"standard errors of zero ({sum(v > 2 for v in z)} of {len(z)} beyond two, largest {max(z):.1f}, as chance "
                f"allows).")
    return f"""
  <section class="mi" id="mjo-impacts">
  <h2>MJO impacts by phase and season: {n_mod} CMIP6 models ({n_mem} members, 1979&ndash;2014), with the observed record for comparison</h2>
  <p class="lede lede--wide">What each MJO phase does to temperature, rainfall and the 500&nbsp;hPa flow, season by
  season, in {n_mod} CMIP6 models whose MJO passes a realism screen: a cell is coloured only where the models agree
  (significant across models and at least {agree} on the same sign). The observed record (1979&ndash;2024) is there
  for comparison; &ldquo;CMIP6 above observed&rdquo; stacks the two for the same field, season and lag. ENSO and the
  forced trend are removed from the index and from the fields. Blank means no significant signal.</p>
  <div class="mi-ctl">
    <label>Field <select id="mi-f">{opts_f}</select></label>
    <label>Data <select id="mi-d"><option value="cmip6">CMIP6 ({n_mod} models)</option><option value="obs">Observed (for comparison)</option></select></label>
    <label>Season <select id="mi-m">{opts_m}</select></label>
    <label>Lag <select id="mi-l"><option value="10">Days {sg(lw['10'][0])} to {sg(lw['10'][1])} after the phase</option><option value="0">Same days ({sg(lw['0'][0])} to {sg(lw['0'][1])})</option></select></label>
    <label>View <select id="mi-v"><option value="strip">All eight phases</option><option value="loop">One phase at a time</option><option value="compare">CMIP6 above observed (all phases)</option></select></label>
  </div>
  <iframe class="anim-embed mi-embed" id="mi-frame" title="MJO impact composites" loading="lazy"></iframe>
  <iframe class="anim-embed mi-embed" id="mi-frame2" title="MJO impact composites, observed" loading="lazy" hidden></iframe>
  <details class="mi-about"><summary>How this is built</summary>
  <p><b>MJO index.</b> CMIP6: the Wheeler &amp; Hendon (2004) RMM computed from each historical run&rsquo;s own daily
  OLR and 850/250&nbsp;hPa zonal wind (the CMIP6 daily archive has no 200&nbsp;hPa), 15&deg;S&ndash;15&deg;N means, the
  run&rsquo;s own seasonal cycle and previous 120-day mean removed, projected on the observed W&amp;H EOFs. Observed,
  for comparison: the Bureau of Meteorology&rsquo;s RMM, 1979&ndash;2024. <b>Both indices are then band-passed to
  20&ndash;100&nbsp;days and renormalised.</b> BoM removes the ENSO signal only up to 2013; after that a strong El
  Ni&ntilde;o&rsquo;s standing pattern projects on RMM as slow &ldquo;phase 6&ndash;8&rdquo; days (DJF 2015/16 was
  &ldquo;active&rdquo; on 89&nbsp;% of days), and the CMIP6 index would carry the same artefact. A linear Ni&ntilde;o-3.4
  regression removed almost none of it, so the band-pass is used for both datasets alike.{ens}{val}</p>
  <p><b>Impact fields</b> (2&nbsp;m temperature, precipitation, 500&nbsp;hPa height), per grid cell and per dataset, in
  three steps. (1) A seasonal cycle that is allowed to change: the mean and three annual harmonics, each with a
  coefficient that varies smoothly in time (a natural cubic spline with knots about every {ff.get('knot_years', 12)}
  years: 4 over the 36 model years, {kn.get('tas_na', 5)} over ERA5 1979&ndash;2024, {kn.get('pr_na', 3)} over GPCP
  1997&ndash;2024), so, for example, the Arctic&rsquo;s faster winter warming is not left in the anomalies. (2) The same
  regression&rsquo;s time-varying mean is the forced trend, smooth and nonlinear rather than a straight line. A model
  with two members is fitted on both together, that is on its ensemble mean, so the forced part is the model&rsquo;s and
  not one member&rsquo;s decade of weather; the daily ensemble mean is not subtracted, because with two members it holds
  half of each member&rsquo;s own MJO. (3) A high-pass at {ff.get('highpass_days', 120)}&nbsp;days: every Fourier
  component with a longer period is removed from what is left, which takes out ENSO and the rest of the interannual
  variability the same way the index filter does; the first and last {ff.get('edge_days_dropped', 60)} days of each
  record are dropped. (A 121-day running mean was tried first and left too much: a running mean cuts off gradually and
  still passes 78&nbsp;% of a 150-day and 17&nbsp;% of a one-year oscillation.){dchk}</p>
  <p><b>Composites.</b> A day counts when the MJO amplitude is at least 1. For every such day in the chosen season (DJF,
  MAM, JJA, SON, or November&ndash;March for a larger winter sample; single months have a third of the sample) the
  anomaly is averaged over days {sg(lw['0'][0])} to {sg(lw['0'][1])} (&ldquo;same days&rdquo;) or {sg(lw['10'][0])}
  to {sg(lw['10'][1])} after it, and the composite is the mean over those days.</p>
  <p><b>Model screen</b> (fixed before looking at any composite): east/west power ratio of 10&deg;S&ndash;10&deg;N rain
  (wavenumbers 1&ndash;3, 30&ndash;96&nbsp;days, November&ndash;April) &ge;&nbsp;{m.get('thresholds', {}).get('ew', 2.0)}
  and eastward propagation (lag-regression pattern against GPCP) r&nbsp;&ge;&nbsp;{m.get('thresholds', {}).get('prop_r', 0.8)};
  GPCP itself gives E/W&nbsp;{obs.get('ew', float('nan')):.1f}. Used: {', '.join(row(r) for r in passed) or 'none'}.
  {('Passed but left out as a near-duplicate of a family member: ' + ', '.join(r['model'] for r in fam) + '. ') if fam else ''}
  Failed: {', '.join(row(r) for r in failed) or 'none'}. {m.get('members_note', '')}</p>
  <p><b>Tests.</b> CMIP6: each model&rsquo;s composite (members pooled; at least 30 active days), across-model
  one-sample t-test with a Benjamini&ndash;Hochberg false-discovery rate of 10&nbsp;% over the map, and at least
  {agree} of the models agreeing on the sign. Observed: an MJO passage lasts days and daily anomalies are
  correlated, so the unit is the event (a run of consecutive active days in one phase), resampled with the whole lag
  window of every one of its days: an event-block bootstrap (2,000 resamples), the same 10&nbsp;% false-discovery rate,
  and only phase-seasons with at least 8 events are tested.</p>
  <p class="chart-sources">CMIP6 historical daily fields via the Pangeo cloud archive (anonymous GCS) &middot; ERA5
  2&nbsp;m temperature and 500&nbsp;hPa height (Copernicus C3S, 1979&ndash;2024) &middot; GPCP 1DD v1.3 daily
  precipitation (NOAA NCEI CDR, 1997&ndash;2024) &middot; RMM index: Australian Bureau of Meteorology &middot;
  static product, redrawn only when the analysis changes.</p>
  </details>
  </section>
  <script>
  (function () {{
    var ids = ["mi-f", "mi-d", "mi-m", "mi-l", "mi-v"], el = {{}};
    ids.forEach(function (k) {{ el[k] = document.getElementById(k); }});
    // default = #mi/tas_na/cmip6/djf/10/strip; a hash value that is not an option is ignored
    var def = ["tas_na", "cmip6", "djf", "10", "strip"];
    function apply(h) {{
      ids.forEach(function (k, i) {{
        var v = h[i], ok = v && [].some.call(el[k].options, function (o) {{ return o.value === v; }});
        el[k].value = ok ? v : def[i];
      }});
    }}
    apply((location.hash.indexOf("#mi/") === 0) ? location.hash.slice(4).split("/") : []);
    var fr1 = document.getElementById("mi-frame"), fr2 = document.getElementById("mi-frame2");
    function src(rid) {{
      return "sst_anim.html?embed=1&base=assets/mjo/impacts/anim&manifest=mjo_impacts_manifest.json&region="
        + rid + "&regions=" + rid + "&start=0";
    }}
    function go(push) {{
      var f = el["mi-f"].value, d = el["mi-d"].value, mo = el["mi-m"].value, l = ("0" + el["mi-l"].value).slice(-2);
      var v = el["mi-v"].value, stem = "mi_" + f + "_", tail = "_" + mo + "_l" + l;
      el["mi-d"].disabled = (v === "compare");
      if (v === "compare") {{
        fr1.src = src(stem + "cmip6" + tail + "_s"); fr2.src = src(stem + "obs" + tail + "_s"); fr2.hidden = false;
      }} else {{
        fr1.src = src(stem + d + tail + (v === "strip" ? "_s" : ""));
        if (!fr2.hidden) {{ fr2.hidden = true; fr2.removeAttribute("src"); }}
      }}
      if (push) history.replaceState(null, "", "#mi/" + ids.map(function (k) {{ return el[k].value; }}).join("/"));
    }}
    ids.forEach(function (k) {{ el[k].addEventListener("change", function () {{ go(true); }}); }});
    go(false);
    // a link to another view of this page (the site finder, a pasted #mi/... link) arrives as a hashchange
    addEventListener("hashchange", function () {{
      var h2 = (location.hash.indexOf("#mi/") === 0) ? location.hash.slice(4).split("/") : null;
      if (!h2 || h2.length !== 5) return;
      apply(h2);                                 // same validation as on load: stale values fall back to the defaults
      go(false);
      var st = document.getElementById("stage"); if (st) st.scrollIntoView({{ block: "start" }});
    }});
  }})();
  </script>"""


def main():
    items = discover()
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    ver = items[0][0] + items[0][1] if items else "0"

    if items:
        d, h, p = items[0]
        write_manifest(items)
        # Animator only (with the slider) — no separate static hero image. The iframe auto-sizes
        # to fit the plot + slider via the sstAnimHeight postMessage listener below.
        body = (f'  <p class="sub" style="margin-bottom:1rem">Latest init: <strong>{label(d, h)}</strong>.'
                ' Pick a figure from the rail: the AIFS-ENS forecast, the same forecast with the El Ni&ntilde;o'
                ' signal removed, and what each phase has meant for temperature, rain and the flow.</p>\n')
    else:
        body = '<p class="empty">No forecasts yet — the first scheduled run will populate this page.</p>'

    # One figure at a time on the shared stage viewer (assets/stage.js), 2026-09-27 (navigation phase 2): the
    # forecast animator is a frame product; the ENSO-removed and impacts sections mount whole, with their own
    # controls (the impacts keep their #mi/f/d/m/l/v hash: ownHash). Old links #mjo-clean / #mjo-impacts are
    # aliases. The rail replaces the "Jump to" line.
    clean, impacts = clean_section(), impacts_section()
    groups = [{"label": "Forecast", "items": [["rmm", "RMM forecast", "AIFS-ENS against IFS-ENS, 15 days"]]
               + ([["clean", "ENSO-removed index", "the El Ni\u00f1o signal taken out"]] if clean else [])}]
    if impacts:
        groups.append({"label": "Impacts", "items": [["mi", "Impacts by phase and season", "CMIP6 and observed composites"]]})
    stage_spec = ("window.STAGE = { groups: " + json.dumps(groups) + ",\n"
                  "  aliases: { \"mjo-clean\": \"clean\", \"mjo-impacts\": \"mi\" },\n"
                  "  products: {\n"
                  "    rmm: { label: \"AIFS-ENS RMM forecast, successive runs\", about: \"about-rmm\",\n"
                  "           frame: function () { return \"sst_anim.html?embed=1&base=assets&manifest=mjo/rmm_manifest.json&region=mjo\"; },\n"
                  "           ratio: function () { return \"1259/1210\"; },\n"
                  "           cap: function () { return \"Drag the slider to step through successive forecast runs (oldest to latest) and watch the predicted MJO track evolve.\"; } },\n"
                  "    clean: { label: \"MJO with the El Ni\u00f1o signal removed\", dom: function () { return \"mjo-clean\"; } },\n"
                  "    mi: { label: \"MJO impacts by phase and season\", ownHash: true, dom: function () { return \"mjo-impacts\"; } }\n"
                  "  }\n};")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MJO Forecast — AIFS-ENS RMM | Shawn Corvec</title>
<meta name="description" content="Real-time Multivariate MJO (RMM) phase-space forecast from the ECMWF AIFS-ENS ensemble — winds plus a precipitation-based pseudo-OLR channel, following Wheeler & Hendon (2004).">
<meta name="robots" content="index, follow">
<meta property="og:title" content="MJO Forecast — AIFS-ENS RMM">
<meta property="og:description" content="Real-time MJO (RMM) phase-space forecast from the ECMWF AIFS-ENS ensemble.">
<meta property="og:type" content="website">
<link rel="canonical" href="https://scorvec.com/mjo.html">\n<link rel="icon" href="/favicon.svg" type="image/svg+xml">\n<meta property="og:url" content="https://scorvec.com/mjo.html">
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #fafaf8; --ink: #1c1c1a; --muted: #6f6b64;
    --accent: #2c4a72; --rule: #e4e2dc;
    --serif: 'Lora', Georgia, serif; --sans: 'Inter', sans-serif;
  }}
  body {{ background: var(--bg); color: var(--ink); font-family: var(--sans);
         font-weight: 300; line-height: 1.6; }}
  nav {{ display: flex; align-items: center; justify-content: space-between;
        padding: 1.5rem 2rem; border-bottom: 1px solid var(--rule);
        max-width: 1500px; margin: 0 auto; }}
  .nav-name {{ font-family: var(--serif); font-size: 1.1rem; color: var(--ink);
              text-decoration: none; font-weight: 500; }}
  .nav-links {{ display: flex; gap: 1.8rem; list-style: none; }}
  .nav-links a {{ color: var(--muted); text-decoration: none; font-size: 0.85rem;
                 letter-spacing: 0.02em; }}
  .nav-links a:hover, .nav-links a.active {{ color: var(--ink); }}
  main {{ max-width: 1500px; margin: 0 auto; padding: 2.5rem 2rem 4rem; }}
  h1 {{ font-family: var(--serif); font-weight: 500; font-size: 1.9rem;
       margin-bottom: 0.4rem; }}
  .lede {{ color: var(--muted); max-width: 70ch; margin-bottom: 2rem; }}
  .hero {{ text-align: center; margin: 0 auto 2.5rem; }}
  .hero img {{ width: 100%; max-width: min(72vh, 920px); height: auto;
              border: 1px solid var(--rule); border-radius: 6px; }}
  .hero figcaption {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.6rem; }}
  h2 {{ font-family: var(--serif); font-weight: 500; font-size: 1.2rem;
       margin: 2rem 0 1rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--rule); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 1rem; }}
  .thumb {{ text-decoration: none; color: var(--muted); font-size: 0.78rem;
           text-align: center; }}
  .thumb img {{ width: 100%; height: auto; border: 1px solid var(--rule);
               border-radius: 4px; transition: border-color .15s; }}
  .thumb:hover img {{ border-color: var(--accent); }}
  .thumb span {{ display: block; margin-top: 0.35rem; }}
  /* Plot is 1259×1140 + a ~70px control row; the animator caps it at 88% of the
     window height, so the width that fits is (88vh - 70px) × 1259/1140 - the old
     72vh / 920px cap left a third of a laptop screen unused. */
  .anim-embed {{ width: 100%; max-width: min(100%, calc((88vh - 70px) * 1259 / 1140));
               margin: 0 auto; display: block;
               border: 1px solid var(--rule); border-radius: 6px; background: #0f0f0d;
               aspect-ratio: 1259 / 1210; max-height: 92vh; }}
  @media (max-width: 768px) {{ .anim-embed {{ aspect-ratio: 1259 / 1330; }} }}
  .meta {{ color: var(--muted); font-size: 0.8rem; margin-top: 2.5rem;
          border-top: 1px solid var(--rule); padding-top: 1rem; }}
  .empty {{ color: var(--muted); padding: 3rem 0; text-align: center; }}
  .mi-ctl {{ display: flex; flex-wrap: wrap; gap: 0.6rem 1.2rem; margin: 0.4rem 0 1rem; font-size: 0.9rem; }}
  .mi-ctl label {{ display: flex; flex-direction: column; gap: 0.2rem; color: var(--muted); }}
  .mi-ctl select {{ font: inherit; color: var(--ink); padding: 0.3rem 0.4rem; border: 1px solid var(--rule);
                   border-radius: 4px; background: #fff; max-width: 100%; }}
  .mi-embed {{ aspect-ratio: 1056 / 900; }}
  .mi-embed + .mi-embed {{ margin-top: 0.8rem; }}
  .mi-embed[hidden] {{ display: none; }}
  .mi-ctl select:disabled {{ color: var(--muted); background: transparent; }}
  .mi-about {{ margin-top: 1rem; color: var(--ink); max-width: 80ch; }}
  .mi-about summary {{ cursor: pointer; color: var(--accent); }}
  .mi-about p {{ margin: 0.6rem 0; font-size: 0.92rem; }}
  .chart-sources {{ color: var(--muted); font-size: 0.82rem; }}
  a {{ color: var(--accent); }}
  /* the stage viewer (assets/stage.css) on this page's palette */
  :root {{ --surface: #ffffff; --edge: #d9d5cc; --ink2: #444; --accent-light: rgba(44, 74, 114, 0.08); }}
  .page-header {{ margin-bottom: 1rem; }}
  .stage section h2 {{ margin-top: 0.2rem; }}
  .stage #mjo-clean {{ margin-top: 0 !important; }}
</style>
<link rel="stylesheet" href="/assets/stage.css">
<script data-goatcounter="https://scorvec.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
<link rel="stylesheet" href="/assets/lightbox.css">
<script src="/assets/lightbox.js" defer></script>
</head>
<body>
{NAV}
<main>
  <div class="page-header">
  <h1>MJO forecast from the AIFS ensemble</h1>
  {body}
  </div>
  <div class="ss-layout">
    <aside class="ss-rail" id="rail" aria-label="Figures"></aside>
    <section class="ss-main">
      <div class="crumb"><div class="path" id="crumb"></div>
        <div class="nav"><button type="button" id="prevBtn" title="previous (←)">&larr; prev</button><button type="button" id="nextBtn" title="next (→)">next &rarr;</button></div></div>
      <div class="opts"><div class="controls" id="opts-a"></div><div class="controls" id="opts-b"></div><div class="controls" id="opts-c"></div></div>
      <div class="stage" id="stage"></div>
      <p class="cap" id="cap"></p>
      <div id="about"></div>
      <p class="hint"><span class="kbd">&uarr;</span> <span class="kbd">&darr;</span> change figure &middot; the impacts keep their own selectors</p>
    </section>
  </div>
  <div class="lightbox" id="lightbox"><img alt=""></div>
  <div class="dom-holder">
{clean}
{impacts}
  </div>
  <template id="about-rmm">
  <p>Real-time Multivariate MJO (RMM) phase-space forecast from the
  ECMWF <strong>AIFS-ENS</strong> ensemble (51 members to day 15), following
  Wheeler &amp; Hendon (2004). Full three-channel projection: U850/U200 plus a
  pseudo-OLR channel built from &minus;standardized tropical precipitation (tropical
  rain and OLR anticorrelate closely); falls back to wind-only if precip is
  unavailable for a cycle. The physics-based <strong>IFS-ENS</strong> (50 members)
  is overlaid in blue through the <em>identical</em> machinery whenever its data
  has landed — a direct AI-vs-physics comparison in the same coordinates.
  Amplitude is the radial distance (rings at 1, 2, 3).
  Observed track is recent ERA5/AIFS analysis (wind-only, verified within a few
  degrees of the official BoM RMM phase).</p>
  <p class="chart-sources">Updated {updated} · Auto-generated from the AIFS-ENS open-data
  feed. Methodology: NOAA CPC / Wheeler &amp; Hendon (2004), EOFs from NOAA OLR +
  NCEP wind with the 120-day low-frequency filter removed; pseudo-OLR from
  &minus;standardized daily precip vs an ERA5 1991&ndash;2020 band climatology.</p>
  </template>
  <p class="meta">Updated {updated} · Auto-generated from the AIFS-ENS open-data feed.</p>
</main>
<script>
{stage_spec}
</script>
<script src="/assets/stage.js" defer></script>
<script>
  // Size the animator iframe to its exact content height (plot + slider) so the slider is
  // never clipped — sst_anim.html posts its height; we match it and drop the fixed aspect-ratio.
  addEventListener('message', function (e) {{
    var d = e.data; if (!d || d.type !== 'sstAnimHeight') return;
    var fr = document.querySelectorAll('iframe.anim-embed');
    for (var i = 0; i < fr.length; i++) {{
      if (fr[i].contentWindow === e.source) {{
        fr[i].style.height = d.h + 'px'; fr[i].style.aspectRatio = 'auto'; fr[i].style.maxHeight = 'none';
        fr[i].style.maxWidth = d.w ? d.w + 'px' : '';   // hug the height-capped plot (see sst.html)
      }}
    }}
  }});
  addEventListener('resize', function () {{
    var fr = document.querySelectorAll('iframe.anim-embed');
    for (var i = 0; i < fr.length; i++) fr[i].style.maxWidth = '';
  }});
</script>
</body>
</html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(items)} plot(s); latest {label(*[items[0][0], items[0][1]]) if items else 'none'})")
    # The template above carries the OLD nav; the shared header/tabs/footer live in
    # scripts/site/apply_chrome.py and must be stamped after every regeneration (the same
    # trap that bit the SST pages 2026-09-06: a green run shipped the old nav for a day).
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("apply_chrome", Path(__file__).resolve().parents[1] / "site" / "apply_chrome.py")
        chrome = importlib.util.module_from_spec(spec); spec.loader.exec_module(chrome)
        chrome.stamp_all(only=[str(OUT)], quiet=True)
    except Exception as e:                                          # noqa: BLE001
        print(f"  WARNING: site chrome not applied ({e!r}) — run scripts/site/apply_chrome.py", flush=True)


if __name__ == "__main__":
    main()
