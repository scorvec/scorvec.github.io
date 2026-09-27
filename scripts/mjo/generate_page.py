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
    """'MJO impacts by phase and month' (2026-09-27): CMIP6 and observed composites drawn by
    scripts/mjo/src/mjo_impacts_render.py (mjo-impacts.yml) from the committed reference. Empty until it exists."""
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
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    opts_m = "".join(f'<option value="{i + 1:02d}">{n}</option>' for i, n in enumerate(months))
    return f"""
  <section class="mi" id="mjo-impacts">
  <h2>MJO impacts by phase and month</h2>
  <p class="lede lede--wide">What each MJO phase has meant for temperature, rainfall and the 500&nbsp;hPa flow, month by
  month: the observed record next to {len(m.get('models', []))} CMIP6 models whose MJO passes a realism screen. Only
  statistically significant cells are coloured; blank means no significant signal.</p>
  <div class="mi-ctl">
    <label>Field <select id="mi-f">{opts_f}</select></label>
    <label>Data <select id="mi-d"><option value="obs">Observed</option><option value="cmip6">CMIP6</option></select></label>
    <label>Month <select id="mi-m">{opts_m}</select></label>
    <label>Lag <select id="mi-l"><option value="10">10 days after the phase</option><option value="0">Same day</option></select></label>
    <label>View <select id="mi-v"><option value="loop">One phase at a time</option><option value="strip">All eight phases</option></select></label>
  </div>
  <iframe class="anim-embed mi-embed" id="mi-frame" title="MJO impact composites" loading="lazy"></iframe>
  <details class="mi-about"><summary>How this is built</summary>
  <p><b>MJO index.</b> Observed: the Bureau of Meteorology&rsquo;s RMM (Wheeler &amp; Hendon 2004), 1979&ndash;2024.
  CMIP6: the same RMM computed from each run&rsquo;s own daily OLR and 850/250&nbsp;hPa zonal wind (the CMIP6 daily
  archive has no 200&nbsp;hPa), 15&deg;S&ndash;15&deg;N means, the run&rsquo;s own seasonal cycle and previous
  120-day mean removed, projected on the observed W&amp;H EOFs. <b>Both indices are then band-passed to
  20&ndash;100&nbsp;days and renormalised.</b> BoM removes the ENSO signal only up to 2013; after that a strong El
  Ni&ntilde;o&rsquo;s standing pattern projects on RMM as slow &ldquo;phase 6&ndash;8&rdquo; days (DJF 2015/16 was
  &ldquo;active&rdquo; on 89&nbsp;% of days), and the CMIP6 index would carry the same artefact. A linear Ni&ntilde;o-3.4
  regression removed almost none of it, so the band-pass is used for both datasets alike.{ens} A day counts when the
  amplitude is at least 1; the
  composite is the mean anomaly on the same day or 10&nbsp;days later, over those days in the chosen month and its two
  neighbours (a centred three-month window, so &ldquo;January&rdquo; uses December&ndash;February days: it triples the
  observed sample). Anomalies are from each dataset&rsquo;s own seasonal cycle and linear trend.{val}</p>
  <p><b>Model screen</b> (fixed before looking at any composite): east/west power ratio of 10&deg;S&ndash;10&deg;N rain
  (wavenumbers 1&ndash;3, 30&ndash;96&nbsp;days, November&ndash;April) &ge;&nbsp;{m.get('thresholds', {}).get('ew', 2.0)}
  and eastward propagation (lag-regression pattern against GPCP) r&nbsp;&ge;&nbsp;{m.get('thresholds', {}).get('prop_r', 0.8)};
  GPCP itself gives E/W&nbsp;{obs.get('ew', float('nan')):.1f}. Used: {', '.join(row(r) for r in passed) or 'none'}.
  {('Passed but left out as a near-duplicate of a family member: ' + ', '.join(r['model'] for r in fam) + '. ') if fam else ''}
  Failed: {', '.join(row(r) for r in failed) or 'none'}. {m.get('members_note', '')}</p>
  <p><b>Tests.</b> CMIP6: each model&rsquo;s composite (members pooled; at least 30 active days), across-model
  one-sample t-test with a Benjamini&ndash;Hochberg false-discovery rate of 10&nbsp;% over the map, and at least
  80&nbsp;% of the models agreeing on the sign. Observed: an MJO passage lasts days and daily anomalies are
  correlated, so the unit is the event (a run of consecutive active days in one phase): an event-block bootstrap of the
  composite, the same 10&nbsp;% false-discovery rate, and only phase-months with at least 8 events are tested.</p>
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
    var h = (location.hash.indexOf("#mi/") === 0) ? location.hash.slice(4).split("/") : null;
    if (h && h.length === 5) {{ ids.forEach(function (k, i) {{ if (h[i]) el[k].value = h[i]; }}); }}
    else {{ el["mi-m"].value = ("0" + (new Date().getMonth() + 1)).slice(-2); }}
    function go(push) {{
      var f = el["mi-f"].value, d = el["mi-d"].value, mo = el["mi-m"].value, l = ("0" + el["mi-l"].value).slice(-2);
      var rid = "mi_" + f + "_" + d + "_m" + mo + "_l" + l + (el["mi-v"].value === "strip" ? "_s" : "");
      document.getElementById("mi-frame").src = "sst_anim.html?embed=1&base=assets/mjo/impacts/anim&manifest=mjo_impacts_manifest.json&region="
        + rid + "&regions=" + rid + "&start=0";
      if (push) history.replaceState(null, "", "#mi/" + ids.map(function (k) {{ return el[k].value; }}).join("/"));
    }}
    ids.forEach(function (k) {{ el[k].addEventListener("change", function () {{ go(true); }}); }});
    go(false);
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
                ' Drag the slider to step through successive forecast runs (oldest → latest) and watch the'
                ' predicted MJO track evolve.</p>\n'
                '  <iframe class="anim-embed" src="sst_anim.html?embed=1&amp;base=assets'
                '&amp;manifest=mjo/rmm_manifest.json&amp;region=mjo" '
                'title="AIFS-ENS RMM forecast — successive runs animation" loading="lazy"></iframe>\n')
    else:
        body = '<p class="empty">No forecasts yet — the first scheduled run will populate this page.</p>'

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
  .mi-about {{ margin-top: 1rem; color: var(--ink); max-width: 80ch; }}
  .mi-about summary {{ cursor: pointer; color: var(--accent); }}
  .mi-about p {{ margin: 0.6rem 0; font-size: 0.92rem; }}
  .chart-sources {{ color: var(--muted); font-size: 0.82rem; }}
  a {{ color: var(--accent); }}
</style>
<script data-goatcounter="https://scorvec.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
<link rel="stylesheet" href="/assets/lightbox.css">
<script src="/assets/lightbox.js" defer></script>
</head>
<body>
{NAV}
<main>
  <h1>MJO forecast from the AIFS ensemble</h1>
  {body}
{clean_section()}
  <p class="lede lede--wide" style="margin-top:1.4rem">Real-time Multivariate MJO (RMM) phase-space forecast from the
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
  <p class="meta">Updated {updated} · Auto-generated from the AIFS-ENS open-data
  feed. Methodology: NOAA CPC / Wheeler &amp; Hendon (2004), EOFs from NOAA OLR +
  NCEP wind with the 120-day low-frequency filter removed; pseudo-OLR from
  &minus;standardized daily precip vs an ERA5 1991&ndash;2020 band climatology.</p>
{impacts_section()}
</main>
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
