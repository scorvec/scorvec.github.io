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
    <li><a href="gallery.html">Gallery</a></li>
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


def main():
    items = discover()
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    ver = items[0][0] + items[0][1] if items else "0"

    if items:
        d, h, p = items[0]
        write_manifest(items)
        # Animator only (with the slider) — no separate static hero image. The iframe auto-sizes
        # to fit the plot + slider via the sstAnimHeight postMessage listener below.
        body = (f'  <p class="lede" style="margin:0.5rem 0 1.2rem">Latest init: <strong>{label(d, h)}</strong>.'
                ' Drag the slider to step through successive forecast runs (oldest → latest) and watch the'
                ' predicted MJO track evolve.</p>\n'
                '  <iframe class="anim-embed" src="sst_anim.html?embed=1&amp;base=assets'
                '&amp;manifest=mjo/rmm_manifest.json&amp;region=mjo" '
                'title="AIFS-ENS RMM forecast — successive runs animation" loading="lazy"></iframe>')
    else:
        body = '<p class="empty">No forecasts yet — the first scheduled run will populate this page.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MJO Forecast — AIFS-ENS RMM | Shawn Corvec</title>
<meta name="description" content="Real-time Multivariate MJO (RMM) phase-space forecast from the ECMWF AIFS-ENS ensemble, following Wheeler & Hendon (2004).">
<meta name="robots" content="index, follow">
<meta property="og:title" content="MJO Forecast — AIFS-ENS RMM">
<meta property="og:description" content="Real-time MJO (RMM) phase-space forecast from the ECMWF AIFS-ENS ensemble.">
<meta property="og:type" content="website">
<link rel="canonical" href="https://scorvec.com/mjo.html">\n<link rel="icon" href="/favicon.svg" type="image/svg+xml">\n<meta property="og:url" content="https://scorvec.com/mjo.html">
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #fafaf8; --ink: #1c1c1a; --muted: #888580;
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
  .anim-embed {{ width: 100%; max-width: min(72vh, 920px); margin: 0 auto; display: block;
               border: 1px solid var(--rule); border-radius: 6px; background: #0f0f0d;
               aspect-ratio: 1259 / 1235; }}
  @media (max-width: 768px) {{ .anim-embed {{ aspect-ratio: 1259 / 1330; }} }}
  .meta {{ color: var(--muted); font-size: 0.8rem; margin-top: 2.5rem;
          border-top: 1px solid var(--rule); padding-top: 1rem; }}
  .empty {{ color: var(--muted); padding: 3rem 0; text-align: center; }}
  a {{ color: var(--accent); }}
</style>
<script data-goatcounter="https://scorvec.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
</head>
<body>
{NAV}
<main>
  <h1>MJO Forecast — AIFS-ENS</h1>
  <p class="lede">Real-time Multivariate MJO (RMM) phase-space forecast from the
  ECMWF <strong>AIFS-ENS</strong> ensemble (51 members to day 15), following
  Wheeler &amp; Hendon (2004). Wind-only projection (U850/U200); amplitude is the
  radial distance (rings at 1, 2, 3). Observed track is recent ERA5/AIFS analysis.</p>
  {body}
  <p class="meta">Updated {updated} · Auto-generated from the AIFS-ENS open-data
  feed. Methodology: NOAA CPC / Wheeler &amp; Hendon (2004), EOFs from NOAA OLR +
  NCEP wind with the 120-day low-frequency filter removed.</p>
</main>
<script>
  // Size the animator iframe to its exact content height (plot + slider) so the slider is
  // never clipped — sst_anim.html posts its height; we match it and drop the fixed aspect-ratio.
  addEventListener('message', function (e) {{
    var d = e.data; if (!d || d.type !== 'sstAnimHeight') return;
    var fr = document.querySelectorAll('iframe.anim-embed');
    for (var i = 0; i < fr.length; i++) {{
      if (fr[i].contentWindow === e.source) {{ fr[i].style.height = d.h + 'px'; fr[i].style.aspectRatio = 'auto'; }}
    }}
  }});
</script>
</body>
</html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(items)} plot(s); latest {label(*[items[0][0], items[0][1]]) if items else 'none'})")


if __name__ == "__main__":
    main()
