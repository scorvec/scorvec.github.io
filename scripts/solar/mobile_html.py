#!/usr/bin/env python3
"""Shared helper for writing mobile-responsive Plotly HTML.

Plotly's fig.write_html() produces a desktop-oriented page:
  - no <meta name="viewport"> tag, so phones render at desktop width
    then shrink the whole thing → tiny, un-tappable controls
  - a fixed-pixel-height plot div that doesn't reflow
  - no responsive config, so the chart doesn't resize with the viewport

This helper wraps write_html to fix all three. Every dashboard/map
generator calls write_responsive_html() instead of fig.write_html().

The key pieces:
  1. config={"responsive": True} → Plotly redraws on container resize
  2. inject <meta name="viewport"> into <head> via post-write string edit
  3. inject a small <style> block so the plot div fills width and uses
     a viewport-relative height (so it's tall enough to use on a phone
     but not absurdly so)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


# Viewport + responsive CSS injected into every generated page.
# The plot is given a height that works in an iframe: it fills the
# iframe's height (100%) with a sensible min so it never collapses.
_MOBILE_HEAD = """  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    html, body { margin: 0; padding: 0; height: 100%; width: 100%; }
    .plotly-graph-div {
      width: 100% !important;
      height: 100% !important;
      min-height: 360px;
    }
    /* Plotly's modebar can overflow on tiny screens; keep it contained */
    .modebar { right: 2px !important; }
    @media (max-width: 640px) {
      .modebar { transform: scale(0.85); transform-origin: top right; }
    }
  </style>
"""


def write_responsive_html(
    fig,
    output_html: Path,
    plotly_js: str = "inline",
    extra_config: Optional[dict] = None,
) -> None:
    """Write a Plotly figure to a mobile-responsive standalone HTML file.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to write.
    output_html : Path
        Destination path.
    plotly_js : str
        Passed through to write_html's include_plotlyjs
        ("inline", "cdn", or "directory").
    extra_config : dict, optional
        Additional Plotly config keys merged over the responsive default.
    """
    config = {
        "responsive": True,
        "displaylogo": False,
        # Drop some rarely-used modebar buttons to declutter on mobile
        "modeBarButtonsToRemove": [
            "select2d", "lasso2d", "autoScale2d",
        ],
    }
    if extra_config:
        config.update(extra_config)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_html),
        include_plotlyjs=plotly_js,
        full_html=True,
        config=config,
        # default_height/width let the div be fluid rather than fixed px
        default_height="100%",
        default_width="100%",
    )

    # Inject the viewport meta + responsive CSS right after <head>.
    # Plotly doesn't expose a head hook, so we patch the file.
    html = output_html.read_text()
    if "<head>" in html and "name=\"viewport\"" not in html:
        html = html.replace("<head>", "<head>\n" + _MOBILE_HEAD, 1)
        output_html.write_text(html)


def write_select_dashboard_html(
    fig,
    output_html: Path,
    options: list,
    visibility_map: dict,
    subtitle_map: dict,
    title: str,
    default_key: str,
    footer_text: str = "",
    theme: str = "dark",
    plotly_cdn: str = "https://cdn.plot.ly/plotly-2.35.2.min.js",
) -> None:
    """Write a dashboard with a NATIVE HTML <select> instead of Plotly's
    in-figure updatemenus dropdown.

    Plotly's updatemenus dropdown is unreliable on touch devices — the
    button overlay is drawn inside the plot and doesn't get native touch
    handling, so taps are flaky and long option lists overflow the
    viewport. A native <select> pops the OS picker on mobile and Just
    Works.

    The select's onchange toggles trace visibility via Plotly.update,
    reproducing exactly what the updatemenus buttons did.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure WITHOUT updatemenus (just the traces + layout).
    output_html : Path
        Destination.
    options : list[str]
        Ordered list of option keys (e.g. region names) for the select.
        Use the exact display strings you want in the dropdown.
    visibility_map : dict[str, list[bool]]
        Maps each option key → the per-trace visibility list to apply.
    subtitle_map : dict[str, str]
        Maps each option key → the subtitle text to show under the title.
    title : str
        Main title (shown above the subtitle).
    default_key : str
        Which option is selected on load.
    footer_text : str
        Small footer caption under the plot.
    theme : {"dark", "light"}
        Visual theme for the wrapper chrome (matches the iframe bg).
    plotly_cdn : str
        Plotly.js CDN URL.
    """
    import json as _json

    if theme == "dark":
        bg = "#0f0f0d"; fg = "#f0ebe0"; muted = "rgba(255,255,255,0.5)"
        sel_bg = "#1a1a17"; sel_border = "rgba(255,255,255,0.25)"
    else:
        bg = "#ffffff"; fg = "#1c1c1a"; muted = "rgba(0,0,0,0.5)"
        sel_bg = "#ffffff"; sel_border = "rgba(0,0,0,0.25)"

    # Figure JSON (data + layout) for Plotly.newPlot
    fig_json = fig.to_json()

    # Build <option> elements
    opts_html = "\n".join(
        f'      <option value="{_escape(o)}"'
        + (' selected' if o == default_key else '')
        + f'>{_escape(o)}</option>'
        for o in options
    )

    default_subtitle = subtitle_map.get(default_key, default_key)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  html, body {{
    margin: 0; padding: 0; height: 100%; width: 100%;
    background: {bg}; color: {fg};
    font-family: Inter, system-ui, -apple-system, sans-serif;
  }}
  #wrap {{ display: flex; flex-direction: column; height: 100vh; width: 100%; }}
  #topbar {{
    flex: 0 0 auto;
    display: flex; align-items: baseline; gap: 0.6rem;
    flex-wrap: wrap;
    padding: 0.7rem 1rem 0.5rem;
  }}
  #title {{ font-size: 1rem; font-weight: 500; }}
  #subtitle {{ font-size: 0.8rem; color: {muted}; }}
  #selector-row {{
    flex: 0 0 auto;
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0 1rem 0.6rem;
  }}
  #selector-row label {{ font-size: 0.78rem; color: {muted}; text-transform: uppercase; letter-spacing: 0.04em; }}
  select {{
    background: {sel_bg}; color: {fg};
    border: 1px solid {sel_border};
    border-radius: 6px;
    padding: 0.5rem 0.7rem;
    font-size: 0.95rem; font-family: inherit;
    min-height: 40px; flex: 1; max-width: 360px;
  }}
  #plot {{ flex: 1 1 auto; min-height: 0; width: 100%; }}
  .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
  #footer {{
    flex: 0 0 auto;
    font-size: 0.7rem; color: {muted};
    padding: 0.4rem 1rem 0.7rem; text-align: center;
  }}
  @media (max-width: 480px) {{
    #title {{ font-size: 0.92rem; }}
    #subtitle {{ font-size: 0.74rem; }}
    select {{ font-size: 0.9rem; }}
  }}
</style>
<script src="{plotly_cdn}"></script>
</head>
<body>
<div id="wrap">
  <div id="topbar">
    <span id="title">{_escape(title)}</span>
    <span id="subtitle">{_escape(default_subtitle)}</span>
  </div>
  <div id="selector-row">
    <label for="region-select">Region</label>
    <select id="region-select">
{opts_html}
    </select>
  </div>
  <div id="plot"></div>
  <div id="footer">{_escape(footer_text)}</div>
</div>

<script>
  const figJson = {fig_json};
  const visibilityMap = {_json.dumps(visibility_map)};
  const subtitleMap = {_json.dumps(subtitle_map)};

  const config = {{
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
  }};

  Plotly.newPlot('plot', figJson.data, figJson.layout, config);

  const sel = document.getElementById('region-select');
  const subtitleEl = document.getElementById('subtitle');
  sel.addEventListener('change', function() {{
    const key = sel.value;
    const vis = visibilityMap[key];
    if (vis) {{
      Plotly.restyle('plot', {{'visible': vis}});
    }}
    if (subtitleMap[key] !== undefined) {{
      subtitleEl.textContent = subtitleMap[key];
    }}
  }});
</script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


def _escape(s: str) -> str:
    """Minimal HTML escaping for text inserted into the template."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
