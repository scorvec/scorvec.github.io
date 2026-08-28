#!/usr/bin/env python3
"""Fold a page's figure cards into plot decks (one card, tabbed panels).

The ENSO pages had grown to 12-22 cards each, nearly all of the same shape:
a heading, a caption, one figure, a source line. That is a long scroll, and
because every animator was its own card, opening a page loaded a dozen iframes
at once. This rewrites the templates so those cards become panels inside a
tabbed deck: one visible figure, the rest a click away and — crucially — not
loaded until asked for.

NOT folded:
  * genuinely interactive charts (Plotly explorers, the recharge oscillator,
    the daily/monthly index charts). Those are the reason to visit the page and
    they need their own space and full width.
  * anything whose panel would lose meaning without its neighbour.

Run against the templates in scripts/sst/pages/, then re-render with enso_site.
Idempotent: a page that already contains a deck is skipped.

    python scripts/sst/build_decks.py --page subsurface
    python scripts/sst/build_decks.py --all --dry-run
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PAGES = Path(__file__).resolve().parent / "pages"

# Cards to leave alone, matched on their <h2>. These are the interactive ones.
KEEP = [
    "Interactive Cross-Section Explorer",
    "Recharge Oscillator",
    "Daily ENSO indices",
    "ONI vs RONI",
]

# Short tab labels. A tab bar needs 2-4 words, not the full heading; deriving
# them by truncation produced things like "Subsurface Heat Content vs. Ana…",
# so they are spelled out.
LABELS = {
    "Today's SST anomaly": "SST anomaly",
    "Beyond ENSO": "PDO · IOD · S Atlantic",
    "Equatorial Pacific at kilometre scale": "MUR 1 km",
    "Ninety days in motion": "90 days, raw SST",
    "Daily SOI": "SOI vs record",
    "MEI.v2 Daily Nowcast": "MEI.v2",
    "Equatorial Convection": "OLR Hovmöller",
    "Live IR Satellite Loop": "IR satellite",
    "Tropical Waves and Variability": "Wave decomposition",
    "Kelvin": "Kelvin · Rossby trackers",
    "Equatorial Pacific Subsurface Temperature": "TAO cross-section",
    "Tropical Pacific Surface Currents": "Surface currents",
    "Equatorial Zonal Current": "Undercurrent",
    "Equatorial Surface Current": "Current Hovmöller",
    "Subsurface Heat Content vs. Analogs": "Heat content vs analogs",
    "Subsurface Cross-Section vs. Analogs": "Cross-section vs analogs",
    "Observed Equatorial Pacific Surface Wind": "ASCAT wind map",
    "Observed Equatorial Zonal Wind": "Zonal wind Hovmöller",
    "Hourly SOI Estimate": "SOI hourly",
    "Southern Oscillation Index Forecast": "SOI forecast",
    "Westerly-Wind-Burst Monitor": "WWB Tarawa",
    "WWB Activity vs Past": "WWB vs past onsets",
    "Equatorial Pacific 10 m Wind Forecast": "10 m wind forecast",
    "MSLP": "MSLP + wind",
    "GDPS Simulated IR": "GDPS sim IR",
    "GDPS 150 hPa": "GDPS 150 hPa",
    "200 hPa Velocity Potential": "Velocity potential",
    "Wave Activity Flux": "Wave activity flux",
    "Atmospheric Angular Momentum": "AAM",
    "AAM Forecast Trend": "AAM run-to-run",
    "AAM Phase": "AAM phase",
    # this one card carries three headings (torque, E-P flux, wave-1), so the
    # tab must not claim it is only the torque budget
    "What Drives AAM": "Torque · E–P flux · Wave-1",
    "E&ndash;P Flux": "E–P flux",
    "Stationary Wave-1": "Wave-1",
    "Where the AAM Lives": "AAM by latitude",
    "Meridional Overturning": "Hadley cell",
    "Zonal Overturning": "Walker cell",
    "Subtropical Jets": "Subtropical jets",
}


def plain(h: str) -> str:
    t = re.sub(r"<[^>]+>", "", h)
    t = re.sub(r"&nbsp;|&#160;", " ", t)
    return " ".join(t.split())


def label_for(h2_html: str) -> str:
    t = plain(h2_html)
    for k, v in LABELS.items():
        if plain(k).lower() in t.lower():
            return v
    return (t[:26] + "…") if len(t) > 27 else t


def split_cards(body: str):
    """Yield (start, end, html) for each top-level chart-card div."""
    out = []
    # Two container classes exist: the modern "chart-card" and a legacy
    # plain "card" (overview/forecasts). The class token must END here:
    # "chart-card" is a prefix of
    # "chart-card-header" and "chart-card-footer", so a loose match finds every
    # card three times (this bit twice before — keep the lookahead)
    for m in re.finditer(r'^(\s*)<div class="(?:chart-)?card(?=[" ])[^"]*">', body, re.M):
        indent, start = m.group(1), m.start()
        depth, i = 0, m.start()
        for tag in re.finditer(r"<div\b|</div>", body[m.start():]):
            depth += 1 if tag.group(0).startswith("<div") else -1
            if depth == 0:
                i = m.start() + tag.end()
                break
        out.append((start, i, body[start:i]))
    return out


def is_keeper(card: str) -> bool:
    h = re.search(r"<h2[^>]*>(.*?)</h2>", card, re.S)
    if not h:
        return True
    t = plain(h.group(1)).lower()
    return any(plain(k).lower() in t for k in KEEP)


def make_deck(cards, deck_id: str) -> str:
    tabs, panels = [], []
    for i, c in enumerate(cards):
        h = re.search(r"<h2[^>]*>(.*?)</h2>", c, re.S)
        tabs.append('        <button%s>%s</button>'
                    % (' class="on"' if i == 0 else "", label_for(h.group(1))))
        inner = re.sub(r'^\s*<div class="(?:chart-)?card(?=[" ])[^"]*">\s*\n', "", c)
        inner = re.sub(r"\n\s*</div>\s*$", "", inner)
        if i > 0:                       # defer everything but the first panel
            inner = inner.replace("<iframe ", '<iframe data-deferred ', 1)
            inner = re.sub(r'(<iframe[^>]*?)\ssrc="', r'\1 data-src="', inner, count=1)
        panels.append('      <div class="deck-panel"%s>\n%s\n      </div>'
                      % ("" if i == 0 else " hidden", inner))
    return ('    <div class="chart-card plot-deck" data-deck="%s">\n'
            '      <div class="deck-tabs">\n%s\n      </div>\n%s\n    </div>'
            % (deck_id, "\n".join(tabs), "\n\n".join(panels)))


def process(slug: str, dry: bool) -> None:
    p = PAGES / f"{slug}.html"
    body = p.read_text()
    if "plot-deck" in body:
        print(f"{slug}: already has a deck — skipped"); return
    cards = split_cards(body)
    groups, cur = [], []
    # group runs of consecutive foldable cards; a keeper or a section header
    # ends the run so decks never straddle a section boundary
    for idx, (a, b, c) in enumerate(cards):
        gap = body[cards[idx - 1][1]:a] if idx else ""
        if "page-section" in gap and cur:
            groups.append(cur); cur = []
        if is_keeper(c):
            if cur:
                groups.append(cur); cur = []
        else:
            cur.append((a, b, c))
    if cur:
        groups.append(cur)
    groups = [g for g in groups if len(g) > 1]
    if not groups:
        print(f"{slug}: nothing to fold"); return
    out = body
    for n, g in enumerate(reversed(groups)):
        deck = make_deck([c for _, _, c in g], f"{slug}{len(groups)-n}")
        out = out[:g[0][0]] + deck + out[g[-1][1]:]
    print(f"{slug}: {len(cards)} cards -> {len(cards)-sum(len(g) for g in groups)+len(groups)} "
          f"({len(groups)} deck(s) folding {sum(len(g) for g in groups)} cards)")
    for g in groups:
        print("    deck:", ", ".join(label_for(re.search(r'<h2[^>]*>(.*?)</h2>', c, re.S).group(1))
                                     for _, _, c in g))
    if not dry:
        p.write_text(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    slugs = ["overview", "subsurface", "atmosphere"] if a.all else [a.page]
    for s in slugs:
        process(s, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
