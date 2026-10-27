"""The site's style rules, checked over the pages and the shared stylesheets (CI: lint.yml; the pre-commit hook).

The one palette and type scale live in assets/site.css (html:root, the site tokens), first in every head, and the
pages draw on them:

  - a page does not redefine a site token (--bg, --panel, --surface, --ink, --ink2, --muted, --rule, --edge, --serif,
    --sans) at :root / html / body; a dark page's own theme (a block whose --bg is dark, or a selector naming its
    theme) is the exception, and a theme accent (--accent*) is set on html:root only, as outlook.css does;
  - a dark page's theme sets every colour the site's type layer reads (DARK_SET), its text tokens at 4.5:1 on its
    --bg, so no site rule falls back to a light-theme grey on the dark ground;
  - no inline h1 rule carries a literal size: a title is var(--t-h1), var(--t-h1-compact) or var(--t-hero);
  - the site stylesheet precedes every other stylesheet and style block in the head (apply_chrome.css_first);
  - every text token reads at 4.5:1 on the surfaces it is used on (WCAG AA), computed from the tokens themselves.

    python scripts/site/style_check.py            # exit 1 with one line per finding
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAGES = (
    sorted(REPO.glob("*.html"))
    + sorted((REPO / "scripts/sst/pages").glob("*.html"))
    + sorted((REPO / "topics").glob("*.html"))
    + sorted((REPO / "skewt").glob("*.html"))
)
SITE_CSS = REPO / "assets/site.css"
TOKENS = ("--bg", "--panel", "--surface", "--ink", "--ink2", "--muted", "--rule", "--edge", "--serif", "--sans")
ACCENTS = ("--accent", "--accent-hi", "--accent-light", "--accent-soft")
DARK_SET = ("--panel", "--ink", "--ink2", "--muted", "--muted-tint", "--rule", "--edge")
ROOT_SEL = re.compile(r"^(:root|html:root|html|body|html,\s*body)$")
STYLE = re.compile(r"<style[^>]*>(.*?)</style>", re.S)
RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
DECL = re.compile(r"(--[a-z0-9-]+)\s*:\s*([^;]+);")
HEX = re.compile(r"#([0-9a-fA-F]{6})\b")
H1_SIZE = re.compile(r"\bh1\b[^{}]*\{[^}]*font-size\s*:(?!\s*var\(--t-(h1|hero))[^;]+;")
FIRST_CSS = re.compile(
    r'<link\b(?![^>]*fonts\.googleapis)[^>]*rel="stylesheet"[^>]*>|<style\b'
)  # the font link is not a stylesheet of ours

# the surfaces each text token is read on: (text token, background token or literal, where)
PAIRS = [
    ("--muted", "--bg", "muted text on the page"),
    ("--muted", "--panel", "muted text on a card"),
    ("--muted-tint", "#e6edf4", "muted text on the viewers' control panel"),
    ("--muted-tint", "#f1e9dc", "muted text on the finder's warmest tint"),
    ("--ink2", "--bg", "card copy on the page"),
    ("--accent", "--bg", "links on the page"),
    ("--accent", "--panel", "links on a card"),
    ("--warm", "--panel", "a positive anomaly in text"),
    ("--cool", "--panel", "a negative anomaly in text"),
    ("#8a4b2a", "--bg", "the outlook pages' sienna accent on the page"),
]


def luminance(hexv: str) -> float:
    def ch(v: int) -> float:
        c = v / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hexv[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def site_tokens() -> dict[str, str]:
    """The html:root tokens of site.css, hex values only (var() references resolved one step)."""
    css = SITE_CSS.read_text()
    out: dict[str, str] = {}
    for sel, body in RULE.findall(css):
        if sel.strip().split("\n")[-1].strip() == "html:root":
            out.update(DECL.findall(body))
    for k, v in list(out.items()):
        m = re.fullmatch(r"var\((--[a-z0-9-]+)\)", v.strip())
        if m and m.group(1) in out:
            out[k] = out[m.group(1)]
    return {k: v.strip() for k, v in out.items()}


def page_findings(path: Path) -> list[str]:
    html = path.read_text()
    rel = path.relative_to(REPO).as_posix()
    if "/assets/site.css" not in html:
        return []  # a redirect stub or an embedded player: no chrome, its own styles
    out = []
    head = html[: html.find("</head>")] if "</head>" in html else html
    m = FIRST_CSS.search(head)
    if m and "site.css" not in m.group(0):
        out.append(f"{rel}: the site stylesheet is not the head's first stylesheet (run scripts/site/apply_chrome.py)")
    for block in STYLE.findall(html):
        for sel, body in RULE.findall(block):
            selector = sel.strip().split("\n")[-1].strip()
            if not ROOT_SEL.match(selector):
                continue
            bg = re.search(r"--bg\s*:\s*(#[0-9a-fA-F]{6})", body)
            if bg and luminance(bg.group(1)) < 0.18:
                out.extend(dark_findings(rel, selector, bg.group(1), dict(DECL.findall(body))))
                continue  # a dark page's own theme
            if "dark" in sel or "theme" in sel:
                continue
            names = [n for n, _ in DECL.findall(body)]
            for n in names:
                if n in TOKENS:
                    out.append(f"{rel}: {selector} redefines the site token {n} (assets/site.css sets it)")
                elif n in ACCENTS and selector != "html:root":
                    out.append(f"{rel}: {selector} sets {n}; a theme accent goes on html:root")
        out.extend(
            f"{rel}: an inline h1 rule sets a literal size ({m.group(0).strip()[:60]}…); use var(--t-h1)"
            for m in H1_SIZE.finditer(block)
        )
    return out


def dark_findings(rel: str, selector: str, bg: str, decl: dict[str, str]) -> list[str]:
    """A dark theme block: the full colour set, and its text tokens readable on its own ground."""
    out = [
        f"{rel}: {selector} is a dark theme without {n} (site rules would read the light one)"
        for n in DARK_SET
        if n not in decl
    ]
    for n in ("--ink", "--ink2", "--muted"):
        v = decl.get(n, "").strip()
        if HEX.fullmatch(v) and contrast(v, bg) < 4.5:
            out.append(f"{rel}: {selector} {n} {v} on --bg {bg} is {contrast(v, bg):.2f}:1 (4.5 needed)")
    return out


def token_findings() -> list[str]:
    tok = site_tokens()
    out = []
    for text, back, where in PAIRS:
        fg = tok.get(text, text)
        bg = tok.get(back, back)
        if not (HEX.fullmatch(fg) and HEX.fullmatch(bg)):
            out.append(f"assets/site.css: cannot read {text} on {back} as hex ({fg!r} on {bg!r})")
            continue
        ratio = contrast(fg, bg)
        if ratio < 4.5:
            out.append(f"assets/site.css: {where}: {text} {fg} on {back} {bg} is {ratio:.2f}:1 (4.5 needed)")
    return out


def main() -> int:
    findings = token_findings()
    for p in PAGES:
        findings += page_findings(p)
    for f in findings:
        print(f)
    print(f"style check: {len(PAGES)} pages, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
