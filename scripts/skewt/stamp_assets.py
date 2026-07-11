#!/usr/bin/env python3
"""Cache-bust the Skew-T assets.

GitHub Pages serves app.js and sharplib.wasm with caching headers, so a browser
happily keeps running yesterday's JavaScript against today's HTML — which is how
a shipped fix can appear "not deployed". Stamp each asset's content hash into
the <script> tags so a changed file is always a new URL.

Run before committing any change to skewt/app.js or skewt/sharplib.*:

    python scripts/skewt/stamp_assets.py
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

SKEWT = Path(__file__).resolve().parents[2] / "skewt"


def digest(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def main() -> int:
    # 1. stamp the wasm hash INTO app.js first — sharplib.js resolves the .wasm
    #    URL itself, so a cached WASM could otherwise pair with fresh JS.
    wasm = SKEWT / "sharplib.wasm"
    app = SKEWT / "app.js"
    if wasm.exists() and app.exists():
        src = app.read_text()
        new_src = re.sub(r'const WASM_V = "[^"]*";',
                         f'const WASM_V = "{digest(wasm)}";', src, count=1)
        if new_src != src:
            app.write_text(new_src)
            print(f"  wasm version -> {digest(wasm)}", flush=True)

    # 2. then hash app.js (now that it's final) into the script tags
    html_path = SKEWT / "index.html"
    html = html_path.read_text()
    changed = []
    for name in ("sharplib.js", "app.js"):
        f = SKEWT / name
        if not f.exists():
            continue
        h = digest(f)
        # match src="app.js" or src="app.js?v=abcd1234"
        pat = re.compile(r'src="(' + re.escape(name) + r')(\?v=[a-f0-9]+)?"')
        new, n = pat.subn(f'src="{name}?v={h}"', html)
        if n and new != html:
            changed.append(f"{name} -> {h}")
        html = new
    html_path.write_text(html)
    if changed:
        print("  stamped: " + ", ".join(changed), flush=True)
    else:
        print("  assets unchanged", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
