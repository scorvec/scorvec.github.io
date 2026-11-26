#!/usr/bin/env python3
"""Cache-bust the Skew-T assets after a WebAssembly rebuild or a new gust model.

GitHub Pages serves app.js and sharplib.wasm with caching headers, so a browser
happily keeps running yesterday's JavaScript against today's HTML. The content
hashes of app.js and sharplib.js are stamped into skewt/index.html by
scripts/site/apply_chrome.py (its `local_assets` entry for the page), which every
chrome re-apply runs. sharplib.js resolves the .wasm URL itself, so a cached WASM
could still pair with fresh JS: this script pins the wasm hash into app.js
(WASM_V), and the gust model's (GUST_V, skewt/gust_model.json from
skewt-gust-verify.yml stage=export), then re-applies the chrome so the page picks
up the new app.js hash.

scripts/skewt/build_wasm.sh runs it at the end of every rebuild; by hand:

    python scripts/skewt/stamp_assets.py
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKEWT = HERE.parents[1] / "skewt"
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "site"))
import apply_chrome


def main() -> int:
    app = SKEWT / "app.js"
    src = new_src = app.read_text()
    for name, const in (("sharplib.wasm", "WASM_V"), ("gust_model.json", "GUST_V")):
        digest = hashlib.md5((SKEWT / name).read_bytes()).hexdigest()[:8]
        new_src = re.sub(rf'const {const} = "[^"]*";', f'const {const} = "{digest}";', new_src, count=1)
        if f'const {const} = "{digest}";' not in src:
            print(f"  {name} version -> {digest}", flush=True)
    if new_src != src:
        app.write_text(new_src)
    if not apply_chrome.stamp_all(only={"skewt/index.html"}):
        print("  assets unchanged", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
