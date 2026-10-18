#!/usr/bin/env python3
"""Cache-bust the Skew-T assets after a WebAssembly rebuild.

GitHub Pages serves app.js and sharplib.wasm with caching headers, so a browser
happily keeps running yesterday's JavaScript against today's HTML. The content
hashes of app.js and sharplib.js are stamped into skewt/index.html by
scripts/site/apply_chrome.py (its `local_assets` entry for the page), which every
chrome re-apply runs. sharplib.js resolves the .wasm URL itself, so a cached WASM
could still pair with fresh JS: this script pins the wasm hash into app.js
(WASM_V) and then re-applies the chrome so the page picks up the new app.js hash.

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
    wasm, app = SKEWT / "sharplib.wasm", SKEWT / "app.js"
    digest = hashlib.md5(wasm.read_bytes()).hexdigest()[:8]
    src = app.read_text()
    new_src = re.sub(r'const WASM_V = "[^"]*";', f'const WASM_V = "{digest}";', src, count=1)
    if new_src != src:
        app.write_text(new_src)
        print(f"  wasm version -> {digest}", flush=True)
    if not apply_chrome.stamp_all(only={"skewt/index.html"}):
        print("  assets unchanged", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
