#!/usr/bin/env python3
"""Small WebP thumbnails for every figure in the plot catalogue (2026-09-27; the catalogue page had been loading the
full figures, ~8.6 MB to scroll the sheet).

Reads the `thumbs` table of assets/site/catalog.json (build_catalog.py): one entry per distinct figure, a file on
main or a loop's first frame on the frames branch. Each figure becomes a ~360 px wide WebP at
assets/site/thumbs/<id>.webp, published on the FRAMES branch like the loop frames (daily churn stays out of
main's history). Incremental: assets/site/thumbs/index.json on the branch records each thumbnail's source
signature (the git blob id on main, the ETag on the frames branch) and only changed figures are fetched and redrawn.

Writes into the workspace: the new/changed thumbnails and index.json under assets/site/thumbs/ (for
`publish_frames_ci.sh +assets/site/thumbs`, which merges them into the branch copy), the obsolete thumbnail paths to
--prune-out (for PRUNE=), and the thumbnail paths back into catalog.json.

    python scripts/site/build_thumbs.py --prune-out /tmp/prune.txt
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_catalog as B                                                      # noqa: E402

SLUG = "scorvec/scorvec.github.io"
RAW_FRAMES = f"https://raw.githubusercontent.com/{SLUG}/frames/"
WIDTH, MAX_H = 360, 540


def get(url: str, method: str = "GET", timeout: int = 60):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "scorvec-catalogue-thumbs"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.headers, (r.read() if method == "GET" else b"")
        except Exception as e:                                               # noqa: BLE001
            last = e
    raise last


def blob_ids() -> dict:
    out = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "HEAD", "--", "assets", "skewt"],
                         capture_output=True, text=True, check=True).stdout
    ids = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        ids[path] = meta.split()[2]
    return ids


def make_thumb(data: bytes, dest: Path) -> None:
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
    im.thumbnail((WIDTH, MAX_H), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "WEBP", quality=72, method=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(B.OUT))
    ap.add_argument("--prev-index", help="local copy of the published index.json (default: fetch from the frames branch)")
    ap.add_argument("--prune-out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    cat = json.loads(Path(a.catalog).read_text())
    table = cat.get("thumbs", {})
    prev = B.thumb_index(a.prev_index)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    blobs = blob_ids()

    jobs = []                                    # (index key, source rel, on frames, url)
    for rec in table.values():
        frames = bool(rec.get("f"))
        key = ("f:" if frames else "m:") + rec["s"]
        url = (RAW_FRAMES if frames else f"https://raw.githubusercontent.com/{SLUG}/{commit}/") + rec["s"]
        jobs.append((key, rec["s"], frames, url))

    def sig_of(job):
        key, rel, frames, url = job
        if not frames:
            return key, blobs.get(rel)
        try:
            h, _ = get(url, "HEAD", 20)
            return key, h.get("ETag") or h.get("Last-Modified")
        except Exception:                                                    # noqa: BLE001
            return key, None
    with ThreadPoolExecutor(a.workers) as ex:
        sigs = dict(ex.map(sig_of, jobs))

    todo = [j for j in jobs if not (prev.get(j[0], {}).get("t") and sigs.get(j[0]) and prev[j[0]].get("sig") == sigs[j[0]])]
    print(f"{len(jobs)} figures: {len(jobs) - len(todo)} thumbnails current, {len(todo)} to draw")

    def draw(job):
        key, rel, frames, url = job
        name = B.thumb_name(("frames/" if frames else "") + rel)
        try:
            _, data = get(url)
            make_thumb(data, REPO / B.THUMB_DIR / f"{name}.webp")
            return key, {"t": name, "sig": sigs.get(key)}, None
        except Exception as e:                                               # noqa: BLE001
            return key, None, f"{rel}: {e.__class__.__name__}: {str(e)[:120]}"
    index, fails = {}, []
    with ThreadPoolExecutor(a.workers) as ex:
        for key, rec, err in ex.map(draw, todo):
            if rec:
                index[key] = rec
            else:
                fails.append(err)
                if prev.get(key, {}).get("t"):
                    index[key] = prev[key]                                   # a stale thumbnail beats none
    for key, _rel, _f, _u in jobs:
        if key not in index and prev.get(key, {}).get("t"):
            index[key] = prev[key]
    for f in fails[:15]:
        print("  failed:", f)
    if len(fails) > 15:
        print(f"  ... {len(fails) - 15} more failures")

    keep = {r["t"] for r in index.values()}
    prune = sorted({f"{B.THUMB_DIR}/{r['t']}.webp" for r in prev.values() if r.get("t") and r["t"] not in keep})
    Path(a.prune_out).write_text(" ".join(prune))
    out_dir = REPO / B.THUMB_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":"), sort_keys=True))
    drawn = len(todo) - len(fails)
    print(f"drew {drawn}, kept {len(index) - drawn}, pruning {len(prune)}; {len(fails)} failed")

    # point the catalogue at the thumbnails
    for rec in table.values():
        hit = index.get(("f:" if rec.get("f") else "m:") + rec["s"])
        if hit:
            rec["t"] = hit["t"]
        else:
            rec.pop("t", None)
    Path(a.catalog).write_text(json.dumps(cat, ensure_ascii=False, separators=(",", ":")))
    changed = drawn > 0 or bool(prune) or json.dumps(prev, sort_keys=True) != json.dumps(index, sort_keys=True)
    print("THUMBS_CHANGED=" + ("1" if changed else "0"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
