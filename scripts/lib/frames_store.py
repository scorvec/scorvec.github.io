#!/usr/bin/env python3
"""Animation frames in an S3-compatible object store (Cloudflare R2).

    python scripts/lib/frames_store.py sync  assets/sst/anim/pacsat ...   # local -> store
    python scripts/lib/frames_store.py seed  assets/sst/anim/mur_ct ...   # store -> local
    python scripts/lib/frames_store.py prune assets/sst/anim/old_thing    # delete from store
    python scripts/lib/frames_store.py ls    [assets/sst/anim]            # dir -> frame count
    python scripts/lib/frames_store.py configured                         # exit 0 if usable

Keys mirror repo paths exactly (assets/<product>/anim/<dir>/<frame>.webp), so
the public bucket URL plays the role raw.githubusercontent.com/.../frames/
did: FRAME_ROOT + path, nothing else changes for the viewers.

WHY AN OBJECT STORE. The `frames` branch was ONE parentless commit force-pushed
by every publisher. Whatever was not in the tree you pushed was deleted, so two
publishers overlapping meant the loser's frames silently vanished (wave1_maps
and vortex_winds, 2026-08-30). Every workaround since - clone-swap-reorphan,
the carry-over loop on the laptop, the GC job that refuses to run while
anything else is in flight - existed to make a global force-push behave like a
per-directory write. An object store IS a per-directory write: `sync` touches
only the prefixes it is given and cannot delete anyone else's.

Environment (the same on a runner and on the laptop):
    FRAMES_S3_BUCKET      bucket name                          (required)
    FRAMES_S3_ENDPOINT    https://<account>.r2.cloudflarestorage.com (required)
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY                  (required)
Nothing set -> `configured` exits 1 and the shell wrappers fall back to the
branch, so this can be wired in before the bucket exists.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
WORKERS = int(os.environ.get("FRAMES_WORKERS", "16"))
# Viewers append ?v=<manifest ver> to every frame URL, so a long browser TTL is
# safe: a re-rendered frame under the same name is fetched fresh as soon as
# the manifest on main changes, and never before.
CACHE_CONTROL = "public, max-age=3600"
CONTENT_TYPES = {".webp": "image/webp", ".png": "image/png", ".json": "application/json",
                 ".txt": "text/plain"}
# 12 h of grace for --guard-stale, matching what publish_frames.sh used against
# the branch date: everything here renders at least daily, so a local copy more
# than 12 h older than what the store already holds is a leftover, not a render.
STALE_GRACE_S = 12 * 3600


def configured() -> bool:
    return all(os.environ.get(k) for k in
               ("FRAMES_S3_BUCKET", "FRAMES_S3_ENDPOINT",
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"))


def client():
    import boto3
    from botocore.config import Config
    # R2 rejects the CRC-based checksum headers boto3 >= 1.36 sends by default;
    # "when_required" restores the plain MD5 behaviour R2 (and S3) accept.
    cfg = Config(region_name="auto", max_pool_connections=WORKERS + 4,
                 retries={"max_attempts": 5, "mode": "standard"},
                 request_checksum_calculation="when_required",
                 response_checksum_validation="when_required")
    return boto3.client("s3", endpoint_url=os.environ["FRAMES_S3_ENDPOINT"], config=cfg)


def bucket() -> str:
    return os.environ["FRAMES_S3_BUCKET"]


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_listing(s3, prefix: str) -> dict[str, dict]:
    """key -> {etag, size, mtime} for everything under prefix/."""
    out: dict[str, dict] = {}
    prefix = prefix.rstrip("/") + "/"
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket(), Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"]] = {"etag": o["ETag"].strip('"'), "size": o["Size"],
                             "mtime": o["LastModified"].timestamp()}
    return out


def local_listing(d: Path) -> dict[str, Path]:
    """relative repo path -> file, for every regular file under d."""
    return {str(p.relative_to(REPO)): p for p in sorted(d.rglob("*")) if p.is_file()}


def _rel(d: str) -> str:
    return d.strip("/").rstrip("/")


# ---------------------------------------------------------------------------
def cmd_sync(dirs: list[str], guard_stale: bool) -> int:
    s3 = client()
    total_up = total_del = 0
    for d in dirs:
        d = _rel(d)
        loc = local_listing(REPO / d) if (REPO / d).is_dir() else {}
        if not loc:
            # A product that failed to render must not blank its own frames.
            print(f"  {d}: nothing rendered locally, store copy kept")
            continue
        rem = remote_listing(s3, d)
        if guard_stale and rem:
            newest_local = max(p.stat().st_mtime for p in loc.values())
            newest_remote = max(v["mtime"] for v in rem.values())
            if newest_local < newest_remote - STALE_GRACE_S:
                age_h = (newest_remote - newest_local) / 3600
                print(f"  {d}: local copy is {age_h:.0f}h older than the store - skipped")
                continue
        to_up = [k for k, p in loc.items()
                 if k not in rem or rem[k]["size"] != p.stat().st_size
                 or rem[k]["etag"] != _md5(p)]
        to_del = [k for k in rem if k not in loc]

        def put(k: str) -> None:
            p = loc[k]
            s3.upload_file(str(p), bucket(), k, ExtraArgs={
                "ContentType": CONTENT_TYPES.get(p.suffix, "application/octet-stream"),
                "CacheControl": CACHE_CONTROL})

        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(put, to_up))
        for i in range(0, len(to_del), 1000):
            s3.delete_objects(Bucket=bucket(), Delete={
                "Objects": [{"Key": k} for k in to_del[i:i + 1000]], "Quiet": True})
        print(f"  {d}: {len(loc)} files, {len(to_up)} uploaded, {len(to_del)} removed")
        total_up += len(to_up)
        total_del += len(to_del)
    print(f"  synced: {total_up} uploaded, {total_del} removed")
    return 0


def cmd_seed(dirs: list[str], exclude: str | None) -> int:
    """store -> local, never overwriting what this run has already produced."""
    s3 = client()
    for d in dirs:
        d = _rel(d)
        rem = remote_listing(s3, d)
        if not rem:
            print(f"  {d}: not in the store yet")
            continue
        want = []
        for k in rem:
            child = k[len(d) + 1:].split("/")[0]
            if exclude and child == exclude:
                continue
            if not (REPO / k).exists():
                want.append(k)

        def get(k: str) -> None:
            dst = REPO / k
            dst.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket(), k, str(dst))

        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(get, want))
        if exclude:
            print(f"  {d}: seeded {len(want)} files ({exclude} skipped - rendered by this run)")
        else:
            print(f"  {d}: seeded {len(want)} files")
    return 0


def cmd_prune(paths: list[str]) -> int:
    s3 = client()
    for d in paths:
        d = _rel(d)
        keys = list(remote_listing(s3, d))
        for i in range(0, len(keys), 1000):
            s3.delete_objects(Bucket=bucket(), Delete={
                "Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})
        print(f"  pruned {d} ({len(keys)} files)")
    return 0


def cmd_ls(prefix: str) -> int:
    """Frame directories under prefix with their counts - what gc_frames.py reads."""
    s3 = client()
    counts: dict[str, int] = {}
    for k in remote_listing(s3, prefix or "assets"):
        if k.endswith(".webp"):
            counts[os.path.dirname(k)] = counts.get(os.path.dirname(k), 0) + 1
    for d in sorted(counts):
        print(f"{counts[d]:6d} {d}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sync"); p.add_argument("dirs", nargs="+")
    p.add_argument("--guard-stale", action="store_true",
                   help="skip a directory whose local copy is >12 h older than the store's")
    p = sub.add_parser("seed"); p.add_argument("dirs", nargs="+")
    p.add_argument("--exclude", help="child directory not to seed")
    p = sub.add_parser("prune"); p.add_argument("paths", nargs="+")
    p = sub.add_parser("ls"); p.add_argument("prefix", nargs="?", default="assets")
    sub.add_parser("configured")
    a = ap.parse_args()

    if a.cmd == "configured":
        return 0 if configured() else 1
    if not configured():
        print("frames_store: FRAMES_S3_BUCKET / FRAMES_S3_ENDPOINT / AWS_* not set", file=sys.stderr)
        return 2
    t0 = time.time()
    rc = {"sync": lambda: cmd_sync(a.dirs, a.guard_stale),
          "seed": lambda: cmd_seed(a.dirs, a.exclude),
          "prune": lambda: cmd_prune(a.paths),
          "ls": lambda: cmd_ls(a.prefix)}[a.cmd]()
    print(f"  ({time.time() - t0:.1f}s)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
