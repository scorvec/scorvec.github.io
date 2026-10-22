#!/usr/bin/env python3
"""Animation frames in an S3-compatible object store (Bunny Storage, through its S3 API).

    python scripts/lib/frames_store.py sync  assets/sst/anim/pacsat +assets/site/thumbs ...  # local -> store
    python scripts/lib/frames_store.py seed  assets/sst/anim/mur_ct ...                      # store -> local
    python scripts/lib/frames_store.py prune assets/sst/anim/old_thing assets/site/thumbs/x.webp  # delete
    python scripts/lib/frames_store.py prune-cycles assets/models/data/hrrr:2026102012       # older cycles
    python scripts/lib/frames_store.py ls    [assets/sst/anim]                               # dir -> frame count
    python scripts/lib/frames_store.py check                                 # round trip on a scratch key
    python scripts/lib/frames_store.py configured                            # exit 0 if usable

Keys mirror repo paths exactly (assets/<product>/anim/<dir>/<frame>.webp), so the store's pull zone plays the role
raw.githubusercontent.com/.../frames/ did: STORE + path (assets/frames_root.js), nothing else changes for the
viewers. Local paths are relative to --root (default: the current directory, the repository root on a runner).

WHY AN OBJECT STORE. The `frames` branch was ONE parentless commit force-pushed
by every publisher. Whatever was not in the tree you pushed was deleted, so two
publishers overlapping meant the loser's frames silently vanished (wave1_maps
and vortex_winds, 2026-08-30). Every workaround since - clone-swap-reorphan,
the carry-over loop, the GC job that refuses to run while
anything else is in flight - existed to make a global force-push behave like a
per-directory write. An object store IS a per-directory write: `sync` touches
only the prefixes it is given and cannot delete anyone else's.

SYNC, as publish_frames_ci.sh treats its arguments: "dir" makes the store's copy exactly the local one (uploads first,
then deletes what the local copy no longer has, so a viewer never meets a gap); "+dir" adds and replaces files and
never deletes, for archives several jobs extend at once. A path with nothing in it locally is skipped: a product that
failed to render must not blank its own frames. *_manifest.json stays out of the store, as off the branch: manifests
live on main and are fetched same-origin. A file is unchanged when its size and the store's ETag match (the ETag of a
single PUT is the MD5 on S3 and R2). Bunny does not document its ETag: if it is not the MD5, nothing is ever skipped
- every sync re-sends its directories in full, correct but slower - and sync says so in its summary (`check` answers
the question once).

BUNNY'S S3 API (public preview) shapes the client: no DeleteObjects (501), so deletes are single requests in
parallel; no Cache-Control, ACL or x-amz-meta headers, so caching is the pull zone's setting; 500 requests/s per zone,
answered past that with 503 SlowDown and a Retry-After, which every request here honours (botocore's own retries are
off, so there is one retry policy); path-style addressing; ranges, delimiter "/" listings and single PUTs as on S3.

Environment:
    FRAMES_S3_BUCKET             the bucket: on Bunny, the storage zone name                   (required)
    FRAMES_S3_ENDPOINT           https://<region>-s3.storage.bunnycdn.com                       (required)
    FRAMES_S3_ACCESS_KEY_ID      on Bunny, the storage zone name                                (required)
    FRAMES_S3_SECRET_ACCESS_KEY  on Bunny, the storage zone password                            (required)
    FRAMES_S3_REGION             the signing region (default us-east-1; `check` reports which ones sign)
    FRAMES_S3_CACHE_CONTROL      a Cache-Control header for every upload, for a store that keeps one (not Bunny)
    FRAMES_WORKERS               parallel requests (default 16)
Nothing set -> `configured` exits 1 and the shell wrappers keep to the branch, so this can be wired in before the
store exists. The credentials are not AWS_*: a job-wide AWS_ACCESS_KEY_ID would be picked up by every other S3 client
in the job.
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import os
import re
import secrets
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

CREDENTIALS = ("FRAMES_S3_BUCKET", "FRAMES_S3_ENDPOINT", "FRAMES_S3_ACCESS_KEY_ID", "FRAMES_S3_SECRET_ACCESS_KEY")
WORKERS = int(os.environ.get("FRAMES_WORKERS", "16"))
CONTENT_TYPES = {
    ".webp": "image/webp",
    ".png": "image/png",
    ".json": "application/json",
    ".txt": "text/plain",
    ".csv": "text/csv",
}
MANIFEST_SUFFIX = "_manifest.json"
CYCLE = re.compile(r"\d{10}")  # a model cycle directory, YYYYMMDDHH (publish_data.sh PRUNE_OLDER)
ATTEMPTS = 8
RETRY_STATUS = {429, 500, 502, 503, 504}  # throttled or transient; 501 (an operation the store lacks) is final
RETRY_AFTER_MAX_S = 60
CHECK_PREFIX = "_frames_check"  # `check` writes one scratch object here and deletes it


def configured() -> bool:
    return all(os.environ.get(k) for k in CREDENTIALS)


def client(region: str | None = None):
    import boto3
    from botocore.config import Config

    cfg = Config(
        region_name=region or os.environ.get("FRAMES_S3_REGION") or "us-east-1",
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        max_pool_connections=WORKERS + 4,
        connect_timeout=10,
        read_timeout=60,
        retries={"total_max_attempts": 1, "mode": "standard"},  # _call retries, honouring Retry-After
        # S3-compatible stores take the plain request: no CRC checksum headers or trailers (which boto3 >= 1.36 sends
        # by default and R2 rejects; Bunny documents none), so the payload is SHA-256 signed instead.
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    return boto3.client(
        "s3",
        endpoint_url=os.environ["FRAMES_S3_ENDPOINT"],
        aws_access_key_id=os.environ["FRAMES_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["FRAMES_S3_SECRET_ACCESS_KEY"],
        config=cfg,
    )


def bucket() -> str:
    return os.environ["FRAMES_S3_BUCKET"]


def _retry_after(value: str | None) -> float | None:
    """Seconds from a Retry-After header (delta-seconds or an HTTP date), capped; None when absent or unreadable."""
    if not value:
        return None
    try:
        return min(RETRY_AFTER_MAX_S, max(0.0, float(value)))
    except ValueError:
        pass
    try:
        at = email.utils.parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        return None
    return min(RETRY_AFTER_MAX_S, max(0.0, at - time.time()))


def _call(fn, /, *a, **kw):
    """fn(*a, **kw), retried when throttled (429 / 503 SlowDown: after the Retry-After the store asks for) or on a
    transient failure (5xx, a dropped or timed-out connection: exponential backoff)."""
    from botocore.exceptions import ClientError, HTTPClientError, IncompleteReadError
    from botocore.exceptions import ConnectionError as BotoConnectionError

    for attempt in range(ATTEMPTS):
        last = attempt == ATTEMPTS - 1
        try:
            return fn(*a, **kw)
        except ClientError as e:
            meta = e.response.get("ResponseMetadata", {})
            if last or meta.get("HTTPStatusCode") not in RETRY_STATUS:
                raise
            wait = _retry_after(meta.get("HTTPHeaders", {}).get("retry-after"))
        except (BotoConnectionError, HTTPClientError, IncompleteReadError):
            if last:
                raise
            wait = None
        time.sleep(wait if wait is not None else min(30.0, 0.5 * 2**attempt))
    raise AssertionError("unreachable")


def _missing(e) -> bool:
    return e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: str) -> str:
    """A repository path, checked: never absolute, never climbing out, never the root."""
    p = path.strip().rstrip("/")
    parts = p.split("/")
    if not p or p.startswith("/") or any(s in ("", ".", "..") for s in parts):
        raise SystemExit(f"frames_store: refusing path {path!r}")
    return p


def _list(s3, prefix: str, delimiter: str | None = None):
    """Yield the ListObjectsV2 pages under prefix, each page retried on its own."""
    kw = {"Bucket": bucket(), "Prefix": prefix}
    if delimiter:
        kw["Delimiter"] = delimiter
    while True:
        page = _call(s3.list_objects_v2, **kw)
        yield page
        token = page.get("NextContinuationToken")
        if not page.get("IsTruncated") or not token:
            return
        kw["ContinuationToken"] = token


def remote_listing(s3, path: str) -> dict[str, dict]:
    """key -> {etag, size} for every file under path/. A key ending in "/" is a directory marker (a store with real
    directories, as Bunny's are underneath, may list them), not a file: seeding one would write a file where a
    directory has to go."""
    out: dict[str, dict] = {}
    for page in _list(s3, path.rstrip("/") + "/"):
        for o in page.get("Contents", []):
            if not o["Key"].endswith("/"):
                out[o["Key"]] = {"etag": o.get("ETag", "").strip('"'), "size": o["Size"]}
    return out


def remote_object(s3, key: str) -> dict[str, dict]:
    """{key: {etag, size}} if the object exists, else {}."""
    from botocore.exceptions import ClientError

    try:
        h = _call(s3.head_object, Bucket=bucket(), Key=key)
    except ClientError as e:
        if _missing(e):
            return {}
        raise
    return {key: {"etag": h.get("ETag", "").strip('"'), "size": h["ContentLength"]}}


def local_listing(root: Path, path: str) -> dict[str, Path]:
    """key -> file for the file root/path, or every regular file under the directory root/path; manifests excepted."""
    p = root / path
    files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else []
    return {q.relative_to(root).as_posix(): q for q in files if not q.name.endswith(MANIFEST_SUFFIX)}


def frame_dirs(s3, prefix: str = "assets") -> dict[str, int]:
    """Directory -> number of .webp frames in it, for everything under prefix/ (what gc_frames.py compares)."""
    counts: Counter[str] = Counter()
    for k in remote_listing(s3, prefix):
        if k.endswith(".webp"):
            counts[os.path.dirname(k)] += 1
    return dict(counts)


def _put(s3, key: str, p: Path) -> str:
    """Upload one file as a single PUT; returns the store's ETag."""
    extra = {"ContentType": CONTENT_TYPES.get(p.suffix, "application/octet-stream")}
    if os.environ.get("FRAMES_S3_CACHE_CONTROL"):
        extra["CacheControl"] = os.environ["FRAMES_S3_CACHE_CONTROL"]

    def once():
        with open(p, "rb") as f:  # reopened on every attempt: a retry re-sends the whole body
            return s3.put_object(Bucket=bucket(), Key=key, Body=f, **extra)

    return _call(once).get("ETag", "").strip('"')


def _get(s3, key: str, dst: Path) -> None:
    """Download one object to dst through a .part file, so a failed download never leaves a file that looks seeded."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")

    def once():
        body = s3.get_object(Bucket=bucket(), Key=key)["Body"]
        with open(part, "wb") as f:
            for chunk in body.iter_chunks(1 << 20):
                f.write(chunk)

    try:
        _call(once)
        os.replace(part, dst)
    finally:
        part.unlink(missing_ok=True)


def _delete(s3, keys: list[str]) -> None:
    """Single deletes in parallel (Bunny answers the batch DeleteObjects with 501)."""
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda k: _call(s3.delete_object, Bucket=bucket(), Key=k), keys))


def _summary(keys: list[str], files: dict[str, Path], top: int = 30) -> None:
    """What a dry run would upload, by directory (a migration names ~28,000 files)."""
    count: Counter[str] = Counter()
    size: Counter[str] = Counter()
    for k in keys:
        count[os.path.dirname(k)] += 1
        size[os.path.dirname(k)] += files[k].stat().st_size
    for d, n in count.most_common(top):
        print(f"      {n:6d} {d}  ({size[d] / 1e6:.1f} MB)")
    if len(count) > top:
        print(f"      ... and {len(count) - top} more directories")
    if keys:
        print(f"      total {sum(size.values()) / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
def cmd_sync(paths: list[str], root: Path, no_replace: bool, dry_run: bool) -> int:
    s3 = client()
    total_up = total_del = 0
    not_md5: list[str] = []
    for arg in paths:
        merge = arg.startswith("+")
        d = _rel(arg.removeprefix("+"))
        loc = local_listing(root, d)
        if not loc:
            # A product that failed to render must not blank its own frames.
            print(f"  {d}: nothing rendered locally, store copy kept")
            continue
        rem = remote_object(s3, d) if (root / d).is_file() else remote_listing(s3, d)
        md5: dict[str, str] = {}

        def digest(k: str, md5=md5, loc=loc) -> str:
            if k not in md5:
                md5[k] = _md5(loc[k])
            return md5[k]

        new = [k for k in loc if k not in rem]
        changed = [
            k for k in loc if k in rem and (rem[k]["size"] != loc[k].stat().st_size or rem[k]["etag"] != digest(k))
        ]
        to_up = new if no_replace else new + changed
        to_del = [] if merge else sorted(k for k in rem if k not in loc)
        mode = "+" if merge else ""
        if dry_run:
            kept = f", {len(changed)} differing kept" if no_replace and changed else ""
            print(f"  {mode}{d}: {len(loc)} files, would upload {len(to_up)}, would remove {len(to_del)}{kept}")
            _summary(to_up, loc)
            continue

        def put(k: str, loc=loc, digest=digest) -> bool:
            return _put(s3, k, loc[k]) == digest(k)

        # uploads first, deletes after: a viewer never meets a gap
        with ThreadPoolExecutor(WORKERS) as ex:
            same = list(ex.map(put, to_up))
        not_md5 += [k for k, ok in zip(to_up, same) if not ok]
        _delete(s3, to_del)
        print(f"  {mode}{d}: {len(loc)} files, {len(to_up)} uploaded, {len(to_del)} removed")
        total_up += len(to_up)
        total_del += len(to_del)
    print(f"  synced: {total_up} uploaded, {total_del} removed")
    if not_md5:
        print(
            f"::warning::frames store: {len(not_md5)} of {total_up} uploads came back with an ETag that is not the "
            "file's MD5, so unchanged files cannot be recognised and are re-sent on every sync"
        )
    return 0


def cmd_seed(dirs: list[str], root: Path) -> int:
    """store -> local, never overwriting what this run has already produced."""
    s3 = client()
    failed = 0
    for arg in dirs:
        d = _rel(arg)
        rem = remote_listing(s3, d)
        if not rem:
            print(f"  {d}: not in the store yet")
            continue
        want = [k for k in rem if not (root / k).exists()]

        def get(k: str) -> bool:
            from botocore.exceptions import BotoCoreError, ClientError

            try:
                _get(s3, k, root / k)
            except (BotoCoreError, ClientError, OSError) as e:
                print(f"    {k}: {str(e)[:160]}")
                return False
            return True

        with ThreadPoolExecutor(WORKERS) as ex:
            ok = list(ex.map(get, want))
        failed += ok.count(False)
        print(f"  {d}: seeded {ok.count(True)} files")
    if failed:
        print(f"::warning::frames store: {failed} file(s) could not be seeded")
    return 1 if failed else 0


def cmd_prune(paths: list[str], dry_run: bool) -> int:
    """Delete each path from the store: the object of that name (a single file) and everything under path/."""
    s3 = client()
    for arg in paths:
        p = _rel(arg)
        keys = sorted(remote_listing(s3, p)) + list(remote_object(s3, p))
        if not dry_run:
            _delete(s3, keys)
        print(f"  {'would prune' if dry_run else 'pruned'} {p} ({len(keys)} files)")
    return 0


def cmd_prune_cycles(specs: list[str], dry_run: bool) -> int:
    """For each DIR:YYYYMMDDHH, delete every child directory of DIR named as a cycle older than that floor - the
    store's PRUNE_OLDER (scripts/models/publish_data.sh). Listed in the store itself, so orphans go too."""
    s3 = client()
    for spec in specs:
        d, _, floor = spec.rpartition(":")
        if not d or not CYCLE.fullmatch(floor):
            raise SystemExit(f"frames_store: prune-cycles {spec!r}: expected DIR:YYYYMMDDHH")
        d = _rel(d)
        old = []
        for page in _list(s3, d + "/", delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                if CYCLE.fullmatch(name) and name < floor:
                    old.append(name)
        for name in sorted(old):
            keys = sorted(remote_listing(s3, f"{d}/{name}"))
            if not dry_run:
                _delete(s3, keys)
            print(f"  prune (older than {floor}): {d}/{name} ({len(keys)} files)")
    return 0


def cmd_ls(prefix: str) -> int:
    counts = frame_dirs(client(), _rel(prefix))
    for d in sorted(counts):
        print(f"{counts[d]:6d} {d}")
    return 0


def cmd_check() -> int:
    """put / head / list / get (whole and a range) / delete on a scratch key, after finding the regions that sign.

    The PUT is the one sync makes (same headers), so a header the store refuses shows up here first."""
    from botocore.exceptions import BotoCoreError, ClientError

    host = urlparse(os.environ["FRAMES_S3_ENDPOINT"]).hostname or ""
    label = host.split(".")[0].removesuffix("-s3")  # de-s3.storage.bunnycdn.com -> de
    candidates = list(dict.fromkeys(r for r in (os.environ.get("FRAMES_S3_REGION"), label, "us-east-1", "auto") if r))
    signs = []
    for r in candidates:
        try:
            _call(client(r).list_objects_v2, Bucket=bucket(), Prefix=CHECK_PREFIX + "/", MaxKeys=1)
        except (BotoCoreError, ClientError) as e:
            print(f"  region {r!r}: {str(e)[:200]}")
            continue
        print(f"  region {r!r}: signs (ListObjectsV2 answered)")
        signs.append(r)
    if not signs:
        print("  no candidate region signs: check the endpoint, bucket and credentials")
        return 1
    region = os.environ.get("FRAMES_S3_REGION") or "us-east-1"
    if region not in signs:
        print(f"  FRAMES_S3_REGION ({region!r}) does not sign; set it to {signs[0]!r}")
        region = signs[0]
    s3 = client(region)

    token = secrets.token_hex(6)
    key = f"{CHECK_PREFIX}/{token}/probe.bin"
    body = os.urandom(200_000)
    md5 = hashlib.md5(body).hexdigest()
    results: dict[str, bool] = {}

    def step(name: str, fn) -> None:
        try:
            ok, note = fn()
        except (BotoCoreError, ClientError, OSError) as e:
            ok, note = False, str(e)[:200]
        results[name] = ok
        print(f"  {name:6s} {'ok  ' if ok else 'FAIL'} {note}")

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "probe.bin"
        src.write_bytes(body)
        etag: dict[str, str] = {}

        def put():
            t0 = time.time()
            etag["put"] = _put(s3, key, src)
            return True, f"{len(body)} bytes in {time.time() - t0:.2f} s, ETag {etag['put']}"

        def head():
            h = _call(s3.head_object, Bucket=bucket(), Key=key)
            tag = h.get("ETag", "").strip('"')
            note = f"length {h['ContentLength']}, type {h.get('ContentType')}, ETag {tag}"
            return h["ContentLength"] == len(body), note

        def listing():
            found = remote_listing(s3, f"{CHECK_PREFIX}/{token}").get(key)
            children = [
                cp["Prefix"]
                for page in _list(s3, CHECK_PREFIX + "/", delimiter="/")
                for cp in page.get("CommonPrefixes", [])
            ]
            ok = found is not None and found["size"] == len(body) and f"{CHECK_PREFIX}/{token}/" in children
            return ok, f"listed {found}; delimiter '/' gives {len(children)} prefix(es)"

        def get():
            def read():  # the body is read inside the retried call, so a dropped stream is retried too
                return s3.get_object(Bucket=bucket(), Key=key)["Body"].read()

            whole = _call(read)
            r = _call(s3.get_object, Bucket=bucket(), Key=key, Range="bytes=1000-1999")
            part = r["Body"].read()
            ok = whole == body and part == body[1000:2000]
            return ok, f"whole {len(whole)} bytes, range -> {len(part)} bytes ({r.get('ContentRange')})"

        def delete():
            _call(s3.delete_object, Bucket=bucket(), Key=key)
            return not remote_object(s3, key), "the object is gone after a single DeleteObject"

        for name, fn in (("put", put), ("head", head), ("list", listing), ("get", get), ("delete", delete)):
            step(name, fn)
        if "put" in etag:
            same = etag["put"] == md5
            print(
                f"  ETag {'IS' if same else 'is NOT'} the MD5 of the body ({md5}): "
                + ("unchanged files are recognised and skipped" if same else "every sync re-sends its files in full")
            )
    print(f"  region to use: FRAMES_S3_REGION={region}")
    return 0 if all(results.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    root = argparse.ArgumentParser(add_help=False)
    root.add_argument("--root", type=Path, default=Path("."), help="the directory local paths are relative to")
    dry = argparse.ArgumentParser(add_help=False)
    dry.add_argument("--dry-run", action="store_true", help="list what would change, change nothing")
    p = sub.add_parser("sync", parents=[root, dry])
    p.add_argument("dirs", nargs="+", help="dir (made exactly the local copy) or +dir (added to, never deleted from)")
    p.add_argument("--no-replace", action="store_true", help="upload only files the store does not have")
    p = sub.add_parser("seed", parents=[root])
    p.add_argument("dirs", nargs="+")
    p = sub.add_parser("prune", parents=[dry])
    p.add_argument("paths", nargs="+")
    p = sub.add_parser("prune-cycles", parents=[dry])
    p.add_argument("specs", nargs="+", metavar="DIR:YYYYMMDDHH")
    p = sub.add_parser("ls")
    p.add_argument("prefix", nargs="?", default="assets")
    sub.add_parser("check")
    sub.add_parser("configured")
    a = ap.parse_args()

    if a.cmd == "configured":
        return 0 if configured() else 1
    if not configured():
        print("frames_store: " + " / ".join(CREDENTIALS) + " not set", file=sys.stderr)
        return 2
    t0 = time.time()
    rc = {
        "sync": lambda: cmd_sync(a.dirs, a.root.resolve(), a.no_replace, a.dry_run),
        "seed": lambda: cmd_seed(a.dirs, a.root.resolve()),
        "prune": lambda: cmd_prune(a.paths, a.dry_run),
        "prune-cycles": lambda: cmd_prune_cycles(a.specs, a.dry_run),
        "ls": lambda: cmd_ls(a.prefix),
        "check": cmd_check,
    }[a.cmd]()
    print(f"  ({time.time() - t0:.1f}s)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
