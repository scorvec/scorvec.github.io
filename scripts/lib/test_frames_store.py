#!/usr/bin/env python3
"""Offline checks of the frames store client (frames_store.py) and of the publish / fetch wrappers around it.

    python scripts/lib/test_frames_store.py

The store is a local stub of Bunny Storage's S3 API as documented: path-style requests; PutObject, GetObject (with
ranges), HeadObject, single DeleteObject, ListObjectsV2 (prefix, delimiter "/", pagination - a few keys a page here, so
every listing pages); DeleteObjects answered 501 NotImplemented. On demand it throttles (503 SlowDown with a
Retry-After) or hands out ETags that are not the MD5. The branch is a local bare repository standing in for GitHub
(FRAMES_URL), so nothing here touches the network; boto3 is the only dependency.

  frames_store.py   sync (add / replace / delete, "+dir", a single file, sibling prefixes, dry run, --no-replace),
                    seed (and --exclude, directory markers), prune (directories and single files), prune-cycles, ls,
                    check, Retry-After, non-MD5 ETags, Cache-Control only on request
  the wrappers      FRAMES_S3_* empty: the branch alone, the store never consulted (publish_frames_ci.sh,
                    fetch_frames_ci.sh, publish_data.sh); store configured: both (seeds still from the branch);
                    FRAMES_BRANCH=off: the store alone, git never touched; off without a usable store: a refusal, and
                    no branch named "off"; another data branch (verify-data): never the store
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames_store as S

LIB = HERE
MODELS = HERE.parent / "models"
STORE_ENV = ("FRAMES_S3_BUCKET", "FRAMES_S3_ENDPOINT", "FRAMES_S3_ACCESS_KEY_ID", "FRAMES_S3_SECRET_ACCESS_KEY")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ── the store ───────────────────────────────────────────────────────────────────────────────────────────────────
class Stub:
    """Bunny Storage's S3 API, as far as frames_store.py uses it."""

    def __init__(self, bucket: str = "zone", page: int = 3):
        self.bucket, self.page = bucket, page
        self.objects: dict[str, dict] = {}  # key -> {body, type, headers}
        self.log: list[tuple[str, str, str]] = []  # (method, key, query)
        self.slowdown = 0  # answer the next N requests with 503 SlowDown, Retry-After: 1
        self.md5_etags = True
        self.lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def reply(self, status, body=b"", headers=None):
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def fail(self, status, code, headers=None):
                body = (
                    f"<?xml version='1.0' encoding='UTF-8'?><Error><Code>{code}</Code><Message>{code}</Message></Error>"
                )
                self.reply(status, body.encode(), {"Content-Type": "application/xml", **(headers or {})})

            def incoming(self):
                """(key, query, body), or None once a fault has been answered."""
                u = urlsplit(self.path)
                bucket, _, key = u.path.lstrip("/").partition("/")  # path-style: /<bucket>/<key>
                q = parse_qs(u.query, keep_blank_values=True)
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                with stub.lock:
                    stub.log.append((self.command, unquote(key), u.query))
                    slow = stub.slowdown > 0
                    stub.slowdown -= slow
                if slow:
                    self.fail(503, "SlowDown", {"Retry-After": "1"})
                    return None
                if unquote(bucket) != stub.bucket:
                    self.fail(404, "NoSuchBucket")
                    return None
                return unquote(key), q, body

            def etag(self, body: bytes) -> str:
                h = hashlib.md5(body if stub.md5_etags else body + b"!").hexdigest()
                return f'"{h}"'

            def do_PUT(self):
                r = self.incoming()
                if r:
                    key, _, body = r
                    with stub.lock:
                        stub.objects[key] = {
                            "body": body,
                            "type": self.headers.get("Content-Type", ""),
                            "headers": {k.lower(): v for k, v in self.headers.items()},
                        }
                    self.reply(200, b"", {"ETag": self.etag(body)})

            def do_HEAD(self):
                r = self.incoming()
                if r:
                    o = stub.objects.get(r[0])
                    if o is None:
                        self.reply(404)
                    else:
                        self.reply(200, o["body"], {"ETag": self.etag(o["body"]), "Content-Type": o["type"]})

            def do_GET(self):
                r = self.incoming()
                if not r:
                    return
                key, q, _ = r
                if not key:
                    self.listing(q)
                    return
                o = stub.objects.get(key)
                if o is None:
                    self.fail(404, "NoSuchKey")
                    return
                body, rng = o["body"], self.headers.get("Range")
                if rng:
                    a, b = (int(x) for x in rng.removeprefix("bytes=").split("-"))
                    part = body[a : b + 1]
                    self.reply(206, part, {"Content-Range": f"bytes {a}-{a + len(part) - 1}/{len(body)}"})
                else:
                    self.reply(200, body, {"ETag": self.etag(body), "Content-Type": o["type"]})

            def do_DELETE(self):
                r = self.incoming()
                if r:
                    with stub.lock:
                        stub.objects.pop(r[0], None)
                    self.reply(204)

            def do_POST(self):
                r = self.incoming()
                if r:
                    self.fail(501, "NotImplemented")  # DeleteObjects (?delete) and anything else

            def listing(self, q):
                arg = {k: v[0] for k, v in q.items()}
                prefix, delim = arg.get("prefix", ""), arg.get("delimiter", "")
                if delim not in ("", "/"):
                    self.fail(400, "InvalidArgument")
                    return
                url = arg.get("encoding-type") == "url"
                size = min(int(arg.get("max-keys", 1000)), stub.page)
                entries = set()
                for k in stub.objects:
                    if k.startswith(prefix):
                        rest = k[len(prefix) :]
                        entries.add((prefix + rest.split("/")[0] + "/", True) if delim and "/" in rest else (k, False))
                entries = sorted(e for e in entries if e[0] > arg.get("continuation-token", ""))
                page, more = entries[:size], len(entries) > size
                enc = (lambda s: escape(quote(s, safe="/"))) if url else escape
                xml = ["<?xml version='1.0' encoding='UTF-8'?><ListBucketResult>"]
                xml.append(f"<Name>{stub.bucket}</Name><Prefix>{enc(prefix)}</Prefix><KeyCount>{len(page)}</KeyCount>")
                xml.append(f"<MaxKeys>{size}</MaxKeys><IsTruncated>{'true' if more else 'false'}</IsTruncated>")
                if url:
                    xml.append("<EncodingType>url</EncodingType>")
                if more:
                    xml.append(f"<NextContinuationToken>{enc(page[-1][0])}</NextContinuationToken>")
                for name, is_prefix in page:
                    if is_prefix:
                        xml.append(f"<CommonPrefixes><Prefix>{enc(name)}</Prefix></CommonPrefixes>")
                    else:
                        b = stub.objects[name]["body"]
                        xml.append(
                            f"<Contents><Key>{enc(name)}</Key><LastModified>2026-10-22T00:00:00.000Z</LastModified>"
                            f"<ETag>{escape(self.etag(b))}</ETag><Size>{len(b)}</Size></Contents>"
                        )
                xml.append("</ListBucketResult>")
                self.reply(200, "".join(xml).encode(), {"Content-Type": "application/xml"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def env(self) -> dict[str, str]:
        return {
            "FRAMES_S3_BUCKET": self.bucket,
            "FRAMES_S3_ENDPOINT": self.url,
            "FRAMES_S3_ACCESS_KEY_ID": self.bucket,
            "FRAMES_S3_SECRET_ACCESS_KEY": "zone-password",
        }

    def listed(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self.objects if k.startswith(prefix))

    def body(self, key: str) -> bytes:
        return self.objects[key]["body"]

    def put(self, key: str, body: bytes) -> None:
        self.objects[key] = {"body": body, "type": "", "headers": {}}

    def writes(self, since: int = 0) -> list[tuple[str, str]]:
        return [(m, k) for m, k, _ in self.log[since:] if m in ("PUT", "DELETE", "POST")]


def write(root: Path, rel: str, data: bytes | str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data.encode() if isinstance(data, str) else data)
    return p


def quiet(fn, *a, **kw) -> tuple[object, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = fn(*a, **kw)
    return rc, out.getvalue()


# ── frames_store.py ─────────────────────────────────────────────────────────────────────────────────────────────
def test_client(stub: Stub) -> None:
    os.environ.update(stub.env())
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        loop = "assets/sst/anim/loop"

        # sync: add; manifests stay out of the store; image/webp; no Cache-Control unless asked for
        for i in range(3):
            write(root, f"{loop}/F0{i}.webp", f"frame {i}")
        write(root, f"{loop}/loop_manifest.json", "{}")
        mark = len(stub.log)
        quiet(S.cmd_sync, [loop], root, False, False)
        check(stub.listed(loop) == [f"{loop}/F00.webp", f"{loop}/F01.webp", f"{loop}/F02.webp"], "sync add")
        check(stub.objects[f"{loop}/F00.webp"]["type"] == "image/webp", "content type")
        check("cache-control" not in stub.objects[f"{loop}/F00.webp"]["headers"], "no Cache-Control by default")
        check(len(stub.writes(mark)) == 3, "three uploads")

        # sync: replace one, delete one, add one; the unchanged frame is skipped (ETag = MD5)
        write(root, f"{loop}/F01.webp", "frame 1, re-rendered")
        (root / f"{loop}/F02.webp").unlink()
        write(root, f"{loop}/F03.webp", "frame 3")
        mark = len(stub.log)
        _, out = quiet(S.cmd_sync, [loop], root, False, False)
        check(stub.listed(loop) == [f"{loop}/F00.webp", f"{loop}/F01.webp", f"{loop}/F03.webp"], "sync swap")
        check(stub.body(f"{loop}/F01.webp") == b"frame 1, re-rendered", "replaced")
        w = stub.writes(mark)
        check(
            sorted(w) == [("DELETE", f"{loop}/F02.webp"), ("PUT", f"{loop}/F01.webp"), ("PUT", f"{loop}/F03.webp")], w
        )
        check(w[-1][0] == "DELETE", "uploads before deletes")
        check("2 uploaded, 1 removed" in out, out)

        # "+dir": adds and replaces, never deletes
        arch = "assets/sst/anim/arch"
        stub.put(f"{arch}/2026-01-01.npz", b"old day")
        stub.put(f"{arch}/2026-01-02.npz", b"day 2")
        write(root, f"{arch}/2026-01-02.npz", "day 2, revised")
        write(root, f"{arch}/2026-01-03.npz", "day 3")
        quiet(S.cmd_sync, ["+" + arch], root, False, False)
        check(stub.listed(arch) == [f"{arch}/2026-01-0{d}.npz" for d in (1, 2, 3)], "+dir deletes nothing")
        check(stub.body(f"{arch}/2026-01-02.npz") == b"day 2, revised", "+dir replaces")

        # nothing rendered: the store copy is kept
        (root / "assets/sst/anim/empty").mkdir(parents=True)
        stub.put("assets/sst/anim/empty/F00.webp", b"keep me")
        _, out = quiet(S.cmd_sync, ["assets/sst/anim/empty", "assets/sst/anim/missing"], root, False, False)
        check("assets/sst/anim/empty/F00.webp" in stub.objects, "empty dir keeps the store copy")
        check(out.count("nothing rendered locally") == 2, out)

        # a single file, then unchanged
        write(root, "assets/site/thumbs/index.json", '{"a": 1}')
        quiet(S.cmd_sync, ["+assets/site/thumbs/index.json"], root, False, False)
        check(stub.objects["assets/site/thumbs/index.json"]["type"] == "application/json", "file arg")
        mark = len(stub.log)
        quiet(S.cmd_sync, ["assets/site/thumbs/index.json"], root, False, False)
        check(stub.writes(mark) == [], "an unchanged file is not re-sent")

        # a swap of dt never touches its sibling dt_npac (prefixes end at "/")
        stub.put("assets/sst/anim/dt/F00.webp", b"x")
        stub.put("assets/sst/anim/dt_npac/F00.webp", b"y")
        write(root, "assets/sst/anim/dt/F01.webp", "z")
        quiet(S.cmd_sync, ["assets/sst/anim/dt"], root, False, False)
        check(
            stub.listed("assets/sst/anim/dt") == ["assets/sst/anim/dt/F01.webp", "assets/sst/anim/dt_npac/F00.webp"],
            "dt",
        )

        # dry run: writes nothing
        write(root, f"{loop}/F04.webp", "frame 4")
        mark = len(stub.log)
        _, out = quiet(S.cmd_sync, [loop], root, False, True)
        check(stub.writes(mark) == [] and "would upload 1" in out, out)

        # --no-replace: only what the store lacks (the migration under a dual write)
        stub.put("assets/mig/a.webp", b"newer, from a publisher")
        write(root, "assets/mig/a.webp", "older, from the branch tip")
        write(root, "assets/mig/b.webp", "only on the branch")
        _, out = quiet(S.cmd_sync, ["+assets/mig"], root, True, False)
        check(stub.body("assets/mig/a.webp") == b"newer, from a publisher", "--no-replace keeps the store's file")
        check(stub.body("assets/mig/b.webp") == b"only on the branch", "--no-replace uploads the missing one")

        # Cache-Control only when FRAMES_S3_CACHE_CONTROL asks for it
        os.environ["FRAMES_S3_CACHE_CONTROL"] = "public, max-age=60"
        write(root, "assets/cc/F00.webp", "cc")
        quiet(S.cmd_sync, ["assets/cc"], root, False, False)
        del os.environ["FRAMES_S3_CACHE_CONTROL"]
        check(stub.objects["assets/cc/F00.webp"]["headers"].get("cache-control") == "public, max-age=60", "cc")

    # seed: never overwrites, skips directory markers; --exclude seeds only the other child directories
    d = "assets/geps/anim"
    stub.put(f"{d}/top.json", b"top")
    stub.put(f"{d}/c1/f.webp", b"c1")
    stub.put(f"{d}/c2/g.webp", b"c2 from the store")
    stub.put(f"{d}/c3/", b"")  # a directory marker
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write(root, f"{d}/c2/g.webp", "rendered by this run")
        rc, out = quiet(S.cmd_seed, [d, "assets/not_there"], root, None)
        check(rc == 0, out)
        check((root / f"{d}/top.json").read_bytes() == b"top" and (root / f"{d}/c1/f.webp").exists(), "seeded")
        check((root / f"{d}/c2/g.webp").read_bytes() == b"rendered by this run", "seed never overwrites")
        check(not (root / f"{d}/c3").is_file(), "a directory marker is not a file")
        check("not in the store yet" in out and not list(root.rglob("*.part")), out)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        quiet(S.cmd_seed, [d], root, "c1")
        got = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        check(got == [f"{d}/c2/g.webp"], f"--exclude: {got}")

    # prune: a directory and a single file; prune-cycles: cycle directories older than the floor
    for k in ("x/dir/a", "x/dir/b", "x/dir_other/a", "x/file.webp"):
        stub.put(f"assets/{k}", b"p")
    quiet(S.cmd_prune, ["assets/x/dir", "assets/x/file.webp"], False)
    check(stub.listed("assets/x/") == ["assets/x/dir_other/a"], stub.listed("assets/x/"))
    h = "assets/models/data/hrrr"
    for c in ("2026102000", "2026102006", "2026102012", "geo"):
        stub.put(f"{h}/{c}/t2m/000.webp", b"m")
        stub.put(f"{h}/{c}/t2m/001.webp", b"m")
    quiet(S.cmd_prune_cycles, [f"{h}:2026102012"], False)
    check(sorted({k.split("/")[4] for k in stub.listed(h)}) == ["2026102012", "geo"], stub.listed(h))
    try:
        S.cmd_prune_cycles([f"{h}:20261020"], False)
        check(False, "a bad floor must be refused")
    except SystemExit:
        pass

    # ls: frame directories and counts
    _, out = quiet(S.cmd_ls, h)
    check(out.split() == ["2", f"{h}/2026102012/t2m", "2", f"{h}/geo/t2m"], out)

    # paths are repository paths
    for bad in ("../x", "/abs", ".", "a//b", ""):
        try:
            S._rel(bad)
            check(False, f"accepted {bad!r}")
        except SystemExit:
            pass

    # throttling: two 503 SlowDown answers, each waited out for its Retry-After, then success
    with tempfile.TemporaryDirectory() as td:
        write(Path(td), "assets/slow/F00.webp", "s")
        stub.slowdown = 2
        t0 = time.time()
        quiet(S.cmd_sync, ["assets/slow"], Path(td), False, False)
        check(time.time() - t0 >= 1.9 and "assets/slow/F00.webp" in stub.objects, "Retry-After honoured")

    # DeleteObjects is 501 here, as on Bunny, and frames_store never sends it
    from botocore.exceptions import ClientError

    try:
        S.client().delete_objects(Bucket=stub.bucket, Delete={"Objects": [{"Key": "assets/mig/a.webp"}]})
        check(False, "DeleteObjects must fail")
    except ClientError as e:
        check(e.response["Error"]["Code"] == "NotImplemented", e.response)
    check(sum(1 for m, _, q in stub.log if m == "POST" and q.startswith("delete")) == 1, "only the probe above")

    # check: the round trip, with the store's ETag the MD5 and then not
    rc, out = quiet(S.cmd_check)
    check(rc == 0 and "ETag IS the MD5" in out and not stub.listed(S.CHECK_PREFIX), out)
    stub.md5_etags = False
    rc, out = quiet(S.cmd_check)
    check(rc == 0 and "ETag is NOT the MD5" in out, out)

    # an ETag that is not the MD5: correct, every file re-sent, and the summary says so
    with tempfile.TemporaryDirectory() as td:
        write(Path(td), "assets/nomd5/F00.webp", "a")
        write(Path(td), "assets/nomd5/F01.webp", "b")
        _, out = quiet(S.cmd_sync, ["assets/nomd5"], Path(td), False, False)
        check("::warning::" in out and "not the file's MD5" in out, out)
        mark = len(stub.log)
        quiet(S.cmd_sync, ["assets/nomd5"], Path(td), False, False)
        check(len(stub.writes(mark)) == 2 and stub.body("assets/nomd5/F01.webp") == b"b", "re-sent, still correct")
    stub.md5_etags = True

    for k in STORE_ENV:
        del os.environ[k]
    check(not S.configured(), "configured() without the variables")


# ── the shell wrappers ──────────────────────────────────────────────────────────────────────────────────────────
GIT_ENV = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_CONFIG_NOSYSTEM": "1",
}
BRANCH_FILES = {
    "assets/sst/anim/loop/F00.webp": "branch loop 0",
    "assets/sst/anim/loop/F01.webp": "branch loop 1",
    "assets/sst/anim/loop/F02.webp": "branch loop 2",
    "assets/sst/anim/arch/2026-01-01.npz": "arch 1",
    "assets/sst/anim/arch/2026-01-02.npz": "arch 2",
    "assets/sst/anim/old/F00.webp": "retired",
    "assets/models/data/hrrr/2026102000/t2m/000.webp": "c00",
    "assets/models/data/hrrr/2026102006/t2m/000.webp": "c06",
    "scripts/sst/data/cmems/store.nc": "cmems",
}


def sh(*a: str, cwd: Path | None = None, env: dict | None = None, check_rc: bool = True) -> str:
    r = subprocess.run(a, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if check_rc and r.returncode:
        raise AssertionError(f"{a}: {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r.stdout


class Site:
    """A bare repository with a `frames` and a `verify-data` branch, a runner workspace, and the environment."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.remote = tmp / "remote.git"
        self.ws = tmp / "ws"
        self.ws.mkdir()
        home = tmp / "home"
        home.mkdir()
        shim = tmp / "bin"
        shim.mkdir()
        if not shutil.which("timeout"):  # coreutils' timeout, which the publisher uses around a push
            (shim / "timeout").write_text('#!/bin/sh\nshift\nexec "$@"\n')
            (shim / "timeout").chmod(0o755)
        self.calls = tmp / "frames_py_calls.txt"
        (shim / "frames_py_spy").write_text(f'#!/bin/sh\necho "$@" >> "{self.calls}"\nexit 1\n')
        (shim / "frames_py_spy").chmod(0o755)
        path = os.pathsep.join([str(shim), str(Path(sys.executable).parent), os.environ.get("PATH", "")])
        self.env = {
            "PATH": path,
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "GH_TOKEN": "unused",
            "GITHUB_REPOSITORY": "owner/repo",
            "FRAMES_URL": self.remote.as_uri(),
            "FRAMES_WARM": "0",
            **GIT_ENV,
        }
        sh("git", "init", "-q", "--bare", str(self.remote), env=self.env)
        sh("git", "-C", str(self.remote), "config", "uploadpack.allowFilter", "true", env=self.env)
        sh("git", "-C", str(self.remote), "config", "uploadpack.allowAnySHA1InWant", "true", env=self.env)
        self.seed_branch("frames", BRANCH_FILES)
        self.seed_branch("verify-data", {"scripts/verify/data/truth/2026-01-01.json": "{}"})

    def seed_branch(self, name: str, files: dict[str, str]) -> None:
        w = self.tmp / f"seed_{name}"
        sh("git", "init", "-q", "-b", name, str(w), env=self.env)
        for rel, text in files.items():
            write(w, rel, text)
        sh("git", "-C", str(w), "add", "-A", env=self.env)
        sh("git", "-C", str(w), "commit", "-q", "-m", "seed", env=self.env)
        sh("git", "-C", str(w), "push", "-q", str(self.remote), name, env=self.env)

    def run(self, script: Path, *args: str, env: dict | None = None, cwd: Path | None = None):
        e = {**self.env, **(env or {})}
        return subprocess.run(
            [shutil.which("bash") or "bash", str(script), *args],
            cwd=cwd or self.ws,
            env=e,
            capture_output=True,
            text=True,
            check=False,
        )

    def tree(self, ref: str = "frames") -> dict[str, str]:
        out = sh("git", "-C", str(self.remote), "ls-tree", "-r", "--full-tree", ref, env=self.env)
        files = {}
        for ln in out.splitlines():
            meta, path = ln.split("\t", 1)
            files[path] = sh("git", "-C", str(self.remote), "cat-file", "blob", meta.split()[2], env=self.env)
        return files

    def tree_id(self, ref: str = "frames") -> str:
        return sh("git", "-C", str(self.remote), "rev-parse", f"{ref}^{{tree}}", env=self.env).strip()

    def refs(self) -> list[str]:
        return sh("git", "-C", str(self.remote), "for-each-ref", "--format=%(refname)", env=self.env).split()


def render(ws: Path) -> None:
    """One run's output: a loop re-rendered whole (fewer frames than before), a day added to an archive."""
    write(ws, "assets/sst/anim/loop/F00.webp", "new loop 0")
    write(ws, "assets/sst/anim/loop/F01.webp", "new loop 1")
    write(ws, "assets/sst/anim/loop_manifest.json", '{"regions": {"loop": {}}}')
    write(ws, "assets/sst/anim/arch/2026-01-03.npz", "arch 3")


def publish_all(site: Site, lib: Path, models: Path, env: dict) -> dict:
    """publish_frames_ci.sh (swap, "+dir", PRUNE), publish_data.sh (a cycle, PRUNE_OLDER) and fetch_frames_ci.sh."""
    render(site.ws)
    r1 = site.run(
        lib / "publish_frames_ci.sh",
        "assets/sst/anim/loop",
        "+assets/sst/anim/arch",
        env={**env, "PRUNE": "assets/sst/anim/old"},
    )
    write(site.ws, "assets/models/data/hrrr/2026102012/t2m/000.webp", "c12")
    r2 = site.run(
        models / "publish_data.sh",
        "assets/models/data/hrrr/2026102012/t2m",
        env={**env, "SRC": str(site.ws), "PRUNE_OLDER": "assets/models/data/hrrr:2026102006"},
    )
    fresh = site.tmp / "fresh"
    fresh.mkdir(exist_ok=True)
    r3 = site.run(lib / "fetch_frames_ci.sh", "assets/sst/anim/arch", "scripts/sst/data/cmems", env=env, cwd=fresh)
    seeded = sorted(p.relative_to(fresh).as_posix() for p in fresh.rglob("*") if p.is_file())
    return {"publish": r1, "data": r2, "fetch": r3, "seeded": seeded}


EXPECTED = {
    "assets/sst/anim/loop/F00.webp": "new loop 0",
    "assets/sst/anim/loop/F01.webp": "new loop 1",
    "assets/sst/anim/arch/2026-01-01.npz": "arch 1",
    "assets/sst/anim/arch/2026-01-02.npz": "arch 2",
    "assets/sst/anim/arch/2026-01-03.npz": "arch 3",
    "assets/models/data/hrrr/2026102006/t2m/000.webp": "c06",
    "assets/models/data/hrrr/2026102012/t2m/000.webp": "c12",
    "scripts/sst/data/cmems/store.nc": "cmems",
}
SEEDED = [
    "assets/sst/anim/arch/2026-01-01.npz",
    "assets/sst/anim/arch/2026-01-02.npz",
    "assets/sst/anim/arch/2026-01-03.npz",
    "scripts/sst/data/cmems/store.nc",
]


def ok(r, what: str) -> None:
    check(r.returncode == 0, f"{what}: exit {r.returncode}\n{r.stdout}\n{r.stderr}")


def inert(lib: Path = LIB, models: Path = MODELS) -> dict:
    """FRAMES_S3_* empty: the branch alone. Returns what a run produced, for comparison with another copy."""
    with tempfile.TemporaryDirectory() as td:
        site = Site(Path(td))
        env = dict.fromkeys(STORE_ENV, "") | {"FRAMES_PY": str(site.tmp / "bin" / "frames_py_spy")}
        res = publish_all(site, lib, models, env)
        for k in ("publish", "data", "fetch"):
            ok(res[k], k)
        stamped = (site.ws / "assets/sst/anim/loop_manifest.json").read_text()
        return {
            "tree": site.tree(),
            "tree_id": site.tree_id(),
            "pins": [r for r in site.refs() if r.startswith("refs/frames-pins/")],
            "tip": sh("git", "-C", str(site.remote), "rev-parse", "frames", env=site.env).strip(),
            "sha_line": res["data"].stdout.splitlines()[-1],
            "stamped": stamped,
            "seeded": res["seeded"],
            "store_consulted": site.calls.exists(),
        }


def test_wrappers(stub: Stub) -> None:
    # 1. FRAMES_S3_* empty: today's path - the branch alone, the store's python never even started
    r = inert()
    check(r["tree"] == EXPECTED, f"branch tree: {r['tree']}")
    check(not r["store_consulted"], "the store's python was run with FRAMES_S3_* empty")
    check(r["sha_line"] == r["tip"], f"publish_data.sh's last line is the new tip: {r}")
    sha = json.loads(r["stamped"])["frames_sha"]  # the manifest names the commit publish_frames_ci.sh pushed, pinned
    check(r["pins"] == [f"refs/frames-pins/{sha}"], r["pins"])
    check(r["seeded"] == SEEDED, r["seeded"])

    # 2. store configured (dual): both get the same edit; seeds still come from the branch
    with tempfile.TemporaryDirectory() as td:
        site = Site(Path(td))
        stub.objects.clear()
        for rel, text in BRANCH_FILES.items():  # as after the migration
            stub.put(rel, text.encode())
        mark = len(stub.log)
        res = publish_all(site, LIB, MODELS, stub.env())
        for k in ("publish", "data", "fetch"):
            ok(res[k], k)
        check(site.tree() == EXPECTED, f"dual: branch {site.tree()}")
        check({k: stub.body(k).decode() for k in stub.listed()} == EXPECTED, f"dual: store {stub.listed()}")
        tip = sh("git", "-C", str(site.remote), "rev-parse", "frames", env=site.env).strip()
        check(res["data"].stdout.splitlines()[-1] == tip, "dual: publish_data.sh's last line is the new tip")
        fetch_gets = [k for m, k, _ in stub.log[mark:] if m == "GET" and k]
        check(not fetch_gets and res["seeded"] == SEEDED, f"dual seeds from the branch: {fetch_gets}")

    # 3. FRAMES_BRANCH=off: the store alone; git is never asked (FRAMES_URL points nowhere)
    with tempfile.TemporaryDirectory() as td:
        site = Site(Path(td))
        stub.objects.clear()
        for rel, text in BRANCH_FILES.items():
            stub.put(rel, text.encode())
        before = site.tree_id()
        env = stub.env() | {"FRAMES_BRANCH": "off", "FRAMES_URL": (site.tmp / "nowhere.git").as_uri()}
        res = publish_all(site, LIB, MODELS, env)
        for k in ("publish", "data", "fetch"):
            ok(res[k], k)
        check({k: stub.body(k).decode() for k in stub.listed()} == EXPECTED, f"off: store {stub.listed()}")
        check(site.tree_id() == before, "off: the branch must not move")
        check(res["data"].stdout.splitlines()[-1:] == [""], f"off: empty SHA line {res['data'].stdout!r}")
        check(res["seeded"] == SEEDED, f"off: seeded from the store {res['seeded']}")
        # nothing rendered: a warning, the store untouched
        empty = site.tmp / "empty"
        empty.mkdir()
        mark = len(stub.log)
        r = site.run(LIB / "publish_frames_ci.sh", "assets/sst/anim/loop", env=env, cwd=empty)
        ok(r, "off, nothing rendered")
        check("nothing rendered" in r.stdout and stub.writes(mark) == [], r.stdout)

    # 4. FRAMES_BRANCH=off with no usable store: refused, and no branch named "off" appears
    with tempfile.TemporaryDirectory() as td:
        site = Site(Path(td))
        render(site.ws)
        env = dict.fromkeys(STORE_ENV, "") | {"FRAMES_BRANCH": "off"}
        r = site.run(LIB / "publish_frames_ci.sh", "assets/sst/anim/loop", env=env)
        check(r.returncode == 1 and "::error::" in r.stdout, f"broken publish: {r.returncode} {r.stdout}")
        write(site.ws, "assets/models/data/hrrr/2026102012/t2m/000.webp", "c12")
        r = site.run(
            MODELS / "publish_data.sh", "assets/models/data/hrrr/2026102012/t2m", env=env | {"SRC": str(site.ws)}
        )
        check(r.returncode == 1 and "::error::" in r.stderr, f"broken publish_data: {r.returncode} {r.stderr}")
        r = site.run(LIB / "fetch_frames_ci.sh", "assets/sst/anim/arch", env=env)
        check(r.returncode == 0 and "::warning::" in r.stdout, f"broken fetch: {r.stdout}")
        check("refs/heads/off" not in site.refs(), site.refs())

    # 5. another data branch never touches the store
    with tempfile.TemporaryDirectory() as td:
        site = Site(Path(td))
        write(site.ws, "scripts/verify/data/truth/2026-01-02.json", "{}")
        mark = len(stub.log)
        env = stub.env() | {"FRAMES_BRANCH": "verify-data"}
        ok(site.run(LIB / "publish_frames_ci.sh", "+scripts/verify/data/truth", env=env), "verify-data")
        check(len(site.tree("verify-data")) == 2 and len(stub.log) == mark, "verify-data: the branch alone")


def main() -> int:
    stub = Stub()
    test_client(stub)
    print("frames_store.py: ok")
    test_wrappers(stub)
    print("publish_frames_ci.sh / fetch_frames_ci.sh / publish_data.sh: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
