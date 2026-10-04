"""Shared HTTP GET with retries — one place for the site's data-pipeline fetches.

Consolidates the ~15 hand-rolled ``urllib.urlopen`` + retry loops scattered across
the SST / MJO / renewables / ASOS scripts. Retries on transient network errors and
transient HTTP statuses (429/500/502/503/504) with linear backoff; raises
immediately on a non-transient 4xx (no point retrying a 404) and re-raises the last
error once ``tries`` is exhausted.

Import from a nested script the same way the pipelines already reach
``scripts/ecmwf/store.py`` — via a sys.path insert of the ``scripts/lib`` dir:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                                if p.name == "scripts") / "lib"))
    from webget import get_json, get_text          # noqa: E402
"""
from __future__ import annotations

import json as _json
import time
import urllib.error
import urllib.request

# Identifying UA — several sources (api.weather.gov, aviationweather.gov) want one.
DEFAULT_UA = "scorvec.com data pipeline (+https://scorvec.com)"
_TRANSIENT = {429, 500, 502, 503, 504}


def get(url, *, headers=None, ua=DEFAULT_UA, timeout=60, tries=4, backoff=4.0):
    """GET ``url`` and return the raw ``bytes``.

    Retries transient failures (network errors and HTTP 429/5xx) up to ``tries``
    times with ``backoff * attempt`` seconds between tries. A non-transient HTTP
    error (e.g. 404) is raised on the first hit; otherwise the last error is
    re-raised after the final attempt.
    """
    hdrs = dict(headers or {})
    if ua and "User-Agent" not in hdrs:
        hdrs["User-Agent"] = ua
    req = urllib.request.Request(url, headers=hdrs)
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in _TRANSIENT:
                raise
        except Exception as e:            # URLError, timeout, socket errors, …
            last = e
        if i < tries - 1:
            time.sleep(backoff * (i + 1))
    raise last


def get_text(url, encoding="utf-8", **kw):
    """GET ``url`` and return decoded text."""
    return get(url, **kw).decode(encoding)


def get_json(url, **kw):
    """GET ``url`` and parse the body as JSON."""
    return _json.loads(get(url, **kw).decode())


def cached(url, dest, *, max_age_h=12.0, ua=DEFAULT_UA, timeout=60, tries=4, backoff=4.0, encoding="utf-8"):
    """Fetch ``url`` into ``dest`` only when needed, and return its text (2026-10-04, user: "collect data
    strategically ... not re-downloading data that already exists").

    * a copy fetched less than ``max_age_h`` hours ago is used as is (no request at all);
    * otherwise a CONDITIONAL GET (If-None-Match / If-Modified-Since from the sidecar ``dest.meta.json``):
      HTTP 304 transfers nothing; 200 replaces the copy;
    * any failure falls back to the existing copy (raises only when there is none).
    The sidecar records fetched/etag/last_modified. Persist ``dest`` between Actions runs with actions/cache.
    Works for http(s); ftp:// has no conditional GET, so use ``max_age_h`` alone to pace it."""
    from pathlib import Path
    from datetime import datetime, timezone
    dest = Path(dest); meta_p = dest.with_name(dest.name + ".meta.json")
    meta = {}
    if meta_p.exists():
        try: meta = _json.loads(meta_p.read_text())
        except Exception: meta = {}
    now = datetime.now(timezone.utc)
    if dest.exists() and meta.get("fetched"):
        age = (now - datetime.fromisoformat(meta["fetched"])).total_seconds() / 3600
        if age < max_age_h:
            return dest.read_text(encoding=encoding)
    hdrs = {"User-Agent": ua}
    if dest.exists() and url.startswith("http"):
        if meta.get("etag"): hdrs["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"): hdrs["If-Modified-Since"] = meta["last_modified"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=hdrs)
        last = None
        for i in range(tries):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    body = r.read()
                    meta = {"etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified")}
                dest.write_bytes(body); break
            except urllib.error.HTTPError as e:
                if e.code == 304:
                    print(f"  {dest.name}: not modified (304, nothing transferred)"); break
                last = e
                if e.code not in _TRANSIENT: raise
            except Exception as e:                    # URLError, timeout, socket errors, ...
                last = e
            if i < tries - 1: time.sleep(backoff * (i + 1))
        else:
            raise last
        meta["fetched"] = now.isoformat(timespec="seconds")
        meta_p.write_text(_json.dumps(meta))
    except Exception as e:                            # noqa: BLE001
        if not dest.exists(): raise
        print(f"  {dest.name}: fetch failed ({repr(e)[:60]}); using the cached copy")
    return dest.read_text(encoding=encoding)
