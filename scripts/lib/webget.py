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
DEFAULT_UA = "scorvec.com data pipeline (scorvec@outlook.com)"
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
