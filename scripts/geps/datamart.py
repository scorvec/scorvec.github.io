"""MSC Datamart access for the GEPS products: one polite, honest, accounted fetcher.

Every GEPS file the subseasonal page reads comes through here (forecast.py, mjo_geps.py, telecon_geps.py and, via
strat_maps.fetch, the vortex, heat-flux and dripping-paint products), so that:

  - the User-Agent is the site's honest one (scripts/lib/webget.py DEFAULT_UA), never a bare library default;
  - transient failures (429/5xx, network) are retried with backoff, a 404 is not (the step is not there yet / any
    more), and nothing is fetched twice in a run (callers cache to disk);
  - each request is written to WORK/fetch_ledger.tsv (bytes, url) so a run reports exactly what it downloaded
    (report_ledger at the end of the workflow).

Data: ECCC GEPS, Open Government Licence - Canada (Contains information licensed under the Open Government Licence -
Canada). https://eccc-msc.github.io/open-data/licence/readme_en/
"""
from __future__ import annotations

import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import webget                                                       # noqa: E402

import paths                                                        # noqa: E402

BASE = "https://dd.weather.gc.ca"
# one extended-range file per (variable, level, step), all 21 members (control + 20 perturbed) in each
URL = (BASE + "/{d}/WXO-DD/ensemble/geps/grib2/raw/{c}/{L:03d}/"
       "CMC_geps-raw_{v}_latlon0p5x0p5_{d}{c}_P{L:03d}_allmbrs.grib2")
# Concurrency is deliberately modest: the Datamart is a shared public service.
WORKERS = int(os.environ.get("GEPS_FETCH_WORKERS", "4"))
_LOCK = threading.Lock()
# GEPS_OFFLINE=1: no request leaves the machine (rebuilding a cycle from files already held, e.g. the laptop check)
OFFLINE = os.environ.get("GEPS_OFFLINE") == "1"


def url(date: str, var: str, L: int, cyc: str = "00") -> str:
    return URL.format(d=date, c=cyc, v=var, L=L)


def _log(nbytes: int, u: str, kind: str = "GET") -> None:
    try:
        paths.WORK.mkdir(parents=True, exist_ok=True)
        with _LOCK, open(paths.WORK / "fetch_ledger.tsv", "a") as f:
            f.write(f"{kind}\t{nbytes}\t{u}\n")
    except OSError:
        pass


def get(u: str, timeout: int = 600) -> bytes | None:
    """The body of `u`, or None if it is not there (404) or keeps failing."""
    if OFFLINE:
        return None
    try:
        b = webget.get(u, timeout=timeout, tries=4, backoff=6.0)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"    HTTP {e.code} {u.rsplit('/', 1)[-1]}", flush=True)
        return None
    except Exception as e:                                          # noqa: BLE001
        print(f"    failed {u.rsplit('/', 1)[-1]}: {str(e)[:60]}", flush=True)
        return None
    _log(len(b), u)
    return b


def download(u: str, dest: Path, timeout: int = 600) -> Path | None:
    """Fetch `u` into `dest` (atomically) unless `dest` already holds it."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    b = get(u, timeout)
    if b is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(b)
    tmp.replace(dest)
    return dest


def exists(u: str, timeout: int = 60) -> bool:
    """HEAD: is this step published yet? (no body transferred)"""
    if OFFLINE:
        return False
    try:
        req = urllib.request.Request(u, method="HEAD", headers=paths.UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ok = r.status == 200
        _log(0, u, "HEAD")
        return ok
    except Exception:                                               # noqa: BLE001
        return False


def report_ledger(path: Path | None = None) -> str:
    p = path or (paths.WORK / "fetch_ledger.tsv")
    if not p.exists():
        return "no Datamart requests this run"
    n = h = tot = 0
    for ln in p.read_text().splitlines():
        k, b, _ = ln.split("\t", 2)
        if k == "HEAD":
            h += 1
        else:
            n += 1; tot += int(b)
    return f"{n} GET ({tot / 1e9:.2f} GB) + {h} HEAD requests to the MSC Datamart"


if __name__ == "__main__":
    print(report_ledger())
