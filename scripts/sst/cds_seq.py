#!/usr/bin/env python3
"""One CDS request at a time for the ONE key, wherever it runs (laptop or Actions).

User rule: the Copernicus key is shared by every CDS user on this project, and two requests in flight at once break
it. Actions jobs serialise through the `seas5` concurrency group (seas5.yml, seas5-lead7.yml, c3s.yml, c3s-u10-skill.yml,
...); a laptop request (allowed since 2026-10-09, "you can rerun anything on the laptop") must therefore yield to that
group. Two independent checks guard every request:

  1. the key's own queue: no other job of this key may be `accepted` or `running` (the CDS jobs API sees requests from
     every client of the key, so it also catches a laptop job started by hand);
  2. on the laptop only: no workflow of the `seas5` concurrency group may be queued, waiting or in progress (gh CLI).

The checks run before a request is submitted AND every POLL seconds while it is in flight; if either trips while our
request is still queued or running, the request is DELETED on the CDS and resubmitted once the key is idle again, so an
overlap lasts at most one poll interval. Retrieval is submit → poll → download, never a blocking retrieve().

    from cds_seq import retrieve
    ok = retrieve("seasonal-original-pressure-levels", request, Path("out.grib"), "label")
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

POLL = float(os.environ.get("CDS_SEQ_POLL", "45"))
REPO = Path(os.environ.get("CDS_SEQ_REPO", Path(__file__).resolve().parents[2]))   # where .github/workflows lives
GH = os.environ.get("GH_BIN", "/opt/homebrew/bin/gh" if Path("/opt/homebrew/bin/gh").exists() else "gh")
GH_REPO = os.environ.get("GITHUB_REPOSITORY", "scorvec/scorvec.github.io")
IN_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"
LOG = Path(os.environ.get("CDS_SEQ_LOG", REPO / "scripts" / "sst" / "data" / "cds_seq.jsonl"))


def _client():
    from ecmwf.datastores import Client
    url = os.environ.get("CDSAPI_URL"); key = os.environ.get("CDSAPI_KEY")
    if not key:
        cfg = {}
        p = Path(os.path.expanduser("~/.cdsapirc"))
        for line in p.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1); cfg[k.strip()] = v.strip()
        url = url or cfg.get("url"); key = cfg.get("key")
    return Client(url=url or "https://cds.climate.copernicus.eu/api", key=key, progress=False, maximum_tries=30,
                  retry_after=60, timeout=120)


def seas5_group_workflows() -> list[str]:
    wf = REPO / ".github" / "workflows"
    out = []
    for f in sorted(wf.glob("*.yml")):
        if re.search(r"^\s*group:.*\bseas5\b", f.read_text(), re.M):   # incl. seas5-lead7's expression form
            out.append(f.name)
    return out


def group_busy() -> list[str]:
    """Runs of the seas5 concurrency group that are not finished (laptop only; in Actions we ARE the group)."""
    if IN_ACTIONS:
        return []
    busy = []
    for w in seas5_group_workflows():
        for st in ("in_progress", "queued", "waiting", "pending", "requested"):
            r = subprocess.run([GH, "run", "list", "--repo", GH_REPO, "-w", w, "-s", st, "--limit", "5",
                                "--json", "databaseId,status"], capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                busy.append(f"{w}: gh failed ({r.stderr.strip()[:80]})")      # cannot tell: treat as busy
                break
            for run in json.loads(r.stdout or "[]"):
                busy.append(f"{w} run {run['databaseId']} {run['status']}")
    return busy


def key_busy(client, mine: str | None = None) -> list[str]:
    """Other jobs of this key that are queued or running."""
    out = []
    try:
        for st in ("accepted", "running"):
            jobs = client.get_jobs(status=st, limit=20)
            for j in jobs.json.get("jobs", []):
                jid = j.get("jobID") or j.get("job_id") or j.get("id")
                if jid and jid != mine:
                    out.append(f"CDS job {jid} {st} ({j.get('processID', '?')})")
    except Exception as e:                                               # noqa: BLE001
        out.append(f"CDS jobs list failed ({str(e)[:80]})")
    return out


def wait_idle(client, label: str, max_wait_h: float) -> bool:
    t0 = time.time(); said = None
    while True:
        why = group_busy() + key_busy(client)
        if not why:
            return True
        if (time.time() - t0) / 3600 > max_wait_h:
            print(f"  {label}: still busy after {max_wait_h} h ({'; '.join(why)[:200]}) - giving up", flush=True)
            return False
        msg = "; ".join(why)[:240]
        if msg != said:
            print(f"  {label}: waiting - {msg}", flush=True); said = msg
        time.sleep(120)


def _log(**kw) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as fh:
            fh.write(json.dumps(dict(kw, at=time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()))) + "\n")
    except OSError:
        pass


_CONSTRAINTS: dict = {}
KEYS = ("originating_centre", "system", "year", "month", "day", "product_type", "variable", "pressure_level")


def available(dataset: str, req: dict) -> bool | None:
    """Does the CDS catalogue's constraints list hold this request's (centre, system, year, month, day, product, variable,
    level)? A catalogue read, not a request: no queue, no key. None when the catalogue cannot be read. 2026-10-09: an
    unpublished issue costs no data but sat 30 min in the queue before failing, so the issue fetchers ask this first."""
    import urllib.request
    if dataset not in _CONSTRAINTS:
        try:
            d = json.load(urllib.request.urlopen(f"https://cds.climate.copernicus.eu/api/catalogue/v1/collections/{dataset}", timeout=60))
            url = [l["href"] for l in d["links"] if l["rel"] == "constraints"][0]
            _CONSTRAINTS[dataset] = json.load(urllib.request.urlopen(url, timeout=120))
        except Exception as e:                                           # noqa: BLE001
            print(f"  constraints for {dataset} unreadable ({str(e)[:100]})", flush=True)
            return None
    want = {k: [str(v) for v in (req[k] if isinstance(req[k], list) else [req[k]])] for k in KEYS if k in req}
    for e in _CONSTRAINTS[dataset]:
        if all(set(v) <= set(e.get(k, v)) for k, v in want.items()):
            return True
    return False


def retrieve(dataset: str, req: dict, dest: Path, label: str, tries: int = 4, max_wait_h: float = 30.0) -> bool:
    """dest written atomically; False when the CDS has no such data or every try failed."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    client = _client()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".part{os.getpid()}")
    attempt = 0
    while attempt < tries:
        if not wait_idle(client, label, max_wait_h):
            return False
        t0 = time.time()
        try:
            remote = client.submit(dataset, req)
        except Exception as e:                                           # noqa: BLE001
            msg = str(e).replace("\n", " ")
            print(f"  {label}: submit failed ({msg[:300]})", flush=True)
            if "no data" in msg.lower() or "not available" in msg.lower() or "invalid" in msg.lower():
                _log(label=label, dataset=dataset, result="refused", msg=msg[:300]); return False
            attempt += 1; time.sleep(60); continue
        rid = remote.request_id
        print(f"  {label}: submitted {rid}", flush=True)
        yielded = False
        while True:
            time.sleep(POLL)
            try:
                st = remote.status
            except Exception as e:                                       # noqa: BLE001
                print(f"  {label}: status check failed ({str(e)[:120]})", flush=True); continue
            if st == "successful":
                break
            if st in ("failed", "rejected", "dismissed", "deleted"):
                break
            why = group_busy()
            if why:                                                      # an Actions CDS job started: step aside
                print(f"  {label}: yielding to {'; '.join(why)[:160]} - deleting {rid}", flush=True)
                try:
                    remote.delete()
                except Exception as e:                                   # noqa: BLE001
                    print(f"  {label}: delete failed ({str(e)[:120]})", flush=True)
                yielded = True
                break
        if yielded:
            _log(label=label, dataset=dataset, result="yielded", minutes=round((time.time() - t0) / 60, 1))
            continue                                                     # not counted as a failed try
        if st != "successful":
            err = ""
            try:
                err = json.dumps(remote.json.get("error") or remote.json.get("message") or "")[:400]
            except Exception:                                            # noqa: BLE001
                pass
            print(f"  {label}: job {rid} {st} {err}", flush=True)
            _log(label=label, dataset=dataset, result=st, msg=err)
            if "no data" in err.lower() or "nodata" in err.lower() or "not available" in err.lower():
                return False
            attempt += 1; time.sleep(60); continue
        try:
            remote.download(str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest)
                dt = (time.time() - t0) / 60
                print(f"  {label}: {dest.stat().st_size / 1e6:.1f} MB in {dt:.1f} min", flush=True)
                _log(label=label, dataset=dataset, result="ok", mb=round(dest.stat().st_size / 1e6, 2), minutes=round(dt, 1))
                try:
                    remote.delete()                                      # tidy the key's job list
                except Exception:                                        # noqa: BLE001
                    pass
                return True
        except Exception as e:                                           # noqa: BLE001
            print(f"  {label}: download failed ({str(e)[:200]})", flush=True)
        attempt += 1; time.sleep(30)
    return False


if __name__ == "__main__":
    import sys
    c = _client()
    print("seas5 group workflows:", ", ".join(seas5_group_workflows()))
    print("group busy:", group_busy() or "no")
    print("key busy:", key_busy(c) or "no")
    sys.exit(0)
