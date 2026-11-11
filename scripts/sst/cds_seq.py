#!/usr/bin/env python3
"""One CDS request at a time on the shared key, wherever it runs (Actions or a manual run).

The Copernicus key is shared by every CDS user on this project, and two requests in flight at once break
it. Actions jobs serialise through the `seas5` concurrency group (seas5.yml, seas5-lead7.yml, c3s.yml, c3s-u10-skill.yml,
...); a request made outside Actions must therefore yield to that group. Two independent checks guard every request:

  1. the key's own queue: no other job of this key may be `accepted` or `running` (the CDS jobs API sees requests from
     every client of the key, so it also catches a job started by hand);
  2. outside Actions only: no workflow of the `seas5` concurrency group may be queued, waiting or in progress (gh CLI).

Both checks run before a request is submitted; the second repeats every POLL seconds while the request is in flight,
and if it trips the request is deleted on the CDS and resubmitted once the key is idle again, so an overlap lasts at
most one poll interval. Retrieval is submit → poll → download, never a blocking retrieve(). Before submitting, a job
of the key that already holds the request (finished, running or queued; e.g. one a run left behind when it hit its
time limit while the job waited in the CDS queue) is adopted instead: adopt(). A job that was lost, failed or would
not download is not adopted again.

The CDS answers a request it holds in its cache within seconds, whatever is queued. retrieve(probe=True) uses that:
with the key busy it sends the request once all the same and keeps it only if it is back within PROBE seconds;
otherwise the job is deleted and the request waits its turn like any other. The CDS also rejects a request while its
queue for the dataset is full: that is waited out and sent again, and does not count as a failed try. Both waits end
at the caller's deadline (start_by), so a run leaves what it could not place for a later one.

    from cds_seq import retrieve
    ok = retrieve("seasonal-original-pressure-levels", request, Path("out.grib"), "label")
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

POLL = float(os.environ.get("CDS_SEQ_POLL", "45"))
STATUS_FAILS = 30  # consecutive failed status checks after which a job is treated as lost
PROBE = 20.0  # s a request sent past a busy key may take: one the CDS holds in its cache is back by then
FULL = "queued requests"  # in the reason of a job rejected because the CDS queue for its dataset is full
SHAPE = ("area", "grid")  # request keys that set the field itself: a job is adopted only if both are the same
REPO = Path(os.environ.get("CDS_SEQ_REPO", Path(__file__).resolve().parents[2]))  # where .github/workflows lives
GH = os.environ.get("GH_BIN") or shutil.which("gh") or "gh"
GH_REPO = os.environ.get("GITHUB_REPOSITORY", "scorvec/scorvec.github.io")
IN_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"
LOG = Path(os.environ.get("CDS_SEQ_LOG", REPO / "scripts" / "sst" / "data" / "cds_seq.jsonl"))


def _client():
    from ecmwf.datastores import Client

    url = os.environ.get("CDSAPI_URL")
    key = os.environ.get("CDSAPI_KEY")
    if not key:
        cfg = {}
        p = Path(os.path.expanduser("~/.cdsapirc"))
        for line in p.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                cfg[k.strip()] = v.strip()
        url = url or cfg.get("url")
        key = cfg.get("key")
    return Client(
        url=url or "https://cds.climate.copernicus.eu/api",
        key=key,
        progress=False,
        maximum_tries=30,
        retry_after=60,
        timeout=120,
    )


def seas5_group_workflows() -> list[str]:
    wf = REPO / ".github" / "workflows"
    return [
        f.name
        for f in sorted(wf.glob("*.yml"))
        if re.search(r"^\s*group:.*\bseas5\b", f.read_text(), re.M)  # incl. seas5-lead7's expression form
    ]


def _workflow_names() -> dict:
    """{display name: file} of the seas5-group workflows (gh run list reports the display name)."""
    out = {}
    for w in seas5_group_workflows():
        m = re.search(r"^name:\s*(.+?)\s*$", (REPO / ".github" / "workflows" / w).read_text(), re.M)
        out[(m.group(1).strip("'\"") if m else w)] = w
    return out


def group_busy(strict: bool = True) -> list[str]:
    """Runs of the seas5 concurrency group that are not finished (outside Actions only; in Actions we ARE the group).
    strict: a failed gh call counts as busy (before submitting); not strict: it is ignored (while a request is in flight,
    so a GitHub API hiccup does not throw away a queue position)."""
    if IN_ACTIONS:
        return []
    names = _workflow_names()
    busy = []
    for st in ("in_progress", "queued", "waiting", "pending", "requested"):
        r = subprocess.run(
            [
                GH,
                "run",
                "list",
                "--repo",
                GH_REPO,
                "-s",
                st,
                "--limit",
                "100",
                "--json",
                "databaseId,status,workflowName",
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if r.returncode != 0:
            if strict:
                busy.append(f"gh failed ({r.stderr.strip()[:80]})")
            continue
        busy.extend(
            f"{names[run['workflowName']]} run {run['databaseId']} {run['status']}"
            for run in json.loads(r.stdout or "[]")
            if run.get("workflowName") in names
        )
    return busy


def key_busy(client) -> list[str]:
    """Jobs of this key that are queued or running."""
    out = []
    try:
        for st in ("accepted", "running"):
            jobs = client.get_jobs(status=st, limit=20)
            for j in jobs.json.get("jobs", []):
                jid = j.get("jobID") or j.get("job_id") or j.get("id")
                if jid:
                    out.append(f"CDS job {jid} {st} ({j.get('processID', '?')})")
    except Exception as e:  # noqa: BLE001
        out.append(f"CDS jobs list failed ({str(e)[:80]})")
    return out


def wait_idle(client, label: str, give_up: float) -> bool:
    """Wait until nothing else of the key (and, outside Actions, of the seas5 group) is in flight. False once the
    time give_up (epoch seconds) has passed."""
    said = None
    while True:
        why = group_busy() + key_busy(client)
        if not why:
            return True
        if time.time() > give_up:
            print(f"  {label}: still busy ({'; '.join(why)[:200]}) - left for a later run", flush=True)
            return False
        msg = "; ".join(why)[:240]
        if msg != said:
            print(f"  {label}: waiting - {msg}", flush=True)
            said = msg
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
    level)? A catalogue read, not a request: no queue, no key. None when the catalogue cannot be read. A request for
    an unpublished issue costs no data but can sit 30 min in the queue before failing, so the issue fetchers ask this
    first."""
    import urllib.request

    if dataset not in _CONSTRAINTS:
        try:
            d = json.load(
                urllib.request.urlopen(
                    f"https://cds.climate.copernicus.eu/api/catalogue/v1/collections/{dataset}", timeout=60
                )
            )
            url = next(l["href"] for l in d["links"] if l["rel"] == "constraints")
            _CONSTRAINTS[dataset] = json.load(urllib.request.urlopen(url, timeout=120))
        except Exception as e:  # noqa: BLE001
            print(f"  constraints for {dataset} unreadable ({str(e)[:100]})", flush=True)
            return None
    want = {k: [str(v) for v in (req[k] if isinstance(req[k], list) else [req[k]])] for k in KEYS if k in req}
    # every combination must be in SOME entry: the catalogue splits one month's start days over several entries (UKMO's
    # September 2026: day 01, the hindcast days 09 / 17 / 25 and the other days each listed apart)
    keys = list(want)
    return all(
        any(all(c[i] in e[k] for i, k in enumerate(keys) if k in e) for e in _CONSTRAINTS[dataset])
        for c in itertools.product(*(want[k] for k in keys))
    )


def _numbers(v) -> tuple:
    """An area or a grid as numbers, however the request wrote it ([90, -180, -90, 180], "1/1", 1.0); () for none."""
    if v is None:
        return ()
    if isinstance(v, str):
        v = v.split("/")
    return tuple(float(x) for x in (v if isinstance(v, (list, tuple)) else [v]))


def _covers(have: dict, want: dict) -> bool:
    """A job's request holds every value of the wanted one (more days or steps are fine: the reader keeps what it
    needs) on the same area and grid."""
    if any(_numbers(have.get(k)) != _numbers(want.get(k)) for k in SHAPE):
        return False
    for k, v in want.items():
        if k in SHAPE:
            continue
        a = {str(x) for x in (v if isinstance(v, list) else [v])}
        b = {str(x) for x in (have.get(k) if isinstance(have.get(k), list) else [have.get(k)])}
        if not a <= b:
            return False
    return True


_HELD: dict[str, dict] = {}  # job id -> the request it holds, read once


def adopt(client, dataset: str, req: dict, skip: set | frozenset = frozenset()):
    """A job of this key that already holds the request - finished, running or queued, e.g. left by a run that hit its
    time limit while the job waited in the CDS queue: its Remote, so it is downloaded or awaited instead of resubmitted.
    skip: job ids already tried."""
    for st in ("successful", "running", "accepted"):
        try:
            jobs = client.get_jobs(status=st, limit=50).json.get("jobs", [])
        except Exception as e:  # noqa: BLE001
            print(f"  CDS jobs list failed ({str(e)[:80]})", flush=True)
            return None
        for j in jobs:
            jid = j.get("jobID")
            if j.get("processID") != dataset or jid in skip:
                continue
            with contextlib.suppress(Exception):  # noqa: BLE001
                if jid not in _HELD:
                    _HELD[jid] = client.get_remote(jid).request
                if _covers(_HELD[jid], req):
                    return client.get_remote(jid)
    return None


def _reason(remote, status: str) -> str:
    """Why a job ended without a result. A rejection's text is only in the answer of the results endpoint."""
    with contextlib.suppress(Exception):  # noqa: BLE001
        err = remote.json.get("error") or remote.json.get("message")
        if err:
            return json.dumps(err)[:400]
    if status in ("failed", "rejected"):
        try:
            remote.get_results()
        except Exception as e:  # noqa: BLE001
            return str(e).replace("\n", " ")[:400]
    return ""


def retrieve(
    dataset: str,
    req: dict,
    dest: Path,
    label: str,
    tries: int = 4,
    max_wait_h: float = 30.0,
    start_by: float | None = None,
    probe: bool = False,
) -> bool:
    """dest written atomically; False when the CDS has no such data, every try failed, or the request found no place
    in the queue in time: within max_wait_h, or by start_by (epoch seconds) if that is sooner. A job already in the
    queue is waited for without a limit: the caller's own time limit ends that, and the job stays for the next run to
    adopt. probe: with the key busy, send the request once all the same and keep it if the CDS answers it from its
    cache (PROBE)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    client = _client()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".part{os.getpid()}")
    give_up = min(time.time() + max_wait_h * 3600, start_by or float("inf"))
    attempt = 0
    tried = set()  # jobs that failed here: a retry submits anew instead of adopting one of them
    while attempt < tries:
        t0 = time.time()
        probing = False
        remote = adopt(client, dataset, req, tried)
        if remote is not None:
            print(
                f"  {label}: adopting the key's job {remote.request_id} ({remote.status}), which holds this request",
                flush=True,
            )
        else:
            probing, probe = probe and not group_busy() and bool(key_busy(client)), False
            if not probing and not wait_idle(client, label, give_up):
                return False
            try:
                remote = client.submit(dataset, req)
            except Exception as e:  # noqa: BLE001
                msg = str(e).replace("\n", " ")
                print(f"  {label}: submit failed ({msg[:300]})", flush=True)
                if "no data" in msg.lower() or "not available" in msg.lower() or "invalid" in msg.lower():
                    _log(label=label, dataset=dataset, result="refused", msg=msg[:300])
                    return False
                attempt += 1
                time.sleep(60)
                continue
            print(f"  {label}: submitted {remote.request_id}{' past a busy key' if probing else ''}", flush=True)
        rid = remote.request_id
        tried.add(rid)
        sent = time.time()
        yielded = False
        failures = 0  # consecutive status checks that raised
        for k in itertools.count():
            # the first checks come quickly: a cached request and a rejection are both settled within seconds
            time.sleep(5.0 if probing else min(POLL, 5.0 * 2 ** min(k, 4)))
            try:
                st = remote.status
                failures = 0
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"  {label}: status check failed ({str(e)[:120]})", flush=True)
                # a job the CDS no longer knows (404: deleted or expired on its side) never comes back, and nor does
                # one that has not answered for STATUS_FAILS checks: count a failed try and submit it again
                if "404" in str(e) or failures >= STATUS_FAILS:
                    st = "lost"
                    break
                continue
            if st == "successful":
                break
            if st in ("failed", "rejected", "dismissed", "deleted"):
                break
            why = group_busy(strict=False)
            if probing and time.time() - sent >= PROBE:
                why = ["the key's other requests (this one is not in the CDS cache)"]
            if why:  # an Actions CDS job started, or the probe has to queue: step aside
                print(f"  {label}: yielding to {'; '.join(why)[:160]} - deleting {rid}", flush=True)
                try:
                    remote.delete()
                except Exception as e:  # noqa: BLE001
                    print(f"  {label}: delete failed ({str(e)[:120]})", flush=True)
                yielded = True
                break
        if yielded:
            _log(label=label, dataset=dataset, result="yielded", minutes=round((time.time() - t0) / 60, 1))
            continue  # not counted as a failed try
        if st != "successful":
            err = _reason(remote, st)
            print(f"  {label}: job {rid} {st} {err}", flush=True)
            _log(label=label, dataset=dataset, result=st, msg=err)
            if "no data" in err.lower() or "nodata" in err.lower() or "not available" in err.lower():
                return False
            if FULL in err.lower():  # no place in the CDS queue: not a failed try
                if probing:
                    continue  # it waits its turn from here
                if time.time() + 120 > give_up:
                    print(f"  {label}: the CDS queue is still full - left for a later run", flush=True)
                    return False
                time.sleep(120)
                continue
            attempt += 1
            time.sleep(60)
            continue
        try:
            remote.download(str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest)
                dt = (time.time() - t0) / 60
                print(f"  {label}: {dest.stat().st_size / 1e6:.1f} MB in {dt:.1f} min", flush=True)
                q = {}
                try:  # queue vs run time, for sizing requests
                    from datetime import datetime

                    J = remote.json
                    c, s0, f = (datetime.fromisoformat(J[k]) for k in ("created", "started", "finished"))
                    q = dict(
                        queue_min=round((s0 - c).total_seconds() / 60, 1),
                        run_min=round((f - s0).total_seconds() / 60, 1),
                    )
                except Exception:  # noqa: BLE001
                    pass
                _log(
                    label=label,
                    dataset=dataset,
                    result="ok",
                    mb=round(dest.stat().st_size / 1e6, 2),
                    minutes=round(dt, 1),
                    **q,
                )
                with contextlib.suppress(Exception):  # noqa: BLE001
                    remote.delete()  # tidy the key's job list
                return True
        except Exception as e:  # noqa: BLE001
            print(f"  {label}: download failed ({str(e)[:200]})", flush=True)
        attempt += 1
        time.sleep(30)
    return False


if __name__ == "__main__":
    import sys

    c = _client()
    print("seas5 group workflows:", ", ".join(seas5_group_workflows()))
    print("group busy:", group_busy() or "no")
    print("key busy:", key_busy(c) or "no")
    sys.exit(0)
