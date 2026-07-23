"""Cross-process ECMWF request budget — the PREVENTIVE rate limiter.

Every ECMWF HTTP request from this machine (rangefetch v2, the legacy
ecmwf-opendata path via store.py's multiurl patch, benchmarks, ad-hoc
scripts) acquires a token here before hitting the network. A shared
flock-protected state file makes the budget hold across processes, so
a pipeline and a benchmark can no longer stack up into a burst that
draws 429/503 SlowDowns; the reactive shedding in the fetchers remains
as the second line of defence.

Model: sliding-window rate caps, one global ("*") plus per-mirror.
Defaults are deliberately below the observed throttle threshold:
  global  ECMWF_RPS       20 req/s  (window-averaged)
  aws     ECMWF_RPS_AWS   14 req/s
  azure   ECMWF_RPS_AZURE 14 req/s
  ecmwf   ECMWF_RPS_ECMWF  6 req/s  (data.ecmwf.int is the frailest)
A 429/503 anywhere calls penalize(host): that host's cap halves for
PENALTY_S seconds (stacking down to a floor), written into the shared
state so every process slows down together.

Benchmarks should set ECMWF_RPS=3 (etc.) rather than bypassing.
Fail-open by design: if the state file is unusable the caller proceeds
unmetered — the limiter must never wedge a pipeline.

    from budget import acquire, penalize
    acquire("aws")            # blocks (bounded) until a token is free
    ...
    penalize("aws")           # server said slow down
"""
from __future__ import annotations

import fcntl
import json
import os
import random
import time
from pathlib import Path

STATE = Path(os.environ.get(
    "ECMWF_BUDGET_STATE",
    str(Path(__file__).resolve().parent / ".budget_state.json")))
WINDOW_S = 2.0                  # sliding window; cap = rps * WINDOW_S
PENALTY_S = 45.0                # halved-cap duration after a throttle signal
FLOOR_FRAC = 0.25               # penalties never cut below this fraction
MAX_WAIT_S = 180.0              # bound the block; then fail-open

_DEFAULTS = {"*": 20.0, "aws": 14.0, "azure": 14.0, "ecmwf": 6.0}


def _rps(host: str) -> float:
    env = {"*": "ECMWF_RPS", "aws": "ECMWF_RPS_AWS",
           "azure": "ECMWF_RPS_AZURE", "ecmwf": "ECMWF_RPS_ECMWF"}
    return float(os.environ.get(env.get(host, ""), _DEFAULTS.get(host, 10.0)))


def _cap(host: str, st: dict, now: float) -> int:
    cap = _rps(host) * WINDOW_S
    pen = st.get("penalty", {}).get(host, 0.0)
    if pen > now:                                   # halved (stacking) while penalized
        halvings = st.get("penalty_n", {}).get(host, 1)
        cap = max(cap * (0.5 ** halvings), cap * FLOOR_FRAC)
    return max(1, int(cap))


class _Locked:
    def __enter__(self):
        self.fh = open(STATE.parent / (STATE.name + ".lock"), "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        try:
            self.st = json.loads(STATE.read_text())
        except Exception:                           # noqa: BLE001 — fresh/corrupt
            self.st = {}
        return self.st

    def __exit__(self, *exc):
        try:
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.st))
            tmp.replace(STATE)
        finally:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
        return False


def acquire(host: str = "*", n: int = 1) -> None:
    """Block (bounded) until n request tokens are available for host AND the
    global pool, then consume them. Fail-open on any state-file trouble."""
    deadline = time.monotonic() + MAX_WAIT_S
    hosts = ("*",) if host == "*" else ("*", host)
    while True:
        try:
            with _Locked() as st:
                now = time.time()
                ts = st.setdefault("ts", {})
                for h in hosts:
                    ts[h] = [t for t in ts.get(h, []) if now - t < WINDOW_S]
                if all(len(ts[h]) + n <= _cap(h, st, now) for h in hosts):
                    for h in hosts:
                        ts[h].extend([now] * n)
                    return
        except Exception:                           # noqa: BLE001 — fail-open
            return
        if time.monotonic() > deadline:
            return                                  # fail-open, never wedge
        time.sleep(0.08 + random.random() * 0.12)


def penalize(host: str = "*") -> None:
    """Server signalled throttling: halve host's cap for PENALTY_S (stacking)."""
    try:
        with _Locked() as st:
            now = time.time()
            pen = st.setdefault("penalty", {})
            pen_n = st.setdefault("penalty_n", {})
            pen_n[host] = min(pen_n.get(host, 0) + 1 if pen.get(host, 0) > now
                              else 1, 4)
            pen[host] = now + PENALTY_S
    except Exception:                               # noqa: BLE001
        pass
