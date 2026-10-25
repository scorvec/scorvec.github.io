#!/usr/bin/env python3
"""Which ECMWF open-data cycle is ready: chosen by what the Google mirror HAS, never by the clock.

WHY. A strat.yml run dispatched at 17:59Z took the 12Z cycle because its first steps were landing: the old
probe (`download_aifs.latest_run`) asked for ONE message of step 6. ECMWF uploads a cycle to the mirror over ~60 min
(AIFS-ENS 12Z: first object 17:59Z, day-15 index 18:54Z, last perturbed-member GRIB 18:58Z), so every product then
asked for steps that did not exist yet, got 404s and rendered nothing. Anything that picks "the latest cycle" early in
that window does the same.

THE RULE. A cycle is ready for a product only if, on the Google mirror, for every file stream the product reads
(`Need`: model / stream / file type / the LAST step it needs):
  1. the last step's `.index` exists,
  2. the last step's `.grib2` exists, and
  3. every step of the stream is fully on the mirror: each of its steps (`Need.every`, ECMWF's open-data step set)
     has both its `.index` and its `.grib2` (one small listing GET per directory). Neither the day-15 index nor the
     day-15 GRIB is the last thing to land: steps are uploaded in PARALLEL, and on 2026-09-26 12Z eight AIFS-ENS pf
     INDEXES (258h..348h) appeared after the day-15 GRIB, the last object 8.7 min after it (11.5 min on 09-24 12Z).
     A GCS object only appears once its upload is complete, so "every step has both files" is exact.
If the listing cannot be read, (3) falls back to a settle time: the last step's GRIB must be `SETTLE_MIN` old.

resolve() walks back from the newest cycle whose init time has passed to at most `back` cycles before it (default 2,
env ECMWF_CYCLE_BACK) and returns the first ready one, logging to stderr why each newer one was passed over. It never
guesses: if none is ready it raises CycleUnavailable. Cycles are 00/12Z unless `every` = 6 (the AIFS-ENS 06/18Z runs
reach day 15 too; the stratosphere products take them).

GOOGLE MIRROR ONLY: every probe goes to storage.googleapis.com/ecmwf-open-data (the object
HEADs, and the JSON listing of the same bucket). Never AWS, Azure or data.ecmwf.int, not even to probe.

    from cycles import resolve, is_ready, AIFS_ENS, IFS_ENS
    date, hh = resolve(AIFS_ENS)                    # e.g. ("20260928", "12")
    ok, why = is_ready("20260928", "12", IFS_ENS)

    python cycles.py --profile aifs-ens --github-output      # date=/time= for a workflow step
    python cycles.py --profile aifs-ens --out /tmp/cyc       # "YYYYMMDD HH" to a file
    python cycles.py --profile aifs-ens --every 6 --out /tmp/cyc   # 00/06/12/18Z
    python cycles.py --profile ifs-ens --date 20260928 --time 12 --check   # exit 0 iff ready
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rangefetch import MIRRORS, path_for  # (the same paths the fetchers read)

GOOGLE = MIRRORS["google"]  # https://storage.googleapis.com/ecmwf-open-data
BUCKET = GOOGLE.rstrip("/").rsplit("/", 1)[-1]  # ecmwf-open-data
LIST_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
BACK = int(os.environ.get("ECMWF_CYCLE_BACK", "2"))
SETTLE_MIN = float(os.environ.get("ECMWF_CYCLE_SETTLE_MIN", "20"))  # fallback only; worst lag seen 11.5 min
TIMEOUT = 20


@dataclass(frozen=True)
class Need:
    """One file stream a product reads, and the last step it needs from it."""

    model: str  # "aifs-ens" | "ifs" | "aifs-single"
    kind: str = "pf"  # file type in the object name: pf / cf (AIFS-ENS), ef (IFS-ENS), fc (aifs-single)
    stream: str = "enfo"  # "enfo" | "oper"
    step: int = 360
    every: tuple = tuple(range(0, 361, 6))  # the stream's published steps up to `step` (all must be on the mirror)
    resol: str = "0p25"
    levtype: str | None = None  # "pl" | "sfc": the 0.1-deg files are split by level type

    def rel(self, date: str, hh: str, step: int | None = None) -> str:
        return path_for(
            date,
            hh,
            self.model,
            self.step if step is None else step,
            self.kind,
            stream=self.stream,
            resol=self.resol,
            levtype=self.levtype,
        )

    def label(self) -> str:
        grid = "" if self.resol == "0p25" else f" {self.resol}-{self.levtype}"
        return f"{self.model}{grid} {self.stream}-{self.kind} +{self.step}h"


# What each product family reads. AIFS-ENS control and perturbed members are separate files that land separately
# (the control first), so both are required.
AIFS_ENS = (Need("aifs-ens", "pf"), Need("aifs-ens", "cf"))  # 0..360 h by 6: 61 steps
IFS_STEPS = tuple(range(0, 144, 3)) + tuple(range(144, 361, 6))  # 0..144 by 3, then by 6: 85 steps
IFS_ENS = (Need("ifs", "ef", every=IFS_STEPS),)  # control + 50 members in one -enfo-ef file per step
# IFS-ENS has begun to land: its first step is complete. It reaches the mirror over ~50 min (00/12Z: step 0 ~T+7.7 h,
# day 15 ~T+8.5 h), and the stratosphere jobs (strat-ifs.yml) fetch each step as it lands from here on.
IFS_ENS_LANDING = (Need("ifs", "ef", step=0, every=(0,)),)
AIFS_SINGLE = (Need("aifs-single", "fc", "oper"),)
# IFS HRES to day 10: the dynamic-tropopause maps (aam.yml) and the TC board's maps and phase space. The 00/12Z run
# reaches day 10 on the mirror ~T+7.5 h.
IFS_HRES_D10 = (Need("ifs", "fc", "oper", step=240, every=IFS_STEPS),)
# The IFS control to day 15 (00/12Z), ~T+7.6 h: the dynamic tropopause (aam.yml) and the other products of the control's
# 0.25-deg run.
IFS_CONTROL = (Need("ifs", "fc", "oper", step=360, every=IFS_STEPS),)
# The TC board (tc.yml): both ensembles' members to day 10 (each member's surface low is followed on its own ensemble's
# pressure) and IFS HRES to day 10. IFS-ENS day 10 is the last to land, ~T+8.6 h.
TC_BOARD = (
    Need("aifs-ens", "pf", step=240),
    Need("ifs", "ef", step=240, every=IFS_STEPS),
    *IFS_HRES_D10,
)
# The IFS control at 0.1 deg to day 10, both of its files (pressure levels and surface): the second stage of the TC
# storm maps (tc.yml) and the AR monitor's IFS IVT (ar-det.yml). It reaches the mirror ~2 h after the 0.25-deg run
# (00Z: day 10 by ~09:40Z, 12Z by ~21:40Z; measured 20-22 Oct 2026).
IFS_CONTROL_0P1 = tuple(
    Need("ifs", "fc", "oper", step=240, every=IFS_STEPS, resol="0p1", levtype=lt) for lt in ("pl", "sfc")
)
# IFS-ENS at 0.1 deg, the surface file at the 6-hourly steps to day 15 (the world ensemble maps' 2 m temperature,
# ensmaps-ifs.yml). The 0.1-deg ensemble files carry the 50 perturbed members only and reach the mirror ~4.5 h after the
# 0.25-deg run (00Z ~13:20Z, 12Z ~01:20Z).
IFS_ENS_0P1 = (Need("ifs", "ef", step=360, every=tuple(range(6, 361, 6)), resol="0p1", levtype="sfc"),)
PROFILES = {
    "aifs-ens": AIFS_ENS,  # strat, mjo, aam, ar
    "ifs-ens": IFS_ENS,
    # strat-ifs.yml can start: the stratosphere page's AIFS-ENS cycle complete and IFS-ENS begun to land on it
    "strat-ifs": AIFS_ENS + IFS_ENS_LANDING,
    "aifs-single": AIFS_SINGLE,
    "aifs-compare": AIFS_SINGLE + AIFS_ENS,  # aifs-compare: single vs control vs ensemble mean
    "ifs-control": IFS_CONTROL,  # aam.yml
    "tc": TC_BOARD,  # tc.yml
    "ifs-0p1": IFS_CONTROL_0P1,  # tc.yml, ar-det.yml: whether the IFS control's maps can be drawn at 0.1 deg
    "ifs-ens-0p1": IFS_ENS_0P1,  # ensmaps-ifs.yml
}


class CycleUnavailable(RuntimeError):
    """No cycle within the look-back window is ready on the Google mirror."""


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _head(url: str) -> tuple[int, dt.datetime | None]:
    """(HTTP status, Last-Modified). Status 0 = transport error."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            lm = r.headers.get("Last-Modified")
            when = None
            if lm:
                from email.utils import parsedate_to_datetime

                when = parsedate_to_datetime(lm).astimezone(dt.UTC)
            return r.status, when
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:  # noqa: BLE001
        return 0, None


def _list_names(prefix: str) -> list[str] | None:
    """Object names under `prefix` in the Google bucket (JSON API, same host). None if the listing failed."""
    names: list[str] = []
    token = None
    for _ in range(20):  # a cycle directory is ~250 objects; 20 pages is a hard stop
        q = {"prefix": prefix, "fields": "items(name),nextPageToken", "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        try:
            with urllib.request.urlopen(f"{LIST_API}?{urllib.parse.urlencode(q)}", timeout=TIMEOUT) as r:
                d = json.load(r)
        except Exception:  # noqa: BLE001
            return None
        names += [it["name"] for it in d.get("items", [])]
        token = d.get("nextPageToken")
        if not token:
            return names
    return None


def _listing(date: str, hh: str, need: Need) -> set[str] | None:
    """The object names in the directory of need's file stream; None if the listing failed."""
    names = _list_names(need.rel(date, hh).rsplit("/", 1)[0] + "/")
    return None if names is None else set(names)


def complete_steps(date: str, hh: str, need: Need) -> set[int] | None:
    """The steps of need.every up to need.step that are completely on the Google mirror - both the .index and the
    .grib2, each of which appears only once its upload is complete - from one listing; None if the listing failed. A
    fetch that follows a run as it lands (ifs_ens.land) takes each of these as soon as it appears."""
    have = _listing(date, hh, need)
    if have is None:
        return None
    return {
        st
        for st in need.every
        if st <= need.step and all(need.rel(date, hh, st) + x in have for x in (".index", ".grib2"))
    }


def _incomplete(date: str, hh: str, need: Need) -> tuple[list[str] | None, int]:
    """(steps of need.every missing their .index or .grib2, steps expected); (None, n) if the listing failed."""
    want = [st for st in need.every if st <= need.step]
    have = _listing(date, hh, need)
    if have is None:
        return None, len(want)
    miss = []
    for st in want:
        rel = need.rel(date, hh, st)
        parts = [ext for ext in (".index", ".grib2") if rel + ext not in have]
        if parts:
            miss.append(f"{st}h" + ("" if len(parts) == 2 else parts[0]))
    return miss, len(want)


def is_ready(
    date: str, hh: str, needs=AIFS_ENS, settle_min: float = SETTLE_MIN, now: dt.datetime | None = None
) -> tuple[bool, str]:
    """(ready, reason) for cycle date/hh against every Need. Google mirror only."""
    hh = f"{int(hh):02d}"
    now = now or dt.datetime.now(dt.UTC)
    notes = []
    for need in needs:
        base = f"{GOOGLE}/{need.rel(date, hh)}"
        st_i, _ = _head(base + ".index")
        if st_i != 200:
            return False, f"{need.label()}: .index {'HTTP ' + str(st_i) if st_i else 'unreachable'}"
        st_g, when = _head(base + ".grib2")
        if st_g != 200:
            return False, f"{need.label()}: .index is up but .grib2 {'HTTP ' + str(st_g) if st_g else 'unreachable'}"
        missing, nwant = _incomplete(date, hh, need)
        if missing is None:  # listing unavailable: fall back to the settle time
            age = (now - when).total_seconds() / 60 if when else -1
            if age < settle_min:
                return False, (
                    f"{need.label()}: listing unavailable and the last GRIB is only {age:.0f} min old "
                    f"(< {settle_min:.0f})"
                )
            notes.append(f"{need.model}-{need.kind} settled {age:.0f} min")
        elif missing:
            return False, (
                f"{need.label()}: {len(missing)} of {nwant} steps not fully uploaded "
                f"({','.join(missing[:6])}{'…' if len(missing) > 6 else ''})"
            )
        else:
            notes.append(f"{need.model}-{need.kind} {nwant}/{nwant} steps")
    return True, "complete (" + "; ".join(notes) + ")"


def candidates(now: dt.datetime | None = None, back: int = BACK, every: int = 12) -> list[tuple[str, str]]:
    """The newest cycle (every `every` hours from 00Z) whose init time has passed, then `back` cycles before it. Only
    ORDERS the probes."""
    now = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC).replace(tzinfo=None)
    c = now.replace(minute=0, second=0, microsecond=0, hour=now.hour // every * every)
    out = []
    for _ in range(back + 1):
        out.append((f"{c:%Y%m%d}", f"{c:%H}"))
        c -= dt.timedelta(hours=every)
    return out


def resolve(
    needs=AIFS_ENS,
    date: str | None = None,
    time_: str | None = None,
    back: int = BACK,
    settle_min: float = SETTLE_MIN,
    now: dt.datetime | None = None,
    what: str = "",
    every: int = 12,
) -> tuple[str, str]:
    """The newest READY cycle (see the module docstring). An explicit date AND time are returned as given (a manual
    override), with a warning when that cycle is not ready. Raises CycleUnavailable when nothing in the window is."""
    what = what or " + ".join(n.label() for n in needs)
    if date and time_:
        ok, why = is_ready(date, time_, needs, settle_min, now)
        _log(
            f"cycle {date} {int(time_):02d}Z given explicitly for {what}: "
            f"{'ready' if ok else 'NOT READY on the Google mirror - ' + why}"
        )
        return date, f"{int(time_):02d}"
    skipped = []
    for d, h in candidates(now, back, every):
        ok, why = is_ready(d, h, needs, settle_min, now)
        if ok:
            msg = f"cycle {d} {h}Z chosen for {what}: {why}"
            if skipped:
                msg += "; passed over " + "; ".join(f"{sd} {sh}Z ({sw})" for sd, sh, sw in skipped)
            _log(msg)
            return d, h
        skipped.append((d, h, why))
        _log(f"  {d} {h}Z not ready: {why}")
    raise CycleUnavailable(
        f"no cycle ready on the Google mirror for {what} within {back} cycles of the newest: "
        + "; ".join(f"{sd} {sh}Z ({sw})" for sd, sh, sw in skipped)
    )


def wait_ready(
    date: str, hh: str, needs, wait_min: float, poll_s: float = 60.0, settle_min: float = SETTLE_MIN
) -> bool:
    """Poll until the cycle is ready or wait_min runs out (strat-ifs waits up to 45 min for IFS-ENS day 15)."""
    deadline = time.monotonic() + 60.0 * wait_min
    while True:
        ok, why = is_ready(date, hh, needs, settle_min)
        if ok:
            _log(f"  {date} {hh}Z ready: {why}")
            return True
        if time.monotonic() >= deadline:
            _log(f"  {date} {hh}Z still not ready after {wait_min:.0f} min: {why}")
            return False
        _log(f"  {date} {hh}Z not ready ({why}); waiting")
        time.sleep(poll_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", choices=sorted(PROFILES), default="aifs-ens")
    ap.add_argument("--date", default="")
    ap.add_argument("--time", default="")
    ap.add_argument("--back", type=int, default=BACK)
    ap.add_argument("--every", type=int, choices=(6, 12), default=12, help="hours between the cycles tried")
    ap.add_argument("--check", action="store_true", help="only report whether --date/--time is ready (exit 0/1)")
    ap.add_argument("--out", help="write 'YYYYMMDD HH' to this file")
    ap.add_argument("--github-output", action="store_true", help="append date=/time= to $GITHUB_OUTPUT")
    a = ap.parse_args()
    needs = PROFILES[a.profile]
    if a.check:
        if not (a.date and a.time):
            ap.error("--check needs --date and --time")
        ok, why = is_ready(a.date, a.time, needs)
        _log(f"{a.date} {a.time}Z {a.profile}: {'ready' if ok else 'not ready'} - {why}")
        return 0 if ok else 1
    try:
        d, h = resolve(needs, a.date or None, a.time or None, back=a.back, what=a.profile, every=a.every)
    except CycleUnavailable as e:
        _log(f"ERROR: {e}")
        return 1
    if a.out:
        Path(a.out).write_text(f"{d} {h}")
    if a.github_output and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"date={d}\ntime={h}\n")
    if not a.out and not a.github_output:
        print(d, h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
