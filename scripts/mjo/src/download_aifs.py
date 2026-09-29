"""
Download AIFS-ENS fields from ECMWF open data.

Variables retrieved:
  - u : zonal wind at 850 and 200 hPa

Note: AIFS-ENS does not output OLR/TTR, so only wind-based RMM is computed.

The AIFS ensemble ('aifs-ens') runs twice daily (00Z / 12Z) with
50 perturbed members + 1 control, stream 'enfo'.

Usage:
    python src/download_aifs.py                        # latest available run
    python src/download_aifs.py --date 20240601 --time 00
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf                                    # shared ECMWF download manager


DATA_DIR = Path(__file__).parent.parent / "data" / "aifs"

# Steps to day 15. Default DAILY (24-hourly): the RMM only needs daily values
# (it builds daily means via step//24 groupby, so one sample/day is enough), and
# daily steps cut the ~1 MB/s-bandwidth-bound download ~4x (≈4 GB → ≈1 GB).
# Set AIFS_STEP_HOURS=6 for the old 4-samples/day behavior if ever needed.
# Step 0 (the analysis) is included so lead_day 0 exists — the zero-lag "truth"
# point archived to obs_history (daily steps otherwise start at lead_day 1).
_STEP_HOURS = int(os.environ.get("AIFS_STEP_HOURS", "24"))
STEPS = [0] + list(range(_STEP_HOURS, 361, _STEP_HOURS))


def rmm_steps(init_time) -> list:
    """Lead steps that land on 00Z VALID times for any init hour, so the RMM samples one
    consistent 00Z point per day across BOTH 00Z and 12Z runs (run-to-run comparable —
    same forecast valid points). 00Z init → 0,24,…,360; 12Z init → 0,12,36,…,348 (forecast
    leads offset +12 h to reach the next 00Z). Step 0 (the init analysis) is always kept —
    it's the zero-lag 'truth' archived to obs_history. Same daily resolution / bandwidth."""
    first = (24 - int(init_time) % 24) % 24                   # 0 for 00Z, 12 for 12Z
    anchored = list(range(first, 361, 24))                    # forecast leads valid at 00Z
    return anchored if first == 0 else [0] + anchored         # 12Z: prepend the 12Z analysis

# Every download goes through the shared store (scripts/ecmwf/store.py), which fetches from the Google Cloud mirror
# ONLY (user rule 2026-09-06). Until 2026-09-29 this module also carried its own legacy retriever (_retrieve /
# retrieve_parallel over ecmwf-opendata) whose SOURCES defaulted to aws,azure,ecmwf (AIFS_SOURCES env); nothing had
# called it since the store took over, and it was deleted rather than left as a way back to the other mirrors.


def latest_run() -> tuple[str, str]:
    """(date, time) of the newest AIFS-ENS cycle that is COMPLETELY on the Google mirror.

    Delegates to the shared resolver (scripts/ecmwf/cycles.py): day 15 of both the perturbed members and the
    control indexed, and every step's GRIB uploaded, falling back up to two cycles. Until 2026-09-28 this probed
    one message of step 6, which is up ~55 min before the cycle is, so a run dispatched early in the upload window
    took a cycle whose later steps were 404 (strat.yml, 17:59Z on 2026-09-28). Logs its choice to stderr only, so
    callers that parse stdout are unaffected; raises RuntimeError when no cycle in the window is ready.
    """
    import cycles
    try:
        return cycles.resolve(cycles.AIFS_ENS, what="AIFS-ENS")
    except cycles.CycleUnavailable as e:
        raise RuntimeError(str(e)) from e


def download(date: str, time: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"aifs_{date}_{time}z"

    print(f"Downloading AIFS-ENS {date} {time}Z via shared store …")

    # u on the 200/850 levels the RMM needs — a light ~1 GB pull, kept separate from the
    # heavy 13-level AAM download so the MJO critical path is never blocked behind it. The
    # AAM builder fetches the other 11 levels and concatenates 200/850 back in (no dup).
    # The RMM reader expects data/aifs/<stem>.{pf,cf}.u.grib2, so hardlink the cache file there.
    cyc = ecmwf.Cycle(date, time); steps = tuple(rmm_steps(time))   # 00Z-anchored valid times
    for typ in ("pf", "cf"):
        src = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "u", "pl", ecmwf.LEVELS_RMM, steps))
        dst = out_dir / f"{stem}.{typ}.u.grib2"
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)                       # free same-filesystem pointer
        except OSError:
            shutil.copy2(src, dst)
        print(f"  {typ} u-wind: {dst.name} -> {Path(src).name}")

    # tp for the precip pseudo-OLR channel of the RMM (accumulated from init;
    # step 0 is skipped — the accumulation is zero there). Non-fatal: on any
    # failure the RMM falls back to wind-only, so the critical path survives
    # a tp outage on the mirrors.
    tp_steps = tuple(s for s in steps if s > 0)
    for typ in ("pf", "cf"):
        try:
            src = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "tp", "sfc", (), tp_steps))
            dst = out_dir / f"{stem}.{typ}.tp.grib2"
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            print(f"  {typ} tp: {dst.name} -> {Path(src).name}")
        except Exception as e:                      # noqa: BLE001
            print(f"  {typ} tp unavailable ({repr(e)[:60]}) — RMM will be wind-only")

    print("Download complete.")


def download_ifs(date: str, time: str, out_dir: Path) -> bool:
    """Fetch IFS-ENS u@850/200 + tp for the RMM comparison overlay.

    IFS-ENS is disseminated ~1-2 h AFTER AIFS-ENS, so this is best-effort:
    returns True only when both wind files landed (tp stays optional — the
    RMM falls back to wind-only per model). The caller re-tries on later
    polls via the plots/<stem>.png.missing sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ifs_{date}_{time}z"
    cyc = ecmwf.Cycle(date, time)
    steps = tuple(rmm_steps(time))
    # pf ONLY: IFS 0.25-deg enfo open data serves no separate cf through this
    # path (the -enfo-cf index 404s and the client finds no cf entries;
    # verified 2026-08-16) — 50 perturbed members are ample for the overlay.
    ok = True
    try:
        src = ecmwf.ensure(cyc, ecmwf.Spec("ifs", "pf", "u", "pl",
                                           ecmwf.LEVELS_RMM, steps))
        dst = out_dir / f"{stem}.pf.u.grib2"
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        print(f"  IFS pf u-wind: {dst.name}")
    except Exception as e:                          # noqa: BLE001
        print(f"  IFS pf u unavailable ({repr(e)[:60]})")
        ok = False
    if ok:
        try:
            tp_steps = tuple(s for s in steps if s > 0)
            src = ecmwf.ensure(cyc, ecmwf.Spec("ifs", "pf", "tp", "sfc", (), tp_steps))
            dst = out_dir / f"{stem}.pf.tp.grib2"
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            print(f"  IFS pf tp: {dst.name}")
        except Exception as e:                      # noqa: BLE001
            print(f"  IFS pf tp unavailable ({repr(e)[:60]}) — wind-only")
    return ok



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYYMMDD (default: latest available)")
    parser.add_argument("--time", default=None, help="00 or 12 (default: latest available)")
    parser.add_argument("--out-dir", default="data/aifs")
    args = parser.parse_args()

    if args.date is None or args.time is None:
        print("No date/time specified — finding latest available AIFS-ENS run …")
        date, time = latest_run()
        print(f"  Latest run: {date} {time}Z")
    else:
        date, time = args.date, args.time

    download(date, time, Path(args.out_dir))


if __name__ == "__main__":
    main()
