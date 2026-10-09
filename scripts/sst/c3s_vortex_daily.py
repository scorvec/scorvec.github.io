#!/usr/bin/env python3
"""The December polar vortex from DAILY C3S data: multi-model member plumes and pre-registered daily skill tests.

User 2026-10-09, after the monthly test (c3s_u10_skill.py, release c3s-u10-v1) found only DECEMBER significant (October
starts, multi-model mean r 0.66, Holm p 0.005): "use multi-model mean for december and follow up with daily data to make
similar charts to c3s". The monthly card is c3s_vortex.py; this is the daily follow-up: the charts the SEAS5 daily product
(seas5_ssw.py) draws - member plumes against the model's own normal, the anomaly panel, the latitude section, the
risk/timing figure - for the C3S multi-model ensemble, and four tests fixed below BEFORE any daily model data were read.

PRE-REGISTRATION (written 2026-10-09 before any daily C3S hindcast was fetched; not to be changed after seeing results)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
DATA. Zonal-mean zonal wind at 10 hPa, 30-90°N on the 2° rows (60°N is the vortex index), for the OCTOBER nominal start
of each C3S centre's current system, built from exactly the initialisations the CDS monthly product uses for that
nominal start (checked against the monthly tables' member dates): UKMO 610 = 9, 17, 25 Sep + 1 Oct (7 members each);
JMA 4 = 13 + 28 Sep; BoM 2 = 23-30 Sep + 1 Oct (3 each); Météo-France 9, DWD 22, CMCC 4, ECCC 4 and 5 = 1 Oct. CDS
`seasonal-original-pressure-levels`, u at 10 hPa, area [90, -180, 30, 180], 2° grid, the 00 UTC steps every 24 h from
the first day after each initialisation to 31 March (or the system's last step). The 12 UTC steps exist for only some
systems and are not used (one consistent sampling across systems). ECMWF SEAS5 (51) is NOT fetched again: its hindcast
is the AWS daily copy already reduced in seas5-ref-v1/hc_ssw_10.npz (25 members, daily means, same 2° rows) and its
operational years seas5-ref-v1/aws_u60_op_10.npz. Hindcast winters 1993/94-2016/17 (n = 24, the C3S common period).
Every request is sequential through cds_seq.retrieve (one key; yields to the `seas5` Actions group); each request's
table is uploaded to the release c3s-daily-u10-v1 as it lands, the raw GRIB kept in scripts/sst/data/c3s_vortex_daily.
TRUTH: MERRA-2 daily u60N 10 hPa (scripts/strat/reference/strat_history.nc `u10`), the series every vortex product uses.

QC (each unit, before use): exactly one field per (initialisation, member, step); |u| < 150 m/s and finite; 60°N row
present; member count per initialisation equal to the monthly product's for the same system and year (c3s-u10-v1 table);
December-mean of each member's daily values against the monthly dataset's December mean of the same member: the
ensemble-mean difference is reported (00 UTC sampling vs a monthly mean of all steps) and a centre whose |mean
difference| exceeds 1 m/s or whose year-to-year r < 0.95 is flagged and left out of everything below.

ANOMALIES. For each system, the normal at each valid day is the mean over all members of the OTHER 23 hindcast winters
(leave-one-year-out), smoothed with an 11-day running mean; a member's anomaly is its wind minus that normal. Members are
aligned by VALID date. The observed normal is MERRA-2's mean over the other 23 winters, same smoothing. Multi-model mean
(MMM) = equal-weight mean over the 8 centres of each centre's ensemble-mean anomaly (ECCC: mean of components 4 and 5).
Member-level multi-model probabilities = equal-weight mean over centres of each centre's member fraction.

TESTS (one family, Holm α = 0.05 over the tests actually run; p one-sided, 20,000 permutations of the 24 outcomes):
  T1  December weak-vortex spell. Observed event: at least 7 consecutive days between 1 and 31 December with MERRA-2 u60N
      below U_obs, the lower tercile of all MERRA-2 December daily values 1993-2016 pooled. A member's event: 7 consecutive
      December days below U_s, the lower tercile of its own system's pooled hindcast member December days in the other 23
      winters (a model-relative threshold, so mean bias does not decide it). Forecast = MMM member probability. Statistic:
      ROC area (calibration-free), H1 > 0.5.
  T2  December mean from the daily data: Pearson r of the MMM December-mean daily anomaly with MERRA-2's December-mean
      anomaly, H1 r > 0. (The consistency check with the monthly result; it uses the current systems.)
  T3  December major SSW: a Charlton-Polvani (2007) major warming with its central date in 1-31 December (detection as
      seas5_ssw.detect, on 1 Oct - 30 Mar for both the members and MERRA-2, so the final-warming rule sees the same days).
      Forecast = MMM member share; statistic ROC area. RUN ONLY IF MERRA-2 has at least 5 such winters and 5 without among
      the 24; otherwise T3 is reported as untestable (observed count and model base rate only) and leaves the family.
  T4  Major SSW between 1 December and 28/29 February, same statistic and the same at-least-5/5 rule.
Descriptive only (never called significant): Brier skill score of the raw MMM probabilities against the leave-one-out
base rate with a year-bootstrap 95 % interval (10,000), reliability counts, the observed and model base rates.
POWER: for T1/T3/T4 the ROC area detectable with 80 % power at the family's Holm level given the observed event counts
(Hanley-McNeil standard error); for T2 the r detectable at 80 % power, n = 24 (Fisher z).
PUBLICATION: the daily charts are shown on stratosphere.html; a probability (T1, T3, T4) is presented as a forecast only if
its test is Holm-significant, otherwise as "the models' view" under a no-track-record banner; no insignificant statistic
is presented as a finding.

    python c3s_vortex_daily.py units                  # list every CDS unit and whether its table is held
    python c3s_vortex_daily.py fetch [--only KEY]     # sequential CDS → raw GRIB → QC'd table (+ release upload)
    python c3s_vortex_daily.py fetch-issue --issue 202610   # the issue's forecast units (Actions, c3s-vortex.yml)
    python c3s_vortex_daily.py test                   # the pre-registered family → data/c3s_vortex_daily_skill.json
    python c3s_vortex_daily.py build --issue 202610   # figures + assets/sst/data/c3s_vortex_daily.json (Actions)
"""
from __future__ import annotations

import pyproj  # noqa: F401  first, before eccodes/xarray load another libproj
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
SITE = Path(os.environ.get("SST_SITE_ROOT", REPO))
DIR = Path(os.environ.get("C3SVD_DIR", HERE / "data" / "c3s_vortex_daily"))
RAW = DIR / "raw"
TAG = os.environ.get("C3SVD_TAG", "")                         # release to upload each table to
RELEASE = "c3s-daily-u10-v1"
YEARS = list(range(1993, 2017))
LATS = np.arange(90.0, 29.0, -2.0)                            # 90 … 30 °N, as seas5_ssw's section
AREA, GRID = [90, -180, 30, 180], [2.0, 2.0]
NDAY = 183                                                    # day index 0 = 1 Oct … 182 = 1 Apr
DATASET = "seasonal-original-pressure-levels"

# (key, centre, system, label, [(month, [days])], last step offered, members per init expected or None)
SYSTEMS = [
    ("ukmo_610", "ukmo", "610", "UKMO 610", [("09", ["09", "17", "25"]), ("10", ["01"])], 4344),
    ("mf_9", "meteo_france", "9", "Météo-France 9", [("10", ["01"])], 4344),
    ("dwd_22", "dwd", "22", "DWD 22", [("10", ["01"])], 4344),
    ("cmcc_4", "cmcc", "4", "CMCC 4", [("10", ["01"])], 4416),
    ("eccc_4", "eccc", "4", "ECCC 4 (GEM5-NEMO)", [("10", ["01"])], 4320),
    ("eccc_5", "eccc", "5", "ECCC 5 (CanESM5)", [("10", ["01"])], 4320),
    ("jma_4", "jma", "4", "JMA 4", [("09", ["13", "28"])], 5160),
    ("bom_2", "bom", "2", "BoM 2", [("09", [f"{d:02d}" for d in range(23, 31)]), ("10", ["01"])], 5160),
]
SYS = {s[0]: s for s in SYSTEMS}
CENTRES = [("ECMWF", ["ecmwf_51"]), ("UKMO", ["ukmo_610"]), ("Météo-France", ["mf_9"]), ("DWD", ["dwd_22"]),
           ("CMCC", ["cmcc_4"]), ("ECCC", ["eccc_4", "eccc_5"]), ("JMA", ["jma_4"]), ("BoM", ["bom_2"])]


# ── units ────────────────────────────────────────────────────────────────────
def _steps(year: int, month: str, days: list[str], last: int) -> list[str]:
    """00 UTC steps every 24 h from day 1 after the EARLIEST initialisation to 31 March, capped at the system's last."""
    d0 = dt.date(year, int(month), int(min(days)))
    end = dt.date(year + 1, 3, 31)
    h = min((end - d0).days * 24, last)
    return [str(x) for x in range(24, h + 1, 24)]


def units(years=None) -> list[dict]:
    out = []
    for key, centre, system, _, groups, last in SYSTEMS:
        for y in (years or YEARS):
            for gi, (month, days) in enumerate(groups):
                name = f"{key}_{y}" + (f"_g{gi}" if len(groups) > 1 else "")
                out.append(dict(unit=name, system=key, year=y, req={
                    "originating_centre": centre, "system": system, "variable": ["u_component_of_wind"],
                    "pressure_level": ["10"], "year": [str(y)], "month": [month], "day": days,
                    "leadtime_hour": _steps(y, month, days, last), "area": AREA, "grid": GRID, "data_format": "grib"}))
    return out


def table_path(unit: str) -> Path:
    return DIR / f"ud_{unit}.npz"


def raw_path(unit: str) -> Path:
    return RAW / f"ud_{unit}.grib"


def upload(*paths: Path) -> None:
    if not TAG:
        return
    gh = "/opt/homebrew/bin/gh" if Path("/opt/homebrew/bin/gh").exists() else "gh"
    for _ in range(3):
        r = subprocess.run([gh, "release", "upload", TAG, "--repo", os.environ.get("GITHUB_REPOSITORY", "scorvec/scorvec.github.io"),
                            "--clobber", *map(str, paths)], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"    uploaded {', '.join(p.name for p in paths)} → {TAG}", flush=True)
            return
        print(f"    upload failed: {r.stderr.strip()[:300]}", flush=True); time.sleep(10)
    print("::warning::release upload failed (the table stays local)", flush=True)


# ── GRIB → table ─────────────────────────────────────────────────────────────
def reduce(unit: str) -> dict:
    """[member, day, lat] zonal means on the 2° rows, members ordered by (init, number); strict QC."""
    import eccodes
    recs = {}
    with open(raw_path(unit), "rb") as fh:
        while True:
            h = eccodes.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                g = lambda k: eccodes.codes_get(h, k)                       # noqa: E731
                if int(g("level")) != 10:
                    raise ValueError(f"level {g('level')}")
                dd, tt = int(g("dataDate")), int(g("dataTime"))
                num = int(g("number"))
                step = int(eccodes.codes_get(h, "endStep"))
                ni, nj = int(g("Ni")), int(g("Nj"))
                la1, la2 = float(g("latitudeOfFirstGridPointInDegrees")), float(g("latitudeOfLastGridPointInDegrees"))
                v = eccodes.codes_get_values(h).astype(float)
                if eccodes.codes_get_long(h, "bitmapPresent"):
                    v[v == eccodes.codes_get_double(h, "missingValue")] = np.nan
                v = v.reshape(nj, ni)
                lo1, lo2 = float(g("longitudeOfFirstGridPointInDegrees")), float(g("longitudeOfLastGridPointInDegrees"))
                if ni > 1 and abs((lo2 - lo1) % 360) < 1e-3:                # −180 and 180 both present
                    v = v[:, :-1]
                lats = np.linspace(la1, la2, nj)
                if abs(la1 - 90) > 1e-3 or abs(la2 - 30) > 1e-3 or nj != len(LATS):
                    raise ValueError(f"rows {la1}..{la2} ({nj})")
                if tt != 0:
                    raise ValueError(f"init time {tt}")
                key = (dd, num, step)
                if key in recs:
                    raise ValueError(f"duplicate field {key}")
                recs[key] = v.mean(axis=1)
            finally:
                eccodes.codes_release(h)
    if not recs:
        raise ValueError("no fields")
    inits = sorted({(d, n) for d, n, _ in recs})
    d00 = min(d for d, _ in inits)
    y0 = d00 // 10000
    oct1 = dt.date(y0, 10, 1)
    u = np.full((len(inits), NDAY, len(LATS)), np.nan, np.float32)
    for (d, n, step), zm in recs.items():
        if step % 24:
            raise ValueError(f"step {step} not a 00 UTC step")
        idate = dt.date(d // 10000, d // 100 % 100, d % 100)
        vday = (idate + dt.timedelta(hours=step) - oct1).days
        if 0 <= vday < NDAY:
            u[inits.index((d, n)), vday] = zm
    fin = u[np.isfinite(u)]
    if fin.size == 0 or np.abs(fin).max() >= 150:
        raise ValueError(f"values out of range (max |u| {np.abs(fin).max() if fin.size else 'none'})")
    per_init = {}
    for d, n in inits:
        per_init[d] = per_init.get(d, 0) + 1
    DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(table_path(unit), u=u, lats=LATS, init=np.array([d for d, _ in inits]),
                        number=np.array([n for _, n in inits]), oct1=str(oct1))
    i60 = int(np.argmin(np.abs(LATS - 60)))
    dec = slice((dt.date(y0, 12, 1) - oct1).days, (dt.date(y0, 12, 31) - oct1).days + 1)
    info = dict(members=len(inits), per_init=per_init, fields=len(recs),
                dec_mean_60n=float(np.nanmean(u[:, dec, i60])), dec_days=int(np.isfinite(u[:, dec, i60]).all(0).sum()))
    print(f"  {unit}: {info}", flush=True)
    return info


# ── fetch ────────────────────────────────────────────────────────────────────
def fetch(only: str | None, years=None) -> int:
    from cds_seq import retrieve
    t0 = time.time(); missing = []
    todo = [u for u in units(years) if (only is None or u["system"] == only)]
    for i, u in enumerate(todo):
        if table_path(u["unit"]).exists():
            continue
        print(f"[{i + 1}/{len(todo)}] {u['unit']}: {len(u['req']['leadtime_hour'])} steps, days {','.join(u['req']['day'])}/{u['req']['month'][0]}", flush=True)
        ok = retrieve(DATASET, u["req"], raw_path(u["unit"]), u["unit"])
        if not ok:
            print(f"::warning::{u['unit']}: CDS request failed or no data", flush=True); missing.append(u["unit"]); continue
        try:
            reduce(u["unit"])
        except Exception as e:                                           # noqa: BLE001
            print(f"::warning::{u['unit']}: QC/decode failed ({e}); raw GRIB kept", flush=True)
            missing.append(u["unit"]); continue
        upload(table_path(u["unit"]))
    print(f"fetch done in {(time.time() - t0) / 3600:.2f} h; missing: {missing or 'none'}", flush=True)
    return 0 if not missing else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["units", "fetch", "reduce"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--unit", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "units":
        for u in units():
            if a.only and u["system"] != a.only:
                continue
            print(u["unit"], len(u["req"]["leadtime_hour"]), u["req"]["month"], ",".join(u["req"]["day"]),
                  "held" if table_path(u["unit"]).exists() else ("raw" if raw_path(u["unit"]).exists() else ""))
        return 0
    if a.cmd == "reduce":
        reduce(a.unit); return 0
    return fetch(a.only)


if __name__ == "__main__":
    raise SystemExit(main())
