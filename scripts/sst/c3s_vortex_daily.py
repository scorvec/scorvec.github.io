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
N = len(YEARS)
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


# CP07 detection, copied verbatim from seas5_ssw.detect (importing seas5_ssw pulls the SEAS5 outlook stack)
def detect(u: np.ndarray, dates) -> dict:
    import pandas as pd
    """CP07 on one daily u60N10 series → {"ssw": [dates], "fw": date or None}. NaN days count as missing:
    a westerly or easterly run is broken by them, so a series with gaps is judged on what it has."""
    u = np.asarray(u, float)
    west = u >= 0
    n = len(u)
    events = []
    last = None
    for i in range(1, n):
        d = dates[i]
        if not (np.isfinite(u[i]) and np.isfinite(u[i - 1])) or not (west[i - 1] and not west[i]):
            continue
        if not ((d.month >= 11) or (d.month <= 3)):
            continue
        if last is not None:
            run = 0; j = i - 1
            while j > last and west[j] and np.isfinite(u[j]):
                run += 1; j -= 1
            if run < 20:
                continue
        # final-warming exclusion: 10 consecutive westerly days before 30 April
        apr30 = pd.Timestamp(year=d.year if d.month <= 4 else d.year + 1, month=4, day=30)
        run, ok = 0, False
        for j in range(i + 1, n):
            if dates[j] > apr30:
                break
            run = run + 1 if (np.isfinite(u[j]) and west[j]) else 0
            if run >= 10:
                ok = True; break
        if ok:
            events.append(d); last = i
    # final warming: the last westerly→easterly transition with no 10-day westerly return afterwards
    fw = None
    for i in range(n - 1, 0, -1):
        if np.isfinite(u[i]) and np.isfinite(u[i - 1]) and west[i - 1] and not west[i]:
            run, back = 0, False
            for j in range(i + 1, n):
                run = run + 1 if (np.isfinite(u[j]) and west[j]) else 0
                if run >= 10:
                    back = True; break
            if not back:
                fw = dates[i]
            break
    return {"ssw": events, "fw": fw}


# ── the issue's forecast units (Actions) ─────────────────────────────────────
SEAS5_REF = Path(os.environ.get("SEAS5_REF_DIR", DIR / "seas5"))       # seas5-ref-v1: hc_ssw_10.npz
SEAS5_SSW = HERE / "data" / "seas5" / "ssw"                             # seas5.yml's cached fc_u10_{issue}.grib
U10_DIR = Path(os.environ.get("C3SU10_DIR", HERE / "data" / "c3s_u10"))  # the monthly tables (QC)
ASSETS = SITE / "assets" / "sst"
OUT_JSON = ASSETS / "data" / "c3s_vortex_daily.json"
NPERM, NBOOT = 20000, 10000
I60 = int(np.argmin(np.abs(LATS - 60)))


def issue_units(issue: str) -> list[dict]:
    """Forecast mode: the CDS keeps every member of a nominal start under day 01 of the issue month."""
    y, m = int(issue[:4]), issue[4:]
    out = []
    for key, centre, system, _, groups, last in SYSTEMS:
        out.append(dict(unit=f"{key}_{issue}", system=key, year=y, req={
            "originating_centre": centre, "system": system, "variable": ["u_component_of_wind"], "pressure_level": ["10"],
            "year": [str(y)], "month": [m], "day": ["01"], "leadtime_hour": _steps(y, m, ["01"], last),
            "area": AREA, "grid": GRID, "data_format": "grib"}))
    return out


def fetch_issue(issue: str) -> int:
    from cds_seq import retrieve
    miss = []
    for u in issue_units(issue):
        if table_path(u["unit"]).exists():
            continue
        if not retrieve(DATASET, u["req"], raw_path(u["unit"]), u["unit"], tries=2,
                        max_wait_h=float(os.environ.get("CDS_MAX_WAIT_H", "0.5"))):
            print(f"  {u['unit']}: not on the CDS yet (or the request failed)", flush=True); miss.append(u["system"]); continue
        try:
            reduce(u["unit"]); upload(table_path(u["unit"]))
        except Exception as e:                                               # noqa: BLE001
            print(f"::warning::{u['unit']}: QC/decode failed ({e})", flush=True); miss.append(u["system"])
        try:
            raw_path(u["unit"]).unlink()                                     # the table is the product; keep runner disk free
        except OSError:
            pass
    print(f"issue {issue}: missing {miss or 'none'}", flush=True)
    return 0


def seas5_forecast(issue: str) -> np.ndarray | None:
    """SEAS5's 51 members from seas5.yml's cached 12-hourly fc_u10 (90-30°N, 2°): daily mean of the 12 and 24 h steps of
    each day (the right-closed rule of seas5_ssw), → [member, day from 1 Oct, lat]."""
    src = SEAS5_SSW / f"fc_u10_{issue}.grib"
    if not src.exists():
        return None
    import eccodes
    acc = {}
    with open(src, "rb") as fh:
        while True:
            h = eccodes.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                num = int(eccodes.codes_get(h, "number")); step = int(eccodes.codes_get(h, "endStep"))
                ni, nj = int(eccodes.codes_get(h, "Ni")), int(eccodes.codes_get(h, "Nj"))
                la1, la2 = eccodes.codes_get_double(h, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get_double(h, "latitudeOfLastGridPointInDegrees")
                v = eccodes.codes_get_values(h).astype(float).reshape(nj, ni)
                lo1, lo2 = eccodes.codes_get_double(h, "longitudeOfFirstGridPointInDegrees"), eccodes.codes_get_double(h, "longitudeOfLastGridPointInDegrees")
                if ni > 1 and abs((lo2 - lo1) % 360) < 1e-3:
                    v = v[:, :-1]
                lats = np.linspace(la1, la2, nj)
                order = [int(np.argmin(np.abs(lats - L))) for L in LATS]
                acc.setdefault(num, {})[step] = v.mean(1)[order]
            finally:
                eccodes.codes_release(h)
    mems = sorted(acc)
    out = np.full((len(mems), NDAY, len(LATS)), np.nan, np.float32)
    for i, m in enumerate(mems):
        for d in range(NDAY):
            a, b = acc[m].get(24 * d + 12), acc[m].get(24 * d + 24)
            if a is not None and b is not None:
                out[i, d] = (a + b) / 2
    return out


# ── loading ──────────────────────────────────────────────────────────────────
def _year_table(key: str, tag) -> np.ndarray | None:
    """[member, day, lat] for one system and one hindcast year (or issue string): its group tables concatenated."""
    _, _, _, _, groups, _ = SYS[key]
    names = [f"{key}_{tag}"] if (len(groups) == 1 or isinstance(tag, str)) else [f"{key}_{tag}_g{gi}" for gi in range(len(groups))]
    parts = []
    for n in names:
        p = table_path(n)
        if not p.exists():
            return None
        parts.append(np.load(p)["u"])
    return np.concatenate(parts, axis=0)


def load_hindcast() -> tuple[dict, dict]:
    """{key: [24 arrays (member, day, lat)]} for every system with all 24 winters, and {key: reason} for the others."""
    out, why = {}, {}
    z = np.load(SEAS5_REF / "hc_ssw_10.npz")
    yrs = [int(v) for v in z["years"]]
    assert np.allclose(z["lats"], LATS), "SEAS5 section rows differ"
    sec = z["sec"]
    out["ecmwf_51"] = [sec[yrs.index(y), :, :NDAY, :].astype(np.float32) for y in YEARS]
    for key in SYS:
        arrs = [_year_table(key, y) for y in YEARS]
        if any(a is None for a in arrs):
            why[key] = f"{sum(a is None for a in arrs)} of {N} winters not fetched yet"
            continue
        out[key] = arrs
    return out, why


def merra2_daily() -> np.ndarray:
    """[24 winters, NDAY] MERRA-2 u60N 10 hPa from 1 Oct."""
    import pandas as pd
    import xarray as xr
    ds = xr.open_dataset(REPO / "scripts" / "strat" / "reference" / "strat_history.nc")
    s = pd.Series(ds["u10"].values.astype(float), index=pd.to_datetime(ds["time"].values)); ds.close()
    out = np.full((N, NDAY), np.nan)
    for i, y in enumerate(YEARS):
        d = pd.date_range(f"{y}-10-01", periods=NDAY, freq="D")
        out[i] = s.reindex(d).values
    return out


def winter_dates(y: int):
    import pandas as pd
    return pd.date_range(f"{y}-10-01", periods=NDAY, freq="D")


DEC = slice(61, 92)                                                          # 1-31 December (1 Oct = day 0)
LAST_DAY = 180                                                              # 30 March (29 March in leap winters)


def smooth(a: np.ndarray, w: int = 11, axis: int = 0) -> np.ndarray:
    """nan-aware centred running mean along `axis`, shrinking at the edges."""
    a = np.moveaxis(np.asarray(a, float), axis, 0)
    out = np.full_like(a, np.nan)
    h = w // 2
    for d in range(a.shape[0]):
        out[d] = np.nanmean(a[max(0, d - h):d + h + 1], axis=0)
    return np.moveaxis(out, 0, axis)


def normals(arrs: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """Leave-one-year-out normals [day, lat] for each year, and the all-years normal; 11-day running mean."""
    S = np.stack([np.nansum(a, axis=0) for a in arrs])                       # (year, day, lat)
    C = np.stack([np.isfinite(a).sum(axis=0) for a in arrs]).astype(float)
    tot_S, tot_C = S.sum(0), C.sum(0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        loo = [smooth((tot_S - S[i]) / (tot_C - C[i])) for i in range(len(arrs))]
        full = smooth(tot_S / tot_C)
    return loo, full


# ── events ───────────────────────────────────────────────────────────────────
def max_run_below(u: np.ndarray, thr: float) -> int:
    best = run = 0
    for v in u:
        run = run + 1 if (np.isfinite(v) and v < thr) else 0
        best = max(best, run)
    return best


def ssw_dates(u: np.ndarray, y: int) -> list:
    d = winter_dates(y)
    return detect(np.asarray(u[:LAST_DAY + 1], float), d[:LAST_DAY + 1])["ssw"]


def in_window(dates, y: int, which: str) -> bool:
    import pandas as pd
    if which == "dec":
        a, b = pd.Timestamp(f"{y}-12-01"), pd.Timestamp(f"{y}-12-31")
    else:
        a, b = pd.Timestamp(f"{y}-12-01"), pd.Timestamp(f"{y + 1}-03-01") - pd.Timedelta(days=1)
    return any(a <= pd.Timestamp(t) <= b for t in dates)


# ── statistics ───────────────────────────────────────────────────────────────
RNG = np.random.default_rng(20261009)
PERM = np.array([RNG.permutation(N) for _ in range(NPERM)])
BOOT = RNG.integers(0, N, size=(NBOOT, N))


def auc(p: np.ndarray, o: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(p); n1 = int(o.sum()); n0 = len(o) - n1
    return float((r[o == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_test(p: np.ndarray, o: np.ndarray) -> dict:
    from scipy.stats import rankdata
    o = o.astype(int); n1 = int(o.sum()); n0 = len(o) - n1
    r = rankdata(p)
    a0 = (r[o == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    op = o[PERM].astype(bool)
    ap = ((r[None, :] * op).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    pval = float((np.sum(ap >= a0 - 1e-12) + 1) / (NPERM + 1))
    bs = []
    for ii in BOOT[:4000]:
        oo = o[ii]
        if 0 < oo.sum() < len(oo):
            bs.append(auc(p[ii], oo))
    return dict(auc=float(a0), auc_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p=pval, n_events=n1, n_non=n0)


def bss(p: np.ndarray, o: np.ndarray) -> dict:
    def one(pp, oo):
        clim = (oo.sum() - oo) / (len(oo) - 1)
        b, bc = np.mean((pp - oo) ** 2), np.mean((clim - oo) ** 2)
        return 1 - b / bc if bc > 0 else np.nan
    v = one(p, o)
    bb = np.array([one(p[ii], o[ii]) for ii in BOOT])
    return dict(bss=float(v), bss_ci95=[float(np.nanpercentile(bb, 2.5)), float(np.nanpercentile(bb, 97.5))],
                mean_forecast=float(p.mean()), obs_rate=float(o.mean()))


def corr_test(f: np.ndarray, o: np.ndarray) -> dict:
    a = f - f.mean(); b = o - o.mean()
    r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    def rows(x, y):
        x = x - x.mean(-1, keepdims=True); y = y - y.mean(-1, keepdims=True)
        return (x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    rp = rows(np.broadcast_to(f, (NPERM, N)), o[PERM])
    rb = rows(f[BOOT], o[BOOT])
    return dict(r=r, r_ci95=[float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))],
                p=float((np.sum(rp >= r) + 1) / (NPERM + 1)))


def holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k]); m = len(keys); out = {}; run = 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k])); out[k] = run
    return out


def auc_power(n1: int, n0: int, alpha: float) -> float:
    """ROC area detectable with 80 % power, one-sided level alpha (Hanley-McNeil SE under the alternative)."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha) + norm.ppf(0.8)
    for A in np.arange(0.5, 1.0, 0.001):
        q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
        se = np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0))
        if (A - 0.5) / se >= z:
            return float(A)
    return 1.0


def r_power(n: int, alpha: float) -> float:
    from scipy.stats import norm
    return float(np.tanh((norm.ppf(1 - alpha) + norm.ppf(0.8)) / np.sqrt(n - 3)))


# ── the pre-registered family ────────────────────────────────────────────────
def centre_members(H: dict) -> dict:
    return {c: [k for k in ks if k in H] for c, ks in CENTRES}


def analyse(H: dict, why: dict) -> dict:
    """Everything the tests need, per system: LOYO anomalies, T1 thresholds and member events, SSW dates; then the
    multi-model quantities per winter and the family."""
    O = merra2_daily()
    o_loo, o_full = normals([O[i][None, :, None] for i in range(N)])
    o_loo = [v[:, 0] for v in o_loo]; o_full = o_full[:, 0]
    oa = np.array([O[i] - o_loo[i] for i in range(N)])
    U_obs = float(np.nanquantile(O[:, DEC], 1 / 3))
    obs_t1 = np.array([max_run_below(O[i, DEC], U_obs) >= 7 for i in range(N)])
    obs_ssw = [ssw_dates(O[i], y) for i, y in enumerate(YEARS)]
    obs_t3 = np.array([in_window(d, y, "dec") for d, y in zip(obs_ssw, YEARS)])
    obs_t4 = np.array([in_window(d, y, "djf") for d, y in zip(obs_ssw, YEARS)])
    sysd = {}
    for key, arrs in H.items():
        loo, full = normals(arrs)
        an = [arrs[i] - loo[i][None] for i in range(N)]
        dec_pool = [arrs[i][:, DEC, I60] for i in range(N)]
        t1f, t3f, t4f, dmean = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        thr = np.zeros(N)
        for i, y in enumerate(YEARS):
            other = np.concatenate([dec_pool[j].ravel() for j in range(N) if j != i])
            thr[i] = np.nanquantile(other, 1 / 3)
            u = arrs[i][:, :, I60]
            ev1 = [max_run_below(m[DEC], thr[i]) >= 7 for m in u]
            ss = [ssw_dates(m, y) for m in u]
            t1f[i] = np.mean(ev1)
            t3f[i] = np.mean([in_window(d, y, "dec") for d in ss])
            t4f[i] = np.mean([in_window(d, y, "djf") for d in ss])
            dmean[i] = np.nanmean(an[i][:, DEC, I60])
        sysd[key] = dict(t1=t1f, t3=t3f, t4=t4f, dec_anom=dmean, thr=thr, normal=full, n_members=[int(a.shape[0]) for a in arrs],
                         dec_raw=np.array([np.nanmean(arrs[i][:, DEC, I60]) for i in range(N)]))
    cm = centre_members(H)
    cen = {c: ks for c, ks in cm.items() if ks and len(ks) == len(dict(CENTRES)[c])}
    def mmm(field):
        return np.mean([np.mean([sysd[k][field] for k in ks], axis=0) for ks in cen.values()], axis=0)
    return dict(O=O, oa=oa, o_full=o_full, U_obs=U_obs, obs_t1=obs_t1, obs_t3=obs_t3, obs_t4=obs_t4, obs_ssw=obs_ssw,
                sys=sysd, centres=cen, missing=why, mmm=dict(t1=mmm("t1"), t3=mmm("t3"), t4=mmm("t4"), dec_anom=mmm("dec_anom")))


def qc_monthly(A: dict) -> dict:
    """Daily December means of each system's ensemble mean against the monthly dataset's (c3s_u10 tables, Oct start)."""
    out = {}
    try:
        import c3s_vortex as V
    except Exception as e:                                                   # noqa: BLE001
        return {"error": str(e)}
    for key, d in A["sys"].items():
        try:
            mon, _ = V.hindcast(key, 10, 3)
        except Exception as e:                                               # noqa: BLE001
            out[key] = dict(error=str(e)[:120]); continue
        diff = d["dec_raw"] - mon
        r = float(np.corrcoef(d["dec_raw"], mon)[0, 1])
        out[key] = dict(mean_diff=float(diff.mean()), r=r, ok=bool(abs(diff.mean()) <= 1.0 and r >= 0.95))
    return out


def family(A: dict) -> dict:
    res, pv = {}, {}
    n1, n0 = int(A["obs_t1"].sum()), N - int(A["obs_t1"].sum())
    res["T1"] = dict(name="December weak-vortex spell (7+ days below the lower tercile)", obs_events=n1,
                     threshold_obs=A["U_obs"], **auc_test(A["mmm"]["t1"], A["obs_t1"]), **bss(A["mmm"]["t1"], A["obs_t1"].astype(float)))
    pv["T1"] = res["T1"]["p"]
    res["T2"] = dict(name="December mean from daily data (r with MERRA-2)", **corr_test(A["mmm"]["dec_anom"], np.nanmean(A["oa"][:, DEC], axis=1)))
    pv["T2"] = res["T2"]["p"]
    for k, ob, nm in (("T3", A["obs_t3"], "major SSW with its central date in December"), ("T4", A["obs_t4"], "major SSW between 1 December and the end of February")):
        e = int(ob.sum())
        if e >= 5 and N - e >= 5:
            res[k] = dict(name=nm, obs_events=e, **auc_test(A["mmm"][k.lower()], ob), **bss(A["mmm"][k.lower()], ob.astype(float)))
            pv[k] = res[k]["p"]
        else:
            res[k] = dict(name=nm, obs_events=e, untestable=True, model_rate=float(A["mmm"][k.lower()].mean()),
                          note=f"{e} of {N} winters had one in MERRA-2: fewer than the 5 events (and 5 non-events) the pre-registration requires")
    adj = holm(pv)
    m = len(pv)
    for k, a in adj.items():
        res[k]["p_holm"] = a; res[k]["significant"] = bool(a < 0.05)
    for k in ("T1", "T3", "T4"):
        if not res[k].get("untestable"):
            res[k]["auc80"] = auc_power(res[k]["n_events"], res[k]["n_non"], 0.05 / m)
    res["T2"]["r80"] = r_power(N, 0.05 / m)
    return dict(tests=res, family_size=m, holm_alpha=0.05)


def test() -> dict:
    H, why = load_hindcast()
    if why:
        print(f"hindcast incomplete: {why}", flush=True)
    A = analyse(H, why)
    qc = qc_monthly(A)
    bad = [k for k, v in qc.items() if isinstance(v, dict) and v.get("ok") is False]
    if bad:
        print(f"QC flags {bad}: left out", flush=True)
        H = {k: v for k, v in H.items() if k not in bad}
        A = analyse(H, why)
    F = family(A)
    winters = [dict(year=y, t1=float(A["mmm"]["t1"][i]), t3=float(A["mmm"]["t3"][i]), t4=float(A["mmm"]["t4"][i]),
                    dec_anom=float(A["mmm"]["dec_anom"][i]), obs_t1=bool(A["obs_t1"][i]), obs_t3=bool(A["obs_t3"][i]),
                    obs_t4=bool(A["obs_t4"][i]), obs_ssw=[str(d.date()) for d in A["obs_ssw"][i]],
                    obs_dec_anom=float(np.nanmean(A["oa"][i, DEC]))) for i, y in enumerate(YEARS)]
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), n=N, years=[YEARS[0], YEARS[-1]],
               centres={c: ks for c, ks in A["centres"].items()}, missing=why, qc=qc, qc_excluded=bad,
               members={k: v["n_members"][0] for k, v in A["sys"].items()},
               base_rates=dict(obs_t1=float(A["obs_t1"].mean()), obs_t3=float(A["obs_t3"].mean()), obs_t4=float(A["obs_t4"].mean()),
                               model_t1=float(A["mmm"]["t1"].mean()), model_t3=float(A["mmm"]["t3"].mean()), model_t4=float(A["mmm"]["t4"].mean())),
               **F, winters=winters)
    return out, A, H


def report(R: dict) -> None:
    print(f"\n=== daily family ({R['family_size']} tests, Holm α 0.05), {R['n']} winters {R['years']} ===")
    print("centres:", R["centres"], "missing:", R["missing"] or "none")
    print("QC (daily Dec mean vs monthly):", {k: (round(v.get('mean_diff', 0), 2), round(v.get('r', 0), 3)) for k, v in R["qc"].items() if isinstance(v, dict)})
    for k, t in R["tests"].items():
        if t.get("untestable"):
            print(f"  {k} {t['name']}: UNTESTABLE - {t['note']}; model member rate {t['model_rate']:.2f}"); continue
        stat = f"AUC {t['auc']:.2f} {np.round(t['auc_ci95'], 2).tolist()}" if "auc" in t else f"r {t['r']:.2f} {np.round(t['r_ci95'], 2).tolist()}"
        extra = f", BSS {t['bss']:.2f} {np.round(t['bss_ci95'], 2).tolist()}" if "bss" in t else ""
        pw = f", 80 % power at {t.get('auc80', t.get('r80')):.2f}"
        print(f"  {k} {t['name']}: {stat}, p {t['p']:.4f}, Holm {t['p_holm']:.3f} {'SIGNIFICANT' if t['significant'] else 'n.s.'}{extra}{pw}"
              + (f"; events {t['obs_events']}" if "obs_events" in t else ""))
    print("base rates:", {k: round(v, 3) for k, v in R["base_rates"].items()})


# ── build: figures + card JSON (Actions) ─────────────────────────────────────
def _clean(v):
    if isinstance(v, dict): return {str(k): _clean(w) for k, w in v.items()}
    if isinstance(v, (list, tuple)): return [_clean(w) for w in v]
    if isinstance(v, np.ndarray): return _clean(v.tolist())
    if isinstance(v, (float, np.floating)): return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    return v


CCOL = {"ECMWF": "#c0392b", "UKMO": "#1f5fa8", "Météo-France": "#2e8b57", "DWD": "#7b4fb3", "CMCC": "#b8860b",
        "ECCC": "#8c564b", "JMA": "#d6618f", "BoM": "#17a2b8"}
QUORUM = 5


def wquant(v: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    """weighted quantiles along axis 0 of v [member, ...] (members weighted so every centre counts equally)."""
    out = np.full((len(qs),) + v.shape[1:], np.nan)
    flat = v.reshape(v.shape[0], -1)
    for j in range(flat.shape[1]):
        x = flat[:, j]; ok = np.isfinite(x)
        if ok.sum() < 3:
            continue
        o = np.argsort(x[ok]); xs = x[ok][o]; ws = w[ok][o]; cw = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
        out[:, j] = np.interp(qs, cw, xs)
    return out.reshape((len(qs),) + v.shape[1:])


def forecast_members(issue: str, A: dict, H: dict) -> dict:
    """The issue's members per system: raw, anomaly against the all-years normal, and the multi-model weights."""
    y = int(issue[:4])
    fc = {}
    for c, ks in A["centres"].items():
        parts = []
        for k in ks:
            u = seas5_forecast(issue) if k == "ecmwf_51" else _year_table(k, issue)
            if u is None:
                parts = None; break
            parts.append((k, u, u - A["sys"][k]["normal"][None]))
        if parts:
            fc[c] = parts
    return fc


def build(issue: str) -> int:
    import pandas as pd
    t0 = time.time()
    y = int(issue[:4]); key_iss = f"{issue[:4]}-{issue[4:]}"
    if int(issue[4:]) != 10:
        print(f"issue {key_iss}: the daily product is for October starts only", flush=True); return 0
    expected = [c for c, _ in CENTRES]
    held = sum(table_path(u["unit"]).exists() for u in units())
    if held < len(units()):                                   # no statistic is computed or shown before the sample is whole
        out = dict(generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), issue=key_iss, state="assembling",
                   hindcast_held=held, hindcast_units=len(units()), centres_expected=expected, centres_present=[], figures={})
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
        print(f"wrote {OUT_JSON} (assembling: {held} of {len(units())} hindcast tables)", flush=True)
        return 0
    R, A, H = test()
    report(R)
    complete_hc = set(R["centres"]) >= set(expected) and not R["qc_excluded"]
    fc = forecast_members(issue, A, H)
    present = [c for c in expected if c in fc]
    dates = winter_dates(y)
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), issue=key_iss, centres_expected=expected,
               centres_present=present, hindcast_complete=bool(complete_hc), skill=R, figures={})
    if not complete_hc:
        out["state"] = "assembling"
    elif len(present) < QUORUM:
        out["state"] = "waiting"
    else:
        out["state"] = "complete" if len(present) == len(expected) else "provisional"
        # members, equal weight per centre (within a centre, per component then per member)
        U, W, CEN, AN = [], [], [], []
        for c in present:
            for k, u, an in fc[c]:
                w = 1.0 / len(present) / len(fc[c]) / u.shape[0]
                U.append(u); AN.append(an); W += [w] * u.shape[0]; CEN += [c] * u.shape[0]
        U = np.concatenate(U); AN = np.concatenate(AN); W = np.array(W); CEN = np.array(CEN)
        corr = A["o_full"][None, :] + AN[:, :, I60]                   # each member on MERRA-2's normal
        # issue probabilities, per centre then averaged (the hindcast's own rules)
        pc = {}
        for c in present:
            vals = []
            for k, u, an in fc[c]:
                thr = float(np.nanquantile(np.concatenate([H[k][i][:, DEC, I60].ravel() for i in range(N)]), 1 / 3))
                ss = [ssw_dates(m, y) for m in u[:, :, I60]]
                vals.append(dict(t1=np.mean([max_run_below(m[DEC], thr) >= 7 for m in u[:, :, I60]]),
                                 t3=np.mean([in_window(d, y, "dec") for d in ss]), t4=np.mean([in_window(d, y, "djf") for d in ss]),
                                 first=[(pd.Timestamp(d[0]) - dates[0]).days if d else None for d in ss], n=u.shape[0],
                                 dec_anom=float(np.nanmean(an[:, DEC, I60]))))
            pc[c] = {k2: float(np.mean([v[k2] for v in vals])) for k2 in ("t1", "t3", "t4", "dec_anom")}
            pc[c]["members"] = int(sum(v["n"] for v in vals))
            pc[c]["first"] = [f for v in vals for f in v["first"]]
        mm = {k2: float(np.mean([pc[c][k2] for c in present])) for k2 in ("t1", "t3", "t4", "dec_anom")}
        out["forecast"] = dict(mmm=mm, centres=pc, members=int(len(U)))
        out["figures"] = render(issue, A, R, U, AN, W, CEN, corr, pc, mm, present, dates)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_clean(out), ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT_JSON} ({out['state']}; centres {present}) in {time.time() - t0:.0f} s", flush=True)
    return 0


def render(issue, A, R, U, AN, W, CEN, corr, pc, mm, present, dates) -> dict:
    import matplotlib
    import matplotlib.ticker  # noqa: F401
    import pandas as pd
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    figs = {}
    x = dates
    T = R["tests"]
    sig = {k: bool(T[k].get("significant")) for k in ("T1", "T2", "T3", "T4")}
    nots = "No demonstrated daily skill: the models' view, not a forecast" if not any(sig[k] for k in ("T1", "T3", "T4")) else ""
    O = A["O"]
    clim = np.nanpercentile(O, [10, 50, 90], axis=0)
    # ── A: plume on MERRA-2's normal + anomaly panel ──
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax.fill_between(x, clim[0], clim[2], color="#bdb8ae", alpha=0.45, lw=0, label="MERRA-2 1993–2016, 10–90 %")
    ax.plot(x, clim[1], color="#6f6b64", lw=1.4, ls="--", label="MERRA-2 median")
    for i in range(len(U)):
        ax.plot(x, corr[i], color="#8a4b2a", alpha=0.10, lw=0.6)
    q = wquant(corr, W, [0.1, 0.5, 0.9])
    ax.fill_between(x, q[0], q[2], color="#8a4b2a", alpha=0.20, lw=0, label="multi-model members, 10–90 %")
    ax.plot(x, np.nansum(corr * W[:, None], 0) / np.nansum(np.isfinite(corr) * W[:, None], 0), color="#8a4b2a", lw=2.8, label="multi-model mean")
    ax.axhline(0, color="#c2185b", lw=1.4, ls="--"); ax.text(x[2], 1.0, "easterly below", color="#c2185b", fontsize=9, va="bottom")
    ax.axvspan(x[61], x[91], color="#f3e9d8", alpha=0.6, lw=0, zorder=0)
    ax.text(x[76], ax.get_ylim()[1] if False else 0.98, "December", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9.5, color="#8a4b2a")
    ax.set_ylabel("u, 60°N 10 hPa (m/s)")
    ax.set_title(f"C3S multi-model, {dates[0]:%B %Y} start: daily vortex wind, each member's anomaly from its own model's normal "
                 f"placed on MERRA-2's normal ({len(present)} centres, {len(U)} members)", loc="left", fontsize=11.5)
    if nots:
        ax.text(0.995, 0.02, nots, transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color="#7a2e12",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff6e8", ec="#c9a227"))
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    ax.grid(axis="y", color="#e8e6e0", lw=0.6)
    for c in present:
        m = CEN == c
        a = np.nansum(AN[m][:, :, I60] * W[m][:, None], 0) / np.nansum(np.isfinite(AN[m][:, :, I60]) * W[m][:, None], 0)
        bx.plot(x, a, color=CCOL[c], lw=1.3, label=c)
    amm = np.nansum(AN[:, :, I60] * W[:, None], 0) / np.nansum(np.isfinite(AN[:, :, I60]) * W[:, None], 0)
    bx.plot(x, amm, color="#1a1a1a", lw=2.6, label="multi-model mean")
    bx.axhline(0, color="#999", lw=0.8); bx.axvspan(x[61], x[91], color="#f3e9d8", alpha=0.6, lw=0, zorder=0)
    bx.set_ylabel("anomaly (m/s)")
    bx.set_title("Ensemble-mean anomaly by centre, each against its own 1993–2016 hindcast normal (11-day smoothed)", loc="left", fontsize=10.5)
    bx.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=5)
    bx.grid(axis="y", color="#e8e6e0", lw=0.6)
    bx.xaxis.set_major_locator(mdates.MonthLocator()); bx.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    bx.set_xlim(x[0], x[LAST_DAY])
    fig.text(0.01, 0.005, "Copernicus Climate Change Service (C3S) seasonal multi-system, daily 00 UTC u at 10 hPa (SEAS5: daily means) · MERRA-2 (NASA GMAO) · scorvec.com",
             fontsize=8, color="#6f6b64")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    figs["u10"] = _save(fig, "c3s_vortex_daily_u10.webp")
    # ── B: latitude–time section of the multi-model anomaly ──
    cen_an = []
    for c in present:
        m = CEN == c
        cen_an.append(np.nansum(AN[m] * W[m][:, None, None], 0) / np.nansum(np.isfinite(AN[m]) * W[m][:, None, None], 0))
    cen_an = np.array(cen_an)                                                # (centre, day, lat)
    sec = cen_an.mean(0)
    agree = np.maximum((cen_an > 0).sum(0), (cen_an < 0).sum(0)) >= max(len(present) - 1, 1)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    lv = np.arange(-12, 13, 2)
    cf = ax.contourf(x[:LAST_DAY + 1], LATS, sec[:LAST_DAY + 1].T, levels=lv, cmap="RdBu_r", extend="both")
    ax.contourf(x[:LAST_DAY + 1], LATS, agree[:LAST_DAY + 1].T.astype(float), levels=[0.5, 1.5], colors="none", hatches=["..."])
    ax.contour(x[:LAST_DAY + 1], LATS, agree[:LAST_DAY + 1].T.astype(float), levels=[0.5], colors="#1a1a1a", linewidths=0.8)
    ax.axhline(60, color="#1a1a1a", lw=0.8, ls=":")
    cb = fig.colorbar(cf, ax=ax, pad=0.01); cb.set_label("m/s")
    ax.set_ylabel("latitude (°N)"); ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title(f"Multi-model mean anomaly of the zonal-mean wind at 10 hPa, 30–90°N ({len(present)} centres, each against its own normal); "
                 f"dotted inside the black contour: at least {max(len(present) - 1, 1)} of {len(present)} centres agree on the sign", loc="left", fontsize=10.5)
    fig.text(0.01, 0.005, "Agreement is a description of the models, not a significance test · C3S seasonal multi-system · scorvec.com", fontsize=8, color="#6f6b64")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    figs["section"] = _save(fig, "c3s_vortex_daily_section.webp")
    # ── C: risk and timing ──
    br = R["base_rates"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    labs = ["December weak spell\n(7+ days, lower third)", "Major SSW\nin December", "Major SSW\nDec–Feb"]
    keys = ["t1", "t3", "t4"]; tk = ["T1", "T3", "T4"]
    xx = np.arange(3)
    ax.bar(xx - 0.27, [mm[k] for k in keys], 0.26, color="#8a4b2a", label=f"this issue ({len(present)} centres)")
    ax.bar(xx, [br["model_" + k] for k in keys], 0.26, color="#c9b8a6", label="the models' 1993–2016 hindcast rate")
    ax.bar(xx + 0.27, [br["obs_" + k] for k in keys], 0.26, color="#4a4744", label="observed 1993–2016 (MERRA-2)")
    for i, k in enumerate(tk):
        t = T[k]
        lab = "untestable (too few events)" if t.get("untestable") else (f"skill: AUC {t['auc']:.2f}, Holm p {t['p_holm']:.2f}" + ("" if t["significant"] else " (n.s.)"))
        ax.text(i, -0.12, lab, ha="center", va="top", fontsize=8.5, color="#1a1a1a" if t.get("significant") else "#6f6b64", transform=ax.get_xaxis_transform())
    ax.set_xticks(xx); ax.set_xticklabels(labs); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("share of members / of winters"); ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_title("Event probabilities, multi-model (equal weight per centre)", loc="left", fontsize=10.5)
    ax.grid(axis="y", color="#e8e6e0", lw=0.6)
    firsts = []; fw = []
    for c in present:
        f = [v for v in pc[c]["first"] if v is not None]
        nmem = pc[c]["members"]
        firsts += f; fw += [1.0 / len(present) / nmem] * len(f)
    bins = np.arange(31, LAST_DAY + 2, 7)
    bx.hist(firsts, bins=bins, weights=fw, color="#8a4b2a", alpha=0.85)
    bx.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([(pd.Timestamp(f"{dates[0].year + (m < 10)}-{m:02d}-01") - dates[0]).days for m in (11, 12, 1, 2, 3)]))
    bx.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["Nov", "Dec", "Jan", "Feb", "Mar"]))
    bx.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    bx.set_title("When the first major SSW falls (share of members, weekly)", loc="left", fontsize=10.5)
    bx.grid(axis="y", color="#e8e6e0", lw=0.6)
    if nots:
        fig.text(0.5, 0.995, "No track record: in the 1993–2016 hindcast none of these probabilities passed its pre-registered test",
                 ha="center", va="top", fontsize=10, color="#7a2e12", bbox=dict(boxstyle="round,pad=0.3", fc="#fff6e8", ec="#c9a227"))
    fig.text(0.01, 0.005, "C3S seasonal multi-system · MERRA-2 (NASA GMAO) · Charlton & Polvani (2007) SSW definition · scorvec.com", fontsize=8, color="#6f6b64")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94 if nots else 1))
    figs["risk"] = _save(fig, "c3s_vortex_daily_risk.webp")
    # ── D: the hindcast, winter by winter ──
    wv = R["winters"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    yrs = [w["year"] for w in wv]
    ax.scatter([w["dec_anom"] for w in wv], [w["obs_dec_anom"] for w in wv], s=34, c=["#b4541f" if w["obs_t1"] else "#2f6690" for w in wv], edgecolor="white", lw=0.6, zorder=3)
    for w in wv:
        ax.annotate(str(w["year"])[2:], (w["dec_anom"], w["obs_dec_anom"]), fontsize=7, color="#6f6b64", xytext=(3, 2), textcoords="offset points")
    t2 = T["T2"]
    ax.set_title(f"December mean from the daily data: r {t2['r']:.2f} (95 % CI {t2['r_ci95'][0]:.2f} to {t2['r_ci95'][1]:.2f}), "
                 f"Holm p {t2['p_holm']:.3f}" + ("" if t2["significant"] else " n.s."), loc="left", fontsize=10)
    ax.set_xlabel("multi-model December anomaly (m/s)"); ax.set_ylabel("MERRA-2 December anomaly (m/s)")
    ax.axhline(0, color="#bbb", lw=0.8); ax.axvline(0, color="#bbb", lw=0.8); ax.grid(color="#eee", lw=0.5)
    t1 = T["T1"]
    order = np.argsort([w["t1"] for w in wv])
    bx.scatter(np.arange(N), [wv[i]["t1"] for i in order], s=38, c=["#b4541f" if wv[i]["obs_t1"] else "#9fb6c8" for i in order], edgecolor="white", lw=0.6)
    bx.set_xticks(np.arange(N)); bx.set_xticklabels([str(wv[i]["year"])[2:] for i in order], fontsize=7, rotation=90)
    bx.set_ylabel("multi-model probability of a weak spell"); bx.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    bx.set_title(f"Weak-spell probability, winters sorted; orange = a spell happened. AUC {t1['auc']:.2f}, Holm p {t1['p_holm']:.2f}"
                 + ("" if t1["significant"] else " n.s."), loc="left", fontsize=10)
    bx.grid(axis="y", color="#eee", lw=0.5)
    fig.text(0.01, 0.005, "Hindcast 1993/94–2016/17, October starts, leave-one-winter-out normals and thresholds · scorvec.com", fontsize=8, color="#6f6b64")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    figs["hindcast"] = _save(fig, "c3s_vortex_daily_hindcast.webp")
    return figs


def _save(fig, name: str) -> str:
    import matplotlib.pyplot as plt
    ASSETS.mkdir(parents=True, exist_ok=True)
    tmp = ASSETS / (name + ".png")
    fig.savefig(tmp, dpi=120); plt.close(fig)
    from PIL import Image
    Image.open(tmp).save(ASSETS / name, "WEBP", quality=88, method=6); tmp.unlink()
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["units", "fetch", "reduce", "fetch-issue", "test", "build"])
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
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
    if a.cmd == "fetch-issue":
        return fetch_issue(a.issue)
    if a.cmd == "test":
        R, _, _ = test(); report(R)
        p = DIR / "c3s_vortex_daily_skill.json"
        p.write_text(json.dumps(_clean(R), ensure_ascii=False, indent=1)); print("wrote", p); return 0
    if a.cmd == "build":
        return build(a.issue)
    return fetch(a.only)


if __name__ == "__main__":
    raise SystemExit(main())
