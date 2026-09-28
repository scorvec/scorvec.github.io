#!/usr/bin/env python3
"""Weather-regime labels from AIFS-ENS and IFS-ENS members, days 1-15 - the ECMWF half of the multi-model regime view
(2026-09-28). NO NEW DOWNLOADS: it reads the 500 hPa height files the stratosphere jobs already fetch from the
Google mirror into the shared store cache, and does nothing if they are not there.

  aifs   strat.yml (light job): wave1_maps.py's 25 perturbed members of z at 500/100/10 hPa and forecast_drip.py's
         control at 11 levels, daily steps 0-360 h -> 26 members
  ifs    strat-ifs.yml (maps job): the shared IFS-ENS gh file (ifs_ens.spec_gh), all 50 perturbed members (open data
         has no IFS control on pressure levels), daily steps 0-360 h -> 50 members

00Z cycles only (day d = the UTC day init + d - 1 = the mean of the 00Z fields at steps 24(d-1) and 24d, so every day
is a whole UTC day, as the GEPS/GEFS days are). Each field is box-averaged to the 2.5 deg NCEP grid (regime_core.box25)
and taken as an anomaly against the ERA5 1991-2020 day-of-year climatology (data/era5_z500_clim.npz) - the base the
regimes are defined on (NCEP R1 1991-2020 anomalies; the two reanalyses' anomalies agree far more closely than their
means do). Neither model has an open reforecast, so the drift cannot be removed the GEPS/GEFS way. Instead each run
archives its ensemble-mean zonal-mean anomaly by latitude and lead in each sector
(assets/regimes/data/regimes_drift_<model>.json); the lead-dependent drift is the mean over past runs of [lead-L
forecast - the day-1 forecast of the run started on that valid day], and it is subtracted, shrunk by n / (n + 10), at
the leads with >= 10 verified pairs. Until then the members are used as they come, and the JSON says so.

Then the same classification as every other model (regime_core: 5-day running mean along the lead, the set of each
valid day's season, "no regime" below pattern correlation 0.25) -> assets/regimes/data/regimes_members_<model>.json.

    python scripts/regimes/regimes_ecmwf.py --model aifs --date 20260928 --time 00
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import regime_core as RC                                                  # noqa: E402

G0 = 9.80665
NDAYS = 15
LEV11 = (1000, 925, 850, 700, 500, 300, 250, 200, 100, 50, 10)
OUT = REPO / "assets" / "regimes" / "data"
DRIFT_MIN, DRIFT_K, DRIFT_KEEP = 10, 10.0, 60
LABEL = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS"}


def files(model, date, time):
    """The cached GRIB files holding member z500 for this cycle, [(path, kind)]; nothing is fetched."""
    import store as E
    cyc = E.Cycle(date, time)
    steps = tuple(E.STEPS)
    if model == "aifs":
        pf = [E.path(cyc, E.Spec("aifs-ens", "pf", "z", "pl", (500, 100, 10), steps, 25)),
              E.path(cyc, E.Spec("aifs-ens", "pf", "z", "pl", LEV11, steps, 10))]
        cf = [E.path(cyc, E.Spec("aifs-ens", "cf", "z", "pl", LEV11, steps)),
              E.path(cyc, E.Spec("aifs-ens", "cf", "z", "pl", (500, 100, 10), steps))]
        got = [next((p for p in pf if p.exists()), None), next((p for p in cf if p.exists()), None)]
        return [p for p in got if p is not None]
    p = E.path(cyc, E.Spec("ifs", "pf", "gh", "pl", LEV11, steps))
    return [p] if p.exists() else []


def read_z500(paths):
    """{(member, step): (73, 144) height in m on the 2.5 deg grid (box mean at read time: 50 members x 16 steps at
    0.25 deg would be 6.6 GB)}."""
    import eccodes as ec
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            while True:
                h = ec.codes_grib_new_from_file(f)
                if h is None:
                    break
                try:
                    if int(ec.codes_get(h, "level")) != 500 or ec.codes_get(h, "shortName") not in ("z", "gh"):
                        continue
                    ni, nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
                    assert abs(ec.codes_get(h, "latitudeOfFirstGridPointInDegrees") - 90.0) < 1e-6
                    v = ec.codes_get_values(h).reshape(nj, ni)
                    lon0 = ec.codes_get(h, "longitudeOfFirstGridPointInDegrees") % 360.0
                    v = np.roll(v, int(round(lon0 / (360.0 / ni))), axis=1)
                    if ec.codes_get(h, "shortName") == "z":
                        v = v / G0
                    num = int(ec.codes_get(h, "number")) if ec.codes_get(h, "dataType") != "cf" else 0
                    out[(num, int(ec.codes_get(h, "step")))] = RC.box25(v, 360.0 / ni)
                finally:
                    ec.codes_release(h)
    return out


def era5_clim(dates):
    z = np.load(RC.DATA / "era5_z500_clim.npz")
    c, nh = z["coef"].astype("float64"), int(z["nharm"])
    out = []
    for d in dates:
        t = 2 * np.pi * (RC.doy(d) - 1) / 366.0
        x = [1.0] + [f(h * t) for h in range(1, nh + 1) for f in (np.cos, np.sin)]
        out.append(np.tensordot(np.array(x), c, 1))                         # (37, 144): 90N .. 0
    return np.stack(out)


def zonal_profile(anom_mean, sk):
    """(days, 73, 144) ensemble-mean anomaly -> (days, lat) sector zonal mean."""
    sec, _, _ = RC.sector_cut(anom_mean, RC.SECTORS[sk])
    return sec.mean(-1)


def drift_table(arch, init):
    """{sk: (drift (days, lat), n (days,))} from the archived runs verified by later runs' day 1."""
    runs = arch.get("runs", {})
    out = {}
    for sk in RC.SECTORS:
        acc, n = None, np.zeros(NDAYS)
        for d0, r in runs.items():
            if sk not in r:
                continue
            f = np.array(r[sk], dtype="float64")                             # (days, lat)
            for L in range(1, f.shape[0]):
                v = (RC.parse_date(d0) + dt.timedelta(days=L)).strftime("%Y%m%d")
                if v >= init.strftime("%Y%m%d") or v not in runs or sk not in runs[v]:
                    continue
                if acc is None:
                    acc = np.zeros((NDAYS, f.shape[1]))
                acc[L] += f[L] - np.array(runs[v][sk], dtype="float64")[0]
                n[L] += 1
        if acc is not None:
            out[sk] = (acc / np.maximum(n, 1)[:, None], n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("aifs", "ifs"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.time != "00":
        print(f"  {a.time}Z cycle: regimes use the 00Z cycles only (whole UTC days) - skipped"); return 0
    paths = files(a.model, a.date, a.time)
    if not paths:
        print(f"::warning::{LABEL[a.model]} {a.date} {a.time}Z: no cached 500 hPa member file - regimes skipped")
        return 0
    fields = read_z500(paths)
    members = sorted({m for m, _ in fields})
    steps = [24 * k for k in range(NDAYS + 1)]
    members = [m for m in members if all((m, s) in fields for s in steps)]
    if len(members) < 10:
        print(f"::warning::{LABEL[a.model]}: only {len(members)} complete members - regimes skipped"); return 0
    init = RC.parse_date(a.date)
    days = [init + dt.timedelta(days=d) for d in range(NDAYS)]
    clim = era5_clim(days)                                                   # (15, 37, 144)
    an = np.full((NDAYS, len(members), 73, 144), np.nan)
    for j, m in enumerate(members):
        b = np.stack([fields[(m, s)] for s in steps])                        # (16, 73, 144)
        an[:, j, :37] = 0.5 * (b[:-1, :37] + b[1:, :37]) - clim
    del fields
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arch_p = out / f"regimes_drift_{a.model}.json"
    try:
        arch = json.loads(arch_p.read_text())
    except Exception:                                                        # noqa: BLE001
        arch = {"runs": {}}
    with np.errstate(invalid="ignore"):
        mean = an.mean(1)                                                    # SH rows stay NaN (unused)
    arch["runs"][a.date] = {sk: np.round(zonal_profile(mean, sk), 1).tolist() for sk in RC.SECTORS}
    keep = sorted(arch["runs"])[-DRIFT_KEEP:]
    arch["runs"] = {k: arch["runs"][k] for k in keep}
    arch["note"] = ("per 00Z run: ensemble-mean 500 hPa anomaly (m, vs ERA5 1991-2020) averaged over each regime sector's "
                    "longitudes, by lead day (rows, day 1 first) and latitude (80N -> 20N, 2.5 deg); regimes_ecmwf.py")
    table = drift_table(arch, init)
    applied = {}
    labs = {}
    for sk, spec in RC.SECTORS.items():
        sec, lat, _ = RC.sector_cut(an, spec)                                # (15, M, lat, lon)
        dr = table.get(sk)
        if dr is not None:
            drift, n = dr
            ok = n >= DRIFT_MIN
            if ok.any():
                w = (n / (n + DRIFT_K))[:, None] * drift * ok[:, None]
                sec = sec - w[:, None, :, None]
            applied[sk] = {"leads_corrected": [int(i + 1) for i in np.flatnonzero(ok)],
                           "pairs": [int(x) for x in n]}
        v = RC.running_partial(sec, RC.SMOOTH, axis=0)
        nd, nm = v.shape[:2]
        ref = RC.Ref()
        lab, _ = ref.classify_dated(v.reshape(nd * nm, *v.shape[2:]), np.repeat(days, nm), sk)
        labs[sk] = lab.reshape(nd, nm)
    drift_note = ("lead-dependent drift removed (zonal mean by latitude, shrunk n/(n+10)) at the leads with >= 10 verified "
                  "runs: " + "; ".join(f"{sk} days {v['leads_corrected']}" for sk, v in applied.items() if v["leads_corrected"])
                  if any(v["leads_corrected"] for v in applied.values())
                  else f"no drift correction yet (accruing: {len(arch['runs'])} archived runs, needs 10 verified pairs per lead)")
    rec = RC.members_record(a.model, LABEL[a.model], init, len(members), days, labs,
                            notes=f"ECMWF {LABEL[a.model]} open data (Google Cloud mirror), 00Z; anomalies vs ERA5 "
                                  f"1991-2020; {drift_note}", max_age_days=2)
    rec["drift"] = {"applied": applied, "note": drift_note}
    (out / f"regimes_members_{a.model}.json").write_text(json.dumps(rec))
    arch_p.write_text(json.dumps(arch, separators=(",", ":")))
    ref = RC.Ref()
    for sk in RC.SECTORS:
        P = RC.family_probs(ref, sk, labs[sk], days)
        print(f"  {LABEL[a.model]} {sk}: {len(members)} members; " + "  ".join(
            f"wk{w['w0'] // 7 + 1} " + "/".join(f"{p * 100:.0f}" for p in w["p"]) for w in RC.weekly(P)))
    print(f"  -> {out / f'regimes_members_{a.model}.json'}; {drift_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
