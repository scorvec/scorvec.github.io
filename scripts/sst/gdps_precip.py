#!/usr/bin/env python3
"""ECCC GDPS precipitation over the Colombian hydro basins.

The Canadian Global Deterministic Prediction System, from Environment and
Climate Change Canada's open Datamart. Two things make it worth testing
against AIFS-ENS on this problem:

  * **0.15 deg** grid against AIFS's 0.25. Over Colombian terrain that
    matters - roughly 4 cells across an ANTIOQUIA-sized catchment instead
    of 1.4, and basin rain here is limited by how well a model resolves
    a cordillera, not by how many members it runs.
  * `Precip-Accum24h` is published directly, so daily totals need no
    differencing and no accumulation-window bookkeeping.

Against that, it is DETERMINISTIC - one run, no spread - so it cannot
produce the ensemble fan the inflow model consumes. The honest question is
therefore not "does it replace AIFS-ENS" but "does its rain verify better,
and does blending it with the AIFS mean help".

Datamart keeps ~30 days, which is the whole reason this is worth doing
now: verification does not have to wait for an archive to accumulate the
way the stage-A quantile maps do.

    python scripts/sst/gdps_precip.py --date 20260820 --cycle 00
    python scripts/sst/gdps_precip.py --backfill 25        # last 25 days

Output: ~/colombia_hydro/raw/fcst_rain/gdps_YYYYMMDD_HHz.json.gz
        same schema as the aifs_/ifs_ cycles, with a single member.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hydro_region_rain import region_weights_energy                # noqa: E402

PRIV = Path.home() / "colombia_hydro"
ARCH = PRIV / "raw" / "fcst_rain"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
BASE = "https://dd.weather.gc.ca/{date}/WXO-DD/model_gdps/15km/{cyc}/{lead:03d}"
FILE = ("{date}T{cyc}Z_MSC_GDPS_Precip-Accum24h_Sfc_LatLon0.15_"
        "PT{lead:03d}H.grib2")
LEADS = list(range(24, 241, 24))          # 1..10 days, 24 h accumulations
_W = {}


def fetch(date: str, cyc: str, lead: int) -> Path | None:
    url = (BASE.format(date=date, cyc=cyc, lead=lead) + "/"
           + FILE.format(date=date, cyc=cyc, lead=lead))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scorvec-hydro/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    tmp = Path(tempfile.mkstemp(suffix=".grib2")[1])
    tmp.write_bytes(data)
    return tmp


def basin_means(path: Path) -> dict | None:
    """Energy-weighted basin mean rain, mm/day, from one GRIB."""
    import xarray as xr
    try:
        d = xr.open_dataset(path, engine="cfgrib",
                            backend_kwargs={"indexpath": ""})
    except Exception:                                   # noqa: BLE001
        return None
    v = list(d.data_vars)[0]
    a = d[v]
    lat = d.latitude.values
    lon = d.longitude.values
    key = (len(lat), len(lon))
    if key not in _W:
        # region_weights_energy wants ascending axes; GDPS lon is 0..360
        _W[key] = region_weights_energy(np.sort(lon), np.sort(lat), ORDER)
    W = _W[key]
    if not W:
        return None
    si, sj = np.argsort(lat), np.argsort(lon)
    g = a.values[np.ix_(si, sj)]
    return {r: float((g * np.asarray(W[r]).reshape(g.shape)).sum())
            for r in ORDER}


def one_cycle(date: str, cyc: str, force=False) -> bool:
    dest = ARCH / f"gdps_{date}_{cyc}z.json.gz"
    if dest.exists() and not force:
        return True
    valid, rows = [], {r: [] for r in ORDER}
    d0 = datetime.strptime(date, "%Y%m%d")
    got = 0
    for lead in LEADS:
        p = fetch(date, cyc, lead)
        if p is None:
            continue
        bm = basin_means(p)
        p.unlink(missing_ok=True)
        Path(str(p) + ".923a8.idx").unlink(missing_ok=True)
        if bm is None:
            continue
        # PT{lead}H is the 24 h total ENDING at that lead, so from a 00Z run
        # it covers the UTC day that STARTS at lead-24 - i.e. PT024H is the
        # init day itself, not the day after. Getting this wrong shifts every
        # GDPS forecast one day late against IMERG's UTC daily fields and
        # would make the model look far worse than it is. AIFS's valid[0] is
        # likewise its init date, so the two now index leads identically.
        valid.append((d0 + timedelta(hours=lead - 24)).strftime("%Y-%m-%d"))
        for r in ORDER:
            rows[r].append(round(bm[r], 2))
        got += 1
    if got < 5:
        print(f"  {date} {cyc}Z: only {got} leads — skipped", flush=True)
        return False
    out = {"model": "gdps", "init_date": date, "init_hh": cyc,
           "valid": valid, "n_members": 1,
           "basins": {r: [rows[r]] for r in ORDER},
           "basins_energy": {r: [rows[r]] for r in ORDER},
           "note": "ECCC GDPS 15km deterministic, Precip-Accum24h, "
                   "energy-weighted basin means (mm/day)"}
    ARCH.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"  {date} {cyc}Z: {got} leads -> {dest.name}", flush=True)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--cycle", default="00")
    ap.add_argument("--backfill", type=int, default=0,
                    help="fetch the last N days that Datamart still holds")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.backfill:
        today = datetime.now(timezone.utc).replace(tzinfo=None)
        ok = 0
        for k in range(a.backfill, -1, -1):
            ds = (today - timedelta(days=k)).strftime("%Y%m%d")
            ok += bool(one_cycle(ds, a.cycle, a.force))
        print(f"backfill: {ok}/{a.backfill + 1} cycles archived")
    else:
        date = a.date or datetime.now(timezone.utc).strftime("%Y%m%d")
        one_cycle(date, a.cycle, a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
