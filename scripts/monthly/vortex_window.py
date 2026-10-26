#!/usr/bin/env python3
"""The lower polar vortex over the 31 days ending on the 7th of the target month, for an update of the month-ahead
outlook (forecast.py --update): the zonal-mean 50 hPa wind over 57-63N (the archive's u50v), from ERA5 as far as the
archive holds it, the IFS analyses (00 and 12 UTC) after that up to the ensembles' start, and then every member of the
four ensembles (IFS-ENS, AIFS-ENS, GEFS and GEPS, at 00 and 12 UTC valid times) to the window's end. Each member's
window mean is one value of the predictor; the ensembles weigh a quarter each.

The run used is the newest 00 or 12 UTC start that all four ensembles have on ens-anomalies.html (their manifests
name only runs that are complete). ECMWF open data from the Google Cloud mirror only (the first rows of each message,
scripts/ecmwf/rangefetch.fetch_rows), GEFS from NOAA's S3 bucket, GEPS from the MSC Datamart.

    python scripts/monthly/vortex_window.py --target 2026-11 --out vortex_202611.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "scripts" / "ensmaps")]
import ensmaps as E  # (imports pyproj before eccodes; puts scripts/ecmwf, lib and geps on the path)
import mo_grid as G

BAND = (57.0, 63.0)
ENSEMBLES = ("ifs", "aifs", "gefs", "geps")
HOURS = (0, 12)  # the valid times sampled in each day


def band_mean(v: np.ndarray, lat: np.ndarray) -> float:
    """The cos-weighted mean over the band's rows of a field (rows, lon), as build_archive's u50v."""
    rows = (lat >= BAND[0] - 1e-6) & (lat <= BAND[1] + 1e-6)
    w = np.cos(np.deg2rad(lat[rows]))
    return float((v[rows].mean(1) * w).sum() / w.sum())


def ecmwf_rows(date: str, hh: str, model: str, step: int, kind: str, stream: str = "enfo") -> dict:
    """{member: band mean} of u at 50 hPa from the first rows of each message (0.25 deg: rows 108-132 are 63-57N)."""
    import rangefetch as rf

    idx = rf.fetch_index(date, hh, model, step, kind, stream=stream)
    want = rf.select(idx, param="u", levelist=[50])
    if not want:
        raise RuntimeError(f"{model} {date} {hh}Z +{step} h: no u at 50 hPa in the index")
    rel = rf.path_for(date, hh, model, step, kind, stream=stream) + ".grib2"
    vals = rf.fetch_rows(rel, want, tuple(range(108, 133)), each=lambda v, lat, lon: band_mean(v, lat))
    return {int(e.get("number", 0)): x for e, x in zip(want, vals, strict=True)}


def ecmwf_members(model: str, date: str, hh: str, step: int) -> dict:
    if model == "ifs":
        return ecmwf_rows(date, hh, "ifs", step, "ef")
    return ecmwf_rows(date, hh, "aifs-ens", step, "cf") | ecmwf_rows(date, hh, "aifs-ens", step, "pf")


def gefs_members(date: str, hh: str, step: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    def one(mem: str) -> bytes:
        url = E.gefs_url(date, hh, mem, step)
        lines = [ln.split(":") for ln in E.s3_get(url + ".idx").decode().splitlines() if ln.strip()]
        i = next(i for i, p in enumerate(lines) if p[3] == "UGRD" and p[4] == "50 mb")
        return E.s3_get(url, (int(lines[i][1]), int(lines[i + 1][1]) - 1))

    with ThreadPoolExecutor(16) as ex:
        msgs = list(ex.map(one, E.GEFS_MEMBERS))
    out = {}
    for m, msg in enumerate(msgs):  # decoded in this thread (ensmaps.decode)
        _, v, inc = E.decode(msg, "")
        out[m] = band_mean(v, 90.0 - inc * np.arange(v.shape[0]))
    return out


def geps_members(date: str, hh: str, step: int) -> dict:
    out = {}
    for msg in E.split_grib(E.geps_file(date, hh, "UGRD_ISBL_0050", step)):
        num, v, inc = E.decode(msg, "perturbationNumber")
        out[num] = band_mean(v, 90.0 - inc * np.arange(v.shape[0]))
    return out


def analysis(day: dt.date) -> float:
    """The IFS analyses' daily mean (00 and 12 UTC, step 0 of the control run, stream oper)."""
    import rangefetch as rf

    vals = []
    for h in HOURS:
        idx = rf.fetch_index(f"{day:%Y%m%d}", f"{h:02d}", "ifs", 0, "fc", stream="oper")
        want = rf.select(idx, param="u", levelist=[50])
        rel = rf.path_for(f"{day:%Y%m%d}", f"{h:02d}", "ifs", 0, "fc", stream="oper") + ".grib2"
        vals += rf.fetch_rows(rel, want, tuple(range(108, 133)), each=lambda v, lat, lon: band_mean(v, lat))
    return float(np.mean(vals))


def newest_common(manifests: Path) -> str:
    """The newest 00/12Z cycle whose whole run every ensemble has on its page."""
    have = []
    for m in ENSEMBLES:
        man = json.loads((manifests / f"ensmaps_{m}.json").read_text())
        have.append({c["cycle"] for c in man["cycles"] if not c.get("through") and c["cycle"][8:] in ("00", "12")})
    common = set.intersection(*have)
    if not common:
        raise SystemExit("no 00/12Z run is on every ensemble's page")
    return max(common)


def window(target: str, manifests: Path, cycle: str | None = None) -> dict:
    ty, tm = map(int, target.split("-"))
    last = dt.date(ty, tm, G.SPLIT_DAY - 1)  # the 7th
    im, iy = (tm - 1, ty) if tm > 1 else (12, ty - 1)
    first = dt.date(iy, im, G.SPLIT_DAY)  # the 8th of the month before
    ndays = (last - first).days + 1
    # ERA5: the archive's block 1 of the month before (its mean over the days it holds)
    arc = G.load_year(iy)
    n_obs = int(arc["n"][im - 1, 1]) // 4  # four 6-hourly samples a day
    obs_mean = float(arc["u50v"][im - 1, 1]) if n_obs else 0.0
    obs_end = first + dt.timedelta(days=n_obs - 1)
    cyc = cycle or newest_common(manifests)
    init = dt.datetime.strptime(cyc, "%Y%m%d%H")
    # the IFS analyses from the day after ERA5's last to the run's start day
    gap = [obs_end + dt.timedelta(days=k) for k in range(1, (init.date() - obs_end).days + 1)]
    an = {d.isoformat(): analysis(d) for d in gap}
    fc_days = [init.date() + dt.timedelta(days=k) for k in range(1, (last - init.date()).days + 1)]
    steps = [int((dt.datetime.combine(d, dt.time(h)) - init).total_seconds() // 3600) for d in fc_days for h in HOURS]
    print(f"{target}: window {first} .. {last}; ERA5 {n_obs} days to {obs_end} (mean {obs_mean:.2f} m/s), IFS analyses "
          f"{len(gap)} days, the {cyc} runs {len(fc_days)} days (steps {steps[0]}-{steps[-1]} h)", flush=True)  # fmt: skip
    date, hh = cyc[:8], cyc[8:]
    get = {
        "ifs": lambda s: ecmwf_members("ifs", date, hh, s),
        "aifs": lambda s: ecmwf_members("aifs", date, hh, s),
        "gefs": lambda s: gefs_members(date, hh, s),
        "geps": lambda s: geps_members(date, hh, s),
    }
    fixed = n_obs * obs_mean + sum(an.values())
    out = {"target": target, "window": [first.isoformat(), last.isoformat()], "days": ndays, "cycle": cyc,
           "era5": {"days": n_obs, "through": obs_end.isoformat(), "mean": round(obs_mean, 3)},
           "analyses": {k: round(v, 3) for k, v in an.items()}, "ensembles": {}}  # fmt: skip
    for m in ENSEMBLES:
        per = [get[m](s) for s in steps]  # [{member: value}] per step
        members = sorted(set.intersection(*(set(p) for p in per)))
        daily = np.array([[np.mean([per[2 * k + j][mem] for j in range(len(HOURS))]) for k in range(len(fc_days))]
                          for mem in members])  # fmt: skip
        win = (fixed + daily.sum(1)) / ndays
        out["ensembles"][m] = {"members": len(members), "window": np.round(win, 3).tolist(),
                               "mean": round(float(win.mean()), 3), "daily_mean": np.round(daily.mean(0), 2).tolist()}  # fmt: skip
        print(f"  {m}: {len(members)} members, window mean {win.mean():.2f} m/s ({win.min():.2f} .. {win.max():.2f})",
              flush=True)  # fmt: skip
    out["mean"] = round(float(np.mean([e["mean"] for e in out["ensembles"].values()])), 3)  # a quarter each
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="YYYY-MM")
    ap.add_argument("--manifests", type=Path, default=ROOT / "assets" / "models")
    ap.add_argument("--cycle", help="YYYYMMDDHH (default: the newest 00/12Z run all four ensembles have)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = window(a.target, a.manifests, a.cycle)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"vortex window {out['window'][0]} .. {out['window'][1]}: {out['mean']:.2f} m/s (equal weight per ensemble)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
