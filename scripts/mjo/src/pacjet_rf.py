#!/usr/bin/env python3
"""GEFS weeks 3-5 of the North Pacific jet and of the Himalayan mountain torque: the drift of GEFS's 35-day runs
against ERA5 there, measured on the GEFSv12 reforecast, and the skill that decides which weeks are shown.

The 15-day jet forecasts (pacjet.py) read the ECMWF ensembles. GEFS runs to day 35 at 00 UTC, and NOAA's reforecast of
the same model (GEFSv12, noaa-gefs-retrospective, anonymous) has a 35-day start every Wednesday of 2000-2019:

  REFORECAST  every Wednesday start, gefs_reforecast.MEMBERS (control + 3), from the Days:10-35 files (0.5 deg,
              byte-ranged through their .idx): the 00 UTC 250 hPa zonal wind of lead days 15-35 as WEEKS means on the
              jet's 1.5 deg grid (pacjet_core.to_eof of the 0.5 deg points), and the surface pressure at 00 and 12 UTC
              as WEEKS means of the Himalayan mountain torque (torque: linear in the pressure, so the week's mean
              pressure gives the week's torque).
  TORQUE      the sum over the box (BOX: 70-105E, 25-45N, torque_map_anim's Himalaya/Tibet) of h dp_s/dlambda times the
              cell area, with the model's own orography h and its surface pressure both averaged over 2.5 deg blocks,
              in Hadley (1e18 N m): torque_history_era5's mtn25 of ERA5, the resolution at which the two can be
              compared (the absolute torque is not converged in resolution; its departures are).
  TRUTH       ERA5's 250 hPa zonal wind at 00 UTC on the same valid days (WeatherBench2, 1.5 deg: the jet's grid) and
              the daily mtn25 Himalayan torque of torque_daily_era5.csv.gz.
  DRIFT       a week's error, the members' mean less ERA5, regressed per point on the start's common-year day (mean
              and annual harmonic) and shrunk by its signal-to-noise (hadley_slp_ref.fit_drift and eval_drift). The
              torque's drift takes with it the offset of the model's orography from ERA5's and of two samples a day
              from four.
  SKILL       the members less the drift (refitted without the start's July-June year): the week's mean wind anomaly
              from pacjet_ref.nc's climatology projected on the jet-phase patterns of the month of the week's middle
              day (PC1 extension/retraction, PC2 poleward/equatorward shift), and the torque, in the weeks whose month
              is in the jet's season; the mean-squared-error skill score of the members' mean against ERA5's
              climatology, its error taken down by (1/len(MEMBERS) - 1/LIVE) times the members' sample variance (the
              expected error of the live 31-member mean), an interval and one-sided p-value from a bootstrap over whole
              July-June years of starts, Benjamini-Hochberg at hadley_slp_ref.FDR_Q over every target and week; a
              target shows a week when it survives and reaches hadley_slp_ref.MIN_SKILL, from week 3 up to the first
              week that fails.

    python src/pacjet_rf.py archive --year 2005 [--limit 2] --out rf_2005.npz   # one year of starts
    python src/pacjet_rf.py truth --year 2005 --out era5_2005.npz       # ERA5's wind on that year's valid days
    python src/pacjet_rf.py verify parts/*.npz                           # -> DRIFT (release) and SKILL (committed)
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gefs"))
import build_hadley_clim as B
import gefs_reforecast as RF
import hadley_slp_ref as R
import pacjet_core as PC
from doycal import noleap_doy

REF_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
REF = REF_DIR / "pacjet_ref.nc"
TORQUE = REF_DIR / "torque_daily_era5.csv.gz"
DRIFT = REF_DIR / "pacjet_rf_drift.npz"  # release pacjet-ref-v1
SKILL = REF_DIR / "pacjet_rf_skill.json"  # committed: the weeks the page shows
WEEKS = tuple(RF.WEEKS[2:])  # lead days of weeks 3-5
DAYS = range(WEEKS[0][0], WEEKS[-1][1] + 1)
LAST_H = 24 * WEEKS[-1][1]  # the runs end here: the last day has no 12 UTC
LIVE = 31  # GEFS's live members: the control and 30
TARGETS = ("pc1", "pc2", "torque")
A, HU, DEG = 6.371e6, 1e18, np.pi / 180.0  # torque_history_era5's
BOX = (70.0, 105.0, 25.0, 45.0)  # lon0, lon1, lat0, lat1: torque_map_anim.RANGES["Himalaya/Tibet"]
BLOCK = 5  # 0.5 deg points per 2.5 deg block
CLIM_YEARS = (1991, 2020)
# the 0.5 deg global grid (90N..90S, 0..359.5E) of the reforecast's days 10-35 and of the live pgrb2a files
_ROWS = np.round((90.0 - PC.WORK_LAT) / 0.5).astype(int)
_COLS = np.round(PC.WORK_LON / 0.5).astype(int)


def jet_grid(f: np.ndarray) -> np.ndarray:
    """A 0.5 deg global field (361, 720), north to south -> the jet's 1.5 deg grid (47, 133)."""
    return PC.to_eof(f[np.ix_(_ROWS, _COLS)])


def blocks(f: np.ndarray) -> np.ndarray:
    """A 0.5 deg global field -> its 2.5 deg block means over BOX and one block either side in longitude,
    (lat south to north, lon)."""
    lon0, lon1, lat0, lat1 = BOX
    rows = np.round((90.0 - np.arange(lat0, lat1, 0.5)) / 0.5).astype(int)
    cols = np.round(np.arange(lon0 - 0.5 * BLOCK, lon1 + 0.5 * BLOCK, 0.5) / 0.5).astype(int)
    sub = f[np.ix_(rows, cols)]
    return sub.reshape(rows.size // BLOCK, BLOCK, cols.size // BLOCK, BLOCK).mean((1, 3))


def torque(h: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """The box's mountain torque (Hadley) from block-mean orography h (m) and surface pressure ps (..., lat, lon; Pa)
    of `blocks`: h dp_s/dlambda (centred) times the cell area, summed."""
    dl = 0.5 * BLOCK * DEG
    lat = np.arange(BOX[2], BOX[3], 0.5 * BLOCK) + 0.25 * (BLOCK - 1)
    area = A**2 * np.cos(lat * DEG)[:, None] * dl * dl
    dpd = (ps[..., 2:] - ps[..., :-2]) / (2 * dl)
    return (h[:, 1:-1] * dpd * area).sum((-2, -1)) / HU


def ps_hours(a: int, z: int) -> list[int]:
    """The 00 and 12 UTC forecast hours of the lead days a..z that the run holds."""
    return [h for d in range(a, z + 1) for h in (24 * d, 24 * d + 12) if h <= LAST_H]


# ── the reforecast ──────────────────────────────────────────────────────────────────────────────────────────────────
def _url(start: str, mem: str, var: str) -> str:
    return f"{RF.S3}/{start[:4]}/{start}00/{mem}/Days:10-35/{var}_{start}00_{mem}.grib2"


def orography(start: str) -> np.ndarray:
    """The reforecast model's orography (m) as `blocks`, from the control's file of one start."""
    url = _url(start, "c00", "hgt_sfc")
    a, z = RF.index(url)[0][:2]
    return blocks(RF.decode(RF.get(url, (a, z))))


def reduce(jobs: list, fetch) -> tuple[np.ndarray, np.ndarray]:
    """(u (week, 47, 133), ps (week, lat, lon) of `blocks`): the WEEKS means of the fetched messages. jobs:
    ("u" | "ps", forecast hour, what `fetch` takes)."""
    u = np.zeros((len(WEEKS), PC.EOF_LAT.size, PC.EOF_LON.size))
    ps, n = None, np.zeros((2, len(WEEKS)))
    with ThreadPoolExecutor(24) as ex:
        for (kind, hour, _), buf in zip(jobs, ex.map(lambda j: fetch(j[2]), jobs), strict=True):
            f = RF.decode(buf)
            x = jet_grid(f) if kind == "u" else blocks(f)
            if ps is None and kind == "ps":
                ps = np.zeros((len(WEEKS), *x.shape))
            for w, (a, z) in enumerate(WEEKS):
                if 24 * a <= hour <= 24 * z + 12:
                    (u if kind == "u" else ps)[w] += x
                    n[int(kind == "ps"), w] += 1
    with np.errstate(invalid="ignore", divide="ignore"):  # a week not asked for: NaN
        return u / n[0][:, None, None], ps / n[1][:, None, None]


def one_member(start: str, mem: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(u (week, 47, 133), ps (week, lat, lon)) of one member of one start, or None when a message is missing."""
    jobs = []
    for kind, var, level, hours in (
        ("u", "ugrd_pres", f"{PC.LEVEL} mb", [24 * d for d in DAYS]),
        ("ps", "pres_sfc", "surface", ps_hours(DAYS[0], DAYS[-1])),
    ):
        url = _url(start, mem, var)
        idx = {(e[2], e[3]): (e[0], e[1]) for e in RF.index(url)}
        for h in hours:
            hit = idx.get((level, f"{h} hour fcst"))
            if hit is None:
                print(f"  {start} {mem}: no {var} at {h} h", flush=True)
                return None
            jobs.append((kind, h, (url, hit)))
    u, ps = reduce(jobs, lambda j: RF.get(*j))
    return u.astype(np.float32), ps.astype(np.float32)


def archive(year: int, limit: int | None = None) -> dict:
    """Every Wednesday start of the year (the first `limit` of them, for a trial) with all MEMBERS complete: u
    (start, member, week, 47, 133), the torque (start, member, week) and the orography blocks."""
    keep, us, tq, h = [], [], [], None
    for start in list(RF.wednesdays(year))[:limit]:
        t0 = time.time()
        got = [one_member(start, m) for m in RF.MEMBERS]
        if any(g is None for g in got):
            print(f"  {start}: incomplete, skipped", flush=True)
            continue
        if h is None:
            h = orography(start)
        keep.append(pd.Timestamp(start))
        us.append(np.stack([g[0] for g in got]))
        tq.append(np.stack([torque(h, g[1]) for g in got]))
        print(
            f"  {start}: {len(got)} members in {time.time() - t0:.0f} s, torque {tq[-1].mean(0).round(1)}", flush=True
        )
    return {
        "start": np.array(keep, "datetime64[D]"),
        "u": np.array(us),
        "torque": np.array(tq, np.float32),
        "h": np.zeros(0) if h is None else h,
    }


def truth(year: int) -> dict:
    """ERA5's 00 UTC 250 hPa zonal wind on the jet's grid on the valid days of the year's starts (weeks 3-5)."""
    starts = [pd.Timestamp(s) for s in RF.wednesdays(year)]
    t0, t1 = starts[0] + pd.Timedelta(days=DAYS[0]), starts[-1] + pd.Timedelta(days=DAYS[-1])
    ds = xr.open_zarr(B.WB2, storage_options={"token": "anon"})
    u = ds["u_component_of_wind"].sel(time=slice(t0, t1), level=PC.LEVEL)
    u = u.isel(time=np.flatnonzero(pd.DatetimeIndex(u.time.values).hour == 0))
    u = u.sel(latitude=slice(PC.EOF_LAT[0] - 0.1, PC.EOF_LAT[-1] + 0.1)).sortby("latitude")
    u = u.sel(longitude=slice(PC.EOF_LON[0] - 0.1, PC.EOF_LON[-1] + 0.1)).transpose("time", "latitude", "longitude")
    assert np.allclose(u.latitude.values, PC.EOF_LAT, atol=1e-3), "WeatherBench2 rows are not the jet's grid"
    assert np.allclose(u.longitude.values, PC.EOF_LON, atol=1e-3), "WeatherBench2 columns are not the jet's grid"
    u = u.load()
    return {"day": pd.DatetimeIndex(u.time.values).normalize().values.astype("datetime64[D]"), "u": u.values}


# ── verification ────────────────────────────────────────────────────────────────────────────────────────────────────
def load(paths) -> tuple[dict, dict]:
    """The reforecast {start, u, torque, h} and ERA5 {day, u}, merged from the year parts."""
    rf, tr = {"start": [], "u": [], "torque": []}, {"day": [], "u": []}
    h = None
    for p in sorted(paths):
        with np.load(p) as f:
            dest = rf if "start" in f.files else tr
            if "h" in f.files and f["h"].ndim == 2:
                h = f["h"]
            for k in dest:
                if f[k].size:
                    dest[k].append(f[k])
    rf = {k: np.concatenate(v) for k, v in rf.items()}
    tr = {k: np.concatenate(v) for k, v in tr.items()}
    return rf | {"h": h}, tr


def week_days(s: pd.Timestamp, w: int) -> pd.DatetimeIndex:
    a, z = WEEKS[w]
    return pd.date_range(s + pd.Timedelta(days=a), s + pd.Timedelta(days=z))


def week_month(s: pd.Timestamp, w: int) -> int:
    """The jet-phase basis of a week: the month of its middle day."""
    return int(week_days(s, w)[3].month)


def torque_obs() -> tuple[pd.Series, np.ndarray]:
    """ERA5's daily Himalayan torque (mtn25) and the harmonics of its CLIM_YEARS climatology (pacjet_core.harm_basis)."""
    t = pd.read_csv(TORQUE, usecols=["date", "mtn25_HIM"], index_col="date", parse_dates=True)["mtn25_HIM"].dropna()
    c = t[(t.index.year >= CLIM_YEARS[0]) & (t.index.year <= CLIM_YEARS[1])]
    coef = np.linalg.lstsq(PC.harm_basis(noleap_doy(c.index)), c.values, rcond=None)[0]
    return t, coef


def weekly_truth(tr: dict, tq: pd.Series, starts: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """ERA5's (start, week, 47, 133) wind and (start, week) torque on each start's valid weeks, NaN where a day is
    missing."""
    pos = {pd.Timestamp(d): i for i, d in enumerate(tr["day"])}
    u = np.full((len(starts), len(WEEKS), *tr["u"].shape[1:]), np.nan, np.float32)
    t = np.full((len(starts), len(WEEKS)), np.nan)
    for i, s in enumerate(starts):
        for w in range(len(WEEKS)):
            days = week_days(s, w)
            rows = [pos.get(d) for d in days]
            if all(r is not None for r in rows):
                u[i, w] = tr["u"][rows].mean(0)
            if days.isin(tq.index).all():
                t[i, w] = tq.loc[days].mean()
    return u, t


def season_year(starts: pd.DatetimeIndex) -> np.ndarray:
    """The July-June year of each start (named by its June): a northern winter's starts fall in one year."""
    return starts.year.values + (starts.month.values >= 7)


def _fits(err: np.ndarray, starts: pd.DatetimeIndex):
    """Per week: the drift fitted on every start and each start's drift refitted without its July-June year."""
    year = season_year(starts)
    cv = np.full_like(err, np.nan)
    fits, low = [], np.ones(err.shape[-1], bool)
    for w in range(len(WEEKS)):
        ok = np.isfinite(err[:, w]).all(-1)
        fits.append(R.fit_drift(err[ok, w], starts[ok], low))
        for y in np.unique(year[ok]):
            a, b = ok & (year != y), ok & (year == y)
            cv[b, w] = R.eval_drift(R.fit_drift(err[a, w], starts[a], low), R.doy_of(starts[b]))
    return fits, cv


def week_clim(ref: xr.Dataset, s: pd.Timestamp, w: int) -> np.ndarray:
    """ERA5's climatological 250 hPa zonal wind of a week on the jet's grid."""
    return PC.harm_eval(ref.u_coef.values, noleap_doy(week_days(s, w))).mean(0)


def verify(paths, seed: int = 1) -> dict:
    rf, tr = load(paths)
    starts = pd.DatetimeIndex(rf["start"])
    ref = xr.open_dataset(REF)
    tq_day, tq_coef = torque_obs()
    Tu, Tt = weekly_truth(tr, tq_day, starts)
    n_s, n_m = rf["u"].shape[:2]
    npt = PC.EOF_LAT.size * PC.EOF_LON.size
    fits_u, cv_u = _fits(rf["u"].mean(1).reshape(n_s, len(WEEKS), npt) - Tu.reshape(n_s, len(WEEKS), npt), starts)
    fits_t, cv_t = _fits((rf["torque"].mean(1) - Tt)[..., None], starts)
    print(
        f"  drift fitted on {np.isfinite(Tu[:, 0, 0, 0]).sum()} starts (wind), {np.isfinite(Tt[:, 0]).sum()} (torque)"
    )
    in_season = ref.in_season.values.astype(bool)
    S = {k: np.full((n_s, len(WEEKS), len(TARGETS)), np.nan) for k in ("fc", "raw", "ref")}
    pair = np.full((2, n_s, len(WEEKS), len(TARGETS)), np.nan)  # the corrected members' mean and ERA5, as anomalies
    fair = 1.0 / n_m - 1.0 / LIVE
    for i, s in enumerate(starts):
        for w in range(len(WEEKS)):
            month = week_month(s, w)
            if not in_season[month - 1]:
                continue
            proj = ref.proj.sel(month=month).values
            if np.isfinite(Tu[i, w]).all() and np.isfinite(cv_u[i, w]).all():
                clim = week_clim(ref, s, w)
                o = PC.project(Tu[i, w] - clim, proj)
                raw = PC.project(rf["u"][i, :, w] - clim, proj)
                mem = PC.project(rf["u"][i, :, w] - cv_u[i, w].reshape(clim.shape) - clim, proj)
                for kind, x, adj in (("fc", mem, fair), ("raw", raw, 0.0)):
                    S[kind][i, w, :2] = (x.mean(0) - o) ** 2 - adj * x.var(0, ddof=1)
                S["ref"][i, w, :2] = o**2
                pair[:, i, w, :2] = mem.mean(0), o
            if np.isfinite(Tt[i, w]) and np.isfinite(cv_t[i, w, 0]):
                clim = float(PC.harm_eval(tq_coef, noleap_doy(week_days(s, w))).mean())
                o = Tt[i, w] - clim
                for kind, x, adj in (
                    ("fc", rf["torque"][i, :, w] - cv_t[i, w, 0] - clim, fair),
                    ("raw", rf["torque"][i, :, w] - clim, 0.0),
                ):
                    S[kind][i, w, 2] = (x.mean() - o) ** 2 - adj * x.var(ddof=1)
                S["ref"][i, w, 2] = o**2
                pair[:, i, w, 2] = rf["torque"][i, :, w].mean() - cv_t[i, w, 0] - clim, o
    names, block = np.unique(season_year(starts), return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(x, minlength=names.size) for x in rng.integers(0, names.size, (R.BOOT, names.size))])
    res = {}
    for kind in ("fc", "raw"):
        tests = [
            R.skill_test(S[kind][..., t], S["ref"][..., t], block, names.size, counts) for t in range(len(TARGETS))
        ]
        res[kind] = [np.stack(x) for x in zip(*tests, strict=True)]  # msss, lo, hi, p: (target, week)
    passed = R.bh(res["fc"][3].ravel()).reshape(len(TARGETS), len(WEEKS))
    doc = {
        "members": n_m,
        "live_members": LIVE,
        "starts": n_s,
        "first": f"{starts[0]:%Y-%m-%d}",
        "last": f"{starts[-1]:%Y-%m-%d}",
        "weeks": [list(w) for w in WEEKS],
        "months": [int(m) + 1 for m in np.flatnonzero(in_season)],
        "score": "MSSS against ERA5's climatology; the members' mean error made fair for the live ensemble",
        "bootstrap": R.BOOT,
        "blocks": "July-June years of starts",
        "cross_validation": "drift refitted without the start's July-June year",
        "fdr_q": R.FDR_Q,
        "min_skill": R.MIN_SKILL,
        "targets": {},
    }
    for t, name in enumerate(TARGETS):
        ok = passed[t] & (res["fc"][0][t] >= R.MIN_SKILL)
        show = 0
        while show < len(WEEKS) and ok[show]:
            show += 1
        f, o = pair[0, ..., t], pair[1, ..., t]
        good = np.isfinite(f) & np.isfinite(o)
        doc["targets"][name] = {
            "show": show,  # weeks 3 .. 3 + show - 1
            "msss": _r3(res["fc"][0][t]),
            "ci": [_r3(res["fc"][1][t]), _r3(res["fc"][2][t])],
            "p": [round(float(x), 4) for x in res["fc"][3][t]],
            "msss_no_drift": _r3(res["raw"][0][t]),
            "correlation": _r3([np.corrcoef(f[good[:, w], w], o[good[:, w], w])[0, 1] for w in range(len(WEEKS))]),
            "obs_sd": _r3([o[good[:, w], w].std() for w in range(len(WEEKS))]),  # ERA5's weekly anomalies, in season
            "scored": [int(x) for x in good.sum(0)],
        }
    SKILL.write_text(json.dumps(doc, indent=1) + "\n")
    save = {"h": rf["h"], "torque_clim": tq_coef}
    for k, fl in (("u", fits_u), ("torque", fits_t)):
        save[f"{k}_coef"] = np.stack([f["coef"] for f in fl])
        save[f"{k}_cov"] = np.stack([f["cov"] for f in fl])
        save[f"{k}_s2"] = np.stack([f["s2"] for f in fl])
        save[f"{k}_kappa"] = np.array([f["kappa"] for f in fl])
    np.savez_compressed(DRIFT, **{k: v.astype(np.float32) for k, v in save.items()})
    for name, t in doc["targets"].items():
        row = [
            " ".join("  .  " if v is None else f"{v:+.2f}" for v in t[k])
            for k in ("msss", "msss_no_drift", "correlation")
        ]
        print(f"  {name:7s} weeks 3-5 {row[0]} | no drift {row[1]} | r {row[2]} | p {t['p']} | show {t['show']}")
    return doc


def _r3(x):
    return [round(float(v), 3) if np.isfinite(v) else None for v in np.atleast_1d(x)]


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("archive", "truth"):
        s = sub.add_parser(name)
        s.add_argument("--year", type=int, required=True)
        s.add_argument("--out", required=True)
        if name == "archive":
            s.add_argument("--limit", type=int)
    s = sub.add_parser("verify")
    s.add_argument("parts", nargs="+")
    a = ap.parse_args()
    if a.cmd == "archive":
        np.savez_compressed(a.out, **archive(a.year, a.limit))
    elif a.cmd == "truth":
        np.savez_compressed(a.out, **truth(a.year))
    else:
        verify(a.parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
