#!/usr/bin/env python3
"""GEFSv12 reforecast -> lead-dependent model climatology for the GEFS extended page (2026-09-26; user: "a GEFS
extended page that mirrors the format of the GEPS page" - plan approved the same day).

Why a model climatology: GEFS drifts with lead, exactly as GEPS does, so anomalies must be taken against GEFS's own
reforecast at the same lead and start date, never against a reanalysis (the GEPS page's rule; see its forecast.py).

Source: NOAA's GEFSv12 reforecast on S3 (noaa-gefs-retrospective, anonymous), 2000-2019. The 35-day runs are the
Wednesday starts (control + 10 members); every file has a .idx, so each field and step is byte-ranged (the site rule
for NOAA models). Days 1-10 are 3-hourly at 0.25 deg, days 10-35 6-hourly at 0.5 deg; above 700 hPa the day 1-10
fields sit in separate *_abv700mb files.

Daily values, identical to what the live pipeline computes (gefs_live.py), so the two sides cancel exactly:
  instantaneous fields (t2m, z500, mslp, u200, u850, u10): the mean of the 12Z and 00Z samples ending the day
  precipitation: the sum of the four 6-hour accumulations (mm/day); OLR: the mean of the four 6-hour averages (W/m2); MSLP in Pa
Everything is box-averaged to 1.5 deg and reduced to weekly means (days 1-7 ... 29-35); u10 is kept daily as the 60N
zonal mean (the vortex panel), and OLR/u200/u850 daily as 15S-15N means by longitude (the MJO index).

    python gefs_reforecast.py year 2019 --out out/            # one year of starts -> out/rf_YYYYMMDD.npz
    python gefs_reforecast.py combine --inp out/ --out clim/  # every year -> clim/gefs_clim_DDD.npz per 5-day doy centre
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

S3 = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
MEMBERS = ("c00", "p01", "p02", "p03")
DAYS = range(1, 36)
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
RES = 1.5
LAT = np.arange(90.0, -90.0 - 1e-6, -RES)          # 121
LON = np.arange(0.0, 360.0 - 1e-6, RES)            # 240
BAND = (LAT >= -15) & (LAT <= 15)

# tag -> (days 1-10 file, days 10-35 file, idx level text, kind)
FIELDS = {
    "t2m":   ("tmp_2m",             "tmp_2m",     "2 m above ground",  "inst"),
    "zg500": ("hgt_pres_abv700mb",  "hgt_pres",   "500 mb",            "inst"),
    "mslp":  ("pres_msl",           "pres_msl",   "mean sea level",    "inst"),
    "u200":  ("ugrd_pres_abv700mb", "ugrd_pres",  "200 mb",            "inst"),
    "u850":  ("ugrd_pres",          "ugrd_pres",  "850 mb",            "inst"),
    "u10":   ("ugrd_pres_abv700mb", "ugrd_pres",  "10 mb",             "inst"),
    "pr":    ("apcp_sfc",           "apcp_sfc",   "surface",           "acc"),
    "olr":   ("ulwrf_tatm",         "ulwrf_tatm", "top of atmosphere", "ave"),
}
WEEKLY = ("t2m", "zg500", "mslp", "u200", "u850", "pr", "olr")
# storage as int16: q = round((value - offset) * factor). float16 overflowed on MSLP in Pa and kept 500 hPa height to
# only 4 m; these keep 0.02 K, 0.2 m, 0.02 hPa, 0.01 m/s, 0.01 mm/day and 0.01 W/m2 in the same two bytes
SCALE = {"t2m": (273.15, 50.0), "zg500": (5500.0, 5.0), "mslp": (100000.0, 0.5), "u200": (0.0, 100.0),
         "u850": (0.0, 100.0), "pr": (0.0, 100.0), "olr": (200.0, 100.0)}


def pack(t, a):
    off, fac = SCALE[t]
    return np.clip(np.round((a - off) * fac), -32767, 32767).astype("int16")


def unpack(t, q):
    off, fac = SCALE[t]
    return q.astype("float64") / fac + off
BANDF = ("olr", "u200", "u850")


def get(url: str, rng: tuple[int, int] | None = None, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            h = {"User-Agent": "scorvec-gefs"}
            if rng:
                h["Range"] = f"bytes={rng[0]}-{rng[1] if rng[1] >= 0 else ''}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                return r.read()
        except Exception:                                          # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def index(url: str) -> list[tuple[int, int, str, str]]:
    """(start, end, level, step text) per message from a .idx."""
    lines = [l.split(":") for l in get(url + ".idx").decode().splitlines() if l.strip()]
    out = []
    for i, p in enumerate(lines):
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else -1
        out.append((int(p[1]), end, p[4], p[5]))
    return out


def decode(buf: bytes):
    import eccodes
    g = eccodes.codes_new_from_message(buf)
    try:
        nj, ni = eccodes.codes_get(g, "Nj"), eccodes.codes_get(g, "Ni")
        v = eccodes.codes_get_values(g).reshape(nj, ni)
        j0 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees")
    finally:
        eccodes.codes_release(g)
    if j0 < 0:                                                   # south-to-north: flip to the 90 -> -90 convention
        v = v[::-1]
    return v


def coarsen(v):
    """Box mean onto the 1.5 deg grid (both native grids include the poles and 0E, and nest exactly)."""
    from scipy.ndimage import uniform_filter1d
    nj = v.shape[0]
    res = 180.0 / (nj - 1)
    n = int(round(RES / res))
    a = uniform_filter1d(v, n, axis=0, mode="nearest")
    a = uniform_filter1d(a, n, axis=1, mode="wrap")
    return a[::n][: len(LAT), ::n][:, : len(LON)]


def zonal60(v):
    nj = v.shape[0]
    j = int(round((90.0 - 60.0) / (180.0 / (nj - 1))))
    return float(v[j].mean())


def wanted(tag: str):
    """(day, which file, step text) for every message the tag needs."""
    kind = FIELDS[tag][3]
    out = []
    for d in DAYS:
        if kind == "inst":
            for h in (24 * d - 12, 24 * d):
                out.append((d, 1 if h <= 240 else 2, f"{h} hour fcst"))
        else:
            for h in range(24 * d - 24, 24 * d, 6):
                out.append((d, 1 if h + 6 <= 240 else 2, f"{h}-{h + 6} hour {kind} fcst"))
    return out


def one_member(start: str, mem: str):
    """Everything one member of one start contributes, or None if its files are incomplete."""
    y = start[:4]
    base = f"{S3}/{y}/{start}00/{mem}"
    idx_cache, jobs = {}, []
    for tag, (f1, f2, lev, kind) in FIELDS.items():
        for d, which, stext in wanted(tag):
            f = f"{base}/Days:{'1-10' if which == 1 else '10-35'}/{f1 if which == 1 else f2}_{start}00_{mem}.grib2"
            if f not in idx_cache:
                try:
                    idx_cache[f] = index(f)
                except Exception:                                  # noqa: BLE001
                    return None
            hit = [e for e in idx_cache[f] if e[2] == lev and e[3] == stext]
            if not hit:
                return None
            jobs.append((tag, d, f, hit[0][0], hit[0][1]))
    with ThreadPoolExecutor(24) as ex:
        bufs = list(ex.map(lambda j: get(j[2], (j[3], j[4])), jobs))
    daily = {t: np.zeros((35, len(LAT), len(LON)), "float64") for t in FIELDS if t != "u10"}
    cnt = {t: np.zeros(35) for t in daily}
    u60 = np.zeros(35); u60n = np.zeros(35)
    for (tag, d, _, _, _), b in zip(jobs, bufs):
        v = decode(b)
        if tag == "u10":
            u60[d - 1] += zonal60(v); u60n[d - 1] += 1
            continue
        daily[tag][d - 1] += coarsen(v); cnt[tag][d - 1] += 1
    for t in daily:
        if FIELDS[t][3] == "acc":
            daily[t] = daily[t]                                      # four 6-h accumulations summed: mm/day
        else:
            daily[t] /= np.maximum(cnt[t], 1)[:, None, None]
    weekly = {t: pack(t, np.stack([daily[t][a - 1:b].mean(0) for a, b in WEEKS])) for t in WEEKLY}
    band = np.stack([pack(t, daily[t][:, BAND, :].mean(1)) for t in BANDF])           # (3, 35, 240) int16
    return weekly, (u60 / np.maximum(u60n, 1)).astype("float32"), band


def wednesdays(year: int):
    d = dt.date(year, 1, 1)
    while d.weekday() != 2:
        d += dt.timedelta(days=1)
    while d.year == year:
        yield d.strftime("%Y%m%d")
        d += dt.timedelta(days=7)


def run_start(args):
    start, out = args
    dest = Path(out) / f"rf_{start}.npz"
    if dest.exists():
        return f"{start}: exists"
    t0 = time.time()
    res = {}
    for m in MEMBERS:
        try:
            r = one_member(start, m)
        except Exception as e:                                     # noqa: BLE001
            r = None; print(f"  {start} {m}: {str(e)[:120]}", flush=True)
        if r is not None:
            res[m] = r
    if not res:
        return f"{start}: no complete member"
    ms = list(res)
    np.savez_compressed(dest, members=np.array(ms),
                        **{f"w_{t}": np.stack([res[m][0][t] for m in ms]) for t in WEEKLY},
                        u60=np.stack([res[m][1] for m in ms]), band=np.stack([res[m][2] for m in ms]))
    return f"{start}: {len(ms)} members in {time.time() - t0:.0f} s"


def cmd_year(a) -> int:
    Path(a.out).mkdir(parents=True, exist_ok=True)
    starts = list(wednesdays(a.year))
    with ProcessPoolExecutor(a.procs) as ex:
        for msg in ex.map(run_start, [(s, a.out) for s in starts]):
            print(msg, flush=True)
    return 0


def cmd_combine(a) -> int:
    """Climatology per 5-day day-of-year centre from the starts within +-15 days (any year), all members."""
    files = sorted(Path(a.inp).rglob("rf_*.npz"))
    doy = np.array([dt.datetime.strptime(f.stem[3:], "%Y%m%d").timetuple().tm_yday for f in files])
    print(f"{len(files)} starts, years {files[0].stem[3:7]}-{files[-1].stem[3:7]}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for c in range(1, 366, 5):
        dd = np.abs(doy - c); dd = np.minimum(dd, 365 - dd)
        sel = [f for f, x in zip(files, dd) if x <= 15]
        acc = {t: np.zeros((5, len(LAT), len(LON))) for t in WEEKLY}
        u60 = np.zeros(35); band = np.zeros((3, 35, len(LON))); n = 0
        for f in sel:
            z = np.load(f)
            for t in WEEKLY:
                acc[t] += unpack(t, z[f"w_{t}"]).sum(0)
            u60 += z["u60"].sum(0); n += len(z["members"])
            band += np.stack([unpack(t, z["band"][:, i]) for i, t in enumerate(BANDF)], 1).sum(0)
        if n == 0:
            continue
        np.savez_compressed(Path(a.out) / f"gefs_clim_{c:03d}.npz", n=n, n_starts=len(sel),
                            **{f"w_{t}": (acc[t] / n).astype("float32") for t in WEEKLY},
                            u60=(u60 / n).astype("float32"), band=(band / n).astype("float32"),
                            lat=LAT, lon=LON)
        print(f"doy {c:3d}: {len(sel)} starts, {n} member-runs", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    y = sp.add_parser("year"); y.add_argument("year", type=int); y.add_argument("--out", default="out"); y.add_argument("--procs", type=int, default=4)
    c = sp.add_parser("combine"); c.add_argument("--inp", default="out"); c.add_argument("--out", default="clim")
    a = ap.parse_args()
    return {"year": cmd_year, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
