#!/usr/bin/env python3
"""GEFSv12 reforecast -> lead-dependent model climatology for the GEFS extended page.

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

The climatology pools, for each 5-day centre, the starts within +-15 days of it on the common-year calendar
(noleap_doy: 1 March is day 60 in every year). The retired gefs-clim-v1 and v2 pooled on the raw day of year, which put
every leap-year start after 28 February one day later, so a leap-year start 16 days before a centre was pooled with
it and one 15 days after was left out (65 of the 73 centres, 89 leap-year starts). Release gefs-clim-v3
(gefs-reforecast.yml) holds both climatologies rebuilt on the common-year calendar: this file's weekly one and
gefs_daily.py's daily one, both fetched in one pass (gefs_daily.py month), plus the per-member weekly hindcast fields
of every start (gefs_hindw_MM.npz) that the tercile maps' skill pass scores (build_gefs_mapskill.py).

    python gefs_reforecast.py year 2019 --out out/              # one year of starts -> out/rf_YYYYMMDD.npz
    python gefs_reforecast.py combine 1 --inp out/ --out clim/  # the centres of January -> clim/gefs_clim_DDD.npz,
                                                                # January's starts -> clim/gefs_hindw_01.npz
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
CLIM_TAG = "gefs-clim-v3"  # the release every reader takes both climatologies from (clim_file)
CLIM_URL = f"https://github.com/scorvec/scorvec.github.io/releases/download/{CLIM_TAG}/{{name}}"
MEMBERS = ("c00", "p01", "p02", "p03")
DAYS = range(1, 36)
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
RES = 1.5
LAT = np.arange(90.0, -90.0 - 1e-6, -RES)  # 121
LON = np.arange(0.0, 360.0 - 1e-6, RES)  # 240
BAND = (LAT >= -15) & (LAT <= 15)

# tag -> (days 1-10 file, days 10-35 file, idx level text, kind)
FIELDS = {
    "t2m": ("tmp_2m", "tmp_2m", "2 m above ground", "inst"),
    "zg500": ("hgt_pres_abv700mb", "hgt_pres", "500 mb", "inst"),
    "mslp": ("pres_msl", "pres_msl", "mean sea level", "inst"),
    "u200": ("ugrd_pres_abv700mb", "ugrd_pres", "200 mb", "inst"),
    "u850": ("ugrd_pres", "ugrd_pres", "850 mb", "inst"),
    "u10": ("ugrd_pres_abv700mb", "ugrd_pres", "10 mb", "inst"),
    "pr": ("apcp_sfc", "apcp_sfc", "surface", "acc"),
    "olr": ("ulwrf_tatm", "ulwrf_tatm", "top of atmosphere", "ave"),
}
WEEKLY = ("t2m", "zg500", "mslp", "u200", "u850", "pr", "olr")
HINDW = ("t2m", "pr", "zg500", "mslp")  # the per-member weekly fields kept per start (the skill-masked map fields)
# storage as int16: q = round((value - offset) * factor). float16 overflowed on MSLP in Pa and kept 500 hPa height to
# only 4 m; these keep 0.02 K, 0.2 m, 0.02 hPa, 0.01 m/s, 0.01 mm/day and 0.01 W/m2 in the same two bytes
SCALE = {
    "t2m": (273.15, 50.0),
    "zg500": (5500.0, 5.0),
    "mslp": (100000.0, 0.5),
    "u200": (0.0, 100.0),
    "u850": (0.0, 100.0),
    "pr": (0.0, 100.0),
    "olr": (200.0, 100.0),
}


def pack(t, a):
    off, fac = SCALE[t]
    return np.clip(np.round((a - off) * fac), -32767, 32767).astype("int16")


def unpack(t, q):
    off, fac = SCALE[t]
    return q.astype("float64") / fac + off


BANDF = ("olr", "u200", "u850")


def noleap_doy(d):
    """The day-of-year key of every GEFS climatology and reference normal: the day of the 365-day common-year calendar,
    float - 1 Jan = 1, 28 Feb = 59, 1 March = 60, 31 Dec = 365 in every year, and 29 Feb = 59.5, midway between 28 Feb
    and 1 March (raw day of year puts 1 March at 61 in leap years). The 5-day centres 1, 6, ..., 361 wrap to 1 after
    361 on this calendar; day-of-year axes 1..366 hold keys 1..365 and repeat 1 January in the 366th slot. A date, an
    array of dates or datetime64 values -> float or float array."""
    a = np.asarray(d, dtype="datetime64[D]")
    y = a.astype("datetime64[Y]")
    doy = (a - y.astype("datetime64[D]")).astype(int) + 1  # raw, 1-based
    yr = y.astype(int) + 1970
    leap = (yr % 4 == 0) & ((yr % 100 != 0) | (yr % 400 == 0))
    x = np.where(leap & (doy == 60), 59.5, np.where(leap & (doy > 60), doy - 1, doy)).astype("float64")
    return float(x) if x.ndim == 0 else x


def at_key(arr, x):
    """arr[k - 1, ...] (leading axis = day-of-year key 1..365, optionally a 366th wrap slot) at keys x from noleap_doy;
    29 Feb (x = 59.5) is the mean of the 28 Feb and 1 March slots."""
    a = np.asarray(arr)
    x = np.asarray(x, dtype="float64")
    lo = np.floor(x).astype(int) - 1
    half = x != np.floor(x)
    if x.ndim == 0:
        return 0.5 * (a[lo] + a[lo + 1]) if half else a[lo]
    out = a[lo]  # fancy indexing: a copy
    if half.any():
        out = out.astype(np.result_type(out.dtype, np.float32), copy=False)
        out[half] = 0.5 * (out[half] + a[lo[half] + 1])
    return out


def get(url: str, rng: tuple[int, int] | None = None, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            h = {"User-Agent": "scorvec-gefs"}
            if rng:
                h["Range"] = f"bytes={rng[0]}-{rng[1] if rng[1] >= 0 else ''}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


_PRUNED: set[Path] = set()


def clim_file(name: str, cache: Path) -> Path:
    """A file of the climatology release CLIM_TAG through the cache (cache/CLIM_TAG/name), downloaded when missing.
    The first call per cache deletes the climatology files of every other release there - the flat gefs_clim_DDD.npz
    and gefs_clim2_DDD.npz of v1/v2 and other gefs-clim-* directories - so a release bump can never read a stale file
    of the same name, and the run cache does not carry the old ones."""
    cache = Path(cache)
    if cache not in _PRUNED:
        _PRUNED.add(cache)
        for p in [*cache.glob("gefs_clim_[0-9][0-9][0-9].npz"), *cache.glob("gefs_clim2_[0-9][0-9][0-9].npz")]:
            p.unlink()
        for d in cache.glob("gefs-clim-*"):
            if d.is_dir() and d.name != CLIM_TAG:
                for p in d.iterdir():
                    p.unlink()
                d.rmdir()
    p = cache / CLIM_TAG / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        tmp.write_bytes(get(CLIM_URL.format(name=name)))
        tmp.replace(p)
    return p


def index(url: str) -> list[tuple[int, int, str, str, str]]:
    """(start, end, level, step text, variable) per message from a .idx. The live files hold every variable in one
    file, so a match must use the variable too: "10 mb" alone finds the geopotential height first."""
    lines = [l.split(":") for l in get(url + ".idx").decode().splitlines() if l.strip()]
    out = []
    for i, p in enumerate(lines):
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else -1
        out.append((int(p[1]), end, p[4], p[5], p[3]))
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
    if j0 < 0:  # south-to-north: flip to the 90 -> -90 convention
        v = v[::-1]
    return v


def coarsen(v):
    """Box mean onto the 1.5 deg grid (both native grids include the poles and 0E, and nest exactly)."""
    from scipy.ndimage import uniform_filter1d

    nj = v.shape[0]
    res = 180.0 / (nj - 1)
    n = round(RES / res)
    a = uniform_filter1d(v, n, axis=0, mode="nearest")
    a = uniform_filter1d(a, n, axis=1, mode="wrap")
    return a[::n][: len(LAT), ::n][:, : len(LON)]


def zonal60(v):
    nj = v.shape[0]
    j = round((90.0 - 60.0) / (180.0 / (nj - 1)))
    return float(v[j].mean())


def wanted(tag: str):
    """(day, which file, step text) for every message the tag needs."""
    kind = FIELDS[tag][3]
    out = []
    for d in DAYS:
        if kind == "inst":
            out.extend((d, 1 if h <= 240 else 2, f"{h} hour fcst") for h in (24 * d - 12, 24 * d))
        else:
            out.extend(
                (d, 1 if h + 6 <= 240 else 2, f"{h}-{h + 6} hour {kind} fcst") for h in range(24 * d - 24, 24 * d, 6)
            )
    return out


class Weekly:
    """One member of one start, message by message: the daily 1.5 deg box mean of every weekly field, the 60N 10 hPa
    zonal wind, and from those (finish) the weekly means and the 15S-15N band of olr, u200 and u850."""

    def __init__(self):
        self.daily = {t: np.zeros((35, len(LAT), len(LON))) for t in FIELDS if t != "u10"}
        self.cnt = {t: np.zeros(35) for t in self.daily}
        self.u60 = np.zeros(35)
        self.u60n = np.zeros(35)

    def add(self, tag: str, day: int, v):
        """One decoded native field (90 -> -90 rows) of `tag` sampled on lead day `day` (1..35)."""
        if tag == "u10":
            self.u60[day - 1] += zonal60(v)
            self.u60n[day - 1] += 1
        else:
            self.daily[tag][day - 1] += coarsen(v)
            self.cnt[tag][day - 1] += 1

    def finish(self):
        """-> weekly {tag: (5, lat, lon) int16}, u60 (35,) float32, band (3, 35, lon) int16. Precipitation's four 6-h
        accumulations stay summed (mm/day); every other field is the mean of its samples."""
        daily = {
            t: d if FIELDS[t][3] == "acc" else d / np.maximum(self.cnt[t], 1)[:, None, None]
            for t, d in self.daily.items()
        }
        weekly = {t: pack(t, np.stack([daily[t][a - 1 : b].mean(0) for a, b in WEEKS])) for t in WEEKLY}
        band = np.stack([pack(t, daily[t][:, BAND, :].mean(1)) for t in BANDF])
        return weekly, (self.u60 / np.maximum(self.u60n, 1)).astype("float32"), band


def one_member(start: str, mem: str):
    """One member's Weekly reduction of one start, or None if its files are incomplete."""
    base = f"{S3}/{start[:4]}/{start}00/{mem}"
    idx_cache, jobs = {}, []
    for tag, (f1, f2, lev, _kind) in FIELDS.items():
        for d, which, stext in wanted(tag):
            f = f"{base}/Days:{'1-10' if which == 1 else '10-35'}/{f1 if which == 1 else f2}_{start}00_{mem}.grib2"
            if f not in idx_cache:
                try:
                    idx_cache[f] = index(f)
                except Exception:  # noqa: BLE001
                    return None
            hit = [e for e in idx_cache[f] if e[2] == lev and e[3] == stext]
            if not hit:
                return None
            jobs.append((tag, d, f, hit[0][0], hit[0][1]))
    acc = Weekly()
    with ThreadPoolExecutor(24) as ex:
        for (tag, d, *_), b in zip(jobs, ex.map(lambda j: get(j[2], (j[3], j[4])), jobs)):
            acc.add(tag, d, decode(b))
    return acc.finish()


def save_weekly(dest: Path, res: dict) -> None:
    """rf_<start>.npz from {member: Weekly.finish()}: members, w_<tag> (m, 5, lat, lon) int16, u60 (m, 35), band (m, 3,
    35, lon) int16."""
    ms = list(res)
    np.savez_compressed(
        dest,
        members=np.array(ms),
        **{f"w_{t}": np.stack([res[m][0][t] for m in ms]) for t in WEEKLY},
        u60=np.stack([res[m][1] for m in ms]),
        band=np.stack([res[m][2] for m in ms]),
    )


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
        except Exception as e:  # noqa: BLE001
            r = None
            print(f"  {start} {m}: {str(e)[:120]}", flush=True)
        if r is not None:
            res[m] = r
    if not res:
        return f"{start}: no complete member"
    save_weekly(dest, res)
    return f"{start}: {len(res)} members in {time.time() - t0:.0f} s"


def cmd_year(a) -> int:
    Path(a.out).mkdir(parents=True, exist_ok=True)
    starts = list(wednesdays(a.year))
    with ProcessPoolExecutor(a.procs) as ex:
        for msg in ex.map(run_start, [(s, a.out) for s in starts]):
            print(msg, flush=True)
    return 0


def centres_in(month: int):
    """The 5-day centres (1, 6, ..., 361) whose common-year day falls in `month`."""
    return [c for c in range(1, 366, 5) if (dt.date(2001, 1, 1) + dt.timedelta(days=c - 1)).month == month]


def pool(doy, c: int, half: int = 15):
    """Mask of the day-of-year keys `doy` (noleap_doy) within `half` days of centre c, the year wrapped at 365."""
    dd = np.abs(np.asarray(doy, dtype="float64") - c)
    return np.minimum(dd, 365 - dd) <= half


def cmd_combine(a) -> int:
    """The weekly climatology of the 5-day centres in a.month from every start within +-15 days of each (any year, all
    members), and the per-member weekly hindcast fields (HINDW) of the month's own starts -> gefs_hindw_MM.npz."""
    files = sorted(Path(a.inp).rglob("rf_*.npz"))
    doy = noleap_doy([dt.datetime.strptime(f.stem[3:], "%Y%m%d").date() for f in files])  # common-year day key
    print(f"{len(files)} starts, years {files[0].stem[3:7]}-{files[-1].stem[3:7]}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for c in centres_in(a.month):
        sel = [f for f, x in zip(files, pool(doy, c)) if x]
        acc = {t: np.zeros((5, len(LAT), len(LON))) for t in WEEKLY}
        u60 = np.zeros(35)
        band = np.zeros((3, 35, len(LON)))
        n = 0
        for f in sel:
            z = np.load(f)
            for t in WEEKLY:
                acc[t] += unpack(t, z[f"w_{t}"]).sum(0)
            u60 += z["u60"].sum(0)
            n += len(z["members"])
            band += np.stack([unpack(t, z["band"][:, i]) for i, t in enumerate(BANDF)], 1).sum(0)
        if n == 0:
            continue
        np.savez_compressed(
            Path(a.out) / f"gefs_clim_{c:03d}.npz",
            n=n,
            n_starts=len(sel),
            **{f"w_{t}": (acc[t] / n).astype("float32") for t in WEEKLY},
            u60=(u60 / n).astype("float32"),
            band=(band / n).astype("float32"),
            lat=LAT,
            lon=LON,
        )
        print(f"doy {c:3d}: {len(sel)} starts, {n} member-runs", flush=True)
    own = [f for f in files if int(f.stem[7:9]) == a.month]
    if own:
        keep = {}
        for f in own:
            with np.load(f) as z:
                st = f.stem[3:]
                keep[f"{st}_members"] = z["members"]
                for t in HINDW:
                    keep[f"{st}_w_{t}"] = z[f"w_{t}"]
        np.savez_compressed(Path(a.out) / f"gefs_hindw_{a.month:02d}.npz", **keep)
        print(f"weekly hindcast members: {len(own)} starts of month {a.month}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    y = sp.add_parser("year")
    y.add_argument("year", type=int)
    y.add_argument("--out", default="out")
    y.add_argument("--procs", type=int, default=4)
    c = sp.add_parser("combine")
    c.add_argument("month", type=int)
    c.add_argument("--inp", default="out")
    c.add_argument("--out", default="clim")
    a = ap.parse_args()
    return {"year": cmd_year, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
