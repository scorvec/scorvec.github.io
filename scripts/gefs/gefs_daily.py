#!/usr/bin/env python3
"""GEFS daily fields - the shared arithmetic of the daily reforecast climatology and the live extended run.

The weekly climatology (gefs_reforecast.py) holds WEEKLY means only. The GEPS page's daily animation, index plumes,
regime assignments and vortex diagnostics are all daily and lead-dependent, so this one keeps, per reforecast start:

  d_<tag>      the member-mean DAILY field at 1.5 deg for every map field and the 10/100 hPa vortex fields
  m_<tag>      every member's daily 500 hPa height, sea-level pressure and 100 hPa height at 2.5 deg (the NCEP R1
               grid the teleconnection and regime patterns live on) - for the hindcast calibration
  s            every member's daily vortex scalars: u(60) and the 65-90 polar-cap T and Z at 10 and 100 hPa, NH/SH

and the combine step turns the starts within +-15 days of each 5-day day-of-year centre into the lead-dependent daily
climatology. The pools are on the common-year calendar (gefs_reforecast.noleap_doy); release gefs-clim-v2 pooled on
the raw day of year and is replaced by gefs-clim-v3 (see gefs_reforecast.py). The month pass also feeds every decoded
message to gefs_reforecast.Weekly, so the weekly climatology's per-start files (rf_YYYYMMDD.npz) come out of the same
fetch. One reduction function (Acc) serves both sides, so forecast and climatology are the same arithmetic by
construction:

  sampling    instantaneous fields = mean of the 12Z and 00Z samples ending each day; precipitation = the four 6-h
              accumulations summed (mm/day); OLR = the four 6-h averages meaned
  grid        the reforecast's days 1-10 are 0.25 deg, everything else (and all of the live run) 0.5 deg. The 0.25 deg
              fields are SUBSAMPLED to the 0.5 deg points first (the 0.5 deg grid nests in the 0.25 deg one), so every
              value is then box-averaged from the same 0.5 deg points: 3x3 to 1.5 deg, 5x5 to 2.5 deg, both centred.
              (v1 box-averaged the 0.25 deg fields 6x6, an even box whose centre sits 0.125 deg off the target point.)

    python gefs_daily.py month 2019 1 --out out/            # the starts of one month -> out/rf2_ and rf_YYYYMMDD.npz
    python gefs_daily.py combine 1 --inp out/ --out clim/   # the 5-day centres in month 1 -> clim/gefs_clim2_DDD.npz
    python gefs_daily.py compare --old old/ --new clim/     # every climatology file in both: largest differences
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_reforecast as R  # (get, index, decode, S3, grids, the weekly reduction)

# tag -> (reforecast file days 1-10, days 10-35, level text, kind, live GRIB shortName)
D2 = {
    "t2m": ("tmp_2m", "tmp_2m", "2 m above ground", "inst", "TMP"),
    "mslp": ("pres_msl", "pres_msl", "mean sea level", "inst", "PRMSL"),
    "zg500": ("hgt_pres_abv700mb", "hgt_pres", "500 mb", "inst", "HGT"),
    "u850": ("ugrd_pres", "ugrd_pres", "850 mb", "inst", "UGRD"),
    "v850": ("vgrd_pres", "vgrd_pres", "850 mb", "inst", "VGRD"),
    "u200": ("ugrd_pres_abv700mb", "ugrd_pres", "200 mb", "inst", "UGRD"),
    "v200": ("vgrd_pres_abv700mb", "vgrd_pres", "200 mb", "inst", "VGRD"),
    "pr": ("apcp_sfc", "apcp_sfc", "surface", "acc", "APCP"),
    "olr": ("ulwrf_tatm", "ulwrf_tatm", "top of atmosphere", "ave", "ULWRF"),
    "zg100": ("hgt_pres_abv700mb", "hgt_pres", "100 mb", "inst", "HGT"),
    "zg10": ("hgt_pres_abv700mb", "hgt_pres", "10 mb", "inst", "HGT"),
    "t100": ("tmp_pres_abv700mb", "tmp_pres", "100 mb", "inst", "TMP"),
    "t10": ("tmp_pres_abv700mb", "tmp_pres", "10 mb", "inst", "TMP"),
    "u100": ("ugrd_pres_abv700mb", "ugrd_pres", "100 mb", "inst", "UGRD"),
    "u10": ("ugrd_pres_abv700mb", "ugrd_pres", "10 mb", "inst", "UGRD"),
}
# Live-only fields: the vortex loop's total wind needs v at 10/100 hPa, an absolute quantity with no climatology, so
# the reforecast never fetches them (and D2, which the reforecast and combine iterate, must not list them).
LIVE_EXTRA = {
    "v100": (None, None, "100 mb", "inst", "VGRD"),
    "v10": (None, None, "10 mb", "inst", "VGRD"),
}
TABLE = {**D2, **LIVE_EXTRA}
# the weekly climatology (gefs_reforecast.Weekly) reads the same messages as D2 in the one fetch of rf_member
assert all(D2[t][:4] == R.FIELDS[t] for t in R.FIELDS), "gefs_reforecast.FIELDS must be messages of D2"
MAPS = ("t2m", "pr", "mslp", "zg500", "olr", "u850", "u200", "v850", "v200")  # the GEPS page's nine map fields
VORTEX = ("zg10", "zg100", "t10", "t100", "u10", "u100")  # the vortex loop's fields
MEMBER25 = ("zg500", "mslp", "zg100")  # per-member, 2.5 deg
BANDF = ("olr", "u200", "u850")  # MJO band, as v1
# int16 storage: q = round((value - offset) * factor)
SCALE = {
    "t2m": (273.15, 50.0),
    "zg500": (5500.0, 5.0),
    "mslp": (100000.0, 0.5),
    "u200": (0.0, 100.0),
    "u850": (0.0, 100.0),
    "v200": (0.0, 100.0),
    "v850": (0.0, 100.0),
    "pr": (0.0, 100.0),
    "olr": (200.0, 100.0),
    "zg100": (16000.0, 5.0),
    "zg10": (30500.0, 5.0),
    "t100": (220.0, 100.0),
    "t10": (220.0, 100.0),
    "u100": (0.0, 100.0),
    "u10": (0.0, 100.0),
}
# vortex scalars: (name, source tag, reduction, hemisphere)
SCALARS = [
    (f"{k}{lev}{h}", f"{src}{lev}", k, h)
    for lev in (10, 100)
    for h in ("N", "S")
    for k, src in (("u60", "u"), ("capT", "t"), ("capZ", "zg"), ("capZd", "zg"))
]
LAT25 = np.arange(90.0, -90.0 - 1e-6, -2.5)  # 73, the NCEP R1 grid
LON25 = np.arange(0.0, 360.0 - 1e-6, 2.5)  # 144
DAYS = 35
WEEKS = R.WEEKS


def pack(t, a):
    off, fac = SCALE[t]
    return np.clip(np.round((np.asarray(a) - off) * fac), -32767, 32767).astype("int16")


def unpack(t, q):
    off, fac = SCALE[t]
    return np.asarray(q).astype("float64") / fac + off


def to_half(v):
    """Native field (90 -> -90 rows, 0E first) on the 0.5 deg points: 0.25 deg subsampled, 0.5 deg as is."""
    nj = v.shape[0]
    if nj == 721:
        return v[::2, ::2]
    if nj == 361:
        return v
    raise ValueError(f"unexpected grid {v.shape}")


def box(h, n):
    """Centred n x n box mean of a 0.5 deg field, sampled every n points (n odd): 3 -> 1.5 deg, 5 -> 2.5 deg."""
    from scipy.ndimage import uniform_filter1d

    a = uniform_filter1d(h, n, axis=0, mode="nearest")
    a = uniform_filter1d(a, n, axis=1, mode="wrap")
    return a[::n, ::n]


LAT_H = np.arange(90.0, -90.0 - 1e-6, -0.5)
COS_H = np.cos(np.deg2rad(LAT_H))


def scalar(h, kind, hemi):
    """One vortex scalar from a 0.5 deg field (strat_series.reduce_field's definitions)."""
    north = hemi == "N"
    if kind == "u60":
        return float(h[round((90.0 - (60.0 if north else -60.0)) / 0.5)].mean())
    zm = h.mean(1)
    cap = (LAT_H >= 65) if north else (LAT_H <= -65)
    v = float((zm[cap] * COS_H[cap]).sum() / COS_H[cap].sum())
    if kind == "capZd":  # relative to the hemisphere poleward of 20
        hm = (LAT_H >= 20) if north else (LAT_H <= -20)
        v -= float((zm[hm] * COS_H[hm]).sum() / COS_H[hm].sum())
    return v


class Acc:
    """Daily accumulation of one member: fields at 1.5 deg, MEMBER25 at 2.5 deg, vortex scalars."""

    def __init__(self, tags=tuple(D2)):
        self.tags = tuple(tags)
        self.d = {t: np.zeros((DAYS, len(R.LAT), len(R.LON))) for t in self.tags}
        self.n = {t: np.zeros(DAYS) for t in self.tags}
        self.m = {t: np.zeros((DAYS, len(LAT25), len(LON25))) for t in MEMBER25 if t in self.tags}
        self.s = np.zeros((len(SCALARS), DAYS))
        self.sn = np.zeros((len(SCALARS), DAYS))

    def add(self, tag, day, v):
        h = to_half(np.asarray(v, dtype="float64"))
        i = day - 1
        self.d[tag][i] += box(h, 3)
        self.n[tag][i] += 1
        if tag in self.m:
            self.m[tag][i] += box(h, 5)
        for k, (_, src, kind, hemi) in enumerate(SCALARS):
            if src == tag:
                self.s[k, i] += scalar(h, kind, hemi)
                self.sn[k, i] += 1

    def finish(self):
        """-> dict of daily arrays; None if any day of any field is missing."""
        out = {}
        for t in self.tags:
            if (self.n[t] == 0).any():
                return None
            f = self.d[t] if TABLE[t][3] == "acc" else self.d[t] / self.n[t][:, None, None]
            out[f"d_{t}"] = f
            if t in self.m:
                out[f"m_{t}"] = self.m[t] if TABLE[t][3] == "acc" else self.m[t] / self.n[t][:, None, None]
        want = [k for k, sc in enumerate(SCALARS) if sc[1] in self.tags]
        if want:
            if (self.sn[want] == 0).any():
                return None
            out["s"] = self.s / np.maximum(self.sn, 1)
        return out


def weekly(daily):
    return np.stack([daily[a - 1 : b].mean(0) for a, b in WEEKS])


def band(daily):
    return daily[:, R.BAND, :].mean(1)


# ── reforecast side ────────────────────────────────────────────────────────────────────────────────────────────
def wanted(tag):
    """(day, which file, step text) for every message the tag needs (gefs_reforecast.wanted, any tag of D2)."""
    kind = D2[tag][3]
    out = []
    for d in range(1, DAYS + 1):
        if kind == "inst":
            out.extend((d, 1 if h <= 240 else 2, f"{h} hour fcst") for h in (24 * d - 12, 24 * d))
        else:
            out.extend(
                (d, 1 if h + 6 <= 240 else 2, f"{h}-{h + 6} hour {kind} fcst") for h in range(24 * d - 24, 24 * d, 6)
            )
    return out


def rf_member(start: str, mem: str):
    """(Acc.finish(), gefs_reforecast.Weekly.finish()) of one member of one start from one fetch, or None if its
    files are incomplete."""
    base = f"{R.S3}/{start[:4]}/{start}00/{mem}"
    idx_cache, jobs = {}, []
    for tag, (f1, f2, lev, _kind, _) in D2.items():
        for d, which, stext in wanted(tag):
            f = f"{base}/Days:{'1-10' if which == 1 else '10-35'}/{f1 if which == 1 else f2}_{start}00_{mem}.grib2"
            if f not in idx_cache:
                try:
                    idx_cache[f] = R.index(f)
                except Exception:  # noqa: BLE001
                    return None
            hit = [e for e in idx_cache[f] if e[2] == lev and e[3] == stext]
            if not hit:
                return None
            jobs.append((tag, d, f, hit[0][0], hit[0][1]))
    acc, wk = Acc(), R.Weekly()
    with ThreadPoolExecutor(24) as ex:
        for (tag, d, *_), b in zip(jobs, ex.map(lambda j: R.get(j[2], (j[3], j[4])), jobs)):
            v = R.decode(b)
            acc.add(tag, d, v)
            if tag in R.FIELDS:
                wk.add(tag, d, v)
    daily = acc.finish()
    return None if daily is None else (daily, wk.finish())


def run_start(args):
    start, out = args
    dest, wdest = Path(out) / f"rf2_{start}.npz", Path(out) / f"rf_{start}.npz"
    if dest.exists() and wdest.exists():
        return f"{start}: exists"
    t0 = time.time()
    res = {}
    for m in R.MEMBERS:
        try:
            r = rf_member(start, m)
        except Exception as e:  # noqa: BLE001
            r = None
            print(f"  {start} {m}: {str(e)[:120]}", flush=True)
        if r is not None:
            res[m] = r
    if not res:
        return f"{start}: no complete member"
    ms = list(res)
    save = {
        "members": np.array(ms),
        "scalars": np.array([s[0] for s in SCALARS]),
        "s": np.stack([res[m][0]["s"] for m in ms]).astype("float32"),
    }
    for t in D2:
        save[f"d_{t}"] = pack(t, np.mean([res[m][0][f"d_{t}"] for m in ms], axis=0))
    for t in MEMBER25:
        save[f"m_{t}"] = pack(t, np.stack([res[m][0][f"m_{t}"] for m in ms]))
    np.savez_compressed(dest, **save)
    R.save_weekly(wdest, {m: res[m][1] for m in ms})
    return f"{start}: {len(ms)} members in {time.time() - t0:.0f} s"


def cmd_month(a) -> int:
    """Every start of one month; fails when a start has no complete member (a climatology pool must not shrink)."""
    Path(a.out).mkdir(parents=True, exist_ok=True)
    starts = [s for s in R.wednesdays(a.year) if int(s[4:6]) == a.month][: a.limit or None]
    with ProcessPoolExecutor(a.procs) as ex:
        for msg in ex.map(run_start, [(s, a.out) for s in starts]):
            print(msg, flush=True)
    missing = [s for s in starts if not all((Path(a.out) / f"{p}_{s}.npz").exists() for p in ("rf2", "rf"))]
    if missing:
        print(f"no complete member: {', '.join(missing)}", flush=True)
        return 1
    return 0


def cmd_combine(a) -> int:
    """Daily climatology per 5-day centre from every start within +-15 days (any year): the member-mean fields
    weighted by their member count, the scalars over every member."""
    files = sorted(Path(a.inp).rglob("rf2_*.npz"))
    doy = R.noleap_doy([dt.datetime.strptime(f.stem[4:], "%Y%m%d").date() for f in files])  # common-year day key
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for c in R.centres_in(a.month):
        sel = [f for f, x in zip(files, R.pool(doy, c)) if x]
        acc = {t: np.zeros((DAYS, len(R.LAT), len(R.LON))) for t in D2}
        a25 = {t: np.zeros((DAYS, len(LAT25), len(LON25))) for t in MEMBER25}
        s = np.zeros((len(SCALARS), DAYS))
        n = 0
        for f in sel:
            z = np.load(f)
            k = len(z["members"])
            for t in D2:
                acc[t] += k * unpack(t, z[f"d_{t}"])
            for t in MEMBER25:
                a25[t] += unpack(t, z[f"m_{t}"]).sum(0)
            s += z["s"].astype("float64").sum(0)
            n += k
        if n == 0:
            continue
        # c25_<tag>: the same climatology on the 2.5 deg grid of the per-member fields (teleconnections, regimes)
        np.savez_compressed(
            Path(a.out) / f"gefs_clim2_{c:03d}.npz",
            n=n,
            n_starts=len(sel),
            scalars=np.array([x[0] for x in SCALARS]),
            s=(s / n).astype("float32"),
            **{f"d_{t}": pack(t, acc[t] / n) for t in D2},
            **{f"c25_{t}": pack(t, a25[t] / n) for t in MEMBER25},
        )
        print(f"doy {c:3d}: {len(sel)} starts, {n} member-runs", flush=True)
    # the per-member hindcast fields of this month's own starts, for the calibrations (teleconnection and regime
    # hindcasts, vortex spread): one compact file per month, keyed by start
    own = [f for f in files if int(f.stem[8:10]) == a.month]
    if own:
        keep = {"scalars": np.array([x[0] for x in SCALARS])}
        for f in own:
            z = np.load(f)
            st = f.stem[4:]
            keep[f"{st}_members"] = z["members"]
            keep[f"{st}_s"] = z["s"]
            for t in MEMBER25:
                keep[f"{st}_m_{t}"] = z[f"m_{t}"]
        np.savez_compressed(Path(a.out) / f"gefs_hind2_{a.month:02d}.npz", **keep)
        print(f"hindcast members: {len(own)} starts of month {a.month}", flush=True)
    return 0


def cmd_compare(a) -> int:
    """Every climatology file (gefs_clim_DDD weekly, gefs_clim2_DDD daily) present under the same name in --old and
    --new: per field, the largest absolute and the rms difference over its leads and grid in GEFS units (packed
    fields unpacked), and the member-runs of each. With --json, the table and the per-field totals as a file."""
    skip = ("n", "n_starts", "lat", "lon", "scalars")
    rows, tot = [], {}
    for f in sorted(Path(a.new).glob("gefs_clim*_*.npz")):
        o = Path(a.old) / f.name
        if not o.exists():
            continue
        with np.load(f) as zn, np.load(o) as zo:
            row = {"file": f.name, "n_old": int(zo["n"]), "n_new": int(zn["n"]), "fields": {}}
            for k in [k for k in zn.files if k in zo.files and k not in skip]:
                new, old = (
                    unpack(k.split("_", 1)[1], z[k]) if k.startswith(("d_", "c25_")) else z[k].astype("float64")
                    for z in (zn, zo)
                )
                d = new - old
                mx, ms = float(np.abs(d).max()), float((d**2).mean())
                row["fields"][k] = {"max": round(mx, 4), "rms": round(float(np.sqrt(ms)), 4)}
                t = tot.setdefault(k, {"max": 0.0, "ms": 0.0, "files": 0})
                t["max"], t["ms"], t["files"] = max(t["max"], mx), t["ms"] + ms, t["files"] + 1
        rows.append(row)
        big = sorted(row["fields"].items(), key=lambda kv: -kv[1]["rms"])[:4]
        print(
            f"{f.name}: member-runs {row['n_old']} -> {row['n_new']}; largest rms "
            + ", ".join(f"{k} {v['rms']:g} (max {v['max']:g})" for k, v in big),
            flush=True,
        )
    fields = {
        k: {"max": round(v["max"], 4), "rms": round(float(np.sqrt(v["ms"] / v["files"])), 4), "files": v["files"]}
        for k, v in tot.items()
    }
    for k, v in fields.items():
        print(f"  {k}: max |new - old| {v['max']:g}, rms {v['rms']:g} over {v['files']} files", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps({"files": rows, "fields": fields}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("month")
    m.add_argument("year", type=int)
    m.add_argument("month", type=int)
    m.add_argument("--out", default="out")
    m.add_argument("--procs", type=int, default=4)
    m.add_argument("--limit", type=int, default=0)
    c = sp.add_parser("combine")
    c.add_argument("month", type=int)
    c.add_argument("--inp", default="out")
    c.add_argument("--out", default="clim")
    k = sp.add_parser("compare")
    k.add_argument("--old", required=True, help="the previous release's files")
    k.add_argument("--new", required=True)
    k.add_argument("--json", help="write the comparison here")
    a = ap.parse_args()
    return {"month": cmd_month, "combine": cmd_combine, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
