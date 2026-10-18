#!/usr/bin/env python3
"""Per-station sounding climatology for the Skew-T Explorer.

For each IGRA station we download the full period-of-record file, compute a set
of indices for every historical sounding, and reduce them to day-of-year
percentile breakpoints + record extremes, separately for each synoptic hour
(00Z and 12Z). The browser then fetches one small JSON per station
(climo/{gid}.json on the skewt-climo branch) and reports where the
live/archived sounding's values rank against that station's own history at the
same hour.

Output (version 2): "hr" holds one table per synoptic hour ("00", "12"), each
{index: {n, p, min, max, minY, maxY}} over the anchors in "doy"; "nh" counts
the soundings behind each hour. "idx" is the same reduction over both hours
pooled, kept for readers that want an all-hours baseline; ranking and records
use "hr". Launches more than HOUR_TOL hours from 00Z or 12Z (mostly event-driven
specials) and launches with no hour are left out of every table.

Indices: pwat 850t 700t 500t h500 thick fzl kidx tott (exact from levels) plus
  ecape ship (via the native SHARPlib helper CLIMO_CAPE_BIN, identical physics
  to the app's WASM).

    python scripts/skewt/build_climo.py OUTDIR [gid1 gid2 ...]   # subset
    python scripts/skewt/build_climo.py OUTDIR --all            # every station
"""

from __future__ import annotations

import concurrent.futures as cf
import functools
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from quantmet.degdays import doy366

# /pub/data/igra/ serves the same files; ncei.noaa.gov/robots.txt disallows /data*
IGRA = "https://www.ncei.noaa.gov/pub/data/igra/data/data-por/"
UA = "scorvec.com skew-t explorer (+https://scorvec.com/skewt/)"
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
# day-of-year climatology: an anchor every 5 days, each pooling a +/-10-day
# window. Far better than calendar months — a July 11 sounding is compared with
# early-July weather, not with everything from July 1 to July 31.
# Anchors and samples are on a 366-slot calendar (1..366; 29 Feb = 60 and
# 1 March = 61 in every year), so a calendar date sits in the same slot in leap
# and common years; skewt/app.js and flag_anomalies.py look up the same way.
DOY_STEP, DOY_WIN = 5, 10
DOY = list(range(1, 366, DOY_STEP))
FORMAT = 2
# Synoptic hours the climatology is split by, and how far (hours) a launch may
# sit from one to count for it. +/-3 h keeps routine launches filed at their
# actual hour (23Z, 11Z) and the 03Z/15Z schedule of the 1940s-50s US network;
# launches further out (04-08Z, 16-20Z) are mostly specials released ahead of
# severe weather or a landfalling storm, a sample conditional on the weather.
SYN_HOURS = (0, 12)
HOUR_TOL = 3.0
# Readers rank only against anchors with at least this many samples; an hour
# with no such anchor is not written.
MIN_N = 30
KEYS = [
    "pwat",
    "850t",
    "700t",
    "500t",
    "850td",
    "700td",
    "h500",
    "thick",
    "fzl",
    "kidx",
    "tott",
    "ecape",
    "ship",
    "850spd",
    "250spd",
]
# Physical plausibility (m / degC / mm) — shared with flag_anomalies.py.
# Outside these bounds is a data error anywhere on Earth; applied at the
# SAMPLE level so one garbled height group can't enter any percentile or
# record. The station-relative global fence below is too loose for
# variables with a wide annual cycle (Green Bay published a 4017 m
# thickness "record" that passed it comfortably).
PHYS = {
    "h500": (4600, 6100),
    "thick": (4700, 6100),
    "850t": (-60, 45),
    "700t": (-55, 35),
    "500t": (-60, 15),
    "850td": (-75, 35),
    "700td": (-75, 30),
    "pwat": (0, 135),
    # wind speed m/s: 850 record ~ hurricane cores; 250 ~ strongest jets
    "850spd": (0, 110),
    "250spd": (0, 165),
}
# Per-sounding values are cached, so re-aggregating (different window, new
# percentiles) never re-downloads the 30-90 MB period-of-record file again.
CACHE = Path(os.environ.get("CLIMO_CACHE", "/tmp/climo_cache"))
CAPE_BIN = os.environ.get("CLIMO_CAPE_BIN", "/tmp/climo_cape")
WORKERS = int(os.environ.get("CLIMO_WORKERS", "6"))
G = 9.80665
HERE = Path(__file__).resolve().parent
STATIONS = HERE.parents[1] / "skewt" / "stations.json"


def _f(s):
    v = int(s)
    return np.nan if v <= -8888 else v


def launch_hour(header: str) -> float:
    """Hour of day (UTC) of an IGRA v2 header: the nominal HOUR, else the release time (HHMM), else NaN."""
    try:
        h = int(header[24:26])
    except ValueError:
        h = 99
    if 0 <= h <= 23:
        return float(h)
    try:
        rt = int(header[27:31])
    except ValueError:
        return np.nan
    hh, mm = divmod(rt, 100)
    if not 0 <= hh <= 23:
        return np.nan
    return hh + (mm / 60.0 if 0 <= mm <= 59 else 0.0)


def sounding_indices(block: list[str], elev: float = 0.0) -> dict | None:
    """block = the level lines of one sounding. Returns index dict or None."""
    P, T, D, H, W = [], [], [], [], []
    for L in block:
        if len(L) < 52:
            continue
        p = _f(L[9:15])
        gph = _f(L[16:21])
        tt = _f(L[22:27])
        dpdp = _f(L[34:39])
        rh = _f(L[28:33])  # pre-~1990 US records carry RH, not DPDP
        ws = _f(L[46:51])  # wind speed, m/s x10
        if np.isnan(p) and not np.isnan(gph):  # pibal — skip for thermo climo
            continue
        if np.isnan(p) or p < 2000:
            continue
        P.append(p)
        H.append(gph)
        W.append(np.nan if np.isnan(ws) else ws / 10.0)
        T.append(np.nan if np.isnan(tt) else tt / 10 + 273.15)
        if np.isnan(tt):
            D.append(np.nan)
        elif not np.isnan(dpdp):
            D.append(tt / 10 - dpdp / 10 + 273.15)
        elif not np.isnan(rh) and rh > 0:  # derive Td from RH (inverse Magnus)
            tc = tt / 10
            g = np.log(min(100.0, rh / 10) / 100.0) + 17.67 * tc / (tc + 243.5)
            D.append(243.5 * g / (17.67 - g) + 273.15)
        else:
            D.append(np.nan)
    if len(P) < 8:
        return None
    P, T, D, H, W = map(np.asarray, (P, T, D, H, W))
    o = np.argsort(-P)  # sfc -> top
    P, T, D, H, W = P[o], T[o], D[o], H[o], W[o]
    if not np.isfinite(H[0]):  # IGRA often omits sfc height
        H = H.copy()
        H[0] = elev

    def at(field, plevel):
        pv = plevel * 100.0
        if pv >= P[0]:
            return field[0]
        for i in range(1, len(P)):
            if P[i] <= pv:
                if not (np.isfinite(field[i]) and np.isfinite(field[i - 1])):
                    return np.nan
                f = np.log(P[i - 1] / pv) / np.log(P[i - 1] / P[i])
                return field[i - 1] + f * (field[i] - field[i - 1])
        return np.nan

    # PWAT: (1/g) ∫ w dp  (mm). Many stations report dewpoint on only a fraction
    # of levels; integrating a shallow/sparse moisture profile yields absurd
    # values (0.5 mm for Argentina), so require real coverage or report nothing.
    pw = np.nan
    m = np.isfinite(P) & np.isfinite(D)
    moist_ok = (
        m.sum() >= 6
        and P[m].min() <= 50000  # moisture to 500 hPa
        and np.isfinite(D[0])  # real surface dewpoint
        and m[P >= 50000].mean() >= 0.6
    )  # 60%+ of levels below 500
    if moist_ok:
        Pm, Dm = P[m], D[m]
        Tdc = Dm - 273.15
        e = 6.112 * np.exp(17.67 * Tdc / (Tdc + 243.5)) * 100.0  # Pa
        w = 0.622 * e / np.maximum(Pm - e, 1.0)
        dP = -np.diff(Pm)  # positive going up
        pw = float(np.sum(0.5 * (w[:-1] + w[1:]) * dP) / G)

    t850, t700, t500 = at(T, 850), at(T, 700), at(T, 500)
    d850, d700 = at(D, 850), at(D, 700)
    h500, h1000 = at(H, 500), at(H, 1000)
    fzl = np.nan
    for i in range(1, len(P)):
        if np.isfinite(T[i]) and np.isfinite(T[i - 1]) and T[i - 1] >= 273.15 > T[i]:
            f = (273.15 - T[i - 1]) / (T[i] - T[i - 1])
            fzl = float(H[i - 1] + f * (H[i] - H[i - 1]) - H[0])
            break

    def c(k):
        return np.nan if np.isnan(k) else k - 273.15

    idx = {
        "pwat": pw,
        "850t": c(t850),
        "700t": c(t700),
        "500t": c(t500),
        "850td": c(d850),
        "700td": c(d700),
        "h500": h500,
        # real 1000 hPa surface required — elevated stations otherwise yield
        # h500 minus station elevation, poisoning the thickness climatology
        "thick": (h500 - h1000) if (np.isfinite(h500) and np.isfinite(h1000) and P[0] >= 99000) else np.nan,
        "fzl": fzl,
        "kidx": ((c(t850) - c(t500)) + c(d850) - (c(t700) - c(d700))) if moist_ok else np.nan,
        "tott": (c(t850) + c(d850) - 2 * c(t500)) if moist_ok else np.nan,
        "850spd": at(W, 850),
        "250spd": at(W, 250),
    }
    return {k: v for k, v in idx.items() if v is not None and np.isfinite(v)}


@functools.cache
def _elevations() -> dict:
    """Station elevations from the station list, read once."""
    return {s["gid"]: s.get("e", 0) or 0 for s in json.loads(STATIONS.read_text())["stations"]}


def station_elev(gid: str) -> float:
    """IGRA often omits the SURFACE geopotential height; it's the station elevation."""
    return float(_elevations().get(gid, 0))


def _samples(gid: str) -> dict | None:
    """Per-sounding index values for a station: {"doy", "yr", "hod", "v": {key: values}}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{gid}.npz"
    if cf.exists():
        try:
            z = np.load(cf)
            if "hod" in z and all(k in z for k in KEYS if k not in ("ecape", "ship")):
                return {"doy": z["doy"], "yr": z["yr"], "hod": z["hod"], "v": {k: z[k] for k in KEYS if k in z}}
            # cache predates the launch hour or a newly added index — refetch to fill it
        except Exception:  # noqa: BLE001
            pass  # corrupt cache: refetch

    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(IGRA + f"{gid}-data.txt.zip", headers={"User-Agent": UA}), timeout=180
        ).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        text = zf.read(zf.namelist()[0]).decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        print(f"  {gid}: fetch failed ({repr(e)[:50]})", flush=True)
        return None

    rows, cape = parse_por(text, gid, station_elev(gid))
    if not rows:
        return None
    s = reduce_rows(rows, cape)
    np.savez_compressed(CACHE / f"{gid}.npz", doy=s["doy"], yr=s["yr"], hod=s["hod"], **s["v"])
    return s


def parse_por(text: str, gid: str, elev: float) -> tuple[list, dict]:
    """Rows (doy, year, month, day, raw header hour, launch hour, {key: val}) of an IGRA period-of-record
    text, and the ECAPE/SHIP values of the native helper keyed by (year, month, day, raw hour)."""
    rows: list = []
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        L = lines[i]
        if L.startswith("#" + gid):
            year, month, day = int(L[13:17]), int(L[18:20]), int(L[21:23])
            nlev = int(L[32:36])
            block = lines[i + 1 : i + 1 + nlev]
            i += 1 + nlev
            vals = sounding_indices(block, elev)
            if vals:
                try:
                    doy = datetime(year, month, max(1, day)).timetuple().tm_yday
                except ValueError:
                    continue
                hraw = int(L[24:26]) if L[24:26].strip().isdigit() else 99
                rows.append((doy, year, month, max(1, day), hraw, launch_hour(L), vals))
            continue
        i += 1

    # ECAPE + SHIP from the native SHARPlib helper (identical physics to the app)
    cape: dict = {}
    if os.path.exists(CAPE_BIN):
        try:
            r = subprocess.run(
                [CAPE_BIN, gid, str(elev)], input=text, capture_output=True, text=True, timeout=900, check=False
            )
            for ln in r.stdout.splitlines():
                q = ln.split()
                if len(q) not in (5, 6):
                    continue
                # "Y M D H ecape ship"; an older helper prints no hour and is matched by date only
                key = (int(q[0]), int(q[1]), int(q[2]), int(q[3]) if len(q) == 6 else None)
                cape.setdefault(key, []).append(
                    (float(q[-2]) if q[-2] != "nan" else np.nan, float(q[-1]) if q[-1] != "nan" else np.nan)
                )
        except Exception as e:  # noqa: BLE001
            print(f"  {gid}: cape helper failed ({repr(e)[:40]})", flush=True)
    return rows, cape


def reduce_rows(rows: list, cape: dict) -> dict:
    """Per-sounding arrays from parse_por's rows; ECAPE/SHIP attached by date and launch hour, in file order."""
    doy = np.array([r[0] for r in rows], dtype="int16")
    yr = np.array([r[1] for r in rows], dtype="int16")
    hod = np.array([r[5] for r in rows], dtype="float32")
    v = {k: np.array([r[6].get(k, np.nan) for r in rows], dtype="float32") for k in KEYS if k not in ("ecape", "ship")}
    for k, j in (("ecape", 0), ("ship", 1)):
        out = np.full(len(rows), np.nan, dtype="float32")
        seen: dict = {}
        for idx, (_, y, mo, dy, hraw, _, _) in enumerate(rows):
            key = (int(y), int(mo), int(dy), int(hraw))
            if key not in cape:
                key = (int(y), int(mo), int(dy), None)
            lst = cape.get(key)
            if not lst:
                continue
            c = seen.get(key, 0)
            if c < len(lst):
                out[idx] = lst[c][j]
                seen[key] = c + 1
        v[k] = out
    return {"doy": doy, "yr": yr, "hod": hod, "v": v}


def hour_classes(hod) -> np.ndarray:
    """Synoptic class (0 or 12) per launch hour, -1 where the hour is unknown or beyond HOUR_TOL of both."""
    h = np.mod(np.asarray(hod, dtype="float64"), 24.0)
    cls = np.full(h.shape, -1, dtype="int8")
    for s in SYN_HOURS:
        d = np.abs(h - s)
        cls[np.isfinite(h) & (np.minimum(d, 24.0 - d) <= HOUR_TOL)] = s
    return cls


def synoptic_class(hour) -> int | None:
    """00Z or 12Z class of one launch hour (UTC), or None; the same rule as hour_classes."""
    try:
        c = int(hour_classes([float(hour)])[0])
    except (TypeError, ValueError):
        return None
    return None if c < 0 else c


def _anchor_table(vals: np.ndarray, ok: np.ndarray, slot: np.ndarray, yr: np.ndarray, fence: bool) -> dict:
    """Percentiles and records at every DOY anchor from the samples where `ok`, each pooling +/-DOY_WIN days."""
    n_l, p_l, mn_l, mx_l, mnY_l, mxY_l = [], [], [], [], [], []

    def empty(c):
        n_l.append(c)
        p_l.append([None] * len(PCTS))
        mn_l.append(None)
        mx_l.append(None)
        mnY_l.append(None)
        mxY_l.append(None)

    for a in DOY:
        d = np.abs(slot - a)
        d = np.minimum(d, 366 - d)  # wrap the year
        m = ok & (d <= DOY_WIN)
        c = int(m.sum())
        if c < 5:
            empty(c)
            continue
        vv, yy = vals[m], yr[m]
        # Window-local fence: the global fence is scaled to the full annual
        # cycle, so it can't see seasonally-impossible values (a winter-grade
        # thickness in an August window). Same 3-tail-span rule, but against
        # THIS window's distribution.
        if fence and c >= 20:
            q1, q50, q99 = np.percentile(vv, [1, 50, 99])
            w_lo = q1 - 3.0 * max(q50 - q1, 1.0)
            w_hi = q99 + 3.0 * max(q99 - q50, 1.0)
            keep = (vv >= w_lo) & (vv <= w_hi)
            if not keep.all():
                vv, yy = vv[keep], yy[keep]
                c = len(vv)
                if c < 5:
                    empty(c)
                    continue
        n_l.append(c)
        p_l.append([round(float(x), 1) for x in np.percentile(vv, PCTS)])
        i0, i1 = int(np.argmin(vv)), int(np.argmax(vv))
        mn_l.append(round(float(vv[i0]), 1))
        mnY_l.append(int(yy[i0]))
        mx_l.append(round(float(vv[i1]), 1))
        mxY_l.append(int(yy[i1]))
    return {"n": n_l, "p": p_l, "min": mn_l, "max": mx_l, "minY": mnY_l, "maxY": mxY_l}


def station_tables(gid: str, s: dict) -> dict | None:
    """The station's climatology JSON (format 2) from its per-sounding samples, or None."""
    doy, yr, V = s["doy"], s["yr"], s["v"]
    # the cache keeps the true day of year (tm_yday); pool on the 366-slot calendar
    dates = (yr.astype("int64") - 1970).astype("datetime64[Y]").astype("datetime64[D]") + (doy.astype("int64") - 1)
    slot = doy366(dates) + 1
    cls = hour_classes(s["hod"])
    syn = cls >= 0
    if not syn.any():
        return None

    pooled: dict = {}
    hours: dict = {f"{h:02d}": {} for h in SYN_HOURS}
    for k in KEYS:
        vals = V.get(k)
        if vals is None:
            continue
        ok = np.isfinite(vals) & syn
        if ok.sum() < 20:
            continue
        # Robust QC: one corrupt sounding must not set a record envelope
        # (seen: a -40 C spike in a tropical station whose real record low
        # was -9). Reject values beyond 3x the distribution's own tail span
        # past the 1st/99th percentile -- scales with the tail, so genuine
        # records (heavy Td dry tails etc.) survive. ecape/ship excluded:
        # zero-inflated distributions make any quantile fence meaningless.
        if k in PHYS:
            lo_p, hi_p = PHYS[k]
            bad = ok & ((vals < lo_p) | (vals > hi_p))
            if bad.any():
                print(f"  {gid} {k}: PHYS rejected {int(bad.sum())} (outside [{lo_p},{hi_p}])", flush=True)
                ok = ok & ~bad
                if ok.sum() < 20:
                    continue
        fence = k not in ("ecape", "ship")
        if fence:
            p1, p50, p99 = np.percentile(vals[ok], [1, 50, 99])
            lo = p1 - 3.0 * max(p50 - p1, 1.0)
            hi = p99 + 3.0 * max(p99 - p50, 1.0)
            bad = ok & ((vals < lo) | (vals > hi))
            if bad.any():
                print(f"  {gid} {k}: QC rejected {int(bad.sum())} (kept [{lo:.0f},{hi:.0f}])", flush=True)
                ok = ok & ~bad
                if ok.sum() < 20:
                    continue
        pooled[k] = _anchor_table(vals, ok, slot, yr, fence)
        for h in SYN_HOURS:
            t = _anchor_table(vals, ok & (cls == h), slot, yr, fence)
            if max(t["n"]) >= MIN_N:
                hours[f"{h:02d}"][k] = t
    hours = {h: t for h, t in hours.items() if t}
    if not hours:
        return None
    return {
        "gid": gid,
        "v": FORMAT,
        "tol": HOUR_TOL,
        "doy": DOY,
        "hr": hours,
        "nh": {h: int((cls == int(h)).sum()) for h in hours},
        "idx": pooled,
        "yr0": int(yr[syn].min()),
        "yr1": int(yr[syn].max()),
    }


def build_station(gid: str, outdir: Path) -> bool:
    s = _samples(gid)
    if not s:
        return False
    out = station_tables(gid, s)
    if not out:
        return False
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{gid}.json").write_text(json.dumps(out, separators=(",", ":")))
    hrs = ", ".join(f"{h}Z {n}" for h, n in out["nh"].items())
    print(f"  {gid}: {len(out['idx'])} indices, soundings {hrs} ({out['yr0']}-{out['yr1']})", flush=True)
    return True


def main() -> int:
    outdir = Path(sys.argv[1])
    args = sys.argv[2:]
    stns = [s["gid"] for s in json.loads(STATIONS.read_text())["stations"]] if "--all" in args else args
    todo = []
    ok = 0
    reagg = os.environ.get("CLIMO_REAGG") == "1"
    for gid in stns:
        fp = outdir / f"{gid}.json"
        if reagg:
            # re-aggregate from the per-sounding cache only (post-hoc QC pass);
            # stations without a cache are left to the normal build
            if (CACHE / f"{gid}.npz").exists():
                todo.append(gid)
            continue
        if fp.exists():
            try:
                have = json.loads(fp.read_text())
                if have.get("v") == FORMAT:
                    ok += 1
                    continue  # already split by synoptic hour
            except Exception:  # noqa: BLE001
                pass  # rebuild on parse error
        todo.append(gid)

    # network-bound (30-90 MB per station) — fetch/compute several at once
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(build_station, g, outdir): g for g in todo}
        for f in cf.as_completed(futs):
            try:
                if f.result():
                    ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  {futs[f]}: failed ({repr(e)[:40]})", flush=True)
    print(f"built {ok}/{len(stns)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
