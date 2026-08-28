#!/usr/bin/env python3
"""Per-station sounding climatology for the Skew-T Explorer.

For each IGRA station we download the full period-of-record file, compute a set
of indices for every historical sounding, and reduce them to monthly percentile
breakpoints + record extremes. The browser then fetches one small JSON per
station (climo/{gid}.json on the skewt-climo branch) and reports where the
live/archived sounding's values rank against that station's own history.

Indices: pwat 850t 700t 500t h500 thick fzl kidx tott (exact from levels) plus
  ecape ship (via the native SHARPlib helper CLIMO_CAPE_BIN, identical physics
  to the app's WASM).

    python scripts/skewt/build_climo.py OUTDIR [gid1 gid2 ...]   # subset
    python scripts/skewt/build_climo.py OUTDIR --all            # every station
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

IGRA = ("https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/"
        "access/data-por/")
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
# day-of-year climatology: an anchor every 5 days, each pooling a +/-10-day
# window. Far better than calendar months — a July 11 sounding is compared with
# early-July weather, not with everything from July 1 to July 31.
DOY_STEP, DOY_WIN = 5, 10
DOY = list(range(1, 366, DOY_STEP))
KEYS = ["pwat", "850t", "700t", "500t", "850td", "700td",
        "h500", "thick", "fzl", "kidx", "tott", "ecape", "ship",
        "850spd", "250spd"]
# Physical plausibility (m / degC / mm) — shared with flag_anomalies.py.
# Outside these bounds is a data error anywhere on Earth; applied at the
# SAMPLE level so one garbled height group can't enter any percentile or
# record. The station-relative global fence below is too loose for
# variables with a wide annual cycle (Green Bay published a 4017 m
# thickness "record" that passed it comfortably).
PHYS = {"h500": (4600, 6100), "thick": (4700, 6100), "850t": (-60, 45),
        "700t": (-55, 35), "500t": (-60, 15), "850td": (-75, 35),
        "700td": (-75, 30), "pwat": (0, 135),
        # wind speed m/s: 850 record ~ hurricane cores; 250 ~ strongest jets
        "850spd": (0, 110), "250spd": (0, 165)}
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
        rh = _f(L[28:33])                 # pre-~1990 US records carry RH, not DPDP
        ws = _f(L[46:51])                 # wind speed, m/s x10
        if np.isnan(p) and not np.isnan(gph):        # pibal — skip for thermo climo
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
        elif not np.isnan(rh) and rh > 0:            # derive Td from RH (inverse Magnus)
            tc = tt / 10
            g = np.log(min(100.0, rh / 10) / 100.0) + 17.67 * tc / (tc + 243.5)
            D.append(243.5 * g / (17.67 - g) + 273.15)
        else:
            D.append(np.nan)
    if len(P) < 8:
        return None
    P, T, D, H, W = map(np.asarray, (P, T, D, H, W))
    o = np.argsort(-P)                                # sfc -> top
    P, T, D, H, W = P[o], T[o], D[o], H[o], W[o]
    if not np.isfinite(H[0]):                         # IGRA often omits sfc height
        H = H.copy(); H[0] = elev

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
    moist_ok = (m.sum() >= 6 and P[m].min() <= 50000            # moisture to 500 hPa
                and np.isfinite(D[0])                            # real surface dewpoint
                and m[P >= 50000].mean() >= 0.6)                 # 60%+ of levels below 500
    if moist_ok:
        Pm, Dm = P[m], D[m]
        Tdc = Dm - 273.15
        e = 6.112 * np.exp(17.67 * Tdc / (Tdc + 243.5)) * 100.0    # Pa
        w = 0.622 * e / np.maximum(Pm - e, 1.0)
        dP = -np.diff(Pm)                             # positive going up
        pw = float(np.sum(0.5 * (w[:-1] + w[1:]) * dP) / G)

    t850, t700, t500 = at(T, 850), at(T, 700), at(T, 500)
    d850, d700 = at(D, 850), at(D, 700)
    h500, h1000 = at(H, 500), at(H, 1000)
    fzl = np.nan
    for i in range(1, len(P)):
        if np.isfinite(T[i]) and np.isfinite(T[i - 1]) and \
                T[i - 1] >= 273.15 > T[i]:
            f = (273.15 - T[i - 1]) / (T[i] - T[i - 1])
            fzl = float(H[i - 1] + f * (H[i] - H[i - 1]) - H[0]); break

    def c(k):
        return np.nan if np.isnan(k) else k - 273.15
    idx = {
        "pwat": pw,
        "850t": c(t850), "700t": c(t700), "500t": c(t500),
        "850td": c(d850), "700td": c(d700),
        "h500": h500,
        # real 1000 hPa surface required — elevated stations otherwise yield
        # h500 minus station elevation, poisoning the thickness climatology
        "thick": (h500 - h1000) if (np.isfinite(h500) and np.isfinite(h1000)
                                    and P[0] >= 99000) else np.nan,
        "fzl": fzl,
        "kidx": ((c(t850) - c(t500)) + c(d850) - (c(t700) - c(d700))) if moist_ok else np.nan,
        "tott": (c(t850) + c(d850) - 2 * c(t500)) if moist_ok else np.nan,
        "850spd": at(W, 850), "250spd": at(W, 250),
    }
    return {k: v for k, v in idx.items() if v is not None and np.isfinite(v)}


_ELEV = None


def station_elev(gid: str) -> float:
    """IGRA often omits the SURFACE geopotential height; it's the station elevation."""
    global _ELEV
    if _ELEV is None:
        _ELEV = {s["gid"]: s.get("e", 0) or 0
                 for s in json.loads(STATIONS.read_text())["stations"]}
    return float(_ELEV.get(gid, 0))


def _samples(gid: str) -> dict | None:
    """Per-sounding index values for a station: {key: (values, doys, years)}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{gid}.npz"
    if cf.exists():
        try:
            z = np.load(cf)
            if all(k in z for k in KEYS if k not in ("ecape", "ship")):
                return {"doy": z["doy"], "yr": z["yr"],
                        "v": {k: z[k] for k in KEYS if k in z}}
            # cache predates a newly added index — refetch to fill it
        except Exception:                                 # noqa: BLE001
            pass                                          # corrupt cache: refetch

    try:
        raw = urllib.request.urlopen(IGRA + f"{gid}-data.txt.zip", timeout=180).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        text = zf.read(zf.namelist()[0]).decode("utf-8", "ignore")
    except Exception as e:                                # noqa: BLE001
        print(f"  {gid}: fetch failed ({repr(e)[:50]})", flush=True)
        return None

    elev = station_elev(gid)
    rows: list = []                                       # (doy, year, {key: val})
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        L = lines[i]
        if L.startswith("#" + gid):
            year, month, day = int(L[13:17]), int(L[18:20]), int(L[21:23])
            nlev = int(L[32:36])
            block = lines[i + 1:i + 1 + nlev]
            i += 1 + nlev
            vals = sounding_indices(block, elev)
            if vals:
                try:
                    doy = datetime(year, month, max(1, day)).timetuple().tm_yday
                except ValueError:
                    continue
                rows.append((doy, year, vals))
            continue
        i += 1

    # ECAPE + SHIP from the native SHARPlib helper (identical physics to the app)
    cape: dict = {}
    if os.path.exists(CAPE_BIN):
        try:
            r = subprocess.run([CAPE_BIN, gid, str(elev)], input=text,
                               capture_output=True, text=True, timeout=900)
            for ln in r.stdout.splitlines():
                q = ln.split()
                if len(q) != 5:
                    continue
                cape.setdefault((int(q[0]), int(q[1]), int(q[2])), []).append(
                    (float(q[3]) if q[3] != "nan" else np.nan,
                     float(q[4]) if q[4] != "nan" else np.nan))
        except Exception as e:                            # noqa: BLE001
            print(f"  {gid}: cape helper failed ({repr(e)[:40]})", flush=True)

    if not rows:
        return None
    doy = np.array([r[0] for r in rows], dtype="int16")
    yr = np.array([r[1] for r in rows], dtype="int16")
    v = {k: np.array([r[2].get(k, np.nan) for r in rows], dtype="float32")
         for k in KEYS if k not in ("ecape", "ship")}
    # cape values are keyed by (year, month) — attach by month, in order
    for k, j in (("ecape", 0), ("ship", 1)):
        out = np.full(len(rows), np.nan, dtype="float32")
        seen: dict = {}
        for idx, (d, y, _) in enumerate(rows):
            date = datetime(int(y), 1, 1) + timedelta(days=int(d) - 1)
            key = (int(y), date.month, date.day)
            lst = cape.get(key)
            if not lst:
                continue
            c = seen.get(key, 0)
            if c < len(lst):
                out[idx] = lst[c][j]
                seen[key] = c + 1
        v[k] = out
    np.savez_compressed(CACHE / f"{gid}.npz", doy=doy, yr=yr, **v)
    return {"doy": doy, "yr": yr, "v": v}


def build_station(gid: str, outdir: Path) -> bool:
    s = _samples(gid)
    if not s:
        return False
    doy, yr, V = s["doy"], s["yr"], s["v"]

    out: dict = {"gid": gid, "doy": DOY, "idx": {}}
    for k in KEYS:
        vals = V.get(k)
        if vals is None:
            continue
        ok = np.isfinite(vals)
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
                print(f"  {gid} {k}: PHYS rejected {int(bad.sum())} "
                      f"(outside [{lo_p},{hi_p}])", flush=True)
                ok = ok & ~bad
                if ok.sum() < 20:
                    continue
        if k not in ("ecape", "ship"):
            p1, p50, p99 = np.percentile(vals[ok], [1, 50, 99])
            lo = p1 - 3.0 * max(p50 - p1, 1.0)
            hi = p99 + 3.0 * max(p99 - p50, 1.0)
            bad = ok & ((vals < lo) | (vals > hi))
            if bad.any():
                print(f"  {gid} {k}: QC rejected {int(bad.sum())} "
                      f"(kept [{lo:.0f},{hi:.0f}])", flush=True)
                ok = ok & ~bad
                if ok.sum() < 20:
                    continue
        n_l, p_l, mn_l, mx_l, mnY_l, mxY_l = [], [], [], [], [], []
        for a in DOY:
            d = np.abs(doy - a)
            d = np.minimum(d, 365 - d)                    # wrap the year
            m = ok & (d <= DOY_WIN)
            c = int(m.sum())
            n_l.append(c)
            if c < 5:
                p_l.append([None] * len(PCTS))
                mn_l.append(None); mx_l.append(None)
                mnY_l.append(None); mxY_l.append(None)
                continue
            vv, yy = vals[m], yr[m]
            # Window-local fence: the global fence above is scaled to the
            # full annual cycle, so it can't see seasonally-impossible
            # values (a winter-grade thickness in an August window). Same
            # 3-tail-span rule, but against THIS window's distribution.
            if k not in ("ecape", "ship") and c >= 20:
                q1, q50, q99 = np.percentile(vv, [1, 50, 99])
                w_lo = q1 - 3.0 * max(q50 - q1, 1.0)
                w_hi = q99 + 3.0 * max(q99 - q50, 1.0)
                keep = (vv >= w_lo) & (vv <= w_hi)
                if not keep.all():
                    vv, yy = vv[keep], yy[keep]
                    c = len(vv)
                    n_l[-1] = c
                    if c < 5:
                        p_l.append([None] * len(PCTS))
                        mn_l.append(None); mx_l.append(None)
                        mnY_l.append(None); mxY_l.append(None)
                        continue
            p_l.append([round(float(x), 1) for x in np.percentile(vv, PCTS)])
            i0, i1 = int(np.argmin(vv)), int(np.argmax(vv))
            mn_l.append(round(float(vv[i0]), 1)); mnY_l.append(int(yy[i0]))
            mx_l.append(round(float(vv[i1]), 1)); mxY_l.append(int(yy[i1]))
        out["idx"][k] = {"n": n_l, "p": p_l, "min": mn_l, "max": mx_l,
                         "minY": mnY_l, "maxY": mxY_l}
    if not out["idx"]:
        return False
    out["yr0"] = int(yr.min()); out["yr1"] = int(yr.max())
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{gid}.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"  {gid}: {len(out['idx'])} indices, {len(doy)} soundings "
          f"({out['yr0']}-{out['yr1']})", flush=True)
    return True


def main() -> int:
    outdir = Path(sys.argv[1])
    args = sys.argv[2:]
    if "--all" in args:
        stns = [s["gid"] for s in json.loads(STATIONS.read_text())["stations"]]
    else:
        stns = args
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
                if have.get("doy") and "850td" in have.get("idx", {}):
                    ok += 1; continue          # already has the split T/Td keys
            except Exception:                   # noqa: BLE001
                pass                            # rebuild on parse error
        todo.append(gid)

    # network-bound (30-90 MB per station) — fetch/compute several at once
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(build_station, g, outdir): g for g in todo}
        for f in cf.as_completed(futs):
            try:
                if f.result():
                    ok += 1
            except Exception as e:              # noqa: BLE001
                print(f"  {futs[f]}: failed ({repr(e)[:40]})", flush=True)
    print(f"built {ok}/{len(stns)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
