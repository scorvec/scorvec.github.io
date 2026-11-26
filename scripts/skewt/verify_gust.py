#!/usr/bin/env python3
"""Verify surface wind gust estimates from soundings against airport observations (runs in GitHub Actions only).

Stages (skewt-gust-verify.yml runs them in order):
    stations OUT   candidate raob sites, co-location check against the airport (<= MAX_KM, |dz| <= MAX_DZ)
    isd      OUT   NOAA Integrated Surface Database station-years from AWS S3 (noaa-global-hourly-pds, anonymous),
                   reduced to METAR/SPECI rows: sustained wind, gust, peak wind, thunder
    build    OUT   IGRA v2 POR files -> the app's profile handling (fogport.py) -> compute_sounding() (the browser's
                   SHARPlib code, built natively; SHARP_LIB) -> every method in gustport.py, matched to the truth
    score    OUT   verify_gust_score.py

Truth over a window after the release (IGRA header release time; nominal - 45 min when missing): the highest of the
reported gusts, the peak-wind remark (PK WND) and the sustained winds, WIN_A minutes from the release, WIN_B as a
check. METAR gusts are reported only when they beat the mean wind by about 10 kt, so a launch with no gust reported
carries its sustained wind (`censored`).
"""

from __future__ import annotations

import csv
import ctypes
import functools
import io
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fogport as FP
import gustport as GP
import verify_fog as VF

S3 = "https://noaa-global-hourly-pds.s3.amazonaws.com"
Y0, Y1 = 2010, 2025
MAX_KM = 5.0
MAX_DZ = 50.0
WIN_A = (0, 120)  # minutes after the release
WIN_B = (-60, 60)
MIN_REPORTS = 2  # METAR/SPECI rows a window needs
KT = 1.943844

# the fog study's sites plus windy ones (co-location still checked; WMO -> airport)
CANDIDATES = VF.CANDIDATES | {
    "72776": "KGTF",
    "72562": "KLBF",
    "72662": "KRAP",
    "72265": "KMAF",
    "72365": "KABQ",
    "72489": "KRNO",
    "72582": "KEKO",
    "72672": "KRIW",
    "72768": "KGGW",
    "72786": "KGEG",
    "72469": "KDEN",
    "72357": "KOUN",
    "72440": "KSGF",
    "72558": "KOMA",
    "72520": "KPIT",
    "72649": "KMSP",
}


# ------------------------------------------------------------- stations ---
def stations(out: Path) -> None:
    sl = VF.get(VF.IGRA + "igra2-station-list.txt").decode()
    igra = {}
    for line in sl.splitlines():
        w = line[6:11]
        if w in CANDIDATES and line[0:2] in ("US", "CA", "AR", "SP", "IT"):
            igra[w] = {"gid": line[0:11], "la": float(line[12:20]), "lo": float(line[21:30]),
                       "el": float(line[31:37]), "name": line[41:71].strip(), "y1": int(line[77:81])}  # fmt: skip
    isd = {}  # ICAO -> [(usaf+wban, la, lo, el, begin, end)]
    for row in csv.reader(io.StringIO(VF.get(VF.ISD_HIST).decode())):
        if len(row) < 11 or row[5] not in CANDIDATES.values() or not row[6] or not row[7] or row[0] == "USAF":
            continue
        isd.setdefault(row[5], []).append(
            (row[0] + row[1], float(row[6]), float(row[7]), float(row[8] or "nan"), row[9], row[10])
        )
    table = []
    rep = ["| WMO | ICAO | IGRA name | km | dz m | ISD ids | kept |", "|---|---|---|---|---|---|---|"]
    for w, icao in CANDIDATES.items():
        g, recs = igra.get(w), isd.get(icao)
        if not g or not recs:
            rep.append(f"| {w} | {icao} | missing ({'IGRA' if not g else 'ISD'}) | | | | no |")
            continue
        cur = max(recs, key=lambda r: r[5])  # the current location
        km = VF.hav_km(g["la"], g["lo"], cur[1], cur[2])
        dz = g["el"] - cur[3]
        ids = sorted({r[0] for r in recs if r[5][:4] >= str(Y0) and r[4][:4] <= str(Y1)})
        keep = km <= MAX_KM and abs(dz) <= MAX_DZ and g["y1"] >= Y0 + 3 and bool(ids)
        rep.append(
            f"| {w} | {icao} | {g['name']} | {km:.1f} | {dz:.0f} | {' '.join(ids)} | {'yes' if keep else 'no'} |"
        )
        table.append({"wmo": w, "icao": icao, **g, "km": round(km, 2), "dz": round(dz, 1), "isd": ids, "keep": keep,
                      "intl": not g["gid"].startswith("US")})  # fmt: skip
    (out / "stations.json").write_text(json.dumps(table, indent=1))
    (out / "stations.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep), flush=True)
    print(f"  kept {sum(s['keep'] for s in table)} of {len(CANDIDATES)}", flush=True)


# ------------------------------------------------------------------ ISD ---
GUST_RE = re.compile(r"\b(?:\d{3}|VRB)(\d{2,3})G(\d{2,3})(KT|MPS)\b")
PK_RE = re.compile(r"\bPK WND (\d{3})(\d{2,3})/(\d{2,4})\b")
TS_RE = re.compile(r"(?:^|\s)[-+]?(?:VC)?TS")


def _speed(field: str):
    """ISD WND 'ddd,q,t,ssss,q' -> m/s (None when missing or failed QC)."""
    p = field.split(",")
    if len(p) < 5 or p[3] == "9999" or p[4] not in ("0", "1", "4", "5", "9", "A", "C", "I", "M", "P", "R", "U"):
        return None
    return int(p[3]) / 10


def parse_isd(text: str):
    """ISD global-hourly CSV -> rows (minute since epoch, sustained, gust, (peak, its minute) or None, thunder),
    METAR (FM-15) and SPECI (FM-16) only; speeds in m/s."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if r.get("REPORT_TYPE", "").strip() not in ("FM-15", "FM-16"):
            continue
        t = datetime.fromisoformat(r["DATE"]).replace(tzinfo=UTC)
        tm = int(t.timestamp() // 60)
        sust = _speed(r.get("WND", ""))
        gust = None
        oc = r.get("OC1", "").split(",")
        if len(oc) == 2 and oc[0] not in ("", "9999") and oc[1] in ("0", "1", "4", "5", "9"):
            gust = int(oc[0]) / 10
        rem = r.get("REM", "")
        metar, _, rmk = rem.partition(" RMK ")
        if m := GUST_RE.search(metar):
            g = int(m.group(2)) / (KT if m.group(3) == "KT" else 1)
            gust = g if gust is None else max(gust, g)
        pk = None
        if m := PK_RE.search(rmk):
            spd, when = int(m.group(2)) / KT, m.group(3)
            if len(when) == 4:
                pt = t.replace(hour=int(when[:2]), minute=int(when[2:]))
                if pt > t:
                    pt -= timedelta(days=1)
            else:
                pt = t.replace(minute=int(when))
                if pt > t:
                    pt -= timedelta(hours=1)
            pk = (spd, int(pt.timestamp() // 60))
        mw = r.get("MW1", "")[:2]
        ts = bool(TS_RE.search(metar)) or mw in ("17", "29") or (mw.isdigit() and 91 <= int(mw) <= 99)
        rows.append((tm, sust, gust, pk, ts))
    return rows


def _fetch_year(args):
    out, icao, sid, year = args
    f = out / "isd" / icao / f"{year}_{sid}.csv"
    if f.exists():
        return f, True
    try:
        rows = parse_isd(VF.get(f"{S3}/{year}/{sid}.csv", timeout=180, tries=2).decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001  (a station-year the bucket lacks)
        return f, False
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("w") as fh:
        for tm, sust, gust, pk, ts in rows:
            pks = "" if pk is None else f"{pk[0]:.1f}"
            pkt = "" if pk is None else pk[1]
            fh.write(
                f"{tm},{'' if sust is None else sust},{'' if gust is None else f'{gust:.1f}'},{pks},{pkt},{int(ts)}\n"
            )
    return f, True


def isd(out: Path) -> None:
    st = [s for s in json.loads((out / "stations.json").read_text()) if s["keep"]]
    jobs = [(out, s["icao"], sid, y) for s in st for sid in s["isd"] for y in range(Y0, Y1 + 1)]
    ok = 0
    with ThreadPoolExecutor(8) as ex:
        for _, got in ex.map(_fetch_year, jobs):
            ok += got
    print(f"  ISD: {ok} of {len(jobs)} station-years", flush=True)


def load_isd(out: Path, icao: str):
    """-> sorted list of (minute, sustained, gust, peak, peak minute, thunder)."""
    rows = []
    for f in sorted((out / "isd" / icao).glob("*.csv")):
        for line in f.read_text().splitlines():
            a = line.split(",")
            rows.append(
                (int(a[0]), *(float(x) if x else None for x in a[1:4]), int(a[4]) if a[4] else None, a[5] == "1")
            )
    rows.sort(key=lambda r: r[0])  # by time alone: two reports in one minute may carry None
    return rows


def truth(rows, times, t0: int, win):
    """Max sustained / gust / peak and thunder over [t0 + win[0], t0 + win[1]] minutes; None with < MIN_REPORTS."""
    import bisect

    a, b = t0 + win[0], t0 + win[1]
    i, j = bisect.bisect_left(times, a), bisect.bisect_right(times, b)
    sel = rows[i:j]
    if len(sel) < MIN_REPORTS:
        return None
    sust = [r[1] for r in sel if r[1] is not None]
    gust = [r[2] for r in sel if r[2] is not None]
    # a peak wind belongs to the window when its own time does (it is reported up to an hour later)
    k = bisect.bisect_right(times, b + 60)
    pk = [r[3] for r in rows[i:k] if r[3] is not None and r[4] is not None and a <= r[4] <= b]
    if not sust:
        return None
    vals = sust + gust + pk
    return {"n": len(sel), "sust": max(sust), "gust": max(gust) if gust else None, "pk": max(pk) if pk else None,
            "peak": max(vals), "censored": not gust and not pk, "ts": any(r[5] for r in sel)}  # fmt: skip


# ---------------------------------------------------------------- build ---
@functools.cache
def _lib():
    return ctypes.CDLL(os.environ["SHARP_LIB"])


def sharp(prof):
    """compute_sounding() through ctypes -> (pbl_top Pa, dcape, sbcape), None where missing."""
    lib = _lib()
    n = len(prof["P"])
    arr = [(ctypes.c_float * n)(*prof[k]) for k in "PHTDUV"]
    nout = lib.out_size()
    out = (ctypes.c_float * nout)()
    tr = [(ctypes.c_float * n)() for _ in range(3)]
    lib.compute_sounding(*arr, n, out, *tr)

    def val(i):
        v = out[i]
        return None if v == -9999 or abs(v) >= 1e7 or not math.isfinite(v) else float(v)

    return val(46), val(39), val(0)


def process_station(args):
    s, out = args
    import numpy as np
    import pandas as pd

    rows = load_isd(out, s["icao"])
    times = [r[0] for r in rows]
    recs, profs = [], []
    qc = {"records": 0, "parse_fail": 0, "no_sfc": 0, "short": 0, "no_truth": 0, "kept": 0}
    for rec in VF.iter_records(out / "igra" / f"{s['gid']}-data.txt.zip"):
        qc["records"] += 1
        h = rec[0]
        ymd, hour, rel = f"{h[13:17]}-{h[18:20]}-{h[21:23]}", int(h[24:26]), h[27:31]
        got = FP.parse_igra_record(rec, s["el"])
        if got is None or got[1]["nan_p"]:
            qc["parse_fail"] += 1
            continue
        prof, info = got
        if not info["sfc_flag"] or not FP.isf(info["raw_t0"]):
            qc["no_sfc"] += 1
            continue
        p = FP.prepare(prof)
        if len(p["P"]) < 8 or p["H"][-1] - p["H"][0] < 3000:
            qc["short"] += 1
            continue
        nom = datetime.strptime(ymd, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(hours=hour)
        if rel.isdigit() and int(rel[:2]) < 24 and int(rel[2:]) < 60:
            rt = nom.replace(hour=int(rel[:2]), minute=int(rel[2:]))
            if rt - nom > timedelta(hours=12):
                rt -= timedelta(days=1)
            elif nom - rt > timedelta(hours=12):
                rt += timedelta(days=1)
        else:
            rt = nom - timedelta(minutes=45)
        t0 = int(rt.timestamp() // 60)
        ta, tb = truth(rows, times, t0, WIN_A), truth(rows, times, t0, WIN_B)
        if ta is None:
            qc["no_truth"] += 1
            continue
        pbl, dcape, sbcape = sharp(p)
        f = GP.features(p, pbl, dcape, sbcape)
        qc["kept"] += 1
        recs.append({"wmo": s["wmo"], "icao": s["icao"], "date": ymd, "hour": hour, "year": int(ymd[:4]),
                     "month": int(ymd[5:7]), "rel_min": t0 - int(nom.timestamp() // 60), "elev": s["el"],
                     **{f"a_{k}": v for k, v in ta.items()},
                     **({f"b_{k}": v for k, v in tb.items()} if tb else {}), **f})  # fmt: skip
        z0 = p["H"][0]
        k = sum(1 for hh in p["H"] if hh - z0 <= GP.GRID_TOP)
        profs.append(np.array([p[x][:k] for x in "PHTDUV"], dtype=np.float32))
    np.savez_compressed(
        out / "profiles" / f"{s['wmo']}.npz",
        offsets=np.cumsum([0] + [a.shape[1] for a in profs]),
        data=np.concatenate(profs, axis=1) if profs else np.zeros((6, 0), np.float32),
        key=np.array([f"{r['date']}_{r['hour']:02d}" for r in recs]),
    )
    return s["wmo"], pd.DataFrame(recs), qc


def build(out: Path) -> None:
    import pandas as pd
    from multiprocessing import Pool

    st = [
        s for s in json.loads((out / "stations.json").read_text()) if s["keep"] and (out / "isd" / s["icao"]).exists()
    ]
    (out / "igra").mkdir(exist_ok=True)
    (out / "profiles").mkdir(exist_ok=True)
    for s in st:
        f = out / "igra" / f"{s['gid']}-data.txt.zip"
        if not f.exists():
            f.write_bytes(VF.get(VF.IGRA + f"data/data-por/{s['gid']}-data.txt.zip"))
            print(f"  IGRA {s['gid']}: {f.stat().st_size >> 20} MB", flush=True)
    frames, qcs = [], {}
    with Pool(os.cpu_count() or 2) as pool:
        for wmo, df, qc in pool.imap_unordered(process_station, [(s, out) for s in st]):
            print(f"  {wmo}: {qc}", flush=True)
            frames.append(df)
            qcs[wmo] = qc
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    df.to_parquet(out / "pairs.parquet", index=False)
    (out / "qc.json").write_text(json.dumps(qcs, indent=1))
    lines = [
        "| WMO | ICAO | launches | kept | gust reported | thunder | peak >= 39 kt |",
        "|---|---|---|---|---|---|---|",
    ]
    for w, g in df.groupby("wmo"):
        lines.append(f"| {w} | {g.icao.iat[0]} | {qcs[w]['records']} | {len(g)} | {(~g.a_censored).mean():.0%} | "
                     f"{g.a_ts.mean():.1%} | {(g.a_peak * KT >= 39).sum()} |")  # fmt: skip
    (out / "dataset.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"  pairs: {len(df)}", flush=True)


def main() -> int:
    cmd, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if cmd == "stations":
        stations(out)
    elif cmd == "isd":
        isd(out)
    elif cmd == "build":
        build(out)
    elif cmd == "score":
        import verify_gust_score

        verify_gust_score.score(out)
    else:
        raise SystemExit(f"unknown command {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
