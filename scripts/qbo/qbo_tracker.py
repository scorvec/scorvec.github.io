#!/usr/bin/env python3
"""QBO tracker: monthly-mean equatorial stratospheric zonal wind from IGRA.

Near the equator the annual cycle in stratospheric zonal wind nearly
vanishes, so the multi-station monthly mean at 10-100 hPa IS the QBO.

    python scripts/qbo/qbo_tracker.py full     # rebuild from period-of-record
    python scripts/qbo/qbo_tracker.py update   # refresh recent months from y2d

Writes qbo/qbo.json: levels, months, u[level][month] (m/s, null=missing),
per-station latest readings.
"""
from __future__ import annotations

import io
import json
import math
import sys
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "qbo" / "qbo.json"
IGRA = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/"
LEVELS = [10, 15, 20, 30, 50, 70, 100]          # hPa
MIN_N = 3                                       # soundings per station-month
# |lat| <= 10 deg — annual cycle negligible; spread in longitude
STATIONS = {
    "48698": "SNM00048698",   # Singapore (WMO reference record)
    "48650": "MYM00048650",   # Kuala Lumpur
    "96441": "MYM00096441",   # Bintulu
    "91334": "FMM00091334",   # Chuuk
    "91348": "FMM00091348",   # Pohnpei
    "91413": "FMM00091413",   # Yap
    "82022": "BRM00082022",   # Boa Vista
    "82099": "BRM00082099",   # Macapa
    "82107": "BRM00082107",   # Sao Gabriel da Cachoeira
    "82244": "BRM00082244",   # Santarem
    "82332": "BRM00082332",   # Manaus
    "82411": "BRM00082411",   # Tabatinga
    "63741": "KEM00063741",   # Nairobi
}


def iv(line, a, b):
    try:
        v = int(line[a:b].replace(" ", "").rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ") or "-9999")
    except ValueError:
        return None
    return None if v <= -8888 else v


def sounding_u(levlines):
    """target-level u (m/s) dict from one sounding's stratospheric lines."""
    pts = []                                    # (ln p_hPa, u)
    for line in levlines:
        p = iv(line, 9, 15)
        if p is None or p > 15000 or p < 500:
            continue
        wd, ws = iv(line, 40, 45), iv(line, 46, 51)
        if wd is None or ws is None:
            continue
        spd = ws / 10.0
        pts.append((math.log(p / 100.0), -spd * math.sin(math.radians(wd))))
    if not pts:
        return {}
    pts.sort(reverse=True)                      # high p (low alt) first
    out = {}
    for tgt in LEVELS:
        lt = math.log(tgt)
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            if b[0] <= lt <= a[0]:
                if a[0] - b[0] < 0.5:           # bracket tight enough (~40%)
                    f = (a[0] - lt) / max(1e-9, a[0] - b[0])
                    out[tgt] = a[1] + f * (b[1] - a[1])
                break
        else:
            if pts and abs(pts[-1][0] - lt) < 0.03:
                out[tgt] = pts[-1][1]
    return out


def station_months(gid, url, y0=1950):
    """{(y,m): {level: [u...]}} accumulated per sounding."""
    acc = {}
    raw = urlopen(url, timeout=600).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            hdr, lines = None, []
            def flush():
                if hdr is None or not lines:
                    return
                us = sounding_u(lines)
                if us:
                    slot = acc.setdefault(hdr, {})
                    for k, v in us.items():
                        slot.setdefault(k, []).append(v)
            for braw in f:
                line = braw.decode("ascii", "replace")
                if line.startswith("#"):
                    flush()
                    hdr, lines = None, []
                    try:
                        y, mo = int(line[13:17]), int(line[18:20])
                    except ValueError:
                        continue
                    if y >= y0:
                        hdr = (y, mo)
                elif hdr is not None:
                    lines.append(line)
            flush()
    return acc


def month_key(y, m):
    return f"{y:04d}-{m:02d}"


def build(mode):
    today = date.today()
    if mode == "update" and OUT.exists():
        prior = json.loads(OUT.read_text())
        y0 = today.year - 1
        keep = {m: i for i, m in enumerate(prior["months"]) if m < f"{y0:04d}-01"}
    else:
        prior, keep, y0 = None, {}, 1950

    # per-station monthly means
    per_stn = {}
    latest = {}
    for wmo, gid in STATIONS.items():
        if mode == "update":
            # NCEI periodically rolls the y2d "beginning year" label (beg2025 ->
            # beg2026 in mid-2026, deleting the old name) — try current year,
            # then previous, then the full period-of-record file.
            urls = [IGRA + f"data-y2d/{gid}-data-beg{y}.txt.zip"
                    for y in (today.year, today.year - 1)]
            urls.append(IGRA + f"data-por/{gid}-data.txt.zip")
        else:
            urls = [IGRA + f"data-por/{gid}-data.txt.zip"]
        acc, err = None, None
        for url in urls:
            try:
                acc = station_months(gid, url, y0)
                break
            except Exception as e:                            # noqa: BLE001
                err = e
        if acc is None:
            print(f"  {wmo}: FAILED {err}", flush=True)
            continue
        means = {}
        for (y, m), levs in acc.items():
            means[month_key(y, m)] = {
                k: sum(v) / len(v) for k, v in levs.items() if len(v) >= MIN_N}
        per_stn[wmo] = means
        if means:
            last = max(means)
            latest[wmo] = {"month": last,
                           "u": {str(k): round(v, 1) for k, v in means[last].items()}}
        print(f"  {wmo}: {len(means)} station-months (latest {max(means, default='—')})",
              flush=True)

    if not per_stn:
        print("ERROR: zero stations fetched — refusing to overwrite qbo.json")
        return 1
    newest = max(max(m) for m in per_stn.values() if m)
    if prior and prior["months"] and newest < prior["months"][-1]:
        print(f"ERROR: fetched record ends {newest}, older than published "
              f"{prior['months'][-1]} — refusing to regress")
        return 1

    # cross-station mean per month/level
    all_months = sorted({m for s in per_stn.values() for m in s})
    if prior:
        all_months = sorted(set(list(keep) + [m for m in all_months if m >= f"{y0:04d}-01"]))
    u = [[None] * len(all_months) for _ in LEVELS]
    n_st = [[0] * len(all_months) for _ in LEVELS]
    for li, lev in enumerate(LEVELS):
        for mi, mon in enumerate(all_months):
            if prior and mon in keep:            # keep the archived value
                pi = keep[mon]
                u[li][mi] = prior["u"][li][pi]
                n_st[li][mi] = prior["n"][li][pi]
                continue
            vals = [s[mon][lev] for s in per_stn.values() if mon in s and lev in s[mon]]
            if vals:
                u[li][mi] = round(sum(vals) / len(vals), 1)
                n_st[li][mi] = len(vals)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "updated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "levels": LEVELS, "months": all_months, "u": u, "n": n_st,
        "stations": {w: STATIONS[w] for w in per_stn},
        "latest": latest,
    }, separators=(",", ":")))
    print(f"wrote {OUT} ({len(all_months)} months)", flush=True)
    return 0


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "update")
