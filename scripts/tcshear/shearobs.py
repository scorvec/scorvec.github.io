#!/usr/bin/env python3
"""Observations and reference data for the TC shear product.

  NHC active storms        https://www.nhc.noaa.gov/CurrentStorms.json (names, ids, positions; AL / EP / CP)
  best track / NHC track   the recon mirror (recon-data branch: storms/<SID>.json, from NHC's ATCF b-deck and
                           forecast advisories), else NHC's ATCF b-deck (ftp.nhc.noaa.gov/atcf/btk)
  vortex-message fixes     the recon mirror's missions (flight-level centre fixes, NHC/CARCAH vortex data messages)
  TDR vortex tilt          the tdr-data branch: NOAA AOML/HRD real-time tail-Doppler-radar centres at 0.5/2/3/6 km and
                           HRD's 2-6 km tilt (hrd/reasor/tdr_centers), screened against the fixes by tdr_mirror.py
  SHIPS                    NHC ATCF real-time SHIPS text (ftp.nhc.noaa.gov/atcf/stext/<yymmddhh><BB><nn><yy>_ships.txt):
                           SHEAR (KT) = 850-200 hPa shear, vortex removed, 0-500 km (SHDC), from the GFS; SHEAR DIR
All public US-government data (NOAA); cited on the page.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path

import numpy as np

import sheardata as D
import shearlib as S

RAW = "https://raw.githubusercontent.com/scorvec/scorvec.github.io"
NHC = "https://www.nhc.noaa.gov/CurrentStorms.json"
STEXT = "https://ftp.nhc.noaa.gov/atcf/stext"
BTK = "https://ftp.nhc.noaa.gov/atcf/btk"
BASIN_OF = {"L": "AL", "E": "EP", "C": "CP", "W": "WP", "A": "IO", "B": "IO", "S": "SH", "P": "SH", "U": "SH"}


def _json(url):
    b = D.http_get(url, tries=2, src="obs")
    return json.loads(b) if b else None


def nhc_active():
    try:
        d = _json(NHC) or {}
    except Exception:                                                  # noqa: BLE001
        return []
    return d.get("activeStorms", [])


def match_sid(s, active, init):
    """ATCF-style id for a tracker storm ('09L' -> AL092026); the NHC id when NHC carries the same number."""
    tid = s["tid"]
    m = re.match(r"^(\d{2})([A-Z])$", tid)
    year = init.year
    if not m:
        return tid, "XX"
    num, b = m.group(1), m.group(2)
    basin = BASIN_OF.get(b, "XX")
    for a in active:
        aid = str(a.get("id", "")).upper()
        if aid[:2] == basin and aid[2:4] == num:
            return aid, basin
    return f"{basin}{num}{year}", basin


def observed_position(s, active):
    """(lat, lon) of the storm now: NHC's current position, else the tracker's t=0 observed position (any member)."""
    for a in active:
        if str(a.get("id", "")).upper() == s.get("sid"):
            try:
                return float(a["latitudeNumeric"]), float(a["longitudeNumeric"])
            except Exception:                                          # noqa: BLE001
                pass
    for t in list(s["tracks"].values()) + ([s["seed"]] if s.get("seed") else []):
        if 0 in list(t["steps"]):
            return float(t["lat"][0]), float(t["lon"][0])
    return None


def _btk_deck(sid):
    """NHC ATCF b-deck -> [[t, lat, lon, vmax kt, pmin hPa, type]] (6-hourly synoptic rows, 34-kt line only)."""
    b = D.http_get(f"{BTK}/b{sid[:2].lower()}{sid[2:4]}{sid[4:]}.dat", tries=2, src="obs")
    if not b:
        return []
    out, seen = [], set()
    for line in b.decode(errors="replace").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 11 or p[2] in seen:
            continue
        try:
            t = int(dt.datetime.strptime(p[2], "%Y%m%d%H").replace(tzinfo=dt.timezone.utc).timestamp())
            la = float(p[6][:-1]) / 10 * (1 if p[6][-1] == "N" else -1)
            lo = float(p[7][:-1]) / 10 * (-1 if p[7][-1] == "W" else 1)
            out.append([t, la, lo, int(p[8]), int(p[9]) or None, p[10]]); seen.add(p[2])
        except Exception:                                              # noqa: BLE001
            continue
    return out


def observations(sid, init):
    """Everything observed about the storm up to now (descriptive, no claims)."""
    obs = dict(btk=[], fcst=None, vdm=[], tdr=[], ships=None, src=[])
    try:
        st = _json(f"{RAW}/recon-data/storms/{sid}.json")
    except Exception:                                                  # noqa: BLE001
        st = None
    if st:
        obs["btk"] = st.get("btk", []); obs["fcst"] = st.get("fcst"); obs["src"].append("recon mirror (NHC ATCF)")
    elif sid[:2] in ("AL", "EP", "CP"):
        obs["btk"] = _btk_deck(sid)
        if obs["btk"]:
            obs["src"].append("NHC ATCF b-deck")
    if sid.startswith("AL"):
        try:
            man = _json(f"{RAW}/recon-data/manifest.json") or {}
            for k, m in (man.get("missions") or {}).items():
                if m.get("storm") == sid:
                    for f in m.get("fixes") or []:
                        obs["vdm"].append([int(f[0]), f[1], f[2], f[3], k])
            obs["vdm"].sort()
        except Exception as e:                                         # noqa: BLE001
            print(f"  vdm: {e}")
        try:
            idx = _json(f"{RAW}/tdr-data/index.json") or {}
            for k, m in (idx.get("missions") or {}).items():
                if m.get("storm") != sid or not m.get("n_ctr"):
                    continue
                fl = _json(f"{RAW}/tdr-data/{m['file']}") or {}
                for c in fl.get("centres") or []:
                    cc = c.get("c") or {}
                    if "tilt" not in c or "2" not in cc or "6" not in cc:
                        continue
                    obs["tdr"].append([int(c["t"]), c["tilt"][0], c["tilt"][1] % 360, cc["2"][0], cc["2"][1],
                                       cc["6"][0], cc["6"][1], fl.get("id"), fl.get("aircraft")])
            obs["tdr"].sort()
            if obs["tdr"]:
                obs["src"].append("NOAA AOML/HRD real-time TDR centres (2-6 km tilt)")
        except Exception as e:                                         # noqa: BLE001
            print(f"  tdr: {e}")
    obs["ships"] = ships(sid, init)
    if obs["ships"]:
        obs["src"].append("NHC ATCF real-time SHIPS (GFS)")
    return obs


def parse_ships(txt):
    """SHIPS text -> {t0, hours, shear kt, shear_dir, sst, pot_int kt, lat, lon, rh} (None for N/A)."""
    rows = {}
    for line in txt.splitlines():
        for key, lab in (("hours", "TIME (HR)"), ("shear", "SHEAR (KT)"), ("shear_dir", "SHEAR DIR"),
                         ("sst", "SST (C)"), ("pot_int", "POT. INT. (KT)"), ("lat", "LAT (DEG N)"),
                         ("lon", "LONG(DEG W)"), ("rh", "700-500 MB RH"), ("vmax", "V (KT) NO LAND")):
            if line.startswith(lab) and key not in rows:
                vals = []
                for x in line[len(lab):].split():
                    try:
                        vals.append(float(x))
                    except ValueError:
                        vals.append(None)
                rows[key] = vals
    m = re.search(r"(\d{2})/(\d{2})/(\d{2})\s+(\d{2}) UTC", txt)
    if not m or "hours" not in rows:
        return None
    t0 = dt.datetime(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)), int(m.group(4)), tzinfo=dt.timezone.utc)
    rows["lon"] = [(-x if x is not None else None) for x in rows.get("lon", [])]
    rows["t0"] = int(t0.timestamp())
    return rows


def ships(sid, init):
    """The SHIPS runs for this storm from init-12 h to init+6 h (whatever exists). -> {cycle 'YYMMDDHH': rows}"""
    out = {}
    for dh in (-12, -6, 0, 6):
        c = init + dt.timedelta(hours=dh)
        name = f"{c:%y%m%d%H}{sid[:2]}{sid[2:4]}{sid[6:8]}_ships.txt"
        try:
            b = D.http_get(f"{STEXT}/{name}", tries=2, src="obs")
        except Exception:                                              # noqa: BLE001
            b = None
        if b:
            r = parse_ships(b.decode(errors="replace"))
            if r:
                out[f"{c:%y%m%d%H}"] = r
    return out or None


# ── running verification archive ──────────────────────────────────────────────────────────────────────────────────
def update_verify(path: Path, sid, name, init, models, obs, keep=4000):
    """Append this cycle's analysis-time (h = 0) values per model against SHIPS's t = 0 shear of the same cycle and the
    TDR tilt nearest in time (within 3 h). One row per (cycle, storm, model)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(path.read_text()) if path.exists() else {"v": 1, "rows": []}
    t_init = int(init.replace(tzinfo=dt.timezone.utc).timestamp())
    sh = (obs.get("ships") or {}).get(f"{init:%y%m%d%H}")
    tdr = None
    for r in obs.get("tdr") or []:
        if abs(r[0] - t_init) <= 3 * 3600 and (tdr is None or abs(r[0] - t_init) < abs(tdr[0] - t_init)):
            tdr = r
    rows = [r for r in doc["rows"] if not (r["t"] == t_init and r["sid"] == sid)]
    for key, m in models.items():
        rec = None
        if key == "aifs-ens":
            rr = (m.get("members") or {}).get("0") or []
        else:
            rr = m.get("run") or []
        rec = next((r for r in rr if r.get("h") == 0), None)
        if not rec or "deep" not in rec:
            continue
        rows.append(dict(t=t_init, sid=sid, name=name, model=key, deep=rec["deep"], shrd=rec.get("shrd"), gs=rec.get("gs"),
                         tilt=rec.get("tilt"), lat=rec["lat"], lon=rec["lon"],
                         ships=[sh["shear"][0], sh["shear_dir"][0]] if sh and sh.get("shear") else None,
                         tdr=[tdr[0], tdr[1], tdr[2]] if tdr else None))
    doc["rows"] = sorted(rows, key=lambda r: (r["t"], r["sid"], r["model"]))[-keep:]
    path.write_text(json.dumps(doc, separators=(",", ":")))
