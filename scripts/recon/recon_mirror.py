#!/usr/bin/env python3
"""Mirror the NHC Atlantic reconnaissance archive to the `recon-data` branch for recon.html.

nhc.noaa.gov sends no CORS header, so the browser cannot read it; this job (recon.yml, GitHub Actions)
reads it and publishes decoded JSON:

    manifest.json                    storms, missions (summary + stats), active storms, newest plan of the day
    missions/<year>/<key>.json       one mission: HDOB track (columnar), dropsondes, vortex messages, RECCOs
    storms/<SID>.json                ATCF best track (b-deck) + the newest official forecast (.fst)
    state.json                       what has been seen (directory mtimes, file names) - the change detector

Live mode (default, every ~15 min from the laptop dispatcher):
  1. GET /archive/recon/<year>/ (~2 KB): product directory mtimes. Unchanged + CurrentStorms unchanged
     -> exit in a second with changed=false (nothing is pushed).
  2. For each changed Atlantic product directory: filtered listings (?P=*.<YYYYMMDD>*) for the days since the
     last run (today and yesterday normally), GET only file names not seen before.
  3. Decode (recon_decode.py), merge into the mission files, assign missions to storms, rebuild the manifest.
Backfill mode (--backfill "2024 AL14", dispatch only): every Atlantic product file in that storm's b-deck
window, kept when the mission belongs to the storm; the storm is pinned (kept across season pruning).

Retention: the current season's missions plus pinned (backfilled) storms; older live missions are pruned.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recon_decode as D  # noqa: E402

UTC = timezone.utc
UA = "scorvec.com recon mirror (+https://scorvec.com/recon.html)"
NHC = "https://www.nhc.noaa.gov"
ATCF = "https://ftp.nhc.noaa.gov/atcf"
PRODUCTS = ["AHONT1", "REPNT1", "REPNT2", "REPNT3", "REPRPD"]     # Atlantic (REPNT0 = non-tasked weather recon)
HCOLS = ["t", "la", "lo", "p", "z", "x", "dv", "t_", "td", "wd", "ws", "wp", "sfmr", "rr", "qc"]
NOW = datetime.now(UTC)


def log(*a):
    print(*a, flush=True)


def get(url, binary=False, tries=3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as r:
                b = r.read()
                return b if binary else b.decode("latin-1")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:  # noqa: BLE001 - network: retry
            last = e
        time.sleep(1.5 * (k + 1))
    log(f"  ! GET failed {url}: {last}")
    return None


LIST_RX = re.compile(r'href="([A-Za-z0-9_.-]+\.txt)".*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+(\S+)', re.S)
DIR_RX = re.compile(r'href="([A-Z0-9]+)/".*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})')


def parse_listing(html):
    out = {}
    if not html:
        return out
    for row in html.split("<tr>"):
        m = LIST_RX.search(row)
        if m:
            out[m.group(1)] = f"{m.group(2)}|{m.group(3)}"
    return out


def parse_dirs(html):
    out = {}
    for row in (html or "").split("<tr>"):
        m = DIR_RX.search(row)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def jload(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return default


def jdump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    if p.exists() and p.read_text() == s:
        return False
    p.write_text(s)
    return True


# ─────────────────────────────────────────────────────────── ATCF ──
def parse_atcf(text, best=True):
    """b-deck / fst lines -> list of [t, la, lo, vmax, mslp, type, name] (one per synoptic time / tau)."""
    rows, seen = [], set()
    for ln in (text or "").splitlines():
        c = [x.strip() for x in ln.split(",")]
        if len(c) < 11:
            continue
        try:
            t0 = datetime.strptime(c[2], "%Y%m%d%H").replace(tzinfo=UTC)
            tau = int(c[5] or 0)
            la = int(c[6][:-1]) / 10.0 * (-1 if c[6][-1] == "S" else 1)
            lo = int(c[7][:-1]) / 10.0 * (-1 if c[7][-1] == "W" else 1)
        except (ValueError, IndexError):
            continue
        t = int((t0 + timedelta(hours=tau)).timestamp())
        if (t, tau) in seen:
            continue
        seen.add((t, tau))
        vmax = int(c[8]) if c[8].strip().isdigit() else None
        mslp = int(c[9]) if len(c) > 9 and c[9].strip().isdigit() and int(c[9]) > 800 else None
        typ = c[10] if len(c) > 10 else ""
        name = c[27] if best and len(c) > 27 else ""
        rows.append([t, round(la, 1), round(lo, 1), vmax, mslp, typ, name, tau, int(t0.timestamp())])
    rows.sort()
    return rows


def storm_name(rows):
    for r in reversed(rows):
        n = (r[6] or "").strip()
        if n and n != "INVEST" and not n.startswith("GENESIS"):
            return n.title()
    return ""


# ──────────────────────────────────────────────────────── geometry ──
def km(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = p2 - p1, math.radians(d - b)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def track_at(rows, t):
    """b-deck position interpolated at epoch t (None outside the track +- 12 h)."""
    if not rows or t < rows[0][0] - 43200 or t > rows[-1][0] + 43200:
        return None
    if t <= rows[0][0]:
        return rows[0][1], rows[0][2]
    if t >= rows[-1][0]:
        return rows[-1][1], rows[-1][2]
    for a, b in zip(rows, rows[1:]):
        if a[0] <= t <= b[0]:
            f = (t - a[0]) / max(1, b[0] - a[0])
            return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])
    return None


# ──────────────────────────────────────────────────────── missions ──
def mission_key(rec, ft):
    ac = rec.get("aircraft") or "UNK"
    mi = rec.get("mission") or "UNK"
    return f"{ft.year}_{ac}_{mi}"


def empty_mission(rec, key, year):
    return dict(key=key, year=year, aircraft=rec.get("aircraft"), type=D.aircraft_type(rec.get("aircraft")),
                mission=rec.get("mission"), name=rec.get("name") or "", tag=rec.get("tag"),
                storm_no=rec.get("storm_no"), hdob={"cols": HCOLS, "rows": []}, drops=[], vdms=[], reccos=[],
                files=[], flagged=0, screened=0)


def screen_track(rows):
    """Drop positions implying > 450 kt between consecutive good positions (the aircraft cannot do that)."""
    it, la, lo = 0, 1, 2
    good = None
    bad = 0
    for r in rows:
        if r[la] is None:
            continue
        if good is not None:
            dt = r[it] - good[it]
            if dt > 0:
                kt = km(good[la], good[lo], r[la], r[lo]) / 1.852 / (dt / 3600.0)
                if kt > 450:
                    r[la] = r[lo] = None
                    bad += 1
                    continue
        good = r
    return bad


def mission_stats(m):
    rows = m["hdob"]["rows"]
    ix = {c: i for i, c in enumerate(HCOLS)}

    def best(col, fn=max):
        cand = [r for r in rows if r[ix[col]] is not None and r[ix["la"]] is not None]
        if not cand:
            return None
        r = fn(cand, key=lambda r: r[ix[col]])
        return dict(v=r[ix[col]], t=r[ix["t"]], la=r[ix["la"]], lo=r[ix["lo"]])

    st = dict(n_hdob=len(rows), n_drop=len(m["drops"]), n_vdm=len(m["vdms"]), n_recco=len(m["reccos"]),
              max_sfmr=best("sfmr"), max_fl=best("ws"), max_peak=best("wp"), min_x=best("x", min))
    ts = [r[0] for r in rows] + [d["spl_t"] or d["rel_t"] for d in m["drops"] if (d.get("spl_t") or d.get("rel_t"))] + \
         [v["t"] for v in m["vdms"]] + [r["t"] for r in m["reccos"]]
    st["first"], st["last"] = (min(ts), max(ts)) if ts else (None, None)
    st["last_hdob"] = rows[-1][0] if rows else None
    pos = [(r[ix["la"]], r[ix["lo"]]) for r in rows if r[ix["la"]] is not None]
    pos += [(d["spg"]["la"], d["spg"]["lo"]) for d in m["drops"] if d.get("spg")]
    if pos:
        las, los = sorted(p[0] for p in pos), sorted(p[1] for p in pos)
        st["med"] = [las[len(las) // 2], los[len(los) // 2]]
        st["bbox"] = [las[0], los[0], las[-1], los[-1]]
    st["vdm_sid"] = next((v.get("sid") for v in m["vdms"] if v.get("sid")), None)
    if m["vdms"]:
        v = min((v for v in m["vdms"] if v.get("mslp")), key=lambda v: v["mslp"], default=None)
        if v:
            st["min_vdm"] = dict(v=v["mslp"], t=v["t"], la=v["la"], lo=v["lo"])
    alts = [r[ix["z"]] for r in rows if r[ix["z"]] is not None]
    if alts:
        alts.sort()
        st["alt_med"] = alts[len(alts) // 2]
    return st


def merge(store, prod, name, rec, ft):
    key = mission_key(rec, ft)
    m = store.get(key)
    if m is None:
        m = store[key] = empty_mission(rec, key, ft.year)
    for k in ("name",):
        if rec.get(k) and not m.get(k):
            m[k] = rec[k]
    if name not in m["files"]:
        m["files"].append(name)
    kind = rec["kind"]
    if kind == "hdob":
        have = {r[0]: i for i, r in enumerate(m["hdob"]["rows"])}
        m["flagged"] += rec["rej"]["flagged"]
        m["screened"] += rec["rej"]["range"]
        for o in rec["obs"]:
            row = [o.get(c) for c in HCOLS]
            if o["t"] in have:
                m["hdob"]["rows"][have[o["t"]]] = row
            else:
                m["hdob"]["rows"].append(row)
        m["hdob"]["rows"].sort(key=lambda r: r[0])
    elif kind == "drop":
        dk = f'{rec.get("obnum")}|{(rec.get("rel") or {}).get("hms") or rec.get("launch_hhmm")}'
        old = next((d for d in m["drops"] if d["dk"] == dk), None)
        if old is not None:
            # a complete re-send / correction replaces; a part that arrives on its own is decoded with the
            # parts already held (a later copy of the same part wins inside decode_drop)
            raw = rec["raw"] if set(rec["parts"]) >= set(old["parts"]) else old["raw"] + "\n" + rec["raw"]
            nr = D.decode_drop(raw, ft) or rec
            nr["raw"] = raw
            rec = nr
            m["drops"].remove(old)
        rec["dk"] = dk
        rec["file"] = name
        m["drops"].append(rec)
        m["drops"].sort(key=lambda d: d.get("rel_t") or d.get("spl_t") or 0)
    elif kind == "vdm":
        vk = (rec.get("obnum"), rec["t"])
        m["vdms"] = [v for v in m["vdms"] if (v.get("obnum"), v["t"]) != vk] + [dict(rec, file=name)]
        m["vdms"].sort(key=lambda v: v["t"])
    elif kind == "recco":
        rk = (rec.get("obnum"), rec["t"])
        m["reccos"] = [r for r in m["reccos"] if (r.get("obnum"), r["t"]) != rk] + [rec]
        m["reccos"].sort(key=lambda r: r["t"])
    return key


# ──────────────────────────────────────────────────────────── run ──
class Mirror:
    def __init__(self, out, prev, workers=6):
        self.out = Path(out)
        self.prev = Path(prev) if prev else None
        self.workers = workers
        self.state = jload(self.out / "state.json", {})
        self.state.setdefault("v", 1)
        self.state.setdefault("dirs", {})
        self.state.setdefault("seen", {})
        self.state.setdefault("btk", {})
        self.state.setdefault("pinned", [])
        self.manifest = jload(self.out / "manifest.json", {})
        self.missions = {}          # key -> mission dict (loaded on demand)
        self.touched = set()
        self.storms = {}            # sid -> rows (best track)
        self.changed = False
        self.newfiles = 0

    # mission store on disk
    def mpath(self, key):
        return self.out / "missions" / key[:4] / f"{key}.json"

    def mission(self, key):
        if key not in self.missions:
            p = self.mpath(key)
            if p.exists():
                self.missions[key] = jload(p, None)
        return self.missions.get(key)

    def fetch_files(self, year, prod, names):
        def one(n):
            return n, get(f"{NHC}/archive/recon/{year}/{prod}/{n}")
        res = []
        with cf.ThreadPoolExecutor(self.workers) as ex:
            for n, txt in ex.map(one, names):
                res.append((n, txt))
        return res

    def ingest(self, year, prod, files, keep=None):
        """Decode + merge. keep(rec) -> bool filters (backfill)."""
        dec = D.DECODERS[prod]
        store = {}
        n_ok = 0
        for name, txt in sorted(files):
            if not txt:
                continue
            ft = D.file_time(name)
            try:
                rec = dec(txt, ft)
            except Exception as e:  # noqa: BLE001 - one malformed bulletin must not stop the mirror
                log(f"  ! decode {prod}/{name}: {type(e).__name__}: {e}")
                continue
            if not rec:
                continue
            if rec["kind"] == "pod":
                pods = self.manifest.setdefault("pods", [])
                if not any(p["t"] == rec["t"] for p in pods):
                    pods.append(rec)
                    pods.sort(key=lambda p: p["t"])
                    del pods[:-4]
                    self.changed = True
                continue
            if keep is not None and not keep(rec):
                continue
            key = mission_key(rec, ft)
            m = self.mission(key)
            if m is not None:
                store[key] = m
            merge(store, prod, name, rec, ft)
            self.missions[key] = store[key]
            self.touched.add(key)
            n_ok += 1
        if n_ok:
            self.changed = True
        return n_ok

    # ─── ATCF ───
    def load_btk(self, year, live=True, only=None):
        """Atlantic b-decks for `year`: current year from atcf/btk (changed files only), older from the archive."""
        st = self.state["btk"].setdefault(str(year), {})
        sdir = self.out / "storms"
        if live and year == NOW.year:
            html = get(f"{ATCF}/btk/")
            listing = {}
            for m in re.finditer(r'href="(bal\d{2}\d{4}\.dat)".*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', html or "", re.S):
                if m.group(1).endswith(f"{year}.dat"):
                    listing[m.group(1)] = m.group(2)
            for fn, mt in listing.items():
                sid = fn[1:9].upper()
                if only and sid not in only:
                    continue
                if st.get(fn) == mt and (sdir / f"{sid}.json").exists():
                    continue
                txt = get(f"{ATCF}/btk/{fn}")
                if txt:
                    self.write_storm(sid, parse_atcf(txt))
                    st[fn] = mt
        else:
            for sid in only or []:
                if (sdir / f"{sid}.json").exists():
                    continue
                fn = f"b{sid.lower()}.dat.gz"
                raw = get(f"{ATCF}/archive/{year}/{fn}", binary=True)
                if raw:
                    self.write_storm(sid, parse_atcf(gzip.decompress(raw).decode("latin-1")))
        for p in sdir.glob(f"*{year}.json"):
            s = jload(p, None)
            if s:
                self.storms[p.stem] = s["btk"]

    def write_storm(self, sid, rows, fst=None):
        p = self.out / "storms" / f"{sid}.json"
        s = jload(p, {"sid": sid})
        if rows is not None:
            s["btk"] = [r[:6] for r in rows]
            s["name"] = storm_name(rows) or s.get("name") or sid
            self.storms[sid] = s["btk"]
        if fst is not None:
            s["fcst"] = fst
        s["updated"] = int(NOW.timestamp())
        if jdump(p, s):
            self.changed = True

    def fetch_fst(self, sid):
        txt = get(f"{ATCF}/fst/{sid.lower()}.fst")
        rows = parse_atcf(txt, best=False)
        if not rows:
            return
        t0 = rows[0][8]
        pts = [[r[7], r[1], r[2], r[3]] for r in rows if r[8] == t0]
        self.write_storm(sid, None, dict(t0=t0, pts=pts))

    # ─── storm assignment ───
    def assign(self, m, st):
        """m: mission or its manifest entry (year, storm_no, mission); st: mission_stats()."""
        y = m.get("year") or int(st.get("key", "0000")[:4])
        if m.get("storm_no") and (m.get("mission") or "")[-1:] == "A":
            return f"AL{m['storm_no']:02d}{y}", "mission number"
        sid = st.get("vdm_sid")
        if sid:
            return sid, "vortex message"
        if not st.get("med") or not st.get("first"):
            return None, None
        tm = (st["first"] + st["last"]) / 2
        best = None
        for sid, rows in self.storms.items():
            if not sid.startswith("AL") or not sid.endswith(str(y)):
                continue
            p = track_at(rows, tm)
            if p is None:
                continue
            d = km(st["med"][0], st["med"][1], p[0], p[1])
            if d < 2000 and (best is None or d < best[1]):
                best = (sid, d)
        if best:
            return best[0], f"nearest storm ({round(best[1])} km from the mission's median position)"
        return None, None

    # ─── outputs ───
    def write_missions(self):
        for key in sorted(self.touched):
            m = self.missions[key]
            m["screened"] = m.get("screened", 0) + screen_track(m["hdob"]["rows"])
            m["updated"] = int(NOW.timestamp())
            if jdump(self.mpath(key), m):
                self.changed = True

    def build_manifest(self, active, pinned):
        man = self.manifest
        missions = {}
        old = man.get("missions", {})
        for p in sorted((self.out / "missions").glob("*/*.json")):
            key = p.stem
            size = p.stat().st_size
            if key not in self.touched and key in old and old[key].get("bytes") == size:
                e = dict(old[key])                       # unchanged file: reuse its stats, re-check the storm
                e["storm"], e["how"] = self.assign(e, e)
                missions[key] = e
                continue
            m = self.missions.get(key) or jload(p, None)
            if not m:
                continue
            st = mission_stats(m)
            sid, how = self.assign(m, st)
            missions[key] = dict(aircraft=m["aircraft"], type=m["type"], mission=m["mission"],
                                 name=m.get("name") or "", year=m["year"], storm_no=m.get("storm_no"),
                                 storm=sid, how=how, bytes=size, file=f"missions/{key[:4]}/{key}.json",
                                 flagged=m.get("flagged", 0), screened=m.get("screened", 0), **st)
        storms = {}
        for sid in sorted({v["storm"] for v in missions.values() if v["storm"]} | {a["id"] for a in active}):
            s = jload(self.out / "storms" / f"{sid}.json", {})
            ms = sorted((k for k, v in missions.items() if v["storm"] == sid),
                        key=lambda k: missions[k]["last"] or 0, reverse=True)
            last = max([missions[k]["last"] or 0 for k in ms], default=0)
            storms[sid] = dict(name=s.get("name") or next((a["name"] for a in active if a["id"] == sid), sid),
                               year=int(sid[-4:]), missions=ms, last=last or None,
                               active=any(a["id"] == sid for a in active), pinned=sid in pinned,
                               file=f"storms/{sid}.json" if s else None)
        man.update(v=1, updated=int(NOW.timestamp()), season=NOW.year, active=active, storms=storms,
                   missions=missions,
                   newest_ob=max([v["last_hdob"] or 0 for v in missions.values()], default=None) or None,
                   credit="NOAA National Hurricane Center reconnaissance archive (USAF 53rd WRS and NOAA AOC)")
        if jdump(self.out / "manifest.json", man):
            self.changed = True

    def prune(self, pinned):
        """Keep the current season + pinned storms; drop older live missions and storms."""
        mdir = self.out / "missions"
        for p in mdir.glob("*/*.json"):
            y = int(p.parent.name)
            if y >= NOW.year:
                continue
            m = jload(p, {})
            st = mission_stats(m) if m else {}
            sid, _ = self.assign(m, st) if m else (None, None)
            if sid not in pinned:
                p.unlink()
                self.changed = True
        for p in (self.out / "storms").glob("*.json"):
            if int(p.stem[-4:]) < NOW.year and p.stem not in pinned:
                p.unlink()
                self.changed = True


def current_storms():
    txt = get(f"{NHC}/CurrentStorms.json")
    try:
        j = json.loads(txt or "")
    except ValueError:
        return None, None
    act = []
    for s in j.get("activeStorms", []):
        sid = (s.get("id") or "").upper()
        if not sid.startswith("AL"):
            continue
        act.append(dict(id=sid, name=s.get("name"), cls=s.get("classification"),
                        vmax=int(s["intensity"]) if str(s.get("intensity", "")).isdigit() else None,
                        mslp=int(s["pressure"]) if str(s.get("pressure", "")).isdigit() else None,
                        la=s.get("latitudeNumeric"), lo=s.get("longitudeNumeric"), t=s.get("lastUpdate"),
                        adv=(s.get("publicAdvisory") or {}).get("advNum")))
    return act, hashlib.md5(json.dumps(act, sort_keys=True).encode()).hexdigest()


def live(mx, days_back):
    year = NOW.year
    st = mx.state
    in_season = 6 <= NOW.month <= 11
    active, cs_hash = current_storms()
    if active is None:
        active = mx.manifest.get("active", [])
        cs_hash = st.get("cs")
    last_full = st.get("last_full", 0)
    if not in_season and not active and NOW.timestamp() - last_full < 2 * 3600 and days_back is None:
        log("off-season, no Atlantic storm, checked < 2 h ago: nothing to do")
        return
    st["last_full"] = int(NOW.timestamp())
    top = parse_dirs(get(f"{NHC}/archive/recon/{year}/"))
    if not top:
        log("  ! no top-level listing (NHC unreachable?) - nothing changed")
        return
    sdirs = st["dirs"].setdefault(str(year), {})
    seen_all = st["seen"]
    last_day = st.get("last_day")
    for prod in PRODUCTS:
        mt = top.get(prod)
        if mt is None or (mt == sdirs.get(prod) and days_back is None):
            continue
        # days to list: since the last day this product was listed (min yesterday), at most 5 (or --days)
        n = days_back if days_back is not None else 1
        if last_day:
            try:
                gap = (NOW.date() - datetime.strptime(last_day, "%Y%m%d").date()).days
                n = max(n, min(gap, 5))
            except ValueError:
                pass
        seen = seen_all.setdefault(f"{year}/{prod}", {})
        new = []
        for k in range(n, -1, -1):
            d = (NOW - timedelta(days=k)).strftime("%Y%m%d")
            if d[:4] != str(year):
                continue
            pat = f"*.{d}*"
            lst = parse_listing(get(f"{NHC}/archive/recon/{year}/{prod}/?P={pat}"))
            for name, sig in lst.items():
                if seen.get(name) != sig:
                    new.append(name)
                    seen[name] = sig
        if new:
            log(f"  {prod}: {len(new)} new file(s)")
            files = mx.fetch_files(year, prod, new)
            got = mx.ingest(year, prod, files)
            mx.newfiles += len(new)
            log(f"    decoded {got}")
        sdirs[prod] = mt
    st["last_day"] = NOW.strftime("%Y%m%d")
    # prune the seen-lists to 10 days (only recent names are ever listed again)
    cut = (NOW - timedelta(days=10)).strftime("%Y%m%d")
    for k, v in seen_all.items():
        for name in [n for n in v if (D.file_time(n) or NOW).strftime("%Y%m%d") < cut]:
            v.pop(name)
    # storms: b-decks (changed files only) every run with new data or new advisory, else hourly
    need_btk = mx.touched or cs_hash != st.get("cs") or NOW.timestamp() - st.get("btk_t", 0) > 3600
    if need_btk:
        mx.load_btk(year)
        st["btk_t"] = int(NOW.timestamp())
    else:
        for p in (mx.out / "storms").glob(f"*{year}.json"):
            s = jload(p, None)
            if s:
                mx.storms[p.stem] = s["btk"]
    if cs_hash != st.get("cs"):
        for a in active:
            mx.fetch_fst(a["id"])
        st["cs"] = cs_hash
        mx.changed = True
    pinned = set(st["pinned"])
    for y in {int(s[-4:]) for s in pinned}:
        if y != year:
            for p in (mx.out / "storms").glob(f"*{y}.json"):
                s = jload(p, None)
                if s:
                    mx.storms[p.stem] = s["btk"]
    if mx.changed or mx.touched:
        mx.write_missions()
        mx.prune(pinned)
        mx.build_manifest(active, pinned)


def backfill(mx, year, sid):
    """All Atlantic recon in the storm's b-deck window; keep missions that belong to `sid`."""
    sid = sid.upper()
    if not re.fullmatch(r"AL\d{6}", sid):
        sid = f"{sid[:4]}{year}" if re.fullmatch(r"AL\d{2}", sid) else sid
    log(f"backfill {sid}")
    mx.load_btk(year, live=(year == NOW.year), only=[sid])
    rows = mx.storms.get(sid)
    if not rows:
        log(f"  ! no b-deck for {sid}")
        return
    t0 = datetime.fromtimestamp(rows[0][0], UTC) - timedelta(days=1)
    t1 = datetime.fromtimestamp(rows[-1][0], UTC) + timedelta(days=1)
    days = []
    d = t0
    while d <= t1:
        days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    log(f"  window {days[0]}..{days[-1]} ({len(days)} days)")
    no = int(sid[2:4])
    mx.state["pinned"] = sorted(set(mx.state["pinned"]) | {sid})
    for prod in PRODUCTS:
        if prod == "REPRPD":
            continue
        names = []
        for dd in days:
            names += list(parse_listing(get(f"{NHC}/archive/recon/{year}/{prod}/?P=*.{dd}*")))
        log(f"  {prod}: {len(names)} files")
        files = mx.fetch_files(year, prod, names)

        def keep(rec, no=no):
            # numbered missions: keep this storm's; unnumbered (surveillance/invest): decided by proximity below
            return rec.get("storm_no") in (None, no) or rec.get("mission", "")[-1:] != "A"
        mx.ingest(year, prod, files, keep=keep)
        mx.newfiles += len(names)
    # proximity pass: drop unnumbered missions that are not near this storm
    for key in list(mx.touched):
        m = mx.missions[key]
        st = mission_stats(m)
        s, _ = mx.assign(m, st)
        if s != sid:
            mx.touched.discard(key)
            mx.missions.pop(key)
            p = mx.mpath(key)
            if p.exists():
                p.unlink()
    log(f"  kept {len(mx.touched)} mission(s): {sorted(mx.touched)}")
    mx.write_missions()
    for y in {int(s[-4:]) for s in mx.state["pinned"]} | {NOW.year}:
        for p in (mx.out / "storms").glob(f"*{y}.json"):
            s = jload(p, None)
            if s:
                mx.storms[p.stem] = s["btk"]
    mx.build_manifest(mx.manifest.get("active", []), set(mx.state["pinned"]))
    mx.changed = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="working tree of the data branch (written in place)")
    ap.add_argument("--prev", help="checkout of the previous recon-data tip (copied into OUT first)")
    ap.add_argument("--backfill", default="", help='"YEAR SID", e.g. "2024 AL14" (dispatch only)')
    ap.add_argument("--days", type=int, default=None, help="live mode: list this many days back")
    a = ap.parse_args()
    out = Path(a.out)
    if a.prev and Path(a.prev).is_dir() and not out.exists():
        shutil.copytree(a.prev, out, ignore=shutil.ignore_patterns(".git"))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mx = Mirror(out, a.prev)
    if a.backfill.strip():
        y, s = a.backfill.split()
        backfill(mx, int(y), s)
    else:
        live(mx, a.days)
    jdump(out / "state.json", mx.state)
    changed = mx.changed
    log(f"done in {time.time() - t0:.1f}s: {mx.newfiles} new file(s), changed={changed}")
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")
            man = jload(out / "manifest.json", {})
            air = bool(man.get("newest_ob") and NOW.timestamp() - man["newest_ob"] < 40 * 60)
            f.write(f"airborne={'true' if air else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
