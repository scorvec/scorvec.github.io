#!/usr/bin/env python3
"""Mirror NOAA AOML/HRD real-time tail Doppler radar (TDR) graphics to the `tdr-data` branch for recon.html.

Source (no CORS header, so the browser cannot read it): https://www.aoml.noaa.gov/ftp/pub/hrd/data/RTradar/<year>/
<YYYYMMDD><letter><n>/{swaths,composites,rzmeans,tilt,zoom,profiles,centers}/ - PNG graphics produced from the
real-time TDR analyses (automated QC; run by NOAA/OMAO AOC since 2026, graphics by HRD). The TDR centre per analysis
(0.5/2/3/6 km, from the analysis' storm-relative flow) comes as text from .../hrd/reasor/tdr_centers/ (linked from
HRD's "Realtime_TDR_graphics_Overview.pdf"). No real-time gridded TDR is published; the post-season Level 2 swaths
(radar/level2/<year>/<storm>/) and the TC-RADAR Level 3 climatology are the gridded products.

What is kept (graphics re-encoded to LOSSLESS WebP, identical pixels, ~40 % of the PNG bytes):
  swaths      reflectivity + wind speed per analysis at 1, 2, 3, 5, 8, 10 km      (every analysis)
  composites  reflectivity + wind, and 0.5 km + flight-level wind                  (every cumulative window)
  rzmeans     azimuthal-mean reflectivity / tangential / radial wind               (latest window only)
  tilt, zoom  composite vorticity + vortex tilt; inner-80-km zoom                  (latest window only)
  profiles    in/outbound wind-speed profiles (every analysis); dbz / vr / PBL     (latest analysis only)
  centers     track map of the 2- and 6-km TDR centres                             (latest)
Everything else (vorticity/w per height, precipitation class, heights 4, 6, 7, 9, 11-18 km) is not mirrored.

Missions: an HRD flight ID (date + aircraft letter + flight number: H = NOAA2 N42RF, I = NOAA3 N43RF, N = NOAA9
N49RF) is matched to the NHC mission in the recon mirror's manifest: same aircraft, storm name in the file names =
the NHC storm name or the mission's own name (SURV etc.), and the analysis times inside the mission's
observation window (+-45 min). Unmatched flights (East Pacific, training, winter storms) are not mirrored.

Layout of the branch:
  index.json                     {missions: {<NHC mission key>: summary + file}, ...}
  m/<year>/<flight>.json         one flight: analyses, composites, latest products, centres (screened)
  img/<year>/<flight>/<name>.webp
  state.json                     listing signatures (mtime|size), mapping cache - the change detector

Usage (recon.yml, after recon_mirror.py):  tdr_mirror.py --manifest /tmp/recon-data [--backfill "2025 AL13"]
The branch is cloned blob-less (--filter=blob:none) so a run downloads only the JSON it reads, never the images,
and pushes one parentless commit (old blobs are referenced, not re-sent).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = timezone.utc
EAST = ZoneInfo("America/New_York")           # the AOML server's listing clock (checked: 1647Z analysis, file 13:41 EDT)
UA = "scorvec.com recon mirror (+https://scorvec.com/recon.html)"
HRD = "https://www.aoml.noaa.gov/ftp/pub/hrd/data/RTradar"
CTR = "https://www.aoml.noaa.gov/ftp/pub/hrd/reasor/tdr_centers"
BR = "tdr-data"
LETTER = {"H": "NOAA2", "I": "NOAA3", "N": "NOAA9"}
TAIL = {"NOAA2": "WP-3D N42RF", "NOAA3": "WP-3D N43RF", "NOAA9": "G-IV N49RF"}
SUBS = ["swaths", "composites", "rzmeans", "tilt", "zoom", "profiles", "centers"]
SWATH_KM = {1, 2, 3, 5, 8, 10}
NOW = datetime.now(UTC)
RECENT_D = 2                 # flights this many days old are listed every run; older ones every FULL_H hours
FULL_H = 6
CODE_V = 1


def log(*a):
    print(*a, flush=True)


def get(url, binary=False, tries=3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
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


# Apache fancy listing (?F=1): name, "YYYY-MM-DD HH:MM" (US Eastern), size
ROW = re.compile(r'<a href="([^"?/][^"]*?)(/?)">[^<]*</a>\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+(\S+)')


def listing(url):
    """{name: (mtime_utc_epoch, size_str, is_dir)} or None when unreachable."""
    html = get(url.rstrip("/") + "/?F=1")
    if html is None:
        return None
    out = {}
    for m in ROW.finditer(html):
        t = datetime.strptime(m.group(3), "%Y-%m-%d %H:%M").replace(tzinfo=EAST).astimezone(UTC)
        out[m.group(1)] = (int(t.timestamp()), m.group(4), m.group(2) == "/")
    return out


# ─────────────────────────────────────────────────────── file names ──
FID = re.compile(r"^(\d{8})([A-Z])(\d)$")
NAME = re.compile(r"^(\d{6}[A-Z]\d)_([A-Za-z0-9-]+)_(.+)\.png$")
KINDS = [
    ("sw", "swaths", re.compile(r"^(\d{4})_dbz_ws_(\d+(?:\.\d+)?)km$")),
    ("comp", "composites", re.compile(r"^(\d{4})_(\d{4})_ws_(dbz|nhc)_planview$")),
    ("rz", "rzmeans", re.compile(r"^(\d{4})_(\d{4})_(dbz|ur|vt)_rzmean$")),
    ("tilt", "tilt", re.compile(r"^(\d{4})_(\d{4})_vort_tilt$")),
    ("zoom", "zoom", re.compile(r"^(\d{4})_(\d{4})_ws_dbz_planview_zoom$")),
    ("prof", "profiles", re.compile(r"^(\d{4})_(ws|dbz|vr_rel|PBL_rel)_profile_(\d{3})_(\d{3})$")),
    ("ctr", "centers", re.compile(r"^tdrcenters$")),
]


def resolve(day, hhmm, mtime=None):
    """Analysis time from the folder date + HHMM. Hours >= 24 run on past midnight (2024 style); otherwise a time
    after midnight is wrapped (2026 style), so the day is the later of D / D+1 that is not after the file's
    modification time (+30 min). Without a modification time: D."""
    hh, mm = int(hhmm[:2]), int(hhmm[2:])
    if hh >= 24:
        return int((day + timedelta(days=1, hours=hh - 24, minutes=mm)).timestamp())
    cands = [day + timedelta(hours=hh, minutes=mm), day + timedelta(days=1, hours=hh, minutes=mm)]
    if mtime:
        ok = [c for c in cands if c.timestamp() <= mtime + 1800]
        if ok:
            return int(ok[-1].timestamp())
    return int(cands[0].timestamp())


def classify(fid, sub, name, mtime):
    """-> dict(kind, name, t | t0,t1, ...) or None (not a product we mirror)."""
    m = NAME.match(name)
    if not m or m.group(1) != fid[2:]:
        return None
    rest = m.group(3)
    day = datetime.strptime(fid[:8], "%Y%m%d").replace(tzinfo=UTC)
    for kind, s, rx in KINDS:
        if s != sub:
            continue
        g = rx.match(rest)
        if not g:
            return None
        r = dict(kind=kind, name=name, storm=m.group(2).upper())
        if kind == "sw":
            km = float(g.group(2))
            if km not in SWATH_KM:
                return None
            r.update(t=resolve(day, g.group(1), mtime), h=int(km))
        elif kind in ("comp", "rz", "tilt", "zoom"):
            t0 = resolve(day, g.group(1), mtime)
            t1 = resolve(day, g.group(2), mtime)
            if t1 < t0:
                t1 += 86400
            r.update(t0=t0, t1=t1)
            if kind in ("comp", "rz"):
                r["v"] = g.group(3)
        elif kind == "prof":
            r.update(t=resolve(day, g.group(1), mtime), v=g.group(2), azi=[int(g.group(3)), int(g.group(4))])
        return r
    return None


def select(items):
    """Which classified files to keep (see the module docstring)."""
    keep = []
    for kind in ("rz", "tilt", "zoom"):
        xs = [i for i in items if i["kind"] == kind]
        if xs:
            t1 = max(i["t1"] for i in xs)
            keep += [i for i in xs if i["t1"] == t1]
    pr = [i for i in items if i["kind"] == "prof"]
    if pr:
        tl = max(i["t"] for i in pr)
        keep += [i for i in pr if i["v"] == "ws" or i["t"] == tl]
    keep += [i for i in items if i["kind"] in ("sw", "comp", "ctr")]
    return keep


# ─────────────────────────────────────────────────────── geometry ──
def km(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = p2 - p1, math.radians(d - b)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def interp(pts, t, extrap):
    """pts: sorted [(t, la, lo)]. Linear in time; outside, extrapolate the end segment up to `extrap` s."""
    if not pts:
        return None
    if len(pts) == 1:
        return (pts[0][1], pts[0][2]) if abs(t - pts[0][0]) <= extrap else None
    if t < pts[0][0] - extrap or t > pts[-1][0] + extrap:
        return None
    a, b = (pts[0], pts[1]) if t < pts[0][0] else (pts[-2], pts[-1])
    for p, q in zip(pts, pts[1:]):
        if p[0] <= t <= q[0]:
            a, b = p, q
            break
    f = (t - a[0]) / max(1, b[0] - a[0])
    return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])


# ─────────────────────────────────────────────────────── centres ──
CTR_HEAD = re.compile(r"TDR center info from (\d{6}[A-Z]\d)_(\d{4})_xy\.nc")
CTR_ROW = re.compile(r"^\s*(0\.5|2\.0|3\.0|6\.0)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.M)
CTR_TILT = re.compile(r"2-6-km Vortex Tilt:\s*([\d.]+)\s*km\s+at\s+(\d+)\s*deg")


def parse_centres(txt, fid6, hhmm):
    """-> (record, reason). Flag value 99.99 = no centre at that level. The header must name this very analysis
    (a stale copy of another flight's file was seen: 261006N1_2022 carried 260908N1_2507)."""
    h = CTR_HEAD.search(txt or "")
    if not h:
        return None, "no header"
    if h.group(1) != fid6 or h.group(2) != hhmm:
        return None, f"header names {h.group(1)}_{h.group(2)}"
    lev = {}
    for m in CTR_ROW.finditer(txt):
        la, lo = float(m.group(2)), float(m.group(3))
        # flag 99.99 (a latitude of 99.99 is impossible; 99.99 deg W alone is a real Gulf longitude)
        if la >= 99 or not (0 < la < 60 and 0 < lo < 180):        # Atlantic: deg N, deg W
            continue
        lev[{"0.5": "05", "2.0": "2", "3.0": "3", "6.0": "6"}[m.group(1)]] = [round(la, 2), round(-lo, 2)]
    if not lev:
        return None, "no centre (flagged 99.99)"
    rec = dict(c=lev)
    tl = CTR_TILT.search(txt)
    if tl and float(tl.group(1)) < 900:
        rec["tilt"] = [float(tl.group(1)), int(tl.group(2))]
    return rec, None


# ─────────────────────────────────────────────────────── git ──
def git(repo, *a, inp=None, check=True):
    r = subprocess.run(["git", "-C", str(repo), *a], input=inp, capture_output=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(a)}: {r.stderr.decode()[:400]}")
    return r.stdout


class Branch:
    """Blob-less clone of the data branch; files are read lazily (git show) and written straight into the index."""

    def __init__(self, path, url):
        self.path = Path(path)
        self.url = url
        self.exists = True
        if not (self.path / ".git").exists():
            r = subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth=1", "-b", BR,
                                url, str(self.path)], capture_output=True)
            if r.returncode:
                log(f"  {BR} not cloned ({r.stderr.decode().strip()[:160]}) - starting a new branch")
                self.path.mkdir(parents=True, exist_ok=True)
                git(self.path, "init", "-q", "-b", BR)
                git(self.path, "remote", "add", "origin", url)
                self.exists = False
            git(self.path, "config", "user.name", "github-actions[bot]")
            git(self.path, "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
        else:
            self.exists = bool(git(self.path, "rev-parse", "-q", "--verify", "HEAD", check=False).strip())
        if self.exists:
            git(self.path, "read-tree", "HEAD")
        self.added, self.removed = 0, 0

    def read(self, p):
        if not self.exists:
            return None
        r = subprocess.run(["git", "-C", str(self.path), "show", f"HEAD:{p}"], capture_output=True)
        return r.stdout if r.returncode == 0 else None

    def write(self, p, data: bytes):
        sha = git(self.path, "hash-object", "-w", "--stdin", inp=data).decode().strip()
        git(self.path, "update-index", "--add", "--cacheinfo", f"100644,{sha},{p}")
        self.added += 1

    def remove(self, p):
        git(self.path, "update-index", "--force-remove", p)
        self.removed += 1

    def commit_push(self, msg):
        tree = git(self.path, "write-tree", "--missing-ok").decode().strip()   # else git fetches every old blob
        if self.exists and tree == git(self.path, "rev-parse", "HEAD^{tree}").decode().strip():
            log("  tree unchanged - nothing to push")
            return False
        c = git(self.path, "commit-tree", tree, "-m", msg).decode().strip()     # parentless: no history growth
        for i in range(3):
            r = subprocess.run(["git", "-C", str(self.path), "push", "-q", "-f", "origin", f"{c}:refs/heads/{BR}"],
                               capture_output=True)
            if r.returncode == 0:
                git(self.path, "update-ref", "HEAD", c)
                self.exists = True
                log(f"  {BR} pushed ({c[:9]})")
                return True
            log(f"  push attempt {i + 1} failed: {r.stderr.decode()[:300]}")
            time.sleep(5 * (i + 1))
        raise RuntimeError(f"{BR} push failed")


# ─────────────────────────────────────────────────────── mirror ──
def jbytes(o):
    return json.dumps(o, separators=(",", ":"), allow_nan=False).encode()


def to_webp(png):
    """Lossless re-encode; None when the PNG does not decode (truncated upload) - retried next run."""
    try:
        from PIL import Image
    except ImportError:          # installed only when a run has images to convert (most runs have none)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--user", "--break-system-packages", "pillow"],
                       check=True)
        import site
        sys.path.insert(0, site.getusersitepackages())
        from PIL import Image
    try:
        im = Image.open(io.BytesIO(png))
        im.load()
    except Exception as e:  # noqa: BLE001
        log(f"    ! undecodable PNG ({type(e).__name__})")
        return None
    if im.width < 200 or im.height < 200:
        log(f"    ! implausible image size {im.size}")
        return None
    b = io.BytesIO()
    im.save(b, "WEBP", lossless=True, method=4)
    return b.getvalue()


class TDR:
    def __init__(self, br, man_dir):
        self.br = br
        self.man_dir = Path(man_dir)
        self.man = json.loads((self.man_dir / "manifest.json").read_text())
        self.state = json.loads(br.read("state.json") or b"{}")
        self.state.setdefault("flights", {})
        self.index = json.loads(br.read("index.json") or b"{}")
        self.index.setdefault("missions", {})
        self.changed = False
        self.ctr_cache = {}          # year -> listing of tdr_centers
        self.stats = dict(new=0, img_bytes=0, ctr_ok=0, ctr_rej={})
        self.scanned = set()

    # NHC side
    def storm_name(self, sid):
        s = self.man["storms"].get(sid) or {}
        return (s.get("name") or "").upper()

    def candidates(self, ac, year):
        return [(k, m) for k, m in self.man["missions"].items()
                if m.get("aircraft") == ac and m.get("year") == year and m.get("storm") and m.get("first")]

    def match(self, fid, items):
        """HRD flight -> NHC mission key (and how), from aircraft + storm name + analysis times."""
        g = FID.match(fid)
        ac = LETTER.get(g.group(2)) if g else None
        if not ac:
            return None, f"aircraft letter {g.group(2) if g else '?'} has no NHC call sign mapping"
        ts = sorted({i.get("t") or i.get("t1") for i in items if (i.get("t") or i.get("t1"))})
        names = {i["storm"] for i in items}
        if not ts:
            return None, "no analyses"
        best = None
        for k, m in self.candidates(ac, int(fid[:4])):
            sn, mn = self.storm_name(m["storm"]), (m.get("name") or "").upper()
            if names and not (names & {sn, mn}):
                continue
            inside = sum(1 for t in ts if m["first"] - 2700 <= t <= m["last"] + 2700)
            if inside and (best is None or inside > best[1]):
                best = (k, inside)
        if not best:
            return None, f"no {ac} mission of {'/'.join(sorted(names))} overlapping the analyses"
        return best[0], f"{ac}, storm name {'/'.join(sorted(names))}, {best[1]} of {len(ts)} analyses inside the mission window"

    # centres
    def centre_list(self, year):
        if year not in self.ctr_cache:
            url = CTR if year == NOW.year else f"{CTR}/{year}"
            self.ctr_cache[year] = listing(url) or {}
            if not self.ctr_cache[year] and year != NOW.year:
                log(f"  (no TDR centre text files for {year})")
        return self.ctr_cache[year]

    def reference(self, sid):
        """Centre reference for the screen: the storm's vortex-message fixes (all missions), else the best track."""
        fx = []
        for k in (self.man["storms"].get(sid) or {}).get("missions", []):
            for f in self.man["missions"][k].get("fixes") or []:
                fx.append((f[0], f[1], f[2]))
        fx.sort()
        bt = []
        p = self.man_dir / "storms" / f"{sid}.json"
        if p.exists():
            bt = [(r[0], r[1], r[2]) for r in json.loads(p.read_text()).get("btk", [])]
        return fx, bt

    def centres(self, fid, sid, fl):
        """Parse + screen this flight's TDR centre files; cached by name+signature in the flight state."""
        year = int(fid[:4])
        lst = self.centre_list(year)
        pref = fid[2:] + "_"
        fx, bt = self.reference(sid)
        cache = fl.setdefault("ctr", {})
        out = []
        day = datetime.strptime(fid[:8], "%Y%m%d").replace(tzinfo=UTC)
        for name, (mt, size, isdir) in sorted(lst.items()):
            if isdir or not name.startswith(pref) or not name.endswith("_tdrcenters.txt"):
                continue
            hhmm = name[len(pref):len(pref) + 4]
            sig = f"{mt}|{size}"
            c = cache.get(name)
            if not c or c.get("sig") != sig:
                self.changed = True
                txt = get(f"{CTR if year == NOW.year else CTR + '/' + str(year)}/{name}")
                if txt is None:
                    continue
                rec, why = parse_centres(txt, fid[2:], hhmm)
                c = dict(sig=sig, rec=rec, why=why)
                cache[name] = c
            rec, why = c["rec"], c["why"]
            t = resolve(day, hhmm, mt)
            if rec:
                p2 = rec["c"].get("2") or rec["c"].get("05") or rec["c"].get("3")
                ref = interp(fx, t, 3 * 3600)
                lim, src = 100, "vortex-message fixes"
                if ref is None:
                    ref, lim, src = interp(bt, t, 12 * 3600), 200, "best track"
                if ref is not None and p2:
                    d = km(p2[0], p2[1], ref[0], ref[1])
                    if d > lim:
                        rec, why = None, f"{round(d)} km from the {src} (limit {lim} km)"
                    else:
                        rec = dict(rec, ref=[src, round(d)])
            if rec:
                out.append(dict(t=t, **rec))
                self.stats["ctr_ok"] += 1
            else:
                self.stats["ctr_rej"][why] = self.stats["ctr_rej"].get(why, 0) + 1
                out.append(dict(t=t, bad=why))
        return out

    # one flight
    def scan(self, year, fid, mt_dir, force=False):
        fl = self.state["flights"].setdefault(fid, {})
        base = f"{HRD}/{year}/{fid}"
        sub_l = listing(base)
        if sub_l is None:
            return
        files = fl.setdefault("files", {})            # sub -> {name: sig}
        smt = fl.setdefault("smt", {})
        for sub in SUBS:
            if sub not in sub_l:
                continue
            mt = sub_l[sub][0]
            if not force and smt.get(sub) == mt and sub in files:
                continue
            lst = listing(f"{base}/{sub}")
            if lst is None:
                continue
            files[sub] = {n: f"{v[0]}|{v[1]}" for n, v in lst.items() if not v[2] and n.endswith(".png")}
            smt[sub] = mt
        if fl.get("dir_mt") != mt_dir:
            fl["dir_mt"] = mt_dir
        self.scanned.add(fid)

    def build(self, year, fid):
        """Map, fetch new kept files, write the flight JSON. Returns the NHC key or None."""
        fl = self.state["flights"][fid]
        items = []
        for sub, names in (fl.get("files") or {}).items():
            for n, sig in names.items():
                c = classify(fid, sub, n, int(sig.split("|")[0]))
                if c:
                    c["sub"], c["sig"] = sub, sig
                    items.append(c)
        key, how = self.match(fid, items)
        if (fl.get("match"), fl.get("how")) != (key, how):
            fl["match"], fl["how"] = key, how
            self.changed = True
        if not key:
            return None
        m = self.man["missions"][key]
        keep = select(items)
        have = fl.setdefault("have", {})              # name -> sig of what is on the branch
        idir = f"img/{year}/{fid}"
        for it in keep:
            p = f"{idir}/{it['name'][:-4]}.webp"
            if (have.get(it["name"]) or [None])[0] == it["sig"]:
                continue
            png = get(f"{HRD}/{year}/{fid}/{it['sub']}/{it['name']}", binary=True)
            if not png:
                continue
            w = to_webp(png)
            if not w:
                continue
            self.br.write(p, w)
            have[it["name"]] = [it["sig"], len(w)]
            self.stats["new"] += 1
            self.stats["img_bytes"] += len(w)
            self.changed = True
        keepn = {i["name"] for i in keep}
        for n in [n for n in have if n not in keepn]:   # superseded "latest" products
            self.br.remove(f"{idir}/{n[:-4]}.webp")
            have.pop(n)
            self.changed = True
        got = [i for i in keep if i["name"] in have]
        img = lambda i: f"{idir}/{i['name'][:-4]}.webp"   # noqa: E731
        an = {}
        for i in got:
            if i["kind"] == "sw":
                an.setdefault(i["t"], {"t": i["t"], "sw": {}})["sw"][str(i["h"])] = img(i)
            elif i["kind"] == "prof":
                an.setdefault(i["t"], {"t": i["t"], "sw": {}}).setdefault("prof", {})[i["v"]] = img(i)
                an[i["t"]]["azi"] = i["azi"]
        comps = {}
        for i in got:
            if i["kind"] == "comp":
                comps.setdefault((i["t0"], i["t1"]), {"t0": i["t0"], "t1": i["t1"]})[i["v"]] = img(i)
        latest = {}
        for i in got:
            if i["kind"] == "rz":
                latest.setdefault("rz", {"t0": i["t0"], "t1": i["t1"]})[i["v"]] = img(i)
            elif i["kind"] in ("tilt", "zoom"):
                latest[i["kind"]] = {"t0": i["t0"], "t1": i["t1"], "img": img(i)}
            elif i["kind"] == "ctr":
                latest["ctr"] = {"img": img(i), "v": i["sig"]}
        ctrs = self.centres(fid, m["storm"], fl)
        names = sorted({i["storm"] for i in items})
        doc = dict(v=1, id=fid, key=key, storm=m["storm"], hrd_name="/".join(names), aircraft=m["aircraft"],
                   tail=TAIL.get(m["aircraft"]), how=how, src=f"{HRD}/{year}/{fid}/",
                   analyses=sorted(an.values(), key=lambda a: a["t"]),
                   composites=sorted(comps.values(), key=lambda c: c["t1"]), latest=latest, centres=ctrs,
                   updated=int(NOW.timestamp()))
        path = f"m/{year}/{fid}.json"
        old = self.br.read(path)
        new = jbytes(doc)
        if old is None or json.loads(old) | {"updated": 0} != doc | {"updated": 0}:
            self.br.write(path, new)
            self.changed = True
        self.index["missions"][key] = dict(id=fid, storm=m["storm"], file=path, n_an=len(an), n_comp=len(comps),
                                           n_ctr=sum(1 for c in ctrs if "c" in c), n_img=len(have),
                                           last=max([a["t"] for a in an.values()] + [c["t1"] for c in comps.values()],
                                                    default=None),
                                           bytes=sum(v[1] for v in have.values()))
        return key

    def prune(self):
        """Drop flights whose NHC mission left the recon mirror (season pruning) and everything unmatched."""
        live = set(self.man["missions"])
        for key in [k for k in self.index["missions"] if k not in live]:
            e = self.index["missions"].pop(key)
            fid = e["id"]
            fl = self.state["flights"].get(fid, {})
            for n in list(fl.get("have", {})):
                self.br.remove(f"img/{fid[:4]}/{fid}/{n[:-4]}.webp")
            fl["have"] = {}
            self.br.remove(e["file"])
            self.changed = True
            log(f"  pruned {fid} ({key})")
        # forget state of flights older than the season (and not mapped) so state.json stays small
        for fid in [f for f, v in self.state["flights"].items() if int(f[:4]) < NOW.year and not v.get("have")]:
            self.state["flights"].pop(fid)

    def finish(self):
        self.prune()
        if not self.changed:
            log("  nothing new")
            return
        tot = sum(e.get("bytes") or 0 for e in self.index["missions"].values())
        self.index.update(v=1, code=CODE_V, updated=int(NOW.timestamp()), bytes=tot,
                          credit="Real-time TDR graphics: NOAA AOML Hurricane Research Division; real-time analyses: "
                                 "NOAA OMAO Aircraft Operations Center (automated QC)")
        self.br.write("index.json", jbytes(self.index))
        self.br.write("state.json", jbytes(self.state))
        log(f"  {self.stats['new']} new image(s), {self.stats['img_bytes'] / 1e6:.2f} MB; centres ok {self.stats['ctr_ok']},"
            f" rejected {self.stats['ctr_rej']}; branch images {tot / 1e6:.1f} MB over {len(self.index['missions'])} flight(s)")
        for k, e in sorted(self.index["missions"].items()):
            log(f"    {e['id']} -> {k}: {e['n_an']} analyses, {e['n_comp']} composites, {e['n_ctr']} centres,"
                f" {e['n_img']} images, {(e['bytes'] or 0) / 1e6:.2f} MB")


def year_flights(year):
    lst = listing(f"{HRD}/{year}")
    if lst is None:
        return None
    return {n: v[0] for n, v in lst.items() if v[2] and FID.match(n)}


def run_live(t):
    """Flights of the last RECENT_D days: listed every run. Older matched flights: re-listed every FULL_H hours
    (HRD sometimes re-renders later). Unmatched flights are re-mapped from the stored listing (no request) in
    case their NHC mission arrives later."""
    year = NOW.year
    fl_all = year_flights(year)
    if fl_all is None:
        log("  ! RTradar listing unreachable - nothing changed")
        return
    full = NOW.timestamp() - t.state.get("full_t", 0) > FULL_H * 3600
    for fid, mt in sorted(fl_all.items()):
        d = datetime.strptime(fid[:8], "%Y%m%d").replace(tzinfo=UTC)
        st = t.state["flights"].get(fid)
        if st is None or (NOW - d).days <= RECENT_D or (full and st.get("match")):
            t.scan(year, fid, mt)
        st = t.state["flights"].get(fid) or {}
        if fid in t.scanned or (st.get("files") and (not st.get("match") or st["match"] not in t.index["missions"])):
            key = t.build(year, fid)
            if fid in t.scanned:
                log(f"  {fid}: {key or '-'} ({st.get('how')})")
    if full:
        t.state["full_t"] = int(NOW.timestamp())
        t.changed = True


def run_backfill(t, year, sid):
    p = t.man_dir / "storms" / f"{sid}.json"
    if not p.exists():
        log(f"  ! no storm file for {sid} in the recon mirror - backfill its HDOBs first")
        return
    bt = json.loads(p.read_text())["btk"]
    t0 = datetime.fromtimestamp(bt[0][0], UTC) - timedelta(days=1)
    t1 = datetime.fromtimestamp(bt[-1][0], UTC) + timedelta(days=1)
    fl_all = year_flights(year) or {}
    for fid, mt in sorted(fl_all.items()):
        d = datetime.strptime(fid[:8], "%Y%m%d").replace(tzinfo=UTC)
        if not (t0 <= d <= t1):
            continue
        t.scan(year, fid, mt, force=True)
        key = t.build(year, fid)
        log(f"  {fid}: {key or '-'} ({t.state['flights'][fid].get('how')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="recon mirror tree (manifest.json, storms/)")
    ap.add_argument("--repo", default="/tmp/tdr-git", help="blob-less clone of the tdr-data branch (created)")
    ap.add_argument("--backfill", default="", help='"YEAR SID", e.g. "2025 AL13"')
    ap.add_argument("--dry", action="store_true", help="do not push")
    a = ap.parse_args()
    if not (Path(a.manifest) / "manifest.json").exists():
        log("no recon manifest - nothing to map TDR flights to")
        return 0
    tok, repo = os.environ.get("GH_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "scorvec/scorvec.github.io")
    url = f"https://x-access-token:{tok}@github.com/{repo}.git" if tok else f"https://github.com/{repo}.git"
    t0 = time.time()
    br = Branch(a.repo, url)
    t = TDR(br, a.manifest)
    if a.backfill.strip():
        y, s = a.backfill.split()
        s = s.upper()
        if re.fullmatch(r"AL\d{2}", s):
            s = f"{s}{y}"
        run_backfill(t, int(y), s)
    else:
        run_live(t)
    t.finish()
    pushed = False
    if t.changed and not a.dry:
        pushed = br.commit_push(f"TDR mirror: {NOW:%Y-%m-%d %H:%M} UTC")
    log(f"done in {time.time() - t0:.1f}s, changed={t.changed}, pushed={pushed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
