#!/usr/bin/env python3
"""TEMPORARY survey (delete after use): radiosonde coverage in the WIS2 Global Cache on AWS.

Runs in GitHub Actions only (.github/workflows/wis2-survey-tmp.yml). Lists the
s3://wis2globalcache bucket politely (sequential ListObjectsV2), downloads the
upper-air BUFR files of the last 24 h (capped), decodes them with ecCodes and
prints a coverage report. Writes a small gzipped JSON summary for the artifact.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import statistics as stx
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config

NOW = datetime.now(timezone.utc)
BUCKET = "wis2globalcache"
OUT = Path(os.environ.get("SURVEY_OUT", "survey-out"))
OUT.mkdir(parents=True, exist_ok=True)
MAX_DL = int(os.environ.get("SURVEY_MAX_DL", 14000))
MAX_BYTES = int(os.environ.get("SURVEY_MAX_BYTES", 400e6))
SLEEP = 0.05
NREQ = Counter()

s3 = boto3.client("s3", region_name="us-east-1",
                  config=Config(signature_version=UNSIGNED, retries={"max_attempts": 5},
                                max_pool_connections=8))


def ls(prefix, delim="/", maxpages=None, bucket=BUCKET, client=None):
    """(common prefixes, objects) of one prefix; sequential paging."""
    c = client or s3
    prefs, objs, tok, pages = [], [], None, 0
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if delim:
            kw["Delimiter"] = delim
        if tok:
            kw["ContinuationToken"] = tok
        r = c.list_objects_v2(**kw)
        NREQ["list"] += 1
        pages += 1
        prefs += [p["Prefix"] for p in r.get("CommonPrefixes", [])]
        objs += r.get("Contents", [])
        time.sleep(SLEEP)
        if not r.get("IsTruncated") or (maxpages and pages >= maxpages):
            break
        tok = r["NextContinuationToken"]
    return prefs, objs


def seg(p):
    return p.rstrip("/").split("/")[-1]


UA_RE = re.compile(r"(temp|upper|radiosond|sonde|pilot|profil|raob|vertical|aerolog|sounding)", re.I)
DATE_RE = re.compile(r"^(\d{4}(-\d{2}(-\d{2})?)?|\d{6,10}|\d{1,2}|\d{4}-\d{2}-\d{2}T?\d*)$")


# ------------------------------------------------------------------ 1. discovery
def discover():
    centres, _ = ls("data/")
    centres = [seg(c) for c in centres]
    print(f"== {len(centres)} centre ids under data/", flush=True)
    print("   " + " ".join(centres), flush=True)
    leaves = []          # (prefix, kind, nchildren/nobjs, sample key, newest LM)
    gw = [c for c in centres if "gts" in c.lower()]
    native = [c for c in centres if c not in gw]
    budget = {"n": 0}

    def explore(prefix, depth, cbud):
        if depth > 9 or cbud["n"] > 160:
            leaves.append((prefix, "budget", 0, "", ""))
            return
        cbud["n"] += 1
        prefs, objs = ls(prefix, maxpages=1)
        if objs:
            lm = max(o["LastModified"] for o in objs)
            leaves.append((prefix, "files", len(objs), objs[0]["Key"], lm.isoformat()))
        if not prefs:
            return
        names = [seg(p) for p in prefs]
        if all(DATE_RE.match(n) for n in names):
            leaves.append((prefix, f"dated({len(names)}:{names[0]}..{names[-1]})", len(names), "", ""))
            explore(sorted(prefs)[-1], depth + 1, cbud)          # newest date only
            return
        if len(prefs) > 60:
            leaves.append((prefix, f"wide({len(prefs)})", len(prefs), names[0], ""))
            for p in prefs:
                if UA_RE.search(seg(p)):
                    explore(p, depth + 1, cbud)
            return
        for p in prefs:
            explore(p, depth + 1, cbud)

    for c in native:
        cb = {"n": 0}
        explore(f"data/{c}/", 0, cb)
        budget["n"] += cb["n"]
    for c in gw:
        top, _ = ls(f"data/{c}/data/core/")
        leaves.append((f"data/{c}/data/core/", "gts-T1:" + ",".join(seg(t) for t in top), len(top), "", ""))
    print(f"== discovery: {NREQ['list']} list requests", flush=True)
    for lf in leaves:
        flag = "UA " if UA_RE.search(lf[0]) else "   "
        print(f"  {flag}{lf[0]}  [{lf[1]}] n={lf[2]} {lf[3][-90:]} {lf[4][:16]}", flush=True)
    return centres, gw, native, leaves


# ------------------------------------------------------------------ 2. listing of upper-air trees
def list_upper_air(gw, leaves):
    files = []   # dict(key, size, lm, src, kind)
    for c in gw:
        pre = f"data/{c}/data/core/I/U/"
        a1, _ = ls(pre)
        print(f"== {pre}: A1 = {[seg(a) for a in a1]}", flush=True)
        n0 = NREQ["list"]
        _, objs = ls(pre, delim=None)
        print(f"   {len(objs)} objects, {NREQ['list'] - n0} list requests", flush=True)
        for o in objs:
            files.append(dict(key=o["Key"], size=o["Size"], lm=o["LastModified"], src=c, kind="gts"))
        # how many distinct leaf dirs (A1/A2ii/CCCC)
        dirs = Counter("/".join(o["Key"].split("/")[4:9]) for o in objs)
        print(f"   {len(dirs)} distinct A1/A2/ii/CCCC dirs", flush=True)
    # native upper-air topics: every leaf whose path matches UA_RE; take the topic root
    roots = set()
    for prefix, kind, *_ in leaves:
        m = UA_RE.search(prefix)
        if not m or "gts" in prefix:
            continue
        # cut the prefix after the matching segment
        parts = prefix.rstrip("/").split("/")
        for i, s in enumerate(parts):
            if UA_RE.search(s):
                roots.add("/".join(parts[:i + 1]) + "/")
                break
    for r in sorted(roots):
        n0 = NREQ["list"]
        _, objs = ls(r, delim=None, maxpages=40)
        c = r.split("/")[1]
        print(f"== native {r}: {len(objs)} objects ({NREQ['list'] - n0} req)", flush=True)
        for o in objs:
            files.append(dict(key=o["Key"], size=o["Size"], lm=o["LastModified"], src=c, kind="native"))
    return files


# ------------------------------------------------------------------ 3. decode
import eccodes as ec  # noqa: E402

MISS = ec.CODES_MISSING_DOUBLE


def _arr(h, k):
    try:
        v = ec.codes_get_array(h, k)
        return [float(x) for x in v]
    except Exception:
        return []


def _one(h, k):
    try:
        v = ec.codes_get_array(h, k)
        if len(v) == 0:
            return None
        x = v[0]
        if isinstance(x, (bytes, str)):
            return str(x).strip()
        if x in (MISS, ec.CODES_MISSING_LONG) or (isinstance(x, float) and x < -1e90):
            return None
        return x
    except Exception:
        return None


def _valid(vals):
    return [v for v in vals if v is not None and abs(v) < 1e90 and v != ec.CODES_MISSING_LONG]


def summarise(h):
    d = {}
    b, s = _one(h, "blockNumber"), _one(h, "stationNumber")
    if b is not None and s is not None:
        d["wmo"] = f"{int(b):02d}{int(s):03d}"
    ws = [_one(h, k) for k in ("wigosIdentifierSeries", "wigosIssuerOfIdentifier",
                                "wigosIssueNumber", "wigosLocalIdentifierCharacter")]
    if ws[3]:
        d["wsi"] = "-".join(str(int(x)) if isinstance(x, (int, float)) else str(x) for x in ws)
    y, mo, dd, hh, mi = (_one(h, k) for k in ("year", "month", "day", "hour", "minute"))
    try:
        d["t"] = datetime(int(y), int(mo), int(dd), int(hh), int(mi or 0), tzinfo=timezone.utc).isoformat()
    except Exception:
        d["t"] = None
    d["la"], d["lo"] = _one(h, "latitude"), _one(h, "longitude")
    p = _valid(_arr(h, "pressure"))
    d["nlev"] = len(p)
    d["ptop"] = round(min(p) / 100, 1) if p else None
    d["nT"] = len(_valid(_arr(h, "airTemperature")))
    d["nTd"] = len(_valid(_arr(h, "dewpointTemperature")))
    d["nW"] = len(_valid(_arr(h, "windSpeed")))
    d["nZ"] = len(_valid(_arr(h, "nonCoordinateGeopotentialHeight")) or _valid(_arr(h, "geopotentialHeight")))
    return d


def decode(blob):
    out, err = [], None
    f = io.BytesIO(blob)
    # ecCodes python reads from real files; write to a temp
    tmp = OUT / f"_tmp_{os.getpid()}_{id(blob)}.bufr"
    tmp.write_bytes(blob)
    try:
        with open(tmp, "rb") as fh:
            while True:
                h = ec.codes_bufr_new_from_file(fh)
                if h is None:
                    break
                try:
                    tmpl = list(ec.codes_get_array(h, "unexpandedDescriptors"))
                    ec.codes_set(h, "unpack", 1)
                    n = ec.codes_get(h, "numberOfSubsets")
                    comp = ec.codes_get(h, "compressedData")
                    if n == 1:
                        sub = [summarise(h)]
                    else:
                        sub = []
                        for k in range(1, n + 1):
                            try:
                                ec.codes_set(h, "extractSubset", k)
                                ec.codes_set(h, "doExtractSubsets", 1)
                                h2 = ec.codes_clone(h)
                                ec.codes_set(h2, "unpack", 1)
                                sub.append(summarise(h2))
                                ec.codes_release(h2)
                            except Exception as e:  # noqa: BLE001
                                sub.append({"err": repr(e)[:80]})
                    for x in sub:
                        x["tmpl"] = tmpl[:3] if len(tmpl) > 3 else tmpl
                        x["nsub"] = n
                        x["comp"] = comp
                    out += sub
                except Exception as e:  # noqa: BLE001
                    err = repr(e)[:120]
                finally:
                    ec.codes_release(h)
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:120]
    finally:
        tmp.unlink(missing_ok=True)
    return out, err


SESSION = requests.Session()
SESSION.headers["User-Agent"] = "scorvec.com skew-t explorer coverage survey (+https://scorvec.com/skewt/)"


def fetch(key):
    for i in range(3):
        try:
            r = SESSION.get(f"https://{BUCKET}.s3.amazonaws.com/{requests.utils.quote(key)}", timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None


GTS_RE = re.compile(r"A_([A-Z]{4})(\d{2})([A-Z]{4})(\d{6})([A-Z]{3})?_C_([A-Z]{4})_(\d{14})")


def nominal_from_name(key, lm):
    m = GTS_RE.search(key)
    if not m:
        return None
    dd, hh, mi = int(m.group(4)[:2]), int(m.group(4)[2:4]), int(m.group(4)[4:6])
    base = lm
    for back in range(0, 3):
        mon = (base.replace(day=1) - timedelta(days=1)) if back else base
        try:
            t = datetime(mon.year, mon.month, dd, hh, mi, tzinfo=timezone.utc)
        except ValueError:
            continue
        if t <= lm + timedelta(hours=2):
            return t
    return None


def region(wmo):
    if not wmo:
        return "unknown"
    b = int(wmo[:2])
    if b <= 19 or b in (22, 26, 27, 33, 34):
        return "Europe"
    if b in (20, 21, 23, 24, 25, 28, 29, 30, 31, 32):
        return "Russia (Asian)"
    if 35 <= b <= 41:
        return "Central Asia / Middle East"
    if b in (42, 43):
        return "South Asia"
    if b in (44, 45, 46, 47) or 50 <= b <= 59:
        return "East Asia"
    if b == 48 or 96 <= b <= 98:
        return "SE Asia"
    if 60 <= b <= 69:
        return "Africa"
    if 70 <= b <= 79:
        return "N & C America"
    if 80 <= b <= 88:
        return "South America"
    if b == 89:
        return "Antarctica"
    if 91 <= b <= 95:
        return "Oceania"
    return "other"


def synop_bucket(t):
    """6-hourly synoptic bucket a launch belongs to (launch ~45-75 min before nominal)."""
    t = t + timedelta(hours=3)
    return t.replace(hour=(t.hour // 6) * 6, minute=0, second=0, microsecond=0)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def jma_cache():
    """The JMA copy: region, layout, and whether the same GTS upper-air tree exists."""
    b = "global-cache-of-japan"
    try:
        r = requests.head(f"https://{b}.s3.amazonaws.com/", timeout=30)
        reg = r.headers.get("x-amz-bucket-region", "us-east-1")
        print(f"== JMA cache: HEAD {r.status_code}, region {reg}", flush=True)
        c = boto3.client("s3", region_name=reg, config=Config(signature_version=UNSIGNED))
        top, objs = ls("", bucket=b, client=c, maxpages=1)
        print(f"   top level: {top[:20]} objs {len(objs)}", flush=True)
        for t in top[:3]:
            sub, _ = ls(t, bucket=b, client=c, maxpages=1)
            print(f"   {t}: {len(sub)} prefixes, e.g. {sub[:12]}", flush=True)
        sub, _ = ls("wis2/data/", bucket=b, client=c)
        print(f"   wis2/data/: {len(sub)} centres: {' '.join(seg(x) for x in sub)}", flush=True)
        for pre in ("wis2/data/de-dwd-gts-to-wis2/data/core/I/U/S/", "wis2/data/jp-jma-gts-to-wis2/data/core/I/U/S/",
                    "wis2/data/br-inmet/data/core/weather/surface-based-observations/temp/"):
            _, o = ls(pre, delim=None, bucket=b, client=c, maxpages=20)
            if o:
                lms = [x["LastModified"] for x in o]
                print(f"   {pre}: {len(o)} objects, oldest {min(lms)}, newest {max(lms)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"   JMA cache check failed: {repr(e)[:160]}", flush=True)


GW = ["de-dwd-gts-to-wis2", "jp-jma-gts-to-wis2"]
NATIVE_EXTRA = ["data/kr-kma/data/core/weather/experimental/temp-drop/"]


def targeted():
    """Run 1 mapped the whole bucket (1003 list requests); radiosondes live only in
    the gateways' I/U/<A1>/ trees and in each centre's .../surface-based-observations/temp/."""
    centres = [seg(c) for c in ls("data/")[0]]
    files, counts = [], {}
    for c in GW:
        for a1 in ("K", "S", "J", "D", "E"):
            pre = f"data/{c}/data/core/I/U/{a1}/"
            n0 = NREQ["list"]
            _, objs = ls(pre, delim=None)
            dirs = {"/".join(o["Key"].split("/")[6:9]) for o in objs}
            counts[pre] = len(objs)
            print(f"== {pre}: {len(objs)} objects in {NREQ['list'] - n0} req, {len(dirs)} A2ii/CCCC dirs", flush=True)
            for o in objs:
                files.append(dict(key=o["Key"], size=o["Size"], lm=o["LastModified"],
                                  src=c, kind="gts-" + c.split("-")[1], a1=a1))
    for c in centres:
        if c in GW:
            continue
        pre = f"data/{c}/data/core/weather/surface-based-observations/temp/"
        _, objs = ls(pre, delim=None)
        if objs:
            print(f"== native {pre}: {len(objs)} objects", flush=True)
            for o in objs:
                files.append(dict(key=o["Key"], size=o["Size"], lm=o["LastModified"], src=c,
                                  kind="native", a1="native"))
    for pre in NATIVE_EXTRA:
        _, objs = ls(pre, delim=None)
        print(f"== native {pre}: {len(objs)} objects", flush=True)
        for o in objs:
            files.append(dict(key=o["Key"], size=o["Size"], lm=o["LastModified"], src=pre.split("/")[1],
                              kind="native", a1="drop"))
    return centres, files


def main():
    t0 = time.time()
    jma_cache()
    centres, files = targeted()
    leaves = []
    if not files:
        print("no upper-air files found", flush=True)
    lms = [f["lm"] for f in files]
    print(f"== {len(files)} upper-air objects listed; oldest LM {min(lms) if lms else None}, "
          f"newest {max(lms) if lms else None}; {NREQ['list']} list requests so far", flush=True)
    by_src = Counter(f["src"] for f in files)
    print("   by source:", dict(by_src), flush=True)
    # retention histogram by hours ago
    ages = Counter(int((NOW - f["lm"]).total_seconds() // 21600) * 6 for f in files)
    print("   age histogram (h ago bucket: n):", dict(sorted(ages.items())), flush=True)
    sizes = [f["size"] for f in files]
    if sizes:
        print(f"   size: median {stx.median(sizes):.0f} B, p90 {pct(sizes, .9)} B, max {max(sizes)} B, "
              f"total {sum(sizes) / 1e6:.1f} MB", flush=True)
    # A1 breakdown in the GTS tree
    a1c = Counter((f["kind"], f["a1"]) for f in files)
    print("   by kind/A1:", dict(a1c), flush=True)
    a24 = defaultdict(lambda: [0, 0])
    for f in files:
        if f["lm"] >= NOW - timedelta(hours=24):
            a24[(f["kind"], f["a1"])][0] += 1
            a24[(f["kind"], f["a1"])][1] += f["size"]
    print("   last 24 h by kind/A1 (files, MB):", {k: (v[0], round(v[1] / 1e6, 1)) for k, v in a24.items()})
    per_src = Counter(f["src"] for f in files if f["kind"] == "native" and f["lm"] >= NOW - timedelta(hours=24))
    print("   native last 24 h per centre:", dict(per_src))

    # last 24 h
    win = [f for f in files if f["lm"] >= NOW - timedelta(hours=24) and f["a1"] in ("K", "S", "native", "drop")]
    win.sort(key=lambda f: f["key"])
    capped, seen_src = [], Counter()
    for f in sorted(win, key=lambda f: f["lm"], reverse=True):
        if f["kind"] == "native":
            seen_src[f["src"]] += 1
            if seen_src[f["src"]] > 1200:
                continue
        capped.append(f)
    win = sorted(capped, key=lambda f: f["key"])
    tot = sum(f["size"] for f in win)
    print(f"== last 24 h: {len(win)} files, {tot / 1e6:.1f} MB", flush=True)
    take = win
    if len(win) > MAX_DL or tot > MAX_BYTES:
        k = max(len(win) / MAX_DL, tot / MAX_BYTES)
        take = win[::int(k) + 1]
        print(f"   sampling every {int(k) + 1}th file -> {len(take)}", flush=True)

    def work(f, b):
        if b is None:
            return f, [], "fetch failed"
        if not b[:4] == b"BUFR" and b"BUFR" not in b[:200]:
            return f, [], f"not BUFR ({b[:8]!r})"
        subs, err = decode(b)
        return f, subs, err

    recs, errs = [], Counter()
    NREQ["get"] = len(take)
    with ThreadPoolExecutor(4) as ex:
        blobs = ex.map(lambda f: fetch(f["key"]), take)       # downloads in 4 threads, decode serially
        for i, (f, b) in enumerate(zip(take, blobs)):
            f, subs, err = work(f, b)
            if err:
                errs[err[:60]] += 1
            for s in subs:
                s.update(key=f["key"], src=f["src"], kind=f["kind"], a1=f["a1"], lm=f["lm"].isoformat(),
                         size=f["size"])
                nom = nominal_from_name(f["key"], f["lm"])
                s["nom"] = nom.isoformat() if nom else None
                recs.append(s)
            if i % 1000 == 0:
                print(f"   ...{i}/{len(take)} decoded, {len(recs)} subsets", flush=True)
    print(f"== decoded {len(recs)} subsets from {len(take)} files; errors: {errs.most_common(8)}", flush=True)
    errsub = Counter(r["err"] for r in recs if "err" in r)
    if errsub:
        print("   subset errors:", errsub.most_common(5), flush=True)

    report(recs, files, take)
    summary = {"generated": NOW.isoformat(), "nreq": dict(NREQ), "centres": centres,
               "leaves": [list(map(str, l)) for l in leaves],
               "files": [dict(key=f["key"], size=f["size"], lm=f["lm"].isoformat(), src=f["src"])
                         for f in files],
               "recs": recs}
    with gzip.open(OUT / "wis2_survey.json.gz", "wt") as fh:
        json.dump(summary, fh, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"== done in {time.time() - t0:.0f} s; requests {dict(NREQ)}", flush=True)


def report(recs, files, take):
    good = [r for r in recs if r.get("t") and (r.get("wmo") or r.get("wsi"))]
    for r in good:
        r["sid"] = r.get("wmo") or ("W:" + r["wsi"])
        r["bucket"] = synop_bucket(datetime.fromisoformat(r["t"]))
    # restrict to launches in the last 24 h (+ a little slack)
    rec24 = [r for r in good if r["bucket"] >= NOW - timedelta(hours=27)]
    print(f"\n######## REPORT ({len(good)} identified subsets; {len(rec24)} with launch in last ~24 h)")
    tm = Counter(tuple(r["tmpl"]) for r in recs if "tmpl" in r)
    print("templates (first descriptors):", tm.most_common(12))
    wsi_only = {r["sid"] for r in good if not r.get("wmo")}
    wsi_non20000 = {r["wsi"] for r in good if r.get("wsi") and not r["wsi"].startswith("0-20000-0-")}
    print(f"stations with WIGOS id only (no block/station): {len(wsi_only)}; WSIs not 0-20000-0-: "
          f"{sorted(wsi_non20000)[:15]}")

    st = defaultdict(lambda: {"b": set(), "src": set(), "nlev": 0, "hires": False, "T": 0, "Td": 0, "W": 0})
    for r in rec24:
        s = st[r["sid"]]
        s["b"].add(r["bucket"].hour)
        s["src"].add(r["src"])
        s["nlev"] = max(s["nlev"], r["nlev"])
        s["T"] |= r["nT"] > 0
        s["Td"] |= r["nTd"] > 0
        s["W"] |= r["nW"] > 0
    print(f"\ndistinct stations (launch in last ~24 h): {len(st)}")
    reg = defaultdict(lambda: Counter())
    for sid, s in st.items():
        g = region(sid if sid[0].isdigit() else None)
        reg[g]["n"] += 1
        reg[g]["00Z"] += 0 in s["b"]
        reg[g]["12Z"] += 12 in s["b"]
        reg[g]["both"] += (0 in s["b"] and 12 in s["b"])
        reg[g]["06/18"] += bool({6, 18} & s["b"])
        reg[g]["hires>=150"] += s["nlev"] >= 150
    print(f"{'region':28s} {'n':>5s} {'00Z':>5s} {'12Z':>5s} {'both':>5s} {'06/18':>6s} {'hires':>6s}")
    for g, c in sorted(reg.items(), key=lambda x: -x[1]["n"]):
        print(f"{g:28s} {c['n']:5d} {c['00Z']:5d} {c['12Z']:5d} {c['both']:5d} {c['06/18']:6d} {c['hires>=150']:6d}")
    print("stations per source:", Counter(src for s in st.values() for src in s["src"]))
    print("T/Td/W available (stations):", sum(s["T"] for s in st.values()), sum(s["Td"] for s in st.values()),
          sum(s["W"] for s in st.values()))

    # level counts per subset (best per station/bucket)
    best = {}
    for r in rec24:
        k = (r["sid"], r["bucket"])
        if k not in best or r["nlev"] > best[k]["nlev"]:
            best[k] = r
    lv = [r["nlev"] for r in best.values()]
    hist = Counter(min(x // 50, 20) * 50 for x in lv)
    print(f"\nsoundings (station x synoptic time, best message): {len(best)}")
    print("max-level histogram (bin start: n):", dict(sorted(hist.items())))
    print(f"  >=150 levels: {sum(x >= 150 for x in lv)}, 30-149: {sum(30 <= x < 150 for x in lv)}, "
          f"<30: {sum(x < 30 for x in lv)}")
    ptop = [r["ptop"] for r in best.values() if r["ptop"]]
    if ptop:
        print(f"top pressure hPa: median {stx.median(ptop):.1f}, p90 {pct(ptop, .9)}, "
              f"share reaching <=100 hPa {sum(p <= 100 for p in ptop) / len(ptop):.0%}, "
              f"<=10 hPa {sum(p <= 10 for p in ptop) / len(ptop):.0%}")
    # per-message level distribution (parts vs full)
    mlev = Counter(min(r["nlev"] // 25, 12) * 25 for r in rec24)
    print("per-message level histogram:", dict(sorted(mlev.items())))
    # hires stations by region
    # latency
    lat_nom, lat_launch = defaultdict(list), defaultdict(list)
    for r in rec24:
        lm = datetime.fromisoformat(r["lm"])
        if r.get("nom"):
            lat_nom[r["kind"]].append((lm - datetime.fromisoformat(r["nom"])).total_seconds() / 60)
        lat_launch[r["kind"] + ("-hires" if r["nlev"] >= 150 else "-std")].append(
            (lm - datetime.fromisoformat(r["t"])).total_seconds() / 60)
    for k, xs in lat_nom.items():
        print(f"latency LastModified - nominal (filename) [{k}] min: n={len(xs)} p10 {pct(xs, .1):.0f} "
              f"median {pct(xs, .5):.0f} p90 {pct(xs, .9):.0f} p99 {pct(xs, .99):.0f}")
    for k, xs in lat_launch.items():
        print(f"latency LastModified - BUFR time [{k}] min: n={len(xs)} p10 {pct(xs, .1):.0f} "
              f"median {pct(xs, .5):.0f} p90 {pct(xs, .9):.0f}")
    # first availability of the best (complete) sounding relative to synoptic time
    first = []
    for (sid, b), r in best.items():
        first.append((datetime.fromisoformat(r["lm"]) - b).total_seconds() / 60)
    print(f"best message arrival after synoptic hour (min): p10 {pct(first, .1):.0f} median {pct(first, .5):.0f} "
          f"p90 {pct(first, .9):.0f}")

    # duplicates
    srcs = defaultdict(set)
    nmsg = Counter()
    for r in rec24:
        srcs[(r["sid"], r["bucket"])].add(r["kind"] + ":" + r["src"])
        nmsg[(r["sid"], r["bucket"])] += 1
    both = [k for k, v in srcs.items() if any(x.startswith("gts") for x in v) and any(x.startswith("native") for x in v)]
    print(f"\nduplicates: {len(both)} station-times in BOTH the GTS gateway and a native centre; "
          f"native-only {sum(all(x.startswith('native') for x in v) for v in srcs.values())}, "
          f"gts-only {sum(all(x.startswith('gts') for x in v) for v in srcs.values())}")
    print("messages per station-time:", dict(sorted(Counter(min(v, 10) for v in nmsg.values()).items())))
    kinds = defaultdict(set)
    for r in rec24:
        kinds[(r["sid"], r["bucket"])].add(r["kind"])
    print("kind combos per station-time:", Counter(tuple(sorted(v)) for v in kinds.values()).most_common())
    for kk in ("gts-dwd", "gts-jma", "native"):
        print(f"  station-times reachable via {kk}: {sum(kk in v for v in kinds.values())}; "
              f"ONLY via {kk}: {sum(v == {kk} for v in kinds.values())}")
    # A1 split: TAC-converted K parts vs S entire soundings
    a1lev = defaultdict(list)
    for r in rec24:
        a1lev[r["a1"]].append(r["nlev"])
    for a, xs in a1lev.items():
        print(f"  levels per message [{a}]: n={len(xs)} median {stx.median(xs)} p10 {pct(xs, .1)} p90 {pct(xs, .9)}; "
              f">=150: {sum(x >= 150 for x in xs)}")
    # first arrival of any message vs of the best one, per station-time
    fa, fb = [], []
    for k, r in best.items():
        ts = [datetime.fromisoformat(x["lm"]) for x in rec24 if (x["sid"], x["bucket"]) == k]
        fa.append((min(ts) - k[1]).total_seconds() / 60)
    print(f"first arrival (any message) after synoptic hour, min: p10 {pct(fa, .1):.0f} median {pct(fa, .5):.0f} "
          f"p90 {pct(fa, .9):.0f}")
    combos = Counter(tuple(sorted(v)) for v in srcs.values())
    print("source combos:", combos.most_common(12))
    # gts vs native level comparison for overlap
    cmpl = []
    for k in both:
        g = max((r["nlev"] for r in rec24 if (r["sid"], r["bucket"]) == k and r["kind"].startswith("gts")), default=0)
        n = max((r["nlev"] for r in rec24 if (r["sid"], r["bucket"]) == k and r["kind"] == "native"), default=0)
        cmpl.append((g, n))
    if cmpl:
        print(f"  overlap: gts max levels median {stx.median(c[0] for c in cmpl)}, native median "
              f"{stx.median(c[1] for c in cmpl)}")

    # compare with the explorer
    sj = json.loads(Path("skewt/stations.json").read_text())["stations"]
    by_wmo = {s["id"]: s for s in sj if s.get("id")}
    iem = set(json.loads(Path("skewt/iem_raob.json").read_text()))
    man = {}
    mp = Path("prev-data/manifest.json")
    if mp.exists():
        man = json.loads(mp.read_text()).get("entries", {})
    igra_active = {k for k, e in man.items() if e.get("src") == "IGRA"}
    iem_active = {k for k, e in man.items() if e.get("src") == "IEM"}
    sj_recent = {s["id"] for s in sj if s.get("id") and s.get("y1", 0) >= 2026} - iem
    got = {sid for sid in st}
    print(f"\nexplorer: manifest IGRA-active {len(igra_active)}, IEM-active {len(iem_active)}; "
          f"stations.json y1>=2026 outside IEM list: {len(sj_recent)}")
    print(f"  IGRA-active covered by WIS2 (last 24 h): {len(igra_active & got)} / {len(igra_active)}")
    print(f"  stations.json 2026 non-IEM covered: {len(sj_recent & got)} / {len(sj_recent)}")
    print(f"  IEM-active also in WIS2: {len(iem_active & got)} / {len(iem_active)}")
    miss = igra_active - got
    mreg = Counter(region(m) for m in miss)
    print("  IGRA-active NOT in WIS2, by region:", mreg.most_common())
    print("  examples missing:", [(m, by_wmo.get(m, {}).get("n")) for m in sorted(miss)[:40]])
    newst = sorted(got - set(by_wmo))
    print(f"  WIS2 stations not in stations.json: {len(newst)}: {newst[:40]}")
    extra = (got & set(by_wmo)) - igra_active - iem_active
    print(f"  WIS2 stations in stations.json but not currently on the map: {len(extra)}: "
          f"{sorted(extra)[:40]}")
    for g in ("South America", "Africa", "SE Asia", "Oceania", "Antarctica"):
        have = sorted(x for x in got if x[0].isdigit() and region(x) == g)
        print(f"  {g}: {have}")

    # example files
    print("\nexample files:")
    seen = set()
    for r in sorted(rec24, key=lambda r: -r["nlev"]):
        tag = (r["kind"], r["src"], r["nlev"] >= 150)
        if tag in seen:
            continue
        seen.add(tag)
        print(f"  [{r['kind']}/{r['src']} nlev={r['nlev']} ptop={r['ptop']} sid={r['sid']} t={r['t']} "
              f"size={r['size']}] {r['key']}")
    for r in rec24[:3]:
        print(f"  [sample] {r['key']} -> {r['sid']} nlev {r['nlev']} nsub {r['nsub']} tmpl {r['tmpl']}")

    # polling cost
    gtsfiles = [f for f in files if f["src"].endswith("gts-to-wis2") or "gts" in f["src"]]
    per_h = len([f for f in files if f["lm"] >= NOW - timedelta(hours=24)]) / 24
    by_hr = Counter(f["lm"].hour for f in files if f["lm"] >= NOW - timedelta(hours=24))
    print(f"\npolling: {len(files)} upper-air objects in the cache now; ~{per_h:.0f} new files/h "
          f"(by LM hour: {dict(sorted(by_hr.items()))})")
    bh = defaultdict(int)
    for f in files:
        if f["lm"] >= NOW - timedelta(hours=24):
            bh[f["lm"].hour] += f["size"]
    print("   bytes per LM hour (kB):", {h: round(v / 1e3) for h, v in sorted(bh.items())})


if __name__ == "__main__":
    main()
