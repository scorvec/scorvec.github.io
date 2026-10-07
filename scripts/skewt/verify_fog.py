#!/usr/bin/env python3
"""Verify the Skew-T explorer's fog / ceiling estimate against METARs (runs in GitHub Actions only).

Stages (skewt-fog-verify.yml runs them in order):
    probe    OUT            what the sources look like (formats, level counts, timing)
    stations OUT            candidate raob sites, co-location check against the METAR site (<= 5 km)
    metar    OUT            IEM ASOS routine METARs 2010-2025, one station at a time, polite + backed off
    build    OUT            IGRA v2 POR files -> faithful port of the app's profile handling (fogport.py) ->
                            old / new / swept rules + fog-model probability per launch, matched to the METAR
    jstest   OUT APP OLD    the Python port vs the real app.js functions under node (IGRA records + synthetic)
    score    OUT            contingency scores, station-block bootstrap CIs, sweeps (fit 2010-19, test 2020-25)

Truth: fog = visibility <= 1 km (vsby <= 0.63 SM, i.e. <= 5/8 SM) or FG/FZFG in present weather; ceiling =
lowest BKN/OVC/VV base. METAR = the routine report in [nominal-90, nominal] min closest to nominal-45.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fogport as FP  # noqa: E402

UA = "scorvec.com skew-t fog verification (+https://scorvec.com/skewt/)"
IGRA = "https://www.ncei.noaa.gov/pub/data/igra/"
ISD_HIST = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
IEM = "https://mesonet.agron.iastate.edu"
Y0, Y1 = 2010, 2025
FIT_LAST = 2019
MAX_KM = 5.0

CANDIDATES = {   # WMO -> METAR ICAO (co-location verified from coordinates at run time)
    "72493": "KOAK", "72694": "KSLE", "72528": "KBUF", "72426": "KILN", "72235": "KJAN",
    "72240": "KLCH", "72248": "KSHV", "72251": "KCRP", "72250": "KBRO", "72363": "KAMA",
    "72451": "KDDC", "72456": "KTOP", "72645": "KGRB", "72747": "KINL", "72208": "KCHS",
    "72214": "KTLH", "72317": "KGSO", "72403": "KIAD", "72518": "KALB", "72712": "KCAR",
    "72476": "KGJT", "72681": "KBOI", "72597": "KMFR", "72797": "KUIL", "70273": "PANC",
    "70261": "PAFA", "70026": "PABR", "72764": "KBIS", "72659": "KABR",
    "87576": "SAEZ", "08221": "LEMD", "16080": "LIML",
}

# sweep grid for the new rule
SATS = [95, 96, 97, 98, 99]
DEPTHS = [0, 15, 30, 60, 100]
NEARS = [200, 300, 500]
BARS = [90, 92, 94, 96, 98]
SHIP = (97, 30, 300)


def get(url: str, timeout: int = 300, tries: int = 3) -> bytes:
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}),
                                          timeout=timeout).read()
        except Exception as e:  # noqa: BLE001
            print(f"  GET failed ({e}) {url}", flush=True)
            if k == tries - 1:
                raise
            time.sleep(30 * (k + 1))
    raise RuntimeError


def hav_km(la1, lo1, la2, lo2):
    r = math.pi / 180
    a = (math.sin((la2 - la1) * r / 2) ** 2 +
         math.cos(la1 * r) * math.cos(la2 * r) * math.sin((lo2 - lo1) * r / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


# ------------------------------------------------------------- stations ---
def stations(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sl = get(IGRA + "igra2-station-list.txt").decode()
    igra = {}
    for line in sl.splitlines():
        w = line[6:11]
        if w in CANDIDATES:
            igra[w] = {"gid": line[0:11], "la": float(line[12:20]), "lo": float(line[21:30]),
                       "el": float(line[31:37]), "name": line[41:71].strip(),
                       "y0": int(line[72:76]), "y1": int(line[77:81])}
    isd = {}
    for row in csv.reader(io.StringIO(get(ISD_HIST).decode())):
        if len(row) < 11 or row[5] not in CANDIDATES.values() or not row[6] or not row[7]:
            continue
        rec = {"la": float(row[6]), "lo": float(row[7]), "el": float(row[8] or "nan"), "end": row[10],
               "name": row[2]}
        if row[5] not in isd or rec["end"] > isd[row[5]]["end"]:
            isd[row[5]] = rec                    # the current location record
    table, rep = [], ["| WMO | ICAO | IGRA name | km | dz m | IGRA years | kept |", "|---|---|---|---|---|---|---|"]
    for w, icao in CANDIDATES.items():
        g, m = igra.get(w), isd.get(icao)
        if not g or not m:
            rep.append(f"| {w} | {icao} | missing ({'IGRA' if not g else 'ISD'}) | | | | no |")
            continue
        km = hav_km(g["la"], g["lo"], m["la"], m["lo"])
        dz = g["el"] - m["el"]
        keep = km <= MAX_KM and g["y1"] >= Y0 + 3 and abs(dz) <= 50
        rep.append(f"| {w} | {icao} | {g['name']} | {km:.1f} | {dz:+.0f} | {g['y0']}-{g['y1']} | "
                   f"{'yes' if keep else 'no'} |")
        table.append({"wmo": w, "icao": icao, **g, "m_la": m["la"], "m_lo": m["lo"], "m_el": m["el"],
                      "km": round(km, 2), "dz": round(dz, 1), "keep": keep})
    (out / "stations.json").write_text(json.dumps(table, indent=1))
    (out / "stations.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep), flush=True)


# --------------------------------------------------------------- METARs ---
ASOS_Q = ("/cgi-bin/request/asos.py?station={st}&data=vsby&data=skyc1&data=skyc2&data=skyc3&data=skyc4"
          "&data=skyl1&data=skyl2&data=skyl3&data=skyl4&data=wxcodes&year1={y0}&month1=1&day1=1&year2={y1}"
          "&month2=1&day2=1&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3")


def iem_get(url: str, deadline_s: int = 600) -> bytes | None:
    """Sequential, polite: back off 60/180/600 s on throttling; give up on the station after that."""
    for k, wait in enumerate([60, 180, 600, None]):
        t0 = time.time()
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120)
            buf = bytearray()
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                if time.time() - t0 > deadline_s:
                    raise TimeoutError("overall deadline")
            return bytes(buf)
        except urllib.error.HTTPError as e:
            print(f"  IEM HTTP {e.code}; attempt {k + 1}", flush=True)
            if e.code not in (429, 500, 502, 503, 504) or wait is None:
                return None
        except Exception as e:  # noqa: BLE001
            print(f"  IEM error {e}; attempt {k + 1}", flush=True)
            if wait is None:
                return None
        time.sleep(wait)
    return None


def metar(out: Path) -> None:
    st = [s for s in json.loads((out / "stations.json").read_text()) if s["keep"]]
    md = out / "metar"
    md.mkdir(parents=True, exist_ok=True)
    fails = 0
    for s in st:
        f = md / f"{s['icao']}.csv"
        if f.exists() and f.stat().st_size > 1000:
            continue
        parts = []
        for y0 in range(Y0, Y1 + 1, 4):
            y1 = min(y0 + 4, Y1 + 1)
            b = iem_get(IEM + ASOS_Q.format(st=s["icao"], y0=y0, y1=y1))
            if b is None:
                fails += 1
                print(f"  {s['icao']} {y0}-{y1}: FAILED", flush=True)
                parts = None
                break
            txt = b.decode(errors="replace").splitlines()
            parts.extend(txt[1:] if parts else txt)
            print(f"  {s['icao']} {y0}-{y1 - 1}: {len(txt) - 1} rows", flush=True)
            time.sleep(5)
        if parts is not None:
            f.write_text("\n".join(parts) + "\n")
        if fails >= 3:
            print("  IEM failing repeatedly - stopping METAR pulls (back off, retry another day)", flush=True)
            break
        time.sleep(10)


def _fnum(s):
    try:
        return float(s)
    except ValueError:
        return None


def load_metars(path: Path, intl: bool):
    """-> sorted list of (epoch_min, vis_sm|None, ceil_ft|None|'none', sky_ok, wx tokens)."""
    rows = []
    with open(path) as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        ix = {k: i for i, k in enumerate(hdr)}
        for r in rd:
            if len(r) != len(hdr):
                continue
            try:
                t = datetime.strptime(r[ix["valid"]], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            vis = _fnum(r[ix["vsby"]])
            if vis is not None and (vis < 0 or vis > 100):
                vis = None
            covers = [(r[ix[f"skyc{k}"]].strip(), _fnum(r[ix[f"skyl{k}"]])) for k in range(1, 5)]
            known = [c for c in covers if c[0] not in ("M", "")]
            ceil, sky_ok = None, True
            if not known:
                # international CAVOK/NSC carry no cloud groups; accept only with CAVOK-like visibility
                sky_ok = bool(intl and vis is not None and vis >= 6.2)
            else:
                bases = []
                for c, h in known:
                    if c in ("BKN", "OVC", "VV"):
                        if h is None:
                            sky_ok = False
                        else:
                            bases.append(h)
                    elif c not in ("CLR", "SKC", "NSC", "NCD", "FEW", "SCT"):
                        sky_ok = False
                ceil = min(bases) if bases else None
            wx = r[ix["wxcodes"]].strip()
            toks = [] if wx in ("M", "") else wx.split()
            rows.append((int(t.timestamp() // 60), vis, ceil, sky_ok, toks))
    rows.sort(key=lambda a: a[0])
    return rows


def pick_metar(rows, times, nominal_min: int):
    import bisect
    lo = bisect.bisect_left(times, nominal_min - 90)
    hi = bisect.bisect_right(times, nominal_min)
    best = None
    for k in range(lo, hi):
        d = abs(times[k] - (nominal_min - 45))
        if best is None or d < best[0]:            # ties keep the earlier report
            best = (d, k)
    return None if best is None else rows[best[1]]


def fog_tokens(toks):
    core = [t.lstrip("+-") for t in toks]
    fg = any(t in ("FG", "FZFG") for t in core)
    shallow = any(t in ("MIFG", "BCFG", "PRFG", "VCFG") for t in core)
    br = any(t == "BR" for t in core)
    return fg, shallow, br


# ---------------------------------------------------------------- build ---
def iter_records(zpath: Path):
    z = zipfile.ZipFile(zpath)
    with z.open(z.namelist()[0]) as fh:
        cur = None
        for raw in io.TextIOWrapper(fh, encoding="ascii", errors="replace"):
            line = raw.rstrip("\n")
            if line.startswith("#"):
                if cur:
                    yield cur
                yr = int(line[13:17])
                cur = [line] if Y0 <= yr <= Y1 and line[24:26] in ("00", "12") else None
            elif cur is not None:
                cur.append(line)
        if cur:
            yield cur


def low_resolution(p):
    z0 = p["H"][0]
    return sum(1 for h in p["H"][1:] if h - z0 <= 500)


def process_station(args):
    s, out = args
    import numpy as np
    zpath = out / "igra" / f"{s['gid']}-data.txt.zip"
    intl = not s["gid"].startswith("US")
    mrows = load_metars(out / "metar" / f"{s['icao']}.csv", intl)
    mtimes = [r[0] for r in mrows]
    rows, profs, cases = [], [], []
    qc = {"records": 0, "parse_fail": 0, "nan_pressure": 0, "no_sfc_flag": 0, "missing_sfc_TTd": 0, "no_metar": 0, "kept": 0}
    rng = random.Random(int(s["wmo"]))
    for rec in iter_records(zpath):
        qc["records"] += 1
        h = rec[0]
        ymd = f"{h[13:17]}-{h[18:20]}-{h[21:23]}"
        hour = int(h[24:26])
        rel = h[27:31]
        psrc = h[37:45].strip()
        got = FP.parse_igra_record(rec, s["el"])
        if rng.random() < 0.012 and len(cases) < 150:      # sample for the node check (incl. later-dropped)
            cases.append({"id": f"{s['wmo']}_{ymd}_{hour:02d}", "text": "\n".join(rec), "gid": s["gid"],
                          "ymd": ymd, "hour": hour, "elev": s["el"]})
        if got is None:
            qc["parse_fail"] += 1
            continue
        prof, info = got
        if info["nan_p"]:                 # a >44 km height with no pressure: JS order undefined, drop
            qc["nan_pressure"] += 1
            continue
        if not info["sfc_flag"]:
            qc["no_sfc_flag"] += 1
            continue
        if not (FP.isf(info["raw_t0"]) and FP.isf(info["raw_d0"])):
            qc["missing_sfc_TTd"] += 1
            continue
        nom = datetime.strptime(ymd, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(hours=hour)
        m = pick_metar(mrows, mtimes, int(nom.timestamp() // 60))
        if m is None:
            qc["no_metar"] += 1
            continue
        p = FP.prepare(prof)
        res = FP.cloud_layers(p)
        if len(res["levs"]) < 3:
            qc["parse_fail"] += 1
            continue
        qc["kept"] += 1
        oc = FP.est_ceiling_old(res)
        nc = FP.est_ceiling_new(res)
        pf = FP.fog_prob(p, FP.doy_of(ymd), hour)
        fg, shallow, br = fog_tokens(m[4])
        vis = m[1]
        row = {"wmo": s["wmo"], "date": ymd, "hour": hour, "year": int(ymd[:4]), "psrc": psrc,
               "rel": rel, "nlev": len(p["P"]), "n500": low_resolution(p),
               "dz1": round(p["H"][1] - p["H"][0], 1),
               "t0": round(p["T"][0] - 273.15, 2), "dd0": round(p["T"][0] - p["D"][0], 2),
               "pfog": pf, "m_off": m[0] - int(nom.timestamp() // 60), "vis": vis,
               "m_ceil": m[2], "sky_ok": m[3], "fg": fg, "fg_shallow": shallow, "br": br,
               "wx": " ".join(m[4]),
               "old_obsc": bool(oc and oc.get("obscured")),
               "old_zb": None if (not oc or oc.get("none")) else oc["zb"],
               "old_conf": None if (not oc or oc.get("none")) else oc["conf"],
               "new_conf": None if (not nc or nc.get("none")) else nc["conf"],
               "new_mist": bool(nc and nc.get("mist")),
               }
        # sweep: fog depends on (SAT, DEPTH); ceilings on (SAT, DEPTH, NEAR); BAR sweep at the shipped rule
        for sat in SATS:
            for dep in DEPTHS:
                for near in NEARS:
                    c = FP.est_ceiling_new(res, sat, dep, near, 94)
                    if near == NEARS[0]:
                        row[f"fog_{sat}_{dep}"] = bool(c and c.get("obscured"))
                    row[f"zb_{sat}_{dep}_{near}"] = (None if (not c or c.get("none")) else c["zb"])
        for bar in BARS:
            c = FP.est_ceiling_new(res, *SHIP, bar)
            row[f"zbbar_{bar}"] = None if (not c or c.get("none")) else c["zb"]
        rows.append(row)
        # compact profile store (<= 6 km AGL) for reuse
        z0 = p["H"][0]
        k = sum(1 for hh in p["H"] if hh - z0 <= 6000)
        profs.append(np.array([p[x][:k] for x in "PHTDUV"], dtype=np.float32))
    import pandas as pd
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c.startswith(("zb", "pfog", "vis", "m_ceil", "old_zb")):
            df[c] = df[c].astype("float32")
    np.savez_compressed(out / "profiles" / f"{s['wmo']}.npz",
                        offsets=np.cumsum([0] + [a.shape[1] for a in profs]),
                        data=np.concatenate(profs, axis=1) if profs else np.zeros((6, 0), np.float32),
                        key=np.array([f"{r['date']}_{r['hour']:02d}" for r in rows]))
    return s["wmo"], df, qc, cases


def build(out: Path) -> None:
    import pandas as pd
    from multiprocessing import Pool
    st = [s for s in json.loads((out / "stations.json").read_text()) if s["keep"]
          and (out / "metar" / f"{s['icao']}.csv").exists()]
    (out / "igra").mkdir(exist_ok=True)
    (out / "profiles").mkdir(exist_ok=True)
    for s in st:
        f = out / "igra" / f"{s['gid']}-data.txt.zip"
        if not f.exists():
            f.write_bytes(get(IGRA + f"data/data-por/{s['gid']}-data.txt.zip"))
            print(f"  IGRA {s['gid']}: {f.stat().st_size >> 20} MB", flush=True)
    frames, qcs, cases = [], {}, []
    with Pool(min(4, os.cpu_count() or 2)) as pool:
        for wmo, rows, qc, cs in pool.imap_unordered(process_station, [(s, out) for s in st]):
            print(f"  {wmo}: {qc}", flush=True)
            frames.append(rows)
            qcs[wmo] = qc
            cases += cs
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out / "pairs.parquet", index=False)
    (out / "qc.json").write_text(json.dumps(qcs, indent=1))
    (out / "jstest_cases.json").write_text(json.dumps(cases + synthetic_cases()))
    print(f"  pairs: {len(df)}", flush=True)


# --------------------------------------------------------------- jstest ---
def synthetic_cases():
    """Hand-built profiles hitting the rule's edges (no data involved)."""
    def mk(levels, ymd="2024-01-15", hour=12):
        # levels: (z m AGL, T C, Td C); pressure hydrostatic from 1000 hPa, light winds
        P, H, T, D, U, V = [], [], [], [], [], []
        p = 100000.0
        for k, (z, t, d) in enumerate(levels):
            if k:
                dz = z - levels[k - 1][0]
                p *= math.exp(-9.80665 * dz / (287.05 * (273.15 + 0.5 * (t + levels[k - 1][1]))))
            P.append(p); H.append(100 + z); T.append(t + 273.15); D.append(d + 273.15)
            U.append(1.0 + z / 1000); V.append(0.5)
        return {"P": P, "H": H, "T": T, "D": D, "U": U, "V": V}
    up = [(z, 8 - z * 0.0065, -20 - z * 0.002) for z in range(1500, 12001, 500)]
    cases = []
    hr = [(z, 5.0 + 0.004 * z, 5.0 + 0.004 * z - (0.1 if z < 80 else 3)) for z in range(0, 300, 5)]
    cases.append(("deep_fog_hires", hr + up))
    cases.append(("mist_sfc_only", [(0, 5, 4.4), (40, 5.2, 4.0), (300, 6, 0)] + up))
    cases.append(("sfc_wet_one_level", [(0, 5, 4.9), (80, 5.5, 2), (500, 4, -5)] + up))
    cases.append(("ice_cold_sfc", [(0, -25, -27.5), (30, -24, -26.5), (200, -22, -26), (900, -26, -40)] + up))
    cases.append(("stratus_400m", [(0, 12, 9), (200, 11, 9.5), (400, 10, 9.9), (450, 9.8, 9.8), (700, 8, 7.9),
                                   (900, 9, -5)] + up))
    cases.append(("low_deck_sfc_layer", [(0, 10, 9.4), (60, 9.8, 9.6), (120, 9.6, 9.55), (160, 9.5, 9.48),
                                         (250, 9.3, 9.2), (600, 9, 0)] + up))
    cases.append(("dry", [(0, 25, 5), (500, 20, 3), (1000, 16, 0)] + up))
    return [{"id": "syn_" + n, "prof": mk(lv), "ymd": "2024-01-15", "hour": 12} for n, lv in cases]


def py_case(c):
    if "text" in c:
        got = FP.parse_igra_record(c["text"].split("\n"), c["elev"])
        if got is None:
            return {"parsed": False}
        prof = got[0]
    else:
        prof = {k: list(v) for k, v in c["prof"].items()}
    p = FP.prepare(prof)
    res = FP.cloud_layers(p)
    return {"parsed": True, "n": len(p["P"]), "H": p["H"], "D": p["D"],
            "layers": [[l["base"], l["top"], l["maxRh"]] for l in res["layers"]],
            "newc": FP.est_ceiling_new(res), "oldc": FP.est_ceiling_old(res),
            "pfog": FP.fog_prob(p, FP.doy_of(c["ymd"]), c["hour"])}


def _close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= tol * max(1, abs(a), abs(b))


def _norm_ec(e):
    if e in ("u", None):
        return ("u",)
    if e == "null" or (isinstance(e, dict) and e.get("none")):
        return ("none",)
    return ("c", bool(e.get("obscured")), e["zb"], e["conf"])


def jstest(out: Path, app: str, old: str) -> None:
    js_out = out / "jstest_js.json"
    subprocess.run(["node", str(Path(__file__).resolve().parent / "fogport_jstest.js"), app, old,
                    str(out / "jstest_cases.json"), str(js_out)], check=True)
    cases = json.loads((out / "jstest_cases.json").read_text())
    js = {r["id"]: r for r in json.loads(js_out.read_text())}
    n = bad = 0
    msgs = []
    skipped = 0
    ties = []          # last-ulp exp() differences (V8 vs glibc) landing exactly on a confidence-bin edge
    for c in cases:
        if "text" in c:
            g = FP.parse_igra_record(c["text"].split("\n"), c["elev"])
            if g is not None and g[1]["nan_p"]:
                skipped += 1               # dropped from the dataset too (NaN pressure: JS sort order undefined)
                continue
        a, b = js[c["id"]], py_case(c)
        n += 1
        probs = []
        if a["parsed"] != b["parsed"]:
            probs.append("parsed")
        elif a["parsed"]:
            if a["n"] != b["n"]:
                probs.append(f"nlev {a['n']} vs {b['n']}")
            else:
                if not all(_close(x, y) for x, y in zip(a["H"], b["H"])):
                    probs.append("H")
                if not all(_close(x, y) for x, y in zip(a["D"], b["D"])):
                    probs.append("D")
            if len(a["layers"]) != len(b["layers"]) or not all(
                    _close(x, y) for la, lb in zip(a["layers"], b["layers"]) for x, y in zip(la, lb)):
                probs.append("layers")
            for k in ("newc", "oldc"):
                ea, eb = _norm_ec(a[k]), _norm_ec(b[k])
                if (ea[0] == eb[0] == "c" and ea[1] == eb[1] and _close(ea[2], eb[2]) and ea[3] != eb[3]
                        and any(abs(l[2] - t) < 1e-9 for l in a["layers"] for t in (94, 96, 98, 100))):
                    ties.append(f"{c['id']} {k}: conf {ea[3]} vs {eb[3]} (layer peak RH on a bin edge to 1e-9)")
                elif ea[0] != eb[0] or (ea[0] == "c" and not (ea[1] == eb[1] and _close(ea[2], eb[2]) and ea[3] == eb[3])):
                    probs.append(f"{k} {ea} vs {eb}")
            if not _close(a["pfog"], b["pfog"], 1e-9):
                probs.append(f"pfog {a['pfog']} vs {b['pfog']}")
        if probs:
            bad += 1
            if len(msgs) < 40:
                msgs.append(f"{c['id']}: " + "; ".join(probs))
    syn = {c["id"]: (_norm_ec(js[c["id"]]["oldc"]), _norm_ec(js[c["id"]]["newc"]),
                     js[c["id"]]["pfog"]) for c in cases if c["id"].startswith("syn_")}
    rep = {"cases": n, "skipped_nan_pressure": skipped, "mismatched": bad, "ulp_bin_ties": ties, "examples": msgs, "synthetic_js(old,new,pfog)": syn}
    (out / "jstest.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str), flush=True)
    if bad:
        raise SystemExit(f"port mismatch on {bad}/{n} cases")


def main() -> int:
    cmd, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if cmd == "stations":
        stations(out)
    elif cmd == "metar":
        metar(out)
    elif cmd == "build":
        build(out)
    elif cmd == "jstest":
        jstest(out, sys.argv[3], sys.argv[4])
    elif cmd == "score":
        import verify_fog_score
        verify_fog_score.score(out)
    else:
        raise SystemExit(f"unknown command {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
