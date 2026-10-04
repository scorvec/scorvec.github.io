#!/usr/bin/env python3
"""Mirror recent radiosonde soundings for the Skew-T Explorer (runs in GitHub Actions).

Two public sources, both credited on the page:

  * IEM RAOB JSON (Iowa Environmental Mesonet, Iowa State University) - real-time
    North America. ONE request per launch hour returns every station that reported
    at that hour (/json/raob.py?ts=YYYYMMDDHHMI, no station filter). IEM's robots.txt
    asks for Crawl-delay: 120, so requests are spaced IEM_DELAY s apart, capped per
    run (MAX_IEM_REQ), and each hour slot is asked at most two or three times over
    its first ~12 h (state.json remembers which slots were asked and when).
  * NOAA NCEI IGRA v2 year-to-date files (Integrated Global Radiosonde Archive) -
    the rest of the world, refreshed by NCEI once a day (~21:40 UTC), so these
    launches are about a day behind. Fetched from /pub/data/igra/ (the /data/ tree
    is disallowed in ncei.noaa.gov/robots.txt; /pub/ is not). The directory listing
    is checked every IGRA_CHECK_H hours; a station's zip is downloaded only when
    its listing timestamp changed since the last run AND the station has no IEM
    launch in the last IEM_COVER_H hours (IEM already covers it in real time).

Outputs (the layout the client reads from the `skewt-data` branch):
  manifest.json             {pipeline, generated, entries: {id: {id,n,la,lo,src,dt,hours}}}
                            id = WMO number (the client's byWmo key); src = "IEM" | "IGRA"
  soundings/{id}_{YYYYMMDDHH}.csv   one per launch, kept RETAIN_H hours (carried forward)
  soundings/{id}.csv                copy of the newest launch
  state.json                fetch bookkeeping (IEM slots asked, IGRA timestamps, per-file source)

CSV columns are fixed (the client and flag_anomalies.py index them by position):
  time,longitude,latitude,pressure_hPa,geopotential height_m,temperature_C,
  dew point temperature_C,ice point temperature_C,relative humidity_%,
  humidity wrt ice_%,mixing ratio_g/kg,wind direction_degree,wind speed_m/s
Neither source carries RH / ice point / mixing ratio, so they are derived from T, Td
and p (Magnus/Bolton over water, Buck-style over ice); winds reported on a subset of
levels are interpolated in log-p onto every thermodynamic level (as the client does
for live profiles). Missing values are written as "nan".

    python scripts/skewt/mirror_soundings.py OUTDIR [PREVDIR]
Env (testing): SKEWT_MAX_IGRA (cap IGRA downloads), SKEWT_MAX_IEM_REQ, SKEWT_IEM_DELAY.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKEWT = HERE.parents[1] / "skewt"
UA = "scorvec.com skew-t explorer (+https://scorvec.com/skewt/)"
IEM_RAOB = "https://mesonet.agron.iastate.edu/json/raob.py"
IEM_NET = "https://mesonet.agron.iastate.edu/geojson/network.py?network=RAOB"
IGRA_Y2D = "https://www.ncei.noaa.gov/pub/data/igra/data/data-y2d/"
PIPELINE = 2                    # manifests without this were made by the retired feed
MAX_LEVELS = 260
RETAIN_H = 96
IEM_DELAY = float(os.environ.get("SKEWT_IEM_DELAY", 121))   # robots.txt Crawl-delay: 120
MAX_IEM_REQ = int(os.environ.get("SKEWT_MAX_IEM_REQ", 6))
MAX_IGRA = int(os.environ.get("SKEWT_MAX_IGRA", 100000))
IEM_LOOKBACK_H = 30
IEM_COVER_H = 36
IGRA_CHECK_H = 3
HEADER = ("time,longitude,latitude,pressure_hPa,geopotential height_m,temperature_C,"
          "dew point temperature_C,ice point temperature_C,relative humidity_%,"
          "humidity wrt ice_%,mixing ratio_g/kg,wind direction_degree,wind speed_m/s")
NOW = datetime.now(timezone.utc)


def get(url: str, timeout: int = 120, tries: int = 3, backoff: int = 30) -> bytes:
    """GET with an honest User-Agent; back off on 429/5xx and network errors."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for k in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == tries:
                raise
            wait = backoff * k
            ra = e.headers.get("Retry-After") if e.headers else None
            if ra and ra.isdigit():
                wait = max(wait, int(ra))
            print(f"    HTTP {e.code}, retry in {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:                                  # noqa: BLE001
            if k == tries:
                raise
            print(f"    retry {k}/{tries - 1} after {repr(e)[:60]}", flush=True)
            time.sleep(backoff * k)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- profile -> CSV
def _num(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _es(tc):                     # saturation vapour pressure over water, hPa (Bolton)
    return 6.112 * math.exp(17.67 * tc / (tc + 243.5))


def _esi(tc):                    # over ice, hPa (Magnus, Alduchov & Eskridge)
    return 6.1121 * math.exp(22.587 * tc / (tc + 273.86))


def _f(v, fmt):
    return format(v, fmt) if _num(v) else "nan"


def to_csv(levels: list, launch: datetime, la: float, lo: float, elev: float) -> str | None:
    """levels: [{p (hPa), h (m|None), t (C|None), td (C|None), wd, ws (m/s)|None, et (s)|None}]."""
    winds = sorted(((L["p"], -L["ws"] * math.sin(math.radians(L["wd"])),
                     -L["ws"] * math.cos(math.radians(L["wd"])))
                    for L in levels if _num(L.get("wd")) and _num(L.get("ws")) and L["ws"] >= 0),
                   key=lambda w: -w[0])
    thermo = sorted((L for L in levels if _num(L.get("t")) and L["p"] >= 20),
                    key=lambda L: -L["p"])

    def wind_at(p):
        if not winds:
            return None
        if p >= winds[0][0]:
            return winds[0][1:]
        for j in range(1, len(winds)):
            if winds[j][0] <= p:
                a, b = winds[j - 1], winds[j]
                if a[0] == b[0]:
                    return b[1:]
                f = math.log(a[0] / p) / math.log(a[0] / b[0])
                return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])
        return winds[-1][1:]

    rows, lastP, lastH, lastT = [], 1e9, elev, None
    for L in thermo:
        p, t = L["p"], L["t"]
        if p >= lastP:
            continue
        h = L.get("h")
        if not _num(h):          # hypsometric fill between reported heights
            tb = t if lastT is None else (t + lastT) / 2
            h = elev if not rows else lastH + 287.05 * (tb + 273.15) / 9.80665 * math.log(lastP / p)
        lastP, lastH, lastT = p, h, t
        td = L.get("td")
        rh = rhi = ice = mr = None
        if _num(td) and td <= t + 0.05:
            e = _es(td)
            rh = min(100.0, 100 * e / _es(t))
            ice = td if td >= 0 else 272.62 * math.log(e / 6.112) / (22.46 - math.log(e / 6.112))
            rhi = rh if t >= 0 else min(100.0, 100 * e / _esi(t))
            mr = 621.97 * e / max(p - e, 1e-3)
        else:
            td = None
        w = wind_at(p)
        wd = ws = None
        if w:
            u, v = w
            ws = math.hypot(u, v)
            wd = (math.degrees(math.atan2(-u, -v)) + 360) % 360 if ws > 0.05 else 0.0
        et = L.get("et")
        ts = launch + timedelta(seconds=et) if _num(et) and et >= 0 else launch
        rows.append(f"{ts:%Y-%m-%d %H:%M:%S},{lo:.4f},{la:.4f},{p:.1f},{h:.0f},{t:.1f},"
                    f"{_f(td, '.1f')},{_f(ice, '.1f')},{_f(rh, '.0f')},{_f(rhi, '.0f')},"
                    f"{_f(mr, '.2f')},{_f(wd, '.0f')},{_f(ws, '.1f')}")
    if len(rows) < 10:
        return None
    if len(rows) > MAX_LEVELS:
        k = -(-len(rows) // MAX_LEVELS)
        rows = [r for i, r in enumerate(rows) if i % k == 0 or i == len(rows) - 1]
    return HEADER + "\n" + "\n".join(rows) + "\n"


def _tag(dt: datetime) -> str:
    return f"{dt:%Y%m%d%H}"


# ---------------------------------------------------------------- stations
def load_stations():
    st = json.loads((SKEWT / "stations.json").read_text())["stations"]
    by_wmo = {s["id"]: s for s in st if s.get("id")}
    by_gid = {s["gid"]: s for s in st}
    icao2wmo = {v: k for k, v in json.loads((SKEWT / "iem_raob.json").read_text()).items()}
    return by_wmo, by_gid, icao2wmo


# ---------------------------------------------------------------- IEM (real time)
def iem_slots_due(state: dict) -> list:
    """Hour slots to ask IEM about this run, most valuable first.

    Every hour of the last IEM_LOOKBACK_H hours is a candidate (specials fly at
    15Z, 18Z...). A slot is asked once when >= 2 h old and again when >= 6 h old
    (late-arriving stations); the routine 00Z/12Z slots get a third look at
    >= 12 h. Empty off-hour slots cost one small request each.
    """
    asked = state.setdefault("iem", {})
    due = []
    h0 = NOW.replace(minute=0, second=0, microsecond=0)
    for k in range(2, IEM_LOOKBACK_H + 1):
        slot = h0 - timedelta(hours=k)
        age = (NOW - slot).total_seconds() / 3600
        prev = asked.get(f"{slot:%Y%m%d%H}", [])
        marks = [2, 6] + ([12] if slot.hour in (0, 12) else [])
        need = [m for m in marks if age >= m and not any(a >= m for a in prev)]
        if need:
            pri = (0 if slot.hour in (0, 12) else 1 if slot.hour in (6, 18) else 2,
                   0 if not prev else 1, age)
            due.append((pri, slot))
    due.sort(key=lambda x: x[0])
    # forget bookkeeping older than the look-back
    cut = f"{h0 - timedelta(hours=IEM_LOOKBACK_H + 2):%Y%m%d%H}"
    for key in [k for k in asked if k < cut]:
        del asked[key]
    return [s for _, s in due[:MAX_IEM_REQ]]


def iem_station_table(state: dict) -> dict:
    """IEM RAOB network table (id -> name/lat/lon), refreshed at most weekly."""
    net = state.get("iemnet") or {}
    if net.get("t") and (NOW - datetime.fromisoformat(net["t"])).days < 7:
        return net.get("st", {})
    try:
        feats = json.loads(get(IEM_NET, tries=2))["features"]
        st = {f["id"]: {"n": f["properties"].get("sname", f["id"]),
                        "la": f["geometry"]["coordinates"][1],
                        "lo": f["geometry"]["coordinates"][0],
                        "e": f["properties"].get("elevation") or 0} for f in feats}
        state["iemnet"] = {"t": NOW.isoformat(), "st": st}
        time.sleep(IEM_DELAY)
        return st
    except Exception as e:                                      # noqa: BLE001
        print(f"  IEM station table failed: {repr(e)[:80]}", flush=True)
        return net.get("st", {})


def run_iem(outdir: Path, state: dict, by_wmo: dict, icao2wmo: dict, meta: dict) -> dict:
    slots = iem_slots_due(state)
    got, nreq, nbytes, unmapped = {}, 0, 0, []
    for i, slot in enumerate(slots):
        if i:
            time.sleep(IEM_DELAY)
        key = f"{slot:%Y%m%d%H}"
        try:
            raw = get(f"{IEM_RAOB}?ts={slot:%Y%m%d%H}00", timeout=90)
        except Exception as e:                                  # noqa: BLE001
            print(f"  IEM {key} failed: {repr(e)[:80]}", flush=True)
            continue
        nreq += 1
        nbytes += len(raw)
        state["iem"].setdefault(key, []).append(
            round((NOW - slot).total_seconds() / 3600, 1))
        try:
            profs = json.loads(raw).get("profiles") or []
        except ValueError:
            continue
        n = 0
        for pr in profs:
            icao = str(pr.get("station", ""))
            wmo = icao2wmo.get(icao)
            if wmo and wmo in by_wmo:
                s = by_wmo[wmo]
                sid, la, lo, el, name = wmo, s["la"], s["lo"], s.get("e") or 0, s["n"]
            else:
                unmapped.append(pr)
                continue
            n += write_iem(outdir, pr, sid, la, lo, el, name, meta, got)
        print(f"  IEM {slot:%Y-%m-%d %H}Z: {len(profs)} profiles, {n} written "
              f"({len(raw) // 1024} KB)", flush=True)
    if unmapped:                 # stations iem_raob.json does not know: IEM's own table
        tbl = iem_station_table(state)
        for pr in unmapped:
            icao = str(pr.get("station", ""))
            s = tbl.get(icao)
            if s:
                write_iem(outdir, pr, icao, s["la"], s["lo"], s["e"], s["n"], meta, got)
    print(f"  IEM: {nreq} request(s), {nbytes / 1e6:.2f} MB, "
          f"{len(got)} launch file(s) written/updated", flush=True)
    meta["iem_req"], meta["iem_bytes"] = nreq, nbytes
    return got


def write_iem(outdir, pr, sid, la, lo, el, name, meta, got) -> int:
    try:
        launch = datetime.strptime(pr["valid"][:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    except (KeyError, ValueError):
        return 0
    if launch < NOW - timedelta(hours=RETAIN_H):
        return 0
    levels = []
    for L in pr.get("profile") or []:
        if not _num(L.get("pres")):
            continue
        ws = L["sknt"] * 0.514444 if _num(L.get("sknt")) else None
        levels.append({"p": L["pres"], "h": L.get("hght"), "t": L.get("tmpc"),
                       "td": L.get("dwpc"), "wd": L.get("drct"), "ws": ws})
    text = to_csv(levels, launch, la, lo, el)
    if not text:
        return 0
    fn = f"{sid}_{_tag(launch)}.csv"
    dest = outdir / "soundings" / fn
    # a later ask can carry more levels (IEM fills in); never shrink a profile
    if dest.exists() and meta["filesrc"].get(fn) == "IEM" and \
            dest.read_text().count("\n") >= text.count("\n"):
        return 0
    dest.write_text(text)
    meta["filesrc"][fn] = "IEM"
    meta["names"][sid] = {"n": name, "la": la, "lo": lo}
    got[fn] = sid
    return 1


# ---------------------------------------------------------------- IGRA (global, daily)
LIST_RE = re.compile(r'href="([A-Z0-9]{11})-data-beg(\d{4})\.txt\.zip".*?'
                     r'<td align="right">([\d-]+ [\d:]+)</td>\s*<td align="right">(\d+)</td>', re.S)


def _iv(s: str, a: int, b: int):
    try:
        v = int(s[a:b])
    except ValueError:
        return None
    return None if v in (-9999, -8888) else v


def parse_igra(text: str, since: datetime):
    """Yield (launch datetime, levels) for soundings at/after `since`."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        h = lines[i]
        if not h.startswith("#"):
            i += 1
            continue
        try:
            y, mo, d, hh = int(h[13:17]), int(h[18:20]), int(h[21:23]), int(h[24:26])
            nlev = int(h[32:36])
        except ValueError:
            i += 1
            continue
        body = lines[i + 1:i + 1 + nlev]
        i += 1 + nlev
        if hh == 99:
            continue                                  # launch hour unknown
        try:
            launch = datetime(y, mo, d, hh, tzinfo=timezone.utc)
        except ValueError:
            continue
        if launch < since:
            continue
        levels = []
        for L in body:
            p = _iv(L, 9, 15)
            if p is None:
                continue                              # height-only pibal level
            et = L[3:8].strip()
            ets = None
            if et and et not in ("-9999", "-8888"):
                try:
                    v = int(et)
                    ets = (v // 100) * 60 + v % 100
                except ValueError:
                    pass
            t = _iv(L, 22, 27)
            dp = _iv(L, 34, 39)
            rh = _iv(L, 28, 33)
            tc = t / 10 if t is not None else None
            td = None
            if tc is not None and dp is not None:
                td = tc - dp / 10
            elif tc is not None and rh is not None and rh > 0:
                g = math.log(min(100, rh / 10) / 100) + 17.67 * tc / (tc + 243.5)
                td = 243.5 * g / (17.67 - g)
            wd, ws = _iv(L, 40, 45), _iv(L, 46, 51)
            levels.append({"p": p / 100, "h": _iv(L, 16, 21), "t": tc, "td": td,
                           "wd": wd, "ws": ws / 10 if ws is not None else None, "et": ets})
        yield launch, levels


def run_igra(outdir: Path, state: dict, by_gid: dict, meta: dict) -> int:
    ig = state.setdefault("igra", {"files": {}})
    last = ig.get("checked")
    if last and (NOW - datetime.fromisoformat(last)).total_seconds() < IGRA_CHECK_H * 3600:
        print(f"  IGRA: listing checked {last[:16]}Z — next check after {IGRA_CHECK_H} h", flush=True)
        return 0
    try:
        listing = get(IGRA_Y2D, timeout=120).decode("utf-8", "replace")
    except Exception as e:                                      # noqa: BLE001
        print(f"::warning::IGRA listing failed: {repr(e)[:80]}", flush=True)
        return 0
    ig["checked"] = NOW.isoformat()
    rows = LIST_RE.findall(listing)
    # stations IEM delivered recently are already real time - skip their IGRA file
    cover = NOW - timedelta(hours=IEM_COVER_H)
    iem_recent = set()
    for fn, src in meta["filesrc"].items():
        if src == "IEM":
            sid, tag = fn[:-4].split("_")
            if datetime.strptime(tag, "%Y%m%d%H").replace(tzinfo=timezone.utc) >= cover:
                iem_recent.add(sid)
    todo = []
    for gid, beg, stamp, size in rows:
        s = by_gid.get(gid)
        if not s or not s.get("id") or s["id"] in iem_recent:
            continue
        if ig["files"].get(gid) == stamp:
            continue                                  # unchanged since our last download
        todo.append((gid, beg, stamp, int(size)))
    todo = todo[:MAX_IGRA]
    print(f"  IGRA: {len(rows)} y2d files listed, {len(todo)} changed and needed "
          f"({sum(t[3] for t in todo) / 1e6:.0f} MB)", flush=True)
    since = NOW - timedelta(hours=RETAIN_H)
    nbytes = nfiles = 0
    for gid, beg, stamp, size in todo:
        s = by_gid[gid]
        try:
            raw = get(f"{IGRA_Y2D}{gid}-data-beg{beg}.txt.zip", timeout=300, backoff=10)
        except Exception as e:                                  # noqa: BLE001
            print(f"    {gid}: {repr(e)[:70]}", flush=True)
            continue
        nbytes += len(raw)
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                text = z.read(z.namelist()[0]).decode("ascii", "replace")
        except zipfile.BadZipFile:
            continue
        ig["files"][gid] = stamp
        sid = s["id"]
        for launch, levels in parse_igra(text, since):
            fn = f"{sid}_{_tag(launch)}.csv"
            if meta["filesrc"].get(fn) == "IEM":
                continue                              # the real-time copy wins
            csv = to_csv(levels, launch, s["la"], s["lo"], s.get("e") or 0)
            if not csv:
                continue
            (outdir / "soundings" / fn).write_text(csv)
            meta["filesrc"][fn] = "IGRA"
            nfiles += 1
        time.sleep(0.2)
    print(f"  IGRA: downloaded {nbytes / 1e6:.0f} MB, wrote {nfiles} launch file(s)", flush=True)
    meta["igra_bytes"] = nbytes
    return nfiles


# ---------------------------------------------------------------- main
def main() -> int:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "skewt-data-out")
    prevdir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    (outdir / "soundings").mkdir(parents=True, exist_ok=True)
    by_wmo, by_gid, icao2wmo = load_stations()
    cutoff = NOW - timedelta(hours=RETAIN_H)

    state, prev_manifest = {}, {}
    if prevdir and (prevdir / "manifest.json").exists():
        prev_manifest = json.loads((prevdir / "manifest.json").read_text())
    if prev_manifest.get("pipeline") == PIPELINE and (prevdir / "state.json").exists():
        state = json.loads((prevdir / "state.json").read_text())
    meta = {"filesrc": {}, "names": state.get("names", {})}
    # Carry forward this pipeline's own per-launch files inside the retention window.
    # Files from a manifest without the pipeline mark came from the retired feed and
    # are deliberately NOT carried.
    if state:
        carried = 0
        for fn, src in (state.get("filesrc") or {}).items():
            f = prevdir / "soundings" / fn
            try:
                ts = datetime.strptime(fn[:-4].split("_")[1], "%Y%m%d%H").replace(tzinfo=timezone.utc)
            except (IndexError, ValueError):
                continue
            if ts >= cutoff and f.exists():
                shutil.copyfile(f, outdir / "soundings" / fn)
                meta["filesrc"][fn] = src
                carried += 1
        print(f"  carried forward {carried} launch file(s)", flush=True)
    else:
        print("  no previous state from this pipeline — starting fresh", flush=True)

    run_iem(outdir, state, by_wmo, icao2wmo, meta)
    run_igra(outdir, state, by_gid, meta)

    # manifest: per station, every launch held + the newest one's source
    hours: dict = {}
    for fn, src in meta["filesrc"].items():
        sid, tag = fn[:-4].split("_")
        hours.setdefault(sid, []).append((f"{tag[:4]}-{tag[4:6]}-{tag[6:8]} {tag[8:10]}:00", src))
    entries = {}
    for sid, hs in hours.items():
        hs.sort()
        dt, src = hs[-1]
        s = by_wmo.get(sid) or meta["names"].get(sid)
        if not s:
            continue
        entries[sid] = {"id": sid, "n": s["n"], "la": round(s["la"], 3), "lo": round(s["lo"], 3),
                        "src": src, "dt": dt, "hours": [h for h, _ in hs]}
        shutil.copyfile(outdir / "soundings" / f"{sid}_{dt.replace('-', '').replace(' ', '')[:10]}.csv",
                        outdir / "soundings" / f"{sid}.csv")
    n_iem = sum(1 for e in entries.values() if e["src"] == "IEM")
    (outdir / "manifest.json").write_text(json.dumps(
        {"pipeline": PIPELINE, "generated": NOW.strftime("%Y-%m-%d %H:%M UTC"),
         "sources": {"IEM": n_iem, "IGRA": len(entries) - n_iem}, "entries": entries}))
    state["filesrc"] = meta["filesrc"]
    state["names"] = {k: v for k, v in meta["names"].items() if k in entries}
    (outdir / "state.json").write_text(json.dumps(state))
    print(f"  manifest: {len(entries)} stations ({n_iem} newest via IEM, "
          f"{len(entries) - n_iem} via IGRA), {len(meta['filesrc'])} launch files → {outdir}",
          flush=True)
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
