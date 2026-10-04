#!/usr/bin/env python3
"""Mirror recent radiosonde soundings for the Skew-T Explorer (runs in GitHub Actions).

Two public sources, both credited on the page:

  * IEM RAOB JSON (Iowa Environmental Mesonet, Iowa State University) - real-time
    North America. ONE request per launch hour returns every station that reported
    at that hour (/json/raob.py?ts=YYYYMMDDHHMI, no station filter). IEM's robots.txt
    asks for Crawl-delay: 120, so requests are spaced IEM_DELAY s apart, capped per
    run (MAX_IEM_REQ), and each hour slot is asked at most two or three times over
    its first ~12 h (state.json remembers which slots were asked and when).
  * NOAA NCEI IGRA v2 (Integrated Global Radiosonde Archive) - the rest of the
    world, which NCEI refreshes once a day (~21:40 UTC), so about a day behind. The
    mirror downloads NO IGRA data: it reads only the y2d directory listing from
    /pub/data/igra/ (the /data/ tree is disallowed in ncei.noaa.gov/robots.txt) at
    most every IGRA_CHECK_H hours, and lists stations whose file grew at a recent
    daily update. The browser fetches a station's y2d zip on demand when opened.

Outputs (the layout the client reads from the `skewt-data` branch):
  manifest.json             {pipeline, generated, entries: {id: {id,n,la,lo,src,dt,hours}}}
                            id = WMO number (the client's byWmo key); src = "IEM" | "IGRA"
                            (IGRA entries: dt null, no hours, "upd" = NCEI update time UTC)
  soundings/{id}_{YYYYMMDDHH}.csv   one per IEM launch, kept RETAIN_H hours (carried forward)
  soundings/{id}.csv                copy of the newest launch
  state.json                bookkeeping (IEM slots asked, IGRA file sizes, per-file source)

CSV columns are fixed (the client and flag_anomalies.py index them by position):
  time,longitude,latitude,pressure_hPa,geopotential height_m,temperature_C,
  dew point temperature_C,ice point temperature_C,relative humidity_%,
  humidity wrt ice_%,mixing ratio_g/kg,wind direction_degree,wind speed_m/s
IEM carries no RH / ice point / mixing ratio, so they are derived from T, Td
and p (Magnus/Bolton over water, Buck-style over ice); winds reported on a subset of
levels are interpolated in log-p onto every thermodynamic level (as the client does
for live profiles). Missing values are written as "nan".

    python scripts/skewt/mirror_soundings.py OUTDIR [PREVDIR]
Env (testing): SKEWT_MAX_IEM_REQ, SKEWT_IEM_DELAY.
"""
from __future__ import annotations

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
IEM_LOOKBACK_H = 30
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
# NO IGRA DATA IS DOWNLOADED HERE (since 2026-10-04). The y2d zips are whole-year
# files that NCEI rewrites every day, so mirroring them meant ~0.5 GB/day of mostly
# unchanged data. Instead the mirror reads only the directory LISTING (one request,
# at most every IGRA_CHECK_H hours) and remembers each file's size: a file that GREW
# at NCEI's latest daily update holds new launches. Those stations go into the
# manifest as src "IGRA" (no per-launch files); when a visitor opens one, the browser
# fetches that one station's y2d zip from NCEI itself (CORS-open).
LIST_RE = re.compile(r'href="([A-Z0-9]{11})-data-beg(\d{4})\.txt\.zip".*?'
                     r'<td align="right">([\d-]+ [\d:]+)</td>\s*<td align="right">(\d+)</td>', re.S)
IGRA_RECENT_H = 60              # grew at one of the last ~2 daily updates


def _listing_utc(stamp: str) -> str:
    """NCEI's index prints US Eastern local time; store UTC."""
    from zoneinfo import ZoneInfo
    t = datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))
    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def run_igra(state: dict, by_gid: dict, prev_manifest: dict, meta: dict) -> dict:
    """{wmo: update stamp (UTC)} for IGRA stations with new launches at a recent update.

    The recent set ("grew") lives in state.json and is re-emitted on every run. The
    listing is re-read when the throttle allows OR whenever there is nothing to compare
    against (no sizes) or nothing to show (empty set) - an empty set must never wait
    out the throttle (2026-10-04: a stale "checked" stamp from the old bulk pipeline
    skipped the listing and the map lost every IGRA station for a run).
    """
    ig = state.setdefault("igra", {})
    ig.pop("files", None)                              # bookkeeping of the old bulk download
    size, grew = ig.setdefault("size", {}), ig.setdefault("grew", {})
    last = ig.get("checked")
    fresh = last and (NOW - datetime.fromisoformat(last)).total_seconds() < IGRA_CHECK_H * 3600
    if fresh and size and grew:
        print(f"  IGRA: listing checked {last[:16]}Z — next check after {IGRA_CHECK_H} h; "
              f"{len(grew)} recent station(s) carried", flush=True)
    else:
        try:
            listing = get(IGRA_Y2D, timeout=120).decode("utf-8", "replace")
            meta["igra_bytes"] = len(listing)
            ig["checked"] = NOW.isoformat()
            rows = LIST_RE.findall(listing)
            newest = max((_listing_utc(r[2]) for r in rows), default="")
            # bootstrap guesses stand only until NCEI's next daily update gives real growth
            replaced = bool(ig.get("boot") and newest and newest != ig["boot"] and size)
            if replaced:
                grew.clear()
                ig.pop("boot", None)
            how = ""
            if not size or (not grew and not replaced):
                # Nothing to compare against yet: seed from the previous manifest's IGRA
                # stations (either manifest format), else from the station list (stations
                # that reported this year). Approximate for at most one daily update.
                w2g = {s["id"]: g for g, s in by_gid.items() if s.get("id")}
                seed = {w2g.get(w) for w, e in (prev_manifest.get("entries") or {}).items()
                        if e.get("src") == "IGRA"} - {None}
                how = "the previous manifest"
                if not seed:
                    seed = {g for g, s in by_gid.items() if s.get("y1", 0) >= NOW.year}
                    how = "the station list (reported this year)"
                listed = {r[0] for r in rows}
                for gid in seed & listed:
                    grew[gid] = newest
                ig["boot"] = newest
            n_grew = 0
            for gid, beg, stamp, sz in rows:
                sz, utc = int(sz), _listing_utc(stamp)
                old = size.get(gid)
                if old is not None and sz > old:
                    grew[gid] = utc
                    n_grew += 1
                size[gid] = sz
            print(f"  IGRA listing: {len(rows)} y2d files ({len(listing) // 1024} KB), newest update "
                  f"{newest}Z, {n_grew} grew since last listing"
                  + (f"; seeded {len(grew)} from {how}" if how else ""), flush=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"::warning::IGRA listing failed: {repr(e)[:80]}", flush=True)
    out = {}
    for gid, utc in list(grew.items()):
        age = (NOW - datetime.strptime(utc, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
               ).total_seconds() / 3600
        if age > IGRA_RECENT_H:
            del grew[gid]
            continue
        s = by_gid.get(gid)
        if s and s.get("id"):
            out[s["id"]] = utc
    if not out:
        print("::warning::no IGRA stations to list this run — the map will show North America only",
              flush=True)
    return out


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
            if src != "IEM":
                continue                # IGRA launches are no longer mirrored (browser fetches them)
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
    igra = run_igra(state, by_gid, prev_manifest, meta)

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
    n_iem = len(entries)
    # IGRA stations: listed without per-launch files; "upd" = the NCEI daily update
    # that brought their newest launches (the launches themselves are ~1-2 days old)
    for sid, upd in igra.items():
        if sid in entries or sid not in by_wmo:
            continue                    # IEM already has it in real time
        s = by_wmo[sid]
        entries[sid] = {"id": sid, "n": s["n"], "la": round(s["la"], 3), "lo": round(s["lo"], 3),
                        "src": "IGRA", "dt": None, "upd": upd}
    (outdir / "manifest.json").write_text(json.dumps(
        {"pipeline": PIPELINE, "generated": NOW.strftime("%Y-%m-%d %H:%M UTC"),
         "sources": {"IEM": n_iem, "IGRA": len(entries) - n_iem}, "entries": entries}))
    state["filesrc"] = meta["filesrc"]
    state["names"] = {k: v for k, v in meta["names"].items() if k in entries}
    (outdir / "state.json").write_text(json.dumps(state))
    print(f"  manifest: {len(entries)} stations ({n_iem} IEM real time with "
          f"{len(meta['filesrc'])} launch files, {len(entries) - n_iem} IGRA on demand) → {outdir}",
          flush=True)
    print(f"  transfer this run: IEM {meta.get('iem_req', 0)} request(s) "
          f"{meta.get('iem_bytes', 0) / 1e6:.2f} MB; IGRA listing "
          f"{meta.get('igra_bytes', 0) / 1e3:.0f} KB", flush=True)
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
