#!/usr/bin/env python3
"""Mirror recent radiosonde soundings for the Skew-T Explorer (runs in GitHub Actions).

Three public sources, all credited on the page:

  * IEM RAOB JSON (Iowa Environmental Mesonet, Iowa State University) - real-time
    North America. ONE request per launch hour returns every station that reported
    at that hour (/json/raob.py?ts=YYYYMMDDHHMI, no station filter). IEM's robots.txt
    asks for Crawl-delay: 120, so requests are spaced IEM_DELAY s apart, capped per
    run (MAX_IEM_REQ), and each hour slot is asked at most two or three times over
    its first ~12 h (state.json remembers which slots were asked and when).
  * WMO WIS 2.0 Global Cache on AWS Open Data (s3://wis2globalcache, anonymous) -
    real-time BUFR radiosondes for the rest of the world, ~2 h after launch. Core
    data under the WMO Unified Data Policy; the originating national service is
    kept per launch and credited. Each run lists the GTS gateways' IUS/IUK trees
    (DWD and JMA copies) and each national centre's .../temp/ topic, GETs only keys
    not seen before (state.json), decodes them with ecCodes and keeps, per station
    and launch hour, the richest message. North American launches stay IEM's: WIS2
    never overwrites an IEM file and IEM replaces a WIS2 one for the same hour.
  * NOAA NCEI IGRA v2 (Integrated Global Radiosonde Archive) - the backfill for
    stations neither real-time feed carries; NCEI refreshes it once a day (~21:40
    UTC), so about a day behind. The mirror downloads NO IGRA data: it reads only
    the y2d directory listing from /pub/data/igra/ (the /data/ tree is disallowed
    in ncei.noaa.gov/robots.txt) at most every IGRA_CHECK_H hours, and lists
    stations whose file grew at a recent daily update. The browser fetches a
    station's y2d zip on demand when opened.

Outputs (the layout the client reads from the `skewt-data` branch):
  manifest.json             {pipeline, generated, sources, centres, entries: {id: {id,n,la,lo,src,dt,hours}}}
                            id = WMO number (the client's byWmo key); src = "IEM" | "WIS2" | "IGRA"
                            of the newest launch. Stations with any WIS2 launch also carry
                            "by" (originating centre code of the newest launch, "IEM" for an
                            IEM one) and "hx" {hour: code} for held launches whose code differs;
                            top-level "centres" maps codes to names. (IGRA entries: dt null, no
                            hours, "upd" = NCEI update time UTC)
  soundings/{id}_{YYYYMMDDHH}.csv   one per IEM/WIS2 launch (nominal hour), kept RETAIN_H hours
  soundings/{id}.csv                copy of the newest launch
  state.json                bookkeeping (IEM slots asked, IGRA file sizes, per-file source,
                            WIS2 keys seen + per-file richness)

CSV columns are fixed (the client and flag_anomalies.py index them by position):
  time,longitude,latitude,pressure_hPa,geopotential height_m,temperature_C,
  dew point temperature_C,ice point temperature_C,relative humidity_%,
  humidity wrt ice_%,mixing ratio_g/kg,wind direction_degree,wind speed_m/s
IEM carries no RH / ice point / mixing ratio, so they are derived from T, Td
and p (Magnus/Bolton over water, Buck-style over ice); winds reported on a subset of
levels are interpolated in log-p onto every thermodynamic level (as the client does
for live profiles). Missing values are written as "nan".

    python scripts/skewt/mirror_soundings.py OUTDIR [PREVDIR]
Env (testing / budgets): SKEWT_MAX_IEM_REQ, SKEWT_IEM_DELAY, SKEWT_WIS2_MAX_GET,
SKEWT_WIS2_MAX_LIST, SKEWT_WIS2_MAX_S, SKEWT_WIS2=0 (skip WIS2).
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
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
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
        print("::warning::no IGRA stations to list this run", flush=True)
    return out


# ---------------------------------------------------------------- WIS2 (global, real time)
# The WMO WIS 2.0 Global Cache on AWS Open Data: anonymous S3, ~72 h retention, core data
# free and unrestricted under the WMO Unified Data Policy (credit the originating service).
# A 2026-10-06 survey found radiosondes in three places, each filling gaps for the others
# (132 station-launches only native, 129 only via JMA, 54 only via DWD), so all are polled:
#   data/{de-dwd,jp-jma}-gts-to-wis2/data/core/I/U/{S,K}/...   GTS bulletins re-published by
#        the two gateways (IUS = entire high-resolution sounding, IUK = TEMP parts; both
#        gateways carry byte-identical copies of most bulletins - one GET serves both)
#   data/<centre>/data/core/weather/surface-based-observations/temp/   national centres
# Skipped: rw-rma (republishes 2025 files), bn-bdmd (TAC text), wind-profiler / pibal /
# dropsonde / ozone trees (other topics, or IUJ/IUD/IUE headers in a temp/ topic).
WIS2_BASE = "https://wis2globalcache.s3.amazonaws.com/"
WIS2_S3NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
WIS2_GATEWAYS = ("de-dwd-gts-to-wis2", "jp-jma-gts-to-wis2")
WIS2_TEMP = "data/core/weather/surface-based-observations/temp/"
# national centres with a temp/ topic (2026-10-06 survey) -> display name for attribution
WIS2_NATIVE = {
    "bb-barbadosmetservices": "Barbados Meteorological Services",
    "be-rmib": "Royal Meteorological Institute of Belgium",
    "br-inmet": "INMET (Brazil)",
    "ck-metservice": "Cook Islands Meteorological Service",
    "cl-meteochile": "Dirección Meteorológica de Chile",
    "cy-dom": "Department of Meteorology (Cyprus)",
    "dz-meteoalgerie": "Météo Algérie",
    "fr-meteofrance": "Météo-France",
    "hk-hko": "Hong Kong Observatory",
    "id-bmkg": "BMKG (Indonesia)",
    "il-ims": "Israel Meteorological Service",
    "in-imd": "India Meteorological Department",
    "ir-irimo": "IRIMO (Iran)",
    "it-meteoam": "Aeronautica Militare (Italy)",
    "kr-kma": "Korea Meteorological Administration",
    "kz-kazhydromet": "Kazhydromet (Kazakhstan)",
    "mx-smn": "Servicio Meteorológico Nacional (Mexico)",
    "nz-metservice": "MetService (New Zealand)",
    "py-dinac-dmh": "DINAC-DMH (Paraguay)",
    "ru-roshydromet": "Roshydromet (Russia)",
    "sa-ncm": "National Center for Meteorology (Saudi Arabia)",
    "sg-mss": "Meteorological Service Singapore",
    "th-tmd": "Thai Meteorological Department",
    "tt-trin-met": "Trinidad and Tobago Meteorological Service",
    "uk-metoffice": "Met Office (UK)",
}
WIS2_SKIP = {"rw-rma", "bn-bdmd"}
# GTS bulletin compiling centres (CCCC) seen carrying radiosondes -> service. Unknown codes
# are shown as the code itself.
WIS2_CCCC = {
    "AMMC": "Bureau of Meteorology (Australia)", "BABJ": "China Meteorological Administration",
    "BIKF": "Icelandic Met Office", "BIRK": "Icelandic Met Office", "DAMM": "Météo Algérie",
    "DBBB": "Météo-Bénin", "DEMS": "India Meteorological Department",
    "DTTA": "INM (Tunisia)", "DTTZ": "INM (Tunisia)", "EBWM": "Royal Meteorological Institute of Belgium",
    "EDZW": "Deutscher Wetterdienst", "EEMH": "Estonian Environment Agency",
    "EFKL": "Finnish Meteorological Institute", "EGRR": "Met Office (UK)", "EHDB": "KNMI (Netherlands)",
    "EIDB": "Met Éireann", "EKMI": "DMI (Denmark)", "ENMI": "MET Norway", "ESWI": "SMHI (Sweden)",
    "EYHM": "Lithuanian Hydrometeorological Service", "FMMI": "Météo Madagascar",
    "HABP": "HungaroMet (Hungary)", "KWBC": "NOAA/NWS (USA)", "LCLK": "Department of Meteorology (Cyprus)",
    "LDZM": "Croatian Meteorological and Hydrological Service", "LEMM": "AEMET (Spain)",
    "LIBN": "Aeronautica Militare (Italy)", "LICT": "Aeronautica Militare (Italy)",
    "LIED": "Aeronautica Militare (Italy)", "LIIB": "Aeronautica Militare (Italy)",
    "LIMN": "Aeronautica Militare (Italy)", "LIMZ": "Aeronautica Militare (Italy)",
    "LIPI": "Aeronautica Militare (Italy)", "LIRE": "Aeronautica Militare (Italy)",
    "LJLM": "Slovenian Environment Agency", "LLBD": "Israel Meteorological Service",
    "LOWM": "GeoSphere Austria", "LPMG": "IPMA (Portugal)", "LSSW": "MeteoSwiss",
    "LTAA": "Turkish State Meteorological Service", "LYBM": "RHMSS (Serbia)",
    "LZIB": "Slovak Hydrometeorological Institute", "MXBA": "Servicio Meteorológico Nacional (Mexico)",
    "NFFN": "Fiji Meteorological Service", "NTAA": "Météo-France (French Polynesia)",
    "NZKL": "MetService (New Zealand)", "OEJD": "National Center for Meteorology (Saudi Arabia)",
    "OEJN": "National Center for Meteorology (Saudi Arabia)", "OKPR": "Czech Hydrometeorological Institute",
    "RJTD": "Japan Meteorological Agency", "RKSL": "Korea Meteorological Administration",
    "RPMM": "PAGASA (Philippines)", "RUHB": "Roshydromet (Russia)", "RUML": "Roshydromet (Russia)",
    "RUMS": "Roshydromet (Russia)", "RUNW": "Roshydromet (Russia)", "SCSC": "Dirección Meteorológica de Chile",
    "SOWR": "IMGW (Poland)", "UAST": "Kazhydromet (Kazakhstan)", "UMRR": "Belhydromet (Belarus)",
    "VGDC": "Bangladesh Meteorological Department", "VHHH": "Hong Kong Observatory",
    "VTBB": "Thai Meteorological Department", "WSSS": "Meteorological Service Singapore",
    "YRBK": "National Meteorological Administration (Romania)",
}
NA_CCCC = {"KWBC", "CWAO", "CWUL"}       # North America: IEM's ground, fetched last
WIS2_MAX_GET = int(os.environ.get("SKEWT_WIS2_MAX_GET", 200))
WIS2_MAX_LIST = int(os.environ.get("SKEWT_WIS2_MAX_LIST", 45))
WIS2_MAX_S = float(os.environ.get("SKEWT_WIS2_MAX_S", 480))
WIS2_MAX_AGE_H = 72               # the cache keeps ~72 h; older launches are not taken
WIS2_SEEN_H = 80                  # forget seen keys this long after their LastModified
WIS2_DISCOVER_PER_RUN = 3         # unknown centres probed for a temp/ topic per run (each daily)
WIS2_MIN_LEVELS = 10              # the client's parseCSV floor
HDR_RE = re.compile(r"(?:^|_)(IU[A-Z]{2}\d{2})([A-Z]{4})(\d{6})([A-Z]{3})?(?=[_.]|$)")
WIGOS_NAME_RE = re.compile(r"WIGOS_(\d+)-(\d+)-(\d+)-([0-9A-Za-z]+)_(\d{8})T(\d{6})")
SFC_BIT = 131072                  # 008042 bit 1: surface level


def _lm(s: str) -> datetime:
    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def s3_list(prefix: str, ctr: Counter, delimiter: str | None = None, max_keys: int = 1000,
            max_pages: int = 10) -> tuple:
    """ListObjectsV2 on the cache, paged sequentially. -> ([(key, size, lm)], [common prefixes])."""
    objs, prefs, token = [], [], None
    for _ in range(max_pages):
        if ctr["list"] >= WIS2_MAX_LIST:
            ctr["list_capped"] += 1
            break
        q = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        raw = get(WIS2_BASE + "?" + urllib.parse.urlencode(q), timeout=60, tries=3, backoff=5)
        ctr["list"] += 1
        ctr["list_bytes"] += len(raw)
        root = ET.fromstring(raw)
        for c in root.findall("s:Contents", WIS2_S3NS):
            objs.append((c.findtext("s:Key", "", WIS2_S3NS), int(c.findtext("s:Size", "0", WIS2_S3NS)),
                         _lm(c.findtext("s:LastModified", "", WIS2_S3NS))))
        prefs += [p.text for p in root.findall("s:CommonPrefixes/s:Prefix", WIS2_S3NS)]
        if root.findtext("s:IsTruncated", "", WIS2_S3NS) != "true":
            break
        token = root.findtext("s:NextContinuationToken", "", WIS2_S3NS)
    return objs, prefs


def _header(name: str):
    """(TTAAii, CCCC, YYGGgg, BBB) of a GTS-style file name, or None."""
    m = HDR_RE.search(name)
    return m.groups() if m else None


def _hdr_time(yyggg: str, ref: datetime) -> datetime | None:
    """Bulletin day/hour/minute resolved to the latest month that is not after ref (+2 h)."""
    dd, hh, mi = int(yyggg[:2]), int(yyggg[2:4]), int(yyggg[4:6])
    base = ref
    for _ in range(3):
        try:
            t = datetime(base.year, base.month, dd, hh, mi, tzinfo=timezone.utc)
            if t <= ref + timedelta(hours=2):
                return t
        except ValueError:
            pass
        base = base.replace(day=1) - timedelta(days=1)
    return None


def nominal_hour(launch: datetime, hints: list) -> datetime:
    """The launch's nominal (synoptic) hour, which names the file and the map's launch hour.

    Balloons go up ~45-75 min before the nominal hour. A hint (the bulletin header's
    YYGGgg, or BUFR section 1's typical time) is used when it sits on the 3-hourly grid
    and within -2 h .. +2 h of the launch; otherwise the launch is put on the 3-hourly
    grid (launch + 90 min, floored): 22:58 -> 00Z, 11:30 -> 12Z, 17:43 -> 18Z, 13:16 -> 12Z."""
    for h in hints:
        if h and h.minute == 0 and h.hour % 3 == 0 and \
                -2 * 3600 <= (launch - h).total_seconds() <= 2 * 3600:
            return h
    t = launch + timedelta(minutes=90)
    return t.replace(hour=t.hour // 3 * 3, minute=0, second=0, microsecond=0)


class Wis2Decoder:
    """ecCodes BUFR decoding of radiosonde messages -> per-subset launch dicts."""

    def __init__(self):
        import eccodes                                    # noqa: PLC0415 - optional dependency
        self.ec = eccodes
        self.MISS_D = eccodes.CODES_MISSING_DOUBLE
        self.MISS_L = eccodes.CODES_MISSING_LONG

    def _ok(self, v):
        return v is not None and not isinstance(v, (str, bytes)) and abs(v) < 1e90 and v != self.MISS_L

    def _arr(self, h, k):
        try:
            return [float(x) if self._ok(x) else None for x in self.ec.codes_get_array(h, k)]
        except Exception:                                 # noqa: BLE001
            return []

    def _one(self, h, k):
        try:
            v = self.ec.codes_get(h, k)
        except Exception:                                 # noqa: BLE001
            return None
        if isinstance(v, (str, bytes)):
            v = v.decode() if isinstance(v, bytes) else v
            return v.strip() or None
        return v if self._ok(v) else None

    def messages(self, blob: bytes):
        """Every BUFR message in a file (GTS files wrap them in an abbreviated heading)."""
        i = blob.find(b"BUFR")
        while i >= 0 and i + 8 <= len(blob):
            n = int.from_bytes(blob[i + 4:i + 7], "big")
            if n < 30 or i + n > len(blob):
                break
            yield blob[i:i + n]
            i = blob.find(b"BUFR", i + n)

    def decode(self, blob: bytes, stats: Counter) -> list:
        ec, out = self.ec, []
        for msg in self.messages(blob):
            try:
                h = ec.codes_new_from_message(msg)
            except Exception:                             # noqa: BLE001
                stats["bad_msg"] += 1
                continue
            try:
                tmpl = [int(x) for x in ec.codes_get_array(h, "unexpandedDescriptors")]
                if 309056 in tmpl:                        # descent profile: not an ascent sounding
                    stats["descent"] += 1
                    continue
                typ = None
                try:
                    typ = datetime(*(int(ec.codes_get(h, k)) for k in (
                        "typicalYear", "typicalMonth", "typicalDay", "typicalHour", "typicalMinute")),
                        tzinfo=timezone.utc)
                except Exception:                         # noqa: BLE001
                    pass
                ec.codes_set(h, "unpack", 1)
                ns = ec.codes_get(h, "numberOfSubsets")
                if ns == 1:
                    s = self.subset(h, stats)
                    if s:
                        s["typ"] = typ
                        out.append(s)
                else:
                    # multi-subset (compressed or not): set the extract keys on the
                    # unpacked handle, CLONE, then unpack the clone - one subset each
                    for k in range(1, ns + 1):
                        try:
                            ec.codes_set(h, "extractSubset", k)
                            ec.codes_set(h, "doExtractSubsets", 1)
                            h2 = ec.codes_clone(h)
                        except Exception:                 # noqa: BLE001
                            stats["bad_subset"] += 1
                            continue
                        try:
                            ec.codes_set(h2, "unpack", 1)
                            s = self.subset(h2, stats)
                            if s:
                                s["typ"] = typ
                                out.append(s)
                        except Exception:                 # noqa: BLE001
                            stats["bad_subset"] += 1
                        finally:
                            ec.codes_release(h2)
            except Exception:                             # noqa: BLE001
                stats["bad_msg"] += 1
            finally:
                ec.codes_release(h)
        return out

    def subset(self, h, stats: Counter) -> dict | None:
        one, arr = self._one, self._arr
        T = arr(h, "airTemperature")
        n = len(T)
        if n < WIS2_MIN_LEVELS:
            stats["few_levels"] += 1
            return None
        # Level arrays line up with airTemperature over its first n entries; extra
        # occurrences (the wind-shear block repeats pressure/timePeriod/significance)
        # come after the level loop.
        P = arr(h, "pressure")
        if len(P) < n:
            stats["no_pressure"] += 1
            return None

        def first(*keys):
            for k in keys:
                a = arr(h, k)
                if len(a) >= n:
                    return a[:n]
            return [None] * n
        Z = first("nonCoordinateGeopotentialHeight", "geopotentialHeight")
        TD = first("dewpointTemperature")
        RH = first("relativeHumidity") if not any(TD) else [None] * n
        WD, WS = first("windDirection"), first("windSpeed")
        TP = first("timePeriod")
        SG = first("extendedVerticalSoundingSignificance")
        # station identity: block/station, else a WIGOS id whose local part is a WMO number
        wmo = None
        b, s = one(h, "blockNumber"), one(h, "stationNumber")
        if b is not None and s is not None and 0 < b < 100 and 0 <= s < 1000:
            wmo = f"{int(b):02d}{int(s):03d}"
        else:
            loc, iss = one(h, "wigosLocalIdentifierCharacter"), one(h, "wigosIssuerOfIdentifier")
            if loc and re.fullmatch(r"\d{5}", str(loc)):
                wmo = str(loc)
                if iss not in (20000, 20001):
                    wmo = "?" + wmo                       # national WSI: accepted only if known
        if not wmo:
            stats["no_station"] += 1
            return None
        try:
            sec = one(h, "second")
            launch = datetime(int(one(h, "year")), int(one(h, "month")), int(one(h, "day")),
                              int(one(h, "hour")), int(one(h, "minute") or 0),
                              int(sec) if sec is not None and 0 <= sec < 60 else 0, tzinfo=timezone.utc)
        except (TypeError, ValueError):
            stats["no_time"] += 1
            return None
        la, lo = one(h, "latitude"), one(h, "longitude")
        elev = one(h, "heightOfStationGroundAboveMeanSeaLevel")
        if elev is None:
            elev = one(h, "heightOfBarometerAboveMeanSeaLevel")
        levels = []
        for i in range(n):
            p = P[i]
            if p is None or not 100 <= p <= 110000:          # Pa
                continue
            t = T[i] - 273.15 if T[i] is not None else None
            if t is not None and not -100 <= t <= 60:
                t = None
            td = TD[i] - 273.15 if TD[i] is not None else None
            if td is None and t is not None and RH[i] is not None and 0 < RH[i] <= 100:
                e = _es(t) * RH[i] / 100                      # dew point from RH (Bolton inverse)
                td = 243.5 * math.log(e / 6.112) / (17.67 - math.log(e / 6.112))
            if td is not None and (t is None or not -110 <= td <= t + 0.5):
                td = None
            ws, wd = WS[i], WD[i]
            if ws is None or wd is None or not (0 <= ws <= 150 and 0 <= wd <= 360):
                ws = wd = None
            z = Z[i] if Z[i] is not None and -500 <= Z[i] <= 50000 else None
            et = TP[i] if TP[i] is not None and 0 <= TP[i] < 14400 else None
            sg = int(SG[i]) if SG[i] is not None else 0
            levels.append({"p": p / 100, "h": z, "t": t, "td": min(td, t) if td is not None else None,
                           "wd": wd, "ws": ws, "et": et, "sig": sg})
        return {"wmo": wmo, "launch": launch, "la": la, "lo": lo, "elev": elev, "levels": levels}


def qc_levels(levels: list, stats: Counter, upper_part: bool = False) -> list | None:
    """Strict checks on one launch's levels; None rejects it.

    Drops levels below the reported surface and isolated temperature spikes; rejects
    profiles whose pressure is not (mostly) monotone in reported order, that start
    above 500 hPa (an upper part only) or that have too few levels. upper_part=True
    relaxes the last two for TEMP parts C/D, which can only join a lower part."""
    psfc = max((L["p"] for L in levels if L["sig"] & SFC_BIT), default=None)
    if psfc:
        levels = [L for L in levels if L["p"] <= psfc + 0.5]
    th = [L for L in levels if L["t"] is not None]
    if len(th) < (3 if upper_part else WIS2_MIN_LEVELS):
        stats["qc_few"] += 1
        return None
    steps = [b["p"] < a["p"] for a, b in zip(th, th[1:]) if b["p"] != a["p"]]
    if steps and sum(steps) / len(steps) < 0.8:
        stats["qc_order"] += 1
        return None
    if max(L["p"] for L in th) < 500 and not upper_part:
        stats["qc_top_only"] += 1
        return None
    srt = sorted(th, key=lambda L: -L["p"])
    for a, b, c in zip(srt, srt[1:], srt[2:]):            # isolated spike vs both neighbours
        if abs(b["t"] - a["t"]) > 15 and abs(b["t"] - c["t"]) > 15 and a["p"] - c["p"] < 100:
            b["t"] = None
            b["td"] = None
    if sum(1 for L in levels if L["t"] is not None and L["p"] >= 20) < (3 if upper_part else WIS2_MIN_LEVELS):
        stats["qc_few"] += 1
        return None
    return levels


def standalone(levels: list) -> bool:
    th = [L for L in levels if L["t"] is not None and L["p"] >= 20]
    return len(th) >= WIS2_MIN_LEVELS and max(L["p"] for L in th) >= 500


def richness(levels: list) -> tuple:
    """(complete, n): complete = T, Td, wind and height all reported; n = thermo levels."""
    th = [L for L in levels if L["t"] is not None and L["p"] >= 20]
    has = lambda k: sum(1 for L in th if L.get(k) is not None) >= 5        # noqa: E731
    return (int(has("td") and has("ws") and has("h")), len(th))


def thin_levels(levels: list, cap: int) -> list:
    """At most `cap` thermo levels: every flagged (standard / significant / tropopause /
    max-wind / surface) level when they are few, the rest evenly spaced."""
    th = [i for i, L in enumerate(levels) if L["t"] is not None and L["p"] >= 20]
    if len(th) <= cap:
        return levels
    keep = {th[0], th[-1]}
    sig = [i for i in th if levels[i]["sig"]]
    if len(sig) <= cap // 2:
        keep.update(sig)
    rest = [i for i in th if i not in keep]
    room = cap - len(keep)
    if room > 0 and rest:
        step = len(rest) / room
        keep.update(rest[int(k * step)] for k in range(min(room, len(rest))))
    return [levels[i] for i in sorted(keep)]


def merge_levels(*sets) -> list:
    """Union of several level lists (TEMP parts) keyed on pressure; the fuller record wins."""
    best = {}
    for levels in sets:
        for L in levels:
            k = round(L["p"], 1)
            fill = sum(L.get(x) is not None for x in ("t", "td", "h", "ws"))
            if k not in best or fill > best[k][0]:
                best[k] = (fill, L)
    return [v[1] for _, v in sorted(best.items(), key=lambda kv: -kv[0])]


def levels_from_csv(text: str) -> list:
    """Re-read a mirror CSV (for merging a later TEMP part into an earlier one)."""
    out = []
    for ln in text.splitlines()[1:]:
        c = ln.split(",")
        if len(c) < 13:
            continue
        f = [None if v in ("nan", "") else float(v) for v in c[3:13]]
        out.append({"p": f[0], "h": f[1], "t": f[2], "td": f[3], "wd": f[8], "ws": f[9],
                    "et": None, "sig": 0})
    return out


def wis2_centres(state: dict, ctr: Counter) -> list:
    """Allow-list + centres found by the slow rediscovery (a few probed per run, each daily)."""
    d = state.setdefault("disc", {})
    found = d.setdefault("found", [])
    checked = d.setdefault("checked", {})
    try:
        listed = d.get("listed")
        if not listed or (NOW - datetime.fromisoformat(listed)).total_seconds() > 86400:
            _, prefs = s3_list("data/", ctr, delimiter="/", max_pages=2)
            if prefs:
                d["all"] = sorted(p.split("/")[1] for p in prefs)
                d["listed"] = NOW.isoformat()
        todo = [c for c in d.get("all", []) if c not in WIS2_NATIVE and c not in WIS2_SKIP
                and c not in WIS2_GATEWAYS and c not in found
                and (c not in checked or (NOW - datetime.fromisoformat(checked[c])).total_seconds() > 86400)]
        todo.sort(key=lambda c: checked.get(c, ""))
        for c in todo[:WIS2_DISCOVER_PER_RUN]:
            objs, _ = s3_list(f"data/{c}/{WIS2_TEMP}", ctr, max_keys=50, max_pages=1)
            checked[c] = NOW.isoformat()
            if any(not k.endswith((".txt", ".json")) and lm > NOW - timedelta(hours=WIS2_MAX_AGE_H)
                   for k, _, lm in objs):
                found.append(c)
                print(f"  WIS2: discovered a new temp/ topic at {c}", flush=True)
    except Exception as e:                                # noqa: BLE001
        print(f"  WIS2 centre rediscovery skipped: {repr(e)[:80]}", flush=True)
    return list(WIS2_NATIVE) + [c for c in found if c not in WIS2_NATIVE]


def run_wis2(outdir: Path, state: dict, by_wmo: dict, meta: dict) -> dict:
    """Mirror new WIS2 radiosonde launches into soundings/; returns {fn: sid} written."""
    t0 = time.time()
    ctr, stats = Counter(), Counter()
    w = state.setdefault("wis2", {})
    seen = w.setdefault("seen", {})                       # dedupe id -> LastModified (epoch h)
    score = w.setdefault("score", {})                     # fn -> [complete, n, kind]
    dec = Wis2Decoder()                                   # ImportError -> caller skips WIS2

    # 1. list: gateways' IUS trees first (most stations, entire soundings), then the
    #    national centres, then the gateways' IUK trees
    centres = wis2_centres(w, ctr)
    prefixes = [(f"data/{g}/data/core/I/U/S/", g, "S") for g in WIS2_GATEWAYS]
    prefixes += [(f"data/{c}/{WIS2_TEMP}", c, "N") for c in centres]
    prefixes += [(f"data/{g}/data/core/I/U/K/", g, "K") for g in WIS2_GATEWAYS]
    cand, oldest = [], NOW - timedelta(hours=WIS2_MAX_AGE_H)
    for pre, src, kind in prefixes:
        try:
            objs, _ = s3_list(pre, ctr)
        except Exception as e:                            # noqa: BLE001
            print(f"  WIS2 list {pre} failed: {repr(e)[:80]}", flush=True)
            continue
        for key, size, lm in objs:
            if lm < oldest or size < 200 or size > 3_000_000:
                continue
            name = key.rsplit("/", 1)[-1]
            if name.endswith((".txt", ".json", ".xml", ".html")):
                continue
            hdr = _header(name)
            k = kind
            if hdr:
                a1 = hdr[0][2]
                if a1 not in "KS":                        # IUJ pibal, IUD dropsonde, IUE ozone...
                    continue
                if kind == "N":
                    k = a1                                # a national copy of TEMP parts / entire
            native = src not in WIS2_GATEWAYS
            # one id for the DWD and JMA copies of a bulletin (same header + size)
            did = (f"G:{''.join(x or '' for x in hdr)}:{size}" if hdr and not native
                   else f"N:{src}/{name}")
            if did in seen:
                continue
            by = src if native else (hdr[1] if hdr else src)
            pri = (0 if k == "N" else 1 if k == "S" else 2) + (2 if by in NA_CCCC else 0)
            cand.append({"key": key, "size": size, "lm": lm, "did": did, "kind": k, "by": by,
                         "hdr": hdr, "pri": pri})
    # in-run dedupe of the two gateway copies, newest launches first within a priority
    uniq = {}
    for c in sorted(cand, key=lambda c: (c["pri"], -c["lm"].timestamp())):
        uniq.setdefault(c["did"], c)
    todo = list(uniq.values())
    print(f"  WIS2: {ctr['list']} list request(s) ({ctr['list_bytes'] / 1e6:.2f} MB), "
          f"{len(centres)} centres + 2 gateways; {len(todo)} new message(s) to fetch", flush=True)

    # 2. GET + decode, within the request and time budgets
    by_fn: dict = {}
    for c in todo:
        if ctr["get"] >= WIS2_MAX_GET or time.time() - t0 > WIS2_MAX_S:
            ctr["deferred"] += 1                          # stays unseen -> next run
            continue
        try:
            blob = get(WIS2_BASE + urllib.parse.quote(c["key"]), timeout=60, tries=2, backoff=5)
        except urllib.error.HTTPError as e:
            ctr["get_err"] += 1
            if e.code in (403, 404):                      # expired from the cache
                seen[c["did"]] = int(c["lm"].timestamp() // 3600)
            continue
        except Exception:                                 # noqa: BLE001
            ctr["get_err"] += 1
            continue
        ctr["get"] += 1
        ctr["get_bytes"] += len(blob)
        seen[c["did"]] = int(c["lm"].timestamp() // 3600)
        hints = [_hdr_time(c["hdr"][2], c["lm"])] if c["hdr"] else []
        for s in dec.decode(blob, stats):
            stats["subsets"] += 1
            sid = s["wmo"]
            if sid.startswith("?"):
                sid = sid[1:]
                if sid not in by_wmo:
                    stats["unknown_wsi"] += 1
                    continue
            launch = s["launch"]
            if launch > NOW + timedelta(hours=1) or launch < oldest:
                stats["too_old_or_future"] += 1
                continue
            st = by_wmo.get(sid)
            la, lo = s["la"], s["lo"]
            if st:
                if la is not None and lo is not None and (
                        abs(la - st["la"]) > 1.5 or
                        abs((lo - st["lo"] + 180) % 360 - 180) * math.cos(math.radians(st["la"])) > 1.5):
                    stats["position_mismatch"] += 1
                    continue
                la, lo, elev, name = st["la"], st["lo"], st.get("e") or s["elev"] or 0, st["n"]
            else:
                if la is None or lo is None or not (-90 <= la <= 90 and -180 <= lo <= 360):
                    stats["no_position"] += 1
                    continue
                lo = (lo + 180) % 360 - 180
                elev, name = s["elev"] or 0, f"WMO {sid}"
            levels = qc_levels(s["levels"], stats)
            upper = False
            if not levels and c["kind"] == "K":           # TEMP parts C/D: kept to merge only
                levels = qc_levels(s["levels"], Counter(), upper_part=True)
                upper = bool(levels)
            if not levels:
                continue
            nom = nominal_hour(launch, hints + [s.get("typ")])
            if nom < NOW - timedelta(hours=RETAIN_H):
                continue
            fn = f"{sid}_{_tag(nom)}.csv"
            by_fn.setdefault(fn, []).append({
                "sid": sid, "levels": levels, "launch": launch, "kind": c["kind"], "by": c["by"],
                "la": la, "lo": lo, "elev": elev, "name": name, "rich": richness(levels), "upper": upper})

    # 3. per station and launch hour keep the richest message; TEMP parts (IUK) merge -
    #    with each other and with parts already written by an earlier run
    got = {}
    for fn, cs in by_fn.items():
        dest = outdir / "soundings" / fn
        if meta["filesrc"].get(fn) == "IEM":
            stats["kept_iem"] += 1                        # North America: IEM's launch stands
            continue
        old = score.get(fn) if meta["filesrc"].get(fn) == "WIS2" and dest.exists() else None
        pool = [x for x in cs if not x["upper"]]
        parts = [x for x in cs if x["kind"] == "K"]
        base = []
        if parts and old and old[2] == "K":
            try:
                base = levels_from_csv(dest.read_text())
            except Exception:                             # noqa: BLE001
                base = []
        if parts and (len(parts) > 1 or base or any(x["upper"] for x in parts)):
            m = dict(next((x for x in parts if not x["upper"]), parts[0]))
            m["levels"] = merge_levels(base, *(x["levels"] for x in parts))
            m["upper"] = False
            if standalone(m["levels"]):
                m["rich"] = richness(m["levels"])
                pool.append(m)
        if not pool:
            stats["upper_part_only"] += 1
            continue
        best = max(pool, key=lambda x: x["rich"])
        if old and tuple(best["rich"]) <= tuple(old[:2]):
            stats["not_richer"] += 1                      # never replace a richer launch
            continue
        text = to_csv(thin_levels(best["levels"], MAX_LEVELS - 2), best["launch"],
                      best["la"], best["lo"], best["elev"])
        if not text:
            stats["csv_short"] += 1
            continue
        dest.write_text(text)
        meta["filesrc"][fn] = "WIS2"
        meta["fileby"][fn] = best["by"]
        meta["names"][best["sid"]] = {"n": best["name"], "la": best["la"], "lo": best["lo"]}
        score[fn] = [best["rich"][0], best["rich"][1], best["kind"]]
        got[fn] = best["sid"]
        ctr["by_" + best["kind"]] += 1

    # bookkeeping: forget keys past the cache's retention, scores of files no longer held
    cut = int((NOW - timedelta(hours=WIS2_SEEN_H)).timestamp() // 3600)
    for k in [k for k, v in seen.items() if v < cut]:
        del seen[k]
    for fn in [fn for fn in score if meta["filesrc"].get(fn) != "WIS2"]:
        del score[fn]
    print(f"  WIS2: {ctr['get']} GET(s) {ctr['get_bytes'] / 1e6:.2f} MB, {ctr['get_err']} failed, "
          f"{ctr['deferred']} deferred to the next run (budget {WIS2_MAX_GET} GETs / {WIS2_MAX_S:.0f} s); "
          f"{stats['subsets']} subsets -> {len(got)} launch file(s) written/upgraded "
          f"(entire/native {ctr['by_S'] + ctr['by_N']}, TEMP parts {ctr['by_K']}) "
          f"in {time.time() - t0:.0f} s", flush=True)
    print("  WIS2 skips: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()) if k != "subsets"),
          flush=True)
    meta.update(wis2_list=ctr["list"], wis2_list_bytes=ctr["list_bytes"], wis2_get=ctr["get"],
                wis2_bytes=ctr["get_bytes"], wis2_written=len(got))
    return got


def centre_label(code: str) -> str:
    return WIS2_NATIVE.get(code) or WIS2_CCCC.get(code) or code


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
    meta = {"filesrc": {}, "fileby": {}, "names": state.get("names", {})}
    # Carry forward this pipeline's own per-launch files inside the retention window.
    # Files from a manifest without the pipeline mark came from the retired feed and
    # are deliberately NOT carried.
    if state:
        carried = Counter()
        oldby = state.get("fileby") or {}
        for fn, src in (state.get("filesrc") or {}).items():
            if src not in ("IEM", "WIS2"):
                continue                # IGRA launches are not mirrored (the browser fetches them)
            f = prevdir / "soundings" / fn
            try:
                ts = datetime.strptime(fn[:-4].split("_")[1], "%Y%m%d%H").replace(tzinfo=timezone.utc)
            except (IndexError, ValueError):
                continue
            if ts >= cutoff and f.exists():
                shutil.copyfile(f, outdir / "soundings" / fn)
                meta["filesrc"][fn] = src
                if src == "WIS2":
                    meta["fileby"][fn] = oldby.get(fn, "")
                carried[src] += 1
        print(f"  carried forward {sum(carried.values())} launch file(s) {dict(carried)}", flush=True)
    else:
        print("  no previous state from this pipeline — starting fresh", flush=True)

    run_iem(outdir, state, by_wmo, icao2wmo, meta)
    # WIS2 must never take the IEM/IGRA publication down with it
    if os.environ.get("SKEWT_WIS2", "1") != "0":
        try:
            run_wis2(outdir, state, by_wmo, meta)
        except ImportError as e:
            print(f"::warning::WIS2 skipped — ecCodes unavailable ({e})", flush=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"::warning::WIS2 failed this run: {repr(e)[:160]}", flush=True)
    igra = run_igra(state, by_gid, prev_manifest, meta)

    # manifest: per station, every launch held + the newest one's source (and, where
    # WIS2 is involved, the originating centre per launch)
    hours: dict = {}
    for fn, src in meta["filesrc"].items():
        sid, tag = fn[:-4].split("_")
        code = (meta["fileby"].get(fn) or "WIS2") if src == "WIS2" else "IEM"
        hours.setdefault(sid, []).append((f"{tag[:4]}-{tag[4:6]}-{tag[6:8]} {tag[8:10]}:00", src, code))
    entries, used = {}, set()
    for sid, hs in hours.items():
        hs.sort()
        dt, src, code = hs[-1]
        s = by_wmo.get(sid) or meta["names"].get(sid)
        if not s:
            continue
        e = {"id": sid, "n": s["n"], "la": round(s["la"], 3), "lo": round(s["lo"], 3),
             "src": src, "dt": dt, "hours": [h[0] for h in hs]}
        if any(h[1] == "WIS2" for h in hs):
            e["by"] = code
            hx = {h[0]: h[2] for h in hs if h[2] != code}
            if hx:
                e["hx"] = hx
            used.update(h[2] for h in hs if h[1] == "WIS2")
        entries[sid] = e
        shutil.copyfile(outdir / "soundings" / f"{sid}_{dt.replace('-', '').replace(' ', '')[:10]}.csv",
                        outdir / "soundings" / f"{sid}.csv")
    n_rt = Counter(e["src"] for e in entries.values())
    # IGRA stations: listed without per-launch files; "upd" = the NCEI daily update
    # that brought their newest launches (the launches themselves are ~1-2 days old).
    # A station either real-time feed holds is never listed as IGRA.
    for sid, upd in igra.items():
        if sid in entries or sid not in by_wmo:
            continue
        s = by_wmo[sid]
        entries[sid] = {"id": sid, "n": s["n"], "la": round(s["la"], 3), "lo": round(s["lo"], 3),
                        "src": "IGRA", "dt": None, "upd": upd}
    n_igra = len(entries) - sum(n_rt.values())
    (outdir / "manifest.json").write_text(json.dumps(
        {"pipeline": PIPELINE, "generated": NOW.strftime("%Y-%m-%d %H:%M UTC"),
         "sources": {"IEM": n_rt["IEM"], "WIS2": n_rt["WIS2"], "IGRA": n_igra},
         "centres": {c: centre_label(c) for c in sorted(used)},
         "entries": entries}, ensure_ascii=False))
    state["filesrc"] = meta["filesrc"]
    state["fileby"] = {fn: b for fn, b in meta["fileby"].items() if meta["filesrc"].get(fn) == "WIS2"}
    state["names"] = {k: v for k, v in meta["names"].items() if k in entries}
    (outdir / "state.json").write_text(json.dumps(state, ensure_ascii=False))
    nf = Counter(meta["filesrc"].values())
    print(f"  manifest: {len(entries)} stations ({n_rt['IEM']} IEM + {n_rt['WIS2']} WIS2 real time "
          f"with {nf['IEM']} + {nf['WIS2']} launch files, {n_igra} IGRA on demand) → {outdir}",
          flush=True)
    print(f"  transfer this run: IEM {meta.get('iem_req', 0)} request(s) "
          f"{meta.get('iem_bytes', 0) / 1e6:.2f} MB; WIS2 {meta.get('wis2_list', 0)} list(s) "
          f"{meta.get('wis2_list_bytes', 0) / 1e6:.2f} MB + {meta.get('wis2_get', 0)} GET(s) "
          f"{meta.get('wis2_bytes', 0) / 1e6:.2f} MB; IGRA listing "
          f"{meta.get('igra_bytes', 0) / 1e3:.0f} KB", flush=True)
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
