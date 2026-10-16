#!/usr/bin/env python3
"""Mirror recent radiosonde soundings for the Skew-T Explorer (runs in GitHub Actions).

Five public sources, all credited on the page:

  * IEM RAOB JSON (Iowa Environmental Mesonet, Iowa State University) - real-time
    North America. ONE request per launch hour returns every station that reported
    at that hour (/json/raob.py?ts=YYYYMMDDHHMI, no station filter). IEM's robots.txt
    asks for Crawl-delay: 120, so requests are spaced IEM_DELAY s apart, capped per
    run (MAX_IEM_REQ), and each hour slot is asked at most two or three times over
    its first ~12 h (state.json remembers which slots were asked and when).
  * ECCC MSC Datamart (Environment and Climate Change Canada, dd.weather.gc.ca) - real-time
    Canada: each Canadian upper-air station's own native ~1-s BUFR (IUK to 100 hPa ~1 h after
    launch, IUS entire ~2 h after), from today's and yesterday's reception-date directories;
    only directories that changed are re-listed, only files that outrank what is held are GET.
    ECCC Data Servers End-use Licence ("Data Source: Environment and Climate Change Canada").
    Wins a tie against a WIS2 relay of the same bulletin and competes with IEM on richness.
  * WMO WIS 2.0 Global Cache on AWS Open Data (s3://wis2globalcache, anonymous) -
    real-time BUFR radiosondes for the rest of the world, ~2 h after launch. Core
    data under the WMO Unified Data Policy; the originating national service is
    kept per launch and credited. Each run lists the GTS gateways' IUS/IUK trees
    (DWD and JMA copies) and each national centre's .../temp/ topic, GETs only keys
    not seen before (state.json), decodes them with ecCodes and keeps, per station
    and launch hour, the richest message. North American launches stay IEM's: WIS2
    never overwrites an IEM file and IEM replaces a WIS2 one for the same hour.
  * GTS TEMP text (FM-35 TAC) via DWD's WIS 2.0 gateway (U/{S,K,L,E} trees) - real time for the
    services that send no BUFR (most of Africa); parts A-D decoded and merged, used only for
    station-hours no BUFR source holds (src "TAC", by "TAC:<CCCC>").
  * NOAA NCEI IGRA v2 (Integrated Global Radiosonde Archive) - the backfill for
    stations neither real-time feed carries; NCEI refreshes it once a day (~21:40
    UTC), so about a day behind. The mirror downloads NO IGRA data: it reads only
    the y2d directory listing from /pub/data/igra/ (the /data/ tree is disallowed
    in ncei.noaa.gov/robots.txt) at most every IGRA_CHECK_H hours, and lists
    stations whose file grew at a recent daily update. The browser fetches a
    station's y2d zip on demand when opened.

Outputs (the layout the client reads from the `skewt-data` branch):
  manifest.json             {pipeline, generated, sources, centres, entries: {id: {id,n,la,lo,src,dt,hours}}}
                            id = WMO number (the client's byWmo key); src = "IEM" | "ECCC" | "WIS2"
                            | "IGRA" of the newest launch. Stations with any WIS2/ECCC launch also
                            carry "by" (originating centre code of the newest launch, "IEM" for an
                            IEM one, "ECCC" for a Datamart one) and "hx" {hour: code} for held
                            launches whose code differs;
                            top-level "centres" maps codes to names. (IGRA entries: dt null, no
                            hours, "upd" = NCEI update time UTC)
  soundings/{id}_{YYYYMMDDHH}.csv   one per IEM/ECCC/WIS2 launch (nominal hour), kept RETAIN_H hours
  soundings/{id}.csv                copy of the newest launch
  state.json                bookkeeping (IEM slots asked, IGRA file sizes, per-file source,
                            WIS2 keys seen, ECCC directories listed + best file per launch,
                            per-file richness "score")

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
SKEWT_WIS2_MAX_LIST, SKEWT_WIS2_MAX_S, SKEWT_WIS2=0 (skip WIS2), SKEWT_ECCC_MAX_LIST,
SKEWT_ECCC_MAX_GET, SKEWT_ECCC_MAX_S, SKEWT_ECCC=0 (skip ECCC), SKEWT_TAC_MAX_GET, SKEWT_TAC_MAX_LIST,
SKEWT_TAC_MAX_S, SKEWT_TAC=0 (skip TAC).
"""

from __future__ import annotations

import contextlib
import itertools
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
from datetime import datetime, timedelta, UTC
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKEWT = HERE.parents[1] / "skewt"
UA = "scorvec.com skew-t explorer (+https://scorvec.com/skewt/)"
IEM_RAOB = "https://mesonet.agron.iastate.edu/json/raob.py"
IEM_NET = "https://mesonet.agron.iastate.edu/geojson/network.py?network=RAOB"
IGRA_Y2D = "https://www.ncei.noaa.gov/pub/data/igra/data/data-y2d/"
PIPELINE = 2  # manifests without this were made by the retired feed
MAX_LEVELS = 260
RETAIN_H = 96
IEM_DELAY = float(os.environ.get("SKEWT_IEM_DELAY", "121"))  # robots.txt Crawl-delay: 120
MAX_IEM_REQ = int(os.environ.get("SKEWT_MAX_IEM_REQ", "6"))
IEM_LOOKBACK_H = 30
IGRA_CHECK_H = 3
HEADER = (
    "time,longitude,latitude,pressure_hPa,geopotential height_m,temperature_C,"
    "dew point temperature_C,ice point temperature_C,relative humidity_%,"
    "humidity wrt ice_%,mixing ratio_g/kg,wind direction_degree,wind speed_m/s"
)
NOW = datetime.now(UTC)


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
        except Exception as e:  # noqa: BLE001
            if k == tries:
                raise
            print(f"    retry {k}/{tries - 1} after {repr(e)[:60]}", flush=True)
            time.sleep(backoff * k)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- profile -> CSV
def _num(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _es(tc):  # saturation vapour pressure over water, hPa (Bolton)
    return 6.112 * math.exp(17.67 * tc / (tc + 243.5))


def _esi(tc):  # over ice, hPa (Magnus, Alduchov & Eskridge)
    return 6.1121 * math.exp(22.587 * tc / (tc + 273.86))


def _f(v, fmt):
    return format(v, fmt) if _num(v) else "nan"


def to_csv(levels: list, launch: datetime, la: float, lo: float, elev: float) -> str | None:
    """levels: [{p (hPa), h (m|None), t (C|None), td (C|None), wd, ws (m/s)|None, et (s)|None}]."""
    winds = sorted(
        (
            (L["p"], -L["ws"] * math.sin(math.radians(L["wd"])), -L["ws"] * math.cos(math.radians(L["wd"])))
            for L in levels
            if _num(L.get("wd")) and _num(L.get("ws")) and L["ws"] >= 0
        ),
        key=lambda w: -w[0],
    )
    thermo = sorted((L for L in levels if _num(L.get("t")) and L["p"] >= 20), key=lambda L: -L["p"])

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
        if round(p, 1) >= round(lastP, 1):  # written to 0.1 hPa: no duplicate pressures
            continue
        h = L.get("h")
        if not _num(h):  # hypsometric fill between reported heights
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
        rows.append(
            f"{ts:%Y-%m-%d %H:%M:%S},{lo:.4f},{la:.4f},{p:.1f},{h:.0f},{t:.1f},"
            f"{_f(td, '.1f')},{_f(ice, '.1f')},{_f(rh, '.0f')},{_f(rhi, '.0f')},"
            f"{_f(mr, '.2f')},{_f(wd, '.0f')},{_f(ws, '.1f')}"
        )
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
            pri = (0 if slot.hour in (0, 12) else 1 if slot.hour in (6, 18) else 2, 0 if not prev else 1, age)
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
        st = {
            f["id"]: {
                "n": f["properties"].get("sname", f["id"]),
                "la": f["geometry"]["coordinates"][1],
                "lo": f["geometry"]["coordinates"][0],
                "e": f["properties"].get("elevation") or 0,
            }
            for f in feats
        }
        state["iemnet"] = {"t": NOW.isoformat(), "st": st}
        time.sleep(IEM_DELAY)
        return st
    except Exception as e:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
            print(f"  IEM {key} failed: {repr(e)[:80]}", flush=True)
            continue
        nreq += 1
        nbytes += len(raw)
        state["iem"].setdefault(key, []).append(round((NOW - slot).total_seconds() / 3600, 1))
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
        print(f"  IEM {slot:%Y-%m-%d %H}Z: {len(profs)} profiles, {n} written ({len(raw) // 1024} KB)", flush=True)
    if unmapped:  # stations iem_raob.json does not know: IEM's own table
        tbl = iem_station_table(state)
        for pr in unmapped:
            icao = str(pr.get("station", ""))
            s = tbl.get(icao)
            if s:
                write_iem(outdir, pr, icao, s["la"], s["lo"], s["e"], s["n"], meta, got)
    print(f"  IEM: {nreq} request(s), {nbytes / 1e6:.2f} MB, {len(got)} launch file(s) written/updated", flush=True)
    meta["iem_req"], meta["iem_bytes"] = nreq, nbytes
    return got


def write_iem(outdir, pr, sid, la, lo, el, name, meta, got) -> int:
    try:
        launch = datetime.strptime(pr["valid"][:16], "%Y-%m-%dT%H:%M").replace(tzinfo=UTC)
    except (KeyError, ValueError):
        return 0
    if launch < NOW - timedelta(hours=RETAIN_H):
        return 0
    levels = []
    for L in pr.get("profile") or []:
        if not _num(L.get("pres")):
            continue
        ws = L["sknt"] * 0.514444 if _num(L.get("sknt")) else None
        levels.append(
            {"p": L["pres"], "h": L.get("hght"), "t": L.get("tmpc"), "td": L.get("dwpc"), "wd": L.get("drct"), "ws": ws}
        )
    text = to_csv(levels, launch, la, lo, el)
    if not text:
        return 0
    fn = f"{sid}_{_tag(launch)}.csv"
    dest = outdir / "soundings" / fn
    # a later ask can carry more levels (IEM fills in); never shrink a profile
    if dest.exists() and meta["filesrc"].get(fn) == "IEM" and dest.read_text().count("\n") >= text.count("\n"):
        return 0
    # Canada's own high-resolution ECCC file stands unless IEM's is richer (see rank())
    if meta["filesrc"].get(fn) == "ECCC":
        held = held_rank(fn, dest, meta, meta["score"])
        if held and rank(richness(levels), 0, "IEM") <= held:
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
LIST_RE = re.compile(
    r'href="([A-Z0-9]{11})-data-beg(\d{4})\.txt\.zip".*?'
    r'<td align="right">([\d-]+ [\d:]+)</td>\s*<td align="right">(\d+)</td>',
    re.S,
)
IGRA_RECENT_H = 60  # grew at one of the last ~2 daily updates


def _listing_utc(stamp: str) -> str:
    """NCEI's index prints US Eastern local time; store UTC."""
    from zoneinfo import ZoneInfo

    t = datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))
    return t.astimezone(UTC).strftime("%Y-%m-%d %H:%M")


def run_igra(state: dict, by_gid: dict, prev_manifest: dict, meta: dict) -> dict:
    """{wmo: update stamp (UTC)} for IGRA stations with new launches at a recent update.

    The recent set ("grew") lives in state.json and is re-emitted on every run. The
    listing is re-read when the throttle allows OR whenever there is nothing to compare
    against (no sizes) or nothing to show (empty set) - an empty set must never wait
    out the throttle (2026-10-04: a stale "checked" stamp from the old bulk pipeline
    skipped the listing and the map lost every IGRA station for a run).
    """
    ig = state.setdefault("igra", {})
    ig.pop("files", None)  # bookkeeping of the old bulk download
    size, grew = ig.setdefault("size", {}), ig.setdefault("grew", {})
    last = ig.get("checked")
    fresh = last and (NOW - datetime.fromisoformat(last)).total_seconds() < IGRA_CHECK_H * 3600
    if fresh and size and grew:
        print(
            f"  IGRA: listing checked {last[:16]}Z — next check after {IGRA_CHECK_H} h; "
            f"{len(grew)} recent station(s) carried",
            flush=True,
        )
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
                seed = {
                    w2g.get(w) for w, e in (prev_manifest.get("entries") or {}).items() if e.get("src") == "IGRA"
                } - {None}
                how = "the previous manifest"
                if not seed:
                    seed = {g for g, s in by_gid.items() if s.get("y1", 0) >= NOW.year}
                    how = "the station list (reported this year)"
                listed = {r[0] for r in rows}
                for gid in seed & listed:
                    grew[gid] = newest
                ig["boot"] = newest
            n_grew = 0
            for gid, _beg, stamp, sz_txt in rows:
                sz, utc = int(sz_txt), _listing_utc(stamp)
                old = size.get(gid)
                if old is not None and sz > old:
                    grew[gid] = utc
                    n_grew += 1
                size[gid] = sz
            print(
                f"  IGRA listing: {len(rows)} y2d files ({len(listing) // 1024} KB), newest update "
                f"{newest}Z, {n_grew} grew since last listing" + (f"; seeded {len(grew)} from {how}" if how else ""),
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"::warning::IGRA listing failed: {repr(e)[:80]}", flush=True)
    out = {}
    for gid, utc in list(grew.items()):
        age = (NOW - datetime.strptime(utc, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)).total_seconds() / 3600
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
    "AMMC": "Bureau of Meteorology (Australia)",
    "BABJ": "China Meteorological Administration",
    "BIKF": "Icelandic Met Office",
    "BIRK": "Icelandic Met Office",
    "DAMM": "Météo Algérie",
    "DBBB": "Météo-Bénin",
    "DEMS": "India Meteorological Department",
    "DTTA": "INM (Tunisia)",
    "DTTZ": "INM (Tunisia)",
    "EBWM": "Royal Meteorological Institute of Belgium",
    "EDZW": "Deutscher Wetterdienst",
    "LFPW": "Météo-France",
    "CWAO": "Environment and Climate Change Canada",
    "EEMH": "Estonian Environment Agency",
    "EFKL": "Finnish Meteorological Institute",
    "EGRR": "Met Office (UK)",
    "EHDB": "KNMI (Netherlands)",
    "EIDB": "Met Éireann",
    "EKMI": "DMI (Denmark)",
    "ENMI": "MET Norway",
    "ESWI": "SMHI (Sweden)",
    "EYHM": "Lithuanian Hydrometeorological Service",
    "FMMI": "Météo Madagascar",
    "HABP": "HungaroMet (Hungary)",
    "KWBC": "NOAA/NWS (USA)",
    "LCLK": "Department of Meteorology (Cyprus)",
    "LDZM": "Croatian Meteorological and Hydrological Service",
    "LEMM": "AEMET (Spain)",
    "LIBN": "Aeronautica Militare (Italy)",
    "LICT": "Aeronautica Militare (Italy)",
    "LIED": "Aeronautica Militare (Italy)",
    "LIIB": "Aeronautica Militare (Italy)",
    "LIMN": "Aeronautica Militare (Italy)",
    "LIMZ": "Aeronautica Militare (Italy)",
    "LIPI": "Aeronautica Militare (Italy)",
    "LIRE": "Aeronautica Militare (Italy)",
    "LJLM": "Slovenian Environment Agency",
    "LLBD": "Israel Meteorological Service",
    "LOWM": "GeoSphere Austria",
    "LPMG": "IPMA (Portugal)",
    "LSSW": "MeteoSwiss",
    "LTAA": "Turkish State Meteorological Service",
    "LYBM": "RHMSS (Serbia)",
    "LZIB": "Slovak Hydrometeorological Institute",
    "MXBA": "Servicio Meteorológico Nacional (Mexico)",
    "NFFN": "Fiji Meteorological Service",
    "NTAA": "Météo-France (French Polynesia)",
    "NZKL": "MetService (New Zealand)",
    "OEJD": "National Center for Meteorology (Saudi Arabia)",
    "OEJN": "National Center for Meteorology (Saudi Arabia)",
    "OKPR": "Czech Hydrometeorological Institute",
    "RJTD": "Japan Meteorological Agency",
    "RKSL": "Korea Meteorological Administration",
    "RPMM": "PAGASA (Philippines)",
    "RUHB": "Roshydromet (Russia)",
    "RUML": "Roshydromet (Russia)",
    "RUMS": "Roshydromet (Russia)",
    "RUNW": "Roshydromet (Russia)",
    "SCSC": "Dirección Meteorológica de Chile",
    "SOWR": "IMGW (Poland)",
    "UAST": "Kazhydromet (Kazakhstan)",
    "UMRR": "Belhydromet (Belarus)",
    "VGDC": "Bangladesh Meteorological Department",
    "VHHH": "Hong Kong Observatory",
    "VTBB": "Thai Meteorological Department",
    "WSSS": "Meteorological Service Singapore",
    "YRBK": "National Meteorological Administration (Romania)",
}
# compiling centres seen only in the TAC TEMP trees (2026-10-07 survey of the DWD gateway's U/ trees)
WIS2_CCCC.update(
    {
        "DFFD": "ANAM (Burkina Faso)",
        "DIAP": "SODEXAM (Côte d'Ivoire)",
        "DRRN": "Direction de la Météorologie Nationale (Niger)",
        "FAPR": "South African Weather Service",
        "FCBB": "ANAC Météorologie (Republic of the Congo)",
        "FEFF": "Direction de la Météorologie (Central African Republic)",
        "FKKD": "Direction de la Météorologie Nationale (Cameroon)",
        "FSIA": "Seychelles Meteorological Authority",
        "FZAA": "METTELSAT (DR Congo)",
        "GABS": "Mali-Météo",
        "GCXO": "AEMET (Spain, Canary Islands)",
        "GOOY": "ANACIM (Senegal)",
        "GQNO": "ONM (Mauritania)",
        "HECA": "Egyptian Meteorological Authority",
        "LGAT": "Hellenic National Meteorological Service",
        "NCRG": "Cook Islands Meteorological Service",
        "OIII": "IRIMO (Iran)",
        "OJAM": "Jordan Meteorological Department",
        "OOMS": "Oman Meteorology",
        "OPKC": "Pakistan Meteorological Department",
        "SABM": "Servicio Meteorológico Nacional (Argentina)",
        "SBBR": "DECEA / INMET (Brazil)",
        "SGAS": "DINAC-DMH (Paraguay)",
        "SKBO": "IDEAM (Colombia)",
        "UBBB": "National Hydrometeorological Service (Azerbaijan)",
        "UGEE": "Hydromet Service (Armenia)",
        "UKMS": "Ukrainian Hydrometeorological Center",
        "UMMN": "Belhydromet (Belarus)",
        "VBRR": "DMH (Myanmar)",
        "VNNN": "Viet Nam Meteorological and Hydrological Administration",
        "MNUB": "NAMEM (Mongolia)",
        "DKPY": "Korea Meteorological Administration",
        "WBSB": "Brunei Darussalam Meteorological Department",
        "WIIX": "BMKG (Indonesia)",
        "WMKK": "MetMalaysia",
        "DTTA": "INM (Tunisia)",
    }
)
# Canadian upper-air stations bulletin under their own indicator (MSC Datamart IU/ tree, 2026-10-06);
# a WIS2 relay of one (e.g. CAYT, St John's) is credited to ECCC
for _c in [
    "CAYT",
    "CGCB",
    "COYU",
    "CWEU",
    "CWLT",
    "CWMW",
    "CWQI",
    "CWSE",
    "CWVK",
    "CYBK",
    "CYCB",
    "CYEV",
    "CYFB",
    "CYJT",
    "CYPH",
    "CYPL",
    "CYQD",
    "CYRB",
    "CYSM",
    "CYUX",
    "CYVP",
    "CYVQ",
    "CYXY",
    "CYYE",
    "CYYQ",
    "CYYR",
    "CYZT",
    "CYZV",
    "CZXS",
]:
    WIS2_CCCC.setdefault(_c, "Environment and Climate Change Canada")
NA_CCCC = {"KWBC", "CWAO", "CWUL"}  # North America: IEM's ground, fetched last
# GTS hubs that compile other countries' soundings into their bulletins: a station outside the
# hub's home (IGRA/FIPS country prefixes) is credited to its own national service, "relayed by" the
# hub (2026-10-06: Bogota's IDEAM sounding arrives in a KWBC bulletin)
RELAY_HOME = {
    "KWBC": {"US", "RQ", "GQ", "AQ", "VQ", "CQ", "RM", "FM", "PS", "AY", "WQ", "JQ"},
    "AMMC": {"AS", "AY", "CK", "KT", "NF", "AT", "HM"},
    "NFFN": {"FJ"},
    "EGRR": {"UK", "FK", "SH", "GI", "AY", "BD"},
    "LFPW": {"FR", "FP", "NC", "RE", "GP", "MB", "FG", "SB", "WF", "AY", "FS"},
}
FIPS_NAME = {
    "AC": "Antigua and Barbuda",
    "AR": "Argentina",
    "BB": "Barbados",
    "BF": "the Bahamas",
    "BH": "Belize",
    "BL": "Bolivia",
    "BR": "Brazil",
    "CI": "Chile",
    "CO": "Colombia",
    "CS": "Costa Rica",
    "CU": "Cuba",
    "DR": "the Dominican Republic",
    "EC": "Ecuador",
    "ES": "El Salvador",
    "GT": "Guatemala",
    "GY": "Guyana",
    "HA": "Haiti",
    "HO": "Honduras",
    "JM": "Jamaica",
    "MX": "Mexico",
    "NU": "Nicaragua",
    "NS": "Suriname",
    "PA": "Paraguay",
    "PE": "Peru",
    "PM": "Panama",
    "TD": "Trinidad and Tobago",
    "UY": "Uruguay",
    "VE": "Venezuela",
    "KR": "Kiribati",
    "TV": "Tuvalu",
    "TN": "Tonga",
    "WS": "Samoa",
    "NH": "Vanuatu",
    "BP": "the Solomon Islands",
    "PP": "Papua New Guinea",
    "NZ": "New Zealand",
    "CW": "the Cook Islands",
    "ID": "Indonesia",
    "TT": "Timor-Leste",
    "CA": "Canada",
    "BD": "Bermuda",
    "FK": "the Falkland Islands",
}
WIS2_MAX_GET = int(os.environ.get("SKEWT_WIS2_MAX_GET", "200"))
WIS2_MAX_LIST = int(os.environ.get("SKEWT_WIS2_MAX_LIST", "45"))
WIS2_MAX_S = float(os.environ.get("SKEWT_WIS2_MAX_S", "480"))
WIS2_MAX_AGE_H = 72  # the cache keeps ~72 h; older launches are not taken
WIS2_SEEN_H = 80  # forget seen keys this long after their LastModified
WIS2_DISCOVER_PER_RUN = 3  # unknown centres probed for a temp/ topic per run (each daily)
WIS2_MIN_LEVELS = 10  # the client's parseCSV floor
HDR_RE = re.compile(r"(?:^|_)(IU[A-Z]{2}\d{2})([A-Z]{4})(\d{6})([A-Z]{3})?(?=[_.]|$)")
WIGOS_NAME_RE = re.compile(r"WIGOS_(\d+)-(\d+)-(\d+)-([0-9A-Za-z]+)_(\d{8})T(\d{6})")
SFC_BIT = 131072  # 008042 bit 1: surface level


def _lm(s: str) -> datetime:
    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)


def s3_list(
    prefix: str, ctr: Counter, delimiter: str | None = None, max_keys: int = 1000, max_pages: int = 10
) -> tuple:
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
        objs.extend(
            (
                c.findtext("s:Key", "", WIS2_S3NS),
                int(c.findtext("s:Size", "0", WIS2_S3NS)),
                _lm(c.findtext("s:LastModified", "", WIS2_S3NS)),
            )
            for c in root.findall("s:Contents", WIS2_S3NS)
        )
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
            t = datetime(base.year, base.month, dd, hh, mi, tzinfo=UTC)
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
        if h and h.minute == 0 and h.hour % 3 == 0 and -2 * 3600 <= (launch - h).total_seconds() <= 2 * 3600:
            return h
    t = launch + timedelta(minutes=90)
    return t.replace(hour=t.hour // 3 * 3, minute=0, second=0, microsecond=0)


class BufrDecoder:
    """ecCodes BUFR decoding of radiosonde messages -> per-subset launch dicts (WIS2 and ECCC)."""

    def __init__(self):
        import eccodes  # optional dependency

        self.ec = eccodes
        self.MISS_D = eccodes.CODES_MISSING_DOUBLE
        self.MISS_L = eccodes.CODES_MISSING_LONG

    def _ok(self, v):
        return v is not None and not isinstance(v, (str, bytes)) and abs(v) < 1e90 and v != self.MISS_L

    def _arr(self, h, k):
        try:
            return [float(x) if self._ok(x) else None for x in self.ec.codes_get_array(h, k)]
        except Exception:  # noqa: BLE001
            return []

    def _one(self, h, k):
        try:
            v = self.ec.codes_get(h, k)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(v, (str, bytes)):
            v = v.decode() if isinstance(v, bytes) else v
            return v.strip() or None
        return v if self._ok(v) else None

    def messages(self, blob: bytes):
        """Every BUFR message in a file (GTS files wrap them in an abbreviated heading)."""
        i = blob.find(b"BUFR")
        while i >= 0 and i + 8 <= len(blob):
            n = int.from_bytes(blob[i + 4 : i + 7], "big")
            if n < 30 or i + n > len(blob):
                break
            yield blob[i : i + n]
            i = blob.find(b"BUFR", i + n)

    def decode(self, blob: bytes, stats: Counter) -> list:
        ec, out = self.ec, []
        for msg in self.messages(blob):
            try:
                h = ec.codes_new_from_message(msg)
            except Exception:  # noqa: BLE001
                stats["bad_msg"] += 1
                continue
            try:
                tmpl = [int(x) for x in ec.codes_get_array(h, "unexpandedDescriptors")]
                if 309056 in tmpl:  # descent profile: not an ascent sounding
                    stats["descent"] += 1
                    continue
                typ = None
                with contextlib.suppress(Exception):
                    typ = datetime(
                        *(
                            int(ec.codes_get(h, k))
                            for k in ("typicalYear", "typicalMonth", "typicalDay", "typicalHour", "typicalMinute")
                        ),
                        tzinfo=UTC,
                    )
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
                        except Exception:  # noqa: BLE001
                            stats["bad_subset"] += 1
                            continue
                        try:
                            ec.codes_set(h2, "unpack", 1)
                            s = self.subset(h2, stats)
                            if s:
                                s["typ"] = typ
                                out.append(s)
                        except Exception:  # noqa: BLE001
                            stats["bad_subset"] += 1
                        finally:
                            ec.codes_release(h2)
            except Exception:  # noqa: BLE001
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
                    wmo = "?" + wmo  # national WSI: accepted only if known
        if not wmo:
            stats["no_station"] += 1
            return None
        try:
            sec = one(h, "second")
            launch = datetime(
                int(one(h, "year")),
                int(one(h, "month")),
                int(one(h, "day")),
                int(one(h, "hour")),
                int(one(h, "minute") or 0),
                int(sec) if sec is not None and 0 <= sec < 60 else 0,
                tzinfo=UTC,
            )
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
            if p is None or not 100 <= p <= 110000:  # Pa
                continue
            t = T[i] - 273.15 if T[i] is not None else None
            if t is not None and not -100 <= t <= 60:
                t = None
            td = TD[i] - 273.15 if TD[i] is not None else None
            if td is None and t is not None and RH[i] is not None and 0 < RH[i] <= 100:
                e = _es(t) * RH[i] / 100  # dew point from RH (Bolton inverse)
                td = 243.5 * math.log(e / 6.112) / (17.67 - math.log(e / 6.112))
            if td is not None and (t is None or not -110 <= td <= t + 0.5):
                td = None
            ws, wd = WS[i], WD[i]
            if ws is None or wd is None or not (0 <= ws <= 150 and 0 <= wd <= 360):
                ws = wd = None
            z = Z[i] if Z[i] is not None and -500 <= Z[i] <= 50000 else None
            et = TP[i] if TP[i] is not None and 0 <= TP[i] < 14400 else None
            sg = int(SG[i]) if SG[i] is not None else 0
            levels.append(
                {
                    "p": p / 100,
                    "h": z,
                    "t": t,
                    "td": min(td, t) if td is not None else None,
                    "wd": wd,
                    "ws": ws,
                    "et": et,
                    "sig": sg,
                }
            )
        return {"wmo": wmo, "launch": launch, "la": la, "lo": lo, "elev": elev, "levels": levels}


def qc_levels(levels: list, stats: Counter, upper_part: bool = False, who: str = "") -> list | None:
    """Strict checks on one launch's levels; None rejects it.

    Drops levels below the reported surface and isolated temperature spikes; rejects
    profiles whose pressure is not (mostly) monotone in reported order, that start
    above 500 hPa (an upper part only) or that have too few levels. upper_part=True
    relaxes the last two for TEMP parts C/D, which can only join a lower part."""
    psfc = max((L["p"] for L in levels if L["sig"] & SFC_BIT), default=None)
    if psfc:
        levels = [L for L in levels if L["p"] <= psfc + 0.5]
    # physically impossible levels: warmer than +15 C above 600 hPa (the global 500 hPa maximum is
    # near 0 C), or a reported height below one reported at a higher pressure (2026-10-06 Kolpasevo:
    # two trailing levels at 493 hPa / 575 m / +47 C)
    zmax, keep = -1e9, []
    for L in sorted(levels, key=lambda L: -L["p"]):
        if L["t"] is not None and L["p"] < 600 and L["t"] > 15:
            stats["qc_hot_aloft"] += 1
            continue
        if L.get("h") is not None:
            if L["h"] < zmax - 100:
                stats["qc_height_order"] += 1
                continue
            zmax = max(zmax, L["h"])
        keep.append(id(L))
    levels = [L for L in levels if id(L) in set(keep)]
    th = [L for L in levels if L["t"] is not None]
    if len(th) < (3 if upper_part else WIS2_MIN_LEVELS):
        stats["qc_few"] += 1
        return None
    steps = [b["p"] < a["p"] for a, b in itertools.pairwise(th) if b["p"] != a["p"]]
    if steps and sum(steps) / len(steps) < 0.8:
        stats["qc_order"] += 1
        return None
    if max(L["p"] for L in th) < 500 and not upper_part:
        stats["qc_top_only"] += 1
        return None
    srt = sorted(th, key=lambda L: -L["p"])
    spikes = [
        b
        for a, b, c in zip(srt, srt[1:], srt[2:])  # isolated spike vs both neighbours
        if abs(b["t"] - a["t"]) > 15 and abs(b["t"] - c["t"]) > 15 and a["p"] - c["p"] < 100
    ]
    for b in spikes:
        b["t"] = None
        b["td"] = None
    if sum(1 for L in levels if L["t"] is not None and L["p"] >= 20) < (3 if upper_part else WIS2_MIN_LEVELS):
        stats["qc_few"] += 1
        return None
    bad = deep_superadiabatic(levels)
    if bad:
        stats["qc_superadiabatic"] += 1
        if stats["qc_superadiabatic"] <= 8:
            print(
                f"    QC {who}: deep superadiabatic {bad[0]:.0f}->{bad[1]:.0f} hPa, {bad[2]} K/km — profile rejected",
                flush=True,
            )
        return None
    return levels


def deep_superadiabatic(levels: list, depth: float = 300.0, lapse: float = 0.015) -> tuple | bool:
    """(p bottom, p top, K/km) of the first offending layer, else False: when the troposphere (above the lowest 200 m) cools faster than 15 K/km over a layer >= 300 m
    deep (dry adiabatic
    is 9.8 K/km; nature never sustains that). Catches mis-coded temperatures that pass every
    per-level check - e.g. PAGASA BUFR 2026-10-06 with Davao at -4.3 C at 850 hPa."""
    th = sorted((L for L in levels if L["t"] is not None and L["p"] >= 100), key=lambda L: -L["p"])
    if len(th) < 3:
        return False
    z = [0.0]
    for a, b in itertools.pairwise(th):  # hypsometric thickness from T alone
        z.append(z[-1] + 29.27 * ((a["t"] + b["t"]) / 2 + 273.15) * math.log(a["p"] / b["p"]))
    j = 0
    for i in range(len(th)):
        # a hot surface layer can be strongly superadiabatic: a layer starting in the lowest 200 m
        # must be >= 500 m deep and cool faster than 22 K/km to count
        d, lr = (depth, lapse) if z[i] >= 200 else (500.0, 0.022)
        j = max(j, i + 1)
        while j < len(th) and z[j] - z[i] < d:
            j += 1
        if j >= len(th):
            break
        if (th[i]["t"] - th[j]["t"]) / (z[j] - z[i]) > lr:
            return (th[i]["p"], th[j]["p"], round((th[i]["t"] - th[j]["t"]) / (z[j] - z[i]) * 1000, 1))
        if z[i] < 200:
            j = i + 1  # the two windows differ: rescan for the next start
    return False


def standalone(levels: list) -> bool:
    th = [L for L in levels if L["t"] is not None and L["p"] >= 20]
    return len(th) >= WIS2_MIN_LEVELS and max(L["p"] for L in th) >= 500


def richness(levels: list) -> tuple:
    """(complete, n): complete = T, Td, wind and height all reported; n = thermo levels."""
    th = [L for L in levels if L["t"] is not None and L["p"] >= 20]

    def has(k):
        return sum(1 for L in th if L.get(k) is not None) >= 5

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
        out.append({"p": f[0], "h": f[1], "t": f[2], "td": f[3], "wd": f[8], "ws": f[9], "et": None, "sig": 0})
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
        todo = [
            c
            for c in d.get("all", [])
            if c not in WIS2_NATIVE
            and c not in WIS2_SKIP
            and c not in WIS2_GATEWAYS
            and c not in found
            and (c not in checked or (NOW - datetime.fromisoformat(checked[c])).total_seconds() > 86400)
        ]
        todo.sort(key=lambda c: checked.get(c, ""))
        for c in todo[:WIS2_DISCOVER_PER_RUN]:
            objs, _ = s3_list(f"data/{c}/{WIS2_TEMP}", ctr, max_keys=50, max_pages=1)
            checked[c] = NOW.isoformat()
            if any(
                not k.endswith((".txt", ".json")) and lm > NOW - timedelta(hours=WIS2_MAX_AGE_H) for k, _, lm in objs
            ):
                found.append(c)
                print(f"  WIS2: discovered a new temp/ topic at {c}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  WIS2 centre rediscovery skipped: {repr(e)[:80]}", flush=True)
    return list(WIS2_NATIVE) + [c for c in found if c not in WIS2_NATIVE]


# Active stations missing from stations.json (no IGRA history yet), named from their BUFR position
EXTRA_NAMES = {"73140": "Chibougamau Ua; Que"}  # 49.78 N 74.53 W, ECCC CGCB (2026-10-06)


def bbb_rank(bbb) -> int:
    """Correction rank of a GTS BBB indicator: CCA/AAA 1, CCB/AAB 2...; original or RRx (late) 0."""
    return ord(bbb[2]) - 64 if bbb and bbb[:2] in ("CC", "AA") and "A" <= bbb[2] <= "Z" else 0


def shared_score(state: dict) -> dict:
    """fn -> [complete, n, kind, cc] of the held WIS2/ECCC launch files (was state.wis2.score)."""
    if "score" not in state:
        state["score"] = (state.get("wis2") or {}).pop("score", {})
    return state["score"]


def take_subset(s: dict, c: dict, hints: list, by_wmo: dict, by_fn: dict, stats: Counter, oldest: datetime) -> bool:
    """QC one decoded subset (WIS2 or ECCC); a usable launch goes into by_fn[file name].

    c: the message's {kind, by, src[, cc]} - kind "K" = GTS TEMP parts (may merge), "S" =
    entire sounding, "N" = national WIS2 topic, "H" = ECCC high-resolution to 100 hPa
    (standalone, never merged); cc = correction rank (0 original, 1 CCA, ...)."""
    stats["subsets"] += 1
    tag = c.get("src", "WIS2")
    sid = s["wmo"]
    if sid.startswith("?"):
        sid = sid[1:]
        if sid not in by_wmo:
            stats["unknown_wsi"] += 1
            return False
    launch = s["launch"]
    if launch > NOW + timedelta(hours=1) or launch < oldest:
        stats["too_old_or_future"] += 1
        return False
    st = by_wmo.get(sid)
    la, lo = s["la"], s["lo"]
    if st:
        if (
            la is not None
            and lo is not None
            and (
                abs(la - st["la"]) > 1.5
                or abs((lo - st["lo"] + 180) % 360 - 180) * math.cos(math.radians(st["la"])) > 1.5
            )
        ):
            stats["position_mismatch"] += 1
            if stats["position_mismatch"] <= 6:
                print(
                    f"    {tag} {sid}: BUFR position {la:.2f},{lo:.2f} vs station list "
                    f"{st['la']:.2f},{st['lo']:.2f} — skipped",
                    flush=True,
                )
            return False
        la, lo, elev, name = st["la"], st["lo"], st.get("e") or s["elev"] or 0, st["n"]
    else:
        if la is None or lo is None or not (-90 <= la <= 90 and -180 <= lo <= 360):
            stats["no_position"] += 1
            return False
        lo = (lo + 180) % 360 - 180
        elev, name = s["elev"] or 0, EXTRA_NAMES.get(sid, f"WMO {sid}")
    levels = qc_levels(s["levels"], stats, who=f"{tag} {sid} {c['by']}")
    upper = False
    if not levels and c["kind"] == "K":  # TEMP parts C/D: kept to merge only
        levels = qc_levels(s["levels"], Counter(), upper_part=True, who=f"{tag} {sid} {c['by']} (part)")
        upper = bool(levels)
    if not levels:
        return False
    nom = nominal_hour(launch, [*hints, s.get("typ")])
    if nom < NOW - timedelta(hours=RETAIN_H):
        return False
    fn = f"{sid}_{_tag(nom)}.csv"
    by_fn.setdefault(fn, []).append(
        {
            "sid": sid,
            "levels": levels,
            "launch": launch,
            "kind": c["kind"],
            "by": c["by"],
            "src": tag,
            "cc": c.get("cc", 0),
            "la": la,
            "lo": lo,
            "elev": elev,
            "name": name,
            "rich": richness(levels),
            "upper": upper,
        }
    )
    return True


# Precedence between the real-time sources for one station and launch hour (file name):
#   * a launch is ranked (complete, n thermo levels, correction rank, source tiebreak) - see rank();
#     a candidate replaces the held file only when it ranks strictly higher, so a coarser file
#     never replaces a richer one and a correction (CCA, CCB...) replaces its original;
#   * ECCC (Canada's own MSC Datamart BUFR) wins a tie with a WIS2 relay of the same bulletin;
#   * IEM keeps North America against WIS2 (the long-standing rule: WIS2 never overwrites an IEM
#     file, IEM replaces a WIS2 one) - but ECCC competes with IEM on richness both ways, because the
#     Datamart file is the native ~1-s ascent while IEM carries mandatory + significant levels.
SRC_TIE = {"ECCC": 1}


def rank(rich, cc: int, src: str) -> tuple:
    return (int(rich[0]), int(rich[1]), int(cc or 0), SRC_TIE.get(src, 0))


def held_rank(fn: str, dest: Path, meta: dict, score: dict) -> tuple | None:
    """Rank of the file currently held for fn, or None when nothing (usable) is held."""
    cur = meta["filesrc"].get(fn)
    if not cur or not dest.exists():
        return None
    sc = score.get(fn)
    if cur in ("WIS2", "ECCC", "TAC") and sc:
        return rank(sc[:2], sc[3] if len(sc) > 3 else 0, cur)
    try:  # IEM (or a score lost): from the file itself
        return rank(richness(levels_from_csv(dest.read_text())), 0, cur)
    except Exception:  # noqa: BLE001
        return None


def commit_launches(
    outdir: Path, by_fn: dict, meta: dict, score: dict, src: str, stats: Counter, ctr: Counter, by_wmo: dict
) -> dict:
    """Per station and launch hour keep the richest candidate; TEMP parts (WIS2 IUK) merge -
    with each other and with parts already written by an earlier run. -> {fn: sid} written."""
    got = {}
    for fn, cs in by_fn.items():
        dest = outdir / "soundings" / fn
        cur = meta["filesrc"].get(fn)
        if cur == "IEM" and src == "WIS2":
            stats["kept_iem"] += 1  # North America: IEM's launch stands
            continue
        if src == "TAC" and cur in ("IEM", "ECCC", "WIS2"):
            stats["kept_bufr"] += 1  # text TEMP only where no BUFR is held
            continue
        old = held_rank(fn, dest, meta, score)
        sc = score.get(fn) if cur in ("WIS2", "ECCC", "TAC") else None
        pool = [x for x in cs if not x["upper"]]
        parts = [x for x in cs if x["kind"] == "K"]
        base = []
        if parts and cur == "WIS2" and sc and sc[2] == "K":
            try:
                base = levels_from_csv(dest.read_text())
            except Exception:  # noqa: BLE001
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
        best = max(pool, key=lambda x: rank(x["rich"], x["cc"], x["src"]))
        if old and rank(best["rich"], best["cc"], best["src"]) <= old:
            stats["not_richer"] += 1  # never replace a richer launch
            continue
        text = to_csv(thin_levels(best["levels"], MAX_LEVELS - 2), best["launch"], best["la"], best["lo"], best["elev"])
        if not text:
            stats["csv_short"] += 1
            continue
        if cur and cur != src:
            stats[f"replaced_{cur}"] += 1
        dest.write_text(text)
        meta["filesrc"][fn] = src
        code = credit_code(best["by"], by_wmo.get(best["sid"]))
        meta["fileby"][fn] = "ECCC" if src == "ECCC" else "TAC:" + code if src == "TAC" else code
        meta["names"][best["sid"]] = {"n": best["name"], "la": best["la"], "lo": best["lo"]}
        score[fn] = [best["rich"][0], best["rich"][1], best["kind"], best["cc"]]
        got[fn] = best["sid"]
        ctr["by_" + best["kind"]] += 1
    return got


def run_wis2(outdir: Path, state: dict, by_wmo: dict, meta: dict) -> dict:
    """Mirror new WIS2 radiosonde launches into soundings/; returns {fn: sid} written."""
    t0 = time.time()
    ctr, stats = Counter(), Counter()
    w = state.setdefault("wis2", {})
    if w.get("v") != 2:  # v1 (first day) marked keys seen, then a bug aborted: start over
        w.pop("seen", None)
        w["v"] = 2
    seen = w.setdefault("seen", {})  # dedupe id -> LastModified (epoch h)
    newseen = {}  # merged into `seen` only after this run's files are written
    score = shared_score(state)  # fn -> [complete, n, kind, cc]
    dec = BufrDecoder()  # ImportError -> caller skips WIS2

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
        except Exception as e:  # noqa: BLE001
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
                if a1 not in "KS":  # IUJ pibal, IUD dropsonde, IUE ozone...
                    continue
                if kind == "N":
                    k = a1  # a national copy of TEMP parts / entire
            native = src not in WIS2_GATEWAYS
            # one id for the DWD and JMA copies of a bulletin (same header + size)
            did = f"G:{''.join(x or '' for x in hdr)}:{size}" if hdr and not native else f"N:{src}/{name}"
            if did in seen:
                continue
            by = src if native else (hdr[1] if hdr else src)
            pri = (0 if k == "N" else 1 if k == "S" else 2) + (2 if by in NA_CCCC else 0)
            cand.append(
                {
                    "key": key,
                    "size": size,
                    "lm": lm,
                    "did": did,
                    "kind": k,
                    "by": by,
                    "hdr": hdr,
                    "pri": pri,
                    "cc": bbb_rank(hdr[3] if hdr else None),
                }
            )
    # in-run dedupe of the two gateway copies. Order: newest 6-hour arrival window first
    # (a backlog - first run, or after an outage - never starves the current launches),
    # within it national/IUS before IUK before North America, newest first
    uniq = {}
    for c in sorted(cand, key=lambda c: ((NOW - c["lm"]).total_seconds() // 21600, c["pri"], -c["lm"].timestamp())):
        uniq.setdefault(c["did"], c)
    todo = list(uniq.values())
    print(
        f"  WIS2: {ctr['list']} list request(s) ({ctr['list_bytes'] / 1e6:.2f} MB), "
        f"{len(centres)} centres + 2 gateways; {len(todo)} new message(s) to fetch",
        flush=True,
    )

    # 2. GET + decode, within the request and time budgets
    by_fn: dict = {}
    for c in todo:
        if ctr["get"] >= WIS2_MAX_GET or time.time() - t0 > WIS2_MAX_S:
            ctr["deferred"] += 1  # stays unseen -> next run
            continue
        try:
            blob = get(WIS2_BASE + urllib.parse.quote(c["key"]), timeout=60, tries=2, backoff=5)
        except urllib.error.HTTPError as e:
            ctr["get_err"] += 1
            if e.code in (403, 404):  # expired from the cache
                newseen[c["did"]] = int(c["lm"].timestamp() // 3600)
            continue
        except Exception:  # noqa: BLE001
            ctr["get_err"] += 1
            continue
        ctr["get"] += 1
        ctr["get_bytes"] += len(blob)
        newseen[c["did"]] = int(c["lm"].timestamp() // 3600)
        hints = [_hdr_time(c["hdr"][2], c["lm"])] if c["hdr"] else []
        try:
            subsets = dec.decode(blob, stats)
        except Exception:  # noqa: BLE001
            stats["decode_error"] += 1
            subsets = []
        for s in subsets:
            try:
                take_subset(s, c, hints, by_wmo, by_fn, stats, oldest)
            except Exception as e:  # noqa: BLE001 - one bad subset only
                stats["process_error"] += 1
                if stats["process_error"] <= 3:
                    print(f"::warning::WIS2 {c['key'].rsplit('/', 1)[-1]}: {repr(e)[:120]}", flush=True)

    # 3. per station and launch hour keep the richest message (shared with ECCC)
    got = commit_launches(outdir, by_fn, meta, score, "WIS2", stats, ctr, by_wmo)

    seen.update(newseen)
    # bookkeeping: forget keys past the cache's retention, scores of files no longer held
    cut = int((NOW - timedelta(hours=WIS2_SEEN_H)).timestamp() // 3600)
    for k in [k for k, v in seen.items() if v < cut]:
        del seen[k]
    print(
        f"  WIS2: {ctr['get']} GET(s) {ctr['get_bytes'] / 1e6:.2f} MB, {ctr['get_err']} failed, "
        f"{ctr['deferred']} deferred to the next run (budget {WIS2_MAX_GET} GETs / {WIS2_MAX_S:.0f} s); "
        f"{stats['subsets']} subsets -> {len(got)} launch file(s) written/upgraded "
        f"(entire/native {ctr['by_S'] + ctr['by_N']}, TEMP parts {ctr['by_K']}) "
        f"in {time.time() - t0:.0f} s",
        flush=True,
    )
    print("  WIS2 skips: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()) if k != "subsets"), flush=True)
    meta.update(
        wis2_list=ctr["list"],
        wis2_list_bytes=ctr["list_bytes"],
        wis2_get=ctr["get"],
        wis2_bytes=ctr["get_bytes"],
        wis2_written=len(got),
    )
    return got


# ---------------------------------------------------------------- ECCC (Canada, real time)
# Environment and Climate Change Canada's MSC Datamart publishes every Canadian upper-air
# station's own BUFR bulletins within minutes of reception (ECCC Data Servers End-use Licence;
# attribution "Data Source: Environment and Climate Change Canada"). Layout (2026-10-06 probe):
#   https://dd.weather.gc.ca/YYYYMMDD/WXO-DD/bulletins/alphanumeric/YYYYMMDD/IU/<CCCC>/<HH>/
#   YYYYMMDD and HH are the RECEPTION date and hour (not the nominal time): one directory per
#   station (CCCC = the station's own indicator, e.g. CWSE = Stony Plain 71119), so a 12Z launch
#   lands in 12/ (the first part, ~12:15) and 13/ (the entire sounding, ~13:15).
#   File names TTAAii_CCCC_YYGGgg_BBB__seq, e.g. IUKB01_CWSE_061200___51534.
#   IUK = native ~1-s ascent (BUFR 309057) from the surface to 100 hPa, posted ~1 h after launch;
#   IUS = the same ascent entire, to burst (~6-7k levels), ~2 h after launch; CCA/CCB = corrections
#   (often the same ascent with a longer tail); IUW / IUJ = PILOT wind-only (309051) - skipped.
#   Off-hour specials sit in their own reception-hour directories (18/, 19/ ...).
ECCC_DD = "https://dd.weather.gc.ca/{d}/WXO-DD/bulletins/alphanumeric/{d}/IU/"
ECCC_NAME = "Environment and Climate Change Canada (MSC Datamart)"
ECCC_MAX_LIST = int(os.environ.get("SKEWT_ECCC_MAX_LIST", "200"))
ECCC_MAX_GET = int(os.environ.get("SKEWT_ECCC_MAX_GET", "90"))
ECCC_MAX_S = float(os.environ.get("SKEWT_ECCC_MAX_S", "240"))
ECCC_DONE_MIN = 75  # a reception-hour directory is final this long after its hour began
ECCC_KIND = {"IUS": ("S", 2), "IUK": ("H", 1)}  # TTA -> (kind, rank): entire beats to-100-hPa
ECCC_NAME_RE = re.compile(r"^(IU[A-Z])[A-Z](\d\d)_([A-Z]{4})_(\d{6})_([A-Z]{3})?_")
DD_ROW_RE = re.compile(r'<a href="([^"?/][^"]*?)(/?)">[^<]*</a>\s+(\d{4}-\d\d-\d\d \d\d:\d\d)\s+(\S+)')


def dd_list(url: str, ctr: Counter) -> list | None:
    """Apache index -> [(name, is_dir, mtime UTC, size text)]; None on 404 / budget."""
    if ctr["list"] >= ECCC_MAX_LIST:
        ctr["list_capped"] += 1
        return None
    try:
        raw = get(url, timeout=40, tries=2, backoff=5)
    except urllib.error.HTTPError as e:
        ctr["list"] += 1
        if e.code == 404:
            return []
        raise
    ctr["list"] += 1
    ctr["list_bytes"] += len(raw)
    return [
        (n, bool(d), datetime.strptime(t, "%Y-%m-%d %H:%M").replace(tzinfo=UTC), sz)
        for n, d, t, sz in DD_ROW_RE.findall(raw.decode("utf-8", "replace"))
    ]


def run_eccc(outdir: Path, state: dict, by_wmo: dict, meta: dict) -> dict:
    """Mirror new Canadian launches from the MSC Datamart; returns {fn: sid} written.

    Polls only today's and yesterday's (UTC) reception dates. A station directory is
    re-listed only when its mtime changed (a new hour directory), an hour directory until
    it is final (ECCC_DONE_MIN after its hour began); per launch (CCCC + YYGGgg) only a
    file that outranks what was already fetched (IUS > IUK, then later corrections) is GET."""
    t0 = time.time()
    ctr, stats = Counter(), Counter()
    e = state.setdefault("eccc", {})
    cdir = e.setdefault("cdir", {})  # "date/CCCC" -> station dir mtime at our last listing
    hdir = e.setdefault("hdir", {})  # "date/CCCC/HH" -> 1 once final and listed
    best = e.setdefault("best", {})  # "CCCC/YYGGgg" -> [kind rank, cc, epoch h]
    cmap = e.setdefault("cccc", {})  # CCCC -> WMO id, learned from the BUFR itself
    daydone = e.setdefault("daydone", {})
    dec = BufrDecoder()  # ImportError -> caller skips ECCC
    today = NOW.strftime("%Y%m%d")
    dates = [today, (NOW - timedelta(days=1)).strftime("%Y%m%d")]
    types, hour_todo, roots_ok = Counter(), [], {}
    for d in dates:
        if daydone.get(d):
            continue
        base = ECCC_DD.format(d=d)
        try:
            root = dd_list(base, ctr)
        except Exception as ex:  # noqa: BLE001
            print(f"  ECCC list {d} failed: {repr(ex)[:80]}", flush=True)
            continue
        if root is None:
            continue
        roots_ok[d] = True
        for cccc, isdir, mt, _ in root:
            if not isdir:
                continue
            key = f"{d}/{cccc}"
            stamp = f"{mt:%Y-%m-%d %H:%M}"
            if cdir.get(key) != stamp:
                try:
                    rows = dd_list(f"{base}{cccc}/", ctr)
                except Exception as ex:  # noqa: BLE001
                    print(f"  ECCC list {key} failed: {repr(ex)[:80]}", flush=True)
                    roots_ok[d] = False
                    continue
                if rows is None:
                    roots_ok[d] = False
                    continue
                for hh, hd, _, _ in rows:
                    if hd and re.fullmatch(r"\d\d", hh):
                        hdir.setdefault(f"{key}/{hh}", 0)
                cdir[key] = stamp
            hour_todo += [k for k, v in hdir.items() if k.startswith(key + "/") and not v]
    # hour directories newest first, so a backlog never starves the current launches
    hour_todo.sort(key=lambda k: (k[:8], k[-2:]), reverse=True)
    files = []
    for hk in hour_todo:
        d, cccc, hh = hk.split("/")
        when = datetime.strptime(d + hh, "%Y%m%d%H").replace(tzinfo=UTC)
        try:
            rows = dd_list(f"{ECCC_DD.format(d=d)}{cccc}/{hh}/", ctr)
        except Exception as ex:  # noqa: BLE001
            print(f"  ECCC list {hk} failed: {repr(ex)[:80]}", flush=True)
            continue
        if rows is None:
            continue
        if when + timedelta(minutes=ECCC_DONE_MIN) <= NOW:
            hdir[hk] = 1
        for name, isdir, mt, _ in rows:
            m = ECCC_NAME_RE.match(name)
            if isdir or not m:
                continue
            tta, _, cc4, yyggg, bbb = m.groups()
            types[tta] += 1
            if tta not in ECCC_KIND or cc4 != cccc:
                continue
            kind, kr = ECCC_KIND[tta]
            files.append(
                {
                    "url": f"{ECCC_DD.format(d=d)}{cccc}/{hh}/{name}",
                    "name": name,
                    "hk": hk,
                    "grp": f"{cccc}/{yyggg}",
                    "kind": kind,
                    "kr": kr,
                    "cc": bbb_rank(bbb),
                    "lm": mt,
                    "yyggg": yyggg,
                    "cccc": cccc,
                }
            )
    # per launch: the single best new file (entire > to-100-hPa, then the latest correction)
    pick = {}
    for f in files:
        old = best.get(f["grp"])
        if old and (f["kr"], f["cc"]) <= (old[0], old[1]):
            continue
        cur = pick.get(f["grp"])
        if not cur or (f["kr"], f["cc"], f["lm"]) > (cur["kr"], cur["cc"], cur["lm"]):
            pick[f["grp"]] = f
    todo = sorted(pick.values(), key=lambda f: f["lm"], reverse=True)
    print(
        f"  ECCC: {ctr['list']} listing(s) ({ctr['list_bytes'] / 1e3:.0f} KB), "
        f"{len(hour_todo)} hour dir(s) open; bulletin types seen "
        f"{dict(sorted(types.items()))}; {len(todo)} new launch file(s) to fetch",
        flush=True,
    )

    by_fn, fetched, oldest = {}, {}, NOW - timedelta(hours=RETAIN_H)
    for f in todo:
        if ctr["get"] >= ECCC_MAX_GET or time.time() - t0 > ECCC_MAX_S:
            ctr["deferred"] += 1
            hdir[f["hk"]] = 0  # list it again next run
            continue
        try:
            blob = get(f["url"], timeout=60, tries=2, backoff=5)
        except Exception:  # noqa: BLE001
            ctr["get_err"] += 1
            hdir[f["hk"]] = 0
            continue
        ctr["get"] += 1
        ctr["get_bytes"] += len(blob)
        best[f["grp"]] = [f["kr"], f["cc"], int(f["lm"].timestamp() // 3600)]
        try:
            subsets = dec.decode(blob, stats)
        except Exception:  # noqa: BLE001
            stats["decode_error"] += 1
            subsets = []
        hints = [_hdr_time(f["yyggg"], f["lm"])]
        c = {"kind": f["kind"], "by": "ECCC", "src": "ECCC", "cc": f["cc"]}
        for s in subsets:
            try:
                if take_subset(s, c, hints, by_wmo, by_fn, stats, oldest):
                    sid = s["wmo"].lstrip("?")
                    cmap[f["cccc"]] = sid
                    fetched.setdefault(
                        f"{sid}_{_tag(nominal_hour(s['launch'], [*hints, s.get('typ')]))}.csv", []
                    ).append((f, s["launch"]))
            except Exception as ex:  # noqa: BLE001
                stats["process_error"] += 1
                if stats["process_error"] <= 3:
                    print(f"::warning::ECCC {f['name']}: {repr(ex)[:120]}", flush=True)
    got = commit_launches(outdir, by_fn, meta, shared_score(state), "ECCC", stats, ctr, by_wmo)
    # latency: reception (Datamart mtime) minus launch, per kind, for the files written
    lat = {"H": [], "S": []}
    for fn in got:
        for f, launch in fetched.get(fn, []):
            lat[f["kind"]].append((f["lm"] - launch).total_seconds() / 60)
    late = [
        f"{fn[:-4]} {f['name'][:6]} +{(f['lm'] - launch).total_seconds() / 3600:.0f} h"
        for fn in got
        for f, launch in fetched.get(fn, [])
        if (f["lm"] - launch).total_seconds() > 6 * 3600
    ]
    if late:
        print(f"  ECCC posted > 6 h after launch (backfill/corrections): {', '.join(late[:12])}", flush=True)

    def med(v):
        return sorted(v)[len(v) // 2] if v else None

    for k, nm in (("H", "IUK to 100 hPa"), ("S", "IUS entire")):
        if lat[k]:
            print(
                f"  ECCC latency {nm}: posted {med(lat[k]):.0f} min after launch (median), "
                f"range {min(lat[k]):.0f}-{max(lat[k]):.0f}, n={len(lat[k])}",
                flush=True,
            )
    # bookkeeping: a past day is finished once every station and hour directory is final
    for d in dates[1:]:
        if (
            roots_ok.get(d)
            and datetime.strptime(d, "%Y%m%d").replace(tzinfo=UTC) + timedelta(days=1, minutes=ECCC_DONE_MIN) <= NOW
            and all(v for k, v in hdir.items() if k.startswith(d + "/"))
        ):
            daydone[d] = 1
    keep = set(dates)
    for dct in (cdir, hdir):
        for k in [k for k in dct if k[:8] not in keep]:
            del dct[k]
    for k in [k for k in daydone if k not in keep]:
        del daydone[k]
    cut = int((NOW - timedelta(hours=RETAIN_H + 24)).timestamp() // 3600)
    for k in [k for k, v in best.items() if v[2] < cut]:
        del best[k]
    print(
        f"  ECCC: {ctr['get']} GET(s) {ctr['get_bytes'] / 1e6:.2f} MB, {ctr['get_err']} failed, "
        f"{ctr['deferred']} deferred (budget {ECCC_MAX_LIST} listings / {ECCC_MAX_GET} GETs / "
        f"{ECCC_MAX_S:.0f} s); {stats['subsets']} subsets -> {len(got)} launch file(s) written/upgraded "
        f"(entire {ctr['by_S']}, to 100 hPa {ctr['by_H']}) in {time.time() - t0:.0f} s",
        flush=True,
    )
    sk = {k: v for k, v in sorted(stats.items()) if k != "subsets"}
    if sk:
        print("  ECCC skips: " + ", ".join(f"{k} {v}" for k, v in sk.items()), flush=True)
    print("  ECCC stations (CCCC=WMO): " + " ".join(f"{k}={v}" for k, v in sorted(cmap.items())), flush=True)
    meta.update(
        eccc_list=ctr["list"],
        eccc_list_bytes=ctr["list_bytes"],
        eccc_get=ctr["get"],
        eccc_bytes=ctr["get_bytes"],
        eccc_written=len(got),
    )
    return got


# ---------------------------------------------------------------- TAC TEMP (GTS text, via WIS2)
# Many services - most of Africa among them - still send their soundings to the GTS only as
# alphanumeric FM-35 TEMP, never as BUFR. DWD's WIS2 gateway republishes those bulletins under
# data/de-dwd-gts-to-wis2/data/core/U/{S,K,L,E}/<A1>/<A2>/<ii>/<CCCC>/ (US = part A TTAA, UK = part
# B TTBB, UL = part C TTCC, UE = part D TTDD; 2026-10-07 survey: 74 compiling centres, ~4400 files
# per 72 h, ~3 MB). Decoded here and used ONLY for a station-hour no BUFR source (IEM, ECCC, WIS2
# BUFR) holds; a BUFR launch that arrives later replaces it when richer, as everywhere else.
# Parts A-D (and corrections / delayed copies) of one launch are merged, across runs too.
# Not decoded: PPBB/PILOT (no sample carried any in that survey; every TTBB had its 21212 winds).
TAC_GATEWAY = "de-dwd-gts-to-wis2"
TAC_TREES = ("S", "K", "L", "E")
TAC_PART = {"S": "A", "K": "B", "L": "C", "E": "D"}
TAC_HDR_RE = re.compile(r"(?:^|_)(U[SKLE][A-Z]{2}\d{2})([A-Z]{4})(\d{6})([A-Z]{3})?(?=[_.]|$)")
TAC_MAX_GET = int(os.environ.get("SKEWT_TAC_MAX_GET", "600"))
TAC_MAX_LIST = int(os.environ.get("SKEWT_TAC_MAX_LIST", "16"))
TAC_MAX_S = float(os.environ.get("SKEWT_TAC_MAX_S", "200"))
TAC_MIN_SIZE = 50  # smaller files are "TTAA 5506/ 61024 NIL=" placeholders
STD_Z = {
    1000: 110,
    925: 760,
    850: 1460,
    700: 3010,
    500: 5570,
    400: 7190,
    300: 9160,
    250: 10360,
    200: 11790,
    150: 13610,
    100: 16180,
    70: 18440,
    50: 20580,
    30: 23850,
    20: 26480,
    10: 31060,
}
SEQ_A = [
    ("00", 1000),
    ("92", 925),
    ("85", 850),
    ("70", 700),
    ("50", 500),
    ("40", 400),
    ("30", 300),
    ("25", 250),
    ("20", 200),
    ("15", 150),
    ("10", 100),
]
SEQ_C = [("70", 70), ("50", 50), ("30", 30), ("20", 20), ("10", 10)]
# Id (last standard level with a wind group): candidates tried in order until the parse is consistent
ID_A = {
    "1": (100, 150),
    "2": (200, 250),
    "3": (300,),
    "4": (400,),
    "5": (500,),
    "7": (700,),
    "8": (850,),
    "9": (925,),
    "0": (1000,),
    "/": (None,),
}
ID_C = {"1": (10,), "2": (20,), "3": (30,), "5": (50,), "7": (70,), "/": (None,)}
TAC_STOP = {"21212", "31313", "41414", "51515", "61616", "71717"}
TROP_BIT, MAXW_BIT = 1 << 13, 1 << 12  # flags only, so thin_levels keeps them


def _tac_t(g: str):
    """TTTaDD -> (t, td) in C."""
    if len(g) != 5 or "/" in g[:3] or not g[:3].isdigit():
        return None, None
    ta = int(g[2])
    t = int(g[:2]) + ta / 10
    if ta % 2:
        t = -t
    dd, td = g[3:], None
    if dd.isdigit():
        d = int(dd)
        dep = d / 10 if d <= 50 else d - 50 if d >= 56 else None
        if dep is not None:
            td = t - dep
    return t, td


def _tac_w(g: str, knots: bool):
    """dddff -> (wd, ws m/s); the units digit of ddd carries ff's hundreds."""
    if len(g) != 5 or not g.isdigit():
        return None, None
    ddd, ff = int(g[:3]), int(g[3:])
    ff += 100 * (ddd % 5)
    ddd -= ddd % 5
    if ddd > 360 or (ddd == 0 and ff):
        return None, None
    return float(ddd), ff * 0.514444 if knots else float(ff)


def _tac_h(p: int, hhh: str):
    """Standard-level hhh -> geopotential m (m for 1000-700 hPa, dam above; nearest plausible)."""
    if not hhh.isdigit():
        return None
    v = int(hhh)
    if p == 1000:
        return float(500 - v if v >= 500 else v)
    unit = 1 if p >= 700 else 10
    return float(min(((v + k * 1000) * unit for k in range(5)), key=lambda z: abs(z - STD_Z[p])))


def _lvl(p, h=None, t=None, td=None, w=(None, None), sig=0):
    return {"p": p, "h": h, "t": t, "td": td, "wd": w[0], "ws": w[1], "et": None, "sig": sig}


def _tac_std(toks, i, seq, last_wind, knots):
    out = []
    for code, p in seq:
        if i >= len(toks) or toks[i][:2] != code or toks[i] in TAC_STOP:
            break
        if i + 1 >= len(toks):
            break
        t, td = _tac_t(toks[i + 1])
        lv = _lvl(float(p), _tac_h(p, toks[i][2:]), t, td)
        i += 2
        if last_wind is not None and p >= last_wind and i < len(toks):
            lv["wd"], lv["ws"] = _tac_w(toks[i], knots)
            i += 1
        out.append(lv)
    nxt = toks[i] if i < len(toks) else ""
    ok = not nxt or nxt in TAC_STOP or nxt[:2] in ("88", "77", "66")
    return out, i, ok


def _tac_trop_maxw(toks, i, knots, scale, out):
    while i < len(toks):
        g = toks[i]
        if g[:2] == "88" and len(g) == 5:
            if g[2:] == "999":
                i += 1
                continue
            if not g[2:].isdigit():
                break
            t, td = _tac_t(toks[i + 1]) if i + 1 < len(toks) else (None, None)
            w = _tac_w(toks[i + 2], knots) if i + 2 < len(toks) else (None, None)
            out.append(_lvl(int(g[2:]) * scale, None, t, td, w, TROP_BIT))
            i += 3
        elif g[:2] in ("77", "66") and len(g) == 5:
            if g[2:] == "999":
                i += 1
                continue
            if not g[2:].isdigit() or i + 1 >= len(toks):
                break
            out.append(_lvl(int(g[2:]) * scale, w=_tac_w(toks[i + 1], knots), sig=MAXW_BIT))
            i += 2
            if i < len(toks) and toks[i][0] == "4" and toks[i] not in TAC_STOP:
                i += 1  # vertical wind shear group
        else:
            break
    return i


def _tac_sig(toks, i, knots, part, out):
    """Parts B/D: nnPPP TTTaDD pairs, then 21212 nnPPP dddff wind pairs."""

    def pres(g):
        v = int(g[2:])
        return v / 10 if part == "D" else float(v + 1000 if v < 100 else v)

    def is_lvl(g):  # nnPPP, nn = 00, 11, ... 99; PPP may be missing ("11///")
        return len(g) == 5 and g[:2].isdigit() and g[0] == g[1] and g not in TAC_STOP

    while i + 1 < len(toks) and is_lvl(toks[i]):
        if toks[i][2:].isdigit():
            t, td = _tac_t(toks[i + 1])
            out.append(_lvl(pres(toks[i]), None, t, td, sig=SFC_BIT if part == "B" and toks[i][:2] == "00" else 0))
        i += 2
    if i < len(toks) and toks[i] == "21212":
        i += 1
        while i + 1 < len(toks) and is_lvl(toks[i]):
            if toks[i][2:].isdigit():
                out.append(
                    _lvl(
                        pres(toks[i]),
                        w=_tac_w(toks[i + 1], knots),
                        sig=SFC_BIT if part == "B" and toks[i][:2] == "00" else 0,
                    )
                )
            i += 2
    return i


def parse_temp(toks: list, stats: Counter) -> dict | None:
    """One FM-35 TEMP report (tokens from MiMiMjMj on) -> {wmo, part, day, hour, knots, launch, levels}."""
    if len(toks) < 3 or toks[0] not in ("TTAA", "TTBB", "TTCC", "TTDD"):
        return None
    part = {"TTAA": "A", "TTBB": "B", "TTCC": "C", "TTDD": "D"}[toks[0]]
    if "NIL" in toks:
        stats["tac_nil"] += 1
        return None
    g1, wmo = toks[1], toks[2]
    if len(g1) != 5 or not g1[:4].isdigit() or not re.fullmatch(r"\d{5}", wmo):
        stats["tac_bad_head"] += 1
        return None
    yy, hour, ind = int(g1[:2]), int(g1[2:4]), g1[4]
    knots = yy > 50
    day = yy - 50 if knots else yy
    if not 1 <= day <= 31 or hour > 23:
        stats["tac_bad_head"] += 1
        return None
    out, i = [], 3
    if part in ("A", "C"):
        if part == "A" and i + 2 < len(toks) and toks[i][:2] == "99" and toks[i][2:].isdigit():
            ps = int(toks[i][2:])
            t, td = _tac_t(toks[i + 1])
            out.append(_lvl(float(ps + 1000 if ps < 100 else ps), None, t, td, _tac_w(toks[i + 2], knots), SFC_BIT))
            i += 3
        seq, ids = (SEQ_A, ID_A) if part == "A" else (SEQ_C, ID_C)
        best = None
        for lw in ids.get(ind, (None,)):
            levs, j, ok = _tac_std(toks, i, seq, lw, knots)
            if ok or best is None:
                best = (levs, j)
            if ok:
                break
        else:
            stats["tac_inconsistent"] += 1
        out += best[0]
        i = _tac_trop_maxw(toks, best[1], knots, 1 if part == "A" else 0.1, out)
    else:
        i = _tac_sig(toks, i, knots, part, out)
    launch = None
    if "31313" in toks:  # 31313 srrarasasa 8GGgg: launch time
        k = toks.index("31313")
        if k + 2 < len(toks) and re.fullmatch(r"8\d{4}", toks[k + 2]):
            gg, mm = int(toks[k + 2][1:3]), int(toks[k + 2][3:5])
            if gg < 24 and mm < 60:
                launch = (gg, mm)
    out = [L for L in out if L["p"] and 0.5 <= L["p"] <= 1100]
    if not out:
        stats["tac_empty"] += 1
        return None
    return {"wmo": wmo, "part": part, "day": day, "hour": hour, "knots": knots, "launch": launch, "levels": out}


def tac_reports(text: str, stats: Counter) -> list:
    """Every TEMP report in a GTS bulletin (multi-station bulletins carry several, '=' separated)."""
    lines = text.replace("\r", "\n").split("\n")
    body = " ".join(ln for ln in lines if not re.match(r"^\s*U[A-Z]{3}\d\d [A-Z]{4} \d{6}", ln))
    out = []
    for chunk in body.split("="):
        toks = chunk.split()
        k = next(
            (j for j, t in enumerate(toks) if t in ("TTAA", "TTBB", "TTCC", "TTDD", "PPBB", "PPAA", "PPCC", "PPDD")),
            None,
        )
        if k is None:
            continue
        if toks[k].startswith("PP"):
            stats["tac_pilot_skipped"] += 1
            continue
        r = parse_temp(toks[k:], stats)
        if r:
            out.append(r)
    return out


def run_tac(outdir: Path, state: dict, by_wmo: dict, meta: dict) -> dict:
    """Mirror TAC TEMP launches for station-hours no BUFR source holds; returns {fn: sid} written."""
    t0 = time.time()
    ctr, stats = Counter(), Counter()
    w = state.setdefault("tac", {})
    seen = w.setdefault("seen", {})  # header+size id -> LastModified epoch h
    newseen = {}
    oldest = NOW - timedelta(hours=WIS2_MAX_AGE_H)
    cand = []
    for tree in TAC_TREES:
        try:
            objs, _ = s3_list(f"data/{TAC_GATEWAY}/data/core/U/{tree}/", ctr)
        except Exception as e:  # noqa: BLE001
            print(f"  TAC list U/{tree} failed: {repr(e)[:80]}", flush=True)
            continue
        for key, size, lm in objs:
            if lm < oldest or size > 100_000:
                continue
            hdr = TAC_HDR_RE.search(key.rsplit("/", 1)[-1])
            if not hdr:
                continue
            did = "".join(x or "" for x in hdr.groups()) + f":{size}"
            if did in seen:
                continue
            if size < TAC_MIN_SIZE:  # NIL placeholder: never fetched
                newseen[did] = int(lm.timestamp() // 3600)
                ctr["nil_small"] += 1
                continue
            cand.append({"key": key, "lm": lm, "did": did, "hdr": hdr.groups()})
    cand.sort(key=lambda c: ((NOW - c["lm"]).total_seconds() // 21600, -c["lm"].timestamp()))
    print(
        f"  TAC: {ctr['list']} list request(s) ({ctr['list_bytes'] / 1e6:.2f} MB); "
        f"{len(cand)} new bulletin(s) to fetch, {ctr['nil_small']} NIL placeholders skipped",
        flush=True,
    )

    # fetch + decode; group the reports by station and nominal hour
    parts: dict = {}
    for c in cand:
        if ctr["get"] >= TAC_MAX_GET or time.time() - t0 > TAC_MAX_S:
            ctr["deferred"] += 1
            continue
        try:
            blob = get(WIS2_BASE + urllib.parse.quote(c["key"]), timeout=40, tries=2, backoff=5)
        except urllib.error.HTTPError as e:
            ctr["get_err"] += 1
            if e.code in (403, 404):
                newseen[c["did"]] = int(c["lm"].timestamp() // 3600)
            continue
        except Exception:  # noqa: BLE001
            ctr["get_err"] += 1
            continue
        ctr["get"] += 1
        ctr["get_bytes"] += len(blob)
        newseen[c["did"]] = int(c["lm"].timestamp() // 3600)
        cccc, yyggg, bbb = c["hdr"][1], c["hdr"][2], c["hdr"][3]
        ref = _hdr_time(yyggg, c["lm"])
        try:
            reps = tac_reports(blob.decode("latin-1"), stats)
        except Exception:  # noqa: BLE001
            stats["tac_decode_error"] += 1
            continue
        for r in reps:
            stats["tac_reports"] += 1
            # the report's YYGG is the nominal hour for most senders but the actual launch hour
            # for some (South Africa: 11Z for a 12Z sounding); 31313 8GGgg is the launch time
            rt = _hdr_time(f"{r['day']:02d}{r['hour']:02d}00", c["lm"])
            if not rt:
                stats["tac_bad_time"] += 1
                continue
            launch = rt
            if r["launch"]:
                gg, mm = r["launch"]
                launch = rt.replace(hour=gg, minute=mm)
                if launch - rt > timedelta(hours=12):
                    launch -= timedelta(days=1)
                elif rt - launch > timedelta(hours=12):
                    launch += timedelta(days=1)
                if abs((launch - rt).total_seconds()) > 3 * 3600:
                    launch = rt
            nom = nominal_hour(launch, [rt, ref])
            if nom < NOW - timedelta(hours=RETAIN_H) or nom > NOW + timedelta(hours=2):
                stats["tac_too_old_or_future"] += 1
                continue
            sid = r["wmo"]
            if sid not in by_wmo:
                stats["tac_unknown_station"] += 1
                continue
            fn = f"{sid}_{_tag(nom)}.csv"
            parts.setdefault(fn, []).append(
                {"r": r, "cccc": cccc, "cc": bbb_rank(bbb), "lm": c["lm"], "launch": launch, "sid": sid}
            )

    # per station-hour: newest correction of each part, merged (with a TAC file held from an
    # earlier run), QC'd, then committed through the shared precedence rules
    by_fn = {}
    for fn, ps in parts.items():
        cur = meta["filesrc"].get(fn)
        if cur in ("IEM", "ECCC", "WIS2"):
            stats["kept_bufr"] += 1  # a BUFR / IEM launch holds this hour
            continue
        newest = {}
        for x in ps:
            k = x["r"]["part"]
            if k not in newest or (x["cc"], x["lm"]) >= (newest[k]["cc"], newest[k]["lm"]):
                newest[k] = x
        sets = [x["r"]["levels"] for x in newest.values()]
        dest = outdir / "soundings" / fn
        if cur == "TAC" and dest.exists():
            with contextlib.suppress(Exception):
                sets.insert(0, levels_from_csv(dest.read_text()))
        levels = merge_levels(*[[dict(L) for L in s] for s in sets])
        levels = qc_levels(levels, stats, who=f"TAC {fn[:-4]} {ps[0]['cccc']}")
        if not levels:
            continue
        x0 = newest.get("A") or newest.get("B") or next(iter(newest.values()))
        st = by_wmo[x0["sid"]]
        by_fn[fn] = [
            {
                "sid": x0["sid"],
                "levels": levels,
                "launch": x0["launch"],
                "kind": "T",
                "by": x0["cccc"],
                "src": "TAC",
                "cc": 0,
                "la": st["la"],
                "lo": st["lo"],
                "elev": st.get("e") or 0,
                "name": st["n"],
                "rich": richness(levels),
                "upper": False,
                "parts": "".join(sorted(newest)),
            }
        ]
    got = commit_launches(outdir, by_fn, meta, shared_score(state), "TAC", stats, ctr, by_wmo)
    seen.update(newseen)
    cut = int((NOW - timedelta(hours=WIS2_SEEN_H)).timestamp() // 3600)
    for k in [k for k, v in seen.items() if v < cut]:
        del seen[k]
    nlev = sorted(by_fn[fn][0]["rich"][1] for fn in got)
    print(
        f"  TAC: {ctr['get']} GET(s) {ctr['get_bytes'] / 1e6:.2f} MB, {ctr['get_err']} failed, "
        f"{ctr['deferred']} deferred (budget {TAC_MAX_GET} GETs / {TAC_MAX_S:.0f} s); "
        f"{stats['tac_reports']} reports -> {len(got)} launch file(s) written/upgraded"
        + (f", median {nlev[len(nlev) // 2]} thermo levels" if nlev else "")
        + f" in {time.time() - t0:.0f} s",
        flush=True,
    )
    print("  TAC skips: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()) if k != "tac_reports"), flush=True)
    meta.update(
        tac_list=ctr["list"],
        tac_list_bytes=ctr["list_bytes"],
        tac_get=ctr["get"],
        tac_bytes=ctr["get_bytes"],
        tac_written=len(got),
    )
    return got


def credit_code(by: str, st: dict | None) -> str:
    """Attribution code of one launch: the centre, or "HUB@FIPS" when a hub relayed a
    station from another country."""
    home = RELAY_HOME.get(by)
    gid = (st or {}).get("gid") or ""
    if home and len(gid) >= 2 and gid[:2] not in home:
        return f"{by}@{gid[:2]}"
    return by


def centre_label(code: str) -> str:
    if code == "ECCC":
        return ECCC_NAME
    if code.startswith("TAC:"):
        return centre_label(code[4:])
    if "@" in code:
        hub, fips = code.split("@", 1)
        who = f"the national service of {FIPS_NAME[fips]}" if fips in FIPS_NAME else "the national service"
        return f"{who}, relayed by {centre_label(hub)}"
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
    meta = {"filesrc": {}, "fileby": {}, "names": state.get("names", {}), "score": shared_score(state)}
    # Carry forward this pipeline's own per-launch files inside the retention window.
    # Files from a manifest without the pipeline mark came from the retired feed and
    # are deliberately NOT carried.
    if state:
        carried = Counter()
        oldby = state.get("fileby") or {}
        for fn, src in (state.get("filesrc") or {}).items():
            if src not in ("IEM", "WIS2", "ECCC", "TAC"):
                continue  # IGRA launches are not mirrored (the browser fetches them)
            f = prevdir / "soundings" / fn
            try:
                ts = datetime.strptime(fn[:-4].split("_")[1], "%Y%m%d%H").replace(tzinfo=UTC)
            except (IndexError, ValueError):
                continue
            if ts >= cutoff and f.exists():
                if src != "IEM":
                    try:  # files written before the deep-superadiabatic check existed
                        bad = deep_superadiabatic(levels_from_csv(f.read_text()))
                        if bad:
                            carried["dropped_superadiabatic"] += 1
                            print(
                                f"    dropped {fn} ({src}, {meta['fileby'].get(fn) or oldby.get(fn) or ''}): "
                                f"deep superadiabatic {bad[0]:.0f}->{bad[1]:.0f} hPa, {bad[2]} K/km",
                                flush=True,
                            )
                            continue
                    except Exception:  # noqa: BLE001
                        pass
                shutil.copyfile(f, outdir / "soundings" / fn)
                meta["filesrc"][fn] = src
                if src == "WIS2":
                    meta["fileby"][fn] = credit_code(oldby.get(fn, ""), by_wmo.get(fn.split("_")[0]))
                elif src == "TAC":
                    meta["fileby"][fn] = oldby.get(fn) or "TAC:?"
                elif src == "ECCC":
                    meta["fileby"][fn] = "ECCC"
                carried[src] += 1
        print(
            f"  carried forward {sum(v for k, v in carried.items() if not k.startswith('dropped'))} "
            f"launch file(s) {dict(carried)}",
            flush=True,
        )
    else:
        print("  no previous state from this pipeline — starting fresh", flush=True)

    run_iem(outdir, state, by_wmo, icao2wmo, meta)
    # ECCC and WIS2 must never take the IEM/IGRA publication down with them
    if os.environ.get("SKEWT_ECCC", "1") != "0":
        try:
            run_eccc(outdir, state, by_wmo, meta)
        except ImportError as e:
            print(f"::warning::ECCC skipped — ecCodes unavailable ({e})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"::warning::ECCC failed this run: {repr(e)[:160]}", flush=True)
    if os.environ.get("SKEWT_WIS2", "1") != "0":
        try:
            run_wis2(outdir, state, by_wmo, meta)
        except ImportError as e:
            print(f"::warning::WIS2 skipped — ecCodes unavailable ({e})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"::warning::WIS2 failed this run: {repr(e)[:160]}", flush=True)
    if os.environ.get("SKEWT_TAC", "1") != "0":
        try:
            run_tac(outdir, state, by_wmo, meta)
        except Exception as e:  # noqa: BLE001
            print(f"::warning::TAC failed this run: {repr(e)[:160]}", flush=True)
    igra = run_igra(state, by_gid, prev_manifest, meta)

    # manifest: per station, every launch held + the newest one's source (and, where
    # WIS2 is involved, the originating centre per launch)
    hours: dict = {}
    for fn, src in meta["filesrc"].items():
        sid, tag = fn[:-4].split("_")
        code = "IEM" if src == "IEM" else "ECCC" if src == "ECCC" else (meta["fileby"].get(fn) or "WIS2")
        hours.setdefault(sid, []).append((f"{tag[:4]}-{tag[4:6]}-{tag[6:8]} {tag[8:10]}:00", src, code))
    entries, used = {}, set()
    for sid, hs in hours.items():
        hs.sort()
        dt, src, code = hs[-1]
        s = by_wmo.get(sid) or meta["names"].get(sid)
        if not s:
            continue
        e = {
            "id": sid,
            "n": s["n"],
            "la": round(s["la"], 3),
            "lo": round(s["lo"], 3),
            "src": src,
            "dt": dt,
            "hours": [h[0] for h in hs],
        }
        if any(h[1] in ("WIS2", "ECCC", "TAC") for h in hs):
            e["by"] = code
            hx = {h[0]: h[2] for h in hs if h[2] != code}
            if hx:
                e["hx"] = hx
            used.update(h[2] for h in hs if h[1] in ("WIS2", "ECCC", "TAC"))
        entries[sid] = e
        shutil.copyfile(
            outdir / "soundings" / f"{sid}_{dt.replace('-', '').replace(' ', '')[:10]}.csv",
            outdir / "soundings" / f"{sid}.csv",
        )
    n_rt = Counter(e["src"] for e in entries.values())
    # IGRA stations: listed without per-launch files; "upd" = the NCEI daily update
    # that brought their newest launches (the launches themselves are ~1-2 days old).
    # A station either real-time feed holds is never listed as IGRA.
    for sid, upd in igra.items():
        if sid in entries or sid not in by_wmo:
            continue
        s = by_wmo[sid]
        entries[sid] = {
            "id": sid,
            "n": s["n"],
            "la": round(s["la"], 3),
            "lo": round(s["lo"], 3),
            "src": "IGRA",
            "dt": None,
            "upd": upd,
        }
    n_igra = len(entries) - sum(n_rt.values())
    (outdir / "manifest.json").write_text(
        json.dumps(
            {
                "pipeline": PIPELINE,
                "generated": NOW.strftime("%Y-%m-%d %H:%M UTC"),
                "sources": {
                    "IEM": n_rt["IEM"],
                    "ECCC": n_rt["ECCC"],
                    "WIS2": n_rt["WIS2"],
                    "TAC": n_rt["TAC"],
                    "IGRA": n_igra,
                },
                "centres": {c: centre_label(c) for c in sorted(used)},
                "entries": entries,
            },
            ensure_ascii=False,
        )
    )
    state["filesrc"] = meta["filesrc"]
    score = shared_score(state)
    for fn in [fn for fn in score if meta["filesrc"].get(fn) not in ("WIS2", "ECCC", "TAC")]:
        del score[fn]
    state["fileby"] = {fn: b for fn, b in meta["fileby"].items() if meta["filesrc"].get(fn) in ("WIS2", "TAC")}
    state["names"] = {k: v for k, v in meta["names"].items() if k in entries}
    (outdir / "state.json").write_text(json.dumps(state, ensure_ascii=False))
    nf = Counter(meta["filesrc"].values())
    print(
        f"  manifest: {len(entries)} stations ({n_rt['IEM']} IEM + {n_rt['ECCC']} ECCC + {n_rt['WIS2']} WIS2 "
        f"+ {n_rt['TAC']} TAC real time with {nf['IEM']} + {nf['ECCC']} + {nf['WIS2']} + {nf['TAC']} launch files, "
        f"{n_igra} IGRA on demand) "
        f"→ {outdir}",
        flush=True,
    )
    print(
        f"  transfer this run: IEM {meta.get('iem_req', 0)} request(s) "
        f"{meta.get('iem_bytes', 0) / 1e6:.2f} MB; ECCC {meta.get('eccc_list', 0)} listing(s) "
        f"{meta.get('eccc_list_bytes', 0) / 1e6:.2f} MB + {meta.get('eccc_get', 0)} GET(s) "
        f"{meta.get('eccc_bytes', 0) / 1e6:.2f} MB; WIS2 {meta.get('wis2_list', 0)} list(s) "
        f"{meta.get('wis2_list_bytes', 0) / 1e6:.2f} MB + {meta.get('wis2_get', 0)} GET(s) "
        f"{meta.get('wis2_bytes', 0) / 1e6:.2f} MB; TAC {meta.get('tac_list', 0)} list(s) "
        f"{meta.get('tac_list_bytes', 0) / 1e6:.2f} MB + {meta.get('tac_get', 0)} GET(s) "
        f"{meta.get('tac_bytes', 0) / 1e6:.3f} MB; IGRA listing "
        f"{meta.get('igra_bytes', 0) / 1e3:.0f} KB",
        flush=True,
    )
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
