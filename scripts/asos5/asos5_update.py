#!/usr/bin/env python
"""Build data for the ASOS 5-minute temperature dashboard (asos5.html).

Sources (official NWS/NOAA only):
  * MADIS HFMETAR — the 5-minute ASOS observations, hourly netCDF files at
    https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF/YYYYMMDD_HH00.gz
  * api.weather.gov — hourly METAR fallback for stations that do not transmit
    the 5-minute feed to MADIS (KJFK, KNYC as of 2026-07).

Writes data/asos5/asos5_latest.js (window.ASOS5 = {...}) and .json with each
station's observations since its local midnight. Hourly MADIS files that are
safely in the past are cached under scripts/asos5/cache/ so a steady-state run
only downloads the two most recent hours (~8 MB).

Run with the wx conda env (needs netCDF4 + numpy):
  /opt/homebrew/Caskroom/miniconda/base/envs/wx/bin/python scripts/asos5/asos5_update.py
"""

import gzip
import io
import json
import os
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
from netCDF4 import Dataset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUT_DIR = os.path.join(REPO, "data", "asos5")

MADIS_URL = "https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF/{:%Y%m%d_%H}00.gz"
NWS_URL = "https://api.weather.gov/stations/{id}/observations?start={start}&limit=500"
AWC_URL = "https://aviationweather.gov/api/data/metar?ids={ids}&format=json&hours=30"
USER_AGENT = "scorvec.com asos5 dashboard (shawncorvec@hotmail.com)"

# MADIS data-descriptor flags to reject: failed / questioned / subjective bad.
BAD_DD = {"X", "Q", "B"}

# How far back the dashboard shows observations, so today can be compared against
# yesterday. The running daily max/min stay anchored to each station's local
# midnight (m0); this only widens the plotted trace.
WINDOW_H = 36

# (id, name, IANA tz, source) — 'hfmetar' = MADIS 5-min, 'nwsapi' = hourly fallback
STATIONS = [
    ("KBOS", "Boston Logan Intl",            "America/New_York",    "hfmetar"),
    ("KLGA", "New York LaGuardia",           "America/New_York",    "hfmetar"),
    ("KNYC", "New York Central Park",        "America/New_York",    "nwsapi"),
    ("KEWR", "Newark Liberty Intl",          "America/New_York",    "hfmetar"),
    ("KJFK", "New York JFK Intl",            "America/New_York",    "nwsapi"),
    ("KPHL", "Philadelphia Intl",            "America/New_York",    "hfmetar"),
    ("KBWI", "Baltimore/Washington Intl",    "America/New_York",    "hfmetar"),
    ("KDCA", "Washington Reagan National",   "America/New_York",    "hfmetar"),
    ("KRIC", "Richmond Intl",                "America/New_York",    "hfmetar"),
    ("KDFW", "Dallas-Fort Worth Intl",       "America/Chicago",     "hfmetar"),
    ("KIAH", "Houston Bush Intercontinental","America/Chicago",     "hfmetar"),
    ("KORD", "Chicago O'Hare Intl",          "America/Chicago",     "hfmetar"),
    ("KBUR", "Burbank Hollywood",            "America/Los_Angeles", "hfmetar"),
    ("KSEA", "Seattle-Tacoma Intl",          "America/Los_Angeles", "hfmetar"),
    ("KPDX", "Portland Intl",                "America/Los_Angeles", "hfmetar"),
]
HF_IDS = {s[0] for s in STATIONS if s[3] == "hfmetar"}


def http_get(url, timeout=60, tries=2):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise


def char_var(ds, name, n):
    """Return a (recNum,) array of stripped strings from a MADIS char variable."""
    raw = np.ma.filled(ds.variables[name][:], b" ")
    if raw.ndim == 1:
        return np.array([b.decode("ascii", "ignore").strip() for b in raw.astype("S1")])
    flat = raw.astype("S1").view(f"S{raw.shape[1]}").ravel()
    return np.array([b[:n].decode("ascii", "ignore").strip() for b in flat])


def parse_hfmetar(nc_bytes):
    """Extract our stations from one hourly HFMETAR file.

    Returns {station_id: {epoch_s: [tempC, dewC, wdir_deg, wspd_kt]}}.
    """
    out = {sid: {} for sid in HF_IDS}
    with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
        tf.write(nc_bytes)
        tf.flush()
        ds = Dataset(tf.name)
        try:
            ids = char_var(ds, "stationId", ds.dimensions["maxStaIdLen"].size)
            keep = np.isin(ids, list(HF_IDS))
            if not keep.any():
                return out
            idx = np.where(keep)[0]
            obst = np.ma.filled(ds.variables["observationTime"][:][idx], np.nan)
            temp = np.ma.filled(ds.variables["temperature"][:][idx], np.nan)
            dew = np.ma.filled(ds.variables["dewpoint"][:][idx], np.nan)
            wdir = np.ma.filled(ds.variables["windDir"][:][idx], np.nan)
            wspd = np.ma.filled(ds.variables["windSpeed"][:][idx], np.nan)  # m/s
            tdd = char_var(ds, "temperatureDD", 1)[idx]
            for k in range(len(idx)):
                if not np.isfinite(obst[k]) or not np.isfinite(temp[k]):
                    continue
                if tdd[k] in BAD_DD:
                    continue
                t_c = float(temp[k]) - 273.15
                if not -60.0 <= t_c <= 60.0:
                    continue
                d_c = float(dew[k]) - 273.15 if np.isfinite(dew[k]) else None
                rec = [
                    round(t_c, 2),
                    round(d_c, 2) if d_c is not None and -60 <= d_c <= 45 else None,
                    int(wdir[k]) if np.isfinite(wdir[k]) else None,
                    round(float(wspd[k]) * 1.94384, 1) if np.isfinite(wspd[k]) else None,
                ]
                out[ids[idx[k]]][int(obst[k])] = rec
        finally:
            ds.close()
    return out


def hour_data(hour_utc, now_utc):
    """Per-hour HFMETAR records, via cache when the hour can no longer change."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, hour_utc.strftime("%Y%m%d_%H") + ".json")
    settled = now_utc - hour_utc > timedelta(hours=2)
    if settled and os.path.exists(cache):
        with open(cache) as f:
            return {k: {int(e): v for e, v in d.items()} for k, d in json.load(f).items()}
    try:
        raw = http_get(MADIS_URL.format(hour_utc))
    except Exception as e:
        print(f"  MADIS {hour_utc:%Y%m%d_%H}00: unavailable ({e})", file=sys.stderr)
        return {}
    data = parse_hfmetar(gzip.decompress(raw))
    if settled:
        with open(cache, "w") as f:
            json.dump(data, f)
    return data


def fetch_nws_hourly(sid, start_utc):
    """Hourly obs from api.weather.gov for stations without a 5-min feed."""
    url = NWS_URL.format(id=sid, start=start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))
    obs = {}
    try:
        feats = json.loads(http_get(url))["features"]
    except Exception as e:
        print(f"  NWS API {sid}: failed ({e})", file=sys.stderr)
        return obs
    for f in feats:
        p = f["properties"]
        t = p.get("temperature") or {}
        if t.get("value") is None or t.get("qualityControl") in BAD_DD:
            continue
        epoch = int(datetime.fromisoformat(p["timestamp"]).timestamp())
        dew = (p.get("dewpoint") or {}).get("value")
        wdir = (p.get("windDirection") or {}).get("value")
        wspd = (p.get("windSpeed") or {}).get("value")  # km/h
        obs[epoch] = [
            round(float(t["value"]), 2),
            round(float(dew), 2) if dew is not None else None,
            int(wdir) if wdir is not None else None,
            round(float(wspd) * 0.539957, 1) if wspd is not None else None,
        ]
    return obs


def fetch_sixhr():
    """Decoded METAR 6-hour max/min temperature groups from the NWS Aviation
    Weather Center, for all stations at once.

    Returns {station_id: [[end_epoch, maxC|None, minC|None], ...]} where each
    entry is the extreme over the 6 hours ending at end_epoch (00/06/12/18Z).
    """
    url = AWC_URL.format(ids=",".join(s[0] for s in STATIONS))
    out = {}
    try:
        metars = json.loads(http_get(url))
    except Exception as e:
        print(f"  AWC METAR API: failed ({e})", file=sys.stderr)
        return out
    seen = set()
    for o in metars:
        sid, mx, mn = o.get("icaoId"), o.get("maxT"), o.get("minT")
        if not sid or (mx is None and mn is None):
            continue
        try:
            end = int(datetime.fromisoformat(
                o["reportTime"].replace("Z", "+00:00")).timestamp())
        except (KeyError, ValueError):
            continue
        ok = lambda v: v is not None and -60.0 <= float(v) <= 60.0
        mx = float(mx) if ok(mx) else None
        mn = float(mn) if ok(mn) else None
        if mx is None and mn is None:
            continue
        if mx is not None and mn is not None and mx < mn:
            continue  # garbled group
        if (sid, end) in seen:
            continue
        seen.add((sid, end))
        out.setdefault(sid, []).append([end, mx, mn])
    return out


def absolutes(recs, groups, m0):
    """Today's absolute max/min: sampled obs merged with 6-h max/min groups.

    recs: {epoch: [tempC, ...]} including samples back to the start of the 6-h
    window that straddles local midnight m0. A straddling window's extreme is
    credited to today only when it beats every sample in the window's
    yesterday portion (otherwise it plausibly occurred before midnight, and
    today's samples already cover the today portion).
    """
    hi = lo = None
    for e, v in sorted(recs.items()):
        if e < m0 or v[0] is None:
            continue
        if hi is None or v[0] > hi["v"] + 1e-9:
            hi = {"v": v[0], "e": e, "src": "obs", "win": None}
        if lo is None or v[0] < lo["v"] - 1e-9:
            lo = {"v": v[0], "e": e, "src": "obs", "win": None}
    for end, gmax, gmin in groups:
        start = end - 6 * 3600
        if end <= m0:
            continue
        ypart = [v[0] for e, v in recs.items() if start <= e < m0 and v[0] is not None]
        win = [max(start, m0), end]
        if gmax is not None and (start >= m0 or not ypart or gmax > max(ypart) + 0.05):
            if hi is None or gmax > hi["v"] + 0.05:
                hi = {"v": gmax, "e": None, "src": "6hr", "win": win}
        if gmin is not None and (start >= m0 or not ypart or gmin < min(ypart) - 0.05):
            if lo is None or gmin < lo["v"] - 0.05:
                lo = {"v": gmin, "e": None, "src": "6hr", "win": win}
    if hi is None or lo is None:
        return None
    return {
        "maxC": round(hi["v"], 2), "max_e": hi["e"], "max_src": hi["src"], "max_win": hi["win"],
        "minC": round(lo["v"], 2), "min_e": lo["e"], "min_src": lo["src"], "min_win": lo["win"],
    }


def prune_cache(now_utc):
    if not os.path.isdir(CACHE_DIR):
        return
    for fn in os.listdir(CACHE_DIR):
        if fn.endswith(".json"):
            try:
                ts = datetime.strptime(fn[:11], "%Y%m%d_%H").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if now_utc - ts > timedelta(days=3):
                os.remove(os.path.join(CACHE_DIR, fn))


def main():
    now = datetime.now(timezone.utc)
    w0 = int(now.timestamp()) - WINDOW_H * 3600      # display window start (epoch s)

    midnights = {}  # sid -> local midnight (as aware UTC datetime)
    for sid, _name, tz, _src in STATIONS:
        loc = now.astimezone(ZoneInfo(tz))
        midnights[sid] = loc.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)

    # reach back to whichever is earlier: WINDOW_H hours ago (the plotted trace) or
    # the start of the 6-h max/min window straddling the earliest local midnight (so
    # straddling groups can still be attributed to today's max/min)
    earliest = min(min(midnights.values()), now - timedelta(hours=WINDOW_H))
    first_hour = earliest.replace(minute=0, second=0, microsecond=0)
    first_hour = first_hour.replace(hour=first_hour.hour // 6 * 6)
    hours = []
    h = first_hour
    while h <= now:
        hours.append(h)
        h += timedelta(hours=1)

    print(f"Fetching MADIS HFMETAR {hours[0]:%Y-%m-%d %H}Z .. {hours[-1]:%H}Z ({len(hours)} files)")
    merged = {sid: {} for sid in HF_IDS}
    for h in hours:
        for sid, recs in hour_data(h, now).items():
            merged[sid].update(recs)

    sixhr = fetch_sixhr()

    stations_out = []
    for sid, name, tz, src in STATIONS:
        if src == "hfmetar":
            recs = merged.get(sid, {})
            label, cadence = "NWS/MADIS 5-min ASOS", 5
        else:
            win_start = datetime.fromtimestamp(w0, timezone.utc).replace(minute=0, second=0, microsecond=0)
            recs = fetch_nws_hourly(sid, win_start)
            label, cadence = "NWS API hourly METAR", 60
        m0 = int(midnights[sid].timestamp())
        groups = sixhr.get(sid, [])
        obs = sorted([e] + v for e, v in recs.items() if e >= w0)   # 36-h trace; max/min stay m0-anchored
        stations_out.append({
            "id": sid, "name": name, "tz": tz,
            "source": label, "cadence_min": cadence,
            "day_local": now.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d"),
            "abs": absolutes(recs, groups, m0),
            "sixhr": [g for g in groups if g[0] > m0],
            "obs": obs,  # [epoch_s, tempC, dewC, wdir_deg, wspd_kt]
        })
        print(f"  {sid}: {len(obs):3d} obs in last {WINDOW_H}h, "
              f"{len([g for g in groups if g[0] > m0]):d} 6-h groups today ({label})")

    payload = {
        "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stations": stations_out,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    blob = json.dumps(payload, separators=(",", ":"))
    with open(os.path.join(OUT_DIR, "asos5_latest.json"), "w") as f:
        f.write(blob)
    with open(os.path.join(OUT_DIR, "asos5_latest.js"), "w") as f:
        f.write("window.ASOS5 = " + blob + ";\n")
    prune_cache(now)
    print(f"Wrote {os.path.join(OUT_DIR, 'asos5_latest.js')}")


if __name__ == "__main__":
    main()
