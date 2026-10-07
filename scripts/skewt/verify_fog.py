#!/usr/bin/env python3
"""Verify the Skew-T explorer's fog / ceiling estimate against METARs (runs in GitHub Actions only).

    python scripts/skewt/verify_fog.py probe OUT
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

UA = "scorvec.com skew-t fog verification (+https://scorvec.com/skewt/)"
IGRA = "https://www.ncei.noaa.gov/pub/data/igra/"
ISD_HIST = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
IEM = "https://mesonet.agron.iastate.edu"

CANDIDATES = {   # WMO -> METAR ICAO (co-location verified from coordinates at run time)
    "72493": "KOAK", "72694": "KSLE", "72528": "KBUF", "72426": "KILN", "72235": "KJAN",
    "72240": "KLCH", "72248": "KSHV", "72251": "KCRP", "72250": "KBRO", "72363": "KAMA",
    "72451": "KDDC", "72456": "KTOP", "72645": "KGRB", "72747": "KINL", "72208": "KCHS",
    "72214": "KTLH", "72317": "KGSO", "72403": "KIAD", "72518": "KALB", "72712": "KCAR",
    "72476": "KGJT", "72681": "KBOI", "72597": "KMFR", "72797": "KUIL", "70273": "PANC",
    "70261": "PAFA", "70026": "PABR", "72764": "KBIS", "72659": "KABR",
    "87576": "SAEZ", "08221": "LEMD", "16080": "LIML",
}


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


def probe(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rep = []
    sl = get(IGRA + "igra2-station-list.txt").decode()
    rep.append("## IGRA station list rows")
    for line in sl.splitlines():
        if line[6:11] in CANDIDATES:
            rep.append(line)
    isd = get(ISD_HIST).decode()
    rep.append("## ISD history rows")
    ic = set(CANDIDATES.values())
    for line in isd.splitlines():
        parts = line.split('","')
        if len(parts) > 5 and parts[5] in ic:
            rep.append(line)
    t0 = time.time()
    try:
        net = json.loads(get(IEM + "/geojson/network.py?network=RAOB"))
        rep.append("## IEM RAOB network")
        for f in net["features"]:
            p = f["properties"]
            s = json.dumps(p)
            if any(w in s for w in CANDIDATES):
                rep.append(s[:300])
    except Exception as e:  # noqa: BLE001
        rep.append(f"IEM RAOB network failed: {e}")
    for url in [IEM + "/cgi-bin/request/raob.py?station=KOAK&sts=2023-01-10T00:00Z&ets=2023-01-11T00:00Z",
                IEM + "/cgi-bin/request/raob.py?station=KOAK&sts=2016-01-10T00:00Z&ets=2016-01-11T00:00Z",
                IEM + "/json/raob.py?ts=202301101200&station=KOAK",
                IEM + "/json/raob.py?ts=202609301200&station=KOAK"]:
        time.sleep(15)
        try:
            t1 = time.time()
            b = get(url, tries=1).decode(errors="replace")
            rep.append(f"## {url} ({len(b)} B, {time.time()-t1:.1f}s)")
            rep.append(b[:1500])
            if url.endswith("KOAK") and "json" in url:
                j = json.loads(b)
                for pr in j.get("profiles", []):
                    lv = pr["profile"]
                    low = [(L.get("pres"), L.get("hght"), L.get("tmpc"), L.get("dwpc")) for L in lv[:25]]
                    rep.append(f"levels {len(lv)} first25 {low}")
            else:
                from collections import Counter
                c = Counter(r.split(",")[1] for r in b.splitlines()[1:] if r)
                rep.append(f"levels per sounding {dict(c)}")
        except Exception as e:  # noqa: BLE001
            rep.append(f"{url} failed: {e}")
    # IGRA y2d for OAK: level counts of the last few soundings + the 2023-01-10 POR record
    for gid in ["USM00072493", "SPM00008221"]:
        for y in (2026, 2025):
            try:
                z = zipfile.ZipFile(io.BytesIO(get(IGRA + f"data/data-y2d/{gid}-data-beg{y}.txt.zip", tries=1)))
                txt = z.read(z.namelist()[0]).decode()
                hdrs = [i for i, L in enumerate(txt.splitlines()) if L.startswith("#")]
                lines = txt.splitlines()
                rep.append(f"## IGRA y2d {gid} beg{y}: {len(hdrs)} soundings")
                for i in hdrs[-4:]:
                    rep.append(lines[i])
                    rep.extend(lines[i + 1:i + 8])
                break
            except Exception as e:  # noqa: BLE001
                rep.append(f"y2d {gid} {y}: {e}")
    time.sleep(15)
    t1 = time.time()
    url = (IEM + "/cgi-bin/request/asos.py?station=KOAK&data=vsby&data=skyc1&data=skyc2&data=skyc3&data=skyc4"
           "&data=skyl1&data=skyl2&data=skyl3&data=skyl4&data=wxcodes&year1=2024&month1=1&day1=1&year2=2025"
           "&month2=1&day2=1&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3")
    try:
        b = get(url, tries=1).decode()
        rep.append(f"## asos KOAK 2024 report_type=3: {len(b)} B {b.count(chr(10))} rows {time.time()-t1:.1f}s")
        rep.append("\n".join(b.splitlines()[:30]))
    except Exception as e:  # noqa: BLE001
        rep.append(f"asos failed: {e}")
    time.sleep(15)
    url = url.replace("KOAK", "LEMD")
    try:
        b = get(url, tries=1).decode()
        rep.append(f"## asos LEMD 2024: {len(b)} B {b.count(chr(10))} rows")
        rep.append("\n".join(b.splitlines()[:30]))
    except Exception as e:  # noqa: BLE001
        rep.append(f"asos LEMD failed: {e}")
    rep.append(f"elapsed {time.time()-t0:.0f}s")
    (out / "probe.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "probe":
        probe(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown command {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
