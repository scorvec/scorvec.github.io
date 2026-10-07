#!/usr/bin/env python3
"""CanSIPS Niño-3.4 / relative Niño-3.4 plume, 12 months, straight from ECCC's MSC Datamart.

PROBE STAGE: `python cansips_dd.py probe` downloads one forecast and one hindcast WaterTemp file
and prints how members, time and the land/ice mask are encoded. The builder follows once the
encoding is known.

Data Source: Environment and Climate Change Canada (MSC Datamart, ECCC Data Servers End-use Licence).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cansips_dd"
DD = "https://dd.weather.gc.ca/{day}/WXO-DD/model_cansips/100km"
UA = {"User-Agent": "scorvec.com ENSO forecasts (CanSIPS Nino-3.4)"}


def dd_root() -> str:
    # Datamart is date-prefixed (a ~30-day rolling window); every prefix serves the same static
    # CanSIPS tree, so today's UTC date is used.
    return DD.format(day=time.strftime("%Y%m%d", time.gmtime()))


def fc_url(ym: str, lead: int) -> str:
    return f"{dd_root()}/forecast/{ym[:4]}/{ym[4:]}/{ym}_MSC_CanSIPS_WaterTemp_Sfc_LatLon1.0_P{lead:02d}M.grib2"


def hc_url(ym: str, lead: int) -> str:
    return f"{dd_root()}/hindcast/{ym[:4]}/{ym[4:]}/{ym}_MSC_CanSIPS-Hindcast_WaterTemp_Sfc_LatLon1.0_P{lead:02d}M.grib2"


def fetch(url: str, dest: Path, tries: int = 4) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return dest
        except Exception as e:                                         # noqa: BLE001
            print(f"  GET {url.rsplit('/', 1)[-1]} attempt {k+1}: {str(e)[:90]}", file=sys.stderr)
            time.sleep(5 * (k + 1))
    return None


KEYS = ["centre", "subCentre", "generatingProcessIdentifier", "typeOfGeneratingProcess",
        "productDefinitionTemplateNumber", "perturbationNumber", "numberOfForecastsInEnsemble",
        "typeOfEnsembleForecast", "dataDate", "dataTime", "stepRange", "startStep", "endStep",
        "stepUnits", "typeOfStatisticalProcessing", "lengthOfTimeRange", "indicatorOfUnitForTimeRange",
        "validityDate", "shortName", "name", "units", "discipline", "parameterCategory", "parameterNumber",
        "typeOfFirstFixedSurface", "bitmapPresent", "numberOfMissing", "missingValue", "Ni", "Nj",
        "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees", "longitudeOfLastGridPointInDegrees", "jScansPositively",
        "productionStatusOfProcessedData", "typeOfProcessedData", "backgroundProcess",
        "NV", "localUsePresent", "section2Length", "numberOfTimeRange", "hoursAfterDataCutoff"]


def _get(h, k):
    import eccodes
    try:
        return eccodes.codes_get(h, k)
    except Exception:                                                  # noqa: BLE001
        return "-"


def n34_of(vals, lats, lons):
    """cos-weighted Niño-3.4 (5S-5N, 190-240E inclusive) over finite cells."""
    la = (lats >= -5) & (lats <= 5)
    lo = (lons >= 190) & (lons <= 240)
    sub = vals[np.ix_(la, lo)]
    w = np.cos(np.deg2rad(lats[la]))[:, None] * np.ones(lo.sum())[None, :]
    ok = np.isfinite(sub)
    return float((sub[ok] * w[ok]).sum() / w[ok].sum()), int((~ok).sum())


def read_msgs(path: Path):
    import eccodes
    out = []
    with open(path, "rb") as fh:
        while True:
            h = eccodes.codes_grib_new_from_file(fh)
            if h is None:
                break
            meta = {k: _get(h, k) for k in KEYS}
            Ni, Nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
            v = eccodes.codes_get_values(h).astype(float)
            if eccodes.codes_get(h, "bitmapPresent"):
                v[v == eccodes.codes_get(h, "missingValue")] = np.nan
            la0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
            la1 = eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
            lo0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
            lats = np.linspace(la0, la1, Nj)
            lons = np.mod(lo0 + np.arange(Ni) * eccodes.codes_get(h, "iDirectionIncrementInDegrees"), 360)
            out.append((meta, v.reshape(Nj, Ni), lats, lons))
            eccodes.codes_release(h)
    return out


def probe() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    print("Datamart root:", dd_root())
    for tag, url in [("forecast 202610 P00M", fc_url("202610", 0)),
                     ("forecast 202609 P00M", fc_url("202609", 0)),
                     ("hindcast 199310 P00M", hc_url("199310", 0)),
                     ("hindcast 199310 P11M", hc_url("199310", 11))]:
        p = fetch(url, CACHE / "probe" / url.rsplit("/", 1)[-1])
        print(f"\n===== {tag}: {url}\n  -> {p} {p.stat().st_size if p else 'MISSING'}")
        if not p:
            continue
        msgs = read_msgs(p)
        print(f"  messages: {len(msgs)}")
        m0 = msgs[0][0]
        for k in KEYS:
            vals = sorted({str(m[0][k]) for m in msgs})
            print(f"  {k:36s} {m0[k]!s:24s} distinct={len(vals)} {vals[:6]}{' ...' if len(vals) > 6 else ''}")
        meta, v, lats, lons = msgs[0]
        print(f"  lats {lats[0]}..{lats[-1]} n={lats.size}  lons {lons[0]}..{lons[-1]} n={lons.size}")
        print(f"  finite cells {np.isfinite(v).sum()} of {v.size}; min {np.nanmin(v):.2f} max {np.nanmax(v):.2f}")
        for name, la, lo in [("Amazon 5S 300E", -5, 300), ("Sahara 20N 10E", 20, 10), ("Antarctic 80S 0E", -80, 0),
                             ("Weddell ice 70S 320E", -70, 320), ("Arctic 85N 0E", 85, 0), ("Hudson 60N 275E", 60, 275),
                             ("Nino34 0N 210E", 0, 210), ("Gulf Guinea 0N 0E", 0, 0)]:
            i = int(np.argmin(np.abs(lats - la))); j = int(np.argmin(np.abs(lons - lo)))
            print(f"    {name:22s} {v[i, j]}")
        n34 = [n34_of(m[1], m[2], m[3]) for m in msgs]
        print("  Nino-3.4 (abs, degC if K-273.15) per message:")
        for idx, (m, (val, nmiss)) in enumerate(zip(msgs, n34)):
            print(f"    msg{idx:02d} pert={m[0]['perturbationNumber']} gen={m[0]['generatingProcessIdentifier']} "
                  f"date={m[0]['dataDate']} step={m[0]['stepRange']} n34={val:.3f} miss={nmiss}")
    # identify which member numbers belong to which component: absolute Niño-3.4 hindcast climatology
    # for September starts, lead 1, by member number, against C3S's eccc_4 (26.77 C) and eccc_5 (26.24 C)
    print("\n===== September-start hindcast lead-1 Niño-3.4 by member, 1993-2016")
    acc = {}
    for y in range(1993, 2017):
        p = fetch(hc_url(f"{y}09", 0), CACHE / "probe" / f"hc_{y}09_P00M.grib2")
        if not p:
            print(f"  {y}09 missing"); continue
        for m in read_msgs(p):
            val, _ = n34_of(m[1], m[2], m[3])
            acc.setdefault(m[0]["perturbationNumber"], []).append(val)
    for k in sorted(acc):
        a = np.array(acc[k])
        kel = a.mean() > 200
        print(f"  member {k:>3}: n={a.size} mean={a.mean() - (273.15 if kel else 0):.3f} C")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe"])
    a = ap.parse_args()
    if a.cmd == "probe":
        return probe()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
