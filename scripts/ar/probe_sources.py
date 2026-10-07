#!/usr/bin/env python3
"""One-off source probe for the AR redesign (runs in Actions only): which humidity / wind / surface-pressure fields and
pressure levels each model publishes, step availability, retention for a past AR case. Prints; downloads only index
files and directory listings."""
from __future__ import annotations

import datetime as dt
import json
import re
from collections import defaultdict

import requests

G = "https://storage.googleapis.com/ecmwf-open-data"
UA = {"User-Agent": "scorvec.com data pipeline (+https://scorvec.com)"}


def get(u, **k):
    try:
        r = requests.get(u, headers=UA, timeout=60, **k)
        return r
    except Exception as e:  # noqa: BLE001
        print("  ERR", u, e)
        return None


def ecmwf_index(date, hh, model, stream, kind, step):
    u = f"{G}/{date}/{hh}z/{model}/0p25/{stream}/{date}{hh}0000-{step}h-{stream}-{kind}.index"
    r = get(u)
    if r is None or r.status_code != 200:
        print(f"  {model}/{stream}/{kind} step {step}: HTTP {None if r is None else r.status_code}  {u}")
        return None
    rows = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    pl = defaultdict(set); sfc = set(); sizes = defaultdict(list)
    for e in rows:
        if e.get("levtype") == "pl":
            pl[e["param"]].add(int(e["levelist"]))
        else:
            sfc.add(e["param"])
        sizes[e["param"]].append(e["_length"])
    print(f"  {model}/{stream}/{kind} {date} {hh}Z step {step}: {len(rows)} msgs")
    for p in sorted(pl):
        print(f"     pl {p}: {sorted(pl[p], reverse=True)}  ~{sum(sizes[p]) / len(sizes[p]) / 1e6:.2f} MB/msg")
    print(f"     sfc: {sorted(sfc)}")
    return rows


def main():
    now = dt.datetime.utcnow()
    d = (now - dt.timedelta(hours=14)).strftime("%Y%m%d")
    print("== ECMWF open data, Google mirror, cycle", d, "00Z")
    ecmwf_index(d, "00", "ifs", "oper", "fc", 0)
    ecmwf_index(d, "00", "ifs", "oper", "fc", 168)
    ecmwf_index(d, "00", "aifs-single", "oper", "fc", 0)
    ecmwf_index(d, "00", "aifs-single", "oper", "fc", 168)
    ecmwf_index(d, "00", "aifs-ens", "enfo", "cf", 0)
    ecmwf_index(d, "00", "aifs-ens", "enfo", "cf", 360)
    ecmwf_index(d, "00", "ifs", "enfo", "ef", 6)
    ecmwf_index(d, "00", "ifs", "enfo", "ef", 240)
    print("== step sets")
    for model, stream, kind in (("ifs", "oper", "fc"), ("aifs-single", "oper", "fc"), ("aifs-ens", "enfo", "pf"), ("ifs", "enfo", "ef")):
        ok = []
        for s in (3, 6, 9, 150, 153, 156, 162, 168, 240, 246, 252, 360):
            r = requests.head(f"{G}/{d}/00z/{model}/0p25/{stream}/{d}000000-{s}h-{stream}-{kind}.index", timeout=30)
            ok.append(f"{s}:{r.status_code}")
        print(f"  {model}/{stream}: {' '.join(ok)}")
    print("== ECMWF retention: 2024-11-20 / 2025-02-13 / 2026-01-15")
    for dd in ("20241120", "20250213", "20260115"):
        for model, stream, kind in (("ifs", "oper", "fc"), ("aifs-single", "oper", "fc")):
            r = requests.head(f"{G}/{dd}/00z/{model}/0p25/{stream}/{dd}000000-0h-{stream}-{kind}.index", timeout=30)
            print(f"  {dd} {model}/{stream}: {r.status_code}")

    print("== GFS 0.25 (AWS noaa-gfs-bdp-pds)")
    for f in (0, 168):
        u = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/00/atmos/gfs.t00z.pgrb2.0p25.f{f:03d}.idx"
        r = get(u)
        if r is None or r.status_code != 200:
            print("  GFS", f, r and r.status_code); continue
        lev = defaultdict(list)
        for l in r.text.splitlines():
            p = l.split(":")
            if len(p) > 5 and p[3] in ("SPFH", "UGRD", "VGRD", "RH", "PRES", "PWAT", "HGT") and (" mb" in p[4] or p[4] in ("surface", "entire atmosphere (considered as a single layer)")):
                lev[p[3]].append(p[4])
        for k, v in lev.items():
            print(f"  f{f:03d} {k}: {v}")
    for dd in ("20241120", "20250213"):
        r = requests.head(f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{dd}/00/atmos/gfs.t00z.pgrb2.0p25.f000.idx", timeout=30)
        print(f"  GFS retention {dd}: {r.status_code}")

    print("== GEFS (AWS noaa-gefs-pds)")
    for sub, name in (("pgrb2ap5", "pgrb2a.0p50"), ("pgrb2bp5", "pgrb2b.0p50"), ("pgrb2sp25", "pgrb2s.0p25")):
        for f in (6, 246, 840):
            u = f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{d}/00/atmos/{sub}/gec00.t00z.{name}.f{f:03d}.idx"
            r = get(u)
            if r is None or r.status_code != 200:
                print(f"  {sub} f{f:03d}: {r and r.status_code}"); continue
            lev = defaultdict(list)
            for l in r.text.splitlines():
                p = l.split(":")
                if len(p) > 5 and p[3] in ("SPFH", "UGRD", "VGRD", "RH", "PRES", "PWAT", "TMP"):
                    lev[p[3]].append(p[4])
            print(f"  {sub} f{f:03d}: " + "; ".join(f"{k}={v}" for k, v in lev.items()))
    for f in (3, 9, 243, 246, 252, 384, 390):
        r = requests.head(f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{d}/00/atmos/pgrb2ap5/gep01.t00z.pgrb2a.0p50.f{f:03d}.idx", timeout=30)
        print(f"  GEFS pgrb2ap5 gep01 f{f:03d}: {r.status_code}")
    print("== GEFSv12 reforecast listing (noaa-gefs-retrospective)")
    for pre in ("GEFSv12/reforecast/2019/2019010200/c00/Days:1-10/", "GEFSv12/reforecast/2019/2019010200/c00/Days:10-35/",
                "GEFSv12/reforecast/2019/2019010200/p01/Days:10-35/"):
        r = get("https://noaa-gefs-retrospective.s3.amazonaws.com/", params={"list-type": "2", "prefix": pre})
        keys = re.findall(r"<Key>([^<]+)</Key>", r.text) if r is not None else []
        szs = re.findall(r"<Size>([^<]+)</Size>", r.text) if r is not None else []
        print(f"  {pre}: {len(keys)} objects")
        for k, s in zip(keys, szs):
            if not k.endswith(".idx"):
                print(f"     {k.rsplit('/', 1)[-1]}  {int(s) / 1e6:.0f} MB")
    r = get("https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/2019/2019010200/c00/Days:10-35/pwat_eatm_2019010200_c00.grib2.idx")
    if r is not None and r.status_code == 200:
        print("  pwat days 10-35 idx:", r.text.splitlines()[:3], "...", len(r.text.splitlines()), "msgs")
    r = get("https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/2019/2019010200/c00/Days:10-35/ugrd_pres_2019010200_c00.grib2.idx")
    if r is not None and r.status_code == 200:
        ls = r.text.splitlines(); print("  ugrd_pres days 10-35 idx:", ls[:6], "...", len(ls), "msgs")

    print("== GDPS 15 km (Datamart)")
    for lead in ("000", "168"):
        u = f"https://dd.weather.gc.ca/{d}/WXO-DD/model_gdps/15km/00/{lead}/"
        r = get(u)
        if r is None or r.status_code != 200:
            print("  GDPS listing", lead, r and r.status_code); continue
        names = sorted(set(re.findall(r'href="([^"]+\.grib2)"', r.text)))
        vars_ = defaultdict(list)
        for n in names:
            m = re.match(rf"{d}T00Z_MSC_GDPS_(.+?)_LatLon0\.15_PT\d+H\.grib2", n)
            if m:
                v = m.group(1)
                if "_IsbL-" in v:
                    a, b = v.split("_IsbL-"); vars_[a].append(b)
                else:
                    vars_["(single)"].append(v)
        for k, v in vars_.items():
            if k == "(single)":
                print(f"  {lead} single-level ({len(v)}): {v}")
            else:
                print(f"  {lead} {k}: {v}")
        sz = re.findall(r'SpecificHumidity_IsbL-0850[^"]*"[^\n]*?(\d+(?:\.\d+)?[KMG])', r.text)
        print("  size hint q850:", sz[:2])
    for sub in ("model_gdps/25km", "model_gdps", "model_gem_global"):
        r = get(f"https://dd.weather.gc.ca/{d}/WXO-DD/{sub}/")
        print(f"  listing {sub}: {r and r.status_code}", (re.findall(r'href="([^"?/][^"]*)"', r.text)[:20] if r is not None and r.status_code == 200 else ""))

    print("== GEPS raw 0.5 (Datamart)")
    for lead in ("006", "246", "384", "840"):
        u = f"https://dd.weather.gc.ca/{d}/WXO-DD/ensemble/geps/grib2/raw/00/{lead}/"
        r = get(u)
        if r is None or r.status_code != 200:
            print("  GEPS listing", lead, r and r.status_code); continue
        names = sorted(set(re.findall(r'href="([^"]+\.grib2)"', r.text)))
        vs = sorted(set(re.sub(rf"CMC_geps-raw_(.+?)_latlon0p5x0p5_{d}00_P{lead}_allmbrs\.grib2", r"\1", n) for n in names))
        print(f"  {lead}: {len(vs)} files: {vs}")
    for lead in ("003", "009", "195", "198", "390"):
        r = requests.head(f"https://dd.weather.gc.ca/{d}/WXO-DD/ensemble/geps/grib2/raw/00/{lead}/CMC_geps-raw_UGRD_ISBL_0850_latlon0p5x0p5_{d}00_P{lead}_allmbrs.grib2", headers=UA, timeout=30)
        print(f"  GEPS step {lead}: {r.status_code} {r.headers.get('Content-Length')}")

    print("== ARCO ERA5 (GCS)")
    try:
        import xarray as xr
        ds = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", chunks=None,
                          storage_options={"token": "anon"}, consolidated=True)
        vv = [v for v in ds.data_vars if "vapour_flux" in v or "water_vapour" in v]
        print("  vars:", vv)
        print("  time range:", str(ds.time.values[0]), str(ds.time.values[-1]))
        av = ds.attrs.get("valid_time_stop") or ds.attrs
        print("  attrs valid_time_stop:", ds.attrs.get("valid_time_stop"), ds.attrs.get("valid_time_stop_era5t"))
    except Exception as e:  # noqa: BLE001
        print("  ARCO error", e)


if __name__ == "__main__":
    main()
