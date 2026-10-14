#!/usr/bin/env python3
"""Independent rain "truths" for the model-rain bias work (2026-10-13, user: re-fetch MERGE and the IDEAM gauges "and other
possible datasets" and "come up with the ultimate rain truth blend"). Everything runs in Actions (rain-truth-backfill.yml);
gridded products land on the 1-degree Americas grid of grid.py (NaN outside their coverage), station data as QC'd tables.

  ideam   IDEAM (Colombia) automatic gauges, datos.gov.co dataset s54a-sgyg (raw 10-minute data), aggregated server-side
          to station-sensor-hour, shifted from Colombian time (UTC-5) to UTC days; a day needs >= 22 complete hours.
          Mostly NOT on the GTS, so mostly independent of the gauge-based grids below - the validation set.
  ghcn    GHCN-Daily PRCP (NOAA open data on AWS), Latin America and the Caribbean (US and Canada left out), Q-flag clean.
          Station observing days (local mornings), so best used at >= 5-day scales.
  merge   MERGE (INPE/CPTEC): IMERG Late merged with South American gauges, daily 0.1 deg, 12Z-12Z; NEST = gauges per cell.
  cpc     CPC unified gauge analysis (NOAA PSL), daily 0.5 deg, land.
  gpcc    GPCC First Guess Daily (DWD), gauge-only 1 deg, land.
  chirps  CHIRPS v2.0 daily 0.25 deg (CHC), land 50S-50N: infrared + gauges; final months only (recent months run dry).

Day definitions differ (UTC, 12Z-12Z, 06Z-06Z, station mornings). Offsets of up to a day add noise to daily comparisons but
do not bias a long-run mean, which is all the bias field uses; the blend documents them.

    python scripts/precip_bias/truth_sources.py {ideam|merge|gpcc} --month 2025-03 --out DIR
    python scripts/precip_bias/truth_sources.py {cpc|chirps|ghcn} --year 2025 --out DIR
-> DIR/{src}_{YYYYMM}.npz (dates, pr (day, lat, lon) mm/day [, gauges]) or DIR/{src}_{YYYYMM|YYYY}.csv.gz + {src}_stations.csv
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid as G                                          # noqa: E402

UA = "scorvec-research/1.0 (+https://scorvec.com; rain verification research)"
_S = None


def session():
    global _S
    if _S is None:
        import requests
        import urllib3.util.connection as uc
        uc.HAS_IPV6 = False
        _S = requests.Session(); _S.headers["User-Agent"] = UA
    return _S


def get(url, tries=5, timeout=180, **kw):
    for k in range(tries):
        try:
            r = session().get(url, timeout=timeout, **kw)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception as ex:                                              # noqa: BLE001
            if k == tries - 1:
                print(f"  GET failed {url[:110]}: {repr(ex)[:90]}", flush=True)
                return None
            time.sleep(5 * (k + 1))


def month_days(ym):
    y, m = map(int, ym.split("-"))
    d0 = date(y, m, 1); d1 = (d0 + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]


def box_mean(a, lat_c, lon_c, step, min_frac=0.5):
    """Regular grid (lat ascending, lon ascending, cell centres lat_c / lon_c at spacing `step`) -> 1-degree Americas boxes:
    mean of the cells whose centres fall in each box, NaN with fewer than min_frac valid. a: (..., nlat, nlon)."""
    out = np.full(a.shape[:-2] + (G.NY, G.NX), np.nan, dtype="float32")
    k = int(round(1.0 / step))
    for i, la in enumerate(G.LAT):
        ri = np.where((lat_c > la - 0.5) & (lat_c < la + 0.5))[0]
        if len(ri) != k:
            continue
        for j, lo in enumerate(G.LON):
            cj = np.where((lon_c > lo - 0.5) & (lon_c < lo + 0.5))[0]
            if len(cj) != k:
                continue
            blk = a[..., ri[0]:ri[-1] + 1, cj[0]:cj[-1] + 1]
            ok = np.isfinite(blk)
            n = ok.sum(axis=(-2, -1))
            with np.errstate(invalid="ignore"):
                m = np.nansum(blk, axis=(-2, -1)) / np.where(n > 0, n, np.nan)
            out[..., i, j] = np.where(n >= min_frac * k * k, m, np.nan)
    return out


def save_grid(out: Path, src, ym, dates, pr, **extra):
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{src}_{ym.replace('-', '')}.npz"
    w = G.area_weights(); ok = np.isfinite(pr)
    mean = float(np.nansum(np.nanmean(pr, 0) * w) / np.sum(w * np.isfinite(np.nanmean(pr, 0)))) if ok.any() else float("nan")
    np.savez_compressed(f, dates=np.array([d.strftime("%Y%m%d") for d in dates]), pr=pr.astype("float32"), lat=G.LAT, lon=G.LON, **extra)
    print(f"{src} {ym}: {len(dates)} days, valid boxes {100 * np.isfinite(pr).mean():.0f} %, mean {mean:.2f} mm/day -> {f.name}", flush=True)
    return f


# ---------------------------------------------------------------- MERGE
MERGE_URL = "https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY/{y:04d}/{m:02d}/MERGE_CPTEC_{y:04d}{m:02d}{d:02d}.grib2"
MLAT = -60.05 + 0.1 * np.arange(924); MLON = -120.05 + 0.1 * np.arange(1001)


def merge_month(ym, out):
    import tempfile
    import eccodes
    dates, pr, ng = [], [], []
    for d in month_days(ym):
        r = get(MERGE_URL.format(y=d.year, m=d.month, d=d.day))
        if r is None:
            continue
        with tempfile.NamedTemporaryFile(suffix=".grib2") as tf:
            tf.write(r.content); tf.flush()
            fields = []
            with open(tf.name, "rb") as fh:
                while True:
                    h = eccodes.codes_grib_new_from_file(fh)
                    if h is None:
                        break
                    v = eccodes.codes_get_values(h).reshape(924, 1001)            # south to north (first point -60.05)
                    if eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees") > eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees"):
                        v = v[::-1]
                    fields.append(np.where((v < 0) | (v >= 9000), np.nan, v)); eccodes.codes_release(h)
        if len(fields) < 2:
            continue
        p, n = fields[0], fields[1]                                               # PREC (mm), NEST (gauges per cell)
        pr.append(box_mean(p, MLAT, MLON, 0.1)); ng.append(box_mean(np.nan_to_num(n), MLAT, MLON, 0.1, 0.0) * 100)  # gauges per box
        dates.append(d)
    if not dates:
        print(f"merge {ym}: nothing"); return None
    return save_grid(out, "merge", ym, dates, np.stack(pr), gauges=np.stack(ng).astype("float32"), day="12Z previous day to 12Z")


# ---------------------------------------------------------------- CPC unified (year file -> months)
def cpc_year(y, out):
    import netCDF4
    r = get(f"https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{y}.nc", timeout=600)
    if r is None:
        return None
    with netCDF4.Dataset("cpc.nc", mode="r", memory=r.content) as f:
        lat = f["lat"][:]; lon = f["lon"][:]; t = netCDF4.num2date(f["time"][:], f["time"].units)
        v = f["precip"][:].astype("float32").filled(np.nan) if hasattr(f["precip"][:], "filled") else np.asarray(f["precip"][:], "float32")
    if lat[0] > lat[-1]:
        lat = lat[::-1]; v = v[:, ::-1]
    lon = np.where(lon > 180, lon - 360, lon); o = np.argsort(lon); lon = lon[o]; v = v[:, :, o]
    days = [date(x.year, x.month, x.day) for x in t]
    made = []
    for m in sorted({(d.year, d.month) for d in days}):
        k = [i for i, d in enumerate(days) if (d.year, d.month) == m]
        made.append(save_grid(out, "cpc", f"{m[0]}-{m[1]:02d}", [days[i] for i in k], box_mean(v[k], lat, lon, 0.5)))
    return made


# ---------------------------------------------------------------- GPCC First Guess Daily (monthly files)
def gpcc_month(ym, out):
    import netCDF4
    y, m = map(int, ym.split("-"))
    r = get(f"https://opendata.dwd.de/climate_environment/GPCC/first_guess_daily/{y}/first_guess_daily_{y}{m:02d}.nc.gz")
    if r is None:
        return None
    with netCDF4.Dataset("gpcc.nc", mode="r", memory=gzip.decompress(r.content)) as f:
        names = list(f.variables)
        pv = next(n for n in names if f[n].ndim == 3 and ("prec" in n.lower() or n.lower() in ("p", "precip")))
        gv = next((n for n in names if f[n].ndim == 3 and n != pv and ("gauge" in n.lower() or n.lower() in ("s", "numgauge"))), None)
        lat = np.asarray(f["lat"][:]); lon = np.asarray(f["lon"][:])
        p = np.ma.filled(f[pv][:].astype("float32"), np.nan); g = np.ma.filled(f[gv][:].astype("float32"), np.nan) if gv else None
        t = netCDF4.num2date(f["time"][:], f["time"].units)
    p = np.where(p < 0, np.nan, p)
    if lat[0] > lat[-1]:
        lat = lat[::-1]; p = p[:, ::-1]; g = g[:, ::-1] if g is not None else None
    days = [date(x.year, x.month, x.day) for x in t]
    print(f"  gpcc variables: precip={pv} gauges={gv}", flush=True)
    extra = {"gauges": box_mean(np.nan_to_num(g), lat, lon, 1.0, 0.0)} if g is not None else {}
    return save_grid(out, "gpcc", ym, days, box_mean(p, lat, lon, 1.0), day="06Z-06Z (SYNOP)", **extra)


# ---------------------------------------------------------------- CHIRPS v2.0 daily p25 (year file -> months)
def chirps_year(y, out, last_final=None):
    import netCDF4
    r = get(f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/chirps-v2.0.{y}.days_p25.nc", timeout=900)
    if r is None:
        return None
    with netCDF4.Dataset("chirps.nc", mode="r", memory=r.content) as f:
        lat = np.asarray(f["latitude"][:]); lon = np.asarray(f["longitude"][:])
        t = netCDF4.num2date(f["time"][:], f["time"].units)
        days = [date(x.year, x.month, x.day) for x in t]
        # only the Americas columns / rows, read in one slice
        li = np.where((lat > -60) & (lat < 60))[0]; lj = np.where((lon > -170) & (lon < -30))[0]
        v = np.ma.filled(f["precip"][:, li[0]:li[-1] + 1, lj[0]:lj[-1] + 1].astype("float32"), np.nan)
    lat, lon = lat[li[0]:li[-1] + 1], lon[lj[0]:lj[-1] + 1]
    if lat[0] > lat[-1]:
        lat = lat[::-1]; v = v[:, ::-1]
    v = np.where(v < 0, np.nan, v)
    made = []
    for m in sorted({(d.year, d.month) for d in days}):
        k = [i for i, d in enumerate(days) if (d.year, d.month) == m]
        made.append(save_grid(out, "chirps", f"{m[0]}-{m[1]:02d}", [days[i] for i in k], box_mean(v[k], lat, lon, 0.25)))
    return made


# ---------------------------------------------------------------- GHCN-Daily (Latin America + Caribbean)
def ghcn_year(y, out):
    import pandas as pd
    st = get("https://noaa-ghcn-pds.s3.amazonaws.com/ghcnd-stations.txt")
    rows = []
    for line in st.text.splitlines():
        sid = line[0:11]; la = float(line[12:20]); lo = float(line[21:30]); el = float(line[31:37])
        if sid[:2] in ("US", "CA") or not (-60 <= la <= 33 and -120 <= lo <= -30):
            continue
        rows.append((sid, la, lo, el, line[41:71].strip()))
    meta = pd.DataFrame(rows, columns=["id", "lat", "lon", "elev", "name"]).set_index("id")
    r = get(f"https://noaa-ghcn-pds.s3.amazonaws.com/csv.gz/by_year/{y}.csv.gz", timeout=900)
    if r is None:
        return None
    keep = []
    for ch in pd.read_csv(io.BytesIO(r.content), compression="gzip", header=None, chunksize=2_000_000, dtype=str,
                          names=["id", "date", "el", "v", "m", "q", "s", "t"], usecols=range(8)):
        ch = ch[(ch.el == "PRCP") & ch.id.isin(meta.index) & ch.q.isna()]
        if len(ch):
            keep.append(ch[["id", "date", "v"]])
    d = pd.concat(keep); d["mm"] = d.v.astype(float) / 10.0; d = d[(d.mm >= 0) & (d.mm < 600)]
    out.mkdir(parents=True, exist_ok=True)
    d[["id", "date", "mm"]].to_csv(out / f"ghcn_{y}.csv.gz", index=False)
    meta.loc[meta.index.isin(d.id.unique())].to_csv(out / "ghcn_stations.csv")
    print(f"ghcn {y}: {len(d):,} station-days, {d.id.nunique()} stations", flush=True)
    return out / f"ghcn_{y}.csv.gz"


# ---------------------------------------------------------------- IDEAM (Colombia), datos.gov.co Socrata
IDEAM = "https://www.datos.gov.co/resource/s54a-sgyg.json"


def ideam_month(ym, out, page=50000):
    import pandas as pd
    days = month_days(ym)
    # local-time window that covers the UTC days of the month: local = UTC - 5 h
    t0 = datetime.combine(days[0], datetime.min.time()) - timedelta(hours=5)
    t1 = datetime.combine(days[-1], datetime.min.time()) + timedelta(hours=19, minutes=59, seconds=59)
    where = f"fechaobservacion between '{t0:%Y-%m-%dT%H:%M:%S}' and '{t1:%Y-%m-%dT%H:%M:%S}'"
    rows, off = [], 0
    while True:
        q = {"$select": "codigoestacion,codigosensor,date_trunc_ymd(fechaobservacion) as d,date_extract_hh(fechaobservacion) as hh,"
                        "sum(valorobservado) as p,count(*) as n,max(valorobservado) as vmax",
             "$where": where, "$group": "codigoestacion,codigosensor,d,hh", "$order": "codigoestacion,codigosensor,d,hh",
             "$limit": page, "$offset": off}
        r = get(IDEAM, params=q, timeout=300)
        if r is None:
            print(f"  ideam {ym}: page at offset {off} failed - month incomplete, not saved"); return None
        chunk = r.json()
        if isinstance(chunk, dict):
            print(f"  ideam {ym}: {str(chunk)[:200]}"); return None
        rows += chunk; off += page
        print(f"  ideam {ym}: {len(rows):,} station-hours", flush=True)
        if len(chunk) < page:
            break
        time.sleep(1.0)
    if not rows:
        return None
    h = pd.DataFrame(rows)
    for c in ("p", "n", "vmax"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h["hh"] = h.hh.astype(int)
    h["utc"] = pd.to_datetime(h.d.str[:10]) + pd.to_timedelta(h.hh + 5, unit="h")                 # local + 5 h = UTC
    h["day"] = h.utc.dt.floor("D")
    # QC per station-sensor-hour: no negatives, no hour above 150 mm, no 10-minute value above 60 mm
    bad = (h.p < 0) | (h.p > 150) | (h.vmax > 60)
    h = h[~bad]
    # expected observations per hour = the sensor's modal count; an hour is complete with >= 80 % of it
    mode = h.groupby(["codigoestacion", "codigosensor"]).n.agg(lambda s: s.value_counts().idxmax())
    h["full"] = h.n >= 0.8 * h.set_index(["codigoestacion", "codigosensor"]).index.map(mode).values
    dd = h[h.full].groupby(["codigoestacion", "codigosensor", "day"]).agg(mm=("p", "sum"), hours=("p", "size")).reset_index()
    dd = dd[dd.hours >= 22]
    # one sensor per station-day: the one with more complete days in the month
    best = dd.groupby(["codigoestacion", "codigosensor"]).size().reset_index(name="k").sort_values("k").drop_duplicates("codigoestacion", keep="last")
    dd = dd.merge(best[["codigoestacion", "codigosensor"]], on=["codigoestacion", "codigosensor"])
    dd = dd[(dd.day >= pd.Timestamp(days[0])) & (dd.day <= pd.Timestamp(days[-1]))]
    dd["date"] = dd.day.dt.strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"ideam_{ym.replace('-', '')}.csv.gz"
    dd[["codigoestacion", "date", "mm", "hours"]].rename(columns={"codigoestacion": "id"}).to_csv(f, index=False)
    print(f"ideam {ym}: {len(dd):,} complete station-days, {dd.codigoestacion.nunique()} stations, "
          f"removed {int(bad.sum())} bad hours -> {f.name}", flush=True)
    return f


def ideam_stations(out):
    """Station coordinates (the raw dataset carries lat/lon on every record)."""
    import pandas as pd
    q = {"$select": "codigoestacion,max(latitud) as lat,max(longitud) as lon,max(nombreestacion) as name,max(departamento) as dep",
         "$where": "fechaobservacion > '2024-01-01T00:00:00'", "$group": "codigoestacion", "$limit": 50000}
    r = get(IDEAM, params=q, timeout=600)
    d = pd.DataFrame(r.json())
    d["lat"] = pd.to_numeric(d.lat); d["lon"] = pd.to_numeric(d.lon)
    d.rename(columns={"codigoestacion": "id"}).to_csv(out / "ideam_stations.csv", index=False)
    print(f"ideam stations: {len(d)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", choices=["ideam", "ideam-stations", "merge", "gpcc", "cpc", "chirps", "ghcn"])
    ap.add_argument("--month"); ap.add_argument("--year", type=int); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out)
    fn = {"ideam": lambda: ideam_month(a.month, out), "ideam-stations": lambda: ideam_stations(out) or True,
          "merge": lambda: merge_month(a.month, out), "gpcc": lambda: gpcc_month(a.month, out),
          "cpc": lambda: cpc_year(a.year, out), "chirps": lambda: chirps_year(a.year, out), "ghcn": lambda: ghcn_year(a.year, out)}[a.src]
    return 0 if fn() else 1


if __name__ == "__main__":
    sys.exit(main())
