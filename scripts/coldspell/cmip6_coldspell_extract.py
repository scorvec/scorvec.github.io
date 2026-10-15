#!/usr/bin/env python3
"""CMIP6 daily reductions for the cold-spell lead-time study (2026-10-15; user: "Did you also look deeply into the
thousands of extra winters in cmip6?"). Runs in Actions (.github/workflows/cmip6-coldspell.yml), one job per model member,
reading the public Pangeo CMIP6 Google Cloud mirror anonymously; only small reductions leave the runner.

Historical run, every year in the store (normally 1850-2014), on the model's own calendar. Per variable:
  tas  (day)       US population-weighted means on a 1° grid (pop_na1.npz: GeoNames places of 15,000+, the ERA5 study's
                   rule): us, east, west, mwne, scent; area colder than -20/-25/-30 °C over 40-90N in the North American
                   (190-310E) and Eurasian (20-180E) sectors (million km2); Niño-3.4 box mean (5S-5N, 190-240E)
  psl  (day)       box means on a 2.5° 20-90N grid: polar cap 65-90N, Azores 33-45N 330-357E, Iceland 57-70N 320-345E,
                   Greenland 60-80N 290-340E, Siberian High 40-60N 80-120E, Aleutian Low 45-60N 165-200E; the 5° field on
                   Oct-Mar days (hPa - 1000, float16) for composites
  zg   (day plev8) polar-cap (65-90N) height at 1000/500/100/50/10 hPa; z500 boxes (Alaska ridge 55-70N 200-240E, EPO
                   55-65N 200-235E, Greenland block 60-80N 290-340E, Ural ridge 55-70N 40-80E, NE Pacific 40-55N 210-235E,
                   Hudson trough 50-65N 260-290E); PNA (Wallace-Gutzler points, as the ERA5 predictors); geostrophic 60N
                   wind from the 55-65N zonal-mean gradient at 100/50/10 hPa; wave-1/2 amplitude at 60N, 100 and 10 hPa;
                   1000-850 hPa thickness temperature -> cold-air areas and cold content by sector (the ERA5 definitions);
                   the z500 5° field on Oct-Mar days (m - 5500, float16)
  pr, rlut (day)   15S-15N cos-weighted meridional means on a 2.5° longitude ring (for a real-time MJO index)
  ua   (EdayZ)     zonal-mean u at 60N and 60S, 10/50/100 hPa (both hemispheres kept: IPSL-CM6A-LR's EdayZ latitude
                   is reversed, the analysis picks the hemisphere whose vortex peaks in DJF)
Each variable is written as soon as it is done (out/<model>_<member>_<var>.npz), so a timeout keeps what finished.

    python cmip6_coldspell_extract.py --catalog pangeo-cmip6.csv --model CanESM5 --member r1i1p1f1 --vars tas,psl,zg,pr,rlut,ua
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

FS = gcsfs.GCSFileSystem(token="anon")
HERE = Path(__file__).resolve().parent
G, R, OMEGA, A = 9.80665, 287.05, 7.292e-5, 6.371e6
NH = (np.arange(20.0, 90.01, 2.5), np.arange(0.0, 360.0, 2.5))
POP = np.load(HERE / "pop_na1.npz")
NA1 = (POP["lat"], POP["lon"])
CAP_BYTES = 1.2e9


def zstore(cat, model, member, var, table):
    s = cat[(cat.source_id == model) & (cat.member_id == member) & (cat.experiment_id == "historical")
            & (cat.table_id == table) & (cat.variable_id == var)]
    return None if s.empty else s.sort_values("version").zstore.iloc[-1]


def interp(block, lat, lon, glat, glon):
    """(..., lat, lon) -> (..., glat, glon), bilinear, cyclic in longitude, latitude clamped at the edges.
    lat ascending, lon ascending in [0, 360)."""
    lonc = np.concatenate([lon[-1:] - 360.0, lon, lon[:1] + 360.0])
    blk = np.concatenate([block[..., -1:], block, block[..., :1]], axis=-1)
    ix = np.interp(glon % 360.0, lonc, np.arange(len(lonc)))
    iy = np.interp(glat, lat, np.arange(len(lat)))
    x0 = np.floor(ix).astype(int); x1 = np.minimum(x0 + 1, len(lonc) - 1); fx = (ix - x0)
    y0 = np.floor(iy).astype(int); y1 = np.minimum(y0 + 1, len(lat) - 1); fy = (iy - y0)[:, None]
    r0 = blk[..., y0, :]; r1 = blk[..., y1, :]
    a = r0[..., x0] * (1 - fx) + r0[..., x1] * fx
    b = r1[..., x0] * (1 - fx) + r1[..., x1] * fx
    return (a * (1 - fy) + b * fy).astype(np.float32)


def boxw(lat, lon, la0, la1, lo0, lo1):
    m = ((lat >= la0) & (lat <= la1))[:, None] & ((lon >= lo0) & (lon <= lo1))[None, :]
    w = np.cos(np.deg2rad(lat))[:, None] * m
    return w / w.sum()


def stream(zs, levels=None):
    """Open one zarr store and yield (dates, block float32 (t, [lev,] lat, lon), lat, lon) in chunk-aligned blocks."""
    ds = xr.open_zarr(FS.get_mapper(zs), consolidated=True, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                      decode_timedelta=False)
    name = [v for v in ds.data_vars if v not in ("time_bnds", "lat_bnds", "lon_bnds", "bnds", "time_bounds")][0]
    da = ds[name]
    for d in list(da.dims):
        if d in ("lon", "longitude", "x") and da.sizes[d] == 1:
            da = da.isel({d: 0}, drop=True)
    latn = "lat" if "lat" in da.dims else "latitude"
    lonn = ("lon" if "lon" in da.dims else "longitude") if ("lon" in da.dims or "longitude" in da.dims) else None
    kpos = None
    if levels is not None:
        pdim = [d for d in da.dims if d not in ("time", latn, lonn)][0]
        p = da[pdim].values.astype(float)
        if p.max() < 2000:
            p = p * 100.0
        kpos = []
        for lev in levels:
            k = int(np.argmin(np.abs(p - lev)))
            if abs(p[k] - lev) > 1.0:
                raise SystemExit(f"level {lev} Pa missing (have {p})")
            kpos.append(k)
        da = da.transpose("time", pdim, latn, *([lonn] if lonn else []))
    lat = da[latn].values.astype(float)
    so_lat = np.argsort(lat); lat = lat[so_lat]
    if lonn:
        lon = da[lonn].values.astype(float) % 360.0; so_lon = np.argsort(lon); lon = lon[so_lon]
    else:
        lon = None
    times = da.time.values
    nt = len(times)
    csz = da.chunks[0] if da.chunks else (nt,)
    per_day = np.prod([s for d_, s in zip(da.dims, da.shape) if d_ != "time"]) * 4
    cap = max(1, int(CAP_BYTES // per_day))
    bounds, cur, start = [], 0, 0
    for c in csz:
        if cur and cur + c > cap:
            bounds.append((start, start + cur)); start += cur; cur = 0
        cur += c
    if cur:
        bounds.append((start, start + cur))
    cal = str(getattr(times[0], "calendar", "standard"))
    for i0, i1 in bounds:
        raw = da.isel(time=slice(i0, i1)).values.astype(np.float32)
        if kpos is not None:
            raw = raw[:, kpos]
        raw = np.take(raw, so_lat, axis=-2 if lonn else -1)
        if lonn:
            raw = raw[..., so_lon]
        raw[~np.isfinite(raw) | (np.abs(raw) > 1e15)] = np.nan
        tt = times[i0:i1]
        dates = np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in tt])
        yield dates, raw, lat, lon, cal


def oct_mar(dates):
    m = np.array([int(d[5:7]) for d in dates])
    return np.isin(m, (10, 11, 12, 1, 2, 3))


def finish(out, tag, series, fields, dates_all, cal, t0, nbytes, extra=None):
    d = np.concatenate(dates_all)
    o = np.argsort(d, kind="stable"); d = d[o]
    keep = np.r_[True, d[1:] != d[:-1]]                                      # duplicate days (rare) -> first
    S = {k: np.concatenate(v)[o][keep].astype(np.float32) for k, v in series.items()}
    F = {}
    for k, (fd, fv) in fields.items():
        fd = np.concatenate(fd); fv = np.concatenate(fv); oo = np.argsort(fd, kind="stable"); fd, fv = fd[oo], fv[oo]
        kk = np.r_[True, fd[1:] != fd[:-1]]
        F[f"{k}_dates"] = fd[kk]; F[k] = fv[kk]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{tag}.npz", dates=d[keep], calendar=cal, **S, **F, **(extra or {}))
    log = dict(file=f"{tag}.npz", days=int(keep.sum()), first=str(d[keep][0]), last=str(d[keep][-1]), calendar=cal,
               gb_decoded=round(nbytes / 1e9, 2), seconds=round(time.time() - t0),
               means={k: round(float(np.nanmean(v)), 4) for k, v in S.items()})
    print(json.dumps(log), flush=True)
    return log


def do_tas(zs, out, tag):
    t0 = time.time(); series, dates_all, nbytes = {}, [], 0
    add = lambda k, v: series.setdefault(k, []).append(np.asarray(v, np.float32))
    W = {k: POP[k] / POP[k].sum() for k in ("us", "east", "west", "mwne", "scent")}
    glat, glon = NH
    cell = (A / 1e3) ** 2 * np.deg2rad(2.5) ** 2 * np.cos(np.deg2rad(glat))
    band = glat >= 40
    sect = {"na": (glon >= 190) & (glon <= 310), "eu": (glon >= 20) & (glon <= 180)}
    cal = "standard"
    for dates, raw, lat, lon, cal in stream(zs):
        nbytes += raw.nbytes
        t = raw - 273.15
        na = interp(t, lat, lon, *NA1)
        for k, w in W.items():
            add(f"t2m_{k}", np.nansum(na * w, axis=(1, 2)))
        nh = interp(t, lat, lon, glat, glon)
        for s, cm in sect.items():
            for thr in (-20, -25, -30):
                m = (nh < thr) & band[None, :, None] & cm[None, None, :]
                add(f"tcold{abs(thr)}_{s}", (m * cell[None, :, None]).sum((1, 2)) / 1e6)
        eq = (lat >= -5) & (lat <= 5); bx = (lon >= 190) & (lon <= 240)
        w34 = np.cos(np.deg2rad(lat[eq]))[:, None] * np.ones((1, bx.sum()))
        add("nino34_tas", np.nansum(t[:, eq][:, :, bx] * w34, axis=(1, 2)) / w34.sum())
        dates_all.append(dates)
    return finish(out, tag, series, {}, dates_all, cal, t0, nbytes)


def do_psl(zs, out, tag):
    t0 = time.time(); series, dates_all, nbytes = {}, [], 0
    fd, fv = [], []
    add = lambda k, v: series.setdefault(k, []).append(np.asarray(v, np.float32))
    glat, glon = NH
    B = {"slp_pcap": (65, 90, 0, 360), "slp_azores": (33, 45, 330, 357), "slp_iceland": (57, 70, 320, 345),
         "slp_greenland": (60, 80, 290, 340), "slp_siberia": (40, 60, 80, 120), "slp_aleutian": (45, 60, 165, 200)}
    BW = {k: boxw(glat, glon, *b) for k, b in B.items()}
    cal = "standard"
    for dates, raw, lat, lon, cal in stream(zs):
        nbytes += raw.nbytes
        nh = interp(raw / 100.0, lat, lon, glat, glon)
        for k, w in BW.items():
            add(k, np.einsum("tij,ij->t", nh, w))
        m = oct_mar(dates)
        if m.any():
            fd.append(dates[m]); fv.append((nh[m][:, ::2, ::2] - 1000.0).astype(np.float16))
        dates_all.append(dates)
    return finish(out, tag, series, {"psl5": (fd, fv)}, dates_all, cal, t0, nbytes,
                  extra=dict(lat5=glat[::2], lon5=glon[::2]))


def do_zg(zs, out, tag):
    t0 = time.time(); series, dates_all, nbytes = {}, [], 0
    fd, fv = [], []
    add = lambda k, v: series.setdefault(k, []).append(np.asarray(v, np.float32))
    levs = [100000.0, 85000.0, 50000.0, 10000.0, 5000.0, 1000.0]
    glat, glon = NH
    cap = boxw(glat, glon, 65, 90, 0, 360)
    B = {"alaska_ridge": (55, 70, 200, 240), "epo": (55, 65, 200, 235), "greenland_block": (60, 80, 290, 340),
         "ural_ridge": (55, 70, 40, 80), "nepac_ridge": (40, 55, 210, 235), "hudson_trough": (50, 65, 260, 290)}
    BW = {k: boxw(glat, glon, *b) for k, b in B.items()}
    j55, j60, j65 = (int(np.argmin(np.abs(glat - x))) for x in (55, 60, 65))
    f60 = 2 * OMEGA * np.sin(np.deg2rad(60)); dy = (glat[j65] - glat[j55]) * np.pi / 180 * A
    cell = (A / 1e3) ** 2 * np.deg2rad(2.5) ** 2 * np.cos(np.deg2rad(glat))
    band = glat >= 40
    sect = {"na": (glon >= 190) & (glon <= 310), "eu": (glon >= 20) & (glon <= 180)}
    pna_pts = [(20.0, 200.0, 1), (45.0, 195.0, -1), (55.0, 245.0, 1), (30.0, 275.0, -1)]
    cal = "standard"
    for dates, raw, lat, lon, cal in stream(zs, levels=levs):
        nbytes += raw.nbytes
        z = {L: interp(raw[:, i], lat, lon, glat, glon) for i, L in enumerate((1000, 850, 500, 100, 50, 10))}
        for L in (1000, 500, 100, 50, 10):
            add(f"pch{L}", np.einsum("tij,ij->t", z[L], cap))
        for k, w in BW.items():
            add(k, np.einsum("tij,ij->t", z[500], w))
        add("pna", 0.25 * sum(s * z[500][:, int(np.argmin(np.abs(glat - la))), int(np.argmin(np.abs(glon - lo)))]
                              for la, lo, s in pna_pts))
        for L in (100, 50, 10):
            zm = np.nanmean(z[L], axis=2)
            add(f"u{L}_60n", -G / f60 * (zm[:, j65] - zm[:, j55]) / dy)
        for L in (100, 10):
            F = np.fft.rfft(np.nan_to_num(z[L][:, j60, :]), axis=1) / z[L].shape[2]
            for k in (1, 2):
                add(f"w{k}amp{L}", 2 * np.abs(F[:, k]))
        tl = G * (z[850] - z[1000]) / (R * np.log(1000 / 850)) - 273.15
        for s, cm in sect.items():
            for thr in (-20, -25, -30):
                m = (tl < thr) & band[None, :, None] & cm[None, None, :]
                add(f"cold{abs(thr)}_{s}", (m * cell[None, :, None]).sum((1, 2)) / 1e6)
            m = (tl < -20) & band[None, :, None] & cm[None, None, :]
            add(f"coldcontent_{s}", (np.where(m, -20 - tl, 0) * cell[None, :, None]).sum((1, 2)) / 1e6)
        m = oct_mar(dates)
        if m.any():
            fd.append(dates[m]); fv.append((z[500][m][:, ::2, ::2] - 5500.0).astype(np.float16))
        dates_all.append(dates)
    return finish(out, tag, series, {"z5005": (fd, fv)}, dates_all, cal, t0, nbytes,
                  extra=dict(lat5=glat[::2], lon5=glon[::2]))


def do_ring(zs, out, tag, var):
    t0 = time.time(); series, dates_all, nbytes = {}, [], 0
    ring = np.arange(0.0, 360.0, 2.5)
    cal = "standard"
    vals = []
    for dates, raw, lat, lon, cal in stream(zs):
        nbytes += raw.nbytes
        rows = (lat >= -15) & (lat <= 15)
        w = np.cos(np.deg2rad(lat[rows]))
        mm = np.nansum(raw[:, rows] * w[None, :, None], axis=1) / w.sum()          # (t, lon)
        mm = mm * (86400.0 if var == "pr" else 1.0)
        lonc = np.concatenate([lon[-1:] - 360.0, lon, lon[:1] + 360.0])
        mmc = np.concatenate([mm[:, -1:], mm, mm[:, :1]], axis=1)
        vals.append(np.stack([np.interp(ring, lonc, r) for r in mmc]).astype(np.float32))
        dates_all.append(dates)
    d = np.concatenate(dates_all); v = np.concatenate(vals); o = np.argsort(d, kind="stable"); d, v = d[o], v[o]
    keep = np.r_[True, d[1:] != d[:-1]]
    off = 0.0 if var == "pr" else 200.0
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{tag}.npz", dates=d[keep], calendar=cal, ring=ring, **{f"{var}_ring": (v[keep] - off).astype(np.float16)},
                        offset=off)
    log = dict(file=f"{tag}.npz", days=int(keep.sum()), first=str(d[keep][0]), last=str(d[keep][-1]), calendar=cal,
               gb_decoded=round(nbytes / 1e9, 2), seconds=round(time.time() - t0), mean=round(float(np.nanmean(v)), 3))
    print(json.dumps(log), flush=True)
    return log


def do_ua(zs, out, tag):
    t0 = time.time()
    ds = xr.open_zarr(FS.get_mapper(zs), consolidated=True, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                      decode_timedelta=False)
    u = ds["ua"]
    for d in list(u.dims):
        if d in ("lon", "longitude", "x") and u.sizes[d] == 1:
            u = u.isel({d: 0}, drop=True)
    u = u.rename({d: "lat" for d in u.dims if d in ("latitude", "y")}).sortby("lat")
    pdim = [d for d in u.dims if d not in ("time", "lat")][0]
    u = u.rename({pdim: "plev"}).transpose("time", "plev", "lat")
    p = u.plev.values.astype(float)
    if p.max() < 2000:
        p = p * 100.0
    kk = {L: int(np.argmin(np.abs(p - L * 100.0))) for L in (10, 50, 100)}
    bad = {L: p[k] for L, k in kk.items() if abs(p[k] - L * 100.0) > 1.0}
    if bad:
        raise SystemExit(f"EdayZ levels missing: {bad}")
    lat = u.lat.values.astype(float)
    S = {f"ua{L}_{h}": [] for L in kk for h in ("60n", "60s")}
    dates = []
    nt = u.sizes["time"]; nbytes = 0
    for i0 in range(0, nt, 3000):
        X = u.isel(time=slice(i0, min(nt, i0 + 3000)), plev=list(kk.values())).values.astype(np.float32)
        X[~np.isfinite(X) | (np.abs(X) > 1e3)] = np.nan
        nbytes += X.nbytes
        for j, L in enumerate(kk):
            for h, la in (("60n", 60.0), ("60s", -60.0)):
                S[f"ua{L}_{h}"].append(np.array([np.interp(la, lat, r) for r in X[:, j]], np.float32))
        dates.append(np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in u.time.values[i0:i0 + 3000]]))
    cal = str(getattr(u.time.values[0], "calendar", "standard"))
    return finish(out, tag, S, {}, dates, cal, t0, nbytes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True); ap.add_argument("--model", required=True); ap.add_argument("--member", required=True)
    ap.add_argument("--vars", default="tas,psl,zg,pr,rlut,ua"); ap.add_argument("--out", default="out")
    a = ap.parse_args()
    import dask
    dask.config.set(scheduler="threads", num_workers=8)
    cat = pd.read_csv(a.catalog)
    out = Path(a.out); tag = f"{a.model}_{a.member}"
    logs = {}
    plan = {"tas": ("tas", "day", do_tas), "psl": ("psl", "day", do_psl), "zg": ("zg", "day", do_zg),
            "pr": ("pr", "day", lambda z, o, t: do_ring(z, o, t, "pr")), "rlut": ("rlut", "day", lambda z, o, t: do_ring(z, o, t, "rlut")),
            "ua": ("ua", "EdayZ", do_ua)}
    for v in [x.strip() for x in a.vars.split(",") if x.strip()]:
        var, table, fn = plan[v]
        zs = zstore(cat, a.model, a.member, var, table)
        if zs is None:
            logs[v] = "no store"; print(f"{tag} {v}: no store", flush=True); continue
        for attempt in range(3):
            try:
                logs[v] = dict(zstore=zs, **fn(zs, out, f"{tag}_{v}")); break
            except SystemExit as e:
                logs[v] = f"skipped: {e}"; print(f"{tag} {v}: {e}", flush=True); break
            except Exception as e:                                                   # noqa: BLE001
                logs[v] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"; print(f"{tag} {v}: {logs[v]}", flush=True)
                time.sleep(30 * (attempt + 1))
        (out / f"{tag}_log.json").write_text(json.dumps(logs, indent=1))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_log.json").write_text(json.dumps(logs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
