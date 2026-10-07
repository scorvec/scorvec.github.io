#!/usr/bin/env python3
"""SEAS5 lead month 7 hindcast: the 1993–2016 reference for the 7th month, built once per start month.

WHY. The CDS monthly product stops at leadtime_month 6, so the SEAS5 page's 7th month (April for an October start,
user 2026-10-07) needs the hindcast from daily data. The CDS sub-daily hindcast would cost tens of GB of 6-hourly
global fields per start month, so the INSTANTANEOUS variables come from Planette's anonymous copy of the C3S SEAS5
daily archive on AWS — s3://planette-c3s-seasonal-forecasts/seas5_ic/hindcast (us-east-2, Copernicus licence,
icechunk 2.2.2 → Python >= 3.12; a third-party copy of C3S data, not ECMWF open data, so the Google-only ECMWF rule
does not apply — user, 2026-09-26). Daily means on a 1° CELL-CENTRED grid (89.5…−89.5, −179.5…179.5); the CDS
grids are point grids on whole degrees, so every field is interpolated bilinearly to the CDS points (a whole-degree
point is the mean of the four cells around it; a pole row takes the nearest row). The ACCUMULATED variables (tp,
snowfall, ssrd, evaporation) and the two levels the AWS copy does not hold (z 50 hPa, u 30/50 hPa) come from the
CDS sub-daily hindcast instead — boundary steps only for the accumulations, so those requests are small and exact.

What is built per monthly kind of seas5_outlook.KINDS: a "pseudo-raw" hindcast `hindcast/hc_{kind}_{MM}_L7.npz`
holding every sample's monthly mean for leads 5, 6 and 7 (samples member-major, year fastest — the order of the
monthly hindcast), on the kind's own CDS grid. seas5_ref derives the lead-7 tables (hc_{unit}_L7_{MM}.npz) from it
with the unchanged product code; leads 5 and 6 are only there for the seasons that end at lead 7 (e.g. FMA) and
for the consistency test against the CDS monthly hindcast (`check`).

    python seas5_aws.py probe                         # layout, variables, coordinates, throughput
    python seas5_aws.py build --month 10 --kinds gl   # pseudo-raw for the kinds (AWS + CDS)
    python seas5_aws.py check --month 10              # AWS vs the CDS monthly hindcast tables at leads 5 and 6

Runs in GitHub Actions only (seas5-lead7.yml / seas5.yml); the laptop never downloads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BUCKET, REGION = "planette-c3s-seasonal-forecasts", "us-east-2"
YEARS = [str(y) for y in range(1993, 2017)]
LEADS = (5, 6, 7)

# (kind, monthly variable key) → source. AWS: (group, variable, level, factor to the monthly product's unit).
# CDS: (seas5_lead7.GROUPS index, levels or None, area override or None, leads).
AWS_SRC = {
    ("sst", "sst"): ("single", "sst", None, 1.0),
    ("gl", "t2m"): ("single", "t2m", None, 1.0),
    ("gl", "sst"): ("single", "sst", None, 1.0),
    ("gl_z500", "z"): ("pressure", "z", 500, 1.0),
    ("gl_u850", "u"): ("pressure", "u", 850, 1.0),
    ("gl_v850", "v"): ("pressure", "v", 850, 1.0),
    ("na_snowdepth", "sd"): ("single", "sdwe", None, 1e-3),          # kg m-2 → m of water equivalent
    ("polar_n", "z_10"): ("pressure", "z", 10, 1.0),
    ("polar_s", "z_10"): ("pressure", "z", 10, 1.0),
    ("nh_z", "z_500"): ("pressure", "z", 500, 1.0),
    ("nh_msl", "msl"): ("single", "slp", None, 1.0),
    ("strat_u", "u_10"): ("pressure", "u", 10, 1.0),
}
# The AWS copy holds no data at z 100 and z 1000 hPa (every chunk empty, seen 2026-10-07), nor z 50 / u 30 / u 50;
# those come from the CDS sub-daily hindcast, lead 7 only, one request per kind (levels together).
CDS_SRC = {
    ("gl", "tprate"): (1, None, None, LEADS),
    ("na_snow", "mtsfr"): (0, None, None, LEADS),
    ("water", "e"): (0, None, None, LEADS),
    ("polar_n", "z_50"): (0, ["50", "100"], None, (7,)),
    ("polar_n", "z_100"): (0, ["50", "100"], None, (7,)),
    ("polar_s", "z_50"): (0, ["50", "100"], None, (7,)),
    ("polar_s", "z_100"): (0, ["50", "100"], None, (7,)),
    ("nh_z", "z_1000"): (0, ["1000"], None, (7,)),
    ("strat_u", "u_30"): (0, ["30", "50"], [4, -180, -4, 180], (7,)),    # QBO band only: the table reads 5°S–5°N
    ("strat_u", "u_50"): (0, ["30", "50"], [4, -180, -4, 180], (7,)),
}
# NOT extended (stay at 6 leads): the `energy` kind — the monthly 10 m wind speed is not reproducible from the
# sub-daily u/v (seas5_lead7 docstring)
SUBSET = {"sfc": ("gl", ("t2m", "tprate"))}
# variables that keep 6 leads: neither the AWS copy nor an affordable CDS request has them (none at present)
UNAVAILABLE: dict = {}


def keys_of(kind: str) -> list[str]:
    if kind in SUBSET:
        return list(SUBSET[kind][1])
    return [k for (kd, k) in list(AWS_SRC) + list(CDS_SRC) if kd == kind]


def kinds() -> list[str]:
    return sorted({kd for (kd, _) in list(AWS_SRC) + list(CDS_SRC)} | set(SUBSET))


def _guard(what: str) -> None:
    from seas5_ref import guard_fetch
    guard_fetch(what)


def open_root(period: str = "hindcast"):
    _guard(f"AWS SEAS5 {period}")
    import icechunk as ic
    import zarr
    st = ic.s3_storage(bucket=BUCKET, prefix=f"seas5_ic/{period}", region=REGION, anonymous=True)
    sess = ic.Repository.open(st).readonly_session("main")
    return zarr.open_group(sess.store, mode="r"), sess


# ── paths ────────────────────────────────────────────────────────────────────
def pseudo_path(kind: str, month: str) -> Path:
    from seas5_outlook import DATA
    return DATA / "hindcast" / f"hc_{kind}_{int(month):02d}_L7.npz"


def cell_cache(group: str, var: str, level, month: str) -> Path:
    from seas5_outlook import DATA
    return DATA / "aws" / f"{group}_{var}_{level or 'sfc'}_{int(month):02d}.npy"


# ── AWS extraction ───────────────────────────────────────────────────────────
AWS_STATS = {"chunks": 0, "seconds": 0.0, "mb_raw": 0.0}


def aws_monthly(group: str, var: str, level, month: str, leads=LEADS) -> np.ndarray:
    """[sample = member × 24 + year, len(leads), 180, 360] monthly means of AWS daily values on the cell grid.
    Lead day d (1-based) = UTC day d of the run, so month L is the days [H0/24, H1/24) of its exact calendar."""
    from seas5_lead7 import month_window
    cache = cell_cache(group, var, level, month)
    if cache.exists():
        return np.load(cache)
    import pandas as pd
    root, _ = open_root("hindcast")
    g = root[f"{group}/1platx1plon"]
    a = g[var]
    it = pd.Timestamp(g["init_time"].attrs["units"].split("since ")[1]) + pd.to_timedelta(g["init_time"][:], "D")
    lead = g["lead"][:]
    assert lead[0] == 1 and np.all(np.diff(lead) == 1), "AWS lead axis is not 1..215 days"
    il = None
    if level is not None:
        levs = [float(x) for x in g["pressure_level"][:]]
        il = levs.index(float(level))
    nm = a.shape[0]
    out = np.full((nm * len(YEARS), len(leads), a.shape[-2], a.shape[-1]), np.nan, dtype=np.float32)
    t0 = time.time()

    def one(job):
        yi, y, m0, m1 = job
        k = int(np.where((it.year == int(y)) & (it.month == int(month)) & (it.day == 1))[0][0])
        win = [month_window(f"{y}{int(month):02d}", L) for L in leads]
        d0, d1 = min(w[0] for w in win) // 24, max(w[1] for w in win) // 24
        tt = time.time()
        blk = a[m0:m1, k, d0:d1, il] if il is not None else a[m0:m1, k, d0:d1]
        blk = np.asarray(blk, dtype=np.float32)
        res = [np.nanmean(blk[:, H0 // 24 - d0:H1 // 24 - d0], axis=1) for H0, H1 in win]
        return yi, m0, m1, np.stack(res, axis=1), time.time() - tt, blk.nbytes

    jobs = [(yi, y, m0, min(m0 + 10, nm)) for yi, y in enumerate(YEARS) for m0 in range(0, nm, 10)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for yi, m0, m1, res, dt, nb in ex.map(one, jobs):
            for j, mem in enumerate(range(m0, m1)):
                out[mem * len(YEARS) + yi] = res[j]
            AWS_STATS["chunks"] += 1; AWS_STATS["mb_raw"] += nb / 1e6
    AWS_STATS["seconds"] += time.time() - t0
    print(f"    AWS {group}/{var}{'@' + str(level) if level else ''} start {month}: {out.shape} in {time.time() - t0:.0f} s "
          f"(finite {np.isfinite(out).mean():.3f})", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def cell_to_points(a: np.ndarray, tlat: np.ndarray, tlon: np.ndarray) -> np.ndarray:
    """Bilinear from the AWS cell centres (lat 89.5…−89.5, lon −179.5…179.5) to CDS points; [..., 180, 360] →
    [..., len(tlat), len(tlon)]. A whole-degree point is the mean of its four surrounding cells; the pole rows
    take the nearest cell row; longitudes wrap. NaN (land in SST) propagates only where all neighbours are NaN."""
    clat = 89.5 - np.arange(a.shape[-2]); clon = -179.5 + np.arange(a.shape[-1])
    r = (89.5 - np.asarray(tlat, float))                                  # fractional row
    r = np.clip(r, 0, a.shape[-2] - 1)
    r0 = np.floor(r).astype(int); r1 = np.minimum(r0 + 1, a.shape[-2] - 1); wr = r - r0
    c = (((np.asarray(tlon, float) + 180.0) % 360.0) - 0.5 + 360.0) % 360.0   # cell index space (cell j centred at j + 0.5 − 180)
    c0 = np.floor(c).astype(int) % a.shape[-1]; c1 = (c0 + 1) % a.shape[-1]; wc = c - np.floor(c)
    out = 0
    wsum = 0
    for ri, wy in ((r0, 1 - wr), (r1, wr)):
        for ci, wx in ((c0, 1 - wc), (c1, wc)):
            w = (wy[:, None] * wx[None, :]).astype(np.float32)
            v = a[..., ri, :][..., ci]
            good = np.isfinite(v)
            out = out + np.where(good, v, 0) * w
            wsum = wsum + good * w
    with np.errstate(invalid="ignore", divide="ignore"):
        return (out / np.where(wsum > 0, wsum, np.nan)).astype(np.float32)


# ── pseudo-raw builder ───────────────────────────────────────────────────────
def target_grid(kind: str):
    """The kind's CDS point grid: from a monthly GRIB of the kind on disk (forecast) or the kind's area/grid."""
    from seas5_outlook import DATA, KINDS
    import glob
    for p in sorted(glob.glob(str(DATA / "forecast" / f"fc_{kind}_[0-9][0-9][0-9][0-9][0-9][0-9].grib")), reverse=True):
        import xarray as xr
        ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": "", "time_dims": ("forecastMonth", "time")})
        lat, lon = ds.latitude.values, ds.longitude.values; ds.close()
        return lat, lon
    n, w, s, e = KINDS[kind]["area"]; g = KINDS[kind]["grid"][0]
    lat = np.arange(n, s - g / 2, -g)
    lon = np.arange(w, e - g / 2 if (e - w) >= 360 - 1e-9 else e + g / 2, g)
    return lat, lon


_CDS_FIELDS: dict = {}


def _cds_hindcast(kind: str, key: str, month: str, lat, lon) -> np.ndarray | None:
    """[600, 3 (leads 5–7), lat, lon] from the CDS sub-daily hindcast; leads not requested stay NaN. Keys that share
    a request (the same kind, group, levels, area and leads) are fetched once."""
    import seas5_lead7 as L7
    gi, levels, area, leads = CDS_SRC[(kind, key)]
    rk = (kind, gi, tuple(levels or ()), tuple(area or ()), tuple(leads), f"{int(month):02d}")
    if rk not in _CDS_FIELDS:
        ds, req = L7.request(kind, gi, YEARS, f"{int(month):02d}", leads)
        if levels:
            req["pressure_level"] = levels
        if area:
            req["area"] = area
        tag = f"hc{int(month):02d}_{'_'.join(levels or ['sfc'])}_{len(leads)}"
        dest = L7.raw_path(kind, gi, tag)
        if not L7._cds(ds, req, dest, f"hindcast {kind} {'/'.join(levels or [])} start {month} leads {list(leads)}"):
            return None
        _CDS_FIELDS.clear()                                             # one request's fields in memory at a time
        _CDS_FIELDS[rk] = L7.reduce_group(dest, YEARS, f"{int(month):02d}", leads)
        dest.unlink()
    fields = _CDS_FIELDS[rk]
    mvar, _, lev = key.partition("_")
    v, flat, flon = fields[(mvar, int(lev) if lev else None)]
    out = np.full((v.shape[0], len(LEADS), len(lat), len(lon)), np.nan, dtype=np.float32)
    ii = [int(np.argmin(np.abs(lat - x))) for x in flat]
    lonn = ((np.asarray(lon) + 180) % 360) - 180
    jj = [int(np.argmin(np.abs(lonn - (((x + 180) % 360) - 180)))) for x in flon]
    assert np.allclose(lat[ii], flat) and np.allclose(lonn[jj], ((flon + 180) % 360) - 180), f"{kind} {key}: grid mismatch"
    for j, L in enumerate(leads):
        out[:, LEADS.index(L)][:, np.ix_(ii, jj)[0], np.ix_(ii, jj)[1]] = v[:, j]
    return out


def build_kind(kind: str, month: str, aws_only: bool = False) -> Path | None:
    """hindcast/hc_{kind}_{MM}_L7.npz: every key of the kind, [600, 3, lat, lon] for leads 5–7. `aws_only` (the
    consistency test) skips the CDS pressure-level keys, which are exact by construction and queue for an hour."""
    dest = pseudo_path(kind, month)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    t0 = time.time()
    arrays = {}
    src_note = {}
    if kind in SUBSET:
        src, ks = SUBSET[kind]
        sp = build_kind(src, month)
        if sp is None:
            return None
        lat, lon = target_grid(kind)
        with np.load(sp) as z:
            slat, slon = z["lat"], z["lon"]
            ii = [int(np.where(np.isclose(slat, x))[0][0]) for x in lat]
            slonn = ((slon + 180) % 360) - 180
            jj = [int(np.where(np.isclose(slonn, ((x + 180) % 360) - 180))[0][0]) for x in lon]
            for k in ks:
                arrays[k] = z[k][:, :, ii][:, :, :, jj]
            src_note = json.loads(str(z["__meta__"]))["source"]
    else:
        lat, lon = target_grid(kind)
        for k in keys_of(kind):
            if (kind, k) in AWS_SRC:
                group, var, level, fac = AWS_SRC[(kind, k)]
                cell = aws_monthly(group, var, level, month)
                arrays[k] = cell_to_points(cell, lat, lon) * np.float32(fac)
                src_note[k] = f"AWS Planette SEAS5 daily {group}/{var}{'@' + str(level) if level else ''}, bilinear to the CDS grid"
            else:
                if aws_only and CDS_SRC[(kind, k)][1]:
                    continue
                v = _cds_hindcast(kind, k, month, lat, lon)
                if v is None:
                    print(f"  {kind} {k}: CDS hindcast fetch failed — pseudo-raw not written", flush=True)
                    return None
                arrays[k] = v
                src_note[k] = "CDS seasonal-original hindcast (sub-daily), reduced like the monthly product"
    arrays["lat"], arrays["lon"] = np.asarray(lat), np.asarray(lon)
    arrays["__meta__"] = np.array(json.dumps(dict(kind=kind, month=f"{int(month):02d}", leads=list(LEADS), years=[YEARS[0], YEARS[-1]],
                                                  members=25, order="member-major, year fastest", source=src_note)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + f".part{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, dest)
    print(f"  pseudo-raw {dest.name}: {', '.join(k for k in arrays if k not in ('lat', 'lon', '__meta__'))} "
          f"{dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min", flush=True)
    return dest


def load_pseudo(path: Path, var: str, level=None):
    """→ (vals[sample, 7, lat, lon] with leads 1–4 NaN, lat, lon) — what seas5_build.load_field returns for a
    hindcast path while seas5_ref derives a lead-7 table."""
    key = var if level is None else f"{var}_{int(level)}"
    with np.load(path) as z:
        if key not in z.files:
            names = [k for k in z.files if k not in ("lat", "lon", "__meta__")]
            if level is None and len(names) == 1:
                key = names[0]
            else:
                raise KeyError(f"{path.name}: no {key} (has {names})")
        v = z[key]; lat, lon = z["lat"], z["lon"]
    out = np.full((v.shape[0], 7) + v.shape[2:], np.nan, dtype=np.float32)
    out[:, 4:7] = v
    return out, lat, lon


# ── consistency test: AWS (pseudo-raw) vs the CDS monthly hindcast tables at leads 5 and 6 ─────────
def check(month: str, out: Path | None = None) -> dict:
    import seas5_ref as R
    month = f"{int(month):02d}"
    res = {}

    def wmean(a, lat):
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones(a.shape[-1])[None]
        ok = np.isfinite(a)
        return float(np.nansum(np.where(ok, a, 0) * w) / np.sum(w * ok))

    def field_cmp(name, cds_by_lead, aws_by_lead, lat, sd=None):
        rec = {}
        for L in (5, 6):
            c, a = cds_by_lead.get(L), aws_by_lead.get(L)
            if c is None or a is None:
                continue
            d = a - c
            rec[f"lead{L}"] = dict(mean_diff=wmean(d, lat), rms=float(np.sqrt(wmean(d ** 2, lat))),
                                   p99_abs=float(np.nanpercentile(np.abs(d), 99)), max_abs=float(np.nanmax(np.abs(d))),
                                   corr=float(np.corrcoef(np.nan_to_num(a[np.isfinite(d)]), np.nan_to_num(c[np.isfinite(d)]))[0, 1]))
            if sd is not None and L in sd:
                rec[f"lead{L}"]["rms_over_interannual_sd"] = float(np.sqrt(wmean(d ** 2, lat)) / np.nanmedian(sd[L][sd[L] > 0]))
        if 5 in cds_by_lead and 6 in cds_by_lead and 5 in aws_by_lead and 6 in aws_by_lead:
            dd = (aws_by_lead[6] - cds_by_lead[6]) - (aws_by_lead[5] - cds_by_lead[5])
            rec["offset_change_5_to_6_rms"] = float(np.sqrt(wmean(dd ** 2, lat)))
        res[name] = rec
        print(f"  {name}: " + " | ".join(f"{k}: " + ", ".join(f"{kk} {vv:.4g}" for kk, vv in v.items()) if isinstance(v, dict) else f"{k} {v:.4g}"
                                       for k, v in rec.items()), flush=True)

    def series_cmp(name, cds, aws):
        rec = {}
        for L in (5, 6):
            c, a = cds[:, L - 1], aws[:, L - 1]
            ok = np.isfinite(c) & np.isfinite(a)
            d = a[ok] - c[ok]
            rec[f"lead{L}"] = dict(n=int(ok.sum()), corr=float(np.corrcoef(a[ok], c[ok])[0, 1]), mean_diff=float(d.mean()),
                                   rms=float(np.sqrt((d ** 2).mean())), sd_cds=float(c[ok].std()),
                                   rms_over_sd=float(np.sqrt((d ** 2).mean()) / c[ok].std()))
        res[name] = rec
        print(f"  {name}: " + " | ".join(f"{k}: " + ", ".join(f"{kk} {vv:.4g}" for kk, vv in v.items()) for k, v in rec.items()), flush=True)

    def pz(kind):
        p = pseudo_path(kind, month)
        return np.load(p) if p.exists() else None

    # 1. mean fields of the global kinds (hc_build_{kind}: hcm_{var} [lead, lat, lon]) and the interannual σ (normals)
    ng = R.load(R.hc_ref("normals_gl", month)); nz = R.load(R.hc_ref("normals_gl_z500", month)); ne = R.load(R.hc_ref("normals_energy", month))
    for kind, var, key, fac, nt, nvar in (("gl", "t2m", "t2m", 1.0, ng, "t2m"), ("gl", "tp", "tprate", 86400e3, ng, "tp"), ("gl", "sst", "sst", 1.0, None, None),
                                          ("gl_z500", "z500", "z", 1 / 9.80665, nz, "z500"), ("gl_u850", "u850", "u", 1.0, None, None),
                                          ("gl_v850", "v850", "v", 1.0, None, None), ("na_snowdepth", "sd", "sd", 1000.0, None, None),
                                          ("energy", "si10", "si10", 1.0, ne, "si10")):
        z = pz(kind)
        if z is None or key not in z.files:
            continue
        if kind == "energy":
            t = ne
            if t is None:
                continue
            cds = {L: t[f"{var}_hcm"][L - 1] for L in (5, 6)}
        else:
            t = R.load(R.hc_ref(f"build_{kind}", month))
            if t is None or f"hcm_{var}" not in t:
                continue
            cds = {L: t[f"hcm_{var}"][L - 1] * fac for L in (5, 6)}
        aw = {L: np.nanmean(z[key][:, L - 5], axis=0) * (fac if kind != "energy" else 1.0) for L in (5, 6)}
        sd = None
        if nt is not None and f"{nvar}_sd" in nt:
            sd = {L: nt[f"{nvar}_sd"][L - 1] for L in (5, 6)}
        field_cmp(f"mean {kind}:{var}", cds, aw, z["lat"], sd)
        # tercile bounds at the single month (normals tables: 33rd/67th percentile of the 600 samples, months 1–6 first)
        if nt is not None and f"{nvar}_lo" in nt:
            for L in (5, 6):
                samp = z[key][:, L - 5] * (fac if kind != "energy" else 1.0)
                lo, hi = np.nanpercentile(samp, [100 / 3, 200 / 3], axis=0)
                field_cmp(f"tercile-lo {kind}:{var} lead {L}", {L: nt[f'{nvar}_lo'][L - 1]}, {L: lo}, z["lat"], sd)
                field_cmp(f"tercile-hi {kind}:{var} lead {L}", {L: nt[f'{nvar}_hi'][L - 1]}, {L: hi}, z["lat"], sd)
        if nvar == "z500" and nt is not None:
            # detrended residual terciles (rlo/rhi) — the bounds the z500 maps use
            ny = 24; yr = np.arange(ny) - (ny - 1) / 2
            for L in (5, 6):
                h = z[key][:, L - 5] / 9.80665
                ym_ = np.nanmean(h.reshape(-1, ny, *h.shape[1:]), axis=0)
                slope = (yr[:, None, None] * (ym_ - ym_.mean(0))).sum(0) / (yr ** 2).sum()
                resid = h - (np.nanmean(h, 0) + slope[None] * np.tile(yr, h.shape[0] // ny)[:, None, None])
                rlo, rhi = np.nanpercentile(resid, [100 / 3, 200 / 3], axis=0)
                field_cmp(f"z500 residual tercile-lo lead {L}", {L: nt["z500_rlo"][L - 1]}, {L: rlo}, z["lat"])
                field_cmp(f"z500 residual tercile-hi lead {L}", {L: nt["z500_rhi"][L - 1]}, {L: rhi}, z["lat"])
                field_cmp(f"z500 interannual sd lead {L}", {L: nt["z500_sd"][L - 1]}, {L: np.nanstd(ym_ - slope[None] * yr[:, None, None], axis=0)}, z["lat"])
    # 2. per-sample series: SST box means, polar caps, teleconnection points/caps, stratospheric winds
    from seas5_build import BOXES, box_mean
    z = pz("sst"); t = R.load(R.hc_ref("build_sst", month))
    if z is not None and t is not None:
        names = list(t["meta"]["boxes"])
        v = np.full((z["sst"].shape[0], 6) + z["sst"].shape[2:], np.nan, np.float32); v[:, 4:6] = z["sst"][:, :2]
        for b in ("nino12", "nino3", "nino34", "nino4", "natl", "iod_w", "iod_e", "atl3", "glob"):
            series_cmp(f"sst box {b}", t["box"][names.index(b)], box_mean(v, z["lat"], z["lon"], BOXES[b]))
    for hemi, kind, box in (("nh", "polar_n", (60, 90, -180, 180)), ("sh", "polar_s", (-90, -60, -180, 180))):
        z = pz(kind); t = R.load(R.hc_ref("build_polar", month))
        if z is None or t is None:
            continue
        for lev in [L for L in (10, 100) if f"z_{L}" in z.files]:
            v = np.full((z[f"z_{lev}"].shape[0], 6) + z[f"z_{lev}"].shape[2:], np.nan, np.float32); v[:, 4:6] = z[f"z_{lev}"][:, :2] / 9.80665
            series_cmp(f"polar cap {hemi} z{lev}", t[f"{hemi}_z{lev}"], box_mean(v, z["lat"], z["lon"], box))
    z = pz("nh_msl"); t = R.load(R.hc_ref("tele_nh", month))
    if z is not None and t is not None:
        from seas5_tele import _point, _boxm
        v = np.full((z["msl"].shape[0], 6) + z["msl"].shape[2:], np.nan, np.float32); v[:, 4:6] = z["msl"][:, :2]
        series_cmp("SOI Tahiti-Darwin msl", t["soi_ht"], _point(v, z["lat"], z["lon"], -17.5, -149.5) - _point(v, z["lat"], z["lon"], -12.5, 130.5))
        series_cmp("EqSOI msl", t["eqsoi_he"], _boxm(v, z["lat"], z["lon"], -5, 5, -130, -80) - _boxm(v, z["lat"], z["lon"], -5, 5, 90, 140))
        field_cmp("mean nh_msl", {L: t["hp_m"][L - 1] for L in (5, 6)}, {L: np.nanmean(z["msl"][:, L - 5], 0) for L in (5, 6)}, z["lat"])
    z = pz("nh_z")
    if z is not None and t is not None:
        from seas5_tele import _cap
        for lev in [L for L in (500, 1000) if f"z_{L}" in z.files]:
            v = np.full((z[f"z_{lev}"].shape[0], 6) + z[f"z_{lev}"].shape[2:], np.nan, np.float32); v[:, 4:6] = z[f"z_{lev}"][:, :2] / 9.80665
            series_cmp(f"NAM cap 65N z{lev}", t[f"cap{lev}"], _cap(v, z["lat"], 65))
        field_cmp("mean nh_z z500", {L: t["hz_m"][L - 1] for L in (5, 6)}, {L: np.nanmean(z["z_500"][:, L - 5], 0) for L in (5, 6)}, z["lat"])
    z = pz("strat_u"); t = R.load(R.hc_ref("tele_strat", month))
    if z is not None and t is not None:
        from seas5_tele import _zonal_at, _band_zonal
        v = np.full((z["u_10"].shape[0], 6) + z["u_10"].shape[2:], np.nan, np.float32); v[:, 4:6] = z["u_10"][:, :2]
        series_cmp("u60N 10 hPa", t["u60n10"], _zonal_at(v, z["lat"], 60))
        series_cmp("u60S 10 hPa", t["u60s10"], _zonal_at(v, z["lat"], -60))
        series_cmp("QBO 10 hPa", t["qbo10"], _band_zonal(v, z["lat"], -5, 5))
    res["_aws_stats"] = dict(AWS_STATS)
    if out:
        out.write_text(json.dumps(res, indent=1))
        print(f"wrote {out}", flush=True)
    return res


# ── probe ────────────────────────────────────────────────────────────────────
def probe() -> None:
    import pandas as pd
    for period in ("hindcast", "forecast"):
        root, sess = open_root(period)
        print(f"===== {period}: groups {list(root.group_keys())}", flush=True)
        for gname in ("single/1platx1plon", "pressure/1platx1plon"):
            try:
                g = root[gname]
            except KeyError:
                print(f"  {gname}: absent", flush=True); continue
            for name, a in g.arrays():
                print(f"  {name:28s} shape={a.shape} chunks={a.chunks} dtype={a.dtype} attrs={json.dumps({k: str(v)[:160] for k, v in a.attrs.items()})}", flush=True)
            it = pd.Timestamp(g["init_time"].attrs["units"].split("since ")[1]) + pd.to_timedelta(g["init_time"][:], "D")
            print(f"  init_time {it[0].date()} .. {it[-1].date()} ({len(it)})", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "build", "check"])
    ap.add_argument("--month", default="10")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--aws-only", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        probe(); return 0
    if a.cmd == "build":
        bad = [k for k in (a.kinds or kinds()) if build_kind(k, a.month, a.aws_only) is None]
        print(f"pseudo-raw start {a.month}: AWS {AWS_STATS['chunks']} chunk reads, {AWS_STATS['mb_raw'] / 1e3:.1f} GB decompressed, "
              f"{AWS_STATS['seconds'] / 60:.1f} min" + (f"; FAILED {bad}" if bad else ""), flush=True)
        return 1 if bad else 0
    check(a.month, Path(a.out) if a.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
