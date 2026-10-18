#!/usr/bin/env python3
"""The equatorial 0-300 m heat-content index (tao_hc.py) rebuilt from two MONTHLY ocean reanalyses, so
events before TAO (1982/83) can sit on the chart, and every past event can be put on one common basis.

Definition - the TAO index's, as closely as gridded monthly fields allow:
  * the same 8 points on the equator (165E, 180, 170W, 155W, 140W, 125W, 110W, 95W);
  * each point = mean over the reanalysis grid cells within +-0.5 deg lat and +-0.5 deg lon of the
    mooring (GODAS: lat -0.50/-0.17/+0.17/+0.50 x 2 lons = 8 cells; ORAS5: the ORCA025 cells in the box);
  * 0-300 m depth-weighted mean POTENTIAL temperature (GODAS: pottmp interpolated to the index's 5 m
    grid 0-300 m exactly as TAO profiles are; ORAS5: its own 0-300 m heat content sohtc300 / (rho0 cp 300 m),
    which is the layer-thickness-weighted integral of potential temperature - checked against a 3-D month
    (--check-oras5-3d: within 0.04 C of the 5 m-grid 0-300 m mean at every point);
  * anomaly at each point vs that reanalysis' OWN 1991-2020 monthly climatology;
  * index = fixed 1/8 weight per point (the reanalyses have no gaps).

Data (fetched once outside Actions, kept in scripts/sst/data/):
  GODAS  NOAA PSL OPeNDAP Datasets/godas/pottmp.YYYY.nc, 1980-, read SEQUENTIALLY (parallel reads
         silently corrupt), each year screened before caching
  ORAS5  ECMWF/C3S CDS reanalysis-oras5, single_level ocean_heat_content_for_the_upper_300m,
         consolidated 1958-2014 + operational 2015-; ONE CDS request at a time
Output (committed, read by sst_subsurface_events.py in Actions - nothing is fetched there):
  reference/reanalysis_hc_monthly.csv   month x {godas,oras5}_{lon} + godas, oras5 (deg C)
  reference/reanalysis_hc_validation.json  vs the TAO index 1993- and vs each other in 1982/83

Usage:  python reanalysis_hc.py --fetch-godas | --fetch-oras5 | --check-oras5-3d | --build
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sst_subsurface as ss
import tao_hc as H

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SST_DATA_DIR", HERE / "data"))  # raw stores (SST_DATA_DIR overrides)
GDIR = DATA / "godas"
ODIR = DATA / "oras5"
REF = HERE / "reference"
OUT_CSV = REF / "reanalysis_hc_monthly.csv"
OUT_JSON = REF / "reanalysis_hc_validation.json"

GODAS_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas/pottmp.{}.nc"
FIRST_YEAR = 1980
BOX = 0.5  # deg: half-width of the lat/lon box around each mooring
CLIM = ("1991-01-01", "2020-12-01")
# ORAS5 (NEMO 3.4) reference density and heat capacity. 1026 x 3991.87 (NEMO >= 3.6) leaves sohtc300 / (rho cp 300 m)
# 0.8-1.1 % above the 0-300 m mean of the 3-D potential temperature at all 8 points (2014-12 check); 1035 x 4000
# matches it to within 0.04 C (check_3d_201412.json; the rest is model layers vs the 5 m interpolation).
RHO0, CP = 1035.0, 4000.0


def log(*a):
    print(*a, flush=True)


# ── GODAS ─────────────────────────────────────────────────────────────────────
def _godas_ok(d: xr.DataArray) -> str:
    v = d.values
    if not np.isfinite(v).all():
        return f"{int((~np.isfinite(v)).sum())} non-finite values"
    if v.min() < 280 or v.max() > 306:
        return f"range {v.min():.1f}-{v.max():.1f} K outside 280-306"
    if d.sizes["lat"] != 4 or d.sizes["lon"] != 102 or d.sizes["level"] != 26:
        return f"shape {dict(d.sizes)}"
    return "ok"


def fetch_godas(years=None):
    """Equatorial slab (lat +-0.55, lon 164-266E, levels 5-303 m) per year, sequential, screened."""
    GDIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    years = years or range(FIRST_YEAR, now.year + 1)
    for y in years:
        p = GDIR / f"godas_eq_pottmp_{y}.nc"
        if p.exists() and y < now.year:
            continue
        for attempt in range(4):
            try:
                ds = xr.open_dataset(GODAS_URL.format(y), engine="netcdf4")
                d = ds["pottmp"].sel(lat=slice(-0.55, 0.55), lon=slice(164.0, 266.0), level=slice(0, 310)).load()
                ds.close()
            except Exception as e:  # noqa: BLE001
                log(f"GODAS {y}: read failed ({e}); retry {attempt + 1}")
                continue
            why = _godas_ok(d)
            if why == "ok":
                d.to_dataset().to_netcdf(p)
                log(f"GODAS {y}: {d.sizes['time']} months cached")
                break
            log(f"GODAS {y}: screen failed ({why}); retry {attempt + 1}")
        else:
            raise SystemExit(f"GODAS {y}: no clean read after 4 attempts")


def godas_points() -> pd.DataFrame:
    """Monthly 0-300 m mean potential temperature (deg C) at the 8 moorings."""
    rows = []
    for p in sorted(GDIR.glob("godas_eq_pottmp_*.nc")):
        d = xr.open_dataset(p)["pottmp"].load() - 273.15
        assert _godas_ok(d + 273.15) == "ok", p
        dep = d.level.values.astype(float)
        t = pd.to_datetime(d.time.values)
        out = np.full((len(t), len(H.LONS)), np.nan)
        for j, lo in enumerate(H.LONS):
            sub = d.sel(lon=slice(lo - BOX, lo + BOX))  # 2 lons x 4 lats
            assert sub.sizes["lon"] == 2, (lo, sub.lon.values)
            V = sub.transpose("time", "lat", "lon", "level").values
            for i in range(len(t)):
                cols = [ss._vinterp(dep, V[i, a, b]).mean() for a in range(V.shape[1]) for b in range(V.shape[2])]
                out[i, j] = np.mean(cols)
        rows.append(pd.DataFrame(out, index=t, columns=H.LONS))
    df = pd.concat(rows).sort_index()
    df.index = df.index.to_period("M").to_timestamp()
    return df


# ── ORAS5 ─────────────────────────────────────────────────────────────────────
def _wait_cds_key():
    """ONE CDS key: before every request wait (15-min checks) until no Actions run of the `seas5`
    concurrency group (seas5*.yml, c3s*.yml, c3s-u10-skill.yml) is queued or running."""
    import shutil
    import subprocess
    import time

    q = '[.[] | select(.workflowName | test("seas5|c3s"; "i"))] | length'
    while True:
        busy = 0
        for st in ("in_progress", "queued", "waiting", "pending", "requested"):
            r = subprocess.run(
                [
                    os.environ.get("GH_BIN") or shutil.which("gh") or "gh",
                    "run",
                    "list",
                    "-R",
                    "scorvec/scorvec.github.io",
                    "--status",
                    st,
                    "--limit",
                    "50",
                    "--json",
                    "workflowName",
                    "--jq",
                    q,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            busy += int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else 1
        if not busy:
            return
        log(f"{datetime.now(UTC):%H:%MZ} CDS key in use by an Actions run; next check in 15 min")
        time.sleep(900)


def _cds_retrieve(req: dict, target: Path):
    import cdsapi

    _wait_cds_key()
    c = cdsapi.Client()
    log(f"CDS request -> {target.name}: {json.dumps(req)[:300]}")
    c.retrieve("reanalysis-oras5", req, str(target))


def fetch_oras5():
    """sohtc300 monthly, one CDS request per block of years, strictly one after another."""
    ODIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    # consolidated 1958-2014, operational 2015- (the CDS lags ~2 months: current year to month-2)
    blocks = [(y, min(y + 9, 2014)) for y in range(FIRST_YEAR, 2015, 10)] + [(2015, now.year - 1), (now.year, now.year)]
    for y0, y1 in blocks:
        z = ODIR / f"oras5_sohtc300_{y0}_{y1}.zip"
        if z.exists() and y1 < now.year and zipfile.is_zipfile(z):
            continue
        last = now.month - 2 if y1 == now.year else 12
        req = {
            "product_type": ["consolidated" if y1 <= 2014 else "operational"],
            "vertical_resolution": "single_level",
            "variable": ["ocean_heat_content_for_the_upper_300m"],
            "year": [str(y) for y in range(y0, y1 + 1)],
            "month": [f"{m:02d}" for m in range(1, last + 1)],
        }
        _cds_retrieve(req, z)
    _oras5_unpack()


def _oras5_unpack():
    """Cut the equatorial strip out of every global monthly file into one small netCDF
    (the CDS serves ORAS5 only on the global native ORCA025 grid - no area subsetting)."""
    out = ODIR / "oras5_sohtc300_eq.nc"
    times, vals, sl, la, lo = [], [], None, None, None
    for z in sorted(ODIR.glob("oras5_sohtc300_*_*.zip")):
        with zipfile.ZipFile(z) as zf:
            for n in sorted(zf.namelist()):
                if not n.endswith(".nc"):
                    continue
                tmp = ODIR / "_tmp.nc"
                tmp.write_bytes(zf.read(n))
                with xr.open_dataset(tmp) as ds:
                    if sl is None:
                        lat, lon = ds["nav_lat"].values, ds["nav_lon"].values % 360
                        jj, ii = np.where((np.abs(lat) <= BOX + 0.3) & (lon >= 163) & (lon <= 267))
                        sl = (slice(jj.min(), jj.max() + 1), slice(ii.min(), ii.max() + 1))
                        la, lo = lat[sl], lon[sl]
                    else:
                        assert np.array_equal(ds["nav_lat"].values[sl], la), n
                    v = ds["sohtc300"].values
                    for k, t in enumerate(pd.to_datetime(ds["time_counter"].values)):
                        times.append(t.to_period("M").to_timestamp())
                        vals.append(v[k][sl])
                tmp.unlink()
    df_t = pd.Index(times)
    keep = ~df_t.duplicated(keep="last")
    order = np.argsort(df_t[keep].values)
    V = np.stack(vals)[keep][order]
    ds = xr.Dataset(
        {
            "sohtc300": (("time", "y", "x"), V, {"units": "J m-2"}),
            "nav_lat": (("y", "x"), la),
            "nav_lon": (("y", "x"), lo),
        },
        coords={"time": df_t[keep][order]},
    )
    ds.to_netcdf(out)
    log(f"ORAS5 strip: {dict(ds.sizes)} {ds.time.values[0]} .. {ds.time.values[-1]} -> {out}")


def oras5_points() -> pd.DataFrame:
    ds = xr.open_dataset(ODIR / "oras5_sohtc300_eq.nc").load()
    lat, lon = ds["nav_lat"].values, ds["nav_lon"].values % 360
    hc = ds["sohtc300"].values  # (time, y, x) J m-2
    t = pd.to_datetime(ds.time.values).to_period("M").to_timestamp()
    out = {}
    for lo in H.LONS:
        m = (np.abs(lat) <= BOX) & (np.abs(lon - lo) <= BOX)
        assert m.sum() >= 4, (lo, m.sum())
        out[lo] = hc[:, m].mean(axis=1) / (RHO0 * CP * 300.0)
    df = pd.DataFrame(out, index=t)
    bad = df[(df < 10) | (df > 30)].stack()
    assert bad.empty, bad.head()
    return df


def check_oras5_3d():
    """One 3-D month (potential temperature, all levels) to confirm sohtc300/(rho0 cp 300) is the
    0-300 m mean of potential temperature as the TAO index takes it (5 m grid)."""
    z = ODIR / "oras5_thetao_201412.zip"
    if not z.exists():
        _cds_retrieve(
            {
                "product_type": ["consolidated"],
                "vertical_resolution": "all_levels",
                "variable": ["potential_temperature"],
                "year": ["2014"],
                "month": ["12"],
            },
            z,
        )
    with zipfile.ZipFile(z) as zf:
        n = next(k for k in zf.namelist() if k.endswith(".nc"))
        tmp = ODIR / "_t3d.nc"
        tmp.write_bytes(zf.read(n))
    ds = xr.open_dataset(tmp)
    lat, lon = ds["nav_lat"].values, ds["nav_lon"].values % 360
    v = next(k for k in ds.data_vars if k.startswith(("votemper", "thetao")))
    th = ds[v]
    zname = next(d for d in th.dims if d.startswith("deptht") or d == "depth")
    dep = ds[zname].values.astype(float)
    hc = oras5_points().loc["2014-12-01"]
    res = {}
    for lo in H.LONS:
        m = (np.abs(lat) <= BOX) & (np.abs(lon - lo) <= BOX)
        jj, ii = np.where(m)
        prof = [ss._vinterp(dep, th.isel(time_counter=0)[:, j, i].values).mean() for j, i in zip(jj, ii)]
        res[H.LON_LABEL[lo]] = {
            "thetao_0_300_mean": round(float(np.mean(prof)), 3),
            "sohtc300_over_rho_cp_300": round(float(hc[lo]), 3),
        }
    tmp.unlink()
    log(json.dumps(res, indent=1))
    (ODIR / "check_3d_201412.json").write_text(json.dumps(res, indent=1))
    return res


# ── index + validation ────────────────────────────────────────────────────────
def anomalies(pts: pd.DataFrame) -> pd.DataFrame:
    base = pts.loc[CLIM[0] : CLIM[1]]
    assert len(base) == 360, len(base)
    clim = base.groupby(base.index.month).mean()
    return pts - clim.reindex(pts.index.month).values


def _stats(a: pd.Series, b: pd.Series) -> dict:
    j = pd.concat([a, b], axis=1).dropna()
    x, y = j.iloc[:, 0], j.iloc[:, 1]
    d = x - y
    r = float(np.corrcoef(x, y)[0, 1])
    a1 = float(pd.Series(d.values).autocorr(1)) if len(d) > 3 else np.nan
    return {
        "n": len(j),
        "r": round(r, 3),
        "bias": round(float(d.mean()), 3),
        "rms": round(float(np.sqrt((d**2).mean())), 3),
        "sd_diff": round(float(d.std()), 3),
        "slope": round(float(np.polyfit(y, x, 1)[0]), 3),
        "lag1_diff": round(a1, 2),
    }


def tao_monthly() -> pd.DataFrame:
    """TAO-only monthly index (all 8 moorings screened-live >= 20 days of the month) and per-mooring
    monthly means (>= 20 live days), from the audit artifact (tao-hc-audit.yml)."""
    aud = DATA / "hc_audit" / "hc_audit"
    m = pd.read_csv(aud / "tao_mooring_anom_strict.csv", index_col=0, parse_dates=True)
    m.columns = [float(c) for c in m.columns]
    g = m.groupby(m.index.to_period("M"))
    mon = g.mean().where(g.count() >= 20)
    mon.index = mon.index.to_timestamp()
    d = pd.read_csv(aud / "daily.csv", index_col=0, parse_dates=True)
    for name, need in (("tao8", 8), ("tao6", 6)):  # all 8 live / >= 6 live (rest interpolated in lon)
        full = d["new_hc"].where(d["n_live"] >= need)
        gi = full.groupby(full.index.to_period("M"))
        idx = gi.mean().where(gi.count() >= 20)
        idx.index = idx.index.to_timestamp()
        mon[name] = idx
    return mon


def build():
    pg, po = godas_points(), oras5_points()
    ag, ao = anomalies(pg), anomalies(po)
    out = pd.DataFrame(index=ag.index.union(ao.index))
    for lo in H.LONS:
        out[f"godas_{H.LON_LABEL[lo]}"] = ag[lo]
        out[f"oras5_{H.LON_LABEL[lo]}"] = ao[lo]
    out["godas"] = ag.mean(axis=1, skipna=False)
    out["oras5"] = ao.mean(axis=1, skipna=False)
    out = out.loc[f"{FIRST_YEAR}-01-01" :]
    REF.mkdir(exist_ok=True)
    out.index.name = "month"
    out.to_csv(OUT_CSV, float_format="%.3f")
    log(f"wrote {OUT_CSV} ({len(out)} months, {out.index.min():%Y-%m} .. {out.index.max():%Y-%m})")

    tm = tao_monthly()
    val = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "definition": {
            "points": [H.LON_LABEL[l] for l in H.LONS],
            "box_deg": BOX,
            "climatology": "each reanalysis' own 1991-2020 monthly mean",
            "godas": "NOAA NCEP GODAS pottmp (PSL), interpolated to 5 m, 0-300 m mean",
            "oras5": "ECMWF ORAS5 (CDS control member) sohtc300 / (1035 kg m-3 x 4000 J kg-1 K-1 x 300 m)",
            "oras5_3d_check_2014_12": json.loads((ODIR / "check_3d_201412.json").read_text())
            if (ODIR / "check_3d_201412.json").exists()
            else None,
        },
    }
    v = {}
    for name in ("oras5", "godas"):
        v[f"{name}_vs_tao8_1993on"] = _stats(out[name], tm["tao8"].loc["1993":])
        v[f"{name}_vs_tao6_1993on"] = _stats(out[name], tm["tao6"].loc["1993":])
        for y in (1997, 2015, 2023):
            v[f"{name}_vs_tao6_{y}_{str(y + 1)[2:]}"] = _stats(
                out[name].loc[f"{y}" : f"{y + 1}"], tm["tao6"].loc[f"{y}" : f"{y + 1}"]
            )
        v[f"{name}_per_mooring_1993on"] = {
            H.LON_LABEL[lo]: _stats(out[f"{name}_{H.LON_LABEL[lo]}"], tm[lo].loc["1993":]) for lo in H.LONS
        }
    v["oras5_vs_godas_1980on"] = _stats(out["oras5"], out["godas"])
    v["oras5_vs_godas_1982_83"] = _stats(out["oras5"].loc["1982":"1983"], out["godas"].loc["1982":"1983"])
    v["oras5_vs_godas_1980_92"] = _stats(out["oras5"].loc["1980":"1992"], out["godas"].loc["1980":"1992"])
    v["oras5_vs_godas_1993on"] = _stats(out["oras5"].loc["1993":], out["godas"].loc["1993":])
    dd = (out["oras5"] - out["godas"]).loc["1982":"1983"]
    v["oras5_minus_godas_1982_83_maxabs"] = {
        "value": round(float(dd.loc[dd.abs().idxmax()]), 2),
        "month": f"{dd.abs().idxmax():%Y-%m}",
    }
    for name in ("oras5", "godas", "tao6"):
        s = out[name] if name != "tao6" else tm["tao6"]
        ev = {}
        for y in (1982, 1997, 2015, 2023, 2026):
            w = s.loc[f"{y}-01-01" : f"{y + 1}-12-01"].dropna()
            if len(w):
                ev[str(y)] = {
                    "max": round(float(w.max()), 2),
                    "max_month": f"{w.idxmax():%Y-%m}",
                    "min": round(float(w.min()), 2),
                    "min_month": f"{w.idxmin():%Y-%m}",
                    "n": len(w),
                }
        v[f"events_{name}"] = ev
    val["validation"] = v
    OUT_JSON.write_text(json.dumps(val, indent=1))
    log(json.dumps(val, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--fetch-godas" in a:
        fetch_godas()
    if "--fetch-oras5" in a:
        fetch_oras5()
    if "--check-oras5-3d" in a:
        check_oras5_3d()
    if "--build" in a:
        build()
