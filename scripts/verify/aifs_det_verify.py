#!/usr/bin/env python3
"""Running deterministic verification: AIFS single vs AIFS-ENS control.

Hypothesis under test: the CRPS-trained ENS control — which keeps a realistic
kinetic-energy spectrum at range while the MSE-trained single blurs — may be
the better *deterministic* product, at the cost of arriving a few minutes
later. Few have benchmarked this publicly.

Fairness rules:
  * both models are block-averaged to a common 1.5 deg grid BEFORE scoring
    (6x6 box means of the 0.25 deg fields), so headline scores are compared
    at a resolution both fully resolve and smoothness is not rewarded;
  * truth is ERA5 at the valid 00Z (ARCO, ~6-day latency) on the same grid —
    neither model verifies against its own analyses;
  * scores: RMSE and anomaly correlation (NH extratropics 20-80N,
    cos-weighted; anomalies vs the WB2 1991-2020 day-of-year climatology),
    plus the ACTIVITY RATIO std(forecast anomaly)/std(ERA5 anomaly) — the
    smoothness monitor: a model can win RMSE by damping variance; the ratio
    exposes it.

Fields: z500, msl, u850 (u850 RMSE/activity only — no clim in the store).
Leads: 24..240 h. One archive npz per init (~2 MB); scores appended to the
committed JSON feed the site page reads.

    python aifs_det_verify.py --collect          # archive today's 00Z pair
    python aifs_det_verify.py --verify           # score matured inits vs ERA5
    python aifs_det_verify.py --collect --verify # the daily cron call
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ARCHIVE = HERE / "data" / "archive"
TRUTH = HERE / "data" / "truth"
CLIM = HERE / "data" / "clim_1p5.npz"
SCORES = REPO / "assets" / "verify" / "aifs_det_scores.json"
WB2_STORE = Path("~/era5_store/wb2_1p5_daily").expanduser()
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

LEADS = list(range(24, 241, 24))
G = 9.80665
MODELS = {"single": ("aifs-single", "fc", "oper"), "control": ("aifs-ens", "cf", "enfo")}
VARS = ("z500", "msl", "u850")


def coarsen(a):
    """0.25 deg (721x1440) -> 1.5 deg block means (120x240), poles trimmed."""
    a = a[:720, :]
    return a.reshape(120, 6, 240, 6).mean(axis=(1, 3))


def grid_1p5():
    lat = 90.0 - 0.25 * (np.arange(720).reshape(120, 6).mean(axis=1))
    lon = 0.25 * (np.arange(1440).reshape(240, 6).mean(axis=1))
    return lat, lon


def collect(date: str) -> bool:
    """Fetch both models' z500/msl/u850 at all leads, coarsen, archive."""
    from ecmwf.opendata import Client
    out = ARCHIVE / f"{date}00.npz"
    if out.exists():
        print(f"{date}: archived")
        return True
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stash = {}
    iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    for mkey, (model, typ, stream) in MODELS.items():
        c = Client(source="ecmwf", model=model)
        for step in LEADS:
            tmp_pl = HERE / "data" / f"_tmp_pl.grib2"
            tmp_sf = HERE / "data" / f"_tmp_sf.grib2"
            try:
                c.retrieve(date=iso, time=0, type=typ, stream=stream, levtype="pl",
                           param=["z", "u"], levelist=[500, 850], step=step,
                           target=str(tmp_pl))
                c.retrieve(date=iso, time=0, type=typ, stream=stream, levtype="sfc",
                           param="msl", step=step, target=str(tmp_sf))
            except Exception as e:                        # noqa: BLE001
                print(f"{date} {mkey} step {step}: fetch failed ({str(e)[:70]})",
                      file=sys.stderr)
                return False
            dpl = xr.open_dataset(tmp_pl, engine="cfgrib",
                                  backend_kwargs={"indexpath": ""})
            dsf = xr.open_dataset(tmp_sf, engine="cfgrib",
                                  backend_kwargs={"indexpath": ""})
            # grids arrive 90..-90 x -180..180; roll longitudes to 0..360
            def field(a):
                v = a.values
                return coarsen(np.roll(v, v.shape[1] // 2, axis=1))
            stash[(mkey, "z500", step)] = field(dpl["z"].sel(isobaricInhPa=500)) / G
            stash[(mkey, "u850", step)] = field(dpl["u"].sel(isobaricInhPa=850))
            stash[(mkey, "msl", step)] = field(dsf["msl"]) / 100.0
            dpl.close(); dsf.close()
        print(f"{date} {mkey}: {len(LEADS)} leads archived", flush=True)
    np.savez_compressed(out, **{f"{m}_{v}_{s}": stash[(m, v, s)].astype(np.float32)
                                for (m, v, s) in stash})
    for t in (HERE / "data" / "_tmp_pl.grib2", HERE / "data" / "_tmp_sf.grib2"):
        t.unlink(missing_ok=True)
    return True


def fetch_truth(valid: pd.Timestamp) -> bool:
    out = TRUTH / f"{valid:%Y%m%d}.npz"
    if out.exists():
        return True
    TRUTH.mkdir(parents=True, exist_ok=True)
    try:
        ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
        t = valid.to_datetime64()
        z = ds["geopotential"].sel(time=t, level=500).values
        if not np.isfinite(z).all():
            return False
        u = ds["u_component_of_wind"].sel(time=t, level=850).values
        m = ds["mean_sea_level_pressure"].sel(time=t).values
    except Exception as e:                                # noqa: BLE001
        print(f"truth {valid:%Y-%m-%d}: {str(e)[:70]}", file=sys.stderr)
        return False
    np.savez_compressed(out, z500=coarsen(z).astype(np.float32) / G,
                        u850=coarsen(u).astype(np.float32),
                        msl=coarsen(m).astype(np.float32) / 100.0)
    print(f"truth {valid:%Y-%m-%d} ✓", flush=True)
    return True


def load_clim():
    """(dayofyear, lat, lon) z500 & msl climatology on our 1.5 grid (NH from
    the WB2 store, mirrored NaN south of 0)."""
    if CLIM.exists():
        z = np.load(CLIM)
        return {"z500": z["z500"], "msl": z["msl"]}
    lat, lon = grid_1p5()
    out = {}
    for var, store, scale in (("z500", "z500", 1.0), ("msl", "slp", 1.0)):
        das = []
        for y in range(1991, 2021):
            f = WB2_STORE / store / f"{store}_{y}.nc"
            das.append(xr.open_dataset(f)[store])
        da = xr.concat(das, dim="time").transpose("time", "latitude", "longitude")
        doy = da.time.dt.dayofyear.values
        vals = da.values * scale
        clim = np.full((366, len(lat), len(lon)), np.nan, np.float32)
        wlat = da.latitude.values
        wlon = da.longitude.values
        for d in range(1, 367):
            sel = np.abs(((doy - d + 183) % 366) - 183) <= 7
            c = vals[sel].mean(axis=0)
            cda = xr.DataArray(c, coords=dict(latitude=wlat, longitude=wlon),
                               dims=("latitude", "longitude"))
            clim[d - 1] = cda.interp(latitude=lat, longitude=lon).values
        out[var] = clim
        print(f"clim {var} built", flush=True)
    np.savez_compressed(CLIM, **out)
    return out


def verify() -> int:
    clim = load_clim()
    lat, lon = grid_1p5()
    band = (lat >= 20) & (lat <= 80)
    w = np.cos(np.deg2rad(lat[band]))[:, None] * np.ones((band.sum(), len(lon)))
    recs = []
    if SCORES.exists():
        recs = json.loads(SCORES.read_text())["records"]
    done = {(r["init"], r["lead"], r["var"], r["model"]) for r in recs}
    n_new = 0
    for arch in sorted(ARCHIVE.glob("*.npz")):
        init = pd.Timestamp(arch.stem[:8])
        z = None
        for lead in LEADS:
            valid = init + pd.Timedelta(hours=lead)
            key0 = (arch.stem, lead, "z500", "single")
            if key0 in done:
                continue
            if not fetch_truth(valid):
                continue
            t = np.load(TRUTH / f"{valid:%Y%m%d}.npz")
            if z is None:
                z = np.load(arch)
            doy = min(valid.dayofyear, 366)
            for var in VARS:
                tv = t[var][band]
                cv = clim.get(var)
                ta = tv - cv[doy - 1][band] if cv is not None else None
                for mkey in MODELS:
                    fv = z[f"{mkey}_{var}_{lead}"][band]
                    err = fv - tv
                    rec = dict(init=arch.stem, lead=lead, var=var, model=mkey,
                               rmse=round(float(np.sqrt(np.nansum(w * err**2)
                                                        / np.nansum(w * np.isfinite(err)))), 3))
                    if ta is not None:
                        fa = fv - cv[doy - 1][band]
                        m = np.isfinite(fa) & np.isfinite(ta)
                        wa = w[m]
                        acc = (np.nansum(wa * fa[m] * ta[m])
                               / np.sqrt(np.nansum(wa * fa[m]**2) * np.nansum(wa * ta[m]**2)))
                        rec["acc"] = round(float(acc), 4)
                        rec["act"] = round(float(np.sqrt(np.nansum(wa * fa[m]**2)
                                                         / np.nansum(wa * ta[m]**2))), 3)
                    recs.append(rec)
                    n_new += 1
    if n_new:
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        SCORES.write_text(json.dumps(
            {"generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
             "grid": "1.5° block means of 0.25° fields", "truth": "ERA5 (ARCO) 00Z",
             "acc_base": "WB2 ERA5 1991–2020 ±7d doy climatology, NH",
             "records": recs}, separators=(",", ":")))
        print(f"scores: +{n_new} records → {len(recs)} total")
    else:
        print("no newly verifiable (ERA5 lag ~6 days)")
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--date", default=pd.Timestamp.utcnow().strftime("%Y%m%d"))
    args = ap.parse_args()
    if args.collect:
        collect(args.date)
    if args.verify:
        verify()
    if not (args.collect or args.verify):
        collect(args.date)
        verify()


if __name__ == "__main__":
    main()
