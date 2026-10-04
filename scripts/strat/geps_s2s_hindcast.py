#!/usr/bin/env python3
"""GEPS (ECCC) reforecast drift for the polar-vortex panel, from the ECMWF S2S archive (2026-09-26; user: the GEPS 10 hPa
winds "don't quite match" a bias-corrected chart elsewhere - "Must be bias corrected").

GEPS drifts with lead: in autumn its stratospheric vortex spins up too slowly, so a raw run slides toward a weak vortex
whatever the real atmosphere does (against a chart corrected with CMC's 2001-2020 reforecasts, ours read ~5 m/s low at
day 20 and ~11 m/s low at day 36 for the 2026-09-24 run). The correction is the standard one:

    corrected(lead) = raw(lead) - [ model(lead) - obs(valid) ]     averaged over the reforecast years

model = the ECCC reforecast (ECMWF Data Store dataset "s2s-reforecasts", origin eccc: control + 3 perturbed members,
2001-2020, issued Mondays and Thursdays, daily leads 24-936 h) from the model-version starts nearest in day of year;
(S2S left the old WEB-API for ECDS on 2026-04-21; the WEB-API closed 2026-05-27. ECDS takes the CDS token.) obs = MERRA-2 on the same valid dates (scripts/telecon/data/
m2_strat, daily means). Licence: ECMWF S2S, non-commercial research (user OK 2026-09-26: "the site is just a hobbyist
page"). LAPTOP ONLY (~/.cdsapirc token, ECDS url).

    python geps_s2s_hindcast.py fetch [--year 2025] [--from 08-14 --to 12-31]   # resumable, one file per start date, nearest dates first
    python geps_s2s_hindcast.py fetch --parts zt_nh                            # T + height 20-90N for the anomaly maps
    python geps_s2s_hindcast.py bias                                           # -> data/geps_s2s/bias_nh.json
    python geps_s2s_hindcast.py mapbias                                        # -> data/geps_s2s/mapbias_{nh,sh}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "geps_s2s"
M2 = HERE.parent / "telecon" / "data" / "m2_strat"
HYEARS = list(range(2001, 2021))
LEADS = list(range(24, 841, 24))              # GEPS extended runs to 840 h; ECDS has no lead 0 for these reforecasts
# ECDS subsets by area but does not regrid (native 1.5 deg): a 60N band for the wind and the 65-90N cap for the
# temperature keep a start date near 30 MB. Cap height is not fetched: its panel is not corrected.
PARTS = {"ut": {"variable": ["u_component_of_wind", "temperature"], "level_value": ["10_hpa", "100_hpa"],
                "area": [90, -180, 58.5, 180]},   # one queue trip per start date: the ECDS S2S queue held 2,150 requests on 2026-09-26
         # the vortex MAPS (2026-09-27; user: "Now that we have the 10mb hindcasts for the geps, can we update the
         # stratosphere maps to look like the [GEFS] version with height and temp anomaly plots?"): T and height at 10 and
         # 100 hPa over each plotted hemisphere, 20-90 deg, for the lead-dependent drift of the anomaly maps (mapbias)
         "zt_nh": {"variable": ["geopotential_height", "temperature"], "level_value": ["10_hpa", "100_hpa"],
                   "area": [90, -180, 20, 180]},
         "zt_sh": {"variable": ["geopotential_height", "temperature"], "level_value": ["10_hpa", "100_hpa"],
                   "area": [-20, -180, -90, 180]},
         # the forecast dripping paint (2026-09-27; user: "zonal mean 'dripping paint' plots from the geps/gefs as
         # well"): polar-cap height drift at the tropospheric and lower-stratospheric levels the zt parts lack
         "ghcap": {"variable": ["geopotential_height"],
                   "level_value": ["50_hpa", "200_hpa", "300_hpa", "500_hpa", "700_hpa", "850_hpa", "925_hpa", "1000_hpa"],
                   "area": [90, -180, 63, 180]}}


def ecds():
    import re
    import cdsapi
    key = re.search(r"key:\s*(\S+)", (Path.home() / ".cdsapirc").read_text()).group(1)     # never printed
    return cdsapi.Client(url="https://ecds.ecmwf.int/api", key=key, quiet=True, progress=False)


def version_dates(year: int, a: str, b: str):
    """ECCC reforecast issue dates (Mondays and Thursdays) in the window, the ones nearest today's day of year first:
    a queued archive takes hours per request, so the current forecast's season is corrected first."""
    d = [x for x in pd.date_range(f"{year}-{a}", f"{year}-{b}", freq="D") if x.dayofweek in (0, 3)]
    today = pd.Timestamp.utcnow().dayofyear
    return sorted(d, key=lambda x: min(abs(x.dayofyear - today), 365 - abs(x.dayofyear - today)))


def fetch(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = ecds()
    parts = a.parts.split(",")
    for d in version_dates(a.year, a.frm, a.to):
        for part, req in ((k, PARTS[k]) for k in parts):
            dest = OUT / f"hc_{d:%Y%m%d}_{part}.grib"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            t0 = time.time()
            try:
                c.retrieve("s2s-reforecasts", {
                    "origin": "eccc", "year": f"{d:%Y}", "month": f"{d:%m}", "day": f"{d:%d}", "time": "00:00",
                    "hyear": [str(y) for y in HYEARS], "hmonth": f"{d:%m}", "hday": f"{d:%d}",
                    "level_type": "pressure", "forecast_type": ["control_forecast", "perturbed_forecast"],
                    "leadtime_hour": [str(h) for h in LEADS], "data_format": "grib", **req}, str(dest) + ".part")
                Path(str(dest) + ".part").rename(dest)
                print(f"{d:%Y-%m-%d} {part}: {dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min", flush=True)
            except Exception as e:                                    # noqa: BLE001  keep going; rerun resumes
                print(f"{d:%Y-%m-%d} {part}: FAILED {str(e)[:200]}", flush=True)


def reduce_file(path: Path) -> dict:
    """[hdate, member, lead] arrays from one start date's file: u60 at 10/100 hPa and the 65-90N cap-mean temperature
    at 100 hPa. Control and perturbed members are separate GRIB data types."""
    import xarray as xr
    arrs = {}
    for dt in ("cf", "pf"):
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": "", "filter_by_keys": {"dataType": dt}})
        for name in ("u", "t"):
            v = ds[name]
            if "number" not in v.dims:
                v = v.expand_dims(number=[0])
            if name == "u":
                for lev in (10, 100):
                    z = v.sel(isobaricInhPa=lev).mean("longitude").sel(latitude=60.0)
                    arrs.setdefault(f"u60_{lev}", []).append(z.transpose("time", "number", "step").values)
            else:
                lat = v.latitude
                w = np.cos(np.deg2rad(lat.where(lat >= 65, drop=True)))
                z = v.sel(isobaricInhPa=100).where(lat >= 65, drop=True).mean("longitude")
                cap = (z * w).sum("latitude") / w.sum()
                arrs.setdefault("capT_100", []).append(cap.transpose("time", "number", "step").values)
        hd = pd.DatetimeIndex(ds.time.values).strftime("%Y-%m-%d").values
        steps = (ds.step.values / np.timedelta64(1, "h")).astype(int)
    out = {k: np.concatenate(v, axis=1).astype("float32") for k, v in arrs.items()}
    out["hdate"], out["step_h"] = hd, steps
    return out


def m2_series():
    """MERRA-2 daily u60 (10, 100 hPa) and cap T (100 hPa) for 2001-2021."""
    import xarray as xr
    parts = [xr.open_dataset(M2 / f"m2_strat_{y}.nc")[["u60", "cap_T"]].sel(lev=[10.0, 100.0], method="nearest")
             for y in range(2001, 2022)]
    d = xr.concat(parts, "time")
    return {"u60_10": d.u60.sel(lev=10.0, method="nearest").to_series(),
            "u60_100": d.u60.sel(lev=100.0, method="nearest").to_series(),
            "capT_100": d.cap_T.sel(lev=100.0, method="nearest").to_series()}


def bias(a) -> None:
    """Per start date and lead: model mean over years x members minus MERRA-2 over the same valid dates.
    Leads start at 24 h (ECDS has no lead 0 for these); step_h says which lead each value is."""
    obs = m2_series()
    table = {"source": "ECMWF S2S cwao enfh (ECCC GEPS reforecast, 4 members, 2001-2020) minus MERRA-2 daily means",
             "starts": {}}
    for f in sorted(OUT.glob("hc_*_ut.grib")):
        d = f.name.split("_")[1]
        try:
            rc = reduce_file(f)
        except Exception as e:                                        # noqa: BLE001
            print(d, "reduce failed:", str(e)[:150]); continue
        ent = {"n_years": int(len(rc["hdate"])), "step_h": rc["step_h"].tolist()}
        for k in ("u60_10", "u60_100", "capT_100"):
            if k not in rc:
                continue
            m = rc[k]                                                 # [hdate, member, step]
            model = np.nanmean(m, axis=(0, 1))
            if k in obs:
                o = np.full((len(rc["hdate"]), len(rc["step_h"])), np.nan)
                for i, h in enumerate(rc["hdate"]):
                    valid = pd.Timestamp(h) + pd.to_timedelta(rc["step_h"], "h")
                    o[i] = obs[k].reindex(valid.normalize()).values
                ent[k] = np.round(model - np.nanmean(o, axis=0), 3).tolist()
            else:
                ent[k + "_drift"] = np.round(model - model[0], 3).tolist()
        table["starts"][d] = ent
        b = ent["u60_10"]
        print(f"{d}: u60 10 hPa bias day 1 {b[0]:+.1f}, day 10 {b[9]:+.1f}, day 20 {b[19]:+.1f}, day {len(b)} {b[-1]:+.1f} m/s", flush=True)
    (OUT / "bias_nh.json").write_text(json.dumps(table))
    print(f"wrote {OUT / 'bias_nh.json'} ({len(table['starts'])} start dates)")


def mapbias(a) -> None:
    """Zonal-mean drift of the ANOMALY MAPS, per start date, hemisphere, level and lead:
    model (years x members) minus ERA5 on the same valid dates, where "ERA5" is the reference the maps are drawn
    against - strat_maps.clim_at (1991-2020 day-of-year climatology moved along its significant trend to each valid
    date) - so subtracting this from a forecast anomaly removes exactly the model's systematic departure from that
    reference at that lead. Zonal mean by latitude: the drift is dominated by the vortex's slow spin-up and the polar
    cap's radiative drift, both zonally symmetric, and 80 samples per lead carry a zonal mean far better than a
    gridpoint (the same choice as the GEFS drift correction). The year-to-year spread gives the standard error.
    -> data/geps_s2s/mapbias_{nh,sh}.json"""
    import xarray as xr
    import strat_maps as sm
    c = xr.open_dataset(sm.CLIM).load()
    tr = xr.open_dataset(sm.TREND).load() if sm.TREND.exists() else None
    for hemi in a.hemi.split(","):
        dest = OUT / f"mapbias_{hemi}.json"
        table = json.loads(dest.read_text()) if dest.exists() else {
            "source": "ECMWF S2S eccc (GEPS reforecast, control + 3 perturbed, 2001-2020) minus ERA5 1991-2020 "
                      "day-of-year climatology at the valid date, trend-referenced as strat_maps.clim_at", "starts": {}}
        for f in sorted(OUT.glob(f"hc_*_zt_{hemi}.grib")):
            d = f.name.split("_")[1]
            if d in table["starts"] and not a.force:
                continue
            zm, hd, steps, lat = {}, None, None, None
            for dt_ in ("cf", "pf"):
                ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": "", "filter_by_keys": {"dataType": dt_}})
                for name, key in (("t", "t"), ("gh", "z")):
                    v = ds[name]
                    v = v.expand_dims(number=[0]) if "number" not in v.dims else v
                    for lev in (10, 100):
                        m = v.sel(isobaricInhPa=lev).mean("longitude").transpose("time", "number", "step", "latitude")
                        zm.setdefault(f"{key}{lev}", []).append(m.values.astype("float64"))
                hd = pd.DatetimeIndex(ds.time.values); steps = (ds.step.values / np.timedelta64(1, "h")).astype(int)
                lat = ds.latitude.values.astype(float)
            zm = {k: np.concatenate(v, axis=1) for k, v in zm.items()}            # [hdate, member, step, lat]
            if a.verbose:
                print(f"  {d} {hemi}: t10 range {np.nanmin(zm['t10']):.0f}..{np.nanmax(zm['t10']):.0f} K, "
                      f"z10 {np.nanmin(zm['z10']):.0f}..{np.nanmax(zm['z10']):.0f} m (units check)", flush=True)
            obs = {k: np.full((len(hd), len(steps), len(lat)), np.nan) for k in zm}
            clat = c.lat.values[::-1]
            for i, h in enumerate(hd):
                for j, st in enumerate(steps):
                    valid = pd.Timestamp(h) + pd.Timedelta(hours=int(st))
                    for key, ckey in (("t", "T"), ("z", "Z")):
                        for lev in (10, 100):
                            ref, _ = sm.clim_at(c, ckey, lev, valid, tr)
                            obs[f"{key}{lev}"][i, j] = np.interp(lat, clat, np.asarray(ref.mean("lon").values)[::-1])
            ent = {"n_years": int(len(hd)), "n_members": int(zm["t10"].shape[1]), "step_h": steps.tolist(),
                   "lat": np.round(lat, 2).tolist()}
            for k in zm:
                diff = np.nanmean(zm[k], axis=1) - obs[k]                        # [hdate, step, lat]: member mean - ERA5
                ent[k] = np.round(np.nanmean(diff, axis=0), 2).tolist()
                ent[k + "_se"] = np.round(np.nanstd(diff, axis=0, ddof=1) / np.sqrt(len(hd)), 2).tolist()
            table["starts"][d] = ent
            cap = lat >= 65 if hemi == "nh" else lat <= -65
            w = np.cos(np.deg2rad(lat[cap]))
            b = np.asarray(ent["t10"]); bz = np.asarray(ent["z10"])
            print(f"{d} {hemi}: 10 hPa cap drift T day 1 {(b[0, cap] * w).sum() / w.sum():+.1f} K, day 15 "
                  f"{(b[14, cap] * w).sum() / w.sum():+.1f}, day 35 {(b[-1, cap] * w).sum() / w.sum():+.1f}; Z day 35 "
                  f"{(bz[-1, cap] * w).sum() / w.sum():+.0f} m", flush=True)
            dest.write_text(json.dumps(table, separators=(",", ":")))
        print(f"wrote {dest} ({len(table['starts'])} start dates)")


def capdrift(a) -> None:
    """Polar-cap (65-90N, cos-weighted) HEIGHT drift per level and lead, for the forecast dripping paint
    (forecast_drip.py): per start date, the mean over years x members of (reforecast cap height - the MERRA-2
    reference mean at the valid date), where the reference is build_drip_ref's harmonics + trend fit, so the drift
    already includes GEPS's own analysis offset from MERRA-2 and the trend is handled year by year. Levels come from
    whichever parts exist for a start date: zt_nh (10, 100 hPa) and ghcap (50 ... 1000 hPa).
    -> data/geps_s2s/capdrift_nh.json"""
    import xarray as xr
    from build_drip_ref import Ref
    R = Ref()
    dest = OUT / "capdrift_nh.json"
    table = json.loads(dest.read_text()) if dest.exists() else {
        "source": "ECMWF S2S eccc (GEPS reforecast, control + 3 perturbed, 2001-2020) polar-cap height minus the "
                  "MERRA-2 reference mean (build_drip_ref) at the valid date", "starts": {}}
    starts = sorted({f.name.split("_")[1] for part in ("zt_nh", "ghcap") for f in OUT.glob(f"hc_*_{part}.grib")})
    for d in starts:
        ent = table["starts"].get(d, {"levels": {}})
        for part in ("zt_nh", "ghcap"):
            f = OUT / f"hc_{d}_{part}.grib"
            if not f.exists() or (part in ent.get("parts", []) and not a.force):
                continue
            caps, hd, steps = [], None, None
            for dt_ in ("cf", "pf"):
                ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": "", "filter_by_keys": {"dataType": dt_}})
                v = ds["gh"]
                v = v.expand_dims(number=[0]) if "number" not in v.dims else v
                v = v.where(v.latitude >= 65, drop=True)
                w = np.cos(np.deg2rad(v.latitude))
                cap = (v.mean("longitude") * w).sum("latitude") / w.sum()
                caps.append(cap.transpose("time", "number", "step", "isobaricInhPa").values)
                hd = pd.DatetimeIndex(ds.time.values); steps = (ds.step.values / np.timedelta64(1, "h")).astype(int)
                levs = [float(x) for x in np.atleast_1d(ds.isobaricInhPa.values)]
            C = np.concatenate(caps, axis=1)                                   # [hdate, member, step, level]
            for k, L in enumerate(levs):
                dif = np.full((len(hd), len(steps)), np.nan)
                for i, h in enumerate(hd):
                    valid = pd.DatetimeIndex(h + pd.to_timedelta(steps, "h")).normalize()
                    dif[i] = np.nanmean(C[i, :, :, k], axis=0) - R.mean(valid, L)
                ent["levels"][str(L)] = {"drift": np.round(np.nanmean(dif, 0), 2).tolist(),
                                         "se": np.round(np.nanstd(dif, 0, ddof=1) / np.sqrt(len(hd)), 2).tolist()}
            ent["step_h"] = steps.tolist(); ent.setdefault("parts", []).append(part)
            ent["parts"] = sorted(set(ent["parts"]))
            print(f"{d} {part}: " + ", ".join(f"{L:.0f} hPa d10 {ent['levels'][str(L)]['drift'][9]:+.0f} / d35 "
                                                  f"{ent['levels'][str(L)]['drift'][-1]:+.0f} m" for L in levs), flush=True)
        table["starts"][d] = ent
    dest.write_text(json.dumps(table, separators=(",", ":")))
    print(f"wrote {dest} ({len(table['starts'])} start dates)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "bias", "mapbias", "capdrift"])
    ap.add_argument("--hemi", default="nh,sh"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--from", dest="frm", default="08-14")
    ap.add_argument("--to", default="12-31")
    ap.add_argument("--parts", default="ut", help="comma list of PARTS to fetch: ut (vortex panel), zt_nh, zt_sh (maps)")
    a = ap.parse_args()
    {"fetch": fetch, "bias": bias, "mapbias": mapbias, "capdrift": capdrift}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
