#!/usr/bin/env python3
"""Ensemble-mean MSLP + 10 m wind map animator - Pacific: the AIFS-ENS + IFS-ENS blend and each model alone.

Pressure (mb) is linear and the 10-m wind speed/barbs are drawn from the ensemble-mean
10u/10v, so we just average the members of each model and combine the two model means
weighted by member count. Daily frames F024..F360 over the tropical/subtropical Pacific,
written as webp + a manifest for the sst_anim.html iframe (region "mslp_wind").

    python src/mslp_wind_anim.py --date 20260609 --time 00 \
        --anim-dir assets/sst/anim/mslp_wind --manifest assets/sst/anim/mslp_wind_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf

EXTENT = (100, 280, -30, 45)                    # lon0, lon1 (0..360), lat0, lat1
# key ENSO/WWB monitoring stations (lon 0..360, lat) shown on the map for reference
STATIONS = {"Darwin (YPDN)": (130.9, -12.4), "Tarawa (NGTA)": (173.0, 1.4),
            "Christmas I. (PLCH)": (202.5, 2.0), "Tahiti (NTAA)": (210.4, -17.5)}
DAILY_STEPS = list(range(24, 361, 24))          # days 1..15
MS2KT = 1.94384
PLEVS = np.arange(900, 1064, 4)                 # MSLP contour levels (hPa)
# 10-m wind-speed shading (kt) — starts at 5 kt, finer steps through the tropical range
WLEV = [5, 8, 11, 14, 17, 20, 23, 27, 31, 36, 41, 47, 53, 60]
WCOLS = ["#dcefff", "#bfe0f5", "#9ccde9", "#73aedb", "#4a86c5", "#3559a8",
         "#5a3f9c", "#8036a0", "#a82f9c", "#cf2592", "#e8408a", "#f57247", "#f59f00"]
WCMAP = ListedColormap(WCOLS); WCMAP.set_under("#ffffff00"); WCMAP.set_over("#d97706")
WNORM = BoundaryNorm(WLEV, WCMAP.N)


def _open(paths, short):
    """One or two cf/pf GRIB paths → (number, step, lat, lon), members 0..N, 0..360°E,
    ascending lat, restricted to DAILY_STEPS. Lazy."""
    parts = []
    for p in paths:
        ds = xr.open_dataset(p, engine="cfgrib", chunks={"number": 1},
                             backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": short}})
        da = ds[[v for v in ds.data_vars][0]]
        if "number" not in da.dims:
            da = da.expand_dims("number")
        parts.append(da)
    da = xr.concat(parts, dim="number").assign_coords(
        number=lambda d: np.arange(d.sizes["number"])).sortby("latitude")
    if float(da.longitude.min()) < 0:
        da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")
    hrs = (da.step / np.timedelta64(1, "h")).values.astype(int)
    return da.isel(step=np.isin(hrs, DAILY_STEPS))


def model_means(cyc, short):
    """(AIFS-ENS mean, n_aifs, IFS-ENS mean on the AIFS grid or None, n_ifs) for one surface field. The blend is
    built by the caller; 2026-10-02 the page shows all three (blend, AIFS-ENS, IFS-ENS) - the old single "super-ensemble"
    loop silently fell back to AIFS-ENS when IFS day 15 was not on the mirror yet, under a title naming both models."""
    sp = lambda m, t: ecmwf.sfc_path(cyc, m, t, short)
    aifs = _open([sp("aifs-ens", "cf"), sp("aifs-ens", "pf")], short)
    am, na = aifs.mean("number").load(), aifs.sizes["number"]
    try:
        ifs = _open([sp("ifs", "pf")], short)
        im, ni = ifs.mean("number"), ifs.sizes["number"]
        im = im.interp(latitude=am.latitude, longitude=am.longitude)
        steps = np.intersect1d(am.step.values, im.step.values)
        return am.sel(step=steps), na, im.sel(step=steps).load(), ni
    except Exception as e:
        print(f"  IFS {short} unavailable ({repr(e)[:60]})", flush=True)
        return am, na, None, 0


# Single runs (2026-10-03, user: "add the aifs-ens member 0, aifs single, IFS op, and GDPS" - for THIS plot). Each gets its
# own loop on the same map and scales; drawn on the AIFS-ENS 0.25-deg grid of the subdomain.
SINGLE = {"aifs0": "AIFS-ENS member 0", "aifs1": "AIFS single", "hres": "IFS HRES", "gdps": "GDPS"}
GDPS_URL = "https://dd.weather.gc.ca/{d}/WXO-DD/model_gdps/15km/{c}/{L:03d}/{d}T{c}Z_MSC_GDPS_{v}_LatLon0.15_PT{L:03d}H.grib2"


def _grib_field(blob):
    """One GRIB message (bytes) -> DataArray(latitude asc, longitude 0..360)."""
    import eccodes as ec
    h = ec.codes_new_from_message(bytes(blob))
    ni, nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
    la0, la1 = ec.codes_get(h, "latitudeOfFirstGridPointInDegrees"), ec.codes_get(h, "latitudeOfLastGridPointInDegrees")
    lo0, di = ec.codes_get(h, "longitudeOfFirstGridPointInDegrees"), ec.codes_get(h, "iDirectionIncrementInDegrees")
    v = ec.codes_get_values(h).reshape(nj, ni); ec.codes_release(h)
    da = xr.DataArray(v, dims=("latitude", "longitude"),
                      coords={"latitude": np.linspace(la0, la1, nj), "longitude": (lo0 + di * np.arange(ni)) % 360})
    return da.sortby("latitude").sortby("longitude")


def single_fields(key, cyc, lat, lon):
    """{msl (Pa), 10u, 10v}: DataArray (step, latitude, longitude) on the target lat/lon, daily steps. ECMWF single runs
    are byte-ranged from the Google Cloud mirror only (oper fc); GDPS (to day 10) comes from the ECCC Datamart."""
    if key == "aifs0":
        sp = lambda t, f: ecmwf.sfc_path(cyc, "aifs-ens", t, f)
        return {f: _open([sp("cf", f)], f).isel(number=0).interp(latitude=lat, longitude=lon) for f in ("msl", "10u", "10v")}
    out = {f: [] for f in ("msl", "10u", "10v")}; steps = []
    if key in ("aifs1", "hres"):
        import rangefetch as rf
        model = "aifs-single" if key == "aifs1" else "ifs"
        for st in DAILY_STEPS:
            try:
                idx = rf.fetch_index(cyc.date, cyc.time, model, st, "fc", stream="oper")
            except Exception as e:                                     # noqa: BLE001
                print(f"  {key} +{st}h: not on the Google mirror ({str(e)[:60]})", flush=True); break
            want = sorted([e for e in idx if e.get("levtype") == "sfc" and e.get("param") in out], key=lambda e: e["_offset"])
            if len(want) < 3:
                break
            blob = rf.fetch_ranges(rf.path_for(cyc.date, cyc.time, model, st, "fc", stream="oper") + ".grib2", rf.coalesce(want))
            pos = 0
            for e in want:
                out[e["param"]].append(_grib_field(blob[pos:pos + e["_length"]]).interp(latitude=lat, longitude=lon)); pos += e["_length"]
            steps.append(st)
    elif key == "gdps":
        import requests
        names = {"msl": "Pressure_MSL", "10u": "WindU_AGL-10m", "10v": "WindV_AGL-10m"}
        for st in [s for s in DAILY_STEPS if s <= 240]:
            got = {}
            for f, nm in names.items():
                r = requests.get(GDPS_URL.format(d=cyc.date, c=cyc.time, L=st, v=nm), timeout=120,
                                 headers={"User-Agent": "scorvec-enso/1.0"})
                if r.status_code != 200:
                    break
                got[f] = _grib_field(r.content).interp(latitude=lat, longitude=lon)
            if len(got) < 3:
                print(f"  gdps +{st}h: not on the Datamart", flush=True); break
            for f in out:
                out[f].append(got[f])
            steps.append(st)
    if not steps:
        raise RuntimeError(f"{key}: no fields")
    td = np.array(steps, "timedelta64[h]").astype("timedelta64[ns]")
    res = {f: xr.concat(v, "step").assign_coords(step=td) for f, v in out.items()}
    if float(res["msl"].median()) < 2000:                                 # GDPS publishes MSLP in hPa? keep Pa throughout
        res["msl"] = res["msl"] * 100.0
    return res


def wait_for_ifs(cyc, minutes):
    """IFS-ENS day 15 reaches the Google mirror ~08:50/20:55Z, often after this job starts: wait (Google only - never
    another mirror) up to `minutes`, re-trying the surface batch every 60 s."""
    import time
    t_end = time.time() + 60 * minutes
    while True:
        try:
            ecmwf.sfc_path(cyc, "ifs", "pf", "msl"); return True
        except Exception as e:
            if time.time() > t_end:
                print(f"  IFS-ENS still not published after {minutes} min ({repr(e)[:60]}); AIFS-ENS only", flush=True)
                return False
            time.sleep(60)


def _hl(p2d, lat, lon, ax, proj):
    """Mark significant pressure highs (H) and lows (L) on the smoothed field."""
    s = gaussian_filter(p2d, 4, mode=("nearest", "wrap"))
    for filt, op, col, sym in ((minimum_filter, np.less_equal, "#c0152f", "L"),
                               (maximum_filter, np.greater_equal, "#1f4fb0", "H")):
        ext = filt(s, size=28, mode=("nearest", "wrap"))
        ys, xs = np.where(op(s, ext))
        seen = []
        for y, x in zip(ys, xs):
            if any(abs(y - yy) < 18 and abs(x - xx) < 18 for yy, xx in seen):
                continue
            seen.append((y, x))
            ax.text(lon[x], lat[y], sym, color=col, fontsize=12, fontweight="bold",
                    ha="center", va="center", transform=proj, clip_on=True)
            ax.text(lon[x], lat[y] - 2.4, f"{s[y, x]:.0f}", color=col, fontsize=6.5,
                    ha="center", va="top", transform=proj, clip_on=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--anim-dir", default="assets/sst/anim/mslp_wind")
    ap.add_argument("--manifest", default="assets/sst/anim/mslp_wind_manifest.json")
    ap.add_argument("--wait-ifs", type=float, default=0, help="minutes to wait for IFS-ENS day 15 on the Google mirror")
    ap.add_argument("--single", default="aifs0,aifs1,hres,gdps", help="single runs to add as their own loops ('' = none)")
    args = ap.parse_args()
    cyc = ecmwf.Cycle(args.date, args.time)
    init = np.datetime64(f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:8]}T{args.time}:00")

    if args.wait_ifs > 0: wait_for_ifs(cyc, args.wait_ifs)
    F = {k: model_means(cyc, k) for k in ("msl", "10u", "10v")}
    na, ni = F["msl"][1], F["msl"][3]
    have_ifs = all(F[k][2] is not None for k in F)
    variants = []                                          # (region, dir suffix, label for the title, msl, u, v)
    if have_ifs:
        bl = {k: (na * F[k][0] + ni * F[k][2].values) / (na + ni) for k in F}
        variants.append(("mslp_wind", "", f"AIFS-ENS + IFS-ENS ({na} + {ni} members)", bl["msl"], bl["10u"], bl["10v"]))
    else:
        variants.append(("mslp_wind", "", f"AIFS-ENS only ({na} members; IFS-ENS not yet published for this run)",
                         F["msl"][0], F["10u"][0], F["10v"][0]))
    variants.append(("mslp_wind_aifs", "_aifs", f"AIFS-ENS ({na} members)", F["msl"][0], F["10u"][0], F["10v"][0]))
    if have_ifs:
        variants.append(("mslp_wind_ifs", "_ifs", f"IFS-ENS ({ni} members)", F["msl"][2], F["10u"][2], F["10v"][2]))
    sub = dict(latitude=slice(EXTENT[2] - 1, EXTENT[3] + 1), longitude=slice(EXTENT[0] - 1, EXTENT[1] + 1))
    glat, glon = F["msl"][0].sel(**sub).latitude, F["msl"][0].sel(**sub).longitude
    for key in [k for k in args.single.split(",") if k]:
        try:
            sf = single_fields(key, cyc, glat, glon)
            n = sf["msl"].sizes["step"]
            variants.append((f"mslp_wind_{key}", f"_{key}", f"{SINGLE[key]}, to day {n}", sf["msl"],
                             sf["10u"], sf["10v"]))
        except Exception as e:                                             # noqa: BLE001
            print(f"  single run {key} skipped: {repr(e)[:100]}", flush=True)
    regions = {}
    for region, suf, who, msl, u10, v10 in variants:
        msl = msl / 100.0                                  # Pa -> hPa
        entries = render(args, init, region, who, msl, u10, v10, Path(args.anim_dir + suf),
                         kind="Ensemble-mean" if region in ("mslp_wind", "mslp_wind_aifs", "mslp_wind_ifs") else "Single run:")
        regions[region] = {"label": {"mslp_wind": "Blend (AIFS-ENS + IFS-ENS)", "mslp_wind_aifs": "AIFS-ENS only",
                                     "mslp_wind_ifs": "IFS-ENS only"}.get(region, SINGLE.get(region.split("_")[-1], region)),
                           "frames": entries}
        print(f"  {region}: {len(entries)} frames ({who})", flush=True)
    mani = {"ver": args.date + args.time, "regions": regions}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(mani))
    print(f"wrote {len(regions)} loops + {args.manifest}", flush=True)
    return 0


def render(args, init, region, who, msl, u10, v10, anim, kind="Ensemble-mean"):
    la0, la1 = EXTENT[2], EXTENT[3]; lo0, lo1 = EXTENT[0], EXTENT[1]
    sub = dict(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
    msl = msl.sel(**sub); u10 = u10.sel(**sub); v10 = v10.sel(**sub)
    lat = msl.latitude.values; lon = msl.longitude.values
    spd = np.hypot(u10.values, v10.values) * MS2KT       # (step, lat, lon) kt
    st = (lat[1] - lat[0]); bstride = max(1, int(round(3.5 / abs(st))))   # barbs ~every 3.5°

    proj = ccrs.PlateCarree(central_longitude=180)
    anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"): old.unlink()          # a fresh set per run (a shorter run must not leave stale frames)
    entries = []
    steps_h = (msl.step / np.timedelta64(1, "h")).values.astype(int)
    for k, h in enumerate(steps_h):
        valid = init + np.timedelta64(int(h), "h")
        fig = plt.figure(figsize=(12.6, 6.2))
        ax = plt.axes(projection=proj)
        ax.set_extent([lo0, lo1, la0, la1], crs=ccrs.PlateCarree())
        cf = ax.contourf(lon, lat, spd[k], levels=WLEV, cmap=WCMAP, norm=WNORM,
                         extend="both", transform=ccrs.PlateCarree())
        p = msl.isel(step=k).values
        cs = ax.contour(lon, lat, gaussian_filter(p, 1.2, mode=("nearest", "wrap")),
                        levels=PLEVS, colors="#333", linewidths=0.6, transform=ccrs.PlateCarree())
        ax.clabel(cs, inline=True, fontsize=6, fmt="%d")
        ax.barbs(lon[::bstride], lat[::bstride],
                 u10.values[k, ::bstride, ::bstride] * MS2KT,
                 v10.values[k, ::bstride, ::bstride] * MS2KT,
                 length=4.2, linewidth=0.4, color="#222", transform=ccrs.PlateCarree())
        _hl(p, lat, lon, ax, ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="0.05", linewidth=1.1, zorder=4)
        ax.coastlines(linewidth=1.1, color="0.05", zorder=4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.4", zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.45", alpha=0.5,
                          linestyle=(0, (3, 3)), zorder=3)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(list(range(-180, 181, 20)))
        gl.ylocator = mticker.FixedLocator(list(range(-30, 46, 15)))
        gl.xlabel_style = gl.ylabel_style = {"size": 6, "color": "0.3"}
        for name, (slon, slat) in STATIONS.items():
            ax.plot(slon, slat, marker="o", ms=4.5, mfc="#ffd400", mec="k", mew=0.7,
                    transform=ccrs.PlateCarree(), zorder=7)
            ax.text(slon, slat + 1.6, name, fontsize=6.2, fontweight="bold", ha="center",
                    va="bottom", color="k", transform=ccrs.PlateCarree(), zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
        cax = fig.add_axes([0.13, 0.06, 0.74, 0.02])
        fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both").set_label(
            "10 m wind speed (kt)", fontsize=8)
        cax.tick_params(labelsize=7)
        ax.set_title(f"{kind} MSLP (mb) + 10 m wind  ·  {who}"
                     f"\ninit {str(init)[:13]}Z  ·  "
                     f"F{int(h):03d} valid {str(valid)[:13]}Z", fontsize=10, loc="left")
        fp = anim / f"F{k:02d}.webp"
        fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.10)
        fig.savefig(fp, dpi=104); plt.close(fig)
        entries.append({"idx": k, "file": fp.name,
                        "date": str(valid)[:10], "label": f"F{int(h):03d} · {str(valid)[:13]}Z"})
    return entries


if __name__ == "__main__":
    raise SystemExit(main())
