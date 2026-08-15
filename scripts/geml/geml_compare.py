#!/usr/bin/env python3
"""ECCC GDPS-GEML (AI emulator) vs operational GDPS — loop suite.

Five loops per cycle (00Z / 12Z), every 6 h through 240 h:
  * geml_z500    — NH polar, 3 panels: GEML | GDPS | GEML − GDPS
  * geml_t2m     — NH polar, 3 panels: 2 m temperature + difference
  * geml_syn_na / _eu / _eas — regional synoptic pairs: MSLP + 1000-500
    thickness on both panels; the GDPS panel is shaded by 6-h precip
    BROKEN INTO TYPE (rain / snow / freezing rain / ice pellets, from the
    per-type accumulations GEM carries). GEML publishes no precipitation
    — its panel shows the dynamics only.

Machinery shared with the AIFS animator (projections, NN warp staging,
Julia/CairoMakie rendering, webp publish). GEML 0.25°, GDPS 0.15°, both
warped to common projected canvases.

    python scripts/geml/geml_compare.py [--date YYYYMMDD --time {00,12}]

Without arguments the newest expected cycle is used (00Z before ~17 UTC,
else 12Z), falling back one cycle if Datamart is incomplete.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "verify"))
import aifs_compare_anim as A                   # projections, warp, julia driver

CACHE = HERE / "data" / "cache"
OVCACHE = HERE / "data"
ANIMROOT = REPO / "assets" / "sst" / "anim"
STEPS = list(range(0, 241, 6))
G = 9.80665
DDGC = "https://dd.weather.gc.ca/today"

# (model, var-file-token, grid token); GEML names hour 0 "PT0H" (unpadded)
GEML = "model_gdps-geml/25km"
GDPS = "model_gdps/15km"
FILES = {
    ("geml", "z500"): (GEML, "GDPS-GEML_Geopotential_IsbL-0500", "LatLon0.25"),
    ("geml", "z1000"): (GEML, "GDPS-GEML_Geopotential_IsbL-1000", "LatLon0.25"),
    ("geml", "msl"): (GEML, "GDPS-GEML_Pressure_MSL", "LatLon0.25"),
    ("geml", "t2m"): (GEML, "GDPS-GEML_AirTemp_AGL-2m", "LatLon0.25"),
    ("gdps", "z500"): (GDPS, "GDPS_GeopotentialHeight_IsbL-0500", "LatLon0.15"),
    ("gdps", "z1000"): (GDPS, "GDPS_GeopotentialHeight_IsbL-1000", "LatLon0.15"),
    ("gdps", "msl"): (GDPS, "GDPS_Pressure_MSL", "LatLon0.15"),
    ("gdps", "t2m"): (GDPS, "GDPS_AirTemp_AGL-2m", "LatLon0.15"),
    ("gdps", "rain"): (GDPS, "GDPS_Rain-Accum_Sfc", "LatLon0.15"),
    ("gdps", "snow"): (GDPS, "GDPS_Snow-Accum_Sfc", "LatLon0.15"),
    ("gdps", "frzr"): (GDPS, "GDPS_FreezingRain-Accum_Sfc", "LatLon0.15"),
    ("gdps", "icep"): (GDPS, "GDPS_IcePellets-Accum_Sfc", "LatLon0.15"),
}
PTYPES = ("rain", "snow", "frzr", "icep")
PT_LEVELS = [0.2, 1, 2.5, 5, 10, 20, 40]

REGIONS = {                                      # key -> (extent, nx, label)
    "na": ([-128, -63, 22, 55], 660, "North America"),
    "eu": ([-32, 45, 32, 72], 660, "Europe"),
    "eas": ([90, 165, 8, 55], 660, "East Asia"),
}


def _url(m, v, s, date, hh):
    root, var, grid = FILES[(m, v)]
    tag = "PT0H" if (m == "geml" and s == 0) else f"PT{s:03d}H"
    fn = f"{date}T{hh}Z_MSC_{var}_{grid}_{tag}.grib2"
    return f"{DDGC}/{root}/{hh}/{s:03d}/{fn}", fn


def fetch(date, hh):
    cyc = CACHE / f"{date}{hh}"
    cyc.mkdir(parents=True, exist_ok=True)
    # per-type accumulations have no hour-0 file (accum starts at 0 by definition)
    jobs = [(m, v, s) for (m, v) in FILES for s in STEPS
            if not (v in PTYPES and s == 0)]

    def get(job):
        m, v, s = job
        url, fn = _url(m, v, s, date, hh)
        p = cyc / fn
        if p.exists() and p.stat().st_size > 10000:
            return (m, v, s), p
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                tmp = p.with_suffix(".part")
                tmp.write_bytes(r.read())
                tmp.replace(p)
            return (m, v, s), p
        except Exception as e:                    # noqa: BLE001
            return (m, v, s), f"{str(e)[:50]}"

    out, bad = {}, []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for key, res in ex.map(get, jobs):
            if isinstance(res, Path):
                out[key] = res
            else:
                bad.append((key, res))
    for old in sorted(CACHE.glob("[0-9]*"))[:-2]:
        shutil.rmtree(old, ignore_errors=True)
    if bad:
        print(f"fetch: {len(bad)} missing (e.g. {bad[0]})", file=sys.stderr)
        return None
    print(f"fetched/cached {len(out)} GRIBs", flush=True)
    return out


def _read(path: Path):
    """(values, lat_desc, lon_0360) from a single-message GRIB."""
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]]
    v = da.values.astype(np.float32)
    lat, lon = da.latitude.values, da.longitude.values % 360
    if lon[0] > lon[-1]:
        roll = int(np.argmin(lon))
        lon = np.roll(lon, -roll); v = np.roll(v, -roll, axis=1)
    if lat[0] < lat[-1]:
        lat = lat[::-1]; v = v[::-1]
    ds.close()
    return v, lat, lon


def _overlays(key, proj, bbox, states=False):
    out = OVCACHE / f"ov_geml_{key}.npz"
    if out.exists():
        return out
    import cartopy.feature as cfeature
    feats = dict(
        coast=cfeature.NaturalEarthFeature("physical", "coastline", "50m"),
        borders=cfeature.NaturalEarthFeature("cultural",
                                             "admin_0_boundary_lines_land", "50m"),
        lakes=cfeature.NaturalEarthFeature("physical", "lakes", "50m"))
    if states:
        feats["states"] = cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces_lines", "50m")
    d = {}
    for k, f in feats.items():
        d[f"{k}_x"], d[f"{k}_y"] = A._project_lines(proj, f.geometries(), bbox)
    np.savez_compressed(out, **d)
    return out


def main():
    ap = argparse.ArgumentParser()
    now = pd.Timestamp.utcnow()
    auto_hh = "00" if now.hour < 17 else "12"
    ap.add_argument("--date", default=now.strftime("%Y%m%d"))
    ap.add_argument("--time", default=auto_hh, choices=("00", "12"))
    args = ap.parse_args()

    # requested cycle; a 12Z ask can fall back to today's 00Z (Datamart /today/
    # holds only the current day, so yesterday's cycles are unreachable)
    tries = [(args.date, args.time)]
    if args.time == "12":
        tries.append((args.date, "00"))
    paths = None
    for date, hh in tries:
        paths = fetch(date, hh)
        if paths:
            break
        print(f"cycle {date} {hh}Z incomplete; falling back", file=sys.stderr)
    if not paths:
        print("no complete cycle available; will retry next scheduled run",
              file=sys.stderr)
        return 1
    base = pd.Timestamp(f"{date} {hh}:00")

    t0 = time.time()
    import cartopy.crs as ccrs
    nps = A._projections()[1]
    projs = {"nh": (nps, [-179.9, 180, 20, 90], 560)}
    for k, (ext, nx, _lab) in REGIONS.items():
        clon = (ext[0] + ext[1]) / 2
        clat = (ext[2] + ext[3]) / 2
        projs[k] = (ccrs.LambertConformal(central_longitude=clon,
                                          central_latitude=clat), ext, nx)

    if A.STAGE.exists():
        shutil.rmtree(A.STAGE)
    A.STAGE.mkdir(parents=True)

    # grids per (region, model); overlays per region
    grids, ovs = {}, {}
    latlon = {}
    for m in ("geml", "gdps"):
        _, glat, glon = _read(paths[(m, "z500", 0)])
        latlon[m] = (glat, glon)
    for rk, (proj, ext, nx) in projs.items():
        for m in ("geml", "gdps"):
            grids[(rk, m)] = A._proj_grid(proj, ext, nx, *latlon[m])
        g = grids[(rk, "geml")]
        ovs[rk] = _overlays(rk, proj, (g["x0"], g["x1"], g["y0"], g["y1"]),
                            states=(rk == "na"))
        shutil.copy(ovs[rk], A.STAGE / f"ov_{rk}.npz")

    def fld(m, v, s):
        vv, _la, _lo = _read(paths[(m, v, s)])
        return vv

    def z_dam(m, v, s):
        # GEML publishes Geopotential (m2 s-2); GDPS GeopotentialHeight (gpm).
        # Explicit per-model — a magnitude test fails at 1000 hPa where mean
        # geopotential (~1100 m2 s-2) sits below any sane threshold.
        vv = fld(m, v, s)
        if m == "geml":
            vv = vv / G
        return vv / 10.0

    frames = {k: [] for k in ("z500", "t2m", "na", "eu", "eas")}
    prev_acc = {}
    for i, s in enumerate(STEPS):
        valid = base + pd.Timedelta(hours=s)
        lab = f"h{s:03d} · {valid:%b %d %HZ}"
        meta = dict(idx=i, step=s, date=f"{valid:%Y-%m-%d}", label=lab)
        tail = f"hour {s} · valid {valid:%a %b %d %HZ} · init {base:%Y-%m-%d %HZ}"

        gz = {m: z_dam(m, "z500", s) for m in ("geml", "gdps")}
        np.savez(A.STAGE / f"za{i:02d}.npz",
                 s_f=A._warp(gz["geml"], grids[("nh", "geml")], smooth=1.2),
                 e_f=A._warp(gz["gdps"], grids[("nh", "gdps")], smooth=1.2))
        frames["z500"].append(dict(meta, id=f"za{i:02d}", npz=f"za{i:02d}.npz",
                                   title=f"500 hPa height (dam) · NH — {tail}"))

        t2 = {m: fld(m, "t2m", s) - 273.15 for m in ("geml", "gdps")}
        np.savez(A.STAGE / f"ta{i:02d}.npz",
                 s_f=A._warp(t2["geml"], grids[("nh", "geml")], smooth=0.8),
                 e_f=A._warp(t2["gdps"], grids[("nh", "gdps")], smooth=0.8))
        frames["t2m"].append(dict(meta, id=f"ta{i:02d}", npz=f"ta{i:02d}.npz",
                                  title=f"2 m temperature (°C) · NH — {tail}"))

        msl = {m: fld(m, "msl", s) / 100.0 for m in ("geml", "gdps")}
        thk = {m: z_dam(m, "z500", s) - z_dam(m, "z1000", s)
               for m in ("geml", "gdps")}
        acc = ({p: np.zeros_like(msl["gdps"]) for p in PTYPES} if s == 0
               else {p: fld("gdps", p, s) for p in PTYPES})
        for rk in REGIONS:
            d = {}
            for m, pre in (("geml", "s_"), ("gdps", "e_")):
                d[pre + "msl"] = A._warp(msl[m], grids[(rk, m)], smooth=2.0)
                d[pre + "thk"] = A._warp(thk[m], grids[(rk, m)], smooth=2.0)
            for p in PTYPES:
                dp = (np.zeros_like(acc[p]) if s == 0
                      else np.clip(acc[p] - prev_acc[p], 0, None))
                d["e_" + p] = A._warp(dp, grids[(rk, "gdps")])
            np.savez(A.STAGE / f"sn_{rk}{i:02d}.npz", **d)
            frames[rk].append(dict(
                meta, id=f"sn_{rk}{i:02d}", npz=f"sn_{rk}{i:02d}.npz",
                title=(f"MSLP · 1000–500 thickness · 6-h precip by type "
                       f"(GDPS panel) · {REGIONS[rk][2]} — {tail}")))
        prev_acc = acc
        if i % 10 == 0:
            print(f"staged h{s:03d}", flush=True)

    def three(loop_frames, name, bounds, diff_bounds, units, cmap, over):
        g = grids[("nh", "geml")]
        return dict(kind="three", overlays="ov_nh.npz",
                    titles=[f"GEML (raw AI emulator) — {name}",
                            f"GDPS (operational hybrid, nudged to GEML) — {name}",
                            "GEML − hybrid GDPS"],
                    xlim=[g["x0"], g["x1"]], ylim=[g["y0"], g["y1"]],
                    bounds=bounds, diff_bounds=diff_bounds, units=units,
                    diff_label=f"GEML − hybrid ({units.split('(')[-1].rstrip(')')})",
                    cmap=cmap, over=over, frames=loop_frames)

    loops = [
        three(frames["z500"], "500 hPa height",
              list(np.arange(486.0, 601.0, 6.0)), list(np.arange(-20.0, 21.0, 2.0)),
              "500 hPa height (dam)", "turbo", "#7a0403"),
        three(frames["t2m"], "2 m temperature",
              list(np.arange(-40.0, 41.0, 4.0)), list(np.arange(-10.0, 11.0, 1.0)),
              "2 m temperature (°C)", "turbo", "#7a0403"),
    ]
    for rk in REGIONS:
        g = grids[(rk, "geml")]
        loops.append(dict(
            kind="ptype", overlays=f"ov_{rk}.npz",
            titles=["GEML (raw AI emulator — no precip output)",
                    "GDPS (operational hybrid) + precip type"],
            xlim=[g["x0"], g["x1"]], ylim=[g["y0"], g["y1"]],
            p_levels=PT_LEVELS, frames=frames[rk]))

    sp = A.STAGE / "spec.json"
    sp.write_text(json.dumps(dict(staging=str(A.STAGE), loops=loops)))
    print(f"staged {sum(len(v) for v in frames.values())} frames "
          f"in {time.time() - t0:.1f}s", flush=True)

    A.render_julia(sp)
    want = {"z500": "geml_z500", "t2m": "geml_t2m",
            "na": "geml_syn_na", "eu": "geml_syn_eu", "eas": "geml_syn_eas"}
    from PIL import Image
    ok_all = True
    for fk, name in want.items():
        pngs = [A.STAGE / (fr["id"] + ".png") for fr in frames[fk]]
        if not all(p.exists() for p in pngs):
            n = sum(p.exists() for p in pngs)
            print(f"{name}: only {n}/{len(pngs)} frames rendered — skipping publish",
                  file=sys.stderr)
            ok_all = False
            continue
        outdir = ANIMROOT / name
        outdir.mkdir(parents=True, exist_ok=True)
        for old in outdir.glob("F*.webp"):
            old.unlink()
        ofr = []
        for fr in frames[fk]:
            im = Image.open(A.STAGE / (fr["id"] + ".png"))
            fn = f"F{fr['idx']:02d}.webp"
            im.save(outdir / fn, quality=92, method=6)
            ofr.append({"idx": fr["idx"], "file": fn, "date": fr["date"],
                        "label": fr["label"]})
        lab = {"geml_z500": "GEML vs GDPS · 500 hPa height + difference (NH)",
               "geml_t2m": "GEML vs GDPS · 2 m temperature + difference (NH)",
               }.get(name, f"GEML vs GDPS · synoptic + precip type · "
                           f"{REGIONS.get(fk, ('', '', fk))[2]}")
        (ANIMROOT / f"{name}_manifest.json").write_text(json.dumps(
            {"ver": int(time.time()), "days": len(ofr),
             "regions": {name: {"label": lab, "n_frames": len(ofr),
                                "frames": ofr}}}))
        print(f"published {len(ofr)} -> {name}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
