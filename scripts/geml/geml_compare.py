#!/usr/bin/env python3
"""ECCC GDPS-GEML (AI emulator) vs operational GDPS — NH 500 hPa loop.

GEML (Global Environmental eMuLator) is ECCC's data-driven AI weather
model, on Datamart since March 2026; the GDPS-SN hybrid launched spring
2026 spectrally nudges the operational GEM toward GEML's large scales.
This animator puts the raw AI emulator side by side with the operational
GDPS it nudges: 0.25° GEML vs 0.15° GDPS, 500 hPa height, NH polar view,
every 6 h through 240 h.

Reuses the AIFS animator machinery: same projection/warp staging, the
same Julia/CairoMakie renderer (panel titles via the spec), the same
publish path. Frames: assets/sst/anim/geml_z500 + manifest.

    python scripts/geml/geml_compare.py [--date YYYYMMDD --time 00]
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "verify"))
import aifs_compare_anim as A                     # projections, warp, julia, publish helpers

CACHE = HERE / "data" / "cache"
ANIM = REPO / "assets" / "sst" / "anim" / "geml_z500"
MANIFEST = REPO / "assets" / "sst" / "anim" / "geml_z500_manifest.json"
STEPS = list(range(0, 241, 6))
G = 9.80665

DDGC = "https://dd.weather.gc.ca/today"
MODELS = {
    "geml": ("model_gdps-geml/25km", "GDPS-GEML_Geopotential_IsbL-0500", "LatLon0.25"),
    "gdps": ("model_gdps/15km", "GDPS_GeopotentialHeight_IsbL-0500", "LatLon0.15"),
}


def fetch(date: str, hh: str) -> dict:
    """Download both models' z500 GRIBs for all steps; returns {(m, step): path}."""
    out = {}
    cyc = CACHE / f"{date}{hh}"
    cyc.mkdir(parents=True, exist_ok=True)
    for m, (root, var, grid) in MODELS.items():
        for s in STEPS:
            # GEML names hour 0 "PT0H" (unpadded); GDPS pads to "PT000H"
            step_tag = "PT0H" if (m == "geml" and s == 0) else f"PT{s:03d}H"
            fn = f"{date}T{hh}Z_MSC_{var}_{grid}_{step_tag}.grib2"
            p = cyc / fn
            if not p.exists() or p.stat().st_size < 10000:
                url = f"{DDGC}/{root}/{hh}/{s:03d}/{fn}"
                try:
                    with urllib.request.urlopen(url, timeout=120) as r:
                        tmp = p.with_suffix(".part")
                        tmp.write_bytes(r.read())
                        tmp.replace(p)
                except Exception as e:                      # noqa: BLE001
                    print(f"{m} h{s:03d}: fetch failed ({str(e)[:60]})",
                          file=sys.stderr)
                    return {}
            out[(m, s)] = p
        print(f"{m}: {len(STEPS)} steps cached", flush=True)
    # keep only this cycle + the previous one
    for old in sorted(CACHE.glob("[0-9]*"))[:-2]:
        shutil.rmtree(old, ignore_errors=True)
    return out


def _read_z500_dam(path: Path):
    """(field_dam, glat, glon) from a single-message GRIB."""
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]]
    v = da.values.astype(np.float32)
    if np.nanmean(v) > 20000:          # geopotential (m2 s-2) -> gpm
        v = v / G
    lat = da.latitude.values
    lon = da.longitude.values % 360
    if lon[0] > lon[-1]:               # keep monotonic 0..360 start
        roll = int(np.argmin(lon))
        lon = np.roll(lon, -roll); v = np.roll(v, -roll, axis=1)
    if lat[0] < lat[-1]:               # warp indices expect descending latitude
        lat = lat[::-1]; v = v[::-1]
    return v / 10.0, lat, lon          # dam


def main():
    ap = argparse.ArgumentParser()
    now = pd.Timestamp.utcnow()
    ap.add_argument("--date", default=now.strftime("%Y%m%d"))
    ap.add_argument("--time", default="00", choices=("00", "12"))
    args = ap.parse_args()
    date, hh = args.date, args.time
    base = pd.Timestamp(f"{date} {hh}:00")

    paths = fetch(date, hh)
    if not paths:
        print("cycle incomplete on Datamart; try the earlier cycle", file=sys.stderr)
        return 1

    t0 = time.time()
    nps = A._projections()[1]
    grids = {}
    if A.STAGE.exists():
        shutil.rmtree(A.STAGE)
    A.STAGE.mkdir(parents=True)

    frames = []
    for i, s in enumerate(STEPS):
        d = {}
        for m, pre in (("geml", "s_"), ("gdps", "e_")):
            v, glat, glon = _read_z500_dam(paths[(m, s)])
            if m not in grids:
                grids[m] = A._proj_grid(nps, [-179.9, 180, 20, 90], 560, glat, glon)
            d[pre + "z5"] = A._warp(v, grids[m], smooth=1.2)
        np.savez(A.STAGE / f"z5{i:02d}.npz", **d)
        valid = base + pd.Timedelta(hours=s)
        frames.append(dict(
            id=f"z5{i:02d}", npz=f"z5{i:02d}.npz", idx=i, step=s,
            date=f"{valid:%Y-%m-%d}", label=f"h{s:03d} · {valid:%b %d %HZ}",
            title=(f"500 hPa geopotential height · NH — hour {s}"
                   f" · valid {valid:%a %b %d %HZ} · init {base:%Y-%m-%d %HZ}")))
    g = grids["geml"]
    ovnh = A._overlays_nh(nps, (g["x0"], g["x1"], g["y0"], g["y1"]))
    shutil.copy(ovnh, A.STAGE / "ov_nh.npz")
    spec = dict(staging=str(A.STAGE), loops=[
        dict(kind="z500", overlays="ov_nh.npz",
             titles=["GDPS-GEML (AI emulator, 28 km)", "GDPS (operational GEM, 15 km)"],
             xlim=[g["x0"], g["x1"]], ylim=[g["y0"], g["y1"]], frames=frames)])
    sp = A.STAGE / "spec.json"
    sp.write_text(json.dumps(spec))
    print(f"staged {len(frames)} frames in {time.time() - t0:.1f}s", flush=True)

    # A.render_julia's success bool assumes the AIFS frame count; check ours
    A.render_julia(sp)
    n_png = len(list(A.STAGE.glob("z5*.png")))
    if n_png != len(frames):
        print(f"julia render incomplete ({n_png}/{len(frames)})", file=sys.stderr)
        return 1

    from PIL import Image
    ANIM.mkdir(parents=True, exist_ok=True)
    for old in ANIM.glob("F*.webp"):
        old.unlink()
    out_frames = []
    for fr in frames:
        im = Image.open(A.STAGE / (fr["id"] + ".png"))
        fn = f"F{fr['idx']:02d}.webp"
        im.save(ANIM / fn, quality=92, method=6)
        out_frames.append({"idx": fr["idx"], "file": fn, "date": fr["date"],
                           "label": fr["label"]})
    MANIFEST.write_text(json.dumps(
        {"ver": int(time.time()), "days": len(out_frames),
         "regions": {"geml_z500": {
             "label": "GDPS-GEML (AI) vs operational GDPS — 500 hPa height (NH)",
             "n_frames": len(out_frames), "frames": out_frames}}}))
    print(f"published {len(out_frames)} -> {ANIM.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
