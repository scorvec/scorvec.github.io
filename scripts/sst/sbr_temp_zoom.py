#!/usr/bin/env python3
"""Southeastern/southern Brazil coastal 2-m temperature zoom — 3 models.

High-detail afternoon (18Z ≈ 3 pm BRT) temperature + 10-m wind barbs for
the Rio / São Paulo / south coast corridor, next four days, from:

  row 1  CMC GDPS 15 km        (Datamart, native 0.15 deg)
  row 2  AIFS-ENS control      (ECMWF open data, 0.25 deg)
  row 3  IFS HRES              (ECMWF open data, 0.25 deg)

City dots carry the model's nearest-gridpoint temperature.  Built to
watch pre-frontal heat spikes on the coast.

    python scripts/sst/sbr_temp_zoom.py [--date 20260824]
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUTPNG = REPO / "brazil_hydro" / "sbr_temp_zoom.webp"

BOX = dict(lon0=-52.0, lon1=-40.0, lat0=-28.0, lat1=-19.5)
STEPS = [18, 42, 66, 90]                     # 18Z valids ≈ 3 pm BRT (default)
GDPS = ("https://dd.weather.gc.ca/{date}/WXO-DD/model_gdps/15km/00/{lead:03d}/"
        "{date}T00Z_MSC_GDPS_{var}_LatLon0.15_PT{lead:03d}H.grib2")
GDPS_VARS = {"t2": "AirTemp_AGL-2m", "u10": "WindU_AGL-10m",
             "v10": "WindV_AGL-10m"}
CITIES = [("Rio", -43.21, -22.87),          # nudged onto the urban land row —
          ("Bangu", -43.47, -22.87),        # the city-center cell is bay-mixed
          ("São Paulo", -46.63, -23.55),
          ("Santos", -46.33, -23.96), ("Campos", -41.33, -21.75),
          ("Curitiba", -49.27, -25.43), ("Florianópolis", -48.55, -27.60),
          ("B. Horizonte", -43.94, -19.92)]


def open_grib(path_or_bytes, **filt):
    import xarray as xr
    if isinstance(path_or_bytes, bytes):
        t = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
        t.write(path_or_bytes)
        t.close()
        path = t.name
    else:
        path = str(path_or_bytes)
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={"indexpath": "",
                                         "filter_by_keys": filt or None}
                         if filt else {"indexpath": ""}).load()
    if isinstance(path_or_bytes, bytes):
        with contextlib.suppress(OSError):
            os.remove(path)
    return ds


def slice_box(da):
    lon = da.longitude
    if float(lon.max()) > 180:
        da = da.assign_coords(longitude=(lon + 180) % 360 - 180)
    da = da.sortby("longitude").sortby("latitude")
    return da.sel(longitude=slice(BOX["lon0"], BOX["lon1"]),
                  latitude=slice(BOX["lat0"], BOX["lat1"]))


def fetch_gdps(date: str, steps: list[int] | None = None):
    """{step: {t2,u10,v10 (2d), lons, lats}}"""
    def one(args):
        var, lead = args
        url = GDPS.format(date=date, lead=lead, var=GDPS_VARS[var])
        req = urllib.request.Request(url, headers={"User-Agent": "scorvec/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return (var, lead, r.read())
        except Exception:                                   # noqa: BLE001
            return (var, lead, None)
    jobs = [(v, s) for v in GDPS_VARS for s in (steps or STEPS)]
    out: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for var, lead, blob in ex.map(one, jobs):
            if blob is None:
                continue
            ds = open_grib(blob)
            da = slice_box(ds[list(ds.data_vars)[0]])
            d = out.setdefault(lead, {})
            d[var] = da.values - (273.15 if var == "t2" else 0.0)
            d["lons"], d["lats"] = da.longitude.values, da.latitude.values
    return out


def fetch_ecmwf(model: str, date: str, stream: str, typ: str,
                steps: list[int] | None = None):
    from ecmwf.opendata import Client
    t = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    t.close()
    try:
        c = Client(source="ecmwf", model=model)
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            c.retrieve(date=date, time=0, stream=stream, type=typ,
                       param=["2t", "10u", "10v"], step=steps or STEPS,
                       target=t.name)
        out: dict[int, dict] = {}
        for sn, key in [("t2m", "t2"), ("u10", "u10"), ("v10", "v10")]:
            try:
                ds = open_grib(t.name, shortName={"t2m": "2t", "u10": "10u",
                                                  "v10": "10v"}[sn])
            except Exception:                               # noqa: BLE001
                continue
            da = slice_box(ds[list(ds.data_vars)[0]])
            steps_h = (np.atleast_1d(da.step.values) /
                       np.timedelta64(1, "h")).astype(int)
            for si, s in enumerate(steps_h):
                d = out.setdefault(int(s), {})
                v = da.isel(step=si).values if "step" in da.dims else da.values
                d[key] = v - (273.15 if key == "t2" else 0.0)
                d["lons"], d["lats"] = da.longitude.values, da.latitude.values
        return out
    finally:
        with contextlib.suppress(OSError):
            os.remove(t.name)


def panel(ax, d, name, tlev, cmap, nrm, states, label_left=None, title=None):
    """One model map panel (shared by grid and animation modes).

    pcolormesh with shading='nearest': every native grid cell rendered
    as-is — no contour binning, no spatial interpolation — so GDPS 15 km
    structure shows at full resolution and the ECMWF grids look exactly
    as coarse as they are."""
    lons, lats = d["lons"], d["lats"]
    ax.pcolormesh(lons, lats, d["t2"], cmap=cmap, norm=nrm,
                  shading="nearest", transform=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", lw=1.0, color="#111")
    ax.add_feature(states, edgecolor="#444", lw=0.5)
    if "u10" in d and "v10" in d:
        stride = max(1, int(round(0.9 / abs(lats[1] - lats[0]))))
        LO, LA = np.meshgrid(lons[::stride], lats[::stride])
        ax.barbs(LO, LA, d["u10"][::stride, ::stride] * 1.94384,
                 d["v10"][::stride, ::stride] * 1.94384,
                 length=4.6, linewidth=0.55, color="#222",
                 transform=ccrs.PlateCarree())
    for nm, x, y in CITIES:
        j = int(np.argmin(np.abs(lons - x)))
        i = int(np.argmin(np.abs(lats - y)))
        tv = d["t2"][i, j]
        ax.plot(x, y, "o", ms=3.5, color="k", transform=ccrs.PlateCarree())
        # Campos clear of the frame edge; Bangu left+down, clear of Rio's label
        dx, dy = {"Campos": (-34, 4), "Bangu": (-42, -12)}.get(nm, (4, 4))
        ax.annotate(f"{nm} {tv:.0f}", xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.8,
                    fontweight="bold", color="#000",
                    bbox=dict(fc="white", alpha=0.75, ec="none", pad=0.8))
    ax.set_extent([BOX["lon0"], BOX["lon1"], BOX["lat0"], BOX["lat1"]])
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    if label_left:
        ax.text(-0.02, 0.5, label_left, transform=ax.transAxes, rotation=90,
                va="center", ha="right", fontsize=11, fontweight="bold")


def interp_steps(data: dict, targets: list[int]) -> dict:
    """Linear time interpolation of a {step: fields} dict onto targets."""
    avail = sorted(s for s in data if "t2" in data.get(s, {}))
    out = {}
    for t in targets:
        if t in avail:
            out[t] = dict(data[t])
            continue
        lo = max((s for s in avail if s < t), default=None)
        hi = min((s for s in avail if s > t), default=None)
        if lo is None or hi is None:
            continue
        w = (t - lo) / (hi - lo)
        d = {k: (1 - w) * data[lo][k] + w * data[hi][k]
             for k in ("t2", "u10", "v10")
             if k in data[lo] and k in data[hi]}
        d["lons"], d["lats"] = data[lo]["lons"], data[lo]["lats"]
        d["interp"] = True
        out[t] = d
    return out


ANIM_DIR = REPO / "assets" / "brazil" / "anim"


def anim(date: str, rows, tlev, cmap, nrm, states, region="sbr_temp"):
    """Per-frame webp files + loop.html manifest, every 3 h through day 10."""
    import io as _io
    import json as _json
    from PIL import Image
    d0 = datetime.strptime(date, "%Y%m%d")
    targets = list(range(3, 241, 3))
    rows = [(name, interp_steps(data, targets)) for name, data in rows]
    fdir = ANIM_DIR / region
    fdir.mkdir(parents=True, exist_ok=True)
    for old in fdir.glob("*.webp"):
        old.unlink()
    frames_meta = []
    for s in targets:
        if not all(s in data for _, data in rows):
            continue
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 17.2),
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        fig.subplots_adjust(left=0.01, right=0.875, top=0.94, bottom=0.005,
                            hspace=0.08)
        for ax, (name, data) in zip(axes, rows):
            d = data[s]
            ttl = name + (" (interp)" if d.get("interp") else "")
            panel(ax, d, name, tlev, cmap, nrm, states, title=ttl)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cmap),
                          ax=list(axes), fraction=0.03, pad=0.015,
                          extend="both", ticks=tlev[::2])
        cb.ax.tick_params(labelsize=8)
        cb.set_label("2-m temperature (°C)", fontsize=9)
        vd = d0 + timedelta(hours=s)
        loc = vd - timedelta(hours=3)                       # BRT
        fig.suptitle(f"SE/S Brazil 2-m temp + 10-m wind\n"
                     f"{vd:%a %b %d %H}Z ({loc:%-I %p} BRT), "
                     f"day {(s - 1) // 24 + 1}  ·  00Z {d0:%Y-%m-%d} runs",
                     fontsize=13, fontweight="bold")
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=105)
        plt.close(fig)
        buf.seek(0)
        fn = f"{s:03d}.webp"
        Image.open(buf).convert("RGB").save(fdir / fn, format="WEBP",
                                            quality=80)
        frames_meta.append({"idx": len(frames_meta), "file": fn,
                            "date": f"{vd:%Y-%m-%d %H}Z",
                            "label": f"{vd:%a %b %d %H}Z ({loc:%-I %p} BRT)"})
        if len(frames_meta) % 20 == 0:
            print(f"  {len(frames_meta)} frames…", flush=True)
    manifest = {"ver": int(datetime.now(timezone.utc).timestamp()),
                "regions": {region: {
                    "label": ("SE/S Brazil 2-m temperature + 10-m wind — "
                              "GDPS / AIFS-ENS control / IFS, 3-hourly"),
                    "n_frames": len(frames_meta), "frames": frames_meta}}}
    (ANIM_DIR / f"{region}_manifest.json").write_text(_json.dumps(manifest))
    print(f"wrote {len(frames_meta)} frames + {region}_manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc)
                    .strftime("%Y%m%d"))
    ap.add_argument("--steps", nargs="+", type=int,
                    help="forecast hours (18Z valids: 18+24k)")
    ap.add_argument("--out", help="output file name (default sbr_temp_zoom.webp)")
    ap.add_argument("--anim", action="store_true",
                    help="animated webp, one frame per day (default d1-10)")
    args = ap.parse_args()
    date = args.date
    global STEPS
    if args.steps:
        STEPS = args.steps
    elif args.anim:
        STEPS = [18 + 24 * k for k in range(10)]
    outpng = (REPO / "brazil_hydro" / args.out) if args.out else \
        (REPO / "brazil_hydro" / "sbr_temp_anim.webp" if args.anim else OUTPNG)

    print("fetching GDPS / AIFS-ENS cf / IFS …", flush=True)
    if args.anim:
        import pickle
        cachef = Path(tempfile.gettempdir()) / f"sbr_anim_{date}.pkl"
        if cachef.exists():
            rows = pickle.loads(cachef.read_bytes())
            print("  (using cached fields)")
        else:
            rows = [
                ("CMC GDPS 15 km", fetch_gdps(date, list(range(3, 241, 3)))),
                ("AIFS-ENS control", fetch_ecmwf("aifs-ens", date, "enfo",
                                                 "cf", list(range(6, 241, 6)))),
                ("IFS HRES", fetch_ecmwf("ifs", date, "oper", "fc",
                                         list(range(3, 145, 3))
                                         + list(range(150, 241, 6))))]
            cachef.write_bytes(pickle.dumps(rows))
    else:
        rows = [("CMC GDPS 15 km", fetch_gdps(date)),
                ("AIFS-ENS control", fetch_ecmwf("aifs-ens", date, "enfo",
                                                 "cf")),
                ("IFS HRES", fetch_ecmwf("ifs", date, "oper", "fc"))]

    # NWS-style fine temperature ramp, 2C steps 8..42
    tlev = list(range(8, 43, 2))
    tcols = ["#4a148c", "#3f51b5", "#1976d2", "#29b6f6", "#80deea",
             "#a5d6a7", "#66bb6a", "#2e7d32", "#c0ca33", "#fdd835",
             "#ffb300", "#fb8c00", "#f4511e", "#e53935", "#b71c1c",
             "#8e0000", "#650a5a"][:len(tlev) - 1]
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("temp_cont", tcols)
    cmap.set_over("#3d0038")
    cmap.set_under("#2a0a4a")
    nrm = Normalize(vmin=tlev[0], vmax=tlev[-1])

    d0 = datetime.strptime(date, "%Y%m%d")
    nc = len(STEPS)
    fig, axes = plt.subplots(3, nc, figsize=(4.4 * nc + 0.6, 11.5),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             squeeze=False)
    fig.subplots_adjust(left=0.015, right=0.93, top=0.90, bottom=0.02,
                        wspace=0.03, hspace=0.12)
    states = cfeature.NaturalEarthFeature("cultural",
                                          "admin_1_states_provinces_lines",
                                          "10m", facecolor="none")
    if args.anim:
        anim(date, rows, tlev, cmap, nrm, states)
        return
    for ri, (name, data) in enumerate(rows):
        for ci, s in enumerate(STEPS):
            ax = axes[ri, ci]
            d = data.get(s)
            if not d or "t2" not in d:
                ax.set_title(f"{name}: unavailable", fontsize=8)
                ax.axis("off")
                continue
            lons, lats = d["lons"], d["lats"]
            ax.contourf(lons, lats, d["t2"], levels=tlev, cmap=cmap, norm=nrm,
                        extend="both", transform=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", lw=1.0, color="#111")
            ax.add_feature(states, edgecolor="#444", lw=0.5)
            if "u10" in d and "v10" in d:
                stride = max(1, int(round(0.9 / abs(lats[1] - lats[0]))))
                LO, LA = np.meshgrid(lons[::stride], lats[::stride])
                ax.barbs(LO, LA, d["u10"][::stride, ::stride] * 1.94384,
                         d["v10"][::stride, ::stride] * 1.94384,
                         length=4.6, linewidth=0.55, color="#222",
                         transform=ccrs.PlateCarree())
            for nm, x, y in CITIES:
                j = int(np.argmin(np.abs(lons - x)))
                i = int(np.argmin(np.abs(lats - y)))
                tv = d["t2"][i, j]
                ax.plot(x, y, "o", ms=3.5, color="k",
                        transform=ccrs.PlateCarree())
                dx, dy = {"Campos": (-34, 4),
                          "Bangu": (-42, -12)}.get(nm, (4, 4))
                ax.annotate(f"{nm} {tv:.0f}", xy=(x, y), xytext=(dx, dy),
                            textcoords="offset points", fontsize=6.8,
                            fontweight="bold", color="#000",
                            path_effects=None,
                            bbox=dict(fc="white", alpha=0.75, ec="none",
                                      pad=0.8))
            ax.set_extent([BOX["lon0"], BOX["lon1"], BOX["lat0"], BOX["lat1"]])
            vd = d0 + timedelta(hours=s)
            if ri == 0:
                ax.set_title(f"{vd:%a %b %d} 18Z (3 pm BRT)", fontsize=10,
                             fontweight="bold")
            if ci == 0:
                ax.text(-0.02, 0.5, name, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=11,
                        fontweight="bold")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cmap),
                      ax=axes.ravel().tolist(), fraction=0.015, pad=0.012,
                      extend="both", ticks=tlev)
    cb.set_label("2-m temperature (°C)", fontsize=9)
    fig.suptitle(f"SE/S Brazil coastal 2-m temperature + 10-m wind — 00Z "
                 f"{d0:%Y-%m-%d} runs, afternoon valids  ·  city labels = "
                 "nearest-gridpoint °C", fontsize=13, fontweight="bold")
    fig.savefig(outpng, dpi=130, bbox_inches="tight")
    print(f"wrote {outpng.name}")


if __name__ == "__main__":
    main()
