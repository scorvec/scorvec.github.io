#!/usr/bin/env python3
"""AIFS-ENS and IFS-ENS daily rainfall, days 1-15, for the IOD + El Nino study (2026-10-05, temporary; runs in Actions
only via .github/workflows/cmip6-iod.yml). ECMWF open data from the Google Cloud mirror ONLY (scripts/ecmwf/store.py).

Per model: total precipitation at the daily 00Z-anchored steps 24..360 h for cf + 50 pf, daily accumulations (mm/day),
1-degree box means over 40S-40N, all longitudes. Output: <out>/ecmwf_tp_<model>_<YYYYMMDD>00.npz (float16).

    python scripts/sst/iod_ecmwf_fetch.py --date 20261005 --out out
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

os.environ.setdefault("ECMWF_SOURCES", "google")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ecmwf"))
import store as ecmwf                                                              # noqa: E402

STEPS = tuple(range(24, 361, 24))
LAT1 = np.arange(-39.5, 40.0, 1.0); LON1 = np.arange(0.5, 360.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00"); ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cyc = ecmwf.Cycle(a.date, a.time)
    for model in ("aifs-ens", "ifs"):
        try:
            mem = []
            for typ in ("cf", "pf"):
                p = ecmwf.ensure(cyc, ecmwf.Spec(model, typ, "tp", "sfc", (), STEPS))
                da = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})["tp"]
                units = da.attrs.get("units", "").strip()
                if units in ("m", "metre", "metres"):
                    da = da * 1000.0
                da = da.sortby("latitude").sel(latitude=slice(-41, 41))
                da = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")
                if "number" not in da.dims:
                    da = da.expand_dims("number")
                da = da.transpose("number", "step", "latitude", "longitude")
                x = da.values.astype("float64")
                daily = np.diff(np.concatenate([np.zeros_like(x[:, :1]), x], axis=1), axis=1)
                lat, lon = da.latitude.values.astype(float), da.longitude.values.astype(float)
                # 1-degree box means via reshape (0.25-degree regular grid, lat -41..41 sliced to the 40S-40N boxes)
                k = int(round(1.0 / abs(lat[1] - lat[0])))
                i0 = int(np.argmin(np.abs(lat + 40.0)))
                sel = daily[..., i0:i0 + 80 * k, :]
                la = lat[i0:i0 + 80 * k]
                if lon.size != 360 * k or abs(lon[0]) > 1e-6 or sel.shape[-2] != 80 * k:
                    raise ValueError(f"unexpected grid lat {lat.size} lon {lon.size} first {lon[0]}")
                box = sel.reshape(sel.shape[0], sel.shape[1], 80, k, 360, k).mean(axis=(3, 5))
                print(model, typ, x.shape, "units", units, "lat", la[0], la[-1], "k", k, flush=True)
                mem.append(box.astype("float32"))
            tp = np.concatenate(mem, axis=0)
            dest = out / f"ecmwf_tp_{model}_{a.date}{a.time}.npz"
            np.savez_compressed(dest, tp=tp.astype("float16"), lat=LAT1, lon=LON1, steps=np.array(STEPS),
                                note="daily accumulations (mm/day), day k = step 24(k-1)..24k h; member 0 = control; "
                                     "1-degree means of the k x k 0.25-degree points starting at each whole degree (true centre = "
                                     "nominal LAT1/LON1 - 0.125 deg)")
            print("wrote", dest, tp.shape, float(np.nanmean(tp)), flush=True)
        except Exception as e:                                                     # noqa: BLE001
            print(f"{model}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
