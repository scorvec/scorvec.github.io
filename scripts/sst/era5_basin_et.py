#!/usr/bin/env python3
"""Basin evapotranspiration from ERA5 latent heat flux (WB2, 1959-2023).

E (mm/day) = -LE (W/m^2) / 28.94 — the latent-heat-to-water conversion
(rho_w * L_v; ERA5's upward flux is negative). Streams the WB2 6-hourly
1.5-degree mean_surface_latent_heat_flux over the basin bounding box,
daily-means it, and reduces to cos-weighted basin series. 1.5-degree
cells are coarse against 1-3-degree basins (few cells each — noted in
the ledger); good enough to quantify the ENSO-ET effect over 64 years,
which a 2-year window cannot separate from rainfall.

Outputs:
  ~/colombia_hydro/raw/era5_basin_et_daily.json   (per-basin mm/day series)
  colombia_hydro/data/et_enso.json                 (ENSO-composite summary)

    python scripts/sst/era5_basin_et.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from matplotlib.path import Path as MplPath      # noqa: E402

REPO = HERE.parent.parent
REGIONS_GJ = HERE / "colombia_hydro_regions.geojson"
NINO_JSON = REPO / "assets" / "sst" / "data" / "nino_history.json"
RAW_OUT = Path.home() / "colombia_hydro" / "raw" / "era5_basin_et_daily.json"
OUT_JSON = REPO / "colombia_hydro" / "data" / "et_enso.json"
WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
LE_TO_MM = 28.94                    # W/m^2 per mm/day


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1959-01-01")
    ap.add_argument("--end", default="2023-01-09")
    a = ap.parse_args()
    if RAW_OUT.exists():
        raw = json.loads(RAW_OUT.read_text())
        print(f"cached: {raw['dates'][0]}..{raw['dates'][-1]}")
    else:
        ds = xr.open_zarr(WB2, storage_options={"token": "anon"})
        le = ds["mean_surface_latent_heat_flux"]
        # basin bbox at 1.5 deg (lon 0-360 in WB2)
        le = le.sel(longitude=slice(279, 291), latitude=slice(-1, 11),
                    time=slice(a.start, a.end))
        print("streaming LE box", dict(le.sizes), flush=True)
        led = le.resample(time="1D").mean().compute()
        E = (-led / LE_TO_MM)                       # mm/day, positive = ET up
        lons = E.longitude.values - 360.0           # geojson convention
        lats = E.latitude.values
        gj = json.loads(REGIONS_GJ.read_text())
        rings = {}
        for ft in gj["features"]:
            nm = (ft["properties"].get("region") or ft["properties"].get("name", "")).upper()
            g = ft["geometry"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            rings.setdefault(nm, []).extend(np.array(p[0]) for p in polys)
        LO, LA = np.meshgrid(lons, lats)
        out = {"dates": [f"{pd.Timestamp(t):%Y-%m-%d}" for t in E.time.values]}
        for r in ORDER:
            paths = [MplPath(rr) for rr in rings[r]]
            inside = np.zeros(LO.shape, bool)
            for pth in paths:
                inside |= pth.contains_points(
                    np.column_stack([LO.ravel(), LA.ravel()])).reshape(LO.shape)
            w = np.where(inside, np.cos(np.deg2rad(LA)), 0.0)
            if w.sum() == 0:                         # basin smaller than a 1.5° cell
                arr = np.vstack(rings[r])
                cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
                i = int(np.argmin(np.abs(lats - cy)))
                j = int(np.argmin(np.abs(lons - cx)))
                w = np.zeros(LO.shape)
                w[i, j] = 1.0
                print(f"  {r}: nearest-cell fallback ({lats[i]:.1f}N {lons[j]:.1f}E)")
            # E dims are (time, lon, lat) in WB2 — transpose to match mask
            v = E.transpose("time", "latitude", "longitude").values
            out[r] = np.round((v * w).sum(axis=(1, 2)) / w.sum(), 3).tolist()
            print(f"  {r}: mean {np.mean(out[r]):.2f} mm/day", flush=True)
        RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
        RAW_OUT.write_text(json.dumps(out, separators=(",", ":")))
        raw = out
        print(f"wrote {RAW_OUT}")

    # ── ENSO composite: monthly ET anomaly vs ONI ───────────────────────────
    dates = pd.to_datetime(raw["dates"])
    nh = json.loads(NINO_JSON.read_text())
    oni = dict(zip(nh["months"], nh["series"]["oni"]["anom"]
                   if isinstance(nh["series"]["oni"], dict) else nh["series"]["oni"]))
    summary = {}
    for r in ORDER:
        s = pd.Series(raw[r], index=dates)
        mo = s.resample("MS").mean()
        clim = mo.groupby(mo.index.month).transform("mean")
        anom = mo - clim
        keys = [f"{t:%Y-%m}" for t in anom.index]
        o = np.array([oni.get(k, np.nan) for k in keys], dtype=float)
        a = anom.values
        m = np.isfinite(o) & np.isfinite(a)
        nino = a[m & (o >= 0.5)].mean()
        nina = a[m & (o <= -0.5)].mean()
        corr = float(np.corrcoef(o[m], a[m])[0, 1])
        summary[r] = {"mean_et_mmday": round(float(s.mean()), 2),
                      "elnino_anom_mmday": round(float(nino), 3),
                      "lanina_anom_mmday": round(float(nina), 3),
                      "corr_oni": round(corr, 3),
                      "n_months": int(m.sum())}
        print(r, summary[r], flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "ERA5 mean_surface_latent_heat_flux (WB2 1.5deg 6h), 1959-2023",
        "conversion": "E[mm/day] = -LE[W/m2] / 28.94",
        "note": "1.5-deg cells are coarse vs 1-3-deg basins; ENSO composite is "
                "robust to that, absolute levels are approximate",
        "summary": summary,
    }, indent=1))
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
