#!/usr/bin/env python3
"""Region masks for the intercomparison, built ONCE (in Actions) exactly as the torque study builds them
(scripts/mjo/src/torque_increments_maps.py masks()), and saved small: data/region_masks_0p25.npz

  ORO  terrain above 1000 m OR ERA5 subgrid-orography std (sdor) above 100 m, plus the 500 km downstream (east)
  OCX  ocean (ERA5 land-sea mask < 0.5) at 30-65 degrees latitude, outside ORO (the storm tracks)
  OCO  other ocean outside ORO;  LND  other land outside ORO
bits: 1 = ORO, 2 = OCX, 4 = OCO, 8 = LND (exclusive). lsm is the ERA5 land-sea fraction. ONE common mask for every
model (nearest neighbour onto each model's grid), so a region means the same place in every model.

Reads the two static fields from ARCO-ERA5 (Google public bucket, anonymous) and scripts/mjo/data/reference/
era5_orography.nc. Contains modified Copernicus Climate Change Service information.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                     # noqa: E402

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def main():
    import gcsfs
    import xarray as xr
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(STORE), chunks=None, decode_timedelta=False)

    def static(name):
        v = ds[name]
        if "time" in v.dims:
            v = v.sel(time="2000-06-15T12:00", method="nearest")
        return np.asarray(v.values, dtype=np.float64)[::-1, :]          # lat ascending
    lat = ds.latitude.values[::-1].astype(np.float64)
    lon = ds.longitude.values.astype(np.float64)
    lsm = static("land_sea_mask")
    sdor = static("standard_deviation_of_orography")
    h = xr.open_dataarray(C.REF / "era5_orography.nc").values.astype(np.float64)
    assert h.shape == lsm.shape and np.isfinite(lsm).all() and 0.2 < lsm.mean() < 0.45, "static fields look wrong"

    def downstream(core):
        out = core.copy()
        for i, la in enumerate(lat):
            if not core[i].any():
                continue
            nshift = int(np.ceil(500e3 / (C.A * np.deg2rad(0.25) * max(np.cos(np.deg2rad(la)), 0.05))))
            row = core[i]
            acc = row.copy()
            for s in range(1, min(nshift, 1440) + 1):
                acc |= np.roll(row, s)
            out[i] = acc
        return out
    oro = downstream(h > 1000.0) | downstream(sdor > 100.0)
    ocean = lsm < 0.5
    ext = (np.abs(lat) >= 30) & (np.abs(lat) <= 65)
    ocx = ocean & ext[:, None] & ~oro
    oco = ocean & ~ext[:, None] & ~oro
    lnd = ~ocean & ~oro
    bits = (oro * 1 + ocx * 2 + oco * 4 + lnd * 8).astype(np.uint8)
    assert (np.isin(bits, [1, 2, 4, 8])).all()
    C.DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.MASKS, bits=bits, lsm=lsm.astype(np.float16), lat=lat, lon=lon)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(lsm)
    for nm, m in (("ORO", oro), ("OCX", ocx), ("OCO", oco), ("LND", lnd)):
        print(f"{nm}: {100 * (w * m).sum() / w.sum():.1f}% of the globe")
    print("wrote", C.MASKS, C.MASKS.stat().st_size // 1024, "kB")


if __name__ == "__main__":
    main()
