#!/usr/bin/env python3
"""Numerical deep-dive validation of the site's AAM calculation.

Independent recomputation (fresh code path, trapezoid over ERA5's full 37
pressure levels, 1000→1 hPa, surface-pressure masked) compared, for a set of
benchmark dates spanning ENSO states, against:

  A. full 37-level AAM (the "traditional physics-model" column, to 1 hPa)
  B. the site's 13-level subset integrated with aam.py's _vert_weights
  C. aam.py's own aam_of() on the same 13-level fields (weights-code check)
  D. the site's archived observed series (aam_history.nc) where dates overlap

Prints a table of absolutes and the decomposition of any offset:
  - quadrature difference (B vs A on the same data)  → vertical-truncation bias
  - code difference (C vs B)                         → would indicate a bug
  - archive difference (D vs C)                      → pipeline consistency

    python src/validate_aam.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from aam import aam_of, LEVELS, A, G, SCALE

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
REF = Path(__file__).resolve().parent.parent / "data" / "reference"

# benchmark dates: El Niño peak, La Niña, neutral, strong Niño, + one exact
# aam_history.nc timestamp for the archive-consistency check
DATES = ["2016-01-15T12:00", "2022-07-15T12:00", "2019-10-15T12:00",
         "2024-01-15T12:00", "2026-07-12T00:00"]


def full_column_aam(u, sp, p_levels_hpa):
    """Independent implementation: M = (a³/g)·Σ_λφ cos²φ dλ dφ · ∫ u dp with a
    plain trapezoid over ALL levels, integrating only where p > level (above
    ground), plus the surface partial layer down to sp using the lowest
    above-ground level's wind."""
    lat = u.latitude.values
    lon = u.longitude.values
    dlam = np.deg2rad(abs(float(lon[1] - lon[0])))
    dphi = np.deg2rad(abs(float(lat[1] - lat[0])))
    p_pa = np.asarray(p_levels_hpa, float) * 100.0
    order = np.argsort(p_pa)
    p_pa = p_pa[order]
    uu = u.values[order]                              # (lev, lat, lon), p ascending
    spv = sp.values                                    # (lat, lon)

    # trapezoid between consecutive levels, each panel clipped to the surface
    integ = np.zeros_like(spv)
    for k in range(1, len(p_pa)):
        p0, p1 = p_pa[k - 1], p_pa[k]
        lo = np.minimum(p0, spv)
        hi = np.minimum(p1, spv)
        thick = np.clip(hi - lo, 0.0, None)
        integ += 0.5 * (uu[k] + uu[k - 1]) * thick
    # top partial layer 0 → lowest p (tiny mass, use top-level wind)
    integ += uu[0] * p_pa[0]
    # bottom partial layer: deepest level above ground → sp, extend lowest
    # above-ground wind (standard practice)
    below = spv > p_pa[-1]
    integ = np.where(below, integ + uu[-1] * (spv - p_pa[-1]), integ)

    cos2 = np.cos(np.deg2rad(lat)) ** 2
    dens = (A ** 3 / G) * integ * cos2[:, None] * dlam * dphi
    eq = 0.5 * dens[np.isclose(lat, 0.0)].sum()
    return (dens.sum(), dens[lat > 0].sum() + eq, dens[lat < 0].sum() + eq)


def main() -> int:
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    all_levels = ds.level.values.tolist()
    print(f"ERA5 levels: {len(all_levels)} ({all_levels[0]}–{all_levels[-1]} hPa); "
          f"site subset: {len(LEVELS)}")

    hist = None
    hp = REF / "aam_history.nc"
    if hp.exists():
        hist = xr.open_dataset(hp)

    rows = []
    for d in DATES:
        t = pd.Timestamp(d)
        try:
            u37 = ds["u_component_of_wind"].sel(time=t).load()
            sp = ds["surface_pressure"].sel(time=t).load()
        except KeyError:
            print(f"  {d}: not in ARCO yet — skipped"); continue

        gA, nA, sA = full_column_aam(u37, sp, all_levels)             # 37-level truth
        u13 = u37.sel(level=LEVELS).sortby("level")
        gB, nB, sB = full_column_aam(u13, sp, sorted(LEVELS))         # 13-lev, my quadrature
        lat = u13.latitude.values
        dlam = np.deg2rad(abs(float(u13.longitude[1] - u13.longitude[0])))
        dphi = np.deg2rad(abs(float(lat[1] - lat[0])))
        gC, nC, sC = aam_of(u13.values, np.asarray(sorted(LEVELS), float) * 100.0,
                            sp.values, lat, dlam, dphi)               # site's code

        row = dict(date=d[:10],
                   g37=gA / SCALE, g13_trap=gB / SCALE, g13_site=gC / SCALE,
                   nh37=nA / SCALE, nh_site=nC / SCALE,
                   sh37=sA / SCALE, sh_site=sC / SCALE)
        if hist is not None and "time" in hist.dims:
            ht = pd.to_datetime(hist.time.values)
            near = np.abs(ht - t)
            if len(near) and near.min() <= pd.Timedelta("1h"):
                i = int(np.argmin(near))
                row["g_archive"] = float(hist["global"].values[i])
                row["nh_archive"] = float(hist["nh"].values[i])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")
    pd.set_option("display.width", 160)
    print("\nAll values ×10²⁵ kg m² s⁻¹:")
    print(df.round(3).to_string())
    print("\nDecomposition (global):")
    print(f"  13-level truncation bias (site levels vs full 37): "
          f"mean {(df.g13_trap - df.g37).mean():+.3f}, "
          f"range {(df.g13_trap - df.g37).min():+.3f}…{(df.g13_trap - df.g37).max():+.3f}")
    print(f"  weights-code difference (aam_of vs independent trapezoid, same levels): "
          f"mean {(df.g13_site - df.g13_trap).mean():+.3f}, "
          f"max |{(df.g13_site - df.g13_trap).abs().max():.3f}|")
    print(f"  hemispheric closure: max |NH+SH−G| site = "
          f"{np.abs(df.nh_site + df.sh_site - df.g13_site).max():.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
