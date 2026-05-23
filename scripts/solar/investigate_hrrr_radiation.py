"""Investigate HRRR shortwave radiation variables.

Goal: figure out whether HRRR's VBDSF and VDDSF (labeled "visible") are
actually visible-band-only (~0.3-0.7 µm) or are mislabeled and actually
contain broadband shortwave (0.3-4.0 µm).

Method: download a recent HRRR analysis cycle, sample DSWRF, VBDSF, and
VDDSF at several clear-sky midday locations across the US, and look at
the ratio (VBDSF + VDDSF) / DSWRF.

Expected results:
  - If ratio ≈ 1.0  → HRRR's "visible" labels are broadband (mislabeled)
  - If ratio ≈ 0.4-0.5 → it really is visible-band-only
  - If ratio is something else → something more interesting going on

Usage:
    cd ~/scorvec.github.io/scripts
    python investigate_hrrr_radiation.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from herbie import Herbie


# Pick a cycle that's been published. HRRR analysis (fxx=0) is published
# ~55 min after the cycle hour. To be safe, look back 2 hours from now.
NOW = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
CYCLE = NOW - timedelta(hours=2)

# Probe locations: places with reliable clear-sky midday conditions in
# May (so radiation will be near maximum). Local solar noon roughly
# 18-20 UTC across CONUS. Lat/lon used to find nearest grid cell.
LOCATIONS = [
    ("Phoenix, AZ",        33.45, -112.07),
    ("Las Vegas, NV",      36.17, -115.14),
    ("Albuquerque, NM",    35.08, -106.65),
    ("Denver, CO",         39.74, -104.99),
    ("Austin, TX",         30.27,  -97.74),
    ("Tampa, FL",          27.95,  -82.46),
    ("Sacramento, CA",     38.58, -121.49),
    ("Atlanta, GA",        33.75,  -84.39),
]


def fetch_field(H: Herbie, search_string: str, label: str):
    """Download one variable from the HRRR wrfsfc file and return as xarray."""
    print(f"  Fetching {label} ({search_string}) ... ", end="", flush=True)
    try:
        ds = H.xarray(search_string)
        # Cfgrib opens with a variable name that depends on the field.
        # The dataset usually has exactly one data variable; grab whichever
        # one isn't a coordinate.
        data_vars = list(ds.data_vars)
        if not data_vars:
            print("FAIL (no data vars)")
            return None
        var = data_vars[0]
        print(f"ok ({var}, shape={ds[var].shape})")
        return ds[var]
    except Exception as e:
        print(f"FAIL ({e})")
        return None


def sample_at(field, lat: float, lon: float) -> float:
    """Return the value of `field` at the grid cell nearest (lat, lon).

    HRRR longitudes are stored as 0-360, so convert lon as needed.
    """
    # HRRR uses a 2D lat/lon coordinate (Lambert conformal grid), so we
    # can't use .sel(latitude=lat); we have to find the nearest cell
    # using a brute-force distance check.
    if "latitude" not in field.coords or "longitude" not in field.coords:
        return float("nan")
    lats = field.latitude.values
    lons = field.longitude.values
    # Normalize HRRR's 0-360 longitude convention
    lons_360 = lons.copy()
    target_lon = lon % 360.0
    dist2 = (lats - lat) ** 2 + (lons_360 - target_lon) ** 2
    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)
    return float(field.values[iy, ix])


def main():
    print("=" * 70)
    print("HRRR shortwave radiation probe")
    print("=" * 70)
    print(f"Cycle:       {CYCLE.strftime('%Y-%m-%d %H:%MZ')}")
    print(f"Forecast:    analysis (fxx=0)")
    print()

    # Try a few cycle hours in case the most recent one hasn't published.
    # Herbie wants tz-naive datetimes (it treats them as UTC internally).
    H = None
    NOW_NAIVE = NOW.replace(tzinfo=None)
    for hours_back in [2, 3, 4, 5, 6]:
        candidate_cycle = NOW_NAIVE - timedelta(hours=hours_back)
        try:
            print(f"Trying cycle: {candidate_cycle.strftime('%Y-%m-%d %H:%MZ')} ...")
            H = Herbie(candidate_cycle, model="hrrr", product="sfc", fxx=0)
            # Check if file actually exists by trying to access source
            if H.grib is None:
                print("  not available, trying older cycle")
                continue
            print(f"  ok: {H.grib}")
            CYCLE_USED = candidate_cycle
            break
        except Exception as e:
            print(f"  failed: {e}")
            H = None
    if H is None:
        print("ERROR: could not find a usable HRRR cycle")
        return 1

    print()
    print("Fetching radiation fields:")
    dswrf = fetch_field(H, ":DSWRF:surface", "DSWRF (downward shortwave)")
    vbdsf = fetch_field(H, ":VBDSF:surface", "VBDSF (visible beam)")
    vddsf = fetch_field(H, ":VDDSF:surface", "VDDSF (visible diffuse)")
    # Also try near-IR if HRRR happens to have them
    nbdsf = fetch_field(H, ":NBDSF:surface", "NBDSF (near-IR beam, optional)")
    nddsf = fetch_field(H, ":NDDSF:surface", "NDDSF (near-IR diffuse, optional)")

    if any(x is None for x in [dswrf, vbdsf, vddsf]):
        print("\nMissing one of the required fields; aborting.")
        return 1

    print()
    print("Sampling at clear-sky probe locations:")
    print(f"{'Location':<22}{'DSWRF':>10}{'VBDSF':>10}{'VDDSF':>10}"
          f"{'V+V/DSWRF':>12}{'NBDSF':>10}{'NDDSF':>10}{'B+D/DSWRF':>12}")
    print("-" * 100)

    rows = []
    for name, lat, lon in LOCATIONS:
        d = sample_at(dswrf, lat, lon)
        vb = sample_at(vbdsf, lat, lon)
        vd = sample_at(vddsf, lat, lon)
        nb = sample_at(nbdsf, lat, lon) if nbdsf is not None else float("nan")
        nd = sample_at(nddsf, lat, lon) if nddsf is not None else float("nan")
        vis_total = vb + vd
        vis_ratio = vis_total / d if d > 1.0 else float("nan")
        all_total = vb + vd + (nb if not np.isnan(nb) else 0) + (nd if not np.isnan(nd) else 0)
        all_ratio = all_total / d if d > 1.0 else float("nan")
        rows.append({
            "location": name,
            "DSWRF": d, "VBDSF": vb, "VDDSF": vd,
            "NBDSF": nb, "NDDSF": nd,
            "vis_ratio": vis_ratio, "all_ratio": all_ratio,
        })
        nbdsf_str = f"{nb:10.1f}" if not np.isnan(nb) else f"{'--':>10}"
        nddsf_str = f"{nd:10.1f}" if not np.isnan(nd) else f"{'--':>10}"
        all_str = f"{all_ratio:12.3f}" if not np.isnan(all_ratio) else f"{'--':>12}"
        print(f"{name:<22}{d:10.1f}{vb:10.1f}{vd:10.1f}{vis_ratio:12.3f}"
              f"{nbdsf_str}{nddsf_str}{all_str}")

    print()
    print("=" * 70)
    print("Interpretation guide:")
    print("=" * 70)
    print()
    print("  (VBDSF + VDDSF) / DSWRF  ratio:")
    print("    ~0.40-0.50  → HRRR really gives visible-only; need scaling")
    print("    ~0.95-1.00  → HRRR's 'visible' fields are actually broadband")
    print("    ~0.55-0.65  → unexpected; may include near-IR partially")
    print()
    print("  If NBDSF/NDDSF are populated:")
    print("    (VBDSF+VDDSF+NBDSF+NDDSF) / DSWRF should be ~1.0")
    print("    → can build broadband direct = VBDSF + NBDSF")

    # Save raw results to CSV for review
    df = pd.DataFrame(rows)
    out_path = "/tmp/hrrr_radiation_probe.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved raw values to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
