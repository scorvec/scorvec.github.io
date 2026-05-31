"""
LOCAL refresh of the CDS-free runtime inputs used by the daily GitHub Action:

  data/reference/wind_map120.nc   forecast 120-day filter map (from ERA5)
  data/reference/obs_history.nc   observed RMM history, seeded from ERA5

Run this locally whenever you want to refresh the slow-moving 120-day map (the
ENSO/low-frequency signal changes slowly, so monthly is plenty).  It needs CDS
for fresh ERA5 (or reuses cached files in data/reference/era5_recent/), so it
stays OUT of CI — the daily Action only reads the committed .nc files.

    python src/seed_recent.py --date 20260530        # anchor date

Then commit wind_map120.nc and obs_history.nc.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import recent_analysis
import archive_truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="anchor date YYYYMMDD (default: today UTC)")
    ap.add_argument("--source", default="era5", choices=["era5", "ncep"])
    args = ap.parse_args()

    init = pd.Timestamp(args.date) if args.date else pd.Timestamp.utcnow().normalize()
    clim = xr.open_dataset("data/reference/climatology.nc")
    eofs = xr.open_dataset("data/reference/eofs.nc")

    mean120, obs = recent_analysis.build(init, clim, eofs, source=args.source)
    recent_analysis.save_map120(mean120)
    archive_truth.seed(obs)

    print(f"wrote {recent_analysis.MAP120_PATH.name} "
          f"({mean120['u850'].size}-lon map) and seeded "
          f"{archive_truth.ARCHIVE.name} ({obs.sizes['time']} obs days)")


if __name__ == "__main__":
    main()
