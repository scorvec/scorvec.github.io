"""
Top-level driver: download AIFS-ENS, compute wind-only RMM, and plot.

CDS-free: reads the committed 120-day filter map (data/reference/wind_map120.nc)
and observed-RMM history (data/reference/obs_history.nc), and extends the history
with today's AIFS analysis.  Refresh those committed files locally (with CDS) via
src/seed_recent.py — this keeps ERA5/CDS out of automated runs.

Usage:
    python run_rmm.py --date 20240601 --time 00

Prerequisites (run once / occasionally):
    python src/setup_reference.py    # W&H reference EOFs + climatology
    python src/seed_recent.py        # wind_map120.nc + obs_history.nc (needs CDS)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from download_aifs import download
from rmm import compute_rmm
from plot import plot_rmm
import recent_analysis
import archive_truth
import pandas as pd
import xarray as xr

CLIM_PATH = Path("data/reference/climatology.nc")
EOFS_PATH = Path("data/reference/eofs.nc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Init date YYYYMMDD")
    parser.add_argument("--time", default="00", help="Init hour: 00 or 12")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()

    if not CLIM_PATH.exists() or not EOFS_PATH.exists():
        print("ERROR: Run src/setup_reference.py first.")
        sys.exit(1)
    if not recent_analysis.MAP120_PATH.exists():
        print("ERROR: Missing wind_map120.nc — run src/seed_recent.py first.")
        sys.exit(1)

    # 1. Download AIFS-ENS
    if not args.skip_download:
        download(args.date, args.time, Path("data/aifs"))

    clim = xr.open_dataset(CLIM_PATH)
    eofs = xr.open_dataset(EOFS_PATH)
    init = pd.Timestamp(f"{args.date}T{args.time}:00")

    # 2. Compute wind-only RMM, filtered by the committed 120-day map
    mean120 = recent_analysis.load_map120()
    print("Computing RMM …")
    rmm = compute_rmm(Path("data/aifs"), args.date, args.time, clim, eofs,
                      mean120=mean120)

    rmm_path = Path("data/aifs") / f"rmm_{args.date}_{args.time}z.nc"
    rmm.to_netcdf(rmm_path)

    # 3. Extend the observed history with today's AIFS analysis (control day 0)
    cf0 = rmm.sel(member="cf", lead_day=0)
    archive_truth.append_truth(init, float(cf0["rmm1"]), float(cf0["rmm2"]))
    obs = archive_truth.load_truth(days=60)

    # 4. Plot
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_png = Path(args.out_dir) / f"rmm_{args.date}_{args.time}z.png"
    plot_rmm(rmm, obs=obs, out_path=out_png)


if __name__ == "__main__":
    main()
