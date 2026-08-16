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

from download_aifs import download, download_ifs
from rmm import compute_rmm
from plot import plot_rmm
import recent_analysis
import archive_truth
import cycle_debias
import numpy as np
import pandas as pd
import xarray as xr

CLIM_PATH = Path("data/reference/climatology.nc")
EOFS_PATH = Path("data/reference/eofs.nc")
PRCP_CLIM_PATH = Path("data/reference/prcp_clim.nc")


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
    # The map drifts <1%/day so a week of staleness is by design (ARCO lag +
    # refresh cadence) — but a silently frozen filter slowly re-admits the
    # ENSO signal into RMM. Surface it where CI annotations pick it up.
    if mean120.get("window_end"):
        stale = (init - pd.Timestamp(mean120["window_end"])).days
        if stale > 14:
            print(f"::warning::wind_map120.nc window ends {mean120['window_end']} "
                  f"({stale} days before init) — low-frequency filter is stale; "
                  "check the map120 refresh job")
    print("Computing RMM …")
    prcp_clim = (xr.open_dataset(PRCP_CLIM_PATH) if PRCP_CLIM_PATH.exists()
                 else None)
    if prcp_clim is None:
        print("::warning::prcp_clim.nc missing — RMM will be wind-only "
              "(run src/build_prcp_clim.py)")
    rmm = compute_rmm(Path("data/aifs"), args.date, args.time, clim, eofs,
                      mean120=mean120, prcp_clim=prcp_clim)
    print(f"  channels: {rmm.attrs.get('channels')}")

    # 2b. De-bias the 12Z cycle onto the 00Z family (see cycle_debias docstring:
    # AIFS-ENS 12Z runs are systematically RMM-offset vs 00Z runs at the same
    # valid times, which made alternating frames "windshield-wiper"). The RAW
    # ensemble mean is archived first — the correction is estimated from raw
    # trailing 00Z/12Z pairs, so re-running a cycle never double-corrects.
    leads = rmm["lead_day"].values
    cycle_debias.record(args.date, args.time,
                        leads, rmm["rmm1"].mean("member").values,
                        rmm["rmm2"].mean("member").values)
    off1, off2 = cycle_debias.offset_for(args.date, args.time, leads)
    if np.any(off1) or np.any(off2):
        rmm["rmm1"] = rmm["rmm1"] - xr.DataArray(off1, dims=["lead_day"])
        rmm["rmm2"] = rmm["rmm2"] - xr.DataArray(off2, dims=["lead_day"])
        rmm.attrs["cycle_debias"] = (
            f"12Z-family offset removed; max |d| = "
            f"{float(np.hypot(off1, off2).max()):.2f} RMM units")
        print(f"12Z cycle de-bias applied (day-14 offset "
              f"({off1[-1]:+.2f}, {off2[-1]:+.2f}))")

    rmm_path = Path("data/aifs") / f"rmm_{args.date}_{args.time}z.nc"
    rmm.to_netcdf(rmm_path)

    # 2c. IFS-ENS through the IDENTICAL machinery — the physics-model reference
    # for the AI forecast (also a methodology cross-check against other vendors).
    # Best-effort: IFS disseminates ~1-2 h after AIFS; when absent the sidecar
    # below tells the hourly poll to re-run this stage until it lands.
    rmm_ifs = None
    try:
        if download_ifs(args.date, args.time, Path("data/aifs")):
            print("Computing IFS RMM …")
            rmm_ifs = compute_rmm(Path("data/aifs"), args.date, args.time,
                                  clim, eofs, mean120=mean120,
                                  prcp_clim=prcp_clim, model="ifs")
            print(f"  IFS channels: {rmm_ifs.attrs.get('channels')}")
            rmm_ifs.to_netcdf(Path("data/aifs") / f"rmm_ifs_{args.date}_{args.time}z.nc")
    except Exception as e:                          # noqa: BLE001
        print(f"IFS RMM leg failed ({repr(e)[:80]}); plotting AIFS only")
        rmm_ifs = None

    # 3. Extend the observed history with today's AIFS analysis (control, earliest
    #    lead). lead_day 0 if step 0 was downloaded, else the first forecast day —
    #    .isel keeps this robust to the daily-vs-6-hourly step choice.
    cf0 = rmm.sel(member="cf").isel(lead_day=0)
    archive_truth.append_truth(init, float(cf0["rmm1_wind"]), float(cf0["rmm2_wind"]))
    obs = archive_truth.load_truth(days=120)      # 12-hourly points → same ~60-day window

    # 4. Plot
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_png = Path(args.out_dir) / f"rmm_{args.date}_{args.time}z.png"
    plot_rmm(rmm, obs=obs, out_path=out_png, ifs=rmm_ifs)
    miss = Path(str(out_png) + ".missing")
    if rmm_ifs is None:
        miss.write_text("ifs\n")
    elif miss.exists():
        miss.unlink()


if __name__ == "__main__":
    main()
