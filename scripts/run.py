#!/usr/bin/env python3
"""
Wind generation forecast — end-to-end runner.

Usage examples
--------------

# Operational: latest HRRR cycle, 0-18h forecast, save to ./output/
python run.py \
    --uswtdb data/uswtdb_v7_0_20240501.csv \
    --eia860 data/2___Plant_Y2023.xlsx \
    --output-dir output/

# Specific cycle (UTC), extended 0-48h forecast (only on 00/06/12/18Z runs)
python run.py \
    --uswtdb data/uswtdb.csv \
    --cycle 2026-05-01T12:00 \
    --fxx-end 48

# Backcast a historical day, all 24 cycles, fxx=1 only (analysis-like)
python run.py \
    --uswtdb data/uswtdb.csv \
    --backcast-date 2024-01-15 \
    --backcast-cycles all \
    --fxx-end 1 \
    --output-dir output/2024-01-15/

# Save per-turbine forecast in addition to BA aggregate
python run.py --uswtdb data/uswtdb.csv --save-per-turbine

# Skip oedb lookup (faster; uses NREL Tier-2 fallback only)
python run.py --uswtdb data/uswtdb.csv --no-oedb

Programmatic use
----------------
    from run import run, latest_hrrr_cycle
    df_ba = run(
        uswtdb_path="data/uswtdb.csv",
        cycle_dt=latest_hrrr_cycle(),
        fxx_range=range(0, 19),
        eia860_path="data/2___Plant_Y2023.xlsx",
        output_dir="output/",
    )
"""

from __future__ import annotations

import os
import sys

# Ensure the script's directory is searched FIRST so local modules
# (forecast.py, aggregation.py, wf_pipeline.py, etc.) always win over
# installed packages with the same generic names. Without this, a user
# who has e.g. PyPI's `forecast` or `pipeline` package installed will
# get the wrong module imported.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
elif sys.path[0] != _HERE:
    sys.path.remove(_HERE)
    sys.path.insert(0, _HERE)

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

log = logging.getLogger("wind_forecast.run")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_hrrr_cycle(latency_hours: float = 2.0) -> datetime:
    """Return the most recent HRRR cycle that is likely to be on the mirrors.

    HRRR runs hourly (00, 01, 02, ... UTC) but takes ~1-2 hours to land
    on NOMADS / AWS. Default `latency_hours=2` gives a safe pick.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=latency_hours)
    return cutoff.replace(minute=0, second=0, microsecond=0)


def parse_cycle(s: str) -> datetime:
    """Parse a CLI cycle string. Accepts ISO-8601 or 'latest'."""
    if s.lower() == "latest":
        return latest_hrrr_cycle()
    # Allow "2026-05-01T12:00", "2026-05-01 12:00", "2026-05-01T12"
    s = s.replace(" ", "T")
    if len(s) == 13:  # YYYY-MM-DDTHH
        s += ":00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy libraries
    for noisy in ("urllib3", "botocore", "s3fs", "fsspec",
                  "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _write_capacity_files(inv: pd.DataFrame, outd: Path, cycle_tag: str,
                          aggregation_levels: Iterable[str]) -> list:
    """Write per-region installed capacity (in MW) for each aggregation level.

    Used by the dashboard to draw reference lines. The grouping columns
    here MUST match what aggregation.aggregate() uses, otherwise the
    dashboard's join won't line up.
    """
    written = []
    level_to_keys = {
        "plant":    ["eia_id", "p_name"],
        "ba":       ["ba_code"],
        "iso":      ["iso"],
        "state":    ["t_state"],
        "national": [],
    }
    for level in aggregation_levels:
        keys = level_to_keys.get(level)
        if keys is None:
            continue
        if not keys:
            cap = pd.DataFrame({"region": ["National"],
                                "capacity_MW": [inv["t_cap"].sum() / 1000.0]})
        else:
            grouped = inv.groupby(keys, dropna=False)["t_cap"].sum() / 1000.0
            cap = grouped.reset_index()
            # Combine multi-key indices into a single 'region' label that
            # matches the forecast CSV's grouping column.
            if len(keys) == 1:
                cap = cap.rename(columns={keys[0]: "region",
                                          "t_cap": "capacity_MW"})
            else:
                cap["region"] = cap[keys].astype(str).agg(" / ".join, axis=1)
                cap = cap[["region", "t_cap"]].rename(
                    columns={"t_cap": "capacity_MW"})
        cap = cap.dropna(subset=["region"])

        # Add project-breakout capacity rows at the iso level so the
        # dashboard's reference line is correct when SunZia (or any
        # other breakout project) is selected. Mirrors the logic in
        # aggregation.aggregate(level="iso").
        if level == "iso":
            from aggregation import PROJECT_BREAKOUTS
            for eia_id, label in PROJECT_BREAKOUTS.items():
                proj_cap_kW = inv.loc[inv["eia_id"] == eia_id, "t_cap"].sum()
                if proj_cap_kW > 0:
                    cap = pd.concat([cap, pd.DataFrame({
                        "region": [label],
                        "capacity_MW": [proj_cap_kW / 1000.0],
                    })], ignore_index=True)

        cap_path = outd / f"capacity_{level}_{cycle_tag}.csv"
        cap.to_csv(cap_path, index=False)
        written.append(cap_path)
    return written


def check_dependencies() -> None:
    """Fail fast with a useful message if a required library is missing."""
    missing = []
    for pkg, install_name in [
        ("herbie", "herbie-data"),
        ("xarray", "xarray"),
        ("scipy", "scipy"),
        ("cfgrib", "cfgrib"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(install_name)
    if missing:
        sys.stderr.write(
            "Missing required packages: " + ", ".join(missing) + "\n"
            "Install with: pip install " + " ".join(missing) + "\n"
            "(eccodes binary may also be needed: `mamba install eccodes` "
            "or your OS package manager)\n"
        )
        sys.exit(1)
    # windpowerlib is optional (Tier-1 catalog); warn but don't fail
    try:
        __import__("windpowerlib")
    except ImportError:
        log.warning("windpowerlib not installed — falling back to NREL "
                    "generic curves only. `pip install windpowerlib` for "
                    "Tier-1 catalog matching.")


# ---------------------------------------------------------------------------
# Programmatic entry point
# ---------------------------------------------------------------------------

def run(uswtdb_path: str | Path,
        cycle_dt: datetime,
        fxx_range: Iterable[int],
        eia860_path: Optional[str | Path] = None,
        use_oedb: bool = True,
        output_dir: Optional[str | Path] = None,
        save_per_turbine: bool = False,
        aggregation_levels: Iterable[str] = ("ba", "iso"),
        ) -> dict[str, pd.DataFrame]:
    """End-to-end runner. Returns a dict of {level: aggregated_df}.

    Side effect: if `output_dir` is given, writes parquet/csv files there.
    """
    # Local imports so dep-check messages happen before module loads
    from wf_pipeline import build_inventory
    from hrrr_winds import fetch_hrrr_forecast, interp_to_points
    from forecast import forecast_fleet, ForecastConfig
    from aggregation import aggregate

    fxx_list = list(fxx_range)
    log.info("=" * 70)
    log.info("HRRR cycle:    %s UTC", cycle_dt.isoformat())
    log.info("Forecast hrs:  fxx=%d..%d (%d hours)",
             fxx_list[0], fxx_list[-1], len(fxx_list))
    log.info("USWTDB:        %s", uswtdb_path)
    log.info("EIA-860:       %s", eia860_path or "(not provided — state fallback)")
    log.info("oedb catalog:  %s", "enabled" if use_oedb else "disabled")
    log.info("=" * 70)

    # 1. Inventory
    t0 = time.time()
    inv = build_inventory(uswtdb_path, eia860_path, use_oedb=use_oedb)
    inv = inv.reset_index(drop=True)
    inv["point_id"] = inv.index
    log.info("Inventory built in %.1fs: %d turbines, %.0f MW",
             time.time() - t0, len(inv), inv["t_cap"].sum() / 1000.0)

    # 2. HRRR
    t0 = time.time()
    log.info("Fetching HRRR... (this is the slow part — minutes for many fxx)")
    ds = fetch_hrrr_forecast(cycle_dt, fxx_list)
    log.info("HRRR fetched in %.1fs", time.time() - t0)

    # 3. Spatial sampling
    t0 = time.time()
    samples = interp_to_points(
        ds,
        lats=inv["ylat"].to_numpy(),
        lons=inv["xlong"].to_numpy(),
    )
    log.info("Sampled HRRR at %d turbines × %d times in %.1fs (%d rows)",
             len(inv), len(fxx_list), time.time() - t0, len(samples))

    # 4. Per-turbine forecast
    t0 = time.time()
    fc = forecast_fleet(inv, samples, cfg=ForecastConfig())
    fleet_peak_MW = fc.groupby("valid_time")["power_net_kW"].sum().max() / 1000.0
    fleet_cap_MW = inv["t_cap"].sum() / 1000.0
    log.info("Forecast computed in %.1fs: peak %.0f MW / %.0f MW capacity (CF=%.0f%%)",
             time.time() - t0, fleet_peak_MW, fleet_cap_MW,
             100 * fleet_peak_MW / fleet_cap_MW)

    # 5. Aggregations
    results: dict[str, pd.DataFrame] = {}
    for level in aggregation_levels:
        results[level] = aggregate(fc, level=level)
        log.info("Aggregated to '%s': %d rows", level, len(results[level]))

    # 6. Output
    if output_dir is not None:
        outd = Path(output_dir)
        outd.mkdir(parents=True, exist_ok=True)
        cycle_tag = cycle_dt.strftime("%Y%m%dT%HZ")

        for level, df in results.items():
            csv_path = outd / f"forecast_{level}_{cycle_tag}.csv"
            df.to_csv(csv_path, index=False)
            log.info("Wrote %s (%d rows)", csv_path, len(df))

        # Per-region installed capacity, for the dashboard's reference lines.
        # Each aggregation level (iso, ba, etc) gets its own capacity file.
        capacity_files = _write_capacity_files(inv, outd, cycle_tag,
                                               aggregation_levels)
        for cap_path in capacity_files:
            log.info("Wrote %s", cap_path)

        # Curve assignment audit
        from curve_assignment import assignment_summary
        asum = assignment_summary(inv)
        asum.to_csv(outd / f"curve_assignment_{cycle_tag}.csv")

        if save_per_turbine:
            # Drop the curve_obj column (not serializable to CSV cleanly)
            cols = [c for c in fc.columns if c != "curve_obj"]
            pq_path = outd / f"forecast_per_turbine_{cycle_tag}.parquet"
            try:
                fc[cols].to_parquet(pq_path, index=False)
                log.info("Wrote %s (%d rows)", pq_path, len(fc))
            except ImportError:
                csv_path = outd / f"forecast_per_turbine_{cycle_tag}.csv"
                fc[cols].to_csv(csv_path, index=False)
                log.info("Wrote %s (%d rows; install pyarrow for parquet)",
                         csv_path, len(fc))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="US wind generation forecast (USWTDB + HRRR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--uswtdb", required=True,
                   help="Path to USWTDB CSV")
    p.add_argument("--eia860", default=None,
                   help="Path to EIA-860 Plant_Y<year>.xlsx for BA mapping")

    cyc = p.add_mutually_exclusive_group()
    cyc.add_argument("--cycle", default="latest",
                     help='HRRR cycle (UTC ISO or "latest")')
    cyc.add_argument("--backcast-date",
                     help="Backcast mode: a date YYYY-MM-DD; runs cycles "
                          "specified by --backcast-cycles")

    p.add_argument("--backcast-cycles", default="0",
                   help='Comma-separated UTC hours (e.g. "0,6,12,18"), '
                        'or "all" for hourly. Only used with --backcast-date')

    p.add_argument("--fxx-start", type=int, default=0,
                   help="First forecast hour")
    p.add_argument("--fxx-end", type=int, default=18,
                   help="Last forecast hour (inclusive)")

    p.add_argument("--output-dir", default="output/",
                   help="Directory for forecast output files")
    p.add_argument("--save-per-turbine", action="store_true",
                   help="Also save the per-turbine forecast (large file)")

    p.add_argument("--no-oedb", action="store_true",
                   help="Skip Tier-1 oedb catalog (use NREL generic only)")
    p.add_argument("--aggregation", default="ba,iso",
                   help="Comma-separated aggregation levels: "
                        "plant, ba, iso, state, national")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    setup_logging(args.verbose)
    check_dependencies()

    fxx_range = range(args.fxx_start, args.fxx_end + 1)
    levels = [s.strip() for s in args.aggregation.split(",") if s.strip()]

    if args.backcast_date:
        # Backcast mode: loop over multiple cycles in one day
        date = datetime.fromisoformat(args.backcast_date).replace(
            tzinfo=timezone.utc)
        if args.backcast_cycles == "all":
            hours = list(range(24))
        else:
            hours = [int(h) for h in args.backcast_cycles.split(",")]
        log.info("Backcast: %s UTC, cycles=%s", date.date(), hours)
        for h in hours:
            cycle_dt = date.replace(hour=h)
            try:
                run(uswtdb_path=args.uswtdb,
                    cycle_dt=cycle_dt,
                    fxx_range=fxx_range,
                    eia860_path=args.eia860,
                    use_oedb=not args.no_oedb,
                    output_dir=args.output_dir,
                    save_per_turbine=args.save_per_turbine,
                    aggregation_levels=levels)
            except Exception as e:
                log.error("Cycle %s failed: %s", cycle_dt, e)
                if args.verbose:
                    raise
    else:
        cycle_dt = parse_cycle(args.cycle)
        run(uswtdb_path=args.uswtdb,
            cycle_dt=cycle_dt,
            fxx_range=fxx_range,
            eia860_path=args.eia860,
            use_oedb=not args.no_oedb,
            output_dir=args.output_dir,
            save_per_turbine=args.save_per_turbine,
            aggregation_levels=levels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
