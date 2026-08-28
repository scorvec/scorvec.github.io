"""
Compute and print the curve assignment summary without running the full
forecast pipeline. Useful for quickly checking how the new oedb alias
table and 12-bin generic curves are covering your fleet.

Usage:
    python3 curve_assignment_check.py
    python3 curve_assignment_check.py --states TX
    python3 curve_assignment_check.py --uswtdb data/uswtdb.csv --states TX OK NM
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uswtdb", default=str(SCRIPT_DIR / "data" / "uswtdb.csv"),
                   help="Path to USWTDB CSV")
    p.add_argument("--states", nargs="*", default=None,
                   help="Filter by state(s); default = all")
    p.add_argument("--output", default=None,
                   help="Optional CSV path for full per-turbine assignment")
    p.add_argument("--verbose", action="store_true",
                   help="Show oedb fuzzy-match log lines")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    from turbine_inventory import load_uswtdb
    from curve_assignment import assign_curves, assignment_summary

    print(f"Loading USWTDB from {args.uswtdb} ...")
    df = load_uswtdb(args.uswtdb)
    print(f"  {len(df):,} turbines after filtering")

    if args.states:
        df = df[df["t_state"].isin(args.states)].copy()
        print(f"  {len(df):,} turbines in {args.states}")

    print("Assigning curves (this loads windpowerlib catalog on first call)...")
    df = assign_curves(df, use_oedb=True)

    summary = assignment_summary(df)
    print()
    print("=" * 80)
    print("Curve assignment summary")
    print("=" * 80)
    # Force pandas to print all rows
    with pd.option_context("display.max_rows", None,
                           "display.float_format", "{:,.1f}".format):
        print(summary)

    # Top-level breakdown: oedb vs generic vs manual
    df["tier"] = df["curve_source"].apply(
        lambda s: ("oem" if s.startswith("oem:")
                   else "oedb" if s.startswith("oedb:")
                   else "generic" if s.startswith("generic_v2") or s.startswith("nrel_generic")
                   else "manual" if s.startswith("manual:")
                   else "other"))
    tier_summary = (df.groupby("tier")
                      .agg(n_turbines=("tier", "size"),
                           total_MW=("t_cap", lambda s: s.sum() / 1000.0))
                      .sort_values("total_MW", ascending=False))
    tier_summary["pct_MW"] = 100.0 * tier_summary["total_MW"] / tier_summary["total_MW"].sum()
    print()
    print("=" * 80)
    print("By tier")
    print("=" * 80)
    print(tier_summary.to_string(float_format=lambda x: f"{x:,.1f}"))

    # Top 20 generic-bin entries by MW (those are the most-improvable ones)
    generic = df[df["tier"] == "generic"]
    if not generic.empty:
        top_generic = (generic.groupby(["t_manu", "t_model"])
                              .agg(n=("case_id", "size"),
                                   mw=("t_cap", lambda s: s.sum() / 1000.0),
                                   curve=("curve_source", "first"))
                              .sort_values("mw", ascending=False)
                              .head(20))
        print()
        print("=" * 80)
        print("Top 20 generic-bin models by MW (still falling back, "
              "candidates for new alias entries):")
        print("=" * 80)
        print(top_generic.to_string(float_format=lambda x: f"{x:,.1f}"))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote per-turbine detail to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
