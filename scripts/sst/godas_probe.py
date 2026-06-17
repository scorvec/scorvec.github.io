#!/usr/bin/env python3
"""Quick probe: what's the latest GODAS month, and what are the var/coord
names? Run this first to confirm structure before the full render."""
import sys
from datetime import datetime, timezone
import pandas as pd
import xarray as xr

BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas"
year = datetime.now(timezone.utc).year

ds = None
for y in (year, year - 1):
    url = f"{BASE}/pottmp.{y}.nc"
    try:
        print(f"Trying {url} ...")
        ds = xr.open_dataset(url)
        print(f"  opened {y}")
        break
    except Exception as e:
        print(f"  failed {y}: {e}")

if ds is None:
    print("Could not open GODAS pottmp from PSL.")
    sys.exit(1)

print("\n=== DATASET STRUCTURE ===")
print(ds)

print("\n=== DATA VARIABLES ===")
for v in ds.data_vars:
    print(f"  {v}: dims={ds[v].dims}, shape={ds[v].shape}")

print("\n=== COORDS ===")
for c in ds.coords:
    vals = ds[c].values
    try:
        print(f"  {c}: {vals.min()} .. {vals.max()}  (n={vals.size})")
    except Exception:
        print(f"  {c}: {vals[:5]} ...")

t = pd.to_datetime(ds["time"].values)
print(f"\n=== TIME ===")
print(f"  range: {t[0]:%Y-%m-%d} -> {t[-1]:%Y-%m-%d}  ({len(t)} steps)")
print(f"  MOST RECENT MONTH AVAILABLE: {t[-1]:%B %Y}")

# Show depth-like coordinate values
for name in ("level", "depth", "lev", "z"):
    if name in ds.coords or name in ds.dims:
        print(f"\n=== DEPTH ('{name}') levels (m) ===")
        print(f"  {ds[name].values}")
        break

ds.close()
