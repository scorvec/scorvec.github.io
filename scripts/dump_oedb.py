"""Dump the windpowerlib turbine catalog to see exact turbine_type strings."""
import sys
import warnings
warnings.filterwarnings("ignore")

from windpowerlib import get_turbine_types

df = get_turbine_types(print_out=False)
print(f"Total turbine types: {len(df)}")
print()

# Group by manufacturer prefix in turbine_type
print(f"{'manufacturer':<35s} turbine_type")
print("-" * 80)
for mfr in sorted(df["manufacturer"].dropna().unique()):
    sub = df[df["manufacturer"] == mfr].sort_values("turbine_type")
    for _, row in sub.iterrows():
        print(f"{mfr:<35s} {row['turbine_type']}")
    print()
