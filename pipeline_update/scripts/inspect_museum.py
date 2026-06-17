"""Diagnose the American Wind Power Center plant entry in USWTDB.

Usage:
    python3 inspect_museum.py [path/to/uswtdb.csv]
"""
import sys
import pandas as pd

uswtdb_path = sys.argv[1] if len(sys.argv) > 1 else "data/uswtdb.csv"
df = pd.read_csv(uswtdb_path, low_memory=False)

# Find by name
hits = df[df["p_name"].astype(str).str.contains(
    "Wind Power Center|Windmill|Museum", case=False, na=False)]
if hits.empty:
    print("No name match — searching by Lubbock coordinates instead...")
    hits = df[(df["t_state"] == "TX") &
              (df["xlong"].between(-101.95, -101.75)) &
              (df["ylat"].between(33.55, 33.65))]

print(f"\n{len(hits)} matching turbine rows\n")
for eia, g in hits.groupby("eia_id", dropna=False):
    print(f"--- eia_id={eia} ---")
    print(f"  p_name(s): {g['p_name'].unique().tolist()}")
    print(f"  turbines:  {len(g)}")
    print(f"  total cap: {g['t_cap'].sum()/1000:.2f} MW")
    if "t_model" in g.columns:
        print(f"  models:    {sorted(g['t_model'].dropna().unique().tolist())[:5]}")
    if "p_year" in g.columns:
        years = sorted(g['p_year'].dropna().unique().astype(int).tolist())
        print(f"  years:     {years}")
    print(f"  bbox:      lon [{g['xlong'].min():.4f}, {g['xlong'].max():.4f}]  "
          f"lat [{g['ylat'].min():.4f}, {g['ylat'].max():.4f}]")
    print()
