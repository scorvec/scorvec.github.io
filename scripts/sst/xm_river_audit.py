#!/usr/bin/env python3
"""Per-river XM inflow audit — is every river's catchment properly captured?

XM meters inflow (AporEner) river-by-river. This ranks all 44 active rivers
by their actual contribution to their region's inflow over the trailing year,
and classifies each river's catchment coverage in our REGION_SZH mapping
(build_hydro_regions.py) by matching the river name against the mapping's
provenance comments:

    mapped    named in exactly one subzona entry
    lumped    named in several entries / shares its subzona with other rivers
    UNMAPPED  not named anywhere — a genuine coverage gap

A big-share UNMAPPED or carelessly-lumped river is where polygon effort pays.

Outputs:
    ~/colombia_hydro/out/xm_river_audit.json
    colombia_hydro/river_audit.webp     (per-region bars, share of inflow)

    python scripts/sst/xm_river_audit.py [--days 365]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_region_rain import fetch_aportes, ORDER, COLORS
from build_hydro_regions import REGION_SZH

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RIVERS_JSON = Path.home() / "colombia_hydro" / "raw" / "xm_listado_rios.json"
OUT = Path.home() / "colombia_hydro" / "out"
OUT_PNG = REPO / "colombia_hydro" / "river_audit.webp"


def active_rivers() -> list[dict]:
    d = json.load(open(RIVERS_JSON))
    out = []
    for it in d["Items"]:
        for e in it["ListEntities"]:
            v = e["Values"]
            if v.get("Status") == "ACTIVO" and v["HydroRegion"] in ORDER:
                out.append(dict(name=v["Name"].strip().upper(),
                                code=v["Code"], region=v["HydroRegion"]))
    return out


def coverage_class(river: str, region: str) -> tuple[str, str]:
    """(class, note) by matching the river name against REGION_SZH comments."""
    hits = []
    for szh, note in REGION_SZH[region].items():
        up = note.upper()
        # match on the distinctive first token(s) of the river name
        key = river.replace(" CP", "").replace("DESV. ", "").split(" (")[0]
        toks = [t for t in key.split() if len(t) > 3] or [key]
        if any(t in up for t in toks):
            hits.append((szh, note))
    if not hits:
        # Ituango-style implicit coverage: intermediate-Cauca rivers
        if region == "ANTIOQUIA" and "ITUANGO" in river:
            hits = [(0, "intermediate Cauca set")]
    if not hits:
        return "UNMAPPED", "no subzona names this river"
    shared = any(";" in n or "," in n for _, n in hits)
    if len(hits) == 1 and not shared:
        return "mapped", f"SZH {hits[0][0]}"
    return "lumped", "; ".join(f"SZH {s}" for s, _ in hits[:3])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    a = ap.parse_args()
    rivers = active_rivers()
    d1 = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2)
    d0 = d1 - timedelta(days=a.days)
    print(f"fetching per-river AporEner {d0:%Y-%m-%d} … {d1:%Y-%m-%d} "
          f"({len(rivers)} rivers)", flush=True)
    apor = fetch_aportes(d0, d1)

    per_river: dict[str, list[float]] = {}
    for day in apor.values():
        for riv, kwh in day.items():
            per_river.setdefault(riv, []).append(kwh / 1e6)

    region_tot = {r: 0.0 for r in ORDER}
    rows = []
    for rv in rivers:
        vals = per_river.get(rv["name"], [])
        mean = float(np.mean(vals)) if vals else 0.0
        region_tot[rv["region"]] += mean
        cls, note = coverage_class(rv["name"], rv["region"])
        rows.append(dict(river=rv["name"], code=rv["code"], region=rv["region"],
                         mean_gwh_d=round(mean, 3), n_days=len(vals),
                         coverage=cls, note=note))
    unmetered = sorted(set(per_river) - {r["name"] for r in rivers})
    for r in rows:
        r["share_pct"] = round(100 * r["mean_gwh_d"] /
                               max(region_tot[r["region"]], 1e-9), 1)
    rows.sort(key=lambda r: (-r["share_pct"] if False else 0,))
    rows.sort(key=lambda r: (r["region"], -r["mean_gwh_d"]))

    print(f"{'region':<10} {'river':<26} {'GWh/d':>7} {'share':>6}  coverage")
    flags = []
    for r in rows:
        mark = "  ⚠" if r["coverage"] == "UNMAPPED" and r["share_pct"] >= 2 else ""
        print(f"{r['region']:<10} {r['river']:<26} {r['mean_gwh_d']:>7.2f} "
              f"{r['share_pct']:>5.1f}%  {r['coverage']}{mark}")
        if mark:
            flags.append(r)
    if unmetered:
        print("rivers in API data but NOT in ListadoRios ACTIVO:", unmetered)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xm_river_audit.json").write_text(json.dumps(
        dict(window=f"{d0:%Y-%m-%d}..{d1:%Y-%m-%d}",
             region_total_gwh_d={k: round(v, 2) for k, v in region_tot.items()},
             rivers=rows, unmatched_api_names=unmetered), indent=1))

    fig, axs = plt.subplots(2, 3, figsize=(13.2, 8.6))
    hatch = {"mapped": "", "lumped": "//", "UNMAPPED": "xx"}
    for ax, reg in zip(axs.ravel(), ORDER):
        rr = [r for r in rows if r["region"] == reg][::-1]
        ys = np.arange(len(rr))
        for y, r in zip(ys, rr):
            ax.barh(y, r["share_pct"], color=COLORS[reg],
                    alpha=0.85 if r["coverage"] == "mapped" else 0.55,
                    hatch=hatch[r["coverage"]], edgecolor="k", linewidth=0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([r["river"].title()[:22] for r in rr], fontsize=6.5)
        ax.set_title(f"{reg} — {region_tot[reg]:.1f} GWh/d", fontsize=9.5,
                     fontweight="bold", loc="left")
        ax.set_xlabel("share of region inflow (%)", fontsize=7.5)
        ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Which rivers carry the water? — XM per-river inflow, trailing year\n"
                 "solid = catchment mapped to its own subzona · hatched = lumped · "
                 "crosshatch = unmapped", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG.name} + xm_river_audit.json"
          + (f" · {len(flags)} coverage flag(s)" if flags else " · no coverage gaps ≥2%"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
