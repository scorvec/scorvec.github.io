#!/usr/bin/env python3
"""GEPS weeks 3-5 precipitation anomaly maps for Brazil.

Weekly totals from the extended GEPS (Mon/Thu 00Z) via the four
week-boundary accumulation files (P336/504/672/840 — APCP accumulates
from init, so week k total = acc(end) - acc(start), ensemble-meaned).
Anomalies vs the IMERG harmonic daily climatology summed over the same
valid days, interpolated to the GEPS 0.5-deg grid.

NOTE: anomalies are vs OBSERVED climatology — the GEPS reforecast
(model) climatology lives in the S2S archive behind a login, so model
drift/bias at these leads is NOT removed.  Numbers per basin are the
area-averaged anomaly (mm/week).

    python scripts/sst/geps_wk35_anom.py [--date 20260824]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brazil_model import basin_weights, MAJORS              # noqa: E402
from build_imerg_clim import OUT as CLIM_NC, eval_clim      # noqa: E402
from geps35_extract import fetch_lead, ARCH                 # noqa: E402
from brazil_rain_daily import basin_paths, centroid, LABEL_XY  # noqa: E402

REPO = HERE.parent.parent
OUTPNG = REPO / "brazil_hydro" / "geps_wk35_anom.webp"
BOX = dict(lon0=-76.0, lon1=-33.0, lat0=-35.0, lat1=6.0)
WEEK_BOUNDS = [336, 504, 672, 840]                          # h; wk3 wk4 wk5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    args = ap.parse_args()
    date = args.date or max(f.name.split("_")[1]
                            for f in ARCH.glob("geps_*_00z.json.gz"))
    print(f"cycle {date} 00Z")

    acc, lons, lats = {}, None, None
    for lead in WEEK_BOUNDS:
        g = fetch_lead(date, lead)
        if g is None:
            raise SystemExit(f"lead {lead} unavailable on Datamart")
        acc[lead], lons, lats = g
    weekly = {}                                             # wk# -> (ny, nx) mm
    for wi, (a, b) in enumerate(zip(WEEK_BOUNDS[:-1], WEEK_BOUNDS[1:])):
        weekly[wi + 3] = np.clip(acc[b] - acc[a], 0, None).mean(axis=0)

    # Anomaly baseline: GEPS S2S reforecast climatology (late-Aug starts
    # 2016-23, each pre-averaged over ~23 hindcast years x 3 members at IRI),
    # so lead-dependent model drift is removed.  tp is accumulated from
    # init on L=1..32 days at 1.5 deg; the reforecast horizon ends at day
    # 32, so week 5 (days 29-35) uses the days 26-32 climatological rate.
    import xarray as xr
    refc = Path.home() / "brazil_hydro" / "raw" / "geps_refc"
    accs, rl_lat, rl_lon = [], None, None
    for f in sorted(refc.glob("tp_*.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        da = ds["tp"].rename({"Y": "lat", "X": "lon"})
        da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon") \
               .sortby("lat")
        accs.append(da)
    if not accs:
        raise SystemExit("no reforecast files in geps_refc/ — run the IRI pull")
    acc = xr.concat(accs, "y").mean("y")                   # (L1, lat, lon)
    acc05 = acc.interp(lat=lats, lon=lons,
                       kwargs={"fill_value": None}).values  # 1.5 -> 0.5 deg
    d0 = datetime.strptime(date, "%Y%m%d").toordinal()
    clim_w = {3: acc05[20] - acc05[13],                    # days 15-21
              4: acc05[27] - acc05[20],                    # days 22-28
              5: acc05[31] - acc05[24]}                    # days 26-32 rate ~ wk5

    W = basin_weights(lons, lats, set(MAJORS))
    paths = basin_paths()
    cents = {b: LABEL_XY.get(b, centroid(r)) for b, r in paths.items()
             if b in MAJORS}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 11.2),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.02,
                        wspace=0.04, hspace=0.10)
    nrm = TwoSlopeNorm(vcenter=0, vmin=-50, vmax=50)
    nrm_t = TwoSlopeNorm(vcenter=0, vmin=-120, vmax=120)
    total_an = np.zeros_like(weekly[3])
    for ax, wk in zip(axes.flat[:3], (3, 4, 5)):
        an = weekly[wk] - clim_w[wk]
        total_an = total_an + an
        v0 = datetime.fromordinal(d0 + (wk - 1) * 7).strftime("%m-%d")
        v1 = datetime.fromordinal(d0 + wk * 7 - 1).strftime("%m-%d")
        ax.contourf(lons, lats, an, levels=np.linspace(-50, 50, 21),
                    cmap="BrBG", norm=nrm, extend="both",
                    transform=ccrs.PlateCarree())
        ax.coastlines(resolution="50m", lw=0.7, color="#333")
        for b, rings in paths.items():
            if b not in MAJORS:
                continue
            for rg in rings:
                ax.plot(rg[:, 0], rg[:, 1], color="#222", lw=0.6,
                        transform=ccrs.PlateCarree())
        for b in MAJORS:
            av = float(np.nansum(an * W[b]))
            x, y = cents[b]
            ax.text(x, y, f"{b.title()}\n{av:+.0f}",
                    transform=ccrs.PlateCarree(), fontsize=5.8, ha="center",
                    va="center", fontweight="bold",
                    bbox=dict(fc="white", alpha=0.7, ec="none", pad=0.7))
        ax.set_extent([BOX[k] for k in ("lon0", "lon1", "lat0", "lat1")])
        ax.set_title(f"week {wk}  ({v0} → {v1})", fontsize=9.5, loc="left",
                     fontweight="bold")
    ax = axes.flat[3]
    ax.contourf(lons, lats, total_an, levels=np.linspace(-120, 120, 21),
                cmap="BrBG", norm=nrm_t, extend="both",
                transform=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", lw=0.7, color="#333")
    for b in MAJORS:
        for rg in paths[b]:
            ax.plot(rg[:, 0], rg[:, 1], color="#222", lw=0.6,
                    transform=ccrs.PlateCarree())
        av = float(np.nansum(total_an * W[b]))
        x, y = cents[b]
        ax.text(x, y, f"{b.title()}\n{av:+.0f}", transform=ccrs.PlateCarree(),
                fontsize=5.8, ha="center", va="center", fontweight="bold",
                bbox=dict(fc="white", alpha=0.7, ec="none", pad=0.7))
    ax.set_extent([BOX[k] for k in ("lon0", "lon1", "lat0", "lat1")])
    ax.set_title("weeks 3-5 total", fontsize=9.5, loc="left",
                 fontweight="bold")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="BrBG"),
                      ax=[axes.flat[1]], fraction=0.04, pad=0.012)
    cb.set_label("mm/week vs climatology", fontsize=8)
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=nrm_t, cmap="BrBG"),
                       ax=[axes.flat[3]], fraction=0.04, pad=0.012)
    cb2.set_label("mm (3-week total)", fontsize=8)
    fig.suptitle(f"GEPS weeks 3-5 rainfall anomaly — {date[:4]}-{date[4:6]}-"
                 f"{date[6:]} 00Z, 21-member mean\nvs GEPS S2S reforecast "
                 "climatology (late-Aug starts, ~1,400 hindcast runs) — "
                 "drift-corrected; basin numbers = area-avg mm",
                 fontsize=11, fontweight="bold")
    fig.savefig(OUTPNG, dpi=115, bbox_inches="tight")
    print(f"wrote {OUTPNG.name}")


if __name__ == "__main__":
    main()
