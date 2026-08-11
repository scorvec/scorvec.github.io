#!/usr/bin/env python3
"""C3S winter snowfall risk — P(exceed hindcast p90 / p95 / p99).

Extreme-snowfall risk maps for the systems that publish snowfall on the CDS.
Thresholds are each model's OWN hindcast percentiles (1993-2016, per valid
calendar month, per gridpoint, all members x years pooled) — model-consistent
by construction, so no observational re-basing is needed and no distribution
is assumed. The forecast risk is the member fraction exceeding the threshold,
averaged across available models; climatological neutral is 10 / 5 / 1 %.

Caveat printed on the figure: percentiles are of the 1993-2016 model climate —
in a warming winter climate the observed frequency of "p90 snow months" has
drifted; read the maps as risk relative to that reference, and lean on the
ratio-to-neutral rather than absolute numbers.

    python c3s_snow_winter.py --fetch --issue 202608
    python c3s_snow_winter.py --issue 202608
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c3s_nino34 as c3s
import xarray as xr
from c3s_t2m_winter import (DATA as T2M_DATA, BOX, GRID, HC_YEARS,
                            TARGET_MONTHS, DATASET, month_samples, MON)


def open_snow(path: Path):
    """Like c3s_t2m_winter.open_fields but WITHOUT the Kelvin offset (which
    annihilates ~1e-8 m w.e./s snowfall rates in float32) and scaled to
    mm w.e. / day."""
    import pandas as _pd
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[[v for v in ds.data_vars][0]].astype("float64") * 86400.0 * 1000.0
    core = [d for d in da.dims if d not in ("latitude", "longitude")]
    vtb = ds["valid_time"].broadcast_like(da.isel(latitude=0, longitude=0, drop=True))
    da = da.stack(sample=core)
    vt = vtb.stack(sample=core)
    return da, _pd.to_datetime(np.asarray(vt.values))

DATA = Path(__file__).resolve().parent / "data" / "c3s_snow"
ASSETS = c3s.ASSETS
PCTLS = (90, 95, 99)


def _retrieve_snow(centre, system, years, month, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = {"originating_centre": centre, "system": system,
           "variable": "snowfall", "product_type": "monthly_mean",
           "year": list(years), "month": month,
           "leadtime_month": ["1", "2", "3", "4", "5", "6"],
           "area": BOX, "grid": GRID, "data_format": "grib"}
    for attempt in range(2):
        try:
            c3s._client().retrieve(DATASET, req, str(dest))
            return dest.exists() and dest.stat().st_size > 0
        except Exception as e:                            # noqa: BLE001
            msg = str(e).replace("\n", " ")
            if "400" in msg or "no data" in msg.lower():
                print(f"    {centre}/{system}: snowfall not published — skipping",
                      file=sys.stderr)
                return False
            print(f"    {centre}/{system} snow: attempt {attempt+1} failed "
                  f"({msg[:90]})", file=sys.stderr)
            time.sleep(8)
    return False


def fetch(issue: str) -> None:
    month = issue[4:]
    for centre, system, label, _c in c3s.MODELS:
        fc = DATA / f"fc_{centre}_{system}_{issue}.grib"
        if not _retrieve_snow(centre, system, [issue[:4]], month, fc):
            continue
        print(f"{label}: snowfall forecast ✓; hindcast (24 yrs) …", flush=True)
        _retrieve_snow(centre, system, HC_YEARS, month,
                       DATA / f"hc_{centre}_{system}_{month}.grib")


def build(issue: str):
    month0 = int(issue[4:])
    rng = pd.date_range(f"{issue[:4]}-{month0:02d}-01", periods=6, freq="MS")
    targets = [(t.month, t.year, L + 1) for L, t in enumerate(rng)
               if t.month in TARGET_MONTHS]
    prods = {}
    lat = lon = None
    per_model = {}
    for centre, system, label, _c in c3s.MODELS:
        hcp = DATA / f"hc_{centre}_{system}_{issue[4:]}.grib"
        fcp = DATA / f"fc_{centre}_{system}_{issue}.grib"
        if not (hcp.exists() and fcp.exists()):
            continue
        hc, hvt = open_snow(hcp)
        fc, fvt = open_snow(fcp)
        lat, lon = hc.latitude.values, hc.longitude.values
        entry = {}
        for (m, yr, L) in targets:
            pool = month_samples(hc, hvt, m)
            fmem = month_samples(fc, fvt, m)
            if pool.size == 0 or fmem.size == 0:
                continue
            thr = {p: np.percentile(pool, p, axis=0) for p in PCTLS}
            entry[(m, L)] = {p: (fmem > thr[p][None]).mean(axis=0) for p in PCTLS}
            j, k = len(pool[0]) // 3, len(pool[0][0]) // 2
            print(f"    m{m:02d}: pool p90@probe={thr[90][j,k]:.3e}  "
                  f"fc mean@probe={fmem[:, j, k].mean():.3e}  "
                  f"domain P>p90 mean={(fmem > thr[90][None]).mean():.3f}", flush=True)
        if entry:
            per_model[label] = entry
            print(f"  {label}: {len(entry)} winter target(s)")
    for (m, yr, L) in targets:
        avail = [per_model[lab][(m, L)] for lab in per_model if (m, L) in per_model[lab]]
        if len(avail) >= 2:
            prods[(m, yr, L)] = {p: np.mean([a[p] for a in avail], axis=0)
                                 for p in PCTLS}
            prods[(m, yr, L)]["nmod"] = len(avail)
    return prods, lat, lon, sorted(per_model)


def render(issue, prods, lat, lon, models, out: Path):
    n = len(prods)
    fig, axes = plt.subplots(n, 3, figsize=(15.0, 4.3 * n),
                             constrained_layout=True,
                             subplot_kw={"projection": ccrs.LambertConformal(
                                 central_longitude=-100, central_latitude=45)})
    axes = np.atleast_2d(axes)
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 1))
    col_cf = [None] * 3
    neutral = {90: 10, 95: 5, 99: 1}
    for i, ((m, yr, L), P) in enumerate(sorted(prods.items(), key=lambda kv: (kv[0][1], kv[0][0]))):
        for j, p in enumerate(PCTLS):
            ax = axes[i, j]
            cf = ax.contourf(lon, lat, 100 * P[p],
                             levels=[neutral[p]*f for f in (0.5, 1, 1.5, 2, 3, 4, 6, 8)],
                             cmap="PuBuGn", extend="max", transform=ccrs.PlateCarree())
            ax.set_extent([-168, -52, 17, 71], ccrs.PlateCarree())
            ax.coastlines(lw=0.6, color="0.25")
            ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.35", facecolor="none")
            ax.add_feature(cfeature.STATES, lw=0.25, edgecolor="0.55", facecolor="none")
            ax.set_title(f"{MON[m][:3]} {yr} · P(snow > p{p})  [neutral {neutral[p]}%]",
                         fontsize=10.5, fontweight="bold", loc="left")
            col_cf[j] = cf
    for j, cf in enumerate(col_cf):
        if cf is not None:
            cb = fig.colorbar(cf, ax=list(axes[:, j]), orientation="horizontal",
                              pad=0.015, fraction=0.035, aspect=36, shrink=0.92)
            cb.ax.tick_params(labelsize=7.5)
    fig.suptitle(f"C3S winter snowfall extremes — issue {issue[:4]}-{issue[4:]} · "
                 f"{len(models)} systems", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.012,
             "Member fraction exceeding each model's OWN 1993–2016 hindcast percentile (per gridpoint, per month) · "
             + ", ".join(models) + " · no distribution assumed",
             fontsize=8.2, ha="center", color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105)
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=pd.Timestamp.utcnow().strftime("%Y%m"))
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default=str(ASSETS / "c3s_winter_snow.webp"))
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch(args.issue)
        return
    prods, lat, lon, models = build(args.issue)
    if not prods:
        raise SystemExit("no snow products buildable yet")
    render(args.issue, prods, lat, lon, models, Path(args.out))


if __name__ == "__main__":
    main()
