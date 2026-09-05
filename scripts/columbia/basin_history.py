"""A 66-year basin-aggregate record: ERA5 spliced onto Daymet.

Daymet starts in 1980, which gave four strong El Nino events -- few enough
that a composite over them said whatever those four happened to do. This
reaches back to 1959 using the local ERA5 store, which doubles the record and
brought the strong-event count to six. That was enough to overturn the
four-event reading: on 1980-2025 alone JFM was wet in 4 of 4 strong events at
+11 to +36%, and adding 1966 and 1973 cut the composite to +7% with 1973
going the other way at -23%.

**Why ERA5 is admissible here despite being poor for precipitation.** At a
point in the Cascades it is bad -- it cannot resolve orographic enhancement
or the rain shadow, and it runs 13% light on January precipitation. But
aggregated over the whole basin, 38 cells of its 1.5 degree grid, it tracks
Daymet at r = 0.989 on Oct-Mar totals and 0.98-0.99 in every calendar month.
The detail it gets wrong averages out; the synoptic year-to-year variability
it gets right is what survives. Those correlations are recomputed on every
run and stored, so the splice stays auditable rather than assumed.

**The hard limit: this is BASIN AGGREGATE only.** At 1.5 degrees many NWRFC
divisions are smaller than one grid cell, so nothing per-division can come
from the ERA5 half. Per-division history stays on Daymet's 1980-2025.

ERA5 is scaled to the Daymet level per calendar month over the 1980-2022
overlap before splicing, so the merged series has one level rather than a
step at 1980. Each month records which source it came from.

Needs the local ERA5 store (~/era5_store), so it runs on the laptop and its
output is committed; the Actions job never reruns it.

    python scripts/columbia/basin_history.py
-> columbia/data/basin_history.json
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P  # noqa: E402
import enso as E  # noqa: E402

STORE = os.path.expanduser("~/era5_store/wb2_1p5_daily")
OUT = os.path.join(P.DATA, "basin_history.json")
SPLICE = 1980                      # Daymet from here, scaled ERA5 before
MON = [10, 11, 12, 1, 2, 3]
NAME = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}


def basin_mask(lon, lat):
    """cos(lat) weights over the 1.5 degree cells inside the basin."""
    from shapely import contains_xy
    from shapely.ops import unary_union
    basin = unary_union(list(P.geoms().values()))
    lon180 = np.where(lon > 180, lon - 360, lon)
    LO, LA = np.meshgrid(lon180, lat, indexing="ij")
    m = contains_xy(basin, LO.ravel(), LA.ravel()).reshape(LO.shape)
    if m.sum() < 4:                                  # fall back to the box
        x0, y0, x1, y1 = basin.bounds
        m = (LO >= x0) & (LO <= x1) & (LA >= y0) & (LA <= y1)
    w = np.cos(np.radians(LA)) * m
    return w / w.sum(), int(m.sum())


def era5_monthly(var, agg):
    """{(year, month): value} from the local store, or {} if it is absent."""
    import xarray as xr
    files = sorted(glob.glob(os.path.join(STORE, var, f"{var}_*.nc")))
    if not files:
        return {}, 0
    d0 = xr.open_dataset(files[0])
    w, ncell = basin_mask(d0.longitude.values, d0.latitude.values)
    d0.close()
    daily = {}
    for f in files:
        try:
            d = xr.open_dataset(f)
        except Exception:
            continue
        if int(d.sizes.get("time", 0)) < 360:        # a partial year is dropped
            d.close(); continue
        v = d[var].values
        b = (v * w[None, :, :]).sum(axis=(1, 2))
        for t, x in zip(d["time"].values, b):
            k = np.datetime64(t, "D").astype("datetime64[D]").item()
            daily.setdefault((k.year, k.month), []).append(float(x))
        d.close()
    out = {k: (float(np.sum(v)) if agg == "sum" else float(np.mean(v)))
           for k, v in daily.items()}
    return out, ncell


def daymet_monthly():
    """Area-weighted basin monthly precip (mm) and mean temperature (C)."""
    src = os.path.join(P.DATA, "daymet_monthly.json")
    if not os.path.exists(src):
        return {}, {}
    dm = json.load(open(src)); area = P.area()
    pr, tp = {}, {}
    for c, rec in dm["div"].items():
        if c not in area:
            continue
        w = area[c]
        for i, (y, m) in enumerate(rec["index"]):
            a = pr.setdefault((y, m), [0.0, 0.0]); a[0] += w * rec["prcp"][i]; a[1] += w
            t = tp.setdefault((y, m), [0.0, 0.0])
            t[0] += w * (rec["tmax"][i] + rec["tmin"][i]) / 2.0; t[1] += w
    return ({k: v[0] / v[1] for k, v in pr.items()},
            {k: v[0] / v[1] for k, v in tp.items()})


def splice(era, day, mode):
    """Merge, scaling (precip) or offsetting (temperature) ERA5 onto Daymet."""
    both = sorted(set(era) & set(day))
    fit, r_by_month = {}, {}
    for m in range(1, 13):
        ks = [k for k in both if k[1] == m]
        if len(ks) < 10:
            continue
        e = np.array([era[k] for k in ks]); d = np.array([day[k] for k in ks])
        fit[m] = float(d.mean() / e.mean()) if mode == "scale" else float(d.mean() - e.mean())
        r_by_month[m] = round(float(np.corrcoef(e, d)[0, 1]), 3)
    out, src = {}, {}
    for k, v in era.items():
        if k[0] < SPLICE and k[1] in fit:
            out[k] = v * fit[k[1]] if mode == "scale" else v + fit[k[1]]
            src[k] = "era5"
    for k, v in day.items():
        out[k] = v; src[k] = "daymet"
    return out, src, r_by_month, len(both)


def composite(series, oni):
    """Oct-Mar monthly anomalies by ENSO phase, with n on every group."""
    def val(wy, m):
        return series.get((wy - 1 if m >= 10 else wy, m))
    years = sorted({(y + 1 if m >= 10 else y) for (y, m) in series})
    years = [y for y in years
             if all(val(y, m) is not None for m in MON) and E.winter(oni, y) is not None]
    if len(years) < 20:
        return None
    ndj = {y: E.winter(oni, y) for y in years}
    clim = {m: float(np.mean([val(y, m) for y in years])) for m in MON}
    groups = {"strong_el_nino": [y for y in years if ndj[y] >= 1.5],
              "moderate_el_nino": [y for y in years if 0.5 <= ndj[y] < 1.5],
              "neutral": [y for y in years if -0.5 < ndj[y] < 0.5],
              "la_nina": [y for y in years if ndj[y] <= -0.5]}
    out = {"years": [years[0], years[-1]], "n_years": len(years),
           "climatology_mm": {NAME[m]: round(clim[m], 1) for m in MON}, "groups": {}}
    for lab, sel in groups.items():
        if not sel:
            continue
        pct = {NAME[m]: round(100 * (float(np.mean([val(y, m) for y in sel])) / clim[m] - 1), 1)
               for m in MON}
        seas = {}
        for nm, ms in (("OND", (10, 11, 12)), ("JFM", (1, 2, 3))):
            a = float(np.mean([sum(val(y, m) for m in ms) for y in sel]))
            seas[nm] = round(100 * (a / sum(clim[m] for m in ms) - 1), 1)
        out["groups"][lab] = {"n": len(sel), "years": sorted(sel),
                              "pct_of_normal": pct, "season_pct": seas,
                              "per_year": {str(y): {nm: round(100 * (sum(val(y, m) for m in ms)
                                                    / sum(clim[m] for m in ms) - 1), 1)
                                                    for nm, ms in (("OND", (10, 11, 12)),
                                                                   ("JFM", (1, 2, 3)))}
                                           for y in sorted(sel)}}
    return out


def main():
    day_p, day_t = daymet_monthly()
    if not day_p:
        print("  no daymet_monthly.json -- run daymet.py first"); return
    era_p, ncell = era5_monthly("prcp", "sum")
    era_t, _ = era5_monthly("t2m", "mean")
    if not era_p:
        print(f"  no ERA5 store at {STORE} -- nothing to splice"); return
    print(f"  ERA5 basin mask: {ncell} cells at 1.5 deg")

    pr, pr_src, pr_r, n_ov = splice(era_p, day_p, "scale")
    tp, tp_src, tp_r, _ = splice({k: v - 273.15 for k, v in era_t.items()}, day_t, "offset")
    print(f"  precip overlap {n_ov} months, r by month "
          + " ".join(f"{NAME[m]}:{pr_r[m]}" for m in MON if m in pr_r))
    worst = min((pr_r[m] for m in pr_r), default=1.0)
    if worst < 0.9:
        print(f"  WARNING: worst monthly r is {worst}; the splice is not safe below ~0.9")

    oni = E.psl_series("oni")
    doc = {"built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "scope": "WHOLE-BASIN aggregate only — ERA5 at 1.5 deg cannot resolve a division",
           "sources": {"pre_1980": "ERA5 (WeatherBench2 1.5 deg daily), scaled to the Daymet "
                                   "level per calendar month over the 1980-2022 overlap",
                       "from_1980": "Daymet V4 R1, area weighted over the NWRFC divisions"},
           "overlap_months": n_ov, "precip_r_by_month": pr_r, "temp_r_by_month": tp_r,
           "era5_cells": ncell,
           "monthly": {f"{y}-{m:02d}": {"prcp": round(pr[(y, m)], 1),
                                        "temp": (round(tp[(y, m)], 2) if (y, m) in tp else None),
                                        "src": pr_src[(y, m)]}
                       for (y, m) in sorted(pr)},
           "enso_composite": composite(pr, oni)}
    json.dump(doc, open(OUT, "w"), separators=(",", ":"))

    c = doc["enso_composite"]
    print(f"\n  {c['n_years']} water years {c['years'][0]}-{c['years'][1]}")
    print(f"  {'group':18s} {'n':>3s} " + " ".join(f"{NAME[m]:>5s}" for m in MON)
          + f" {'OND':>6s} {'JFM':>6s}")
    for lab, g in c["groups"].items():
        print(f"  {lab:18s} {g['n']:3d} "
              + " ".join(f"{g['pct_of_normal'][NAME[m]]:+5.0f}" for m in MON)
              + f" {g['season_pct']['OND']:+6.0f} {g['season_pct']['JFM']:+6.0f}")
    print(f"\n  wrote {OUT} ({os.path.getsize(OUT)/1000:.0f} kB)")


if __name__ == "__main__":
    main()
