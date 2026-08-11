#!/usr/bin/env python3
"""C3S winter 2m-temperature outlook — re-based anomalies, skill-weighted.

Smarter than the raw multi-model mean, in three specific ways:

1. ANOMALY RE-BASING. Each model's anomaly is first taken against its OWN
   hindcast climatology (1993-2016, per start month and lead — removing bias
   and drift, the C3S convention), then re-expressed against a chosen
   OBSERVED base by the exact per-gridpoint, per-calendar-month shift
       anom_base = anom_hc + ERA5(1993-2016) - ERA5(base)
   Two bases are produced: 1991-2020 (WMO standard) and 2016-2025 (the
   recent-decade normal used on energy desks — which absorbs the trend by
   construction).

2. SKILL WEIGHTS BY LEAD. Every hindcast year is run through the identical
   anomaly pipeline and verified against ERA5; per (model, lead) the mean
   CRPS over years and the domain sets the weight w = (1/CRPS)^2, normalized
   across models at that lead. A model's month-1 credibility survives even if
   its month-6 forecasts are noise.

3. WEIGHTED MEMBER COUNTING. Probability maps are built by counting members,
   each carrying weight w_model / n_members_model, against ERA5 1991-2020
   terciles — no Gaussian assumptions.

Domain: North America, 15-72N x 170-50W at 1 deg. The August issue reaches
December (lead 5) and January (lead 6); February joins with the September
issue. Data: seasonal-monthly-single-levels t2m, hindcast + forecast, cached
GRIBs, same retrieval conventions as c3s_nino34 (data_format + grid, member x
init x step raveled and pooled by valid calendar month, so lagged burst
ensembles reduce identically to clean ones).

    python c3s_t2m_winter.py --fetch            # queue/refresh all downloads
    python c3s_t2m_winter.py --issue 202608     # build products
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c3s_nino34 as c3s

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "c3s_t2m"
ASSETS = c3s.ASSETS
ERA5 = HERE / "data" / "era5_t2m_mon.nc"

BOX, GRID = [72, -170, 15, -50], [1.0, 1.0]
HC_YEARS = [str(y) for y in range(1993, 2017)]
BASES = {"9120": (1991, 2020), "1625": (2016, 2025)}
TARGET_MONTHS = (12, 1, 2)
DATASET = "seasonal-monthly-single-levels"


def _retrieve_t2m(centre, system, years, month, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = {"originating_centre": centre, "system": system,
           "variable": "2m_temperature", "product_type": "monthly_mean",
           "year": list(years), "month": month,
           "leadtime_month": ["1", "2", "3", "4", "5", "6"],
           "area": BOX, "grid": GRID, "data_format": "grib"}
    for attempt in range(2):
        try:
            c3s._client().retrieve(DATASET, req, str(dest))
            return dest.exists() and dest.stat().st_size > 0
        except Exception as e:                            # noqa: BLE001
            msg = str(e).replace("\n", " ")
            if "no data" in msg.lower():
                return False
            print(f"    {centre}/{system} t2m {month}: attempt {attempt+1} "
                  f"failed ({msg[:90]})", file=sys.stderr)
            time.sleep(8)
    return False


def fetch(issue: str) -> None:
    month = issue[4:]
    for centre, system, label, _c in c3s.MODELS:
        hc = DATA / f"hc_{centre}_{system}_{month}.grib"
        fc = DATA / f"fc_{centre}_{system}_{issue}.grib"
        print(f"{label}: forecast …", flush=True)
        _retrieve_t2m(centre, system, [issue[:4]], month, fc)
        print(f"{label}: hindcast (24 yrs) …", flush=True)
        _retrieve_t2m(centre, system, HC_YEARS, month, hc)


def open_fields(path: Path):
    """DataArray of t2m with a raveled 'sample' dim carrying valid_time and
    init year — member x init x step flattened, so burst ensembles pool the
    same way c3s_nino34 pools its box series."""
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    da = ds[[v for v in ds.data_vars][0]] - 273.15
    core = [d for d in da.dims if d not in ("latitude", "longitude")]
    vtb = ds["valid_time"].broadcast_like(da.isel(latitude=0, longitude=0, drop=True))
    da = da.stack(sample=core)
    vt = vtb.stack(sample=core)
    return da, pd.to_datetime(np.asarray(vt.values))


def month_samples(da, vts, month, year=None):
    m = vts.month == month
    if year is not None:
        m &= (vts.year == year)
    idx = np.where(m)[0]
    sub = da.isel(sample=idx).transpose("sample", "latitude", "longitude").values
    good = np.isfinite(sub).all(axis=(1, 2))
    return sub[good]


def era5_monthly(base_y0, base_y1, month, lat, lon):
    ds = xr.open_dataset(ERA5)
    da = ds["t2m"].sel(time=(ds.time.dt.month == month) &
                       (ds.time.dt.year >= base_y0) & (ds.time.dt.year <= base_y1))
    clim = da.mean("time")
    if float(clim.max()) > 150:                # stored in K, not °C
        clim = clim - 273.15
    lon360 = np.where(lon < 0, lon + 360, lon)
    out = clim.interp(lat=("latitude", lat), lon=("longitude", lon360)).values
    ds.close()
    return out


def era5_year(month, year, lat, lon):
    ds = xr.open_dataset(ERA5)
    sel = ds["t2m"].sel(time=f"{year}-{month:02d}-01")
    if float(sel.max()) > 150:                 # stored in K, not °C
        sel = sel - 273.15
    lon360 = np.where(lon < 0, lon + 360, lon)
    out = sel.interp(lat=("latitude", lat), lon=("longitude", lon360)).values
    ds.close()
    return out


def crps_members(mem, y):
    """Empirical CRPS field: mem (n, ny, nx), y (ny, nx)."""
    t1 = np.abs(mem - y[None]).mean(axis=0)
    n = mem.shape[0]
    if n > 60:                                  # subsample the pair term
        sel = np.random.default_rng(0).choice(n, 60, replace=False)
        m2 = mem[sel]
    else:
        m2 = mem
    t2 = 0.5 * np.abs(m2[:, None] - m2[None, :]).mean(axis=(0, 1))
    return t1 - t2


def build(issue: str):
    month0 = int(issue[4:])
    rng = pd.date_range(f"{issue[:4]}-{month0:02d}-01", periods=6, freq="MS")
    targets = [(t.month, t.year, L + 1) for L, t in enumerate(rng)
               if t.month in TARGET_MONTHS]
    if not targets:
        raise SystemExit("no winter months within reach of this issue")
    print(f"issue {issue}: winter targets {[(m, y, 'L'+str(L)) for m, y, L in targets]}")

    per_model = {}
    weights = {}
    lat = lon = None
    for centre, system, label, _c in c3s.MODELS:
        hcp = DATA / f"hc_{centre}_{system}_{issue[4:]}.grib"
        fcp = DATA / f"fc_{centre}_{system}_{issue}.grib"
        if not (hcp.exists() and fcp.exists()):
            print(f"  {label}: data not ready — skipped")
            continue
        hc, hvt = open_fields(hcp)
        fc, fvt = open_fields(fcp)
        lat = hc.latitude.values
        lon = hc.longitude.values
        entry = {}
        for (m, yr, L) in targets:
            clim = month_samples(hc, hvt, m).mean(axis=0)          # hindcast climo
            fmem = month_samples(fc, fvt, m) - clim[None]          # anom vs hindcast
            if fmem.size == 0:
                continue
            # per-year hindcast verification for the weights
            obs_clim_hc = np.mean([era5_year(m, y2, lat, lon)
                                   for y2 in range(1993, 2017)], axis=0)
            crps_years = []
            for y2 in range(1993, 2017):
                hm = month_samples(hc, hvt, m, year=y2) - clim[None]
                if hm.size == 0:
                    continue
                oa = era5_year(m, y2, lat, lon) - obs_clim_hc
                crps_years.append(np.nanmean(crps_members(hm, oa)))
            entry[(m, L)] = dict(anom_members=fmem, crps=float(np.mean(crps_years)))
        if entry:
            per_model[label] = entry
            print(f"  {label}: {len(entry)} target(s), "
                  + ", ".join(f"m{m:02d} CRPS {v['crps']:.3f}" for (m, _l), v in entry.items()))

    # ── weights, shifts, products ──
    prods = {}
    for (m, yr, L) in targets:
        avail = {lab: e[(m, L)] for lab, e in per_model.items() if (m, L) in e}
        if len(avail) < 3:
            continue
        w = np.array([1.0 / max(avail[lab]["crps"], 1e-3) ** 2 for lab in avail])
        w /= w.sum()
        shift = {}
        clim_hc_obs = era5_monthly(1993, 2016, m, lat, lon)
        for tag, (y0, y1) in BASES.items():
            shift[tag] = clim_hc_obs - era5_monthly(y0, y1, m, lat, lon)
        # ERA5 terciles vs 1991-2020 (anomaly space, per gridpoint)
        yrs = np.arange(1991, 2021)
        obs_anoms = np.array([era5_year(m, y2, lat, lon) for y2 in yrs])
        obs_anoms = obs_anoms - obs_anoms.mean(axis=0)
        t_lo, t_hi = np.percentile(obs_anoms, [100 / 3, 200 / 3], axis=0)

        mean_maps = {}
        for tag in BASES:
            mm = np.array([avail[lab]["anom_members"].mean(axis=0) + shift[tag]
                           for lab in avail])
            mean_maps[tag] = (w[:, None, None] * mm).sum(axis=0)
        p_above = np.zeros_like(mean_maps["9120"])
        p_below = np.zeros_like(p_above)
        for wi, lab in zip(w, avail):
            mem = avail[lab]["anom_members"] + shift["9120"][None]
            p_above += wi * (mem > t_hi[None]).mean(axis=0)
            p_below += wi * (mem < t_lo[None]).mean(axis=0)
        prods[(m, yr, L)] = dict(mean=mean_maps, p_above=p_above,
                                 p_below=p_below, weights=dict(zip(avail, w.round(3))),
                                 nmem=sum(avail[lab]["anom_members"].shape[0] for lab in avail))
    return prods, lat, lon


MON = {12: "December", 1: "January", 2: "February"}


def render(issue, prods, lat, lon, out: Path):
    n = len(prods)
    fig, axes = plt.subplots(n, 4, figsize=(18.6, 4.3 * n),
                             constrained_layout=True,
                             subplot_kw={"projection": ccrs.LambertConformal(
                                 central_longitude=-100, central_latitude=45)})
    axes = np.atleast_2d(axes)
    for i, ((m, yr, L), P) in enumerate(sorted(prods.items(), key=lambda kv: (kv[0][1], kv[0][0]))):
        panels = [
            (P["mean"]["9120"], np.linspace(-4, 4, 17), "RdBu_r",
             f"{MON[m]} {yr} · anomaly vs 1991–2020 (°C)"),
            (P["mean"]["1625"], np.linspace(-4, 4, 17), "RdBu_r",
             f"{MON[m]} {yr} · anomaly vs 2016–2025 (°C)"),
            (100 * P["p_above"], np.arange(20, 91, 10), "YlOrRd",
             f"P(above normal) % · terciles vs 1991–2020"),
            (100 * P["p_below"], np.arange(20, 91, 10), "YlGnBu",
             f"P(below normal) %"),
        ]
        for j, (fld, levels, cmap, title) in enumerate(panels):
            ax = axes[i, j]
            cf = ax.contourf(lon, lat, fld, levels=levels, cmap=cmap,
                             extend="both" if j < 2 else "max",
                             transform=ccrs.PlateCarree())
            ax.set_extent([-168, -52, 17, 71], ccrs.PlateCarree())
            ax.coastlines(lw=0.6, color="0.25")
            ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.35", facecolor="none")
            ax.add_feature(cfeature.STATES, lw=0.25, edgecolor="0.55", facecolor="none")
            ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
            cb = fig.colorbar(cf, ax=ax, orientation="horizontal",
                              pad=0.02, fraction=0.05, aspect=32)
            cb.ax.tick_params(labelsize=7.5)
        wtxt = "  ".join(f"{k.split()[0]} {v:.2f}" for k, v in P["weights"].items())
        axes[i, 0].set_xlabel(f"lead {L} · {P['nmem']} weighted members · w: {wtxt}",
                              fontsize=7.5, color="0.3", labelpad=3)
    fig.suptitle(f"C3S winter 2 m temperature — issue {issue[:4]}-{issue[4:]} · "
                 "anomalies re-based to observed climates · CRPS-weighted members",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             "Each model anomalized vs its own 1993–2016 hindcast (bias+drift removed), re-based with exact ERA5 per-gridpoint monthly shifts · "
             "weights ∝ 1/CRPS² per (model, lead) from 24-yr hindcast verification vs ERA5 · probabilities by weighted member counting vs ERA5 terciles",
             fontsize=8.5, ha="center", color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105)
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=pd.Timestamp.utcnow().strftime("%Y%m"))
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default=str(ASSETS / "c3s_winter_t2m.webp"))
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch(args.issue)
        return
    prods, lat, lon = build(args.issue)
    if not prods:
        raise SystemExit("no products buildable (data still downloading?)")
    render(args.issue, prods, lat, lon, Path(args.out))
    meta = {(f"{m:02d}-{yr}"): {"lead": L, "weights": P["weights"], "nmem": P["nmem"]}
            for (m, yr, L), P in prods.items()}
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
