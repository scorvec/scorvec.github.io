#!/usr/bin/env python3
"""SFS beta DAILY products: L48 degree days and daily teleconnections.

The daily stores carry all 31 members for the first 47 days at 1°, with a
matching 30-year × 11-member daily reforecast — enough for member-resolved
degree-day traces and daily teleconnection indices judged against the
model's own hindcast climatology.

Products (feed: assets/sfs/data/sfs_daily.json):
  * L48 HDD/CDD — population-weighted (built-in metro list, base 65 °F)
    daily degree days per member, with the hindcast climatological
    percentile band per lead day (same init month, 330 samples/day).
  * Daily teleconnections — AO (projection onto EOF1 of the model's own
    reforecast 1000-hPa height anomalies, 20-90°N, computed at 2°),
    NAO (Hurrell-style MSLP dipole Azores−Iceland), PNA (Wallace-Gutzler
    500-hPa centers), EPO (500-hPa dipole, mid-Pacific south minus
    Alaska/Bering north). All standardized per lead day against the
    reforecast climatology, so member values are in hindcast-σ units and
    P(index<0 | members) is directly comparable to the 50% base rate.

Climatology cache (one ~9 GB reforecast stream per init month, then free):
scripts/sfs/data/daily_climo_{MM}.npz.

    python scripts/sfs/sfs_daily.py [--issue 202608] [--climo-only]
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "assets" / "sfs" / "data" / "sfs_daily.json"
CLIMDIR = HERE / "data"
BASE = "https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/beta1"
CLIM_Y0, CLIM_Y1 = 1991, 2020
NDAYS = 47

# ── population weights: top US metros (approx 2020 CBSA pop, millions) ──────
METROS = [
    ("NYC", 40.7, -74.0, 19.8), ("LA", 34.05, -118.2, 13.0),
    ("Chicago", 41.9, -87.6, 9.4), ("Dallas", 32.8, -96.8, 7.6),
    ("Houston", 29.8, -95.4, 7.1), ("DC", 38.9, -77.0, 6.4),
    ("Philadelphia", 40.0, -75.2, 6.2), ("Atlanta", 33.7, -84.4, 6.1),
    ("Miami", 25.8, -80.2, 6.1), ("Phoenix", 33.4, -112.1, 4.8),
    ("Boston", 42.4, -71.1, 4.9), ("SF", 37.8, -122.4, 4.7),
    ("Riverside", 34.0, -117.4, 4.6), ("Detroit", 42.3, -83.0, 4.4),
    ("Seattle", 47.6, -122.3, 4.0), ("Minneapolis", 44.98, -93.3, 3.7),
    ("SanDiego", 32.7, -117.2, 3.3), ("Tampa", 27.9, -82.5, 3.2),
    ("Denver", 39.7, -105.0, 3.0), ("StLouis", 38.6, -90.2, 2.8),
    ("Baltimore", 39.3, -76.6, 2.8), ("Charlotte", 35.2, -80.8, 2.7),
    ("Orlando", 28.5, -81.4, 2.7), ("SanAntonio", 29.4, -98.5, 2.6),
    ("Portland", 45.5, -122.7, 2.5), ("Sacramento", 38.6, -121.5, 2.4),
    ("Pittsburgh", 40.4, -80.0, 2.4), ("LasVegas", 36.2, -115.1, 2.3),
    ("Austin", 30.3, -97.7, 2.3), ("Cincinnati", 39.1, -84.5, 2.3),
    ("KansasCity", 39.1, -94.6, 2.2), ("Columbus", 40.0, -83.0, 2.1),
    ("Indianapolis", 39.8, -86.2, 2.1), ("Cleveland", 41.5, -81.7, 2.1),
    ("Nashville", 36.2, -86.8, 2.0), ("SanJose", 37.3, -121.9, 2.0),
    ("VirginiaBeach", 36.8, -76.1, 1.8), ("Providence", 41.8, -71.4, 1.7),
    ("Milwaukee", 43.0, -87.9, 1.6), ("Jacksonville", 30.3, -81.7, 1.6),
    ("OklahomaCity", 35.5, -97.5, 1.4), ("Raleigh", 35.8, -78.6, 1.4),
    ("Memphis", 35.1, -90.0, 1.3), ("Richmond", 37.5, -77.5, 1.3),
    ("NewOrleans", 30.0, -90.1, 1.3), ("Louisville", 38.3, -85.8, 1.3),
    ("SaltLake", 40.8, -111.9, 1.3), ("Hartford", 41.8, -72.7, 1.2),
    ("Buffalo", 42.9, -78.9, 1.2), ("Birmingham", 33.5, -86.8, 1.1),
]

# teleconnection point/box definitions (lon in 0-360)
NAO_S = (38.0, 334.0)      # Azores (Ponta Delgada)
NAO_N = (65.0, 338.0)      # Iceland (Reykjavik/Stykkisholmur)
PNA_PTS = [(20, 200, +1), (45, 195, -1), (55, 245, +1), (30, 275, -1)]
EPO_S = (20, 35, 200, 235)   # latmin, latmax, lonmin, lonmax  (south box, +)
EPO_N = (55, 65, 200, 235)   # north box, −
AO_DOM = (20, 90)            # 1000-hPa EOF domain, all longitudes, at 2°


def _open(url):
    import fsspec
    import xarray as xr
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True)


def _metro_idx(lat, lon):
    ii, jj, ww = [], [], []
    for _n, la, lo, pop in METROS:
        ii.append(int(np.argmin(np.abs(lat - la))))
        jj.append(int(np.argmin(np.abs(lon - (lo % 360)))))
        ww.append(pop)
    w = np.array(ww); w = w / w.sum()
    return np.array(ii), np.array(jj), w


def _degree_days(t2m_k, ii, jj, w):
    """(..., day) population-weighted HDD and CDD from Kelvin daily means."""
    tf = (t2m_k[..., ii, jj] - 273.15) * 9 / 5 + 32          # (..., nmetro)
    hdd = np.clip(65.0 - tf, 0, None) @ w
    cdd = np.clip(tf - 65.0, 0, None) @ w
    return hdd, cdd


def _pna(z500, lat, lon):
    v = 0.0
    for la, lo, s in PNA_PTS:
        v = v + s * z500[..., np.argmin(np.abs(lat - la)),
                         np.argmin(np.abs(lon - lo))]
    return 0.25 * v


def _box(z, lat, lon, b):
    la = (lat >= b[0]) & (lat <= b[1])
    lo = (lon >= b[2]) & (lon < b[3])
    w = np.cos(np.deg2rad(lat[la]))[:, None] * np.ones((la.sum(), lo.sum()))
    return np.nansum(z[..., la, :][..., lo] * w, axis=(-2, -1)) / w.sum()


def _coarse2(field):
    """1° -> 2° block mean over the trailing lat/lon dims (180 lat kept)."""
    f = field[..., :180, :]
    s = f.shape
    return f.reshape(*s[:-2], 90, 2, 180, 2).mean(axis=(-3, -1))


def _vt100_band(v, t, lat):
    """zonal-mean 100-hPa eddy heat flux, cos-weighted 45-75N mean."""
    ve = v - v.mean(axis=-1, keepdims=True)
    te = t - t.mean(axis=-1, keepdims=True)
    vt = (ve * te).mean(axis=-1)
    la = (lat >= 45) & (lat <= 75)
    w = np.cos(np.deg2rad(lat[la]))
    return (vt[..., la] * w).sum(axis=-1) / w.sum()


def build_strat_climo(month: int) -> dict:
    """100-hPa v'T' and 50-hPa 60°N zonal wind hindcast samples, cached."""
    f = CLIMDIR / f"daily_climo_strat_{month:02d}.npz"
    if f.exists():
        return dict(np.load(f))
    ds = _open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
    ds = ds.sel(init=slice(str(CLIM_Y0), str(CLIM_Y1)))
    lat = ds.lat.values
    j60 = int(np.argmin(np.abs(lat - 60)))
    n_init, n_mem = ds.sizes["init"], ds.sizes["member"]
    vt = np.zeros((n_init * n_mem, NDAYS), np.float32)
    u50 = np.zeros_like(vt)
    k = 0
    for yi in range(n_init):
        v = ds.VGRD_100mb.isel(init=yi).values
        t = ds.TMP_100mb.isel(init=yi).values
        u = ds.UGRD_50mb.isel(init=yi).values
        for mi in range(n_mem):
            vt[k] = _vt100_band(v[mi], t[mi], lat)
            u50[k] = u[mi][..., j60, :].mean(axis=-1)
            k += 1
        print(f"strat climo: init {yi + 1}/{n_init}", flush=True)
    out = {"vt100": vt, "u50": u50}
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    print(f"strat climo {month:02d}: cached ({k} samples/day)", flush=True)
    return out


def build_field_climo(month: int) -> dict:
    """Per-lead-day fields from the daily reforecast: pooled mean and σ
    (the anomaly yardstick) AND the hindcast's own mean ENSEMBLE spread as
    a function of lead (the spread-ratio denominator — comparing today's
    member spread to pooled climatological σ conflates interannual
    variance with ensemble dispersion and trends below 1 by construction)."""
    f = CLIMDIR / f"field_climo3_{month:02d}.npz"
    if f.exists():
        return dict(np.load(f))
    ds = _open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
    ds = ds.sel(init=slice(str(CLIM_Y0), str(CLIM_Y1)))
    years = ds.init.dt.year.values.astype(np.float64)
    tbar = years.mean()
    stt = ((years - tbar) ** 2).sum()
    n_init, n_mem = ds.sizes["init"], ds.sizes["member"]
    acc = {"tbar": np.float64(tbar)}
    for var, key in (("TMP_2maboveground", "t2m"), ("HGT_500mb", "z500")):
        s1 = np.zeros((NDAYS, 181, 360), np.float64)
        s2 = np.zeros_like(s1)
        ev = np.zeros_like(s1)                     # Σ_years var_members (ddof=1)
        sm2 = np.zeros_like(s1)                    # Σ_years (member-mean)²
        smt = np.zeros_like(s1)                    # Σ_years (t−t̄)·member-mean
        for yi in range(n_init):
            v = ds[var].isel(init=yi).values.astype(np.float64)  # (11,47,181,360)
            m = v.mean(axis=0)
            s1 += v.sum(axis=0)
            s2 += (v * v).sum(axis=0)
            ev += v.var(axis=0, ddof=1)
            sm2 += m * m
            smt += (years[yi] - tbar) * m
            if yi % 10 == 0:
                print(f"field climo {key}: init {yi + 1}/{n_init}", flush=True)
        n = n_init * n_mem
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu * mu, 1e-9) * n / (n - 1))
        esd = np.sqrt(ev / n_init)                 # RMS ensemble spread per lead
        # hindcast trend: per-gridpoint slope of yearly ensemble means, pooled
        # over all leads (the climate trend is lead-independent; 47x more
        # samples than a per-lead fit)
        b = smt.sum(axis=0) / (stt * NDAYS)                        # (181,360)
        # detrended signal variance: Var_y(mean_y) − trend-explained part
        var_sig = sm2 / n_init - (s1 / n) ** 2
        var_sig_dt = np.maximum(var_sig - (b[None] ** 2) * stt / n_init, 0.0)
        acc[key + "_mu"] = mu.astype(np.float32)
        acc[key + "_sd"] = sd.astype(np.float32)
        acc[key + "_esd"] = esd.astype(np.float32)
        acc[key + "_trend"] = b.astype(np.float32)
        acc[key + "_sigdt"] = np.sqrt(var_sig_dt).astype(np.float32)
        print(f"field climo {key}: done (trend range "
              f"{b.min():+.3f}..{b.max():+.3f}/yr)", flush=True)
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **acc)
    return acc


def precip_daily_clim(month: int) -> np.ndarray:
    """Per-lead-day reforecast mean precip rate (mm/day), cached.

    The daily precip stream is CONSISTENT with the monthly stream (4%%
    global-mean difference, r=0.84 spatially vs sub-month sampling), unlike
    t2m/z500 — so the own-hindcast baseline is sound here."""
    f = CLIMDIR / f"precip_climo_{month:02d}.npz"
    if f.exists():
        return np.load(f)["clim"]
    ds = _open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
    ds = ds.sel(init=slice(str(CLIM_Y0), str(CLIM_Y1)))
    n_init = ds.sizes["init"]
    s1 = np.zeros((NDAYS, 181, 360), np.float64)
    n = 0
    for yi in range(n_init):
        v = ds.PRATE_surface.isel(init=yi).values.astype(np.float64)
        s1 += v.sum(axis=0)
        n += v.shape[0]
        if yi % 10 == 0:
            print(f"precip climo: init {yi + 1}/{n_init}", flush=True)
    clim = (s1 / n * 86400).astype(np.float32)
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, clim=clim)
    print(f"precip climo {month:02d}: cached", flush=True)
    return clim


def render_chi_loop(issue, t0, lead_days):
    """Daily 200-hPa velocity-potential anomaly loop, reusing the site's
    spherical-harmonic tools and the committed ERA5 1991-2020 harmonic χ
    climatology (same baseline as the AIFS vpot product)."""
    import sys as _sys, time as _time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import xarray as xr
    _sys.path.insert(0, str(REPO / "scripts" / "mjo" / "src"))
    from wind200_vpot import velocity_potential, eval_vp_clim, CLIM, VP_CMAP, VP_LEVELS
    ds = _open(f"{BASE}/forecast/{issue}/atm_daily.zarr")
    lat, lon = ds.lat.values, ds.lon.values
    u = ds.UGRD_200mb.values
    v = ds.VGRD_200mb.values
    ksel = np.where(np.isfinite(u[0, :, 90, 180]) & (lead_days >= 16))[0]
    cds = xr.open_dataset(CLIM)
    coef = cds[list(cds.data_vars)[0]].values
    ANIM = REPO / "assets" / "sfs" / "anim"
    name = "sfs_chid"
    outdir = ANIM / name
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("F*.webp"):
        old.unlink()
    frames = []
    for i, k in enumerate(ksel):
        valid = t0 + pd.Timedelta(days=int(lead_days[k]))
        u_da = xr.DataArray(np.nanmean(u[:, k], axis=0),
                            coords={"latitude": lat, "longitude": lon},
                            dims=("latitude", "longitude"))
        v_da = xr.DataArray(np.nanmean(v[:, k], axis=0),
                            coords={"latitude": lat, "longitude": lon},
                            dims=("latitude", "longitude"))
        chi, dlat, dlon = velocity_potential(u_da, v_da)
        anom = (chi - eval_vp_clim(coef, int(valid.dayofyear))) / 1e6
        fig = plt.figure(figsize=(13.0, 5.6))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([-180, 180, -75, 75], crs=ccrs.PlateCarree())
        cf = ax.contourf(dlon, dlat, anom, levels=VP_LEVELS, cmap=VP_CMAP,
                         extend="both", transform=ccrs.PlateCarree())
        ax.coastlines(lw=0.5, color="0.25")
        ax.set_title("ens-mean 200-hPa velocity potential anomaly vs ERA5 "
                     "1991–2020 harmonic normal (10⁶ m²/s; negative = "
                     "divergent outflow / enhanced convection)",
                     fontsize=10, loc="left")
        fig.colorbar(cf, ax=ax, orientation="vertical", fraction=0.025,
                     pad=0.015)
        fig.suptitle(f"SFS beta — 200-hPa χ · {valid:%a %b %d %Y} "
                     f"(day {int(lead_days[k])}) · 31-member mean · issue "
                     f"{t0:%b %Y}", fontsize=12, fontweight="bold", y=0.99)
        fig.subplots_adjust(top=0.88)
        fn = f"F{i:02d}.webp"
        fig.savefig(outdir / fn, dpi=130, bbox_inches="tight")
        plt.close(fig)
        frames.append({"idx": i, "file": fn, "date": f"{valid:%Y-%m-%d}",
                       "label": f"{valid:%b %d} · day {int(lead_days[k])}"})
    (ANIM / f"{name}_manifest.json").write_text(json.dumps(
        {"ver": int(_time.time()), "days": len(frames),
         "regions": {name: {"label": "SFS 200-hPa velocity potential anomaly",
                            "n_frames": len(frames), "frames": frames}}}))
    print(f"loop {name}: {len(frames)} frames", flush=True)


def era5_daily_clim():
    """ERA5 1991-2020 day-of-year climatology (±7 d window) for t2m/z500,
    NH 1.5° from the local WB2 store; cached once, month-independent."""
    f = CLIMDIR / "era5_daily_clim.npz"
    if f.exists():
        return dict(np.load(f))
    import xarray as xr
    W = Path("~/era5_store/wb2_1p5_daily").expanduser()
    out = {}
    for v in ("t2m", "z500"):
        das = []
        for y in range(1991, 2021):
            ds = xr.open_dataset(W / v / f"{v}_{y}.nc")
            das.append(ds[list(ds.data_vars)[0]].transpose(
                "time", "latitude", "longitude"))
        da = xr.concat(das, dim="time")
        doy = da.time.dt.dayofyear.values
        vals = da.values
        clim = np.full((366,) + vals.shape[1:], np.nan, np.float32)
        for d in range(1, 367):
            m = np.abs(((doy - d + 183) % 366) - 183) <= 7
            clim[d - 1] = vals[m].mean(axis=0)
        out[v] = clim
        out["lat"] = da.latitude.values.astype(np.float32)
        out["lon"] = da.longitude.values.astype(np.float32)
        print(f"era5 clim {v}: built", flush=True)
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


def render_daily_maps(issue, t0, sel, lead_days, t2, z5, F, lat, lon):
    """Daily anomaly loops vs ERA5 1991-2020 climatology (the site's fixed
    base): the beta's NRT daily stream is inconsistent with its own
    reforecast AND monthly stream (~+2 K structured), so the own-hindcast
    daily baseline is unreliable; ERA5 is the trustworthy reference. z500
    frames carry ensemble-mean absolute height contours."""
    import time as _time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    E = era5_daily_clim()
    elat, elon = E["lat"], E["lon"]
    ANIM = REPO / "assets" / "sfs" / "anim"
    la = lat >= 20
    LON, LAT = np.meshgrid(lon, lat[la])
    proj = ccrs.PlateCarree(central_longitude=-100)

    def clim_on_grid(cfield):
        """interp one ERA5 1.5° clim field to the SFS 1° grid (20-90N)."""
        import xarray as xr
        # cyclic pad so the 358.5..360 wrap interpolates instead of NaN-striping
        cf = np.concatenate([cfield, cfield[:, :1]], axis=1)
        elon_p = np.concatenate([elon, [elon[0] + 360.0]])
        da = xr.DataArray(cf, coords={"latitude": elat, "longitude": elon_p},
                          dims=("latitude", "longitude"))
        return da.interp(latitude=lat[la], longitude=lon,
                         kwargs={"fill_value": None}).values

    # PRATE is valid on the OPPOSITE day parity to t2m/z500 (the beta
    # interleaves variable groups across alternating days) — select its own
    # finite leads independently
    import fsspec as _fs, xarray as _xr
    _dsd = _xr.open_zarr(_fs.get_mapper(
        f"{BASE}/forecast/{issue}/atm_daily.zarr"), consolidated=True,
        decode_timedelta=True)
    pr_full = _dsd.PRATE_surface.values * 86400
    psel = np.where(np.isfinite(pr_full[0, :, 90, 180])
                    & (lead_days >= 16))[0]
    pr = pr_full[:, psel]                                 # (31, np, 181, 360)
    pclim = precip_daily_clim(int(issue[4:6]))[psel]

    for key, fld, label in (("t2md", t2, "2 m temperature"),
                            ("z500d", z5, "500 hPa height"),
                            ("prcpd", pr, "precipitation")):
        glob = key == "prcpd"
        ksel = psel if glob else sel
        if glob:
            ens = np.nanmean(fld, axis=0) - pclim         # mm/day, global
            zabs = None
            LONg, LATg = np.meshgrid(lon, lat)
        else:
            vk = "t2m" if key == "t2md" else "z500"
            scale = 1.0 if vk == "t2m" else 0.1           # K; gpm -> dam
            ens_abs = np.nanmean(fld[:, :, la], axis=0)
            doys = [(t0 + pd.Timedelta(days=int(d))).dayofyear
                    for d in lead_days[sel]]
            clim = np.stack([clim_on_grid(E[vk][min(d, 366) - 1]) for d in doys])
            ens = (ens_abs - clim) * scale
            zabs = ens_abs * 0.1 if key == "z500d" else None
            LONg, LATg = LON, LAT

        name = f"sfs_{key}"
        outdir = ANIM / name
        outdir.mkdir(parents=True, exist_ok=True)
        for old in outdir.glob("F*.webp"):
            old.unlink()
        frames = []
        for i in range(ens.shape[0]):
            valid = t0 + pd.Timedelta(days=int(lead_days[ksel][i]))
            gproj = (ccrs.PlateCarree(central_longitude=180) if glob else proj)
            fig, ax = plt.subplots(figsize=(13.0, 5.6 if glob else 3.9),
                                   subplot_kw=dict(projection=gproj))
            if glob:
                pm0 = ax.pcolormesh(LONg, LATg, ens[i], cmap="BrBG",
                                    vmin=-15, vmax=15,
                                    transform=ccrs.PlateCarree(), rasterized=True)
            else:
                vmax0 = 10 if key == "t2md" else 24
                pm0 = ax.pcolormesh(LONg, LATg, ens[i], cmap="RdBu_r",
                                    vmin=-vmax0, vmax=vmax0,
                                    transform=ccrs.PlateCarree(), rasterized=True)
            if zabs is not None:
                cs = ax.contour(LONg, LATg, zabs[i], levels=np.arange(480, 601, 6),
                                colors="k", linewidths=0.7,
                                transform=ccrs.PlateCarree())
                ax.clabel(cs, levels=np.arange(480, 601, 12), fmt="%d",
                          fontsize=7, inline=True)
            ax.coastlines(lw=0.5, color="0.25")
            ax.add_feature(cfeature.BORDERS, lw=0.25, edgecolor="0.45")
            if glob:
                ax.set_global()
            else:
                ax.set_extent([-180, 180, 20, 90], ccrs.PlateCarree())
            if glob:
                ttl = "ens-mean precip anomaly vs own reforecast daily climatology (mm/day)"
            else:
                unit0 = "°C" if key == "t2md" else "dam"
                ttl = ("ens-mean anomaly vs ERA5 1991–2020 daily normal "
                       f"({unit0})")
                if zabs is not None:
                    ttl += " · contours: ens-mean z500 (dam)"
            ax.set_title(ttl, fontsize=10, loc="left")
            fig.colorbar(pm0, ax=ax, orientation="vertical",
                         fraction=0.025, pad=0.015, extend="both")
            fig.suptitle(f"SFS beta — {label} · {valid:%a %b %d %Y} "
                         f"(day {int(lead_days[ksel][i])}) · 31-member mean "
                         f"· issue {t0:%b %Y}",
                         fontsize=12, fontweight="bold", y=0.99)
            fig.subplots_adjust(top=0.86)
            fn = f"F{i:02d}.webp"
            fig.savefig(outdir / fn, dpi=130, bbox_inches="tight")
            plt.close(fig)
            frames.append({"idx": i, "file": fn, "date": f"{valid:%Y-%m-%d}",
                           "label": f"{valid:%b %d} · day {int(lead_days[ksel][i])}"})
        (ANIM / f"{name}_manifest.json").write_text(json.dumps(
            {"ver": int(_time.time()), "days": len(frames),
             "regions": {name: {"label": f"SFS {label} anomaly + spread",
                                "n_frames": len(frames), "frames": frames}}}))
        print(f"loop {name}: {len(frames)} frames", flush=True)


MLEADS = list(range(2, 12))          # monthly extension: leads 2-11


def _coarse2m(field):
    """0.5° (361,720) -> 2° block mean over trailing lat/lon dims."""
    f = field[..., :360, :]
    sh = f.shape
    return f.reshape(*sh[:-2], 90, 4, 180, 4).mean(axis=(-3, -1))


def build_monthly_climo(month: int) -> dict:
    """Monthly-lead index samples from the reforecast atm_monthly store."""
    f = CLIMDIR / f"monthly_idx_climo_{month:02d}.npz"
    if f.exists():
        return dict(np.load(f))
    ds = _open(f"{BASE}/reforecast/{month:02d}/atm_monthly.zarr")
    ds = ds.sel(init=slice(str(CLIM_Y0), str(CLIM_Y1))).isel(lead=MLEADS)
    lat, lon = ds.lat.values, ds.lon.values
    ii, jj, w = _metro_idx(lat, lon)
    n_init, n_mem = ds.sizes["init"], ds.sizes["member"]
    nS, nL = n_init * n_mem, len(MLEADS)
    lat2 = _coarse2m(np.broadcast_to(lat[:361, None], (361, 720)))[:, 0]
    dom2 = lat2 >= AO_DOM[0]
    hdd = np.zeros((nS, nL), np.float32); cdd = np.zeros_like(hdd)
    nao = np.zeros_like(hdd); pna = np.zeros_like(hdd); epo = np.zeros_like(hdd)
    slp2 = np.zeros((nS, nL, int(dom2.sum()), 180), np.float32)
    k = 0
    for yi in range(n_init):
        pm = ds.prmsl.isel(init=yi).values / 100.0
        z5 = ds.z500.isel(init=yi).values
        t2 = ds.tmp2m.isel(init=yi).values
        for mi in range(n_mem):
            h, c = _degree_days(t2[mi], ii, jj, w)
            hdd[k], cdd[k] = h, c
            nao[k] = (pm[mi][..., np.argmin(np.abs(lat - NAO_S[0])),
                             np.argmin(np.abs(lon - NAO_S[1]))]
                      - pm[mi][..., np.argmin(np.abs(lat - NAO_N[0])),
                               np.argmin(np.abs(lon - NAO_N[1]))])
            pna[k] = _pna(z5[mi], lat, lon)
            epo[k] = _box(z5[mi], lat, lon, EPO_S) - _box(z5[mi], lat, lon, EPO_N)
            slp2[k] = _coarse2m(pm[mi])[:, dom2, :]
            k += 1
        print(f"monthly climo: init {yi + 1}/{n_init}", flush=True)
    climfield = slp2.mean(axis=0)
    z_anom = slp2 - climfield[None]
    wgt = np.sqrt(np.cos(np.deg2rad(lat2[dom2])))[:, None]
    X = (z_anom * wgt).reshape(nS * nL, -1)
    X = X - X.mean(axis=0)
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    e1 = vt[0].astype(np.float32)
    if np.nanmean(e1.reshape(int(dom2.sum()), 180)[lat2[dom2] >= 70]) > 0:
        e1 = -e1                                   # +AO = low polar pressure
    ao = (z_anom * wgt).reshape(nS, nL, -1) @ e1
    out = {"lat2m": lat2[dom2], "ao_pattern_m": e1,
           "ao_climfield_m": climfield.astype(np.float32),
           "hdd_m": hdd, "cdd_m": cdd, "ao_m": ao.astype(np.float32),
           "nao_m": nao, "pna_m": pna, "epo_m": epo}
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    print(f"monthly idx climo {month:02d}: cached", flush=True)
    return out


def build_climo(month: int) -> dict:
    """Stream the reforecast dailies once; cache everything small."""
    f = CLIMDIR / f"daily_climo_{month:02d}.npz"
    if f.exists():
        return dict(np.load(f))
    import xarray as xr  # noqa: F401
    ds = _open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
    ds = ds.sel(init=slice(str(CLIM_Y0), str(CLIM_Y1)))
    lat, lon = ds.lat.values, ds.lon.values
    ii, jj, w = _metro_idx(lat, lon)
    n_init, n_mem = ds.sizes["init"], ds.sizes["member"]
    nS = n_init * n_mem

    hdd = np.zeros((nS, NDAYS), np.float32)
    cdd = np.zeros((nS, NDAYS), np.float32)
    nao_s = np.zeros((nS, NDAYS), np.float32); nao_n = np.zeros_like(nao_s)
    pna = np.zeros_like(nao_s); epo_s = np.zeros_like(nao_s)
    epo_n = np.zeros_like(nao_s)
    # AO: keep coarsened z1000 fields north of AO_DOM[0] for EOF + projections
    lat2 = _coarse2(np.broadcast_to(lat[:181, None], (181, 360)))[:, 0]
    dom2 = lat2 >= AO_DOM[0]
    z1k = np.zeros((nS, NDAYS, int(dom2.sum()), 180), np.float32)

    k = 0
    for yi in range(n_init):
        t2 = ds.TMP_2maboveground.isel(init=yi).values      # (11, 47, 181, 360)
        z5 = ds.HGT_500mb.isel(init=yi).values
        z1 = ds.HGT_1000mb.isel(init=yi).values
        pm = ds.PRMSL_meansealevel.isel(init=yi).values / 100.0
        for mi in range(n_mem):
            h, c = _degree_days(t2[mi], ii, jj, w)
            hdd[k], cdd[k] = h, c
            nao_s[k] = pm[mi][..., np.argmin(np.abs(lat - NAO_S[0])),
                              np.argmin(np.abs(lon - NAO_S[1]))]
            nao_n[k] = pm[mi][..., np.argmin(np.abs(lat - NAO_N[0])),
                              np.argmin(np.abs(lon - NAO_N[1]))]
            pna[k] = _pna(z5[mi], lat, lon)
            epo_s[k] = _box(z5[mi], lat, lon, EPO_S)
            epo_n[k] = _box(z5[mi], lat, lon, EPO_N)
            z1k[k] = _coarse2(z1[mi])[:, dom2, :]
            k += 1
        print(f"climo stream: init {yi + 1}/{n_init}", flush=True)

    # AO EOF on lead-day-anomaly z1000 (every 2nd day to bound the SVD);
    # the per-lead-day reforecast mean field is ALSO the climatology the NRT
    # members are anomalized against (never the NRT ensemble mean — that
    # would center the forecast AO on zero by construction)
    ao_climfield = z1k.mean(axis=0)                          # (47, pts, 180)
    z_anom = z1k - ao_climfield[None]
    wgt = np.sqrt(np.cos(np.deg2rad(lat2[dom2])))[:, None]
    X = (z_anom[:, ::2] * wgt).reshape(nS * len(range(0, NDAYS, 2)), -1)
    X = X - X.mean(axis=0)
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    e1 = vt[0].astype(np.float32)
    # sign: positive AO = LOW polar heights (poleward of 70N loading < 0)
    grid = e1.reshape(int(dom2.sum()), 180)
    if np.nanmean(grid[lat2[dom2] >= 70]) > 0:
        e1 = -e1
    ao = ((z_anom * wgt).reshape(nS, NDAYS, -1) @ e1)

    out = {"lat2": lat2[dom2], "ao_pattern": e1,
           "ao_climfield": ao_climfield.astype(np.float32),
           "hdd": hdd, "cdd": cdd,
           "ao": ao.astype(np.float32),
           "nao": (nao_s - nao_n).astype(np.float32),
           "pna": pna, "epo": (epo_s - epo_n).astype(np.float32)}
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    print(f"daily climo {month:02d}: cached ({nS} samples/day)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=datetime.now(timezone.utc).strftime("%Y%m"))
    ap.add_argument("--climo-only", action="store_true")
    args = ap.parse_args()
    issue, month = args.issue, int(args.issue[4:6])
    t0 = pd.Timestamp(f"{issue[:4]}-{issue[4:6]}-01")

    C = build_climo(month)
    S = build_strat_climo(month)
    M = build_monthly_climo(month)
    F = build_field_climo(month)
    if args.climo_only:
        return 0

    ds = _open(f"{BASE}/forecast/{issue}/atm_daily.zarr")
    lat, lon = ds.lat.values, ds.lon.values
    ii, jj, w = _metro_idx(lat, lon)
    t2 = ds.TMP_2maboveground.values                        # (31, 47, 181, 360)
    z5 = ds.HGT_500mb.values
    z1 = ds.HGT_1000mb.values
    pm = ds.PRMSL_meansealevel.values / 100.0
    # the beta's NRT daily output is disseminated every OTHER day (odd leads
    # all-NaN; the reforecast is fully daily). Keep the finite leads from day
    # 16 on — the subseasonal weeks-3-to-6.5 window where this system adds
    # value beyond medium-range NWP.
    lead_days = pd.to_timedelta(ds.lead.values).days.values
    finite = np.isfinite(t2[0, :, 90, 180])
    sel = np.where(finite & (lead_days >= 16))[0]
    t2, z5, z1, pm = t2[:, sel], z5[:, sel], z1[:, sel], pm[:, sel]
    C = {k: (v[:, sel] if getattr(v, "ndim", 0) == 2 else v) for k, v in C.items()}
    C["ao_climfield"] = C["ao_climfield"][sel]
    S = {k: v[:, sel] for k, v in S.items()}
    nsel = len(sel)
    print(f"NRT dailies loaded ({nsel} valid subseasonal steps)", flush=True)

    hdd, cdd = _degree_days(t2, ii, jj, w)                  # (31, 47) each
    raw = {
        "nao": pm[..., np.argmin(np.abs(lat - NAO_S[0])),
                 np.argmin(np.abs(lon - NAO_S[1]))]
               - pm[..., np.argmin(np.abs(lat - NAO_N[0])),
                    np.argmin(np.abs(lon - NAO_N[1]))],
        "pna": _pna(z5, lat, lon),
        "epo": _box(z5, lat, lon, EPO_S) - _box(z5, lat, lon, EPO_N),
    }
    lat2full = _coarse2(np.broadcast_to(lat[:181, None], (181, 360)))[:, 0]
    dom2 = lat2full >= AO_DOM[0]
    wgt = np.sqrt(np.cos(np.deg2rad(C["lat2"])))[:, None]
    z1c = _coarse2(z1)[..., dom2, :]
    a_f = z1c - C["ao_climfield"][None]
    # remove the area-weighted domain mean per member/day before projecting:
    # the beta's daily stream carries a polar-amplified height excess (an
    # annular artifact) whose common-mode part would otherwise project onto
    # the AO; the differential part cannot be removed without destroying the
    # AO signal itself and is flagged in the page caption.
    aw = np.cos(np.deg2rad(C["lat2"]))[:, None]
    dm = (a_f * aw).sum(axis=(-2, -1)) / (aw.sum() * a_f.shape[-1])
    a_f = a_f - dm[..., None, None]
    raw["ao"] = (a_f * wgt).reshape(z1c.shape[0], nsel, -1) @ C["ao_pattern"]

    def standardized(key):
        cs = C[key]                                          # (samples, 47)
        mu, sd = cs.mean(axis=0), cs.std(axis=0, ddof=1)
        return (raw[key] - mu[None, :]) / sd[None, :]

    tele = {}
    for key, label in (("ao", "AO"), ("nao", "NAO"), ("pna", "PNA"),
                       ("epo", "EPO")):
        s = standardized(key)
        tele[key] = {
            "label": label,
            "members": np.round(s, 2).tolist(),
            "p_neg": np.round((s < 0).mean(axis=0), 3).tolist(),
            "p_plus1": np.round((s > 1).mean(axis=0), 3).tolist(),
            "p_minus1": np.round((s < -1).mean(axis=0), 3).tolist(),
        }

    def climo_band(cs):
        q = np.percentile(cs, [10, 25, 50, 75, 90], axis=0)
        out = {p: np.round(q[i], 1).tolist()
               for i, p in enumerate(("p10", "p25", "p50", "p75", "p90"))}
        out["mean"] = np.round(cs.mean(axis=0), 1).tolist()
        return out

    v100 = ds.VGRD_100mb.values[:, sel]
    t100 = ds.TMP_100mb.values[:, sel]
    u50f = ds.UGRD_50mb.values[:, sel]
    j60 = int(np.argmin(np.abs(lat - 60)))
    strat = {
        "vt100": {"label": "100-hPa eddy heat flux (45–75°N)",
                  "units": "K·m/s",
                  "members": np.round(_vt100_band(v100, t100, lat), 2).tolist(),
                  "climo": climo_band(S["vt100"])},
        "u50": {"label": "50-hPa zonal-mean wind at 60°N", "units": "m/s",
                "members": np.round(u50f[..., j60, :].mean(axis=-1), 2).tolist(),
                "climo": climo_band(S["u50"])},
    }

    render_daily_maps(issue, t0, sel, lead_days, t2, z5, F, lat, lon)
    render_chi_loop(issue, t0, lead_days)

    # ── monthly extension: leads 2-11 from atm_monthly ──────────────────────
    dsm = _open(f"{BASE}/forecast/{issue}/atm_monthly.zarr")
    mlat, mlon = dsm.lat.values, dsm.lon.values
    mii, mjj, mw = _metro_idx(mlat, mlon)
    pm_m = dsm.prmsl.isel(lead=MLEADS).values / 100.0
    z5_m = dsm.z500.isel(lead=MLEADS).values
    t2_m = dsm.tmp2m.isel(lead=MLEADS).values
    hdd_m, cdd_m = _degree_days(t2_m, mii, mjj, mw)
    raw_m = {
        "nao": (pm_m[..., np.argmin(np.abs(mlat - NAO_S[0])),
                     np.argmin(np.abs(mlon - NAO_S[1]))]
                - pm_m[..., np.argmin(np.abs(mlat - NAO_N[0])),
                       np.argmin(np.abs(mlon - NAO_N[1]))]),
        "pna": _pna(z5_m, mlat, mlon),
        "epo": _box(z5_m, mlat, mlon, EPO_S) - _box(z5_m, mlat, mlon, EPO_N),
    }
    wgt_m = np.sqrt(np.cos(np.deg2rad(M["lat2m"])))[:, None]
    lat2mf = _coarse2m(np.broadcast_to(mlat[:361, None], (361, 720)))[:, 0]
    dom2m = lat2mf >= AO_DOM[0]
    slp2 = _coarse2m(pm_m)[..., dom2m, :]
    raw_m["ao"] = ((slp2 - M["ao_climfield_m"][None]) * wgt_m
                   ).reshape(slp2.shape[0], len(MLEADS), -1) @ M["ao_pattern_m"]
    tele_m = {}
    for key in ("ao", "nao", "pna", "epo"):
        cs = M[key + "_m"]
        mu, sd = cs.mean(axis=0), cs.std(axis=0, ddof=1)
        sm = (raw_m[key] - mu[None, :]) / sd[None, :]
        tele_m[key] = {"members": np.round(sm, 2).tolist(),
                       "p_neg": np.round((sm < 0).mean(axis=0), 3).tolist(),
                       "p_plus1": np.round((sm > 1).mean(axis=0), 3).tolist(),
                       "p_minus1": np.round((sm < -1).mean(axis=0), 3).tolist()}
    months_m = [(t0 + pd.DateOffset(months=k)).strftime("%Y-%m") for k in MLEADS]
    monthly = {
        "months": months_m,
        "telecons": tele_m,
        "hdd": {"members": np.round(hdd_m, 1).tolist(),
                "climo": climo_band(M["hdd_m"])},
        "cdd": {"members": np.round(cdd_m, 1).tolist(),
                "climo": climo_band(M["cdd_m"])},
    }

    days = [(t0 + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
            for d in lead_days[sel]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "issue": f"{issue[:4]}-{issue[4:6]}",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "days": days,
        "hdd": {"members": np.round(hdd, 1).tolist(), "climo": climo_band(C["hdd"])},
        "cdd": {"members": np.round(cdd, 1).tolist(), "climo": climo_band(C["cdd"])},
        "telecons": tele,
        "strat": strat,
        "monthly": monthly,
        "notes": ("subseasonal window: valid leads day 16-46, 48-h steps (NRT daily output is disseminated every other day); "
                  "degree days: population-weighted (50 largest metros), base 65F; "
                  "telecons standardized per lead day vs own reforecast "
                  f"{CLIM_Y0}-{CLIM_Y1} (330 samples/day)"),
    }, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(REPO)}")
    print("HDD ens-mean d1-5:", np.round(hdd.mean(0)[:5], 1).tolist())
    print("CDD ens-mean d1-5:", np.round(cdd.mean(0)[:5], 1).tolist())
    for k in tele:
        print(k, "ens-mean d1-10:",
              np.round(np.array(tele[k]["members"]).mean(0)[:10], 2).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
