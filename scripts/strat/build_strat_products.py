#!/usr/bin/env python3
"""Stratosphere monitor products (SH live; NH panels activate in NDJFM).

Per cycle:
  1. maintain a rolling ~150-day observed cache (ARCO ERA5, 12Z):
     u/gh at 10 hPa (maps + plume tails) and v/T at 6 levels (TEM descent),
     data/strat_obs.nc  (gitignored, incremental, ~6-day ARCO lag)
  2. render, for the active hemisphere:
     assets/strat/vortex_maps.webp     two-panel: wind+heights+sinking / eddy-T
     assets/strat/u10_plume.webp       obs + climatology envelope + AIFS-ENS members
     assets/strat/descent_index.webp   polar-cap descent + cap T, with 1974- clim band
  3. stamp ?v= cache-busters in stratosphere.html

AIFS-ENS u@10 comes free from the AAM cycle cache (LEVELS_AAM includes 10 hPa).

    python scripts/strat/build_strat_products.py            # auto hemisphere
    python scripts/strat/build_strat_products.py --hemi sh
"""
from __future__ import annotations
import argparse
import glob
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE / "data"
ASSETS = REPO / "assets" / "strat"
TELECON = REPO / "scripts" / "telecon" / "data"
CACHE = REPO / "scripts" / "ecmwf" / "cache"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
A = 6.371e6
KAP = 0.2854
TEM_LEVS = [10, 20, 30, 50, 70, 100]
ROLL_DAYS = 150


# ── rolling observed cache ──────────────────────────────────────────────────
def update_obs() -> xr.Dataset:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "strat_obs.nc"
    have = None
    t_last = None
    if path.exists():
        have = xr.open_dataset(path).load(); have.close()
        if "vt_k2_100" not in have:     # schema upgrade -> full re-backfill
            have = None
        else:
            t_last = pd.Timestamp(have.time.values[-1])
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = (end - pd.Timedelta(days=ROLL_DAYS)) if t_last is None else \
            (t_last + pd.Timedelta(days=1))
    if start > end - pd.Timedelta(days=3):          # ARCO lag; nothing new yet
        return have
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    sub = dict(latitude=slice(None, None, 4), longitude=slice(None, None, 4))
    sel = dict(time=slice(str(start.date()), None))
    def pull(var, lev):
        x = ds[var].sel(level=lev, **sel).isel(**sub)
        x = x.sel(time=x.time.dt.hour == 12).compute()
        return x
    t0 = time.time()
    u10 = pull("u_component_of_wind", 10)
    g10 = pull("geopotential", 10) / 9.80665
    T10 = pull("temperature", 10)
    g100 = pull("geopotential", 100) / 9.80665
    z500f = pull("geopotential", 500) / 9.80665
    u50 = pull("u_component_of_wind", 50)
    w = sum(pull("vertical_velocity", l) for l in (30, 50, 70)) / 3 * 864.0
    vT = {}
    for l in TEM_LEVS:
        vT[("v", l)] = pull("v_component_of_wind", l)
        vT[("T", l)] = pull("temperature", l) if l != 10 else T10
    lat = u10.latitude.values
    tt = pd.DatetimeIndex(u10.time.values).normalize()
    # wave-1 heat-flux cospectrum at 100 hPa (SRI input vt_k1)
    v100 = vT[("v", 100)]; T100 = vT[("T", 100)]
    vp = (v100 - v100.mean("longitude")).values
    Tp = (T100 - T100.mean("longitude")).values
    n = vp.shape[-1]
    Vf = np.fft.rfft(vp, axis=-1) / n
    Tf = np.fft.rfft(Tp, axis=-1) / n
    vtk1 = 2.0 * (Vf[..., 1] * np.conj(Tf[..., 1])).real     # (time, lat)
    vtk2 = 2.0 * (Vf[..., 2] * np.conj(Tf[..., 2])).real     # wave-2 channel
    # equatorial u50 zonal mean (QBO)
    eqm = np.abs(u50.latitude.values) <= 5
    u50eq = u50.values[:, eqm, :].mean(axis=(1, 2))
    vbar = np.stack([vT[("v", l)].mean("longitude").values for l in TEM_LEVS], axis=1)
    Tbar = np.stack([vT[("T", l)].mean("longitude").values for l in TEM_LEVS], axis=1)
    vt = np.stack([((vT[("v", l)] - vT[("v", l)].mean("longitude"))
                    * (vT[("T", l)] - vT[("T", l)].mean("longitude"))
                    ).mean("longitude").values for l in TEM_LEVS], axis=1)
    new = xr.Dataset(
        {"u10": (("time", "lat", "lon"), u10.values),
         "z10": (("time", "lat", "lon"), g10.values),
         "T10": (("time", "lat", "lon"), T10.values),
         "z100": (("time", "lat", "lon"), g100.values),
         "z500": (("time", "lat", "lon"), z500f.values),
         "w357": (("time", "lat", "lon"), w.values),
         "vbar": (("time", "lev", "lat"), vbar),
         "Tbar": (("time", "lev", "lat"), Tbar),
         "vt": (("time", "lev", "lat"), vt),
         "vt_k1_100": (("time", "lat"), vtk1),
         "vt_k2_100": (("time", "lat"), vtk2),
         "u50_eq": (("time",), u50eq)},
        coords={"time": tt, "lat": lat, "lon": u10.longitude.values,
                "lev": np.array(TEM_LEVS, float)})
    ok = np.isfinite(new.u10.values).all(axis=(1, 2))
    new = new.isel(time=ok)
    merged = new if have is None else xr.concat([have, new], dim="time")
    merged = merged.isel(time=~pd.Index(merged.time.values).duplicated(keep="last"))
    merged = merged.sortby("time").isel(time=slice(-ROLL_DAYS, None))
    tmp = path.with_suffix(".nc.tmp"); merged.to_netcdf(tmp); tmp.replace(path)
    print(f"  obs cache -> {pd.Timestamp(merged.time.values[-1]).date()} "
          f"({new.sizes['time']} new days, {time.time()-t0:.0f}s)", flush=True)
    return merged


# ── TEM descent from the cache ──────────────────────────────────────────────
def descent_series(obs, hemi):
    p = obs.lev.values * 100.0
    lat = obs.lat.values
    fac = (1e5 / p)[None, :, None] ** KAP
    dthdp = np.gradient(obs.Tbar.values * fac, p, axis=1)
    dthdp = np.where(np.abs(dthdp) < 2e-4, -2e-4, dthdp)
    vstar = obs.vbar.values - np.gradient(obs.vt.values * fac / dthdp, p, axis=1)
    phi0 = 60.0 if hemi == "nh" else -60.0
    j = np.argmin(np.abs(lat - phi0))
    sgn = 1.0 if phi0 > 0 else -1.0
    cosb = np.cos(np.deg2rad(60)); area = 2*np.pi*A**2*(1 - np.sin(np.deg2rad(60)))
    om = -(2*np.pi*A*cosb/area) * np.cumsum(vstar[:, :, j] * sgn
                                            * np.gradient(p)[None, :], axis=1) * 864.0
    capm = lat >= 60 if hemi == "nh" else lat <= -60
    wgt = np.cos(np.deg2rad(lat[capm]))
    capT = (obs.Tbar.values[:, 2][:, capm] * wgt).sum(1) / wgt.sum()
    t = pd.DatetimeIndex(obs.time.values)
    return (pd.Series(om[:, 2], index=t), pd.Series(om[:, 4], index=t),
            pd.Series(capT, index=t))


def descent_clim(hemi):
    """day-of-year 10-90% band of cap descent @30 hPa from the 1974- record."""
    ds = xr.open_dataset(TELECON / "tem_series.nc")
    p = ds.lev.values * 100.0; lat = ds.lat.values
    fac = (1e5 / p)[None, :, None] ** KAP
    dthdp = np.gradient(ds.Tbar.values * fac, p, axis=1)
    dthdp = np.where(np.abs(dthdp) < 2e-4, -2e-4, dthdp)
    vstar = ds.vbar.values - np.gradient(ds.vt.values * fac / dthdp, p, axis=1)
    phi0 = 60.0 if hemi == "nh" else -60.0
    j = np.argmin(np.abs(lat - phi0)); sgn = 1.0 if phi0 > 0 else -1.0
    cosb = np.cos(np.deg2rad(60)); area = 2*np.pi*A**2*(1 - np.sin(np.deg2rad(60)))
    om30 = -(2*np.pi*A*cosb/area) * np.cumsum(vstar[:, :, j] * sgn
             * np.gradient(p)[None, :], axis=1)[:, 2] * 864.0
    om30 = pd.Series(om30, index=pd.DatetimeIndex(ds.time.values)) \
             .rolling(5, center=True, min_periods=1).mean()
    doy = om30.index.dayofyear
    lo = om30.groupby(doy).quantile(0.10)
    hi = om30.groupby(doy).quantile(0.90)
    md = om30.groupby(doy).mean()
    return lo, hi, md


# ── AIFS-ENS u10 plume ──────────────────────────────────────────────────────
def latest_aifs_u10():
    for d in sorted(glob.glob(str(CACHE / "*z")), reverse=True):
        hits = glob.glob(f"{d}/aifs-ens/pf_u_10-*m25.grib2")
        cfh = glob.glob(f"{d}/aifs-ens/cf_u_10-*x16.grib2")
        if hits and cfh:
            kw = dict(engine="cfgrib", backend_kwargs=dict(
                filter_by_keys={"shortName": "u", "level": 10}, indexpath=""))
            pf = xr.open_dataset(hits[0], **kw)
            cf = xr.open_dataset(cfh[0], **kw)
            tag = Path(d).name          # e.g. 2026072712z
            base = pd.Timestamp(f"{tag[:4]}-{tag[4:6]}-{tag[6:8]} {tag[8:10]}:00")
            return pf, cf, base
    return None, None, None


def u10_clim(hemi):
    f = "strat_series.nc" if hemi == "nh" else "strat_series_sh.nc"
    v = "u10_60n" if hemi == "nh" else "u10_60s"
    s = xr.open_dataset(TELECON / f).to_dataframe()[v]
    s.index = pd.DatetimeIndex(s.index)
    doy = s.index.dayofyear
    return s.groupby(doy).quantile(0.10), s.groupby(doy).quantile(0.90), \
           s.groupby(doy).mean()


# ── renderers ───────────────────────────────────────────────────────────────
def smooth(f, slon, slat):
    g = ndi.gaussian_filter1d(f, slon, axis=1, mode="wrap")
    return ndi.gaussian_filter1d(g, slat, axis=0, mode="nearest")


def render_maps(obs, hemi, out):
    i = -1
    lat, lon = obs.lat.values, obs.lon.values
    day = pd.Timestamp(obs.time.values[i]).date()
    sel = slice(-5, None)
    spd = np.hypot(obs.u10.values[i], 0)            # speed from u alone underestimates;
    # use gh-gradient-free approach: mean of last 5 days of |V| ~ use u,w fields we have
    spd = np.hypot(obs.u10.values[sel].mean(0), 0)
    # (v10 not cached to keep the pull light; zonal speed dominates the night jet)
    z = obs.z10.values[sel].mean(0)
    Ted = obs.T10.values[sel].mean(0)
    Ted = smooth(Ted - Ted.mean(axis=1, keepdims=True), 3, 2)
    wsm = smooth(obs.w357.values[sel].mean(0), 10, 6)
    proj = ccrs.SouthPolarStereo() if hemi == "sh" else ccrs.NorthPolarStereo()
    ext = [-180, 180, -90, -42] if hemi == "sh" else [-180, 180, 42, 90]
    ylocs = [-80, -70, -60, -50] if hemi == "sh" else [50, 60, 70, 80]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8), subplot_kw={"projection": proj})
    for ax in axes:
        ax.set_extent(ext, ccrs.PlateCarree())
        ax.coastlines(lw=0.6, color="0.35")
        ax.gridlines(color="0.6", lw=0.35, alpha=0.7, ylocs=ylocs,
                     xlocs=range(-180, 181, 30))
    ax = axes[0]
    pc0 = ax.contourf(lon, lat, spd, levels=np.arange(30, 101, 10), cmap="plasma",
                      extend="max", transform=ccrs.PlateCarree())
    ax.contour(lon, lat, z/1000, levels=np.arange(27, 32.5, 0.3),
               colors="0.55", linewidths=0.6, transform=ccrs.PlateCarree())
    cw = ax.contour(lon, lat, wsm, levels=[0.5, 1.0, 1.5, 2.0, 3.0],
                    colors="#5c3317", linewidths=[0.9, 1.2, 1.5, 1.8, 2.2],
                    transform=ccrs.PlateCarree())
    ax.clabel(cw, fontsize=7.5, fmt="%.1f")
    ax.set_title("10 hPa zonal wind (fill, ≥30 m/s) + heights\n"
                 "+ 30–70 hPa sinking (brown, hPa/day)", fontsize=10.5)
    fig.colorbar(pc0, ax=ax, orientation="horizontal", fraction=0.05, pad=0.05,
                 label="m/s")
    ax = axes[1]
    pc1 = ax.contourf(lon, lat, Ted, levels=np.linspace(-6, 6, 17), cmap="RdBu_r",
                      extend="both", transform=ccrs.PlateCarree())
    ax.contour(lon, lat, z/1000, levels=np.arange(27, 32.5, 0.3),
               colors="0.3", linewidths=0.6, transform=ccrs.PlateCarree())
    ax.set_title("10 hPa eddy temperature (K, smoothed) + heights", fontsize=10.5)
    fig.colorbar(pc1, ax=ax, orientation="horizontal", fraction=0.05, pad=0.05,
                 label="K")
    fig.suptitle(f"{'Southern' if hemi == 'sh' else 'Northern'} stratosphere — "
                 f"ERA5, 5-day mean to {day}", fontsize=13, fontweight="bold", y=0.99)
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.name}", flush=True)


def render_plume(obs, hemi, out):
    lat = obs.lat.values
    j = np.argmin(np.abs(lat - (60.0 if hemi == "nh" else -60.0)))
    uobs = pd.Series(obs.u10.values[:, j, :].mean(1),
                     index=pd.DatetimeIndex(obs.time.values))
    lo, hi, md = u10_clim(hemi)
    pf, cf, base = latest_aifs_u10()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    cd = uobs.index.dayofyear
    ax.fill_between(uobs.index, lo.reindex(cd).values, hi.reindex(cd).values,
                    color="0.88", label="climatology 10–90% (1974–)")
    ax.plot(uobs.index, md.reindex(cd).values, color="0.6", ls=":", lw=1)
    ax.plot(uobs.index, uobs.values, color="#1d2833", lw=2, label="ERA5 observed")
    if pf is not None:
        phi0 = 60.0 if hemi == "nh" else -60.0
        upf = pf["u"].sel(latitude=phi0).mean("longitude")
        ucf = cf["u"].sel(latitude=phi0).mean("longitude")
        fd = base + pd.to_timedelta(pf.step.values)
        for m in range(upf.sizes["number"]):
            ax.plot(fd, upf.isel(number=m).values, color="#4fa3d6", lw=0.7, alpha=0.5)
        ax.plot(fd, ucf.values, color="#0b5394", lw=2.2, label="AIFS-ENS control")
        ax.plot(fd, upf.mean("number").values, color="#e07b39", lw=2.2, ls="--",
                label="ensemble mean")
        fdd = pd.DatetimeIndex(fd).dayofyear
        ax.fill_between(fd, lo.reindex(fdd).values, hi.reindex(fdd).values,
                        color="0.88")
        ax.plot(fd, md.reindex(fdd).values, color="0.6", ls=":", lw=1)
    ax.axhline(0, color="0.4", lw=1, ls=":")
    ax.set_ylabel(f"zonal-mean u, 10 hPa, 60°{'N' if hemi == 'nh' else 'S'} (m/s)")
    ax.legend(loc="best", fontsize=8.5); ax.grid(alpha=0.25)
    ax.set_title(f"{'NH' if hemi == 'nh' else 'SH'} polar-night jet: observed + "
                 f"AIFS-ENS 15-day forecast (cycle {base})", fontsize=11.5,
                 fontweight="bold")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.name}", flush=True)


def render_descent(obs, hemi, out):
    om30, om70, capT = descent_series(obs, hemi)
    lo, hi, md = descent_clim(hemi)
    sm5 = lambda s: s.rolling(5, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    cd = om30.index.dayofyear
    ax.fill_between(om30.index, lo.reindex(cd).values, hi.reindex(cd).values,
                    color="0.88", label="climatology 10–90% (1974–)")
    ax.plot(om30.index, md.reindex(cd).values, color="0.6", ls=":", lw=1)
    ax.plot(om30.index, sm5(om30).values, color="#6d51b8", lw=2.2,
            label="cap descent ω* @30 hPa")
    ax.plot(om30.index, sm5(om70).values, color="#4fa3d6", lw=1.5, label="@70 hPa")
    ax.axhline(0, color="0.6", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(capT.index, capT.values, color="#b3475f", lw=2.2, alpha=0.85)
    ax2.set_ylabel("polar-cap T @30 hPa (K)", color="#b3475f")
    ax2.tick_params(axis="y", labelcolor="#b3475f")
    ax.set_ylabel("descent (hPa/day, + = sinking)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(f"{'NH' if hemi == 'nh' else 'SH'} polar-cap residual descent "
                 f"(TEM continuity across 60°) + temperature", fontsize=11.5,
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.name}", flush=True)


def stamp_html():
    page = REPO / "stratosphere.html"
    if not page.exists():
        return
    v = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M")
    txt = page.read_text()
    txt = re.sub(r'(assets/strat/[a-z0-9_]+\.webp)\?v=\w+', rf'\1?v={v}', txt)
    page.write_text(txt)
    print(f"  stamped stratosphere.html v={v}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemi", choices=["nh", "sh", "auto"], default="auto")
    a = ap.parse_args()
    hemi = a.hemi
    if hemi == "auto":
        m = pd.Timestamp.utcnow().month
        hemi = "nh" if m in (11, 12, 1, 2, 3) else "sh"
    ASSETS.mkdir(parents=True, exist_ok=True)
    obs = update_obs()
    render_maps(obs, hemi, ASSETS / "vortex_maps.webp")
    render_plume(obs, "sh", ASSETS / "u10_plume_sh.webp")
    render_plume(obs, "nh", ASSETS / "u10_plume_nh.webp")
    render_descent(obs, hemi, ASSETS / "descent_index.webp")
    import subprocess as _sp, sys as _sys
    _sp.run([_sys.executable, str(HERE / "build_strat_anim.py")], check=False)
    import subprocess, sys
    subprocess.run([sys.executable, str(HERE / "sri_score.py")], check=False)
    stamp_html()
    print("done", flush=True)


if __name__ == "__main__":
    main()