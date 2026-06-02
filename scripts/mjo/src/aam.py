#!/usr/bin/env python3
"""Global & hemispheric atmospheric angular momentum (AAM) from AIFS-ENS.

Relative (wind) AAM:  M = (a³/g) ∫∫∫ u·cos²φ dλ dφ dp  — the axial angular
momentum carried by the winds. Computed globally and per hemisphere from the
13-level zonal wind, mass-weighted in the vertical with surface pressure as the
lower bound (so sub-surface levels over terrain are excluded), for every member
of the combined AIFS-ENS + IFS-ENS… (AIFS-ENS only for now — IFS pl on open data
differs) ensemble → an absolute forecast plume.

    python src/aam.py --date 20260601 --time 00 --out plots/aam.webp
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from download_aifs import _retrieve, retrieve_parallel

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
DAILY_STEPS = [0] + list(range(24, 361, 24))          # analysis + days 1..15
A = 6.371e6
G = 9.80665
SCALE = 1e25                                          # plot units: 10²⁵ kg m² s⁻¹


def download(date: str, time: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    base = dict(model="aifs-ens", date=date, time=int(time), stream="enfo", step=DAILY_STEPS)
    for typ in ("cf", "pf"):
        # 13 levels × 50 members is a lot of byte-range requests; cap parallel
        # streams at 2 so AWS doesn't rate-limit ("SlowDown").
        def fetch(req, tgt):
            retrieve_parallel(req, tgt, workers=2) if typ == "pf" else _retrieve(req, tgt)
        up = out_dir / f"u_{date}_{time}z_{typ}.grib2"
        sp = out_dir / f"sp_{date}_{time}z_{typ}.grib2"
        if not up.exists():
            print(f"  {typ}: downloading u (13 levels) …", flush=True)
            fetch(dict(**base, type=typ, levtype="pl", levelist=LEVELS, param="u"), str(up))
        if not sp.exists():
            print(f"  {typ}: downloading sp …", flush=True)
            fetch(dict(**base, type=typ, levtype="sfc", param="sp"), str(sp))
        paths[typ] = (up, sp)
    return paths


def _vert_weights(p_pa: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """Mass-layer thickness Δp (Pa) per level, clipped to the surface pressure
    so sub-surface layers vanish. p ascending; sp (lat,lon). -> (lev,lat,lon)."""
    edges = np.empty(len(p_pa) + 1)
    edges[1:-1] = 0.5 * (p_pa[1:] + p_pa[:-1])
    edges[0] = max(p_pa[0] - 0.5 * (p_pa[1] - p_pa[0]), 0.0)
    edges[-1] = p_pa[-1] + 0.5 * (p_pa[-1] - p_pa[-2])
    lo = np.minimum(edges[:-1][:, None, None], sp)
    hi = np.minimum(edges[1:][:, None, None], sp)
    return np.clip(hi - lo, 0.0, None)


def aam_of(u, p_pa, sp, lat, dlon, dlat):
    """(global, NH, SH) relative AAM for one (lev,lat,lon) field."""
    Uint = np.sum(u * _vert_weights(p_pa, sp), axis=0)             # ∫u dp (lat,lon)
    cos2 = np.cos(np.deg2rad(lat)) ** 2
    dens = (A**3 / G) * Uint * cos2[:, None] * dlon * dlat
    return dens.sum(), dens[lat > 0].sum(), dens[lat < 0].sum()


def compute(paths: dict):
    """-> members×steps arrays for global/NH/SH, and valid step hours."""
    rows = {"global": [], "nh": [], "sh": []}
    steps_h = None
    for typ, (up, sp_path) in paths.items():
        du = xr.open_dataset(up, engine="cfgrib", backend_kwargs={"indexpath": ""},
                             chunks={"number": 1})
        ds = xr.open_dataset(sp_path, engine="cfgrib", backend_kwargs={"indexpath": ""},
                             chunks={"number": 1})
        u = du["u"].sortby("isobaricInhPa")
        p_pa = u.isobaricInhPa.values * 100.0
        lat = u.latitude.values
        dlon = np.deg2rad(abs(float(u.longitude[1] - u.longitude[0])))
        dlat = np.deg2rad(abs(float(lat[1] - lat[0])))
        spvar = ds[[v for v in ds.data_vars][0]]
        members = u.number.values if "number" in u.dims else [None]
        if steps_h is None:
            steps_h = (u.step / np.timedelta64(1, "h")).values.astype(int)
        for m in members:                                          # one member at a time
            um = (u if m is None else u.sel(number=m)).values      # (step,lev,lat,lon)
            spm = (spvar if m is None else spvar.sel(number=m)).values   # (step,lat,lon)
            g, n, s = [], [], []
            for k in range(um.shape[0]):
                G_, N_, S_ = aam_of(um[k], p_pa, spm[k], lat, dlon, dlat)
                g.append(G_); n.append(N_); s.append(S_)
            rows["global"].append(g); rows["nh"].append(n); rows["sh"].append(s)
        print(f"  {typ}: processed {len(members)} member(s)", flush=True)
    return {k: np.array(v) / SCALE for k, v in rows.items()}, steps_h


CLIM_PATH = Path(__file__).resolve().parent.parent / "data" / "reference" / "aam_clim.nc"
REG = (("global", "Global"), ("nh", "Northern Hemisphere"), ("sh", "Southern Hemisphere"))


def eval_clim(coef, doy):
    w = 2 * np.pi * np.asarray(doy) / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    return X @ coef


ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
HIST_PATH = Path(__file__).resolve().parent.parent / "data" / "reference" / "aam_history.nc"


def _aam_era5(ds, t):
    u = ds["u_component_of_wind"].sel(time=t, level=LEVELS).sortby("level")
    sp = ds["surface_pressure"].sel(time=t)
    lat = u.latitude.values
    dlon = np.deg2rad(abs(float(u.longitude[1] - u.longitude[0])))
    dlat = np.deg2rad(abs(float(lat[1] - lat[0])))
    return [x / SCALE for x in aam_of(u.values, u.level.values * 100.0, sp.values, lat, dlon, dlat)]


def observed_history(year):
    """This-year observed AAM (global/NH/SH) from ERA5, cached + incrementally
    extended (ERA5 lags ~5 days, so it stops a few days short of the init)."""
    from concurrent.futures import ThreadPoolExecutor
    try:
        ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    except Exception as e:                                          # noqa: BLE001
        print(f"  ERA5 history unavailable ({repr(e)[:60]})"); return None
    end = min(pd.Timestamp(ds.time.values[-1]).normalize(), pd.Timestamp(year, 12, 31))
    want = pd.date_range(pd.Timestamp(year, 1, 1), end, freq="3D")
    have = pd.DataFrame(columns=["global", "nh", "sh"])
    if HIST_PATH.exists():
        h = xr.open_dataset(HIST_PATH).to_dataframe(); h.index = pd.to_datetime(h.index)
        have = h
    todo = [d for d in want if d not in have.index]
    if todo:
        print(f"  computing {len(todo)} ERA5 history day(s) for {year} …", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(lambda d: _aam_era5(ds, d.strftime("%Y-%m-%dT12:00")), todo))
        new = pd.DataFrame(res, columns=["global", "nh", "sh"], index=pd.DatetimeIndex(todo))
        have = pd.concat([have, new]).sort_index()
        have = have[~have.index.duplicated(keep="last")]
        HIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({k: ("time", have[k].values) for k in ("global", "nh", "sh")},
                   coords={"time": have.index}).to_netcdf(HIST_PATH)
    return have[have.index.year == year].sort_index()


def _interp_doy(clim, key, region, doy):
    xp = clim.doy.values; fp = clim[key].sel(region=region).values
    return np.interp(doy, np.concatenate([xp - 365, xp, xp + 365]),
                     np.concatenate([fp, fp, fp]))


def plot(data, steps_h, init, hist, out: Path):
    valid = pd.to_datetime([init + pd.Timedelta(hours=int(h)) for h in steps_h])
    vdoy = np.array([d.dayofyear for d in valid])
    clim = xr.open_dataset(CLIM_PATH) if CLIM_PATH.exists() else None
    doy_year = np.arange(1, 366)
    mticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, (key, title) in enumerate(REG):
        arr = data[key]
        a0, a1 = axes[0, j], axes[1, j]
        coef = clim["coeffs"].sel(region=key).values if clim is not None else None
        # ---- row 1: absolute on the seasonal cycle (record range, ±1σ, mean) ----
        if clim is not None:
            cmean = eval_clim(coef, doy_year)
            if "min" in clim:
                a0.fill_between(doy_year, _interp_doy(clim, "min", key, doy_year),
                                _interp_doy(clim, "max", key, doy_year),
                                color="0.86", label="1991–2020 record range")
            if "std" in clim:
                cstd = _interp_doy(clim, "std", key, doy_year)
                a0.fill_between(doy_year, cmean - cstd, cmean + cstd, color="0.62",
                                alpha=0.55, label="±1σ")
            a0.plot(doy_year, cmean, color="0.3", lw=1.5, label="climatology mean")
        if hist is not None and len(hist):
            hd = np.array([d.dayofyear for d in hist.index])
            a0.plot(hd, hist[key].values, color="k", lw=1.7, label=f"{init.year} observed")
        a0.plot(vdoy, arr.T, color="#c62828", lw=0.3, alpha=0.10)
        a0.plot(vdoy, arr.mean(0), color="#b71c1c", lw=2.4, label="forecast")
        a0.axvline(vdoy[0], color="0.5", lw=0.8, ls=":")
        a0.set_title(title, fontsize=11, fontweight="bold")
        a0.set_xticks(mticks); a0.set_xticklabels(list("JFMAMJJASOND"), fontsize=8)
        a0.set_xlim(1, 365); a0.grid(True, alpha=0.2)
        # ---- row 2: anomaly + % of climatology ----
        if clim is not None:
            cbase = eval_clim(coef, vdoy)
            anom = arr - cbase[None, :]
            a1.plot(valid, anom.T, color="#c62828", lw=0.4, alpha=0.10)
            lo, hi = np.percentile(anom, [10, 90], axis=0)
            a1.fill_between(valid, lo, hi, color="#c62828", alpha=0.15)
            a1.plot(valid, anom.mean(0), color="#b71c1c", lw=2.4)
            if hist is not None and len(hist):
                ha = hist[key].values - eval_clim(coef, np.array([d.dayofyear for d in hist.index]))
                a1.plot(hist.index, ha, color="k", lw=1.4)
            a1.axhline(0, color="0.5", lw=0.9); a1.grid(True, alpha=0.2)
            for lab in a1.get_xticklabels():
                lab.set_rotation(45); lab.set_ha("right"); lab.set_fontsize(7)
            base = float(np.nanmean(cbase))                        # % of climatological AAM
            if abs(base) > 0.1:
                ax2 = a1.twinx(); lo_, hi_ = a1.get_ylim()
                ax2.set_ylim(lo_ / base * 100, hi_ / base * 100)
                ax2.set_ylabel("% of climatology", fontsize=8); ax2.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("Absolute AAM (10²⁵ kg m² s⁻¹)")
    axes[1, 0].set_ylabel("AAM anomaly (10²⁵ kg m² s⁻¹)")
    axes[0, 0].legend(fontsize=7, loc="lower left", ncol=2, framealpha=0.9)
    fig.suptitle(f"Atmospheric angular momentum (relative/wind term) — AIFS-ENS init {init:%Y-%m-%d %HZ}\n"
                 f"top: {init.year} observed + forecast on the seasonal cycle (mean, ±1σ, record range)   ·   "
                 f"bottom: anomaly vs. 1991–2020", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    g0 = data["global"].mean(0)
    print(f"saved {out} (global AAM init {g0[0]:.1f} → day15 {g0[-1]:.1f} ×10²⁵)")


ARCHIVE_PATH = Path(__file__).resolve().parent.parent / "data" / "reference" / "aam_forecast_archive.nc"


def archive_forecast(init, data, steps_h):
    """Append this run's ensemble-mean AAM (per lead, per region) to the archive,
    so successive forecasts can be overlaid to show the run-to-run trend."""
    leads = (np.array(steps_h) / 24.0).round().astype(int)
    means = np.stack([data[k].mean(0) for k in ("global", "nh", "sh")])      # (region, lead)
    new = xr.Dataset({"fc_mean": (("region", "lead"), means)},
                     coords={"region": ["global", "nh", "sh"], "lead": leads,
                             "init": pd.Timestamp(init)}).expand_dims("init")
    if ARCHIVE_PATH.exists():
        with xr.open_dataset(ARCHIVE_PATH) as ds:
            old = ds.load()
        if pd.Timestamp(init) in pd.to_datetime(old.init.values):
            old = old.drop_sel(init=pd.Timestamp(init))
        new = xr.concat([old, new], dim="init").sortby("init")
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARCHIVE_PATH.with_suffix(".tmp.nc"); new.to_netcdf(tmp); os.replace(tmp, ARCHIVE_PATH)
    print(f"  archived forecast {pd.Timestamp(init):%Y-%m-%d %HZ} ({new.sizes['init']} runs total)")


def plot_trend(out: Path, n_runs: int = 14):
    """Overlay the last n_runs ensemble-mean AAM-anomaly forecasts (per region),
    coloured by init date (newest bold) — the run-to-run forecast trend."""
    import matplotlib.dates as mdates
    from matplotlib import cm, colors as mcolors
    if not ARCHIVE_PATH.exists():
        return
    arch = xr.open_dataset(ARCHIVE_PATH)
    clim = xr.open_dataset(CLIM_PATH) if CLIM_PATH.exists() else None
    inits = pd.to_datetime(arch.init.values)[-n_runs:]
    n = len(inits); cmap = cm.plasma
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    handles = []
    for j, (key, title) in enumerate(REG):
        ax = axes[j]
        coef = clim["coeffs"].sel(region=key).values if clim is not None else None
        for i, it in enumerate(inits):
            fc = arch["fc_mean"].sel(region=key, init=it).values
            valid = pd.to_datetime([it + pd.Timedelta(days=int(l)) for l in arch.lead.values])
            y = fc - eval_clim(coef, np.array([d.dayofyear for d in valid])) if coef is not None else fc
            newest = i == n - 1
            ln, = ax.plot(valid, y, color=cmap(i / max(n - 1, 1)), lw=2.8 if newest else 1.1,
                          alpha=1.0 if newest else 0.75, zorder=5 if newest else 2,
                          label=f"{it:%b %d} {it:%H}Z" + ("  (latest)" if newest else ""))
            if j == 0:
                handles.append(ln)
        ax.axhline(0, color="0.5", lw=0.8); ax.grid(True, alpha=0.2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(45); lab.set_ha("right"); lab.set_fontsize(7.5)
    axes[0].set_ylabel("AAM anomaly (10²⁵ kg m² s⁻¹)")
    fig.legend(handles[::-1], [h.get_label() for h in handles[::-1]], loc="center left",
               bbox_to_anchor=(0.995, 0.5), fontsize=7.5, framealpha=0.9,
               title="forecast init", title_fontsize=8)
    fig.suptitle("AAM anomaly forecast trend — successive AIFS-ENS ensemble-mean runs "
                 "(newest = bold)", fontsize=12, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out} ({len(inits)} runs)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/aam")
    ap.add_argument("--out", default="plots/aam.webp")
    args = ap.parse_args()
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    paths = download(args.date, args.time, Path(args.data_dir))
    data, steps_h = compute(paths)
    hist = observed_history(init.year)
    plot(data, steps_h, init, hist, Path(args.out))
    archive_forecast(init, data, steps_h)
    plot_trend(Path(args.out).with_name("aam_trend.webp"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
