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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "era5"))
import store as ecmwf                                    # shared ECMWF download manager
import era5_store                                        # persistent local ARCO-ERA5 store

# 14 levels incl. 10 hPa: with midpoint layer edges the top layer spans 0–30 hPa,
# so the integral covers the FULL column (validated vs 37-level ERA5: adding 10 hPa
# cut the truncation error from ±1.0×10²⁵ — the winter polar-night jet — to ≲0.5,
# the remainder being the unsampled 20–30 hPa QBO shear zone; open data has no
# levels between 10 and 50).
LEVELS = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
DAILY_STEPS = [0] + list(range(24, 361, 24))          # analysis + days 1..15
A = 6.371e6
G = 9.80665
SCALE = 1e25                                          # plot units: 10²⁵ kg m² s⁻¹


def download(date: str, time: str, out_dir: Path = None) -> dict:
    """Ensure the AAM u-levels + surface pressure (cf+pf) via the shared store. To avoid
    re-downloading 200/850 (the RMM already pulls those), the heavy AAM-only 11 levels and
    the light RMM 200/850 file are fetched SEPARATELY and concatenated back to 13 in
    compute(). Returns {typ: (u_rest_path, u_rmm_path, sp_path)} of canonical cache paths."""
    cyc = ecmwf.Cycle(date, time); steps = tuple(DAILY_STEPS)
    paths = {}
    for typ in ("cf", "pf"):
        u_rest = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "u", "pl", ecmwf.LEVELS_AAM_REST, steps,
                                              ecmwf.AAM_PF_MEMBERS))
        u_rmm = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "u", "pl", ecmwf.LEVELS_RMM, steps))
        sp = ecmwf.sfc_path(cyc, "aifs-ens", typ, "sp")   # sp out of the batched surface file
        paths[typ] = (u_rest, u_rmm, sp)
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
    """(global, NH, SH) relative AAM for one (lev,lat,lon) field.
    The φ=0 row (present on the 0.25°/1.5° grids, and carrying maximal cos²
    weight) is split half-and-half so NH + SH = global exactly — excluding it
    from both sides dropped ~0.3% of the global integral from the hemispheres."""
    Uint = np.sum(u * _vert_weights(p_pa, sp), axis=0)             # ∫u dp (lat,lon)
    cos2 = np.cos(np.deg2rad(lat)) ** 2
    dens = (A**3 / G) * Uint * cos2[:, None] * dlon * dlat
    eq_half = 0.5 * dens[np.isclose(lat, 0.0)].sum()
    return (dens.sum(),
            dens[lat > 0].sum() + eq_half,
            dens[lat < 0].sum() + eq_half)


def compute(paths: dict):
    """-> members×steps arrays for global/NH/SH, and valid step hours."""
    rows = {"global": [], "nh": [], "sh": []}
    steps_h = None
    for typ, (up_rest, up_rmm, sp_path) in paths.items():
        kw = dict(engine="cfgrib", backend_kwargs={"indexpath": ""}, chunks={"number": 1})
        du_rest = xr.open_dataset(up_rest, **kw)         # 11 AAM-only levels
        du_rmm = xr.open_dataset(up_rmm, **kw)           # 200/850 (shared with the RMM)
        ds = xr.open_dataset(sp_path, engine="cfgrib",
                             backend_kwargs={"filter_by_keys": {"shortName": "sp"}, "indexpath": ""},
                             chunks={"number": 1})       # sp out of the batched surface file
        # concatenate the two u files back to the full 13 levels; the rest file may
        # carry a member SUBSET (AAM_PF_MEMBERS) — align the full-ensemble RMM file
        # to it so the concat doesn't outer-join NaN members
        ur, um = du_rest["u"], du_rmm["u"]
        if "number" in ur.dims and "number" in um.dims:
            um = um.sel(number=ur.number)
        u = xr.concat([ur, um], dim="isobaricInhPa").sortby("isobaricInhPa")
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

# ── GFZ ESMGFZ wind-term AAM: the independent full-column reference ──────────
# χ3_motion (dimensionless, anomaly vs 2003-2014) → M_r [kg m² s⁻¹]:
#   M_r = (χ3 + GFZ_CHI3_MEAN) × Ω·C/α3.  Files issued 2023+ print a wrong
# subtracted-mean header line — ALWAYS add this constant, never the header's.
GFZ_URL = ("https://datacenter.iers.org/products/geofluids/atmosphere/eamf/"
           "GGFC2010/GFZ/ESMGFZ_AAM_v1.0_03h_{year}.asc")
GFZ_CHI3_MEAN = 2.734436408977633e-8
GFZ_TO_KGM2S = 5.2092e33
GFZ_CACHE = Path.home() / "mjo" / "aam_validation"


def gfz_series(year):
    """Daily-mean global relative AAM from GFZ's operational ESMGFZ effective
    angular momentum functions (ECMWF full model column, 3-hourly) — the
    external verification trace. Returns a pd.Series (date → ×10²⁵ kg m² s⁻¹)
    or None; network failure falls back to the cached copy."""
    import requests
    GFZ_CACHE.mkdir(parents=True, exist_ok=True)
    cache = GFZ_CACHE / f"esmgfz_{year}.asc"
    try:
        r = requests.get(GFZ_URL.format(year=year), timeout=25)
        r.raise_for_status()
        cache.write_text(r.text)
        text = r.text
    except Exception as e:                                     # noqa: BLE001
        if not cache.exists():
            print(f"  GFZ reference unavailable ({str(e)[:50]})"); return None
        text = cache.read_text()
        print("  GFZ reference: using cached file")
    rows = []
    for line in text.splitlines():
        c = line.split()
        if len(c) < 11 or not (c[0].isdigit() and len(c[0]) == 4):
            continue
        try:
            rows.append((pd.Timestamp(int(c[0]), int(c[1]), int(c[2])),
                         float(c[-1])))                        # z-motion term
        except ValueError:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "chi3"])
    daily = df.groupby("date")["chi3"].mean()
    return (daily + GFZ_CHI3_MEAN) * GFZ_TO_KGM2S / SCALE


def _aam_era5(t):
    """One observed AAM sample, via the persistent local ERA5 store (streams
    from ARCO only on first touch; local netCDF afterwards)."""
    u = era5_store.get_u(t, LEVELS)
    sp = era5_store.get_sp(t)
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
    # ARCO-ERA5's time axis is padded with NaNs out to ~2050, so ds.time[-1] is NOT the
    # data edge — cap at today. ERA5 itself backfills ~5–14 days late.
    today = pd.Timestamp(np.datetime64("today", "D"))
    end = min(today, pd.Timestamp(year, 12, 31))
    want = pd.date_range(pd.Timestamp(year, 1, 1), end, freq="3D")
    have = pd.DataFrame(columns=["global", "nh", "sh"])
    if HIST_PATH.exists():
        h = xr.open_dataset(HIST_PATH).to_dataframe(); h.index = pd.to_datetime(h.index)
        have = h
    # Retry any wanted date that's missing OR cached as NaN: recent 3-day slots that were
    # empty on an earlier run (data not yet in ERA5) must be re-fetched as it backfills,
    # otherwise the observed history freezes a couple weeks short of the init.
    cached_ok = set(have.dropna(how="all").index)
    todo = [d for d in want if d not in cached_ok]
    if todo:
        print(f"  computing {len(todo)} ERA5 history day(s) for {year} …", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(lambda d: _aam_era5(d.strftime("%Y-%m-%dT12:00")), todo))
        new = pd.DataFrame(res, columns=["global", "nh", "sh"], index=pd.DatetimeIndex(todo))
        have = pd.concat([have[~have.index.isin(new.index)], new]).sort_index()
        have = have[~have.index.duplicated(keep="last")]
        have = have.dropna(how="all")          # never persist not-yet-available days as permanent NaN
        HIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({k: ("time", have[k].values) for k in ("global", "nh", "sh")},
                   coords={"time": have.index}).to_netcdf(HIST_PATH)
    return have[have.index.year == year].sort_index()


def _interp_doy(clim, key, region, doy):
    xp = clim.doy.values; fp = clim[key].sel(region=region).values
    return np.interp(doy, np.concatenate([xp - 365, xp, xp + 365]),
                     np.concatenate([fp, fp, fp]))


def plot(data, steps_h, init, hist, out: Path, gfz=None):
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
        if gfz is not None and key == "global":
            gd = np.array([d.dayofyear for d in gfz.index])
            a0.plot(gd, gfz.values, color="#2e7d32", lw=1.1, alpha=0.85,
                    label="GFZ ESMGFZ (independent)")
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
            if gfz is not None and key == "global":
                ga = gfz.values - eval_clim(coef, np.array([d.dayofyear for d in gfz.index]))
                a1.plot(gfz.index, ga, color="#2e7d32", lw=1.0, alpha=0.85)
            a1.axhline(0, color="0.5", lw=0.9); a1.grid(True, alpha=0.2)
            for lab in a1.get_xticklabels():
                lab.set_rotation(45); lab.set_ha("right"); lab.set_fontsize(7)
            base = float(np.nanmean(cbase))                        # anomaly as % of clim mean
            if abs(base) > 0.1:
                ax2 = a1.twinx(); lo_, hi_ = a1.get_ylim()
                ax2.set_ylim(lo_ / base * 100, hi_ / base * 100)
                ax2.set_ylabel("anomaly, % of clim mean", fontsize=8)
                ax2.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("Absolute AAM (10²⁵ kg m² s⁻¹)")
    axes[1, 0].set_ylabel("AAM anomaly (10²⁵ kg m² s⁻¹)")
    axes[0, 0].legend(fontsize=7, loc="lower left", ncol=2, framealpha=0.9)
    fig.suptitle(f"Atmospheric angular momentum (relative/wind term) — AIFS-ENS init {init:%Y-%m-%d %HZ}\n"
                 f"top: {init.year} observed + forecast on the seasonal cycle (mean, ±1σ, record range)   ·   "
                 f"bottom: anomaly vs. 1991–2020", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.text(0.5, 0.001,
             "M = (a³/g)∫∫∫ u cos²φ dλ dφ dp, full column via 14 pressure levels (10–1000 hPa; top layer 0–30 hPa) · "
             "green = GFZ ESMGFZ wind term (ECMWF full model column, independent of this pipeline) — the ongoing external verification",
             ha="center", fontsize=7.5, color="0.42")
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.1))
    handles = []
    for j, (key, title) in enumerate(REG):
        ax = axes[j]
        coef = clim["coeffs"].sel(region=key).values if clim is not None else None
        for i, it in enumerate(inits):
            fc = arch["fc_mean"].sel(region=key, init=it).values
            valid = pd.to_datetime([it + pd.Timedelta(days=int(l)) for l in arch.lead.values])
            y = fc - eval_clim(coef, np.array([d.dayofyear for d in valid])) if coef is not None else fc
            newest = i == n - 1
            if newest:                              # latest run: bold black, on top
                color, lw, alpha, z = "k", 3.2, 1.0, 6
            else:                                   # older runs: purple→orange, skipping plasma's gross-yellow top
                frac = i / max(n - 2, 1)
                color, lw, alpha, z = cmap(0.08 + 0.66 * frac), 1.1, 0.7, 2
            ln, = ax.plot(valid, y, color=color, lw=lw, alpha=alpha, zorder=z,
                          label=f"{it:%b %d} {it:%H}Z" + ("  (latest)" if newest else ""))
            if j == 0:
                handles.append(ln)
        ax.axhline(0, color="0.5", lw=0.8); ax.grid(True, alpha=0.2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(45); lab.set_ha("right"); lab.set_fontsize(7.5)
    axes[0].set_ylabel("AAM anomaly (10²⁵ kg m² s⁻¹)")
    fig.suptitle(f"AAM anomaly forecast trend — successive AIFS-ENS ensemble-mean runs "
                 f"through {inits[-1]:%Y-%m-%d %HZ} (newest = bold)", fontsize=12, fontweight="bold")
    # legend in a row UNDER the panels (not to the right → panels use the full width)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.27)
    fig.legend(handles[::-1], [h.get_label() for h in handles[::-1]],
               loc="upper center", bbox_to_anchor=(0.5, 0.13), ncol=min(n, 7),
               fontsize=8, framealpha=0.9, title="forecast init", title_fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120); plt.close(fig)
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
    plot(data, steps_h, init, hist, Path(args.out), gfz=gfz_series(init.year))
    archive_forecast(init, data, steps_h)
    plot_trend(Path(args.out).with_name("aam_trend.webp"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
