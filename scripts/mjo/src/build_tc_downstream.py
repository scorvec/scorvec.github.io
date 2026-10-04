#!/usr/bin/env python3
"""What usually follows a recurving tropical cyclone: ERA5 composites, 1991-2020 (laptop, once; static reference).

User 2026-10-02 ("1, 3 and 5 sound fine"): item 1 = historical downstream composites, item 3 = live alerts built on them.

Storms     IBTrACS v04r01 (NOAA NCEI, ~/data_archive/ibtracs/ibtracs.since1980.csv), main tracks, basin AT RECURVATURE WP/EP/NA,
           seasons 1991-2020, reaching 34 kt (WMO or USA wind), 6-hourly synoptic fixes. Recurvature = the SAME rule as the
           live card (tc_jet.recurvature: 24-h zonal motion from <= -1 to >= +2 m/s), first turn only, at 15-45N, Jun-Nov.
           Extratropical transition = first fix with NATURE 'ET' (recorded, not used to select).
Strength   the live outflow index needs T on levels (PV), which the local ERA5 store lacks. PROXY: at 200 hPa only,
           -v_chi . grad(eta) (eta = absolute vorticity, v_chi = irrotational wind by spherical-harmonic inversion, T63),
           from ERA5 daily means on the 1.5 deg grid; index = area mean of its negative part within 500 km of the storm,
           largest of recurvature day -1/0/+1, in 1e-5 s^-1 per day. Terciles per basin split strong / weak interaction.
           How the proxy tracks the live PV index is checked separately (tc_downstream_proxy_check.py).
Fields     ERA5 daily z500 (m), t2m (K), precip (mm/day) on the 1.5 deg NH grid (~/era5_store/wb2_1p5_daily); anomaly =
           value minus a per-gridpoint fit of mean + 3 annual harmonics + linear trend (z500, t2m; precip no trend), 1991-2020.
Composite  lag 0-14 days after the recurvature DAY (UTC). Storms recurving within 4 days of each other in the same basin
           are merged into one EPISODE (their lag fields averaged) - they are not independent. Significance per grid cell:
           one-sample t-test across episodes (strong minus weak: Welch), Benjamini-Hochberg FDR 10 % over each map's cells.
           Area-mean lag curves: bootstrap over episodes (4000), 90 % CI, BH across the lags of each curve.
Outputs    scripts/mjo/data/reference/tc_downstream_events.csv, tc_downstream_comp.nc (significant cells only, int16),
           tc_downstream.json (lag curves, thresholds, sample sizes). Figures: tc_downstream_render.py (Actions).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import tc_jet as TJ                                                       # noqa: E402

IBT = Path.home() / "data_archive" / "ibtracs" / "ibtracs.since1980.csv"
STORE = Path.home() / "era5_store"
REF = Path(__file__).resolve().parents[1] / "data" / "reference"
Y0, Y1 = 1991, 2020
BASINS = ("WP", "EP", "NA")
SEASONS = {"jun_nov": (6, 7, 8, 9, 10, 11), "aug_sep": (8, 9), "oct_nov": (10, 11)}
LAGS = np.arange(0, 15)
MAP_LAGS = (0, 2, 4, 6, 8, 10, 12, 14)
EPISODE_GAP = 4
A = 6.371e6
BASIN_REGIONS = {"WP": ("wcan_pnw", "west_us", "central_us", "east_us"), "EP": ("wcan_pnw", "west_us", "central_us", "east_us"),
                 "NA": ("east_us", "w_europe", "n_europe")}
REGIONS = {                                    # area means (lat0, lat1, lon0, lon1 in deg E 0..360)
    "wcan_pnw": ("Pacific Northwest & western Canada", 45, 60, 225, 250),
    "west_us": ("Western US", 32, 45, 235, 255),
    "central_us": ("Central US", 30, 48, 255, 270),
    "east_us": ("Eastern US", 30, 45, 270, 290),
    "w_europe": ("Western Europe", 42, 60, 350, 380),
    "n_europe": ("Northern Europe", 55, 70, 360, 390),
}


# ── storms ───────────────────────────────────────────────────────────────────
def events() -> pd.DataFrame:
    cols = ["SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "WMO_WIND", "USA_WIND", "TRACK_TYPE"]
    d = pd.read_csv(IBT, usecols=cols, skiprows=[1], low_memory=False, keep_default_na=False)
    d = d[(d.TRACK_TYPE == "main")]
    d["SEASON"] = pd.to_numeric(d.SEASON, errors="coerce")
    d = d[(d.SEASON >= Y0) & (d.SEASON <= Y1)].copy()
    d["t"] = pd.to_datetime(d.ISO_TIME)
    d = d[d.t.dt.hour.isin([0, 6, 12, 18]) & (d.t.dt.minute == 0)]
    for c in ("LAT", "LON", "WMO_WIND", "USA_WIND"):
        d[c] = pd.to_numeric(d[c].replace("", np.nan), errors="coerce")
    out = []
    for sid, g in d.groupby("SID", sort=False):
        g = g.sort_values("t")
        vmax = np.nanmax(np.r_[g.WMO_WIND.values, g.USA_WIND.values, 0])
        if vmax < 34:
            continue
        s = dict(steps=((g.t - g.t.iloc[0]) / pd.Timedelta(hours=1)).values.astype(int), lat=g.LAT.values,
                 lon=g.LON.values)
        r = TJ.recurvature(s)
        if r is None:
            continue
        k = int(np.where(s["steps"] == r)[0][0])
        tr = g.t.iloc[k]
        basin = g.BASIN.iloc[k]                       # the basin WHERE it recurves (Faxai 2019 formed in the EP, recurved at 139E)
        if basin not in BASINS:
            continue
        if not (15 <= s["lat"][k] <= 45) or tr.month not in SEASONS["jun_nov"] or not (Y0 <= tr.year <= Y1):
            continue
        et = g.t[g.NATURE == "ET"]
        out.append(dict(sid=sid, name=g.NAME.iloc[0], basin=basin, season=int(g.SEASON.iloc[0]), t_recurve=tr,
                        lat=round(float(s["lat"][k]), 2), lon=round(float(s["lon"][k]) % 360, 2), vmax_kt=float(vmax),
                        t_et=et.iloc[0] if len(et) else pd.NaT))
    return pd.DataFrame(out)


# ── the 200 hPa proxy for the outflow-jet index ─────────────────────────────
def _open(var, year, glob=False):
    root = STORE / ("wb2_1p5_daily_global" if glob else "wb2_1p5_daily") / var / f"{var}_{year}.nc"
    da = xr.open_dataset(root)[var]
    return da.transpose("time", "latitude", "longitude")             # the store mixes dimension orders: by NAME


def proxy_field(u2d: np.ndarray, v2d: np.ndarray, lat: np.ndarray, lon: np.ndarray, lmax=63) -> np.ndarray:
    """-v_chi . grad(eta) at 200 hPa in 1e-5 s^-1 per day on the input (global, lat ascending) grid."""
    from wind200_vpot import irrotational_wind, velocity_potential
    U = xr.DataArray(u2d, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
    V = xr.DataArray(v2d, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
    chi, dlat, dlon = velocity_potential(U, V, lmax=lmax)
    uc, vc = irrotational_wind(chi, dlat, dlon)
    dl = np.asarray(dlon, float)
    if dl[-1] < 360 - 1e-6:
        dl = np.r_[dl, dl[0] + 360]; uc = np.concatenate([uc, uc[:, :1]], 1); vc = np.concatenate([vc, vc[:, :1]], 1)
    UC = xr.DataArray(uc, dims=("la", "lo"), coords={"la": dlat, "lo": dl}).sortby("la")
    VC = xr.DataArray(vc, dims=("la", "lo"), coords={"la": dlat, "lo": dl}).sortby("la")
    tg = dict(la=xr.DataArray(lat, dims="y"), lo=xr.DataArray(lon % 360, dims="x"))
    uc, vc = UC.interp(**tg).values, VC.interp(**tg).values
    phi = np.deg2rad(lat); lam = np.deg2rad(lon)
    cosp = np.clip(np.cos(phi), 1e-3, None)[:, None]
    dlam = lam[1] - lam[0]
    ddx = lambda a: (np.roll(a, -1, 1) - np.roll(a, 1, 1)) / (2 * dlam) / (A * cosp)
    ddy = lambda a: np.gradient(a, phi, axis=0) / A
    eta = ddx(v2d) - ddy(u2d * cosp) / cosp + 2 * 7.2921e-5 * np.sin(phi)[:, None]
    adv = -(uc * ddx(eta) + vc * ddy(eta))
    adv[np.abs(lat) > 85] = np.nan
    return adv * 86400.0 * 1e5


def proxy_index(ev: pd.DataFrame) -> pd.DataFrame:
    """The proxy at each event: max over recurvature day -1..+1 of the 500 km disc mean of the negative part."""
    vals = []
    cache = {}
    for i, e in enumerate(ev.itertuples()):
        best = np.nan
        for dd in (-1, 0, 1):
            day = (e.t_recurve.normalize() + pd.Timedelta(days=dd))
            if not (Y0 <= day.year <= Y1):
                continue
            if day.year not in cache:
                cache = {day.year: (_open("u200", day.year, True), _open("v200", day.year, True))}
            u, v = cache[day.year]
            ud, vd = u.sel(time=day).values, v.sel(time=day).values
            lat, lon = u.latitude.values, u.longitude.values
            f = proxy_field(ud, vd, lat, lon)
            x = TJ.disc_index(f, lat, lon, e.lat, e.lon)
            best = x if not np.isfinite(best) else max(best, x)
        vals.append(best)
        if i % 50 == 0:
            print(f"    proxy {i}/{len(ev)}", flush=True)
    ev = ev.copy(); ev["proxy"] = np.round(vals, 4)
    return ev


# ── anomalies ────────────────────────────────────────────────────────────────
def anomalies(var: str) -> xr.DataArray:
    """Daily NH anomaly 1991-2020 (+ early 2021 for lags) against mean + 3 harmonics (+ trend except precip)."""
    das = [_open(var, y) for y in range(Y0, Y1 + 2) if (STORE / "wb2_1p5_daily" / var / f"{var}_{y}.nc").exists()]
    da = xr.concat(das, "time").sel(latitude=slice(10, 85) if das[0].latitude[0] < das[0].latitude[-1] else slice(85, 10))
    da = da.sortby("latitude").astype("float32")
    t = da.time.values
    fit_t = (pd.DatetimeIndex(t).year <= Y1)
    doy = pd.DatetimeIndex(t).dayofyear.values
    yr = (pd.DatetimeIndex(t) - pd.Timestamp(f"{Y0}-01-01")).days.values / 365.25
    w = 2 * np.pi * doy / 365.25
    X = [np.ones_like(w)]
    for k in (1, 2, 3):
        X += [np.cos(k * w), np.sin(k * w)]
    if var != "prcp":
        X.append(yr - yr[fit_t].mean())
    X = np.stack(X, 1)
    Y = da.values.reshape(len(t), -1)
    beta, *_ = np.linalg.lstsq(X[fit_t], Y[fit_t], rcond=None)
    an = (Y - X @ beta).reshape(da.shape).astype("float32")
    return xr.DataArray(an, dims=da.dims, coords=da.coords)


def bh(p: np.ndarray, q=0.10) -> np.ndarray:
    p = np.asarray(p, float); ok = np.isfinite(p); out = np.zeros(p.shape, bool)
    if not ok.any():
        return out
    pv = p[ok]; o = np.argsort(pv); m = len(pv)
    passed = pv[o] <= q * np.arange(1, m + 1) / m
    if passed.any():
        kmax = np.where(passed)[0].max(); sel = np.zeros(m, bool); sel[o[:kmax + 1]] = True
        tmp = np.zeros(m, bool); tmp[:] = sel; out[ok] = tmp
    return out


def episodes(ev: pd.DataFrame) -> list[list[int]]:
    """Group events whose recurvature days are within EPISODE_GAP days (chained) into episodes."""
    ev = ev.sort_values("t_recurve")
    out, cur, last = [], [], None
    for i, e in zip(ev.index, ev.t_recurve):
        if last is not None and (e - last).days > EPISODE_GAP:
            out.append(cur); cur = []
        cur.append(i); last = e
    if cur:
        out.append(cur)
    return out


def lagstack(an: xr.DataArray, days: list[pd.Timestamp]) -> np.ndarray:
    """(n, lag, lat, lon) anomalies after each day; NaN where beyond the record."""
    tt = pd.DatetimeIndex(an.time.values)
    pos = {t: i for i, t in enumerate(tt)}
    arr = an.values
    out = np.full((len(days), len(LAGS)) + arr.shape[1:], np.nan, np.float32)
    for i, d in enumerate(days):
        for j, L in enumerate(LAGS):
            k = pos.get(d + pd.Timedelta(days=int(L)))
            if k is not None:
                out[i, j] = arr[k]
    return out


def ttest1(x):                                             # (n, ...) -> p
    from scipy import stats
    return stats.ttest_1samp(x, 0.0, axis=0, nan_policy="omit").pvalue


def welch(a, b):
    from scipy import stats
    return stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit").pvalue


def storm_relative(an, ev, res, comp_vars):
    """z500 composites with every field shifted so the recurvature longitude sits at relative longitude 0 (the downstream
    ridge/trough forms relative to the STORM; Earth-relative compositing smears it over the 100-175E spread of
    recurvature points). Latitude is not shifted. Stored on a relative-longitude axis -180..178.5."""
    for b in BASINS:
        for sk, months in SEASONS.items():
            base = ev[(ev.basin == b) & ev.month.isin(months)]
            groups = {"all": base, "strong": base[base.tercile == "strong"], "weak": base[base.tercile == "weak"]}
            E = {}
            for gk, gdf in groups.items():
                if len(episodes(gdf)) < 8:
                    E[gk] = None; continue
                order = gdf.sort_values("t_recurve")
                st = lagstack(an, list(order.day))
                sh = np.round(order.lon.values / 1.5).astype(int)
                st = np.stack([np.roll(st[i], -sh[i] + 120, axis=-1) for i in range(len(order))])   # rel lon 0 at index 120
                pos = {i: n for n, i in enumerate(order.index)}
                E[gk] = np.stack([np.nanmean(st[[pos[i] for i in ep]], 0) for ep in episodes(gdf)])
            info = res["sets"][f"{b}|{sk}"]
            for gk in ("all", "diff"):
                if gk == "all":
                    if E["all"] is None:
                        continue
                    mean = np.nanmean(E["all"], 0); p = ttest1(E["all"])
                else:
                    if E["strong"] is None or E["weak"] is None:
                        continue
                    mean = np.nanmean(E["strong"], 0) - np.nanmean(E["weak"], 0); p = welch(E["strong"], E["weak"])
                keep = np.zeros_like(mean, bool)
                dom = np.zeros(mean.shape[-1], bool); dom[120 - 40:120 + 101] = True      # -60..+150 deg relative
                for j in range(len(LAGS)):
                    kj = np.zeros(mean.shape[1:], bool)
                    kj[:, dom] = bh(p[j][:, dom].ravel()).reshape(p[j][:, dom].shape)
                    keep[j] = kj
                sig = np.where(keep, mean, np.nan)
                for L in MAP_LAGS:
                    comp_vars[f"z500rel|{b}|{sk}|{gk}|{L}"] = sig[L]
                info.setdefault("sig_frac", {})[f"z500rel|{gk}"] = [round(float(keep[j][:, dom].mean()), 4) for j in range(len(LAGS))]
            info["recurve_lat_median"] = round(float(base.lat.median()), 1)


def main() -> int:
    t0 = time.time()
    REF.mkdir(parents=True, exist_ok=True)
    evf = REF / "tc_downstream_events.csv"
    if evf.exists() and "--fresh" not in sys.argv:
        ev = pd.read_csv(evf, parse_dates=["t_recurve", "t_et"], keep_default_na=False, na_values=[""])   # basin "NA" is not NaN
    else:
        ev = events(); print(f"  {len(ev)} recurving storms ({ev.basin.value_counts().to_dict()})", flush=True)
        ev = proxy_index(ev)
        ev.to_csv(evf, index=False)
    ev["day"] = ev.t_recurve.dt.normalize()
    ev["month"] = ev.t_recurve.dt.month
    # strong / weak terciles of the proxy, per basin (all months together, so a season's thresholds share one scale)
    thr = {}
    ev["tercile"] = ""
    for b in BASINS:
        m = ev.basin == b
        q1, q2 = np.nanpercentile(ev.loc[m, "proxy"], [33.33, 66.67])
        thr[b] = dict(p33=round(float(q1), 4), p67=round(float(q2), 4))
        ev.loc[m & (ev.proxy >= q2), "tercile"] = "strong"
        ev.loc[m & (ev.proxy <= q1), "tercile"] = "weak"
    print("  proxy terciles", thr, flush=True)

    res = {"generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d"), "years": [Y0, Y1], "lags": LAGS.tolist(),
           "map_lags": list(MAP_LAGS), "thresholds": thr, "regions": {k: v[0] for k, v in REGIONS.items()},
           "proxy_units": "1e-5 s^-1 per day", "sets": {}}
    comp_vars = {}
    for var in ("z500", "t2m", "prcp"):
        an = anomalies(var); print(f"  {var} anomalies {dict(an.sizes)} ({time.time() - t0:.0f}s)", flush=True)
        lat, lon = an.latitude.values, an.longitude.values
        lon360 = lon % 360
        for b in BASINS:
            for sk, months in SEASONS.items():
                base = ev[(ev.basin == b) & ev.month.isin(months)]
                groups = {"all": base, "strong": base[base.tercile == "strong"], "weak": base[base.tercile == "weak"]}
                E = {}
                for gk, gdf in groups.items():
                    eps = episodes(gdf)
                    if len(eps) < 8:
                        E[gk] = None; continue
                    order = gdf.sort_values("t_recurve")
                    st = lagstack(an, list(order.day))
                    pos = {i: n for n, i in enumerate(order.index)}
                    E[gk] = np.stack([np.nanmean(st[[pos[i] for i in ep]], 0) for ep in eps])
                key = f"{b}|{sk}"
                info = res["sets"].setdefault(key, {"basin": b, "season": sk,
                                                    "n_storms": {g: int(len(groups[g])) for g in groups},
                                                    "n_episodes": {g: (None if E[g] is None else int(len(E[g]))) for g in groups},
                                                    "curves": {}})
                for gk in ("all", "diff"):
                    if gk == "all":
                        if E["all"] is None:
                            continue
                        mean = np.nanmean(E["all"], 0); p = ttest1(E["all"])
                    else:
                        if E["strong"] is None or E["weak"] is None:
                            continue
                        mean = np.nanmean(E["strong"], 0) - np.nanmean(E["weak"], 0); p = welch(E["strong"], E["weak"])
                    keep = np.zeros_like(mean, bool)
                    for j in range(len(LAGS)):
                        keep[j] = bh(p[j].ravel()).reshape(p[j].shape)
                    sig = np.where(keep, mean, np.nan)
                    for L in MAP_LAGS:
                        comp_vars[f"{var}|{key}|{gk}|{L}"] = sig[L]
                    info.setdefault("sig_frac", {})[f"{var}|{gk}"] = [round(float(keep[j].mean()), 4) for j in range(len(LAGS))]
                # area-mean lag curves (all-storm composite and strong-minus-weak), bootstrap over episodes. Only the
                # basin's own downstream regions; the false-discovery test runs JOINTLY over those regions x 15 lags.
                rng = np.random.default_rng(7)
                for gk in ("all", "diff"):
                    if gk == "all" and E["all"] is None:
                        continue
                    if gk == "diff" and (E["strong"] is None or E["weak"] is None):
                        continue
                    pend = {}
                    for rk in BASIN_REGIONS[b]:
                        _, la0, la1, lo0, lo1 = REGIONS[rk]
                        mj = (lat >= la0) & (lat <= la1)
                        mi = ((lon360 - lo0) % 360 <= (lo1 - lo0))
                        wj = np.cos(np.deg2rad(lat[mj]))
                        am = lambda X: np.nansum(np.nanmean(X[:, :, mj][:, :, :, mi], 3) * wj, 2) / wj.sum()
                        if gk == "all":
                            a_ = am(E["all"]); mean = a_.mean(0)
                            boots = np.stack([a_[rng.integers(0, len(a_), len(a_))].mean(0) for _ in range(4000)])
                        else:
                            s_, w_ = am(E["strong"]), am(E["weak"]); mean = s_.mean(0) - w_.mean(0)
                            boots = np.stack([s_[rng.integers(0, len(s_), len(s_))].mean(0) - w_[rng.integers(0, len(w_), len(w_))].mean(0)
                                              for _ in range(4000)])
                        lo_, hi_ = np.percentile(boots, [5, 95], 0)
                        p = np.minimum(1, 2 * np.minimum((boots <= 0).mean(0), (boots >= 0).mean(0)))
                        pend[rk] = (mean, lo_, hi_, p)
                    keep = bh(np.concatenate([v[3] for v in pend.values()])).reshape(len(pend), -1)
                    for (rk, (mean, lo_, hi_, p)), sg in zip(pend.items(), keep):
                        info["curves"][f"{var}|{rk}|{gk}"] = dict(mean=np.round(mean, 3).tolist(), lo=np.round(lo_, 3).tolist(),
                                                                  hi=np.round(hi_, 3).tolist(), sig=sg.tolist())
        if var == "z500":
            storm_relative(an, ev, res, comp_vars)
        print(f"  {var} done ({time.time() - t0:.0f}s)", flush=True)
        lat_keep, lon_keep = lat, lon
        del an
    # pack the significant maps: one int16 variable per (var, basin, season, group), dims (lag, lat, lon)
    scale = {"z500": 0.1, "z500rel": 0.1, "t2m": 0.01, "prcp": 0.01}
    ds = xr.Dataset(coords={"lag": list(MAP_LAGS), "latitude": lat_keep, "longitude": lon_keep})
    names = sorted({k.rsplit("|", 1)[0] for k in comp_vars})
    for nm in names:
        var = nm.split("|")[0]
        arr = np.stack([comp_vars.get(f"{nm}|{L}", np.full((len(lat_keep), len(lon_keep)), np.nan)) for L in MAP_LAGS])
        q = np.where(np.isfinite(arr), np.round(arr / scale[var]), -32768).clip(-32768, 32767).astype("int16")
        vname = nm.replace("|", "__")
        ds[vname] = (("lag", "latitude", "longitude"), q)
        ds[vname].attrs.update(scale_factor=scale[var], _FillValue=np.int16(-32768),
                               units={"z500": "m", "z500rel": "m", "t2m": "K", "prcp": "mm/day"}[var])
        if var == "z500rel":
            ds[vname].attrs["longitude_axis"] = "relative to the recurvature longitude: value = longitude coordinate - 180"
    enc = {v: {"zlib": True, "complevel": 6} for v in ds.data_vars}
    ds.attrs.update(description="Significant (BH FDR 10%) composite anomalies after TC recurvature; see build_tc_downstream.py")
    ds.to_netcdf(REF / "tc_downstream_comp.nc", encoding=enc)
    old = REF / "tc_downstream.json"
    if old.exists() and "proxy_check" in json.loads(old.read_text()):      # written by tc_downstream_proxy_check.py
        res["proxy_check"] = json.loads(old.read_text())["proxy_check"]
    (REF / "tc_downstream.json").write_text(json.dumps(res))
    print(f"  wrote {REF / 'tc_downstream_comp.nc'} ({(REF / 'tc_downstream_comp.nc').stat().st_size / 1e6:.1f} MB) "
          f"and tc_downstream.json in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
