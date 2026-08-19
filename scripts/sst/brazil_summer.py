#!/usr/bin/env python3
"""P(warmest population-weighted Brazilian summer on record), from C3S.

Target: the DJF mean 2-m temperature over Brazil, weighted by metro
population — the quantity that drives cooling demand, not an area mean
over the Amazon.  Weights are the 14 metros already used by the
degree-day tracker (~79 M people), so this is consistent with the rest of
the stack.

Observed record: ERA5 (1.5 deg global store), DJF 2000/01 .. 2025/26 —
26 summers.  "On record" therefore means *in the ERA5 record we hold*,
not all time; stated on the page rather than implied.

BIAS CORRECTION.  A seasonal model's absolute temperature is unusable —
its climatology is offset from observed and drifts with lead.  Each
member is converted to an anomaly against that system's OWN hindcast
climatology for the same (target month, lead), then added to the ERA5
observed climatology.  Model drift cancels; only the anomaly crosses.
The hindcast is retrieved for exactly the same init month and leads as
the forecast, so the drift being removed is the drift that applies.

    python scripts/sst/brazil_summer.py

Outputs: ~/colombia_hydro/out/brazil_summer.json
         ~/colombia_hydro/site/brazil_summer.webp
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PRIV = Path.home() / "colombia_hydro"
CACHE = PRIV / "raw" / "c3s_t2m_brazil"
ERA5 = Path.home() / "era5_store" / "wb2_1p5_daily_global" / "t2m"
OUT_JSON = PRIV / "out" / "brazil_summer.json"
OUT_PNG = PRIV / "site" / "brazil_summer.webp"
AREA = [6, -74, -34, -34]            # N, W, S, E — Brazil
SEASON = (11, 12, 1, 2, 3)           # NDJFM — the full warm season.  From an
                                     # August init that is leads 3-7, so the
                                     # far end sits at the edge of SEAS5's
                                     # 7-month horizon and is the least
                                     # reliable month of the five.
_MLET = {11: "N", 12: "D", 1: "J", 2: "F", 3: "M"}
SEASON_NAME = "".join(_MLET[m] for m in SEASON)      # label follows the data


def season_label(months):
    """Label the months actually available, so a truncated season is not
    advertised as the full one — SEAS5 runs 6 leads, so an August init
    cannot reach March and NDJFM silently becomes NDJF."""
    return "".join(_MLET[m] for m in SEASON if m in months)
GRID = "1.0/1.0"
CENTRE, SYSTEM = "ecmwf", "51"
HIND_YEARS = [str(y) for y in range(1993, 2017)]
# metro population weights, identical to scripts/energy/dd_surprise.py
METROS = [("Sao Paulo", -23.55, -46.6, 22.0), ("Rio", -22.9, -43.2, 13.6),
          ("Belo Horizonte", -19.9, -43.9, 6.1), ("Brasilia", -15.8, -47.9, 4.7),
          ("Fortaleza", -3.7, -38.5, 4.2), ("Salvador", -13.0, -38.5, 3.9),
          ("Recife", -8.05, -34.9, 4.2), ("Curitiba", -25.4, -49.3, 3.7),
          ("Porto Alegre", -30.0, -51.2, 4.3), ("Manaus", -3.1, -60.0, 2.7),
          ("Belem", -1.45, -48.5, 2.5), ("Goiania", -16.7, -49.3, 2.7),
          ("Campinas", -22.9, -47.1, 3.3), ("Sao Luis", -2.5, -44.3, 1.6)]


def retrieve(years, month, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    import cdsapi
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = {"originating_centre": CENTRE, "system": SYSTEM,
           "variable": "2m_temperature", "product_type": "monthly_mean",
           "year": list(years), "month": [month],
           "leadtime_month": ["3", "4", "5", "6", "7"],   # Aug init -> Nov..Mar
           "area": AREA, "grid": GRID, "data_format": "grib"}
    try:
        cdsapi.Client(timeout=1800, quiet=True, progress=False,
                      wait_until_complete=True, retry_max=1
                      ).retrieve("seasonal-monthly-single-levels", req, str(dest))
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:                              # noqa: BLE001
        print(f"  retrieve failed: {repr(e)[:140]}", flush=True)
        return False


def pop_weight(lats, lons):
    """Weight grid → metro populations at the nearest gridpoint."""
    lons = np.asarray(lons); lats = np.asarray(lats)
    lo = ((lons + 180) % 360) - 180
    Wt = np.zeros((len(lats), len(lons)))
    for _, la, ln, pop in METROS:
        i = int(np.argmin(np.abs(lats - la)))
        j = int(np.argmin(np.abs(lo - ln)))
        Wt[i, j] += pop
    return Wt / Wt.sum()


def c3s_djf(path):
    """{(init_year, target_month): [members]} pop-weighted degC."""
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds["t2m"] - 273.15
    lon = da.longitude.values
    if lon.max() > 180:
        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180)
    da = da.sortby("longitude").sortby("latitude")
    Wt = pop_weight(da.latitude.values, da.longitude.values)
    for d_ in ("number", "time", "step"):
        if d_ not in da.dims:
            da = da.expand_dims(d_)
    da = da.transpose("number", "time", "step", "latitude", "longitude")
    arr = da.values
    times = np.atleast_1d(np.asarray(da["time"].values, dtype="datetime64[D]"))
    vt = np.atleast_2d(np.asarray(ds["valid_time"].values, dtype="datetime64[D]"))
    if vt.shape[0] != len(times) and vt.shape[1] == len(times):
        vt = vt.T
    if vt.shape[0] == 1 and len(times) > 1:
        vt = np.repeat(vt, len(times), axis=0)
    out = {}
    wf = Wt.ravel()
    for ti in range(arr.shape[1]):
        iy = int(str(times[ti])[:4])
        for si in range(arr.shape[2]):
            if si >= vt.shape[1]:
                continue
            v = vt[ti, si]
            tm = int(str(v)[5:7])
            if tm not in SEASON:
                continue
            vals = []
            for mi in range(arr.shape[0]):
                g = arr[mi, ti, si].ravel()
                if not np.isfinite(g).any():
                    continue
                vals.append(float(np.dot(np.nan_to_num(g), wf)))
            if vals:
                out.setdefault((iy, tm), []).extend(vals)
    return out


def era5_djf():
    """Observed pop-weighted DJF means, keyed by the DJF start year."""
    import xarray as xr
    files = sorted(ERA5.glob("t2m_*.nc"))
    if not files:
        return {}
    ds0 = xr.open_dataset(files[0])
    Wt = pop_weight(ds0.latitude.values, ds0.longitude.values).ravel()
    daily = {}
    for f in files:
        ds = xr.open_dataset(f)
        t = np.asarray(ds["time"].values, dtype="datetime64[D]")
        # the store is (time, longitude, latitude) — transpose, never reshape
        a = ds["t2m"].transpose("time", "latitude", "longitude").values
        if np.nanmax(a) > 200:
            a = a - 273.15
        v = a.reshape(a.shape[0], -1) @ Wt
        for d, x in zip(t, v):
            if np.isfinite(x):
                daily[str(d)] = float(x)
    out = {}
    for d, x in daily.items():
        y, m = int(d[:4]), int(d[5:7])
        if m not in SEASON:
            continue
        # keyed by the season's START year: Nov-Dec of Y, Jan-Mar of Y+1
        out.setdefault(y if m >= 11 else y - 1, []).append(x)
    need = 28 * len(SEASON)
    return {y: float(np.mean(v)) for y, v in out.items() if len(v) >= need}


def crps_gauss(mu, sig, y):
    from math import sqrt, pi, erf, exp
    z = (y - mu) / sig
    return float(sig * (z * (2 * (0.5 * (1 + erf(z / sqrt(2)))) - 1)
                        + 2 * (exp(-0.5 * z * z) / sqrt(2 * pi)) - 1 / sqrt(pi)))


def emos_fit(fmean, fsd, obs, yrs, embargo=1):
    """Calibrate the seasonal temperature ensemble: obs ~ N(a+b*mean, c^2+d*var).

    The same treatment applied to C3S rainfall, for the same reason: a raw
    seasonal ensemble usually has an over-confident mean and a spread that
    does not represent the error, so a raw tail probability like
    "P(warmest on record)" is not trustworthy.  Fitted by direct CRPS
    minimisation, leave-one-year-out with a +/-1 year embargo, on the
    hindcast years that overlap the ERA5 store.  Returns the fitted
    parameters plus the out-of-sample CRPS skill they achieve.
    """
    from scipy.optimize import minimize
    fmean, fsd, obs, yrs = map(np.asarray, (fmean, fsd, obs, yrs))
    mu = np.full(len(obs), np.nan); sg = np.full(len(obs), np.nan)
    par = []
    for y in yrs:
        te = yrs == y
        tr = np.abs(yrs - y) > embargo
        if tr.sum() < 8:
            continue

        def loss(pp, tr=tr):
            a, b, c, dd = pp
            v = np.sqrt(c ** 2 + dd ** 2 * fsd[tr] ** 2)
            m = a + b * fmean[tr]
            return float(np.mean([crps_gauss(mi, vi, oi)
                                  for mi, vi, oi in zip(m, v, obs[tr])]))

        # BOUNDED: with only ~16 overlapping years an unbounded Nelder-Mead
        # runs the spread coefficient off to nonsense (d = -2849 was the
        # first attempt).  c and d enter squared so variance stays positive,
        # and both are held to physically sensible ranges.
        r = minimize(loss, x0=[float(np.mean(obs[tr]) - np.mean(fmean[tr])),
                               1.0, float(np.std(obs[tr])), 1.0],
                     method="L-BFGS-B",
                     bounds=[(-10, 10), (0.0, 2.0), (0.05, 3.0), (0.0, 3.0)])
        a, b, c, dd = r.x
        c, dd = abs(c), abs(dd)
        par.append(r.x)
        mu[te] = a + b * fmean[te]
        sg[te] = np.sqrt(c ** 2 + dd ** 2 * fsd[te] ** 2)
    ok = np.isfinite(mu)
    if ok.sum() < 8:
        return None
    cr = float(np.mean([crps_gauss(m, s_, o)
                        for m, s_, o in zip(mu[ok], sg[ok], obs[ok])]))
    sd0, m0 = float(np.std(obs[ok])), float(np.mean(obs[ok]))
    crc = float(np.mean([crps_gauss(m0, sd0, o) for o in obs[ok]]))
    P = np.mean(np.asarray(par), axis=0)
    return {"a": round(float(P[0]), 3), "b": round(float(P[1]), 3),
            "c": round(float(P[2]), 3), "d": round(float(P[3]), 3),
            "n_years": int(ok.sum()), "crps": round(cr, 3),
            "crps_clim": round(crc, 3),
            "crps_skill": round(1 - cr / crc, 3),
            "raw_ens_sd": round(float(np.mean(fsd)), 3),
            "calibrated_sd": round(float(np.mean(sg[ok])), 3)}


def anomaly_map(fc_path, hc_path, out_png, season_name):
    """Gridded seasonal temperature anomaly, forecast minus own hindcast."""
    import xarray as xr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def seas_mean(path):
        ds = xr.open_dataset(path, engine="cfgrib",
                             backend_kwargs={"indexpath": ""})
        da = ds["t2m"] - 273.15
        lon = da.longitude.values
        if lon.max() > 180:
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180)
        da = da.sortby("longitude").sortby("latitude")
        for d_ in ("number", "time", "step"):
            if d_ not in da.dims:
                da = da.expand_dims(d_)
        da = da.transpose("number", "time", "step", "latitude", "longitude")
        vt = np.atleast_2d(np.asarray(ds["valid_time"].values,
                                      dtype="datetime64[D]"))
        times = np.atleast_1d(np.asarray(da["time"].values, dtype="datetime64[D]"))
        if vt.shape[0] != len(times) and vt.shape[1] == len(times):
            vt = vt.T
        if vt.shape[0] == 1 and len(times) > 1:
            vt = np.repeat(vt, len(times), axis=0)
        arr = da.values
        keep = []
        for ti in range(arr.shape[1]):
            for si in range(min(arr.shape[2], vt.shape[1])):
                if int(str(vt[ti, si])[5:7]) in SEASON:
                    keep.append(arr[:, ti, si])
        if not keep:
            return None, None, None
        g = np.nanmean(np.stack(keep), axis=(0, 1))     # over season and members
        return g, da.latitude.values, da.longitude.values

    fg, la, lo = seas_mean(fc_path)
    hg, _, _ = seas_mean(hc_path)
    if fg is None or hg is None:
        return False
    anom = fg - hg
    fig = plt.figure(figsize=(9.4, 8.6))
    ax = fig.add_axes([0.08, 0.09, 0.80, 0.80])
    lim = float(np.nanpercentile(np.abs(anom), 99))
    lim = max(lim, 0.5)
    pc = ax.pcolormesh(lo, la, anom, cmap="RdYlBu_r", vmin=-lim, vmax=lim,
                       shading="auto")
    cs = ax.contour(lo, la, anom, levels=np.arange(-4, 4.1, 0.5),
                    colors="k", linewidths=0.35, alpha=0.45)
    ax.clabel(cs, fmt="%.1f", fontsize=6.5)
    for name, mlat, mlon, pop in METROS:
        ax.plot(mlon, mlat, "o", ms=3 + pop ** 0.5, color="k",
                markerfacecolor="white", markeredgewidth=1.1, zorder=5)
        ha = "right" if mlon > lo.max() - 6 else "left"
        dx = -0.7 if ha == "right" else 0.7
        ax.text(mlon + dx, mlat + 0.35, name, fontsize=7.6, color="k",
                zorder=6, ha=ha)
    ax.set_xlim(lo.min(), lo.max()); ax.set_ylim(la.min(), la.max())
    ax.set_xlabel("longitude", fontsize=8.5); ax.set_ylabel("latitude", fontsize=8.5)
    ax.set_title(f"{season_name} 2-m temperature anomaly — C3S SEAS5 ensemble mean\n"
                 "against its own 1993–2016 hindcast (marker size = metro population)",
                 fontsize=11, fontweight="bold", loc="left", color="#1a2733")
    ax.tick_params(labelsize=8); ax.grid(lw=0.2, alpha=0.35)
    cb = fig.colorbar(pc, ax=ax, fraction=0.043, pad=0.02)
    cb.set_label("anomaly, °C", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.savefig(out_png, dpi=125)
    plt.close(fig)
    return True


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    fc_p = CACHE / "fcst_202608.grib"
    hc_p = CACHE / "hind_08.grib"
    now = datetime.now(timezone.utc)
    ok_f = retrieve([str(now.year)], "08", fc_p)
    ok_h = retrieve(HIND_YEARS, "08", hc_p)
    print(f"forecast {'ok' if ok_f else 'FAILED'} · hindcast "
          f"{'ok' if ok_h else 'FAILED'}", flush=True)
    if not (ok_f and ok_h):
        return 1

    obs = era5_djf()
    yrs = sorted(y for y in obs if y >= 2000 and y <= now.year - 1)
    print(f"ERA5 observed {SEASON_NAME} pop-weighted: {yrs[0]}/{str(yrs[0]+1)[2:]}.."
          f"{yrs[-1]}/{str(yrs[-1]+1)[2:]}  ({len(yrs)} summers)", flush=True)
    ranked = sorted(yrs, key=lambda y: -obs[y])
    for y in ranked[:5]:
        print(f"   {y}/{str(y+1)[2:]}  {obs[y]:.2f} °C")

    hc = c3s_djf(hc_p)
    fc = c3s_djf(fc_p)
    print(f"C3S hindcast cases {len(hc)} · forecast cases {len(fc)} "
          f"({sorted({k[1] for k in fc})} target months)", flush=True)
    if not hc or not fc:
        print("C3S parse produced nothing — aborting"); return 1
    # hindcast climatology per target month, and observed climatology
    hclim = {m: float(np.mean([v for (iy, tm), vv in hc.items() if tm == m
                               for v in vv])) for m in SEASON}
    # observed monthly climatology on the same hindcast years
    import xarray as xr
    files = sorted(ERA5.glob("t2m_*.nc"))
    ds0 = xr.open_dataset(files[0])
    Wt = pop_weight(ds0.latitude.values, ds0.longitude.values).ravel()
    omon = {}
    for f in files:
        ds = xr.open_dataset(f)
        t = np.asarray(ds["time"].values, dtype="datetime64[D]")
        a = ds["t2m"].transpose("time", "latitude", "longitude").values
        if np.nanmax(a) > 200:
            a = a - 273.15
        v = a.reshape(a.shape[0], -1) @ Wt
        for d, x in zip(t, v):
            # the current year's file is padded with NaN past the last
            # observed day — including those would poison the climatology
            if not np.isfinite(x):
                continue
            m = int(str(d)[5:7])
            if m in SEASON:
                omon.setdefault(m, []).append(float(x))
    oclim = {m: float(np.mean(v)) for m, v in omon.items()}

    # bias-corrected member DJF means for the coming summer
    per_month = {}
    for (iy, tm), vals in fc.items():
        per_month[tm] = np.asarray(vals, float) - hclim[tm] + oclim[tm]
    missing = [m for m in SEASON if m not in per_month]
    if missing:
        print(f"  forecast missing months {missing} — season truncated to "
              f"{sorted(per_month)}", flush=True)
    have = [m for m in SEASON if m in per_month]
    label = season_label(have)
    n = min(len(per_month[m]) for m in have)
    djf = np.mean([per_month[m][:n] for m in have], axis=0)
    # ---- EMOS on the seasonal mean, fitted on the hindcast overlap ----
    hy = {}
    for (iy, tm), vals in hc.items():
        if tm not in SEASON:
            continue
        sy = iy if tm >= 11 else iy - 1          # season start year
        hy.setdefault(sy, {})[tm] = np.asarray(vals, float) - hclim[tm] + oclim[tm]
    fm, fs, ob, oy = [], [], [], []
    for sy, bym in sorted(hy.items()):
        if not all(m in bym for m in have) or sy not in obs:
            continue
        k = min(len(bym[m]) for m in have)
        ens = np.mean([bym[m][:k] for m in have], axis=0)
        fm.append(float(np.mean(ens))); fs.append(float(np.std(ens)))
        ob.append(obs[sy]); oy.append(sy)
    cal = emos_fit(fm, fs, ob, oy) if len(ob) >= 12 else None
    if cal:
        print(f"EMOS on {cal['n_years']} hindcast years: b={cal['b']:.2f} "
              f"d={cal['d']:.2f} · CRPS skill {cal['crps_skill']:+.3f} · "
              f"sd {cal['raw_ens_sd']:.2f} -> {cal['calibrated_sd']:.2f} °C",
              flush=True)

    record = max(obs[y] for y in yrs)
    rec_year = max(yrs, key=lambda y: obs[y])
    p_record = float(np.mean(djf > record))
    p_record_cal = None
    if cal:
        from math import erf, sqrt
        mu_c = cal["a"] + cal["b"] * float(np.mean(djf))
        sg_c = float(np.sqrt(cal["c"] ** 2 + cal["d"] ** 2
                             * float(np.std(djf)) ** 2))
        p_record_cal = float(1 - 0.5 * (1 + erf((record - mu_c) /
                                                (sg_c * sqrt(2)))))
        top3 = sorted([obs[y] for y in yrs])[-3]
        p_top3_cal = float(1 - 0.5 * (1 + erf((top3 - mu_c) / (sg_c * sqrt(2)))))
    clim_recent = float(np.mean([obs[y] for y in yrs[-10:]]))

    out = {"generated": now.strftime("%Y-%m-%d %H:%M UTC"),
           "target": label + " 2m temperature over Brazil, metro-pop weighted",
           "season_label": label, "season_requested": SEASON_NAME,
           "season_months": list(SEASON), "season": SEASON_NAME,
           "metros": len(METROS),
           "pop_millions": round(sum(m[3] for m in METROS), 1),
           "observed_record_period": f"{yrs[0]}/{str(yrs[0]+1)[2:]}"
                                     f"..{yrs[-1]}/{str(yrs[-1]+1)[2:]}",
           "observed": {f"{y}": round(obs[y], 3) for y in yrs},
           "record_value_c": round(record, 3),
           "record_summer": f"{rec_year}/{str(rec_year+1)[2:]}",
           "forecast_members": int(len(djf)),
           "forecast_p10_p50_p90": [round(float(np.percentile(djf, p)), 2)
                                    for p in (10, 50, 90)],
           "p_warmest_on_record_raw": round(p_record, 3),
           "p_warmest_on_record": (round(p_record_cal, 3)
                                   if cal and cal["crps_skill"] > 0
                                   else round(p_record, 3)),
           "emos_validated": bool(cal and cal["crps_skill"] > 0),
           "emos": cal,
           "emos_mu_sigma": [round(mu_c, 2), round(sg_c, 2)] if cal else None,
           "p_top3_raw": round(float(np.mean(djf > sorted(
               [obs[y] for y in yrs])[-3])), 3),
           "p_top3": (round(p_top3_cal, 3) if cal and cal["crps_skill"] > 0
                      else round(float(np.mean(djf > sorted(
                          [obs[y] for y in yrs])[-3])), 3)),
           "anomaly_vs_last10_c": round(float(np.median(djf)) - clim_recent, 2),
           "bias_correction": "member anomaly vs the system's own 1993-2016 "
                              "hindcast climatology, added to the ERA5 climatology"}
    print(f"\nforecast {label} {now.year}/{str(now.year+1)[2:]}: "
          f"p10/p50/p90 = {out['forecast_p10_p50_p90']} °C")
    print(f"record is {record:.2f} °C ({out['record_summer']})")
    if cal:
        print(f"\nRAW  P(warmest) = {p_record*100:.0f}%   "
              f"EMOS-calibrated = {p_record_cal*100:.0f}%")
    print(f"\nP(warmest on record) = {out['p_warmest_on_record']*100:.0f}%   "
          f"P(top-3) = {out['p_top3']*100:.0f}%   "
          f"median anomaly vs last 10 summers {out['anomaly_vs_last10_c']:+.2f} °C")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    try:
        mp = PRIV / "site" / "brazil_summer_map.webp"
        if anomaly_map(fc_p, hc_p, mp, label):
            print(f"wrote {mp}")
            out["map"] = str(mp.name)
            OUT_JSON.write_text(json.dumps(out, indent=1))
    except Exception as e:                              # noqa: BLE001
        print(f"map failed: {repr(e)[:150]}")
    try:
        figure(out, obs, yrs, djf, record)
    except Exception as e:                              # noqa: BLE001
        print(f"figure failed: {repr(e)[:150]}")
    print(f"wrote {OUT_JSON}")
    return 0


def figure(out, obs, yrs, djf, record):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, INK = "#13273d", "#1a2733"
    fig = plt.figure(figsize=(11.69, 8.27))
    hd = fig.add_axes([0, 0.935, 1, 0.065]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.015, 0.62, "BRAZIL — WARMEST SUMMER ON RECORD?",
            transform=hd.transAxes, color="white", fontsize=14.5,
            fontweight="bold", va="center")
    hd.text(0.015, 0.20, f"{out.get('season_label', SEASON_NAME)} 2-m temperature, weighted by metro population "
            f"({out['metros']} metros, {out['pop_millions']:.0f} M) · "
            f"C3S bias-corrected against its own 1993-2016 hindcast",
            transform=hd.transAxes, color="#b9c6d4", fontsize=8.6, va="center")
    ax = fig.add_axes([0.075, 0.45, 0.60, 0.42])
    v = [obs[y] for y in yrs]
    ax.bar(yrs, v, color=["#c62828" if x >= record - 1e-9 else "#9db8d8" for x in v])
    ax.axhline(record, color="#c62828", lw=1.2, ls="--",
               label=f"record {record:.2f} °C ({out['record_summer']})")
    p = out["forecast_p10_p50_p90"]
    ny = yrs[-1] + 1
    ax.errorbar([ny], [p[1]], yerr=[[p[1] - p[0]], [p[2] - p[1]]], fmt="o",
                color="#b35806", ms=9, lw=2.2, capsize=5,
                label=f"forecast {ny}/{str(ny+1)[2:]} (10–90%)")
    ax.set_ylim(min(v) - 0.5, max(max(v), p[2]) + 0.4)
    ax.set_ylabel(f"{out.get('season_label', SEASON_NAME)} mean, °C (pop-weighted)", fontsize=9)
    ax.set_xlabel("season start year (Nov)", fontsize=9)
    ax.set_title("Observed summers and the coming one", fontsize=11,
                 fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=8); ax.grid(lw=0.25, alpha=0.5, axis="y")
    ax.tick_params(labelsize=8)

    ax2 = fig.add_axes([0.72, 0.45, 0.24, 0.42])
    ax2.hist(djf, bins=18, orientation="horizontal", color="#b35806", alpha=0.85)
    ax2.axhline(record, color="#c62828", lw=1.4, ls="--")
    ax2.set_title("member spread", fontsize=10, fontweight="bold",
                  loc="left", color=INK)
    ax2.set_xlabel("members", fontsize=8.5)
    ax2.tick_params(labelsize=8); ax2.grid(lw=0.25, alpha=0.5, axis="y")
    ax2.set_ylim(ax.get_ylim())

    ax3 = fig.add_axes([0.075, 0.06, 0.885, 0.32]); ax3.set_axis_off()
    ax3.text(0, 0.98, f"P(warmest on record) = {out['p_warmest_on_record']*100:.0f}%"
             f"      P(top-3) = {out['p_top3']*100:.0f}%",
             fontsize=17, fontweight="bold", color="#c62828", va="top")
    lines = [
        f"Median forecast {p[1]:.2f} °C, {out['anomaly_vs_last10_c']:+.2f} °C "
        f"against the last ten summers; the record is {record:.2f} °C in "
        f"{out['record_summer']}.",
        "\"On record\" means within the ERA5 store held here — "
        f"{out['observed_record_period']}, {len(yrs)} summers — not all time.",
        "Population weighting uses the 14 metros of the degree-day tracker, so a hot "
        "Amazon does not outvote São Paulo. It is metro-weighted, not gridded "
        "population, so smaller cities are unrepresented.",
        "Members are bias-corrected as anomalies against the system's own 1993-2016 "
        "hindcast for the same init month and lead, then added to the ERA5 "
        "climatology — raw seasonal temperature is offset and drifts with lead.",
        "Single system (ECMWF SEAS5) and a 51-member spread that seasonal "
        "verification generally shows to be under-dispersed: treat the probability "
        "as indicative, and note it is conditioned on an El Niño already forecast to "
        "be the strongest on record.",
    ]
    y = 0.74
    import textwrap
    for t in lines:
        for ln in textwrap.wrap(t, 140):
            ax3.text(0, y, ln, fontsize=8.8, color=INK, va="top")
            y -= 0.075
        y -= 0.02
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=118); plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    raise SystemExit(main())
