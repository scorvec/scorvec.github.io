#!/usr/bin/env python3
"""ERA5 reference for the North Pacific jet product (pacjet.py) — built once on the laptop from the LOCAL store.

Input: ~/era5_store/wb2_1p5_daily/{u250,v250}/ (WeatherBench-2 ERA5, daily means, 1.5 deg, 0-90N), 1991-2020, on the
1.5 deg jet grid of pacjet_core (10.5-79.5N, 100.5E-61.5W).

  1. Day-of-year climatology of the 250 hPa zonal wind (mean + three harmonics per grid point) - the anomaly base.
  2. SEASON-APPROPRIATE jet phases. The patterns are the canonical cold-season pair (the two leading EOFs of the
     area-weighted (sqrt cos lat) Nov-Mar daily anomalies over the Winters et al. (2019) sector 10-80N, 100E-120W:
     EOF1 = extension/retraction, +PC1 = westerly anomaly in the exit region 30-40N 170E-150W; EOF2 = poleward/
     equatorward shift, +PC2 = westerly anomaly north of the January jet axis). The SCALE is seasonal: in each month
     the PCs are standardised over the days of a sliding three-month window centred on it, so 1 sigma in October means
     an October-sized departure, not a January one. Rotating EOFs month by month was tried and rejected: in October
     the window's own EOF1/EOF2 are mixtures of extension and shift (pattern correlations 0.71/0.55 and 0.48/0.60), so
     "extension" would change meaning from month to month, while the canonical PLANE still holds most of October's
     variance. A month is IN SEASON when the canonical pair captures >= CAPTURE_OK of the variance the window's own two
     leading EOFs capture and the two planes are close (smaller principal-angle cosine >= COS_OK). Outside the season
     the live product shows a note, not sigma values.
  3. The ERA5 daily PC record (each day projected on its own month's basis) and its phase labels (pacjet_core.phase_of)
     - the input of the impact composites and of the torque regression.
  4. The climatological jet axis for every day of the year: the axis (pacjet_core.jet_axis) of the harmonic
     day-of-year mean of the daily wind SPEED.
  5. The monthly frequency of daily speed >= 50 m/s (context for the probability maps).
  6. The Himalayan mountain-torque anomaly 1991-2020 (data/reference/him_torque_1991_2020.nc, from WeatherBench-2
     surface pressure over 70-105E 25-45N: build_pacjet_torque.py --series) for the torque regression.

    python src/build_pacjet_ref.py        # ~2 min -> scripts/mjo/data/reference/pacjet_ref.nc
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pacjet_core as PC                                                    # noqa: E402

REF = HERE.parent / "data" / "reference"
OUT = REF / "pacjet_ref.nc"
STORE = Path.home() / "era5_store" / "wb2_1p5_daily"
TORQUE = REF / "him_torque_1991_2020.nc"    # build_pacjet_torque.py --series (WB2 surface pressure)
Y0, Y1 = 1991, 2020
EXIT = dict(lat=(30.0, 40.0), lon=(170.0, 210.0))
CANON = (11, 12, 1, 2, 3)
CAPTURE_OK, COS_OK = 0.80, 0.75
NH = 3


def load(var):
    parts = []
    for y in range(Y0, Y1 + 1):
        with xr.open_dataset(STORE / var / f"{var}_{y}.nc") as ds:
            a = ds[var].transpose("time", "latitude", "longitude")
            a = a.assign_coords(latitude=np.round(a.latitude.values.astype(float), 3),
                                longitude=np.round(a.longitude.values.astype(float), 3))
            parts.append(a.sel(latitude=PC.EOF_LAT, longitude=PC.EOF_LON).load())
    u = xr.concat(parts, "time")
    assert 5 < float(u.mean()) < 40 or var.startswith("v"), f"{var} mean {float(u.mean())} - not m/s?"
    return u


def harm_fit(doy, y):
    B = PC.harm_basis(doy, NH)
    coef, *_ = np.linalg.lstsq(B, y.reshape(len(y), -1), rcond=None)
    return coef.reshape(B.shape[1], *y.shape[1:])


def leading_modes(X, n=3):
    """X (time, space) centred -> (eigvecs (n, space), explained fraction (n,))."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:n], (S ** 2 / (S ** 2).sum())[:n]


def main() -> int:
    t0 = time.time()
    u = load("u250"); v = load("v250")
    t = pd.DatetimeIndex(u.time.values)
    doy = t.dayofyear.values; mon = t.month.values
    print(f"  u250/v250 {dict(u.sizes)} {t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d} ({time.time() - t0:.0f}s)", flush=True)
    U = u.values.astype("float32"); V = v.values.astype("float32")
    coef = harm_fit(doy, U)
    anom = U - PC.harm_eval(coef, doy)
    lat, lon = PC.EOF_LAT, PC.EOF_LON
    sec = PC.sector_mask()
    w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * sec[None, :].astype(float)       # zero outside the sector
    npts = int(sec.sum()) * lat.size
    exit_m = ((lat >= EXIT["lat"][0]) & (lat <= EXIT["lat"][1]))[:, None] & ((lon >= EXIT["lon"][0]) & (lon <= EXIT["lon"][1]))[None]

    def modes_for(sel):
        X = (anom[sel] * w[None]).reshape(sel.sum(), -1)[:, (w > 0).ravel()]
        X = X - X.mean(0)
        return leading_modes(X, 3)

    def full(vec):
        f = np.zeros(lat.size * lon.size, "float64"); f[(w > 0).ravel()] = vec
        return f.reshape(lat.size, lon.size)

    # canonical cold-season pair
    ccold = np.isin(mon, CANON)
    ev, ex = modes_for(ccold)
    can = [full(ev[0]), full(ev[1])]
    if (can[0] / np.where(w > 0, w, 1))[exit_m].mean() < 0:
        can[0] = -can[0]
    print(f"  canonical Nov-Mar EOF1/2 explain {ex[0]:.1%}/{ex[1]:.1%} (EOF3 {ex[2]:.1%})", flush=True)

    # speed climatology and the climatological axis by day of year
    spd = np.hypot(U, V)
    scoef = harm_fit(doy, spd)
    doys = np.arange(1, 367)
    spd_doy = PC.harm_eval(scoef, doys)                                        # (366, lat, lon)
    axis_clim = PC.jet_axis(spd_doy, lat=lat)
    axis_jan = np.nanmean(axis_clim[14][sec])                                  # mid-January axis over the sector
    north = (lat[:, None] > axis_jan + 4) & ((lon >= 140) & (lon <= 220))[None]
    if (can[1] / np.where(w > 0, w, 1))[north].mean() < 0:
        can[1] = -can[1]
    print(f"  canonical signs: +PC1 exit-region westerly, +PC2 westerly north of the Jan axis ({axis_jan:.1f}N)", flush=True)
    daily_axis = PC.jet_axis(spd, lat=lat)                                    # (time, lon)
    with np.errstate(all="ignore"):
        axis_p10 = np.stack([np.nanpercentile(daily_axis[mon == m], 10, axis=0) for m in range(1, 13)])
        axis_p90 = np.stack([np.nanpercentile(daily_axis[mon == m], 90, axis=0) for m in range(1, 13)])
        axis_def = np.stack([np.isfinite(daily_axis[mon == m]).mean(0) for m in range(1, 13)])
    axis_p10[axis_def < 0.5] = np.nan; axis_p90[axis_def < 0.5] = np.nan       # no range where the core is usually absent
    freq50 = np.stack([(spd[mon == m] >= PC.CORE).mean(0) for m in range(1, 13)]).astype("float32")

    # The PATTERNS are the canonical pair in every month (the phases mean the same thing all season); the SCALE is the
    # month's: each PC is standardised over its sliding three-month window. A month is in season when the canonical
    # pair still describes that window: it captures >= CAPTURE_OK of the variance the window's own two leading EOFs
    # capture, and the two planes are close (smaller principal-angle cosine >= COS_OK).
    proj = np.zeros((12, 2, lat.size, lon.size), "float32")
    reg = np.zeros((12, 2, lat.size, lon.size), "float32")
    sd = np.zeros((12, 2)); expl = np.zeros((12, 3)); capture = np.zeros(12); cosang = np.zeros((12, 2))
    cvec = [c[(w > 0)] for c in can]                                        # unit weighted eigenvectors (sector points)
    for m in range(1, 13):
        win = [((m - 2) % 12) + 1, m, (m % 12) + 1]
        sel = np.isin(mon, win)
        X = (anom[sel] * w[None]).reshape(sel.sum(), -1)[:, (w > 0).ravel()]
        X = X - X.mean(0)
        evm, exm = leading_modes(X, 3)
        tot = (X ** 2).sum()
        capture[m - 1] = float(((X @ np.stack(cvec).T) ** 2).sum() / tot / exm[:2].sum())
        cosang[m - 1] = np.linalg.svd(evm[:2] @ np.stack(cvec).T, compute_uv=False)
        expl[m - 1] = exm
        for k in range(2):
            pcs = np.einsum("tij,ij->t", anom[sel] * w[None], can[k])
            s = pcs.std()
            proj[m - 1, k] = (can[k] * w / s).astype("float32")
            reg[m - 1, k] = np.einsum("tij,t->ij", anom[sel], pcs / s) / sel.sum()
            sd[m - 1, k] = s
    ok = (capture >= CAPTURE_OK) & (cosang.min(1) >= COS_OK)
    for m in range(1, 13):
        print(f"  {pd.Timestamp(2001, m, 1):%b}: canonical pair captures {capture[m - 1]:.0%} of the window's own EOF1+2 "
              f"({expl[m - 1, :2].sum():.1%} of variance), plane cosines {cosang[m - 1, 0]:.2f}/{cosang[m - 1, 1]:.2f}, "
              f"PC sd {sd[m - 1, 0]:.0f}/{sd[m - 1, 1]:.0f} -> {'in season' if ok[m - 1] else 'OUT'}", flush=True)

    # ERA5 daily PCs on each day's own month basis, and the phases
    pcs = np.zeros((len(t), 2), "float32")
    for m in range(1, 13):
        s = mon == m
        pcs[s] = PC.project(anom[s], proj[m - 1])
    ph = PC.phase_of(pcs[:, 0], pcs[:, 1])
    ins = ok[mon - 1]
    for k, name in enumerate(PC.PHASES):
        print(f"  {name:12s} {np.mean(ph[ins] == k):.1%} of in-season days", flush=True)
    print(f"  neutral      {np.mean(ph[ins] == -1):.1%}", flush=True)

    data = {"u_coef": (("harm", "latitude", "longitude"), coef.astype("float32")),
            "spd_coef": (("harm", "latitude", "longitude"), scoef.astype("float32")),
            "proj": (("month", "mode", "latitude", "longitude"), proj),
            "eof_reg": (("month", "mode", "latitude", "longitude"), reg),
            "pc_sd": (("month", "mode"), sd.astype("float32")),
            "explained": (("month", "rank"), expl.astype("float32")),
            "capture": (("month",), capture.astype("float32")),
            "plane_cos": (("month", "mode"), cosang.astype("float32")),
            "in_season": (("month",), ok),
            "axis_clim": (("doy", "longitude"), axis_clim.astype("float32")),
            "freq50": (("month", "latitude", "longitude"), freq50),
            "axis_p10": (("month", "longitude"), axis_p10.astype("float32")),
            "axis_p90": (("month", "longitude"), axis_p90.astype("float32")),
            "axis_defined": (("month", "longitude"), axis_def.astype("float32")),
            "pc": (("time", "mode"), pcs), "phase": (("time",), ph.astype("int8"))}
    if TORQUE.exists():
        with xr.open_dataset(TORQUE) as old:
            data["him_torque_anom"] = (("time",), old.him_torque_anom.reindex(time=u.time.values).values.astype("float32"))
    ds = xr.Dataset(data, coords={"harm": np.arange(2 * NH + 1), "latitude": lat, "longitude": lon, "month": np.arange(1, 13),
                                  "mode": ["pc1", "pc2"], "rank": [1, 2, 3], "doy": doys, "time": u.time.values})
    ds.attrs.update(note=f"ERA5 (WB2 1.5 deg daily, local store) 250 hPa wind {Y0}-{Y1}. EOFs per month on a sliding 3-month "
                         f"window over 10-80N 100E-120W: the Nov-Mar EOF pair as patterns, standardised per window; in season when "
                         f"the pair captures >= {CAPTURE_OK:.0%} of the window's own EOF1+2 variance and the planes' cosines "
                         f">= {COS_OK}. PC = sum(anom * proj), unit variance in the window. "
                         "Phases (Winters et al. 2019): |PC| >= 1, nearest axis.",
                    canonical_explained=[float(ex[0]), float(ex[1])])
    ds.to_netcdf(OUT, encoding={k: {"zlib": True, "complevel": 4} for k in ds.data_vars})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
