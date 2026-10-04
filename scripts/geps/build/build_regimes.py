#!/usr/bin/env python3
"""Weather regimes for the Euro-Atlantic and North American sectors — built
once on NCEP/NCAR R1 500 hPa height, 1991-2020.

Method (Michelangeli et al. 1995; Cassou 2008; Grams et al. 2017 for the
Euro-Atlantic sector; Lee et al. 2019 / Robertson & Ghil 1999 for North
America):
  anomalies   daily 500 hPa height minus the 1991-2020 day-of-year
              climatology, 5-day running mean (synoptic noise out, blocking
              and regime transitions kept), cos-lat weighted;
  seasons     two regime sets per sector, cold (Oct-Mar) and warm (Apr-Sep):
              summer regimes are not winter regimes with less amplitude,
              they have different shapes;
  EOFs        the leading 14 EOFs of the sector (typically ~85-90% of the
              variance), k-means with k = 4 in that space, 30 restarts, the
              partition with the lowest inertia kept;
  centroids   stored as physical anomaly maps (m) on the 2.5 deg sector grid,
              so a forecast member-day is assigned by nearest centroid in
              the same weighted Euclidean sense;
  no regime   a day whose pattern correlation with its nearest centroid is
              below R_MIN is "no regime", as in the ECMWF product; the share
              of such days in the climatology is reported.
Names are given by hand after looking at figs/geps_regimes_{sector}.webp —
k-means numbers clusters arbitrarily — and kept in telecon/regime_names.json;
the centroids never change unless this is rerun, so the names hold.

    python build_regimes.py
    -> telecon/regimes.nc, telecon/regime_names.json (defaults), figs/geps_regimes_{ea,na}.webp
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.vq import kmeans2

HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path[:0] = [str(HERE), str(HERE.parent)]
from paths import LOCAL_STORE as _LS                             # noqa: E402  laptop-only inputs/outputs
from build_telecon_patterns import daily, doy_clim, NCEP, TC   # noqa: E402

FIGS = _LS / "figs"
Y0, Y1 = 1991, 2020
K = 4
NEOF = 14
R_MIN = 0.25
SMOOTH = 5
SECTORS = {"ea": dict(label="Euro-Atlantic", lat=(80, 20), lon=(-90, 30)),
           "na": dict(label="North America / Pacific", lat=(80, 20), lon=(-170, -30))}
SEASONS = {"cold": (10, 11, 12, 1, 2, 3), "warm": (4, 5, 6, 7, 8, 9)}


def sector_field(z, spec):
    """Sector cut of a (time, lat, lon) 2.5 deg field, lon in -180..180 order."""
    lon = ((z.lon.values + 180) % 360) - 180
    z = z.assign_coords(lon=lon).sortby("lon")
    return z.sel(lat=slice(*spec["lat"]), lon=slice(*spec["lon"]))


def running(a, n):
    k = np.ones(n) / n
    out = np.full_like(a, np.nan)
    v = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 0, a)
    out[n // 2: n // 2 + v.shape[0]] = v
    return out


def cluster(A, w):
    """A: (time, lat, lon) anomalies; w: (lat, lon) sqrt(cos) weights.
    Returns centroids (K, lat, lon) in anomaly units, labels, inertia."""
    X = (A * w).reshape(A.shape[0], -1)
    Xc = X - X.mean(0)
    # EOFs via the time-time Gram matrix (see svd_safe in build_telecon_patterns)
    C = Xc @ Xc.T
    ev, U = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1][:NEOF]
    ev, U = ev[o], U[:, o]
    pcs = U * np.sqrt(ev)                                  # (time, NEOF), variance-scaled
    eofs = (Xc.T @ U) / np.sqrt(ev)                        # (space, NEOF), unit norm
    explained = ev.sum() / np.trace(C)
    best = None
    rng = np.random.default_rng(7)
    for _ in range(30):
        cen, lab = kmeans2(pcs, K, minit="++", seed=rng.integers(1 << 30), iter=50)
        inertia = ((pcs - cen[lab]) ** 2).sum()
        if best is None or inertia < best[2]:
            best = (cen, lab, inertia)
    cen, lab, inertia = best
    cent = (eofs @ cen.T).T.reshape(K, *A.shape[1:]) / w    # back to physical anomaly (m)
    return cent, lab, explained


def assign(A, cent, w):
    """Nearest centroid (weighted Euclidean) and pattern correlation to it."""
    X = (A * w).reshape(A.shape[0], -1)
    Cw = (cent * w).reshape(K, -1)
    d2 = ((X[:, None, :] - Cw[None]) ** 2).sum(-1)
    lab = d2.argmin(1)
    xm = X - X.mean(1, keepdims=True); cm = Cw - Cw.mean(1, keepdims=True)
    corr = (xm @ cm.T) / (np.linalg.norm(xm, axis=1)[:, None] * np.linalg.norm(cm, axis=1)[None] + 1e-9)
    return lab, corr[np.arange(len(lab)), lab]


def main() -> int:
    z = daily("hgt", range(Y0, Y1 + 1), 500)
    clim = doy_clim(z)
    an = z.copy(data=(z.values - clim.values[z.time.dt.dayofyear.values - 1]).astype("float32"))
    t = pd.DatetimeIndex(z.time.values)
    out_c, meta = {}, {}
    names = {}
    for sk, spec in SECTORS.items():
        sec = sector_field(an, spec)
        A = running(sec.values, SMOOTH)
        lat, lon = sec.lat.values, sec.lon.values
        w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
        for season, months in SEASONS.items():
            sel = np.isin(t.month, months) & np.isfinite(A).all(axis=(1, 2))
            cent, lab, expl = cluster(A[sel], w)
            # order clusters by frequency, most common first, for stable numbering
            freq = np.bincount(lab, minlength=K) / len(lab)
            order = np.argsort(freq)[::-1]
            cent = cent[order]
            lab2, corr = assign(A[sel], cent, w)
            none = corr < R_MIN
            freq = np.bincount(lab2[~none], minlength=K) / (~none).sum()
            # persistence: mean run length of the same regime in days
            runs = np.diff(np.flatnonzero(np.r_[True, np.diff(lab2) != 0, True]))
            out_c[(sk, season)] = cent
            meta[f"{sk}_{season}"] = dict(explained=round(float(expl), 3), freq=[round(float(f), 3) for f in freq],
                                          no_regime=round(float(none.mean()), 3),
                                          mean_persistence_days=round(float(runs.mean()), 1),
                                          n_days=int(sel.sum()))
            for k in range(K):
                names[f"{sk}_{season}_{k}"] = f"R{k+1}"
            print(f"  {sk} {season}: EOFs explain {expl*100:.0f}%, freq {np.round(freq, 2)}, "
                  f"no-regime {none.mean()*100:.0f}%, persistence {runs.mean():.1f} d")
        # figure: 2 seasons x K centroids
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.LambertConformal(central_longitude=np.mean(spec["lon"]), standard_parallels=(35, 55))
        fig = plt.figure(figsize=(4.0 * K, 3.6 * len(SEASONS) + 0.8))
        for si, season in enumerate(SEASONS):
            cent = out_c[(sk, season)]
            lim = float(np.nanpercentile(np.abs(cent), 99))
            for k in range(K):
                ax = fig.add_subplot(len(SEASONS), K, si * K + k + 1, projection=proj)
                ax.set_extent([spec["lon"][0], spec["lon"][1], spec["lat"][1], spec["lat"][0]], ccrs.PlateCarree())
                m = ax.contourf(lon, lat, cent[k], levels=np.linspace(-lim, lim, 21), cmap="RdBu_r", extend="both",
                                transform=ccrs.PlateCarree())
                ax.contour(lon, lat, cent[k], levels=np.linspace(-lim, lim, 11), colors="k", linewidths=0.35,
                           transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.6, edgecolor="#333")
                mt = meta[f"{sk}_{season}"]
                ax.set_title(f"{season} · regime {k+1} · {mt['freq'][k]*100:.0f}% of days", fontsize=9.5,
                             fontweight="bold")
        cax = fig.add_axes([0.3, 0.035, 0.4, 0.018])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal"); cb.set_label("500 hPa height anomaly (m)", fontsize=9)
        fig.suptitle(f"{spec['label']} weather regimes — k-means (k={K}) on 5-day-mean 500 hPa anomalies, "
                     f"NCEP/NCAR R1 {Y0}–{Y1}", fontsize=11.5, fontweight="bold", y=0.99)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.09, wspace=0.05, hspace=0.18)
        FIGS.mkdir(exist_ok=True)
        fig.savefig(FIGS / f"geps_regimes_{sk}.webp", dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
        plt.close(fig)
        print(f"  geps_regimes_{sk}.webp")
    # save centroids on each sector's own grid
    ds = {}
    for (sk, season), cent in out_c.items():
        spec = SECTORS[sk]
        sec = sector_field(an.isel(time=slice(0, 1)), spec)
        ds[f"{sk}_{season}"] = xr.DataArray(cent, dims=("k", f"lat_{sk}", f"lon_{sk}"),
                                            coords={"k": np.arange(K), f"lat_{sk}": sec.lat.values, f"lon_{sk}": sec.lon.values})
    xr.Dataset(ds, attrs=dict(method="see build_regimes.py", r_min=R_MIN, smooth_days=SMOOTH, k=K, neof=NEOF,
                              base=f"{Y0}-{Y1}", meta=json.dumps(meta))).to_netcdf(TC / "regimes.nc")
    q = TC / "regime_names.json"
    if not q.exists():
        q.write_text(json.dumps(names, indent=1))
    print("  regimes.nc written; name the clusters in regime_names.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
