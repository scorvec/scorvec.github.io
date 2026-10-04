#!/usr/bin/env python3
"""Reference files of the weather-regime products (laptop, once; rerun only if build_regimes.py is rerun).

  data/regimes_ref.npz   the centroids of ~/data_archive/geps_subx/telecon/regimes.nc (k-means, k = 4, NCEP/NCAR R1
                         1991-2020, cold and warm sets, EA and NA sectors) with their names; the family pairing checked
                         against the centroid correlations (regime_core.FAMILIES, >= 0.6 or the build stops); the
                         base-period statistics with every day classified by its OWN season's set: frequency by
                         calendar month, overall frequency, mean episode length.
  data/r1_labels.npz     the R1 regime of every day 1991-2020 in both sets (int8; lab_<sector>_<season> from the
                         centred 5-day mean, trail_<sector>_<season> from the 3 days ending that day, the persistence
                         forecast's input, own_<sector> in the day's own set) - the observed regime days of the impact
                         composites and of the hindcast verification.
  data/era5_z500_clim.npz  ERA5 1991-2020 day-of-year 500 hPa height climatology on the 2.5 deg grid, 90N-0, as
                         annual harmonics (the ~/era5_store daily 1.5 deg global file, box-averaged to 2.5 deg like
                         the forecasts, 10 harmonics) - the anomaly base of AIFS-ENS and IFS-ENS, which have no reforecast.

    python scripts/regimes/build_regimes_ref.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regime_core as RC                                                  # noqa: E402

GEPS = Path.home() / "data_archive" / "geps_subx"
sys.path.insert(0, str(GEPS))
ERA5 = Path.home() / "era5_store" / "wb2_1p5_daily_global" / "z500"
NHARM = 10


def centroid_corr(a, b, lat):
    w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None]
    A = (a * w).reshape(len(a), -1); B = (b * w).reshape(len(b), -1)
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return (A @ B.T) / np.outer(np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1))


def r1_labels(cent, lat_s, lon_s):
    """R1 daily z500 1991-2020 -> anomalies vs the 1991-2020 day-of-year climatology (build_regimes' own), 5-day
    running mean, classified in both sets."""
    from build_telecon_patterns import daily, doy_clim
    z = daily("hgt", range(1991, 2021), 500)       # R1 daily files on disk: 1991-2020 (+2025-26)
    base = z.sel(time=slice("1991-01-01", "2020-12-31"))
    clim = doy_clim(base)
    t = pd.DatetimeIndex(z.time.values)
    an = z.values - clim.values[t.dayofyear.values - 1]
    assert np.allclose(z.lat.values, RC.LAT25) and np.allclose(z.lon.values, RC.LON25)
    labs = {}
    for sk, spec in RC.SECTORS.items():
        sec, la, lo = RC.sector_cut(an, spec)
        assert np.allclose(la, lat_s[sk]) and np.allclose(lo, lon_s[sk])
        # build_regimes used a CENTRED 5-day mean with NaN at the two ends of the record; running_partial is the
        # same away from the ends
        v = RC.running_partial(sec, RC.SMOOTH, axis=0)
        # the regime a forecaster sees on day D: the 3-day mean ending on D (no look-ahead) - the persistence
        # forecast of a start on D + 1
        h = RC.SMOOTH // 2
        cs = np.cumsum(np.concatenate([np.zeros((1,) + sec.shape[1:]), sec], 0), 0)
        tr = np.stack([(cs[i + 1] - cs[max(0, i - h)]) / (i + 1 - max(0, i - h)) for i in range(len(sec))])
        ref_like = _Tmp(cent, lat_s, lon_s)
        for s in RC.SEASONS:
            labs[f"{sk}_{s}"] = ref_like.classify(v, sk, s)[0]
            labs[f"trail_{sk}_{s}"] = ref_like.classify(tr, sk, s)[0]
    return t, labs


class _Tmp(RC.Ref):
    def __init__(self, cent, lat, lon):                                   # noqa: D401  (no file yet)
        self.cent, self.lat, self.lon, self.names, self.meta = cent, lat, lon, {}, {}


def era5_clim():
    """ERA5 z500 1991-2020 day-of-year mean (31-day circular smoothing), 2.5 deg box mean, 90N-0 -> harmonics."""
    acc = np.zeros((366, 121, 240)); n = np.zeros(366)
    for y in range(1991, 2021):
        d = xr.open_dataset(ERA5 / f"z500_{y}.nc")
        a = d["z500"].transpose("time", "latitude", "longitude")
        lat = a.latitude.values
        v = a.values.astype("float64")
        if lat[0] < lat[-1]:
            v = v[:, ::-1]
        assert 4500 < np.nanmean(v) < 6000, "z500 must be metres (store attributes lie - README)"
        dy = pd.DatetimeIndex(a.time.values).dayofyear.values - 1
        np.add.at(acc, dy, v); np.add.at(n, dy, 1)
        d.close()
    c = acc / n[:, None, None]
    c[365] = 0.5 * (c[364] + c[0]) if n[365] < 5 else c[365]
    from scipy.ndimage import uniform_filter1d
    sm = uniform_filter1d(c, 31, axis=0, mode="wrap")
    g = RC.box25(sm, 1.5)[:, :37]                                          # (366, 37, 144): 90N .. 0
    t = 2 * np.pi * np.arange(366) / 366.0
    X = np.column_stack([np.ones(366)] + [f(h * t) for h in range(1, NHARM + 1) for f in (np.cos, np.sin)])
    coef, *_ = np.linalg.lstsq(X, g.reshape(366, -1), rcond=None)
    fit = (X @ coef).reshape(g.shape)
    err = np.abs(fit - g)[:, 4:25]                                         # 80N-20N
    print(f"  ERA5 z500 clim: {NHARM} harmonics, |fit - daily clim| 20-80N median {np.median(err):.2f} m, "
          f"max {err.max():.2f} m", flush=True)
    return coef.reshape(-1, 37, 144).astype("float32")


def main() -> int:
    ds = xr.open_dataset(GEPS / "telecon" / "regimes.nc")
    names = json.loads((GEPS / "telecon" / "regime_names.json").read_text())
    names = {k: v for k, v in names.items() if not k.startswith("_")}
    cent, lat_s, lon_s = {}, {}, {}
    for sk in RC.SECTORS:
        lat_s[sk] = ds[f"lat_{sk}"].values.astype("float64")
        lon_s[sk] = ds[f"lon_{sk}"].values.astype("float64")
        for s in RC.SEASONS:
            cent[f"{sk}_{s}"] = ds[f"{sk}_{s}"].values.astype("float64")
    # the family pairing must match the centroids
    fam_r = {}
    for sk in RC.SECTORS:
        r = centroid_corr(cent[f"{sk}_cold"], cent[f"{sk}_warm"], lat_s[sk])
        fam_r[sk] = np.round(r, 3).tolist()
        for f in RC.FAMILIES[sk]:
            if f["cold"] is not None and f["warm"] is not None:
                rr = r[f["cold"], f["warm"]]
                nc, nw = names[sk + "_cold_" + str(f["cold"])], names[sk + "_warm_" + str(f["warm"])]
                print(f"  {sk} family {f['key']:6s} cold {nc!r:30s} ~ warm {nw!r:34s} r = {rr:+.2f}")
                if rr < RC.FAMILY_MIN_R:
                    raise SystemExit(f"{sk} {f['key']}: centroid correlation {rr:.2f} < {RC.FAMILY_MIN_R}")
            else:
                s, k = ("cold", f["cold"]) if f["cold"] is not None else ("warm", f["warm"])
                o = "warm" if s == "cold" else "cold"
                best = r[k].max() if s == "cold" else r[:, k].max()
                nm = names[sk + "_" + s + "_" + str(k)]
                print(f"  {sk} family {f['key']:6s} {s} only {nm!r}: best {o} match r = {best:+.2f}")
                if best >= RC.FAMILY_MIN_R:
                    raise SystemExit(f"{sk} {f['key']} has a counterpart (r {best:.2f}): pair it")
    t, labs = r1_labels(cent, lat_s, lon_s)
    own = {}
    base = (t.year >= 1991) & (t.year <= 2020)
    stats = {}
    for sk in RC.SECTORS:
        cold = np.array([RC.season_of(m) == "cold" for m in t.month])
        lab = np.where(cold, labs[f"{sk}_cold"], labs[f"{sk}_warm"])
        own[sk] = lab
        for s in RC.SEASONS:
            m = base & ((cold if s == "cold" else ~cold))
            f = np.bincount(lab[m], minlength=RC.K + 1) / m.sum()
            months = RC.SEASONS[s]
            bym = {int(mo): (np.bincount(lab[base & (t.month == mo)], minlength=RC.K + 1)
                             / (base & (t.month == mo)).sum()).round(4).tolist() for mo in months}
            # mean episode length (consecutive days in one regime, inside the season)
            x = np.where(m, lab, -9)
            cut = np.flatnonzero(np.diff(x) != 0) + 1
            ep = [len(g) for g in np.split(x, cut) if g[0] >= 0 and g[0] != RC.NONE]
            stats[f"{sk}_{s}"] = dict(freq=f.round(4).tolist(), by_month=bym,
                                      mean_episode_days=round(float(np.mean(ep)), 1), n_days=int(m.sum()))
            print(f"  {sk} {s}: freq {np.round(f * 100).astype(int).tolist()} (last = none), "
                  f"episodes {np.mean(ep):.1f} d")
    meta = dict(names=names, stats=stats, family_corr=fam_r, families=RC.FAMILIES,
                source="~/data_archive/geps_subx/telecon/regimes.nc (build_regimes.py): k-means k=4, 14 EOFs, 5-day "
                       "running-mean NCEP/NCAR R1 500 hPa anomalies 1991-2020, cold Oct-Mar / warm Apr-Sep",
                r_min=RC.R_MIN, smooth_days=RC.SMOOTH,
                note="stats: every day classified with its own season's set, 1991-2020")
    RC.DATA.mkdir(exist_ok=True)
    np.savez_compressed(RC.REF_FILE, meta_json=np.array(json.dumps(meta)),
                        **{f"cent_{k}": v.astype("float32") for k, v in cent.items()},
                        **{f"lat_{sk}": lat_s[sk] for sk in RC.SECTORS}, **{f"lon_{sk}": lon_s[sk] for sk in RC.SECTORS})
    np.savez_compressed(RC.DATA / "r1_labels.npz", t=np.array([int(x.strftime("%Y%m%d")) for x in t], "int32"),
                        **{(k if k.startswith("trail_") else f"lab_{k}"): v.astype("int8") for k, v in labs.items()},
                        **{f"own_{sk}": v.astype("int8") for sk, v in own.items()})
    coef = era5_clim()
    np.savez_compressed(RC.DATA / "era5_z500_clim.npz", coef=coef, nharm=NHARM, lat=RC.LAT25[:37], lon=RC.LON25,
                        note="ERA5 1991-2020 daily z500 (m), day-of-year mean, 31-d circular smoothing, 2.5 deg box "
                             "mean; value(doy) = coef[0] + sum_h coef[2h-1] cos(h t) + coef[2h] sin(h t), "
                             "t = 2 pi (doy - 1) / 366")
    for f in (RC.REF_FILE, RC.DATA / "r1_labels.npz", RC.DATA / "era5_z500_clim.npz"):
        print(f"  {f.relative_to(HERE.parents[1])}  {f.stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
