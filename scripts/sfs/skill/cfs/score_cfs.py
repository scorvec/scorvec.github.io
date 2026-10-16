"""Part A: SFS beta2 vs CFSv2 weekly hindcast skill, Sep 1 / Oct 1 starts, against ERA5 (NH 1.5 deg).

SFS: 4-member means, scores averaged over all 15 4-of-6 subsets (precip: members 0-3). CFSv2: the four cycles
00/06/12/18Z of the start day, member 1 each (IRI SubX hindcast 1999-2016 / real-time 2018-2025; z500 2019-2025 from
the operational files on S3). Anomalies: each system minus its own mean over the OTHER years (leave-one-year-out, same
year set for both systems); truth = ERA5 minus its 1991-2020 daily normal over the same valid days. Per case over a
region (cos-lat): uncentred pattern ACC (Fisher-z averaged over cases), MSSS (sum MSE / sum clim MSE), fair CRPS vs a
Gaussian climatology N(0, sd_LOO). 95% CIs: year-block bootstrap (both starts of a year together), 5000 resamples."""

import itertools
import sys
import os
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2 import YEARS, OUT, WNAMES, ELAT, ELON

S = np.load(OUT + "sfs_weekly2.npz")
C = np.load(OUT + "cfs_weekly.npz")
E = np.load(OUT + "era5_truth2.npz")
assert list(S["years"]) == list(C["years"]) == list(E["years"]) == YEARS
W = np.cos(np.deg2rad(ELAT))[:, None] * np.ones((1, len(ELON)))


def conus_mask():
    import cartopy.io.shapereader as shp
    from shapely.geometry import Point
    from shapely.prepared import prep

    geoms = [
        r.geometry
        for r in shp.Reader(shp.natural_earth("110m", "cultural", "admin_0_countries")).records()
        if r.attributes.get("ADM0_A3") == "USA"
    ]
    g = prep(geoms[0])
    m = np.zeros((len(ELAT), len(ELON)), bool)
    for i, la in enumerate(ELAT):
        if not 24 <= la <= 50:
            continue
        for j, lo in enumerate(ELON):
            x = lo - 360 if lo > 180 else lo
            if -125 <= x <= -66 and g.contains(Point(x, la)):
                m[i, j] = True
    return m


REG = {"NH 20-90N": (ELAT >= 20)[:, None] * np.ones((1, len(ELON)), bool), "CONUS land": conus_mask()}


def loo(F):
    """F (2, ny, mem, 4, lat, lon) -> members minus the other years' ensemble-mean (same member count)."""
    Fm = F.mean(2)
    ny = F.shape[1]
    L = (Fm.sum(1, keepdims=True) - Fm) / (ny - 1)
    return F - L[:, :, None]


def case_stats(f, o, mask):
    w = W * mask * np.isfinite(f) * np.isfinite(o)
    f = np.nan_to_num(f)
    o = np.nan_to_num(o)
    sw = w.sum()
    return (
        (w * f * o).sum() / np.sqrt((w * f * f).sum() * (w * o * o).sum()),
        (w * (f - o) ** 2).sum() / sw,
        (w * o * o).sum() / sw,
    )


def fair_crps(ens, o):
    m = ens.shape[0]
    return np.mean(np.abs(ens - o[None]), 0) - np.abs(ens[:, None] - ens[None, :]).sum((0, 1)) / (2 * m * (m - 1))


def gauss_crps(sd, o):
    from scipy.stats import norm

    z = o / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def fz(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


rng = np.random.default_rng(7)
rows = []
for var in ("t2m", "z500", "pr"):
    ok = np.array(
        [
            np.isfinite(C[var][:, i, :, :, 30, 10]).all()
            and np.isfinite(S[var][:, i, :, :, 30, 10]).all()
            and np.isfinite(E[var][:, i, :, 30, 10]).all()
            for i in range(len(YEARS))
        ]
    )
    for era, yr_sel in (("all", ok), ("1999-2016", ok & (np.array(YEARS) <= 2016))):
        yrs = [y for y, k in zip(YEARS, yr_sel) if k]
        idx = np.where(yr_sel)[0]
        Fs = S[var][:, idx].astype(np.float64)
        Fc = C[var][:, idx].astype(np.float64)
        O = E[var][:, idx].astype(np.float64)
        As, Ac = loo(Fs), loo(Fc)
        subs = list(itertools.combinations(range(6), 4)) if Fs.shape[2] >= 6 else [tuple(range(4))]
        ny = len(yrs)
        for wi, wn in enumerate(WNAMES):
            for rn, mask in REG.items():
                st = {
                    k: np.zeros((2, ny))
                    for k in ("sfs_z", "sfs_mse", "clim", "cfs_z", "cfs_mse", "sfs_crps", "cfs_crps", "crpsc")
                }
                ww = W * mask
                for mi in range(2):
                    for yi in range(ny):
                        o = O[mi, yi, wi]
                        zs, es, cs = [], [], []
                        for sub in subs:
                            a, e, c = case_stats(As[mi, yi, list(sub), wi].mean(0), o, mask)
                            zs.append(fz(a))
                            es.append(e)
                            cs.append(np.nansum(ww * fair_crps(As[mi, yi, list(sub), wi], o)) / ww.sum())
                        st["sfs_z"][mi, yi] = np.mean(zs)
                        st["sfs_mse"][mi, yi] = np.mean(es)
                        st["clim"][mi, yi] = c
                        st["sfs_crps"][mi, yi] = np.mean(cs)
                        a, e, _ = case_stats(Ac[mi, yi, :, wi].mean(0), o, mask)
                        st["cfs_z"][mi, yi] = fz(a)
                        st["cfs_mse"][mi, yi] = e
                        st["cfs_crps"][mi, yi] = np.nansum(ww * fair_crps(Ac[mi, yi, :, wi], o)) / ww.sum()
                        sd = np.sqrt(np.nanmean(np.delete(O[:, :, wi], yi, axis=1) ** 2, axis=(0, 1)))
                        st["crpsc"][mi, yi] = np.nansum(ww * gauss_crps(sd, o)) / ww.sum()

                def metric(kind, s, ix, st=st):
                    if kind == "acc":
                        return float(np.tanh(st[f"{s}_z"][:, ix].mean()))
                    if kind == "msss":
                        return float(1 - st[f"{s}_mse"][:, ix].sum() / st["clim"][:, ix].sum())
                    return float(1 - st[f"{s}_crps"][:, ix].sum() / st["crpsc"][:, ix].sum())

                boot = rng.integers(0, ny, (5000, ny))
                allix = np.arange(ny)
                for kind in ("acc", "msss", "crpss"):
                    for s in ("sfs", "cfs"):
                        b = [metric(kind, s, bi) for bi in boot]
                        rows.append(
                            dict(
                                var=var,
                                window=wn,
                                region=rn,
                                metric=kind,
                                system=s,
                                years=era,
                                n_years=ny,
                                value=round(metric(kind, s, allix), 3),
                                lo=round(float(np.percentile(b, 2.5)), 3),
                                hi=round(float(np.percentile(b, 97.5)), 3),
                                sig="",
                            )
                        )
                    d = metric(kind, "sfs", allix) - metric(kind, "cfs", allix)
                    b = np.array([metric(kind, "sfs", bi) - metric(kind, "cfs", bi) for bi in boot])
                    lo, hi = np.percentile(b, [2.5, 97.5])
                    rows.append(
                        dict(
                            var=var,
                            window=wn,
                            region=rn,
                            metric=kind,
                            system="sfs-cfs",
                            years=era,
                            n_years=ny,
                            value=round(d, 3),
                            lo=round(float(lo), 3),
                            hi=round(float(hi), 3),
                            sig="yes" if (lo > 0 or hi < 0) else "no",
                        )
                    )
        print(var, era, ny, "years:", yrs[0], "-", yrs[-1], flush=True)

with open(OUT + "partA_sfs_vs_cfs_weekly.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("wrote", len(rows), "rows")
