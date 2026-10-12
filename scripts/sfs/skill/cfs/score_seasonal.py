"""Part B: SFS beta2 vs CFSv2 seasonal hindcasts, Sep 1 / Oct 1 starts 1991-2025.

Equal ensembles: SFS all 11 members; CFSv2 11-member subsets drawn at random from its 24 (20 draws, scores averaged per
draw); CFSv2 with all 24 members also listed (cfs24). Every system against its own leave-one-year-out climatology
(same years for both), truth likewise against its own LOO mean.
  Nino-3.4: per start month and lead 0-8 (lead 0 = start month), temporal ACC (Pearson over years) and RMSE (C) of the
  ensemble-mean anomaly vs OISST v2.1 (the repo's oisst_box_monthly.json). cfs_split: CFSv2 climatology split at 1999
  (CPC's practice for the CFSR ocean-analysis break), LOO within each segment.
  3-month t2m (global, NH 20-90N, CONUS land) and precipitation (NH 20-90N, CONUS land) vs ERA5: uncentred pattern ACC
  per case, Fisher-z averaged; MSSS = 1 - sum MSE / sum clim MSE.
95% CIs: year bootstrap (both starts of a year together), 5000 resamples; sig = the SFS-minus-CFS interval excludes 0."""
import sys, os, csv, json, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2 import OUT

YRS = np.arange(1991, 2026)
GLAT = np.arange(-90.0, 90.0 + 1e-6, 1.5); GLON = np.arange(0.0, 360.0 - 1e-6, 1.5)
rng = np.random.default_rng(11)
NB, ND = 5000, 20
BOOT = rng.integers(0, len(YRS), (NB, len(YRS)))
rows = []


def ci(vals):
    return round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)


def loo_anom(x, axis_years=1):
    """x (..., years, ...) with NaN years allowed -> x minus the mean over the other finite years."""
    fin = np.isfinite(x)
    s = np.nansum(x, axis=axis_years, keepdims=True); n = fin.sum(axis=axis_years, keepdims=True)
    return x - (s - np.nan_to_num(x)) / (n - 1)


# ---------------------------------------------------------------- Nino-3.4
o = json.load(open(Path(__file__).resolve().parents[3] / "sst" / "reference" / "oisst_box_monthly.json"))
om = dict(zip(o["months"], o["n34"]))
NL = 9                                                                  # leads 0-8: real-time CFSv2 has only 8 members at lead 9
OBS = np.full((2, len(YRS), NL), np.nan)
for mi, m in enumerate((9, 10)):
    for yi, y in enumerate(YRS):
        for l in range(NL):
            mm = m + l; yy = y + (mm - 1) // 12; mm = (mm - 1) % 12 + 1
            OBS[mi, yi, l] = om.get(f"{yy}-{mm:02d}", np.nan)
SN = np.load(OUT + "sfs_n34.npy")[..., :NL]                             # (2, 35, 11, 10)
CN = np.load(OUT + "cfs_n34.npy")[..., :NL]                                     # (2, 35, 28, 10)
assert np.isfinite(OBS).all() and np.isfinite(SN).all(), "gaps in Nino-3.4"
cmem = [np.where(np.isfinite(CN[mi, yi]).all(-1))[0] for mi in range(2) for yi in range(len(YRS))]   # members with all 10 leads
print("CFS members with all leads, min/max:", min(len(c) for c in cmem), max(len(c) for c in cmem), flush=True)
assert min(len(c) for c in cmem) >= 11
# member draws: 11 of each start's complete members
draws = [[rng.choice(cmem[mi * len(YRS) + yi], 11, replace=False) for mi in range(2) for yi in range(len(YRS))] for _ in range(ND)]
Oa = loo_anom(OBS)
Sa = loo_anom(SN.mean(2))                                               # (2, 35, 10)
C24 = loo_anom(np.nanmean(CN, 2))
seg = YRS >= 1999


def split_anom(x):
    out = np.empty_like(x)
    for s in (~seg, seg):
        out[:, s] = loo_anom(x[:, s])
    return out


C24s = split_anom(np.nanmean(CN, 2))
Cd = np.stack([loo_anom(np.stack([[CN[mi, yi, d[mi * len(YRS) + yi]].mean(0) for yi in range(len(YRS))] for mi in range(2)]))
               for d in draws])                                         # (ND, 2, 35, 10)
Cds = np.stack([split_anom(np.stack([[CN[mi, yi, d[mi * len(YRS) + yi]].mean(0) for yi in range(len(YRS))] for mi in range(2)]))
                for d in draws])


def acc_rmse(f, ob, ix):
    """f, ob (2, ny, 10) -> acc (2, 10), rmse (2, 10) over the resampled years ix."""
    f = f[:, ix]; ob = ob[:, ix]
    fc = f - f.mean(1, keepdims=True); oc = ob - ob.mean(1, keepdims=True)
    return (fc * oc).sum(1) / np.sqrt((fc ** 2).sum(1) * (oc ** 2).sum(1)), np.sqrt(((f - ob) ** 2).mean(1))


def nino_metric(name, ix):
    if name == "sfs": return acc_rmse(Sa, Oa, ix)
    if name == "cfs24": return acc_rmse(C24, Oa, ix)
    if name == "cfs24_split": return acc_rmse(C24s, Oa, ix)
    D = Cd if name == "cfs" else Cds
    r = [acc_rmse(D[k], Oa, ix) for k in range(ND)]
    return np.tanh(np.mean([np.arctanh(np.clip(a, -.999, .999)) for a, _ in r], 0)), np.mean([e for _, e in r], 0)


allix = np.arange(len(YRS))
names = ["sfs", "cfs", "cfs_split", "cfs24", "cfs24_split"]
val = {n: nino_metric(n, allix) for n in names}
bt = {n: [nino_metric(n, b) for b in BOOT] for n in names}
for mi, m in enumerate(("Sep", "Oct")):
    for l in range(NL):
        for k, met in enumerate(("acc", "rmse")):
            for n in names:
                lo, hi = ci([b[k][mi, l] for b in bt[n]])
                rows.append(dict(part="nino34", var="nino34", start=m, period=f"lead{l}", region="5S-5N 170-120W", metric=met,
                                 system=n, n_years=len(YRS), anom="loo", value=round(float(val[n][k][mi, l]), 3), lo=lo, hi=hi, sig=""))
            for other in ("cfs", "cfs_split"):
                d = val["sfs"][k][mi, l] - val[other][k][mi, l]
                lo, hi = ci([bs[k][mi, l] - bc[k][mi, l] for bs, bc in zip(bt["sfs"], bt[other])])
                rows.append(dict(part="nino34", var="nino34", start=m, period=f"lead{l}", region="5S-5N 170-120W", metric=met,
                                 system=f"sfs-{other}", n_years=len(YRS), anom="loo", value=round(float(d), 3), lo=lo, hi=hi,
                                 sig="yes" if (lo > 0 or hi < 0) else "no"))
print("nino34 done", flush=True)

# ---------------------------------------------------------------- 3-month t2m / precipitation
E = np.load(OUT + "era5_seasonal.npz")


def conus_mask(lat):
    import cartopy.io.shapereader as shp
    from shapely.geometry import Point
    from shapely.prepared import prep
    g = prep([r.geometry for r in shp.Reader(shp.natural_earth("110m", "cultural", "admin_0_countries")).records()
              if r.attributes.get("ADM0_A3") == "USA"][0])
    msk = np.zeros((len(lat), len(GLON)), bool)
    for i, la in enumerate(lat):
        if not 24 <= la <= 50: continue
        for j, lo in enumerate(GLON):
            x = lo - 360 if lo > 180 else lo
            if -125 <= x <= -66 and g.contains(Point(x, la)): msk[i, j] = True
    return msk


SEASN = {(0, 0): ("Sep", "OND"), (0, 1): ("Sep", "NDJ"), (1, 0): ("Oct", "NDJ"), (1, 1): ("Oct", "DJF")}
for var, cvar in (("t2m", "tref"), ("pr", "prec")):
    lat = GLAT if var == "t2m" else GLAT[GLAT >= 0]
    W = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(GLON)))
    regs = {"NH 20-90N": (lat >= 20)[:, None] * np.ones((1, len(GLON)), bool), "CONUS land": conus_mask(lat)}
    if var == "t2m": regs = {"Global": np.ones((len(lat), len(GLON)), bool), **regs}
    SF = np.load(OUT + f"sfs_seas_{var}.npy")                            # (2, 35, 11, 2, 121, 240)
    if var == "pr": SF = SF[..., GLAT >= 0, :]
    O = E[var]                                                          # (2, 35, 2, lat, lon)
    for (mi, k), (sm, sn) in SEASN.items():
        CF = np.full((len(YRS), 28, len(lat), len(GLON)), np.nan, np.float32)
        for yi, y in enumerate(YRS):
            CF[yi] = np.load(f"{OUT}raw_nmme/{cvar}_{y}{(9, 10)[mi]:02d}_{k}.npy")
        ok = np.isfinite(O[mi, :, k, len(lat) // 2 + 5, 10]) & np.isfinite(SF[mi, :, 0, k, 10, 10]) & np.isfinite(CF[:, :, 10, 10]).any(1)
        yix = np.where(ok)[0]; ny = len(yix); tyr = YRS[yix].astype(float)
        Obr = O[mi, yix, k].astype(np.float64)
        Smr = SF[mi, yix, :, k].mean(1).astype(np.float64)
        Cs = CF[yix].astype(np.float64)
        C24r = np.nanmean(Cs, 1)
        Cdr = [np.stack([Cs[i, rng.choice(np.where(np.isfinite(Cs[i, :, 10, 10]))[0], 11, replace=False)].mean(0) for i in range(ny)])
               for _ in range(ND)]

        def anom(x, mode):
            if mode == "loo":
                return x - (x.sum(0, keepdims=True) - x) / (ny - 1)
            out = np.empty_like(x)                                      # leave-one-year-out linear trend
            for i in range(ny):
                o_ = np.arange(ny) != i
                t = tyr[o_] - tyr[o_].mean(); xm = x[o_].mean(0)
                b = np.tensordot(t, x[o_] - xm, axes=(0, 0)) / (t ** 2).sum()
                out[i] = x[i] - (xm + b * (tyr[i] - tyr[o_].mean()))
            return out
        for mode in ("loo", "loo_detrended"):
          Ob = anom(Obr, mode); Sm = anom(Smr, mode); C24m = anom(C24r, mode); Cdm = [anom(x, mode) for x in Cdr]
          for rn, mask in regs.items():
              w = W * mask; sw = w.sum()

              def stats(f):                                               # per case: z, mse ; clim mse
                  fo = (w * f * Ob).sum((1, 2)); ff = (w * f * f).sum((1, 2)); oo = (w * Ob * Ob).sum((1, 2))
                  return np.arctanh(np.clip(fo / np.sqrt(ff * oo), -.999, .999)), (w * (f - Ob) ** 2).sum((1, 2)) / sw
              clim = (w * Ob * Ob).sum((1, 2)) / sw
              st = {"sfs": [stats(Sm)], "cfs24": [stats(C24m)], "cfs": [stats(x) for x in Cdm]}

              def met(n, ix):
                  z = np.mean([s[0][ix].mean() for s in st[n]]); e = np.mean([s[1][ix].sum() for s in st[n]])
                  return np.tanh(z), 1 - e / clim[ix].sum()
              bix = rng.integers(0, ny, (NB, ny)); aix = np.arange(ny)
              v = {n: met(n, aix) for n in st}; b = {n: np.array([met(n, i) for i in bix]) for n in st}
              for j, mt in enumerate(("acc", "msss")):
                  for n in st:
                      lo, hi = ci(b[n][:, j])
                      rows.append(dict(part="seasonal", var=var, start=sm, period=sn, region=rn, metric=mt, system=n, n_years=ny, anom=mode,
                                       value=round(float(v[n][j]), 3), lo=lo, hi=hi, sig=""))
                  lo, hi = ci(b["sfs"][:, j] - b["cfs"][:, j])
                  rows.append(dict(part="seasonal", var=var, start=sm, period=sn, region=rn, metric=mt, system="sfs-cfs", n_years=ny, anom=mode,
                                   value=round(float(v["sfs"][j] - v["cfs"][j]), 3), lo=lo, hi=hi, sig="yes" if (lo > 0 or hi < 0) else "no"))
        print(var, sm, sn, ny, "years", YRS[yix[0]], "-", YRS[yix[-1]], flush=True)

with open(OUT + "partB_sfs_vs_cfs_seasonal.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("wrote", len(rows), "rows")
