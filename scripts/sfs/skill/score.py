"""Score SFS beta2 vs GEFSv12 vs GEPS8 weekly hindcasts (Sep 1 / Oct 1 starts, 2001-2019) against ERA5.

Anomalies: model A = (F - N) - mean over the OTHER years (F' - N'), N = ERA5 1991-2020 normal over that case's own valid
days (removes the seasonal cycle for GEFS starts that fall up to 3 days off the 1st; for SFS/GEPS it is constant across
years). Truth O = ERA5 - N (same normal). Equal ensemble size: every system as a 4-member mean (SFS: all 15 4-of-6
subsets, scores averaged). Metrics per case over a region (cos-lat weights): uncentred pattern ACC (Fisher-z averaged),
MSSS (sum MSE / sum clim MSE), fair CRPS for SFS and GEFS (members) vs a Gaussian climatology N(0, sd_LOO).
CIs: year-block bootstrap (both start months of a year together), 5000 resamples, 95%."""
import itertools, sys, numpy as np, csv
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import YEARS, MONTHS, OUT, WNAMES, ELAT, ELON

rng = np.random.default_rng(7)
S = np.load(OUT + "sfs_weekly.npz"); G = np.load(OUT + "gefs_weekly.npz"); P = np.load(OUT + "geps_weekly.npz"); E = np.load(OUT + "era5_truth.npz")
W = np.cos(np.deg2rad(ELAT))[:, None] * np.ones((1, len(ELON)))


def conus_mask():
    import cartopy.io.shapereader as shp
    from shapely.geometry import Point
    from shapely.prepared import prep
    geoms = [r.geometry for r in shp.Reader(shp.natural_earth("110m", "cultural", "admin_0_countries")).records()
             if r.attributes.get("ADM0_A3") == "USA"]
    g = prep(geoms[0])
    m = np.zeros((len(ELAT), len(ELON)), bool)
    for i, la in enumerate(ELAT):
        if not 24 <= la <= 50: continue
        for j, lo in enumerate(ELON):
            x = lo - 360 if lo > 180 else lo
            if -125 <= x <= -66 and g.contains(Point(x, la)): m[i, j] = True
    return m


REG = {"NH 20-90N": (ELAT >= 20)[:, None] * np.ones((1, len(ELON)), bool), "CONUS land": conus_mask()}
print({k: int(v.sum()) for k, v in REG.items()}, "grid cells")


def anoms(F, N):
    """F (2, ny, [mem,] 4, lat, lon) forecasts; N (2, ny, 4, lat, lon) the normal. -> anomalies, same shape as F."""
    mem = F.ndim == 6
    Fm = F.mean(2) if mem else F                                 # ensemble mean for the drift estimate
    D = Fm - N                                                   # (2, ny, 4, lat, lon)
    ny = D.shape[1]
    loo = (np.nansum(D, axis=1, keepdims=True) - D) / (ny - 1)    # mean over the other years
    if mem:
        return F - N[:, :, None] - loo[:, :, None]
    return F - N - loo


def case_stats(f, o, mask):
    """f, o (lat, lon) -> (acc, mse, mse_clim) over mask with cos weights."""
    w = W * mask * np.isfinite(f) * np.isfinite(o)
    f = np.nan_to_num(f); o = np.nan_to_num(o)
    sw = w.sum()
    acc = (w * f * o).sum() / np.sqrt((w * f * f).sum() * (w * o * o).sum())
    return acc, (w * (f - o) ** 2).sum() / sw, (w * o * o).sum() / sw


def fair_crps(ens, o):
    """ens (m, lat, lon), o (lat, lon) -> fair CRPS field."""
    m = ens.shape[0]
    t1 = np.mean(np.abs(ens - o[None]), 0)
    t2 = np.abs(ens[:, None] - ens[None, :]).sum((0, 1)) / (2 * m * (m - 1))
    return t1 - t2


def gauss_crps(sd, o):
    from scipy.stats import norm
    z = o / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def fz(r): return np.arctanh(np.clip(r, -0.999, 0.999))


rows = []
SUBS = list(itertools.combinations(range(6), 4))
for var in ("t2m", "z500", "pr"):
    O_s, N_s = E[f"sfs_{var}"], E[f"clim_sfs_{var}"]
    O_g, N_g = (E[f"gefs_{var}"], E[f"clim_gefs_{var}"]) if f"gefs_{var}" in E else (None, None)
    O_p, N_p = E[f"geps_{var}"], E[f"clim_geps_{var}"]
    A_s = anoms(S[var], N_s)                                      # (2, ny, M, 4, lat, lon) members
    A_p = anoms(P[var], N_p)                                      # (2, ny, 4, lat, lon) GEPS 4-member mean
    A_g = anoms(G[var], N_g) if var in G.files else None          # (2, ny, 4mem, 4, lat, lon)
    M = A_s.shape[2]; subs = SUBS if M >= 6 else [tuple(range(4))]
    # LOO sd of truth weekly anomalies per gridpoint (for the Gaussian clim CRPS)
    for wi, wn in enumerate(WNAMES):
        for rn, mask in REG.items():
            if var == "pr" and rn == "NH 20-90N":
                pass
            stat = {}                                             # name -> per case (2*ny,) arrays
            def add(name, vals):
                stat.setdefault(name, []).append(vals)
            for mi in range(2):
                for yi in range(len(YEARS)):
                    o_s = O_s[mi, yi, wi]
                    # SFS: average the scores of the 4-member subsets
                    acc_l, mse_l = [], []
                    for sub in subs:
                        f = A_s[mi, yi, list(sub), wi].mean(0)
                        a, m_, c_ = case_stats(f, o_s, mask); acc_l.append(fz(a)); mse_l.append(m_)
                    add("sfs_z", np.mean(acc_l)); add("sfs_mse", np.mean(mse_l)); add("sfs_clim", c_)
                    a, m_, c_ = case_stats(A_s[mi, yi, :, wi].mean(0), o_s, mask)
                    add("sfsall_z", fz(a)); add("sfsall_mse", m_)
                    o_p = O_p[mi, yi, wi]
                    a, m_, c_ = case_stats(A_p[mi, yi, wi], o_p, mask); add("geps_z", fz(a)); add("geps_mse", m_); add("geps_clim", c_)
                    if A_g is not None:
                        o_g = O_g[mi, yi, wi]
                        a, m_, c_ = case_stats(A_g[mi, yi, :, wi].mean(0), o_g, mask); add("gefs_z", fz(a)); add("gefs_mse", m_); add("gefs_clim", c_)
                        # multi-model (each system verified against ITS truth is not possible for a blend: use the SFS/GEPS
                        # valid days (the 1st); GEFS anomalies are on days up to 3 off - stated as a caveat)
                        mm2 = 0.5 * (A_g[mi, yi, :, wi].mean(0) + A_p[mi, yi, wi])
                        a, m_, _ = case_stats(mm2, o_s, mask); add("mm2_z", fz(a)); add("mm2_mse", m_)
                        z3, e3 = [], []
                        for sub in subs:
                            mm3 = (A_g[mi, yi, :, wi].mean(0) + A_p[mi, yi, wi] + A_s[mi, yi, list(sub), wi].mean(0)) / 3
                            a, m_, _ = case_stats(mm3, o_s, mask); z3.append(fz(a)); e3.append(m_)
                        add("mm3_z", np.mean(z3)); add("mm3_mse", np.mean(e3))
                        # CRPS: SFS (avg over subsets) and GEFS, 4 members each, vs a Gaussian clim
                        ww = W * mask
                        sd_s = np.sqrt(np.nanmean(np.delete(O_s[:, :, wi], yi, axis=1) ** 2, axis=(0, 1)))
                        sd_g = np.sqrt(np.nanmean(np.delete(O_g[:, :, wi], yi, axis=1) ** 2, axis=(0, 1)))
                        cs = np.mean([np.nansum(ww * fair_crps(A_s[mi, yi, list(sub), wi], o_s)) for sub in subs]) / ww.sum()
                        cg = np.nansum(ww * fair_crps(A_g[mi, yi, :, wi], o_g)) / ww.sum()
                        add("sfs_crps", cs); add("sfs_crpsc", np.nansum(ww * gauss_crps(sd_s, o_s)) / ww.sum())
                        add("gefs_crps", cg); add("gefs_crpsc", np.nansum(ww * gauss_crps(sd_g, o_g)) / ww.sum())
            st = {k: np.array(v).reshape(2, len(YEARS)) for k, v in stat.items()}   # (month, year)

            def metric(kind, sysn, idx):
                if kind == "acc":
                    return float(np.tanh(st[f"{sysn}_z"][:, idx].mean()))
                if kind == "msss":
                    ck = f"{sysn}_clim" if f"{sysn}_clim" in st else "sfs_clim"
                    return float(1 - st[f"{sysn}_mse"][:, idx].sum() / st[ck][:, idx].sum())
                if kind == "crpss":
                    return float(1 - st[f"{sysn}_crps"][:, idx].sum() / st[f"{sysn}_crpsc"][:, idx].sum())
            boot = rng.integers(0, len(YEARS), (5000, len(YEARS)))
            allidx = np.arange(len(YEARS))
            systems = ["sfs", "sfsall", "geps"] + (["gefs", "mm2", "mm3"] if A_g is not None else [])
            for kind in ("acc", "msss", "crpss"):
                sl = systems if kind != "crpss" else (["sfs", "gefs"] if A_g is not None else [])
                for sysn in sl:
                    v = metric(kind, sysn, allidx)
                    b = np.array([metric(kind, sysn, bi) for bi in boot])
                    row = {"var": var, "window": wn, "region": rn, "metric": kind, "system": sysn, "value": round(v, 3),
                           "lo": round(float(np.percentile(b, 2.5)), 3), "hi": round(float(np.percentile(b, 97.5)), 3)}
                    rows.append(row)
                pairs = [("sfs", "geps")] + ([("sfs", "gefs"), ("mm3", "mm2")] if A_g is not None else [])
                if kind == "crpss": pairs = [("sfs", "gefs")] if A_g is not None else []
                for x, y in pairs:
                    d = metric(kind, x, allidx) - metric(kind, y, allidx)
                    b = np.array([metric(kind, x, bi) - metric(kind, y, bi) for bi in boot])
                    lo, hi = np.percentile(b, [2.5, 97.5])
                    rows.append({"var": var, "window": wn, "region": rn, "metric": kind, "system": f"{x}-{y}", "value": round(d, 3),
                                 "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                                 "sig": "yes" if (lo > 0 or hi < 0) else "no"})
            print(var, wn, rn, "done", flush=True)

with open(OUT + "skill_table.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=["var", "window", "region", "metric", "system", "value", "lo", "hi", "sig"])
    wr.writeheader(); wr.writerows(rows)
print("wrote", OUT + "skill_table.csv", len(rows), "rows")
