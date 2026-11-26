"""Surface wind gust estimates from one sounding, for the gust verification (verify_gust.py).

Profiles come in as the Skew-T explorer prepares them (fogport.prepare: P Pa, H m MSL, T and Td K, U and V m/s, the
surface first). Methods, all from the profile alone:
  sfc        the sounding's surface wind
  pbl_mean   the app's "PBL mean wind": the height-weighted mean wind from the ground to the PBL top (compute_sounding's
             pbl_top, the first level where theta-v exceeds the surface value by 0.5 K), app.js meanWindAgl
  pbl_max    the strongest wind below the PBL top (momentum-transfer rule of thumb)
  brasseur   after Brasseur (2001, Mon. Wea. Rev. 129, 5-25): the wind at a level z_p can be brought down by turbulent
             eddies when the mean turbulent kinetic energy between the ground and z_p is at least the buoyant energy
             the descending parcel must overcome,
                 alpha * (1/z_p) * int_0^zp E dz  >=  int_0^zp g (theta_v(z_p) - theta_v(z)) / theta_v(z) dz,
             and the gust is the strongest wind among the levels meeting it. A radiosonde measures no turbulence, so E
             is diagnosed from shear and stability on a GRID_DZ grid with a mixing-length closure:
                 K = l^2 |dV/dz| f(Ri),  K = C_K l sqrt(E)  ->  E = (l |dV/dz| f(Ri) / C_K)^2,
                 l = kappa z / (1 + kappa z / lam),
                 f = (1 - Ri/Ric)^2 for 0 <= Ri < Ric, 0 above Ric, sqrt(1 - 16 Ri) for Ri < 0.
             `upper` is the strongest wind within the turbulent layer (E above E_MIN from the ground up), Brasseur's
             upper bound; `z_src` the height the gust comes from.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

G = 9.80665
KAPPA = 0.4
C_K = 0.5
E_MIN = 0.01  # m2 s-2: the top of the turbulent layer
GRID_TOP = 4000.0  # m AGL
GRID_DZ = 25.0
# the calibration grid the dataset carries (Brasseur's own constants are alpha 1, lam 150 m, Ric 0.25)
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
LAMS = (75.0, 150.0, 300.0)
RICS = (0.25, 0.5, 1.0)


def interp_hagl(prof, p_pa):
    """app.js interpHagl: height AGL (m) at a pressure, linear in log p; None above the profile."""
    P, H = prof["P"], prof["H"]
    if not math.isfinite(p_pa):
        return None
    if p_pa >= P[0]:
        return 0.0
    for i in range(1, len(P)):
        if P[i] <= p_pa:
            f = math.log(P[i - 1] / p_pa) / math.log(P[i - 1] / P[i])
            return H[i - 1] + f * (H[i] - H[i - 1]) - H[0]
    return None


def mean_wind_agl(prof, z0, z1):
    """app.js meanWindAgl: height-weighted mean (u, v) through an AGL layer, None when the winds span < 80 % of it."""
    sfc = prof["H"][0]
    Z, U, V = [], [], []
    for h, u, v in zip(prof["H"], prof["U"], prof["V"]):
        if math.isfinite(h) and math.isfinite(u) and math.isfinite(v):
            Z.append(h - sfc)
            U.append(u)
            V.append(v)
    if len(Z) < 2:
        return None
    a, b = max(z0, Z[0]), min(z1, Z[-1])
    if b - a < 0.8 * (z1 - z0):
        return None

    def at(z):
        i = 1
        while i < len(Z) - 1 and Z[i] < z:
            i += 1
        f = (z - Z[i - 1]) / (Z[i] - Z[i - 1])
        return U[i - 1] + f * (U[i] - U[i - 1]), V[i - 1] + f * (V[i] - V[i - 1])

    pts = [(a, at(a))] + [(Z[i], (U[i], V[i])) for i in range(len(Z)) if a < Z[i] < b] + [(b, at(b))]
    su = sv = 0.0
    for (za, (ua, va)), (zb, (ub, vb)) in itertools.pairwise(pts):
        su += 0.5 * (ua + ub) * (zb - za)
        sv += 0.5 * (va + vb) * (zb - za)
    return su / (b - a), sv / (b - a)


def theta_v(P, T, D):
    """Virtual potential temperature (K) from pressure (Pa), temperature and dewpoint (K)."""
    e = 611.2 * np.exp(17.67 * (D - 273.15) / (D - 29.65))
    q = 0.622 * e / (P - 0.378 * e)
    return T * (1e5 / P) ** 0.2857 * (1 + 0.608 * q)


def grid(prof):
    """(z, u, v, theta_v) on GRID_DZ steps from the ground to GRID_TOP AGL (or the profile's top), linear in height."""
    H = np.asarray(prof["H"], float)
    z = H - H[0]
    top = min(GRID_TOP, z[-1])
    zg = np.arange(0.0, top + 1e-6, GRID_DZ)
    P = np.asarray(prof["P"], float)
    thv = theta_v(P, np.asarray(prof["T"], float), np.asarray(prof["D"], float))
    return (
        zg,
        np.interp(zg, z, np.asarray(prof["U"], float)),
        np.interp(zg, z, np.asarray(prof["V"], float)),
        np.interp(zg, z, thv),
    )


def tke(zg, u, v, thv, lam=150.0, ric=0.25):
    """Diagnosed turbulent kinetic energy (m2 s-2) at the grid levels (layer values moved to the upper level; E(0) = the
    first layer's)."""
    dz = np.diff(zg)
    s2 = (np.diff(u) / dz) ** 2 + (np.diff(v) / dz) ** 2
    zm = 0.5 * (zg[1:] + zg[:-1])
    n2 = G / (0.5 * (thv[1:] + thv[:-1])) * np.diff(thv) / dz
    ri = n2 / np.maximum(s2, 1e-8)
    f = np.where(ri < 0, np.sqrt(np.clip(1 - 16 * ri, 0, None)), np.clip(1 - ri / ric, 0, None) ** 2)
    l = KAPPA * zm / (1 + KAPPA * zm / lam)
    e = (l * np.sqrt(s2) * f / C_K) ** 2
    return np.r_[e[:1], e]


def brasseur(zg, u, v, thv, alpha=1.0, lam=150.0, ric=0.25):
    """-> (gust, upper bound, source height AGL) in m/s, m/s, m."""
    spd = np.hypot(u, v)
    if len(zg) < 3:
        return float(spd[0]), float(spd[0]), 0.0
    E = tke(zg, u, v, thv, lam, ric)
    dz = np.diff(zg)
    cumE = np.r_[0.0, np.cumsum(0.5 * (E[1:] + E[:-1]) * dz)]
    meanE = np.where(zg > 0, cumE / np.maximum(zg, 1e-9), E[0])
    # int_0^zk g (thv_k - thv) / thv dz = g (thv_k * int_0^zk dz / thv - z_k): one running integral for every level
    inv = np.r_[0.0, np.cumsum(0.5 * (1 / thv[1:] + 1 / thv[:-1]) * dz)]
    ok = alpha * meanE >= G * (thv * inv - zg)
    ok[0] = True
    k = int(np.argmax(np.where(ok, spd, -1)))
    turb = np.cumprod(alpha * E >= E_MIN).astype(bool)
    turb[0] = True
    return float(spd[k]), float(spd[turb].max()), float(zg[k])


def features(prof, pbl_top_pa, dcape, sbcape):
    """Every method and the profile features of one prepared profile -> dict (speeds in m/s, heights in m AGL)."""
    zg, u, v, thv = grid(prof)
    spd = np.hypot(u, v)
    out = {"sfc": float(spd[0]), "dcape": dcape, "sbcape": sbcape}
    zp = interp_hagl(prof, pbl_top_pa) if pbl_top_pa is not None and pbl_top_pa > 0 else None
    out["pbl_m"] = zp
    mw = mean_wind_agl(prof, 0.0, zp) if zp is not None and zp >= 50 else None  # app.js: below 50 m, no row
    out["pbl_mean"] = None if mw is None else float(math.hypot(*mw))
    out["pbl_max"] = float(spd[zg <= zp].max()) if zp is not None and zp >= 50 else None
    for h in (250, 500, 1000, 1500, 2000, 3000):
        out[f"w{h}"] = float(np.interp(h, zg, spd)) if zg[-1] >= h else None
        out[f"dthv{h}"] = float(np.interp(h, zg, thv) - thv[0]) if zg[-1] >= h else None
    out["top_m"] = float(zg[-1])
    for a in ALPHAS:
        for lam in LAMS:
            for ric in RICS:
                g, up, zs = brasseur(zg, u, v, thv, a, lam, ric)
                key = f"{a:g}_{lam:g}_{ric:g}"
                out[f"bw_{key}"], out[f"bu_{key}"], out[f"bz_{key}"] = g, up, zs
    return out
