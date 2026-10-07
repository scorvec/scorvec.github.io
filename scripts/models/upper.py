#!/usr/bin/env python3
"""Upper-air fields for models.html (2026-10-06, user: "Can we add any upper air charts as well? Maybe 925mb winds, dynamic
tropopause, 250mb winds?"). The physics only; models.py fetches, encodes and publishes.

Fields (every UA_EVERY = 3 forecast hours; the page's slider walks those hours on these maps):
  ua925  925 hPa wind (earth-relative u, v) and height; HRRR/RRFS on the 6 km half-resolution grid ("lo"), RDPS native.
         Masked where the surface pressure is below 925 hPa (the level is underground there; the models extrapolate).
  ua250  250 hPa wind and height, same grids.
  dt     the dynamic tropopause (2-PVU surface): potential temperature and the wind on it, on the 12 km ("sub") grid
         (RDPS: 20 km).
  z500   500 hPa height and absolute vorticity, on the "sub" grid.

Dynamic tropopause, HRRR and RRFS (neither writes PV-surface fields; checked in the .idx inventories 2026-10-06):
  T, u, v on the isobaric levels UA_LEVS (100-400 hPa every 25, 450-700 every 50) are first smoothed and subsampled to
  ~12 km (a centred box of F+1 cells, half weight at the ends, then every F-th point - the "sub" grid of the MSLP files),
  then Ertel PV on the pressure levels, as scripts/mjo/src/dt_charts.py:
      PV = -g [ (zeta + f) dtheta/dp - dv/dp dtheta/dx + du/dp dtheta/dy ]      (1 PVU = 1e-6 K m2 kg-1 s-1)
  with the grid-relative winds and the conformal map factor m of the Lambert grid: zeta = m^2 [d(v/m)/dX - d(u/m)/dY],
  d/dx = m d/dX (X, Y = projection metres along the grid axes). PV gets one 1-2-1 pass in each direction.
  The 2-PVU crossing is searched DOWNWARD from 100 hPa: the first level whose PV and the next level's are both below 2
  PVU (so a one-level dip in the stratosphere and the low-level diabatic PV of convection or fronts cannot make a false,
  low tropopause); theta, ln p and the wind are interpolated linearly in PV between that level and the one above it.
  Where PV is already below 2 PVU at 100 hPa (deep tropics) the 100 hPa values are used (as dt_charts.py), and where no
  crossing is found above 700 hPa the 700 hPa values.
Dynamic tropopause, RDPS: the model's own 2-PVU pressure (Pressure_PVU-2 on the MSC Datamart), with theta and the wind
  interpolated to it in ln p from the isobaric levels.
"""
from __future__ import annotations

import math

import numpy as np

UA_EVERY = 3
UA_LEVS = [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550, 600, 650, 700]
# RRFS comes from NOMADS (no real-time S3 copy): 11 levels, 38 messages per 3 h instead of 62 (its 3 km prslev messages are
# ~0.7 MB each; the 19-level set added ~1.9 GB per cycle there, measured 2026-10-07)
UA_LEVS_RRFS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700]


def levs_of(model):
    return UA_LEVS_RRFS if model == "rrfs" else UA_LEVS_HRRR
RDPS_T_LEVS = [100, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700]
RDPS_UV_LEVS = [100, 150, 200, 250, 300, 400, 500, 600, 700]
G0, OMEGA, KAPPA, THR = 9.80665, 7.2921e-5, 0.2857, 2.0


def ua_hour(lead):
    return lead % UA_EVERY == 0


def smooth_dec(a, F):
    """Centred box of F+1 cells (half weight at both ends = an F-cell block centred on the kept point), edges clamped,
    then every F-th point from grid row/column 0 - the grid of the "sub" files (la1/lo1 unchanged, dx * F)."""
    if F <= 1:
        return np.asarray(a, np.float32)
    h = F // 2
    w = np.ones(2 * h + 1)
    w[0] = w[-1] = 0.5
    w /= w.sum()
    ny, nx = a.shape
    oy, ox = -(-ny // F), -(-nx // F)
    p = np.pad(np.asarray(a, np.float64), ((h, h + F), (h, h + F)), mode="edge")
    out = np.zeros((oy, ox))
    for dj in range(2 * h + 1):
        for di in range(2 * h + 1):
            out += w[dj] * w[di] * p[dj:dj + oy * F:F, di:di + ox * F:F][:oy, :ox]
    return out.astype(np.float32)


def lcc_mapfac(lat, lat1, lat2):
    """Map factor of a spherical Lambert conformal grid (tangent when lat1 == lat2)."""
    l1, l2, la = math.radians(lat1), math.radians(lat2), np.radians(lat)
    n = math.sin(l1) if abs(l1 - l2) < 1e-9 else (
        math.log(math.cos(l1) / math.cos(l2)) / math.log(math.tan(math.pi / 4 + l2 / 2) / math.tan(math.pi / 4 + l1 / 2)))
    return (math.cos(l1) / np.cos(la)) * (math.tan(math.pi / 4 + l1 / 2) / np.tan(np.pi / 4 + la / 2)) ** n


def _121(a):
    b = a.copy()
    b[..., 1:-1, :] = 0.25 * a[..., :-2, :] + 0.5 * a[..., 1:-1, :] + 0.25 * a[..., 2:, :]
    c = b.copy()
    c[..., :, 1:-1] = 0.25 * b[..., :, :-2] + 0.5 * b[..., :, 1:-1] + 0.25 * b[..., :, 2:]
    return c


def ertel_pv(T, U, V, levs_hpa, lat, m, dX):
    """Ertel PV (PVU) on pressure levels of a conformal grid. T, U, V: (nlev, ny, nx), grid-relative winds; lat, m:
    (ny, nx); dX: grid spacing in projection metres. -> pv, theta"""
    p = np.asarray(levs_hpa, np.float64) * 100.0
    th = T * (1e5 / p[:, None, None]) ** KAPPA
    m3 = m[None]

    def dX_(a):
        return np.gradient(a, dX, axis=-1)

    def dY_(a):
        return np.gradient(a, dX, axis=-2)
    zeta = m3 ** 2 * (dX_(V / m3) - dY_(U / m3))
    f = (2 * OMEGA * np.sin(np.radians(lat)))[None]
    th_p = np.gradient(th, p, axis=0)
    u_p = np.gradient(U, p, axis=0)
    v_p = np.gradient(V, p, axis=0)
    pv = -G0 * ((zeta + f) * th_p - v_p * m3 * dX_(th) + u_p * m3 * dY_(th)) * 1e6
    return _121(pv), th, zeta + f


def dt_downward(pv, th, U, V, levs_hpa, thr=THR):
    """theta, p (hPa), u, v on the thr-PVU surface, searched downward from the top level (see the module docstring)."""
    nl = pv.shape[0]
    below = pv < thr
    cand = below.copy()
    cand[:-1] &= below[1:]
    k = np.argmax(cand, axis=0)
    none = ~cand.any(axis=0)
    k1 = np.where(none, nl - 1, k)
    k0 = np.maximum(k1 - 1, 0)
    jj, ii = np.indices(k1.shape)
    p0, p1 = pv[k0, jj, ii], pv[k1, jj, ii]
    den = p0 - p1
    w = np.clip((p0 - thr) / np.where(np.abs(den) < 1e-6, 1e-6, den), 0, 1)
    w = np.where(k1 == 0, 0.0, w)                                  # PV < 2 already at the top: the top level's values
    w = np.where(none, 1.0, w)                                     # no crossing above the bottom level: the bottom's
    lnp = np.log(np.asarray(levs_hpa, np.float64))[:, None, None] * np.ones_like(pv)

    def at(a):
        return a[k0, jj, ii] * (1 - w) + a[k1, jj, ii] * w
    return dict(theta=at(th), p=np.exp(at(lnp)), u=at(U), v=at(V), top=(k1 == 0), none=none)


def interp_lnp(stack, levs_hpa, p_hpa):
    """Values of stack (nlev, ny, nx) at pressure p_hpa (ny, nx), linear in ln p; outside the levels: the end values."""
    lv = np.log(np.asarray(levs_hpa, np.float64))
    x = np.log(np.clip(p_hpa, levs_hpa[0], levs_hpa[-1]))
    k = np.clip(np.searchsorted(lv, x) - 1, 0, len(lv) - 2)        # lv ascending (100 .. 700 hPa)
    jj, ii = np.indices(p_hpa.shape)
    w = (x - lv[k]) / (lv[k + 1] - lv[k])
    return stack[k, jj, ii] * (1 - w) + stack[k + 1, jj, ii] * w


def q_auto(z, steps=(1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 30.0)):
    """Per-frame linear code for heights: offset = floor of the frame minimum to the step, the finest step that spans the
    frame in 254 codes. -> (codes uint8 with 255 = missing, [offset, step])"""
    fin = np.isfinite(z)
    if not fin.any():
        return np.full(z.shape, 255, np.uint8), [0.0, steps[-1]]
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    for s in steps:
        if (hi - math.floor(lo / s) * s) / s <= 254:
            break
    off = math.floor(lo / s) * s
    c = np.clip(np.rint((z - off) / s), 0, 254)
    return np.where(fin, c, 255).astype(np.uint8), [off, s]


# ── potential-vorticity charts (2026-10-07, user: "Can we add some potential vorticity charts to the short-term model
# viewer?"): upper-level layer mean 300-250 hPa, PV on isentropes (315/320/330/340 K), low-level layer mean 850-700 hPa,
# all from the same Ertel PV on pressure levels as the dynamic tropopause. HRRR: UA_LEVS + 750/800/850 (the DT search
# still uses 100-700 hPa only); RDPS: computed here the same way on its rotated lat-lon grid from RDPS_PV_LEVS.
PV_LOW = [750, 800, 850]
UA_LEVS_HRRR = UA_LEVS + PV_LOW
RDPS_PV_LEVS = [100, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]
THETAS = (315.0, 320.0, 330.0, 340.0)
PV_UP = (250, 275, 300)
PV_LO = (700, 750, 800, 850)


def ertel_pv_rot(T, U, V, levs_hpa, rlat_deg, lat_deg, dlam_deg, dphi_deg):
    """Ertel PV (PVU) on pressure levels of a (rotated) regular lat-lon grid: winds relative to the grid, rlat the grid's own
    latitude (rows), lat the true latitude (for f). dlam/dphi: grid spacing in degrees."""
    a = 6.371e6
    p = np.asarray(levs_hpa, np.float64) * 100.0
    th = T * (1e5 / p[:, None, None]) ** KAPPA
    cph = np.cos(np.radians(rlat_deg))[None]
    dl, dp = math.radians(dlam_deg), math.radians(dphi_deg)

    def ddx(q):
        return np.gradient(q, dl, axis=-1) / (a * cph)

    def ddy(q):
        return np.gradient(q, dp, axis=-2) / a
    zeta = ddx(V) - ddy(U * cph) / cph
    f = (2 * OMEGA * np.sin(np.radians(lat_deg)))[None]
    pv = -G0 * ((zeta + f) * np.gradient(th, p, axis=0) - np.gradient(V, p, axis=0) * ddx(th) + np.gradient(U, p, axis=0) * ddy(th)) * 1e6
    return _121(pv), th


def layer_mean(pv, levs, layer):
    idx = [list(levs).index(p) for p in layer]
    return pv[idx].mean(axis=0)


def on_theta(pv, th, U, V, levs_hpa, theta0, pmax=700):
    """PV, p (hPa), u, v on the isentrope theta0, linear in theta between the levels (ln p for pressure); searched upward
    from pmax hPa (the first crossing from below); NaN where theta0 is outside the levels 100..pmax hPa"""
    lv = list(levs_hpa)
    kmax = max(i for i, p in enumerate(lv) if p <= pmax)
    lnp = np.log(np.asarray(lv, np.float64))
    out = {k: np.full(pv.shape[1:], np.nan) for k in ("pv", "p", "u", "v")}
    done = np.zeros(pv.shape[1:], bool)
    for k in range(kmax, 0, -1):                        # k: lower level (higher p), k-1 above it
        t0, t1 = th[k], th[k - 1]
        hit = ~done & (t0 <= theta0) & (t1 >= theta0) & (t1 > t0)
        if not hit.any():
            continue
        w = np.where(hit, (theta0 - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
        for name, arr in (("pv", pv), ("u", U), ("v", V)):
            out[name] = np.where(hit, arr[k] * (1 - w) + arr[k - 1] * w, out[name])
        out["p"] = np.where(hit, np.exp(lnp[k] * (1 - w) + lnp[k - 1] * w), out["p"])
        done |= hit
    return out
