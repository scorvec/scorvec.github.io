#!/usr/bin/env python3
"""Daily vortex strength / structure / geometry metrics (2026-09-30 SSW precursor study). LAPTOP, local data only.

From MERRA-2 zonal means (40-90N, 1 deg; 100..1 hPa; jet-width tail 20-40N from the 0.5 deg m2_zmu file):
  u60_{100,50,10,1}   zonal wind at 60N (m/s)
  vT100               [v'T'] 100 hPa, 45-75N cos-weighted (K m/s)
  jet_lat, jet_u      latitude and speed of the 10 hPa jet maximum (40-85N, parabolic refinement)
  jet_width           full width (deg) where u10 >= half the maximum (20-90N profile)
  depth_ratio         u60(50 hPa) / u60(10 hPa)  (lower-stratosphere share of the vortex; NaN when u60_10 < 5)
  top_shear           u60(1 hPa) - u60(10 hPa)
  qy_edge, qy_edge_lat  max of the QG zonal-mean PV gradient at 10 hPa over 50-85N, in units of 2*Omega/a
                       (a zonal-mean proxy for the edge PV gradient; no isentropic PV maps are used)
  qy_pole             mean QG PV gradient on the poleward flank (jet_lat+3 .. jet_lat+15) at 10 hPa
  bt_neg_upper        area fraction of 40-85N x 10..1 hPa where beta - u_yy < 0 (barotropic instability indicator)
  qg_neg_upper        same for the full QG PV gradient (barotropic + baroclinic, Charney-Stern necessary condition)
  bt_neg_flank        area fraction of the two jet flanks (|lat - jet_lat| 3..15 deg) at 10..1 hPa with beta-u_yy < 0
  n2k1, n2k2          median stationary-wave refractive index a^2 n_k^2 (k = 1, 2) over 55-75N, 50-10 hPa
  n2k2_pos            share of that region where wave 2 can propagate (n^2 > 0)
From NCEP R1 10 hPa heights (20-90N, 2.5 deg; Oct-Apr, to 2026-03-17), moment diagnostics of the area below the
site's vortex edge (30,221 m), equal-area polar coordinates (Mitchell et al. 2011 / Seviour et al. 2013 style):
  cen_lat, cen_lon, aspect, area, kurt? (no: excess kurtosis is not computed)
Output: ~/research/ssw_structure/data/metrics_daily.pkl
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import xarray as xr

import sswr_common as C

A = 6.371e6
OM = 7.292e-5
H = 7000.0
R = 287.05
KAP = 2 / 7
EDGE = 30221.23


def smooth_lat(a, w=5):
    """Running mean along the last axis (latitude), width w points, edge-truncated."""
    k = np.ones(w) / w
    out = np.empty_like(a)
    pad = w // 2
    ap = np.concatenate([np.repeat(a[..., :1], pad, -1), a, np.repeat(a[..., -1:], pad, -1)], -1)
    for i in range(a.shape[-1]):
        out[..., i] = ap[..., i:i + w].mean(-1)
    return out


def qg_terms(u, T, lat, lev):
    """u, T: (time, lev, lat). Returns barotropic beta - u_yy and full QG q_y (1/(m s)), and N^2."""
    phi = np.deg2rad(lat)
    cph = np.cos(phi)
    us = smooth_lat(u.astype("float64"))
    z = -H * np.log(lev / 1000.0)
    # barotropic part
    d1 = np.gradient(us * cph, phi, axis=-1) / np.where(cph < 1e-3, 1e-3, cph)
    d2 = np.gradient(d1, phi, axis=-1)
    beta = 2 * OM * cph / A
    bt = beta - d2 / A ** 2
    # baroclinic part: -(f^2/rho) d/dz(rho/N^2 du/dz), rho ~ exp(-z/H)
    f = 2 * OM * np.sin(phi)
    dTdz = np.gradient(T.astype("float64"), z, axis=1)
    N2 = (R / H) * (dTdz + KAP * T / H)
    N2 = np.clip(N2, 1e-5, None)
    rho = np.exp(-z / H)[None, :, None]
    inner = rho / N2 * np.gradient(us, z, axis=1)
    bc = -(f ** 2) / rho * np.gradient(inner, z, axis=1)
    return bt, bt + bc, N2, us, beta


def main():
    F = C.load_m2_fields()
    t = pd.DatetimeIndex(F["time"])
    lat, lev = F["lat"], F["lev"]
    u, T = F["u"], F["T"]
    il = {L: int(np.where(np.isclose(lev, L))[0][0]) for L in lev}
    j60 = int(np.where(lat == 60)[0][0])
    out = pd.DataFrame(index=t)
    for L in (100, 50, 10, 1):
        out[f"u60_{L}"] = u[:, il[L], j60]
    w = np.cos(np.deg2rad(lat)); m = (lat >= 45) & (lat <= 75)
    out["vT100"] = (F["vT100"][:, m] * w[m]).sum(1) / w[m].sum()
    out["filled"] = F["filled"]
    # ---------------- 10 hPa jet
    Z = C.load_zmu_low()
    tz = pd.DatetimeIndex(Z["time"])
    lowu = pd.DataFrame(Z["u10"], index=tz).reindex(t).interpolate(limit=9).values
    latlow = Z["lat"][Z["lat"] < 40]
    lowu = lowu[:, Z["lat"] < 40]
    prof_lat = np.concatenate([latlow, lat])
    prof = np.concatenate([lowu, u[:, il[10]]], 1)
    sel = (prof_lat >= 40) & (prof_lat <= 85)
    idx = np.where(sel)[0]
    jl, ju, wid = np.full(len(t), np.nan), np.full(len(t), np.nan), np.full(len(t), np.nan)
    for i in range(len(t)):
        p = prof[i]
        if not np.isfinite(p[idx]).all():
            continue
        k = idx[np.argmax(p[idx])]
        if 0 < k < len(p) - 1:
            y0, y1, y2 = p[k - 1], p[k], p[k + 1]
            den = y0 - 2 * y1 + y2
            dx = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den < 0 else 0.0
            dlat = prof_lat[k + 1] - prof_lat[k]
            jl[i] = prof_lat[k] + dx * dlat
            ju[i] = y1 - 0.25 * (y0 - y2) * dx
        else:
            jl[i], ju[i] = prof_lat[k], p[k]
        if ju[i] <= 5:
            continue
        half = ju[i] / 2
        a = k
        while a > 0 and p[a] >= half:
            a -= 1
        b = k
        while b < len(p) - 1 and p[b] >= half:
            b += 1
        if np.isfinite(p[a]) and p[a] < half:      # interpolate the crossing
            la = prof_lat[a] + (half - p[a]) / (p[a + 1] - p[a]) * (prof_lat[a + 1] - prof_lat[a])
        else:
            la = np.nan                               # never fell to half the maximum by 20N
        lb = prof_lat[b] - (half - p[b]) / (p[b - 1] - p[b]) * (prof_lat[b] - prof_lat[b - 1]) if p[b] < half else 90.0
        wid[i] = lb - la
    out["jet_lat"], out["jet_u"], out["jet_width"] = jl, ju, wid
    # ---------------- depth
    out["depth_ratio"] = np.where(out.u60_10 > 5, out.u60_50 / out.u60_10.clip(lower=5), np.nan)
    out["top_shear"] = out.u60_1 - out.u60_10
    # ---------------- QG PV gradient, barotropic instability, refractive index
    bt, qy, N2, us, beta = qg_terms(u, T, lat, lev)
    unit = 2 * OM / A
    q10 = qy[:, il[10]] / unit
    m = (lat >= 50) & (lat <= 85)
    out["qy_edge"] = np.nanmax(q10[:, m], 1)
    out["qy_edge_lat"] = lat[m][np.nanargmax(q10[:, m], 1)]
    pol = np.full(len(t), np.nan)
    flank = np.full(len(t), np.nan)
    up = [il[L] for L in (10, 7, 5, 3, 2, 1)]
    wl = np.cos(np.deg2rad(lat))
    reg = (lat >= 40) & (lat <= 85)
    btn = (bt[:, up][:, :, reg] < 0)
    qgn = (qy[:, up][:, :, reg] < 0)
    out["bt_neg_upper"] = (btn * wl[reg]).sum((1, 2)) / (wl[reg].sum() * len(up))
    out["qg_neg_upper"] = (qgn * wl[reg]).sum((1, 2)) / (wl[reg].sum() * len(up))
    for i in range(len(t)):
        if not np.isfinite(jl[i]):
            continue
        mp = (lat >= jl[i] + 3) & (lat <= min(jl[i] + 15, 88))
        if mp.any():
            pol[i] = q10[i, mp].mean()
        mf = ((np.abs(lat - jl[i]) >= 3) & (np.abs(lat - jl[i]) <= 15) & (lat >= 40) & (lat <= 88))
        flank[i] = ((bt[i][up][:, mf] < 0) * wl[mf]).sum() / (wl[mf].sum() * len(up))
    out["qy_pole"] = pol
    out["bt_neg_flank"] = flank
    phi = np.deg2rad(lat)
    f = 2 * OM * np.sin(phi)
    mr = (lat >= 55) & (lat <= 75)
    lr = [il[L] for L in (50, 30, 20, 10)]
    for k in (1, 2):
        uu = us[:, lr][:, :, mr]
        n2 = (qy[:, lr][:, :, mr] / np.where(uu > 1, uu, np.nan) - (k / (A * np.cos(phi[mr]))) ** 2
              - f[mr] ** 2 / (4 * N2[:, lr][:, :, mr] * H ** 2)) * A ** 2
        n2 = np.where(uu > 1, n2, -999.0)            # easterlies / critical line: no stationary propagation
        out[f"n2k{k}"] = np.median(np.clip(n2, -200, 200).reshape(len(t), -1), 1)
        if k == 2:
            out["n2k2_pos"] = (n2 > 0).reshape(len(t), -1).mean(1)
    # ---------------- R1 z10 moment diagnostics
    fs = sorted(glob.glob(str(C.TD / "z10_r1_nh" / "z10_*.nc")))
    z = xr.open_mfdataset(fs, combine="by_coords").z10.load()
    zt = pd.DatetimeIndex(z.time.values).normalize()
    la, lo = np.deg2rad(z.lat.values), np.deg2rad(z.lon.values)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    r = 2 * np.sin((np.pi / 2 - LA) / 2)
    X, Y = r * np.cos(LO), r * np.sin(LO)
    dA = np.cos(LA)
    keep = LA >= np.deg2rad(30)
    rows = []
    for fld in z.values:
        if not np.isfinite(fld).all():
            rows.append((np.nan,) * 4); continue
        q = np.where(keep, np.clip(EDGE - fld, 0, None), 0) * dA
        inside = np.where(keep & (fld < EDGE), dA, 0)
        m0 = q.sum()
        if m0 <= 0:
            rows.append((np.nan, np.nan, np.nan, 0.0)); continue
        xc, yc = (q * X).sum() / m0, (q * Y).sum() / m0
        J11 = (q * (X - xc) ** 2).sum() / m0
        J22 = (q * (Y - yc) ** 2).sum() / m0
        J12 = (q * (X - xc) * (Y - yc)).sum() / m0
        s = np.sqrt((J11 - J22) ** 2 + 4 * J12 ** 2)
        ar = np.sqrt((J11 + J22 + s) / max(J11 + J22 - s, 1e-12))
        rc = np.hypot(xc, yc)
        clat = 90 - np.rad2deg(2 * np.arcsin(min(rc / 2, 1)))
        clon = np.rad2deg(np.arctan2(yc, xc)) % 360
        rows.append((clat, clon, ar, inside.sum() / (dA * keep).sum()))
    g = pd.DataFrame(rows, index=zt, columns=["cen_lat", "cen_lon", "aspect", "area"])
    g = g[~g.index.duplicated()]
    out = out.join(g, how="left")
    out.to_pickle(C.DATA / "metrics_daily.pkl")
    print(out.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
