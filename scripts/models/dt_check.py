#!/usr/bin/env python3
"""Validation of the dynamic tropopause in models.html (upper.py), run in Actions (models-dtcheck.yml), never on the
laptop. One valid time; the reference is the GFS's own 2-PVU fields (UPP, "PV=2e-06 (Km^2/kg/s) surface", 0.25 deg,
AWS open data), theta = T (1000/p)^kappa.

  1. method test on ONE model: upper.py's downward search applied to GFS T/u/v on its isobaric levels vs the GFS's own
     2-PVU theta (same model, so only the method differs); also the upward search from 700 hPa for comparison
  2. HRRR: upper.py on wrfprs (as models.py) vs GFS, at HRRR "sub" points, GFS bilinear
  3. RDPS: its own Pressure_PVU-2 with theta interpolated from its levels (as models.py) vs GFS

    python scripts/models/dt_check.py --cycle 2026100612 --lead 6
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import models as M                                      # noqa: E402
import upper as UPR                                     # noqa: E402
sb = M.sb

GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{h:02d}/atmos/gfs.t{h:02d}z.pgrb2.0p25.f{f:03d}"
GFS_LEVS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]


def stats(a, b, tag):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    r = np.corrcoef(a, b)[0, 1] if len(a) > 2 else float("nan")
    print(f"{tag:58s} N={len(a):7d}  bias {d.mean():+6.2f}  MAE {np.abs(d).mean():5.2f}  RMSE {np.sqrt((d ** 2).mean()):5.2f}  "
          f"r {r:.3f}  |d|<5K {100 * (np.abs(d) < 5).mean():5.1f}%  |d|>15K {100 * (np.abs(d) > 15).mean():4.1f}%", flush=True)


def gfs_fields(date, cyc, lead):
    f = M.fetcher("hrrr")                                        # an S3 fetcher (16 threads)
    url = GFS.format(d=date, h=cyc, f=lead)
    want = {f"pv:{v}": rf"^{v}:PV=2e-06 \(Km\^2/kg/s\) surface:" for v in ("PRES", "TMP")}
    for p in GFS_LEVS:
        for v in ("TMP", "UGRD", "VGRD"):
            want[f"{v}:{p}"] = rf"^{v}:{p} mb:(anl|\d+ hour fcst):?$"
    raw = M.fetch_idx_fields("hrrr", url, want)
    v0, g = sb.decode(raw["pv:PRES"], want_grid=True)
    lat, lon = g["lat"][:, 0], g["lon"][0]
    out = {k: sb.decode(b) for k, b in raw.items()}
    return out, lat, lon


def gfs_interp(field, glat, glon, lat, lon):
    """bilinear from the GFS 0.25 deg grid (lat descending or ascending, lon 0..360 or -180..180) to points"""
    lat_asc = glat[0] < glat[-1]
    la = glat if lat_asc else glat[::-1]
    fld = field if lat_asc else field[::-1]
    lo = np.mod(glon, 360.0)
    order = np.argsort(lo)
    lo, fld = lo[order], fld[:, order]
    x = np.mod(lon, 360.0)
    fi = np.interp(x, lo, np.arange(len(lo)))
    fj = np.interp(lat, la, np.arange(len(la)))
    i0, j0 = np.clip(np.floor(fi).astype(int), 0, len(lo) - 2), np.clip(np.floor(fj).astype(int), 0, len(la) - 2)
    wx, wy = fi - i0, fj - j0
    return ((fld[j0, i0] * (1 - wx) + fld[j0, i0 + 1] * wx) * (1 - wy) + (fld[j0 + 1, i0] * (1 - wx) + fld[j0 + 1, i0 + 1] * wx) * wy)


def pv_latlon(T, U, V, levs, lat, lon):
    """Ertel PV on a regular lat/lon grid (dt_charts.py's formula); lat ascending"""
    a = 6.371e6
    p = np.asarray(levs, float) * 100.0
    th = T * (1e5 / p[:, None, None]) ** UPR.KAPPA
    phi, lam = np.radians(lat), np.radians(lon)
    cph = np.cos(phi)[None, :, None]
    dl, dp = lam[1] - lam[0], phi[1] - phi[0]
    ddx = lambda q: (np.roll(q, -1, -1) - np.roll(q, 1, -1)) / (2 * dl) / (a * cph)   # noqa: E731
    ddy = lambda q: np.gradient(q, dp, axis=1) / a                                     # noqa: E731
    zeta = ddx(V) - ddy(U * cph) / cph
    f = (2 * UPR.OMEGA * np.sin(phi))[None, :, None]
    pv = -UPR.G0 * ((zeta + f) * np.gradient(th, p, axis=0) - np.gradient(V, p, axis=0) * ddx(th) + np.gradient(U, p, axis=0) * ddy(th)) * 1e6
    return UPR._121(pv), th


def dt_upward(pv, th, levs):
    """the search dt_charts.py uses: first level at/above 2 PVU going UP from the bottom level"""
    pvr, thr_, lv = pv[::-1], th[::-1], np.log(np.asarray(levs, float))[::-1]
    above = pvr >= UPR.THR
    k = np.argmax(above, 0)
    never = ~above.any(0)
    k1 = np.clip(k, 1, len(lv) - 1)
    k0 = k1 - 1
    jj, ii = np.indices(k.shape)
    p0, p1 = pvr[k0, jj, ii], pvr[k1, jj, ii]
    w = np.clip((UPR.THR - p0) / np.where(np.abs(p1 - p0) < 1e-6, 1e-6, p1 - p0), 0, 1)
    w = np.where(k == 0, 0.0, w)
    t = thr_[k0, jj, ii] * (1 - w) + thr_[k1, jj, ii] * w
    return np.where(never, thr_[-1], t)


def agree(ours, ref, tag):
    """contingency of two yes/no fields: share agreeing, and the Heidke skill score"""
    ok = np.isfinite(ours) & np.isfinite(ref)
    a, b = ours[ok].astype(bool), ref[ok].astype(bool)
    n = a.size
    hits, fa, miss, cn = (a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()
    exp = ((hits + miss) * (hits + fa) + (cn + miss) * (cn + fa)) / max(n, 1)
    hss = (hits + cn - exp) / max(n - exp, 1)
    print(f"{tag:58s} N={n:7d}  agree {100 * (hits + cn) / max(n, 1):5.1f}%  HSS {hss:.2f}  ours-yes {100 * a.mean():4.1f}%  ref-yes {100 * b.mean():4.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--lead", type=int, default=6)
    a = ap.parse_args()
    cyc_dt = dt.datetime.strptime(a.cycle, "%Y%m%d%H")
    date, cyc, lead = a.cycle[:8], cyc_dt.hour, a.lead
    print(f"valid {cyc_dt + dt.timedelta(hours=lead):%Y-%m-%d %HZ} (cycle {a.cycle} +{lead} h)", flush=True)

    # GFS reference + the method test on GFS itself
    G, glat, glon = gfs_fields(date, cyc, lead)
    th_upp = G["pv:TMP"] * (1000.0 / (G["pv:PRES"] / 100.0)) ** UPR.KAPPA
    p_upp = G["pv:PRES"] / 100.0
    st_ = lambda v: np.stack([G[f"{v}:{p}"] for p in GFS_LEVS]).astype(np.float64)    # noqa: E731
    asc = glat[0] < glat[-1]
    flip = (lambda x: x) if asc else (lambda x: x[..., ::-1, :])                        # noqa: E731
    T, U, V = flip(st_("TMP")), flip(st_("UGRD")), flip(st_("VGRD"))
    la = glat if asc else glat[::-1]
    pv, th = pv_latlon(T, U, V, GFS_LEVS, la, glon)
    dn = UPR.dt_downward(pv, th, U, V, GFS_LEVS)
    up = dt_upward(pv, th, GFS_LEVS)
    ref = flip(th_upp)
    LA = la[:, None] * np.ones((1, len(glon)))
    box = (LA > 20) & (LA < 60) & ((np.mod(glon, 360) > 230) & (np.mod(glon, 360) < 300))[None, :]
    nh = (LA > 15) & (LA < 80)
    print("1. method test on the GFS (its own T/u/v on 13 levels vs its UPP 2-PVU theta)")
    stats(dn["theta"][box], ref[box], "   downward search, North America 20-60N 130-60W")
    stats(up[box], ref[box], "   upward search from 700 hPa (dt_charts.py), same box")
    stats(dn["theta"][nh], ref[nh], "   downward search, NH 15-80N")
    stats(up[nh], ref[nh], "   upward search, NH 15-80N")
    stats(dn["p"][box], flip(p_upp)[box], "   downward search, 2-PVU pressure (hPa), North America")

    # HRRR, exactly as models.py computes it
    M._CTX.clear()
    url0 = M.file_url("hrrr", date, cyc, lead)
    z = M.fetch_idx_fields("hrrr", url0, {"zsfc": M.IDX_WANT["hrrr"]["zsfc"], "psfc": M.IDX_WANT["hrrr"]["psfc"]})
    lat, lon, grel = M.grid_latlon(z["zsfc"])
    gd = M.grid_desc(z["zsfc"])
    F = M.SUB["hrrr"]
    M._CTX.update(sub=F, grid_rel=grel, alpha=M.alpha_of(lat, lon) if grel else None, lat_sub=lat[::F, ::F].astype(np.float64),
                  lat1=gd["lat1"], lat2=gd["lat2"], dx=gd["dx"])
    blobs = M.fetch_idx_fields("hrrr", M.prs_url("hrrr", date, cyc, lead), M.ua_want())
    blobs["psfc"] = z["psfc"]
    o = M.ua_lcc(blobs)
    thh = o["dt"][1]
    plat, plon = lat[::F, ::F], lon[::F, ::F]
    gref = gfs_interp(th_upp, glat, glon, plat, plon)
    print("2. HRRR (upper.py on wrfprs, 12 km) vs GFS UPP 2-PVU theta")
    stats(thh.ravel(), gref.ravel(), "   HRRR whole domain")
    inner = (plat > 25) & (plat < 50)
    stats(thh[inner], gref[inner], "   HRRR 25-50N")
    print(f"   HRRR theta range {np.nanpercentile(thh, 1):.0f}-{np.nanpercentile(thh, 99):.0f} K, GFS {np.nanpercentile(gref, 1):.0f}-{np.nanpercentile(gref, 99):.0f} K")
    # the PV charts vs the GFS's own 2-PVU surface: stratospheric air in the 300-250 hPa layer <=> GFS 2-PVU below 275 hPa;
    # PV > 2 on 330 K <=> GFS 2-PVU theta below 330 K
    gp = gfs_interp(p_upp, glat, glon, plat, plon)
    print("2b. HRRR PV charts vs GFS 2-PVU surface (yes/no consistency)")
    if "pvu" in o:
        agree(o["pvu"][1] > 2, np.where(np.isfinite(gp), gp > 275, np.nan), "   300-250 hPa PV > 2  vs  GFS p(2 PVU) > 275 hPa")
    for t in (315, 320, 330, 340):
        k = f"pv{t}"
        if k in o:
            x = o[k][1]
            agree(np.where(np.isfinite(x), x > 2, np.nan), np.where(np.isfinite(gref), gref < t, np.nan), f"   PV({t} K) > 2  vs  GFS theta(2 PVU) < {t} K")
            print(f"      {t} K: masked (outside 100-700 hPa) {100 * np.isnan(x).mean():.0f} %; PV p5/p50/p95 {np.nanpercentile(x, 5):.2f}/{np.nanpercentile(x, 50):.2f}/{np.nanpercentile(x, 95):.2f}")
    if "pvl" in o:
        x = o["pvl"][1]
        print(f"   850-700 hPa PV: masked {100 * np.isnan(x).mean():.0f} %, p5/p50/p95/p99.5 {np.nanpercentile(x, 5):.2f}/{np.nanpercentile(x, 50):.2f}/{np.nanpercentile(x, 95):.2f}/{np.nanpercentile(x, 99.5):.2f} PVU, < 0: {100 * (x < 0).mean():.1f} %")

    # RDPS own 2-PVU
    M._CTX.clear()
    f = M.fetcher("rdps")
    b = f.get(M.rdps_url(date, cyc, max(lead, 1), M.RDPS_VARS["u80"]))
    lat2, lon2, grel2 = M.grid_latlon(b)
    m = (lat2 >= M.RDPS_BOX[2]) & (lat2 <= M.RDPS_BOX[3]) & (lon2 >= M.RDPS_BOX[0]) & (lon2 <= M.RDPS_BOX[1])
    jj, ii = np.where(m)
    crop = ((int(jj.min()), int(jj.max()) + 1), (int(ii.min()), int(ii.max()) + 1))
    sl = (slice(*crop[0]), slice(*crop[1]))
    F2 = M.SUB["rdps"]
    M._CTX.update(sub=F2, crop=crop, grid_rel=grel2, alpha=M.alpha_of(lat2, lon2)[sl] if grel2 else None)
    gm = M.grid_desc(b, crop)
    M._CTX.update(rd_grid=gm, rd_rlat=gm["la1"] + (gm.get("j0", 0) + F2 * np.arange(-(-gm["ny"] // F2))) * gm["dj"],
                  lat_sub=UPR.smooth_dec(lat2[sl], F2).astype(np.float64))
    jobs = M.rdps_ua_jobs(date, cyc, lead)
    jobs["psfc"] = M.rdps_url(date, cyc, lead, M.RDPS_VARS["psfc"])
    rb = {k: f.get(u) for k, u in jobs.items()}
    o2 = M.ua_rdps(rb, crop)
    thr = o2["dt"][1]
    rlat, rlon = lat2[sl][::F2, ::F2], lon2[sl][::F2, ::F2]
    rref = gfs_interp(th_upp, glat, glon, rlat, rlon)
    print("3. RDPS (its Pressure_PVU-2, theta from its levels, 20 km) vs GFS UPP 2-PVU theta")
    stats(thr.ravel(), rref.ravel(), "   RDPS North America crop")
    inner2 = (rlat > 25) & (rlat < 50) & (rlon > -125) & (rlon < -65)
    stats(thr[inner2], rref[inner2], "   RDPS 25-50N 125-65W")
    pd = M.dec(rb["ua:PDT"], crop)
    pd = pd / 100.0 if np.nanmedian(pd) > 2000 else pd
    gp2 = gfs_interp(p_upp, glat, glon, rlat, rlon)
    print("3b. RDPS PV charts (computed here) vs GFS and vs the RDPS's own 2-PVU pressure")
    pds = UPR.smooth_dec(np.where(np.isfinite(M.dec(rb["ua:PDT"], crop)), M.dec(rb["ua:PDT"], crop), np.nan), F2)
    pds = pds / 100.0 if np.nanmedian(pds) > 2000 else pds
    if "pvu" in o2:
        agree(o2["pvu"][1] > 2, np.where(np.isfinite(gp2), gp2 > 275, np.nan), "   RDPS 300-250 hPa PV > 2  vs  GFS p(2 PVU) > 275 hPa")
        agree(o2["pvu"][1] > 2, np.where(np.isfinite(pds), pds > 275, np.nan), "   RDPS 300-250 hPa PV > 2  vs  RDPS's own p(2 PVU) > 275 hPa")
    if "pv330" in o2:
        x = o2["pv330"][1]
        agree(np.where(np.isfinite(x), x > 2, np.nan), np.where(np.isfinite(rref), rref < 330, np.nan), "   RDPS PV(330 K) > 2  vs  GFS theta(2 PVU) < 330 K")
    else:
        print("   RDPS: no PV charts (missing levels?)", sorted(k for k in o2))
    print(f"   RDPS 2-PVU pressure p5/p50/p95 {np.nanpercentile(pd, 5):.0f}/{np.nanpercentile(pd, 50):.0f}/{np.nanpercentile(pd, 95):.0f} hPa")
    for k in ("ua925", "ua250", "z500"):
        for name, oo in (("HRRR", o), ("RDPS", o2)):
            if k in oo:
                x = oo[k]
                print(f"   {name} {k}: " + ", ".join(f"{np.nanmin(c):.1f}..{np.nanmax(c):.1f}" for c in x[1:]))


if __name__ == "__main__":
    main()
