#!/usr/bin/env python3
"""Physics of the tropical-cyclone shear and environment product (tcshear.py drives it; no I/O here).

Everything is computed on a STORM-CENTRED grid: an azimuthal-equidistant plane around the storm's surface centre
(RADIUS_KM x RADIUS_KM, DX_KM spacing), onto which the model's lat-lon fields are interpolated. Horizontal derivatives
are taken on the sphere first (vorticity, divergence) and interpolated afterwards.

1. Vortex removal (Galarneau & Davis 2013, MWR 141, 405-430; the same idea as Kurihara et al. 1993, MWR 121, 2030-
   2045, and the "vortex removed" SHIPS shear SHDC): at each pressure level the storm's OWN circulation is the wind
   induced by the relative vorticity and divergence within R_REMOVE_KM of the centre. It is recovered by inverting
   them with the free-space Green's function of the Laplacian (a Biot-Savart convolution, no boundary condition):
       u_v = -(1/2 pi) int zeta' (y-y')/r^2 dA' + (1/2 pi) int delta' (x-x')/r^2 dA'
       v_v =  (1/2 pi) int zeta' (x-x')/r^2 dA' + (1/2 pi) int delta' (y-y')/r^2 dA'
   and subtracted. The ENVIRONMENTAL wind at that level is the area mean of what is left within R_ENV_KM (SHDC: 0-500
   km). For reference the plain 200-800 km annulus mean of the total wind (SHRD) is kept too: a centred annulus mean
   already cancels the storm's axisymmetric wind, but not its asymmetries or the projection of a tilted vortex.
2. The environmental hodograph is that wind at every level 1000-100 hPa. Shear measures from it (shear()).
3. Vortex tilt: vorticity centroids at 850 and 500 hPa (centroid of CYCLONIC relative vorticity, zeta x sign(lat),
   within R_CENTROID_KM, iterated from the surface centre), tilt = 500 minus 850 hPa centre. A centroid whose walk ends
   more than R_CENTROID_KM from where it started has left its own vortex: no tilt then.
Hemispheres: relative vorticity and Ertel PV are signed (negative for a Southern Hemisphere cyclone and for its
stratospheric air), so every vorticity / PV threshold here is applied to the field times sign(lat) (hemi()).
4. Ventilation (Tang & Emanuel 2012, BAMS 93, 1901-1912): VI = shear x entropy deficit / potential intensity, with
   moist entropy as their eq. 3 (Bryan 2008) and PI from tcpyPI (Gilford 2021, GMD 14, 2351-2369; Bister & Emanuel
   2002).
"""
from __future__ import annotations

import math

import numpy as np

A_EARTH = 6.371e6
OMEGA = 7.292e-5
KT = 1.0 / 0.514444                  # m/s -> kt

RADIUS_KM = 1000.0                   # half-width of the storm-centred plane
DX_KM = 20.0
R_REMOVE_KM = 500.0                  # vorticity / divergence of the storm inside this radius = the storm's own flow
R_ENV_KM = 500.0                     # environmental wind = mean of the residual inside this radius (SHIPS SHDC)
ANN_KM = (200.0, 800.0)              # SHIPS SHRD annulus (total wind)
R_CENTROID_KM = 150.0
CENT_DX_KM = 5.0
MAX_WALK_KM = R_CENTROID_KM          # a centroid that wanders further than its own search radius is another vortex

LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100)
SHIPS_GS_LEVELS = (1000, 850, 700, 500, 400, 300, 250, 200, 150, 100)    # SHIPS lsdiag.f gshear levels (no 925 / 600)


def hemi(lat) -> float:
    """+1 in the Northern Hemisphere, -1 in the Southern: cyclonic vorticity and stratospheric PV times hemi() > 0."""
    return -1.0 if float(lat) < 0 else 1.0


def norm_lon(lon):
    """Longitude in -180..180."""
    return (np.asarray(lon, float) + 180.0) % 360.0 - 180.0


# ── geometry ──────────────────────────────────────────────────────────────────────────────────────────────────────
def gc_km(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    dl = np.radians(np.asarray(lo2) - np.asarray(lo1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * A_EARTH / 1e3 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def aeqd_inverse(clat, clon, x_km, y_km):
    """lat, lon of plane points (x east, y north, km) on the azimuthal-equidistant plane centred at clat, clon."""
    r = np.hypot(x_km, y_km) / (A_EARTH / 1e3)                       # angular distance
    az = np.arctan2(x_km, y_km)
    p1, l1 = math.radians(clat), math.radians(clon)
    p2 = np.arcsin(np.sin(p1) * np.cos(r) + np.cos(p1) * np.sin(r) * np.cos(az))
    l2 = l1 + np.arctan2(np.sin(az) * np.sin(r) * np.cos(p1), np.cos(r) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 180.0) % 360.0 - 180.0


def offset_km(clat, clon, lat, lon):
    """(x east, y north) km of lat, lon on the plane centred at clat, clon (azimuthal equidistant)."""
    d = gc_km(clat, clon, lat, lon)
    p1, p2 = math.radians(clat), np.radians(lat)
    dl = np.radians(np.asarray(lon) - clon)
    az = np.arctan2(np.sin(dl) * np.cos(p2), math.cos(p1) * np.sin(p2) - math.sin(p1) * np.cos(p2) * np.cos(dl))
    return d * np.sin(az), d * np.cos(az)


def step_km(clat, clon, x_km, y_km):
    la, lo = aeqd_inverse(clat, clon, np.atleast_1d(float(x_km)), np.atleast_1d(float(y_km)))
    return float(la[0]), float(lo[0])


class Plane:
    """The storm-centred plane and the interpolation weights from one lat-lon box onto it."""
    def __init__(self, clat, clon, radius_km=RADIUS_KM, dx_km=DX_KM):
        n = int(round(radius_km / dx_km))
        ax = np.arange(-n, n + 1) * dx_km
        self.x, self.y = np.meshgrid(ax, ax)                           # [j (y), i (x)]
        self.r = np.hypot(self.x, self.y)
        self.dx = dx_km
        self.clat, self.clon = clat, clon
        self.lat, self.lon = aeqd_inverse(clat, clon, self.x, self.y)

    def sample(self, box_lat, box_lon, field):
        """Bilinear interpolation of field[..., lat, lon] (box_lat ascending, box_lon ascending and continuous, in any
        360-degree window, e.g. -197..-158 for a box across the dateline) onto the plane. NaN outside the box."""
        from scipy.ndimage import map_coordinates
        lo0 = float(box_lon[0])
        # every plane longitude into [lo0, lo0 + 360): a storm at -178 has plane points at +179 that the box holds as -181
        lon = lo0 + np.mod(self.lon - lo0 + 1e-6, 360.0) - 1e-6
        fj = (self.lat - box_lat[0]) / (box_lat[1] - box_lat[0])
        fi = (lon - lo0) / (box_lon[1] - box_lon[0])
        out_of = (fj < 0) | (fj > len(box_lat) - 1) | (fi < 0) | (fi > len(box_lon) - 1)
        f = np.asarray(field)
        flat = f.reshape((-1,) + f.shape[-2:])
        res = np.empty((flat.shape[0],) + self.x.shape, np.float64)
        for k in range(flat.shape[0]):
            res[k] = map_coordinates(flat[k].astype(np.float64), [fj, fi], order=1, mode="nearest")
            res[k][out_of] = np.nan
        return res.reshape(f.shape[:-2] + self.x.shape)


def box(lat, lon, clat, clon, half_lat=11.5, half_lon=None):
    """Index slices of a lat-lon box around (clat, clon) on a global grid (lat descending or ascending, lon 0..360).
    Returns (jj, ii, box_lat ascending, box_lon ascending in -180..180 continuous); the caller takes f[..., jj][..., ii]."""
    if half_lon is None:
        half_lon = min(40.0, half_lat / max(0.3, math.cos(math.radians(min(abs(clat) + half_lat, 75.0)))))
    jo = np.argsort(lat)
    la = lat[jo]
    jsel = jo[(la >= clat - half_lat) & (la <= clat + half_lat)]
    dl = (np.asarray(lon) - clon + 180.0) % 360.0 - 180.0
    isel = np.where(np.abs(dl) <= half_lon)[0]
    isel = isel[np.argsort(dl[isel])]
    return jsel, isel, lat[jsel], clon + dl[isel]


# ── vorticity, divergence, inversion ──────────────────────────────────────────────────────────────────────────────
def vort_div(u, v, lat, lon):
    """Relative vorticity and divergence (s-1) on a regular lat-lon box (lat ascending, lon ascending, degrees)."""
    phi = np.radians(lat)[:, None]
    dlam = math.radians(lon[1] - lon[0]); dphi = math.radians(lat[1] - lat[0])
    cos = np.cos(phi)
    du_dl = np.gradient(u, dlam, axis=-1); dv_dl = np.gradient(v, dlam, axis=-1)
    ducos = np.gradient(u * cos, dphi, axis=-2); dvcos = np.gradient(v * cos, dphi, axis=-2)
    zeta = (dv_dl - ducos) / (A_EARTH * cos)
    div = (du_dl + dvcos) / (A_EARTH * cos)
    return zeta, div


_KERN: dict = {}


def _kernels(n, dx_km):
    """FFTs of the Biot-Savart kernels x/(2 pi r^2) dA and y/(2 pi r^2) dA on a zero-padded grid big enough for a
    LINEAR convolution of a (2n+1)^2 field (no wrap-around), cached per plane size."""
    from scipy import fft as sfft
    key = (n, dx_km)
    if key not in _KERN:
        m = 2 * n + 1
        L = sfft.next_fast_len(3 * m)
        ax = np.arange(-(2 * n), 2 * n + 1) * dx_km * 1e3
        X, Y = np.meshgrid(ax, ax)
        r2 = X ** 2 + Y ** 2
        r2[r2 == 0] = np.inf
        dA = (dx_km * 1e3) ** 2
        def spec(K):
            P = np.zeros((L, L))
            P[:K.shape[0], :K.shape[1]] = K
            return sfft.rfft2(P)
        _KERN[key] = (L, spec(X / (2 * np.pi * r2) * dA), spec(Y / (2 * np.pi * r2) * dA))
    return _KERN[key]


def vortex_wind(zeta, div, plane: Plane, r0_km=R_REMOVE_KM):
    """The wind induced by zeta and div inside r0 of the centre, everywhere on the plane (free-space inversion)."""
    from scipy import fft as sfft
    n = (plane.x.shape[0] - 1) // 2
    m = 2 * n + 1
    L, Kx, Ky = _kernels(n, plane.dx)
    inside = plane.r <= r0_km
    Z = sfft.rfft2(np.where(inside, np.nan_to_num(zeta), 0.0), s=(L, L))
    D = sfft.rfft2(np.where(inside, np.nan_to_num(div), 0.0), s=(L, L))
    u = sfft.irfft2(-Ky * Z + Kx * D, s=(L, L))[2 * n:2 * n + m, 2 * n:2 * n + m]
    v = sfft.irfft2(Kx * Z + Ky * D, s=(L, L))[2 * n:2 * n + m, 2 * n:2 * n + m]
    return u, v


def environment(u, v, zeta, div, plane: Plane):
    """Per level (arrays [lev, j, i] on the plane): environmental wind (vortex removed, mean within R_ENV_KM), the SHRD-
    style annulus mean of the total wind, and the fraction of valid points (coverage). -> dict of arrays [lev]."""
    nl = u.shape[0]
    env = np.full((nl, 2), np.nan); ann = np.full((nl, 2), np.nan); cov = np.zeros(nl)
    disc = plane.r <= R_ENV_KM
    ring = (plane.r >= ANN_KM[0]) & (plane.r <= ANN_KM[1])
    for k in range(nl):
        ok = np.isfinite(u[k]) & np.isfinite(v[k])
        cov[k] = ok[disc].mean()                     # a limited-area model (HRRR) must cover the whole 0-500 km disc
        if cov[k] < 0.98:
            continue
        uv, vv = vortex_wind(zeta[k], div[k], plane)
        m = disc & ok
        env[k] = (np.mean(u[k][m] - uv[m]), np.mean(v[k][m] - vv[m]))
        m2 = ring & ok
        if ok[ring].mean() >= 0.9:
            ann[k] = (np.mean(u[k][m2]), np.mean(v[k][m2]))
    return dict(env=env, ann=ann, cov=cov)


# ── shear measures ────────────────────────────────────────────────────────────────────────────────────────────────
def vec(du, dv):
    """(magnitude m/s, 'from' direction deg: the meteorological convention SHIPS uses - westerly shear = 270)."""
    return float(np.hypot(du, dv)), float((np.degrees(np.arctan2(-du, -dv)) + 360.0) % 360.0)


def layer_shear(env, levels, top, bot):
    k1, k0 = list(levels).index(top), list(levels).index(bot)
    return env[k1, 0] - env[k0, 0], env[k1, 1] - env[k0, 1]


def generalized_shear(env, levels, p_top=100.0, p_bot=1000.0):
    """Generalized shear (Knaff, Sampson & DeMaria 2005, WAF 20, 688-699, eq. 3; the 1000-100 hPa form of the
    operational SHIPS code, lsdiag.f gshear): 4 x the mass-weighted mean magnitude of the wind's departure from its
    mass-weighted column mean,
        GS = 4 sum_p w_p |V_p - Vbar|,   Vbar = sum_p w_p V_p,   w_p = trapezoidal dp / (p_bot - p_top).
    For a wind that changes linearly with pressure it equals the wind difference across the WHOLE 1000-100 hPa column,
    i.e. 900/650 = 1.38 times its 200-850 hPa shear (10 m/s of 200-850 hPa shear -> 13.8 m/s; the trapezoid weights on
    the SHIPS levels give 14.4); a profile with its shear concentrated in one layer, or turning with height, scores
    higher relative to its two-level shear. The product uses the SHIPS levels (SHIPS_GS_LEVELS), as validated."""
    p = np.asarray(levels, float)
    sel = (p <= p_bot) & (p >= p_top) & np.isfinite(env[:, 0])
    p, U = p[sel], env[sel]
    if len(p) < 6 or p.max() < p_bot - 1 or p.min() > p_top + 1:
        return np.nan
    o = np.argsort(-p); p, U = p[o], U[o]
    w = np.zeros(len(p))
    dp = -np.diff(p)
    w[:-1] += dp / 2; w[1:] += dp / 2
    w /= w.sum()
    mean = (w[:, None] * U).sum(0)
    return float(4.0 * (w * np.hypot(*(U - mean).T)).sum())


def _profile_50(env, levels):
    """The environmental wind interpolated (linear in p) to 850, 800, ..., 200 hPa."""
    p = np.asarray(levels, float)
    ok = np.isfinite(env[:, 0])
    o = np.argsort(p[ok])
    pp, U = p[ok][o], env[ok][o]
    if pp.min() > 200 or pp.max() < 850:
        return None, None
    g = np.arange(200.0, 851.0, 50.0)
    return g, np.stack([np.interp(g, pp, U[:, 0]), np.interp(g, pp, U[:, 1])], 1)


def shear_height_depth(env, levels):
    """Shear height and depth after Finocchio & Majumdar (2017, MWR 145, 361-378), from the profile on a 50-hPa grid:
      local shear  |V(p+50) - V(p-50)| / 100 hPa at each interior level;
      DEPTH        the deepest layer within 200-850 hPa where the local shear stays above 66 % of its profile average (hPa);
      HEIGHT       h = sum_p [CSD_prof(p) - CSD_lin(p)], CSD_prof = shear accumulated from 200 hPa downward / its total,
                   CSD_lin = (p - 200) / 650: positive = shear concentrated in the upper troposphere.
    Returns (h, depth hPa). FM17 found neither explains more than 5 % of intensity-change variance; Finocchio et al.
    (2016) found shallow, low-level shear the most destructive in idealized runs."""
    g, U = _profile_50(env, levels)
    if g is None:
        return np.nan, np.nan
    loc = np.full(len(g), np.nan)
    loc[1:-1] = np.hypot(*(U[2:] - U[:-2]).T) / 100.0
    loc[0] = np.hypot(*(U[1] - U[0])) / 50.0
    loc[-1] = np.hypot(*(U[-1] - U[-2])) / 50.0
    mean = loc.mean()
    if mean <= 0:
        return np.nan, np.nan
    strong = loc > 0.66 * mean
    best = run = 0
    for k in range(len(g)):
        run = run + 1 if strong[k] else 0
        best = max(best, run)
    depth = max(0, best - 1) * 50.0
    seg = np.hypot(*np.diff(U, axis=0).T)                               # 200->250, ..., 800->850
    csd = np.r_[0.0, np.cumsum(seg)] / max(seg.sum(), 1e-6)            # accumulated from 200 hPa downward
    h = float(np.sum(csd - (g - 200.0) / 650.0))
    return h, depth


# ── tilt ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def centroid(box_lat, box_lon, zeta, clat, clon, r_km=R_CENTROID_KM, iters=25, max_walk_km=MAX_WALK_KM):
    """Centroid of CYCLONIC relative vorticity (zeta x sign(clat) > 0) within r_km, iterated from (clat, clon).
    The walk is capped: once the centroid is more than max_walk_km from where it started it has locked onto another
    vorticity feature (or this level has no coherent vortex), and the result is None, not a 300-700 km 'tilt'.
    -> (lat, lon, moved_km) or None."""
    sgn = hemi(clat)
    la, lo = clat, clon
    tot = 0.0
    for _ in range(iters):
        pl = Plane(la, lo, radius_km=r_km + 2 * CENT_DX_KM, dx_km=CENT_DX_KM)
        z = pl.sample(box_lat, box_lon, zeta)
        m = (pl.r <= r_km) & np.isfinite(z)
        w = np.where(m, np.maximum(sgn * z, 0.0), 0.0)
        if w.sum() <= 0:
            return None
        cx, cy = (w * pl.x).sum() / w.sum(), (w * pl.y).sum() / w.sum()
        la, lo = step_km(la, lo, cx, cy)
        tot = float(gc_km(clat, clon, la, lo))
        if tot > max_walk_km:
            return None
        if math.hypot(cx, cy) < 0.5:
            break
    return la, lo, tot


def bearing_of(dx, dy):
    """Compass bearing (deg, 0 = north, clockwise) TOWARD which the vector (east, north) points."""
    return float((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)


def rel_angle(bearing, shear_from):
    """Angle of a direction relative to DOWNSHEAR, counter-clockwise positive: 0 = downshear, +90 = downshear-left
    side (left of the shear vector), 180 = upshear, -90 = right of shear. shear_from: meteorological 'from' deg."""
    downshear = (shear_from + 180.0) % 360.0
    return float(((downshear - bearing) + 180.0) % 360.0 - 180.0)


# ── thermodynamics ────────────────────────────────────────────────────────────────────────────────────────────────
CPD, RD, RV, CL, CPV, LV0 = 1005.7, 287.04, 461.5, 4190.0, 1870.0, 2.501e6
EPS = RD / RV


def esat(T):
    """Saturation vapour pressure over liquid (hPa), Bolton (1980)."""
    Tc = T - 273.15
    return 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))


def mixing_ratio(q):
    return q / (1.0 - q)


LV_B08 = 2.555e6                     # Bryan (2008) pseudoadiabatic latent heat constant, as Tang & Emanuel (2012) eq. 3


def entropy(T, r, p_hpa, saturated=False):
    """Moist entropy (J kg-1 K-1), Tang & Emanuel (2012) eq. 3 (Bryan 2008, pseudoadiabatic):
       s = cp ln T - Rd ln pd + Lvo r / T - Rv r ln H,   Lvo = 2.555e6 J/kg.
    saturated=True: s* (r = r*, H = 1). (Emanuel's 1994 form with (cpd + r cl) ln T carries a reference-state term
    that does not cancel between s* and s at different water contents, inflating the deficit three-fold.)"""
    es = esat(T)
    rs = EPS * es / (p_hpa - es)
    if saturated:
        r = rs
    e = r * p_hpa / (EPS + r)
    H = np.clip(e / es, 1e-6, 1.0)
    pd = (p_hpa - e) * 100.0
    return CPD * np.log(T) - RD * np.log(pd) + LV_B08 * r / T - r * RV * np.log(H)


def potential_intensity(sst_c, msl_hpa, levels, T, r, full=False):
    """Bister-Emanuel PI (m/s, 10 m wind with V_reduc 0.8) from tcpyPI (Gilford 2021, GMD 14, 2351-2369), pseudo-
    adiabatic ascent as Tang & Emanuel (2012). T K, r kg/kg, levels hPa. full=True -> (vmax, pmin, outflow T K) or None."""
    from tcpyPI import pi
    p = np.asarray(levels, float)
    o = np.argsort(-p)
    vmax, pmin, ifl, to, otl = pi(float(sst_c), float(msl_hpa), p[o], np.asarray(T, float)[o] - 273.15,
                                  np.asarray(r, float)[o] * 1e3, CKCD=0.9, ascent_flag=1, diss_flag=1, V_reduc=0.8,
                                  ptop=50, miss_handle=1)
    if ifl != 1 or not np.isfinite(vmax):
        return None if full else (np.nan, np.nan)
    return (float(vmax), float(pmin), float(to)) if full else (float(vmax), float(pmin))


# ── one storm, one time, one model member ─────────────────────────────────────────────────────────────────────────
P_MID = 600                          # Tang & Emanuel (2012): entropy deficit at 600 hPa
R_CORE_DISC_KM = 100.0               # TE12: inner-core saturation entropy averaged over a 100 km disc ...
R_ENV600_KM = (100.0, 300.0)         # ... the environmental 600 hPa entropy over a 100-300 km annulus


def ventilation(shear_ms, T_core600, T_env, r_env, levels, sst_c, msl_hpa, T600_ann, r600_ann):
    """Tang & Emanuel (2012, BAMS 93, 1901-1912) ventilation index  VI = u_shear chi_m / u_PI.
       chi_m = (s*_m - s_m) / (s*_SST - s_b)
         s*_m   saturation entropy at 600 hPa in the inner core (0-100 km mean temperature)
         s_m    entropy at 600 hPa in the 100-300 km annulus
         s*_SST - s_b  the air-sea disequilibrium; TE12 evaluate it at the radius of maximum wind of a storm AT its PI
                with the Bister-Emanuel algorithm, so here it comes from the same PI calculation:
                (s*_SST - s_b) = V_grad^2 / [(Ck/Cd) (Ts/To) (Ts - To)]   (BE02 with dissipative heating)
       u_PI     Bister-Emanuel PI (tcpyPI, pseudoadiabatic ascent as TE12, Ck/Cd 0.9, 10 m wind = 0.8 x gradient wind)
                from the 200-800 km environmental sounding and the OISST under the storm.
       Entropy: TE12 eq. 3. -> dict(vi, chi, chi_env, pi, pmin, sst, diseq, vi_env)."""
    out = dict(vi=np.nan, chi=np.nan, pi=np.nan, pmin=np.nan, sst=sst_c)
    if not np.isfinite(sst_c):
        return out
    pi_ = potential_intensity(sst_c, msl_hpa, levels, T_env, r_env, full=True)
    if pi_ is None:
        return out
    vmax, pmin, to = pi_
    ts = sst_c + 273.15
    vg = vmax / 0.8
    diseq = vg ** 2 / (0.9 * (ts / to) * (ts - to)) if to < ts else np.nan
    s_m_star = entropy(T_core600, 0.0, P_MID, saturated=True)
    s_m = entropy(T600_ann, r600_ann, P_MID)
    chi = (s_m_star - s_m) / diseq if np.isfinite(diseq) and diseq > 1.0 else np.nan
    # sensitivity: the inner core taken saturated at the ENVIRONMENTAL 600 hPa temperature (no resolved warm core), the
    # closer analogue of TE12's reanalysis (their 100 km disc barely resolves a warm core; they note s*_m is then
    # underestimated for strong storms)
    chi_e = (entropy(T600_ann, 0.0, P_MID, saturated=True) - s_m) / diseq if np.isfinite(diseq) and diseq > 1.0 else np.nan
    out.update(pi=vmax, pmin=pmin, chi=float(chi), chi_env=float(chi_e), diseq=float(diseq))
    if np.isfinite(chi) and vmax > 5.0:
        out["vi"] = float(shear_ms * chi / vmax)
    if np.isfinite(chi_e) and vmax > 5.0:
        out["vi_env"] = float(shear_ms * chi_e / vmax)
    return out


def pv330(box, levels=LEVELS, theta=330.0):
    """Ertel PV on the 330 K isentrope over the box, with the models page's code (scripts/models/upper.py:
    ertel_pv_rot + on_theta, the same PV as the dynamic-tropopause charts). -> (pv [lat, lon] PVU, lat, lon)."""
    import upper
    lat, lon = box["lat"], box["lon"]
    o = np.argsort(np.asarray(levels))                                 # on_theta wants pressure ascending (top first)
    lv = np.asarray(levels)[o]
    T, U, V = (np.asarray(box[k], np.float64)[o] for k in ("t", "u", "v"))
    LA = np.repeat(lat[:, None], len(lon), 1)
    pv, th = upper.ertel_pv_rot(T, U, V, lv, LA, LA, float(lon[1] - lon[0]), float(lat[1] - lat[0]))
    on = upper.on_theta(pv, th, U, V, lv, theta, pmax=700)
    return on["pv"], lat, lon


R_CORE_KM = 400.0                    # the storm's own diabatic PV tower reaches 330 K: trough metrics start outside it


def trough_metrics(pv, lat, lon, clat, clon, thr=2.0, r_max=1700.0):
    """Nearest stratospheric air on 330 K (|PV| >= thr: PV x sign(lat), PV is negative in the Southern Hemisphere) and
    the strongest 330 K PV gradient between R_CORE_KM and 1000 km (the storm's own core PV excluded): the upstream
    trough as the shear's source.
    -> dict(d2pvu km or None, b2pvu bearing, gmax PVU per 100 km, and d2far=True only when there is truly no such air
       within r_max: the PV is defined over at least 95 % of the R_CORE_KM..r_max disc and the box reaches r_max.
       d2pvu None without d2far = not determined (box edge, missing PV), never 'beyond r_max')."""
    from scipy.ndimage import gaussian_filter
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    d = gc_km(clat, clon, LA, LO)
    pv = hemi(clat) * np.asarray(pv, float)
    # smooth ~1 deg first: isolated convective (diabatic) PV blobs on 330 K are not the trough
    ok = np.isfinite(pv)
    w = gaussian_filter(ok.astype(float), 3.0)
    pv = np.where(ok & (w > 0.3), gaussian_filter(np.where(ok, pv, 0.0), 3.0) / np.maximum(w, 1e-6), np.nan)
    hit = np.isfinite(pv) & (pv >= thr) & (d <= r_max) & (d >= R_CORE_KM)
    out = dict(d2pvu=None, b2pvu=None, gmax=None)
    if hit.any():
        k = np.argmin(np.where(hit, d, np.inf))
        x, y = offset_km(clat, clon, LA.flat[k], LO.flat[k])
        out["d2pvu"] = round(float(d.flat[k])); out["b2pvu"] = round(bearing_of(float(x), float(y)))
    else:
        disc = (d >= R_CORE_KM) & (d <= r_max)
        edge = min(d[0].min(), d[-1].min(), d[:, 0].min(), d[:, -1].min())
        if disc.any() and np.isfinite(pv[disc]).mean() >= 0.95 and edge >= r_max:
            out["d2far"] = True
    dy = (lat[1] - lat[0]) * 111.2
    dx = (lon[1] - lon[0]) * 111.2 * np.cos(np.radians(LA))
    gy, gx = np.gradient(pv, axis=0) / dy, np.gradient(pv, axis=1) / dx
    g = np.hypot(gx, gy) * 100.0
    m = (d <= 1000.0) & (d >= R_CORE_KM) & np.isfinite(g)
    if m.any():
        out["gmax"] = round(float(np.nanmax(g[m])), 2)
    return out


def analyze(box, clat, clon, sst_c, levels=LEVELS, want_pv=True):
    """Everything for one storm-time-member from its lat-lon box. -> dict (JSON-ready, rounded)."""
    lat, lon = box["lat"], box["lon"]
    u, v = np.asarray(box["u"], float), np.asarray(box["v"], float)
    z, d = vort_div(u, v, lat, lon)
    pl = Plane(clat, clon)
    U = pl.sample(lat, lon, np.stack([u, v, z, d]))
    E = environment(U[0], U[1], U[2], U[3], pl)
    env, ann = E["env"], E["ann"]
    rec = dict(lat=round(clat, 2), lon=round(float(norm_lon(clon)), 2))
    if not np.isfinite(env[levels.index(850)]).all() or not np.isfinite(env[levels.index(200)]).all():
        rec["cov"] = round(float(np.nanmin(E["cov"])), 2)
        return rec
    rec["env"] = [[round(float(a), 1), round(float(b), 1)] if np.isfinite(a) else None for a, b in env]
    rec["ann"] = [[round(float(a), 1), round(float(b), 1)] if np.isfinite(a) else None for a, b in ann]
    for name, top, bot in (("deep", 200, 850), ("low", 500, 850), ("up", 200, 500)):
        mag, frm = vec(*layer_shear(env, levels, top, bot))
        rec[name] = [round(mag, 1), round(frm)]
    mag, frm = vec(*layer_shear(ann, levels, 200, 850))
    rec["shrd"] = [round(mag, 1), round(frm)] if np.isfinite(mag) else None
    gsel = [list(levels).index(p) for p in SHIPS_GS_LEVELS]                # the SHIPS form, as validated (verify.py)
    gs = generalized_shear(env[gsel], SHIPS_GS_LEVELS)
    rec["gs"] = round(gs, 1) if np.isfinite(gs) else None
    gr = generalized_shear(ann[gsel], SHIPS_GS_LEVELS)
    rec["gsr"] = round(gr, 1) if np.isfinite(gr) else None              # SHIPS SHRG analogue (200-800 km, total wind)
    if rec["shrd"] is not None:
        rec["shrs"] = [round(x) if i else round(x, 1) for i, x in enumerate(vec(*layer_shear(ann, levels, 500, 850)))]
    h, dp = shear_height_depth(env, levels)
    rec["sh_h"] = round(h, 2) if np.isfinite(h) else None
    rec["sh_d"] = round(dp) if np.isfinite(dp) else None
    # tilt: 850 -> 500 hPa vorticity centroids
    c850 = centroid(lat, lon, z[levels.index(850)], clat, clon)
    c500 = centroid(lat, lon, z[levels.index(500)], *(c850[:2] if c850 else (clat, clon)))
    if c850 and c500:
        x, y = offset_km(c850[0], c850[1], c500[0], c500[1])
        t_km = float(math.hypot(x, y))
        rec["c850"] = [round(c850[0], 3), round(c850[1], 3)]
        rec["c500"] = [round(c500[0], 3), round(c500[1], 3)]
        rec["tilt"] = [round(t_km, 1), round(bearing_of(float(x), float(y)))]
        rec["tilt_rel"] = round(rel_angle(bearing_of(float(x), float(y)), rec["deep"][1]))
    # thermodynamics: environmental sounding = 200-800 km annulus mean of T and q (PI); TE12 discs for the deficit
    ring = (pl.r >= ANN_KM[0]) & (pl.r <= ANN_KM[1])
    TQ = pl.sample(lat, lon, np.stack([np.asarray(box["t"], float), np.asarray(box["q"], float)]))
    T_env = np.array([np.nanmean(TQ[0][k][ring]) for k in range(len(levels))])
    q_env = np.array([np.nanmean(TQ[1][k][ring]) for k in range(len(levels))])
    r_env = mixing_ratio(np.clip(q_env, 0, 0.05))
    k6 = levels.index(P_MID)
    core = pl.r <= R_CORE_DISC_KM
    a600 = (pl.r >= R_ENV600_KM[0]) & (pl.r <= R_ENV600_KM[1])
    T_core600 = float(np.nanmean(TQ[0][k6][core]))
    T600a = float(np.nanmean(TQ[0][k6][a600])); r600a = float(mixing_ratio(np.nanmean(TQ[1][k6][a600])))
    MS = pl.sample(lat, lon, np.asarray(box["msl"], float)[None])[0]
    msl_env = float(np.nanmean(MS[ring]))
    rec["msl_min"] = round(float(np.nanmin(MS[pl.r <= 150])), 1)
    vi = ventilation(rec["deep"][0], T_core600, T_env, r_env, levels, sst_c, msl_env, T600a, r600a)
    rec.update({k: (round(v_, 3 if k in ("vi", "chi", "vi_env", "chi_env") else 1) if np.isfinite(v_) else None)
                for k, v_ in vi.items()})
    rh600 = q_env[levels.index(600)] / (EPS * esat(T_env[levels.index(600)]) / (600 - esat(T_env[levels.index(600)])))
    rec["rh600"] = round(float(100 * rh600))
    if want_pv:
        try:
            pv, plat, plon = pv330(box, levels)
            rec.update(trough_metrics(pv, plat, plon, clat, clon))
        except Exception as e:                                         # noqa: BLE001
            rec["pv_err"] = str(e)[:80]
    return rec


def find_centre(box, glat, glon, r_search_km=300.0, ring=(300.0, 500.0), min_depth=1.0):
    """Surface centre for models without ECMWF's tracker (GFS, HRRR): the minimum of the lightly smoothed MSLP within
    r_search_km of the guess, accepted only as a closed low (at least min_depth hPa below the mean of the ring around
    it). -> (lat, lon, pmin) or None."""
    from scipy.ndimage import gaussian_filter
    lat, lon = box["lat"], box["lon"]
    m = np.asarray(box["msl"], float)
    if not np.isfinite(m).any():
        return None
    fill = np.where(np.isfinite(m), m, np.nanmax(m))
    sm = gaussian_filter(fill, 1.0)
    sm = np.where(np.isfinite(m), sm, np.nan)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    d = gc_km(glat, glon, LA, LO)
    cand = np.where((d <= r_search_km) & np.isfinite(sm), sm, np.inf)
    k = int(np.argmin(cand))
    if not np.isfinite(cand.flat[k]):
        return None
    la, lo, p = float(LA.flat[k]), float(LO.flat[k]), float(m.flat[k])
    dd = gc_km(la, lo, LA, LO)
    lo = float(norm_lon(lo))                                           # a box across the dateline holds e.g. -181
    rm = sm[(dd >= ring[0]) & (dd <= ring[1])]
    if np.isfinite(rm).mean() < 0.7 or np.nanmean(rm) - float(sm.flat[k]) < min_depth:
        return None
    return la, lo, p
