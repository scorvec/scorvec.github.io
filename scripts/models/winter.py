#!/usr/bin/env python3
"""Winter accumulations for models.html: FRAM ice accretion, snowfall at a column snow-to-liquid ratio and at 10:1, and
the snow-depth increase. The physics only; models.py accumulates hour by hour in its sequential pass and encodes them
like the other accumulations (16-bit, since the run started).

afram  ice accretion, FRAM (Sanders and Barjenbruch 2016, Wea. Forecasting 31, 1041-1060, doi:10.1175/WAF-D-15-0118.1):
       per period, the ice-to-liquid ratio from the freezing-rain liquid RATE P (in/h), the 2 m wet-bulb temperature Tw (C,
       values below -7 C set to -7 as in the paper) and the 10 m wind V (kt):
         ILR_P = 0.1395 P^-0.541                                        (their Eq. 2)
         ILR_T = -0.0071 Tw^3 - 0.1039 Tw^2 - 0.3904 Tw + 0.5545        (Eq. 3)
         ILR_V = 0.0014 V^2 + 0.0027 V + 0.7574                         (Eq. 4)
         Tw > -0.35 C:            ILR = 0.70 ILR_P + 0.29 ILR_T + 0.01 ILR_V    (Eq. 5)
         Tw <= -0.35 C, V > 12 kt: ILR = 0.73 ILR_P + 0.01 ILR_T + 0.26 ILR_V   (Eq. 6)
         Tw <= -0.35 C, V <= 12 kt: ILR = 0.79 ILR_P + 0.20 ILR_T + 0.01 ILR_V  (Eq. 7)
       ice on an elevated horizontal surface Ti = ILR x liquid (Eq. 8), and the RADIAL accretion shown,
       Req = 0.394 Ti (their Eq. 1, after Ryerson and Ramsay 2007) - the number for power lines and branches.
       The liquid is the model's freezing-rain accumulation (HRRR/RRFS FRZR, RDPS FreezingRain-Accum) differenced between
       consecutive published hours (1 h; RRFS 3 h past 36 h, where P is the 3-h mean rate).
       Tw from 2 m T and RH by Stull (2011, J. Appl. Meteor. Climatol. 50, 2267-2269).
acobb  snowfall with a column snow-to-liquid ratio (HRRR only: it needs hourly T, RH and vertical-velocity profiles, which
       only the HRRR's S3 files give cheaply): the Cobb and Waldstreicher (2005) method in the form the NBM v4.1 uses
       (WPC HMT seminar 2024-03-14): every 25 hPa from 925 to 300 hPa (above ground), the layer ratio SR(T) from the 2011
       Cobb-Waldstreicher table (NWS MDL, SLR.pdf), weighted by sqrt(upward motion in microbar/s, at least 1) x RH^2/6400;
       only layers saturated with respect to ice (RH_ice >= 90 %, Cobb and Waldstreicher's criterion) count; no such
       layer -> 10:1. Capped at 30 (as MDL) and floored at 3. Applied to the model's snow-phase liquid: the APCP of the
       period where the surface type is snow (rain-snow mix: half). No melting or compaction: new-snow depth, like ASNOW.
as10   the same snow-phase liquid at 10:1 (HRRR, RRFS; the RDPS's own snowfall map is already 10:1).
asnod  snow-depth increase: the sum of the POSITIVE changes of the model's snow depth (HRRR/RRFS SNOD, RDPS SnowDepth)
       between consecutive published hours; melt and settling are ignored, so it is the depth the snowfall would build.
"""

from __future__ import annotations

import numpy as np

# 2011 Cobb-Waldstreicher layer snow ratios (NWS MDL, "Snow/Liquid Ratio", SLR.pdf), layer temperature C -> ratio
LT = np.array([-300, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 1, 300], float)
LSR = np.array([7.2, 7.2, 6.8, 7.0, 8.8, 12, 18, 23, 26, 22.5, 17.5, 12, 9.5, 9, 8.5, 7, 3, 0, 0], float)
COBB_LEVS = list(range(925, 299, -25))
SLR_MIN, SLR_MAX, SLR_NONE = 3.0, 30.0, 10.0


def wetbulb_stull(t_c, rh):
    """Stull (2011): Tw (C) from T (C) and RH (%), for RH 5-99 % and T -20..50 C near sea-level pressure"""
    rh = np.clip(rh, 5.0, 100.0)
    return (
        t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def rh_from_td(t_c, td_c):
    def es(x):
        return 6.112 * np.exp(17.67 * x / (x + 243.5))

    return 100.0 * es(td_c) / es(t_c)


def fram_ilr(p_in_h, tw_c, v_kt):
    """Sanders and Barjenbruch (2016) ice-to-liquid ratio (elevated horizontal ice); see the module docstring"""
    p = np.maximum(p_in_h, 1e-4)
    t = np.maximum(tw_c, -7.0)
    ip = 0.1395 * p**-0.541
    it = -0.0071 * t**3 - 0.1039 * t**2 - 0.3904 * t + 0.5545
    iv = 0.0014 * v_kt**2 + 0.0027 * v_kt + 0.7574
    ilr = np.where(
        t > -0.35,
        0.70 * ip + 0.29 * it + 0.01 * iv,
        np.where(v_kt > 12.0, 0.73 * ip + 0.01 * it + 0.26 * iv, 0.79 * ip + 0.20 * it + 0.01 * iv),
    )
    return np.maximum(ilr, 0.0)


def fram_radial(liq_in, hours, tw_c, v_kt):
    """radial ice accretion (in) from the freezing-rain liquid of a period of `hours`"""
    ok = liq_in > 0
    out = np.zeros(liq_in.shape, np.float32)
    if ok.any():
        out[ok] = 0.394 * fram_ilr(liq_in[ok] / hours, tw_c[ok], v_kt[ok]) * liq_in[ok]
    return out


def _esw(t_c):
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def _esi(t_c):
    return 6.112 * np.exp(22.46 * t_c / (t_c + 272.62))


def cobb_slr(T_k, RH, OMEGA, levs_hpa, psfc_pa):
    """Column snow-to-liquid ratio (see the module docstring). T_k, RH (% w.r.t. water), OMEGA (Pa/s): (nlev, ...).
    -> (slr, frac of points with no qualifying layer)"""
    num = np.zeros(T_k.shape[1:], np.float64)
    den = np.zeros(T_k.shape[1:], np.float64)
    for k, p in enumerate(levs_hpa):
        t = T_k[k] - 273.15
        rh = np.clip(RH[k], 0, 100)
        rhi = np.where(t < 0, rh * _esw(t) / _esi(t), rh)
        ok = (rhi >= 90.0) & (p * 100.0 < psfc_pa)
        vv = np.maximum(-OMEGA[k] * 10.0, 1.0)  # upward motion, microbar/s, at least 1
        w = np.where(ok, np.sqrt(vv) * rh * rh / 6400.0, 0.0)
        num += w * np.interp(t, LT, LSR)
        den += w
    none = den <= 0
    slr = np.where(none, SLR_NONE, num / np.where(none, 1, den))
    return np.clip(slr, SLR_MIN, SLR_MAX).astype(np.float32), none


def kuchera(T_k, levs_hpa, psfc_pa, t2_k):
    """Kuchera-type ratio from the warmest temperature between the ground and 500 hPa (comparison only)"""
    tmax = t2_k.astype(np.float64).copy()
    for k, p in enumerate(levs_hpa):
        if p >= 500:
            tmax = np.where(p * 100.0 < psfc_pa, np.maximum(tmax, T_k[k]), tmax)
    return np.where(tmax > 271.16, 12 + 2 * (271.16 - tmax), 12 + (271.16 - tmax))
