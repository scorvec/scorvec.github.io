#!/usr/bin/env python3
"""CanSIPS on the seasonal page: every field ECCC's Datamart publishes, 12 months, plus ECCC's own
probability products. Fetching and screening come from cansips_dd.py (the single CanSIPS fetcher).

Anomaly maps (ours): 40-member mean minus CanSIPS's own 1993-2016 hindcast mean for the same start
month and lead, each component model (members 1-20, 21-40) against its own hindcast, monthly for
months 1-12 and 3-month for months 1-3 ... 10-12. A cell is shaded only where the forecast shift is
significant: Welch t-test of the 40 forecast anomalies against the hindcast's single-member spread
(effective hindcast n = 24 years, conservative), Benjamini-Hochberg FDR q = 0.10 across the cells
of the map (Wilks 2016). Fields: SST, 2 m T, 850 hPa T, precipitation, MSLP, 500 hPa height, sea-
surface height, 200 and 850 hPa wind (shading: zonal-wind anomaly; arrows: the vector anomaly where
either component is significant) -- the SFS group's wind convention on the same page.

Probability maps (ECCC's, as published, not recomputed): tercile above/below normal for 2 m T,
precipitation and snow water equivalent, and the decile tails from the ProbGT10Pct / ProbGT90Pct
products, against ECCC's 1991-2020 hindcast climatology.

Hindcast climatology fields are cached per start month (scripts/sst/data/cansips_dd/clim/, ~17 MB a
field) and kept in the GitHub release cansips-ref-v1 by the workflow, so a start month is paid for
once (~3,200 Datamart files) and is free every later year.

Output: assets/cansips/anim/cs_<product>/F<nn>.webp (frames branch) + cs_<product>_manifest.json
(main) + assets/cansips/data/cansips_maps.json.

  python cansips_maps.py [--issue YYYYMM] [--only sst,t2m] [--no-probs]

Data Source: Environment and Climate Change Canada.
"""

from __future__ import annotations

import pyproj  # noqa: F401  -- before eccodes: findlibs preloads eckit's libproj otherwise
import argparse
import calendar
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from quantmet.stats import benjamini_hochberg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cansips_dd as DDM
import mapstyle as MS

SITE = DDM.SITE_ROOT
ANIM = SITE / "assets" / "cansips" / "anim"
DOC = SITE / "assets" / "cansips" / "data" / "cansips_maps.json"
CLIMDIR = DDM.CACHE / "clim"
NL = DDM.NLEAD
# periods: 12 single months, then the 10 three-month windows ECCC also uses (P00M-P02M ... P09M-P11M)
PERIODS = [[k] for k in range(NL)] + [[k, k + 1, k + 2] for k in range(NL - 2)]
C3S_HORIZON = 6  # months the C3S systems reach
N_EFF_HC = len(DDM.CLIM_YEARS)
FDR_Q = 0.10
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# product: (vars, map kind, units, title, cmap, levels, domain mask)
FIELDS = {
    "sst": (
        ("sst",),
        "sst",
        "°C",
        "Sea-surface temperature anomaly",
        "RdBu_r",
        [-3, -2.5, -2, -1.5, -1, -0.6, -0.3, 0.3, 0.6, 1, 1.5, 2, 2.5, 3],
        "ocean",
    ),
    "t2m": (
        ("t2m",),
        "atm",
        "°C",
        "2 m temperature anomaly",
        "RdBu_r",
        [-5, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 5],
        "land",
    ),
    "t850": (
        ("t850",),
        "atm",
        "°C",
        "850 hPa temperature anomaly",
        "RdBu_r",
        [-5, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 5],
        None,
    ),
    "precip": (
        ("prate",),
        "atm",
        "mm/day",
        "Precipitation anomaly",
        "BrBG",
        [-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6],
        None,
    ),
    "mslp": (
        ("mslp",),
        "atm",
        "hPa",
        "Mean sea-level pressure anomaly",
        "RdBu_r",
        [-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6],
        None,
    ),
    "z500": (
        ("z500",),
        "atm",
        "dam",
        "500 hPa height anomaly",
        "RdBu_r",
        [-9, -6, -4.5, -3, -1.5, -0.5, 0.5, 1.5, 3, 4.5, 6, 9],
        None,
    ),
    "ssh": (
        ("ssh",),
        "sst",
        "cm",
        "Sea-surface height anomaly",
        "RdBu_r",
        [-20, -15, -10, -7.5, -5, -2.5, 2.5, 5, 7.5, 10, 15, 20],
        "ocean",
    ),
    "wind850": (
        ("u850", "v850"),
        "atm",
        "m/s",
        "850 hPa wind anomaly (shading: zonal)",
        "RdBu_r",
        [-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6],
        None,
    ),
    "wind200": (
        ("u200", "v200"),
        "atm",
        "m/s",
        "200 hPa wind anomaly (shading: zonal)",
        "RdBu_r",
        [-12, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 12],
        None,
    ),
}
VMAX_ARROW = {"wind850": 6.0, "wind200": 12.0}

# the outlook pages' tercile bins and colours (seas5_build.TERC_BINS / TERC_PALETTES; copied, not
# imported, so this module never loads the SEAS5 builder)
TERC_BINS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]
PAL = {
    "warm": ["#fde4cf", "#fbc39c", "#f59d68", "#e8703c", "#c8451c", "#8f2a0d"],
    "cool": ["#dbe9f6", "#b7d2ec", "#8ab6df", "#5c95cd", "#3672b6", "#1f4f8f"],
    "wet": ["#dcf0d6", "#b6dfad", "#89c983", "#5aae5c", "#338c3f", "#1b6229"],
    "dry": ["#f3e6cd", "#e6cd9f", "#d3af6f", "#b98d45", "#976c27", "#6b4b14"],
}
TAIL_BINS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.001]
# probability products: key -> (Datamart variable, level, title, (upper palette, lower palette), extent)
PROBS = {
    "t2m": ("AirTemp", "AGL-2m", "2 m temperature", ("warm", "cool"), None),
    "precip": ("PrecipAccum", "Sfc", "Precipitation", ("wet", "dry"), None),
    "swe": ("SnowWaterEquiv", "Sfc", "Snow water equivalent", ("cool", "dry"), [-180, 180, 20, 84]),
}
TAILS = {  # key -> (prob var, product, upper tail?, title, palette)
    "t2m_hot": ("t2m", "GT90Pct", True, "2 m temperature in its warmest 10%", "warm"),
    "t2m_cold": ("t2m", "GT10Pct", False, "2 m temperature in its coldest 10%", "cool"),
    "precip_wet": ("precip", "GT90Pct", True, "Precipitation in its wettest 10%", "wet"),
    "precip_dry": ("precip", "GT10Pct", False, "Precipitation in its driest 10%", "dry"),
}


def ym_add(ym, n):
    return DDM.add_months(ym, n)


def period_name(ym: str, idx: list[int]) -> str:
    a, b = ym_add(ym, idx[0]), ym_add(ym, idx[-1])
    if len(idx) == 1:
        return f"{MONTHS[int(a[4:]) - 1]} {a[:4]}"
    lets = "".join(MONTHS[int(ym_add(ym, k)[4:]) - 1][0] for k in idx)
    return f"{lets} {a[:4]}" + (f"–{b[2:4]}" if b[:4] != a[:4] else "")


def lead_text(idx: list[int]) -> str:
    return f"month {idx[0] + 1}" if len(idx) == 1 else f"months {idx[0] + 1}–{idx[-1] + 1}"


def beyond_c3s(idx: list[int]) -> bool:
    return idx[-1] >= C3S_HORIZON


# ── hindcast climatology fields ──────────────────────────────────────────────
def clim_path(var: str, month: int) -> Path:
    return CLIMDIR / f"cansips_clim_{var}_{month:02d}.npz"


def unit_factor(var: str, units: str) -> tuple[float, float]:
    """(scale, offset) taking file units to the map units; refuses an unexpected unit."""
    u = units.replace(" ", "")
    if var in ("sst", "t2m", "t850"):
        if u == "K":
            return 1.0, -273.15
    elif var == "prate":
        if u in ("kgm**-2s**-1", "kgm-2s-1"):
            return 86400.0, 0.0
        if u in ("ms**-1", "m/s", "ms-1"):
            return 86400.0 * 1000.0, 0.0
    elif var == "z500":
        if u == "gpm":
            return 0.1, 0.0
        if u in ("m**2s**-2", "m2s-2"):
            return 0.1 / 9.80665, 0.0
    elif var == "mslp":
        if u == "Pa":
            return 0.01, 0.0
    elif var == "ssh":
        # eccodes has no name for ECCC's SeaSfcHeight-Geoid parameter and reports 'unknown'; read_members has already
        # screened every member mean into -5..5, which only metres satisfy (cm would be +-500)
        if u in ("m", "unknown"):
            return 100.0, 0.0
    elif var in ("u200", "v200", "u850", "v850"):  # noqa: SIM102  one branch per variable, parallel to the others
        if u in ("ms**-1", "m/s", "ms-1"):
            return 1.0, 0.0
    raise DDM.DataError(f"{var}: unexpected units '{units}'")


def period_mean(X: np.ndarray, ym: str, idx: list[int], days: bool) -> np.ndarray:
    """Mean of X (lead, ...) over the lead months idx of issue ym (lead 0 = the issue month). `days`: each month weighted
    by its length (a time mean; a leap February counts 29 days); else the equal-weight mean that the hindcast
    climatologies written without the `weighting` flag were built with, so forecast and climatology always match."""
    if not days or len(idx) == 1:
        return X[idx].mean(0)
    y, m = int(ym[:4]), int(ym[4:6])
    w = np.array([calendar.monthrange(y + (m - 1 + k) // 12, (m - 1 + k) % 12 + 1)[1] for k in idx], np.float32)
    return np.tensordot(w / w.sum(), X[idx], axes=1)


def build_clim(var: str, month: int) -> dict:
    """Per component (2), per period (22): mean and single-member SD of the hindcast, map units. `days`: the
    multi-month periods are day-weighted means (files written since the weighting carry `weighting`)."""
    p = clim_path(var, month)
    if p.exists():
        z = np.load(p)
        return {
            "mean": z["mean"].astype(np.float32),
            "sd": z["sd"].astype(np.float32),
            "units": str(z["units"]),
            "days": "weighting" in z.files,
        }
    t0 = time.time()
    jobs = [
        (DDM.hc_dir(f"{y}{month:02d}") + DDM.hc_name(var, f"{y}{month:02d}", L), DDM.hc_path(var, f"{y}{month:02d}", L))
        for y in DDM.CLIM_YEARS
        for L in range(NL)
    ]
    paths = DDM.fetch_many(jobs, workers=4)
    if any(q is None for q in paths):
        raise DDM.DataError(f"{var}: hindcast start month {month:02d} incomplete on the Datamart")
    nP, comps = len(PERIODS), list(DDM.COMPONENTS.values())
    S1 = np.zeros((2, nP, 180, 360))
    S2 = np.zeros((2, nP, 180, 360))
    units = None
    fac = None
    for y in DDM.CLIM_YEARS:
        ym = f"{y}{month:02d}"
        leads = []
        for L in range(NL):
            arr, meta = DDM.read_members(DDM.hc_path(var, ym, L), var, ym, L)
            if fac is None:
                units = meta["units"]
                fac = unit_factor(var, units)
            leads.append(arr * np.float32(fac[0]) + np.float32(fac[1]))
        X = np.stack(leads)  # (12, 40, 180, 360)
        for pi, idx in enumerate(PERIODS):
            x = period_mean(X, ym, idx, True)  # (40, 180, 360)
            for ci, sl in enumerate(comps):
                xs = x[sl].astype(np.float64)
                S1[ci, pi] += xs.sum(0)
                S2[ci, pi] += (xs * xs).sum(0)
        del X, leads
    n = 20 * len(DDM.CLIM_YEARS)
    mean = S1 / n
    sd = np.sqrt(np.maximum(S2 / n - mean * mean, 0.0) * n / (n - 1))
    if var == "prate":
        g = float(
            (mean[:, 0] * np.cos(np.deg2rad(DDM.LAT))[None, :, None]).sum()
            / (np.cos(np.deg2rad(DDM.LAT)).sum() * 360 * 2)
        )
        if not 2.0 <= g <= 4.5:
            raise DDM.DataError(f"prate: hindcast global mean {g:.2f} mm/day is not plausible (units {units})")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p, mean=mean.astype(np.float32), sd=sd.astype(np.float16), units=np.array(units), weighting=np.array("days")
    )
    for q in paths:  # free the disk field by field
        with contextlib.suppress(OSError):
            q.unlink()
    print(f"  clim {var} {month:02d}: {len(jobs)} files, {time.time() - t0:.0f} s", flush=True)
    return build_clim(var, month)


# ── forecast anomalies + significance ───────────────────────────────────────
def forecast_members(var: str, ym: str) -> tuple[np.ndarray, str]:
    jobs = [(DDM.fc_dir(ym) + DDM.fc_name(var, ym, L), DDM.fc_path(var, ym, L)) for L in range(NL)]
    paths = DDM.fetch_many(jobs, workers=6)
    if any(q is None for q in paths):
        raise DDM.DataError(f"{var}: forecast {ym} incomplete on the Datamart")
    out, units = [], None
    for L in range(NL):
        arr, meta = DDM.read_members(DDM.fc_path(var, ym, L), var, ym, L)
        units = meta["units"]
        f = unit_factor(var, units)
        out.append(arr * np.float32(f[0]) + np.float32(f[1]))
    return np.stack(out), units  # (12, 40, 180, 360)


def bh_mask(p: np.ndarray, domain: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg over the finite p-values inside `domain` (quantmet.stats; was a local copy)."""
    return benjamini_hochberg(np.where(domain, p, np.nan), q)


def anomaly_and_sig(X: np.ndarray, clim: dict, pi: int, domain: np.ndarray, ym: str):
    from scipy import stats

    x = period_mean(X, ym, PERIODS[pi], clim.get("days", False))  # (40, 180, 360), as the climatology was built
    a = np.empty_like(x)
    for ci, sl in enumerate(DDM.COMPONENTS.values()):
        a[sl] = x[sl] - clim["mean"][ci, pi]
    A = a.mean(0)
    sf2 = a.var(0, ddof=1) / a.shape[0]
    sh2 = (clim["sd"][:, pi].astype(np.float64) ** 2).mean(0) / N_EFF_HC
    se2 = sf2 + sh2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = A / np.sqrt(se2)
        df = se2**2 / (sf2**2 / (a.shape[0] - 1) + sh2**2 / (N_EFF_HC - 1))
    p = 2 * stats.t.sf(np.abs(t), np.where(np.isfinite(df), df, 1.0))
    p = np.where(se2 > 0, p, np.nan)
    return A, bh_mask(p, domain)


# ── drawing ──────────────────────────────────────────────────────────────────
LON_E = np.arange(0, 361, 1.0)
LAT_E = np.arange(-90, 91, 1.0)


def _heading_text(ym, idx):
    t = f"{period_name(ym, idx)} ({lead_text(idx)})"
    return t + (" · CanSIPS only" if beyond_c3s(idx) else "")


def _beyond_note(idx):
    return (
        "Months 7–12: CanSIPS alone, past the six months the C3S systems reach, so one model's forecast, not a consensus. "
        if beyond_c3s(idx)
        else ""
    )


def draw_field(key, ym, idx, A, sig, comps, out: Path, nsig: float):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    vs, kind, units, title, cmap, levels, dom = FIELDS[key]
    pc_kind = "sst" if kind == "sst" else "atm"
    fig, ax, H, pc = MS.open_map(kind=pc_kind)
    shade = np.where(sig, A[0] if len(vs) == 2 else A, np.nan)
    sst_sty = MS.sst_anom_style(5.0) if key == "sst" else None  # the site's shared SST palette and ladder
    if sst_sty is not None:
        levels = sst_sty["levels"]
        cm = sst_sty["cmap"]
        nrm = sst_sty["norm"]
    else:
        cm = plt.get_cmap(cmap, len(levels) - 1)
        nrm = BoundaryNorm(levels, len(levels) - 1)
    pm = ax.pcolormesh(
        LON_E,
        LAT_E,
        np.ma.masked_invalid(shade),
        cmap=cm,
        norm=nrm,
        transform=pc,
        shading="flat",
        rasterized=True,
        zorder=1,
    )
    if len(vs) == 2:
        st = 8
        u, v = A
        keep = sig if comps is None else (comps[0] | comps[1])
        uu = np.where(keep, u, np.nan)[st // 2 :: st, st // 2 :: st]
        vv = np.where(keep, v, np.nan)[st // 2 :: st, st // 2 :: st]
        LO, LA = np.meshgrid(DDM.LON[st // 2 :: st], DDM.LAT[st // 2 :: st])
        ax.quiver(
            LO, LA, uu, vv, transform=pc, color="k", scale=VMAX_ARROW[key] * 30, width=0.0016, alpha=0.75, zorder=3
        )
    MS.features(
        ax, land_only=(dom == "land"), states=(kind != "sst"), land_color=MS.SST_LAND if key == "sst" else "#f1f0eb"
    )  # SST: land darker than the white 'no shift'
    MS.heading(
        fig,
        H,
        f"CanSIPS {title} · {_heading_text(ym, idx)} · {MONTHS[int(ym[4:]) - 1]} {ym[:4]} issue",
        _beyond_note(idx)
        + "Mean of 40 members (GEM5.2-NEMO + CanESM5) minus CanSIPS's own 1993–2016 hindcast for the same start month and "
        "lead, each model against its own. Shaded only where the shift is significant (Welch t-test, FDR q = 0.10; "
        f"{nsig:.0%} of the map); blank = no significant shift. Data Source: Environment and Climate Change Canada.",
        wrap=205,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if sst_sty is not None:
        cb = MS.colorbar(fig, H, pm, f"anomaly ({units})")
        MS.sst_colorbar(cb, sst_sty, f"anomaly ({units})", size=8.5, tick_size=7)
        MS.save_webp_palette(fig, out, dpi=110)  # flat 1-degree classes: lossless, sharp
        return
    MS.colorbar(fig, H, pm, f"anomaly ({units})", levels)
    MS.save(fig, out, dpi=100, quality=78)


def draw_terciles(key, ym, idx, above, below, normal, out: Path):
    from matplotlib.colors import ListedColormap, BoundaryNorm

    _var, _lev, title, (pu, pl), extent = PROBS[key]
    fig, ax, H, pc = MS.open_map(kind="atm", extent=extent, central=-100.0 if extent else None)
    for arr, other, cols in ((above, below, PAL[pu]), (below, above, PAL[pl])):
        show = np.where((arr >= TERC_BINS[0]) & (arr >= np.fmax(other, normal)), arr, np.nan)
        ax.pcolormesh(
            LON_E,
            LAT_E,
            np.ma.masked_invalid(show),
            cmap=ListedColormap(cols),
            norm=BoundaryNorm(TERC_BINS, len(cols)),
            transform=pc,
            shading="flat",
            zorder=1,
        )
    MS.features(ax, land_only=True)
    labs = {
        "t2m": ("Above normal most likely", "Below normal most likely"),
        "precip": ("Wetter than normal most likely", "Drier than normal most likely"),
        "swe": ("More snow than normal most likely", "Less snow than normal most likely"),
    }[key]
    MS.tercile_legends(fig, PAL[pu], PAL[pl], *labs, TERC_BINS)
    MS.heading(
        fig,
        H,
        f"{title}: most likely tercile — CanSIPS · {_heading_text(ym, idx)} · {MONTHS[int(ym[4:]) - 1]} {ym[:4]} issue",
        _beyond_note(idx)
        + "ECCC's own tercile probabilities as published (not recomputed here), against ECCC's 1991–2020 hindcast "
        "climatology. White: no category reaches 40 % or near-normal is most likely; near-normal is not drawn."
        + (" No snow on the ground in the hindcast leaves a cell empty." if key == "swe" else "")
        + " Data Source: Environment and Climate Change Canada.",
        wrap=190,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    MS.save(fig, out, dpi=100, quality=78)


def draw_tail(key, ym, idx, prob, out: Path):
    from matplotlib.colors import ListedColormap, BoundaryNorm

    _pv, prod, _upper, title, pal = TAILS[key]
    fig, ax, H, pc = MS.open_map(kind="atm")
    show = np.where(prob >= TAIL_BINS[0], prob, np.nan)
    pm = ax.pcolormesh(
        LON_E,
        LAT_E,
        np.ma.masked_invalid(show),
        cmap=ListedColormap(PAL[pal]),
        norm=BoundaryNorm(TAIL_BINS, len(PAL[pal])),
        transform=pc,
        shading="flat",
        zorder=1,
    )
    MS.features(ax, land_only=True)
    MS.heading(
        fig,
        H,
        f"Chance of {title[0].lower() + title[1:]} — CanSIPS · {_heading_text(ym, idx)} · {MONTHS[int(ym[4:]) - 1]} {ym[:4]} issue",
        _beyond_note(idx)
        + f"ECCC's own probability as published (from its Prob{prod} product, not recomputed here), against ECCC's "
        "1991–2020 hindcast climatology. The climatological chance is 10 %; white = under 20 %, twice that. "
        "Data Source: Environment and Climate Change Canada.",
        wrap=190,
    )
    cb = MS.colorbar(fig, H, pm, "probability", None, extend="neither")
    cb.set_ticks([*TAIL_BINS[:-1], 1.0])
    cb.set_ticklabels([f"{int(b * 100)}%" for b in TAIL_BINS[:-1]] + ["100%"])
    out.parent.mkdir(parents=True, exist_ok=True)
    MS.save(fig, out, dpi=100, quality=78)


# ── ECCC probability products ────────────────────────────────────────────────
def prob_name(ym, var, lev, prod, idx):
    P = f"P{idx[0]:02d}M" if len(idx) == 1 else f"P{idx[0]:02d}M-P{idx[-1]:02d}M"
    return f"{ym}_MSC_CanSIPS_{var}-Prob{prod}_{lev}_LatLon1.0_{P}.grib2"


def read_prob(path: Path, ym: str) -> np.ndarray:
    """One probability field in 0..1 (NaN where ECCC leaves it missing); screened."""
    import eccodes

    vals = []
    with open(path, "rb") as fh:
        while True:
            h = eccodes.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                if (eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")) != (360, 180) or abs(
                    eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees") + 89.5
                ) > 1e-6:
                    raise DDM.DataError(f"{path.name}: unexpected grid")
                if str(eccodes.codes_get(h, "dataDate"))[:6] != ym:
                    raise DDM.DataError(f"{path.name}: dataDate {eccodes.codes_get(h, 'dataDate')} for issue {ym}")
                v = eccodes.codes_get_values(h).astype(np.float64)
                if eccodes.codes_get(h, "bitmapPresent"):
                    v[v == eccodes.codes_get(h, "missingValue")] = np.nan
                vals.append(v.reshape(180, 360))
            finally:
                eccodes.codes_release(h)
    if len(vals) != 1:
        raise DDM.DataError(f"{path.name}: {len(vals)} messages, expected 1")
    v = vals[0]
    # ECCC publishes these in percent (Datamart readme: 0 % to 100 %); a field whose maximum stays under 1.5 would be
    # a change of convention, so refuse it rather than guess. GRIB packing leaves values a hair past 100.
    mx = np.nanmax(v) if np.isfinite(v).any() else 0
    if mx <= 1.5:
        raise DDM.DataError(f"{path.name}: maximum {mx:.3g} - not the documented percent scale")
    v = v / 100.0
    if np.nanmin(v) < -1e-3 or np.nanmax(v) > 1 + 1e-3:
        raise DDM.DataError(
            f"{path.name}: probabilities outside 0..100 % ({100 * np.nanmin(v):.4g}..{100 * np.nanmax(v):.4g})"
        )
    return np.clip(v, 0.0, 1.0)


def prob_fields(ym, key, prods, idx):
    var, lev = PROBS[key][0], PROBS[key][1]
    jobs = [
        (DDM.fc_dir(ym) + prob_name(ym, var, lev, p, idx), DDM.CACHE / "prob" / ym / prob_name(ym, var, lev, p, idx))
        for p in prods
    ]
    paths = DDM.fetch_many(jobs, workers=4)
    if any(q is None for q in paths):
        return None
    return [read_prob(q, ym) for q in paths]


# ── manifests ────────────────────────────────────────────────────────────────
def write_loop(name: str, label: str, frames: list):
    ANIM.mkdir(parents=True, exist_ok=True)
    (ANIM / f"{name}_manifest.json").write_text(
        json.dumps(
            {
                "ver": int(time.time()),
                "days": len(frames),
                "regions": {name: {"label": label, "n_frames": len(frames), "frames": frames}},
            },
            separators=(",", ":"),
        )
    )


def frame_entry(ym, k, idx, fn):
    vm = ym_add(ym, idx[len(idx) // 2])
    return {
        "idx": k,
        "file": fn,
        "date": f"{vm[:4]}-{vm[4:]}",
        "label": f"{period_name(ym, idx)} · {lead_text(idx)}" + (" · CanSIPS only" if beyond_c3s(idx) else ""),
    }


def dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.glob("*.webp")) if d.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue")
    ap.add_argument("--only", default="")
    ap.add_argument("--no-probs", action="store_true")
    ap.add_argument("--clim-only", action="store_true")
    ap.add_argument(
        "--if-stale", action="store_true", help="exit at once when the published maps are this issue's, complete"
    )
    ap.add_argument("--stale-check", action="store_true", help="print stale=true|false for GITHUB_OUTPUT and exit")
    a = ap.parse_args()
    if a.stale_check:
        ym = a.issue or DDM.latest_issue()
        d0 = json.loads(DOC.read_text()) if DOC.exists() else {}
        fresh = (
            bool(ym)
            and d0.get("issue") == f"{ym[:4]}-{ym[4:]}"
            and not d0.get("failed")
            and d0.get("posted") == {str(k): v for k, v in (DDM.issue_posted(ym) or {}).items()}
        )
        print(f"stale={'false' if fresh else 'true'}")
        return 0
    ym = a.issue or DDM.latest_issue()
    if not ym:
        print("::warning::no complete CanSIPS issue on the Datamart; keeping the published maps")
        return 0
    if a.if_stale and DOC.exists():
        d0 = json.loads(DOC.read_text())
        if (
            d0.get("issue") == f"{ym[:4]}-{ym[4:]}"
            and not d0.get("failed")
            and d0.get("posted") == {str(k): v for k, v in (DDM.issue_posted(ym) or {}).items()}
        ):
            print(f"maps for {ym} are published and complete; nothing to do")
            return 0
    month = int(ym[4:])
    want = [k for k in FIELDS if not a.only or k in a.only.split(",")]
    ocean = DDM.ocean_mask()
    domains = {"ocean": ocean, "land": ~ocean, None: np.ones_like(ocean)}
    doc = json.loads(DOC.read_text()) if DOC.exists() else {}
    doc["products"] = {}  # this issue's loops only
    doc.update(
        {
            "issue": f"{ym[:4]}-{ym[4:]}",
            "posted": {str(k): v for k, v in (DDM.issue_posted(ym) or {}).items()},
            "attribution": DDM.ATTRIBUTION,
            "licence": DDM.LICENCE,
        }
    )
    doc.setdefault("products", {})
    failures = []
    for key in want:
        vs, _kind, _units, title, _cmap, _levels, dom = FIELDS[key]
        try:
            clims = [build_clim(v, month) for v in vs]
            if a.clim_only:
                continue
            Xs = [forecast_members(v, ym)[0] for v in vs]
        except (DDM.DataError, RuntimeError) as e:
            print(f"::warning::{key}: {e}")
            failures.append(key)
            continue
        for suffix, sel in (("m", range(NL)), ("s", range(NL, len(PERIODS)))):
            name = f"cs_{key}_{suffix}"
            d = ANIM / name
            shutil.rmtree(d, ignore_errors=True)
            frames, fracs = [], []
            for k, pi in enumerate(sel):
                idx = PERIODS[pi]
                res = [anomaly_and_sig(X, c, pi, domains[dom], ym) for X, c in zip(Xs, clims)]
                if len(vs) == 2:
                    A = (res[0][0], res[1][0])
                    sig = res[0][1]
                    comps = (res[0][1], res[1][1])
                else:
                    A, sig = res[0]
                    comps = None
                frac = float(sig[domains[dom]].mean())
                fn = f"F{k:02d}.webp"
                draw_field(key, ym, idx, A, sig, comps, d / fn, frac)
                frames.append(frame_entry(ym, k, idx, fn))
                fracs.append(round(frac, 3))
            write_loop(name, f"CanSIPS · {title} · {'monthly' if suffix == 'm' else '3-month'}", frames)
            doc["products"][name] = {"frames": len(frames), "bytes": dir_bytes(d), "sig_frac": fracs}
            print(f"loop {name}: {len(frames)} frames, {dir_bytes(d) // 1024} KB, significant {fracs}", flush=True)
        del Xs
    if not a.no_probs and not a.clim_only:
        for key in PROBS:
            for suffix, sel in (("m", range(NL)), ("s", range(NL, len(PERIODS)))):
                name = f"cs_terc_{key}_{suffix}"
                d = ANIM / name
                shutil.rmtree(d, ignore_errors=True)
                frames = []
                for k, pi in enumerate(sel):
                    idx = PERIODS[pi]
                    try:
                        f3 = prob_fields(ym, key, ["AboveNormal", "BelowNormal", "NearNormal"], idx)
                    except (DDM.DataError, RuntimeError) as e:
                        print(f"::warning::{name} {idx}: {e}")
                        f3 = None
                    if f3 is None:
                        continue
                    ab, be, nn = f3
                    s = ab + be + nn
                    # where the hindcast has no snow (or no defined terciles) ECCC writes 0 for all three: no forecast there
                    none = np.isfinite(s) & (s < 0.02)
                    ab, be, nn = (np.where(none, np.nan, f) for f in (ab, be, nn))
                    ok = np.isfinite(s) & ~none
                    bad = float(np.mean(np.abs(s[ok] - 1) > 0.03)) if ok.any() else 1.0
                    if bad > 0.01:
                        print(
                            f"::warning::{name} {idx}: terciles do not sum to 1 on {bad:.1%} of the defined cells; skipped"
                        )
                        continue
                    if key == "swe" and ok.mean() < 0.01:
                        print(f"  {name} {idx}: snow terciles defined on {ok.mean():.2%} of the globe; frame skipped")
                        continue
                    fn = f"F{k:02d}.webp"
                    draw_terciles(key, ym, idx, ab, be, nn, d / fn)
                    frames.append(frame_entry(ym, k, idx, fn))
                if frames:
                    write_loop(
                        name,
                        f"CanSIPS (ECCC probabilities) · {PROBS[key][2]} terciles · {'monthly' if suffix == 'm' else '3-month'}",
                        frames,
                    )
                    doc["products"][name] = {"frames": len(frames), "bytes": dir_bytes(d)}
                    print(f"loop {name}: {len(frames)} frames, {dir_bytes(d) // 1024} KB", flush=True)
                else:
                    failures.append(name)
        for key, (pv, prod, upper, title, _pal) in TAILS.items():
            for suffix, sel in (("m", range(NL)), ("s", range(NL, len(PERIODS)))):
                name = f"cs_tail_{key}_{suffix}"
                d = ANIM / name
                shutil.rmtree(d, ignore_errors=True)
                frames = []
                for k, pi in enumerate(sel):
                    idx = PERIODS[pi]
                    try:
                        f1 = prob_fields(ym, pv, [prod], idx)
                    except (DDM.DataError, RuntimeError) as e:
                        print(f"::warning::{name} {idx}: {e}")
                        f1 = None
                    if f1 is None:
                        continue
                    pr = f1[0] if upper else 1.0 - f1[0]
                    fn = f"F{k:02d}.webp"
                    draw_tail(key, ym, idx, pr, d / fn)
                    frames.append(frame_entry(ym, k, idx, fn))
                if frames:
                    write_loop(
                        name,
                        f"CanSIPS (ECCC probabilities) · {title} · {'monthly' if suffix == 'm' else '3-month'}",
                        frames,
                    )
                    doc["products"][name] = {"frames": len(frames), "bytes": dir_bytes(d)}
                    print(f"loop {name}: {len(frames)} frames, {dir_bytes(d) // 1024} KB", flush=True)
                else:
                    failures.append(name)
    if a.clim_only:
        return 0
    tot = sum(v.get("bytes", 0) for v in doc["products"].values())
    doc.update(
        {
            "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "total_bytes": tot,
            "failed": failures,
            "hindcast": "1993-2016, 40 members x 24 years, each component model against its own",
            "significance": "Welch t-test, forecast members vs hindcast single-member spread (n_eff 24), BH FDR q=0.10",
        }
    )
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    print(
        f"wrote {DOC.relative_to(SITE)}; {len(doc['products'])} loops, {tot / 1e6:.1f} MB of frames; failed: {failures or 'none'}"
    )
    return 1 if len(failures) > len(doc["products"]) // 2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
