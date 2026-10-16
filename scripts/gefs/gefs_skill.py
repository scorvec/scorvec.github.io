#!/usr/bin/env python3
"""GEFS weekly-map skill mask, and the date / re-basing bookkeeping the GEFS tercile maps share with it (2026-09-26;
port of the GEPS page's render_maps.skill_mask / hatch_skill and hindcast_maps.py).

What the mask is. For every field (t2m, pr, zg500, mslp), calendar month of the INIT (reforecast starts within +-45
days of the 15th, as GEPS pools them) and week 1-5, the gridpoint anomaly correlation r of the GEFSv12 reforecast's
4-member ensemble-mean weekly anomaly (2000-2019 Wednesday starts) with the observed weekly anomaly on the same valid
days - the truths the GEPS page uses: NCEP/NCAR R1 2 m air, 500 hPa height and sea-level pressure, CPC unified gauge
precipitation (land only). Built once, offline, by build_gefs_mapskill.py; see its docstring for the arithmetic
(leave-season-out model climatology, re-basing, conservative 1.5 deg remap of the gauge analysis).
A cell is skilful only if r passes a one-sided t-test of r > 0 at ALPHA on N_EFF - 2 degrees of freedom AND survives a
Benjamini-Hochberg false-discovery control at FDR_Q over its map (noskill). N_EFF is the number of reforecast YEARS,
not the ~260 pairs behind r: the starts of one season are a week apart and serially correlated (and the GEPS
hindcast's are 1-3 days apart, so its weekly windows overlap), and a season can be no more dependent than one sample a
year, so the count of years bounds the effective sample size without an estimate of that correlation (one-sided 5 %
at 20 years: r >= 0.378).

Reference files (release gefs-ref-v1, downloaded by the daily job into one flat directory `ref_dir`):
  gefs_mapskill_{tag}.npz   r (12, 5, 121, 240) float16, NaN = no verification data (precipitation over the ocean);
                            t2m and pr also a, s (regression slope and residual sd, for the calibrated terciles)
  gefs_terciles_{tag}.npz   t2m, pr: 2016-2025 weekly-anomaly median / terciles at the 73 5-day centres (gefs_probs.py)
  era5_shift.npz            the maps' 2001-2020 -> 1991-2020 re-basing

Dates (the GEFS convention, NOT the GEPS page's base + L labelling): day d is the UTC calendar day init + (d - 1), so
week k covers init + 7k - 7 ... init + 7k - 1 and week 1 starts ON the init date. The skill table is picked by the
calendar month of the init date.

Use from gefs_live.py's weekly figure (the `_masked` variants):

    import gefs_skill as SK
    m = SK.masks(tag, base, ref_dir)                  # None if the tag has no skill file (not one of MASK_TAGS)
    ...
    if m is not None:
        SK.hatch(ax, R.LAT, R.LON, m[0][i], m[1][i], ccrs.PlateCarree())   # i = week index 0-4
    note: SK.MASK_NOTE[tag]

numpy, scipy and quantmet (the significance test), matplotlib for hatch.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from gefs_reforecast import noleap_doy  # the climatologies' day-of-year key (numpy only)

ALPHA = 0.05  # one-sided level of the gridpoint test of positive skill
FDR_Q = 0.10  # Benjamini-Hochberg false discovery rate over each map (the site's map standard, as on the CanSIPS maps)
N_EFF = 20  # effective sample size of a gridpoint r: the 2000-2019 reforecast years (see the docstring)
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
CENTRES = np.arange(1, 366, 5)  # the 73 5-day day-of-year centres of the climatologies
MASK_TAGS = ("t2m", "pr", "zg500", "mslp")
LAT = np.arange(90.0, -90.0 - 1e-6, -1.5)  # gefs_reforecast.LAT (121, north first)
LON = np.arange(0.0, 360.0 - 1e-6, 1.5)  # gefs_reforecast.LON (240)


def test_note(n_eff: int = N_EFF, what: str = "yrs") -> str:
    """The figures' wording of the noskill test."""
    return f"one-sided {ALPHA:.0%}, {n_eff} {what}, FDR {FDR_Q:.2f}"


# the footnote the GEPS weekly_masked figures carry (render_maps.weekly), GEFS facts
MASK_NOTE = {
    t: (
        f"hatched: no significant hindcast skill for this week and season ({test_note()})"
        + ("; cross-hatched: no gauge truth (ocean)" if t == "pr" else "")
    )
    for t in MASK_TAGS
}


# ── dates ────────────────────────────────────────────────────────────────────────────────────────────────────────
def as_date(d) -> dt.date:
    """date, datetime, numpy datetime64 or 'YYYYMMDD' / 'YYYY-MM-DD' -> datetime.date."""
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    s = str(d)[:10].replace("-", "")
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def week_dates(init, k: int) -> list[dt.date]:
    """The seven valid dates of week k (1-5): init + 7k-7 ... init + 7k-1."""
    b = as_date(init)
    return [b + dt.timedelta(days=d) for d in range(7 * k - 7, 7 * k)]


def bracket(doy: float):
    """(i_lo, w_lo, i_hi, w_hi): linear weights between the two 5-day centres around the day-of-year key `doy`
    (gefs_reforecast.noleap_doy, 1-365), indices into CENTRES - exactly gefs_live.clim_at's interpolation (keys
    362-365 blend centre 361 into centre 1)."""
    lo = int(CENTRES[CENTRES <= doy].max())  # noqa: SIM300  array of window centres, not a constant
    hi = lo + 5 if lo + 5 <= 361 else 1
    w = (doy - lo) / 5.0
    return (lo - 1) // 5, 1.0 - w, (hi - 1) // 5, w


def at_doy(arr, doy: float):
    """arr (73, ...) at the 5-day centres -> the value at day-of-year key `doy`, linear between centres."""
    i, wi, j, wj = bracket(doy)
    return wi * arr[i] + wj * arr[j]


def weekly_shift(s73, init) -> np.ndarray:
    """The re-basing term per week: the 7-day mean over each week's VALID dates of the ERA5 1991-2020 minus
    2001-2020 normal difference (era5_shift.npz s15_<tag>, 73 centres). Anomaly vs 1991-2020 =
    [F - M_GEFS(init doy, week)] - weekly_shift. -> (5, lat, lon) float64."""
    out = np.zeros((len(WEEKS), *s73.shape[1:]))
    for k in range(1, len(WEEKS) + 1):
        for d in week_dates(init, k):
            out[k - 1] += at_doy(s73, noleap_doy(d))
        out[k - 1] /= 7.0
    return out


# ── skill tables ─────────────────────────────────────────────────────────────────────────────────────────────────
def noskill(r, n_eff: int = N_EFF) -> np.ndarray:
    """True where a finite r is NOT significant skill: the one-sided p of r > 0 (t on n_eff - 2 degrees of freedom)
    above ALPHA, or the cell not a Benjamini-Hochberg discovery at FDR_Q among the finite cells of its map. A map is
    the last two axes (a 1-D input is one map); NaN (no truth) is never flagged here."""
    from quantmet.stats import benjamini_hochberg
    from scipy.stats import t as tdist

    r = np.asarray(r, dtype=float)
    df = n_eff - 2
    ok = np.isfinite(r)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(ok, tdist.sf(r * np.sqrt(df / np.clip(1.0 - r * r, 1e-12, None)), df), np.nan)
    maps = p.reshape(-1, *p.shape[-2:]) if p.ndim >= 2 else p[None]
    disc = np.stack([benjamini_hochberg(m, FDR_Q) for m in maps]).reshape(p.shape)
    return ok & ~(disc & (p <= ALPHA))


_SK: dict = {}


def load(tag: str, ref_dir) -> dict | None:
    """The skill file as a dict of float32 arrays (month, week, lat, lon); None if absent."""
    key = (tag, str(ref_dir))
    if key not in _SK:
        p = Path(ref_dir) / f"gefs_mapskill_{tag}.npz"
        if not p.exists():
            _SK[key] = None
        else:
            z = np.load(p)
            _SK[key] = {k: (z[k].astype("float32") if z[k].dtype.kind == "f" else z[k]) for k in z.files}
    return _SK[key]


def skill(tag: str, init, ref_dir) -> np.ndarray | None:
    """Anomaly correlation r (5, 121, 240) for the init's calendar month, weeks 1-5; NaN = no truth."""
    z = load(tag, ref_dir)
    if z is None:
        return None
    return z["r"][as_date(init).month - 1]


def mask(tag: str, init, week: int, ref_dir) -> np.ndarray | None:
    """(121, 240) bool on the 1.5 deg grid (gefs_reforecast.LAT 90 -> -90, LON 0 -> 358.5): True where the hindcast
    anomaly correlation for this init month and week (1-5) is not significant skill (noskill) - the cells hatched ////."""
    r = skill(tag, init, ref_dir)
    if r is None:
        return None
    return noskill(r[week - 1])


def notruth(tag: str, init, week: int, ref_dir) -> np.ndarray | None:
    """(121, 240) bool: True where there is no verification data at all (precipitation over the ocean) - the cells
    GEPS cross-hatches xxxx."""
    r = skill(tag, init, ref_dir)
    if r is None:
        return None
    return ~np.isfinite(r[week - 1])


def masks(tag: str, init, ref_dir):
    """(noskill, notruth), each (5, 121, 240) bool, for weeks 1-5 of this init; None if the tag was never scored."""
    r = skill(tag, init, ref_dir)
    if r is None:
        return None
    return noskill(r), ~np.isfinite(r)


def hatch(ax, lat, lon, mask_skill, mask_notruth, transform, zorder=None):
    """Draw GEPS's skill hatching on one map panel (render_maps.hatch_skill): //// where mask_skill, xxxx where
    mask_notruth, black at matplotlib's default hatch width, no fill. lat/lon are the mask's grid (1-D, lon 0-360
    ascending; the 0E column is repeated at 360 so the hatching closes across the seam). Either mask may be None."""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    lon_w = np.r_[lon, lon[0] + 360.0]
    kw = {} if zorder is None else {"zorder": zorder}
    for fld, pat in ((mask_skill, "////"), (mask_notruth, "xxxx")):
        if fld is None or not np.any(fld):
            continue
        f = np.concatenate([fld, fld[:, :1]], axis=1).astype(float)
        cs = ax.contourf(lon_w, lat, f, levels=[0.5, 1.5], colors="none", hatches=[pat], transform=transform, **kw)
        # pin the hatch colour (GEPS draws matplotlib's default black) and no outline, whatever the rcParams
        cs.set_edgecolor("black")
        cs.set_linewidth(0.0)
