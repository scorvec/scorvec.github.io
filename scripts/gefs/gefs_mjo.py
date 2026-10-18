#!/usr/bin/env python3
"""MJO diagnostics from the daily GEFS extended run - the GEPS page's "Tropics" group for the GEFS page. Port of
scripts/geps/mjo_geps.py and mjo_hindcast.py.

Three figures, in the GEPS figures' format:
  gefs_mjo_phase.webp      Wheeler & Hendon (2004) RMM phase diagram: the observed track of the last 40 days, every
                           member (the plume) and the ensemble-mean track, days 1-35
  gefs_mjo_hovmoller.webp  15S-15N OLR and 850 hPa zonal-wind anomaly, lead x longitude (ensemble mean)
  gefs_mjo_skill.webp      RMM skill of GEFS from its own reforecast (2000-2019), redrawn every run for the "now
                           +-45 d" curve, plus data/gefs_mjo_skill.json (the numbers for the page's table)

The RMM is the FULL three-field projection (OLR + U850 + U200) onto the site's WH04 reference EOFs
(scripts/mjo/data/reference/eofs.nc, packed into the reference file with the WH04 climatology and normalisers), the
same arithmetic as the GEPS product: 15S-15N band means on the 2.5 deg EOF longitudes, anomalies, the trailing 120-day
mean removed, each field divided by its WH04 standard deviation.

Dates. GEFS day d (index d-1 of every 35-day array) is the mean of the samples at 24d-12 h and 24d h (OLR: the four
6-h averages over 24d-24..24d h), i.e. the UTC calendar day init + (d-1): day 1 IS the init date. Every date on these
figures, and every pairing with the observed RMM in the hindcast, uses that. (GEPS labels its day n as init + n; its day n
is the 24n-18..24n h mean, so that is one day late there. Not copied.)

The observed side is GEFS's OWN analysis, all three fields - the GEPS approach ("grow the observed track from the
forecast model's own zero-lag analysis, so there is no reanalysis-to-model handoff jump"), carried one step further:
  winds  u850/u200 from the control's analysis (gec00 f000) at 12Z and 00Z, averaged exactly as a forecast day is
         (the 12Z and next 00Z samples = that calendar day)
  OLR    the 0-6 h average of the control at 00/06/12/18Z, meaned (the four 6-h averages over the calendar day =
         exactly how a forecast day's OLR is made); GEFS writes no analysis OLR, and these are the shortest-range OLR
         it has
  120-d  the WH04 filter maps (the trailing 120-day mean of each anomaly) from the same analyses: running (the 120
         days before each day) for the observed track, the 120 days before the init for the forecast
Checked against the observed RMM (scripts/mjo/src/rmm_era5.py, ERA5) on the reforecast: day 1 of its 1,028 starts,
the start date and the same arithmetic as an analysis day, has a bivariate correlation of 0.990 with it, RMSE 0.22
and amplitude ratio 1.01 (ENSO removed from both). So the
observed track, its filter and the forecast are one model's one frame, and nothing leaves the S3 bucket the
forecast comes from (~2,000 small byte-ranged requests on a cold cache, a handful a day on a warm one: per-cycle band
means are kept in CACHE/mjo_anl/). GEPS takes its OLR track from ERA5 / the GMGSI proxy (olr_channel.py) and its wind
filter map from ERA5 (wind_map120.nc), and filters only its forecast OLR with its own day-1 OLR
(mjo_geps.forecast_olr_map): the Datamart keeps a month of GEPS runs, too few for a track filtered by its own 120 days.

Reference frames (GEPS's fix, A(L) = [F(L) - M(L)] + [M(L=0) - O]). F is the forecast, M the GEFS reforecast model
climatology at the INIT day of year and the same lead, O the WH04 observed climatology (scripts/mjo climatology.nc,
1979-2001, three harmonics) at the valid date. F - M removes the drift but leaves the forecast in the frame of the
reforecast's climate; the observed track sits in the frame of the analysis. GEFS's climatology has no lead 0: its
first value is day 1, the 12 h and 24 h samples (OLR: 0-24 h). So M(L=0) is taken as M(day 1) at the VALID day of
year - "a start on the valid date, first day": the same choice GEPS makes (its L=0.5, also its first daily mean).
Because GEFS day 1 is valid ON the init date, at L = 1 the two M terms are the same number and A(1) = F(1) - O
EXACTLY, i.e. forecast day 1 and the analysis day it overlaps are the same arithmetic on the same valid window; at
long lead A removes the drift RELATIVE TO DAY 1, M(L) - M(1). With the filter maps from the same analyses, O cancels
against the map to within its seasonal change over 120 days, so nothing here depends on how NOAA OLR and GEFS OLR
differ (+18 W/m2 in the band mean). Tested on the reforecast (build_gefs_mjo.py, 1,028 starts): with the shift, COR /
RMSE at days 1, 10, 20 are 0.965/0.40, 0.781/1.02, 0.483/1.54; with F - M alone 0.951/0.47, 0.769/1.04, 0.467/1.55;
with the shift frozen at the init date (6-year subset), equal at day 1 and below GEPS's form from day 10. Kept.

GEFS's members are too active in RMM space: a member's RMM amplitude grows from 1.03x the observed at day 1 to 1.5x by
day 20-35 (reforecast, 4 members), so a few-member mean still reads ~1.05-1.15x and the 31-member mean track can sit
well outside the unit circle at long lead. The skill figure's amplitude panel shows it; nothing here rescales it.
The live mean track is checked against the reforecast instead (track_check): a peak after day 10 beyond the 95th
percentile of the reforecast's 4-member-mean peaks (3.4) is printed on the figure as model bias, with the verification
of such peaks (on the 52 reforecast starts that went beyond it the observed RMM averaged amplitude 2.1 on the same
day against a forecast 3.8). The excess is state-dependent: in the strong El Nino autumn of 2015, 4 of 13 Sep-Nov starts went
beyond it, their means drifting to negative RMM2 (day 35 -1.3 against the observed -0.7) - a standing signal that grows
after the init, which neither the unconditional reforecast climatology, the 120-day filter of the analyses nor the
ENSO removal below (fixed at the init) can remove.

ENSO. The RMM removes the 120-day mean only, as the observed RMM does (scripts/mjo/src/rmm_era5.py), and while an
El Nino grows the 120-day mean lags the warming, so the standing anomaly projects onto RMM1. Every track has it removed (enso_removal): RMM - dN34 * c, dN34 the OISST
Nino-3.4 on the day minus its mean over the 120 days of the day's filter (the forecast holds the init's), c the
three-field correction of scripts/mjo/src/enso_rmm.py (enso_rmm3_correction.json, carried in the reference file as
enso_c: -0.59 per kelvin on RMM1, 2003-2022 fit; the RMM2 part is not significant and is 0). GEFS's own reforecast
RMM1 depends on the start's dN34 by -0.52 at day 1 and -0.46 to -0.57 out to day 35, close to c, and in the
reforecast the observed Nino-3.4 of each valid date (an upper bound on any SST forecast) scored no better than the
init's held. The skill and track_check pairs have ENSO removed on both sides (hindcast_pairs).

Model climatology: the band means of the reforecast climatology (the weekly files' `band` of release gefs-clim-v3,
pooled on the common-year day of year), packed per 5-day centre into the reference file by build_gefs_mjo.py and read
at the init day of year exactly as gefs_live.clim_at reads the maps (linear between the two centres around it).

The ENSEMBLE MEAN is projected for the bold track; its amplitude damps with lead as members decohere - a shrinking
radius is loss of agreement, not a forecast that the MJO decays. A GROWING mean radius is the opposite case: every
member shares the same standing drift (track_check above). The plume is every member projected alike.

    python gefs_mjo.py --npz /tmp/gefs_work/gefs2_20260925.npz --date 20260925 --site . --cache /tmp/gefs_work/clim \
        [--ref DIR]      # DIR holds gefs_mjo_ref.npz + gefs_mjo_hindcast.npz (release gefs-ref-v1); default: cache
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_reforecast as R  # (get, index, decode, grid, packing)
import gefs_daily as G  # (to_half, box: the live reduction)

S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
REF_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-ref-v1/{}"
REF_FILE, HIND_FILE = "gefs_mjo_ref.npz", "gefs_mjo_hindcast.npz"
TAGS = ("olr", "u200", "u850")  # order of the `band` arrays (gefs_daily.BANDF)
LON25 = np.arange(0.0, 360.0, 2.5)  # the WH04 EOF longitudes
TRACK_DAYS = 40  # observed track shown
MAP_DAYS = 120  # WH04 low-frequency filter
MAP_MIN = 110  # days of the 120 that must exist (olr_channel's min_periods)
MUTED = "#6f6b64"
WARN = "#8b1a1a"
TRACK_PCT = 95  # a mean-track peak after day 10 beyond this percentile of the reforecast's own peaks is flagged
STRONG_NINO = 1.5  # deg C: from this Nino-3.4 the reforecast's one strong El Nino autumn is cited beside the flag
NINO_AUTUMN = (2015, (9, 10, 11))  # Sep-Nov 2015, ONI +2.4: the strongest El Nino of the 2000-2019 reforecast
# ECMWF extended range, for the reference marks - GEPS's values, drawn exactly as there (mjo_hindcast.py): published
# bivariate-correlation skill of the ECMWF reforecasts (Vitart 2017, QJRMS 143:2210, Fig. 2, and the ECMWF annual
# verification memoranda): COR 0.6 at about day 27 in the 2016 cycle and near day 30 by the 2020s; 0.5 around day 33.
ECMWF_REF = {"COR 0.6": (27, 30), "COR 0.5": (31, 34)}
PHASE_ANGLES = {1: 202.5, 2: 247.5, 3: 292.5, 4: 337.5, 5: 22.5, 6: 67.5, 7: 112.5, 8: 157.5}
REGION_LABELS = [
    ("West. Hem.\nand Africa", -3.4, 0.0, 90),
    ("Indian Ocean", 0.0, -3.4, 0),
    ("Maritime\nContinent", 3.4, 0.0, -90),
    ("Western Pacific", 0.0, 3.4, 0),
]


def _clear_offset(x, y, base, labels, unit=0.019):
    """Offset (points) for a "day N" label at (x, y): `base` unless its box overlaps a region name (`labels` =
    (text, x, y, rotation) in data units), else the first candidate that does not (e.g. "day 35" landing on
    "Indian Ocean"). Boxes are approximate (axes span ~8.6 units over ~6.5 in, so 1 pt ~ 0.019 units)."""

    def box(o):  # annotate's text: left/baseline at the offset
        x0, y0 = x + o[0] * unit, y + o[1] * unit
        return x0, x0 + 0.85, y0 - 0.05, y0 + 0.28

    def regions():
        for t, lx, ly, rot in labels:
            w = 0.16 * max(len(s) for s in t.split("\n")) / 2 + 0.1
            h = 0.2 * (t.count("\n") + 1) / 2 + 0.06
            yield (lx - h, lx + h, ly - w, ly + w) if rot else (lx - w, lx + w, ly - h, ly + h)

    def hits(o):
        a = box(o)
        return sum(
            max(0.0, min(a[1], r[1]) - max(a[0], r[0])) * max(0.0, min(a[3], r[3]) - max(a[2], r[2])) for r in regions()
        )

    cands = [
        base,
        (11, 9),
        (11, -13),
        (-13, 11),
        (-13, -13),
        (11, -24),
        (-13, -24),
        (11, 22),
        (-13, 22),
        (14, -4),
        (-58, -4),
    ]
    return min(cands, key=lambda o: (round(hits(o), 3), cands.index(o)))


# ── reference ─────────────────────────────────────────────────────────────────────────────────────────────────
def ref_path(name: str, dirs) -> Path:
    """A reference file from the first directory that has it, else downloaded from release gefs-ref-v1 into the
    last directory given (the cache)."""
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    p = Path(dirs[-1]) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(R.get(REF_URL.format(name)))
    return p


def load_ref(dirs) -> dict:
    z = np.load(ref_path(REF_FILE, dirs))
    return {k: z[k] for k in z.files}


def to25(a):
    """Last axis 1.5 deg (0 ... 358.5) -> the 2.5 deg EOF longitudes, linear, across the dateline."""
    a = np.asarray(a, dtype="float64")
    lw = np.r_[R.LON, 360.0]
    aw = np.concatenate([a, a[..., :1]], axis=-1)
    flat = aw.reshape(-1, aw.shape[-1])
    return np.stack([np.interp(LON25, lw, r) for r in flat]).reshape((*a.shape[:-1], len(LON25)))


def centre_weights(doy: float):
    """gefs_live.clim_at's interpolation: linear between the two 5-day centres around the day-of-year key (dkey)."""
    centres = list(range(1, 366, 5))
    lo = max(c for c in centres if c <= doy)
    hi = lo + 5 if lo + 5 <= 361 else 1
    w = (doy - lo) / 5.0
    return [(lo, 1.0 - w), (hi, w)]


def model_clim(ref: dict, doy: float) -> np.ndarray:
    """Reforecast band climatology (3, 35, 240) [olr, u200, u850] at a day-of-year key (dkey)."""
    cen = list(ref["centres"])
    out = 0.0
    for c, w in centre_weights(doy):
        q = ref["mclim"][cen.index(c)]
        out = out + w * np.stack([R.unpack(t, q[i]) for i, t in enumerate(TAGS)])
    return out


def obs_clim(ref: dict, doy: float) -> np.ndarray:
    """WH04 observed climatology (3, 144) at a day-of-year key (dkey). Its 366 slots (climatology.nc) are common-year
    days 1..365 plus a wrap slot; 29 Feb (59.5) is the mean of 28 Feb and 1 March."""
    O = ref["O"]
    return 0.5 * (O[:, int(np.floor(doy)) - 1, :].astype("float64") + O[:, int(np.ceil(doy)) - 1, :])


def yday(d: dt.date) -> int:
    return d.timetuple().tm_yday


def dkey(d: dt.date) -> float:
    """Day-of-year key of the climatologies: the common-year day, 29 Feb = 59.5 (gefs_reforecast.noleap_doy)."""
    return R.noleap_doy(d)


def project(fo, f850, f200, ref):
    """Three-field WH04 projection -> (rmm1, rmm2), unit-variance normalised (mjo_geps._project)."""
    e1 = np.concatenate([ref["eof_u850"][0], ref["eof_u200"][0]])
    e2 = np.concatenate([ref["eof_u850"][1], ref["eof_u200"][1]])
    o1, o2 = ref["eof_olr"][0], ref["eof_olr"][1]
    comb = np.concatenate([f850, f200], axis=-1)
    return ((fo @ o1 + comb @ e1) / float(ref["pc_std"][0]), (fo @ o2 + comb @ e2) / float(ref["pc_std"][1]))


def channels_rmm(ch: np.ndarray, ref: dict):
    """(..., 3, 144) normalised channels [olr, u200, u850] -> rmm1, rmm2 of shape (...)."""
    return project(ch[..., 0, :], ch[..., 2, :], ch[..., 1, :], ref)


def channels_rmm_wind(ch: np.ndarray, ref: dict, pc_wind_std) -> tuple[np.ndarray, np.ndarray]:
    """The wind-only RMM (U850 + U200 on the wind parts of the EOFs, normalised by the wind-only projection's own
    standard deviation): the index the MJO page's multi-model diagram shares with the ECMWF ensembles."""
    comb = np.concatenate([ch[..., 2, :], ch[..., 1, :]], axis=-1)
    e1 = np.concatenate([ref["eof_u850"][0], ref["eof_u200"][0]])
    e2 = np.concatenate([ref["eof_u850"][1], ref["eof_u200"][1]])
    return (comb @ e1) / float(pc_wind_std[0]), (comb @ e2) / float(pc_wind_std[1])


def write_wind_track(m1: np.ndarray, m2: np.ndarray, init: dt.date, out: Path) -> None:
    """One model's track in the layout of scripts/mjo/src/rmm_export.track (the MJO page reads it): the ensemble mean,
    the members, lead days and valid times. GEFS day d is the UTC day init + d - 1."""

    def r(a, nd=3):
        return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(a, float).ravel()]

    n_lead = m1.shape[1]
    out.write_text(
        json.dumps(
            {
                "label": "GEFS",
                "init": f"{init:%Y-%m-%d} 00Z",
                "n": int(m1.shape[0]),
                "lead": list(range(1, n_lead + 1)),
                "t": [(init + dt.timedelta(days=k)).strftime("%Y-%m-%dT00") for k in range(n_lead)],
                "r1": r(m1.mean(0)),
                "r2": r(m2.mean(0)),
                "m1": [r(x, 2) for x in m1],
                "m2": [r(x, 2) for x in m2],
                "note": "wind-only RMM (U850 + U200), GEFS analyses for the 120-day filter",
            },
            separators=(",", ":"),
        )
    )


# ── GEFS's own analysis ───────────────────────────────────────────────────────────────────────────────────────
def _band(v) -> np.ndarray:
    """A native 0.5 deg field -> the live run's 15S-15N band (gefs_daily: 3x3 box to 1.5 deg, the BAND rows)."""
    return G.box(G.to_half(np.asarray(v, dtype="float64")), 3)[R.BAND].mean(0)


def anl_cycle(cyc: dt.datetime, cache: Path):
    """Band means from one control cycle: u850/u200 analysis (00Z/12Z cycles) and the 0-6 h mean OLR (every cycle).
    Cached per cycle; an incomplete cycle is not cached, so it is asked for again next run."""
    p = cache / f"{cyc:%Y%m%d%H}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    d, hh = cyc.strftime("%Y%m%d"), cyc.hour
    url = f"{S3}/gefs.{d}/{hh:02d}/atmos/pgrb2ap5/gec00.t{hh:02d}z.pgrb2a.0p50.f{{:03d}}"
    out = {}
    try:
        if hh in (0, 12):
            u = url.format(0)
            idx = R.index(u)
            for tag, lev in (("u850", "850 mb"), ("u200", "200 mb")):
                hit = [e for e in idx if e[4] == "UGRD" and e[2] == lev and e[3] == "anl"]
                out[tag] = _band(R.decode(R.get(u, hit[0][:2]))).astype("float32")
        u = url.format(6)
        idx = R.index(u)
        hit = [e for e in idx if e[4] == "ULWRF" and e[2] == "top of atmosphere" and e[3] == "0-6 hour ave fcst"]
        out["olr"] = _band(R.decode(R.get(u, hit[0][:2]))).astype("float32")
    except Exception:  # noqa: BLE001
        return None
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(p, **out)
    return out


def analysis_days(first: dt.date, last: dt.date, cache: Path, workers: int = 16):
    """Daily band means [olr, u200, u850] (n, 3, 240) for the calendar days first..last, NaN where incomplete.

    A day's winds are the 12Z analysis and the next 00Z analysis averaged (a forecast day's two samples); its OLR is
    the four 0-6 h averages from the 00/06/12/18Z cycles (the forecast day's four 6-h averages). All-or-nothing per
    field and day: a missing sample leaves the day NaN rather than a biased mean."""
    days = [first + dt.timedelta(days=k) for k in range((last - first).days + 1)]
    cycles = sorted(
        {dt.datetime(d.year, d.month, d.day) + dt.timedelta(hours=h) for d in days for h in (0, 6, 12, 18, 24)}
    )
    with ThreadPoolExecutor(workers) as ex:
        got = dict(zip(cycles, ex.map(lambda c: anl_cycle(c, cache), cycles)))
    out = np.full((len(days), 3, len(R.LON)), np.nan)
    for i, d in enumerate(days):
        t0 = dt.datetime(d.year, d.month, d.day)
        o = [got.get(t0 + dt.timedelta(hours=h)) for h in (0, 6, 12, 18)]
        if all(x is not None and "olr" in x for x in o):
            out[i, 0] = np.mean([x["olr"] for x in o], axis=0)
        w = [got.get(t0 + dt.timedelta(hours=h)) for h in (12, 24)]
        for j, tag in ((1, "u200"), (2, "u850")):
            if all(x is not None and tag in x for x in w):
                out[i, j] = np.mean([x[tag] for x in w], axis=0)
    miss = [f"{d:%m-%d}" for d, r in zip(days, out) if not np.isfinite(r).all()]
    if miss:
        print(f"  analysis: {len(miss)} incomplete day(s): {', '.join(miss[:12])}", flush=True)
    return days, out


def observed(init: dt.date, ref: dict, cache: Path):
    """Observed three-field RMM for the TRACK_DAYS days before the init, each day filtered with the mean of the 120
    days before it (WH04), plus the filter map of the 120 days before the init for the forecast.

    -> (dates, rmm1, rmm2, map_now (3, 144) in anomaly units, analysis anomalies (n, 3, 144), days)"""
    last = init - dt.timedelta(days=1)
    first = last - dt.timedelta(days=TRACK_DAYS + MAP_DAYS - 1)
    days, A = analysis_days(first, last, cache)
    anom = to25(A) - np.stack([obs_clim(ref, dkey(d)) for d in days])  # (n, 3, 144)

    def filt(i):  # mean of the 120 days before day i
        w = anom[max(0, i - MAP_DAYS) : i]
        ok = np.isfinite(w).all(-1)  # (k, 3)
        if w.shape[0] < MAP_DAYS or (ok.sum(0) < MAP_MIN).any():
            return None
        return np.stack([np.nanmean(w[:, j], axis=0) for j in range(3)])

    std = ref["std"][:, None]
    dates, r1, r2 = [], [], []
    for i in range(len(days) - TRACK_DAYS, len(days)):
        m = filt(i)
        if m is None or not np.isfinite(anom[i]).all():
            continue
        a, b = channels_rmm((anom[i] - m) / std, ref)
        dates.append(days[i])
        r1.append(float(a))
        r2.append(float(b))
    map_now = filt(len(days))
    if map_now is None:
        raise SystemExit("fewer than 110 of the 120 analysis days before the init: no filter map")
    if not dates:
        raise SystemExit("no complete observed day in the track window")
    return dates, np.array(r1), np.array(r2), map_now, anom, days


# ── forecast ──────────────────────────────────────────────────────────────────────────────────────────────────
def forecast_channels(band: np.ndarray, init: dt.date, ref: dict, map_now: np.ndarray):
    """(M, 3, 35, 240) raw member band means -> normalised channels (M, 35, 3, 144) in the analysis frame, and the
    de-drifted anomaly F - M(L) (M, 3, 35, 240) for the Hovmoller.

    A(L) = [F(L) - M(L; init doy)] + [M(1; valid doy) - O(valid doy)] - map120; valid date of day L = init + L - 1,
    so at L = 1 the M terms cancel and A(1) = F(1) - O - map120."""
    M = model_clim(ref, dkey(init))  # (3, 35, 240)
    anom = band.astype("float64") - M[None]
    shift = np.empty((3, 35, len(LON25)))
    for k in range(35):
        vdoy = dkey(init + dt.timedelta(days=k))
        shift[:, k] = to25(model_clim(ref, vdoy)[:, 0]) - obs_clim(ref, vdoy)
    ch = (to25(anom) + shift[None] - map_now[None, :, None, :]) / ref["std"][None, :, None, None]
    return np.moveaxis(ch, 2, 1), anom


def nino34_now(site: Path) -> float | None:
    """The latest daily OISST Nino-3.4 anomaly (assets/sst/data/enso_daily.json, deg C), or None without the file."""
    p = Path(site) / "assets" / "sst" / "data" / "enso_daily.json"
    try:
        v = [x for x in json.loads(p.read_text())["daily"]["nino34"] if x is not None]
    except (OSError, KeyError, ValueError):
        return None
    return float(v[-1]) if v else None


def n34_smoothed(site: Path):
    """The OISST Nino-3.4 anomaly as the ENSO correction reads it (scripts/mjo/src/enso_rmm.n34_smoothed): daily from
    assets/sst/data/enso_daily.json, gaps interpolated, 31-day trailing mean of at least 20 days.
    -> (days datetime64[D], values), or None without the file"""
    p = Path(site) / "assets" / "sst" / "data" / "enso_daily.json"
    try:
        d = json.loads(p.read_text())["daily"]
    except (OSError, KeyError, ValueError):
        return None
    t = np.array(d["dates"], dtype="datetime64[D]")
    v = np.array([np.nan if x is None else x for x in d["nino34"]], dtype="float64")
    days = np.arange(t[0], t[-1] + 1)
    x = np.interp(days.astype("int64"), t[np.isfinite(v)].astype("int64"), v[np.isfinite(v)])
    k = np.arange(len(x))
    lo = np.maximum(k - 30, 0)
    cs = np.r_[0.0, np.cumsum(x)]
    keep = k + 1 - lo >= 20
    return days[keep], ((cs[k + 1] - cs[lo]) / (k + 1 - lo))[keep]


def delta_n34(n34, t, end) -> float:
    """enso_rmm.delta_n34: the Nino-3.4 at t (its latest value at or before t, so a forecast holds the init's) minus
    its mean over the MAP_DAYS days ending on `end`; NaN with fewer than 90 of them or nothing at t."""
    days, v = n34
    t, end = np.datetime64(t, "D"), np.datetime64(end, "D")
    win = (days > end - MAP_DAYS) & (days <= end)
    now = days <= t
    if win.sum() < 90 or not now.any():
        return np.nan
    return float(v[now][-1] - v[win].mean())


def hindcast_pairs(hind_path: Path, c: np.ndarray):
    """The reforecast pairs of gefs_mjo_hindcast.npz with ENSO removed as from the live track: every member by its
    start's dN34 (dn_f) at every lead, the observed RMM by the valid date's (dn_o); c = the applied (RMM1, RMM2)
    correction, zeros for the pairs as they are (so too a file without dn_f).
    -> starts (str), fm1, fm2 (S, members, L), o1, o2 (S, L), amp0 (S,) the observed amplitude on the start date, note"""
    z = np.load(hind_path)
    fm1, fm2 = z["fm1"].astype("float64"), z["fm2"].astype("float64")
    o1, o2 = z["o1"].astype("float64"), z["o2"].astype("float64")
    if "dn_f" in z.files:
        f, o = z["dn_f"].astype("float64")[:, None, None], z["dn_o"].astype("float64")
        fm1, fm2, o1, o2 = fm1 - f * c[0], fm2 - f * c[1], o1 - o * c[0], o2 - o * c[1]
    return z["starts"], fm1, fm2, o1, o2, np.hypot(o1[:, 0], o2[:, 0]), str(z["note"])


def track_check(hind_path: Path, r1: np.ndarray, r2: np.ndarray, nino34: float | None, c) -> list[str] | None:
    """The live ensemble-mean track against the reforecast (the same projection, gefs_mjo_hindcast.npz, ENSO removed
    by c as the track is): when its peak amplitude after day 10 is beyond what the reforecast's 4-member means reach on
    TRACK_PCT % of starts (a 31-member mean should reach less, as it damps more), the figure says so with the
    reforecast's own verification of such peaks, and, from STRONG_NINO, how its strong El Nino autumn behaved.
    -> footer lines, or None within range."""
    starts, fm1, fm2, o1, o2, _, _ = hindcast_pairs(hind_path, c)
    ok = np.isfinite(fm1).all((1, 2)) & np.isfinite(o1).all(1)
    a = np.hypot(fm1.mean(1), fm2.mean(1))[:, 10:]
    k = np.argmax(np.where(ok[:, None], a, -1.0), axis=1)
    peak = a[np.arange(len(a)), k]
    thr = float(np.percentile(peak[ok], TRACK_PCT))
    amp = np.hypot(r1, r2)[10:]
    j = int(np.argmax(amp))
    if amp[j] <= thr:
        return None
    hi = ok & (peak > thr)
    seen = np.hypot(o1, o2)[:, 10:][np.arange(len(a)), k]
    out = [
        f"Model bias, not MJO strength: the mean track reaches amplitude {amp[j]:.1f} on day {j + 11}, beyond the "
        f"{thr:.1f} that {TRACK_PCT} % of the reforecast's 4-member means stay within after day 10; on the {int(hi.sum())} "
        f"reforecast starts that went beyond it, the observed RMM averaged {seen[hi].mean():.1f} on the same day against a "
        f"forecast {peak[hi].mean():.1f}."
    ]
    if nino34 is not None and nino34 >= STRONG_NINO:
        y, mons = NINO_AUTUMN
        en = ok & np.array([s[:4] == str(y) and int(s[5:7]) in mons for s in starts])
        out.append(
            f"In its strong El Niño autumn (Sep–Nov {y}) {int((en & hi).sum())} of {int(en.sum())} starts went beyond "
            f"it (against {100 - TRACK_PCT} % of all starts); their day-35 mean RMM2 was {fm2.mean(1)[en, -1].mean():+.1f} "
            f"against the observed {o2[en, -1].mean():+.1f}."
        )
    return out


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────────────
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def draw_phase_wheel(ax, radius: float = 4.0, amp_rings=(1, 2, 3)):
    """The AIFS-ENS / GEPS wheel (scripts/mjo/src/plot.draw_phase_wheel, copied so the Actions job needs neither
    xarray nor pandas): phase 1 in the left sector, numbering counterclockwise."""
    ax.set_aspect("equal")
    ax.axhline(0, color="0.5", lw=0.7, zorder=1)
    ax.axvline(0, color="0.5", lw=0.7, zorder=1)
    for angle in (45, 135):
        rad = np.deg2rad(angle)
        ax.plot(
            [-radius, radius], [-radius * np.tan(rad), radius * np.tan(rad)], color="0.6", lw=0.6, ls="--", zorder=1
        )
    for phase, centre in PHASE_ANGLES.items():
        rad = np.deg2rad(centre)
        ax.text(
            radius * 0.78 * np.cos(rad),
            radius * 0.78 * np.sin(rad),
            str(phase),
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
            fontweight="bold",
            zorder=2,
        )
    for text, x, y, rot in REGION_LABELS:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color="0.4",
            style="italic",
            rotation=rot,
            rotation_mode="anchor",
        )
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in amp_rings:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="0.45", ls="--", lw=0.8, zorder=1, alpha=0.6)
        ax.text(
            r * np.cos(np.deg2rad(-45)),
            r * np.sin(np.deg2rad(-45)),
            str(r),
            ha="center",
            va="center",
            fontsize=8,
            color="0.4",
            zorder=2,
            bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.7),
        )
    lim = radius + 0.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("RMM1", fontsize=11)
    ax.set_ylabel("RMM2", fontsize=11)


def plot_phase(init, odates, or1, or2, m1, m2, n_members, out: Path, enso_note: str, warn: list[str] | None = None):
    """m1, m2: (members, 35) RMM per member; the bold track is the ensemble mean's (= the mean of the members' RMM,
    the projection being linear). enso_note: the footer's line on the ENSO removal. warn: track_check's lines, printed
    under the footer (the figure grows to fit)."""
    plt = _plt()
    import textwrap
    import matplotlib.colors as mcolors

    r1, r2 = m1.mean(0), m2.mean(0)
    L = np.arange(1, m1.shape[1] + 1)
    note = "\n".join(textwrap.fill(t, 108) for t in warn) if warn else ""
    extra = 0.15 * (note.count("\n") + 1) + 0.05 if note else 0.0  # inches added below the wheel
    H = 9.0 + extra
    fig, ax = plt.subplots(figsize=(8.5, H))
    fig.subplots_adjust(left=0.082, right=0.978, top=1 - 0.558 / H, bottom=(1.305 + extra) / H)
    draw_phase_wheel(ax)
    ax.plot(or1, or2, color="0.55", lw=1.0, alpha=0.75, zorder=5)
    months = [d.strftime("%Y-%m") for d in odates]
    cmap_m = plt.get_cmap("tab10")
    for k, mo in enumerate(dict.fromkeys(months)):
        sel = np.array([m == mo for m in months])
        lab = dt.datetime.strptime(mo, "%Y-%m").strftime("%b %Y")
        ax.scatter(
            or1[sel], or2[sel], s=22, zorder=6, color=cmap_m(k % 10), edgecolor="0.3", linewidth=0.3, label=f"Obs {lab}"
        )
    ax.scatter(or1[-1], or2[-1], marker="*", s=210, zorder=8, color="#111111", label=f"analysis {odates[-1]:%b %d}")
    for k in range(m1.shape[0]):
        ax.plot(np.r_[or1[-1], m1[k]], np.r_[or2[-1], m2[k]], color="#4a7bb5", alpha=0.42, lw=0.9, zorder=3)
    cmap, norm = plt.get_cmap("plasma_r"), mcolors.Normalize(0, len(L) - 1)
    x, y = np.r_[or1[-1], r1], np.r_[or2[-1], r2]
    for i in range(len(x) - 1):
        ax.plot(x[i : i + 2], y[i : i + 2], color=cmap(norm(max(i - 1, 0))), lw=2.4, zorder=7)
    for k in range(4, len(L), 5):
        ax.scatter(r1[k], r2[k], s=42, color=cmap(norm(k)), edgecolor="0.25", linewidth=0.4, zorder=8)
    for k, (dx0, dy0) in zip((9, 19, 34), ((11, 9), (11, -13), (-13, 11))):
        if k < len(L):
            dx, dy = _clear_offset(r1[k], r2[k], (dx0, dy0), REGION_LABELS)
            ax.annotate(
                f"day {L[k]}",
                (r1[k], r2[k]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=9,
                fontweight="bold",
                zorder=10,
                color=cmap(norm(k)),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.82),
            )
    ax.scatter(
        r1[-1],
        r2[-1],
        marker="s",
        s=70,
        color=cmap(norm(len(L) - 1)),
        edgecolor="0.25",
        linewidth=0.5,
        zorder=9,
        label=f"GEFS day {L[-1]}",
    )
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
    ax.set_title(f"GEFS Ensemble RMM  —  Init: {init:%Y-%m-%d} 00Z", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        (0.198 + extra) / H,
        f"Three-field RMM (OLR + U850 + U200) · {n_members} members\n"
        f"observed track: GEFS's own analysis, each day filtered by the 120 days before it\n{enso_note}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )
    if note:
        fig.text(0.5, 0.15 / H, note, ha="center", va="bottom", fontsize=8.6, color=WARN, linespacing=1.25)
    p = out / "gefs_mjo_phase.webp"
    fig.savefig(p, dpi=110, facecolor="white", pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    amp = np.hypot(r1, r2)
    ma = np.hypot(m1, m2)
    print(
        f"  {p.name}  obs amp {np.hypot(or1[-1], or2[-1]):.2f} ({odates[-1]}), mean track day1 {amp[0]:.2f} -> "
        f"day35 {amp[-1]:.2f}; members day35 p10-p90 {np.percentile(ma[:, -1], 10):.2f}-"
        f"{np.percentile(ma[:, -1], 90):.2f}",
        flush=True,
    )
    return p.name


def _hov_panel(ax, lon, L, F, cmap, lim, init, dates_on_right):
    m = ax.pcolormesh(lon, L, F, cmap=cmap, vmin=-lim, vmax=lim, shading="auto")
    ax.invert_yaxis()
    ax.set_xlim(lon[0] - 0.75, lon[-1] + 0.75)
    ax.set_xticks([0, 60, 120, 180, 240, 300])
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W"], fontsize=9)
    for x in (100, 150):  # Indian | Maritime | Pacific
        ax.axvline(x, color="#00000033", lw=0.8, ls=(0, (4, 3)))
    for x, t in ((58, "Indian Ocean"), (132, "Maritime"), (215, "Pacific"), (310, "Atlantic")):
        ax.text(
            x,
            1.4,
            t,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#4a453f",
            fontweight="bold",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78),
        )
    ax.tick_params(labelsize=9)
    if dates_on_right:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ticks = [1, 7, 14, 21, 28, 35]
        ax2.set_yticks(ticks)
        ax2.set_yticklabels([f"{init + dt.timedelta(days=t - 1):%b %d}" for t in ticks], fontsize=8.5)
    return m


def plot_hovmoller(init, anom_mean, olr_offset, out: Path):
    """Two filled panels, convection (OLR) and the 850 hPa zonal wind, as the GEPS figure: the MJO's signature is the
    phase relation between the two. anom_mean: (3, 35, 240) ensemble-mean F - M(L) at 1.5 deg; olr_offset: (35,)
    uniform OLR offset removed per lead (the live run's global-mean OLR anomaly), as GEPS does."""
    plt = _plt()
    O = anom_mean[0] - olr_offset[:, None]
    U = anom_mean[2]  # m/s, as on every other figure of the site
    lon, L = R.LON, np.arange(1, O.shape[0] + 1)
    fig = plt.figure(figsize=(11.4, 9.2))
    left, right, bot, top = 0.062, 0.918, 0.108, 0.885
    gap = 0.055
    w = (right - left - gap) / 2
    axO = fig.add_axes([left, bot, w, top - bot])
    axU = fig.add_axes([left + w + gap, bot, w, top - bot])
    limO = float(np.nanpercentile(np.abs(O), 98))
    limU = float(np.nanpercentile(np.abs(U), 98))
    mO = _hov_panel(axO, lon, L, O, "BrBG_r", limO, init, False)
    mU = _hov_panel(axU, lon, L, U, "RdBu_r", limU, init, True)
    axO.set_ylabel("forecast lead (days)", fontsize=10.5)
    axU.set_yticklabels([])
    axO.set_title("Convection — OLR anomaly", fontsize=11.5, fontweight="bold", pad=6)
    axU.set_title("850 hPa zonal wind anomaly", fontsize=11.5, fontweight="bold", pad=6)
    for ax, m, lab in ((axO, mO, "OLR anomaly (W m⁻²)"), (axU, mU, "850 hPa u anomaly (m s⁻¹)")):
        bb = ax.get_position()
        cax = fig.add_axes([bb.x0 + 0.10 * bb.width, 0.042, 0.80 * bb.width, 0.015])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(lab, fontsize=9, labelpad=2)
        cb.ax.tick_params(labelsize=8, pad=1.5)
    fig.suptitle(
        f"Tropical convection and low-level wind, 15°S–15°N\nGEFS extended · init {init:%Y-%m-%d}",
        fontsize=13,
        fontweight="bold",
        y=0.972,
        va="top",
    )
    p = out / "gefs_mjo_hovmoller.webp"
    fig.savefig(p, dpi=110, facecolor="white", pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    print(f"  {p.name}  OLR ±{limO:.0f} W/m², u850 ±{limU:.1f} m/s", flush=True)
    return p.name


# ── hindcast skill (scores + figure; the pairs come from build_gefs_mjo.py) ────────────────────────────────────
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def _curves(F1, F2, O1, O2):
    """Bivariate COR, RMSE, amplitude ratio by lead over the starts given (rows), NaN-aware."""
    good = np.isfinite(F1) & np.isfinite(O1)
    cor, rmse, ampr, nn = [], [], [], []
    for li in range(F1.shape[1]):
        g = good[:, li]
        a1, a2, b1, b2 = F1[g, li], F2[g, li], O1[g, li], O2[g, li]
        cor.append(float((a1 * b1 + a2 * b2).sum() / np.sqrt((a1**2 + a2**2).sum() * (b1**2 + b2**2).sum())))
        rmse.append(float(np.sqrt(((a1 - b1) ** 2 + (a2 - b2) ** 2).mean())))
        ampr.append(float(np.hypot(a1, a2).mean() / np.hypot(b1, b2).mean()))
        nn.append(int(g.sum()))
    return dict(cor=cor, rmse=rmse, amp_ratio=ampr, n=nn)


def cross(cor, thr):
    """Forecast day at which COR first falls through thr, linearly interpolated (mjo_hindcast.cross)."""
    c = np.asarray(cor)
    d = np.arange(1, len(c) + 1)
    below = np.where(c < thr)[0]
    if not len(below):
        return None
    if below[0] == 0:
        return 0.0
    i = below[0]
    return float(d[i] - 1 + (c[i - 1] - thr) / (c[i - 1] - c[i]))


def _fd(x, nd=1):
    """A crossing day for print: '>35' when the curve never falls through the threshold."""
    return ">35" if x is None else f"{x:.{nd}f}"


def _boot_cross(F1, F2, O1, O2, years, thr, n=2000, seed=1):
    """90% interval of the crossing day, resampling whole YEARS of starts (the MJO state persists across a year's
    consecutive weekly starts, so starts are not independent)."""
    uy = np.unique(years)
    num = np.zeros((len(uy), F1.shape[1]))
    fa = np.zeros_like(num)
    ob = np.zeros_like(num)
    for k, y in enumerate(uy):
        s = years == y
        g = np.isfinite(F1[s]) & np.isfinite(O1[s])
        num[k] = np.where(g, F1[s] * O1[s] + F2[s] * O2[s], 0).sum(0)
        fa[k] = np.where(g, F1[s] ** 2 + F2[s] ** 2, 0).sum(0)
        ob[k] = np.where(g, O1[s] ** 2 + O2[s] ** 2, 0).sum(0)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(0, len(uy), len(uy))
        c = num[k].sum(0) / np.sqrt(fa[k].sum(0) * ob[k].sum(0))
        x = cross(c, thr)
        out.append(np.nan if x is None else x)
    out = np.array(out)
    return [float(np.nanpercentile(out, 5)), float(np.nanpercentile(out, 95))] if np.isfinite(out).any() else None


def _scaled_members(fm1, fm2, O1, O2, M):
    """Bivariate COR and RMSE of an M-member ensemble mean, from the 4 hindcast members taken as exchangeable: the
    covariance with the observations does not depend on M, and the ensemble mean's power is  a + b/M  (a = the mean
    product of two different members, b = a member's power minus a). Exact for M = 4 on these starts."""
    ok = np.isfinite(fm1).all(1) & np.isfinite(O1)  # (S, L): all members and obs
    k = fm1.shape[1]
    cor, rmse = [], []
    for li in range(fm1.shape[2]):
        g = ok[:, li]
        a1, a2, b1, b2 = fm1[g, :, li], fm2[g, :, li], O1[g, li], O2[g, li]
        cov = ((a1.mean(1) * b1) + (a2.mean(1) * b2)).mean()
        p1 = (a1**2 + a2**2).mean()  # member power
        s1, s2 = a1.sum(1), a2.sum(1)
        pair = ((s1**2 - (a1**2).sum(1)) + (s2**2 - (a2**2).sum(1))).mean() / (k * (k - 1))
        pw = pair + (p1 - pair) / M
        po = (b1**2 + b2**2).mean()
        cor.append(float(cov / np.sqrt(pw * po)))
        rmse.append(float(np.sqrt(max(pw - 2 * cov + po, 0.0))))
    return cor, rmse


def skill(hind_path: Path, doy_now: int, out: Path, data: Path, c, n_live: int = 31):
    """Score the reforecast pairs (ENSO removed by c, hindcast_pairs) for every subset, write data/gefs_mjo_skill.json
    and gefs_mjo_skill.webp."""
    plt = _plt()
    t0 = time.time()
    starts, fm1, fm2, o1, o2, amp0, note = hindcast_pairs(hind_path, c)  # fm: (S, members, L)
    starts = [dt.date.fromisoformat(s) for s in starts]
    have = np.isfinite(fm1).any(1)  # (S, L): any member
    with np.errstate(invalid="ignore"):
        f1 = np.where(have, np.nansum(fm1, 1) / np.maximum(np.isfinite(fm1).sum(1), 1), np.nan)  # ensemble mean
        f2 = np.where(have, np.nansum(fm2, 1) / np.maximum(np.isfinite(fm2).sum(1), 1), np.nan)  # (projection linear)
    ok = np.isfinite(f1).all(1)
    L = np.arange(1, f1.shape[1] + 1)
    mon = np.array([s.month for s in starts])
    doy = np.array([yday(s) for s in starts])
    years = np.array([s.year for s in starts])
    dn = ((doy - doy_now + 183) % 366) - 183
    subsets = {
        "all": ok,
        "strong initial (amp ≥ 1)": ok & (amp0 >= 1.0),
        f"now ±45 d (doy {doy_now})": ok & (np.abs(dn) <= 45),
    }
    subsets.update({s: ok & np.isin(mon, m) for s, m in SEASONS.items()})
    res = {k: _curves(f1[v], f2[v], o1[v], o2[v]) for k, v in subsets.items()}
    full = ok & np.isfinite(fm1).all((1, 2))
    live, live_rmse = _scaled_members(fm1[full], fm2[full], o1[full], o2[full], n_live)
    four, four_rmse = _scaled_members(fm1[full], fm2[full], o1[full], o2[full], fm1.shape[1])
    info = {
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "leads": [int(x) for x in L],
        "valid_date": "day d is valid on init + (d - 1) days (day 1 = the init date)",
        "doy_now": int(doy_now),
        "ecmwf_reference": ECMWF_REF,
        "hindcast": note,
        "n_members_hindcast": int(fm1.shape[1]),
        "subsets": {},
        f"cor_{n_live}_members_estimate": {
            "cor": live,
            "rmse": live_rmse,
            "day_cor_0.6": cross(live, 0.6),
            "day_cor_0.5": cross(live, 0.5),
            "n_starts": int(full.sum()),
            "cor_4_members_same_starts": four,
            "rmse_4_members_same_starts": four_rmse,
            "method": "exchangeable-member scaling of the ensemble-mean power (a + b/M) from the 4 hindcast members",
        },
    }
    for k, v in res.items():
        m = subsets[k]
        info["subsets"][k] = {
            **v,
            "day_cor_0.6": cross(v["cor"], 0.6),
            "day_cor_0.5": cross(v["cor"], 0.5),
            "day_cor_0.6_ci90": _boot_cross(f1[m], f2[m], o1[m], o2[m], years[m], 0.6),
            "day_cor_0.5_ci90": _boot_cross(f1[m], f2[m], o1[m], o2[m], years[m], 0.5),
            "n_starts": int(m.sum()),
        }
        x = info["subsets"][k]

        def fmt(c):
            return f"{c[0]:.1f}-{c[1]:.1f}" if c else "-"

        print(
            f"  {k:28s} n {int(m.sum()):4d}  COR d5 {v['cor'][4]:.2f} d10 {v['cor'][9]:.2f} d15 {v['cor'][14]:.2f} "
            f"d20 {v['cor'][19]:.2f} d25 {v['cor'][24]:.2f}  0.6 at {_fd(x['day_cor_0.6'])} ({fmt(x['day_cor_0.6_ci90'])})"
            f"  0.5 at {_fd(x['day_cor_0.5'])} ({fmt(x['day_cor_0.5_ci90'])})",
            flush=True,
        )
    print(
        f"  {n_live}-member estimate: COR 0.6 at {_fd(cross(live, 0.6))}, 0.5 at {_fd(cross(live, 0.5))} "
        f"(4 members on the same starts {_fd(cross(four, 0.6))} / {_fd(cross(four, 0.5))})",
        flush=True,
    )
    data.mkdir(parents=True, exist_ok=True)
    (data / "gefs_mjo_skill.json").write_text(json.dumps(info))

    fig = plt.figure(figsize=(14.5, 5.4))
    now = f"now ±45 d (doy {doy_now})"
    cols = {
        "all": "#16335c",
        "DJF": "#2c5fa8",
        "MAM": "#3f9b6a",
        "JJA": "#c8781e",
        "SON": "#b4453c",
        "strong initial (amp ≥ 1)": "#6a3d9a",
        now: "#111111",
    }
    for pi, (key, ylab, title) in enumerate(
        (
            ("cor", "bivariate correlation", "Skill: bivariate correlation vs the observed RMM"),
            ("rmse", "bivariate RMSE", "Error"),
            ("amp_ratio", "forecast / observed amplitude", "Amplitude"),
        )
    ):
        ax = fig.add_axes([0.05 + pi * 0.325, 0.19, 0.27, 0.67])
        for k, v in res.items():
            lw = 2.6 if k == "all" else (2.2 if k.startswith("now") else 1.4)
            ls = "-" if not k.startswith("strong") else (0, (4, 2))
            ax.plot(L, v[key], color=cols[k], lw=lw, ls=ls, label=k)
        if key == "cor":
            ax.plot(L, live, color=cols["all"], lw=1.3, ls=":", label=f"all, {n_live}-member estimate")
            ax.axhline(0.6, color="#8a8680", lw=0.8, ls=":")
            ax.axhline(0.5, color="#8a8680", lw=0.8, ls=":")
            for lab, (d0, d1) in ECMWF_REF.items():
                yv = float(lab.split()[-1])
                ax.plot([d0, d1], [yv, yv], color="#8a1d1d", lw=4, alpha=0.6, solid_capstyle="butt")
                ax.text(
                    d1,
                    yv - 0.02,
                    f"ECMWF, published:\n{lab} at day {d0}–{d1}",
                    fontsize=7.0,
                    color="#8a1d1d",
                    va="top",
                    ha="right",
                )
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7.6, loc="lower left", framealpha=0.92)
        elif key == "rmse":
            ax.plot(L, live_rmse, color=cols["all"], lw=1.3, ls=":")
            ax.axhline(np.sqrt(2.0), color="#8a8680", lw=0.8, ls=":")
            ax.text(1.5, np.sqrt(2.0) + 0.02, "√2 = climatology forecast", fontsize=7, ha="left", color="#5a5650")
        elif key == "amp_ratio":
            ax.axhline(1.0, color="#8a8680", lw=0.8, ls=":")
            ax.set_ylim(0.4, max(1.2, 0.05 + max(max(v["amp_ratio"]) for v in res.values())))
        ax.set_xlim(1, 35)
        ax.set_xlabel("forecast day", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8.5)
    a = info["subsets"]["all"]
    ci = a["day_cor_0.6_ci90"]
    y0, y1 = min(years[ok]), max(years[ok])
    fig.suptitle(
        f"GEFS MJO skill from the GEFSv12 reforecast, {y0}–{y1} · {int(ok.sum())} starts, ensemble mean · "
        f"COR 0.6 at day {_fd(a['day_cor_0.6'], 0)}"
        + (f" (90%: {ci[0]:.0f}–{ci[1]:.0f})" if ci else "")
        + f", 0.5 at day {_fd(a['day_cor_0.5'], 0)}",
        fontsize=12.5,
        fontweight="bold",
        y=0.985,
        va="top",
    )
    fig.text(
        0.5,
        0.012,
        "same projection as the live RMM (three-field WH04, lead-matched de-drift, 120-day mean "
        + (
            "and ENSO removed), verified against the observed RMM (computed here from ERA5 the same way) on the valid "
            "date (day 1 = the start date; ENSO removed from it too);\n"
            if np.any(c)
            else "removed), verified against the observed RMM (computed here from ERA5 the same way) on the valid date "
            "(day 1 = the start date);\n"
        )
        + f"weekly starts, hindcast mean of {fm1.shape[1]} members; the title's bracket is the 90% interval with "
        f"whole years resampled; dotted: the same starts scaled to the {n_live}-member operational mean\n"
        f"(members taken as exchangeable), COR 0.6 at day {_fd(cross(live, 0.6), 0)}; ECMWF marks are published "
        "reforecast values (Vitart 2017, ECMWF verification memoranda), not a like-for-like computation",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#5a5650",
    )
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gefs_mjo_skill.webp", dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"  gefs_mjo_skill.webp  ({time.time() - t0:.0f} s)", flush=True)
    return "gefs_mjo_skill.webp"


# ── driver ────────────────────────────────────────────────────────────────────────────────────────────────────
def enso_removal(ref: dict, site: Path, init: dt.date, odates):
    """The ENSO correction of the live tracks (enso_c of the reference, from scripts/mjo enso_rmm3_correction.json):
    dN34, the OISST Nino-3.4 on a day minus its mean over the 120 days of that day's filter (the 120 days before it,
    for the forecast the init's, its Nino-3.4 held), times c. Without the reference's c or the Nino-3.4 file nothing is
    removed. -> c, the observed days' shifts (n, 2), the forecast's (2,), its dN34, the figure's footer line"""
    one = dt.timedelta(days=1)
    c, n34 = ref.get("enso_c"), n34_smoothed(site)
    if c is None or n34 is None:
        print(f"::warning::ENSO not removed ({'no enso_c in the reference' if c is None else 'no Nino-3.4 file'})")
        return np.zeros(2), np.zeros((len(odates), 2)), np.zeros(2), np.nan, "ENSO not removed"
    dn_o = np.array([delta_n34(n34, d, d - one) for d in odates])
    dn = delta_n34(n34, init, init - one)
    if not np.isfinite([*dn_o, dn]).all():
        print("::warning::no Nino-3.4 change on some day(s): ENSO left in there", flush=True)
    so, sf = np.nan_to_num(dn_o)[:, None] * c, np.nan_to_num(dn) * c
    note = (
        f"ENSO removed: Niño-3.4 {dn:+.2f} °C against its mean over the 120-day filter, every forecast point moved "
        f"{np.hypot(*sf):.2f}"
    )
    return c, so, sf, dn, note


def live_olr_offset(z, init, cache: Path) -> np.ndarray:
    """The live run's uniform OLR offset against the reforecast, per lead, for the Hovmoller only (GEPS removes its
    ~7.6 W/m2): the global-mean weekly OLR anomaly of the ensemble mean against the weekly reforecast climatology
    (gefs_live.clim_at, through the shared cache). ~1.4 W/m2 on 2026-09-25 (GEPS: 7.6); in the RMM it would be 0.09
    sd of uniform OLR, which projects to about 0.01 - so the RMM keeps it, the Hovmoller does not."""
    try:
        import gefs_live as GL

        C = GL.clim_at(init, cache)
        wt = np.cos(np.deg2rad(R.LAT))[:, None] * np.ones((1, len(R.LON)))
        wk = z["w_olr"].mean(0) - C["w_olr"]
        g = np.array([(w * wt).sum() / wt.sum() for w in wk])
        return np.concatenate([np.full(b - a + 1, g[i]) for i, (a, b) in enumerate(R.WEEKS)])
    except Exception as e:  # noqa: BLE001
        print(f"  OLR offset unavailable ({str(e)[:60]}); none removed", flush=True)
        return np.zeros(35)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--site", default=str(HERE.parents[1]))
    ap.add_argument("--cache", default="/tmp/gefs_work/clim")
    ap.add_argument("--ref", default=None, help="directory holding gefs_mjo_ref.npz / gefs_mjo_hindcast.npz")
    a = ap.parse_args()
    t0 = time.time()
    init = dt.date(int(a.date[:4]), int(a.date[4:6]), int(a.date[6:]))
    cache = Path(a.cache)
    dirs = ([a.ref] if a.ref else []) + [str(cache)]
    out = Path(a.site) / "assets" / "gefs"
    data = out / "data"
    out.mkdir(parents=True, exist_ok=True)
    ref = load_ref(dirs)
    z = np.load(a.npz)
    band = z["band"].astype("float64")  # (M, 3, 35, 240)
    if band.shape[1:] != (3, 35, len(R.LON)):
        raise SystemExit(f"unexpected band shape {band.shape}")
    import gefs_drift

    dr = gefs_drift.from_env()  # operational drift the reforecast lacks
    if dr is not None:
        for k, t in enumerate(("olr", "u200", "u850")):  # gefs_daily.BANDF order
            band[:, k] -= dr.zonal(t)[:, R.BAND].mean(1)[None, :, None]
    n = band.shape[0]
    print(f"GEFS MJO {a.date}: {n} members, climatology {ref['mclim_source']!s}", flush=True)

    odates, or1, or2, map_now, _, _ = observed(init, ref, cache / "mjo_anl")
    print(
        f"  observed: {odates[0]} -> {odates[-1]} ({len(odates)} d), amp {np.hypot(or1[-1], or2[-1]):.2f}; "
        f"analyses {time.time() - t0:.0f} s",
        flush=True,
    )
    ch, anom = forecast_channels(band, init, ref, map_now)
    m1, m2 = channels_rmm(ch, ref)  # (M, 35)
    c, so, sf, dn, enso_note = enso_removal(ref, Path(a.site), init, odates)
    raw = np.hypot(m1.mean(0), m2.mean(0)), np.hypot(or1[-1], or2[-1])
    or1, or2, m1, m2 = or1 - so[:, 0], or2 - so[:, 1], m1 - sf[0], m2 - sf[1]
    amp = np.hypot(m1.mean(0), m2.mean(0))
    print(
        f"  ENSO removed: dN34 {dn:+.2f} K at the init, forecast moved {0 - sf[0]:+.2f}/{0 - sf[1]:+.2f} (RMM1/RMM2); "
        "mean-track amplitude raw -> ENSO removed, days 1/10/35: "
        + ", ".join(f"{raw[0][k]:.2f} -> {amp[k]:.2f}" for k in (0, 9, 34))
        + f"; observed {odates[-1]:%m-%d} {raw[1]:.2f} -> {np.hypot(or1[-1], or2[-1]):.2f}",
        flush=True,
    )
    norm = HERE / "data" / "mjo_wind_norm.json"
    if norm.exists():
        data.mkdir(parents=True, exist_ok=True)
        w1, w2 = channels_rmm_wind(ch, ref, json.loads(norm.read_text())["pc_wind_std"])
        write_wind_track(w1, w2, init, data / "gefs_rmm_wind.json")
    else:
        print("::warning::data/mjo_wind_norm.json missing: no wind-only track for the MJO page", flush=True)
    warn = track_check(ref_path(HIND_FILE, dirs), m1.mean(0), m2.mean(0), nino34_now(Path(a.site)), c)
    if warn:
        print("  mean track flagged: " + " ".join(warn), flush=True)
    names = [plot_phase(init, odates, or1, or2, m1, m2, n, out, enso_note, warn)]
    off = live_olr_offset(z, init, cache)
    print(f"  OLR offset vs reforecast (global mean, by week): {', '.join(f'{x:+.1f}' for x in off[::7])} W/m2")
    names.append(plot_hovmoller(init, anom.mean(0), off, out))
    names.append(skill(ref_path(HIND_FILE, dirs), yday(init), out, data, c, n_live=n))
    print(f"{len(names)} MJO figures in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
