#!/usr/bin/env python3
"""MJO diagnostics from the daily GEFS extended run - the GEPS page's "Tropics" group for the GEFS page (2026-09-26;
user: "the GEFS pages should have all the same maps/charts the GEPS one does"). Port of geps_subx/mjo_geps.py and
mjo_hindcast.py.

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
figures, and every pairing with BoM's RMM in the hindcast, uses that. (GEPS labels its day n as init + n; its day n
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
Checked against BoM's official RMM over 150 days (2023-09-28 .. 2024-02-24, the last BoM months): bivariate
correlation 0.993, RMSE 0.22, amplitude ratio 0.95 (0.990 / 0.957 with BoM shifted a day later / earlier). So the
observed track, its filter and the forecast are one model's one frame, and nothing leaves the S3 bucket the
forecast comes from (~2,000 small byte-ranged requests on a cold cache, a handful a day on a warm one: per-cycle band
means are kept in CACHE/mjo_anl/). GEPS takes its OLR track and both filter maps from ERA5 / the GMGSI proxy
(olr_channel.py, wind_map120.nc); here that would join a satellite-scale OLR track to a model-scale forecast.

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

Model climatology: the band means of the reforecast climatology (gefs-clim-v1 `band`, or the 15S-15N rows of v2's
daily fields), packed per 5-day centre into the reference file by build_gefs_mjo.py and read at the init day of
year exactly as gefs_live.clim_at reads the maps (linear between the two centres around it).

The ENSEMBLE MEAN is projected for the bold track; its amplitude damps with lead as members decohere - a shrinking
radius is loss of agreement, not a forecast that the MJO decays. The plume is every member projected alike.

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
import gefs_reforecast as R                                           # noqa: E402  (get, index, decode, grid, packing)
import gefs_daily as G                                               # noqa: E402  (to_half, box: the live reduction)

S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
REF_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-ref-v1/{}"
REF_FILE, HIND_FILE = "gefs_mjo_ref.npz", "gefs_mjo_hindcast.npz"
TAGS = ("olr", "u200", "u850")                   # order of the `band` arrays (gefs_daily.BANDF)
LON25 = np.arange(0.0, 360.0, 2.5)               # the WH04 EOF longitudes
TRACK_DAYS = 40                                  # observed track shown
MAP_DAYS = 120                                   # WH04 low-frequency filter
MAP_MIN = 110                                    # days of the 120 that must exist (olr_channel's min_periods)
MUTED = "#6f6b64"
# ECMWF extended range, for the reference marks - GEPS's values, drawn exactly as there (mjo_hindcast.py): published
# bivariate-correlation skill of the ECMWF reforecasts (Vitart 2017, QJRMS 143:2210, Fig. 2, and the ECMWF annual
# verification memoranda): COR 0.6 at about day 27 in the 2016 cycle and near day 30 by the 2020s; 0.5 around day 33.
ECMWF_REF = {"COR 0.6": (27, 30), "COR 0.5": (31, 34)}
PHASE_ANGLES = {1: 202.5, 2: 247.5, 3: 292.5, 4: 337.5, 5: 22.5, 6: 67.5, 7: 112.5, 8: 157.5}
REGION_LABELS = [("West. Hem.\nand Africa", -3.4, 0.0, 90), ("Indian Ocean", 0.0, -3.4, 0),
                 ("Maritime\nContinent", 3.4, 0.0, -90), ("Western Pacific", 0.0, 3.4, 0)]


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
    return np.stack([np.interp(LON25, lw, r) for r in flat]).reshape(a.shape[:-1] + (len(LON25),))


def centre_weights(doy: int):
    """gefs_live.clim_at's interpolation: linear between the two 5-day centres around the day of year."""
    centres = list(range(1, 366, 5))
    lo = max(c for c in centres if c <= doy)
    hi = lo + 5 if lo + 5 <= 361 else 1
    w = (doy - lo) / 5.0
    return [(lo, 1.0 - w), (hi, w)]


def model_clim(ref: dict, doy: int) -> np.ndarray:
    """Reforecast band climatology (3, 35, 240) [olr, u200, u850] at a day of year."""
    cen = list(ref["centres"])
    out = 0.0
    for c, w in centre_weights(doy):
        q = ref["mclim"][cen.index(c)]
        out = out + w * np.stack([R.unpack(t, q[i]) for i, t in enumerate(TAGS)])
    return out


def obs_clim(ref: dict, doy: int) -> np.ndarray:
    """WH04 observed climatology (3, 144) at a day of year (1..366)."""
    return ref["O"][:, min(doy, 366) - 1, :].astype("float64")


def yday(d: dt.date) -> int:
    return d.timetuple().tm_yday


def project(fo, f850, f200, ref):
    """Three-field WH04 projection -> (rmm1, rmm2), unit-variance normalised (mjo_geps._project)."""
    e1 = np.concatenate([ref["eof_u850"][0], ref["eof_u200"][0]])
    e2 = np.concatenate([ref["eof_u850"][1], ref["eof_u200"][1]])
    o1, o2 = ref["eof_olr"][0], ref["eof_olr"][1]
    comb = np.concatenate([f850, f200], axis=-1)
    return ((fo @ o1 + comb @ e1) / float(ref["pc_std"][0]),
            (fo @ o2 + comb @ e2) / float(ref["pc_std"][1]))


def channels_rmm(ch: np.ndarray, ref: dict):
    """(..., 3, 144) normalised channels [olr, u200, u850] -> rmm1, rmm2 of shape (...)."""
    return project(ch[..., 0, :], ch[..., 2, :], ch[..., 1, :], ref)


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
    except Exception:                                                  # noqa: BLE001
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
    cycles = sorted({dt.datetime(d.year, d.month, d.day) + dt.timedelta(hours=h)
                     for d in days for h in (0, 6, 12, 18, 24)})
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
    days before it (WH04, as BoM computes it), plus the filter map of the 120 days before the init for the forecast.

    -> (dates, rmm1, rmm2, map_now (3, 144) in anomaly units, analysis anomalies (n, 3, 144), days)"""
    last = init - dt.timedelta(days=1)
    first = last - dt.timedelta(days=TRACK_DAYS + MAP_DAYS - 1)
    days, A = analysis_days(first, last, cache)
    anom = to25(A) - np.stack([obs_clim(ref, yday(d)) for d in days])            # (n, 3, 144)

    def filt(i):                                                               # mean of the 120 days before day i
        w = anom[max(0, i - MAP_DAYS):i]
        ok = np.isfinite(w).all(-1)                                            # (k, 3)
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
        dates.append(days[i]); r1.append(float(a)); r2.append(float(b))
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
    M = model_clim(ref, yday(init))                                           # (3, 35, 240)
    anom = band.astype("float64") - M[None]
    shift = np.empty((3, 35, len(LON25)))
    for k in range(35):
        vdoy = yday(init + dt.timedelta(days=k))
        shift[:, k] = to25(model_clim(ref, vdoy)[:, 0]) - obs_clim(ref, vdoy)
    ch = (to25(anom) + shift[None] - map_now[None, :, None, :]) / ref["std"][None, :, None, None]
    return np.moveaxis(ch, 2, 1), anom


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
        ax.plot([-radius, radius], [-radius * np.tan(rad), radius * np.tan(rad)], color="0.6", lw=0.6, ls="--", zorder=1)
    for phase, centre in PHASE_ANGLES.items():
        rad = np.deg2rad(centre)
        ax.text(radius * 0.78 * np.cos(rad), radius * 0.78 * np.sin(rad), str(phase), ha="center", va="center",
                fontsize=10, color="0.35", fontweight="bold", zorder=2)
    for text, x, y, rot in REGION_LABELS:
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color="0.4", style="italic", rotation=rot,
                rotation_mode="anchor")
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in amp_rings:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="0.45", ls="--", lw=0.8, zorder=1, alpha=0.6)
        ax.text(r * np.cos(np.deg2rad(-45)), r * np.sin(np.deg2rad(-45)), str(r), ha="center", va="center", fontsize=8,
                color="0.4", zorder=2, bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.7))
    lim = radius + 0.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("RMM1", fontsize=11); ax.set_ylabel("RMM2", fontsize=11)


def plot_phase(init, odates, or1, or2, m1, m2, n_members, out: Path):
    """m1, m2: (members, 35) RMM per member; the bold track is the ensemble mean's (= the mean of the members' RMM,
    the projection being linear)."""
    plt = _plt()
    import matplotlib.colors as mcolors
    r1, r2 = m1.mean(0), m2.mean(0)
    L = np.arange(1, m1.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    fig.subplots_adjust(left=0.082, right=0.978, top=0.938, bottom=0.145)
    draw_phase_wheel(ax)
    ax.plot(or1, or2, color="0.55", lw=1.0, alpha=0.75, zorder=5)
    months = [d.strftime("%Y-%m") for d in odates]
    cmap_m = plt.get_cmap("tab10")
    for k, mo in enumerate(dict.fromkeys(months)):
        sel = np.array([m == mo for m in months])
        lab = dt.datetime.strptime(mo, "%Y-%m").strftime("%b %Y")
        ax.scatter(or1[sel], or2[sel], s=22, zorder=6, color=cmap_m(k % 10), edgecolor="0.3", linewidth=0.3,
                   label=f"Obs {lab}")
    ax.scatter(or1[-1], or2[-1], marker="*", s=210, zorder=8, color="#111111", label=f"analysis {odates[-1]:%b %d}")
    for k in range(m1.shape[0]):
        ax.plot(np.r_[or1[-1], m1[k]], np.r_[or2[-1], m2[k]], color="#4a7bb5", alpha=0.42, lw=0.9, zorder=3)
    cmap, norm = plt.get_cmap("plasma_r"), mcolors.Normalize(0, len(L) - 1)
    x, y = np.r_[or1[-1], r1], np.r_[or2[-1], r2]
    for i in range(len(x) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], color=cmap(norm(max(i - 1, 0))), lw=2.4, zorder=7)
    for k in range(4, len(L), 5):
        ax.scatter(r1[k], r2[k], s=42, color=cmap(norm(k)), edgecolor="0.25", linewidth=0.4, zorder=8)
    for k, (dx, dy) in zip((9, 19, 34), ((11, 9), (11, -13), (-13, 11))):
        if k < len(L):
            ax.annotate(f"day {L[k]}", (r1[k], r2[k]), textcoords="offset points", xytext=(dx, dy), fontsize=9,
                        fontweight="bold", zorder=10, color=cmap(norm(k)),
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.82))
    ax.scatter(r1[-1], r2[-1], marker="s", s=70, color=cmap(norm(len(L) - 1)), edgecolor="0.25", linewidth=0.5,
               zorder=9, label=f"GEFS day {L[-1]}")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
    ax.set_title(f"GEFS Ensemble RMM  —  Init: {init:%Y-%m-%d} 00Z", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.022, f"Three-field RMM (OLR + U850 + U200) · {n_members} members\n"
             "observed track: GEFS's own analysis, each day filtered by the 120 days before it",
             ha="center", va="bottom", fontsize=9, color=MUTED)
    p = out / "gefs_mjo_phase.webp"
    fig.savefig(p, dpi=110, facecolor="white", pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    amp = np.hypot(r1, r2)
    ma = np.hypot(m1, m2)
    print(f"  {p.name}  obs amp {np.hypot(or1[-1], or2[-1]):.2f} ({odates[-1]}), mean track day1 {amp[0]:.2f} -> "
          f"day35 {amp[-1]:.2f}; members day35 p10-p90 {np.percentile(ma[:, -1], 10):.2f}-"
          f"{np.percentile(ma[:, -1], 90):.2f}", flush=True)
    return p.name


def _hov_panel(ax, lon, L, F, cmap, lim, init, dates_on_right):
    m = ax.pcolormesh(lon, L, F, cmap=cmap, vmin=-lim, vmax=lim, shading="auto")
    ax.invert_yaxis()
    ax.set_xlim(lon[0] - 0.75, lon[-1] + 0.75)
    ax.set_xticks([0, 60, 120, 180, 240, 300])
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W"], fontsize=9)
    for x in (100, 150):                                                       # Indian | Maritime | Pacific
        ax.axvline(x, color="#00000033", lw=0.8, ls=(0, (4, 3)))
    for x, t in ((62, "Indian Ocean"), (125, "Maritime"), (215, "Pacific"), (310, "Atlantic")):
        ax.text(x, 1.4, t, ha="center", va="top", fontsize=8.5, color="#4a453f", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78))
    ax.tick_params(labelsize=9)
    if dates_on_right:
        ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim())
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
    U = anom_mean[2] * 1.94384                                                 # m/s -> knots
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
    for ax, m, lab in ((axO, mO, "OLR anomaly (W m⁻²)"), (axU, mU, "850 hPa u anomaly (knots)")):
        bb = ax.get_position()
        cax = fig.add_axes([bb.x0 + 0.10 * bb.width, 0.042, 0.80 * bb.width, 0.015])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(lab, fontsize=9, labelpad=2)
        cb.ax.tick_params(labelsize=8, pad=1.5)
    fig.suptitle("Tropical convection and low-level wind, 15°S–15°N\n"
                 f"GEFS extended · init {init:%Y-%m-%d}", fontsize=13, fontweight="bold", y=0.972, va="top")
    p = out / "gefs_mjo_hovmoller.webp"
    fig.savefig(p, dpi=110, facecolor="white", pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    print(f"  {p.name}  OLR ±{limO:.0f} W/m², u850 ±{limU:.0f} kt", flush=True)
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
        cor.append(float((a1 * b1 + a2 * b2).sum() / np.sqrt((a1 ** 2 + a2 ** 2).sum() * (b1 ** 2 + b2 ** 2).sum())))
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
    num = np.zeros((len(uy), F1.shape[1])); fa = np.zeros_like(num); ob = np.zeros_like(num)
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
    ok = np.isfinite(fm1).all(1) & np.isfinite(O1)                            # (S, L): all members and obs
    k = fm1.shape[1]
    cor, rmse = [], []
    for li in range(fm1.shape[2]):
        g = ok[:, li]
        a1, a2, b1, b2 = fm1[g, :, li], fm2[g, :, li], O1[g, li], O2[g, li]
        cov = ((a1.mean(1) * b1) + (a2.mean(1) * b2)).mean()
        p1 = (a1 ** 2 + a2 ** 2).mean()                                        # member power
        s1, s2 = a1.sum(1), a2.sum(1)
        pair = ((s1 ** 2 - (a1 ** 2).sum(1)) + (s2 ** 2 - (a2 ** 2).sum(1))).mean() / (k * (k - 1))
        pw = pair + (p1 - pair) / M
        po = (b1 ** 2 + b2 ** 2).mean()
        cor.append(float(cov / np.sqrt(pw * po)))
        rmse.append(float(np.sqrt(max(pw - 2 * cov + po, 0.0))))
    return cor, rmse


def skill(hind_path: Path, doy_now: int, out: Path, data: Path, n_live: int = 31):
    """Score the reforecast pairs for every subset, write data/gefs_mjo_skill.json and gefs_mjo_skill.webp."""
    plt = _plt()
    t0 = time.time()
    z = np.load(hind_path)
    starts = [dt.date.fromisoformat(s) for s in z["starts"]]
    fm1, fm2 = z["fm1"].astype("float64"), z["fm2"].astype("float64")          # (S, members, L)
    have = np.isfinite(fm1).any(1)                                            # (S, L): any member
    with np.errstate(invalid="ignore"):
        f1 = np.where(have, np.nansum(fm1, 1) / np.maximum(np.isfinite(fm1).sum(1), 1), np.nan)   # ensemble mean
        f2 = np.where(have, np.nansum(fm2, 1) / np.maximum(np.isfinite(fm2).sum(1), 1), np.nan)   # (projection linear)
    o1, o2 = z["o1"].astype("float64"), z["o2"].astype("float64")
    amp0 = z["amp0"].astype("float64")
    ok = np.isfinite(f1).all(1)
    L = np.arange(1, f1.shape[1] + 1)
    mon = np.array([s.month for s in starts]); doy = np.array([yday(s) for s in starts])
    years = np.array([s.year for s in starts])
    dn = ((doy - doy_now + 183) % 366) - 183
    subsets = {"all": ok, "strong initial (amp ≥ 1)": ok & (amp0 >= 1.0),
               f"now ±45 d (doy {doy_now})": ok & (np.abs(dn) <= 45)}
    subsets.update({s: ok & np.isin(mon, m) for s, m in SEASONS.items()})
    res = {k: _curves(f1[v], f2[v], o1[v], o2[v]) for k, v in subsets.items()}
    full = ok & np.isfinite(fm1).all((1, 2))
    live, live_rmse = _scaled_members(fm1[full], fm2[full], o1[full], o2[full], n_live)
    four, four_rmse = _scaled_members(fm1[full], fm2[full], o1[full], o2[full], fm1.shape[1])
    info = {"generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "leads": [int(x) for x in L],
            "valid_date": "day d is valid on init + (d - 1) days (day 1 = the init date)",
            "doy_now": int(doy_now), "ecmwf_reference": ECMWF_REF,
            "hindcast": str(z["note"]), "n_members_hindcast": int(fm1.shape[1]), "subsets": {},
            f"cor_{n_live}_members_estimate": {"cor": live, "rmse": live_rmse, "day_cor_0.6": cross(live, 0.6),
                                               "day_cor_0.5": cross(live, 0.5), "n_starts": int(full.sum()),
                                               "cor_4_members_same_starts": four, "rmse_4_members_same_starts": four_rmse,
                                               "method": "exchangeable-member scaling of the ensemble-mean power "
                                                         "(a + b/M) from the 4 hindcast members"}}
    for k, v in res.items():
        m = subsets[k]
        info["subsets"][k] = {**v, "day_cor_0.6": cross(v["cor"], 0.6), "day_cor_0.5": cross(v["cor"], 0.5),
                              "day_cor_0.6_ci90": _boot_cross(f1[m], f2[m], o1[m], o2[m], years[m], 0.6),
                              "day_cor_0.5_ci90": _boot_cross(f1[m], f2[m], o1[m], o2[m], years[m], 0.5),
                              "n_starts": int(m.sum())}
        x = info["subsets"][k]
        fmt = lambda c: f"{c[0]:.1f}-{c[1]:.1f}" if c else "-"                  # noqa: E731
        print(f"  {k:28s} n {int(m.sum()):4d}  COR d5 {v['cor'][4]:.2f} d10 {v['cor'][9]:.2f} d15 {v['cor'][14]:.2f} "
              f"d20 {v['cor'][19]:.2f} d25 {v['cor'][24]:.2f}  0.6 at {_fd(x['day_cor_0.6'])} ({fmt(x['day_cor_0.6_ci90'])})"
              f"  0.5 at {_fd(x['day_cor_0.5'])} ({fmt(x['day_cor_0.5_ci90'])})", flush=True)
    print(f"  {n_live}-member estimate: COR 0.6 at {_fd(cross(live, 0.6))}, 0.5 at {_fd(cross(live, 0.5))} "
          f"(4 members on the same starts {_fd(cross(four, 0.6))} / {_fd(cross(four, 0.5))})", flush=True)
    data.mkdir(parents=True, exist_ok=True)
    (data / "gefs_mjo_skill.json").write_text(json.dumps(info))

    fig = plt.figure(figsize=(14.5, 5.4))
    now = f"now ±45 d (doy {doy_now})"
    cols = {"all": "#16335c", "DJF": "#2c5fa8", "MAM": "#3f9b6a", "JJA": "#c8781e", "SON": "#b4453c",
            "strong initial (amp ≥ 1)": "#6a3d9a", now: "#111111"}
    for pi, (key, ylab, title) in enumerate((("cor", "bivariate correlation", "Skill: bivariate correlation vs BoM RMM"),
                                             ("rmse", "bivariate RMSE", "Error"),
                                             ("amp_ratio", "forecast / observed amplitude", "Amplitude"))):
        ax = fig.add_axes([0.05 + pi * 0.325, 0.19, 0.27, 0.67])
        for k, v in res.items():
            lw = 2.6 if k == "all" else (2.2 if k.startswith("now") else 1.4)
            ls = "-" if not k.startswith("strong") else (0, (4, 2))
            ax.plot(L, v[key], color=cols[k], lw=lw, ls=ls, label=k)
        if key == "cor":
            ax.plot(L, live, color=cols["all"], lw=1.3, ls=":", label=f"all, {n_live}-member estimate")
            ax.axhline(0.6, color="#8a8680", lw=0.8, ls=":"); ax.axhline(0.5, color="#8a8680", lw=0.8, ls=":")
            for lab, (d0, d1) in ECMWF_REF.items():
                yv = float(lab.split()[-1])
                ax.plot([d0, d1], [yv, yv], color="#8a1d1d", lw=4, alpha=0.6, solid_capstyle="butt")
                ax.text(d1, yv - 0.02, f"ECMWF, published:\n{lab} at day {d0}–{d1}", fontsize=7.0,
                        color="#8a1d1d", va="top", ha="right")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7.6, loc="lower left", framealpha=0.92)
        elif key == "rmse":
            ax.plot(L, live_rmse, color=cols["all"], lw=1.3, ls=":")
            ax.axhline(np.sqrt(2.0), color="#8a8680", lw=0.8, ls=":")
            ax.text(1.5, np.sqrt(2.0) + 0.02, "√2 = climatology forecast", fontsize=7, ha="left", color="#5a5650")
        elif key == "amp_ratio":
            ax.axhline(1.0, color="#8a8680", lw=0.8, ls=":")
            ax.set_ylim(0.4, max(1.2, 0.05 + max(max(v["amp_ratio"]) for v in res.values())))
        ax.set_xlim(1, 35); ax.set_xlabel("forecast day", fontsize=9); ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        ax.grid(alpha=0.25, lw=0.5); ax.tick_params(labelsize=8.5)
    a = info["subsets"]["all"]
    ci = a["day_cor_0.6_ci90"]
    y0, y1 = min(years[ok]), max(years[ok])
    fig.suptitle(f"GEFS MJO skill from the GEFSv12 reforecast, {y0}–{y1} · {int(ok.sum())} starts, ensemble mean · "
                 f"COR 0.6 at day {_fd(a['day_cor_0.6'], 0)}" + (f" (90%: {ci[0]:.0f}–{ci[1]:.0f})" if ci else "")
                 + f", 0.5 at day {_fd(a['day_cor_0.5'], 0)}", fontsize=12.5, fontweight="bold", y=0.985, va="top")
    fig.text(0.5, 0.012, "same projection as the live RMM (three-field WH04, lead-matched de-drift, 120-day mean "
             "removed), verified against BoM RMM on the valid date (day 1 = the start date);\n"
             f"weekly starts, hindcast mean of {fm1.shape[1]} members; the title's bracket is the 90% interval with "
             f"whole years resampled; dotted: the same starts scaled to the {n_live}-member operational mean\n"
             f"(members taken as exchangeable), COR 0.6 at day {_fd(cross(live, 0.6), 0)}; ECMWF marks are published "
             "reforecast values (Vitart 2017, ECMWF verification memoranda), not a like-for-like computation",
             ha="center", va="bottom", fontsize=7.8, color="#5a5650")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gefs_mjo_skill.webp", dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"  gefs_mjo_skill.webp  ({time.time() - t0:.0f} s)", flush=True)
    return "gefs_mjo_skill.webp"


# ── driver ────────────────────────────────────────────────────────────────────────────────────────────────────
def live_olr_offset(z, init, cache: Path) -> np.ndarray:
    """The live run's uniform OLR offset against the reforecast, per lead, for the Hovmoller only (GEPS removes its
    ~7.6 W/m2): the global-mean weekly OLR anomaly of the ensemble mean against the v1 weekly climatology (the file
    gefs_live.py has already put in the cache, else none). ~1.4 W/m2 on 2026-09-25 (GEPS: 7.6); in the RMM it would
    be 0.09 sd of uniform OLR, which projects to about 0.01 - so the RMM keeps it, the Hovmoller does not."""
    try:
        import gefs_live as GL
        C = GL.clim_at(init, cache)
        wt = np.cos(np.deg2rad(R.LAT))[:, None] * np.ones((1, len(R.LON)))
        wk = z["w_olr"].mean(0) - C["w_olr"]
        g = np.array([(w * wt).sum() / wt.sum() for w in wk])
        return np.concatenate([np.full(b - a + 1, g[i]) for i, (a, b) in enumerate(R.WEEKS)])
    except Exception as e:                                             # noqa: BLE001
        print(f"  OLR offset unavailable ({str(e)[:60]}); none removed", flush=True)
        return np.zeros(35)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--date", required=True)
    ap.add_argument("--site", default=str(HERE.parents[1])); ap.add_argument("--cache", default="/tmp/gefs_work/clim")
    ap.add_argument("--ref", default=None, help="directory holding gefs_mjo_ref.npz / gefs_mjo_hindcast.npz")
    a = ap.parse_args()
    t0 = time.time()
    init = dt.date(int(a.date[:4]), int(a.date[4:6]), int(a.date[6:]))
    cache = Path(a.cache)
    dirs = ([a.ref] if a.ref else []) + [str(cache)]
    out = Path(a.site) / "assets" / "gefs"; data = out / "data"
    out.mkdir(parents=True, exist_ok=True)
    ref = load_ref(dirs)
    z = np.load(a.npz)
    band = z["band"].astype("float64")                                     # (M, 3, 35, 240)
    if band.shape[1:] != (3, 35, len(R.LON)):
        raise SystemExit(f"unexpected band shape {band.shape}")
    import gefs_drift
    dr = gefs_drift.from_env()                                             # operational drift the reforecast lacks
    if dr is not None:
        for k, t in enumerate(("olr", "u200", "u850")):                    # gefs_daily.BANDF order
            band[:, k] -= dr.zonal(t)[:, R.BAND].mean(1)[None, :, None]
    n = band.shape[0]
    print(f"GEFS MJO {a.date}: {n} members, climatology {str(ref['mclim_source'])}", flush=True)

    odates, or1, or2, map_now, _, _ = observed(init, ref, cache / "mjo_anl")
    print(f"  observed: {odates[0]} -> {odates[-1]} ({len(odates)} d), amp {np.hypot(or1[-1], or2[-1]):.2f}; "
          f"analyses {time.time() - t0:.0f} s", flush=True)
    ch, anom = forecast_channels(band, init, ref, map_now)
    m1, m2 = channels_rmm(ch, ref)                                            # (M, 35)
    names = [plot_phase(init, odates, or1, or2, m1, m2, n, out)]
    off = live_olr_offset(z, init, cache)
    print(f"  OLR offset vs reforecast (global mean, by week): {', '.join(f'{x:+.1f}' for x in off[::7])} W/m2")
    names.append(plot_hovmoller(init, anom.mean(0), off, out))
    names.append(skill(ref_path(HIND_FILE, dirs), yday(init), out, data, n_live=n))
    print(f"{len(names)} MJO figures in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
