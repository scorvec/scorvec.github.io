#!/usr/bin/env python3
"""Teleconnection indices from every GEFS member, days 1-35 - plumes, weekly probabilities, change vs the previous run,
and an observed tail. The GEPS page's telecon_geps.py ported to the GEFS extended page (2026-09-26).

The projectors are the GEPS page's (CPC-style patterns on NCEP/NCAR R1 1991-2020 - regression on CPC's monthly NH
indices, SLP EOF1 for the AO/AAO, PSL's EPO/WPO boxes, CPC's SOI/EQSOI boxes, NAM at 100 hPa as the demeaned EOF1),
shipped in gefs_patterns_ref.npz. They are applied on the same 2.5 deg grid and in the same units to:

  members    every GEFS member's de-drifted daily anomaly (gefs_patterns.member_anoms: forecast minus the lead-matched
             GEFSv12 reforecast climatology at the init day of year, re-based to 1991-2020 with the ERA5 period shift),
             so the index carries no model drift. NAM 100 is de-drifted the same way (GEPS could not: its hindcast has
             no 100 hPa) and calibrated like the rest.
  observed   GEFS's own control analyses for the last 45 days, each day the 12Z + next 00Z pair - exactly a forecast
             day's samples - against the reforecast climatology at lead day 1 (gefs_patterns.observed_anoms). The
             GEPS page takes the 0/6/12/18 h daily mean for sea-level pressure to cancel the atmospheric tides against
             a tide-free climatology; GEFS's reforecast climatology is 00Z/12Z-sampled and so carries the semidiurnal
             tide S2 itself, and only the matching 00Z/12Z pair cancels it: a four-sample tail would sit +1.0 sd
             (SOI) and +1.3 sd (EQSOI) off, measured on 30 days of GEFS analyses (gefs_patterns.py).
  CPC        CPC's published daily AO/NAO/PNA/AAO, PSL's daily EPO/WPO and the Long Paddock daily SOI as crosses on
             the tail, CPC's monthly SOI/EQSOI as bars, all rescaled to seasonal sd (telecon_geps.py verbatim).

Calibration (gefs_patterns_calib.npz, build_gefs_patterns_hindcast.py): the GEFSv12 reforecast 2000-2019 (Wednesday
starts, 4 members) scored with the same projectors against NCEP/NCAR R1 on the valid dates; for the starts within
+-45 days of the init day of year, per index and week, the anomaly correlation r, the regression obs = a*fcst + b and
its residual sd s. The live ensemble-mean weekly index m becomes P(week >= +0.5) = 1 - Phi((0.5 - a m - b) / s). A
week counts as skilful only when r >= 0.25 AND r is significant at 5% (one-sided t-test on the effective sample size
n (1 - r1 r2) / (1 + r1 r2), r1 r2 = the week-to-week lag-1 autocorrelations of the forecast and observed series of
consecutive weekly starts, Bretherton et al. 1999); otherwise the cell is hatched "no skill".

Units: seasonal standard deviations of the DAILY index over 1991-2020 (std within +-15 d of the day of year).

    python gefs_telecon.py --npz /tmp/gefs_work/gefs2_20260925.npz --date 20260925 --site . --cache /tmp/gefs_work
    -> assets/gefs/gefs_telecon_series.webp, gefs_telecon_plume_<id>.webp, gefs_telecon_probs.webp,
       assets/gefs/data/gefs_telecon.json; run archive gefs_telecon_<date>.npz (+ the analysis tail files)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import textwrap
from math import erf, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_patterns as GP                                           # noqa: E402

SKILL_MIN = 0.25                # hindcast anomaly correlation below this = "no skill", hatched
P_MAX = 0.05                    # ... and so is a correlation not significant at this level
THR = 0.5                       # +-0.5 seasonal sd = positive / negative phase
TAIL_DAYS = GP.TAIL_DAYS
FIT_COLS = ["n", "r", "a", "b", "s", "obs_sd", "fc_sd", "n_eff", "p"]
CPC_URL = {"ao": "cwlinks/norm.daily.ao.cdas.z1000.19500101_current.csv",
           "nao": "cwlinks/norm.daily.nao.cdas.z500.19500101_current.csv",
           "pna": "cwlinks/norm.daily.pna.cdas.z500.19500101_current.csv",
           "aao": "cwlinks/norm.daily.aao.cdas.z700.19790101_current.csv"}
PSL_URL = "https://downloads.psl.noaa.gov/Public/map/teleconnections/{sid}.reanalysis.t10trunc.1948-present.txt"
CPC_MONTHLY_URL = {"soi": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
                   "eqsoi": "https://www.cpc.ncep.noaa.gov/data/indices/reqsoi.for"}
LONGPADDOCK_URL = ("https://data.longpaddock.qld.gov.au/SeasonalClimateOutlook/SouthernOscillationIndex/"
                   "SOIDataFiles/DailySOI1887-1989Base.txt")


def phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ── published indices (telecon_geps / build_telecon_patterns parsers without pandas) ─────────────────────────
def _series(pairs):
    pairs = sorted((d, v) for d, v in pairs if np.isfinite(v))
    return (np.array([p[0] for p in pairs], dtype="datetime64[D]"), np.array([p[1] for p in pairs], dtype="float64"))


def cpc_daily(path: Path):
    """CPC daily csv: year, month, day, value; -999 missing."""
    if not path.exists():
        return None
    out = []
    for ln in path.read_text().splitlines()[1:]:
        p = [x.strip() for x in ln.split(",")]
        try:
            v = float(p[3])
            out.append((dt.date(int(p[0]), int(p[1]), int(p[2])), np.nan if v <= -999 else v))
        except (ValueError, IndexError):
            continue
    return _series(out) if out else None


def psl_daily(path: Path):
    """PSL daily EPO/WPO: 'yyyy mm dd value' in metres."""
    if not path.exists():
        return None
    out = []
    for ln in path.read_text().splitlines():
        p = ln.split()
        try:
            out.append((dt.date(int(p[0]), int(p[1]), int(p[2])), float(p[3])))
        except (ValueError, IndexError):
            continue
    return _series(out) if out else None


def cpc_soi_monthly(path: Path):
    """CPC 'soi', the STANDARDIZED table; -999.9 missing; fixed width ('-1.8-999.9' is two numbers)."""
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    k = [i for i, ln in enumerate(lines) if "STANDARDIZED" in ln]
    out = []
    for ln in lines[k[0]:] if k else []:
        m = re.match(r"^\s*(\d{4})(.*)$", ln)
        if not m:
            continue
        v = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(2))]
        if len(v) == 12:
            out += [(dt.date(int(m.group(1)), mo, 1), np.nan if x <= -999 else x) for mo, x in enumerate(v, 1)]
    return _series(out) if out else None


def cpc_eqsoi_monthly(path: Path):
    """CPC reqsoi.for (standardized equatorial SOI); 999.9 missing."""
    if not path.exists():
        return None
    out = []
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) == 13 and p[0].isdigit():
            out += [(dt.date(int(p[0]), mo, 1), np.nan if float(x) > 999 else float(x)) for mo, x in enumerate(p[1:], 1)]
    return _series(out) if out else None


def longpaddock_daily_soi(path: Path):
    """Long Paddock daily SOI: 'Year Day Tahiti Darwin SOI' (header), Troup x10 on the 1887-1989 base."""
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    head = [c.strip().lower() for c in lines[0].split()]
    iy, iday, isoi = head.index("year"), head.index("day"), head.index("soi")
    out = []
    for ln in lines[1:]:
        p = ln.split()
        try:
            out.append((dt.date(int(p[iy]), 1, 1) + dt.timedelta(days=int(p[iday]) - 1), float(p[isoi])))
        except (ValueError, IndexError):
            continue
    return _series(out) if out else None


def _doys(t):
    return np.array([GP.doy(x) for x in t.astype(object)])


def _std_9120(t, v):
    sel = (t >= np.datetime64("1991-01-01")) & (t <= np.datetime64("2020-12-31"))
    return float(np.std(v[sel]))


def published(ref: GP.Ref, cache: Path, base: dt.date):
    """{sid: (dates, values)} daily published indices and {sid_monthly: [(month start, value)]}, in seasonal sd."""
    cpc = {}
    d = Path(cache) / "published"
    for sid, u in CPC_URL.items():
        c = cpc_daily(GP.refresh(f"https://ftp.cpc.ncep.noaa.gov/{u}", d / f"{sid}.daily.csv"))
        if c is not None and sid in ref.pat:                       # CPC publishes annually standardized values
            cpc[sid] = (c[0], c[1] * ref.to_seasonal(sid, _doys(c[0])))
    for sid in ("epo", "wpo"):
        ps = psl_daily(GP.refresh(PSL_URL.format(sid=sid), d / f"{sid}.daily.txt"))
        if ps is not None and sid in ref.pat:                      # PSL publishes metres: annual sd, then seasonal
            cpc[sid] = (ps[0], ps[1] / _std_9120(*ps) * ref.to_seasonal(sid, _doys(ps[0])))
    for sid, url in CPC_MONTHLY_URL.items():
        if sid not in ref.pat:
            continue
        f = GP.refresh(url, d / url.rsplit("/", 1)[-1])
        cm = cpc_soi_monthly(f) if sid == "soi" else cpc_eqsoi_monthly(f)
        if cm is None:
            continue
        keep = cm[0] >= np.datetime64(base - dt.timedelta(days=TAIL_DAYS + 31))
        t0 = cm[0][keep].astype(object)
        mid = np.array([GP.doy(x + dt.timedelta(days=14)) for x in t0])
        cpc[f"{sid}_monthly"] = list(zip(t0, cm[1][keep] * ref.pat[sid]["monthly_sd"] * ref.to_seasonal(sid, mid)))
    if "soi" in ref.pat:
        lp = longpaddock_daily_soi(GP.refresh(LONGPADDOCK_URL, d / "DailySOI1887-1989Base.txt"))
        if lp is not None:
            cpc["soi"] = (lp[0], lp[1] / _std_9120(*lp) * ref.to_seasonal("soi", _doys(lp[0])))
    return cpc


# ── forecast, previous run, observed, calibration ─────────────────────────────────────────────────────────────
def observed(ref: GP.Ref, base: dt.date, runs: Path):
    """{sid: (dates, values)} daily indices of the analysis tail."""
    fields, ts = GP.observed_anoms(base, ref, runs)
    if not ts:
        return {}
    mo = np.array([d.month for d in ts]); dy = np.array([GP.doy(d) for d in ts])
    out = {}
    for sid in ref.ids:
        tag = next(t for t in GP.TAGS if GP.FIELD[t] == ref.pat[sid]["field"])
        if tag in fields:
            out[sid] = (list(ts), ref.index(sid, fields[tag], mo, dy))
    return out


def calibration(ref: GP.Ref, base: dt.date):
    """{sid: [row per week or None]} from the calibration file's fit table at the init day of year."""
    if ref.calib is None or "fit" not in ref.calib.files:
        return {}
    ids = ref.calib_meta["fit_ids"]
    F = ref.calib["fit"]                                             # (index, 366 doy, 5 weeks, FIT_COLS)
    cols = ref.calib_meta.get("fit_cols", FIT_COLS)
    out = {}
    for i, sid in enumerate(ids):
        rows = []
        for w in range(len(GP.WEEKS)):
            v = dict(zip(cols, F[i, GP.doy(base) - 1, w].tolist()))
            if not np.isfinite(v["r"]) or v["n"] < 30:
                rows.append(None); continue
            rows.append(dict(n=int(v["n"]), r=round(v["r"], 2), a=round(v["a"], 2), b=round(v["b"], 2),
                             s=round(v["s"], 2), obs_sd=round(v["obs_sd"], 2), fc_sd=round(v["fc_sd"], 2),
                             n_eff=int(round(v["n_eff"])), p=round(v["p"], 4)))
        out[sid] = rows
    return out


def weekly_stats(t, M, base, prev=None, cal=None):
    """telecon_geps.weekly_stats with GEFS valid dates: week k covers init + (7k-7) .. init + (7k-1)."""
    rows = []
    t = np.array(t, dtype="datetime64[D]")
    for k, (w0, w1) in enumerate(GP.WEEKS):
        d0, d1 = base + dt.timedelta(days=w0 - 1), base + dt.timedelta(days=w1 - 1)
        sel = (t >= np.datetime64(d0)) & (t <= np.datetime64(d1)) & np.isfinite(M).all(1)
        if not sel.any():
            continue
        wm = M[sel].mean(0)
        row = dict(week=k + 1, d0=d0.isoformat(), d1=d1.isoformat(), ndays=int(sel.sum()),
                   mean=round(float(wm.mean()), 2),
                   p10=round(float(np.percentile(wm, 10)), 2), p25=round(float(np.percentile(wm, 25)), 2),
                   p50=round(float(np.percentile(wm, 50)), 2), p75=round(float(np.percentile(wm, 75)), 2),
                   p90=round(float(np.percentile(wm, 90)), 2),
                   p_neg=round(float((wm <= -THR).mean()), 3), p_pos=round(float((wm >= THR).mean()), 3),
                   p_strong_neg=round(float((wm <= -1.0).mean()), 3), p_strong_pos=round(float((wm >= 1.0).mean()), 3))
        if prev is not None:
            pt, pM = prev
            pt = np.array(pt, dtype="datetime64[D]")
            psel = (pt >= np.datetime64(d0)) & (pt <= np.datetime64(d1)) & np.isfinite(pM).all(1)
            if psel.any():
                pw = pM[psel].mean(0)
                row.update(prev_mean=round(float(pw.mean()), 2), prev_ndays=int(psel.sum()),
                           prev_p_neg=round(float((pw <= -THR).mean()), 3), prev_p_pos=round(float((pw >= THR).mean()), 3))
        if cal and k < len(cal) and cal[k]:
            c = cal[k]
            mu = c["a"] * row["mean"] + c["b"]
            row.update(skill_r=c["r"], skill_p=c["p"], no_skill=bool(c["r"] < SKILL_MIN or c["p"] > P_MAX),
                       cal_n=c["n"], cal_neff=c["n_eff"], cal_a=c["a"], cal_b=c["b"], cal_s=c["s"],
                       cal_mean=round(float(mu), 2),
                       cal_p_pos=round(1.0 - phi((THR - mu) / c["s"]), 3),
                       cal_p_neg=round(phi((-THR - mu) / c["s"]), 3),
                       cal_p_strong_pos=round(1.0 - phi((1.0 - mu) / c["s"]), 3),
                       cal_p_strong_neg=round(phi((-1.0 - mu) / c["s"]), 3))
        rows.append(row)
    return rows


# ── figures (telecon_geps.py geometry, colours and wording) ───────────────────────────────────────────────────
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plume_panel(ax, ref, sid, base, prev_date, fc, pfc, obs, cpc, n_members, big=False):
    import matplotlib.dates as mdates
    t, M = fc[sid]
    m = ref.pat[sid]
    fs = 1.25 if big else 1.0
    ax.axhspan(-THR, THR, color="#000", alpha=0.035, lw=0)
    ax.fill_between(t, np.nanpercentile(M, 10, 1), np.nanpercentile(M, 90, 1), color="#2c4a72", alpha=0.16, lw=0,
                    label="p10–p90")
    ax.fill_between(t, np.nanpercentile(M, 25, 1), np.nanpercentile(M, 75, 1), color="#2c4a72", alpha=0.28, lw=0,
                    label="p25–p75")
    ax.plot(t, np.nanmean(M, 1), color="#16335c", lw=2.2 * fs, label="ensemble mean")
    if pfc and sid in pfc:
        pt, pM = pfc[sid]
        keep = [i for i, x in enumerate(pt) if t[0] <= x <= t[-1]]
        if keep:
            ax.plot([pt[i] for i in keep], np.nanmean(pM, 1)[keep], ls=(0, (4, 2)), color="#c8781e", lw=1.7 * fs,
                    label=f"previous run mean, init {GP.parse_date(prev_date):%b %d}")
    lo_x = base - dt.timedelta(days=TAIL_DAYS)
    ov = []
    if sid in obs and len(obs[sid][0]):
        ot, ovv = obs[sid]
        ax.plot(ot, ovv, color="#3f3a33", lw=1.5 * fs, label="GEFS analyses (12Z + 00Z), same projection")
        ax.plot(ot, ovv, "o", color="#3f3a33", ms=2.4 * fs)
        ov = ovv[np.isfinite(ovv)]
    if sid in cpc and cpc[sid] is not None:
        ct, cv = cpc[sid]
        keep = (ct >= np.datetime64(lo_x)) & (ct <= np.datetime64(base + dt.timedelta(days=2)))
        if keep.any():
            ax.plot(ct[keep].astype(object), cv[keep], "x", color="#8a1d1d", ms=3.6 * fs, mew=1.0,
                    label=f"published daily index ({m.get('daily_reference', 'CPC; PSL for EPO/WPO')})")
    if f"{sid}_monthly" in cpc:
        first = True
        for t0, v in cpc[f"{sid}_monthly"]:
            nxt = (t0.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
            t1 = min(nxt - dt.timedelta(days=1), base - dt.timedelta(days=1))
            if t1 < lo_x or t0 > base:
                continue
            ax.plot([t0, t1], [v, v], color="#8a1d1d", lw=1.8 * fs, solid_capstyle="butt",
                    label="published monthly index (CPC), in daily-index sd" if first else None)
            first = False
    ax.axhline(0, color="#555", lw=0.9)
    ax.axvline(dt.datetime.combine(base, dt.time()) - dt.timedelta(hours=12), color="#8a8680", lw=0.8, ls=":")
    ax.set_xlim(lo_x, t[-1] + dt.timedelta(days=1))
    ylim = max(2.0, float(np.nanmax(np.abs(np.r_[np.nanpercentile(M, [5, 95], 1).ravel(), ov if len(ov) else 0]))) * 1.1)
    ax.set_ylim(-ylim, ylim)
    ax.set_ylabel("seasonal sd", fontsize=8.5 * fs)
    ref_ = m.get("reference", "CPC")
    rt = (f"r vs {ref_} daily {m['r_daily']:+.2f}" if "r_daily" in m else
          f"r vs CPC monthly {m['r_monthly']:+.2f}"
          + (f" · vs {m['daily_reference']} daily {m['r_daily_ref']:+.2f}" if "r_daily_ref" in m else "")
          if "r_monthly" in m else "no external reference")
    ax.set_title(f"{sid.upper()} — {m['name']}", fontsize=9.6 * fs, fontweight="bold", loc="left", pad=4)
    ax.text(0.985, 0.955, rt, transform=ax.transAxes, ha="right", va="top", fontsize=6.9 * fs, color=GP.MUTED,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
    ax.tick_params(labelsize=8 * fs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10 if not big else 5))
    ax.grid(alpha=0.25, lw=0.5)


BASIS = ("forecast de-drifted against the lead-matched GEFSv12 reforecast climatology (2000–2019), re-based to "
         "1991–2020 with the ERA5 2001–2020 → 1991–2020 shift")


def plot_plumes(ref, base, prev_date, fc, pfc, obs, cpc, n_members, out: Path):
    plt = _plt()
    for sid in [i for i in ref.ids if i in fc]:
        fig = plt.figure(figsize=(12.4, 5.6))
        ax = fig.add_axes([0.06, 0.13, 0.925, 0.77])
        _plume_panel(ax, ref, sid, base, prev_date, fc, pfc, obs, cpc, n_members, big=True)
        ax.legend(fontsize=8, ncol=3, loc="lower left", framealpha=0.92)
        fig.suptitle(f"GEFS extended ensemble, {n_members} members · init {base:%Y-%m-%d} 00Z"
                     + (f" · dashed: previous run {GP.parse_date(prev_date):%b %d}" if prev_date else "") + GP.TEST_NOTE,
                     fontsize=10.5, y=0.985, va="top", color=GP.INK)
        note = ("seasonal standard deviations of the daily index (std within ±15 d of the date, 1991–2020); shaded band "
                "±0.5 sd = neutral; " + BASIS + (" (100 hPa: shift built the same way from ERA5)"
                                                   if ref.pat[sid]["field"] == "z100" else ""))
        fig.text(0.5, 0.012, "\n".join(textwrap.wrap(note, 190)), ha="center", va="bottom", fontsize=8, color=GP.MUTED)
        GP.save_webp(fig, out / f"gefs_telecon_plume_{sid}.webp")
        plt.close(fig)
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    for sid in [i for i in ref.ids if i not in fc]:                  # out of season: say so, never leave a stale plume
        fig = plt.figure(figsize=(12.4, 5.6))
        ax = fig.add_axes([0.06, 0.13, 0.925, 0.77]); ax.axis("off")
        ax.set_title(f"{sid.upper()} — {ref.pat[sid]['name']}", fontsize=12, fontweight="bold", loc="left", pad=4)
        dm = sorted(ref.defined[sid])
        ax.text(0.5, 0.55, f"CPC defines this pattern only in {', '.join(months[m - 1] for m in dm)};\nnone of this "
                f"forecast's valid days ({base:%b %d} – {base + dt.timedelta(days=GP.DAYS - 1):%b %d}) falls in them, "
                "so there is no index to plot.", ha="center", va="center", fontsize=12, color=GP.INK,
                transform=ax.transAxes)
        fig.suptitle(f"GEFS extended ensemble, {n_members} members · init {base:%Y-%m-%d} 00Z" + GP.TEST_NOTE,
                     fontsize=10.5, y=0.985, va="top", color=GP.INK)
        GP.save_webp(fig, out / f"gefs_telecon_plume_{sid}.webp")
        plt.close(fig)
    print(f"  gefs_telecon_plume_*.webp ({sum(1 for i in ref.ids if i in fc)} indices, "
          f"{sum(1 for i in ref.ids if i not in fc)} out of season)", flush=True)


def plot_series(ref, base, prev_date, fc, pfc, obs, cpc, n_members, out: Path):
    plt = _plt()
    ids = [i for i in ref.ids if i in fc]
    ncol = 4
    nrow = (len(ids) + ncol - 1) // ncol
    fig = plt.figure(figsize=(17.0, 3.15 * nrow + 0.9))
    L0, R0, T0, B0 = 0.038, 0.995, 0.905, 0.06
    PW, PH = (R0 - L0) / ncol, (T0 - B0) / nrow
    for i, sid in enumerate(ids):
        col, row = i % ncol, i // ncol
        ax = fig.add_axes([L0 + col * PW + 0.028, T0 - (row + 1) * PH + 0.055, PW * 0.86, PH * 0.74])
        _plume_panel(ax, ref, sid, base, prev_date, fc, pfc, obs, cpc, n_members)
        if i == 0:
            ax.legend(fontsize=6.4, ncol=2, loc="lower left", framealpha=0.92)
    fig.suptitle(f"Teleconnection indices — GEFS extended ensemble, {n_members} members · init {base:%Y-%m-%d} 00Z"
                 + (f"  ·  dashed: previous run {GP.parse_date(prev_date):%b %d}" if prev_date else "") + GP.TEST_NOTE,
                 fontsize=13.5, fontweight="bold", y=0.985, va="top")
    note = ("seasonal standard deviations of the daily index (std within ±15 d of the date, 1991–2020; NCEP/NCAR R1, "
            "CPC-style patterns); shaded band ±0.5 sd = neutral; " + BASIS + " (NAM 100 included); tail: GEFS control "
            "analyses, each day the 12Z + next 00Z pair, anomalised the same way")
    fig.text(0.5, 0.008, "\n".join(textwrap.wrap(note, 230)), ha="center", va="bottom", fontsize=8.2, color=GP.MUTED)
    GP.save_webp(fig, out / "gefs_telecon_series.webp")
    plt.close(fig)
    print("  gefs_telecon_series.webp", flush=True)


def plot_probs(ref, base, prev_date, weeks, n_members, out: Path):
    plt = _plt()
    from matplotlib.patches import Patch, Rectangle
    ids = [i for i in ref.ids if i in weeks]
    n = len(ids)
    has_cal = any("skill_r" in r for sid in ids for r in weeks[sid])
    fig = plt.figure(figsize=(13.6, 0.80 * n + 2.6))
    ax = fig.add_axes([0.205, 0.075, 0.765, 0.79])
    W, bw = len(GP.WEEKS), 0.78

    def bar(y, h, pn, pp, x0, txt=True):
        pz = max(0.0, 1.0 - pn - pp)
        ax.barh(y, pn * bw, left=x0, height=h, color="#2c5fa8", lw=0)
        ax.barh(y, pz * bw, left=x0 + pn * bw, height=h, color="#d9d6d0", lw=0)
        ax.barh(y, pp * bw, left=x0 + (pn + pz) * bw, height=h, color="#b4453c", lw=0)
        if txt:
            for frac, off in ((pn, pn / 2), (pp, pn + pz + pp / 2)):
                if frac >= 0.12:
                    ax.text(x0 + off * bw, y, f"{frac * 100:.0f}%", ha="center", va="center", fontsize=7.4,
                            color="white", fontweight="bold")

    for r_, sid in enumerate(ids):
        y = n - 1 - r_
        for row in weeks[sid]:
            x = row["week"] - 1
            x0 = x - bw / 2
            bar(y + 0.10, 0.44, row["p_neg"], row["p_pos"], x0, txt=not row.get("no_skill"))
            dm = f"  Δ{row['mean'] - row['prev_mean']:+.1f}" if "prev_mean" in row else ""
            line = f"mean {row['mean']:+.1f}{dm}"
            if "skill_r" in row:
                bar(y - 0.24, 0.16, row["cal_p_neg"], row["cal_p_pos"], x0, txt=False)
                ax.text(x0 + bw + 0.02, y - 0.24, f"{row['cal_p_neg'] * 100:.0f}/{row['cal_p_pos'] * 100:.0f}",
                        ha="left", va="center", fontsize=7.0, color=GP.INK)
                line += f"  ·  skill r {row['skill_r']:+.2f}"
                if row.get("no_skill"):
                    ax.add_patch(Rectangle((x0 - 0.02, y - 0.36), bw + 0.04, 0.72, facecolor="white", alpha=0.55,
                                           edgecolor="#8a8680", hatch="////", lw=0.6, zorder=5))
                    ax.text(x, y + 0.10, f"no skill  ({row['p_neg'] * 100:.0f}% / {row['p_pos'] * 100:.0f}%)",
                            ha="center", va="center", fontsize=7.4, fontweight="bold", color=GP.INK, zorder=6,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
            else:
                ax.text(x0 + bw / 2, y - 0.24, "no hindcast — uncalibrated", ha="center", va="center", fontsize=6.8,
                        color=GP.MUTED, style="italic")
            ax.text(x, y - 0.43, line, ha="center", va="center", fontsize=7.3, color=GP.INK)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.7, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{sid.upper()}  {ref.pat[sid]['name'].replace('Northern Annular Mode, 100 hPa', 'N. Annular Mode 100 hPa')}"
                        for sid in ids][::-1], fontsize=8.8)
    ax.set_xticks(range(W))
    ax.set_xticklabels([f"Week {k + 1}\n{base + dt.timedelta(days=w0 - 1):%b %d} – {base + dt.timedelta(days=w1 - 1):%b %d}"
                        for k, (w0, w1) in enumerate(GP.WEEKS)], fontsize=8.8)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    fig.legend(handles=[Patch(color="#2c5fa8", label=f"P(weekly mean ≤ −{THR} sd)"), Patch(color="#d9d6d0", label="neutral"),
                        Patch(color="#b4453c", label=f"P(weekly mean ≥ +{THR} sd)")],
               loc="upper center", bbox_to_anchor=(0.59, 0.945), ncol=3, fontsize=8.4, frameon=False)
    fig.suptitle(f"Teleconnection phase probabilities by week — GEFS {n_members} members · init {base:%Y-%m-%d}"
                 + (f"  ·  Δ = change in weekly mean vs the {GP.parse_date(prev_date):%b %d} run" if prev_date else "")
                 + GP.TEST_NOTE, fontsize=12.5, fontweight="bold", y=0.985, va="top")
    step = 100.0 / n_members
    note = (f"upper bar: raw member fractions ({n_members} members → {step:.1f}% steps; 0% = no member, not certainty); "
            "index in seasonal sd for the time of year"
            + ("\nthin lower bar: hindcast-calibrated probabilities — observed weekly index (NCEP/NCAR R1) regressed on the "
               "forecast one over\n2000–2019 GEFSv12 reforecast starts within ±45 d of this date (Wednesday starts, 4-member "
               "mean); skill r = anomaly correlation of the hindcast mean;\n"
               f"hatched = r below {SKILL_MIN} or not significant at 5% (t-test on the effective sample size of "
               "consecutive weekly starts): no usable skill" if has_cal else ""))
    fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=7.8, color=GP.MUTED)
    GP.save_webp(fig, out / "gefs_telecon_probs.webp")
    plt.close(fig)
    print("  gefs_telecon_probs.webp", flush=True)


def _finite(v):
    """Rounded list with None where a day has no value (a pattern CPC does not define in that month): JSON has no NaN."""
    return [round(float(x), 2) if np.isfinite(x) else None for x in v]


# ── main ──────────────────────────────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = GP.common_args(argparse.ArgumentParser())
    a = ap.parse_args()
    base, out, data, runs, cache, ref, clim, z = GP.setup(a)
    if clim is None:
        print("no daily climatology (gefs-clim-v2): teleconnections skipped"); return 0
    days = GP.valid_days(base)
    n_members = int(len(z["members"]))
    fields = GP.member_anoms(z, base, ref, clim)
    idx = GP.indices_from(ref, fields, days)
    del fields
    fc = {sid: (days, v) for sid, v in idx.items()}
    import warnings
    warnings.filterwarnings("ignore", message="All-NaN slice")               # undefined days of a pattern
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    prev_date, pz = GP.previous_run(runs, "telecon", a.date)
    pfc = None
    if pz is not None:
        pdays = GP.valid_days(GP.parse_date(prev_date))
        pfc = {sid: (pdays, pz[sid].astype("float64")) for sid in pz.files if sid in ref.pat}
    print(f"cycle {a.date}, {n_members} members, previous {prev_date or 'none'}, {len(fc)} indices", flush=True)
    obs = observed(ref, base, runs)
    GP.prune_tail(runs, base)
    cpc = published(ref, cache, base)
    cal = calibration(ref, base)
    print(f"  hindcast calibration: {'yes, ' + str(len(cal)) + ' indices' if cal else 'none'}", flush=True)
    weeks = {sid: weekly_stats(t, M, base, pfc.get(sid) if pfc else None, cal.get(sid)) for sid, (t, M) in fc.items()}
    for sid in [i for i in ref.ids if i in weeks]:
        print(f"  {sid:6s} " + "  ".join(
            f"wk{r['week']} {r['mean']:+.1f} [{r['p_neg'] * 100:.0f}/{r['p_pos'] * 100:.0f}]"
            + (f" Δ{r['mean'] - r['prev_mean']:+.1f}" if "prev_mean" in r else "")
            + (f" r{r['skill_r']:+.2f}{'*' if not r['no_skill'] else ''} cal[{r['cal_p_neg'] * 100:.0f}/{r['cal_p_pos'] * 100:.0f}]"
               if "skill_r" in r else "") for r in weeks[sid]), flush=True)
    plot_series(ref, base, prev_date, fc, pfc, obs, cpc, n_members, out)
    plot_plumes(ref, base, prev_date, fc, pfc, obs, cpc, n_members, out)
    plot_probs(ref, base, prev_date, weeks, n_members, out)
    js = {"cycle": a.date, "prev_cycle": prev_date, "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
          "threshold": THR, "skill_min": SKILL_MIN, "skill_p_max": P_MAX,
          "units": "seasonal sd: std of the 1991-2020 daily index within +-15 d of the day of year",
          "n_members": n_members, "valid_convention": "day d valid on init + (d - 1)",
          "hindcast": "GEFSv12 reforecast 2000-2019, Wednesday starts, 4-member mean, vs NCEP/NCAR R1"
                      if cal else None,
          "test": bool(GP.TEST_NOTE), "indices": {}}
    for sid in [i for i in ref.ids if i in fc]:
        t, M = fc[sid]
        m = ref.pat[sid]
        rec = {"name": m["name"], "cpc": m.get("cpc"), "field": m["field"],
               "reference": m.get("reference", "CPC" if m.get("cpc") else None),
               "has_hindcast": any("skill_r" in r for r in weeks[sid]),
               "defined_months": sorted(ref.defined[sid]),
               "r_daily": m.get("r_daily"), "r_monthly": m.get("r_monthly"),
               "r_daily_ref": m.get("r_daily_ref"), "daily_reference": m.get("daily_reference"),
               "weeks": weeks[sid],
               "daily": {"t": [x.isoformat() for x in t], "mean": _finite(np.nanmean(M, 1)),
                         "p10": _finite(np.nanpercentile(M, 10, 1)), "p90": _finite(np.nanpercentile(M, 90, 1))}}
        if sid in obs:
            ot, ov = obs[sid]
            ok = np.isfinite(ov)
            rec["obs"] = {"t": [x.isoformat() for x, k in zip(ot, ok) if k], "v": np.round(ov[ok], 2).tolist()}
        js["indices"][sid] = rec
    (data / "gefs_telecon.json").write_text(json.dumps(js, allow_nan=False))
    print("  data/gefs_telecon.json", flush=True)
    np.savez_compressed(runs / f"gefs_telecon_{a.date}.npz", **{sid: M.astype("float16") for sid, (t, M) in fc.items()})
    GP.prune_runs(runs, "telecon", base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
