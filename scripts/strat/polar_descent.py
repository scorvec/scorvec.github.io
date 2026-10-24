#!/usr/bin/env python3
"""Descent over the polar cap, days 1-15: where the air over the Arctic sinks (and warms by compression) or rises, from
the stratosphere down through the troposphere, for the cap as a whole and six sectors of it, from every ensemble.

The quantity is the log-pressure vertical velocity w* = -(H / p) omega (H = 7 km), the TEM convention: omega (Pa/s) is
what the models publish, and w* puts the stratosphere and the troposphere on one scale (mm/s). Shown as DESCENT,
positive downward. Each member's omega is averaged onto polarcap.py's 1.5-deg cells first, then
  cap       the 65-90N cos-weighted mean, at every level
  sectors   the same over six 60-deg sectors of the cap: Greenland (60W-0), Barents and Scandinavia (0-60E), Kara and
            West Siberia (60-120E), East Siberia (120-180E), Alaska and Bering (180-120W), Canadian Arctic (120-60W)
  layers    the cap and sector means averaged in log-pressure over 10-100 hPa (the stratosphere the models share) and
            100-300 hPa (the tropopause region the descent has to cross to reach the weather)
  maps      north of 50N, each member smoothed to planetary scales first (zonal wavenumbers 0-8, a 3-deg Gaussian in
            latitude), the ensemble mean shaded only where enough members agree on descent or ascent for a two-sided
            sign test at 5 % (polarcap.agreeing); orographic gravity waves (Greenland, the Rockies) are averaged away
The models: AIFS-ENS control + 25 and IFS-ENS 50 members (ECMWF open data, Google Cloud mirror; w at 850, 500, 300,
250, 200, 150, 100, 50 and 10 hPa), the IFS control and the AIFS-ENS control drawn on their ensembles' figures and
recorded on their own, and GEFS control + 30 (VVEL from its pgrb2b files, to 1 hPa: strat_ens.py part "omega"). Each
value is the field at its valid time (the run's hour each day). The observed record is GEOS FP's analysis (NCCS
OPeNDAP), the daily mean of its 00/06/12/18Z omega (which cancels the atmospheric tides), `tail` below (bdc.yml).

Raw model output, no normal. omega is the EULERIAN motion, the air's own movement through each pressure surface, not the
Brewer-Dobson (residual) circulation: while planetary waves carry heat poleward, as before a sudden warming, the
Eulerian mean over the cap RISES, its adiabatic cooling offsetting part of the eddies' warming, while the residual
circulation sinks (Andrews, Holton and Leovy 1987). The sectors and maps show where the air sinks and warms.

  residual  for GEFS and GEOS FP, whose v and T reach 1 hPa, the cap mean of the residual (TEM) vertical velocity
            omega* = [omega] + (1/(a cos phi)) d/dphi (cos phi [v'theta'] / d[theta]/dp), which over the cap is its
            Eulerian mean less cos phi0 X / (a (1 - sin phi0)), X = [v'theta'] / d[theta]/dp on its edge PHI0 (the
            zonal means of the grid rows within 0.3 deg of it; d/dp centred in pressure between the levels): the descent
            that carries a warming down, from RES_BOTTOM up. The ECMWF ensembles' open data have v and T at too few
            levels for it.

    python polar_descent.py --model ifs --date 20261024 --time 00          # IFS-ENS and the IFS control
    python polar_descent.py tail --days 5     # GEOS FP: -> assets/sst/data/descent_tail.json (bdc.yml)
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "lib"), str(REPO / "scripts" / "ecmwf")]
import polarcap as PC
import run_record as RR

H = 7000.0  # m, the log-pressure scale height
LEVELS_EC = (850, 500, 300, 250, 200, 150, 100, 50, 10)  # hPa: ECMWF open data's w above the boundary layer
LEVELS_GEFS = (850, 500, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1)  # and GEOS FP's tail
SECTORS = (
    ("Greenland", 300, 360),
    ("Barents and Scandinavia", 0, 60),
    ("Kara and West Siberia", 60, 120),
    ("East Siberia", 120, 180),
    ("Alaska and Bering", 180, 240),
    ("Canadian Arctic", 240, 300),
)
REGIONS = ("Polar cap", *(s[0] for s in SECTORS))  # the series' last axis
LAYERS = {"strat": (10, 100), "ut": (100, 300)}  # hPa
MAP_LAT = 49.5  # the maps' southern edge (cell centre)
MAP_LEVELS = (10, 50, 100)
KMAX = 8  # zonal wavenumbers kept in the maps
SIGMA_LAT = 2.0  # cells (3 deg): the maps' meridional smoothing
PHI0 = 65.25  # deg N: the cap cells' southern edge (polarcap.CLAT >= CAP: the cells from 65.25N)
A_EARTH = 6.371e6  # m
RES_BOTTOM = 300.0  # hPa: the residual descent is drawn above it (below, the polar stability is too weak for the TEM
# eddy term to be read from daily fields)
KAPPA = 0.2857  # R / cp
TAIL = REPO / "assets" / "sst" / "data" / "descent_tail.json"
TAIL_DAYS = 45
OUT = REPO / "assets" / "sst"
ANIM = OUT / "anim"
ENSEMBLES = ("aifs", "ifs", "gefs")
CONTROL = PC.CONTROL
RATE = np.arange(-4.0, 4.01, 0.5)  # mm/s, the column chart
MAP_RATE = np.arange(-6.0, 6.01, 1.0)  # mm/s, the maps


# ── the reduction ────────────────────────────────────────────────────────────────────────────────────────────────
def descent(omega: np.ndarray, levels) -> np.ndarray:
    """mm/s of descent, -w* = +H omega / p, for omega (Pa/s) with a level axis third from the end (..., level, lat,
    lon)."""
    p = np.asarray(levels, float)[:, None, None] * 100.0
    return 1000.0 * H * omega / p


def area_means(d: np.ndarray, lat: np.ndarray = PC.CLAT, lon: np.ndarray = PC.CLON) -> np.ndarray:
    """(..., 7) the cap and sector means of d (..., lat, lon), cos-weighted, missing points left out."""
    rows = lat >= PC.CAP
    lon = lon % 360
    ok = np.isfinite(d[..., rows, :])
    x = np.where(ok, d[..., rows, :], 0.0)
    w = np.cos(np.deg2rad(lat[rows]))[:, None] * np.ones(lon.size)
    out = []
    for m in [np.ones(lon.size, bool)] + [(lon >= a) & (lon < b) for _, a, b in SECTORS]:
        wm = w * m
        out.append((x * wm).sum((-2, -1)) / (ok * wm).sum((-2, -1)))
    return np.stack(out, -1)


def smooth(d: np.ndarray) -> np.ndarray:
    """Planetary scales of d (..., lat, lon) on the cells: zonal wavenumbers 0-KMAX, then a Gaussian in latitude."""
    from scipy.ndimage import gaussian_filter1d

    f = np.fft.rfft(d, axis=-1)
    f[..., KMAX + 1 :] = 0
    return gaussian_filter1d(np.fft.irfft(f, n=d.shape[-1], axis=-1), SIGMA_LAT, axis=-2, mode="nearest")


def boundary_eddies(v: np.ndarray, t: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """([v'T'], [T]) on the cap's edge PHI0 from v, t (..., level, lat, lon): the zonal-mean eddy heat flux and
    temperature of the grid rows within 0.3 deg of it, averaged over those rows -> (..., level)."""
    rows = np.abs(lat - PHI0) <= 0.3
    vv, tt = v[..., rows, :], t[..., rows, :]
    vt = ((vv - vv.mean(-1, keepdims=True)) * (tt - tt.mean(-1, keepdims=True))).mean(-1)
    return vt.mean(-1), tt.mean((-2, -1))


def residual_shift(vt: np.ndarray, tbar: np.ndarray, levels) -> np.ndarray:
    """The eddy part of the cap-mean residual descent, mm/s (..., level): -cos phi0 X / (a (1 - sin phi0)) with
    X = [v'theta'] / d[theta]/dp on the edge, in the descent units of `descent`."""
    p = np.asarray(levels, float) * 100.0
    fac = (1e5 / p) ** KAPPA  # theta / T
    th_p = np.gradient(tbar * fac, p, axis=-1)
    domega = -np.cos(np.deg2rad(PHI0)) * (vt * fac / th_p) / (A_EARTH * (1 - np.sin(np.deg2rad(PHI0))))
    return np.where(p <= RES_BOTTOM * 100.0, 1000.0 * H * domega / p, np.nan)


def reduce_step(omega: np.ndarray, levels, vt: np.ndarray | None = None, tbar: np.ndarray | None = None) -> dict:
    """One lead's reductions from every member's omega (member, level, CLAT, CLON), Pa/s: the cap and sector series
    (member, level, 7), north of MAP_LAT the smoothed members' mean (level, lat, lon) and the number of members
    descending there (level, lat, lon), and with the edge's eddies (boundary_eddies, (member, level)) the cap's residual
    descent (member, level)."""
    d = descent(omega, levels)
    rows = PC.CLAT >= MAP_LAT
    s = smooth(d[..., rows, :])
    out = {
        "series": area_means(d).astype("float32"),
        "mean": s.mean(0).astype("float32"),
        "ndesc": (s > 0).sum(0).astype("int16"),
    }
    if vt is not None:
        out["res"] = (out["series"][..., 0] + residual_shift(vt, tbar, levels)).astype("float32")
    return out


def stack_steps(steps: list) -> dict:
    """The leads' reduce_step outputs stacked: series (member, lead, level, 7), mean and ndesc (lead, level, lat, lon),
    res (member, lead, level) when the steps carry it."""
    out = {
        "series": np.stack([s["series"] for s in steps], 1),
        "mean": np.stack([s["mean"] for s in steps]),
        "ndesc": np.stack([s["ndesc"] for s in steps]),
    }
    if "res" in steps[0]:
        out["res"] = np.stack([s["res"] for s in steps], 1)
    return out


def layer(x: np.ndarray, levels, top: float, bot: float) -> np.ndarray:
    """The log-pressure mean over top..bot hPa of x (..., level, region): trapezoid weights in ln p."""
    lv = np.asarray(levels, float)
    keep = (lv >= top) & (lv <= bot)
    lp = np.log(lv[keep])
    o = np.argsort(lp)
    w = np.zeros(keep.sum())
    seg = np.diff(lp[o])
    w[o[:-1]] += seg / 2
    w[o[1:]] += seg / 2
    return np.tensordot(x[..., keep, :], w / w.sum(), axes=([-2], [0]))


# ── the models ───────────────────────────────────────────────────────────────────────────────────────────────────
def _from_grib(paths: list, levels) -> dict:
    """{member: omega (lead, level, CLAT, CLON)} from GRIB files of w (members and the control as their own messages);
    a member missing any lead or level is left out."""
    got = {}
    for path in paths:
        for k, lev in enumerate(levels):
            for (m, step), cells in PC.read_grib(path, "w", (lev,))[lev].items():
                got.setdefault(m, np.full((len(PC.LEADS), len(levels), PC.CLAT.size, PC.CLON.size), np.nan, "float32"))
                got[m][PC.LEADS.index(step), k] = cells
    return {m: a for m, a in got.items() if np.isfinite(a).all()}


def _reduce_members(members: dict, levels) -> dict:
    ms = sorted(members)
    a = np.stack([members[m] for m in ms])  # (member, lead, level, lat, lon)
    return {**stack_steps([reduce_step(a[:, i], levels) for i in range(a.shape[1])]), "n": len(ms)}


def load_aifs(date: str, hh: str) -> dict:
    import store as E

    cyc, steps = E.Cycle(date, hh), tuple(E.STEPS)
    paths = {
        k: E.ensure(cyc, E.Spec("aifs-ens", k, "w", "pl", LEVELS_EC, steps, n))
        for k, n in (("cf", 0), ("pf", PC.AIFS_MEMBERS))
    }
    mem = _from_grib([paths["cf"], paths["pf"]], LEVELS_EC)  # the control is member 0
    return {"aifs": _reduce_members(mem, LEVELS_EC), "aifsc": _reduce_members({0: mem[0]}, LEVELS_EC)}


def load_ifs(date: str, hh: str) -> dict:
    import ifs_ens as IE

    cyc, out = IE.E.Cycle(date, hh), {}
    for m, ready, spec in (("ifs", IE.published, IE.spec_w), ("hres", IE.ctrl_published, IE.spec_ctrl_w)):
        if ready(date, hh):
            out[m] = _reduce_members(_from_grib([IE.ensure(cyc, spec())], LEVELS_EC), LEVELS_EC)
    return out


def load_gefs(date: str, hh: str) -> dict:
    import strat_ens as SE

    d = SE.load("gefs", date, hh, ("omega",))
    keep = SE.complete_members(d, "om_series")
    out = {"series": d["om_series"][keep], "mean": d["om_mean"], "ndesc": d["om_ndesc"], "n": int(d["om_n"].min())}
    if "om_res" in d:
        out["res"] = d["om_res"][keep]
    return {"gefs": out}


LEVELS_OF = {"aifs": LEVELS_EC, "aifsc": LEVELS_EC, "ifs": LEVELS_EC, "hres": LEVELS_EC, "gefs": LEVELS_GEFS}


# ── the observed record ──────────────────────────────────────────────────────────────────────────────────────────
def tail(a) -> int:
    """GEOS FP: the cap and sector descent (mm/s) per level and the cap's residual descent, daily means of the
    00/06/12/18Z analyses, for the last `days` days not yet held -> TAIL (the last KEEP kept)."""
    import time
    import warnings

    import geosfp_cap as FP

    have = json.loads(TAIL.read_text()).get("days", {}) if TAIL.exists() else {}
    fp = FP.Fp().open()
    lev = fp.d.lev.values
    li = [int(np.argmin(np.abs(lev - L))) for L in LEVELS_GEFS]
    l0, l1 = min(li), max(li)
    r0 = int(np.where(fp.d.lat.values >= PC.CAP)[0][0])
    lat, lon = fp.d.lat.values[r0::2], fp.d.lon.values[::4]  # 0.5 x 1.25 deg, as geosfp_cap: a cap mean loses nothing
    edge = np.flatnonzero(np.abs(fp.d.lat.values - PHI0) <= 0.3)  # the rows on the cap's edge, for the residual
    end = fp.tn.normalize() - pd.Timedelta(days=1) if fp.tn.hour < 18 else fp.tn.normalize()
    days = [
        d
        for d in pd.date_range(end - pd.Timedelta(days=a.days - 1), end)
        if a.force or not isinstance(have.get(f"{d:%Y-%m-%d}"), dict)  # a day held without its residual is refetched
    ]

    def read(var, it, lat_sl):
        for k in range(7):  # NCCS answers 504 under load
            try:
                return fp.d[var].isel(time=it, lev=slice(l0, l1 + 1), lat=lat_sl, lon=slice(None, None, 4)).values
            except Exception as e:  # noqa: BLE001
                if k == 6:
                    raise SystemExit(f"GEOS FP {var} step {it}: {str(e)[:80]}") from e
                time.sleep(15 * (k + 1))

    for day in days:
        snaps, vts, tbars = [], [], []
        for h in (0, 6, 12, 18):
            it = round((day + pd.Timedelta(hours=h) - fp.t0) / pd.Timedelta(hours=3))
            om = read("omega", it, slice(r0, None, 2))
            snaps.append(np.where(np.abs(om) < 1e5, om, np.nan)[[i - l0 for i in li]])  # below ground: missing
            sl = slice(int(edge[0]), int(edge[-1]) + 1)
            vt, tb = boundary_eddies(*(read(v, it, sl)[[i - l0 for i in li]] for v in ("v", "t")), fp.d.lat.values[sl])
            vts.append(vt)
            tbars.append(tb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x = area_means(descent(np.nanmean(snaps, 0), LEVELS_GEFS), lat, lon)  # (level, 7)
        res = x[:, 0] + residual_shift(np.mean(vts, 0), np.mean(tbars, 0), LEVELS_GEFS)
        have[f"{day:%Y-%m-%d}"] = {"w": np.round(x, 3).tolist(), "res": np.round(res, 3).tolist()}
        k10 = LEVELS_GEFS.index(10)
        print(
            f"  GEOS FP {day:%Y-%m-%d}: cap descent 10 hPa {x[k10, 0]:+.2f} mm/s, residual {res[k10]:+.2f}", flush=True
        )
    keep = sorted(have)[-a.keep :]
    TAIL.write_text(json.dumps({"source": "GEOS FP inst3_3d_asm_Np omega (NASA GMAO, NCCS OPeNDAP), daily mean of "
                                "00/06/12/18Z, on polarcap.py's 1.5-deg cells; descent -w = H omega / p, H = 7 km, mm/s "
                                "(w: the cap and sectors, Eulerian; res: the cap's residual, with the eddies on 65.25N)",
                                "levels": list(LEVELS_GEFS), "regions": list(REGIONS),
                                "days": {k: have[k] for k in keep}}, separators=(",", ":")))  # fmt: skip
    print(f"descent tail: {len(days)} day(s) fetched, {len(keep)} kept -> {TAIL}")
    return 0


def observed(levels) -> pd.DataFrame | None:
    """The GEOS FP tail at the model's levels: a frame with a (level, region) column index, by day; region "Residual"
    is the cap's residual descent."""
    if not TAIL.exists():
        return None
    j = json.loads(TAIL.read_text())
    idx = [j["levels"].index(L) for L in levels if L in j["levels"]]
    if len(idx) < len(levels):
        return None
    days = sorted(d for d, v in j["days"].items() if isinstance(v, dict))
    if not days:
        return None
    w = np.array([np.array(j["days"][d]["w"], float)[idx] for d in days])  # (day, level, region)
    r = np.array([np.array(j["days"][d]["res"], float)[idx] for d in days])[..., None]
    cols = pd.MultiIndex.from_product([list(levels), [*REGIONS, "Residual"]])
    return pd.DataFrame(np.concatenate([w, r], -1).reshape(len(days), -1), index=pd.DatetimeIndex(days), columns=cols)


# ── figures ──────────────────────────────────────────────────────────────────────────────────────────────────────
INK, MUTED = "#1f2328", "#6f6b64"
CREDIT = {**PC.CREDIT, "gefs": "NOAA GEFS (AWS Open Data), VVEL from the pgrb2b files"}


def column(model, init, valid, levels, series, res, ctrl, obs, out: Path) -> None:
    """Time-height of the cap's descent - the residual descent where the model has it (res, (member, lead, level)),
    else the Eulerian one: GEOS FP's last TAIL_DAYS days, then the members' mean; beneath, the cap's 10-100 and
    100-300 hPa layers (with the residual, the Eulerian 10-100 hPa as well)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import forecast_drip as FD

    region = "Polar cap" if res is None else "Residual"
    fc = series[..., 0] if res is None else res  # (member, lead, level)
    cap = np.nanmean(fc, 0)
    t_obs = obs.index[(obs.index < init.normalize()) & (obs.index >= init.normalize() - pd.Timedelta(days=TAIL_DAYS))] if obs is not None else []  # fmt: skip
    xo = np.asarray((pd.DatetimeIndex(t_obs) - init.normalize()).days, float)
    xf = np.asarray((valid - init.normalize()) / pd.Timedelta(days=1), float)[1:]
    Zo = np.array([[obs.loc[t, (L, region)] for L in levels] for t in t_obs]) if len(t_obs) else np.empty((0, len(levels)))  # fmt: skip
    X, Z = np.r_[xo, xf], np.vstack([Zo, cap[1:]])
    fig = plt.figure(figsize=(13.4, 8.6), dpi=120)
    lab = RR.MODEL_LABEL[model]
    what = "descent" if res is None else "residual descent"
    fig.text(0.07, 0.975, f"{lab} · {what} over the polar cap (65–90°N), observed and forecast · run {init:%d %b %Y %HZ}",
             fontsize=15, fontweight="bold", color=INK, va="top")  # fmt: skip
    n = series.shape[0]
    kind = (
        "Log-pressure vertical velocity, downward positive (red: the air sinks and warms by compression). Eulerian "
        "motion: while waves carry heat poleward the cap-mean air rises, offsetting part of their warming."
        if res is None
        else "The residual (TEM) vertical velocity −(H/p) ω*, downward positive: the Eulerian motion plus the eddies' "
        "heat-flux term on the cap's edge, the descent that carries a warming down."
    )
    fig.text(0.07, 0.935, f"{kind} Left of the line: GEOS FP analyses, daily means; right: "
             f"{'the run' if n == 1 else f'the mean of {n} members'}, at the run's hour each day. Raw output, no normal.",
             fontsize=9.4, color=MUTED, va="top", wrap=True)  # fmt: skip
    ax = fig.add_axes([0.07, 0.42, 0.84, 0.45])
    cf = ax.contourf(X, np.array(levels, float), Z.T, levels=RATE, cmap="RdBu_r", extend="both")
    ax.axvline(0, color=INK, lw=1.4)
    FD._paxis(ax, levels)
    ax.set_xlim(X[0], X[-1])
    ticks = list(range(int(np.ceil(X[0] / 7)) * 7, int(X[-1]) + 1, 7))
    labels = [f"{init.normalize() + pd.Timedelta(days=t):%d %b}" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8.5)
    cb = fig.colorbar(cf, cax=fig.add_axes([0.925, 0.42, 0.012, 0.45]), ticks=RATE[::2])
    cb.set_label("descent, mm/s (red = sinking)", fontsize=9)
    ax2 = fig.add_axes([0.07, 0.08, 0.84, 0.26])
    col = RR.MODEL_COLOUR[model]
    for key, ls in (("strat", "-"), ("ut", "--")):
        top, bot = LAYERS[key]
        name = f"{top}–{bot} hPa"
        if obs is not None and len(t_obs):
            o = layer(Zo[:, :, None], levels, top, bot)[:, 0]
            ax2.plot(xo, o, color=INK, lw=1.5, ls=ls, label=f"{name}, observed (GEOS FP)")
        f = layer(fc[:, 1:, :, None], levels, top, bot)[..., 0]  # (member, lead)
        if n > 1 and key == "strat":
            ax2.fill_between(
                xf, *np.nanpercentile(f, [10, 90], 0), color=col, alpha=0.15, lw=0, label="members 10–90 %"
            )
        ax2.plot(
            xf, np.nanmean(f, 0), color=col, lw=2.0, ls=ls, label=f"{name}, {'run' if n == 1 else 'members’ mean'}"
        )
        if ctrl is not None:
            c = layer(ctrl[:, 1:, :, :1], levels, top, bot)[0, :, 0]
            ax2.plot(
                xf,
                c,
                color="#1f1f1f",
                lw=1.2,
                ls=":" if key == "ut" else "-.",
                label=f"{name}, {RR.MODEL_LABEL[CONTROL[model]]}",
            )
    if res is not None:  # the Eulerian 10-100 hPa mean beside it: the eddies' term is the gap
        top, bot = LAYERS["strat"]
        e = layer(series[:, 1:, :, :1], levels, top, bot)[..., 0]
        ax2.plot(
            xf, np.nanmean(e, 0), color="#8a8680", lw=1.4, ls=":", label=f"{top}–{bot} hPa, Eulerian, members’ mean"
        )
        if obs is not None and len(t_obs):
            eo = np.array([[obs.loc[t, (L, "Polar cap")] for L in levels] for t in t_obs])
            ax2.plot(xo, layer(eo[:, :, None], levels, top, bot)[:, 0], color="#8a8680", lw=1.2, ls=":",
                     label=f"{top}–{bot} hPa, Eulerian, observed")  # fmt: skip
    ax2.axhline(0, color="#555", lw=0.8)
    ax2.axvline(0, color=INK, lw=1.0)
    ax2.set_xlim(X[0], X[-1])
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("descent, mm/s", fontsize=10, color=INK)
    ax2.legend(fontsize=8, ncol=3, frameon=True, framealpha=0.9, edgecolor="none", loc="upper left")
    ax2.set_title("The cap's stratosphere (10–100 hPa) and tropopause region (100–300 hPa), log-pressure means",
                  loc="left", fontsize=10.2, fontweight="bold", color=INK)  # fmt: skip
    for a in (ax, ax2):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    how = (
        "−(H/p) ω, H = 7 km, from each model's ω averaged onto 1.5° cells."
        if res is None
        else f"ω* over the cap = its Eulerian mean less cos φ0 [v′θ′]/(∂θ/∂p) / (a (1 − sin φ0)) on its edge, φ0 = {PHI0}°N."
    )
    fig.text(0.07, 0.012, f"{how} {CREDIT[model]} · GEOS FP (NASA GMAO).", fontsize=8.2, color=MUTED)
    fig.savefig(out, dpi=120, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def sectors(model, init, valid, levels, series, ctrl, obs, out: Path) -> None:
    """The cap's six sectors: each one's 10-100 hPa descent (the members' mean and 10-90 %) after GEOS FP's last
    TAIL_DAYS days, and its 100-300 hPa descent dashed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    col, lab, n = RR.MODEL_COLOUR[model], RR.MODEL_LABEL[model], series.shape[0]
    vf = valid[1:]
    t0 = init.normalize() - pd.Timedelta(days=TAIL_DAYS)
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.8), sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat, start=1):
        for key, ls in (("strat", "-"), ("ut", "--")):
            top, bot = LAYERS[key]
            f = layer(series[:, 1:, :, k : k + 1], levels, top, bot)[..., 0]
            if obs is not None:
                o = obs[(obs.index >= t0) & (obs.index < init.normalize())]
                ov = layer(np.stack([o[(L, REGIONS[k])].values for L in levels], -1)[:, :, None], levels, top, bot)[
                    :, 0
                ]
                ax.plot(o.index, ov, color=INK, lw=1.4, ls=ls)
            if n > 1 and key == "strat":
                ax.fill_between(vf, *np.nanpercentile(f, [10, 90], 0), color=col, alpha=0.15, lw=0)
            ax.plot(vf, np.nanmean(f, 0), color=col, lw=2.0, ls=ls)
            if ctrl is not None and key == "strat":
                ax.plot(
                    vf, layer(ctrl[:, 1:, :, k : k + 1], levels, top, bot)[0, :, 0], color="#1f1f1f", lw=1.1, ls="-."
                )
        ax.axhline(0, color="#555", lw=0.8)
        ax.axvline(init, color=INK, lw=1.0)
        lo, hi = SECTORS[k - 1][1], SECTORS[k - 1][2]
        ax.set_title(f"{REGIONS[k]} ({_lon(lo)}–{_lon(hi)})", fontsize=10.5, fontweight="bold", loc="left")
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=8.5)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    for ax in axes[:, 0]:
        ax.set_ylabel("descent, mm/s", fontsize=9.5)
    fig.suptitle(f"{lab} · descent by sector of the polar cap (65–90°N) · run {init:%d %b %Y %HZ}", fontsize=14,
                 fontweight="bold", x=0.06, ha="left", y=0.985)  # fmt: skip
    who = "the run" if n == 1 else f"the mean of {n} members (shaded 10–90 %)"
    ctl = f", {RR.MODEL_LABEL[CONTROL[model]]} dash-dot" if ctrl is not None else ""
    fig.text(0.06, 0.935, f"Solid: 10–100 hPa, dashed: 100–300 hPa (log-pressure means), downward positive · black: GEOS FP "
             f"analyses; colour: {who}{ctl}", fontsize=9.4, color=MUTED)  # fmt: skip
    fig.text(0.06, 0.01, textwrap.fill("Raw output, no normal. The sector values carry what the polar-cap mean cancels: a "
             "displaced vortex sinks on one side of the pole and rises on the other. " + CREDIT[model] + " · GEOS FP (NASA "
             "GMAO).", 200), fontsize=8, color=MUTED)  # fmt: skip
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.08, hspace=0.3, wspace=0.08)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def _lon(x: float) -> str:
    """A longitude east of 0 as text, 0-180E or W."""
    x = (x + 180) % 360 - 180
    if x in (0, -180):
        return "0°" if x == 0 else "180°"
    return f"{abs(x):g}°{'E' if x > 0 else 'W'}"


def map_frame(model, init, valid, n, levels, mean, ndesc, out: Path) -> None:
    """MAP_LEVELS side by side: the smoothed ensemble mean's descent, shaded where enough members agree on its sign."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import vortex_winds as VW

    lat = PC.CLAT[PC.CLAT >= MAP_LAT]
    lon = np.append(PC.CLON, 360.0)
    cyc = lambda a: np.concatenate([a, a[:, :1]], axis=1)  # noqa: E731
    k = PC.agreeing(n) if n > 1 else 0
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14.3, 6.2), dpi=100)
    shown = [L for L in MAP_LEVELS if L in levels]
    for j, L in enumerate(shown):
        i = list(levels).index(L)
        m, c = mean[i], ndesc[i]
        gate = np.ones(m.shape, bool) if n == 1 else (c >= k) | (n - c >= k)
        ax = fig.add_axes([0.01 + j * 0.33, 0.19, 0.31, 0.68], projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, MAP_LAT + 0.5, 90], crs=pc)
        VW.circular_boundary(ax)
        ax.contourf(lon, lat, cyc(np.where(gate, np.nan, 1.0)), levels=[0.5, 1.5], colors=[PC.NEUTRAL], transform=pc)
        cf = ax.contourf(
            lon, lat, cyc(np.where(gate, m, np.nan)), levels=MAP_RATE, cmap="RdBu_r", extend="both", transform=pc
        )
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#2b2b2b", linewidth=0.55)
        gl = ax.gridlines(linewidth=0.3, color="#8a8a8a", alpha=0.5)
        gl.ylocator = plt.FixedLocator([60, 70, 80])
        ax.set_title(f"{L} hPa", fontsize=11.5, fontweight="bold")
    cb = fig.colorbar(cf, cax=fig.add_axes([0.3, 0.12, 0.4, 0.025]), orientation="horizontal", ticks=MAP_RATE[::2])
    cb.set_label("descent, mm/s (red = sinking)", fontsize=9)
    how = "one run" if n == 1 else f"ensemble mean, shaded where at least {k} of the {n} members agree on descent or ascent (a sign test at 5 %); grey: split"  # fmt: skip
    fig.text(0.01, 0.965, f"{RR.MODEL_LABEL[model]} · descent north of 50°N: {valid:%a %-d %b %HZ} (day {(valid - init).days}) · "
             f"run {init:%Y-%m-%d %HZ}", fontsize=13, fontweight="bold")  # fmt: skip
    fig.text(
        0.01,
        0.92,
        f"{how} · each member smoothed to zonal waves 0–{KMAX} and 3° in latitude first",
        fontsize=9,
        color="#333",
    )
    fig.text(
        0.01,
        0.015,
        f"Log-pressure vertical velocity, downward positive. Raw model output. {CREDIT[model]}.",
        fontsize=8,
        color=MUTED,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def maps(model, init, valid, levels, red) -> None:
    d = ANIM / f"descent_{model}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.webp"):
        old.unlink()
    frames = []
    for k in range(1, len(valid)):
        map_frame(model, init, valid[k], red["n"], levels, red["mean"][k], red["ndesc"][k], d / f"D{k:02d}.webp")
        frames.append(
            {"file": f"D{k:02d}.webp", "date": f"{valid[k]:%Y-%m-%d}", "label": f"Day {k} · {valid[k]:%a %-d %b %HZ}"}
        )
    PC.manifest(f"descent_{model}", init, f"Descent over the Arctic, {RR.MODEL_LABEL[model]}", frames)
    print(f"  {len(frames)} descent maps -> {d}", flush=True)


# ── one run ──────────────────────────────────────────────────────────────────────────────────────────────────────
def run(model: str, date: str, hh: str, figures: bool = True) -> int:
    init = pd.Timestamp(f"{date} {hh}:00")
    valid = pd.DatetimeIndex([init + pd.Timedelta(hours=h) for h in PC.LEADS])
    got = {"aifs": load_aifs, "ifs": load_ifs, "gefs": load_gefs}[model](date, hh)
    if not got:
        print(f"::warning::{model} {date} {hh}Z: nothing on the mirror to draw", flush=True)
        return 1
    for m, red in got.items():
        lv, x = LEVELS_OF[m], red["series"]
        cap = {key: layer(x[..., :1], lv, *LAYERS[key])[..., 0] for key in LAYERS}  # (member, lead)
        if "res" in red:
            cap["res"] = layer(red["res"][..., None], lv, *LAYERS["strat"])[..., 0]
        print(f"  {RR.MODEL_LABEL[m]}: {x.shape[0]} members; cap 10-100 hPa descent {np.nanmean(cap['strat'][:, 1:6]):+.2f} "
              f"mm/s days 1-5, {np.nanmean(cap['strat'][:, 6:]):+.2f} days 6-15"
              + (f"; residual {np.nanmean(cap['res'][:, 1:6]):+.2f} / {np.nanmean(cap['res'][:, 6:]):+.2f}" if "res" in cap else ""),
              flush=True)  # fmt: skip
        PC.record("descent", m, init, cap, valid)
        if figures:
            maps(m, init, valid, lv, red)
    if not figures:
        return 0
    for m in (e for e in ENSEMBLES if e in got):
        lv, c = LEVELS_OF[m], got.get(CONTROL.get(m))
        ctrl = c["series"] if c else None
        obs = observed(lv)
        column(m, init, valid, lv, got[m]["series"], got[m].get("res"), ctrl, obs, OUT / f"descent_{m}.webp")
        sectors(m, init, valid, lv, got[m]["series"], ctrl, obs, OUT / f"descent_sectors_{m}.webp")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("tail",), help="tail: the GEOS FP record")
    ap.add_argument("--model", choices=ENSEMBLES)
    ap.add_argument("--date")
    ap.add_argument("--time")
    ap.add_argument("--record-only", action="store_true", help="the run into its records, no figures (a backfill)")
    ap.add_argument("--days", type=int, default=5, help="tail: the days back to fill")
    ap.add_argument("--keep", type=int, default=90, help="tail: the days kept")
    ap.add_argument("--force", action="store_true", help="tail: refetch the days held")
    a = ap.parse_args()
    if a.cmd == "tail":
        return tail(a)
    if not (a.model and a.date and a.time):
        ap.error("--model, --date and --time are required (or tail)")
    return run(a.model, a.date, f"{int(a.time):02d}", figures=not a.record_only)


if __name__ == "__main__":
    raise SystemExit(main())
