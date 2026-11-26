#!/usr/bin/env python3
"""Polar-cap height of the dynamic tropopause, IFS control, against its ERA5 normals.

The 2-PVU surface from geopotential alone on ten levels (700-50 hPa) at 1.5 degrees, the same calculation as the ERA5
normals (build_dt_cap_ref.py):
  temperature    hydrostatic, T = -(g/R) dZ/dln p
  wind           geostrophic, from Z on each level
  PV             -g (f + zeta_g) dtheta/dp, dtheta/dp through ln theta in ln p
  tropopause     first crossing of 2 PVU searching upward from 700 hPa, height interpolated linearly in PV
then averaged (cos-weighted) over 65-87N: the whole cap and its Atlantic side (90W-30E).

Each run adds its analysis to a record (assets/sst/data/dt_cap.json), fills missing 00/12Z analyses of the last
HISTORY_DAYS from the Google mirror (at most --max-fill a run), keeps the forecast to day 10 (and the previous run's),
and draws the last HISTORY_DAYS plus the forecast against the normal's percentiles (assets/sst/dt_cap.webp).
    python src/dt_cap.py --date 20261126 --time 00
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from quantmet.degdays import doy366

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))

ROOT = Path(__file__).resolve().parents[3]
CLIM = ROOT / "scripts/mjo/data/reference/dt_cap_clim.json"
RECORD = ROOT / "assets/sst/data/dt_cap.json"
FIG = ROOT / "assets/sst/dt_cap.webp"

MODEL, MODEL_LAB = "ifs", "IFS control"
LEVS = np.array([700.0, 600, 500, 400, 300, 250, 200, 150, 100, 50])  # bottom up
LAT = np.round(np.arange(37.5, 88.51, 1.5), 3)  # computed rows (the cap mean uses 65-87N)
LON = np.round(np.arange(0.0, 360.0, 1.5), 3)
CAP = (65.0, 87.0)
SECTORS = {"cap": None, "atl": (270.0, 30.0)}  # lon range in 0..360, may wrap
LABELS = {"cap": "Whole polar cap, 65–87°N", "atl": "Atlantic side, 65–87°N, 90°W–30°E"}
THR = 2.0
G, R_D, KAPPA, OMEGA, A_EARTH = 9.80665, 287.04, 0.2857, 7.2921e-5, 6.371e6
STEPS = tuple(range(0, 241, 12))
HISTORY_DAYS = 60


def dt_height(Z: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Height (m) of the 2-PVU surface from geopotential height Z (time, LEVS, lat, lon), lat ascending, regular lon."""
    phi = np.deg2rad(lat)
    dlam = np.deg2rad(lon[1] - lon[0])
    f = 2 * OMEGA * np.sin(phi)
    cos = np.cos(phi)
    Z = Z.astype(np.float64)
    lnp = np.log(LEVS)
    T = -(G / R_D) * np.gradient(Z, lnp, axis=1)
    th = T * ((1000.0 / LEVS) ** KAPPA)[None, :, None, None]
    dthdp = th * np.gradient(np.log(th), lnp, axis=1) / (LEVS * 100.0)[None, :, None, None]
    ug = -(G / (f * A_EARTH))[None, None, :, None] * np.gradient(Z, phi, axis=2)
    vg = (G / (f * A_EARTH * cos))[None, None, :, None] * (np.roll(Z, -1, 3) - np.roll(Z, 1, 3)) / (2 * dlam)
    zeta = (
        (np.roll(vg, -1, 3) - np.roll(vg, 1, 3)) / (2 * dlam) - np.gradient(ug * cos[None, None, :, None], phi, axis=2)
    ) / (A_EARTH * cos[None, None, :, None])
    pv = -G * (f[None, None, :, None] + zeta) * dthdp * 1e6
    above = pv >= THR
    idx = np.argmax(above, axis=1)  # first level at or above 2 PVU, from the bottom
    k1 = np.clip(idx, 1, len(LEVS) - 1)
    k0 = k1 - 1

    def take(a, k):
        return np.take_along_axis(a, k[:, None], axis=1)[:, 0]

    pv0, pv1 = take(pv, k0), take(pv, k1)
    w = np.clip((THR - pv0) / np.where(np.abs(pv1 - pv0) < 1e-6, 1e-6, pv1 - pv0), 0, 1)
    z = take(Z, k0) * (1 - w) + take(Z, k1) * w
    z = np.where(idx == 0, Z[:, 0], z)  # stratospheric already at 700 hPa
    return np.where(above.any(axis=1), z, Z[:, -1])  # no crossing below 50 hPa: the top level


def cap_means(zdt: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict:
    """Cos-weighted 65-87N mean of a (time, lat, lon) field over each of SECTORS."""
    out = {}
    rows = (lat >= CAP[0]) & (lat <= CAP[1])
    w = np.cos(np.deg2rad(lat[rows]))
    for name, sec in SECTORS.items():
        if sec is None:
            cols = np.ones(len(lon), bool)
        else:
            a, b = sec
            cols = (lon >= a) | (lon <= b) if a > b else (lon >= a) & (lon <= b)
        sub = zdt[:, rows][:, :, cols].mean(axis=2)
        out[name] = (sub * w).sum(axis=1) / w.sum()
    return out


def _weights(src: np.ndarray, dst: np.ndarray, half: float, period: float | None = None) -> np.ndarray:
    """(dst, src) averaging weights: each destination cell spans +-half, its edge points count half."""
    d = src[None, :] - dst[:, None]
    if period is not None:
        d = (d + period / 2) % period - period / 2
    eps = 1e-6
    w = np.where(np.abs(d) < half - eps, 1.0, np.where(np.abs(d) <= half + eps, 0.5, 0.0))
    return w / w.sum(axis=1, keepdims=True)


def coarsen(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """A (..., lat, lon) field on a fine regular grid averaged onto the 1.5-degree LAT x LON cells."""
    wy = _weights(lat, LAT, 0.75) * np.cos(np.deg2rad(lat))[None, :]
    wy /= wy.sum(axis=1, keepdims=True)
    wx = _weights(lon, LON, 0.75, period=360.0)
    return np.einsum("yj,...jx,lx->...yl", wy, field, wx)


def step_caps(date: str, hh: str, step: int) -> dict:
    """Cap means of one step of the IFS control (gh on LEVS from the Google mirror, coarsened to 1.5 degrees)."""
    import tclib as TC

    f = TC.oper_fields(MODEL, date, hh, step, pl=("gh",), levels=tuple(int(x) for x in LEVS))
    lat, lon = f["lat"], f["lon"]
    order = np.argsort(lon)
    rows = lat >= LAT[0] - 1.0
    Z = np.stack([f[("gh", int(L))][rows][:, order] for L in LEVS])
    Zc = coarsen(Z, lat[rows], lon[order])
    return {k: float(v[0]) for k, v in cap_means(dt_height(Zc[None], LAT, LON), LAT, LON).items()}


def load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def update_record(rec: dict, init: pd.Timestamp, max_fill: int) -> dict:
    """Add this run's forecast (and its analysis) and fill missing 00/12Z analyses of the last HISTORY_DAYS."""
    date, hh = f"{init:%Y%m%d}", f"{init:%H}"
    fc = []
    for h in STEPS:
        v = init + pd.Timedelta(hours=h)
        fc.append({"valid": f"{v:%Y-%m-%dT%H}", **step_caps(date, hh, h)})
    ana = rec.get("analyses", {})
    ana[f"{init:%Y-%m-%dT%H}"] = {k: fc[0][k] for k in SECTORS}
    want = pd.date_range(init.floor("D") - pd.Timedelta(days=HISTORY_DAYS), init, freq="12h")
    filled = 0
    for t in want[::-1]:
        key = f"{t:%Y-%m-%dT%H}"
        if key in ana or filled >= max_fill:
            continue
        try:
            ana[key] = step_caps(f"{t:%Y%m%d}", f"{t:%H}", 0)
            filled += 1
        except RuntimeError as e:
            print(f"  analysis {key}: {e}", flush=True)
    cut = f"{init - pd.Timedelta(days=2 * HISTORY_DAYS):%Y-%m-%dT%H}"
    ana = {k: ana[k] for k in sorted(ana) if k >= cut}
    this = f"{init:%Y-%m-%dT%H}"
    if rec.get("init") not in (None, this):  # a newer run: the last one becomes the previous run
        prev, prev_init = rec.get("forecast"), rec.get("init")
    else:  # first run, or this run redone
        prev, prev_init = rec.get("previous"), rec.get("previous_init")
    print(f"  {len(fc)} forecast steps, {filled} analyses filled, {len(ana)} kept", flush=True)
    return {
        "init": this,
        "model": MODEL_LAB,
        "analyses": ana,
        "forecast": fc,
        "previous": prev,
        "previous_init": prev_init,
    }


def _series(rows: list | dict, name: str) -> pd.Series:
    if isinstance(rows, dict):
        return pd.Series({pd.Timestamp(k): v[name] for k, v in rows.items()}).sort_index()
    return pd.Series({pd.Timestamp(r["valid"]): r[name] for r in rows}).sort_index()


def render(rec: dict, clim: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import house

    house.apply()
    init = pd.Timestamp(rec["init"])
    t0, t1 = init.floor("D") - pd.Timedelta(days=HISTORY_DAYS), init + pd.Timedelta(hours=STEPS[-1])
    days = pd.date_range(t0, t1.ceil("D"), freq="D")
    slot = doy366(days.values)
    fig, axes = plt.subplots(2, 1, figsize=house.size(9.6, 0.7), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.9, bottom=0.085, hspace=0.24)
    for ax, name in zip(axes, SECTORS):
        q = {p: np.asarray(clim[name][p]) + clim.get("offset", {}).get(name, 0.0) for p in clim["percentiles"]}
        km = 1e-3
        ax.fill_between(days, q["2.5"][slot] * km, q["97.5"][slot] * km, color=house.RULE, lw=0, label="95 % of days")
        ax.fill_between(days, q["10"][slot] * km, q["90"][slot] * km, color=house.EDGE, lw=0, alpha=0.7, label="80 %")
        ax.plot(days, q["50"][slot] * km, color=house.MUTED, lw=1.0, ls="--", label="median")
        a = _series(rec["analyses"], name)
        a = a[a.index >= t0]
        ax.plot(a.index, a.values * km, color=house.INK, lw=1.8, label="analyses")
        if rec.get("previous"):
            p = _series(rec["previous"], name)
            ax.plot(p.index, p.values * km, color=house.MUTED, lw=0.9, label="previous run")
        f = _series(rec["forecast"], name)
        ax.plot(f.index, f.values * km, color=house.ACCENT, lw=2.0, label="forecast")
        ax.axvline(init, color=house.MUTED, lw=0.8)
        ax.set_title(LABELS[name], loc="left")
        ax.set_ylabel("height (km)")
        ax.grid(axis="y", color=house.GRID, lw=0.6)
    axes[0].legend(loc="upper left", ncol=6, frameon=False, fontsize=9)
    axes[-1].set_xlim(t0, t1)
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.suptitle(f"Polar-cap tropopause height — {MODEL_LAB}, init {init:%d %b %HZ}", x=0.01, y=0.985, ha="left")
    w0, w1 = clim["window"]
    house.source(fig, f"2-PVU surface from geopotential, 1.5°. Shading: ERA5 winters {w0}–{w1} for the date.")
    for ax in axes:
        house.legend_room(ax)
    house.clear(fig, 110)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--record", default=str(RECORD))
    ap.add_argument("--figure", default=str(FIG))
    ap.add_argument("--max-fill", type=int, default=40, help="most missing analyses fetched in one run")
    a = ap.parse_args()
    init = pd.Timestamp(f"{a.date}T{int(a.time):02d}:00")
    clim = load(CLIM, None)
    if clim is None:
        raise SystemExit(f"{CLIM.name} missing: build it with build_dt_cap_ref.py")
    rec = update_record(load(Path(a.record), {}), init, a.max_fill)
    Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    Path(a.record).write_text(json.dumps(rec, separators=(",", ":")))
    render(rec, clim, Path(a.figure))
    print(f"wrote {a.record} and {a.figure}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
