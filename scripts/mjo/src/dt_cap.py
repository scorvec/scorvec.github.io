#!/usr/bin/env python3
"""Polar-cap height of the dynamic tropopause from four single runs, against its ERA5 normals.

The 2-PVU surface from geopotential alone on ten levels (700-50 hPa) at 1.5 degrees, the same calculation as the ERA5
normals (build_dt_cap_ref.py):
  temperature    hydrostatic, T = -(g/R) dZ/dln p
  wind           geostrophic, from Z on each level
  PV             -g (f + zeta_g) dtheta/dp, dtheta/dp through ln theta in ln p
  tropopause     first crossing of 2 PVU searching upward from 700 hPa, height interpolated linearly in PV
then averaged (cos-weighted) over 65-87N: the whole cap and its Atlantic side (90W-30E). Every model's geopotential is
averaged onto the same 1.5-degree cells first, so the runs and the normal compare cell for cell:
  ifs    IFS control, gh from ECMWF open data (Google Cloud mirror, tclib.oper_fields)
  aifs1  AIFS single, the same way
  gfs    GFS 0.5 deg, HGT from NOAA's AWS bucket by byte range (strat_maps' S3 transport)
  gdps   GDPS 15 km, GeopotentialHeight from ECCC's HPFX Datamart (strat_maps' Datamart getter)
12-hourly to day 10, all from the same 00/12Z cycle. A model that fails is left out of that run's figure.

The record (assets/sst/data/dt_cap.json) holds the IFS control's analyses (step 0 of each 00/12Z run; missing ones in the
last HISTORY_DAYS are filled from the mirror, at most --max-fill a run), each model's newest run, and the onsets of
sudden warmings: the analysed onset ssw.html reports (ssw_summary.json, event.onset), kept while in the window. The
figure (assets/sst/dt_cap.webp) draws the last HISTORY_DAYS and the runs against the normal's percentiles.
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

SCRIPTS = Path(__file__).resolve().parents[2]
# scripts/strat goes in first, so last in the search order: its own strat_maps.py would hide scripts/models' one
for sub in ("strat", "geps", "snowband", "models", "ecmwf", "lib"):
    sys.path.insert(0, str(SCRIPTS / sub))

ROOT = SCRIPTS.parent
CLIM = ROOT / "scripts/mjo/data/reference/dt_cap_clim.json"
RECORD = ROOT / "assets/sst/data/dt_cap.json"
SUMMARY = ROOT / "assets/sst/data/ssw_summary.json"
FIG = ROOT / "assets/sst/dt_cap.webp"

MODELS = {"ifs": "IFS control", "aifs1": "AIFS single", "gfs": "GFS", "gdps": "GDPS"}
COLOUR_KEY = {"ifs": "hres", "aifs1": "aifs1", "gfs": "gfs", "gdps": "gdps"}  # run_record.MODEL_COLOUR keys
ANALYSIS_MODEL = "ifs"
LEVS = np.array([700.0, 600, 500, 400, 300, 250, 200, 150, 100, 50])  # bottom up
Z_STD = np.array([3010.0, 4210, 5570, 7180, 9160, 10360, 11790, 13610, 16180, 20580])  # m, standard atmosphere
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


def _metres(Z: np.ndarray, model: str) -> np.ndarray:
    """Geopotential height in metres, whatever unit a model's GRIB carries (m, dam or m2/s2), told by its magnitude."""
    r = np.nanmedian(Z.reshape(len(LEVS), -1), axis=1) / Z_STD
    for f in (1.0, 10.0, 1.0 / G):
        if np.all(np.abs(r * f - 1.0) < 0.2):
            return Z * f
    raise RuntimeError(f"{model} heights {np.round(r * Z_STD)} m: not geopotential height in any known unit")


def _ecmwf(model: str, date: str, hh: str, step: int):
    import tclib as TC

    f = TC.oper_fields(model, date, hh, step, pl=("gh",), levels=tuple(int(x) for x in LEVS))
    return np.stack([f[("gh", int(L))] for L in LEVS]), f["lat"], f["lon"]


def _gfs(date: str, hh: str, step: int):
    import snowband as sb
    import strat_maps as SM

    msgs, grid = sb.fetch_messages(SM.fetcher(), SM.gfs_url(date, hh, step), {("HGT", f"{p:g} mb") for p in LEVS})
    return np.stack([msgs[("HGT", f"{p:g} mb")] for p in LEVS]), grid["lat"][:, 0], grid["lon"][0]


def _gdps(date: str, hh: str, step: int):
    from concurrent.futures import ThreadPoolExecutor

    import datamart as DM
    import strat_maps as SM

    urls = [SM.gdps_url(date, hh, step, f"GeopotentialHeight_IsbL-{int(p):04d}") for p in LEVS]
    with ThreadPoolExecutor(DM.WORKERS) as ex:
        bodies = list(ex.map(DM.get, urls))
    if any(b is None for b in bodies):
        raise RuntimeError(f"GDPS {date} {hh}Z +{step} h: {sum(b is None for b in bodies)} height files missing")
    Z = np.stack([SM.decode_ll(b)[0] for b in bodies])  # rows 90N -> 90S, columns from 0E
    return Z, np.linspace(90.0, -90.0, Z.shape[1]), np.arange(Z.shape[2]) * 360.0 / Z.shape[2]


FETCH = {
    "ifs": lambda d, h, s: _ecmwf("ifs", d, h, s),
    "aifs1": lambda d, h, s: _ecmwf("aifs-single", d, h, s),
    "gfs": _gfs,
    "gdps": _gdps,
}


def step_caps(model: str, date: str, hh: str, step: int) -> dict:
    """Cap means of one step of one model: its gh on LEVS, coarsened to the 1.5-degree cells."""
    Z, lat, lon = FETCH[model](date, hh, step)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float) % 360.0
    iy, ix = np.argsort(lat), np.argsort(lon)
    Z, lat, lon = Z[:, iy][:, :, ix], lat[iy], lon[ix]
    rows = lat >= LAT[0] - 1.0
    Zc = coarsen(_metres(Z[:, rows], model), lat[rows], lon)
    return {k: float(v[0]) for k, v in cap_means(dt_height(Zc[None], LAT, LON), LAT, LON).items()}


def load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _key(t: pd.Timestamp) -> str:
    return f"{t:%Y-%m-%dT%H}"


def run_model(model: str, init: pd.Timestamp) -> list | None:
    """One model's forecast steps, or None when any step fails."""
    date, hh = f"{init:%Y%m%d}", f"{init:%H}"
    try:
        return [{"valid": _key(init + pd.Timedelta(hours=h)), **step_caps(model, date, hh, h)} for h in STEPS]
    except Exception as e:  # noqa: BLE001  one model's failure leaves the others
        print(f"  {MODELS[model]}: left out ({e})", flush=True)
        return None


def update_record(rec: dict, init: pd.Timestamp, max_fill: int) -> dict:
    """Add each model's run, the IFS control's analysis, missing analyses of the last HISTORY_DAYS and the onsets."""
    runs = {}
    for m in MODELS:
        fc = run_model(m, init)
        if fc is not None:
            runs[m] = {"init": _key(init), "forecast": fc}
            print(f"  {MODELS[m]}: day 0 cap {fc[0]['cap']:.0f} m, atl {fc[0]['atl']:.0f} m", flush=True)
    if not runs:
        raise SystemExit(f"no model run for {init:%Y-%m-%d %HZ}")
    ana = rec.get("analyses", {})
    if ANALYSIS_MODEL in runs:
        ana[_key(init)] = {k: runs[ANALYSIS_MODEL]["forecast"][0][k] for k in SECTORS}
    filled = 0
    for t in pd.date_range(init.floor("D") - pd.Timedelta(days=HISTORY_DAYS), init, freq="12h")[::-1]:
        if _key(t) in ana or filled >= max_fill:
            continue
        try:
            ana[_key(t)] = step_caps(ANALYSIS_MODEL, f"{t:%Y%m%d}", f"{t:%H}", 0)
            filled += 1
        except RuntimeError as e:
            print(f"  analysis {_key(t)}: {e}", flush=True)
    cut = init - pd.Timedelta(days=2 * HISTORY_DAYS)
    ana = {k: ana[k] for k in sorted(ana) if pd.Timestamp(k) >= cut}
    onsets = set(rec.get("onsets", []))
    onset = (load(SUMMARY, {}).get("event") or {}).get("onset")
    if onset:
        onsets.add(onset)
    onsets = sorted(d for d in onsets if pd.Timestamp(d) >= cut)
    print(f"  {len(runs)} models, {filled} analyses filled, {len(ana)} kept, onsets {onsets}", flush=True)
    return {"init": _key(init), "analyses": ana, "runs": runs, "onsets": onsets}


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
    from run_record import MODEL_COLOUR

    house.apply()
    init = pd.Timestamp(rec["init"])
    t0, t1 = init.floor("D") - pd.Timedelta(days=HISTORY_DAYS), init + pd.Timedelta(hours=STEPS[-1])
    days = pd.date_range(t0, t1.ceil("D"), freq="D")
    slot = doy366(days.values)
    runs = {m: r for m, r in rec["runs"].items() if r["init"] == rec["init"]}
    fig, axes = plt.subplots(2, 1, figsize=house.size(9.6, 0.7), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.9, bottom=0.085, hspace=0.24)
    km = 1e-3
    for ax, name in zip(axes, SECTORS):
        q = {p: np.asarray(clim[name][p]) + clim.get("offset", {}).get(name, 0.0) for p in clim["percentiles"]}
        ax.fill_between(days, q["2.5"][slot] * km, q["97.5"][slot] * km, color=house.RULE, lw=0, label="95 % of days")
        ax.fill_between(days, q["10"][slot] * km, q["90"][slot] * km, color=house.EDGE, lw=0, alpha=0.7, label="80 %")
        ax.plot(days, q["50"][slot] * km, color=house.MUTED, lw=1.0, ls="--", label="median")
        a = _series(rec["analyses"], name)
        a = a[a.index >= t0]
        ax.plot(a.index, a.values * km, color=house.INK, lw=1.8, label="analyses")
        for m, r in runs.items():
            f = _series(r["forecast"], name)
            ax.plot(f.index, f.values * km, color=MODEL_COLOUR[COLOUR_KEY[m]], lw=1.7, label=MODELS[m])
        ax.axvline(init, color=house.MUTED, lw=0.8)
        for d in rec.get("onsets", []):
            ts = pd.Timestamp(d)
            if t0 <= ts <= t1:
                ax.axvline(ts, color=house.WARM, lw=1.2, ls=(0, (4, 3)))
                if ax is axes[0]:
                    ax.text(ts, 1.01, f" SSW {ts:%-d %b}", transform=ax.get_xaxis_transform(), color=house.WARM,
                            fontsize=9, ha="left", va="bottom")  # fmt: skip
        ax.set_title(LABELS[name], loc="left")
        ax.set_ylabel("height (km)")
        ax.grid(axis="y", color=house.GRID, lw=0.6)
    axes[0].legend(loc="upper left", ncol=4, frameon=False, fontsize=9)
    axes[-1].set_xlim(t0, t1)
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.suptitle(f"Polar-cap tropopause height — runs of {init:%d %b %HZ}", x=0.01, y=0.985, ha="left")
    w0, w1 = clim["window"]
    house.source(
        fig,
        f"2-PVU surface from geopotential, 1.5°. Analyses: IFS control. Shading: ERA5 winters {w0}–{w1} for the date.",
    )
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
