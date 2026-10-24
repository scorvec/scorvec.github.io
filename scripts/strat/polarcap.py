#!/usr/bin/env python3
"""Polar-cap sea-level pressure and 500 hPa height, days 1-15, from every ensemble: plumes, maps and the run record.

What the troposphere under the polar vortex is doing: each member's 65-90N mean sea-level pressure and 500 hPa height
against the ERA5 1991-2020 normal of that day and hour (high pressure over the Arctic is the negative Arctic
Oscillation that often follows a sudden warming), and polar maps of the ensemble mean.

Every model's fields are averaged onto the 1.5-deg cells the normals are on (ERA5's in WeatherBench 2; reference/
z500_clim.nc, msl_clim.nc from build_polarcap_clim.py; scripts/lib/cellmeans.py), so the models compare cell for cell:
  aifs  AIFS-ENS control + 25 perturbed members (as its other stratosphere products): msl, and z at 500 hPa from the
        wave maps' file; ECMWF open data, Google Cloud mirror.
  ifs   IFS-ENS, all 50 perturbed members, from the shared IFS-ENS files (ifs_ens.py gh and msl), and the IFS control
        (its own stream: ctrl_gh, ctrl_msl), drawn on the IFS-ENS plume and recorded as a model of its own (hres).
  gefs  GEFS control + 30 members, geps GEPS control + 20 members: strat_ens.py's "polarcap" part (PRMSL, HGT 500).
Daily to 360 h, each value the field at its valid time (00 or 12 UTC, as the normal). The sea-level pressure cap leaves
out the cells whose ERA5 surface lies above HIGH (Greenland's ice sheet, the higher Siberian and Canadian ranges), where
sea-level pressure is an extrapolation each model makes its own way; the maps leave it unshaded above MAP_HIGH. Raw
model output: no drift is removed (the dripping paint's cap heights are drift-corrected and standardised).

Outputs, model m:
  assets/sst/polarcap_{m}.webp      the plumes: every member's cap anomaly and the members' mean (IFS-ENS: the control)
  assets/sst/anim/polarcap_{m}/ + polarcap_{m}_manifest.json   maps of the ensemble mean, the anomaly coloured where
                                     enough members give it its sign for a two-sided sign test at 5 % (agreeing()):
                                     days 1-5, 6-10, 11-15, then each day
  assets/sst/data/runs/polarcap_{m}.json (+ hres)   the run record (run_record.py, which draws the run-to-run figures)
  --combined: assets/sst/polarcap_models.webp, every model's newest run from the records (strat-combine.yml)

    python polarcap.py --model ifs --date 20261024 --time 00
    python polarcap.py --combined
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "lib"), str(REPO / "scripts" / "ecmwf")]
import cellmeans as CM
import harmonic_clim as HC
import run_record as RR

CELL = 1.5
CLAT = np.round(np.arange(13, 61) * CELL, 2)  # 19.5..90 N: the normals' cells
CLON = np.round(np.arange(240) * CELL, 2)  # 0..358.5 E
CAP = 65.0
HIGH = 500.0  # m: the sea-level pressure cap leaves out cells whose ERA5 surface is higher
MAP_HIGH = 1500.0  # m: the maps leave sea-level pressure anomalies unshaded above this
LEADS = tuple(range(0, 361, 24))
WINDOWS = ((1, 5), (6, 10), (11, 15))
AIFS_MEMBERS = 25
G = 9.80665
OUT = REPO / "assets" / "sst"
ANIM = OUT / "anim"
ENSEMBLES = ("aifs", "ifs", "gefs", "geps")
WHO = {
    "aifs": "control + 25 members",
    "ifs": "50 members",
    "gefs": "control + 30 members",
    "geps": "control + 20 members",
}
CREDIT = {
    "aifs": "ECMWF AIFS-ENS open data (CC BY 4.0), Google Cloud mirror",
    "ifs": "ECMWF IFS-ENS and IFS control open data (CC BY 4.0), Google Cloud mirror",
    "gefs": "NOAA GEFS (AWS Open Data)",
    "geps": "ECCC GEPS (MSC Datamart)",
}
NEUTRAL = "#d9d6d0"
MSL_LEVELS = np.arange(-20, 21, 2)  # hPa
Z_LEVELS = np.arange(-200, 201, 20)  # m


# ── the grid, the normals, the cap ───────────────────────────────────────────────────────────────────────────────
def coarse(f: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cell means on CLAT x CLON of f (..., lat, lon) on a regular global grid (rows south of the cells left out)."""
    rows = lat >= CLAT[0] - CELL
    h = abs(float(lat[1] - lat[0]))
    return CM.coarsen(f[..., rows, :], lat[rows], lon, h, CLAT, CLON, CELL, 360.0)


class Normals:
    """ERA5 1991-2020 on the cells: sea-level pressure (hPa) at 00 and 12 UTC, 500 hPa height (m), surface height."""

    def __init__(self):
        import xarray as xr

        with xr.open_dataset(HERE / "reference" / "z500_clim.nc") as z, xr.open_dataset(HERE / "reference" / "msl_clim.nc") as p:  # fmt: skip
            for d in (z, p):
                if not (np.allclose(d.latitude, CLAT) and np.allclose(d.longitude, CLON)):
                    raise SystemExit("the polar-cap normals are not on the cells CLAT x CLON")
            self.z = z["coef"].values.astype(np.float64)
            self.p = {int(h): p["coef"].sel(hour=h).values.astype(np.float64) for h in p.hour.values}
            self.orog = p["orog"].values.astype(np.float64)

    def at(self, valid: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        """(sea-level pressure, 500 hPa height) normal at that day and hour, (CLAT, CLON)."""
        if valid.hour not in self.p:
            raise SystemExit(f"no sea-level pressure normal for {valid:%H} UTC")
        t = HC.times_of([valid])
        return HC.evaluate(self.p[valid.hour], t)[0], HC.evaluate(self.z, t)[0]


def cap(x: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """The 65-90N area mean of x (..., CLAT, CLON) over the cells `keep` (every cell when None)."""
    rows = CLAT >= CAP
    w = np.cos(np.deg2rad(CLAT[rows]))[:, None] * np.ones(CLON.size)
    if keep is not None:
        w = w * keep[rows]
    return (x[..., rows, :] * w).sum((-2, -1)) / w.sum()


# ── the fields ───────────────────────────────────────────────────────────────────────────────────────────────────
def read_grib(path: Path, short: str, level: int | None = None, scale: float = 1.0) -> dict:
    """{(member, step h): cell means} of the messages of `short` (at `level`) in a GRIB file of global regular grids;
    a message without a member number is member 0 (the control)."""
    import eccodes

    out, grid = {}, None
    with open(path, "rb") as fh:
        while (gid := eccodes.codes_grib_new_from_file(fh)) is not None:
            try:
                if eccodes.codes_get(gid, "shortName") != short:
                    continue
                if level is not None and eccodes.codes_get(gid, "level") != level:
                    continue
                try:
                    num = int(eccodes.codes_get(gid, "number"))
                except eccodes.KeyValueNotFoundError:
                    num = 0
                step = int(eccodes.codes_get(gid, "endStep"))
                if step not in LEADS:
                    continue
                if grid is None:
                    ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
                    la0 = eccodes.codes_get_double(gid, "latitudeOfFirstGridPointInDegrees")
                    lo0 = eccodes.codes_get_double(gid, "longitudeOfFirstGridPointInDegrees")
                    dj = eccodes.codes_get_double(gid, "jDirectionIncrementInDegrees")
                    di = eccodes.codes_get_double(gid, "iDirectionIncrementInDegrees")
                    up = eccodes.codes_get(gid, "jScansPositively")
                    lat = la0 + (dj if up else -dj) * np.arange(nj)
                    grid = (nj, ni, lat, lo0 + di * np.arange(ni))
                nj, ni, lat, lon = grid
                v = eccodes.codes_get_values(gid).reshape(nj, ni) * scale
                out[num, step] = coarse(v, lat, lon)
            finally:
                eccodes.codes_release(gid)
    return out


def stack(*parts: dict) -> np.ndarray:
    """(member, lead, CLAT, CLON) from read_grib dicts, NaN where a member lacks a lead; members with nothing dropped."""
    got = {k: v for p in parts for k, v in p.items()}
    mems = sorted({m for m, _ in got})
    a = np.full((len(mems), len(LEADS), CLAT.size, CLON.size), np.nan, np.float32)
    for (m, s), v in got.items():
        a[mems.index(m), LEADS.index(s)] = v
    return a


def load_aifs(date: str, hh: str) -> dict:
    import store as E
    import wave_maps as WM

    cyc, steps = E.Cycle(date, hh), tuple(E.STEPS)
    msl, z = [], []
    for kind, n in (("cf", 0), ("pf", AIFS_MEMBERS)):
        msl.append(read_grib(E.ensure(cyc, E.Spec("aifs-ens", kind, "msl", "sfc", (), steps, n)), "msl", scale=0.01))
        zs = E.Spec("aifs-ens", kind, "z", "pl", WM.LEVELS, steps, n)  # the wave maps' file
        z.append(read_grib(E.ensure(cyc, zs), "z", 500, 1 / G))
    return {"aifs": (stack(*msl), stack(*z))}


def load_ifs(date: str, hh: str) -> dict:
    """IFS-ENS (when complete on the mirror) and the IFS control, from the shared files."""
    import ifs_ens as IE

    cyc, out = IE.E.Cycle(date, hh), {}
    if IE.published(date, hh):
        out["ifs"] = (
            stack(read_grib(IE.ensure(cyc, IE.spec_msl()), "msl", scale=0.01)),
            stack(read_grib(IE.ensure(cyc, IE.spec_gh()), "gh", 500)),
        )
    if IE.ctrl_published(date, hh):
        out["hres"] = (
            stack(read_grib(IE.ensure(cyc, IE.spec_ctrl_msl()), "msl", scale=0.01)),
            stack(read_grib(IE.ensure(cyc, IE.spec_ctrl_gh()), "gh", 500)),
        )
    return out


def load_strat_ens(model: str, date: str, hh: str) -> dict:
    import strat_ens as SE

    d = SE.load(model, date, hh, ("polarcap",))
    keep = SE.complete_members(d, "pc_z500")
    return {model: (d["pc_msl"][keep], d["pc_z500"][keep])}


# ── one run ──────────────────────────────────────────────────────────────────────────────────────────────────────
def anomalies(init: pd.Timestamp, msl: np.ndarray, z: np.ndarray, N: Normals):
    """The members' anomalies (member, lead, CLAT, CLON) of both fields, and the valid times."""
    valid = [init + pd.Timedelta(hours=h) for h in LEADS]
    nm = np.stack([np.stack(N.at(v)) for v in valid])  # (lead, 2, CLAT, CLON)
    return msl - nm[None, :, 0], z - nm[None, :, 1], pd.DatetimeIndex(valid)


def caps(am: np.ndarray, az: np.ndarray, N: Normals) -> tuple[np.ndarray, np.ndarray]:
    """(member, lead) cap anomalies: sea-level pressure over the cells below HIGH, 500 hPa height over all."""
    return cap(am, N.orog < HIGH), cap(az)


def record(model: str, init: pd.Timestamp, cm: np.ndarray, cz: np.ndarray, valid: pd.DatetimeIndex) -> None:
    frames = {k: pd.DataFrame(v.T, index=valid) for k, v in (("msl", cm), ("z500", cz))}
    RR.record("polarcap", model, init, RR.entry("polarcap", frames))
    RR.render("polarcap", model)


def agreeing(n: int) -> int:
    """The fewest of n members that must share a sign for a two-sided sign test against an even split to pass at 5 %
    (16 of 21, 19 of 26, 22 of 31, 33 of 50)."""
    from scipy.stats import binom

    return next(k for k in range(n // 2 + 1, n + 1) if 2 * binom.sf(k - 1, n, 0.5) < 0.05)


def agreed(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the members' mean, where enough members share its sign: agreeing) for x (member, ...) without gaps."""
    m = x.mean(0)
    same = (np.sign(x) == np.sign(m)[None]).sum(0)
    return m, (same >= agreeing(x.shape[0])) & (m != 0)


# ── figures ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _style(ax, ylab: str) -> None:
    import matplotlib.dates as mdates

    ax.axhline(0, color="#8a8680", lw=1.0, zorder=2)
    ax.set_ylabel(ylab, fontsize=9.5)
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))


def plumes(model: str, init: pd.Timestamp, valid, cm, cz, ctrl=None, out: Path | None = None) -> Path:
    """Every member's cap anomaly, the members' mean bold, (the control dashed), both fields."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col, lab = RR.MODEL_COLOUR[model], RR.MODEL_LABEL[model]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True)
    for ax, x, k, (title, unit) in zip(
        axes,
        (cm, cz),
        (0, 1),
        (("Sea-level pressure, 65–90°N (cells below 500 m)", "hPa"), ("500 hPa height, 65–90°N", "m")),
    ):
        for row in x:
            ax.plot(valid, row, color=col, lw=0.8, alpha=0.35, zorder=3)
        mean = np.nanmean(x, 0)
        ax.plot(valid, mean, color=col, lw=2.8, zorder=5, label=f"{lab} mean, {x.shape[0]} members")
        if ctrl is not None:
            ax.plot(valid, ctrl[k], color="#1f1f1f", lw=1.6, ls="--", zorder=6, label="IFS control")
        late = (valid >= init + pd.Timedelta(days=6)) & (valid <= init + pd.Timedelta(days=15))
        lm = np.nanmean(x[:, late], 1)
        fmt = "{:+.1f}" if unit == "hPa" else "{:+.0f}"
        ax.set_title(f"{title}: anomaly from ERA5 1991–2020", fontsize=11, fontweight="bold", loc="left", pad=16)
        ax.text(
            0,
            1.015,
            f"days 6–15: members' mean {fmt.format(np.nanmean(lm)).replace('-', '−')} {unit}, "
            f"{(lm > 0).mean():.0%} of the members above normal",
            transform=ax.transAxes,
            fontsize=9,
            color="0.25",
        )
        _style(ax, unit)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    fig.suptitle(
        f"{lab} · the troposphere under the polar vortex · run {init:%Y-%m-%d %HZ}",
        fontsize=13.5,
        fontweight="bold",
        x=0.07,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.07,
        0.945,
        "every member thin, the members' mean bold · above zero: high pressure and heights over the Arctic "
        "(the negative Arctic Oscillation)",
        fontsize=9.5,
        color="0.25",
    )
    fig.text(
        0.07,
        0.012,
        f"Raw model output against the ERA5 normal of each day and hour. {CREDIT[model]}.",
        fontsize=8,
        color="#6f6b64",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.07, hspace=0.32)
    out = out or OUT / f"polarcap_{model}.webp"
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out


def _pole(a: np.ndarray) -> np.ndarray:
    """The cells' 90N row (wedges of the pole cap, a different value at each longitude) as one value, the pole's: a map
    draws the row as a single point, where differing values would meet in a star."""
    a = a.copy()
    a[-1] = a[-1].mean() >= 0.5 if a.dtype == bool else a[-1].mean()
    return a


def _panel(fig, pos, mean, anom, gate, levels, contour_step, unit, title, high=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import vortex_winds as VW

    pc = ccrs.PlateCarree()
    ax = fig.add_axes(pos, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, CLAT[0] + 0.5, 90], crs=pc)
    VW.circular_boundary(ax)
    lon = np.append(CLON, 360.0)
    cyc = lambda a: np.concatenate([a, a[:, :1]], axis=1)  # noqa: E731
    mean, anom, gate = _pole(mean), _pole(anom), _pole(gate)
    ax.contourf(
        lon, CLAT, cyc(np.where(gate, np.nan, 1.0)), levels=[0.5, 1.5], colors=[NEUTRAL], transform=pc, zorder=1
    )
    shown = np.where(gate, anom, np.nan)
    cf = ax.contourf(lon, CLAT, cyc(shown), levels=levels, cmap="RdBu_r", extend="both", transform=pc, zorder=2)
    if high is not None:
        ax.contourf(lon, CLAT, cyc(np.where(high, 1.0, np.nan)), levels=[0.5, 1.5], colors=["white"], transform=pc,
                    zorder=2.5)  # fmt: skip
    lo, hi = (
        np.floor(np.nanmin(mean) / contour_step) * contour_step,
        np.ceil(np.nanmax(mean) / contour_step) * contour_step,
    )
    cs = ax.contour(lon, CLAT, cyc(mean), levels=np.arange(lo, hi + 1, contour_step), colors="#111", linewidths=0.8,
                    transform=pc, zorder=3)  # fmt: skip
    ax.clabel(cs, fmt="%d", fontsize=6.5, inline_spacing=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#2b2b2b", linewidth=0.55, zorder=4)
    gl = ax.gridlines(linewidth=0.3, color="#8a8a8a", alpha=0.5, zorder=5)
    gl.ylocator = plt.FixedLocator([30, 45, 60, 75])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    cax = fig.add_axes([pos[0] + 0.04, pos[1] - 0.05, pos[2] - 0.08, 0.018])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=levels[::2])
    cb.set_label(f"anomaly from ERA5 1991–2020, {unit}", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)


def map_frame(model, init, span, n, mm, ma, mg, zm, za, zg, high, out: Path) -> None:
    """Two polar panels: the mean sea-level pressure (hPa) and 500 hPa height (zm, dam) contoured, their anomalies (m)
    shaded where the gates mg, zg pass."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = RR.MODEL_LABEL[model]
    fig = plt.figure(figsize=(13.0, 7.4), dpi=110)
    _panel(fig, [0.02, 0.175, 0.46, 0.69], mm, ma, mg, MSL_LEVELS, 4, "hPa", "Sea-level pressure", high)
    _panel(fig, [0.52, 0.175, 0.46, 0.69], zm, za, zg, Z_LEVELS, 6, "m", "500 hPa height")
    fig.text(0.02, 0.965, f"{lab} · sea-level pressure and 500 hPa height: {span}", fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.928,
        f"Run {init:%Y-%m-%d %HZ} · {n} members · ensemble mean (black, every 4 hPa and 6 dam); shaded: its anomaly "
        f"where at least {agreeing(n)} of the {n} members agree on the sign (a sign test at 5 %)",
        fontsize=9.5,
        color="#333",
    )
    fig.text(
        0.02,
        0.008,
        "Grey: the members are split. Sea-level pressure is not shaded where the surface is above 1500 m.\n"
        f"Raw model output against ERA5 1991–2020 at each day and hour. {CREDIT[model]}.",
        fontsize=8,
        color="#6f6b64",
        linespacing=1.4,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def maps(model: str, init: pd.Timestamp, msl, z, am, az, N: Normals) -> None:
    """The frames and the manifest: days 1-5, 6-10, 11-15, then each day."""
    d = ANIM / f"polarcap_{model}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.webp"):
        old.unlink()
    high = N.orog > MAP_HIGH
    spans = [(f"P{i + 1}", a, b) for i, (a, b) in enumerate(WINDOWS)] + [(f"D{k:02d}", k, k) for k in range(1, 16)]
    frames = []
    for name, a, b in spans:
        sel = [LEADS.index(24 * k) for k in range(a, b + 1)]
        v0, v1 = init + pd.Timedelta(days=a), init + pd.Timedelta(days=b)
        what = f"day {a}" if a == b else f"days {a}–{b}"
        when = f"{v0:%a %-d %b %HZ}" if a == b else f"{v0:%a %-d %b} – {v1:%a %-d %b}"
        span = f"{what}, {when}"
        mm, (ma, mg) = np.nanmean(msl[:, sel], (0, 1)), agreed(np.nanmean(am[:, sel], 1))
        zm, (za, zg) = np.nanmean(z[:, sel], (0, 1)), agreed(np.nanmean(az[:, sel], 1))
        map_frame(model, init, span, msl.shape[0], mm, ma, mg, zm / 10.0, za, zg, high, d / f"{name}.webp")
        frames.append(
            {"file": f"{name}.webp", "date": f"{v0:%Y-%m-%d}", "label": f"{what[0].upper()}{what[1:]} · {when}"}
        )
    rid = f"polarcap_{model}"
    man = {"ver": f"{init:%Y%m%d%H}", "days": len(frames), "selectorLabel": "Model", "default": rid,
           "regions": {rid: {"label": f"Sea-level pressure and 500 hPa height, {RR.MODEL_LABEL[model]}",
                             "n_frames": len(frames), "frames": [dict(f, idx=i) for i, f in enumerate(frames)]}}}  # fmt: skip
    (ANIM / f"{rid}_manifest.json").write_text(json.dumps(man))
    print(f"  {len(frames)} map frames -> {d}", flush=True)


def combined(out: Path = OUT / "polarcap_models.webp", max_age_days: float = 2.0) -> Path | None:
    """Every model's newest run from the records: the members' mean and 10-90 % band, the IFS control dashed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    newest = {m: rs[-1] for m in (*ENSEMBLES, "hres") if (rs := RR.runs("polarcap", m))}
    if not newest:
        return None
    t1 = max(pd.Timestamp(r["init"]) for r in newest.values())
    newest = {m: r for m, r in newest.items() if t1 - pd.Timestamp(r["init"]) <= pd.Timedelta(days=max_age_days)}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True)
    for ax, key, (title, unit) in zip(
        axes,
        ("msl", "z500"),
        (("Sea-level pressure, 65–90°N (cells below 500 m)", "hPa"), ("500 hPa height, 65–90°N", "m")),
    ):
        for m, r in newest.items():
            mean = RR.series(r, key)
            col = RR.MODEL_COLOUR[m]
            lab = f"{RR.MODEL_LABEL[m]} {pd.Timestamp(r['init']):%-d %b %HZ}"
            if m == "hres":
                ax.plot(mean.index, mean.values, color="#1f1f1f", lw=1.6, ls="--", zorder=6, label=lab)
                continue
            ax.fill_between(mean.index, RR.series(r, key, "p10"), RR.series(r, key, "p90"), color=col, alpha=0.13, lw=0)
            ax.plot(mean.index, mean.values, color=col, lw=2.4, zorder=5, label=f"{lab}, {r['members']} members")
        ax.set_title(f"{title}: anomaly from ERA5 1991–2020", fontsize=11, fontweight="bold", loc="left")
        _style(ax, unit)
        ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.9)
    fig.suptitle(
        f"Every ensemble's newest run · the troposphere under the polar vortex (newest {t1:%Y-%m-%d %HZ})",
        fontsize=13.5,
        fontweight="bold",
        x=0.07,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.07,
        0.945,
        "each model's members' mean with its 10–90 % range, the IFS control dashed · above zero: high pressure and "
        "heights over the Arctic (the negative Arctic Oscillation)",
        fontsize=9.5,
        color="0.25",
    )
    fig.text(
        0.07,
        0.012,
        "Raw model output against the ERA5 normal of each day and hour; each model is labelled with its run.",
        fontsize=8,
        color="#6f6b64",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.07, hspace=0.32)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def run(model: str, date: str, hh: str) -> int:
    init = pd.Timestamp(f"{date} {hh}:00")
    N = Normals()
    load = {"aifs": load_aifs, "ifs": load_ifs}.get(model) or (lambda d, h: load_strat_ens(model, d, h))
    fields = load(date, hh)
    if not fields:
        print(f"::warning::{model} {date} {hh}Z: nothing on the mirror to draw", flush=True)
        return 1
    done, ctrl = {}, None
    for m, (msl_all, z_all) in fields.items():
        full = np.isfinite(msl_all).all((1, 2, 3)) & np.isfinite(z_all).all((1, 2, 3))  # members with every lead
        msl, z = msl_all[full], z_all[full]
        am, az, valid = anomalies(init, msl, z, N)
        cm, cz = caps(am, az, N)
        print(f"  {RR.MODEL_LABEL[m]}: {msl.shape[0]} members; analysis cap {np.nanmean(cm[:, 0]):+.1f} hPa, "
              f"{np.nanmean(cz[:, 0]):+.0f} m; days 6-15 {np.nanmean(cm[:, 6:]):+.1f} hPa, {np.nanmean(cz[:, 6:]):+.0f} m",
              flush=True)  # fmt: skip
        record(m, init, cm, cz, valid)
        if m == "hres":
            ctrl = (cm[0], cz[0])
        else:
            done[m] = (msl, z, am, az, cm, cz, valid)
    for m, (msl, z, am, az, cm, cz, valid) in done.items():
        plumes(m, init, valid, cm, cz, ctrl if m == "ifs" else None)
        maps(m, init, msl, z, am, az, N)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=ENSEMBLES)
    ap.add_argument("--date")
    ap.add_argument("--time")
    ap.add_argument("--combined", action="store_true", help="every model's newest run from the records")
    a = ap.parse_args()
    if a.combined:
        combined()
        return 0
    if not (a.model and a.date and a.time):
        ap.error("--model, --date and --time are required (or --combined)")
    return run(a.model, a.date, f"{int(a.time):02d}")


if __name__ == "__main__":
    raise SystemExit(main())
