#!/usr/bin/env python3
"""The polar vortex and the troposphere under it, days 1-15, from every ensemble: the polar cap at sea level and 500
hPa, the Arctic and North Atlantic Oscillations, the 100 hPa height, and at 10 hPa the temperature and the vortex's
shape member by member (split, displaced: vortex_shape.py) - plumes, maps and run records.

Each member's sea-level pressure, 500 and 100 hPa height against the ERA5 1991-2020 normal of that day and hour (high
pressure over the Arctic is the negative Arctic Oscillation that often follows a sudden warming). Every model's fields
are averaged onto the 1.5-deg cells the normals are on (ERA5's in WeatherBench 2; reference/{msl,z500,z100}_clim.nc from
build_polarcap_clim.py; scripts/lib/cellmeans.py), so the models compare cell for cell:
  aifs  AIFS-ENS control + 50 perturbed members: msl and t at 10 hPa from the 90-18N rows of each message (all the
        cells need, store.ensure_rows), and z at 500, 100 and 10 hPa from the wave maps' file; ECMWF open data, Google
        Cloud mirror. The control (member 0) is also drawn on the AIFS-ENS plumes and recorded and mapped on its own
        (aifsc), as the IFS control is.
  ifs   IFS-ENS, all 50 perturbed members, from the shared IFS-ENS files (ifs_ens.py gh, gh_wind, msl and t10), and
        the IFS control (its own stream: ctrl_gh, ctrl_msl), drawn on the IFS-ENS plumes and recorded and mapped as a
        model of its own (hres; its 10 hPa temperature maps are t10_maps.py's).
  gefs  GEFS control + 30 members, geps GEPS control + 20 members: strat_ens.py's "polarcap" part (PRMSL, HGT 500, 100,
        10, TMP 10).
  aifs1 AIFS single, ECMWF's deterministic AIFS run, on its own (no ensemble): msl and t at 10 hPa from the 90-18N rows
        of each message, gh at 500, 100 and 10 hPa from the shared files of its stratosphere products (aifs_single.py);
        recorded and mapped (100 hPa, 10 hPa) as the control runs are.
Daily to 360 h, each value the field at its valid time (00, 06, 12 or 18 UTC, as the normal: AIFS-ENS runs four times a
day, the others at 00 and 12 UTC). Raw model output: no drift is removed (the dripping paint's cap heights are
drift-corrected and standardised).

  polar cap  65-90N means. The sea-level pressure cap leaves out the cells whose ERA5 surface lies above HIGH
             (Greenland's ice sheet, the higher Siberian and Canadian ranges), where sea-level pressure is an
             extrapolation each model makes its own way; the maps leave it unshaded above MAP_HIGH.
  AO, NAO    each member's anomalies projected on the site's teleconnection patterns (the GEPS and GEFS pages' CPC-style
             projectors on NCEP/NCAR R1 1991-2020: the AO the leading sea-level pressure EOF, the NAO the rotated 500
             hPa mode; gefs_patterns.Ref, release gefs-ref-v1), the cells' anomalies averaged onto the patterns' 2.5 deg
             grid first; in standard deviations of the daily index at that time of year.
  100 hPa    the height anomaly and its 24 h change: where a lobe of the lower-stratospheric vortex fills or deepens.
  10 hPa     the temperature anomaly (the normal: t10_maps.py's ERA5 1991-2020 monthly means, averaged onto the cells)
             under the height, and every member's vortex outline coloured by its shape that day.
Maps show the ensemble mean, an anomaly or change coloured only where enough members give it its sign for a two-sided
sign test at 5 % (agreeing()); a control run, one run, is drawn as it is.

Outputs, model m:
  assets/sst/polarcap_{m}.webp, telecon_{m}.webp   plumes: every member, the members' mean (IFS-ENS: the control)
  assets/sst/anim/polarcap_{m}/ + manifest         sea level and 500 hPa: days 1-5, 6-10, 11-15, then each day
  assets/sst/anim/z100_{m}/ + manifest (+ hres)    100 hPa anomaly and 24 h change, each day
  assets/sst/anim/t10_{m}/ + manifest (+ aifsc)    10 hPa temperature and every member's vortex, days 0-15
  assets/sst/vshape_{m}.webp                       every member's vortex shape each day, the shares beneath
  assets/sst/data/runs/{polarcap,telecon,vshape}_{m}.json (+ hres)   the run records (run_record.py draws the
                                                   run-to-run figures)
  aifs1, a single run: its records and its z100_aifs1 and t10_aifs1 loops only (no plumes, sea-level maps or shape
  barcode of its own)
  --combined: assets/sst/polarcap_models.webp, telecon_models.webp, every model's newest run (strat-combine.yml)

    python polarcap.py --model ifs --date 20261024 --time 00
    python polarcap.py --combined
    python polarcap.py --model gefs --date 20261010 --time 12 --record-only     # the records only (strat-backfill.yml)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

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
FIELDS = ("msl", "z500", "z100", "z10", "t10")  # each member's, on the cells
NORMALS = ("msl", "z500", "z100", "t10")  # the fields with an ERA5 normal (anomalies)
ROWS = tuple(range(round((90.0 - (CLAT[0] - CELL)) / 0.25) + 1))  # 90-18N: the ECMWF 0.25 deg rows the cells need
G = 9.80665
OUT = REPO / "assets" / "sst"
ANIM = OUT / "anim"
ENSEMBLES = ("aifs", "ifs", "gefs", "geps")
CONTROL = {"aifs": "aifsc", "ifs": "hres"}  # an ensemble's control run, drawn on its plumes and recorded on its own
SINGLE = (*CONTROL.values(), "aifs1")  # one run: recorded and mapped on its own, no plumes or shape barcode of its own
CREDIT = {
    "aifs": "ECMWF AIFS-ENS open data (CC BY 4.0), Google Cloud mirror",
    "ifs": "ECMWF IFS-ENS and IFS control open data (CC BY 4.0), Google Cloud mirror",
    "hres": "ECMWF IFS control open data (CC BY 4.0), Google Cloud mirror",
    "aifsc": "ECMWF AIFS-ENS open data (CC BY 4.0), the control (member 0), Google Cloud mirror",
    "aifs1": "ECMWF AIFS single open data (CC BY 4.0), Google Cloud mirror",
    "gefs": "NOAA GEFS (AWS Open Data)",
    "geps": "ECCC GEPS (MSC Datamart)",
}
NEUTRAL = "#d9d6d0"
MSL_LEVELS = np.arange(-20, 21, 2)  # hPa
Z_LEVELS = np.arange(-200, 201, 20)  # m
Z100_LEVELS = np.arange(-400, 401, 40)  # m
T10_LEVELS = np.arange(-36, 37, 4)  # K, as the IFS control's 10 hPa maps (t10_maps.py)
TEND_LEVELS = np.arange(-200, 201, 20)  # m per day
CAP_PANELS = (
    ("msl", "Sea-level pressure, 65–90°N (cells below 500 m)", "hPa"),
    ("z500", "500 hPa height, 65–90°N", "m"),
)
TELE_PANELS = (("ao", "Arctic Oscillation", "sd"), ("nao", "North Atlantic Oscillation", "sd"))
TELE_REF = HERE / "data" / "telecon_ref"  # gefs_patterns_ref.npz, fetched from release gefs-ref-v1 once per job


# ── the grid, the normals, the cap ───────────────────────────────────────────────────────────────────────────────
def coarse(f: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cell means on CLAT x CLON of f (..., lat, lon) on a regular global grid (rows south of the cells left out)."""
    rows = lat >= CLAT[0] - CELL
    h = abs(float(lat[1] - lat[0]))
    return CM.coarsen(f[..., rows, :], lat[rows], lon, h, CLAT, CLON, CELL, 360.0)


class Normals:
    """ERA5 1991-2020 on the cells: sea-level pressure (hPa) at 00, 06, 12 and 18 UTC, 500 and 100 hPa height (m),
    10 hPa temperature (K), the surface height (m)."""

    def __init__(self):
        import xarray as xr

        self.coef = {}
        with xr.open_dataset(HERE / "reference" / "t10_clim.nc") as d:  # 1 deg (t10_maps.py's): averaged onto the cells
            self.coef["t10"] = coarse(d["coef"].values.astype(np.float64), d.latitude.values, d.longitude.values)
        for f in ("msl", "z500", "z100"):  # on the cells already (build_polarcap_clim.py)
            with xr.open_dataset(HERE / "reference" / f"{f}_clim.nc") as d:
                if not (np.allclose(d.latitude, CLAT) and np.allclose(d.longitude, CLON)):
                    raise SystemExit(f"{f}_clim.nc is not on the cells CLAT x CLON")
                if f == "msl":
                    self.coef[f] = {int(h): d["coef"].sel(hour=h).values.astype(np.float64) for h in d.hour.values}
                    self.orog = d["orog"].values.astype(np.float64)
                else:
                    self.coef[f] = d["coef"].values.astype(np.float64)

    def at(self, field: str, valid: pd.Timestamp) -> np.ndarray:
        """The field's normal at that day and hour, (CLAT, CLON)."""
        c = self.coef[field]
        if isinstance(c, dict):
            if valid.hour not in c:
                raise SystemExit(f"no {field} normal for {valid:%H} UTC")
            c = c[valid.hour]
        return HC.evaluate(c, HC.times_of([valid]))[0]


def cap(x: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """The 65-90N area mean of x (..., CLAT, CLON) over the cells `keep` (every cell when None)."""
    rows = CLAT >= CAP
    w = np.cos(np.deg2rad(CLAT[rows]))[:, None] * np.ones(CLON.size)
    if keep is not None:
        w = w * keep[rows]
    return (x[..., rows, :] * w).sum((-2, -1)) / w.sum()


# ── the fields ───────────────────────────────────────────────────────────────────────────────────────────────────
def read_grib(path: Path, short: str, levels=(None,), scale: float = 1.0) -> dict:
    """{level: {(member, step h): cell means}} of the messages of `short` at each of `levels` (None: any level) in a
    GRIB file of global regular grids; a message without a member number is member 0 (the control)."""
    import eccodes

    out, grid = {lev: {} for lev in levels}, None
    with open(path, "rb") as fh:
        while (gid := eccodes.codes_grib_new_from_file(fh)) is not None:
            try:
                if eccodes.codes_get(gid, "shortName") != short:
                    continue
                lev = None if levels == (None,) else eccodes.codes_get(gid, "level")
                if lev not in out:
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
                out[lev][num, step] = coarse(v, lat, lon)
            finally:
                eccodes.codes_release(gid)
    return out


def _cells(v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return coarse(v.astype(np.float64), lat, lon).astype(np.float32)


def read_rows(cyc, spec, scale: float = 1.0) -> dict:
    """read_grib's {level: {(member, step h): cell means}} (level None at the surface) of an AIFS-ENS or AIFS single
    spec, from the 90-18N rows of each message (store.ensure_rows, ~45 % of its bytes)."""
    import store as E

    a, mems, steps, levs = E.load_rows(E.ensure_rows(cyc, spec, ROWS, reduce=_cells, tag="cells"))
    return {
        (L or None): {(m, s): a[i, j, k] * scale for i, m in enumerate(mems) for j, s in enumerate(steps) if s in LEADS}
        for k, L in enumerate(levs)
    }


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
    got = {f: [] for f in FIELDS}
    for kind in ("cf", "pf"):
        got["msl"].append(read_rows(cyc, E.Spec("aifs-ens", kind, "msl", "sfc", (), steps), 0.01)[None])
        z = read_grib(E.ensure(cyc, E.Spec("aifs-ens", kind, "z", "pl", WM.LEVELS, steps)), "z", (500, 100, 10), 1 / G)
        for lev in (500, 100, 10):  # the wave maps' file
            got[f"z{lev}"].append(z[lev])
        got["t10"].append(read_rows(cyc, E.Spec("aifs-ens", kind, "t", "pl", (10,), steps))[10])
    return {
        "aifs": {f: stack(*parts) for f, parts in got.items()},
        "aifsc": {f: stack(parts[0]) for f, parts in got.items()},
    }


def load_aifs1(date: str, hh: str) -> dict:
    """AIFS single: msl and t at 10 hPa from the rows, gh at 500, 100 and 10 hPa from the shared files (aifs_single.py;
    read_grib keeps the daily steps of the 12-hourly one)."""
    import aifs_single as A1
    import store as E

    cyc = E.Cycle(date, hh)
    msl = read_rows(cyc, E.Spec("aifs-single", "fc", "msl", "sfc", (), A1.STEPS), 0.01)[None]
    t10 = read_rows(cyc, E.Spec("aifs-single", "fc", "t", "pl", (10,), A1.STEPS))[10]
    z = {lev: read_grib(A1.height(cyc, lev), "gh", (lev,))[lev] for lev in A1.HEIGHTS}
    return {"aifs1": {"msl": stack(msl), **{f"z{lev}": stack(z[lev]) for lev in A1.HEIGHTS}, "t10": stack(t10)}}


def load_ifs(date: str, hh: str) -> dict:
    """IFS-ENS (when complete on the mirror) and the IFS control, from the shared files. The control has no 10 hPa
    temperature here: its own maps of it (t10_maps.py) are 12-hourly and in both hemispheres."""
    import ifs_ens as IE

    cyc, out = IE.E.Cycle(date, hh), {}
    for m, ready, msl, gh, t10 in (
        ("ifs", IE.published, IE.spec_msl, IE.gh_spec, IE.spec_t10),
        ("hres", IE.ctrl_published, IE.spec_ctrl_msl, lambda lev: IE.spec_ctrl_gh(), None),
    ):
        if ready(date, hh):
            z = {lev: read_grib(IE.ensure(cyc, gh(lev)), "gh", (lev,))[lev] for lev in (500, 100, 10)}
            out[m] = {"msl": stack(read_grib(IE.ensure(cyc, msl()), "msl", scale=0.01)[None]),
                      **{f"z{lev}": stack(z[lev]) for lev in (500, 100, 10)}}  # fmt: skip
            if t10 is not None:
                out[m]["t10"] = stack(read_grib(IE.ensure(cyc, t10()), "t", (10,))[10])
    return out


def load_strat_ens(model: str, date: str, hh: str) -> dict:
    import strat_ens as SE

    d = SE.load(model, date, hh, ("polarcap",))
    keep = SE.complete_members(d, "pc_z500")
    return {model: {f: d[f"pc_{f}"][keep] for f in FIELDS if f"pc_{f}" in d}}


# ── one run ──────────────────────────────────────────────────────────────────────────────────────────────────────
def anomalies(init: pd.Timestamp, fields: dict, N: Normals) -> tuple[dict, pd.DatetimeIndex]:
    """The members' anomalies {field: (member, lead, CLAT, CLON)} of the fields with a normal, and the valid times."""
    valid = pd.DatetimeIndex([init + pd.Timedelta(hours=h) for h in LEADS])
    return {f: x - np.stack([N.at(f, v) for v in valid])[None] for f, x in fields.items() if f in NORMALS}, valid


def caps(A: dict, N: Normals) -> dict:
    """{msl, z500: (member, lead)} cap anomalies: sea-level pressure over the cells below HIGH, 500 hPa height over all."""
    return {"msl": cap(A["msl"], N.orog < HIGH), "z500": cap(A["z500"])}


def teleconnections(A: dict, valid: pd.DatetimeIndex) -> dict:
    """{ao, nao: (member, lead)} in seasonal standard deviations of the daily index (gefs_patterns.Ref.index)."""
    sys.path.insert(0, str(REPO / "scripts" / "gefs"))
    import gefs_patterns as GP

    ref = GP.Ref(GP.ref_path(GP.REF_NAME, TELE_REF))
    lat25 = GP.LAT25
    rows = lat25 - 2.5 / 2 >= CLAT[0] - CELL / 2  # the 2.5-deg rows whose cells the 1.5-deg cells cover: 20 N and north
    months = np.array([t.month for t in valid])
    doys = np.array([GP.doy(t.date()) for t in valid])
    out = {}
    for sid, f in (("ao", "msl"), ("nao", "z500")):
        a = A[f]
        on25 = np.full((*a.shape[:2], GP.LAT25.size, GP.LON25.size), np.nan)
        on25[..., rows, :] = CM.coarsen(a, CLAT, CLON, CELL, lat25[rows], GP.LON25, 2.5, 360.0)
        n, L = a.shape[:2]
        flat = ref.index(sid, on25.reshape(n * L, *on25.shape[2:]), np.tile(months, n), np.tile(doys, n))
        out[sid] = flat.reshape(n, L)
    return out


def record(product: str, model: str, init: pd.Timestamp, series: dict, valid: pd.DatetimeIndex, scalars=None) -> None:
    frames = {k: pd.DataFrame(v.T, index=valid) for k, v in series.items()}
    RR.record(product, model, init, RR.entry(product, frames, scalars))
    RR.render(product, model)


def agreeing(n: int) -> int:
    """The fewest of n members that must share a sign for a two-sided sign test against an even split to pass at 5 %
    (16 of 21, 22 of 31, 33 of 50 or 51)."""
    from scipy.stats import binom

    return next(k for k in range(n // 2 + 1, n + 1) if 2 * binom.sf(k - 1, n, 0.5) < 0.05)


def agreed(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the members' mean, where enough members share its sign: agreeing) for x (member, ...) without gaps; a single
    run is shown everywhere."""
    m = x.mean(0)
    if x.shape[0] == 1:
        return m, np.ones(m.shape, bool)
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


def plumes(model: str, init: pd.Timestamp, valid, series: dict, panels, title: str, ctrl=None, out=None) -> Path:
    """Every member's series (member, lead), the members' mean bold (the control dashed), one panel per (key, title,
    unit)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import house

    house.apply()

    col, lab = RR.MODEL_COLOUR[model], RR.MODEL_LABEL[model]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11.5, 8.2), sharex=True)
    for ax, (key, name, unit) in zip(axes, panels):
        x = series[key]
        for row in x:
            ax.plot(valid, row, color=col, lw=0.8, alpha=0.35, zorder=3)
        ax.plot(valid, np.nanmean(x, 0), color=col, lw=2.8, zorder=5, label=f"{lab} mean, {x.shape[0]} members")
        if ctrl is not None:
            ax.plot(valid, ctrl[key], color="#1f1f1f", lw=1.6, ls="--", zorder=6, label=RR.MODEL_LABEL[CONTROL[model]])
        late = (valid >= init + pd.Timedelta(days=6)) & (valid <= init + pd.Timedelta(days=15))
        lm = np.nanmean(x[:, late], 1)
        fmt = "{:+.0f}" if unit == "m" else "{:+.1f}"
        ax.set_title(f"{name}: anomaly from ERA5 1991–2020" if unit != "sd" else name, fontsize=11,
                     fontweight="bold", loc="left", pad=16)  # fmt: skip
        ax.text(
            0,
            1.015,
            f"days 6–15: members' mean {fmt.format(np.nanmean(lm)).replace('-', '−')} {unit}, "
            f"{(lm > 0).mean():.0%} of the members above zero",
            transform=ax.transAxes,
            fontsize=9,
            color="0.25",
        )
        _style(ax, unit if unit != "sd" else "sd of the daily index")
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    fig.suptitle(f"{lab} · {title} · run {init:%Y-%m-%d %HZ}", fontsize=13.5, fontweight="bold", x=0.07, ha="left",
                 y=0.985)  # fmt: skip
    fig.text(0.07, 0.945, SUBTITLE[panels], fontsize=9.5, color="0.25")
    fig.text(0.07, 0.008, textwrap.fill(f"{NOTE[panels]} {CREDIT[model]}.", 190), fontsize=8, color="#6f6b64",
             linespacing=1.4)  # fmt: skip
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.085, hspace=0.32)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out


SUBTITLE = {
    CAP_PANELS: "every member thin, the members' mean bold · above zero: high pressure and heights over the Arctic "
    "(the negative Arctic Oscillation)",
    TELE_PANELS: "every member thin, the members' mean bold · negative AO and NAO: high pressure over the Arctic and "
    "Greenland, the westerlies pushed south",
}
NOTE = {
    CAP_PANELS: "Raw model output against the ERA5 normal of each day and hour.",
    TELE_PANELS: "Raw model output: anomalies against ERA5 1991–2020 on the site's CPC-style patterns (NCEP/NCAR R1); "
    "in standard deviations of the daily index at this time of year.",
}


def _pole(a: np.ndarray) -> np.ndarray:
    """The cells' 90N row (wedges of the pole cap, a different value at each longitude) as one value, the pole's: a map
    draws the row as a single point, where differing values would meet in a star."""
    a = a.copy()
    a[-1] = a[-1].mean() >= 0.5 if a.dtype == bool else a[-1].mean()
    return a


def _panel(fig, pos, mean, anom, gate, levels, contour_step, label, title, high=None):
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
    cb.set_label(label, fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)


ANOM_LABEL = "anomaly from ERA5 1991–2020, {}"


def map_frame(model, init, span, n, mm, ma, mg, zm, za, zg, high, out: Path) -> None:
    """Two polar panels: the mean sea-level pressure (hPa) and 500 hPa height (zm, dam) contoured, their anomalies (m)
    shaded where the gates mg, zg pass."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = RR.MODEL_LABEL[model]
    fig = plt.figure(figsize=(13.0, 7.4), dpi=110)
    _panel(fig, [0.02, 0.175, 0.46, 0.69], mm, ma, mg, MSL_LEVELS, 4, ANOM_LABEL.format("hPa"), "Sea-level pressure",
           high)  # fmt: skip
    _panel(fig, [0.52, 0.175, 0.46, 0.69], zm, za, zg, Z_LEVELS, 6, ANOM_LABEL.format("m"), "500 hPa height")
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
    manifest(f"polarcap_{model}", init, f"Sea-level pressure and 500 hPa height, {RR.MODEL_LABEL[model]}", frames)
    print(f"  {len(frames)} map frames -> {d}", flush=True)


def manifest(rid: str, init: pd.Timestamp, label: str, frames: list) -> None:
    man = {"ver": f"{init:%Y%m%d%H}", "days": len(frames), "selectorLabel": "Model", "default": rid,
           "regions": {rid: {"label": label, "n_frames": len(frames), "frames": [dict(f, idx=i) for i, f in enumerate(frames)]}}}  # fmt: skip
    (ANIM / f"{rid}_manifest.json").write_text(json.dumps(man))


def z100_frame(model, init, valid, n, zm, za, zg, tm, tg, out: Path) -> None:
    """Two polar panels at 100 hPa: the height (zm, dam) contoured on both, its anomaly and its 24 h change shaded."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = RR.MODEL_LABEL[model]
    fig = plt.figure(figsize=(13.0, 7.4), dpi=110)
    _panel(fig, [0.02, 0.175, 0.46, 0.69], zm, za, zg, Z100_LEVELS, 16, ANOM_LABEL.format("m"), "Height anomaly")
    _panel(fig, [0.52, 0.175, 0.46, 0.69], zm, tm, tg, TEND_LEVELS, 16, "change over the last 24 h, m",
           "Height change over 24 h")  # fmt: skip
    fig.text(0.02, 0.965, f"{lab} · 100 hPa height: {valid:%a %-d %b %HZ} (day {(valid - init).days})", fontsize=14,
             fontweight="bold")  # fmt: skip
    how = (
        "one run, drawn as it is"
        if n == 1
        else f"ensemble mean; shaded where at least {agreeing(n)} of the {n} members agree on the sign (a sign test at 5 %)"
    )
    fig.text(0.02, 0.928, f"Run {init:%Y-%m-%d %HZ} · black: height, every 16 dam · {how}", fontsize=9.5, color="#333")
    fig.text(
        0.02,
        0.008,
        ("Grey: the members are split. " if n > 1 else "")
        + "The change is this day's height minus the day before's, so a lobe of the vortex filling in shows red over "
        f"it.\nRaw model output against ERA5 1991–2020 at each day and hour. {CREDIT[model]}.",
        fontsize=8,
        color="#6f6b64",
        linespacing=1.4,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def z100_maps(model: str, init: pd.Timestamp, z, az) -> None:
    """Each day 1-15: the 100 hPa height anomaly and 24 h change (ensemble means, gated; the control as it is)."""
    d = ANIM / f"z100_{model}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.webp"):
        old.unlink()
    frames = []
    for k in range(1, len(LEADS)):
        valid = init + pd.Timedelta(hours=LEADS[k])
        za, zg = agreed(az[:, k])
        tm, tg = agreed(z[:, k] - z[:, k - 1])
        z100_frame(model, init, valid, z.shape[0], z[:, k].mean(0) / 10.0, za, zg, tm, tg, d / f"D{k:02d}.webp")
        frames.append(
            {"file": f"D{k:02d}.webp", "date": f"{valid:%Y-%m-%d}", "label": f"Day {k} · {valid:%a %-d %b %HZ}"}
        )
    manifest(f"z100_{model}", init, f"100 hPa height, {RR.MODEL_LABEL[model]}", frames)
    print(f"  {len(frames)} 100 hPa frames -> {d}", flush=True)


def _vortices(fig, pos, z, states, title) -> None:
    """Every member's vortex: its contour at the split level of z (member, CLAT, CLON), coloured by its shape that
    day (vortex_shape.py), the splits drawn last."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import vortex_shape as VS
    import vortex_winds as VW
    from matplotlib.lines import Line2D

    pc = ccrs.PlateCarree()
    ax = fig.add_axes(pos, projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, CLAT[0] + 0.5, 90], crs=pc)
    VW.circular_boundary(ax)
    lon = np.append(CLON, 360.0)
    lw = 1.8 if len(z) == 1 else 0.8
    for i in sorted(range(len(z)), key=lambda i: (states[i] == 2, states[i] == 1)):
        f = _pole(z[i])
        ax.contour(lon, CLAT, np.concatenate([f, f[:, :1]], axis=1), levels=[VS.EDGE - VS.SPLIT_OFFSET],
                   colors=[VS.LINE_COLOUR[VS.STATES[states[i]]]], linewidths=lw, transform=pc, zorder=3)  # fmt: skip
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#2b2b2b", linewidth=0.55, zorder=4)
    gl = ax.gridlines(linewidth=0.3, color="#8a8a8a", alpha=0.5, zorder=5)
    gl.ylocator = plt.FixedLocator([30, 45, 60, 75])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    counts = [(k, int((states == VS.STATES.index(k)).sum())) for k in VS.STATES]
    handles = [Line2D([], [], color=VS.LINE_COLOUR[k], lw=2) for k, c in counts if c]
    labels = [f"{VS.STATE_LABEL[k]}: {c}" for k, c in counts if c]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(pos[0] + pos[2] / 2, pos[1] - 0.012), ncol=2,
               fontsize=8.5, frameon=False)  # fmt: skip


def t10_frame(model, init, valid, n, zm, ta, tg, z, states, out: Path) -> None:
    """Two polar panels at 10 hPa: the temperature anomaly shaded under the height (zm, dam), and every member's
    vortex."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import vortex_shape as VS

    lab = RR.MODEL_LABEL[model]
    fig = plt.figure(figsize=(13.0, 7.4), dpi=110)
    _panel(fig, [0.02, 0.175, 0.46, 0.69], zm, ta, tg, T10_LEVELS, 24, ANOM_LABEL.format("K"),
           "Temperature anomaly")  # fmt: skip
    _vortices(fig, [0.52, 0.175, 0.46, 0.69], z, states, "The vortex" if n == 1 else "Every member's vortex")
    fig.text(0.02, 0.965, f"{lab} · 10 hPa temperature and the vortex: {valid:%a %-d %b %HZ} (day {(valid - init).days})",
             fontsize=14, fontweight="bold")  # fmt: skip
    how = (
        "one run, drawn as it is"
        if n == 1
        else f"ensemble mean; shaded where at least {agreeing(n)} of the {n} members agree on the sign (a sign test at 5 %)"
    )
    fig.text(0.02, 0.928, f"Run {init:%Y-%m-%d %HZ} · black: height, every 24 dam · {how}", fontsize=9.5, color="#333")
    fig.text(
        0.02,
        0.008,
        ("Grey: the members are split. " if n > 1 else "")
        + f"Right: the {VS.EDGE - VS.SPLIT_OFFSET:,.0f} m height line of "
        + ("the run" if n == 1 else "each member")
        + " (150 m inside the winter vortex edge), coloured by the vortex's shape that day.\n"
        f"Raw model output against ERA5 1991–2020 at each day. {CREDIT[model]}.",
        fontsize=8,
        color="#6f6b64",
        linespacing=1.4,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def t10_maps(model: str, init: pd.Timestamp, at, z, shape: dict) -> None:
    """Each day 0-15: the 10 hPa temperature anomaly (the ensemble mean, gated; a single run as it is) under the mean
    height, and every member's vortex."""
    d = ANIM / f"t10_{model}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.webp"):
        old.unlink()
    frames = []
    for k in range(len(LEADS)):
        valid = init + pd.Timedelta(hours=LEADS[k])
        ta, tg = agreed(at[:, k])
        t10_frame(model, init, valid, z.shape[0], z[:, k].mean(0) / 10.0, ta, tg, z[:, k], shape["state"][:, k],
                  d / f"D{k:02d}.webp")  # fmt: skip
        frames.append(
            {"file": f"D{k:02d}.webp", "date": f"{valid:%Y-%m-%d}", "label": f"Day {k} · {valid:%a %-d %b %HZ}"}
        )
    manifest(f"t10_{model}", init, f"10 hPa temperature and the vortex, {RR.MODEL_LABEL[model]}", frames)
    print(f"  {len(frames)} 10 hPa frames -> {d}", flush=True)


def combined(product: str, panels, title: str, out: Path, max_age_days: float = 2.0) -> Path | None:
    """Every model's newest run from the records: the members' mean and 10-90 % band, the single runs dashed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    newest = {m: rs[-1] for m in (*ENSEMBLES, *SINGLE) if (rs := RR.runs(product, m))}
    if not newest:
        return None
    t1 = max(pd.Timestamp(r["init"]) for r in newest.values())
    newest = {m: r for m, r in newest.items() if t1 - pd.Timestamp(r["init"]) <= pd.Timedelta(days=max_age_days)}
    fig, axes = plt.subplots(len(panels), 1, figsize=(11.5, 8.2), sharex=True)
    for ax, (key, name, unit) in zip(axes, panels):
        for m, r in newest.items():
            mean = RR.series(r, key)
            col = RR.MODEL_COLOUR[m]
            lab = f"{RR.MODEL_LABEL[m]} {pd.Timestamp(r['init']):%-d %b %HZ}"
            if m in SINGLE:
                ax.plot(mean.index, mean.values, color="#1f1f1f" if m == "hres" else col, lw=1.6, ls="--", zorder=6,
                        label=lab)  # fmt: skip
                continue
            ax.fill_between(mean.index, RR.series(r, key, "p10"), RR.series(r, key, "p90"), color=col, alpha=0.13, lw=0)
            ax.plot(mean.index, mean.values, color=col, lw=2.4, zorder=5, label=f"{lab}, {r['members']} members")
        ax.set_title(f"{name}: anomaly from ERA5 1991–2020" if unit != "sd" else name, fontsize=11, fontweight="bold",
                     loc="left")  # fmt: skip
        _style(ax, unit if unit != "sd" else "sd of the daily index")
        ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.9)
    fig.suptitle(f"Every model's newest run · {title}", fontsize=13.5, fontweight="bold",
                 x=0.07, ha="left", y=0.985)  # fmt: skip
    fig.text(0.07, 0.945, SUBTITLE[panels].split(" · ", 1)[1], fontsize=9.5, color="0.25")
    fig.text(0.07, 0.008, textwrap.fill(f"Each model's members' mean and 10–90 % range, the single runs dashed, each "
             f"labelled with its run. {NOTE[panels]}", 190), fontsize=8, color="#6f6b64", linespacing=1.4)  # fmt: skip
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.085, hspace=0.32)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


CAP_TITLE = "the troposphere under the polar vortex"
TELE_TITLE = "the Arctic and North Atlantic Oscillations"


def run(model: str, date: str, hh: str, figures: bool = True) -> int:
    init = pd.Timestamp(f"{date} {hh}:00")
    N = Normals()
    load = {"aifs": load_aifs, "aifs1": load_aifs1, "ifs": load_ifs}.get(model) or (
        lambda d, h: load_strat_ens(model, d, h)
    )
    got = load(date, hh)
    if not got:
        print(f"::warning::{model} {date} {hh}Z: nothing on the mirror to draw", flush=True)
        return 1
    import vortex_shape as VS

    done, ctrls, shapes = {}, {}, {}
    for m, raw in got.items():
        full = np.logical_and.reduce([np.isfinite(x).all((1, 2, 3)) for x in raw.values()])  # every lead, field
        fields = {f: x[full] for f, x in raw.items()}
        A, valid = anomalies(init, fields, N)
        cp, tc = caps(A, N), teleconnections(A, valid)
        print(f"  {RR.MODEL_LABEL[m]}: {full.sum()} members; cap {np.nanmean(cp['msl'][:, 0]):+.1f} hPa, "
              f"{np.nanmean(cp['z500'][:, 0]):+.0f} m at the start, {np.nanmean(cp['msl'][:, 6:]):+.1f} hPa, "
              f"{np.nanmean(cp['z500'][:, 6:]):+.0f} m days 6-15; AO {np.nanmean(tc['ao'][:, 0]):+.2f} -> "
              f"{np.nanmean(tc['ao'][:, 6:]):+.2f}, NAO {np.nanmean(tc['nao'][:, 0]):+.2f} -> {np.nanmean(tc['nao'][:, 6:]):+.2f}",
              flush=True)  # fmt: skip
        record("polarcap", m, init, cp, valid)
        record("telecon", m, init, tc, valid)
        if "z100" in fields and figures:
            z100_maps(m, init, fields["z100"], A["z100"])
        if "z10" in fields:
            shapes[m] = vs = VS.shape(fields["z10"], CLAT, CLON)
            share = [f"{100 * (vs['state'][:, k] == 2).mean():.0f}" for k in (0, 5, 10, 15)]
            print(f"  {RR.MODEL_LABEL[m]}: vortex split in {'/'.join(share)} % of members on days 0/5/10/15, on any "
                  f"day 1-15 in {VS.any_split(vs):.0%}", flush=True)  # fmt: skip
            record("vshape", m, init, VS.series(vs), valid, {"anysplit": round(VS.any_split(vs), 3)})
            if "t10" in fields and figures:
                t10_maps(m, init, A["t10"], fields["z10"], vs)
        if m in SINGLE:
            ctrls[m] = {**{k: v[0] for k, v in cp.items()}, **{k: v[0] for k, v in tc.items()}}
        else:
            done[m] = (fields, A, cp, tc, valid)
    if not figures:
        return 0
    for m, (fields, A, cp, tc, valid) in done.items():
        c = ctrls.get(CONTROL.get(m))
        plumes(m, init, valid, cp, CAP_PANELS, CAP_TITLE, c, OUT / f"polarcap_{m}.webp")
        plumes(m, init, valid, tc, TELE_PANELS, TELE_TITLE, c, OUT / f"telecon_{m}.webp")
        maps(m, init, fields["msl"], fields["z500"], A["msl"], A["z500"], N)
        if m in shapes:
            cm = CONTROL.get(m)
            cs = {**shapes[cm], "model": cm} if cm in shapes else None
            VS.members(m, init, shapes[m], cs, CREDIT[m], OUT / f"vshape_{m}.webp")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=(*ENSEMBLES, "aifs1"))
    ap.add_argument("--date")
    ap.add_argument("--time")
    ap.add_argument("--combined", action="store_true", help="every model's newest run from the records")
    ap.add_argument("--record-only", action="store_true", help="the run into the records, no figures (a backfill)")
    a = ap.parse_args()
    if a.combined:
        combined("polarcap", CAP_PANELS, CAP_TITLE, OUT / "polarcap_models.webp")
        combined("telecon", TELE_PANELS, TELE_TITLE, OUT / "telecon_models.webp")
        return 0
    if not (a.model and a.date and a.time):
        ap.error("--model, --date and --time are required (or --combined)")
    return run(a.model, a.date, f"{int(a.time):02d}", figures=not a.record_only)


if __name__ == "__main__":
    raise SystemExit(main())
