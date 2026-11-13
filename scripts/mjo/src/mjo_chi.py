#!/usr/bin/env python3
"""
The MJO in 200 hPa velocity potential with the El Niño pattern taken out: one time-longitude chart, the last
months of analyses joined to every member of the AIFS-ENS forecast.

  record     the AIFS-ENS 0-h analyses of the tropical χ history (walker_chi), averaged 15°S-15°N, one value a day:
             the mean of the 00 and 12 UTC analyses, a day holding one of them moved to that mean by the record's
             mean 00-minus-12 UTC difference
  forecast   the control and the 50 perturbed members at each 00 UTC valid time to day 15, χ by the same solver on the
             same grid, less the model's drift: the mean error of the runs of the last DRIFT_WINDOW days against the
             analyses, at each longitude a quadratic in lead time through zero, zonal wavenumbers ≤ DRIFT_KMAX. Those
             runs' forecasts are a rolling archive beside the χ history; a seed of control forecasts
             (build_mjo_chi_drift.py) covers its first weeks
  El Niño    removed by regression, not by a running mean that would take the slow MJO with it: the ERA5 1991-2020
             normal and trend, then the walker_chi_clim.nc coefficients on Niño-3.4 and on Niño-3.4 three months
             earlier times the daily OISST Niño-3.4, detrended, 31-day mean; then the record's own mean of what is
             left at each longitude, the standing part the regression misses. The regression is taken out at every
             longitude (a field cut at the edge of its significant region would leave that edge in what is filtered);
             the strip above the chart draws it only where it is significant

What remains is filtered in wavenumber-frequency space member by member, the member's forecast standing in for the
future the filter needs: the MJO's band (eastward wavenumbers 1-5, periods 30-96 days) and the Kelvin wedge
(kelvin_tracker). Past the last analysis the member mean is drawn only where the members agree on its sign
(two-sided sign test, false discovery rate FDR over the cells tested); the filtered fields are tested over the last
TEST_BACK analysis days as well, which the forecast still shapes.

The column beside the chart reads the MJO's band as an RMM phase: each member's field fitted by the two patterns that
go with the observed RMM1 and RMM2 in ERA5 (build_mjo_chi_phase.py; within one phase of the RMM on 94 % of the days
both are active). A phase is given where the member mean's amplitude is 1 or more and PHASE_SHARE of the members
point within 45° of it; below 1 there is no MJO and no phase.

Output: assets/mjo/chi_hov.webp, and the run's own entry in the drift archive.

    python mjo_chi.py --selftest
    python mjo_chi.py --date 20261113 --time 00
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import house

house.apply()
from matplotlib.colors import BoundaryNorm
from quantmet.stats import benjamini_hochberg
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import kelvin_tracker as K
import walker_chi as wc
from enso_rmm import phase_of

REF = HERE.parent / "data" / "reference"
BAND = 15.0  # degrees of latitude either side of the equator
RECORD, MIN_RECORD, SHOW = 130, 75, 110  # analysis days the filters run on (most, fewest); days of them drawn
MJO_K, MJO_DAYS = (1, 5), (30.0, 96.0)  # eastward zonal wavenumbers and periods of the MJO's band
PAD, RAMP = 120, 10  # days of padding after the last forecast day, the first RAMP of them a slide to zero
MIN_PAIRS = 30  # days holding both analyses, for the 00-minus-12 UTC difference
LEADS = np.arange(12, 361, 12)  # hours: the drift archive's leads (a 00 UTC run fills the days, a 12 UTC run the rest)
DRIFT_WINDOW = 60  # days of earlier runs behind the drift
DRIFT_KMAX = 6  # zonal wavenumbers kept in it
DRIFT_MIN = 8  # runs verified to day 14 or later, below which there is no drift and no figure
DRIFT_KEEP = 2 * DRIFT_WINDOW + 10  # runs kept in the rolling archive
DRIFT_NAME = "mjo_chi_drift.nc"
DRIFT_SEED = REF / "mjo_chi_drift_seed.nc"
PHASE_REF = REF / "mjo_chi_phase.nc"  # build_mjo_chi_phase.py
FDR = 0.05  # false discovery rate of the sign tests
TEST_BACK = 30  # analysis days before the forecast over which the filtered fields are tested too
PHASE_SHARE = 0.75  # of the members within 45° of the member mean's angle, for a phase to be given
SST_STALE = 14  # days: an SST record ending longer than this before the run is not used
SHADE = [-12, -8, -5, -3, -1.5, 1.5, 3, 5, 8, 12]  # 10⁶ m² s⁻¹
MJO_STEP, KELVIN_LEVELS = 1.5, [-5.0, -2.5]
KELVIN_COLOUR = "#7b2d8e"
PHASE_TINT = "#b9c6d8"  # the amplitude bar on the days with a phase


# ---- band mean and filters ------------------------------------------------------------------- #
def band_mean(chi: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """χ (..., lat, lon) in m² s⁻¹ -> its area-weighted mean over BAND, (..., lon) in 10⁶ m² s⁻¹."""
    m = np.abs(lat) <= BAND
    w = np.cos(np.deg2rad(lat[m]))
    return (chi[..., m, :] * (w / w.sum())[:, None]).sum(-2) / 1e6


def mjo_mask(nt: int, nlon: int) -> np.ndarray:
    """Boolean (frequency, wavenumber) mask of the MJO's band, in numpy's fft2 layout (signs as in kelvin_mask)."""
    f = np.fft.fftfreq(nt, d=1.0)
    s = np.fft.fftfreq(nlon, d=1.0 / nlon)
    F, S = np.meshgrid(f, s, indexing="ij")
    k = np.where(F > 0, -S, S)
    af = np.abs(F)
    return (k >= MJO_K[0]) & (k <= MJO_K[1]) & (af >= 1.0 / MJO_DAYS[1]) & (af <= 1.0 / MJO_DAYS[0])


def selftest() -> int:
    nt, nx = 150, 172
    t = np.arange(nt)[:, None]
    lon = np.arange(nx)[None, :] * 2 * np.pi / nx
    for name, x, keep in (
        ("eastward k=2 45 d", np.cos(2 * lon - 2 * np.pi * t / 45.0), True),
        ("westward k=2 45 d", np.cos(2 * lon + 2 * np.pi * t / 45.0), False),
        ("eastward k=5 8 d", np.cos(5 * lon - 2 * np.pi * t / 8.0), False),  # a Kelvin wave
        ("standing k=1", np.cos(lon) + 0 * t, False),
    ):
        y = K.wave_filter(x[:, None, :].astype(float), mjo_mask, PAD, RAMP)[:, 0]
        r = np.std(y[20:-20]) / np.std(x[20:-20])
        ok = r > 0.7 if keep else r < 0.1
        print(f"  MJO band, {name}: kept {r:.2f} of the amplitude -> {'ok' if ok else 'FAIL'}")
        if not ok:
            return 1
    return K.selftest()


# ---- the analyses ---------------------------------------------------------------------------- #
def observed(hist: xr.DataArray, init: pd.Timestamp):
    """The record up to `init`: (days, day (n, lon), a00, doff). `day` is the band-mean χ200 of each day (the mean of
    its two analyses), contiguous, RECORD days at most; `a00` the 00 UTC analyses on their own, for the drift; `doff`
    the mean of 00 UTC minus the day's mean, by longitude. Raises LookupError when the record cannot carry the chart."""
    h = hist.sel(level=200, time=slice(init.normalize() - pd.Timedelta(days=RECORD - 1), init))
    if h.sizes["time"] == 0:
        raise LookupError("no analysis in the χ history's last months")
    b = xr.DataArray(
        band_mean(h.values.astype(float), h.latitude.values),
        dims=("time", "longitude"),
        coords={"time": h.time.values, "longitude": h.longitude.values},
    )
    t = pd.DatetimeIndex(b.time.values)
    a00, a12 = (b.isel(time=t.hour == hh).assign_coords(time=t[t.hour == hh].normalize()) for hh in (0, 12))
    both = np.intersect1d(a00.time.values, a12.time.values)
    if both.size < MIN_PAIRS:
        raise LookupError(
            f"{both.size} days hold both analyses; {MIN_PAIRS} are needed for the 00-minus-12 UTC difference"
        )
    doff = ((a00.sel(time=both) - a12.sel(time=both)) / 2).mean("time")
    have = np.union1d(a00.time.values, a12.time.values)
    x00, x12 = a00.reindex(time=have), a12.reindex(time=have)
    day = xr.where(x12.isnull(), x00 - doff, xr.where(x00.isnull(), x12 + doff, (x00 + x12) / 2))
    # up to three missing days in a row are interpolated; the record starts after the last longer hole
    df = pd.DataFrame(day.transpose("time", "longitude").values, index=pd.DatetimeIndex(have))
    df = df.reindex(pd.date_range(have[0], have[-1], freq="D")).interpolate(limit=3, limit_area="inside").dropna()
    brk = np.flatnonzero(np.diff(df.index.values).astype("timedelta64[D]").astype(int) > 1)
    if brk.size:
        df = df.iloc[brk[-1] + 1 :]
    days = df.index
    if len(days) < MIN_RECORD:
        raise LookupError(f"{len(days)} contiguous analysis days from {days[0]:%Y-%m-%d}; {MIN_RECORD} are needed")
    if days[-1] < init.normalize() - pd.Timedelta(days=1):
        raise LookupError(f"the last analysis is {days[-1]:%Y-%m-%d}, more than a day before the run")
    return days, df.values, a00, doff.values


# ---- the El Niño part ------------------------------------------------------------------------ #
def nino34(path: Path, wref: xr.Dataset) -> pd.Series:
    """Daily OISST Niño-3.4 anomaly less the 1991-2020 trend (the regression's predictor is detrended), as a 31-day
    centred mean; the last 15 days are the mean of the days there are."""
    e = json.loads(path.read_text())["daily"]
    s = pd.Series(e["nino34"], index=pd.to_datetime(e["dates"]), dtype=float).asfreq("D").interpolate()
    (la0, la1), (lo0, lo1) = json.loads(wref.attrs["boxes"])["n34"]
    la, lo = wref.lat.values, wref.lon_e.values
    j, i = (la >= la0) & (la <= la1), (lo >= lo0) & (lo <= lo1)
    w = np.cos(np.deg2rad(la[j]))[:, None]
    slope = (wref.sst_slope.values[:, j][:, :, i] * w).sum((1, 2)) / (w.sum() * i.sum())  # °C a year, by month
    yr = s.index.year + (s.index.dayofyear - 1) / 365.25
    return (s - slope[s.index.month - 1] * (yr - 2005.5)).rolling(31, center=True, min_periods=16).mean()


def enso_coefs(clim: xr.Dataset, lat: np.ndarray, nlon: int):
    """Band means (12 months, lon) of χ200's regression coefficients on Niño-3.4 and on Niño-3.4 three months
    earlier, 10⁶ m² s⁻¹ per °C; and where (12 months, lon) the ENSO regression is significant over at least half of
    the band's area."""
    coefs = [band_mean(clim.coef.sel(pred=p).values[..., :nlon], lat) for p in ("n34", "n34_lag3")]
    sig = np.stack([wc.r2_pvalues(clim, m)[0] for m in range(1, 13)])[..., :nlon]
    return coefs, band_mean(sig * 1e6, lat) >= 0.5


def monthly(c: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    """c (12, ...) holding each month's value at its middle -> linear in time between the middles, on `times`."""
    pos = times.month.values - 1 + (times.day.values - 0.5) / times.days_in_month.values - 0.5
    i0 = np.floor(pos).astype(int)
    w = (pos - i0).reshape(-1, *(1,) * (c.ndim - 1))
    return (1 - w) * c[i0 % 12] + w * c[(i0 + 1) % 12]


def enso_part(times: pd.DatetimeIndex, sst: pd.Series, coefs):
    """The regression's El Niño part of the band-mean χ200 on `times` (t, lon), with the two Niño-3.4 values used.
    Past the SST record's end the last value is held."""
    n34 = sst.reindex(times, method="ffill").values
    lag = sst.reindex(times - pd.Timedelta(days=91), method="ffill").values
    return monthly(coefs[0], times) * n34[:, None] + monthly(coefs[1], times) * lag[:, None], n34, lag


# ---- the forecast and its drift -------------------------------------------------------------- #
def member_forecast(date: str, hh: str, lat: np.ndarray, lon: np.ndarray):
    """Band-mean χ200 of every member at the run's 00 UTC valid times: (lead hours, χ (lead, member, lon)), the control
    first. The winds are the fields the run's RMM and velocity-potential products fetched; the control's v is added."""
    import download_aifs
    import store as ecmwf

    steps = tuple(download_aifs.rmm_steps(hh))
    cyc = ecmwf.Cycle(date, hh)

    def wind(typ, par, levels):
        da = xr.open_dataset(
            ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, par, "pl", levels, steps)),
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )[par]
        return da.sel(isobaricInhPa=200) if "isobaricInhPa" in da.dims else da

    w = {
        (typ, par): wind(typ, par, lev) for typ in ("cf", "pf") for par, lev in (("u", ecmwf.LEVELS_RMM), ("v", (200,)))
    }
    hours = np.array([s for s in steps if s > 0])
    out = np.empty((hours.size, 1 + ecmwf.PF_MEMBERS, lon.size))
    for i, h in enumerate(hours):
        step = np.timedelta64(int(h), "h")
        # one step's members at a time: 0.2 GB a component, not the 3 GB of all the steps
        u, v = w["pf", "u"].sel(step=step).load(), w["pf", "v"].sel(step=step).load()
        if u.sizes["number"] != ecmwf.PF_MEMBERS or v.sizes["number"] != ecmwf.PF_MEMBERS:
            raise SystemExit(f"mjo_chi: {u.sizes['number']} members of u and {v.sizes['number']} of v at +{h} h")
        members = [(w["cf", "u"].sel(step=step), w["cf", "v"].sel(step=step))]
        members += [(u.isel(number=k), v.isel(number=k)) for k in range(ecmwf.PF_MEMBERS)]
        for k, (uk, vk) in enumerate(members):
            chi, la, lo = wc.chi_strip(uk, vk)
            if not (np.allclose(la, lat) and np.allclose(lo[: lon.size], lon)):
                raise SystemExit("mjo_chi: the forecast's χ is not on the history's grid")
            out[i, k] = band_mean(chi[:, : lon.size].astype(float), lat)
    return hours, out


def drift_entry(init: pd.Timestamp, hours: np.ndarray, fc: np.ndarray, lon: np.ndarray, members: int) -> xr.Dataset:
    """One run's archive entry: its band-mean χ200 (lead, lon) on LEADS, NaN at the leads the run does not hold, and
    the number of members behind it."""
    full = np.full((1, LEADS.size, lon.size), np.nan, "float32")
    full[0, np.searchsorted(LEADS, hours)] = fc
    return xr.Dataset(
        {"fc": (("init", "lead", "longitude"), full), "members": (("init",), np.array([members], "int16"))},
        coords={"init": [init], "lead": LEADS, "longitude": lon},
        attrs={"units": "1e6 m2 s-1", "note": "AIFS-ENS 200 hPa velocity potential, 15S-15N mean, by lead hour"},
    )


def read_drift(path: Path, lon: np.ndarray) -> xr.Dataset | None:
    if not path.exists():
        return None
    try:
        ds = xr.open_dataset(path).load()
    except Exception as ex:  # noqa: BLE001
        print(f"  mjo_chi: cannot read {path.name} ({str(ex)[:60]})", flush=True)
        return None
    if ds.sizes["longitude"] != lon.size or not np.allclose(ds.longitude.values, lon) or ds.sizes["lead"] != LEADS.size:
        print(f"  mjo_chi: {path.name} is on another grid; not used", flush=True)
        return None
    return ds


def write_drift(path: Path, ds: xr.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding={"fc": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    tmp.replace(path)


def update_drift(path: Path, cur: xr.Dataset) -> xr.Dataset:
    """The rolling archive with this run's entry in it (the last DRIFT_KEEP runs), written back, then joined to the
    seed: the rolling file's entry where both hold a run."""
    lon = cur.longitude.values
    old = read_drift(path, lon)
    if old is not None:
        old = old.sel(init=old.init != cur.init.values[0])
        cur = xr.concat([old, cur], dim="init").sortby("init") if old.sizes["init"] else cur
    cur = cur.isel(init=slice(-DRIFT_KEEP, None))
    write_drift(path, cur)
    seed = read_drift(DRIFT_SEED, lon)
    if seed is None:
        return cur
    seed = seed.sel(init=~seed.init.isin(cur.init))
    return xr.concat([seed, cur], dim="init").sortby("init") if seed.sizes["init"] else cur


def lowpass(x: np.ndarray, kmax: int) -> np.ndarray:
    f = np.fft.rfft(x, axis=-1)
    f[..., kmax + 1 :] = 0
    return np.fft.irfft(f, x.shape[-1], axis=-1)


def drift_fit(arc: xr.Dataset, a00: xr.DataArray, init: pd.Timestamp, hours: np.ndarray):
    """The model's mean error at `hours` (lead, lon), from the archive's runs of the DRIFT_WINDOW days before `init`
    against the 00 UTC analyses: both cycles pooled, at each longitude a least-squares a·L + b·L² in the lead L, then
    zonal wavenumbers ≤ DRIFT_KMAX. Returns (drift, runs used, runs verified to day 14); drift is None below DRIFT_MIN."""
    t = pd.DatetimeIndex(arc.init.values)
    use = np.flatnonzero((t >= init - pd.Timedelta(days=DRIFT_WINDOW)) & (t < init))
    at = pd.DatetimeIndex(a00.time.values)
    lead = arc.lead.values
    err, ld, deep = [], [], 0
    for i in use:
        valid = t[i] + pd.to_timedelta(lead, "h")
        fc = arc.fc.values[i]
        ok = np.isfinite(fc).all(-1) & valid.isin(at)
        if not ok.any():
            continue
        err.append(fc[ok] - a00.sel(time=valid[ok]).values)
        ld.append(lead[ok] / 24.0 / 15.0)
        deep += bool(lead[ok].max() >= 336)
    if deep < DRIFT_MIN:
        return None, len(err), deep
    L = np.concatenate(ld)
    c, *_ = np.linalg.lstsq(np.stack([L, L**2], 1), np.concatenate(err), rcond=None)
    Lf = hours / 24.0 / 15.0
    return lowpass(np.stack([Lf, Lf**2], 1) @ c, DRIFT_KMAX), len(err), deep


def drift_by_source(arc: xr.Dataset, a00: xr.DataArray, init: pd.Timestamp, hours: np.ndarray) -> None:
    """Log line: the drift from the seed's control forecasts beside that from the archived ensemble means, once both
    have enough runs. They estimate one model's drift, so a low correlation means the correction cannot be trusted."""
    fits = [drift_fit(arc.sel(init=m), a00, init, hours)[0] for m in (arc.members == 1, arc.members > 1)]
    if any(f is None for f in fits):
        return
    i = int(np.argmin(np.abs(hours - 240)))
    r = float(np.corrcoef(fits[0][i], fits[1][i])[0, 1])
    rms = float(np.sqrt(((fits[0][i] - fits[1][i]) ** 2).mean()))
    print(
        f"  mjo_chi: drift at +{hours[i]} h, control runs against ensemble means: r {r:.2f}, rms difference {rms:.2f}"
    )


# ---- the members' agreement ------------------------------------------------------------------ #
def agreed(x: np.ndarray) -> np.ndarray:
    """x (time, member, lon) -> the cells where the members' signs differ from a coin toss: two-sided sign test, the
    false discovery rate held at FDR over the cells."""
    n = x.shape[1]
    neg = (x < 0).sum(1)
    return benjamini_hochberg(np.minimum(1.0, 2.0 * binom.sf(np.maximum(neg, n - neg) - 1, n, 0.5)), FDR)


def rmm_like(mjo: np.ndarray, lon: np.ndarray):
    """mjo (time, member, lon), the MJO's band -> (amplitude (time,) of the member mean in RMM units, phase (time,):
    1 to 8, 0 where none is given - amplitude under 1, or fewer than PHASE_SHARE of the members within 45° of the
    mean's angle, longitude (8,) of the active centre - the reference patterns' minimum - in the middle of each
    phase)."""
    ref = xr.open_dataset(PHASE_REF)
    if ref.sizes["longitude"] != lon.size or not np.allclose(ref.longitude.values, lon):
        raise SystemExit("mjo_chi: the phase reference is on another grid")
    z = mjo @ ref.proj.values.T
    z = z[..., 0] + 1j * z[..., 1]
    mean = z.mean(1)
    near = (np.abs(np.angle(z * np.conj(mean)[:, None])) <= np.pi / 4).mean(1) >= PHASE_SHARE
    amp = np.abs(mean)
    mid = np.radians(-157.5 + 45.0 * np.arange(8))
    centre = lon[
        (np.outer(np.cos(mid), ref.pattern.values[0]) + np.outer(np.sin(mid), ref.pattern.values[1])).argmin(1)
    ]
    return amp, np.where((amp >= 1) & near, phase_of(mean.real, mean.imag), 0), centre


# ---- rendering ------------------------------------------------------------------------------- #
def render(days, n_an, lon, shade, mjo, kel, reg, left, amp, phase, centre, init, n34, lag, nmem, out: Path):
    """shade, mjo, kel: masked (day, lon) fields in 10⁶ m² s⁻¹, the first n_an rows analyses. reg, left: the El Niño
    regression at the last analysis (masked where it is not significant) and the record's remaining mean, by
    longitude. amp, phase: the RMM-like amplitude of each day and its phase (0: none given)."""
    lon_p = np.r_[lon, lon[0] + 360]

    def closed(a):
        return np.ma.concatenate([a, a[..., :1]], -1)

    y = np.arange(len(days))
    fig = plt.figure(figsize=(11.5, 10.5))
    ax = fig.add_axes([0.075, 0.07, 0.775, 0.69])
    norm = BoundaryNorm(SHADE, K.VP_CMAP.N, extend="both")
    cf = ax.contourf(lon_p, y, closed(shade), levels=SHADE, cmap=K.VP_CMAP, norm=norm, extend="both")
    top = MJO_STEP * max(4, int(np.ceil(np.ma.abs(mjo).max() / MJO_STEP)))
    lv = np.arange(MJO_STEP, top + 0.1, MJO_STEP)
    ax.contour(lon_p, y, closed(mjo), levels=-lv[::-1], colors=house.INK, linewidths=1.25, linestyles="solid")
    ax.contour(lon_p, y, closed(mjo), levels=lv, colors=house.INK, linewidths=0.8, linestyles=[(0, (4, 2.5))])
    ax.contour(lon_p, y, closed(kel), levels=KELVIN_LEVELS, colors=KELVIN_COLOUR, linewidths=0.9, linestyles="solid")
    ax.axhline(n_an - 0.5, color="#1c2430", lw=1.8)
    lab = dict(
        fontsize=9,
        fontweight="bold",
        color="#1c2430",
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )
    ax.text(357, n_an - 0.9, "▲ analysis", ha="right", va="bottom", **lab)
    ax.text(
        357,
        n_an - 0.1,
        f"▼ forecast: mean of {nmem} members, drawn where they agree on its sign",
        ha="right",
        va="top",
        **lab,
    )
    ax.set_ylim(len(days) - 0.5, -0.5)  # time runs down
    ticks = [i for i, d in enumerate(days) if d.day in (1, 8, 15, 22)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([days[i].strftime("%b %-d") for i in ticks], fontsize=9)
    ax.set_xticks(range(0, 361, 60))
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"], fontsize=9)
    ax.set_xlim(0, 360)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # the pattern taken out, by longitude
    sx = fig.add_axes([0.075, 0.8, 0.775, 0.07], sharex=ax)
    tot = np.r_[reg.data + left, (reg.data + left)[0]]
    sx.fill_between(lon_p, 0, tot, where=tot <= 0, color="#86c98a", lw=0, interpolate=True)
    sx.fill_between(lon_p, 0, tot, where=tot >= 0, color="#f0a64b", lw=0, interpolate=True)
    sx.plot(lon_p, tot, color=house.INK, lw=1.3, label="standing pattern taken out")
    sx.plot(
        lon_p,
        closed(reg),
        color=house.ACCENT,
        lw=1.2,
        label="its El Niño part: regression on Niño-3.4, where significant",
    )
    sx.axhline(0, color=house.MUTED, lw=0.6)
    sx.tick_params(labelbottom=False, labelsize=8)
    sx.set_ylabel("10⁶ m² s⁻¹", fontsize=8)
    for s in ("top", "right"):
        sx.spines[s].set_visible(False)
    sx.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.32),
        ncol=2,
        fontsize=8,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
    )

    def above(text, at, lift, **kw):
        ax.annotate(
            text,
            (at, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, lift),
            textcoords="offset points",
            ha="center",
            va="bottom",
            annotation_clip=False,
            **kw,
        )

    for lo_, name in (
        (75, "Indian Ocean"),
        (120, "Maritime Cont."),
        (165, "W Pacific"),
        (240, "E Pacific"),
        (325, "Atlantic"),
    ):
        above(name, lo_, 16, fontsize=8, color=house.MUTED)
    # where the MJO's active centre sits in each phase
    for k, lo_ in enumerate(centre):
        above(str(k + 1), lo_, 3, fontsize=8.5, fontweight="bold", color=house.INK2)
    ax.annotate(
        "RMM phase",
        (0, 1),
        xycoords="axes fraction",
        xytext=(-6, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=house.MUTED,
        annotation_clip=False,
    )

    # the phase the MJO's band reads as, day by day, on a bar of its amplitude
    px = fig.add_axes([0.858, 0.07, 0.047, 0.69], sharey=ax)
    xmax = max(3.0, 0.5 * np.ceil(2 * amp.max()) + 0.5)
    px.barh(y, amp, height=1.0, color=np.where(phase > 0, PHASE_TINT, house.RULE), lw=0)
    px.axvline(1, color=house.MUTED, lw=0.7, ls=(0, (3, 2)))
    px.axhline(n_an - 0.5, color="#1c2430", lw=1.8)
    cut = np.flatnonzero(np.diff(phase)) + 1
    for i0, i1 in zip(np.r_[0, cut], np.r_[cut, phase.size], strict=True):
        if phase[i0] and i1 - i0 >= 3:  # a numeral needs three days of room
            px.text(
                0.84 * xmax,
                (i0 + i1 - 1) / 2,
                str(phase[i0]),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=house.INK,
            )
    px.set_xlim(0, xmax)
    px.set_xticks([1])
    px.tick_params(labelleft=False, left=False, labelsize=8)
    for s in ("top", "right"):
        px.spines[s].set_visible(False)
    last = n_an - 1
    head = dict(transform=px.transAxes, ha="left", clip_on=False)
    px.text(0, 1.012, "phase, on a bar\nof the amplitude", va="bottom", fontsize=8, color=house.MUTED, **head)
    px.text(0, 1.142, f"{days[last]:%b %-d}", va="bottom", fontsize=8.5, color=house.MUTED, **head)
    if amp[last] < 1:
        now, why = "No MJO", f"amplitude {amp[last]:.1f}"
    elif phase[last]:
        now, why = f"Phase {phase[last]}", f"amplitude {amp[last]:.1f}"
    else:
        now, why = "No phase", f"amplitude {amp[last]:.1f},\nmembers apart"
    px.text(0, 1.105, now, va="center", fontsize=15, fontweight="bold", color=house.INK, **head)
    px.text(0, 1.085, why, va="top", fontsize=9, color=house.INK2, **head)

    fig.text(
        0.075,
        0.978,
        f"AIFS-ENS · MJO without the El Niño pattern: 200 hPa velocity potential, 15°S–15°N · init {init:%Y-%m-%d %H} UTC",
        fontsize=12.5,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.075,
        0.953,
        textwrap.fill(
            "Shading: the anomaly once the standing pattern in the strip is out: the regression on Niño-3.4 "
            f"({n34:+.1f} °C now, {lag:+.1f} °C three months ago) and the rest of the record's mean; green is upper-level "
            "divergence, the rainy phase. Black: the slow eastward part, the MJO's band (wavenumbers 1–5, 30–96 days), "
            f"every {MJO_STEP:g}; solid active, dashed suppressed. Purple: active Kelvin waves, {-KELVIN_LEVELS[-1]:g} "
            "and stronger. Right: the RMM phase of the black contours, at amplitude 1 or more with the members agreed.",
            170,
        ),
        fontsize=8.6,
        color="#444",
        va="top",
        linespacing=1.3,
    )
    cax = fig.add_axes([0.925, 0.07, 0.018, 0.69])
    cb = fig.colorbar(cf, cax=cax, extend="both", ticks=SHADE)
    cb.set_label("χ200 anomaly without the El Niño pattern (10⁶ m² s⁻¹)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.text(
        0.075,
        0.012,
        textwrap.fill(
            "Analyses: AIFS-ENS control at 0 h, 00 and 12 UTC averaged; anomalies from ERA5 1991–2020. Forecast: every "
            f"member less the model's mean error of the last {DRIFT_WINDOW} days. Agreement: two-sided sign test, false "
            f"discovery rate {100 * FDR:g} %. Phase: the ERA5 1991–2020 patterns that go with RMM1 and RMM2; above the "
            "chart, where each phase's active centre sits. SST: OISST.",
            175,
        ),
        fontsize=7.8,
        color=house.MUTED,
        linespacing=1.3,
    )
    house.clear(fig, 118)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=118, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="run date, YYYYMMDD")
    ap.add_argument("--time", default="00")
    ap.add_argument("--history", default=str(REPO / "assets/sst/anim/walker" / wc.HIST_NAME))
    ap.add_argument("--drift", default=str(REPO / "assets/sst/anim/walker" / DRIFT_NAME))
    ap.add_argument("--enso", default=str(REPO / "assets/sst/data/enso_daily.json"))
    ap.add_argument("--out", default=str(REPO / "assets/mjo/chi_hov.webp"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if selftest():
        raise SystemExit("mjo_chi: filter self-test failed; not rendering")
    init = pd.Timestamp(f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:]}T{int(a.time):02d}:00")
    hist = wc.load_history(Path(a.history))
    if hist is None:
        raise SystemExit("mjo_chi: no χ history; the previous figure stays")
    if np.isclose(float(hist.longitude[-1] - hist.longitude[0]), 360.0):
        hist = hist.isel(longitude=slice(0, -1))
    lat, lon = hist.latitude.values, hist.longitude.values

    # the run's forecast goes into the drift archive whether or not a figure follows
    hours, fc = member_forecast(a.date, a.time, lat, lon)
    arc = update_drift(Path(a.drift), drift_entry(init, hours, fc.mean(1), lon, fc.shape[1]))
    try:
        days, day, a00, doff = observed(hist, init)
    except LookupError as ex:
        raise SystemExit(f"mjo_chi: {ex}; the previous figure stays") from None
    drift, runs, deep = drift_fit(arc, a00, init, hours)
    if drift is None:
        raise SystemExit(
            f"mjo_chi: {deep} earlier runs verified to day 14 ({DRIFT_MIN} needed); the previous figure stays"
        )
    drift_by_source(arc, a00, init, hours)
    sst = nino34(Path(a.enso), xr.open_dataset(REF / "walker_basins_clim.nc"))
    if sst.index[-1] < init.normalize() - pd.Timedelta(days=SST_STALE):
        raise SystemExit(f"mjo_chi: the SST record ends {sst.index[-1]:%Y-%m-%d}; the previous figure stays")

    # what is left of each day once the normal and the El Niño part are out, analyses then members
    clim = xr.open_dataset(K.CLIM)
    coefs, sig = enso_coefs(clim, lat, lon.size)
    fdays = pd.DatetimeIndex(init + pd.to_timedelta(hours, "h"))
    enso_o, n34, lag = enso_part(days, sst, coefs)
    r_obs = day - band_mean(K.chi_normal(days, clim, lat, lon), lat) - enso_o
    r_fc = (
        fc
        - (drift + doff + band_mean(K.chi_normal(fdays, clim, lat, lon), lat) + enso_part(fdays, sst, coefs)[0])[
            :, None
        ]
    )
    left = r_obs.mean(0)
    nmem = fc.shape[1]
    x = np.concatenate([np.repeat((r_obs - left)[:, None], nmem, 1), r_fc - left], 0)
    alld = days.append(fdays)
    n_an = len(days)
    if days[-1] < fdays[0] - pd.Timedelta(days=1):  # the run day's analysis is missing: halfway between its neighbours
        x = np.insert(x, n_an, (x[n_an - 1] + x[n_an]) / 2, 0)
        alld = alld.insert(n_an, days[-1] + pd.Timedelta(days=1))
    mjo = K.wave_filter(x, mjo_mask, PAD, RAMP)
    kel = K.wave_filter(x, K.kelvin_mask, PAD, RAMP)

    t0 = max(0, n_an - TEST_BACK)
    ok_s = np.ones(x.shape[::2], bool)
    ok_s[n_an:] = agreed(x[n_an:])
    ok_m, ok_k = ok_s.copy(), ok_s.copy()
    ok_m[t0:], ok_k[t0:] = agreed(mjo[t0:]), agreed(kel[t0:])
    amp, phase, centre = rmm_like(mjo, lon)
    first = max(n_an - SHOW, 2, int(0.1 * len(alld)))  # past the filter's soft start
    print(
        f"  mjo_chi: {n_an} analysis days {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d} + {len(alld) - n_an} forecast days, "
        f"{nmem} members; drift from {runs} runs ({deep} to day 14), rms {np.sqrt((drift[-1] ** 2).mean()):.2f} at "
        f"+{hours[-1]} h; Niño-3.4 {n34[-1]:+.2f} °C, {lag[-1]:+.2f} °C three months ago; record mean left rms "
        f"{np.sqrt((left**2).mean()):.2f}; forecast cells drawn: shading {100 * ok_s[n_an:].mean():.0f} %, MJO band "
        f"{100 * ok_m[n_an:].mean():.0f} %, Kelvin {100 * ok_k[n_an:].mean():.0f} %; MJO band rms "
        f"{np.sqrt((mjo.mean(1)[first:] ** 2).mean()):.2f}, Kelvin {np.sqrt((kel.mean(1)[first:] ** 2).mean()):.2f}; "
        f"on {alld[n_an - 1]:%Y-%m-%d} amplitude {amp[n_an - 1]:.2f}, phase {phase[n_an - 1] or 'none'}; phase given on "
        f"{(phase[n_an:] > 0).sum()} of {len(alld) - n_an} forecast days",
        flush=True,
    )

    def shown(f, ok):
        return np.ma.masked_where(~ok[first:], f.mean(1)[first:])

    render(
        alld[first:],
        n_an - first,
        lon,
        shown(x, ok_s),
        shown(mjo, ok_m),
        shown(kel, ok_k),
        np.ma.masked_where(~(monthly(sig.astype(float), days[-1:])[0] >= 0.5), enso_o[-1]),
        left,
        amp[first:],
        phase[first:],
        centre,
        init,
        n34[-1],
        lag[-1],
        nmem,
        Path(a.out),
    )
    print(f"  mjo_chi: wrote {Path(a.out).name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
