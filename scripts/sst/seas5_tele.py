#!/usr/bin/env python3
"""SEAS5 teleconnection and stratospheric indices, member by member, month by month.

Teleconnections use the SAME projectors as the GEPS subseasonal page
(~/data_archive/geps_subx/telecon/patterns.nc, built by build_telecon_patterns.py):
CPC-style loading patterns for NAO, EA, WP, EP/NP, PNA, EA/WR, SCA, TNH and POL as
per-calendar-month regression maps of 500 hPa height on CPC's own indices; AO as
the first EOF of sea-level pressure poleward of 20°N; EPO and WPO as PSL's box
definitions. Applied to each member's monthly 500 hPa height (or msl) anomaly
against SEAS5's own 1993–2016 hindcast, interpolated to the 2.5° NCEP grid the
patterns live on. The result is on CPC's monthly scale (annual standard
deviations), so it reads like the published index.

Also, from the same fields and the stratospheric wind pull:
  soi      Southern Oscillation index: msl Tahiti minus Darwin, standardised by
           the hindcast spread at each lead.
  u60n10 / u60s10   zonal-mean 10 hPa zonal wind at 60°N / 60°S (m/s), the
           standard vortex strength; P(easterly) is the SSW-side probability.
  qbo10/30/50       equatorial (5°S–5°N) zonal-mean zonal wind, SEAS5's own QBO.
  nam_{10,50,100,500,1000}  polar-cap (65–90°N) height anomaly standardised by the
           hindcast and sign-flipped, an annular-mode index by level.
  wave1_100 / wave2_100   amplitude (m) of zonal wavenumbers 1 and 2 of 100 hPa
           height along 60°N, the planetary-wave forcing.

Skill: every index is also computed from the 24 hindcast years (ensemble mean per
year) and, where ERA5 monthly fields are cached (seas5_era5.py), from ERA5 with
the same projector against its 1993–2016 mean; the correlation across the 24
years at each lead is stored as `skill`, so the page can say how much a tilt is
worth for that month and lead.

Output: assets/sst/data/seas5_tele.json (this issue with members; previous issue
summary). Imported by seas5_outlook's runner via `python seas5_tele.py --issue`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seas5_outlook import ASSETS, fc_path, hc_path, previous_issues  # noqa: E402
from seas5_build import G0, _open, _summ, detrend_pair, load_field, valid_months  # noqa: E402

TC = Path.home() / "data_archive" / "geps_subx" / "telecon"
OUT_JSON = ASSETS / "data" / "seas5_tele.json"
ERA5 = Path(__file__).resolve().parent / "data" / "seas5" / "era5"

TELE_ORDER = ["nao", "pna", "ao", "ea", "wp", "epnp", "eawr", "sca", "tnh", "pol", "epo", "wpo"]
DEFINED_MONTHS = {"tnh": {12, 1, 2}, "epnp": set(range(1, 13)) - {8, 9}}
TELE_LABEL = {"nao": "NAO", "pna": "PNA", "ao": "Arctic Oscillation", "ea": "East Atlantic", "wp": "West Pacific",
              "epnp": "East Pacific / North Pacific", "eawr": "East Atlantic / West Russia", "sca": "Scandinavia",
              "tnh": "Tropical / Northern Hemisphere", "pol": "Polar / Eurasia", "epo": "EPO", "wpo": "WPO"}


# ── the projectors ───────────────────────────────────────────────────────────
class Patterns:
    def __init__(self):
        self.ds = xr.open_dataset(TC / "patterns.nc")
        self.meta = json.loads((TC / "validation.json").read_text())
        lat = self.ds.lat.values
        self.w = np.sqrt(np.clip(np.cos(np.deg2rad(lat.astype("float64"))), 0, None))[:, None] * np.ones((1, self.ds.lon.size))
        self.ids = [i for i in TELE_ORDER if i in self.meta and i in self.ds.index.values]

    def to_ncep(self, a: xr.DataArray) -> np.ndarray:
        """(..., lat, lon) on any regular grid → NCEP 2.5° (90..−90, 0..357.5), NaN where absent."""
        lon = a.longitude.values
        a = a.assign_coords(longitude=np.where(lon < 0, lon + 360, lon)).sortby("longitude")
        wrap = a.isel(longitude=0).assign_coords(longitude=360.0)
        a = xr.concat([a, wrap], dim="longitude").sortby("latitude")
        out = a.interp(latitude=self.ds.lat.values, longitude=self.ds.lon.values, method="linear")
        return out.transpose(..., "latitude", "longitude").values

    def index(self, sid: str, anom: np.ndarray, months) -> np.ndarray:
        """anom: (n, lat, lon) on the NCEP grid (z500 m or slp hPa); months: (n,) calendar month.
        → CPC-scale monthly index (annual standard deviations); NaN in months CPC does not define."""
        m = self.meta[sid]
        mo = np.asarray(months) - 1
        P = self.ds.projector.sel(index=sid).values[mo]
        a = np.asarray(anom, dtype="float64")
        if m.get("standardize"):
            sd = self.ds.std_month_z500.values[mo]
            a = a / np.where(sd > 0, sd, np.nan)
        if m.get("demean"):
            lat = self.ds.lat.values
            cosw = (np.cos(np.deg2rad(lat)) * (lat >= 20))[:, None] * np.ones((1, self.ds.lon.size))
            a = a - (np.nan_to_num(a) * cosw).sum(axis=(-2, -1), keepdims=True) / cosw.sum()
        raw = np.einsum("nij,nij->n", np.nan_to_num(a * self.w), P)
        # CPC does not define every pattern in every month (TNH Dec–Feb only; EP/NP not Aug–Sep)
        defined = DEFINED_MONTHS.get(sid, set(range(1, 13)))
        raw[~np.isin(np.asarray(months), sorted(defined))] = np.nan
        return raw


# ── field helpers ────────────────────────────────────────────────────────────
def _grid(path: Path, var: str, **filt):
    vals, lat, lon = load_field(path, var, **filt)
    return vals, lat, lon


def _pair(kind: str, ym: str, var: str, **filt):
    """(fc[sample, lead, lat, lon], hc[...], lat, lon) or None."""
    f, h = fc_path(kind, ym), hc_path(kind, ym[4:])
    if not (f.exists() and h.exists()):
        return None
    fc, lat, lon = _grid(f, var, **filt)
    hc, _, _ = _grid(h, var, **filt)
    return fc, hc, lat, lon


def _zonal_at(vals, lat, lat0):
    i = int(np.argmin(np.abs(lat - lat0)))
    return np.nanmean(vals[:, :, i, :], axis=-1)                    # [sample, lead]


def _band_zonal(vals, lat, la0, la1):
    m = (lat >= la0) & (lat <= la1)
    w = np.cos(np.deg2rad(lat[m]))
    return np.nansum(np.nanmean(vals[:, :, m, :], axis=-1) * w, axis=-1) / w.sum()


def _cap(vals, lat, la0):
    m = lat >= la0 if la0 > 0 else lat <= la0
    w = np.cos(np.deg2rad(lat[m]))
    return np.nansum(np.nanmean(vals[:, :, m, :], axis=-1) * w, axis=-1) / w.sum()


def _wave_amp(vals, lat, lat0, k):
    i = int(np.argmin(np.abs(lat - lat0)))
    row = vals[:, :, i, :]
    spec = np.fft.rfft(row, axis=-1)
    return 2.0 * np.abs(spec[..., k]) / row.shape[-1]


def _point(vals, lat, lon, la, lo):
    return vals[:, :, int(np.argmin(np.abs(lat - la))), int(np.argmin(np.abs(lon - lo)))]


# ── per-issue indices ────────────────────────────────────────────────────────
def compute(ym: str, pats: Patterns, with_members: bool, scale: dict | None = None) -> dict:
    """scale: {key: {calendar_month: σ}} from ERA5 (1991–2020) — the unit for the indices that have no
    built-in standardisation (AO, EPO, WPO, SOI), so that the model and the observed tail share one scale.
    Falls back to the hindcast spread at each lead when ERA5 is unavailable."""
    out = {}
    vm = valid_months(ym)
    months = [int(v[5:]) for v in vm]
    nl = len(vm)

    def to_sigma(key, fi, hi):
        if scale and key in scale and all(m in scale[key] for m in months):
            sd = np.array([scale[key][m] for m in months])[None]
            return fi / sd, hi / sd, "ERA5 1991–2020 standard deviation for the calendar month"
        sd = np.nanstd(hi, axis=0); sd = np.where(sd > 0, sd, np.nan)
        return fi / sd, hi / sd, "hindcast spread at each lead"

    def entry(fc_idx, hc_idx, label, units, note, group, sign_note=None):
        """fc_idx [sample, lead], hc_idx [sample, lead] (hindcast, anomaly or absolute as the index wants)."""
        e = dict(label=label, units=units, note=note, group=group, valid=vm, **_summ(fc_idx),
                 clim_sd=np.nanstd(hc_idx, axis=0).round(3).tolist(), clim_mean=np.nanmean(hc_idx, axis=0).round(3).tolist(),
                 p_pos=np.nanmean(fc_idx > 0, axis=0).round(3).tolist(), p_neg=np.nanmean(fc_idx < 0, axis=0).round(3).tolist(),
                 hc_years=None)
        if with_members:
            e["members"] = np.where(np.isfinite(fc_idx), np.round(fc_idx, 3), None).tolist()
        return e

    # 500 hPa / msl patterns
    z = _pair("nh_z", ym, "z", level=500)
    p = _pair("nh_msl", ym, "msl")
    if z and p:
        fz, hz, lat, lon = z; fp, hp, _, _ = p
        hz_m = np.nanmean(hz, 0); hp_m = np.nanmean(hp, 0)
        az = xr.DataArray((fz - hz_m[None]) / G0, dims=("s", "l", "latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
        ahz = xr.DataArray((hz - hz_m[None]) / G0, dims=("s", "l", "latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
        ap = xr.DataArray((fp - hp_m[None]) / 100.0, dims=("s", "l", "latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
        ahp = xr.DataArray((hp - hp_m[None]) / 100.0, dims=("s", "l", "latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
        nz, nhz, npm, nhp = (pats.to_ncep(a) for a in (az, ahz, ap, ahp))
        for sid in pats.ids:
            m = pats.meta[sid]
            src, hsrc = (npm, nhp) if m.get("field") == "slp" else (nz, nhz)
            def run(arr):
                n, L = arr.shape[:2]
                flat = arr.reshape(n * L, *arr.shape[2:])
                mo = np.tile(months, n)
                return pats.index(sid, flat, mo).reshape(n, L)
            fi, hi = run(src), run(hsrc)
            unit_note = "CPC-scale (annual standard deviations)"
            if not m.get("standardize"):                   # AO (EOF, raw) and the EPO/WPO boxes (metres) need a unit
                fi, hi, unit_note = to_sigma(sid, fi, hi)
            out[sid] = entry(fi, hi, TELE_LABEL.get(sid, sid), "σ",
                             (m.get("name", sid) + f". CPC-style pattern applied to the member's 500 hPa height anomaly; {unit_note}."
                              if m.get("field") != "slp" else f"First EOF of sea-level pressure poleward of 20°N; positive = low pressure over the pole; {unit_note}."),
                             "tele")
            out[sid]["_hc"] = hi                                          # kept for skill, dropped before writing
        # SOI from msl points
        ft = _point(fp, lat, lon, -17.5, -149.5) - _point(fp, lat, lon, -12.5, 130.5)
        ht = _point(hp, lat, lon, -17.5, -149.5) - _point(hp, lat, lon, -12.5, 130.5)
        fa, ha, unit_note = to_sigma("soi", (ft - ht.mean(0)) / 100.0, (ht - ht.mean(0)) / 100.0)
        out["soi"] = entry(fa, ha, "Southern Oscillation index", "σ",
                           f"Tahiti minus Darwin sea-level pressure anomaly; unit: {unit_note}; negative with El Niño.", "tele")
        out["soi"]["_hc"] = ha
    # stratospheric wind
    u = _pair("strat_u", ym, "u", level=10)
    if u:
        fu, hu, lat, lon = u
        for key, la, lab in (("u60n10", 60, "60°N 10 hPa zonal wind"), ("u60s10", -60, "60°S 10 hPa zonal wind")):
            fi, hi = _zonal_at(fu, lat, la), _zonal_at(hu, lat, la)
            e = entry(fi, hi, lab, "m/s", "Zonal-mean zonal wind, the standard vortex-strength metric; below zero is an easterly (reversed) vortex.", "strat")
            e["p_easterly"] = np.nanmean(fi < 0, axis=0).round(3).tolist()
            e["absolute"] = True
            e["_hc"] = hi
            out[key] = e
        for lev in (10, 30, 50):
            fu, hu, lat, lon = _pair("strat_u", ym, "u", level=lev)
            fi, hi = _band_zonal(fu, lat, -5, 5), _band_zonal(hu, lat, -5, 5)
            e = entry(fi, hi, f"QBO: equatorial {lev} hPa wind", "m/s", "5°S–5°N zonal-mean zonal wind, SEAS5's own QBO; westerly positive.", "strat")
            e["absolute"] = True; e["_hc"] = hi
            out[f"qbo{lev}"] = e
    # NAM by level and planetary waves
    for lev, kind in ((10, "polar_n"), (50, "polar_n"), (100, "polar_n"), (500, "nh_z"), (1000, "nh_z")):
        r = _pair(kind, ym, "z", level=lev)
        if not r:
            continue
        fz, hz, lat, lon = r
        fc_cap, hc_cap = _cap(fz / G0, lat, 65), _cap(hz / G0, lat, 65)
        fa, hr = detrend_pair(fc_cap, hc_cap, ym)                   # heights carry the warming trend: anomaly vs the hindcast trend line
        sd = np.nanstd(hr, axis=0)
        fi = -fa / sd; hi = -hr / sd
        e = entry(fi, hi, f"NAM at {lev} hPa", "σ", "Polar-cap (65–90°N) height anomaly against the hindcast's linear trend (detrended), standardised and sign-flipped: positive = strong vortex / low polar heights.", "strat")
        e["_hc"] = hi
        out[f"nam{lev}"] = e
    r = _pair("polar_n", ym, "z", level=100)
    if r:
        fz, hz, lat, lon = r
        for k in (1, 2):
            fi, hi = _wave_amp(fz / G0, lat, 60, k), _wave_amp(hz / G0, lat, 60, k)
            e = entry(fi, hi, f"Wave-{k} amplitude, 100 hPa, 60°N", "m", "Amplitude of the zonal wavenumber along 60°N: the planetary-wave forcing that precedes a warming.", "strat")
            e["absolute"] = True; e["_hc"] = hi
            out[f"wave{k}_100"] = e
    return out


# ── skill against ERA5 ───────────────────────────────────────────────────────
def era5_indices(pats: Patterns):
    """(series, scale): series = {key: {(year, month): value}} observed indices in the SAME units as the
    model's; scale = {key: {month: σ}} the ERA5 1991–2020 calendar-month standard deviation used as
    the unit for AO/EPO/WPO/SOI. From the LOCAL ERA5 store: the 0–90°N 500 hPa height and sea-level
    pressure sets (1959–2026, monthly means of daily fields) for the CPC patterns, the AO and the NAM
    rungs; the global pressure set (1991–2020) for the SOI. Anomalies are against the 1993–2016 mean of
    each calendar month — the model's hindcast base. The stratospheric winds (10/30/50 hPa) are not in
    the store, so u60N/u60S/QBO carry no observed reference until a CDS pressure-level file exists."""
    import era5_local
    if not era5_local.available():
        return None, None
    out, scale = {}, {}

    def ym_keys(da):
        t = da.time.values
        return np.array([int(str(x)[:4]) for x in t]), np.array([int(str(x)[5:7]) for x in t])

    def anom(vals, yrs, mos):
        base = (yrs >= 1993) & (yrs <= 2016)
        clim = np.stack([np.nanmean(vals[(mos == m) & base], axis=0) for m in range(1, 13)])
        return vals - clim[mos - 1]

    def month_sd(vals, yrs, mos):
        sel = (yrs >= 1991) & (yrs <= 2020)
        return {m: float(np.nanstd(vals[sel & (mos == m)], ddof=1)) for m in range(1, 13)}

    def series(vals, yrs, mos):
        return {(int(y), int(m)): float(v) for y, m, v in zip(yrs, mos, vals) if np.isfinite(v)}

    z = era5_local.monthly("z", 500); pm = era5_local.monthly("slp_nh")
    if z is not None and pm is not None:
        z, pm = era5_local.to_lon180(z), era5_local.to_lon180(pm)
        yrs, mos = ym_keys(z)
        za = anom(z.values.astype(np.float64), yrs, mos)
        ypm, mpm = ym_keys(pm)
        pa = anom(pm.values.astype(np.float64), ypm, mpm)
        nz = pats.to_ncep(xr.DataArray(za, dims=("t", "latitude", "longitude"), coords={"latitude": z.latitude.values, "longitude": z.longitude.values}))
        npm = pats.to_ncep(xr.DataArray(pa, dims=("t", "latitude", "longitude"), coords={"latitude": pm.latitude.values, "longitude": pm.longitude.values}))
        for sid in pats.ids:
            m = pats.meta[sid]
            slp = m.get("field") == "slp"
            src, yy, mm = (npm, ypm, mpm) if slp else (nz, yrs, mos)
            v = pats.index(sid, src, mm)
            if not m.get("standardize"):
                scale[sid] = month_sd(v, yy, mm)
                v = v / np.array([scale[sid][k] for k in mm])
            out[sid] = series(v, yy, mm)
    g = era5_local.monthly("slp")                                        # global set: Tahiti and Darwin
    if g is not None:
        g = era5_local.to_lon180(g)
        yg, mg = ym_keys(g)
        lat, lon = g.latitude.values, g.longitude.values
        tah = g.values[:, int(np.argmin(np.abs(lat + 17.5))), int(np.argmin(np.abs(lon + 149.5)))]
        dar = g.values[:, int(np.argmin(np.abs(lat + 12.5))), int(np.argmin(np.abs(lon - 130.5)))]
        soi = anom((tah - dar).astype(np.float64), yg, mg)                # hPa
        scale["soi"] = month_sd(soi, yg, mg)
        out["soi"] = series(soi / np.array([scale["soi"][k] for k in mg]), yg, mg)
    for lev in (50, 100, 500, 1000):
        zl = era5_local.monthly("z", lev)
        if zl is None:
            continue
        yy, mm = ym_keys(zl)
        m = zl.latitude.values >= 65; w = np.cos(np.deg2rad(zl.latitude.values[m]))
        cap = np.nansum(np.nanmean(zl.values[:, m, :], axis=-1) * w, axis=-1) / w.sum()
        # same construction as the model's NAM: anomaly against a linear trend (fit 1991–2020 per
        # calendar month, extrapolated beyond), in units of the residual spread, sign-flipped
        nam = np.full(cap.shape, np.nan)
        for k in range(1, 13):
            sel = mm == k; fit = sel & (yy >= 1991) & (yy <= 2020)
            x = yy[fit] - yy[fit].mean(); yv = cap[fit]
            b = (x * (yv - yv.mean())).sum() / (x ** 2).sum(); a0 = yv.mean()
            resid_sd = np.std(yv - (a0 + b * x), ddof=2)
            nam[sel] = -(cap[sel] - (a0 + b * (yy[sel] - yy[fit].mean()))) / resid_sd
        out[f"nam{lev}"] = series(nam, yy, mm)
        if lev == 100:
            i = int(np.argmin(np.abs(zl.latitude.values - 60)))
            spec = np.fft.rfft(zl.values[:, i, :], axis=-1)
            for k in (1, 2):
                out[f"wave{k}_100"] = series(2.0 * np.abs(spec[:, k]) / zl.shape[-1], yy, mm)
    return (out or None), (scale or None)


# ── published indices (CPC) as an independent check on the observed tail ─────
CPC_FILES = {
    "tele_index.nh": "https://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/tele_index.nh",
    "monthly.ao.index.txt": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii",
    "soi.txt": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
}
CPC_COLS = {"nao": "NAO", "ea": "EA", "wp": "WP", "epnp": "EP/NP", "pna": "PNA", "eawr": "EA/WR", "sca": "SCA", "tnh": "TNH", "pol": "POL"}


def cpc_published() -> dict:
    """{key: {(year, month): value}} CPC's own monthly indices (their standardisation, so the overlay
    is a check on sign and rough size, not a same-unit comparison). Files refreshed when > 12 h old."""
    import re, urllib.request
    d = TC / "cpc"; d.mkdir(parents=True, exist_ok=True)
    for name, url in CPC_FILES.items():
        f = d / name
        if f.exists() and time.time() - f.stat().st_mtime < 12 * 3600:
            continue
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if data:
                f.write_bytes(data)
        except Exception as e:                                             # noqa: BLE001
            print(f"  CPC {name}: not refreshed ({str(e)[:60]})", flush=True)
    out = {}
    f = d / "tele_index.nh"
    if f.exists():
        cols = None; rows = {}
        for ln in f.read_text(errors="replace").splitlines():
            if cols is None and "NAO" in ln and "PNA" in ln:
                cols = [c for c in re.split(r"\s+", ln.strip()) if c][2:]; continue
            m = re.match(r"^\s*(\d{4})\s+(\d{1,2})\s+(.*)$", ln)
            if not m or cols is None:
                continue
            vals = [float(x) for x in re.findall(r"-?\d+\.\d+", m.group(3))]
            rows[(int(m.group(1)), int(m.group(2)))] = dict(zip(cols, vals))
        for key, col in CPC_COLS.items():
            out[key] = {k: v[col] for k, v in rows.items() if col in v and v[col] > -99}
    f = d / "monthly.ao.index.txt"
    if f.exists():
        ao = {}
        for ln in f.read_text(errors="replace").splitlines():
            m = re.match(r"^\s*(\d{4})\s+(\d{1,2})\s+(-?\d+\.\d+)", ln)
            if m:
                ao[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
        out["ao"] = ao
    f = d / "soi.txt"
    if f.exists():
        soi = {}; std_part = False
        for ln in f.read_text(errors="replace").splitlines():
            if "STANDARDIZED" in ln:
                std_part = True; continue
            m = re.match(r"^\s*(\d{4})((?:\s*-?\d+\.\d)+)\s*$", ln)
            if std_part and m:
                vals = [float(x) for x in re.findall(r"-?\d+\.\d", m.group(2))]
                for k, v in enumerate(vals[:12]):
                    if v > -999:
                        soi[(int(m.group(1)), k + 1)] = v
        out["soi"] = soi
    return out


THR = 0.5          # phase threshold in index units (σ)
SKILL_MIN = 0.25   # below this hindcast correlation the calibrated probabilities are climatology and the cell is hatched
TAIL_MONTHS = 18


def _phi(x):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def add_skill(idx: dict, ym: str, era: dict | None, cpc: dict | None = None) -> None:
    """Per index and lead: hindcast skill and a regression calibration against ERA5 at the valid month.
    The 24 hindcast years' ensemble mean x_y is regressed on the observed index o_y (o = a·x + b, residual
    s); the calibrated forecast is N(a·x̄ + b, s²), so P(≥ +THR) and P(≤ −THR) collapse to climatology when
    the slope is ~0. In-sample over 24 years (no cross-validation), which flatters r slightly at n = 24.
    Also attaches the last TAIL_MONTHS observed values (ERA5) and CPC's published index where one exists."""
    vm = valid_months(ym)
    y0, m0 = int(ym[:4]), int(ym[4:])
    tail = [((y0 * 12 + m0 - 1 - k) // 12, (y0 * 12 + m0 - 1 - k) % 12 + 1) for k in range(TAIL_MONTHS, 0, -1)]
    for key, e in idx.items():
        hc = e.pop("_hc", None)
        fc = np.array([[np.nan if v is None else v for v in row] for row in e["members"]], dtype=float) if e.get("members") is not None else None
        with np.errstate(invalid="ignore"):
            e["p_neg05"] = [None if not np.isfinite(fc[:, L]).any() else round(float(np.nanmean(fc[:, L] <= -THR)), 3) for L in range(fc.shape[1])] if fc is not None and not e.get("absolute") else None
            e["p_pos05"] = [None if not np.isfinite(fc[:, L]).any() else round(float(np.nanmean(fc[:, L] >= THR)), 3) for L in range(fc.shape[1])] if fc is not None and not e.get("absolute") else None
        obs = era.get(key) if era else None
        e["obs_tail"] = [[f"{y}-{m:02d}", round(obs[(y, m)], 3)] for (y, m) in tail if obs and (y, m) in obs] if obs else None
        pub = cpc.get(key) if cpc else None
        e["cpc_tail"] = [[f"{y}-{m:02d}", round(pub[(y, m)], 2)] for (y, m) in tail if (y, m) in pub] if pub else None
        if hc is None or obs is None:
            e["skill"] = None; e["cal"] = None; continue
        n = hc.shape[0] // 24                                          # members per year; samples stack member-major
        r, cal = [], []                                                # (xarray stack over (number, time): time varies fastest)
        for L, v in enumerate(vm):
            mo = int(v[5:])
            yrs = list(range(1993, 2017))
            hmean = np.nanmean(hc[:, L].reshape(n, 24), axis=0)        # → per-year ensemble mean, years in order
            o = np.array([obs.get((y + (1 if (m0 - 1 + L) >= 12 else 0), mo), np.nan) for y in yrs])
            ok = np.isfinite(hmean) & np.isfinite(o)
            if ok.sum() < 12 or e["mean"][L] is None:
                r.append(None); cal.append(None); continue
            x, oo = hmean[ok], o[ok]
            rr = float(np.corrcoef(x, oo)[0, 1]); r.append(round(rr, 2))
            a, b = np.polyfit(x, oo, 1)
            s_ = float(np.std(oo - (a * x + b), ddof=2))
            mu = float(a * e["mean"][L] + b)
            c = dict(n=int(ok.sum()), a=round(float(a), 3), b=round(float(b), 3), s=round(s_, 3), mu=round(mu, 3), obs_sd=round(float(oo.std(ddof=1)), 3))
            if not e.get("absolute"):
                c["p_neg"] = round(_phi((-THR - mu) / s_), 3); c["p_pos"] = round(1.0 - _phi((THR - mu) / s_), 3)
            c["no_skill"] = bool(rr < SKILL_MIN)
            cal.append(c)
        e["skill"] = r; e["cal"] = cal


# ── figures ──────────────────────────────────────────────────────────────────
INK, MUTED = "#222", "#6b6b6b"


def _month_num(s: str) -> int:
    return int(s[:4]) * 12 + int(s[5:7]) - 1


def _fmt_month(n: int) -> str:
    import calendar
    return f"{calendar.month_abbr[n % 12 + 1]}\n{n // 12}" if n % 12 == 0 else calendar.month_abbr[n % 12 + 1]


def plot_series(doc: dict, group: str, out: Path) -> None:
    """One panel per index: ERA5 observed tail (and CPC's published index where one exists), the
    forecast bands and mean of this issue, the previous issue's mean, the hindcast skill per month."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    keys = doc["groups"][group]
    ncol = 4; nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 2.9 * nrow + 1.3), squeeze=False)
    issue, prev = doc["issue"], doc["previous"]
    for k, key in enumerate(keys):
        ax = axes[k // ncol, k % ncol]
        e = doc["indices"][key]; pe = (doc.get("previous_indices") or {}).get(key)
        xs = [_month_num(v) for v in e["valid"]]
        x0 = xs[0] - TAIL_MONTHS
        absolute = bool(e.get("absolute"))
        cm = np.array(e["clim_mean"], dtype=float) if absolute else np.zeros(len(xs))
        band = np.array(e["clim_sd"], dtype=float) if absolute else np.full(len(xs), THR)
        if absolute:
            ax.fill_between(xs, cm - band, cm + band, color="#000", alpha=0.05, zorder=0)
            ax.plot(xs, cm, color="#888", lw=0.9, ls="--", zorder=1)
        else:
            ax.axhspan(-THR, THR, color="#000", alpha=0.05, zorder=0)
            ax.axhline(0, color="#555", lw=0.8, zorder=1)
        f = lambda a: np.array([np.nan if v is None else v for v in a], dtype=float)
        ax.fill_between(xs, f(e["p10"]), f(e["p90"]), color="#3f5f8f", alpha=0.16, zorder=2, lw=0)
        ax.fill_between(xs, f(e["p25"]), f(e["p75"]), color="#3f5f8f", alpha=0.30, zorder=2, lw=0)
        if pe:
            ax.plot([_month_num(v) for v in pe["valid"]], f(pe["mean"]), color="#d0801a", lw=1.6, ls="--", zorder=4)
        ax.plot(xs, f(e["mean"]), color="#1b365d", lw=2.2, marker="o", ms=3.5, zorder=5)
        if e.get("obs_tail"):
            ox = [_month_num(t) for t, _ in e["obs_tail"]]; oy = [v for _, v in e["obs_tail"]]
            ax.plot(ox, oy, color="#333", lw=1.4, marker="o", ms=3, zorder=6)
        if e.get("cpc_tail"):
            ax.plot([_month_num(t) for t, _ in e["cpc_tail"]], [v for _, v in e["cpc_tail"]], ls="none", marker="x", ms=5, mew=1.2, color="#8b1a1a", zorder=7)
        if key.startswith("u60"):
            ax.axhline(0, color="#c2185b", lw=1.0, ls="--", zorder=3)
        ax.axvline(xs[0] - 0.5, color="#999", lw=0.8, ls=":", zorder=3)
        if e.get("skill"):
            for xv, rr in zip(xs, e["skill"]):
                if rr is not None:
                    ax.text(xv, 0.015, f"r {rr:+.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.4, rotation=90,
                            color=(INK if rr >= SKILL_MIN else "#aaa"), fontweight=("bold" if rr >= 0.5 else "normal"))
        ax.set_xlim(x0 - 0.5, xs[-1] + 0.5)
        ticks = list(range(x0, xs[-1] + 1, 2)) if (xs[-1] - x0) > 14 else list(range(x0, xs[-1] + 1))
        ax.set_xticks(ticks); ax.set_xticklabels([_fmt_month(t) for t in ticks], fontsize=7.2)
        ax.tick_params(axis="y", labelsize=7.5); ax.grid(axis="y", lw=0.4, alpha=0.4)
        ax.set_title(f"{e['label']}", fontsize=10, loc="left", fontweight="bold")
        ax.set_ylabel(e["units"], fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if not e.get("obs_tail"):
            ax.text(0.02, 0.96, ("ERA5 series not in the local store; CPC's published index shown" if e.get("cpc_tail") else "no ERA5 reference in the local store"),
                    transform=ax.transAxes, fontsize=7, color=MUTED, va="top", style="italic")
    for k in range(len(keys), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    import calendar
    il = lambda ym: f"{calendar.month_abbr[int(ym[4:])]} {ym[:4]}"
    handles = [Line2D([], [], color="#333", lw=1.4, marker="o", ms=3), Line2D([], [], ls="none", marker="x", color="#8b1a1a", mew=1.2),
               Line2D([], [], color="#1b365d", lw=2.2, marker="o", ms=3.5), Line2D([], [], color="#d0801a", lw=1.6, ls="--"),
               plt.Rectangle((0, 0), 1, 1, color="#3f5f8f", alpha=0.16), plt.Rectangle((0, 0), 1, 1, color="#3f5f8f", alpha=0.30), plt.Rectangle((0, 0), 1, 1, color="#000", alpha=0.05)]
    labels = ["observed, ERA5 (same projection)", "published index (CPC), its own scale", f"{il(issue)} issue: ensemble mean", f"{il(prev)} issue mean",
              "p10–p90", "p25–p75", ("hindcast mean ±1σ" if group == "strat" else f"neutral, ±{THR}σ")]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(("Teleconnection indices" if group == "tele" else "Stratosphere and annular-mode indices") + f" — SEAS5, 51 members, {il(issue)} issue", x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.01, 1 - 0.36 / fig.get_figheight(), "Monthly means. Anomalies against the 1993–2016 hindcast (model) and the 1993–2016 ERA5 mean (observed); r = correlation of the 24 hindcast years' ensemble mean with ERA5 for that "
             "calendar month and lead (grey below 0.25). Dotted line: last observed month in the store.", fontsize=8, color=MUTED, va="top")
    fig.subplots_adjust(left=0.045, right=0.99, top=1 - 0.85 / fig.get_figheight(), bottom=0.7 / fig.get_figheight(), hspace=0.5, wspace=0.22)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    print(f"  wrote {out.name}", flush=True)


def plot_probs(doc: dict, out: Path) -> None:
    """Phase-probability board, index × month: raw member fractions (upper bar), hindcast-calibrated
    probabilities (thin lower bar), mean, change vs the previous issue, skill; hatched where r < SKILL_MIN."""
    import calendar
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    keys = [k for k in doc["groups"]["tele"] if doc["indices"][k].get("p_neg05")]
    n = len(keys); vm = doc["indices"][keys[0]]["valid"]; W = len(vm)
    fig = plt.figure(figsize=(13.6, 0.80 * n + 2.6))
    ax = fig.add_axes([0.205, 0.075, 0.765, 0.79])
    bw = 0.78

    def bar(y, h, pn, pp, x0, txt=True):
        pz = max(0.0, 1.0 - pn - pp)
        ax.barh(y, pn * bw, left=x0, height=h, color="#2c5fa8", lw=0)
        ax.barh(y, pz * bw, left=x0 + pn * bw, height=h, color="#d9d6d0", lw=0)
        ax.barh(y, pp * bw, left=x0 + (pn + pz) * bw, height=h, color="#b4453c", lw=0)
        if txt:
            for frac, off in ((pn, pn / 2), (pp, pn + pz + pp / 2)):
                if frac >= 0.12:
                    ax.text(x0 + off * bw, y, f"{frac*100:.0f}%", ha="center", va="center", fontsize=7.4, color="white", fontweight="bold")

    for r_, key in enumerate(keys):
        e = doc["indices"][key]; pe = (doc.get("previous_indices") or {}).get(key)
        y = n - 1 - r_
        for L, v in enumerate(vm):
            x0 = L - bw / 2
            if e["mean"][L] is None:
                ax.text(L, y, "not defined by CPC in this month", ha="center", va="center", fontsize=7, color=MUTED, style="italic"); continue
            c = (e.get("cal") or [None] * W)[L]
            if e["p_neg05"][L] is None:
                ax.text(L, y, "not defined by CPC in this month", ha="center", va="center", fontsize=7, color=MUTED, style="italic"); continue
            bar(y + 0.10, 0.44, e["p_neg05"][L], e["p_pos05"][L], x0, txt=not (c and c.get("no_skill")))
            dm = ""
            if pe and v in pe["valid"] and pe["mean"][pe["valid"].index(v)] is not None:
                dm = f"  Δ{e['mean'][L] - pe['mean'][pe['valid'].index(v)]:+.1f}"
            line = f"mean {e['mean'][L]:+.1f}{dm}"
            if c and "p_neg" in c:
                bar(y - 0.24, 0.16, c["p_neg"], c["p_pos"], x0, txt=False)
                ax.text(x0 + bw + 0.02, y - 0.24, f"{c['p_neg']*100:.0f}/{c['p_pos']*100:.0f}", ha="left", va="center", fontsize=7.0, color=INK)
                line += f"  ·  skill r {e['skill'][L]:+.2f}"
                if c.get("no_skill"):
                    ax.add_patch(Rectangle((x0 - 0.02, y - 0.36), bw + 0.04, 0.72, facecolor="white", alpha=0.55, edgecolor="#8a8680", hatch="////", lw=0.6, zorder=5))
                    ax.text(L, y + 0.10, f"no skill  ({e['p_neg05'][L]*100:.0f}% / {e['p_pos05'][L]*100:.0f}%)", ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK, zorder=6,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
            else:
                ax.text(x0 + bw / 2, y - 0.24, "no ERA5 reference — uncalibrated", ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")
            ax.text(L, y - 0.43, line, ha="center", va="center", fontsize=7.3, color=INK)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(-0.7, n - 0.4)
    ax.set_yticks(range(n)); ax.set_yticklabels([doc["indices"][k]["label"] for k in keys][::-1], fontsize=8.8)
    ax.set_xticks(range(W)); ax.set_xticklabels([f"{calendar.month_name[int(v[5:])]} {v[:4]}" for v in vm], fontsize=9)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    handles = [Patch(color="#2c5fa8", label=f"P(monthly index ≤ −{THR}σ)"), Patch(color="#d9d6d0", label="neutral"), Patch(color="#b4453c", label=f"P(monthly index ≥ +{THR}σ)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.59, 0.945), ncol=3, fontsize=8.4, frameon=False)
    il = lambda ym: f"{calendar.month_abbr[int(ym[4:])]} {ym[:4]}"
    fig.suptitle(f"Teleconnection phase probabilities by month — SEAS5, 51 members, {il(doc['issue'])} issue  ·  Δ = change in the mean vs the {il(doc['previous'])} issue",
                 fontsize=12.5, fontweight="bold", y=0.985, va="top")
    note = ("upper bar: raw member fractions (51 members → 2% steps; 0% = no member, not certainty)\n"
            "thin lower bar: hindcast-calibrated probabilities — the observed monthly index (ERA5) regressed on the hindcast ensemble mean over the 24 hindcast years 1993–2016 for this calendar month and lead, "
            f"Gaussian with the residual spread;\nskill r = correlation of the hindcast mean with ERA5; hatched = r below {SKILL_MIN}, no usable skill (the calibrated bar is then climatology by construction)")
    fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=7.8, color=MUTED)
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6}); plt.close(fig)
    print(f"  wrote {out.name}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    pats = Patterns()
    ym = a.issue; prev = previous_issues(ym, 1)[0]
    era, scale = None, None
    try:
        era, scale = era5_indices(pats)
    except Exception as e:                                             # noqa: BLE001
        print(f"  ERA5 reference skipped ({str(e)[:80]})", flush=True)
    cpc = {}
    try:
        cpc = cpc_published()
    except Exception as e:                                             # noqa: BLE001
        print(f"  CPC indices skipped ({str(e)[:80]})", flush=True)
    cur = compute(ym, pats, with_members=True, scale=scale)
    if not cur:
        raise SystemExit("no teleconnection fields on disk yet")
    add_skill(cur, ym, era, cpc)
    pv = compute(prev, pats, with_members=False, scale=scale)
    for e in pv.values():
        e.pop("_hc", None)
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": ym, "previous": prev,
           "groups": {"tele": [k for k in cur if cur[k]["group"] == "tele"], "strat": [k for k in cur if cur[k]["group"] == "strat"]},
           "indices": cur, "previous_indices": pv,
           "figures": {"tele_series": "seas5_tele_series.webp", "tele_probs": "seas5_tele_probs.webp", "strat_series": "seas5_strat_series.webp"},
           "thr": THR, "skill_min": SKILL_MIN,
           "skill_note": "correlation of the 24 hindcast years' ensemble mean with ERA5 (local store) at the valid month; blank where ERA5 lacks the field (stratospheric winds)" if era else None}
    for group, name in (("tele", "seas5_tele_series.webp"), ("strat", "seas5_strat_series.webp")):
        try:
            plot_series(doc, group, ASSETS / name)
        except Exception as e:                                         # noqa: BLE001
            print(f"  {name} FAILED ({str(e)[:100]})", flush=True); doc["figures"].pop(f"{group}_series", None)
    try:
        plot_probs(doc, ASSETS / "seas5_tele_probs.webp")
    except Exception as e:                                             # noqa: BLE001
        print(f"  seas5_tele_probs.webp FAILED ({str(e)[:100]})", flush=True); doc["figures"].pop("tele_probs", None)
    def clean(o):                                                  # NaN is not JSON; browsers reject the whole file
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return o
    OUT_JSON.write_text(json.dumps(clean(doc), separators=(",", ":")))
    print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size / 1e3:.0f} kB, {len(cur)} indices) in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
