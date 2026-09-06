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
def compute(ym: str, pats: Patterns, with_members: bool) -> dict:
    out = {}
    vm = valid_months(ym)
    months = [int(v[5:]) for v in vm]
    nl = len(vm)

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
            if not m.get("standardize"):                   # AO (EOF, raw) and the EPO/WPO boxes (metres): hindcast σ units
                sd = np.nanstd(hi, axis=0); sd = np.where(sd > 0, sd, np.nan)
                fi, hi = fi / sd, hi / sd
            out[sid] = entry(fi, hi, TELE_LABEL.get(sid, sid), "σ",
                             (m.get("name", sid) + ". CPC-style pattern applied to the member's 500 hPa height anomaly; annual standard deviations."
                              if m.get("field") != "slp" else "First EOF of sea-level pressure poleward of 20°N; positive = low pressure over the pole."),
                             "tele")
            out[sid]["_hc"] = hi                                          # kept for skill, dropped before writing
        # SOI from msl points
        ft = _point(fp, lat, lon, -17.5, -149.5) - _point(fp, lat, lon, -12.5, 130.5)
        ht = _point(hp, lat, lon, -17.5, -149.5) - _point(hp, lat, lon, -12.5, 130.5)
        sd = np.nanstd(ht, axis=0)
        fa = (ft - ht.mean(0)) / sd; ha = (ht - ht.mean(0)) / sd
        out["soi"] = entry(fa, ha, "Southern Oscillation index", "σ",
                           "Tahiti minus Darwin sea-level pressure, standardised by the hindcast spread at each lead; negative with El Niño.", "tele")
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
def era5_indices(pats: Patterns) -> dict | None:
    """{key: DataFrame-like dict of {(year, month): value}} from ERA5 monthly means, anomalies vs 1993–2016."""
    gl_pl, gl_sfc = ERA5 / "era5_gl_pl_1991-2025.grib", ERA5 / "era5_gl_sfc_1991-2025.grib"
    if not (gl_pl.exists() and gl_sfc.exists()):
        return None
    out = {}
    def monthly(ds_var):
        t = ds_var.time.values
        yrs = np.array([int(str(x)[:4]) for x in t]); mos = np.array([int(str(x)[5:7]) for x in t])
        return yrs, mos
    def anom(da, yrs, mos):
        base = (yrs >= 1993) & (yrs <= 2016)
        clim = np.stack([np.nanmean(da[(mos == m) & base], axis=0) for m in range(1, 13)])
        return da - clim[mos - 1]
    ds = _open(gl_pl, shortName="z", level=500)
    z = ds["z"].transpose("time", "latitude", "longitude"); yrs, mos = monthly(z)
    za = anom(z.values / G0, yrs, mos)
    dsm = _open(gl_sfc, shortName="msl")
    pm = dsm["msl"].transpose("time", "latitude", "longitude")
    pa = anom(pm.values / 100.0, yrs, mos)
    lat, lon = z.latitude.values, z.longitude.values
    nz = pats.to_ncep(xr.DataArray(za, dims=("t", "latitude", "longitude"), coords={"latitude": lat, "longitude": lon}))
    npm = pats.to_ncep(xr.DataArray(pa, dims=("t", "latitude", "longitude"), coords={"latitude": lat, "longitude": lon}))
    for sid in pats.ids:
        src = npm if pats.meta[sid].get("field") == "slp" else nz
        out[sid] = dict(zip(zip(yrs, mos), pats.index(sid, src, mos)))
    tah = pm.values[:, int(np.argmin(np.abs(lat + 17.5))), int(np.argmin(np.abs(lon + 149.5)))]
    dar = pm.values[:, int(np.argmin(np.abs(lat + 12.5))), int(np.argmin(np.abs(lon - 130.5)))]
    d = (tah - dar) / 100.0
    out["soi"] = dict(zip(zip(yrs, mos), d))                         # raw hPa; correlation is scale-free
    for lev, key in ((10, "u60n10"),):
        dsu = _open(gl_pl, shortName="u", level=lev); u = dsu["u"].transpose("time", "latitude", "longitude")
        vals = u.values; i = int(np.argmin(np.abs(u.latitude.values - 60)))
        out[key] = dict(zip(zip(yrs, mos), np.nanmean(vals[:, i, :], axis=-1)))
        j = int(np.argmin(np.abs(u.latitude.values + 60)))
        out["u60s10"] = dict(zip(zip(yrs, mos), np.nanmean(vals[:, j, :], axis=-1)))
        for lv in (10, 30, 50):
            dsu = _open(gl_pl, shortName="u", level=lv); uu = dsu["u"].transpose("time", "latitude", "longitude")
            m = (uu.latitude.values >= -5) & (uu.latitude.values <= 5)
            out[f"qbo{lv}"] = dict(zip(zip(yrs, mos), np.nanmean(uu.values[:, m, :], axis=(1, 2))))
    for lev in (10, 50, 100, 500, 1000):
        dsz = _open(gl_pl, shortName="z", level=lev); zz = dsz["z"].transpose("time", "latitude", "longitude")
        m = zz.latitude.values >= 65; w = np.cos(np.deg2rad(zz.latitude.values[m]))
        cap = np.nansum(np.nanmean(zz.values[:, m, :], axis=-1) * w, axis=-1) / w.sum() / G0
        out[f"nam{lev}"] = dict(zip(zip(yrs, mos), -cap))              # sign as the model index
    return out


def add_skill(idx: dict, ym: str, era: dict | None) -> None:
    """Correlate the hindcast ensemble-mean index per year with ERA5 at the valid month."""
    vm = valid_months(ym)
    for key, e in idx.items():
        hc = e.pop("_hc", None)
        if hc is None or era is None or key not in era:
            e["skill"] = None; continue
        n = hc.shape[0] // 24                                          # members per year; samples stack member-major
        r = []                                                         # (xarray stack over (number, time): time varies fastest)
        for L, v in enumerate(vm):
            mo = int(v[5:])
            yrs = list(range(1993, 2017))
            hmean = np.nanmean(hc[:, L].reshape(n, 24), axis=0)        # → per-year ensemble mean, years in order
            obs = np.array([era[key].get((y + (1 if (int(ym[4:]) - 1 + L) >= 12 else 0), mo), np.nan) for y in yrs])
            ok = np.isfinite(hmean) & np.isfinite(obs)
            r.append(round(float(np.corrcoef(hmean[ok], obs[ok])[0, 1]), 2) if ok.sum() >= 12 else None)
        e["skill"] = r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    pats = Patterns()
    ym = a.issue; prev = previous_issues(ym, 1)[0]
    cur = compute(ym, pats, with_members=True)
    if not cur:
        raise SystemExit("no teleconnection fields on disk yet")
    era = None
    try:
        era = era5_indices(pats)
    except Exception as e:                                             # noqa: BLE001
        print(f"  ERA5 skill skipped ({str(e)[:80]})", flush=True)
    add_skill(cur, ym, era)
    pv = compute(prev, pats, with_members=False)
    for e in pv.values():
        e.pop("_hc", None)
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": ym, "previous": prev,
           "groups": {"tele": [k for k in cur if cur[k]["group"] == "tele"], "strat": [k for k in cur if cur[k]["group"] == "strat"]},
           "indices": cur, "previous_indices": pv, "skill_note": "correlation of the 24 hindcast years' ensemble mean with ERA5 at the valid month" if era else None}
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
