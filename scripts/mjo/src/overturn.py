#!/usr/bin/env python3
"""Ensemble forecasts of the Hadley and Walker circulations: members corrected for their model's drift against ERA5.

The Hadley and Walker products read the AIFS-ENS analysis against ERA5 1991-2020 (mmsf.py, walker.py,
hadley_timeseries.py). A raw forecast minus that climatology is not the same reading: it carries the model's own
drift, which grows with lead time. So every model's members are first corrected by the model's MEAN error against
ERA5 at the same lead and season, measured on its own archived runs:

    x_hat(member, L) = x(member, L) - d(L, season),    d(L, season) = < x(L) - x_ERA5(valid time) >
                                                        over archived starts within +-45 days of the season

x is the zonal-mean v (vbar, on the 13 levels and ERA5's 0.25 deg latitudes, the climatology's grid) or the 5S-5N
divergent zonal wind (ud, walker.band_ud, on the chi solver's longitudes). Both are linear in the wind, so a drift
taken out of them is taken out of the streamfunctions; cell intensities and edges are then read per member.

ECMWF publishes no reforecast in its open data, so the archive is its own past runs on the Google mirror (IFS-ENS
from 2024-03, AIFS-ENS from 2025-07). d is measured at leads 0, 5, 10 and 15 days on a 5-day lattice of 00Z starts,
so every verifying time is a lattice day and ERA5 is read on the lattice only (one ARCO chunk holds all 37 levels:
~120 MB per field). Four members per start: d is the mean error, not the spread, and four members raise its standard
error by ~14 % over fifty. Between the measured leads d is interpolated linearly. d(0) is the model's analysis
offset from ERA5.

    python src/overturn.py truth   --since 2024-03-01 --out era5.npz            # ERA5 on the lattice
    python src/overturn.py archive --model ifs --since 2024-03-01 --out ifs.npz  # archived runs on the lattice
    python src/overturn.py pack era5*.npz ifs*.npz aifs*.npz                     # -> data/reference/overturn_ref.npz
    python src/overturn.py check                                                 # drift summary per model and lead

A part of a lattice range runs with --part k/n (every n-th lattice day from the k-th), so a backfill splits into
parallel jobs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "era5"))
from mmsf import LEVELS
from walker import band_ud
from doycal import noleap_doy

REF = Path(__file__).resolve().parent.parent / "data" / "reference" / "overturn_ref.npz"
LAT = np.linspace(90.0, -90.0, 721)  # ERA5's rows: the Psi climatology's latitudes
NLON_UD = 173  # the chi solver's DH2 longitudes at LMAX 42 (walker_clim_coeffs.nc)
EPOCH = pd.Timestamp("2024-03-01")  # lattice origin: starts are EPOCH + k * LATTICE days, 00Z
LATTICE = 5
DRIFT_LEADS = (0, 5, 10, 15)  # days; every verifying time is a lattice day
WINDOW = 45  # +- days of the season pooled into one drift estimate (triangular weights)
MODELS = {
    "aifs": {"label": "AIFS-ENS", "store": "aifs-ens", "first": "2025-07-01", "cf": True},
    "ifs": {"label": "IFS-ENS", "store": "ifs", "first": "2024-03-01", "cf": False},
}
ARCHIVE_PF = {"aifs": 3, "ifs": 4}  # control + 3 for AIFS-ENS, 4 perturbed for IFS-ENS (no control at pl)


# ── reductions shared by every dataset ──────────────────────────────────────────────────────────────────────────────
def zonal_v(v: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """(lat, lon) v -> zonal mean on LAT (ECMWF and ERA5 grids already are; GEFS 0.5 deg is interpolated)."""
    z = v.mean(axis=-1)
    if lat.size == LAT.size and np.allclose(lat, LAT):
        return z
    o = np.argsort(lat)
    return np.interp(LAT, lat[o], z[o])


def eq_ud(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(lat, lon) u, v -> the 5S-5N divergent zonal wind on the chi solver's longitudes (walker.band_ud)."""
    c = {"latitude": lat, "longitude": lon}
    ud, *_ = band_ud(xr.DataArray(u, dims=c, coords=c), xr.DataArray(v, dims=c, coords=c))
    return ud


def index_grib(path: Path) -> dict:
    """{(shortName, number, level, step_h): byte offset} for every message of a file."""
    import eccodes

    out = {}
    with open(path, "rb") as fh:
        while (h := eccodes.codes_grib_new_from_file(fh)) is not None:
            try:
                n = eccodes.codes_get(h, "number") if eccodes.codes_is_defined(h, "number") else 0
                key = (
                    eccodes.codes_get(h, "shortName"),
                    int(n),
                    int(eccodes.codes_get(h, "level")),
                    int(eccodes.codes_get(h, "endStep")),
                )
                out[key] = int(eccodes.codes_get(h, "offset"))
            finally:
                eccodes.codes_release(h)
    return out


def read_at(path: Path, offset: int):
    """(field (lat, lon) float64, lat, lon in 0..360) of the message at `offset`."""
    import eccodes

    with open(path, "rb") as fh:
        fh.seek(offset)
        h = eccodes.codes_grib_new_from_file(fh)
        try:
            nj, ni = eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni")
            la1 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
            la2 = eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
            lo1 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
            f = eccodes.codes_get_values(h).reshape(nj, ni)
        finally:
            eccodes.codes_release(h)
    lon = (lo1 + np.arange(ni) * 360.0 / ni) % 360.0
    o = np.argsort(lon)  # the chi solver and the zonal mean want 0..360 ascending
    return f[:, o], np.linspace(la1, la2, nj), lon[o]


def reduce_file(path: Path, members, steps_h, levels=LEVELS):
    """vbar (member, step, lev, 721) and ud (member, step, lev, 173) from a GRIB file holding u and v."""
    idx = index_grib(path)
    vb = np.full((len(members), len(steps_h), len(levels), LAT.size), np.nan, np.float32)
    ud = np.full((len(members), len(steps_h), len(levels), NLON_UD), np.nan, np.float32)
    for i, m in enumerate(members):
        for j, s in enumerate(steps_h):
            for k, lev in enumerate(levels):
                ku, kv = ("u", m, lev, s), ("v", m, lev, s)
                if ku not in idx or kv not in idx:
                    continue
                fu, lat, lon = read_at(path, idx[ku])
                fv, _, _ = read_at(path, idx[kv])
                vb[i, j, k] = zonal_v(fv, lat)
                ud[i, j, k] = eq_ud(fu, fv, lat, lon)
    return vb, ud


# ── the lattice ─────────────────────────────────────────────────────────────────────────────────────────────────────
def lattice(since, until, part=(0, 1)) -> pd.DatetimeIndex:
    """00Z lattice days in [since, until], every n-th from the k-th for --part k/n."""
    d0 = EPOCH + pd.Timedelta(days=LATTICE * int(np.ceil((pd.Timestamp(since) - EPOCH).days / LATTICE)))
    days = pd.date_range(d0, pd.Timestamp(until), freq=f"{LATTICE}D")
    k, n = part
    return days[k::n]


def truth(days) -> dict:
    """ERA5 vbar, ud at 00Z on each day (ARCO via era5_store; days ARCO has not published are skipped)."""
    import era5_store

    keep, vb, ud = [], [], []
    for d in days:
        t = pd.Timestamp(d).normalize()
        try:
            u = era5_store.get_u(t, LEVELS).sortby("level")
            v = era5_store.get_v(t, LEVELS).sortby("level")
        except Exception as e:  # noqa: BLE001
            print(f"  ERA5 {t:%Y-%m-%d}: {type(e).__name__} {str(e)[:80]}", flush=True)
            continue
        if bool(np.isnan(v.values).all()):  # ARCO pads days it has not published
            continue
        lat, lon = v.latitude.values, v.longitude.values
        vb.append(np.stack([zonal_v(v.values[k], lat) for k in range(len(LEVELS))]))
        ud.append(np.stack([eq_ud(u.values[k], v.values[k], lat, lon) for k in range(len(LEVELS))]))
        keep.append(t)
        print(f"  ERA5 {t:%Y-%m-%d} done", flush=True)
    return {
        "era5_time": np.array(keep, "datetime64[s]"),
        "era5_vbar": np.array(vb, np.float32).reshape(len(keep), len(LEVELS), LAT.size),
        "era5_ud": np.array(ud, np.float32).reshape(len(keep), len(LEVELS), NLON_UD),
    }


def archive(model: str, days) -> dict:
    """The model's archived 00Z runs at the drift leads: ensemble means of vbar, ud over the archived members."""
    import store as ecmwf

    cfg = MODELS[model]
    steps = tuple(24 * L for L in DRIFT_LEADS)
    keep, vb, ud, nm = [], [], [], []
    for d in days:
        cyc = ecmwf.Cycle(f"{d:%Y%m%d}", "00")
        parts = []
        try:
            if cfg["cf"]:
                p = ecmwf.ensure(cyc, ecmwf.Spec(cfg["store"], "cf", ("u", "v"), "pl", tuple(LEVELS), steps))
                parts.append(reduce_file(p, [0], steps))
                p.unlink(missing_ok=True)
            p = ecmwf.ensure(
                cyc, ecmwf.Spec(cfg["store"], "pf", ("u", "v"), "pl", tuple(LEVELS), steps, ARCHIVE_PF[model])
            )
            parts.append(reduce_file(p, list(range(1, ARCHIVE_PF[model] + 1)), steps))
            p.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {cfg['label']} {d:%Y-%m-%d}: {type(e).__name__} {str(e)[:80]}", flush=True)
            continue
        v = np.concatenate([x[0] for x in parts])
        u = np.concatenate([x[1] for x in parts])
        if np.isnan(v).all(axis=(0, 2, 3)).any():  # a lead missing from every member
            print(f"  {cfg['label']} {d:%Y-%m-%d}: incomplete, skipped", flush=True)
            continue
        keep.append(d)
        vb.append(np.nanmean(v, axis=0))
        ud.append(np.nanmean(u, axis=0))
        nm.append(v.shape[0])
        print(f"  {cfg['label']} {d:%Y-%m-%d}: {v.shape[0]} members", flush=True)
    n = len(keep)
    return {
        f"{model}_start": np.array(keep, "datetime64[s]"),
        f"{model}_vbar": np.array(vb, np.float32).reshape(n, len(DRIFT_LEADS), len(LEVELS), LAT.size),
        f"{model}_ud": np.array(ud, np.float32).reshape(n, len(DRIFT_LEADS), len(LEVELS), NLON_UD),
        f"{model}_nmem": np.array(nm, np.int16),
    }


def pack(paths, out: Path = REF) -> dict:
    """Merge partial truth/archive files (any order, overlaps allowed) into the reference, sorted and de-duplicated."""
    acc: dict[str, list] = {}
    for p in paths:
        with np.load(p) as f:
            for k in f.files:
                acc.setdefault(k, []).append(f[k])
    doc = {}
    for key_t, rest in [("era5_time", ("era5_vbar", "era5_ud"))] + [
        (f"{m}_start", (f"{m}_vbar", f"{m}_ud", f"{m}_nmem")) for m in MODELS
    ]:
        if key_t not in acc:
            continue
        t = np.concatenate(acc[key_t])
        _, first = np.unique(t, return_index=True)
        for k in (key_t, *rest):
            doc[k] = np.concatenate(acc[k])[first]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: (v.astype(np.float16) if v.dtype == np.float32 else v) for k, v in doc.items()})
    return doc


# ── the drift ───────────────────────────────────────────────────────────────────────────────────────────────────────
def load_ref(path: Path = REF) -> dict | None:
    if not path.exists():
        return None
    with np.load(path) as f:
        return {k: (f[k].astype(np.float32) if f[k].dtype == np.float16 else f[k]) for k in f.files}


def drift(ref: dict, model: str, doy: float, var: str = "vbar"):
    """d at DRIFT_LEADS for a start on common-year day `doy`: (len(DRIFT_LEADS), lev, n) and the starts used per lead.
    None if the reference has no pairs for this model."""
    if ref is None or f"{model}_start" not in ref or "era5_time" not in ref:
        return None
    starts = pd.DatetimeIndex(ref[f"{model}_start"])
    et = pd.DatetimeIndex(ref["era5_time"])
    epos = {t: i for i, t in enumerate(et)}
    dd = np.abs(noleap_doy(starts) - doy)
    dd = np.minimum(dd, 365.0 - dd)
    w = np.clip(1.0 - dd / WINDOW, 0.0, None)
    x, e = ref[f"{model}_{var}"], ref[f"era5_{var}"]
    out = np.full((len(DRIFT_LEADS), *x.shape[2:]), np.nan, np.float32)
    used = []
    for li, L in enumerate(DRIFT_LEADS):
        num, den, n = 0.0, 0.0, 0
        for si, s in enumerate(starts):
            j = epos.get(s + pd.Timedelta(days=L))
            if w[si] <= 0 or j is None:
                continue
            num = num + w[si] * (x[si, li] - e[j])
            den += w[si]
            n += 1
        if den > 0:
            out[li] = num / den
        used.append(n)
    if not any(used):
        return None
    return out, used


def drift_at(d: np.ndarray, lead_days) -> np.ndarray:
    """Linear interpolation of d (len(DRIFT_LEADS), ...) to daily leads (beyond 15 days: held at day 15)."""
    lead_days = np.asarray(lead_days, float)
    L = np.asarray(DRIFT_LEADS, float)
    i = np.clip(np.searchsorted(L, lead_days, side="right") - 1, 0, len(L) - 2)
    f = np.clip((lead_days - L[i]) / (L[i + 1] - L[i]), 0.0, 1.0)
    shape = (-1,) + (1,) * (d.ndim - 1)
    return d[i] * (1 - f).reshape(shape) + d[i + 1] * f.reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("truth", "archive"):
        s = sub.add_parser(name)
        s.add_argument("--since", default=str(EPOCH.date()))
        s.add_argument(
            "--until", default=str((pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=6)).date())
        )
        s.add_argument("--part", default="0/1")
        s.add_argument("--out", required=True)
        if name == "archive":
            s.add_argument("--model", required=True, choices=sorted(MODELS))
    s = sub.add_parser("pack")
    s.add_argument("parts", nargs="+")
    s.add_argument("--out", default=str(REF))
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd in ("truth", "archive"):
        k, n = (int(x) for x in a.part.split("/"))
        if a.cmd == "truth":
            days = lattice(a.since, a.until, (k, n))
            print(f"ERA5 on {len(days)} lattice days", flush=True)
            doc = truth(days)
        else:
            # a run verifies until 15 days after its start: archive starts whose leads ERA5 can still score later
            days = lattice(max(pd.Timestamp(a.since), pd.Timestamp(MODELS[a.model]["first"])), a.until, (k, n))
            print(f"{MODELS[a.model]['label']} on {len(days)} lattice starts", flush=True)
            doc = archive(a.model, days)
        np.savez_compressed(a.out, **doc)
        return 0
    if a.cmd == "pack":
        doc = pack(a.parts, Path(a.out))
        print({k: v.shape for k, v in doc.items()})
        return 0
    ref = load_ref()
    if ref is None:
        print(f"no reference at {REF}")
        return 1
    p_hpa = np.asarray(LEVELS, float)
    for m, cfg in MODELS.items():
        for doy in (15, 105, 196, 288):
            r = drift(ref, m, doy)
            if r is None:
                continue
            d, used = r
            k5 = int(np.argmin(np.abs(p_hpa - 200)))
            tropics = np.abs(LAT) <= 30
            print(
                f"  {cfg['label']:9s} doy {doy:3d}: starts per lead {used}; 200 hPa vbar drift, tropics RMS "
                + " ".join(f"d{L}={np.sqrt(np.nanmean(d[i, k5, tropics] ** 2)):.3f}" for i, L in enumerate(DRIFT_LEADS))
                + " m/s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
