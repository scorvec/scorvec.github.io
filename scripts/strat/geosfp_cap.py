#!/usr/bin/env python3
"""GEOS FP polar-cap (65-90N) geopotential height, daily, for the observed tail of the FORECAST dripping-paint plots
(2026-09-27; user: "zonal mean 'dripping paint' plots from the geps/gefs as well").

MERRA-2 is the reference the drips are standardised against (build_strat_history.py), but our MERRA-2 series ends on
2026-06-26 (GES DISC's on-prem OPeNDAP was retired). GEOS FP runs the same GEOS system and its assimilation archive
(NCCS OPeNDAP, inst3_3d_asm_Np, Dec 2017 -> today) carries height on the same 42 levels, so it supplies the recent
record. The reduction copies fetch_m2_strat_series.py exactly: the daily mean of the 00/06/12/18Z snapshots (which
cancels the semidiurnal tide), zonal mean, then the cos-weighted mean over 65-90N. A 0.5 x 1.25 deg subsample of the
0.25 x 0.3125 grid is used: a cap mean of a planetary-scale field loses nothing to it.

The GEOS FP minus MERRA-2 difference is measured on same days (`overlap`: every 7th day, 2024-07 -> 2026-06) and fitted
per level as a mean plus an annual harmonic, so the tail can be put on the MERRA-2 scale before it is standardised.

    python geosfp_cap.py overlap                       # -> reference/geosfp_cap_offset.json   (laptop, once)
    python geosfp_cap.py tail --days 90 --out reference/geosfp_cap_seed.json    # seed history
    python geosfp_cap.py tail --days 5 --out assets/sst/data/geosfp_cap_history.json   # daily append (Actions)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
URL = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Np"
LEVELS = [1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 250.0, 200.0, 100.0, 50.0, 10.0]   # the GEPS/GEFS/AIFS levels
OFFSET = HERE / "reference" / "geosfp_cap_offset.json"
SEED = HERE / "reference" / "geosfp_cap_seed.json"
M2 = REPO / "scripts" / "telecon" / "data" / "m2_strat"


class Fp:
    def __init__(self):
        self.d = None

    def open(self):
        for k in range(4):
            try:
                d = xr.open_dataset(URL)
                if not np.issubdtype(d.time.dtype, np.datetime64):   # a 504 on the .das leaves time undecoded
                    raise RuntimeError("time axis not decoded")
                self.d = d
                break
            except Exception as e:                                               # noqa: BLE001
                print(f"  open failed ({str(e)[:80]}); retry", flush=True)
                time.sleep(20 * (k + 1))
        if self.d is None:
            raise SystemExit("GEOS FP OPeNDAP unreachable")
        lev = self.d.lev.values
        self.li = [int(np.argmin(np.abs(lev - L))) for L in LEVELS]
        self.l0, self.l1 = min(self.li), max(self.li)
        lat = self.d.lat.values
        self.la = np.where(lat >= 65.0)[0][::2]
        self.t0 = pd.Timestamp(str(self.d.time.values[0])[:19])
        self.tn = pd.Timestamp(str(self.d.time.values[-1])[:19])
        return self

    def day(self, day: pd.Timestamp):
        """Cap mean per LEVEL (m) for one day, or None if a snapshot is missing. 1000 hPa may be NaN (see below)."""
        lat = self.d.lat.values[self.la]
        w = np.cos(np.deg2rad(lat))
        zms = []
        for h in (0, 6, 12, 18):
            t = day + pd.Timedelta(hours=h)
            if t > self.tn:
                return None
            it = int(round((t - self.t0) / pd.Timedelta(hours=3)))
            for k in range(7):                                                   # NCCS answers 504 under load
                try:
                    a = self.d["h"].isel(time=it, lev=slice(self.l0, self.l1 + 1),
                                         lat=slice(int(self.la[0]), int(self.la[-1]) + 1, 2),
                                         lon=slice(None, None, 4)).values
                    break
                except Exception as e:                                           # noqa: BLE001
                    if k == 6:
                        print(f"  {t} failed: {str(e)[:80]}", flush=True)
                        return None
                    time.sleep(15 * (k + 1))
            a = a[[i - self.l0 for i in self.li]]                               # (level, lat, lon)
            a = np.where(np.abs(a) < 1e5, a, np.nan)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                zms.append(np.nanmean(a, -1))                                   # (level, lat)
        # the MERRA-2 order (fetch_m2_strat_series): NaN-aware zonal mean per snapshot, NaN-aware mean over the four
        # snapshots, then the cos-weighted cap mean, which is NaN for a level with any all-missing latitude row.
        # Below-ground points are missing, and in summer the whole 1000 hPa surface is underground over much of the
        # Arctic (47 % of the cap on 2026-07-22, whole rows at 85-87N), so 1000 hPa is NaN on such days, as in MERRA-2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            zm = np.nanmean(np.array(zms), 0)
        cap = (zm * w).sum(-1) / w.sum()
        if not np.isfinite(cap[1:]).all():
            return None
        return cap


_FP = None


def _worker_day(day):
    """Process-pool worker: one GEOS FP handle per process (netCDF/HDF5 OPeNDAP reads are not thread-safe)."""
    global _FP
    if _FP is None:
        _FP = Fp().open()
    return day, _FP.day(day)


def many(days, procs=6):
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    with ProcessPoolExecutor(procs) as ex:
        for i, (d, c) in enumerate(ex.map(_worker_day, list(days))):
            out[d] = c
            if i % 15 == 0:
                print(f"  {i + 1}/{len(days)} {d:%Y-%m-%d} {'ok' if c is not None else 'missing'}", flush=True)
    return out


def m2_cap():
    import glob
    fs = sorted(glob.glob(str(M2 / "m2_strat_*.nc")))
    d = xr.open_mfdataset(fs, combine="by_coords")["cap_z"].sel(lev=LEVELS, method="nearest").load()
    return d.to_pandas()


def overlap(a) -> int:
    m2 = m2_cap()
    days = [d for d in pd.date_range(a.start, a.end, freq=f"{a.step}D")
            if d in m2.index and np.isfinite(m2.loc[d].values[1:]).all()]
    got = many(days, a.procs)
    rows = [(d, got[d] - m2.loc[d].values) for d in days if got.get(d) is not None]
    t = pd.DatetimeIndex([r[0] for r in rows]); D = np.array([r[1] for r in rows])
    x = 2 * np.pi * (t.dayofyear.values - 1) / 365.25
    X = np.column_stack([np.ones(len(t)), np.cos(x), np.sin(x)])
    out = {"levels": LEVELS, "n_days": len(t), "period": [f"{t.min():%Y-%m-%d}", f"{t.max():%Y-%m-%d}"],
           "method": "GEOS FP minus MERRA-2 polar-cap (65-90N) height, same days, daily mean of 00/06/12/18Z; per level "
                     "mean + annual harmonic (coefficients on 1, cos, sin of 2 pi (doy-1)/365.25)", "coef": {}, "resid_sd": {}}
    for k, L in enumerate(LEVELS):
        ok = np.isfinite(D[:, k])                                              # 1000 hPa: summer days drop out
        b = np.linalg.lstsq(X[ok], D[ok, k], rcond=None)[0]
        out["coef"][str(L)] = [round(float(v), 3) for v in b]
        out["resid_sd"][str(L)] = round(float(np.std(D[ok, k] - X[ok] @ b, ddof=3)), 2)
        out.setdefault("n_days_level", {})[str(L)] = int(ok.sum())
        print(f"  {L:6.0f} hPa: mean {b[0]:+7.1f} m, annual amplitude {np.hypot(b[1], b[2]):5.1f} m, residual sd "
              f"{out['resid_sd'][str(L)]:5.1f} m")
    OFFSET.write_text(json.dumps(out, indent=1))
    print(f"wrote {OFFSET} ({len(t)} days)")
    return 0


def offset_at(days: pd.DatetimeIndex) -> np.ndarray:
    """(n_days, level) GEOS FP - MERRA-2 offset to SUBTRACT from a GEOS FP cap series."""
    if not OFFSET.exists():                                   # not measured yet: no adjustment (|offset| < ~15 m)
        return np.zeros((len(days), len(LEVELS)))
    o = json.loads(OFFSET.read_text())
    x = 2 * np.pi * (days.dayofyear.values - 1) / 365.25
    X = np.column_stack([np.ones(len(days)), np.cos(x), np.sin(x)])
    return np.column_stack([X @ np.array(o["coef"][str(L)]) for L in LEVELS])


def load_history(extra: Path | None = None) -> pd.DataFrame:
    """Seed + live history, raw GEOS FP cap heights (m), columns = LEVELS."""
    h = {}
    for p in (SEED, extra):
        if p is not None and Path(p).exists():
            h.update(json.loads(Path(p).read_text()).get("days", {}))
    if not h:
        return pd.DataFrame(columns=LEVELS)
    df = pd.DataFrame.from_dict(h, orient="index")
    df.index = pd.DatetimeIndex(df.index); df.columns = [float(c) for c in df.columns]
    return df.sort_index()[LEVELS]


def tail(a) -> int:
    out = Path(a.out); out = out if out.is_absolute() else REPO / out
    have = json.loads(out.read_text()).get("days", {}) if out.exists() else {}
    fp = Fp().open()
    end = fp.tn.normalize() - pd.Timedelta(days=1) if fp.tn.hour < 18 else fp.tn.normalize()
    days = [d for d in pd.date_range(end - pd.Timedelta(days=a.days - 1), end, freq="D")
            if a.force or f"{d:%Y-%m-%d}" not in have]
    got = many(days, a.procs) if len(days) > 2 else {d: fp.day(d) for d in days}
    n = 0
    for dd, c in got.items():
        if c is None:
            continue
        have[f"{dd:%Y-%m-%d}"] = {str(L): (round(float(v), 2) if np.isfinite(v) else None) for L, v in zip(LEVELS, c)}
        n += 1
    keep = sorted(have)[-a.keep:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": "GEOS FP inst3_3d_asm_Np (NCCS OPeNDAP), 65-90N cos-weighted cap height, daily "
                                         "mean of 00/06/12/18Z, m (raw GEOS FP scale; subtract geosfp_cap_offset.json "
                                         "for the MERRA-2 scale)", "levels": LEVELS,
                               "days": {k: have[k] for k in keep}}, separators=(",", ":")))
    print(f"GEOS FP cap tail: {n} new day(s), {len(keep)} kept, last {keep[-1] if keep else None} -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["overlap", "tail"])
    ap.add_argument("--start", default="2024-07-01"); ap.add_argument("--end", default="2026-06-26")
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--days", type=int, default=5); ap.add_argument("--keep", type=int, default=120)
    ap.add_argument("--out", default=str(REPO / "assets" / "sst" / "data" / "geosfp_cap_history.json"))
    ap.add_argument("--force", action="store_true"); ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    return {"overlap": overlap, "tail": tail}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
