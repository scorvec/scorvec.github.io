#!/usr/bin/env python3
"""Observed side of the "East-based vs central" item of the El Niño monitor (enso.html).

events       every El Niño winter since 1950-51 as the monitor and the ENSO forecasts page class it: the event from
             CPC's ONI (c3s_enso_flavour.events_table), its DJF E and C from ERSSTv6 on the fixed 1991-2020 base
             projected on the site's EOFs (obs_seasons), east-based / basin-wide / central by c3s_enso_flavour.classify.
             The class on the CMIP6 composites' moving base (below) is recorded alongside.
regressions  DJF 2 m temperature, precipitation and 500 hPa height on E and C together, per standard deviation of
             each with the other held fixed (OLS), t-test per grid point, Benjamini-Hochberg FDR 10 % over the map.
             Indices and fields are anomalies from the mean of the other winters within +-10 years at every point,
             exactly as the CMIP6 runs (cmip6_enso_flavour.local_anom). Fewer than 8 observed east-based winters exist,
             too few for a composite of the class, so the observations are tested on the two indices.
fields       ERA5 2 m temperature (global) and 500 hPa height (Northern Hemisphere) from the ERA5 daily store
             (era5_local.py; a month counts only when every day is there), GPCP v2.3 monthly precipitation (NOAA NCEI)
             from scripts/sst/data/obs/gpcp; DJF means weighted by days in month; area-overlap means onto the CMIP6
             composites' 2.5-degree grids.

  python scripts/sst/build_enso_flavour_impacts_obs.py      -> reference/enso_flavour_impacts_obs.{npz,json}
"""

from __future__ import annotations

import pyproj  # noqa: F401  before xarray: the land mask goes through cartopy (libproj load order)
import calendar
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c3s_enso_flavour as F
import cmip6_enso_flavour as CF
import era5_local
from quantmet.stats import benjamini_hochberg

GPCP = HERE / "data" / "obs" / "gpcp"
OUT_STEM = "enso_flavour_impacts_obs"
YEARS = np.arange(1951, 2027)  # each DJF by its January; ERSSTv6 from 1950-01
ERA5 = {  # var: (store directory, file stem, destination grid, plausible DJF range)
    "tas": ("wb2_1p5_daily_global/t2m", "t2m", CF.AM, CF.RANGE["tas"]),
    "z500": ("wb2_1p5_daily/z500", "z500", CF.NH, CF.RANGE["z500"]),  # metres, whatever the attribute says
}


def _mdays(m: str) -> int:
    return calendar.monthrange(int(m[:4]), int(m[5:7]))[1]


def era5_djf(var: str) -> np.ndarray:
    """(len(YEARS), lat, lon) DJF means of the ERA5 daily store on the CMIP6 grid; NaN for an incomplete winter."""
    sub, stem, dst, (lo, hi) = ERA5[var]
    months, fields = [], []
    for f in era5_local._files(sub, stem):
        if "tail" in f.stem:
            continue
        with xr.open_dataset(f) as ds:
            da = era5_local._align(ds[stem].transpose("time", "latitude", "longitude")).sortby("latitude")
            t = da.indexes["time"]
            for m in (12, 1, 2):
                sel = t.month == m
                if not sel.any() or sel.sum() != _mdays(f"{t[sel][0].year}-{m:02d}"):
                    continue  # every day of the month, 29 February included
                x = da.isel(time=np.flatnonzero(sel)).mean("time").values.astype("float64")
                months.append(f"{t[sel][0].year:04d}-{m:02d}")
                fields.append(x)
            lat, lon = da.latitude.values, da.longitude.values
    A, B = CF.overlap_ops(lat, lon, *dst)
    x = CF.regrid(np.stack(fields), A, B)
    if np.nanmin(x) < lo or np.nanmax(x) > hi:
        raise ValueError(f"ERA5 {var} monthly means outside {lo}..{hi}: {np.nanmin(x):.1f}..{np.nanmax(x):.1f}")
    return CF.djf(x, months, np.array([_mdays(m) for m in months], float), YEARS)


def gpcp_djf() -> np.ndarray:
    """(len(YEARS), lat, lon) DJF means of GPCP v2.3 monthly precipitation (mm/day) on the Americas grid."""
    files = {}
    for f in sorted(GPCP.glob("gpcp_v02r03_monthly_d*_c*.nc")):  # a month re-issued later: the newest file wins
        ym = f.name.split("_d")[1][:6]
        files[f"{ym[:4]}-{ym[4:]}"] = f
    months = [m for m in sorted(files) if int(m[5:7]) in (12, 1, 2)]
    fields = []
    for m in months:
        with xr.open_dataset(files[m]) as ds:
            v = ds["precip"].isel(time=0).sortby("latitude")
            x = v.values.astype("float64")
            lat, lon = v.latitude.values, v.longitude.values
        if not np.isfinite(x).all() or x.min() < 0 or x.max() > 200:
            raise ValueError(f"GPCP {m}: values outside 0..200 mm/day or missing")
        fields.append(x)
    A, B = CF.overlap_ops(lat, lon, *CF.AM)
    x = CF.regrid(np.stack(fields), A, B)
    return CF.djf(x, months, np.array([_mdays(m) for m in months], float), YEARS)


def indices(obs: dict, eof: dict) -> tuple[np.ndarray, np.ndarray]:
    """DJF E and C on the moving base (anomalies from the mean of the other winters within +-10 years)."""
    a = CF.djf(obs["a"], obs["months"], np.array([_mdays(m) for m in obs["months"]], float), YEARS)
    return F.project(CF.local_anom(a[None])[0], eof)


def events(obs: dict, eof: dict, E: np.ndarray, C: np.ndarray) -> list[dict]:
    """The observed El Niño winters on the site's rule (fixed base), with the moving-base class beside it."""
    seas = F.obs_seasons(obs, eof)
    table = F.events_table(seas)
    djf_ = seas["DJF"]
    out = []
    for y, e in table.iterrows():
        if e["kind"] != "el nino" or y not in djf_.index:
            continue
        r = djf_.loc[y]
        k = int(np.searchsorted(YEARS, y + 1))  # the DJF that event peaks in
        out.append(
            dict(
                label=e["label"],
                year=int(y),
                strength=e["strength"],
                peak_oni=round(float(e["peak_oni"]), 2),
                E=round(float(r["E"]), 2),
                C=round(float(r["C"]), 2),
                klass=str(F.classify(r["E"], r["C"])),
                E_moving=round(float(E[k]), 2),
                C_moving=round(float(C[k]), 2),
                klass_moving=str(F.classify(E[k], C[k])),
            )
        )
    return out


def main() -> int:
    grid, eof = F.load_eof_reference()
    obs = F.load_obs()
    if not (np.allclose(obs["lat"], grid["lat"]) and np.allclose(obs["lon"], grid["lon"])):
        raise SystemExit("ERSST box and EOF grid differ")
    E, C = indices(obs, eof)
    ev = events(obs, eof, E, C)
    land = CF.land_fraction(*CF.AM) >= 0.5
    npz = {"lat": CF.AM[0], "lon": CF.AM[1], "zlat": CF.NH[0], "zlon": CF.NH[1]}
    meta = {
        "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "events": ev,
        "counts": {k: sum(e["klass"] == k for e in ev) for k in CF.CLASSES},
        "counts_moving": {k: sum(e["klass_moving"] == k for e in ev) for k in CF.CLASSES},
        "maps": {},
    }
    fields = {"tas": era5_djf("tas"), "pr": gpcp_djf(), "z500": era5_djf("z500")}
    npz["zclim"] = np.nanmean(fields["z500"], axis=0).astype("float32")  # ERA5 DJF mean height, for contours
    for v, X in fields.items():
        A = CF.local_anom(X[None])[0]
        ok = np.isfinite(E) & np.isfinite(C) & np.isfinite(A).all(axis=(1, 2))
        beta, p = CF.ols(np.column_stack([np.ones(ok.sum()), E[ok], C[ok]]), A[ok])
        yrs = YEARS[ok]
        for j, k in ((1, "E"), (2, "C")):
            f, s = beta[j], benjamini_hochberg(p[j], CF.ALPHA)
            if v == "pr":
                f, s = CF.to_pct(f, s, np.nanmean(X[ok], axis=0))
            npz[f"{v}|{k}|f"], npz[f"{v}|{k}|s"] = f.astype("float32"), s
            meta["maps"][f"{v}|{k}"] = dict(
                n=int(ok.sum()), span=[int(yrs[0]), int(yrs[-1])], sig_frac=CF.sig_frac(v, s, f, land)
            )
            print(
                f"  {v} {k}: {ok.sum()} winters {yrs[0]}-{yrs[-1]}, {100 * meta['maps'][f'{v}|{k}']['sig_frac']:.0f} % significant"
            )
    np.savez_compressed(CF.REF / f"{OUT_STEM}.npz", **npz)
    (CF.REF / f"{OUT_STEM}.json").write_text(
        json.dumps(meta, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    )
    print(f"events {meta['counts']} (moving base {meta['counts_moving']}); wrote {OUT_STEM}.{{npz,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
