#!/usr/bin/env python3
"""Day-of-year climatology + interannual spread of the Hadley cell diagnostics.

Reads the ERA5 zonal-mean v samples already on disk (5-day stride, 1991-2020,
built once by build_mmsf_clim.py) — nothing is downloaded — computes the cell
INTENSITY and EDGE LATITUDES for every individual sample, and only then forms
the day-of-year statistics.

That order matters. The existing mmsf_clim_coeffs.nc is a harmonic fit to Ψ
itself, so every derived quantity taken from it is a property of the 30-year
MEAN circulation, not the typical circulation:

  * cell edges vary by ~10° between years, so the mean Ψ has a smeared, weak
    subtropical gradient — differentiating it understates a real week's descent
    rate by ~2x, and its argmax latitude jumps 26 -> 34 -> 46 deg through one
    summer as competing weak maxima trade places.
  * the mean summer cell is so washed out that Ψ_mid barely crosses zero, which
    is why edge latitudes read off the harmonic clim came back at +80 deg.

Averaging the metric per sample instead gives the distribution a given week
should actually be scored against, and hands us percentiles for free.

    python src/build_hadley_clim.py                 # -> data/reference/hadley_clim.nc
    python src/build_hadley_clim.py --window 10     # +/- days pooled per doy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from mmsf import LEVELS, hadley_cells, streamfunction   # noqa: E402

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OUT = REF / "hadley_clim.nc"
CACHE = Path.home() / "mjo" / "era5_cache" / "mmsf_vbar_1991-2020_s5.nc"

# nh/sh edge lists come back south->north, so the poleward edge is the far one
METRICS = ("nh_psi", "nh_core", "nh_edge_eq", "nh_edge_pole",
           "sh_psi", "sh_core", "sh_edge_eq", "sh_edge_pole")
PCTS = (10, 25, 50, 75, 90)
# Below this the cell is not really there (the summer cell collapses for whole
# weeks). Intensity stays reportable — it is a real measurement of a weak cell —
# but an "edge latitude" of a cell that barely exists is noise, and unmasked it
# both spikes the observed series and blows the climatological band open. Gated
# identically here and in the plotted series so the two stay comparable.
MIN_PSI = 1.0


def sample_metrics(psi, p_hpa, lat) -> dict:
    """One Ψ field → the eight scalars, NaN where a cell/edge is unresolved."""
    c = hadley_cells(psi, p_hpa, lat)
    out = {m: np.nan for m in METRICS}
    nh, sh = c.get("nh"), c.get("sh")
    if nh:
        out["nh_psi"], out["nh_core"] = nh["psi"], nh["core_lat"]
        if abs(nh["psi"]) >= MIN_PSI:
            lo, hi = nh["edges"]                   # south, north
            out["nh_edge_eq"] = np.nan if lo is None else lo
            out["nh_edge_pole"] = np.nan if hi is None else hi
    if sh:
        out["sh_psi"], out["sh_core"] = sh["psi"], sh["core_lat"]
        if abs(sh["psi"]) >= MIN_PSI:
            lo, hi = sh["edges"]                   # south = poleward for the SH cell
            out["sh_edge_pole"] = np.nan if lo is None else lo
            out["sh_edge_eq"] = np.nan if hi is None else hi
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--window", type=int, default=7,
                    help="+/- days averaged WITHIN each year to mimic a weekly mean")
    ap.add_argument("--smooth", type=int, default=15,
                    help="running-mean length (days) applied along doy at the end")
    a = ap.parse_args()

    cpath = Path(a.cache)
    if not cpath.exists():
        print(f"ERA5 sample cache {cpath} not found — build_mmsf_clim.py creates it.",
              file=sys.stderr)
        return 1
    ds = xr.open_dataset(cpath)
    lat = ds.latitude.values.astype(float)
    p_hpa = ds.level.values.astype(float)
    p_pa = p_hpa * 100.0
    vbar = ds["vbar"].values
    times = pd.to_datetime(ds.time.values)
    print(f"{len(times)} ERA5 samples {times[0]:%Y-%m-%d}…{times[-1]:%Y-%m-%d} "
          f"({cpath.name}, no download)", flush=True)

    rows = []
    for i in range(len(times)):
        psi = streamfunction(vbar[i], p_pa, lat)
        rows.append(sample_metrics(psi, p_hpa, lat))
        if i % 400 == 0:
            print(f"  {i}/{len(times)} … {times[i]:%Y-%m-%d}", flush=True)
    df = pd.DataFrame(rows, index=times)
    doy = df.index.dayofyear.values

    # The observed frames are 7-day means, so the climatology must be a
    # distribution of comparable objects — not of instantaneous 12Z fields,
    # whose spread is far wider and would flatter any weekly value into
    # "normal". So: average each YEAR's samples inside the +/-window band
    # first (5-day stride -> 2-3 samples, ~weekly), then take the spread
    # ACROSS the 30 years. n per doy is therefore ~30, one value per year.
    yrs = df.index.year.values
    uy = np.unique(yrs)
    stats = {m: np.full((len(PCTS) + 2, 366), np.nan) for m in METRICS}
    counts = np.zeros(366, int)
    for d in range(1, 367):
        off = np.abs(((doy - d + 182) % 365) - 182)          # circular distance
        band = off <= a.window
        for k in METRICS:
            col = df[k].values
            per_year = []
            for y in uy:
                v = col[band & (yrs == y)]
                v = v[np.isfinite(v)]
                if v.size:
                    per_year.append(v.mean())
            v = np.asarray(per_year)
            if k == METRICS[0]:
                counts[d - 1] = v.size
            if v.size < 15:                                   # too thin to trust
                continue
            stats[k][0, d - 1] = v.mean()
            stats[k][1, d - 1] = v.std(ddof=1)
            for j, q in enumerate(PCTS):
                stats[k][2 + j, d - 1] = np.percentile(v, q)

    # light circular running mean along doy — n~30 makes raw percentile curves
    # jitter from year-swapping at the tails, which is estimator noise, not signal
    if a.smooth > 1:
        w = int(a.smooth)
        for k in METRICS:
            for r in range(stats[k].shape[0]):
                row = stats[k][r]
                ok = np.isfinite(row)
                if ok.sum() < w:
                    continue
                filled = np.interp(np.arange(366), np.flatnonzero(ok), row[ok])
                pad = np.concatenate([filled[-w:], filled, filled[:w]])
                sm = np.convolve(pad, np.ones(w) / w, mode="same")[w:-w]
                stats[k][r] = np.where(ok, sm, np.nan)

    out = xr.Dataset(
        {k: (("stat", "doy"), stats[k]) for k in METRICS},
        coords={"stat": ["mean", "sd"] + [f"p{q}" for q in PCTS],
                "doy": np.arange(1, 367)},
        attrs={"source": cpath.name, "window_days": a.window,
               "smooth_days": a.smooth, "n_samples": len(times),
               "basis": ("spread is ACROSS YEARS of each year's +/-window mean "
                         "(~weekly), matching the 7-day-mean observed frames"),
               "note": ("per-sample Hadley metrics from ERA5 1991-2020 (5-day "
                        "stride), pooled by day-of-year; intensity in 10^10 kg/s, "
                        "latitudes in degrees north")})
    out["n_pooled"] = ("doy", counts)
    REF.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT)
    print(f"wrote {OUT}  (pooled n/doy: {counts.min()}–{counts.max()})")

    for lab, d in (("1 Feb", 32), ("1 May", 121), ("1 Aug", 213), ("1 Nov", 305)):
        r = out.sel(doy=d)
        print(f"  {lab}: NH Ψmax {float(r.nh_psi.sel(stat='p50')):+6.2f} "
              f"[{float(r.nh_psi.sel(stat='p10')):+5.2f},{float(r.nh_psi.sel(stat='p90')):+5.2f}]"
              f"  edge {float(r.nh_edge_pole.sel(stat='p50')):+5.1f}"
              f"  |  SH Ψmax {float(r.sh_psi.sel(stat='p50')):+6.2f} "
              f"edge {float(r.sh_edge_pole.sel(stat='p50')):+6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
