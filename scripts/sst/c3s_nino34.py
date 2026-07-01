#!/usr/bin/env python3
"""C3S multi-model ENSO seasonal forecast — traditional ONI and RONI.

The Copernicus Climate Change Service (C3S) multi-system seasonal ensemble, one
trajectory per originating centre, shown two ways:

  ONI  — the traditional Niño-3.4 SST anomaly (3-month running mean).
  RONI — the Relative Oceanic Niño Index (CPC/ECMWF): the Niño-3.4 anomaly minus
         the 20°S–20°N tropical-mean anomaly, rescaled per calendar month by
         σ(ONI)/σ(relative) so it stays in °C on ONI's thresholds. RONI removes
         the background tropical warming that inflates the traditional index.

Both panels show every model plus the multi-model mean and the model-to-model
spread. Each model is anomalized against its own 1993–2016 hindcast (drift/bias
removed), putting all centres on one scale.

Only the 20°S–20°N tropical band is pulled from the CDS (an area-subset, coarsened
to keep it small), which holds both the Niño-3.4 box and the tropical mean and
keeps the job out of the big-request MARS queue. Each model's hindcast climatology
(Niño-3.4 and tropical mean, per start-month, per lead) is cached in
c3s_nino34_clim.csv (committed), so only a never-seen start-month pays the
reforecast download. The RONI rescale reuses roni_sigma.json (the site's
OISST-derived σ(ONI)/σ(relative), CPC/ECMWF method).

Data: CDS `seasonal-monthly-single-levels`, sea_surface_temperature, monthly_mean.
Requires ~/.cdsapirc (or CDSAPI_URL/CDSAPI_KEY).

  python c3s_nino34.py                 # latest issue, all models
  python c3s_nino34.py --issue 202606  # specific issue
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE.parents[1]
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data" / "c3s"
CLIM_PATH = HERE / "c3s_nino34_clim.csv"         # committed cache (per model, start-month, lead)
SIGMA_PATH = HERE / "roni_sigma.json"            # RONI σ(ONI)/σ(relative) per calendar month

DATASET = "seasonal-monthly-single-levels"
AREA = [20, -180, -20, 180]                       # tropical band (N, W, S, E) — holds N3.4 + tropical mean
# Coarse grid on purpose: the lagged burst ensembles (NCEP/UKMO/BoM) decode into a
# huge (number × init × step) hypercube, and at 1° the tropical band peaks ~22 GB
# RAM (OOMs a loaded machine). At 5° it is ~6 GB. Resolution bias cancels in the
# forecast−hindcast anomaly (both sampled identically), so ONI/RONI are unaffected;
# the Niño-3.4 and 20°S–20°N means are large-area averages of a smooth field.
GRID = [5.0, 5.0]
LEADS = ["1", "2", "3", "4", "5", "6"]
MAXLEAD = 6
CLIM_YEARS = [str(y) for y in range(1993, 2017)]  # C3S common hindcast period
N34_LAT, N34_LON = (-5, 5), (-170, -120)          # Niño-3.4 in −180..180 longitude
TROP_LAT = (-20, 20)                              # tropical mean, all longitudes

# (centre, system, label, colour) — C3S centres with current Niño-3.4 data on the
# CDS. CMCC (35) and JMA have no recent data as of 2026-07 and are omitted; add
# them back here if/when they return.
MODELS = [
    ("ecmwf",        "51",  "ECMWF SEAS5",    "#d62728"),
    ("ukmo",         "610", "UKMO GloSea6",   "#1f77b4"),
    ("meteo_france", "9",   "Météo-France 9", "#2ca02c"),
    ("dwd",          "22",  "DWD GCFS2",      "#9467bd"),
    ("ncep",         "2",   "NCEP CFSv2",     "#ff7f0e"),
    ("eccc",         "4",   "ECCC CanSIPS",   "#8c564b"),
    ("bom",          "2",   "BoM ACCESS-S2",  "#17becf"),
]


def _client():
    import cdsapi
    return cdsapi.Client(timeout=600, quiet=True, progress=False,
                         wait_until_complete=True, retry_max=1)


def _retrieve(centre, system, years, month, dest: Path) -> bool:
    """Area-subset SST retrieval → dest. False on MarsNoData / failure (skip)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = {
        "originating_centre": centre, "system": system,
        "variable": "sea_surface_temperature", "product_type": "monthly_mean",
        "year": list(years), "month": month, "leadtime_month": LEADS,
        "area": AREA, "grid": GRID, "data_format": "grib",
    }
    for attempt in range(2):
        try:
            _client().retrieve(DATASET, req, str(dest))
            return dest.exists() and dest.stat().st_size > 0
        except Exception as e:
            msg = str(e).replace("\n", " ")
            if "no data" in msg.lower():
                print(f"    {centre}/{system} {month}: no data on CDS — skipping", file=sys.stderr)
                return False
            print(f"    {centre}/{system} {month}: attempt {attempt+1} failed ({msg[:90]})", file=sys.stderr)
            time.sleep(8)
    return False


def _wmean(da, lat_rng, lon_rng=None):
    """Cosine-latitude-weighted mean over a lat (and optional lon) box, skipna."""
    m = (da["latitude"] >= lat_rng[0]) & (da["latitude"] <= lat_rng[1])
    sub = da.where(m)
    if lon_rng is not None:
        ml = (da["longitude"] >= lon_rng[0]) & (da["longitude"] <= lon_rng[1])
        sub = sub.where(ml)
    w = np.cos(np.deg2rad(da["latitude"]))
    return sub.weighted(w).mean(("latitude", "longitude"), skipna=True)


def _indices_by_lead(path: Path, start_month: int):
    """{lead(1..MAXLEAD) → (n34_members[], trop_members[])} in °C for one GRIB.

    Model-agnostic: pool every finite (member × init × step) cell by the calendar
    month of its valid_time so clean monthly-mean files and lagged burst ensembles
    reduce identically. lead = months after start_month (mod-12)."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds["sst"] - 273.15
    n34 = _wmean(da, N34_LAT, N34_LON)
    trop = _wmean(da, TROP_LAT)
    vt = da["valid_time"]
    (n34_b, _), (trop_b, vt_b) = xr.broadcast(n34, vt), xr.broadcast(trop, vt)
    n34_f = np.asarray(n34_b.values).ravel()
    trop_f = np.asarray(trop_b.values).ravel()
    vtm = pd.to_datetime(np.asarray(vt_b.values).ravel()).month
    ds.close()
    out = {}
    for L in range(1, MAXLEAD + 1):
        want = ((vtm - start_month) % 12) == L
        n = n34_f[want & np.isfinite(n34_f)]
        t = trop_f[want & np.isfinite(trop_f)]
        if n.size and t.size:
            out[L] = (n, t)
    return out


# ── hindcast climatology (per model, per start-month), cached ─────────────────
def _load_clim() -> pd.DataFrame:
    if CLIM_PATH.exists():
        return pd.read_csv(CLIM_PATH)
    return pd.DataFrame(columns=["model", "month", "lead", "clim_n34", "clim_trop"])


def _clim_rows(df, model, month):
    sub = df[(df.model == model) & (df.month == month)].sort_values("lead")
    if len(sub) < MAXLEAD:
        return None
    n = sub.set_index("lead")["clim_n34"].reindex(range(1, MAXLEAD + 1)).values
    t = sub.set_index("lead")["clim_trop"].reindex(range(1, MAXLEAD + 1)).values
    return (n, t) if np.isfinite(n).all() and np.isfinite(t).all() else None


def _save_clim(model, month, clim_n34, clim_trop):
    df = _load_clim()
    df = df[~((df.model == model) & (df.month == month))]
    rows = pd.DataFrame({"model": model, "month": month, "lead": range(1, MAXLEAD + 1),
                         "clim_n34": np.round(clim_n34, 4), "clim_trop": np.round(clim_trop, 4)})
    df = pd.concat([df, rows], ignore_index=True).sort_values(["model", "month", "lead"])
    tmp = CLIM_PATH.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    os.replace(tmp, CLIM_PATH)


def climatology(model, centre, system, month):
    """(clim_n34[lead], clim_trop[lead]) for a model & start-month, cached."""
    cached = _clim_rows(_load_clim(), model, month)
    if cached is not None:
        return cached
    print(f"  building hindcast climatology: {model} start-month {month:02d} "
          f"({CLIM_YEARS[0]}–{CLIM_YEARS[-1]}) …", flush=True)
    dest = DATA / "hindcast" / f"{centre}_{system}_{month:02d}.grib"
    if not _retrieve(centre, system, CLIM_YEARS, f"{month:02d}", dest):
        return None
    by_lead = _indices_by_lead(dest, month)
    if len(by_lead) < MAXLEAD:
        return None
    cn = np.array([by_lead[L][0].mean() for L in range(1, MAXLEAD + 1)])
    ct = np.array([by_lead[L][1].mean() for L in range(1, MAXLEAD + 1)])
    _save_clim(model, month, cn, ct)
    return cn, ct


def forecast_series(model, centre, system, ym):
    """(oni[lead], roni[lead]) ensemble-mean anomaly series for one model & issue,
    each a length-MAXLEAD array of 3-month-running values (°C); None if missing."""
    month = int(ym[4:])
    clim = climatology(model, centre, system, month)
    if clim is None:
        return None
    cn, ct = clim
    dest = DATA / "forecast" / f"{centre}_{system}_{ym}.grib"
    if not _retrieve(centre, system, [ym[:4]], ym[4:], dest):
        return None
    by_lead = _indices_by_lead(dest, month)
    if len(by_lead) < MAXLEAD:
        return None
    n34 = np.array([by_lead[L][0].mean() for L in range(1, MAXLEAD + 1)])
    trop = np.array([by_lead[L][1].mean() for L in range(1, MAXLEAD + 1)])
    oni_anom = n34 - cn
    rel_anom = (n34 - trop) - (cn - ct)
    issue = pd.Timestamp(int(ym[:4]), month, 1)
    idx = [issue + pd.DateOffset(months=L) for L in range(1, MAXLEAD + 1)]
    oni = pd.Series(oni_anom, idx).rolling(3, center=True, min_periods=1).mean()
    rel = pd.Series(rel_anom, idx).rolling(3, center=True, min_periods=1).mean()
    roni = _scale_roni(rel)
    return oni.values, roni.values


def _scale_roni(rel: pd.Series) -> pd.Series:
    """rel × per-month σ(ONI)/σ(relative) (CPC/ECMWF) → RONI in °C."""
    if not SIGMA_PATH.exists():
        return rel
    tab = json.loads(SIGMA_PATH.read_text()).get("scale_by_month", {})
    s = {int(k): float(v) for k, v in tab.items()}
    fac = [s.get(m, 1.0) for m in pd.DatetimeIndex(rel.index).month]
    return rel * pd.Series(fac, index=rel.index)


def latest_issue() -> str:
    now = pd.Timestamp.utcnow()
    for back in range(0, 3):
        ym = (now - pd.DateOffset(months=back)).strftime("%Y%m")
        if _retrieve("ecmwf", "51", [ym[:4]], ym[4:], DATA / "probe" / f"ecmwf_{ym}.grib"):
            return ym
    raise SystemExit("no recent ECMWF SEAS5 issue found on the CDS")


def plot(ym: str, out: Path):
    issue = pd.Timestamp(int(ym[:4]), int(ym[4:]), 1)
    valid = [issue + pd.DateOffset(months=L) for L in range(1, MAXLEAD + 1)]

    oni_s, roni_s = {}, {}
    for centre, system, label, colour in MODELS:
        r = forecast_series(centre, centre, system, ym)
        if r is None or not np.isfinite(r[0]).any():
            print(f"  {label:16s} unavailable — skipped", flush=True)
            continue
        oni_s[label] = (r[0], colour)
        roni_s[label] = (r[1], colour)
        print(f"  {label:16s} ONI {r[0][0]:+.2f}→{r[0][-1]:+.2f}  RONI {r[1][0]:+.2f}→{r[1][-1]:+.2f}", flush=True)
    if not oni_s:
        raise SystemExit("no models returned data")

    fig, axes = plt.subplots(2, 1, figsize=(11, 9.4), sharex=True)

    def panel(ax, series, title, ylab):
        stack = []
        for label, (y, colour) in series.items():
            ax.plot(valid, y, color=colour, lw=1.5, marker="o", ms=3.5, label=label, zorder=3)
            stack.append(y)
        stack = np.vstack(stack)
        ax.fill_between(valid, np.nanmin(stack, 0), np.nanmax(stack, 0),
                        color="0.6", alpha=0.14, zorder=1, label="Model range")
        ax.plot(valid, np.nanmean(stack, 0), color="k", lw=2.8, label="Multi-model mean", zorder=5)
        ax.axhline(0, color="0.5", lw=0.8)
        for g in (0.5, 1.0, 1.5, -0.5, -1.0, -1.5):
            ax.axhline(g, color="0.75", lw=0.6, ls=":")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="x", alpha=0.2)

    panel(axes[0], oni_s, "Traditional ONI  (Niño-3.4 anomaly, 3-month running mean)",
          "ONI (°C)")
    panel(axes[1], roni_s, "RONI  (relative index, CPC/ECMWF — tropical-mean removed)",
          "RONI (°C)")
    axes[0].legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for lab in axes[1].get_xticklabels():
        lab.set_rotation(45); lab.set_ha("right"); lab.set_fontsize(8)
    fig.suptitle(f"C3S multi-model ENSO forecast — issued {issue:%b %Y}\n"
                 f"{len(oni_s)} systems · ONI and RONI · anomaly vs each model's 1993–2016 hindcast",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} ({len(oni_s)} models)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", help="YYYYMM (default: latest available)")
    ap.add_argument("--out", default=str(ASSETS / "c3s_nino34.webp"))
    args = ap.parse_args()
    ym = args.issue or latest_issue()
    print(f"C3S multi-model ENSO (ONI+RONI) — issue {ym}")
    plot(ym, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
