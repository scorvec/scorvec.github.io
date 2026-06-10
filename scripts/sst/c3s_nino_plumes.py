#!/usr/bin/env python3
"""C3S multi-model Niño-3.4 seasonal forecast plumes (Copernicus CDS).

Pulls the C3S seasonal-forecast SST over the Niño-3.4 box (5°S-5°N, 170°W-120°W) for every
contributing centre, turns each ensemble member into a Niño-3.4 SST anomaly vs that model's own
1993-2016 hindcast climatology (the standard C3S bias-removal), and plots the member plumes,
each model's mean, and the multi-model "super-ensemble" mean. The previous month's init mean is
overlaid so the month-on-month shift is visible.

Subsetting to the tiny Niño-3.4 box keeps every CDS request small; requests are issued one at a
time (the dataset rate-limits concurrent jobs) and every download is cached, so re-runs and the
monthly Action only fetch the new forecast init (hindcast climatologies are reused forever).

    python scripts/sst/c3s_nino_plumes.py                       # latest init + prior, then plot
    python scripts/sst/c3s_nino_plumes.py --init 2026-06        # a specific init
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "c3s"                       # raw .nc cache (gitignored)
SYSJSON = HERE / "metar" / "c3s_systems.json"       # discovered working systems (committed)
DS_FC = "seasonal-monthly-single-levels"            # raw ensemble members (absolute SST)
DS_PP = "seasonal-postprocessed-single-levels"      # pre-computed ensemble-mean anomaly (no hindcast pull)
NINO34_AREA = [5, 190, -5, 240]                     # N, W, S, E
LEADS = ["1", "2", "3", "4", "5", "6"]

# candidate `system` numbers per centre (first that downloads wins; cached to SYSJSON)
CENTRES = {
    "ecmwf":        ([51],            "#d62728", "ECMWF SEAS5"),
    "ukmo":         ([604, 603, 602], "#1f77b4", "UK Met Office"),
    "meteo_france": ([9, 8],          "#2ca02c", "Météo-France"),
    "dwd":          ([22, 21],        "#9467bd", "DWD"),
    "cmcc":         ([35],            "#ff7f0e", "CMCC"),
    "ncep":         ([2],             "#8c564b", "NCEP"),
    "jma":          ([3],             "#e377c2", "JMA"),
    "eccc":         ([5, 4, 3, 6, 2], "#17becf", "ECCC"),
}


# --------------------------------------------------------------------------- CDS fetch
def _client():
    import cdsapi
    return cdsapi.Client()


def _retrieve(dataset: str, centre: str, system: int, year: int, month: int,
              product_type: str, dest: Path) -> bool:
    import time
    if dest.exists():
        return True
    req = {
        "originating_centre": centre, "system": str(system),
        "variable": "sea_surface_temperature", "product_type": product_type,
        "year": str(year), "month": f"{month:02d}",
        "leadtime_month": LEADS, "area": NINO34_AREA, "data_format": "netcdf",
    }
    for attempt in range(1, 9):
        try:
            _client().retrieve(dataset, req, str(dest))
            return dest.exists()
        except Exception as e:                                   # noqa: BLE001
            msg = str(e)
            if "temporarily limited" in msg or "rejected" in msg:    # CDS per-dataset queue cap
                wait = min(90 * attempt, 360)
                print(f"    {centre} sys {system}: rate-limited, wait {wait}s (try {attempt})", flush=True)
                time.sleep(wait); continue
            print(f"    {centre} sys {system} [{product_type}]: {repr(e)[:100]}", flush=True)
            return False                                          # genuine bad request (wrong system) → next candidate
    return False


def _systems() -> dict:
    return json.loads(SYSJSON.read_text()) if SYSJSON.exists() else {}


def fetch_centre(centre: str, init_year: int, init_month: int) -> Path | None:
    """Ensure the member forecast .nc is cached; returns its path. Only the fast forecast
    dataset is pulled — anomalies are drift-anchored to the observed Niño-3.4 (see model_anomaly),
    so no hindcast / postprocessed request (those are slow / often CDS-backlogged)."""
    cands, _, _ = CENTRES[centre]
    known = _systems().get(centre)
    order = ([known] + [s for s in cands if s != known]) if known else cands
    CACHE.mkdir(parents=True, exist_ok=True)
    for sysn in order:
        fc = CACHE / f"{centre}_s{sysn}_{init_year}{init_month:02d}_fc.nc"      # members (absolute)
        if not _retrieve(DS_FC, centre, sysn, init_year, init_month, "monthly_mean", fc):
            continue
        s = _systems(); s[centre] = sysn                              # remember the working system
        SYSJSON.parent.mkdir(parents=True, exist_ok=True); SYSJSON.write_text(json.dumps(s, indent=2))
        print(f"  {centre}: system {sysn} ✓", flush=True)
        return fc
    print(f"  {centre}: no working system in {order}", flush=True)
    return None


# observed Niño-3.4 anomaly (CPC sstoi.indices) — the anchor for each init
def obs_anomaly() -> pd.Series:
    """Monthly observed Niño-3.4 anomaly (°C), indexed by Timestamp, from CPC."""
    import urllib.request, io
    txt = urllib.request.urlopen("https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices", timeout=30).read().decode()
    df = pd.read_csv(io.StringIO(txt), sep=r"\s+")
    df.columns = [c.strip() for c in df.columns]
    idx = pd.to_datetime(dict(year=df["YR"], month=df["MON"], day=1))
    return pd.Series(df["ANOM.3"].values, index=idx)


def obs_anchor(obs: pd.Series, init_year: int, init_month: int) -> float:
    """Latest observed Niño-3.4 anomaly at or before the init month (the plume's starting point)."""
    s = obs[obs.index <= pd.Timestamp(init_year, init_month, 1)]
    return float(s.iloc[-1]) if len(s) else 0.0


# ----------------------------------------------------------------------- computation
def _nino34(ds: xr.Dataset) -> xr.DataArray:
    """Cos-latitude-weighted Niño-3.4 area mean; keeps any number + forecastMonth dims."""
    var = "sst" if "sst" in ds.data_vars else list(ds.data_vars)[0]
    w = np.cos(np.deg2rad(ds["latitude"]))
    return ds[var].weighted(w).mean(("latitude", "longitude"))


def model_anomaly(fc_path: Path, anchor: float, init_year: int, init_month: int) -> pd.DataFrame:
    """Per-member Niño-3.4 anomaly, drift-anchored to the observed state at init time:
    anom_member(t) = member_SST(t) − ensemble_mean_SST(lead-1) + observed_anomaly.
    Subtracting the model's own lead-1 ensemble mean removes its SST bias; anchoring to the
    observed anomaly makes every model's plume start from the real current Niño-3.4."""
    n = _nino34(xr.open_dataset(fc_path)).squeeze(drop=True)         # (number, forecastMonth) absolute SST
    lead1_mean = n.mean("number").isel(forecastMonth=0)             # model's first-month ensemble mean
    anom = (n - lead1_mean + anchor).transpose("number", "forecastMonth")
    months = [pd.Timestamp(init_year, init_month, 1) + pd.DateOffset(months=int(m) - 1)
              for m in anom["forecastMonth"].values]
    return pd.DataFrame(anom.values.T, index=pd.DatetimeIndex(months))   # rows=month, cols=members


def collect(init_year: int, init_month: int, anchor: float) -> dict[str, pd.DataFrame]:
    """Fetch + compute the per-model anomaly DataFrames for one init (cached)."""
    out = {}
    for centre in CENTRES:
        fc = fetch_centre(centre, init_year, init_month)
        if fc:
            out[centre] = model_anomaly(fc, anchor, init_year, init_month)
    return out


# ------------------------------------------------------------------------------ plot
def _super_mean(models: dict[str, pd.DataFrame]) -> pd.Series:
    """Multi-model mean = equal-weight average of each model's ensemble mean."""
    return pd.concat([df.mean(axis=1) for df in models.values()], axis=1).mean(axis=1)


def plot(cur: dict, prev: dict, cur_lbl: str, prev_lbl: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for centre, df in cur.items():
        cands, color, name = CENTRES[centre]
        ax.plot(df.index, df.values, color=color, lw=0.4, alpha=0.16, zorder=2)   # member plumes
        ax.plot(df.index, df.mean(axis=1).values, color=color, lw=1.6, alpha=0.9,
                zorder=4, label=f"{name} ({df.shape[1]})")
    sm = _super_mean(cur)
    ax.plot(sm.index, sm.values, color="black", lw=3.4, zorder=6, label="Super-ensemble mean")
    if prev:
        pm = _super_mean(prev)
        ax.plot(pm.index, pm.values, color="0.35", lw=2.2, ls=(0, (5, 2)), zorder=5,
                label=f"{prev_lbl} init mean")
        # annotate the month-on-month change at the last common target month
        common = sm.index.intersection(pm.index)
        if len(common):
            t = common[-1]; dv = sm[t] - pm[t]
            ax.annotate(f"Δ vs {prev_lbl}: {dv:+.2f} °C", (t, sm[t]), textcoords="offset points",
                        xytext=(6, 10), fontsize=9, fontweight="bold")
    for y, lab in [(0.5, "weak El Niño"), (-0.5, "weak La Niña")]:
        ax.axhline(y, color="0.7", lw=0.6, ls=":")
    ax.axhspan(0.5, 5, color="#d62728", alpha=0.05); ax.axhspan(-5, -0.5, color="#1f77b4", alpha=0.05)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("Niño-3.4 SST anomaly (°C)")
    ax.set_title(f"C3S multi-model Niño-3.4 plumes — {cur_lbl} init\n"
                 "drift-anchored to observed Niño-3.4 (each model's SST bias removed)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(-2.2, 2.6); ax.grid(alpha=0.15)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}  (super-ens latest {sm.iloc[-1]:+.2f} °C, {len(cur)} models)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="init month YYYY-MM (default: latest published)")
    ap.add_argument("--out", default=str(HERE.parent.parent / "assets" / "sst" / "nino_plumes.webp"))
    args = ap.parse_args(argv)
    if args.init:
        iy, im = map(int, args.init.split("-"))
    else:
        t = date.today(); iy, im = (t.year, t.month) if t.day >= 13 else \
            ((t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12))
    py, pm = (iy, im - 1) if im > 1 else (iy - 1, 12)
    cur_lbl = f"{date(iy, im, 1):%b %Y}"; prev_lbl = f"{date(py, pm, 1):%b %Y}"
    obs = obs_anomaly()
    print(f"=== current init {cur_lbl} (anchor {obs_anchor(obs, iy, im):+.2f} °C) ===", flush=True)
    cur = collect(iy, im, obs_anchor(obs, iy, im))
    print(f"=== prior init {prev_lbl} (anchor {obs_anchor(obs, py, pm):+.2f} °C) ===", flush=True)
    prev = collect(py, pm, obs_anchor(obs, py, pm))
    if not cur:
        print("no models for current init"); return 1
    plot(cur, prev, cur_lbl, prev_lbl, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
