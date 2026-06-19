#!/usr/bin/env python3
"""Southern Oscillation Index tracker for the El Niño monitor.

Observed SOI from LongPaddock (BoM standard 1887-1989 base) plus a COMBINED AIFS-ENS +
IFS-ENS forecast (super-ensemble mean + member plume). Both the 30-day running
SOI (bold) and the raw daily Troup SOI (faint) are shown on one plot.

The forecast SOI is Tahiti(17.5°S,149.6°W) − Darwin(12.4°S,130.9°E) MSL from the
ensembles, standardized with the Troup monthly normals (mean pressure-difference
and its SD) recovered by per-month regression from the LongPaddock daily file —
so observed and forecast share one scale — then bias-corrected to the recent
observed level (model gridpoint vs station). Negative SOI ⇒ El Niño-favorable.

    python src/soi_forecast.py --date 20260601 --time 00 --out plots/soi.webp
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf                                    # shared ECMWF download manager

SOI_URL = ("https://data.longpaddock.qld.gov.au/SeasonalClimateOutlook/"
           "SouthernOscillationIndex/SOIDataFiles/DailySOI1887-1989Base.txt")
TAHITI = dict(latitude=-17.5, longitude=210.4)        # 149.6°W
DARWIN = dict(latitude=-12.4, longitude=130.9)
DAILY_STEPS = list(range(24, 361, 24))                # forecast days 1..15
MODELS = [dict(model="aifs-ens", types=["cf", "pf"], label="AIFS-ENS"),
          dict(model="ifs",      types=["pf"],        label="IFS-ENS")]
PAST_DAYS = 75                                        # observed history shown
WIN = 30                                              # 30-day running SOI window


# ── observed SOI + Troup normals ──────────────────────────────────────────────
def fetch_obs(cache: Path) -> pd.DataFrame:
    """LongPaddock daily SOI -> DataFrame indexed by date: Tahiti, Darwin, SOI."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        txt = urllib.request.urlopen(SOI_URL, timeout=60).read().decode()
        cache.write_text(txt)
    except Exception as e:                            # fall back to cached copy
        if not cache.exists():
            raise
        print(f"  SOI fetch failed ({repr(e)[:60]}); using cached {cache.name}")
        txt = cache.read_text()
    df = pd.read_csv(io.StringIO(txt), sep=r"\s+")
    df.columns = [c.strip() for c in df.columns]
    df["date"] = (pd.to_datetime(df["Year"].astype(int).astype(str) + "-01-01")
                  + pd.to_timedelta(df["Day"].astype(int) - 1, unit="D"))
    df = df.set_index("date")[["Tahiti", "Darwin", "SOI"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]        # drop any repeated dates
    for c in ("Tahiti", "Darwin"):                    # mask -999.9 missing sentinels
        df.loc[df[c] < 900, c] = np.nan
    df.loc[df["Tahiti"].isna() | df["Darwin"].isna(), "SOI"] = np.nan
    return df


def troup_normals(obs: pd.DataFrame) -> dict:
    """Recover per-calendar-month (mean diff, SD) from SOI = 10·(diff−m)/sd.
    Regress diff = Tahiti−Darwin on SOI within each month: slope=sd/10, intc=mean."""
    d = pd.DataFrame({"diff": obs["Tahiti"] - obs["Darwin"],
                      "soi": obs["SOI"], "m": obs.index.month}).dropna()
    out = {}
    for m, g in d.groupby("m"):
        slope, intc = np.polyfit(g["soi"].values, g["diff"].values, 1)
        out[m] = (float(intc), float(slope * 10.0))   # (mean diff, SD of diff)
    return out


def soi_of(diff_hpa: np.ndarray, months: np.ndarray, normals: dict) -> np.ndarray:
    """Troup SOI from a Tahiti−Darwin pressure difference (hPa) per month."""
    dmean = np.array([normals[m][0] for m in months])
    sd = np.array([normals[m][1] for m in months])
    return 10.0 * (diff_hpa - dmean) / sd


# ── forecast: ensemble Tahiti−Darwin MSL ──────────────────────────────────────
def download_msl(cfg: dict, date: str, time: str, out_dir: Path = None) -> dict:
    """Ensure msl (forecast days) for this model via the shared store; AIFS cf+pf is
    deduped with the torque budget. Returns {typ: cache_path}."""
    cyc = ecmwf.Cycle(date, time)
    return {typ: ecmwf.sfc_path(cyc, cfg["model"], typ, "msl") for typ in cfg["types"]}


def member_diff(paths: dict) -> xr.DataArray:
    """(member, step) Tahiti−Darwin MSL in hPa, members stacked across cf/pf."""
    parts = []
    for p in paths.values():
        ds = xr.open_dataset(p, engine="cfgrib",
                             backend_kwargs={"filter_by_keys": {"shortName": "msl"}, "indexpath": ""},
                             chunks={"number": 1})       # msl out of the batched surface file
        msl = ds[[v for v in ds.data_vars][0]]
        if float(msl.longitude.min()) < 0:
            msl = msl.assign_coords(longitude=msl.longitude % 360).sortby("longitude")
        d = (msl.sel(**TAHITI, method="nearest")
             - msl.sel(**DARWIN, method="nearest")) / 100.0      # Pa -> hPa
        if "number" not in d.dims:
            d = d.expand_dims("number")
        parts.append(d.compute())
    out = xr.concat(parts, dim="number")
    return out.assign_coords(number=np.arange(out.sizes["number"]))


# ── plot ──────────────────────────────────────────────────────────────────────
def plot(obs: pd.DataFrame, normals: dict, diff: xr.DataArray,
         init: pd.Timestamp, out: Path):
    steps_h = (diff.step / np.timedelta64(1, "h")).values.astype(int)
    init_d = init.normalize()                                    # calendar day (obs is daily)
    fdates = pd.to_datetime([(init + pd.Timedelta(hours=int(h))).normalize() for h in steps_h])
    fc = np.vstack([soi_of(diff.isel(number=j).values, fdates.month.values, normals)
                    for j in range(diff.sizes["number"])])      # (member, day)

    # bias-correct forecast daily SOI to the recent observed level (gridpoint↔station).
    # Anchor to the 30-day RUNNING-MEAN observed SOI (the bold black line we plot), not a
    # raw 10-day median: the combined ensemble's raw SOI is essentially flat and heavily
    # biased, so this constant offset sets the whole forecast LEVEL. A 10-day median gets
    # yanked ~10 pts by a few-day daily swing (e.g. a transient +SOI spike), pushing the
    # forecast spuriously positive; the 30-day mean ties it to the smoothed observed state.
    obs_soi = obs["SOI"]
    recent = obs_soi.rolling(WIN, min_periods=WIN - 5).mean().loc[:init_d].iloc[-1]
    # nan-robust: some ensemble members can have missing MSL at early steps (seen in
    # AIFS-ENS open data), which would otherwise turn np.median → NaN and blank the run.
    bias = np.nanmedian(fc[:, :5]) - recent
    fc -= bias

    # 30-day running SOI per member: observed tail (shared) + member forecast
    full = pd.date_range(init_d - pd.Timedelta(days=WIN - 1), fdates[-1], freq="D")
    obs_d = obs_soi.reindex(full)
    # min_periods < WIN tolerates the 1–few-day gap between the last observation
    # and the first forecast day (init+24h) — otherwise that hole blanks every
    # forecast window. A 30-day mean from ≥25 days is fine.
    run_fc = []
    for j in range(fc.shape[0]):
        s = obs_d.copy()
        s.loc[fdates] = fc[j]
        run_fc.append(s.rolling(WIN, min_periods=WIN - 5).mean())
    run_fc = pd.DataFrame(run_fc).T                              # index=full, cols=member
    run_obs = obs_soi.rolling(WIN, min_periods=WIN - 5).mean()

    p0 = init_d - pd.Timedelta(days=PAST_DAYS)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    # El Niño / La Niña reference bands
    ax.axhspan(-7, 7, color="0.5", alpha=0.06)
    for y in (7, -7):
        ax.axhline(y, color="0.6", lw=0.7, ls="--")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.text(p0, 9, "La Niña", fontsize=8, color="#2c4a72")
    ax.text(p0, -11, "El Niño", fontsize=8, color="#9a2c2c")

    # observed: faint daily + bold 30-day
    od = obs_soi.loc[p0:init_d]
    ax.plot(od.index, od.values, color="#5577a6", lw=1.5, alpha=0.95, label="Observed daily SOI")
    ro = run_obs.loc[p0:init_d]
    ax.plot(ro.index, ro.values, color="k", lw=2.4, label="Observed 30-day SOI")

    # forecast: member plume (30-day running) + ensemble mean; daily 10–90% band
    fwin = run_fc.loc[init_d:]
    ax.plot(fwin.index, fwin.values, color="#1f77b4", lw=0.9, alpha=0.18)   # member plume — legible
    ax.plot(fwin.index, fwin.mean(axis=1).values, color="#d62728", lw=2.8,
            label="Forecast 30-day SOI (ens. mean)")
    lo, hi = np.nanpercentile(fc, [10, 90], axis=0)
    ax.fill_between(fdates, lo, hi, color="#d62728", alpha=0.12,
                    label="Forecast daily SOI (10–90%)")
    ax.plot(fdates, np.nanmean(fc, axis=0), color="#d62728", lw=1.6, ls=":", alpha=0.9)
    ax.axvline(init_d, color="0.4", lw=0.8, ls=":")

    ax.set_xlim(p0, fdates[-1])
    ax.set_ylabel("Southern Oscillation Index")
    ax.set_title(f"Southern Oscillation Index — observed + AIFS-ENS/IFS-ENS forecast "
                 f"(init {init:%Y-%m-%d %HZ})\nTroup SOI · negative ⇒ El Niño-favorable · "
                 f"{diff.sizes['number']}-member combined ensemble",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.92)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    last = run_fc.iloc[-1].mean()
    print(f"saved {out} (obs SOI {obs_soi.dropna().iloc[-1]:+.1f} {obs_soi.dropna().index[-1]:%b %d}; "
          f"forecast 30-day ens-mean day15 {last:+.1f}; bias {bias:+.1f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/msl")
    ap.add_argument("--out", default="plots/soi_forecast.webp")
    args = ap.parse_args()

    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    obs = fetch_obs(Path("data/soi/DailySOI.txt"))
    normals = troup_normals(obs)

    diffs, included = [], []
    for cfg in MODELS:
        try:
            diffs.append(member_diff(download_msl(cfg, args.date, args.time,
                                                  Path(args.data_dir))))
            included.append(cfg["model"])
        except Exception as e:
            print(f"  {cfg['label']}: skipped ({repr(e)[:80]})", flush=True)
    if not diffs:
        raise SystemExit("no ensemble MSL available for the SOI forecast")
    diff = xr.concat(diffs, dim="number")
    diff = diff.assign_coords(number=np.arange(diff.sizes["number"]))
    plot(obs, normals, diff, init, Path(args.out))
    # Record any skipped model (e.g. IFS-ENS not yet on the portal) so the pipeline knows the
    # render is incomplete and re-renders once it lands. One token/line ("aifs-ens"/"ifs").
    missing = [c["model"] for c in MODELS if c["model"] not in included]
    flag = Path(str(args.out) + ".missing")
    if missing:
        flag.write_text("\n".join(missing) + "\n")
    else:
        flag.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
