#!/usr/bin/env python3
"""SFS beta MJO — 31-member RMM trajectories through the site's own machinery.

The exact operational wind-only RMM path used for the AIFS product, with
NO model-specific adjustment: 15°S–15°N cos-weighted band-mean U850/U200
→ day-of-year climatology (climatology.nc) → minus the trailing-120-day
analysis-mean maps (wind_map120.nc, the WH04 low-frequency/ENSO filter,
held fixed across lead) → divide by std_u850/std_u200 → project onto the
wind portions of the reference EOFs → divide by the recalibrated per-mode
pc_wind_std. Every constant comes from the same committed reference files
as the AIFS RMM, so the two products are directly comparable.

SFS gives 31 members every OTHER day out to day 46 — a genuine
subseasonal MJO forecast. At long leads the model's own tropical wind
drift projects onto the index (it is visible as slowly growing amplitude);
per the site owner's call this is shown raw rather than hindcast-adjusted.

Output: assets/sfs/mjo_rmm.webp (WH04 phase diagram: observed trail +
members + ensemble mean) and assets/sfs/data/sfs_mjo.json.

    python scripts/sfs/sfs_mjo.py [--issue 202608]
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MREF = REPO / "scripts" / "mjo" / "data" / "reference"
OUTPNG = REPO / "assets" / "sfs" / "mjo_rmm.webp"
OUTJSON = REPO / "assets" / "sfs" / "data" / "sfs_mjo.json"
BASE = "https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/beta1"


def _open(url):
    import fsspec
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True,
                        decode_timedelta=True)


def band_mean(u, lat):
    """15S-15N cos-weighted band mean over the lat axis: (..., lat, lon) -> (..., lon)."""
    m = (lat >= -15) & (lat <= 15)
    w = np.cos(np.deg2rad(lat[m]))
    return (u[..., m, :] * w[:, None]).sum(axis=-2) / w.sum()


def fetch_bom_rmm(days_back=45):
    """Official BoM RMM via the IRI Data Library mirror (bom.gov.au blocks
    direct downloads). Returns (dates, rmm1, rmm2) or None on any failure —
    the caller falls back to the site's own AIFS-analysis history so the
    monthly cron never dies on a mirror outage."""
    import urllib.request
    from urllib.parse import quote
    t1 = datetime.now(timezone.utc)
    t0 = t1 - pd.Timedelta(days=days_back)
    d0, d1 = t0.strftime("%d %b %Y"), t1.strftime("%d %b %Y")
    try:
        out = {}
        for var in ("RMM1", "RMM2"):
            url = (f"http://iridl.ldeo.columbia.edu/SOURCES/.BoM/.MJO/.RMM/"
                   f".{var}/T/{quote(f'(0000 {d0})')}/{quote(f'(0000 {d1})')}/"
                   "RANGE/gridtable.tsv")
            with urllib.request.urlopen(url, timeout=60) as r:
                rows = [ln.split() for ln in
                        r.read().decode().splitlines()[2:] if ln.strip()]
            jd = np.array([float(a) for a, _ in rows])
            out[var] = (jd, np.array([float(b) for _, b in rows]))
        jd1, v1 = out["RMM1"]
        jd2, v2 = out["RMM2"]
        if len(jd1) < 10 or len(jd1) != len(jd2) or not np.allclose(jd1, jd2):
            return None
        dates = pd.to_datetime(jd1 - 2440587.5, unit="D").normalize()
        return dates, v1, v2
    except Exception as e:
        print(f"BoM RMM fetch failed ({e}); falling back to AIFS analysis",
              flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=datetime.now(timezone.utc).strftime("%Y%m"))
    args = ap.parse_args()
    issue = args.issue
    t0 = pd.Timestamp(f"{issue[:4]}-{issue[4:6]}-01")

    clim = xr.open_dataset(MREF / "climatology.nc")
    eofs = xr.open_dataset(MREF / "eofs.nc")
    m120 = xr.open_dataset(MREF / "wind_map120.nc")
    elon = eofs.longitude.values
    e1 = np.concatenate([eofs["eof_u850"].sel(mode=1).values,
                         eofs["eof_u200"].sel(mode=1).values])
    e2 = np.concatenate([eofs["eof_u850"].sel(mode=2).values,
                         eofs["eof_u200"].sel(mode=2).values])
    s1 = float(eofs["pc_wind_std"].sel(mode=1))
    s2 = float(eofs["pc_wind_std"].sel(mode=2))

    ds = _open(f"{BASE}/forecast/{issue}/atm_daily.zarr")
    band = ds.where((ds.lat >= -16) & (ds.lat <= 16), drop=True)
    lat, lon = band.lat.values, band.lon.values
    u850 = band.UGRD_850mb.values                     # (31, 47, ~33, 360)
    u200 = band.UGRD_200mb.values
    lead_days = pd.to_timedelta(ds.lead.values).days.values
    sel = np.where(np.isfinite(u850[0, :, 0, 180]))[0]
    valid = [t0 + pd.Timedelta(days=int(d)) for d in lead_days[sel]]

    def to_eof(x):
        """(..., 360 lons at 1°) -> EOF 2.5° grid via linear interp (cyclic)."""
        xi = np.concatenate([x, x[..., :1]], axis=-1)
        loni = np.concatenate([lon, [lon[0] + 360]])
        return np.stack([np.interp(elon, loni, xi[idx])
                         for idx in np.ndindex(x.shape[:-1])]
                        ).reshape(*x.shape[:-1], len(elon))

    b850 = to_eof(band_mean(u850[:, sel], lat))       # (31, n, 144)
    b200 = to_eof(band_mean(u200[:, sel], lat))
    doys = np.array([min(v.dayofyear, 366) for v in valid])
    a850 = (b850 - clim["clim_u850"].sel(dayofyear=doys).values[None]
            - m120["u850"].values[None, None]) / clim.attrs["std_u850"]
    a200 = (b200 - clim["clim_u200"].sel(dayofyear=doys).values[None]
            - m120["u200"].values[None, None]) / clim.attrs["std_u200"]
    comb = np.concatenate([a850, a200], axis=-1)      # (31, n, 288)
    rmm1 = comb @ e1 / s1
    rmm2 = comb @ e2 / s2
    print(f"day-0 ens-mean RMM: ({rmm1[:, 0].mean():+.2f}, "
          f"{rmm2[:, 0].mean():+.2f}) amp "
          f"{np.hypot(rmm1, rmm2)[:, 0].mean():.2f}")

    # observed trail: official BoM RMM (IRI mirror), AIFS analysis fallback
    bom = fetch_bom_rmm()
    if bom is not None:
        obs_dates, o1, o2 = bom
        obs_label = "observed (BoM RMM)"
    else:
        obs = xr.open_dataset(MREF / "obs_history.nc")
        otr = obs.isel(time=slice(-40, None))
        obs_dates = pd.to_datetime(otr.time.values)
        o1, o2 = otr.rmm1.values, otr.rmm2.values
        obs_label = "observed (AIFS analysis)"

    # ── WH04 phase diagram — same wheel as the AIFS-ENS product ─────────────
    sys.path.insert(0, str(REPO / "scripts" / "mjo" / "src"))
    from plot import draw_phase_wheel
    fig, ax = plt.subplots(figsize=(9.6, 9.6))
    draw_phase_wheel(ax)
    for m in range(rmm1.shape[0]):
        ax.plot(rmm1[m], rmm2[m], color="#2e97ad", lw=0.7, alpha=0.35)
    ax.plot(o1, o2, color="0.15", lw=2.2, marker="o", ms=3, label=obs_label)
    ax.annotate(f"{obs_dates[-1]:%b %d}", (o1[-1], o2[-1]), fontsize=8,
                color="0.15", xytext=(6, -9), textcoords="offset points")
    mm1, mm2 = rmm1.mean(axis=0), rmm2.mean(axis=0)
    ax.plot(mm1, mm2, color="#c62828", lw=3, marker="o", ms=4.5,
            label="SFS ensemble mean")
    for i in range(0, len(valid), 4):
        ax.annotate(f"{valid[i]:%b %d}", (mm1[i], mm2[i]), fontsize=8,
                    color="#7a1d1d", xytext=(5, 5), textcoords="offset points")
    ax.set_title(f"SFS beta — MJO (wind-only RMM), 31 members to day "
                 f"{int(lead_days[sel][-1])} · issue {t0:%b %Y}\n"
                 "same machinery as the AIFS-ENS product · raw projection, "
                 "no model adjustment", fontsize=11, fontweight="bold",
                 loc="left")
    ax.legend(fontsize=9, loc="lower left")
    OUTPNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPNG, dpi=140, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    OUTJSON.parent.mkdir(parents=True, exist_ok=True)
    OUTJSON.write_text(json.dumps({
        "issue": f"{issue[:4]}-{issue[4:6]}",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "days": [f"{v:%Y-%m-%d}" for v in valid],
        "rmm1": np.round(rmm1, 3).tolist(),
        "rmm2": np.round(rmm2, 3).tolist(),
        "obs": {"source": obs_label,
                "days": [f"{d:%Y-%m-%d}" for d in obs_dates],
                "rmm1": np.round(o1, 3).tolist(),
                "rmm2": np.round(o2, 3).tolist()},
        "filter_window_end": m120.attrs.get("window_end", "?"),
    }, separators=(",", ":")))
    print(f"wrote {OUTPNG.relative_to(REPO)} + sfs_mjo.json")
    print("ens-mean amp by step:",
          np.round(np.hypot(mm1, mm2), 2).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
