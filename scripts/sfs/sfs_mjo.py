#!/usr/bin/env python3
"""SFS beta MJO — 31-member RMM trajectories through the site's own machinery.

The exact operational wind-only RMM path used for the AIFS product:
15°S–15°N cos-weighted band-mean U850/U200 → day-of-year climatology
(climatology.nc) → minus the trailing-120-day analysis-mean maps
(wind_map120.nc, the WH04 low-frequency/ENSO filter, held fixed across
lead) → divide by std_u850/std_u200 → project onto the wind portions of
the reference EOFs → divide by the recalibrated per-mode pc_wind_std.
Every constant comes from the same committed reference files as the AIFS
RMM, so the two products are directly comparable.

SFS gives 31 members every OTHER day out to day 46 — a genuine
subseasonal MJO forecast. Output: assets/sfs/mjo_rmm.webp (WH04 phase
diagram: observed trail + members + ensemble mean) and
assets/sfs/data/sfs_mjo.json.

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


def mjo_prate_clim(month, psel, elon, to_eof):
    """Per-lead reforecast PRATE band-mean statistics on the EOF grid:
    mean (climatology + drift in one, since there is no external precip
    climatology) and σ across the 30 years x 11 members. Basis for the
    pseudo-OLR channel: tropical precip and OLR anticorrelate tightly, so
    −(standardized precip anomaly) stands in for olr_anom/std_olr in the
    full three-channel WH04 projection. Cached per init month."""
    f = HERE / "data" / f"mjo_prate_clim_{month:02d}.npz"
    if f.exists():
        z = np.load(f)
        return z["mu"], z["sd"]
    ds = _open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
    ds = ds.sel(init=slice("1991", "2020"))
    band = ds.where((ds.lat >= -16) & (ds.lat <= 16), drop=True)
    lat = band.lat.values
    n_init = band.sizes["init"]
    s1_ = np.zeros((len(psel), len(elon)))
    s2_ = np.zeros_like(s1_)
    n = 0
    for yi in range(n_init):
        p = band.PRATE_surface.isel(init=yi).values[:, psel]  # (11, n, lat, lon)
        pb = to_eof(band_mean(p, lat))                        # (11, n, 144)
        s1_ += pb.sum(axis=0)
        s2_ += (pb ** 2).sum(axis=0)
        n += pb.shape[0]
        print(f"prate clim: init {yi + 1}/{n_init}", flush=True)
    mu = s1_ / n
    sd = np.sqrt(np.maximum(s2_ / n - mu ** 2, 1e-20))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, mu=mu, sd=sd)
    print(f"prate clim {month:02d}: cached ({n} samples/lead)", flush=True)
    return mu, sd


def mjo_drift(month, sel, doys, clim, elon, to_eof):
    """Lead-dependent model drift of band-mean U850/U200 vs the reference
    day-of-year climatology, from the 1991-2020 x 11-member reforecast.
    Averaging (model - clim) over 30 years removes MJO/ENSO variability;
    what survives is the model's own bias growth with lead — without
    removing it, every member 'amplifies' along the drift direction and the
    ensemble-mean RMM ramps artificially (it reached amp ~2.9 by day 40
    uncorrected for the Aug 2026 issue). Cached per init month."""
    f = HERE / "data" / f"mjo_drift_{month:02d}.npz"
    if f.exists():
        z = np.load(f)
        return z["d850"], z["d200"]
    ds = _open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
    ds = ds.sel(init=slice("1991", "2020"))
    band = ds.where((ds.lat >= -16) & (ds.lat <= 16), drop=True)
    lat = band.lat.values
    n_init = band.sizes["init"]
    c850 = clim["clim_u850"].sel(dayofyear=doys).values   # (n, 144)
    c200 = clim["clim_u200"].sel(dayofyear=doys).values
    d850 = np.zeros((len(sel), len(elon)))
    d200 = np.zeros_like(d850)
    for yi in range(n_init):
        u8 = band.UGRD_850mb.isel(init=yi).values[:, sel]  # (11, n, lat, lon)
        u2 = band.UGRD_200mb.isel(init=yi).values[:, sel]
        d850 += (to_eof(band_mean(u8, lat)) - c850).mean(axis=0)
        d200 += (to_eof(band_mean(u2, lat)) - c200).mean(axis=0)
        print(f"mjo drift: init {yi + 1}/{n_init}", flush=True)
    d850 /= n_init
    d200 /= n_init
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, d850=d850, d200=d200)
    print(f"mjo drift {month:02d}: cached "
          f"(|d850| max {np.abs(d850).max():.2f}, |d200| max {np.abs(d200).max():.2f} m/s)",
          flush=True)
    return d850, d200


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
    band = ds.sel(lat=slice(None)).where(
        (ds.lat >= -16) & (ds.lat <= 16), drop=True)
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
    d850, d200 = mjo_drift(t0.month, sel, doys, clim, elon, to_eof)
    # Anchor at initialization: subtract only the drift GROWTH d(lead)-d(0).
    # The reforecast's day-0 term is its initial-state bias vs the ERA5-era
    # climatology (older-analysis inits) — the NRT analysis doesn't share it,
    # and subtracting it pushed day 0 away from the observed RMM.
    d850 = d850 - d850[0]
    d200 = d200 - d200[0]
    a850 = (b850 - clim["clim_u850"].sel(dayofyear=doys).values[None]
            - m120["u850"].values[None, None] - d850[None]) / clim.attrs["std_u850"]
    a200 = (b200 - clim["clim_u200"].sel(dayofyear=doys).values[None]
            - m120["u200"].values[None, None] - d200[None]) / clim.attrs["std_u200"]
    comb = np.concatenate([a850, a200], axis=-1)      # (31, n, 288)
    rmm1 = comb @ e1 / s1                             # wind-only (AIFS-style)
    rmm2 = comb @ e2 / s2
    amp = np.hypot(rmm1, rmm2)
    print(f"day-0 ens-mean RMM (wind-only): ({rmm1[:, 0].mean():+.2f}, "
          f"{rmm2[:, 0].mean():+.2f}) amp {amp[:, 0].mean():.2f}")

    # ── pseudo-OLR channel from PRATE → full 3-channel WH04 projection ──────
    # PRATE is a flux field living on the ODD leads (parity interleave), so
    # standardize on its own lead grid, then interpolate the standardized
    # anomaly to the even wind days. Sign flip: more rain = deeper convection
    # = lower OLR. The reforecast per-lead mean is climatology + drift in one;
    # the WH04 120-day low-frequency filter has no precip counterpart here,
    # so slow ENSO-ish precip signal can leak into this channel at long leads.
    pcol = band.PRATE_surface.isel(member=0, lat=0, lon=180).values
    psel = np.where(np.isfinite(pcol))[0]
    pb = to_eof(band_mean(band.PRATE_surface.values[:, psel], lat))
    pmu, psd = mjo_prate_clim(t0.month, psel, elon, to_eof)
    # scalar standardization per lead (RMS σ over longitude), like WH04's
    # single std_olr — per-longitude division would erase the spatial
    # variance structure and blow up noise over dry longitudes
    phat = (pb - pmu[None]) / np.sqrt((psd ** 2).mean(axis=1))[None, :, None]
    pdays = lead_days[psel].astype(float)
    wdays = lead_days[sel].astype(float)
    olr_hat = np.empty((phat.shape[0], len(wdays), phat.shape[2]))
    for m in range(phat.shape[0]):
        for j in range(phat.shape[2]):
            olr_hat[m, :, j] = np.interp(wdays, pdays, phat[m, :, j])
    olr_hat = -olr_hat                                # pseudo olr_anom/std_olr
    eo1 = eofs["eof_olr"].sel(mode=1).values
    eo2 = eofs["eof_olr"].sel(mode=2).values
    f1 = float(eofs["pc_std"].sel(mode=1))
    f2 = float(eofs["pc_std"].sel(mode=2))
    rmm1f = (olr_hat @ eo1 + comb @ e1) / f1          # full WH04 normalization
    rmm2f = (olr_hat @ eo2 + comb @ e2) / f2
    print(f"day-0 ens-mean RMM (full, precip proxy): ({rmm1f[:, 0].mean():+.2f}, "
          f"{rmm2f[:, 0].mean():+.2f}) amp "
          f"{np.hypot(rmm1f, rmm2f)[:, 0].mean():.2f}")

    # observed trail for context (already RMM-scaled)
    obs = xr.open_dataset(MREF / "obs_history.nc")
    otr = obs.isel(time=slice(-40, None))

    # ── WH04 phase diagram — same wheel as the AIFS-ENS product ─────────────
    sys.path.insert(0, str(REPO / "scripts" / "mjo" / "src"))
    from plot import draw_phase_wheel
    fig, ax = plt.subplots(figsize=(9.6, 9.6))
    draw_phase_wheel(ax)
    for m in range(rmm1.shape[0]):
        ax.plot(rmm1[m], rmm2[m], color="#2e97ad", lw=0.7, alpha=0.35)
    ax.plot(otr.rmm1, otr.rmm2, color="0.15", lw=2.2, marker="o", ms=3,
            label="observed (AIFS analysis)")
    # precip-as-OLR full projection verified WORSE vs official BOM RMM
    # (phase error +46 to +56 deg vs +4 deg wind-only, Aug 2026 issue) —
    # shown as an experimental reference only, wind-only stays primary
    ax.plot(rmm1f.mean(axis=0), rmm2f.mean(axis=0), color="0.4", lw=1.6,
            ls="--", label="ens mean + precip pseudo-OLR (exp.)")
    mm1, mm2 = rmm1.mean(axis=0), rmm2.mean(axis=0)
    ax.plot(mm1, mm2, color="#c62828", lw=3, marker="o", ms=4.5,
            label="SFS ensemble mean (wind-only)")
    for i in range(0, len(valid), 4):
        ax.annotate(f"{valid[i]:%b %d}", (mm1[i], mm2[i]), fontsize=8,
                    color="#7a1d1d", xytext=(5, 5), textcoords="offset points")
    ax.set_title(f"SFS beta — MJO (wind-only RMM), 31 members to day "
                 f"{int(lead_days[sel][-1])} · issue {t0:%b %Y}\n"
                 "same machinery as the AIFS-ENS product · drift-corrected "
                 "vs 1991–2020 hindcast · dashed: experimental precip pseudo-OLR",
                 fontsize=11, fontweight="bold", loc="left")
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
        "rmm1_full_exp": np.round(rmm1f, 3).tolist(),
        "rmm2_full_exp": np.round(rmm2f, 3).tolist(),
        "filter_window_end": m120.attrs.get("window_end", "?"),
    }, separators=(",", ":")))
    print(f"wrote {OUTPNG.relative_to(REPO)} + sfs_mjo.json")
    print("ens-mean amp by step:", np.round(np.hypot(mm1, mm2), 2).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
