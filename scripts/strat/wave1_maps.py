#!/usr/bin/env python3
"""
Planetary wave-1 in geopotential height: 100 and 500 hPa, both hemispheres.

Four polar panels — 100 hPa NH / SH on top, 500 hPa NH / SH below — showing the
zonal wavenumber-1 component of geopotential height (shaded) with the FULL
height field contoured over it.

Why a map and not an amplitude series. Wave-1 amplitude answers "how big", which
is the least interesting half of the question. What actually matters for the
vortex is WHERE the ridge sits and whether it is vertically coherent: a wave-1
that leans westward with height is actively driving the vortex, while one that
sits over the same longitude at both levels is not doing much. An amplitude
curve cannot show either. Two levels on one figure, with phase visible directly,
can.

  100 hPa  the lower stratosphere, where wave driving reaches the vortex.
  500 hPa  the mid-troposphere source region. Comparing the two shows whether a
           tropospheric ridge is actually connected upward.

The wave-1 field is extracted per latitude by an FFT in longitude, keeping only
k=1 and transforming back. That component is an anomaly by construction — the
zonal mean is k=0 and is discarded — so no separate climatology is needed, and
the figure is meaningful on any single cycle with nothing to go stale.

Shading is symmetric about zero with a per-level scale. Wave amplitude grows
with height, so 100 and 500 hPa cannot share one; NH and SH DO share a scale at
each level, so the hemispheres stay comparable.

Usage:
  python wave1_maps.py --date 20260829 --time 12 --out assets/sst/wave1_maps.webp
  python wave1_maps.py --step 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
import store as ecmwf

PC = ccrs.PlateCarree()
G = 9.80665
LEVELS = (100, 500)          # rendered top row first
LAT_EDGE = 20.0

# Symmetric per-level scales in geopotential metres, fixed so a colour means the
# same wave amplitude every day. 100 hPa runs roughly twice 500 hPa.
SCALE = {100: 400.0, 500: 200.0}
# Contour interval for the FULL height field drawn over the shading.
FULL_CI = {100: 120.0, 500: 60.0}
CMAP = "RdBu_r"


def open_field(path, short: str, lev: int, step_h: int):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
        filter_by_keys={"shortName": short, "level": lev}, indexpath=""))
    da = ds[short] if short in ds else ds[list(ds.data_vars)[0]]
    if "step" in da.dims:
        steps = (da.step.values / np.timedelta64(1, "h")).astype(int)
        if step_h not in set(steps):
            raise SystemExit(f"step {step_h}h not in z@{lev} "
                             f"(have {steps.min()}..{steps.max()})")
        da = da.isel(step=int(np.where(steps == step_h)[0][0]))
    return da


def fetch(date: str, time: str, step_h: int):
    """Geopotential at both levels, control only.

    z is not in the store's bulk list - it was dropped 2026-07-24 as ~0.5 GB a
    cycle of dead weight when nothing consumed it. Control at two levels is a
    small fraction of that.
    """
    cyc = ecmwf.Cycle(date, time)
    path = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "z", "pl",
                                        LEVELS, tuple(ecmwf.STEPS)))
    return {lev: open_field(path, "z", lev, step_h) for lev in LEVELS}


def wave1(field2d: np.ndarray) -> np.ndarray:
    """Zonal wavenumber-1 component, per latitude row.

    rfft along longitude, keep only k=1, transform back. Dropping k=0 removes
    the zonal mean, which is what makes the result an anomaly without needing a
    climatology; dropping k>=2 removes the shorter waves that would otherwise
    clutter the phase we are trying to read.
    """
    spec = np.fft.rfft(field2d, axis=1)
    keep = np.zeros_like(spec)
    keep[:, 1] = spec[:, 1]
    return np.fft.irfft(keep, n=field2d.shape[1], axis=1)


def circular_boundary(ax):
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts * 0.5 + 0.5), transform=ax.transAxes)


def panel(ax, z, lev, hemi):
    lat = z.latitude.values
    lon = z.longitude.values
    north = hemi == "NH"
    sel = (lat >= LAT_EDGE) if north else (lat <= -LAT_EDGE)
    la = lat[sel]
    Z = z.values[sel, :] / G                      # geopotential -> gpm
    W1 = wave1(Z)

    ax.set_extent([-180, 180, LAT_EDGE if north else -90,
                   90 if north else -LAT_EDGE], crs=PC)
    circular_boundary(ax)

    s = SCALE[lev]
    lv = np.linspace(-s, s, 17)
    cf = ax.contourf(lon, la, W1, levels=lv, cmap=CMAP, extend="both",
                     transform=PC, zorder=1)

    # Full height field over the top: the shading says where wave-1 is, these
    # say what the actual flow looks like, and the reader can see at a glance how
    # much of the pattern wave-1 accounts for.
    ci = FULL_CI[lev]
    lo, hi = np.floor(Z.min() / ci) * ci, np.ceil(Z.max() / ci) * ci
    ax.contour(lon, la, Z, levels=np.arange(lo, hi + ci, ci), colors="#3a3a3a",
               linewidths=0.5, transform=PC, zorder=3)

    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#555",
                   linewidth=0.5, zorder=4)
    gl = ax.gridlines(linewidth=0.3, color="#8a8a8a", alpha=0.45, zorder=5)
    gl.ylocator = plt.FixedLocator([20, 40, 60, 80] if north else [-80, -60, -40, -20])
    lat60 = 60.0 if north else -60.0
    ax.plot(np.linspace(-180, 180, 361), np.full(361, lat60), transform=PC,
            color="#444", linewidth=0.85, linestyle=(0, (5, 3)), zorder=5)

    # Amplitude quoted at 60 deg, the latitude the vortex diagnostics use.
    j = int(np.argmin(np.abs(la - lat60)))
    amp = float(np.abs(np.fft.rfft(Z[j])[1]) * 2.0 / Z.shape[1])
    ax.set_title(f"{hemi} · {lev} hPa    wave-1 amp at 60° = {amp:.0f} gpm",
                 fontsize=10.5, pad=6)
    return cf


def render(fields, date, time, step_h, out_path: Path):
    fig = plt.figure(figsize=(11.4, 11.0), dpi=125)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045],
                          left=0.02, right=0.93, top=0.90, bottom=0.055,
                          wspace=0.06, hspace=0.14)
    for r, lev in enumerate(LEVELS):
        cf = None
        for c, hemi in enumerate(("NH", "SH")):
            proj = (ccrs.NorthPolarStereo(central_longitude=0) if hemi == "NH"
                    else ccrs.SouthPolarStereo(central_longitude=0))
            ax = fig.add_subplot(gs[r, c], projection=proj)
            cf = panel(ax, fields[lev], lev, hemi)
        cax = fig.add_subplot(gs[r, 2])
        cb = fig.colorbar(cf, cax=cax, extend="both")
        cb.set_label(f"{lev} hPa wave-1 height anomaly (gpm)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    valid = f"F{step_h:03d}" if step_h else "analysis"
    fig.suptitle("Planetary wave-1 in geopotential height — 100 and 500 hPa",
                 fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.943,
             f"ECMWF AIFS-ENS control · {date[:4]}-{date[4:6]}-{date[6:]} {time}Z "
             f"{valid} · shaded: zonal wavenumber-1 component · "
             f"contours: full height field · dashed ring: 60°",
             fontsize=9, color="#555", ha="left")
    # Two short lines, not one long one: at this figure width a single line of
    # this text runs off the right edge (it did on the first render).
    fig.text(0.02, 0.030,
             "Wave-1 is an anomaly by construction — the zonal mean is k = 0 and is "
             "discarded — so no climatology is involved.",
             fontsize=8, color="#777", ha="left")
    fig.text(0.02, 0.011,
             "Compare the phase between levels: a ridge that leans westward with "
             "height is actively driving the vortex.",
             fontsize=8, color="#777", ha="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125, facecolor="white",
                pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--out", default="assets/sst/wave1_maps.webp")
    a = ap.parse_args(argv)
    fields = fetch(a.date, a.time, a.step)
    print(f"  wrote {render(fields, a.date, a.time, a.step, Path(a.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
