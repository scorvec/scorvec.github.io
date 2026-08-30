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
import pandas as pd
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
# Perturbed members averaged for the ensemble mean. 25 rather than all 50: at
# k=1 the mean converges long before the member count runs out, and the pull is
# one z field per member per level per step, so 50 would double a ~0.5 GB fetch
# for a change too small to see on the map.
MEMBERS = 25

# Symmetric per-level scales in geopotential metres, fixed so a colour means the
# same wave amplitude every day. 100 hPa runs roughly twice 500 hPa.
SCALE = {100: 400.0, 500: 200.0}
# Amplitudes inside the innermost pair are left WHITE rather than tinted. A pale
# wash over a whole summer hemisphere reads as "something is happening here"
# when nothing is; blanking the weak values makes the hemisphere that is
# actually active obvious at a glance, and stops the eye chasing noise in the
# other one.
#
# Written out rather than computed with linspace so the colourbar carries round
# numbers a reader can actually use - an evenly divided range gave ticks like
# 116.7 and 343.3.
POS = {100: [60, 100, 150, 200, 300, 400],
       500: [30, 50, 75, 100, 150, 200]}
# Contour interval for the FULL height field drawn over the shading.
FULL_CI = {100: 120.0, 500: 60.0}
CMAP = "RdBu_r"


def open_level(path, lev: int):
    """The whole forecast for one level, step dimension intact.

    Opened once per level rather than once per (level, step): the animation
    walks 16 steps and re-opening the grib for each would decode the same file
    16 times.
    """
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
        filter_by_keys={"shortName": "z", "level": lev}, indexpath=""))
    da = ds["z"] if "z" in ds else ds[list(ds.data_vars)[0]]
    # Perturbed files carry a member dimension; the ensemble MEAN is what is
    # plotted, so collapse it here and everything downstream is unchanged.
    #
    # Taking the mean before the k=1 extraction is not an approximation: the
    # FFT is linear, so the wave-1 of the mean IS the mean of the members'
    # wave-1. It is simply the cheaper order.
    if "number" in da.dims:
        da = da.mean("number", keep_attrs=True)
    return da


def at_step(da, step_h: int):
    if "step" not in da.dims:
        return da
    steps = (da.step.values / np.timedelta64(1, "h")).astype(int)
    if step_h not in set(steps):
        raise SystemExit(f"step {step_h}h not available (have {steps.min()}..{steps.max()})")
    return da.isel(step=int(np.where(steps == step_h)[0][0]))


def fetch(date: str, time: str, members: int = MEMBERS):
    """Geopotential at both levels, as the ENSEMBLE MEAN.

    z is not in the store's bulk list - it was dropped 2026-07-24 as ~0.5 GB a
    cycle of dead weight when nothing consumed it - so this pulls its own.

    Why the mean and not the control: the control is one realisation, and by
    day 10 its wave-1 phase is largely noise. The ensemble mean is the part of
    the wave the forecast actually agrees on, which is the only part worth
    reading a phase off. The cost is real - one member per level per step, so
    N members is N times the control - and AIFS-ENS publishes no `em` product
    to shortcut it (probed 2026-08-30: only cf and pf exist for z).

    ECMWF open data may not have every member at every step; if the perturbed
    pull fails outright we fall back to the control rather than lose the
    figure, and the subtitle says which was used.
    """
    cyc = ecmwf.Cycle(date, time)
    try:
        path = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "pf", "z", "pl",
                                            LEVELS, tuple(ecmwf.STEPS), members))
        return {lev: open_level(path, lev) for lev in LEVELS}, f"ensemble mean ({members} members)"
    except Exception as e:                                   # noqa: BLE001
        print(f"  perturbed pull failed ({type(e).__name__}: {e}); "
              f"falling back to the control", flush=True)
        path = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "z", "pl",
                                            LEVELS, tuple(ecmwf.STEPS)))
        return {lev: open_level(path, lev) for lev in LEVELS}, "control"


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


CLIM = Path(__file__).resolve().parent / "reference" / "wave1_clim.nc"
_CLIM_CACHE = {}


def clim_coeff(lev: int, hemi: str, valid: pd.Timestamp):
    """Climatological complex k=1 coefficient for this level/hemisphere/day.

    None when the climatology file is absent, so the figure still renders (the
    index panel is simply omitted) rather than the product failing.
    """
    if not CLIM.exists():
        return None, None
    if not _CLIM_CACHE:
        _CLIM_CACHE["ds"] = xr.open_dataset(CLIM)
    d = _CLIM_CACHE["ds"]
    key = f"{hemi.lower()}{lev}"
    if f"{key}_re" not in d:
        return None, None
    doy = min(int(valid.dayofyear), 366)
    i = doy - 1
    c = complex(float(d[f"{key}_re"][i]), float(d[f"{key}_im"][i]))
    # Annual maximum for this level/hemisphere, used to decide whether a
    # standing wave exists at all on this date.
    amax = float(np.max(np.hypot(d[f"{key}_re"].values, d[f"{key}_im"].values)))
    return c, amax


def superposition(Zband_coeff: complex, clim, amax=None):
    """How the forecast wave-1 interferes with the climatological standing wave.

    Total wave activity goes as |Z|^2. Splitting the forecast into the
    climatological standing wave plus an anomaly, Z = Zc + Za, gives

        |Z|^2 = |Zc|^2 + 2 Re(Zc conj(Za)) + |Za|^2

    and the middle term is the LINEAR INTERFERENCE - the only term whose sign
    can change. Positive means the anomaly is piling onto the standing wave and
    the total is larger than climatology; negative means it is cancelling it.
    This is the standard linear-interference diagnostic used for stratospheric
    wave driving, and it is why the phase matters more than the amplitude: an
    anomaly in quadrature with the climatology (90 deg out) contributes nothing
    to the total at first order however big it is.

    Reported normalised by |Zc|^2 so it is dimensionless and comparable across
    levels and seasons:

        S = Re(Za conj(Zc)) / |Zc|^2  =  (|Za|/|Zc|) cos(dphi)

    S = +1 means the anomaly is as large as the climatological wave and exactly
    in phase with it. Also returns the phase difference in degrees of longitude,
    which is the more physical way to read it.
    """
    if clim is None or abs(clim) == 0:
        return None, None
    # An interference index needs something to interfere WITH. The NH
    # climatological wave-1 at 100 hPa falls to 8 gpm in midsummer against a
    # 147 gpm winter maximum, and S divides by |Zc|^2 - so an ordinary anomaly
    # over a near-absent standing wave returns a meaningless S of several
    # units. Below a quarter of the annual maximum there is no standing wave to
    # reinforce or cancel (and in the summer hemisphere the easterlies stop
    # planetary waves propagating anyway), so the index is simply not defined.
    if amax is not None and abs(clim) < 0.25 * amax:
        return None, None
    anom = Zband_coeff - clim
    s = (anom * np.conj(clim)).real / (abs(clim) ** 2)
    # k=1, so a radian of phase is a radian of longitude.
    dphi = np.degrees(np.angle(anom) - np.angle(clim))
    dphi = (dphi + 180.0) % 360.0 - 180.0
    return float(s), float(dphi)


def band_coeff(Z, la, north: bool) -> complex:
    """Forecast complex k=1 coefficient over the same 55-65 deg band as the clim."""
    lo, hi = (55.0, 65.0) if north else (-65.0, -55.0)
    m = (la >= lo) & (la <= hi)
    if not m.any():
        return complex(0.0, 0.0)
    w = np.cos(np.deg2rad(la[m]))
    prof = (Z[m, :] * w[:, None]).sum(axis=0) / w.sum()
    return complex(np.fft.rfft(prof)[1] * 2.0 / prof.size)


def circular_boundary(ax):
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts * 0.5 + 0.5), transform=ax.transAxes)


def panel(ax, z, lev, hemi, valid=None):
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

    pos = POS[lev]
    # 12 edges -> 11 bands: 5 negative, the white deadband spanning
    # -pos[0]..+pos[0], then 5 positive.
    lv = np.array([-v for v in reversed(pos)] + pos, float)
    cmap = plt.get_cmap(CMAP)
    cols = ([cmap(x) for x in np.linspace(0.02, 0.40, 5)] + ["#ffffff"] +
            [cmap(x) for x in np.linspace(0.60, 0.98, 5)])
    cf = ax.contourf(lon, la, W1, levels=lv, colors=cols, extend="both",
                     transform=PC, zorder=1)

    # Full height field over the top: the shading says where wave-1 is, these
    # say what the actual flow looks like, and the reader can see at a glance how
    # much of the pattern wave-1 accounts for.
    ci = FULL_CI[lev]
    lo, hi = np.floor(Z.min() / ci) * ci, np.ceil(Z.max() / ci) * ci
    # Height contours are CONTEXT - they sit behind the geography, so they are
    # lighter than the coastline rather than darker (they were #3a3a3a/0.5
    # against a #555/0.5 coast, which made the data contours the most
    # prominent lines on a map).
    ax.contour(lon, la, Z, levels=np.arange(lo, hi + ci, ci), colors="#6e6e6e",
               linewidths=0.45, transform=PC, zorder=3)

    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#0d0d0d",
                   linewidth=1.0, zorder=4)
    gl = ax.gridlines(linewidth=0.3, color="#8a8a8a", alpha=0.45, zorder=5)
    gl.ylocator = plt.FixedLocator([20, 40, 60, 80] if north else [-80, -60, -40, -20])
    lat60 = 60.0 if north else -60.0
    ax.plot(np.linspace(-180, 180, 361), np.full(361, lat60), transform=PC,
            color="#444", linewidth=0.85, linestyle=(0, (5, 3)), zorder=5)

    # Amplitude quoted at 60 deg, the latitude the vortex diagnostics use.
    j = int(np.argmin(np.abs(la - lat60)))
    amp = float(np.abs(np.fft.rfft(Z[j])[1]) * 2.0 / Z.shape[1])
    title = f"{hemi} · {lev} hPa    wave-1 amp at 60° = {amp:.0f} gpm"

    # Superposition against the climatological standing wave.
    S = dphi = None
    if valid is not None:
        cc, amax = clim_coeff(lev, hemi, valid)
        S, dphi = superposition(band_coeff(Z, la, north), cc, amax)
    if S is not None:
        if abs(dphi) <= 60:      word, col = "reinforcing", "#b3122b"
        elif abs(dphi) >= 120:   word, col = "cancelling", "#1f4e9c"
        else:                    word, col = "in quadrature", "#666666"
        title += f"\nsuperposition S = {S:+.2f}  (Δφ = {dphi:+.0f}°)"
        ax.set_title(title, fontsize=10.0, pad=6)
        # The verdict carries the meaning, so it gets the colour rather than
        # tinting the whole title. Inside the panel, top-left: a polar plot is
        # clipped to a circle, so the corner is empty white and the badge sits
        # clear of both the title and the figure footnotes (placing it BELOW
        # the axes put it straight through the footer text).
        ax.text(0.02, 0.99, word.upper(), transform=ax.transAxes, ha="left",
                va="top", fontsize=9.5, fontweight="bold", color=col)
    else:
        ax.set_title(title, fontsize=10.5, pad=6)
        # Say why it is missing. Silence reads as a bug; this is a real result -
        # in the summer hemisphere there is no standing wave to interfere with,
        # and the easterlies would stop planetary waves propagating anyway.
        ax.text(0.02, 0.99, "NO STANDING WAVE", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.5, color="#999999")
    return cf


def render(z, lev, date, time, step_h, out_path: Path, source="ensemble mean"):
    """ONE level, both hemispheres, side by side.

    Unstacked deliberately. The two levels used to share a 2x2 figure, which
    made the animation nearly square and very tall on screen - the scrubber
    ended up below the fold and the loop was awkward to control. One level per
    frame halves the height and the animator's own selector switches between
    them, which is also the honest division: the reader compares 100 and 500
    hPa by flipping, and each view is big enough to actually read the phase.
    """
    fig = plt.figure(figsize=(11.4, 6.15), dpi=125)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045],
                          left=0.02, right=0.93, top=0.83, bottom=0.10,
                          wspace=0.06)
    init = pd.Timestamp(f"{date}T{time}:00")
    valid = init + pd.Timedelta(hours=step_h)
    cf = None
    for c, hemi in enumerate(("NH", "SH")):
        proj = (ccrs.NorthPolarStereo(central_longitude=0) if hemi == "NH"
                else ccrs.SouthPolarStereo(central_longitude=0))
        ax = fig.add_subplot(gs[0, c], projection=proj)
        cf = panel(ax, z, lev, hemi, valid)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(cf, cax=cax, extend="both")
    cb.set_label(f"{lev} hPa wave-1 height anomaly (gpm)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    tag = f"F{step_h:03d}" if step_h else "analysis"
    fig.suptitle(f"Planetary wave-1 in geopotential height — {lev} hPa",
                 fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.925,
             f"ECMWF AIFS-ENS {source} · {date[:4]}-{date[4:6]}-{date[6:]} {time}Z "
             f"{tag} · valid {valid:%a %d %b %HZ} · shaded: zonal wavenumber-1 "
             f"component · contours: full height field",
             fontsize=9, color="#555", ha="left")
    fig.text(0.02, 0.038,
             "Wave-1 is an anomaly by construction — the zonal mean is k = 0 and is "
             "discarded — so no climatology is involved in the SHADING.",
             fontsize=8, color="#777", ha="left")
    fig.text(0.02, 0.014,
             "S is the linear-interference term Re(Za·conj(Zc))/|Zc|²  against the "
             "1991–2020 standing wave: + reinforcing, − cancelling, 0 in quadrature.",
             fontsize=8, color="#777", ha="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125, facecolor="white",
                pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    return out_path


def build_loop(full, date, time, anim_root: Path, manifest: Path,
               source="ensemble mean") -> int:
    """One frame per step PER LEVEL, and a manifest the animator can switch on.

    anim_root is the animation ROOT (assets/sst/anim); this writes
    wave1_100/ and wave1_500/ under it, one region each.

    Two regions rather than one figure with both levels: see render(). The
    animator renders `selectorLabel` as a control, so the reader picks the
    level and the loop stays short and wide.
    """
    import json
    init = pd.Timestamp(f"{date}T{time}:00")
    regions = {}
    for lev in LEVELS:
        d = anim_root / f"wave1_{lev}"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("F*.webp"):
            old.unlink()
        frames = []
        for i, step_h in enumerate(ecmwf.STEPS):
            render(at_step(full[lev], step_h), lev, date, time, step_h,
                   d / f"F{i:02d}.webp", source)
            valid = init + pd.Timedelta(hours=step_h)
            frames.append({"idx": i, "file": f"F{i:02d}.webp",
                           "date": f"{valid:%Y-%m-%d}",
                           "label": f"F{step_h:03d} · valid {valid:%a %d %b %HZ}"})
            print(f"    {lev} hPa  F{step_h:03d}", flush=True)
        regions[f"wave1_{lev}"] = {
            "label": f"Wave-1 height anomaly — {lev} hPa, both hemispheres",
            "n_frames": len(frames), "frames": frames}
        print(f"  {lev} hPa: {len(frames)} frames -> {d}")
    man = {"ver": f"{date}{time}", "days": len(ecmwf.STEPS),
           "selectorLabel": "Level", "default": "wave1_100",
           "regions": regions}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(man))
    print(f"  wrote {manifest.name} with {len(regions)} regions")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--out", default="assets/sst/wave1_maps.webp",
                    help="static figure; --out-level picks which level it shows")
    ap.add_argument("--out-level", type=int, default=100, choices=list(LEVELS))
    ap.add_argument("--members", type=int, default=MEMBERS)
    ap.add_argument("--anim-dir", help="frames ROOT; wave1_100/ and wave1_500/ are created under it")
    ap.add_argument("--manifest", help="animator manifest path")
    a = ap.parse_args(argv)
    full, source = fetch(a.date, a.time, a.members)
    # The static figure stays: it is what the page shows before the loop is
    # loaded, and what a reader sees if the animator fails. 100 hPa by default -
    # it is the level the vortex actually feels.
    print(f"  wrote {render(at_step(full[a.out_level], a.step), a.out_level, a.date, a.time, a.step, Path(a.out), source)}")
    if a.anim_dir and a.manifest:
        build_loop(full, a.date, a.time, Path(a.anim_dir), Path(a.manifest), source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
