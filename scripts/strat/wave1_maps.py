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

IFS-ENS (2026-09-27, user: "We also add in the ifs-ens for all strat products"): --model ifs maps the mean of ALL 50
IFS-ENS perturbed members from the shared IFS-ENS gh file (ifs_ens.py, Google Cloud mirror only) into wave1_ifs_nh/ and
wave1_ifs_sh/. Open data has no IFS control at pressure levels, so its analysis frame is the IFS HRES step 0 - the
analysis the ensemble starts from.

Usage:
  python wave1_maps.py --date 20260829 --time 12 --anim-dir assets/sst/anim --manifest assets/sst/anim/wave1_maps_manifest.json
  python wave1_maps.py --model ifs --date 20260927 --time 00 --anim-dir assets/sst/anim \
      --manifest assets/sst/anim/wave1_maps_ifs_manifest.json
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
LEVELS = (500, 100, 10)      # left to right, bottom-up (10 hPa added 2026-09-27)
ROLE = {500: "troposphere (the source)", 100: "lower stratosphere", 10: "middle stratosphere (the vortex)"}
LAT_EDGE = 20.0
# Perturbed members averaged for the ensemble mean. 25 rather than all 50: at
# k=1 the mean converges long before the member count runs out, and the pull is
# one z field per member per level per step, so 50 would double a ~0.5 GB fetch
# for a change too small to see on the map.
MEMBERS = 25
# The still shows the level the vortex actually feels; the loop carries both.
STILL_LEVEL = 100

# Symmetric per-level scales in geopotential metres, fixed so a colour means the
# same wave amplitude every day. 100 hPa runs roughly twice 500 hPa.
SCALE = {10: 800.0, 100: 400.0, 500: 200.0}
# Amplitudes inside the innermost pair are left WHITE rather than tinted. A pale
# wash over a whole summer hemisphere reads as "something is happening here"
# when nothing is; blanking the weak values makes the hemisphere that is
# actually active obvious at a glance, and stops the eye chasing noise in the
# other one.
#
# Written out rather than computed with linspace so the colourbar carries round
# numbers a reader can actually use - an evenly divided range gave ticks like
# 116.7 and 343.3.
POS = {10: [100, 200, 300, 450, 600, 800],
       100: [60, 100, 150, 200, 300, 400],
       500: [30, 50, 75, 100, 150, 200]}
# Contour interval for the FULL height field drawn over the shading.
FULL_CI = {10: 160.0, 100: 120.0, 500: 60.0}
CMAP = "RdBu_r"


MODEL = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS"}
# Region/directory prefix per model: the AIFS loop keeps its original wave1_nh/, wave1_sh/.
PREFIX = {"aifs": "wave1", "ifs": "wave1_ifs"}


def open_level(path, lev: int, short: str = "z"):
    """The whole forecast for one level, step dimension intact.

    Opened once per level rather than once per (level, step): the animation
    walks 16 steps and re-opening the grib for each would decode the same file
    16 times.
    """
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
        filter_by_keys={"shortName": short, "level": lev}, indexpath=""))
    da = ds[short] if short in ds else ds[list(ds.data_vars)[0]]
    # z is geopotential (m2 s-2), gh already geopotential metres; split() converts by this factor.
    da.attrs["gpm_per_unit"] = 1.0 if short == "gh" else 1.0 / G
    # Perturbed files carry a member dimension; the ensemble MEAN is what is
    # plotted, so collapse it here and everything downstream is unchanged.
    #
    # Taking the mean before the k=1 extraction is not an approximation: the
    # FFT is linear, so the wave-1 of the mean IS the mean of the members'
    # wave-1. It is simply the cheaper order.
    # (2026-09-27) The members are KEPT now: the maps stipple where they disagree on the wave's sign and quote their
    # phase agreement. Subsampled to 1 deg first - k=1 needs nothing finer - so members x steps x levels stays small.
    out = da.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))
    out.attrs["gpm_per_unit"] = da.attrs["gpm_per_unit"]
    return out


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
    N members is N times member 0 - and AIFS-ENS publishes no `em` product
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


def fetch_ifs(date: str, time: str):
    """IFS-ENS: ALL 50 perturbed members of gh at 500/100/10 hPa from the shared IFS-ENS file (ifs_ens.py)."""
    import ifs_ens as IE
    if not IE.published(date, time):
        raise SystemExit(f"IFS-ENS {date} {time}Z is not on the Google mirror yet")
    path = IE.ensure(IE.E.Cycle(date, time), IE.spec_gh())
    return {lev: open_level(path, lev, "gh") for lev in LEVELS}, f"ensemble mean ({IE.MEMBERS} members)"


def fetch_analysis_ifs(date: str, time: str):
    """IFS has no control at pressure levels on open data; the HRES step 0 is the analysis the ensemble starts from."""
    import ifs_ens as IE
    path = IE.hres_analysis(IE.E.Cycle(date, time), ("gh",), LEVELS)
    return {lev: open_level(path, lev, "gh") for lev in LEVELS}


def fetch_analysis(date: str, time: str):
    """Step 0 from the CONTROL, for the analysis frame.

    The rest of the loop is the ensemble mean, but a mean is the wrong thing at
    step 0: the perturbed members differ from the analysis only BY their
    perturbations, so averaging them smooths the very state the analysis is. The
    control at step 0 IS the analysis. One extra field per level, ~1 MB.
    """
    cyc = ecmwf.Cycle(date, time)
    path = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "z", "pl", LEVELS, (0,)))
    return {lev: open_level(path, lev) for lev in LEVELS}


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
        return None, None, None
    # An interference index needs something to interfere WITH. The NH
    # climatological wave-1 at 100 hPa falls to 8 gpm in midsummer against a
    # 147 gpm winter maximum, and S divides by |Zc|^2 - so an ordinary anomaly
    # over a near-absent standing wave returns a meaningless S of several
    # units. Below a quarter of the annual maximum there is no standing wave to
    # reinforce or cancel (and in the summer hemisphere the easterlies stop
    # planetary waves propagating anyway), so the index is simply not defined.
    if amax is not None and abs(clim) < 0.25 * amax:
        return None, None, None
    anom = Zband_coeff - clim
    s = (anom * np.conj(clim)).real / (abs(clim) ** 2)
    # k=1, so a radian of phase is a radian of longitude.
    dphi = np.degrees(np.angle(anom) - np.angle(clim))
    dphi = (dphi + 180.0) % 360.0 - 180.0
    # TOTAL wave-1 power against climatology: |Zc+Za|^2/|Zc|^2 = 1 + 2S + r^2.
    #
    # S alone is a FIRST-ORDER diagnostic and only means what people take it to
    # mean while the anomaly is small against the standing wave, so the cross
    # term dominates the r^2 term. On 2026-08-30 the SH ran S = -1.28 at 100 hPa
    # with r = 1.88: the anomaly was nearly twice the climatological wave, the
    # quadratic term won, and TOTAL wave-1 power came out at 1.96x climatology.
    # Labelling that "cancelling" off the sign of S was simply wrong - the wave
    # had reversed phase and grown, not cancelled. Report the total so the
    # reader can see when the linear reading has stopped applying.
    tot = 1.0 + 2.0 * s + abs(anom / clim) ** 2
    return float(s), float(dphi), float(tot)


def band_coeff(Z, la, north: bool, lon0: float = 0.0) -> complex:
    """Forecast complex k=1 coefficient over the same 55-65 deg band as the clim, on the CLIMATOLOGY'S longitude
    origin.

    The climatology was built on WeatherBench2's grid, which starts at 0 E; ECMWF open data starts at -180. For a
    profile sampled from lon0, the k=1 coefficient carries a factor exp(i*lon0), so without the correction every
    forecast phase was 180 deg off the climatology's. The amplitude ratio (|Z|/|Zc|) never depended on it, but the
    interference index and its phase did: until 2026-09-27 an in-phase wave read as reversed, e.g. S -2.30 at
    -175 deg for the 2026-09-27 00Z F360 NH 100 hPa map, which is +0.30 corrected (found when the user asked for the
    plot to be made more useful)."""
    lo, hi = (55.0, 65.0) if north else (-65.0, -55.0)
    m = (la >= lo) & (la <= hi)
    if not m.any():
        return complex(0.0, 0.0)
    w = np.cos(np.deg2rad(la[m]))
    prof = (Z[..., m, :] * w[:, None]).sum(axis=-2) / w.sum()
    c = np.fft.rfft(prof, axis=-1)[..., 1] * 2.0 / prof.shape[-1]
    return c * np.exp(-1j * np.deg2rad(lon0))


def ridge_lon(c) -> float:
    """Longitude (deg E, -180..180) of the wave-1 ridge for a 0-E-origin coefficient: profile = |c| cos(lon + arg c)."""
    return float((-np.degrees(np.angle(c)) + 180.0) % 360.0 - 180.0)


def lon_label(x: float) -> str:
    x = (x + 180.0) % 360.0 - 180.0
    return f"{abs(x):.0f}°{'E' if x > 0 else 'W'}" if abs(x) > 0.5 and abs(abs(x) - 180) > 0.5 else f"{abs(x):.0f}°"


def circular_boundary(ax):
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts * 0.5 + 0.5), transform=ax.transAxes)


def split(z):
    """(ensemble-mean field, member fields or None) in gpm. The analysis frame is the control: no members."""
    k = z.attrs.get("gpm_per_unit", 1.0 / G)
    if "number" in z.dims:
        m = z.transpose("number", "latitude", "longitude").values * k
        return m.mean(0), m
    return z.values * k, None


def panel(ax, z, lev, hemi, valid=None):
    """One level: wave-1 shaded, full heights contoured, ridge markers (this forecast vs normal) on the 60 deg circle,
    stippling where the members disagree on the sign. Returns (contour set, info) for the tilt line."""
    lat = z.latitude.values
    lon = z.longitude.values
    north = hemi == "NH"
    sel = (lat >= LAT_EDGE) if north else (lat <= -LAT_EDGE)
    la = lat[sel]
    Zall, M = split(z)
    Z = Zall[sel, :]
    W1 = wave1(Z)

    ax.set_extent([-180, 180, LAT_EDGE if north else -90, 90 if north else -LAT_EDGE], crs=PC)
    circular_boundary(ax)

    pos = POS[lev]
    lv = np.array([-v for v in reversed(pos)] + pos, float)
    cmap = plt.get_cmap(CMAP)
    cols = ([cmap(x) for x in np.linspace(0.02, 0.40, 5)] + ["#ffffff"] +
            [cmap(x) for x in np.linspace(0.60, 0.98, 5)])
    cf = ax.contourf(lon, la, W1, levels=lv, colors=cols, extend="both", transform=PC, zorder=1)

    info = {"lev": lev, "coherence": None}
    if M is not None:
        # Where the members disagree on the sign of the wave-1 component, the ensemble mean there is a compromise
        # between opposite phases: stipple it (only inside the shaded range - the white band is already "weak").
        Wm = np.stack([wave1(m[sel, :]) for m in M])
        agree = (np.sign(Wm) == np.sign(W1)[None]).mean(0)
        mask = (agree < 0.8) & (np.abs(W1) >= pos[0])
        yy, xx = np.meshgrid(la, lon, indexing="ij")
        st = (slice(None, None, 3), slice(None, None, 3))
        ax.scatter(xx[st][mask[st]], yy[st][mask[st]], s=0.9, c="#222", alpha=0.6, linewidths=0, transform=PC, zorder=2)
        # Phase agreement of the members' band wave-1: |mean coefficient| / mean |coefficient| (1 = all in phase).
        cm = band_coeff(M[:, sel, :], la, north, lon[0])
        info["coherence"] = float(np.abs(cm.mean()) / np.mean(np.abs(cm)))

    ci = FULL_CI[lev]
    lo, hi = np.floor(Z.min() / ci) * ci, np.ceil(Z.max() / ci) * ci
    ax.contour(lon, la, Z, levels=np.arange(lo, hi + ci, ci), colors="#6e6e6e", linewidths=0.45, transform=PC, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#0d0d0d", linewidth=0.9, zorder=4)
    gl = ax.gridlines(linewidth=0.3, color="#8a8a8a", alpha=0.45, zorder=5)
    gl.ylocator = plt.FixedLocator([20, 40, 60, 80] if north else [-80, -60, -40, -20])
    lat60 = 60.0 if north else -60.0
    ax.plot(np.linspace(-180, 180, 361), np.full(361, lat60), transform=PC, color="#444", linewidth=0.85,
            linestyle=(0, (5, 3)), zorder=5)

    c = band_coeff(Z, la, north, lon[0])
    amp = float(abs(c))
    info.update(amp=amp, ridge=ridge_lon(c))
    ax.plot(info["ridge"], lat60, marker="^", ms=11, mfc="#b3122b", mec="white", mew=1.0, transform=PC, zorder=7)
    cc, amax = clim_coeff(lev, hemi, valid) if valid is not None else (None, None)
    head = f"{lev} hPa · {ROLE[lev]}"
    line2 = f"wave-1 {amp:.0f} gpm at 55–65°"
    word = None
    if cc is not None and amax is not None and abs(cc) >= 0.25 * amax:
        ratio = amp / abs(cc)
        info.update(ratio=ratio, clim_ridge=ridge_lon(cc))
        ax.plot(info["clim_ridge"], lat60, marker="^", ms=11, mfc="none", mec="#333", mew=1.4, transform=PC, zorder=6)
        d = (info["ridge"] - info["clim_ridge"] + 180.0) % 360.0 - 180.0
        line2 += f" = {ratio:.1f}× normal · ridge {abs(d):.0f}° {'west' if d < 0 else 'east'} of normal"
        word = ("amplified", "#b3122b") if ratio >= 1.25 else ("weak", "#1f4e9c") if ratio <= 0.75 else ("near normal", "#555555")
    else:
        line2 += " · no standing wave in the normal"
    if info["coherence"] is not None:
        line2 += f"\nmembers' phase agreement {info['coherence']:.2f} (1 = all in phase)"
    ax.set_title(head + "\n" + line2, fontsize=9.6, pad=5, linespacing=1.3)
    if word:
        ax.text(0.02, 0.99, word[0], transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold",
                color=word[1])
    return cf, info


def tilt_sentence(infos) -> str:
    """How the ridge moves with height, 500 -> 100 -> 10 hPa, as two lines: where it is, and what that means.
    Westward with height = the wave is carrying activity upward (positive heat flux); stacked = little vertical
    propagation; eastward = not propagating up."""
    shifts, kinds = [], []
    for a, b in zip(infos[:-1], infos[1:]):
        d = (b["ridge"] - a["ridge"] + 180.0) % 360.0 - 180.0
        shifts.append(f"{abs(d):.0f}° {'west' if d < 0 else 'east'} ({a['lev']}→{b['lev']})")
        kinds.append("stacked" if abs(d) < 20 else "west" if -150 < d <= -20 else "east" if 20 <= d < 150 else "none")
    ridges = " → ".join(f"{i['lev']} hPa {lon_label(i['ridge'])}" for i in infos)
    if all(k == "west" for k in kinds):
        verdict = "leans WEST all the way up: carrying wave activity into the vortex"
    elif kinds[0] == "west" and kinds[1] != "west":
        verdict = "leans west into the lower stratosphere, then " + {"stacked": "stacks: the activity is not reaching 10 hPa",
                   "east": "leans east: not reaching 10 hPa", "none": "loses coherence above 100 hPa"}[kinds[1]]
    elif kinds[0] != "west" and kinds[1] == "west":
        verdict = "no westward tilt out of the troposphere; the tilt is confined to the stratosphere"
    else:
        verdict = "no westward tilt: little wave activity propagating up"
    return (f"Ridge longitude at 55–65°: {ridges}\nShift with height: {'; '.join(shifts)} — {verdict}")


def render(levels_at_step, hemi, date, time, step_h, out_path: Path, source="ensemble mean", model="aifs"):
    """ONE hemisphere, three levels bottom-up, left to right: 500 hPa (the tropospheric source), 100 hPa (where wave
    driving reaches the vortex) and 10 hPa (the vortex itself). The comparison that carries the physics is between
    levels - a ridge that moves WEST with height is carrying wave activity up - so the ridge longitudes and the shift
    between levels are written out under the maps rather than left to the eye (2026-09-27 rework, user: "Is there a
    way we can improve this plot a bit to make it more useful?")."""
    fig = plt.figure(figsize=(15.6, 6.9), dpi=110)
    init = pd.Timestamp(f"{date}T{time}:00")
    valid = init + pd.Timedelta(hours=step_h)
    proj = (ccrs.NorthPolarStereo(central_longitude=0) if hemi == "NH" else ccrs.SouthPolarStereo(central_longitude=0))
    infos = []
    w, gap, left = 0.305, 0.022, 0.018
    for c, lev in enumerate(LEVELS):
        x0 = left + c * (w + gap)
        ax = fig.add_axes([x0, 0.215, w, 0.60], projection=proj)
        cf, info = panel(ax, levels_at_step[lev], lev, hemi, valid)
        infos.append(info)
        cax = fig.add_axes([x0 + 0.03, 0.175, w - 0.06, 0.018])
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both", ticks=cf.levels)
        cb.ax.tick_params(labelsize=7.2, pad=1)
        cb.set_label(f"{lev} hPa wave-1: height above (+) / below (−) the zonal mean, gpm", fontsize=8, labelpad=1)
    full = "Northern Hemisphere" if hemi == "NH" else "Southern Hemisphere"
    tag = f"F{step_h:03d}" if step_h else "analysis"
    fig.text(left, 0.975, f"ECMWF {MODEL[model]} · planetary wave-1 from the troposphere to the vortex — {full}",
             fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(left, 0.925, f"{source} · {date[:4]}-{date[4:6]}-{date[6:]} {time}Z {tag} · valid {valid:%a %d %b %HZ} · "
             "shaded: zonal wavenumber-1 of geopotential height · grey contours: full height field",
             fontsize=9.2, color="#6f6b64", ha="left", va="top")
    fig.text(left, 0.118, tilt_sentence(infos), fontsize=9.6, ha="left", va="top", color="#1a1a1a", fontweight="bold",
             linespacing=1.4)
    fig.text(left, 0.058,
             "Filled ▲: this map's wave-1 ridge; open △: where the 1991–2020 normal puts it on this date. "
             "× normal = wave-1 amplitude over the normal standing wave's (ERA5, 55–65°). Dots: under 80 % of the members "
             "agree on the sign.\nA ridge that shifts WEST going up (500 → 100 → 10 hPa) is carrying wave activity into "
             "the vortex (positive eddy heat flux); one stacked over the same longitude is not.",
             fontsize=8.3, color="#6f6b64", ha="left", va="top", linespacing=1.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    return out_path


def active_hemisphere(full) -> str:
    """Whichever hemisphere currently carries the larger 100 hPa wave-1 (the still/loop default follows the season)."""
    amp = {}
    z = at_step(full[100], 0)
    Zall, _ = split(z)
    lat = z.latitude.values
    for hemi in ("NH", "SH"):
        north = hemi == "NH"
        sel = (lat >= LAT_EDGE) if north else (lat <= -LAT_EDGE)
        amp[hemi] = abs(band_coeff(Zall[sel, :], lat[sel], north))
    return "NH" if amp["NH"] >= amp["SH"] else "SH"


def build_loop(full, date, time, anim_root: Path, manifest: Path,
               source="ensemble mean", default_hemi="NH", model="aifs") -> int:
    """One frame per step PER HEMISPHERE, and a manifest the animator switches on.

    anim_root is the animation ROOT (assets/sst/anim); this writes wave1_nh/
    and wave1_sh/ under it, one region each.
    """
    import json
    init = pd.Timestamp(f"{date}T{time}:00")
    # Analysis first, then the forecast. The analysis frame comes from the
    # CONTROL (see fetch_analysis) while every forecast frame is the ensemble
    # mean, so the loop starts from the state the forecast was launched from
    # rather than from a smoothed version of it.
    steps = [0] + [h for h in ecmwf.STEPS if h > 0]
    ana = fetch_analysis_ifs(date, time) if model == "ifs" else fetch_analysis(date, time)
    ana_label = ("IFS HRES analysis, the state IFS-ENS starts from (no IFS-ENS control on open data)"
                 if model == "ifs" else "control (analysis)")
    regions = {}
    for hemi in ("NH", "SH"):
        key = f"{PREFIX[model]}_{hemi.lower()}"
        d = anim_root / key
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("F*.webp"):
            old.unlink()
        frames = []
        for i, step_h in enumerate(steps):
            if step_h == 0:
                fields = {lev: at_step(ana[lev], 0) for lev in LEVELS}
                src = ana_label
            else:
                fields = {lev: at_step(full[lev], step_h) for lev in LEVELS}
                src = source
            render(fields, hemi, date, time, step_h, d / f"F{i:02d}.webp", src, model)
            valid = init + pd.Timedelta(hours=step_h)
            lab = (f"analysis · {valid:%a %d %b %HZ}" if step_h == 0
                   else f"F{step_h:03d} · valid {valid:%a %d %b %HZ}")
            frames.append({"idx": i, "file": f"F{i:02d}.webp",
                           "date": f"{valid:%Y-%m-%d}", "label": lab})
            print(f"    {hemi}  {'analysis' if step_h == 0 else f'F{step_h:03d}'}", flush=True)
        regions[key] = {
            "label": ("Northern" if hemi == "NH" else "Southern")
                     + " Hemisphere — wave-1 at 500, 100 and 10 hPa" + (", IFS-ENS" if model == "ifs" else ""),
            "n_frames": len(frames), "frames": frames}
        print(f"  {hemi}: {len(frames)} frames -> {d}")
    man = {"ver": f"{date}{time}", "days": len(steps),
           "selectorLabel": "Hemisphere",
           "default": f"{PREFIX[model]}_{default_hemi.lower()}",
           "regions": regions}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(man))
    print(f"  wrote {manifest.name}, default {default_hemi}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--hemi", choices=["NH", "SH", "auto"], default="auto",
                    help="hemisphere for the still (default: whichever is active)")
    ap.add_argument("--members", type=int, default=MEMBERS)
    ap.add_argument("--model", choices=("aifs", "ifs"), default="aifs")
    ap.add_argument("--anim-dir", help="frames ROOT; wave1_nh/ and wave1_sh/ go under it")
    ap.add_argument("--manifest", help="animator manifest path")
    a = ap.parse_args(argv)
    full, source = fetch_ifs(a.date, a.time) if a.model == "ifs" else fetch(a.date, a.time, a.members)
    hemi = active_hemisphere(full) if a.hemi == "auto" else a.hemi
    print(f"  leading hemisphere: {hemi}", flush=True)
    # No still. It duplicated the loop's first frame, and on the page it was the
    # analysis - the one view that cannot answer whether the wave is building.
    if not (a.anim_dir and a.manifest):
        print("  nothing to do: --anim-dir and --manifest are required "
              "(the loop is the product; there is no still)")
        return 2
    build_loop(full, a.date, a.time, Path(a.anim_dir), Path(a.manifest), source, hemi, a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
