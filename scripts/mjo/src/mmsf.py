#!/usr/bin/env python3
"""Meridional mass streamfunction Ψ(φ,p) from the AIFS-ENS 0-h analysis.

    Ψ(φ,p) = (2π a cosφ / g) ∫_0^p [v] dp'        [v] = zonal-mean meridional wind
    units 10^10 kg s^-1.  Ψ>0 ⇒ clockwise cell in the (φ,p) plane (northward aloft).

Each cycle's zonal-mean v is appended to a small history file; the last N cycles
are re-rendered as Ψ ANOMALY frames (Ψ − ERA5 harmonic climatology, build_mmsf_clim.py)
into a sliding-window animator, so you can watch the Hadley/Ferrel cells respond to
the current event. A fixed colour scale keeps the animation steady frame-to-frame.

    python src/mmsf.py --date 20260602 --time 00 \
        --anim-dir assets/sst/anim/mmsf --manifest assets/sst/anim/mmsf_manifest.json \
        --out assets/sst/mmsf_anom.webp
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "era5"))
import store as ecmwf                                    # shared ECMWF download manager
import era5_store                                        # ARCO-ERA5 gap-fill for outages

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
A = 6.371e6; G = 9.80665; SCALE = 1e10        # Earth radius, gravity, plot units
REF = Path(__file__).resolve().parent.parent / "data" / "reference"
CLIM = REF / "mmsf_clim_coeffs.nc"            # ERA5 1991-2020 Ψ harmonic coeffs
HIST = REF / "mmsf_vbar_history.nc"           # rolling per-cycle zonal-mean v
MAXN = 120                                    # frames to keep (≈ 60 days twice-daily)


def download_v(date: str, time: str, dd: Path = None) -> Path:
    """0-h analysis meridional wind on the 13 pressure levels (control = analysis at
    t=0), via the shared ECMWF store. Returns the cache path."""
    cyc = ecmwf.Cycle(date, time)
    return ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "v", "pl", tuple(LEVELS), (0,)))


def streamfunction(vbar: np.ndarray, p_pa: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Ψ(p,lat) in 10^10 kg/s from zonal-mean v (lev,lat), p ascending (Pa)."""
    psi = np.zeros_like(vbar)
    for k in range(1, len(p_pa)):
        dp = p_pa[k] - p_pa[k - 1]
        psi[k] = psi[k - 1] + 0.5 * (vbar[k] + vbar[k - 1]) * dp
    psi *= (2 * np.pi * A * np.cos(np.deg2rad(lat))[None, :] / G)
    return psi / SCALE


def zonal_mean_v(vpath: Path):
    ds = xr.open_dataset(vpath, engine="cfgrib", backend_kwargs={"indexpath": ""})
    v = ds["v"].sortby("isobaricInhPa")
    lat = v.latitude.values
    p_pa = v.isobaricInhPa.values * 100.0
    vbar = v.mean("longitude").values            # (lev, lat)
    return vbar, p_pa, lat


def update_history(vbar, p_pa, lat, valid: datetime) -> xr.DataArray:
    """Append this cycle's zonal-mean v to the rolling history; dedupe + prune to MAXN."""
    cur = xr.DataArray(
        vbar[None], dims=("time", "level", "latitude"),
        coords={"time": [pd.Timestamp(valid)],
                "level": (np.asarray(p_pa) / 100).astype(int), "latitude": lat}, name="vbar")
    if HIST.exists():
        old = xr.open_dataarray(HIST)
        old = old.sel(time=old.time != np.datetime64(valid))   # drop same cycle on re-run
        da = xr.concat([old, cur], dim="time").sortby("time") if old.time.size else cur
    else:
        da = cur
    da = da.isel(time=slice(-MAXN, None))
    HIST.parent.mkdir(parents=True, exist_ok=True)
    tmp = HIST.with_suffix(".tmp.nc"); da.to_netcdf(tmp); tmp.replace(HIST)
    return da


def backfill_history_gaps(hist: xr.DataArray) -> xr.DataArray:
    """Fill missing 12Z days in the trailing 30 days of the history from ERA5
    (ARCO), so pipeline outages don't stall the weekly-mean frames: after a
    multi-day gap no trailing window spans the required ~week and the newest
    emitted frame freezes at the last dense stretch. ERA5 lags ~5 days; days
    ARCO hasn't published yet come back all-NaN and are simply skipped."""
    t = pd.to_datetime(hist.time.values)
    end = t.max().normalize()
    have_days = {x.normalize() for x in t}
    days = pd.date_range(end - pd.Timedelta(days=30), end, freq="D")
    lat = hist.latitude.values
    added = 0
    for day in days:
        if day in have_days:
            continue
        stamp = day + pd.Timedelta(hours=12)
        try:
            v = era5_store.get_v(stamp, LEVELS)
        except Exception as e:                             # noqa: BLE001
            print(f"  gap-fill {stamp:%Y-%m-%d}: ERA5 unavailable ({repr(e)[:60]})")
            continue
        if bool(np.isnan(v.values).all()):                 # ARCO NaN padding: not published yet
            continue
        vbar = v.mean("longitude").interp(latitude=lat).values
        hist = update_history(vbar, np.asarray(LEVELS, float) * 100.0, lat,
                              stamp.to_pydatetime())
        added += 1
    if added:
        print(f"  gap-fill: appended {added} ERA5 day(s) to the v̄ history")
    return hist


def clim_psi(coeffs: xr.DataArray, doy: int) -> np.ndarray:
    """Evaluate the mean+annual+semiannual Ψ climatology for a day-of-year → (lev,lat)."""
    w = 2 * np.pi * doy / 365.25
    basis = np.array([1.0, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    return np.tensordot(basis, coeffs.values, axes=(0, 0))


# ── Hadley-cell strength ────────────────────────────────────────────────────
# The two Hadley cells are the two same-sign Ψ bands either side of the ITCZ
# rising branch, NOT the two hemispheres: in solstice season the winter cell is
# cross-equatorial (right now its core sits ON the equator), so splitting at 0°
# measures the same cell twice. Sign identifies the cell — Ψ>0 is northward flow
# aloft, i.e. sinking to the north = the NORTHERN cell; Ψ<0 = the southern one.
# The band walk is also what keeps the SH Ferrel cell (positive, like the
# northern Hadley cell, and in JJA the stronger of the two) out of the answer.
HAD_LAT = 45.0            # search limit: Ferrel cell cores sit poleward of this
HAD_P = (200.0, 900.0)    # excludes the TOA zero line the integral starts from
                          # and the shallow surface return flow


def _sign_band(row, j, eps):
    """Index range of the same-sign run of `row` containing j. `eps` is a
    deadband: near-zero wobble does not end a band, a real opposite cell does."""
    s = 1.0 if row[j] >= 0 else -1.0
    a = b = j
    while a > 0 and row[a - 1] * s > -eps:
        a -= 1
    while b < row.size - 1 and row[b + 1] * s > -eps:
        b += 1
    return a, b


def psi_mid(psi, p_hpa):
    """Mid-tropospheric mean Ψ(lat), 300-700 hPa — the smooth single-valued
    profile the cell EDGES are read off. Ψ itself is used for intensity."""
    return psi[(p_hpa >= 300) & (p_hpa <= 700)].mean(axis=0)


def _zero_cross(prof, la, j0, step):
    """Latitude of the first Ψ_mid = 0 crossing walking from index j0 north
    (step=+1) or south (step=-1), linearly interpolated. NaN if the profile
    never changes sign inside the search box (then the cell is unbounded there
    and an edge latitude would be fiction)."""
    s = 1.0 if prof[j0] >= 0 else -1.0
    j = j0
    while 0 <= j + step < prof.size:
        k = j + step
        if prof[k] * s <= 0:
            d = abs(prof[j]) + abs(prof[k])
            f = abs(prof[j]) / d if d else 0.0
            return float(la[j] + f * (la[k] - la[j]))
        j = k
    return float("nan")


def cell_edges(pm, la, core_lat):
    """(equatorward, poleward) edge latitudes of the cell containing core_lat,
    from the mid-tropospheric profile. Returned in south→north order."""
    j0 = int(np.argmin(np.abs(la - core_lat)))
    lo, hi = _zero_cross(pm, la, j0, -1), _zero_cross(pm, la, j0, +1)
    # a crossing sitting on the search boundary is the box edge, not a cell edge
    lim = HAD_LAT - 1.0
    lo = lo if abs(lo) <= lim else float("nan")
    hi = hi if abs(hi) <= lim else float("nan")
    return lo, hi


def hadley_cells(psi, p_hpa, lat):
    """Both Hadley cells of one Ψ field → {'nh': {...}, 'sh': {...}}, values in
    10¹⁰ kg s⁻¹, keyed by the sign convention above (missing cell → None)."""
    o = np.argsort(lat)                       # ECMWF grids run 90 → -90
    la = lat[o]
    ml = np.abs(la) <= HAD_LAT
    mp = (p_hpa >= HAD_P[0]) & (p_hpa <= HAD_P[1])
    box = psi[np.ix_(mp, o)][:, ml]           # (lev, lat) inside the search box
    lb, pb = la[ml], p_hpa[mp]
    pm = psi_mid(psi, p_hpa)[o][ml]           # edge profile, same box
    if not np.isfinite(box).any():
        return {"nh": None, "sh": None}

    # Cell profile: at each latitude the signed Ψ where the column overturns
    # hardest. Taking a single level instead loses the summer cell at solstice —
    # it is strong near 400 hPa and ~0 at the winter cell's 600 hPa core level,
    # so the band walk sails straight through it and returns a Ferrel cell.
    kk = np.nanargmax(np.where(np.isfinite(box), np.abs(box), -1.0), axis=0)
    row = box[kk, np.arange(box.shape[1])]
    kj = int(np.nanargmax(np.abs(row)))
    peak = abs(float(row[kj]))
    # deadband: near-zero wobble must not end a band, but keep an absolute
    # floor so a weak Ferrel cell still stops the walk when peak is large.
    eps = min(0.02 * peak, 0.3)
    a, b = _sign_band(row, kj, eps)

    # Which edge of the dominant band is the ITCZ? The one where air RISES.
    # Crossing the ITCZ the two Hadley cells hand over, so Ψ increases northward
    # (dΨ/dφ > 0 = ascent); at the subtropical edge it decreases (descent). A
    # nearest-the-equator rule gets this wrong whenever the summer cell is too
    # weak to stop the walk: the band then runs to the search limit, the
    # subtropical edge looks "nearer", and the companion picked up is the FERREL
    # cell — which shares the summer cell's sign and outweighs it, so the cell
    # is reported at ~25-30 deg in the wrong hemisphere. Ascent settles it, and
    # when neither edge ascends there is genuinely no second cell to report.
    cand = []
    if a > 0 and row[a] - row[a - 1] > 0:
        cand.append((abs(lb[a]), a - 1))
    if b < row.size - 1 and row[b + 1] - row[b] > 0:
        cand.append((abs(lb[b]), b + 1))
    bands = [(a, b)]
    if cand:
        bands.append(_sign_band(row, min(cand)[1], eps))

    out = {"nh": None, "sh": None}
    for (i0, i1) in bands:
        sub = np.array(box[:, i0:i1 + 1], dtype=float)
        bs = 1.0 if row[i0:i1 + 1][np.nanargmax(np.abs(row[i0:i1 + 1]))] >= 0 else -1.0
        sub[sub * bs < 0] = np.nan            # never let a leaked Ferrel cell win
        if not np.isfinite(sub).any():
            continue
        vi, vj = np.unravel_index(np.nanargmax(np.abs(sub)), sub.shape)
        val = float(sub[vi, vj])
        clat = float(lb[i0:i1 + 1][vj])
        if abs(clat) > 38.0:
            continue                          # a Ferrel cell core, not a Hadley one
        key = "nh" if val >= 0 else "sh"
        if out[key] is not None and abs(val) <= abs(out[key]["psi"]):
            continue                          # keep the stronger claimant
        e_lo, e_hi = cell_edges(pm, lb, clat)
        out[key] = {"psi": round(val, 2),
                    "core_lat": round(clat, 1),
                    "core_p": round(float(pb[vi]), 0),
                    # edges from Ψ_mid = 0, south→north; the poleward one is the
                    # descending branch (north for the NH cell, south for the SH)
                    "edges": [None if np.isnan(e_lo) else round(e_lo, 1),
                              None if np.isnan(e_hi) else round(e_hi, 1)]}
    return out


HADCLIM = REF / "hadley_clim.nc"          # per-sample ERA5 metric climatology
_HC = {}


def _hadclim(doy: int):
    """Median/mean/sd of each metric for this day of year, from hadley_clim.nc.

    Preferred over reading the normals off the harmonic Ψ field: that field is a
    30-YEAR MEAN circulation, and cell edges wander ~10° between years, so its
    summer cell is washed out — it puts the late-August northern normal at 1.8
    against a true across-year median of 3.6. Returns None if the file has not
    been built (build_hadley_clim.py), and the caller falls back."""
    if not HADCLIM.exists():
        return None
    if "ds" not in _HC:
        try:
            _HC["ds"] = xr.open_dataset(HADCLIM)
        except Exception:                                      # noqa: BLE001
            _HC["ds"] = None
    ds = _HC["ds"]
    if ds is None:
        return None
    r = ds.sel(doy=int(min(max(doy, 1), 366)))
    out = {}
    for key in ("nh", "sh"):
        out[key] = {st: float(r[f"{key}_psi"].sel(stat=st)) for st in ("p50", "mean", "sd")}
    return out


def hadley_strength(psi_abs, psi_clim, p_hpa, lat, doy=None):
    """This frame's cells against the ERA5 normal for the same day of year —
    each field's cells found independently, then paired by sign."""
    now, nor = hadley_cells(psi_abs, p_hpa, lat), hadley_cells(psi_clim, p_hpa, lat)
    hc = _hadclim(doy) if doy is not None else None
    out = {}
    for key in ("nh", "sh"):
        c, n = now.get(key), nor.get(key)
        if c is None:
            out[key] = None
            continue
        d = dict(c)
        d["normal_core_lat"] = n["core_lat"] if n else None
        if hc and np.isfinite(hc[key]["p50"]):
            ref, sd = hc[key]["p50"], abs(hc[key]["sd"])
            d["normal_basis"] = "era5_median"
            d["sigma"] = (round((c["psi"] - hc[key]["mean"]) / sd
                                * (1.0 if key == "nh" else -1.0), 2)
                          if np.isfinite(sd) and sd else None)
        else:                                   # harmonic-Ψ fallback
            ref = n["psi"] if n else None
            d["normal_basis"] = "harmonic_psi"
            d["sigma"] = None
        d["normal"] = None if ref is None else round(ref, 2)
        d["anom"] = None if ref is None else round(c["psi"] - ref, 2)
        # % of normal is meaningless against a near-zero normal (the summer cell
        # nearly vanishes at solstice) — the absolute anomaly carries it there.
        d["pct_of_normal"] = (round(100.0 * c["psi"] / ref, 1)
                              if ref is not None and abs(ref) >= 2.0 else None)
        out[key] = d
    return out


def strength_texts(hs):
    """One self-describing corner label per cell."""
    txt = []
    for key, x, name in (("sh", -27.0, "Southern"), ("nh", 27.0, "Northern")):
        d = hs.get(key)
        if not d or d.get("normal") is None:
            continue
        lab = ("ERA5 median" if d.get("normal_basis") == "era5_median" else "normal")
        sig = "" if d.get("sigma") is None else f"   ({d['sigma']:+.1f}\u03c3)"
        if d["pct_of_normal"] is not None:
            second = (f"{lab} {abs(d['normal']):.1f}   \u2192   "
                      f"{d['pct_of_normal']:.0f}%{sig}")
        else:
            second = (f"{lab} {abs(d['normal']):.1f}   \u2192   "
                      f"{d['anom']:+.1f}{sig}")
        txt.append({"x": x, "y": 118.0, "size": 8.5, "color": "#1a1a1a",
                    "s": (f"{name} cell   |\u03a8|max = {abs(d['psi']):.1f} "
                          f"@ {d['core_lat']:+.0f}\u00b0\n{second}")})
    return txt


def plot_psi(psi, p_hpa, lat, out: Path, title: str, anom=False, vlim=None, psi_abs=None,
             texts=None):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    mlat = (lat >= -45) & (lat <= 45)                      # tropics/subtropics focus
    lim = vlim if vlim is not None else float(np.nanpercentile(np.abs(psi[:, mlat]), 99.5))
    levs = np.linspace(-lim, lim, 21)
    cf = ax.contourf(lat[mlat], p_hpa, psi[:, mlat], levels=levs, cmap="RdBu_r", extend="both")
    if psi_abs is not None:
        # absolute Ψ as black contours = the actual overturning cells (every 5 units)
        clev = np.array([v for v in range(-100, 101, 5) if v != 0])
        cs = ax.contour(lat[mlat], p_hpa, psi_abs[:, mlat], levels=clev, colors="k", linewidths=1.0)
        ax.clabel(cs, levels=[v for v in clev if v % 20 == 0], fmt="%d", fontsize=6, inline=True)
        ax.contour(lat[mlat], p_hpa, psi_abs[:, mlat], levels=[0], colors="k", linewidths=0.9)  # cell boundary
        # vertical-motion arrows: length ∝ ascent/descent strength. ω ∝ −∂Ψ/∂φ/cosφ,
        # so "up" (rising) ∝ +∂Ψ/∂lat/cosφ; flat/weak columns are hidden.
        LATg = lat[mlat]; PA = psi_abs[:, mlat]
        wvert = np.gradient(PA, LATg, axis=1) / np.cos(np.deg2rad(LATg))[None, :]
        wn = wvert / (np.nanpercentile(np.abs(wvert), 96) or 1.0)
        wn[np.abs(wn) < 0.12] = np.nan
        sj = max(1, LATg.size // 22)
        LATm, Pm = np.meshgrid(LATg, p_hpa)
        ax.quiver(LATm[:, ::sj], Pm[:, ::sj], np.zeros_like(wn)[:, ::sj], wn[:, ::sj],
                  angles="uv", scale_units="height", scale=9, width=0.0028,
                  headwidth=4, headlength=5, color="0.15", alpha=0.8, pivot="mid", zorder=6)
    else:
        ax.contour(lat[mlat], p_hpa, psi[:, mlat], levels=[0], colors="0.3", linewidths=0.7)
    ax.set_ylim(1000, 100); ax.set_yscale("log")
    ax.set_yticks([1000, 850, 700, 500, 300, 200, 100]); ax.set_yticklabels([1000, 850, 700, 500, 300, 200, 100])
    ax.set_xlabel("latitude"); ax.set_ylabel("pressure (hPa)")
    ax.set_xlim(-45, 45); ax.set_xticks([-45, -30, -15, 0, 15, 30, 45]); ax.axvline(0, color="0.6", lw=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(("Ψ anomaly" if anom else "Ψ") + "  (10¹⁰ kg s⁻¹)", fontsize=9)
    for t in (texts or []):
        ax.text(t["x"], t["y"], t["s"], ha="center", va="top", fontsize=t["size"],
                color=t["color"], zorder=8, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.75", alpha=0.85, lw=0.6))
    if psi_abs is not None:
        fig.text(0.5, 0.005, "colour = Ψ′ anomaly (vs ERA5 clim)   ·   black contours = absolute Ψ (10¹⁰ kg s⁻¹)   ·   arrows = vertical motion (up = ascent; length ∝ strength)",
                 ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def _stage_mmsf(jl, frame_id, out, psi_anom, psi_abs, p_hpa, lat, title, vlim,
                fill_colors, texts=None):
    """Serialize one frame for the Julia rasterizer (mjo_render.jl spec)."""
    mlat = (lat >= -45) & (lat <= 45)
    LATg = lat[mlat]
    clev = [float(v) for v in range(-100, 101, 5) if v != 0]
    wvert = np.gradient(psi_abs[:, mlat], LATg, axis=1) / np.cos(np.deg2rad(LATg))[None, :]
    wn = wvert / (np.nanpercentile(np.abs(wvert), 96) or 1.0)
    wn[np.abs(wn) < 0.12] = np.nan
    sj = max(1, LATg.size // 22)
    LATm, Pm = np.meshgrid(LATg, p_hpa)
    xs, ys, ws = LATm[:, ::sj].ravel(), Pm[:, ::sj].ravel(), wn[:, ::sj].ravel()
    m = np.isfinite(ws)
    xs, ys, ws = xs[m], ys[m], ws[m]
    av = ys * (10.0 ** (-0.09 * ws) - 1.0)      # uniform visual length on log-p
    latfield = np.broadcast_to(LATg[None, :], (p_hpa.size, LATg.size))
    jl.stage(frame_id, out,
             arrays=dict(lat=LATg, p=p_hpa, za=psi_anom[:, mlat],
                         zabs=psi_abs[:, mlat], latf=latfield,
                         ax=xs, ay=ys, au=np.zeros_like(av), av=av),
             meta=dict(
                 out_png="", figsize=[10, 5.2], title=title,
                 footer=("colour = Ψ′ anomaly (vs ERA5 clim)   ·   black contours = absolute Ψ "
                         "(10¹⁰ kg s⁻¹)   ·   arrows = vertical motion (up = ascent; length ∝ strength)"),
                 xlabel="latitude", ylabel="pressure (hPa)",
                 ylog=True, yreversed=True,
                 xlim=[-45, 45], ylim=[1000, 100],
                 xticks=[-45, -30, -15, 0, 15, 30, 45],
                 xticklabels=["-45", "-30", "-15", "0", "15", "30", "45"],
                 yticks=[1000, 850, 700, 500, 300, 200, 100],
                 fill=dict(npz="za", x="lat", y="p",
                           levels=[float(v) for v in np.linspace(-vlim, vlim, 21)],
                           colors=fill_colors,
                           cbar_label="Ψ anomaly  (10¹⁰ kg s⁻¹)"),
                 contours=[dict(npz="zabs", levels=[c for c in clev if c < 0],
                                color="#000000", width=1.0, dash=True),
                           dict(npz="zabs", levels=[c for c in clev if c > 0],
                                color="#000000", width=1.0, dash=False),
                           dict(npz="zabs", levels=[0.0], color="#000000", width=0.9, dash=False),
                           dict(npz="latf", levels=[0.0], color="#999999", width=0.6, dash=False)],
                 arrows=dict(x="ax", y="ay", u="au", v="av", scale=1.0),
                 texts=list(texts or [])))


def build_anim(hist: xr.DataArray, anim_dir: Path, manifest: Path, static_out: Path) -> int:
    """Re-render the last MAXN cycles as Ψ′ frames with a common colour scale."""
    if not CLIM.exists():
        print(f"  clim coeffs {CLIM} missing — run build_mmsf_clim.py first; skipping anim.", file=sys.stderr)
        return 1
    coeffs = xr.open_dataarray(CLIM)
    lat = hist.latitude.values
    coeffs = coeffs.interp(latitude=lat)                    # align clim grid to the analysis grid
    assert list(hist.level.values) == LEVELS, \
        f"history level axis {list(hist.level.values)} != LEVELS — stale mmsf_vbar_history.nc?"
    p_pa = np.asarray(LEVELS, float) * 100.0
    st_all = pd.to_datetime(hist.time.values)
    vbar_all = hist.values                                  # (time, lev, lat)

    # Weekly mean: each frame is the TRAILING 7-day mean of the zonal-mean v (Ψ is linear in
    # v, so this equals the 7-day-mean Ψ). The day-to-day wobble is noise; smoothing it makes
    # the Hadley/Ferrel cell-strength EVOLUTION legible. Only emit a frame once ≥6 days of
    # history precede it, so every frame is a near-full weekly mean.
    WIN, MINSPAN = pd.Timedelta(days=7), pd.Timedelta(days=6)
    st, vbar_wk = [], []
    for i in range(len(st_all)):
        w = (st_all > st_all[i] - WIN) & (st_all <= st_all[i])
        if st_all[i] - st_all[w][0] < MINSPAN:             # window not yet ~a week wide
            continue
        st.append(st_all[i]); vbar_wk.append(vbar_all[w].mean(axis=0))
    if not st:                                             # <6 days of history: fall back to raw
        st, vbar_wk = list(st_all), [vbar_all[i] for i in range(len(st_all))]
    st = pd.DatetimeIndex(st)

    # pass 1: absolute Ψ + anomaly per (weekly-mean) frame → common vlim (steady animation)
    psia, psip, hstr = [], [], []
    for i in range(len(st)):
        psi = streamfunction(vbar_wk[i], p_pa, lat)
        psia.append(psi)
        # frames are TRAILING 7-day means, so evaluate the harmonic clim at the
        # window MIDPOINT (end − 3.5 d) — at the end-day it lags the seasonal
        # cycle by half a window and leaks a spurious anomaly in transitions.
        pc = clim_psi(coeffs, (st[i] - pd.Timedelta(days=3.5)).dayofyear)
        psip.append(psi - pc)
        hstr.append(hadley_strength(psi, pc, p_pa / 100, lat,
                                    doy=int((st[i] - pd.Timedelta(days=3.5)).dayofyear)))
    mlat = (lat >= -45) & (lat <= 45)
    vlim = float(np.nanpercentile(np.abs(np.stack(psip)[:, :, mlat]), 99.0)) or 1.0

    anim_dir = Path(anim_dir); anim_dir.mkdir(parents=True, exist_ok=True)
    for old in anim_dir.glob("F*.webp"):
        old.unlink()

    # Julia fast-path (MJO_RENDERER=julia) — same shape as walker.build_anim
    try:
        import julia_mjo as _jl
    except ImportError:
        _jl = None
    use_jl = _jl is not None and _jl.available()
    jstat: dict = {}
    if use_jl:
        _jl.reset()
        fill_colors = _jl.cmap_hex("RdBu_r", 20)
        for i in range(len(st)):
            _stage_mmsf(_jl, f"F{i:02d}", anim_dir / f"F{i:02d}.webp",
                        psip[i], psia[i], p_pa / 100, lat,
                        f"Meridional mass streamfunction anomaly Ψ′ — 7-day mean ending {st[i]:%Y-%m-%d}",
                        vlim, fill_colors, texts=strength_texts(hstr[i]))
        _stage_mmsf(_jl, "Fstatic", Path(static_out),
                    psip[-1], psia[-1], p_pa / 100, lat,
                    f"Meridional mass streamfunction anomaly Ψ′ — 7-day mean ending {st[-1]:%Y-%m-%d}",
                    vlim, fill_colors, texts=strength_texts(hstr[-1]))
        jstat = _jl.render()

    frames = []
    for i in range(len(st)):
        fp = anim_dir / f"F{i:02d}.webp"
        if not jstat.get(f"F{i:02d}"):
            plot_psi(psip[i], p_pa / 100, lat, fp,
                     f"Meridional mass streamfunction anomaly Ψ′ — 7-day mean ending {st[i]:%Y-%m-%d}",
                     anom=True, vlim=vlim, psi_abs=psia[i], texts=strength_texts(hstr[i]))
        frames.append({"idx": i, "file": fp.name, "date": f"{st[i]:%Y-%m-%d}",
                       "label": f"week ending {st[i]:%a %b %d, %Y}",
                       "hadley": hstr[i]})
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "days": len(frames),
            "hadley_latest": hstr[-1], "regions": {"mmsf": {
        "label": "Meridional mass streamfunction anomaly (7-day mean, Ψ′ vs ERA5 1991-2020)",
        "n_frames": len(frames), "frames": frames}}}
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(mani))
    # static "latest" for the card thumbnail
    if not jstat.get("Fstatic"):
        plot_psi(psip[-1], p_pa / 100, lat, Path(static_out),
                 f"Meridional mass streamfunction anomaly Ψ′ — 7-day mean ending {st[-1]:%Y-%m-%d}",
                 anom=True, vlim=vlim, psi_abs=psia[-1], texts=strength_texts(hstr[-1]))
    print(f"  Ψ′ range {np.nanmin(psip[-1]):.1f} … {np.nanmax(psip[-1]):.1f} ×10¹⁰ kg/s "
          f"(vlim ±{vlim:.0f}); wrote {len(frames)} frames + manifest")
    for k, nm in (("nh", "Northern"), ("sh", "Southern")):
        d = hstr[-1].get(k)
        if not d:
            print(f"  {nm} Hadley cell: not resolved this frame"); continue
        vs = (f"{d['pct_of_normal']:.0f}% of normal" if d.get("pct_of_normal") is not None
              else f"anom {d['anom']:+.1f}" if d.get("anom") is not None else "no normal")
        vs += "" if d.get("sigma") is None else f" ({d['sigma']:+.1f}σ)"
        vs += f" [{d.get('normal_basis')}]"
        print(f"  {nm} Hadley cell: |Ψ|max {abs(d['psi']):.1f} vs normal "
              f"{abs(d['normal']):.1f} ×10¹⁰ kg/s = {vs} "
              f"(core {d['core_lat']:+.1f}°, {d['core_p']:.0f} hPa; "
              f"edges {d['edges'][0]:+.0f}…{d['edges'][1]:+.0f}°)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/mmsf")
    ap.add_argument("--anim-dir", default="assets/sst/anim/mmsf")
    ap.add_argument("--manifest", default="assets/sst/anim/mmsf_manifest.json")
    ap.add_argument("--out", default="assets/sst/mmsf_anom.webp")
    args = ap.parse_args()
    dd = Path(args.data_dir); dd.mkdir(parents=True, exist_ok=True)
    vbar, p_pa, lat = zonal_mean_v(download_v(args.date, args.time, dd))
    valid = datetime.strptime(f"{args.date}{args.time}", "%Y%m%d%H")
    hist = update_history(vbar, p_pa, lat, valid)
    hist = backfill_history_gaps(hist)
    print(f"  history: {hist.time.size} cycles "
          f"({pd.to_datetime(hist.time.values[0]):%Y-%m-%d} … {valid:%Y-%m-%d %HZ})")
    return build_anim(hist, Path(args.anim_dir), Path(args.manifest), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
