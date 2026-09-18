#!/usr/bin/env python3
"""
Convectively coupled Kelvin wave tracker: where the waves are, on a map, through the AIFS-ENS forecast.

The record is 200 hPa velocity potential, the cleanest large-scale footprint of a coupled Kelvin
wave (upper-level divergence over the active convection, convergence over the suppressed phase):

  analysis   every AIFS-ENS 0-h analysis in the rolling tropical χ history (walker_chi; 00 and
             12 UTC averaged per day, |lat| <= 32 deg on the lmax-42 grid)
  forecast   the 50-member ensemble-mean u, v at 200 hPa for each daily step to day 15, turned into
             χ with the same solver on the same grid (the fields wind200_vpot has already fetched)
  anomaly    against the ERA5 1991-2020 harmonic climatology and trend (walker_chi_clim.nc)

The joined record is Kelvin-filtered in wavenumber-frequency space at every latitude (Wheeler and
Kiladis 1999; the site's OLR trackers use the same wedge): eastward zonal wavenumbers 1-14,
periods 2.5-20 days, equivalent depths 8-90 m. The end of the record is padded with a 10-day ramp
to zero so the filter's taper eats the padding, not the forecast. The ensemble mean is used
deliberately: the filtered field of a mean is the part of the wave the members agree on.

Outputs
  assets/sst/anim/kelvin/F*.webp + kelvin_manifest.json   one map per day, D-10 .. D+15
  assets/sst/kelvin_hov.webp                                Hovmoller 5S-5N, 45 days + forecast
  assets/sst/data/kelvin_tracker.json                       crest longitudes by day

    python kelvin_tracker.py --selftest
    python kelvin_tracker.py --date 20260918 --time 12
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "sst"))
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))

G, A_EARTH, DAY = 9.81, 6.371e6, 86400.0
KMIN, KMAX, PMIN, PMAX, HMIN, HMAX = 1, 14, 2.5, 20.0, 8.0, 90.0
CLIM = HERE.parent / "data" / "reference" / "walker_chi_clim.nc"
MAP_DAYS_BACK, HOV_DAYS_BACK = 10, 45
LAT_MAP = 25.0
TRAP_LAT = 15.0                  # Kelvin waves are equatorially trapped: the filtered field fades out poleward of this
VP_CMAP = LinearSegmentedColormap.from_list("vpot", ["#1b5e20", "#43a047", "#86c98a", "#cfe8cf", "#ffffff",
                                                     "#fbe2bd", "#f0a64b", "#df6a1e", "#a8330f"])


# ---- the filter ------------------------------------------------------------------------------ #
def kelvin_mask(nt: int, nlon: int) -> np.ndarray:
    """Boolean (freq, wavenumber) mask for the Kelvin wedge, in numpy's fft2 layout.

    numpy's inverse transform builds exp(+2 pi i (f t + s x)); a crest moves east when f and s have
    OPPOSITE signs, so the eastward zonal wavenumber is k = -s for f > 0 (and k = s for f < 0)."""
    f = np.fft.fftfreq(nt, d=1.0)                          # cycles per day
    s = np.fft.fftfreq(nlon, d=1.0 / nlon)                 # planetary wavenumber
    F, S = np.meshgrid(f, s, indexing="ij")
    k = np.where(F > 0, -S, S)
    af = np.abs(F)
    band = (k >= KMIN) & (k <= KMAX) & (af >= 1.0 / PMAX) & (af <= 1.0 / PMIN)
    # dispersion: f = sqrt(g h) * k / (2 pi a) per second -> per day
    fk = lambda h: np.sqrt(G * h) * k / (2 * np.pi * A_EARTH) * DAY           # noqa: E731
    return band & (af >= fk(HMIN)) & (af <= fk(HMAX)) & (F != 0)


def kelvin_filter(x: np.ndarray, pad: int = 60, ramp: int = 10) -> np.ndarray:
    """x (time, lat, lon) -> Kelvin-filtered, same shape. Time mean removed per point; the end is padded
    with a `ramp`-day slide to zero, then zeros, and the padding is dropped after filtering."""
    nt, nl, nx = x.shape
    x = x - x.mean(0, keepdims=True)
    r = x[-1][None] * (1 - np.arange(1, ramp + 1) / (ramp + 1))[:, None, None]
    xp = np.concatenate([x, r, np.zeros((pad - ramp, nl, nx))], 0)
    # soft start too: a 10% cosine taper on the first days only
    n0 = max(2, int(0.1 * nt))
    xp[:n0] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(n0) / n0))[:, None, None]
    m = kelvin_mask(xp.shape[0], nx)
    out = np.empty_like(xp)
    for j in range(nl):
        out[:, j] = np.real(np.fft.ifft2(np.fft.fft2(xp[:, j]) * m))
    return out[:nt]


def selftest() -> int:
    nt, nx = 150, 144
    t = np.arange(nt)[:, None]; lon = np.arange(nx)[None, :] * 2 * np.pi / nx
    east = np.cos(5 * lon - 2 * np.pi * t / 8.0)           # k=5, 8-day period, eastward: h ~ 14 m, in the band
    west = np.cos(5 * lon + 2 * np.pi * t / 8.0)           # same, westward
    slow = np.cos(2 * lon - 2 * np.pi * t / 45.0)          # eastward but MJO-slow: outside the band
    for name, x, keep in (("eastward k=5 8 d", east, True), ("westward k=5 8 d", west, False), ("eastward k=2 45 d", slow, False)):
        y = kelvin_filter(x[:, None, :].astype(float))[:, 0]
        r = np.std(y[20:-20]) / np.std(x[20:-20])
        ok = r > 0.7 if keep else r < 0.1
        print(f"  {name}: kept {r:.2f} of the amplitude -> {'ok' if ok else 'FAIL'}")
        if not ok:
            return 1
    print("selftest passed")
    return 0


# ---- the record ------------------------------------------------------------------------------ #
def chi_anomaly(chi: np.ndarray, times: pd.DatetimeIndex, clim: xr.Dataset, lat, lon) -> np.ndarray:
    """chi (t, lat, lon) at 200 hPa minus the harmonic climatology and the 1991-2020 trend."""
    c = clim.sel(level=200)
    if not np.allclose(c.latitude.values, lat):
        raise SystemExit("walker_chi_clim.nc is not on the chi history's grid")
    nx = len(lon)                                            # the history's 360-degree duplicate column is already dropped
    coef = c.chi_coef.values[..., :nx]
    slope = c.chi_slope.values[..., :nx]
    doy = times.dayofyear.values; ang = 2 * np.pi * doy / 365.25
    B = np.stack([np.ones_like(ang), np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)], -1)
    cl = np.einsum("th,hij->tij", B, coef)
    yr = times.year.values + (doy - 1) / 365.25
    return chi - cl - slope[times.month.values - 1] * (yr - 2005.5)[:, None, None]


def analysis_days(hist: xr.DataArray) -> xr.DataArray:
    """00 and 12 UTC analyses -> one field per day (the day's mean)."""
    h = hist.sel(level=200)
    return h.groupby(h.time.dt.floor("D")).mean("time").rename({"floor": "time"})


def forecast_chi(date: str, hh: str, lat_ref, lon_ref):
    """Daily 00 UTC ensemble-mean chi200 from the AIFS-ENS forecast, on the history's grid."""
    import walker_chi as WC
    import download_aifs
    from wind200_vpot import _ens_mean
    import store as ecmwf
    steps = list(download_aifs.rmm_steps(hh))
    ds = _ens_mean(ecmwf.Cycle(date, hh), steps, steps)
    init = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]}T{hh}:00")
    out, days = [], []
    for i, sh in enumerate((ds.step / np.timedelta64(1, "h")).round().astype(int).values):
        if sh == 0:
            continue
        chi, la, lo = WC.chi_strip(ds["u"].isel(step=i), ds["v"].isel(step=i))
        if not (np.allclose(la, lat_ref) and np.allclose(lo, lon_ref)):
            chi = xr.DataArray(chi, dims=("latitude", "longitude"), coords={"latitude": la, "longitude": lo}).interp(latitude=lat_ref, longitude=lon_ref).values
        out.append(chi); days.append((init + pd.Timedelta(hours=int(sh))).normalize())
    return pd.DatetimeIndex(days), np.array(out)


# ---- rendering ------------------------------------------------------------------------------- #
def crests(kf_eq: np.ndarray, lon: np.ndarray, thresh: float) -> list[float]:
    """Longitudes of the active (negative, divergent) Kelvin crests along the equator: local minima below -thresh."""
    x = np.r_[kf_eq[-1], kf_eq, kf_eq[0]]
    out = []
    for i in range(1, len(x) - 1):
        if x[i] < -thresh and x[i] <= x[i - 1] and x[i] <= x[i + 1]:
            out.append(float(lon[i - 1]))
    return out


def render_map(kf, anom, lat, lon, day, tag, levels, init, out: Path, thresh):
    import mapstyle as MS
    fig, ax, H, pc = MS.open_map(extent=[-180, 180, -LAT_MAP, LAT_MAP], central=180, width=15.0, top=0.80, bot=0.95)
    norm = BoundaryNorm(levels, VP_CMAP.N, extend="both")
    cf = ax.contourf(lon, lat, kf / 1e6, levels=levels, cmap=VP_CMAP, norm=norm, extend="both", transform=pc, zorder=1)
    st = max(1.0, float(np.ceil(np.nanpercentile(np.abs(anom), 95) / 1e6 / 4)))
    cl = np.arange(-8 * st, 8 * st + 0.1, st); cl = cl[cl != 0]
    ax.contour(lon, lat, anom / 1e6, levels=cl, colors="#3b3b3b", linewidths=0.55, alpha=0.55, transform=pc, zorder=4)
    MS.features(ax, states=False, gridlines=False)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#6f6b64", alpha=0.5, xlocs=range(-180, 181, 30), ylocs=[-20, -10, 0, 10, 20], zorder=4)
    gl.top_labels = gl.right_labels = False; gl.xlabel_style = gl.ylabel_style = {"size": 7.5, "color": "#444"}
    ax.plot([-180, 180], [0, 0], color="#6f6b64", lw=0.5, ls=(0, (4, 4)), transform=pc, zorder=4)
    j0 = int(np.argmin(np.abs(lat)))
    for c in crests(kf[max(0, j0 - 2):j0 + 3].mean(0), lon, thresh):
        ax.annotate("K", xy=(c, 0), xycoords=pc._as_mpl_transform(ax), ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white", zorder=9, bbox=dict(boxstyle="circle,pad=0.25", fc="#1b5e20", ec="white", lw=1.2))
    kind = "analysis" if tag == "analysis" else f"forecast {tag}"
    MS.heading(fig, H, f"AIFS-ENS · Kelvin waves in 200 hPa velocity potential · {day:%a %d %b %Y} ({kind})",
               f"shading: Kelvin-filtered χ200 anomaly (green = upper-level divergence over the ACTIVE, rainy phase; brown = suppressed) · "
               f"K = active crest on the equator · grey contours: unfiltered χ200 anomaly every {st:g}×10⁶ m² s⁻¹ · init {init:%Y-%m-%d %H} UTC, ensemble mean",
               title_size=12.5, sub_size=8.2, wrap=190)
    cb = MS.colorbar(fig, H, cf, "Kelvin-filtered χ200 anomaly (10⁶ m² s⁻¹)")
    cb.set_ticks([x for x in levels if abs(x / (levels[-1] / 6)) >= 1 - 1e-6]); cb.ax.tick_params(labelsize=7.5)
    MS.save(fig, out, dpi=118)


def render_hov(kf_eq, anom_eq, lon, days, init_day, levels, out: Path):
    fig = plt.figure(figsize=(11.5, 10.5))
    ax = fig.add_axes([0.075, 0.12, 0.83, 0.74])
    lon_p = np.r_[lon, lon[0] + 360]
    kfp = np.c_[kf_eq, kf_eq[:, :1]]; anp = np.c_[anom_eq, anom_eq[:, :1]]
    y = np.arange(len(days))
    norm = BoundaryNorm(levels, VP_CMAP.N, extend="both")
    cf = ax.contourf(lon_p, y, kfp / 1e6, levels=levels, cmap=VP_CMAP, norm=norm, extend="both")
    st = max(1.0, float(np.ceil(np.nanpercentile(np.abs(anp), 95) / 1e6 / 4)))
    cl = np.arange(-8 * st, 8 * st + 0.1, st); cl = cl[cl != 0]
    ax.contour(lon_p, y, anp / 1e6, levels=cl, colors="#3b3b3b", linewidths=0.5, alpha=0.5)
    i0 = int(np.flatnonzero(days <= init_day)[-1])
    ax.axhline(i0 + 0.5, color="#1c2430", lw=1.8)
    lab = dict(fontsize=9, fontweight="bold", color="#1c2430", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.text(357, i0 + 0.1, "▲ analysis", ha="right", va="bottom", **lab)
    ax.text(357, i0 + 0.9, "▼ forecast (AIFS-ENS ensemble mean)", ha="right", va="top", **lab)
    ax.set_ylim(len(days) - 0.5, -0.5)                      # time runs DOWN, the usual Hovmoller
    ticks = [i for i, d in enumerate(days) if d.day in (1, 8, 15, 22)]
    ax.set_yticks(ticks); ax.set_yticklabels([days[i].strftime("%b %-d") for i in ticks], fontsize=9)
    ax.set_xticks(range(0, 361, 60)); ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"], fontsize=9)
    ax.set_xlim(0, 360)
    for lo_, name in ((75, "Indian Ocean"), (120, "Maritime Cont."), (165, "W Pacific"), (240, "E Pacific"), (325, "Atlantic")):
        ax.text(lo_, -0.9, name, ha="center", va="bottom", fontsize=8, color="#6f6b64", clip_on=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.075, 0.975, "AIFS-ENS · Kelvin waves along the equator (5°S–5°N), analysis and 15-day forecast",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.075, 0.948, textwrap.fill("Shading: Kelvin-filtered 200 hPa velocity potential anomaly; green is upper-level divergence over the "
             "active, rainy phase. A coupled Kelvin wave is a green streak tilting down to the right, moving east at 10–20 m/s "
             "(about 1,000–1,700 km a day). Grey contours: the unfiltered anomaly, whose broad slow pattern is the MJO and the ENSO "
             "background.", 150), fontsize=8.6, color="#444", va="top", linespacing=1.3)
    cax = fig.add_axes([0.925, 0.12, 0.018, 0.74])
    cb = fig.colorbar(cf, cax=cax, extend="both", ticks=[x for x in levels if abs(x / (levels[-1] / 6)) >= 1 - 1e-6])
    cb.set_label("Kelvin-filtered χ200 anomaly (10⁶ m² s⁻¹)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.text(0.075, 0.03, textwrap.fill("Filter: eastward wavenumbers 1–14, periods 2.5–20 days, equivalent depths 8–90 m (Wheeler & Kiladis 1999). "
             "Analysis: AIFS-ENS 0-h analyses, 00 and 12 UTC averaged. Anomalies vs ERA5 1991–2020. Averaged 5°S–5°N.", 170),
             fontsize=7.8, color="#6f6b64", linespacing=1.3)
    fig.savefig(out, dpi=118, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--time", default="00")
    ap.add_argument("--history", default=str(REPO / "assets/sst/anim/walker/walker_chi_history.nc"))
    ap.add_argument("--anim-dir", default=str(REPO / "assets/sst/anim/kelvin"))
    ap.add_argument("--manifest", default=str(REPO / "assets/sst/anim/kelvin_manifest.json"))
    ap.add_argument("--hov", default=str(REPO / "assets/sst/kelvin_hov.webp"))
    ap.add_argument("--json", default=str(REPO / "assets/sst/data/kelvin_tracker.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if selftest():
        raise SystemExit("filter self-test failed; not rendering")
    import walker_chi as WC
    hist = WC.load_history(Path(a.history))
    if hist is None or hist.sizes["time"] < 20:
        print("  kelvin: no usable chi history; skipping"); return 0
    lat, lon = hist.latitude.values, hist.longitude.values
    dup = np.isclose(lon[-1] - lon[0], 360.0)
    ad = analysis_days(hist)
    init_day = pd.Timestamp(f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:]}")
    ad = ad.sel(time=slice(None, init_day))
    fdays, fchi = forecast_chi(a.date, a.time, lat, lon)
    days = pd.DatetimeIndex(list(pd.DatetimeIndex(ad.time.values)) + [d for d in fdays if d > init_day])
    chi = np.concatenate([ad.values, fchi[[i for i, d in enumerate(fdays) if d > init_day]]], 0)
    full = pd.date_range(days[0], days[-1], freq="D")                        # fill any missing day by interpolation
    da = xr.DataArray(chi, dims=("time", "latitude", "longitude"), coords={"time": days, "latitude": lat, "longitude": lon})
    da = da.reindex(time=full).interpolate_na("time", limit=3).dropna("time", how="all")
    days = pd.DatetimeIndex(da.time.values)
    if dup:
        da = da.isel(longitude=slice(0, -1)); lon = lon[:-1]
    clim = xr.open_dataset(CLIM)
    anom = chi_anomaly(da.values, days, clim, lat, lon)
    kf = kelvin_filter(anom)
    n_an = int((days <= init_day).sum())
    print(f"  kelvin: {n_an} analysis days {days[0]:%Y-%m-%d}..{init_day:%Y-%m-%d} + {len(days) - n_an} forecast days", flush=True)
    # the filter picks up subtropical noise that is not a Kelvin wave: fade it beyond TRAP_LAT
    trap = np.exp(-(np.clip(np.abs(lat) - TRAP_LAT, 0, None) / 5.0) ** 2)
    kf = kf * trap[None, :, None]
    vtop = float(np.nanpercentile(np.abs(kf[-60:]), 99.5) / 1e6)
    step = next(x for x in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0) if x * 6 >= vtop)
    levels = np.r_[-np.arange(6, 0, -1) * step, -step / 2, step / 2, np.arange(1, 7) * step]
    thresh = 2.0 * step * 1e6
    anim = Path(a.anim_dir); anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    frames, crest_log = [], {}
    idx = [i for i, d in enumerate(days) if d >= init_day - pd.Timedelta(days=MAP_DAYS_BACK)]
    init = pd.Timestamp(f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:]}T{int(a.time):02d}:00")
    for n, i in enumerate(idx):
        d = days[i]; lead = (d - init_day).days
        tag = "analysis" if lead <= 0 else f"+{lead} d"
        fp = anim / f"F{n:02d}.webp"
        render_map(kf[i], anom[i], lat, lon, d, tag, levels, init, fp, thresh)
        j0 = int(np.argmin(np.abs(lat)))
        crest_log[f"{d:%Y-%m-%d}"] = [round(c, 1) for c in crests(kf[i][max(0, j0 - 2):j0 + 3].mean(0), lon, thresh)]
        frames.append({"idx": n, "file": fp.name, "date": f"{d:%Y-%m-%d}",
                       "label": (f"{d:%a %b %d} · analysis" if lead <= 0 else f"{d:%a %b %d} · day +{lead}")})
    mani = {"ver": int(_time.time()), "start": next(k for k, f in enumerate(frames) if f["date"] == f"{init_day:%Y-%m-%d}"),
            "regions": {"kelvin": {"label": "Kelvin waves, 200 hPa velocity potential (AIFS-ENS)", "n_frames": len(frames), "frames": frames}}}
    Path(a.manifest).write_text(json.dumps(mani))
    eq = (np.abs(lat) <= 5.5)
    hs = [i for i, d in enumerate(days) if d >= init_day - pd.Timedelta(days=HOV_DAYS_BACK)]
    render_hov(kf[hs][:, eq].mean(1), anom[hs][:, eq].mean(1), lon, days[hs], init_day, levels, Path(a.hov))
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps({"init": f"{init:%Y-%m-%d %H}", "analysis_days": n_an, "first_day": f"{days[0]:%Y-%m-%d}",
                                        "units": "longitude (deg E) of active Kelvin crests on the equator", "crests": crest_log}, separators=(",", ":")))
    print(f"  kelvin: {len(frames)} map frames, Hovmoller, crests today {crest_log.get(f'{init_day:%Y-%m-%d}')}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    # outputs are written; skip interpreter teardown (a C-extension destructor segfaults at exit
    # under the cartopy 0.25 pin, as in ar_monitor, and turned a complete run into a warning)
    sys.stdout.flush(); sys.stderr.flush()
    import os
    os._exit(rc or 0)
