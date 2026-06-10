#!/usr/bin/env python3
"""Wheeler-Kiladis wave decomposition of equatorial OLR.

Space-time (wavenumber-frequency) bandpass filtering of the 15°S-15°N OLR anomaly into the
canonical convectively-coupled equatorial wave bands — MJO, Kelvin, equatorial Rossby (ER)
and a low-frequency background — then overlays each filtered convective signal as contours
on the recent OLR Hovmöller. Eastward/westward selectivity and the Kelvin/ER equivalent-depth
dispersion bounds (8-90 m) follow Wheeler & Kiladis (1999) / CPC's operational definitions.

Run the self-test first — it pushes synthetic eastward/westward waves through the filters and
asserts the direction selectivity, which is the part that's easy to get backwards:

    python scripts/sst/olr_waves.py --selftest
    python scripts/sst/olr_waves.py --hist 2021-06-01 --days 150   # historical demo render
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

HERE = Path(__file__).resolve().parent
HIST = HERE / "data" / "olr_history.nc"         # full 1979-2022 backbone (gitignored, local)
CLIM = HERE / "metar" / "olr_clim.nc"           # committed daily climatology (for real-time)
STORE_RT = HERE / "metar" / "olr_rt.nc"         # committed rolling GMGSI store

# physical constants (equatorial beta-plane shallow water)
G = 9.81
A_EARTH = 6.371e6
BETA = 2.0 * 7.292e-5 / A_EARTH          # df/dy at the equator
DAY = 86400.0

# Wave bands: (kmin, kmax) planetary zonal wavenumber (+ eastward, − westward),
# (pmin, pmax) period days, (hmin, hmax) equivalent depth m (None = no dispersion bound).
# colours follow Carl Schreck's NCICS "OLR with CFS forecasts" convention.
# clev = ± W/m² contour level (LF/ER are broad, lower-amplitude fields → lower threshold).
WAVES = {
    "Kelvin": dict(k=(1, 14),   p=(2.5, 30), h=(8, 90),   n=None, color="#1f57d8", lw=1.6, clev=13),
    "ER":     dict(k=(-10, -1), p=(9.7, 48), h=(8, 90),   n=1,    color="#e8000b", lw=1.6, clev=11),
    "MJO":    dict(k=(1, 5),    p=(30, 96),  h=None,      n=None, color="#000000", lw=1.8, clev=13),
    "LF":     dict(k=(-3, 3),   p=(120, 1e9), h=None,     n=None, color="#9b30d0", lw=1.8, clev=8,
                   label="Low"),               # all periods >120 d incl. interannual ENSO standing envelope
}


# ----------------------------------------------------------------------
# dispersion helpers (frequency in cycles/day, s = planetary wavenumber)
# ----------------------------------------------------------------------
def kelvin_he(s: float, f: float) -> float:
    """Equivalent depth implied by a Kelvin wave at (s>0, f>0): ω = (s/a)·√(g·he)."""
    if s <= 0 or f <= 0:
        return np.nan
    omega = 2 * np.pi * f / DAY
    c = omega / (s / A_EARTH)
    return c * c / G


def er_freq(s: int, he: float, n: int = 1) -> float:
    """|frequency| (cyc/day) of the n-mode equatorial-Rossby wave at wavenumber s, depth he.
    Solves the nondim cubic ν³−(K²+2n+1)ν−K=0 and takes the Rossby (smallest-|ν|) root."""
    c = np.sqrt(G * he)
    k_dim = s / A_EARTH
    K = k_dim * np.sqrt(c / BETA)
    roots = np.roots([1.0, 0.0, -(K * K + 2 * n + 1), -K])
    real = roots[np.abs(roots.imag) < 1e-9].real
    nu = real[np.argmin(np.abs(real))]               # Rossby branch
    omega = nu * np.sqrt(BETA * c)
    return abs(omega) / (2 * np.pi) * DAY


# precompute ER frequency band edges per westward wavenumber
_ER_BOUNDS = {}
for _s in range(-20, 0):
    _f = sorted(er_freq(_s, h) for h in (8.0, 90.0))
    _ER_BOUNDS[_s] = (_f[0], _f[1])


def in_band(s: float, f: float, w: dict) -> bool:
    """Is the positive-frequency physical wave (s, f>0) inside band w?"""
    if f <= 0:
        return False
    kmin, kmax = w["k"]; pmin, pmax = w["p"]
    if not (kmin <= s <= kmax):
        return False
    period = 1.0 / f
    if not (pmin <= period <= pmax):
        return False
    if w["h"] is not None:
        hmin, hmax = w["h"]
        if w["n"] is None:                           # Kelvin: depth from (s,f)
            he = kelvin_he(s, f)
            return hmin <= he <= hmax
        si = int(round(s))                           # ER: frequency band per wavenumber
        if si not in _ER_BOUNDS:
            return False
        flo, fhi = _ER_BOUNDS[si]
        return flo <= f <= fhi
    return True


# ----------------------------------------------------------------------
# the filter
# ----------------------------------------------------------------------
def _taper(a: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Split-cosine-bell taper over the first/last `frac` of the time axis (axis 0)."""
    nt = a.shape[0]
    w = np.ones(nt)
    m = max(1, int(frac * nt))
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 1) / (m + 1)))
    w[:m] = ramp; w[-m:] = ramp[::-1]
    return a * w[:, None]


def wk_filter(anom: np.ndarray, wave: str) -> np.ndarray:
    """Bandpass `anom` (ntime × nlon, detrended) to one wave band via 2-D FFT.
    numpy convention: cell (m,n) reconstructs exp(2πi(f_m t + s_n x/nx)); the physical
    wave exp(i(sλ − 2π f t)) therefore sits at s_phys = wavenum[n], f_phys = −freq_t[m].
    A cell is kept iff its positive-frequency representative is in the band (the conjugate
    cell maps to the same representative, so the mask is Hermitian → real inverse)."""
    w = WAVES[wave]
    nt, nx = anom.shape
    F = np.fft.fft2(_taper(anom))
    ft = np.fft.fftfreq(nt, d=1.0)                   # cycles/day
    sx = np.fft.fftfreq(nx, d=1.0) * nx              # planetary wavenumber (integers)
    S = np.broadcast_to(sx[None, :], (nt, nx)).astype(float).copy()
    Fr = np.broadcast_to(-ft[:, None], (nt, nx)).astype(float).copy()
    neg = Fr < 0                                     # fold every cell to positive frequency
    S[neg] *= -1; Fr[neg] *= -1
    with np.errstate(divide="ignore", invalid="ignore"):
        period = np.where(Fr > 0, 1.0 / Fr, np.inf)
    kmin, kmax = w["k"]; pmin, pmax = w["p"]
    m = (S >= kmin) & (S <= kmax) & (period >= pmin) & (period <= pmax) & (Fr > 0)
    if w["h"] is not None:
        hmin, hmax = w["h"]
        if w["n"] is None:                           # Kelvin: depth from (s,f)
            with np.errstate(divide="ignore", invalid="ignore"):
                c = (2 * np.pi * Fr / DAY) / (S / A_EARTH)
                he = c * c / G
            m &= (he >= hmin) & (he <= hmax)
        else:                                        # ER: per-wavenumber frequency bounds
            flo = np.full((nt, nx), np.nan); fhi = np.full((nt, nx), np.nan)
            si = np.round(S).astype(int)
            for s, (lo, hi) in _ER_BOUNDS.items():
                cell = si == s
                flo[cell] = lo; fhi[cell] = hi
            m &= (Fr >= flo) & (Fr <= fhi)
    return np.fft.ifft2(F * m.astype(float)).real


def _anomaly(ds: xr.Dataset, band: str = "15") -> xr.DataArray:
    """OLR anomaly = OLR − smooth daily climatology (by dayofyear)."""
    olr = ds[f"olr_{band}"]
    clim = ds[f"clim_{band}"]
    doy = olr["time"].dt.dayofyear.values
    base = clim.sel(dayofyear=xr.DataArray(doy, dims="time")).values
    a = (olr.values - base)
    a = a - a.mean(0, keepdims=True)                 # remove time-mean per lon (keep low-freq fluctuations)
    return xr.DataArray(a, coords={"time": olr["time"], "lon": olr["lon"]}, dims=("time", "lon"))


# ----------------------------------------------------------------------
# self-test: synthetic waves must be selected by the right band only
# ----------------------------------------------------------------------
def selftest() -> int:
    nt, nx = 1024, 144
    t = np.arange(nt)[:, None]
    lam = (np.arange(nx) / nx)[None, :]              # 0..1 around the globe
    def wave(s, period, amp=1.0):
        return amp * np.cos(2 * np.pi * (s * lam - t / period))   # eastward if s,period>0
    ok = True
    # eastward Kelvin-like: s=+4, period 8 d → he≈... should land in Kelvin, not ER
    east = wave(4, 8.0)
    ke = np.std(wk_filter(east, "Kelvin")); er = np.std(wk_filter(east, "ER"))
    print(f"  eastward s=+4 P=8d : Kelvin={ke:.3f}  ER={er:.3f}  → {'OK' if ke > 5 * er else 'FAIL'}")
    ok &= ke > 5 * er
    # westward ER-like: s=-4 (i.e. eastward arg negative), period 20 d → ER, not Kelvin
    west = wave(-4, 20.0)
    ke2 = np.std(wk_filter(west, "Kelvin")); er2 = np.std(wk_filter(west, "ER"))
    print(f"  westward s=-4 P=20d: ER={er2:.3f}  Kelvin={ke2:.3f}  → {'OK' if er2 > 5 * ke2 else 'FAIL'}")
    ok &= er2 > 5 * ke2
    # eastward MJO: s=+2, period 50 d → MJO, not Kelvin
    mjo = wave(2, 50.0)
    mj = np.std(wk_filter(mjo, "MJO")); mk = np.std(wk_filter(mjo, "Kelvin"))
    print(f"  eastward s=+2 P=50d: MJO={mj:.3f}  Kelvin={mk:.3f}  → {'OK' if mj > 5 * mk else 'FAIL'}")
    ok &= mj > 5 * mk
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------------
# render
# ----------------------------------------------------------------------
def _anomaly_rt(rt: xr.Dataset, clim_ds: xr.Dataset, band: str = "15") -> xr.DataArray:
    """Real-time anomaly from the GMGSI store, deseasonalised with the interp_OLR climatology.
    Reindexes to a gap-free daily axis (the filter needs evenly-spaced samples) and removes the
    window-mean per longitude — which also absorbs any constant GMGSI-vs-true-OLR offset."""
    olr = rt[f"olr_{band}"]
    full = pd.date_range(pd.Timestamp(olr["time"].values[0]), pd.Timestamp(olr["time"].values[-1]), freq="D")
    df = olr.reindex(time=full).to_pandas().interpolate(axis=0, limit_direction="both")  # fill day gaps
    olr = xr.DataArray(df.values, coords={"time": full, "lon": olr["lon"].values}, dims=("time", "lon"))
    clim = clim_ds[f"clim_{band}"]
    base = clim.sel(dayofyear=xr.DataArray(olr["time"].dt.dayofyear.values, dims="time")).values
    a = olr.values - base
    a = a - a.mean(0, keepdims=True)                 # remove offset only; keep low-freq (ENSO) evolution
    return xr.DataArray(a, coords={"time": olr["time"], "lon": olr["lon"]}, dims=("time", "lon"))


def render(anom: xr.DataArray, end: datetime, days: int, out: Path,
           waves=("Kelvin", "ER", "MJO", "LF"), prelim_days: int = 0, subtitle: str = "") -> None:
    """Schreck-style global OLR-anomaly Hovmöller (5°S-5°N) with the filtered convectively-
    coupled waves overlaid as ±16 W/m² contours (solid = enhanced convection, dashed = suppressed)."""
    from scipy.ndimage import gaussian_filter
    from matplotlib.colors import BoundaryNorm
    filt = {w: wk_filter(anom.values, w) for w in waves}        # filter the whole (global) series
    lon = anom["lon"].values                                    # full globe 0-360
    time = anom["time"].values
    sel = (time <= np.datetime64(end)) & (time > np.datetime64(end) - np.timedelta64(days, "D"))
    tt = time[sel]; xx = lon
    base = gaussian_filter(anom.values[sel], sigma=(1.4, 1.0))
    fig, ax = plt.subplots(figsize=(7.8, 9.8))
    levels = [-72, -56, -40, -24, -8, 8, 24, 40, 56, 72]        # green = convection, brown = suppressed
    cmap = plt.get_cmap("BrBG_r")
    pm = ax.contourf(xx, tt, base, levels=levels, cmap=cmap,
                     norm=BoundaryNorm(levels, cmap.N), extend="both")
    for w in waves:
        sig = (3.0, 2.0) if w == "LF" else (0.8, 0.5)           # keep wave cores sharp; LF broad
        fw = gaussian_filter(filt[w][sel], sigma=sig)
        c = WAVES[w]["color"]; lw = WAVES[w]["lw"]; cl = WAVES[w].get("clev", 13)
        ax.contour(xx, tt, fw, levels=[-cl], colors=c, linewidths=lw, linestyles="solid")
        ax.contour(xx, tt, fw, levels=[cl], colors=c, linewidths=lw * 0.8, linestyles="dashed")
    if prelim_days:                                             # flag the taper-affected recent end
        ax.axhline(np.datetime64(end) - np.timedelta64(prelim_days, "D"), color="0.25", lw=0.8)
        ax.text(184, np.datetime64(end) - np.timedelta64(prelim_days, "D"), " preliminary →",
                color="0.25", fontsize=7, va="bottom", ha="center", zorder=6)
    ax.set_xlim(0, 360); ax.set_xlabel("longitude"); ax.set_ylabel("date"); ax.invert_yaxis()
    ax.set_xticks(range(0, 361, 60))
    ax.set_xticklabels(["0", "60E", "120E", "180", "120W", "60W", "0"])
    ax.axvline(180, color="0.6", lw=0.4, ls=":")
    ax.yaxis.set_major_formatter(DateFormatter("%d %b"))
    ax.set_title("OLR anomaly + convectively-coupled waves  ·  5°S–5°N"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=10)
    handles = [plt.Line2D([], [], color=WAVES[w]["color"], lw=WAVES[w]["lw"],
                          label=WAVES[w].get("label", w)) for w in waves]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92,
                    title="solid = convection")
    leg.get_title().set_fontsize(7)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.055, aspect=42)
    cb.set_label("OLR anomaly (W m⁻²)   ·   green = convection, brown = suppressed")
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="render current period from the GMGSI store")
    ap.add_argument("--hist", help="end date YYYY-MM-DD for a historical demo render")
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--out", default=str(HERE.parent.parent / "assets" / "sst" / "olr_waves.webp"))
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.realtime:
        clim = xr.open_dataset(CLIM)
        rt = xr.open_dataset(STORE_RT)
        anom = _anomaly_rt(rt, clim, "05")
        end = pd.Timestamp(anom["time"].values[-1]).to_pydatetime()
        render(anom, end, min(args.days, 120), Path(args.out), prelim_days=14,
               subtitle="GMGSI longwave-IR proxy · deseasonalised vs interp-OLR 1979-2022")
    else:
        clim = xr.open_dataset(HIST)
        end = datetime.strptime(args.hist, "%Y-%m-%d") if args.hist else \
            datetime.utcfromtimestamp(clim["time"].values[-1].astype("datetime64[s]").astype(int))
        render(_anomaly(clim, "05"), end, args.days, Path(args.out),
               subtitle="NOAA interpolated OLR 1979-2022")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
