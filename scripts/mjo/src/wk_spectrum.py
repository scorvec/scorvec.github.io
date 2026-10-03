#!/usr/bin/env python3
"""Wheeler-Kiladis wavenumber-frequency spectrum of OBSERVED tropical OLR, raw power (Wheeler & Kiladis 1999).

User 2026-10-03: "just compute the WK spectra, don't compare anything to normal ... from real OLR data". So: no
background, no ratio, no significance test -- the power spectrum of the last 96 days of satellite-measured OLR.

Data: NASA CERES FLASHFlux daily gridded TOA fluxes from NOAA-20 (FLASH_TISA_NOAA20 Version1A, `Fluxes/TOA/All_Sky/
toa_lw_all`, 1 deg, Dec 2023 on, about four days behind; NASA LaRC ASDC via Earthdata login). It is broadband OLR
measured by the CERES instrument, not a model field and not an infrared proxy. The other real-OLR record, NOAA's daily
HIRS CDR, has no working distribution in 2026 (its AWS bucket holds only documentation; NCEI's paths 404 / 503), and
NOAA's interpolated OLR ends in 2022.

Method (WK99, as NCL wkSpaceTime): 15S-15N; the 96-day window's mean and linear trend removed per grid point (no
seasonal cycle is subtracted: that would need a climatology); 10 % split cosine taper in time; split into the parts
symmetric and antisymmetric about the equator; 2-D FFT in longitude and time per latitude; power summed over the
latitudes; one 1-2-1 pass in frequency for display. Shown as log10 power with the equatorial shallow-water curves at
equivalent depths 12, 25 and 50 m.

    python wk_spectrum.py --out ../../../assets/sst/wk_raw.webp       # from scripts/mjo/src; olr-waves.yml, daily
Each day is reduced to its 15S-15N band (band_YYYYMMDD.npz, ~45 KB) in WK_CACHE (default ~/data_archive/wk/flashflux;
in Actions a cached directory) and the 6 MB download deleted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

NSEG = 96                                   # WK99 window
KMAX, FMAX = 15, 0.5                        # daily data: 0.5 cycles per day is the Nyquist frequency
SHORT = "FLASH_TISA_NOAA20"
VAR = "Fluxes/TOA/All_Sky/toa_lw_all"
CACHE = Path(os.environ.get("WK_CACHE", str(Path.home() / "data_archive" / "wk" / "flashflux")))


# ── data ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def _band(f: Path):
    """One FLASHFlux file -> (lat centres, lon centres 0-360 ascending, OLR) on 15S-15N."""
    import netCDF4
    with netCDF4.Dataset(f) as nc:
        v = np.ma.filled(nc[VAR][:].astype("float32"), np.nan)
        la, lo = np.asarray(nc["lat"][:]), np.asarray(nc["lon"][:])
    la = la + 0.5; lo = (lo + 0.5) % 360                              # coordinates are cell EDGES (89..-90, -180..179)
    sel = np.abs(la) < 15
    v, la = v[sel], la[sel]
    o = np.argsort(la); v, la = v[o], la[o]
    p = np.argsort(lo)
    return la, lo[p], v[:, p]


def fetch(days: int = NSEG + 12) -> dict:
    """{date: band file} for the newest `days` days. FLASHFlux republishes a day under a newer production number
    (..._406410.YYYYMMDD.nc): the highest wins, and a band file remembers which it came from."""
    import earthaccess
    earthaccess.login(strategy="environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc")
    end = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    g = earthaccess.search_data(short_name=SHORT, temporal=(f"{end - pd.Timedelta(days=days + 10):%Y-%m-%d}", f"{end:%Y-%m-%d}"))
    best = {}
    for x in g:
        url = next((u for u in x.data_links() if u.endswith(".nc")), None)
        if not url:
            continue
        prod, day = url.rsplit("/", 1)[1].split("_")[-1].split(".")[:2]
        if day not in best or prod > best[day][0]:
            best[day] = (prod, x)
    want = sorted(best)[-days:]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for d in want:
        bf = CACHE / f"band_{d}.npz"
        if bf.exists() and str(np.load(bf)["prod"]) == best[d][0]:
            out[pd.Timestamp(d)] = bf
            continue
        nc = earthaccess.download([best[d][1]], local_path=str(CACHE))[0]
        la, lo, v = _band(Path(nc))
        np.savez_compressed(bf, lat=la, lon=lo, olr=v, prod=best[d][0])
        Path(nc).unlink()
        out[pd.Timestamp(d)] = bf
    for f in CACHE.glob("band_*.npz"):                                   # keep the cache to the window
        if f.stem[5:] < want[0]:
            f.unlink()
    return out


def load(files: dict):
    """(time, lat, lon) OLR on 15S-15N from the band files."""
    days, arr, lat, lon = [], [], None, None
    for d, f in sorted(files.items()):
        z = np.load(f)
        lat, lon = z["lat"], z["lon"]
        days.append(d); arr.append(z["olr"])
    return pd.DatetimeIndex(days), lat, lon, np.stack(arr)


# ── spectrum ──────────────────────────────────────────────────────────────────────────────────────────────────────
def taper(n, frac=0.1):
    w = np.ones(n); m = int(round(frac * n))
    x = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
    w[:m] = x; w[-m:] = x[::-1]
    return w


def spectrum(seg):
    """seg (time, lat, lon), lat symmetric about 0 -> freq >= 0, k (eastward > 0), P_sym, P_asym summed over lat."""
    nt, ny, nx = seg.shape
    t = np.arange(nt); tc = t - t.mean()
    a = seg - seg.mean(0)
    a = a - np.einsum("t,yx->tyx", tc, (tc[:, None, None] * a).sum(0) / (tc ** 2).sum())
    a = a * taper(nt)[:, None, None]
    flip = a[:, ::-1, :]
    out = []
    for f in (0.5 * (a + flip), 0.5 * (a - flip)):
        F = np.fft.fft2(f, axes=(0, 2)) / (nt * nx)
        P = np.fft.fftshift((np.abs(F) ** 2).sum(1), axes=(0, 1))
        out.append(P)
    freq = np.fft.fftshift(np.fft.fftfreq(nt, d=1.0)); kk = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0 / nx))
    pos = freq >= 0
    # numpy's exp(-i(kx + wt)) convention: positive frequency with POSITIVE k is a WESTWARD wave; flip k
    return freq[pos], -kk[::-1], out[0][pos][:, ::-1], out[1][pos][:, ::-1]


def smooth121(a, axis, n=1):
    a = a.copy()
    for _ in range(n):
        p = np.pad(a, [(1, 1) if i == axis else (0, 0) for i in range(a.ndim)], mode="edge")
        sl = lambda s: tuple(s if i == axis else slice(None) for i in range(a.ndim))      # noqa: E731
        a = 0.25 * p[sl(slice(0, -2))] + 0.5 * p[sl(slice(1, -1))] + 0.25 * p[sl(slice(2, None))]
    return a


# ── figure ────────────────────────────────────────────────────────────────────────────────────────────────────────
def dispersion(ax, part):
    """Equatorial shallow-water dispersion curves (Matsuno 1966) at h = 12, 25 (dashed), 50 m, cycles per day."""
    g, beta, a_e = 9.81, 2.28e-11, 6.371e6
    kk = np.linspace(-KMAX, KMAX, 400)
    kw = kk / a_e
    cpd = lambda w: w * 86400 / (2 * np.pi)                                          # noqa: E731
    for h in (12, 25, 50):
        c = np.sqrt(g * h)
        if part == "sym":
            lines = [(kk[kk > 0], cpd(c * kw[kk > 0])),
                     (kk[kk < 0], cpd((-beta * kw / (kw ** 2 + 3 * beta / c))[kk < 0])),
                     (kk, cpd(np.sqrt(3 * beta * c + (c * kw) ** 2)))]
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                w_p = kw * c / 2 * (1 + np.sqrt(1 + 4 * beta / (kw ** 2 * c)))
                w_m = kw * c / 2 * (1 - np.sqrt(1 + 4 * beta / (kw ** 2 * c)))
            lines = [(kk[kk < 0], cpd(np.abs(w_m[kk < 0]))), (kk[kk > 0], cpd(w_p[kk > 0])),
                     (kk, cpd(np.sqrt(5 * beta * c + (c * kw) ** 2)))]
        for x, y in lines:
            ax.plot(x, y, color="#222", lw=0.7, ls="--" if h == 25 else "-", alpha=0.8)
    ax.axvline(0, color="#777", lw=0.6, ls=":")
    for per in (3, 6, 30):
        ax.axhline(1 / per, color="#999", lw=0.5, ls=":")
        ax.text(KMAX - 0.3, 1 / per, f"{per} d", fontsize=7, color="#6f6b64", ha="right", va="bottom")
    labs = ({"Kelvin": (8, 0.33), "n=1 ER": (-9, 0.06), "MJO": (2, 0.025), "n=1 WIG": (-11, 0.46), "n=1 EIG": (12, 0.47)}
            if part == "sym" else {"MRG": (-8, 0.28), "n=0 EIG": (8, 0.42)})
    for t, (x, y) in labs.items():
        ax.text(x, y, t, fontsize=8, color="#111", weight="bold", ha="center", clip_on=True)


def render(freq, kk, ps, pa, t0, t1, out: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    m = np.abs(kk) <= KMAX; f = (freq > 0) & (freq <= FMAX)
    K, Fq = np.meshgrid(kk[m], freq[f])
    L = {p: np.log10(smooth121(P, 0, 1)[np.ix_(f, m)]) for p, P in (("sym", ps), ("asym", pa))}
    allv = np.concatenate([L["sym"].ravel(), L["asym"].ravel()])
    lo, hi = np.floor(np.nanpercentile(allv, 2) * 4) / 4, np.ceil(np.nanpercentile(allv, 99.8) * 4) / 4
    levels = np.arange(lo, hi + 0.25, 0.25)
    cmap = plt.get_cmap("YlGnBu"); norm = BoundaryNorm(levels, cmap.N, extend="both")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, part in zip(axes, ("sym", "asym")):
        cf = ax.contourf(K, Fq, L[part], levels=levels, cmap=cmap, norm=norm, extend="both")
        dispersion(ax, part)
        ax.set_xlim(-KMAX, KMAX); ax.set_ylim(0, FMAX)
        ax.set_xlabel("zonal wavenumber (eastward > 0)", fontsize=9)
        ax.set_title(("Symmetric about the equator: Kelvin, equatorial Rossby, MJO" if part == "sym"
                      else "Antisymmetric: mixed Rossby–gravity, n = 0 eastward gravity"), fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("frequency (cycles per day)", fontsize=9)
    cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.05, pad=0.13, aspect=50)
    cb.set_label("log10 power, (W m⁻²)² per wavenumber–frequency bin, summed over 15°S–15°N", fontsize=8.5)
    fig.suptitle(f"Wheeler–Kiladis spectrum of observed OLR, 15°S–15°N, {t0:%d %b} – {t1:%d %b %Y} (96 days)",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.03, "Raw power: no background removed, no comparison with normal; the window's mean and trend are removed "
             "at each point, so a standing anomaly (an El Niño's, the seasonal cycle) does not appear.\nCurves: equatorial "
             "shallow-water waves at equivalent depths 12, 25 (dashed) and 50 m. Data: NASA CERES FLASHFlux NOAA-20 daily "
             "TOA longwave flux (NASA LaRC ASDC).", fontsize=8, color="#6f6b64")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../../assets/sst/wk_raw.webp")
    a = ap.parse_args()
    t, lat, lon, olr = load(fetch())
    t = t[-NSEG:]; olr = olr[-NSEG:]
    gaps = (t[-1] - t[0]).days + 1 - len(t)
    assert len(t) == NSEG and gaps == 0, f"{len(t)} days, {gaps} missing in the window"
    assert np.allclose(lat, -lat[::-1]), "latitudes must be symmetric about the equator"
    bad = ~np.isfinite(olr)
    if bad.any():                                                               # a few missing cells: fill from the day's zonal mean
        print(f"  {bad.mean():.4%} of cells missing, filled with the latitude's daily zonal mean", flush=True)
        zm = np.nanmean(olr, axis=2, keepdims=True); olr = np.where(bad, zm, olr)
    freq, kk, ps, pa = spectrum(olr)
    out = Path(a.out)
    render(freq, kk, ps, pa, t[0], t[-1], out)
    js = out.parent / "data" / "wk_spectrum.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps({"source": f"{SHORT} {VAR}", "window": [f"{t[0]:%Y-%m-%d}", f"{t[-1]:%Y-%m-%d}"],
                              "days": NSEG, "mean_olr": round(float(np.nanmean(olr)), 1)}))
    print(f"window {t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d}, mean OLR {np.nanmean(olr):.1f} W/m2", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
