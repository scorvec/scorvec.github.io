#!/usr/bin/env python3
"""Wheeler-Kiladis wavenumber-frequency spectrum of OBSERVED tropical OLR (Wheeler & Kiladis 1999), with an OLR archive.

Data: NASA CERES FLASHFlux daily gridded TOA fluxes from NOAA-20 (FLASH_TISA_NOAA20 Version1A, `Fluxes/TOA/All_Sky/
toa_lw_all`, 1 deg, Dec 2023 on, about four days behind; NASA LaRC ASDC via Earthdata login): broadband OLR measured by
the CERES instrument, not a model field and not an infrared proxy.

ARCHIVE (user 2026-10-03: "start building an OLR archive going forward"): every day's 15S-15N band (30 x 360, 1 deg) is
kept in monthly files olr_band_YYYYMM.npz (dates, olr, production id) in WK_ARCHIVE. In Actions that directory lives on
the frames branch (assets/sst/anim/wk_olr, seeded and published by olr-waves.yml); on the laptop it is
~/data_archive/wk/archive. A day is replaced only when FLASHFlux republishes it under a newer production number.
`--backfill` fills the archive from the start of the record (Dec 2023).

BACKGROUND (user 2026-10-03, option 1): the spectrum is divided by a background built from the SAME observed record:
every 96-day segment of the archive, overlapping by 60 days, mean and trend removed, 10 % taper, symmetric and
antisymmetric power averaged over segments; background = the mean of the two, smoothed by 1-2-1 passes in wavenumber
(5/10/20/40 as frequency rises) and 10 in frequency (WK99, NCL wkSpaceTime). Written to reference/wk_bg_ceres.nc by
`--build-bg` and rebuilt when the archive has grown by a season.

Figure: top row = the record's own spectrum / background (the textbook WK diagram, every shaded bin significant over
its segments); bottom row = the last 96 days / background, shaded only where significant: one 1-2-1 pass in frequency and
one in wavenumber (~10.7 equivalent degrees of freedom), chi-square p per bin, Benjamini-Hochberg FDR 10 % over the
plotted domain (the site's significance rule). The window's mean and trend are removed at each point.

    python wk_spectrum.py --backfill 2023-12-01        # laptop, once
    python wk_spectrum.py --build-bg                     # laptop, after a backfill or each season
    python wk_spectrum.py --out ../../../assets/sst/wk_spectrum.webp   # daily (olr-waves.yml): append + render
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
CACHE = Path(os.environ.get("WK_CACHE", str(Path.home() / "data_archive" / "wk" / "flashflux")))      # transient downloads
ARCHIVE = Path(os.environ.get("WK_ARCHIVE", str(Path.home() / "data_archive" / "wk" / "archive")))
REF = Path(__file__).resolve().parent.parent / "data" / "reference"
BG = REF / "wk_bg_ceres.nc"
STEP = 36                                   # segments overlap by 60 days
DOF_NOW = 2 * 2 * (16 / 6)                  # one 1-2-1 pass in f and in k on a single window
KSTORE = 30                                 # wavenumbers kept in the background file


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


def _month_file(d) -> Path:
    return ARCHIVE / f"olr_band_{pd.Timestamp(d):%Y%m}.npz"


def archive_read() -> tuple[dict, dict]:
    """{date: olr band}, {date: production id} from every monthly file."""
    olr, prod = {}, {}
    for f in sorted(ARCHIVE.glob("olr_band_*.npz")):
        z = np.load(f)
        for d, v, p in zip(z["dates"], z["olr"], z["prod"]):
            olr[pd.Timestamp(str(d))] = v; prod[pd.Timestamp(str(d))] = str(p)
    return olr, prod


def archive_write(olr: dict, prod: dict, lat, lon, months=None) -> None:
    """Write the monthly files; only `months` (YYYYMM) when given, so untouched months keep their bytes
    (a rewrite changes the zip timestamps, and the frames branch would carry every month again)."""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    by_m = {}
    for d in sorted(olr):
        by_m.setdefault(f"{d:%Y%m}", []).append(d)
    for m, ds in by_m.items():
        if months is not None and m not in months:
            continue
        tmp = ARCHIVE / f".olr_band_{m}.tmp.npz"
        np.savez_compressed(tmp, dates=np.array([f"{d:%Y-%m-%d}" for d in ds]), olr=np.stack([olr[d] for d in ds]).astype("float32"),
                            prod=np.array([prod[d] for d in ds]), lat=lat, lon=lon)
        tmp.replace(ARCHIVE / f"olr_band_{m}.npz")


def fetch(since=None, days: int = NSEG + 12) -> tuple[dict, np.ndarray, np.ndarray]:
    """Bring the archive up to date (the newest `days` days, or everything from `since`) and return it.
    FLASHFlux republishes a day under a newer production number (..._406410.YYYYMMDD.nc): the highest wins."""
    import earthaccess
    earthaccess.login(strategy="environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc")
    end = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    start = pd.Timestamp(since) if since else end - pd.Timedelta(days=days + 10)
    best = {}
    for a, b in zip(pd.date_range(start, end, freq="90D"), list(pd.date_range(start, end, freq="90D"))[1:] + [end]):
        for x in earthaccess.search_data(short_name=SHORT, temporal=(f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}")):
            url = next((u for u in x.data_links() if u.endswith(".nc")), None)
            if not url:
                continue
            pr, day = url.rsplit("/", 1)[1].split("_")[-1].split(".")[:2]
            if day not in best or pr > best[day][0]:
                best[day] = (pr, x)
    olr, prod = archive_read()
    lat = lon = None
    CACHE.mkdir(parents=True, exist_ok=True)
    new = 0; touched = set()
    for d in sorted(best):
        t = pd.Timestamp(d)
        if t in prod and prod[t] >= best[d][0]:
            continue
        nc = earthaccess.download([best[d][1]], local_path=str(CACHE))[0]
        lat, lon, v = _band(Path(nc))
        Path(nc).unlink()
        olr[t] = v; prod[t] = best[d][0]; new += 1; touched.add(f"{t:%Y%m}")
        if new % 30 == 0:
            archive_write(olr, prod, lat, lon, touched); touched = set(); print(f"  archived {new} days (to {t:%Y-%m-%d})", flush=True)
    if lat is None:
        f = sorted(ARCHIVE.glob("olr_band_*.npz"))
        z = np.load(f[-1]); lat, lon = z["lat"], z["lon"]
    if touched:
        archive_write(olr, prod, lat, lon, touched)
    print(f"  OLR archive: {len(olr)} days, {min(olr):%Y-%m-%d} -> {max(olr):%Y-%m-%d} ({new} added or updated)", flush=True)
    return olr, lat, lon


def load(olr: dict):
    days = sorted(olr)
    return pd.DatetimeIndex(days), np.stack([olr[d] for d in days])


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


def band_masks(freq, kk):
    """Wave bands of the WK99 / CPC filters on the (freq, k) grid: Kelvin, n=1 ER and MJO (symmetric), MRG and n=0 EIG
    (antisymmetric). Kelvin/ER/MRG/EIG lie between their dispersion curves for equivalent depths 8 and 90 m."""
    g, beta, a_e = 9.81, 2.28e-11, 6.371e6
    F, K = np.meshgrid(freq, kk, indexing="ij")
    kw = K / a_e
    cpd = lambda w: w * 86400 / (2 * np.pi)                                          # noqa: E731
    def kel(h): return cpd(np.sqrt(g * h) * kw)
    def er(h):
        c = np.sqrt(g * h); return cpd(-beta * kw / (kw ** 2 + 3 * beta / c))
    with np.errstate(invalid="ignore", divide="ignore"):
        def mrg(h):
            c = np.sqrt(g * h); return cpd(np.abs(kw * c / 2 * (1 - np.sqrt(1 + 4 * beta / (kw ** 2 * c)))))
        def eig0(h):
            c = np.sqrt(g * h); return cpd(kw * c / 2 * (1 + np.sqrt(1 + 4 * beta / (kw ** 2 * c))))
        between = lambda fn: (F >= fn(8)) & (F <= fn(90))                            # noqa: E731
        return {
            "sym": {"Kelvin": (K >= 1) & (K <= 14) & (F >= 1 / 20) & (F <= 1 / 2.5) & between(kel),
                    "n=1 ER": (K >= -10) & (K <= -1) & (F >= 1 / 48) & (F <= 1 / 9.7) & (F <= er(90)) & (F >= er(8)),
                    "MJO": (K >= 1) & (K <= 5) & (F >= 1 / 96) & (F <= 1 / 30)},
            "asym": {"MRG": (K >= -10) & (K <= -1) & (F >= 1 / 6) & (F <= 1 / 3) & between(mrg),
                     "n=0 EIG": (K >= 1) & (K <= 14) & (F >= 1 / 5) & (F <= 1 / 2.5) & between(eig0)}}


def band_test(P, bg, masks, dof_bin, q=0.10):
    """Band-summed power over band-summed background, chi-square with dof_bin per bin, BH over all bands."""
    from scipy.stats import chi2
    res = {}
    for name, m in masks.items():
        n = int(m.sum())
        r = float(P[m].sum() / bg[m].sum())
        res[name] = {"ratio": r, "bins": n, "p": float(chi2.sf(r * dof_bin * n, dof_bin * n))}
    return res


def fill_gaps(t: pd.DatetimeIndex, arr: np.ndarray):
    """Missing cells -> the latitude's daily zonal mean; missing DAYS (FLASHFlux has a few) -> linear in time.
    Returns a continuous daily series and how many days were interpolated."""
    bad = ~np.isfinite(arr)
    if bad.any():
        zm = np.nanmean(arr, axis=2, keepdims=True); arr = np.where(bad, zm, arr)
    full = pd.date_range(t[0], t[-1], freq="D")
    if len(full) == len(t):
        return full, arr, 0
    idx = np.searchsorted(t, full)
    have = np.isin(full, t)
    out = np.empty((len(full),) + arr.shape[1:], arr.dtype)
    out[have] = arr
    x = np.arange(len(full))
    flat = out.reshape(len(full), -1)
    for i in np.where(~have)[0]:
        lo = np.max(np.where(have[:i])[0]); hi = i + np.min(np.where(have[i:])[0])
        w = (i - lo) / (hi - lo); flat[i] = (1 - w) * flat[lo] + w * flat[hi]
    return full, out, int((~have).sum())


def background(ps, pa, freq):
    bg = 0.5 * (ps + pa)
    for lo, hi, n in ((0.0, 0.1, 5), (0.1, 0.2, 10), (0.2, 0.3, 20), (0.3, 9.9, 40)):
        m = (freq >= lo) & (freq < hi)
        if m.any():
            bg[m] = smooth121(bg, 1, n)[m]
    return smooth121(bg, 0, 10)


def build_bg() -> int:
    import xarray as xr
    olr, prod = archive_read()
    t, arr = load(olr)
    t, arr, ngap = fill_gaps(t, arr)
    if ngap > 0.03 * len(t):
        print(f"refusing: {ngap} of {len(t)} days would be interpolated - backfill the archive first", flush=True)
        return 1
    ps = pa = 0; n = 0
    for i in range(0, len(t) - NSEG + 1, STEP):
        f, kk, a_s, a_a = spectrum(arr[i:i + NSEG]); ps = ps + a_s; pa = pa + a_a; n += 1
    ps, pa = ps / n, pa / n
    bg = background(ps, pa, f)
    keep = np.abs(kk) <= KSTORE
    REF.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"p_sym": (("freq", "k"), ps[:, keep]), "p_asym": (("freq", "k"), pa[:, keep]),
                "background": (("freq", "k"), bg[:, keep])}, coords={"freq": f, "k": kk[keep]},
               attrs={"source": f"{SHORT} {VAR}, 15S-15N, 1 deg", "first": f"{t[0]:%Y-%m-%d}", "last": f"{t[-1]:%Y-%m-%d}",
                      "segments": n, "segment_days": NSEG, "step_days": STEP, "interpolated_days": ngap}
               ).to_netcdf(BG)
    print(f"wrote {BG}: {n} segments, {t[0]:%Y-%m-%d} -> {t[-1]:%Y-%m-%d}, {ngap} missing days interpolated", flush=True)
    return 0


def bh(p, q=0.10):
    pf = p.ravel(); o = np.argsort(pf); n = len(pf)
    ok = pf[o] <= q * np.arange(1, n + 1) / n
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    out = np.zeros(n, bool); out[o[:k]] = True
    return out.reshape(p.shape)


def render(freq, kk, now, clim, t0, t1, out: Path, bands=None, masks=None):
    """now / clim: {"sym": (ratio, sig), "asym": (ratio, sig)} on (freq, kk)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = np.abs(kk) <= KMAX; f = (freq > 0) & (freq <= FMAX)
    K, Fq = np.meshgrid(kk[m], freq[f])
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
    rows = (("clim", clim, [1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.4],
             f"Record mean, {clim['_t0']:%b %Y} – {clim['_t1']:%b %Y} ({clim['_n']} windows)"),
            ("now", now, [1.5, 2, 2.5, 3, 4, 5, 6],
             f"Last 96 days, {t0:%d %b} – {t1:%d %b %Y}"))
    for i, (tag, res, lev, title) in enumerate(rows):
        for j, part in enumerate(("sym", "asym")):
            ax = axes[i, j]
            r, sig = res[part]
            Z = r[np.ix_(f, m)].copy()
            Z = np.where(sig[np.ix_(f, m)], Z, np.nan)
            cf = ax.contourf(K, Fq, Z, levels=lev, cmap="YlOrRd", extend="max")
            if sig[np.ix_(f, m)].any():
                ax.contour(K, Fq, sig[np.ix_(f, m)].astype(float), levels=[0.5], colors="#7a1f00", linewidths=0.8)
            else:
                ax.text(0.5, 0.62, "no single bin significant in this window",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#6f6b64", style="italic")
            dispersion(ax, part)
            if bands:
                txt = []
                for nm, b in bands[tag][part].items():
                    if b["sig"]:
                        ax.contour(K, Fq, masks[part][nm][np.ix_(f, m)].astype(float), levels=[0.5], colors="#1f5fa8",
                                   linewidths=1.3, linestyles="--")
                        txt.append(f"{nm}: {b['ratio']:.2f}× background (p {b['p']:.1g})")
                    else:
                        txt.append(f"{nm}: not significant")
                ax.text(0.98, 0.98, "Wave bands (blue dashed = significant)\n" + "\n".join(txt), transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5, family="monospace", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.6", lw=0.5))
            ax.set_xlim(-KMAX, KMAX); ax.set_ylim(0, FMAX)
            ax.set_title(f"{title}\n{'Symmetric: Kelvin, equatorial Rossby, MJO' if part == 'sym' else 'Antisymmetric: mixed Rossby–gravity, n = 0 eastward gravity'}",
                         fontsize=9.5, loc="left")
            ax.tick_params(labelsize=8)
            if i == 1:
                ax.set_xlabel("zonal wavenumber (eastward > 0)", fontsize=9)
            if j == 0:
                ax.set_ylabel("frequency (cycles per day)", fontsize=9)
        cb = fig.colorbar(cf, ax=axes[i, :].tolist(), orientation="vertical", fraction=0.025, pad=0.02)
        cb.set_label("power / background", fontsize=8.5); cb.ax.tick_params(labelsize=8)
    fig.suptitle("Wheeler–Kiladis spectrum of observed OLR, 15°S–15°N (NASA CERES FLASHFlux): power / background, "
                 "significant bins only", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.01, "Background: the mean of the symmetric and antisymmetric power over every 96-day window of the record, "
             "heavily smoothed (WK99). Shaded only where chi-square significant at Benjamini–Hochberg FDR 10 %\n(top row credited with the record's independent length only). "
             "Curves: equatorial shallow-water waves at equivalent depths 12, 25 (dashed) and 50 m. Data: NASA LaRC ASDC CERES FLASHFlux "
             "NOAA-20 daily TOA longwave flux.", fontsize=8, color="#6f6b64")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out, flush=True)


def main() -> int:
    import xarray as xr
    from scipy.stats import chi2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../../assets/sst/wk_spectrum.webp")
    ap.add_argument("--backfill", help="fill the archive from this date (YYYY-MM-DD)")
    ap.add_argument("--build-bg", action="store_true")
    ap.add_argument("--no-fetch", action="store_true", help="render from the archive as it is")
    a = ap.parse_args()
    if a.backfill:
        fetch(since=a.backfill)
        return 0
    if a.build_bg:
        return build_bg()
    olr = archive_read()[0] if a.no_fetch else fetch()[0]
    t, arr = load(olr)
    t, arr = t[-NSEG:], arr[-NSEG:]
    t, arr, ngap = fill_gaps(t, arr)
    if len(t) != NSEG:                                                   # a gap inside the last 96 days: shorten to it
        t, arr = t[-NSEG:], arr[-NSEG:]
    lat = np.load(sorted(ARCHIVE.glob("olr_band_*.npz"))[-1])["lat"]
    assert np.allclose(lat, -lat[::-1]), "latitudes must be symmetric about the equator"
    freq, kk, ps, pa = spectrum(arr)
    B = xr.open_dataset(BG)
    sel = np.isin(kk, B.k.values)
    freq_b = B.freq.values
    assert np.allclose(freq, freq_b), "background built with a different window length"
    bg = B["background"].values
    now, clim = {}, {"_t0": pd.Timestamp(B.attrs["first"]), "_t1": pd.Timestamp(B.attrs["last"]), "_n": int(B.attrs["segments"])}
    kk_s = kk[sel]
    clim_days = (clim["_t1"] - clim["_t0"]).days + 1
    dom = (np.abs(kk_s)[None, :] <= KMAX) & (freq[:, None] > 0) & (freq[:, None] <= FMAX)
    for part, P, Pc in (("sym", ps[:, sel], B["p_sym"].values), ("asym", pa[:, sel], B["p_asym"].values)):
        r = smooth121(smooth121(P, 0, 1), 1, 1) / bg
        p = chi2.sf(r * DOF_NOW, DOF_NOW)
        sig = np.zeros_like(r, bool); sig[dom] = bh(p[dom])
        now[part] = (r, sig)
        # the record mean: smoothed the same way, and credited only with the record's INDEPENDENT length
        # (overlapping windows add no data): DOF = DOF_NOW x days / 96
        dof_c = DOF_NOW * clim_days / NSEG
        rc = smooth121(smooth121(Pc, 0, 1), 1, 1) / bg
        pc = chi2.sf(rc * dof_c, dof_c)
        sigc = np.zeros_like(rc, bool); sigc[dom] = bh(pc[dom])
        clim[part] = (rc, sigc)
    # wave-band test: band-summed power / background (raw, unsmoothed power; 2 dof per bin, conservative), BH over bands
    masks = band_masks(freq, kk_s)
    bands = {}
    for tag, P, dof in (("now", {"sym": ps[:, sel], "asym": pa[:, sel]}, 2.0),
                        ("clim", {"sym": B["p_sym"].values, "asym": B["p_asym"].values}, 2.0 * clim_days / NSEG)):
        res = {part: band_test(P[part], bg, masks[part], dof) for part in ("sym", "asym")}
        flat = [(part, nm) for part in res for nm in res[part]]
        sig = bh(np.array([res[pt][nm]["p"] for pt, nm in flat]))
        for (pt, nm), sg in zip(flat, sig):
            res[pt][nm]["sig"] = bool(sg) and res[pt][nm]["ratio"] > 1
        bands[tag] = res
    out = Path(a.out)
    render(freq, kk_s, now, clim, t[0], t[-1], out, bands, masks)
    js = out.parent / "data" / "wk_spectrum.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    summ = {"source": f"{SHORT} {VAR}", "window": [f"{t[0]:%Y-%m-%d}", f"{t[-1]:%Y-%m-%d}"], "days": NSEG,
            "interpolated_days": ngap, "archive_days": len(olr), "archive_first": f"{min(olr):%Y-%m-%d}",
            "background": {"first": B.attrs["first"], "last": B.attrs["last"], "segments": int(B.attrs["segments"])},
            "significant_bins": {p: int(now[p][1].sum()) for p in ("sym", "asym")},
            "significant_bins_record": {p: int(clim[p][1].sum()) for p in ("sym", "asym")},
            "bands": {tag: {pt: {nm: {"ratio": round(b["ratio"], 3), "p": float(f"{b['p']:.3g}"), "sig": b["sig"]}
                                 for nm, b in v.items()} for pt, v in r.items()} for tag, r in bands.items()}, "mean_olr": round(float(np.nanmean(arr)), 1)}
    js.write_text(json.dumps(summ))
    print(summ, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
