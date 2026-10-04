#!/usr/bin/env python3
"""HRRR vs RRFS kinetic-energy power spectra from NATIVE-level winds.

Method: Corvec & Separovic DCT approach (github.com/scorvec/nwp_power_spectra)
— 2D DCT (ortho) of u and v on a common-size interior box, elliptic radial
binning to a 1D KE spectrum. Native hybrid levels only (pressure-level output
is smoothed by post-processing). RRFS levels are matched to the HRRR target
level by domain-mean pressure. Reference slopes k^-3 (synoptic) and k^-5/3
(mesoscale); the wavelength where a model's spectrum peels off the reference
is its effective resolution.

    python ke_spectra.py --date 20260718 --cycle 00 --fxx 6 --level 40
"""
from __future__ import annotations

import argparse
import io
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import scipy.fft
import scipy.stats as stats

HRRR_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
RRFS_BASE = "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a"
BOX = 512                    # interior box (grid points ≈ 1536 km at 3 km)
OUT = Path(__file__).resolve().parents[2] / "assets" / "spectra"


def _fetch_records(idx_url: str, grib_url: str, wanted: list[str]) -> dict:
    """Byte-range fetch of GRIB records matching the wanted idx substrings."""
    with urllib.request.urlopen(idx_url, timeout=60) as r:
        lines = r.read().decode().splitlines()
    starts = [int(l.split(":")[1]) for l in lines]
    out = {}
    for i, line in enumerate(lines):
        for w in wanted:
            if w in line:
                end = starts[i + 1] - 1 if i + 1 < len(starts) else ""
                req = urllib.request.Request(
                    grib_url, headers={"Range": f"bytes={starts[i]}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    out[w] = r.read()
    return out


def _decode(grib_bytes: bytes, with_latlon: bool = False):
    import eccodes as ec
    with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
        f.write(grib_bytes); f.flush()
        with open(f.name, "rb") as fh:
            gid = ec.codes_grib_new_from_file(fh)
            ny = ec.codes_get(gid, "Ny"); nx = ec.codes_get(gid, "Nx")
            vals = ec.codes_get_values(gid).reshape(ny, nx)
            if with_latlon:
                lats = ec.codes_get_array(gid, "latitudes").reshape(ny, nx)
                lons = ec.codes_get_array(gid, "longitudes").reshape(ny, nx)
                ec.codes_release(gid)
                return vals, lats, lons
            ec.codes_release(gid)
    return vals


# Common analysis region = the largest lat/lon rectangle INSCRIBED in the
# HRRR domain (trimmed of its boundary sponge). Both models are cropped to
# their own index boxes covering this region, so each spectrum uses its
# full native grid over identical geography.
_HRRR_MARGIN = 25            # points trimmed from HRRR edges (blend zone)
_bounds_cache: dict = {}
_crop_cache: dict = {}


def _hrrr_bounds(lats, lons):
    """Inscribed lat/lon rectangle of the trimmed HRRR Lambert grid."""
    m = _HRRR_MARGIN
    la, lo = lats[m:-m, m:-m], lons[m:-m, m:-m]
    return (float(la[0, :].max()), float(la[-1, :].min()),
            float(lo[:, 0].max()), float(lo[:, -1].min()))


def _crop_idx(model, grib_bytes):
    """Index box of this model's grid covering the common region."""
    if model in _crop_cache:
        return _crop_cache[model]
    _, lats, lons = _decode(grib_bytes, with_latlon=True)
    lons = np.where(lons > 180, lons - 360, lons)
    if "hrrr" not in _bounds_cache:
        if model == "hrrr":
            _bounds_cache["hrrr"] = _hrrr_bounds(lats, lons)
        else:                       # need HRRR first; caller order guarantees it
            raise RuntimeError("hrrr must be fetched before rrfs for bounds")
    b0, b1, l0, l1 = _bounds_cache["hrrr"]
    inside = (lats >= b0) & (lats <= b1) & (lons >= l0) & (lons <= l1)
    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    _crop_cache[model] = (int(rows[0]), int(rows[-1] + 1),
                          int(cols[0]), int(cols[-1] + 1))
    return _crop_cache[model]


def fetch_uv(model: str, date: str, cycle: str, fxx: int, level: int):
    """(u, v, mean_p_hPa) at one native hybrid level."""
    if model == "hrrr":
        base = f"{HRRR_BASE}/hrrr.{date}/conus/hrrr.t{cycle}z.wrfnatf{fxx:02d}.grib2"
    else:
        base = (f"{RRFS_BASE}/rrfs.{date}/{cycle}/"
                f"rrfs.t{cycle}z.natlev.3km.f{fxx:03d}.na.grib2")
    tags = [f":UGRD:{level} hybrid level:", f":VGRD:{level} hybrid level:",
            f":PRES:{level} hybrid level:"]
    recs = _fetch_records(base + ".idx", base, tags)
    if len(recs) < 2:
        raise RuntimeError(f"{model}: records not found for hybrid level {level}")
    u = _decode(recs[tags[0]]); v = _decode(recs[tags[1]])
    y0, y1, x0, x1 = _crop_idx(model, recs[tags[0]])
    p = float(np.mean(_decode(recs[tags[2]]))) / 100 if tags[2] in recs else np.nan
    return u[y0:y1, x0:x1], v[y0:y1, x0:x1], p, None


def find_level(model, date, cycle, fxx, target_hpa, lo=2, hi=60):
    """Hybrid level whose domain-mean pressure is nearest target (probes PRES
    records only — a few hundred KB each)."""
    if model == "hrrr":
        base = f"{HRRR_BASE}/hrrr.{date}/conus/hrrr.t{cycle}z.wrfnatf{fxx:02d}.grib2"
    else:
        base = (f"{RRFS_BASE}/rrfs.{date}/{cycle}/"
                f"rrfs.t{cycle}z.natlev.3km.f{fxx:03d}.na.grib2")
    best = (1e9, None)
    for lev in range(lo, hi + 1, 4):
        try:
            recs = _fetch_records(base + ".idx", base, [f":PRES:{lev} hybrid level:"])
            if not recs:
                continue
            pm = float(np.mean(_decode(next(iter(recs.values()))))) / 100
            if abs(pm - target_hpa) < best[0]:
                best = (abs(pm - target_hpa), lev, pm)
        except Exception:                                      # noqa: BLE001
            continue
    if best[1] is None:
        raise RuntimeError(f"{model}: no PRES levels found")
    print(f"  {model}: hybrid L{best[1]} ≈ {best[2]:.0f} hPa (target {target_hpa})")
    return best[1]


def interior(a: np.ndarray, center: tuple, n: int = BOX) -> np.ndarray:
    cy, cx = center
    y0 = max(0, min(a.shape[0] - n, cy - n // 2))
    x0 = max(0, min(a.shape[1] - n, cx - n // 2))
    return a[y0:y0 + n, x0:x0 + n]


def ke_spectrum(u: np.ndarray, v: np.ndarray, res_km: float = 3.0):
    """1D KE spectrum via 2D DCT + elliptic-ring binning (nwp_power_spectra)."""
    du = np.abs(scipy.fft.dctn(u, norm="ortho") / np.sqrt(u.size)) ** 2
    dv = np.abs(scipy.fft.dctn(v, norm="ortho") / np.sqrt(v.size)) ** 2
    p2 = 0.5 * (du + dv)
    s0, s1 = p2.shape
    nbwv = min(s0, s1); nbwx = max(s0, s1)
    fr = np.arange(s0)[:, None] / s0
    fc = np.arange(s1)[None, :] / s1
    knrm = (np.sqrt(fr ** 2 + fc ** 2) * nbwv).flatten()
    eps = 0.1 * np.sqrt(1 / nbwx)
    kbins = np.arange(-1.0 + eps, nbwv + eps, 1.0)
    abins, _, _ = stats.binned_statistic(knrm, p2.flatten(),
                                         statistic=np.nansum, bins=kbins)
    p1 = abins[1:]
    p1 = p1[0::2][:len(p1[1::2])] + p1[1::2]          # pair-sum (as in the repo)
    wavenumbers = np.arange(1, len(p1))
    wavelengths = res_km * nbwv / wavenumbers
    return wavelengths, p1[1:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--cycle", default="00")
    ap.add_argument("--fxx", type=int, nargs="+", default=[6])
    ap.add_argument("--target-hpa", type=float, default=700.0)
    ap.add_argument("--out", default=str(OUT / "ke_spectra.webp"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(args.fxx)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 5.6), squeeze=False)
    for j, fxx in enumerate(args.fxx):
        ax = axes[0, j]
        for model, color in (("hrrr", "#d62728"), ("rrfs", "#1f77b4")):
            try:
                lev = find_level(model, args.date, args.cycle, fxx, args.target_hpa)
                u, v, p, ctr = fetch_uv(model, args.date, args.cycle, fxx, lev)
            except Exception as e:                             # noqa: BLE001
                print(f"  {model} f{fxx:02d}: {str(e)[:80]}")
                continue
            wl, spec = ke_spectrum(u, v)
            ax.loglog(wl, spec, color=color, lw=1.4,
                      label=f"{model.upper()} (hyb L{lev}, ~{p:.0f} hPa)")
            print(f"  {model} f{fxx:02d}: L{lev} ≈ {p:.0f} hPa, {u.shape} grid")
        wl_ref = np.geomspace(12, 800, 50)
        a53 = spec[np.argmin(np.abs(wl - 300))] / (300.0 ** (5 / 3))
        ax.loglog(wl_ref, a53 * wl_ref ** (5 / 3), color="0.6", lw=0.9, ls="--",
                  label="k$^{-5/3}$")
        a3 = spec[np.argmin(np.abs(wl - 600))] / (600.0 ** 3)
        ax.loglog(wl_ref, a3 * wl_ref ** 3, color="0.35", lw=0.9, ls=":",
                  label="k$^{-3}$")
        ax.invert_xaxis()
        ax.set_xlabel("wavelength (km)")
        if j == 0:
            ax.set_ylabel("KE power spectral density (m² s⁻²)")
        ax.set_title(f"F{fxx:02d}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"HRRR vs RRFS kinetic-energy spectra — ~{args.target_hpa:.0f} hPa native winds · "
                 f"{args.date} {args.cycle}Z · full common domain (HRRR-inscribed)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
