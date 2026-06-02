#!/usr/bin/env python3
"""Meridional mass streamfunction Ψ(φ,p) from the AIFS-ENS 0-h analysis.

    Ψ(φ,p) = (2π a cosφ / g) ∫_0^p [v] dp'        [v] = zonal-mean meridional wind
    units 10^10 kg s^-1.  Ψ>0 ⇒ clockwise cell in the (φ,p) plane (northward aloft).

Hadley / Ferrel / polar cells fall straight out. Anomalies vs the ERA5 climatology
(build_mmsf_clim.py) show how the Hadley cell is responding to the current event;
daily 0-h analyses are stashed into an animator.

    python src/mmsf.py --date 20260602 --time 00 --out /tmp/mmsf.webp
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from download_aifs import _retrieve

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
A = 6.371e6; G = 9.80665; SCALE = 1e10        # Earth radius, gravity, plot units


def download_v(date: str, time: str, dd: Path) -> Path:
    """0-h analysis meridional wind on the 13 pressure levels (control = analysis at t=0)."""
    p = dd / f"v_{date}_{time}z_cf.grib2"
    if not p.exists():
        print("  downloading 0-h analysis v (13 levels) …", flush=True)
        _retrieve(dict(model="aifs-ens", date=date, time=int(time), stream="enfo",
                       type="cf", levtype="pl", levelist=LEVELS, param="v", step=0), str(p))
    return p


def streamfunction(vbar: np.ndarray, p_pa: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Ψ(p,lat) in 10^10 kg/s from zonal-mean v (lev,lat), p ascending (Pa)."""
    # cumulative trapezoidal integral of [v] from the top (p=0) down to each level
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


def plot_psi(psi, p_hpa, lat, out: Path, title: str, anom: bool = False):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    mlat = (lat >= -88) & (lat <= 88)
    lim = float(np.nanpercentile(np.abs(psi[:, mlat]), 99.5))
    levs = np.linspace(-lim, lim, 21)
    cf = ax.contourf(lat[mlat], p_hpa, psi[:, mlat], levels=levs, cmap="RdBu_r", extend="both")
    ax.contour(lat[mlat], p_hpa, psi[:, mlat], levels=[0], colors="0.3", linewidths=0.7)
    ax.set_ylim(1000, 50); ax.set_yscale("log")
    ax.set_yticks([1000, 850, 700, 500, 300, 200, 100, 50]); ax.set_yticklabels([1000, 850, 700, 500, 300, 200, 100, 50])
    ax.set_xlabel("latitude"); ax.set_ylabel("pressure (hPa)")
    ax.set_xticks([-60, -30, 0, 30, 60]); ax.axvline(0, color="0.6", lw=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(("Ψ anomaly" if anom else "Ψ") + "  (10¹⁰ kg s⁻¹)" +
                 ("   red = anomalous clockwise (↑N aloft)" if anom else ""), fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--data-dir", default="data/mmsf")
    ap.add_argument("--out", default="/tmp/mmsf.webp")
    args = ap.parse_args()
    dd = Path(args.data_dir); dd.mkdir(parents=True, exist_ok=True)
    vbar, p_pa, lat = zonal_mean_v(download_v(args.date, args.time, dd))
    psi = streamfunction(vbar, p_pa, lat)
    print(f"  Ψ range {np.nanmin(psi):.1f} … {np.nanmax(psi):.1f} ×10¹⁰ kg/s")
    plot_psi(psi, p_pa / 100.0, lat, Path(args.out),
             f"Meridional mass streamfunction — AIFS-ENS analysis {args.date} {args.time}Z")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
