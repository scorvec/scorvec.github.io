#!/usr/bin/env python3
"""Kelvin & equatorial Rossby wave trackers — filtered Hovmöllers with tracks.

Companion to olr_waves.py (which overlays every band as contours on one
Hovmöller): here each wave gets its OWN panel — the Wheeler-Kiladis
bandpassed OLR anomaly as shading — and active enhanced-convection
packets are TRACKED: the recent global phase speed of each band is
estimated by lag cross-correlation of the filtered field, today's packet
centres are located, and their characteristics are extrapolated 12 days
ahead at that speed in a marked extension strip. Kelvin uses the 5°S-5°N
band; ER (n=1) the broader 15°S-15°N band where its off-equator OLR
signal lives.

Reuses olr_waves' filter, stores and climatology verbatim.

    python scripts/sst/wave_trackers.py                       # real-time
    python scripts/sst/wave_trackers.py --out /tmp/test.webp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from olr_waves import CLIM, STORE_RT, _anomaly_rt, wk_filter  # noqa: E402


def wk_filter_padded(anom: np.ndarray, wave: str, pad: int = 48) -> np.ndarray:
    """wk_filter with a soft-landing end pad so its internal split-cosine
    taper (5% of the record ≈ 4 weeks) attenuates the PADDING, not the most
    recent — operationally crucial — days: the field ramps from the last
    observed row to zero over 10 days, then stays zero. The padded rows are
    discarded after filtering."""
    n = anom.shape[0]
    ramp_n = 10
    ramp = anom[-1][None, :] * (1 - (np.arange(1, ramp_n + 1) / (ramp_n + 1)))[:, None]
    padded = np.vstack([anom, ramp, np.zeros((pad - ramp_n, anom.shape[1]))])
    return wk_filter(padded, wave)[:n]

OUT = HERE.parent.parent / "assets" / "sst" / "wave_tracker.webp"
M_PER_DEG = 111320.0                 # metres per degree longitude at the equator
SHOW_DAYS = 90                       # history shown
EXT_DAYS = 12                        # extrapolation strip
SPEED_WIN = 15                       # days used for the lag-correlation speed fit

PANELS = {
    "Kelvin": dict(band="05", cexp=(5.0, 30.0),   # plausible speed window m/s (east)
                   title="Convectively coupled Kelvin waves (5°S–5°N)"),
    "ER":     dict(band="15", cexp=(-12.0, -1.0),  # westward
                   title="Equatorial Rossby waves n=1 (15°S–15°N)"),
}


def phase_speed(f: np.ndarray, dlon: float, cexp: tuple[float, float]) -> float:
    """Mean zonal phase speed (m/s) of the filtered field over the last
    SPEED_WIN days, from circular lag-1-day cross-correlation: for each
    consecutive-day pair the best cyclic shift is found by FFT correlation
    and pairs are weighted by their combined variance. Clamped to the
    band's plausible window so one noisy day can't flip the direction."""
    nlon = f.shape[1]
    g = f[-(SPEED_WIN + 1):]
    shifts, weights = [], []
    for i in range(len(g) - 1):
        a, b = g[i], g[i + 1]
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        corr = np.fft.ifft(np.fft.fft(b) * np.conj(np.fft.fft(a))).real
        s = int(np.argmax(corr))
        if s > nlon // 2:
            s -= nlon                            # signed cyclic shift (cells/day)
        # limit to physically sane daily displacement (< a third of the circle)
        if abs(s) > nlon // 3:
            continue
        shifts.append(s)
        weights.append(a.std() * b.std())
    if not shifts:
        return float(np.mean(cexp))
    cells = np.average(shifts, weights=weights)
    c = cells * dlon * M_PER_DEG / 86400.0       # m/s
    return float(np.clip(c, min(cexp), max(cexp)))


def packet_centres(f: np.ndarray, lons: np.ndarray, nsig: float = 1.0) -> list[float]:
    """Longitudes of active enhanced-convection packets: minima of the
    3-day-mean filtered field below -nsig of the panel's own recent std."""
    recent = f[-3:].mean(axis=0)
    sm = gaussian_filter1d(recent, 2, mode="wrap")
    thr = -nsig * f[-60:].std()
    cent = []
    n = len(sm)
    for i in range(n):
        if sm[i] < thr and sm[i] <= sm[(i - 1) % n] and sm[i] <= sm[(i + 1) % n]:
            cent.append(float(lons[i]))
    return cent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    clim = xr.open_dataset(CLIM)
    rt = xr.open_dataset(STORE_RT)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 8.8), sharey=True)
    now = None
    for ax, (wave, cfg) in zip(axes, PANELS.items()):
        anom = _anomaly_rt(rt, clim, cfg["band"])
        lons = anom["lon"].values
        dlon = float(lons[1] - lons[0])
        times = pd.to_datetime(anom["time"].values)
        now = times[-1]
        filt = wk_filter_padded(anom.values, wave)

        show = filt[-SHOW_DAYS:]
        tshow = times[-SHOW_DAYS:]
        sd = max(filt[-180:].std(), 1e-6)
        ext_t = pd.date_range(now + pd.Timedelta(days=1), periods=EXT_DAYS)
        yfull = tshow.append(ext_t)

        pm = ax.pcolormesh(lons, tshow, show, cmap="BrBG_r",
                           vmin=-3 * sd, vmax=3 * sd, shading="nearest")
        # extension strip: light grey ground, dashed boundary at 'today'
        ax.axhspan(now + pd.Timedelta(hours=12), yfull[-1], color="0.94", zorder=0)
        ax.axhline(now + pd.Timedelta(hours=12), color="0.25", lw=1.0, ls="--")
        ax.annotate("constant-phase-speed extrapolation", xy=(0.985, 0.028),
                    xycoords="axes fraction", ha="right", fontsize=8, color="0.35")

        c = phase_speed(filt, dlon, cfg["cexp"])
        deg_day = c * 86400.0 / M_PER_DEG
        for lon0 in packet_centres(filt, lons):
            # characteristic through the packet: 8 days back over the data
            # (visual fit check) + EXT_DAYS forward into the strip
            tt = pd.date_range(now - pd.Timedelta(days=8), yfull[-1])
            dl = np.array([(t - now).days + (t - now).seconds / 86400 for t in tt])
            ll = (lon0 + deg_day * dl) % 360
            # break the line at the dateline wrap so it doesn't streak across
            seg = np.where(np.abs(np.diff(ll)) > 180)[0]
            ll2 = ll.copy()
            for s in seg:
                ll2[s + 1:] = ll2[s + 1:]        # (kept simple: draw as points)
            ax.plot(ll, tt, ls="none", marker=".", ms=2.6, color="#111111")
            ax.annotate(f"{c:+.0f} m/s", xy=(lon0, now + pd.Timedelta(days=2)),
                        fontsize=8, fontweight="bold", color="#111111",
                        ha="center", clip_on=True)

        ax.set_title(cfg["title"], fontsize=11, fontweight="bold", loc="left")
        ax.set_xticks([0, 60, 120, 180, 240, 300])
        ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W"])
        ax.set_ylim(yfull[-1], tshow[0])          # newest at the BOTTOM (site convention)
        ax.yaxis.set_major_formatter(DateFormatter("%b %d"))
        ax.grid(lw=0.2, color="0.6", alpha=0.4)
        cb = fig.colorbar(pm, ax=ax, orientation="horizontal", fraction=0.04,
                          pad=0.045, aspect=38)
        cb.set_label(f"{wave}-filtered OLR′ (W/m² · green = enhanced convection)",
                     fontsize=8.5)

    fig.suptitle(f"Equatorial wave trackers — through {now:%b %d %Y}\n"
                 "Wheeler–Kiladis bandpass of the GMGSI OLR proxy · dotted lines: "
                 "packet characteristics at the fitted phase speed",
                 fontsize=11.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.06, right=0.99, wspace=0.06)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
