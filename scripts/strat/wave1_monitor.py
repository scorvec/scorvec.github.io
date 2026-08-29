#!/usr/bin/env python3
"""Stationary wave-1 monitor, both hemispheres, through the AIFS-ENS forecast.

The forcing side of the E-P flux diagnostics: how much wave-1 is there, and is
it amplifying? Two products per cycle from the control's geopotential:

assets/sst/anim/wave1_nh/F##.webp (+ manifest), same for wave1_sh
    Longitude x height cross-section of the wavenumber-1 geopotential-height
    perturbation, meridionally averaged over the 55-65 deg band (the vortex
    forcing latitudes), 1000 -> 10 hPa, one 16-frame day-0..15 loop per
    hemisphere. The Holton figure, live: amplitude growth with height
    (~rho^-1/2 for propagating waves) and the WESTWARD PHASE TILT with height
    that is the visual signature of upward wave-activity flux (poleward heat
    flux). No tilt = trapped/barotropic; sheared-off top = the C-D lid at
    work. Colour range fixed across the loop per hemisphere.

assets/sst/wave1_amp.webp
    Amplitude vs lead time: |Z1| (solid) and |Z2| (dashed) at 500 / 100 / 10
    hPa for each hemisphere — "is the forcing increasing?" at a glance.

Rendering: Julia (CairoMakie) via the shared mjo_render.jl rasterizer, with a
matplotlib re-render for any frame Julia drops. z rides a small dedicated cf
pull (~180 MB/cycle) through the shared ecmwf store.

    python scripts/strat/wave1_monitor.py
"""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qbo_duct import latest_cycle, open_field, ecmwf, REPO, JULIA_SCRIPT

ANIM = REPO / "assets" / "sst" / "anim"
STAGING = HERE / "data" / ".wave1_staging"
G = 9.80665
BANDS = {"nh": (55.0, 65.0), "sh": (-65.0, -55.0)}
KEY_LEVS = [500, 100, 10]
LEV_COLORS = {500: "#8d6e63", 100: "#1e88e5", 10: "#7b1fa2"}


def wave_component(zband, k):
    """(wave_field(lon), amplitude) of zonal wavenumber k from Z(lon)."""
    F = np.fft.rfft(zband)
    amp = 2.0 * np.abs(F[k]) / zband.size
    keep = np.zeros_like(F)
    keep[k] = F[k]
    return np.fft.irfft(keep, n=zband.size), amp


def load_sections(tag):
    """{hemi: {"steps", "lon", "levs", "w1": (nstep, nlev, nlon),
    "amp": {k: (nstep, nlev)}}} from the control geopotential."""
    cyc = ecmwf.Cycle(tag[:8], tag[8:10])
    S = tuple(ecmwf.STEPS)
    zpath = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", "z", "pl",
                                         tuple(ecmwf.LEVELS_AAM), S))
    out = {h: {"w1": [], "amp": {1: [], 2: []}} for h in BANDS}
    levs_all = sorted(ecmwf.LEVELS_AAM)
    lon = lat = None
    for lev in levs_all:
        da = open_field(zpath, "z", lev)
        lat = da.latitude.values
        lon = da.longitude.values
        for h, (l0, l1) in BANDS.items():
            m = (lat >= l0) & (lat <= l1)
            w = np.cos(np.deg2rad(lat[m]))
            band = (da.values[:, m, :] * w[None, :, None]).sum(axis=1) / w.sum() / G
            w1_steps, a1_steps, a2_steps = [], [], []
            for i in range(band.shape[0]):
                f1, a1 = wave_component(band[i], 1)
                _, a2 = wave_component(band[i], 2)
                w1_steps.append(f1)
                a1_steps.append(a1)
                a2_steps.append(a2)
            out[h]["w1"].append(w1_steps)
            out[h]["amp"][1].append(a1_steps)
            out[h]["amp"][2].append(a2_steps)
        da.close()
    steps = [int(s) for s in ecmwf.STEPS]
    for h in BANDS:
        out[h]["w1"] = np.array(out[h]["w1"]).transpose(1, 0, 2)     # step,lev,lon
        for k in (1, 2):
            out[h]["amp"][k] = np.array(out[h]["amp"][k]).T           # step,lev
        out[h]["steps"] = steps
        out[h]["lon"] = lon
        out[h]["levs"] = np.array(levs_all, float)
    return out


def cmap_hex(name, n):
    import matplotlib.colors as mcolors
    cmap = plt.get_cmap(name)
    return [mcolors.to_hex(cmap((k + 0.5) / n)) for k in range(n)]


def render_hemi(h, D, base):
    name = f"wave1_{h}"
    outdir = ANIM / name
    outdir.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    for f in STAGING.glob("*"):
        f.unlink()
    steps, lon, levs, W = D["steps"], D["lon"], D["levs"], D["w1"]
    lab = "55–65°N" if h == "nh" else "55–65°S"
    cap = max(50, float(np.ceil(np.percentile(np.abs(W), 99.5) / 50) * 50))
    ncols = 20
    fill_cols = cmap_hex("RdBu_r", ncols)
    fill_levels = list(np.linspace(-cap, cap, ncols + 1))
    lon_p = np.where(lon < 0, lon + 360, lon)
    order = np.argsort(lon_p)
    frames_meta, frames = [], []
    for i, s in enumerate(steps):
        valid = base + pd.Timedelta(hours=s)
        fid = f"F{i:02d}"
        a10 = D["amp"][1][i][levs == 10][0]
        a100 = D["amp"][1][i][levs == 100][0]
        np.savez_compressed(STAGING / f"{fid}.npz",
                            lon=lon_p[order].astype(float), lev=levs,
                            w1=W[i][:, order].astype(float))
        meta = dict(
            out_png=fid + ".png", figsize=[11.5, 6.4],
            title=(f"Wave-1 geopotential height · {lab} · AIFS-ENS control · "
                   f"day {s // 24} · valid {valid:%a %b %d %HZ} · "
                   f"|Z₁| 100 hPa: {a100:.0f} m · 10 hPa: {a10:.0f} m"),
            footer=("ECMWF AIFS-ENS open data (CC BY 4.0) · wavenumber-1 component of Z, "
                    f"cos-weighted {lab} mean · shading fixed across the loop (±{cap:.0f} m) · "
                    "westward tilt with height = upward wave-activity flux"),
            xlabel="longitude", ylabel="pressure (hPa)",
            ylog=True, yreversed=True,
            xlim=[0.0, 360.0], ylim=[1000.0, 10.0],
            xticks=[0, 60, 120, 180, 240, 300, 360],
            xticklabels=["0°", "60°E", "120°E", "180°",
                         "120°W", "60°W", "0°"],
            yticks=[1000, 700, 500, 300, 200, 100, 50, 10],
            fill=dict(npz="w1", x="lon", y="lev", levels=fill_levels,
                      colors=fill_cols, cbar_label="wave-1 Z′ (m)"),
            contours=[dict(npz="w1", levels=[0.0], color="#555555",
                           width=0.8, dash=False)],
            arrows=None, texts=[], frame_id=fid)
        frames_meta.append(meta)
        frames.append({"idx": i, "file": f"{fid}.webp", "date": f"{valid:%Y-%m-%d}",
                       "label": f"day {s // 24} · {valid:%b %d}"})

    spec = dict(staging=str(STAGING), frames=frames_meta)
    (STAGING / "spec.json").write_text(json.dumps(spec))
    ok = False
    if shutil.which("julia"):
        try:
            r = subprocess.run(
                ["julia", "--project=" + str(JULIA_SCRIPT.parent),
                 str(JULIA_SCRIPT), str(STAGING / "spec.json")],
                capture_output=True, text=True, timeout=1200)
            ok = r.returncode == 0 and "JULIA RENDER DONE" in r.stdout
            if not ok:
                print(f"  wave1 julia failed:\n{(r.stderr or r.stdout)[-300:]}", flush=True)
        except Exception as e:                            # noqa: BLE001
            print(f"  wave1 julia failed ({str(e)[:80]})", flush=True)
    n_jl = 0
    for i, meta in enumerate(frames_meta):
        fid = meta["frame_id"]
        png = STAGING / f"{fid}.png"
        out = outdir / f"{fid}.webp"
        if ok and png.exists():
            Image.open(png).convert("RGB").save(out, quality=84, method=6)
            png.unlink()
            n_jl += 1
        else:
            _frame_mpl(meta, out)
    man = {"ver": int(time.time()), "days": len(frames),
           "regions": {name: {
               "label": f"Wave-1 geopotential ({lab}) — AIFS-ENS",
               "n_frames": len(frames), "frames": frames}}}
    (ANIM / f"{name}_manifest.json").write_text(json.dumps(man))
    print(f"wrote {len(frames)} {name} frames ({n_jl} via julia) + manifest")


def _frame_mpl(meta, out):
    z = np.load(STAGING / f"{meta['frame_id']}.npz")
    fig, ax = plt.subplots(figsize=meta["figsize"])
    f = meta["fill"]
    cf = ax.contourf(z["lon"], z["lev"], z["w1"], levels=f["levels"],
                     colors=f["colors"], extend="both")
    ax.contour(z["lon"], z["lev"], z["w1"], levels=[0.0], colors="#555555",
               linewidths=0.8)
    ax.set_yscale("log")
    ax.set_ylim(1000, 10)
    ax.set_yticks(meta["yticks"])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(meta["xticks"])
    ax.set_xticklabels(meta["xticklabels"], fontsize=9)
    ax.set_xlabel(meta["xlabel"]); ax.set_ylabel(meta["ylabel"])
    cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label(f["cbar_label"], fontsize=9)
    ax.set_title(meta["title"], fontsize=11, fontweight="bold", loc="left")
    fig.text(0.5, 0.005, meta["footer"], fontsize=8, ha="center",
             va="bottom", color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=115)
    plt.close(fig)


AMP_CLIM = Path(__file__).resolve().parent / "data" / "wave_amp_clim.nc"
PAL = {10: "#b4541f", 100: "#1f4b6e", 500: "#8a8680"}


def amplitude_chart(D, base, out):
    """Forecast stationary-wave amplitude against its day-of-year normal.

    The chart used to plot six raw curves with no reference, so a 90 m wave-1 at
    10 hPa could not be judged: that is enormous in September and unremarkable in
    January. Each level now carries its climatological amplitude for the matching
    day of year (dotted, same colour), computed from the WB2 daily Z climatology,
    and the panel titles quote the day-15 value as a percent of normal. Wave-2 is
    kept but demoted to a thin line — it is context, not the headline.

    z_ltm has 10 and 100 hPa only, so 500 hPa is drawn without a reference.
    """
    clim = xr.open_dataset(AMP_CLIM) if AMP_CLIM.exists() else None
    days = np.array(D["nh"]["steps"]) / 24.0
    doys = [(base + pd.Timedelta(days=float(x))).dayofyear for x in days]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True)
    for ax, h in zip(axes, ("nh", "sh")):
        levs = D[h]["levs"]
        pct = None
        for kl in KEY_LEVS:
            i = np.where(levs == kl)[0][0]
            c = PAL.get(kl, "#8a8680")
            a1 = D[h]["amp"][1][:, i]
            ax.plot(days, a1, color=c, lw=2.4, marker="o", ms=3.2,
                    label=f"wave-1 · {kl} hPa", zorder=4)
            ax.plot(days, D[h]["amp"][2][:, i], color=c, lw=0.9, ls=":",
                    alpha=0.55, label=f"wave-2 · {kl} hPa", zorder=3)
            key = f"{h}_{kl}_k1"
            if clim is not None and key in clim:
                ref = clim[key].values[[dd - 1 for dd in doys]]
                ax.plot(days, ref, color=c, lw=1.4, ls="--", alpha=0.75, zorder=2,
                        label=f"normal · {kl} hPa")
                if kl == 10 and ref[-1] > 1:
                    pct = 100.0 * a1[-1] / ref[-1]
        ttl = ("Northern Hemisphere (55–65°N)" if h == "nh"
               else "Southern Hemisphere (55–65°S)")
        if pct is not None:                      # second line, or it runs into the next panel
            ttl += f"\n10 hPa wave-1 at day 15: {pct:.0f}% of normal"
        ax.set_title(ttl, fontsize=10.5, fontweight="bold", loc="left", linespacing=1.4)
        ax.set_xlabel("forecast day")
        ax.grid(alpha=0.22)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("amplitude (m)")
    axes[0].legend(fontsize=6.6, ncol=3, framealpha=0.85, loc="upper center",
               bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.suptitle(f"Stationary-wave amplitude vs normal — AIFS-ENS control, "
                 f"init {base:%Y-%m-%d %HZ}", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             "Wavenumber-1 (bold) and wavenumber-2 (dotted) amplitude of geopotential height, 55–65° cos-weighted band mean. "
             "Dashed = the day-of-year normal from the WB2 daily Z climatology;\n"
             "500 hPa has no reference because that climatology carries only 10 and 100 hPa. "
             "Amplitude is what drives the vertical E–P flux, so a wave-1 well above normal is the precursor to watch.",
             fontsize=8, ha="center", va="bottom", color="#8a8680", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


def main():
    u_full, u_rmm, v_a14, tag = latest_cycle()
    base = pd.Timestamp(f"{tag[:4]}-{tag[4:6]}-{tag[6:8]} {tag[8:10]}:00")
    D = load_sections(tag)
    for h in ("nh", "sh"):
        render_hemi(h, D[h], base)
    amplitude_chart(D, base, REPO / "assets" / "sst" / "wave1_amp.webp")


if __name__ == "__main__":
    main()
