#!/usr/bin/env python3
"""Subtropical jet monitor: strength & position vs normal, observed + forecast.

Data: the 13-level AIFS-ENS u already pulled for the AAM products (cf+pf, daily
steps to day 15) → per-member zonal-mean [u]. Reference: ERA5 1991–2020 harmonic
climatology + σ(doy) envelope from build_jet_clim.py.

Renders one static figure (assets/sst/jets.webp):
  top    — [u](φ,p) cross-section at the analysis: colour = anomaly vs the ERA5
           day-of-year normal, black contours = absolute [u] (the jets), ▼ = the
           200 hPa subtropical cores.
  middle — NH & SH subtropical jet-core SPEED (max [u]@200 hPa in 15–45°|lat|)
           through the forecast: members (thin), ensemble mean (bold), ERA5
           normal (dashed) ± 1σ (band). Header gives the analysis σ-departure.
  bottom — same for the core LATITUDE (poleward/equatorward displacement).

    python src/jets.py --date 20260718 --time 00 --out ../../assets/sst/jets.webp
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store as ecmwf
from aam import DAILY_STEPS
from build_jet_clim import jet_metrics, LEVELS, JET_LEV

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
CLIM = REF / "jet_clim.nc"


def load_ubar(date: str, time: str):
    """Per-member zonal-mean [u](member, step, lev, lat) from the shared store
    (cache hits whenever the AAM heavy pull ran this cycle)."""
    cyc = ecmwf.Cycle(date, time); steps = tuple(DAILY_STEPS)
    parts = []
    for typ in ("cf", "pf"):
        up_rest = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "u", "pl",
                                               ecmwf.LEVELS_AAM_REST, steps))
        up_rmm = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", typ, "u", "pl",
                                              ecmwf.LEVELS_RMM, steps))
        kw = dict(engine="cfgrib", backend_kwargs={"indexpath": ""}, chunks={"number": 1})
        u = xr.concat([xr.open_dataset(up_rest, **kw)["u"],
                       xr.open_dataset(up_rmm, **kw)["u"]],
                      dim="isobaricInhPa").sortby("isobaricInhPa")
        u = u.sel(isobaricInhPa=LEVELS)              # files carry 14 levels (incl. 10 hPa);
                                                     # jet clim + k200 index are 13-level
        ub = u.mean("longitude")                     # zonal mean, lazily per member
        if "number" not in ub.dims:
            ub = ub.expand_dims(number=[0])
        parts.append(ub.transpose("number", "step", "isobaricInhPa", "latitude").load())
    ubar = xr.concat(parts, dim="number")
    steps_h = (ubar.step / np.timedelta64(1, "h")).values.astype(int)
    return (ubar.values, steps_h, ubar.isobaricInhPa.values.astype(float),
            ubar.latitude.values)


def _harm(coefs: np.ndarray, doy: float) -> np.ndarray:
    w = 2 * np.pi * doy / 365.25
    b = np.array([1.0, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    return np.tensordot(b, coefs, axes=(0, 0))


def _xsec_frame(ubar_k, anom_k, lat, p_hpa, lim, title, cores, clim_cores, out_fp):
    """One [u](φ,p) cross-section frame: anomaly shading, absolute contours,
    ▼ = forecast cores, ▽ = climatological core positions."""
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    cf = ax.contourf(lat, p_hpa, anom_k, levels=np.linspace(-lim, lim, 21),
                     cmap="RdBu_r", extend="both")
    cs = ax.contour(lat, p_hpa, ubar_k, levels=np.arange(10, 91, 10), colors="k", linewidths=0.8)
    ax.clabel(cs, levels=[20, 40, 60], fmt="%d", fontsize=7, inline=True)
    ax.contour(lat, p_hpa, ubar_k, levels=[0], colors="k", linewidths=1.1, linestyles="--")
    for lc in clim_cores:
        ax.plot(lc, JET_LEV, marker="v", ms=9, color="none", mec="0.45", mew=1.4, zorder=5)
    for lc in cores:
        ax.plot(lc, JET_LEV, marker="v", ms=9, color="#111", mec="w", mew=1.2, zorder=6)
    ax.set_ylim(1000, 100); ax.set_yscale("log")
    ax.set_yticks([1000, 850, 700, 500, 300, 200, 100])
    ax.set_yticklabels([1000, 850, 700, 500, 300, 200, 100])
    ax.minorticks_off()
    ax.set_xlim(-75, 75); ax.set_xticks(range(-75, 76, 15))
    ax.set_ylabel("pressure (hPa)")
    ax.set_title(title, fontsize=10.6, fontweight="bold", linespacing=1.4)
    cb = fig.colorbar(cf, ax=ax, pad=0.012)
    cb.set_label("[u] anomaly (m/s)", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, dpi=112, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_anim(ub, steps_h, p_hpa, lat, init, c, met_cl, sig_cl,
               anim_dir: Path, manifest: Path) -> None:
    """Per-forecast-day cross-section frames (ens mean, day 0..15) + manifest,
    same player conventions as the walker/WAF/torque anims. Fixed colour scale."""
    import json
    k200 = int(np.where(p_hpa == JET_LEV)[0][0])
    valid = [init + pd.Timedelta(hours=int(h)) for h in steps_h]
    doys = np.array([v.dayofyear for v in valid], float)
    ubm = ub.mean(axis=0)                                        # (step, lev, lat)
    latc, levc = c.latitude.values, c.level.values
    anoms, cores_fc = [], []
    for k in range(ubm.shape[0]):
        ucl = xr.DataArray(_harm(c["u_coeffs"].values, doys[k]),
                           dims=("level", "latitude"),
                           coords={"level": levc, "latitude": latc}
                           ).interp(latitude=lat, kwargs={"fill_value": None}).values
        anoms.append(ubm[k] - ucl)
        cores_fc.append(jet_metrics(ubm[k, k200], lat))          # (nh_s, nh_l, sh_s, sh_l)
    lim = float(np.ceil(max(float(np.nanpercentile(np.abs(np.stack(anoms)), 99.5)), 4.0)))

    anim_dir = Path(anim_dir); anim_dir.mkdir(parents=True, exist_ok=True)
    for old in anim_dir.glob("F*.webp"):
        old.unlink()
    frames = []
    for k in range(ubm.shape[0]):
        nh_s, nh_l, sh_s, sh_l = cores_fc[k]
        zn = (nh_s - met_cl[k, 0]) / sig_cl[k, 0]
        zs = (sh_s - met_cl[k, 2]) / sig_cl[k, 2]
        lead = int(round(steps_h[k] / 24))
        fp = anim_dir / f"F{k:02d}.webp"
        _xsec_frame(ubm[k], anoms[k], lat, p_hpa, lim,
                    f"Zonal-mean zonal wind — AIFS-ENS mean · init {init:%Y-%m-%d %HZ} · "
                    f"day {lead} (valid {valid[k]:%a %b %d})\n▼ = subtropical cores "
                    f"(▽ = climatological position):  NH {nh_s:.0f} m/s ({zn:+.1f}σ) · "
                    f"SH {sh_s:.0f} m/s ({zs:+.1f}σ)",
                    (nh_l, sh_l), (met_cl[k, 1], met_cl[k, 3]), fp)
        frames.append({"idx": k, "file": fp.name, "date": f"{valid[k]:%Y-%m-%d}",
                       "label": f"day {lead} · valid {valid[k]:%a %b %d} · "
                                f"NH {zn:+.1f}σ · SH {zs:+.1f}σ"})
    mani = {"ver": int(pd.Timestamp.now().timestamp()), "days": len(frames),
            "regions": {"jets": {
                "label": "Subtropical jets: zonal-mean [u] anomaly vs ERA5 1991-2020 (AIFS-ENS mean)",
                "n_frames": len(frames), "frames": frames}}}
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(mani))
    print(f"  jets anim: {len(frames)} frames (colour scale ±{lim:.0f} m/s) → {anim_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--out", default="../../assets/sst/jets.webp")
    ap.add_argument("--anim-dir", default="")          # e.g. assets/sst/anim/jets
    ap.add_argument("--manifest", default="")          # e.g. assets/sst/anim/jets_manifest.json
    args = ap.parse_args()
    if not CLIM.exists():
        print(f"clim {CLIM} missing — run build_jet_clim.py first", file=sys.stderr)
        return 1
    c = xr.open_dataset(CLIM)
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    print("== subtropical jet monitor ==", flush=True)
    ub, steps_h, p_hpa, lat = load_ubar(args.date, args.time)     # (mem,step,lev,lat)
    valid = [init + pd.Timedelta(hours=int(h)) for h in steps_h]
    k200 = int(np.where(p_hpa == JET_LEV)[0][0])

    # per-member jet metrics at every lead → (mem, step, 4)
    mets = np.array([[jet_metrics(ub[m, k, k200], lat) for k in range(ub.shape[1])]
                     for m in range(ub.shape[0])])
    mmean = mets.mean(axis=0)

    # clim curves along the forecast (evaluated per valid day-of-year)
    doys = np.array([v.dayofyear for v in valid], float)
    met_cl = np.stack([_harm(c["met_coeffs"].values, d) for d in doys])   # (step, 4)
    var_cl = np.stack([_harm(c["var_coeffs"].values, d) for d in doys])
    floor = np.maximum(float(c.attrs.get("floor_frac", 0.1)) ** 2
                       * np.abs(np.nanmean(var_cl, axis=0)), 1e-6)
    sig_cl = np.sqrt(np.maximum(var_cl, floor[None, :]))

    # analysis cross-section anomaly vs clim
    ubar0 = ub[:, 0].mean(axis=0)                                 # ens-mean analysis (lev,lat)
    uclim0 = _harm(c["u_coeffs"].values, doys[0])
    uclim0 = xr.DataArray(uclim0, dims=("level", "latitude"),
                          coords={"level": c.level.values, "latitude": c.latitude.values}
                          ).interp(latitude=lat, kwargs={"fill_value": None}).values
    anom0 = ubar0 - uclim0

    fig = plt.figure(figsize=(11.2, 11.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 1, 1], hspace=0.34, wspace=0.16,
                          left=0.065, right=0.985, top=0.94, bottom=0.075)

    ax = fig.add_subplot(gs[0, :])
    lim = max(float(np.nanpercentile(np.abs(anom0), 99.5)), 2.0)
    cf = ax.contourf(lat, p_hpa, anom0, levels=np.linspace(-lim, lim, 21),
                     cmap="RdBu_r", extend="both")
    cs = ax.contour(lat, p_hpa, ubar0, levels=np.arange(10, 91, 10), colors="k", linewidths=0.8)
    ax.clabel(cs, levels=[20, 40, 60], fmt="%d", fontsize=7, inline=True)
    ax.contour(lat, p_hpa, ubar0, levels=[0], colors="k", linewidths=1.1, linestyles="--")
    nh_s, nh_l, sh_s, sh_l = jet_metrics(ubar0[k200], lat)
    for lc in (nh_l, sh_l):
        ax.plot(lc, JET_LEV, marker="v", ms=9, color="#111", mec="w", mew=1.2, zorder=6)
    ax.set_ylim(1000, 100); ax.set_yscale("log")
    ax.set_yticks([1000, 850, 700, 500, 300, 200, 100])
    ax.set_yticklabels([1000, 850, 700, 500, 300, 200, 100])
    ax.minorticks_off()
    ax.set_xlim(-75, 75); ax.set_xticks(range(-75, 76, 15))
    ax.set_ylabel("pressure (hPa)")
    zs_n = (nh_s - met_cl[0, 0]) / sig_cl[0, 0]
    zs_s = (sh_s - met_cl[0, 2]) / sig_cl[0, 2]
    ax.set_title(f"Zonal-mean zonal wind — analysis {init:%Y-%m-%d %HZ} · colour = "
                 f"anomaly vs ERA5 1991–2020 normal\n▼ = subtropical cores:  "
                 f"NH {nh_s:.0f} m/s ({zs_n:+.1f}σ) · SH {sh_s:.0f} m/s ({zs_s:+.1f}σ)",
                 fontsize=10.6, fontweight="bold", linespacing=1.4)
    cb = fig.colorbar(cf, ax=ax, pad=0.012)
    cb.set_label("[u] anomaly (m/s)", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)

    panels = [("NH jet-core speed", 0, "m/s"), ("SH jet-core speed", 2, "m/s"),
              ("NH jet-core latitude", 1, "°lat"), ("SH jet-core latitude", 3, "°lat")]
    for i, (ttl, j, unit) in enumerate(panels):
        a = fig.add_subplot(gs[1 + i // 2, i % 2])
        a.plot(valid, mets[:, :, j].T, color="#1565c0", lw=0.4, alpha=0.12)
        a.plot(valid, mmean[:, j], color="#0d47a1", lw=2.2, label="AIFS-ENS mean")
        a.plot(valid, met_cl[:, j], color="0.35", lw=1.4, ls="--", label="ERA5 normal")
        a.fill_between(valid, met_cl[:, j] - sig_cl[:, j], met_cl[:, j] + sig_cl[:, j],
                       color="0.6", alpha=0.22, label="±1σ")
        a.grid(True, alpha=0.25)
        a.set_title(ttl + f"  ({unit})", fontsize=9.5, fontweight="bold")
        for lab in a.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right"); lab.set_fontsize(7)
        a.tick_params(labelsize=7.5)
        if i == 0:
            a.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.text(0.5, 0.006,
             "core = max zonal-mean u at 200 hPa within 15–45°|lat| (parabolic refinement) · "
             "members thin blue · clim: ERA5 1991–2020 harmonic day-of-year normal ± 1σ",
             ha="center", fontsize=8, color="0.4")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=112, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  NH {nh_s:.1f} m/s @ {nh_l:.1f}° ({zs_n:+.1f}σ) · "
          f"SH {sh_s:.1f} m/s @ {sh_l:.1f}° ({zs_s:+.1f}σ); wrote {out}")
    if args.anim_dir and args.manifest:
        build_anim(ub, steps_h, p_hpa, lat, init, c, met_cl, sig_cl,
                   Path(args.anim_dir), Path(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
