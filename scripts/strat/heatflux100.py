#!/usr/bin/env python3
"""
100 hPa eddy heat flux [v'T'], 45-75 deg: is the wave activity actually going UP?

The wave-1 maps say how big the planetary wave is and where its ridge sits. They
do not say whether it propagates into the stratosphere: that is the vertical
component of the Eliassen-Palm flux, which at 100 hPa is proportional to the
zonal-mean poleward eddy heat flux [v'T']. A large wave-1 that is vertically
stacked carries almost none; one that tilts westward with height carries a lot.
This product computes the flux itself from the AIFS-ENS forecast.

  * PER MEMBER, then averaged. The flux is quadratic, so the flux of the ensemble
    mean fades with lead as the members' phases decorrelate (the WAF lesson); the
    mean of the members' fluxes does not.
  * Wavenumbers 1..K_MAX only (K_MAX = 72, the resolution of the 2.5 deg NCEP R1
    climatology), so forecast and climatology resolve the same eddies. The k-th
    contribution is 2 Re(V_k conj(T_k)) / N^2; wave-1 and wave-2 are kept apart.
  * Positive = POLEWARD in both hemispheres (the SH flux is sign-flipped).
  * Against the 1991-2020 R1 climatology (build_vt100_clim.py): day-of-year mean,
    sd, p10-p90, and the mean/sd of the trailing 40-day mean, which is the
    quantity the vortex responds to (Polvani & Waugh 2004): a few strong days do
    little, six weeks of above-normal flux weakens the vortex.
  * Observed tail: the AIFS control's step-0 fields, which are the analysis, appended
    every cycle to assets/sst/data/heatflux100_history.json (a committed seed,
    reference/heatflux100_history_seed.json, carries the months before the product
    existed; --backfill N rebuilds it). NOT a reanalysis: PSL's R1 and R2 both stopped
    updating in March 2026 (checked 2026-09-18).

Outputs  assets/sst/heatflux100_{nh,sh}.webp  +  assets/sst/data/heatflux100.json

    python scripts/strat/heatflux100.py --date 20260918 --time 00 --out-dir assets/sst
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import store as ecmwf                                   # noqa: E402

CLIM = Path(__file__).resolve().parent / "reference" / "vt100_clim.nc"
MEMBERS = 25
K_MAX = 72
BAND = (45.0, 75.0)
OBS_DAYS = 60
INK, MUTED, GRID = "#1c2430", "#6f6b64", "#d9dde2"
C_MEAN, C_MEM, C_CTRL, C_OBS, C_K1, C_K2, C_CLIM = "#9b2d20", "#9aa3ad", "#2b5f8a", "#1c2430", "#9b2d20", "#2b5f8a", "#6f6b64"


def open_field(path, short: str):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(filter_by_keys={"shortName": short, "level": 100}, indexpath=""))
    da = ds[short] if short in ds else ds[list(ds.data_vars)[0]]
    if "number" not in da.dims:
        da = da.expand_dims("number")
    if "step" not in da.dims:
        da = da.expand_dims("step")
    return da.transpose("number", "step", "latitude", "longitude")


def fetch(cyc, typ: str, members: int = 0):
    out = {}
    for short in ("v", "t"):
        spec = ecmwf.Spec("aifs-ens", typ, short, "pl", (100,), tuple(ecmwf.STEPS), members) if typ == "pf" \
            else ecmwf.Spec("aifs-ens", typ, short, "pl", (100,), tuple(ecmwf.STEPS))
        out[short] = open_field(ecmwf.ensure(cyc, spec), short)
    return out["v"], out["t"]


def fluxes(v, t):
    """(member, step, lat, lon) -> per-latitude flux by wavenumber group, positive northward.
    Returns lat (descending as stored) and {tot, k1, k2}: (member, step, lat)."""
    a, b = np.asarray(v.values, np.float64), np.asarray(t.values, np.float64)
    n = a.shape[-1]
    V = np.fft.rfft(a, axis=-1)[..., :K_MAX + 1]; T = np.fft.rfft(b, axis=-1)[..., :K_MAX + 1]
    per_k = 2.0 * np.real(V * np.conj(T)) / n ** 2
    return v.latitude.values, {"tot": per_k[..., 1:].sum(-1), "k1": per_k[..., 1], "k2": per_k[..., 2]}


def band_mean(lat, f, north: bool):
    lo, hi = BAND if north else (-BAND[1], -BAND[0])
    m = (lat >= lo) & (lat <= hi)
    w = np.cos(np.deg2rad(lat[m])); w = w / w.sum()
    return (f[..., m] * w).sum(-1) * (1.0 if north else -1.0)


SEED = Path(__file__).resolve().parent / "reference" / "heatflux100_history_seed.json"


def analysis_flux(cyc):
    """The control's step-0 fields ARE the analysis: {hemi: {tot, k1, k2}} for one cycle."""
    out = {}
    fields = {}
    for short in ("v", "t"):
        fields[short] = open_field(ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", short, "pl", (100,), (0,))), short)
    lat, f = fluxes(fields["v"], fields["t"])
    for hemi, north in (("nh", True), ("sh", False)):
        out[hemi] = {k: round(float(band_mean(lat, f[k], north)[0, 0]), 3) for k in f}
    return out


def load_history(path: Path | None) -> dict:
    """{'YYYY-MM-DD HH': {hemi: {tot,k1,k2}}}: the committed seed under the live file."""
    h = {}
    for p in (SEED, path):
        if p is not None and Path(p).exists():
            try: h.update(json.loads(Path(p).read_text()))
            except Exception as e:                                       # noqa: BLE001
                print(f"  history {p}: unreadable ({e})", flush=True)
    return h


def history_frame(h: dict, hemi: str):
    """Daily means of the analyses on hand (00 and 12 UTC where both exist)."""
    if not h:
        return None
    df = pd.DataFrame({k: v[hemi] for k, v in h.items() if hemi in v}).T
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H")
    return df.astype(float).groupby(df.index.normalize()).mean().sort_index()


def clim_at(clim, hemi: str, key: str, stat: str, dates):
    doy = np.minimum(pd.DatetimeIndex(dates).dayofyear.values, 366) - 1
    return clim[f"{hemi}_{key}_{stat}"].values[doy]


def style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa3ad")
    ax.tick_params(colors=INK, labelsize=9)


def render(hemi, north, valid, mem, ctrl, lat, hov, obs, clim, init, n_mem, out):
    H = "Northern" if north else "Southern"
    fig = plt.figure(figsize=(13.2, 8.6), dpi=130)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], hspace=0.30, wspace=0.16, left=0.055, right=0.985, top=0.885, bottom=0.07)
    fig.suptitle(f"AIFS-ENS · 100 hPa eddy heat flux [v′T′], {int(BAND[0])}–{int(BAND[1])}°{'N' if north else 'S'} · {H} Hemisphere",
                 x=0.055, y=0.975, ha="left", fontsize=15, fontweight="bold", color=INK)
    sub_obs = "" if obs is not None else " · no analysed tail yet"
    fig.text(0.055, 0.938, f"init {init:%Y-%m-%d %H} UTC · {n_mem} members, flux per member then averaged · positive = poleward = wave activity going up · "
             f"climatology NCEP R1 1991–2020, waves 1–{K_MAX}{sub_obs}", fontsize=10, color=MUTED)
    x_all = (obs.index.tolist() if obs is not None else []) + list(valid)
    span = pd.date_range(min(x_all), max(x_all))

    # A: daily flux
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    if clim is not None:
        ax.fill_between(span, clim_at(clim, hemi, "tot", "p10", span), clim_at(clim, hemi, "tot", "p90", span), color="#e8ebef", lw=0, label="climatology 10–90%")
        ax.plot(span, clim_at(clim, hemi, "tot", "mean", span), color=C_CLIM, lw=1.6, ls="--", label="climatological mean")
    for k in range(mem["tot"].shape[0]):
        ax.plot(valid, mem["tot"][k], color=C_MEM, lw=0.6, alpha=0.7, label="members" if k == 0 else None)
    if obs is not None:
        ax.plot(obs.index, obs["tot"], color=C_OBS, lw=1.8, label="analysed (AIFS control, step 0)")
    ax.plot(valid, ctrl["tot"], color=C_CTRL, lw=1.3, ls=(0, (4, 2)), label="control")
    ax.plot(valid, mem["tot"].mean(0), color=C_MEAN, lw=2.6, label="ensemble mean")
    ax.axvline(init, color="#9aa3ad", lw=0.9, ls=":"); ax.axhline(0, color="#9aa3ad", lw=0.7)
    ax.set_ylabel("[v′T′], K m s⁻¹", fontsize=10, color=INK)
    ax.set_title("Daily flux, all waves", loc="left", fontsize=11.5, fontweight="bold", color=INK)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")

    # B: wave-1 and wave-2
    ax = fig.add_subplot(gs[0, 1]); style(ax)
    for key, col, nm in (("k1", C_K1, "wave-1"), ("k2", C_K2, "wave-2")):
        q25, q75 = np.percentile(mem[key], [25, 75], axis=0)
        ax.fill_between(valid, q25, q75, color=col, alpha=0.16, lw=0)
        ax.plot(valid, mem[key].mean(0), color=col, lw=2.3, label=f"{nm}, ensemble mean (band 25–75%)")
        if obs is not None:
            ax.plot(obs.index[-21:], obs[key].iloc[-21:], color=col, lw=1.2, alpha=0.75)
        if clim is not None:
            ax.plot(span[-(len(valid) + 21):], clim_at(clim, hemi, key, "mean", span[-(len(valid) + 21):]), color=col, lw=1.2, ls="--", label=f"{nm} climatology")
    ax.axvline(init, color="#9aa3ad", lw=0.9, ls=":"); ax.axhline(0, color="#9aa3ad", lw=0.7)
    ax.set_title("By zonal wavenumber", loc="left", fontsize=11.5, fontweight="bold", color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    # C: trailing 40-day mean as a standardised anomaly (needs the observed tail)
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    z_last = None
    if obs is not None and clim is not None and len(obs) >= 40:
        def z40(series):
            r = series.rolling(40, min_periods=40).mean().dropna()
            return (r - clim_at(clim, hemi, "tot", "m40", r.index)) / clim_at(clim, hemi, "tot", "sd40", r.index)
        zo = z40(obs["tot"])
        gap = pd.date_range(obs.index[-1] + pd.Timedelta(days=1), valid[0] - pd.Timedelta(days=1)) if valid[0] > obs.index[-1] + pd.Timedelta(days=1) else []
        zs = []
        for k in range(mem["tot"].shape[0]):
            fc = pd.Series(mem["tot"][k], index=valid)
            fc = fc[fc.index > obs.index[-1]]
            bridge = pd.Series(np.interp([g.value for g in gap], [obs.index[-1].value, fc.index[0].value], [obs["tot"].iloc[-1], fc.iloc[0]]), index=gap) if len(gap) else pd.Series(dtype=float)
            zs.append(z40(pd.concat([x for x in (obs["tot"], bridge, fc) if len(x)])).loc[lambda s: s.index > obs.index[-1]])
        Z = pd.concat(zs, axis=1)
        ax.fill_between(Z.index, Z.quantile(0.1, axis=1), Z.quantile(0.9, axis=1), color=C_MEAN, alpha=0.15, lw=0, label="members 10–90%")
        ax.plot(zo.index, zo, color=C_OBS, lw=1.8, label="analysed")
        ax.plot(Z.index, Z.mean(axis=1), color=C_MEAN, lw=2.6, label="analysed + forecast, ensemble mean")
        z_last = float(Z.mean(axis=1).iloc[-1])
        ax.axhspan(-1, 1, color="#eef0f3", lw=0, zorder=0)
        top = max(2.0, float(np.nanmax(np.abs(np.r_[zo.values[-45:], Z.values.ravel()]))) * 1.15)
        ax.set_ylim(-top, top)
        ax.text(0.995, 0.86, "above +1σ for weeks: vortex weakens", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
        ax.text(0.995, 0.04, "below −1σ for weeks: vortex strengthens", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
        ax.axvline(init, color="#9aa3ad", lw=0.9, ls=":"); ax.axhline(0, color="#9aa3ad", lw=0.7)
        ax.set_xlim(zo.index[-min(len(zo), 45)], valid[-1])
        ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3)
        ax.set_ylabel("standard deviations", fontsize=10, color=INK)
    else:
        ax.text(0.5, 0.5, "needs 40 analysed days and the climatology", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=MUTED)
    ax.set_title("Trailing 40-day mean flux, standardised (what the vortex responds to)", loc="left", fontsize=11.5, fontweight="bold", color=INK)

    # D: latitude vs lead, ensemble mean
    ax = fig.add_subplot(gs[1, 1]); style(ax); ax.grid(False)
    m = (lat >= 20) & (lat <= 88) if north else (lat <= -20) & (lat >= -88)
    F = hov[:, m].T * (1.0 if north else -1.0)
    vmax = max(10.0, float(np.ceil(np.nanpercentile(np.abs(F), 99) / 10.0) * 10.0))
    lev = np.linspace(-vmax, vmax, 21)                       # steps of vmax/10: whole numbers
    cs = ax.contourf(valid, np.abs(lat[m]), F, levels=lev, cmap="RdBu_r", extend="both")
    for b in BAND:
        ax.axhline(b, color=INK, lw=0.8, ls=(0, (4, 3)))
    cb = fig.colorbar(cs, ax=ax, pad=0.015, fraction=0.05, ticks=np.linspace(-vmax, vmax, 11)); cb.ax.tick_params(labelsize=8)
    cb.ax.set_yticklabels([f"{x:.0f}" for x in np.linspace(-vmax, vmax, 11)]); cb.set_label("K m s⁻¹", fontsize=9)
    ax.set_ylabel(f"latitude, °{'N' if north else 'S'}", fontsize=10, color=INK)
    ax.set_title("Ensemble-mean flux by latitude", loc="left", fontsize=11.5, fontweight="bold", color=INK)
    for a_ in fig.axes[:4]:
        if a_ is not cb.ax:
            loc = matplotlib.dates.AutoDateLocator(minticks=4, maxticks=7)
            a_.xaxis.set_major_locator(loc); a_.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %-d"))
    fig.savefig(out, format="webp", pil_kwargs={"quality": 88})
    plt.close(fig)
    return z_last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--out-dir", default="assets/sst"); ap.add_argument("--members", type=int, default=MEMBERS)
    ap.add_argument("--history", default="assets/sst/data/heatflux100_history.json")
    ap.add_argument("--backfill", type=int, default=0, help="rebuild the seed from the last N days of 00/12 UTC control analyses, then exit")
    a = ap.parse_args()
    if a.backfill:
        h = load_history(None); now = pd.Timestamp(f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:8]}")
        for k in range(a.backfill, -1, -1):
            for hh in ("00", "12"):
                d = now - pd.Timedelta(days=k); key = f"{d:%Y-%m-%d} {hh}"
                if key in h:
                    continue
                try:
                    h[key] = analysis_flux(ecmwf.Cycle(f"{d:%Y%m%d}", hh))
                    print(f"  {key}: NH {h[key]['nh']['tot']:+.1f}  SH {h[key]['sh']['tot']:+.1f}", flush=True)
                except Exception as e:                                   # noqa: BLE001
                    print(f"  {key}: {type(e).__name__} {str(e)[:70]}", flush=True)
            SEED.write_text(json.dumps(dict(sorted(h.items())), separators=(",", ":")))
        return 0
    cyc = ecmwf.Cycle(a.date, a.time)
    init = pd.Timestamp(f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:8]} {int(a.time):02d}:00")
    v, t = fetch(cyc, "pf", a.members)
    vc, tc = fetch(cyc, "cf")
    steps = (v.step.values / np.timedelta64(1, "h")).astype(int)
    valid = pd.DatetimeIndex([init + pd.Timedelta(hours=int(s)) for s in steps])
    lat, fm = fluxes(v, t)
    _, fc = fluxes(vc, tc)
    clim = xr.open_dataset(CLIM).load() if CLIM.exists() else None
    hpath = REPO / a.history if not Path(a.history).is_absolute() else Path(a.history)
    hist = load_history(hpath)
    try:                                               # this cycle's analysis joins the record
        _, f0 = fluxes(vc.isel(step=[0]), tc.isel(step=[0]))
        hist[f"{init:%Y-%m-%d %H}"] = {hm: {k: round(float(band_mean(lat, f0[k], nth)[0, 0]), 3) for k in f0} for hm, nth in (("nh", True), ("sh", False))}
        keep = sorted(hist)[-800:]
        hpath.parent.mkdir(parents=True, exist_ok=True)
        hpath.write_text(json.dumps({k: hist[k] for k in keep}, separators=(",", ":")))
    except Exception as e:                                       # noqa: BLE001
        print(f"  history not updated ({type(e).__name__}: {e})", flush=True)
    out_dir = REPO / a.out_dir if not Path(a.out_dir).is_absolute() else Path(a.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    doc = {"init": init.isoformat(), "members": int(v.sizes["number"]), "band": list(BAND), "k_max": K_MAX, "valid": [d.strftime("%Y-%m-%d %H:%M") for d in valid],
           "units": "K m s-1", "sign": "positive = poleward", "hemispheres": {}}
    for hemi, north in (("nh", True), ("sh", False)):
        mem = {k: band_mean(lat, fm[k], north) for k in fm}
        ctrl = {k: band_mean(lat, fc[k], north)[0] for k in fc}
        obs = history_frame(hist, hemi)
        if obs is not None:
            obs = obs[obs.index <= init.normalize()]
            gaps = np.flatnonzero(np.diff(obs.index.values) > np.timedelta64(2, "D"))     # a hole would bend the 40-day mean
            if len(gaps):
                obs = obs.iloc[gaps[-1] + 1:]
            obs = obs.asfreq("D").interpolate(limit=1).iloc[-(OBS_DAYS + 40):]
            if len(obs) < 5:
                obs = None
        z_last = render(hemi, north, valid, mem, ctrl, lat, fm["tot"].mean(0), obs, clim, init, int(v.sizes["number"]),
                        out_dir / f"heatflux100_{hemi}.webp")
        h = {k: {"mean": np.round(mem[k].mean(0), 2).tolist(), "p10": np.round(np.percentile(mem[k], 10, 0), 2).tolist(),
                 "p90": np.round(np.percentile(mem[k], 90, 0), 2).tolist(), "control": np.round(ctrl[k], 2).tolist()} for k in mem}
        if clim is not None:
            h["clim_mean"] = np.round(clim_at(clim, hemi, "tot", "mean", valid), 2).tolist()
            h["clim_sd"] = np.round(clim_at(clim, hemi, "tot", "sd", valid), 2).tolist()
            z = (mem["tot"].mean(0) - clim_at(clim, hemi, "tot", "mean", valid)) / clim_at(clim, hemi, "tot", "sd", valid)
            h["z_daily_mean"] = np.round(z, 2).tolist()
            h["k1_share_d6_15"] = round(float(mem["k1"].mean(0)[6:].mean() / max(1e-6, mem["tot"].mean(0)[6:].mean())), 2)
        if obs is not None:
            h["observed"] = {"dates": [d.strftime("%Y-%m-%d") for d in obs.index[-OBS_DAYS:]], "tot": np.round(obs["tot"].values[-OBS_DAYS:], 2).tolist()}
        h["z40_end_of_forecast"] = None if z_last is None else round(z_last, 2)
        doc["hemispheres"][hemi] = h
        print(f"  {hemi}: mean flux d1-5 {mem['tot'].mean(0)[1:6].mean():+.1f}, d6-15 {mem['tot'].mean(0)[6:].mean():+.1f} K m/s"
              + (f"; 40-d standardised at the end {z_last:+.2f}" if z_last is not None else ""), flush=True)
    (out_dir / "data" / "heatflux100.json").write_text(json.dumps(doc, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
