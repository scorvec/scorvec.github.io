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
  * Wavenumbers 1..K_MAX only (K_MAX = 72, which the climatology's 1.5 deg ERA5 grid
    and every forecast grid resolve), so forecast and climatology resolve the same
    eddies. The k-th contribution is 2 Re(V_k conj(T_k)) / N^2; wave-1 and wave-2
    are kept apart.
  * Positive = POLEWARD in both hemispheres (the SH flux is sign-flipped).
  * Against the ERA5 1991-2020 climatology (build_vt100_clim.py, the same flux from
    single 00 and 12 UTC fields): day-of-year mean, sd, p10-p90, and the mean/sd of
    the trailing 40-day mean, which is the quantity the vortex responds to (Polvani &
    Waugh 2004): a few strong days do little, six weeks of above-normal flux weakens
    the vortex.
  * Model error: every member, and the analysed tail, has the model's mean error
    against ERA5 removed where heatflux100_bias.py found it significant (by season
    and lead, measured on the model's archived runs; reference/heatflux100_bias.json),
    and nothing elsewhere.
  * SSW warning meter (Northern Hemisphere): every member's trailing 10-day mean flux through the forecast, read
    against the level that preceded the sudden warmings on record (reference/heatflux100_warning.json,
    build_hf_warning.py: ERA5 1990-2026 and the stratosphere history's CP07 central dates). The meter quotes the
    share of members crossing it, the day they first do, and the record's odds: how many SSWs had a crossing in
    the 30 days before, how far ahead, and the chance of an SSW within 30 days of a warning day against the
    base rate. Nothing on the record for the Southern Hemisphere, so no meter there.
  * Observed tail: the AIFS control's step-0 fields, which are the analysis, appended
    every cycle to assets/sst/data/heatflux100_history.json (a committed seed,
    reference/heatflux100_history_seed.json, carries the months before the product
    existed; --backfill N rebuilds it). ERA5 itself arrives about five days late.

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
from quantmet.degdays import doy366

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:  # only the AIFS fetch needs it; the GEPS/GEFS versions
    import store as ecmwf  # (heatflux100_ext.py) import this module without it
except Exception:  # noqa: BLE001
    ecmwf = None

CLIM = Path(__file__).resolve().parent / "reference" / "vt100_clim.nc"
BIAS = Path(__file__).resolve().parent / "reference" / "heatflux100_bias.json"
WARN = Path(__file__).resolve().parent / "reference" / "heatflux100_warning.json"
MEMBERS = 25
K_MAX = 72
BAND = (45.0, 75.0)
OBS_DAYS = 60
INK, MUTED, GRID = "#1c2430", "#6f6b64", "#d9dde2"
C_MEAN, C_MEM, C_CTRL, C_OBS, C_K1, C_K2, C_CLIM = (
    "#9b2d20",
    "#9aa3ad",
    "#2b5f8a",
    "#1c2430",
    "#9b2d20",
    "#2b5f8a",
    "#6f6b64",
)


def open_field(path, short: str):
    ds = xr.open_dataset(
        path, engine="cfgrib", backend_kwargs=dict(filter_by_keys={"shortName": short, "level": 100}, indexpath="")
    )
    da = ds[short] if short in ds else ds[next(iter(ds.data_vars))]
    if "number" not in da.dims:
        da = da.expand_dims("number")
    if "step" not in da.dims:
        da = da.expand_dims("step")
    return da.transpose("number", "step", "latitude", "longitude")


def fetch(cyc, typ: str, members: int = 0):
    out = {}
    for short in ("v", "t"):
        spec = (
            ecmwf.Spec("aifs-ens", typ, short, "pl", (100,), tuple(ecmwf.STEPS), members)
            if typ == "pf"
            else ecmwf.Spec("aifs-ens", typ, short, "pl", (100,), tuple(ecmwf.STEPS))
        )
        out[short] = open_field(ecmwf.ensure(cyc, spec), short)
    return out["v"], out["t"]


def fluxes(v, t):
    """(member, step, lat, lon) -> per-latitude flux by wavenumber group, positive northward.
    Returns lat (descending as stored) and {tot, k1, k2}: (member, step, lat)."""
    a, b = np.asarray(v.values, np.float64), np.asarray(t.values, np.float64)
    n = a.shape[-1]
    V = np.fft.rfft(a, axis=-1)[..., : K_MAX + 1]
    T = np.fft.rfft(b, axis=-1)[..., : K_MAX + 1]
    per_k = 2.0 * np.real(V * np.conj(T)) / n**2
    if n % 2 == 0 and n // 2 <= K_MAX:
        per_k[..., n // 2] /= 2.0  # Nyquist counts once (a no-op on the 0.25/0.5 deg grids: k=72 < n/2)
    return v.latitude.values, {"tot": per_k[..., 1:].sum(-1), "k1": per_k[..., 1], "k2": per_k[..., 2]}


def band_mean(lat, f, north: bool):
    lo, hi = BAND if north else (-BAND[1], -BAND[0])
    m = (lat >= lo) & (lat <= hi)
    w = np.cos(np.deg2rad(lat[m]))
    w = w / w.sum()
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
            try:
                h.update(json.loads(Path(p).read_text()))
            except Exception as e:  # noqa: BLE001
                print(f"  history {p}: unreadable ({e})", flush=True)
    return h


def history_frame(h: dict, hemi: str):
    """Daily means of the analyses on hand (00 and 12 UTC where both exist), less the analysis's significant mean
    error against ERA5."""
    if not h:
        return None
    df = pd.DataFrame({k: v[hemi] for k, v in h.items() if hemi in v}).T
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H")
    df = df.astype(float).groupby(df.index.normalize()).mean().sort_index()
    an = load_bias("analysis")
    return df.apply(lambda col: col - bias_at(an, hemi, col.name, df.index, 0.0))


def clim_at(clim, hemi: str, key: str, stat: str, dates):
    return clim[f"{hemi}_{key}_{stat}"].values[doy366(dates)]


def load_bias(model: str) -> dict | None:
    """One row of heatflux100_bias.json ('analysis' = the analysed record): the model's mean errors against ERA5 by
    season and lead block, or None when the model was not tested."""
    if not BIAS.exists():
        return None
    doc = json.loads(BIAS.read_text())
    tab = doc["models"].get(model)
    return None if tab is None or "untested" in tab else tab | {"season_mid": doc["season_mid"]}


def bias_at(tab, hemi: str, key: str, valid, lead_days) -> np.ndarray:
    """The model's significant mean error (K m/s, forecast minus ERA5) at each valid time and lead in days. At each
    measured lead block's centre it is the block's mean error if that is significant and zero if not (or untested);
    between the centres it is linear in lead, so it tapers to zero rather than dropping, and beyond the last block it
    keeps that block's value (zero unless significant). Between the seasons' middles it is linear in day of year."""
    valid = pd.DatetimeIndex(valid)
    if tab is None:
        return np.zeros(len(valid))
    c = tab[f"{hemi}_{key}"]
    lead = np.broadcast_to(np.asarray(lead_days, float), (len(valid),))
    b = np.where(np.array(c["significant"], dtype=bool), np.array(c["bias"], dtype=float), 0.0)  # (season, block)
    per = np.array([np.interp(lead, tab["lead_centres"], row) for row in b])  # (season, time)
    mid = doy366(pd.to_datetime([f"2000-{md}" for md in tab["season_mid"]]))
    xs = np.r_[mid[-1] - 366, mid, mid[0] + 366]  # periodic over the year
    per = np.vstack([per[-1:], per, per[:1]])
    return np.array([np.interp(s, xs, per[:, j]) for j, s in enumerate(doy366(valid))])


def load_warning() -> dict | None:
    """The SSW warning reference (build_hf_warning.py), or None without the file."""
    return json.loads(WARN.read_text()) if WARN.exists() else None


def member_paths(obs: pd.DataFrame, mem_tot: np.ndarray, valid) -> pd.DataFrame | None:
    """One daily series per member: the analysed tail, a linear bridge over any gap to the first forecast day, then
    the member's forecast at its daily valid times (the 12Z cycles' half-day offsets are kept as given)."""
    if obs is None or len(obs) < 5:
        return None
    fc0 = valid[valid > obs.index[-1]]
    if not len(fc0):
        return None
    gap = pd.date_range(obs.index[-1] + pd.Timedelta(days=1), fc0[0] - pd.Timedelta(days=1))
    cols = {}
    for k in range(mem_tot.shape[0]):
        fc = pd.Series(mem_tot[k], index=valid)
        fc = fc[fc.index > obs.index[-1]]
        bridge = (
            pd.Series(
                np.interp(
                    [g.value for g in gap], [obs.index[-1].value, fc.index[0].value], [obs["tot"].iloc[-1], fc.iloc[0]]
                ),
                index=gap,
            )
            if len(gap)
            else pd.Series(dtype=float)
        )
        cols[k] = pd.concat([x for x in (obs["tot"], bridge, fc) if len(x)])
    return pd.DataFrame(cols)


def warning_meter(obs, mem_tot, valid, ref: dict | None, init) -> dict | None:
    """The SSW warning meter's numbers: the current W-day mean, each member's maximum W-day mean through the
    forecast and the day it first reaches the threshold, and the ensemble mean's path."""
    if ref is None:
        return None
    paths = member_paths(obs, mem_tot, valid)
    if paths is None:
        return None
    W, T = int(ref["window_days"]), float(ref["threshold"])
    roll = paths.rolling(W, min_periods=W).mean()
    now = obs["tot"].rolling(W, min_periods=W).mean().dropna()
    if now.empty:
        return None
    fc = roll[roll.index > obs.index[-1]]
    if fc.empty:
        return None
    mx = fc.max(axis=0)
    first = fc.ge(T).idxmax().where(fc.ge(T).any())  # NaT where a member never crosses
    lead_days = (first.dropna() - init.normalize()) / pd.Timedelta(days=1)
    ens = fc.mean(axis=1)
    return {
        "window_days": W,
        "threshold": T,
        "now": round(float(now.iloc[-1]), 2),
        "now_date": now.index[-1].strftime("%Y-%m-%d"),
        "horizon_days": round((fc.index[-1] - init.normalize()) / pd.Timedelta(days=1)),
        "members_crossing": round(float(fc.ge(T).any().mean()), 3),
        "first_crossing_day_median": None if lead_days.empty else round(float(lead_days.median())),
        "member_max": {k: round(float(mx.quantile(q)), 2) for k, q in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))},
        "ens_mean_max": round(float(ens.max()), 2),
        "ens_mean_end": round(float(ens.iloc[-1]), 2),
        "ens_mean_path": {d.strftime("%Y-%m-%d"): round(float(v), 2) for d, v in ens.items()},
        "record": {
            k: ref[k]
            for k in (
                "n_events",
                "hit_rate",
                "false_alarm_rate",
                "peirce_skill",
                "precision",
                "precision_ci90",
                "base_rate",
                "lead_days_first_warning",
                "winter_metric_quantiles",
                "record",
            )
        },
    }


def draw_meter(ax, warn: dict | None, north: bool, n_mem: int):
    """The SSW warning meter: a gauge of the W-day mean flux with the record's warning level, where the analysis
    stands now, the members' maxima through the forecast and the record's odds."""
    ax.set_facecolor("#ffffff")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c9cfd6")
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    ax.set_yticks([])
    if not north or warn is None:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.text(
            0.0,
            0.5,
            "No SSW warning meter: the record of sudden warmings used for it is Northern Hemisphere only."
            if not north
            else "SSW warning meter needs the analysed tail and the warning reference.",
            transform=ax.transAxes,
            fontsize=9.5,
            color=MUTED,
            va="center",
        )
        return None
    T, W, rec = warn["threshold"], warn["window_days"], warn["record"]
    q = rec["winter_metric_quantiles"]
    hi = max(45.0, float(np.ceil(max(warn["member_max"]["p90"], warn["now"], T) / 5.0) * 5.0 + 5.0))
    ax.set_xlim(0, hi)
    ax.set_ylim(0, 1)
    ax.axvspan(0, T, color="#eef0f3", lw=0, zorder=0)
    ax.axvspan(T, hi, color="#fbe9e4", lw=0, zorder=0)
    for key, lab, y, va in (("p50", "winter median", 0.97, "top"), ("p90", "winter p90", 0.04, "bottom")):
        ax.axvline(q[key], color="#9aa3ad", lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.text(q[key] - 0.3, y, lab, fontsize=7.5, color=MUTED, ha="right", va=va)
    ax.axvline(T, color="#b3261e", lw=2.0, zorder=3)
    ax.text(
        T + 0.4, 0.97, f"SSW warning level {T:g}", fontsize=8.5, color="#b3261e", ha="left", va="top", fontweight="bold"
    )
    # members' maxima through the forecast: a 10-90 % bar with the median, the ensemble mean's maximum as a square
    mm = warn["member_max"]
    ax.plot([mm["p10"], mm["p90"]], [0.42, 0.42], color=C_MEAN, lw=7, solid_capstyle="butt", alpha=0.45, zorder=4)
    ax.plot([mm["p50"]], [0.42], marker="|", ms=16, mew=2.2, color=C_MEAN, zorder=5)
    ax.plot([warn["ens_mean_max"]], [0.42], marker="s", ms=8, color=C_MEAN, mec="white", mew=1.2, zorder=6)
    ax.plot([warn["now"]], [0.42], marker="D", ms=8, color=C_OBS, mec="white", mew=1.2, zorder=7)
    ax.set_xlabel(f"trailing {W}-day mean [v′T′], K m s⁻¹", fontsize=9, color=INK, labelpad=2)
    pct = round(100 * warn["members_crossing"])
    when = warn["first_crossing_day_median"]
    lead = rec["lead_days_first_warning"]
    line1 = (
        f"◆ now {warn['now']:.1f} ({warn['now_date'][5:]}) · ■ ensemble-mean maximum {warn['ens_mean_max']:.1f} · "
        f"bar: members' maxima, 10–90 % · {pct} % of {n_mem} members reach the level by day {warn['horizon_days']}"
        + (f", median first crossing day {when}" if when is not None else "")
    )
    line2 = (
        f"Record 1991–2026: {round(rec['hit_rate'] * rec['n_events'])} of {rec['n_events']} SSWs had the {W}-day mean at this level in the 30 days before "
        f"(median {lead['median']:.0f} days ahead); {100 * rec['false_alarm_rate']:.0f} % of quiet Nov–Mar windows reach it too."
    )
    line3 = (
        f"On a warning day the chance of an SSW within 30 days is {100 * rec['precision']:.0f} % "
        f"[{100 * rec['precision_ci90'][0]:.0f}–{100 * rec['precision_ci90'][1]:.0f}] against {100 * rec['base_rate']:.0f} % on any Nov–Mar day: "
        "three times the odds, not a forecast of a warming."
    )
    return line1, line2, line3


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa3ad")
    ax.tick_params(colors=INK, labelsize=9)


def bias_note(tab, corrected: bool) -> str:
    """The subtitle's account of the model-error correction."""
    if tab is None:
        return "not tested against ERA5 (too few archived runs), uncorrected"
    if corrected:
        return "members less the model's mean error against ERA5 where significant"
    return "no significant mean error against ERA5 at these dates and leads"


def render(hemi, north, valid, mem, ctrl, lat, hov, obs, clim, init, n_mem, out, note, model="AIFS-ENS", warn=None):
    H = "Northern" if north else "Southern"
    fig = plt.figure(figsize=(13.2, 10.6), dpi=130)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.55, 1],
        height_ratios=[1, 1, 0.3],
        hspace=0.36,
        wspace=0.16,
        left=0.055,
        right=0.955,
        top=0.89,
        bottom=0.125,
    )  # right margin: room for the colour bar's label (the figure is saved uncropped)
    fig.suptitle(
        f"{model} · 100 hPa eddy heat flux [v′T′], {int(BAND[0])}–{int(BAND[1])}°{'N' if north else 'S'} · {H} Hemisphere",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    sub_obs = "" if obs is not None else " · no analysed tail yet"
    fig.text(
        0.055,
        0.948,
        f"init {init:%Y-%m-%d %H} UTC · {n_mem} members, flux per member then averaged · positive = poleward = wave activity going up\n"
        f"climatology ERA5 1991–2020, waves 1–{K_MAX} · {note}{sub_obs}",
        fontsize=10,
        color=MUTED,
        va="top",
        linespacing=1.45,
    )
    x_all = (obs.index.tolist() if obs is not None else []) + list(valid)
    span = pd.date_range(min(x_all), max(x_all))

    # A: daily flux
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    if clim is not None:
        ax.fill_between(
            span,
            clim_at(clim, hemi, "tot", "p10", span),
            clim_at(clim, hemi, "tot", "p90", span),
            color="#e8ebef",
            lw=0,
            label="climatology 10–90%",
        )
        ax.plot(
            span, clim_at(clim, hemi, "tot", "mean", span), color=C_CLIM, lw=1.6, ls="--", label="climatological mean"
        )
    for k in range(mem["tot"].shape[0]):
        ax.plot(valid, mem["tot"][k], color=C_MEM, lw=0.6, alpha=0.7, label="members" if k == 0 else None)
    if obs is not None:
        ax.plot(obs.index, obs["tot"], color=C_OBS, lw=1.8, label="analysed (AIFS control, step 0)")
    if ctrl is not None:  # None when the run's control did not arrive
        ax.plot(valid, ctrl["tot"], color=C_CTRL, lw=1.3, ls=(0, (4, 2)), label="control")
    ax.plot(valid, mem["tot"].mean(0), color=C_MEAN, lw=2.6, label="ensemble mean")
    ax.axvline(init, color="#9aa3ad", lw=0.9, ls=":")
    ax.axhline(0, color="#9aa3ad", lw=0.7)
    ax.set_ylabel("[v′T′], K m s⁻¹", fontsize=10, color=INK)
    ax.set_title("Daily flux, all waves", loc="left", fontsize=11.5, fontweight="bold", color=INK)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.85, edgecolor="none", ncol=3, loc="upper left")

    # B: wave-1 and wave-2
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    for key, col, nm in (("k1", C_K1, "wave-1"), ("k2", C_K2, "wave-2")):
        q25, q75 = np.percentile(mem[key], [25, 75], axis=0)
        ax.fill_between(valid, q25, q75, color=col, alpha=0.16, lw=0)
        ax.plot(valid, mem[key].mean(0), color=col, lw=2.3, label=f"{nm} ensemble mean, 25–75 %")
        if obs is not None:
            ax.plot(obs.index[-21:], obs[key].iloc[-21:], color=col, lw=1.2, alpha=0.75)
        if clim is not None:
            # the mean and the 90th percentile: a wave's flux above its own 90th percentile is rare for the date
            for stat, ls, lab in (("mean", "--", "climatology"), ("p90", ":", "90th percentile")):
                ax.plot(
                    span[-(len(valid) + 21) :],
                    clim_at(clim, hemi, key, stat, span[-(len(valid) + 21) :]),
                    color=col,
                    lw=1.2,
                    ls=ls,
                    label=f"{nm} {lab}",
                )
    ax.axvline(init, color="#9aa3ad", lw=0.9, ls=":")
    ax.axhline(0, color="#9aa3ad", lw=0.7)
    ax.set_title("By zonal wavenumber", loc="left", fontsize=11.5, fontweight="bold", color=INK)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * 1.45)  # headroom: the three-row legend sits above the lines, not over them
    ax.legend(
        fontsize=8.5,
        frameon=True,
        framealpha=0.85,
        edgecolor="none",
        loc="upper left",
        ncol=2,
        handlelength=1.6,
        columnspacing=0.9,
    )

    # C: trailing 40-day mean as a standardised anomaly (needs the observed tail)
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    z_last = None
    if obs is not None and clim is not None and len(obs) >= 40:

        def z40(series):
            r = series.rolling(40, min_periods=40).mean().dropna()
            return (r - clim_at(clim, hemi, "tot", "m40", r.index)) / clim_at(clim, hemi, "tot", "sd40", r.index)

        zo = z40(obs["tot"])
        gap = (
            pd.date_range(obs.index[-1] + pd.Timedelta(days=1), valid[0] - pd.Timedelta(days=1))
            if valid[0] > obs.index[-1] + pd.Timedelta(days=1)
            else []
        )
        zs = []
        for k in range(mem["tot"].shape[0]):
            fc = pd.Series(mem["tot"][k], index=valid)
            fc = fc[fc.index > obs.index[-1]]
            bridge = (
                pd.Series(
                    np.interp(
                        [g.value for g in gap],
                        [obs.index[-1].value, fc.index[0].value],
                        [obs["tot"].iloc[-1], fc.iloc[0]],
                    ),
                    index=gap,
                )
                if len(gap)
                else pd.Series(dtype=float)
            )
            zs.append(
                z40(pd.concat([x for x in (obs["tot"], bridge, fc) if len(x)])).loc[lambda s: s.index > obs.index[-1]]
            )
        Z = pd.concat(zs, axis=1)
        ax.fill_between(
            Z.index,
            Z.quantile(0.1, axis=1),
            Z.quantile(0.9, axis=1),
            color=C_MEAN,
            alpha=0.15,
            lw=0,
            label="members 10–90%",
        )
        ax.plot(zo.index, zo, color=C_OBS, lw=1.8, label="analysed")
        ax.plot(Z.index, Z.mean(axis=1), color=C_MEAN, lw=2.6, label="analysed + forecast, ensemble mean")
        z_last = float(Z.mean(axis=1).iloc[-1])
        ax.axhspan(-1, 1, color="#eef0f3", lw=0, zorder=0)
        top = max(2.0, float(np.nanmax(np.abs(np.r_[zo.values[-45:], Z.values.ravel()]))) * 1.15)
        ax.set_ylim(-top, top)
        ax.text(
            0.995,
            0.86,
            "above +1σ for weeks: vortex weakens",
            transform=ax.transAxes,
            ha="right",
            fontsize=8.5,
            color=MUTED,
        )
        ax.text(
            0.995,
            0.04,
            "below −1σ for weeks: vortex strengthens",
            transform=ax.transAxes,
            ha="right",
            fontsize=8.5,
            color=MUTED,
        )
        ax.axvline(init, color="#9aa3ad", lw=0.9, ls=":")
        ax.axhline(0, color="#9aa3ad", lw=0.7)
        ax.set_xlim(zo.index[-min(len(zo), 45)], valid[-1])
        ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3)
        ax.set_ylabel("standard deviations", fontsize=10, color=INK)
    else:
        ax.text(
            0.5,
            0.5,
            "needs 40 analysed days and the climatology",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color=MUTED,
        )
    ax.set_title(
        "Trailing 40-day mean flux, standardised (what the vortex responds to)",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )

    # D: latitude vs lead, ensemble mean
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    ax.grid(False)
    m = (lat >= 20) & (lat <= 88) if north else (lat <= -20) & (lat >= -88)
    F = hov[:, m].T * (1.0 if north else -1.0)
    vmax = max(10.0, float(np.ceil(np.nanpercentile(np.abs(F), 99) / 10.0) * 10.0))
    lev = np.linspace(-vmax, vmax, 21)  # steps of vmax/10: whole numbers
    cs = ax.contourf(valid, np.abs(lat[m]), F, levels=lev, cmap="RdBu_r", extend="both")
    for b in BAND:
        ax.axhline(b, color=INK, lw=0.8, ls=(0, (4, 3)))
    cb = fig.colorbar(cs, ax=ax, pad=0.015, fraction=0.05, ticks=np.linspace(-vmax, vmax, 11))
    cb.ax.tick_params(labelsize=8)
    cb.ax.set_yticklabels([f"{x:.0f}".replace("-", "−") for x in np.linspace(-vmax, vmax, 11)])
    cb.set_label("[v′T′], K m s⁻¹ (+ = poleward)", fontsize=9, color=INK)
    ax.set_ylabel(f"latitude, °{'N' if north else 'S'}", fontsize=10, color=INK)
    ax.set_title(
        f"Ensemble mean by latitude (dashed: {int(BAND[0])}–{int(BAND[1])}° band)",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    for a_ in fig.axes[:4]:
        if a_ is not cb.ax:
            loc = matplotlib.dates.AutoDateLocator(minticks=4, maxticks=7)
            a_.xaxis.set_major_locator(loc)
            a_.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %-d"))

    # E: the SSW warning meter
    ax = fig.add_subplot(gs[2, :])
    lines = draw_meter(ax, warn, north, n_mem)
    if lines:
        for y, txt, col in zip((0.058, 0.040, 0.022), lines, (INK, MUTED, MUTED)):
            fig.text(0.055, y, txt, fontsize=8.6, color=col, va="top")
    ax.set_title(
        "SSW warning meter: the 10-day mean flux against the level that preceded the warmings on record",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    fig.savefig(out, format="webp", pil_kwargs={"quality": 88})
    plt.close(fig)
    return z_last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--out-dir", default="assets/sst")
    ap.add_argument("--members", type=int, default=MEMBERS)
    ap.add_argument("--history", default="assets/sst/data/heatflux100_history.json")
    ap.add_argument(
        "--backfill",
        type=int,
        default=0,
        help="rebuild the seed from the last N days of 00/12 UTC control analyses, then exit",
    )
    a = ap.parse_args()
    if a.backfill:
        h = load_history(None)
        now = pd.Timestamp(f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:8]}")
        for k in range(a.backfill, -1, -1):
            for hh in ("00", "12"):
                d = now - pd.Timedelta(days=k)
                key = f"{d:%Y-%m-%d} {hh}"
                if key in h:
                    continue
                try:
                    h[key] = analysis_flux(ecmwf.Cycle(f"{d:%Y%m%d}", hh))
                    print(f"  {key}: NH {h[key]['nh']['tot']:+.1f}  SH {h[key]['sh']['tot']:+.1f}", flush=True)
                except Exception as e:  # noqa: BLE001
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
    try:  # this cycle's analysis joins the record
        _, f0 = fluxes(vc.isel(step=[0]), tc.isel(step=[0]))
        hist[f"{init:%Y-%m-%d %H}"] = {
            hm: {k: round(float(band_mean(lat, f0[k], nth)[0, 0]), 3) for k in f0}
            for hm, nth in (("nh", True), ("sh", False))
        }
        keep = sorted(hist)[-800:]
        hpath.parent.mkdir(parents=True, exist_ok=True)
        hpath.write_text(json.dumps({k: hist[k] for k in keep}, separators=(",", ":")))
    except Exception as e:  # noqa: BLE001
        print(f"  history not updated ({type(e).__name__}: {e})", flush=True)
    out_dir = REPO / a.out_dir if not Path(a.out_dir).is_absolute() else Path(a.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    doc = {
        "init": init.isoformat(),
        "members": int(v.sizes["number"]),
        "band": list(BAND),
        "k_max": K_MAX,
        "valid": [d.strftime("%Y-%m-%d %H:%M") for d in valid],
        "units": "K m s-1",
        "sign": "positive = poleward",
        "hemispheres": {},
    }
    tab = load_bias("aifs")
    wref = load_warning()
    lead = (valid - init) / pd.Timedelta(days=1)
    for hemi, north in (("nh", True), ("sh", False)):
        err = {k: bias_at(tab, hemi, k, valid, lead) for k in fm}  # zero where no significant error was found
        mem = {k: band_mean(lat, fm[k], north) - err[k] for k in fm}
        ctrl = {k: band_mean(lat, fc[k], north)[0] - err[k] for k in fc}
        obs = history_frame(hist, hemi)
        if obs is not None:
            obs = obs[obs.index <= init.normalize()]
            gaps = np.flatnonzero(
                np.diff(obs.index.values) > np.timedelta64(2, "D")
            )  # a hole would bend the 40-day mean
            if len(gaps):
                obs = obs.iloc[gaps[-1] + 1 :]
            obs = obs.asfreq("D").interpolate(limit=1).iloc[-(OBS_DAYS + 40) :]
            if len(obs) < 5:
                obs = None
        warn = warning_meter(obs, mem["tot"], valid, wref, init) if north else None
        z_last = render(
            hemi,
            north,
            valid,
            mem,
            ctrl,
            lat,
            fm["tot"].mean(0),
            obs,
            clim,
            init,
            int(v.sizes["number"]),
            out_dir / f"heatflux100_{hemi}.webp",
            bias_note(tab, any(np.any(e != 0) for e in err.values())),
            warn=warn,
        )
        h = {
            k: {
                "mean": np.round(mem[k].mean(0), 2).tolist(),
                "p10": np.round(np.percentile(mem[k], 10, 0), 2).tolist(),
                "p90": np.round(np.percentile(mem[k], 90, 0), 2).tolist(),
                "control": np.round(ctrl[k], 2).tolist(),
                "model_error_removed": np.round(err[k], 2).tolist(),
            }
            for k in mem
        }
        if clim is not None:
            h["clim_mean"] = np.round(clim_at(clim, hemi, "tot", "mean", valid), 2).tolist()
            h["clim_sd"] = np.round(clim_at(clim, hemi, "tot", "sd", valid), 2).tolist()
            z = (mem["tot"].mean(0) - clim_at(clim, hemi, "tot", "mean", valid)) / clim_at(
                clim, hemi, "tot", "sd", valid
            )
            h["z_daily_mean"] = np.round(z, 2).tolist()
            h["k1_share_d6_15"] = round(
                float(mem["k1"].mean(0)[6:].mean() / max(1e-6, mem["tot"].mean(0)[6:].mean())), 2
            )
        if obs is not None:
            h["observed"] = {
                "dates": [d.strftime("%Y-%m-%d") for d in obs.index[-OBS_DAYS:]],
                "tot": np.round(obs["tot"].values[-OBS_DAYS:], 2).tolist(),
            }
        h["z40_end_of_forecast"] = None if z_last is None else round(z_last, 2)
        if warn is not None:
            h["ssw_warning"] = warn
        doc["hemispheres"][hemi] = h
        print(
            f"  {hemi}: mean flux d1-5 {mem['tot'].mean(0)[1:6].mean():+.1f}, d6-15 {mem['tot'].mean(0)[6:].mean():+.1f} K m/s"
            + (f"; 40-d standardised at the end {z_last:+.2f}" if z_last is not None else ""),
            flush=True,
        )
    (out_dir / "data" / "heatflux100.json").write_text(json.dumps(doc, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
