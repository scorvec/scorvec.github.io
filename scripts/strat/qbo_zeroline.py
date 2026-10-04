#!/usr/bin/env python3
"""QBO zero-wind line tracker (2026-09-28; user: "what could we add to see the impact of the QBO zero-wind line").

Holton and Tan (1980): planetary waves propagating up and equatorward from the winter midlatitudes meet a critical line
where the zonal-mean wind [u] turns easterly. With easterly QBO winds in the lower stratosphere that line sits in the
winter subtropics, the extratropical waveguide is narrower and more wave activity is steered toward the vortex. This
product tracks WHERE that line is, day by day, at 70, 50, 30 and 10 hPa.

THE INDEX (zero_line): [u] on a latitude grid, smoothed with a Gaussian of sigma 2 deg latitude; start at the latitude of
the strongest westerlies between 25 and 45 deg in the hemisphere asked for (if even that maximum is <= 0 there are no
winter westerlies to bound: the index is undefined, flag 1 - most of May-September in the NH); scan toward and across
the equator; the index is the first zero crossing (linear interpolation) whose easterly region beyond it reaches at
least UMIN = 1 m/s. Shallower dips (|u| < 1 m/s, inside analysis noise) are stepped over, so a -0.3 m/s wobble at 18N
in a westerly-QBO autumn does not count as a critical line. No qualifying crossing before 45 deg in the other
hemisphere: censored at that bound (flag 2). Positive = north. The choice of sigma/UMIN is documented and tested in
build_qbo_zeroline.py (made on robustness, before looking at any vortex relation).

Subcommands (live, in Actions):
  forecast --model aifs|ifs   per-member [u] at 50 and 10 hPa saved by qbo_duct.py (data/zmu_members_<model>.npz; the
                              E-P flux loop's own fields, no extra download) -> assets/sst/data/qbo_zeroline_<model>.json
  render                      MERRA-2 reference (reference/qbo_zeroline_ref.nc) + GEOS FP analysis tail (the BDC
                              product's daily 00Z zonal means, frames-branch cache assets/sst/anim/bdc_tail) + the
                              forecast JSONs -> assets/sst/qbo_zeroline_<lev>.webp, assets/sst/data/qbo_zeroline.json
ECMWF open data has u at 50 and 10 hPa but not 70 or 30, so those two levels are analysis only.
"""
from __future__ import annotations

import argparse
import glob
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REF = HERE / "reference" / "qbo_zeroline_ref.nc"
TEST = HERE / "reference" / "qbo_zeroline_test.json"
LEVELS = (70, 50, 30, 10)
FC_LEVELS = (50, 10)
SIGMA, UMIN, START, BOUND = 2.0, 1.0, (25.0, 45.0), 45.0
LAT = np.arange(-50.0, 50.01, 1.0)                        # common 1 deg grid for sections
MODEL = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS"}
INK, MUTED, GRID = "#1c2430", "#6f6b64", "#d9dde2"
COL = {"ana": "#eb6834", "aifs": "#4a3aa7", "ifs": "#138a5e", "clim": "#8a8780"}


# ------------------------------------------------------------------------------------------------------------ index ---
def zero_line(u, lat, hemi="nh", sigma=SIGMA, umin=UMIN, start=START, bound=BOUND):
    """u (..., lat) on any latitude order -> (index lat, flag) with the leading shape. flag 0 ok, 1 undefined (no
    winter westerlies), 2 censored at the far bound. SH = the same rule on the mirrored grid (index negative)."""
    lat = np.asarray(lat, float)
    u = np.asarray(u, float)
    o = np.argsort(lat)
    lat, u = lat[o], u[..., o]
    if hemi == "sh":
        lat, u = -lat[::-1], u[..., ::-1]
    if sigma > 0:
        u = gaussian_filter1d(u, sigma / abs(lat[1] - lat[0]), axis=-1, mode="nearest")
    shp = u.shape[:-1]
    U = u.reshape(-1, u.shape[-1])
    keep = lat >= -bound
    la, U = lat[keep], U[:, keep]
    st = np.where((la >= start[0]) & (la <= start[1]))[0]
    z = np.full(U.shape[0], np.nan)
    fl = np.zeros(U.shape[0], int)
    for k in range(U.shape[0]):
        row = U[k]
        if not np.isfinite(row).all():
            fl[k] = 1; continue
        j = st[np.argmax(row[st])]
        if row[j] <= 0:
            fl[k] = 1; continue
        found = False
        while j > 0:
            if row[j - 1] <= 0 < row[j]:
                jj = j - 1
                while jj >= 0 and row[jj] <= 0:
                    jj -= 1
                if row[jj + 1:j].min() <= -umin:
                    z[k] = la[j - 1] + (la[j] - la[j - 1]) * (0 - row[j - 1]) / (row[j] - row[j - 1])
                    found = True
                    break
                if jj < 0:
                    break
                j = jj + 1
                continue
            j -= 1
        if not found:
            z[k], fl[k] = -bound, 2
    z = z.reshape(shp)
    return (-z if hemi == "sh" else z), fl.reshape(shp)


def to_grid(u, lat, grid=LAT):
    """Interpolate (..., lat) onto the common grid (ascending)."""
    lat = np.asarray(lat, float)
    o = np.argsort(lat)
    lat, u = lat[o], np.asarray(u, float)[..., o]
    flat = u.reshape(-1, u.shape[-1])
    return np.stack([np.interp(grid, lat, r) for r in flat]).reshape(u.shape[:-1] + (len(grid),))


# --------------------------------------------------------------------------------------------------------- forecast ---
def forecast(model, npz=None, out_dir=REPO / "assets" / "sst" / "data"):
    npz = Path(npz or HERE / "data" / f"zmu_members_{model}.npz")
    if not npz.exists():
        raise SystemExit(f"{npz.name}: not written this run (qbo_duct.py found no {MODEL[model]} cycle); "
                         f"the figure keeps the analysis and any recent forecast JSON")
    d = np.load(npz)
    init = pd.Timestamp(str(d["init"]))
    steps = [int(s) for s in d["steps"]]
    lat = d["lat"]
    out = {"model": MODEL[model], "init": init.strftime("%Y-%m-%d %HZ"), "rule": rule_text(), "levels": {}}
    for L in FC_LEVELS:
        if f"u{L}" not in d:
            continue
        u = d[f"u{L}"]                                        # (step, member, lat)
        z, fl = zero_line(u, lat, "nh")
        zs, fls = zero_line(u, lat, "sh")
        rec = {"valid": [(init + pd.Timedelta(hours=s)).strftime("%Y-%m-%d") for s in steps],
               "members": int(u.shape[1]), "mean_u": np.round(to_grid(u.mean(axis=1), lat), 2).tolist()}
        for tag, zz, ff in (("nh", z, fl), ("sh", zs, fls)):
            q = np.where(ff == 1, np.nan, zz)
            with np.errstate(all="ignore"):
                pct = np.nanpercentile(q, [10, 50, 90], axis=1)
            rec[tag] = {"p10": _r(pct[0]), "p50": _r(pct[1]), "p90": _r(pct[2]),
                        "defined": (ff != 1).sum(axis=1).tolist(), "censored": (ff == 2).sum(axis=1).tolist()}
        out["levels"][str(L)] = rec
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"qbo_zeroline_{model}.json").write_text(json.dumps(out, separators=(",", ":")))
    l50 = out["levels"].get("50", {}).get("nh", {})
    print(f"  {MODEL[model]} {out['init']}: 50 hPa NH zero line day 0 {l50.get('p50', [None])[0]}, "
          f"day 15 {l50.get('p50', [None])[-1]} ({out['levels'].get('50', {}).get('members')} members)", flush=True)


def _r(a):
    return [None if not np.isfinite(x) else round(float(x), 2) for x in np.asarray(a, float)]


def rule_text():
    return (f"first zero crossing of [u] (sigma {SIGMA:g} deg smoothing) scanning equatorward from the strongest "
            f"25-45 deg westerlies whose easterly region reaches {UMIN:g} m/s; censored at {BOUND:g} deg in the other hemisphere")


# ----------------------------------------------------------------------------------------------------------- render ---
def load_tail(tail_dir):
    """GEOS FP 00Z zonal means, one npz a day (bdc_dc.py's analysis cache) -> (dates, {lev: (day, LAT)})."""
    files = sorted(glob.glob(str(Path(tail_dir) / "*.npz")))
    days, rows = [], {L: [] for L in LEVELS}
    for f in files:
        try:
            z = np.load(f)
            lev = np.asarray(z["lev"], float)
            ub = np.asarray(z["ubar"], float)
            if not np.isfinite(ub).any():
                continue
            for L in LEVELS:
                rows[L].append(to_grid(ub[int(np.argmin(np.abs(lev - L)))], z["lat"]))
            days.append(pd.Timestamp(Path(f).stem))
        except Exception as e:                                                   # noqa: BLE001
            print(f"  tail {Path(f).name}: skipped ({str(e)[:60]})", flush=True)
    return pd.DatetimeIndex(days), {L: np.array(rows[L]) for L in LEVELS}


def broken(x, z, fl=None, jump=8.0):
    """Line arrays with NaN breaks where the index jumps more than `jump` degrees in a day (a flip between two
    crossings is a jump, not a path) and at censored days (drawn separately)."""
    z = np.asarray(z, float).copy()
    if fl is not None:
        z[np.asarray(fl) == 2] = np.nan
    xs, zs = [], []
    for i in range(len(z)):
        if i and np.isfinite(z[i]) and np.isfinite(z[i - 1]) and abs(z[i] - z[i - 1]) > jump:
            xs.append(np.nan); zs.append(np.nan)
        xs.append(x[i]); zs.append(z[i])
    return np.array(xs), np.array(zs)


def style(ax):
    ax.tick_params(colors=INK, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa3ad")


def lat_fmt(v, _=None):
    return "EQ" if abs(v) < 0.01 else f"{abs(v):.0f}°{'N' if v > 0 else 'S'}"


def render(out_dir=REPO / "assets" / "sst", data_dir=None, tail_dir=None, days_back=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates
    import xarray as xr

    out_dir = Path(out_dir)
    data_dir = Path(data_dir or out_dir / "data")
    ref = xr.open_dataset(REF)
    test = json.loads(TEST.read_text()) if TEST.exists() else {}
    tail_dir = Path(tail_dir or out_dir / "anim" / "bdc_tail")
    if not list(tail_dir.glob("*.npz")) and (HERE / "data" / "bdc_dc" / "fp_tail").exists():
        tail_dir = HERE / "data" / "bdc_dc" / "fp_tail"                  # laptop fallback
    tdays, tail = load_tail(tail_dir)
    fc = {}
    for m in ("aifs", "ifs"):
        p = data_dir / f"qbo_zeroline_{m}.json"
        if p.exists():
            fc[m] = json.loads(p.read_text())
    # the forecasts must be recent to be drawn: a stale JSON (a model that has not run for days) is dropped
    now = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    for m in list(fc):
        if pd.Timestamp(fc[m]["init"][:10]) < now - pd.Timedelta(days=3):
            print(f"  {MODEL[m]} forecast {fc[m]['init']} is stale; not drawn", flush=True); fc.pop(m)

    m2_time = pd.DatetimeIndex(ref["recent_time"].values)
    m2_end = m2_time[-1]
    ana_end = max([m2_end] + ([tdays[-1]] if len(tdays) else []))
    t0 = ana_end - pd.Timedelta(days=days_back)
    summary = {"as_of": ana_end.strftime("%Y-%m-%d"), "merra2_end": m2_end.strftime("%Y-%m-%d"),
               "geosfp_days": int(len(tdays)), "rule": rule_text(), "levels": {}, "forecasts": {}}
    clim_doy = ref["doy"].values
    comp_day = pd.DatetimeIndex(ref["cday"].values)                          # a nominal Jul-Mar calendar

    for L in LEVELS:
        # --- the analysed record on the common grid: MERRA-2, then GEOS FP after MERRA-2 ends
        u_m2 = ref["recent_u"].sel(lev=L).values                            # (time, LAT)
        keep_t = tdays > m2_end
        times = m2_time.append(tdays[keep_t])
        U = np.concatenate([u_m2, tail[L][keep_t]]) if keep_t.any() else u_m2
        sel = times >= t0
        times, U = times[sel], U[sel]
        z, fl = zero_line(U, LAT, "nh")
        z = np.where(fl == 1, np.nan, z)
        zs, fls = zero_line(U, LAT, "sh")
        zs = np.where(fls == 1, np.nan, zs)
        summary["levels"][str(L)] = {"nh": _r(z[-1:])[0], "sh": _r(zs[-1:])[0], "nh_flag": int(fl[-1]),
                                     "nh_30d": _r([np.nanmean(z[-30:])])[0] if np.isfinite(z[-30:]).any() else None}

        fig = plt.figure(figsize=(12.6, 9.4))
        H = fig.get_figheight()
        title = f"QBO zero-wind line at {L} hPa: MERRA-2 and GEOS FP analyses" + \
            (", " + " and ".join(MODEL[m] for m in fc if str(L) in fc[m]["levels"]) + " forecasts"
             if any(str(L) in fc[m]["levels"] for m in fc) else "")
        fig.text(0.045, 1 - 0.25 / H, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
        sub = (f"Zonal-mean wind [u] by latitude and day, westerly red, easterly blue; black = [u] = 0. The orange line is "
               f"the zero-line index: where planetary waves coming equatorward from the northern westerlies first meet "
               f"easterlies at least 1 m/s deep. Grey dashed = its MERRA-2 1980–2025 median for the date, band 10–90 %.")
        fig.text(0.045, 1 - 0.62 / H, "\n".join(textwrap.wrap(sub, 172)), ha="left", va="top", fontsize=9.6,
                 color=MUTED, linespacing=1.35)
        axA = fig.add_axes([0.062, 0.50, 0.80, 0.36])
        axE = fig.add_axes([0.062, 0.105, 0.385, 0.275])
        axW = fig.add_axes([0.477, 0.105, 0.385, 0.275])
        cax = fig.add_axes([0.885, 0.105, 0.014, 0.755])
        fig.text(0.062, 0.405, "Past winters by QBO phase (Singapore 50 hPa, Oct–Nov): MERRA-2 mean [u] by calendar day, "
                 "with this year's analysed zero line", fontsize=9.6, color=MUTED, ha="left")
        vmax = 40 if L <= 30 else 30
        lev_b = np.array([-vmax, -30, -25, -20, -15, -10, -6, -3, -1, 1, 3, 6, 10, 15, 20, 25, 30, vmax])
        lev_b = np.unique(np.clip(lev_b, -vmax, vmax))
        cmap = plt.get_cmap("RdBu_r", len(lev_b) - 1)
        norm = mcolors.BoundaryNorm(lev_b, cmap.N)

        # --- panel A: recent months + forecast
        X = mdates.date2num(times.to_pydatetime())
        cf = axA.contourf(X, LAT, U.T, levels=lev_b, cmap=cmap, norm=norm, extend="both")
        axA.contour(X, LAT, U.T, levels=[0], colors=INK, linewidths=0.8)
        fc_models = [m for m in ("aifs", "ifs") if m in fc and str(L) in fc[m]["levels"]]
        x_end = X[-1]
        if fc_models:
            rec0 = fc[fc_models[0]]["levels"][str(L)]
            fv = pd.DatetimeIndex(rec0["valid"])
            fX = mdates.date2num(fv.to_pydatetime())
            fu = np.array(rec0["mean_u"])
            axA.contourf(fX, LAT, fu.T, levels=lev_b, cmap=cmap, norm=norm, extend="both")
            axA.contour(fX, LAT, fu.T, levels=[0], colors=INK, linewidths=0.8, linestyles="--")
            axA.axvspan(fX[0], fX[-1], color="white", alpha=0.18, lw=0)
            axA.axvline(fX[0], color=INK, lw=1.0)
            axA.text(fX[0], 1.01, f" forecast from {fc[fc_models[0]]['init']} →", transform=axA.get_xaxis_transform(),
                     fontsize=8.6, color=INK, va="bottom")
            x_end = fX[-1]
            for m in fc_models:
                r = fc[m]["levels"][str(L)]
                vv = mdates.date2num(pd.DatetimeIndex(r["valid"]).to_pydatetime())
                p10, p50, p90 = (np.array([np.nan if v is None else v for v in r["nh"][k]]) for k in ("p10", "p50", "p90"))
                axA.fill_between(vv, p10, p90, color=COL[m], alpha=0.28, lw=0)
                axA.plot(vv, p50, color=COL[m], lw=2.0, label=f"{MODEL[m]} zero line, median and 10–90 % of {r['members']} members")
                summary["forecasts"][f"{m}_{L}"] = {"init": fc[m]["init"], "day15_p50": r["nh"]["p50"][-1],
                                                    "day15_p10": r["nh"]["p10"][-1], "day15_p90": r["nh"]["p90"][-1]}
        # climatology of the index over the whole window, by day of year
        allt = pd.date_range(times[0], pd.Timestamp(mdates.num2date(x_end)).tz_localize(None).normalize())
        di = np.clip(allt.dayofyear.values, 1, 365) - 1
        cq = ref["zl_clim"].sel(lev=L, hemi="nh").values                    # (doy, q)
        ok = ref["zl_defined"].sel(lev=L, hemi="nh").values >= 0.5
        cX = mdates.date2num(allt.to_pydatetime())
        c10, c50, c90 = (np.where(ok[di], cq[di, k], np.nan) for k in range(3))
        band = (c10 > -BOUND + 0.5) & (c90 > -BOUND + 0.5)                   # a censored percentile has no band
        axA.fill_between(cX, np.where(band, c10, np.nan), np.where(band, c90, np.nan), color=COL["clim"], alpha=0.16, lw=0)
        c50 = np.where(c50 > -BOUND + 0.5, c50, np.nan)
        axA.plot(cX, c50, color=COL["clim"], lw=1.4, ls="--", label="MERRA-2 1980–2025 median, 10–90 %")
        axA.plot(*broken(X, z, fl), color=COL["ana"], lw=2.4, label="analysed zero line (MERRA-2, then GEOS FP 00Z)")
        cen = fl == 2
        if cen.any():
            axA.plot(X[cen], np.full(cen.sum(), -33.0), "v", ms=4, color=COL["ana"], ls="none",
                     label="no qualifying crossing north of 45°S")
        if (tdays > m2_end).any() and m2_end > times[0]:
            xm = mdates.date2num(m2_end.to_pydatetime())
            axA.axvline(xm, color=MUTED, lw=0.7, ls=":")
            axA.text(xm, 1.01, "MERRA-2 | GEOS FP", transform=axA.get_xaxis_transform(), fontsize=8.4, color=MUTED,
                     ha="center", va="bottom")
        axA.set_xlim(X[0], x_end)
        axA.set_ylim(-35, 50)
        axA.xaxis.set_major_locator(mdates.MonthLocator())
        axA.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axA.yaxis.set_major_formatter(plt.FuncFormatter(lat_fmt))
        axA.set_yticks(np.arange(-30, 46, 15))
        axA.grid(axis="y", color=GRID, lw=0.5)
        axA.legend(loc="upper left", bbox_to_anchor=(0.0, -0.075), fontsize=8.6, frameon=False, ncol=3,
                   handlelength=2.2, columnspacing=1.6)
        style(axA)
        if L in (70, 30):
            axA.text(0.0, -0.15, "No forecast at this level: ECMWF open data has u at 50 and 10 hPa only.",
                     transform=axA.transAxes, ha="left", va="top", fontsize=8.6, color=MUTED)

        # --- panels B/C: easterly- and westerly-QBO composites, Aug-Mar, with this year's analysed line
        cX2 = mdates.date2num(comp_day.to_pydatetime())
        yr0 = pd.Timestamp(comp_day[0]).year
        # this year's track on the nominal calendar (Jul of the current winter onward)
        wy = ana_end.year if ana_end.month >= 7 else ana_end.year - 1
        for ax, ph, nm in ((axE, "E", "Easterly"), (axW, "W", "Westerly")):
            cu = ref["comp_u"].sel(phase=ph, lev=L).values                   # (cday, LAT)
            ax.contourf(cX2, LAT, cu.T, levels=lev_b, cmap=cmap, norm=norm, extend="both")
            ax.contour(cX2, LAT, cu.T, levels=[0], colors=INK, linewidths=0.8)
            n = int(ref["comp_n"].sel(phase=ph).values)
            this = (times >= pd.Timestamp(wy, 7, 1))
            if this.any():
                shift = pd.DatetimeIndex([t - pd.DateOffset(years=wy - yr0) for t in times[this]])
                ax.plot(*broken(mdates.date2num(shift.to_pydatetime()), z[this], fl[this]), color=COL["ana"], lw=2.2,
                        label=f"{wy}/{str(wy + 1)[2:]} zero line so far")
            ax.set_title(f"{nm} QBO: mean of {n} winters", fontsize=10.5, color=INK, loc="left", fontweight="bold")
            ax.set_ylim(-35, 50); ax.set_yticks(np.arange(-30, 46, 15))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lat_fmt))
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[8, 10, 12, 2]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.set_xlim(cX2[0], cX2[-1])
            ax.grid(axis="y", color=GRID, lw=0.5)
            style(ax)
            if this.any():
                ax.legend(loc="lower left", fontsize=8.2, frameon=True, framealpha=0.9, edgecolor="none")
        axW.set_yticklabels([])
        cb = fig.colorbar(cf, cax=cax, ticks=[t for t in lev_b if t in (-30, -20, -10, -3, 3, 10, 20, 30)])
        cb.set_label("zonal-mean u (m/s)", fontsize=9, color=INK, labelpad=8); cb.ax.tick_params(labelsize=8.5)
        verdict = test.get("test", {}).get("verdict_short", "")
        foot = ("MERRA-2 (NASA GMAO, M2I3NPASM daily means, 1980–" + m2_end.strftime("%b %Y") + ") · GEOS FP assimilation, "
                "00Z (NASA GMAO, NCCS) · ECMWF AIFS-ENS control + 25 members and IFS-ENS 50 members, open data "
                "(CC BY 4.0, Google Cloud mirror)" + ("; forecast shading = " + MODEL[fc_models[0]] + " ensemble-mean [u]" if fc_models else "") +
                " · QBO phase: KIT / FU Berlin Singapore soundings. " + verdict)
        fig.text(0.045, 0.012, "\n".join(textwrap.wrap(foot, 190)), ha="left", va="bottom", fontsize=8.0, color=MUTED,
                 linespacing=1.3)
        out = out_dir / f"qbo_zeroline_{L}.webp"
        fig.savefig(out, format="webp", dpi=110, pil_kwargs={"quality": 88})
        plt.close(fig)
        print(f"  wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out} (NH zero line {summary['levels'][str(L)]['nh']})",
              flush=True)
    (data_dir / "qbo_zeroline.json").write_text(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("forecast"); f.add_argument("--model", choices=("aifs", "ifs"), required=True)
    f.add_argument("--npz"); f.add_argument("--out-dir", default=str(REPO / "assets" / "sst" / "data"))
    r = sp.add_parser("render"); r.add_argument("--out-dir", default=str(REPO / "assets" / "sst"))
    r.add_argument("--tail-dir"); r.add_argument("--data-dir")
    a = ap.parse_args()
    if a.cmd == "forecast":
        forecast(a.model, a.npz, Path(a.out_dir))
    else:
        render(Path(a.out_dir), a.data_dir, a.tail_dir)


if __name__ == "__main__":
    main()
