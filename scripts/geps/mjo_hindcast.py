#!/usr/bin/env python3
"""MJO (RMM) skill of GEPS from the GEPS8 hindcast, 2001-2020, days 1-35.

Every hindcast start (four a week, ensemble mean) is projected onto the site's
Wheeler-Hendon reference EOFs with exactly the arithmetic mjo_geps.py uses for
the live forecast — 15S-15N band means of OLR, u850 and u200; the lead-matched
GEPS8 model climatology removed (de-drift); the model-to-observed frame shift
M(L=0.5) - O added; the WH04 120-day mean removed; divided by the reference
stds — and verified against the Bureau of Meteorology's official RMM series on
the valid date. Scores: the bivariate correlation of Lin et al. (2008) and
Rashid et al. (2011), the bivariate RMSE, and the amplitude ratio, by lead,
for all starts, by season of the start, for starts with observed initial
amplitude >= 1, and for the starts within +-45 days of a chosen day of year
(the season now).

One approximation, and it is the only one: the WH04 120-day mean of the
observed channels before each start is not available for 2001-2020 in this
archive, so it is taken from the hindcast's own de-drifted day-1 channels of
the starts in the preceding 120 days (four a week, ~68 samples). Day-1
skill is ~0.95, so this stands in well for the observed 120-day mean and,
being a slowly varying term, it barely touches the day-to-day score.

    python mjo_hindcast.py --doy-now 246
    -> telecon/mjo_hindcast.nc, telecon/mjo_skill.json, figs/geps_mjo_skill.webp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths                                                     # noqa: E402
sys.path.insert(0, str(paths.REPO / "scripts" / "mjo"))
import mjo_geps as mg                                            # noqa: E402

# the full pass (no --replot) reads the laptop's hindcast store; --replot needs only REF/telecon/mjo_hindcast.nc
STORE = paths.LOCAL_STORE / "geps8_hindcast"
CLIM = paths.LOCAL_STORE / "climatology"
TC = paths.TC_REF if (paths.TC_REF / "mjo_hindcast.nc").exists() else paths.LOCAL_STORE / "telecon"
TC_OUT = paths.TC_STATE
FIGS = paths.FIGS
SITE = paths.SITE_ASSETS
TAGS = {"olr": "clim_olr", "u850": "clim_u850", "u200": "clim_u200"}
Y0, Y1 = 2001, 2020
FILTER_DAYS = 120

# ECMWF extended range, for the reference line. Published bivariate-correlation
# skill of the ECMWF reforecasts (Vitart 2017, QJRMS 143:2210, Fig. 2 and the
# ECMWF annual verification memoranda): COR 0.6 reached at about day 27 in the
# 2016 cycle and near day 30 by the 2020s; 0.5 around day 33. Drawn as the
# two crossing points, not as a curve: they are the numbers the papers quote.
ECMWF_REF = {"COR 0.6": (27, 30), "COR 0.5": (31, 34)}


def bom_rmm():
    """BoM RMM1/RMM2 by date (WH04 method 1974-2013, Gottschalck 2010 after)."""
    rows = []
    for ln in (paths.LOCAL_STORE / "telecon" / "bom" / "rmm.74toRealtime.txt").read_text().splitlines():
        p = ln.split()
        if len(p) < 7 or not p[0].isdigit():
            continue
        y, m, d = int(p[0]), int(p[1]), int(p[2])
        r1, r2 = float(p[3]), float(p[4])
        if abs(r1) > 900 or abs(r2) > 900:
            continue
        rows.append((pd.Timestamp(y, m, d), r1, r2))
    df = pd.DataFrame(rows, columns=["t", "r1", "r2"]).set_index("t")
    return df


def band_of(a, lon):
    """(S, L, lon) 15S-15N cos-weighted band mean of a (S, L, Y, X) hindcast array."""
    a = a.rename({"Y": "latitude", "X": "longitude"})
    lat = a.latitude
    w = np.cos(np.deg2rad(lat)).where((lat >= mg.LAT_MIN) & (lat <= mg.LAT_MAX), 0.0)
    b = a.weighted(w).mean(dim="latitude")
    b = xr.concat([b, b.isel(longitude=0).assign_coords(longitude=360.0)], dim="longitude")
    return b.interp(longitude=lon).values


def model_clim_band(tag, lon):
    """Model climatology band mean on (doy, L, lon), day-of-year axis wrapped."""
    c = xr.open_dataset(CLIM / f"geps8_clim_{tag}.nc")
    cv = c[[v for v in c.data_vars if v != "n_starts"][0]].squeeze(drop=True)
    lat = cv.Y
    w = np.cos(np.deg2rad(lat)).where((lat >= mg.LAT_MIN) & (lat <= mg.LAT_MAX), 0.0)
    b = cv.weighted(w).mean(dim="Y")
    b = xr.concat([b, b.isel(X=0).assign_coords(X=360.0)], dim="X").interp(X=lon)
    b = xr.concat([b.isel(doy=-1).assign_coords(doy=b.doy[-1] - 366), b,
                   b.isel(doy=0).assign_coords(doy=b.doy[0] + 366)], dim="doy")
    return b.load()                                              # (doy, L, X)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doy-now", type=int, default=int(pd.Timestamp.utcnow().dayofyear))
    ap.add_argument("--replot", action="store_true",
                    help="re-score and re-draw from telecon/mjo_hindcast.nc (seconds), for the "
                         "'now +-45 d' curve on each cycle; the full pass takes ~20 min")
    a = ap.parse_args()
    t0 = time.time()
    if a.replot and (TC / "mjo_hindcast.nc").exists():
        return replot(a)
    clim = xr.open_dataset(mg.REF / "climatology.nc")
    eofs = xr.open_dataset(mg.REF / "eofs.nc")
    lon = eofs.longitude.values.astype(float)
    obs = bom_rmm()

    # ---- pass 1: de-drifted, re-based channels for every start and lead ----
    chan, starts, Lref = {}, None, None
    for tag, ckey in TAGS.items():
        mcb = model_clim_band(tag, lon)
        # frame shift M(0.5) - O by day of year of the VALID date
        rr = np.stack([mcb.sel(L=0.5, method="nearest").interp(doy=d).values
                       - clim[ckey].sel(dayofyear=min(d, 366)).values for d in range(1, 367)])
        parts = []
        S_all = []
        for f in sorted(STORE.glob(f"geps8_{tag}_*.nc")):
            ds = xr.open_dataset(f)
            v = ds[[k for k in ds.data_vars][0]]
            if "P" in v.dims:
                v = v.isel(P=0)
            v = v.load().astype("float32")
            S = pd.DatetimeIndex(v.S.values)
            L = v.L.values
            b = band_of(v, lon)                                   # (S, L, lon)
            for si, s0 in enumerate(S):
                mc = mcb.interp(doy=s0.dayofyear).sel(L=L, method="nearest").values
                valid = s0 + pd.to_timedelta(np.floor(L).astype(int), unit="D")   # L = 0.5 is the start date itself
                b[si] = b[si] - mc + rr[valid.dayofyear.values - 1]
            parts.append(b); S_all.append(S.values)
            Lref = L
            ds.close()
        chan[tag] = np.concatenate(parts)                          # (S, L, lon), model->obs frame
        starts = pd.DatetimeIndex(np.concatenate(S_all))
        print(f"  {tag}: {chan[tag].shape[0]} starts, {time.time()-t0:.0f} s", flush=True)
    order = np.argsort(starts.values)
    starts = starts[order]
    for tag in chan:
        chan[tag] = chan[tag][order]

    # ---- 120-day mean from the day-1 channels of the preceding starts ------
    day1 = {tag: chan[tag][:, 0, :] for tag in chan}
    tsec = starts.values.astype("datetime64[D]").astype(int)
    n = len(starts)
    for tag in chan:
        m120 = np.full_like(day1[tag], np.nan)
        for i in range(n):
            sel = (tsec >= tsec[i] - FILTER_DAYS) & (tsec < tsec[i])
            if sel.sum() >= 20:
                m120[i] = day1[tag][sel].mean(0)
        chan[tag] = (chan[tag] - m120[:, None, :]) / clim.attrs[f"std_{tag}"]

    # ---- project, verify ---------------------------------------------------
    nL = len(Lref)
    f1 = np.full((n, nL), np.nan); f2 = np.full_like(f1, np.nan)
    for li in range(nL):
        r1, r2 = mg._project(chan["olr"][:, li, :], chan["u850"][:, li, :], chan["u200"][:, li, :], eofs)
        f1[:, li], f2[:, li] = r1, r2
    ok = np.isfinite(f1).all(1)
    valid_day = np.ceil(Lref).astype(int)
    o1 = np.full_like(f1, np.nan); o2 = np.full_like(f1, np.nan)
    for i, s0 in enumerate(starts):
        vd = s0 + pd.to_timedelta(valid_day - 1, unit="D")
        o = obs.reindex(vd)
        o1[i], o2[i] = o.r1.values, o.r2.values
    o_init = obs.reindex(starts)
    amp0 = np.hypot(o_init.r1.values, o_init.r2.values)
    ds = xr.Dataset({"f1": (("S", "L"), f1.astype("float32")), "f2": (("S", "L"), f2.astype("float32")),
                     "o1": (("S", "L"), o1.astype("float32")), "o2": (("S", "L"), o2.astype("float32")),
                     "amp0": (("S",), amp0.astype("float32"))},
                    coords={"S": starts.values, "L": Lref})
    ds.attrs.update(source="GEPS8 hindcast ensemble mean projected as mjo_geps.py; obs = BoM RMM",
                    note="120-day mean from the hindcast day-1 channels of the preceding 120 days")
    ds.to_netcdf(paths.LOCAL_STORE / "telecon" / "mjo_hindcast.nc")   # then make_reference.py

    return score_and_plot(a, f1, f2, o1, o2, amp0, starts, Lref, ok, t0)
    return 0


def replot(a):
    ds = xr.open_dataset(TC / "mjo_hindcast.nc")
    starts = pd.DatetimeIndex(ds.S.values)
    f1, f2, o1, o2 = ds.f1.values.astype(float), ds.f2.values.astype(float), ds.o1.values.astype(float), ds.o2.values.astype(float)
    ok = np.isfinite(f1).all(1)
    return score_and_plot(a, f1, f2, o1, o2, ds.amp0.values.astype(float), starts, ds.L.values, ok, time.time())


def score_and_plot(a, f1, f2, o1, o2, amp0, starts, Lref, ok, t0):
    nL = len(Lref)
    valid_day = np.ceil(Lref).astype(int)
    def scores(mask):
        F1, F2, O1, O2 = f1[mask], f2[mask], o1[mask], o2[mask]
        good = np.isfinite(F1) & np.isfinite(O1)
        cor, rmse, ampr, nn = [], [], [], []
        for li in range(nL):
            g = good[:, li]
            a1, a2, b1, b2 = F1[g, li], F2[g, li], O1[g, li], O2[g, li]
            cor.append(float((a1 * b1 + a2 * b2).sum() / np.sqrt((a1**2 + a2**2).sum() * (b1**2 + b2**2).sum())))
            rmse.append(float(np.sqrt(((a1 - b1)**2 + (a2 - b2)**2).mean())))
            ampr.append(float(np.hypot(a1, a2).mean() / np.hypot(b1, b2).mean()))
            nn.append(int(g.sum()))
        return dict(cor=cor, rmse=rmse, amp_ratio=ampr, n=nn)

    def cross(cor, thr):
        c = np.array(cor); d = np.arange(1, nL + 1)
        below = np.where(c < thr)[0]
        return None if not len(below) else (float(d[below[0]] - 1 + (c[below[0]-1] - thr) / (c[below[0]-1] - c[below[0]]))
                                            if below[0] > 0 else 0.0)

    mon = starts.month.values
    doy = starts.dayofyear.values
    seasons = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    dn = ((doy - a.doy_now + 183) % 366) - 183
    subsets = {"all": ok,
               "strong initial (amp ≥ 1)": ok & (amp0 >= 1.0),
               f"now ±45 d (doy {a.doy_now})": ok & (np.abs(dn) <= 45)}
    subsets.update({s: ok & np.isin(mon, m) for s, m in seasons.items()})
    res = {k: scores(v) for k, v in subsets.items()}
    out = {"generated": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "leads": [int(x) for x in valid_day],
           "doy_now": a.doy_now, "ecmwf_reference": ECMWF_REF, "subsets": {}}
    for k, v in res.items():
        out["subsets"][k] = {**v, "day_cor_0.6": cross(v["cor"], 0.6), "day_cor_0.5": cross(v["cor"], 0.5),
                             "n_starts": int(subsets[k].sum())}
        print(f"  {k:28s} n {int(subsets[k].sum()):5d}  COR d5 {v['cor'][4]:.2f} d10 {v['cor'][9]:.2f} "
              f"d15 {v['cor'][14]:.2f} d20 {v['cor'][19]:.2f} d25 {v['cor'][24]:.2f} d30 {v['cor'][29]:.2f}  "
              f"0.6 at {cross(v['cor'], 0.6)}  0.5 at {cross(v['cor'], 0.5)}", flush=True)
    TC_OUT.mkdir(parents=True, exist_ok=True)
    (TC_OUT / "mjo_skill.json").write_text(json.dumps(out))

    # ---- figure --------------------------------------------------------------
    fig = plt.figure(figsize=(14.5, 5.4))
    days = valid_day
    cols = {"all": "#16335c", "DJF": "#2c5fa8", "MAM": "#3f9b6a", "JJA": "#c8781e", "SON": "#b4453c",
            "strong initial (amp ≥ 1)": "#6a3d9a", f"now ±45 d (doy {a.doy_now})": "#111111"}
    for pi, (key, ylab, title) in enumerate((("cor", "bivariate correlation", "Skill: bivariate correlation vs BoM RMM"),
                                             ("rmse", "bivariate RMSE", "Error"),
                                             ("amp_ratio", "forecast / observed amplitude", "Amplitude"))):
        ax = fig.add_axes([0.05 + pi * 0.325, 0.19, 0.27, 0.67])
        for k, v in res.items():
            lw = 2.6 if k == "all" else (2.2 if k.startswith("now") else 1.4)
            ls = "-" if not k.startswith("strong") else (0, (4, 2))
            ax.plot(days[:35], v[key][:35], color=cols[k], lw=lw, ls=ls, label=k)
        if key == "cor":
            ax.axhline(0.6, color="#8a8680", lw=0.8, ls=":"); ax.axhline(0.5, color="#8a8680", lw=0.8, ls=":")
            for lab, (d0, d1) in ECMWF_REF.items():
                yv = float(lab.split()[-1])
                ax.plot([d0, d1], [yv, yv], color="#8a1d1d", lw=4, alpha=0.6, solid_capstyle="butt")
                ax.text(d1, yv - 0.02, f"ECMWF, published:\n{lab} at day {d0}–{d1}", fontsize=7.0,
                        color="#8a1d1d", va="top", ha="right")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7.6, loc="lower left", framealpha=0.92)
        elif key == "amp_ratio":
            ax.axhline(1.0, color="#8a8680", lw=0.8, ls=":")
            ax.set_ylim(0.4, 1.2)
        else:
            ax.axhline(np.sqrt(2.0), color="#8a8680", lw=0.8, ls=":")
            ax.text(1.5, np.sqrt(2.0) + 0.02, "√2 = climatology forecast", fontsize=7, ha="left", color="#5a5650")
        ax.set_xlim(1, 35); ax.set_xlabel("forecast day", fontsize=9); ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        ax.grid(alpha=0.25, lw=0.5); ax.tick_params(labelsize=8.5)
    c_all = res["all"]
    fig.suptitle(f"GEPS MJO skill from the GEPS8 hindcast, {Y0}–{Y1} · {int(ok.sum())} starts, ensemble mean · "
                 f"COR 0.6 at day {cross(c_all['cor'], 0.6):.0f}, 0.5 at day {cross(c_all['cor'], 0.5):.0f}",
                 fontsize=12.5, fontweight="bold", y=0.985, va="top")
    fig.text(0.5, 0.012, "same projection as the live RMM (three-field WH04, lead-matched de-drift, 120-day mean removed), "
             "verified against BoM RMM on the valid date; the hindcast mean is 4 members — the 21-member\n"
             "operational mean should score a few days better; ECMWF marks are published reforecast values "
             "(Vitart 2017, ECMWF verification memoranda), not a like-for-like computation",
             ha="center", va="bottom", fontsize=7.8, color="#5a5650")
    for d in (FIGS, SITE):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "geps_mjo_skill.webp", dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    (SITE / "mjo_skill.json").write_text(json.dumps(out))
    print(f"  geps_mjo_skill.webp  ({time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
