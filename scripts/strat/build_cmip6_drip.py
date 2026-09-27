#!/usr/bin/env python3
"""Dripping paint from the CMIP6 sample: u(60N) composites from the ground to 1 hPa around thousands of model sudden
warmings and strong-vortex events, next to MERRA-2 drawn the same way (2026-09-27; user: "can the dripping paint plots
be updated with this much larger set of cmip6 data?").

LAPTOP-ONLY, run once. Reads the CMIP6 extracts (scripts/strat/data/cmip6/<model>_<experiment>_<member>.nc from
cmip6_strat_extract.py, ~25,500 model-years) and the MERRA-2 zonal-mean archive (scripts/telecon/data/m2_strat), and
writes one small committed reference that strat_history.py draws in Actions:

    scripts/strat/reference/cmip6_drip.nc

WHY u(60N) AND NOT POLAR-CAP HEIGHT. The site's MERRA-2 drips are standardised 65-90N geopotential height. The CMIP6
archive has no daily zonal-mean height for these models (DynVarMIP EdayZ carries ua, ta and the TEM terms, not zg), so
the CMIP6 composites are of the zonal-mean zonal wind at 60N, which the extracts hold daily on 39 levels. MERRA-2 is
redrawn in the same variable so model and reanalysis compare like for like. The drip plots -u' (sign flipped) so a
weak vortex is red, as in the height drips; the lower strip plots +u' at 700 hPa, a daily NAM proxy with the AO's sign
(negative = weaker westerlies at 60N). 700 hPa, not 1000: CESM2-WACCM's zonal means are missing at 1000-850 hPa
wherever any longitude is below ground.

MODELS: the stratosphere-resolving, QBO-generating set of cmip6_ssw_expanded.py (GOOD). Every experiment on file is
pooled - piControl, historical, ssp126/245/370/585, amip - because each run is anomalised against its own running
climatology (below), so forced change in the mean state drops out; the events are the model's own.

ANOMALIES (per member, per level): minus a running climatology - a 15-day running mean along the calendar, then a
31-year centred running mean of each calendar day across years (control-run drift and forced trends drop out) - and
divided by the model's standard deviation for that calendar day and level (all its members pooled, a 31-day running
mean, floored at a third of the level's winter peak, the same floor as the MERRA-2 drips). Leap days are dropped;
360-day calendars keep their 360 days. MERRA-2: build_strat_history.standardise (4 harmonics + trend, MLS step above
5 hPa, 3-harmonic sd with the same floor), after its gap filling and QC.

EVENTS (the MERRA-2 drips' rules, in u(60N) terms):
  SSW          Charlton & Polvani (2007) on u(60N, 10 hPa), Dec-Mar (cmip6_strat_analysis.cp07_djfm: first easterly
               day, 20 westerly days between events, back to westerlies for 10 days by the end of April). The 28 MERRA-2
               events the height drips use all fall in Dec-Mar, so the window matches.
  deep/shallow the MERRA-2 drips split at the median of the standardised polar-cap HEIGHT at 100 and 150 hPa, days
               +1..+30; here the same window and levels in standardised -u' (weaker lower-stratospheric vortex =
               deeper), split at each model's own median (MERRA-2: at the median of its 28).
  strong vortex  standardised u'(60N, 10 hPa) first crossing +1.5 upward, Nov-Mar, events at least 60 days apart (the
               MERRA-2 rule on the height-based 10 hPa annular index, in wind terms).
  splits / displacements need the vortex's shape, which zonal means do not have: MERRA-2 height drips only.

TEST. CMIP6: one composite per model (all its events pooled), models weighted equally; a cell is robust where the
across-model one-sample t-test passes Benjamini-Hochberg FDR 10 % over the chart AND >= 80 % of the models agree on
the sign (the house CMIP6 rule). MERRA-2: t-test across events, FDR 10 % over the chart (the height drips' test).

    python scripts/strat/build_cmip6_drip.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cmip6_strat_analysis as A                                   # noqa: E402

DATA = HERE / "data" / "cmip6"
OUT = HERE / "reference" / "cmip6_drip.nc"
GOOD = ["UKESM1-0-LL", "MRI-ESM2-0", "CESM2-WACCM", "CESM2-WACCM-FV2", "CNRM-CM6-1", "CNRM-ESM2-1", "HadGEM3-GC31-LL",
        "MPI-ESM1-2-HR", "MIROC6"]                                  # = cmip6_ssw_expanded.GOOD
LAGS = np.arange(-40, 91)                                          # = build_strat_history.LAGS
SETS = ["ssw_all", "ssw_deep", "ssw_shallow", "sv"]
STRIP_P = 700.0
PMIN = 1.0
SV_THR, SV_SEP = 1.5, 60
CUM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def files_for(model):
    out = []
    for f in sorted(glob.glob(str(DATA / f"{model}_*.nc"))):
        parts = Path(f).stem.split("_")
        if "_".join(parts[:-2]) == model:
            out.append((f, parts[-2], parts[-1]))
    return out


def read_member(path):
    d = xr.open_dataset(path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    Y, M, D = A.ymd(d.time.values)
    cal = getattr(d.time.values[0], "calendar", "standard")
    p = d.plev.values / 100.0
    keep = p >= PMIN - 1e-6
    u = d.u60.values[:, keep].astype("float64")
    if cal == "360_day":
        nd, dix = 360, (M - 1) * 30 + (D - 1)
    else:
        ok = ~((M == 2) & (D == 29))                                # leap days out: a fixed 365-day calendar
        Y, M, D, u = Y[ok], M[ok], D[ok], u[ok]
        nd, dix = 365, CUM[M - 1] + D - 1
    return Y, M, dix, nd, u, p[keep]


def running_anom(u, Y, dix, nd):
    """u minus a 15-day x 31-year running climatology of each calendar day."""
    ys = Y - Y.min(); ny = ys.max() + 1; L = u.shape[1]
    X = np.full((ny, nd, L), np.nan)
    X[ys, dix] = u
    sm = pd.DataFrame(X.reshape(ny * nd, L)).rolling(15, center=True, min_periods=8).mean().values.reshape(ny, nd * L)
    C = pd.DataFrame(sm).rolling(31, center=True, min_periods=10).mean().values.reshape(ny, nd, L)
    return u - C[ys, dix]


def windows(q, idx):
    """(events, lags, ...) windows of q around each index; NaN outside the record."""
    n = len(q)
    out = np.full((len(idx), len(LAGS)) + q.shape[1:], np.nan, "float32")
    for e, i in enumerate(idx):
        j = i + LAGS
        ok = (j >= 0) & (j < n)
        out[e, ok] = q[j[ok]]
    return out


def strong_vortex(z10, M):
    ev, last = [], -10 ** 9
    up = np.where((z10[1:] >= SV_THR) & (z10[:-1] < SV_THR))[0] + 1
    for i in up:
        if M[i] not in (11, 12, 1, 2, 3) or i - last < SV_SEP:
            continue
        ev.append(int(i)); last = i
    return ev


def model_composites(model):
    t0 = time.time()
    mem = []
    ss, cnt, p = None, None, None
    for f, exp, member in files_for(model):
        try:
            Y, M, dix, nd, u, p = read_member(f)
        except Exception as e:                                      # noqa: BLE001
            print(f"  {model} {exp} {member}: skipped ({str(e)[:80]})", flush=True); continue
        if len(np.unique(Y)) < 20:
            continue
        a = running_anom(u, Y, dix, nd)
        if ss is None:
            ss, cnt = np.zeros((nd, u.shape[1])), np.zeros((nd, u.shape[1]))
        ok = np.isfinite(a)
        np.add.at(ss, dix, np.where(ok, a * a, 0.0)); np.add.at(cnt, dix, ok)
        mem.append((exp, member, Y, M, dix, u, a))
    if not mem:
        return model, None
    nd = ss.shape[0]
    var = ss / np.maximum(cnt, 1)
    pad = np.concatenate([var[-15:], var, var[:15]])
    sd = np.sqrt(pd.DataFrame(pad).rolling(31, center=True, min_periods=16).mean().values[15:-15])
    sd = np.maximum(sd, np.nanmax(sd, 0, keepdims=True) / 3)
    k10, k100, k150, kst = (int(np.argmin(np.abs(p - x))) for x in (10.0, 100.0, 150.0, STRIP_P))
    ssw_w, ssw_s, ssw_depth, sv_w, sv_s = [], [], [], [], []
    years, runs = 0, []
    for exp, member, Y, M, dix, u, a in mem:
        z = a / sd[dix]
        qw = -z                                                     # weak vortex positive (the height drips' sign)
        ev = A.cp07_djfm(u[:, k10], Y, M)
        ev = [i for i in ev if i + 30 < len(u)]
        sv = strong_vortex(z[:, k10], M)
        if ev:
            w = windows(qw, ev); ssw_w.append(w); ssw_s.append(windows(z[:, kst], ev))
            ssw_depth += [float(np.nanmean(qw[i + 1:i + 31][:, [k100, k150]])) for i in ev]
        if sv:
            sv_w.append(windows(qw, sv)); sv_s.append(windows(z[:, kst], sv))
        years += len(np.unique(Y)); runs.append(f"{exp}/{member}")
    W = np.concatenate(ssw_w) if ssw_w else np.zeros((0, len(LAGS), len(p)), "float32")
    S = np.concatenate(ssw_s) if ssw_s else np.zeros((0, len(LAGS)), "float32")
    Wv = np.concatenate(sv_w) if sv_w else np.zeros((0, len(LAGS), len(p)), "float32")
    Sv = np.concatenate(sv_s) if sv_s else np.zeros((0, len(LAGS)), "float32")
    depth = np.array(ssw_depth)
    med = float(np.median(depth)) if len(depth) else np.nan
    deep = depth >= med
    sets = {"ssw_all": (W, S), "ssw_deep": (W[deep], S[deep]), "ssw_shallow": (W[~deep], S[~deep]), "sv": (Wv, Sv)}
    comp = {s: np.nanmean(v[0], 0) if len(v[0]) else np.full((len(LAGS), len(p)), np.nan) for s, v in sets.items()}
    strip = {s: np.nanmean(v[1], 0) if len(v[1]) else np.full(len(LAGS), np.nan) for s, v in sets.items()}
    n = {s: int(len(v[0])) for s, v in sets.items()}
    exps = sorted({r.split("/")[0] for r in runs})
    print(f"  {model:16s} {len(runs):3d} runs {years:6d} yrs  SSW {n['ssw_all']:5d} ({n['ssw_all'] / years * 10:.1f}/decade)  "
          f"deep {n['ssw_deep']} shallow {n['ssw_shallow']}  SV {n['sv']}  [{', '.join(exps)}]  {time.time() - t0:.0f} s", flush=True)
    return model, {"comp": comp, "strip": strip, "n": n, "years": years, "runs": len(runs), "exps": exps, "p": p,
                   "depth_median": med}


def merra2():
    """MERRA-2 u(60N) drips on the same rules."""
    import build_strat_history as B
    d = B.load_m2()
    t = pd.DatetimeIndex(d.time.values)
    lev = d.lev.values
    keep = lev >= PMIN - 1e-6
    u = d.u60.values[:, keep].astype(float)
    p = lev[keep]
    z = np.full_like(u, np.nan)
    for k, L in enumerate(p):
        z[:, k] = B.standardise(u[:, k], t, step=L <= 5.0)[1]
    qw = -z
    E = json.load(open(B.OUT_JS))
    used = [pd.Timestamp(e["date"]) for e in E["ssw"] if not e.get("marginal")]
    pos = {x: i for i, x in enumerate(t)}
    idx = [pos[c] for c in used]
    k10, k100, k150, kst = (int(np.argmin(np.abs(p - x))) for x in (10.0, 100.0, 150.0, STRIP_P))
    depth = np.array([float(np.nanmean(qw[i + 1:i + 31][:, [k100, k150]])) for i in idx])
    med = float(np.median(depth)); deep = depth >= med
    zdepth = {e["date"]: e["depth"] for e in E["ssw"] if not e.get("marginal")}
    agree = sum((zdepth[f"{c:%Y-%m-%d}"] == "deep") == bool(dp) for c, dp in zip(used, deep))
    M = t.month.values
    sv = strong_vortex(z[:, list(p).index(10.0)], M)
    sv_h = {e["date"] for e in E["strong_vortex"]}
    sv_dates = [f"{t[i]:%Y-%m-%d}" for i in sv]
    near = sum(any(abs((pd.Timestamp(a) - pd.Timestamp(b)).days) <= 10 for b in sv_h) for a in sv_dates)
    print(f"  MERRA-2: {len(idx)} SSWs, u-based deep/shallow agrees with the height-based split on {agree}/{len(idx)}; "
          f"{len(sv)} strong-vortex events in u terms, {near} within 10 days of a height-based one ({len(sv_h)})")
    sets = {"ssw_all": idx, "ssw_deep": [i for i, dp in zip(idx, deep) if dp], "ssw_shallow": [i for i, dp in zip(idx, deep) if not dp],
            "sv": sv}
    out = {}
    for s, ii in sets.items():
        w = windows(qw, ii)
        n = np.isfinite(w).sum(0)
        m = np.nanmean(w, 0); sd = np.nanstd(w, 0, ddof=1)
        out[s] = {"mean": m, "t": m / (sd / np.sqrt(np.maximum(n, 1))), "n": len(ii), "strip": windows(z[:, kst], ii),
                  "dates": [f"{t[i]:%Y-%m-%d}" for i in ii]}
    return p, out, {"deep_agree_height": int(agree), "n_ssw": len(idx), "sv_near_height": int(near), "n_sv_height": len(sv_h),
                    "depth_median": med}


def main():
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(5) as ex:
        for model, r in ex.map(model_composites, GOOD):
            if r is not None:
                res[model] = r
    models = [m for m in GOOD if m in res]
    p = res[models[0]]["p"]
    for m in models:
        assert np.allclose(res[m]["p"], p), m
    comp = np.stack([[res[m]["comp"][s] for m in models] for s in SETS]).astype("float32")      # set, model, lag, lev
    strip = np.stack([[res[m]["strip"][s] for m in models] for s in SETS]).astype("float32")    # set, model, lag
    nev = np.array([[res[m]["n"][s] for m in models] for s in SETS], "int32")
    mp, m2, m2info = merra2()
    ds = xr.Dataset(
        {"cmip6_comp": (("set", "model", "lag", "plev"), comp),
         "cmip6_strip": (("set", "model", "lag"), strip),
         "cmip6_n": (("set", "model"), nev),
         "cmip6_years": ("model", np.array([res[m]["years"] for m in models], "int32")),
         "cmip6_runs": ("model", np.array([res[m]["runs"] for m in models], "int32")),
         "m2_mean": (("set", "lag", "m2lev"), np.stack([m2[s]["mean"] for s in SETS]).astype("float32")),
         "m2_t": (("set", "lag", "m2lev"), np.stack([m2[s]["t"] for s in SETS]).astype("float32")),
         "m2_n": ("set", np.array([m2[s]["n"] for s in SETS], "int32")),
         "m2_strip_ssw": (("m2ssw", "lag"), m2["ssw_all"]["strip"].astype("float32")),
         "m2_strip_deep": (("m2deep", "lag"), m2["ssw_deep"]["strip"].astype("float32")),
         "m2_strip_shallow": (("m2shallow", "lag"), m2["ssw_shallow"]["strip"].astype("float32")),
         "m2_strip_sv": (("m2sv", "lag"), m2["sv"]["strip"].astype("float32"))},
        coords={"set": SETS, "model": models, "lag": LAGS, "plev": p, "m2lev": mp,
                "m2ssw": m2["ssw_all"]["dates"], "m2deep": m2["ssw_deep"]["dates"], "m2shallow": m2["ssw_shallow"]["dates"],
                "m2sv": m2["sv"]["dates"]},
        attrs={"title": "Dripping paint in u(60N): CMIP6 per-model composites and MERRA-2, standardised; drip = -u' "
                        "(weak vortex positive), strip = +u' at 700 hPa",
               "built": pd.Timestamp.utcnow().strftime("%Y-%m-%d"), "builder": "scripts/strat/build_cmip6_drip.py",
               "strip_hpa": STRIP_P,
               "experiments": json.dumps({m: res[m]["exps"] for m in models}),
               "merra2": json.dumps(m2info)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT, encoding={v: {"zlib": True, "complevel": 5} for v in ds.data_vars})
    tot = {s: int(nev[i].sum()) for i, s in enumerate(SETS)}
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB): {len(models)} models, "
          f"{int(ds.cmip6_years.sum())} model-years, events {tot}; MERRA-2 {dict(zip(SETS, ds.m2_n.values.tolist()))}; "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
