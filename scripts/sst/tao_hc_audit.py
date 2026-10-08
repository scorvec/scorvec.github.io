#!/usr/bin/env python3
"""Audit of the equatorial 0-300 m heat-content index (TAO 8 moorings) - Actions only.

  * fetches the whole TAO DISDEL equatorial record (1980 -> today) with quality/source codes;
  * rebuilds the index the old way (exact replica of sst_subsurface_events.hc before 2026-10-07)
    and the screened way (tao_hc);
  * daily coverage table since 1 Aug of the current year, carry-forward ages, isolated reports;
  * GLORYS12 (Copernicus Marine: reanalysis my + interim myint + analysis anfc) 0-300 m anomaly at
    the same 8 points, same depth integration; mooring-dropout experiments on fully covered days;
  * CPC GODAS heat-content index (monthly, 180W-100W etc.) and PMEL GODAS T300 east/west.
Outputs go to $AUDIT_OUT (default scripts/sst/data/hc_audit): summary.json + csv files.
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import tao_subsurface as tao
import sst_subsurface as ss
import tao_hc as H

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("AUDIT_OUT", HERE / "data" / "hc_audit"))
CACHE = HERE / "data" / "hc_audit_cache"
UA = {"User-Agent": "Mozilla/5.0 (scorvec.com data pipeline)"}
NOW = datetime.now(timezone.utc)

CPC_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ocean/index/heat_content_index.txt"
PMEL_URL = "https://www.pmel.noaa.gov/tao/wwv/data/{}.dat"

GLORYS = [("my", "cmems_mod_glo_phy_my_0.083deg_P1D-m", "thetao"),
          ("anfc", "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", "thetao")]


def log(*a):
    print(*a, flush=True)


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read().decode("latin-1")


# ── TAO ───────────────────────────────────────────────────────────────────────
def fetch_tao():
    CACHE.mkdir(parents=True, exist_ok=True)
    chunks = [(y, min(y + 4, NOW.year)) for y in range(1980, NOW.year + 1, 5)]
    paths = []
    for y0, y1 in chunks:
        p = CACHE / f"tao_{y0}_{y1}.ascii"
        end = datetime(y1, 12, 31) if y1 < NOW.year else NOW
        if not (p.exists() and p.stat().st_size > 0) or y1 == NOW.year:
            tao.deliver(datetime(y0, 1, 1), end, p)
        paths.append(p)
    return paths


def old_index(paths):
    """Exact replica of the pre-2026-10-07 hc(): raw parse, depth mean skipna, ffill(limit=10),
    NaN below 6 used, centred 15-d mean with trailing fill."""
    rows = []
    for p in paths:
        d = tao.parse(p)
        a = ss.to_depth_grid(d)
        coeffs = ss.load_or_build_coeffs(a.longitude.values)
        doy = pd.to_datetime(a.time.values).dayofyear.values
        an = a - ss.eval_climatology(coeffs, doy)
        col = an.sel(depth=slice(0, 300)).mean("depth", skipna=True)
        rows.append(pd.DataFrame(col.values, index=pd.to_datetime(a.time.values),
                                 columns=[float(x) for x in col.longitude.values]))
    df = pd.concat(rows).sort_index()
    df = df[~df.index.duplicated(keep="last")].reindex(columns=H.LONS)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    live = df.notna()
    filled = df.ffill(limit=H.PERSIST_DAYS)
    used = filled.notna()
    s = filled.mean(axis=1, skipna=True)
    s[used.sum(axis=1) < H.MIN_MOORINGS] = np.nan
    centred = s.rolling(15, center=True, min_periods=15).mean()
    trailing = s.rolling(15, min_periods=8).mean()
    return df, s, centred.fillna(trailing), live, used


# ── references ────────────────────────────────────────────────────────────────
def cpc_monthly():
    txt = _get(CPC_URL)
    (OUT / "cpc_heat_content_index.txt").write_text(txt)
    rows = []
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) == 5 and p[0].isdigit():
            rows.append((pd.Timestamp(int(p[0]), int(p[1]), 1), float(p[2]), float(p[3]), float(p[4])))
    df = pd.DataFrame(rows, columns=["month", "cpc_130E_80W", "cpc_160E_80W", "cpc_180_100W"]).set_index("month")
    return df, txt.splitlines()[0].strip()


def pmel_monthly():
    out = {}
    for name in ["t300", "t300_east", "t300_west"]:
        try:
            txt = _get(PMEL_URL.format(name))
        except Exception as e:      # noqa: BLE001
            log(f"PMEL {name} failed: {e}")
            continue
        rows = []
        for ln in txt.splitlines():
            p = ln.split()
            if len(p) == 3 and p[0].isdigit() and len(p[0]) == 6:
                rows.append((pd.Timestamp(int(p[0][:4]), int(p[0][4:]), 1), float(p[2])))
        out["pmel_" + name] = pd.Series(dict(rows))
    return pd.DataFrame(out)


def _fetch_one(tag, dsid, var, lo):
    import copernicusmarine as cm
    import time as _t
    glon = lo if lo < 180 else lo - 360          # GLORYS -180..180
    fn = f"gl_{tag}_{int(lo)}.nc"
    t0 = _t.time()
    kw = dict(dataset_id=dsid, variables=[var], minimum_longitude=glon, maximum_longitude=glon,
              minimum_latitude=-0.25, maximum_latitude=0.25, minimum_depth=0, maximum_depth=330,
              output_filename=fn + ".part", output_directory=str(CACHE), overwrite=True,
              coordinates_selection_method="nearest", disable_progress_bar=True)
    if tag == "anfc":
        kw.update(start_datetime="2022-06-01", end_datetime=NOW.strftime("%Y-%m-%d"))
    try:
        cm.subset(**kw)
        (CACHE / (fn + ".part")).replace(CACHE / fn)
        log(f"  fetched {fn} in {_t.time() - t0:.0f} s")
    except Exception as e:  # noqa: BLE001
        log(f"GLORYS {tag} {lo} failed: {e}")


def glorys_series(fetch: bool = False):
    """Daily 0-300 m T profile (5-m grid) at each mooring, 0.25S-0.25N, from my (reanalysis, which
    now runs to ~6 weeks ago) and anfc (analysis)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if fetch:
        from concurrent.futures import ThreadPoolExecutor
        jobs = [(tag, dsid, var, lo) for tag, dsid, var in GLORYS for lo in H.LONS
                if not (CACHE / f"gl_{tag}_{int(lo)}.nc").exists()]
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(lambda j: _fetch_one(*j), jobs))
        return None
    res = {}
    for tag, dsid, var in GLORYS:
        cols = {}
        for lo in H.LONS:
            p = CACHE / f"gl_{tag}_{int(lo)}.nc"
            if not p.exists():
                continue
            d = xr.open_dataset(p)[var]
            d = d.mean([c for c in ("latitude", "longitude") if c in d.dims])
            dep = d.depth.values.astype(float)
            V = d.transpose("time", "depth").values
            G = np.full((V.shape[0], len(ss.DEPTH_GRID)), np.nan)
            for i in range(V.shape[0]):
                m = np.isfinite(V[i])
                if m.sum() >= 3 and dep[m].max() >= 300:
                    G[i] = np.interp(ss.DEPTH_GRID, dep[m], V[i][m])
            t = pd.to_datetime(d.time.values).normalize()
            cols[lo] = pd.DataFrame(G, index=t)
            log(f"GLORYS {tag} {H.LON_LABEL[lo]}: {t.min():%Y-%m-%d} -> {t.max():%Y-%m-%d} ({len(t)} d)")
        res[tag] = cols
    return res


def harmonic_fit(t: pd.DatetimeIndex, Y: np.ndarray):
    w = 2 * np.pi * t.dayofyear.values / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    m = np.isfinite(Y).all(1)
    c, *_ = np.linalg.lstsq(X[m], Y[m], rcond=None)
    return c                                                     # (5, ndepth)


def harmonic_eval(t, c):
    w = 2 * np.pi * t.dayofyear.values / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    return X @ c


def glorys_anomaly(gl):
    """Per-mooring 0-300 m anomaly vs a GLORYS 1993-2020 harmonic climatology (per 5-m level).
    my+myint form one record; anfc gets its own constant offset vs myint on the overlap."""
    coeffs = {}
    anom = {}
    for lo in H.LONS:
        parts = [gl[k][lo] for k in ("my",) if lo in gl.get(k, {})]
        if not parts:
            continue
        rea = pd.concat(parts).sort_index()
        rea = rea[~rea.index.duplicated(keep="first")]
        base = rea[(rea.index >= "1993-01-01") & (rea.index <= "2020-12-31")]
        c = harmonic_fit(base.index, base.values)
        coeffs[lo] = c
        a_rea = pd.Series((rea.values - harmonic_eval(rea.index, c)).mean(1), index=rea.index)
        s = a_rea
        if lo in gl.get("anfc", {}):
            af = gl["anfc"][lo]
            af = af[af.index <= pd.Timestamp(NOW.date()) - pd.Timedelta(days=1)]   # analysis days only
            a_af = pd.Series((af.values - harmonic_eval(af.index, c)).mean(1), index=af.index)
            ov = a_af.index.intersection(a_rea.index)
            off = float((a_rea[ov] - a_af[ov]).mean()) if len(ov) > 60 else 0.0
            r = float(np.corrcoef(a_rea[ov], a_af[ov])[0, 1]) if len(ov) > 60 else np.nan
            log(f"  anfc vs my {H.LON_LABEL[lo]}: {len(ov)} d overlap, offset {off:+.3f}, r {r:.3f}")
            s = pd.concat([a_rea, (a_af + off)[a_af.index > a_rea.index.max()]])
        anom[lo] = s
    return pd.DataFrame(anom).reindex(columns=H.LONS), coeffs


def stats(a: pd.Series, b: pd.Series):
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return {"n": int(m.sum())}
    x, y = a[m].values, b[m].values
    sl, ic = np.polyfit(y, x, 1)
    return {"n": int(m.sum()), "r": round(float(np.corrcoef(x, y)[0, 1]), 3),
            "bias": round(float((x - y).mean()), 3), "rmse": round(float(np.sqrt(((x - y) ** 2).mean())), 3),
            "slope": round(float(sl), 3), "intercept": round(float(ic), 3),
            "rmse_after_fit": round(float(np.sqrt(((x - (sl * y + ic)) ** 2).mean())), 3)}


def lag_corr(a, b, lags=range(-60, 61, 5)):
    best = None
    for L in lags:
        s = stats(a, b.shift(L))
        if "r" in s and (best is None or s["r"] > best[1]):
            best = (L, s["r"])
    return best


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"generated": NOW.isoformat(timespec="seconds")}

    # ---- TAO ------------------------------------------------------------------
    paths = fetch_tao()
    ds = H.concat([H.parse_qc(p) for p in paths])
    log(f"TAO record {pd.Timestamp(ds.time.values[0]):%Y-%m-%d} -> {pd.Timestamp(ds.time.values[-1]):%Y-%m-%d}, "
        f"lons {ds.longitude.values.tolist()}")
    # quality / source code census inside 0-300 m at the 8 moorings
    sub = ds.sel(longitude=[l for l in H.LONS if l in ds.longitude.values], depth=ds.depth[ds.depth <= 300])
    fin = np.isfinite(sub["temp"].values)
    q = sub["q"].values[fin]; s = sub["s"].values[fin]
    yrs = pd.to_datetime(sub.time.values).year.values[:, None, None] * np.ones(fin.shape, int)
    yrs = yrs[fin]
    census = pd.DataFrame({"year": yrs, "q": q, "s": s})
    qtab = census.pivot_table(index="year", columns="q", aggfunc="size", fill_value=0)
    stab = census.pivot_table(index="year", columns="s", aggfunc="size", fill_value=0)
    qtab.to_csv(OUT / "qc_quality_by_year.csv"); stab.to_csv(OUT / "qc_source_by_year.csv")
    log("quality codes by year (finite values, 0-300 m, 8 moorings):\n" + qtab.to_string())
    log("source codes by year:\n" + stab.to_string())

    # climatology coefficients: NaN cells?
    cf = ss.load_or_build_coeffs(H.LONS)
    nanc = {H.LON_LABEL[l]: int(np.isnan(cf[0, :, j]).sum()) for j, l in enumerate(H.LONS)}
    summary["clim_nan_levels_per_mooring"] = nanc
    log(f"climatology NaN 5-m levels per mooring: {nanc}")

    strict, info = H.mooring_anomaly(ds, strict=True)
    loose, info_l = H.mooring_anomaly(ds, strict=False)
    strict4, info4 = H.mooring_anomaly(ds, strict=True, qset=(1, 2, 3, 4))
    idx_new = H.index(strict)
    idx_q4 = H.index(strict4)
    # why profiles fail, by year (q123 screen and shape-only screen)
    for tag, inf in [("q123", info), ("q1234", info4)]:
        r = inf["reason"].stack().rename("reason").reset_index()
        r.columns = ["time", "lon", "reason"]
        r["year"] = pd.to_datetime(r["time"]).dt.year
        tab = r.pivot_table(index="year", columns="reason", aggfunc="size", fill_value=0)
        tab.to_csv(OUT / f"profile_reasons_{tag}.csv")
        log(f"profile verdicts by year ({tag}; mooring-days):\n" + tab.to_string())
        rm = r.pivot_table(index="lon", columns="reason", aggfunc="size", fill_value=0)
        log(f"profile verdicts by mooring ({tag}):\n" + rm.rename(index=H.LON_LABEL).to_string())
    both = strict.notna() & strict4.notna()
    d4 = (strict4 - strict).where(both)
    summary["q4_inclusion_effect_where_both_ok"] = {H.LON_LABEL[c]: {"n": int(both[c].sum()),
        "mean": round(float(d4[c].mean()), 3) if both[c].any() else None,
        "sd": round(float(d4[c].std()), 3) if both[c].sum() > 2 else None} for c in H.LONS}
    old_df, old_raw, old_sm, old_live, old_used = old_index(paths)

    # partial profiles that the old rule counted as live
    part = (loose.notna() & strict4.isna())
    py = part.groupby(part.index.year).sum()
    py.columns = [H.LON_LABEL[c] for c in py.columns]
    py.to_csv(OUT / "partial_profiles_by_year.csv")
    log("mooring-days the OLD rule used whose profile does not span 0-300 m (shape rule, Q4 kept):\n" + py.to_string())

    # per-year coverage
    cov = pd.DataFrame({
        "days": idx_new.groupby(idx_new.index.year).size(),
        "mean_live_strict": idx_new["n_live"].groupby(idx_new.index.year).mean().round(2),
        "days_ge6_strict": (idx_new["hc"].notna()).groupby(idx_new.index.year).sum(),
        "days_ge6_q1234": (idx_q4["hc"].notna()).groupby(idx_q4.index.year).sum(),
        "mean_live_q1234": idx_q4["n_live"].groupby(idx_q4.index.year).mean().round(2),
        "days_ge6_old": old_raw.notna().groupby(old_raw.index.year).sum(),
        "days_all8_live": (idx_new["n_live"] == 8).groupby(idx_new.index.year).sum(),
    })
    cov.to_csv(OUT / "coverage_by_year.csv")
    log("coverage by year:\n" + cov.to_string())

    # ---- carry-forward audit ----------------------------------------------------
    # old rule: how long could a value be carried, counting from its last REAL complete report?
    _, age_strict = H.carry(strict, persist=10 ** 6)
    old_used_r = old_used.reindex(age_strict.index).fillna(False)
    old_age_vs_complete = age_strict.where(old_used_r)
    worst = old_age_vs_complete.max()
    summary["old_rule_max_age_since_complete_profile_days"] = {H.LON_LABEL[k]: (None if pd.isna(v) else int(v))
                                                                for k, v in worst.items()}
    log(f"old rule: max days since the mooring's last COMPLETE screened profile among values it used: "
        f"{summary['old_rule_max_age_since_complete_profile_days']}")
    # isolated reports: a live run of <=2 days bracketed by >10-day gaps
    iso = {}
    for c in strict.columns:
        ok = strict[c].reindex(age_strict.index).notna().astype(int)
        runs = (ok.diff().fillna(ok.iloc[0]) != 0).cumsum()
        g = ok.groupby(runs)
        cnt = 0
        for k, v in g:
            if v.iloc[0] == 1 and len(v) <= 2:
                cnt += 1
        iso[H.LON_LABEL[c]] = cnt
    summary["isolated_short_reports"] = iso

    # old smoothed line drawn on days with no valid raw index
    ghost = old_sm.notna() & old_raw.reindex(old_sm.index).isna()
    summary["old_line_days_drawn_without_index"] = int(ghost.sum())
    gy = ghost.groupby(ghost.index.year).sum()
    log(f"old smoothed line drawn on days whose raw index was NaN: {int(ghost.sum())} (by year: "
        f"{ {int(k): int(v) for k, v in gy.items() if v} })")

    # ---- recent coverage table --------------------------------------------------
    t0 = pd.Timestamp(NOW.year, 8, 1)
    _, age10 = H.carry(strict)
    rec = []
    for d in pd.date_range(t0, pd.Timestamp(NOW.date())):
        row = {"date": d.strftime("%Y-%m-%d")}
        for c in H.LONS:
            lab = H.LON_LABEL[c]
            has = d in loose.index and pd.notna(loose.at[d, c])
            ok = d in strict.index and pd.notna(strict.at[d, c])
            a = age10.at[d, c] if d in age10.index else np.nan
            ns = int(info["nsens"].at[d, c]) if d in info["nsens"].index and pd.notna(info["nsens"].at[d, c]) else 0
            row[lab] = "L" if ok else ("p%d" % ns if has else (f"c{int(a)}" if pd.notna(a) else "."))
        row["n_live"] = int(idx_new["n_live"].get(d, 0)) if d in idx_new.index else 0
        row["n_carried"] = int(idx_new["n_carried"].get(d, 0)) if d in idx_new.index else 0
        row["old_n_used"] = int(old_used.loc[d].sum()) if d in old_used.index else 0
        row["old_raw"] = None if d not in old_raw.index or pd.isna(old_raw[d]) else round(float(old_raw[d]), 3)
        row["old_plotted"] = None if d not in old_sm.index or pd.isna(old_sm[d]) else round(float(old_sm[d]), 3)
        row["new_raw"] = None if d not in idx_new.index or pd.isna(idx_new.at[d, "hc"]) else round(float(idx_new.at[d, "hc"]), 3)
        rec.append(row)
    rt = pd.DataFrame(rec)
    rt.to_csv(OUT / "coverage_recent.csv", index=False)
    log("recent daily coverage (L live, pN partial with N sensors <=300 m, cK carried K days, . none):\n"
        + rt.to_string(index=False))

    # ---- references -------------------------------------------------------------
    cpc, cpc_hdr = cpc_monthly()
    summary["cpc_header"] = cpc_hdr
    pm = pmel_monthly()
    try:
        if os.environ.get("AUDIT_GLORYS", "1") == "0":
            raise RuntimeError("GLORYS disabled for this pass (AUDIT_GLORYS=0)")
        gl = glorys_series()
        g_anom, g_coeffs = glorys_anomaly(gl)
        np.savez(OUT / "glorys_clim_coeffs.npz", lons=np.array(list(g_coeffs)),
                 coeffs=np.stack([g_coeffs[l] for l in g_coeffs]))
    except Exception as e:      # noqa: BLE001
        log(f"GLORYS failed entirely: {e}")
        g_anom = pd.DataFrame(columns=H.LONS, dtype=float)
    g_anom.to_csv(OUT / "glorys_mooring_anom.csv")
    strict.to_csv(OUT / "tao_mooring_anom_strict.csv")
    strict4.to_csv(OUT / "tao_mooring_anom_q1234.csv")
    loose.to_csv(OUT / "tao_mooring_anom_old.csv")
    idx_g = g_anom.mean(axis=1).where(g_anom.notna().sum(axis=1) == 8)

    # mooring-level agreement TAO vs GLORYS
    ml = {}
    for c in H.LONS:
        if c in g_anom:
            ml[H.LON_LABEL[c]] = stats(strict[c], g_anom[c])
    summary["mooring_tao_vs_glorys_daily"] = ml
    q4only = strict4.notna() & strict.isna()
    qe = {}
    for c in H.LONS:
        g = g_anom[c].reindex(strict.index) if c in g_anom else None
        if g is None:
            continue
        e123 = (strict[c] - g).dropna(); e4 = (strict4[c].where(q4only[c]) - g).dropna()
        qe[H.LON_LABEL[c]] = {"q123_days": int(len(e123)), "q123_rmse": round(float(np.sqrt((e123 ** 2).mean())), 3) if len(e123) else None,
                              "q4only_days": int(len(e4)), "q4only_rmse": round(float(np.sqrt((e4 ** 2).mean())), 3) if len(e4) else None,
                              "q4only_bias_minus_q123_bias": round(float(e4.mean() - e123.mean()), 3) if len(e4) and len(e123) else None}
    summary["q4_vs_glorys"] = qe
    log("Q4-dependent mooring-days vs GLORYS: " + json.dumps(qe))
    old_vs_g = {}
    lo_only = loose.notna() & strict4.isna()
    for c in H.LONS:
        if c not in g_anom:
            continue
        g = g_anom[c].reindex(strict.index)
        e = (loose[c].where(lo_only[c]) - g).dropna()
        old_vs_g[H.LON_LABEL[c]] = {"partial_days": int(len(e)), "rmse": round(float(np.sqrt((e ** 2).mean())), 3) if len(e) else None,
                                    "bias": round(float(e.mean()), 3) if len(e) else None}
    summary["partial_profile_days_vs_glorys"] = old_vs_g
    log("partial-profile mooring-days (old rule only) vs GLORYS: " + json.dumps(old_vs_g))
    summary["index_q4_vs_glorys_daily_ge6"] = stats(idx_q4["hc"], g_anom.mean(axis=1).where(g_anom.notna().sum(axis=1) == 8))
    summary["index_old_vs_glorys_daily"] = stats(old_raw, g_anom.mean(axis=1).where(g_anom.notna().sum(axis=1) == 8))
    log("per-mooring TAO (screened) vs GLORYS 0-300 m anomaly, daily: " + json.dumps(ml))

    # index agreement, all-8-live days
    full = idx_new["n_live"] == 8
    summary["index_tao_vs_glorys_daily_all8"] = stats(idx_new["hc"].where(full), idx_g)
    summary["index_tao_vs_glorys_daily_ge6"] = stats(idx_new["hc"], idx_g)
    byy = {}
    for y in range(1993, NOW.year + 1):
        m = idx_new.index.year == y
        byy[y] = stats(idx_new["hc"][m], idx_g.reindex(idx_new.index)[m])
    summary["index_tao_vs_glorys_by_year"] = byy

    # ---- dropout experiments on fully covered days ------------------------------
    # truth = mean of the 8 live moorings; remove a set, estimate by (a) mean of the rest,
    # (b) longitude interpolation from live neighbours, (c) GLORYS at the dropped moorings plus
    # the mooring's trailing-365-d TAO-GLORYS offset (computed only from days BEFORE the test day)
    diff = (strict - g_anom.reindex(strict.index))
    off = diff.rolling(365, min_periods=120).mean().shift(1)
    S8 = strict[full.reindex(strict.index).fillna(False)]
    truth = S8.mean(axis=1)
    lonarr = np.array(H.LONS)
    sets = {"west3 (165E,180,170W)": [165.0, 180.0, 190.0],
            "sep14 set (165E,180,155W,95W)": [165.0, 180.0, 205.0, 265.0],
            "2026-09 (165E,180,170W,155W,95W)": [165.0, 180.0, 190.0, 205.0, 265.0]}
    for c in H.LONS:
        sets[f"single {H.LON_LABEL[c]}"] = [c]
    drop = {}
    for name, ds_ in sets.items():
        keep = [c for c in H.LONS if c not in ds_]
        est_mean = S8[keep].mean(axis=1)
        est_int = S8.apply(lambda r: np.interp(lonarr, np.array(keep), r[keep].values).mean(), axis=1)
        gfill = S8.copy()
        for c in ds_:
            gfill[c] = g_anom.reindex(S8.index)[c] + off.reindex(S8.index)[c] if c in g_anom else np.nan
        est_g = gfill.mean(axis=1).where(gfill.notna().all(axis=1))
        res = {}
        for k, e in [("mean_of_rest", est_mean), ("lon_interp", est_int), ("glorys_fill", est_g)]:
            err = (e - truth).dropna()
            res[k] = {"n": int(len(err)), "bias": round(float(err.mean()), 3) if len(err) else None,
                      "rmse": round(float(np.sqrt((err ** 2).mean())), 3) if len(err) else None,
                      "p95_abs": round(float(err.abs().quantile(0.95)), 3) if len(err) else None}
            if len(err) and k != "glorys_fill":
                e2 = err[err.index >= "2023-01-01"]
                res[k]["rmse_2023on"] = round(float(np.sqrt((e2 ** 2).mean())), 3) if len(e2) else None
            if len(err) and k == "glorys_fill":
                e2 = err[err.index >= "2023-01-01"]
                res[k]["rmse_2023on"] = round(float(np.sqrt((e2 ** 2).mean())), 3) if len(e2) else None
                res[k]["n_2023on"] = int(len(e2))
        drop[name] = res
    summary["dropout_tests"] = drop
    log("dropout tests:\n" + json.dumps(drop, indent=1))

    # ---- monthly comparison with CPC / PMEL --------------------------------------
    def monthly(s, minday=20):
        g = s.groupby(s.index.to_period("M"))
        m = g.mean().where(g.count() >= minday)
        m.index = m.index.to_timestamp()
        return m
    sub6 = [180.0, 190.0, 205.0, 220.0, 235.0, 250.0]
    hc6 = strict[sub6].mean(axis=1).where(strict[sub6].notna().all(axis=1))
    east4 = [205.0, 220.0, 235.0, 250.0, 265.0]
    hce = strict[east4].mean(axis=1).where(strict[east4].notna().all(axis=1))
    mon = pd.DataFrame({
        "tao8_new": monthly(idx_new["hc"]),
        "tao8_q4": monthly(idx_q4["hc"]),
        "tao8_old": monthly(old_raw),
        "tao_180_110W_live": monthly(hc6),
        "tao_155W_95W_live": monthly(hce),
        "glorys8": monthly(idx_g) if len(idx_g.dropna()) else np.nan,
        "tao_days_ge6_new": idx_new["hc"].groupby(idx_new.index.to_period("M")).count().rename(lambda p: p.to_timestamp()),
        "tao_days_ge6_old": old_raw.groupby(old_raw.index.to_period("M")).count().rename(lambda p: p.to_timestamp()),
        "mean_live_new": idx_new["n_live"].groupby(idx_new.index.to_period("M")).mean().rename(lambda p: p.to_timestamp()),
    }).join(cpc, how="outer").join(pm, how="outer")
    mon.to_csv(OUT / "monthly_compare.csv")
    comp = {}
    for a, b in [("tao8_q4", "cpc_180_100W"), ("tao8_q4", "glorys8"), ("tao8_new", "cpc_180_100W"), ("tao8_new", "cpc_160E_80W"), ("tao8_old", "cpc_180_100W"),
                 ("tao_180_110W_live", "cpc_180_100W"), ("glorys8", "cpc_180_100W"), ("tao8_new", "glorys8"),
                 ("tao_155W_95W_live", "pmel_t300_east"), ("tao8_new", "pmel_t300")]:
        if a in mon and b in mon:
            comp[f"{a} vs {b}"] = {"1981-2026": stats(mon[a], mon[b]),
                                   "1993-2026": stats(mon[a][mon.index >= "1993"], mon[b][mon.index >= "1993"])}
    summary["monthly_comparisons"] = comp
    log("monthly comparisons:\n" + json.dumps(comp, indent=1))

    # yearly residuals vs CPC after one regression over 1981-2026; flag > 0.4 C monthly
    fitp = stats(mon["tao8_new"], mon["cpc_180_100W"])
    if "slope" in fitp:
        pred = fitp["slope"] * mon["cpc_180_100W"] + fitp["intercept"]
        resid = (mon["tao8_new"] - pred)
        rold = mon["tao8_old"] - (fitp["slope"] * mon["cpc_180_100W"] + fitp["intercept"])
        yr = pd.DataFrame({
            "months": resid.groupby(resid.index.year).count(),
            "r_new": [stats(mon["tao8_new"][mon.index.year == y], mon["cpc_180_100W"][mon.index.year == y]).get("r")
                      for y in sorted(set(resid.index.year))],
            "mean_resid_new": resid.groupby(resid.index.year).mean().round(2),
            "max_abs_resid_new": resid.abs().groupby(resid.index.year).max().round(2),
            "mean_resid_q4": (mon["tao8_q4"] - pred).groupby(mon.index.year).mean().round(2),
            "max_abs_resid_q4": (mon["tao8_q4"] - pred).abs().groupby(mon.index.year).max().round(2),
            "months_q4": (mon["tao8_q4"] - pred).groupby(mon.index.year).count(),
            "mean_resid_old": rold.groupby(rold.index.year).mean().round(2),
            "max_abs_resid_old": rold.abs().groupby(rold.index.year).max().round(2),
            "months_old": rold.groupby(rold.index.year).count(),
        })
        yr.to_csv(OUT / "yearly_vs_cpc.csv")
        log("yearly vs CPC 180W-100W (residual after one regression, deg C):\n" + yr.to_string())
        flag = resid[resid.abs() > 0.4]
        summary["months_resid_gt_0p4"] = {k.strftime("%Y-%m"): round(float(v), 2) for k, v in flag.items()}
        summary["resid_sd"] = round(float(resid.std()), 3)

    # timing
    summary["lag_best_tao8_vs_cpc_180_100W_months"] = None
    lc = []
    for L in range(-3, 4):
        st = stats(mon["tao8_new"], mon["cpc_180_100W"].shift(L))
        lc.append((L, st.get("r")))
    summary["lag_corr_tao8_vs_cpc_months"] = lc

    # 1998 trough
    y98 = idx_new["hc"]["1998"]
    sm98 = H.smooth(idx_new["hc"])["1998"]
    summary["trough_1998"] = {
        "new_raw_min": round(float(y98.min()), 2), "new_raw_min_date": str(y98.idxmin().date()),
        "new_smoothed_min": round(float(sm98.min()), 2),
        "old_smoothed_min": round(float(old_sm["1998"].min()), 2),
        "old_smoothed_min_date": str(old_sm["1998"].idxmin().date()),
        "tao8_monthly_min": round(float(mon["tao8_new"]["1998"].min()), 2),
        "cpc_180_100W_min": float(cpc["cpc_180_100W"]["1998"].min()),
        "cpc_160E_80W_min": float(cpc["cpc_160E_80W"]["1998"].min()),
        "cpc_180_100W_min_month": cpc["cpc_180_100W"]["1998"].idxmin().strftime("%Y-%m"),
        "glorys8_monthly_min": None if mon["glorys8"]["1998"].isna().all() else round(float(mon["glorys8"]["1998"].min()), 2),
        "mooring_min_1998": {H.LON_LABEL[c]: round(float(strict[c]["1998"].min()), 2) for c in H.LONS
                             if strict[c]["1998"].notna().any()},
    }
    log("1998 trough: " + json.dumps(summary["trough_1998"]))

    # current value
    last = idx_new["hc"].last_valid_index()
    summary["current"] = {"last_valid_new": str(last.date()) if last is not None else None,
                          "value_new": None if last is None else round(float(idx_new.at[last, "hc"]), 3),
                          "old_last_plotted": str(old_sm.last_valid_index().date()),
                          "old_last_plotted_value": round(float(old_sm.dropna().iloc[-1]), 3),
                          "glorys8_last": None if not len(idx_g.dropna()) else
                          [str(idx_g.last_valid_index().date()), round(float(idx_g.dropna().iloc[-1]), 3)]}
    # daily frame for everything
    daily = pd.DataFrame({"q4_hc": idx_q4["hc"], "q4_n_live": idx_q4["n_live"], "new_hc": idx_new["hc"], "n_live": idx_new["n_live"], "n_carried": idx_new["n_carried"],
                          "old_raw": old_raw, "old_plotted": old_sm, "glorys8": idx_g})
    daily.to_csv(OUT / "daily.csv")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log("SUMMARY " + json.dumps(summary, default=str)[:6000])
    return 0


if __name__ == "__main__":
    if "--glorys-fetch" in sys.argv:
        glorys_series(fetch=True)
        sys.exit(0)
    sys.exit(main())
