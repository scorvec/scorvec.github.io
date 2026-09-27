#!/usr/bin/env python3
"""Site reduction of the CMIP6 ENSO-impact study (2026-09-27; user: "can the enso regression maps be added to the site
(for north and south america variables)?"). LAPTOP, run once after cmip6_enso_impacts.py: reads its saved state and the
member files, adds what the site needs and the study did not keep, and writes a compact reference that the Actions
renderer (enso_impacts_render.py) draws:

    scripts/sst/reference/enso_impacts_site.npz   fields + significance masks on the 2.5 deg Americas grid
    scripts/sst/reference/enso_impacts_site.json  counts, tests, the regional table (values only where significant)

What is added here:
  regression  per member, per season and grid point: OLS slope of the anomaly (own quadratic trend removed) on the
              same season's Nino-3.4 (own quadratic trend removed) - K per K and mm/day per K. Members averaged within
              a model (equal weight), models weighted equally;
              robust = across-model t-test BH-FDR 10 % AND >= 80 % of models agreeing on the sign (as every CMIP6
              map here). Observed: ERA5 t2m (1960-2026) and GPCP v2.3 (1979-2026) on CPC's ONI, per-point OLS t-test,
              BH-FDR 10 % over the map.
  % of normal precipitation: every CMIP6 map is TESTED in mm/day (the study's test, unchanged) and SHOWN as % of the
              multi-model normal; observed maps as % of the observed normal (a per-point scale leaves the per-point
              test unchanged). Cells with a normal under 0.3 mm/day are not scored (a percentage of nothing).
  regional observed composites tested (Welch t-test against neutral seasons, FDR 10 % within each season x class row
              of 20 regions) - the study stored them untested.
  super El Nino month by month over North America (cmip6_super_nino_t2m_monthly.py, recomputed to keep the masks).

    python scripts/sst/cmip6_enso_site.py
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cmip6_enso_impacts as E                                            # noqa: E402

REF = HERE / "reference"
DRY = 0.3                                                                  # mm/day: no % of normal below this


def to_pct(mm, sig, clim):
    """A multi-model precipitation field tested in mm/day, shown as % of the multi-model normal; cells whose normal is
    under DRY mm/day are left unscored (a percentage of nothing)."""
    ok = clim >= DRY
    return np.where(ok, 100.0 * mm / np.maximum(clim, 1e-6), np.nan), sig & ok


def robust_nan(stack):
    """E.robust with cells missing in any model left untested."""
    ok = np.isfinite(stack).all(0)
    mm, sig, p, agree = E.robust(np.where(ok, stack, 0.0))
    return np.where(ok, mm, np.nan), sig & ok, np.where(ok, p, np.nan)


def regressions():
    """Per-model mean regression maps: {(v, season): (names, stack)}; pr in % of normal per K."""
    files = [f for f in sorted(glob.glob(str(E.SRC / "*.nc"))) if ".part" not in f]
    acc = defaultdict(lambda: defaultdict(list))
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        model = ds.attrs["source_id"]; years = ds["year"].values.astype(int)
        F = {v: ds[v].values.astype("float32") for v in E.VARS}
        idx, _ = E.season_index(ds); ds.close()
        for si, s in enumerate(E.SEASONS):
            ok = np.array([y in idx[s] for y in years])
            for v in E.VARS:
                ok &= np.isfinite(F[v][si]).all(axis=(1, 2))
            yrs = years[ok]
            if len(yrs) < 20:
                continue
            x = E.detrend(np.array([idx[s][y] for y in yrs]), yrs.astype(float)).astype(float)
            xc = x - x.mean()
            for v in E.VARS:
                A = E.detrend(F[v][si][ok], yrs.astype(float))
                b = np.tensordot(xc, A - A.mean(0), axes=(0, 0)) / (xc @ xc)
                acc[(v, s)][model].append(b.astype("float32"))
        if (i + 1) % 100 == 0:
            print(f"  regression: {i + 1}/{len(files)} members", flush=True)
    out = {}
    for key, d in acc.items():
        names = sorted(d)
        out[key] = (names, np.stack([np.mean(d[m], axis=0) for m in names]), sum(len(d[m]) for m in names))
    return out


def obs_regression(S):
    """{season: (slope, p, n, span, clim)} per point, OLS on the ONI."""
    out = {}
    for s in E.SEASONS:
        yrs, x, A, clim = S[s]
        xc = x - x.mean(); sxx = xc @ xc
        b = np.tensordot(xc, A - A.mean(0), axes=(0, 0)) / sxx
        res = (A - A.mean(0)) - b[None] * xc[:, None, None]
        n = len(x); se = np.sqrt((res ** 2).sum(0) / (n - 2) / sxx)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = 2 * stats.t.sf(np.abs(b / se), n - 2)
        out[s] = (b, p, n, [int(yrs[0]), int(yrs[-1])], clim)
    return out


def obs_regional_tests(S, W, v):
    """Observed regional class composites with a Welch test vs neutral seasons, FDR within each (season, class) row."""
    out = {}
    for s in E.SEASONS:
        yrs, x, A, clim = S[s]
        R = E.regional(A, W); rcl = E.regional(clim, W); neu = np.abs(x) < E.NEUTRAL
        for c, *_, cond in E.CLASSES:
            sel = cond(x)
            if sel.sum() < E.OBS_TEST_N:
                out[(s, c)] = dict(n=int(sel.sum()), tested=False); continue
            _, p = stats.ttest_ind(R[sel], R[neu], axis=0, equal_var=False)
            val = R[sel].mean(0) - R[neu].mean(0)
            if v == "pr":
                val = 100.0 * val / rcl
            out[(s, c)] = dict(n=int(sel.sum()), tested=True, value=val, p=p, sig=E.fdr(p))
    return out


def super_monthly():
    import cmip6_super_nino_t2m as T
    import cmip6_super_nino_t2m_monthly as M
    pairs = [(f, M.MON / (Path(f).stem + ".npz")) for f in sorted(glob.glob(str(M.SRC / "*.nc"))) if ".part" not in f]
    members = [M.member(a, b) for a, b in pairs if b.exists()]
    per = defaultdict(list)
    for m in members:
        x, A = m["x"], m["A"]; sup, neu = x >= T.SUPER, np.abs(x) < T.NEUTRAL
        if sup.sum() == 0 or neu.sum() < 5:
            continue
        per[m["model"]].append((A[sup].mean(0) - A[neu].mean(0), int(sup.sum())))
    names = sorted(per)
    S = np.stack([np.average([c for c, _ in per[k]], axis=0, weights=[n for _, n in per[k]]) for k in names])
    mm, sig = [], []
    for k in range(S.shape[1]):
        a, b, _, _ = T.robust(S[:, k]); mm.append(a); sig.append(b)
    n_sup = sum(n for k in names for _, n in per[k])
    return dict(lat=members[0]["lat"], lon=members[0]["lon"], mm=np.stack(mm), sig=np.stack(sig), models=len(names),
                members=len(members), events=int(n_sup))


def main() -> int:
    state = pickle.load(open(E.STATE, "rb"))
    lat, lon, W, maps = state["lat"], state["lon"], state["W"], state["maps"]
    npz, meta = {"lat": lat.astype("float32"), "lon": lon.astype("float32"), "land": (state["lf"] >= 0.5)}, {}
    meta["models"] = len(state["members"]); meta["members"] = int(sum(state["members"].values()))
    meta["classes"] = {c: {"name": n, "thr": t} for c, n, t, _ in E.CLASSES}
    meta["maps"] = {}

    def put(key, field, sig, info):
        npz[key + "|f"] = np.asarray(field, "float32"); npz[key + "|s"] = np.asarray(sig, bool); meta["maps"][key] = info

    SEA = dict(JJA0="JJA", SON0="SON", MAM1="MAM", DJF="DJF")
    # CMIP6 composites (classes, derived) and the super lifecycle, pr in % of normal
    for key, e in maps.items():
        kind, v, s, c = key
        if kind not in ("x", "life") or e.get("per_model") is None:
            continue
        names = e["models"]; S = e["per_model"]
        if len(names) < E.MIN_MODELS:
            meta["maps"][f"{kind}|{v}|{s}|{c}"] = dict(models=len(names), events=e["n_events"], tested=False); continue
        mm, sig, _ = robust_nan(S)                  # tested in the study's own units (mm/day for precipitation)
        if v == "pr":
            mm, sig = to_pct(mm, sig, state["clim"][SEA.get(s, s)][1].mean(0))
        land = npz["land"] & np.isfinite(mm)
        put(f"{kind}|{v}|{s}|{c}", mm, sig, dict(models=len(names), events=e["n_events"], members=e["n_members"],
                                                  tested=True, land_robust=round(float(sig[land].mean()), 3)))
    # CMIP6 regressions
    print("CMIP6 regressions ...", flush=True)
    for (v, s), (names, S, nmem) in regressions().items():
        mm, sig, _ = robust_nan(S)
        if v == "pr":
            mm, sig = to_pct(mm, sig, state["clim"][s][1].mean(0))
        land = npz["land"] & np.isfinite(mm)
        put(f"reg|{v}|{s}|cmip6", mm, sig, dict(models=len(names), members=nmem, tested=True,
                                                land_robust=round(float(sig[land].mean()), 3)))
    # observations
    print("observations ...", flush=True)
    oni = E.oni_table()
    OS = {"tas": E.obs_seasons(E.era5_monthly(lat, lon), oni), "pr": E.obs_seasons(E.gpcp_monthly(lat, lon), oni)}
    meta["regions_obs"] = {}
    for v in E.VARS:
        for s, (b, p, n, span, clim) in obs_regression(OS[v]).items():
            ok = np.isfinite(p) & ((clim >= DRY) if v == "pr" else True)
            sig = E.fdr(np.where(ok, p, np.nan))
            f = np.where(clim >= DRY, 100.0 * b / np.maximum(clim, 1e-6), np.nan) if v == "pr" else b
            put(f"reg|{v}|{s}|obs", f, sig, dict(n=n, span=span, tested=True,
                                                  land_sig=round(float(sig[npz["land"] & np.isfinite(f)].mean()), 3)))
        obs = state["obs"][v]
        for s in E.SEASONS:
            clim = obs[(s, "clim")]
            for c, *_ in E.CLASSES:
                o = obs[(s, c)]
                if not o.get("tested"):
                    meta["maps"][f"obs|{v}|{s}|{c}"] = dict(n=o["n"], years=o["years"], tested=False); continue
                f = np.where(clim >= DRY, 100.0 * o["field"] / np.maximum(clim, 1e-6), np.nan) if v == "pr" else o["field"]
                put(f"obs|{v}|{s}|{c}", f, o["sig"] & np.isfinite(f), dict(n=o["n"], years=o["years"], span=o["span"], tested=True))
        # observed super lifecycle placed among 3-event model draws (gridded, FDR 10 %)
        for ph, *_ in E.LIFE:
            o = obs[("life", ph, "super")]
            clim = obs[(SEA[ph], "clim")]
            f = np.where(clim >= DRY, 100.0 * o["field"] / np.maximum(clim, 1e-6), np.nan) if v == "pr" else o["field"]
            put(f"ovm|{v}|{ph}|super", f, o["sig_grid"] & np.isfinite(f),
                dict(n=o["n"], years=o["years"], tested=True, cells=int((o["sig_grid"] & np.isfinite(f)).sum())))
        # regional table: CMIP6 robust values + observed where its own test passes
        rt = obs_regional_tests(OS[v], W, v)
        for (s, c), d in rt.items():
            meta["regions_obs"][f"{v}|{s}|{c}"] = (dict(n=d["n"], tested=False) if not d["tested"] else
                                                   dict(n=d["n"], tested=True,
                                                        value=[round(float(a), 2) for a in d["value"]],
                                                        sig=[bool(a) for a in d["sig"]], p=[float(a) for a in d["p"]]))
    table = E.region_stats(state)
    meta["regions"] = E.RNAMES
    meta["regions_cmip6"] = {}
    for v in E.VARS:
        for s, row in table[v].items():
            for c, e in row.items():
                if s.startswith("life_"):
                    continue
                vals = [e["regions"][r].get("model_pct_of_normal" if v == "pr" else "model_mean") for r in E.RNAMES]
                meta["regions_cmip6"][f"{v}|{s}|{c}"] = dict(
                    models=e["models"], events=e["events"], value=[round(float(a), 2) for a in vals],
                    robust=[bool(e["regions"][r]["robust"]) for r in E.RNAMES],
                    agree=[e["regions"][r]["same_sign"] for r in E.RNAMES],
                    obs_outside=[bool(e["regions"][r].get("obs_outside_models", False)) for r in E.RNAMES],
                    obs_outside_tablewide=[bool(e["regions"][r].get("obs_outside_models_tablewide_fdr", False)) for r in E.RNAMES],
                    obs_pctile=[e["regions"][r].get("obs_percentile_in_models") for r in E.RNAMES],
                    obs_value=[e["regions"][r].get("obs_pct_of_normal" if v == "pr" else "obs") for r in E.RNAMES],
                    obs_n=[e["regions"][r].get("obs_n") for r in E.RNAMES],
                    obs_p=[e["regions"][r].get("obs_vs_models_p") for r in E.RNAMES])
    # super El Nino over North America month by month (its own grid)
    print("super El Nino by month ...", flush=True)
    sm = super_monthly()
    npz["na_lat"] = sm["lat"].astype("float32"); npz["na_lon"] = sm["lon"].astype("float32")
    for k, mname in enumerate(("Nov", "Dec", "Jan", "Feb", "Mar")):
        put(f"namon|tas|{mname}|super", sm["mm"][k], sm["sig"][k],
            dict(models=sm["models"], members=sm["members"], events=sm["events"], tested=True,
                 robust=round(float(sm["sig"][k].mean()), 3)))
    # climatological normal (multi-model mean, mm/day) so the renderer can grey the dry cells
    for s in E.SEASONS:
        npz[f"clim|pr|{s}"] = state["clim"][s][1].mean(0).astype("float32")
    REF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF / "enso_impacts_site.npz", **npz)
    (REF / "enso_impacts_site.json").write_text(json.dumps(meta, separators=(",", ":"), default=float))
    print(f"wrote {REF / 'enso_impacts_site.npz'} ({(REF / 'enso_impacts_site.npz').stat().st_size / 1e6:.1f} MB, "
          f"{len(meta['maps'])} maps) + json ({(REF / 'enso_impacts_site.json').stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
