#!/usr/bin/env python3
"""SEAS5 hindcast case study: one start date, one target season, against ERA5.

    python scripts/sst/seas5_case.py --start 199709 --season DJF

WHY. The SEAS5 page shows what the model says now; this asks how the same
model, from the same start month, did in an analogue year. First case: the
September 1997 start for the winter of 1997/98, the strongest El Niño in the
hindcast period, against the 2026 event (user, 2026-09-16).

DATA. The start-month hindcast already cached for the page
(scripts/sst/data/seas5/hindcast/hc_gl_{MM}.grib: t2m, tprate, sst; hc_gl_z500_{MM}.grib:
z; 1993-2016 × 25 members × 6 leads, 1°). Observed: the local ERA5 store
(~/era5_store, 1.5° daily: t2m and z500 global, precipitation 0-90N). The model
anomaly is member minus the 1993-2016 hindcast mean at the same lead (the same
convention as the page); the observed anomaly is against ERA5's 1993/94-2016/17
mean of the same season, so both sides share the base. Niño-3.4 observed from
assets/sst/data/nino_history.json (CPC ERSST monthly), re-based to 1993-2016.

OUTPUT. {out}/seas5_case_{YYYYMM}_{SEASON}_maps.png (3 × 2 maps: model ensemble
mean vs ERA5), _plume.png (Niño-3.4 members vs observed + pattern correlations
by region), and _summary.json with every number quoted on the figures.
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, xarray as xr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
HC = ROOT / "scripts" / "sst" / "data" / "seas5" / "hindcast"
E5 = Path.home() / "era5_store"
sys.path.insert(0, str(ROOT / "scripts" / "sst"))
G0 = 9.80665
MON = "JFMAMJJASOND"
HC_YEARS = (1993, 2016)
REGIONS = {  # name: (lat0, lat1, lon0, lon1) in 0..360; lon0 > lon1 wraps
    "NH 20-90N": (20, 90, 0, 360), "North America": (20, 75, 190, 310), "Europe": (30, 75, 330, 60),
    "Tropics 30S-30N": (-30, 30, 0, 360), "US + S Canada": (25, 55, 235, 295),
}
BOXES = {  # small boxes for the regional table (lat0, lat1, lon0, lon1)
    "Northern Plains / Upper Midwest": (42, 50, 258, 275), "US Southeast": (30, 36, 270, 285),
    "California": (34, 42, 236, 243), "Pacific NW / BC": (44, 54, 235, 245),
    "Northern Europe": (55, 65, 5, 30), "UK / France": (44, 58, 352, 5),
}
NINO34 = (-5, 5, 190, 240)


def leads_for(start_m: int, season: str) -> list[int]:
    """Lead index (1..6) of each month of `season` after a start in month start_m."""
    i0 = MON.index(season[0]); out = []
    for k, ch in enumerate(season):
        vm = (i0 + k) % 12 + 1
        lead = (vm - start_m) % 12 + 1
        assert 1 <= lead <= 6 and MON[vm - 1] == ch, f"{season} not reachable from month {start_m}"
        out.append(lead)
    return out


def open_hc(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": "", "time_dims": ("forecastMonth", "time")})


def model_season(path: Path, var: str, year: int, mm: int, leads: list[int], fac: float = 1.0):
    """→ (members[25, lat, lon] anomaly, clim[lat, lon], lat, lon[0..360 sorted]) for the season mean."""
    ds = open_hc(path); da = ds[var]
    sel = da.sel(forecastMonth=leads)
    yr = sel.sel(time=f"{year}-{mm:02d}-01").mean("forecastMonth").values.astype(np.float32) * fac   # (number, lat, lon)
    clim = sel.mean(("number", "time", "forecastMonth")).values.astype(np.float32) * fac               # (lat, lon)
    lat, lon = da.latitude.values, da.longitude.values; ds.close()
    lon360 = np.where(lon < 0, lon + 360, lon); order = np.argsort(lon360)
    return (yr - clim)[:, :, order], clim[:, order], lat, lon360[order]


def era5_season(var: str, sub: str, year0: int, season: str, years: tuple[int, int]):
    """Observed season mean for the target winter and the climatology over `years` (season starting in
    those years) → (anom[lat, lon], lat, lon). Handles the store's mixed dimension order by name."""
    i0 = MON.index(season[0]); months = [((i0 + k) % 12 + 1) for k in range(len(season))]
    cache = {}
    def yearfile(y):
        if y not in cache:
            ds = xr.open_dataset(E5 / sub / var / f"{var}_{y}.nc"); cache[y] = ds[var].transpose("time", "latitude", "longitude").load(); ds.close()
        return cache[y]
    def smean(y0):
        parts = []
        for k, m in enumerate(months):
            y = y0 + (1 if (i0 + k) >= 12 else 0)
            x = yearfile(y); parts.append(x.sel(time=x.time.dt.month == m).mean("time"))
        return xr.concat(parts, "m").mean("m")
    tgt = smean(year0)
    clim = xr.concat([smean(y) for y in range(years[0], years[1] + 1)], "y").mean("y")
    a = (tgt - clim); return a.values.astype(np.float32), a.latitude.values, a.longitude.values


def to_grid(field, lat, lon, lat_t, lon_t):
    """Bilinear model (1°) → ERA5 (1.5°) grid, with a wrapped longitude column."""
    da = xr.DataArray(field, dims=("y", "x") if field.ndim == 2 else ("n", "y", "x"),
                      coords={"y": lat, "x": lon}, name="f")
    da = xr.concat([da, da.isel(x=0).assign_coords(x=lon[0] + 360)], "x")
    return da.interp(y=lat_t, x=lon_t, kwargs={"fill_value": None}).values


def region_mask(lat, lon, box):
    la0, la1, lo0, lo1 = box
    L, N = np.meshgrid(lon, lat)
    ok_lat = (N >= la0) & (N <= la1)
    ok_lon = (L >= lo0) & (L <= lo1) if lo0 <= lo1 else ((L >= lo0) | (L <= lo1))
    return ok_lat & ok_lon


def pattern_r(a, b, lat, lon, box):
    m = region_mask(lat, lon, box) & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10: return None
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(a); w = w[m]; x, y = a[m], b[m]
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - xm) * (y - ym), weights=w)
    return float(cov / np.sqrt(np.average((x - xm) ** 2, weights=w) * np.average((y - ym) ** 2, weights=w)))


def box_mean(field, lat, lon, box):
    m = region_mask(lat, lon, box)
    w = (np.cos(np.deg2rad(lat))[:, None] * np.ones((lat.size, lon.size)))
    if field.ndim == 2:
        ok = m & np.isfinite(field); return float(np.average(field[ok], weights=w[ok]))
    return np.array([box_mean(f, lat, lon, box) for f in field])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="199709"); ap.add_argument("--season", default="DJF")
    ap.add_argument("--out", default=str(ROOT / "scripts" / "sst" / "data" / "seas5" / "case"))
    ap.add_argument("--render-only", action="store_true", help="redraw from the cached fields (npz) and summary")
    ap.add_argument("--apply", default=None, help="YYYYMM of a real-time forecast: add the case's observed-minus-hindcast error field to it (same season)")
    ap.add_argument("--composite", default=None, help="comma-separated case starts (YYYYMM, same month): composite of their observed / hindcast / error fields; with --apply, adds the composite error to that forecast"); a = ap.parse_args()
    if a.composite:
        return composite_mode(a)
    if a.apply:
        return apply_mode(a)
    year, mm, season = int(a.start[:4]), int(a.start[4:]), a.season.upper()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    leads = leads_for(mm, season); i0 = MON.index(season[0])
    # the season's first calendar month falls in `year` if it comes after the start month, else year+1
    y_season = year if (i0 + 1) > mm else year + 1
    label = f"{season} {y_season}/{str(y_season + 1)[2:]}" if (i0 + len(season) - 1) >= 12 else f"{season} {y_season}"
    print(f"start {year}-{mm:02d}, {label}, leads {leads}", flush=True)
    stem = out / f"seas5_case_{a.start}_{season}"
    if a.render_only:
        S = json.load(open(str(stem) + "_summary.json")); z = np.load(str(stem) + "_fields.npz")
        F = {v: {k: z[f"{v}_{k}"] for k in ("em", "lat", "lon", "obs", "olat", "olon", "sign", "em_i")} for v in ("t2m", "tp", "z500")}
        render(S, F, stem); print("re-rendered", stem); return

    S = {"start": a.start, "season": season, "label": label, "leads": leads, "members": 25, "hindcast_years": list(HC_YEARS)}
    fields = {}
    for var, path, short, fac, e5var, e5sub in (("t2m", HC / f"hc_gl_{mm:02d}.grib", "t2m", 1.0, "t2m", "wb2_1p5_daily_global"),
                                                ("tp", HC / f"hc_gl_{mm:02d}.grib", "tprate", 86400.0 * 1000, "prcp", "wb2_1p5_daily"),
                                                ("z500", HC / f"hc_gl_z500_{mm:02d}.grib", "z", 1.0 / G0, "z500", "wb2_1p5_daily_global")):
        print(f"  model {var} ...", flush=True)
        mem, clim, lat, lon = model_season(path, short, year, mm, leads, fac)
        print(f"  ERA5 {e5var} ...", flush=True)
        obs, olat, olon = era5_season(e5var, e5sub, y_season, season, HC_YEARS)
        em = mem.mean(0)
        em_i = to_grid(em, lat, lon, olat, olon); mem_i = to_grid(mem, lat, lon, olat, olon)
        rs = {k: pattern_r(em_i, obs, olat, olon, b) for k, b in REGIONS.items()}
        # member-level: pattern r of each member, and the fraction of members with the observed sign per cell
        r_mem = {k: [pattern_r(m_, obs, olat, olon, b) for m_ in mem_i] for k, b in REGIONS.items()}
        sign_agree = (np.sign(mem_i) == np.sign(obs)[None]).mean(0)
        boxes = {}
        for k, b in BOXES.items():
            if b[0] < olat.min() or b[1] > olat.max(): continue
            mm_ = box_mean(mem_i, olat, olon, b)
            boxes[k] = {"obs": round(box_mean(obs, olat, olon, b), 2), "ens_mean": round(float(mm_.mean()), 2),
                        "p10": round(float(np.percentile(mm_, 10)), 2), "p90": round(float(np.percentile(mm_, 90)), 2),
                        "frac_sign": round(float((np.sign(mm_) == np.sign(box_mean(obs, olat, olon, b))).mean()), 2)}
        fields[var] = {"em": em, "lat": lat, "lon": lon, "obs": obs, "olat": olat, "olon": olon, "sign": sign_agree, "em_i": em_i}
        S[var] = {"pattern_r": {k: (None if v is None else round(v, 3)) for k, v in rs.items()},
                  "member_r_median": {k: (None if not any(x is not None for x in v) else round(float(np.median([x for x in v if x is not None])), 3)) for k, v in r_mem.items()},
                  "boxes": boxes}
        print("   pattern r:", S[var]["pattern_r"], flush=True)
        del mem, mem_i

    # Niño-3.4 plume: model members per lead (start month .. last lead) vs hindcast clim; observed re-based to 1993-2016
    print("  Niño-3.4 ...", flush=True)
    ds = open_hc(HC / f"hc_gl_{mm:02d}.grib"); sst = ds["sst"]
    lat, lon = sst.latitude.values, sst.longitude.values; lon360 = np.where(lon < 0, lon + 360, lon)
    m = region_mask(lat, lon360, NINO34); w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(m, dtype=float)
    def bm(arr):  # arr (..., lat, lon)
        x = np.where(m & np.isfinite(arr), arr, np.nan); ww = np.where(m & np.isfinite(arr), w, 0)
        return np.nansum(x * ww, axis=(-2, -1)) / ww.sum(axis=(-2, -1))
    yr = bm(sst.sel(time=f"{year}-{mm:02d}-01").values)            # (number, lead)
    clim = bm(sst.mean("number").values).mean(1)                    # bm → (lead, year); mean over the hindcast years
    ds.close()
    plume = yr - clim[None, :]
    nh = json.load(open(ROOT / "assets" / "sst" / "data" / "nino_history.json"))
    months, absv = nh["months"], nh["series"]["nino34"]["abs"]
    mon_clim = {k: np.mean([v for mo, v in zip(months, absv) if int(mo[:4]) in range(HC_YEARS[0], HC_YEARS[1] + 1) and int(mo[5:]) == k and v is not None]) for k in range(1, 13)}
    obs_n34 = []
    for L in range(6):
        vm = (mm - 1 + L) % 12 + 1; vy = year + ((mm - 1 + L) // 12)
        key = f"{vy}-{vm:02d}"; v = absv[months.index(key)] if key in months else None
        obs_n34.append(None if v is None else round(float(v - mon_clim[vm]), 2))
    S["nino34"] = {"lead_months": [f"{year + ((mm - 1 + L) // 12)}-{((mm - 1 + L) % 12 + 1):02d}" for L in range(6)],
                   "members": np.round(plume, 2).tolist(), "ens_mean": np.round(plume.mean(0), 2).tolist(), "obs": obs_n34,
                   "obs_base": f"{HC_YEARS[0]}-{HC_YEARS[1]} monthly mean of CPC ERSST Niño-3.4"}
    print("   Niño-3.4 ens mean", S["nino34"]["ens_mean"], "obs", obs_n34, flush=True)
    json.dump(S, open(str(stem) + "_summary.json", "w"), indent=1)
    np.savez_compressed(str(stem) + "_fields.npz", **{f"{v}_{k}": arr for v, d in fields.items() for k, arr in d.items()})
    render(S, fields, stem)
    print("wrote", out)


def model_forecast_season(path: Path, hc_path: Path, var: str, leads: list[int], fac: float = 1.0):
    """Real-time forecast members (51) for the season minus the start-month hindcast mean at the same leads."""
    ds = open_hc(path); da = ds[var]
    fc = da.sel(forecastMonth=leads).mean("forecastMonth").values.astype(np.float32) * fac        # (number, lat, lon)
    lat, lon = da.latitude.values, da.longitude.values; ds.close()
    hs = open_hc(hc_path); clim = hs[var].sel(forecastMonth=leads).mean(("number", "time", "forecastMonth")).values.astype(np.float32) * fac; hs.close()
    lon360 = np.where(lon < 0, lon + 360, lon); order = np.argsort(lon360)
    return (fc - clim)[:, :, order], lat, lon360[order]


def apply_mode(a):
    """The user's ask (2026-09-16): 'apply the same spatial pattern of errors from the hindcast to
    this year's forecast'. Error = observed minus the hindcast ensemble mean for the case season,
    on the ERA5 grid; adjusted = this year's ensemble-mean anomaly (same season, same leads,
    against the same 1993-2016 hindcast base) plus that error. It is ONE case's error, so it
    carries that winter's unforecastable weather (the 1997/98 NAO+ over Europe) as well as any
    systematic amplitude shortfall; the figure shows all three so the reader can see which is which."""
    year, mm, season = int(a.start[:4]), int(a.start[4:]), a.season.upper()
    fy, fm = int(a.apply[:4]), int(a.apply[4:])
    assert fm == mm, "the forecast must share the case's start month (same leads, same hindcast base)"
    out = Path(a.out); stem = out / f"seas5_case_{a.start}_{season}"
    S = json.load(open(str(stem) + "_summary.json")); z = np.load(str(stem) + "_fields.npz")
    leads = leads_for(mm, season); i0 = MON.index(season[0])
    y_f = fy if (i0 + 1) > mm else fy + 1
    flabel = f"{season} {y_f}/{str(y_f + 1)[2:]}" if (i0 + len(season) - 1) >= 12 else f"{season} {y_f}"
    FC = str(ROOT / "scripts" / "sst" / "data" / "seas5" / "forecast")
    A = {"case": a.start, "forecast": a.apply, "season": season, "case_label": S["label"], "forecast_label": flabel, "leads": leads}
    F = {}
    for var, fpath, hpath, short, fac in (("t2m", f"{FC}/fc_gl_{a.apply}.grib", HC / f"hc_gl_{mm:02d}.grib", "t2m", 1.0),
                                         ("tp", f"{FC}/fc_gl_{a.apply}.grib", HC / f"hc_gl_{mm:02d}.grib", "tprate", 86400.0 * 1000),
                                         ("z500", f"{FC}/fc_gl_z500_{a.apply}.grib", HC / f"hc_gl_z500_{mm:02d}.grib", "z", 1.0 / G0)):
        print(f"  forecast {var} ...", flush=True)
        mem, lat, lon = model_forecast_season(Path(fpath), hpath, short, leads, fac)
        olat, olon, obs, em_i = z[f"{var}_olat"], z[f"{var}_olon"], z[f"{var}_obs"], z[f"{var}_em_i"]
        err = obs - em_i                                                    # 1997/98: observed minus hindcast ensemble mean
        mem_i = to_grid(mem, lat, lon, olat, olon); em_f = mem_i.mean(0); adj = em_f + err; adj_m = mem_i + err[None]
        boxes = {}
        for k, b in BOXES.items():
            if b[0] < olat.min() or b[1] > olat.max(): continue
            raw_m = box_mean(mem_i, olat, olon, b); adj_mm = box_mean(adj_m, olat, olon, b)
            boxes[k] = {"raw_ens_mean": round(float(raw_m.mean()), 2), "raw_p10": round(float(np.percentile(raw_m, 10)), 2), "raw_p90": round(float(np.percentile(raw_m, 90)), 2),
                        "error_1997": round(box_mean(err, olat, olon, b), 2), "adjusted": round(float(adj_mm.mean()), 2),
                        "adj_p10": round(float(np.percentile(adj_mm, 10)), 2), "adj_p90": round(float(np.percentile(adj_mm, 90)), 2)}
        A[var] = {"boxes": boxes, "pattern_r_raw_vs_1997obs": {k: (None if v is None else round(v, 3)) for k, v in ((kk, pattern_r(em_f, obs, olat, olon, bb)) for kk, bb in REGIONS.items())}}
        F[var] = {"raw": em_f, "err": err, "adj": adj, "olat": olat, "olon": olon}
        print("   boxes:", {k: (v["raw_ens_mean"], v["error_1997"], v["adjusted"]) for k, v in boxes.items()}, flush=True)
        del mem, mem_i, adj_m
    astem = out / f"seas5_case_{a.start}_{season}_apply_{a.apply}"
    np.savez_compressed(str(out / f"seas5_forecast_{a.apply}_{season}_fields.npz"), **{f"{v}_{k}": F[v][k] for v in F for k in ("raw", "olat", "olon")})
    json.dump(A, open(str(astem) + "_summary.json", "w"), indent=1)
    render_apply(A, F, astem); print("wrote", astem)


def forecast_fields(a, season, mm, leads, out):
    """Ensemble-mean anomaly of the real-time forecast on the ERA5 grid per variable, cached as npz."""
    cache = out / f"seas5_forecast_{a.apply}_{season}_fields.npz"
    if cache.exists():
        z = np.load(cache); return {v: {k: z[f"{v}_{k}"] for k in ("raw", "olat", "olon")} for v in ("t2m", "tp", "z500")}
    FC = str(ROOT / "scripts" / "sst" / "data" / "seas5" / "forecast"); F = {}
    ref = np.load(str(out / f"seas5_case_{a.composite.split(',')[0]}_{season}_fields.npz"))
    for var, fpath, hpath, short, fac in (("t2m", f"{FC}/fc_gl_{a.apply}.grib", HC / f"hc_gl_{mm:02d}.grib", "t2m", 1.0),
                                         ("tp", f"{FC}/fc_gl_{a.apply}.grib", HC / f"hc_gl_{mm:02d}.grib", "tprate", 86400.0 * 1000),
                                         ("z500", f"{FC}/fc_gl_z500_{a.apply}.grib", HC / f"hc_gl_z500_{mm:02d}.grib", "z", 1.0 / G0)):
        print(f"  forecast {var} ...", flush=True)
        mem, lat, lon = model_forecast_season(Path(fpath), hpath, short, leads, fac)
        olat, olon = ref[f"{var}_olat"], ref[f"{var}_olon"]
        F[var] = {"raw": to_grid(mem, lat, lon, olat, olon).mean(0), "olat": olat, "olon": olon}; del mem
    np.savez_compressed(str(cache), **{f"{v}_{k}": F[v][k] for v in F for k in ("raw", "olat", "olon")})
    return F


def composite_mode(a):
    """Composite of several September-start cases (same season): mean observed, mean hindcast ensemble
    mean, mean error, and where the cases AGREE on the error's sign (the systematic part); with --apply,
    the composite error added to this year's forecast."""
    cases = a.composite.split(","); season = a.season.upper(); mm = int(cases[0][4:])
    assert all(c[4:] == cases[0][4:] for c in cases), "composite cases must share the start month"
    out = Path(a.out); leads = leads_for(mm, season)
    Ss = {c: json.load(open(out / f"seas5_case_{c}_{season}_summary.json")) for c in cases}
    Zs = {c: np.load(out / f"seas5_case_{c}_{season}_fields.npz") for c in cases}
    C = {"cases": cases, "labels": {c: Ss[c]["label"] for c in cases}, "season": season}
    F = {}
    for var in ("t2m", "tp", "z500"):
        obs = np.stack([Zs[c][f"{var}_obs"] for c in cases]); em = np.stack([Zs[c][f"{var}_em_i"] for c in cases]); err = obs - em
        olat, olon = Zs[cases[0]][f"{var}_olat"], Zs[cases[0]][f"{var}_olon"]
        agree = (np.sign(err) == np.sign(err.mean(0))[None]).all(0)             # every case's error has the composite's sign
        F[var] = {"obs": obs.mean(0), "em": em.mean(0), "err": err.mean(0), "agree": agree, "olat": olat, "olon": olon, "err_cases": err}
        C[var] = {"pattern_r_composite": {k: (None if v is None else round(v, 3)) for k, v in ((kk, pattern_r(em.mean(0), obs.mean(0), olat, olon, bb)) for kk, bb in REGIONS.items())},
                  "pattern_r_cases": {c: Ss[c][var]["pattern_r"] for c in cases},
                  "agree_frac": {k: round(float(agree[region_mask(olat, olon, b)].mean()), 3) for k, b in REGIONS.items()},
                  "boxes": {k: {"obs": round(box_mean(obs.mean(0), olat, olon, b), 2), "hindcast": round(box_mean(em.mean(0), olat, olon, b), 2),
                                "error": round(box_mean(err.mean(0), olat, olon, b), 2), "error_cases": [round(box_mean(e, olat, olon, b), 2) for e in err]}
                            for k, b in BOXES.items() if b[0] >= olat.min() and b[1] <= olat.max()}}
        print(var, "composite r", C[var]["pattern_r_composite"], "| agree", C[var]["agree_frac"], flush=True)
    tag = "_".join(c[:4] for c in cases); cstem = out / f"seas5_composite_{tag}_{season}"
    if a.apply:
        fy = int(a.apply[:4]); i0 = MON.index(season[0]); y_f = fy if (i0 + 1) > mm else fy + 1
        C["forecast"] = a.apply; C["forecast_label"] = f"{season} {y_f}/{str(y_f + 1)[2:]}" if (i0 + len(season) - 1) >= 12 else f"{season} {y_f}"
        FF = forecast_fields(a, season, mm, leads, out)
        for var in ("t2m", "tp", "z500"):
            F[var]["raw"] = FF[var]["raw"]; F[var]["adj"] = FF[var]["raw"] + F[var]["err"]
            F[var]["adj_agree"] = FF[var]["raw"] + np.where(F[var]["agree"], F[var]["err"], 0.0)
            olat, olon = F[var]["olat"], F[var]["olon"]
            C[var]["apply_boxes"] = {k: {"raw": round(box_mean(FF[var]["raw"], olat, olon, b), 2), "adjusted": round(box_mean(F[var]["adj"], olat, olon, b), 2),
                                         "adjusted_agree_only": round(box_mean(F[var]["adj_agree"], olat, olon, b), 2)}
                                     for k, b in BOXES.items() if b[0] >= olat.min() and b[1] <= olat.max()}
            C[var]["pattern_r_raw_vs_composite_obs"] = {k: (None if v is None else round(v, 3)) for k, v in ((kk, pattern_r(FF[var]["raw"], F[var]["obs"], olat, olon, bb)) for kk, bb in REGIONS.items())}
        cstem = out / f"seas5_composite_{tag}_{season}_apply_{a.apply}"
    json.dump(C, open(str(cstem) + "_summary.json", "w"), indent=1)
    render_composite(C, F, cstem); print("wrote", cstem)


def render_composite(C, F, stem):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.colors as mcolors
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    INK, MUTED = "#1a1a1a", "#6f6b64"
    pc = ccrs.PlateCarree(); proj = ccrs.PlateCarree(central_longitude=-140)
    spec = {"t2m": ("2 m temperature anomaly, K", [-6, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 6], "RdBu_r"),
            "tp": ("Precipitation anomaly, mm/day", [-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3], "BrBG"),
            "z500": ("500 hPa height anomaly, m", [-120, -80, -60, -40, -20, -10, 10, 20, 40, 60, 80, 120], "RdBu_r")}
    labs = " + ".join(C["labels"][c] for c in C["cases"]); n = len(C["cases"])
    def draw(fig, gs, r, c, var, fld, title, bold, hatch=None, cbar=False):
        lab, lev, cm = spec[var]; f = F[var]; norm = mcolors.BoundaryNorm(lev, 256, extend="both")
        ax = fig.add_subplot(gs[r, c], projection=proj); ax.set_extent([-180, 180, 0 if var == "tp" else -60, 85], crs=pc)
        lon = f["olon"]; lon_c = np.concatenate([lon, [lon[0] + 360]]); fld_c = np.concatenate([fld, fld[:, :1]], axis=1)
        im = ax.pcolormesh(lon_c, f["olat"], fld_c, cmap=cm, norm=norm, transform=pc, shading="auto")
        if hatch is not None:
            h = np.concatenate([hatch, hatch[:, :1]], axis=1)
            ax.contourf(lon_c, f["olat"], np.where(~h, 1, np.nan), levels=[0.5, 1.5], colors="none", hatches=["...."], transform=pc)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.55, edgecolor="#333"); ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#777")
        ax.set_title(title, loc="left", fontsize=8.4, fontweight="bold" if bold else "normal", color=INK, pad=3)
        if cbar:
            cax = ax.inset_axes([1.012, 0.04, 0.016, 0.92]); cb = fig.colorbar(im, cax=cax, extend="both"); cb.ax.tick_params(labelsize=6.5, colors=MUTED, length=0, pad=1.5); cb.set_label(lab, fontsize=7.2, color=MUTED, labelpad=2)
    # figure 1: composite hindcast | composite observed | composite error (dots: cases disagree on sign)
    fig = plt.figure(figsize=(16, 8.2)); gs = fig.add_gridspec(3, 3, left=0.012, right=0.962, top=0.895, bottom=0.01, hspace=0.14, wspace=0.02, height_ratios=[145, 85, 145])
    for r, var in enumerate(("t2m", "tp", "z500")):
        rc = C[var]["pattern_r_composite"]; rcs = C[var]["pattern_r_cases"]
        sub = "N America r " + " · ".join(f"{C['labels'][c][-5:]} {rcs[c]['North America']:+.2f}" for c in C["cases"]) + f" · comp {rc['North America']:+.2f}"
        draw(fig, gs, r, 0, var, F[var]["em"], f"SEAS5 1 Sep hindcast ensemble mean, {n}-event composite\n{sub}", True)
        draw(fig, gs, r, 1, var, F[var]["obs"], f"ERA5 observed, {n}-event composite\nEurope r composite {rc['Europe']:+.2f} · " + " · ".join(f"{C['labels'][c][-5:]} {rcs[c]['Europe']:+.2f}" for c in C["cases"]), False)
        draw(fig, gs, r, 2, var, F[var]["err"], f"Composite error: observed − hindcast (dots: the {n} events disagree on the sign)", True, hatch=F[var]["agree"], cbar=True)
    fig.text(0.012, 0.985, f"SEAS5 September-start hindcasts, strong El Niño composite: {labs}", fontsize=15.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.948, f"Each event: 25-member ensemble mean from the 1 September start, months 4–6, minus the 1993–2016 September-start hindcast mean; ERA5 minus its own 1993/94–2016/17 mean of the same season. The composite averages the {n} events on each side. "
             "Where the dots are absent, every event's error had the composite's sign — the repeatable part of what the model gets wrong. ERA5 1.5° grid; precipitation 0–90 N only.", fontsize=8, color=MUTED, va="top", wrap=True)
    base = str(stem).split("_apply_")[0]                      # the composite figure keeps its own name when --apply is on
    fig.savefig(base + "_maps.png", dpi=120, facecolor="white"); plt.close(fig)
    if "forecast" not in C: return
    # figure 2: this year's forecast | composite error | forecast + error where the events agree
    fig = plt.figure(figsize=(16, 8.2)); gs = fig.add_gridspec(3, 3, left=0.012, right=0.962, top=0.895, bottom=0.01, hspace=0.14, wspace=0.02, height_ratios=[145, 85, 145])
    for r, var in enumerate(("t2m", "tp", "z500")):
        rr = C[var]["pattern_r_raw_vs_composite_obs"]
        draw(fig, gs, r, 0, var, F[var]["raw"], f"SEAS5 1 Sep {C['forecast'][:4]} forecast, 51 members: {C['forecast_label']}\nN America r vs the composite observed winter {rr['North America']:+.2f} · Europe {rr['Europe']:+.2f}", True)
        draw(fig, gs, r, 1, var, F[var]["err"], f"Composite hindcast error of {n} strong El Niños (dots: sign not shared by all)", False, hatch=F[var]["agree"])
        draw(fig, gs, r, 2, var, F[var]["adj_agree"], f"{C['forecast_label']} forecast + composite error where the {n} events agree", True, cbar=True)
    fig.text(0.012, 0.985, f"This winter's SEAS5 forecast with the strong-El Niño composite error added: {C['forecast_label']}", fontsize=15.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.948, f"Left: ensemble-mean anomaly of the 1 Sep {C['forecast'][:4]} forecast against the 1993–2016 September-start hindcast (months 4–6). Middle: the average of what the same model got wrong from 1 September in {labs}. "
             "Right: left plus middle, but only where all events erred the same way; elsewhere the forecast is left alone. That keeps the repeatable amplitude shortfall and drops each winter's own weather. ERA5 1.5° grid; precipitation 0–90 N only.", fontsize=8, color=MUTED, va="top", wrap=True)
    fig.savefig(str(stem) + "_maps.png", dpi=120, facecolor="white"); plt.close(fig)


def render_apply(A, F, stem):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.colors as mcolors
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    INK, MUTED = "#1a1a1a", "#6f6b64"
    pc = ccrs.PlateCarree(); proj = ccrs.PlateCarree(central_longitude=-140)
    spec = {"t2m": ("2 m temperature anomaly, K", [-6, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 6], "RdBu_r"),
            "tp": ("Precipitation anomaly, mm/day", [-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3], "BrBG"),
            "z500": ("500 hPa height anomaly, m", [-120, -80, -60, -40, -20, -10, 10, 20, 40, 60, 80, 120], "RdBu_r")}
    fig = plt.figure(figsize=(16, 8.2))
    gs = fig.add_gridspec(3, 3, left=0.012, right=0.962, top=0.895, bottom=0.01, hspace=0.14, wspace=0.02, height_ratios=[145, 85, 145])
    cols = ((f"SEAS5 1 Sep {A['forecast'][:4]} forecast, 51 members: {A['forecast_label']}", "raw"),
            (f"Error of the 1 Sep {A['case'][:4]} hindcast: observed − ensemble mean, {A['case_label']}", "err"),
            (f"{A['forecast_label']} forecast + {A['case_label']} error", "adj"))
    for r, var in enumerate(("t2m", "tp", "z500")):
        lab, lev, cm = spec[var]; f = F[var]; norm = mcolors.BoundaryNorm(lev, 256, extend="both")
        for c, (title, key) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c], projection=proj)
            ax.set_extent([-180, 180, 0 if var == "tp" else -60, 85], crs=pc)
            lon = f["olon"]; fld = f[key]; lon_c = np.concatenate([lon, [lon[0] + 360]]); fld_c = np.concatenate([fld, fld[:, :1]], axis=1)
            im = ax.pcolormesh(lon_c, f["olat"], fld_c, cmap=cm, norm=norm, transform=pc, shading="auto")
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.55, edgecolor="#333"); ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#777")
            ax.set_title(title, loc="left", fontsize=8.6, fontweight="bold" if c != 1 else "normal", color=INK, pad=3)
            if c == 2:
                cax = ax.inset_axes([1.012, 0.04, 0.016, 0.92]); cb = fig.colorbar(im, cax=cax, extend="both"); cb.ax.tick_params(labelsize=6.5, colors=MUTED, length=0, pad=1.5); cb.set_label(lab, fontsize=7.2, color=MUTED, labelpad=2)
    fig.text(0.012, 0.985, f"This winter's SEAS5 forecast with the {A['case_label']} hindcast error added", fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.948, f"Left: ensemble-mean anomaly of the 1 Sep {A['forecast'][:4]} forecast for {A['forecast_label']} against the 1993–2016 September-start hindcast (months 4–6). Middle: what the same model, from 1 Sep {A['case'][:4]}, "
             f"got wrong for {A['case_label']} (ERA5 minus the 25-member hindcast mean, both against 1993–2016). Right: left plus middle. The error is ONE winter's, so it carries that winter's weather as well as any systematic shortfall — "
             "read the middle column for what is being added. ERA5 1.5° grid; precipitation 0–90 N only.", fontsize=8, color=MUTED, va="top", wrap=True)
    fig.savefig(str(stem) + "_maps.png", dpi=120, facecolor="white"); plt.close(fig)


def render(S, F, stem):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.colors as mcolors
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    INK, MUTED = "#1a1a1a", "#6f6b64"
    pc = ccrs.PlateCarree(); proj = ccrs.PlateCarree(central_longitude=-140)
    spec = {"t2m": ("2 m temperature anomaly, K", [-6, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 6], "RdBu_r"),
            "tp": ("Precipitation anomaly, mm/day", [-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3], "BrBG"),
            "z500": ("500 hPa height anomaly, m", [-120, -80, -60, -40, -20, -10, 10, 20, 40, 60, 80, 120], "RdBu_r")}
    fig = plt.figure(figsize=(14, 10.2))
    gs = fig.add_gridspec(3, 2, left=0.015, right=0.955, top=0.905, bottom=0.01, hspace=0.12, wspace=0.025, height_ratios=[145, 85, 145])
    for r, var in enumerate(("t2m", "tp", "z500")):
        lab, lev, cm = spec[var]; f = F[var]
        norm = mcolors.BoundaryNorm(lev, 256, extend="both")
        for c, (title, fld, lat, lon) in enumerate(((f"SEAS5 ensemble mean, 25 members, start 1 Sep {S['start'][:4]}", f["em"], f["lat"], f["lon"]),
                                                   ("ERA5 observed", f["obs"], f["olat"], f["olon"]))):
            ax = fig.add_subplot(gs[r, c], projection=proj)
            ax.set_extent([-180, 180, -60 if var != "tp" else 0, 85], crs=pc) if var != "tp" else ax.set_extent([-180, 180, 0, 85], crs=pc)
            lon_c = np.concatenate([lon, [lon[0] + 360]]); fld_c = np.concatenate([fld, fld[:, :1]], axis=1)
            im = ax.pcolormesh(lon_c, lat, fld_c, cmap=cm, norm=norm, transform=pc, shading="auto")
            if c == 0 and var != "tp":
                # hatch where fewer than 70 % of members agree with the OBSERVED sign (a hindsight consistency mask)
                sg = f["sign"]; ol, oo = f["olat"], f["olon"]
                sgc = np.concatenate([sg, sg[:, :1]], axis=1); ooc = np.concatenate([oo, [oo[0] + 360]])
                ax.contourf(ooc, ol, np.where(sgc < 0.7, 1, np.nan), levels=[0.5, 1.5], colors="none", hatches=["...."], transform=pc)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, edgecolor="#333")
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#777")
            r_nh = S[var]["pattern_r"].get("NH 20-90N"); r_na = S[var]["pattern_r"].get("North America"); r_eu = S[var]["pattern_r"].get("Europe")
            sub = f"pattern r vs ERA5: NH {r_nh:+.2f} · N America {r_na:+.2f} · Europe {r_eu:+.2f}" if c == 0 and r_nh is not None else ""
            ax.set_title(title + (f"\n{sub}" if sub else ""), loc="left", fontsize=9.5, fontweight="bold" if c == 0 else "normal", color=INK, pad=3)
            if c == 1:
                cax = ax.inset_axes([1.012, 0.04, 0.016, 0.92]); cb = fig.colorbar(im, cax=cax, extend="both"); cb.ax.tick_params(labelsize=6.8, colors=MUTED, length=0, pad=1.5); cb.set_label(lab, fontsize=7.5, color=MUTED, labelpad=2)
    fig.text(0.015, 0.985, f"SEAS5 September {S['start'][:4]} hindcast vs what happened: {S['label']}", fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.952, "Model: ECMWF SEAS5 hindcast, 25 members, start 1 September, months 4–6 (Dec–Feb); anomaly = member mean minus the 1993–2016 September-start hindcast mean at the same leads. "
             "Observed: ERA5 (1.5°), anomaly against ERA5's 1993/94–2016/17 season mean. Dots: fewer than 70 % of members share the observed sign. Precipitation is compared on 0–90 N only (ERA5 store coverage).",
             fontsize=8.2, color=MUTED, va="top", wrap=True)
    fig.savefig(str(stem) + "_maps.png", dpi=120, facecolor="white"); plt.close(fig)

    # plume + correlations
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1], "left": 0.05, "right": 0.99, "top": 0.78, "bottom": 0.16, "wspace": 0.22})
    N = S["nino34"]; x = np.arange(6); labs = [f"{MON[int(m[5:]) - 1]}\n{m[:4]}" for m in N["lead_months"]]
    for mem in N["members"]: a1.plot(x, mem, color="#2a78d6", lw=0.7, alpha=0.35)
    a1.plot(x, N["ens_mean"], color="#2a78d6", lw=2.4, label="SEAS5 members (thin) and ensemble mean")
    ov = [np.nan if v is None else v for v in N["obs"]]; a1.plot(x, ov, color=INK, lw=2.4, marker="o", ms=5, label="Observed (CPC ERSST Niño-3.4, re-based 1993–2016)")
    for xi, (m_, o_) in enumerate(zip(N["ens_mean"], ov)):
        if np.isfinite(o_): a1.annotate(f"{o_:+.1f}", (xi, o_), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    a1.annotate(f"{N['ens_mean'][-1]:+.1f}", (5, N["ens_mean"][-1]), xytext=(6, 0), textcoords="offset points", fontsize=7.5, color="#2a78d6", va="center")
    a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=8); a1.axhline(0, color="#c3c2b7", lw=0.7); a1.grid(axis="y", color="#ebe9e3", lw=0.6)
    for s in ("top", "right"): a1.spines[s].set_visible(False)
    a1.set_ylabel("Niño-3.4 anomaly, °C", fontsize=8.5, color=MUTED); a1.tick_params(labelsize=8, colors=MUTED)
    a1.legend(loc="upper left", fontsize=7.8, frameon=False); a1.set_title("Niño-3.4 from the 1 Sep start: members vs observed", loc="left", fontsize=10, fontweight="bold", color=INK)
    regs = list(REGIONS); cols = {"t2m": "#2a78d6", "tp": "#1baf7a", "z500": "#4a3aa7"}; names = {"t2m": "2 m temperature", "tp": "Precipitation", "z500": "500 hPa height"}
    yb = np.arange(len(regs)); hgt = 0.26
    for i, var in enumerate(("t2m", "tp", "z500")):
        vals = [S[var]["pattern_r"].get(rg) for rg in regs]; vals_med = [S[var]["member_r_median"].get(rg) for rg in regs]
        a2.barh(yb + (i - 1) * hgt, [np.nan if v is None else v for v in vals], height=hgt - 0.03, color=cols[var], label=names[var])
        for yy, v, vm in zip(yb + (i - 1) * hgt, vals, vals_med):
            if v is not None: a2.text(v + (0.012 if v >= 0 else -0.012), yy, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.2, color=INK)
            if vm is not None: a2.plot([vm], [yy], marker="|", ms=9, mew=1.6, color="white"); a2.plot([vm], [yy], marker="|", ms=7, mew=1.0, color=INK)
    a2.set_yticks(yb); a2.set_yticklabels(regs, fontsize=8.5); a2.invert_yaxis(); a2.axvline(0, color="#c3c2b7", lw=0.7); a2.set_xlim(-0.25, 1.08)
    a2.grid(axis="x", color="#ebe9e3", lw=0.6); a2.tick_params(labelsize=8, colors=MUTED)
    for s in ("top", "right"): a2.spines[s].set_visible(False)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=7.8, frameon=False)
    a2.set_title(f"Pattern correlation vs ERA5, {S['label']} (tick: median member)", loc="left", fontsize=10, fontweight="bold", color=INK)
    fig.text(0.05, 0.97, f"SEAS5 September {S['start'][:4]} hindcast: ENSO plume and skill by region, {S['label']}", fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.05, 0.915, "Cosine-weighted pattern correlation of the ensemble-mean anomaly with the observed anomaly over each region; the tick marks the median single member, so the gap to the bar is what the ensemble mean buys over one run. Precipitation regions are cut at the equator.", fontsize=8.2, color=MUTED, va="top", wrap=True)
    fig.savefig(str(stem) + "_plume.png", dpi=120, facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    main()
