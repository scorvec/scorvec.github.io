#!/usr/bin/env python3
"""Colombia inflow forecast: AIFS-ENS + IFS-ENS rain through the kernel model.

Rides the MJO pipeline's 00/12Z tp downloads (scripts/mjo/data/aifs/
{aifs,ifs}_YYYYMMDD_HHz.*.tp.grib2, global 0.25 deg, accumulated from
init).  Four stages, all idempotent:

  1. EXTRACT  — every tp cycle not yet archived is sliced to the Colombia
     box, de-accumulated to daily totals, basin-averaged per ensemble
     member, and appended to ~/colombia_hydro/raw/fcst_rain/ (kept
     forever; a few kB per cycle).  The GRIBs themselves are pruned at
     +7 d by the MJO runner — the archive is the permanent record.
  2. VERIFY   — matured 00Z leads are paired against RAW-IMERG basin
     rain (the kernel model's calibration space: the model was fit on
     raw-IMERG anomalies, so NWP must be mapped into raw-IMERG units,
     NOT gauge-corrected units).  Regularized per-basin, per-lead-band
     multiplicative bias factors F = (sum obs + P)/(sum fcst + P), prior
     P = K_PRIOR days x basin clim — factors start at 1 and earn their
     way off it as pairs accrue.
  3. BLEND    — per basin/band inverse-MSE weights between the two
     bias-corrected ensemble means; 50/50 until MIN_PAIRS pairs.
  4. FAN      — latest cycle of each model: every member's daily rain is
     bias-corrected, converted to an anomaly vs the IMERG harmonic clim,
     spliced onto the observed anomaly history, and propagated through
     the fitted EMA kernel (tau, lag, gain, ENSO term per basin).  The
     lag means lead day d predicts inflow at d+lag.  Weighted quantiles
     across the pooled ~101 members -> % of norm fan.

Outputs:
  ~/colombia_hydro/raw/fcst_rain/{model}_{date}_{hh}z.json.gz  (archive)
  ~/colombia_hydro/raw/imerg_basin_daily.json                  (truth cache)
  ~/colombia_hydro/out/fcst_verif.json                         (private, tracked)
  colombia_hydro/data/inflow_forecast.json                     (fan, public)

    python scripts/sst/colombia_forecast.py
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                   # noqa: E402
from hydro_region_rain import region_weights, gauge_correction, gauge_correction_mtime  # noqa: E402
from build_imerg_clim import OUT as CLIM_NC, eval_clim      # noqa: E402
from rain_inflow_model import ema                           # noqa: E402
from matplotlib.path import Path as MplPath                 # noqa: E402

REPO = HERE.parent.parent
GRIB_DIR = REPO / "scripts" / "mjo" / "data" / "aifs"
REGIONS_GJ = HERE / "colombia_hydro_regions.geojson"
PRIV = Path.home() / "colombia_hydro"
ARCH = PRIV / "raw" / "fcst_rain"
TRUTH_CACHE = PRIV / "raw" / "imerg_basin_daily.json"
VERIF_JSON = PRIV / "out" / "fcst_verif.json"
MODEL_JSON = REPO / "colombia_hydro" / "data" / "rain_inflow_model.json"
ENSO_JSON = REPO / "assets" / "sst" / "data" / "enso_daily.json"
OUT_JSON = REPO / "colombia_hydro" / "data" / "inflow_forecast.json"
VERIF_PNG = REPO / "colombia_hydro" / "rain_verif.webp"
VERIF_PUB = REPO / "colombia_hydro" / "data" / "rain_verif.json"
STORAGE_JSON = REPO / "colombia_hydro" / "data" / "storage.json"
INFLOW_CLIM = REPO / "colombia_hydro" / "data" / "inflow_clim.json"
GEN_MODEL = REPO / "colombia_hydro" / "data" / "gen_model.json"
DAM_MODEL = REPO / "colombia_hydro" / "data" / "dam_models.json"
_CATCH_W: dict = {}                         # catchment masks, keyed by grid shape

ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
BANDS = [(1, 3), (4, 7), (8, 15)]           # lead-day bands for bias/weights
MAX_LEAD = 15                               # days
K_PRIOR = 6.0                               # prior strength, days of clim —
                                            # light on purpose: data dominates
                                            # within ~a week of matured leads
PRIOR_RATIO = {"aifs": 1.0, "ifs": 0.75}    # prior bias-factor center: IFS
                                            # carries a known tropical wet bias
                                            # (user-asserted, ledger 2026-08-16);
                                            # the archive overrides it quickly
MIN_PAIRS = 6                               # per band before weights move off 0.5
SPREAD_MIN_PAIRS = 30                       # per band before spread inflation acts
RES_TAU = 10.0                              # days, decay of the obs-residual anchor
TAU_SLOW_D = 90.0                           # slow kernel for dam fans
STORAGE_JSON_IN = None                      # set below (public data dir)
# Colombia box, generous around the basins (geojson lons are -180..180)
BOX = dict(lon0=-81.0, lon1=-68.0, lat0=-1.5, lat1=12.5)


# ── basin masks on an arbitrary lat/lon grid ────────────────────────────────
def basin_weights(lons: np.ndarray, lats: np.ndarray) -> dict[str, np.ndarray]:
    gj = json.loads(REGIONS_GJ.read_text())
    rings: dict[str, list] = {}
    for ft in gj["features"]:
        nm = (ft["properties"].get("region") or ft["properties"].get("name", "")).upper()
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        rings.setdefault(nm, []).extend(np.array(p[0]) for p in polys)
    LO, LA = np.meshgrid(lons, lats)
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    W = {}
    for r in ORDER:
        inside = np.zeros(LO.shape, bool)
        for rr in rings[r]:
            inside |= MplPath(rr).contains_points(pts).reshape(LO.shape)
        w = np.where(inside, np.cos(np.deg2rad(LA)), 0.0)
        if w.sum() == 0:
            arr = np.vstack(rings[r])
            i = int(np.argmin(np.abs(lats - arr[:, 1].mean())))
            j = int(np.argmin(np.abs(lons - arr[:, 0].mean())))
            w = np.zeros(LO.shape)
            w[i, j] = 1.0
        W[r] = w / w.sum()
    return W


# ── stage 1: extract new cycles from the MJO GRIBs ──────────────────────────
def extract_cycle(model: str, date: str, hh: str) -> dict | None:
    """Basin-mean daily rain (mm/day) per member from one cycle's tp GRIBs."""
    import xarray as xr
    stem = f"{model}_{date}_{hh}z"
    parts = []
    types = ("cf", "pf") if model == "aifs" else ("pf",)
    for typ in types:
        p = GRIB_DIR / f"{stem}.{typ}.tp.grib2"
        if not p.exists():
            if typ == "pf":
                return None                     # pf is the ensemble — required
            continue
        ds = xr.open_dataset(p, engine="cfgrib", chunks={},
                             backend_kwargs={"filter_by_keys": {"shortName": "tp"},
                                             "indexpath": ""})
        da = ds["tp"]
        if da.attrs.get("units", "").strip() in ("m", "metre", "metres", "meters"):
            da = da * 1000.0                    # -> mm; open-data tp is kg m-2 already
        lons = da.longitude.values
        if lons.max() > 180:                    # 0..360 grid -> shift to -180..180
            da = da.assign_coords(longitude=(da.longitude + 180) % 360 - 180)
        da = da.sortby("longitude").sortby("latitude")
        da = da.sel(longitude=slice(BOX["lon0"], BOX["lon1"]),
                    latitude=slice(BOX["lat0"], BOX["lat1"]))
        if "number" not in da.dims:
            da = da.expand_dims("number")
        parts.append(da.compute())
    if not parts:
        return None
    da = parts[0] if len(parts) == 1 else __import__("xarray").concat(parts, dim="number")

    # daily buckets: the tp downloads carry 00Z-valid boundaries only (00Z init
    # -> steps 24,48,...; 12Z init -> 12,36,...), so every 24h bucket between
    # consecutive boundaries is an exact calendar day.  Prepend the implicit
    # zero accumulation at init; keep only buckets spanning a full 24h that
    # start at a 00Z instant (drops the 12h partial first bucket of 12Z runs).
    steps_h = (da.step.values / np.timedelta64(1, "h")).astype(int)
    order = np.argsort(steps_h)
    steps_h = steps_h[order]
    v = da.isel(step=order).transpose("number", "step", "latitude", "longitude").values
    init_dt = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{hh}:00")
    bh = np.concatenate([[0], steps_h])                       # boundary hours
    bv = np.concatenate([np.zeros((v.shape[0], 1) + v.shape[2:]), v], axis=1)
    valid, keep = [], []
    for k in range(len(bh) - 1):
        t0 = init_dt + np.timedelta64(int(bh[k]), "h")
        if bh[k + 1] - bh[k] == 24 and t0 == t0.astype("datetime64[D]"):
            valid.append(str(t0.astype("datetime64[D]")))
            keep.append(k)
    if len(valid) < 3:
        print(f"  {stem}: only {len(valid)} full calendar-day buckets — skipped")
        return None
    daily = np.clip(np.stack([bv[:, k + 1] - bv[:, k] for k in keep], axis=1),
                    0, None)                                  # (nmem, nday, ny, nx)
    W = basin_weights(da.longitude.values, da.latitude.values)
    WC = {}
    if DAM_MODEL.exists():
        try:
            from dam_models import catchment_weights, DAMS
            key = (len(da.longitude), len(da.latitude))
            if key not in _CATCH_W:
                _CATCH_W[key] = catchment_weights(
                    da.longitude.values, da.latitude.values, set(DAMS))
            WC = _CATCH_W[key]
        except Exception as e:              # noqa: BLE001 — dams are optional
            print(f"  catchment masks unavailable: {repr(e)[:60]}")
    out = {"model": model, "init_date": date, "init_hh": hh, "valid": valid,
           "n_members": int(daily.shape[0]), "basins": {}}
    for r in ORDER:
        w = W[r]
        out["basins"][r] = np.round(
            (daily * w[None, None]).sum(axis=(2, 3)), 2).tolist()   # [mem][lead]
    if WC:
        out["rivers"] = {nm: np.round(
            (daily * w[None, None]).sum(axis=(2, 3)), 2).tolist()
            for nm, w in WC.items()}
    return out


def stage_extract() -> int:
    ARCH.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for f in sorted(glob.glob(str(GRIB_DIR / "*_*z.pf.tp.grib2"))):
        m = re.match(r"(aifs|ifs)_(\d{8})_(\d{2})z", Path(f).name)
        if not m:
            continue
        model, date, hh = m.groups()
        dest = ARCH / f"{model}_{date}_{hh}z.json.gz"
        if dest.exists():
            continue
        print(f"extracting {model} {date} {hh}Z ...", flush=True)
        try:
            rec = extract_cycle(model, date, hh)
        except Exception as e:                  # noqa: BLE001 — one bad GRIB must not kill the run
            print(f"  extract failed: {repr(e)[:120]}")
            continue
        if rec is None:
            continue
        with gzip.open(dest, "wt") as fh:
            json.dump(rec, fh, separators=(",", ":"))
        n_new += 1
    return n_new


# ── truth: RAW-IMERG basin daily means, incrementally cached ────────────────
def truth_series() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(dates, rain[r], clim[r]) over the whole IMERG daily cache.

    GAUGE-CORRECTED space (rain and clim both x F) — matches kernel model
    v2, which was refit on corrected rain. The cache stores the F-field
    mtime it was built with and rebuilds in full when the field updates
    (weekly rebuilds shift it only marginally, but keep it exact)."""
    import xarray as xr
    cache = json.loads(TRUTH_CACHE.read_text()) if TRUTH_CACHE.exists() else {"dates": []}
    fmt = gauge_correction_mtime()
    if cache.get("corr_mtime") != fmt:
        cache = {"dates": [], "corr_mtime": fmt}
    files = sorted(IP.DAILY_CACHE.glob("*.npy"))
    days = [f.stem for f in files]
    known = set(cache["dates"])
    new = [d for d in days if d not in known]
    if new:
        ml, mt = IP._grid_axes()
        lons = np.sort(IP._LON[ml])
        lats = np.sort(IP._LAT[mt])
        W = region_weights(REGIONS_GJ, lons, lats)
        F = gauge_correction(lons, lats)
        clim = xr.open_dataset(CLIM_NC)["coef"].values
        for r in ORDER:
            cache.setdefault(r, [])
            cache.setdefault(r + "_clim", [])
        for d in new:
            g = np.load(IP.DAILY_CACHE / f"{d}.npy") * F
            doy = min(datetime.strptime(d, "%Y%m%d").timetuple().tm_yday, 365)
            c = eval_clim(clim, doy) * F
            for r in ORDER:
                w = W[r]
                sw = w.sum()
                cache[r].append(round(float((g * w).sum() / sw), 3))
                cache[r + "_clim"].append(round(float((c * w).sum() / sw), 3))
        cache["dates"] = cache["dates"] + new
        # keep sorted (new days always append in order, but be safe)
        order = np.argsort(cache["dates"])
        for k in list(cache):
            if isinstance(cache[k], list):
                cache[k] = [cache[k][i] for i in order]
        TRUTH_CACHE.write_text(json.dumps(cache, separators=(",", ":")))
        print(f"truth cache: +{len(new)} days -> {len(cache['dates'])}", flush=True)
    dates = np.array([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in cache["dates"]],
                     dtype="datetime64[D]")
    rain = {r: np.array(cache[r], float) for r in ORDER}
    rclim = {r: np.array(cache[r + "_clim"], float) for r in ORDER}
    return dates, rain, rclim


# ── stages 2+3: verification -> bias factors + blend weights ────────────────
def stage_verify(dates, rain, rclim) -> dict:
    dmap = {str(d): i for i, d in enumerate(dates)}
    # pairs[model][r][band] -> [sum_f, sum_o, sse, n]
    acc = {mdl: {r: {b: [0.0, 0.0, 0.0, 0, 0.0] for b in range(len(BANDS))} for r in ORDER}
           for mdl in ("aifs", "ifs")}
    for f in sorted(ARCH.glob("*.json.gz")):    # all buckets are calendar-aligned
        rec = json.loads(gzip.open(f, "rt").read())
        mdl = rec["model"]
        d0 = np.datetime64(f"{rec['init_date'][:4]}-{rec['init_date'][4:6]}-"
                           f"{rec['init_date'][6:8]}")
        for li, vd in enumerate(rec["valid"]):
            i = dmap.get(vd)
            if i is None:
                continue
            lead = int((np.datetime64(vd) - d0).astype(int)) + 1
            band = next((bi for bi, (a, b) in enumerate(BANDS) if a <= lead <= b), None)
            if band is None:
                continue
            for r in ORDER:
                obs = rain[r][i]
                if not np.isfinite(obs):
                    continue
                mems = np.array([mem[li] for mem in rec["basins"][r]])
                fc = float(mems.mean())
                a = acc[mdl][r][band]
                a[0] += fc
                a[1] += obs
                a[2] += (fc - obs) ** 2
                a[3] += 1
                a[4] += float(mems.std())
    clim_mean = {r: float(np.nanmean(rclim[r])) for r in ORDER}
    factors, weights, counts, spread = {}, {}, {}, {}
    for r in ORDER:
        factors[r], weights[r], counts[r], spread[r] = {}, {}, {}, {}
        for bi in range(len(BANDS)):
            P = K_PRIOR * clim_mean[r]
            fA, fI = {}, {}
            for mdl in ("aifs", "ifs"):
                s_f, s_o, sse, n, ssp = acc[mdl][r][bi]
                d_ = fA if mdl == "aifs" else fI
                r0 = PRIOR_RATIO[mdl]
                d_["F"] = (s_o + P * r0) / (s_f + P) if s_f > 0 else r0
                d_["mse"] = sse / n if n else np.nan
                d_["n"] = n
                # spread ratio RMSE/mean-spread: >1 = underdispersive; used
                # to inflate the fan once >=30 pairs (SPREAD_MIN_PAIRS)
                d_["sr"] = round(float(np.sqrt(sse / n) / (ssp / n)), 3) \
                    if n and ssp > 0 else None
            nA, nI = fA["n"], fI["n"]
            if nA >= MIN_PAIRS and nI >= MIN_PAIRS and fA["mse"] > 0 and fI["mse"] > 0:
                wa = (1 / fA["mse"]) / (1 / fA["mse"] + 1 / fI["mse"])
            else:
                wa = 0.5
            factors[r][bi] = {"aifs": round(fA["F"], 3), "ifs": round(fI["F"], 3)}
            weights[r][bi] = round(float(wa), 3)
            counts[r][bi] = {"aifs": nA, "ifs": nI}
            spread[r][bi] = {"aifs": fA["sr"], "ifs": fI["sr"]}
    verif = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
             "truth": "GAUGE-CORRECTED IMERG basin-daily (kernel model v2 space)",
             "bands_lead_days": BANDS, "k_prior_days": K_PRIOR,
             "prior_ratio": PRIOR_RATIO,
             "min_pairs": MIN_PAIRS,
             "bias_factors": factors, "weight_aifs": weights, "pairs": counts,
             "spread_ratio": spread}
    VERIF_JSON.parent.mkdir(parents=True, exist_ok=True)
    VERIF_JSON.write_text(json.dumps(verif, indent=1))
    return verif


# ── stage 4: the fan ────────────────────────────────────────────────────────
def weighted_quantile(vals: np.ndarray, wts: np.ndarray, qs) -> np.ndarray:
    i = np.argsort(vals)
    v, w = vals[i], wts[i]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(qs, cw, v)


def band_of(lead: int) -> int:
    return next((bi for bi, (a, b) in enumerate(BANDS) if a <= lead <= b),
                len(BANDS) - 1)


def stage_fan(dates, rain, rclim, verif) -> None:
    params = json.loads(MODEL_JSON.read_text())["params"]
    # latest archived cycle per model
    latest = {}
    for mdl in ("aifs", "ifs"):
        fs = sorted(ARCH.glob(f"{mdl}_*.json.gz"))
        if fs:
            latest[mdl] = json.loads(gzip.open(fs[-1], "rt").read())
    if not latest:
        print("no archived cycles yet — no fan")
        return
    # RONI held at latest value
    ed = json.loads(ENSO_JSON.read_text())["daily"]
    roni = float(np.array(ed["roni_d"], float)[np.isfinite(np.array(ed["roni_d"], float))][-1])

    # observed inflow %-of-norm (5-day trailing) for the residual anchor:
    # the fan starts at reality and relaxes to the model over ~RES_TAU days
    obs_now = {}
    if INFLOW_CLIM.exists():
        rec_ = json.loads(INFLOW_CLIM.read_text())["recent"]
        for r in ORDER:
            v = np.array(rec_["pct_of_norm"][r], float)
            v[v == 0] = np.nan
            k5 = np.convolve(np.where(np.isfinite(v), v, np.nan),
                             np.ones(5) / 5, mode="full")[:len(v)]
            fin = k5[np.isfinite(k5)]
            obs_now[r] = float(fin[-1]) if len(fin) else None

    obs_last = dates[-1]
    horizon = max(np.datetime64(rec["valid"][-1]) for rec in latest.values())
    fdays = np.arange(obs_last + np.timedelta64(1, "D"), horizon + np.timedelta64(1, "D"))
    doys = np.array([min(d.item().timetuple().tm_yday, 365) for d in fdays])

    # pooled members: anomaly traces on the fdays axis (NaN -> 0 = climatology)
    members, mwts, inits, mvalid = [], [], {}, []
    for mdl, rec in latest.items():
        inits[mdl] = f"{rec['init_date']} {rec['init_hh']}Z"
        vmap = {np.datetime64(v): i for i, v in enumerate(rec["valid"])}
        n = rec["n_members"]
        valid = np.array([vmap.get(d) is not None for d in fdays])
        for mi in range(n):
            mvalid.append(valid)
            tr = {}
            for r in ORDER:
                x = np.zeros(len(fdays))
                for di, d in enumerate(fdays):
                    li = vmap.get(d)
                    if li is None:
                        continue
                    lead = (d - np.datetime64(rec["init_date"][:4] + "-" +
                            rec["init_date"][4:6] + "-" + rec["init_date"][6:8])
                            ).astype(int)
                    F = verif["bias_factors"][r][band_of(max(lead, 1))][mdl]
                    x[di] = F * rec["basins"][r][mi][li]
                tr[r] = x
            members.append(tr)
            # member weight: model blend weight (band-averaged) split over members
            wbar = float(np.mean([verif["weight_aifs"][r][bi] for r in ORDER
                                  for bi in range(len(BANDS))]))
            mwts.append((wbar if mdl == "aifs" else 1 - wbar) / n)
    mwts = np.array(mwts)

    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "inits": inits, "n_members": len(members), "roni": round(roni, 2),
           "truth_last_day": str(obs_last),
           "note": ("%-of-norm fan: pooled bias-corrected AIFS-ENS + IFS-ENS "
                    "members through the per-basin v3 model (fast + slow rain "
                    "kernels, ENSO term, storage-state term held at latest); "
                    "weights/bias from out-of-sample verification vs gauge-corrected IMERG"),
           "basins": {}}
    all_traces = {}
    for r in ORDER:
        p = params[r]
        tau, lag = p["tau_days"], p["lag_days"]
        tau_slow = p.get("tau_slow_days", 90)
        a3, b3 = p["intercept4_pct"], p["gain4_pct_per_mmday"]
        c3, d3 = p["enso4_coef_pct_per_roni"], p["slow4_pct_per_mmday"]
        e3 = p["stor4_pct_per_pt"]
        # storage anomaly held at latest observed value across the fan
        # (months-scale decorrelation; measured, so no feedback loop)
        s_an = 0.0
        if STORAGE_JSON.exists():
            sreg = json.loads(STORAGE_JSON.read_text())["regions"]
            s_an = float(sreg.get(r, {}).get("pct_anom_latest", 0.0))
        hist_anom = rain[r] - rclim[r]
        clim_f = np.array([np.nan] * len(fdays), float)
        # forecast-day climatology from the same harmonic fit (via truth clim by doy)
        # rclim is a smooth harmonic evaluated per doy — reuse by doy lookup
        doy_hist = np.array([min(d.item().timetuple().tm_yday, 365) for d in dates])
        clim_by_doy = {}
        for dd, cc in zip(doy_hist, rclim[r]):
            clim_by_doy.setdefault(int(dd), cc)
        clim_f = np.array([clim_by_doy.get(int(d), float(np.nanmean(rclim[r])))
                           for d in doys])
        traces = np.zeros((len(members), len(fdays)))
        for mi, tr in enumerate(members):
            x = np.concatenate([np.where(np.isfinite(hist_anom), hist_anom, 0.0),
                                tr[r] - clim_f])
            k = ema(x, tau)
            ks = ema(x, tau_slow)
            # y at day t uses kernels at t-lag; fan day j is fdays[j]
            nh = len(hist_anom)
            ix = [min(nh + j - lag, len(k) - 1) for j in range(len(fdays))]
            kk, kks = k[ix], ks[ix]
            traces[mi] = a3 + b3 * kk + c3 * roni + d3 * kks + e3 * s_an
            if mi == 0:
                # model value at the last observed day (same member-independent
                # history) -> residual vs observed, decayed along the fan
                k_now = k[len(hist_anom) - 1 - lag]
                ks_now = ks[len(hist_anom) - 1 - lag]
                fit_now = a3 + b3 * k_now + c3 * roni + d3 * ks_now + e3 * s_an
                resid = (obs_now[r] - fit_now) if obs_now.get(r) is not None else 0.0
                decay = resid * np.exp(-(np.arange(len(fdays)) + 1.0) / RES_TAU)
        traces = traces + decay[None, :]
        # spread inflation about the weighted median once verification says
        # the ensemble is over/under-dispersive (needs SPREAD_MIN_PAIRS pairs)
        srs = [verif["spread_ratio"][r][bi][mdl]
               for bi in range(len(BANDS)) for mdl in ("aifs", "ifs")
               if verif["spread_ratio"][r][bi][mdl] is not None
               and min(verif["pairs"][r][bi].values()) >= SPREAD_MIN_PAIRS]
        if srs:
            sr = float(np.clip(np.mean(srs), 0.7, 1.8))
            med = np.array([weighted_quantile(traces[:, j], mwts, [0.5])[0]
                            for j in range(len(fdays))])
            traces = med[None, :] + sr * (traces - med[None, :])
        all_traces[r] = traces
        out["basins"][r] = {
            "tau": tau, "lag": lag,
            "q": {f"p{int(q*100)}": np.round(
                [weighted_quantile(traces[:, j], mwts, [q])[0]
                 for j in range(len(fdays))], 1).tolist() for q in qs}}
    out["dates"] = [str(d) for d in fdays]
    # bias-corrected basin rain fan (mm/day), masking days beyond a model's
    # first full bucket / horizon so zeros never pollute the quantiles
    mv = np.array(mvalid)
    out["rain"] = {}
    for r in ORDER:
        arr = np.array([m[r] for m in members])
        qd = {f"p{int(q*100)}": [] for q in qs}
        for j in range(len(fdays)):
            ok = mv[:, j]
            if ok.sum() >= 10:
                for q in qs:
                    qd[f"p{int(q*100)}"].append(round(float(
                        weighted_quantile(arr[ok, j], mwts[ok], [q])[0]), 2))
            else:
                for q in qs:
                    qd[f"p{int(q*100)}"].append(None)
        out["rain"][r] = qd

    # ── storage fan: S' = S + inflow(member) - outflow_fcst, % of capacity ──
    if STORAGE_JSON.exists() and INFLOW_CLIM.exists():
        st = json.loads(STORAGE_JSON.read_text())
        iclim = json.loads(INFLOW_CLIM.read_text())["clim"]
        regs = [r for r in ORDER if r in st["regions"]]
        s_last = np.datetime64(st["last_day"])
        stor = {"last_day": st["last_day"], "basins": {}}
        S = {}      # member storage state, kWh
        for r in regs:
            S[r] = np.full(len(members), float(st["regions"][r]["vol_kwh"]))
        # gap days between storage obs and fan start: advance on climatology
        gap = np.arange(s_last + np.timedelta64(1, "D"), fdays[0])
        norm_gwh = {r: np.array(iclim[r]["mean"], float) for r in regs}
        for r in regs:
            ofc = np.array(st["regions"][r]["outflow_fcst_kwh"], float)
            for d in gap:
                dy = min(d.item().timetuple().tm_yday, 365) - 1
                S[r] += norm_gwh[r][dy] * 1e6 - ofc[dy]
        traj = {r: np.zeros((len(members), len(fdays))) for r in regs}
        natI = np.zeros((len(members), len(fdays)))      # national inflow, GWh
        for r in regs:
            cap = float(st["regions"][r]["cap_kwh"])
            ofc = np.array(st["regions"][r]["outflow_fcst_kwh"], float)
            for j, d in enumerate(fdays):
                dy = min(d.item().timetuple().tm_yday, 365) - 1
                infl_kwh = all_traces[r][:, j] / 100.0 * norm_gwh[r][dy] * 1e6
                natI[:, j] += infl_kwh / 1e6
                S[r] = np.clip(S[r] + infl_kwh - ofc[dy], 0, cap)
                traj[r][:, j] = S[r]
            stor["basins"][r] = {
                f"p{int(q*100)}": np.round(
                    [weighted_quantile(100 * traj[r][:, j] / cap, mwts, [q])[0]
                     for j in range(len(fdays))], 2).tolist() for q in qs}
        cap_nat = sum(float(st["regions"][r]["cap_kwh"]) for r in regs)
        Snat = np.sum([traj[r] for r in regs], axis=0)
        stor["basins"]["NATIONAL"] = {
            f"p{int(q*100)}": np.round(
                [weighted_quantile(100 * Snat[:, j] / cap_nat, mwts, [q])[0]
                 for j in range(len(fdays))], 2).tolist() for q in qs}
        out["storage"] = stor
        print(f"storage fan: {len(regs)} regions + NATIONAL "
              f"(state {st['last_day']}, gap {len(gap)} d on climatology)")

        # ── generation fan: persistence + state blend (gen_model.json) ──────
        if GEN_MODEL.exists():
            gm = json.loads(GEN_MODEL.read_text())
            c = gm["coefs"]
            a0, ptau = gm["persistence"]["a0"], gm["persistence"]["tau_days"]
            Gn365 = np.array(gm["gn365_gwh"], float)
            In365 = np.array(gm["in365_gwh"], float)
            Wd = np.array(gm["weekday_offsets_gwh"], float)
            ia_hist = list(gm["ia_recent_gwh"])
            gtr = np.zeros((len(members), len(fdays)))
            state_const = (c["c0"] + c["storage_gwh_per_pt"] * gm["storage_anom_now"]
                           + c["nino_gwh_per_degc"] * gm["nino_now"])
            br = gm.get("inflow_bridge", {"a": 0.0, "b": 1.0})
            for mi in range(len(members)):
                buf = list(ia_hist)
                for j, d in enumerate(fdays):
                    dy = min(d.item().timetuple().tm_yday, 365) - 1
                    buf.append(natI[mi, j] - In365[dy])
                    # bridge: modeled (skill-corrected) composite anomaly ->
                    # observed-inflow units the state model was trained on
                    ia7 = br["a"] + br["b"] * float(np.mean(buf[-7:]))
                    alpha = a0 * np.exp(-(j + 1.0) / ptau)
                    state = state_const + c["inflow7_gwh_per_gwh"] * ia7
                    wd_ = d.item().weekday()
                    gtr[mi, j] = (Gn365[dy] + Wd[wd_]
                                  + alpha * gm["ga_now_gwh"]
                                  + (1 - alpha) * state)
            # member spread only carries the rain signal; the dominant
            # uncertainty is the model residual — widen by lead using the
            # blend's effective skill r_eff(h) = max(alpha(h), LOYO r)
            sig = gm.get("sigma_ga_gwh", 15.0)
            r_loyo = gm.get("r_loyo", 0.55)
            zq = {0.1: -1.2816, 0.25: -0.6745, 0.5: 0.0, 0.75: 0.6745,
                  0.9: 1.2816}
            out["generation"] = {}
            for q in qs:
                row = []
                for j in range(len(fdays)):
                    alpha = a0 * np.exp(-(j + 1.0) / ptau)
                    r_eff = max(alpha, r_loyo)
                    s_h = sig * np.sqrt(max(0.0, 1 - r_eff ** 2))
                    row.append(round(float(
                        weighted_quantile(gtr[:, j], mwts, [q])[0]
                        + zq[q] * s_h), 1))
                out["generation"][f"p{int(q*100)}"] = row
            print(f"generation fan: {gtr[:, 0].mean():.0f} -> "
                  f"{gtr[:, -1].mean():.0f} GWh/d (mean)")

    # ── per-dam fans: propagate saved kernel states with catchment rain ─────
    if DAM_MODEL.exists():
        dm = json.loads(DAM_MODEL.read_text())["params"]
        have_rivers = {mdl: rec for mdl, rec in latest.items() if "rivers" in rec}
        if have_rivers:
            out["dams"] = {}
            for rv, p in dm.items():
                tau, lag = p["tau_days"], p["lag_days"]
                c0, c1, c2, c3, c4 = p["coefs"]
                clim365 = np.array(p["clim365_mmday"], float)
                anchor = ((p["obs_now_pct"] - p["fit_now_pct"])
                          if p["obs_now_pct"] is not None else 0.0)
                traces, tw = [], []
                for mdl, rec in have_rivers.items():
                    if rv not in rec["rivers"]:
                        continue
                    vmap = {np.datetime64(v): i for i, v in enumerate(rec["valid"])}
                    wbar = float(np.mean([verif["weight_aifs"][p["region"]][bi]
                                          for bi in range(len(BANDS))]))
                    wm = (wbar if mdl == "aifs" else 1 - wbar) / rec["n_members"]
                    d0 = np.datetime64(f"{rec['init_date'][:4]}-"
                                       f"{rec['init_date'][4:6]}-"
                                       f"{rec['init_date'][6:8]}")
                    for mem in rec["rivers"][rv]:
                        kf, ksl = p["k_now"], p["ks_now"]
                        kf_h, ks_h = [kf], [ksl]
                        # gap days between last rain day and fan start: x = 0
                        gap_n = int((fdays[0] - np.datetime64(p["last_rain_day"])
                                     ).astype(int)) - 1
                        for _ in range(max(gap_n, 0)):
                            kf = (1 - 1 / tau) * kf
                            ksl = (1 - 1 / TAU_SLOW_D) * ksl
                            kf_h.append(kf)
                            ks_h.append(ksl)
                        ys = []
                        for j, d in enumerate(fdays):
                            li = vmap.get(d)
                            dy = min(d.item().timetuple().tm_yday, 365) - 1
                            lead = int((d - d0).astype(int))
                            Fb = verif["bias_factors"][p["region"]][
                                band_of(max(lead, 1))][mdl]
                            x = (Fb * mem[li] - clim365[dy]) if li is not None else 0.0
                            kf = (1 - 1 / tau) * kf + x / tau
                            ksl = (1 - 1 / TAU_SLOW_D) * ksl + x / TAU_SLOW_D
                            kf_h.append(kf)
                            ks_h.append(ksl)
                            y = (c0 + c1 * kf_h[max(len(kf_h) - 1 - lag, 0)]
                                 + c2 * p["roni_now"]
                                 + c3 * ks_h[max(len(ks_h) - 1 - lag, 0)]
                                 + c4 * p["storage_anom_now"]
                                 + anchor * np.exp(-(j + 1.0) / RES_TAU))
                            ys.append(y)
                        traces.append(ys)
                        tw.append(wm)
                if traces:
                    traces = np.array(traces)
                    tw = np.array(tw)
                    out["dams"][rv] = {
                        f"p{int(q*100)}": np.round(
                            [weighted_quantile(traces[:, j], tw, [q])[0]
                             for j in range(len(fdays))], 1).tolist()
                        for q in qs}
            print(f"dam fans: {len(out.get('dams', {}))} rivers")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT_JSON.relative_to(REPO)}: {len(fdays)} days from "
          f"{fdays[0]} ({len(members)} members, inits {inits})")



# ── day-1 rain scorecard: each cycle's first full 24h vs corrected IMERG ────
def stage_rain_scorecard(dates, rain) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime as _dt

    dmap = {str(d): i for i, d in enumerate(dates)}
    recs = []                      # (valid, model, hh, basin, fcst, obs)
    for f in sorted(ARCH.glob("*.json.gz")):
        rec = json.loads(gzip.open(f, "rt").read())
        vd = rec["valid"][0]       # first full 24h bucket = the day-1 forecast
        i = dmap.get(vd)
        if i is None:
            continue
        for r in ORDER:
            obs = rain[r][i]
            if not np.isfinite(obs):
                continue
            fc = float(np.mean([mem[0] for mem in rec["basins"][r]]))
            recs.append((vd, rec["model"], rec["init_hh"], r, round(fc, 2),
                         round(float(obs), 2)))

    stats = {}
    for mdl in ("aifs", "ifs"):
        stats[mdl] = {}
        for r in ORDER:
            v = [(f_, o_) for (_, m_, _, r_, f_, o_) in recs
                 if m_ == mdl and r_ == r]
            if not v:
                stats[mdl][r] = {"n": 0}
                continue
            fa = np.array([x[0] for x in v])
            oa = np.array([x[1] for x in v])
            st = {"n": len(v), "mae": round(float(np.mean(np.abs(fa - oa))), 2),
                  "bias_ratio": round(float(fa.mean() / oa.mean()), 2)
                  if oa.mean() > 0 else None}
            if len(v) >= 5 and fa.std() > 0 and oa.std() > 0:
                st["r"] = round(float(np.corrcoef(fa, oa)[0, 1]), 2)
            stats[mdl][r] = st

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.5), sharex=True)
    cols = {"aifs": "#1f4e8c", "ifs": "#c62828"}
    for ax, r in zip(axes.flat, ORDER):
        vr = sorted({(vd, o_) for (vd, _, _, r_, _, o_) in recs if r_ == r})
        if vr:
            t = [_dt.strptime(x[0], "%Y-%m-%d") for x in vr]
            ax.plot(t, [x[1] for x in vr], color="k", lw=1.4, marker="o",
                    ms=3, label="corrected IMERG (obs)")
        for mdl in ("aifs", "ifs"):
            vm = sorted([(vd, f_) for (vd, m_, _, r_, f_, _) in recs
                         if r_ == r and m_ == mdl])
            if vm:
                t = [_dt.strptime(x[0], "%Y-%m-%d") for x in vm]
                st = stats[mdl][r]
                lab = (f"{mdl.upper()} day-1 (n={st['n']}, "
                       f"MAE {st.get('mae', '-')}, x{st.get('bias_ratio', '-')})")
                ax.plot(t, [x[1] for x in vm], color=cols[mdl], lw=0, marker="D",
                        ms=4, alpha=0.85, label=lab)
        if not vr:
            ax.text(0.5, 0.5, "awaiting matured forecasts\n(first pairs ~1 day "
                    "after each cycle)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="0.4")
        ax.set_title(r, fontsize=10, fontweight="bold", loc="left")
        ax.grid(lw=0.25, alpha=0.5)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("mm/day", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("Day-1 basin rainfall: AIFS-ENS / IFS-ENS ensemble mean vs "
                 "gauge-corrected IMERG — every 00/12Z cycle, accumulating "
                 "from 2026-08-16", fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(VERIF_PNG, dpi=120)
    plt.close(fig)
    VERIF_PUB.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "truth": "gauge-corrected IMERG basin-daily",
        "stats_day1": stats,
        "records": [{"valid": v, "model": m, "init_hh": h, "basin": b,
                     "fcst_mmday": f_, "obs_mmday": o_}
                    for (v, m, h, b, f_, o_) in recs],
    }, separators=(",", ":")))
    print(f"rain scorecard: {len(recs)} day-1 pairs")


def main() -> int:
    n = stage_extract()
    print(f"extract: {n} new cycle(s)")
    dates, rain, rclim = truth_series()
    verif = stage_verify(dates, rain, rclim)
    npairs = sum(c["aifs"] + c["ifs"] for r in ORDER for c in verif["pairs"][r].values())
    print(f"verify: {npairs} matured basin-lead pairs")
    stage_rain_scorecard(dates, rain)
    stage_fan(dates, rain, rclim, verif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
