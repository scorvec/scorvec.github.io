#!/usr/bin/env python3
"""Brazil ENA forecast engine: AIFS-ENS + IFS-ENS rain over SIN basins.

Same architecture as colombia_forecast: rides the MJO pipeline's tp
GRIBs (global — includes the 3-week AWS backfill), extracts basin-mean
member rain over the 12 major SIN basins, verifies matured leads
against the IMERG basin truth, maintains regularized bias factors +
inverse-MSE blend weights per basin/lead-band, and propagates each
basin's fitted kernel state (brazil_models.json) with bias-corrected
member rain into ENA %-of-MLT fans, anchored to the observed level.

Outputs:
  ~/brazil_hydro/raw/fcst_rain/{model}_{date}_{hh}z.json.gz
  ~/brazil_hydro/out/fcst_verif.json
  brazil_hydro/data/ena_forecast.json

    python scripts/sst/brazil_forecast.py
"""
from __future__ import annotations

import glob
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brazil_model import basin_weights, MAJORS, TAU_SLOW   # noqa: E402
from rain_inflow_model import ema                          # noqa: E402

REPO = HERE.parent.parent
GRIB_DIR = REPO / "scripts" / "mjo" / "data" / "aifs"
PRIV = Path.home() / "brazil_hydro"
ARCH = PRIV / "raw" / "fcst_rain"
TRUTH = PRIV / "raw" / "imerg_basin_daily.json"
MODELS = PRIV / "out" / "brazil_models.json"
VERIF_JSON = PRIV / "out" / "fcst_verif.json"
OUT_JSON = REPO / "brazil_hydro" / "data" / "ena_forecast.json"

BANDS = [(1, 3), (4, 7), (8, 15)]
K_PRIOR = 6.0
PRIOR_RATIO = {"aifs": 1.0, "ifs": 0.8}    # tropical wet bias, lighter than
MIN_PAIRS = 6                              # the Andes (flatter terrain)
RES_TAU = 10.0
_W: dict = {}                              # basin masks per grid shape


def extract_cycle(model: str, date: str, hh: str) -> dict | None:
    import xarray as xr
    stem = f"{model}_{date}_{hh}z"
    parts = []
    for typ in (("cf", "pf") if model == "aifs" else ("pf",)):
        p = GRIB_DIR / f"{stem}.{typ}.tp.grib2"
        if not p.exists():
            if typ == "pf":
                return None
            continue
        ds = xr.open_dataset(p, engine="cfgrib", chunks={},
                             backend_kwargs={"filter_by_keys":
                                             {"shortName": "tp"},
                                             "indexpath": ""})
        da = ds["tp"]
        if da.attrs.get("units", "").strip() in ("m", "metre", "metres"):
            da = da * 1000.0
        if da.longitude.values.max() > 180:
            da = da.assign_coords(longitude=(da.longitude + 180) % 360 - 180)
        da = da.sortby("longitude").sortby("latitude")
        da = da.sel(longitude=slice(-76.0, -33.0), latitude=slice(-35.0, 6.0))
        if "number" not in da.dims:
            da = da.expand_dims("number")
        parts.append(da.compute())
    if not parts:
        return None
    da = parts[0] if len(parts) == 1 else __import__("xarray").concat(
        parts, dim="number")
    steps_h = (da.step.values / np.timedelta64(1, "h")).astype(int)
    order = np.argsort(steps_h)
    steps_h = steps_h[order]
    v = da.isel(step=order).transpose("number", "step", "latitude",
                                      "longitude").values
    init_dt = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{hh}:00")
    bh = np.concatenate([[0], steps_h])
    bv = np.concatenate([np.zeros((v.shape[0], 1) + v.shape[2:]), v], axis=1)
    valid, keep = [], []
    for k in range(len(bh) - 1):
        t0 = init_dt + np.timedelta64(int(bh[k]), "h")
        if bh[k + 1] - bh[k] == 24 and t0 == t0.astype("datetime64[D]"):
            valid.append(str(t0.astype("datetime64[D]")))
            keep.append(k)
    if len(valid) < 3:
        return None
    daily = np.clip(np.stack([bv[:, k + 1] - bv[:, k] for k in keep], axis=1),
                    0, None)
    key = (len(da.longitude), len(da.latitude))
    if key not in _W:
        _W[key] = basin_weights(da.longitude.values, da.latitude.values,
                                set(MAJORS))
    out = {"model": model, "init_date": date, "init_hh": hh, "valid": valid,
           "n_members": int(daily.shape[0]), "basins": {}}
    for b, w in _W[key].items():
        out["basins"][b] = np.round(
            (daily * w[None, None]).sum(axis=(2, 3)), 2).tolist()
    return out


def stage_extract() -> int:
    ARCH.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(str(GRIB_DIR / "*_*z.pf.tp.grib2"))):
        m = re.match(r"(aifs|ifs)_(\d{8})_(\d{2})z", Path(f).name)
        if not m:
            continue
        model, date, hh = m.groups()
        dest = ARCH / f"{model}_{date}_{hh}z.json.gz"
        if dest.exists():
            continue
        print(f"extracting {model} {date} {hh}Z (Brazil) …", flush=True)
        try:
            rec = extract_cycle(model, date, hh)
        except Exception as e:                  # noqa: BLE001
            print(f"  failed: {repr(e)[:100]}")
            continue
        if rec is None:
            continue
        with gzip.open(dest, "wt") as fh:
            json.dump(rec, fh, separators=(",", ":"))
        n += 1
    return n


def band_of(lead: int) -> int:
    return next((i for i, (a, b) in enumerate(BANDS) if a <= lead <= b),
                len(BANDS) - 1)


def stage_verify(tc) -> dict:
    rdates = {f"{d[:4]}-{d[4:6]}-{d[6:8]}": i
              for i, d in enumerate(tc["dates"])}
    acc = {m: {b: {bi: [0.0, 0.0, 0.0, 0] for bi in range(len(BANDS))}
               for b in MAJORS} for m in ("aifs", "ifs")}
    for f in sorted(ARCH.glob("*.json.gz")):
        rec = json.loads(gzip.open(f, "rt").read())
        mdl = rec["model"]
        d0 = np.datetime64(f"{rec['init_date'][:4]}-{rec['init_date'][4:6]}-"
                           f"{rec['init_date'][6:8]}")
        for li, vd in enumerate(rec["valid"]):
            i = rdates.get(vd)
            if i is None:
                continue
            lead = int((np.datetime64(vd) - d0).astype(int)) + 1
            bi = band_of(lead)
            for b in MAJORS:
                if b not in rec["basins"] or b not in tc:
                    continue
                obs = tc[b][i]
                fc = float(np.mean([mem[li] for mem in rec["basins"][b]]))
                a = acc[mdl][b][bi]
                a[0] += fc
                a[1] += obs
                a[2] += (fc - obs) ** 2
                a[3] += 1
    clim_mean = {b: float(np.nanmean(tc[b])) for b in MAJORS if b in tc}
    factors, weights, counts = {}, {}, {}
    for b in MAJORS:
        if b not in clim_mean:
            continue
        factors[b], weights[b], counts[b] = {}, {}, {}
        for bi in range(len(BANDS)):
            P = K_PRIOR * clim_mean[b]
            F, mse, npair = {}, {}, {}
            for mdl in ("aifs", "ifs"):
                s_f, s_o, sse, n = acc[mdl][b][bi]
                r0 = PRIOR_RATIO[mdl]
                F[mdl] = (s_o + P * r0) / (s_f + P) if s_f > 0 else r0
                mse[mdl] = sse / n if n else np.nan
                npair[mdl] = n
            if (min(npair.values()) >= MIN_PAIRS and mse["aifs"] > 0
                    and mse["ifs"] > 0):
                wa = (1 / mse["aifs"]) / (1 / mse["aifs"] + 1 / mse["ifs"])
            else:
                wa = 0.5
            factors[b][bi] = {m: round(F[m], 3) for m in F}
            weights[b][bi] = round(float(wa), 3)
            counts[b][bi] = npair
    verif = {"generated": datetime.now(timezone.utc)
             .strftime("%Y-%m-%d %H:%M UTC"),
             "truth": "raw IMERG SIN-basin daily means",
             "bands_lead_days": BANDS, "prior_ratio": PRIOR_RATIO,
             "bias_factors": factors, "weight_aifs": weights,
             "pairs": counts}
    VERIF_JSON.parent.mkdir(parents=True, exist_ok=True)
    VERIF_JSON.write_text(json.dumps(verif, indent=1))
    return verif


def weighted_quantile(vals, wts, qs):
    i = np.argsort(vals)
    v, w = np.asarray(vals)[i], np.asarray(wts)[i]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(qs, cw, v)


def stage_fan(tc, verif) -> None:
    if not MODELS.exists():
        return
    dm = json.loads(MODELS.read_text())["params"]
    latest = {}
    for mdl in ("aifs", "ifs"):
        fs = sorted(ARCH.glob(f"{mdl}_*.json.gz"))
        if fs:
            latest[mdl] = json.loads(gzip.open(fs[-1], "rt").read())
    if not latest:
        return
    last_day = np.datetime64(f"{tc['dates'][-1][:4]}-{tc['dates'][-1][4:6]}-"
                             f"{tc['dates'][-1][6:8]}")
    horizon = max(np.datetime64(rec["valid"][-1]) for rec in latest.values())
    fdays = np.arange(last_day + np.timedelta64(1, "D"),
                      horizon + np.timedelta64(1, "D"))
    if not len(fdays):
        return
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    out = {"generated": datetime.now(timezone.utc)
           .strftime("%Y-%m-%d %H:%M UTC"),
           "inits": {m: f"{r['init_date']} {r['init_hh']}Z"
                     for m, r in latest.items()},
           "dates": [str(d) for d in fdays], "basins": {}, "rain": {}}
    for b, p in dm.items():
        tau, lag = p["tau_days"], p["lag_days"]
        c0, c1, c2, c3 = p["coefs"]
        clim365 = np.array(p["clim365_mmday"], float)
        anchor = p["obs_now_pct"] - p["fit_now_pct"]
        traces, tw = [], []
        for mdl, rec in latest.items():
            if b not in rec["basins"]:
                continue
            vmap = {np.datetime64(v): i for i, v in enumerate(rec["valid"])}
            wbar = float(np.mean([verif["weight_aifs"][b][bi]
                                  for bi in verif["weight_aifs"][b]]))
            wm = (wbar if mdl == "aifs" else 1 - wbar) / rec["n_members"]
            d0 = np.datetime64(f"{rec['init_date'][:4]}-"
                               f"{rec['init_date'][4:6]}-"
                               f"{rec['init_date'][6:8]}")
            for mem in rec["basins"][b]:
                kf, ksl = p["k_now"], p["ks_now"]
                kf_h, ks_h = [kf], [ksl]
                gap = int((fdays[0] - np.datetime64(p["last_rain_day"])
                           ).astype(int)) - 1
                for _ in range(max(gap, 0)):
                    kf *= (1 - 1 / tau)
                    ksl *= (1 - 1 / TAU_SLOW)
                    kf_h.append(kf)
                    ks_h.append(ksl)
                ys = []
                for j, d in enumerate(fdays):
                    li = vmap.get(d)
                    dy = min(d.item().timetuple().tm_yday, 365) - 1
                    lead = max(int((d - d0).astype(int)), 1)
                    Fb = verif["bias_factors"][b][band_of(lead)][mdl]
                    x = (Fb * mem[li] - clim365[dy]) if li is not None else 0.0
                    kf = (1 - 1 / tau) * kf + x / tau
                    ksl = (1 - 1 / TAU_SLOW) * ksl + x / TAU_SLOW
                    kf_h.append(kf)
                    ks_h.append(ksl)
                    y = (c0 + c1 * kf_h[max(len(kf_h) - 1 - lag, 0)]
                         + c2 * ks_h[max(len(ks_h) - 1 - lag, 0)]
                         + c3 * p["ear_anom_now"]
                         + anchor * np.exp(-(j + 1.0) / RES_TAU))
                    ys.append(y)
                traces.append(ys)
                tw.append(wm)
        if traces:
            traces = np.array(traces)
            tw = np.array(tw)
            out["basins"][b] = {
                f"p{int(q*100)}": np.round(
                    [weighted_quantile(traces[:, j], tw, [q])[0]
                     for j in range(len(fdays))], 1).tolist() for q in qs}
        # bias-corrected rain quantiles (mm/day), per-day valid masking
        rr, rv, rw = [], [], []
        for mdl, rec in latest.items():
            if b not in rec["basins"]:
                continue
            vmap = {np.datetime64(v): i for i, v in enumerate(rec["valid"])}
            wbar = float(np.mean([verif["weight_aifs"][b][bi]
                                  for bi in verif["weight_aifs"][b]]))
            wm = (wbar if mdl == "aifs" else 1 - wbar) / rec["n_members"]
            d0 = np.datetime64(f"{rec['init_date'][:4]}-"
                               f"{rec['init_date'][4:6]}-"
                               f"{rec['init_date'][6:8]}")
            valid = np.array([vmap.get(d) is not None for d in fdays])
            for mem in rec["basins"][b]:
                row = np.zeros(len(fdays))
                for j, d in enumerate(fdays):
                    li = vmap.get(d)
                    if li is None:
                        continue
                    lead = max(int((d - d0).astype(int)), 1)
                    row[j] = (verif["bias_factors"][b][band_of(lead)][mdl]
                              * mem[li])
                rr.append(row)
                rv.append(valid)
                rw.append(wm)
        if rr:
            rr = np.array(rr)
            rv = np.array(rv)
            rw = np.array(rw)
            qd = {f"p{int(q*100)}": [] for q in qs}
            for j in range(len(fdays)):
                ok = rv[:, j]
                for q in qs:
                    qd[f"p{int(q*100)}"].append(
                        round(float(weighted_quantile(rr[ok, j], rw[ok],
                                                      [q])[0]), 2)
                        if ok.sum() >= 8 else None)
            out["rain"][b] = qd
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, separators=(",", ":")))
    print(f"ENA fans: {len(out['basins'])} basins, {len(fdays)} days "
          f"(inits {out['inits']})", flush=True)


def main() -> int:
    n = stage_extract()
    print(f"extract: {n} new cycle(s)")
    tc = json.loads(TRUTH.read_text())
    verif = stage_verify(tc)
    npairs = sum(c[m] for b in verif["pairs"]
                 for c in verif["pairs"][b].values() for m in c)
    print(f"verify: {npairs} matured basin-lead pairs")
    stage_fan(tc, verif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
