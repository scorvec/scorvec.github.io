#!/usr/bin/env python3
"""AIFS single vs AIFS-ENS control — spectrally fair, observation-based verification.

Successor to aifs_det_verify.py (which scored 1.5° block means against ERA5).
Two things it fixes, both raised on 2026-09-02:

  1. SAME SPECTRUM, NOT JUST SAME GRID. A 1.5° box mean still passes plenty of
     300-600 km power, so the sharper model kept variance the blurrier one had
     damped and paid the double penalty for it. Here every field is expanded in
     spherical harmonics (pyshtools), its power spectrum is kept, and BOTH
     models are truncated at the same degree LMAX_T before anything is scored.
     The spectra go to the page, so the smoothing itself is visible by lead.

  2. INDEPENDENT TRUTH. ERA5 is an IFS analysis with its own smoothness and both
     models were trained on it. Radiosondes are independent, global, and here
     within hours: 500 hPa height and 850 hPa temperature are interpolated in
     log-pressure from each station's mirrored sounding (the explorer's own
     skewt-data / skewt-archive branches). A radiosonde cannot verify 2 m
     temperature, so that stays ERA5-only. ERA5 scores are kept (from the
     truncated fields, so they are consistent) for MSLP, 2 m temperature and
     continuity.

Station anomalies (z500, t850) are against each station's OWN day-of-year
median from its full IGRA record (the explorer's skewt-climo branch, 5-day
anchors, n ≥ 30) — an observation-based baseline at the point itself. ERA5
scores use the ERA5 1991-2020 ±7-day climatology (z500, msl, t2m; none for
t850, so it gets RMSE/bias only there). ACC and the activity ratio
std(forecast anomaly)/std(observed anomaly) are computed across stations.

    python aifs_station_verify.py --collect --date YYYYMMDD --time 00|12
    python aifs_station_verify.py --verify
    python aifs_station_verify.py            # both, latest cycle

Data layout (all under scripts/verify/data, carried on the frames branch):
    st_archive/{date}{hh}.npz   per cycle: station samples, 1.5° fields, spectra
    truth_st/{valid}.npz        radiosonde truth per valid time
    truth/{valid}.npz           ERA5 truth per valid time (1.5°)
    clim/clim_1p5.npz           z500 / msl / t2m day-of-year climatology
Scores: assets/verify/aifs_scores_v2.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
DATA = HERE / "data"
ST_ARCHIVE = DATA / "st_archive"
TRUTH_ST = DATA / "truth_st"
TRUTH = DATA / "truth"
CLIM = DATA / "clim" / "clim_1p5.npz"
CLIM_LEGACY = DATA / "clim_1p5.npz"
UW_CACHE = DATA / "uw_cache"
SCORES = REPO / "assets" / "verify" / "aifs_scores_v2.json"
STATIONS_JSON = REPO / "skewt" / "stations.json"
UW_ARCHIVE = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-archive/uw-{d}.zip"
UW_LIVE = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-data/soundings/{sid}_{tag}.csv"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

LEADS = list(range(24, 241, 24))
G = 9.80665
LMAX_T = 120                      # common truncation: T120 ≈ 1.5° (matches the legacy grid)
MODELS = {"single": ("aifs-single", "fc"), "control": ("aifs-ens", "cf")}
VARS = ("z500", "t850", "msl", "t2m")
RAOB_VARS = ("z500", "t850")      # what a radiosonde can verify (2 m temperature it cannot)
CLIMO_ST = DATA / "clim" / "station_climo.npz"
CLIMO_URL = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-climo/climo/{gid}.json"
KEEP_CYCLES = 400                 # archive retention on the branch (~200 days of 2 cycles)
GRID_DAYS = 12                    # keep the 1.5° grids this long (ERA5 lags ~6 d), then strip them


# ─────────────────────────────────────────────────────────── grids & helpers
def coarsen(a):
    """0.25° (720×1440, 90→-89.75) → 1.5° block means (120×240)."""
    return a.reshape(120, 6, 240, 6).mean(axis=(1, 3))


def grid_1p5():
    lat = 90.0 - 0.25 * (np.arange(720).reshape(120, 6).mean(axis=1))
    lon = 0.25 * (np.arange(1440).reshape(240, 6).mean(axis=1))
    return lat, lon


def to_dh(da):
    """xarray field → (720×1440) float64 array on the pyshtools DH grid:
    latitude 90 → -89.75 (drop the -90 row), longitude 0 → 359.75."""
    v = da.transpose("latitude", "longitude").values.astype(np.float64)
    lat = da.latitude.values
    lon = da.longitude.values
    if lat[0] < lat[-1]:
        v = v[::-1]; lat = lat[::-1]
    if lon.min() < 0:                                     # -180..180 → 0..360
        k = int(np.argmin(np.abs(lon)))
        v = np.roll(v, -k, axis=1)
    if v.shape[0] == 721:
        v = v[:720]
    assert v.shape == (720, 1440), v.shape
    return v


def spectrum_and_truncate(v, lmax_t=LMAX_T):
    """Power spectrum per degree (0..359) and the field truncated at lmax_t,
    back on the same 720×1440 grid."""
    import pyshtools as sh
    g = sh.SHGrid.from_array(v, grid="DH")
    c = g.expand()
    spec = c.spectrum().astype(np.float32)
    c.coeffs[:, lmax_t + 1:, :] = 0.0
    vt = c.expand(grid="DH2", extend=False).to_array()          # 720×1440, no -90 row / 360° column
    return spec, vt


def sample_points(v, lats, lons):
    """Bilinear sample of a DH-grid field at station lat/lon (deg, lon any range)."""
    from scipy.interpolate import RegularGridInterpolator
    glat = 90.0 - 0.25 * np.arange(720)                  # descending
    glon = 0.25 * np.arange(1441)                        # wrap column for 360
    vv = np.concatenate([v, v[:, :1]], axis=1)
    f = RegularGridInterpolator((glat[::-1], glon), vv[::-1], bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([np.clip(lats, -89.75, 90.0), np.mod(lons, 360.0)])
    return f(pts).astype(np.float32)


def load_stations():
    """Active radiosonde stations with a WMO id: id → (lat, lon)."""
    st = json.loads(STATIONS_JSON.read_text())["stations"]
    y = pd.Timestamp.utcnow().year - 1
    out = {s["id"]: (float(s["la"]), float(s["lo"])) for s in st
           if s.get("id") and s.get("y1", 0) >= y and np.isfinite(s.get("la", np.nan))}
    ids = sorted(out)
    lats = np.array([out[i][0] for i in ids]); lons = np.array([out[i][1] for i in ids])
    return ids, lats, lons


# ──────────────────────────────────────────────────────────────── collect
def collect(date: str, hh: str) -> bool:
    import store as ecmwf
    out = ST_ARCHIVE / f"{date}{hh}.npz"
    if out.exists():
        print(f"{date} {hh}Z: archived"); return True
    ST_ARCHIVE.mkdir(parents=True, exist_ok=True)
    ids, lats, lons = load_stations()
    cyc = ecmwf.Cycle(date, hh)
    S = tuple(LEADS)
    stash, specs = {}, {}
    for mkey, (model, typ) in MODELS.items():
        try:
            ppl = ecmwf.ensure(cyc, ecmwf.Spec(model, typ, "z", "pl", (500,), S))
            pt8 = ecmwf.ensure(cyc, ecmwf.Spec(model, typ, "t", "pl", (850,), S))
            pms = ecmwf.ensure(cyc, ecmwf.Spec(model, typ, "msl", "sfc", (), S))
            p2t = ecmwf.ensure(cyc, ecmwf.Spec(model, typ, "2t", "sfc", (), S))
        except Exception as e:                            # noqa: BLE001
            print(f"{date} {mkey}: fetch failed ({str(e)[:80]})", file=sys.stderr)
            return False
        kw = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
        dz = xr.open_dataset(ppl, **kw); dm = xr.open_dataset(pms, **kw); dt = xr.open_dataset(p2t, **kw)
        d8 = xr.open_dataset(pt8, **kw)
        t0 = time.time()
        for step in LEADS:
            sd = pd.Timedelta(hours=step)
            zsel = dz["z"].sel(step=sd)
            if "isobaricInhPa" in zsel.dims:                  # scalar coord when one level was fetched
                zsel = zsel.sel(isobaricInhPa=500)
            tsel = d8["t"].sel(step=sd)
            if "isobaricInhPa" in tsel.dims:
                tsel = tsel.sel(isobaricInhPa=850)
            fields = {
                "z500": to_dh(zsel) / G,
                "t850": to_dh(tsel) - 273.15,
                "msl": to_dh(dm["msl"].sel(step=sd)) / 100.0,
                "t2m": to_dh(dt["t2m"].sel(step=sd)) - 273.15,
            }
            for var, v in fields.items():
                spec, vt = spectrum_and_truncate(v)
                specs[(mkey, var, step)] = spec
                stash[(mkey, var, step)] = (sample_points(vt, lats, lons),
                                            coarsen(vt).astype(np.float32),
                                            coarsen(v).astype(np.float32))
        dz.close(); dm.close(); dt.close(); d8.close()
        print(f"{date} {hh}Z {mkey}: {len(LEADS)} leads, spectra + T{LMAX_T} in {time.time() - t0:.0f} s",
              flush=True)
    payload = {"ids": np.array(ids), "lats": lats.astype(np.float32), "lons": lons.astype(np.float32),
               "lmax_t": np.int32(LMAX_T)}
    for (m, v, st), (pts, g_t, g_raw) in stash.items():
        payload[f"st_{m}_{v}_{st}"] = pts
        payload[f"gt_{m}_{v}_{st}"] = g_t          # truncated, 1.5°
        payload[f"gr_{m}_{v}_{st}"] = g_raw        # untruncated block mean (legacy method)
    for (m, v, st), sp in specs.items():
        payload[f"spec_{m}_{v}_{st}"] = sp
    np.savez_compressed(out, **payload)
    return True


# ────────────────────────────────────────────────────────────── truth: RAOB
def _interp_logp(P, X, p0, lo, hi):
    """X at pressure p0 by log-p interpolation between the bracketing levels
    (P sorted descending); NaN unless the brackets sit within (lo, hi) hPa."""
    hit = np.where(np.abs(P - p0) < 0.6)[0]
    if hit.size and np.isfinite(X[hit[0]]):
        return float(X[hit[0]])
    ok = np.isfinite(X)
    below = np.where((P > p0) & ok)[0]; above = np.where((P < p0) & ok)[0]
    if not (below.size and above.size):
        return np.nan
    i, j = below[-1], above[0]
    if not (lo > P[i] and P[j] > hi):
        return np.nan
    f = (np.log(P[i]) - np.log(p0)) / (np.log(P[i]) - np.log(P[j]))
    return float(X[i] + f * (X[j] - X[i]))


def _z500_from_csv(text):
    """(z500 m, t850 °C, surface T °C, surface p hPa) from a UW TEXT:CSV sounding."""
    rows = text.strip().split("\n")
    if not rows or not rows[0].startswith("time"):
        return None
    P, Z, T = [], [], []
    for r in rows[1:]:
        c = r.split(",")
        if len(c) < 7:
            continue
        try:
            p, z, t = float(c[3]), float(c[4]), float(c[5])
        except ValueError:
            continue
        if not (np.isfinite(p) and np.isfinite(z)) or p <= 0:
            continue
        P.append(p); Z.append(z); T.append(t if np.isfinite(t) else np.nan)
    if len(P) < 5:
        return None
    P = np.array(P); Z = np.array(Z); T = np.array(T)
    order = np.argsort(-P); P, Z, T = P[order], Z[order], T[order]
    sfc_t, sfc_p = T[0], P[0]
    z500 = _interp_logp(P, Z, 500.0, 700.0, 350.0)
    if not (4500 < z500 < 6200):
        z500 = np.nan
    t850 = _interp_logp(P, T, 850.0, 1000.0, 700.0) if sfc_p > 860 else np.nan   # below ground otherwise
    if not (-45 < t850 < 40):
        t850 = np.nan
    return z500, t850, sfc_t, sfc_p


def _http(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "scorvec.com aifs verification"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def truth_raob(valid: pd.Timestamp) -> bool:
    """Radiosonde truth at `valid` (00/12Z) from the explorer's day bundle
    (skewt-archive, permanent) with the live mirror as a fallback for today."""
    out = TRUTH_ST / f"{valid:%Y%m%d%H}.npz"
    if out.exists():
        return True
    TRUTH_ST.mkdir(parents=True, exist_ok=True); UW_CACHE.mkdir(parents=True, exist_ok=True)
    dkey = f"{valid:%Y%m%d}"; tag = f"{valid:%Y%m%d%H}"
    zpath = UW_CACHE / f"uw-{dkey}.zip"
    files = {}
    try:
        if not zpath.exists():
            zpath.write_bytes(_http(UW_ARCHIVE.format(d=dkey)))
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                base = name.rsplit("/", 1)[-1]
                if base.endswith(f"_{tag}.csv"):
                    files[base.split("_")[0]] = zf.read(name).decode("utf-8", "ignore")
    except Exception as e:                                # noqa: BLE001
        zpath.unlink(missing_ok=True)
        print(f"  raob {tag}: day bundle unavailable ({str(e)[:60]}); trying the live mirror", flush=True)
    if not files:
        # today's bundle is written after the day closes: use the live mirror
        try:
            man = json.loads(_http("https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-data/manifest.json"))
            want = f"{valid:%Y-%m-%d %H}:00"
            for sid, e in man.get("entries", {}).items():
                if want in (e.get("hours") or []):
                    try:
                        files[sid] = _http(UW_LIVE.format(sid=sid, tag=tag), timeout=30).decode("utf-8", "ignore")
                    except Exception:                     # noqa: BLE001
                        pass
        except Exception as e:                            # noqa: BLE001
            print(f"  raob {tag}: live mirror failed ({str(e)[:60]})", flush=True)
    if len(files) < 50:
        print(f"  raob {tag}: only {len(files)} soundings — not enough yet", flush=True)
        return False
    ids, z, t8, t, p = [], [], [], [], []
    for sid, text in files.items():
        r = _z500_from_csv(text)
        if r is None:
            continue
        ids.append(sid); z.append(r[0]); t8.append(r[1]); t.append(r[2]); p.append(r[3])
    np.savez_compressed(out, ids=np.array(ids), z500=np.array(z, np.float32),
                        t850=np.array(t8, np.float32), tsfc=np.array(t, np.float32),
                        psfc=np.array(p, np.float32))
    print(f"  raob {tag}: {len(ids)} stations, {int(np.isfinite(z).sum())} with z500, "
          f"{int(np.isfinite(t8).sum())} with t850", flush=True)
    return True


# ────────────────────────────────────────────────────────────── truth: ERA5
def truth_era5(valid: pd.Timestamp) -> bool:
    out = TRUTH / f"{valid:%Y%m%d%H}.npz"
    if out.exists():
        d = np.load(out)
        if "t2m" in d.files and "t850" in d.files:
            return True
    TRUTH.mkdir(parents=True, exist_ok=True)
    try:
        ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
        t = valid.to_datetime64()
        z = ds["geopotential"].sel(time=t, level=500).values
        if not np.isfinite(z).all():
            return False
        m = ds["mean_sea_level_pressure"].sel(time=t).values
        t2 = ds["2m_temperature"].sel(time=t).values
        t8 = ds["temperature"].sel(time=t, level=850).values
    except Exception as e:                                # noqa: BLE001
        print(f"  era5 {valid:%Y-%m-%d %HZ}: {str(e)[:70]}", file=sys.stderr)
        return False
    np.savez_compressed(out, z500=coarsen(z[:720]).astype(np.float32) / G,
                        msl=coarsen(m[:720]).astype(np.float32) / 100.0,
                        t2m=coarsen(t2[:720]).astype(np.float32) - 273.15,
                        t850=coarsen(t8[:720]).astype(np.float32) - 273.15)
    print(f"  era5 {valid:%Y-%m-%d %HZ} ✓", flush=True)
    return True


# ────────────────────────────────────────────────────────────────── scoring
CLIM_T2M_6H = DATA / "clim" / "clim_1p5_t2m6h.npz"


def load_clim():
    p = CLIM if CLIM.exists() else CLIM_LEGACY
    if not p.exists():
        return {}
    d = np.load(p)
    out = {k: d[k].astype(np.float32) for k in d.files}
    # hour-of-day 2 m normals (00/06/12/18Z, NH, K): a 12Z field scored
    # against a daily-mean normal carries the diurnal cycle in its anomaly
    if CLIM_T2M_6H.exists():
        h = np.load(CLIM_T2M_6H)
        for k in h.files:
            if k.startswith("t2m_h"):
                full = np.full((366, 120, 240), np.nan, np.float32)
                full[:, :h[k].shape[1]] = h[k].astype(np.float32) - 273.15
                out[k] = full
    if "t2m" in out and np.nanmean(out["t2m"]) > 100:      # stored in K → °C
        out["t2m"] = out["t2m"] - 273.15
    for k, c in out.items():                              # WB2 ends at 358.5E: the last
        if c.ndim == 3 and np.isnan(c[:, :, -1]).all():   # 1.5° column was never filled,
            c[:, :, -1] = 0.5 * (c[:, :, -2] + c[:, :, 0])   # which NaN'd stations near 0E
    return out


def station_climo(ids):
    """Per-station day-of-year MEDIAN of h500 and 850 T from each station's
    full IGRA record (skewt-climo branch: 73 five-day anchors, percentile list
    [1,5,10,25,50,75,90,95,99], n per anchor). Built once (~900 small fetches)
    and cached; stations without a climatology get NaN → no anomaly scores."""
    ids = list(ids)
    if CLIMO_ST.exists():
        d = np.load(CLIMO_ST, allow_pickle=False)
        if list(d["ids"]) == ids:
            return {"z500": d["z500"], "t850": d["t850"]}
    st = json.loads(STATIONS_JSON.read_text())["stations"]
    gid = {s["id"]: s["gid"] for s in st if s.get("id")}
    z = np.full((len(ids), 73), np.nan, np.float32); t = np.full((len(ids), 73), np.nan, np.float32)
    got = 0
    for i, sid in enumerate(ids):
        g = gid.get(sid)
        if not g:
            continue
        try:
            c = json.loads(_http(CLIMO_URL.format(gid=g), timeout=30))
        except Exception:                                 # noqa: BLE001
            continue
        for var, key, arr in (("z500", "h500", z), ("t850", "850t", t)):
            a = (c.get("idx") or {}).get(key)
            if not a:
                continue
            nn = a.get("n") or [0] * 73
            for k in range(min(73, len(a.get("p", [])))):
                if a["p"][k] and a["p"][k][4] is not None and nn[k] >= 30:
                    arr[i, k] = a["p"][k][4]
        got += 1
    CLIMO_ST.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CLIMO_ST, ids=np.array(ids), z500=z, t850=t)
    print(f"  station climatology: {got}/{len(ids)} stations", flush=True)
    return {"z500": z, "t850": t}


def station_clim_at(sc, var, doy):
    """Nearest 5-day anchor (anchors at doy 1, 6, 11, …, 361)."""
    k = int(round((min(doy, 366) - 1) / 5.0)) % 73
    return sc[var][:, k].astype(float)


def clim_at_points(clim, var, doy, lats, lons):
    from scipy.interpolate import RegularGridInterpolator
    c = clim.get(var)
    if c is None:
        return None
    lat, lon = grid_1p5()
    field = c[min(doy, 366) - 1]
    ff = np.concatenate([field, field[:, :1]], axis=1)
    lon2 = np.concatenate([lon, [lon[0] + 360.0]])
    f = RegularGridInterpolator((lat[::-1], lon2), ff[::-1], bounds_error=False, fill_value=np.nan)
    return f(np.column_stack([np.clip(lats, lat.min(), lat.max()), np.mod(lons, 360.0)]))


def _scores(f, o, fa=None, oa=None, w=None):
    m = np.isfinite(f) & np.isfinite(o)
    if fa is not None:
        m &= np.isfinite(fa) & np.isfinite(oa)
    n = int(m.sum())
    if n < 20:
        return None
    ww = np.ones(n) if w is None else w[m]
    e = f[m] - o[m]
    rec = {"n": n,
           "rmse": round(float(np.sqrt(np.sum(ww * e**2) / ww.sum())), 3),
           "bias": round(float(np.sum(ww * e) / ww.sum()), 3)}
    if fa is not None:
        a, b = fa[m], oa[m]
        a = a - np.sum(ww * a) / ww.sum(); b = b - np.sum(ww * b) / ww.sum()
        den = np.sqrt(np.sum(ww * a**2) * np.sum(ww * b**2))
        if den > 0:
            rec["acc"] = round(float(np.sum(ww * a * b) / den), 4)
            rec["act"] = round(float(np.sqrt(np.sum(ww * a**2) / np.sum(ww * b**2))), 3)
    return rec


def verify() -> int:
    clim = load_clim()
    lat, lon = grid_1p5()
    band = (lat >= 20) & (lat <= 80)
    wg = (np.cos(np.deg2rad(lat[band]))[:, None] * np.ones((band.sum(), len(lon))))
    recs, spectra = [], {}
    sclim = None
    if SCORES.exists():
        old = json.loads(SCORES.read_text())
        recs = old.get("records", []); spectra = old.get("spectra", {})
    done = {(r["init"], r["lead"], r["var"], r["model"], r["truth"], r["region"]) for r in recs}
    n_new = 0
    for arch in sorted(ST_ARCHIVE.glob("*.npz"))[-KEEP_CYCLES:]:
        init = pd.Timestamp(arch.stem[:8]) + pd.Timedelta(hours=int(arch.stem[8:10]))
        A = None
        # spectra first: they exist for every lead the moment the cycle is
        # archived, unlike scores, which wait for the valid time and its truth
        for lead in LEADS:
            for var in VARS:
                for mkey in MODELS:
                    k = f"{mkey}_{var}_{lead}"
                    ent = spectra.get(k)
                    if ent and arch.stem in ent.get("inits", []):
                        continue
                    if A is None:
                        A = np.load(arch, allow_pickle=False)
                        ids = list(A["ids"]); lats = A["lats"].astype(float); lons = A["lons"].astype(float)
                    if f"spec_{k}" not in A.files:
                        continue
                    sp = A[f"spec_{k}"].astype(float)
                    ent = ent or {"n": 0, "mean": [0.0] * len(sp), "inits": []}
                    n = ent["n"]; mean = np.array(ent["mean"])
                    mean = (mean * n + sp) / (n + 1)
                    spectra[k] = {"n": n + 1, "mean": [float(f"{x:.6g}") for x in mean],
                                  "last": [float(f"{x:.6g}") for x in sp], "last_init": arch.stem,
                                  "inits": (ent["inits"] + [arch.stem])[-KEEP_CYCLES:]}
                    n_new += 1
        for lead in LEADS:
            valid = init + pd.Timedelta(hours=lead)
            if valid > pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=3):
                continue
            key = (arch.stem, lead, "z500", "single", "raob", "nh")
            if key in done and (arch.stem, lead, "z500", "single", "era5", "nh") in done:
                continue
            if A is None:
                A = np.load(arch, allow_pickle=False)
                ids = list(A["ids"]); lats = A["lats"].astype(float); lons = A["lons"].astype(float)
            doy = min(valid.dayofyear, 366)
            # ── radiosonde truth ──
            if key not in done and truth_raob(valid):
                T = np.load(TRUTH_ST / f"{valid:%Y%m%d%H}.npz", allow_pickle=False)
                tid = list(T["ids"]); pos = {s: i for i, s in enumerate(tid)}
                sel = np.array([pos.get(s, -1) for s in ids])
                has = sel >= 0
                if sclim is None:
                    sclim = station_climo(ids)
                for var in RAOB_VARS:
                    if var not in T.files or f"st_single_{var}_{lead}" not in A.files:
                        continue
                    o = np.full(len(ids), np.nan, np.float32); o[has] = T[var][sel[has]]
                    ca = station_clim_at(sclim, var, doy)
                    oa = o - ca
                    for region, msk in (("nh", (lats >= 20) & (lats <= 80)), ("glb", np.ones(len(ids), bool))):
                        for mkey in MODELS:
                            f = A[f"st_{mkey}_{var}_{lead}"].astype(float)
                            fa = f - ca
                            fm = f.copy(); om = o.copy(); fm[~msk] = np.nan; om[~msk] = np.nan
                            fam = None if fa is None else np.where(msk, fa, np.nan)
                            oam = None if oa is None else np.where(msk, oa, np.nan)
                            sc = _scores(fm, om, fam, oam)
                            if sc is None:
                                continue
                            recs.append(dict(init=arch.stem, lead=lead, var=var, model=mkey,
                                             truth="raob", region=region, **sc)); n_new += 1
            # ── ERA5 truth (1.5°, NH extratropics), truncated and legacy fields ──
            ekey = (arch.stem, lead, "z500", "single", "era5", "nh")
            if ekey not in done and truth_era5(valid):
                E = np.load(TRUTH / f"{valid:%Y%m%d%H}.npz")
                for var in VARS:
                    if var not in E.files or f"gt_single_{var}_{lead}" not in A.files:
                        continue
                    o = E[var][band]
                    cv = clim.get(var)
                    if var == "t2m":                                   # hour-matched when built
                        cv = clim.get(f"t2m_h{valid.hour:02d}", cv)
                    ca = None if cv is None else cv[doy - 1][band]
                    oa = None if ca is None else o - ca
                    for mkey in MODELS:
                        for method, pref in (("era5", "gt"), ("era5-block", "gr")):
                            f = A[f"{pref}_{mkey}_{var}_{lead}"][band]
                            fa = None if ca is None else f - ca
                            sc = _scores(f.ravel(), o.ravel(),
                                         None if fa is None else fa.ravel(),
                                         None if oa is None else oa.ravel(), wg.ravel())
                            if sc is None:
                                continue
                            recs.append(dict(init=arch.stem, lead=lead, var=var, model=mkey,
                                             truth=method, region="nh", **sc)); n_new += 1
    if n_new:
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        SCORES.write_text(json.dumps(
            {"generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
             "method": f"spherical-harmonic truncation of both models at T{LMAX_T} before scoring",
             "truths": {"raob": "radiosondes (explorer mirror): z500 and 850 hPa T, log-p interpolated",
                        "era5": "ERA5 (ARCO) 1.5° block means of the truncated fields, NH 20-80N",
                        "era5-block": "legacy: 1.5° block means of the raw 0.25° fields"},
             "acc_base": {"raob": "each station's own IGRA day-of-year median (skewt-climo)",
                          "era5": "ERA5 1991-2020 ±7d day-of-year climatology (WB2), NH; none for t850"},
             "lmax_t": LMAX_T, "leads": LEADS,
             "records": recs, "spectra": spectra}, separators=(",", ":")))
        print(f"scores: +{n_new} records → {len(recs)} total; spectra keys {len(spectra)}")
    else:
        print("no newly verifiable cycles")
    # retention on the branch: the 1.5° grids (10 MB/cycle) are only needed
    # until ERA5 has caught up (~6 days), so archives older than GRID_DAYS are
    # rewritten with just the station samples and spectra (~0.2 MB); whole
    # cycles beyond KEEP_CYCLES are dropped.
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=GRID_DAYS)
    for f in sorted(ST_ARCHIVE.glob("*.npz")):
        init = pd.Timestamp(f.stem[:8]) + pd.Timedelta(hours=int(f.stem[8:10]))
        if init > cutoff:
            continue
        d = np.load(f, allow_pickle=False)
        if not any(k.startswith(("gt_", "gr_")) for k in d.files):
            continue
        np.savez_compressed(f, **{k: d[k] for k in d.files if not k.startswith(("gt_", "gr_"))})
    for old in sorted(ST_ARCHIVE.glob("*.npz"))[:-KEEP_CYCLES]:
        old.unlink()
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--date", default=pd.Timestamp.utcnow().strftime("%Y%m%d"))
    ap.add_argument("--time", default="00", choices=("00", "12"))
    a = ap.parse_args()
    if a.collect:
        collect(a.date, a.time)
    if a.verify:
        verify()
    if not (a.collect or a.verify):
        collect(a.date, a.time); verify()


if __name__ == "__main__":
    main()
