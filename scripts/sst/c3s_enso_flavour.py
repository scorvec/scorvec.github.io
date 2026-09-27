#!/usr/bin/env python3
"""How east-based will the El Niño be? C3S members projected onto Takahashi's E and C modes.

2026-09-27, user: "let's re-do the enso forecasts page ... We should also add a measure of how
east-based it will be". Writes assets/sst/data/enso_flavour.json for enso-forecasts.html.

THE INDICES (Takahashi, Montecinos, Goubanova & Dewitte 2011, GRL 38, L10704)
  EOF1 and EOF2 of monthly tropical Pacific SST anomalies, 10°S–10°N, 140°E–80°W, from NOAA
  ERSSTv6 on its native 2° grid, anomalies against a FIXED 1991–2020 monthly climatology (no
  detrending), sqrt(cos lat) weighted, EOFs trained on 1979–2020. PCs are standardised over the
  training period. EOF1 is the basin-wide warming (sign: warm over Niño-3.4); EOF2 the east–west
  dipole (sign: warm in Niño-4, cold in Niño-1+2, as Takahashi). Then
      E = (PC1 − PC2)/√2    eastern-Pacific (coastal, "canonical") warming
      C = (PC1 + PC2)/√2    central-Pacific warming
  both in units of their own 1979–2020 standard deviation. Every other month since 1950 is
  PROJECTED onto the same EOFs and scaled by the same training σ.

  Also, for readers who think in boxes: Niño-1+2 (0–10°S, 90–80°W), Niño-3 (5°N–5°S, 150–90°W),
  Niño-3.4 (5°N–5°S, 170–120°W), Niño-4 (5°N–5°S, 160°E–150°W), and the "centre of the warming":
  the longitude of the largest equatorial (2°S–2°N) anomaly after a 10°-wide running mean along
  the equator, searched between 150°E and 80°W.

THE FORECAST
  C3S `seasonal-postprocessed-single-levels`, product_type monthly_mean, variable
  sea_surface_temperature_anomaly: every MEMBER's anomaly already taken against its own model's
  1993–2016 hindcast climate — the same construction c3s_nino34.py does by hand for the plume,
  which it reproduces to the third decimal (checked 2026-09-27: ECMWF Niño-3.4 per member), so no
  hindcast download at all. One strip per system, 10°S–10°N × 120°E–70°W at 1° (~2–8 MB). The
  anomalies are moved from the models' 1993–2016 base to the observed 1991–2020 base with the
  observed difference of the two climatologies (per calendar month and grid point, a few
  hundredths of a degree), so the forecast sits on the same ruler as the past events. The strip is
  sampled at the ERSST 2° points and projected onto the observed EOFs.

  ECCC's two components (GEM5-NEMO + CanESM5) are merged into one CanSIPS, as in the plume.

Runs on the LAPTOP (the Copernicus key stays there), monthly, from run_c3s_nino34.sh.

    python scripts/sst/c3s_enso_flavour.py                 # issue from enso_forecast.json
    python scripts/sst/c3s_enso_flavour.py --issue 202609
    python scripts/sst/c3s_enso_flavour.py --fetch-only --issue 202608
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c3s_nino34 import MODELS, MERGE, _client          # one model registry, one CDS client  # noqa: E402

SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE.parents[1]
ASSETS = SITE_ROOT / "assets" / "sst"
OUT = ASSETS / "data" / "enso_flavour.json"
FC_JSON = ASSETS / "data" / "enso_forecast.json"
HIST_JSON = ASSETS / "data" / "nino_history.json"
ERSST = HERE / "data" / "ersst_v6_mnmean.nc"
ERSST_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v6/sst.mnmean.nc"
CACHE = HERE / "data" / "c3s" / "strip"

DATASET = "seasonal-postprocessed-single-levels"
AREA = [10, 120, -10, -70]                 # N, W, S, E  → 120°E … 290°E (70°W)
GRID = [1.0, 1.0]
LEADS = ["1", "2", "3", "4", "5", "6"]

EOF_LAT, EOF_LON = (-10.0, 10.0), (140.0, 280.0)
TRAIN = ("1979-01", "2020-12")
BASE = (1991, 2020)
HINDCAST_BASE = (1993, 2016)
BOXES = {                                   # (lat range, lon range in °E)
    "n12": ((-10, 0), (270, 280)),
    "n3":  ((-5, 5), (210, 270)),
    "n34": ((-5, 5), (190, 240)),
    "n4":  ((-5, 5), (160, 210)),
}
EQ_LAT = (-2, 2)
CENTRE_SEARCH = (150.0, 280.0)
SMOOTH_DEG = 10.0
SEASONS = {"SON": (9, 10, 11), "OND": (10, 11, 12), "NDJ": (11, 12, 1), "DJF": (12, 1, 2)}


# ─────────────────────────────────────────────────────────────────────────── C3S strip
def strip_path(centre: str, system: str, issue: str) -> Path:
    return CACHE / issue / f"{centre}_{system}.grib"


def fetch_strip(centre: str, system: str, issue: str) -> Path | None:
    dest = strip_path(centre, system, issue)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = {
        "originating_centre": centre, "system": system,
        "variable": ["sea_surface_temperature_anomaly"], "product_type": ["monthly_mean"],
        "year": [issue[:4]], "month": [issue[4:6]], "leadtime_month": LEADS,
        "area": AREA, "grid": GRID, "data_format": "grib",
    }
    tmp = dest.with_suffix(".part")
    for attempt in range(2):
        try:
            _client().retrieve(DATASET, req, str(tmp))
            os.replace(tmp, dest)
            print(f"  {centre}/{system} {issue}: {dest.stat().st_size / 1e6:.1f} MB", flush=True)
            return dest
        except Exception as e:                                        # noqa: BLE001
            msg = str(e).replace("\n", " ")
            tmp.unlink(missing_ok=True)
            if "no data" in msg.lower():
                print(f"  {centre}/{system} {issue}: no data on the CDS", file=sys.stderr, flush=True)
                return None
            print(f"  {centre}/{system} {issue}: attempt {attempt + 1} failed ({msg[:100]})",
                  file=sys.stderr, flush=True)
            time.sleep(8)
    return None


def fetch_all(issue: str) -> dict:
    """{(centre, system): path} for every registry system that has this issue."""
    got = {}
    for centre, system, label, _c in MODELS:
        p = fetch_strip(centre, system, issue)
        if p is not None:
            got[(centre, system)] = p
    return got




def load_strip(path: Path) -> xr.DataArray:
    """Member anomalies (member, lead, lat, lon) in °C, lat ascending, lon 120..290 °E.

    Read message by message with eccodes rather than through cfgrib's hypercube: a lagged
    ensemble (UKMO, NCEP: 31 start dates, steps in days) becomes a mostly-empty
    number × start × step cube that way. Each GRIB message carries its own `forecastMonth`
    (lead 1 = the issue month) and start date, so a member is one (number, start date) pair."""
    import eccodes
    fields, grid = {}, None
    with open(path, "rb") as fh:
        while True:
            h = eccodes.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                if grid is None:
                    grid = dict(ni=eccodes.codes_get(h, "Ni"), nj=eccodes.codes_get(h, "Nj"),
                                la1=eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
                                la2=eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees"),
                                lo1=eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees"),
                                lo2=eccodes.codes_get(h, "longitudeOfLastGridPointInDegrees"),
                                miss=eccodes.codes_get(h, "missingValue"))
                key = (int(eccodes.codes_get(h, "dataDate")), int(eccodes.codes_get(h, "number")))
                lead = int(eccodes.codes_get(h, "forecastMonth"))
                v = eccodes.codes_get_values(h).astype("float64")
                v[v == grid["miss"]] = np.nan
                fields.setdefault(key, {})[lead] = v.reshape(grid["nj"], grid["ni"])
            finally:
                eccodes.codes_release(h)
    nl = len(LEADS)
    keys = sorted(k for k, d in fields.items() if all(L in d for L in range(1, nl + 1)))
    if not keys:
        raise ValueError(f"{path.name}: no member has all {nl} leads")
    x = np.stack([np.stack([fields[k][L] for L in range(1, nl + 1)]) for k in keys])
    lat = np.linspace(grid["la1"], grid["la2"], grid["nj"])
    lo2 = grid["lo2"] if grid["lo2"] > grid["lo1"] else grid["lo2"] + 360
    lon = np.linspace(grid["lo1"], lo2, grid["ni"]) % 360
    da = xr.DataArray(x, dims=("member", "lead", "latitude", "longitude"),
                      coords=dict(member=np.arange(len(keys)), lead=np.arange(1, nl + 1),
                                  latitude=np.round(lat, 3), longitude=np.round(lon, 3)))
    return da.sortby("latitude")


# ─────────────────────────────────────────────────────────────────────────── observed
def _ym(t) -> str:
    return f"{t.year:04d}-{t.month:02d}"


def load_obs() -> dict:
    """ERSSTv6 over the EOF box, 1950 → latest, anomalies against the fixed 1991–2020 base.
    Also the 1993–2016 climatology, for moving the models onto that base."""
    if not ERSST.exists():
        from urllib.request import urlretrieve
        ERSST.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(ERSST_URL, ERSST)                                  # noqa: S310
    ds = xr.open_dataset(ERSST)
    sst = (ds["sst"].sortby("lat")
           .sel(lat=slice(*EOF_LAT), lon=slice(*EOF_LON), time=slice("1950-01-01", None)).load())
    ds.close()
    t = pd.DatetimeIndex(sst["time"].values)
    x = sst.values.astype("float64")
    mo, yr = t.month.values, t.year.values

    def clim(y0, y1):
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.stack([np.nanmean(x[(mo == m) & (yr >= y0) & (yr <= y1)], 0) for m in range(1, 13)])

    c91, c93 = clim(*BASE), clim(*HINDCAST_BASE)
    return dict(lat=sst["lat"].values.astype(float), lon=sst["lon"].values.astype(float),
                months=[_ym(v) for v in t], a=x - c91[mo - 1], c91=c91, c93=c93)


def fit_eofs(obs: dict) -> dict:
    lat, lon, a, months = obs["lat"], obs["lon"], obs["a"], obs["months"]
    valid = np.isfinite(a).all(0)
    w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones_like(lon)[None, :]
    tr = np.array([TRAIN[0] <= m <= TRAIN[1] for m in months])
    X = a[tr][:, valid] * w[valid][None]
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    e1, e2 = Vt[0].copy(), Vt[1].copy()

    def as_map(v):
        P = np.full(valid.shape, np.nan)
        P[valid] = v / w[valid]
        return P

    b = {k: _box_mask(lat, lon, *BOXES[k]) for k in ("n34", "n4", "n12")}
    if np.nanmean(as_map(e1)[b["n34"]]) < 0:
        e1 = -e1
    P2 = as_map(e2)
    if np.nanmean(P2[b["n4"]]) - np.nanmean(P2[b["n12"]]) < 0:          # warm Niño-4, cold Niño-1+2
        e2 = -e2
    pc1, pc2 = X @ e1, X @ e2
    s1, s2 = float(pc1.std()), float(pc2.std())
    return dict(valid=valid, w=w, e1=e1, e2=e2, s1=s1, s2=s2, var=var[:4],
                pat1=as_map(e1) * s1, pat2=as_map(e2) * s2)          # °C per standard deviation


def project(field: np.ndarray, eof: dict) -> tuple[np.ndarray, np.ndarray]:
    """(..., lat, lon) anomalies on the EOF grid → (E, C). NaN at an EOF cell → that sample is NaN."""
    X = field[..., eof["valid"]] * eof["w"][eof["valid"]]
    pc1 = (X @ eof["e1"]) / eof["s1"]
    pc2 = (X @ eof["e2"]) / eof["s2"]
    return (pc1 - pc2) / np.sqrt(2), (pc1 + pc2) / np.sqrt(2)


def _box_mask(lat, lon, la, lo):
    return ((lat >= la[0]) & (lat <= la[1]))[:, None] & ((lon >= lo[0]) & (lon <= lo[1]))[None, :]


def box_means(field: np.ndarray, lat, lon) -> dict:
    """cos-weighted box means over the last two axes, NaN-skipping."""
    out = {}
    for k, (la, lo) in BOXES.items():
        m = _box_mask(lat, lon, la, lo)
        wgt = np.cos(np.deg2rad(lat))[:, None] * m
        num = np.nansum(field * wgt, axis=(-2, -1))
        den = np.sum(wgt * np.isfinite(field), axis=(-2, -1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = num / den
    return out


def warming_centre(field: np.ndarray, lat, lon) -> np.ndarray:
    """Longitude (°E) of the centre of the equatorial warming.

    The 2°S–2°N mean anomaly, run through a 10°-wide running mean along the equator, is searched
    between 150°E and 80°W; the centre is the anomaly-weighted mean longitude of the stretch that
    is within 30 % of the peak. A bare maximum jumps between two near-equal humps (DJF 2023–24
    reads 102°W by maximum and 131°W this way, 2018–19 90°W against 144°W) — the flavour of an
    event should not flip on a tenth of a degree. NaN when no part of the equator is warm."""
    eq = (lat >= EQ_LAT[0]) & (lat <= EQ_LAT[1])
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmean(field[..., eq, :], axis=-2)              # (..., lon)
            dx = float(np.median(np.diff(lon)))
            k = int(round(SMOOTH_DEG / dx)) // 2
            sm = np.stack([np.nanmean(prof[..., max(0, j - k): j + k + 1], -1)
                           for j in range(prof.shape[-1])], -1)
    srch = (lon >= CENTRE_SEARCH[0]) & (lon <= CENTRE_SEARCH[1])
    q = np.where(srch, sm, np.nan)
    mx = np.nanmax(q, axis=-1, keepdims=True)
    wgt = np.where((q >= 0.7 * mx) & (mx > 0), q, 0.0)
    wgt = np.nan_to_num(wgt)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (wgt * lon).sum(-1) / wgt.sum(-1)
    return np.where(mx[..., 0] > 0, c, np.nan)


def eq_profile(field: np.ndarray, lat, lon, at_lon) -> np.ndarray:
    """2°S–2°N mean anomaly sampled at the longitudes `at_lon` (the ERSST 2° points)."""
    eq = (lat >= EQ_LAT[0]) & (lat <= EQ_LAT[1])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.nanmean(field[..., eq, :], axis=-2)
    j = [int(np.argmin(np.abs(lon - v))) for v in at_lon]
    return prof[..., j]


def obs_indices(obs: dict, eof: dict) -> pd.DataFrame:
    a, lat, lon = obs["a"], obs["lat"], obs["lon"]
    E, C = project(a, eof)
    bx = box_means(a, lat, lon)
    df = pd.DataFrame({"E": E, "C": C, **bx, "lon": warming_centre(a, lat, lon)},
                      index=pd.Index(obs["months"], name="month"))
    return df


def obs_seasons(obs: dict, eof: dict) -> dict:
    """{season: DataFrame indexed by onset year} from season-MEAN fields (the centre is nonlinear)."""
    a, lat, lon, months = obs["a"], obs["lat"], obs["lon"], obs["months"]
    idx = {m: i for i, m in enumerate(months)}
    out = {}
    for s, mons in SEASONS.items():
        rows = {}
        for y in range(int(months[0][:4]), int(months[-1][:4]) + 1):
            keys = [f"{y + (1 if (m < mons[0]) else 0)}-{m:02d}" for m in mons]
            if not all(k in idx for k in keys):
                continue
            f = np.mean([a[idx[k]] for k in keys], 0)
            E, C = project(f[None], eof)
            bx = box_means(f[None], lat, lon)
            rows[y] = dict(E=float(E[0]), C=float(C[0]), lon=float(warming_centre(f[None], lat, lon)[0]),
                           prof=eq_profile(f, lat, lon, lon),
                           **{k: float(v[0]) for k, v in bx.items()})
        df = pd.DataFrame(rows).T
        for c in df.columns:
            if c != "prof":
                df[c] = df[c].astype(float)
        out[s] = df
    return out


# ─────────────────────────────────────────────────────────────────────────── forecast
def model_indices(da: xr.DataArray, issue: str, obs: dict, eof: dict) -> dict:
    """Per-member indices by lead and by season for one system.

    The anomaly is moved from the model's 1993–2016 base onto the observed 1991–2020 base by adding
    the observed climatology difference for the valid calendar month. E and C are projected at the
    ERSST 2° points (which sit on the 1° strip exactly); boxes and the centre use the 1° strip."""
    start = int(issue[4:6])
    vmon = [((start - 1 + L - 1) % 12) + 1 for L in range(1, len(LEADS) + 1)]
    dmap = obs["c93"] - obs["c91"]                                      # (12, lat2, lon2)
    d_obsgrid = xr.DataArray(dmap, dims=("m", "latitude", "longitude"),
                             coords=dict(m=np.arange(1, 13), latitude=obs["lat"], longitude=obs["lon"]))
    lat1, lon1 = da["latitude"].values, da["longitude"].values
    # the base shift on the 1° strip: interpolated inside the EOF box, nearest outside it
    d1 = (d_obsgrid.interp(latitude=lat1, longitude=lon1)
          .combine_first(d_obsgrid.interp(latitude=lat1, longitude=lon1, method="nearest",
                                          kwargs={"fill_value": None}))).values
    d1 = np.nan_to_num(d1)
    x1 = da.values.copy()                                                # (mem, lead, lat, lon)
    for li, m in enumerate(vmon):
        x1[:, li] += d1[m - 1]
    # the ERSST points
    ilat = [int(np.argmin(np.abs(lat1 - v))) for v in obs["lat"]]
    ilon = [int(np.argmin(np.abs(lon1 - v))) for v in obs["lon"]]
    assert np.allclose(lat1[ilat], obs["lat"]) and np.allclose(lon1[ilon], obs["lon"]), "strip is not on the 2° points"
    x2 = x1[:, :, ilat][:, :, :, ilon]
    # an ocean cell ERSST has that the model masks (a coastline at 1°): take the nearest ocean
    # cell of the same row (the land mask is the same for every member and lead)
    probe = np.isfinite(x2).all(axis=(0, 1))
    bad = eof["valid"] & ~probe
    for i, j in zip(*np.where(bad)):
        ok = np.where(probe[i])[0]
        if ok.size:
            x2[:, :, i, j] = x2[:, :, i, ok[np.argmin(np.abs(ok - j))]]
    nbad = int(bad.sum())

    def pack(f1, f2):
        E, C = project(f2, eof)
        bx = box_means(f1, lat1, lon1)
        return dict(E=E, C=C, lon=warming_centre(f1, lat1, lon1),
                    prof=eq_profile(f1, lat1, lon1, obs["lon"]), **bx)

    by_lead = pack(x1, x2)                                               # each (mem, lead)
    seas = {}
    for s, mons in SEASONS.items():
        li = [vmon.index(m) for m in mons if m in vmon]
        if len(li) < 3:
            continue
        seas[s] = pack(x1[:, li].mean(1), x2[:, li].mean(1))             # each (mem,)
    raw_n34 = box_means(da.values, lat1, lon1)["n34"]                    # on the model base, for the check
    return dict(lead=by_lead, season=seas, raw_n34=raw_n34, n=da.sizes["member"], filled=nbad,
                valid_months=[f"{int(issue[:4]) + (start - 1 + L - 1) // 12}-{m:02d}"
                              for L, m in zip(range(1, len(LEADS) + 1), vmon)])


def collect(issue: str, obs: dict, eof: dict) -> dict:
    """{label: (indices, colour)} with the ECCC pair merged into CanSIPS."""
    out = {}
    for centre, system, label, colour in MODELS:
        p = strip_path(centre, system, issue)
        if not p.exists():
            p = fetch_strip(centre, system, issue)
        if p is None:
            print(f"  {label:16s} unavailable — skipped", flush=True)
            continue
        try:
            r = model_indices(load_strip(p), issue, obs, eof)
        except Exception as e:                                          # noqa: BLE001
            print(f"  {label:16s} unreadable ({str(e)[:90]}) — skipped", flush=True)
            continue
        out[label] = (r, colour)
        d = r["season"].get("DJF")
        print(f"  {label:16s} {r['n']:>3d} mem" + (f" · DJF E {np.mean(d['E']):+.2f} C {np.mean(d['C']):+.2f}"
              f" centre {_lonlab(np.nanmean(d['lon']))}" if d else "") +
              (f" · {r['filled']} coastal cells filled" if r["filled"] else ""), flush=True)
    for merged, parts in MERGE.items():
        have = [q for q in parts if q in out]
        if len(have) < 2:
            continue
        colour = out[have[0]][1]
        rs = [out[q][0] for q in have]
        m = dict(rs[0])
        m["lead"] = {k: np.concatenate([r["lead"][k] for r in rs]) for k in rs[0]["lead"]}
        m["season"] = {s: {k: np.concatenate([r["season"][s][k] for r in rs]) for k in rs[0]["season"][s]}
                       for s in rs[0]["season"] if all(s in r["season"] for r in rs)}
        m["raw_n34"] = np.concatenate([r["raw_n34"] for r in rs])
        m["n"] = sum(r["n"] for r in rs)
        for q in have:
            del out[q]
        out[merged] = (m, colour)
        print(f"  {merged:16s} {m['n']:>3d} mem (merged: {' + '.join(have)})", flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────── summary
EAST_MARGIN = 1.0      # E − C ≥ 1 σ: east-based; ≤ −1 σ: central; between: basin-wide


def classify(E, C):
    B = np.asarray(E) - np.asarray(C)
    return np.where(B >= EAST_MARGIN, "east", np.where(B <= -EAST_MARGIN, "central", "basin"))


def _lonlab(L) -> str:
    if L is None or not np.isfinite(L):
        return "—"
    return f"{L:.0f}°E" if L <= 180 else f"{360 - L:.0f}°W"


def season_summary(res: dict, s: str, ev: pd.DataFrame) -> dict | None:
    """Everything the verdict needs, each model counted once (the plume's convention)."""
    from scipy import stats
    labs = [k for k in res if s in res[k][0]["season"]]
    if len(labs) < 3:
        return None
    frac = {c: 0.0 for c in ("east", "basin", "central")}
    means, cls_models = {}, {}
    for k in labs:
        d = res[k][0]["season"][s]
        cl = classify(d["E"], d["C"])
        for c in frac:
            frac[c] += float(np.mean(cl == c)) / len(labs)
        means[k] = dict(E=float(np.mean(d["E"])), C=float(np.mean(d["C"])),
                        lon=float(np.nanmedian(d["lon"])), n12=float(np.mean(d["n12"])),
                        n3=float(np.mean(d["n3"])), n34=float(np.mean(d["n34"])), n4=float(np.mean(d["n4"])),
                        east_gt=float(np.mean(np.asarray(d["E"]) > np.asarray(d["C"]))))
        cls_models[k] = str(classify(means[k]["E"], means[k]["C"]))
    B = np.array([means[k]["E"] - means[k]["C"] for k in labs])
    t = stats.ttest_1samp(B, 0.0)
    mm = {q: float(np.mean([means[k][q] for k in labs])) for q in ("E", "C", "n12", "n3", "n34", "n4")}
    mm["lon"] = float(np.median([means[k]["lon"] for k in labs]))
    # east-of-120°W share of members (model-weighted)
    f_lon_e = float(np.mean([np.mean(np.asarray(res[k][0]["season"][s]["lon"]) >= 240.0) for k in labs]))
    f_e_gt_c = float(np.mean([means[k]["east_gt"] for k in labs]))
    # nearest past El Niño in (E, C)
    nino = ev[ev["kind"] == "el nino"]
    d2 = np.hypot(nino["E"] - mm["E"], nino["C"] - mm["C"])
    near = [dict(label=nino.loc[i, "label"], E=round(float(nino.loc[i, "E"]), 2),
                 C=round(float(nino.loc[i, "C"]), 2), dist=round(float(d2[i]), 2))
            for i in d2.sort_values().index[:3]]
    cls_mm = str(classify(mm["E"], mm["C"]))
    n_same = sum(1 for k in labs if cls_models[k] == cls_mm)
    sig = bool(t.pvalue < 0.05)
    # the verdict: a class is asserted only when the direction is significant across models AND
    # two thirds of the members AND all but one model sit in it; otherwise the page says the
    # models disagree (or that the lean is not significant) and names the split.
    if sig and frac[cls_mm] >= 2 / 3 and n_same >= len(labs) - 1:
        verdict = cls_mm
    elif sig and cls_mm != "basin":
        verdict = f"lean-{cls_mm}"
    else:
        verdict = "split"
    return dict(models=len(labs), frac={c: round(v, 3) for c, v in frac.items()},
                frac_e_gt_c=round(f_e_gt_c, 3), frac_lon_east_of_120w=round(f_lon_e, 3),
                mmm={k: round(v, 2) for k, v in mm.items()},
                model_means={k: {q: round(v, 2) for q, v in m.items()} for k, m in means.items()},
                model_class=cls_models, n_models_in_class=n_same,
                spread_E_minus_C=dict(min=round(float(B.min()), 2), max=round(float(B.max()), 2),
                                      sd=round(float(B.std(ddof=1)), 2)),
                ttest=dict(stat=round(float(t.statistic), 2), p=float(f"{t.pvalue:.2g}"), df=len(labs) - 1,
                           test="one-sample t-test of the model-mean E − C against 0, each model once"),
                significant=sig, verdict=verdict, klass=cls_mm, analogues=near)


def events_table(seas: dict) -> pd.DataFrame:
    """Every winter since 1950 with its CPC class, keyed by onset year, for each season."""
    off = {}
    p = ASSETS / "data" / "cpc_official.json"
    if p.exists():
        off = json.loads(p.read_text()).get("oni", {})
    rows = []
    djf = seas["DJF"]
    for y in djf.index:
        y = int(y)
        win = [off.get(f"{yy}-{mm:02d}") for yy, mm in
               [(y, 7), (y, 8), (y, 9), (y, 10), (y, 11), (y, 12), (y + 1, 1), (y + 1, 2)]]
        win = [v for v in win if v is not None]
        nd = [off.get(f"{y}-12"), off.get(f"{y + 1}-01")]               # NDJ, DJF (centred months)
        nd = [v for v in nd if v is not None]
        if not nd:
            continue
        kind = "el nino" if max(nd) >= 0.5 else ("la nina" if min(nd) <= -0.5 else "neutral")
        peak = max(win) if kind == "el nino" else (min(win) if kind == "la nina" else None)
        if kind == "el nino":
            strength = ("very strong" if peak >= 2.0 else "strong" if peak >= 1.5
                        else "moderate" if peak >= 1.0 else "weak")
        else:
            strength = None
        rows.append(dict(year=y, label=f"{y}–{str(y + 1)[2:]}", kind=kind,
                         peak_oni=peak, strength=strength))
    ev = pd.DataFrame(rows).set_index("year")
    return ev


NAMED = {1957, 1965, 1972, 1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009, 2015, 2018, 2023}


def build(issue: str, prev_issue: str | None) -> dict:
    obs = load_obs()
    eof = fit_eofs(obs)
    print(f"EOFs {TRAIN[0]}–{TRAIN[1]} (ERSSTv6, fixed {BASE[0]}–{BASE[1]} base): variance "
          f"{eof['var'][0]:.1%} / {eof['var'][1]:.1%}", flush=True)
    mon = obs_indices(obs, eof)
    seas = obs_seasons(obs, eof)
    ev = events_table(seas)

    res = collect(issue, obs, eof)
    if len(res) < 3:
        raise SystemExit(f"only {len(res)} systems for {issue}")
    # the sanity check against the plume: same product, same members, same box
    chk = {}
    if FC_JSON.exists():
        F = json.loads(FC_JSON.read_text())
        if F.get("issue", "").replace("-", "") == issue:
            for k, (r, _c) in res.items():
                if k in F["models"]:
                    a_ = np.array([np.mean(x) for x in F["models"][k]["n34"]])
                    b_ = np.array([np.mean(r["raw_n34"][:, L]) for L in range(len(LEADS))])
                    chk[k] = round(float(np.max(np.abs(a_ - b_))), 3)
            print("  Niño-3.4 vs the plume's own (max |Δ| of the model mean over 6 leads): " +
                  ", ".join(f"{k} {v:.3f}" for k, v in chk.items()), flush=True)

    per_season = {}
    for s in SEASONS:
        sd = seas[s].reindex(ev.index)
        e2 = ev.join(sd)
        per_season[s] = e2
    summaries = {s: season_summary(res, s, per_season[s]) for s in SEASONS}

    prev = None
    if prev_issue:
        try:
            pr = collect(prev_issue, obs, eof)
            if len(pr) >= 3:
                prev = {"issue": f"{prev_issue[:4]}-{prev_issue[4:]}",
                        "seasons": {s: season_summary(pr, s, per_season[s]) for s in SEASONS}}
        except SystemExit:
            prev = None

    first = next(iter(res.values()))[0]
    r2 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 2)   # noqa: E731
    r0 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 1)   # noqa: E731
    models = {}
    for k, (r, colour) in res.items():
        # per member: what the page draws (the box indices go into the summary as model means)
        models[k] = {"color": colour, "n": int(r["n"]),
                     "lead": {q: [[r2(v) if q != "lon" else r0(v) for v in r["lead"][q][:, L]]
                                  for L in range(len(LEADS))] for q in ("E", "C", "lon")},
                     "season": {s: {q: [r2(v) if q != "lon" else r0(v) for v in r["season"][s][q]]
                                    for q in ("E", "C", "lon")}
                                for s in r["season"]}}

    # equatorial anomaly profiles (2°S–2°N, at the ERSST 2° longitudes): the forecast per season
    # (each model's mean, the multi-model mean, pooled-member P10/P50/P90), the named past events
    # in the same seasons, and the latest observed months
    rl = lambda arr: [r2(v) for v in arr]                               # noqa: E731
    profiles = {"lon": [float(v) for v in obs["lon"]], "forecast": {}, "events": {}, "obs": {}}
    for s in SEASONS:
        labs = [k for k in res if s in res[k][0]["season"]]
        if not labs:
            continue
        mp = {k: np.nanmean(res[k][0]["season"][s]["prof"], 0) for k in labs}
        pool = np.concatenate([res[k][0]["season"][s]["prof"] for k in labs], 0)
        profiles["forecast"][s] = {
            "models": {k: rl(v) for k, v in mp.items()},
            "mmm": rl(np.mean(list(mp.values()), 0)),
            "p10": rl(np.nanpercentile(pool, 10, 0)), "p50": rl(np.nanpercentile(pool, 50, 0)),
            "p90": rl(np.nanpercentile(pool, 90, 0))}
    for y in sorted(NAMED):
        if y not in per_season["DJF"].index:
            continue
        lab = per_season["DJF"].loc[y, "label"]
        profiles["events"][lab] = {s: rl(per_season[s].loc[y, "prof"]) for s in SEASONS
                                   if y in per_season[s].index and isinstance(per_season[s].loc[y, "prof"], np.ndarray)}
    for i in range(-3, 0):
        profiles["obs"][obs["months"][i]] = rl(eq_profile(obs["a"][i], obs["lat"], obs["lon"], obs["lon"]))

    hist_prov = None
    if HIST_JSON.exists():
        hist_prov = json.loads(HIST_JSON.read_text()).get("provisional_after")
    last = mon.index[-1]
    provisional_after = min(hist_prov, _prev_month(last)) if hist_prov else _prev_month(last)
    tail = mon.iloc[-24:]
    events = []
    for y, row in per_season["DJF"].iterrows():
        if row["kind"] == "neutral":
            continue
        e = dict(year=int(y), label=row["label"], kind=row["kind"], strength=row["strength"],
                 peak_oni=r2(row["peak_oni"]), named=bool(int(y) in NAMED))
        for s in SEASONS:
            q = per_season[s].loc[y]
            e[s] = dict(E=r2(q.get("E")), C=r2(q.get("C")), lon=r0(q.get("lon")),
                        n12=r2(q.get("n12")), n34=r2(q.get("n34")), n4=r2(q.get("n4")))
            if e[s]["E"] is not None:
                e[s]["class"] = str(classify(q["E"], q["C"]))
        events.append(e)

    doc = {
        "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "issue": f"{issue[:4]}-{issue[4:]}",
        "valid_months": first["valid_months"],
        "seasons": {s: list(m) for s, m in SEASONS.items()},
        "method": {
            "indices": "Takahashi et al. (2011) E and C: EOF1/EOF2 of monthly SST anomalies, 10°S–10°N, "
                       "140°E–80°W; E = (PC1 − PC2)/√2, C = (PC1 + PC2)/√2, in standard deviations",
            "obs": f"NOAA ERSSTv6 monthly, 2° grid, fixed {BASE[0]}–{BASE[1]} base, no detrending",
            "eof_training": f"{TRAIN[0]} to {TRAIN[1]}",
            "variance_explained": [round(float(v), 3) for v in eof["var"][:2]],
            "forecast": "C3S seasonal-postprocessed-single-levels, member SST anomalies vs each model's own "
                        "1993–2016 hindcast, moved onto the 1991–2020 base with the observed climatology "
                        "difference, projected at the ERSST 2° points",
            "classes": f"east-based: E − C ≥ {EAST_MARGIN:g}; central: E − C ≤ −{EAST_MARGIN:g}; basin-wide between",
            "centre": "anomaly-weighted mean longitude of the 2°S–2°N anomaly (10° running mean) where it is "
                      "within 30% of its peak, searched 150°E–80°W",
            "verdict_rule": "a class is named only when the across-model t-test of E − C against 0 gives "
                            "p < 0.05, two thirds of the members (each model weighted equally) fall in the "
                            "class, and all but at most one model mean does; a significant lean that misses "
                            "the other two is reported as a lean; anything else as a split",
            "plume_check_max_abs_diff_n34": chk,
            "trend_note": "On the fixed base the western Pacific's faster warming puts a slow drift into C "
                          "(see trend_per_decade); E barely moves. Older events therefore sit slightly low on C.",
            "trend_per_decade": _trends(mon),
        },
        "eof_patterns": {
            "lon": [float(v) for v in obs["lon"]],
            "eq1": [r2(v) for v in _eqprof(eof["pat1"], obs["lat"])],
            "eq2": [r2(v) for v in _eqprof(eof["pat2"], obs["lat"])],
        },
        "obs": {
            "source": "ERSSTv6 (NOAA/NCEI via PSL), 1991–2020 base",
            "latest": last, "provisional_after": provisional_after,
            "months": list(tail.index),
            **{q: [r2(v) if q != "lon" else r0(v) for v in tail[q].values] for q in ("E", "C", "lon", "n12", "n3", "n34", "n4")},
            "record_E": _record(mon, "E", last[:4]),
            "record_C": _record(mon, "C", last[:4]),
        },
        "events": events,
        "profiles": profiles,
        "summary": summaries,
        "previous": prev,
        "models": models,
    }
    return doc


def _record(mon: pd.DataFrame, q: str, before_year: str) -> dict:
    sub = mon[[m[:4] < before_year for m in mon.index]][q]
    m = sub.idxmax()
    return {"month": m, "value": round(float(sub[m]), 2)}


def _clean(o):
    """NaN/inf → None all the way down (the JSON is written with allow_nan=False)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _prev_month(ym: str) -> str:
    y, m = int(ym[:4]), int(ym[5:7]) - 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y:04d}-{m:02d}"


def _eqprof(P, lat):
    eq = (lat >= EQ_LAT[0]) & (lat <= EQ_LAT[1])
    return np.nanmean(P[eq], 0)


def _trends(mon: pd.DataFrame) -> dict:
    yrs = np.array([int(m[:4]) + (int(m[5:7]) - 0.5) / 12 for m in mon.index])
    ok = yrs < 2026
    return {q: round(float(np.polyfit(yrs[ok], mon[q].values[ok], 1)[0] * 10), 3) for q in ("E", "C")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", help="YYYYMM (default: the plume's issue in enso_forecast.json)")
    ap.add_argument("--prev", help="YYYYMM of the issue to compare against (default: the month before)")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    issue = a.issue or json.loads(FC_JSON.read_text())["issue"].replace("-", "")
    if a.fetch_only:
        got = fetch_all(issue)
        print(f"{len(got)}/{len(MODELS)} systems cached for {issue}")
        return 0 if got else 1
    prev = a.prev or (pd.Timestamp(f"{issue[:4]}-{issue[4:]}-01") - pd.DateOffset(months=1)).strftime("%Y%m")
    doc = build(issue, prev)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean(doc), separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    os.replace(tmp, out)
    s = doc["summary"].get("DJF") or {}
    print(f"saved {out.name} ({out.stat().st_size // 1024} KB) · DJF verdict {s.get('verdict')} "
          f"({s.get('frac')}, p {s.get('ttest', {}).get('p')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
