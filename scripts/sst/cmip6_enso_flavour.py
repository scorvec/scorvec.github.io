#!/usr/bin/env python3
"""East-based, basin-wide and central El Niño over the Americas in CMIP6, on the site's one east/central rule.

The CMIP6 side of the "East-based vs central" item of the El Niño monitor (enso.html). Every December-February of the
429 historical runs of the ENSO-impact study (16 models, 1950-2014) is placed on the same ruler as the observed winters
and the C3S forecast members of the ENSO forecasts page:

  E and C       Takahashi et al. (2011, GRL 38, L10704) from the OBSERVED ERSSTv6 EOFs of c3s_enso_flavour.py
                (10S-10N, 140E-80W, trained 1979-2020; reference/enso_flavour_eof.npz): each model winter's SST anomaly
                on the 2-degree EOF cells, projected with c3s_enso_flavour.project, so E and C are in standard deviations
                of the observed indices.
  classes       c3s_enso_flavour.classify: an El Niño winter (DJF Niño-3.4 >= +0.5 K) is east-based when E - C >= 1,
                central when E - C <= -1 and basin-wide in between; neutral winters have |Niño-3.4| < 0.5 K.
  anomalies     every field and index against the run's own mean of the other winters within +-10 years, at every grid
                point (fewer at the ends of the run), so the forced warming neither moves the early winters between
                classes nor enters the composites. The monitor's observed classes keep CPC's fixed 1991-2020 base; a
                65-year run with its forced trend needs the moving one.
  DJF means     the three monthly means weighted by their days on the run's own calendar (noleap and 360-day included).

Tests
  per model     its members' winters pooled. Class minus neutral: Welch t-test per grid point across the events,
                Benjamini-Hochberg FDR 10 % over the map. East minus central at the same strength: OLS on an east dummy
                with Niño-3.4 held fixed by bin effects (0.5-1, 1-1.5, 1.5-2, >= 2 K; bins holding both classes) and a
                linear term (east-based events run stronger even within a bin), t-test, FDR 10 %.
                A model with fewer than MIN_EVENTS events in a class has no map for it.
  multi-model   each model with a map counts once: across-model one-sample t-test, FDR 10 % over the map, AND >= 80 %
                of those models agreeing on the sign; fewer than MIN_MODELS models: not tested.
  E and C       per model, the pooled OLS of the field on E and C together (per observed sd); multi-model test as above.
  regions       the 20 regions of the ENSO-impact study: regional means per model, the same multi-model test with the
                FDR over the 20 regions; a value is kept only where it passes.
Precipitation is tested in mm/day and shown as % of the normal (cells under 0.3 mm/day are not scored).

Data: Amon ts, tas and pr of every member in reference/cmip6_enso_members.csv, and Amon zg for its `zg` subset (up to 10
members per model: each zg chunk carries all 19 levels), 1949-12 to 2014-12, from the Pangeo CMIP6 archive on Google
Cloud (anonymous). `extract` reduces one model's members to DJF means - ts on the EOF cells (area-overlap mean of the
ocean cells; an EOF cell with no model ocean takes the nearest one in its row, as the daily OISST projection does), tas
and pr on the 2.5-degree Americas grid, z500 on the 2.5-degree Northern Hemisphere grid (area-overlap means) - plus E and
C from the member's OWN two EOFs, a sensitivity check of the observed-pattern projection. `build` turns the per-model
files into reference/enso_flavour_impacts_cmip6.{npz,json}. Both run in .github/workflows/cmip6-enso-flavour.yml.

    python scripts/sst/cmip6_enso_flavour.py extract --model CanESM5 --out DIR [--max-members N] [--workers 2]
    python scripts/sst/cmip6_enso_flavour.py build --src DIR [--require-all]
"""

from __future__ import annotations

import pyproj  # noqa: F401  before xarray: the land mask goes through cartopy (libproj load order)
import argparse
import datetime as dt
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c3s_enso_flavour as F
import cmip6_pacific_ts_extract as PT  # cell edges, 1-D overlaps, the Natural Earth land, the sea-ice threshold
from enso_flavour_obs import _fill_row
from quantmet.stats import benjamini_hochberg

REF = HERE / "reference"
MEMBERS = REF / "cmip6_enso_members.csv"
OUT_STEM = "enso_flavour_impacts_cmip6"
SO = {"token": "anon"}
T0, T1 = "1949-12", "2014-12"  # whole months, valid on every calendar (a 360-day year has no 31 December)
NMON = 781
YEARS = np.arange(1950, 2015)  # each DJF by its January
AM = (np.arange(-60.0, 75.01, 2.5), np.arange(190.0, 330.01, 2.5))  # the Americas grid of the ENSO-impact study
NH = (np.arange(0.0, 90.01, 2.5), np.arange(0.0, 357.51, 2.5))
HALF = 10  # years either side in the local climatology
NEUTRAL = 0.5
BINS = (0.5, 1.0, 1.5, 2.0)  # Niño-3.4 strength bins of the east-minus-central fit
MIN_EVENTS = 8  # as the observed composites: fewer events, no map
MIN_MODELS = 5
AGREE = 0.8
ALPHA = 0.10
DRY = 0.3  # mm/day: no % of normal below this
_LAND_LOCK = threading.Lock()  # the prepared Natural Earth geometry is shared by the member threads
CLASSES = ("east", "basin", "central")
VARS = ("tas", "pr", "z500")
RANGE = {"tas": (180.0, 330.0), "pr": (0.0, 100.0), "z500": (4500.0, 6100.0)}  # plausible DJF means (K, mm/day, m)
REGIONS = {  # (lat0, lat1, lon0, lon1), longitude in degrees E (-180..180); the ENSO-impact study's 20 regions
    "Pacific Northwest": (42, 49, -125.5, -116),
    "California": (32, 42, -124.5, -117),
    "US Southwest": (31, 37, -115, -103),
    "Gulf Coast": (28, 33, -98, -84),
    "Southeast US": (25, 36.5, -84, -75.5),
    "Ohio Valley": (36.5, 42, -90, -79),
    "Great Lakes": (41, 49, -93, -76),
    "N Plains / Prairies": (45, 57, -115, -95),
    "Alaska": (58, 71, -165, -140),
    "Quebec / Labrador": (50, 60, -76, -58),
    "Mexico": (16, 31, -110, -94),
    "Central America": (7, 17, -93, -77),
    "Colombia": (-2, 11, -78, -71),
    "Venezuela": (1, 11, -70, -60),
    "Peru / Ecuador coast": (-12.5, -2, -81.5, -78.5),
    "Amazon": (-10, 2, -70, -50),
    "Northeast Brazil": (-12, -2, -45, -35),
    "La Plata basin": (-35, -22, -62, -48),
    "Central Chile": (-38, -30, -73, -69),
    "Patagonia": (-52, -40, -74, -64),
}


# ─────────────────────────────────────────────────────────────────────────── shared helpers
def overlap_ops(src_lat, src_lon, dst_lat, dst_lon) -> tuple[np.ndarray, np.ndarray]:
    """Separable area-overlap weights (destination x source) of a global native grid on a regular destination grid:
    latitude by d(sin lat), longitude periodic."""
    s_la = np.clip(PT._bounds(np.asarray(src_lat, float)), -90, 90)
    d_la = np.clip(PT._bounds(np.asarray(dst_lat, float)), -90, 90)
    A = PT._overlap(np.sin(np.deg2rad(s_la)), np.sin(np.deg2rad(d_la)))
    B = PT._overlap(PT._bounds(np.asarray(src_lon, float), periodic=True), PT._bounds(np.asarray(dst_lon, float)), True)
    return A, B


def regrid(x: np.ndarray, A: np.ndarray, B: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """(T, ny, nx) -> (T, my, mx) area-overlap means over the `valid` source cells (default: the finite ones); NaN where
    a destination cell overlaps none."""
    if valid is None:
        valid = np.isfinite(x)
    num = np.einsum("il,tlm,jm->tij", A, np.where(valid, x, 0.0), B, optimize=True)
    den = np.einsum("il,tlm,jm->tij", A, valid.astype("float64"), B, optimize=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def djf(x: np.ndarray, months: list[str], ndays: np.ndarray, years: np.ndarray = YEARS) -> np.ndarray:
    """Monthly (T, ...) -> (len(years), ...) DJF means (December of the year before), each month weighted by its days;
    NaN for a winter with a month missing."""
    row = {m: i for i, m in enumerate(months)}
    out = np.full((len(years), *x.shape[1:]), np.nan)
    for k, y in enumerate(years):
        ii = [row.get(m) for m in (f"{y - 1}-12", f"{y}-01", f"{y}-02")]
        if None not in ii:
            out[k] = np.average(x[ii], axis=0, weights=ndays[ii])
    return out


def local_anom(x: np.ndarray, half: int = HALF) -> np.ndarray:
    """(M, Y, ...) with years along axis 1 -> anomalies from the mean of the OTHER years within +-half (fewer at the ends),
    member by member and point by point; NaN-aware."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    pad = np.zeros_like(v[:, :1])
    s = np.concatenate([pad, np.cumsum(v, 1)], 1)
    n = np.concatenate([pad, np.cumsum(ok, 1)], 1)
    y = np.arange(x.shape[1])
    lo, hi = np.clip(y - half, 0, None), np.clip(y + half + 1, None, x.shape[1])
    S, N = s[:, hi] - s[:, lo] - v, n[:, hi] - n[:, lo] - ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok & (N > 0), x - S / np.maximum(N, 1), np.nan)


def land_fraction(lat: np.ndarray, lon: np.ndarray, sub: int = 10) -> np.ndarray:
    """Land fraction of each cell (centred on lat/lon, regular spacing) from Natural Earth 50 m, sub x sub samples."""
    import shapely

    land = PT.land_geom()
    d = float(abs(lat[1] - lat[0]))
    off = ((np.arange(sub) + 0.5) / sub - 0.5) * d
    lo = np.where(lon > 180, lon - 360, lon)
    Y = np.broadcast_to(lat[:, None, None, None] + off[None, None, :, None], (len(lat), len(lo), sub, sub))
    X = np.broadcast_to(lo[None, :, None, None] + off[None, None, None, :], (len(lat), len(lo), sub, sub))
    return shapely.contains_xy(land, X.ravel(), Y.ravel()).reshape(Y.shape).mean((2, 3))


def region_weights(lat: np.ndarray, lon: np.ndarray, lf: np.ndarray) -> np.ndarray:
    """(regions, lat, lon) land-fraction x cos(lat) weights, each summing to 1."""
    lo = np.where(lon > 180, lon - 360, lon)
    W = np.zeros((len(REGIONS), len(lat), len(lon)))
    for k, (a, b, c, e) in enumerate(REGIONS.values()):
        W[k] = (
            ((lat >= a) & (lat <= b))[:, None]
            * ((lo >= c) & (lo <= e))[None, :]
            * lf
            * np.cos(np.deg2rad(lat))[:, None]
        )
        W[k] /= W[k].sum()
    return W


def robust(stack: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """(models, ...) -> multi-model mean and mask: across-model one-sample t-test passing Benjamini-Hochberg FDR `alpha`
    over the map AND >= AGREE of the models agreeing on the sign."""
    with np.errstate(invalid="ignore", divide="ignore"):
        p = stats.ttest_1samp(stack, 0.0, axis=0).pvalue
    agree = np.maximum((stack > 0).mean(0), (stack < 0).mean(0))
    return stack.mean(0), benjamini_hochberg(p, alpha) & (agree >= AGREE)


def ols(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns of X (n, k) on Y (n, ...) -> coefficients (k, ...) and two-sided p of each."""
    Yf = Y.reshape(len(Y), -1)
    beta, *_ = np.linalg.lstsq(X, Yf, rcond=None)
    dof = len(X) - X.shape[1]
    s2 = ((Yf - X @ beta) ** 2).sum(0) / dof
    se = np.sqrt(np.outer(np.diag(np.linalg.inv(X.T @ X)), s2))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 2 * stats.t.sf(np.abs(beta / se), dof)
    return beta.reshape(X.shape[1], *Y.shape[1:]), p.reshape(X.shape[1], *Y.shape[1:])


def to_pct(field: np.ndarray, sig: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A precipitation field tested in mm/day, shown as % of `normal`; cells under DRY mm/day are not scored."""
    ok = normal >= DRY
    return np.where(ok, 100.0 * field / np.maximum(normal, 1e-6), np.nan), sig & ok


# ─────────────────────────────────────────────────────────────────────────── extract (one model, in Actions)
def _open(store: str, var: str) -> xr.DataArray:
    ds = xr.open_zarr(store, storage_options=SO, consolidated=True)
    v = ds[var].sel(time=slice(T0, T1))
    if "lon" in v.coords:
        v = v.assign_coords(lon=v.lon % 360).sortby("lon")
    return v.sortby("lat")


def _calendar(v: xr.DataArray) -> tuple[list[str], np.ndarray]:
    """'YYYY-MM' labels and days of each month on the store's own calendar; the record must be 1949-12 .. 2014-12."""
    months = [f"{t.year:04d}-{t.month:02d}" for t in v.indexes["time"]]
    if len(months) != NMON or len(set(months)) != NMON or (months[0], months[-1]) != ("1949-12", "2014-12"):
        raise ValueError(f"time axis {months[:1]}..{months[-1:]} ({len(months)} months, {len(set(months))} distinct)")
    return months, v.time.dt.days_in_month.values.astype("float64")


def _screen(x: np.ndarray, var: str) -> np.ndarray:
    lo, hi = RANGE[var]
    if not np.isfinite(x).all() or x.min() < lo or x.max() > hi:
        raise ValueError(f"{var} DJF means outside {lo}..{hi} or missing: {np.nanmin(x):.1f}..{np.nanmax(x):.1f}")
    return x


def ts_strip(store: str, grid: dict, eof: dict) -> tuple[list[str], np.ndarray, np.ndarray, str]:
    """Monthly surface temperature (K) on the EOF cells: area-overlap mean of the native ocean cells (centre not on
    Natural Earth land, not sea ice); EOF cells with no model ocean take the nearest one in their row."""
    import shapely

    v = _open(store, "ts").sel(lat=slice(-16, 16))
    months, nd = _calendar(v)
    lat, lon = v.lat.values.astype(float), v.lon.values.astype(float)
    x = v.values.astype("float64")
    X, Y = np.meshgrid(np.where(lon > 180, lon - 360, lon), lat)
    with _LAND_LOCK:
        ocean = ~shapely.contains_xy(PT.land_geom(), X.ravel(), Y.ravel()).reshape(X.shape)
    f = regrid(x, *overlap_ops(lat, lon, grid["lat"], grid["lon"]), np.isfinite(x) & (x >= PT.ICE) & ocean[None])
    f = _fill_row(f, eof["valid"])
    tv = f[:, eof["valid"]]
    if not np.isfinite(tv).all() or tv.min() < 280 or tv.max() > 315:
        raise ValueError(f"tropical ts outside 280..315 K or missing: {np.nanmin(tv):.1f}..{np.nanmax(tv):.1f}")
    return months, nd, f, str(v.time.dt.calendar)


def own_ec(f: np.ndarray, months: list[str], nd: np.ndarray, eof: dict) -> tuple[np.ndarray, np.ndarray]:
    """E and C from the member's OWN two EOFs, DJF means: monthly anomalies from its own +-10-year climatology of each
    calendar month, sqrt(cos lat)-weighted EOFs over the EOF cells, PCs standardised over the run, signs fixed against
    the observed patterns."""
    mon = np.array([int(m[5:7]) for m in months])
    a = np.full_like(f, np.nan)
    for m in range(1, 13):
        a[mon == m] = local_anom(f[mon == m][None])[0]
    X = a[:, eof["valid"]] * eof["w"][eof["valid"]]
    ok = np.isfinite(X).all(1)
    _, _, Vt = np.linalg.svd(X[ok], full_matrices=False)
    pc = X @ Vt[:2].T
    pc *= np.sign([Vt[0] @ eof["e1"], Vt[1] @ eof["e2"]])
    pc = (pc - np.nanmean(pc, 0)) / np.nanstd(pc, 0)
    E, C = (pc[:, 0] - pc[:, 1]) / np.sqrt(2), (pc[:, 0] + pc[:, 1]) / np.sqrt(2)
    return djf(E, months, nd), djf(C, months, nd)


def field_djf(store: str, var: str, dst: tuple[np.ndarray, np.ndarray], scale: float = 1.0) -> np.ndarray:
    """DJF means of one Amon field on a destination grid (area-overlap means of every native cell). zg: 500 hPa."""
    v = _open(store, var)
    if var == "zg":
        plev = v.plev.values.astype(float)
        want = 50000.0 if plev.max() > 2000 else 500.0  # Pa, or hPa in a few stores
        k = int(np.argmin(np.abs(plev - want)))
        if abs(plev[k] - want) > 1:
            raise ValueError(f"no 500 hPa level: {plev}")
        v = v.isel(plev=k)
    months, nd = _calendar(v)
    keep = np.flatnonzero(np.isin([int(m[5:7]) for m in months], (12, 1, 2)))
    lat0, lat1 = dst[0][0] - 6, dst[0][-1] + 6
    v = v.isel(time=keep).sel(lat=slice(lat0, lat1))
    x = v.values.astype("float64") * scale
    f = regrid(x, *overlap_ops(v.lat.values, v.lon.values, *dst))
    return djf(f, [months[i] for i in keep], nd[keep])


def extract_member(row, grid: dict, eof: dict, tries: int = 3) -> tuple[str, dict | str]:
    name = f"{row.source_id}_{row.member_id}"
    err = ""
    for attempt in range(tries):
        try:
            months, nd, f, cal = ts_strip(row.ts, grid, eof)
            out = {"ts": djf(f, months, nd), "calendar": cal}
            out["own_E"], out["own_C"] = own_ec(f, months, nd, eof)
            out["tas"] = _screen(field_djf(row.tas, "tas", AM), "tas")
            out["pr"] = _screen(field_djf(row.pr, "pr", AM, 86400.0), "pr")  # kg m-2 s-1 -> mm/day
            if row.zg:
                out["z500"] = _screen(field_djf(row.zg_store, "zg", NH), "z500")
            return name, out
        except Exception as e:  # noqa: BLE001  one bad store must not stop the model
            err = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  {name} attempt {attempt + 1}: {err}", flush=True)
            time.sleep(15)
    return name, err


def _i16(x: np.ndarray, offset: float, scale: float) -> np.ndarray:
    return np.where(np.isfinite(x), np.round((x - offset) * scale), -32768).clip(-32768, 32767).astype("int16")


def cmd_extract(a) -> int:
    grid, eof = F.load_eof_reference()
    rows = pd.read_csv(MEMBERS)
    rows = rows[rows.source_id == a.model]
    if rows.empty:
        raise SystemExit(f"{a.model}: not in {MEMBERS.name}")
    if a.max_members:
        rows = rows.head(a.max_members)
    t0 = time.time()
    print(f"{a.model}: {len(rows)} members, {int(rows.zg.sum())} with zg ({a.workers} workers)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda r: extract_member(r, grid, eof), rows.itertuples()))
    good = [(n, r) for n, r in res if isinstance(r, dict)]
    bad = {n: r for n, r in res if isinstance(r, str)}
    for n, r in bad.items():
        print(f"::warning::{n} skipped: {r}", flush=True)
    if not good:
        raise SystemExit(f"{a.model}: no member extracted")
    cals = {r["calendar"] for _, r in good}
    zg = [(n, r) for n, r in good if "z500" in r]
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out / f"{a.model}.npz",
        source_id=a.model,
        calendar=",".join(sorted(cals)),
        members=np.array([n.rsplit("_", 1)[1] for n, _ in good]),
        years=YEARS,
        ts=np.stack([r["ts"] for _, r in good]).astype("float32"),
        own_E=np.stack([r["own_E"] for _, r in good]).astype("float32"),
        own_C=np.stack([r["own_C"] for _, r in good]).astype("float32"),
        tas=_i16(np.stack([r["tas"] for _, r in good]), 273.15, 100),
        pr=_i16(np.stack([r["pr"] for _, r in good]), 0.0, 100),
        zg_members=np.array([n.rsplit("_", 1)[1] for n, _ in zg]),
        z500=_i16(np.stack([r["z500"] for _, r in zg]), 5000.0, 1) if zg else np.zeros((0, len(YEARS), 0, 0), "int16"),
        skipped=np.array(sorted(bad)),
    )
    print(
        f"{a.model}: {len(good)} members ({len(zg)} with z500), {len(bad)} skipped, calendar {cals}, "
        f"{(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────── build (all models)
def load_model(path: Path) -> dict:
    """One model's DJF tables, decoded: ts (K), tas (K), pr (mm/day), z500 (m)."""
    z = np.load(path)

    def dec(q, offset, scale):
        return np.where(q == -32768, np.nan, q / scale + offset)

    return dict(
        model=str(z["source_id"]),
        members=[str(m) for m in z["members"]],
        zg_members=[str(m) for m in z["zg_members"]],
        skipped=[str(m) for m in z["skipped"]],
        ts=z["ts"].astype("float64"),
        own_E=z["own_E"].astype("float64"),
        own_C=z["own_C"].astype("float64"),
        tas=dec(z["tas"], 273.15, 100.0),
        pr=dec(z["pr"], 0.0, 100.0),
        z500=dec(z["z500"], 5000.0, 1.0),
    )


def class_maps(A: np.ndarray, n34: np.ndarray, cls: np.ndarray) -> dict:
    """One model's pooled winters (n, lat, lon) -> {class: composite minus neutral, FDR mask, events, mean Niño-3.4} and
    'diff', east minus central with Niño-3.4 held fixed; a class with fewer than MIN_EVENTS events gets no map."""
    neu, en = np.abs(n34) < NEUTRAL, n34 >= NEUTRAL
    out = {}
    for k in CLASSES:
        sel = en & (cls == k)
        out[k] = dict(n=int(sel.sum()), x=round(float(n34[sel].mean()), 3) if sel.any() else None)
        if sel.sum() >= MIN_EVENTS:
            with np.errstate(invalid="ignore", divide="ignore"):
                p = stats.ttest_ind(A[sel], A[neu], axis=0, equal_var=False).pvalue
            out[k].update(f=A[sel].mean(0) - A[neu].mean(0), s=benjamini_hochberg(p, ALPHA))
    b = np.digitize(n34, BINS)
    east, cent = en & (cls == "east"), en & (cls == "central")
    shared = [k for k in range(1, len(BINS) + 1) if (east & (b == k)).any() and (cent & (b == k)).any()]
    sel = (east | cent) & np.isin(b, shared)
    ne, nc = int((east & sel).sum()), int((cent & sel).sum())
    out["diff"] = dict(
        n=ne,
        n_b=nc,
        x=round(float(n34[east & sel].mean()), 3) if ne else None,
        x_b=round(float(n34[cent & sel].mean()), 3) if nc else None,
    )
    if ne >= MIN_EVENTS and nc >= MIN_EVENTS:
        strength = n34[sel] - n34[sel].mean()
        X = np.column_stack([east[sel], strength] + [b[sel] == k for k in shared]).astype("float64")
        beta, p = ols(X, A[sel])
        out["diff"].update(f=beta[0], s=benjamini_hochberg(p[0], ALPHA))
    return out


def model_stats(m: dict, grid: dict, eof: dict) -> dict:
    """Indices, classes and every per-model map of one model."""
    ts = local_anom(m["ts"])
    E, C = F.project(ts, eof)
    n34 = F.box_means(ts, grid["lat"], grid["lon"])["n34"]
    cls = F.classify(E, C)
    ok = np.isfinite(E) & np.isfinite(C) & np.isfinite(n34)
    en = ok & (n34 >= NEUTRAL)
    own = F.classify(m["own_E"], m["own_C"])
    own_ok = en & np.isfinite(m["own_E"]) & np.isfinite(m["own_C"])
    res = dict(
        counts=dict(
            members=len(m["members"]),
            zg_members=len(m["zg_members"]),
            winters=int(ok.sum()),
            neutral=int((ok & (np.abs(n34) < NEUTRAL)).sum()),
            el_nino=int(en.sum()),
            **{k: int((en & (cls == k)).sum()) for k in CLASSES},
            own_same_class=round(float(np.mean(own[own_ok] == cls[own_ok])), 3) if own_ok.any() else None,
            **{f"own_{k}": int((own_ok & (own == k)).sum()) for k in CLASSES},
        ),
        events=np.column_stack([E[en], C[en], n34[en]]),
        maps={},
    )
    # tas and pr exist for every member, z500 for the zg members only (rows of the indices)
    zrows = np.array([m["members"].index(z) for z in m["zg_members"]], int)
    for v, rows in (("tas", np.arange(len(m["members"]))), ("pr", np.arange(len(m["members"]))), ("z500", zrows)):
        if not len(rows):
            continue
        A = local_anom(m[v])
        good = ok[rows] & np.isfinite(A).all(axis=(2, 3))
        Ap, xp, cp = A[good], n34[rows][good], cls[rows][good]
        res["maps"][v] = class_maps(Ap, xp, cp)
        beta, _ = ols(np.column_stack([np.ones(len(Ap)), E[rows][good], C[rows][good]]), Ap)
        res["maps"][v]["E"], res["maps"][v]["C"] = dict(f=beta[1]), dict(f=beta[2])
    res["pr_normal"] = np.nanmean(m["pr"], axis=(0, 1))  # mm/day, for % of normal
    return res


def regional_values(stack: np.ndarray, W: np.ndarray, normals: np.ndarray | None) -> dict:
    """Per-model maps (models, lat, lon) -> {region: multi-model mean or "n.s."} with the multi-model test over the 20
    regions; precipitation as % of each model's regional normal."""
    R = np.tensordot(stack, W, axes=([-2, -1], [1, 2]))
    if normals is not None:
        R = 100.0 * R / np.tensordot(normals, W, axes=([-2, -1], [1, 2]))
    mm, sig = robust(R)
    return {r: (round(float(mm[i]), 3) if sig[i] else "n.s.") for i, r in enumerate(REGIONS)}


def sig_frac(v: str, sig: np.ndarray, field: np.ndarray, land: np.ndarray) -> float:
    """Share of the scored map that passes: land for tas and pr, 20-90N for z500."""
    area = np.broadcast_to((NH[0] >= 20)[:, None], field.shape) if v == "z500" else land
    return round(float(sig[area & np.isfinite(field)].mean()), 3)


def cmd_build(a) -> int:
    t0 = time.time()
    grid, eof = F.load_eof_reference()
    want = sorted(pd.read_csv(MEMBERS).source_id.unique())
    files = {p.stem: p for p in sorted(a.src.glob("*.npz"))}
    missing = [m for m in want if m not in files]
    if missing and a.require_all:
        raise SystemExit(f"missing models: {missing}")
    names = [m for m in want if m in files]
    st, skipped = {}, {}
    for n in names:
        m = load_model(files[n])
        st[n] = model_stats(m, grid, eof)
        if m["skipped"]:
            skipped[n] = m["skipped"]
        c = st[n]["counts"]
        print(
            f"  {n:15s} {c['members']:3d} members ({c['zg_members']} z500): {c['el_nino']:5d} El Niño winters, east "
            f"{c['east']:4d} basin {c['basin']:5d} central {c['central']:4d}; own EOFs, same class {c['own_same_class']}",
            flush=True,
        )
    lf = land_fraction(*AM)
    land = lf >= 0.5
    W = region_weights(*AM, lf)
    npz = {"lat": AM[0], "lon": AM[1], "zlat": NH[0], "zlon": NH[1], "models": np.array(names)}
    ev = [np.column_stack([st[n]["events"], np.full(len(st[n]["events"]), i)]) for i, n in enumerate(names)]
    npz["events"] = np.concatenate(ev).astype("float32")  # E, C, Niño-3.4, model index: every El Niño winter
    meta = {
        "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "rule": f"El Niño: DJF Niño-3.4 >= +{NEUTRAL:g} K; east-based E - C >= {F.EAST_MARGIN:g}, central E - C <= "
        f"-{F.EAST_MARGIN:g}, basin-wide between (c3s_enso_flavour.classify); E and C on the observed ERSSTv6 EOFs",
        "anomalies": f"each run's own mean of the other winters within +-{HALF} years, per grid point",
        "min_events": MIN_EVENTS,
        "min_models": MIN_MODELS,
        "models": names,
        "missing_models": missing,
        "skipped_members": skipped,
        "counts": {n: st[n]["counts"] for n in names},
        "maps": {},
        "per_model": {},
        "regions": {},
    }
    normals = {n: st[n]["pr_normal"] for n in names}
    mm_normal = np.mean(list(normals.values()), axis=0)  # the multi-model precipitation normal
    for v in VARS:
        for k in (*CLASSES, "diff", "E", "C"):
            key = f"{v}|{k}"
            per = {n: st[n]["maps"][v][k] for n in names if v in st[n]["maps"]}
            have = [n for n, e in per.items() if "f" in e]
            info = dict(models=have, n_models=len(have), tested=len(have) >= MIN_MODELS)
            if k in (*CLASSES, "diff"):
                info["events"] = sum(per[n]["n"] for n in have)
                info["x"] = round(float(np.mean([per[n]["x"] for n in have])), 3) if have else None
                if k == "diff":
                    info["events_b"] = sum(per[n]["n_b"] for n in have)
                    info["x_b"] = round(float(np.mean([per[n]["x_b"] for n in have])), 3) if have else None
                meta["per_model"][key] = {
                    n: {q: e[q] for q in ("n", "n_b", "x", "x_b") if q in e} for n, e in per.items()
                }
                for n in have:  # every model's own map, tested across its events
                    f, s = per[n]["f"], per[n]["s"]
                    if v == "pr":
                        f, s = to_pct(f, s, normals[n])
                    npz[f"{key}|{n}|f"], npz[f"{key}|{n}|s"] = f.astype("float32"), s
                    meta["per_model"][key][n]["sig_frac"] = sig_frac(v, s, f, land)
            if info["tested"]:
                S = np.stack([per[n]["f"] for n in have])
                f, s = robust(S)
                if v != "z500" and k in (*CLASSES, "diff"):
                    meta["regions"][key] = regional_values(
                        S, W, np.stack([normals[n] for n in have]) if v == "pr" else None
                    )
                if v == "pr":
                    f, s = to_pct(f, s, mm_normal)
                npz[f"{key}|f"], npz[f"{key}|s"] = f.astype("float32"), s
                info["sig_frac"] = sig_frac(v, s, f, land)
                print(f"  {key}: {len(have)} models, {100 * info['sig_frac']:.0f} % robust", flush=True)
            meta["maps"][key] = info
    np.savez_compressed(REF / f"{OUT_STEM}.npz", **npz)
    (REF / f"{OUT_STEM}.json").write_text(json.dumps(meta, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    print(f"wrote {OUT_STEM}.{{npz,json}} ({len(names)} models, {time.time() - t0:.0f} s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model", required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--max-members", type=int, default=0, help="the first N members only (a validation run)")
    e.add_argument("--workers", type=int, default=2)
    b = sub.add_parser("build")
    b.add_argument("--src", type=Path, required=True)
    b.add_argument("--require-all", action="store_true", help="fail unless every model of the member list is there")
    a = ap.parse_args()
    return cmd_extract(a) if a.cmd == "extract" else cmd_build(a)


if __name__ == "__main__":
    raise SystemExit(main())
