#!/usr/bin/env python3
"""Point time series for models.html meteograms, and the projection math that finds a point on a model grid
(2026-10-01, user: "click-an-airport meteograms ... all three models, every forecast hour of the current run").

Extracted in models.py's worker WHILE the decoded grids are in memory - nothing is downloaded for it - and written as
ONE compact JSON per model run next to the run's WebP frames:

    assets/models/data/<model>/<cycle>/points.json          (models.yml: surface fields)
    assets/models/data/hrrr/<cycle>/points_ecape.json       (ecape.yml: ECAPE / CAPE, from its own WebP output)

Why one file per run and not per-station shards: the page needs it only when someone clicks a marker, and then wants
the next click to be instant; ~700 shards per run would be ~2,000 tiny files per cycle on the frames branch (tree churn
on every push, and gzip cannot compress a 600-byte file), while the whole run gzips to a few hundred KB.

Locating a point: the grid description the page already uses (manifest "grids") is inverted analytically -
spherical Lambert conformal for HRRR/RRFS, rotated lat-lon for RDPS/RAQDPS - to a FRACTIONAL grid index (fi, fj),
fj counted from the SOUTHERN row (the decoded arrays' row 0; the WebP files are flipped, row 0 north). Checked against
eccodes' own latitude/longitude arrays (`python scripts/models/points.py check`).

Sampling: BILINEAR for continuous fields (temperature, dewpoint, wind components, pressure, shortwave, smoke,
accumulations); NEAREST cell for fields where an average of neighbours is not a value the model produced (ceiling,
visibility, reflectivity with its precipitation type, CAPE/ECAPE whose skipped columns are flagged). A point outside
the grid (or within half a cell of its edge) gets nulls.

File format (all values integers in the stated unit, null = missing / none; series follow `hours`):
    {"v":1, "model":"hrrr", "cycle":"2026100118", "hours":[0,1,...], "ids":["KATL",...],
     "fields": {"t2m": {"unit":"F", "scale":0.1, "method":"bilinear", "d":[[...per id...], ...]}, ...}}
value = integer * scale. "d" is id-major: d[k] is the series for ids[k]; a series that is null in every hour is stored as
a bare null and one that is zero in every hour as a bare 0. `delta: true` on a field means the series is
stored as its first value followed by hour-to-hour differences (cumulative sum restores it; nulls break the chain and
the value after a null is absolute) - used for the smooth fields, where it roughly halves the gzipped size.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SITE = HERE.parent.parent
AIRPORTS = SITE / "data" / "models_airports.json"
CITIES = SITE / "data" / "models_cities.json"
D2R = math.pi / 180.0
# the HRRR's (and RRFS's) 3 km CONUS grid as models.py's grid_desc reads it - the ECAPE job uses it when it has no manifest
HRRR_GRID = dict(type="lcc", nx=1799, ny=1059, lat1=38.5, lat2=38.5, lon0=-97.5, la1=21.138123, lo1=-122.719528,
                 dx=3000.0, dy=3000.0, R=6371229.0)


# ── projection: lat/lon -> fractional grid index ────────────────────────────────────────────────────────────────────
def lcc_ij(g, lat, lon):
    """Spherical Lambert conformal (the HRRR/RRFS grid, R = 6371229 m; the page's shader inverts the same formulas)."""
    lat = np.asarray(lat, float) * D2R
    lon = np.asarray(lon, float)
    l1, l2 = g["lat1"] * D2R, g["lat2"] * D2R
    n = math.sin(l1) if abs(l1 - l2) < 1e-9 else (
        math.log(math.cos(l1) / math.cos(l2)) / math.log(math.tan(math.pi / 4 + l2 / 2) / math.tan(math.pi / 4 + l1 / 2)))
    F = math.cos(l1) * math.tan(math.pi / 4 + l1 / 2) ** n / n
    R = g.get("R", 6371229.0)

    def xy(la, lo):
        rho = R * F / np.tan(np.pi / 4 + la / 2) ** n
        th = n * (((lo - g["lon0"] + 180) % 360) - 180) * D2R
        return rho * np.sin(th), -rho * np.cos(th)
    x, y = xy(lat, lon)
    x1, y1 = xy(np.array(g["la1"] * D2R), np.array(g["lo1"]))
    return (x - x1) / g["dx"] - g.get("i0", 0), (y - y1) / g["dy"] - g.get("j0", 0)


def rot_ll(g, lat, lon):
    """Geographic -> rotated lat/lon (degrees) for a GRIB rotated_ll grid with its southern pole at (splat, splon)."""
    la, lo = np.asarray(lat, float) * D2R, (np.asarray(lon, float) - g["splon"]) * D2R
    th = (90.0 + g["splat"]) * D2R
    x, y, z = np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)
    x2 = math.cos(th) * x + math.sin(th) * z
    z2 = -math.sin(th) * x + math.cos(th) * z
    return np.arcsin(np.clip(z2, -1, 1)) / D2R, np.arctan2(y, x2) / D2R


def rotll_ij(g, lat, lon):
    rla, rlo = rot_ll(g, lat, lon)
    fi = (((rlo - g["lo1"]) + 180) % 360 - 180) / g["di"] - g.get("i0", 0)
    fj = (rla - g["la1"]) / g["dj"] - g.get("j0", 0)
    return fi, fj


def grid_ij(g, lat, lon):
    return (lcc_ij if g["type"] == "lcc" else rotll_ij)(g, lat, lon)


# ── samplers ────────────────────────────────────────────────────────────────────────────────────────────────────────
class Sampler:
    """Precomputed indices for one grid: bilinear corners + weights, and the nearest cell."""

    def __init__(self, g, lat, lon):
        fi, fj = grid_ij(g, lat, lon)
        nx, ny = int(g["nx"]), int(g["ny"])
        self.inside = (fi >= 0) & (fj >= 0) & (fi <= nx - 1) & (fj <= ny - 1) & np.isfinite(fi) & np.isfinite(fj)
        fi = np.where(self.inside, fi, 0.0)
        fj = np.where(self.inside, fj, 0.0)
        i0 = np.clip(np.floor(fi).astype(int), 0, max(nx - 2, 0))
        j0 = np.clip(np.floor(fj).astype(int), 0, max(ny - 2, 0))
        self.i0, self.j0 = i0, j0
        self.wx, self.wy = fi - i0, fj - j0
        self.ni = np.clip(np.rint(fi).astype(int), 0, nx - 1)
        self.nj = np.clip(np.rint(fj).astype(int), 0, ny - 1)
        self.fi, self.fj = fi, fj

    def bilinear(self, a):
        a = np.asarray(a, np.float64)
        i0, j0, wx, wy = self.i0, self.j0, self.wx, self.wy
        v = ((1 - wx) * (1 - wy) * a[j0, i0] + wx * (1 - wy) * a[j0, i0 + 1]
             + (1 - wx) * wy * a[j0 + 1, i0] + wx * wy * a[j0 + 1, i0 + 1])
        return np.where(self.inside, v, np.nan).astype(np.float32)

    def nearest(self, a):
        v = np.asarray(a)[self.nj, self.ni]
        if v.dtype.kind == "f":
            return np.where(self.inside, v, np.nan).astype(np.float32)
        return np.where(self.inside, v, 255).astype(np.uint8)


# ── the point list ──────────────────────────────────────────────────────────────────────────────────────────────────
def load_points():
    """-> (ids, lat, lon): every airport on the map, then the degree-day tracker's cities that are not airports."""
    ids, la, lo = [], [], []
    for path in (AIRPORTS, CITIES):
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        f = d["fields"]
        key = "airports" if "airports" in d else "cities"
        for r in d[key]:
            r = dict(zip(f, r))
            if r["icao"] in ids:
                continue
            ids.append(r["icao"])
            la.append(float(r["lat"]))
            lo.append(float(r["lon"]))
    return ids, np.array(la), np.array(lo)


# ── encoding ────────────────────────────────────────────────────────────────────────────────────────────────────────
# name -> unit, scale (value = int * scale), sampling, delta-coded
SPEC = {
    "t2m": ("F", 0.1, "bilinear", True),
    "td2m": ("F", 0.1, "bilinear", True),
    "ws80": ("m/s", 0.1, "bilinear (u, v)", True),
    "wd80": ("deg from", 5, "bilinear (u, v)", False),
    "ws10": ("m/s", 0.1, "bilinear (u, v)", True),        # 10 m wind and surface gust (user 2026-10-04)
    "wd10": ("deg from", 5, "bilinear (u, v)", False),
    "gust": ("m/s", 0.1, "bilinear", True),
    "ceil": ("ft AGL", 10, "nearest", False),        # null = no ceiling
    "vis": ("mi", 0.1, "nearest", False),            # capped at 10 (= 10 mi or more, as a METAR reports it)
    "refl": ("dBZ", 1, "nearest", False),            # HRRR/RRFS: REFD 1 km AGL; null below 5 dBZ
    "prate": ("mm/h", 0.1, "nearest", False),        # RDPS (no reflectivity): precipitation rate
    "ptype": ("code", 1, "nearest", False),          # 0 none, 1 rain, 2 snow, 3 FZRA, 4 sleet, 5 rain-snow mix
    "smoke": ("mg/m2", 0.1, "bilinear", False),
    "sw": ("W/m2", 5, "bilinear", False),
    "mslp": ("hPa", 0.1, "bilinear", True),
    "apcp": ("in", 0.01, "bilinear", True),          # accumulated since the run started; hourly = difference
    "asnow": ("in", 0.1, "bilinear", True),
    "afzra": ("in", 0.01, "bilinear", True),
    "aip": ("in", 0.01, "bilinear", True),
    "ecape_mu": ("J/kg", 1, "nearest", False),
    "ecape_ml": ("J/kg", 1, "nearest", False),
    "mucape": ("J/kg", 1, "nearest", False),
    "mlcape": ("J/kg", 1, "nearest", False),
}
ORDER = list(SPEC)


def quantise(name, x):
    """float series (nan = missing) -> list of int-or-None, delta-coded where the spec says so."""
    unit, scale, _, delta = SPEC[name]
    x = np.asarray(x, float)
    q = [None if not np.isfinite(v) else int(round(v / scale)) for v in x]
    if not delta:
        return q
    out, prev = [], None
    for v in q:
        if v is None:
            out.append(None)
            prev = None
        elif prev is None:
            out.append(v)
            prev = v
        else:
            out.append(v - prev)
            prev = v
    return out


def dequantise(name, q):
    """The inverse, as the page does it (for tests)."""
    unit, scale, _, delta = SPEC[name]
    out, prev = [], None
    for v in q:
        if v is None:
            out.append(np.nan)
            prev = None
            continue
        if delta and prev is not None:
            v = prev + v
        prev = v
        out.append(v * scale)
    return np.array(out)


def _compact(q):
    """A series that is all null is stored as null, one that is all zero as 0 (dry hours, no echo, no smoke)."""
    if all(v is None for v in q):
        return None
    if all(v == 0 for v in q):
        return 0
    return q


def expand(d, n):
    """The page's inverse of _compact."""
    return [None] * n if d is None else [0] * n if d == 0 else d


def build(model, cycle, hours, ids, series, extra=None):
    """series: {name: array (npoints, nhours) float, nan = missing}. Points with no data at all in every field are
    dropped from the file (outside the model's grid)."""
    names = [n for n in ORDER if n in series]
    have = np.zeros(len(ids), bool)
    for n in names:
        have |= np.isfinite(series[n]).any(axis=1)
    keep = np.flatnonzero(have)
    fields = {}
    for n in names:
        unit, scale, method, delta = SPEC[n]
        e = dict(unit=unit, scale=scale, method=method, d=[_compact(quantise(n, series[n][k])) for k in keep])
        if delta:
            e["delta"] = True
        fields[n] = e
    doc = dict(v=1, model=model, cycle=cycle, hours=list(map(int, hours)), ids=[ids[k] for k in keep], fields=fields)
    if extra:
        doc.update(extra)
    return doc


def dumps(doc):
    return json.dumps(doc, separators=(",", ":"))


# ── self-check against eccodes ──────────────────────────────────────────────────────────────────────────────────────
def _check(pkl):
    import pickle
    import sys
    sys.path.insert(0, str(HERE))
    import models as M
    blobs = pickle.load(open(pkl, "rb"))
    for name, buf in blobs.items():
        g = M.grid_desc(buf)
        lat, lon, _ = M.grid_latlon(buf)
        lon = ((lon + 180) % 360) - 180
        rng = np.random.default_rng(0)
        jj = rng.integers(0, g["ny"], 2000)
        ii = rng.integers(0, g["nx"], 2000)
        fi, fj = grid_ij(g, lat[jj, ii], lon[jj, ii])
        err = np.hypot(fi - ii, fj - jj)
        print(f"{name}: {g['type']} max index error {err.max():.4f} cells, median {np.median(err):.5f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "check":
        _check(sys.argv[2])
