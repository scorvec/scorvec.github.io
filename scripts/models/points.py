#!/usr/bin/env python3
"""Point time series for models.html meteograms, and the projection math that finds a point on a model grid.

Clicking an airport shows every model's series for every forecast hour of the current run.

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

The same content, addressable per station, is written beside it (points.bin, points_ecape.bin), so a click reads one
station with an HTTP Range request instead of the whole run (points.json stays for older pages and the airport labels
on the temperature map, which want every station at once):

    b"PTB1" | uint32 LE n | index = zlib(JSON, n bytes) | record 0 | record 1 | ...
    index : every key of points.json except "ids" and the fields' "d" (so v, model, cycle, init, hours, fields with
            unit/scale/method/delta, ...), plus "ids" and "off": record k is bytes [8 + n + off[k], 8 + n + off[k+1])
    record: zlib(JSON {field: that station's "d" entry exactly as in points.json (list, null or 0)})

The manifest lists it next to points.json as "points_bin": {file, bytes, head = 8 + n, n}, so the page reads the head in
one request. `python scripts/models/points.py parity <points.json> [<points.bin>]` checks that every station and field
decodes to the same series from both.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SITE = HERE.parent.parent
AIRPORTS = SITE / "data" / "models_airports.json"
CITIES = SITE / "data" / "models_cities.json"
D2R = math.pi / 180.0
# the HRRR's (and RRFS's) 3 km CONUS grid as models.py's grid_desc reads it - the ECAPE job uses it when it has no manifest
HRRR_GRID = dict(
    type="lcc",
    nx=1799,
    ny=1059,
    lat1=38.5,
    lat2=38.5,
    lon0=-97.5,
    la1=21.138123,
    lo1=-122.719528,
    dx=3000.0,
    dy=3000.0,
    R=6371229.0,
)


# ── projection: lat/lon -> fractional grid index ────────────────────────────────────────────────────────────────────
def lcc_ij(g, lat, lon):
    """Spherical Lambert conformal (the HRRR/RRFS grid, R = 6371229 m; the page's shader inverts the same formulas)."""
    lat = np.asarray(lat, float) * D2R
    lon = np.asarray(lon, float)
    l1, l2 = g["lat1"] * D2R, g["lat2"] * D2R
    n = (
        math.sin(l1)
        if abs(l1 - l2) < 1e-9
        else (
            math.log(math.cos(l1) / math.cos(l2))
            / math.log(math.tan(math.pi / 4 + l2 / 2) / math.tan(math.pi / 4 + l1 / 2))
        )
    )
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
        v = (
            (1 - wx) * (1 - wy) * a[j0, i0]
            + wx * (1 - wy) * a[j0, i0 + 1]
            + (1 - wx) * wy * a[j0 + 1, i0]
            + wx * wy * a[j0 + 1, i0 + 1]
        )
        return np.where(self.inside, v, np.nan).astype(np.float32)

    def nearest(self, a):
        v = np.asarray(a)[self.nj, self.ni]
        if v.dtype.kind == "f":
            return np.where(self.inside, v, np.nan).astype(np.float32)
        return np.where(self.inside, v, 255).astype(np.uint8)


# ── the point list ──────────────────────────────────────────────────────────────────────────────────────────────────
def load_points():
    """-> (ids, lat, lon): every airport on the map, then the extra cities that are not airports."""
    ids, la, lo = [], [], []
    for path in (AIRPORTS, CITIES):
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        f = d["fields"]
        key = "airports" if "airports" in d else "cities"
        for row in d[key]:
            r = dict(zip(f, row))
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
    "ws10": ("m/s", 0.1, "bilinear (u, v)", True),  # 10 m wind and surface gust
    "wd10": ("deg from", 5, "bilinear (u, v)", False),
    "gust": ("m/s", 0.1, "bilinear", True),
    "ceil": ("ft AGL", 10, "nearest", False),  # null = no ceiling
    "vis": ("mi", 0.1, "nearest", False),  # capped at 10 (= 10 mi or more, as a METAR reports it)
    "refl": ("dBZ", 1, "nearest", False),  # HRRR/RRFS: REFD 1 km AGL; null below 5 dBZ
    "prate": ("mm/h", 0.1, "nearest", False),  # RDPS (no reflectivity): precipitation rate
    "ptype": ("code", 1, "nearest", False),  # 0 none, 1 rain, 2 snow, 3 FZRA, 4 sleet, 5 rain-snow mix
    "smoke": ("mg/m2", 0.1, "bilinear", False),
    "sw": ("W/m2", 5, "bilinear", False),
    "mslp": ("hPa", 0.1, "bilinear", True),
    "apcp": ("in", 0.01, "bilinear", True),  # accumulated since the run started; hourly = difference
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
    _unit, scale, _, delta = SPEC[name]
    x = np.asarray(x, float)
    q = [None if not np.isfinite(v) else int(np.rint(v / scale)) for v in x]
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
    _unit, scale, _, delta = SPEC[name]
    out, prev = [], None
    for qv in q:
        if qv is None:
            out.append(np.nan)
            prev = None
            continue
        v = prev + qv if delta and prev is not None else qv
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


def bin_bytes(doc):
    """points.json document -> (points.bin bytes, head length): one zlib record per station behind a zlib index."""
    recs = [
        zlib.compress(dumps({n: e["d"][k] for n, e in doc["fields"].items()}).encode(), 9)
        for k in range(len(doc["ids"]))
    ]
    idx = {key: v for key, v in doc.items() if key not in ("ids", "fields")}
    idx["fields"] = {n: {key: v for key, v in e.items() if key != "d"} for n, e in doc["fields"].items()}
    off = [0]
    for r in recs:
        off.append(off[-1] + len(r))
    idx.update(ids=doc["ids"], off=off)
    head = zlib.compress(dumps(idx).encode(), 9)
    return b"PTB1" + struct.pack("<I", len(head)) + head + b"".join(recs), 8 + len(head)


def bin_read(b):
    """points.bin bytes -> a points.json-shaped document (the page's reading of it, for checks)."""
    if b[:4] != b"PTB1":
        raise ValueError("not a PTB1 file")
    n = struct.unpack("<I", b[4:8])[0]
    idx = json.loads(zlib.decompress(b[8 : 8 + n]))
    base, off = 8 + n, idx.pop("off")
    recs = [json.loads(zlib.decompress(b[base + off[k] : base + off[k + 1]])) for k in range(len(idx["ids"]))]
    doc = dict(idx)
    doc["fields"] = {f: dict(e, d=[r[f] for r in recs]) for f, e in idx["fields"].items()}
    return doc


def parity(json_path, bin_path=None):
    """Every station and field of points.json against points.bin: the stored entries and the decoded series. -> problems"""
    jp = Path(json_path)
    bp = Path(bin_path) if bin_path else jp.with_suffix(".bin")
    a, b = json.loads(jp.read_text()), bin_read(bp.read_bytes())
    bad = [
        f"top-level {key!r} differs" for key in sorted(set(a) | set(b)) if key != "fields" and a.get(key) != b.get(key)
    ]
    if a["ids"] != b["ids"] or list(a["fields"]) != list(b["fields"]):
        bad.append("station or field list differs")
        return bad
    n, nser = len(a["hours"]), 0
    for f, ea in a["fields"].items():
        eb = b["fields"][f]
        if {k: v for k, v in ea.items() if k != "d"} != {k: v for k, v in eb.items() if k != "d"}:
            bad.append(f"{f}: field metadata differs")
        for k, sid in enumerate(a["ids"]):
            da, db = ea["d"][k], eb["d"][k]
            xa, xb = _decode(ea, da, n), _decode(eb, db, n)
            if da != db or not np.array_equal(xa, xb, equal_nan=True):
                bad.append(f"{f} {sid}: series differ")
            nser += 1
    print(
        f"{jp.name} vs {bp.name}: {len(a['ids'])} stations x {len(a['fields'])} fields = {nser} series, "
        f"{len(bad)} differences; {jp.stat().st_size / 1e3:.0f} KB json, {bp.stat().st_size / 1e3:.0f} KB bin"
    )
    return bad


def _decode(e, d, n):
    """One stored series -> values, as the page decodes it (mgSeries)."""
    out, prev = np.full(n, np.nan), None
    for k, xq in enumerate(expand(d, n)):
        if xq is None:
            prev = None
            continue
        x = prev + xq if e.get("delta") and prev is not None and d != 0 else xq
        prev = x
        out[k] = x * e["scale"]
    return out


# ── self-check against eccodes ──────────────────────────────────────────────────────────────────────────────────────
def _check(pkl):
    import pickle
    import sys

    sys.path.insert(0, str(HERE))
    import models as M

    with open(pkl, "rb") as fh:
        blobs = pickle.load(fh)
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
    elif len(sys.argv) > 2 and sys.argv[1] == "parity":
        problems = parity(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        for p in problems[:20]:
            print("  " + p)
        sys.exit(1 if problems else 0)
