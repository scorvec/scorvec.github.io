#!/usr/bin/env python3
"""Vector boundaries for models.html: coastlines, lakes, country borders, US states, Canadian and Mexican
provinces and US counties, as compact delta-encoded polylines in lon/lat (1e-3 degree units, ~100 m) so the page
can project them into whatever map projection it draws and stroke them crisply at every zoom.

Sources (read once; the output is static and only rebuilt when missing):
  Natural Earth 10m coastline, lakes, admin-0 boundary lines, admin-1 lines (CAN/MEX) - public domain,
      raw.githubusercontent.com/nvkelso/natural-earth-vector (jsDelivr refuses files over 20 MB)
  us-atlas@3 counties-10m (US Census cartographic boundaries 1:10M) - states and counties, via jsDelivr npm

    python scripts/models/build_geo.py --out assets/models/geo.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/{}.geojson"
USATLAS = "https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json"
BOX = (-179.9, -40.0, 10.0, 84.0)          # lon0, lon1, lat0, lat1: everything any of the three models covers
Q = 1000                                   # 1e-3 degree


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "scorvec-models/1.0"}),
                                        timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:                                    # noqa: BLE001
            if k == 3:
                raise
            print(f"  retry {url}: {e}")
            time.sleep(10 * (k + 1))


def inside(lon, lat):
    return BOX[0] <= lon <= BOX[1] and BOX[2] <= lat <= BOX[3]


def clip(line):
    """Split a polyline into the runs that fall inside BOX (one point of margin kept so edges meet the frame)."""
    out, cur = [], []
    for i, (lo, la) in enumerate(line):
        if inside(lo, la):
            if not cur and i > 0:
                cur.append(line[i - 1])
            cur.append((lo, la))
        elif cur:
            cur.append((lo, la))
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [c for c in out if len(c) >= 2]


def simplify(pts, tol):
    """Douglas-Peucker in degrees (iterative)."""
    if len(pts) < 3 or tol <= 0:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (x1, y1), (x2, y2) = pts[a], pts[b]
        dx, dy = x2 - x1, y2 - y1
        n2 = dx * dx + dy * dy
        best, bi = -1.0, -1
        for i in range(a + 1, b):
            x, y = pts[i]
            if n2 == 0:
                d = (x - x1) ** 2 + (y - y1) ** 2
            else:
                t = ((x - x1) * dx + (y - y1) * dy) / n2
                t = min(1.0, max(0.0, t))
                d = (x - x1 - t * dx) ** 2 + (y - y1 - t * dy) ** 2
            if d > best:
                best, bi = d, i
        if bi > 0 and best > tol * tol:
            keep[bi] = True
            stack += [(a, bi), (bi, b)]
    return [p for p, k in zip(pts, keep) if k]


def encode(lines, tol):
    """[[x0, y0, dx1, dy1, ...], ...] in 1e-3 degree integers, consecutive duplicates dropped."""
    enc, npts = [], 0
    for ln in lines:
        for part in clip(ln):
            part = simplify(part, tol)
            q = []
            for lo, la in part:
                p = (round(lo * Q), round(la * Q))
                if not q or q[-1] != p:
                    q.append(p)
            if len(q) < 2:
                continue
            flat = [q[0][0], q[0][1]]
            for (x0, y0), (x1, y1) in zip(q, q[1:]):
                flat += [x1 - x0, y1 - y0]
            enc.append(flat)
            npts += len(q)
    return enc, npts


def geo_lines(fc, keep=lambda p: True):
    out = []
    for f in fc["features"]:
        if not keep(f.get("properties") or {}):
            continue
        g = f["geometry"]
        if g is None:
            continue
        t, c = g["type"], g["coordinates"]
        if t == "LineString":
            out.append(c)
        elif t == "MultiLineString":
            out += c
        elif t == "Polygon":
            out += c
        elif t == "MultiPolygon":
            for poly in c:
                out += poly
    return [[(float(x), float(y)) for x, y, *_ in ln] for ln in out]


def topo_arcs(topo):
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * sx + tx, y * sy + ty))
        arcs.append(pts)
    return arcs


def topo_used(obj):
    """Indices of every arc referenced by a TopoJSON GeometryCollection (each shared border once)."""
    used = set()

    def walk(a):
        if isinstance(a, list):
            for b in a:
                walk(b)
        else:
            used.add(a if a >= 0 else ~a)
    for g in obj["geometries"]:
        walk(g.get("arcs", []))
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/models/geo.json")
    a = ap.parse_args()
    layers = {}
    print("Natural Earth ...")
    coast = geo_lines(get(NE.format("ne_10m_coastline")))
    lakes = geo_lines(get(NE.format("ne_10m_lakes")), keep=lambda p: (p.get("scalerank", p.get("SCALERANK")) or 9) <= 5)
    ctry = geo_lines(get(NE.format("ne_10m_admin_0_boundary_lines_land")))
    prov = geo_lines(get(NE.format("ne_10m_admin_1_states_provinces_lines")),
                     keep=lambda p: (p.get("ADM0_A3") or p.get("adm0_a3")) in ("CAN", "MEX"))
    print("us-atlas ...")
    topo = get(USATLAS)
    arcs = topo_arcs(topo)
    st = topo_used(topo["objects"]["states"])
    co = topo_used(topo["objects"]["counties"]) - st
    # (name, lines, simplification tolerance in degrees, minimum zoom in km per screen pixel - the page draws a layer
    # only when zoomed in past it; 0 = always)
    spec = [("coast", coast, 0.004, 0), ("lakes", lakes, 0.004, 0), ("countries", ctry, 0.004, 0),
            ("provinces", prov, 0.004, 0), ("states", [arcs[i] for i in sorted(st)], 0.003, 0),
            ("counties", [arcs[i] for i in sorted(co)], 0.003, 2.2)]
    for name, lines, tol, kpp in spec:
        enc, n = encode(lines, tol)
        layers[name] = dict(q=Q, maxKmPerPx=kpp or None, lines=enc)
        print(f"  {name}: {len(enc)} lines, {n} points")
    src = "Natural Earth 10m (public domain); US states and counties: us-atlas 3 / US Census 1:10M"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # two files: the base map (first paint) and the counties (fetched after the first frame is up)
    for path, names in ((out, [n for n in layers if n != "counties"]),
                        (out.with_name(out.stem + "_counties.json"), ["counties"])):
        path.write_text(json.dumps(dict(sources=src, layers={n: layers[n] for n in names}), separators=(",", ":")))
        print(f"wrote {path}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
