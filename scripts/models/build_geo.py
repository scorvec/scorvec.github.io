#!/usr/bin/env python3
"""Vector boundaries for models.html: coastlines, lakes, country borders, US states, Canadian and Mexican
provinces and US counties, as compact delta-encoded polylines in lon/lat (1e-3 degree units, ~100 m) so the page
can project them into whatever map projection it draws and stroke them crisply at every zoom.

Sources (read once; the output is static and only rebuilt when missing):
  Natural Earth 10m coastline, lakes, admin-0 boundary lines, admin-1 lines (CAN/MEX) - public domain,
      raw.githubusercontent.com/nvkelso/natural-earth-vector (jsDelivr refuses files over 20 MB)
  us-atlas@3 counties-10m (US Census cartographic boundaries 1:10M) - states and counties, via jsDelivr npm

Three files: geo.json (the base map), geo_counties.json (drawn only zoomed in) and geo_coarse.json, the base map
already simplified to the level the page draws at the national view (COARSE_KM), which the page paints first and
replaces with geo.json once the first map frame is up. The coarse file is derived from geo.json alone (no download),
with the page's own projection and Douglas-Peucker, so its lines are the points the page keeps from geo.json at that
tolerance:

    python scripts/models/build_geo.py --out assets/models/geo.json                 # all three (downloads)
    python scripts/models/build_geo.py --out assets/models/geo.json --coarse-only   # geo_coarse.json from geo.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
import urllib.request
from pathlib import Path

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/{}.geojson"
USATLAS = "https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json"
BOX = (-179.9, -40.0, 10.0, 84.0)  # lon0, lon1, lat0, lat1: everything any of the three models covers
Q = 1000  # 1e-3 degree


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "scorvec-models/1.0"}), timeout=120
            ) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
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
        for raw_part in clip(ln):
            part = simplify(raw_part, tol)
            q = []
            for lo, la in part:
                p = (round(lo * Q), round(la * Q))
                if not q or q[-1] != p:
                    q.append(p)
            if len(q) < 2:
                continue
            flat = [q[0][0], q[0][1]]
            for (x0, y0), (x1, y1) in itertools.pairwise(q):
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
        elif t == "MultiLineString" or t == "Polygon":
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


# ── the coarse first-paint file ─────────────────────────────────────────────────────────────────────────────────────
# models.html projects every line to its display projection (the HRRR's Lambert conformal, tangent at 38.5 N 97.5 W,
# sphere 6371229 m; km) and simplifies each layer at 0.1/0.3/0.9/2.7 km (worker geoProcess/simplifyLine). The national
# view draws the 2.7 km level on phones (0.6 device px), so that level is shipped on its own: same projection, same
# Douglas-Peucker (same distances, same first-maximum tie-break), same rule dropping lines smaller than the tolerance,
# and the kept points are the ORIGINAL quantised vertices, re-delta-coded - nothing is moved.
COARSE_KM = 2.7


def page_projection():
    """(n, RF, rho0, lon0) exactly as models.html computes P."""
    d2r = math.pi / 180
    R, phi, lon0 = 6371229, 38.5 * d2r, -97.5 * d2r
    n, t = math.sin(phi), math.tan(math.pi / 4 + phi / 2)
    F = math.cos(phi) * math.pow(t, n) / n
    RF = R * F
    return n, RF, RF / math.pow(t, n), lon0


def project_km(pts, q, P):
    """[(lon_int, lat_int)] -> [(x, y)] km, the worker's arithmetic in the same order."""
    n, RF, rho0, lon0 = P
    D = math.pi / 180
    out = []
    for x, y in pts:
        rho = RF / math.pow(math.tan(math.pi / 4 + y / q * D / 2), n)
        dl = x / q * D - lon0
        dl = math.atan2(math.sin(dl), math.cos(dl))
        out.append((rho * math.sin(n * dl) / 1000, (rho0 - rho * math.cos(n * dl)) / 1000))
    return out


def dp_keep(xy, tol):
    """The page's simplifyLine: indices kept by Douglas-Peucker at `tol` km."""
    m = len(xy)
    keep = [False] * m
    keep[0] = keep[m - 1] = True
    st, t2 = [0, m - 1], tol * tol
    while st:
        j = st.pop()
        i = st.pop()
        x0, y0 = xy[i]
        dx, dy = xy[j][0] - x0, xy[j][1] - y0
        L2, best, bd = dx * dx + dy * dy, -1, t2
        for k in range(i + 1, j):
            px, py = xy[k][0] - x0, xy[k][1] - y0
            if L2 > 0:
                cr = px * dy - py * dx
                d = cr * cr / L2
                tt = (px * dx + py * dy) / L2
                if tt < 0:
                    d = px * px + py * py
                elif tt > 1:
                    ex, ey = px - dx, py - dy
                    d = ex * ex + ey * ey
            else:
                d = px * px + py * py
            if d > bd:
                bd, best = d, k
        if best > 0:
            keep[best] = True
            st += [i, best, best, j]
    return [k for k in range(m) if keep[k]]


def coarse_doc(doc, tol=COARSE_KM):
    """geo.json document -> geo_coarse.json document (same format, every layer simplified at `tol` km)."""
    P = page_projection()
    layers = {}
    for name, L in doc["layers"].items():
        q, lines = L["q"], []
        for ln in L["lines"]:
            x, y, pts = ln[0], ln[1], []
            for k in range(len(ln) // 2):
                if k:
                    x += ln[2 * k]
                    y += ln[2 * k + 1]
                pts.append((x, y))
            xy = project_km(pts, q, P)
            xs, ys = [p[0] for p in xy], [p[1] for p in xy]
            if max(xs) - min(xs) < tol and max(ys) - min(ys) < tol:  # smaller than the tolerance: the page skips it
                continue
            kept = [pts[k] for k in dp_keep(xy, tol)]
            flat = [kept[0][0], kept[0][1]]
            for (x0, y0), (x1, y1) in itertools.pairwise(kept):
                flat += [x1 - x0, y1 - y0]
            lines.append(flat)
        layers[name] = dict(L, lines=lines)
    return dict(sources=doc.get("sources", ""), simplified_km=tol, layers=layers)


def write_coarse(full_path):
    full_path = Path(full_path)
    doc = json.loads(full_path.read_text())
    cdoc = coarse_doc(doc)
    path = full_path.with_name(full_path.stem + "_coarse.json")
    path.write_text(json.dumps(cdoc, separators=(",", ":")))
    n0 = sum(len(ln) // 2 for L in doc["layers"].values() for ln in L["lines"])
    n1 = sum(len(ln) // 2 for L in cdoc["layers"].values() for ln in L["lines"])
    print(f"wrote {path}: {path.stat().st_size / 1e6:.3f} MB, {n1} of {n0} points at {COARSE_KM} km")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/models/geo.json")
    ap.add_argument("--coarse-only", action="store_true", help="only derive <out>_coarse.json from an existing <out>")
    a = ap.parse_args()
    if a.coarse_only:
        write_coarse(a.out)
        return
    layers = {}
    print("Natural Earth ...")
    coast = geo_lines(get(NE.format("ne_10m_coastline")))
    lakes = geo_lines(get(NE.format("ne_10m_lakes")), keep=lambda p: (p.get("scalerank", p.get("SCALERANK")) or 9) <= 5)
    ctry = geo_lines(get(NE.format("ne_10m_admin_0_boundary_lines_land")))
    prov = geo_lines(
        get(NE.format("ne_10m_admin_1_states_provinces_lines")),
        keep=lambda p: (p.get("ADM0_A3") or p.get("adm0_a3")) in ("CAN", "MEX"),
    )
    print("us-atlas ...")
    topo = get(USATLAS)
    arcs = topo_arcs(topo)
    st = topo_used(topo["objects"]["states"])
    co = topo_used(topo["objects"]["counties"]) - st
    # (name, lines, simplification tolerance in degrees, minimum zoom in km per screen pixel - the page draws a layer
    # only when zoomed in past it; 0 = always)
    spec = [
        ("coast", coast, 0.004, 0),
        ("lakes", lakes, 0.004, 0),
        ("countries", ctry, 0.004, 0),
        ("provinces", prov, 0.004, 0),
        ("states", [arcs[i] for i in sorted(st)], 0.003, 0),
        ("counties", [arcs[i] for i in sorted(co)], 0.003, 2.2),
    ]
    for name, lines, tol, kpp in spec:
        enc, n = encode(lines, tol)
        layers[name] = dict(q=Q, maxKmPerPx=kpp or None, lines=enc)
        print(f"  {name}: {len(enc)} lines, {n} points")
    src = "Natural Earth 10m (public domain); US states and counties: us-atlas 3 / US Census 1:10M"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # two files: the base map (first paint) and the counties (fetched after the first frame is up)
    for path, names in (
        (out, [n for n in layers if n != "counties"]),
        (out.with_name(out.stem + "_counties.json"), ["counties"]),
    ):
        path.write_text(json.dumps(dict(sources=src, layers={n: layers[n] for n in names}), separators=(",", ":")))
        print(f"wrote {path}: {path.stat().st_size / 1e6:.2f} MB")
    write_coarse(out)  # from the file just written: the same path as --coarse-only


if __name__ == "__main__":
    main()
