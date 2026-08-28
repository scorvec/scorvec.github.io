#!/usr/bin/env python3
"""Global ensemble tropical-cyclone tracker from AIFS-ENS (+ IFS-ENS) MSLP.

Zero extra downloads: runs on the per-member surface files (sp/10u/10v/msl)
already pulled each cycle for the MSLP/wind animation — 51 AIFS members (and
50 IFS members when their surface file is cached), daily steps to day 15.

Method (per member):
  detect — MSLP local minima (0.25°) that are ≥ RING_DEPTH hPa deeper than the
           mean pressure on a ~5° ring (closed circulation), with the peak
           10 m wind within 3° recorded as intensity. Genesis gated to
           |lat| ≤ 35 and p < 1012 hPa; tracking follows storms to |lat| 60.
  link   — greedy nearest-candidate linking at 24 h spacing within a motion-
           extrapolated 800 km gate; tracks kept if ≥ 3 days long and peaking
           ≥ WIND_MIN m/s (10 m, grid-scale — well below true intensity).
  cluster— tracks from all members grouped into "storms" when their genesis
           points agree within 600 km / 2 days; clusters supported by ≥ 15%
           of members are labelled BASIN-n.

Output: weathernerds-style per-storm track panels (tc_storms.webp) and
per-day animation frames for three zoom regions x two models (AIFS/IFS),
light maps with 200-km strike probability accumulating through the shown day.
Detection runs in a C kernel (detect.c, auto-compiled) with a scipy fallback;
frames render in a process pool.

    python tc_tracker.py --date 20260718 --time 00 --out-dir ../../assets/tc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ecmwf"))
import store as ecmwf

R_EARTH = 6371.0                     # km
P_GATE = 1012.0                      # hPa: candidate ceiling at genesis
P_TRACK = 1016.0                     # hPa: ceiling while tracking (filling storms)
RING_DEG = 5.0                       # closed-low ring radius (deg)
RING_DEPTH = 2.0                     # hPa below ring mean required
LAT_GENESIS = 30.0                   # genesis latitude gate, summer hemisphere
LAT_GENESIS_WINTER = 20.0            # winter hemisphere: subtropical lows are not TCs
LAT_TRACK = 60.0                     # tracking latitude ceiling
LAT_ET = 42.0                        # truncate tracks here (extratropical transition proxy)
LINK_KMD = 900.0                     # linking gate, km per 24 h (motion-extrapolated)
GAP_H = 36                           # max hours between fixes before a track dies
MIN_SPAN_H = 48                      # minimum track lifetime (hours)
WIND_MIN = 15.5                      # m/s (~30 kt): track's peak 10 m wind to keep it
DEPTH_PEAK = 3.0                     # hPa: track's peak ring depth (env-relative —
                                     # absolute MSLP screens fail in heat-trough zones)
WIND_SHOW = 25 / 1.94384             # m/s: fixes below 25 kt are neither drawn nor
                                     # counted in strike probability (noise floor)
CLUSTER_KM = 600.0                   # genesis agreement radius
CLUSTER_DT = 2                       # genesis agreement window (days)
MIN_SUPPORT = 0.25                   # fraction of members that must develop the storm
STRIKE_KM = 200.0                    # strike-probability radius

BASINS = [  # (key, label, lon_w, lon_e, lat_s, lat_n)  lons in -180..180
    ("natl", "North Atlantic", -100, -20, 5, 50),
    ("epac", "East Pacific", -140, -75, 5, 35),
    ("cpac", "Central Pacific", -180, -140, 5, 35),
    ("wpac", "West Pacific", 100, 180, 5, 45),
    ("nio", "North Indian", 45, 100, 3, 30),
    ("sio", "South Indian", 35, 115, -35, -3),
    ("aus", "Australia / SW Pacific", 115, 180, -35, -3),
]


_LM = np.load(Path(__file__).parent / "land_mask_0p5.npz")
# land dilated by 2 cells (~110 km): genesis must be OPEN ocean, which kills
# semi-permanent coastal troughs (Panama Bight / lee-of-Andes Colombian low,
# gulf-coast thermal lows) that are technically over water
_NEAR_LAND = ndimage.binary_dilation(_LM["land"], iterations=2)

# no-TC zones (climatologically cyclone-free enclosed/marginal seas whose
# trough lows otherwise pass the MSLP gates): (lon_w, lon_e, lat_s, lat_n)
EXCLUDE = [(32, 44, 12, 30),     # Red Sea
           (46, 57, 22, 31),     # Persian Gulf
           (-6, 37, 29, 47),     # Mediterranean
           (42, 58, 8, 18),      # Gulf of Aden / Somali jet / Socotra zone (the
                                 # July monsoon trough re-seeds a fake "storm"
                                 # every few days; real Arabian Sea genesis
                                 # is east of ~60°E)
           (-78.5, -70.5, 8, 12.5),  # Colombian Caribbean coast (semi-permanent
                                     # lee-of-Andes trough; real Caribbean genesis
                                     # is north of ~12.5°N)
           (-81, -77, 2, 9)]     # Panama Bight / Colombian Pacific coast trough


SEASONAL_EXCLUDE = [
    # Gulf of Oman / Arabian & Makran coasts / Persian Gulf approaches: the
    # summer monsoon heat trough spawns broad non-tropical lows here Jun-Sep;
    # real Oman-coast TCs (Gonu, Shaheen) are pre/post-monsoon
    (48, 64, 16, 28, {6, 7, 8, 9}),
]
_MONTH = [0]                       # set from the init in main()
INIT_ISO = [""]                    # init timestamp for the Julia renderer


def excluded(lat, lon):
    if any(w <= lon <= e and s <= lat <= n for w, e, s, n in EXCLUDE):
        return True
    return any(w <= lon <= e and s <= lat <= n and _MONTH[0] in mo
               for w, e, s, n, mo in SEASONAL_EXCLUDE)


def _cell(lat, lon):
    i = int(round((lat - _LM["lat"][0]) / 0.5))
    j = int(round((((lon + 180) % 360) - 180 - _LM["lon"][0]) / 0.5)) % len(_LM["lon"])
    return i, j


def is_ocean(lat, lon):
    """Over water (0.5° Natural Earth mask, nearest cell)."""
    i, j = _cell(lat, lon)
    if i < 0 or i >= len(_LM["lat"]):
        return False
    return not bool(_LM["land"][i, j])


def seg_over_water(a, b):
    """Both endpoints AND the midpoint over water — long link jumps must not
    draw chords across mountain ranges/isthmuses."""
    mid_lon = a[2] + (((b[2] - a[2] + 180) % 360) - 180) / 2
    return (is_ocean(a[1], a[2]) and is_ocean(b[1], b[2])
            and is_ocean((a[1] + b[1]) / 2, ((mid_lon + 180) % 360) - 180))


def is_open_ocean(lat, lon):
    """≥ ~110 km from any coast — the genesis standard. Kills desert heat
    lows AND semi-permanent coastal troughs (Panama Bight, lee of the Andes)."""
    i, j = _cell(lat, lon)
    if i < 0 or i >= len(_LM["lat"]):
        return False
    return not bool(_NEAR_LAND[i, j])


def _zeta(u, v, lat, lon):
    """Relative vorticity from 10 m winds on the lat-lon grid (s^-1)."""
    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    a = 6.371e6
    cosp = np.cos(latr)[:, None]
    dvdx = np.gradient(v, lonr, axis=1) / (a * cosp)
    dudy = np.gradient(u, latr, axis=0) / a
    return dvdx - dudy


def refine_centroid(cands_k, zeta, lat, lon, rad_px=10):
    """Move each candidate to the cyclonic-vorticity centroid within ~2.5°:
    the rotation centre is far more stable than the MSLP-minimum gridpoint
    for weak/disorganised systems (cf. TRACK / TempestExtremes)."""
    out = []
    dlat = abs(float(lat[1] - lat[0]))
    for c in cands_k:
        la, lo = c[0], c[1]
        iy = int(round((lat[0] - la) / dlat))            # lat descending
        ix = int(round(((lo - float(lon[0])) % 360) / dlat))
        y0, y1 = max(0, iy - rad_px), min(len(lat), iy + rad_px + 1)
        xs = np.arange(ix - rad_px, ix + rad_px + 1) % len(lon)
        z = zeta[y0:y1][:, xs]
        w = np.maximum((z if la >= 0 else -z) - 3e-5, 0.0)   # cyclonic only
        if w.sum() <= 0:
            out.append(c); continue
        yy = lat[y0:y1][:, None] * np.ones_like(w)
        lon0 = lon[ix]
        xoff = (np.arange(-rad_px, rad_px + 1) * dlat)[None, :] * np.ones_like(w)
        cla = float((yy * w).sum() / w.sum())
        clo = float(lon0 + (xoff * w).sum() / w.sum())
        clo = ((clo + 180) % 360) - 180
        # never move further than the search box itself
        if abs(cla - la) <= rad_px * dlat + 0.01:
            out.append((cla, clo, c[2], c[3], c[4]))
        else:
            out.append(c)
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def open_members(date: str, time: str):
    """Locate every cached per-member surface file with msl for this cycle
    (AIFS cf/pf + IFS pf) and open msl/10u/10v. Never triggers a download —
    only cache-present files are used."""
    out = []
    cycle_dir = ecmwf.CACHE / f"{date}{time}z"
    import re as _re
    best = {}
    for sub in sorted(cycle_dir.glob("*")):
        if not sub.is_dir():
            continue
        for typ in ("cf", "pf"):
            for fp in sorted(sub.glob(f"{typ}_*msl*_sfc_*.grib2")):
                m = _re.search(r"x(\d+)\.grib2$", fp.name)
                nst = int(m.group(1)) if m else 0
                key = (sub.name.replace("-ens", ""), typ)
                if key not in best or nst > best[key][0]:
                    best[key] = (nst, fp)
    hits = [(k[0], k[1], v[1]) for k, v in sorted(best.items())]
    for model, typ, p in hits:
        try:
            msl = xr.open_dataset(p, backend_kwargs={"filter_by_keys": {"shortName": "msl"},
                                                     "indexpath": ""},
                                  engine="cfgrib", chunks={"number": 1})["msl"]
            u10 = xr.open_dataset(p, backend_kwargs={"filter_by_keys": {"shortName": "10u"},
                                                     "indexpath": ""},
                                  engine="cfgrib", chunks={"number": 1})["u10"]
            v10 = xr.open_dataset(p, backend_kwargs={"filter_by_keys": {"shortName": "10v"},
                                                     "indexpath": ""},
                                  engine="cfgrib", chunks={"number": 1})["v10"]
        except Exception as e:                                     # noqa: BLE001
            print(f"  {model} {typ}: unreadable ({str(e)[:60]}) — skipped")
            continue
        members = msl.number.values if "number" in msl.dims else [None]
        out.append((model, typ, p, msl, u10, v10, members))
        print(f"  {model} {typ}: {len(members)} member(s) from {Path(p).name}")
    return out


# ── C detection kernel (fused local-min + SAT ring + wind scan; ~10x scipy) ──
def _load_ckernel():
    import ctypes, subprocess
    src = Path(__file__).parent / "detect.c"
    lib = Path(__file__).parent / "detect.dylib"
    try:
        if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-o", str(lib), str(src)],
                           check=True, capture_output=True)
        dl = ctypes.CDLL(str(lib))
        f = dl.detect_step
        f.restype = ctypes.c_int
        F = ctypes.POINTER(ctypes.c_float); I = ctypes.POINTER(ctypes.c_int)
        f.argtypes = [F, F, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                      ctypes.c_int, I, I, F, F, F, ctypes.c_int]
        return f
    except Exception as e:                                     # noqa: BLE001
        print(f"  C kernel unavailable ({str(e)[:60]}) — scipy fallback")
        return None


_CKERNEL = _load_ckernel()
_MAX_CAND = 4096


def _detect_c(msl_hpa, wind, lat, ring_px):
    """C-kernel candidate detection for all steps of one member."""
    import ctypes
    F = ctypes.POINTER(ctypes.c_float)
    nstep, nlat, nlon = msl_hpa.shape
    band = np.where(np.abs(lat) <= LAT_TRACK)[0]
    iy0, iy1 = int(band[0]), int(band[-1] + 1)
    oy = np.empty(_MAX_CAND, np.int32); ox = np.empty(_MAX_CAND, np.int32)
    op = np.empty(_MAX_CAND, np.float32); od = np.empty(_MAX_CAND, np.float32)
    ow = np.empty(_MAX_CAND, np.float32)
    lon_vals = None
    cands = [[] for _ in range(nstep)]
    for k in range(nstep):
        f = np.ascontiguousarray(msl_hpa[k], np.float32)
        w = np.ascontiguousarray(wind[k], np.float32)
        n = _CKERNEL(f.ctypes.data_as(F), w.ctypes.data_as(F),
                     nlat, nlon, iy0, iy1, ctypes.c_float(P_TRACK), 4, ring_px,
                     ctypes.c_float(RING_DEPTH), 8,
                     oy.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                     ox.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                     op.ctypes.data_as(F), od.ctypes.data_as(F),
                     ow.ctypes.data_as(F), _MAX_CAND)
        if n < 0:
            raise MemoryError("detect_step")
        cands[k] = [( float(_detect_c.lat[oy[i]]), float(_detect_c.lon[ox[i]]),
                      float(op[i]), float(od[i]), float(ow[i]) ) for i in range(n)]
    return cands


def detect_member(msl_hpa, wind, lat, lon, ring_px):
    """Candidate minima for one member: list per step of
    (lat, lon, p_min, ring_depth, wmax)."""
    if _CKERNEL is not None:
        _detect_c.lat = lat; _detect_c.lon = lon
        return _detect_c(msl_hpa, wind, lat, ring_px)
    nstep = msl_hpa.shape[0]
    cands = [[] for _ in range(nstep)]
    # ring mean via difference of uniform filters (cheap annulus approximation):
    # mean over disc(r_out) minus disc(r_in) — on the 0.25° grid px sizes vary
    # with latitude for zonal distance, but the ring test is a depth gate, not
    # a precise geometry, so grid-space annulus is fine.
    for k in range(nstep):
        f = msl_hpa[k]
        mins = (f == ndimage.minimum_filter(f, size=9, mode="wrap")) & (f < P_TRACK)
        iy, ix = np.where(mins)
        if not len(iy):
            continue
        big = ndimage.uniform_filter(f, size=2 * ring_px + 1, mode="wrap")
        small = ndimage.uniform_filter(f, size=ring_px, mode="wrap")
        npix_big = (2 * ring_px + 1) ** 2
        npix_small = ring_px ** 2
        ring = (big * npix_big - small * npix_small) / (npix_big - npix_small)
        wpad = 8                                                 # 2° at 0.25°
        for y, x in zip(iy, ix):
            if abs(lat[y]) > LAT_TRACK:
                continue
            depth = ring[y, x] - f[y, x]
            if depth < RING_DEPTH:
                continue
            y0, y1 = max(0, y - wpad), min(f.shape[0], y + wpad + 1)
            xs = (np.arange(x - wpad, x + wpad + 1)) % f.shape[1]
            wmax = float(wind[k][y0:y1][:, xs].max())
            cands[k].append((float(lat[y]), float(lon[x]), float(f[y, x]),
                             float(depth), wmax))
    return cands


def link_tracks(cands, steps_h, summer_nh: bool):
    """Greedy linking of per-step candidates into tracks (one member).
    Track points carry the VALID HOUR (not step index) so 6-hourly and daily
    step sets — and models with different step grids — mix cleanly.
    Point tuple: (hour, lat, lon, p, depth, wmax)."""
    tracks = []
    live = []
    for k in range(len(cands)):
        h = int(steps_h[k])
        dt_prev = int(steps_h[k] - steps_h[k - 1]) if k else 0
        pool = list(cands[k])
        used = set()
        # try to extend live tracks first (nearest candidate inside gate)
        for tr in live:
            hp, la, lo = tr[-1][0], tr[-1][1], tr[-1][2]
            dt = h - hp
            if dt > GAP_H:                       # too long since last fix → dead
                continue
            gla, glo = la, lo
            if len(tr) >= 2 and tr[-1][0] > tr[-2][0]:
                f = dt / (tr[-1][0] - tr[-2][0])     # extrapolate last motion
                gla = la + (la - tr[-2][1]) * f
                glo = lo + (lo - tr[-2][2]) * f
            gate = max(150.0, LINK_KMD * dt / 24.0)
            best, bd = None, gate
            for i, c in enumerate(pool):
                if i in used:
                    continue
                d = haversine_km(gla, glo, c[0], c[1])
                if d < bd:
                    best, bd = i, d
            if best is not None:
                c = pool[best]; used.add(best)
                tr.append((h, *c))
        # unclaimed candidates in the genesis window start new tracks
        for i, c in enumerate(pool):
            if i in used:
                continue
            glim = LAT_GENESIS if (c[0] >= 0) == summer_nh else LAT_GENESIS_WINTER
            if abs(c[0]) > glim or c[2] >= P_GATE or not is_ocean(c[0], c[1]) \
                    or excluded(c[0], c[1]):
                continue
            if abs(c[0]) > 20 and c[3] < 2.5:
                continue                         # 20-30°: frontal/subtropical junk zone —
                                                 # demand a solidly closed low
            live.append([(h, *c)])
        # retire stale tracks (collect, don't drop)
        still = []
        for tr in live:
            (still if h - tr[-1][0] <= GAP_H else tracks).append(tr)
        live = still
    tracks = list(live) + tracks
    keep = []
    for tr in tracks:
        cut = next((i for i, pt in enumerate(tr) if abs(pt[1]) > LAT_ET), len(tr))
        tr = tr[:max(cut, 1)]
        if tr[-1][0] - tr[0][0] < MIN_SPAN_H:
            continue
        if max(p[5] for p in tr) < WIND_MIN:
            continue
        if max(p[4] for p in tr) < DEPTH_PEAK:
            continue                              # never digs below its surroundings:
                                                  # broad monsoon/thermal trough, not a TC
        if sum(is_ocean(p[1], p[2]) for p in tr) < 0.4 * len(tr):
            continue                              # lives on/along land: monsoon/lee trough
        path = sum(haversine_km(a[1], a[2], b[1], b[2])
                   for a, b in zip(tr[:-1], tr[1:]))
        if path < 500.0:
            continue                              # quasi-stationary: monsoon/heat low, not a TC
        keep.append(tr)
    return keep


def cluster_storms(all_tracks):
    """Group member tracks into storms by genesis proximity (greedy)."""
    clusters = []
    for mem, tr in all_tracks:
        g = tr[0]
        placed = False
        for cl in clusters:
            g0 = cl["genesis"]
            if (abs(g[0] - g0[0]) <= CLUSTER_DT * 24
                    and haversine_km(g[1], g[2], g0[1], g0[2]) <= CLUSTER_KM):
                cl["tracks"].append((mem, tr)); placed = True
                break
        if not placed:
            clusters.append({"genesis": (g[0], g[1], g[2]), "tracks": [(mem, tr)]})
    # merge pass (clusters seeded far apart can converge)
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                gi, gj = clusters[i]["genesis"], clusters[j]["genesis"]
                if (abs(gi[0] - gj[0]) <= CLUSTER_DT * 24
                        and haversine_km(gi[1], gi[2], gj[1], gj[2]) <= CLUSTER_KM):
                    clusters[i]["tracks"] += clusters[j]["tracks"]
                    del clusters[j]; merged = True
                    break
            if merged:
                break
    return clusters


def basin_of(lat, lon):
    if (-105 <= lon <= -84 and 5 <= lat <= 18) or (-84 <= lon <= -77 and 0 <= lat <= 9):
        return "epac"                              # Pacific side of Central America / Panama Bight
    for key, label, w, e, s, n in BASINS:
        if w <= lon <= e and s <= lat <= n:
            return key
    return "other"


def strike_probability(all_tracks, n_members):
    """0.5° grid: fraction of members with any track point (6-hourly linear
    interpolation between daily fixes) within STRIKE_KM."""
    glat = np.arange(-60, 60.01, 0.5)
    glon = np.arange(-180, 180.0, 0.5)
    hit = {}
    for mem, tr in all_tracks:
        pts = []
        for a, b in zip(tr[:-1], tr[1:]):
            for f in np.linspace(0, 1, 5)[:-1]:
                la = a[1] + f * (b[1] - a[1])
                dlon = ((b[2] - a[2] + 180) % 360) - 180
                pts.append((la, ((a[2] + f * dlon + 180) % 360) - 180))
        pts.append((tr[-1][1], tr[-1][2]))
        m = hit.setdefault(mem, np.zeros((len(glat), len(glon)), bool))
        for la, lo in pts:
            dlat_deg = STRIKE_KM / 111.0
            dlon_deg = STRIKE_KM / max(111.0 * np.cos(np.radians(la)), 20.0)
            y = np.where(np.abs(glat - la) <= dlat_deg)[0]
            x = np.where(np.abs(((glon - lo + 180) % 360) - 180) <= dlon_deg)[0]
            if len(y) and len(x):
                yy, xx = np.meshgrid(y, x, indexing="ij")
                d = haversine_km(la, lo, glat[yy], glon[xx])
                m[yy[d <= STRIKE_KM], xx[d <= STRIKE_KM]] = True
    prob = np.zeros((len(glat), len(glon)))
    for m in hit.values():
        prob += m
    return glat, glon, prob / max(n_members, 1)


# ── rendering ────────────────────────────────────────────────────────────────
DARK_BG = "#0b1220"
PANEL_BG = "#101a2e"

# animator zooms: (key, label, lon_w, lon_e, lat_s, lat_n)
ANIM_REGIONS = [
    ("globe", "Global", -180, 180, -48, 48),
    ("natl", "Atlantic · Gulf of Mexico · Caribbean", -100, -15, 5, 48),
    ("epac", "East Pacific", -140, -75, 5, 35),
    ("wpac", "West Pacific", 100, 180, 5, 45),
]


def _wind_color(w):
    """10 m grid-scale wind → color (Saffir-flavored, thresholds lowered for
    ensemble-mean-ish 0.25° daily winds)."""
    for thr, c in ((33, "#ff5ce1"), (28, "#e53935"), (24, "#fb8c00"),
                   (18, "#fdd835"), (14, "#66bb6a")):
        if w >= thr:
            return c
    return "#4fc3f7"


def _style_ax(ax, extent=None):
    ax.set_facecolor(DARK_BG)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#1b2b45", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#41609a",
                   linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor="#2a4066",
                   linewidth=0.3, zorder=3)
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())


def _mean_track(cl):
    """Per-hour mean position of a cluster's OVER-WATER member fixes (hours
    with ≥30% of members and ≥3 fixes) — overland stragglers made low-support
    means scrawl across terrain."""
    from collections import defaultdict
    bag = defaultdict(list)
    for _, tr in cl["tracks"]:
        for pt in tr:
            bag[pt[0]].append((pt[1], pt[2], pt[5]))
    n_mem = len({m for m, _ in cl["tracks"]})
    out = []
    for k in sorted(bag):
        pts = bag[k]
        if len(pts) < max(3, 0.3 * n_mem):
            continue
        la = np.mean([q[0] for q in pts])
        # circular-safe lon mean (storm spread is < 60°, so anchor to first)
        l0 = pts[0][1]
        lo = l0 + np.mean([((q[1] - l0 + 180) % 360) - 180 for q in pts])
        out.append((k, la, ((lo + 180) % 360) - 180, np.mean([q[2] for q in pts])))
    return out


# weathernerds-style wind bins (kt) on 10 m grid-scale winds (m/s → kt)
_WN_BINS = [(0, "#b8b8b8"), (20, "#25c8c8"), (30, "#2a52dd"), (40, "#2fb52f"),
            (50, "#e6d82e"), (60, "#f08a1e"), (70, "#e2231e"), (80, "#e01e9d"),
            (100, "#f66ef0")]


def _wn_color(w_ms):
    kt = w_ms * 1.94384
    c = _WN_BINS[0][1]
    for thr, col in _WN_BINS:
        if kt >= thr:
            c = col
    return c


def fetch_nhc_invests():
    """Latest positions of active NHC systems + invests (Atlantic/EPAC/CPAC).
    Best-effort: returns [] on any failure so the product never breaks."""
    import requests
    out = []
    try:
        r = requests.get("https://www.nhc.noaa.gov/CurrentStorms.json", timeout=15)
        for s in r.json().get("activeStorms", []):
            out.append({"id": s.get("binNumber", s.get("id", "?")).upper(),
                        "name": s.get("name", ""),
                        "lat": float(s["latitudeNumeric"]),
                        "lon": float(s["longitudeNumeric"])})
    except Exception:                                          # noqa: BLE001
        pass
    try:
        import re
        listing = requests.get("https://ftp.nhc.noaa.gov/atcf/btk/", timeout=15).text
        for fn in set(re.findall(r"b(?:al|ep|cp)9\d2026\.dat", listing)):
            try:
                txt = requests.get(f"https://ftp.nhc.noaa.gov/atcf/btk/{fn}",
                                   timeout=15).text.strip().splitlines()
                last = txt[-1].split(",")
                la = last[6].strip(); lo = last[7].strip()
                lat = float(la[:-1]) / 10 * (1 if la.endswith("N") else -1)
                lon = float(lo[:-1]) / 10 * (1 if lo.endswith("E") else -1)
                out.append({"id": fn[1:5].upper(), "name": "INVEST",
                            "lat": lat, "lon": lon})
            except Exception:                                  # noqa: BLE001
                continue
    except Exception:                                          # noqa: BLE001
        pass
    return out


def match_invests(labeled, invests):
    """Tag day-0/1 genesis clusters with the nearest NHC system within 500 km."""
    for cl in labeled:
        cl["invest"] = ""
        g = cl["genesis"]
        if g[0] > 36:                            # only current systems match invests
            continue
        best, bd = None, 500.0
        for inv in invests:
            d = haversine_km(g[1], g[2], inv["lat"], inv["lon"])
            if d < bd:
                best, bd = inv, d
        if best:
            nm = best["name"].title() if best["name"] not in ("", "INVEST") else "Invest"
            cl["invest"] = f"{nm} {best['id']}"


def _draw_storm_panel(ax, cl, model, other_mean, w0, e0, s0, n0):
    ax.set_extent([w0, e0, s0, n0], crs=ccrs.PlateCarree())
    ax.set_facecolor("#d5ecf5")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8dcb8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#8a8a7a",
                   linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#b0b0a0",
                   linewidth=0.35, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#b9cfd9",
                      linestyle="--", xlocs=range(-180, 181, 10),
                      ylocs=range(-60, 61, 5))
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"color": "#667", "size": 6.5}
    HURR = 64 / 1.94384                              # 64 kt in m/s
    mtracks = [(m, tr) for m, tr in cl["tracks"] if m.startswith(model)]
    n_hur = 0
    for _, tr in mtracks:
        for a, b in zip(tr[:-1], tr[1:]):
            if b[5] < WIND_SHOW:
                continue                          # sub-25 kt noise floor (the
                                                  # colored endpoint must qualify)
            ax.plot([a[2], b[2]], [a[1], b[1]], color=_wn_color(b[5]),
                    lw=0.9, alpha=0.75, transform=ccrs.Geodetic(), zorder=4)
        vis = [pt for pt in tr if pt[5] >= WIND_SHOW]
        if vis:
            ax.scatter([pt[2] for pt in vis], [pt[1] for pt in vis], s=5,
                       c=[_wn_color(pt[5]) for pt in vis], edgecolors="none",
                       alpha=0.85, transform=ccrs.PlateCarree(), zorder=5)
        peak = max(tr, key=lambda pt: pt[5])
        if peak[5] >= HURR:
            n_hur += 1                               # hurricane-force member
            ax.plot(peak[2], peak[1], marker="$H$", ms=8, color="#c62828",
                    mew=0.4, transform=ccrs.PlateCarree(), zorder=8)
    # a mean track is only meaningful when this model genuinely develops the
    # storm — low-support scribbles were worse than nothing
    own_sup = cl["sup_m"].get(model, 0.0)
    mt = _mean_track({"tracks": mtracks}) if (mtracks and own_sup >= 0.25) else []
    if other_mean and len(other_mean) >= 2:
        ax.plot([q[2] for q in other_mean], [q[1] for q in other_mean],
                color="0.35", lw=1.6, ls="--", alpha=0.8,
                transform=ccrs.Geodetic(), zorder=5)
    if len(mt) >= 2:
        ax.plot([q[2] for q in mt], [q[1] for q in mt], color="k", lw=2.8,
                transform=ccrs.Geodetic(), zorder=6)
        for q in mt:
            if q[0] % 24 != 0 or q[0] == 0:
                continue                          # hour labels at each daily position
            ax.annotate(str(q[0]), xy=(q[2], q[1] + 0.7),
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        fontsize=6, color="0.3", ha="center", zorder=7,
                        annotation_clip=True, clip_on=True)
    return mt, n_hur


def _storm_timeseries(cl, model_counts, init, fp: Path):
    """MSLP + 10 m wind plumes for one storm: members thin (AIFS steel-blue,
    IFS orange), model means bold, TS/hurricane thresholds marked."""
    fig, (ap, aw) = plt.subplots(1, 2, figsize=(13.5, 4.6))
    colors = {"aifs": ("#1f77b4", "#0b3d66"), "ifs": ("#ff9f43", "#b35a00")}
    for model in sorted(model_counts):
        thin, bold = colors.get(model, ("0.6", "0.2"))
        mtracks = [tr for m, tr in cl["tracks"] if m.startswith(model)]
        from collections import defaultdict
        pbag, wbag = defaultdict(list), defaultdict(list)
        for tr in mtracks:
            tv = [init + pd.Timedelta(hours=int(pt[0])) for pt in tr]
            ap.plot(tv, [pt[3] for pt in tr], color=thin, lw=0.5, alpha=0.25)
            aw.plot(tv, [pt[5] * 1.94384 for pt in tr], color=thin, lw=0.5, alpha=0.25)
            for pt in tr:
                pbag[int(pt[0])].append(pt[3])
                wbag[int(pt[0])].append(pt[5] * 1.94384)
        n_model = model_counts[model]
        hrs = sorted(h for h in pbag if len(pbag[h]) >= max(3, 0.3 * max(len(mtracks), 1)))
        if hrs:
            tv = [init + pd.Timedelta(hours=h) for h in hrs]
            mdl = "AIFS" if model == "aifs" else "IFS"
            ap.plot(tv, [np.mean(pbag[h]) for h in hrs], color=bold, lw=2.4,
                    label=f"{mdl} mean")
            aw.plot(tv, [np.mean(wbag[h]) for h in hrs], color=bold, lw=2.4,
                    label=f"{mdl} mean")
    aw.axhline(34, color="0.55", lw=0.9, ls="--")
    aw.axhline(64, color="#c62828", lw=0.9, ls="--")
    aw.text(0.005, 34, " TS (34 kt)", fontsize=6.5, color="0.45", va="bottom",
            transform=aw.get_yaxis_transform())
    aw.text(0.005, 64, " hurricane (64 kt)", fontsize=6.5, color="#c62828",
            va="bottom", transform=aw.get_yaxis_transform())
    ap.set_ylabel("central MSLP (hPa)", fontsize=9)
    aw.set_ylabel("peak 10 m wind (kt, grid-scale)", fontsize=9)
    ap.set_title("Central pressure", fontsize=10, fontweight="bold")
    aw.set_title("Peak 10 m wind near centre", fontsize=10, fontweight="bold")
    for a in (ap, aw):
        a.grid(True, alpha=0.25)
        a.tick_params(labelsize=7.5)
        for lab in a.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
    ap.legend(fontsize=8, loc="best", framealpha=0.9)
    inv = f"  ·  ≈ {cl['invest']}" if cl.get("invest") else ""
    fig.suptitle(f"{cl['id']} — ensemble intensity{inv} · init {init:%Y-%m-%d %HZ} · "
                 f"grid-scale winds understate true intensity",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=140, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)


def render_storm_panels(labeled, model_counts, init, out_dir: Path):
    """One image per labeled storm (support ≥ 20%): side-by-side AIFS | IFS
    panels, member tracks colored by 10 m wind (kt), black = that model's
    mean track, grey dashed = the OTHER model's mean. Files:
    storms/storm_NN.webp + entries in the returned meta list (for the page
    gallery, click-to-enlarge)."""
    sdir = out_dir / "storms"
    sdir.mkdir(parents=True, exist_ok=True)
    for old_f in sdir.glob("storm_*.webp"):
        old_f.unlink()
    storms = sorted([cl for cl in labeled],
                    key=lambda c: -c["support"])[:12]
    meta = []
    for i, cl in enumerate(storms):
        las = [pt[1] for _, tr in cl["tracks"] for pt in tr]
        l0 = cl["genesis"][2]
        los = [l0 + (((pt[2] - l0 + 180) % 360) - 180) for _, tr in cl["tracks"] for pt in tr]
        w0, e0 = np.percentile(los, 2) - 5, np.percentile(los, 98) + 5
        s0, n0 = np.percentile(las, 2) - 4, np.percentile(las, 98) + 4
        if e0 - w0 < 24: pad = (24 - (e0 - w0)) / 2; w0 -= pad; e0 += pad
        if n0 - s0 < 16: pad = (16 - (n0 - s0)) / 2; s0 -= pad; n0 += pad
        if e0 - w0 > 70: c = (e0 + w0) / 2; w0, e0 = c - 35, c + 35
        if n0 - s0 > 40: c = (n0 + s0) / 2; s0, n0 = c - 20, c + 20
        aspect = (n0 - s0) / (e0 - w0)
        figw = 19.0
        figh = figw / 2 * aspect * 1.02 + 0.7
        fig = plt.figure(figsize=(figw, figh), facecolor="white")
        means = {}
        for j, model in enumerate(sorted(model_counts)):
            ax = fig.add_subplot(1, 2, j + 1,
                                 projection=ccrs.PlateCarree(central_longitude=(w0 + e0) / 2))
            other = [m for m in sorted(model_counts) if m != model]
            om = means.get(other[0]) if other else None
            means[model], n_hur = _draw_storm_panel(ax, cl, model, om, w0, e0, s0, n0)
            mdl = "AIFS-ENS" if model == "aifs" else "IFS-ENS"
            hur = f" · {n_hur} reach hurricane force" if n_hur else ""
            ax.set_title(f"{mdl} — {cl['sup_m'].get(model, 0)*100:.0f}% of "
                         f"{model_counts[model]} members{hur}",
                         fontsize=10, fontweight="bold")
        # second pass so the FIRST panel also gets the other model's mean
        if len(means) == 2:
            ms = sorted(model_counts)
            ax0 = fig.axes[0]
            om = means[ms[1]]
            if om and len(om) >= 2:
                ax0.plot([q[2] for q in om], [q[1] for q in om], color="0.35",
                         lw=1.6, ls="--", alpha=0.8, transform=ccrs.Geodetic(), zorder=5)
        from matplotlib.lines import Line2D
        bins = [("25+", "#25c8c8"), ("30+", "#2a52dd"),
                ("40+", "#2fb52f"), ("50+", "#e6d82e"), ("60+", "#f08a1e"),
                ("70+", "#e2231e"), ("80+ kt", "#e01e9d")]
        handles = [Line2D([], [], color=c, lw=2.2, label=l) for l, c in bins]
        handles += [Line2D([], [], color="k", lw=2.8, label="model mean"),
                    Line2D([], [], color="0.35", lw=1.6, ls="--", label="other model")]
        fig.axes[0].legend(handles=handles, loc="best", fontsize=6.8, ncol=2,
                           framealpha=0.92, borderpad=0.5, columnspacing=0.9,
                           handletextpad=0.4, title="10 m wind (kt)",
                           title_fontsize=7)
        inv = f"  ·  ≈ {cl['invest']}" if cl.get("invest") else ""
        g = cl["genesis"]
        fig.suptitle(f"{cl['id']} — support {cl['support']*100:.0f}%{inv} · "
                     f"genesis day {g[0] // 24} · init {init:%Y-%m-%d %HZ}",
                     fontsize=12, fontweight="bold", y=1.01)
        fig.tight_layout()
        fn = f"storm_{i:02d}.webp"
        sdir.mkdir(parents=True, exist_ok=True)   # robust to concurrent cleanup
        fig.savefig(sdir / fn, dpi=150, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 92, "method": 6})
        plt.close(fig)
        tsfn = f"storm_{i:02d}_ts.webp"
        _storm_timeseries(cl, model_counts, init, sdir / tsfn)
        meta.append({"id": cl["id"], "file": f"storms/{fn}", "ts_file": f"storms/{tsfn}",
                     "support": round(cl["support"], 3),
                     "sup_m": {k: round(v, 3) for k, v in cl["sup_m"].items()},
                     "invest": cl.get("invest", ""),
                     "genesis_day": int(g[0] // 24),
                     "genesis": [round(g[1], 1), round(g[2], 1)]})
    print(f"  storm panels: {len(storms)} storms → {sdir}")
    return meta


def _render_frame(job):
    """One animation frame (top-level for multiprocessing)."""
    (key, label, w, e, s, nn, model, n_model, k, ndays, upto, rclusters,
     init_iso, fp, figw, figh) = job
    init = pd.Timestamp(init_iso)
    glat, glon, prob = strike_probability(upto, n_model)
    fig = plt.figure(figsize=(figw, figh), facecolor="white")
    clon = -160 if key == "globe" else (w + e) / 2
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=clon))
    if key == "globe":
        ax.set_extent([-179.9, 179.9, s, nn], crs=ccrs.PlateCarree(central_longitude=clon))
    else:
        ax.set_extent([w, e, s, nn], crs=ccrs.PlateCarree())
    ax.set_facecolor("#d5ecf5")                                # sea
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8dcb8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#8a8a7a",
                   linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#b0b0a0",
                   linewidth=0.35, zorder=3)
    sel = (glon >= w - 5) & (glon <= e + 5)
    pshow = np.where(prob > 0.05, prob, np.nan)
    pm = ax.pcolormesh(glon[sel], glat, pshow[:, sel],
                       transform=ccrs.PlateCarree(), cmap="YlOrRd",
                       vmin=0, vmax=0.9, alpha=0.75, zorder=1)
    for m, tr in upto:
        la = [p[1] for p in tr]; lo = [p[2] for p in tr]
        ax.plot(lo, la, color="#3a6ea8", lw=0.5,
                alpha=0.3, transform=ccrs.Geodetic(), zorder=4)
        for p in tr:
            if p[0] == k * 24:
                ax.plot(p[2], p[1], marker="o",
                        ms=3.0 if key == "globe" else 4.0, color=_wn_color(p[5]),
                        mec="k", mew=0.35, alpha=0.95,
                        transform=ccrs.PlateCarree(), zorder=5)
    placed = []
    for cl in rclusters:                       # (genesis, id, support) tuples
        g, cid, sup = cl
        if g[0] > k * 24:
            continue
        ax.plot(g[2], g[1], marker="D", ms=3.8 if key == "globe" else 5.5,
                color="#222", mec="w", mew=0.7,
                transform=ccrs.PlateCarree(), zorder=6)
        if not (w + 2 <= g[2] <= e - 2 and s + 1.5 <= g[1] <= nn - 1.5):
            continue
        sy, sx = (4.5, 16) if key == "globe" else (3.2, 8)
        if any(abs(g[1] - py) < sy and abs(g[2] - px) < sx for py, px in placed):
            continue
        placed.append((g[1], g[2]))
        dy = -2.4 if g[1] > nn - 4 else 1.6
        lfs = 6.2 if key == "globe" else 8.5
        ax.annotate(f"{cid}  {sup*100:.0f}%", xy=(g[2], g[1] + dy),
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    color="#1a1a2e", fontsize=lfs, fontweight="bold",
                    ha="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.45))
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#b9cfd9",
                      linestyle="--", xlocs=range(-180, 181, 10),
                      ylocs=range(-60, 61, 5))
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"color": "#667", "size": 6.5}
    # dot-colour legend (10 m wind, kt)
    from matplotlib.lines import Line2D
    bins = [("25+", "#25c8c8"), ("30+", "#2a52dd"),
            ("40+", "#2fb52f"), ("50+", "#e6d82e"), ("60+", "#f08a1e"),
            ("70+", "#e2231e"), ("80+ kt", "#e01e9d")]
    handles = [Line2D([], [], marker="o", ls="none", ms=6, mec="k", mew=0.3,
                      color=c, label=l) for l, c in bins]
    handles.append(Line2D([], [], marker="D", ls="none", ms=6, color="#222",
                          mec="w", label="genesis"))
    lgfs = 5.4 if key == "globe" else 6.6
    ax.legend(handles=handles, loc="upper right", fontsize=lgfs, ncol=3,
              framealpha=0.85, borderpad=0.4, columnspacing=0.8,
              handletextpad=0.3, title="10 m wind (kt)", title_fontsize=lgfs + 0.2)
    mdl = "AIFS-ENS" if model == "aifs" else "IFS-ENS"
    valid = init + pd.Timedelta(days=k)
    ax.set_title(f"{label} — TC-strength lows · {mdl} ({n_model} members) · "
                 f"init {init:%Y-%m-%d %HZ}\n"
                 f"day {k} (valid {valid:%a %b %d}) · shading = strike "
                 f"probability through day {k} · ● = day-{k} member positions",
                 color="#1a1a2e", fontsize=10.3, fontweight="bold", pad=8)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.05,
                      fraction=0.04, aspect=48)
    cb.set_label("fraction of members with a track within 200 km",
                 color="#333", fontsize=8)
    cb.ax.tick_params(colors="#333", labelsize=7)
    cb.outline.set_edgecolor("#999")
    fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)
    return fp


def _render_with_julia(spec_regions, ndays, anim_dir: Path) -> bool:
    """Render all frames via scripts/julia/tc_render.jl (PNG), then convert to
    webp. Returns False on any failure so the matplotlib pool takes over."""
    import json
    import shutil
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    if not shutil.which("julia"):
        return False
    jdir = Path(__file__).resolve().parents[1] / "julia"
    spec = {"init": INIT_ISO[0], "ndays": ndays,
            "land_mask": str(Path(__file__).parent / "land_mask_0p5.npz"),
            "out_dir": str(anim_dir), "regions": spec_regions}
    sp = anim_dir / "render_spec.json"
    sp.write_text(json.dumps(spec))
    try:
        r = subprocess.run(["julia", str(jdir / "tc_render.jl"), str(sp)],
                           capture_output=True, text=True, timeout=1500)
        if r.returncode != 0 or "JULIA RENDER DONE" not in r.stdout:
            print(f"  julia renderer failed — matplotlib fallback\n{r.stderr[-400:]}")
            return False
        print("  " + r.stdout.strip().splitlines()[-1])
    except Exception as e:                                     # noqa: BLE001
        print(f"  julia renderer failed ({str(e)[:80]}) — matplotlib fallback")
        return False
    from PIL import Image

    def _conv(png):
        Image.open(png).save(png.with_suffix(".webp"), quality=92, method=6)
        png.unlink()
    pngs = list(anim_dir.glob("*/F*.png"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_conv, pngs))
    print(f"  converted {len(pngs)} frames to webp")
    return True


def build_anim(coherent, labeled, model_counts, init, anim_dir: Path):
    """Per-day frames per zoom region, SPLIT BY MODEL (AIFS-ENS vs IFS-ENS)
    with the player's picker as the model dropdown. Frames render in a
    process pool."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    anim_dir.mkdir(parents=True, exist_ok=True)
    INIT_ISO[0] = init.isoformat()
    kmax = max((tr[-1][0] for _, tr in coherent), default=0)
    ndays = max(kmax // 24 + 1, 1)
    ver = int(pd.Timestamp.now().timestamp())
    jobs, mani_frames, spec_regions = [], {}, []
    for key, label, w, e, s, nn in ANIM_REGIONS:
        if key == "globe":
            # seasonal bounds: the winter hemisphere's TC-dead high latitudes
            # are just empty map
            s, nn = (-15, 45) if init.month in (5, 6, 7, 8, 9, 10) else (-45, 15)
        def in_box(tr):
            return any(w - 6 <= p[2] <= e + 6 and s - 4 <= p[1] <= nn + 4 for p in tr)
        rclusters = [cl for cl in labeled
                     if w - 3 <= cl["genesis"][2] <= e + 3
                     and s - 2 <= cl["genesis"][1] <= nn + 2]
        aspect = (nn - s) / (e - w)
        figw = 14.5 if key == "globe" else 10.4
        figh = figw * aspect * (1.12 if key == "globe" else 1.28) + 0.9
        reg_spec = {"key": key, "label": label, "w": w, "e": e, "s": s, "n": nn,
                    "figw": figw, "figh": figh, "models": []}
        spec_regions.append(reg_spec)
        for model in sorted(model_counts):
            rid = f"{key}_{model}"
            rtracks = [(m, [pt for pt in tr if pt[5] >= WIND_SHOW])
                       for m, tr in coherent if m.startswith(model) and in_box(tr)]
            rtracks = [(m, tr) for m, tr in rtracks if len(tr) >= 2]
            # per-model labels: support within THIS model's members
            rcl = sorted(((cl["genesis"], cl["id"], cl["sup_m"].get(model, 0.0))
                          for cl in rclusters), key=lambda x: -x[2])
            rcl = [c for c in rcl if c[2] >= 0.10]
            reg_spec["models"].append({
                "model": model, "n": model_counts[model], "rid": rid,
                "tracks": [{"member": m,
                            "fix": [[int(pt[0])] + [round(float(x), 2) for x in pt[1:]]
                                    for pt in tr]} for m, tr in rtracks],
                "clusters": [{"id": cid, "sup": round(sup, 3),
                              "g": [int(g[0]), round(float(g[1]), 2), round(float(g[2]), 2)]}
                             for g, cid, sup in rcl]})
            rdir = anim_dir / rid
            rdir.mkdir(parents=True, exist_ok=True)
            for old_f in rdir.glob("F*.webp"):
                old_f.unlink()
            frames = []
            for k in range(ndays):
                upto = [(m, [p for p in tr if p[0] <= k * 24]) for m, tr in rtracks]
                upto = [(m, tr) for m, tr in upto if tr]
                fp = rdir / f"F{k:02d}.webp"
                valid = init + pd.Timedelta(days=k)
                jobs.append((key, label, w, e, s, nn, model,
                             model_counts[model], k, ndays, upto, rcl,
                             init.isoformat(), str(fp), figw, figh))
                frames.append({"idx": k, "file": fp.name,
                               "date": f"{valid:%Y-%m-%d}",
                               "label": f"day {k} · valid {valid:%a %b %d}"})
            mani_frames[rid] = frames
    # renderer dispatch: Julia/Makie (one warm process, ~5x faster) with the
    # matplotlib pool as automatic fallback (TC_RENDERER=python forces it)
    rendered = False
    if os.environ.get("TC_RENDERER", "julia") != "python":
        rendered = _render_with_julia(spec_regions, ndays, anim_dir)
    if not rendered:
        with ProcessPoolExecutor(max_workers=10) as ex:
            list(ex.map(_render_frame, jobs, chunksize=2))
    for key, label, w, e, s, nn in ANIM_REGIONS:
        regions = {}
        for model in sorted(model_counts):
            rid = f"{key}_{model}"
            mdl = "AIFS-ENS" if model == "aifs" else "IFS-ENS"
            regions[rid] = {"label": f"{mdl} ({model_counts[model]} members)",
                            "n_frames": len(mani_frames[rid]),
                            "frames": mani_frames[rid]}
        mani = {"ver": ver, "days": ndays, "default": f"{key}_aifs",
                "selectorLabel": "model", "regions": regions}
        (anim_dir / f"{key}_manifest.json").write_text(json.dumps(mani))
    print(f"  anim: {len(jobs)} frames across "
          f"{len(ANIM_REGIONS)} regions × {len(model_counts)} models")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", default="00")
    ap.add_argument("--out-dir", default="../../assets/tc")
    args = ap.parse_args()
    init = pd.Timestamp(f"{args.date}T{args.time}:00")
    print("== global ensemble TC tracker ==", flush=True)

    sources = open_members(args.date, args.time)
    if not sources:
        print("no cached MSLP surface files for this cycle — nothing to do",
              file=sys.stderr)
        return 1

    summer_nh = init.month in (5, 6, 7, 8, 9, 10)
    _MONTH[0] = init.month
    all_tracks = []
    n_members = 0
    model_counts = {}
    for model, typ, path, msl, u10, v10, members in sources:
        lat = msl.latitude.values
        lon = msl.longitude.values
        ring_px = int(round(RING_DEG / abs(float(lat[1] - lat[0]))))
        steps_h = (msl.step / np.timedelta64(1, "h")).values.astype(int)
        # decode in member batches (u+v kept for vorticity — full-file decode
        # of all three components would need ~23 GB for the 6-hourly pf file)
        BATCH = 10
        has_num = "number" in msl.dims
        for b0 in range(0, len(members), BATCH):
            bsl = slice(b0, b0 + BATCH)
            fa = ((msl.isel(number=bsl) if has_num else msl).values / 100.0).astype(np.float32)
            ua = (u10.isel(number=bsl) if has_num else u10).values.astype(np.float32)
            va = (v10.isel(number=bsl) if has_num else v10).values.astype(np.float32)
            if fa.ndim == 3:                                # cf: no member dim
                fa = fa[None]; ua = ua[None]; va = va[None]
            wa = np.hypot(ua, va)
            for mi, m in enumerate(members[bsl]):
                cands = detect_member(fa[mi], wa[mi], lat, lon, ring_px)
                for k in range(len(cands)):                 # vorticity-centroid refine
                    if cands[k]:
                        zk = _zeta(ua[mi, k], va[mi, k], lat, lon)
                        cands[k] = refine_centroid(cands[k], zk, lat, lon)
                tracks = link_tracks(cands, steps_h, summer_nh)
                mid = f"{model}-{typ}{'' if m is None else int(m)}"
                for tr in tracks:
                    all_tracks.append((mid, tr))
                n_members += 1
                model_counts[model] = model_counts.get(model, 0) + 1
            del fa, ua, va, wa
        del msl, u10, v10
        print(f"  {model} {typ}: tracked ({n_members} members cumulative, "
              f"{len(all_tracks)} tracks)", flush=True)

    clusters = cluster_storms(all_tracks)
    supported = [cl for cl in clusters
                 if len({m for m, _ in cl["tracks"]}) >= MIN_SUPPORT * n_members]
    coherent = [mt for cl in supported for mt in cl["tracks"]]
    # label ids + overall and per-model support
    by_basin = {}
    for cl in sorted(supported, key=lambda c: c["genesis"][0]):
        b = basin_of(cl["genesis"][1], cl["genesis"][2])
        by_basin.setdefault(b, []).append(cl)
        cl["id"] = f"{b.upper()}-{len(by_basin[b])}"
        mems = {m for m, _ in cl["tracks"]}
        cl["support"] = len(mems) / n_members
        cl["sup_m"] = {mod: len({m for m in mems if m.startswith(mod)}) / cnt
                       for mod, cnt in model_counts.items()}
    labeled = supported
    match_invests(labeled, fetch_nhc_invests())
    storm_meta = render_storm_panels(labeled, model_counts, init, Path(args.out_dir))
    build_anim(coherent, labeled, model_counts, init,
               Path(args.out_dir) / "anim")
    print(f"  {len(all_tracks)} member-tracks → {len(clusters)} clusters, "
          f"{len(labeled)} labeled storms:")
    for cl in labeled:
        g = cl["genesis"]
        print(f"    {cl['id']}: genesis day {g[0] // 24} near ({g[1]:.1f}, {g[2]:.1f}), "
              f"support {cl['support']*100:.0f}%")
    (Path(args.out_dir) / "tc_meta.json").write_text(json.dumps(
        {"init": f"{init:%Y-%m-%d %H}Z", "members": n_members,
         "storms": storm_meta}))
    # full track dump for external renderers (Julia/Makie experiments etc.)
    (Path(args.out_dir) / "tracks.json").write_text(json.dumps(
        {"init": f"{init:%Y-%m-%d %H}Z",
         "storms": [{"id": cl["id"], "support": round(cl["support"], 3),
                     "invest": cl.get("invest", ""),
                     "tracks": [{"member": m,
                                 "fix": [[int(pt[0])] + [round(float(x), 2) for x in pt[1:]]
                                         for pt in tr]}
                                for m, tr in cl["tracks"]]}
                    for cl in labeled]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
