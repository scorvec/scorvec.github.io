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
LINK_KM = 800.0                      # 24 h linking gate (motion-extrapolated)
MIN_DAYS = 3                         # minimum track length (points)
WIND_MIN = 14.0                      # m/s: track's peak 10 m wind to keep it
CLUSTER_KM = 600.0                   # genesis agreement radius
CLUSTER_DT = 2                       # genesis agreement window (days)
MIN_SUPPORT = 0.15                   # fraction of members to label a storm
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
           (42, 58, 8, 18)]      # Gulf of Aden / Somali jet / Socotra zone (the
                                 # July monsoon trough re-seeds a fake "storm"
                                 # every few days; real Arabian Sea genesis
                                 # is east of ~60°E)


def excluded(lat, lon):
    return any(w <= lon <= e and s <= lat <= n for w, e, s, n in EXCLUDE)


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


def is_open_ocean(lat, lon):
    """≥ ~110 km from any coast — the genesis standard. Kills desert heat
    lows AND semi-permanent coastal troughs (Panama Bight, lee of the Andes)."""
    i, j = _cell(lat, lon)
    if i < 0 or i >= len(_LM["lat"]):
        return False
    return not bool(_NEAR_LAND[i, j])


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
    hits = []
    for sub in sorted(cycle_dir.glob("*")):
        if not sub.is_dir():
            continue
        for typ in ("cf", "pf"):
            for p in sorted(sub.glob(f"{typ}_*msl*_sfc_*.grib2")):
                hits.append((sub.name.replace("-ens", ""), typ, p))
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
    """Greedy linking of per-step candidates into tracks (one member)."""
    tracks = []                                  # each: list of (k, lat, lon, p, depth, w)
    live = []
    for k in range(len(cands)):
        pool = list(cands[k])
        used = set()
        # try to extend live tracks first (nearest candidate inside gate)
        for tr in live:
            kp, la, lo = tr[-1][0], tr[-1][1], tr[-1][2]
            if k - kp > 1:                       # missed more than one step → dead
                continue
            gla, glo = la, lo
            if len(tr) >= 2:                     # extrapolate last motion
                gla = la + (la - tr[-2][1])
                glo = lo + (lo - tr[-2][2])
            best, bd = None, LINK_KM
            for i, c in enumerate(pool):
                if i in used:
                    continue
                d = haversine_km(gla, glo, c[0], c[1])
                if d < bd:
                    best, bd = i, d
            if best is not None:
                c = pool[best]; used.add(best)
                tr.append((k, *c))
        # unclaimed candidates in the genesis window start new tracks
        for i, c in enumerate(pool):
            if i in used:
                continue
            glim = LAT_GENESIS if (c[0] >= 0) == summer_nh else LAT_GENESIS_WINTER
            if abs(c[0]) > glim or c[2] >= P_GATE or not is_ocean(c[0], c[1]) \
                    or excluded(c[0], c[1]):
                continue
            if abs(c[0]) > 20 and (c[3] < 3.0 or c[4] < 15.0):
                continue                         # 20-30°: frontal/subtropical junk zone —
                                                 # demand a deep closed low with real wind
            live.append([(k, *c)])
        # retire tracks that missed a step (collect, don't drop)
        still = []
        for tr in live:
            (still if k - tr[-1][0] <= 1 else tracks).append(tr)
        live = still
    tracks = list(live) + tracks
    keep = []
    for tr in tracks:
        cut = next((i for i, pt in enumerate(tr) if abs(pt[1]) > LAT_ET), len(tr))
        tr = tr[:max(cut, 1)]
        if len(tr) < MIN_DAYS:
            continue
        if max(p[5] for p in tr) < WIND_MIN:
            continue
        if sum(is_ocean(p[1], p[2]) for p in tr) < 0.4 * len(tr):
            continue                              # lives on/along land: monsoon/lee trough
        path = sum(haversine_km(a[1], a[2], b[1], b[2])
                   for a, b in zip(tr[:-1], tr[1:]))
        if path < 700.0:
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
            if (abs(g[0] - g0[0]) <= CLUSTER_DT
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
                if (abs(gi[0] - gj[0]) <= CLUSTER_DT
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
    """Per-day mean position of a cluster's member fixes (days with ≥30% of
    the cluster's members present)."""
    from collections import defaultdict
    bag = defaultdict(list)
    for _, tr in cl["tracks"]:
        for pt in tr:
            bag[pt[0]].append((pt[1], pt[2], pt[5]))
    n_mem = len({m for m, _ in cl["tracks"]})
    out = []
    for k in sorted(bag):
        pts = bag[k]
        if len(pts) < max(2, 0.3 * n_mem):
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


def render_storm_panels(labeled, model_counts, init, out_dir: Path):
    """One zoomed panel per labeled storm (support ≥ 20%), weathernerds-style:
    light map, member tracks colored by 10 m wind, bold black ensemble-mean
    track with day labels. Written as a single grid sheet: tc_storms.webp"""
    storms = sorted([cl for cl in labeled if cl["support"] >= 0.20],
                    key=lambda c: -c["support"])[:8]
    if not storms:
        return 0
    ncol = 2
    nrow = (len(storms) + ncol - 1) // ncol
    fig = plt.figure(figsize=(7.4 * ncol, 6.0 * nrow + 0.7), facecolor="white")
    for i, cl in enumerate(storms):
        las = [p[1] for _, tr in cl["tracks"] for p in tr]
        l0 = cl["genesis"][2]
        los = [l0 + (((p[2] - l0 + 180) % 360) - 180) for _, tr in cl["tracks"] for p in tr]
        # 10-90% envelope (outlier link-chains must not set the zoom), clamped
        w0, e0 = np.percentile(los, 8) - 4, np.percentile(los, 92) + 4
        s0, n0 = np.percentile(las, 8) - 3, np.percentile(las, 92) + 3
        if e0 - w0 < 24: pad = (24 - (e0 - w0)) / 2; w0 -= pad; e0 += pad
        if n0 - s0 < 16: pad = (16 - (n0 - s0)) / 2; s0 -= pad; n0 += pad
        if e0 - w0 > 55: c = (e0 + w0) / 2; w0, e0 = c - 27.5, c + 27.5
        if n0 - s0 > 32: c = (n0 + s0) / 2; s0, n0 = c - 16, c + 16
        ax = fig.add_subplot(nrow, ncol, i + 1,
                             projection=ccrs.PlateCarree(central_longitude=(w0 + e0) / 2))
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
        for _, tr in cl["tracks"]:
            for a, b in zip(tr[:-1], tr[1:]):
                ax.plot([a[2], b[2]], [a[1], b[1]], color=_wn_color(b[5]),
                        lw=0.9, alpha=0.75, transform=ccrs.Geodetic(), zorder=4)
        mt = _mean_track(cl)
        if len(mt) >= 2:
            ax.plot([q[2] for q in mt], [q[1] for q in mt], color="k", lw=2.8,
                    transform=ccrs.Geodetic(), zorder=6)
            for q in mt[::2]:
                ax.annotate(f"d{q[0]}", xy=(q[2], q[1] + 0.7),
                            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                            fontsize=6.5, color="0.35", ha="center", zorder=7)
        sup = " · ".join(f"{m.upper()} {cl['sup_m'].get(m, 0)*100:.0f}%"
                         for m in sorted(model_counts))
        g = cl["genesis"]
        ax.set_title(f"{cl['id']} — support {cl['support']*100:.0f}% ({sup}) · "
                     f"genesis day {g[0]}", fontsize=10, fontweight="bold")
    fig.suptitle(f"Ensemble storm tracks — AIFS-ENS + IFS-ENS init {init:%Y-%m-%d %HZ} · "
                 f"black = ensemble-mean track · color = 10 m wind (kt): "
                 f"grey<20 · cyan≥20 · blue≥30 · green≥40 · yellow≥50 · orange≥60 · "
                 f"red≥70 · magenta≥80",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "tc_storms.webp", dpi=115, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  storm panels: {len(storms)} storms → tc_storms.webp")
    return len(storms)


def _render_frame(job):
    """One animation frame (top-level for multiprocessing)."""
    (key, label, w, e, s, nn, model, n_model, k, ndays, upto, rclusters,
     init_iso, fp, figw, figh) = job
    init = pd.Timestamp(init_iso)
    glat, glon, prob = strike_probability(upto, n_model)
    fig = plt.figure(figsize=(figw, figh), facecolor="white")
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree(central_longitude=(w + e) / 2))
    ax.set_extent([w, e, s, nn], crs=ccrs.PlateCarree())
    ax.set_facecolor("#d5ecf5")                                # sea
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8dcb8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#8a8a7a",
                   linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#b0b0a0",
                   linewidth=0.35, zorder=3)
    sel = (glon >= w - 5) & (glon <= e + 5)
    pm = ax.pcolormesh(glon[sel], glat, np.where(prob[:, sel] > 0.05,
                                                 prob[:, sel], np.nan),
                       transform=ccrs.PlateCarree(), cmap="YlOrRd",
                       vmin=0, vmax=0.9, alpha=0.75, zorder=1)
    for m, tr in upto:
        la = [p[1] for p in tr]; lo = [p[2] for p in tr]
        ax.plot(lo, la, color="#3a6ea8", lw=0.5, alpha=0.3,
                transform=ccrs.Geodetic(), zorder=4)
        for p in tr:
            if p[0] == k:
                ax.plot(p[2], p[1], marker="o", ms=4.0, color=_wn_color(p[5]),
                        mec="k", mew=0.35, alpha=0.95,
                        transform=ccrs.PlateCarree(), zorder=5)
    placed = []
    for cl in rclusters:                       # (genesis, id, support) tuples
        g, cid, sup = cl
        if g[0] > k:
            continue
        ax.plot(g[2], g[1], marker="D", ms=5.5, color="#222", mec="w",
                mew=0.8, transform=ccrs.PlateCarree(), zorder=6)
        if not (w + 2 <= g[2] <= e - 2 and s + 1.5 <= g[1] <= nn - 1.5):
            continue
        if any(abs(g[1] - py) < 3.2 and abs(g[2] - px) < 8 for py, px in placed):
            continue
        placed.append((g[1], g[2]))
        dy = -2.4 if g[1] > nn - 4 else 1.6
        ax.annotate(f"{cid}  {sup*100:.0f}%", xy=(g[2], g[1] + dy),
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    color="#1a1a2e", fontsize=8.5, fontweight="bold",
                    ha="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.65))
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#b9cfd9",
                      linestyle="--", xlocs=range(-180, 181, 10),
                      ylocs=range(-60, 61, 5))
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"color": "#667", "size": 6.5}
    # dot-colour legend (10 m wind, kt)
    from matplotlib.lines import Line2D
    bins = [("<20", "#b8b8b8"), ("20+", "#25c8c8"), ("30+", "#2a52dd"),
            ("40+", "#2fb52f"), ("50+", "#e6d82e"), ("60+", "#f08a1e"),
            ("70+", "#e2231e"), ("80+ kt", "#e01e9d")]
    handles = [Line2D([], [], marker="o", ls="none", ms=6, mec="k", mew=0.3,
                      color=c, label=l) for l, c in bins]
    handles.append(Line2D([], [], marker="D", ls="none", ms=6, color="#222",
                          mec="w", label="genesis"))
    ax.legend(handles=handles, loc="upper right", fontsize=6.6, ncol=3,
              framealpha=0.9, borderpad=0.5, columnspacing=0.9,
              handletextpad=0.35, title="10 m wind (kt)", title_fontsize=6.8)
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
    fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fp


def build_anim(coherent, labeled, model_counts, init, anim_dir: Path):
    """Per-day frames per zoom region, SPLIT BY MODEL (AIFS-ENS vs IFS-ENS)
    with the player's picker as the model dropdown. Frames render in a
    process pool."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    anim_dir.mkdir(parents=True, exist_ok=True)
    kmax = max((tr[-1][0] for _, tr in coherent), default=0)
    ndays = max(kmax + 1, 1)
    ver = int(pd.Timestamp.now().timestamp())
    jobs, mani_frames = [], {}
    for key, label, w, e, s, nn in ANIM_REGIONS:
        def in_box(tr):
            return any(w - 6 <= p[2] <= e + 6 and s - 4 <= p[1] <= nn + 4 for p in tr)
        rclusters = [cl for cl in labeled
                     if w - 3 <= cl["genesis"][2] <= e + 3
                     and s - 2 <= cl["genesis"][1] <= nn + 2]
        aspect = (nn - s) / (e - w)
        figw = 10.4
        figh = figw * aspect * 1.28 + 0.9
        for model in sorted(model_counts):
            rid = f"{key}_{model}"
            rtracks = [(m, tr) for m, tr in coherent
                       if m.startswith(model) and in_box(tr)]
            # per-model labels: support within THIS model's members
            rcl = sorted(((cl["genesis"], cl["id"], cl["sup_m"].get(model, 0.0))
                          for cl in rclusters), key=lambda x: -x[2])
            rcl = [c for c in rcl if c[2] >= 0.10]
            rdir = anim_dir / rid
            rdir.mkdir(parents=True, exist_ok=True)
            for old_f in rdir.glob("F*.webp"):
                old_f.unlink()
            frames = []
            for k in range(ndays):
                upto = [(m, [p for p in tr if p[0] <= k]) for m, tr in rtracks]
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
    all_tracks = []
    n_members = 0
    model_counts = {}
    for model, typ, path, msl, u10, v10, members in sources:
        lat = msl.latitude.values
        lon = msl.longitude.values
        ring_px = int(round(RING_DEG / abs(float(lat[1] - lat[0]))))
        steps_h = (msl.step / np.timedelta64(1, "h")).values.astype(int)
        # decode each variable ONCE — cfgrib per-member selection re-scans the
        # GRIB every time; a single bulk decode is far faster (~10 GB transient)
        fa = (msl.values / 100.0).astype(np.float32)
        wa = np.hypot(u10.values, v10.values).astype(np.float32)
        del msl, u10, v10
        if fa.ndim == 3:                                    # cf: no member dim
            fa = fa[None]; wa = wa[None]
        for mi, m in enumerate(members):
            cands = detect_member(fa[mi], wa[mi], lat, lon, ring_px)
            tracks = link_tracks(cands, steps_h, summer_nh)
            mid = f"{model}-{typ}{'' if m is None else int(m)}"
            for tr in tracks:
                all_tracks.append((mid, tr))
            n_members += 1
            model_counts[model] = model_counts.get(model, 0) + 1
        del fa, wa
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
    render_storm_panels(labeled, model_counts, init, Path(args.out_dir))
    build_anim(coherent, labeled, model_counts, init,
               Path(args.out_dir) / "anim")
    print(f"  {len(all_tracks)} member-tracks → {len(clusters)} clusters, "
          f"{len(labeled)} labeled storms:")
    for cl in labeled:
        g = cl["genesis"]
        print(f"    {cl['id']}: genesis day {g[0]} near ({g[1]:.1f}, {g[2]:.1f}), "
              f"support {cl['support']*100:.0f}%")
    (Path(args.out_dir) / "tc_meta.json").write_text(json.dumps(
        {"init": f"{init:%Y-%m-%d %H}Z", "members": n_members,
         "storms": [{"id": cl["id"], "support": round(cl["support"], 3)}
                    for cl in labeled]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
