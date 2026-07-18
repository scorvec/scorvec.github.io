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

Output: dark-theme global map (strike probability within 200 km + member
spaghetti + cluster labels) and per-active-basin zoom panels.

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

# no-TC zones (climatologically cyclone-free enclosed/marginal seas whose
# trough lows otherwise pass the MSLP gates): (lon_w, lon_e, lat_s, lat_n)
EXCLUDE = [(32, 44, 12, 30),     # Red Sea
           (46, 57, 22, 31),     # Persian Gulf
           (-6, 37, 29, 47)]     # Mediterranean


def excluded(lat, lon):
    return any(w <= lon <= e and s <= lat <= n for w, e, s, n in EXCLUDE)


def is_ocean(lat, lon):
    """Genesis must be over water (kills desert/plateau heat lows and lee
    troughs). 0.5° Natural Earth mask, nearest-cell lookup."""
    i = int(round((lat - _LM["lat"][0]) / 0.5))
    j = int(round((((lon + 180) % 360) - 180 - _LM["lon"][0]) / 0.5)) % len(_LM["lon"])
    if i < 0 or i >= len(_LM["lat"]):
        return False
    return not bool(_LM["land"][i, j])


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


def detect_member(msl_hpa, wind, lat, lon, ring_px):
    """Candidate minima for one member: list per step of
    (lat, lon, p_min, ring_depth, wmax)."""
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


def render(clusters, all_tracks, prob_grid, n_members, init, out_dir: Path):
    glat, glon, prob = prob_grid
    labeled = [cl for cl in clusters
               if len({m for m, _ in cl["tracks"]}) >= MIN_SUPPORT * n_members]
    # order labels per basin by genesis day
    by_basin = {}
    for cl in sorted(labeled, key=lambda c: c["genesis"][0]):
        b = basin_of(cl["genesis"][1], cl["genesis"][2])
        by_basin.setdefault(b, []).append(cl)
        cl["id"] = f"{b.upper()}-{len(by_basin[b])}"
        cl["support"] = len({m for m, _ in cl["tracks"]}) / n_members

    fig = plt.figure(figsize=(15.5, 8.3), facecolor=DARK_BG)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=-160))
    _style_ax(ax)
    ax.set_global()
    pm = ax.pcolormesh(glon, glat, np.where(prob > 0.02, prob, np.nan),
                       transform=ccrs.PlateCarree(), cmap="magma",
                       vmin=0, vmax=0.9, alpha=0.85, zorder=1)
    for mem, tr in all_tracks:
        la = [p[1] for p in tr]; lo = [p[2] for p in tr]
        ax.plot(lo, la, color="#9fd8ff", lw=0.45, alpha=0.28,
                transform=ccrs.Geodetic(), zorder=4)
    # numbered genesis dots + a roster below the map (on-map text piles up
    # where ensemble genesis points crowd)
    roster = sorted(labeled, key=lambda c: -c["support"])
    for i, cl in enumerate(roster):
        g = cl["genesis"]
        ax.plot(g[2], g[1], marker="o", ms=6.5, color="#fff59d", mec="k", mew=0.6,
                transform=ccrs.PlateCarree(), zorder=6)
        ax.annotate(str(i + 1), xy=ax.projection.transform_point(g[2], g[1], ccrs.PlateCarree()),
                    color="k", fontsize=5.2, fontweight="bold", ha="center", va="center",
                    zorder=7)
    ncol = 4
    for i, cl in enumerate(roster[:16]):
        g = cl["genesis"]
        lat_s = f"{abs(g[1]):.0f}{'N' if g[1] >= 0 else 'S'}"
        lon_s = f"{abs(g[2]):.0f}{'E' if g[2] >= 0 else 'W'}"
        fig.text(0.055 + 0.24 * (i % ncol), -0.012 - 0.022 * (i // ncol),
                 f"❶{i+1:>2} {cl['id']}  {cl['support']*100:3.0f}%  ·  day {g[0]} · {lat_s} {lon_s}"
                 .replace("❶", "●"),
                 color="#fff59d", fontsize=8, family="monospace",
                 transform=fig.transFigure)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.03,
                      fraction=0.045, aspect=45)
    cb.set_label(f"probability of a TC-strength low passing within {STRIKE_KM:.0f} km "
                 f"(next 15 days, {n_members} members)", color="0.85", fontsize=9)
    cb.ax.tick_params(colors="0.8", labelsize=8)
    cb.outline.set_edgecolor("#41609a")
    ax.set_title(f"Global ensemble tropical-cyclone tracker — AIFS-ENS"
                 f"{' + IFS-ENS' if n_members > 51 else ''} init {init:%Y-%m-%d %HZ}\n"
                 f"thin lines = per-member MSLP-minimum tracks · ● = ensemble genesis "
                 f"(labels: basin-number + member support)",
                 color="0.92", fontsize=12.5, fontweight="bold", pad=10)
    fig.text(0.5, 0.012,
             "tracks: closed MSLP minima (≥2 hPa ring depth) linked at 24 h steps, ≥3 days, "
             "peak 10 m wind ≥ 14 m/s · grid-scale daily winds understate true intensity",
             ha="center", color="0.55", fontsize=7.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "tc_global.webp", dpi=115, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)

    # per-basin zooms for basins with labeled storms
    active = [b for b in by_basin if b != "other"]
    if active:
        n = len(active)
        ncol = min(2, n); nrow = (n + ncol - 1) // ncol
        fig = plt.figure(figsize=(7.6 * ncol, 4.9 * nrow + 0.5), facecolor=DARK_BG)
        for i, b in enumerate(active):
            key, label, w, e, s, nn = next(x for x in BASINS if x[0] == b)
            ax = fig.add_subplot(nrow, ncol, i + 1,
                                 projection=ccrs.PlateCarree(central_longitude=(w + e) / 2))
            _style_ax(ax, extent=[w, e, s, nn])
            sel = (glon >= w - 5) & (glon <= e + 5)
            ax.pcolormesh(glon[sel], glat, np.where(prob[:, sel] > 0.02,
                                                    prob[:, sel], np.nan),
                          transform=ccrs.PlateCarree(), cmap="magma",
                          vmin=0, vmax=0.9, alpha=0.8, zorder=1)
            placed = []
            for cl in sorted(by_basin[b], key=lambda c: -c["support"]):
                for mem, tr in cl["tracks"]:
                    la = [p[1] for p in tr]; lo = [p[2] for p in tr]
                    ax.plot(lo, la, color="#9fd8ff", lw=0.5, alpha=0.3,
                            transform=ccrs.Geodetic(), zorder=4)
                    for p in tr:
                        ax.plot(p[2], p[1], marker="o", ms=1.6,
                                color=_wind_color(p[5]), alpha=0.55,
                                transform=ccrs.PlateCarree(), zorder=5)
                g = cl["genesis"]
                if not (w + 2 <= g[2] <= e - 2 and s + 1.5 <= g[1] <= nn - 1.5):
                    continue                            # genesis off-panel: dot only
                if any(abs(g[1] - py) < 3.5 and abs(g[2] - px) < 7 for py, px in placed):
                    continue                            # too close to a placed label
                placed.append((g[1], g[2]))
                dy = -2.6 if g[1] > nn - 4 else 1.8
                ax.annotate(f"{cl['id']}  {cl['support']*100:.0f}%",
                            xy=(g[2], g[1] + dy), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                            color="#fff59d", fontsize=9, fontweight="bold",
                            ha="center", zorder=7)
            gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="#2a4066",
                              xlocs=range(-180, 181, 15), ylocs=range(-60, 61, 10))
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = gl.ylabel_style = {"color": "0.6", "size": 6.5}
            ax.set_title(label, color="0.9", fontsize=11, fontweight="bold")
        fig.suptitle(f"Active-basin detail — init {init:%Y-%m-%d %HZ} · dots colored by "
                     f"10 m wind (blue < 14 · green ≥ 14 · yellow ≥ 18 · orange ≥ 24 · "
                     f"red ≥ 28 · magenta ≥ 33 m/s)",
                     color="0.92", fontsize=11.5, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / "tc_basins.webp", dpi=115, bbox_inches="tight",
                    facecolor=DARK_BG)
        plt.close(fig)
    return labeled


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
    for model, typ, path, msl, u10, v10, members in sources:
        lat = msl.latitude.values
        lon = msl.longitude.values
        ring_px = int(round(RING_DEG / abs(float(lat[1] - lat[0]))))
        for m in members:
            sel = dict(number=m) if m is not None else {}
            f = (msl.sel(**sel) if sel else msl).values / 100.0
            w = np.hypot((u10.sel(**sel) if sel else u10).values,
                         (v10.sel(**sel) if sel else v10).values)
            cands = detect_member(f, w, lat, lon, ring_px)
            steps_h = (msl.step / np.timedelta64(1, "h")).values.astype(int)
            tracks = link_tracks(cands, steps_h, summer_nh)
            mid = f"{model}-{typ}{'' if m is None else int(m)}"
            for tr in tracks:
                all_tracks.append((mid, tr))
            n_members += 1
        print(f"  {model} {typ}: tracked ({n_members} members cumulative, "
              f"{len(all_tracks)} tracks)", flush=True)

    clusters = cluster_storms(all_tracks)
    supported = [cl for cl in clusters
                 if len({m for m, _ in cl["tracks"]}) >= MIN_SUPPORT * n_members]
    coherent = [mt for cl in supported for mt in cl["tracks"]]
    prob_grid = strike_probability(coherent, n_members)
    labeled = render(clusters, coherent, prob_grid, n_members, init,
                     Path(args.out_dir))
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
