#!/usr/bin/env python3
"""Polar jet, subtropical jet and superposition forecasts from the 50 IFS-ENS members (jet_branches.py).

DATA. ECMWF IFS-ENS open data, the 00 UTC run's 50 perturbed members (open data has no control on pressure levels):
t, u and v on the nine levels 700-100 hPa (jet_branches.LEVS), 12-hourly to --last-step, byte-ranged one step at a
time from the Google Cloud mirror only (scripts/ecmwf/rangefetch.py: index, select, coalesce, ranged GETs, metered by
the shared request budget) and decoded in memory; nothing is cached. No other job holds these fields: aam-ifs.yml
keeps u on these levels at daily steps only, strat-ifs.yml u and v at 100 hPa and t at 100 hPa, in jobs of their own.

IDENTIFICATION. Every member at every step on its own (PV does not survive averaging), by jet_branches.masks_025:
the ERA5 reference's method, grid and thresholds.

PRODUCTS, the cold season only: an init from 1 November to 31 March, leads past 31 March dropped.
  maps     per lead, the share of members with each branch within NEAR_KM (300 km) of the column (jet_branches.near),
           against how often ERA5 had it within 300 km on the valid day of the season, 1991-2020 (the centred 31-day
           window; near_count of jet_branches_clim.nc, release jet-branches-ref-v1) -> loop frames + manifest
  events   per region (western North Pacific, North America), the share of members with a superposition event at
           each lead - a group of superposed columns, each within 1000 km of another, at least as large as the
           region's threshold (reference/jet_branches_ref.json) - against ERA5's rate of analyses with one on the
           valid day (reference/jet_branches_regions.nc, the same window) -> a still
  json     the numbers behind both and what the fetch cost
Outside the season the still and a one-frame loop say so, and the json has in_season false.

EVIDENCE. With reference/jet_branches_skill.json (jet_branches_verify.py: archived runs against ERA5) the shares are
recalibrated where that verification adopted it, and a branch's map or a region's event odds are drawn only at the
leads where they beat the ERA5 normal; elsewhere the panel says so. Without the file every lead is drawn as it comes.

    python jet_branches_fc.py run --date 20260110 --out fc.npz       # fetch and identify (the costly part)
    python jet_branches_fc.py render --fc fc.npz --clim jet_branches_clim.nc --out-dir assets/sst
    python jet_branches_fc.py forecast --date 20260110 --clim jet_branches_clim.nc --out-dir assets/sst
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import json
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ecmwf"))
sys.path.insert(0, str(HERE.parents[1] / "strat"))
import ifs_ens as IE  # the IFS-ENS publication check (scripts/ecmwf/cycles.py's shared rule)
import jet_branches as JB

REF = HERE.parent / "data" / "reference"
SKILL = REF / "jet_branches_skill.json"
MODEL, LABEL, NAME = "ifs", "IFS-ENS", "jetbr_ifs"
HOUR = "00"
PARAMS = ("t", "u", "v")
MEMBERS = 50
MIN_SHARE = 0.9  # a lead with fewer than this share of the members complete is dropped
SOURCES = ["google"]  # the Google Cloud mirror only
STEP_TRIES = 3
INK, MUTED = "#1a1a1a", "#6f6b64"
P_LEV = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # the North Pacific jet card's probability scale
P_COL = ["#fff3c4", "#fde28a", "#fdc458", "#fca03f", "#f7792f", "#e8512a", "#cc2f2f", "#a41d3f", "#7a1450", "#4d0f52"]
CLIM_LEV = {"polar": (25, 50, 75), "stj": (25, 50, 75), "sup": (2, 5, 10)}  # % contours of the ERA5 frequency
TITLE = {
    "polar": "Polar jet (315–330 K PV wall)",
    "stj": "Subtropical jet (340–355 K PV wall)",
    "sup": "Superposition (both in one column)",
}
CREDIT = (
    "ECMWF IFS-ENS open data (CC BY 4.0), 50 perturbed members, 0.25° reduced to 0.5°, Google Cloud mirror · "
    "normals: ERA5 1991–2020 (Copernicus C3S) · method: Winters & Martin 2014, Christenson et al. 2017, "
    "Winters et al. 2020"
)


# ── fetch and identify ──────────────────────────────────────────────────────────────────────────────────────────────
def lead_steps(last_step):
    return list(range(0, int(last_step) + 1, 12))


def _grid(gid, eccodes):
    """The message's latitudes and the column order that puts its longitudes at 0..359.75."""
    nj, ni = eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")
    la0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
    la1 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
    lo0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
    lat = np.linspace(la0, la1, nj)
    lon = (lo0 + (360.0 / ni) * np.arange(ni)) % 360
    return nj, ni, JB.band_rows025(lat), np.argsort(np.round(lon, 3))


def step_masks(job):
    """One step of the run: t, u, v at LEVS for perturbed members 1..n, every member identified.
    -> {step, members (n,), masks (n, 3, packed), cost}; None when the step could not be fetched."""
    import eccodes
    import rangefetch as rf

    date, step, n = job
    for attempt in range(1, STEP_TRIES + 1):
        try:
            t0 = time.time()
            idx = rf.fetch_index(date, HOUR, MODEL, step, "ef", sources=SOURCES)
            want = sorted(
                rf.select(idx, param=list(PARAMS), levelist=list(JB.LEVS), numbers=list(range(1, n + 1)), type="pf"),
                key=lambda e: e["_offset"],
            )
            have = {(e["param"], int(e["levelist"])) for e in want}
            lacking = sorted({(p, lev) for p in PARAMS for lev in JB.LEVS} - have)
            if lacking:  # not a transport failure: the run does not carry these fields
                print(f"  step {step:3d}: the run lacks {lacking}; skipped", flush=True)
                return None
            ranges = rf.coalesce(want)
            wire = {}
            blob = rf.fetch_ranges(
                rf.path_for(date, HOUR, MODEL, step, "ef") + ".grib2", ranges, sources=SOURCES, stats=wire
            )
            t1 = time.time()
            where, pos = {}, 0
            for e in want:
                where[(int(e["number"]), e["param"], int(e["levelist"]))] = (pos, e["_length"])
                pos += e["_length"]
            if pos != len(blob):
                raise RuntimeError(f"{len(blob)} bytes for {pos} indexed")
            grid, members, packed = None, [], []
            for num in sorted({k[0] for k in where}):
                keys = [(num, p, lev) for p in PARAMS for lev in JB.LEVS]
                if not all(k in where for k in keys):
                    continue
                fields = []
                for k in keys:
                    o, n = where[k]
                    gid = eccodes.codes_new_from_message(bytes(blob[o : o + n]))
                    try:
                        grid = grid or _grid(gid, eccodes)
                        nj, ni, rows, order = grid
                        fields.append(eccodes.codes_get_values(gid).reshape(nj, ni)[rows][:, order].astype(np.float32))
                    finally:
                        eccodes.codes_release(gid)
                t, u, v = (np.stack(fields[i * len(JB.LEVS) : (i + 1) * len(JB.LEVS)]) for i in range(3))
                members.append(num)
                packed.append(np.packbits(JB.masks_025(t, u, v).reshape(3, -1), axis=-1))
            del blob
            if not members:
                print(f"  step {step:3d}: no member complete; skipped", flush=True)
                return None
            cost = dict(
                messages=len(want),
                bytes=pos,
                wire=int(sum(wire.values())),
                ranges=len(ranges),
                fetch_s=round(t1 - t0, 1),
                identify_s=round(time.time() - t1, 1),
            )
            print(
                f"  step {step:3d}: {len(members)} members, {cost['messages']} messages, {pos / 1e6:,.0f} MB "
                f"({cost['wire'] / 1e6:,.0f} on the wire) in {cost['ranges']} ranges; fetch {cost['fetch_s']:.0f} s, "
                f"identify {cost['identify_s']:.0f} s",
                flush=True,
            )
            return dict(step=step, members=np.array(members), masks=np.stack(packed), cost=cost)
        except Exception as e:  # noqa: BLE001  one step must not end the run; the lead is dropped instead
            print(f"  step {step:3d}: attempt {attempt} failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
            time.sleep(20 * attempt)
    return None


def carries_levels(date) -> bool:
    """Whether the run's open data has t, u and v on all of LEVS (step 0's index): IFS-ENS open data gained 600, 400,
    150 and 100 hPa during 2025, so older runs cannot be identified."""
    import rangefetch as rf

    idx = rf.fetch_index(date, HOUR, MODEL, 0, "ef", sources=SOURCES)
    have = {
        (e["param"], int(e["levelist"])) for e in rf.select(idx, param=list(PARAMS), type="pf") if e.get("levelist")
    }
    return {(p, lev) for p in PARAMS for lev in JB.LEVS} <= have


def run(date, last_step, out, workers=3, wait_min=0.0, members=MEMBERS):
    """Fetch and identify every step for perturbed members 1..members -> npz (per-step member masks and the fetch's
    cost). Waits up to wait_min for the run to be on the Google mirror to last_step."""
    deadline = time.time() + 60 * wait_min
    while not IE.published(date, HOUR, step=last_step):
        if time.time() >= deadline:
            raise SystemExit(f"IFS-ENS {date} {HOUR}Z is not on the Google mirror to step {last_step}")
        print(f"  IFS-ENS {date} {HOUR}Z not complete on the Google mirror yet; checking again in 2 min", flush=True)
        time.sleep(120)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        got = [r for r in ex.map(step_masks, [(date, s, members) for s in lead_steps(last_step)]) if r is not None]
    wall = time.time() - t0
    keep = [r for r in got if r["members"].size >= MIN_SHARE * members]
    if not keep:
        raise SystemExit("no step was fetched with enough members")
    cost = {k: sum(r["cost"][k] for r in got) for k in ("messages", "bytes", "wire", "ranges")}
    cost.update(index_requests=len(got), steps=len(got), wall_s=round(wall, 1), workers=workers)
    print(
        f"IFS-ENS {date} {HOUR}Z: {cost['steps']} steps, {cost['messages']:,} messages, {cost['bytes'] / 1e9:.2f} GB "
        f"wanted, {cost['wire'] / 1e9:.2f} GB on the wire, {cost['ranges']:,} range requests + {cost['index_requests']} "
        f"index requests, {wall:.0f} s wall",
        flush=True,
    )
    n = max(r["members"].size for r in keep)
    masks = np.zeros((len(keep), n, *keep[0]["masks"].shape[1:]), np.uint8)
    members = np.zeros((len(keep), n), np.int16)
    for i, r in enumerate(keep):
        masks[i, : r["members"].size] = r["masks"]
        members[i, : r["members"].size] = r["members"]
    np.savez_compressed(
        out,
        init=np.datetime64(pd.Timestamp(f"{date} {HOUR}:00"), "s"),
        steps=np.array([r["step"] for r in keep]),
        members=members,
        masks=masks,
        cost=json.dumps(cost),
    )
    return Path(out)


def load_run(path):
    """npz from `run` -> (init, steps, [per step (n, 3, LAT, LON) masks], cost)."""
    with np.load(path) as f:
        init = pd.Timestamp(f["init"][()])
        steps, members, packed, cost = f["steps"], f["members"], f["masks"], json.loads(str(f["cost"]))
    masks = []
    for i in range(steps.size):
        n = int((members[i] > 0).sum())
        bits = np.unpackbits(packed[i, :n], axis=-1, count=JB.LAT.size * JB.LON.size)  # (n, 3, columns)
        masks.append(bits.reshape(n, 3, JB.LAT.size, JB.LON.size).astype(bool))
    return init, steps, masks, cost


# ── the products ────────────────────────────────────────────────────────────────────────────────────────────────────
def thresholds():
    ref = json.loads((REF / "jet_branches_ref.json").read_text())
    return {r: int(ref["regions"][r]["threshold"]) for r in JB.REGIONS}


def map_normals(clim_path, valid):
    """ERA5 1991-2020 share of analyses with each branch within NEAR_KM, for each valid time's day of the season:
    (lead, 3, LAT, LON)."""
    with xr.open_dataset(clim_path) as ds:
        if "near_count" not in ds or float(ds.attrs.get("near_km", -1)) != JB.NEAR_KM:
            raise SystemExit(f"{clim_path} has no {JB.NEAR_KM:g} km near_count: rebuild jet-branches-ref-v1")
        counts = ds["near_count"].transpose("slot", "branch", "latitude", "longitude").values
        samples = ds["samples"].values
    return JB.smooth_season(counts, samples, at=JB.season_slot(valid))


def event_normals(valid, thr, regions_path=REF / "jet_branches_regions.nc"):
    """ERA5 1991-2020 share of analyses with a superposition event in each region on each valid time's day of the
    season: {region: (lead,)}."""
    with xr.open_dataset(regions_path) as ds:
        slot = JB.season_slot(ds.time.values)
        samples = np.bincount(slot, minlength=152)
        out = {}
        for r in JB.REGIONS:
            hit = ds["largest_group"].sel(region=r).values >= thr[r]
            out[r] = JB.smooth_season(np.bincount(slot, weights=hit, minlength=152), samples, at=JB.season_slot(valid))
    return out


def member_events(masks, thr):
    """(n, 3, LAT, LON) masks of one lead -> {region: share of members with an event}."""
    inside = {r: JB.region_mask(JB.LAT, JB.LON, r) for r in JB.REGIONS}
    return {
        r: float(np.mean([any(g[0] >= thr[r] for g in JB.groups_of(m[2], inside[r])) for m in masks]))
        for r in JB.REGIONS
    }


def products(init, steps, masks, clim_path):
    """Per in-season lead: the neighbourhood shares, their normals, the regions' event shares and normals."""
    valid = pd.DatetimeIndex([init + pd.Timedelta(hours=int(s)) for s in steps])
    ok = JB.season_slot(valid.values) >= 0
    keep = np.flatnonzero(ok)
    valid = valid[keep]
    thr = thresholds()
    share = np.stack([JB.near(masks[i]).mean(axis=0) for i in keep]).astype(np.float32)
    ev = [member_events(masks[i], thr) for i in keep]
    return dict(
        init=init,
        steps=np.asarray(steps)[keep],
        valid=valid,
        members=[int(masks[i].shape[0]) for i in keep],
        share=share,
        normal=map_normals(clim_path, valid.values).astype(np.float32),
        events={r: np.array([e[r] for e in ev]) for r in JB.REGIONS},
        event_normal=event_normals(valid.values, thr),
        thresholds=thr,
    )


def _days(group):
    a, b = (int(x) for x in group.split("-"))
    return range(a, b + 1)


def gate(pr, skill):
    """The verification applied to the products: the adopted recalibration of the shares (its table maps the share
    of the verified members, k/n, to the observed frequency; a share in between is interpolated; 'map_cal' (lead,
    branch) marks where) and which product-leads are shown ('map_shown' (lead, branch), 'event_shown' (lead, region)).
    skill None: all, as they are."""
    days = np.asarray(pr["steps"]) // 24
    out = dict(pr, map_shown=np.ones((days.size, len(JB.BRANCHES)), bool))
    out["map_cal"] = np.zeros((days.size, len(JB.BRANCHES)), bool)
    out["event_shown"] = np.ones((days.size, len(JB.REGIONS)), bool)
    if skill is None:
        return out
    share = pr["share"].copy()
    for b, name in enumerate(JB.BRANCHES):
        m = skill["maps"][name]
        lv = np.linspace(0.0, 1.0, len(m["table"][0]))
        for i, d in enumerate(days):
            known = d < len(m["calibrated"])
            if known and m["calibrated"][d]:
                share[i, b] = np.interp(share[i, b], lv, m["table"][d])
                out["map_cal"][i, b] = True
            out["map_shown"][i, b] = known and m["show"] is not None and m["show"][0] <= d <= m["show"][1]
    out["share"] = share
    events = {}
    for r, name in enumerate(JB.REGIONS):
        e = skill["events"][name]
        group = {d: g for g, grp in enumerate(e["groups"]) for d in _days(grp)}
        lv = np.linspace(0.0, 1.0, len(e["table"][0]))
        x = pr["events"][name].copy()
        for i, d in enumerate(days):
            g = group.get(int(d))
            if g is not None and e["calibrated"][g]:
                x[i] = np.interp(x[i], lv, e["table"][g])
            out["event_shown"][i, r] = g is not None and e["show"] is not None and e["show"][0] <= g <= e["show"][1]
        events[name] = x
    out["events"] = events
    return out


# ── rendering ───────────────────────────────────────────────────────────────────────────────────────────────────────
def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans"})
    return plt


def cyclic(f):
    """Append the 0E column at 360E so the maps close."""
    return np.concatenate([f, f[..., :1]], axis=-1), np.append(JB.LON, 360.0)


def nh_map(fig, rect, title):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax = fig.add_axes(rect, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent((0, 359.99, JB.LAT[0], JB.LAT[-1]), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.6, edgecolor="#333", zorder=4)
    gl = ax.gridlines(
        draw_labels=True, lw=0.3, color="#bbb", xlocs=range(-180, 181, 30), ylocs=range(10, 81, 10), zorder=3
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8.5, "color": INK}
    ax.set_title(title, fontsize=11.5, loc="left", fontweight="bold", color=INK)
    return ax


def draw_regions(ax):
    import cartopy.crs as ccrs

    for r, box in JB.REGIONS.items():
        (la0, la1), (lo0, lo1) = box["lat"], box["lon"]
        ax.plot(
            [lo0, lo1, lo1, lo0, lo0],
            [la0, la0, la1, la1, la0],
            color=INK,
            lw=1.0,
            ls=(0, (2, 2)),
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        ax.text(
            lo0 + 1,
            la1 - 1,
            r.upper(),
            fontsize=8.5,
            color=INK,
            va="top",
            transform=ccrs.PlateCarree(),
            zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1),
        )


def frame(pr, i, out, dpi=90):
    """One lead: the three branches' neighbourhood shares with the ERA5 normal as contours."""
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt = mpl()
    map_w, left, right, top, gap, bot = 11.4, 0.55, 1.05, 1.35, 0.5, 0.62
    mh = map_w * (JB.LAT[-1] - JB.LAT[0]) / 360.0
    W, H = left + map_w + right, top + 3 * mh + 2 * gap + bot
    fig = plt.figure(figsize=(W, H))
    cmap = ListedColormap(P_COL)
    v = pr["valid"][i]
    for k, b in enumerate(JB.BRANCHES):
        rect = [left / W, (bot + (2 - k) * (mh + gap)) / H, map_w / W, mh / H]
        ax = nh_map(fig, rect, TITLE[b])
        if not pr["map_shown"][i, k]:
            ax.text(
                0.5,
                0.5,
                f"Not shown at day {int(pr['steps'][i]) / 24:g}: at this lead the forecast of this branch did not beat "
                "the ERA5 normal on the archived runs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10.5,
                color=MUTED,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
            )
            continue
        f, lon = cyclic(100 * pr["share"][i, k])
        cf = ax.contourf(
            lon, JB.LAT, f, levels=P_LEV, cmap=cmap, norm=BoundaryNorm(P_LEV, cmap.N), transform=ccrs.PlateCarree()
        )
        g, _ = cyclic(100 * pr["normal"][i, k])
        cs = ax.contour(
            lon, JB.LAT, g, levels=CLIM_LEV[b], colors=INK, linewidths=(0.7, 1.0, 1.4), transform=ccrs.PlateCarree()
        )
        ax.clabel(cs, fmt="%d%%", fontsize=7.5, inline_spacing=2)
        if b == "sup":
            draw_regions(ax)
        cax = fig.add_axes([rect[0] + rect[2] + 0.12 / W, rect[1] + 0.05 * rect[3], 0.16 / W, 0.9 * rect[3]])
        cb = fig.colorbar(cf, cax=cax, ticks=P_LEV[::2])
        cb.set_label("chance (%)" if pr["map_cal"][i, k] else "% of members", fontsize=9)
        cb.ax.tick_params(labelsize=8.5)
    day = int(pr["steps"][i]) / 24
    fig.text(
        left / W,
        1 - 0.14 / H,
        f"Jet branches, day {day:g} (valid {v:%a %-d %b %HZ}) — {LABEL} {pr['members'][i]} members, init "
        f"{pr['init']:%Y-%m-%d %HZ}",
        fontsize=13.5,
        fontweight="bold",
        va="top",
        color=INK,
    )
    shading = (
        f"the chance of the branch within {JB.NEAR_KM:g} km (the members' share, recalibrated on archived runs where "
        "that improved it)"
        if pr["map_cal"][i].any()
        else f"share of members with the branch within {JB.NEAR_KM:g} km"
    )
    sub = (
        f"Shading: {shading}. Contours: how often ERA5 had it within {JB.NEAR_KM:g} km on this date, 1991–2020. A jet "
        f"is a 400–100 hPa wind above {JB.V_MIN:g} m/s over a strong PV gradient in the 1–3 PVU channel of its "
        "isentropic layer; dashed boxes: the superposition regions."
    )
    fig.text(left / W, 1 - 0.5 / H, textwrap.fill(sub, int(W * 12.4)), fontsize=9.4, va="top", color=MUTED)
    fig.text(left / W, 0.06 / H, textwrap.fill(CREDIT, int(W * 15.5)), fontsize=8.2, va="bottom", color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def events_still(pr, out):
    """The regions' event shares by lead against the ERA5 rate for the valid day."""
    plt = mpl()
    W, H = 13.0, 5.4
    fig, axs = plt.subplots(1, len(JB.REGIONS), figsize=(W, H), sharey=True)
    days = np.asarray(pr["steps"]) / 24
    for k, (ax, r) in enumerate(zip(axs, JB.REGIONS)):
        on = pr["event_shown"][:, k]
        ax.bar(days[on], 100 * pr["events"][r][on], width=0.4, color="#cc2f2f", label=f"{LABEL} chance of an event")
        for d in days[~on]:
            ax.axvspan(d - 0.25, d + 0.25, color="#ececec", lw=0, zorder=0)
        if not on.all():
            ax.text(
                0.99,
                0.6,
                "grey: leads where the odds did not\nbeat the ERA5 rate on the archived runs",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
                color=MUTED,
            )
        ax.plot(days, 100 * pr["event_normal"][r], color=INK, lw=1.6, ls=(0, (5, 3)), label="ERA5 1991–2020, same date")
        box = JB.REGIONS[r]
        ax.set_title(
            f"{box['label']} ({box['lat'][0]:g}–{box['lat'][1]:g}°N, {box['lon'][0]:g}–{box['lon'][1]:g}°E)",
            fontsize=11.5,
            loc="left",
            fontweight="bold",
            color=INK,
        )
        ax.text(
            0.01,
            0.97,
            f"event: ≥ {pr['thresholds'][r]} superposed columns linked within {JB.LINK_KM:g} km",
            transform=ax.transAxes,
            fontsize=9,
            color=MUTED,
            va="top",
        )
        ax.set_xlabel("lead (days)")
        ax.set_xlim(-0.4, days.max() + 0.4)
        ax.grid(axis="y", color="#e3e3e3", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("chance of an event (%)")
    axs[0].set_ylim(0, 100)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.06, 0.1), ncol=2, fontsize=9.5,
               frameon=False)  # fmt: skip
    fig.suptitle(
        f"Superposition events by region — {LABEL} {max(pr['members'])} members, init {pr['init']:%Y-%m-%d %HZ}",
        x=0.06,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(0.06, 0.012, textwrap.fill(CREDIT, int(W * 15.5)), fontsize=8.2, color=MUTED)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.3, wspace=0.06)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


def season_note(init, out):
    """The still shown outside the cold season."""
    plt = mpl()
    fig = plt.figure(figsize=(13.0, 4.0))
    fig.text(
        0.04,
        0.9,
        f"Jet branches — not shown in {init:%B} ({LABEL} init {init:%Y-%m-%d %HZ})",
        fontsize=15,
        fontweight="bold",
        color=INK,
        va="top",
    )
    msg = (
        "The polar and subtropical jets are told apart by the isentropic layers that house them in the cold season, "
        "315–330 K and 340–355 K (Christenson et al. 2017). In the warm season they become a single jet on other "
        "isentropes (Orr & Martin 2026) and the scheme no longer separates them, so these maps and event odds are made "
        "for runs started from 1 November to 31 March, against the ERA5 1991–2020 cold-season reference."
    )
    fig.text(0.04, 0.72, textwrap.fill(msg, 140), fontsize=11.5, color=INK, va="top", linespacing=1.4)
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.18])
    for m in range(1, 13):
        on = m in (11, 12, 1, 2, 3)
        ax.bar(m, 1, color=("#2e86c1" if on else "#e0e0e0"), edgecolor="white", width=0.95)
        ax.text(
            m,
            0.5,
            pd.Timestamp(2001, m, 1).strftime("%b"),
            ha="center",
            va="center",
            fontsize=10.5,
            color=("white" if on else MUTED),
            fontweight=("bold" if m == init.month else "normal"),
        )
    ax.set_xlim(0.5, 12.5)
    ax.axis("off")
    fig.text(0.04, 0.03, "blue: the months the product covers", fontsize=9, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


def write_manifest(frames, path):
    mani = {
        "ver": int(time.time()),
        "regions": {NAME: {"label": f"Jet branches ({LABEL})", "n_frames": len(frames), "frames": frames}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mani))


def in_season(init) -> bool:
    return bool(JB.season_slot(np.array([pd.Timestamp(init).to_datetime64()]))[0] >= 0)


def _outputs(out_dir):
    """(frames dir emptied of the last run's frames, manifest, events still, json)."""
    out_dir = Path(out_dir)
    anim = out_dir / "anim" / NAME
    anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    return (
        anim,
        out_dir / "anim" / f"{NAME}_manifest.json",
        out_dir / f"{NAME}_events.webp",
        out_dir / "data" / f"{NAME}.json",
    )


def _write_json(doc, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")


def render_off_season(init, out_dir):
    """Outside the cold season: the note as the still and as a one-frame loop, and in_season false."""
    anim, manifest, still, data = _outputs(out_dir)
    season_note(init, still)
    season_note(init, anim / "F00.webp")
    write_manifest(
        [{"idx": 0, "file": "F00.webp", "date": f"{init:%Y-%m-%d}", "label": "outside the cold season"}], manifest
    )
    doc = dict(model=LABEL, init=f"{init:%Y-%m-%dT%H:%MZ}", in_season=False)
    _write_json(doc, data)
    return doc


def render(fc_path, clim_path, out_dir):
    """Frames, manifest, events still and json under out_dir (assets/sst) from a run's npz."""
    init, steps, masks, cost = load_run(fc_path)
    if not in_season(init):
        return render_off_season(init, out_dir)
    anim, manifest, still, data = _outputs(out_dir)
    doc = dict(model=LABEL, init=f"{init:%Y-%m-%dT%H:%MZ}", near_km=JB.NEAR_KM, fetch=cost)
    skill = json.loads(SKILL.read_text()) if SKILL.exists() else None
    pr = gate(products(init, steps, masks, clim_path), skill)
    frames = []
    for i, v in enumerate(pr["valid"]):
        if not pr["map_shown"][i].any():
            continue
        fn = f"F{len(frames):02d}.webp"
        frame(pr, i, anim / fn)
        frames.append(
            {
                "idx": len(frames),
                "file": fn,
                "date": f"{v:%Y-%m-%d}",
                "label": f"day {pr['steps'][i] / 24:g} · valid {v:%a %b %d %HZ} · superposition up to "
                f"{100 * pr['share'][i, 2].max():.0f}%",
            }
        )
    write_manifest(frames, manifest)
    events_still(pr, still)
    doc.update(
        in_season=True,
        verified=skill is not None,
        map_shown={b: [bool(x) for x in pr["map_shown"][:, k]] for k, b in enumerate(JB.BRANCHES)},
        event_shown={r: [bool(x) for x in pr["event_shown"][:, k]] for k, r in enumerate(JB.REGIONS)},
        members=pr["members"],
        lead_h=[int(s) for s in pr["steps"]],
        valid=[f"{v:%Y-%m-%dT%H:%MZ}" for v in pr["valid"]],
        max_share={
            b: [round(float(x), 3) for x in pr["share"][:, k].max(axis=(1, 2))] for k, b in enumerate(JB.BRANCHES)
        },
        regions={
            r: dict(
                label=JB.REGIONS[r]["label"],
                threshold=pr["thresholds"][r],
                share=[round(float(x), 3) for x in pr["events"][r]],
                normal=[round(float(x), 4) for x in pr["event_normal"][r]],
            )
            for r in JB.REGIONS
        },
    )
    print(f"  {len(frames)} frames -> {anim}", flush=True)
    _write_json(doc, data)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "forecast"):
        s = sub.add_parser(name)
        s.add_argument("--date", required=True, help="YYYYMMDD of the 00 UTC run")
        s.add_argument("--last-step", type=int, default=240)
        s.add_argument("--workers", type=int, default=3)
        s.add_argument("--members", type=int, default=MEMBERS, help="perturbed members 1..N")
        s.add_argument("--wait-min", type=float, default=0.0, help="wait this long for the run on the mirror")
        s.add_argument("--out", default="jetbr_fc.npz")
        if name == "forecast":
            s.add_argument("--clim", required=True)
            s.add_argument("--out-dir", required=True)
    s = sub.add_parser("render")
    s.add_argument("--fc", required=True)
    s.add_argument("--clim", required=True)
    s.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "render":
        render(a.fc, a.clim, a.out_dir)
        return 0
    init = pd.Timestamp(f"{a.date} {HOUR}:00")
    if a.cmd == "forecast" and not in_season(init):
        render_off_season(init, a.out_dir)  # nothing to fetch
        return 0
    run(a.date, a.last_step, a.out, a.workers, a.wait_min, a.members)
    if a.cmd == "forecast":
        render(a.out, a.clim, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
