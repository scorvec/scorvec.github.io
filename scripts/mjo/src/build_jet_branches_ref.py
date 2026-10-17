#!/usr/bin/env python3
"""ERA5 reference of the polar jet, the subtropical jet and their superposition (jet_branches.py): the cold seasons
(November-March) of 1991-2020 at 00 and 12 UTC. Built in Actions (jet-branches-ref.yml).

Source: WeatherBench 2's ERA5 at 0.25 deg, 13 pressure levels, 6-hourly (anonymous GCS), which holds the nine levels
the forecasts carry. One zarr chunk is one variable at one time on all 13 levels (30-43 MB), so an analysis costs
three chunks, ~115 MB; a season ~35 GB.

  fetch    one cold season (November of the year before --winter to March of --winter) -> npz:
             the per-analysis masks of the three branches on the 0.5 deg grid (bit-packed), the monthly sums of the
             400-100 hPa wind under each branch, histograms of the in-channel PV gradient of the windy columns, and the
             sensitivity variants below as season counts and per-analysis region counts
  combine  parts -> the reference files:
             jet_branches_clim.nc     per-slot counts of each branch (152-slot season, jet_branches.season_slot), the
                                      analyses per slot, monthly counts and mean 400-100 hPa wind of each branch
             jet_branches_calib.nc    the variants' cold-season counts
             reference/jet_branches_regions.nc   per analysis and region: superposed columns, the largest group
             reference/jet_branches_events.csv   the superposition events
             reference/jet_branches_ref.json     thresholds, each region's event threshold, the median in-channel
                                                 gradients (the published thresholds lie above them), the variants
                                                 against the published climatologies
           and, with --figure, the validation maps (scratch: cold-season frequency of each branch and of the
           superposition on the 2.5 deg grid of the published climatology)

Events (Winters et al. 2020): in each region, the analyses in the top 10 % by superposed columns among the analyses
with any (the region's threshold n); in those, a group of >= n superposed columns, each within 1000 km of another, is
an event at that analysis; an event at an analysis whose group centre lies within 1500 km of one at an analysis up to
30 h earlier continues that event.

Sensitivity variants (all from the same fields, so the differences are the choice alone). The papers name "the PV
gradient within the 1-3 PVU channel" in a layer; three readings of it (METHODS):
  mean     the gradient of the mean PV of the layer's four isentropes (jet_branches.py; the product, "main")
  iso      the gradient on each of the layer's isentropes in its own channel, the layer's largest
  mass     the gradient of the layer's bulk PV: the mass-weighted (dp-weighted) mean over its three 5 K sublayers, what
           an isentropic-layer PV, -g (zeta + f) dtheta/dp across the layer, computes
and
  scan_x   the product's thresholds times x
  r1, r1_iso, r1_mass   the three on the 2.5 deg grid of the NCEP-NCAR reanalysis (5 x 5 means of the 0.5 deg grid,
           the same nine levels) with Christenson et al.'s threshold, 0.64e-5 PVU/m for both layers: the setting of
           the published climatology (Christenson et al. 2017; Handlos & Martin 2016)

    python build_jet_branches_ref.py fetch --winter 2010 --out parts/jb_2010.npz
    python build_jet_branches_ref.py combine parts/jb_*.npz --release-dir out --figure out/jet_branches_check.png
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.pv import on_isentrope

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jet_branches as JB

REF = HERE.parent / "data" / "reference"
WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
VARS = {"t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind"}
HOURS = (0, 12)
WINTERS = range(1991, 2021)  # cold seasons by the year of their January
MONTHS = (11, 12, 1, 2, 3)
SCAN = (0.6, 0.7, 0.8, 0.9, 1.1, 1.25)
R1_THRESH = {"polar": 0.64e-5, "stj": 0.64e-5}  # Christenson et al. (2017) on the 2.5 deg grid
R1_BAND = np.round(np.arange(5.0, 85.0 + 1e-6, 2.5), 3)  # 5 x 5 blocks of the 0.5 deg band rows 4..86N
R1_LON = np.round(np.arange(0.0, 360.0, 2.5), 3)
R1_LAT = R1_BAND[JB.output_rows(R1_BAND)]
METHODS = ("mean", "iso", "mass")
R1_NAMES = ("r1", "r1_iso", "r1_mass")  # the METHODS on the 2.5 deg grid
HIST_EDGES = np.linspace(-7.0, -3.0, 401)  # log10 of the in-channel |grad PV| (PVU/m)
TRACK_KM, TRACK_H = 1500.0, 30.0
# one row per event: its first and last analysis, how many analyses it held, its largest group, its first centre
EVENT_COLUMNS = ("region", "event", "onset", "end", "analyses", "peak_columns", "lat", "lon")
# The published cold-season superposition climatology, for the validation: frequency maxima (percent of analyses) by
# area, from Christenson et al. (2017) Fig. 3c as reproduced in Handlos & Martin (2021) Fig. 1f and its text (NCEP-NCAR,
# 2.5 deg, NDJFM 1960-2010), and the western North Pacific box counts of Handlos & Martin (2016, Table 2.1 of Handlos's
# thesis; DJF 1979/80-2009/10, 6-hourly).
CHECK_BOXES = {
    "wnp": dict(label="western North Pacific", lat=(25.0, 45.0), lon=(120.0, 180.0), published="1-2 %"),
    "na": dict(label="southern North America and the western Atlantic", lat=(20.0, 50.0), lon=(230.0, 310.0),
               published="about 0.5 %"),
    "naf": dict(label="northern Africa and the eastern Mediterranean", lat=(20.0, 40.0), lon=(350.0, 50.0),
                published="about 0.5 %"),
}  # fmt: skip
HANDLOS_DJF = {1: 13.0, 2: 7.72, 3: 4.77, 4: 2.77, 5: 1.65, 6: 1.02, 7: 0.591}  # % of DJF analyses with >= X in the box
WINTERS2020 = dict(na_threshold=18, na_events=326, seasons=31)  # CFSR 0.5 deg, 6-hourly NDJFM 1979-2010


# ── fetch ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def screen(t, u, v, what):
    """Refuse a field that is not temperature and wind on the nine levels."""
    if not (np.isfinite(t).all() and np.isfinite(u).all() and np.isfinite(v).all()):
        raise RuntimeError(f"{what}: non-finite values")
    if t.min() < 170 or t.max() > 330 or max(np.abs(u).max(), np.abs(v).max()) > 200:
        raise RuntimeError(f"{what}: T {t.min():.0f}..{t.max():.0f} K, wind up to {np.abs(u).max():.0f} m/s")


def to_r1(f05):
    """0.5 deg band (..., BAND, 720) -> the 2.5 deg band (..., R1_BAND, 144): the mean of the 5 x 5 cells whose union is
    the 2.5 deg cell (centred on it; longitudes wrap)."""
    i0 = int(np.flatnonzero(JB.BAND == R1_BAND[0] - 1.0)[0])  # noqa: SIM300  coordinate array BAND, not a constant
    a = np.roll(f05[..., i0 : i0 + 5 * R1_BAND.size, :], 2, axis=-1)
    s = a.shape[:-2]
    return a.reshape(*s, R1_BAND.size, 5, R1_LON.size, 5).mean(axis=(-3, -1))


def gradient_sets(d, lat, rows):
    """{method: {layer: in-channel gradient}} at the output rows for the three METHODS, from jet_branches.diagnose."""
    p = np.broadcast_to(np.asarray(JB.LEVS, float)[:, None, None], d["theta"].shape)
    out = {"mean": d["grad"], "iso": {}, "mass": {}}
    for name, thetas in JB.LAYERS.items():
        pv = d["pv"][name]
        out["iso"][name] = np.fmax.reduce([JB.channel_gradient(x, lat) for x in pv])
        dp = -np.diff(np.stack([on_isentrope(p, d["theta"], p, p, th0)["pv"] for th0 in thetas]), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            bulk = np.where(dp.sum(0) > 0, (0.5 * (pv[:-1] + pv[1:]) * dp).sum(0) / dp.sum(0), np.nan)
        out["mass"][name] = JB.channel_gradient(bulk, lat)
    return {m: {k: x[rows] for k, x in g.items()} for m, g in out.items()}


def masks_of(m):
    return np.stack([m[b] for b in JB.BRANCHES])


def analyses(winter, days=None):
    d = pd.date_range(f"{winter - 1}-11-01", f"{winter}-03-31")
    if days:
        d = d[:days]
    return [x + pd.Timedelta(hours=h) for x in d for h in HOURS]


def reader(ds):
    lat = ds.latitude.values
    rows = JB.band_rows025(lat)
    sl = slice(int(rows.min()), int(rows.max()) + 1)
    flip = rows[0] > rows[-1]  # the store runs north to south

    def read(ts):
        for attempt in range(4):
            try:
                out = []
                for name in VARS.values():
                    da = ds[name].sel(time=ts, level=list(JB.LEVS)).isel(latitude=sl)
                    a = da.transpose("level", "latitude", "longitude").values
                    out.append(a[:, ::-1] if flip else a)
                return ts, out
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                print(f"  {ts:%Y-%m-%d %H}: {type(e).__name__}, retrying", flush=True)
                time.sleep(10 * (attempt + 1))

    return read


def fetch(winter, out, workers=8, days=None):
    times = analyses(winter, days)
    ds = xr.open_zarr(WB2, chunks=None, storage_options={"token": "anon"})
    read = reader(ds)
    sl = JB.output_rows(JB.BAND)
    reg = {r: JB.region_mask(JB.LAT, JB.LON, r) for r in JB.REGIONS}
    reg1 = {r: JB.region_mask(R1_LAT, R1_LON, r) for r in JB.REGIONS}
    names = ["main", *(f"scan_{x:g}" for x in SCAN), *METHODS[1:]]
    shape, shape1 = (3, JB.LAT.size, JB.LON.size), (3, R1_LAT.size, R1_LON.size)
    var_maps = np.zeros((len(names), *shape), np.uint16)
    r1_maps = np.zeros((len(METHODS), *shape1), np.uint16)
    vsum = np.zeros((len(MONTHS), *shape))
    hist = np.zeros((2, len(METHODS), 2, HIST_EDGES.size - 1), np.int64)  # grid (0.5, 2.5 deg), method, layer
    packed, var_reg, r1_reg = [], [], []
    t0 = time.time()

    def histogram(k, vbar, gs):
        windy = vbar > JB.V_MIN
        for j, m in enumerate(METHODS):
            for i, name in enumerate(JB.LAYERS):
                x = gs[m][name][windy & np.isfinite(gs[m][name])]
                hist[k, j, i] += np.histogram(np.log10(x), HIST_EDGES)[0]

    with ThreadPoolExecutor(workers) as ex:
        it = iter(times)
        pending = deque(ex.submit(read, ts) for ts in islice(it, 2 * workers))  # a bounded read-ahead
        while pending:
            ts, (t, u, v) = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(ex.submit(read, nxt))
            screen(t, u, v, f"{ts:%Y-%m-%d %H}")
            t05, u05, v05 = (JB.to_half(x) for x in (t, u, v))
            d = JB.diagnose(t05, u05, v05, JB.BAND, JB.LON)
            vb = d["vbar"][sl]
            gs = gradient_sets(d, JB.BAND, sl)
            g = gs["mean"]
            ms = [JB.identify(vb, g["polar"], g["stj"])]
            ms += [JB.identify(vb, g["polar"], g["stj"], {k: x * c for k, c in JB.THRESH.items()}) for x in SCAN]
            ms += [JB.identify(vb, gs[m]["polar"], gs[m]["stj"]) for m in METHODS[1:]]
            main = masks_of(ms[0])
            packed.append(np.packbits(main.reshape(3, -1), axis=-1))
            vsum[MONTHS.index(ts.month)] += main * vb
            for k, m in enumerate(ms):
                var_maps[k] += masks_of(m)
            var_reg.append([[int(m["sup"][reg[r]].sum()) for r in JB.REGIONS] for m in ms])
            histogram(0, vb, gs)
            # the 2.5 deg grid of the published climatology
            d1 = JB.diagnose(*(to_r1(x) for x in (t05, u05, v05)), R1_BAND, R1_LON)
            s1 = JB.output_rows(R1_BAND)
            gs1 = gradient_sets(d1, R1_BAND, s1)
            vb1 = d1["vbar"][s1]
            m1 = [JB.identify(vb1, gs1[m]["polar"], gs1[m]["stj"], R1_THRESH) for m in METHODS]
            for k, m in enumerate(m1):
                r1_maps[k] += masks_of(m)
            r1_reg.append([[int(m["sup"][reg1[r]].sum()) for r in JB.REGIONS] for m in m1])
            histogram(1, vb1, gs1)
            if ts.hour == 0 and ts.day == 1:
                print(f"  {ts:%Y-%m}: {len(packed)} analyses, {time.time() - t0:.0f} s", flush=True)
    print(f"winter {winter}: {len(packed)} analyses in {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(
        out,
        winter=winter,
        time=np.array(times, "datetime64[s]"),
        masks=np.stack(packed),
        vsum=vsum.astype(np.float32),
        var_names=np.array(names),
        var_maps=var_maps,
        var_reg=np.array(var_reg, np.int16),
        r1_maps=r1_maps,
        r1_reg=np.array(r1_reg, np.int16),
        hist=hist,
    )


# ── combine ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_parts(paths, allow_partial):
    parts = sorted((dict(np.load(p)) for p in paths), key=lambda f: int(f["winter"]))
    got = [int(f["winter"]) for f in parts]
    missing = sorted(set(WINTERS) - set(got))
    if missing and not allow_partial:
        raise SystemExit(f"missing winters {missing}: refusing a partial reference")
    if len(set(got)) != len(got):
        raise SystemExit(f"a winter appears twice: {got}")
    names = parts[0]["var_names"]
    if any(not np.array_equal(f["var_names"], names) for f in parts):
        raise SystemExit("the parts were built with different variants")
    return parts, got


def unpack(packed):
    n = JB.LAT.size * JB.LON.size
    return np.unpackbits(packed, axis=-1, count=n).reshape(3, JB.LAT.size, JB.LON.size).astype(bool)


def box_mask(lat, lon, box):
    la = (lat >= box["lat"][0]) & (lat <= box["lat"][1])
    lo0, lo1 = box["lon"]
    lo = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else (lon >= lo0) | (lon <= lo1)
    return la[:, None] & lo[None, :]


def box_max(freq, lat, lon):
    """{box: (max %, lat, lon)} of a frequency map."""
    out = {}
    for k, box in CHECK_BOXES.items():
        f = np.where(box_mask(lat, lon, box), freq, -1.0)
        i, j = np.unravel_index(int(np.argmax(f)), f.shape)
        out[k] = [round(100 * float(f[i, j]), 2), float(lat[i]), float(lon[j])]
    return out


def region_stats(counts):
    """counts (time,) of superposed columns in a region -> share of analyses with any, and the event threshold: the
    count at the 90th percentile of the analyses with any (Winters et al. 2020: the top 10 %)."""
    pos = counts[counts > 0]
    thr = int(np.percentile(pos, 90, method="higher")) if pos.size else 0
    return {"share_any": round(100 * pos.size / max(counts.size, 1), 2), "threshold": thr}


def groups_of(mask, region_mask):
    """Superposed columns of a region at one analysis -> [(size, centre lat, centre lon)], largest first."""
    ii, jj = np.nonzero(mask & region_mask)
    if not ii.size:
        return []
    la, lo = JB.LAT[ii], JB.LON[jj]
    lab = JB.groups(la, lo)
    out = []
    for g in range(lab.max() + 1):
        sel = lab == g
        z = np.exp(1j * np.deg2rad(lo[sel])).mean()
        out.append((int(sel.sum()), float(la[sel].mean()), float(np.rad2deg(np.angle(z)) % 360)))
    return sorted(out, reverse=True)


def track(times, found, thr):
    """found: per analysis, its groups -> event rows. A group of >= thr columns is an event at its analysis; it
    continues the event of a group up to TRACK_H earlier whose centre is within TRACK_KM (the nearest), else starts
    one."""
    rows, recent = [], []  # recent: (time, lat, lon, event index)
    for t, gs in zip(times, found):
        recent = [r for r in recent if (t - r[0]) <= pd.Timedelta(hours=TRACK_H)]
        now = []
        for size, la, lo in gs:
            if size < thr:
                continue
            near = [(JB.great_circle_km(la, lo, r[1], r[2]), r[3]) for r in recent if r[0] < t]
            near = [x for x in near if x[0] <= TRACK_KM]
            if near:
                e = min(near)[1]
                ev = rows[e]
                if ev["end"] != t:  # two groups of one analysis joining one event count it once
                    ev["end"], ev["analyses"] = t, ev["analyses"] + 1
                ev["peak_columns"] = max(ev["peak_columns"], size)
            else:
                e = len(rows)
                rows.append(dict(onset=t, end=t, analyses=1, peak_columns=size, lat=round(la, 2), lon=round(lo, 2)))
            now.append((t, la, lo, e))
        recent += now
    return rows


def hist_median(h):
    c = np.cumsum(h)
    if not c[-1]:
        return float("nan")
    k = int(np.searchsorted(c, 0.5 * c[-1]))
    return float(10 ** (0.5 * (HIST_EDGES[k] + HIST_EDGES[k + 1])))


def combine(paths, release_dir, figure=None, allow_partial=False):
    parts, got = load_parts(paths, allow_partial)
    times = pd.DatetimeIndex(np.concatenate([f["time"] for f in parts]))
    packed = np.concatenate([f["masks"] for f in parts])
    print(f"{len(got)} winters ({got[0]}-{got[-1]}), {len(times)} analyses", flush=True)
    names = [str(x) for x in parts[0]["var_names"]]

    # per-slot and monthly counts, the monthly wind under each branch
    slot = JB.season_slot(times.values)
    samples = np.bincount(slot, minlength=152)
    counts = np.zeros((152, 3, JB.LAT.size, JB.LON.size), np.uint8)
    month = np.array([MONTHS.index(m) for m in times.month])
    month_samples = np.bincount(month, minlength=len(MONTHS))
    mcount = np.zeros((len(MONTHS), 3, JB.LAT.size, JB.LON.size), np.uint16)
    regmask = {r: JB.region_mask(JB.LAT, JB.LON, r) for r in JB.REGIONS}
    found = {r: [] for r in JB.REGIONS}
    nsup = {r: np.zeros(len(times), np.int16) for r in JB.REGIONS}
    for k in range(len(times)):
        m = unpack(packed[k])
        counts[slot[k]] += m
        mcount[month[k]] += m
        for r in JB.REGIONS:
            nsup[r][k] = int((m[2] & regmask[r]).sum())
            found[r].append(groups_of(m[2], regmask[r]))
    vsum = sum(f["vsum"].astype(np.float64) for f in parts)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(mcount > 0, vsum / mcount, np.nan).astype(np.float32)

    stats = {r: region_stats(nsup[r]) for r in JB.REGIONS}
    events = []
    for r in JB.REGIONS:
        ev = track(times, found[r], max(stats[r]["threshold"], 1))
        stats[r]["events"] = len(ev)
        stats[r]["events_per_season"] = round(len(ev) / len(got), 2)
        events += [dict(region=r, event=i + 1, **e) for i, e in enumerate(ev)]
    largest = {r: np.array([g[0][0] if g else 0 for g in found[r]], np.int16) for r in JB.REGIONS}

    # outputs
    release_dir = Path(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    attrs = dict(
        source="ERA5 (WeatherBench 2, 0.25 deg, 13 levels) reduced to 0.5 deg, 00 and 12 UTC",
        seasons=f"November-March, winters {got[0]}-{got[-1]} (by the year of January)",
        method="jet_branches.py: Winters & Martin 2014; Christenson et al. 2017; Winters et al. 2020",
        threshold_polar=JB.THRESH["polar"],
        threshold_stj=JB.THRESH["stj"],
        wind_min=JB.V_MIN,
    )
    coords = dict(branch=list(JB.BRANCHES), latitude=JB.LAT, longitude=JB.LON)
    clim = xr.Dataset(
        {
            "count": (("slot", "branch", "latitude", "longitude"), counts),
            "samples": (("slot",), samples.astype(np.int16)),
            "month_count": (("month", "branch", "latitude", "longitude"), mcount),
            "month_samples": (("month",), month_samples.astype(np.int16)),
            "month_wind": (("month", "branch", "latitude", "longitude"), speed),
        },
        coords=dict(coords, slot=np.arange(152), slot_position=("slot", JB.slot_position()), month=list(MONTHS)),
        attrs=dict(
            attrs,
            slots="jet_branches.season_slot: 1 November = 0 ... 29 February = 120 ... 31 March = 151",
            frequency="jet_branches.smooth_season(count, samples): the centred 31-day window",
        ),
    )
    clim["month_wind"].attrs = dict(units="m s-1", long_name="mean 400-100 hPa wind of the columns holding the branch")
    enc = {k: dict(zlib=True, complevel=6) for k in clim.data_vars}
    enc["month_wind"].update(dtype="int16", scale_factor=0.01, _FillValue=-32768)
    clim.to_netcdf(release_dir / "jet_branches_clim.nc", encoding=enc)

    var_maps = sum(f["var_maps"].astype(np.int32) for f in parts)
    r1_maps = sum(f["r1_maps"].astype(np.int32) for f in parts)
    calib = xr.Dataset(
        {
            "count": (("variant", "branch", "latitude", "longitude"), var_maps),
            "count_r1": (("variant_r1", "branch", "latitude_r1", "longitude_r1"), r1_maps),
        },
        coords=dict(coords, variant=names, variant_r1=list(R1_NAMES), latitude_r1=R1_LAT, longitude_r1=R1_LON),
        attrs=dict(attrs, analyses=len(times), scan=list(SCAN), threshold_r1=R1_THRESH["polar"]),
    )
    calib.to_netcdf(
        release_dir / "jet_branches_calib.nc", encoding={k: dict(zlib=True, complevel=6) for k in calib.data_vars}
    )

    regions = xr.Dataset(
        {
            "superposed": (("region", "time"), np.stack([nsup[r] for r in JB.REGIONS])),
            "largest_group": (("region", "time"), np.stack([largest[r] for r in JB.REGIONS])),
        },
        coords=dict(region=list(JB.REGIONS), time=times.values),
        attrs=dict(attrs, groups=f"single linkage within {JB.LINK_KM:g} km"),
    )
    enc = {k: dict(zlib=True, complevel=6) for k in regions.data_vars}
    regions.to_netcdf(REF / "jet_branches_regions.nc", encoding=enc)
    pd.DataFrame(events, columns=EVENT_COLUMNS).to_csv(
        REF / "jet_branches_events.csv", index=False, date_format="%Y-%m-%d %H:%M"
    )

    summary = validation(parts, names, var_maps, r1_maps, len(times), times)
    ref = dict(
        attrs,
        layers={k: list(v) for k, v in JB.LAYERS.items()},
        levels=list(JB.LEVS),
        channel_pvu=list(JB.CHANNEL),
        analyses=len(times),
        regions={r: dict(JB.REGIONS[r], **stats[r]) for r in JB.REGIONS},
        smoothing_days=2 * JB.SMOOTH_HALF + 1,
        **summary,
    )
    (REF / "jet_branches_ref.json").write_text(json.dumps(ref, indent=1) + "\n")
    print(json.dumps({r: ref["regions"][r] for r in JB.REGIONS}, indent=1))
    if figure:
        draw(var_maps[0], r1_maps[0], len(times), figure, f"{got[0]}-{got[-1]}")


def validation(parts, names, var_maps, r1_maps, n, times):
    """The median in-channel gradients, and each variant against the published climatologies."""
    hist = sum(f["hist"] for f in parts)
    med = {
        grid: {
            method: {layer: hist_median(hist[g, k, i]) for i, layer in enumerate(JB.LAYERS)}
            for k, method in enumerate(METHODS)
        }
        for g, grid in enumerate(("0.5deg", "2.5deg"))
    }
    var_reg = np.concatenate([f["var_reg"] for f in parts]).astype(int)  # (time, variant, region)
    r1_reg = np.concatenate([f["r1_reg"] for f in parts]).astype(int)
    djf = times.month.isin((12, 1, 2))
    out = {}
    for k, name in enumerate(names):
        out[name] = dict(
            sup_max=box_max(var_maps[k, 2] / n, JB.LAT, JB.LON),
            **{r: region_stats(var_reg[:, k, i]) for i, r in enumerate(JB.REGIONS)},
        )
    for k, name in enumerate(R1_NAMES):
        wnp = r1_reg[djf, k, list(JB.REGIONS).index("wnp")]
        out[name] = dict(
            sup_max=box_max(r1_maps[k, 2] / n, R1_LAT, R1_LON),
            wnp_djf_at_least={x: round(100 * float((wnp >= x).mean()), 3) if wnp.size else None for x in HANDLOS_DJF},
        )
    print("Median in-channel |grad PV| of the windy columns, PVU/m (Christenson et al. set their threshold above it)")
    for grid, by in med.items():
        for method, v in by.items():
            print(f"  {grid} {method:5s} polar {v['polar']:.2e}  stj {v['stj']:.2e}")
    print("  published: 0.5 deg 1.4e-5 / 0.9e-5 (Winters et al. 2020), 2.5 deg 0.64e-5 both (Christenson et al.)")
    print("Cold-season superposition frequency maxima (%, lat, lon) by area; published (2.5 deg): "
          + ", ".join(f"{b['label']} {b['published']}" for b in CHECK_BOXES.values()))  # fmt: skip
    for name, v in out.items():
        extra = "" if name.startswith("r1") else f"  NA threshold {v['na']['threshold']} (published 18)"
        print(f"  {name:10s} " + "  ".join(f"{k} {x}" for k, x in v["sup_max"].items()) + extra)
    for name in R1_NAMES:
        got = out[name]["wnp_djf_at_least"]
        print(f"  {name} DJF analyses with >= X superposed columns in the western North Pacific box (%): "
              + ", ".join(f"{x}: {got[x]} ({HANDLOS_DJF[x]})" for x in HANDLOS_DJF))  # fmt: skip
    return dict(
        threshold_rule_medians=med,
        variants=out,
        published=dict(
            cold_season_max_by_area={k: b["published"] for k, b in CHECK_BOXES.items()},
            wnp_djf_at_least=HANDLOS_DJF,
            winters2020=WINTERS2020,
        ),
    )


def draw(main, r1, n, path, period):
    """Scratch validation maps: cold-season frequency (% of analyses) of each branch on the 0.5 deg grid and of the
    superposition on the 2.5 deg grid with the published threshold."""
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(
        4, 1, figsize=(11, 13.5), subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180))
    )
    panels = (
        (JB.LAT, JB.LON, main[0], "Polar jet, 0.5 deg", 60),
        (JB.LAT, JB.LON, main[1], "Subtropical jet, 0.5 deg", 60),
        (JB.LAT, JB.LON, main[2], "Superposition, 0.5 deg (1.4e-5 / 0.9e-5 PVU/m)", 2),
        (R1_LAT, R1_LON, r1[2], "Superposition, 2.5 deg (0.64e-5 PVU/m), the published setting", 2),
    )
    for ax, (la, lo, c, title, vmax) in zip(axs, panels):
        f = np.ma.masked_less(100.0 * c / n, 0.05)
        pm = ax.pcolormesh(
            lo, la, f, transform=ccrs.PlateCarree(), cmap="viridis", vmin=0, vmax=vmax, shading="nearest"
        )
        ax.coastlines(lw=0.6)
        ax.set_extent((0, 359.9, 0, 80), crs=ccrs.PlateCarree())
        ax.set_title(f"{title}: % of analyses, NDJFM {period}", fontsize=10)
        fig.colorbar(pm, ax=ax, shrink=0.8, pad=0.01)
    fig.savefig(path, dpi=110, bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch")
    s.add_argument("--winter", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--days", type=int, help="only the first DAYS days of the season (a quick check)")
    s = sub.add_parser("combine")
    s.add_argument("parts", nargs="+")
    s.add_argument("--release-dir", required=True)
    s.add_argument("--figure")
    s.add_argument("--allow-partial", action="store_true", help="combine fewer winters than 1991-2020 (a check run)")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.winter, a.out, a.workers, a.days)
    else:
        combine(a.parts, a.release_dir, a.figure, a.allow_partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
