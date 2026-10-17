#!/usr/bin/env python3
"""Offline checks of jet_branches_fc.py and jet_branches_hindcast.py: a step of the IFS-ENS stream decoded from real
GRIB2 messages (encoded here, served by a stand-in for the range fetcher), the products, the renders, the season rule,
and the hindcast's scan events and comparison.

    python scripts/mjo/src/test_jet_branches_fc.py

The fields are test_jet_branches.py's synthetic stepped tropopause on the 0.25 deg global grid. The messages are laid
out as an IFS-ENS `ef` file is - every member's every field in an arrival order that mixes members and parameters -
and their grid starts at 180W, as ECMWF open data's does, so the decode's member keying and column order are tested.
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jet_branches as JB
import jet_branches_fc as FC
import jet_branches_hindcast as HC
import test_jet_branches as T

LAT025 = np.round(np.arange(90.0, -90.0 - 1e-6, -0.25), 3)
LON180 = np.round(np.arange(-180.0, 180.0, 0.25), 3)  # the open-data grid starts at 180W


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def global_fields(shift):
    """t, u, v (lev, 721, 1440) on the open-data grid: the synthetic atmosphere (mirrored south) with its steps moved
    `shift` degrees north."""
    t, u, v, _, _ = T.synthetic(lat=np.abs(LAT025) - shift, lon=LON180 % 360)
    return t, u, v


def ef_file(members):
    """GRIB2 bytes and index entries of an `ef` step holding t, u, v at LEVS for the given members, in a shuffled
    arrival order."""
    import eccodes

    keys = [(m, p, lev) for m in members for p in FC.PARAMS for lev in JB.LEVS]
    order = np.random.default_rng(7).permutation(len(keys))
    fields = {m: dict(zip(FC.PARAMS, global_fields(0.5 * m))) for m in members}
    blob, index = bytearray(), []
    for k in order:
        m, p, lev = keys[k]
        g = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
        try:
            for key, val in (
                ("Ni", 1440),
                ("Nj", 721),
                ("latitudeOfFirstGridPointInDegrees", 90.0),
                ("longitudeOfFirstGridPointInDegrees", -180.0),
                ("latitudeOfLastGridPointInDegrees", -90.0),
                ("longitudeOfLastGridPointInDegrees", 179.75),
                ("iDirectionIncrementInDegrees", 0.25),
                ("jDirectionIncrementInDegrees", 0.25),
                ("level", lev),
                ("bitsPerValue", 24),
            ):
                eccodes.codes_set(g, key, val)
            eccodes.codes_set_values(g, fields[m][p][JB.LEVS.index(lev)].ravel().astype(float))
            msg = eccodes.codes_get_message(g)
        finally:
            eccodes.codes_release(g)
        index.append(dict(_offset=len(blob), _length=len(msg), param=p, levelist=str(lev), number=str(m), type="pf"))
        blob += msg
    return bytes(blob), index


def test_step():
    import rangefetch as rf

    blob, index = ef_file([1, 2])
    orig = rf.fetch_index, rf.fetch_ranges
    rf.fetch_index = lambda *a, **k: index
    rf.fetch_ranges = lambda rel, ranges, sources=None, workers=None, stats=None: b"".join(
        blob[e["_offset"] : e["_offset"] + e["_length"]] for _, _, es in ranges for e in es
    )
    try:
        r = FC.step_masks(("20260110", 24))
    finally:
        rf.fetch_index, rf.fetch_ranges = orig
    check(r is not None and r["members"].tolist() == [1, 2], f"members {None if r is None else r['members']}")
    rows = JB.band_rows025(LAT025)
    order = np.argsort(LON180 % 360)
    for i, m in enumerate((1, 2)):
        want = JB.masks_025(*(x[:, rows][:, :, order] for x in global_fields(0.5 * m)))
        got = np.unpackbits(r["masks"][i], axis=-1, count=JB.LAT.size * JB.LON.size).reshape(want.shape).astype(bool)
        # 24-bit packing leaves the identification unchanged up to a few columns where a gradient sits on its threshold
        check(np.mean(got != want) < 1e-4 and got.sum() > 100, f"member {m}: {int((got != want).sum())} columns differ")
    check(r["cost"]["messages"] == 54 and r["cost"]["bytes"] == len(blob), f"cost {r['cost']}")


def fake_run(path, init, steps, members=6):
    """A run npz like jet_branches_fc.run writes: members' masks from the synthetic atmosphere, steps moving north."""
    rows = JB.band_rows025(LAT025)
    order = np.argsort(LON180 % 360)
    masks = []
    for s in steps:
        per = [
            JB.masks_025(*(x[:, rows][:, :, order] for x in global_fields(0.1 * m + s / 48))) for m in range(members)
        ]
        masks.append([np.packbits(x.reshape(3, -1), axis=-1) for x in per])
    np.savez_compressed(
        path,
        init=np.datetime64(init, "s"),
        steps=np.array(steps),
        members=np.tile(np.arange(1, members + 1), (len(steps), 1)),
        masks=np.array(masks, np.uint8),
        cost=json.dumps({"bytes": 1}),
    )


def fake_clim(path):
    """A clim file shaped as jet-branches-ref-v1's, with a uniform 4 % near_count frequency."""
    shape = (152, 3, JB.LAT.size, JB.LON.size)
    ds = xr.Dataset(
        {"near_count": (("slot", "branch", "latitude", "longitude"), np.full(shape, 2, np.uint8)),
         "samples": (("slot",), np.full(152, 50, np.int16))},
        coords=dict(branch=list(JB.BRANCHES), latitude=JB.LAT, longitude=JB.LON),
        attrs=dict(near_km=JB.NEAR_KM),
    )  # fmt: skip
    ds.to_netcdf(path)


def test_products_and_render():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        fake_run(d / "fc.npz", pd.Timestamp("2026-03-30 00:00"), [0, 12, 24, 36, 48, 60])
        fake_clim(d / "clim.nc")
        init, steps, masks, _ = FC.load_run(d / "fc.npz")
        pr = FC.products(init, steps, masks, d / "clim.nc")
        # 2026-03-30 00Z + 60 h runs past 31 March: the leads valid in April are dropped
        check(pr["steps"].tolist() == [0, 12, 24, 36], f"in-season leads {pr['steps']}")
        check(np.allclose(pr["normal"], 0.04), "the normal from the clim file")
        share = pr["share"][0]
        near = JB.near(masks[0]).mean(axis=0)
        check(np.array_equal(share, near.astype(np.float32)), "the shares are the members' mean neighbourhood masks")
        check(
            set(pr["events"]) == set(JB.REGIONS) and all(0 <= x <= 1 for r in pr["events"].values() for x in r),
            "events",
        )
        check(all(0 < x < 1 for r in pr["event_normal"].values() for x in r), f"event normals {pr['event_normal']}")
        doc = FC.render(d / "fc.npz", d / "clim.nc", d / "out")
        mani = json.loads((d / "out" / "anim" / f"{FC.NAME}_manifest.json").read_text())
        frames = mani["regions"][FC.NAME]["frames"]
        check(len(frames) == 4 and all((d / "out" / "anim" / FC.NAME / f["file"]).exists() for f in frames), "frames")
        check((d / "out" / f"{FC.NAME}_events.webp").exists() and doc["in_season"], "the events still")
        check(len(doc["regions"]["wnp"]["share"]) == 4, "json leads")
        # outside the season: the note, a one-frame loop, in_season false, the old frames gone
        doc = FC.render_off_season(pd.Timestamp("2026-10-17 00:00"), d / "out")
        files = sorted(p.name for p in (d / "out" / "anim" / FC.NAME).glob("*.webp"))
        check(not doc["in_season"] and files == ["F00.webp"], f"off season {files}")


def test_hindcast():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        steps = [0, 12, 24]
        init = pd.Timestamp("2026-01-10 00:00")
        fake_run(d / "fc.npz", init, steps)
        fake_clim(d / "clim.nc")
        # a scan whose analyses are the run's member 0 at each step: a perfect member
        _, _, masks, _ = FC.load_run(d / "fc.npz")
        times = pd.DatetimeIndex([init + pd.Timedelta(hours=s) for s in steps])
        packed = np.stack([np.packbits(m[0].reshape(3, -1), axis=-1) for m in masks])
        events = HC.scan_events(times, packed)
        np.savez_compressed(
            d / "scan.npz", time=np.array(times, "datetime64[s]"), masks=packed, events=json.dumps(events)
        )
        doc = HC.compare(d / "fc.npz", d / "scan.npz", d / "clim.nc", d / "out")
        check(len(doc["leads"]) == 3 and (d / "out" / "compare_20260110.webp").exists(), "compare outputs")
        check(all(row["polar"]["bss"] > 0.5 for row in doc["leads"]), "a near-perfect ensemble beats a flat normal")
        ev = {r: [{"onset": "2026-01-12 00:00", "end": "2026-01-12 12:00", "analyses": 2, "peak_columns": 40}]
              for r in JB.REGIONS}  # fmt: skip
        np.savez_compressed(d / "scan2.npz", time=np.array(times, "datetime64[s]"), masks=packed, events=json.dumps(ev))
        check(HC.pick(d / "scan2.npz", [3, 6]) == ["20260106", "20260109"], "pick")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
