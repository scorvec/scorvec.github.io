#!/usr/bin/env python3
"""Tercile probability maps for every C3S system, and for the systems pooled.

Each system's live members are counted against its own 1993-2016 hindcast distribution at
each grid point, so model bias and spread drift are removed before anything is compared;
that is the only way the eight systems can be shown on one page. Temperature is counted
against the hindcast's linear trend extrapolated to the valid year -- against a flat
1993-2016 base the warming trend alone puts most of the tropics in the upper tercile and
the map stops saying anything about the season.

The multi-model map averages the per-system probabilities with equal weight per system,
which is what pooling their members would give if every system had the same ensemble size.

Inputs come from c3s_terciles.py (and, for ECMWF, the SEAS5 page's own global files).

    SST_SITE_ROOT=. python scripts/sst/c3s_terciles_render.py --issue 202609
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c3s_grib
import mapstyle as MS
from c3s_nino34 import MODELS
from c3s_terciles import VARS, issue_table
from c3s_maps_render import MONTHS, PERIODS, lead_note, on_grid, period_label
from seas5_build import TERC_BINS, TERC_PALETTES, TERC_TEST_NOTE, tercile_signal

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
OUT = SITE / "assets" / "sst" / "c3s"
MIN_SYSTEMS = 4  # below this the pooled map is effectively one or two models
TITLES = {"t2m": "2 m temperature", "tp": "Precipitation"}
PALETTES = {"t2m": ("warm", "cool"), "tp": ("wet", "dry")}
SIDES = {
    "t2m": ("Above normal most likely", "Below normal most likely"),
    "tp": ("Wetter than normal most likely", "Drier than normal most likely"),
}
DETREND = ("t2m",)  # precipitation has no trend worth extrapolating
MIN_SPREAD = {"tp": 0.05}  # mm/day between the two boundaries, below which the terciles are noise
# The CDS serves each system on its own 1 deg grid: ECMWF pole-inclusive on whole degrees,
# UKMO cell-centred. Per-system maps are drawn on the native grid; the pooled map needs one.
REF_LAT = np.arange(-89.5, 90.0, 1.0)
REF_LON = np.arange(0.5, 360.0, 1.0)


def to_ref(a: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear onto the reference grid, wrapping in longitude."""
    return on_grid(a, lat, lon, REF_LAT, REF_LON)


def period_means(path: Path, short: str, fac: float, periods: dict):
    """{period: values[sample, lat, lon]}, lat, lon, year_of_sample.

    A sample is one member of one start (number, start date and time), its period value the mean
    of the period's leads (`periods` holds lead indices, 0 = the issue month). Only members holding
    every lead of the file count, and a period only where the file has all of its leads. Read
    message by message (c3s_grib), one period sum per sample: averaged out of cfgrib's hypercube,
    a lagged system is mostly padding, and one NCEP period alone peaked near 25 GB. A sample's year
    is that of its nominal start, so a January member started in December counts with the
    hindcast year it forecasts.

    A file can hold several variables (the SEAS5 page's own file: t2m, tprate, sst); `short` picks
    one, and a file of a single other variable is read as that one."""
    keys = ("cfVarName", "number", "dataDate", "dataTime", "forecastMonth", "indexingDate")

    def read(name: str | None):
        sums, leads, year, names, grid = {}, {}, {}, set(), None
        for m, v, g in c3s_grib.messages(path, keys):
            names.add(m["cfVarName"])
            if m["cfVarName"] != name or not np.isfinite(v).any():
                continue
            grid, i = g, m["forecastMonth"] - 1
            k = (m["number"], m["dataDate"], m["dataTime"])
            leads.setdefault(k, set()).add(i)
            year[k] = (m["indexingDate"] or m["dataDate"]) // 10000
            for pid, want in periods.items():
                if i in want:
                    s = sums.setdefault(pid, {})
                    s[k] = s[k] + v.astype(np.float32) if k in s else v.astype(np.float32)
        return sums, leads, year, names, grid

    sums, leads, year, names, grid = read(short)
    if not leads and len(names) == 1:
        sums, leads, year, names, grid = read(next(iter(names)))
    if not leads:
        return {}, None, None, None
    have = set().union(*leads.values())
    samples = sorted(k for k, s in leads.items() if s == have)
    # one convention for every system: latitude ascending, longitude 0..360 ascending
    lat_o, lon_o = np.argsort(grid.lat), np.argsort(grid.lon)
    out = {}
    for pid, want in periods.items():
        if not set(want) <= have:
            continue
        m = np.stack([sums[pid].pop(k) for k in samples]) * np.float32(fac / len(want))
        out[pid] = m[:, lat_o][:, :, lon_o]
    sums.clear()
    return out, grid.lat[lat_o], grid.lon[lon_o], np.array([year[k] for k in samples])


def bounds(h: np.ndarray, years: np.ndarray | None, detrend: bool) -> dict:
    """The hindcast half of `probs`, kept as a table (c3s_ref hc_terc): per grid point the 1/3 and 2/3
    bounds of the hindcast season means and, with `detrend`, the per-point linear-trend slope b."""
    out = {}
    if detrend and years is not None:
        x = (years - years.mean()).astype("float32")[:, None, None]
        b = np.nansum(x * (h - np.nanmean(h, axis=0, keepdims=True)), axis=0) / float((x[:, 0, 0] ** 2).sum())
        h = h - b[None] * x
        out["b"] = b
        out["ymean"] = np.float64(years.mean())
    out["lo"], out["hi"] = np.nanpercentile(h, [100 / 3, 200 / 3], axis=0)
    return out


def probs_from_bounds(f: np.ndarray, tb: dict, target_year: int | None, min_spread: float = 0.0) -> dict:
    """The forecast half: members counted against stored bounds, moved by the trend extrapolated to
    `target_year` when the bounds are detrended."""
    if "b" in tb and target_year is not None:
        f = f - tb["b"][None] * (target_year - tb["ymean"])
    lo, hi = tb["lo"], tb["hi"]
    below = (f < lo[None]).mean(0)
    above = (f > hi[None]).mean(0)
    dead = ~np.isfinite(lo) | ~np.isfinite(hi) | ((hi - lo) <= min_spread)
    below = np.where(dead, np.nan, below)
    above = np.where(dead, np.nan, above)
    return {"above": above, "below": below, "normal": 1.0 - above - below}


def probs(
    f: np.ndarray, h: np.ndarray, years: np.ndarray | None, target_year: int | None, min_spread: float = 0.0
) -> dict:
    """Tercile probabilities for one lead group, from the season means of the members (f)
    and of the hindcast (h); `target_year` set = counted against the hindcast trend.

    Cells whose two boundaries sit within `min_spread` are dropped: in a desert, or in a
    dry season, a third of the hindcast can be the same near-zero number, and "below the
    lower tercile" then means nothing. Masking them is honest; drawing them is not.
    Split in two halves (bounds / probs_from_bounds) so the hindcast half can be stored as a table."""
    tb = bounds(h, years, target_year is not None)
    return probs_from_bounds(f, tb, target_year, min_spread)


def draw(pr: dict, lat, lon, var: str, title: str, sub: str, out: Path, n_eff) -> None:
    """The most likely tercile where it reaches the first colour bin and the split passes tercile_signal (n_eff
    members)."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from cartopy.util import add_cyclic_point

    above_c, below_c = (TERC_PALETTES[k] for k in PALETTES[var])
    fig, ax, H, pc = MS.open_map(kind="atm")
    normal = pr["normal"]
    sig = tercile_signal(pr, n_eff)
    for arr, other, cols in ((pr["above"], pr["below"], above_c), (pr["below"], pr["above"], below_c)):
        show = np.where(sig & (arr >= TERC_BINS[0]) & (arr >= np.maximum(other, normal)), arr, np.nan)
        a, lo = add_cyclic_point(show, coord=lon)
        ax.pcolormesh(
            lo,
            lat,
            a,
            cmap=ListedColormap(cols),
            norm=BoundaryNorm(TERC_BINS, len(cols)),
            transform=pc,
            shading="auto",
            zorder=1,
        )
    MS.features(ax, land_only=True)
    MS.tercile_legends(fig, above_c, below_c, *SIDES[var], TERC_BINS)
    MS.heading(fig, H, title, sub, wrap=190)
    MS.save(fig, out, dpi=110)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--vars", default="t2m,tp")
    a = ap.parse_args()
    issue = a.issue
    y, mo = int(issue[:4]), int(issue[4:6])
    want = [v for v in a.vars.split(",") if v in VARS]
    OUT.mkdir(parents=True, exist_ok=True)

    models, pool = [], {}
    for entry in MODELS:
        centre, system, label = entry[0], str(entry[1]), entry[2]
        mid = f"{centre}_{system}"
        have = []
        for var in want:
            # the issue table (c3s_ref is_terc): built from the raw files when they are on disk,
            # otherwise read from the release -- the raw files are deleted once reduced
            t = issue_table(centre, system, var, issue)
            if t is None:
                continue
            lat, lon, meta = t["lat"], t["lon"], t["meta"]
            for pid in PERIODS:
                if pid not in meta["periods"]:
                    continue
                idx = list(PERIODS[pid])
                pr = {k: t[f"{k}_{pid}"] for k in ("above", "below", "normal")}
                pname, lead = period_label(idx, y, mo), lead_note(idx)
                sub = (
                    f"{label} · {pname} ({lead}) · {MONTHS[mo - 1]} {y} issue · {meta['nmem'][pid]} members counted "
                    f"against this system's own 1993–2016 hindcast ({meta['nsamp'][pid]} samples) at each grid point. "
                    f"Coloured where a category reaches 40 % {TERC_TEST_NOTE}; near-normal is not drawn."
                    + (
                        "  Counted against the hindcast's linear trend extrapolated to the valid year."
                        if var in DETREND
                        else ""
                    )
                )
                draw(
                    pr,
                    lat,
                    lon,
                    var,
                    f"{TITLES[var]}: most likely tercile — {label}",
                    sub,
                    OUT / f"c3s_terc_{mid}_{var}_{pid}.webp",
                    meta["nmem"][pid],
                )
                pool.setdefault((var, pid), []).append(
                    {**{k: to_ref(pr[k], lat, lon) for k in ("above", "below", "normal")}, "n": meta["nmem"][pid]}
                )
            del t
            have.append(var)
        # every listed system must carry every field: the picker offers one field list for
        # all of them, so a system with half the set would 404 on the other half
        if have == want:
            models.append({"id": mid, "label": label, "fields": have})
            print(f"  {label}: {', '.join(have)}", flush=True)
        elif have:
            print(f"  {label}: only {', '.join(have)}; left off the picker", flush=True)

    made = 0
    for (var, pid), items in sorted(pool.items()):
        if len(items) < MIN_SYSTEMS:
            print(f"  pooled {var} {pid}: only {len(items)} system(s); skipped", flush=True)
            continue
        keep = items
        pr = {k: np.nanmean(np.stack([p[k] for p in keep]), axis=0) for k in ("above", "below", "normal")}
        idx = PERIODS[pid]
        pname, lead = period_label(idx, y, mo), lead_note(idx)
        sub = (
            f"{len(keep)} C3S systems, each weighted equally · {pname} ({lead}) · {MONTHS[mo - 1]} {y} issue · "
            "every system counted against its own 1993–2016 hindcast first. "
            f"Coloured where a category reaches 40 % {TERC_TEST_NOTE}; near-normal is not drawn."
            + (
                "  Counted against the hindcast's linear trend extrapolated to the valid year."
                if var in DETREND
                else ""
            )
        )
        draw(
            pr,
            REF_LAT,
            REF_LON,
            var,
            f"{TITLES[var]}: most likely tercile — multi-model",
            sub,
            OUT / f"c3s_terc_mmm_{var}_{pid}.webp",
            1.0 / sum(1.0 / (len(keep) ** 2 * p["n"]) for p in keep),  # the equal-weight mean's effective members
        )
        made += 1
    if made:
        models.insert(0, {"id": "mmm", "label": "Multi-model", "fields": sorted({v for v, _ in pool})})

    doc_path = SITE / "assets" / "sst" / "data" / "c3s_maps.json"
    doc = json.loads(doc_path.read_text()) if doc_path.exists() else {}
    doc["terciles"] = {
        "issue": f"{y}-{mo:02d}",
        "issue_label": f"{MONTHS[mo - 1]} {y}",
        "models": models,
        "fields": {v: TITLES[v] for v in want},
        "detrended": list(DETREND),
        "periods": {p: period_label(PERIODS[p], y, mo) for p in PERIODS},
    }
    doc_path.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote tercile maps for {len(models)} model(s) + the pooled map")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
