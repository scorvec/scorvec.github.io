#!/usr/bin/env python3
"""Tercile probability maps for every C3S system, and for the systems pooled.

Each system's live members are counted against ITS OWN 1993-2016 hindcast distribution at
each grid point, so model bias and spread drift are removed before anything is compared;
that is the only way the eight systems can be shown on one page. Temperature is counted
against the hindcast's linear trend extrapolated to the valid year -- against a flat
1993-2016 base the warming trend alone puts most of the tropics in the upper tercile and
the map stops saying anything about the season (seen on the SEAS5 page, 2026-09-07).

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
import mapstyle as MS
from c3s_nino34 import MODELS
from c3s_terciles import VARS, path_for
from c3s_maps_render import MONTHS, PERIODS, lead_note, period_label
from seas5_build import TERC_BINS, TERC_PALETTES

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
OUT = SITE / "assets" / "sst" / "c3s"
MIN_SYSTEMS = 4                       # below this the pooled map is one or two models in a trench coat
TITLES = {"t2m": "2 m temperature", "tp": "Precipitation"}
PALETTES = {"t2m": ("warm", "cool"), "tp": ("wet", "dry")}
SIDES = {"t2m": ("Above normal most likely", "Below normal most likely"),
         "tp": ("Wetter than normal most likely", "Drier than normal most likely")}
DETREND = ("t2m",)                    # precipitation has no trend worth extrapolating
MIN_SPREAD = {"tp": 0.05}             # mm/day between the two boundaries, below which the terciles are noise
# The CDS serves each system on its own 1 deg grid: ECMWF pole-inclusive on whole degrees,
# UKMO cell-centred. Per-system maps are drawn on the native grid; the pooled map needs one.
REF_LAT = np.arange(-89.5, 90.0, 1.0)
REF_LON = np.arange(0.5, 360.0, 1.0)


def to_ref(a: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear onto the reference grid, wrapping in longitude."""
    if a.shape == (REF_LAT.size, REF_LON.size) and np.allclose(lat, REF_LAT) and np.allclose(lon, REF_LON):
        return a
    from scipy.interpolate import RegularGridInterpolator
    lon_e = np.concatenate([lon, lon[:1] + 360.0])
    a_e = np.concatenate([a, a[:, :1]], axis=1)
    f = RegularGridInterpolator((lat, lon_e), a_e, bounds_error=False, fill_value=None)
    la, lo = np.meshgrid(REF_LAT, REF_LON, indexing="ij")
    return f((la, lo)).astype(np.float32)


def period_means(path: Path, short: str, fac: float, periods: dict):
    """{period: values[sample, lat, lon]}, lat, lon, year_of_sample.

    Never materialises the whole (sample, lead, lat, lon) cube: NCEP's hindcast alone is
    ~3,000 samples x 6 leads x 180 x 360, and holding a forecast and a hindcast cube at once
    is what the first version of this script died of (exit 137). Each lead group is averaged
    straight out of the file instead, which is all the terciles ever need."""
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib", chunks={"forecastMonth": 1},
                         backend_kwargs={"indexpath": "", "time_dims": ("forecastMonth", "time")})
    name = short if short in ds.data_vars else list(ds.data_vars)[0]
    da = ds[name]
    dims = [d for d in ("number", "time") if d in da.dims]
    da = da.transpose(*dims, "forecastMonth", "latitude", "longitude")
    lat, lon = da.latitude.values, da.longitude.values
    years = None
    if "time" in dims:
        yr = np.asarray([int(str(t)[:4]) for t in ds["time"].values])
        # sample index = the flattened leading dims in THIS order, so the repeat follows it
        years = np.tile(yr, len(ds["number"])) if dims == ["number", "time"] else np.repeat(yr, len(ds["number"])) \
            if "number" in dims else yr

    def flat(a: np.ndarray) -> np.ndarray:
        return a.reshape((-1,) + a.shape[-2:]) if dims else a[None]

    # a lagged system (UKMO: 62 members x 31 start dates) does not fill its own hypercube, and
    # cfgrib pads the combinations that never existed with NaN. Left in, they would count as
    # members that are never above the tercile and quietly deflate every probability.
    real = np.isfinite(flat(da.isel(forecastMonth=0).values)).any(axis=(1, 2))
    out = {}
    for pid, want in periods.items():
        idx = [i for i in want if i < da.sizes["forecastMonth"]]
        if not idx:
            continue
        m = flat(da.isel(forecastMonth=idx).mean("forecastMonth").values.astype(np.float32))[real]
        if fac != 1.0:
            m *= fac
        out[pid] = m
    ds.close()
    if years is not None:
        years = years[real]
    # one convention for every system: latitude ascending, longitude 0..360 ascending
    lon = np.where(lon < 0, lon + 360, lon)
    o = np.argsort(lon)
    lon = lon[o]
    flip = lat[0] > lat[-1]
    for pid in out:
        out[pid] = out[pid][..., o]
        if flip:
            out[pid] = out[pid][..., ::-1, :]
    if flip:
        lat = lat[::-1]
    return out, lat, lon, years


def probs(f: np.ndarray, h: np.ndarray, years: np.ndarray | None,
          target_year: int | None, min_spread: float = 0.0) -> dict:
    """Tercile probabilities for one lead group, from the season means of the members (f)
    and of the hindcast (h); `target_year` set = counted against the hindcast trend.

    Cells whose two boundaries sit within `min_spread` are dropped: in a desert, or in a
    dry season, a third of the hindcast can be the same near-zero number, and "below the
    lower tercile" then means nothing. Masking them is honest; drawing them is not."""
    if target_year is not None and years is not None:
        x = (years - years.mean()).astype("float32")[:, None, None]
        b = np.nansum(x * (h - np.nanmean(h, axis=0, keepdims=True)), axis=0) / float((x[:, 0, 0] ** 2).sum())
        h = h - b[None] * x
        f = f - b[None] * (target_year - years.mean())
    lo, hi = np.nanpercentile(h, [100 / 3, 200 / 3], axis=0)
    below = (f < lo[None]).mean(0)
    above = (f > hi[None]).mean(0)
    dead = ~np.isfinite(lo) | ~np.isfinite(hi) | ((hi - lo) <= min_spread)
    below = np.where(dead, np.nan, below); above = np.where(dead, np.nan, above)
    return {"above": above, "below": below, "normal": 1.0 - above - below}


def draw(pr: dict, lat, lon, var: str, title: str, sub: str, out: Path) -> None:
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from cartopy.util import add_cyclic_point

    above_c, below_c = (TERC_PALETTES[k] for k in PALETTES[var])
    fig, ax, H, pc = MS.open_map(kind="atm")
    normal = pr["normal"]
    for arr, other, cols in ((pr["above"], pr["below"], above_c), (pr["below"], pr["above"], below_c)):
        show = np.where((arr >= TERC_BINS[0]) & (arr >= np.maximum(other, normal)), arr, np.nan)
        a, lo = add_cyclic_point(show, coord=lon)
        ax.pcolormesh(lo, lat, a, cmap=ListedColormap(cols), norm=BoundaryNorm(TERC_BINS, len(cols)),
                      transform=pc, shading="auto", zorder=1)
    MS.features(ax, land_only=True)
    top_lab, bot_lab = SIDES[var]
    def patches(cols):
        return [Patch(color=c, label=f"{int(TERC_BINS[k] * 100)}–{int(min(TERC_BINS[k + 1], 1) * 100)}%")
                for k, c in enumerate(cols)]
    l1 = fig.legend(handles=patches(above_c), loc="lower left", bbox_to_anchor=(0.04, 0.004), ncol=6,
                    frameon=False, title=top_lab, fontsize=8, title_fontsize=8.5)
    fig.add_artist(l1)
    fig.legend(handles=patches(below_c), loc="lower right", bbox_to_anchor=(0.96, 0.004), ncol=6,
               frameon=False, title=bot_lab, fontsize=8, title_fontsize=8.5)
    MS.heading(fig, H, title, sub, wrap=190)
    MS.save(fig, out, dpi=110)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--vars", default="t2m,tp")
    a = ap.parse_args()
    issue = a.issue; y, mo = int(issue[:4]), int(issue[4:6])
    want = [v for v in a.vars.split(",") if v in VARS]
    OUT.mkdir(parents=True, exist_ok=True)

    models, pool = [], {}
    for entry in MODELS:
        centre, system, label = entry[0], str(entry[1]), entry[2]
        mid = f"{centre}_{system}"
        have = []
        for var in want:
            short, fac = VARS[var][1], VARS[var][2]
            fp, hp = path_for("fc", centre, system, var, issue), path_for("hc", centre, system, var, issue)
            if not (fp.exists() and hp.exists()):
                continue
            try:
                fcm, lat, lon, _ = period_means(fp, short, fac, PERIODS)
                hcm, lat_h, lon_h, years = period_means(hp, short, fac, PERIODS)
            except Exception as e:                                    # noqa: BLE001
                print(f"  {label} {var}: unreadable ({str(e)[:70]})", file=sys.stderr); continue
            if not fcm or next(iter(fcm.values())).shape[-2:] != next(iter(hcm.values())).shape[-2:]:
                print(f"  {label} {var}: forecast and hindcast grids differ; skipped", file=sys.stderr); continue
            for pid in PERIODS:
                if pid not in fcm or pid not in hcm:
                    continue
                idx = [i for i in PERIODS[pid]]
                target = y + (mo - 1 + idx[len(idx) // 2]) // 12 if var in DETREND else None
                pr = probs(fcm[pid], hcm[pid], years, target, min_spread=MIN_SPREAD.get(var, 0.0))
                pname, lead = period_label(idx, y, mo), lead_note(idx)
                sub = (f"{label} · {pname} ({lead}) · {MONTHS[mo - 1]} {y} issue · {fcm[pid].shape[0]} members counted "
                       f"against this system's own 1993–2016 hindcast ({hcm[pid].shape[0]} samples) at each grid point. "
                       "White: no category reaches 40 %; near-normal is not drawn."
                       + ("  Counted against the hindcast's linear trend extrapolated to the valid year." if var in DETREND else ""))
                draw(pr, lat, lon, var, f"{TITLES[var]}: most likely tercile — {label}",
                     sub, OUT / f"c3s_terc_{mid}_{var}_{pid}.webp")
                pool.setdefault((var, pid), []).append(
                    {k: to_ref(pr[k], lat, lon) for k in ("above", "below", "normal")})
            del fcm, hcm
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
            print(f"  pooled {var} {pid}: only {len(items)} system(s); skipped", flush=True); continue
        keep = items
        pr = {k: np.nanmean(np.stack([p[k] for p in keep]), axis=0) for k in ("above", "below", "normal")}
        idx = PERIODS[pid]
        pname, lead = period_label(idx, y, mo), lead_note(idx)
        sub = (f"{len(keep)} C3S systems, each weighted equally · {pname} ({lead}) · {MONTHS[mo - 1]} {y} issue · "
               "every system counted against its own 1993–2016 hindcast first. "
               "White: no category reaches 40 %; near-normal is not drawn."
               + ("  Counted against the hindcast's linear trend extrapolated to the valid year." if var in DETREND else ""))
        draw(pr, REF_LAT, REF_LON, var, f"{TITLES[var]}: most likely tercile — multi-model", sub,
             OUT / f"c3s_terc_mmm_{var}_{pid}.webp")
        made += 1
    if made:
        models.insert(0, {"id": "mmm", "label": "Multi-model", "fields": sorted({v for v, _ in pool})})

    doc_path = SITE / "assets" / "sst" / "data" / "c3s_maps.json"
    doc = json.loads(doc_path.read_text()) if doc_path.exists() else {}
    doc["terciles"] = {"models": models, "fields": {v: TITLES[v] for v in want},
                       "detrended": list(DETREND),
                       "periods": {p: period_label(PERIODS[p], y, mo) for p in PERIODS}}
    doc_path.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote tercile maps for {len(models)} model(s) + the pooled map")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
