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


def load(path: Path, short: str):
    """(values[sample, lead, lat, lon], lat, lon, year_of_sample or None), on a 1 deg grid."""
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={"indexpath": "", "time_dims": ("forecastMonth", "time")})
    name = short if short in ds.data_vars else list(ds.data_vars)[0]
    da = ds[name]
    extra = [d for d in da.dims if d not in ("forecastMonth", "latitude", "longitude")]
    years = None
    if "time" in extra:
        years = np.asarray([int(str(t)[:4]) for t in ds["time"].values])
    da = da.expand_dims("sample") if not extra else da.stack(sample=extra)
    da = da.transpose("sample", "forecastMonth", "latitude", "longitude")
    vals = da.values.astype(np.float32)
    lat, lon = da.latitude.values, da.longitude.values
    if years is not None and "number" in extra:          # sample = number x time, in stack order
        n_time = len(ds["time"]); n_num = vals.shape[0] // n_time
        years = np.tile(years, n_num) if extra.index("time") > extra.index("number") else np.repeat(years, n_num)
    ds.close()
    # one convention for every system: latitude ascending, longitude 0..360 ascending
    lon = np.where(lon < 0, lon + 360, lon)
    o = np.argsort(lon); lon, vals = lon[o], vals[..., o]
    if lat[0] > lat[-1]:
        lat, vals = lat[::-1], vals[..., ::-1, :]
    return vals, lat, lon, years


def probs(fc: np.ndarray, hc: np.ndarray, idx: list[int], years: np.ndarray | None,
          target_year: int | None, min_spread: float = 0.0) -> dict:
    """Tercile probabilities for one lead group; `target_year` set = detrended.

    Cells whose two boundaries sit within `min_spread` are dropped: in a desert, or in a
    dry season, a third of the hindcast can be the same near-zero number, and "below the
    lower tercile" then means nothing. Masking them is honest; drawing them is not."""
    f = fc[:, idx].mean(axis=1)
    h = hc[:, idx].mean(axis=1)
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
                fc, lat, lon, _ = load(fp, short)
                hc, lat_h, lon_h, years = load(hp, short)
            except Exception as e:                                    # noqa: BLE001
                print(f"  {label} {var}: unreadable ({str(e)[:70]})", file=sys.stderr); continue
            if fc.shape[-2:] != hc.shape[-2:]:
                print(f"  {label} {var}: forecast and hindcast grids differ; skipped", file=sys.stderr); continue
            fc, hc = fc * fac, hc * fac
            for pid, wantidx in PERIODS.items():
                idx = [i for i in wantidx if i < fc.shape[1] and i < hc.shape[1]]
                if not idx:
                    continue
                target = y + (mo - 1 + idx[len(idx) // 2]) // 12 if var in DETREND else None
                pr = probs(fc, hc, idx, years, target, min_spread=MIN_SPREAD.get(var, 0.0))
                pname, lead = period_label(idx, y, mo), lead_note(idx)
                nh = hc.shape[0] // max(len(set(years.tolist())) if years is not None else 1, 1)
                sub = (f"{label} · {pname} ({lead}) · {MONTHS[mo - 1]} {y} issue · {fc.shape[0]} members counted against "
                       f"this system's own 1993–2016 hindcast ({hc.shape[0]} samples) at each grid point. "
                       "White: no category reaches 40 %; near-normal is not drawn."
                       + ("  Counted against the hindcast's linear trend extrapolated to the valid year." if var in DETREND else ""))
                draw(pr, lat, lon, var, f"{TITLES[var]}: most likely tercile — {label}",
                     sub, OUT / f"c3s_terc_{mid}_{var}_{pid}.webp")
                pool.setdefault((var, pid), []).append((pr, lat, lon))
            have.append(var)
            del fc, hc
        if have:
            models.append({"id": mid, "label": label, "fields": have})
            print(f"  {label}: {', '.join(have)}", flush=True)

    made = 0
    for (var, pid), items in sorted(pool.items()):
        if len(items) < MIN_SYSTEMS:
            print(f"  pooled {var} {pid}: only {len(items)} system(s); skipped", flush=True); continue
        lat, lon = items[0][1], items[0][2]
        keep = [p for p, la, lo in items if p["above"].shape == items[0][0]["above"].shape]
        pr = {k: np.nanmean(np.stack([p[k] for p in keep]), axis=0) for k in ("above", "below", "normal")}
        idx = PERIODS[pid]
        pname, lead = period_label(idx, y, mo), lead_note(idx)
        sub = (f"{len(keep)} C3S systems, each weighted equally · {pname} ({lead}) · {MONTHS[mo - 1]} {y} issue · "
               "every system counted against its own 1993–2016 hindcast first. "
               "White: no category reaches 40 %; near-normal is not drawn."
               + ("  Counted against the hindcast's linear trend extrapolated to the valid year." if var in DETREND else ""))
        draw(pr, lat, lon, var, f"{TITLES[var]}: most likely tercile — multi-model", sub,
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
