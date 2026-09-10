#!/usr/bin/env python3
"""Render the C3S multi-model seasonal anomaly maps, one model at a time plus the multi-model mean.

Reads what c3s_maps.py cached (the postprocessed ensemble-mean anomalies, already taken against
each centre's own 1993-2016 hindcast) and draws them in the shared outlook style (mapstyle.py), so
they sit beside the SEAS5 and SFS maps without looking like a different site.

Outputs assets/sst/c3s/c3s_{model}_{field}_{period}.webp and assets/sst/data/c3s_maps.json,
which names every model, field and period the page can offer.

    SST_SITE_ROOT=. python scripts/sst/c3s_maps_render.py --issue 202609
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapstyle
from c3s_maps import CACHE, FIELDS, path_for
from c3s_nino34 import MODELS

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
OUT = SITE / "assets" / "sst" / "c3s"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# field → (label, units, colour map, vmax, map kind)
SPEC = {
    "t2m":  ("2 m temperature", "°C", "RdBu_r", 3.0, "atm"),
    "tp":   ("Precipitation", "mm/day", "BrBG", 3.0, "atm"),
    "mslp": ("Mean sea-level pressure", "hPa", "RdBu_r", 6.0, "atm"),
    "sst":  ("Sea-surface temperature", "°C", "RdBu_r", 2.0, "sst"),
}
# lead groups: month 1 is the start month itself, so 2-4 and 2-7 are the useful seasons
PERIODS = {"m1": ("lead month 1", [0]), "s1": ("months 2–4", [1, 2, 3]), "s2": ("months 5–7", [4, 5])}


def read(key: str, centre: str, system: int, issue: str):
    """(lead, lat, lon) anomaly array in display units, or None."""
    import xarray as xr
    p = path_for(key, centre, system, issue)
    if not p.exists():
        return None
    try:
        ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
    except Exception as e:                                            # noqa: BLE001
        print(f"  {centre}/{system} {key}: unreadable ({str(e)[:70]})", file=sys.stderr)
        return None
    da = ds[list(ds.data_vars)[0]]
    lead_dim = next((d for d in da.dims if d not in ("latitude", "longitude")), None)
    if lead_dim is None:
        da = da.expand_dims("lead")
        lead_dim = "lead"
    da = da.transpose(lead_dim, "latitude", "longitude")
    v = da.values.astype("float64")
    if key == "mslp":
        v = v / 100.0                                                 # Pa -> hPa
    if key == "tp":
        v = v * 86400.0 * 1000.0                                      # m/s -> mm/day
    return v, da.latitude.values, da.longitude.values


def draw(v, lat, lon, key: str, title: str, sub: str, out: Path):
    label, units, cmap, vmax, kind = SPEC[key]
    fig, ax, H, pc = mapstyle.open_map(kind=kind)
    lv = mapstyle.symmetric_levels(vmax)
    if lon.max() > 180:                                               # 0-360 -> -180..180 for plotting
        order = np.argsort(((lon + 180) % 360) - 180)
        lon = ((lon + 180) % 360 - 180)[order]; v = v[:, order]
    # Close the antimeridian: without a cyclic column contourf leaves a white seam down the
    # dateline, which on a Pacific-centred map runs straight through the subject.
    from cartopy.util import add_cyclic_point
    v, lon = add_cyclic_point(v, coord=lon)
    m = ax.contourf(lon, lat, v, levels=lv, cmap=cmap, extend="both", transform=pc)
    mapstyle.features(ax)
    mapstyle.heading(fig, H, title, sub)
    mapstyle.colorbar(fig, H, m, f"{label} anomaly ({units})", levels=lv)
    out.parent.mkdir(parents=True, exist_ok=True)
    mapstyle.save(fig, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True)
    ap.add_argument("--fields", default="t2m,tp,mslp,sst")
    a = ap.parse_args()
    issue = a.issue; y, mo = int(issue[:4]), int(issue[4:6])
    keys = [k for k in a.fields.split(",") if k in SPEC]
    models, made = [], 0
    stack: dict[tuple[str, str], list] = {}

    for entry in MODELS:
        centre, system, label = entry[0], entry[1], entry[2]
        mid = f"{centre}_{system}"
        have = []
        for key in keys:
            r = read(key, centre, system, issue)
            if r is None:
                continue
            v, lat, lon = r
            for pid, (pname, idx) in PERIODS.items():
                idx = [i for i in idx if i < v.shape[0]]
                if not idx:
                    continue
                field = v[idx].mean(axis=0)
                sub = (f"{label} · {issue[:4]}-{issue[4:6]} issue · {pname} · ensemble-mean anomaly "
                       f"against the model's own 1993–2016 hindcast (C3S postprocessed)")
                out = OUT / f"c3s_{mid}_{key}_{pid}.webp"
                draw(field, lat, lon, key, f"{SPEC[key][0]} anomaly — {label}", sub, out)
                made += 1
                stack.setdefault((key, pid), []).append((field, lat, lon))
            have.append(key)
        if have:
            models.append({"id": mid, "label": label, "fields": have})
            print(f"  {label}: {', '.join(have)}", flush=True)

    # multi-model mean: every model that supplied the field, equally weighted
    for (key, pid), items in stack.items():
        if len(items) < 2:
            continue
        base_lat, base_lon = items[0][1], items[0][2]
        fields = [f for f, la, lo in items if f.shape == items[0][0].shape]
        field = np.nanmean(np.stack(fields), axis=0)
        pname = PERIODS[pid][0]
        sub = (f"Mean of {len(fields)} C3S systems, equally weighted · {issue[:4]}-{issue[4:6]} issue · {pname} "
               f"· each model's ensemble-mean anomaly against its own 1993–2016 hindcast")
        draw(field, base_lat, base_lon, key, f"{SPEC[key][0]} anomaly — multi-model mean", sub, OUT / f"c3s_mmm_{key}_{pid}.webp")
        made += 1
    if stack:
        models.insert(0, {"id": "mmm", "label": "Multi-model mean", "fields": sorted({k for k, _ in stack})})

    doc = {"issue": f"{issue[:4]}-{issue[4:6]}", "issue_label": f"{MONTHS[mo-1]} {y}",
           "source": "C3S seasonal-postprocessed-single-levels (ensemble-mean anomaly vs each model's 1993-2016 hindcast)",
           "models": models, "fields": {k: {"label": SPEC[k][0], "units": SPEC[k][1]} for k in keys},
           "periods": {p: PERIODS[p][0] for p in PERIODS}}
    (SITE / "assets" / "sst" / "data").mkdir(parents=True, exist_ok=True)
    (SITE / "assets" / "sst" / "data" / "c3s_maps.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {made} maps + c3s_maps.json ({len(models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
