#!/usr/bin/env python3
"""Ocean and ENSO indices for every C3S system, from the anomaly fields already on disk.

The SEAS5 card behind these indices is member-level: it reads the full ensemble and can
draw a fan. The other seven systems publish their members only as tens of gigabytes, and
the postprocessed collection this page already caches carries the ENSEMBLE MEAN anomaly --
against each centre's own 1993-2016 hindcast, which is exactly the reference an index
wants. So every system gets its index CURVE here at no download cost, and only ECMWF keeps
a spread; the card says so rather than implying eight fans.

Boxes, the PDO projection and the calibration all come from seas5_build so there is one
definition of each index on the site.

    SST_SITE_ROOT=. python scripts/sst/c3s_indices.py --issue 202609
-> assets/sst/data/c3s_indices.json
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
from c3s_maps import path_for
from c3s_nino34 import MODELS
from seas5_build import BOXES, box_mean, _pdo_index

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
OUT = SITE / "assets" / "sst" / "data" / "c3s_indices.json"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# key -> (label, units, note). Kept to what an ensemble-mean anomaly field can honestly
# support: no Trans-Niño (its standardisation needs the hindcast spread) and a relative
# Niño-3.4 that is the plain tropical-mean subtraction rather than the RONI scaling.
INDICES = [
    ("nino12", "Niño-1+2", "°C", "0–10°S, 90–80°W — the coastal, east-based index"),
    ("nino3", "Niño-3", "°C", "5°N–5°S, 150–90°W — the eastern basin"),
    ("nino34", "Niño-3.4", "°C", "5°N–5°S, 170–120°W — the ONI region"),
    ("nino4", "Niño-4", "°C", "5°N–5°S, 160°E–150°W — the central-western basin"),
    ("rnino34", "Niño-3.4 minus the tropics", "°C", "Niño-3.4 less the 20°S–20°N mean: the part that is not basin-wide warming"),
    ("pdo", "PDO", "index", "North Pacific EOF projection on the NCEI scale, global mean removed"),
    ("amo", "AMO (relative)", "°C", "North Atlantic 0–60°N minus the 60°S–60°N mean"),
    ("iod", "Indian Ocean Dipole", "°C", "west (50–70°E) minus east (90–110°E) box"),
    ("atl3", "Atlantic Niño (ATL3)", "°C", "3°S–3°N, 20°W–0°"),
]


def read_ssta(centre: str, system: str, issue: str):
    """(lead, lat, lon) ensemble-mean SST anomaly, or None."""
    import xarray as xr
    p = path_for("sst", centre, system, issue)
    if not p.exists():
        return None, None, None
    try:
        ds = xr.open_dataset(p, engine="cfgrib",
                             backend_kwargs={"indexpath": "", "time_dims": ("forecastMonth", "time")})
    except Exception as e:                                            # noqa: BLE001
        print(f"  {centre}/{system}: unreadable ({str(e)[:60]})", file=sys.stderr)
        return None, None, None
    da = ds[list(ds.data_vars)[0]]
    if "forecastMonth" in da.dims:
        da = da.transpose("forecastMonth", "latitude", "longitude")
    v = da.values.astype("float32")
    lat, lon = da.latitude.values, da.longitude.values
    ds.close()
    # The CDS serves 0..360; every box on this site is written -180..180 (Niño-3.4 is
    # -170..-120), and box_mean's wrap branch cannot rescue a box whose BOTH edges are
    # negative. Convert once, here, rather than teaching every index about the grid.
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    o = np.argsort(lon)
    lon, v = lon[o], v[..., o]
    if lat[0] > lat[-1]:
        lat, v = lat[::-1], v[..., ::-1, :]
    return v, lat, lon


def indices_for(v, lat, lon) -> dict:
    """{key: [value per lead]} from one system's anomaly field."""
    a = v[None]                                    # [sample=1, lead, lat, lon] for box_mean
    def bx(name):
        return box_mean(a, lat, lon, BOXES[name])[0]
    out = {}
    for k in ("nino12", "nino3", "nino34", "nino4"):
        out[k] = bx(k)
    trop, glob, natl = bx("trop"), bx("glob"), bx("natl")
    out["rnino34"] = out["nino34"] - trop
    out["amo"] = natl - glob
    out["iod"] = bx("iod_w") - bx("iod_e")
    out["atl3"] = bx("atl3")
    try:
        out["pdo"] = _pdo_index(a, lat, lon, glob[None])[0]
    except Exception as e:                                            # noqa: BLE001
        print(f"    PDO projection failed ({str(e)[:60]})", file=sys.stderr)
    return {k: [None if not np.isfinite(x) else round(float(x), 3) for x in vals]
            for k, vals in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    a = ap.parse_args()
    issue = a.issue
    y, mo = int(issue[:4]), int(issue[4:6])

    values, models, nlead = {}, [], 0
    for entry in MODELS:
        centre, system, label = entry[0], str(entry[1]), entry[2]
        colour = entry[3] if len(entry) > 3 else None
        v, lat, lon = read_ssta(centre, system, issue)
        if v is None:
            continue
        vals = indices_for(v, lat, lon)
        mid = f"{centre}_{system}"
        values[mid] = vals
        models.append({"id": mid, "label": label, "color": colour})
        nlead = max(nlead, v.shape[0])
        print(f"  {label}: Niño-3.4 {vals['nino34']}", flush=True)

    if not values:
        raise SystemExit("no cached SST anomaly fields — run c3s_maps.py first")

    # Multi-model mean: equal weight per SYSTEM, the same convention as the maps, and
    # only where at least half the systems have the index at that lead.
    mm = {}
    for key, _, _, _ in INDICES:
        series = [values[m["id"]].get(key) for m in models if values[m["id"]].get(key)]
        if not series:
            continue
        row = []
        for i in range(nlead):
            xs = [s[i] for s in series if i < len(s) and s[i] is not None]
            row.append(round(float(np.mean(xs)), 3) if len(xs) >= max(2, len(series) // 2) else None)
        mm[key] = row
    values["mmm"] = mm
    models.insert(0, {"id": "mmm", "label": "Multi-model mean", "color": "#111111"})

    months = []
    for i in range(nlead):
        m = mo + i
        months.append(f"{MONTHS[(m - 1) % 12]} {y + (m - 1) // 12}")

    doc = {"issue": f"{y}-{mo:02d}", "issue_label": f"{MONTHS[mo - 1]} {y}",
           "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "months": months, "models": models,
           "indices": [{"key": k, "label": lb, "units": u, "note": n} for k, lb, u, n in INDICES],
           "values": values,
           "basis": ("ensemble-mean anomaly against each system's own 1993–2016 hindcast "
                     "(C3S seasonal-postprocessed-single-levels); no member spread on this basis")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT.name}: {len(models)} model(s) incl. the mean, {len(INDICES)} indices, {nlead} leads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
