#!/usr/bin/env python3
"""What the RRFS bucket holds for one cycle (models-probe.yml, mode rrfsna): the file kinds and their sizes, the grid of
each kind, whether the North American 3 km files carry the messages models.py reads, and what one hour of the
North American grid costs to decode and to store against the CONUS grid.

    python rrfs_probe.py --cycle 2026111006 --lead 6
"""

import argparse
import re
import resource
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M

HOST = "https://noaa-rrfs-ops-pds.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
GRID_KEYS = (
    "gridType",
    "gridDefinitionTemplateNumber",
    "Ni",
    "Nj",
    "Nx",
    "Ny",
    "latitudeOfFirstGridPointInDegrees",
    "longitudeOfFirstGridPointInDegrees",
    "latitudeOfLastGridPointInDegrees",
    "longitudeOfLastGridPointInDegrees",
    "latitudeOfSouthernPoleInDegrees",
    "longitudeOfSouthernPoleInDegrees",
    "angleOfRotationInDegrees",
    "iDirectionIncrementInDegrees",
    "jDirectionIncrementInDegrees",
    "DxInMetres",
    "DyInMetres",
    "Latin1InDegrees",
    "Latin2InDegrees",
    "LoVInDegrees",
    "iScansNegatively",
    "jScansPositively",
    "resolutionAndComponentFlags",
    "packingType",
    "bitsPerValue",
    "bitmapPresent",
    "numberOfMissing",
    "shapeOfTheEarth",
)


def list_keys(prefix):
    keys, tok = [], None
    while True:
        u = f"{HOST}/?list-type=2&prefix={prefix}&max-keys=1000"
        if tok:
            u += "&continuation-token=" + urllib.parse.quote(tok, safe="")
        with urllib.request.urlopen(urllib.request.Request(u, headers=M.UA), timeout=60) as r:
            x = ET.fromstring(r.read())
        keys += [(c.find(NS + "Key").text, int(c.find(NS + "Size").text)) for c in x.iter(NS + "Contents")]
        tok = x.findtext(NS + "NextContinuationToken")
        if not tok:
            return keys


def grid_keys(buf):
    import eccodes

    g = eccodes.codes_new_from_message(buf)
    try:
        out = {}
        for k in GRID_KEYS:
            try:
                out[k] = eccodes.codes_get(g, k)
            except Exception:  # noqa: BLE001
                continue
        return out
    finally:
        eccodes.codes_release(g)


def look(f, base, name, want):
    """Which of the wanted messages one file carries, and their bytes."""
    lines = M.idx_lines(f, f"{base}.{name}")
    print(f"\n== {name}: {len(lines)} messages")
    tot, miss = 0, []
    for key, rx in want.items():
        hit = next(((s, e, d) for s, e, d in lines if re.search(rx, d)), None)
        if hit is None:
            miss.append(key)
            continue
        n = hit[1] - hit[0] + 1 if hit[1] >= 0 else 0
        tot += n
        print(f"  {key:14s} {n / 1e6:7.2f} MB  {hit[2]}")
    print(f"  wanted messages {tot / 1e6:.1f} MB/hour; MISSING {miss}")
    for key in miss:
        v = want[key].lstrip("^").split(":")[0]
        print(f"   lines with {v}:", [d for _, _, d in lines if d.startswith(v + ":")][:12])


def cost(base, name):
    """Fetch, decode and lossless-WebP cost of a few surface fields at every 1st, 2nd and 3rd grid point."""
    want = {k: M.IDX_WANT["rrfs"][k] for k in ("zsfc", "t2m", "refd", "u10", "v10", "ir")}
    t0 = time.time()
    blobs = M.fetch_idx_fields("rrfs", f"{base}.{name}", want)
    t1 = time.time()
    z, g = M.sb.decode(blobs["zsfc"], want_grid=True)
    t2 = time.time()
    a = {k: M.sb.decode(b) for k, b in blobs.items()}
    t3 = time.time()
    lat, lon = g["lat"], g["lon"]
    mb = sum(map(len, blobs.values())) / 1e6
    print(
        f"\n{name}: grid {z.shape}, fetch {t1 - t0:.1f}s ({mb:.1f} MB), lat/lon {t2 - t1:.1f}s, 6 decodes {t3 - t2:.1f}s"
    )
    print(f"  grid-relative winds {g['grid_rel']}, lat {float(lat.min()):.3f} to {float(lat.max()):.3f}")
    print("  corners", [(round(float(lat[j, i]), 3), round(float(lon[j, i]), 3)) for j in (0, -1) for i in (0, -1)])
    for k, v in a.items():
        print(f"  {k}: NaN {np.isnan(v).mean():.4f} min {np.nanmin(v):.2f} max {np.nanmax(v):.2f}")
    enc = {
        "t2m": lambda s: M.encode("t2m", M.f_of_k(a["t2m"][::s, ::s]))[0],
        "refd": lambda s: np.where(
            a["refd"][::s, ::s] >= 5, np.clip(np.rint(a["refd"][::s, ::s] * 2), 1, 254), 0
        ).astype(np.uint8),
        "ir": lambda s: M.q_lin(a["ir"][::s, ::s] - 273.15, -100.0, 0.6),
        "wind10": lambda s: np.dstack([M.q_lin(a[c][::s, ::s] * 1.94384, -127.0, 1.0) for c in ("u10", "v10", "u10")]),
    }
    for k, fn in enc.items():
        row = []
        for s in (1, 2, 3):
            t0 = time.time()
            b = M.webp(fn(s))
            row.append(f"x{s} {len(b) / 1024:.0f} KB {time.time() - t0:.1f}s")
        print(f"  webp {k}: " + "; ".join(row))
    print(f"  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--lead", type=int, default=6)
    a = ap.parse_args()
    date, cyc, tag = a.cycle[:8], int(a.cycle[8:]), f".f{a.lead:03d}."
    keys = list_keys(f"rrfs.{date}/{cyc:02d}/")
    kinds = defaultdict(list)
    for k, n in keys:
        kinds[re.sub(r"\.f\d{3}\.", ".fFFF.", k.split("/", 2)[2])].append(n)
    print(f"{len(keys)} keys, {sum(n for _, n in keys) / 1e9:.1f} GB under rrfs.{date}/{cyc:02d}/")
    for k in sorted(kinds):
        v = kinds[k]
        print(f"  {k:64s} n={len(v):3d} median {np.median(v) / 1e6:9.1f} MB  sum {sum(v) / 1e9:7.2f} GB")

    f, base = M.fetcher("rrfs"), M.sb.rrfs_base(date, cyc)
    have = {k.split("/", 2)[2].split("z.", 1)[-1] for k, _ in keys if k.count("/") == 2}
    upper = {**M.ua_want(), **M.sb_want_lcc("rrfs"), "sb:REFC": r"^REFC:entire atmosphere"}
    gribs = sorted(n for n in have if tag in n and n.endswith(".grib2"))
    for name in gribs:
        if ".3km." in name and name.split(".")[0] in ("2dfld", "prslev", "natlev"):
            try:
                look(f, base, name, M.IDX_WANT["rrfs"] if name.startswith("2dfld") else upper)
            except Exception as ex:  # noqa: BLE001
                print(name, "idx failed:", repr(ex)[:200])
    print()
    for name in gribs:
        try:
            s, e, d = M.idx_lines(f, f"{base}.{name}")[0]
            print(name, "|", d, "|", grid_keys(f.get(f"{base}.{name}", (s, e))))
        except Exception as ex:  # noqa: BLE001
            print(name, "idx/grid failed:", repr(ex)[:200])
    for dom in ("conus", "na"):
        name = f"2dfld.3km{tag}{dom}.grib2"
        if name in have:
            cost(base, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
