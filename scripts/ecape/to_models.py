#!/usr/bin/env python3
"""ECAPE / CAPE grids in the models-viewer data format (2026-09-30, user: "put the ecape plots into the short-term
models interface as well").

The ECAPE job already computes full-resolution grids for the HRRR's 00/06/12/18Z runs - the same runs models.html
shows - so it also writes them as the viewer's lossless 8-bit WebP, nothing recomputed:

  assets/models/data/hrrr/<cycle>/{ecape_mu,ecape_ml,mucape,mlcape}/<fff>.webp

Encoding (enc "sqrt", K = 2.8): code = round(K * sqrt(J/kg)), 0-254, decoded value = (code / K)^2, so the top code is
8,229 J/kg and above; 255 = a column the kernel skipped. Quantisation error (half a step) is at most sqrt(x)/K:
+-3.6 J/kg at 100, +-11 at 1,000, +-20 at 3,000, +-28 at 6,000 (<= 3.6 % of the value above 100 J/kg). Measured
on HRRR 2026093018 f03: 1.50 MB per hour for the four fields; the 16-bit 1 J/kg alternative was 3.25 MB and 10 J/kg
16-bit about the same as this with a coarser low end. The ECAPE/CAPE ratio is not shipped: the page divides the two
grids (masked where CAPE <= 100 J/kg, as render_ecape.py did).

    python scripts/ecape/to_models.py encode <stem> --out assets/models/data/hrrr/2026093018
    python scripts/ecape/to_models.py entry --cycle 2026093018 --site . --out /tmp/ecape_entry.json [--sha SHA]
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

K = 2.8
FIELDS = ["ecape_mu", "ecape_ml", "mucape", "mlcape"]
METHOD = ("SHARPlib (Peters et al. 2023 entraining CAPE) on every column of the HRRR's native hybrid levels "
          "(wrfnat, lowest 42 levels), the same routine the sounding explorer runs")


def encode_cape(x):
    """J/kg -> uint8 codes, row 0 at the NORTH (the file's row 0 is the grid's southern row)."""
    x = np.asarray(x, dtype=np.float64)
    bad = ~np.isfinite(x) | (x < -1.0)                      # ecape_grid leaves -9999 in a skipped column
    c = np.clip(np.rint(K * np.sqrt(np.maximum(x, 0.0))), 0, 254)
    c = np.where(bad, 255, c).astype(np.uint8)
    return c[::-1]


def decode_cape(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c == 255, np.nan, (c / K) ** 2)


def webp(arr):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(b, "WEBP", lossless=True, quality=100, method=4)
    return b.getvalue()


def cmd_encode(a):
    stem = Path(a.stem)
    em = json.loads(Path(str(stem) + "_ecape.json").read_text())
    meta = json.loads(stem.with_suffix(".json").read_text())
    nf, ny, nx = em["shape"]
    arr = np.fromfile(str(stem) + "_ecape.f32", dtype=np.float32).reshape(nf, ny, nx)
    fxx = int(meta["cycle"].get("fxx", 0))
    out = Path(a.out)
    tot = 0
    for i, name in enumerate(em["fields"]):
        if name not in FIELDS:
            continue
        codes = encode_cape(arr[i])
        b = webp(codes)
        if a.check:                                           # lossless round trip, and the error bound
            from PIL import Image
            back = np.asarray(Image.open(io.BytesIO(b)))
            back = back if back.ndim == 2 else back[..., 0]
            assert np.array_equal(back, codes), f"{name}: WebP round trip is not exact"
            x = arr[i][::-1].astype(np.float64)
            ok = (x >= 0) & (x <= 8000)
            err = np.abs(decode_cape(codes)[ok] - x[ok])
            assert np.all(err <= np.sqrt(x[ok]) / K + 0.5 / K ** 2 + 1e-3), f"{name}: error bound broken"
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{fxx:03d}.webp").write_bytes(b)
        tot += len(b)
    print(f"  models format: f{fxx:03d} {tot / 1e6:.2f} MB -> {out}", flush=True)
    return 0


def cmd_entry(a):
    root = Path(a.site) / "assets/models/data/hrrr" / a.cycle
    fields, nbytes = {}, 0
    for name in FIELDS:
        d = root / name
        hrs = sorted(int(m.group(1)) for m in (re.match(r"^(\d{3})\.webp$", p.name) for p in d.glob("*.webp")) if m) if d.is_dir() else []
        if not hrs:
            continue
        nbytes += sum((d / f"{h:03d}.webp").stat().st_size for h in hrs)
        fields[name] = dict(hours=hrs, grid="main", enc="sqrt", k=K)
        if a.sha:
            fields[name]["sha"] = a.sha
    if not fields:
        print(f"no ECAPE frames under {root}", file=sys.stderr)
        return 1
    hours = sorted({h for f in fields.values() for h in f["hours"]})
    entry = dict(cycle=a.cycle, fields=fields, hours=hours, bytes=nbytes, method=METHOD,
                 made=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"))
    Path(a.out).write_text(json.dumps({"ecape": {a.cycle: entry}}, separators=(",", ":")))
    print(f"ECAPE entry {a.cycle}: {len(fields)} fields x {len(hours)} hours, {nbytes / 1e6:.1f} MB")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("encode")
    p.add_argument("stem")
    p.add_argument("--out", required=True)
    p.add_argument("--check", action="store_true", help="verify the lossless round trip and the error bound")
    p = sub.add_parser("entry")
    p.add_argument("--cycle", required=True)
    p.add_argument("--site", default=".")
    p.add_argument("--sha", default="")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"encode": cmd_encode, "entry": cmd_entry}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
