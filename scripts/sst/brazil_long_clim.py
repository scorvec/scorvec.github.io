#!/usr/bin/env python3
"""Long IMERG climatology for the Brazil/Colombia crop from monthly IMERG.

GPM_3IMERGM V07 monthlies, 2001-2024 (24 complete years): each granule
is downloaded (earthaccess auth), the shared crop subset read, the
granule deleted. Accumulates per-calendar-month mean rain rate
(mm/day) per cell, then fits the 5-coefficient annual+semiannual
harmonic to the 12 monthly climatology values per cell.

The gauge-correction field (2-yr, INMET) is applied at EVALUATION
time, not baked in — corrected clim = F x harmonic(clim24yr).

Output: ~/brazil_hydro/raw/imerg_longclim.npz
  coef (5, ny, nx) raw harmonic coefficients, mm/day
  monthly (12, ny, nx) raw monthly means, mm/day
  progress sidecar allows resume.

    python scripts/sst/brazil_long_clim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                   # noqa: E402

OUT = Path.home() / "brazil_hydro" / "raw" / "imerg_longclim.npz"
PROG = Path.home() / "brazil_hydro" / "raw" / "imerg_longclim_progress.json"
SCRATCH = Path.home() / "brazil_hydro" / "raw" / "granule_tmp"
Y0, Y1 = 2001, 2024


def main() -> int:
    import earthaccess
    IP._login()
    ml, mt = IP._grid_axes()
    ny, nx = int(mt.sum()), int(ml.sum())
    prog = json.loads(PROG.read_text()) if PROG.exists() else {}
    ssum = np.array(prog.get("sum", np.zeros((12, ny, nx)).tolist()),
                    dtype="float64") if "sum" in prog else np.zeros((12, ny, nx))
    cnt = np.array(prog.get("cnt", [0] * 12), dtype=int)
    done = set(prog.get("done", []))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for y in range(Y0, Y1 + 1):
        for m in range(1, 13):
            key = f"{y}{m:02d}"
            if key in done:
                continue
            res = earthaccess.search_data(
                short_name="GPM_3IMERGM", version="07",
                temporal=(f"{y}-{m:02d}-01", f"{y}-{m:02d}-27"))
            if not res:
                print(f"  {key}: no granule", flush=True)
                done.add(key)
                continue
            try:
                files = earthaccess.download(res[:1], str(SCRATCH))
                p = Path(files[0])
                with h5py.File(p, "r") as f:
                    arr = f["Grid/precipitation"][0].astype("float32")  # (lon, lat)
                sub = np.where(arr < 0, np.nan,
                               arr)[np.ix_(ml, mt)].T * 24.0     # mm/hr -> mm/day
                p.unlink(missing_ok=True)
            except Exception as e:                  # noqa: BLE001
                print(f"  {key}: {repr(e)[:80]}", flush=True)
                continue
            ssum[m - 1] += np.nan_to_num(sub)
            cnt[m - 1] += 1
            done.add(key)
            if len(done) % 24 == 0:
                PROG.write_text(json.dumps(
                    {"done": sorted(done), "cnt": cnt.tolist(),
                     "sum": np.round(ssum, 4).tolist()}))
                print(f"  progress: {len(done)} months", flush=True)
    monthly = ssum / np.maximum(cnt[:, None, None], 1)
    # harmonic fit on 12 mid-month doys per cell
    mid = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    th = 2 * np.pi * mid / 365.0
    X = np.column_stack([np.ones_like(th), np.sin(th), np.cos(th),
                         np.sin(2 * th), np.cos(2 * th)])
    beta, *_ = np.linalg.lstsq(X, monthly.reshape(12, -1), rcond=None)
    coef = beta.reshape(5, ny, nx).astype("float32")
    np.savez_compressed(OUT, coef=coef, monthly=monthly.astype("float32"),
                        years=f"{Y0}-{Y1}", n_months=cnt.tolist())
    PROG.write_text(json.dumps({"done": sorted(done), "cnt": cnt.tolist(),
                                "sum": np.round(ssum, 4).tolist()}))
    print(f"wrote {OUT} (per-month n = {cnt.tolist()})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
