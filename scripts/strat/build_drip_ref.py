#!/usr/bin/env python3
"""Standardisation reference for the FORECAST dripping-paint plots (2026-09-27): the MERRA-2 polar-cap (65-90N) height
fit of build_strat_history.standardise, stored as coefficients so the Actions renderers and the laptop GEPS run put
forecasts and the observed tail on exactly the scale of the MERRA-2 history drips.

Per level: mean = 4 annual harmonics + a linear trend (per decade from 2000; no MLS step - that applies above 5 hPa
only and every level here is >= 10 hPa); sd = sqrt(3-harmonic fit to the squared residual, the top 0.1 % clipped),
floored at a third of its seasonal maximum. The fit uses MERRA-2 1980-01-01 -> 2026-06-26 with the same QC and gap
filling as build_strat_history.load_m2.

    python build_drip_ref.py            # -> reference/drip_std.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "reference" / "drip_std.json"
LEVELS = [1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 250.0, 200.0, 100.0, 50.0, 10.0]


def design(t: pd.DatetimeIndex, nh: int, trend: bool):
    x = 2 * np.pi * (t.dayofyear.values - 1) / 365.25
    cols = [np.ones(len(t))]
    for k in range(1, nh + 1):
        cols += [np.cos(k * x), np.sin(k * x)]
    if trend:
        cols.append((t.year.values + t.dayofyear.values / 365.25 - 2000) / 10)
    return np.column_stack(cols)


def fit(y: np.ndarray, t: pd.DatetimeIndex):
    """The same algebra as build_strat_history.standardise (step=False), returning the coefficients."""
    X = design(t, 4, True)
    ok = np.isfinite(y)
    b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    r = y - X @ b
    Xs = design(t, 3, False)
    r2 = np.minimum(r ** 2, np.nanpercentile(r ** 2, 99.9))
    bs = np.linalg.lstsq(Xs[ok], r2[ok], rcond=None)[0]
    sd = np.sqrt(np.clip(Xs @ bs, 1e-6 * np.nanvar(r), None))
    return b, bs, float(sd.max() / 3), float(1e-6 * np.nanvar(r))


class Ref:
    """ref.mean(days, level) and ref.sd(days, level) on the MERRA-2 scale (m)."""

    def __init__(self, path: Path = OUT):
        self.j = json.loads(Path(path).read_text())

    def mean(self, days, L):
        c = self.j["levels"][str(float(L))]
        return design(pd.DatetimeIndex(days), 4, True) @ np.array(c["b"])

    def sd(self, days, L):
        c = self.j["levels"][str(float(L))]
        v = design(pd.DatetimeIndex(days), 3, False) @ np.array(c["bs"])
        return np.maximum(np.sqrt(np.clip(v, c["var_floor"], None)), c["sd_floor"])

    def z(self, days, L, height):
        return (np.asarray(height) - self.mean(days, L)) / self.sd(days, L)


def main() -> int:
    import sys
    sys.path.insert(0, str(HERE))
    import build_strat_history as B
    d = B.load_m2()
    t = pd.DatetimeIndex(d.time.values)
    out = {"source": "MERRA-2 cap_z (65-90N cos-weighted, daily mean of 00/06/12/18Z), "
                     f"{t[0]:%Y-%m-%d}..{t[-1]:%Y-%m-%d}, build_strat_history QC and gap filling",
           "method": "mean: 4 annual harmonics + linear trend per decade from 2000; sd: sqrt of a 3-harmonic fit to "
                     "the squared residual (top 0.1 % clipped), floored at a third of its seasonal maximum",
           "levels": {}}
    for L in LEVELS:
        y = d.cap_z.sel(lev=L, method="nearest").values.astype(float)
        b, bs, floor, vfloor = fit(y, t)
        out["levels"][str(L)] = {"b": [float(v) for v in b], "bs": [float(v) for v in bs], "sd_floor": floor,
                                 "var_floor": vfloor}
        # check against the history reference's own standardisation
        z_hist = B.standardise(y, t)[1]
        R = Ref.__new__(Ref); R.j = out
        z_ref = R.z(t, L, y)
        err = np.nanmax(np.abs(z_hist - z_ref))
        print(f"  {L:6.0f} hPa: sd range {np.sqrt(np.clip(design(t, 3, False) @ bs, vfloor, None)).min():6.1f}-"
              f"{np.sqrt(np.clip(design(t, 3, False) @ bs, vfloor, None)).max():6.1f} m, floor {floor:6.1f} m; "
              f"max |z - history z| {err:.2e}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
