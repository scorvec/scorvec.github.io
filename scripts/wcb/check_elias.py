#!/usr/bin/env python3
"""Hold elias.py to the published pipeline it follows (wcb-ref.yml, job check).

    python scripts/wcb/check_elias.py ARCHIVE     # the unzipped Zenodo archive: its code, models and example fields

The archive's own code is run on its example (ERA-Interim, 3-5 October 2016 00 UTC) under Keras, and for 4 October
  1  the example's layout is the one elias.py assumes (grid, the levels either side of 700, 500 and 300 hPa), and
     the thresholds and the climatology read the same;
  2  each network's image is the image the archive hands its model (the outflow's irrotational wind taken the
     archive's way, from the winds, so that only the predictors are compared);
  3  each network gives Keras' output from the same image, block by block and at the end;
  4  the three probabilities and the belts' footprints come out as the archive's, end to end;
  5  the irrotational wind elias.py derives from a divergence is the one windspharm derives from the winds at the
     same truncation; how far the outflow moves between truncations of 89 and 180 is printed for information.
Any difference beyond TOL fails the run. Needs the archive's environment beside this module's: tensorflow with
Keras 2, windspharm, xarray, matplotlib.
"""

from __future__ import annotations

import os
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elias as E

DAY = datetime(2016, 10, 4)
LAG = timedelta(hours=E.LAG_H)
TOL = 1.0e-4
ROWS = slice(83, 181)  # elias.ROWS and elias.LAT in the example's latitudes (90S-90N)
INNER = slice(84, 180)


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def gap(a, b):
    """Largest difference between two arrays, relative to the size of the second where that exceeds one."""
    return float(np.abs(np.asarray(a, float) - b).max() / max(1.0, np.abs(b).max()))


def archive_code(root):
    """The archive's module, importable under a current Keras 2 (three of its import paths have since moved)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import keras

    moved = {"convolutional": ("Conv2D", "Conv2DTranspose"), "pooling": ("MaxPooling2D",), "merge": ("concatenate",)}
    for name, members in moved.items():
        module = types.ModuleType(f"keras.layers.{name}")
        for member in members:
            setattr(module, member, getattr(keras.layers, member))
        sys.modules[module.__name__] = module
    sys.path.insert(0, str(root))
    import calc_wcbmetric_v2
    import dict_wcbmetric_v2

    dict_wcbmetric_v2.calcdict["basedir"] = str(root)
    return calc_wcbmetric_v2


def example(root, day):
    """The example's fields of `day` as the archive's notebook hands them on - {name: (level, lat, lon)}, levels
    from 1000 hPa up, humidity never negative - in double precision, with the levels (hPa), latitudes, longitudes."""
    with Dataset(root / "data" / f"Z{day:%Y%m%d_%H}") as nc:
        lev = np.asarray(nc.variables["plev"][:], float)
        order = np.argsort(-lev)
        f = {k: np.asarray(nc.variables[k.upper()][0], float)[order] for k in "zuvqt"}
        lat, lon = (np.asarray(nc.variables[k][:], float) for k in ("lat", "lon"))
    f["q"] = np.maximum(f["q"], 0.0)
    return f, lev[order] / (100.0 if lev.max() > 10000.0 else 1.0), lat, lon


def ported(f, lev):
    """elias.py's input from the example's fields."""
    at = {int(p): i for i, p in enumerate(lev)}
    return {k: {p: f[k][at[p]][ROWS] for p in levels} for k, levels in E.LEVELS.items()}


def wind_speed(f, lev, truncation=None):
    """The 300 hPa irrotational wind speed the archive's way, on elias.LAT x elias.LON: windspharm on the global
    winds, rows from the north."""
    from windspharm.standard import VectorWind

    k = list(lev).index(300.0)
    wind = VectorWind(f["u"][k][::-1], f["v"][k][::-1])
    return np.hypot(*wind.irrotationalcomponent(truncation=truncation))[::-1][INNER], wind


class Tap:
    """A Keras model that keeps the images it is handed."""

    def __init__(self, model):
        self.model, self.seen = model, []

    def predict_on_batch(self, x):
        self.seen.append(np.array(x, np.float32))
        return self.model.predict_on_batch(x)


def main(root):
    import keras

    cw = archive_code(root)
    days = (DAY - LAG, DAY, DAY + LAG)
    data = {d: example(root, d) for d in days}
    f, lev, lat, lon = data[DAY]

    print("1  layout", flush=True)
    print(f"      levels (hPa): {lev.tolist()}")
    check(np.array_equal(lat, np.arange(-90.0, 91.0)) and np.array_equal(lon, E.LON), "1 degree, 90S-90N, 180W-179E")
    check(np.array_equal(lat[ROWS], E.ROWS) and np.array_equal(lat[INNER], E.LAT), "the domain's rows")
    for level, (below, above) in E.NEIGH.items():
        i = list(lev).index(level)
        check((lev[i - 1], lev[i + 1]) == (below, above), f"{level} hPa lies between {below} and {above} hPa")
    box = (INNER.start, INNER.stop, 0, lon.size)
    for stage, name in E.STAGES.items():
        ocv, hit = E.thresholds(root, stage, DAY), E.climatology(root, stage, DAY)
        print(
            f"      {stage}: thresholds {np.nanmin(ocv):.3f}-{np.nanmax(ocv):.3f}, climatology "
            f"{np.nanmin(hit):.4f}-{np.nanmax(hit):.4f} (mean {np.nanmean(hit):.4f}), "
            f"{int(np.isnan(ocv).sum())} + {int(np.isnan(hit).sum())} cells without a value"
        )
        published = np.ma.filled(cw.read_ocv(name, f"{DAY:%m%d}"), np.nan)
        check(np.array_equal(ocv, published, equal_nan=True), f"{stage} thresholds")
        published = np.ma.filled(cw.read_wcb(name, f"{DAY:%Y%m%d_%H}"), np.nan)[box[0] : box[1], box[2] : box[3]]
        check(np.array_equal(hit, published, equal_nan=True), f"{stage} climatology")

    # the archive, as its notebook runs it
    models = {stage: Tap(cw.load_model(name)) for stage, name in E.STAGES.items()}
    theirs = {}
    for d in days:
        g = data[d][0]
        clim = cw.read_wcb("MIDTROP", f"{d:%Y%m%d_%H}")
        theirs[d] = cw.calc_MIDTROP(models["ascent"], g["z"], g["u"], g["v"], g["q"], g["t"], clim, lon, lat, lev, *box)
    args = (f["z"], f["u"], f["v"], f["q"], f["t"])
    theirs["inflow"] = cw.calc_GT800(models["inflow"], *args, theirs[DAY + LAG], lon, lat, lev, *box)
    theirs["outflow"] = cw.calc_LT400(models["outflow"], *args, theirs[DAY - LAG], lon, lat, lev, *box)
    theirs["ascent"] = theirs[DAY]

    print("2  predictors", flush=True)
    speed, wind = wind_speed(f, lev)
    for d, seen in zip(days, models["ascent"].seen, strict=True):
        image = E.ascent_image(ported(*data[d][:2]), E.climatology(root, "ascent", d))
        for c in range(5):
            check((g := gap(image[..., c], seen[0, ..., c])) < TOL, f"ascent {d:%d %b} predictor {c + 1}: {g:.1e}")
    images = {
        "inflow": E.inflow_image(ported(f, lev), theirs[DAY + LAG]),
        "outflow": E.outflow_image(ported(f, lev), speed, theirs[DAY - LAG]),
    }
    for stage, image in images.items():
        for c in range(5):
            g = gap(image[..., c], models[stage].seen[0][0, ..., c])
            check(g < TOL, f"{stage} predictor {c + 1}: {g:.1e}")

    print("3  networks", flush=True)
    nets = {stage: E.Net(root / "models" / f"cnn_{name}.h5") for stage, name in E.STAGES.items()}
    for stage, tap in models.items():
        x = tap.seen[len(tap.seen) // 2]  # the image of DAY (the ascent's model also saw the days either side)
        inner = [a for a in tap.model.layers if type(a).__name__ in ("Activation", "Conv2DTranspose")]
        ref = keras.Model(tap.model.input, [a.output for a in inner]).predict_on_batch(x)
        trace = []
        p = nets[stage](x, trace)
        check(len(trace) == len(ref) == 13, f"{stage}: 13 blocks and upsamplings")
        for a, mine, keras_out in zip(inner, trace, ref, strict=True):
            same = mine.shape == keras_out.shape
            g = gap(mine, keras_out) if same else np.inf
            check(same and g < TOL, f"{stage} {a.name} {keras_out.shape[1:]}: {g:.1e}")
        check((g := gap(p[0], theirs[stage])) < TOL, f"{stage} probability from the archive's image: {g:.1e}")

    print("4  end to end", flush=True)
    ascent = {
        d: nets["ascent"](E.ascent_image(ported(*data[d][:2]), E.climatology(root, "ascent", d))[None])[0] for d in days
    }
    ours = {
        "ascent": ascent[DAY],
        "inflow": nets["inflow"](E.inflow_image(ported(f, lev), ascent[DAY + LAG])[None])[0],
        "outflow": nets["outflow"](E.outflow_image(ported(f, lev), speed, ascent[DAY - LAG])[None])[0],
    }
    for stage in E.STAGES:
        ocv = E.thresholds(root, stage, DAY)
        a, b = ours[stage] > ocv, theirs[stage] > ocv
        g = gap(ours[stage], theirs[stage])
        check(g < TOL, f"{stage}: probability {g:.1e}; footprint {int(b.sum())} cells, {int((a != b).sum())} differ")

    print("5  irrotational wind", flush=True)
    at89 = wind_speed(f, lev, 89)[0]
    dh = np.roll(wind.divergence(truncation=89)[:-1], 180, axis=-1)  # without the south pole, from 0E
    g = float(np.abs(np.hypot(*E.irrotational_wind(dh, 89)) - at89).max() / at89.max())
    check(g < TOL, f"from the divergence, truncated at 89: {g:.1e} of the largest speed ({at89.max():.1f} m/s)")
    coarse = nets["outflow"](E.outflow_image(ported(f, lev), at89, ascent[DAY - LAG])[None])[0]
    ocv = E.thresholds(root, "outflow", DAY)
    print(
        f"      outflow with the wind truncated at 89 instead of 180: probability moves {gap(coarse, ours['outflow']):.3f} "
        f"at most, {int(((coarse > ocv) != (ours['outflow'] > ocv)).sum())} of {int((ours['outflow'] > ocv).sum())} "
        "footprint cells change"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
