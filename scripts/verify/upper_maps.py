#!/usr/bin/env python3
"""The single runs' upper-air error maps and cross-sections of verification.html, from the fields database
(upper_fields.py): each model's newest WINDOW runs at days 1, 3, 5, 7 and 10 against each truth.

  era5   ERA5 at 00Z of the valid day (era5-upa_)
  own    the model's own analysis: its 00Z run of the valid day, step 0. AIFS starts from the IFS analysis, so its
         step 0 is IFS's

A run is in a truth's window when that truth covers all five valid days. For each model, lead, variable and level:
  mean error   forecast less truth over the window's runs at each gridpoint, shown only where its t test passes
               Benjamini-Hochberg at FDR_Q over the map (verify_figs.mean_test: the variance inflated for the lag-1
               autocorrelation pooled over the map). Wind: the speed's error
  RMSE         the root-mean-square error, untested. Wind: the vector error's
and the same of the zonal mean over latitude and pressure (the cross-sections), tested over each section.

Written to OUT (published to the frames store as assets/verification/upper/):
  <truth>_d<day>_<var><level>.png   the grids, 16-bit integers in the PNG's red and green (high and low byte,
                                    offset by 32768, PACK steps), a flag in blue (0 no value, 1 value, 2 a significant
                                    mean error), alpha 255: mean error left, RMSE right, a model to a row (MODELS)
  <truth>_xs_<var>.json             the cross-sections
  coast.png                         coastlines and borders over the maps (plate carree, -180 to 180)
and the manifest assets/verification/upper.json.

    upper_maps.py --list --have have.txt            # the database files the maps are made from
    upper_maps.py --derived fields --out assets/verification/upper --manifest assets/verification/upper.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import upper_fields as U
from verify_figs import FDR_Q, mean_test

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from latlon import G15

WINDOW = 30  # runs
MIN_RUNS = 20  # a model with fewer runs in the window is left off that truth's maps
DAYS = (1, 3, 5, 7, 10)
TRUTHS = ("era5", "own")
MODELS = tuple(U.MODELS)
MAP_VARS = ("z", "t", "u", "v", "wind", "q")
# the grids' 16-bit steps (z m, t K, winds m/s, q kg/kg) and the units the page shows
PACK = {"z": 0.1, "t": 0.01, "u": 0.01, "v": 0.01, "wind": 0.01, "q": 1e-6}
UNITS = {"z": "m", "t": "K", "u": "m/s", "v": "m/s", "wind": "m/s", "q": "g/kg"}
SHOW = {"q": 1000.0}  # factor to the page's unit
DECIMALS = {"z": 1, "t": 2, "u": 2, "v": 2, "wind": 2, "q": 3}  # the cross-sections', in the page's units


def levels_of(var: str) -> tuple[int, ...]:
    return U.levels_of("q" if var == "q" else "z")


def inits_of(have: set[str], model: str) -> list[str]:
    pre = f"upa-{U.MODELS[model]}_"
    return sorted(n[len(pre) : len(pre) + 10] for n in have if n.startswith(pre))


def valid(init: str, day: int) -> pd.Timestamp:
    return pd.to_datetime(init, format="%Y%m%d%H") + pd.Timedelta(days=day)


def windows(have: set[str]) -> dict[str, dict[str, list[str]]]:
    """Each truth's window of runs for each model: its newest WINDOW runs whose valid days the truth covers."""
    out = {t: {} for t in TRUTHS}
    for m in MODELS:
        runs = inits_of(have, m)
        held = set(runs)
        for truth in TRUTHS:
            if truth == "era5":
                ok = [r for r in runs if all(U.truth_file(valid(r, d)) in have for d in DAYS)]
            else:
                ok = [r for r in runs if all(f"{valid(r, d):%Y%m%d%H}" in held for d in DAYS)]
            out[truth][m] = ok[-WINDOW:]
    return out


def needed(have: set[str]) -> list[str]:
    """The database files the maps are made from: each window's runs and their truths."""
    names = set()
    for truth, per in windows(have).items():
        for m, runs in per.items():
            for r in runs:
                names.add(U.file_of(m, r))
                for d in DAYS:
                    names.add(U.truth_file(valid(r, d)) if truth == "era5" else U.file_of(m, f"{valid(r, d):%Y%m%d%H}"))
    return sorted(names)


class Fields:
    """The database files of one model (or ERA5), packed as stored, unpacked a field at a time."""

    def __init__(self, paths: dict[str, Path]):
        self.packed, self.off = {}, {}
        for key, p in paths.items():
            with np.load(p) as z:
                self.packed[key] = {v: z[v] for v in U.VARS}
                self.off[key] = U.offsets(z)

    def get(self, key: str, var: str, lead: int | None) -> np.ndarray:
        """(level, lat, lon) of ``var`` of file ``key``, at lead index ``lead`` (None: a truth file)."""
        a = self.packed[key][var]
        return U.unpack(a if lead is None else a[lead], var, self.off[key][var])


def errors(fc: Fields, truth: Fields, runs: list[str], day: int, own: bool) -> dict[str, np.ndarray]:
    """(runs, level, lat, lon) errors of each map variable at ``day``; wind: the speed's error, and "wind_vec" the
    vector error's square."""
    k = U.LEADS.index(24 * day)
    out = {}
    for var in U.VARS:
        f = np.stack([fc.get(r, var, k) for r in runs])
        t = np.stack([truth.get(key, var, lead) for key, lead in (tkey(r, day, own) for r in runs)])
        out[var] = f - t
        if var in ("u", "v"):
            out[var + "_f"], out[var + "_t"] = f, t
    out["wind"] = np.hypot(out.pop("u_f"), out.pop("v_f")) - np.hypot(out.pop("u_t"), out.pop("v_t"))
    out["wind_vec"] = out["u"] ** 2 + out["v"] ** 2
    return out


def tkey(run: str, day: int, own: bool) -> tuple[str, int | None]:
    """The truth's file key and lead index for a run's valid day."""
    v = valid(run, day)
    return (f"{v:%Y%m%d%H}", 0) if own else (f"{v:%Y%m%d}", None)


def stats(err: np.ndarray, sq: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean error, its significance and RMSE of one map or section (``err``: runs x cells; ``sq`` the squared error
    when it is not err squared: the vector wind's)."""
    from mverify.significance import fdr

    mean, p = mean_test(err.T, MIN_RUNS)
    sig = fdr(p, FDR_Q) & np.isfinite(mean)
    rmse = np.sqrt(np.nanmean(err**2 if sq is None else sq, axis=0))
    return mean, sig, rmse


def as_list(x: np.ndarray, decimals: int, shape: tuple) -> list:
    """Nested lists of ``x`` rounded, None where it is not finite (JSON has no NaN)."""
    r = np.round(x.astype(np.float64), decimals).reshape(shape).astype(object)
    r[~np.isfinite(x.reshape(shape))] = None
    return r.tolist()


def to_png(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None], step: float, path: Path) -> None:
    """A model to a row of [mean error | RMSE], 16-bit in red and green, the flag in blue (module docstring)."""
    from PIL import Image

    ny, nx = G15.shape
    img = np.zeros((ny * len(rows), 2 * nx, 4), np.uint8)
    img[..., 3] = 255
    for i, row in enumerate(rows):
        if row is None:
            continue
        mean, sig, rmse = row
        for j, (val, flag) in enumerate(((mean, np.where(sig, 2, 1)), (rmse, np.ones(rmse.shape, int)))):
            v, ok = val.reshape(ny, nx), np.isfinite(val.reshape(ny, nx))
            q = (np.clip(np.round(np.where(ok, v, 0) / step), -32767, 32767) + 32768).astype(np.uint16)
            tile = img[i * ny : (i + 1) * ny, j * nx : (j + 1) * nx]
            tile[..., 0], tile[..., 1] = q >> 8, q & 255
            tile[..., 2] = np.where(ok, flag.reshape(ny, nx), 0)
    Image.fromarray(img, "RGBA").save(path, optimize=True)


def overlay(path: Path) -> None:
    """Coastlines and borders on a transparent plate carree of the whole globe, -180 to 180, for the page's maps."""
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14.4, 7.2), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1), projection=ccrs.PlateCarree())
    ax.set_extent((-180, 180, -90, 90), ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.7, edgecolor="#2b2b2b")
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.35, edgecolor="#555555")
    ax.axis("off")
    fig.savefig(path, transparent=True, dpi=100)
    plt.close(fig)


def build(a) -> None:
    import warnings

    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)  # cells no run reaches stay NaN
    have = {p.name for p in a.derived.glob("*.npz")}
    win = windows(have)
    era5 = Fields({n[9:17]: a.derived / n for n in sorted(have) if n.startswith("era5-upa_")})
    a.out.mkdir(parents=True, exist_ok=True)
    maps: dict[tuple, list] = {}
    sections: dict[tuple, dict] = {}
    used = {t: {} for t in TRUTHS}
    for mi, m in enumerate(MODELS):
        pre = f"upa-{U.MODELS[m]}_"
        fc = Fields({n[len(pre) : len(pre) + 10]: a.derived / n for n in sorted(have) if n.startswith(pre)})
        for truth in TRUTHS:
            runs = win[truth][m]
            if len(runs) < MIN_RUNS:
                print(f"{m} against {truth}: {len(runs)} runs, left off", flush=True)
                continue
            used[truth][m] = {"runs": len(runs), "first": runs[0], "last": runs[-1]}
            for day in DAYS:
                e = errors(fc, fc if truth == "own" else era5, runs, day, truth == "own")
                for var in MAP_VARS:
                    sq = e["wind_vec"] if var == "wind" else None
                    for li, lev in enumerate(levels_of(var)):
                        key = (truth, day, var, lev)
                        maps.setdefault(key, [None] * len(MODELS))[mi] = stats(
                            e[var][:, li].reshape(len(runs), -1),
                            None if sq is None else sq[:, li].reshape(len(runs), -1),
                        )
                    zm = np.nanmean(e[var], axis=3)  # runs x level x lat
                    zsq = np.nanmean(sq if sq is not None else e[var] ** 2, axis=3)
                    mean, sig, _ = stats(zm.reshape(len(runs), -1), None)
                    rmse = np.sqrt(np.nanmean(zsq, axis=0)).ravel()
                    f, nd = SHOW.get(var, 1.0), DECIMALS[var]
                    sections.setdefault((truth, var), {}).setdefault(str(day), {})[m] = {
                        "mean": as_list(mean * f, nd, zm.shape[1:]),
                        "sig": sig.astype(int).reshape(zm.shape[1:]).tolist(),
                        "rmse": as_list(rmse * f, nd, zm.shape[1:]),
                    }
            print(f"{m} against {truth}: {len(runs)} runs, {runs[0]} to {runs[-1]}", flush=True)
        del fc
    shares = {}
    for (truth, day, var, lev), rows in maps.items():
        to_png(rows, PACK[var], a.out / f"{truth}_d{day}_{var}{lev}.png")
        shares.setdefault(truth, {}).setdefault(str(day), {})[f"{var}{lev}"] = [
            None if r is None else [int(np.isfinite(r[0]).sum()), int(r[1].sum())] for r in rows
        ]
    for (truth, var), body in sections.items():
        (a.out / f"{truth}_xs_{var}.json").write_text(
            json.dumps({"levels": levels_of(var), "lat": G15.lat.tolist(), "days": body}, separators=(",", ":"))
        )
    overlay(a.out / "coast.png")
    manifest = {
        "updated": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%MZ}",
        "models": list(MODELS),
        "days": list(DAYS),
        "truths": used,
        "vars": {
            v: {"levels": list(levels_of(v)), "unit": UNITS[v], "step": PACK[v] * SHOW.get(v, 1.0)} for v in MAP_VARS
        },
        "grid": {
            "lat0": float(G15.lat[0]),
            "dlat": float(G15.lat[1] - G15.lat[0]),
            "lon0": float(G15.lon[0]),
            "dlon": float(G15.lon[1] - G15.lon[0]),
            "ny": G15.shape[0],
            "nx": G15.shape[1],
        },
        "shares": shares,
        "fdr_q": FDR_Q,
    }
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(f"{len(maps)} maps, {len(sections)} sections", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the database files the maps are made from")
    ap.add_argument("--have", type=Path, help="names of the files the database holds, one per line (--list)")
    ap.add_argument("--derived", type=Path, default=Path("fields"), help="those files, downloaded")
    ap.add_argument("--out", type=Path, default=Path("assets/verification/upper"))
    ap.add_argument("--manifest", type=Path, default=Path("assets/verification/upper.json"))
    a = ap.parse_args()
    if a.list:
        print(*needed(set(a.have.read_text().split())), sep="\n")
        return
    build(a)


if __name__ == "__main__":
    main()
