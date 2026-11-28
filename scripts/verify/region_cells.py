"""The surface stations' errors run by run on a grid of cells, for verification.html's own-region view: the page sums
the cells inside a box the reader draws into that box's errors for each run, and tests them there as this module's
siblings test theirs (verify_figs.mean_test and mverify.significance.paired_test, written again in the page).

For each kind of model (the single runs from their station sums stn_<init>.parquet, the ensemble means from
ens-stn_<init>.parquet), the newest REGION_RUNS runs and each variable of the station maps' surface layers, a
station's pairs at each lead of the station maps are summed into the CELL x CELL degree cell it reports from. A cell's
numbers in a run are its pairs (one a station: a lead verifies once in a run) and each model's mean error and RMSE
over them, in hundredths of the unit (STEP), so that the page's sums over cells are the stations' own sums to the
rounding. The cells are filed TILE x TILE degrees to a file, region_<kind>_<variable>_<row>_<column>.json, so that a
box fetches only the files it reaches:

    {"cells": [cell, ...], "n": [...], "models": {model: {"me": [...], "rm": [...]}}}

the cell numbered row * NX + column from 90 S and 180 W, each array flat over lead, run and cell (cell fastest). The
manifest (region.json) gives the runs, leads and models of each kind, which model has which run at each lead, and
each variable's cells with the stations of its median run, for the page's map of where the stations are.

In a run every model of a kind is scored on the same pairs (mverify.verify.common_sample), so a cell's pairs are one
count for all its models. The single runs' wind is the speed's error and the vector's RMSE, the ensemble means' the
members' mean speed's, both (as the country series, country_series.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CELL = 2.5  # degrees
TILE = 20.0  # degrees: a file's cells
NX = round(360 / CELL)
PER = round(TILE / CELL)  # cells along a tile's side
STEP = 0.01  # the stored mean errors and RMSEs, in their unit
REGION_RUNS = 90  # the newest runs of each kind
LEADS = (24, 72, 120, 168, 240)  # the station sums' leads (mverify_scores.MAP_LEADS)
KINDS = {  # kind: its station sums' files, its models in the page's order
    "single": ("stn", ("GFS", "IFS", "AIFS", "GDPS")),
    "means": ("ens-stn", ("GEFS", "IFS-ENS", "AIFS-ENS", "GEPS")),
}
VARS = {  # variable: its unit, its name for the single runs and for the ensemble means (verify_figs.COUNTRY_VARS)
    "t2m": ("°C", "2 m temperature", "2 m temperature"),
    "td2m": ("°C", "2 m dew point", "2 m dew point"),
    "msl": ("hPa", "Sea-level pressure", "Sea-level pressure"),
    "wind10": ("m/s", "10 m wind", "10 m wind speed"),
}
SPEED = {"ws10": "wind10"}  # the ensemble means' wind, named as the single runs'


def run_cells(path: Path) -> pd.DataFrame:
    """One run's station sums summed into cells: model, variable, lead_h, cell, n, se, sse."""
    cols = ["model", "network", "variable", "lead_h", "lat", "lon", "n", "se", "sse"]
    s = pd.read_parquet(path, columns=cols, filters=[("network", "==", "GHCNH"), ("lead_h", "in", list(LEADS))])
    s = s.astype({"model": str, "variable": str}).replace({"variable": SPEED})
    s = s[s["variable"].isin(VARS) & np.isfinite(s["lat"]) & np.isfinite(s["lon"])]
    row = np.clip(np.floor((s["lat"].to_numpy(float) + 90) / CELL), 0, 180 / CELL - 1).astype(int)
    col = np.floor((s["lon"].to_numpy(float) + 180) / CELL).astype(int) % NX
    s = s.assign(cell=row * NX + col)
    return s.groupby(["model", "variable", "lead_h", "cell"], as_index=False)[["n", "se", "sse"]].sum()


def kind_cells(derived: Path, kind: str) -> tuple[list[str], pd.DataFrame]:
    """The newest REGION_RUNS runs of one kind and their cells, each row numbered by its run (``run``)."""
    prefix, _ = KINDS[kind]
    name = re.compile(rf"{re.escape(prefix)}_(\d{{10}})\.parquet")
    found = sorted((m.group(1), f) for f in derived.glob(f"{prefix}_*.parquet") if (m := name.fullmatch(f.name)))
    found = found[-REGION_RUNS:]
    if not found:
        return [], pd.DataFrame()
    parts = [run_cells(f).assign(run=k) for k, (_, f) in enumerate(found)]
    return [i for i, _ in found], pd.concat(parts, ignore_index=True)


def tile_of(cell: np.ndarray) -> np.ndarray:
    return (cell // NX) // PER * 1000 + (cell % NX) // PER


def write_var(t: pd.DataFrame, kind: str, var: str, models: list[str], runs: int, out: Path) -> dict:
    """One kind's variable: its tiles' files, and for the manifest its cells with the stations of their median run
    at day 1 and which model has which run at each lead."""
    import verify_figs as F

    cells = np.sort(t.loc[t["n"] > 0, "cell"].unique())
    at = {c: k for k, c in enumerate(cells)}
    shape = (len(LEADS), runs, len(cells))
    lead_ix = t["lead_h"].map({h: k for k, h in enumerate(LEADS)}).to_numpy()
    cell_ix = t["cell"].map(at).to_numpy()
    n = np.zeros(shape)
    me, rm, present = {}, {}, {}
    for m in models:
        one = (t["model"] == m).to_numpy() & np.isfinite(cell_ix.astype(float))
        if not one.any():
            continue
        idx = (lead_ix[one], t["run"].to_numpy()[one], cell_ix[one].astype(int))
        k, se, sse = (np.zeros(shape) for _ in range(3))
        np.add.at(k, idx, t["n"].to_numpy(float)[one])
        np.add.at(se, idx, t["se"].to_numpy(float)[one])
        np.add.at(sse, idx, t["sse"].to_numpy(float)[one])
        n = np.maximum(n, k)  # one count for every model of the run (module docstring)
        pos = np.where(k > 0, k, 1.0)
        me[m] = np.where(k > 0, np.rint(se / pos / STEP), 0).astype(int)
        rm[m] = np.where(k > 0, np.rint(np.sqrt(sse / pos) / STEP), 0).astype(int)
        present[m] = ["".join("1" if x else "0" for x in (k[li].sum(1) > 0)) for li in range(len(LEADS))]
    if not me:
        return {}
    ms = [m for m in models if m in me]
    tiles, tile = [], tile_of(cells)
    for tl in np.unique(tile):
        inside = np.flatnonzero(tile == tl)
        flat = {m: {"me": me[m][:, :, inside].ravel().tolist(), "rm": rm[m][:, :, inside].ravel().tolist()} for m in ms}
        body = {"cells": cells[inside].tolist(), "n": n[:, :, inside].astype(int).ravel().tolist(), "models": flat}
        F.dump(body, out / f"region_{kind}_{var}_{tl // 1000}_{tl % 1000}.json")
        tiles.append(f"{tl // 1000}_{tl % 1000}")
    day1 = n[0]
    stations = np.nanmedian(np.where(day1.sum(1, keepdims=True) > 0, day1, np.nan), axis=0)
    unit, single, means = VARS[var]
    return {
        "label": single if kind == "single" else means,
        "unit": unit,
        "tiles": tiles,
        "cells": [[int(c), int(np.rint(s))] for c, s in zip(cells, np.nan_to_num(stations), strict=True)],
        "present": present,
    }


def region_figs(derived: Path, out: Path) -> dict:
    """Each kind's cells (module docstring); the manifest region.json, named in figs.json."""
    import verify_figs as F

    manifest = {"cell": CELL, "tile": TILE, "nx": NX, "step": STEP, "leads": list(LEADS), "kinds": {}}
    for kind, (_, order) in KINDS.items():
        inits, t = kind_cells(derived, kind)
        if t.empty:
            continue
        models = [m for m in order if m in set(t["model"])]
        variables = {}
        for var in VARS:
            if found := write_var(t[t["variable"] == var], kind, var, models, len(inits), out):
                variables[var] = found
        if variables:
            manifest["kinds"][kind] = {"inits": inits, "models": models, "vars": variables}
            print(f"region {kind}: {len(inits)} runs, " + ", ".join(f"{v} {len(x['cells'])} cells in "
                  f"{len(x['tiles'])} files" for v, x in variables.items()), flush=True)  # fmt: skip
    if not manifest["kinds"]:
        return {}
    return {"file": F.dump(manifest, out / "region.json")}
