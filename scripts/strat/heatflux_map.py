#!/usr/bin/env python3
"""Maps of the 100 hPa eddy heat flux every 12 hours: where round the hemisphere the waves carry heat poleward.

[v'T'] (heatflux100.py) is a zonal mean. What it averages is v*T*, the product at each point of the departures of v
and T from their zonal means, and that product is far from even round a latitude circle: a pulse into the
stratosphere is usually one or two regions of strong poleward flux. Each member's v and T at a forecast hour are
averaged onto the polar-cap cells (polarcap.CLAT x CLON, 1.5 deg) and multiplied there, so a map carries the waves
longer than about 3 deg (the line chart keeps waves 1-72) and nothing is averaged in time. The ensemble mean of the
members' v*T* is coloured only where enough of them share its sign (polarcap.agreed, a sign test at 5 %).

heatflux100.py --map draws AIFS-ENS and heatflux100_ext.py --map IFS-ENS (both through ecmwf() below), GEFS and
GEPS (shared(): strat_ens.py's part "fluxmap").

Outputs  assets/sst/anim/heatflux_map_{aifs,ifs,gefs,geps}/F<hour>.webp, assets/sst/anim/heatflux_map_<model>_manifest.json
"""

from __future__ import annotations

from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)
import numpy as np
import pandas as pd

import polarcap as PC
import run_record as RR

STEPS = tuple(range(0, 361, 12))  # h
LEVELS = np.arange(-150, 151, 15)  # K m s-1


def flux(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """v*T* of v and T on the cells (..., CLAT, CLON): the product of their departures from the zonal mean."""
    return (v - v.mean(-1, keepdims=True)) * (t - t.mean(-1, keepdims=True))


def cells(lat, lon, v, t) -> np.ndarray:
    """flux() of v and T (..., lat, lon) on a regular global grid."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return flux(*(PC._cells(np.asarray(x), lat, lon) for x in (v, t)))


def ecmwf(cyc, model: str, local=()) -> np.ndarray:
    """flux() for the perturbed members of an ECMWF ensemble (`model`: its name in the open data) on STEPS:
    (member, step, CLAT, CLON). v and T come from the 90-18N rows of each message, averaged onto the cells as they
    arrive (store.ensure_rows); a step that a whole-field file of `local` (specs of this cycle) holds on disk is read
    from it, not fetched again."""
    import store as E

    v, t = (
        E.load_rows(
            E.ensure_rows(
                cyc, E.Spec(model, "pf", x, "pl", (100,), STEPS), PC.ROWS, reduce=PC._cells, tag="cells", local=local
            )
        )[0][:, :, 0]
        for x in ("v", "t")
    )
    return flux(v, t)


def ifs(date: str, time: str) -> np.ndarray:
    """ecmwf() for IFS-ENS: v on every step and T on the daily ones are in the files ifs_ens.py fetched."""
    import ifs_ens as IE

    return ecmwf(IE.E.Cycle(date, time), "ifs", local=(IE.spec_uv(), IE.spec_t100()))


def shared(model: str, date: str, time: str) -> np.ndarray:
    """(member, step, CLAT, CLON) on STEPS from the shared GEFS/GEPS fetch (strat_ens.py): the members complete on
    every step."""
    import strat_ens as SE

    d = SE.load(model, date, f"{int(time):02d}", ("fluxmap",))
    return d["flux_map"][SE.complete_members(d, "flux_map")]


def frame(model: str, init: pd.Timestamp, hour: int, x: np.ndarray, out: Path) -> None:
    """One polar panel: the ensemble mean of x (member, CLAT, CLON), the members' v*T* at init + hour, shaded where
    they agree on its sign."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = x.shape[0]
    valid = init + pd.Timedelta(hours=hour)
    mean, gate = PC.agreed(x)
    fig = plt.figure(figsize=(8.0, 8.5), dpi=110)
    PC._panel(fig, [0.085, 0.155, 0.83, 0.745], None, mean, gate, LEVELS, None,
              "K m s$^{-1}$: red poleward, blue equatorward", "")  # fmt: skip
    fig.text(0.03, 0.966, f"{RR.MODEL_LABEL[model]} · 100 hPa eddy heat flux v*T*: {valid:%a %-d %b %HZ} (+{hour} h)",
             fontsize=14, fontweight="bold")  # fmt: skip
    fig.text(
        0.03,
        0.934,
        f"Run {init:%Y-%m-%d %HZ} · mean of the {n} members' flux, shaded where at least {PC.agreeing(n)} agree on "
        "the sign (a sign test at 5 %)",
        fontsize=9.5,
        color="#333",
    )
    fig.text(
        0.03,
        0.008,
        "Grey: the members are split. v* and T* are the departures of the meridional wind and temperature from their "
        "zonal means,\nhere on 1.5° cells at this hour; the mean of v*T* round a latitude circle is [v′T′], and the "
        f"line chart is its average between\nthe 45° and 75°N circles drawn here. {PC.CREDIT[model]}.",
        fontsize=8,
        color="#6f6b64",
        linespacing=1.4,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def loop(model: str, init: pd.Timestamp, x: np.ndarray) -> None:
    """The frames and the manifest: x (member, step, CLAT, CLON) on STEPS, a map each 12 hours."""
    rid = f"heatflux_map_{model}"
    d = PC.ANIM / rid
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.webp"):
        old.unlink()
    frames = []
    for k, hour in enumerate(STEPS):
        valid = init + pd.Timedelta(hours=hour)
        frame(model, init, hour, x[:, k], d / f"F{hour:03d}.webp")
        frames.append(
            {"file": f"F{hour:03d}.webp", "date": f"{valid:%Y-%m-%d}", "label": f"+{hour} h · {valid:%a %-d %b %HZ}"}
        )
    PC.manifest(rid, init, f"100 hPa eddy heat flux, {RR.MODEL_LABEL[model]}", frames)
    print(f"  {len(frames)} heat-flux map frames -> {d}", flush=True)
