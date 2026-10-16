#!/usr/bin/env python3
"""Hadley and Walker forecasts, member by member, from the AIFS-ENS and IFS-ENS 00Z runs.

Every member's zonal-mean v and equatorial divergent zonal wind (overturn.reduce_file, the same reductions as the
analysis) at days 0..15, corrected by its model's drift against ERA5 at that lead and season (overturn.drift, the
reference in release overturn-ref-v1), then put through exactly what the analysed series goes through
(hadley_timeseries.py): TRAILING 7-day means, so the window of an early lead still holds analysed days; the cell
intensity and poleward edges (build_hadley_clim.sample_metrics); and the Pacific Walker cell index (walker.pac_index).
The analysed days carry the analysis offset d(0), so analysis and forecast both stand on ERA5's footing.

Without a drift reference for a model the model is left out: a raw forecast against ERA5 would show its drift as an
anomaly.

    python src/overturn_fc.py --date 20261016          # -> assets/sst/data/overturn_fc.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import overturn as O
from build_hadley_clim import sample_metrics
from doycal import noleap_doy
from mmsf import HIST as VHIST, LEVELS, streamfunction as psi_merid
from walker import HIST as UHIST, pac_index, streamfunction as psi_zonal

LEADS = list(range(16))  # days
WIN = 7  # trailing window, days (hadley_timeseries.py)
KEYS = ("nh_psi", "nh_edge_pole", "sh_psi", "sh_edge_pole")
P_PA = np.asarray(LEVELS, float) * 100.0


def fetch(model: str, init: pd.Timestamp, work: Path):
    """vbar (member, lead, lev, 721) and ud (member, lead, lev, 173) for every member, one step file at a time."""
    import store as ecmwf

    cfg = O.MODELS[model]
    cyc = ecmwf.Cycle(f"{init:%Y%m%d}", "00")
    members = ([0] if cfg["cf"] else []) + list(range(1, 51))
    vb = np.full((len(members), len(LEADS), len(LEVELS), O.LAT.size), np.nan, np.float32)
    ud = np.full((len(members), len(LEADS), len(LEVELS), O.NLON_UD), np.nan, np.float32)
    for j, L in enumerate(LEADS):
        s = (24 * L,)
        types = (["cf"] if cfg["cf"] else []) + ["pf"]
        for typ in types:
            p = ecmwf.ensure(cyc, ecmwf.Spec(cfg["store"], typ, ("u", "v"), "pl", tuple(LEVELS), s))
            mem = [0] if typ == "cf" else list(range(1, 51))
            v, u = O.reduce_file(p, mem, s)
            rows = [members.index(m) for m in mem]
            vb[rows, j], ud[rows, j] = v[:, 0], u[:, 0]
            p.unlink(missing_ok=True)  # ~0.9 GB a step: never keep them
        print(f"  {cfg['label']} day {L}: {np.isfinite(vb[:, j, 0, 0]).sum()} members", flush=True)
    np.savez_compressed(work / f"{model}_{init:%Y%m%d}.npz", vbar=vb, ud=ud, members=np.array(members))
    return vb, ud


def analysed(init: pd.Timestamp, offset_v, offset_u):
    """Daily means of the analysis history (00 and 12 UTC) for the WIN-1 days before init, less the analysis offset:
    (WIN-1, lev, lat) and (WIN-1, lev, lon)."""
    days = pd.date_range(init - pd.Timedelta(days=WIN - 1), init - pd.Timedelta(days=1))
    out = []
    for path, off in ((VHIST, offset_v), (UHIST, offset_u)):
        h = xr.open_dataarray(path) if Path(path).exists() else None
        if h is None:
            return None
        t = pd.DatetimeIndex(h.time.values).normalize()
        rows = []
        for d in days:
            m = t == d
            if not m.any():
                return None  # a hole in the window: no blended week can be formed
            rows.append(h.values[m].mean(0) - off)
        out.append(np.stack(rows))
    return out


def series(vb, ud, hist_v, hist_u, lon):
    """Per member and lead: the trailing-WIN-day means of the corrected fields -> {key: (member, lead)}."""
    n_m = vb.shape[0]
    res = {k: np.full((n_m, len(LEADS)), np.nan) for k in (*KEYS, "pac")}
    lat = O.LAT
    for i in range(n_m):
        full_v = np.concatenate([hist_v, vb[i]])  # analysed days, then lead 0..15
        full_u = np.concatenate([hist_u, ud[i]])
        for j in range(len(LEADS)):
            a = j + 1  # window = the WIN days ending at lead j
            wv, wu = full_v[a : a + WIN].mean(0), full_u[a : a + WIN].mean(0)
            m = sample_metrics(psi_merid(wv, P_PA, lat), np.asarray(LEVELS, float), lat)
            for k in KEYS:
                res[k][i, j] = m[k]
            res["pac"][i, j] = pac_index(psi_zonal(wu, P_PA), np.asarray(LEVELS, float), lon)
    return res


def _json(a):
    """Rounded nested lists with None for NaN (a cell or edge the metric could not resolve)."""
    return [_json(x) for x in a] if np.ndim(a) else (None if not np.isfinite(a) else round(float(a), 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="00Z run, YYYYMMDD")
    ap.add_argument("--models", default="aifs,ifs")
    ap.add_argument("--work", default="scripts/mjo/data/overturn")
    ap.add_argument("--out", default="assets/sst/data/overturn_fc.json")
    a = ap.parse_args()
    init = pd.Timestamp(a.date)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ref = O.load_ref()
    doy = float(noleap_doy(pd.DatetimeIndex([init]))[0])
    d_a = O.drift(ref, "aifs", doy, "vbar"), O.drift(ref, "aifs", doy, "ud")
    if d_a[0] is None:
        print("no AIFS-ENS drift: the analysis offset is unknown, nothing is drawn", file=sys.stderr)
        return 1
    hist = analysed(init, d_a[0][0][0], d_a[1][0][0])  # the analysis is the AIFS control's step 0
    if hist is None:
        print("the analysis history has a hole in the last week", file=sys.stderr)
        return 1
    doc = {
        "init": f"{init:%Y-%m-%d}T00:00Z",
        "valid": [f"{init + pd.Timedelta(days=L):%Y-%m-%d}" for L in LEADS],
        "window_days": WIN,
        "footing": "ERA5: members less their model's drift, analysed days less the analysis offset",
        "models": {},
    }
    for model in a.models.split(","):
        dv, du = O.drift(ref, model, doy, "vbar"), O.drift(ref, model, doy, "ud")
        if dv is None or du is None:
            print(f"  {O.MODELS[model]['label']}: no drift reference, left out", flush=True)
            continue
        vb, ud = fetch(model, init, work)
        vb = vb - O.drift_at(dv[0], LEADS)[None]
        ud = ud - O.drift_at(du[0], LEADS)[None]
        res = series(vb, ud, *hist, xr.open_dataarray(UHIST).longitude.values)
        doc["models"][model] = {
            "label": O.MODELS[model]["label"],
            "members": int(vb.shape[0]),
            "drift_starts": dv[1],
            **{k: _json(v) for k, v in res.items()},
        }
        med = {k: np.nanmedian(v[:, -1]) for k, v in res.items()}
        print(
            f"  {O.MODELS[model]['label']} day 15 medians: " + " ".join(f"{k} {v:+.2f}" for k, v in med.items()),
            flush=True,
        )
    if not doc["models"]:
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
