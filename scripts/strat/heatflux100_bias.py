#!/usr/bin/env python3
"""The 100 hPa eddy heat flux forecasts against ERA5: each model's systematic error by lead, from its own past runs.

heatflux100.py (AIFS-ENS) and heatflux100_ext.py (IFS-ENS, GEPS, GEFS) read their members against the ERA5 1991-2020
climatology (build_vt100_clim.py). A model whose flux runs systematically high or low against ERA5 would show that as
an anomaly, so each model's flux is computed the products' way on its archived 00Z runs and kept for comparison with
ERA5 (reference/vt100_era5.nc) at the same valid time:

  aifs  AIFS-ENS, Google Cloud mirror of ECMWF open data, from July 2025: control + 4 perturbed members, days 0-15
  ifs   IFS-ENS, the same mirror, from January 2025: 5 perturbed members (no control at 100 hPa), days 0-15, at 0.5 deg
        as in the product
  gefs  NOAA GEFS, NOAA's S3 bucket, from January 2025: control + 4 perturbed members, days 0-35
  geps  ECCC GEPS, MSC Datamart, which keeps only recent runs: control + 20 perturbed, days 0-16 (0-35 on the extended
        runs)

Starts every 3 days (GEPS: every day the Datamart still holds). The control and the mean of the perturbed members are
kept apart, so the products' own mix (AIFS-ENS 1 + 25, GEFS 1 + 30, GEPS 1 + 20, IFS-ENS 50) can be formed from them:
the expected flux of a perturbed member does not depend on which members are drawn.

    python heatflux100_bias.py archive --model aifs --part 0/8 --out aifs_0.npz
    python heatflux100_bias.py pack parts/*.npz      # -> reference/heatflux100_archive.npz, merged with the committed one
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "ecmwf"))
import build_vt100_clim as C
import heatflux100 as HF
import heatflux100_ext as HE

ARCHIVE = HERE / "reference" / "heatflux100_archive.npz"
EPOCH = pd.Timestamp("2025-01-01")  # lattice origin: starts are EPOCH + k * every days, 00Z
DAYS_ECMWF = list(range(16))  # the ECMWF products' steps, 0..360 h
DAYS_35 = list(range(36))  # GEFS and GEPS, 0..840 h
MODELS = {
    "aifs": {"label": "AIFS-ENS", "first": "2025-07-01", "every": 3, "days": DAYS_ECMWF, "pf": 4},
    "ifs": {"label": "IFS-ENS", "first": "2025-01-10", "every": 3, "days": DAYS_ECMWF, "pf": 5},
    "gefs": {"label": "GEFS", "first": "2025-01-10", "every": 3, "days": DAYS_35, "pf": 4},
    "geps": {"label": "GEPS", "first": None, "every": 1, "days": DAYS_35, "pf": 20},  # first: the Datamart's window
}
GEPS_WINDOW = 40  # days back the Datamart is asked for


def band(v, t, lat) -> np.ndarray:
    """(member, step, lat, lon) v and T -> (member, step, 6): the band fluxes in build_vt100_clim.KEYS order, the
    products' flux (heatflux100.fluxes and band_mean), member by member."""
    v, t = np.asarray(v), np.asarray(t)
    return np.stack([C.band_fluxes(v[i], t[i], lat) for i in range(len(v))])


def ecmwf_start(model: str, d: pd.Timestamp):
    """(control (step, 6) or None, perturbed (member, step, 6)) of one archived AIFS-ENS / IFS-ENS 00Z run. Checks the
    Google mirror's index first, so a run without 100 hPa v and T is skipped in seconds rather than retried."""
    import rangefetch as rf
    import store as E

    cfg = MODELS[model]
    name = "aifs-ens" if model == "aifs" else "ifs"
    date = f"{d:%Y%m%d}"
    try:
        idx = rf.fetch_index(date, "00", name, 360, "pf", sources=["google"])
    except rf.IndexUnavailable:
        return None
    if len({e["param"] for e in rf.select(idx, param=["v", "t"], levelist=[100], type="pf")}) < 2:
        return None
    cyc = E.Cycle(date, "00")
    steps = tuple(24 * L for L in cfg["days"])
    out = []
    for typ in ("cf", "pf") if model == "aifs" else ("pf",):  # IFS-ENS has no control at pressure levels
        spec = E.Spec(name, typ, ("v", "t"), "pl", (100,), steps, cfg["pf"] if typ == "pf" else 0)
        p = E.ensure(cyc, spec)
        v, t = HF.open_field(p, "v"), HF.open_field(p, "t")
        if model == "ifs":  # the product's 0.5 deg: every other point (where the longitudes start does not matter)
            out.append(band(v.values[:, :, ::2, ::2], t.values[:, :, ::2, ::2], v.latitude.values[::2]))
        else:
            out.append(band(v.values, t.values, v.latitude.values))
        p.unlink(missing_ok=True)
        p.with_suffix(p.suffix + ".json").unlink(missing_ok=True)
    return (out[0][0], out[1]) if model == "aifs" else (None, out[0])


def gefs_start(d: pd.Timestamp):
    mems = ["gec00"] + [f"gep{i:02d}" for i in range(1, MODELS["gefs"]["pf"] + 1)]
    v, t = HE.gefs_fields(f"{d:%Y%m%d}", mems, [24 * L for L in DAYS_35])
    if not np.isfinite(v[:, 0]).all():
        return None
    f = band(v, t, HE.LAT_H)
    return f[0], f[1:]


def geps_start(d: pd.Timestamp):
    """GEPS from the Datamart: to day 35 when the run is an extended one, else day 16; a missing lead stays NaN."""
    import strat_maps as sm

    date = f"{d:%Y%m%d}"
    last = 840 if sm.datamart.exists(sm.GEPS.format(d=date, c="00", v="TMP", lev=100, L=840)) else 384
    f = np.full((21, len(DAYS_35), 6), np.nan)
    for j, h in enumerate(range(0, last + 1, 24)):
        got = HE.geps_step(date, h)
        if got is None:
            if h == 0:
                return None
            continue
        f[:, j] = band(got[0][:, None], got[1][:, None], HE.LAT_H)[:, 0]
        for var in ("VGRD", "TMP"):  # a run's files are read once: keep the runner's disk flat
            (sm.DL / f"geps_{date}00_{var}100_{h:03d}.grib2").unlink(missing_ok=True)
    return f[0], f[1:]


def lattice(model: str, since, until, part=(0, 1)) -> pd.DatetimeIndex:
    cfg = MODELS[model]
    first = pd.Timestamp(cfg["first"]) if cfg["first"] else pd.Timestamp(until) - pd.Timedelta(days=GEPS_WINDOW)
    s = max(first, pd.Timestamp(since)) if since else first
    every = cfg["every"]
    d0 = EPOCH + pd.Timedelta(days=every * int(np.ceil((s - EPOCH).days / every)))
    k, n = part
    return pd.date_range(d0, pd.Timestamp(until), freq=f"{every}D")[k::n]


def archive(model: str, days) -> dict:
    run = {"aifs": lambda d: ecmwf_start("aifs", d), "ifs": lambda d: ecmwf_start("ifs", d)}
    run |= {"gefs": gefs_start, "geps": geps_start}
    nd = len(MODELS[model]["days"])
    keep, cf, pf, npf = [], [], [], []
    for d in days:
        try:
            got = run[model](d)
        except Exception as e:  # noqa: BLE001  (one bad run must not stop the archive)
            print(f"  {MODELS[model]['label']} {d:%Y-%m-%d}: {type(e).__name__} {str(e)[:90]}", flush=True)
            continue
        if got is None:
            print(f"  {MODELS[model]['label']} {d:%Y-%m-%d}: not available", flush=True)
            continue
        c, p = got
        cnt = np.isfinite(p).sum(0)  # members on hand at each lead (a GEPS or GEFS member-step can be missing)
        keep.append(d)
        cf.append(np.full((nd, 6), np.nan) if c is None else c)
        pf.append(np.where(cnt > 0, np.nansum(p, 0) / np.maximum(cnt, 1), np.nan))
        npf.append(int(cnt[0, 0]))
        print(f"  {MODELS[model]['label']} {d:%Y-%m-%d}: {int(cnt[0, 0])} perturbed members", flush=True)
    n = len(keep)
    return {
        f"{model}_start": np.array(keep, "datetime64[s]"),
        f"{model}_cf": np.array(cf, np.float32).reshape(n, nd, 6),
        f"{model}_pf": np.array(pf, np.float32).reshape(n, nd, 6),
        f"{model}_npf": np.array(npf, np.int16),
    }


def pack(paths, out: Path = ARCHIVE) -> dict:
    """Merge partial archives (any order, overlaps allowed: the later file wins) with the committed one."""
    acc: dict[str, list] = {}
    for p in ([out] if out.exists() else []) + [Path(x) for x in paths]:
        with np.load(p) as f:
            for k in f.files:
                acc.setdefault(k, []).append(f[k])
    doc = {}
    for m in MODELS:
        if f"{m}_start" not in acc:
            continue
        t = np.concatenate(acc[f"{m}_start"])
        last = len(t) - 1 - np.unique(t[::-1], return_index=True)[1]  # the last occurrence of each start, in order
        for k in ("start", "cf", "pf", "npf"):
            doc[f"{m}_{k}"] = np.concatenate(acc[f"{m}_{k}"])[last]
        print(
            f"  {MODELS[m]['label']}: {len(last)} runs, {pd.Timestamp(t.min()):%Y-%m-%d} .. {pd.Timestamp(t.max()):%Y-%m-%d}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **doc)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("archive")
    s.add_argument("--model", required=True, choices=sorted(MODELS))
    s.add_argument("--since", default="", help="first start (empty = the model's first archived run)")
    s.add_argument("--until", default=str((pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=6)).date()))
    s.add_argument("--part", default="0/1")
    s.add_argument("--out", required=True)
    sub.add_parser("pack").add_argument("parts", nargs="+")
    a = ap.parse_args()
    if a.cmd == "archive":
        k, n = (int(x) for x in a.part.split("/"))
        days = lattice(a.model, a.since, a.until, (k, n))
        print(f"{MODELS[a.model]['label']}: {len(days)} starts in part {k}/{n}", flush=True)
        np.savez_compressed(a.out, **archive(a.model, days))
    else:
        pack(a.parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
