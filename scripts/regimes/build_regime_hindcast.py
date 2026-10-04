#!/usr/bin/env python3
"""Regime labels of every hindcast start, SEASON-AWARE (laptop, once per model; rerun if the regimes change).

  --model geps   GEPS8 hindcast 2001-2020 (4 starts a week, 4-member ensemble means, from IRI; zg500 monthly files in
                 ~/data_archive/geps_subx/geps8_hindcast). Anomalies exactly as the live GEPS product and the old
                 telecon_hindcast.py: minus the lead-dependent GEPS8 climatology at the start's day of year, plus the
                 ERA5 1991-2020 minus 2001-2020 shift at the valid date, on the 2.5 deg grid.
  --model gefs   GEFSv12 reforecast 2000-2019 (Wednesday starts, 4 members; the per-member daily 2.5 deg 500 hPa
                 height of release gefs-clim-v2, gefs_hind2_MM.npz). Anomalies exactly as the live GEFS product:
                 member mean minus the lead-matched reforecast climatology at the start's day of year (recomputed from
                 the same starts, +-15 d pooling, linear between 5-day centres - build_gefs_patterns_hindcast.py's
                 arithmetic), minus the ERA5 shift at the valid date (gefs_patterns_ref.npz).

Each start's ensemble-mean anomaly is smoothed with the regimes' 5-day running mean along the lead (partial windows
at the ends, as live) and classified with BOTH seasonal sets; the stored label takes the set of each VALID day's
season. Observed (data/r1_labels.npz, NCEP/NCAR R1): the centred 5-day-mean regime on the valid day in that day's set;
persistence: the regime of the 3-day mean ending the day before day 1, classified with the valid day's set.
Day d is valid on start + (d - 1) for both models (GEPS8 lead L = d - 0.5).

    python scripts/regimes/build_regime_hindcast.py --model geps [--workers 2]
    python scripts/regimes/build_regime_hindcast.py --model gefs --hind ~/data_archive/regimes_work/gefs_hind2
    -> scripts/regimes/data/regime_hindcast_{model}.npz  S (yyyymmdd), fc_<sector> / ob_<sector> / pers_<sector>
       (start, day) int8 in the valid day's set (0-3, 4 = no regime, -1 missing), fc_<sector>_<season> both sets
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import regime_core as RC                                                  # noqa: E402

GEPS = Path.home() / "data_archive" / "geps_subx"
NDAYS = 35
_REF = None


def ref():
    global _REF
    if _REF is None:
        _REF = RC.Ref()
    return _REF


def classify_both(an, dates):
    """an (days, 73, 144) ensemble-mean anomaly (m) -> {sk: {season: labels (days,)}}."""
    out = {}
    for sk, spec in RC.SECTORS.items():
        sec, _, _ = RC.sector_cut(an, spec)
        v = RC.running_partial(sec, RC.SMOOTH, axis=0)
        out[sk] = {s: ref().classify(v, sk, s)[0] for s in RC.SEASONS}
    return out


# ── GEPS8 ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def geps_file(path):
    """Labels of every start in one monthly GEPS8 zg500 file (telecon_hindcast.one_file's anomaly arithmetic)."""
    sys.path.insert(0, str(GEPS))
    import pandas as pd
    import xarray as xr
    import telecon_hindcast as th
    cw = th.clim_wrapped25("zg500")
    shift = th.period_shift("zg500")
    ds = xr.open_dataset(path)
    a = ds["zg"].isel(P=0).load().astype("float32")
    ds.close()
    S = pd.DatetimeIndex(a.S.values)
    L = a.L.values
    a = xr.concat([a, a.isel(X=0).assign_coords(X=360.0)], dim="X")
    a25 = a.interp(Y=RC.LAT25, X=RC.LON25).values.astype("float32")          # (S, L, 73, 144)
    del a
    out = []
    for si, s0 in enumerate(S):
        refc = cw.interp(doy=s0.dayofyear).sel(L=L, method="nearest").values
        an = a25[si] - refc
        valid = s0 + pd.to_timedelta(np.floor(L).astype(int), unit="D")      # L = 0.5 is the start date itself
        if shift is not None:
            an = an + shift[valid.dayofyear.values - 1]
        an = an[:NDAYS]
        if not np.isfinite(an).all():
            continue
        out.append((s0.strftime("%Y%m%d"), classify_both(an, None)))
    return out


def geps_starts(workers):
    files = sorted((GEPS / "geps8_hindcast").glob("geps8_zg500_*.nc"))
    res, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(geps_file, files), 1):
            res += r
            if i % 20 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files, {len(res)} starts, {time.time() - t0:.0f} s", flush=True)
    return sorted({s: lab for s, lab in res}.items(), key=lambda x: x[0])


# ── GEFSv12 reforecast ────────────────────────────────────────────────────────────────────────────────────────────
def gefs_starts(hind: Path):
    sys.path.insert(0, str(REPO / "scripts" / "gefs"))
    import gefs_patterns as GP
    import gefs_daily as G
    gref = GP.Ref(Path.home() / "data_archive" / "gefs_ref" / "patterns" / GP.REF_NAME)
    files = sorted(Path(hind).glob("gefs_hind2_*.npz"))
    if len(files) != 12:
        raise SystemExit(f"need the 12 monthly gefs_hind2 files, found {len(files)}")
    means, acc, n = {}, np.zeros((len(GP.CENTRES), NDAYS, 73, 144)), np.zeros(len(GP.CENTRES))
    t0 = time.time()
    for f in files:
        with np.load(f) as z:
            starts = sorted({k.split("_")[0] for k in z.files if k.endswith("_members")})
            for s in starts:
                m = G.unpack("zg500", z[f"{s}_m_zg500"]).astype("float64")        # (members, 35, 73, 144)
                means[s] = m.mean(0).astype("float32")
                d = GP.doy(GP.parse_date(s))
                for c in GP.CENTRES:
                    if min(abs(d - c), 365 - abs(d - c)) <= 15:
                        acc[GP.centre_index(c)] += m.sum(0); n[GP.centre_index(c)] += m.shape[0]
        print(f"  {f.name}: {len(starts)} starts ({time.time() - t0:.0f} s)", flush=True)
    clim = acc / np.maximum(n, 1)[:, None, None, None]
    print(f"  climatology: member-runs per centre {int(n.min())}-{int(n.max())}", flush=True)
    out = []
    for s in sorted(means):
        base = GP.parse_date(s)
        lo, hi, w = GP.bracket(GP.doy(base))
        c = (1 - w) * clim[GP.centre_index(lo)] + w * clim[GP.centre_index(hi)]
        days = GP.valid_days(base)
        sh = gref.shift("zg500", days)
        an = (means[s] - c - (sh if sh is not None else 0.0)) * GP.UNITS["zg500"]
        if np.isfinite(an[:, 4:25]).all():
            out.append((s, classify_both(an, days)))
    return out


# ── pairing with R1 ───────────────────────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("geps", "gefs"))
    ap.add_argument("--hind", help="directory of gefs_hind2_MM.npz (gefs)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    res = geps_starts(a.workers) if a.model == "geps" else gefs_starts(Path(a.hind))
    r1 = np.load(RC.DATA / "r1_labels.npz")
    t1 = [RC.parse_date(str(x)) for x in r1["t"]]
    pos = {d: i for i, d in enumerate(t1)}
    S = [s for s, _ in res]
    save = {"S": np.array([int(s) for s in S], "int32")}
    for sk in RC.SECTORS:
        fc_both = {s: np.stack([lab[sk][s] for _, lab in res]) for s in RC.SEASONS}
        fc = np.full((len(S), NDAYS), -1, "int8"); ob = fc.copy(); ps = fc.copy()
        for i, s in enumerate(S):
            s0 = RC.parse_date(s)
            ip = pos.get(s0 - dt.timedelta(days=1))
            for j in range(NDAYS):
                v = s0 + dt.timedelta(days=j)
                se = RC.season_of(v.month)
                fc[i, j] = fc_both[se][i, j]
                k = pos.get(v)
                if k is not None:
                    ob[i, j] = r1[f"lab_{sk}_{se}"][k]
                if ip is not None:
                    ps[i, j] = r1[f"trail_{sk}_{se}"][ip]
        save[f"fc_{sk}"], save[f"ob_{sk}"], save[f"pers_{sk}"] = fc, ob, ps
        for s in RC.SEASONS:
            save[f"fc_{sk}_{s}"] = fc_both[s].astype("int8")
        ok = (fc >= 0) & (ob >= 0)
        print(f"  {sk}: hit day 1/7/14/21/28 " + " ".join(
            f"{np.mean((fc[:, j] == ob[:, j])[ok[:, j]]):.2f}" for j in (0, 6, 13, 20, 27))
            + f"  (persistence " + " ".join(f"{np.mean((ps[:, j] == ob[:, j])[ok[:, j] & (ps[:, j] >= 0)]):.2f}"
                                             for j in (0, 6, 13, 20, 27)) + ")", flush=True)
    out = RC.HINDCAST[a.model]
    np.savez_compressed(out, **save, note=np.array(
        "season-aware regime labels (build_regime_hindcast.py): fc = hindcast ensemble-mean regime, ob = NCEP R1, "
        "pers = R1 3-day mean ending the day before day 1; every label in the valid day's seasonal set; day d valid "
        "on start + d - 1"))
    print(f"{out.relative_to(REPO)}: {len(S)} starts {S[0]} .. {S[-1]}, {out.stat().st_size / 1e3:.0f} kB "
          f"({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
