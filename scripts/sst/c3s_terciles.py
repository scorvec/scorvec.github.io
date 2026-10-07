#!/usr/bin/env python3
"""Fetch what the C3S tercile maps need: forecast members and the matching hindcast.

The anomaly maps on seasonal.html come from the postprocessed collection, which already
holds the anomaly against each centre's own hindcast and is tiny. Tercile probabilities
cannot be taken from an anomaly alone -- they need the hindcast DISTRIBUTION per grid
point (the 33rd and 67th percentiles of that model's own climate) and the spread of the
live members around it. So this pulls the raw fields instead:

  hindcast   1993-2016, the C3S common period, same start month, leads 1-6
  forecast   this issue's members, leads 1-6

on the 1 deg grid the SEAS5 page already uses, so every system is counted on one grid.
ECMWF SEAS5 is read from the SEAS5 page's own global files when they are on disk
(scripts/sst/data/seas5/{hindcast,forecast}/hc_gl_{MM}.grib / fc_gl_{YYYYMM}.grib -- c3s.yml
restores the issue's forecast from seas5.yml's cache); otherwise its t2m and precipitation are
fetched like any other system.

SINCE 2026-10-07 (c3s.yml, Actions): every raw file is reduced to a table the moment it lands
(c3s_ref: hc_terc_* per start month, is_terc_* per issue) and deleted, so a start month's ~9 GB
of hindcast is downloaded once, ever, and an interrupted issue resumes without fetching again.
The renderers (c3s_terciles_render, c3s_popt_monthly, c3s_popt_daily) read the tables.

    python scripts/sst/c3s_terciles.py --issue 202610 --budget-min 120   # Actions
    python scripts/sst/c3s_terciles.py --only dwd_22 --kind hc
"""
from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray (libproj double-free trap)
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c3s_nino34 import MODELS, _client
import c3s_ref as CR

STORE = HERE / "data" / "c3s" / "terc"
SEAS5 = HERE / "data" / "seas5"
DATASET = "seasonal-monthly-single-levels"
HC_YEARS = [str(y) for y in range(1993, 2017)]          # the C3S common hindcast period
LEADS = ["1", "2", "3", "4", "5", "6"]
GRID = ["1", "1"]
# variable -> (CDS name, grib short name, factor to display units)
VARS = {
    "t2m": ("2m_temperature", "t2m", 1.0),
    "tp":  ("total_precipitation", "tprate", 86400.0 * 1000.0),
}
POP_LEADS = {f"m{i + 1}": [i] for i in range(6)}        # one lead month per column (population cards)
POP_REGIONS = ("us", "br")


def local_seas5(kind: str, issue: str) -> Path:
    """The SEAS5 page's own global file, which holds t2m, tprate and sst on the 1 deg grid."""
    return (SEAS5 / "hindcast" / f"hc_gl_{issue[4:6]}.grib") if kind == "hc" \
        else (SEAS5 / "forecast" / f"fc_gl_{issue}.grib")


def path_for(kind: str, centre: str, system: str, var: str, issue: str) -> Path:
    if centre == "ecmwf" and local_seas5(kind, issue).exists():
        return local_seas5(kind, issue)
    stem = f"{kind}_{centre}_{system}_{var}_" + (issue[4:6] if kind == "hc" else issue)
    return STORE / f"{stem}.grib"


def fetch(kind: str, centre: str, system: str, var: str, issue: str, force: bool = False) -> bool:
    p = path_for(kind, centre, system, var, issue)
    if centre == "ecmwf" and p == local_seas5(kind, issue):
        print(f"  {centre}/{system} {var} {kind}: local SEAS5 file", flush=True)
        return True
    if p.exists() and not force and p.stat().st_size > 1_000_000:
        print(f"  {centre}/{system} {var} {kind}: cached ({p.stat().st_size / 1e6:.0f} MB)", flush=True)
        return True
    STORE.mkdir(parents=True, exist_ok=True)
    req = {
        "originating_centre": centre, "system": str(system),
        "variable": [VARS[var][0]], "product_type": ["monthly_mean"],
        "year": HC_YEARS if kind == "hc" else [issue[:4]],
        "month": [issue[4:6]], "leadtime_month": LEADS,
        "data_format": "grib", "grid": GRID,
    }
    tmp = p.with_suffix(".part")
    t0 = time.time()
    try:
        _client().retrieve(DATASET, req, str(tmp))
    except Exception as e:                                            # noqa: BLE001
        print(f"  {centre}/{system} {var} {kind}: FAILED {str(e)[:120]}", flush=True)
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(p)
    print(f"  {centre}/{system} {var} {kind}: {p.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    return True


def _drop_raw(p: Path) -> None:
    """Delete a reduced raw file in Actions -- never the SEAS5 page's own files."""
    if CR.discard_raw() and p.exists() and STORE in p.parents:
        p.unlink()


# ── tables (c3s_ref) ─────────────────────────────────────────────────────────
def _pop_w(region: str, lat: np.ndarray, lon: np.ndarray):
    w = CR.pop_weights(region, lat, np.where(lon > 180.0, lon - 360.0, lon))
    return None if w.sum() <= 0 else w / w.sum()


def derive_hc_table(centre: str, system: str, var: str, month: str, hp: Path) -> dict | None:
    """hc_terc table from the raw hindcast: tercile bounds per period (+ trend for t2m) and, for
    t2m, the population-weighted hindcast mean per lead (c3s_popt_* anomaly reference)."""
    from c3s_maps_render import PERIODS
    from c3s_terciles_render import DETREND, bounds, period_means
    short, fac = VARS[var][1], VARS[var][2]
    hcm, lat, lon, years = period_means(hp, short, fac, PERIODS)
    if not hcm:
        return None
    arrs, meta = {"lat": lat, "lon": lon}, {"nsamp": {}, "periods": sorted(hcm)}
    for pid, h in hcm.items():
        tb = bounds(h, years, var in DETREND)
        for k, v in tb.items():
            arrs[f"{k}_{pid}"] = v
        meta["nsamp"][pid] = int(h.shape[0])
    del hcm
    if var == "t2m":
        hl, lat2, lon2, _ = period_means(hp, "t2m", 1.0, POP_LEADS)
        for region in POP_REGIONS:
            w = _pop_w(region, lat2, lon2)
            if w is None:
                continue
            arrs[f"hcpop_{region}"] = np.array([float((hl[k] * w[None]).sum(axis=(1, 2)).mean()) for k in POP_LEADS
                                                if k in hl])
    CR.save(CR.hc_terc(centre, system, var, month), meta=meta, **arrs)
    return CR.load(CR.hc_terc(centre, system, var, month))


def hc_table(centre: str, system: str, var: str, issue: str) -> dict | None:
    """The start month's hindcast table, derived from the raw hindcast when only that is on disk."""
    month = issue[4:6]
    t = CR.load(CR.hc_terc(centre, system, var, month))
    if t is not None:
        return t
    hp = path_for("hc", centre, system, var, issue)
    if not hp.exists():
        return None
    try:
        return derive_hc_table(centre, system, var, month, hp)
    except Exception as e:                                            # noqa: BLE001
        print(f"  {centre}/{system} {var} hc: unreadable ({str(e)[:90]})", file=sys.stderr, flush=True)
        return None


def derive_issue_table(centre: str, system: str, var: str, issue: str, fp: Path, tb: dict) -> dict | None:
    """is_terc table: tercile probabilities per period (exactly c3s_terciles_render.probs), and for t2m
    every member's population-weighted monthly mean per region and the members' start dates."""
    from c3s_maps_render import PERIODS
    from c3s_terciles_render import DETREND, MIN_SPREAD, period_means, probs_from_bounds
    import xarray as xr
    short, fac = VARS[var][1], VARS[var][2]
    y, mo = int(issue[:4]), int(issue[4:6])
    fcm, lat, lon, _ = period_means(fp, short, fac, PERIODS)
    if not fcm or fcm[next(iter(fcm))].shape[-2:] != (tb["lat"].size, tb["lon"].size) \
            or not (np.allclose(lat, tb["lat"]) and np.allclose(lon, tb["lon"])):
        print(f"  {centre}/{system} {var}: forecast and hindcast grids differ; skipped", file=sys.stderr, flush=True)
        return None
    arrs, meta = {"lat": lat, "lon": lon}, {"nmem": {}, "nsamp": tb["meta"]["nsamp"], "periods": []}
    for pid, f in fcm.items():
        if f"lo_{pid}" not in tb:
            continue
        idx = PERIODS[pid]
        target = y + (mo - 1 + idx[len(idx) // 2]) // 12 if var in DETREND else None
        b = {k: tb[f"{k}_{pid}"] for k in ("lo", "hi", "b", "ymean") if f"{k}_{pid}" in tb}
        pr = probs_from_bounds(f, b, target, min_spread=MIN_SPREAD.get(var, 0.0))
        for k in ("above", "below", "normal"):
            arrs[f"{k}_{pid}"] = pr[k]
        meta["nmem"][pid] = int(f.shape[0]); meta["periods"].append(pid)
    del fcm
    if var == "t2m":
        fl, lat2, lon2, _ = period_means(fp, "t2m", 1.0, POP_LEADS)
        for region in POP_REGIONS:
            w = _pop_w(region, lat2, lon2)
            if w is None:
                continue
            keys = [k for k in POP_LEADS if k in fl]
            arrs[f"fcpop_{region}"] = np.stack([(fl[k] * w[None]).sum(axis=(1, 2)) for k in keys], axis=1)
        try:                                       # a lagged system's start dates (c3s_sixh needs them)
            ds = xr.open_dataset(fp, engine="cfgrib",
                                 backend_kwargs={"indexpath": "", "time_dims": ("forecastMonth", "time")})
            arrs["starts"] = np.array(sorted({str(t)[:10] for t in np.atleast_1d(ds["time"].values)}))
            ds.close()
        except Exception:                                             # noqa: BLE001
            pass
    CR.save(CR.is_terc(centre, system, var, issue), meta=meta, **arrs)
    return CR.load(CR.is_terc(centre, system, var, issue))


def issue_table(centre: str, system: str, var: str, issue: str) -> dict | None:
    t = CR.load(CR.is_terc(centre, system, var, issue))
    if t is not None:
        return t
    fp = path_for("fc", centre, system, var, issue)
    if not fp.exists():
        return None
    tb = hc_table(centre, system, var, issue)
    if tb is None:
        return None
    try:
        return derive_issue_table(centre, system, var, issue, fp, tb)
    except Exception as e:                                            # noqa: BLE001
        print(f"  {centre}/{system} {var} fc: unreadable ({str(e)[:90]})", file=sys.stderr, flush=True)
        return None


def hc_pop_means(centre: str, system: str, region: str, issue: str) -> dict | None:
    """{'m1': K, ...} population-weighted hindcast mean per lead (the popt anomaly reference)."""
    t = hc_table(centre, system, "t2m", issue)
    if t is None or f"hcpop_{region}" not in t:
        return None
    return {f"m{i + 1}": float(v) for i, v in enumerate(t[f"hcpop_{region}"])}


def fc_pop_members(centre: str, system: str, region: str, issue: str):
    """[member, lead] population-weighted monthly means (K) of this issue, or None."""
    t = issue_table(centre, system, "t2m", issue)
    if t is None or f"fcpop_{region}" not in t:
        return None
    return t[f"fcpop_{region}"]


def derive_hc_tables(month: str) -> None:
    """Laptop bootstrap: hc_terc tables from whatever raw hindcasts are on disk for this start month."""
    for entry in MODELS:
        centre, system = entry[0], str(entry[1])
        for var in VARS:
            if CR.load(CR.hc_terc(centre, system, var, month)) is not None:
                print(f"  {centre}/{system} {var} {month}: table exists")
                continue
            hp = path_for("hc", centre, system, var, f"2000{month}")
            if not hp.exists():
                print(f"  {centre}/{system} {var} {month}: no raw hindcast here")
                continue
            t0 = time.time()
            derive_hc_table(centre, system, var, month, hp)
            print(f"  {centre}/{system} {var} {month}: {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--only", default="", help="centre_system")
    ap.add_argument("--kind", default="hc,fc")
    ap.add_argument("--vars", default="t2m,tp")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--budget-min", type=float, default=0.0,
                    help="stop STARTING new downloads after this many minutes (0 = no limit)")
    a = ap.parse_args()
    kinds = [k for k in a.kind.split(",") if k in ("hc", "fc")]
    want = [v for v in a.vars.split(",") if v in VARS]
    got = need = 0
    t_start = time.time()

    def budget_left() -> bool:
        return a.budget_min <= 0 or (time.time() - t_start) / 60 < a.budget_min

    for entry in MODELS:
        centre, system = entry[0], str(entry[1])
        if a.only and f"{centre}_{system}" != a.only:
            continue
        for var in want:
            need += 1
            # the tables are the product; raw files are only a means to them
            if not a.force and CR.load(CR.is_terc(centre, system, var, a.issue)) is not None:
                got += 1
                print(f"  {centre}/{system} {var}: issue table exists", flush=True)
                continue
            if "hc" in kinds and hc_table(centre, system, var, a.issue) is None:
                if not budget_left():
                    print(f"  {centre}/{system} {var} hc: budget spent; next run", flush=True)
                    continue
                ok = fetch("hc", centre, system, var, a.issue, a.force) and \
                    hc_table(centre, system, var, a.issue) is not None
                _drop_raw(path_for("hc", centre, system, var, a.issue))
                if not ok:
                    continue
            if "fc" in kinds:
                if not path_for("fc", centre, system, var, a.issue).exists():
                    if not budget_left():
                        print(f"  {centre}/{system} {var} fc: budget spent; next run", flush=True)
                        continue
                    if not fetch("fc", centre, system, var, a.issue, a.force):
                        continue
                if issue_table(centre, system, var, a.issue) is not None:
                    got += 1
                _drop_raw(path_for("fc", centre, system, var, a.issue))
    print(f"{got}/{need} issue table(s) ready under {CR.REF}")
    # the multi-model tercile is only honest if most systems are in it
    return 0 if got >= 0.75 * need else 1


if __name__ == "__main__":
    raise SystemExit(main())
