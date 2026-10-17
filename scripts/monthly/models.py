"""Dynamical seasonal-model guidance for the month-ahead outlook: ECMWF SEAS5, UKMO GloSea and ECCC CanSIPS through
the C3S multi-system archive, as per-year hindcast tables and the live issue, on the outlook's 3° grid (0-90N).

The outlook for month M is issued on the 15th of M-1; the C3S issue of month M-1 (published around the 13th) is then
the newest, and M is its lead month 2. CanSIPS enters C3S as its two component models (eccc system 4, GEM5-NEMO, and
system 5, CanESM5, 20 members each), each against its own hindcast, merged here with equal weight into one CanSIPS.

Tables (scripts/monthly/data/):
    c3s_hc_<centre>_<system>_<MM>.npz   per system and start month MM: years 1993-2016, and for each of t2m, prcp,
                                        z500 the per-year ensemble mean at lead month 2 as a leave-one-out anomaly
                                        against the 1993-2016 hindcast mean (float16, (24, P)). Fields are added as
                                        they arrive, so the single-level and pressure-level parts can come from
                                        different places.
    c3s_live_<YYYYMM>.npz               per issue: each system's ensemble-mean anomaly at lead month 2
                                        (`<centre>_<system>_<var>`, (P,)) from the C3S post-processed anomaly sets.

Where they come from:
    Actions (monthly-models.yml, the 14th): CDS requests limited to lead month 2 and 0-90N - the hindcast of the
        coming start month when its table is missing (then other start months, under a time budget), and the live
        issue. `seasonal-monthly-{single,pressure}-levels` and `seasonal-postprocessed-{single,pressure}-levels`.
    A local C3S archive (--reduce-local DIR): raw hindcast GRIBs already on disk, named as the site's C3S scripts
        name them (hc_<centre>_<system>_<var>_<MM>.grib, var t2m / tp, leads 1-6, global 1°), reduced the same way;
        SEAS5's own hc_gl_<MM>.grib is read too. Nothing is downloaded on this path.

All anomalies are against the model's own 1993-2016 hindcast; forecast.py moves them to 1991-2020 with ERA5 (the
outlook's base) and blends them with the statistical forecast. Downloads run in GitHub Actions only (C3S_ALLOW_FETCH=1
overrides, as for the site's other C3S scripts).

    python scripts/monthly/models.py --fetch-live --issue 202610                  # Actions
    python scripts/monthly/models.py --fetch-hindcast --months 10 --budget-min 60  # Actions
    python scripts/monthly/models.py --reduce-local ~/scorvec/scripts/sst/data/c3s/terc   # a local C3S archive
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mo_grid as G

REPO = HERE.parents[1]
DATA = G.DATA


def _registry() -> dict:
    """The current C3S system of each centre, from the seasonal page's model list (assets/sst/data/c3s_maps.json, kept
    by c3s.yml), so a new GloSea or SEAS5 system switches the hindcast tables and the live issue together."""
    import json

    cur = {"ecmwf": ["51"], "ukmo": ["610"], "eccc": ["4", "5"]}
    try:
        ids = [m["id"] for m in json.loads((REPO / "assets/sst/data/c3s_maps.json").read_text())["models"]]
        for centre in cur:
            got = sorted({i.rsplit("_", 1)[1] for i in ids if i.rsplit("_", 1)[0] == centre})
            if got:
                cur[centre] = got
    except Exception:  # noqa: BLE001  the registry is optional; the defaults are the systems of October 2026
        pass
    return cur


_REG = _registry()
# the systems blended: SEAS5, GloSea, and CanSIPS as its two C3S components (GEM5-NEMO and CanESM5)
MODELS = {
    "SEAS5": [("ecmwf", s) for s in _REG["ecmwf"]],
    "UKMO": [("ukmo", s) for s in _REG["ukmo"]],
    "CanSIPS": [("eccc", s) for s in _REG["eccc"]],
}
SYSTEMS = tuple(cs for parts in MODELS.values() for cs in parts)
HC_YEARS = list(range(1993, 2017))
LEAD = 2  # lead month 2 of the issue month's start = the target month
AREA = [90, -180, 0, 180]  # N, W, S, E
GRAV = 9.80665
# outlook var -> (raw dataset variable, post-processed anomaly variable, pressure level or None, factor to units)
VARS = {
    "t2m": ("2m_temperature", "2m_temperature_anomaly", None, 1.0),
    "prcp": ("total_precipitation", "total_precipitation_anomalous_rate_of_accumulation", None, 86400.0 * 1000.0),
    "z500": ("geopotential", "geopotential_anomaly", "500", 1.0 / GRAV),
}
KEYS = ("cfVarName", "number", "dataDate", "dataTime", "forecastMonth", "indexingDate")


def tag(centre: str, system: str) -> str:
    return f"{centre}_{system}"


def hc_path(centre: str, system: str, mm: int) -> Path:
    return DATA / f"c3s_hc_{tag(centre, system)}_{mm:02d}.npz"


def live_path(issue: str) -> Path:
    return DATA / f"c3s_live_{issue}.npz"


def fetch_allowed() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("C3S_ALLOW_FETCH") == "1"


# ------------------------------------------------------------------------------------------------- reading GRIBs
def _coarsen(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    co = G.Coarsener(lat, lon)
    return co(np.asarray(field, np.float64), "nh").reshape(*np.shape(field)[:-2], -1).astype(np.float32)


def lead_means(path: Path, var: str | None, lead: int, by_year: bool) -> tuple[dict, int]:
    """Ensemble means of lead month `lead` from a C3S monthly GRIB, message by message (lagged systems included:
    a member counts under its NOMINAL start, `indexingDate`). by_year: {year: (P,)} for a hindcast; else {0: (P,)}.
    `var` picks one variable of a multi-variable file by cfVarName prefix (None: the file's only variable)."""
    sys.path.insert(0, str(REPO / "scripts" / "sst"))
    import c3s_grib

    sums, counts, grid = {}, {}, None
    for m, field, g in c3s_grib.messages(path, KEYS):
        if var is not None and not str(m["cfVarName"]).startswith(var):
            continue
        if m["forecastMonth"] is not None and int(m["forecastMonth"]) != lead:
            continue
        if not np.isfinite(field).any():
            continue
        grid = g
        y = int((m["indexingDate"] or m["dataDate"]) // 10000) if by_year else 0
        sums[y] = sums.get(y, 0.0) + field
        counts[y] = counts.get(y, 0) + 1
    if grid is None:
        return {}, 0
    out = {y: _coarsen(sums[y] / counts[y], grid.lat, grid.lon) for y in sums}
    return out, max(counts.values())


def loo_anom(x: np.ndarray) -> np.ndarray:
    """Per-year values (ny, P) -> anomalies against the mean of the OTHER years."""
    n = len(x)
    return ((x - x.mean(0)) * n / (n - 1)).astype(np.float32)


def write_hindcast(centre: str, system: str, mm: int, var: str, per_year: dict) -> bool:
    """Add one variable to a system's start-month table (complete years only)."""
    if sorted(per_year) != HC_YEARS:
        print(
            f"  {tag(centre, system)} {mm:02d} {var}: years {sorted(per_year)[:2]}.. incomplete, not kept", flush=True
        )
        return False
    p = hc_path(centre, system, mm)
    d = {}
    if p.exists():
        with np.load(p) as f:
            d = {k: f[k] for k in f.files}
    d["years"] = np.array(HC_YEARS)
    d[var] = loo_anom(np.stack([per_year[y] for y in HC_YEARS])).astype(np.float16)
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **d)
    tmp.replace(p)
    print(f"  {tag(centre, system)} {mm:02d} {var}: table written", flush=True)
    return True


# --------------------------------------------------------------------------------------------------- CDS fetches
def _client():
    if not fetch_allowed():
        raise RuntimeError("CDS downloads run in GitHub Actions only (set C3S_ALLOW_FETCH=1 to override)")
    import cdsapi

    return cdsapi.Client(timeout=600, quiet=True, progress=False, wait_until_complete=True, retry_max=1)


def _retrieve(dataset: str, req: dict, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    t0 = time.time()
    try:
        _client().retrieve(dataset, req, str(tmp))
    except Exception as e:  # noqa: BLE001  no data for this system/month, or a CDS failure: skipped, logged
        print(f"    {dataset} {req.get('originating_centre')}/{req.get('system')}: FAILED {str(e)[:160]}", flush=True)
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    print(f"    {dest.name}: {dest.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)
    return True


def _request(centre, system, var, years, month, post: bool) -> tuple[str, dict]:
    raw, anom, level, _ = VARS[var]
    kind = "pressure" if level else "single"
    dataset = f"seasonal-{'postprocessed' if post else 'monthly'}-{kind}-levels"
    req = {
        "originating_centre": centre,
        "system": system,
        "variable": [anom if post else raw],
        "product_type": ["monthly_mean"],
        "year": [str(y) for y in years],
        "month": [f"{month:02d}"],
        "leadtime_month": [str(LEAD)],
        "area": AREA,
        "grid": ["1", "1"],
        "data_format": "grib",
    }
    if level:
        req["pressure_level"] = [level]
    return dataset, req


def fetch_hindcast(mm: int, work: Path, deadline: float) -> None:
    for centre, system in SYSTEMS:
        have = set()
        if hc_path(centre, system, mm).exists():
            with np.load(hc_path(centre, system, mm)) as f:
                have = set(f.files)
        for var, (_, _, _, fac) in VARS.items():
            if var in have:
                continue
            if time.time() > deadline:
                print("  time budget spent; the rest resumes next run", flush=True)
                return
            dest = work / f"hc_{tag(centre, system)}_{var}_{mm:02d}_l{LEAD}.grib"
            if not dest.exists() and not _retrieve(*_request(centre, system, var, HC_YEARS, mm, post=False), dest):
                continue
            per_year, _ = lead_means(dest, None, LEAD, by_year=True)
            write_hindcast(centre, system, mm, var, {y: v * fac for y, v in per_year.items()})
            dest.unlink(missing_ok=True)


def fetch_live(issue: str, work: Path) -> None:
    """Each system's live ensemble-mean anomaly at lead month 2 (only systems whose start month has a table)."""
    y, mm = int(issue[:4]), int(issue[4:])
    p = live_path(issue)
    d = {}
    if p.exists():
        with np.load(p) as f:
            d = {k: f[k] for k in f.files}
    for centre, system in SYSTEMS:
        for var, (_, _, _, fac) in VARS.items():
            key = f"{tag(centre, system)}_{var}"
            if key in d:
                continue
            dest = work / f"live_{key}_{issue}.grib"
            if not dest.exists() and not _retrieve(*_request(centre, system, var, [y], mm, post=True), dest):
                continue
            mean, nmem = lead_means(dest, None, LEAD, by_year=False)
            if 0 in mean and nmem >= 5:
                d[key] = (mean[0] * fac).astype(np.float32)
                d[f"{key}_n"] = np.array(nmem)
                print(f"  live {key}: {nmem} members", flush=True)
            dest.unlink(missing_ok=True)
    if d:
        np.savez_compressed(p, **d)


# ------------------------------------------------------------------------------------------ a local C3S archive
def reduce_local(root: Path, months: list[int]) -> None:
    """Reduce raw hindcast GRIBs already on disk (the site's C3S naming) into the tables, at lead month 2."""
    seas5 = root.parent.parent / "seas5" / "hindcast"  # scripts/sst/data/seas5/hindcast next to .../c3s/terc
    for mm in months:
        for centre, system in SYSTEMS:
            for var, short in (("t2m", "t2m"), ("prcp", "tp")):
                if hc_path(centre, system, mm).exists():
                    with np.load(hc_path(centre, system, mm)) as f:
                        if var in f.files:
                            continue
                fac = VARS[var][3]
                p = root / f"hc_{tag(centre, system)}_{short}_{mm:02d}.grib"
                pick = None
                if not p.exists() and centre == "ecmwf" and (seas5 / f"hc_gl_{mm:02d}.grib").exists():
                    p, pick = seas5 / f"hc_gl_{mm:02d}.grib", ("t2m" if var == "t2m" else "tprate")
                if not p.exists():
                    print(f"  {p.name}: not on disk", flush=True)
                    continue
                t0 = time.time()
                per_year, _ = lead_means(p, pick, LEAD, by_year=True)
                write_hindcast(centre, system, mm, var, {y: v * fac for y, v in per_year.items() if y in HC_YEARS})
                print(f"    from {p.name} in {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------------------------------- for the blend
def hindcast(model: str, mm: int, var: str) -> np.ndarray | None:
    """(24, P) per-year anomalies of one model (CanSIPS: the mean of its two components) for start month mm."""
    parts = []
    for centre, system in MODELS[model]:
        p = hc_path(centre, system, mm)
        if not p.exists():
            return None
        with np.load(p) as f:
            if var not in f.files:
                return None
            parts.append(f[var].astype(np.float32))
    return np.mean(parts, 0)


def live(model: str, issue: str, var: str) -> np.ndarray | None:
    p = live_path(issue)
    if not p.exists():
        return None
    with np.load(p) as f:
        keys = [f"{tag(c, s)}_{var}" for c, s in MODELS[model]]
        if not all(k in f.files for k in keys):
            return None
        return np.mean([f[k] for k in keys], 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-live", action="store_true")
    ap.add_argument("--fetch-hindcast", action="store_true")
    ap.add_argument("--reduce-local", type=Path, help="directory of raw C3S hindcast GRIBs (hc_<centre>_<system>_...)")
    ap.add_argument("--issue", help="YYYYMM of the C3S issue (default: this month)")
    ap.add_argument("--months", help="start months for the hindcast (default: the issue month, then the rest)")
    ap.add_argument("--budget-min", type=float, default=60.0)
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "c3s_monthly")
    a = ap.parse_args()
    issue = a.issue or datetime.now(UTC).strftime("%Y%m")
    first = int(issue[4:])
    months = [int(x) for x in a.months.split(",")] if a.months else [first] + [m for m in range(1, 13) if m != first]
    deadline = time.time() + 60 * a.budget_min
    if a.reduce_local:
        reduce_local(a.reduce_local.expanduser(), months)
    if a.fetch_hindcast:
        for mm in months:
            fetch_hindcast(mm, a.work, deadline)
    if a.fetch_live:
        fetch_live(issue, a.work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
