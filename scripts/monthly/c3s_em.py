#!/usr/bin/env python3
"""C3S seasonal ensemble means on the long-range outlook's 3° Northern Hemisphere cells, for the blend (blend.py).

Per system (c3s_nino34.MODELS: seven systems from six centres) and variable (2 m temperature, precipitation, 500 hPa
height), the ensemble mean of every lead month 1-6 (lead 1 = the start month) is box-averaged from the CDS's 1°
grid onto the outlook's cells and kept as two kinds of table, assets of the release `c3s-em-v1`:

    hc_em_{centre}_{system}_{var}_{MM}.npz    the 1993-2016 hindcast of start month MM: per year and lead the
                                              anomaly against the system's own 1993-2016 mean (float16), and that mean
    fc_em_{centre}_{system}_{var}_{YYYYMM}.npz  the issue of YYYYMM: the ensemble mean per lead and the member count

A hindcast table is built once per start month and kept for good; an issue's when C3S publishes it (around the 13th).
Raw GRIBs (0-90N at 1°, a few hundred MB for a hindcast) are reduced as they land and deleted. CDS downloads run in
Actions only (c3s-em.yml, in the `seas5` concurrency group: one CDS key, never two requests at once).

    python c3s_em.py --hindcast 11 12 --budget-min 240     # hindcast tables still missing for these start months
    python c3s_em.py --issue 202611                         # the issue's tables
    python c3s_em.py --status 202611                        # what is in C3S_EM_DIR for that issue
    python c3s_em.py --pull --status 202611                 # first fetch the issue's tables from the release

--pull fetches the tables the other arguments name from the release's public download links (no GitHub API: the
repository's hourly API allowance is shared by every workflow and ran out on the first run, 2026-10-29); a table not
there yet is simply missing.
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes (libproj double-free trap)
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sst"))
sys.path.insert(0, str(HERE))
import mo_grid as G

EM_DIR = Path(os.environ.get("C3S_EM_DIR", HERE / "data" / "c3s_em"))
TAG = "c3s-em-v1"
RELEASE_URL = f"https://github.com/scorvec/scorvec.github.io/releases/download/{TAG}/{{}}"
HC_YEARS = range(1993, 2017)
LEADS = range(1, 7)
VARS = {  # outlook variable -> (CDS dataset, CDS variable, pressure level, eccodes cfVarName)
    "t2m": ("seasonal-monthly-single-levels", "2m_temperature", None, ("t2m",)),
    "tp": ("seasonal-monthly-single-levels", "total_precipitation", None, ("tprate", "tp")),
    "z500": ("seasonal-monthly-pressure-levels", "geopotential", "500", ("z",)),
}
OUTLOOK_VAR = {"t2m": "t2m", "prcp": "tp", "z500": "z500"}
AREA = [90, -180, -2, 180]  # N, W, S, E: the 0-90N cells reach 1.5° south of the equator
MIN_SYSTEMS = 4  # groups (CanSIPS's two components count once) needed for a multi-system mean


def systems() -> list[tuple[str, str]]:
    import c3s_nino34 as C

    return [(c, s) for c, s, _, _ in C.MODELS]


def group(centre: str, system: str) -> str:
    """Systems pooled into one member of the multi-system mean: CanSIPS's two components, as on seasonal.html."""
    return centre if centre == "eccc" else f"{centre}_{system}"


def hc_path(centre: str, system: str, var: str, mm: int) -> Path:
    return EM_DIR / f"hc_em_{centre}_{system}_{var}_{mm:02d}.npz"


def fc_path(centre: str, system: str, var: str, issue: str) -> Path:
    return EM_DIR / f"fc_em_{centre}_{system}_{var}_{issue}.npz"


# ------------------------------------------------------------------------------------------------------- fetching
def retrieve(centre: str, system: str, var: str, years, month: int, dest: Path) -> bool:
    from c3s_ref import guard_fetch
    from cds_seq import available

    guard_fetch("Copernicus CDS")
    import cdsapi

    ds, cvar, level, _ = VARS[var]
    req = {
        "originating_centre": centre,
        "system": system,
        "variable": cvar,
        "product_type": "monthly_mean",
        "year": [str(y) for y in years],
        "month": f"{month:02d}",
        "leadtime_month": [str(L) for L in LEADS],
        "area": AREA,
        "grid": "1.0/1.0",
        "data_format": "grib",
    }
    if level:
        req["pressure_level"] = level
    # a start the CDS does not hold yet (UKMO's hindcast of a month appears with its issue) waits hours, then fails
    if available(ds, req) is False:
        print(f"  {centre}_{system} {var} {month:02d}: not in the CDS catalogue yet", flush=True)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            cl = cdsapi.Client(timeout=900, quiet=True, progress=False, wait_until_complete=True, retry_max=1)
            cl.retrieve(ds, req, str(dest))
            return dest.exists() and dest.stat().st_size > 0
        except Exception as e:  # noqa: BLE001  a system or issue the CDS does not hold (yet)
            msg = str(e).replace("\n", " ")
            print(f"  {centre}_{system} {var} {month:02d}: attempt {attempt + 1} failed ({msg[:120]})", flush=True)
            if "no data" in msg.lower():
                return False
            time.sleep(10)
    return False


def reduce(path: Path, var: str, hindcast: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(years, ensemble mean (Y, 6, P) in °C / mm/day / m on the outlook's 0-90N cells, members (Y, 6)) of a
    monthly-mean GRIB. A lagged member is filed under its nominal start (indexingDate) where the file says so."""
    import c3s_grib

    names = VARS[var][3]
    acc, cnt, co = {}, {}, None
    for m, v, g in c3s_grib.messages(path, ("dataDate", "forecastMonth", "indexingDate", "cfVarName")):
        if m["cfVarName"] not in names or not 1 <= int(m["forecastMonth"]) <= 6:
            continue
        if co is None:
            co = G.Coarsener(g.lat, g.lon)
        y = int(str(m["indexingDate"] or m["dataDate"])[:4])
        x = {"t2m": v - 273.15, "tp": v * 86400e3, "z500": v / 9.80665}[var]  # K, m s-1, m2 s-2
        k = (y, int(m["forecastMonth"]))
        if k not in acc:
            acc[k], cnt[k] = np.zeros_like(x), 0
        acc[k] += x
        cnt[k] += 1
    years = sorted({k[0] for k in acc})
    if hindcast:
        years = [y for y in years if y in HC_YEARS]
    mean = np.full((len(years), 6, len(G.LAT_NH) * len(G.LON)), np.nan, np.float32)
    nmem = np.zeros((len(years), 6), np.int32)
    for (y, L), a in acc.items():
        if y in years:
            mean[years.index(y), L - 1] = co(a / cnt[(y, L)], "nh").ravel()
            nmem[years.index(y), L - 1] = cnt[(y, L)]
    return np.array(years), mean, nmem


def build_hindcast(centre: str, system: str, var: str, mm: int) -> bool:
    out = hc_path(centre, system, var, mm)
    raw = EM_DIR / "raw" / f"hc_{centre}_{system}_{var}_{mm:02d}.grib"
    if not retrieve(centre, system, var, HC_YEARS, mm, raw):
        return False
    years, mean, nmem = reduce(raw, var, hindcast=True)
    raw.unlink()
    if list(years) != list(HC_YEARS) or not np.isfinite(mean).all():
        print(f"  {out.name}: incomplete hindcast ({len(years)} years) - not kept", flush=True)
        return False
    clim = mean.mean(0)
    np.savez_compressed(out, years=years, clim=clim, anom=(mean - clim).astype(np.float16), nmem=nmem)
    print(f"  {out.name}: {years[0]}-{years[-1]}, {int(nmem.min())}-{int(nmem.max())} members", flush=True)
    return True


def build_issue(centre: str, system: str, var: str, issue: str) -> bool:
    out = fc_path(centre, system, var, issue)
    raw = EM_DIR / "raw" / f"fc_{centre}_{system}_{var}_{issue}.grib"
    if not retrieve(centre, system, var, [int(issue[:4])], int(issue[4:]), raw):
        return False
    years, mean, nmem = reduce(raw, var, hindcast=False)
    raw.unlink()
    if len(years) != 1 or not np.isfinite(mean).all():
        print(f"  {out.name}: incomplete ({len(years)} start years) - not kept", flush=True)
        return False
    np.savez_compressed(out, em=mean[0], nmem=nmem[0])
    print(f"  {out.name}: {int(nmem[0].min())} members", flush=True)
    return True


# ------------------------------------------------------------------------------------------------------- reading
def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def target(issue: str, v: str, month: int, span: int, live: bool = True) -> dict | None:
    """The multi-system mean for the outlook variable v over the `span` months from calendar month `month`, from the
    C3S issue of month `issue` (YYYYMM): {"hind": (24, P) anomalies of the 1993-2016 starts, "live": (P,) the issue's
    anomaly, "years": the 24 hindcast years of the TARGET's first month, "systems": the groups used}, both against
    the systems' own 1993-2016 hindcast (blend.py moves them to 1991-2020); None when fewer than MIN_SYSTEMS groups
    have both tables (the hindcast alone with live=False: a skill-only run) or the target runs past lead 6."""
    var = OUTLOOK_VAR[v]
    mm = int(issue[4:])
    l0 = (month - mm) % 12  # lead index (0 = the start month) of the target's first month
    if l0 + span > 6:
        return None
    groups: dict[str, list] = {}
    for c, s in systems():
        h, f = _load(hc_path(c, s, var, mm)), _load(fc_path(c, s, var, issue)) if live else None
        if h is None or (live and f is None):
            continue
        hind = h["anom"][:, l0 : l0 + span].astype(np.float64).mean(1)
        now = (f["em"][l0 : l0 + span] - h["clim"][l0 : l0 + span]).mean(0) if live else None
        groups.setdefault(group(c, s), []).append((hind, now))
    if len(groups) < MIN_SYSTEMS:
        return None
    g = {k: (np.mean([a for a, _ in x], 0), np.mean([b for _, b in x], 0) if live else None) for k, x in groups.items()}
    return {
        "hind": np.mean([a for a, _ in g.values()], 0),
        "live": np.mean([b for _, b in g.values()], 0) if live else None,
        "years": np.array(HC_YEARS) + int(month < mm),  # a target in the next calendar year
        "systems": sorted(g),
    }


def pull(paths: list[Path]) -> int:
    """Download the named tables that are not on disk from the release; returns how many arrived."""
    import urllib.error
    import urllib.request

    got = 0
    for p in paths:
        if p.exists():
            continue
        try:
            with urllib.request.urlopen(RELEASE_URL.format(p.name), timeout=120) as r:
                p.write_bytes(r.read())
            got += 1
        except urllib.error.HTTPError as e:
            if e.code != 404:  # 404: not built yet
                raise
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindcast", nargs="*", type=int, help="start months whose missing hindcast tables to build")
    ap.add_argument("--issue", help="YYYYMM: build the issue's missing tables")
    ap.add_argument("--budget-min", type=float, default=240.0, help="stop starting new requests after this")
    ap.add_argument("--status", help="YYYYMM: list the tables an issue's blend needs")
    ap.add_argument("--pull", action="store_true", help="first fetch the named tables from the release")
    a = ap.parse_args()
    EM_DIR.mkdir(parents=True, exist_ok=True)
    if a.pull:
        want = []
        for c, s in systems():
            for var in VARS:
                want += [fc_path(c, s, var, i) for i in {a.issue, a.status} - {None}]
                want += [
                    hc_path(c, s, var, mm) for mm in {*(a.hindcast or []), *([int(a.status[4:])] if a.status else [])}
                ]
        print(f"pulled {pull(want)} of {len(want)} tables from {TAG}", flush=True)
    t0 = time.time()
    left = lambda: (time.time() - t0) / 60 < a.budget_min  # noqa: E731
    if a.issue:
        for c, s in systems():
            for var in VARS:
                if not fc_path(c, s, var, a.issue).exists() and left():
                    build_issue(c, s, var, a.issue)
    for mm in a.hindcast or []:
        for c, s in systems():
            for var in VARS:
                if not hc_path(c, s, var, mm).exists() and left():
                    build_hindcast(c, s, var, mm)
    if a.status:
        mm = int(a.status[4:])
        for var in VARS:
            hc = [f"{c}_{s}" for c, s in systems() if hc_path(c, s, var, mm).exists()]
            fc = [f"{c}_{s}" for c, s in systems() if fc_path(c, s, var, a.status).exists()]
            print(f"{var}: hindcast {len(hc)}/{len(systems())} {hc}; issue {len(fc)}/{len(systems())} {fc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
