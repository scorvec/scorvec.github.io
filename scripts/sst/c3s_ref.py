#!/usr/bin/env python3
"""Derived tables + network guard for the C3S multi-system build (.github/workflows/c3s.yml).

The raw inputs are gigabytes of GRIB: each system's 1993-2016 hindcast per start month (~9 GB
for the tercile maps alone), the 6-hourly members (~4 GB an issue), the CPC daily record, geonames
and the 2.2 GB OISST monthly file. Downloads and processing run in Actions only, and nothing is
downloaded twice. So, as for SEAS5 (seas5_ref.py, release seas5-ref-v1), every product reads
compact tables kept as assets of the GitHub release `c3s-ref-v1`, and a raw file is reduced the
moment it lands and then deleted:

  hc_terc_{centre}_{system}_{var}_{MM}.npz   per system, variable and hindcast START MONTH: the
        tercile bounds per period (and the per-point trend for t2m) and, for t2m, the population-
        weighted hindcast mean per lead and region. Built once per start month.
  is_terc_{centre}_{system}_{var}_{YYYYMM}.npz   per issue: the tercile probabilities per period,
        member count, and for t2m each member's population-weighted monthly mean per region plus
        the start dates a lagged system's members carry (the 6-hourly request needs them).
  is_sixh_{centre}_{system}_{region}_{YYYYMM}_m{k}.npz   per issue and forecast month: every
        member's population-weighted DAILY mean (c3s_popt_daily.daily_members), the only thing
        the daily card needs from a 6-hourly chunk.
  obs_c3s_places.npz        geonames cities15000 populated places per region (lat, lon, pop),
        the same filter as seas5_popT.pop_grid_from_file, so weights can be laid on ANY grid.
  obs_c3s_cpc_clim.npz      CPC daily 1993-2016 national (population-weighted) mean per calendar
        month and region -- the base of the observed months on the population-weighted cards.

Two small observed series are committed instead (they grow every month):
  scripts/sst/reference/oisst_box_monthly.json   OISST v2.1 monthly Nino-3.4 and 20S-20N box
        means, extended from PSL OPeNDAP (sequential) -- the monthly relative-index line on the
        static plume figure.

The `obs` tables were built once, outside Actions, from files already held (geonames, a CPC store,
the OISST file) -- the seas5-ref-v1 precedent -- and are never rebuilt in Actions.

    python c3s_ref.py obs               # one-off bootstrap: places + CPC clim + OISST box series
    python c3s_ref.py hc --month 09     # one-off bootstrap: hc_terc tables from raw hindcasts on disk
    python c3s_ref.py ls
    python c3s_ref.py oisst             # Actions: extend the OISST box series (PSL OPeNDAP)
    python c3s_ref.py cpc --issue 202610   # Actions: CPC daily for the six months before the issue
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray (libproj double-free trap)
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REF = Path(os.environ["C3S_REF_DIR"]).resolve() if os.environ.get("C3S_REF_DIR") else HERE / "data" / "c3s" / "ref"
TAG = "c3s-ref-v1"
OISST_BOX = HERE / "reference" / "oisst_box_monthly.json"
OISST_OPENDAP = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
CPC_OPENDAP = "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_global_temp/{stat}.{year}.nc"
CPC_VAR = {"tx": "tmax", "tn": "tmin"}


# ── network guard: downloads run in Actions only ─────────────────────────────
def fetch_allowed() -> bool:
    """Downloads run in GitHub Actions only. C3S_OFFLINE=1 forbids them anywhere; outside Actions
    they need an explicit C3S_ALLOW_FETCH=1, which nothing sets by default."""
    if os.environ.get("C3S_OFFLINE") == "1":
        return False
    return os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("C3S_ALLOW_FETCH") == "1"


def guard_fetch(what: str) -> None:
    if not fetch_allowed():
        raise RuntimeError(
            f"download refused ({what}): C3S data is fetched in GitHub Actions only "
            "(.github/workflows/c3s.yml; set C3S_ALLOW_FETCH=1 to override deliberately)"
        )


def discard_raw() -> bool:
    """Delete a raw GRIB as soon as its table is written (Actions: disk and the never-twice rule)."""
    return os.environ.get("C3S_DISCARD_RAW") == "1"


# ── paths and I/O ────────────────────────────────────────────────────────────
def hc_terc(centre: str, system: str, var: str, month: str) -> Path:
    return REF / f"hc_terc_{centre}_{system}_{var}_{int(month):02d}.npz"


def is_terc(centre: str, system: str, var: str, issue: str) -> Path:
    return REF / f"is_terc_{centre}_{system}_{var}_{issue}.npz"


def is_sixh(centre: str, system: str, region: str, issue: str, k: int) -> Path:
    return REF / f"is_sixh_{centre}_{system}_{region}_{issue}_m{k}.npz"


def obs(name: str) -> Path:
    return REF / f"obs_{name}.npz"


def save(path: Path, meta: dict | None = None, **arrays) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".part{os.getpid()}.npz")
    if meta is not None:
        arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    print(f"    ref {path.name}: {path.stat().st_size / 1e6:.2f} MB", flush=True)
    return path


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "__meta__"}
        if "__meta__" in z.files:
            out["meta"] = json.loads(str(z["__meta__"]))
    return out


# ── population weights on any grid ───────────────────────────────────────────
_PLACES: dict = {}


def places(region: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lat, lon, pop) of the region's populated places, in geonames file order."""
    if region not in _PLACES:
        t = load(obs("c3s_places"))
        if t is None:
            raise FileNotFoundError(f"table {obs('c3s_places').name} missing (release {TAG})")
        for r in ("us", "br"):
            if f"lat_{r}" in t:
                _PLACES[r] = (t[f"lat_{r}"], t[f"lon_{r}"], t[f"pop_{r}"])
    return _PLACES[region]


def pop_weights(region: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Population per cell of any (lat, lon) grid: seas5_popT.pop_grid_from_file exactly (nearest
    lat row and lon column per place, in file order), from the places table instead of the file."""
    la, lo, pop = places(region)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    grid = np.zeros((lat.size, lon.size))
    if la.size:
        i = np.abs(lat[None, :] - la[:, None]).argmin(axis=1)
        j = np.abs(lon[None, :] - lo[:, None]).argmin(axis=1)
        np.add.at(grid, (i, j), pop.astype(float))
    return grid


def derive_places() -> None:
    from seas5_popT import POP_FILE, PPL_CODES, REGIONS

    arrs = {}
    for region, (_label, cc, area, _unit) in REGIONS.items():
        la, lo, pp = [], [], []
        with open(POP_FILE, encoding="utf-8") as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) < 15 or f[8] != cc or f[6] != "P" or f[7] not in PPL_CODES:
                    continue
                try:
                    a, b, p = float(f[4]), float(f[5]), int(f[14])
                except ValueError:
                    continue
                if not (area[2] <= a <= area[0] and area[1] <= b <= area[3]):
                    continue
                la.append(a)
                lo.append(b)
                pp.append(p)
        arrs[f"lat_{region}"] = np.array(la)
        arrs[f"lon_{region}"] = np.array(lo)
        arrs[f"pop_{region}"] = np.array(pp, dtype=np.int64)
        print(f"  {region}: {len(pp)} places, {sum(pp) / 1e6:.1f} M people", flush=True)
    save(
        obs("c3s_places"),
        meta={
            "source": "GeoNames cities15000 (CC BY 4.0), populated places "
            "(seas5_popT.PPL_CODES) inside each region's CDS area"
        },
        **arrs,
    )


# ── CPC daily: recent years from PSL OPeNDAP, the 1993-2016 base from the table ──
def fetch_cpc(region: str, years, force_recent: bool = True) -> None:
    """The region's CPC Global daily Tmax/Tmin for `years`, the layout of the CPC store
    (seas5_extremes.cpc_path: lat descending, lon -180..180). Sequential: PSL OPeNDAP silently
    corrupts parallel reads. The current and previous year are refreshed every call (CPC fills
    and revises its recent days)."""
    import time as _t
    import xarray as xr
    from seas5_extremes import cpc_path
    from seas5_popT import REGIONS

    n, w, s, e = REGIONS[region][2]
    this_year = int(_t.strftime("%Y", _t.gmtime()))
    for yy in years:
        for stat, var in CPC_VAR.items():
            dest = cpc_path(stat, region, yy)
            if dest.exists() and dest.stat().st_size > 0 and not (force_recent and yy >= this_year - 1):
                continue
            guard_fetch(f"CPC {var} {yy}")
            url = CPC_OPENDAP.format(stat=var, year=yy)
            for attempt in range(4):
                try:
                    ds = xr.open_dataset(url)
                    da = ds[var].sel(lat=slice(n, s), lon=slice(w % 360, e % 360)).load()
                    ds.close()
                    break
                except Exception as ex:  # noqa: BLE001
                    print(f"  CPC {var} {yy} {region}: attempt {attempt + 1} failed ({str(ex)[:80]})", flush=True)
                    _t.sleep(20 * (attempt + 1))
            else:
                print(f"::warning::CPC {var} {yy} {region} unavailable", flush=True)
                continue
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
            v = da.values
            if not np.isfinite(v).any() or np.nanmax(np.abs(v)) == 0:
                print(f"::warning::CPC {var} {yy} {region}: empty or all-zero read; not kept", flush=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part.nc")
            da.to_dataset(name=var).to_netcdf(tmp)
            os.replace(tmp, dest)
            print(f"  CPC {var} {yy} {region}: {da.sizes['time']} days to {str(da.time.values[-1])[:10]}", flush=True)


def cpc_clim(region: str) -> dict | None:
    t = load(obs("c3s_cpc_clim"))
    if t is None or f"clim_{region}" not in t:
        return None
    return {m: float(v) for m, v in zip(range(1, 13), t[f"clim_{region}"])}


def derive_cpc_clim() -> None:
    from c3s_popt_daily import cpc_daily_national, cpc_base_clim

    arrs = {}
    for region in ("us", "br"):
        daily = cpc_daily_national(region, range(1993, 2017))
        if not daily:
            print(f"  {region}: no CPC files", flush=True)
            continue
        clim = cpc_base_clim(daily)
        arrs[f"clim_{region}"] = np.array([clim[m] for m in range(1, 13)], dtype=np.float64)
        print(f"  {region}: CPC 1993-2016 national monthly means (degC) {np.round(arrs[f'clim_{region}'], 2)}")
    save(
        obs("c3s_cpc_clim"),
        meta={
            "source": "CPC Global Unified daily Tmax/Tmin (NOAA PSL), (Tmax+Tmin)/2, "
            "population-weighted national means, 1993-2016 per calendar month"
        },
        **arrs,
    )


# ── OISST monthly box means (the plume figure's observed relative index) ─────
def _box_means(sst, months_sel=None):
    """{'n34': [...], 'trop': [...]} cos-weighted box means, as c3s_nino34.observed_relative."""

    def box(la0, la1, lo0, lo1):
        sub = sst.sel(lat=slice(la0, la1), lon=slice(lo0, lo1))
        w = np.cos(np.deg2rad(sub.lat))
        return sub.weighted(w.fillna(0)).mean(("lat", "lon")).compute()

    return box(-5, 5, 190, 240), box(-20, 20, 0, 360)


def derive_oisst_boxes(path: Path) -> None:
    import xarray as xr

    ds = xr.open_dataset(path, chunks={"time": 120})
    n34, trop = _box_means(ds["sst"])
    keys = [f"{y}-{m:02d}" for y, m in zip(n34.time.dt.year.values, n34.time.dt.month.values)]
    doc = {
        "months": keys,
        "n34": [None if not np.isfinite(v) else float(v) for v in n34.values],
        "trop": [None if not np.isfinite(v) else float(v) for v in trop.values],
        "source": "NOAA OISST v2.1 monthly (PSL sst.mon.mean.nc): cos-weighted Nino-3.4 (5S-5N, 170-120W) and "
        "20S-20N means, degC; extended monthly by c3s_ref.update_oisst_boxes",
    }
    OISST_BOX.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"  {OISST_BOX.name}: {keys[0]} .. {keys[-1]}")


def update_oisst_boxes(redo: int = 3) -> bool:
    """Append the months PSL has published since the series' last month (and re-read the last
    `redo`, which OISST revises while its dailies are preliminary). One month per OPeNDAP read,
    sequential. True when the file changed."""
    import xarray as xr

    guard_fetch("PSL OISST monthly")
    doc = json.loads(OISST_BOX.read_text())
    ds = xr.open_dataset(OISST_OPENDAP)
    times = ds["time"].values
    have = doc["months"]
    start = have[-redo] if len(have) >= redo else have[0]
    by = {m: (a, b) for m, a, b in zip(have, doc["n34"], doc["trop"])}
    changed = False
    for t in times:
        key = str(t)[:7]
        if key < start:
            continue
        sl = ds["sst"].sel(time=[t]).sel(lat=slice(-20, 20))
        sl = sl.load()
        if not np.isfinite(sl.values).any():
            print(f"  OISST {key}: empty read; skipped", flush=True)
            continue
        n34, trop = _box_means(sl)
        val = (float(n34.values[0]), float(trop.values[0]))
        old = by.get(key)
        if old is None or any(o is None or abs(o - v) > 1e-6 for o, v in zip(old, val)):
            by[key] = val
            changed = True
            print(f"  OISST {key}: Nino-3.4 {val[0]:.3f}  20S-20N {val[1]:.3f}", flush=True)
    ds.close()
    if changed:
        ks = sorted(by)
        doc.update(months=ks, n34=[by[k][0] for k in ks], trop=[by[k][1] for k in ks])
        OISST_BOX.write_text(json.dumps(doc, separators=(",", ":")))
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["obs", "hc", "ls", "oisst", "cpc"])
    ap.add_argument("--month", help="MM for `hc`")
    ap.add_argument("--issue", help="YYYYMM for `cpc`: the six observed months before it")
    ap.add_argument("--only", default="", help="places,cpc,oisst for `obs`")
    a = ap.parse_args(argv)
    if a.cmd == "ls":
        for p in sorted(REF.glob("*.npz")):
            print(f"{p.stat().st_size / 1e6:8.2f} MB  {p.name}")
        return 0
    if a.cmd == "oisst":
        update_oisst_boxes()
        return 0
    if a.cmd == "cpc":
        y, mo = int(a.issue[:4]), int(a.issue[4:6])
        years = sorted({y + (mo - k - 1) // 12 for k in range(1, 7)})
        for region in ("us", "br"):
            fetch_cpc(region, years)
        return 0
    if a.cmd == "obs":
        want = set(a.only.split(",")) if a.only else {"places", "cpc", "oisst"}
        if "places" in want:
            derive_places()
        if "cpc" in want:
            derive_cpc_clim()
        if "oisst" in want:
            derive_oisst_boxes(HERE / "data" / "sst.mon.mean.nc")
        return 0
    if a.cmd == "hc":
        from c3s_terciles import derive_hc_tables

        derive_hc_tables(f"{int(a.month):02d}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
