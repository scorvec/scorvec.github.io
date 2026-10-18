#!/usr/bin/env python3
"""CanSIPS straight from ECCC's MSC Datamart: the ONE fetcher, plus the 12-month Niño-3.4 plume.

C3S carries CanSIPS for six months; ECCC's own Datamart publishes the same 40-member forecast to
twelve, with the full 1990-2020 hindcast beside it. This module fetches and screens those GRIBs
(cansips_maps.py imports the same functions for the seasonal-page maps) and builds the ENSO
forecasts page's CanSIPS entry, assets/sst/data/cansips_dd.json.

Datamart layout (verified 2026-10-07):
  https://dd.weather.gc.ca/YYYYMMDD/WXO-DD/model_cansips/100km/{forecast,hindcast}/YYYY/MM/
  YYYYMMDD is a ~30-day rolling window of date prefixes, every one of which serves the SAME static
  CanSIPS tree (forecasts from 2023, hindcasts 1990-2020), so today's prefix is used.
  {YYYYMM}_MSC_CanSIPS[-Hindcast]_{Var}_LatLon1.0_P{nn}M.grib2, P00M..P11M; P00M IS the issue month
  (dataDate = YYYYMM01, a one-month mean ending on the 1st of the next month). The issue is posted
  on the last day of the previous month (October 2026: 2026-09-30 15:22 UTC).

GRIB facts, measured from the files (`cansips_dd.py probe`):
  * 40 messages per file, perturbationNumber 1..40, PDT 4.11, 1 deg cell-centred grid
    (-89.5..89.5, 0.5..359.5), units K for WaterTemp.
  * Members 1-20 and 21-40 are the two component models. ECCC's Datamart readme: 1-10 GEM5.2-NEMO
    initialised on the last day of the month, 11-20 GEM5.2-NEMO four days earlier, 21-30 CanESM5
    last day, 31-40 CanESM5 four days earlier. The hindcast has the same 40 members, 1990-2020.
  * NO BITMAP: WaterTemp is filled over land (Amazon 297 K, Sahara 298 K), so every ocean average
    here applies an explicit ocean mask (ocean_mask(), from global-land-mask) — without it the
    20S-20N mean of the relative index would include land.

Indices exactly as c3s_nino34.py builds them for the page's other systems: Niño-3.4 = cos-weighted
mean over the one-degree points 5S-5N, 170W-120W (sampled as the CDS samples them, see indices());
the tropical mean over the 5-degree ocean points 20S-20N; each COMPONENT anomalised against its own 1993-2016 hindcast for the same start
month and lead, then pooled; rNiño-3.4 = (n34 - trop) anomaly x sigma(ONI)/sigma(rel) for the valid
calendar month (roni_sigma.json).

  python cansips_dd.py build [--issue YYYYMM] [--force]   # newest posted issue (+ the C3S issue)
  python cansips_dd.py probe

Data Source: Environment and Climate Change Canada (MSC Datamart; ECCC Data Servers End-use Licence,
https://eccc-msc.github.io/open-data/licence/readme_en/).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import functools
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE.parents[1]
CACHE = HERE / "data" / "cansips_dd"  # gitignored (scripts/sst/data/)
OUT_JSON = SITE_ROOT / "assets" / "sst" / "data" / "cansips_dd.json"
C3S_JSON = SITE_ROOT / "assets" / "sst" / "data" / "enso_forecast.json"
CLIM_CSV = HERE / "cansips_dd_clim.csv"  # committed: per component, start month, lead
SIGMA_PATH = HERE / "roni_sigma.json"

DD = "https://dd.weather.gc.ca/{day}/WXO-DD/model_cansips/100km"
UA = {"User-Agent": "scorvec.com seasonal pages (CanSIPS monthly, a few hundred files a month)"}
ATTRIBUTION = "Data Source: Environment and Climate Change Canada"
LICENCE = "https://eccc-msc.github.io/open-data/licence/readme_en/"
NLEAD = 12
NMEM = 40
CLIM_YEARS = list(range(1993, 2017))  # the page's common hindcast period (C3S)
COMPONENTS = {"GEM5.2-NEMO": slice(0, 20), "CanESM5": slice(20, 40)}  # ECCC Datamart readme
KEEP_ISSUES = 6

# Datamart variable names, and the physical range a screened member mean must fall in (file units)
VARS = {
    "sst": ("WaterTemp_Sfc", (265.0, 312.0)),
    "t2m": ("AirTemp_AGL-2m", (180.0, 335.0)),
    "t850": ("AirTemp_ISBL-0850", (190.0, 320.0)),
    "z500": ("GeopotentialHeight_ISBL-0500", (4400.0, 6200.0)),
    "prate": ("PrecipRate_Sfc", (-1e-6, 1.0)),  # units checked separately
    "mslp": ("Pressure_MSL", (90000.0, 110000.0)),
    "ssh": ("SeaSfcHeight-Geoid", (-5.0, 5.0)),
    "u200": ("WindU_ISBL-0200", (-120.0, 160.0)),
    "v200": ("WindV_ISBL-0200", (-120.0, 120.0)),
    "u850": ("WindU_ISBL-0850", (-80.0, 80.0)),
    "v850": ("WindV_ISBL-0850", (-80.0, 80.0)),
}


# ── Datamart access ──────────────────────────────────────────────────────────
def dd_root() -> str:
    return DD.format(day=time.strftime("%Y%m%d", time.gmtime()))


def _session():
    s = requests.Session()
    s.headers.update(UA)
    return s


@functools.cache
def http():
    return _session()


def listing(url: str) -> dict:
    """Apache index -> {name: (mtime 'YYYY-MM-DD HH:MM', size string)}; {} if the directory is absent."""
    for k in range(4):
        try:
            r = http().get(url, timeout=60)
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            out = {}
            for m in re.finditer(r'href="([^"?/][^"]*)">[^<]*</a>\s+(\d{4}-\d\d-\d\d \d\d:\d\d)\s+(\S+)', r.text):
                out[m.group(1)] = (m.group(2), m.group(3))
            return out
        except Exception as e:  # noqa: BLE001
            print(f"  listing {url}: attempt {k + 1} {str(e)[:80]}", file=sys.stderr)
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"Datamart listing failed: {url}")


def fc_name(var: str, ym: str, lead: int) -> str:
    return f"{ym}_MSC_CanSIPS_{VARS[var][0]}_LatLon1.0_P{lead:02d}M.grib2"


def hc_name(var: str, ym: str, lead: int) -> str:
    return f"{ym}_MSC_CanSIPS-Hindcast_{VARS[var][0]}_LatLon1.0_P{lead:02d}M.grib2"


def fc_dir(ym: str) -> str:
    return f"{dd_root()}/forecast/{ym[:4]}/{ym[4:]}/"


def hc_dir(ym: str) -> str:
    return f"{dd_root()}/hindcast/{ym[:4]}/{ym[4:]}/"


def fetch(url: str, dest: Path, tries: int = 8) -> Path | None:
    """GET to dest (atomic). None on 404. A GRIB that does not end in '7777' is a truncated transfer."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            r = http().get(url, timeout=180)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            b = r.content
            if not (b[:4] == b"GRIB" and b[-4:] == b"7777"):
                raise OSError(f"truncated GRIB ({len(b)} bytes)")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(b)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"  GET {url.rsplit('/', 1)[-1]} attempt {k + 1}: {str(e)[:90]}", file=sys.stderr)
            time.sleep(min(90, 8 * 2**k))  # the Datamart drops connections under load: back off hard
    raise RuntimeError(f"Datamart download failed after {tries} tries: {url}")


def fetch_many(jobs: list[tuple[str, Path]], workers: int = 4) -> list[Path | None]:
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: fetch(*j), jobs))


def fc_path(var, ym, lead):
    return CACHE / "fc" / ym / fc_name(var, ym, lead)


def hc_path(var, ym, lead):
    return CACHE / "hc" / ym[4:] / hc_name(var, ym, lead)


def issue_posted(ym: str, var: str = "sst") -> dict | None:
    """{lead: mtime} when all 12 lead files of `var` are listed for issue ym, else None."""
    L = listing(fc_dir(ym))
    got = {}
    for lead in range(NLEAD):
        n = fc_name(var, ym, lead)
        if n not in L:
            return None
        got[lead] = L[n][0]
    return got


def latest_issue() -> str | None:
    """Newest issue with all 12 WaterTemp leads posted: next month (posted on the last day of this
    one), else this month, else last month."""
    t = time.gmtime()
    y, m = t.tm_year, t.tm_mon
    cands = []
    for d in (1, 0, -1):
        mm = m + d
        yy = y + (mm - 1) // 12
        mm = (mm - 1) % 12 + 1
        cands.append(f"{yy}{mm:02d}")
    for ym in cands:
        if issue_posted(ym):
            return ym
    return None


def add_months(ym: str, n: int) -> str:
    y, m = int(ym[:4]), int(ym[4:]) - 1 + n
    return f"{y + m // 12}{m % 12 + 1:02d}"


# ── GRIB reading + screens ───────────────────────────────────────────────────
class DataError(RuntimeError):
    pass


def read_members(path: Path, var: str, ym: str, lead: int, nmem: int = NMEM) -> tuple[np.ndarray, dict]:
    """(members[nmem, 180, 360] float32 in file units, ordered by perturbationNumber, meta).

    Screens (strict-data-quality): message count, member numbers exactly 1..nmem, one dataDate equal to
    the issue's first day, validity at the end of the right month, the expected grid, no missing values,
    no duplicated member, and a physically plausible range."""
    import eccodes

    rows, meta = {}, {}
    with open(path, "rb") as fh:
        while True:
            h = eccodes.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:

                def g(k, h=h):
                    return eccodes.codes_get(h, k)

                pn = int(g("perturbationNumber"))
                if pn in rows:
                    raise DataError(f"{path.name}: member {pn} appears twice")
                if (
                    (g("Ni"), g("Nj")) != (360, 180)
                    or abs(g("latitudeOfFirstGridPointInDegrees") + 89.5) > 1e-6
                    or abs(g("longitudeOfFirstGridPointInDegrees") - 0.5) > 1e-6
                    or g("jScansPositively") != 1
                ):
                    raise DataError(f"{path.name}: unexpected grid")
                dd, vd = str(g("dataDate")), str(g("validityDate"))
                meta.setdefault("dataDate", set()).add(dd)
                meta.setdefault("validityDate", set()).add(vd)
                meta["units"] = g("units")
                meta["shortName"] = g("shortName")
                v = eccodes.codes_get_values(h).astype(np.float32)
                if g("bitmapPresent"):
                    v[v == g("missingValue")] = np.nan
                rows[pn] = v.reshape(180, 360)
            finally:
                eccodes.codes_release(h)
    if sorted(rows) != list(range(1, nmem + 1)):
        raise DataError(f"{path.name}: members {sorted(rows)[:3]}..{sorted(rows)[-3:]} ({len(rows)}), want 1..{nmem}")
    if meta["dataDate"] != {f"{ym}01"}:
        raise DataError(f"{path.name}: dataDate {meta['dataDate']} for issue {ym}")
    want_v = add_months(ym, lead + 1) + "01"
    if meta["validityDate"] != {want_v}:
        raise DataError(f"{path.name}: validity {meta['validityDate']}, expected {want_v} (end of lead {lead})")
    arr = np.stack([rows[k] for k in range(1, nmem + 1)])
    if not np.isfinite(arr).all():
        raise DataError(f"{path.name}: {int((~np.isfinite(arr)).sum())} missing values")
    lo, hi = VARS[var][1]
    mm = arr.mean(0)
    if mm.min() < lo or mm.max() > hi:
        raise DataError(f"{path.name}: member mean {mm.min():.4g}..{mm.max():.4g} outside {lo}..{hi} {meta['units']}")
    flat = arr.reshape(nmem, -1)[:, ::97]
    for i in range(nmem):  # a copied member is a feed defect
        if np.any(np.all(flat[i + 1 :] == flat[i], axis=1)):
            raise DataError(f"{path.name}: member {i + 1} duplicated")
    return arr, meta


LAT = np.arange(-89.5, 90, 1.0)
LON = np.arange(0.5, 360, 1.0)


def ocean_mask() -> np.ndarray:
    """Boolean (180, 360): True where at least half of the 1-degree cell is ocean (global-land-mask,
    sampled 10 x 10 per cell). Inland lakes count as land, as in the C3S fields."""
    p = CACHE / "ocean_mask_1deg.npy"
    if p.exists():
        return np.load(p)
    from global_land_mask import globe

    sub = (np.arange(10) + 0.5) / 10.0 - 0.5
    la = (LAT[:, None] + sub[None, :]).ravel()
    lo = (LON[:, None] + sub[None, :]).ravel()
    lo = np.where(lo > 180, lo - 360, lo)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    oc = globe.is_ocean(LA, LO).reshape(180, 10, 360, 10).mean(axis=(1, 3))
    m = oc >= 0.5
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, m)
    return m


def _box_weights(lat_rng, lon_rng=None) -> np.ndarray:
    la = (LAT > lat_rng[0]) & (LAT < lat_rng[1])  # noqa: SIM300  coordinate array LAT, not a constant
    lo = np.ones(LON.size, bool) if lon_rng is None else (LON > lon_rng[0]) & (LON < lon_rng[1])  # noqa: SIM300  coordinate array LON, not a constant
    w = np.cos(np.deg2rad(LAT))[:, None] * (la[:, None] & lo[None, :]) * ocean_mask()
    return w / w.sum()


def _point_weights(lats, lons, need_ocean: bool) -> np.ndarray:
    """(180, 360) weights that reproduce sampling at integer-degree POINTS, as the CDS request does: each point is
    bilinearly interpolated from the cell-centred grid, i.e. the mean of its four neighbouring cells, and points are
    cos-weighted. need_ocean: a point counts only when all four cells are ocean (the land fill never enters)."""
    oc = ocean_mask()
    w = np.zeros((180, 360))
    for la in lats:
        i0 = round(la + 89.0)  # the cell centred at la - 0.5
        for lo in lons:
            j0 = round(lo - 1.0) % 360  # the cell centred at lo - 0.5
            cells = [(i0, j0), (i0 + 1, j0), (i0, (j0 + 1) % 360), (i0 + 1, (j0 + 1) % 360)]
            if need_ocean and not all(oc[c] for c in cells):
                continue
            for c in cells:
                w[c] += 0.25 * np.cos(np.deg2rad(la))
    return w / w.sum()


def indices(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-member (Niño-3.4, 20S-20N ocean mean) in degC from a WaterTemp member stack.

    Sampled exactly as c3s_nino34.py's CDS requests sample the C3S systems: Niño-3.4 on the 11 x 51 one-degree POINTS
    5S..5N, 170W..120W inclusive, the tropical mean on the 5-degree points 20S..20N. Measured on the September 2026
    issue (`checkbox`): the cell box (the 10 x 50 cells strictly inside the region) reads 0.09-0.15 C warmer than
    C3S's copy of the same 40 runs, because a strong El Nino's anomaly is concentrated on the equator and the point
    box reaches half a degree further off it; the point box agrees with C3S to 0.02 C."""
    w34, wtr = _index_weights()
    a = arr.astype(np.float64) - 273.15
    return (a * w34).sum(axis=(1, 2)), (a * wtr).sum(axis=(1, 2))


@functools.cache
def _index_weights() -> tuple[np.ndarray, np.ndarray]:
    """(Niño-3.4 point weights, 20S-20N ocean point weights), built once on first use."""
    return (
        _point_weights(range(-5, 6), range(190, 241), need_ocean=False),
        _point_weights(range(-20, 21, 5), range(0, 360, 5), need_ocean=True),
    )


# ── hindcast climatology of the indices ──────────────────────────────────────
def _load_clim() -> dict:
    out = {}
    if CLIM_CSV.exists():
        for ln in CLIM_CSV.read_text().splitlines()[1:]:
            c, mo, lead, n34, tr, ny, nm = ln.split(",")
            out[(c, int(mo), int(lead))] = (float(n34), float(tr), int(ny), int(nm))
    return out


def _save_clim(d: dict):
    rows = ["component,month,lead,clim_n34,clim_trop,n_years,n_members"]
    for (c, mo, lead), (n, t, ny, nm) in sorted(d.items()):
        rows.append(f"{c},{mo},{lead},{n:.4f},{t:.4f},{ny},{nm}")
    tmp = CLIM_CSV.with_suffix(".tmp")
    tmp.write_text("\n".join(rows) + "\n")
    tmp.replace(CLIM_CSV)


def index_climatology(month: int) -> dict:
    """{component: (clim_n34[12], clim_trop[12])} for a start month, cached in the committed CSV.
    Downloads 24 years x 12 leads of the WaterTemp hindcast (~1 GB) the first time a month comes round."""
    d = _load_clim()
    if all((c, month, L) in d for c in COMPONENTS for L in range(1, NLEAD + 1)):
        return {
            c: (
                np.array([d[(c, month, L)][0] for L in range(1, NLEAD + 1)]),
                np.array([d[(c, month, L)][1] for L in range(1, NLEAD + 1)]),
            )
            for c in COMPONENTS
        }
    print(f"  index hindcast climatology, start month {month:02d}, {CLIM_YEARS[0]}-{CLIM_YEARS[-1]} ...", flush=True)
    jobs = []
    for y in CLIM_YEARS:
        ym = f"{y}{month:02d}"
        jobs.extend((hc_dir(ym) + hc_name("sst", ym, lead), hc_path("sst", ym, lead)) for lead in range(NLEAD))
    paths = fetch_many(jobs)
    if any(p is None for p in paths):
        miss = [j[0].rsplit("/", 1)[-1] for j, p in zip(jobs, paths) if p is None]
        raise DataError(f"hindcast incomplete for start month {month:02d}: {len(miss)} missing, e.g. {miss[:3]}")
    acc = {c: {L: ([], []) for L in range(NLEAD)} for c in COMPONENTS}
    for y in CLIM_YEARS:
        ym = f"{y}{month:02d}"
        for lead in range(NLEAD):
            arr, _ = read_members(hc_path("sst", ym, lead), "sst", ym, lead)
            n34, tr = indices(arr)
            for c, sl in COMPONENTS.items():
                acc[c][lead][0].append(n34[sl])
                acc[c][lead][1].append(tr[sl])
    for c in COMPONENTS:
        for lead in range(NLEAD):
            n = np.concatenate(acc[c][lead][0])
            t = np.concatenate(acc[c][lead][1])
            d[(c, month, lead + 1)] = (float(n.mean()), float(t.mean()), len(CLIM_YEARS), n.size // len(CLIM_YEARS))
    _save_clim(d)
    return index_climatology(month)


# ── the forecast ─────────────────────────────────────────────────────────────
def roni_scale() -> dict:
    tab = json.loads(SIGMA_PATH.read_text()).get("scale_by_month", {})
    return {int(k): float(v) for k, v in tab.items()}


def build_issue(ym: str) -> dict:
    posted = issue_posted(ym)
    if not posted:
        raise DataError(f"issue {ym}: not all 12 WaterTemp leads are posted")
    month = int(ym[4:])
    clim = index_climatology(month)
    jobs = [(fc_dir(ym) + fc_name("sst", ym, L), fc_path("sst", ym, L)) for L in range(NLEAD)]
    fetch_many(jobs)
    scale = roni_scale()
    n34_mem, rn_mem, valid, comp_means = [], [], [], {c: {"n34": [], "rnino": []} for c in COMPONENTS}
    for L in range(NLEAD):
        arr, _meta = read_members(fc_path("sst", ym, L), "sst", ym, L)
        n34, tr = indices(arr)
        vm = add_months(ym, L)
        valid.append(f"{vm[:4]}-{vm[4:]}")
        an, ar = np.empty(NMEM), np.empty(NMEM)
        for c, sl in COMPONENTS.items():
            cn, ct = clim[c][0][L], clim[c][1][L]
            an[sl] = n34[sl] - cn
            ar[sl] = ((n34[sl] - tr[sl]) - (cn - ct)) * scale.get(int(vm[4:]), 1.0)
            comp_means[c]["n34"].append(round(float(an[sl].mean()), 3))
            comp_means[c]["rnino"].append(round(float(ar[sl].mean()), 3))
        if np.abs(an).max() > 8:
            raise DataError(f"{ym} lead {L}: Niño-3.4 anomaly {np.abs(an).max():.1f} °C is not physical")
        n34_mem.append([round(float(v), 3) for v in an])
        rn_mem.append([round(float(v), 3) for v in ar])
    nm = np.array(n34_mem)
    rm = np.array(rn_mem)
    print(
        f"  CanSIPS {ym}: {NMEM} members, Niño-3.4 mean "
        + " ".join(f"{v:+.2f}" for v in nm.mean(1))
        + "\n"
        + " " * 14
        + "rNiño-3.4 mean "
        + " ".join(f"{v:+.2f}" for v in rm.mean(1)),
        flush=True,
    )
    return {
        "issue": f"{ym[:4]}-{ym[4:]}",
        "valid_months": valid,
        "posted_utc": posted[0],
        "files_mtime": posted,
        "members": NMEM,
        "components": {c: {"members": f"{sl.start + 1}-{sl.stop}", **comp_means[c]} for c, sl in COMPONENTS.items()},
        "n34": n34_mem,
        "rnino": rn_mem,
    }


def validate(entry: dict) -> dict | None:
    """Months 1-6 against the C3S CanSIPSv3 entry of the same issue, if the page holds one."""
    if not C3S_JSON.exists():
        return None
    F = json.loads(C3S_JSON.read_text())
    M = F.get("models", {}).get("ECCC CanSIPSv3")
    if F.get("issue") != entry["issue"] or not M:
        return None
    out = {"c3s_issue": F["issue"], "leads": len(M["n34"])}
    for k in ("n34", "rnino"):
        c = np.array([np.mean(r) for r in M[k]])
        d = np.array([np.mean(r) for r in entry[k][: len(c)]])
        cs = np.array([np.std(r) for r in M[k]])
        ds = np.array([np.std(r) for r in entry[k][: len(c)]])
        out[k] = {
            "c3s_mean": [round(float(v), 3) for v in c],
            "datamart_mean": [round(float(v), 3) for v in d],
            "diff": [round(float(v), 3) for v in d - c],
            "max_abs_diff": round(float(np.abs(d - c).max()), 3),
            "c3s_sd": [round(float(v), 3) for v in cs],
            "datamart_sd": [round(float(v), 3) for v in ds],
            "c3s_members": len(M[k][0]),
        }
        # the C3S entry concatenates its two component systems, 20 + 20, in c3s_nino34.MERGE order
        if len(M[k][0]) == 40:
            out[k]["c3s_first20_mean"] = [round(float(np.mean(r[:20])), 3) for r in M[k]]
            out[k]["c3s_last20_mean"] = [round(float(np.mean(r[20:])), 3) for r in M[k]]
    print(
        f"  validation vs C3S {F['issue']}: Niño-3.4 diff {out['n34']['diff']}  rNiño diff {out['rnino']['diff']}",
        flush=True,
    )
    return out


def build(issue: str | None, force: bool) -> int:
    doc = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {"issues": {}}
    latest = issue or latest_issue()
    if not latest:
        print("::warning::no complete CanSIPS issue on the Datamart (files late?); keeping the published one")
        return 0
    want = [latest]
    if not issue and C3S_JSON.exists():  # the page's C3S issue too, for its months 7-12 extension
        ci = json.loads(C3S_JSON.read_text()).get("issue", "").replace("-", "")
        if ci and ci != latest and ci not in want:
            want.append(ci)
    changed = False
    for ym in want:
        key = f"{ym[:4]}-{ym[4:]}"
        have = doc["issues"].get(key)
        if have and not force:
            try:
                same = issue_posted(ym) == {int(k): v for k, v in have.get("files_mtime", {}).items()}
            except Exception:  # noqa: BLE001
                same = True
            if same:
                print(f"  {key}: already built from the same files")
                if "validation" not in have:
                    v = validate(have)
                    if v:
                        have["validation"] = v
                        changed = True
                continue
        if not issue_posted(ym):
            print(f"  {key}: not (fully) posted on the Datamart; skipped")
            continue
        e = build_issue(ym)
        v = validate(e)
        if v:
            e["validation"] = v
        doc["issues"][key] = e
        changed = True
    if not changed:
        print("nothing new")
        return 0
    keys = sorted(doc["issues"])[-KEEP_ISSUES:]
    doc["issues"] = {k: doc["issues"][k] for k in keys}
    doc.update(
        {
            "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "latest": keys[-1],
            "system": "ECCC CanSIPS (GEM5.2-NEMO + CanESM5, 40 members)",
            "hindcast": f"{CLIM_YEARS[0]}-{CLIM_YEARS[-1]}, each component model against its own hindcast for the same start "
            "month and lead (40 members x 24 years)",
            "indices": "Niño-3.4 5S-5N 170W-120W; relative = minus the 20S-20N ocean mean, sigma-scaled per calendar month "
            "(roni_sigma.json), as for the C3S systems",
            "source": "ECCC MSC Datamart, model_cansips/100km WaterTemp_Sfc, P00M-P11M",
            "attribution": ATTRIBUTION,
            "licence": LICENCE,
        }
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    tmp.replace(OUT_JSON)
    print(f"wrote {OUT_JSON.relative_to(SITE_ROOT)} ({OUT_JSON.stat().st_size // 1024} KB; issues {', '.join(keys)})")
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as f:
            f.write("changed=true\n")
    return 0


# ── why the direct copy and C3S's differ: the box (validation only) ──────────
def n34_points(arr: np.ndarray) -> np.ndarray:
    """Niño-3.4 the way the CDS request samples it: 11 x 51 points at integer degrees (5S..5N, 190..240E), each
    bilinearly interpolated from the cell-centred grid (= the mean of its four neighbouring cells), cos-weighted."""
    a = arr.astype(np.float64) - 273.15
    la = np.arange(-5, 6)
    lo = np.arange(190, 241)
    i0 = (la + 89.5 - 0.5).astype(int)
    j0 = (lo - 0.5 - 0.5).astype(int)  # cell below / left of each point
    v = 0.25 * (a[:, i0][:, :, j0] + a[:, i0 + 1][:, :, j0] + a[:, i0][:, :, j0 + 1] + a[:, i0 + 1][:, :, j0 + 1])
    w = np.cos(np.deg2rad(la))[:, None] * np.ones(lo.size)[None, :]
    return (v * w).sum(axis=(1, 2)) / w.sum()


def checkbox(ym: str) -> int:
    """Re-derive the Niño-3.4 anomaly with the C3S point box and compare both boxes with the C3S CanSIPS entry."""
    month = int(ym[4:])
    jobs = [
        (hc_dir(f"{y}{month:02d}") + hc_name("sst", f"{y}{month:02d}", L), hc_path("sst", f"{y}{month:02d}", L))
        for y in CLIM_YEARS
        for L in range(6)
    ]
    jobs += [(fc_dir(ym) + fc_name("sst", ym, L), fc_path("sst", ym, L)) for L in range(6)]
    fetch_many(jobs)
    F = json.loads(C3S_JSON.read_text())
    M = F["models"]["ECCC CanSIPSv3"]["n34"]
    for L in range(6):
        cells, pts = {c: [] for c in COMPONENTS}, {c: [] for c in COMPONENTS}
        for y in CLIM_YEARS:
            arr, _ = read_members(hc_path("sst", f"{y}{month:02d}", L), "sst", f"{y}{month:02d}", L)
            n_c, _ = indices(arr)
            n_p = n34_points(arr)
            for c, sl in COMPONENTS.items():
                cells[c].append(n_c[sl])
                pts[c].append(n_p[sl])
        arr, _ = read_members(fc_path("sst", ym, L), "sst", ym, L)
        f_c, _ = indices(arr)
        f_p = n34_points(arr)
        an_c, an_p = np.empty(NMEM), np.empty(NMEM)
        for c, sl in COMPONENTS.items():
            an_c[sl] = f_c[sl] - np.concatenate(cells[c]).mean()
            an_p[sl] = f_p[sl] - np.concatenate(pts[c]).mean()
        c3 = np.array(M[L])  # C3S order: eccc_4 (= Datamart 21-40) then eccc_5 (= Datamart 1-20)
        dd_c = np.concatenate([an_c[20:], an_c[:20]])
        dd_p = np.concatenate([an_p[20:], an_p[:20]])
        print(
            f"lead {L + 1}: C3S mean {c3.mean():+.3f} | cell box {dd_c.mean():+.3f} (diff {dd_c.mean() - c3.mean():+.3f}, "
            f"member rms {np.sqrt(((dd_c - c3) ** 2).mean()):.3f}) | point box {dd_p.mean():+.3f} (diff {dd_p.mean() - c3.mean():+.3f}, "
            f"member rms {np.sqrt(((dd_p - c3) ** 2).mean()):.3f}) | sd C3S {c3.std():.3f} cell {dd_c.std():.3f} point {dd_p.std():.3f}",
            flush=True,
        )
    return 0


# ── probe (kept: re-run it if ECCC changes the encoding) ─────────────────────
def probe() -> int:
    import eccodes

    ym = latest_issue()
    print("Datamart root:", dd_root(), "latest issue:", ym)
    p = fetch(fc_dir(ym) + fc_name("sst", ym, 0), CACHE / "probe" / fc_name("sst", ym, 0))
    with open(p, "rb") as fh:
        h = eccodes.codes_grib_new_from_file(fh)
        for k in (
            "centre",
            "productDefinitionTemplateNumber",
            "perturbationNumber",
            "numberOfForecastsInEnsemble",
            "dataDate",
            "stepRange",
            "validityDate",
            "shortName",
            "units",
            "bitmapPresent",
        ):
            print(f"  {k:36s} {eccodes.codes_get(h, k)}")
        eccodes.codes_release(h)
    arr, _meta = read_members(p, "sst", ym, 0)
    n34, tr = indices(arr)
    print(f"  screened: {arr.shape}, Niño-3.4 {n34.mean():.2f} C, 20S-20N ocean {tr.mean():.2f} C")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "probe", "checkbox"])
    ap.add_argument("--issue", help="YYYYMM (default: newest posted)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "probe":
        return probe()
    if a.cmd == "checkbox":
        return checkbox(a.issue or "202609")
    return build(a.issue, a.force)


if __name__ == "__main__":
    raise SystemExit(main())
