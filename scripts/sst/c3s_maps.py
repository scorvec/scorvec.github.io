#!/usr/bin/env python3
"""C3S multi-model seasonal anomaly maps — every centre, one selectable model at a time, each tested.

seas5_outlook.py pulls SEAS5's 51 members globally and caches 17 GB per start month; doing that for all six C3S
centres would be ~100 GB and days of queue. The maps need far less: C3S publishes `seasonal-postprocessed-single-
levels`, "Seasonal forecast anomalies on single levels" — every MEMBER's anomaly already taken against its own model's
1993-2016 hindcast, which is exactly the quantity a model-vs-model comparison wants, and no forecast hindcast to
download. The map is their ensemble mean.

Whether that mean is more than noise is a test (c3s_maps_render, seas5_build.shift_signal): the members' spread
around it, and the hindcast's own spread -- one member's period mean over the 24 hindcast years, which only the raw
hindcast (`seasonal-monthly-single-levels`, 1993-2016, the start month, 1 deg) has. The tercile maps fetch that raw
hindcast for 2 m temperature and precipitation anyway (c3s_terciles.py, same file, reduced for both before it goes);
for sea-level pressure and SST it is fetched for this test alone, once per start month.

Everything is reduced to a table the moment it lands (c3s_ref) and the raw GRIB deleted in Actions:
  is_anom_{centre}_{system}_{field}_{YYYYMM}   ensemble-mean anomaly per lead, the members' spread per period
  hc_anom_{centre}_{system}_{field}_{MM}       the hindcast's single-member spread per period
Periods are c3s_maps_render.PERIODS, day-weighted means of the lead months (a leap February counts 29 days; a
hindcast member is weighted by the calendar of its own nominal start year).

    python scripts/sst/c3s_maps.py --issue 202610 --budget-min 120   # Actions: members, then the hindcasts
    python scripts/sst/c3s_maps.py --issue 202610 --only ecmwf_51
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray (libproj double-free trap)
import argparse
import calendar
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c3s_nino34 import MODELS, _client  # one model registry, one CDS client
import c3s_ref as CR
import c3s_terciles as CT

DATASET = "seasonal-postprocessed-single-levels"
CACHE = HERE / "data" / "c3s" / "maps"
LEADS = ["1", "2", "3", "4", "5", "6"]
# field -> (postprocessed anomaly name, factor to display units). The raw hindcast of each field: c3s_terciles.RAW_VARS.
FIELDS = {
    "t2m": ("2m_temperature_anomaly", 1.0),
    "tp": ("total_precipitation_anomalous_rate_of_accumulation", 86400.0 * 1000.0),
    "mslp": ("mean_sea_level_pressure_anomaly", 0.01),
    "sst": ("sea_surface_temperature_anomaly", 1.0),
}


def path_for(key: str, centre: str, system: str, issue: str) -> Path:
    return CACHE / issue / f"{centre}_{system}_{key}.grib"


def fetch(key: str, centre: str, system: str, issue: str, force: bool = False) -> Path | None:
    """This issue's member anomalies of one field (product_type monthly_mean: every member, leads 1-6)."""
    dest = path_for(key, centre, system, issue)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = {
        "originating_centre": centre,
        "system": str(system),
        "variable": [FIELDS[key][0]],
        "product_type": ["monthly_mean"],
        "year": [issue[:4]],
        "month": [issue[4:6]],
        "leadtime_month": LEADS,
        "data_format": "grib",
    }
    tmp = dest.with_suffix(".part")
    try:
        _client().retrieve(DATASET, req, str(tmp))
        tmp.replace(dest)
        print(f"  {centre}/{system} {key}: {dest.stat().st_size / 1e6:.1f} MB", flush=True)
        return dest
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        print(f"  {centre}/{system} {key}: unavailable ({msg[:110]})", file=sys.stderr, flush=True)
        tmp.unlink(missing_ok=True)
        return None


# ── reading members: message by message (c3s_grib), never cfgrib's hypercube ────
def lead_weight(year: int, month: int, idx: list[int], i: int) -> float:
    """Weight of lead index i (0 = the start month) in the day-weighted mean of the leads idx of a run nominally
    started in (year, month): the valid month's length over the period's, in that start's own calendar."""

    def days(j: int) -> int:
        return calendar.monthrange(year + (month - 1 + j) // 12, (month - 1 + j) % 12 + 1)[1]

    return days(i) / sum(days(j) for j in idx)


def member_stats(path: Path, short: str | None, fac: float, month: int, periods: dict, ensemble_mean: bool) -> dict:
    """From a GRIB of members (a forecast) or of members x years (a hindcast), in display units:
      mean       [lead, lat, lon] ensemble mean (only with `ensemble_mean`)
      sd_{pid}   standard deviation over the samples of their day-weighted period mean
      n, ny      samples, distinct nominal start years
    A sample is one member of one start (number, start date and time); only samples holding every lead of the file
    count, and a period only where the file has all its leads. Read one message at a time (c3s_grib): a lagged system
    (NCEP: 120 members, each its own start) costs one running period sum per sample, never the padded hypercube. A
    sample is weighted by the calendar of its NOMINAL start (indexingDate), so a January member started in December
    counts with the year it forecasts. `short` picks one variable of a multi-variable file (the SEAS5 page's own:
    t2m, tprate, sst); None, or a name the file does not have, reads a single-variable file as its one variable."""
    import c3s_grib

    keys = ("cfVarName", "number", "dataDate", "dataTime", "forecastMonth", "indexingDate")

    def read(name: str | None):
        sums, lead, leads, year, names, grid = {}, {}, {}, {}, set(), None
        for m, field, g in c3s_grib.messages(path, keys):
            names.add(m["cfVarName"])
            if (name is not None and m["cfVarName"] != name) or not np.isfinite(field).any():
                continue
            grid, i = g, m["forecastMonth"] - 1
            k = (m["number"], m["dataDate"], m["dataTime"])
            y = (m["indexingDate"] or m["dataDate"]) // 10000
            leads.setdefault(k, set()).add(i)
            year[k] = y
            v = field.astype(np.float32)
            if ensemble_mean:
                lead.setdefault(i, {})[k] = v
            for pid, ix in periods.items():
                if i in ix:
                    s = sums.setdefault(pid, {})
                    w = np.float32(lead_weight(y, month, ix, i))
                    s[k] = s[k] + w * v if k in s else w * v
        return sums, lead, leads, year, names, grid

    sums, lead, leads, year, names, grid = read(short)
    if not leads and len(names) == 1:
        sums, lead, leads, year, names, grid = read(next(iter(names)))
    if not leads:
        raise ValueError(f"{path.name}: no {short or 'field'} messages")
    have = set().union(*leads.values())
    samples = sorted(k for k, s in leads.items() if s == have)
    lat_o, lon_o = np.argsort(grid.lat), np.argsort(grid.lon)  # latitude ascending, longitude 0..360 ascending

    def stack(d: dict) -> np.ndarray:
        return (np.stack([d.pop(k) for k in samples]) * np.float32(fac))[:, lat_o][:, :, lon_o]

    out = {"lat": grid.lat[lat_o], "lon": grid.lon[lon_o], "n": len(samples), "periods": []}
    out["ny"] = len({year[k] for k in samples})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells (land in SST) stay NaN
        for pid, ix in periods.items():
            if set(ix) <= have:
                out[f"sd_{pid}"] = np.nanstd(stack(sums.pop(pid)), axis=0, ddof=1).astype(np.float32)
                out["periods"].append(pid)
        if ensemble_mean:
            nl = max(have) + 1
            mean = np.full((nl, lat_o.size, lon_o.size), np.nan, np.float32)
            for i in sorted(have):
                mean[i] = np.nanmean(stack(lead.pop(i)), axis=0)
            out["mean"] = mean
    return out


# ── the tables ───────────────────────────────────────────────────────────────
def _drop(p: Path) -> None:
    if CR.discard_raw() and p.exists():
        p.unlink()


def issue_anom(centre: str, system: str, key: str, issue: str) -> dict | None:
    """is_anom table of one system and field (derived from this issue's member GRIB when only that is on disk)."""
    from c3s_maps_render import PERIODS

    t = CR.load(CR.is_anom(centre, system, key, issue))
    if t is not None:
        return t
    p = path_for(key, centre, system, issue)
    if not p.exists():
        return None
    try:
        s = member_stats(p, None, FIELDS[key][1], int(issue[4:6]), PERIODS, ensemble_mean=True)
    except Exception as e:  # noqa: BLE001
        print(f"  {centre}/{system} {key}: unreadable ({str(e)[:90]})", file=sys.stderr, flush=True)
        return None
    if s["n"] < 2:
        print(f"  {centre}/{system} {key}: {s['n']} member(s), no spread; not kept", file=sys.stderr, flush=True)
        return None
    arrs = {k: v for k, v in s.items() if k.startswith("sd_")}
    CR.save(
        CR.is_anom(centre, system, key, issue),
        meta={"nmem": s["n"], "periods": s["periods"]},
        mean=s["mean"],
        lat=s["lat"],
        lon=s["lon"],
        **arrs,
    )
    _drop(p)
    return CR.load(CR.is_anom(centre, system, key, issue))


def hindcast_spread(centre: str, system: str, key: str, issue: str) -> dict | None:
    """hc_anom table of one system, field and start month (derived from the raw hindcast when only that is on disk;
    the raw file itself is c3s_terciles', which deletes it)."""
    from c3s_maps_render import PERIODS

    month = issue[4:6]
    t = CR.load(CR.hc_anom(centre, system, key, month))
    if t is not None:
        return t
    hp = CT.path_for("hc", centre, system, key, issue)
    if not hp.exists():
        return None
    short, fac = CT.RAW_VARS[key][1], CT.RAW_VARS[key][2]
    t0 = time.time()
    try:
        s = member_stats(hp, short, fac, int(month), PERIODS, ensemble_mean=False)
    except Exception as e:  # noqa: BLE001
        print(f"  {centre}/{system} {key} hc: unreadable ({str(e)[:90]})", file=sys.stderr, flush=True)
        return None
    if s["ny"] < 20:
        print(f"  {centre}/{system} {key} hc: {s['ny']} hindcast years; not kept", file=sys.stderr, flush=True)
        return None
    arrs = {k: v for k, v in s.items() if k.startswith("sd_")}
    CR.save(
        CR.hc_anom(centre, system, key, month),
        meta={"nsamp": s["n"], "years": s["ny"], "periods": s["periods"]},
        lat=s["lat"],
        lon=s["lon"],
        **arrs,
    )
    print(f"  {centre}/{system} {key} hc: spread table in {time.time() - t0:.0f}s", flush=True)
    return CR.load(CR.hc_anom(centre, system, key, month))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()), help="YYYYMM, default this month")
    ap.add_argument("--only", default="", help="centre_system, e.g. ecmwf_51")
    ap.add_argument("--fields", default="t2m,tp,mslp,sst")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--budget-min",
        type=float,
        default=0.0,
        help="stop STARTING hindcast downloads after this many minutes (0 = no limit)",
    )
    a = ap.parse_args()
    want = [k for k in a.fields.split(",") if k in FIELDS]
    systems = [(e[0], str(e[1])) for e in MODELS if not a.only or f"{e[0]}_{e[1]}" == a.only]
    t_start = time.time()
    got = 0
    # 1. this issue's members, every system and field (small: one request each)
    for centre, system in systems:
        for key in want:
            if a.force:
                CR.is_anom(centre, system, key, a.issue).unlink(missing_ok=True)
            if issue_anom(centre, system, key, a.issue) is None and fetch(key, centre, system, a.issue, a.force):
                issue_anom(centre, system, key, a.issue)
            got += CR.is_anom(centre, system, key, a.issue).exists()
    want_n = len(want) * len(systems)
    print(f"{got}/{want_n} member table(s) for {a.issue} under {CR.REF}", flush=True)
    # 2. the hindcast spread each map is tested against: once per system, field and start month, ever. A hard stop
    # (SIGINT from the workflow's timeout) ends the fetching, not the run: the members above are what gates the render
    try:
        for centre, system in systems:
            for key in want:
                if hindcast_spread(centre, system, key, a.issue) is not None:
                    continue
                if a.budget_min > 0 and (time.time() - t_start) / 60 >= a.budget_min:
                    print(f"  {centre}/{system} {key} hc: budget spent; next run", flush=True)
                    continue
                if CT.fetch("hc", centre, system, key, a.issue):
                    hindcast_spread(centre, system, key, a.issue)
                    if key in CT.VARS:  # the same raw hindcast makes the tercile table too: never fetched twice
                        CT.hc_table(centre, system, key, a.issue)
                    CT._drop_raw(CT.path_for("hc", centre, system, key, a.issue))
    except KeyboardInterrupt:
        print("  hindcast fetching stopped (time limit); the next run continues", flush=True)
    tested = sum(CR.hc_anom(c, s, k, a.issue[4:6]).exists() for c, s in systems for k in want)
    print(f"{tested}/{want_n} hindcast spread table(s) for start month {a.issue[4:6]}", flush=True)
    # a half-published issue must not reach the renderer: last month's complete set is better than this month's two
    # systems
    return 0 if got >= 0.75 * want_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
