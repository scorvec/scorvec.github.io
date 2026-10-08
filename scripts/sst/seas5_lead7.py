#!/usr/bin/env python3
"""SEAS5 lead month 7: monthly means from the CDS sub-daily product, defined like the monthly product.

WHY. The SEAS5 page reaches the 7th month of each run (April for an October start; user 2026-10-07: "The seas5
forecasts should go out to april"), but the CDS monthly product (seasonal-monthly-single-levels /
-pressure-levels) only offers leadtime_month 1–6. SEAS5 itself runs 215 days (5160 h), so the 7th month is in
the sub-daily product (seasonal-original-single-levels / -pressure-levels). This module rebuilds the monthly
mean of every monthly kind of seas5_outlook.KINDS from it, on the SAME area and grid (the CDS interpolates both
products identically), so lead 7 drops into the arrays of leads 1–6:

  instantaneous fields (t2m, sst, msl, snow depth, z/u/v)   mean of the steps inside the month — STEP_RULE
                                                            "right" = the first step AFTER 00 UTC on the 1st
                                                            (06 UTC; 12 UTC for pressure levels) … 00 UTC on the
                                                            1st of the next month inclusive
  accumulated fields (tp, snowfall, ssrd, evaporation)      (accumulation at the month's end − at its start)
                                                            / seconds in the month = the monthly product's rate
The rule and the units were fixed by the lead-6 test (2026-10-07, run 37629317631): the same reduction applied to
month 6 (March 2027 of the October 2026 issue, 51 members) against the monthly product's own month 6. With the
"right" rule every field matches to GRIB packing precision — max |difference| t2m 0.002 K, SST 0.001 K, z500
0.28 m²/s², u/v850 0.001 m/s, msl 0.18 Pa, z10 0.56 m²/s², u10 hPa 0.002 m/s, snow depth 5e-6 m; the accumulated
rates tp / snowfall / evaporation to 1e-11 m/s (ratio 1.0000). The "left" rule (00 UTC on the 1st … 18 UTC on the
last day) is off by up to 0.35 K in t2m and 430 m²/s² in z10. Not reproducible, so NOT extended: the 10 m wind
SPEED (the monthly product's si10 is 1.2 % above the mean of the 6-hourly speeds from u10/v10, rms 0.18 member σ —
it is computed from the model's own speed field) and with it the `energy` kind (si10, ssrd) stays at 6 leads.

    python seas5_lead7.py forecast --issue 202610           # this issue's month 7 → forecast/fc_{kind}_{ym}_L7.npz
    python seas5_lead7.py check --issue 202610              # month 6 the same way vs the monthly GRIBs (one-time)

Every request is sequential (one CDS key) and runs in GitHub Actions only (seas5_ref.guard_fetch).
"""
from __future__ import annotations

import pyproj  # noqa: F401  first, before eccodes/xarray/cartopy load another libproj
import argparse
import calendar
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seas5_outlook import CENTRE, CLIM_YEARS, DATA, KINDS, SYSTEM, UNUSED_KINDS, fc_path  # noqa: E402

LEAD = 7
RAW = DATA / "l7raw"                      # sub-daily GRIBs: reduced at once, then deleted
STEP_RULE = "right"                       # (H0, H1] — fixed by the lead-6 test (module docstring)
SECONDS_PER_HOUR = 3600.0

# What each monthly kind needs from the sub-daily product: request groups (dataset, CDS variables, levels, cadence).
# cadence: hours between requested steps for instantaneous fields; "acc" = only the month boundaries.
SINGLE, PRESSURE = "seasonal-original-single-levels", "seasonal-original-pressure-levels"
GROUPS = {
    "sst": [(SINGLE, ["sea_surface_temperature"], None, 6)],
    "gl": [(SINGLE, ["2m_temperature"], None, 6), (SINGLE, ["total_precipitation"], None, "acc"),
           (SINGLE, ["sea_surface_temperature"], None, 6)],
    "gl_z500": [(PRESSURE, ["geopotential"], ["500"], 12)],
    "gl_u850": [(PRESSURE, ["u_component_of_wind"], ["850"], 12)],
    "gl_v850": [(PRESSURE, ["v_component_of_wind"], ["850"], 12)],
    "na_snow": [(SINGLE, ["snowfall"], None, "acc")],
    "na_snowdepth": [(SINGLE, ["snow_depth"], None, 6)],
    "polar_n": [(PRESSURE, ["geopotential"], ["10", "50", "100"], 12)],
    "polar_s": [(PRESSURE, ["geopotential"], ["10", "50", "100"], 12)],
    "nh_z": [(PRESSURE, ["geopotential"], ["500", "1000"], 12)],
    "nh_msl": [(SINGLE, ["mean_sea_level_pressure"], None, 6)],
    "strat_u": [(PRESSURE, ["u_component_of_wind"], ["10", "30", "50"], 12)],
    "water": [(SINGLE, ["evaporation"], None, "acc")],
}
SUBSET_OF = {"sfc": "gl"}
# The CDS queue charges per REQUEST, not per byte (2026-10-08: one pressure-level request waited 230 min), so the
# pressure-level kinds that share a grid are fetched together and cut per kind: the polar caps and the NH heights are
# all on the 2° grid (one global z request, 5 levels), the 850 hPa winds both on the global 1° grid (u and v in one).
# MARS interpolates to the target grid and then crops to the area, so a kind's points are the same in either request.
COMBINED = {
    "z2": dict(dataset=PRESSURE, variables=["geopotential"], levels=["10", "50", "100", "500", "1000"], area=[90, -180, -90, 180],
               grid=[2.0, 2.0], cadence=12, kinds={"polar_n": {"z": [10, 50, 100]}, "polar_s": {"z": [10, 50, 100]}, "nh_z": {"z": [500, 1000]}}),
    "uv850": dict(dataset=PRESSURE, variables=["u_component_of_wind", "v_component_of_wind"], levels=["850"], area=[90, -180, -90, 180],
                  grid=[1.0, 1.0], cadence=12, kinds={"gl_u850": {"u": [None, 850]}, "gl_v850": {"v": [None, 850]}}),
}


def _cut(fields: dict, keep: dict, lat_t, lon_t) -> dict:
    """The kind's variables/levels of a combined request, on the kind's own points."""
    out = {}
    for (mvar, lev), (v, lat, lon) in fields.items():
        if mvar not in keep or lev not in keep[mvar]:
            continue
        ii = [int(np.where(np.isclose(lat, x))[0][0]) for x in lat_t]
        lon_n = ((lon + 180) % 360) - 180
        jj = [int(np.where(np.isclose(lon_n, ((x + 180) % 360) - 180))[0][0]) for x in lon_t]
        key = (mvar, lev if len([l for l in keep[mvar] if l is not None]) > 1 else None)
        out[key] = (v[..., ii, :][..., jj], np.asarray(lat_t), np.asarray(lon_t))
    return out                 # the Americas t2m/tp kind is a cut of the global 1° kind (same grid points)

# sub-daily GRIB shortName → (monthly-product variable name as load_field asks for it, kind of reduction)
SHORT = {"2t": ("t2m", "mean"), "sst": ("sst", "mean"), "tp": ("tprate", "rate"), "z": ("z", "mean"),
         "u": ("u", "mean"), "v": ("v", "mean"), "msl": ("msl", "mean"), "sf": ("mtsfr", "rate"),
         "sd": ("sd", "mean"), "ssrd": ("ssrd", "rate"), "e": ("e", "rate")}
# monthly-product units per accumulated variable: value = FACTOR × (Δaccumulation / seconds). Fixed by the lead-6
# test (CHECK): the monthly tprate / mtsfr / e are rates per second; ssrd is given per second too.
FACTOR = {"tprate": 1.0, "mtsfr": 1.0, "e": 1.0, "ssrd": 1.0}


def month_window(ym: str, L: int) -> tuple[int, int]:
    """(H0, H1): lead hours of the start and end of forecast month L (1 = the start month) of a run started on
    the 1st of ym. Exact calendar, so a leap February moves every later boundary by 24 h."""
    y, m = int(ym[:4]), int(ym[4:])
    h = 0
    for j in range(L - 1):
        mm = (m - 1 + j) % 12 + 1; yy = y + (m - 1 + j) // 12
        h += calendar.monthrange(yy, mm)[1] * 24
    mm = (m - 1 + L - 1) % 12 + 1; yy = y + (m - 1 + L - 1) // 12
    return h, h + calendar.monthrange(yy, mm)[1] * 24


def _windows(years: list[str], month: str, leads) -> dict:
    """{year: {L: (H0, H1)}} for every start year."""
    return {y: {L: month_window(f"{y}{month}", L) for L in leads} for y in years}


def _hours(win: dict, cadence) -> list[int]:
    """Every leadtime_hour a request needs to cover all windows (union over years: the hindcast mixes calendars)."""
    hs = set()
    for per in win.values():
        for H0, H1 in per.values():
            if cadence == "acc":
                hs.update([H0, H1])
            else:
                hs.update(range(H0, H1 + 1, int(cadence)))
    return sorted(h for h in hs if h <= 5160)


def _cds(dataset: str, req: dict, dest: Path, label: str) -> bool:
    """One CDS request, sequential, at most 4 tries; a throttled queue backs off for minutes."""
    from seas5_ref import guard_fetch
    if dest.exists() and dest.stat().st_size > 0:
        return True
    guard_fetch(f"CDS {label}")
    from seas5_outlook import _client
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f".part{os.getpid()}")
    for attempt in range(4):
        t0 = time.time()
        try:
            print(f"  CDS {label}: {len(req.get('leadtime_hour', req.get('leadtime_month', [])))} steps × {len(req['year'])} yr → {dest.name} …", flush=True)
            _client().retrieve(dataset, req, str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest)
                dt = (time.time() - t0) / 60
                print(f"    done {dest.stat().st_size / 1e6:.0f} MB in {dt:.1f} min", flush=True)
                _log_volume(label, dest.stat().st_size, dt)
                return True
        except Exception as e:                                          # noqa: BLE001
            msg = str(e).replace("\n", " ")
            print(f"    {label}: attempt {attempt + 1} failed ({msg[:300]})", flush=True)
            if "temporarily limited" in msg.lower() or "queued requests" in msg.lower() or ("404" in msg and "/jobs/" in msg):
                time.sleep(240 * (attempt + 1))
            else:
                time.sleep(30)
    return False


VOLUME_LOG = DATA / "l7_volume.jsonl"


def _log_volume(label: str, nbytes: int, minutes: float) -> None:
    VOLUME_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(VOLUME_LOG, "a") as fh:
        fh.write(json.dumps(dict(label=label, mb=round(nbytes / 1e6, 1), minutes=round(minutes, 2),
                                 at=time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()))) + "\n")


def request(kind: str, gi: int, years: list[str], month: str, leads) -> tuple[str, dict]:
    ds, variables, levels, cadence = GROUPS[kind][gi]
    k = KINDS[kind]
    win = _windows(years, month, leads)
    req = {"originating_centre": CENTRE, "system": SYSTEM, "variable": variables, "year": list(years), "month": [month],
           "day": ["01"], "leadtime_hour": [str(h) for h in _hours(win, cadence)], "area": k["area"], "grid": k["grid"],
           "data_format": "grib"}
    if levels:
        req["pressure_level"] = levels
    return ds, req


def raw_path(kind: str, gi: int, tag: str) -> Path:
    return RAW / f"{tag}_{kind}_g{gi}.grib"


def fetch_group(kind: str, gi: int, years: list[str], month: str, leads, tag: str) -> Path | None:
    ds, req = request(kind, gi, years, month, leads)
    dest = raw_path(kind, gi, tag)
    return dest if _cds(ds, req, dest, f"{tag} {kind} g{gi} leads {list(leads)}") else None


# ── reduction ────────────────────────────────────────────────────────────────
def _open_short(path: Path):
    """{shortName: xarray.DataArray} of a sub-daily GRIB, one hypercube per shortName (cfgrib cannot merge
    variables whose step sets differ)."""
    import xarray as xr
    import cfgrib
    out = {}
    for ds in cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}):
        for name, da in ds.data_vars.items():
            sn = da.attrs.get("GRIB_shortName", name)
            out[sn] = da
    return out


def _sample_major(da):
    """→ (values[sample, step, (level,) lat, lon], hours[step], years[sample] or None, levels or None, lat, lon).
    Samples are member-major with the year fastest, as seas5_build._stack_samples orders the monthly hindcast."""
    dims = list(da.dims)
    lev = "isobaricInhPa" if "isobaricInhPa" in dims else None
    if "step" not in dims:
        da = da.expand_dims("step")
    if "number" not in da.dims:
        da = da.expand_dims("number")
    order = ["number"] + (["time"] if "time" in da.dims else []) + ["step"] + ([lev] if lev and lev in da.dims else []) + ["latitude", "longitude"]
    da = da.transpose(*order).sortby("step")
    hours = (da.step.values / np.timedelta64(1, "h")).astype(int)
    v = da.values.astype(np.float32)
    years = None
    if "time" in da.dims:
        yt = [str(t)[:4] for t in np.atleast_1d(da.time.values)]
        years = [y for _ in range(v.shape[0]) for y in yt]                 # per sample
        v = v.reshape(-1, *v.shape[2:])                                   # (number × time) member-major, year fastest
    levels = [int(x) for x in np.atleast_1d(da[lev].values)] if lev and lev in da.dims else None
    return v, hours, years, levels, da.latitude.values, da.longitude.values


def reduce_month(arr: np.ndarray, hours: np.ndarray, how: str, H0: int, H1: int, rule: str = STEP_RULE) -> np.ndarray:
    """arr [sample, step, ...] → [sample, ...] monthly value for the window [H0, H1)."""
    if how == "rate":
        i0, i1 = list(hours).index(H0), list(hours).index(H1)
        return (arr[:, i1] - arr[:, i0]) / ((H1 - H0) * SECONDS_PER_HOUR)
    sel = (hours >= H0) & (hours < H1) if rule == "left" else (hours > H0) & (hours <= H1)
    if not sel.any():
        raise ValueError(f"no steps inside [{H0}, {H1})")
    return arr[:, sel].mean(axis=1)


def reduce_group(path: Path, years: list[str] | None, month: str, leads, rule: str = STEP_RULE) -> dict:
    """{(monthly var, level|None): (vals[sample, len(leads), lat, lon], lat, lon)} of one raw sub-daily GRIB.
    `years` = the start years of the samples (hindcast) or None (forecast, one year in the request)."""
    out = {}
    fields = _open_short(path)
    speed = None
    for sn, da in fields.items():
        v, hours, yrs, levels, lat, lon = _sample_major(da)
        if sn in ("10u", "u10", "10v", "v10"):
            speed = speed or {}
            speed["u" if "u" in sn else "v"] = (v, hours, yrs, lat, lon)
            continue
        if sn not in SHORT:
            print(f"    {path.name}: unexpected shortName {sn} — ignored", flush=True); continue
        mvar, how = SHORT[sn]
        lev_list = levels or [None]
        for li, lev in enumerate(lev_list):
            vv = v[:, :, li] if levels else v
            out[(mvar, lev)] = (_by_year(vv, hours, yrs, month, leads, how, rule) * (FACTOR.get(mvar, 1.0) if how == "rate" else 1.0), lat, lon)
    if speed and "u" in speed and "v" in speed:
        (u, hours, yrs, lat, lon), (vv, _, _, _, _) = speed["u"], speed["v"]
        out[("si10", None)] = (_by_year(np.hypot(u, vv), hours, yrs, month, leads, "mean", rule), lat, lon)
    return out


def _by_year(v: np.ndarray, hours: np.ndarray, yrs, month: str, leads, how: str, rule: str) -> np.ndarray:
    """Apply each sample's own calendar (its start year) to the month windows → [sample, len(leads), ...]."""
    years_s = yrs if yrs is not None else [None] * v.shape[0]
    res = np.empty((v.shape[0], len(leads)) + v.shape[2:], dtype=np.float32)
    cache = {}
    for s in range(v.shape[0]):
        y = years_s[s] or _FC_YEAR[0]
        if y not in cache:
            cache[y] = [month_window(f"{y}{month}", L) for L in leads]
        for j, (H0, H1) in enumerate(cache[y]):
            res[s, j] = reduce_month(v[s:s + 1], hours, how, H0, H1, rule)[0]
    return res


_FC_YEAR = ["2026"]


# ── forecast supplements ─────────────────────────────────────────────────────
def l7_path(kind: str, ym: str) -> Path:
    return DATA / "forecast" / f"fc_{kind}_{ym}_L7.npz"


def _key(mvar: str, lev) -> str:
    return mvar if lev is None else f"{mvar}_{lev}"


def save_supplement(path: Path, fields: dict, meta: dict) -> None:
    arrays = {}
    for (mvar, lev), (v, lat, lon) in fields.items():
        arrays[_key(mvar, lev)] = v.astype(np.float32)
        arrays["lat"], arrays["lon"] = lat, lon
    arrays["__meta__"] = np.array(json.dumps(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".part{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    print(f"    {path.name}: {', '.join(k for k in arrays if k not in ('lat', 'lon', '__meta__'))} "
          f"({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def _subset(fields: dict, lat_t: np.ndarray, lon_t: np.ndarray, keep: set) -> dict:
    out = {}
    for (mvar, lev), (v, lat, lon) in fields.items():
        if mvar not in keep:
            continue
        ii = [int(np.where(np.isclose(lat, x))[0][0]) for x in lat_t]
        lon_n = ((lon + 180) % 360) - 180
        jj = [int(np.where(np.isclose(lon_n, ((x + 180) % 360) - 180))[0][0]) for x in lon_t]
        out[(mvar, lev)] = (v[..., ii, :][..., jj], np.asarray(lat_t), np.asarray(lon_t))
    return out


def grid_of(kind: str, ym: str):
    """lat/lon of this issue's monthly GRIB of the kind (the target grid), or None."""
    p = fc_path(kind, ym)
    if not p.exists():
        return None
    import xarray as xr
    ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": "", "time_dims": ("forecastMonth", "time")})
    out = (ds.latitude.values, ds.longitude.values); ds.close()
    return out


def forecast(ym: str, lead: int = LEAD, kinds=None, tag: str | None = None, keep_raw: bool = False,
             budget_min: float | None = None) -> dict:
    """This issue's month `lead` for every monthly kind → fc_{kind}_{ym}_L7.npz (or the _m{lead}chk name for the
    test). Returns {kind: ok}. With `budget_min`, no new request starts after that many minutes (the CDS queued
    pressure-level requests 30–90 min each on 2026-10-07); what is left is fetched by the next run."""
    _FC_YEAR[0] = ym[:4]
    t_start = time.time()
    tag = tag or f"fc{ym}m{lead}"
    want = [k for k in KINDS if k not in UNUSED_KINDS and (kinds is None or k in kinds)]
    got = {}
    cache: dict = {}
    dest_of = lambda k: l7_path(k, ym) if lead == LEAD else DATA / "forecast" / f"fc_{k}_{ym}_m{lead}chk.npz"
    combined_done = set()
    for name, c in COMBINED.items():
        todo = [k for k in c["kinds"] if k in want and not dest_of(k).exists()]
        if not todo:
            continue
        if budget_min is not None and (time.time() - t_start) / 60 > budget_min:
            print(f"  {name}: lead-7 fetch budget ({budget_min:.0f} min) spent — left for the next run", flush=True); continue
        win = _windows([ym[:4]], ym[4:], [lead])
        req = {"originating_centre": CENTRE, "system": SYSTEM, "variable": c["variables"], "pressure_level": c["levels"],
               "year": [ym[:4]], "month": [ym[4:]], "day": ["01"], "leadtime_hour": [str(h) for h in _hours(win, c["cadence"])],
               "area": c["area"], "grid": c["grid"], "data_format": "grib"}
        dest = RAW / f"{tag}_{name}.grib"
        if not _cds(c["dataset"], req, dest, f"{tag} {name} (for {', '.join(todo)})"):
            continue
        fields = {k: (v[:, 0], la, lo) for k, (v, la, lo) in reduce_group(dest, None, ym[4:], [lead]).items()}
        for kind in todo:
            g = grid_of(kind, ym)
            if g is None:
                print(f"  {kind}: monthly grid missing — skipped", flush=True); continue
            H0, H1 = month_window(ym, lead)
            save_supplement(dest_of(kind), _cut(fields, c["kinds"][kind], g[0], g[1]),
                            dict(issue=ym, lead=lead, window_h=[H0, H1], rule=STEP_RULE, request=name,
                                 source="CDS seasonal-original-pressure-levels (sub-daily), reduced to the monthly-product definition"))
            combined_done.add(kind)
        if not keep_raw:
            dest.unlink()
    for kind in [k for k in want if k in GROUPS] + [k for k in want if k in SUBSET_OF]:
        if any(kind in c["kinds"] for c in COMBINED.values()):
            got[kind] = dest_of(kind).exists(); continue
        dest = l7_path(kind, ym) if lead == LEAD else DATA / "forecast" / f"fc_{kind}_{ym}_m{lead}chk.npz"
        if dest.exists():
            got[kind] = True; continue
        if budget_min is not None and kind not in SUBSET_OF and (time.time() - t_start) / 60 > budget_min:
            print(f"  {kind}: lead-7 fetch budget ({budget_min:.0f} min) spent — left for the next run", flush=True)
            got[kind] = False; continue
        if kind in SUBSET_OF:
            src = SUBSET_OF[kind]
            g = grid_of(kind, ym)
            sp = l7_path(src, ym) if lead == LEAD else DATA / "forecast" / f"fc_{src}_{ym}_m{lead}chk.npz"
            if src not in cache and sp.exists():
                with np.load(sp) as z:
                    cache[src] = {(k, None): (z[k], z["lat"], z["lon"]) for k in z.files if k not in ("lat", "lon", "__meta__")}
            if src not in cache or g is None:
                print(f"  {kind}: source {src} or the monthly grid missing — skipped", flush=True); got[kind] = False; continue
            fields = _subset(cache[src], g[0], g[1], {"t2m", "tprate"})
        else:
            fields = {}
            ok = True
            for gi in range(len(GROUPS[kind])):
                p = fetch_group(kind, gi, [ym[:4]], ym[4:], [lead], tag)
                if p is None:
                    ok = False; break
                fields.update({k: (v[:, 0], la, lo) for k, (v, la, lo) in reduce_group(p, None, ym[4:], [lead]).items()})
                if not keep_raw:
                    p.unlink()
            if not ok:
                got[kind] = False; continue
            cache[kind] = fields
        H0, H1 = month_window(ym, lead)
        save_supplement(dest, fields, dict(issue=ym, lead=lead, window_h=[H0, H1], rule=STEP_RULE,
                                           source="CDS seasonal-original-* (sub-daily), reduced to the monthly-product definition"))
        got[kind] = True
    return got


# ── the lead-6 test ──────────────────────────────────────────────────────────
def check(ym: str, out: Path | None = None) -> dict:
    """Month 6 from the sub-daily product (both step rules) against the monthly product's own month 6, per kind and
    variable: mean / max |difference| and the difference relative to the member spread. The verdict fixes STEP_RULE
    and FACTOR."""
    from seas5_build import load_field
    _FC_YEAR[0] = ym[:4]
    res = {}
    for kind in [k for k in GROUPS]:
        if not fc_path(kind, ym).exists():
            print(f"  check {kind}: monthly GRIB not on disk — skipped", flush=True); continue
        fields_by_rule = {}
        for gi in range(len(GROUPS[kind])):
            p = fetch_group(kind, gi, [ym[:4]], ym[4:], [6], f"chk{ym}m6")
            if p is None:
                continue
            for rule in ("left", "right"):
                for k, (v, la, lo) in reduce_group(p, None, ym[4:], [6], rule).items():
                    fields_by_rule.setdefault(rule, {})[k] = (v[:, 0], la, lo)
            p.unlink()
        for (mvar, lev), (vl, lat, lon) in fields_by_rule.get("left", {}).items():
            try:
                mon, mlat, mlon = load_field(fc_path(kind, ym), mvar, **({"level": lev} if lev is not None else {}), lead7=False)
            except Exception as e:                                       # noqa: BLE001
                print(f"  check {kind} {mvar}: monthly read failed ({e})", flush=True); continue
            m6 = mon[:, 5]
            assert m6.shape == vl.shape, (m6.shape, vl.shape)
            rec = {}
            for rule in ("left", "right"):
                d = fields_by_rule[rule][(mvar, lev)][0]
                diff = d - m6
                ok = np.isfinite(diff)
                sd = np.nanstd(m6, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = float(np.nanmedian(d[ok] / m6[ok])) if np.any(m6[ok] != 0) else float("nan")
                rec[rule] = dict(mean_diff=float(np.nanmean(diff)), mean_abs=float(np.nanmean(np.abs(diff))),
                                 p99_abs=float(np.nanpercentile(np.abs(diff), 99)), max_abs=float(np.nanmax(np.abs(diff))),
                                 rms_over_member_sd=float(np.sqrt(np.nanmean(diff ** 2)) / np.nanmean(sd)),
                                 median_ratio=ratio, n=int(ok.sum()))
            rec["monthly_mean"] = float(np.nanmean(m6)); rec["monthly_member_sd"] = float(np.nanmean(np.nanstd(m6, axis=0)))
            res[f"{kind}:{_key(mvar, lev)}"] = rec
            print(f"  check {kind} {_key(mvar, lev)}: " + " | ".join(
                f"{r}: mean {rec[r]['mean_diff']:+.4g}, mean|d| {rec[r]['mean_abs']:.4g}, p99 {rec[r]['p99_abs']:.4g}, "
                f"max {rec[r]['max_abs']:.4g}, rms/sd {rec[r]['rms_over_member_sd']:.3f}, ratio {rec[r]['median_ratio']:.5g}"
                for r in ("left", "right")) + f" (monthly mean {rec['monthly_mean']:.4g}, member sd {rec['monthly_member_sd']:.3g})", flush=True)
    if out:
        out.write_text(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["forecast", "check", "missing"])
    ap.add_argument("--issue", required=True)
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--budget-min", type=float, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "forecast":
        got = forecast(a.issue, kinds=a.kinds, budget_min=a.budget_min)
        bad = [k for k, v in got.items() if not v]
        print(f"lead-7 supplements {a.issue}: {len(got) - len(bad)} ok" + (f", missing {bad}" if bad else ""), flush=True)
        return 0 if not bad else 1
    if a.cmd == "missing":
        from seas5_outlook import nleads
        miss = [k for k in list(GROUPS) + list(SUBSET_OF) if nleads(a.issue) >= LEAD and fc_path(k, a.issue).exists() and not l7_path(k, a.issue).exists()]
        print(" ".join(miss))
        return 1 if miss else 0
    check(a.issue, Path(a.out) if a.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
