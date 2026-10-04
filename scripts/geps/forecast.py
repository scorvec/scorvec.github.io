#!/usr/bin/env python3
"""Fetch the live GEPS extended cycle and turn it into de-drifted anomalies.

The anomaly is taken against the GEPS8 hindcast climatology at the SAME lead
(build_clim.py), not against a reanalysis. That is the whole point: GEPS drifts
with lead, and differencing against ERA5 would fold that drift into what looks
like signal.

Three joins have to be got right, and each of them is a silent-failure risk:

  units   The hindcast `pr` is a RATE in kg m-2 s-1; the live product is APCP,
          accumulated mm since init. They are differenced here into a daily rate
          and converted, so both sides mean the same thing. (An unconverted
          comparison would be wrong by ~10^5 and still plot.)
  grid    Hindcast is 1 deg, live GEPS is 0.5 deg, and the grids are exactly
          nested (clim on integer degrees, forecast on half degrees). Earlier this
          interpolated the CLIMATOLOGY up to 0.5 deg to keep forecast detail, which
          was wrong: the forecast resolves terrain and coastlines the 1 deg
          reference cannot represent, so that structure did not cancel and came
          out as grid-scale speckle over every mountain range and shoreline —
          mean state masquerading as anomaly. The difference is now taken on the
          climatology's OWN grid, with the forecast band-limited to it first, and
          the resulting anomaly interpolated back to 0.5 deg for output. An anomaly
          cannot carry more resolution than the reference it is measured against.
  calendar Climatology is indexed by the day-of-year of the START, at 5-day
          centres — build_clim.py bins on S, so cell (doy=c, L) is the mean over
          runs INITIALISED near doy c, valid at c+L. It must therefore be looked
          up at the forecast's INIT day-of-year, the same for every lead; the
          valid-date seasonal cycle is already carried by the lead axis. Reading
          it at the valid date instead silently compares a day-35 forecast against
          runs started 35 days later, which over autumn North America made the
          reference 8 K too cold and produced a fake warm anomaly growing with
          lead. Fixed 2026-08-29.

Extended (day 35) cycles run Monday and Thursday only; other days stop at day 16.

Downloads (2026-10-04, moved to Actions): every file comes through datamart.py (honest User-Agent, retries, ledger)
and is read ONCE. The same GRIBs also feed what other products used to fetch again for themselves:

  u850/u200/olr  the per-member 15S-15N band means of the three RMM channels (live/geps_rmm_members_<date>.npz),
                 which mjo_geps.py read by re-downloading the same 316 files
  zg500 24-h     the raw GRIB is kept in the vortex GRIB cache (strat_maps.DL) for the 500 hPa level of
                 forecast_drip.py, which fetched the same 35 steps
  mslp 6/12/18 h the 2.5 deg ensemble mean of the cycle's first day goes to the observed-tail cache that
                 telecon_geps.analysis_daily otherwise refetched

    python forecast.py                 # latest extended cycle
    python forecast.py --date 20260827
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datamart                                                   # noqa: E402
import paths                                                      # noqa: E402
import refpack                                                    # noqa: E402

CLIM = paths.CLIM
OUT = paths.LIVE
URL = datamart.URL.replace("{c}", "00")
# the vortex products' GRIB cache (scripts/strat/strat_maps.DL); forecast_drip reads 500 hPa from it
STRAT_DL = Path(os.environ.get("STRAT_DL", paths.REPO / "scripts" / "strat" / "data" / "vortex_dl"))

# hindcast tag -> (live GEPS variable name, needs de-accumulation)
VARS = {
    "zg500": ("HGT_ISBL_0500", False),
    "t2m":   ("TMP_TGL_2m",    False),
    # OLR arrives ACCUMULATED (J m-2 since init), not as a flux: day 1 read
    # 1.245e7, i.e. 144 W/m2 x 86400 s. Differencing the 24 h endpoints and
    # dividing by 86400 recovers the daily-mean W/m2 the hindcast stores --
    # the same arithmetic as pr, so it shares the de-accumulation branch.
    "olr":   ("OLR_NTAT_0",    True),
    "u850":  ("UGRD_ISBL_0850", False),
    "u200":  ("UGRD_ISBL_0200", False),
    "v850":  ("VGRD_ISBL_0850", False),
    "v200":  ("VGRD_ISBL_0200", False),
    "pr":    ("APCP_SFC_0",    True),      # accumulated mm -> daily rate
    "mslp":  ("PRMSL_MSL_0",   False),
}
# The hindcast is a DAILY MEAN with L at the centre of the interval (0.5, 1.5, …),
# so forecast day n must be a daily mean too, and maps to clim L = n - 0.5.
# Comparing an instantaneous 00Z field against a daily-mean climatology would be
# badly wrong for t2m in particular — 00Z is a night-time sample over the Americas.
# GEPS publishes 6-hourly steps all the way to day 35, so day n averages the four
# steps 24n-18, 24n-12, 24n-6, 24n.
DAYS = list(range(1, 36))
SUBSTEPS = (-18, -12, -6, 0)


def latest_extended(explicit=None) -> str | None:
    """Most recent cycle that actually reaches day 35 (Mon/Thu). HEAD requests only."""
    if explicit:
        return explicit
    day = pd.Timestamp.utcnow().tz_localize(None).normalize()
    for back in range(0, 8):
        d = (day - pd.Timedelta(days=back)).strftime("%Y%m%d")
        if datamart.exists(URL.format(d=d, v="HGT_ISBL_0500", L=840)):
            return d
    return None


# NCEP/NCAR R1 grid: the teleconnection loading patterns (telecon) are defined
# on it, so member fields are coarsened straight onto it. 5 x 5 box mean of the
# 0.5 deg field, then the 2.5 deg points — the box removes what the coarse grid
# cannot carry rather than aliasing it.
LAT25 = np.arange(90.0, -90.1, -2.5)
LON25 = np.arange(0.0, 360.0, 2.5)


# member grid per field: 2.5 deg (NCEP) for the teleconnection fields, 1 deg
# for the probability maps (2.5 deg cells read as blocks on a regional map)
MEMBER_STEP = {"t2m": 1.0, "pr": 1.0}
# RMM channels: per-member 15S-15N band means on the Wheeler-Hendon 2.5 deg longitudes (mjo_geps.member_bands format)
RMM_TAGS = ("u850", "u200", "olr")
RMM_LAT = (-15.0, 15.0)


def coarsen_to(a, step):
    n = int(round(step / 0.5))
    lat = np.arange(90.0, -90.0 - 1e-6, -step)
    lon = np.arange(0.0, 360.0, step)
    a = a.rolling(latitude=n, center=True, min_periods=1).mean()
    a = a.rolling(longitude=n, center=True, min_periods=1).mean()
    a = a.sel(latitude=lat, longitude=lon, method="nearest")
    return a.assign_coords(latitude=lat, longitude=lon)


def coarsen25(a):
    return coarsen_to(a, 2.5)


def band_members(v, lat, lo, lon=LON25):
    """(member, lat, lon) -> cos-weighted 15S-15N mean on the 2.5 deg EOF grid, (member, nlon).
    Exactly mjo_geps._band_members (moved here so the members are reduced from the GRIBs this script already holds)."""
    w = np.where((lat >= RMM_LAT[0]) & (lat <= RMM_LAT[1]), np.cos(np.deg2rad(lat)), 0.0)
    b = (v * w[None, :, None]).sum(1) / w.sum()                 # (member, nlon)
    o = np.argsort(lo % 360)
    lo_s = (lo % 360)[o]
    b = b[:, o]
    lo_w = np.r_[lo_s, 360.0]
    b_w = np.concatenate([b, b[:, :1]], axis=1)
    return np.stack([np.interp(lon, lo_w, row) for row in b_w])


def _grab(date: str, var: str, L: int, keep: bool = False, step: float = 2.5, band: bool = False,
          save_as: Path | None = None, ana_tag: str | None = None):
    """Ensemble-mean field for one variable and lead, or None.

    keep=True also returns every member on the `step` grid, as (mean, members[number, lat, lon]); the members are
    what the teleconnection indices and tercile maps are computed from. band=True returns (mean, band) with the
    per-member RMM band means instead. save_as keeps the raw GRIB there (shared GRIB cache); ana_tag writes the
    2.5 deg ensemble mean to the observed-tail cache as <ana_tag>_P<L>_<date>.npz."""
    # mkstemp hands back an OPEN fd as well as a path. Keeping only the
    # path leaks that fd on every call: unlink() drops the name, not the
    # descriptor. At 35 days x 4 substeps = 140 grabs per field it blew
    # through launchd's 256-fd limit partway into the second field and the
    # 2026-08-31 cycle died as "Too many open files" (plus DNS failures,
    # which were sockets that could no longer be opened either).
    none = (None, None) if (keep or band) else None
    _fd, _name = tempfile.mkstemp(suffix=".grib2", dir=os.environ.get("GEPS_TMP"))
    os.close(_fd)
    tmp = Path(_name)
    try:
        if save_as is not None and save_as.exists() and save_as.stat().st_size > 0:
            src = save_as                                       # already cached this run: no second download
        else:
            body = datamart.get(URL.format(d=date, v=var, L=L))
            if body is None:
                print(f"    miss {var} P{L:03d}", flush=True)
                return none
            tmp.write_bytes(body)
            src = tmp
            if save_as is not None:
                save_as.parent.mkdir(parents=True, exist_ok=True)
                save_as.write_bytes(body)
        # allmbrs files hold the control AND the perturbed members; opening them
        # unfiltered raises "multiple values for unique key" and drops the model
        parts, mem = [], []
        for dt in ("pf", "cf"):
            try:
                d = xr.open_dataset(src, engine="cfgrib", backend_kwargs=dict(
                    filter_by_keys={"dataType": dt}, indexpath=""))
                a = d[[v for v in d.data_vars][0]]
                n = a.sizes["number"] if "number" in a.dims else 1
                parts.append((a.mean("number") if "number" in a.dims else a, n))
                if keep or band:
                    mem.append(a if "number" in a.dims else a.expand_dims(number=[0]))
            except Exception:                                  # noqa: BLE001
                pass
        if not parts:
            return none
        # Weight by MEMBER COUNT, not by file. pf holds 20 perturbed members and cf
        # a single control; averaging the pf mean and the cf as two equal things
        # gave the control 1/2 the weight instead of 1/21, which put a single
        # member's small-scale noise into every "ensemble mean" map (errors to
        # 12 K, rms 1.7 K at day 21). Fixed 2026-08-29.
        tot = sum(n for _, n in parts)
        out = sum(a * n for a, n in parts) / tot
        out = out.load()
        if ana_tag is not None:
            paths.ANA.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(paths.ANA / f"{ana_tag}_P{L:03d}_{date}.npz",
                                v=coarsen25(out).values.astype("float32"))
        if keep or band:
            m = xr.concat([x.drop_vars([c for c in x.coords if c not in
                                        ("latitude", "longitude", "number")])
                           for x in mem], dim="number")
            m = m.assign_coords(number=np.arange(m.sizes["number"]))
            if band:
                return out, band_members(m.values, m.latitude.values, m.longitude.values)
            return out, coarsen_to(m, step).load().astype("float32")
        return out
    except Exception as e:                                     # noqa: BLE001
        print(f"    miss {var} P{L:03d}: {str(e)[:60]}", flush=True)
        return none
    finally:
        tmp.unlink(missing_ok=True)


def fetch_var(date: str, tag: str, workers: int, keep: bool = False):
    """Ensemble-mean daily field (L, lat, lon); with keep=True a tuple of that
    and the member daily fields (L, number, lat, lon) on the member grid. For the RMM channels the second element is
    the per-member band means (L, number, nlon) instead."""
    step = MEMBER_STEP.get(tag, 2.5)
    var, accum = VARS[tag]
    band = tag in RMM_TAGS
    if accum:
        # APCP is accumulated from init: the day-n amount is the difference of the
        # endpoints, so only the 24-hour steps are needed.
        want = sorted({24 * d for d in DAYS} | {24})
    else:
        want = sorted({24 * d + s for d in DAYS for s in SUBSTEPS})

    def grab(L):
        save = (STRAT_DL / f"geps_{date}00_HGT500_{L:03d}.grib2") if (tag == "zg500" and L % 24 == 0) else None
        ana = tag if (tag == "mslp" and L in (6, 12, 18)) else None
        return _grab(date, var, L, keep, step, band=band, save_as=save, ana_tag=ana)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        raw = dict(zip(want, ex.map(grab, want)))
    if keep or band:
        got = {L: a for L, (a, _) in raw.items() if a is not None}
        gotm = {L: m for L, (_, m) in raw.items() if m is not None}
    else:
        got = {L: a for L, a in raw.items() if a is not None}
        gotm = {}
    if not got:
        return (None, (None, None)) if band else (None, None) if keep else None

    def daily_series(g):
        fields, leads = [], []
        for d in DAYS:
            if accum:
                end, start = g.get(24 * d), g.get(24 * (d - 1))
                if end is None:
                    continue
                daily = end - start if start is not None else end
                fields.append(daily / 86400.0)   # pr: mm/day -> kg m-2 s-1; olr: J m-2 -> W m-2
            else:
                have = [g[24 * d + s] for s in SUBSTEPS if (24 * d + s) in g]
                if not have:
                    continue
                fields.append(xr.concat(have, dim="__t").mean("__t"))
            leads.append(d)
        if not fields:
            return None
        arr = xr.concat(fields, dim="L").assign_coords(L=leads).sortby("L")
        if accum:
            arr.attrs["units"] = "W m-2" if tag == "olr" else "kg m-2 s-1"
        arr.attrs["cell_methods"] = ("time: mean over the 24 h ending at the lead "
                                     "(matches the hindcast daily mean)")
        return arr

    def daily_bands(g):
        """mjo_geps.member_bands' daily reduction of the band means: (lead, member, nlon), leads."""
        rows, leads = [], []
        for d in DAYS:
            if accum:
                end, start = g.get(24 * d), g.get(24 * (d - 1))
                if end is None:
                    continue
                rows.append(((end - start) if start is not None else end) / 86400.0)
            else:
                have = [g[24 * d + s] for s in SUBSTEPS if (24 * d + s) in g]
                if not have:
                    continue
                rows.append(np.mean(have, axis=0))
            leads.append(d)
        return (np.stack(rows), np.array(leads)) if rows else (None, None)

    arr = daily_series(got)
    if band:
        return arr, (daily_bands(gotm) if gotm else (None, None))
    if keep:
        marr = daily_series(gotm) if gotm else None
        return arr, (marr.transpose("L", "number", "latitude", "longitude")
                     if marr is not None else None)
    return arr


def to_clim_grid(fc, lat: str, lon: str, ylat, xlon):
    """Band-limit the 0.5 deg forecast to the 1 deg climatology grid.

    A plain subsample would alias the half-degree detail down onto the coarse
    grid; a separable [1,2,1]/4 filter removes it first. Longitude wraps, latitude
    uses edge values at the poles. The grids are point-registered and exactly
    nested, so after filtering the 1 deg points are a straight selection."""
    a = fc
    for dim, periodic in ((lat, False), (lon, True)):
        lo = a.shift({dim: 1})
        hi = a.shift({dim: -1})
        if periodic:
            lo = xr.where(lo.isnull(), a.roll({dim: 1}, roll_coords=False), lo)
            hi = xr.where(hi.isnull(), a.roll({dim: -1}, roll_coords=False), hi)
        else:
            lo = lo.fillna(a); hi = hi.fillna(a)
        a = 0.25 * lo + 0.5 * a + 0.25 * hi
    return a.sel({lat: ylat, lon: xlon}, method="nearest").assign_coords(
        {lat: ylat, lon: xlon})


def anomaly(tag: str, fc, base: pd.Timestamp):
    """forecast - climatology(day-of-year of the VALID date, same lead)."""
    p = refpack.pack_path(tag)
    cv = refpack.clim(tag)
    if cv is None:
        print(f"    no climatology for {tag}"); return None
    # wrap the day-of-year axis so late-December interpolates into January
    cw = xr.concat([cv.isel(doy=-1).assign_coords(doy=cv.doy[-1] - 366),
                    cv,
                    cv.isel(doy=0).assign_coords(doy=cv.doy[0] + 366)], dim="doy")
    lat = "latitude" if "latitude" in fc.dims else "lat"
    lon = "longitude" if "longitude" in fc.dims else "lon"
    out = []
    # one day-of-year for every lead: the INIT's, because the climatology is
    # binned by start date (see the calendar note above)
    init_doy = base.dayofyear
    ylat, xlon = cv["Y"].values, cv["X"].values
    coarse = to_clim_grid(fc.assign_coords({lon: fc[lon].values % 360}).sortby(lon),
                          lat, lon, ylat, xlon)
    for L in fc.L.values:
        ref = (cw.sel(L=float(L) - 0.5, method="nearest").interp(doy=init_doy)
                 .rename({"Y": lat, "X": lon}))
        out.append(coarse.sel(L=L) - ref.values)
    a = xr.concat(out, dim="L").assign_coords(L=fc.L.values)
    # back to the forecast grid so downstream code and the maps are unchanged;
    # the field is genuinely 1 deg, this is interpolation for rendering only.
    # Longitude must WRAP: the climatology stops at 359 deg E and the forecast grid
    # runs to 359.5, so a plain interp left a NaN column there — invisible on the
    # Americas maps but fatal to any global weighted mean, since 0 * NaN is NaN.
    a = xr.concat([a, a.isel({lon: 0}).assign_coords({lon: 360.0})], dim=lon)
    a = a.interp({lat: fc[lat].values, lon: np.sort(fc[lon].values % 360)})
    if bool(np.isnan(a.values).any()):
        raise SystemExit(f"{tag}: NaN survived the regrid — check the wrap")
    a.attrs["anomaly_reference"] = f"GEPS8 hindcast climatology 2001-2020, lead-matched ({p.stem})"

    # The hindcast climatology is centred on ~2010; the forecast is 2026. For
    # geopotential height that gap is a real climate signal, measured here as
    # +17 m at day 1 rising to +34 m by day 35 — a uniform ridge that would sit
    # over every map and hide the circulation. Record the area-weighted global
    # mean per lead so a consumer can subtract it (the standard treatment for
    # height anomalies) while keeping the raw anomaly for fields like t2m where
    # the warming IS part of what you want to see.
    w = np.cos(np.deg2rad(a[lat].values))
    gm = ((a * xr.DataArray(w, dims=lat)).sum((lat, lon))
          / (w.sum() * a.sizes[lon]))
    a = a.assign_coords(global_mean=("L", gm.values))
    a.attrs["global_mean_note"] = (
        "coord `global_mean` is the area-weighted global mean per lead: the "
        "offset between the 2001-2020 hindcast climate and today. Subtract it "
        "to see circulation rather than trend.")
    return a


def anomaly_coarse(tag: str, mem, base: pd.Timestamp):
    """Member anomalies: forecast - lead-matched hindcast climatology, 2.5 deg.

    Same reference and same calendar rule as anomaly(); the 1 deg climatology is
    interpolated to the 2.5 deg member grid (a smooth field, nothing lost)."""
    p = refpack.pack_path(tag)
    cv = refpack.clim(tag)
    if cv is None:
        return None
    cw = xr.concat([cv.isel(doy=-1).assign_coords(doy=cv.doy[-1] - 366),
                    cv,
                    cv.isel(doy=0).assign_coords(doy=cv.doy[0] + 366)], dim="doy")
    init_doy = base.dayofyear
    out = []
    for L in mem.L.values:
        ref = (cw.sel(L=float(L) - 0.5, method="nearest").interp(doy=init_doy)
                 .rename({"Y": "latitude", "X": "longitude"}))
        ref = xr.concat([ref, ref.isel(longitude=0).assign_coords(longitude=360.0)],
                        dim="longitude")
        ref = ref.interp(latitude=mem.latitude.values, longitude=mem.longitude.values)
        out.append(mem.sel(L=L) - ref.values[None])
    a = xr.concat(out, dim="L").assign_coords(L=mem.L.values).astype("float32")
    if bool(np.isnan(a.values).any()):
        raise SystemExit(f"{tag}: NaN in the member anomaly")
    a.attrs["anomaly_reference"] = f"GEPS8 hindcast climatology 2001-2020, lead-matched ({p.stem})"
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--members", default="zg500,mslp,t2m,pr",
                    help="tags whose 21 members are also kept, at 2.5 deg, for the "
                         "teleconnection indices (geps_members_{tag}_{date}.nc)")
    ap.add_argument("--tags", default=",".join(VARS))
    ap.add_argument("--workers", type=int, default=datamart.WORKERS)
    a = ap.parse_args()
    date = latest_extended(a.date)
    if not date:
        print("no extended (day-35) cycle found in the last week"); return 1
    base = pd.Timestamp(f"{date} 00:00")
    print(f"cycle {base:%Y-%m-%d %a} 00Z  (extended, day 35)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    keep_tags = {t.strip() for t in a.members.split(",") if t.strip()}
    rmm = {}
    written = 0
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        keep = tag in keep_tags
        fc = fetch_var(date, tag, a.workers, keep=keep)
        if tag in RMM_TAGS:
            fc, (bands, bL) = fc
            if bands is not None:
                rmm[tag] = (bands, bL)
        elif keep:
            fc, mem = fc
        if fc is None:
            print(f"  {tag}: no data"); continue
        an = anomaly(tag, fc, base)
        if an is None:
            continue
        if keep and mem is not None:
            am = anomaly_coarse(tag, mem, base)
            if am is not None:
                dm = xr.Dataset({f"{tag}_fc": mem, f"{tag}_anom": am})
                dm.attrs = {"cycle": date, "source": "ECCC GEPS raw, all members (cf + 20 pf)",
                            "grid": f"{MEMBER_STEP.get(tag, 2.5)} deg, box mean of the 0.5 deg field",
                            "reference": am.attrs.get("anomaly_reference", "")}
                dst = OUT / f"geps_members_{tag}_{date}.nc"
                dm.to_netcdf(dst, encoding={k: {"zlib": True, "complevel": 4} for k in dm.data_vars})
                print(f"  {tag}: {mem.sizes['number']} members x {mem.sizes['L']} days -> {dst.name}",
                      flush=True)
        ds = xr.Dataset({f"{tag}_fc": fc, f"{tag}_anom": an})
        ds.attrs = {"cycle": date, "source": "ECCC GEPS raw, ensemble mean of cf+pf",
                    "reference": an.attrs.get("anomaly_reference", "")}
        dst = OUT / f"geps_live_{tag}_{date}.nc"
        ds.to_netcdf(dst)
        written += 1
        v = an.values
        print(f"  {tag}: L{int(fc.L.min())}-{int(fc.L.max())}  "
              f"anom mean {np.nanmean(v):+.3g} sd {np.nanstd(v):.3g}  -> {dst.name}",
              flush=True)
    if all(t in rmm for t in RMM_TAGS):
        # mjo_geps.member_bands reads this instead of downloading the three channels a second time
        common = sorted(set.intersection(*(set(rmm[t][1].tolist()) for t in RMM_TAGS)))
        sel = {t: [list(rmm[t][1]).index(x) for x in common] for t in RMM_TAGS}
        np.savez_compressed(OUT / f"geps_rmm_members_{date}.npz", L=np.array(common),
                            **{t: rmm[t][0][sel[t]] for t in RMM_TAGS})
        print(f"  RMM band members: {len(common)} days x {rmm['olr'][0].shape[1]} -> geps_rmm_members_{date}.npz")
    if written == 0 and not datamart.OFFLINE:
        print("nothing fetched for this cycle - not building on stale or missing fields")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
