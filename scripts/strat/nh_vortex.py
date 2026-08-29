#!/usr/bin/env python3
"""NH polar vortex FORECAST monitor -- AIFS-ENS, IFS-ENS, GEPS and GDPS.

Three panels, forecast-first (a short analysis tail for context, then every
model that publishes the stratosphere):

  1. u(60N) at 10 hPa  -- the WMO sudden-stratospheric-warming diagnostic.
     Easterly here IS an SSW, so the zero line is heavy and the cross-model
     member fraction below it is called out.
  2. u(60N) at 100 hPa -- the coupling level. A 10 hPa event that never shows
     at 100 hPa rarely reaches the surface.
  3. 100 hPa polar-cap (65-90N) height anomaly -- positive = weak/displaced
     vortex; the field that leads the AO/NAO response.

Models (probed 2026-08-28; all four publish u AND height at 10 and 100 hPa):

  aifs  AIFS-ENS  25 pf + cf, 0.25 deg, day 15.  FREE -- the AAM task already
                  caches pf_u_10-50-100-... every cycle. Height is control-only:
                  open data publishes no perturbed z at these levels.
  geps  GEPS      20 members, 0.5 deg, day 16. One `allmbrs` GRIB per
                  variable/level/step (~2.4 MB UGRD, ~1.3 MB HGT) -> ~98 MB/cycle.
  gdps  GDPS      deterministic, 0.15 deg, day 10 -> ~31 MB/cycle.
  ifs   IFS-ENS   50 pf, 0.25 deg, day 15. There is NO control at these levels --
                  open data publishes only type=pf for u/gh at 10 and 100 hPa, so
                  "cheap control-only IFS" is not an option. 89 MB per step for all
                  50 members = 28.5 MB per member per cycle (1.44 GB for the full
                  ensemble), so IFS is OFF by default: pass --ifs-members N.

DETRENDING (the reason the reference is built the way it is)
------------------------------------------------------------
Polar-cap 100 hPa height has a strong secular trend -- tropospheric warming
lifts the 100 hPa surface. Measured on MERRA-2 1980-2026 with the seasonal cycle
removed first:

    z100 cap    +18.69 m/decade   (+86.9 m over the record, residual sd 160.5 m)
    u60 @10hPa   -0.23 m/s/decade (-1.0 m/s total,  residual sd 10.34 m/s)
    u60 @100hPa  -0.03 m/s/decade (-0.2 m/s total,  residual sd  3.92 m/s)

At today's epoch the z100 trend alone is +44 m against the record-mean level, so
an undetrended 1980-2026 band would place every recent day high by construction
and manufacture a "weak vortex" that is really just climate drift. The height
climatology is therefore referenced to the CURRENT epoch: the trend is removed
and every historical day is adjusted to today's level before the day-of-year
percentiles are taken. The winds are left alone -- their trends are under 2.5% of
their own variability, and pretending to correct that would be false precision.

Reference: MERRA-2 (scripts/telecon/data/m2_strat), 1980-2026 day-of-year
10th-90th percentiles. The polar cap is rebuilt from zbar with the same
cos-weighted 65-90N definition used on every forecast. The analysis tail is ERA5;
_consistency() verifies on every run that mixing the two reanalyses is legitimate
(measured 2026-08-28 over 94 overlapping days: u60 at 10 hPa -0.17 m/s r 0.997,
cap height +0.30 m r 1.000 -- negligible, so no offset is applied).

    python scripts/strat/nh_vortex.py                     # aifs+geps+gdps
    python scripts/strat/nh_vortex.py --models aifs,geps  # no downloads beyond GEPS
    python scripts/strat/nh_vortex.py --ifs-members 10    # opt into IFS spread
"""
from __future__ import annotations

import argparse
import glob
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE / "data"
M2 = REPO / "scripts" / "telecon" / "data" / "m2_strat"
CACHE = REPO / "scripts" / "ecmwf" / "cache"
DL = DATA / "vortex_dl"                    # downloaded GRIB, reused within a cycle
OUT = REPO / "assets" / "sst" / "nh_vortex.webp"

PHI = 60.0                 # WMO 60N for the SSW criterion
CAP = (65.0, 90.0)         # polar cap for the 100 hPa height anomaly
TAIL_DAYS = 21             # short analysis tail: this is a forecast product
G = 9.80665                # ECMWF `z` is geopotential; `gh` / HGT are already gpm
UA = {"User-Agent": "scorvec-enso/1.0"}
DETREND = {"zcap"}         # see the module docstring for why only this one

STYLE = {
    "aifs": ("#b4541f", "AIFS-ENS"),
    "ifs":  ("#1f4b6e", "IFS-ENS"),
    "geps": ("#2f7d4f", "GEPS"),
    "gdps": ("#6b4c9a", "GDPS"),
}


# ── shared reductions ───────────────────────────────────────────────────────
def zonal_at(da, phi=PHI):
    return da.sel(latitude=phi, method="nearest").mean("longitude")


def cap_mean(da, lo=CAP[0], hi=CAP[1]):
    lat = da.latitude
    sel = da.where((lat >= lo) & (lat <= hi), drop=True)
    w = np.cos(np.deg2rad(sel.latitude))
    return (sel.mean("longitude") * w).sum("latitude") / w.sum()


def _grib(path, **keys):
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
        filter_by_keys=keys, indexpath=""))


def _first(ds):
    return ds[list(ds.data_vars)[0]]


def _frame(da, reduce_fn, base, scale=1.0):
    r = reduce_fn(da) * scale
    t = base + pd.to_timedelta(r.step.values)
    if "number" in r.dims:
        return pd.DataFrame(r.transpose("step", "number").values, index=t)
    return pd.DataFrame({0: np.asarray(r.values)}, index=t)


def _fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=240) as r:
            dest.write_bytes(r.read())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"    miss {url.rsplit('/', 1)[-1][:52]}: {str(e)[:44]}", flush=True)
        return None
    return dest


# ── model loaders: each returns {"u10","u100","zcap"} of DataFrames ─────────
def load_aifs(cdir, base):
    pf = glob.glob(f"{cdir}/aifs-ens/pf_u_10-*.grib2")
    cf = glob.glob(f"{cdir}/aifs-ens/cf_u_10-*.grib2")
    if not (pf and cf):
        return None
    out = {}
    for lev, key in ((10, "u10"), (100, "u100")):
        d = _frame(_grib(pf[0], shortName="u", level=lev)["u"], zonal_at, base)
        d["cf"] = _frame(_grib(cf[0], shortName="u", level=lev)["u"], zonal_at, base)[0]
        out[key] = d
    zf = glob.glob(f"{cdir}/aifs-ens/cf_z_10-*.grib2")
    out["zcap"] = (_frame(_grib(zf[0], shortName="z", level=100)["z"], cap_mean,
                          base, 1.0 / G) if zf else None)
    return out


GEPS_URL = ("https://dd.weather.gc.ca/{d}/WXO-DD/ensemble/geps/grib2/raw/{c}/{L:03d}/"
            "CMC_geps-raw_{v}_ISBL_{lev:04d}_latlon0p5x0p5_{d}{c}_P{L:03d}_allmbrs.grib2")
GDPS_URL = ("https://dd.weather.gc.ca/{d}/WXO-DD/model_gdps/15km/{c}/{L:03d}/"
            "{d}T{c}Z_MSC_GDPS_{v}_IsbL-{lev:04d}_LatLon0.15_PT{L:03d}H.grib2")


def _eccc(kind, url_tpl, date, cyc, leads, spec):
    base = pd.Timestamp(f"{date} {cyc}:00")
    out = {}
    for key, var, lev, red in spec:
        rows = {}
        for L in leads:
            f = _fetch(url_tpl.format(d=date, c=cyc, v=var, lev=lev, L=L),
                       DL / f"{kind}_{date}{cyc}_{var}{lev}_{L:03d}.grib2")
            if f is None:
                continue
            # a GEPS `allmbrs` GRIB carries the 20 perturbed members AND the
            # control; opening it unfiltered raises "multiple values for unique
            # key". Take pf for the spread and append cf as one more column.
            vals = []
            for dt in (("pf", "cf") if kind == "geps" else (None,)):
                try:
                    ds = _grib(f, dataType=dt) if dt else _grib(f)
                    vals.append(np.atleast_1d(np.asarray(red(_first(ds)).values)).ravel())
                except Exception as e:                        # noqa: BLE001
                    if dt == "pf" or dt is None:
                        print(f"    {kind} {var}{lev} +{L}h unreadable: {str(e)[:50]}",
                              flush=True)
            if not vals:
                continue
            rows[base + pd.Timedelta(hours=L)] = np.concatenate(vals)
        if not rows:
            return None
        out[key] = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return out


def load_geps(date, cyc, leads):
    return _eccc("geps", GEPS_URL, date, cyc, leads,
                 [("u10", "UGRD", 10, zonal_at), ("u100", "UGRD", 100, zonal_at),
                  ("zcap", "HGT", 100, cap_mean)])


def load_gdps(date, cyc, leads):
    return _eccc("gdps", GDPS_URL, date, cyc, leads,
                 [("u10", "WindU", 10, zonal_at), ("u100", "WindU", 100, zonal_at),
                  ("zcap", "GeopotentialHeight", 100, cap_mean)])


def load_ifs(date, cyc, leads, members):
    """IFS-ENS via the repo's byte-ranged fetcher.

    NOT ecmwf.opendata's Client.download(): with param/levelist set it still
    pulled the whole step file -- one call landed a 2.76 GB GRIB for a single
    field. rangefetch parses the .index and requests only the matching message
    byte ranges, which is what the AAM/RMM tasks already use.
    """
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
    try:
        import rangefetch as rf
    except ImportError:
        print("    rangefetch unavailable; skipping IFS", flush=True)
        return None
    if not members:
        print("    IFS skipped: open data has no control (type=cf) for u/gh at 10/100 hPa, "
              "so members are the only option — pass --ifs-members N (~28.5 MB each)",
              flush=True)
        return None
    kind = "ef"          # NOT cf/pf: those object names 404 for IFS enfo
    typ = "pf"
    nums = list(range(1, members + 1))
    base = pd.Timestamp(f"{date} {cyc}:00")
    out, mb = {}, 0
    for param, lev, key in (("u", 10, "u10"), ("u", 100, "u100"), ("gh", 100, "zcap")):
        red = cap_mean if key == "zcap" else zonal_at
        rows = {}
        for L in leads:
            tgt = DL / f"ifs_{date}{cyc}_{param}{lev}_{typ}{members or ''}_{L:03d}.grib2"
            if not tgt.exists():
                try:
                    idx = rf.fetch_index(date, cyc, "ifs", int(L), kind, stream="enfo")
                    idx = [e for e in idx if e.get("type") == typ]
                    want = rf.select(idx, param=param, levelist=[lev], numbers=nums)
                    if not want:
                        continue
                    blob = rf.fetch_ranges(
                        rf.path_for(date, cyc, "ifs", int(L), kind, stream="enfo") + ".grib2",
                        rf.coalesce(want))
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    tgt.write_bytes(blob)
                except Exception as e:                        # noqa: BLE001
                    print(f"    IFS {param}{lev} +{L}h: {str(e)[:60]}", flush=True)
                    continue
            mb += tgt.stat().st_size / 1e6
            try:
                rows[base + pd.Timedelta(hours=L)] = np.atleast_1d(
                    np.asarray(red(_first(_grib(tgt))).values)).ravel()
            except Exception as e:                            # noqa: BLE001
                print(f"    IFS {param}{lev} +{L}h unreadable: {str(e)[:50]}", flush=True)
        if not rows:
            return None
        out[key] = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    print(f"    IFS transferred {mb:.0f} MB ({typ}{members or ''})", flush=True)
    return out


# ── reference: MERRA-2, epoch-referenced where a trend matters ──────────────
def _detrend_to_epoch(s, epoch):
    """Adjust every historical day to `epoch`'s climate level.

    Seasonal cycle out first, then a linear fit in decimal years; the fitted
    slope is used to shift history forward. Returns (adjusted series, m/decade).
    """
    ix = pd.DatetimeIndex(s.index)
    yr = np.asarray(ix.year + (ix.dayofyear - 1) / 365.25, dtype=float)
    r = (s - s.groupby(ix.dayofyear).transform("mean")).values
    b, _ = np.polyfit(yr, r, 1)
    return pd.Series(s.values + b * (epoch - yr), index=ix), 10.0 * b


def clim_doy(series):
    doy = pd.DatetimeIndex(series.index).dayofyear
    g = series.groupby(doy)
    q = lambda p: g.quantile(p).reindex(range(1, 367)).interpolate().bfill().ffill()
    return q(0.10), q(0.50), q(0.90)


def load_clim(epoch):
    fs = sorted(glob.glob(str(M2 / "m2_strat_*.nc")))
    if not fs:
        raise SystemExit(f"no MERRA-2 files under {M2}")
    d = xr.open_mfdataset(fs, combine="by_coords", chunks=None)
    idx = pd.DatetimeIndex(d.time.values)
    zb = d["zbar"].sel(lev=100.0)
    sel = zb.where((zb.lat >= CAP[0]) & (zb.lat <= CAP[1]), drop=True)
    w = np.cos(np.deg2rad(sel.lat))
    out = {
        "u10": pd.Series(np.asarray(d["u60"].sel(lev=10.0).values), index=idx).dropna(),
        "u100": pd.Series(np.asarray(d["u60"].sel(lev=100.0).values), index=idx).dropna(),
        "zcap": pd.Series(np.asarray(((sel * w).sum("lat") / w.sum()).values),
                          index=idx).dropna(),
    }
    print(f"  MERRA-2 reference {idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d} ({len(idx)} days)",
          flush=True)
    for k in list(out):
        ix = pd.DatetimeIndex(out[k].index)
        yr = np.asarray(ix.year + (ix.dayofyear - 1) / 365.25, dtype=float)
        r = (out[k] - out[k].groupby(ix.dayofyear).transform("mean")).values
        b = np.polyfit(yr, r, 1)[0]
        if k in DETREND:
            out[k], dec = _detrend_to_epoch(out[k], epoch)
            print(f"    {k}: trend {dec:+.2f} /decade — climatology referenced to "
                  f"epoch {epoch:.1f} (shifts the band {b*(epoch-yr.mean()):+.1f})", flush=True)
        else:
            print(f"    {k}: trend {10*b:+.2f} /decade vs resid sd "
                  f"{np.std(r - np.polyval(np.polyfit(yr, r, 1), yr)):.2f} — left as-is",
                  flush=True)
    return out


def analysis():
    p = DATA / "strat_obs.nc"
    if not p.exists():
        return {}
    d = xr.open_dataset(p)
    idx = pd.DatetimeIndex(d.time.values)
    ren = lambda v: d[v].rename({"lat": "latitude", "lon": "longitude"})
    z = pd.Series(cap_mean(ren("z100")).values, index=idx)
    if float(np.nanmedian(np.abs(z.values))) > 1e5:
        z = z / G
    return {"u10": pd.Series(zonal_at(ren("u10")).values, index=idx), "zcap": z}


def _consistency(obs, clim, name, unit, tol):
    if obs is None or clim is None:
        return
    ix = obs.index.intersection(clim.index)
    if len(ix) < 20:
        print(f"  {name}: <20 overlapping days, consistency unchecked", flush=True)
        return
    dd = (obs.loc[ix] - clim.loc[ix]).dropna()
    flag = "  <-- EXCEEDS TOLERANCE" if abs(dd.mean()) > tol else ""
    print(f"  {name}: ERA5-MERRA2 {dd.mean():+.2f} {unit} (sd {dd.std():.2f}, "
          f"r {obs.loc[ix].corr(clim.loc[ix]):.3f}, n={len(dd)}){flag}", flush=True)


# ── plot ────────────────────────────────────────────────────────────────────
def draw_model(ax, df, key, sub=None):
    col, lab = STYLE[key]
    v = df.values if sub is None else df.values - sub[:, None]
    n = df.shape[1]
    if n >= 5:
        ax.fill_between(df.index, np.nanpercentile(v, 10, axis=1),
                        np.nanpercentile(v, 90, axis=1),
                        color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(df.index, np.nanmean(v, axis=1), color=col, lw=2.4, zorder=5,
                label=f"{lab} ({n}m)")
    else:
        ax.plot(df.index, np.nanmean(v, axis=1), color=col, lw=2.0, ls="--", zorder=5,
                label=f"{lab} ({'control' if key == 'ifs' else 'det'})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="aifs,geps,gdps,ifs")
    ap.add_argument("--ifs-members", type=int, default=0)
    ap.add_argument("--cycle")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    want = [m.strip() for m in a.models.split(",") if m.strip()]

    cdirs = sorted(glob.glob(str(CACHE / "*z")))
    if a.cycle:
        cdirs = [d for d in cdirs if Path(d).name == a.cycle]
    cdir = base = None
    for d in reversed(cdirs):
        if glob.glob(f"{d}/aifs-ens/pf_u_10-*.grib2"):
            t = Path(d).name
            cdir, base = Path(d), pd.Timestamp(f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:00")
            break
    if base is None:
        print("no AIFS-ENS cycle in the cache to anchor on"); return 1
    date, cyc = f"{base:%Y%m%d}", f"{base:%H}"
    epoch = base.year + (base.dayofyear - 1) / 365.25
    print(f"cycle {base:%Y-%m-%d %HZ}  models={want}", flush=True)

    # the GRIB cache is ~250 MB a cycle — keep only this cycle's files
    tag = f"_{date}{cyc}_"
    for f in DL.glob("*.grib2"):
        if tag not in f.name:
            f.unlink(missing_ok=True)

    clim = load_clim(epoch)
    obs = analysis()
    _consistency(obs.get("u10"), clim["u10"], "u60 @10 hPa", "m/s", 1.0)
    _consistency(obs.get("zcap"), clim["zcap"], "z100 cap", "m", 60.0)

    fc = {}
    if "aifs" in want:
        fc["aifs"] = load_aifs(cdir, base)
    if "geps" in want:
        fc["geps"] = load_geps(date, cyc, list(range(0, 385, 24)))
    if "gdps" in want:
        fc["gdps"] = load_gdps(date, cyc, list(range(0, 241, 24)))
    if "ifs" in want:
        fc["ifs"] = load_ifs(date, cyc, list(range(0, 361, 24)), a.ifs_members)
    fc = {k: v for k, v in fc.items() if v}
    if not fc:
        print("no model data available"); return 1
    for k, v in fc.items():
        print(f"  {STYLE[k][1]}: " + ", ".join(
            f"{p}={v[p].shape[1]}m x{len(v[p])}" for p in v if v[p] is not None), flush=True)

    tmax = max(d[p].index[-1] for d in fc.values() for p in d if d[p] is not None)
    t0 = base - pd.Timedelta(days=TAIL_DAYS)
    span = pd.date_range(t0, tmax, freq="D")
    doy = pd.DatetimeIndex(span).dayofyear

    fig, axes = plt.subplots(3, 1, figsize=(13, 11.4), sharex=True)
    panels = [("u10", "u at 10 hPa, 60°N  (m s$^{-1}$)", False),
              ("u100", "u at 100 hPa, 60°N  (m s$^{-1}$)", False),
              ("zcap", f"100 hPa {CAP[0]:.0f}–{CAP[1]:.0f}°N height anomaly (m)", True)]
    for ax, (key, ylab, anom) in zip(axes, panels):
        q10, q50, q90 = clim_doy(clim[key])
        b = q50.reindex(doy).values if anom else np.zeros(len(span))
        lab = ("MERRA-2 climatology 10–90% (1980–, detrended to "
               f"{base:%Y})" if key in DETREND else "MERRA-2 climatology 10–90% (1980–)")
        ax.fill_between(span, q10.reindex(doy).values - b, q90.reindex(doy).values - b,
                        color="#e6e3dc", zorder=0, label=lab)
        if not anom:
            ax.plot(span, q50.reindex(doy).values, color="#9a958c", ls=":", lw=1.1, zorder=1)
        o = obs.get(key)
        if o is not None:
            o = o[(o.index >= t0) & (o.index <= base)]
            if len(o):
                ob = q50.reindex(pd.DatetimeIndex(o.index).dayofyear).values if anom else 0
                ax.plot(o.index, o.values - ob, color="#1a1a1a", lw=2.4, zorder=7,
                        label="ERA5 analysis")
        for k, d in fc.items():
            df = d.get(key)
            if df is None or not len(df):
                continue
            sub = q50.reindex(pd.DatetimeIndex(df.index).dayofyear).values if anom else None
            draw_model(ax, df, k, sub)
        ax.axvline(base, color="#8a8680", lw=1, zorder=3)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", fontsize=7.4, ncol=3, framealpha=0.9)
        if key == "u10":
            ax.axhline(0, color="#a33", lw=1.5, zorder=4)
            # models run to different lead counts, so pool the FRACTION per model
            # rather than concatenating rows of unequal length
            msg = "easterly at 10 hPa / 60°N = SSW (WMO)"
            hits = {k: float((d["u10"].values < 0).mean()) for k, d in fc.items()
                    if d.get("u10") is not None and d["u10"].shape[1] >= 5}
            if hits and max(hits.values()) > 0:
                worst = max(hits, key=hits.get)
                msg = (f"{100*hits[worst]:.1f}% of {STYLE[worst][1]} members easterly at "
                       "some lead — easterly here = SSW (WMO)")
            ax.text(0.005, 0.05, msg, transform=ax.transAxes, fontsize=8,
                    color="#a33", style="italic")
        if key == "zcap":
            ax.axhline(0, color="#8a8680", lw=1, ls=":", zorder=4)
            ax.text(0.005, 0.93, "positive = weaker / displaced vortex",
                    transform=ax.transAxes, fontsize=8, color="#8a8680", style="italic")
        if key == "u100" and obs.get("u100") is None:
            ax.text(0.5, 0.05, "no analysis tail at this level — the rolling ERA5 store "
                               "carries u only at 10 hPa",
                    transform=ax.transAxes, ha="center", fontsize=8, style="italic",
                    color="#8a8680")

    fig.suptitle(f"Northern polar vortex forecast — {base:%Y-%m-%d %HZ} cycle",
                 fontsize=14, fontweight="bold", y=0.996)
    fig.text(0.5, 0.004,
             "Zonal-mean zonal wind at 60°N and the 65–90°N polar-cap height. Shading is each ensemble's 10th–90th member "
             "percentile, the solid line its mean; dashed = deterministic or control-only.\n"
             "AIFS-ENS and IFS-ENS to day 15, GEPS to day 16, GDPS to day 10. Reference: MERRA-2 1980–2026 day-of-year "
             "percentiles; the height climatology is detrended to the current year (+18.7 m/decade), the winds are not "
             "(<0.25 m/s/decade). Vertical line = analysis time.",
             ha="center", va="bottom", fontsize=8, color="#8a8680", linespacing=1.5)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.026, 1, 0.985))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=112, facecolor="white", bbox_inches="tight",
                pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
