#!/usr/bin/env python3
"""Key stratospheric diagnostics as time series — GEPS ensemble, day 0 to 35.

Five panels, the numbers a vortex monitor actually turns on:

  u(60) 10 hPa   the WMO sudden-stratospheric-warming diagnostic. Drawn ABSOLUTE,
                 not as an anomaly, with the zero line heavy — an SSW is defined
                 as this crossing zero, and an anomaly plot hides exactly that.
  u(60) 100 hPa  the coupling level. A 10 hPa reversal that never reaches here
                 rarely reaches the surface either.
  cap T 10/100   polar-cap (65-90) temperature anomaly. The warming IS the event,
                 and it leads the wind reversal, so this turns first.
  cap z 100 hPa  the field that leads the AO/NAO response. Epoch-referenced,
                 because its secular trend (+18.9 m/decade) is large enough to
                 fake a weak vortex on a 1991-2020 base — see build_strat_trend.py.

Spread is drawn as bands (p10-p90, p25-p75) with the ensemble mean over them,
not as 21 spaghetti traces: with this many members the envelope is the readable
object and individual members are noise on the eye.

An analysis tail is prepended from earlier cycles' lead-0 fields, so the forecast
starts from somewhere rather than from nothing. Datamart keeps about 30 days.

Diagnostics are computed on the GEPS grid and the climatological ones on the 1
degree climatology grid: these are large-scale integrals (a zonal mean and a
polar-cap mean), and the grid-induced difference is far below the signal.

    python strat_series.py --hemi nh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyproj  # noqa: F401,E402  before xarray/eccodes: eccodes preloads its own libproj (CLAUDE.md gotcha)
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strat_maps as sm                                        # noqa: E402

OUT = sm.OUT
LEADS = list(range(0, 841, 24))
PANELS = [("u60", "UGRD", "U", 10,  "u(60{h}) 10 hPa",  "m/s", False),
          ("u60", "UGRD", "U", 100, "u(60{h}) 100 hPa", "m/s", False),
          ("capT", "TMP", "T", 100, "polar-cap T anomaly, 100 hPa", "K", True),
          ("capZ", "HGT", "Z", 100, "polar-cap height anomaly, 100 hPa", "m", True)]


def members(path):
    """All 21 members as a stacked DataArray (number, lat, lon)."""
    out = []
    for dt in ("pf", "cf"):
        try:
            ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(
                filter_by_keys={"dataType": dt}, indexpath=""))
            a = ds[list(ds.data_vars)[0]].load()
            out.append(a if "number" in a.dims else a.expand_dims(number=[0]))
        except Exception:                                        # noqa: BLE001
            pass
    if not out:
        return None
    return xr.concat([o.drop_vars("number") .assign_coords(
        number=np.arange(o.sizes["number"])) if False else o for o in out],
        dim="number", coords="minimal", compat="override")


def reduce_field(a, kind, north, demean=False):
    """Reduce a field to its diagnostic. `demean` removes the hemispheric mean
    first, which for HEIGHT is essential: the QBO alone moves stratospheric
    heights by >100 m from year to year (2026 is +216 m in the tropics at
    10 hPa, the largest in 36 years), and that offset is not a vortex signal.
    What couples to the surface is the cap relative to its surroundings."""
    lat = a["latitude"] if "latitude" in a.coords else a["lat"]
    ln = lat.name
    lonname = "longitude" if "longitude" in a.dims else "lon"
    if demean:
        hm = (lat >= 20) if north else (lat <= -20)
        wh = np.cos(np.deg2rad(lat.where(hm, drop=True)))
        zh = a.where(hm, drop=True).mean(lonname)
        a = a - (zh * wh).sum(ln) / wh.sum()
    if kind == "u60":
        z = a.mean("longitude" if "longitude" in a.dims else "lon")
        return z.interp({ln: 60.0 if north else -60.0}).values
    m = (lat >= 65) if north else (lat <= -65)
    w = np.cos(np.deg2rad(lat.where(m, drop=True)))
    z = a.where(m, drop=True).mean("longitude" if "longitude" in a.dims else "lon")
    return ((z * w).sum(ln) / w.sum()).values


def clim_value(c, tr, ckey, lev, kind, valid, north):
    ref, _ = sm.clim_at(c, ckey, lev, valid, tr)
    lat = ref["lat"]
    if kind == "capZ":
        hm = (lat >= 20) if north else (lat <= -20)
        wh = np.cos(np.deg2rad(lat.where(hm, drop=True)))
        ref = ref - (ref.where(hm, drop=True).mean("lon") * wh).sum("lat") / wh.sum()
    if kind == "u60":
        return float(ref.mean("lon").interp(lat=60.0 if north else -60.0))
    m = (lat >= 65) if north else (lat <= -65)
    w = np.cos(np.deg2rad(lat.where(m, drop=True)))
    z = ref.where(m, drop=True).mean("lon")
    return float((z * w).sum("lat") / w.sum())


# one JSON per (hemisphere, run); in Actions STRAT_SERIES_DIR points into the GEPS state carried between runs
SERIES_DIR = Path(__import__("os").environ.get("STRAT_SERIES_DIR", sm.DATA / "series"))
# the observed tail: each day's lead-0 diagnostic, reduced once and kept (the GRIBs are not)
TAIL_CACHE = SERIES_DIR / "analysis_tail.json"

# Lead-dependent drift of GEPS against MERRA-2, from ECCC's 2001-2020 reforecasts in the ECMWF S2S archive
# (geps_s2s_hindcast.py, 2026-09-26). corrected = raw - bias(start day of year, lead). Northern panels only; cap z is
# not corrected (no MERRA-2 counterpart for its >=20N demeaning). Missing table or a start more than 10 days from
# any reforecast start -> no correction, and the chart says so.
BIAS_FILE = Path(__file__).resolve().parent / "data" / "geps_s2s" / "bias_nh.json"
BIAS_KEY = {("u60", 10): "u60_10", ("u60", 100): "u60_100", ("capT", 100): "capT_100"}
_BIAS = None


def bias_table():
    global _BIAS
    if _BIAS is None:
        import json
        _BIAS = {}
        if BIAS_FILE.exists():
            t = json.loads(BIAS_FILE.read_text())
            _BIAS = {pd.Timestamp(k).dayofyear: v for k, v in t.get("starts", {}).items()}
    return _BIAS


def bias_at(base, lead_h, kind, lev, north):
    """Drift to subtract, linear in day of year between the two nearest reforecast starts; held at its last value
    beyond the last reforecast lead. None = no correction available."""
    key = BIAS_KEY.get((kind, lev))
    B = bias_table()
    if not north or not key or not B:
        return None
    doy = base.dayofyear
    ds = sorted(B)
    dist = lambda d: min(abs(d - doy), 365 - abs(d - doy))
    near = sorted(ds, key=dist)[:2]
    if dist(near[0]) > 10:
        return None
    def at(d):
        e = B[d]; arr = e.get(key)
        if arr is None:
            return None
        sh = e.get("step_h") or list(range(0, 24 * len(arr), 24))
        i = int(np.argmin(np.abs(np.asarray(sh) - lead_h)))   # nearest lead; ECDS leads start at 24 h, so 0 -> 24 h
        return arr[i]
    if len(near) == 1 or dist(near[1]) > 10 or dist(near[0]) == 0:
        return at(near[0])
    w0, w1 = dist(near[1]), dist(near[0])                    # inverse-distance in day of year
    a0, a1 = at(near[0]), at(near[1])
    if a0 is None or a1 is None:
        return a0 if a1 is None else a1
    return (a0 * w0 + a1 * w1) / (w0 + w1)


def forecast_series(date, cyc, kind, var, ckey, lev, north, as_anom, c, tr, raw_out=None):
    """(valid times, members[lead, member], normal[lead]) for one diagnostic, bias-corrected where a reforecast
    drift is available (raw_out, if a list, receives the uncorrected ensemble mean per lead)."""
    base = pd.Timestamp(f"{date} {cyc}:00")
    ft, fm, fc_clim = [], [], []
    for L in LEADS:
        f = sm.fetch(date, cyc, var, lev, L)
        mm = members(f) if f else None
        if mm is None:
            continue
        valid = base + pd.Timedelta(hours=L)
        vals = np.atleast_1d(reduce_field(mm, kind, north,
                                          demean=(kind == "capZ"))).astype(float).ravel()
        cval = clim_value(c, tr, ckey, lev, kind, valid, north)
        b = bias_at(base, L, kind, lev, north)
        if raw_out is not None:
            raw_out.append(float(np.mean(vals)) - (cval if as_anom else 0.0))
        if b is not None:
            vals = vals - b
        if as_anom:
            vals = vals - cval
        ft.append(valid); fm.append(vals); fc_clim.append(cval)
    return ft, (np.array(fm) if fm else None), fc_clim


_TAIL = None


def tail_value(d, cyc, var, lev, kind, north):
    """The ensemble-mean lead-0 diagnostic of one earlier cycle, reduced once and cached in TAIL_CACHE (so each
    day's analysis GRIB is downloaded once ever, not once per run that shows it in its 14-day tail)."""
    import json
    global _TAIL
    if _TAIL is None:
        try:
            _TAIL = json.loads(TAIL_CACHE.read_text()) if TAIL_CACHE.exists() else {}
        except ValueError:
            _TAIL = {}
    key = f"{d}{cyc}_{var}{lev}_{kind}_{'nh' if north else 'sh'}"
    if key in _TAIL:
        return _TAIL[key]
    f = sm.fetch(d, cyc, var, lev, 0)
    m = sm.ens_mean(f) if f else None
    if m is None:
        return None
    v = float(reduce_field(m, kind, north, demean=(kind == "capZ")))
    _TAIL[key] = v
    newest = sorted(_TAIL)[-600:]                   # ~40 days x 4 diagnostics x 2 hemispheres, plenty
    _TAIL = {k: _TAIL[k] for k in newest}
    TAIL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TAIL_CACHE.write_text(json.dumps(_TAIL))
    return v


def previous_extended(date, cyc):
    """Newest extended cycle before `date` with a full lead set in the GRIB cache, or a saved series."""
    have = sorted({q.name.split("_")[1][:8] for q in sm.DL.glob(f"geps_*{cyc}_UGRD10_840.grib2")}
                  | {q.stem.split("_")[2][:8] for q in SERIES_DIR.glob(f"strat_*_*{cyc}.json")})
    older = [d for d in have if d < date]
    return older[-1] if older else None


def series_path(hemi, date, cyc):
    return SERIES_DIR / f"strat_{hemi}_{date}{cyc}.json"


def load_series(hemi, date, cyc):
    q = series_path(hemi, date, cyc)
    if not q.exists():
        return None
    import json
    d = json.loads(q.read_text())
    base = pd.Timestamp(f"{date} {cyc}:00")
    for key, v in d["panels"].items():
        v["t"] = pd.to_datetime(v["t"])
        v["mean"] = np.array(v["mean"])
        if not d.get("bias_corrected") and bias_table():          # cached before the correction existed
            kind = next(k for k in ("u60", "capT", "capZ") if key.startswith(k)); lev = int(key[len(kind):])   # "u6010" = u60 at 10 hPa
            leads = ((v["t"] - base) / pd.Timedelta(hours=1)).astype(int)
            b = [bias_at(base, L, kind, lev, hemi == "nh") for L in leads]
            v["mean"] = np.array([m - bb if bb is not None else m for m, bb in zip(v["mean"], b)])
    return d


def save_series(hemi, date, cyc, panels):
    """Keep the ensemble statistics of every run: the next run overlays them,
    and a run-over-run history of the day-35 vortex forecast is worth having."""
    import json
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    out = {"date": date, "cyc": cyc, "hemi": hemi, "bias_corrected": bool(bias_table()), "panels": {}}
    for key, (ft, M) in panels.items():
        out["panels"][key] = {
            "t": [t.strftime("%Y-%m-%dT%H") for t in ft],
            "mean": np.round(M.mean(1), 3).tolist(),
            "p10": np.round(np.percentile(M, 10, 1), 3).tolist(),
            "p25": np.round(np.percentile(M, 25, 1), 3).tolist(),
            "p75": np.round(np.percentile(M, 75, 1), 3).tolist(),
            "p90": np.round(np.percentile(M, 90, 1), 3).tolist()}
    series_path(hemi, date, cyc).write_text(json.dumps(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--cyc", default="00")
    ap.add_argument("--prev", help="previous run to overlay (default: newest cached extended cycle before --date)")
    ap.add_argument("--hemi", default="nh", choices=("nh", "sh"))
    ap.add_argument("--tail-days", type=int, default=14)
    ap.add_argument("--clim", default=str(sm.CLIM))
    a = ap.parse_args()
    north = a.hemi == "nh"
    c = xr.open_dataset(a.clim)
    tr = xr.open_dataset(sm.TREND) if sm.TREND.exists() else None
    date = a.date or sm.latest_extended(a.cyc)
    base = pd.Timestamp(f"{date} {a.cyc}:00")
    print(f"GEPS extended {base:%Y-%m-%d %H}Z, {a.hemi.upper()}", flush=True)

    # 2 x 2 rather than four stacked strips: at a fixed page width four rows
    # made each panel short and wide, which is the wrong shape for reading a
    # 50-day trace against an ensemble envelope.
    prev_date = a.prev or previous_extended(date, a.cyc)
    prev_series = load_series(a.hemi, prev_date, a.cyc) if prev_date else None
    if prev_date and prev_series is None:
        # first time through: build the previous run's series from the cache
        print(f"  previous run {prev_date}: no cached series, computing", flush=True)
        pp = {}
        for kind, var, ckey, lev, title, unit, as_anom in PANELS:
            pft, pM, _ = forecast_series(prev_date, a.cyc, kind, var, ckey, lev, north, as_anom, c, tr)
            if pM is not None:
                pp[f"{kind}{lev}"] = (pft, pM)
        if pp:
            save_series(a.hemi, prev_date, a.cyc, pp)
            prev_series = load_series(a.hemi, prev_date, a.cyc)
    panels_out = {}
    fig = plt.figure(figsize=(13.4, 8.4))
    n = len(PANELS)
    NCOL = 2
    L0, R0, T0, B0 = 0.055, 0.988, 0.885, 0.070
    PW = (R0 - L0) / NCOL
    PH = (T0 - B0) / ((n + NCOL - 1) // NCOL)
    for pi, (kind, var, ckey, lev, title, unit, as_anom) in enumerate(PANELS):
        # analysis tail from earlier cycles' lead 0
        tt, tv = [], []
        for k in range(a.tail_days, 0, -1):
            d = (base - pd.Timedelta(days=k)).strftime("%Y%m%d")
            v = tail_value(d, a.cyc, var, lev, kind, north)
            if v is None:
                continue
            valid = pd.Timestamp(f"{d} {a.cyc}:00")
            b0 = bias_at(base, 0, kind, lev, north)
            if b0 is not None:
                v -= b0                                           # same analysis-level offset as the forecast's step 0
            if as_anom:
                v -= clim_value(c, tr, ckey, lev, kind, valid, north)
            tt.append(valid); tv.append(v)
        # forecast, per member
        raw_mean = []
        ft, M, fc_clim = forecast_series(date, a.cyc, kind, var, ckey, lev, north, as_anom, c, tr, raw_out=raw_mean)
        if M is None:
            continue
        corrected = bias_at(base, 24, kind, lev, north) is not None
        pkey = f"{kind}{lev}"
        panels_out[pkey] = (ft, M)
        col, row = pi % NCOL, pi // NCOL
        ax = fig.add_axes([L0 + col * PW + 0.042,
                           T0 - (row + 1) * PH + 0.070,
                           PW * 0.895, PH * 0.735])
        ax.fill_between(ft, np.percentile(M, 10, 1), np.percentile(M, 90, 1),
                        color="#2c4a72", alpha=0.16, lw=0, label="p10–p90")
        ax.fill_between(ft, np.percentile(M, 25, 1), np.percentile(M, 75, 1),
                        color="#2c4a72", alpha=0.28, lw=0, label="p25–p75")
        ax.plot(ft, M.mean(1), "#16335c", lw=2.2,
                label="ensemble mean, bias-corrected" if corrected else "ensemble mean")
        if corrected:
            ax.plot(ft, raw_mean, color="#16335c", lw=1.1, ls=(0, (1, 1.6)), alpha=0.75,
                    label="raw GEPS mean (model drift left in)")
        if tt:
            ax.plot(tt + ft[:1], tv + [M.mean(1)[0]], color="#3f3a33", lw=1.6,
                    label=f"analysis, last {len(tt)} d")
            ax.plot(tt, tv, "o", color="#3f3a33", ms=3.0)
        # previous extended run, ensemble mean only, on the valid days both runs
        # cover: the run-over-run shift is the question this answers, and a
        # second set of bands would bury the first
        if prev_series and pkey in prev_series["panels"]:
            pv = prev_series["panels"][pkey]
            keep = (pv["t"] >= ft[0]) & (pv["t"] <= ft[-1])
            if keep.any():
                ax.plot(pv["t"][keep], pv["mean"][keep], ls=(0, (4, 2)), color="#c8781e",
                        lw=1.7, label=f"previous run mean, init {pd.Timestamp(prev_series['date']):%b %d}")
                # change on the common days, for the log
                cm = pd.Series(M.mean(1), index=pd.DatetimeIndex(ft))
                pm = pd.Series(pv["mean"][keep], index=pv["t"][keep])
                dd = (cm - pm).dropna()
                if len(dd):
                    print(f"    vs previous run ({prev_series['date']}): mean change over "
                          f"{len(dd)} common days {dd.mean():+.2f} {unit}, at the last "
                          f"common day {dd.iloc[-1]:+.2f}", flush=True)
        if as_anom:
            ax.axhline(0, color="#8a8680", lw=0.9)
        else:
            ax.plot(ft, fc_clim, ls="--", color="#b4453c", lw=1.5,
                    label="1991–2020 normal")
            ax.axhline(0, color="k", lw=1.6)          # the SSW threshold itself
        ax.axvline(base, color="#8a8680", lw=0.8, ls=":")
        ax.set_ylabel(unit, fontsize=9.5)
        ax.set_title(title.format(h="N" if north else "S"), fontsize=11,
                     fontweight="bold", loc="left", pad=4)
        ax.tick_params(labelsize=8.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        ax.grid(alpha=0.25, lw=0.5)
        if pi == 0:
            ax.legend(fontsize=7.2, ncol=2, loc="upper left", framealpha=0.9)
        print(f"  {title.format(h='N' if north else 'S'):38s} "
              f"day35 mean {M.mean(1)[-1]:+8.2f} {unit}  "
              f"(p10 {np.percentile(M,10,1)[-1]:+.2f} p90 {np.percentile(M,90,1)[-1]:+.2f})",
              flush=True)
    if north and bias_table():
        fig.text(0.5, 0.012, "Bias-corrected for the model's drift with lead time: GEPS reforecasts 2001–2020 (ECCC, via the "
                 "ECMWF S2S archive) minus MERRA-2 on the same dates, by start date and lead. Polar-cap height is not "
                 "corrected. Bands and mean are after correction; the dotted line is the raw model.",
                 ha="center", fontsize=8.3, color="#5c6b73", wrap=True)
    fig.suptitle(f"{'Northern' if north else 'Southern'} polar vortex\n"
                 f"GEPS extended ensemble · init {base:%Y-%m-%d %H}Z",
                 fontsize=13, fontweight="bold", y=0.985, va="top")
    if panels_out:
        save_series(a.hemi, date, a.cyc, panels_out)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"strat_{a.hemi}_series.webp"
    fig.savefig(p, dpi=110, facecolor="white", pil_kwargs={"quality": 90, "method": 6})
    plt.close(fig)
    print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
