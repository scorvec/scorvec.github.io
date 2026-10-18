#!/usr/bin/env python3
"""ENSO correction for the three-field RMM (OLR + U850 + U200) of the GEPS and GEFS pages (offline, once).

The fit of enso_rmm_correction.json (the wind-only RMM of the MJO page) with the OLR channel added: the
120-day-filtered band anomalies are regressed on dN34, Nino-3.4 minus its mean over the same 120 days, longitude by
longitude, and the slopes are projected onto the WH04 EOFs and divided by the three-field normalisers (pc_std).
RMM_clean = RMM - dN34 * c.

  winds   ARCO/ERA5 15S-15N band-mean u850/u200 (data/reference/arco_eq_u), as the wind-only fit
  OLR     NOAA interpolated OLR, 15S-15N cos-weighted (the source of the reference climatology and EOFs), from
          $MJO_CACHE/data/reference/olr.day.mean.nc. It ends in 2022, which sets the period: 2003-2022, all three
          channels over the same days.
  N34     ERSST v6 monthly, daily between mid-months (enso_rmm.ersst_n34)

Every anomaly is taken against climatology.nc (common-year day of year, doycal.sel_doy) and filtered with the mean
of the 120 days ending on its day, the window dN34 uses. Uncertainty: 2000 bootstrap resamples of whole years (the
slopes are rebuilt from per-year sums). A component whose 95 % interval includes 0 is not significant and is applied
as 0 (the site's significance rule); the file keeps the fitted value beside it. The OLR and wind parts of c are kept
apart because GEPS filters its winds with a map that ends about a week before its OLR map's.

Validation (validation()) on the observed RMM (rmm_era5.nc, ERA5 1979 onwards), with each correction as applied:
the three-field index with this c, the wind-only index of the MJO page with enso_rmm_correction.json's. For each, its
correlation with dN34 and its share of active days (amplitude >= 1) before and after the correction, over 1980-2002
(years neither fit used) and over the fit's years; for the wind-only index also its agreement with the three-field one
(raw with raw, corrected with corrected), and the winters and El Nino growth phases the MJO page cites. --validate
rewrites the validation of both files from the committed series alone (the wind-only correction's fit is not rebuilt
here, and this fit's inputs are local).

    python src/build_enso_rmm3.py              # -> data/reference/enso_rmm3_correction.json
    python src/build_enso_rmm3.py --validate   # the validation of both corrections, from the committed series
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from doycal import sel_doy
import enso_rmm
import rmm_era5

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from localdirs import env_dir

REF = Path(__file__).resolve().parents[1] / "data" / "reference"
OUT = REF / "enso_rmm3_correction.json"
Y0, Y1 = 2003, 2022
OUT_OF_SAMPLE = (1980, 2002)  # validation years before both fits (the wind-only one used 2003-2024)
WIND_FIT = (2003, 2024)
# the seasons the MJO page cites: winters with an MJO, and the months while an El Nino grew
WINDOWS = {
    "2009/10": ("2009-12-01", "2010-02-28"),
    "2015/16": ("2015-12-01", "2016-02-29"),
    "2023/24": ("2023-12-01", "2024-02-29"),
    "JAS 2009 (growth)": ("2009-07-01", "2009-09-30"),
    "JAS 2015 (growth)": ("2015-07-01", "2015-09-30"),
    "JAS 2023 (growth)": ("2023-07-01", "2023-09-30"),
}
TAGS = ("olr", "u850", "u200")
N_BOOT = 2000


def filtered(a: np.ndarray, t: pd.DatetimeIndex) -> np.ndarray:
    """(time, lon) anomaly minus its mean over the WINDOW days ending on each day (NaN until a full window)."""
    df = pd.DataFrame(a, index=t).asfreq("D")
    return a - df.rolling(enso_rmm.WINDOW, min_periods=enso_rmm.WINDOW).mean().loc[t].values


def channels(clim: xr.Dataset, lon: np.ndarray) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Filtered anomalies of the three channels on the days Y0-Y1 (NaN until the first full window)."""
    u = xr.open_mfdataset([REF / "arco_eq_u" / f"arco_eq_u_{y}.nc" for y in range(Y0, Y1 + 1)], combine="by_coords")
    u = u["uband"]
    u = u.sel(lev=[850, 200]).load()
    tu = pd.DatetimeIndex(u.time.values).normalize()
    o = xr.open_dataset(env_dir("MJO_CACHE", "data", "reference", "olr.day.mean.nc"))["olr"]
    o = o.sel(time=slice(f"{Y0}-01-01", f"{Y1}-12-31"), lat=slice(15, -15))
    ob = o.weighted(np.cos(np.deg2rad(o.lat))).mean("lat").transpose("time", "lon").load()
    if not np.allclose(ob.lon.values, lon):
        raise SystemExit("NOAA OLR is not on the EOF longitudes")
    to = pd.DatetimeIndex(ob.time.values).normalize()
    raw = {
        "u850": (tu, u.sel(lev=850).interp(lon=lon).values),  # 1 deg -> the 2.5 deg EOF longitudes
        "u200": (tu, u.sel(lev=200).interp(lon=lon).values),
        "olr": (to, ob.values),
    }
    days = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31")
    out = {}
    for tag, (t, v) in raw.items():
        f = filtered(v - sel_doy(clim[f"clim_{tag}"], t), t)
        i = t.get_indexer(days)
        if (i < 0).any():
            raise SystemExit(f"{tag}: {int((i < 0).sum())} days missing in {Y0}-{Y1}")
        out[tag] = f[i]
    return days, out


def year_sums(x: np.ndarray, d: np.ndarray, years: np.ndarray) -> dict:
    """Per-year sums for a centred least-squares slope of every column of x on d."""
    return {
        y: (m.sum(), d[m].sum(), (d[m] ** 2).sum(), x[m].sum(0), (x[m] * d[m, None]).sum(0))
        for y in np.unique(years)
        if (m := years == y).any()
    }


def slope(sums: dict, years) -> np.ndarray:
    n, sd, sdd, sx, sxd = (sum(sums[y][k] for y in years) for k in range(5))
    return (sxd - sx * sd / n) / (sdd - sd**2 / n)


def project(beta: dict[str, np.ndarray], clim: xr.Dataset, eofs: xr.Dataset) -> dict[str, np.ndarray]:
    """Per-channel slopes -> the OLR and wind parts of c, (RMM1, RMM2) each, on the three-field normalisers."""
    part = {}
    for grp, tags in (("olr", ("olr",)), ("wind", ("u850", "u200"))):
        part[grp] = np.array(
            [
                sum((beta[t] / clim.attrs[f"std_{t}"]) @ eofs[f"eof_{t}"].sel(mode=m).values for t in tags)
                / float(eofs["pc_std"].sel(mode=m))
                for m in (1, 2)
            ]
        )
    return part


def phase(x: np.ndarray) -> np.ndarray:
    """WH04 phase 1-8 of (2, time) RMM."""
    return np.floor(((np.degrees(np.arctan2(x[1], x[0])) - 180.0) % 360.0) / 45.0) + 1


def scores(r: np.ndarray, dn: np.ndarray, sel: np.ndarray, ref: np.ndarray | None = None) -> dict:
    """The index r (2, time) over the days sel: its correlation with dN34 by component and its share of active days;
    with ref, its agreement with that index (correlation by component, the share of ref's active days in the same
    phase)."""
    g = sel & np.isfinite(r).all(0) & np.isfinite(dn)
    out = {
        "n": int(g.sum()),
        "corr_dn34_r1": round(float(np.corrcoef(r[0, g], dn[g])[0, 1]), 3),
        "corr_dn34_r2": round(float(np.corrcoef(r[1, g], dn[g])[0, 1]), 3),
        "active": round(float((np.hypot(*r[:, g]) >= 1.0).mean()), 3),
    }
    if ref is not None:
        act = np.hypot(*ref[:, g]) >= 1.0
        out |= {
            "r1": round(float(np.corrcoef(r[0, g], ref[0, g])[0, 1]), 3),
            "r2": round(float(np.corrcoef(r[1, g], ref[1, g])[0, 1]), 3),
            "phase_exact": round(float((phase(r[:, g]) == phase(ref[:, g]))[act].mean()), 3),
        }
    return out


def window_stats(r: np.ndarray, days: pd.DatetimeIndex, t0: str, t1: str) -> dict:
    """Share of active days, mean amplitude and median eastward phase speed (deg/day) over t0..t1."""
    k = (days >= t0) & (days <= t1)
    amp, spd = enso_rmm._speed_amp(r[0, k], r[1, k], np.arange(k.sum(), dtype=float))
    return {
        "active": round(float((amp >= 1.0).mean()), 2),
        "amp": round(float(amp.mean()), 2),
        "speed": round(float(np.median(spd)), 1),
    }


def validation(c3: np.ndarray, c_wind: np.ndarray) -> dict[str, dict]:
    """The validation sections of both corrections on the observed RMM, with c3 (three-field) and c_wind (wind-only)
    as applied -> {"three": ..., "wind": ...}."""
    s = rmm_era5.load()
    days = s.index
    dn = enso_rmm.window_dn34(enso_rmm.ersst_n34(), days, days)  # the window of the series' own 120-day filter
    three = s[["rmm1", "rmm2"]].values.T
    wind = s[["rmm1_wind", "rmm2_wind"]].values.T
    three_c, wind_c = three - dn * c3[:, None], wind - dn * c_wind[:, None]
    years = days.year.values
    note = (
        "the observed RMM (rmm_era5.nc: ERA5, previous 120-day mean removed, ENSO left in); clean = RMM - dN34 * c as "
        "applied, dN34 = ERSST v6 Nino-3.4 minus its mean over the 120 days ending on the day; active = amplitude >= 1"
    )
    out = {
        "three": {"index": "three-field rmm1/rmm2 of " + note},
        "wind": {"index": "wind-only rmm1_wind/rmm2_wind of " + note},
    }
    for (y0, y1), tag in ((OUT_OF_SAMPLE, "out_of_sample"), ((Y0, Y1), "fit_years")):
        sel = (years >= y0) & (years <= y1)
        out["three"][f"era5_rmm_{y0}_{y1}_{tag}"] = {"raw": scores(three, dn, sel), "clean": scores(three_c, dn, sel)}
    for (y0, y1), tag in ((OUT_OF_SAMPLE, "out_of_sample"), (WIND_FIT, "fit_years")):
        sel = (years >= y0) & (years <= y1)
        out["wind"][f"era5_rmm_{y0}_{y1}_{tag}"] = {
            "raw": scores(wind, dn, sel, three),
            "clean": scores(wind_c, dn, sel, three_c),
        }
    out["wind"]["winters_and_growth_phases"] = {
        k: {
            "raw": window_stats(wind, days, *w),
            "clean": window_stats(wind_c, days, *w),
            "three_field": window_stats(three, days, *w),
            "dn34_mean": round(float(np.nanmean(dn[(days >= w[0]) & (days <= w[1])])), 2),
        }
        for k, w in WINDOWS.items()
    }
    for kind, v in out.items():
        for k, e in v.items():
            if k.startswith("era5_rmm"):
                r, c = e["raw"], e["clean"]
                print(
                    f"  {kind} {k}: corr with dN34 RMM1 {r['corr_dn34_r1']:+.3f} -> {c['corr_dn34_r1']:+.3f}, "
                    f"active {r['active']:.3f} -> {c['active']:.3f}"
                    + (f"; vs three-field r {c['r1']:.3f}/{c['r2']:.3f}" if "r1" in c else "")
                )
    return out


def validate_files() -> None:
    """Rewrite the validation of both correction files from the committed series."""
    d3, dw = json.loads(OUT.read_text()), json.loads(enso_rmm.CORR.read_text())
    v = validation(sum(enso_rmm.load_c3().values()), enso_rmm.load_c())
    d3["validation"], dw["validation"] = v["three"], v["wind"]
    OUT.write_text(json.dumps(d3, indent=1))
    enso_rmm.CORR.write_text(json.dumps(dw, indent=1))
    print(f"wrote the validation of {OUT.name} and {enso_rmm.CORR.name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ENSO correction for the three-field RMM")
    ap.add_argument("--validate", action="store_true", help="only rewrite both corrections' validation")
    if ap.parse_args(argv).validate:
        validate_files()
        return 0
    clim = xr.open_dataset(REF / "climatology.nc")
    eofs = xr.open_dataset(REF / "eofs.nc")
    lon = eofs.longitude.values.astype(float)
    days, ch = channels(clim, lon)
    dn = enso_rmm.window_dn34(enso_rmm.ersst_n34(), days, days)
    ok = np.isfinite(dn) & np.all([np.isfinite(ch[t]).all(1) for t in TAGS], axis=0)
    years = days.year.values
    print(f"{Y0}-{Y1}: {ok.sum()} days, dN34 sd {dn[ok].std():.3f} K", flush=True)

    sums = {t: year_sums(ch[t][ok], dn[ok], years[ok]) for t in TAGS}
    uy = np.unique(years[ok])
    beta = {t: slope(sums[t], uy) for t in TAGS}
    part = project(beta, clim, eofs)
    c = part["olr"] + part["wind"]
    # the wind-only fit of enso_rmm_correction.json, recomputed here as a check of the arithmetic (its normalisers)
    wind_only = np.array(
        [
            sum((beta[t] / clim.attrs[f"std_{t}"]) @ eofs[f"eof_{t}"].sel(mode=m).values for t in ("u850", "u200"))
            / float(eofs["pc_wind_std"].sel(mode=m))
            for m in (1, 2)
        ]
    )
    rng = np.random.default_rng(1)
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(uy, len(uy))  # one draw of years for all three channels
        p = project({t: slope(sums[t], pick) for t in TAGS}, clim, eofs)
        boot.append([*(p["olr"] + p["wind"]), *p["olr"]])
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    sig = (lo[:2] > 0) | (hi[:2] < 0)
    applied = {g: np.where(sig, part[g], 0.0) for g in part}
    print(
        f"c = {c[0]:+.4f}, {c[1]:+.4f} (OLR part {part['olr'][0]:+.4f}, {part['olr'][1]:+.4f}); 95 %: RMM1 "
        f"{lo[0]:+.3f}..{hi[0]:+.3f}, RMM2 {lo[1]:+.3f}..{hi[1]:+.3f} -> applied {applied['olr'] + applied['wind']}"
    )
    print(
        f"wind-only check: {wind_only[0]:+.4f}, {wind_only[1]:+.4f} (enso_rmm_correction.json: {enso_rmm.load_c(fitted=True)})"
    )

    c_ap = applied["olr"] + applied["wind"]
    val = validation(c_ap, enso_rmm.load_c())["three"]

    def comp(a):
        return {"rmm1": round(float(a[0]), 4), "rmm2": round(float(a[1]), 4)}

    out = {
        "title": "ENSO correction for the three-field RMM (OLR + U850 + U200) of the GEPS and GEFS products",
        "made": f"{dt.date.today():%Y-%m-%d}",
        "builder": "scripts/mjo/src/build_enso_rmm3.py",
        "index": "Nino-3.4 anomaly. Fit: ERSST v6 monthly (PSL nina34.anom.data) interpolated to daily between "
        "mid-months. Real time: NOAA OISST v2.1 daily (assets/sst/data/enso_daily.json), 31-day trailing mean.",
        "fit": f"{Y0}-{Y1}: ARCO/ERA5 15S-15N band-mean u850/u200 and NOAA interpolated OLR (15S-15N, cos-weighted), "
        "anomalies against climatology.nc, each filtered with the mean of the 120 days ending on its day; slopes "
        "on dN34 (Nino-3.4 minus its mean over the same 120 days) by longitude, projected onto the WH04 EOFs, "
        "divided by pc_std",
        "apply": "RMM_clean = RMM - dN34_olr * applied.olr - dN34_wind * applied.wind, each dN34 = Nino-3.4 at the "
        "day (forecasts: at the init, held through the forecast) minus its mean over the 120 days of that channel "
        "group's filter map",
        "fitted": {"c": comp(c), "olr": comp(part["olr"]), "wind": comp(part["wind"])},
        "bootstrap": {
            "method": f"{N_BOOT} resamples of whole years ({len(uy)}), slopes refitted from per-year sums",
            "c_sd": comp(boot[:, :2].std(0)),
            "c_ci95": {"rmm1": [round(lo[0], 4), round(hi[0], 4)], "rmm2": [round(lo[1], 4), round(hi[1], 4)]},
            "olr_part_ci95": {"rmm1": [round(lo[2], 4), round(hi[2], 4)], "rmm2": [round(lo[3], 4), round(hi[3], 4)]},
            "significant": {"rmm1": bool(sig[0]), "rmm2": bool(sig[1])},
        },
        "applied": {"c": comp(c_ap), "olr": comp(applied["olr"]), "wind": comp(applied["wind"])},
        "wind_only_check": {"c": comp(wind_only), "enso_rmm_correction": comp(enso_rmm.load_c(fitted=True))},
        "longitude": [float(x) for x in lon],
        "beta_olr_wm2_per_K": [round(float(x), 4) for x in beta["olr"]],
        "beta_u850_ms_per_K": [round(float(x), 4) for x in beta["u850"]],
        "beta_u200_ms_per_K": [round(float(x), 4) for x in beta["u200"]],
        "validation": val,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
