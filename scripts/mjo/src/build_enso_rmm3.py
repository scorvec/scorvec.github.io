#!/usr/bin/env python3
"""ENSO correction for the three-field RMM (OLR + U850 + U200) of the GEPS and GEFS pages (offline, once).

The fit of enso_rmm_correction.json (the wind-only RMM of the MJO page) with the OLR channel added: the
120-day-filtered band anomalies are regressed on dN34, Nino-3.4 minus its mean over the same 120 days, longitude by
longitude, and the slopes are projected onto the WH04 EOFs and divided by the three-field normalisers (pc_std).
RMM_clean = RMM - dN34 * c.

  winds   ARCO/ERA5 15S-15N band-mean u850/u200 (data/reference/arco_eq_u), as the wind-only fit
  OLR     NOAA interpolated OLR, 15S-15N cos-weighted (the source of the reference climatology and EOFs, and of
          BoM's RMM), from $MJO_CACHE/data/reference/olr.day.mean.nc. It ends in 2022, which sets the period:
          2003-2022, all three channels over the same days.
  N34     ERSST v6 monthly, daily between mid-months (enso_rmm.ersst_n34)

Every anomaly is taken against climatology.nc (common-year day of year, doycal.sel_doy) and filtered with the mean
of the 120 days ending on its day, the window dN34 uses. Uncertainty: 2000 bootstrap resamples of whole years (the
slopes are rebuilt from per-year sums). A component whose 95 % interval includes 0 is not significant and is applied
as 0 (the site's significance rule); the file keeps the fitted value beside it. The OLR and wind parts of c are kept
apart because GEPS filters its winds with a map that ends about a week before its OLR map's.

Validation against BoM's RMM (rmm.74toRealtime.txt, $DATA_ARCHIVE/geps_subx/telecon/bom/): 2003-2013, when BoM
removed ENSO itself (its SST1 regression), and 2014-2022, when it did not; plus a fit on 2014-2022 alone scored on
2003-2013, out of sample.

    python src/build_enso_rmm3.py            # -> data/reference/enso_rmm3_correction.json
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from doycal import sel_doy
import enso_rmm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from localdirs import env_dir

REF = Path(__file__).resolve().parents[1] / "data" / "reference"
OUT = REF / "enso_rmm3_correction.json"
Y0, Y1 = 2003, 2022
CV_SPLIT = 2013  # cross-validation: fit after this year, score up to it
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


def rmm(ch: dict[str, np.ndarray], clim: xr.Dataset, eofs: xr.Dataset) -> np.ndarray:
    """Three-field RMM (2, time) of the filtered anomalies."""
    return np.stack(
        [
            sum((ch[t] / clim.attrs[f"std_{t}"]) @ eofs[f"eof_{t}"].sel(mode=m).values for t in TAGS)
            / float(eofs["pc_std"].sel(mode=m))
            for m in (1, 2)
        ]
    )


def bom(days: pd.DatetimeIndex) -> np.ndarray:
    """BoM's RMM1/RMM2 on `days` (2, time), NaN where missing."""
    src = env_dir("DATA_ARCHIVE", "geps_subx", "telecon", "bom", "rmm.74toRealtime.txt")
    rows = {}
    for ln in src.read_text().splitlines()[2:]:
        p = ln.split()
        if len(p) >= 7 and abs(float(p[3])) < 900 and abs(float(p[4])) < 900:
            rows[pd.Timestamp(int(p[0]), int(p[1]), int(p[2]))] = (float(p[3]), float(p[4]))
    b = pd.DataFrame.from_dict(rows, orient="index").reindex(days)
    return b.values.T


def scores(r: np.ndarray, b: np.ndarray, dn: np.ndarray, sel: np.ndarray) -> dict:
    """Agreement with BoM over the days `sel`: correlation by component, the share of active days (BoM amplitude
    >= 1) in the same phase, and each index's correlation with dN34."""
    g = sel & np.isfinite(b).all(0) & np.isfinite(r).all(0) & np.isfinite(dn)

    def phase(x):
        return np.floor(((np.degrees(np.arctan2(x[1], x[0])) - 180.0) % 360.0) / 45.0)

    act = np.hypot(*b[:, g]) >= 1.0
    return {
        "r1": round(float(np.corrcoef(r[0, g], b[0, g])[0, 1]), 3),
        "r2": round(float(np.corrcoef(r[1, g], b[1, g])[0, 1]), 3),
        "phase_exact": round(float((phase(r[:, g]) == phase(b[:, g]))[act].mean()), 3),
        "n": int(g.sum()),
        "n_active": int(act.sum()),
        "corr_dn34_r1": round(float(np.corrcoef(r[0, g], dn[g])[0, 1]), 3),
        "corr_dn34_r2": round(float(np.corrcoef(r[1, g], dn[g])[0, 1]), 3),
        "bom_corr_dn34_r1": round(float(np.corrcoef(b[0, g], dn[g])[0, 1]), 3),
    }


def main() -> int:
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
    print(f"wind-only check: {wind_only[0]:+.4f}, {wind_only[1]:+.4f} (enso_rmm_correction.json: {enso_rmm.load_c()})")

    # validation against BoM; the out-of-sample fit uses the years after CV_SPLIT only
    r = rmm(ch, clim, eofs)
    b = bom(days)
    late = uy[uy > CV_SPLIT]
    p_cv = project({t: slope(sums[t], late) for t in TAGS}, clim, eofs)
    c_cv = np.where(sig, p_cv["olr"] + p_cv["wind"], 0.0)
    c_ap = applied["olr"] + applied["wind"]
    early, after = ok & (years <= CV_SPLIT), ok & (years > CV_SPLIT)
    val = {
        f"vs_bom_{Y0}_{CV_SPLIT}_enso_removed": {
            "raw": scores(r, b, dn, early),
            "clean": scores(r - dn * c_ap[:, None], b, dn, early),
            f"clean_fit_{CV_SPLIT + 1}_{Y1}_only": scores(r - dn * c_cv[:, None], b, dn, early),
        },
        f"vs_bom_{CV_SPLIT + 1}_{Y1}_enso_not_removed": {
            "raw": scores(r, b, dn, after),
            "clean": scores(r - dn * c_ap[:, None], b, dn, after),
        },
    }
    for k, v in val.items():
        for m, s in v.items():
            print(f"  {k} {m}: r {s['r1']:.3f}/{s['r2']:.3f}, corr with dN34 {s['corr_dn34_r1']:+.3f}")

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
        "wind_only_check": {"c": comp(wind_only), "enso_rmm_correction": comp(enso_rmm.load_c())},
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
