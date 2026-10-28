#!/usr/bin/env python3
"""IOD correction for the wind-only RMM of the MJO page (offline, once), on top of the ENSO correction.

Why. The RMM removes only the previous 120-day mean, so a growing Indian Ocean Dipole projects onto it as the growing
El Nino does (enso_rmm.py): a positive IOD's easterlies over the equatorial Indian Ocean stand in phases 8-1-2 (autumn
2026, 1997, 1982). The ENSO correction takes out the part tied to the Nino-3.4 change; this file takes out the part tied
to the DMI change that is left after it.

What. dDMI = the DMI minus its mean over the 120 days of the wind filter window, as dN34 (enso_rmm.window_dn34). d is
the least-squares slope of the ENSO-corrected observed index (rmm_era5.nc wind-only RMM minus dN34 * c, c as applied)
on dDMI over the IOD's season (June to December; dDMI is 0 in January-May), fitted over 2003-2024, the ENSO fit's years.
The RMM is a linear projection of the filtered winds, so this slope is the per-longitude regression projected onto the
wind EOFs. Uncertainty: 2000 bootstrap resamples of whole years; a component whose 95 % interval includes 0 is applied
as 0 (the site's significance rule). RMM_clean = RMM_wind - dN34 * c - dDMI * d.

  DMI     Saji et al. (1999): western (50-70E, 10S-10N) minus south-eastern (90-110E, 10S-0) tropical Indian Ocean,
          ERSST v6 monthly anomalies against 1991-2020 (the real-time OISST DMI's base), written here to
          data/reference/dmi_ersst6.csv and read daily between mid-months (enso_rmm.ersst_dmi). Real time: the NOAA
          OISST v2.1 daily DMI (assets/sst/data/enso_daily.json, sst-roni.py), 31-day trailing mean.

Validation on the observed RMM over 1980-2002 (years the fit did not use): the correlation of RMM1 with dDMI over the
IOD's season and the September-December variance of RMM1 removed, ENSO-corrected against ENSO- and IOD-corrected; the
slope refitted on 1980-2002 alone (the same term in the other half of the record); and the September 15-November 30
mean vector of the positive-IOD autumns. --validate rewrites the validation from the committed series alone.

    python src/build_iod_rmm.py --ersst ../sst/data/ersst_v6_mnmean.nc   # -> data/reference/{dmi_ersst6.csv,iod_rmm_correction.json}
    python src/build_iod_rmm.py --validate
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import enso_rmm
import rmm_era5

HERE = Path(__file__).resolve().parent
ERSST = HERE.parents[1] / "sst" / "data" / "ersst_v6_mnmean.nc"  # kept current by sst.yml
FIT = (2003, 2024)
OUT_OF_SAMPLE = (1980, 2002)
WEST = dict(lat=(-10, 10), lon=(50, 70))  # as sst-roni.py's IOD_WEST / IOD_EAST
EAST = dict(lat=(-10, 0), lon=(90, 110))
AUTUMNS = (1982, 1994, 1997, 2006, 2015, 2019, 2023)  # positive-IOD autumns (2015 a weak one, with a strong El Nino)
N_BOOT = 2000


def dmi_monthly(path: Path) -> pd.Series:
    """ERSST v6 DMI, monthly anomalies against 1991-2020."""
    sst = xr.open_dataset(path)["sst"]

    def box(b):
        s = sst.sel(lat=slice(b["lat"][1], b["lat"][0]), lon=slice(*b["lon"]))
        s = s.weighted(np.cos(np.deg2rad(s.lat))).mean(("lat", "lon")).to_series()
        base = s[(s.index.year >= 1991) & (s.index.year <= 2020)]
        return s - base.groupby(base.index.month).mean().reindex(s.index.month).values

    dmi = (box(WEST) - box(EAST)).loc["1950":]
    dmi.index = dmi.index.to_period("M")
    return dmi


def terms() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """The observed days, the ENSO-corrected wind-only RMM (2, time), dDMI over the IOD's season (0 outside it) and dN34."""
    s = rmm_era5.load()
    days = s.index
    dn = enso_rmm.window_dn34(enso_rmm.ersst_n34(), days, days)
    dd = enso_rmm.window_dn34(enso_rmm.ersst_dmi(), days, days)
    dd = np.where(enso_rmm.iod_on(days), dd, 0.0)
    e = s[["rmm1_wind", "rmm2_wind"]].values.T - dn * enso_rmm.load_c()[:, None]
    return days, e, dd, dn


def fit(e: np.ndarray, x: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Least-squares slope (RMM1, RMM2) of e on x over the days sel (a mask, or indices that may repeat)."""
    xc = x[sel] - x[sel].mean()
    return (e[:, sel] - e[:, sel].mean(1, keepdims=True)) @ xc / (xc @ xc)


def validation(days, e, dd, dn, d: np.ndarray) -> dict:
    years = days.year.values
    ok = np.isfinite(e).all(0) & np.isfinite(dd) & np.isfinite(dn)
    oos = ok & (years >= OUT_OF_SAMPLE[0]) & (years <= OUT_OF_SAMPLE[1])
    season = oos & enso_rmm.iod_on(days)
    sond = oos & np.isin(days.month, (9, 10, 11, 12))
    c2 = e - dd * d[:, None]
    out = {
        "index": "the observed wind-only RMM (rmm_era5.nc) with the ENSO correction as applied (dN34 * c), without "
        "and with the IOD term; dDMI from the ERSST v6 DMI, dN34 from ERSST v6 Nino-3.4, each minus its mean over "
        "the 120 days ending on the day",
        f"era5_rmm_{OUT_OF_SAMPLE[0]}_{OUT_OF_SAMPLE[1]}_out_of_sample": {
            "n_season": int(season.sum()),
            "corr_ddmi_r1": {
                "enso": round(float(np.corrcoef(e[0, season], dd[season])[0, 1]), 3),
                "enso_iod": round(float(np.corrcoef(c2[0, season], dd[season])[0, 1]), 3),
            },
            "sond_r1_variance_removed_pct": round(
                float(100 * (1 - (c2[0, sond] ** 2).sum() / (e[0, sond] ** 2).sum())), 1
            ),
            "refit_d_rmm1": round(float(fit(e, dd, oos)[0]), 4),
        },
        "autumns_sep15_nov30": {},
    }
    for y in AUTUMNS:
        k = ok & (days >= f"{y}-09-15") & (days <= f"{y}-11-30")
        row = {"ddmi_mean": round(float(dd[k].mean()), 2), "dn34_mean": round(float(dn[k].mean()), 2)}
        for tag, r in (("enso", e), ("enso_iod", c2)):
            z = r[0, k] + 1j * r[1, k]
            row[tag] = {
                "amp": round(float(abs(z.mean())), 2),
                "stationarity": round(float(abs(z.mean()) / np.abs(z).mean()), 2),
            }
        out["autumns_sep15_nov30"][str(y)] = row
    v = out[f"era5_rmm_{OUT_OF_SAMPLE[0]}_{OUT_OF_SAMPLE[1]}_out_of_sample"]
    print(
        f"  {OUT_OF_SAMPLE[0]}-{OUT_OF_SAMPLE[1]}: corr RMM1 with dDMI {v['corr_ddmi_r1']['enso']:+.3f} -> "
        f"{v['corr_ddmi_r1']['enso_iod']:+.3f}; Sep-Dec RMM1 variance removed {v['sond_r1_variance_removed_pct']:.1f} %; "
        f"refitted d {v['refit_d_rmm1']:+.3f}"
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IOD correction for the wind-only RMM")
    ap.add_argument("--ersst", type=Path, default=ERSST, help="ERSST v6 monthly grid (sst.yml's ersst_v6_mnmean.nc)")
    ap.add_argument("--validate", action="store_true", help="only rewrite the validation from the committed series")
    a = ap.parse_args(argv)
    if a.validate:
        out = json.loads(enso_rmm.CORR_IOD.read_text())
        out["validation"] = validation(*terms(), enso_rmm.load_d())
        enso_rmm.CORR_IOD.write_text(json.dumps(out, indent=1))
        print(f"wrote the validation of {enso_rmm.CORR_IOD.name}")
        return 0
    dmi = dmi_monthly(a.ersst)
    enso_rmm.ERSST_DMI.write_text("month,dmi\n" + "".join(f"{p},{v:.3f}\n" for p, v in dmi.items()))
    print(f"wrote {enso_rmm.ERSST_DMI.name}: {dmi.index[0]}..{dmi.index[-1]}")

    days, e, dd, dn = terms()
    years = days.year.values
    sel = np.isfinite(e).all(0) & np.isfinite(dd) & np.isfinite(dn) & (years >= FIT[0]) & (years <= FIT[1])
    d = fit(e, dd, sel)
    uy = np.unique(years[sel])
    rng = np.random.default_rng(1)
    boot = []
    for _ in range(N_BOOT):
        pick = np.concatenate([np.flatnonzero(sel & (years == y)) for y in rng.choice(uy, len(uy))])
        boot.append(fit(e, dd, pick))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    sig = (lo > 0) | (hi < 0)
    applied = np.where(sig, d, 0.0)
    print(
        f"d = {d[0]:+.4f}, {d[1]:+.4f}; 95 %: RMM1 {lo[0]:+.3f}..{hi[0]:+.3f}, RMM2 {lo[1]:+.3f}..{hi[1]:+.3f} -> applied {applied}"
    )

    def comp(x):
        return {"rmm1": round(float(x[0]), 4), "rmm2": round(float(x[1]), 4)}

    out = {
        "title": "IOD correction for the wind-only RMM, on top of the ENSO correction (enso_rmm_correction.json)",
        "made": f"{dt.date.today():%Y-%m-%d}",
        "builder": "scripts/mjo/src/build_iod_rmm.py",
        "index": "DMI (Saji et al. 1999), 50-70E 10S-10N minus 90-110E 10S-0, anomalies against 1991-2020. Fit: "
        "ERSST v6 monthly (data/reference/dmi_ersst6.csv) interpolated to daily between mid-months. Real time: NOAA "
        "OISST v2.1 daily (assets/sst/data/enso_daily.json), 31-day trailing mean.",
        "fit": f"{FIT[0]}-{FIT[1]}: slope of the observed wind-only RMM (rmm_era5.nc) minus dN34 * c (c as applied) on "
        "dDMI, the DMI minus its mean over the 120 days ending on the day, over June-December (dDMI 0 in January-May)",
        "apply": "RMM_clean = RMM_wind - dN34 * c - dDMI * d in June-December, dDMI = the DMI at the day (forecasts: at "
        "the init, held through the forecast) minus its mean over the 120 days of the wind filter window",
        "months": list(enso_rmm.IOD_MONTHS),
        "fitted": comp(d),
        "bootstrap": {
            "method": f"{N_BOOT} resamples of whole years ({len(uy)}), slope refitted",
            "d_ci95": {"rmm1": [round(lo[0], 4), round(hi[0], 4)], "rmm2": [round(lo[1], 4), round(hi[1], 4)]},
            "significant": {"rmm1": bool(sig[0]), "rmm2": bool(sig[1])},
        },
        "applied": comp(applied),
        "validation": validation(days, e, dd, dn, applied),
    }
    enso_rmm.CORR_IOD.write_text(json.dumps(out, indent=1))
    print(f"wrote {enso_rmm.CORR_IOD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
