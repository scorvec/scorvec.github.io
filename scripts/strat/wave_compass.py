#!/usr/bin/env python3
"""
The wave compass (stratosphere.html#compass): why a 500 hPa wave pattern feeds the heat flux into the vortex.

LINEAR INTERFERENCE. The 500 hPa height along 55-65N always carries a standing wave-1 and wave-2, the normal N for the
date (reference/wave_clim.nc: forced by the mountains and the land-sea contrast). A forecast's wave-k coefficient is
that normal plus an anomaly A, and the wave's size squared is

    |N + A|^2 = |N|^2 + 2 Re(N A*) + |A|^2,    2 Re(N A*) = 2 |N| |A| cos(offset)

where the offset is the angle, in phase, between the anomaly's ridges and the normal ridges. The cross term is the
interference: an anomaly sitting on the normal ridges builds the wave, the same anomaly half a wavelength away cancels
it. The upward wave activity that reaches the vortex, the 100 hPa eddy heat flux, grows with the planetary wave's
size, so where the anomaly sits matters more than how big it is (Garfinkel et al. 2010; Smith and Kushner 2012).

ERA5 RELATION (`relation`). Within WINDOW days of a run's calendar date, every ERA5 day's 500 hPa wave-k against the
100 hPa wave-k heat flux over that day and the LEAD - 1 after it (reference/vt100_era5.nc, 45-75N):
  fit       flux = a + b 2 Re(N A*) + c |A|^2 by least squares; b is the interference's weight, and the page draws the
            expected flux only where its 90 % range (BOOT resamples of whole winters) is above zero
  bins      for the largest third of anomalies (|A| at or above the window's 2/3 quantile), the mean flux in 30-degree
            bins of the offset, each with its 90 % range from the same resampling, and the mean over all the window's
            days
The window's days are grouped by winter (each window lies inside one shifted year), so the resampling keeps a winter's
days, which are far from independent, together.

PER RUN (`write`, called by wave_maps.py after an ensemble's loops): every member's 500 and 100 hPa wave-1 and wave-2
coefficient for each day to day 15 and the 100 hPa members' mean (the wave's lean between the levels; ssw.html's
checklist reads each member's own lean), the normal at 500 and 100 hPa and a Hotelling test of the members' mean
anomaly against zero (does the ensemble agree where the anomaly sits?), with the ERA5 relation for the run's date ->
assets/sst/data/wave_compass_<model>.json. The page computes the expected flux member by member from the fit (the flux
is quadratic, so the flux of the mean is not the mean flux) and turns every member's anomaly by the same angle when the
reader moves the ridge.

REFERENCE (`build`, offline from the local ERA5 store; scripts/strat/reference/wave_compass_era5.npz): the daily-mean
500 hPa wave-1 and wave-2 coefficients over 55-65N (WeatherBench 2's 1.5 deg ERA5, the band and rfft scaling of
build_wave_clim.py) beside the daily 100 hPa wave-1 and wave-2 heat flux (mean of 00 and 12 UTC), for every day both
exist. Daily means against the 00 UTC normal: the stationary waves have no meaningful diurnal cycle.

    python scripts/strat/wave_compass.py build
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.degdays import doy366

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "lib"))
REF = HERE / "reference" / "wave_compass_era5.npz"
CLIM = HERE / "reference" / "wave_clim.nc"
VT = HERE / "reference" / "vt100_era5.nc"
OUT = REPO / "assets" / "sst" / "data"
WAVES = (1, 2)
WINDOW = 30  # days either side of the run's calendar date
LEAD = 5  # the flux over the 500 hPa day and the four after it
BIG = 2 / 3  # the binned curve's anomalies: the window's largest third
BIN = 30  # degrees of offset per bin
BOOT, SEED = 1000, 7


def normal(dates, k: int, lev: int = 500) -> np.ndarray:
    """The normal standing wave-k at `lev` over 55-65N on each date (wave_clim.nc, 366-slot calendar), complex gpm."""
    with xr.open_dataset(CLIM) as d:
        c = d[f"nh{lev}_k{k}_re"].values + 1j * d[f"nh{lev}_k{k}_im"].values
    return c[doy366(pd.DatetimeIndex(dates))]


def build() -> None:
    """The ERA5 reference series (module docstring) -> REF."""
    from build_wave_clim import band_coeff
    from localdirs import env_dir

    store = env_dir("ERA5_STORE", "wb2_1p5_daily", "z500")
    with xr.open_dataset(VT) as v:
        vt = v[[f"nh_k{k}" for k in WAVES]].mean("hour").to_dataframe()
    parts = []
    for y in range(vt.index.year.min(), vt.index.year.max() + 1):
        path = store / f"z500_{y}.nc"
        if not path.exists():
            continue
        with xr.open_dataset(path) as d:
            z = d["z500"].transpose("time", "latitude", "longitude")  # the store's z500 holds metres
            c = band_coeff(z.values.astype("float64"), z.latitude.values)
            parts.append(pd.DataFrame(c, index=pd.DatetimeIndex(z.time.values).normalize(), columns=list(WAVES)))
    c = pd.concat(parts)
    days = pd.date_range(max(c.index[0], vt.index[0]), min(c.index[-1], vt.index[-1]), freq="D")
    c, vt = c.reindex(days), vt.reindex(days)
    amp = np.abs(c[1].dropna())
    if not 20 < amp.median() < 300:
        raise SystemExit(f"wave-1 amplitude median {amp.median():.0f}: not geopotential metres")
    np.savez_compressed(
        REF,
        date=days.values.astype("datetime64[D]"),
        c=c[list(WAVES)].to_numpy().astype("complex64"),
        f=vt[[f"nh_k{k}" for k in WAVES]].to_numpy().astype("float32"),
        meta=json.dumps({
            "z500": "WeatherBench 2 ERA5 1.5 deg daily means, cos-weighted 55-65N band, rfft x 2/N, 0 E origin",
            "flux": "ERA5 100 hPa [v'T'] wave-1 and wave-2, 45-75N, mean of 00 and 12 UTC (vt100_era5.nc)",
            "waves": list(WAVES),
        }),
    )  # fmt: skip
    print(f"{REF.name}: {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}, {len(days)} days")


def _ref():
    with np.load(REF) as z:
        return {"date": z["date"], "c": z["c"], "f": z["f"]}


def _sig(x: float) -> float:
    return float(f"{x:.4g}")


def relation(day, k: int, ref=None) -> dict:
    """The ERA5 link (module docstring) between the 500 hPa wave-k and the next LEAD days' 100 hPa wave-k heat flux,
    within WINDOW days of `day`'s calendar date."""
    r = _ref() if ref is None else ref
    t = pd.DatetimeIndex(r["date"])
    j = WAVES.index(k)
    flux = pd.Series(r["f"][:, j].astype("float64"), index=t).rolling(LEAD, min_periods=LEAD).mean().shift(1 - LEAD)
    slot, s0 = doy366(t), int(doy366(pd.DatetimeIndex([day]))[0])
    gap = np.abs(slot - s0)
    keep = (np.minimum(gap, 366 - gap) <= WINDOW) & flux.notna().to_numpy() & np.isfinite(r["c"][:, j])
    t, y, c = t[keep], flux.to_numpy()[keep], r["c"][keep, j].astype("complex128")
    N = normal(t, k)
    A = c - N
    lin, a2 = 2 * np.real(N * np.conj(A)), np.abs(A) ** 2
    winter = (t - pd.to_timedelta(s0 - 183, unit="D")).year.to_numpy()  # each window inside one shifted year
    groups, g = np.unique(winter, return_inverse=True)
    X = np.column_stack([np.ones(len(y)), lin, a2])
    XtX = np.zeros((len(groups), 3, 3))
    Xty = np.zeros((len(groups), 3))
    np.add.at(XtX, g, X[:, :, None] * X[:, None, :])
    np.add.at(Xty, g, X * y[:, None])
    beta = np.linalg.solve(XtX.sum(0), Xty.sum(0))
    offset = np.degrees(np.abs(np.angle(A * np.conj(N))))
    big = np.abs(A) >= np.quantile(np.abs(A), BIG)
    edges = np.arange(0, 180 + BIN, BIN)
    b = np.clip(np.digitize(offset, edges) - 1, 0, len(edges) - 2)
    sums, cnt = np.zeros((len(groups), len(edges) - 1)), np.zeros((len(groups), len(edges) - 1))
    np.add.at(sums, (g[big], b[big]), y[big])
    np.add.at(cnt, (g[big], b[big]), 1)
    pick = np.random.default_rng(SEED).integers(0, len(groups), (BOOT, len(groups)))
    w = np.stack([np.bincount(p, minlength=len(groups)) for p in pick]).astype("float64")  # (BOOT, groups) counts
    boot_beta = np.linalg.solve(np.einsum("bg,gij->bij", w, XtX), np.einsum("bg,gi->bi", w, Xty)[..., None])[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_bins = (w @ sums) / (w @ cnt)
    b90 = np.percentile(boot_beta[:, 1], [5, 95])
    bins = []
    for i in range(len(edges) - 1):
        n = int(cnt[:, i].sum())
        lo, hi = np.nanpercentile(boot_bins[:, i], [5, 95]) if n else (np.nan, np.nan)
        bins.append({"lo": int(edges[i]), "hi": int(edges[i + 1]), "n": n, "mean": round(float(sums[:, i].sum() / n), 2) if n else None,
                     "lo90": round(float(lo), 2) if n else None, "hi90": round(float(hi), 2) if n else None})  # fmt: skip
    return {
        "window_days": WINDOW, "lead_days": [0, LEAD - 1], "years": [int(t.year.min()), int(t.year.max())],
        "n_days": len(y), "winters": len(groups), "mean": round(float(y.mean()), 2),
        "fit": {"a": _sig(beta[0]), "b": _sig(beta[1]), "c": _sig(beta[2]), "b90": [_sig(b90[0]), _sig(b90[1])],
                "r": round(float(np.corrcoef(X @ beta, y)[0, 1]), 2)},
        "linked": bool(b90[0] > 0), "big_m": round(float(np.quantile(np.abs(A), BIG)), 1), "bins": bins,
    }  # fmt: skip


def agreement(A: np.ndarray) -> float | None:
    """p-value of Hotelling's test that the members' mean anomaly (complex, one per member) is zero: small when the
    ensemble agrees where the anomaly sits. F(2, n-2) has the closed-form tail (1 + 2F/(n-2))^(-(n-2)/2). A spread
    with no width in some direction (members identical there) is inverted on the directions it has."""
    n = len(A)
    if n < 4:
        return None
    X = np.column_stack([A.real, A.imag])
    m = X.mean(0)
    t2 = n * m @ np.linalg.pinv(np.cov(X, rowvar=False)) @ m
    f = (n - 2) / (2 * (n - 1)) * t2
    return float((1 + 2 * f / (n - 2)) ** (-(n - 2) / 2))


def _pairs(c) -> list:
    return np.round(np.stack([np.real(c), np.imag(c)], -1), 1).tolist()


def write(init: pd.Timestamp, valid, c500: dict, c100: dict, model: str, label: str, out_dir: Path = OUT, ref=None):
    """One ensemble run -> wave_compass_<model>.json (module docstring). c500[k], c100[k]: (day, member) complex 500
    and 100 hPa coefficients; valid: the days' valid times."""
    valid = pd.DatetimeIndex(valid)
    waves = {}
    for k in WAVES:
        N = normal(valid, k)
        waves[str(k)] = {
            "normal": _pairs(N),
            "normal_100": _pairs(normal(valid, k, 100)),
            "members": _pairs(c500[k]),
            "members_100": _pairs(c100[k]),
            "mean_100": _pairs(np.mean(c100[k], 1)),
            "p_anomaly": [None if p is None else round(p, 4) for p in (agreement(c - n) for c, n in zip(c500[k], N))],
            "era5": relation(init, k, ref),
        }
    doc = {
        "model": model, "label": label, "init": f"{init:%Y-%m-%dT%H:%MZ}", "n_members": int(np.shape(c500[1])[1]),
        "valid": [f"{v:%Y-%m-%d %H:%M}" for v in valid], "band": [55, 65], "hemisphere": "NH",
        "convention": "coefficient c: height along the band = |c| cos(k lon + arg c), lon east from 0 E; ridge at -arg(c)/k",
        "waves": waves,
    }  # fmt: skip
    path = Path(out_dir) / f"wave_compass_{model}.json"
    path.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"  wrote {path.name}: {len(valid)} days, {doc['n_members']} members")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["build"], help="build: the ERA5 reference series from the local store")
    ap.parse_args(argv)
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
