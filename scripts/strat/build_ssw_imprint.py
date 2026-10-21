#!/usr/bin/env python3
"""Reference for the SSW surface-imprint index (ssw_imprint.py): the CMIP6 pattern, the ERA5 baseline, the ERA5 index
since 1959 and the two tests that decide what the stratosphere page may say about it.

    python build_ssw_imprint.py        # -> reference/ssw_imprint.nc (inputs: ERA5_STORE, committed references)

INPUTS. The CMIP6 composite (reference/cmip6_follow.nc); ERA5 daily sea-level pressure, WeatherBench2 1.5 deg, 0-90N,
1959- (the local ERA5 store, wb2_1p5_daily/slp: "Pa" in the attributes, hPa in the values); the MERRA-2 SSW list
(reference/strat_history_events.json, marginal events out) and 100 hPa polar-cap series (reference/strat_history.nc
z100s, the scale of the dripping-paint plots).

TESTS (stored as attributes and printed):
  imprint   the index over days 1-30, 31-60 and 1-60 after the MERRA-2 SSWs: across events (one-sample t-test) and
            against month-matched random onsets (N_NULL draws, one-sided).
  coupling  every November-April day since 1980 (the product's season): the next 30 days' mean index regressed on the
            100 hPa cap that day and the index's own mean over the five days before (the surface's persistence),
            winters resampled N_BOOT times for the intervals. The 100 hPa coefficient is the stratosphere's share, over
            and above persistence.
  outlook   that regression as a forecast of P(next 30 days' index > 0), each winter predicted from the others
            (leave-one-winter-out), Brier skill against each training set's base rate, on every day and on the days
            with the 100 hPa cap at least 1 sd from normal; winters resampled for the intervals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
import ssw_imprint as SI
from localdirs import env_dir
from strat_history import robust

FOLLOW = HERE / "reference" / "cmip6_follow.nc"
EVENTS = HERE / "reference" / "strat_history_events.json"
HISTORY = HERE / "reference" / "strat_history.nc"
BASE = (1991, 2020)
WINDOWS = ((1, 30), (31, 60), (1, 60))
WINTER = (11, 12, 1, 2, 3, 4)  # November-April, the season the monitor runs in
N_NULL, N_BOOT = 2000, 1000
LAGS = np.arange(-40, 91)


def pattern():
    """(pattern, robust mask, lat, lon, models, leave-one-out patterns): the models' mean days 1-60 composite, zero where
    it is not robust; and the same from all models but one, for testing a model on a pattern it did not help make."""
    F = xr.open_dataset(FOLLOW)
    X = F.psl.sel(set="ssw_all").mean("window").values  # (model, lat, lon), hPa; both windows are 30 days
    m, sig, _, _ = robust(X)
    loo = []
    for k in range(X.shape[0]):
        mk, sk, _, _ = robust(np.delete(X, k, axis=0))
        loo.append(np.where(sk, mk, 0.0))
    return np.where(sig, m, 0.0), sig, F.lat.values, F.lon.values, [str(x) for x in F.model.values], np.array(loo)


def era5_boxes():
    """(times, (time, lat, lon) hPa) ERA5 daily sea-level pressure on the pattern's boxes."""
    files = sorted(env_dir("ERA5_STORE", "wb2_1p5_daily", "slp").glob("slp_*.nc"))
    times, parts, boxes = [], [], None
    for f in files:
        a = xr.open_dataset(f).slp.transpose("time", "latitude", "longitude")
        if boxes is None:
            boxes = SI.Boxes(a.latitude.values, a.longitude.values)
        v = a.values.astype(float)
        if np.nanmedian(v) > 5000:  # some years may hold Pa as the attribute says
            v /= 100.0
        parts.append(boxes(v))
        times.append(pd.DatetimeIndex(a.time.values))
    return times[0].append(times[1:]), np.concatenate(parts)


def window_means(index: pd.Series, onsets, a0: int, a1: int) -> np.ndarray:
    return np.array([index[d + pd.Timedelta(days=a0) : d + pd.Timedelta(days=a1)].mean() for d in onsets])


def main() -> int:
    P, sig, lat, lon, models, loo = pattern()
    nmod = len(models)
    print(f"pattern: {int(sig.sum())} of {sig.size} boxes robust, {P.min():+.2f} to {P.max():+.2f} hPa", flush=True)
    t, S = era5_boxes()
    print(f"ERA5: {len(t)} days, {t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d}", flush=True)
    X = SI.design(t)
    base = (t.year >= BASE[0]) & (t.year <= BASE[1])
    V = S.reshape(len(t), -1)
    # the seasonal cycle from the base period, then each box's linear trend over the whole record
    harm = np.linalg.lstsq(X[base, :-1], V[base], rcond=None)[0]
    slope, icpt = np.polyfit(X[:, -1], V - X[:, :-1] @ harm, 1)
    harm[0] += icpt
    coef = np.vstack([harm, slope]).reshape(X.shape[1], *S.shape[1:])

    ds = xr.Dataset(
        {
            "pattern": (("lat", "lon"), P.astype("float32")),
            "robust": (("lat", "lon"), sig.astype("int8")),
            "baseline": (("term", "lat", "lon"), coef.astype("float64")),
            "pattern_loo": (("model", "lat", "lon"), loo.astype("float32")),
        },
        coords={"lat": lat, "lon": lon, "term": np.arange(X.shape[1]), "model": models},
    )
    ds.to_netcdf(SI.REF)  # the index below goes through the same Ref a forecast uses
    ref = SI.Ref()
    index = pd.Series(ref.index(t, S), index=t)
    winter = index[index.index.month.isin(WINTER)]
    print(f"index sd: all days {index.std():.2f}, Nov-Apr {winter.std():.2f}", flush=True)

    ev = json.loads(EVENTS.read_text())
    onsets = [pd.Timestamp(e["date"]) for e in ev["ssw"] if not e.get("marginal")]
    rng = np.random.default_rng(2026)
    pool = {m: index[(index.index.month == m) & (index.index.year >= 1980)].index for m in {d.month for d in onsets}}
    tests = {}
    for a0, a1 in WINDOWS:
        x = window_means(index, onsets, a0, a1)
        x = x[np.isfinite(x)]
        null = np.array(
            [
                np.nanmean(
                    window_means(index, [pool[d.month][rng.integers(len(pool[d.month]))] for d in onsets], a0, a1)
                )
                for _ in range(N_NULL)
            ]
        )
        tt = stats.ttest_1samp(x, 0.0)
        tests[f"d{a0}_{a1}"] = {
            "n": len(x),
            "mean": round(float(x.mean()), 3),
            "share_positive": round(float((x > 0).mean()), 3),
            "t_p": float(f"{tt.pvalue:.3g}"),
            "random_p": float(f"{(np.sum(null >= x.mean()) + 1) / (N_NULL + 1):.3g}"),
        }
        print(f"  after {len(x)} SSWs, days {a0}-{a1}: {tests[f'd{a0}_{a1}']}", flush=True)
    comp = np.array([[index.get(d + pd.Timedelta(days=int(k)), np.nan) for k in LAGS] for d in onsets])

    z = pd.Series(xr.open_dataset(HISTORY).z100s.values, index=pd.DatetimeIndex(xr.open_dataset(HISTORY).time.values))
    rows = []
    for d in z.index[z.index.month.isin(WINTER)]:
        if d + pd.Timedelta(days=30) > index.index[-1] or d.year < 1980:
            continue
        rows.append(
            (
                d,
                z[d],
                index[d - pd.Timedelta(days=5) : d].mean(),
                index[d + pd.Timedelta(days=1) : d + pd.Timedelta(days=30)].mean(),
            )
        )
    D = pd.DataFrame(rows, columns=["d", "z100", "i0", "i30"]).dropna()
    D["winter"] = D.d.dt.year - (D.d.dt.month < 7)

    def fit(frame):
        A = np.column_stack([np.ones(len(frame)), frame.z100, frame.i0])
        return np.linalg.lstsq(A, frame.i30, rcond=None)[0]

    b = fit(D)
    ws = D.winter.unique()
    by = {w: D[D.winter == w] for w in ws}
    boot = np.array([fit(pd.concat([by[w] for w in rng.choice(ws, len(ws))])) for _ in range(N_BOOT)])
    resid = D.i30 - np.column_stack([np.ones(len(D)), D.z100, D.i0]) @ b
    coupling = {
        "n_days": len(D),
        "n_winters": len(ws),
        "coef": [round(float(v), 3) for v in b],
        "ci95": [[round(float(v), 3) for v in np.percentile(boot[:, k], [2.5, 97.5])] for k in range(3)],
        "resid_sd": round(float(resid.std()), 3),
        "base_rate": round(float((D.i30 > 0).mean()), 3),
        "terms": ["intercept", "100 hPa cap (sd)", "index over days -5..0"],
    }
    print(f"  coupling: {coupling}", flush=True)

    # outlook skill, each winter left out in turn
    prob, clim = np.full(len(D), np.nan), np.full(len(D), np.nan)
    for w in ws:
        out = (D.winter == w).values
        tr = D[~out]
        bw = fit(tr)
        sd = (tr.i30 - np.column_stack([np.ones(len(tr)), tr.z100, tr.i0]) @ bw).std()
        mu = np.column_stack([np.ones(out.sum()), D.z100[out], D.i0[out]]) @ bw
        prob[out] = stats.norm.sf(0.0, mu, sd)
        clim[out] = (tr.i30 > 0).mean()
    D["hit"], D["prob"], D["clim"] = (D.i30 > 0).astype(float), prob, clim

    def bss(frame):
        return 1 - ((frame.prob - frame.hit) ** 2).mean() / ((frame.clim - frame.hit) ** 2).mean()

    outlook = {}
    for name, sel in (
        ("all_days", D.z100.abs() >= 0),
        ("cap_1sd", D.z100.abs() >= 1.0),
        ("cap_weak_1sd", D.z100 >= 1.0),
    ):
        F_ = D[sel.values]
        byw = {w: F_[F_.winter == w] for w in F_.winter.unique()}
        boot_s = [bss(pd.concat([byw[w] for w in rng.choice(list(byw), len(byw))])) for _ in range(N_BOOT)]
        outlook[name] = {
            "n_days": len(F_),
            "bss": round(float(bss(F_)), 3),
            "ci95": [round(float(v), 3) for v in np.percentile(boot_s, [2.5, 97.5])],
        }
    print(f"  outlook skill (leave-one-winter-out): {outlook}", flush=True)

    ds["index"] = ("time", index.values.astype("float32"))
    ds["index30"] = ("time", index.rolling(30).mean().values.astype("float32"))  # trailing 30-day mean
    ds["comp"] = (("event", "lag"), comp.astype("float32"))
    ds = ds.assign_coords(time=t, lag=LAGS, event=[f"{d:%Y-%m-%d}" for d in onsets])
    ds.attrs = {
        "title": "SSW surface-imprint index: ERA5 sea-level pressure projected on the CMIP6 post-SSW composite",
        "builder": "scripts/strat/build_ssw_imprint.py",
        "pattern": f"cmip6_follow.nc psl, set ssw_all, days 1-60, mean of {nmod} models, robust boxes only (FDR 10 %, "
        ">= 80 % sign agreement)",
        "baseline": f"ERA5 {BASE[0]}-{BASE[1]}, {SI.HARMONICS} annual harmonics on the 366-slot day of year, linear "
        f"trend per box over the whole record, years from {SI.EPOCH:%Y-%m-%d}",
        "units": "index: 1 = the CMIP6 composite's amplitude; pattern and baseline hPa",
        "imprint_tests": json.dumps(tests),
        "coupling": json.dumps(coupling),
        "outlook_skill": json.dumps(outlook),
        "sd7_by_month": json.dumps(
            [round(float(v), 3) for v in index.rolling(7, center=True).mean().groupby(index.index.month).std()]
        ),
    }
    ds.to_netcdf(SI.REF)
    print(f"wrote {SI.REF.relative_to(REPO)} ({SI.REF.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
