#!/usr/bin/env python3
"""Reference for the checklist on ssw.html: what the days before the MERRA-2 sudden warmings since 1980 had that quiet
winter days did not, the threshold each ensemble is checked against, and how often past warmings and quiet windows met
it.

    python build_ssw_checklist.py      # -> reference/ssw_checklist.json

INPUTS. The Charlton-Polvani onsets of MERRA-2 (reference/strat_history_events.json); MERRA-2 u(60N) at 10 and 1 hPa
(build_strat_history.load_m2: the raw archive, its QC and gap filling); ERA5 100 hPa [v'T'] 45-75N, mean of 00 and 12
UTC (reference/vt100_era5.nc), against the ERA5 day-of-year normal the forecasts are read against
(reference/vt100_clim.nc, heatflux100.clim_at); ERA5 500 and 100 hPa heights (WeatherBench 2 1.5 deg daily means in
the local ERA5 store) over the wave compass's 55-65N band (build_wave_clim.band_coeff).

INGREDIENTS, each over the last L days before the onset (L = WINDOW unless a forecast covers fewer):
  source   the 500 hPa wave-1 size |c1|^2 against its normal (sd; NORMAL_YEARS, +-HALF days of the date), mean >= 0
  tilt     how far west the wave-1 ridge at 100 hPa sits of the ridge at 500 hPa (degrees longitude), mean >= 45
  flux     the 100 hPa [v'T'] against its normal (sd), mean >= +0.5
  upward   the 100 hPa [v'T'] above zero on every day: no waves coming back down
  slowing  the largest fall of u(60N, 10 hPa) within 10 days, the window and the onset day, >= 20 m/s
QUIET WINDOWS end on a November-March day at least CALM[0] days before and CALM[1] days after every onset.
TEST, per ingredient and L: the onsets meeting the threshold against the quiet windows' rate for the onsets' months,
one-sided binomial. The page lists an ingredient only when its WINDOW-day test passes, and ticks a shorter forecast
window only when that L's test passes too.
NOT ON THE LIST (`others`): the 500 hPa wave-2 size and tilt over the WINDOW days, the 10 hPa wind PRE days before the
onset (a weakened vortex beforehand) and the share of the WINDOW days with u(60N) weaker at 1 than at 10 hPa: the
onsets' mean against N_DRAW draws of quiet windows with the onsets' months, two-sided.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.degdays import doy366
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import heatflux100 as HF

EVENTS = HERE / "reference" / "strat_history_events.json"
VT = HERE / "reference" / "vt100_era5.nc"
OUT = HERE / "reference" / "ssw_checklist.json"
WINDOW = 15
MIN_DAYS = 3
CALM = (60, 30)
MONTHS = (11, 12, 1, 2, 3)
NORMAL_YEARS = (1991, 2020)
HALF = 15
PRE = (30, 16)
SLOW_SPAN = 10
THRESHOLDS = {"source": 0.0, "tilt": 45.0, "flux": 0.5, "upward": 0.0, "slowing": 20.0}
ALPHA = 0.05
N_DRAW = 20000


def normal(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """366-slot mean and sd of `s` over NORMAL_YEARS, each slot from the days within HALF of it."""
    base = s.loc[f"{NORMAL_YEARS[0]}" : f"{NORMAL_YEARS[1]}"].dropna()
    slot = doy366(base.index)
    mu, sd = np.empty(366), np.empty(366)
    for k in range(366):
        near = np.abs((slot - k + 183) % 366 - 183) <= HALF
        mu[k], sd[k] = base.values[near].mean(), base.values[near].std(ddof=1)
    return mu, sd


def zscore(s: pd.Series, mu: np.ndarray, sd: np.ndarray) -> pd.Series:
    i = doy366(s.index)
    return (s - mu[i]) / sd[i]


def waves() -> pd.DataFrame:
    """Daily ERA5 wave-1 and wave-2 band coefficients at 500 and 100 hPa (gpm)."""
    from build_wave_clim import band_coeff
    from localdirs import env_dir

    store = env_dir("ERA5_STORE", "wb2_1p5_daily", "zplev")
    parts = []
    for path in sorted(store.glob("zplev_*.nc")):
        with xr.open_dataset(path) as d:
            z = d["zplev"].sel(level=[500.0, 100.0], latitude=slice(54, 66)).load()  # the store's zplev holds metres
        cols = {}
        for lev in (500, 100):
            zz = z.sel(level=float(lev)).transpose("time", "latitude", "longitude")
            c = band_coeff(zz.values.astype("float64"), zz.latitude.values)
            cols |= {f"c{lev}_{k}": c[:, j] for j, k in enumerate((1, 2))}
        parts.append(pd.DataFrame(cols, index=pd.DatetimeIndex(z.time.values).normalize()))
    w = pd.concat(parts)
    w = w[~w.index.duplicated()].asfreq("D")
    if not 20 < np.abs(w["c500_1"]).median() < 300:
        raise SystemExit(f"500 hPa wave-1 amplitude median {np.abs(w['c500_1']).median():.0f}: not metres")
    return w


def tilt(c500: pd.Series, c100: pd.Series, k: int) -> pd.Series:
    """Degrees longitude the 100 hPa ridge sits west of the 500 hPa ridge (ridge at -arg(c)/k)."""
    return pd.Series(np.angle(c100 * np.conj(c500), deg=True) / k, c500.index)


def largest_fall(u: pd.Series) -> float:
    """The largest fall of u from one day to a later day at most SLOW_SPAN days on."""
    u = u.asfreq("D")
    return float(max((u.shift(lag) - u).max() for lag in range(1, SLOW_SPAN + 1)))


VALUE = {  # ingredient -> (series key, the window's value); ends at the day before the onset, slowing at the onset
    "source": ("source", lambda x: x.mean()),
    "tilt": ("tilt", lambda x: x.mean()),
    "flux": ("flux", lambda x: x.mean()),
    "upward": ("vt", lambda x: x.min()),
    "slowing": ("u10", largest_fall),
}


def window(s: pd.Series, end: pd.Timestamp, days: int, through: bool = False) -> pd.Series | None:
    """The `days` days before `end` (and `end` itself when `through`), or None when any is missing."""
    x = s.loc[end - pd.Timedelta(days=days) : end - pd.Timedelta(days=0 if through else 1)]
    return None if len(x) < days + through or x.isna().any() else x


def meets(key: str, v: float) -> bool:
    return bool(v > THRESHOLDS[key] if key == "upward" else v >= THRESHOLDS[key])


def quiet_ends(onsets: list[pd.Timestamp], days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ok = np.isin(days.month, MONTHS)
    for e in onsets:
        ok &= ~((days > e - pd.Timedelta(days=CALM[0])) & (days < e + pd.Timedelta(days=CALM[1])))
    return days[ok]


def month_rate(values: dict, onset_months, key) -> float:
    """The quiet windows' rate of meeting the threshold, weighted to the onsets' months."""
    by = {}
    for t, v in values.items():
        by.setdefault(t.month, []).append(meets(key, v))
    return float(np.mean([np.mean(by[m]) for m in onset_months]))


def main() -> int:
    import build_strat_history as B

    ev = json.loads(EVENTS.read_text())
    onsets = sorted(pd.Timestamp(x) for w in ev["winters"] for x in w["ssw"])
    d = B.load_m2()
    t = pd.DatetimeIndex(d.time.values)
    S = {"u10": pd.Series(d.u10.values.astype(float), t),
         "u1": pd.Series(d.u60.sel(lev=1.0).values.astype(float), t)}  # fmt: skip
    with xr.open_dataset(VT) as v:
        vt = v["nh_tot"].mean("hour").to_series()
    vt.index = pd.DatetimeIndex(vt.index).normalize()
    S["vt"] = vt.asfreq("D")
    with xr.open_dataset(HF.CLIM) as clim:
        S["flux"] = (S["vt"] - HF.clim_at(clim, "nh", "tot", "mean", S["vt"].index)) / HF.clim_at(
            clim, "nh", "tot", "sd", S["vt"].index
        )
    w = waves()
    mu, sd = normal(np.abs(w["c500_1"]) ** 2)
    S["source"] = zscore(np.abs(w["c500_1"]) ** 2, mu, sd)
    S["tilt"] = tilt(w["c500_1"], w["c100_1"], 1)
    mu2, sd2 = normal(np.abs(w["c500_2"]) ** 2)
    S["source2"] = zscore(np.abs(w["c500_2"]) ** 2, mu2, sd2)
    S["tilt2"] = tilt(w["c500_2"], w["c100_2"], 2)
    S["u10z"] = zscore(S["u10"], *normal(S["u10"]))
    S["weaker_aloft"] = (S["u1"] < S["u10"]).astype(float)

    items = {}
    for key, (src, f) in VALUE.items():
        s = S[src]
        through = key == "slowing"
        ends = quiet_ends(onsets, s.dropna().index)
        by_days, used = {}, None
        for L in range(WINDOW, MIN_DAYS - 1, -1):
            ev_v = {e: window(s, e, L, through) for e in onsets}
            ev_v = {e: f(x) for e, x in ev_v.items() if x is not None}
            if used is None:
                used = sorted(ev_v)  # the onsets with the full window: the same set at every L
            ev_v = {e: ev_v[e] for e in used}
            q_v = {q: window(s, q, L, through) for q in ends}
            q_v = {q: f(x) for q, x in q_v.items() if x is not None}
            hits = int(sum(meets(key, v) for v in ev_v.values()))
            rate = month_rate(q_v, [e.month for e in used], key)
            p = stats.binomtest(hits, len(used), rate, alternative="greater").pvalue
            by_days[str(L)] = {"hits": hits, "n": len(used), "quiet": round(rate, 3), "p": float(f"{p:.3g}")}
        items[key] = {
            "threshold": THRESHOLDS[key],
            "onsets": [f"{e:%Y-%m-%d}" for e in used],
            "quiet_windows": len(ends),
            "by_days": by_days,
            "passes": by_days[str(WINDOW)]["p"] < ALPHA,
        }
        b = by_days[str(WINDOW)]
        print(f"{key:8s} {b['hits']}/{b['n']} onsets vs {b['quiet']:.0%} of quiet windows, p {b['p']:.2g}"
              f"  (shortest passing window: {min(int(L) for L, x in by_days.items() if x['p'] < ALPHA)} d)")  # fmt: skip

    rng = np.random.default_rng(20261030)
    others = {}
    for key, src, (a, b) in (
        ("wave2_size", "source2", (WINDOW, 1)),
        ("wave2_tilt", "tilt2", (WINDOW, 1)),
        ("weak_before", "u10z", PRE),
        ("weaker_aloft", "weaker_aloft", (WINDOW, 1)),
    ):
        # the mean over days a..b before each window's end
        s, lag, days = S[src], pd.Timedelta(days=b - 1), a - b + 1
        ev_v = {e: float(x.mean()) for e in onsets if (x := window(s, e - lag, days)) is not None}
        q_v = {
            q: float(x.mean())
            for q in quiet_ends(onsets, s.dropna().index)
            if (x := window(s, q - lag, days)) is not None
        }
        by = {m: np.array([v for q, v in q_v.items() if q.month == m]) for m in MONTHS}
        months = [e.month for e in ev_v]
        null = np.mean([rng.choice(by[m], N_DRAW) for m in months], axis=0)
        obs = float(np.mean(list(ev_v.values())))
        p = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (N_DRAW + 1)
        others[key] = {"onsets": len(ev_v), "mean": round(obs, 2), "quiet": round(float(null.mean()), 2),
                       "days_before": [a, b], "p": round(float(p), 4)}  # fmt: skip
        print(f"{key:13s} onsets {obs:+.2f} vs quiet {null.mean():+.2f}, p {p:.3g}")

    OUT.write_text(json.dumps({
        "built": f"{pd.Timestamp.now('UTC'):%Y-%m-%d}",
        "definitions": {"window": WINDOW, "min_days": MIN_DAYS, "calm": list(CALM), "months": list(MONTHS),
                        "normal_years": list(NORMAL_YEARS), "half": HALF, "slow_span": SLOW_SPAN, "alpha": ALPHA,
                        "n_draw": N_DRAW},
        "sources": {"source": "ERA5 500 hPa, 55-65N", "tilt": "ERA5 500 and 100 hPa, 55-65N",
                    "flux": "ERA5 100 hPa [v'T'], 45-75N", "upward": "ERA5 100 hPa [v'T'], 45-75N",
                    "slowing": "MERRA-2 u(60N, 10 hPa)"},
        "normal_source": {"mean": [round(float(x), 1) for x in mu], "sd": [round(float(x), 1) for x in sd]},
        "items": items,
        "others": others,
    }, indent=1))  # fmt: skip
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
