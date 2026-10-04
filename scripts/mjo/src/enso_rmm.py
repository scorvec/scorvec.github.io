"""ENSO-removed wind-only RMM (2026-09-27; user asked for an ENSO-cleaned MJO index beside the raw RMM).

Why. Since 2014 the Bureau of Meteorology's RMM removes only the previous-120-day mean; before that (1974-2013) the
part linearly related to ENSO (their SST1) was removed first (Wheeler & Hendon 2004). This page's wind-only RMM uses
the 120-day filter only. While an El Nino is GROWING, the 120-day mean lags the warming and the standing convection
and wind anomaly over the central Pacific project onto phases 6-8/1, which reads as a stalled MJO (Sep 2026: phases
6-8/1 for 60 days at +0.6 deg/day, where a real MJO moves at about +7 deg/day).

What. RMM_clean = RMM_wind - dN34 * c. dN34 is Nino-3.4 minus its mean over the 120 days of the wind filter window,
and c is the projection onto the two wind EOFs of the per-longitude regression of the 120-day-filtered u850/u200
anomalies on dN34, fitted on ARCO/ERA5 2003-2024 (data/reference/enso_rmm_correction.json). Validation against BoM's
ENSO-removed RMM, 2003-2013: r(RMM1) 0.945 -> 0.963, r(RMM2) 0.960 -> 0.961; the index's correlation with dN34 falls
from -0.21 to -0.01. The forecast holds Nino-3.4 at its latest value (ENSO changes little in 15 days).

The linear correction removes part of a growing El Nino's projection, not all of it, so the page also shows NOAA's
OLR-based real-time OMI (ROMI; 40-day mean removed, which suppresses slow signals) as an independent check.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "data" / "reference"
CORR = REF / "enso_rmm_correction.json"
REPO = HERE.parents[2]
ENSO_DAILY = REPO / "assets" / "sst" / "data" / "enso_daily.json"
ROMI_URL = "https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt"
LAG_DAYS = 7          # the wind filter map's window typically ends about a week before init (ARCO lag + refresh cadence)


def load_c():
    d = json.loads(CORR.read_text())
    return np.array([d["c"]["rmm1"], d["c"]["rmm2"]], float)


def n34_smoothed() -> pd.Series | None:
    """Daily OISST Nino-3.4 anomaly, 31-day trailing mean (the fit used monthly ERSST, so day-to-day noise is
    smoothed out). None when the file is missing, so the caller can skip the cleaned index."""
    if not ENSO_DAILY.exists():
        return None
    d = json.loads(ENSO_DAILY.read_text())["daily"]
    s = pd.Series(d["nino34"], index=pd.to_datetime(d["dates"]), dtype=float).asfreq("D").interpolate()
    return s.rolling(31, min_periods=20).mean().dropna()


def delta_n34(n34: pd.Series, t, window_end=None) -> float:
    """N34 at t (latest value at or before t, i.e. persisted into the forecast) minus the mean over the 120 days
    ending at window_end (the wind filter's window) or t - LAG_DAYS."""
    t = pd.Timestamp(t).tz_localize(None).normalize()
    end = pd.Timestamp(window_end).normalize() if window_end else t - pd.Timedelta(days=LAG_DAYS)
    win = n34.loc[end - pd.Timedelta(days=119): end]
    now = n34.loc[:t]
    if len(win) < 90 or now.empty:
        return float("nan")
    return float(now.iloc[-1] - win.mean())


def clean_forecast(rmm: xr.Dataset, dn: float, c: np.ndarray, label: str) -> xr.Dataset:
    """The cleaned wind-only forecast in the shape plot.plot_rmm expects (rmm1/rmm2 over member x lead_day)."""
    out = xr.Dataset({"rmm1": rmm["rmm1_wind"] - dn * c[0], "rmm2": rmm["rmm2_wind"] - dn * c[1]},
                     attrs=dict(rmm.attrs))
    out.attrs.update(model_label=label, channels="u850+u200 (ENSO removed)", enso_dn34=round(dn, 3))
    return out


def clean_obs(obs: xr.Dataset | None, n34: pd.Series, c: np.ndarray) -> xr.Dataset | None:
    if obs is None or obs.sizes.get("time", 0) == 0:
        return obs
    t = pd.to_datetime(obs["time"].values)
    dn = np.array([delta_n34(n34, x) for x in t])
    dn = np.where(np.isfinite(dn), dn, 0.0)
    return xr.Dataset({"rmm1": ("time", obs["rmm1"].values - dn * c[0]),
                       "rmm2": ("time", obs["rmm2"].values - dn * c[1])}, coords={"time": obs["time"].values})


def _speed_amp(x1, x2, days):
    amp = np.hypot(x1, x2)
    ang = np.unwrap(np.arctan2(x2, x1))
    step = np.diff(days) if len(days) > 1 else np.array([1.0])
    spd = np.degrees(np.diff(ang)) / np.where(step > 0, step, 1.0)
    return amp, spd


def phase_of(x1, x2):
    ang = (np.degrees(np.arctan2(x2, x1)) + 360.0) % 360.0
    return (((ang - 180.0) % 360.0) // 45.0).astype(int) + 1


def romi_status(timeout=40):
    """NOAA PSL real-time OMI (CPC OLR). Plotted as (PC2, -PC1) to match the RMM orientation. None if unreachable."""
    try:
        req = urllib.request.Request(ROMI_URL, headers={"User-Agent": "scorvec.com MJO page"})
        txt = urllib.request.urlopen(req, timeout=timeout).read().decode()
    except Exception as e:                                                      # noqa: BLE001
        print(f"  ROMI fetch failed ({repr(e)[:60]})")
        return None
    rows = []
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) >= 7 and p[0].isdigit():
            rows.append((pd.Timestamp(int(p[0]), int(p[1]), int(p[2])), float(p[4]), float(p[5]), float(p[6])))
    if not rows:
        return None
    R = pd.DataFrame(rows, columns=["t", "pc1", "pc2", "amp"]).set_index("t").sort_index()
    W = R.loc[R.index.max() - pd.Timedelta(days=60):]
    x, y = W.pc2.values, -W.pc1.values
    _, spd = _speed_amp(x, y, np.arange(len(x), dtype=float))
    last = R.iloc[-1]
    return {"date": f"{R.index.max():%Y-%m-%d}", "amp": round(float(last.amp), 2),
            "phase": int(phase_of(np.array([last.pc2]), np.array([-last.pc1]))[0]),
            "active_60d": round(float((W.amp >= 1).mean()), 2), "amp_60d": round(float(W.amp.mean()), 2),
            "speed_60d": round(float(np.median(spd)), 1)}


def diagnose(clean: xr.Dataset, clean_tail: xr.Dataset | None, dn: float, romi: dict | None) -> dict:
    m1 = clean["rmm1"].mean("member").values
    m2 = clean["rmm2"].mean("member").values
    lead = clean["lead_day"].values.astype(float)
    amp, spd = _speed_amp(m1, m2, lead)
    fc = {"amp_mean": round(float(amp.mean()), 2), "amp_max": round(float(amp.max()), 2),
          "speed": round(float(np.median(spd)), 1), "phase_start": int(phase_of(m1[:1], m2[:1])[0]),
          "phase_end": int(phase_of(m1[-1:], m2[-1:])[0]), "days": int(round(lead[-1]))}
    coherent = fc["amp_mean"] >= 1.0 and fc["speed"] >= 3.0
    out = {"dn34": round(dn, 2), "forecast": fc, "coherent_mjo_forecast": bool(coherent), "romi": romi}
    if clean_tail is not None and clean_tail.sizes.get("time", 0):
        t = pd.to_datetime(clean_tail["time"].values)
        sel = t >= t.max() - pd.Timedelta(days=60)
        a, s = _speed_amp(clean_tail["rmm1"].values[sel], clean_tail["rmm2"].values[sel],
                          (t[sel] - t[sel][0]) / pd.Timedelta(days=1))
        out["tail_60d"] = {"amp": round(float(a.mean()), 2), "active": round(float((a >= 1).mean()), 2),
                           "speed": round(float(np.median(s)), 1)}
    return out


def status_sentence(st: dict) -> str:
    fc = st["forecast"]
    if st["coherent_mjo_forecast"]:
        s = (f"The ENSO-removed forecast shows an MJO: ensemble-mean amplitude {fc['amp_mean']:.1f} moving east at about "
             f"{fc['speed']:+.0f}&deg;/day, phase {fc['phase_start']} &rarr; {fc['phase_end']} over {fc['days']} days.")
    else:
        s = (f"<b>No coherent MJO in the ENSO-removed forecast:</b> ensemble-mean amplitude {fc['amp_mean']:.1f} "
             f"(max {fc['amp_max']:.1f}) and a phase speed of {fc['speed']:+.1f}&deg;/day, where a real MJO moves at about "
             f"+7&deg;/day.")
    r = st.get("romi")
    if r:
        s += (f" NOAA&rsquo;s OLR-based ROMI on {r['date']}: amplitude {r['amp']:.2f} (phase {r['phase']}); over the last "
              f"60 days {int(round(100 * r['active_60d']))}&nbsp;% of days reached amplitude 1.")
    s += f" Correction applied: &Delta;Ni&ntilde;o-3.4 = {st['dn34']:+.2f}&nbsp;K."
    return s
