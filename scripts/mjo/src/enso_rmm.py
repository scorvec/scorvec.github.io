"""ENSO-removed wind-only RMM: an ENSO-cleaned MJO index shown beside the raw RMM.

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

The three-field RMM of the GEPS and GEFS pages (OLR + U850 + U200) has its own c, fitted the same way with the OLR
channel included (build_enso_rmm3.py -> data/reference/enso_rmm3_correction.json). It is kept as two parts, the OLR
channel's and the winds', because a product may filter the two with 120-day windows that end on different days.
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
CORR3 = REF / "enso_rmm3_correction.json"
ERSST_N34 = REF / "nina34.anom.data"
REPO = HERE.parents[2]
ENSO_DAILY = REPO / "assets" / "sst" / "data" / "enso_daily.json"
ROMI_URL = "https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt"
LAG_DAYS = 7  # the wind filter map's window typically ends about a week before init (ARCO lag + refresh cadence)
WINDOW = 120  # days of the RMM's low-frequency filter, over which dN34 takes its mean
BOM_ENSO_REMOVED_TO = pd.Timestamp("2013-12-31")  # BoM's RMM had its ENSO (SST1) part removed up to this day


def load_c(fitted: bool = False):
    """The wind-only correction, (RMM1, RMM2) per kelvin of dN34. As applied, RMM2's slope is 0: its fit (+0.068) is not
    significant (95 % interval -0.32..+0.32, 2,000 whole-year resamples over 2003-2022, build_enso_rmm3.py), as for the
    three-field RMM. fitted=True gives the fitted pair."""
    c = json.loads(CORR.read_text())["c"]
    return np.array([c["rmm1"], c["rmm2"] if fitted else 0.0], float)


def load_c3() -> dict[str, np.ndarray]:
    """The three-field RMM's correction as applied, {"olr": c, "wind": c}: each the (RMM1, RMM2) part per kelvin of
    dN34 that channel group carries. A component whose fit is not significant is 0 there (enso_rmm3_correction.json)."""
    d = json.loads(CORR3.read_text())["applied"]
    return {k: np.array([d[k]["rmm1"], d[k]["rmm2"]], float) for k in ("olr", "wind")}


def ersst_n34() -> pd.Series:
    """ERSST v6 Nino-3.4 anomaly (PSL nina34.anom.data, monthly) interpolated to daily between mid-months: the index
    both corrections were fitted on, and the one the hindcast scores use."""
    rows = []
    for ln in ERSST_N34.read_text().splitlines()[1:]:
        p = ln.split()
        if len(p) != 13 or not p[0].isdigit():
            continue
        for m, v in enumerate(p[1:], 1):
            if float(v) > -99:
                t0 = pd.Timestamp(int(p[0]), m, 1)
                rows.append((t0 + pd.Timedelta(days=(t0.days_in_month - 1) / 2), float(v)))
    s = pd.Series(dict(rows), dtype=float)
    days = pd.date_range(s.index[0].normalize() + pd.Timedelta(days=1), s.index[-1].normalize())
    return s.reindex(s.index.union(days)).interpolate("time").reindex(days)


def window_dn34(n34: pd.Series, days, ends) -> np.ndarray:
    """dN34 for a daily series: n34 on each of `days` minus its mean over the WINDOW days ending on the matching `ends`
    (inclusive). NaN where either is missing."""
    m = n34.rolling(WINDOW, min_periods=WINDOW).mean()
    return n34.reindex(pd.DatetimeIndex(days)).values - m.reindex(pd.DatetimeIndex(ends)).values


def hindcast_dn34(starts, valid) -> tuple[np.ndarray, np.ndarray]:
    """The ENSO terms of an RMM hindcast scored against BoM's RMM, from ERSST: the forecast's dN34 (S,), the
    Nino-3.4 at the start minus its mean over the WINDOW days before it (the hindcast's filter window), held at every
    lead as in the live forecast; and the truth's (S, L), the Nino-3.4 on each valid date minus its mean over the WINDOW
    days ending there, 0 up to BOM_ENSO_REMOVED_TO, where BoM had removed ENSO itself. valid: (S, L) dates."""
    n34 = ersst_n34()
    s = pd.DatetimeIndex(starts).normalize()
    v = pd.DatetimeIndex(np.asarray(valid, dtype="datetime64[D]").ravel())
    dn_f = window_dn34(n34, s, s - pd.Timedelta(days=1))
    dn_o = np.where(v > BOM_ENSO_REMOVED_TO, window_dn34(n34, v, v), 0.0).reshape(np.shape(valid))
    return dn_f, dn_o


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
    win = n34.loc[end - pd.Timedelta(days=119) : end]
    now = n34.loc[:t]
    if len(win) < 90 or now.empty:
        return float("nan")
    return float(now.iloc[-1] - win.mean())


def clean_forecast(rmm: xr.Dataset, dn: float, c: np.ndarray, label: str) -> xr.Dataset:
    """The cleaned wind-only forecast in the shape plot.plot_rmm expects (rmm1/rmm2 over member x lead_day)."""
    out = xr.Dataset(
        {"rmm1": rmm["rmm1_wind"] - dn * c[0], "rmm2": rmm["rmm2_wind"] - dn * c[1]}, attrs=dict(rmm.attrs)
    )
    out.attrs.update(model_label=label, channels="u850+u200 (ENSO removed)", enso_dn34=round(dn, 3))
    return out


def clean_obs(obs: xr.Dataset | None, n34: pd.Series, c: np.ndarray) -> xr.Dataset | None:
    if obs is None or obs.sizes.get("time", 0) == 0:
        return obs
    t = pd.to_datetime(obs["time"].values)
    dn = np.array([delta_n34(n34, x) for x in t])
    dn = np.where(np.isfinite(dn), dn, 0.0)
    return xr.Dataset(
        {"rmm1": ("time", obs["rmm1"].values - dn * c[0]), "rmm2": ("time", obs["rmm2"].values - dn * c[1])},
        coords={"time": obs["time"].values},
    )


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
    except Exception as e:  # noqa: BLE001
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
    W = R.loc[R.index.max() - pd.Timedelta(days=60) :]
    x, y = W.pc2.values, -W.pc1.values
    _, spd = _speed_amp(x, y, np.arange(len(x), dtype=float))
    last = R.iloc[-1]
    return {
        "date": f"{R.index.max():%Y-%m-%d}",
        "amp": round(float(last.amp), 2),
        "phase": int(phase_of(np.array([last.pc2]), np.array([-last.pc1]))[0]),
        "active_60d": round(float((W.amp >= 1).mean()), 2),
        "amp_60d": round(float(W.amp.mean()), 2),
        "speed_60d": round(float(np.median(spd)), 1),
    }


def diagnose(clean: xr.Dataset, clean_tail: xr.Dataset | None, dn: float, romi: dict | None) -> dict:
    m1 = clean["rmm1"].mean("member").values
    m2 = clean["rmm2"].mean("member").values
    lead = clean["lead_day"].values.astype(float)
    amp, spd = _speed_amp(m1, m2, lead)
    fc = {
        "amp_mean": round(float(amp.mean()), 2),
        "amp_max": round(float(amp.max()), 2),
        "speed": round(float(np.median(spd)), 1),
        "phase_start": int(phase_of(m1[:1], m2[:1])[0]),
        "phase_end": int(phase_of(m1[-1:], m2[-1:])[0]),
        "amp_start": round(float(amp[0]), 2),
        "amp_end": round(float(amp[-1]), 2),
        "days": int(np.rint(lead[-1])),
    }
    coherent = fc["amp_mean"] >= 1.0 and fc["speed"] >= 3.0
    out = {"dn34": round(dn, 2), "forecast": fc, "coherent_mjo_forecast": bool(coherent), "romi": romi}
    if clean_tail is not None and clean_tail.sizes.get("time", 0):
        t = pd.to_datetime(clean_tail["time"].values)
        sel = t >= t.max() - pd.Timedelta(days=60)
        a, s = _speed_amp(
            clean_tail["rmm1"].values[sel], clean_tail["rmm2"].values[sel], (t[sel] - t[sel][0]) / pd.Timedelta(days=1)
        )
        out["tail_60d"] = {
            "amp": round(float(a.mean()), 2),
            "active": round(float((a >= 1).mean()), 2),
            "speed": round(float(np.median(s)), 1),
        }
    return out


ENSO_STRONG_DN34 = 0.5  # |dN34| (K) above which the ENSO change visibly projects on the raw RMM: the cleaned view leads


def _phases(fc: dict) -> str:
    """' phase a -> b' for the coherent case, naming a phase only where the mean track's amplitude is at least 1 there
    (below 1 there is no MJO to have a phase); '' when neither end qualifies or the status predates amp_start/amp_end."""
    a0, a1 = fc.get("amp_start", 0.0), fc.get("amp_end", 0.0)
    if a0 >= 1.0 and a1 >= 1.0:
        return f", phase {fc['phase_start']} &rarr; {fc['phase_end']} over {fc['days']} days"
    if a0 >= 1.0:
        return f", from phase {fc['phase_start']}"
    if a1 >= 1.0:
        return f", reaching phase {fc['phase_end']} by day {fc['days']}"
    return ""


def raw_caption(st: dict) -> str:
    """The lead sentence of the raw RMM view: the ENSO-removed verdict, so the standard index is never shown without
    it. Names no phase (the cleaned view's status line does, under the amplitude rule)."""
    fc, dn = st["forecast"], st["dn34"]
    head = "<b>ENSO not removed.</b> "
    if st["coherent_mjo_forecast"]:
        return head + (
            f"With the ENSO signal taken out (&Delta;Ni&ntilde;o-3.4 {dn:+.2f}&nbsp;&deg;C) the forecast still shows an MJO: "
            f"ensemble-mean amplitude {fc['amp_mean']:.1f}, moving east at about {fc['speed']:+.0f}&deg;/day (see the "
            "ENSO-removed index)."
        )
    s = head + (
        f"With the ENSO signal taken out (&Delta;Ni&ntilde;o-3.4 {dn:+.2f}&nbsp;&deg;C) there is no coherent MJO in this "
        f"forecast: ensemble-mean amplitude {fc['amp_mean']:.1f} and a phase speed of {fc['speed']:+.1f}&deg;/day, where "
        "a real MJO moves at about +7&deg;/day."
    )
    if abs(dn) >= ENSO_STRONG_DN34:
        s += " A large or slow track in this diagram is most likely the ENSO change projecting onto the index, not an MJO."
    return s


def status_sentence(st: dict) -> str:
    fc = st["forecast"]
    if st["coherent_mjo_forecast"]:
        s = (
            f"The ENSO-removed forecast shows an MJO: ensemble-mean amplitude {fc['amp_mean']:.1f} moving east at about "
            f"{fc['speed']:+.0f}&deg;/day{_phases(fc)}."
        )
    else:
        s = (
            f"<b>No coherent MJO in the ENSO-removed forecast:</b> ensemble-mean amplitude {fc['amp_mean']:.1f} "
            f"(max {fc['amp_max']:.1f}) and a phase speed of {fc['speed']:+.1f}&deg;/day, where a real MJO moves at about "
            f"+7&deg;/day."
        )
    r = st.get("romi")
    if r:
        # below amplitude 1 there is no MJO to have a phase: say so rather than quote one
        now = (
            f"amplitude {r['amp']:.2f}, phase {r['phase']}"
            if r["amp"] >= 1.0
            else f"amplitude {r['amp']:.2f}, below the 1.0 of an active MJO, so there is no MJO right now"
        )
        s += (
            f" NOAA&rsquo;s OLR-based ROMI on {r['date']}: {now}; over the last "
            f"60 days {round(100 * r['active_60d'])}&nbsp;% of days reached amplitude 1."
        )
    s += f" Correction applied: &Delta;Ni&ntilde;o-3.4 = {st['dn34']:+.2f}&nbsp;&deg;C."
    return s
