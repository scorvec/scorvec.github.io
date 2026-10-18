#!/usr/bin/env python3
"""High-fidelity 2026 AAM budget: does the El Nino AAM rise close against the surface torques?

Inputs (committed by .github/workflows/torque-history.yml, nothing downloaded here):
  scripts/mjo/data/reference/torque_1h_2026.csv.gz    ERA5 HOURLY 2026-03-01 -> latest ERA5T:
        mountain / friction / GWD torques (global, bands, ranges), relative (motion) AAM from the
        vertical integral of eastward mass flux, mass-term AAM from p_s
  scripts/mjo/data/reference/torque_daily_era5.csv.gz ERA5 4x-daily history 1959 -> (climatologies,
        other years' windows)
  GFZ ESMGFZ 3-hourly effective AAM functions (ECMWF operational analysis), the external reference:
        aam_validation/esmgfz_2026.asc (aam.py's cache)

Budget (total AAM = motion + mass term; only surface torques change it):
  dM/dt = T_mountain + T_friction + T_gwd + residual
Hourly: dM/dt over [t, t+1h] against the trapezoid mountain torque and the 1-h mean stresses stamped
t+1h (ERA5 mean rates are over the hour ending at the stamp). The residual collects analysis
increments (ERA5 4D-Var windows 09-21 / 21-09 UTC), numerical non-conservation and the
difference between the model's own torques and torques recomputed on the 0.25-degree output grid.

    python scripts/mjo/src/torque_budget_2026.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from localdirs import env_dir
import torque_enso_study as S
from doycal import noleap_doy

REF = S.REF
H1 = REF / "torque_1h_2026.csv.gz"
GFZ_CHI3_MEAN = 2.734436408977633e-8  # aam.py: add to chi3_motion
GFZ_TO_KGM2S = 5.2092e33  # C_m * Omega
PER_H = 1e25 / 1e18 / 3600.0  # (1e25 kg m2/s per hour) -> Hadley
HD = S.HADLEY_DAY_TO_AAM
W0, W1 = "2026-04-15", "2026-09-19"  # the rise analysed in the main session
OTHER = ["HIM", "ROC", "GRL", "ANT", "ALP"]


def load_hourly():
    h = pd.read_csv(H1, parse_dates=["time"]).set_index("time").sort_index()
    h["M"] = h["aam_glob"] + h["aamm_glob"]
    h["T"] = h["mtn_glob"] + h["fric_glob"] + h["gwd_glob"]
    return h


def interval_budget(h: pd.DataFrame, step_h: int):
    """Budget over consecutive samples `step_h` hours apart, using only those samples (as a
    3-hourly or 12-hourly product would). Torque over an interval: trapezoid of the instantaneous
    mountain torque; the stamped mean stresses (1-h means) averaged over the interval's end samples."""
    s = h[(h.index.hour % step_h) == 0]
    dt = np.diff(s.index.values).astype("timedelta64[h]").astype(float)
    ok = dt == step_h
    dM = np.diff(s["M"].values)[ok] / step_h * PER_H
    dMr = np.diff(s["aam_glob"].values)[ok] / step_h * PER_H
    dMm = np.diff(s["aamm_glob"].values)[ok] / step_h * PER_H
    mt = 0.5 * (s["mtn_glob"].values[:-1] + s["mtn_glob"].values[1:])[ok]
    fr = 0.5 * (s["fric_glob"].values[:-1] + s["fric_glob"].values[1:])[ok]
    gw = 0.5 * (s["gwd_glob"].values[:-1] + s["gwd_glob"].values[1:])[ok]
    if step_h == 1:  # exact: the stamped 1-h means
        fr = s["fric_glob"].values[1:][ok]
        gw = s["gwd_glob"].values[1:][ok]
    T = mt + fr + gw
    end = s.index[1:][ok]
    return pd.DataFrame(
        {"dMdt": dM, "dMrdt": dMr, "dMmdt": dMm, "mtn": mt, "fric": fr, "gwd": gw, "T": T, "R": dM - T, "Rr": dMr - T},
        index=end,
    )


def window_hours(b: pd.DataFrame):
    """Residual by end hour + inside-window vs window-crossing comparison (4D-Var 09-21 / 21-09 UTC)."""
    g = b.groupby(b.index.hour)["R"].agg(["mean", "std", "count"])
    return g


def welch(x, fs=24.0, nper=24 * 32):
    from scipy.signal import welch as w

    f, p = w(x - np.nanmean(x), fs=fs, nperseg=min(nper, len(x)), detrend="linear")
    return f, p  # f in cycles/day


def load_gfz():
    rows = []
    for line in env_dir("MJO_CACHE", "aam_validation", "esmgfz_2026.asc").read_text().splitlines():
        c = line.split()
        if len(c) < 10 or not (c[0].isdigit() and len(c[0]) == 4):
            continue
        try:
            t = pd.Timestamp(int(c[0]), int(c[1]), int(c[2]), int(c[3]))
            rows.append((t, float(c[-4]), float(c[-1])))  # chi3_mass, chi3_motion
        except ValueError:
            continue
    g = pd.DataFrame(rows, columns=["time", "chi3_mass", "chi3_motion"]).set_index("time")
    g["Mr"] = (g.chi3_motion + GFZ_CHI3_MEAN) * GFZ_TO_KGM2S / 1e25
    g["Mm"] = g.chi3_mass * GFZ_TO_KGM2S / 0.756 / 1e25
    g["M"] = g.Mr + g.Mm
    return g


def daily_clims(span=(1991, 2020)):
    """4x-daily climatologies (harmonic, 1991-2020) of the history columns, keyed by column, evaluated
    as functions of the date, plus the 12Z-only climatologies of the global per-hour columns."""
    d = S.load_era5()
    return d


def clim_on(dhist: pd.DataFrame, col: str, idx: pd.DatetimeIndex):
    _, fit = S.harmonic_anom(dhist[col])
    # evaluate the fitted harmonic climatology on arbitrary dates via the common-year day (1 March = 60 in every year,
    # 29 Feb = 59.5), so a date is matched to the same calendar day of the other years
    doy = fit.groupby(noleap_doy(fit.index)).mean()
    return pd.Series(np.interp(noleap_doy(idx), doy.index.values, doy.values), index=idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    res = {}
    h = load_hourly()
    res["span"] = [str(h.index[0]), str(h.index[-1]), len(h)]
    gaps = np.diff(h.index.values).astype("timedelta64[h]").astype(int)
    res["hourly_gaps"] = int((gaps != 1).sum())

    # ---------- hourly / 3-hourly / 6-hourly / 12-hourly closures (absolute) ----------
    clos = {}
    for st in (1, 3, 6, 12, 24):
        b = interval_budget(h, st)
        sub = b.loc["2026-03-02":]
        r = sub["R"]
        clos[st] = dict(
            mean_R=float(r.mean()),
            sd_R=float(r.std()),
            mean_Rmotion_only=float(sub["Rr"].mean()),
            sd_Rmotion_only=float(sub["Rr"].std()),
            n=len(r),
            corr_dMdt_T=float(np.corrcoef(sub.dMdt, sub["T"])[0, 1]),
            sd_dMdt=float(sub.dMdt.std()),
            sd_T=float(sub["T"].std()),
        )
        if st == 1:
            hb = b
    res["closure_by_step"] = clos
    gh = window_hours(hb)
    res["hourly_residual_by_end_hour"] = gh.round(3).reset_index().to_dict(orient="records")
    # 4D-Var windows: compare the interval ending at the hour after each boundary to the rest
    jump_hours = gh["std"].sort_values(ascending=False).index[:2].tolist()
    inside = hb[~hb.index.hour.isin(jump_hours)]["R"]
    across = hb[hb.index.hour.isin(jump_hours)]["R"]
    res["window_test"] = dict(
        jump_end_hours=[int(x) for x in jump_hours],
        inside_mean=float(inside.mean()),
        inside_sd=float(inside.std()),
        across_mean=float(across.mean()),
        across_sd=float(across.std()),
        note="end hour h = the interval (h-1, h] UTC",
    )
    # the same test on a 3-hourly product (00, 03, ... 21 UTC, like GFZ's sampling)
    b3 = interval_budget(h, 3)
    g3 = b3.groupby(b3.index.hour)["R"].agg(["mean", "std", "count"])
    res["three_hourly_residual_by_end_hour"] = g3.round(3).reset_index().to_dict(orient="records")
    across3 = b3.index.hour.isin([12, 0])  # (09,12] and (21,24] contain 09->10 / 21->22
    res["three_hourly_window_test"] = dict(
        inside_mean=float(b3.R[~across3].mean()),
        inside_sd=float(b3.R[~across3].std()),
        across_mean=float(b3.R[across3].mean()),
        across_sd=float(b3.R[across3].std()),
    )
    # increment size per window: (residual at the jump hour - the inside-window mean) x 1 h, in 1e25
    inc = {int(hh): float((gh.loc[hh, "mean"] - inside.mean()) * 3600 * 1e18 / 1e25) for hh in jump_hours}
    res["increment_per_window_1e25"] = inc
    res["residual_by_month"] = {f"{k:%Y-%m}": float(v) for k, v in hb["R"].resample("MS").mean().round(2).items()}
    # residual spectrum
    f, p = welch(hb["R"].values)
    _fT, pT = welch(hb["T"].values)
    _fM, pM = welch(hb["dMdt"].values)
    pd.DataFrame({"cpd": f, "psd_R": p, "psd_T": pT, "psd_dMdt": pM}).to_csv(
        od / "budget2026_spectrum.csv", index=False
    )

    # daily closure from hourly (00Z to 00Z)
    d = pd.DataFrame(
        {"T": hb["T"].resample("D").mean(), "dMdt": hb["dMdt"].resample("D").mean(), "n": hb["T"].resample("D").count()}
    )
    d = d[d.n == 24]
    d["R"] = d.dMdt - d["T"]
    res["daily_from_hourly"] = dict(
        mean_R=float(d.R.mean()), sd_R=float(d.R.std()), n=len(d), lag1=float(d.R.autocorr(1))
    )

    # ---------- the Apr 15 - Sep 19 rise: closure variants ----------
    dh = S.load_era5()
    w0, w1 = pd.Timestamp(W0), pd.Timestamp(W1)
    var = {}

    # (a) 12Z-daily, motion only, ANOMALY (the earlier budget): per-hour 12Z climatologies from the history
    def anom12(col):
        x = dh[col]
        an, _ = S.harmonic_anom(x)
        return an

    T12 = anom12("mtn_glob_h12") + anom12("fric_glob_h12") + anom12("gwd_glob_h12")
    Mr12 = anom12("aam_glob_h12")
    Mm12 = anom12("aamm_glob_h12") if "aamm_glob_h12" in dh else None

    def wint(series):  # integral over [w0, w1] in 1e25
        return float(series.loc[w0:w1].sum() * HD)

    def wdelta(series):  # change between the window edges
        return float(series.loc[w1 + pd.Timedelta(days=1)] - series.loc[w0])

    var["12Z daily, motion only, anomaly"] = dict(torque=wint(T12), dM=wdelta(Mr12))
    if Mm12 is not None:
        var["12Z daily, motion+mass, anomaly"] = dict(torque=wint(T12), dM=wdelta(Mr12 + Mm12))
    # (b) 4x daily means (history), motion+mass, anomaly
    A4 = {
        c: S.harmonic_anom(dh[c])[0]
        for c in ("mtn_glob", "fric_glob", "gwd_glob", "aam_glob") + (("aamm_glob",) if "aamm_glob" in dh else ())
    }
    T4 = A4["mtn_glob"] + A4["fric_glob"] + A4["gwd_glob"]
    M4 = A4["aam_glob"] + A4.get("aamm_glob", 0.0)
    # daily-mean AAM is centred ~09Z; the window edge change uses day means (w0 .. w1+1)
    var["4x daily, motion+mass, anomaly"] = dict(torque=wint(T4), dM=wdelta(M4))
    # (c) hourly and 3-hourly ABSOLUTE closures over the same window
    for st in (1, 3):
        b = interval_budget(h, st)
        sub = b.loc[w0 : w1 + pd.Timedelta(hours=23)]
        var[f"{st}-hourly, motion+mass, absolute"] = dict(
            torque=float(sub["T"].sum() * st * 3600 * 1e18 / 1e25),
            dM=float(sub["dMdt"].sum() * st * 3600 * 1e18 / 1e25),
        )
    # (d) hourly-based ANOMALY: daily means of the hourly torques minus the 4x-daily climatology
    #     (the 24x vs 4x sampling offset measured in 2026 itself and reported)
    hd = h[["mtn_glob", "fric_glob", "gwd_glob", "aam_glob", "aamm_glob"]].resample("D").mean()
    h4 = (
        h[h.index.hour.isin([0, 6, 12, 18])][["mtn_glob", "fric_glob", "gwd_glob", "aam_glob", "aamm_glob"]]
        .resample("D")
        .mean()
    )
    res["sampling_offset_24x_minus_4x_2026"] = (hd - h4).mean().round(3).to_dict()
    clim = {c: clim_on(dh, c, hd.index) for c in hd.columns if c in dh}
    Th = sum(hd[c] - clim[c] for c in ("mtn_glob", "fric_glob", "gwd_glob"))
    Mh = (hd["aam_glob"] - clim["aam_glob"]) + ((hd["aamm_glob"] - clim["aamm_glob"]) if "aamm_glob" in clim else 0)
    var["hourly daily-means, motion+mass, anomaly vs 4x clim"] = dict(torque=wint(Th), dM=wdelta(Mh))
    for v in var.values():
        v["residual"] = v["dM"] - v["torque"]
        v["residual_mean_hadley"] = v["residual"] / (((w1 - w0).days + 1) * HD)
    res["window_variants"] = var

    # GFZ comparison (3-hourly, ECMWF operational)
    try:
        g = load_gfz()
        gd = g["M"].resample("D").mean()
        res["gfz"] = dict(
            dM_window_total=float(gd.loc[w1 + pd.Timedelta(days=1)] - gd.loc[w0]),
            dMr_window=float(
                g["Mr"].resample("D").mean().loc[w1 + pd.Timedelta(days=1)] - g["Mr"].resample("D").mean().loc[w0]
            ),
            dMm_window=float(
                g["Mm"].resample("D").mean().loc[w1 + pd.Timedelta(days=1)] - g["Mm"].resample("D").mean().loc[w0]
            ),
            corr_daily_dMdt_vs_era5=float(
                np.corrcoef(
                    gd.diff().reindex(d.index).dropna(),
                    (hd["aam_glob"] + hd["aamm_glob"]).diff().reindex(gd.diff().reindex(d.index).dropna().index),
                )[0, 1]
            ),
            note="GFZ removes the surface-pressure tides and applies an inverted-barometer correction "
            "over the ocean to the mass term; ERA5 here is the atmosphere's own mass term",
        )
        era5_dM = float(
            (hd["aam_glob"] + hd["aamm_glob"]).loc[w1 + pd.Timedelta(days=1)]
            - (hd["aam_glob"] + hd["aamm_glob"]).loc[w0]
        )
        res["gfz"]["era5_dM_window_total_absolute"] = era5_dM
    except Exception as e:  # noqa: BLE001
        res["gfz"] = {"error": str(e)}

    # ---------- attribution of the rise (anomalies vs 4x 1991-2020 clim, hourly 2026 daily means) ----------
    terms = {
        "Andes": ["mtn_AND"],
        "other ranges": [f"mtn_{r}" for r in OTHER],
        "rest of mountain": None,
        "friction tropics": ["fric_b3000s", "fric_b0030n"],
        "friction NH ext": ["fric_b3060n", "fric_b6090n"],
        "friction SH ext": ["fric_b9060s", "fric_b6030s"],
        "GWD": ["gwd_glob"],
    }
    cols_needed = sorted({c for v in terms.values() if v for c in v} | {"mtn_glob"} | {f"mtn_{r}" for r in OTHER})
    hdd = h[cols_needed].resample("D").mean()
    hist_an = {c: S.harmonic_anom(dh[c])[0] for c in cols_needed}
    an26 = {c: hdd[c] - clim_on(dh, c, hdd.index) for c in cols_needed}
    att = {}
    years = list(range(1979, 2026))
    en, _sup, _neu = S.onset_years(S.load_oni(), (1979, 2024))

    def series_for(name, src):
        if name == "rest of mountain":
            return src["mtn_glob"] - src["mtn_AND"] - sum(src[f"mtn_{r}"] for r in OTHER)
        return sum(src[c] for c in terms[name])

    for name in terms:
        x26 = series_for(name, an26)
        v26 = float(x26.loc[w0:w1].sum() * HD)
        xh = series_for(name, hist_an)
        hist = np.array([xh.loc[pd.Timestamp(f"{y}-04-15") : pd.Timestamp(f"{y}-09-19")].sum() * HD for y in years])
        att[name] = dict(
            v2026=v26,
            hist_mean=float(hist.mean()),
            hist_sd=float(hist.std(ddof=1)),
            z=float((v26 - hist.mean()) / hist.std(ddof=1)),
            rank_of_48=int((hist < v26).sum() + 1),
            en_onset_mean=float(np.mean([hist[years.index(y)] for y in en if y in years])),
            n_en=len(en),
        )
    tot26 = sum(att[k]["v2026"] for k in att)
    # sensitivity: remove each term's 1979-2025 linear trend (ERA5's torques drift; AAM does not), extrapolated to 2026
    for name in terms:
        xh = series_for(name, hist_an)
        sub = xh.loc["1979":"2025"]
        tt = (sub.index - pd.Timestamp("1979-01-01")).days.values.astype(float)
        c = np.polyfit(tt, sub.values, 1)
        x26 = series_for(name, an26).loc[w0:w1]
        t26 = (x26.index - pd.Timestamp("1979-01-01")).days.values.astype(float)
        att[name]["v2026_detrended"] = float((x26.values - np.polyval(c, t26)).sum() * HD)
        att[name]["trend_hadley_per_decade"] = float(c[0] * 3652.5)
    res["attribution"] = att
    res["attribution_total_torque"] = tot26
    res["attribution_dM_hourly_anomaly"] = var["hourly daily-means, motion+mass, anomaly vs 4x clim"]["dM"]
    # interannual spread of the window residual (how precisely ANY year's window closes)
    Rh = []
    for y in years:
        a0 = pd.Timestamp(f"{y}-04-15")
        a1 = pd.Timestamp(f"{y}-09-19")
        Rh.append(float((M4.loc[a1 + pd.Timedelta(days=1)] - M4.loc[a0]) - T4.loc[a0:a1].sum() * HD))
    res["history_window_residual"] = dict(
        mean=float(np.mean(Rh)),
        sd=float(np.std(Rh, ddof=1)),
        r2026_4x=var["4x daily, motion+mass, anomaly"]["residual"],
    )
    (od / "budget2026.json").write_text(json.dumps(res, indent=1, default=float))
    hb.to_csv(od / "budget2026_hourly.csv.gz")
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
