#!/usr/bin/env python3
"""Momentum budget of past sudden warmings (2026-09-27; user: the historical counterpart of the live Momentum budget
item). LAPTOP, run once, after the daily NCCS fetch of the tropical-precursor study.

Every term of the transformed-Eulerian-mean zonal momentum budget at 55-65N, composited around the SSW central dates,
days -40..+40, at 50, 10 and 1 hPa. The OPERATOR IS THE LIVE PRODUCT'S: momentum_budget.tem_terms (Coriolis on the
residual flow fhat v*, vertical advection -omega* du/dp, the EP-flux divergence and its wave-1 / wave-2 / wave-3+ parts)
and momentum_budget.band (55-65 deg, cos-weighted), on momentum_budget.LEVS, with the same day definition: du/dt is the
00Z -> next-00Z change of [u], every other term the trapezoid mean of the day's snapshots (here the two bounding 00Z
states: the only times on file) , GWD the calendar-day mean. residual = du/dt - (cor + vad + epd + gwd [+ ana]).

Data (~/research/ssw_tropical_precursors, fetch_nccs_daily.py):
  MERRA-2 GMI replay, 00Z u v T omega, 2 x 2.5 deg, 40S-90N, every Oct-Apr day 1980-2019 (event windows complete, other
    days a shuffled subset); daily-mean DUDTGWD and DUDTORO (orographic; non-orographic = difference). The replay carries
    no analysis-increment tendency, so its residual holds the assimilation's push.
  GEOS FP assimilation around the 2018-2026 events: 00Z state + the mean of four tavg3 windows of DUDTGWD and DUDTANA
    (the analysis increment, its own term); GEOS FP publishes no orographic split.
Events: the site's MERRA-2 catalogue (strat_history_events.json; confirmed events only) plus 29 Feb 1980 (NCEP R1);
  "strong" = the top third by depth (the catalogue's 100-150 hPa polar-cap measure, days 1-30).
Tests: composite against season-matched random dates (same calendar day +-10 days, another winter, 2,000 draws) drawn from
  the days on file, with event-window days thinned to the other days' coverage; window means over days -40..-21, -20..-1,
  0..+20, +21..+40 with p and a leave-one-event-out range; the share of the wave drag the Coriolis term offsets with an
  event-bootstrap 90 % interval.
Output: scripts/strat/reference/mbudget_history.json (drawn by ssw_precursors.py in Actions).

    python scripts/strat/build_mbudget_history.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import momentum_budget as MB                                          # noqa: E402  (the live operator)

STUDY = Path.home() / "research" / "ssw_tropical_precursors"
OUT = HERE / "reference" / "mbudget_history.json"
LEVELS = (50, 10, 1)
DAY = 86400.0
RNG = np.random.default_rng(20260927)
LAGS = np.arange(-40, 41)
WINDOWS = ((-40, -21), (-20, -1), (0, 20), (21, 40))
TERMS = ("dudt", "cor", "vad", "epd", "epd_k1", "epd_k2", "epd_k3p", "gwd", "oro", "nog", "ana", "resid")


def levels_idx(lev):
    return np.array([int(np.argmin(np.abs(lev - L))) for L in MB.LEVS])


def rec_from_daily(z):
    k = levels_idx(z["lev"])
    g = lambda key: z[key][k].astype("float64")                                    # noqa: E731
    r = {"ubar": g("ubar"), "vbar": g("vbar"), "tbar": g("tbar"), "wbar": g("wbar"), "vT": g("vT"), "uv": g("uv"), "uw": g("uw")}
    r["vT_k1"], r["vT_k2"], r["uv_k1"], r["uv_k2"] = g("vT_k1"), g("vT_k2"), g("uv_k1"), g("uv_k2")
    r["vT_k3p"] = r["vT"] - r["vT_k1"] - r["vT_k2"]; r["uv_k3p"] = r["uv"] - r["uv_k1"] - r["uv_k2"]
    zero = np.zeros_like(r["uv"])
    for w in MB.WAVES:                    # no wave-split [u'omega'] (as the live MERRA-2 normals); the total is exact
        r[f"uw_{w}"] = zero
    return r, k


def snapshots(ddir, fp=False):
    """00Z band means (55-65N) at LEVELS for every file: ubar and the TEM terms (m/s^2); GWD/ORO/ANA daily means."""
    rows = []
    li = [int(np.argmin(np.abs(MB.LEVS - L))) for L in LEVELS]
    for f in sorted(glob.glob(str(ddir / "*.npz"))):
        z = np.load(f)
        lat = z["lat"].astype("float64")
        r, k = rec_from_daily(z)
        if not all(np.isfinite(r[x]).all() for x in ("ubar", "tbar", "vT", "uv", "uw")):
            continue
        tm = MB.tem_terms(lat, r)
        d = {"t": pd.Timestamp(Path(f).stem)}
        for L, i in zip(LEVELS, li):
            for key in ("ubar", "cor", "vad", "epd", "epd_k1", "epd_k2", "epd_k3p"):
                d[f"{key}{L}"] = float(MB.band(tm[key][i], lat, 55, 65))
            if fp:
                d[f"gwd{L}"] = float(MB.band(z["dudtgwd"][k][i].astype("float64"), lat, 55, 65))
                d[f"ana{L}"] = float(MB.band(z["dudtana"][k][i].astype("float64"), lat, 55, 65))
        rows.append(d)
    S = pd.DataFrame(rows).set_index("t").sort_index()
    if not fp:
        g = []
        for f in sorted(glob.glob(str(STUDY / "data" / "gmi_gwd" / "*.npz"))):
            z = np.load(f); k = levels_idx(z["lev"]); lat = z["lat"].astype("float64")
            d = {"t": pd.Timestamp(Path(f).stem)}
            for L, i in zip(LEVELS, li):
                d[f"gwd{L}"] = float(MB.band(np.nan_to_num(z["dudtgwd"][k][i].astype("float64")), lat, 55, 65))
                d[f"oro{L}"] = float(MB.band(np.nan_to_num(z["dudtoro"][k][i].astype("float64")), lat, 55, 65))
            g.append(d)
        S = S.join(pd.DataFrame(g).set_index("t").sort_index(), how="left")
    return S


def window_budget(S, fp=False, half=2):
    """5-day (+-half) window budget from 00Z snapshots, robust to missing days: each term = mean of the snapshots in the
    window (m/s per day; GWD/ORO/ANA are calendar-day means), du/dt = least-squares slope of the 00Z wind over +-(half+1)
    days. Inside the event windows every day is on file, so this equals the live day definition (00Z->00Z change and
    trapezoid means) averaged over five days; outside them it tolerates the shuffled fetch's gaps for the null."""
    S = S.asfreq("D")
    out = pd.DataFrame(index=S.index)
    for L in LEVELS:
        for key in ("cor", "vad", "epd", "epd_k1", "epd_k2", "epd_k3p"):
            out[f"{key}{L}"] = S[f"{key}{L}"].rolling(2 * half + 1, center=True, min_periods=min(2, 2 * half + 1)).mean() * DAY
        for key in ("gwd", "oro", "ana"):
            if f"{key}{L}" in S:
                out[f"{key}{L}"] = S[f"{key}{L}"].rolling(2 * half + 1, center=True, min_periods=min(2, 2 * half + 1)).mean() * DAY
        if not fp:
            out[f"nog{L}"] = out[f"gwd{L}"] - out[f"oro{L}"]; out[f"ana{L}"] = 0.0
        else:
            out[f"oro{L}"] = out[f"nog{L}"] = np.nan
        u = S[f"ubar{L}"]; ok = u.notna().astype(float); uu = u.fillna(0.0)
        tt = pd.Series(np.arange(len(u), dtype=float), index=u.index)
        w = 2 * half + 3
        n = ok.rolling(w, center=True).sum(); st = (tt * ok).rolling(w, center=True).sum(); su = uu.rolling(w, center=True).sum()
        stu = (tt * uu).rolling(w, center=True).sum(); stt = (tt * tt * ok).rolling(w, center=True).sum()
        den = stt - st * st / n
        out[f"dudt{L}"] = ((stu - st * su / n) / den).where((n >= 3) & (den > 0))
        out[f"u{L}"] = u.rolling(2 * half + 1, center=True, min_periods=1).mean()
        out[f"resid{L}"] = out[f"dudt{L}"] - (out[f"cor{L}"] + out[f"vad{L}"] + out[f"epd{L}"] + out[f"gwd{L}"] + out[f"ana{L}"].fillna(0))
    return out


def null_dates(dates, years, B):
    out = np.empty((B, len(dates)), dtype="datetime64[ns]")
    pool = np.array(sorted(years))
    for j, d in enumerate(dates):
        wy = d.year if d.month >= 7 else d.year - 1
        cand = pool[pool != wy]
        ys = RNG.choice(cand, B); js = RNG.integers(-10, 11, B)
        for b in range(B):
            y = ys[b] + (1 if d.month < 7 else 0)
            try:
                dd = pd.Timestamp(y, d.month, d.day)
            except ValueError:
                dd = pd.Timestamp(y, d.month, 28)
            out[b, j] = (dd + pd.Timedelta(days=int(js[b]))).to_datetime64()
    return out


def lagmat(s, dates, lags):
    idx = pd.DatetimeIndex(dates)
    return np.column_stack([s.reindex(idx + pd.Timedelta(days=int(L))).values for L in lags])


def composite(D, cols, ev, null_pool_mask, years, B=2000):
    Dsm = D[cols]
    nd = null_dates(ev, years, B)
    res = {}
    for c in cols:
        s = Dsm[c]
        M = lagmat(s, ev, LAGS)
        Mn = lagmat(s * null_pool_mask, pd.DatetimeIndex(nd.ravel()), LAGS).reshape(B, len(ev), len(LAGS))
        comp = np.nanmean(M, 0); nm = np.nanmean(Mn, 1); mu = np.nanmean(nm, 0)
        p = 2 * np.minimum((nm >= comp).mean(0), (nm <= comp).mean(0)); p = np.maximum(p, 1 / B)
        wins = {}
        for a, b in WINDOWS:
            w = (LAGS >= a) & (LAGS <= b)
            cm = float(np.nanmean(comp[w])); nw = np.nanmean(nm[:, w], 1)
            pw = float(max(2 * min((nw >= cm).mean(), (nw <= cm).mean()), 1 / B))
            loo = [float(np.nanmean(np.nanmean(np.delete(M, i, 0), 0)[w])) for i in range(len(ev))]
            wins[f"{a}..{b}"] = {"comp": round(cm, 3), "null": round(float(np.nanmean(nw)), 3), "p": round(pw, 4),
                                 "loo": [round(min(loo), 3), round(max(loo), 3)]}
        res[c] = {"comp": [None if not np.isfinite(v) else round(float(v), 3) for v in comp],
                  "null": [None if not np.isfinite(v) else round(float(v), 3) for v in mu],
                  "p": [round(float(v), 4) for v in p], "windows": wins, "M": M}
    return res


def offset_fraction(M_cor, M_epd, nul_cor, nul_epd, window=(-20, 5), B=2000):
    """Share of the ANOMALOUS resolved-wave drag offset by the anomalous Coriolis term over the window: -dcor/depd, with
    anomalies = event composite minus the season-matched random-date normal (which also removes the fixed-hour bias of
    00Z-only terms). Event-bootstrap 90 % interval."""
    w = (LAGS >= window[0]) & (LAGS <= window[1])
    nc, ne = np.nanmean(np.asarray(nul_cor, float)[w]), np.nanmean(np.asarray(nul_epd, float)[w])
    def frac(ix):
        return -(np.nanmean(M_cor[ix][:, w]) - nc) / (np.nanmean(M_epd[ix][:, w]) - ne)
    n = M_cor.shape[0]
    f = frac(np.arange(n)); bs = [frac(RNG.integers(0, n, n)) for _ in range(B)]
    return {"fraction": round(float(f), 3), "ci90": [round(float(np.percentile(bs, 5)), 3), round(float(np.percentile(bs, 95)), 3)],
            "window": list(window)}


def main():
    spec = importlib.util.spec_from_file_location("fnd", STUDY / "fetch_nccs_daily.py"); fnd = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = ["x", "gmi", "0", "1"]; spec.loader.exec_module(fnd); sys.argv = argv
    ev_js = json.loads((HERE / "reference" / "strat_history_events.json").read_text())
    conf = {e["date"]: e for e in ev_js["ssw"] if not e["marginal"]}
    gmi_ev = [pd.Timestamp(d) for d in fnd.M2_EVENTS if d in conf or d == "1980-02-29"]
    gmi_ev = [d for d in gmi_ev if d <= pd.Timestamp("2019-11-01")]
    print("GMI events:", len(gmi_ev))
    S = snapshots(STUDY / "data" / "gmi_daily")
    D = window_budget(S)
    cols = [f"{k}{L}" for L in LEVELS for k in TERMS if k != "ana"] + [f"u{L}" for L in LEVELS]
    # null pool: event-window days thinned to the coverage of the other Oct-Apr days
    allg = fnd.day_list("gmi")
    evw = set(pd.Timestamp(e) + pd.Timedelta(days=k) for e in fnd.M2_EVENTS for k in range(fnd.LAG0, fnd.LAG1 + 1))
    have = set(S.dropna(subset=["cor10"]).index)
    rest = [d for d in allg if d not in evw]
    p_rest = float(np.mean([d in have for d in rest]))
    drop = [d for d in allg if d in evw and d in have and RNG.random() > p_rest]
    mask = pd.Series(1.0, index=D.index); mask[mask.index.isin(drop)] = np.nan
    years = list(range(1979, 2019))
    depth = {pd.Timestamp(k): v["lsz"] for k, v in conf.items()}
    dep = [(d, depth.get(d, np.nan)) for d in gmi_ev]
    ranked = sorted([x for x in dep if np.isfinite(x[1])], key=lambda x: -x[1])
    strong = [d for d, _ in ranked[: len(ranked) // 3]]
    res = {"built": pd.Timestamp.now().strftime("%Y-%m-%d"), "levels": LEVELS, "lags": LAGS.tolist(), "windows": [f"{a}..{b}" for a, b in WINDOWS],
           "events_all": [str(d.date()) for d in gmi_ev], "events_strong": [str(d.date()) for d in strong],
           "strong_rule": "top third by depth: mean standardised 100-150 hPa polar-cap height, days 1-30 (MERRA-2 catalogue)",
           "null": {"other_days_coverage": round(p_rest, 3), "event_days_thinned": len(drop)}, "sets": {}}
    for name, ev in (("all", gmi_ev), ("strong", strong)):
        C = composite(D, cols, ev, mask, years)
        out = {"n": len(ev)}
        for L in LEVELS:
            out[str(L)] = {k: {kk: vv for kk, vv in C[f"{k}{L}"].items() if kk != "M"} for k in TERMS if f"{k}{L}" in C}
            out[str(L)]["u"] = {kk: vv for kk, vv in C[f"u{L}"].items() if kk != "M"}
            out[str(L)]["offset"] = offset_fraction(C[f"cor{L}"]["M"], C[f"epd{L}"]["M"], C[f"cor{L}"]["null"], C[f"epd{L}"]["null"])
            out[str(L)]["offset_m20_m1"] = offset_fraction(C[f"cor{L}"]["M"], C[f"epd{L}"]["M"], C[f"cor{L}"]["null"], C[f"epd{L}"]["null"], window=(-20, -1))
            # size of each term during the breakdown (days -20..+5), mean of |composite|
            w = (LAGS >= -20) & (LAGS <= 5)
            an = lambda k: np.array([np.nan if v is None else v for v in C[f"{k}{L}"]["comp"]], float) - np.array([np.nan if v is None else v for v in C[f"{k}{L}"]["null"]], float)  # noqa: E731
            out[str(L)]["anomaly_size_breakdown"] = {k: round(float(np.nanmean(np.abs(an(k)[w]))), 3) for k in ("dudt", "cor", "vad", "epd", "gwd", "resid")}
        res["sets"][name] = out
        print(name, len(ev), {L: (out[str(L)]["offset"], out[str(L)]["offset_m20_m1"], out[str(L)]["anomaly_size_breakdown"]) for L in LEVELS})
    # noise: residual scatter on days outside every event window
    Dd = window_budget(S, half=0)                                      # single days, for the noise statistics
    quiet = Dd[~Dd.index.isin(list(evw))]
    res["noise"] = {str(L): {"resid_rms": round(float(np.sqrt(np.nanmean(quiet[f"resid{L}"] ** 2))), 3),
                             "resid_mean": round(float(np.nanmean(quiet[f"resid{L}"])), 3),
                             "dudt_sd": round(float(np.nanstd(quiet[f"dudt{L}"])), 3),
                             "epd_sd": round(float(np.nanstd(quiet[f"epd{L}"])), 3),
                             "r2_daily": round(float(1 - np.nanvar(quiet[f"resid{L}"]) / np.nanvar(quiet[f"dudt{L}"])), 3)} for L in LEVELS}
    # GEOS FP events with the analysis increment (descriptive + one-sample tests over events)
    fpdir = STUDY / "data" / "fp_daily"
    if any(fpdir.glob("*.npz")):
        SF = snapshots(fpdir, fp=True); DF = window_budget(SF, fp=True)
        fev = [pd.Timestamp(d) for d in fnd.FP_EVENTS if d in conf]
        fp = {"events": [str(d.date()) for d in fev]}
        for L in LEVELS:
            fp[str(L)] = {}
            for k in ("dudt", "cor", "vad", "epd", "gwd", "ana", "resid"):
                M = lagmat(DF[f"{k}{L}"], fev, LAGS)
                row = {}
                for a, b in WINDOWS:
                    w = (LAGS >= a) & (LAGS <= b)
                    per = np.nanmean(M[:, w], 1); per = per[np.isfinite(per)]
                    if len(per) >= 3:
                        from scipy import stats
                        tt = stats.ttest_1samp(per, 0.0)
                        row[f"{a}..{b}"] = {"mean": round(float(per.mean()), 3), "n": int(len(per)), "p_vs_zero": round(float(tt.pvalue), 4)}
                fp[str(L)][k] = row
        res["geosfp"] = fp
        print("GEOS FP:", json.dumps(fp)[:600])
    OUT.write_text(json.dumps(res, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
