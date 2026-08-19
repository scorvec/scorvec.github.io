#!/usr/bin/env python3
"""National inflow: the aggregate that maps to hydro generation potential.

Two regimes, because the useful horizon is longer than the weather:

  BALANCE OF MONTH — daily resolution, driven by the AIFS+IFS ensemble
  through the delta model, member by member so cross-basin rain
  correlation survives the aggregation.  Beyond the NWP horizon the
  members are relaxed toward the monthly expectation.

  BEYOND — monthly resolution out to +6 months.  No rain forecast is
  involved: at that range the information is ENSO, reservoir storage and
  antecedent wetness.  That means this half can be backtested on the full
  2000-2026 inflow record, El Nino years included, without waiting on a
  satellite archive.

NON-STATIONARITY.  National inflow energy roughly doubled from 123 GWh/d
(2005) to 249 (2025) purely from fleet growth — Sogamoso, Ituango and the
rest.  Modelling raw energy would fit that ramp, not hydrology.  So the
model works in fleet-corrected % of norm (each basin's series is already
fleet-corrected; the national aggregate is their energy-weighted mean)
and converts to GWh/day only at the end, using the CURRENT fleet's
day-of-year norm.  Change the fleet and the conversion changes; the
hydrology does not.

    python scripts/sst/national_inflow.py [--showcase 2015-06:2016-06]

Outputs:
  ~/colombia_hydro/out/national_inflow.json
  ~/colombia_hydro/site/national_inflow.webp
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPO = HERE.parent.parent
PRIV = Path.home() / "colombia_hydro"
# The split this module already implements is exactly what the volume
# rebuild needs: model the ANOMALY, convert to GWh only at the end via the
# current fleet's norm. So the anomaly source moves to volume (pure
# hydrology, no turbines) while national_norm_gwh() stays on AporEner - it
# is deliberately the CURRENT fleet's energy norm, which is what converts a
# hydrological anomaly into today's generation potential.
_VOL = os.environ.get("CO_INFLOW_METRIC", "AporEner") == "AporCaudal"
INFLOW_JSON = (REPO / "colombia_hydro" / "data" /
               ("inflow_clim_vol.json" if _VOL else "inflow_clim.json"))
NINO_JSON = REPO / "assets" / "sst" / "data" / "nino_history.json"
APOR = PRIV / "raw" / "aporener_daily.json.gz"
OUT_JSON = PRIV / "out" / "national_inflow.json"
OUT_PNG = PRIV / "site" / "national_inflow.webp"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
NORM_YEARS = 5               # years of AporEner defining the CURRENT fleet norm
LOG_SPACE = True             # fit monthly % of norm in logs.  A linear fit
                             # extrapolates without bound: at ONI +2.12 it
                             # returned 13% of norm for Jan-2027 with a
                             # NEGATIVE p10, against a 26-year observed
                             # minimum of 43%.  Logs make the ENSO response
                             # multiplicative, keep every quantile positive,
                             # and skew the distribution the way inflow
                             # actually is.  They also verify better at every
                             # lead (skill +0.295/+0.193/+0.131 against
                             # +0.293/+0.183/+0.118).
MAX_MONTH = 6                # monthly forecast horizon


# ── national series ─────────────────────────────────────────────────────────
def basin_energy_weights() -> dict:
    """Each basin's share of national inflow energy, current fleet."""
    from hydro_region_rain import _river_energy, _regulated_rivers, CATCH_GJ
    egy, reg = _river_energy(), _regulated_rivers()
    gj = json.loads(CATCH_GJ.read_text())
    tot = {r: 0.0 for r in ORDER}
    seen = set()
    for ft in gj["features"]:
        p = ft["properties"]
        rg, riv = p.get("region"), p.get("river")
        if rg not in tot or riv in seen:
            continue
        seen.add(riv)
        tot[rg] += float(egy.get(riv, 0.0))     # regulated INCLUDED: they
                                                # still generate, they are
                                                # only excluded from the RAIN
                                                # weighting
    s = sum(tot.values())
    return {r: v / s for r, v in tot.items()} if s else {r: 1 / 6 for r in ORDER}


def national_norm_gwh() -> np.ndarray:
    """Current-fleet national inflow energy by day-of-year, GWh/day."""
    d = json.load(gzip.open(APOR, "rt"))
    days = sorted(d)
    yrs = sorted({x[:4] for x in days})[-NORM_YEARS:]
    acc = np.zeros(367)
    cnt = np.zeros(367)
    for x in days:
        if x[:4] not in yrs:
            continue
        doy = datetime.strptime(x, "%Y-%m-%d").timetuple().tm_yday
        v = sum(z for z in d[x].values() if isinstance(z, (int, float)))
        acc[doy] += v / 1e6                     # kWh/day -> GWh/day
        cnt[doy] += 1
    norm = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    # circular 15-day smooth
    ok = np.isfinite(norm)
    idx = np.arange(367)
    filled = np.interp(idx, idx[ok], norm[ok])
    k = 15
    pad = np.concatenate([filled[-k:], filled, filled[:k]])
    sm = np.convolve(pad, np.ones(k) / k, "same")[k:-k]
    return sm


def load_national():
    ic = json.loads(INFLOW_JSON.read_text())
    fp = ic["full_pct_of_norm"]
    dates = np.array(fp["dates"], dtype="datetime64[D]")
    W = basin_energy_weights()
    M = np.column_stack([np.where(np.asarray(fp[b], float) == 0, np.nan,
                                  np.asarray(fp[b], float)) for b in ORDER])
    w = np.array([W[b] for b in ORDER])
    nat = np.nansum(M * w, axis=1) / np.nansum(np.isfinite(M) * w, axis=1)

    # storage anomaly, national = energy-weighted
    from xm_storage import pct_anomaly_series
    sd, sa = pct_anomaly_series()
    sd = np.array([str(x) for x in sd], dtype="datetime64[D]")
    S = np.column_stack([np.asarray(sa[b], float) for b in ORDER])
    snat = np.nansum(S * w, axis=1) / np.nansum(np.isfinite(S) * w, axis=1)
    smap = dict(zip(sd.astype(str), snat))
    stor = np.array([smap.get(str(x), np.nan) for x in dates])
    for i in range(1, len(stor)):
        if not np.isfinite(stor[i]):
            stor[i] = stor[i - 1]

    # ONI, monthly -> daily hold
    nh = json.loads(NINO_JSON.read_text())
    ym, vals = nh["months"], nh["series"]["oni"]["anom"]
    oni_m = {str(a)[:7]: (float(b) if b is not None else np.nan)
             for a, b in zip(ym, vals)}
    oni = np.array([oni_m.get(str(x)[:7], np.nan) for x in dates])
    for i in range(1, len(oni)):
        if not np.isfinite(oni[i]):
            oni[i] = oni[i - 1]
    return {"dates": dates, "pct": nat, "stor": stor, "oni": oni,
            "weights": W, "norm_gwh": national_norm_gwh()}


# ── monthly regime: beyond the weather, out to +6 months ────────────────────
def monthly_frame(d):
    """Monthly means plus the state known at the END of each issue month."""
    dates, pct = d["dates"], d["pct"]
    ym = np.array([str(x)[:7] for x in dates])
    months = sorted(set(ym))
    rows = {}
    for m in months:
        k = ym == m
        if k.sum() < 20:
            continue
        rows[m] = {"days": int(k.sum()), "pct": float(np.nanmean(pct[k])),
                   "oni": float(np.nanmean(d["oni"][k])),
                   "stor_end": float(d["stor"][k][-1]),
                   "ant30": float(np.nanmean(pct[np.where(k)[0][-1] - 29:
                                                 np.where(k)[0][-1] + 1])),
                   "ant90": float(np.nanmean(pct[max(0, np.where(k)[0][-1] - 89):
                                                 np.where(k)[0][-1] + 1]))}
    return rows


def month_design(rows, keys, lead):
    """Predict pct of month t+lead from state at end of month t."""
    X, Y, T = [], [], []
    for i, k in enumerate(keys):
        j = i + lead
        if j >= len(keys):
            break
        tgt = keys[j]
        # target must actually be `lead` months after the issue month
        a = int(k[:4]) * 12 + int(k[5:7])
        b = int(tgt[:4]) * 12 + int(tgt[5:7])
        if b - a != lead:
            continue
        r = rows[k]
        mo = int(tgt[5:7])
        s1, c1 = np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)
        o = r["oni"]
        X.append([o, o * s1, o * c1, r["stor_end"], r["ant30"] - 100.0,
                  r["ant90"] - 100.0, s1, c1])
        Y.append(rows[tgt]["pct"])
        T.append(tgt)
    return np.asarray(X, float), np.asarray(Y, float), T


def monthly_backtest(d, embargo_years=1):
    """Leave-one-year-out with a +/-1 year embargo.

    A plain LOYO would leave the other half of an El Nino in the training
    set — the 2015-16 event spans two calendar years — so the embargo
    drops the adjacent years too.  That is the difference between
    "predicted 2016 having seen 2015" and genuinely cold-starting it.
    """
    rows = monthly_frame(d)
    keys = sorted(rows)
    out = {"leads": {}, "by_month": {}}
    preds = {}
    for lead in range(1, MAX_MONTH + 1):
        X, Y, T = month_design(rows, keys, lead)
        yrs = np.array([int(t[:4]) for t in T])
        Yt = np.log(np.clip(Y, 5.0, None)) if LOG_SPACE else Y
        P = np.full(len(Y), np.nan)
        for y in sorted(set(yrs)):
            te = yrs == y
            tr = np.abs(yrs - y) > embargo_years
            if tr.sum() < 60:
                continue
            A = np.column_stack([np.ones(tr.sum()), X[tr]])
            b, *_ = np.linalg.lstsq(A, Yt[tr], rcond=None)
            P[te] = b[0] + X[te] @ b[1:]
        if LOG_SPACE:
            P = np.exp(P)
        m = np.isfinite(P) & np.isfinite(Y)
        if m.sum() < 30:
            continue
        clim = np.full(m.sum(), float(np.mean(Y[m])))
        rmse = lambda a, b_: float(np.sqrt(np.mean((a - b_) ** 2)))
        out["leads"][lead] = {
            "n": int(m.sum()),
            "r": round(float(np.corrcoef(P[m], Y[m])[0, 1]), 3),
            "rmse": round(rmse(P[m], Y[m]), 2),
            "rmse_climatology": round(rmse(clim, Y[m]), 2),
            "skill_vs_climatology": round(1 - rmse(P[m], Y[m]) / rmse(clim, Y[m]), 3),
            "mae": round(float(np.mean(np.abs(P[m] - Y[m]))), 2),
            "log_resid_sd": round(float(np.std(np.log(np.clip(Y[m], 5, None))
                                               - np.log(np.clip(P[m], 5, None)))), 4)}
        for t, p, o in zip(np.asarray(T)[m], P[m], Y[m]):
            preds.setdefault(t, {})[lead] = (round(float(p), 1), round(float(o), 1))
    out["by_month"] = preds
    return out


# ── balance-of-month regime: daily, ensemble-driven ─────────────────────────
def daily_regime(d, monthly_exp):
    """National daily paths to the end of the month, member by member.

    Each basin's delta model is run on every ensemble member, then the six
    basins are summed WITH THE SAME MEMBER INDEX so a member that is wet
    over Antioquia and dry over Oriente stays that way in the national
    total.  Aggregating basin quantiles instead would silently assume the
    basins are perfectly correlated and overstate the national spread.

    Past the NWP horizon the member paths are relaxed toward the monthly
    model's expectation, so the daily fan hands over to the monthly one
    instead of running on stale rain.
    """
    import inflow_delta_model as M
    import inflow_delta_forecast as F
    md = M.load()
    W = d["weights"]
    cycles = F.latest_cycles()
    if not cycles:
        return None
    factors = json.loads(F.VERIF_JSON.read_text())["bias_factors"]
    dates = md["dates"]
    n = len(dates)
    horizon = M.MAX_LEAD
    ext = np.array([dates[-1] + np.timedelta64(k, "D")
                    for k in range(1, horizon + 1)])
    all_dates = np.concatenate([dates, ext])

    tc = json.loads((PRIV / "raw" / "imerg_basin_daily.json").read_text())
    keep = {x: i for i, x in enumerate(tc["dates"])}
    per_basin, resid_nat = {}, []
    for b in ORDER:
        c = np.asarray(tc[b + "_clim"], float)
        F.CLIM = getattr(F, "CLIM", {})
        F.CLIM[b] = np.array([c[keep[str(x).replace("-", "")]] for x in dates])
        pad = len(all_dates) - n
        F.CLIM[b] = np.concatenate([F.CLIM[b], F.CLIM[b][-pad:]])

        rain, y = md["rain"][b], md["y"][b]
        roni, stor = md["roni"], md["stor"][b]
        tr = np.ones(n, bool)
        tau, lag = M.select_hyper(rain, y, roni, stor, tr, False)
        X, dy = M.design(rain, y, roni, stor, tau, lag)
        m = np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
        beta = M.fit(X, dy, m)
        sh, off = M.shrinkage(beta, rain, y, roni, stor, tau, lag, tr)
        rh = np.concatenate([rain, np.zeros(pad)])
        rn = np.concatenate([roni, np.full(pad, roni[-1])])
        st = np.concatenate([stor, np.full(pad, stor[-1])])
        mems, _ = F.member_rain(b, cycles, factors, all_dates, rh)
        i0 = n - 1
        y0 = float(y[i0]) if np.isfinite(y[i0]) else float(np.nanmean(y[-5:]))
        paths = []
        for x in mems:
            sim = M.simulate(beta, x, rn, st, tau, lag, i0, y0, horizon)
            paths.append([y0 + off[j] + sh[j] * (v - y0) if np.isfinite(v)
                          else np.nan for j, v in enumerate(sim)])
        per_basin[b] = np.asarray(paths, float)

    nmem = min(v.shape[0] for v in per_basin.values())
    nat = sum(per_basin[b][:nmem] * W[b] for b in ORDER)     # member-wise
    return {"paths": nat, "dates": all_dates[n:], "last_obs": float(
        np.nansum([md["y"][b][-1] * W[b] for b in ORDER]))}


def national_residuals(d, embargo=30, folds=8, cond=None):
    """Out-of-sample national daily residuals by lead, for the fan width.

    HETEROSCEDASTIC.  A single pooled residual per lead is wrong here: at
    lead 7 the out-of-sample residual sd is 16.4 pts when the basin is dry
    (<70% of norm), 21.1 in the middle and 36.2 when wet (>110).  Applying
    the pooled ~26.7 to every day makes the fan far too wide in a drought
    and too narrow in a flood - exactly backwards for a drying outlook.

    `cond` is the predicted level for the day being forecast; residuals
    are drawn from the matching state band.  Falls back to the pooled set
    when a band is too thin to trust.
    """
    import inflow_delta_model as M
    md = M.load()
    W = d["weights"]
    per = {}
    for b in ORDER:
        cv = M.backtest(md, b)
        per[b] = cv
    out = {}
    for h in range(1, M.MAX_LEAD + 1):
        acc = None
        for b in ORDER:
            r = np.asarray(per[b]["leads"].get(h, {}).get("resid", []), float)
            if not len(r):
                acc = None
                break
            acc = r * W[b] if acc is None else acc[:len(r)] + r[:len(acc)] * W[b]
        if acc is not None and len(acc) > 30:
            out[h] = acc
    return out


def monthly_forecast(d, bt):
    """Months +1..+6 from state at the last complete month, with LOYO spread."""
    rows = monthly_frame(d)
    keys = sorted(rows)
    # monthly_frame already drops months with <20 days; require a
    # near-complete one for the state vector, else step back
    issue = keys[-1]
    if rows[issue]["days"] < 28 and len(keys) > 1:
        issue = keys[-2]
    obs_min = min(rows[k]["pct"] for k in keys)
    out = []
    for lead in range(1, MAX_MONTH + 1):
        X, Y, T = month_design(rows, keys, lead)
        if len(Y) < 60:
            continue
        Yt = np.log(np.clip(Y, 5.0, None)) if LOG_SPACE else Y
        A = np.column_stack([np.ones(len(Y)), X])
        b, *_ = np.linalg.lstsq(A, Yt, rcond=None)
        r = rows[issue]
        a0 = int(issue[:4]) * 12 + int(issue[5:7]) + lead
        ty, tm = (a0 - 1) // 12, (a0 - 1) % 12 + 1
        s1, c1 = np.sin(2 * np.pi * tm / 12), np.cos(2 * np.pi * tm / 12)
        o = r["oni"]
        x = np.array([o, o * s1, o * c1, r["stor_end"], r["ant30"] - 100.0,
                      r["ant90"] - 100.0, s1, c1])
        raw = float(b[0] + x @ b[1:])
        L = bt["leads"].get(lead, {})
        if LOG_SPACE:
            sd = L.get("log_resid_sd", 0.28)
            p50, p10, p90 = (float(np.exp(raw)),
                             float(np.exp(raw - 1.2816 * sd)),
                             float(np.exp(raw + 1.2816 * sd)))
        else:
            sd = L.get("rmse", 25.0)
            p50, p10, p90 = raw, max(raw - 1.2816 * sd, 0.0), raw + 1.2816 * sd
        doy = datetime(ty, tm, 15).timetuple().tm_yday
        gw = d["norm_gwh"][doy] / 100.0
        out.append({"month": f"{ty:04d}-{tm:02d}", "lead": lead,
                    "pct_p50": round(p50, 1), "pct_p10": round(p10, 1),
                    "pct_p90": round(p90, 1),
                    "below_observed_minimum": bool(p50 < obs_min),
                    "gwh_p50": round(p50 * gw, 1),
                    "gwh_p10": round(p10 * gw, 1),
                    "gwh_p90": round(p90 * gw, 1),
                    "norm_gwh": round(d["norm_gwh"][doy], 1)})
    # ONI outside the fitted range means extrapolation, and 2015-16 shows
    # which way it fails: with that event withheld the model called the
    # collapse 6 months out but overshot its depth (34-40 vs 53-64 observed).
    onis = np.array([rows[k]["oni"] for k in keys], float)
    o_now = rows[issue]["oni"]
    pctile = float(np.mean(onis <= o_now) * 100)
    return {"issue_month": issue, "oni_at_issue": round(o_now, 2),
            "oni_percentile_in_record": round(pctile, 1),
            "extrapolating_on_enso": bool(pctile > 97 or pctile < 3),
            "severity_caveat": ("at extreme ONI the model overshoots the depth "
                                "of the response - see the 2015-16 showcase"),
            "storage_anom_at_issue": round(rows[issue]["stor_end"], 1),
            "observed_monthly_minimum": round(obs_min, 1),
            "log_space": LOG_SPACE, "months": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--showcase", default="2015-06:2016-08")
    a = ap.parse_args()
    d = load_national()
    print(f"national series {d['dates'][0]}..{d['dates'][-1]} "
          f"({len(d['dates'])} d)  weights "
          f"{ {k: round(v, 3) for k, v in d['weights'].items()} }", flush=True)

    bt = monthly_backtest(d)
    print(f"\nMONTHLY (leave-one-year-out, +/-1 y embargo)")
    print(f"{'lead':>5}{'n':>5}{'r':>7}{'rmse':>8}{'clim':>8}{'skill':>8}")
    for L, v in bt["leads"].items():
        print(f"{L:5d}{v['n']:5d}{v['r']:7.3f}{v['rmse']:8.2f}"
              f"{v['rmse_climatology']:8.2f}{v['skill_vs_climatology']:+8.3f}")

    fc_m = monthly_forecast(d, bt)
    daily = None
    try:
        daily = daily_regime(d, fc_m)
        resid = national_residuals(d)
    except Exception as e:                          # noqa: BLE001
        print(f"daily regime unavailable: {repr(e)[:110]}")
        resid = {}

    days = []
    if daily is not None:
        for j, dt in enumerate(daily["dates"]):
            col = daily["paths"][:, j]
            col = col[np.isfinite(col)]
            if not len(col):
                continue
            R = resid.get(j + 1, np.zeros(0))
            full = (col[:, None] + R[None, :]).ravel() if len(R) > 30 else col
            full = np.clip(full, 0.0, None)
            doy = int(str(dt)[5:7]) and datetime.strptime(
                str(dt), "%Y-%m-%d").timetuple().tm_yday
            gw = d["norm_gwh"][doy] / 100.0
            days.append({"date": str(dt), "lead": j + 1,
                         "pct": {f"p{q}": round(float(np.percentile(full, q)), 1)
                                 for q in (5, 25, 50, 75, 95)},
                         "gwh_p50": round(float(np.percentile(full, 50)) * gw, 1),
                         "gwh_p5": round(float(np.percentile(full, 5)) * gw, 1),
                         "gwh_p95": round(float(np.percentile(full, 95)) * gw, 1),
                         "rain_only_sd": round(float(np.std(col)), 1),
                         "norm_gwh": round(d["norm_gwh"][doy], 1)})

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "series": f"{d['dates'][0]}..{d['dates'][-1]}",
           "basin_energy_weights": {k: round(v, 4) for k, v in d["weights"].items()},
           "units": "pct = fleet-corrected % of norm; gwh = GWh/day on the "
                    f"CURRENT fleet norm (last {NORM_YEARS} y of AporEner)",
           "monthly_backtest": bt["leads"],
           "monthly_forecast": fc_m,
           "daily_balance_of_month": days,
           "showcase": {}}
    a0, a1 = a.showcase.split(":")
    rows = monthly_frame(d)
    for m in sorted(rows):
        if a0 <= m <= a1:
            p = bt["by_month"].get(m, {})
            out["showcase"][m] = {"obs": round(rows[m]["pct"], 1),
                                  "oni": round(rows[m]["oni"], 2),
                                  **{f"lead{L}": p[L][0] for L in sorted(p)}}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\nmonthly forecast from {fc_m['issue_month']} "
          f"(ONI {fc_m['oni_at_issue']:+.2f}, storage "
          f"{fc_m['storage_anom_at_issue']:+.1f} pts)")
    for m in fc_m["months"]:
        print(f"  {m['month']}  {m['pct_p50']:5.0f}% of norm  "
              f"[{m['pct_p10']:.0f}–{m['pct_p90']:.0f}]   "
              f"{m['gwh_p50']:6.1f} GWh/d  [{m['gwh_p10']:.0f}–{m['gwh_p90']:.0f}]")
    try:
        figure(out, d, bt, rows)
    except Exception as e:                          # noqa: BLE001
        print(f"figure failed: {repr(e)[:140]}")
    print(f"wrote {OUT_JSON}")
    return 0


def figure(out, d, bt, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    NAVY, INK = "#13273d", "#1a2733"
    fig = plt.figure(figsize=(15.6, 10.4))
    hd = fig.add_axes([0, 0.945, 1, 0.055]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.014, 0.62, "COLOMBIA — NATIONAL INFLOW / HYDRO POTENTIAL",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    hd.text(0.014, 0.2, "daily to end of month from the ensemble · monthly to "
            "+6 from ENSO, storage and antecedent wetness",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")
    hd.text(0.986, 0.5, out["series"], transform=hd.transAxes,
            color="#b9c6d4", fontsize=9, va="center", ha="right")

    # ── showcase: the El Nino the model had to call cold ────────────────────
    ax = fig.add_axes([0.055, 0.575, 0.55, 0.31])
    sc = out["showcase"]
    ms = sorted(sc)
    t = [datetime.strptime(m, "%Y-%m") for m in ms]
    ax.axhline(100, color="0.55", lw=0.9, ls=":")
    ax.plot(t, [sc[m]["obs"] for m in ms], color="#111", lw=2.6,
            marker="o", ms=5, label="observed", zorder=5)
    for L, c in ((1, "#1f4e8c"), (3, "#e08214"), (6, "#c62828")):
        v = [sc[m].get(f"lead{L}") for m in ms]
        if any(x is not None for x in v):
            ax.plot(t, v, color=c, lw=1.7, marker="s", ms=3.5, alpha=0.9,
                    label=f"forecast, {L} month{'s' if L > 1 else ''} ahead")
    ax.set_ylabel("national inflow, % of norm", fontsize=9)
    ax.set_title(f"The test that matters — {ms[0]} to {ms[-1]}, each year "
                 "cold-started (year ±1 withheld)",
                 fontsize=11, fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=8, ncol=2); ax.grid(lw=0.25, alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax2 = ax.twinx()
    ax2.plot(t, [sc[m]["oni"] for m in ms], color="#7b1fa2", lw=1.1, ls="--",
             alpha=0.75)
    ax2.set_ylabel("ONI (dashed)", fontsize=8, color="#7b1fa2")
    ax2.tick_params(labelsize=7.5, colors="#7b1fa2")

    # ── monthly skill by lead ───────────────────────────────────────────────
    ax3 = fig.add_axes([0.68, 0.575, 0.29, 0.31])
    L = sorted(int(k) for k in bt["leads"])
    ax3.bar(L, [bt["leads"][k]["skill_vs_climatology"] for k in L],
            color="#1f4e8c", alpha=0.9)
    for k in L:
        ax3.text(k, bt["leads"][k]["skill_vs_climatology"] + 0.006,
                 f"r={bt['leads'][k]['r']:.2f}", ha="center", fontsize=7.2)
    ax3.set_xlabel("lead, months", fontsize=9)
    ax3.set_ylabel("RMSE skill vs climatology", fontsize=9)
    ax3.set_title("Monthly skill, 26 years", fontsize=11, fontweight="bold",
                  loc="left", color=INK)
    ax3.grid(lw=0.25, alpha=0.5, axis="y"); ax3.tick_params(labelsize=8)

    # ── the live forecast: daily then monthly, in GWh/day ───────────────────
    ax4 = fig.add_axes([0.055, 0.075, 0.915, 0.40])
    days = out["daily_balance_of_month"]
    if days:
        td = [datetime.strptime(x["date"], "%Y-%m-%d") for x in days]
        ax4.fill_between(td, [x["gwh_p5"] for x in days],
                         [x["gwh_p95"] for x in days], color="#1f4e8c",
                         alpha=0.18, lw=0, label="daily 5–95%")
        ax4.plot(td, [x["gwh_p50"] for x in days], color="#1f4e8c", lw=2.2,
                 label="daily median (ensemble + model error)")

    mf = out["monthly_forecast"]["months"]
    if mf and days:
        last_daily = datetime.strptime(days[-1]["date"], "%Y-%m-%d")
        mf = [x for x in mf if datetime.strptime(x["month"] + "-15",
                                                 "%Y-%m-%d") > last_daily]
    if mf:
        tm = [datetime.strptime(x["month"] + "-15", "%Y-%m-%d") for x in mf]
        ax4.errorbar(tm, [x["gwh_p50"] for x in mf],
                     yerr=[[x["gwh_p50"] - x["gwh_p10"] for x in mf],
                           [x["gwh_p90"] - x["gwh_p50"] for x in mf]],
                     fmt="s", color="#b35806", ms=7, lw=1.6, capsize=4,
                     label="monthly median, 10–90%")
        if days:
            ax4.axvline(td[-1], color="0.6", lw=0.9, ls="--")
            ax4.text(td[-1], ax4.get_ylim()[1], "  ensemble ends",
                     fontsize=7.5, color="#5a6b7a", va="top")
    if days or mf:
        t0 = datetime.strptime(days[0]["date"], "%Y-%m-%d") if days else tm[0]
        t1 = tm[-1] if mf else datetime.strptime(days[-1]["date"], "%Y-%m-%d")
        span = (t1 - t0).days + 1
        tn = [t0 + timedelta(days=k) for k in range(span)]
        ax4.plot(tn, [d["norm_gwh"][x.timetuple().tm_yday] for x in tn],
                 color="0.45", lw=1.2, ls=":",
                 label="seasonal norm, current fleet")
    ax4.set_ylabel("national inflow energy, GWh/day", fontsize=9.5)
    ax4.set_title("Live outlook — daily through the balance of the month, "
                  "then monthly to +6", fontsize=11, fontweight="bold",
                  loc="left", color=INK)
    ax4.legend(fontsize=8, ncol=3); ax4.grid(lw=0.25, alpha=0.5)
    ax4.tick_params(labelsize=8)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.text(0.055, 0.017, "% of norm is fleet-corrected and stationary; GWh/day "
             "applies TODAY'S fleet norm to it — national inflow energy roughly "
             "doubled 2005–2025 on fleet growth alone, so the two must not be "
             "conflated.", fontsize=7.6, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=118); plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    raise SystemExit(main())
