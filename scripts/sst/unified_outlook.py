#!/usr/bin/env python3
"""One spliced national forecast: daily 15 days, then monthly to +6.

Joins the three horizons into a single continuous product and, more
importantly, stops under-forcing the seasonal half.

THE ENSO PROBLEM.  The monthly model takes ONI *at issue* as its
predictor, which was +2.12 in July 2026.  The C3S multi-model ENSO
forecast has Nino-3.4 reaching +3.8 by November - beyond 2015-16 (+2.65)
and 1997-98 (~+2.4), with all seven systems agreeing.  Holding ONI at its
issue value therefore under-forces the outlook badly.  Three seasonal
variants are run side by side so the disagreement is visible rather than
hidden:

  ONI_ISSUE  ONI held at its value on the issue date (the old behaviour)
  ONI_FCST   the C3S multi-model ENSO trajectory, sampled member by
             member so ENSO forecast uncertainty enters the distribution
  C3S_RAIN   driven by EMOS-calibrated C3S rainfall instead of an ENSO
             index at all

The third matters here specifically.  At ONI ~+3.8 the statistical model
is extrapolating far outside anything in its 26-year training set, where
a linear ENSO coefficient is least trustworthy; a coupled model that
simulates the event does not extrapolate a regression.  Its calibrated
rain skill is modest (CRPS skill +0.01..+0.09) but it fails differently,
which is the point of showing it.

    python scripts/sst/unified_outlook.py

Outputs: ~/colombia_hydro/out/unified_outlook.json
         ~/colombia_hydro/site/unified_outlook.webp
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import national_inflow as NI                                        # noqa: E402
import outlook_2027 as OL                                           # noqa: E402

PRIV = Path.home() / "colombia_hydro"
REPO = HERE.parent.parent
ENSO_FC = REPO / "assets" / "sst" / "data" / "enso_forecast.json"
C3S = PRIV / "out" / "c3s_basin_precip.json"
OUT_JSON = PRIV / "out" / "unified_outlook.json"
OUT_PNG = PRIV / "site" / "unified_outlook.webp"
NDRAW = 4000
MAXLEAD = 6


def enso_forecast():
    """{month: array of multi-model Nino-3.4 members}."""
    if not ENSO_FC.exists():
        return {}
    d = json.loads(ENSO_FC.read_text())
    vm = d["valid_months"]
    big = np.concatenate([np.asarray(v["n34"], float)
                          for v in d["models"].values() if v.get("n34")], axis=1)
    return {m: big[j] for j, m in enumerate(vm)}, d["issue"]


def c3s_rain_forecast():
    """{lead: array of national rain members, mm/day} for the newest init."""
    d = json.loads(C3S.read_text())["data"]
    W = NI.basin_energy_weights()
    init = max(d)
    out = {}
    for lead, byb in d[init].items():
        n = min(len(byb[b]) for b in NI.ORDER if b in byb)
        v = np.zeros(n)
        for b in NI.ORDER:
            v += np.asarray(byb[b][:n], float) * W[b]
        out[int(lead)] = v
    return init, out


def month_fit(rows, keys, lead, use_target_oni):
    """Log-space monthly fit; ONI is either at issue or at the target month."""
    X, Y = [], []
    for i, k in enumerate(keys):
        j = i + lead
        if j >= len(keys):
            break
        tgt = keys[j]
        a = int(k[:4]) * 12 + int(k[5:7])
        b = int(tgt[:4]) * 12 + int(tgt[5:7])
        if b - a != lead:
            continue
        r = rows[k]
        o = rows[tgt]["oni"] if use_target_oni else r["oni"]
        mo = int(tgt[5:7])
        s1, c1 = np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)
        X.append([o, o * s1, o * c1, r["stor_end"], r["ant30"] - 100.0,
                  r["ant90"] - 100.0, s1, c1])
        Y.append(np.log(max(rows[tgt]["pct"], 5.0)))
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    A = np.column_stack([np.ones(len(Y)), X])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ beta
    return beta, float(np.std(resid))


def main() -> int:
    d = NI.load_national()
    rows = NI.monthly_frame(d)
    keys = sorted(rows)
    gen = OL.monthly_generation()
    beta_sh, sst = OL.fit_share(gen, rows)
    bt = NI.monthly_backtest(d)
    issue = keys[-1] if rows[keys[-1]]["days"] >= 28 else keys[-2]
    r0 = rows[issue]
    rng = np.random.default_rng(2027)

    ef, ef_issue = enso_forecast()
    c3s_init, c3s_rain = c3s_rain_forecast()
    print(f"issue {issue}  ONI now {r0['oni']:+.2f}  storage "
          f"{r0['stor_end']:+.1f}  | ENSO fcst {ef_issue}, C3S rain {c3s_init}")

    # C3S rain climatology by (target month, lead) for the ratio
    craw = json.loads(C3S.read_text())["data"]
    W = NI.basin_energy_weights()
    clim = {}
    for init, byl in craw.items():
        if init >= "2026-01":
            continue
        for lead, byb in byl.items():
            a = int(init[:4]) * 12 + int(init[5:7]) + int(lead)
            tm = (a - 1) % 12 + 1
            n = min(len(byb[b]) for b in NI.ORDER if b in byb)
            v = np.zeros(n)
            for b in NI.ORDER:
                v += np.asarray(byb[b][:n], float) * W[b]
            clim.setdefault((tm, int(lead)), []).extend(v.tolist())
    clim = {k: float(np.mean(v)) for k, v in clim.items()}

    # historical C3S ratio per (init month, lead) for fitting the rain variant
    hist_ratio = {}
    for init, byl in craw.items():
        for lead, byb in byl.items():
            a = int(init[:4]) * 12 + int(init[5:7]) + int(lead)
            ty, tm = (a - 1) // 12, (a - 1) % 12 + 1
            base = clim.get((tm, int(lead)))
            if base is None or base <= 0:
                continue
            n = min(len(byb[b]) for b in NI.ORDER if b in byb)
            v = np.zeros(n)
            for b in NI.ORDER:
                v += np.asarray(byb[b][:n], float) * W[b]
            hist_ratio[(f"{int(init[:4]):04d}-{int(init[5:7]):02d}", int(lead))] = \
                (f"{ty:04d}-{tm:02d}", float(np.mean(v)) / base)

    # fit the C3S-rain monthly model on the hindcast overlap
    Xr, Yr = [], []
    for (im, lead), (tgt, ratio) in hist_ratio.items():
        if im not in rows or tgt not in rows:
            continue
        mo = int(tgt[5:7])
        s1, c1 = np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)
        lr = np.log(max(ratio, 0.05))
        Xr.append([lr, lr * s1, lr * c1, rows[im]["stor_end"],
                   rows[im]["ant90"] - 100.0, s1, c1])
        Yr.append(np.log(max(rows[tgt]["pct"], 5.0)))
    Xr, Yr = np.asarray(Xr, float), np.asarray(Yr, float)
    Ar = np.column_stack([np.ones(len(Yr)), Xr])
    beta_r, *_ = np.linalg.lstsq(Ar, Yr, rcond=None)
    sd_r = float(np.std(Yr - Ar @ beta_r))
    print(f"C3S-rain monthly fit: n={len(Yr)}  resid sd(log)={sd_r:.3f}")

    variants = {}
    for name in ("ONI_ISSUE", "ONI_FCST", "C3S_RAIN"):
        months = []
        prev_logit = np.full(NDRAW, OL._lg(gen[max(gen)]["share"]))
        for lead in range(1, MAXLEAD + 1):
            a = int(issue[:4]) * 12 + int(issue[5:7]) + lead
            ty, tm = (a - 1) // 12, (a - 1) % 12 + 1
            tgt = f"{ty:04d}-{tm:02d}"
            s1, c1 = np.sin(2 * np.pi * tm / 12), np.cos(2 * np.pi * tm / 12)
            if name == "C3S_RAIN":
                mem = c3s_rain.get(lead)
                base = clim.get((tm, lead))
                if mem is None or not base:
                    continue
                lr = np.log(np.maximum(mem / base, 0.05))
                lr = rng.choice(lr, NDRAW)
                mu = (beta_r[0] + beta_r[1] * lr + beta_r[2] * lr * s1
                      + beta_r[3] * lr * c1 + beta_r[4] * r0["stor_end"]
                      + beta_r[5] * (r0["ant90"] - 100.0) + beta_r[6] * s1
                      + beta_r[7] * c1)
                sd = sd_r
            else:
                use_tgt = name == "ONI_FCST"
                beta, sd = month_fit(rows, keys, lead, use_tgt)
                if use_tgt:
                    ens = ef.get(tgt)
                    o = (rng.choice(ens, NDRAW) if ens is not None
                         else np.full(NDRAW, r0["oni"]))
                else:
                    o = np.full(NDRAW, r0["oni"])
                mu = (beta[0] + beta[1] * o + beta[2] * o * s1 + beta[3] * o * c1
                      + beta[4] * r0["stor_end"] + beta[5] * (r0["ant30"] - 100.0)
                      + beta[6] * (r0["ant90"] - 100.0) + beta[7] * s1
                      + beta[8] * c1)
            draws = np.exp(mu + rng.normal(0, sd, NDRAW))
            t = (ty - 2013) + (tm - 6) / 12.0
            z = (beta_sh[0] + beta_sh[1] * np.log(np.maximum(draws, 5) / 100.0)
                 + beta_sh[2] * r0["stor_end"] + beta_sh[3] * t
                 + beta_sh[4] * s1 + beta_sh[5] * c1 + beta_sh[6] * prev_logit
                 + rng.normal(0, sst["resid_sd_logit"], NDRAW))
            prev_logit = z
            share = 100.0 / (1 + np.exp(-z))
            doy = datetime(ty, tm, 15).timetuple().tm_yday
            gw = d["norm_gwh"][doy] / 100.0
            q = lambda x, p: [round(float(np.percentile(x, y)), 1) for y in p]
            months.append({"month": tgt, "lead": lead,
                           "inflow_pct": dict(zip(("p10", "p50", "p90"),
                                                  q(draws, (10, 50, 90)))),
                           "inflow_gwh": dict(zip(("p10", "p50", "p90"),
                                                  q(draws * gw, (10, 50, 90)))),
                           "hydro_share_pct": dict(zip(("p10", "p50", "p90"),
                                                       q(share, (10, 50, 90)))),
                           "norm_gwh": round(float(d["norm_gwh"][doy]), 1)})
        variants[name] = months

    # daily balance-of-month
    daily = None
    try:
        dr = NI.daily_regime(d, None)
        resid = NI.national_residuals(d)
        rows_d = []
        for j, dt in enumerate(dr["dates"]):
            col = dr["paths"][:, j]
            col = col[np.isfinite(col)]
            R = resid.get(j + 1, np.zeros(0))
            full = (col[:, None] + R[None, :]).ravel() if len(R) > 30 else col
            full = np.clip(full, 0, None)
            doy = datetime.strptime(str(dt), "%Y-%m-%d").timetuple().tm_yday
            gw = d["norm_gwh"][doy] / 100.0
            rows_d.append({"date": str(dt), "lead": j + 1,
                           "p10": round(float(np.percentile(full, 10)), 1),
                           "p50": round(float(np.percentile(full, 50)), 1),
                           "p90": round(float(np.percentile(full, 90)), 1),
                           "gwh_p50": round(float(np.percentile(full, 50)) * gw, 1)})
        daily = rows_d
    except Exception as e:                          # noqa: BLE001
        print(f"daily regime unavailable: {repr(e)[:110]}")

    # ---- FUSE the daily and monthly halves for the current month ----
    # The seam is not a missing-rain problem.  GEFS was ingested and
    # verified for days 10-35 (gefs_verify.py): anomaly correlation against
    # gauge-corrected IMERG is +0.10 at d10-15, +0.07 at d16-22 and +0.01 by
    # d23-35, so subseasonal rain carries no usable day-to-day information
    # past the AIFS horizon and cannot close the gap.
    #
    # What actually causes the disagreement is that the monthly model is
    # issued from LAST month's state and never sees the month in progress.
    # For the current month the answer is already largely determined: the
    # observed days are known and the ensemble covers most of the rest.  So
    # that month is replaced by the fused value - observed month-to-date plus
    # the daily fan for the balance - with the residual monthly uncertainty
    # scaled by the fraction of the month still unresolved.
    reconcile = None
    if daily:
        cm = str(d["dates"][-1])[:7]
        mtd = [float(v) for dt, v in zip(d["dates"], d["pct"])
               if str(dt)[:7] == cm and np.isfinite(v)]
        fwd = [r["p50"] for r in daily if r["date"][:7] == cm]
        if mtd and fwd:
            implied = (sum(mtd) + sum(fwd)) / (len(mtd) + len(fwd))
            m1 = next((m for m in variants["ONI_ISSUE"] if m["month"] == cm), None)
            said = m1["inflow_pct"]["p50"] if m1 else None
            reconcile = (
                f"{cm}: {len(mtd)} days observed at {np.mean(mtd):.0f}% of norm, "
                f"{len(fwd)} days forecast at {np.mean(fwd):.0f}%  \u2192  implied "
                f"monthly mean {implied:.0f}%."
                + (f"  The monthly model, issued from {issue} state, said "
                   f"{said:.0f}% \u2014 it had not seen this month. Where they "
                   f"disagree inside the ensemble horizon, trust the daily half."
                   if said is not None else ""))
            print("\n" + reconcile)
    fused = {}
    if daily:
        cm = str(d["dates"][-1])[:7]
        mtd = [float(v) for dt, v in zip(d["dates"], d["pct"])
               if str(dt)[:7] == cm and np.isfinite(v)]
        fwd = [r for r in daily if r["date"][:7] == cm]
        if mtd and fwd:
            import calendar
            ndays = calendar.monthrange(int(cm[:4]), int(cm[5:7]))[1]
            nres = ndays - len(mtd) - len(fwd)          # days still unresolved
            frac_open = max(nres, 0) / ndays
            for name, months in variants.items():
                for m in months:
                    if m["month"] != cm:
                        continue
                    for q, key in ((10, "p10"), (50, "p50"), (90, "p90")):
                        fq = [r[key] for r in fwd]
                        # unresolved days fall back to the monthly quantile
                        tail = m["inflow_pct"][key] * nres
                        m["inflow_pct"][key] = round(
                            (sum(mtd) + sum(fq) + tail) / ndays, 1)
                    m["fused_with_daily"] = True
                    m["days_observed"] = len(mtd)
                    m["days_from_ensemble"] = len(fwd)
                    m["days_unresolved"] = int(max(nres, 0))
                    m["fraction_open"] = round(frac_open, 3)
                    fused[name] = m["inflow_pct"]["p50"]
    obs_recent = [{"date": str(dt), "pct": round(float(v), 1)}
                  for dt, v in zip(d["dates"][-75:], d["pct"][-75:])
                  if np.isfinite(v)]
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "reconcile": reconcile, "fused_current_month": fused,
           "observed_recent": obs_recent,
           "issue_month": issue, "oni_at_issue": round(r0["oni"], 2),
           "storage_anom": round(r0["stor_end"], 1),
           "enso_forecast_issue": ef_issue,
           "enso_forecast_n34": {m: [round(float(np.percentile(v, p)), 2)
                                     for p in (10, 50, 90)] for m, v in ef.items()},
           "c3s_rain_init": c3s_init, "share_model": sst,
           "daily": daily, "variants": variants}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))

    print(f"\n{'month':9}" + "".join(f"{k:>26}" for k in variants))
    print(f"{'':9}" + "".join(f"{'inflow p10/p50/p90':>26}" for _ in variants))
    ms = [m["month"] for m in variants["ONI_ISSUE"]]
    for i, mo in enumerate(ms):
        line = f"{mo:9}"
        for k in variants:
            v = variants[k][i]["inflow_pct"] if i < len(variants[k]) else None
            line += (f"{v['p10']:9.0f}{v['p50']:8.0f}{v['p90']:9.0f}" if v
                     else f"{'--':>26}")
        print(line)
    print(f"\n{'month':9}" + "".join(f"{k:>26}" for k in variants))
    print(f"{'':9}" + "".join(f"{'hydro share p10/p50/p90':>26}" for _ in variants))
    for i, mo in enumerate(ms):
        line = f"{mo:9}"
        for k in variants:
            v = variants[k][i]["hydro_share_pct"] if i < len(variants[k]) else None
            line += (f"{v['p10']:9.0f}{v['p50']:8.0f}{v['p90']:9.0f}" if v
                     else f"{'--':>26}")
        print(line)
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
