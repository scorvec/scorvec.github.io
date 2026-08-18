#!/usr/bin/env python3
"""Verify the C3S seasonal RAINFALL forecasts themselves against IMERG.

The seasonal inflow study found C3S rain adds ~+0.01 CRPS skill over a
statistical model.  That is indirect: it could mean the rain forecast is
poor, or that its information is already carried by ENSO state.  This
tests the rain forecast on its own terms.

Everything is in anomaly space, each side against ITS OWN climatology by
(target month, lead) - the model's from its hindcast, the observations'
from IMERG - so systematic wet/dry drift is removed and only the
year-to-year signal is scored.  That is the generous framing: a model
gets no credit or blame for its mean state.

Reference forecasts, both of which the model must beat to be useful:
  climatology  - zero anomaly every time
  ONI regression - a one-predictor least-squares fit of observed basin
    rain on the ONI known at init, cross-validated the same way.  If a
    single number available on the issue date forecasts Colombian rain
    as well as a coupled global model, the model is adding nothing.

    python scripts/sst/c3s_verify.py

Outputs: ~/colombia_hydro/out/c3s_verify.json
         ~/colombia_hydro/site/c3s_verify.webp
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perfect_rain_backtest as PR                                  # noqa: E402
import national_inflow as NI                                        # noqa: E402
import seasonal_inflow as SI                                        # noqa: E402

PRIV = Path.home() / "colombia_hydro"
OUT_JSON = PRIV / "out" / "c3s_verify.json"
OUT_PNG = PRIV / "site" / "c3s_verify.webp"
ORDER = NI.ORDER
MAXLEAD = 6


def observed_monthly():
    """Observed monthly-mean basin + national rain, mm/day, from IMERG."""
    d = PR.load_all()
    W = NI.basin_energy_weights()
    dates = d["dates"]
    ym = np.array([str(x)[:7] for x in dates])
    nat = sum(np.nan_to_num(d["rain_abs"][b]) * W[b] for b in ORDER)
    out = {}
    for m in sorted(set(ym)):
        k = ym == m
        if k.sum() < 20:
            continue
        rec = {b: float(np.nanmean(d["rain_abs"][b][k])) for b in ORDER}
        rec["NATIONAL"] = float(np.nanmean(nat[k]))
        out[m] = rec
    return out


def crps_ens(members, obs):
    m = np.sort(np.asarray(members, float))
    n = len(m)
    i = np.arange(n)
    return float(np.mean(np.abs(m - obs)) - np.sum((2 * i - n + 1) * m) / (n * n))


def crps_gauss(mu, sig, y):
    """Closed-form CRPS of a Gaussian — exact, no sampling noise."""
    from math import sqrt, pi, erf, exp
    z = (y - mu) / sig
    pdf = exp(-0.5 * z * z) / sqrt(2 * pi)
    cdf = 0.5 * (1 + erf(z / sqrt(2)))
    return float(sig * (z * (2 * cdf - 1) + 2 * pdf - 1 / sqrt(pi)))


def emos(fa, ens_sd, oa, yr, embargo=1):
    """Calibrate the ensemble: obs ~ N(a + b*ensmean, c + d*ens_var).

    The raw ensemble has a skilful mean but is not a usable probability -
    CRPS against climatology is negative because the spread does not
    represent monthly-mean uncertainty.  EMOS (nonhomogeneous Gaussian
    regression) fixes both halves at once: an affine correction of the
    mean, and a variance that is allowed to scale with the ensemble's own
    spread, so that spread earns its keep as a skill predictor rather
    than being trusted at face value.  b < 1 shrinks an over-confident
    mean; d > 0 means spread genuinely carries information about the
    error on that day.

    Fitted by direct CRPS minimisation, leave-one-year-out with a +/-1
    year embargo, so no case informs its own calibration.
    """
    from scipy.optimize import minimize
    fa, ens_sd, oa, yr = map(np.asarray, (fa, ens_sd, oa, yr))
    mu = np.full(len(oa), np.nan); sg = np.full(len(oa), np.nan)
    par = []
    for y in sorted(set(yr)):
        te = yr == y
        tr = np.abs(yr - y) > embargo
        if tr.sum() < 50 or te.sum() == 0:
            continue

        def loss(p, tr=tr):
            a, b, c, dd = p
            v = np.sqrt(np.maximum(c, 1e-4) ** 2 + np.maximum(dd, 0.0) *
                        ens_sd[tr] ** 2)
            m = a + b * fa[tr]
            return float(np.mean([crps_gauss(mi, vi, oi)
                                  for mi, vi, oi in zip(m, v, oa[tr])]))

        r = minimize(loss, x0=[0.0, 0.5, float(np.std(oa[tr])), 0.5],
                     method="Nelder-Mead",
                     options={"maxiter": 900, "xatol": 1e-3, "fatol": 1e-4})
        a, b, c, dd = r.x
        mu[te] = a + b * fa[te]
        sg[te] = np.sqrt(max(c, 1e-4) ** 2 + max(dd, 0.0) * ens_sd[te] ** 2)
        par.append([round(float(v), 3) for v in (a, b, c, dd)])
    ok = np.isfinite(mu)
    if ok.sum() < 40:
        return None
    cr = float(np.mean([crps_gauss(m, s_, o)
                        for m, s_, o in zip(mu[ok], sg[ok], oa[ok])]))
    sd0 = float(np.std(oa[ok]))
    crc = float(np.mean([crps_gauss(0.0, sd0, o) for o in oa[ok]]))
    pit = np.array([0.5 * (1 + __import__("math").erf((o - m) / (s_ * np.sqrt(2))))
                    for m, s_, o in zip(mu[ok], sg[ok], oa[ok])])
    return {"n": int(ok.sum()), "crps": round(cr, 3),
            "crps_clim": round(crc, 3),
            "crps_skill": round(1 - cr / crc, 3),
            "pit_mean": round(float(np.mean(pit)), 3),
            "coverage_10_90": round(float(np.mean((pit > .1) & (pit < .9))), 3),
            "mean_params_a_b_c_d": [round(float(np.mean([p[i] for p in par])), 3)
                                    for i in range(4)]}


def main() -> int:
    obs = observed_monthly()
    raw = json.loads((PRIV / "out" / "c3s_basin_precip.json").read_text())["data"]
    W = NI.basin_energy_weights()
    nh = json.loads((HERE.parent.parent / "assets" / "sst" / "data" /
                     "nino_history.json").read_text())
    oni = dict(zip(nh["months"], nh["series"]["oni"]["anom"]))

    # gather cases
    cases = []
    for init, byl in raw.items():
        im = init[:7]
        for lead_s, byb in byl.items():
            lead = int(lead_s)
            a = int(init[:4]) * 12 + int(init[5:7]) + lead
            ty, tm = (a - 1) // 12, (a - 1) % 12 + 1
            tgt = f"{ty:04d}-{tm:02d}"
            if tgt not in obs:
                continue
            n = min(len(byb[b]) for b in ORDER if b in byb)
            mem = np.zeros(n)
            for b in ORDER:
                mem += np.asarray(byb[b][:n], float) * W[b]
            cases.append({"init": im, "lead": lead, "target": tgt, "tmonth": tm,
                          "year": ty, "mem": mem, "obs": obs[tgt]["NATIONAL"],
                          "oni": oni.get(im, np.nan)})
    yrs = sorted({c["year"] for c in cases})
    print(f"cases: {len(cases)}  target years {yrs[0]}..{yrs[-1]}", flush=True)

    # climatologies by (target month, lead): model from its own ensemble
    mclim, oclim = {}, {}
    for c in cases:
        mclim.setdefault((c["tmonth"], c["lead"]), []).extend(c["mem"])
        oclim.setdefault(c["tmonth"], []).append(c["obs"])
    mclim = {k: float(np.mean(v)) for k, v in mclim.items()}
    oclim = {k: float(np.mean(v)) for k, v in oclim.items()}

    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "target": "monthly mean NATIONAL rainfall, mm/day, energy-weighted",
           "period": f"{yrs[0]}..{yrs[-1]}", "by_lead": {}, "by_season": {}}
    print(f"\n{'lead':>5}{'n':>5}{'ACC':>8}{'ONI-reg':>9}{'CRPSss':>9}"
          f"{'sd ratio':>10}{'bias mm/d':>11}")
    for lead in range(1, MAXLEAD + 1):
        sub = [c for c in cases if c["lead"] == lead]
        if len(sub) < 40:
            continue
        fa = np.array([np.mean(c["mem"]) - mclim[(c["tmonth"], lead)] for c in sub])
        oa = np.array([c["obs"] - oclim[c["tmonth"]] for c in sub])
        yr = np.array([c["year"] for c in sub])
        on = np.array([c["oni"] for c in sub], float)
        acc = float(np.corrcoef(fa, oa)[0, 1])
        # ONI regression, LOYO with +/-1y embargo — the cheap rival
        pr = np.full(len(oa), np.nan)
        for y in sorted(set(yr)):
            te = yr == y; tr = (np.abs(yr - y) > 1) & np.isfinite(on)
            if tr.sum() < 40:
                continue
            s1 = np.sin(2 * np.pi * np.array([c["tmonth"] for c in sub]) / 12)
            c1 = np.cos(2 * np.pi * np.array([c["tmonth"] for c in sub]) / 12)
            A = np.column_stack([np.ones(tr.sum()), on[tr], on[tr] * s1[tr],
                                 on[tr] * c1[tr]])
            b, *_ = np.linalg.lstsq(A, oa[tr], rcond=None)
            pr[te] = (b[0] + b[1] * on[te] + b[2] * on[te] * s1[te]
                      + b[3] * on[te] * c1[te])
        m = np.isfinite(pr)
        acc_oni = float(np.corrcoef(pr[m], oa[m])[0, 1]) if m.sum() > 40 else np.nan
        # probabilistic: ensemble anomaly vs observed anomaly
        cr = float(np.mean([crps_ens(c["mem"] - mclim[(c["tmonth"], lead)],
                                     c["obs"] - oclim[c["tmonth"]]) for c in sub]))
        crc = float(np.mean([crps_ens(oa, o) for o in oa]))
        out["by_lead"][lead] = {
            "n": len(sub), "acc": round(acc, 3),
            "acc_oni_regression": round(acc_oni, 3) if np.isfinite(acc_oni) else None,
            "crps": round(cr, 3), "crps_clim": round(crc, 3),
            "crps_skill": round(1 - cr / crc, 3),
            "sd_ratio_fcst_obs": round(float(np.std(fa) / np.std(oa)), 3),
            "bias_mm_day": round(float(np.mean(
                [np.mean(c["mem"]) - c["obs"] for c in sub])), 3)}
        es = np.array([float(np.std(c["mem"])) for c in sub])
        cal = emos(fa, es, oa, yr)
        if cal:
            out["by_lead"][lead]["emos_calibrated"] = cal
        v = out["by_lead"][lead]
        print(f"{lead:5d}{v['n']:5d}{v['acc']:8.3f}"
              f"{(v['acc_oni_regression'] if v['acc_oni_regression'] is not None else float('nan')):9.3f}"
              f"{v['crps_skill']:+9.3f}{v['sd_ratio_fcst_obs']:10.2f}"
              f"{v['bias_mm_day']:+11.2f}", flush=True)

    # seasonal breakdown at short lead — is DJF (peak ENSO) any better?
    print(f"\n{'season (lead 1-3)':20}{'n':>5}{'ACC':>8}{'ONI-reg':>9}")
    seas = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    for name, mons in seas.items():
        sub = [c for c in cases if c["tmonth"] in mons and c["lead"] <= 3]
        if len(sub) < 40:
            continue
        fa = np.array([np.mean(c["mem"]) - mclim[(c["tmonth"], c["lead"])] for c in sub])
        oa = np.array([c["obs"] - oclim[c["tmonth"]] for c in sub])
        on = np.array([c["oni"] for c in sub], float)
        mm = np.isfinite(on)
        out["by_season"][name] = {
            "n": len(sub), "acc": round(float(np.corrcoef(fa, oa)[0, 1]), 3),
            "corr_obs_vs_oni": round(float(np.corrcoef(on[mm], oa[mm])[0, 1]), 3)}
        v = out["by_season"][name]
        print(f"{name:20}{v['n']:5d}{v['acc']:8.3f}{v['corr_obs_vs_oni']:9.3f}")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    try:
        figure(out)
    except Exception as e:                          # noqa: BLE001
        print(f"figure failed: {repr(e)[:150]}")
    print(f"\nwrote {OUT_JSON}")
    return 0


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, INK = "#13273d", "#1a2733"
    fig = plt.figure(figsize=(14.6, 6.4))
    hd = fig.add_axes([0, 0.925, 1, 0.075]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes, facecolor=NAVY))
    hd.text(0.013, 0.62, "C3S SEASONAL RAINFALL — VERIFIED AGAINST IMERG",
            transform=hd.transAxes, color="white", fontsize=15,
            fontweight="bold", va="center")
    hd.text(0.013, 0.2, "national basin rain, monthly means, anomalies against each "
            f"side's own climatology · {out['period']}",
            transform=hd.transAxes, color="#b9c6d4", fontsize=9, va="center")
    L = sorted(int(k) for k in out["by_lead"])
    G = lambda k: [out["by_lead"][str(l)][k] if str(l) in out["by_lead"]
                   else out["by_lead"][l][k] for l in L]
    ax = fig.add_axes([0.055, 0.13, 0.27, 0.70])
    ax.plot(L, G("acc"), color="#1f7a4d", lw=2.3, marker="o", ms=6,
            label="C3S ensemble mean")
    ax.plot(L, [v if v is not None else np.nan for v in G("acc_oni_regression")],
            color="#c62828", lw=1.8, marker="s", ms=5, ls="--",
            label="ONI regression (cheap rival)")
    ax.axhline(0, color="0.4", lw=1.0)
    ax.set_xlabel("lead, months", fontsize=9.5)
    ax.set_ylabel("anomaly correlation (ACC)", fontsize=9.5)
    ax.set_title("It does have rain skill", fontsize=11, fontweight="bold",
                 loc="left", color=INK)
    ax.legend(fontsize=8.5); ax.grid(lw=0.25, alpha=0.5); ax.tick_params(labelsize=8.5)

    ax2 = fig.add_axes([0.385, 0.13, 0.27, 0.70])
    ax2.bar(L, G("crps_skill"), color=["#c62828" if v < 0 else "#1f7a4d"
                                       for v in G("crps_skill")], alpha=0.9)
    ax2.axhline(0, color="0.4", lw=1.1)
    ax2.set_xlabel("lead, months", fontsize=9.5)
    ax2.set_ylabel("CRPS skill vs climatology", fontsize=9.5)
    ax2.set_title("…but not as a distribution", fontsize=11, fontweight="bold",
                  loc="left", color=INK)
    ax2.grid(lw=0.25, alpha=0.5, axis="y"); ax2.tick_params(labelsize=8.5)

    ax3 = fig.add_axes([0.715, 0.13, 0.26, 0.70])
    ax3.plot(L, G("bias_mm_day"), color="#b35806", lw=2.3, marker="o", ms=6)
    ax3.axhline(0, color="0.4", lw=1.0)
    ax3.set_xlabel("lead, months", fontsize=9.5)
    ax3.set_ylabel("mean bias, mm/day", fontsize=9.5)
    ax3.set_title("…and it is very wet", fontsize=11, fontweight="bold",
                  loc="left", color=INK)
    ax3.grid(lw=0.25, alpha=0.5); ax3.tick_params(labelsize=8.5)
    fig.text(0.055, 0.02, "ACC is computed on anomalies, so the bias costs it nothing "
             "there — but a raw C3S rainfall number is unusable without correction, "
             "and the ensemble is not calibrated enough to use as a probability.",
             fontsize=7.8, color="#5a6b7a")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=118); plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    raise SystemExit(main())
