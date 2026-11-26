"""Scoring for verify_gust.py: every gust method against the airport truth, fitted on 2010-2019 and scored on
2020-2025, with station-block bootstrap CIs (the same draws for every method, so differences are paired).

Methods (m/s internally, knots in the report):
  sfc x k        the sounding's surface wind times a gust factor fitted on the fit years
  PBL mean       the app's row as shown, and recalibrated (a + b x, fitted)
  PBL max        the strongest wind below the PBL top, as is and recalibrated
  DCAPE          a + b sqrt(2 DCAPE)
  Brasseur       Brasseur's constants (alpha 1, lam 150 m, Ric 0.25), as is and recalibrated
  Brasseur*      ours: the constants chosen on the fit years from the dataset's grid, then a + b gust + c sfc
  GBM            gradient-boosted quantile regression (p10, p50, p90) on the profile features and the physics
                 estimates: the ceiling on what the profile alone can tell
A method with no value for a launch (no PBL top, a profile too short) falls back to the surface wind, so every method
is scored on every launch. Fits minimise the mean absolute error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gustport import ALPHAS, LAMS, RICS
from verify_gust import KT, Y0, Y1

FIT_LAST = 2019
B = 2000
RNG = np.random.default_rng(20261126)
THRESH_KT = (39, 50)  # Wind Advisory (45 mph) and High Wind Warning (58 mph) gusts
BRASSEUR = "1_150_0.25"
FEATURES = ["sfc", "pbl_m", "pbl_mean", "pbl_max", "dcape", "sbcape", "top_m", "hour", "month", "elev",
            *[f"w{h}" for h in (250, 500, 1000, 1500, 2000, 3000)], *[f"dthv{h}" for h in (250, 500, 1000, 1500, 2000, 3000)],
            f"bw_{BRASSEUR}", f"bu_{BRASSEUR}", f"bz_{BRASSEUR}"]  # fmt: skip


class Boot:
    """Station-block bootstrap over per-station sums."""

    def __init__(self, groups):
        self.keys, self.idx = np.unique(np.asarray(groups), return_inverse=True)
        S = len(self.keys)
        draws = RNG.integers(0, S, size=(B, S))
        self.W = np.zeros((B, S))
        for k in range(B):
            np.add.at(self.W[k], draws[k], 1)

    def sums(self, x):
        return np.bincount(self.idx, weights=np.asarray(x, float), minlength=len(self.keys))

    def mean(self, x, n=None):
        """(point, lo, hi) of sum(x)/sum(n) (n = ones by default)."""
        n = np.ones(len(x)) if n is None else n
        sx, sn = self.sums(x), self.sums(n)
        with np.errstate(invalid="ignore", divide="ignore"):
            bt = (self.W @ sx) / (self.W @ sn)
        return (float(sx.sum() / sn.sum()), *map(float, np.nanpercentile(bt, [2.5, 97.5])))

    def contingency(self, pred, obs):
        """POD, FAR, CSI -> {name: (point, lo, hi)}."""
        pred, obs = np.asarray(pred, bool), np.asarray(obs, bool)
        a, b, c = (self.sums(m) for m in (pred & obs, pred & ~obs, ~pred & obs))
        out = {}
        for name, f in (("POD", lambda a, b, c: a / (a + c)), ("FAR", lambda a, b, c: b / (a + b)),
                        ("CSI", lambda a, b, c: a / (a + b + c))):  # fmt: skip
            with np.errstate(invalid="ignore", divide="ignore"):
                pt = f(a.sum(), b.sum(), c.sum())
                bt = f(self.W @ a, self.W @ b, self.W @ c)
            out[name] = (float(pt), *map(float, np.nanpercentile(bt, [2.5, 97.5])))
        return out


def mae_fit(cols, y):
    """Least-absolute-deviation fit of y on [1, cols...] -> coefficients."""
    X = np.column_stack([np.ones(len(y)), *cols])
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    res = minimize(lambda b: np.abs(X @ b - y).mean(), beta0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})  # fmt: skip
    return res.x


def apply(beta, cols):
    return beta[0] + sum(b * c for b, c in zip(beta[1:], cols, strict=True))


def fill(df, col):
    """A method's value, the surface wind where it has none."""
    return df[col].astype(float).fillna(df["sfc"].astype(float)).to_numpy()


def fmt(t, d=1):
    p, lo, hi = t
    return "-" if not np.isfinite(p) else f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def score(out: Path) -> None:
    df = pd.read_parquet(out / "pairs.parquet")
    df = df[(df.year >= Y0) & (df.year <= Y1) & df.a_peak.notna() & df.sfc.notna()].reset_index(drop=True)
    fit, test = df[df.year <= FIT_LAST], df[df.year > FIT_LAST]
    yf, yt = fit.a_peak.to_numpy(float), test.a_peak.to_numpy(float)
    preds, notes = {}, {}

    # baselines
    k = mae_fit([fill(fit, "sfc")], yf)
    preds["sfc x k"] = apply(k, [fill(test, "sfc")])
    notes["sfc x k"] = f"{k[0] * KT:+.1f} kt + {k[1]:.2f} x sfc"
    preds["PBL mean (app)"] = fill(test, "pbl_mean")
    for name, col in (("PBL mean", "pbl_mean"), ("PBL max", "pbl_max"), ("Brasseur", f"bw_{BRASSEUR}")):
        if name != "PBL mean":
            preds[f"{name} (as is)"] = fill(test, col)
        b = mae_fit([fill(fit, col)], yf)
        preds[f"{name}, recalibrated"] = apply(b, [fill(test, col)])
        notes[f"{name}, recalibrated"] = f"{b[0] * KT:+.1f} kt + {b[1]:.2f} x"
    dc = lambda d: np.sqrt(2 * d.dcape.astype(float).clip(lower=0).fillna(0).to_numpy())  # noqa: E731
    b = mae_fit([dc(fit)], yf)
    preds["DCAPE"] = apply(b, [dc(test)])

    # ours: the Brasseur constants chosen on the fit years, then blended with the surface wind
    best = None
    for a in ALPHAS:
        for lam in LAMS:
            for ric in RICS:
                key = f"{a:g}_{lam:g}_{ric:g}"
                cols = [fill(fit, f"bw_{key}"), fill(fit, "sfc")]
                beta = mae_fit(cols, yf)
                m = np.abs(apply(beta, cols) - yf).mean()
                if best is None or m < best[0]:
                    best = (m, key, beta)
    _, key, beta = best
    preds["Brasseur* (ours)"] = apply(beta, [fill(test, f"bw_{key}"), fill(test, "sfc")])
    notes["Brasseur* (ours)"] = (f"alpha, lam, Ric = {key.replace('_', ', ')}; {beta[0] * KT:+.1f} kt + {beta[1]:.2f} x gust"
                                 f" + {beta[2]:.2f} x sfc")  # fmt: skip

    # the ceiling: gradient-boosted quantiles
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xf, Xt = fit[FEATURES].astype(float), test[FEATURES].astype(float)
    q = {}
    for p in (0.1, 0.5, 0.9):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=p, max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=31, min_samples_leaf=200, random_state=0)  # fmt: skip
        q[p] = m.fit(Xf, yf).predict(Xt)
    preds["GBM median"] = q[0.5]

    bt = Boot(test.wmo.to_numpy())
    rows = []
    ref_name = None
    for name, pr in preds.items():
        e = pr - yt
        rows.append({"method": name, "MAE": bt.mean(np.abs(e) * KT), "bias": bt.mean(e * KT)})
    base = [r for r in rows if not r["method"].startswith(("Brasseur*", "GBM"))]
    ref_name = min(base, key=lambda r: r["MAE"][0])["method"]
    lines = [f"# Gust estimates from soundings: test years {FIT_LAST + 1}-{Y1}",
             f"{len(test)} launches at {test.wmo.nunique()} stations (fit {len(fit)} launches, {Y0}-{FIT_LAST}); truth "
             f"= highest gust, peak wind or sustained wind in the 2 h after release; {test.a_censored.mean():.0%} with "
             "no gust reported. 95 % station-block bootstrap intervals.", "",
             f"| method | MAE kt | bias kt | MAE minus {ref_name} | fitted |", "|---|---|---|---|---|"]  # fmt: skip
    ref = np.abs(preds[ref_name] - yt) * KT
    for r in rows:
        d = bt.mean(np.abs(preds[r["method"]] - yt) * KT - ref)
        lines.append(
            f"| {r['method']} | {fmt(r['MAE'])} | {fmt(r['bias'])} | {fmt(d, 2)} | {notes.get(r['method'], '')} |"
        )
    for thr in THRESH_KT:
        obs = yt * KT >= thr
        lines += ["", f"## Gusts of {thr} kt or more ({int(obs.sum())} test launches)", "",
                  "| method | POD | FAR | CSI |", "|---|---|---|---|"]  # fmt: skip
        for name, pr in preds.items():
            c = bt.contingency(pr * KT >= thr, obs)
            lines.append(f"| {name} | {fmt(c['POD'], 2)} | {fmt(c['FAR'], 2)} | {fmt(c['CSI'], 2)} |")
    inside = (yt >= q[0.1]) & (yt <= q[0.9])
    lines += ["", f"GBM p10-p90 range: covers {fmt(bt.mean(inside.astype(float) * 100), 0)} % of test launches "
              f"(80 % nominal), mean width {np.mean(q[0.9] - q[0.1]) * KT:.1f} kt."]  # fmt: skip
    # where it matters: by subset, the two best physics methods and the reference
    subsets = {"00Z": test.hour.to_numpy() == 0, "12Z": test.hour.to_numpy() == 12,
               "thunder": test.a_ts.to_numpy(bool), "no thunder": ~test.a_ts.to_numpy(bool),
               "DJF": test.month.isin([12, 1, 2]).to_numpy(), "MAM": test.month.isin([3, 4, 5]).to_numpy(),
               "JJA": test.month.isin([6, 7, 8]).to_numpy(), "SON": test.month.isin([9, 10, 11]).to_numpy(),
               "truth >= 30 kt": yt * KT >= 30}  # fmt: skip
    show = [ref_name, "Brasseur* (ours)", "GBM median"]
    lines += [
        "",
        "## MAE (kt) by subset",
        "",
        "| subset | n | " + " | ".join(show) + " |",
        "|---|---|" + "---|" * len(show),
    ]
    for sname, m in subsets.items():
        sb = Boot(test.wmo.to_numpy()[m])
        lines.append(
            f"| {sname} | {int(m.sum())} | "
            + " | ".join(fmt(sb.mean(np.abs(preds[s][m] - yt[m]) * KT)) for s in show)
            + " |"
        )
    (out / "score.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps({"reference": ref_name, "brasseur_star": {"constants": key,
                                                  "beta": [float(x) for x in beta]},
                                                  "rows": rows}, indent=1, default=float))  # fmt: skip
    print("\n".join(lines), flush=True)
