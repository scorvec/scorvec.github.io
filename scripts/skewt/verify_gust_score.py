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
export() builds the model the Skew-T app ships (skewt/gust_model.json) and a parity fixture for its JS port.
A method with no value for a launch (no PBL top, a profile too short) falls back to the surface wind, so every method
is scored on every launch. Fits minimise the mean absolute error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import gustport as GP
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


# ---- the shipped model (skewt/gust_model.json): profile-only quantile GBMs, the range conformally widened ----
# No hour, month or elevation: the stations are almost all in the US, so those would stand for US local time, season
# and terrain and mislead anywhere else; the sounding's own stability says day or night.
SHIP_FEATURES = ["sfc", "pbl_m", "pbl_mean", "pbl_max", "dcape", "sbcape", "top_m",
                 *[f"w{h}" for h in (250, 500, 1000, 1500, 2000, 3000)],
                 *[f"dthv{h}" for h in (250, 500, 1000, 1500, 2000, 3000)],
                 f"bw_{BRASSEUR}", f"bu_{BRASSEUR}", f"bz_{BRASSEUR}"]  # fmt: skip
TRAIN_LAST, CAL_LAST = 2017, 2019  # train 2010-2017, widen the range on 2018-2019, score 2020-2025
COVER = 0.8
CONFIGS = [(400, 31, 0.05), (300, 15, 0.07), (200, 15, 0.1), (150, 7, 0.1)]  # (iterations, leaves, learning rate)


def _gbm(p, cfg):
    from sklearn.ensemble import HistGradientBoostingRegressor

    it, leaves, lr = cfg
    return HistGradientBoostingRegressor(loss="quantile", quantile=p, max_iter=it, learning_rate=lr,
                                         max_leaf_nodes=leaves, min_samples_leaf=200, random_state=0)  # fmt: skip


def _trees(m):
    """A fitted HistGradientBoostingRegressor -> {base, root, f, t, l, r, m}: every tree's nodes in flat arrays (f = feature
    index, -1 on a leaf; t = threshold, or the leaf's value; l/r = child node indices; m = 1 when a missing value goes
    left). A sample goes left when x <= t (the threshold nudged up a part in 1e9)."""
    f, t, lft, rgt, miss, root = [], [], [], [], [], []
    for (pred,) in m._predictors:
        off = len(f)
        root.append(off)
        for n in pred.nodes:
            leaf = bool(n["is_leaf"])
            f.append(-1 if leaf else int(n["feature_idx"]))
            # thresholds sit ON data values (winds in whole knots): nudged up a part in 1e9 so a tie goes left in Python
            # and in the browser alike, whatever the last bit of their arithmetic
            th = float(n["num_threshold"])  # +inf on a split by missingness alone: 1e308 keeps JSON (and JS) valid
            t.append(float(n["value"]) if leaf else min(th + 1e-9 * max(1.0, abs(th)), 1e308))
            lft.append(0 if leaf else off + int(n["left"]))
            rgt.append(0 if leaf else off + int(n["right"]))
            miss.append(int(bool(n["missing_go_to_left"])))
    base = float(np.ravel(m._baseline_prediction)[0])
    return {"base": base, "root": root, "f": f, "t": t, "l": lft, "r": rgt, "m": miss}


def _walk(tr, X):
    """Predict from _trees() output (the JS app does exactly this)."""
    out = np.empty(len(X))
    f, t, lft, rgt, miss = (np.asarray(tr[k]) for k in "ftlrm")
    for i, x in enumerate(X):
        s = tr["base"]
        for root in tr["root"]:
            k = root
            while f[k] >= 0:
                v = x[f[k]]
                k = lft[k] if (miss[k] if np.isnan(v) else v <= t[k]) else rgt[k]
            s += t[k]
        out[i] = s
    return out


def _cut(prof, z_agl):
    """Pressure (Pa) at a height AGL on the profile, linear in log p; None above it."""
    z = np.asarray(prof["H"], float) - prof["H"][0]
    if z_agl is None or not np.isfinite(z_agl) or z_agl > z[-1]:
        return None
    return float(np.exp(np.interp(z_agl, z, np.log(np.asarray(prof["P"], float)))))


def export(out: Path) -> None:
    import tarfile

    df = pd.read_parquet(out / "pairs.parquet")
    df = df[(df.year >= Y0) & (df.year <= Y1) & df.a_peak.notna() & df.sfc.notna()].reset_index(drop=True)
    tr, cal, te = df[df.year <= TRAIN_LAST], df[(df.year > TRAIN_LAST) & (df.year <= CAL_LAST)], df[df.year > CAL_LAST]
    X = {k: d[SHIP_FEATURES].astype(float).to_numpy() for k, d in (("tr", tr), ("cal", cal), ("te", te))}
    y = {k: d.a_peak.to_numpy(float) for k, d in (("tr", tr), ("cal", cal), ("te", te))}
    lines = [f"# Shipped gust model: trained {Y0}-{TRAIN_LAST} ({len(tr)} launches), range widened on "
             f"{TRAIN_LAST + 1}-{CAL_LAST} ({len(cal)}), scored on {CAL_LAST + 1}-{Y1} ({len(te)}, {te.wmo.nunique()} "
             "stations)", "", "## Model size, chosen on the calibration years (median MAE, kt)", ""]  # fmt: skip
    # the smallest model within 0.03 kt of the best on the calibration years (never the test years)
    maes = {}
    for cfg in CONFIGS:
        maes[cfg] = float(np.abs(_gbm(0.5, cfg).fit(X["tr"], y["tr"]).predict(X["cal"]) - y["cal"]).mean() * KT)
        lines.append(f"- {cfg[0]} trees x {cfg[1]} leaves (lr {cfg[2]}): {maes[cfg]:.3f}")
    best = min(maes.values())
    cfg = min((c for c in CONFIGS if maes[c] <= best + 0.03), key=lambda c: c[0] * c[1])
    models = {p: _gbm(p, cfg).fit(X["tr"], y["tr"]) for p in (0.1, 0.5, 0.9)}
    q = {k: {p: m.predict(X[k]) for p, m in models.items()} for k in ("cal", "te")}
    # conformalised quantile regression (Romano et al. 2019): widen p10-p90 by the 80th percentile of the misses
    sc = np.maximum(q["cal"][0.1] - y["cal"], y["cal"] - q["cal"][0.9])
    n = len(sc)
    widen = float(np.quantile(sc, min(1.0, np.ceil((n + 1) * COVER) / n)))
    lo = np.maximum(0, np.minimum(q["te"][0.1] - widen, q["te"][0.5]))
    hi = np.maximum(q["te"][0.9] + widen, q["te"][0.5])
    yt, med = y["te"], q["te"][0.5]
    bt = Boot(te.wmo.to_numpy())
    app = fill(te, "pbl_mean")
    inside = (yt >= lo) & (yt <= hi)
    lines += ["", f"Chosen: {cfg[0]} trees x {cfg[1]} leaves. Range widened by {widen * KT:+.2f} kt each side.", "",
              "## Test years", "", "| | value |", "|---|---|",
              f"| median MAE kt | {fmt(bt.mean(np.abs(med - yt) * KT))} |",
              f"| app PBL mean wind MAE kt | {fmt(bt.mean(np.abs(app - yt) * KT))} |",
              f"| median minus PBL mean, MAE kt | {fmt(bt.mean((np.abs(med - yt) - np.abs(app - yt)) * KT), 2)} |",
              f"| median bias kt | {fmt(bt.mean((med - yt) * KT))} |",
              f"| range covers % (nominal {COVER:.0%}) | {fmt(bt.mean(inside * 100.0), 0)} |",
              f"| range mean width kt | {np.mean(hi - lo) * KT:.1f} |"]  # fmt: skip
    for thr in THRESH_KT:
        obs = yt * KT >= thr
        for name, pr in (("median", med), ("top of range", hi), ("PBL mean (app)", app)):
            c = bt.contingency(pr * KT >= thr, obs)
            lines.append(f"| >= {thr} kt by {name}: POD / FAR / CSI | {fmt(c['POD'], 2)} / {fmt(c['FAR'], 2)} / "
                         f"{fmt(c['CSI'], 2)} |")  # fmt: skip
    ts = te.a_ts.to_numpy(bool)
    for name, m in (("thunder", ts), ("no thunder", ~ts)):
        sb = Boot(te.wmo.to_numpy()[m])
        lines.append(f"| {name} ({int(m.sum())}): median MAE / covers % | {fmt(sb.mean(np.abs(med[m] - yt[m]) * KT))} / "
                     f"{fmt(sb.mean(inside[m] * 100.0), 0)} |")  # fmt: skip
    trees = {f"p{int(p * 100)}": _trees(m) for p, m in models.items()}
    for p, m in models.items():  # the flat trees reproduce sklearn
        assert np.allclose(_walk(trees[f"p{int(p * 100)}"], X["te"][:2000]), m.predict(X["te"][:2000]), atol=1e-9)
    stations = sorted(df.icao.unique().tolist())
    model = {"features": SHIP_FEATURES, "units": "m/s, m AGL, J/kg, K", "widen": float(f"{widen:.5g}"),
             "trained": f"{Y0}-{TRAIN_LAST}", "stations": stations, **trees}  # fmt: skip
    (out / "gust_model.json").write_text(json.dumps(model, separators=(",", ":"), allow_nan=False))  # JS reads it
    lines.append(f"\nModel file {(out / 'gust_model.json').stat().st_size / 1e6:.2f} MB.")
    # parity fixture for the app's port: test launches' stored profiles (<= 4 km AGL), the features gustport computes
    # from them and the model's output
    with tarfile.open(out / "profiles_npz.tar") as tf:
        tf.extractall(out, filter="data")
    rng = np.random.default_rng(7)
    cases = []
    for w in rng.choice(te.wmo.unique(), 12, replace=False):
        z = np.load(out / "profiles" / f"{w}.npz")
        keys = {k: i for i, k in enumerate(z["key"])}
        sub = te[te.wmo == w]
        for _, r in sub.iloc[rng.choice(len(sub), min(25, len(sub)), replace=False)].iterrows():
            i = keys.get(f"{r.date}_{int(r.hour):02d}")
            if i is None:
                continue
            a = z["data"][:, z["offsets"][i] : z["offsets"][i + 1]].astype(float)
            prof = {k: a[j].tolist() for j, k in enumerate("PHTDUV")}
            pa = _cut(prof, r.pbl_m)
            num = lambda v: None if v is None or not np.isfinite(v) else float(v)  # noqa: E731
            dc, sb = num(r.dcape), num(r.sbcape)
            feats = GP.features(prof, pa, dc, sb)
            xv = np.array([[np.nan if feats[k] is None else feats[k] for k in SHIP_FEATURES]], float)
            pr = [float(models[p].predict(xv)[0]) for p in (0.1, 0.5, 0.9)]
            cases.append({"prof": prof, "pbl_pa": pa, "dcape": dc, "sbcape": sb,
                          "feats": {k: feats[k] for k in SHIP_FEATURES}, "pred": pr})  # fmt: skip
    (out / "gust_fixture.json").write_text(json.dumps(cases, separators=(",", ":")))
    lines.append(f"Parity fixture: {len(cases)} test launches.")
    (out / "export.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
