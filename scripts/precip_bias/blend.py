#!/usr/bin/env python3
"""Model weights of the tropical rainfall forecast's blend (circulation.html#rainfc), fitted and tested out of sample.

The blend is a mixture of the four ensembles' bias-corrected members in which model m's members share its weight w_m, so
the blend's tercile chances are P = sum_m w_m p_m (p_m: the share of m's members in the tercile).

Score  the fair ranked probability score of the weekly tercile forecast: the cumulative categories "below normal" and
       "below or near normal" of the GPCP 2001-2020 normal week (gpcp_clim.py), against IMERG Late scaled to Final (the
       truth of the bias correction), on every second 1° cell in each direction (neighbours carry nearly the same
       information) where a normal week is testable (build_bias.DRY_Q67), cos-latitude weighted. Fair (Ferro 2014): the
       sampling noise of a finite ensemble, sum_m w_m^2 p_m (1 - p_m) / (n_m - 1) per category, is taken off, so the score
       is that of each model's full distribution and the six archived members do not reward spreading weight for its own
       sake. The score is then a quadratic form in w: per run A, b, c with score(w) = w'Aw - 2b'w + c.
Data   the bias archive (members 1-6 of every 00Z run, current model cycles: build_bias.VERSIONS), the runs all fitted
       models share and IMERG has verified. Each run is corrected with ratios estimated WITHOUT its own 21-day block and
       the runs within BUFFER_DAYS of it, whose verifying weeks overlap or adjoin the block's (global_bias.estimate: the
       same estimator and significance gate, its season kernel centred on that block), so no run is scored with a
       correction its own block of observations shaped.
Fit    per lead week, one weight per model on the simplex in steps of STEP, minimising the summed score. A model joins the
       fit once the archive it shares with the models already in holds FIT_BLOCKS 21-day blocks of verified runs (one
       more than global_bias.MIN_BLOCKS, so its bias can still be estimated with any block left out); until then it stays
       out of the blend (weight 0; it is still shown on its own), so a model whose rain has not been verified never
       carries weight, and the fitted models split the whole blend. With fewer than two fitted models there is nothing to
       fit: the blend is the equal mix of the fitted one, or of every model while none is.
Test   leave-one-block-out cross-validation: weights fitted without block k and without the runs within BUFFER_DAYS of
       it (their verifying weeks overlap or adjoin block k's) score block k, and equal weights score the same runs.
       Skill = 1 - score(fitted) / score(equal), summed over blocks, with a 90 % interval from resampling whole blocks.
       The fitted weights (refitted on all runs) are used only when that interval lies above zero, a one-sided 5 % test
       per lead week (10 % for the two together); otherwise the blend keeps equal weights and the page says the
       optimisation did not beat them.

    python scripts/precip_bias/blend.py --data DIR --ref DIR --out DIR
-> DIR/blend_weights_global.json: per week the weights used, the fitted ones, the cross-validated skill with its interval,
   the equal-weight blend's skill against the climatological tercile forecast and against each fitted model alone,
   runs, blocks and period, and a descriptive bias summary (bias_summary) of each model and the blend, raw and corrected.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid as G
import build_bias as BB
import global_bias as GB

STEP, NBOOT, BUFFER_DAYS, SUB = 0.025, 2000, 14, 2
FIT_BLOCKS = GB.MIN_BLOCKS + 1
CLIM_CUM = (1 / 3, 2 / 3)  # the climatological forecast's cumulative chances
BAND, WET_MM = 30.0, 7.0  # the bias summary: 30°S-30N; local bias only where the verifying weeks average >= 1 mm/day


# ── the score ────────────────────────────────────────────────────────────────────────────────────────────────────────
def cum_probs(X: np.ndarray, q33: np.ndarray, q67: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Members X (member, cells) -> the chances of "below normal" and "below or near normal" (2, cells) and their fair
    sampling variances p (1 - p) / (n - 1)."""
    p = np.stack([(q33[None] > X).mean(0), (q67[None] >= X).mean(0)])
    return p, p * (1 - p) / (X.shape[0] - 1)


def quad(P: np.ndarray, V: np.ndarray, O: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fair-RPS statistics of one run: P, V (model, 2, cells), O (2, cells) 0/1, w (cells) -> A (M, M), b (M), c such that
    the summed score of weights x is x'Ax - 2b'x + c."""
    A = np.einsum("mkc,lkc,c->ml", P, P, w) - np.diag(np.einsum("mkc,c->m", V, w))
    b = np.einsum("mkc,kc,c->m", P, O, w)
    return A, b, float(np.einsum("kc,c->", O, w))  # O is 0 or 1: O^2 = O


def score(A, b, c, x) -> float:
    return float(x @ A @ x - 2 * b @ x + c)


def simplex(m: int, step: float = STEP) -> np.ndarray:
    """Every weight vector of m models on the simplex in steps of `step` (rows)."""
    n = round(1 / step)
    return np.array([[*k, n - sum(k)] for k in itertools.product(range(n + 1), repeat=m - 1) if sum(k) <= n]) / n


def fit(A, b, cand) -> np.ndarray:
    """The candidate weight vector with the lowest summed score."""
    return cand[np.argmin(np.einsum("km,ml,kl->k", cand, A, cand) - 2 * cand @ b)]


def cross_validate(A, b, c, s_clim, days, cand, rng, names=()) -> dict:
    """Leave-one-block-out test of fitted against equal weights. A (run, M, M), b (run, M), c, s_clim (run): per-run
    statistics; days: run dates (sorted). -> per-block scores, the skill of the fitted weights over equal ones, of the
    equal-weight blend over climatology and over each model alone (`names`, in A's order; a single model needs no fit, so
    its score is out of sample as it stands), each with a 90 % block-bootstrap interval."""
    days = np.array(days, dtype="datetime64[D]")
    blk = (days - days[0]).astype(int) // BB.BLOCK_DAYS
    ub = np.unique(blk)
    eq = np.full(A.shape[1], 1.0 / A.shape[1])
    buf = np.timedelta64(BUFFER_DAYS, "D")
    s_fit, s_eq, s_cl, s_one = [], [], [], []
    for k in ub:
        test = blk == k
        near = (days >= days[test][0] - buf) & (days <= days[test][-1] + buf)
        x = fit(A[~near].sum(0), b[~near].sum(0), cand)
        At, bt, ct = A[test].sum(0), b[test].sum(0), c[test].sum()
        s_fit.append(score(At, bt, ct, x))
        s_eq.append(score(At, bt, ct, eq))
        s_cl.append(s_clim[test].sum())
        s_one.append([score(At, bt, ct, e) for e in np.eye(A.shape[1])])
    s_fit, s_eq, s_cl, s_one = map(np.array, (s_fit, s_eq, s_cl, s_one))
    W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=NBOOT).astype("float64")
    gain = 1 - (W @ s_fit) / (W @ s_eq)
    rpss = 1 - (W @ s_eq) / (W @ s_cl)
    vs_one = {}
    for j, m in enumerate(names):
        g = 1 - (W @ s_eq) / (W @ s_one[:, j])
        vs_one[m] = [
            round(float(1 - s_eq.sum() / s_one[:, j].sum()), 4),
            *np.round(np.percentile(g, [5, 95]), 4).tolist(),
        ]
    return {
        "blocks": len(ub),
        "skill": [round(float(1 - s_fit.sum() / s_eq.sum()), 4), *np.round(np.percentile(gain, [5, 95]), 4).tolist()],
        "blend_rpss": [
            round(float(1 - s_eq.sum() / s_cl.sum()), 4),
            *np.round(np.percentile(rpss, [5, 95]), 4).tolist(),
        ],
        "blend_vs_single": vs_one,
    }


def bias_summary(F: dict, T: np.ndarray, lat: np.ndarray) -> dict:
    """Descriptive bias of each model's weekly rain and of their equal-weight blend over the verified runs. F: model ->
    members (run, member, NY, NX) mm; T: the verifying weeks (run, NY, NX) mm; lat: the rows' latitudes. Per entry: the
    30°S-30°N area-weighted forecast/observed ratio, and the area-weighted mean of |forecast/observed - 1| over the cells
    whose verifying weeks average >= WET_MM (local biases of opposite sign cancel in the ratio, not here); null without
    any such cell."""
    ok = np.isfinite(T).all(0)
    for x in F.values():
        ok &= np.isfinite(x).all((0, 1))
    tbar = np.where(ok, np.nanmean(T, 0), np.nan)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(tbar)
    band = ok & (np.abs(lat)[:, None] <= BAND)
    wet = band & (tbar >= WET_MM)
    means = {m: np.nanmean(x, (0, 1)) for m, x in F.items()}
    means["blend"] = np.mean(list(means.values()), 0)

    def ratio(num: float, den: float) -> float | None:
        return round(float(num / den), 3) if den > 0 else None

    out = {}
    for m, f in means.items():
        out[m] = {
            "band_ratio": ratio((w * f)[band].sum(), (w * tbar)[band].sum()),
            "local_abs_bias": ratio((w * np.abs(f / np.where(wet, tbar, 1.0) - 1))[wet].sum(), w[wet].sum()),
        }
    out["wet_cells"] = int(wet.sum())
    return out


# ── data ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def members_at(data: Path, model: str, runs: list[date], wk: str) -> np.ndarray:
    """The week's rain of members 1-6 of the given runs, (run, member, NY, NX) mm on the 1° grid."""
    want = {d: i for i, d in enumerate(runs)}
    out = None
    for f in sorted(data.glob(f"{model}_[0-9]*.npz")):
        z = np.load(f)
        rows = [(i, want[BB._d(s)]) for i, s in enumerate(z["inits"]) if BB._d(s) in want]
        if rows:
            acc = z["acc"][[i for i, _ in rows]].astype("float32")
            if out is None:
                out = np.full((len(runs), acc.shape[2], G.NY, G.NX), np.nan, "float32")
            out[[j for _, j in rows]] = acc[:, 0] if wk == "wk1" else np.maximum(acc[:, 1] - acc[:, 0], 0.0)
    return out


def fit_set(ver: dict) -> tuple[list[str], list[date]]:
    """Models joining the fit, earliest archive first, while the runs they all share fill FIT_BLOCKS 21-day blocks; ->
    (models, shared runs)."""
    have = sorted((v[0][0], m) for m, v in ver.items() if v[0])
    models, shared = [], None
    for _, m in have:
        s = set(ver[m][0]) if shared is None else shared & set(ver[m][0])
        days = sorted(s)
        if days and len({(d - days[0]).days // BB.BLOCK_DAYS for d in days}) >= FIT_BLOCKS:
            models.append(m)
            shared = s
    return models, sorted(shared or [])


def week(wk, data, arch, obs1, obs2, C, rng) -> dict:
    a, b = GB.WEEKS[wk]
    li1, lp1, D1 = obs1
    li2, lp2, D2 = obs2
    lat2, lon2 = G.LAT.reshape(-1, GB.K).mean(1), G.LON.reshape(-1, GB.K).mean(1)
    ver = {m: GB.verified(m, *arch[m], li2, lp2, D2, wk) if arch[m][0] else ([], None, None, None) for m in GB.MODELS}
    fitted, runs = fit_set(ver)
    out = {"models": list(GB.MODELS), "fitted": fitted, "optimised": False}
    if len(fitted) < 2:
        pool = fitted or list(GB.MODELS)
        out["weights"] = {m: round(1 / len(pool), 4) if m in pool else 0.0 for m in GB.MODELS}
        out["note"] = "fewer than two models share a long enough verified archive: equal weights"
        return out
    days = np.array(runs, dtype="datetime64[D]")
    blk = (days - days[0]).astype(int) // BB.BLOCK_DAYS
    # truth and normal on every SUB-th 1° cell; the week's totals in mm
    cells = (slice(None, None, SUB), slice(None, None, SUB))
    mids = [(d + timedelta(days=(a + b) / 2)).month for d in runs]
    T, _ = BB.adjust_late(BB.obs_rate(li1, lp1, runs, a, b), mids, D1)
    if T is None:
        raise SystemExit(f"{wk}: no 1° Late-to-Final ratio for some season")
    T = T[(slice(None), *cells)] * 7.0
    area = np.cos(np.deg2rad(G.LAT[cells[0]]))[:, None] * np.ones((1, len(G.LON[cells[1]])))
    # members of each fitted model, corrected block by block with ratios estimated without that block and its buffer
    buf = np.timedelta64(BUFFER_DAYS, "D")
    X, RAW = {}, {}
    for m in fitted:
        ini, F, Tm, se = ver[m]
        cyc = BB.VERSIONS[m][-1][0]
        raw = members_at(data, m, runs, wk)
        RAW[m] = raw[(slice(None), slice(None), *cells)]
        xm = np.full((len(runs), raw.shape[1], *T.shape[1:]), np.nan, "float32")
        idays = np.array(ini, dtype="datetime64[D]")
        for k in np.unique(blk):
            sel = blk == k
            keep = (idays < days[sel][0] - buf) | (idays > days[sel][-1] + buf)
            ki = [d for d, x in zip(ini, keep) if x]
            centre = runs[int(np.flatnonzero(sel)[sel.sum() // 2])]
            r = GB.estimate(ki, F[keep], Tm[keep], se, wk, centre, lat2, lon2, rng, cyc)["ratio"]
            xm[sel] = (raw[sel] * G.refine(r, lat2, lon2)[None, None])[(slice(None), slice(None), *cells)]
        X[m] = xm
    A, B, Cc, S0 = [], [], [], []
    for i, d in enumerate(runs):
        cl = BB.clim_at(C, d + timedelta(days=a))
        q33, q67 = cl["q33"][cells], cl["q67"][cells]
        t = T[i]
        ok = np.isfinite(t) & np.isfinite(q67) & (q67 >= BB.DRY_Q67)
        for m in fitted:
            ok &= np.isfinite(X[m][i]).all(0)
        O = np.stack([t[ok] < q33[ok], t[ok] <= q67[ok]]).astype("float64")
        PV = [cum_probs(X[m][i][:, ok], q33[ok], q67[ok]) for m in fitted]
        w = area[ok]
        Ai, bi, ci = quad(np.stack([p for p, _ in PV]), np.stack([v for _, v in PV]), O, w)
        A.append(Ai)
        B.append(bi)
        Cc.append(ci)
        S0.append(float(sum(((q - O[k]) ** 2 * w).sum() for k, q in enumerate(CLIM_CUM))))
    A, B, Cc, S0 = map(np.array, (A, B, Cc, S0))
    cand = simplex(len(fitted))
    cv = cross_validate(A, B, Cc, S0, runs, cand, rng, fitted)
    x = fit(A.sum(0), B.sum(0), cand)
    out.update(
        runs=len(runs),
        first=f"{runs[0]}",
        last=f"{runs[-1]}",
        blocks=cv["blocks"],
        cv_skill=cv["skill"],
        blend_rpss=cv["blend_rpss"],
        fitted_weights={m: round(float(v), 4) for m, v in zip(fitted, x)},
        blend_vs_single=cv["blend_vs_single"],
        bias={"raw": bias_summary(RAW, T, G.LAT[cells[0]]), "corrected": bias_summary(X, T, G.LAT[cells[0]])},
    )
    out["optimised"] = cv["skill"][1] > 0
    use = out["fitted_weights"] if out["optimised"] else {m: 1 / len(fitted) for m in fitted}
    out["weights"] = {m: round(use.get(m, 0.0), 4) for m in GB.MODELS}  # a model not yet fitted stays out
    s, lo_, hi_ = (100 * v for v in cv["skill"])
    print(
        f"{wk}: fitted {fitted} on {len(runs)} runs {runs[0]}..{runs[-1]} ({cv['blocks']} blocks); fitted weights "
        f"{out['fitted_weights']}; CV skill over equal weights {s:+.2f} % [{lo_:+.2f}, {hi_:+.2f}] -> "
        f"{'fitted' if out['optimised'] else 'equal'} weights {out['weights']}; equal-weight blend RPSS "
        f"{100 * cv['blend_rpss'][0]:.1f} % [{100 * cv['blend_rpss'][1]:.1f}, {100 * cv['blend_rpss'][2]:.1f}]",
        flush=True,
    )
    for kind in ("raw", "corrected"):
        bs = out["bias"][kind]
        print(
            f"{wk} bias ({kind}): "
            + ", ".join(
                f"{m} ratio {v['band_ratio']} local {v['local_abs_bias']}" for m, v in bs.items() if m != "wet_cells"
            ),
            flush=True,
        )
    print(f"{wk} equal blend over each model alone (fair RPS): {out['blend_vs_single']}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the archive: model month files and IMERG late/final")
    ap.add_argument("--ref", required=True, help="dir with gpcp_clim_week_global.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = Path(a.data)
    rng = np.random.default_rng(20261016)
    C = np.load(Path(a.ref) / "gpcp_clim_week_global.npz")
    obs = {}
    for k in (1, GB.K):  # the scored truth on 1° cells, the bias estimator's on 2° boxes (as global_bias.py)
        li, lp = BB.load_obs(data, "late", k)
        fi, fp = BB.load_obs(data, "final", k)
        obs[k] = (li, lp, BB.truth_terms(li, lp, fi, fp, rng))
    arch = {m: GB.load_global_model(data, m, GB.K) for m in GB.MODELS}
    res = {
        "made": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "score": "fair RPS of weekly terciles vs GPCP 2001-2020, IMERG Late scaled to Final",
        "step": STEP,
        "block_days": BB.BLOCK_DAYS,
        "buffer_days": BUFFER_DAYS,
        "fit_blocks": FIT_BLOCKS,
        "weeks": {wk: week(wk, data, arch, obs[1], obs[GB.K], C, rng) for wk in GB.WEEKS},
    }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "blend_weights_global.json").write_text(json.dumps(res, indent=1, allow_nan=False))  # published
    return 0


if __name__ == "__main__":
    sys.exit(main())
