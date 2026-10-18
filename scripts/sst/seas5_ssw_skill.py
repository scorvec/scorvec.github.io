#!/usr/bin/env python3
"""Conditional skill of SEAS5's sudden-warming (SSW) index: by signal strength and by ENSO amplitude.

SAMPLE. October starts 1981–2025 (45 winters): the AWS SEAS5 daily copy's hindcast (1981–2016, 25 members; the table
hc_ssw_10.npz of seas5_ssw) and its operational forecasts 2017–2025 (51 members; pressure levels exist on the AWS copy
through the October 2025 start). So that every winter's forecast fraction carries the same sampling noise, the
operational years use members 0–24 (25 of 51, member 0 = the control) for every probability test; the full 51 are used
only for the continuous ensemble-mean vortex wind. SSWs are detected in every member and in MERRA-2 with the same code
(seas5_ssw.detect, Charlton & Polvani 2007, Nov–Mar). ENSO: CPC's official DJF ONI (nino_history.json).

PRE-REGISTERED TESTS (fixed 2026-10-08 before anything was computed; Holm-corrected as one family of six, α = 0.05):
  T1  (a) logistic regression of observed SSW (Nov–Mar, yes/no) on the ensemble SSW fraction: slope, likelihood-ratio p
  T2  (a) observed SSW rate, top vs bottom tercile of the forecast fraction: Fisher exact test (two-sided)
  T3  (b) correlation of the ensemble-mean DJF u60N 10 hPa with MERRA-2 (51 members in the operational years)
  T4  (b) the same for January
  T5  (c) logistic model obs ~ fraction × |ONI|: the interaction term (does skill rise with ENSO amplitude?), LR p
  T6  (d) BSS where the forecast signal is strong (|fraction − model base| in its top tercile) and |ONI| ≥ 1.0:
          one-sided bootstrap p of BSS ≤ 0
Descriptive (not in the family, n per group stated): per ENSO group (ONI ≥ 1.5, 0.5–1.5, neutral, ≤ −0.5) the mean
fraction, the observed rate, BSS and ROC area with bootstrap 90 % intervals where n ≥ 8 and both outcomes occur.
POWER: by simulation, the smallest calibrated skill 45 winters detect with 80 % power at α = 0.05.

    python seas5_ssw_skill.py operational        # AWS → data/seas5/ssw/aws_u60_op_10.npz (no CDS)
    python seas5_ssw_skill.py build              # tests, figure, data/seas5_ssw_skill.json
"""

from __future__ import annotations

import pyproj  # noqa: F401  first, before eccodes/xarray load another libproj
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seas5_outlook import ASSETS
import seas5_ref as R
from seas5_ssw import DIR, NDAYS, detect, merra2_u60, oni_djf

OP_YEARS = list(range(2017, 2026))
OUT_JSON = ASSETS / "data" / "seas5_ssw_skill.json"
NSUB = 25
NBOOT = 5000
RNG = np.random.default_rng(20261008)


# ── operational October starts from AWS ─────────────────────────────────────
def op_path() -> Path:
    return DIR / "aws_u60_op_10.npz"


def operational() -> bool:
    import seas5_aws

    if op_path().exists():
        return True
    t0 = time.time()
    root, _ = seas5_aws.open_root("forecast")
    g = root["pressure/1platx1plon"]
    a = g["u"]
    it = pd.Timestamp(g["init_time"].attrs["units"].split("since ")[1]) + pd.to_timedelta(g["init_time"][:], "D")
    il = [float(x) for x in g["pressure_level"][:]].index(10.0)
    lat = g["lat"][:]
    rows = np.where(np.abs(lat - 60.0) < 0.6)[0]
    assert len(rows) == 2, lat[rows]
    nm = a.shape[0]
    out = np.full((len(OP_YEARS), nm, NDAYS), np.nan, np.float32)

    def one(job):
        yi, y, m0, m1 = job
        k = int(np.where((it.year == y) & (it.month == 10) & (it.day == 1))[0][0])
        blk = np.asarray(a[m0:m1, k, :, il, rows[0] : rows[1] + 1, :], dtype=np.float32)
        return yi, m0, m1, np.nanmean(blk, axis=(-2, -1))

    jobs = [(yi, y, m0, min(m0 + 10, nm)) for yi, y in enumerate(OP_YEARS) for m0 in range(0, nm, 10)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for yi, m0, m1, zm in ex.map(one, jobs):
            out[yi, m0:m1] = zm
    fin = np.isfinite(out).mean(axis=(1, 2))
    print(
        f"  AWS operational u60N10: {out.shape} in {(time.time() - t0) / 60:.1f} min; finite per year {np.round(fin, 3).tolist()}",
        flush=True,
    )
    DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(op_path(), u60=out, years=np.array(OP_YEARS))
    return bool(fin.min() > 0.99)


# ── statistics ───────────────────────────────────────────────────────────────
def logit_fit(X: np.ndarray, y: np.ndarray, iters: int = 100):
    """IRLS logistic regression → (beta, se, loglik). X includes the intercept column."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * w[:, None])
        g = X.T @ (y - p)
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
    H = X.T @ (X * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.pinv(H)))
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return b, se, ll


def lrt(X1, X0, y):
    from scipy.stats import chi2

    b1, se1, l1 = logit_fit(X1, y)
    _, _, l0 = logit_fit(X0, y)
    d = 2 * (l1 - l0)
    return b1, se1, float(chi2.sf(max(d, 0.0), X1.shape[1] - X0.shape[1]))


def bss_auc(p, o):
    clim = (o.sum() - o) / (len(o) - 1)
    bs, bsc = np.mean((p - o) ** 2), np.mean((clim - o) ** 2)
    bss = 1 - bs / bsc if bsc > 0 else np.nan
    pos, neg = p[o == 1], p[o == 0]
    auc = (
        np.nan
        if not len(pos) or not len(neg)
        else (np.sum(pos[:, None] > neg[None]) + 0.5 * np.sum(pos[:, None] == neg[None])) / (len(pos) * len(neg))
    )
    return float(bss), float(auc)


def boot(p, o, n=NBOOT):
    vals = np.array([bss_auc(p[ii], o[ii]) for ii in (RNG.integers(0, len(o), len(o)) for _ in range(n))])
    return vals


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    out = {}
    run = 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def power(n: int, base: float, sims: int = 2000) -> list:
    """Calibrated forecasts p_i = base + s·z (clipped), outcomes ~ Bernoulli(p_i): power of the T1 slope test at α = 0.05,
    with the population BSS and ROC area of such a forecast."""
    out = []
    for s in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        hits = 0
        for _ in range(sims):
            p = np.clip(base + s * RNG.standard_normal(n), 0.02, 0.98)
            o = (RNG.random(n) < p).astype(float)
            if o.min() == o.max():
                continue
            X1 = np.column_stack([np.ones(n), p])
            X0 = np.ones((n, 1))
            _, _, pv = lrt(X1, X0, o)
            hits += pv < 0.05
        # population skill of that forecast (large sample)
        P = np.clip(base + s * RNG.standard_normal(200000), 0.02, 0.98)
        O = (RNG.random(200000) < P).astype(float)
        bss = 1 - np.mean((P - O) ** 2) / np.mean((O.mean() - O) ** 2)
        from scipy.stats import mannwhitneyu

        u = mannwhitneyu(P[O == 1][:20000], P[O == 0][:20000]).statistic / (20000 * 20000)
        out.append(dict(spread=s, power=round(hits / sims, 3), bss=round(float(bss), 3), auc=round(float(u), 3)))
    return out


# ── build ────────────────────────────────────────────────────────────────────
def build() -> dict:
    t0 = time.time()
    H = R.need(R.hc_ref("ssw", "10"), "hindcast daily u10 (seas5_ssw)")
    hsec, lats, hyears = H["sec"], H["lats"], [int(y) for y in H["years"]]
    i60 = int(np.argmin(np.abs(lats - 60)))
    op = np.load(op_path())
    win = (11, 12, 1, 2, 3)
    rows = []
    m2 = merra2_u60()
    oni = oni_djf()
    cat = json.loads((HERE.parents[1] / "scripts" / "strat" / "reference" / "strat_history_events.json").read_text())[
        "ssw"
    ]
    cat_w = {int(e["date"][:4]) - (1 if int(e["date"][5:7]) <= 6 else 0) for e in cat}

    def members(u, y):
        dates = pd.date_range(f"{y}-10-01", periods=NDAYS, freq="D")
        flags = np.array([any(d.month in win for d in detect(u[m], dates)["ssw"]) for m in range(u.shape[0])])
        djf = (dates.month == 12) | (dates.month == 1) | (dates.month == 2)
        jan = dates.month == 1
        return flags, np.nanmean(u[:, djf]), np.nanmean(u[:, jan])

    for yi, y in enumerate(hyears):
        f, d, j = members(hsec[yi, :, :, i60], y)
        rows.append(dict(year=y, src="hindcast", n=len(f), frac=f.mean(), djf=d, jan=j))
    for yi, y in enumerate(op["years"]):
        u = op["u60"][yi]
        f25, _, _ = members(u[:NSUB], int(y))
        _, d, j = members(u, int(y))
        rows.append(dict(year=int(y), src="operational", n=NSUB, frac=f25.mean(), djf=d, jan=j))
    for r in rows:
        y = r["year"]
        s = m2[f"{y}-10-01" : f"{y + 1}-05-03"]
        ev = detect(s.values, s.index)["ssw"]
        r["obs"] = float(any(d.month in win for d in ev))
        r["obs_cat"] = float(y in cat_w)
        r["obs_djf"] = float(s[f"{y}-12-01" : f"{y + 1}-02"].mean())
        r["obs_jan"] = float(
            s[f"{y + 1}-01-01" : f"{y + 1}-01-31"].mean()
        )  # DJF through the last day of February (the 29th in a leap year)
        r["oni"] = oni.get(y, np.nan)
    df = pd.DataFrame(rows)
    agree = int((df.obs == df.obs_cat).sum())
    print(
        f"  {len(df)} winters; observed SSW winters {int(df.obs.sum())}; detection vs catalogue agree in {agree}/{len(df)}",
        flush=True,
    )
    p, o, a = df.frac.values, df.obs.values, np.abs(df.oni.values)
    base_model = float(p.mean())
    res = {
        "n": len(df),
        "years": f"{df.year.min()}/{(df.year.min() + 1) % 100:02d}–{df.year.max()}/{(df.year.max() + 1) % 100:02d}",
        "obs_rate": round(float(o.mean()), 3),
        "model_base": round(base_model, 3),
        "detection_vs_catalogue": f"{agree}/{len(df)}",
        "tests": {},
    }
    # T1
    b, se, pv = lrt(np.column_stack([np.ones(len(o)), p]), np.ones((len(o), 1)), o)
    res["tests"]["T1"] = dict(
        name="Observed SSW ~ ensemble SSW fraction (logistic)",
        n=len(o),
        stat="slope",
        value=round(float(b[1]), 2),
        ci95=[round(float(b[1] - 1.96 * se[1]), 2), round(float(b[1] + 1.96 * se[1]), 2)],
        p=pv,
    )
    # T2
    from scipy.stats import fisher_exact, pearsonr

    lo_t, hi_t = np.quantile(p, [1 / 3, 2 / 3])
    top, bot = p >= hi_t, p <= lo_t
    tab = [[int(o[top].sum()), int((1 - o[top]).sum())], [int(o[bot].sum()), int((1 - o[bot]).sum())]]
    res["tests"]["T2"] = dict(
        name="Observed SSW rate, top vs bottom tercile of the forecast fraction (Fisher exact)",
        n=int(top.sum() + bot.sum()),
        stat="rates",
        value=[round(float(o[top].mean()), 2), round(float(o[bot].mean()), 2)],
        n_groups=[int(top.sum()), int(bot.sum())],
        p=float(fisher_exact(tab)[1]),
    )
    # T3, T4
    for key, col, ocol, lab in (("T3", "djf", "obs_djf", "DJF"), ("T4", "jan", "obs_jan", "January")):
        x, yv = df[col].values, df[ocol].values
        r, pr = pearsonr(x, yv)
        bs = [pearsonr(x[ii], yv[ii])[0] for ii in (RNG.integers(0, len(x), len(x)) for _ in range(NBOOT))]
        res["tests"][key] = dict(
            name=f"Ensemble-mean {lab} u60N 10 hPa vs MERRA-2 (correlation)",
            n=len(x),
            stat="r",
            value=round(float(r), 2),
            ci90=[round(float(np.percentile(bs, 5)), 2), round(float(np.percentile(bs, 95)), 2)],
            p=float(pr),
        )
    # T5
    ok = np.isfinite(a)
    X1 = np.column_stack([np.ones(ok.sum()), p[ok], a[ok], p[ok] * a[ok]])
    X0 = X1[:, :3]
    b5, se5, p5 = lrt(X1, X0, o[ok])
    res["tests"]["T5"] = dict(
        name="Interaction fraction × |ONI| in obs ~ fraction × |ONI| (logistic)",
        n=int(ok.sum()),
        stat="interaction",
        value=round(float(b5[3]), 2),
        ci95=[round(float(b5[3] - 1.96 * se5[3]), 2), round(float(b5[3] + 1.96 * se5[3]), 2)],
        p=p5,
    )
    # T6
    sig = np.abs(p - base_model)
    sel = (sig >= np.quantile(sig, 2 / 3)) & (a >= 1.0)
    if sel.sum() >= 4 and 0 < o[sel].sum() < sel.sum():
        clim_all = o.mean()

        def bss_sub(ii):
            pp, oo = p[sel][ii], o[sel][ii]
            return 1 - np.mean((pp - oo) ** 2) / np.mean((clim_all - oo) ** 2)

        v = bss_sub(np.arange(sel.sum()))
        bsv = np.array([bss_sub(RNG.integers(0, sel.sum(), sel.sum())) for _ in range(NBOOT)])
        res["tests"]["T6"] = dict(
            name="BSS when the signal is strong (top tercile of |fraction − model base|) and |ONI| ≥ 1.0",
            n=int(sel.sum()),
            stat="BSS",
            value=round(float(v), 2),
            ci90=[round(float(np.percentile(bsv, 5)), 2), round(float(np.percentile(bsv, 95)), 2)],
            p=float((np.sum(bsv <= 0) + 1) / (NBOOT + 1)),
            years=[int(y) for y in df.year[sel]],
        )
    else:
        res["tests"]["T6"] = dict(
            name="BSS when the signal is strong and |ONI| ≥ 1.0",
            n=int(sel.sum()),
            stat="BSS",
            value=None,
            ci90=None,
            p=1.0,
            note="too few winters or one outcome only",
            years=[int(y) for y in df.year[sel]],
        )
    adj = holm({k: v["p"] for k, v in res["tests"].items()})
    for k, v in res["tests"].items():
        v["p_holm"] = round(adj[k], 4)
        v["p"] = round(v["p"], 4)
        v["significant"] = bool(adj[k] < 0.05)
    # descriptive ENSO groups
    grp = {
        "strong El Niño (ONI ≥ 1.5)": df.oni >= 1.5,
        "moderate El Niño (0.5–1.5)": (df.oni >= 0.5) & (df.oni < 1.5),
        "neutral (|ONI| < 0.5)": df.oni.abs() < 0.5,
        "La Niña (ONI ≤ −0.5)": df.oni <= -0.5,
    }
    res["groups"] = {}
    for gname, mask in grp.items():
        m = mask.values
        e = dict(
            n=int(m.sum()),
            years=[int(y) for y in df.year[m]],
            frac=round(float(p[m].mean()), 3) if m.any() else None,
            obs_rate=round(float(o[m].mean()), 3) if m.any() else None,
        )
        if m.sum() >= 8 and 0 < o[m].sum() < m.sum():
            bs_, au_ = bss_auc(p[m], o[m])
            bv = boot(p[m], o[m])
            e.update(
                bss=round(bs_, 2),
                bss_ci90=[
                    round(float(np.nanpercentile(bv[:, 0], 5)), 2),
                    round(float(np.nanpercentile(bv[:, 0], 95)), 2),
                ],
                auc=round(au_, 2),
                auc_ci90=[
                    round(float(np.nanpercentile(bv[:, 1], 5)), 2),
                    round(float(np.nanpercentile(bv[:, 1], 95)), 2),
                ],
            )
        else:
            e["note"] = "n < 8 or one outcome only: no skill estimate"
        res["groups"][gname] = e
    res["power"] = power(len(df), float(o.mean()))
    mde = next((x for x in res["power"] if x["power"] >= 0.8), None)
    res["mde"] = mde
    res["winters"] = [
        dict(
            year=int(r.year),
            src=r.src,
            frac=round(float(r.frac), 3),
            obs=int(r.obs),
            oni=(None if not np.isfinite(r.oni) else round(float(r.oni), 2)),
            djf=round(float(r.djf), 1),
            obs_djf=round(float(r.obs_djf), 1),
        )
        for r in df.itertuples()
    ]
    cur = (
        json.loads((ASSETS / "data" / "seas5_ssw.json").read_text())
        if (ASSETS / "data" / "seas5_ssw.json").exists()
        else None
    )
    res["current"] = dict(issue=cur["issue"], frac=cur["windows"]["Nov–Mar"]["forecast"]) if cur else None
    res["figure"] = render(df, res)
    res["generated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    res["note"] = (
        "Operational years 2017–2025 use members 0–24 of 51 for the SSW fraction so every winter has 25 members; the "
        "continuous vortex test (T3, T4) uses all 51. Tests T1–T6 were fixed before computing and are Holm-corrected as one family."
    )
    for t in res["tests"].values():
        print(
            f"  {t['name']}: n {t['n']}, {t['stat']} {t['value']}, CI {t.get('ci95') or t.get('ci90')}, p {t['p']}, Holm {t['p_holm']}",
            flush=True,
        )
    for g, e in res["groups"].items():
        print(
            f"  {g}: n {e['n']} {e['years']}, frac {e['frac']}, obs {e['obs_rate']}, BSS {e.get('bss')} {e.get('bss_ci90')}, AUC {e.get('auc')} {e.get('auc_ci90')}",
            flush=True,
        )
    print(f"  power: {res['power']}", flush=True)

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x

    OUT_JSON.write_text(json.dumps(clean(res), separators=(",", ":")))
    print(f"wrote {OUT_JSON} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return res


def render(df, res) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    jit = np.random.default_rng(3).uniform(-0.06, 0.06, len(df))
    sc = ax.scatter(
        df.frac,
        df.obs + jit,
        c=df.oni,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(0, -2.5, 2.5),
        s=[60 if s == "operational" else 42 for s in df.src],
        marker="o",
        edgecolors=["#222" if s == "operational" else "none" for s in df.src],
        linewidths=0.8,
        zorder=3,
    )
    # winter labels placed clear of each other (strong-ENSO winters at the same share would print on top of one another
    # at the no-SSW row)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.12, 1.12)
    ren = fig.canvas.get_renderer()
    placed = []
    spots = [
        (5, 4, "left"),
        (5, -10, "left"),
        (-5, 4, "right"),
        (-5, -10, "right"),
        (5, 14, "left"),
        (5, -20, "left"),
        (-5, 14, "right"),
        (-5, -20, "right"),
    ]
    for r, j in sorted(zip(df.itertuples(), jit), key=lambda t: t[0].frac):
        if abs(r.oni) >= 1.5:
            for dx, dy, ha in spots:
                t_ = ax.annotate(
                    f"{r.year}/{(r.year + 1) % 100:02d}",
                    (r.frac, r.obs + j),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    ha=ha,
                    fontsize=7.5,
                    color="#444",
                )
                bb = t_.get_window_extent(ren)
                fr = ax.get_window_extent(ren)
                if bb.y0 >= fr.y0 and bb.y1 <= fr.y1 and bb.x1 <= fr.x1 and not any(bb.overlaps(o) for o in placed):
                    placed.append(bb)
                    break
                t_.remove()
            else:
                placed.append(
                    ax.annotate(
                        f"{r.year}/{(r.year + 1) % 100:02d}",
                        (r.frac, r.obs + j),
                        textcoords="offset points",
                        xytext=(5, 4),
                        fontsize=7.5,
                        color="#444",
                    ).get_window_extent(ren)
                )
    t1 = res["tests"]["T1"]
    xs = np.linspace(0, 1, 100)
    X = np.column_stack([np.ones(len(df)), df.frac.values])
    b, _, _ = logit_fit(X, df.obs.values)
    ax.plot(
        xs,
        1 / (1 + np.exp(-(b[0] + b[1] * xs))),
        color="#8a4b2a",
        lw=1.6,
        label=f"logistic fit (slope p {t1['p']:.2f}, Holm {t1['p_holm']:.2f})",
    )
    ax.axhline(df.obs.mean(), color="#8a8680", lw=0.8, ls=":", label=f"observed rate {df.obs.mean():.0%}")
    ax.axvline(res["model_base"], color="#8a8680", lw=0.8, ls="--", label=f"model base rate {res['model_base']:.0%}")
    if res.get("current"):
        ax.scatter(
            [res["current"]["frac"]],
            [0.5],
            marker="*",
            s=260,
            color="#c8451c",
            edgecolors="#222",
            zorder=5,
            label=f"this winter: {res['current']['frac']:.0%} of members (outcome unknown)",
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["no SSW", "", "SSW"])
    ax.set_xlabel("share of 25 SEAS5 members with a major SSW, Nov–Mar (October start)")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("DJF ONI (CPC)")
    ax.legend(fontsize=7.8, frameon=True, framealpha=0.9, edgecolor="none", loc="center left")  # clear of the rate line
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"SEAS5 SSW fraction against what happened, {res['years']} ({res['n']} winters)", loc="left", fontsize=12
    )
    import textwrap  # wrapped: one line widened the saved picture past the chart

    fig.text(
        0.01,
        -0.03,
        textwrap.fill(
            "Each dot is a winter: the October-start ensemble's SSW share against the MERRA-2 outcome (jittered), "
            "coloured by DJF ONI; outlined dots are the operational years 2017–2025 (members 0–24). Labelled: ONI ≥ 1.5.",
            175,
        ),
        fontsize=7.6,
        color="#6f6b64",
        va="top",
        linespacing=1.35,
    )
    out = ASSETS / "seas5_ssw_skill.webp"
    fig.savefig(out, dpi=105, bbox_inches="tight", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)
    return out.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["operational", "build"])
    a = ap.parse_args(argv)
    if a.cmd == "operational":
        return 0 if operational() else 1
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
