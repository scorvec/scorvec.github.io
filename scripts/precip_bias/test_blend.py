#!/usr/bin/env python3
"""Offline checks of the tropical rainfall forecast's blend fit, its cross-validated test and the day-on-day change test,
on synthetic ensembles with known answers (no data are fetched and no figure is drawn).

    python scripts/precip_bias/test_blend.py

  score    blend.quad's quadratic form equals the fair RPS summed cell by cell, and the fair score of six-member
           ensembles averages to the score of the full distributions (the reason the archive's six members suffice)
  null     three models with errors of the same size: the cross-validated test may call fitted weights better than equal
           ones in about 5 % of trials at most (it is a one-sided 5 % test)
  planted  one model with a far smaller error: the fit gives it most of the weight and the test finds the gain
  change   tropical_rain.welch on skewed (gamma) members of four ensembles of 50/50/31/21: with no change about 5 % of
           cells fall below p = 0.05 and at most about q of the maps get any shaded cell after BH-FDR; a planted change is
           found, with a false-discovery share near or under q
  grid     GEFS-like (90°N, 0°E first) and GEPS-like (90°S, 0°E first) 0.5° GRIB2 messages written with eccodes decode
           (ens_tp.decode) to the field they were written from, and to the same 1° boxes
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blend as BL
import tropical_rain as TR  # (imports pyproj before eccodes)
import ens_tp as ET
import grid as G
import eccodes

Q33 = -0.430727  # standard normal terciles
RNG = np.random.default_rng(20261016)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}", flush=True)


def ar1(rng, n, cells, rho):
    """(n, cells) unit-variance AR(1) series along the first axis: errors and signals that persist from run to run."""
    x = np.empty((n, cells))
    x[0] = rng.standard_normal(cells)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(cells)
    return x


def season(rng, err_sd, runs=147, cells=300, members=6, rho=0.9):
    """Per-run fair-RPS statistics of models whose forecasts are the predictable signal plus their own persistent error
    (sd err_sd each) plus member noise, scored against truth = signal + noise. -> A, b, c, s_clim, days."""
    s = 0.8 * ar1(rng, runs, cells, rho)
    y = s + rng.standard_normal((runs, cells))
    sd = np.sqrt(0.64 + 1.0)
    q33, q67 = np.full(cells, Q33 * sd), np.full(cells, -Q33 * sd)
    X = [
        s[:, None] + e * ar1(rng, runs, cells, rho)[:, None] + rng.standard_normal((runs, members, cells))
        for e in err_sd
    ]
    w = np.ones(cells)
    A, b, c, s0 = [], [], [], []
    for i in range(runs):
        O = np.stack([y[i] < q33, y[i] <= q67]).astype(float)
        PV = [BL.cum_probs(x[i], q33, q67) for x in X]
        Ai, bi, ci = BL.quad(np.stack([p for p, _ in PV]), np.stack([v for _, v in PV]), O, w)
        A.append(Ai)
        b.append(bi)
        c.append(ci)
        s0.append(sum(((q - O[k]) ** 2 * w).sum() for k, q in enumerate(BL.CLIM_CUM)))
    days = [date(2026, 5, 13) + timedelta(days=i) for i in range(runs)]
    return np.array(A), np.array(b), np.array(c), np.array(s0), days


print("score")
cells, M = 500, 3
X = [RNG.normal(m * 0.3, 1.0, (6, cells)) for m in range(M)]
q33, q67 = np.full(cells, -0.43), np.full(cells, 0.43)
y = RNG.normal(0, 1, cells)
O = np.stack([y < q33, y <= q67]).astype(float)
w = RNG.uniform(0.2, 1.0, cells)
PV = [BL.cum_probs(x, q33, q67) for x in X]
P, V = np.stack([p for p, _ in PV]), np.stack([v for _, v in PV])
A, b, c = BL.quad(P, V, O, w)
for x in RNG.dirichlet(np.ones(M), 5):
    direct = sum(
        (w * ((np.einsum("m,mc->c", x, P[:, k]) - O[k]) ** 2 - np.einsum("m,mc->c", x**2, V[:, k]))).sum()
        for k in range(2)
    )
    check(abs(BL.score(A, b, c, x) - direct) < 1e-8 * abs(direct), f"quadratic form = direct fair RPS ({direct:.3f})")
# fair: six-member chances, averaged over many ensembles, score like the full distributions
mu = np.array([0.0, 0.5, -0.4])
x = np.array([0.5, 0.3, 0.2])
Pt = np.stack([np.stack([norm.cdf(-0.43 - m) * np.ones(cells), norm.cdf(0.43 - m) * np.ones(cells)]) for m in mu])
exact = sum(((np.einsum("m,mc->c", x, Pt[:, k]) - O[k]) ** 2).sum() for k in range(2))
fair, raw = [], []
for _ in range(400):
    PV = [BL.cum_probs(RNG.normal(m, 1.0, (6, cells)), q33, q67) for m in mu]
    A1, b1, c1 = BL.quad(np.stack([p for p, _ in PV]), np.stack([v for _, v in PV]), O, np.ones(cells))
    fair.append(BL.score(A1, b1, c1, x))
    raw.append(BL.score(A1 + np.diag(np.einsum("mkc->m", np.stack([v for _, v in PV]))), b1, c1, x))
check(
    abs(np.mean(fair) - exact) < 0.01 * exact,
    f"fair 6-member score {np.mean(fair):.1f} ~ full-distribution {exact:.1f}",
)
check(
    np.mean(raw) > exact * 1.02,
    f"the unadjusted 6-member score is worse ({np.mean(raw):.1f}): what the fair form removes",
)

print("simplex and fit")
cand = BL.simplex(3)
check(np.allclose(cand.sum(1), 1) and (cand >= 0).all() and len(cand) == 861, "861 weight vectors in steps of 0.025")
check(len(BL.simplex(4)) == 12341, "12341 for four models")

print("null: three models of equal skill")
trials, hits, gains = 200, 0, []
for _ in range(trials):
    A, b, c, s0, days = season(RNG, (0.6, 0.6, 0.6))
    cv = BL.cross_validate(A, b, c, s0, days, cand, RNG)
    hits += cv["skill"][1] > 0
    gains.append(cv["skill"][0])
check(hits / trials <= 0.08, f"fitted weights called better in {100 * hits / trials:.1f} % of {trials} null trials")
check(
    np.mean(gains) < 0, f"mean cross-validated gain under the null {100 * np.mean(gains):+.2f} % (fitting costs skill)"
)

print("planted: one model far better")
wins, top = 0, []
for _ in range(20):
    A, b, c, s0, days = season(RNG, (0.1, 1.0, 1.0))
    cv = BL.cross_validate(A, b, c, s0, days, cand, RNG)
    wins += cv["skill"][1] > 0
    top.append(BL.fit(A.sum(0), b.sum(0), cand)[0])
check(wins >= 18, f"the gain is found in {wins} of 20 trials")
check(min(top) > 0.6, f"the better model's fitted weight {min(top):.2f}-{max(top):.2f}")
check(cv["blend_rpss"][1] > 0, f"the equal-weight blend beats climatology (RPSS {100 * cv['blend_rpss'][0]:.1f} %)")

print("change: Welch's test on four ensembles")
sizes = {"ifs": 50, "aifs-ens": 50, "gefs": 31, "geps": 21}
wts = dict.fromkeys(sizes, 0.25)


def runs(rng, cells, shift=None, k=0.7):
    scale = rng.uniform(1.0, 40.0, cells)  # mm per week, cell by cell
    today = {m: rng.gamma(k, scale, (n, cells)) for m, n in sizes.items()}
    prev = {m: rng.gamma(k, scale, (n, cells)) for m, n in sizes.items()}
    if shift is not None:
        for m in today:
            today[m][:, shift] += 0.5 * np.sqrt(k) * scale[shift]  # +0.5 standard deviations of a member, every model
    return today, prev


ps, maps = [], 0
for _ in range(300):
    today, prev = runs(RNG, 400)
    d, p = TR.welch(today, prev, wts)
    ps.append(p)
    maps += TR.change_test(today, prev, wts)[1].any()
rate = float(np.mean(np.concatenate(ps) < 0.05))
check(0.04 <= rate <= 0.065, f"no change: {100 * rate:.2f} % of cells below p = 0.05")
check(maps / 300 <= 0.14, f"no change: {100 * maps / 300:.1f} % of maps with any shaded cell after BH-FDR q 0.10")
fdp, power = [], []
for _ in range(50):
    true = np.zeros(2000, bool)
    true[RNG.choice(2000, 200, replace=False)] = True
    today, prev = runs(RNG, 2000, shift=true)
    _, sig, _ = TR.change_test(today, prev, wts)
    fdp.append((sig & ~true).sum() / max(sig.sum(), 1))
    power.append((sig & true).sum() / true.sum())
check(np.mean(power) > 0.5, f"planted change found in {100 * np.mean(power):.0f} % of its cells")
check(np.mean(fdp) <= 0.10, f"false-discovery share {100 * np.mean(fdp):.1f} % (q 0.10)")
d, p = TR.welch({"x": np.zeros((5, 3))}, {"x": np.zeros((5, 3))}, {"x": 1.0})
check(np.isnan(p).all(), "cells where no member varies are not tested")

print("grid: 0.5° GRIB2 messages")
lat = np.linspace(90, -90, 361)
lon = np.arange(0, 360, 0.5)
field = (5 + 3 * np.cos(np.deg2rad(lat))[:, None] * (1 + np.sin(np.deg2rad(2 * lon)))[None, :]).astype("float64")
for name, centre, south_first in (("GEFS", "kwbc", False), ("GEPS", "cwao", True)):
    g = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    for k, v in (
        ("centre", centre),  # NCEP's tables name APCP "tp" in kg m**-2; under ECCC's eccodes leaves the units unknown
        ("productDefinitionTemplateNumber", 11),
        ("discipline", 0),
        ("parameterCategory", 1),
        ("parameterNumber", 8),
        ("typeOfStatisticalProcessing", 1),
        ("lengthOfTimeRange", 168),
        ("Ni", 720),
        ("Nj", 361),
        ("iDirectionIncrementInDegrees", 0.5),
        ("jDirectionIncrementInDegrees", 0.5),
        ("longitudeOfFirstGridPointInDegrees", 0.0),
        ("longitudeOfLastGridPointInDegrees", 359.5),
        ("latitudeOfFirstGridPointInDegrees", -90.0 if south_first else 90.0),
        ("latitudeOfLastGridPointInDegrees", 90.0 if south_first else -90.0),
        ("jScansPositively", int(south_first)),
        ("perturbationNumber", 7),
        ("bitsPerValue", 24),
    ):
        eccodes.codes_set(g, k, v)
    eccodes.codes_set_values(g, (field[::-1] if south_first else field).ravel())
    msg = eccodes.codes_get_message(g)
    eccodes.codes_release(g)
    meta, f, res = ET.decode(ET._messages(msg + msg)[1])
    want = np.roll(field, 360, axis=1)  # columns from 180°W
    check(res == 0.5 and meta["number"] == 7 and meta["end"] == 168, f"{name}: 0.5° grid, member 7, ends at 168 h")
    check(np.abs(f - want).max() < 1e-3, f"{name}: decoded in mm to rows 90°N -> 90°S, columns from 180°W")
    box = G.to_1deg(f, 0.5)
    check(np.abs(box - G.to_1deg(want, 0.5)).max() < 1e-3, f"{name}: the same 1° boxes")
print("all checks passed")
