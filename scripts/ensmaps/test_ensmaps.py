#!/usr/bin/env python3
"""Offline checks of the global ensemble maps (ensmaps.py, ensmaps_ref.py) on synthetic data with known answers: no
data are fetched.

    python scripts/ensmaps/test_ensmaps.py

  grids    to_half keeps a constant, gives the exact cell means of a field linear in longitude (0.1 and 0.25 deg) and
           wraps at the date line; tiles overlap their neighbours by one point and the last column wraps to 180W
  grib     decode rolls a south-first message starting at 0E to 90N first from 180W; to_units reads Pa, dam and
           geopotential from their magnitudes; read_field does the same for a reanalysis message with missing values
  normals  bilinear interpolation of a 1.5-deg curve is exact for a field linear in latitude and wraps in longitude;
           the 2 m file's float16 harmonics give back the curve to 0.02 K
  codes    the anomaly, mean and change codes round-trip within half a step and map NaN to 0
  gate     agreeing counts the members sharing the mean's sign; sign_k is 33/50, 33/51, 22/31, 16/21
  test     welch_p equals scipy's Welch test; var_p equals the F-test from scipy's F distribution, finds a planted
           doubling of the spread across a map and nothing between two runs of the same spread; against band_ratio's
           typical change at the latitude it finds a planted rise and not a uniform shrinkage; the spread codes
           round-trip
  chain    a run's statistics serve the runs 6 to 48 h later (IFS-ENS 06/18Z to +144 h among them), the last window
           cut to the shared days; the manifest keeps the four newest runs, the store those that left in the last hour
           and the newest before them; a rebuild publishes add-only
  regimes  members whose 500 hPa anomaly is a regime's centroid are labelled with it (every day in its own season's
           set) and the record is regime_core's format; a period's label day is the middle day, whose 5-day running
           mean is the period's own mean; the run-to-run test finds a planted shift and nothing in identical runs; a
           12Z run's day 16 needs +384 h
  clusters a scenario keeps its letter through a one-scenario day, a split takes a new letter and a scenario that
           re-forms its old one; the members' moves between two periods' clusters
  super    a first stage's statistics of an older span layout are carried by span; the weighted sign test's gate is
           sign_k / n for one ensemble and holds its 5 % for four (simulated); the super-ensemble of four synthetic
           runs gives the equal-weight mean anomaly, the agreeing share's gate, the variance of the weighted mean, the
           change from the run 12 h earlier (a planted GEFS shift) and the manifest; its plan waits for all four runs
           and makes a cycle again when one of them is remade
  telecon  Capture keeps a member's field at each day's step on the patterns' grid (a constant stays constant, other
           steps are skipped, a missing lead is marked); an index of a reference with known projectors is the planted
           amplitude, read against the normal of the field's own hour, and the ensemble mean's index is the members'
           mean; the hindcast fit finds a planted correlation, counts consecutive daily starts as partly one case, and
           its predictive probabilities are reliable on fresh data; the gate passes the skilful leads only
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ensmaps as E  # (imports pyproj before eccodes)
import ensmaps_clusters as EC
import ensmaps_ref as R
import ensmaps_regimes as ER
import ensmaps_super as ES
import ensmaps_telecon as ET
import regime_core as RC
import regimes_ecmwf as RE
import telecon15 as T15
import eccodes

RNG = np.random.default_rng(20261025)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}", flush=True)


def grib(field: np.ndarray, lat0: float, lon0: float, inc: float, up: bool) -> bytes:
    g = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    nj, ni = field.shape
    for k, v in (
        ("Ni", ni),
        ("Nj", nj),
        ("latitudeOfFirstGridPointInDegrees", lat0),
        ("longitudeOfFirstGridPointInDegrees", lon0),
        ("latitudeOfLastGridPointInDegrees", -lat0),
        ("longitudeOfLastGridPointInDegrees", lon0 + inc * (ni - 1)),
        ("iDirectionIncrementInDegrees", inc),
        ("jDirectionIncrementInDegrees", inc),
        ("jScansPositively", int(up)),
        ("bitsPerValue", 24),
    ):
        eccodes.codes_set(g, k, v)
    eccodes.codes_set_values(g, field.ravel())
    b = eccodes.codes_get_message(g)
    eccodes.codes_release(g)
    return b


def grids():
    print("grids")
    for g in (E.G10, E.G25):
        a = np.full(g.shape, 7.5, np.float32)
        check(np.allclose(E.to_half(a, g.res), 7.5), f"to_half keeps a constant ({g.res} deg)")
        lon = np.broadcast_to(g.lon[None], g.shape).astype(np.float64)
        h = E.to_half(np.sin(np.deg2rad(lon)), g.res)
        if g.res == 0.1:  # five points of 0.1 deg centred on the cell: their mean of sin
            exact = np.mean([np.sin(np.deg2rad(E.G50.lon + d)) for d in (-0.2, -0.1, 0.0, 0.1, 0.2)], axis=0)
        else:  # 1-2-1 over 0.25-deg points
            exact = (
                np.sin(np.deg2rad(E.G50.lon - 0.25))
                + 2 * np.sin(np.deg2rad(E.G50.lon))
                + np.sin(np.deg2rad(E.G50.lon + 0.25))
            ) / 4
        check(np.abs(h[180] - exact).max() < 1e-6, f"to_half: exact cell means across the date line ({g.res} deg)")
    img = np.arange(1801 * 3600, dtype=np.int64).reshape(1801, 3600) % 251
    img = np.dstack([img] * 3).astype(np.uint8)
    t = E.tiles(img)
    check(len(t) == 37 and t["o"].shape == (361, 720, 3), "36 tiles and the every-5th-point overview")
    check(t["00"].shape == (301, 601, 3) and (t["00"][:, -1] == t["01"][:, 0]).all(), "tiles share an edge column")
    check(
        (t["05"][:, -1] == img[0:301, 0]).all() and (t["10"][0] == t["00"][-1]).all(),
        "the last column wraps; rows overlap",
    )


def grib_checks():
    print("grib")
    lat = np.arange(-90, 90.01, 0.5)
    lon = np.arange(0, 360, 0.5)
    f = 1000 * lat[:, None] + ((lon + 180) % 360 - 180)[None]
    _, v, inc = E.decode(grib(f, -90.0, 0.0, 0.5, up=True), "")
    check(inc == 0.5 and v.shape == (361, 720), "a 0.5-deg global message decodes")
    check(
        abs(v[0, 0] - (90000 - 180)) < 0.5 and abs(v[-1, -1] - (-90000 + 179.5)) < 0.5,
        "rolled to 90N first, 180W first",
    )
    check(E.to_units("msl", np.full((10, 10), 101000.0))[1] == "Pa", "msl in Pa -> hPa")
    check(E.to_units("z500", np.full((10, 10), 555.0))[0][0, 0] == 5550.0, "500 hPa height in dam -> m")
    check(abs(E.to_units("z500", np.full((10, 10), 54000.0))[0][0, 0] - 54000 / 9.80665) < 1e-3, "geopotential -> m")
    g = eccodes.codes_new_from_message(grib(np.where(np.abs(f) < 1e9, f, 0), 90.0, 0.0, 0.5, up=False))
    try:
        out = R.read_field(g, np.round(np.arange(90, -90.01, -0.5), 2), np.round(np.arange(-180, 180, 0.5), 2))
    finally:
        eccodes.codes_release(g)
    check(abs(out[0, 0] - (-90000 - 180)) < 1, "read_field rolls a reanalysis message to 180W first")


def normals():
    print("normals")
    coef = np.zeros((7, R.LATU.size, R.LONU.size), np.float32)
    coef[0] = R.LATU[:, None] * 2.0 + 0 * R.LONU[None]
    coef[1] = np.cos(np.deg2rad(R.LONU))[None] + 0 * R.LATU[:, None]
    b = E.bilinear(coef, R.LATU, R.LONU, E.G50)
    check(np.abs(b[0] - 2 * E.G50.lat[:, None]).max() < 1e-4, "bilinear is exact for a field linear in latitude")
    i = np.abs(E.G50.lon - 179.5).argmin()
    exact = np.cos(np.deg2rad(178.5)) / 3 + 2 * np.cos(np.deg2rad(180.0)) / 3  # nodes 178.5E and 180W, 2/3 of the way
    check(abs(b[1, 100, i] - exact) < 1e-5, "bilinear wraps at the date line")
    with tempfile.TemporaryDirectory() as d:
        c = np.stack(
            [np.full((R.LAT10.size, R.LON10.size), 280.0)]
            + [np.full((R.LAT10.size, R.LON10.size), a) for a in (-12.3, 3.1, 0.7, -0.4, 0.2, 0.05)]
        )
        np.savez_compressed(
            Path(d) / "n.npz",
            mean=c[0].astype(np.float32),
            harm=c[1:].astype(np.float16),
            lat=R.LAT10,
            lon=R.LON10,
            meta="{}",
        )
        N = R.Normals.t2m(Path(d) / "n.npz")
        t = pd.date_range("2026-01-01", "2026-12-31", freq="7D")
        exact = R.HC.evaluate(c[:, :1, :1], R.HC.times_of(t))[:, 0, 0]
        check(
            np.abs(N.at(t)[:, 0, 0] - exact).max() < 0.02, "the 2 m normals' float16 harmonics hold the curve to 0.02 K"
        )


def codes():
    print("codes")
    v = np.array([-31.0, -0.12, 0.0, 0.13, 31.5, np.nan])
    c = E.code_signed(v, 0.25)
    back = np.where(c > 0, (c.astype(float) - 128) * 0.25, np.nan)
    check(
        np.allclose(back[:-1], [-31.0, -0.0, 0.0, 0.25, 31.5], atol=0.13) and c[-1] == 0,
        "anomaly codes round-trip, NaN -> 0",
    )
    m = E.code_mean(np.array([1013.2, np.nan]), 920.0, 0.6)
    check(abs(920 + (m[0] - 1) * 0.6 - 1013.2) <= 0.3 and m[1] == 0, "mean codes round-trip, NaN -> 0")


def gate():
    print("gate")
    x = RNG.normal(0.3, 1.0, (50, 40, 30)).astype(np.float32)
    m, n = E.agreeing(x, np.arange(50))
    check(
        np.allclose(m, x.mean(0), atol=1e-5) and (n == (np.sign(x) == np.sign(x.mean(0))).sum(0)).all(),
        "agreeing counts the members sharing the mean's sign",
    )
    m2, n2 = E.agreeing(x, np.arange(0, 50, 2))
    check(np.allclose(m2, x[::2].mean(0), atol=1e-5) and n2.max() <= 25, "agreeing on a subset of the members")
    check([E.sign_k(k) for k in (50, 51, 31, 21)] == [33, 33, 22, 16], "sign_k: 33/50, 33/51, 22/31, 16/21")


def test():
    print("test")
    a, b = RNG.normal(0.4, 1.0, (31, 200)), RNG.normal(0.0, 1.5, (21, 200))
    p = E.welch_p(a.mean(0), a.std(0, ddof=1), 31, b.mean(0), b.std(0, ddof=1), 21)
    check(np.allclose(p, ttest_ind(a, b, equal_var=False).pvalue), "welch_p equals scipy's Welch test")
    from scipy.stats import f as fisher

    s1, s0 = a.std(0, ddof=1), b.std(0, ddof=1)
    F = s1**2 / s0**2
    ref = 2 * np.minimum(fisher.cdf(F, 30, 20), fisher.sf(F, 30, 20))
    check(np.allclose(E.var_p(s1, 31, s0, 21), ref), "var_p equals the two-sided F-test")
    wide, base, same = RNG.normal(0, 2.0, (50, 2000)), RNG.normal(0, 1.0, (50, 2000)), RNG.normal(0, 1.0, (50, 2000))
    sd = lambda x: x.std(0, ddof=1)  # noqa: E731
    found = E.significant(E.var_p(sd(wide), 50, sd(base), 50)).mean()
    none = E.significant(E.var_p(sd(same), 50, sd(base), 50)).mean()
    check(
        found > 0.95 and none < 0.01,
        f"a doubled spread is found ({found:.0%} of points), identical spreads not ({none:.1%})",
    )
    check(np.isnan(E.var_p(np.zeros(3), 50, np.ones(3), 50)).all(), "var_p: NaN where a run's members do not vary")
    # a newer run's spread 20 % smaller everywhere (the shorter lead): against the band's typical ratio nothing stands
    # out; with twice that spread in a band of columns, those are found and the false discoveries stay near the FDR
    old = RNG.normal(0, 1.0, (50, 60, 400))
    new = RNG.normal(0, 0.8, (50, 60, 400))
    rb = E.band_ratio(sd(new) / sd(old), half=10)
    flat = E.significant(E.var_p(sd(new), 50, sd(old), 50, rb)).mean()
    check(
        np.allclose(rb, 0.8, atol=0.03) and flat < 0.01,
        f"a uniform shrinkage is the typical ratio ({np.median(rb):.2f}) and is not flagged ({flat:.1%})",
    )
    new[:, :, :40] *= 2.0
    s1, s0 = sd(new), sd(old)
    sig = E.significant(E.var_p(s1, 50, s0, 50, E.band_ratio(s1 / s0, half=10)))
    hit, fdp = sig[:, :40].mean(), sig[:, 40:].sum() / max(sig.sum(), 1)
    check(
        hit > 0.9 and fdp < 0.15, f"a planted rise is found ({hit:.0%}); false discoveries {fdp:.0%} of those flagged"
    )
    c = E.code_mean(np.array([0.0, 37.4, np.nan]), 0.0, E.ENC["z500"]["spread"])
    check(
        c[0] == 1 and abs((c[1] - 1) * E.ENC["z500"]["spread"] - 37.4) <= 0.5 and c[2] == 0, "spread codes round-trip"
    )


def chain():
    print("chain")
    pairs = (  # a run and one 6-48 h earlier (CHANGES); IFS-ENS 06/18Z runs to +144 h
        ("2026102412", "2026102500", 360, 360),
        ("2026102500", "2026102512", 360, 360),
        ("2026102500", "2026102506", 360, 144),
        ("2026102418", "2026102506", 144, 144),
        ("2026102506", "2026102512", 144, 360),
        ("2026102418", "2026102512", 360, 360),
        ("2026102406", "2026102500", 144, 360),
        ("2026102400", "2026102500", 360, 360),
        ("2026102412", "2026102600", 360, 360),
        ("2026102318", "2026102506", 360, 144),
        ("2026102400", "2026102600", 360, 360),
        ("2026102412", "2026102612", 360, 360),
    )
    for c0, c1, l0, l1 in pairs:
        i0, i1 = (pd.Timestamp(f"{c[:8]} {c[8:]}:00") for c in (c0, c1))
        d0, d1 = E.full_days(i0, l0), E.full_days(i1, l1)
        S = {"t2m": RNG.normal(0, 1, (21, len(d0), 2, 3)).astype(np.float32)}
        done = {"t2m": np.ones((21, len(d0)), bool)}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.npz"
            E.save_stats(p, i0, d0, S, done)
            prev = E.load_stats(p, i0)
        cut = [E.prev_row(d1, k0, k1, prev) for _, k0, k1 in E.spans(len(d1))]
        held = [d1[k0][0] <= d0[-1][0] for _, k0, _ in E.spans(len(d1))]
        check(
            [c is not None for c in cut] == held,
            f"{c1[8:]}Z: every span starting on a day the {c0[8:]}Z run holds finds its statistics ({sum(held)} of {len(cut)})",
        )
        j = 2 if cut[2] is not None else next(i for i, c in enumerate(cut) if c is not None)
        name, k0, _ = E.spans(len(d1))[j]
        last, row = cut[j]
        j0 = next(j for j, (day, _) in enumerate(d0) if day == d1[k0][0])
        j1 = next(j for j, (day, _) in enumerate(d0) if day == d1[last][0])
        x = S["t2m"][:, j0 : j1 + 1].mean(1)
        check(
            np.allclose(prev["t2m"]["mean"][row], x.mean(0), atol=1e-5),
            f"{name}: the previous run's mean over the shared days ({len(d1)} days, cut to day {last + 1})",
        )
        check(
            np.allclose(prev["t2m"]["sd"][row].astype(float), x.std(0, ddof=1), atol=2e-3)
            and prev["t2m"]["n"][row] == 21,
            f"{name}: its standard deviation and member count",
        )
    check(
        len(E.full_days(pd.Timestamp("2026-10-25 00:00"))) == 15
        and len(E.full_days(pd.Timestamp("2026-10-25 12:00"))) == 14
        and len(E.full_days(pd.Timestamp("2026-10-25 06:00"))) == 14
        and len(E.full_days(pd.Timestamp("2026-10-25 06:00"), 144)) == 5
        and len(E.full_days(pd.Timestamp("2026-10-25 18:00"), 144)) == 5,
        "15 full days from 00Z, 14 from 06 or 12Z, 5 from an IFS-ENS 06/18Z run to +144 h",
    )
    mod = E.MODELS["aifs"]

    def made(c: str, late: int = 0) -> str:  # each run published 8 h after its start
        return f"{pd.Timestamp(f'{c[:8]} {c[8:]}:00') + pd.Timedelta(hours=8, minutes=late):%Y-%m-%dT%H:%MZ}"

    def write(c: str, late: int = 0) -> tuple[str, str, str]:
        E.write_manifest(site, mod, {"cycle": c, "made": made(c, late)}, None, lists)
        return tuple((lists / f"{k}.txt").read_text().strip() for k in ("publish", "prune", "floor"))

    with tempfile.TemporaryDirectory() as d:
        site, lists = Path(d), Path(d) / "lists"
        for c in ("2026102400", "2026102406", "2026102412", "2026102418", "2026102500"):
            write(c)
        _, prune6, floor6 = write("2026102506")
        man = json.loads((site / "assets/models/ensmaps_aifs.json").read_text())
        rerun = write("2026102506", 20)[0]  # a rebuild 20 min later
        stale = write("2026102318", 30)[0]  # older than the four
        _, prune7, floor7 = write("2026102512")
    check(
        [c["cycle"] for c in man["cycles"]] == ["2026102506", "2026102500", "2026102418", "2026102412"]
        and [x["cycle"] for x in man["spare"]] == ["2026102406", "2026102400"]
        and (prune6, floor6.split(":")[1]) == ("", "2026102400"),
        "the manifest keeps the four newest runs; the store also the one that just left and the newest one before",
    )
    check(rerun.startswith("+") and rerun.endswith("/2026102506") and stale == "", "a rebuild is published add-only; "
          "an older run is not published")  # fmt: skip
    check(
        prune7.endswith("/2026102400") and floor7.endswith(":2026102406"),
        "a run that left over an hour before a newer leaver is pruned",
    )


def regimes():
    print("regimes")
    ref = RC.Ref()
    init = pd.Timestamp("2026-10-25 00:00")
    days = E.full_days(init)
    dates = [d.date() for d, _ in days]
    nd, nm = len(dates), 12
    pattern = {}
    an = np.zeros((nd, 73, 144))
    for sk in RC.SECTORS:
        cent, lat, lon = ref.centroids(sk, "cold")
        k = 1 if sk == "ea" else 2
        lon360 = lon % 360
        rows = [int(np.rint((90 - la) / 2.5)) for la in lat]
        cols = [int(np.rint(lo / 2.5)) % 144 for lo in lon360]
        an[:, np.ix_(rows, cols)[0], np.ix_(rows, cols)[1]] = cent[k]
        pattern[sk] = k
    Z25 = np.zeros((nm, nd, 73, 144), np.float32)
    clim = RE.era5_clim(dates)
    Z25[:, :, :37] = (clim + an[:, :37])[None]
    ok = np.ones((nm, nd), bool)

    class Mod:
        key, label, source = "test", "Test", "synthetic members"

    with tempfile.TemporaryDirectory() as d:
        got = ER.labels(Mod, "2026102500", init, dates, Z25, ok, Path(d))
        rec = json.loads((Path(d) / "ensmaps_members_test.json").read_text())
        arch = json.loads((Path(d) / "ensmaps_drift_test.json").read_text())
    labs = got[0]
    check(all((labs[sk] == pattern[sk]).all() for sk in labs), "members on a centroid are labelled with its regime")
    check(
        rec["schema"] == RC.SCHEMA and rec["cycle"] == "2026102500" and len(rec["valid"]) == nd,
        "the record is regime_core's format",
    )
    check(
        list(arch["runs"]) == ["20261025"] and len(arch["runs"]["20261025"]["ea"]) == RE.NDAYS,
        "the 00Z run joins the drift archive",
    )
    a = RNG.normal(size=14)
    rp = RC.running_partial(a, RC.SMOOTH)
    check(
        all(abs(rp[min(k0 + 2, k1)] - a[k0 : k1 + 1].mean()) < 1e-12 for _, k0, k1 in E.spans(14)[:3]),
        "a period's label day carries the period's mean",
    )
    counts = np.array([[20, 20, 20], [5, 5, 5], [6, 6, 6]])
    prev = {"valid": ["2026-10-25", "2026-10-26", "2026-10-27"], "n": 31, "count": counts.tolist()}
    same = ER.vs_previous(counts, 31, dates[:3], prev)
    moved = counts.copy()
    moved[:, 2] = [2, 23, 6]
    shift = ER.vs_previous(moved, 31, dates[:3], prev)
    check(
        not np.any(same["sig"]) and shift["sig"][0][2] and shift["sig"][1][2] and not shift["sig"][2][2],
        "the run-to-run test finds a planted shift only",
    )
    check(len(E.full_days(pd.Timestamp("2026-10-25 12:00"), 384)) == 15, "a 12Z run to +384 h has 15 full days")


def clusters():
    print("clusters")
    keys = [("ifs", i) for i in range(20)]
    w = np.full(20, 0.05)

    def sol(groups):
        lab = np.zeros(20, int)
        for j, g in enumerate(groups):
            lab[list(g)] = j
        return {"k": len(groups), "lab": lab}

    seq = [
        sol([range(10), range(10, 20)]),
        sol([range(20)]),
        sol([range(10), range(10, 17), range(17, 20)]),
        sol([[*range(10), 17, 18, 19], range(10, 17)]),
        sol([range(10), range(10, 17), range(17, 20)]),
    ]
    EC.chain_letters([(keys, s, w) for s in seq])
    check(
        [s["letters"] for s in seq] == [["A", "B"], [None], ["A", "B", "C"], ["A", "B"], ["A", "B", "C"]],
        "letters: kept through a one-scenario day, a new one for a split, the same one when it re-forms",
    )
    check(EC.flow((keys, seq[2]), (keys, seq[3], w)) == [[0.5, 0.0], [0.0, 0.35], [0.15, 0.0]], "the members' moves")


def super_ens():
    print("super")
    from PIL import Image

    ns = {"ifs": 50, "aifs": 51, "gefs": 31, "geps": 21}
    check(all(round(ES.gate((n,)) * n) == E.sign_k(n) for n in ns.values()), "one ensemble's gate is sign_k(n) / n")
    t = ES.gate(tuple(ns.values()))
    T = (RNG.binomial(list(ns.values()), 0.5, (400_000, 4)) / np.array(list(ns.values()))).mean(1)
    check(
        2 * (t - 1e-9 <= T).mean() < 0.052 and 2 * (t - 0.01 <= T).mean() > 0.05,
        f"four ensembles' gate {t:.3f} holds the sign test's 5 % (simulated)",
    )
    g50, E.G50 = E.G50, E.Grid(30.0)  # 7 x 12 cells
    try:
        init = pd.Timestamp("2026-10-25 12:00")
        days = E.full_days(init)
        sp_old = E.stat_spans(len(days), cuts=False)
        z = RNG.normal(0, 1, (len(sp_old), *E.G50.shape)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "s.npz"
            keys = np.array([E.key(days, k0, k1) for k0, k1 in sp_old])
            np.savez(
                f,
                init="2026102512",
                keys=keys,
                mean_z500=z,
                sd_z500=z.astype(np.float16),
                n_z500=np.full(len(keys), 50),
            )
            S = {"t2m": RNG.normal(0, 1, (50, len(days), *E.G50.shape)).astype(np.float32)}
            E.save_stats(f, init, days, S, {"t2m": np.ones((50, len(days)), bool)})
            with np.load(f) as out:
                row = {str(k): i for i, k in enumerate(out["keys"])}
                old_ok = all(np.array_equal(out["mean_z500"][row[k]], z[i]) for i, k in enumerate(keys))
                new = [row[k] for k in row if k not in set(keys)]
                check(
                    old_ok and len(new) > 0 and np.isnan(out["mean_z500"][new]).all() and not out["n_z500"][new].any(),
                    f"a first stage's statistics of an older layout are carried by span ({len(new)} new spans empty)",
                )
        c0, c1 = "2026102500", "2026102512"
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            raw = {}
            for cyc in (c0, c1):
                init = pd.Timestamp(f"{cyc[:8]} {cyc[8:]}:00")
                for m, n in ns.items():
                    days = E.full_days(init, E.MODELS[m].last(cyc[8:]))
                    mu = 0.3 + (0.5 if (cyc, m) == (c1, "gefs") else 0.0)
                    S = {f: RNG.normal(mu, 1.0, (n, len(days), *E.G50.shape)).astype(np.float32) for f in E.FIELDS}
                    done = {f: np.ones((n, len(days)), bool) for f in E.FIELDS}
                    N50 = {f: np.zeros((len(days), *E.G50.shape), np.float32) for f in E.FIELDS}
                    high = np.zeros(E.G50.shape, bool)
                    high[0, 0] = m == "geps"
                    E.save_stats(d / "in" / m / f"{cyc}.npz", init, days, S, done, N50, high)
                    raw[cyc, m] = S
                a = argparse.Namespace(cycle=cyc, stats_dir=d / "in", prev_dir=d / "prev", save=d / "prev" / f"{cyc}.npz",
                                       site=d / "site", manifests=d / "man", manifest_in=None, lists=d / "l", threads=1)  # fmt: skip
                ES.run(a)
            man = json.loads((d / "site/assets/models/ensmaps_super.json").read_text())
            e = man["cycles"][0]
            p1 = next(s for s in e["spans"] if s["key"] == "P1")
            p3 = next(s for s in e["spans"] if s["key"] == "P3")
            check(
                [c["cycle"] for c in man["cycles"]] == [c1, c0] and e["prev"] == "2026-10-25T00:00Z"
                and p1["n"]["z500"] == 153 and (p3["first"], p3["last"]) == ("2026-11-05", "2026-11-08") and len(e["days"]) == 14
                and set(p3["n"]) == set(E.FIELDS),
                "the 12Z super run: 14 days (GEFS and GEPS cut to them), 153 members, the run 12 h earlier",
            )  # fmt: skip
            img = np.asarray(Image.open(d / f"site/{E.DATA}/super/{c1}/z500/a/P1.webp"))
            x = {m: raw[c1, m]["z500"][:, 0:5].mean(1) for m in ns}  # the members' days 1-5 anomalies
            A = sum(x[m].mean(0) for m in ns) / 4
            share = sum((x[m] > 0).mean(0) for m in ns) / 4
            share = np.where(A > 0, share, 1 - share)
            check(
                np.abs(img[..., 0].astype(int) - E.code_signed(A, E.ENC["z500"]["anom"])).max() <= 1,
                "the anomaly is the equal-weight mean of the four ensembles' means",
            )
            check(
                np.array_equal(img[..., 1] >= p1["k"]["z500"], share >= t - 1e-9)
                and np.abs(img[..., 1] / ES.SHARE - share).max() <= 1 / ES.SHARE,
                f"dots where the weighted share agreeing reaches the gate ({p1['k']['z500']} of {ES.SHARE})",
            )
            mslp = np.asarray(Image.open(d / f"site/{E.DATA}/super/{c1}/msl/a/P1.webp"))
            check(
                mslp[0, 0, 0] == 0 and mslp[0, 0, 1] == 255 and mslp[1, 1, 0] > 0,
                "no pressure anomaly where an ensemble's ground is high",
            )
            runs = {m: ES.load_run(d / "in" / m / f"{c1}.npz", pd.Timestamp("2026-10-25 12:00")) for m in ns}
            k = E.key(E.full_days(pd.Timestamp("2026-10-25 12:00")), 0, 4)
            _, V, D, n = ES.combine(runs, "z500", k)
            Vd = sum(x[m].var(0, ddof=1) / x[m].shape[0] for m in ns) / 16
            Dd = sum((x[m].var(0, ddof=1) / x[m].shape[0] / 16) ** 2 / (x[m].shape[0] - 1) for m in ns)
            check(np.allclose(V, Vd, rtol=2e-3) and np.allclose(D, Dd, rtol=5e-3) and n == (50, 51, 31, 21),
                  "the weighted mean's variance and its Satterthwaite denominator")  # fmt: skip
            c = np.asarray(Image.open(d / f"site/{E.DATA}/super/{c1}/t2m/c/P1.webp"))
            y1 = {m: raw[c1, m]["t2m"][:, 0:5].mean(1) for m in ns}  # the 12Z run's days 1-5: the 00Z run's 2-6
            y0 = {m: raw[c0, m]["t2m"][:, 1:6].mean(1) for m in ns}
            dd = sum(y1[m].mean(0) - y0[m].mean(0) for m in ns) / 4
            terms = [y[m].var(0, ddof=1) / y[m].shape[0] / 16 for y in (y1, y0) for m in ns]
            nn = [y[m].shape[0] for y in (y1, y0) for m in ns]
            pv = ES.welch_p(dd, sum(terms), sum(t * t / (n - 1) for t, n in zip(terms, nn, strict=True)), 0.0, 0.0, 0.0)
            check(
                np.abs(c[..., 0].astype(int) - E.code_signed(dd, E.ENC["t2m"]["change"])).max() <= 1
                and np.array_equal(c[..., 1] == 1, E.significant(pv))
                and p1["change"]["t2m"]["n0"] == 153,
                f"the change from the run 12 h earlier: the weighted Welch test over the eight terms (the GEFS shift "
                f"x 1/4 = 0.125 K; {dd.mean():.3f} K, {c[..., 1].mean():.0%} of the cells significant)",
            )
            mans = d / "plan"
            mans.mkdir()
            for m in ns:
                cyc = [{"cycle": c1, "made": "2026-10-26T01:00Z", "last_step": E.MODELS[m].last("12")}]
                (mans / f"ensmaps_{m}.json").write_text(json.dumps({"cycles": cyc}))

            def plan(**kw) -> str:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    ES.cmd_plan(argparse.Namespace(manifests=mans, cycle="", force=False, **kw))
                return out.getvalue().strip()

            first = plan()
            sup = {
                "cycles": [{"cycle": c1, "ensembles": {m: {"made": "2026-10-26T01:00Z", "stage": None} for m in ns}}]
            }
            (mans / "ensmaps_super.json").write_text(json.dumps(sup))
            again = plan()
            sup["cycles"][0]["ensembles"]["ifs"]["made"] = "2026-10-26T00:00Z"
            (mans / "ensmaps_super.json").write_text(json.dumps(sup))
            remade = plan()
            (mans / "ensmaps_gefs.json").write_text(json.dumps({"cycles": []}))
            check(first == c1 and again == "" and remade == c1 and plan() == "",
                  "plan: a cycle all four hold, not again, again when one is remade, not without all four")  # fmt: skip
    finally:
        E.G50 = g50


def telecon():
    from latlon import G25N

    print("telecon")
    lat, lon = E.G50.lat, E.G50.lon
    cap = ET.Capture(2)
    for step in (18, 24, 48):
        cap.put(0, step, np.full(E.G50.shape, 5600.0, np.float32), lat, lon)
    cap.put(1, 24, np.full(E.G50.shape, 5500.0, np.float32), lat, lon)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "gefs.npz"
        cap.save(f, "gefs", "2026102800", pd.Timestamp("2026-10-28"))
        z = np.load(f)
        check(np.all(z["z"][0, :2] == 5600) and np.all(z["z"][1, 0] == 5500) and z["ok"][0, :2].all()
              and not z["ok"][0, 2:].any() and z["ok"][1].tolist() == [True] + [False] * 14,
              "Capture: a constant 500 hPa field stays constant on the 2.5-deg grid; off-day steps skipped, gaps marked")  # fmt: skip

        # a reference with known projectors: the index of normal + c x pattern is c x <p, p> / sigma
        ids, rows = T15.IDS, G25N.shape
        pat = np.linalg.qr(RNG.normal(size=(np.prod(rows), len(ids))))[0].T.reshape(len(ids), *rows)  # orthonormal
        proj = np.stack([np.broadcast_to(p / (p**2).sum(), (12, *rows)) for p in pat])
        normal = np.zeros((366, 2, *rows), np.float32)
        normal[:, 0], normal[:, 1] = 5500.0, 5510.0  # 00 and 12 UTC
        meta = {"patterns": {s: {"name": s.upper()} for s in ids}}
        refp = Path(tmp) / "ref.npz"
        np.savez(refp, ids=np.array(ids), proj=proj, std_month=np.full((12, *rows), 2.0),
                 standardize=np.array([True, False, False]), sigma=np.full((3, 366), 4.0),
                 defined=np.ones((3, 12), bool), normal=normal, meta=np.array(json.dumps(meta)))  # fmt: skip
        ref = T15.Ref(refp)
        t = pd.DatetimeIndex(["2026-01-15 00:00", "2026-07-15 12:00"])
        fields = np.stack([5500 + 3 * pat[0] + 8 * pat[1], 5510 + 3 * pat[0] + 8 * pat[1]])
        v = ref.index(ref.anomaly(fields, t), t)
        check(np.allclose(v[:, 0], 3 / 2 / 4) and np.allclose(v[:, 1], 8 / 4),
              "index: the planted amplitude (over std_month where standardized, over sigma), each hour's own normal")  # fmt: skip
        run = {"key": "gefs", "init": pd.Timestamp("2026-10-28"), "lead_h": np.array(T15.LEADS),
               "ok": np.ones((5, 15), bool),
               "z": (5500 + RNG.normal(0, 40, (5, 15, *rows))).astype(np.int16)}  # fmt: skip
        tt, mem, mean = ET.member_indices(ref, run)
        whole = ref.index(ref.anomaly(run["z"].astype(float).mean(0), tt), tt)
        check(np.allclose(mean, whole) and np.allclose(mean, mem.mean(0)),
              "the ensemble mean's index is the index of the members' mean field")  # fmt: skip
        site = Path(tmp) / "site"
        added = []
        for back in (12 * 24, 36, 24, 0, 0):  # one run older than KEEP_DAYS before the newest, the newest twice
            r = dict(run, init=run["init"] - pd.Timedelta(hours=back))
            added.append(ET.record(site, ref, r, (tt - pd.Timedelta(hours=back), mem, mean)))
        rec = ET.recorded(site)["gefs"]
        check(added == [True, True, True, True, False] and [x["init"] for x in rec] == ["2026-10-26T12", "2026-10-27T00",
              "2026-10-28T00"] and len(rec[-1]["pna"]["members"]) == 15 and len(rec[-1]["pna"]["members"][0]) == 5
              and np.allclose(rec[-1]["epo"]["mean"], np.round(mean[:, 1], 2)),
              "run record: a run is added once, runs over KEEP_DAYS older than the newest dropped, every member kept")  # fmt: skip

    # the hindcast fit on daily starts: an AR(1) observed index, forecasts that lose skill with lead
    starts = pd.date_range("2024-09-01", "2026-10-20", freq="D")
    obs = np.zeros(len(starts) + 16)
    for k in range(1, obs.size):
        obs[k] = 0.85 * obs[k - 1] + RNG.normal(0, np.sqrt(1 - 0.85**2))
    days = pd.date_range(starts[0], periods=obs.size, freq="D")
    rho = {24: 0.95, 240: 0.4, 360: 0.0}
    fc = []
    for h, r in rho.items():
        o = obs[h // 24 : h // 24 + len(starts)]
        f = r * o + np.sqrt(1 - r * r) * RNG.normal(size=len(starts)) if r else RNG.normal(size=len(starts))
        fc.append(pd.DataFrame({"model": "GEFS", "init": starts, "lead_h": h} | dict.fromkeys(T15.IDS, f)))
    fc = pd.concat(fc, ignore_index=True)
    ob = pd.DataFrame({"valid": days} | dict.fromkeys(T15.IDS, obs))
    t0 = pd.Timestamp("2026-10-28")
    cal = ET.calibrate(fc, ob, "GEFS", t0)
    c1, c10, c15 = cal["pna", 24], cal["pna", 240], cal["pna", 360]
    near = np.abs((starts.dayofyear - t0.dayofyear + 183) % 366 - 183) <= ET.WINDOW
    check(abs(c1["r"] - 0.95) < 0.03 and abs(c10["r"] - 0.4) < 0.15 and abs(c15["r"]) < 0.2 and c1["n"] == near.sum()
          and cal["pna", 48] is None, f"the fit's correlations at days 1, 10, 15 ({c1['r']:.2f}, {c10['r']:.2f}, "
          f"{c15['r']:.2f}) from the {c1['n']} starts within {ET.WINDOW} days of the date")  # fmt: skip
    check(
        c1["n_eff"] < 0.4 * c1["n"],
        f"consecutive daily starts are partly one case: day 1's n_eff {c1['n_eff']:.0f} of {c1['n']}",
    )
    allc = {"gefs": cal}
    ET.gate(allc)
    check(ET.skilful(c1) and ET.skilful(c10) and not ET.skilful(c15), "the gate passes days 1 and 10, not day 15")

    # fitted on a large sample of independent cases, the chance of the positive phase matches its frequency on fresh ones
    def cases(n):
        y = 0.3 + 1.2 * RNG.normal(size=n)
        return 0.5 * y + RNG.normal(size=n), y

    c = ET.fit(*cases(4000), np.full(4000, -1))
    x, y = cases(20000)
    p_pos = 1.0 - ET.cal_cdf(0.5, c, x)
    hit = y >= 0.5
    top = p_pos > np.quantile(p_pos, 0.8)
    check(abs(p_pos.mean() - hit.mean()) < 0.04 and abs(p_pos[top].mean() - hit[top].mean()) < 0.06,
          f"P(>= +0.5) is reliable on fresh cases: {p_pos.mean():.3f} forecast, {hit.mean():.3f} observed; the top "
          f"fifth {p_pos[top].mean():.3f}, {hit[top].mean():.3f}")  # fmt: skip


def main() -> int:
    grids()
    grib_checks()
    normals()
    codes()
    gate()
    test()
    chain()
    regimes()
    clusters()
    super_ens()
    telecon()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
