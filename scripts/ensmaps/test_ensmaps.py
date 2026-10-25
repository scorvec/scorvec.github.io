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
  test     welch_p equals scipy's Welch test
  chain    a run's statistics serve the runs 6 and 12 h later (IFS-ENS 06/18Z to +144 h among them), the last window
           cut to the shared days; the manifest keeps the four newest runs, the store those that left in the last hour
           and the newest before them; a rebuild publishes add-only
  regimes  members whose 500 hPa anomaly is a regime's centroid are labelled with it (every day in its own season's
           set) and the record is regime_core's format; a period's label day is the middle day, whose 5-day running
           mean is the period's own mean; the run-to-run test finds a planted shift and nothing in identical runs; a
           12Z run's day 16 needs +384 h
  clusters a scenario keeps its letter through a one-scenario day, a split takes a new letter and a scenario that
           re-forms its old one; the members' moves between two periods' clusters
  interim  column_height gives back the height of a standard column from its surface and sea-level pressures
"""

from __future__ import annotations

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
import regime_core as RC
import regimes_ecmwf as RE
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


def chain():
    print("chain")
    pairs = (  # the run 12 h earlier, and 6 h earlier; IFS-ENS 06/18Z runs to +144 h
        ("2026102412", "2026102500", 360, 360),
        ("2026102500", "2026102512", 360, 360),
        ("2026102500", "2026102506", 360, 144),
        ("2026102418", "2026102506", 144, 144),
        ("2026102506", "2026102512", 144, 360),
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


def interim():
    print("interim")
    z, t0 = np.array([0.0, 500.0, 1500.0, 4000.0]), 288.15
    t = t0 - R.GAMMA * z  # a standard column: its sea-level pressure from the surface one
    sp = 101325.0 * (t / t0) ** (9.80665 / (R.RD * R.GAMMA))
    check(
        np.allclose(R.column_height(np.full(4, 101325.0), sp, t), z, atol=0.01), "column_height inverts the reduction"
    )


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
    interim()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
