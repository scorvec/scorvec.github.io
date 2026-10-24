#!/usr/bin/env python3
"""Offline checks of the North America 2 m temperature product (natemp.py, natemp_ref.py) on synthetic data with known
answers: no data are fetched and no figure is drawn.

    python scripts/natemp/test_natemp.py

  normals  a known three-harmonic curve: its month averages fitted back exactly (the saved file's Normals reproduce it),
           a day's normal is the curve's mean at the day's four forecast times; ERA5's sea fill takes no value from a
           land node when a sea node is near
  grib     read_field maps a 180..334 E message with missing values onto the box; decode keeps the box of a global
           0.1-deg 2t message starting at 0 E or at 180 E, and refuses another grid
  days     a 00Z run has 15 full days (steps 6..360), a 12Z run 14 (18..348); windows 1-5, 6-10, 11-15/14
  read     read_run on fake steps: the members' daily means, a member missing one field loses that day only
  gate     the two-thirds agreement on the sign of the mean, also when the members are skewed; that share passes
           a sign test at 5 % for every member count drawn
  test     welch_p equals scipy's Welch test; with no change about q of the maps get any significant point after
           Benjamini-Hochberg, a planted change is found
  chain    a run's statistics serve the run 12 h later, 00Z -> 12Z and 12Z -> 00Z: every window and day it holds, the
           last window cut to the shared days, the means and standard deviations those of the members
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_ind

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import natemp as T  # (imports pyproj before eccodes)
import natemp_ref as R
import eccodes
import harmonic_clim as HC

RNG = np.random.default_rng(20261023)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}", flush=True)


def grib(field: np.ndarray, lat0: float, lon0: float, inc: float, number: int = 1, bits: int = 16) -> bytes:
    """A GRIB2 2 m temperature message of one ensemble member on a regular grid, north first."""
    g = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    nj, ni = field.shape
    for k, v in (
        ("centre", "ecmf"),
        ("productDefinitionTemplateNumber", 1),
        ("shortName", "2t"),
        ("perturbationNumber", number),
        ("Ni", ni),
        ("Nj", nj),
        ("iDirectionIncrementInDegrees", inc),
        ("jDirectionIncrementInDegrees", inc),
        ("latitudeOfFirstGridPointInDegrees", lat0),
        ("latitudeOfLastGridPointInDegrees", round(lat0 - inc * (nj - 1), 6)),
        ("longitudeOfFirstGridPointInDegrees", lon0),
        ("longitudeOfLastGridPointInDegrees", round(lon0 + inc * (ni - 1), 6)),
        ("jScansPositively", 0),
        ("bitsPerValue", bits),
    ):
        eccodes.codes_set(g, k, v)
    if np.isnan(field).any():
        eccodes.codes_set(g, "bitmapPresent", 1)
        eccodes.codes_set_double(g, "missingValue", 9999.0)
        field = np.where(np.isnan(field), 9999.0, field)
    eccodes.codes_set_values(g, field.ravel().astype(np.float64))
    msg = eccodes.codes_get_message(g)
    eccodes.codes_release(g)
    return msg


print("normals: harmonic curve, daily normal, sea fill")
true = np.array([280.0, -12.0, -3.0, 1.5, 0.8, -0.6, 0.3])  # mean, cos 1..3, sin 1..3 (harmonic_clim order)
coef_true = true[:, None, None] * np.ones((1, R.LAT.size, R.LON.size))
months = np.tensordot(HC.design(), coef_true, axes=(1, 0))
coef = HC.fit(months)
check(np.abs(coef - coef_true).max() < 1e-9, "the month averages of a three-harmonic curve give back its coefficients")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "n.npz"
    np.savez_compressed(path, coef=coef.astype(np.float32), lat=R.LAT, lon=R.LON, meta='{"x": 1}')
    N = R.Normals(path)
day = pd.Timestamp("2028-02-29")
want = np.mean([HC.evaluate(true, HC.times_of(day + pd.Timedelta(hours=h)))[0] for h in T.SAMPLES])
got = T.day_normals(N, [day])[0]
check(np.abs(got - want).max() < 1e-3, "a day's normal is the curve's mean at its 06, 12, 18 and 24 UTC (29 Feb)")
point = R.Normals.__new__(R.Normals)  # one grid point's curve
point.coef = true[:, None, None]
days = pd.date_range("2026-01-01", "2026-12-31")
daily = T.day_normals(point, days)[:, 0, 0]
mm = np.array([daily[days.month == m].mean() for m in range(1, 13)])
# within 0.05 K: the four times centre on 15 UTC, three hours after the middle of the day the month averages integrate
check(np.abs(mm - months[:, 0, 0]).max() < 0.05, "the daily normals average to the monthly normals (within 0.05 K)")
sea = np.ones((R.LAT25.size, R.LON25.size))
t25 = np.full((12, R.LAT25.size, R.LON25.size), 285.0)
sea[100:, 300] = 0.0  # a column of land nodes, 25 K colder
t25[:, 100:, 300] = 260.0
sea[150, 290:292] = 0.0  # ...and a 2 x 2 block of land, so one 0.1-deg point has no sea node around it
sea[151, 290:292] = 0.0
t25[:, 150:152, 290:292] = 260.0
fill = R.sea_interp(t25, sea)
j, i = R.cell(R.LAT25[120], R.LON25[300] + 0.1)  # between the land column and the sea column east of it
check(abs(fill[0, j, i] - 285.0) < 1e-6, "the sea fill takes nothing from a land node beside a sea node")
j, i = R.cell(R.LAT25[150] - 0.1, R.LON25[290] + 0.1)  # inside the land block: plain bilinear
check(abs(fill[0, j, i] - 260.0) < 1e-6, "a point with no sea node around takes the plain bilinear value")

print("grib: the box, missing values, the grid check")
f = RNG.normal(280, 5, (R.LAT.size, R.LON.size))
f[:50] = np.nan
g = eccodes.codes_new_from_message(grib(f, 77.0, 180.0, 0.1, bits=24))
try:
    v = R.read_field(g, R.LAT, R.LON)
finally:
    eccodes.codes_release(g)
check(
    np.isnan(v[:50]).all() and np.abs(v[50:] - f[50:]).max() < 1e-3, "read_field: 180..334 E onto -180..-26, NaN kept"
)
lat = np.round(np.arange(900, -901, -1) / 10.0, 1)
lon = np.round(np.arange(3600) / 10.0, 1)
glob = 250 + 0.5 * lat[:, None] + 0.01 * lon[None, :]
num, box = T.decode(grib(glob, 90.0, 0.0, 0.1, number=7))
want = glob[R.ROW0 : R.ROW0 + T.NY, 1800 : 1800 + T.NX]
check(num == 7 and np.abs(box - want).max() < 2e-3, "decode: member 7 and the box rows 77N..9N, columns 180W..26W")
num, box = T.decode(grib(np.roll(glob, 1800, axis=1), 90.0, 180.0, 0.1, number=8))
check(num == 8 and np.abs(box - want).max() < 2e-3, "decode: the same box from a grid that starts at 180 E")
try:
    T.decode(grib(np.full((721, 1440), 280.0), 90.0, 0.0, 0.25))
    check(False, "decode refuses a 0.25-deg message")
except RuntimeError:
    check(True, "decode refuses a 0.25-deg message")

print("days: the run's full days and windows")
d00 = T.full_days(pd.Timestamp("2026-10-23 00:00"))
d12 = T.full_days(pd.Timestamp("2026-10-23 12:00"))
check(len(d00) == 15 and d00[0][1] == [6, 12, 18, 24] and d00[-1][1][-1] == 360, "00Z: 15 days, steps 6..360")
check(len(d12) == 14 and d12[0][1] == [18, 24, 30, 36] and d12[-1][1][-1] == 348, "12Z: 14 days, steps 18..348")
check(d12[0][0] == pd.Timestamp("2026-10-24") and d12[-1][0] == pd.Timestamp("2026-11-06"), "12Z: Oct 24 to Nov 6")
check(
    T.windows(15) == [(0, 4), (5, 9), (10, 14)] and T.windows(14) == [(0, 4), (5, 9), (10, 13)],
    "windows 1-5, 6-10, 11-",
)
check(
    [s[0] for s in T.spans(14)][:4] == ["P1", "P2", "P3", "D01"] and len(T.spans(14)) == 17, "frames P1-P3, then days"
)

print("read: daily means on fake steps")
init = pd.Timestamp("2026-10-23 00:00")
one = d00[:2]
cache = {}


def fake(date, hh, step):
    msgs = []
    for m in (1, 2, 3):
        if m == 3 and step == 18:
            continue  # member 3 misses day 1's 18 UTC field
        if (m, step) not in cache:
            cache[m, step] = grib(glob + m + step / 6.0, 90.0, 0.0, 0.1, number=m)
        msgs.append(cache[m, step])
    return msgs, sum(map(len, msgs))


A, done, info = T.read_run("20261023", "00", one, fake)
w1 = want + 1 + np.mean([6, 12, 18, 24]) / 6.0
w2 = want + 2 + np.mean([30, 36, 42, 48]) / 6.0
check(
    np.abs(A[0, 0] - w1).max() < 2e-3 and np.abs(A[1, 1] - w2).max() < 2e-3, "a member's day: the mean of its 4 fields"
)
check(done[:3].tolist() == [[True, True], [True, True], [False, True]], "a missing field drops that member-day only")
check(not done[3:].any() and np.isnan(A[3:]).all(), "members without messages stay missing")
check(info["messages"] == 2 * 4 * 3 - 1, "every message decoded once")

print("gate: agreement on the sign of the mean")
x = np.ones((50, 4))
x[:16, 1] = -1.0  # 68 % positive: coloured
x[:17, 2] = -1.0  # 66 % positive: split
x[:, 3] = 0.1
x[:16, 3] = -5.0  # 68 % positive but the mean is negative: split
m, gate = T.agreed(x)
check(gate.tolist() == [True, True, False, False], "coloured at >= 2/3 on the mean's sign, never on a skewed mean")
x = np.ones((48, 1))
x[:16] = -1.0  # exactly two thirds
check(T.agreed(x)[1].tolist() == [True], "exactly two thirds of 48 members is coloured")
p = [binomtest(int(np.ceil(T.AGREE * n - 1e-9)), n).pvalue for n in range(T.MIN_MEMBERS, T.MEMBERS + 1)]
check(max(p) < 0.05, f"the share passes a two-sided 5 % sign test for {T.MIN_MEMBERS}-{T.MEMBERS} members")

print("test: Welch, FDR")
a, b = RNG.normal(0.3, 1.0, (50, 200)), RNG.normal(0.0, 2.0, (50, 200))
p = T.welch_p(a.mean(0), a.std(0, ddof=1), 50, b.mean(0), b.std(0, ddof=1), 50)
check(np.allclose(p, ttest_ind(a, b, equal_var=False).pvalue), "welch_p equals scipy's Welch test")
fam = np.ones((40, 50), bool)
maps = 0
for _ in range(200):
    a, b = RNG.normal(0, 1, (50, 40, 50)), RNG.normal(0, 1, (50, 40, 50))
    p = T.welch_p(a.mean(0), a.std(0, ddof=1), 50, b.mean(0), b.std(0, ddof=1), 50)
    maps += T.significant(p, fam).any()
check(maps / 200 <= T.FDR_Q + 0.05, f"no change: {maps} of 200 maps get a significant point (q = {T.FDR_Q})")
a = RNG.normal(0, 1, (50, 40, 50))
a[:, :10] += 1.5
b = RNG.normal(0, 1, (50, 40, 50))
sig = T.significant(T.welch_p(a.mean(0), a.std(0, ddof=1), 50, b.mean(0), b.std(0, ddof=1), 50), fam)
check(sig[:10].mean() > 0.9 and sig[10:].mean() < 0.05, "a planted 1.5-sd change is found, little else")

print("chain: statistics from one run to the next")
with tempfile.TemporaryDirectory() as tmp:
    for a_init, b_init in (("2026-10-23 00:00", "2026-10-23 12:00"), ("2026-10-23 12:00", "2026-10-24 00:00")):
        ia, ib = pd.Timestamp(a_init), pd.Timestamp(b_init)
        da, db = T.full_days(ia), T.full_days(ib)
        Aa = RNG.normal(280, 2, (50, len(da), 3, 4)).astype(np.float32)
        mem = np.arange(48)
        path = Path(tmp) / f"{ia:%Y%m%d%H}.npz"
        T.save_stats(path, ia, da, Aa, mem)
        prev = T.load_stats(path, ib - pd.Timedelta(hours=12))
        got = {name: T.prev_row(db, k0, k1, prev) for name, k0, k1 in T.spans(len(db))}
        tag = f"{ia:%HZ} -> {ib:%HZ}"
        held = [n for n, v in got.items() if v is not None]
        if len(db) == 14:
            check(len(held) == 17, f"{tag}: every window and day of the new run found")
        else:
            check(
                len(held) == 17 and got["D15"] is None and got["P3"][0] == 13, f"{tag}: days 11-14 of 11-15, no day 15"
            )
        name, k0, k1 = T.spans(len(db))[0]
        c1, row = got[name]
        off = [d for d, _ in da].index(db[k0][0])
        x = T.member_mean(Aa, mem, off + k0, off + c1)
        ok = np.allclose(prev["mean"][row], x.mean(0), atol=1e-4) and np.allclose(
            prev["sd"][row], x.std(0, ddof=1), atol=2e-3
        )
        check(ok and prev["n"] == 48, f"{tag}: the stored days 1-5 are the members' mean and sd over the same dates")
    check(T.load_stats(path, pd.Timestamp("2026-10-22 00:00")) is None, "statistics of another run are refused")
print("all checks passed")
