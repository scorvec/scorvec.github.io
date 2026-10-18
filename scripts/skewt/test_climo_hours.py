#!/usr/bin/env python3
"""Offline checks that the sounding climatology is split by synoptic hour, on synthetic soundings.

python scripts/skewt/test_climo_hours.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_climo as B
import flag_anomalies as F


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def header(gid, y, m, d, hh, rel, nlev):
    return f"#{gid} {y:04d} {m:02d} {d:02d} {hh:>2} {rel:>4} {nlev:4d} ncdc-gts ncdc-gts  352472  -974436"


def level(p_pa, gph, t10, dpdp10):
    # LVLTYP1/2, ETIME, PRESS, PFLAG, GPH, ZFLAG, TEMP, TFLAG, RH, DPDP, WDIR, WSPD (IGRA v2 fixed columns),
    # padded to the 52 characters the readers require
    return f"10 {-9999:5d} {p_pa:6d}A{gph:5d}A{t10:5d}A{-9999:5d} {dpdp10:5d} {270:5d} {100:5d} "


# launch hour: the nominal HOUR, else the release time, else unknown
check(B.launch_hour(header("USM00072357", 2020, 7, 11, "00", "2315", 10)) == 0.0, "nominal hour")
check(B.launch_hour(header("USM00072357", 2020, 7, 11, "99", "1415", 10)) == 14.25, "release time when no hour")
check(np.isnan(B.launch_hour(header("USM00072357", 2020, 7, 11, "99", "9999", 10))), "no hour at all")

# synoptic class: +/-3 h of 00Z or 12Z, inclusive; everything else excluded
for h, want in [(0, 0), (23, 0), (21, 0), (3, 0), (2.5, 0), (4, None), (6, None), (9, 12), (14, 12), (15, 12)]:
    check(B.synoptic_class(h) == want, (h, B.synoptic_class(h)))
for h in (16, 18, 20, float("nan"), None, "x"):
    check(B.synoptic_class(h) is None, h)
check(list(B.hour_classes([0, 6, 12, 18, 23, np.nan])) == [0, -1, 12, -1, 0, -1], "vectorised classes")

# a station with warm 00Z and cold 12Z launches, 14Z specials and extreme 06Z specials
rng = np.random.default_rng(1)
years = np.arange(1980, 2020)
doy, yr, hod, t850 = [], [], [], []
for y in years:
    for d in range(1, 366):
        for h, base in ((0.0, 20.0), (12.0, 10.0)):
            doy.append(d)
            yr.append(y)
            hod.append(h)
            t850.append(base + rng.normal(0, 1))
    for d in range(150, 200, 7):
        doy.append(d)
        yr.append(y)
        hod.append(6.0)  # off-hour special: must not enter any table
        t850.append(30.0)
        doy.append(d)
        yr.append(y)
        hod.append(14.0)  # 14Z special: counts for 12Z
        t850.append(10.0)
doy.append(180)
yr.append(2019)
hod.append(np.nan)  # no hour: excluded
t850.append(-40.0)
s = {
    "doy": np.array(doy, dtype="int16"),
    "yr": np.array(yr, dtype="int16"),
    "hod": np.array(hod, dtype="float32"),
    "v": {"850t": np.array(t850, dtype="float32")},
}
out = B.station_tables("TEST0000001", s)
check(out["v"] == B.FORMAT and set(out["hr"]) == {"00", "12"}, out.keys())
a = B.DOY.index(181)
m00 = out["hr"]["00"]["850t"]["p"][a][4]
m12 = out["hr"]["12"]["850t"]["p"][a][4]
check(abs(m00 - 20) < 0.3 and abs(m12 - 10) < 0.3, (m00, m12))
pool_n = out["idx"]["850t"]["n"][a]
check(pool_n == out["hr"]["00"]["850t"]["n"][a] + out["hr"]["12"]["850t"]["n"][a], "pooled table = both hours")
check(max(out["hr"]["00"]["850t"]["max"][a], out["hr"]["12"]["850t"]["max"][a]) < 30, "06Z specials excluded")
check(min(out["hr"]["12"]["850t"]["min"][a], out["hr"]["00"]["850t"]["min"][a]) > -40, "hourless launch excluded")
n_win = 21 * len(years)
check(out["hr"]["00"]["850t"]["n"][a] == n_win, out["hr"]["00"]["850t"]["n"][a])
check(out["hr"]["12"]["850t"]["n"][a] > n_win, "14Z specials counted at 12Z")
check(out["nh"]["00"] == 365 * len(years) and out["nh"]["12"] == 365 * len(years) + 8 * len(years), out["nh"])

# a 00Z-only station has no 12Z table
only00 = {k: (v[s["hod"] == 0] if k != "v" else {"850t": v["850t"][s["hod"] == 0]}) for k, v in s.items()}
o2 = B.station_tables("TEST0000002", only00)
check(set(o2["hr"]) == {"00"}, o2["hr"].keys())

# readers: like with like, no table off-hour, older pooled files as they are
t, ts = F.climo_tables(out, "2026-06-30 00:00")
check(t is out["hr"]["00"] and len(ts) == 2, "00Z launch -> 00Z table")
t, _ = F.climo_tables(out, "2026-06-30 13:00")
check(t is out["hr"]["12"], "13Z launch -> 12Z table")
check(F.climo_tables(out, "2026-06-30 06:00")[0] is None, "06Z launch not ranked")
check(F.climo_tables(out, "2026-06-30")[0] is None, "no hour, not ranked")
check(F.climo_tables(o2, "2026-06-30 12:00")[0] is None, "no 12Z climatology at a 00Z-only station")
v1 = {"doy": B.DOY, "idx": out["idx"]}
check(F.climo_tables(v1, "2026-06-30 12:00")[0] is out["idx"], "format-1 file pooled")

# 20 C at 00Z is ordinary; at 12Z it beats the record for the time of year
d00 = {k: out["hr"]["00"]["850t"][k][a] for k in ("p", "min", "max")}
d12 = {k: out["hr"]["12"]["850t"][k][a] for k in ("p", "min", "max")}
check(40 < F.pct_of(d00, 20.0) < 60 and F.pct_of(d12, 20.0) == 100.0, (F.pct_of(d00, 20.0), F.pct_of(d12, 20.0)))
# the all-time record spans every hour's anchors
hi = F.all_time(ts, "850t", "high")
check(hi[0] == max(max(x for x in out["hr"][h]["850t"]["max"] if x is not None) for h in ("00", "12")), hi)

# ECAPE/SHIP attach by date AND hour: a 00Z sounding the helper skipped must not take the 12Z value
rows = [(180, 2020, 6, 28, 0, 0.0, {"850t": 20.0}), (180, 2020, 6, 28, 12, 12.0, {"850t": 10.0})]
r = B.reduce_rows(rows, {(2020, 6, 28, 12): [(1500.0, 1.2)]})
check(np.isnan(r["v"]["ecape"][0]) and r["v"]["ecape"][1] == 1500.0, r["v"]["ecape"])
r = B.reduce_rows(rows, {(2020, 6, 28, None): [(100.0, 0.1), (200.0, 0.2)]})  # helper without hours
check(list(r["v"]["ecape"]) == [100.0, 200.0], r["v"]["ecape"])

# the period-of-record parser carries each launch's hour
levels = [
    level(100000, 110, 250, 30),
    level(92500, 790, 210, 30),
    level(85000, 1510, 170, 30),
    level(70000, 3150, 80, 50),
    level(50000, 5850, -80, 100),
    level(40000, 7550, -180, 100),
    level(30000, 9650, -320, 100),
    level(25000, 10900, -420, 100),
    level(20000, 12350, -530, 100),
    level(10000, 16500, -700, 100),
]
txt = "\n".join(
    [
        header("USM00072357", 2020, 7, 11, "00", "2315", len(levels)),
        *levels,
        header("USM00072357", 2020, 7, 11, "99", "1805", len(levels)),
        *levels,
    ]
)
rows, _ = B.parse_por(txt, "USM00072357", 357.0)
check([(r[4], round(r[5], 2)) for r in rows] == [(0, 0.0), (99, 18.08)], [(r[4], r[5]) for r in rows])
check(abs(rows[0][6]["850t"] - 17.0) < 1e-6, rows[0][6])
check(list(B.hour_classes([r[5] for r in rows])) == [0, -1], "18Z release excluded")
print("all climatology hour checks passed")
