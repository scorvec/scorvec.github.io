#!/usr/bin/env python3
"""Offline checks of wave_compass.py on synthetic waves and series (no network, no figures).

  python scripts/strat/test_wave_compass.py

ridges      a 55-65N band profile cos(k (lon - lon0)) through build_wave_clim.band_coeff has its ridge where the page
            puts it, -arg(c)/k, at amplitude |c|, for wave-1 and wave-2
relation    a synthetic ERA5 series whose flux follows the interference term gives a significant positive weight and
            a binned flux falling from lined up to opposed; a flux that ignores the waves is not "linked"
agreement   members scattered round a common anomaly agree (small p), members with random phases do not
write       the run's file carries every member and day, the normals and the relation, in the stated convention
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_wave_clim as BWC
import wave_compass as WC


def check(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL {msg}")
    print(f"ok   {msg}")


def ridges():
    lat = np.arange(40.0, 80.5, 1.0)
    lon = np.arange(0.0, 360.0, 1.5)
    for k, lon0, amp in ((1, 70.0, 120.0), (2, -35.0, 80.0)):
        z = amp * np.cos(k * np.deg2rad(lon - lon0))[None, None, :] + 0 * lat[None, :, None]
        c = BWC.band_coeff(z, lat)[0, k - 1]
        ridge = (-np.degrees(np.angle(c)) / k) % (360 / k)
        check(abs(ridge - lon0 % (360 / k)) < 1e-6 and abs(abs(c) - amp) < 1e-6,
              f"wave-{k}: ridge {ridge:.1f} E for {lon0 % (360 / k):.1f}, amplitude {abs(c):.1f} for {amp}")  # fmt: skip


def series(linked: bool, seed=1):
    """1991-2020 daily: wave coefficients round the real normal, and a flux that follows 2 Re(N A*) or ignores it."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    c, f = [], []
    for k in WC.WAVES:
        N = WC.normal(t, k)
        A = 90 * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t))) / np.sqrt(2)
        c.append(N + A)
        lin = 2 * np.real(N * np.conj(A))
        f.append(6 + (4e-4 * lin if linked else 0) + rng.standard_normal(len(t)))
    return {"date": t.values.astype("datetime64[D]"), "c": np.stack(c, 1), "f": np.stack(f, 1)}


def relation():
    day = pd.Timestamp("2026-12-01")
    for k in WC.WAVES:
        r = WC.relation(day, k, series(True))
        m = [b["mean"] for b in r["bins"]]
        check(r["linked"] and r["fit"]["b90"][0] > 0, f"wave-{k} linked: b {r['fit']['b']:.2e}, 90% {r['fit']['b90']}")
        check(m[0] > r["mean"] > m[-1] and all(np.diff(m) < 0), f"wave-{k} bins fall from lined up to opposed: {m}")
        # 61 days a year, less the series' last LEAD - 1 days (no full flux window after them)
        check(r["n_days"] == 30 * 61 - (WC.LEAD - 1) and r["winters"] == 30,
              f"wave-{k} window: {r['n_days']} days in {r['winters']} winters")  # fmt: skip
        n = WC.relation(day, k, series(False))
        check(
            not n["linked"] or abs(n["fit"]["b"]) < 1e-5,
            f"wave-{k} no link when the flux ignores the waves: b {n['fit']['b']:.1e}",
        )


def agreement():
    rng = np.random.default_rng(3)
    shared = 60 * np.exp(1j * 0.7) + 25 * (rng.standard_normal(50) + 1j * rng.standard_normal(50))
    random = 60 * np.exp(1j * rng.uniform(0, 2 * np.pi, 50))
    p1, p2 = WC.agreement(shared), WC.agreement(random)
    check(p1 < 1e-6 and p2 > 0.05, f"agreement: shared anomaly p {p1:.1e}, random phases p {p2:.2f}")
    check(WC.agreement(shared[:3]) is None, "agreement: too few members gives no test")


def write():
    init = pd.Timestamp("2026-10-29T12:00")
    valid = pd.date_range(init, periods=16, freq="D")
    rng = np.random.default_rng(5)
    c500 = {k: WC.normal(valid, k)[:, None] + 50 * (rng.standard_normal((16, 31)) + 1j * rng.standard_normal((16, 31)))
            for k in WC.WAVES}  # fmt: skip
    c100 = {k: c500[k].mean(1) * 1.8 for k in WC.WAVES}
    with tempfile.TemporaryDirectory() as d:
        WC.write(init, valid, c500, c100, "gefs", "GEFS", Path(d), series(True))
        doc = json.loads((Path(d) / "wave_compass_gefs.json").read_text())
    w = doc["waves"]["2"]
    check(doc["n_members"] == 31 and len(doc["valid"]) == 16, "write: 31 members, 16 days")
    check(np.shape(w["members"]) == (16, 31, 2) and np.shape(w["normal"]) == (16, 2) and len(w["p_anomaly"]) == 16,
          "write: members (day, member, re/im), normals and tests per day")  # fmt: skip
    check(abs(w["members"][3][7][0] - round(c500[2][3, 7].real, 1)) < 1e-9, "write: member values to 0.1 m")
    check(w["era5"]["linked"] and len(w["era5"]["bins"]) == 6, "write: the ERA5 relation for the run's date")


if __name__ == "__main__":
    ridges()
    relation()
    agreement()
    write()
    print("all checks passed")
