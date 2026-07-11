#!/usr/bin/env python3
"""Flag near-record soundings for the Skew-T map's "record watch".

For each just-mirrored latest sounding, compute the level indices and compare
them to that station's climatology (climo/{gid}.json). Any index below the 5th
or above the 95th monthly percentile is flagged. Writes anomalies.json
{wmo: {dt, flags: [{k, v, pct, sense}]}} into the mirror output, which the map
fetches to color and annotate the anomalous stations.

    python scripts/skewt/flag_anomalies.py SOUNDINGS_DIR CLIMO_DIR OUT.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STATIONS = HERE.parents[1] / "skewt" / "stations.json"
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
# Only the indices worth interrupting someone for. PWAT and K-index fire on any
# humid day and would drown the signal; these four actually mark an unusual
# airmass or an unusual amount of instability.
WATCH = ("h500", "thick", "850t", "ecape")
# Some indices are only interesting at ONE end. A station with no convective
# energy is in its normal state — "record low ECAPE" is not news, it's just a
# quiet day, and it crowds out the genuinely unstable soundings. Heights,
# thicknesses and 850 mb temperatures matter at both ends (a record-cold airmass
# is as notable as a record-warm one).
HIGH_ONLY = {"ecape"}
# A value must also be big enough to mean anything. Where an index is almost
# always zero (SHIP in the Arctic; ECAPE over the poles), the percentile is
# computed against a spike at zero and any trace value scores "P99".
FLOOR = {"ecape": 100.0, "ship": 0.5}
LABELS = {"h500": "500mb hgt", "thick": "1000-500 thick", "850t": "850mb T",
          "ecape": "ECAPE"}
G = 9.80665


def indices(P, T, D, H):
    """P (Pa), T/D (K), H (m), sfc→top. Returns index dict (°C / m / mm)."""
    P, T, D, H = map(np.asarray, (P, T, D, H))

    def at(field, plevel):
        pv = plevel * 100.0
        if pv >= P[0]:
            return field[0]
        for i in range(1, len(P)):
            if P[i] <= pv:
                if not (np.isfinite(field[i]) and np.isfinite(field[i - 1])):
                    return np.nan
                f = np.log(P[i - 1] / pv) / np.log(P[i - 1] / P[i])
                return field[i - 1] + f * (field[i] - field[i - 1])
        return np.nan

    m = np.isfinite(P) & np.isfinite(D)
    pw = np.nan
    if m.sum() >= 3:
        Pm, Dm = P[m], D[m]
        Tdc = Dm - 273.15
        e = 6.112 * np.exp(17.67 * Tdc / (Tdc + 243.5)) * 100.0
        w = 0.622 * e / np.maximum(Pm - e, 1.0)
        pw = float(np.sum(0.5 * (w[:-1] + w[1:]) * -np.diff(Pm)) / G)

    t850, t700, t500 = at(T, 850), at(T, 700), at(T, 500)
    d850, d700 = at(D, 850), at(D, 700)
    h500, h1000 = at(H, 500), at(H, 1000)
    fzl = np.nan
    for i in range(1, len(P)):
        if np.isfinite(T[i]) and np.isfinite(T[i - 1]) and T[i - 1] >= 273.15 > T[i]:
            f = (273.15 - T[i - 1]) / (T[i] - T[i - 1])
            fzl = float(H[i - 1] + f * (H[i] - H[i - 1]) - H[0]); break

    def c(k):
        return np.nan if np.isnan(k) else k - 273.15
    d = {"pwat": pw, "850t": c(t850), "700t": c(t700), "500t": c(t500), "h500": h500,
         "thick": (h500 - h1000) if np.isfinite(h500) and np.isfinite(h1000) else np.nan,
         "fzl": fzl, "kidx": (c(t850) - c(t500)) + c(d850) - (c(t700) - c(d700)),
         "tott": c(t850) + c(d850) - 2 * c(t500)}
    return {k: v for k, v in d.items() if np.isfinite(v)}


def parse_csv(text):
    P, T, D, H = [], [], [], []
    for ln in text.splitlines()[1:]:
        c = ln.split(",")
        if len(c) < 7:
            continue
        try:
            p = float(c[3]); t = float(c[5]); dp = float(c[6]); h = float(c[4])
        except ValueError:
            continue
        if p < 20:
            continue
        P.append(p * 100); H.append(h); T.append(t + 273.15); D.append(dp + 273.15)
    return P, T, D, H


def pct_of(d, v):
    X = [d["min"]] + d["p"] + [d["max"]]
    Y = [0] + PCTS + [100]
    if v <= X[0]:
        return 0.0
    if v >= X[-1]:
        return 100.0
    for i in range(1, len(X)):
        if v <= X[i]:
            f = (v - X[i - 1]) / ((X[i] - X[i - 1]) or 1)
            return Y[i - 1] + f * (Y[i] - Y[i - 1])
    return 50.0


def ecape_for(sdir: Path) -> dict:
    """ECAPE per station, from the SAME WebAssembly build the browser runs — so
    the live value and its climatological percentile share identical physics."""
    js = HERE / "ecape_latest.js"
    if not js.exists():
        return {}
    try:
        r = subprocess.run(["node", str(js), str(sdir)], capture_output=True,
                           text=True, timeout=300)
    except Exception as e:                                # noqa: BLE001
        print(f"  ecape helper unavailable ({repr(e)[:40]})", flush=True)
        return {}
    out = {}
    for ln in r.stdout.splitlines():
        q = ln.split()
        if len(q) == 2:
            try:
                out[q[0]] = float(q[1])
            except ValueError:
                pass
    print(f"  ECAPE computed for {len(out)} soundings", flush=True)
    return out


def main() -> int:
    sdir, cdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    wmo2gid = {s["id"]: s["gid"] for s in json.loads(STATIONS.read_text())["stations"]}
    ecape = ecape_for(sdir)
    manifest = json.loads((sdir.parent / "manifest.json").read_text()) \
        if (sdir.parent / "manifest.json").exists() else {"entries": {}}
    entries = manifest.get("entries", {})

    anomalies = {}
    for wmo, e in entries.items():
        gid = wmo2gid.get(wmo)
        cf = cdir / f"{gid}.json" if gid else None
        csv = sdir / f"{wmo}.csv"
        if not gid or not cf or not cf.exists() or not csv.exists():
            continue
        try:
            idx = indices(*parse_csv(csv.read_text()))
            climo = json.loads(cf.read_text())
        except Exception:                                 # noqa: BLE001
            continue
        if wmo in ecape:
            idx["ecape"] = ecape[wmo]
        idx = {k: v for k, v in idx.items() if k in WATCH}   # only what's worth flagging
        # day-of-year climatology: nearest 5-day anchor to the sounding's date
        try:
            dt = datetime.strptime(e.get("dt", "")[:10], "%Y-%m-%d")
            doy = min(365, dt.timetuple().tm_yday)
        except ValueError:
            continue
        anchors = climo.get("doy") or []
        if not anchors:
            continue
        s = min(range(len(anchors)),
                key=lambda i: min(abs(anchors[i] - doy), 365 - abs(anchors[i] - doy)))
        flags = []
        for k, v in idx.items():
            A = climo.get("idx", {}).get(k)
            if not A or not A["p"][s] or A["p"][s][0] is None or A["n"][s] < 30:
                continue
            d = {"p": A["p"][s], "min": A["min"][s], "max": A["max"][s]}
            pct = pct_of(d, v)
            # A degenerate tail is not an anomaly. ECAPE is zero on almost every
            # Antarctic sounding, so a zero there sits at the MEDIAN — but a naive
            # percentile calls it P0 and flags a "record low". Require the value to
            # be genuinely separated from the bulk: below p5 AND strictly below p25
            # (or above p95 AND strictly above p75).
            pp = A["p"][s]
            if k in FLOOR and v < FLOOR[k]:
                continue                                 # too small to be news
            if pct >= 95 and not (pp[6] > pp[4]):
                continue                                 # distribution is a spike
            low_ok = (pct <= 5 and v < pp[3]            # p[3] = 25th percentile
                      and k not in HIGH_ONLY)
            high_ok = pct >= 95 and v > pp[5]            # p[5] = 75th percentile
            if low_ok or high_ok:
                flags.append({"k": k, "lab": LABELS.get(k, k),
                              "v": round(v, 1), "pct": round(pct),
                              "sense": "high" if pct >= 95 else "low"})
        if flags:
            flags.sort(key=lambda f: abs(f["pct"] - 50), reverse=True)
            anomalies[wmo] = {"dt": e.get("dt", ""), "flags": flags}

    out.write_text(json.dumps(anomalies, separators=(",", ":")))
    print(f"  flagged {len(anomalies)} near-record stations", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
