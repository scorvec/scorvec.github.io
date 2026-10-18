#!/usr/bin/env python3
"""Flag near-record soundings for the Skew-T map's "record watch".

For each just-mirrored latest sounding, compute the level indices and compare
them to that station's climatology (climo/{gid}.json) for the same day of year
and synoptic hour (00Z or 12Z). Any index below the 5th or above the 95th
percentile is flagged. Writes anomalies.json
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
from quantmet.degdays import doy366

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_climo import PCTS, PHYS as CLIMO_PHYS, synoptic_class

STATIONS = HERE.parents[1] / "skewt" / "stations.json"
# Only the indices worth interrupting someone for. PWAT and K-index fire on any
# humid day and would drown the signal; these four actually mark an unusual
# airmass or an unusual amount of instability.
WATCH = ("h500", "thick", "850t", "ecape")
# Variables checked ONLY for actual records (tie/beat the station's stored
# extreme) — they never produce P95 "near record" noise, but a record-humid
# column or record-warm 700 mb is exactly the kind of thing the record map
# should carry. All exist in the climo JSONs.
RECORD_ONLY = ("700t", "500t", "850td", "700td", "pwat", "850spd", "250spd")
# Some indices are only interesting at ONE end. A station with no convective
# energy is in its normal state — "record low ECAPE" is not news, it's just a
# quiet day, and it crowds out the genuinely unstable soundings. Heights,
# thicknesses and 850 mb temperatures matter at both ends (a record-cold airmass
# is as notable as a record-warm one).
HIGH_ONLY = {"ecape", "850spd", "250spd"}  # a record CALM is not news
# A value must also be big enough to mean anything. Where an index is almost
# always zero (SHIP in the Arctic; ECAPE over the poles), the percentile is
# computed against a spike at zero and any trace value scores "P99".
FLOOR = {"ecape": 100.0, "ship": 0.5}
LABELS = {
    "h500": "500mb hgt",
    "thick": "1000-500 thick",
    "850t": "850mb T",
    "ecape": "ECAPE",
    "700t": "700mb T",
    "500t": "500mb T",
    "850td": "850mb Td",
    "700td": "700mb Td",
    "pwat": "PWAT",
    "850spd": "850mb wind",
    "250spd": "250mb wind",
}
# Physical plausibility (m / °C / mm / m/s): outside these bounds is a data error
# anywhere on Earth, whatever the station's own envelope says - the climatology
# builder's bounds (the 3-tail-span station fence alone lets a corrupt height
# through: heights have fat summer tails), plus the two convective indices the
# builder does not fence. The WASM helper can emit a missing-value sentinel
# (~1e8) for ECAPE; without a bound it would be flagged as an all-time record.
PHYS = {**CLIMO_PHYS, "ecape": (0, 12000), "ship": (0, 15)}
G = 9.80665


def indices(P, T, D, H, W=None):
    """P (Pa), T/D (K), H (m), W (m/s), sfc→top. Index dict (°C / m / mm / m/s)."""
    P, T, D, H = map(np.asarray, (P, T, D, H))
    W = np.asarray(W) if W is not None else np.full_like(P, np.nan, dtype=float)

    def at(field, plevel):
        pv = plevel * 100.0
        # A level BELOW the surface does not exist: high stations (Concordia, 3233 m,
        # surface ~630 hPa) used to report their surface temperature as "850 hPa T"
        # (-61.5 C) and get flagged. Same 20 hPa tolerance as the browser's interpP.
        if pv >= P[0]:
            return field[0] if pv - P[0] < 2000.0 else np.nan
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
            fzl = float(H[i - 1] + f * (H[i] - H[i - 1]) - H[0])
            break

    def c(k):
        return np.nan if np.isnan(k) else k - 273.15

    # Thickness needs a REAL 1000 hPa surface: at elevated stations (Reno,
    # 1516 m) the interpolator falls back to the surface height and the
    # "thickness" is h500 minus station elevation — junk that once flagged as
    # an all-time record.
    thick = (h500 - h1000) if (np.isfinite(h500) and np.isfinite(h1000) and P[0] >= 99000) else np.nan
    d = {
        "pwat": pw,
        "850t": c(t850),
        "700t": c(t700),
        "500t": c(t500),
        "850td": c(d850),
        "700td": c(d700),
        "h500": h500,
        "thick": thick,
        "fzl": fzl,
        "kidx": (c(t850) - c(t500)) + c(d850) - (c(t700) - c(d700)),
        "tott": c(t850) + c(d850) - 2 * c(t500),
        "850spd": at(W, 850),
        "250spd": at(W, 250),
    }
    return {k: v for k, v in d.items() if np.isfinite(v)}


def parse_csv(text):
    P, T, D, H, W = [], [], [], [], []
    for ln in text.splitlines()[1:]:
        c = ln.split(",")
        if len(c) < 7:
            continue
        try:
            p = float(c[3])
            t = float(c[5])
            dp = float(c[6])
            h = float(c[4])
        except ValueError:
            continue
        if p < 20:
            continue
        try:
            w = float(c[12])
        except (ValueError, IndexError):
            w = np.nan
        P.append(p * 100)
        H.append(h)
        T.append(t + 273.15)
        D.append(dp + 273.15)
        W.append(w)
    return P, T, D, H, W


def pct_of(d, v):
    # The climo sanitizer nulls physically-impossible record min/max entries,
    # so either end may be None: fall back to p1/p99 — percentile
    # scoring still works, and the record-tier logic independently skips
    # slots whose record value is absent.
    lo = d["min"] if d["min"] is not None else d["p"][0]
    hi = d["max"] if d["max"] is not None else d["p"][-1]
    X = [lo] + d["p"] + [hi]
    Y = [0, *PCTS, 100]
    if v <= X[0]:
        return 0.0
    if v >= X[-1]:
        return 100.0
    # scan from the RIGHT so a tied plateau (p75==p90==p95) scores the highest
    # percentile it equals — the left scan under-scored plateau highs to P75
    for i in range(len(X) - 1, 0, -1):
        if v > X[i - 1]:
            f = (v - X[i - 1]) / ((X[i] - X[i - 1]) or 1)
            return Y[i - 1] + min(1.0, f) * (Y[i] - Y[i - 1])
    return 0.0


def climo_tables(climo: dict, dt: str):
    """(table for the sounding's synoptic hour, every hour's table) of a station climatology.

    Format-2 files hold one table per synoptic hour; a launch more than the builder's tolerance from 00Z
    and 12Z, or at an hour the station has no climatology for, gets no table. Older files pool every hour
    in "idx" and are used as they are."""
    hours = climo.get("hr")
    if hours is None:
        t = climo.get("idx") or {}
        return t, [t]
    try:
        cls = synoptic_class(int(dt[11:13]))
    except ValueError:
        cls = None
    return (hours.get(f"{cls:02d}") if cls is not None else None), list(hours.values())


def all_time(tables: list, k: str, sense: str):
    """(value, year) of the station's all-time extreme of index k across every hour's anchors, or None."""
    v, y = ("max", "maxY") if sense == "high" else ("min", "minY")
    marks = [(m, yr) for t in tables if k in t for m, yr in zip(t[k][v], t[k][y]) if m is not None]
    if not marks:
        return None
    return max(marks) if sense == "high" else min(marks)


def ecape_for(sdir: Path) -> dict:
    """ECAPE per station, from the SAME WebAssembly build the browser runs — so
    the live value and its climatological percentile share identical physics."""
    js = HERE / "ecape_latest.js"
    if not js.exists():
        return {}
    try:
        r = subprocess.run(["node", str(js), str(sdir)], capture_output=True, text=True, timeout=300, check=False)
    except Exception as e:  # noqa: BLE001
        print(f"  ecape helper unavailable ({repr(e)[:40]})", flush=True)
        return {}
    out = {}
    for ln in r.stdout.splitlines():
        q = ln.split()
        if len(q) == 2:
            try:
                v = float(q[1])
            except ValueError:
                continue
            # the WASM helper returns a ~1e8 missing-value sentinel when a profile is
            # too short for a parcel (truncated soundings ending near 650 hPa): that is
            # "no value", not an implausible one to be logged and skipped downstream
            if np.isfinite(v) and 0.0 <= v < 1.0e5:
                out[q[0]] = v
    print(f"  ECAPE computed for {len(out)} soundings", flush=True)
    return out


def main() -> int:
    sdir, cdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    stations = sorted(json.loads(STATIONS.read_text())["stations"], key=lambda s: s.get("y1", 0))
    wmo2gid = {s["id"]: s["gid"] for s in stations}  # a reused WMO index maps to its most recent station
    ecape = ecape_for(sdir)
    manifest = (
        json.loads((sdir.parent / "manifest.json").read_text())
        if (sdir.parent / "manifest.json").exists()
        else {"entries": {}}
    )
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
        except Exception:  # noqa: BLE001
            continue
        if wmo in ecape:
            idx["ecape"] = ecape[wmo]
        idx = {k: v for k, v in idx.items() if k in WATCH or k in RECORD_ONLY}
        # day-of-year climatology: nearest 5-day anchor to the sounding's date, same synoptic hour
        try:
            dt = datetime.strptime(e.get("dt", "")[:10], "%Y-%m-%d")
            doy = int(doy366([dt])[0]) + 1  # build_climo's 366-slot calendar: 1 March = 61
        except ValueError:
            continue
        anchors = climo.get("doy") or []
        table, tables = climo_tables(climo, e.get("dt") or "")
        if not anchors or not table:
            continue
        s = min(range(len(anchors)), key=lambda i: min(abs(anchors[i] - doy), 366 - abs(anchors[i] - doy)))
        flags = []
        for k, v in idx.items():
            A = table.get(k)
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
            if k in PHYS and not (PHYS[k][0] <= v <= PHYS[k][1]):
                print(f"    {wmo} {k}: {v:.1f} physically implausible — skipped", file=sys.stderr)
                continue
            if k in FLOOR and v < FLOOR[k]:
                continue  # too small to be news
            if pct >= 95 and not (pp[6] > pp[4]):
                continue  # distribution is a spike
            low_ok = (
                pct <= 5
                and v < pp[3]  # p[3] = 25th percentile
                and k not in HIGH_ONLY
            )
            high_ok = pct >= 95 and v > pp[5]  # p[5] = 75th percentile
            if low_ok or high_ok:
                # An actual record (ties or beats the stored extreme) is a
                # different claim from "P100": rounding used to promote P99.6
                # to "P100 high", which reads as a record when the value was
                # 15 m short of one. Non-records now cap at P99; records carry
                # the mark they broke. The all-time envelope is exact — every
                # sounding lands inside some ±10-day anchor window of its hour,
                # so the max over anchors and hours IS the station's all-time
                # extreme.
                # A "record" that beats the station's ALL-HISTORY extreme by
                # several tail-widths is almost certainly corrupt source data,
                # not weather (Reno published a 6224 m 500-hPa height — 230 m
                # over its all-time max). Same fence build_climo uses.
                if k not in FLOOR:
                    span = max(pp[8] - pp[4], 1.0)  # p99 − p50
                    mxa = A["max"][s]
                    mna = A["min"][s]
                    if (mxa is not None and v > mxa + 3.0 * span) or (mna is not None and v < mna - 3.0 * span):
                        print(
                            f"    {wmo} {k}: {v:.1f} beyond plausibility fence "
                            f"(env [{mna},{mxa}], span {span:.1f}) — skipped",
                            file=sys.stderr,
                        )
                        continue
                rec = None
                mx, mn = A["max"][s], A["min"][s]
                if high_ok and mx is not None and v >= mx:
                    rec = {"t": "high", "prev": mx, "y": A["maxY"][s]}
                    at = all_time(tables, k, "high")
                    if v >= at[0]:
                        rec.update(tier="all", prev=at[0], y=at[1])
                elif low_ok and mn is not None and v <= mn:
                    rec = {"t": "low", "prev": mn, "y": A["minY"][s]}
                    at = all_time(tables, k, "low")
                    if v <= at[0]:
                        rec.update(tier="all", prev=at[0], y=at[1])
                if k in RECORD_ONLY and not rec:
                    continue  # these only speak when a record falls
                f = {
                    "k": k,
                    "lab": LABELS.get(k, k),
                    "v": round(v, 1),
                    "pct": 100 if rec else min(99, round(pct)),
                    "sense": "high" if pct >= 95 else "low",
                }
                if rec:
                    f["rec"] = rec
                flags.append(f)
        if flags:
            flags.sort(key=lambda f: abs(f["pct"] - 50), reverse=True)
            anomalies[wmo] = {"dt": e.get("dt", ""), "flags": flags}

    out.write_text(json.dumps(anomalies, separators=(",", ":")))
    print(f"  flagged {len(anomalies)} near-record stations", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
