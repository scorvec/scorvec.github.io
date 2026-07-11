#!/usr/bin/env python3
"""Per-station sounding climatology for the Skew-T Explorer.

For each IGRA station we download the full period-of-record file, compute a set
of indices for every historical sounding, and reduce them to monthly percentile
breakpoints + record extremes. The browser then fetches one small JSON per
station (climo/{gid}.json on the skewt-climo branch) and reports where the
live/archived sounding's values rank against that station's own history.

Indices: pwat 850t 700t 500t h500 thick fzl kidx tott (exact from levels) plus
  ecape ship (via the native SHARPlib helper CLIMO_CAPE_BIN, identical physics
  to the app's WASM).

    python scripts/skewt/build_climo.py OUTDIR [gid1 gid2 ...]   # subset
    python scripts/skewt/build_climo.py OUTDIR --all            # every station
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

IGRA = ("https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/"
        "access/data-por/")
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
CAPE_BIN = os.environ.get("CLIMO_CAPE_BIN", "/tmp/climo_cape")
G = 9.80665
HERE = Path(__file__).resolve().parent
STATIONS = HERE.parents[1] / "skewt" / "stations.json"


def _f(s):
    v = int(s)
    return np.nan if v <= -8888 else v


def sounding_indices(block: list[str]) -> dict | None:
    """block = the level lines of one sounding. Returns index dict or None."""
    P, T, D, H = [], [], [], []
    for L in block:
        if len(L) < 52:
            continue
        p = _f(L[9:15])
        gph = _f(L[16:21])
        tt = _f(L[22:27])
        dpdp = _f(L[34:39])
        if np.isnan(p) and not np.isnan(gph):        # pibal — skip for thermo climo
            continue
        if np.isnan(p) or p < 2000:
            continue
        P.append(p)
        H.append(gph)
        T.append(np.nan if np.isnan(tt) else tt / 10 + 273.15)
        D.append(np.nan if (np.isnan(tt) or np.isnan(dpdp)) else tt / 10 - dpdp / 10 + 273.15)
    if len(P) < 8:
        return None
    P, T, D, H = map(np.asarray, (P, T, D, H))
    o = np.argsort(-P)                                # sfc -> top
    P, T, D, H = P[o], T[o], D[o], H[o]

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

    # PWAT: (1/g) ∫ w dp  (mm). Many stations report dewpoint on only a fraction
    # of levels; integrating a shallow/sparse moisture profile yields absurd
    # values (0.5 mm for Argentina), so require real coverage or report nothing.
    pw = np.nan
    m = np.isfinite(P) & np.isfinite(D)
    moist_ok = (m.sum() >= 6 and P[m].min() <= 50000            # moisture to 500 hPa
                and np.isfinite(D[0])                            # real surface dewpoint
                and m[P >= 50000].mean() >= 0.6)                 # 60%+ of levels below 500
    if moist_ok:
        Pm, Dm = P[m], D[m]
        Tdc = Dm - 273.15
        e = 6.112 * np.exp(17.67 * Tdc / (Tdc + 243.5)) * 100.0    # Pa
        w = 0.622 * e / np.maximum(Pm - e, 1.0)
        dP = -np.diff(Pm)                             # positive going up
        pw = float(np.sum(0.5 * (w[:-1] + w[1:]) * dP) / G)

    t850, t700, t500 = at(T, 850), at(T, 700), at(T, 500)
    d850, d700 = at(D, 850), at(D, 700)
    h500, h1000 = at(H, 500), at(H, 1000)
    fzl = np.nan
    for i in range(1, len(P)):
        if np.isfinite(T[i]) and np.isfinite(T[i - 1]) and \
                T[i - 1] >= 273.15 > T[i]:
            f = (273.15 - T[i - 1]) / (T[i] - T[i - 1])
            fzl = float(H[i - 1] + f * (H[i] - H[i - 1]) - H[0]); break

    def c(k):
        return np.nan if np.isnan(k) else k - 273.15
    idx = {
        "pwat": pw,
        "850t": c(t850), "700t": c(t700), "500t": c(t500),
        "h500": h500,
        "thick": (h500 - h1000) if np.isfinite(h500) and np.isfinite(h1000) else np.nan,
        "fzl": fzl,
        "kidx": ((c(t850) - c(t500)) + c(d850) - (c(t700) - c(d700))) if moist_ok else np.nan,
        "tott": (c(t850) + c(d850) - 2 * c(t500)) if moist_ok else np.nan,
    }
    return {k: v for k, v in idx.items() if v is not None and np.isfinite(v)}


def build_station(gid: str, outdir: Path) -> bool:
    try:
        raw = urllib.request.urlopen(IGRA + f"{gid}-data.txt.zip", timeout=120).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        text = zf.read(zf.namelist()[0]).decode("utf-8", "ignore")
    except Exception as e:                            # noqa: BLE001
        print(f"  {gid}: fetch failed ({repr(e)[:50]})", flush=True)
        return False

    by_month: dict = {}                               # mm -> {index -> [(val, year)]}
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        L = lines[i]
        if L.startswith("#" + gid):
            year, month = int(L[13:17]), int(L[18:20])
            nlev = int(L[32:36])
            block = lines[i + 1:i + 1 + nlev]
            i += 1 + nlev
            vals = sounding_indices(block)
            if vals:
                mb = by_month.setdefault(f"{month:02d}", {})
                for k, v in vals.items():
                    mb.setdefault(k, []).append((v, year))
            continue
        i += 1

    # ECAPE + SHIP from the native SHARPlib helper (same physics as the app)
    if os.path.exists(CAPE_BIN):
        try:
            r = subprocess.run([CAPE_BIN, gid], input=text, capture_output=True,
                               text=True, timeout=300)
            for ln in r.stdout.splitlines():
                q = ln.split()
                if len(q) != 4:
                    continue
                yr, mm = int(q[0]), q[1]
                mb = by_month.setdefault(mm, {})
                for k, s in (("ecape", q[2]), ("ship", q[3])):
                    if s != "nan":
                        mb.setdefault(k, []).append((float(s), yr))
        except Exception as e:                            # noqa: BLE001
            print(f"  {gid}: cape helper failed ({repr(e)[:40]})", flush=True)

    out = {"gid": gid, "months": {}}
    total = 0
    for mm, inds in by_month.items():
        mo = {}
        for k, vy in inds.items():
            arr = np.array([v for v, _ in vy])
            yrs = [y for _, y in vy]
            total = max(total, len(arr))
            imin, imax = int(np.argmin(arr)), int(np.argmax(arr))
            mo[k] = {
                "n": len(arr),
                "p": [round(float(x), 1) for x in np.percentile(arr, PCTS)],
                "min": round(float(arr[imin]), 1), "minY": yrs[imin],
                "max": round(float(arr[imax]), 1), "maxY": yrs[imax],
            }
        mo["yr0"] = min(y for vy in inds.values() for _, y in vy)
        mo["yr1"] = max(y for vy in inds.values() for _, y in vy)
        out["months"][mm] = mo
    if not out["months"]:
        return False
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{gid}.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"  {gid}: {len(out['months'])} months, ~{total} soundings/mo peak", flush=True)
    return True


def main() -> int:
    outdir = Path(sys.argv[1])
    args = sys.argv[2:]
    if "--all" in args:
        stns = [s["gid"] for s in json.loads(STATIONS.read_text())["stations"]]
    else:
        stns = args
    ok = 0
    for gid in stns:
        fp = outdir / f"{gid}.json"
        if fp.exists():
            try:
                have = json.loads(fp.read_text())
                if any("ecape" in mo or "ship" in mo
                       for mo in have.get("months", {}).values()):
                    ok += 1; continue          # already has cape indices
            except Exception:                   # noqa: BLE001
                pass                            # rebuild on parse error
        if build_station(gid, outdir):
            ok += 1
    print(f"built {ok}/{len(stns)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
