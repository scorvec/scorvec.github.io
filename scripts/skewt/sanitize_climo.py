"""Post-hoc QC scrub of published climatology JSONs (skewt-climo branch).

The published climo was built before build_climo.py grew its physical-
plausibility QC (the P[0]>=990 hPa thickness gate, PHYS absolute bounds),
and the per-sounding cache needed for a cheap re-aggregation was lost —
a full rebuild means re-downloading ~35 GB from NCEI. Until that rebuild,
this script scrubs the worst of the published files in place:

  1. Drops the `thick` index ENTIRELY where the whole distribution is
     poisoned (elevated stations: 1000 hPa is underground, so "thickness"
     was h500 minus an extrapolated subterranean height — Albuquerque's
     July median was 4290 m). Detector: median of window p50s < 4950 m,
     comfortably below any real station's annual median.
  2. Nulls record min/max (+ their years) outside the PHYS absolute bounds
     shared with flag_anomalies.py — e.g. Green Bay's 4017 m 1961
     "thickness record".
  3. Nulls record min/max beyond 3x the window's own tail span past the
     window p1/p99 (same fence the live flagger applies, but window-local
     — the build's global fence was blind to seasonally-impossible values
     like a 4832 m thickness "record" in August).
  4. Thickness cross-check vs h500: thick = h500 - h1000 and h1000 sits
     within roughly [-60, +320] m, so a thick record below the window's
     h500 record minus 320 m (or above h500 max + 80 m) is impossible
     whatever the spans say. Catches transition-season corruption the
     span fence is too loose for (Green Bay May 2014: 4840 m).

Run against a checkout of the skewt-climo branch:
    python scripts/skewt/sanitize_climo.py <path-to>/climo
Prints per-rule counts; rewrites only files that changed.
"""
import json
import statistics
import sys
from pathlib import Path

PHYS = {"h500": (4600, 6100), "thick": (4700, 6100), "850t": (-60, 45),
        "700t": (-55, 35), "500t": (-60, 15), "850td": (-75, 35),
        "700td": (-75, 30), "pwat": (0, 135)}
# Window-span floors so ultra-tight (tropical) distributions don't get a
# near-zero fence that nulls genuine records.
FLOOR = {"h500": 40.0, "thick": 40.0, "pwat": 4.0}
FLOOR_DEFAULT = 2.0     # temperatures / dewpoints (deg C)
P1, P50, P99 = 0, 4, 8  # indices into the PCTS list [1,5,10,25,50,75,90,95,99]


def sanitize(d: dict) -> dict:
    counts = {"thick_dropped": 0, "phys": 0, "fence": 0, "xh500": 0}
    idx = d.get("idx", {})

    t = idx.get("thick")
    if t:
        p50s = [p[P50] for p in t["p"] if p and p[P50] is not None]
        if p50s and statistics.median(p50s) < 4950:
            del idx["thick"]
            counts["thick_dropped"] = 1

    h5 = idx.get("h500")
    for k, (lo, hi) in PHYS.items():
        a = idx.get(k)
        if not a:
            continue
        floor = FLOOR.get(k, FLOOR_DEFAULT)
        for i in range(len(a["min"])):
            p = a["p"][i]
            for side, other in (("min", "max"), ("max", "min")):
                v = a[side][i]
                if v is None:
                    continue
                kill = None
                if not (lo <= v <= hi):
                    kill = "phys"
                elif p and p[P1] is not None:
                    if side == "min":
                        span = max(p[P50] - p[P1], floor)
                        if v < p[P1] - 3.0 * span:
                            kill = "fence"
                    else:
                        span = max(p[P99] - p[P50], floor)
                        if v > p[P99] + 3.0 * span:
                            kill = "fence"
                if kill is None and k == "thick" and h5:
                    if side == "min" and h5["min"][i] is not None \
                            and v < h5["min"][i] - 320:
                        kill = "xh500"
                    elif side == "max" and h5["max"][i] is not None \
                            and v > h5["max"][i] + 80:
                        kill = "xh500"
                if kill:
                    a[side][i] = None
                    a[side + "Y"][i] = None
                    counts[kill] += 1
    return counts


def main() -> int:
    root = Path(sys.argv[1])
    files = sorted(root.glob("*.json"))
    tot = {"thick_dropped": 0, "phys": 0, "fence": 0, "xh500": 0}
    changed = 0
    for fp in files:
        d = json.loads(fp.read_text())
        before = json.dumps(d, separators=(",", ":"))
        c = sanitize(d)
        after = json.dumps(d, separators=(",", ":"))
        if after != before:
            fp.write_text(after)
            changed += 1
        for k in tot:
            tot[k] += c[k]
    print(f"{len(files)} files scanned, {changed} rewritten")
    print(f"  thick dropped whole-station: {tot['thick_dropped']}")
    print(f"  records nulled — PHYS: {tot['phys']}, window fence: "
          f"{tot['fence']}, thick-vs-h500: {tot['xh500']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
