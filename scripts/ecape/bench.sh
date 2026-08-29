#!/bin/bash
# Where is the ECAPE loop actually bound - the network or the cores?
#
# This exists because it was not possible to tell from ad-hoc measurements taken
# 2026-08-29: bandwidth samples ranged 27-41 MB/s depending on what else was
# downloading, and kernel timings ranged 474%-1595% CPU depending on whether WRF
# had the machine. The two branches cross at about 36 MB/s, which is exactly
# where that noise sat, so neither answer was established.
#
# Runs ONE forecast hour end to end, times each stage, and says which branch
# binds. It also reports machine load before and after, and refuses to present a
# clean verdict if the box was busy - a contended number is worse than no number
# because it looks authoritative.
#
#     scripts/ecape/bench.sh                  # newest extended cycle, F00
#     scripts/ecape/bench.sh --fxx 6
#     scripts/ecape/bench.sh --threads 8      # cap the kernel
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ECAPE_PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
SCRATCH="${ECAPE_SCRATCH:-${TMPDIR:-/tmp}/ecape}/bench"
FXX=0
THREADS=""
FRAMES=29                       # what a full 48 h loop renders, for extrapolation

while [ $# -gt 0 ]; do
  case "$1" in
    --fxx) FXX="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO" || exit 1
mkdir -p "$SCRATCH"
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
NTHREADS="${THREADS:-$CORES}"

load1() { uptime | sed -E 's/.*load averages?: *//' | awk '{print $1}' | tr -d ','; }
busy_procs() {
  ps -Ao pcpu,comm -r 2>/dev/null | awk 'NR>1 && $1>50 {print $2}' | head -5 | tr '\n' ' '
}

LOAD_BEFORE=$(load1)
BUSY_BEFORE=$(busy_procs)

echo "ECAPE bench — $CORES cores, kernel threads $NTHREADS"
echo "  load before: $LOAD_BEFORE   busy: ${BUSY_BEFORE:-none}"
echo

[ -x "$REPO/scripts/ecape/ecape_grid" ] || { echo "building kernel…"; "$REPO/scripts/ecape/build.sh" >/dev/null || exit 1; }
CYCLE="$($PY scripts/ecape/fetch_hrrr.py --print-cycle --extended-only --probe-fxx 48)" || exit 1
STEM="$SCRATCH/b"
rm -f "$STEM".* 2>/dev/null

# bash's `time` is portable across macOS and Linux and gives sub-second real/user/sys.
TIMEFORMAT='%R %U %S'

echo "cycle $CYCLE  F$(printf '%02d' "$FXX")"
T_FETCH=$( { time $PY scripts/ecape/fetch_hrrr.py --date "${CYCLE:0:8}" --hour "${CYCLE:8:2}" \
             --fxx "$FXX" --out "$STEM" --quiet >/dev/null 2>&1; } 2>&1 )
BYTES=$(stat -f%z "$STEM.f32" 2>/dev/null || stat -c%s "$STEM.f32" 2>/dev/null || echo 0)
# The download is ~18% of the decoded cube (GRIB2 is packed); measure it directly
# rather than assuming, by asking the index how much we requested.
DL_MB=$($PY - "$CYCLE" "$FXX" <<'EOF' 2>/dev/null || echo 0
import sys, importlib.util
spec = importlib.util.spec_from_file_location("fh", "scripts/ecape/fetch_hrrr.py")
fh = importlib.util.module_from_spec(spec); spec.loader.exec_module(fh)
cyc, fxx = sys.argv[1], int(sys.argv[2])
rows = fh.fetch_index(fh._url(cyc[:8], int(cyc[8:]), fxx=fxx))
rng, _ = fh.wanted_ranges(rows)
print(round(sum(e - o + 1 for o, e in rng if e) / 1e6, 1))
EOF
)

T_KERNEL=$( { time OMP_NUM_THREADS="$NTHREADS" ./scripts/ecape/ecape_grid "$STEM" >/dev/null 2>&1; } 2>&1 )
T_RENDER=$( { time $PY scripts/ecape/render_ecape.py "$STEM" --anim-root "$SCRATCH/a" >/dev/null 2>&1; } 2>&1 )

f_real=$(echo "$T_FETCH"  | awk '{print $1}')
k_real=$(echo "$T_KERNEL" | awk '{print $1}'); k_cpu=$(echo "$T_KERNEL" | awk '{print $2+$3}')
r_real=$(echo "$T_RENDER" | awk '{print $1}')

LOAD_AFTER=$(load1)
BUSY_AFTER=$(busy_procs)
rm -rf "$SCRATCH"

$PY - "$f_real" "$k_real" "$k_cpu" "$r_real" "$DL_MB" "$FRAMES" "$CORES" \
     "$LOAD_BEFORE" "$LOAD_AFTER" "$BUSY_BEFORE$BUSY_AFTER" <<'EOF'
import sys
f, k, kc, r, mb, frames, cores, lb, la, busy = sys.argv[1:]
f, k, kc, r, mb = float(f), float(k), float(kc), float(r), float(mb)
frames, cores, lb, la = int(frames), int(cores), float(lb), float(la)

bw   = mb / f if f > 0 else 0          # fetch stage includes decode; see note below
par  = kc / k if k > 0 else 0
comp = k + r
print()
print(f"  fetch + decode   {f:6.2f} s   ({mb:.0f} MB requested -> {bw:5.1f} MB/s incl. decode)")
print(f"  kernel           {k:6.2f} s   ({kc:.0f} s CPU -> {par:4.1f}x effective parallelism of {cores})")
print(f"  render           {r:6.2f} s")
print(f"  compute branch   {comp:6.2f} s   (kernel + render, overlapped against the next fetch)")
print()
bound = "FETCH" if f > comp else "COMPUTE"
per   = max(f, comp)
print(f"  BINDING: {bound}   per-frame {per:.1f} s -> {frames} frames ≈ {frames*per/60:.1f} min")
# The branches cross where fetch time equals compute time.
if comp > 0:
    print(f"  crossover: this box flips to compute-bound above {mb/comp:.0f} MB/s")

# A contended sample is worse than none: it looks authoritative and is not.
noisy = lb > cores * 0.6 or la > cores * 0.6 or busy.strip()
if noisy:
    print()
    print(f"  ** SUSPECT ** load {lb:.1f} -> {la:.1f} on {cores} cores"
          + (f", busy: {busy.strip()}" if busy.strip() else ""))
    print("     Both stages compete with whatever else is running. Re-run on a quiet")
    print("     box and connection before trusting the verdict above.")
else:
    print()
    print(f"  machine looked quiet (load {lb:.1f} -> {la:.1f} on {cores} cores) — verdict is usable.")
EOF
