#!/bin/bash
# Local driver for the gridded HRRR ECAPE loop.
#
# WHY LOCAL: the kernel is the whole cost, and it is core-bound. This laptop does
# all 1,905,141 columns in 14.7 s wall (199 s CPU across 18 threads); a free
# 2-core Actions runner needs ~96 s for the same frame, which turned a 29-frame
# 48 h loop into a ~52 min job. Here the fetch becomes the slow part instead, so
# the run is bounded by bandwidth rather than compute. The Actions workflow is
# kept dispatch-only as a fallback, not as the operational path.
#
#     scripts/ecape/run_local_ecape.sh                  # newest extended cycle
#     scripts/ecape/run_local_ecape.sh --cycle 2026082912
#     scripts/ecape/run_local_ecape.sh --fxx "0 3 6"    # short test loop
#     scripts/ecape/run_local_ecape.sh --no-push        # render only
#
# Frames go to the orphan `frames` branch via scripts/lib/publish_frames.sh
# (assets/ecape/anim is registered in its DIRS); only the manifest and the F00
# stills are committed to main.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ECAPE_PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
SCRATCH="${ECAPE_SCRATCH:-${TMPDIR:-/tmp}/ecape}"
ANIM="$REPO/assets/ecape/anim"
CYCLE=""
FXX=""
PUSH=1

while [ $# -gt 0 ]; do
  case "$1" in
    --cycle) CYCLE="$2"; shift 2 ;;
    --fxx)   FXX="$2"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO" || exit 1
mkdir -p "$SCRATCH"

# Single-runner lock: a cycle takes ~15 min and the launchd timer can fire on top
# of a manual run. Two concurrent runs would interleave frames from different
# cycles into one anim tree.
LOCK="$REPO/scripts/ecape/.run.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  owner="$(cat "$LOCK/pid" 2>/dev/null)"
  if { [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; } \
     || [ -n "$(find "$LOCK" -maxdepth 0 -mmin -1 2>/dev/null)" ]; then
    echo "another ecape run is active (pid ${owner:-?}) — exiting"; exit 0
  fi
  echo "  (stale lock — reclaiming)"; rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || exit 0
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK" 2>/dev/null' EXIT

# --- kernel ------------------------------------------------------------------
if [ ! -x "$REPO/scripts/ecape/ecape_grid" ]; then
  echo "building the SHARPlib kernel (~5 s) …"
  "$REPO/scripts/ecape/build.sh" || { echo "build failed"; exit 1; }
fi

# --- cycle -------------------------------------------------------------------
if [ -z "$CYCLE" ]; then
  # Probe F48: only 00/06/12/18Z run that far, and a cycle's tail publishes well
  # after its analysis, so this picks a run that is complete end to end.
  CYCLE="$($PY scripts/ecape/fetch_hrrr.py --print-cycle --extended-only --probe-fxx 48)" || exit 1
fi
[ -z "$FXX" ] && FXX="$(seq 0 1 18) $(seq 21 3 48)"
read -r -a HOURS <<< "$FXX"
echo "ECAPE  cycle $CYCLE  ${#HOURS[@]} forecast hours"

# A previous cycle's frames must go: build_manifest globs this tree, so a
# leftover F45 would be republished under the new cycle's valid time.
rm -rf "$ANIM"; mkdir -p "$ANIM"

fetch_one() {
  $PY scripts/ecape/fetch_hrrr.py --date "${CYCLE:0:8}" --hour "${CYCLE:8:2}" \
      --fxx "$1" --out "$SCRATCH/f$1" --quiet
}

# Fetch (~30 s) overlaps the kernel (~15 s here) — bandwidth is the binding
# constraint locally, so prefetching the next hour is what keeps cores busy.
t0=$(date +%s)
fetch_one "${HOURS[0]}" || { echo "fetch F${HOURS[0]} failed"; exit 1; }
for i in "${!HOURS[@]}"; do
  f="${HOURS[$i]}"; nxt="${HOURS[$((i+1))]:-}"
  BG=""
  if [ -n "$nxt" ]; then fetch_one "$nxt" & BG=$!; fi
  printf '  F%-3s ' "$f"
  if ./scripts/ecape/ecape_grid "$SCRATCH/f$f" >/dev/null 2>&1 \
     && $PY scripts/ecape/render_ecape.py "$SCRATCH/f$f" --anim-root "$ANIM" >/dev/null 2>&1; then
    printf 'ok\n'
  else
    # One bad hour must not sink the loop; the manifest is built from whatever
    # frames exist, so the animation simply skips it.
    printf 'FAILED (skipped)\n'
  fi
  rm -f "$SCRATCH/f$f".f32 "$SCRATCH/f${f}_ecape".f32 \
        "$SCRATCH/f$f".json "$SCRATCH/f${f}_ecape".json
  [ -n "$BG" ] && wait "$BG"
done
echo "  frames rendered in $(( $(date +%s) - t0 )) s"

$PY scripts/ecape/build_manifest.py "$ANIM" --cycle "$CYCLE" || exit 1
for fld in ecape_ml ecape_mu ratio_ml ratio_mu; do
  [ -f "$ANIM/$fld/F00.webp" ] && cp "$ANIM/$fld/F00.webp" "$REPO/assets/ecape/$fld.webp"
done

[ "$PUSH" = "0" ] && { echo "  --no-push: leaving results uncommitted"; exit 0; }

# --- publish -----------------------------------------------------------------
# Frames to the orphan branch (never main: 29 x 4 frames is ~29 MB a cycle, the
# churn that forced the 2026-08-28 history collapse).
"$REPO/scripts/lib/publish_frames.sh" || echo "  (frame publish reported a problem)"

source "$REPO/scripts/lib/gitlock.sh"
git_lock || exit 0
trap 'git_unlock; rm -rf "$LOCK" 2>/dev/null' EXIT
git add assets/ecape/anim/ecape_manifest.json assets/ecape/*.webp
if git diff --staged --quiet; then
  echo "  no changes to commit"; git_unlock; exit 0
fi
git commit -q -m "data update: $(date -u +%Y-%m-%dT%H:%MZ)"
for i in 1 2 3 4 5; do
  if git pull --rebase --autostash -X theirs origin main && git push; then
    echo "  pushed (attempt $i)"; git_unlock; exit 0
  fi
  git rebase --abort 2>/dev/null || true
  sleep 5
done
echo "  push failed after retries"; git_unlock; exit 1
