#!/bin/bash
# Seed a runner's workspace with existing animation frames from the `frames`
# branch, so incremental renderers append instead of rebuilding.
#
#     scripts/lib/fetch_frames_ci.sh assets/sst/anim/mur_ct ...
#
# WHY THIS EXISTS. Several renderers are incremental - mur_eqpac.py renders a
# day only `if not p.exists()`, and each frame it does render costs its own
# upstream fetch. That worked while the frames were tracked on main and arrived
# with the checkout. When 483 duplicated frame webp were taken off main on
# 2026-08-30 the runner started every job with an empty directory, so mur-sst
# tried to rebuild ~150 frames from April onward and was killed by its 25 min
# timeout. Same trap for any other product that appends.
#
# Read-only and best-effort: a missing branch, or a directory not on it, is not
# an error - the renderer simply builds what it does not find, which is the old
# behaviour rather than a failure.
set -uo pipefail

[ $# -gt 0 ] || { echo "usage: $0 <frame-dir> [frame-dir ...]"; exit 2; }
BRANCH="${FRAMES_BRANCH:-frames}"

. "$(dirname "$0")/frames_env.sh"
if frames_store_ready; then
  "$FRAMES_PY" "$FRAMES_LIB/frames_store.py" seed ${EXCLUDE:+--exclude "$EXCLUDE"} "$@"
  exit 0
fi
REPO_URL="https://github.com/${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}.git"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if ! git clone --depth 1 --branch "$BRANCH" --single-branch -q "$REPO_URL" "$TMP/f" 2>/dev/null; then
  echo "  no $BRANCH branch to seed from - rendering everything from scratch"
  exit 0
fi

# EXCLUDE names a CHILD directory not to seed - used by the ECAPE archive so a
# cycle this run has just rendered is never topped up with its own older frames
# from the branch (a leftover F45 from a previous attempt would otherwise be
# published carrying the new run's valid times).
EXCLUDE="${EXCLUDE:-}"

seed_one() {
  local src="$1" dst="$2"
  mkdir -p "$dst"
  # -n: never overwrite something this run has already produced.
  cp -rn "$src/." "$dst/" 2>/dev/null || true
  echo "  $dst: seeded $(find "$dst" -name '*.webp' -type f 2>/dev/null | wc -l | tr -d ' ') frames"
}

for d in "$@"; do
  if [ ! -d "$TMP/f/$d" ]; then
    echo "  $d: not on $BRANCH yet"
    continue
  fi
  if [ -n "$EXCLUDE" ]; then
    for child in "$TMP/f/$d"/*; do
      [ -d "$child" ] || continue
      name="$(basename "$child")"
      if [ "$name" = "$EXCLUDE" ]; then
        echo "  $d/$name: skipped (rendered by this run)"
        continue
      fi
      seed_one "$child" "$d/$name"
    done
  else
    seed_one "$TMP/f/$d" "$d"
  fi
done
