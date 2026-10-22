#!/bin/bash
# Seed a runner's workspace with existing animation frames from the `frames`
# branch (from the frames store once the branch is retired), so incremental
# renderers append instead of rebuilding.
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
# ONLY THE ASKED-FOR DIRECTORIES TRAVEL (2026-10-16): a blobless clone of the branch's trees, then a sparse checkout of
# just these paths, which fetches their files in one batch. The full clone it replaces moved the whole branch (3.2 GB)
# to seed a single loop.
#
# Read-only and best-effort: a missing branch, or a directory not on it, is not
# an error - the renderer simply builds what it does not find, which is the old
# behaviour rather than a failure.
set -uo pipefail

[ $# -gt 0 ] || { echo "usage: $0 <frame-dir> [frame-dir ...]"; exit 2; }
BRANCH="${FRAMES_BRANCH:-frames}"

# FRAMES_MODE: see frames_env.sh. Seeds come from the branch for as long as it is published (dual too): it is the
# complete copy, while a store still being filled would hand a swapped directory a partial set, which its publisher
# would then write to both. Only once the branch is off does the store seed.
. "$(dirname "$0")/frames_env.sh"
case "$FRAMES_MODE" in
  store)
    "$FRAMES_PY" "$FRAMES_LIB/frames_store.py" seed "$@" \
      || echo "::warning::not everything could be seeded from the frames store - rendering what is missing"
    exit 0 ;;
  broken)
    echo "::warning::FRAMES_BRANCH=off but the frames store is not usable - rendering everything from scratch"
    exit 0 ;;
esac
REPO_URL="${FRAMES_URL:-https://github.com/${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}.git}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if ! git clone -q --depth 1 --filter=blob:none --no-checkout --single-branch --branch "$BRANCH" "$REPO_URL" "$TMP/f" 2>/dev/null; then
  echo "::warning::could not read the $BRANCH branch to seed from - rendering everything from scratch"
  exit 0
fi
pats=()
for d in "$@"; do pats+=("/${d%/}"); done
if ! git -C "$TMP/f" sparse-checkout set --no-cone -- "${pats[@]}" 2>/dev/null || ! git -C "$TMP/f" checkout -q "$BRANCH"; then
  echo "::warning::could not check out ${*} from $BRANCH - rendering what is missing from scratch"
  exit 0
fi

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
  seed_one "$TMP/f/$d" "$d"
done
