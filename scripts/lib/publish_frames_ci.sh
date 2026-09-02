#!/bin/bash
# Publish animation frames to the orphan `frames` branch FROM A RUNNER.
#
#     GH_TOKEN=... scripts/lib/publish_frames_ci.sh assets/sst/anim/pacsat ...
#
# Takes frame directories as arguments and swaps ONLY those into the branch.
#
# WHY THIS EXISTS AS A SCRIPT. The clone-swap-reorphan dance was pasted inline
# into ecape.yml and strat.yml, and four more workflows (pacific-satellite,
# samerica-satellite, gdps-charts, mur-sst) needed it when their frames moved
# off main on 2026-08-30. Six copies of a routine whose failure mode is
# "silently delete every other product's frames" is not a thing to maintain.
#
# THE RULE THIS ENCODES: the branch is force-pushed as ONE PARENTLESS COMMIT,
# so whatever is not in the tree we push is deleted. A job must therefore clone
# what is already there and replace only its own subdirectories. Force-pushing
# just your own tree wipes everyone else's frames - which is exactly what
# happened to wave1_maps and vortex_winds earlier that day.
#
# An empty or missing directory is SKIPPED rather than swapped in: a product
# that failed to render must not blank its own frames on the branch.
#
# PRUNE="<path> ..." deletes paths from the branch, for retention.
set -euo pipefail

# No directories is valid when PRUNE is set: that is the GC's prune-only call.
if [ $# -eq 0 ] && [ -z "${PRUNE:-}" ]; then
  echo "usage: $0 <frame-dir> [frame-dir ...]   (or set PRUNE=... for prune-only)"
  exit 2
fi
: "${GH_TOKEN:?GH_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
BRANCH="${FRAMES_BRANCH:-frames}"

# Object store first. `sync` writes only the directories it is given, so there
# is no race to lose here - see frames_store.py. While the viewers still read
# the branch (FRAMES_BRANCH != off) the branch is published as well, so the two
# stay in step until the cutover commit flips assets/frames_root.js.
. "$(dirname "$0")/frames_env.sh"
if frames_store_ready; then
  [ $# -gt 0 ] && "$FRAMES_PY" "$FRAMES_LIB/frames_store.py" sync "$@"
  [ -n "${PRUNE:-}" ] && "$FRAMES_PY" "$FRAMES_LIB/frames_store.py" prune ${PRUNE}
  [ "$BRANCH" = "off" ] && exit 0
fi

# [ -d ] first: under `set -euo pipefail` a find over a missing directory exits
# non-zero, pipefail carries it through the pipe and set -e kills the script
# before it prints anything. That cost a silent exit-1 in strat.yml on
# 2026-08-30 - the step failed with zero output and a green-looking run.
count_frames() {
  [ -d "$1" ] || { echo 0; return 0; }
  # *.nc: the sst workflow round-trips its Copernicus stores through the
  # branch under scripts/sst/data/cmems; they count as content too.
  find "$1" -type f \( -name '*.webp' -o -name '*.nc' -o -name '*.npz' \) 2>/dev/null | wc -l | tr -d ' '
}

n=0
for d in "$@"; do
  c=$(count_frames "$d")
  echo "  $d: $c frames"
  n=$((n + c))
done
# Nothing to publish is normally a reason to stop - but not when the caller is
# here to PRUNE. The GC job passes no directories at all, and an early exit
# would have made it a no-op that reported success.
if [ "$n" -eq 0 ] && [ -z "${PRUNE:-}" ]; then
  echo "::warning::nothing rendered; leaving the frames branch untouched"
  exit 0
fi

URL="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
# THE RACE THIS CLOSES (2026-09-02): sst, mjo, aam, aifs-compare, strat, ecape,
# satellite and gdps all publish here and can overlap. Each clones the branch,
# swaps its own dirs in and force-pushes ONE parentless commit - so if two ran
# at once the second push silently threw away the first one's frames. The
# push below is --force-with-lease against the tip we cloned: if the branch
# moved meanwhile the push is refused, and we re-clone (now containing the
# other job's frames), re-swap and try again. Nothing is ever overwritten
# unseen.
ORIG_PWD="$PWD"
publish_once() {
cd "$ORIG_PWD" || return 1                 # a retry starts from the workspace again
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' RETURN
if git clone --depth 1 --branch "$BRANCH" --single-branch -q "$URL" "$TMP/f"; then
  echo "  cloned $BRANCH ($(du -sh "$TMP/f" | cut -f1))"
  BASE=$(git -C "$TMP/f" rev-parse HEAD)
else
  echo "  no $BRANCH yet - creating it"
  mkdir -p "$TMP/f" && git -C "$TMP/f" init -q && git -C "$TMP/f" remote add origin "$URL"
  BASE=""
fi

for d in "$@"; do
  c=$(count_frames "$d")
  if [ "$c" -eq 0 ]; then
    echo "  $(basename "$d"): nothing rendered, keeping the branch copy"
    continue
  fi
  rm -rf "${TMP:?}/f/$d"
  mkdir -p "$TMP/f/$(dirname "$d")"
  cp -r "$d" "$TMP/f/$(dirname "$d")/"
  # Manifests live on main so the page fetches them same-origin.
  find "$TMP/f/$d" -name '*_manifest.json' -delete 2>/dev/null || true
done

# PRUNE removes paths from the branch outright - retention, not publication.
# The swap loop above only ever REPLACES what it is given, so without this an
# archive would grow forever: nothing that stops being rendered is ever removed.
for d in ${PRUNE:-}; do
  if [ -e "$TMP/f/$d" ]; then
    rm -rf "${TMP:?}/f/$d"
    echo "  pruned $d from $BRANCH"
  fi
done

cd "$TMP/f"
git config user.name "Shawn Corvec"
git config user.email "shawncorvec@hotmail.com"
git checkout -q --orphan fresh
git add -A
if git diff --cached --quiet; then echo "  no frame changes"; return 0; fi
git commit -q -m "animation frames $(date -u +%Y-%m-%dT%H:%MZ)"
git config http.postBuffer 524288000
if [ -n "$BASE" ]; then
  git push -q --force-with-lease="$BRANCH:$BASE" origin "fresh:$BRANCH" || return 75
else
  git push -q --force origin "fresh:$BRANCH" || return 75
fi
echo "  pushed $(find assets -name '*.webp' | wc -l | tr -d ' ') frames total to $BRANCH"
return 0
}
for attempt in 1 2 3 4 5; do
  publish_once "$@" && exit 0
  rc=$?
  [ "$rc" -eq 75 ] || exit "$rc"
  echo "  $BRANCH moved under us (another job published) - re-cloning, attempt $((attempt + 1))/5"
  sleep $((attempt * 7))
done
echo "::error::could not publish to $BRANCH after 5 attempts (branch kept moving)"
exit 1
