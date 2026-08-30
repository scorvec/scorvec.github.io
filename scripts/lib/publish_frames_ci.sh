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

# [ -d ] first: under `set -euo pipefail` a find over a missing directory exits
# non-zero, pipefail carries it through the pipe and set -e kills the script
# before it prints anything. That cost a silent exit-1 in strat.yml on
# 2026-08-30 - the step failed with zero output and a green-looking run.
count_frames() {
  [ -d "$1" ] || { echo 0; return 0; }
  find "$1" -name '*.webp' -type f 2>/dev/null | wc -l | tr -d ' '
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

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
URL="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
if git clone --depth 1 --branch "$BRANCH" --single-branch -q "$URL" "$TMP/f"; then
  echo "  cloned $BRANCH ($(du -sh "$TMP/f" | cut -f1))"
else
  echo "  no $BRANCH yet - creating it"
  mkdir -p "$TMP/f" && git -C "$TMP/f" init -q && git -C "$TMP/f" remote add origin "$URL"
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
if git diff --cached --quiet; then echo "  no frame changes"; exit 0; fi
git commit -q -m "animation frames $(date -u +%Y-%m-%dT%H:%MZ)"
git config http.postBuffer 524288000
git push --force -q origin "fresh:$BRANCH"
echo "  pushed $(find assets -name '*.webp' | wc -l | tr -d ' ') frames total to $BRANCH"
