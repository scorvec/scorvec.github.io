#!/bin/bash
# Publish animation frames to the orphan `frames` branch.
#
# WHY: the daily frame churn (GDPS loops, OISST panels, satellite loops, AIFS
# animators) rewrites tens of MB of webp per day. Committed to main that is
# ~6 GB/month of permanent history — the repo hit 11 GB and had to be collapsed
# 2026-08-28. Force-pushing a SINGLE PARENTLESS commit instead means the branch
# never accumulates history: each push orphans the previous tree and GitHub GCs
# it. Same pattern already proven by skewt-data and asos5-data.
#
# WHAT MOVES: only the frame webp (298 MB). The *_manifest.json files (276 KB
# total) STAY on main, so the animator still fetches them same-origin and needs
# no CORS; only <img> requests go cross-origin, and images never need CORS.
#
#     scripts/lib/publish_frames.sh            # publish if anything changed
#     scripts/lib/publish_frames.sh --force    # publish regardless
#     PRUNE="wave1_nh wave1_sh" scripts/lib/publish_frames.sh --force   # retire products
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BRANCH="frames"
# every animation frame dir on the site. cptec/brazil/sfs are smaller than
# sst but churn the same way, and keeping ONE branch for all of them means
# the viewers need a single frame root rather than a per-product mapping.
DIRS=(assets/sst/anim assets/cptec/anim assets/brazil/anim assets/sfs/anim assets/ecape/anim assets/geps/anim)
STAMP="$REPO/scripts/lib/.frames_published"
cd "$REPO" || exit 1

# Fingerprint the frame set so an unchanged cycle costs nothing. Names+sizes are
# enough: a rewritten frame with identical size AND name is not a thing here
# (the renderers stamp dates into the image), and hashing 300 MB every 15 min
# would cost more than the push it saves.
fingerprint() {
  for d in "${DIRS[@]}"; do
    [ -d "$d" ] && find "$d" -name '*.webp' -type f -exec stat -f '%N %z' {} + 2>/dev/null
  done | sort | shasum | cut -d' ' -f1
}

# Single-publisher lock. The launchd job fires every 15 min and a manual run
# can land on top of it; two simultaneous force-pushes race and GitHub rejects
# one with "cannot lock ref refs/heads/frames" (seen 2026-08-28). Serialise.
LOCK="$REPO/scripts/lib/.frames.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +20 2>/dev/null)" ]; then
    echo "  (stale frames lock >20 min — reclaiming)"; rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || exit 0
  else
    echo "frames publish already running — skipping"; exit 0
  fi
fi
trap 'rm -rf "$LOCK" 2>/dev/null' EXIT

FP="$(fingerprint)"
if [ "${1:-}" != "--force" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$FP" ]; then
  echo "frames unchanged — nothing to publish"; exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP" "$LOCK" 2>/dev/null' EXIT
n=0
for d in "${DIRS[@]}"; do
  [ -d "$d" ] || continue
  mkdir -p "$TMP/$d"
  # frames only — manifests stay on main
  ( cd "$d" && find . -name '*.webp' -type f -print0 \
      | while IFS= read -r -d '' f; do
          mkdir -p "$TMP/$d/$(dirname "$f")"; cp "$f" "$TMP/$d/$f"
        done )
  c=$(find "$TMP/$d" -name '*.webp' | wc -l | tr -d ' ')
  echo "  $d: $c frames"; n=$((n + c))
done
[ "$n" -eq 0 ] && { echo "no frames found — refusing to publish an empty branch"; exit 1; }

# Carry over every product the LAPTOP does not render.
#
# This branch is force-pushed as one parentless commit, so whatever is not in
# the tree we build is deleted. That was safe while every animation was made
# here, but the stratosphere loops (wave1_maps, vortex_winds) now render only
# in GitHub Actions. On 2026-08-30 CI published them at 05:24Z and this script
# force-pushed at 11:06Z with no such directories on disk, silently deleting 47
# frames - the manifests on main then pointed at 404s and the panels showed
# only their static image.
#
# So: anything present on the branch but absent locally is restored from the
# branch itself. Retiring a product is now EXPLICIT via --prune, rather than a
# side effect of it not happening to be on this disk.
PRUNE="${PRUNE:-}"
git fetch -q origin "$BRANCH" 2>/dev/null || true
if git rev-parse -q --verify "origin/$BRANCH" >/dev/null; then
  kept=0
  for d in "${DIRS[@]}"; do
    # every immediate subdirectory of an anim root on the branch
    for sub in $(git ls-tree --name-only "origin/$BRANCH" "$d/" 2>/dev/null); do
      name="$(basename "$sub")"
      [ -d "$sub" ] && continue                       # rendered locally: ours wins
      case " $PRUNE " in *" $name "*)
        echo "  $name: pruned (explicitly retired)"; continue ;;
      esac
      mkdir -p "$TMP/$(dirname "$sub")"
      git archive "origin/$BRANCH" "$sub" | tar -x -C "$TMP" 2>/dev/null || continue
      c=$(find "$TMP/$sub" -name '*.webp' 2>/dev/null | wc -l | tr -d ' ')
      [ "$c" -gt 0 ] && { echo "  $name: $c frames carried over (not rendered here)"; \
                          kept=$((kept + c)); n=$((n + c)); }
    done
  done
  [ "$kept" -gt 0 ] && echo "  carried over $kept frames from the existing branch"
fi

REMOTE="$(git config --get remote.origin.url)"
(
  cd "$TMP" || exit 1
  git init -q -b "$BRANCH"
  git add -A
  git -c user.name="Shawn Corvec" -c user.email="scorvec@outlook.com" \
      commit -q -m "animation frames $(date -u +%FT%H:%MZ)"
  git config http.postBuffer 524288000
  git config http.version HTTP/1.1
  git push -q --force "$REMOTE" "$BRANCH:$BRANCH"
) || { echo "frames push FAILED"; exit 1; }

echo "$FP" > "$STAMP"
echo "published $n frames to '$BRANCH'"
