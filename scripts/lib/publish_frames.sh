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
BRANCH="${FRAMES_BRANCH:-frames}"               # override only for a scratch test (e.g. frames-test)
# every animation frame dir on the site. cptec/brazil/sfs are smaller than
# sst but churn the same way, and keeping ONE branch for all of them means
# the viewers need a single frame root rather than a per-product mapping.
# 2026-09-03: GEPS is the ONLY loop still rendered on the laptop (user rule:
# the laptop dispatches, it does not render). Everything else is CI-owned and
# carried over from the branch untouched. This used to list every anim dir,
# and at 00:43Z on 2026-09-03 it pushed the laptop's stale 19:03Z AIFS loops
# over the CI render because their mtime looked "fresh" - the exact clobber
# the carry-over rule was meant to prevent.
DIRS=(assets/geps/anim)
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

# Object store: every immediate product directory under the anim roots that
# exists on this disk, guarded so a directory CI renders (ECAPE, the strat
# loops) is never overwritten by a stale local leftover - the same 12 h rule the
# branch carry-over below applies. Per-directory writes, so nothing needs to
# be "carried over": what this machine does not render is simply not touched.
. "$REPO/scripts/lib/frames_env.sh"
if frames_store_ready; then
  subs=()
  for d in "${DIRS[@]}"; do
    for sub in "$d"/*/; do
      sub="${sub%/}"; [ -d "$sub" ] || continue
      [ -n "$(find "$sub" -name '*.webp' -type f -print -quit 2>/dev/null)" ] && subs+=("$sub")
    done
  done
  [ ${#subs[@]} -gt 0 ] && "$FRAMES_PY" "$REPO/scripts/lib/frames_store.py" sync --guard-stale "${subs[@]}"
  if [ -n "${PRUNE:-}" ]; then
    pp=(); for name in $PRUNE; do for d in "${DIRS[@]}"; do pp+=("$d/$name"); done; done
    "$FRAMES_PY" "$REPO/scripts/lib/frames_store.py" prune "${pp[@]}"
  fi
  [ "${FRAMES_BRANCH:-frames}" = "off" ] && { echo "published to the frames store"; exit 0; }
fi

[ "$BRANCH" = "off" ] && { echo "FRAMES_BRANCH=off but no frames store configured - nothing published"; exit 0; }

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
  # The branch is one orphan commit, so every file on it shares this date.
  # Local frames OLDER than it are stale leftovers, not a fresh render.
  branch_epoch=$(git log -1 --format=%ct "origin/$BRANCH" 2>/dev/null || echo 0)
  for d in "${DIRS[@]}"; do
    # every immediate subdirectory of an anim root on the branch
    for sub in $(git ls-tree --name-only "origin/$BRANCH" "$d/" 2>/dev/null); do
      name="$(basename "$sub")"
      # "Do I have this?" is not the same question as "did I just render it?".
      # ECAPE moved to Actions but its old frame directories stayed on this
      # disk; on 2026-08-30 that made this script push day-old ECAPE frames
      # over the runner's fresh ones, so the manifest advertised the 12Z cycle
      # while the images were from the previous afternoon. Local wins only if
      # it is genuinely NEWER than what the branch already carries.
      if [ -d "$sub" ]; then
        newest=$(find "$sub" -name '*.webp' -type f -exec stat -f '%m' {} + 2>/dev/null | sort -rn | head -1)
        newest=${newest:-0}
        # 12 h of grace, not a bare comparison. The branch's date moves every
        # time ANY product publishes, including from CI, so a strict test would
        # reject frames this machine had rendered perfectly well an hour before
        # some unrelated runner pushed. Everything here renders at least daily,
        # so "more than 12 h older than the branch" is staleness, not lag.
        if [ "$newest" -ge $(( branch_epoch - 43200 )) ]; then
          continue                                    # freshly rendered here: ours wins
        fi
        age_h=$(( (branch_epoch - newest) / 3600 ))
        echo "  $name: local copy is ${age_h}h stale - keeping the branch's"
      fi
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
  # Everything on the branch OUTSIDE this publisher's DIRS is CI-owned and
  # must survive untouched (the sst workflow's Copernicus stores, the AIFS
  # verification state, every CI-rendered loop). Extract the WHOLE branch tree
  # minus DIRS underneath what was staged above. The earlier version walked
  # top-level paths and called `assets` "owned" because assets/geps/anim is
  # under it - and on 2026-09-03 01:15Z that dropped every other loop from
  # the branch. Never reason about ownership above the DIRS entries themselves.
  git archive "origin/$BRANCH" | tar -x -C "$TMP" $(for d in "${DIRS[@]}"; do printf -- "--exclude=%s " "$d"; done) 2>/dev/null \
    && echo "  carried over everything outside ${DIRS[*]} from the branch" \
    || echo "  WARNING: could not carry over the branch tree"
fi

REMOTE="$(git config --get remote.origin.url)"
# HOW THE COMMIT IS MADE AND PUSHED (2026-09-28). This used to `git init` a fresh
# repository in $TMP and force-push from there. The branch is one PARENTLESS commit
# and git negotiates a push by commit ancestry, so from any full repository - fresh
# or not - the push re-sent every object on the branch: 2.24 GB for a one-file
# change. Past GitHub's 2 GiB pack limit every push failed ("pack exceeds maximum
# allowed size") and the GEPS loops stopped publishing for a day (~40 failures).
# The CI publishers never hit it only because they push from a --depth 1 clone:
# in a SHALLOW repository git treats the boundary commit's whole tree as present
# and sends only the new objects.
# So: (1) build the tree and the parentless commit in THIS repository's object
# store (a scratch index, $TMP as the work tree), where origin/$BRANCH was just
# fetched; (2) push it from a throwaway bare repository that borrows this object
# store (objects/info/alternates) and is marked shallow at origin/$BRANCH, with
# --force-with-lease against that same tip. Same bytes as a CI push (~400 bytes
# for a one-file change), an atomic lease, still one parentless commit, and no
# hooks in the throwaway repository - exactly as the old `git init` had none.
# DRY=1 builds and checks the commit and reports what would be sent, without pushing.
BASE=$(git rev-parse -q --verify "origin/$BRANCH^{commit}" 2>/dev/null || true)
(
  GD="$(git rev-parse --absolute-git-dir)"
  OBJ="$(git rev-parse --path-format=absolute --git-path objects)"   # the common store, also from a worktree
  export GIT_DIR="$GD" GIT_WORK_TREE="$TMP" GIT_INDEX_FILE="$TMP.index"
  cd "$TMP" || exit 1
  git -c gc.auto=0 add -A -f . || exit 1        # -f: this repo's ignore rules must not drop frames
  tree=$(git write-tree) || exit 1
  rm -f "$GIT_INDEX_FILE"
  unset GIT_WORK_TREE GIT_INDEX_FILE
  if [ -n "$BASE" ] && [ "$tree" = "$(git rev-parse "$BASE^{tree}")" ]; then
    echo "  tree identical to origin/$BRANCH - nothing to publish"; exit 5
  fi
  commit=$(git -c user.name="Shawn Corvec" -c user.email="26825570+scorvec@users.noreply.github.com" \
           commit-tree "$tree" -m "animation frames $(date -u +%FT%H:%MZ)") || exit 1

  # GUARD (2026-09-04). This is a FORCE push of a whole tree, so anything that
  # is stale in $TMP silently replaces the CI's copy. It has bitten twice: the
  # 09-03 wipe of every CI loop, and on 09-04 the equatorial frames reverted a
  # day (the page then showed Sep 1 under a Sep 2 manifest). Outside DIRS this
  # tree must be byte-identical to the branch we fetched; if it is not, the
  # branch moved under us - skip and let the next tick rebuild from a fresh
  # fetch. Never force-push a tree you did not just derive from the branch.
  # (The lease below then guarantees the branch is STILL that tip when we land.)
  own=$(IFS='|'; echo "${DIRS[*]}")
  ours=$(git ls-tree -r "$commit" --format='%(objectname) %(path)' \
         | grep -vE "^[0-9a-f]+ ($own)/" | sort)
  theirs=$(git ls-tree -r "origin/$BRANCH" --format='%(objectname) %(path)' \
           | grep -vE "^[0-9a-f]+ ($own)/" | sort)
  if [ "$ours" != "$theirs" ]; then
    echo "  GUARD: tree outside ${DIRS[*]} differs from origin/$BRANCH - not pushing"
    diff <(printf '%s\n' "$theirs") <(printf '%s\n' "$ours") | head -8
    exit 3
  fi
  if [ -n "$BASE" ]; then
    echo "  commit ${commit:0:9} (no parent): $(git diff --name-only "$BASE" "$commit" | wc -l | tr -d ' ') file(s) differ from origin/$BRANCH ${BASE:0:9}"
  fi
  [ "${DRY:-0}" = "1" ] && { echo "  DRY=1: not pushing"; exit 0; }

  PR="$TMP.push"
  rm -rf "$PR"
  unset GIT_DIR            # MUST go: with GIT_DIR exported, `git -C "$PR"` still pushes from THIS
                           # (full) repository and re-sends the whole branch (found in the scratch test)
  git init -q --bare "$PR" || exit 1
  echo "$OBJ" > "$PR/objects/info/alternates"
  if [ -n "$BASE" ]; then
    echo "$BASE" > "$PR/shallow"
    git -C "$PR" -c gc.auto=0 push -q --force-with-lease="refs/heads/$BRANCH:$BASE" \
        "$REMOTE" "$commit:refs/heads/$BRANCH"
  else
    git -C "$PR" -c gc.auto=0 push -q --force "$REMOTE" "$commit:refs/heads/$BRANCH"
  fi
  prc=$?
  rm -rf "$PR"
  [ "$prc" = "0" ] || exit 1
)
rc=$?
[ "$rc" = "3" ] && { echo "frames push skipped (branch moved); next tick retries"; exit 0; }
[ "$rc" = "5" ] && { [ "${DRY:-0}" = "1" ] || echo "$FP" > "$STAMP"; exit 0; }
[ "$rc" = "0" ] || { echo "frames push FAILED (a 'stale info' rejection means the branch moved: next tick retries)"; exit 1; }
[ "${DRY:-0}" = "1" ] && exit 0                    # a dry run leaves the stamp alone

echo "$FP" > "$STAMP"
echo "published $n frames to '$BRANCH'"
