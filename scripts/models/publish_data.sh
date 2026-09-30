#!/bin/bash
# Publish one model cycle's data to the orphan `frames` branch WITHOUT downloading the branch.
#
#     GH_TOKEN=... SRC=<workspace> PRUNE="<path> ..." scripts/models/publish_data.sh assets/models/data/hrrr/2026093012
#
# prints the new frames commit SHA on the last line of stdout (the page pins its jsDelivr fallback to it).
#
# WHY NOT publish_frames_ci.sh. That script clones the whole branch (--depth 1, ~1.4 GB of every product's frames) to
# swap a few directories in. This job runs every hour and only ever ADDS one new cycle directory and deletes old ones,
# so it works on the branch's TREES alone: a blobless shallow clone (--filter=blob:none: commits and trees, no file
# contents - a few MB), the tree edited in a scratch index (git rm --cached / git add), `write-tree --missing-ok`
# (the other products' blobs are referenced, not present) and a parentless commit pushed with --force-with-lease
# against the tip we read. From a shallow repo the push sends only the new objects (see publish_frames_ci.sh on
# --depth 1: the boundary commit's whole tree counts as present), so the branch never travels in either direction.
#
# The same rules as publish_frames_ci.sh: a lease, re-read and retry when another publisher moved the branch, never
# publish an empty directory, and a guard that everything OUTSIDE assets/models is byte-for-byte the tree we read -
# a bug here must not be able to delete another product's frames.
set -euo pipefail
: "${GH_TOKEN:?GH_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
BRANCH="${FRAMES_BRANCH:-frames}"
SRC="$(cd "${SRC:-.}" && pwd)"
URL="${FRAMES_URL:-https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git}"
[ $# -gt 0 ] || [ -n "${PRUNE:-}" ] || { echo "usage: $0 <dir> ... (or PRUNE=...)" >&2; exit 2; }
for d in "$@"; do
  case "$d" in assets/models/*) ;; *) echo "refusing $d: this publisher only writes assets/models/" >&2; exit 2 ;; esac
  n=$(find "$SRC/$d" -type f 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] || { echo "refusing $d: nothing rendered" >&2; exit 2; }
done
for p in ${PRUNE:-}; do
  case "$p" in assets/models/data/*) ;; *) echo "refusing to prune $p" >&2; exit 2 ;; esac
done

publish_once() {
  local TMP; TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' RETURN
  git clone -q --depth 1 --filter=blob:none --no-checkout --single-branch --branch "$BRANCH" "$URL" "$TMP/f" >&2
  cd "$TMP/f"
  local BASE; BASE=$(git rev-parse HEAD)
  export GIT_INDEX_FILE="$TMP/index"
  git read-tree HEAD
  for p in ${PRUNE:-} "$@"; do git rm -r -q --cached --ignore-unmatch -- "$p" >&2; done
  for d in "$@"; do git --work-tree="$SRC" add -f -- "$d" >&2; done
  local TREE; TREE=$(git write-tree --missing-ok)
  # guard: outside assets/models the new tree must equal the one we read
  if [ "$(git ls-tree HEAD -- . | grep -v $'\tassets$' | git hash-object --stdin)" != \
       "$(git ls-tree "$TREE" -- . | grep -v $'\tassets$' | git hash-object --stdin)" ] || \
     [ "$(git ls-tree HEAD:assets | grep -v $'\tmodels$' | git hash-object --stdin)" != \
       "$(git ls-tree "$TREE":assets | grep -v $'\tmodels$' | git hash-object --stdin)" ]; then
    echo "::error::the new frames tree differs outside assets/models - refusing to push" >&2
    return 2
  fi
  local C; C=$(git -c user.name="Shawn Corvec" -c user.email="26825570+scorvec@users.noreply.github.com" \
                commit-tree "$TREE" -m "animation frames $(date -u +%Y-%m-%dT%H:%MZ) (models)")
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "dry run: would push $C over $BASE" >&2; echo "$C"; return 0; fi
  git push -q --force-with-lease="refs/heads/$BRANCH:$BASE" origin "$C:refs/heads/$BRANCH" >&2 || return 75
  echo "$C"
}
for attempt in 1 2 3 4 5 6; do
  rc=0
  (publish_once "$@") && exit 0 || rc=$?
  [ "$rc" -eq 75 ] || exit "$rc"
  echo "  $BRANCH moved under us (another job published) - re-reading, attempt $((attempt + 1))/6" >&2
  sleep $((attempt * 7))
done
echo "::error::could not publish to $BRANCH after 6 attempts" >&2
exit 1
