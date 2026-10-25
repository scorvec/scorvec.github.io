#!/bin/bash
# Publish one model cycle's data to the orphan `frames` branch WITHOUT downloading the branch, and to the frames store
# once it is configured (scripts/lib/frames_store.py; see the store block below).
#
#     GH_TOKEN=... SRC=<workspace> PRUNE="<path> ..." scripts/models/publish_data.sh assets/models/data/hrrr/2026093012
#
# prints the new frames commit SHA on the last line of stdout (the page pins its jsDelivr fallback to it; an empty
# line once the branch is retired).
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
# A directory named "+<dir>" is added to and replaced in, never deleted from: a rebuild of a cycle a page may still
# name (its manifest lags the store), whose leftovers go with the cycle.
# PRUNE_OLDER="<dir>:<name> ...": also remove every child of <dir> whose name sorts before <name>, evaluated against the
# branch as it is at push time (models.yml / ecape.yml: assets/models/data/hrrr:<oldest kept cycle>). Cycles only ever
# move forward, so this is safe against a concurrent publisher, and it collects orphans no manifest remembers.
[ $# -gt 0 ] || [ -n "${PRUNE:-}" ] || [ -n "${PRUNE_OLDER:-}" ] || { echo "usage: $0 <dir> ... (or PRUNE=... / PRUNE_OLDER=...)" >&2; exit 2; }
for d in "$@"; do
  d="${d#+}"
  case "$d" in assets/models/*) ;; *) echo "refusing $d: this publisher only writes assets/models/" >&2; exit 2 ;; esac
  n=$(find "$SRC/$d" -type f 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] || { echo "refusing $d: nothing rendered" >&2; exit 2; }
done
for p in ${PRUNE:-}; do
  case "$p" in assets/models/data/*) ;; *) echo "refusing to prune $p" >&2; exit 2 ;; esac
done
for p in ${PRUNE_OLDER:-}; do
  case "${p%%:*}" in assets/models/data/*) ;; *) echo "refusing to prune under ${p%%:*}" >&2; exit 2 ;; esac
  [[ "${p##*:}" =~ ^[0-9]{10}$ ]] || { echo "refusing PRUNE_OLDER $p: the floor must be a YYYYMMDDHH cycle" >&2; exit 2; }
done

# The frames store (FRAMES_MODE: see scripts/lib/frames_env.sh) gets the same edit first: the directories replaced,
# PRUNE deleted, then every cycle directory older than its PRUNE_OLDER floor, listed in the store itself so orphans go
# too - and the old cycles only once the new one is up. While the branch is still published a store failure is a
# warning; once it is off the store is the only copy, nothing is pushed, and the SHA line is empty (the page then has
# no commit to pin, and models.py stamp records "").
. "$(dirname "$0")/../lib/frames_env.sh"
if [ "$FRAMES_MODE" = broken ]; then
  echo "::error::FRAMES_BRANCH=off but the frames store is not usable - nothing published" >&2
  exit 1
fi
store_publish() {
  local py="$FRAMES_PY" lib="$FRAMES_LIB/frames_store.py" dry=""
  [ "${DRY_RUN:-0}" != "1" ] || dry="--dry-run"
  if [ $# -gt 0 ]; then "$py" "$lib" sync $dry --root "$SRC" "$@" || return; fi
  if [ -n "${PRUNE:-}" ]; then "$py" "$lib" prune $dry ${PRUNE} || return; fi
  if [ -n "${PRUNE_OLDER:-}" ]; then "$py" "$lib" prune-cycles $dry ${PRUNE_OLDER} || return; fi
}
if [ "$FRAMES_MODE" != branch ]; then
  rc=0; store_publish "$@" >&2 || rc=$?
  if [ "$FRAMES_MODE" = store ]; then
    if [ "$rc" -eq 0 ]; then echo ""; fi
    exit "$rc"
  fi
  [ "$rc" -eq 0 ] || echo "::warning::the frames store was not updated (exit $rc); publishing to the $BRANCH branch alone" >&2
fi

publish_once() {
  local TMP; TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' RETURN
  git clone -q --depth 1 --filter=blob:none --no-checkout --single-branch --branch "$BRANCH" "$URL" "$TMP/f" >&2
  cd "$TMP/f"
  local BASE; BASE=$(git rev-parse HEAD)
  export GIT_INDEX_FILE="$TMP/index"
  git read-tree HEAD
  for p in ${PRUNE:-} "$@"; do [[ "$p" == +* ]] || git rm -r -q --cached --ignore-unmatch -- "$p" >&2; done
  local spec d n
  for spec in ${PRUNE_OLDER:-}; do
    d="${spec%%:*}"
    for n in $(git ls-tree --name-only "HEAD:$d" 2>/dev/null); do
      if [[ "$n" =~ ^[0-9]{10}$ && "$n" < "${spec##*:}" ]]; then echo "  prune (older than ${spec##*:}): $d/$n" >&2; git rm -r -q --cached -- "$d/$n" >&2; fi
    done
  done
  for d in "$@"; do git --work-tree="$SRC" add -f -- "${d#+}" >&2; done
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
  # pack.window=0: the thin push looks up the old blob at each replaced path as a delta base, and in this blobless
  # clone every one would be downloaded first; a zero delta window keeps the push thin and asks for none (the same
  # reasoning as publish_frames_ci.sh).
  git -c pack.window=0 push -q --force-with-lease="refs/heads/$BRANCH:$BASE" origin "$C:refs/heads/$BRANCH" >&2 || return 75
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
