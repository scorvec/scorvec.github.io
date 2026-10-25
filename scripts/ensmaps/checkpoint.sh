#!/bin/bash
# A checkpoint of a run followed as it lands (ensmaps.py run --follow --checkpoint): the frames written so far go to
# the frames store and the manifest to main, as the workflow's last steps do for the whole run. The commit is built
# on origin/main with plumbing (a scratch index, commit-tree), so the job's working tree and HEAD stay as they are.
#
#   env: MODEL, CYC (the job's), ENSMAPS_LISTS, ENSMAPS_THROUGH (set by ensmaps.py)
set -euo pipefail
L="${ENSMAPS_LISTS:?}"
M="assets/models/ensmaps_${MODEL:?}.json"
SRC=. PRUNE="$(tr '\n' ' ' < "$L/prune.txt")" PRUNE_OLDER="$(cat "$L/floor.txt")" \
  scripts/models/publish_data.sh $(cat "$L/publish.txt") > /dev/null
blob=$(git hash-object -w "$M")
export GIT_AUTHOR_NAME="github-actions[bot]" GIT_AUTHOR_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
for i in 1 2 3 4 5; do
  if ! timeout 120 git fetch -q origin main; then echo "fetch $i failed"; sleep 5; continue; fi
  idx=$(mktemp)
  GIT_INDEX_FILE="$idx" git read-tree FETCH_HEAD
  GIT_INDEX_FILE="$idx" git update-index --add --cacheinfo "100644,$blob,$M"
  tree=$(GIT_INDEX_FILE="$idx" git write-tree)
  rm -f "$idx"
  if [ "$tree" = "$(git rev-parse 'FETCH_HEAD^{tree}')" ]; then echo "the manifest on main is this one"; exit 0; fi
  c=$(git commit-tree "$tree" -p FETCH_HEAD -m "ensmaps: $MODEL ${CYC:?} through ${ENSMAPS_THROUGH:?}")
  if timeout 180 git push -q origin "$c:refs/heads/main"; then echo "on main: $MODEL $CYC through $ENSMAPS_THROUGH"; exit 0; fi
  echo "push $i rejected; again"
  sleep 5
done
echo "ERROR: the checkpoint could not be pushed" >&2
exit 1
