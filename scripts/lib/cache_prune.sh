#!/bin/bash
# Delete superseded Actions cache entries: for each key PREFIX, keep the newest entry on this ref and delete the rest.
#
#     GH_TOKEN=... scripts/lib/cache_prune.sh oisst-data- ascat-data-
#
# The data caches are keyed on their contents (<prefix><hash of the files>) and saved only when the files changed, so
# the newest entry is the current state and every older one is dead weight. Without this they stayed until GitHub
# evicted them, and the repository sat at its 10 GB cap: per-run keys left three 1.4 GB OISST copies and six 174 MB GEFS
# ones, and the eviction that made room took live entries (geps-ref-v1) with them.
#
# Prefixes must be complete (end in "-") so one cannot match another product's keys. Needs `actions: write`. Best
# effort: a failed listing or delete is a warning, never a failed run.
set -uo pipefail
: "${GH_TOKEN:?GH_TOKEN is required}"
REF="${GITHUB_REF:-refs/heads/main}"
for p in "$@"; do
  case "$p" in *-) ;; *) echo "::warning::cache prefix '$p' does not end in '-'; skipped"; continue ;; esac
  if ! ids=$(gh cache list --ref "$REF" --key "$p" --sort created_at --order desc --limit 100 \
               --json id,key -q ".[] | select(.key | startswith(\"$p\")) | .id" 2>&1); then
    echo "::warning::could not list the $p caches: $ids"
    continue
  fi
  n=0; first=1
  for id in $ids; do
    if [ "$first" = 1 ]; then first=0; continue; fi      # the newest stays
    if gh cache delete "$id" >/dev/null 2>&1; then n=$((n + 1)); else echo "::warning::could not delete cache $id ($p)"; fi
  done
  echo "  $p: kept the newest, deleted $n superseded"
done
exit 0
