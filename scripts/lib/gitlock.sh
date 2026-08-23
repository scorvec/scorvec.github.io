# Shared advisory lock for the site pipelines (mjo / sst / synoptic / ens), which all
# commit + push to the SAME repo. Concurrent `git pull --rebase --autostash` runs were
# autostashing and then reverting each other's freshly-rendered (uncommitted) files,
# losing whole cycles of products. Serialising the commit/push critical section with this
# mutex prevents two pipelines from doing git mutations at the same time.
#
# Usage (set REPO first):
#     source "$REPO/scripts/lib/gitlock.sh"
#     git_lock || exit 0          # blocks up to 10 min; stale lock (>15 min) is reclaimed
#     ... git add / commit / pull --rebase / push ...
#     git_unlock
# A `trap git_unlock EXIT` is recommended so the lock is always released.
: "${REPO:?REPO must be set before sourcing gitlock.sh}"
GITLOCK="$REPO/.gitlock"

git_lock() {
  local waited=0
  while ! mkdir "$GITLOCK" 2>/dev/null; do
    if [ -n "$(find "$GITLOCK" -maxdepth 0 -mmin +15 2>/dev/null)" ]; then
      echo "  (stale git lock >15 min — reclaiming)"; rm -rf "$GITLOCK"; continue
    fi
    sleep 3; waited=$((waited + 3))
    if [ "$waited" -ge 600 ]; then echo "  (git lock busy >10 min — skipping push this run)"; return 1; fi
  done
  echo "$(basename "${0:-?}") $$ $(date -u +%FT%TZ)" > "$GITLOCK/owner" 2>/dev/null
  return 0
}

git_unlock() { rm -rf "$GITLOCK" 2>/dev/null; }

# Call after a failed `git pull --rebase -X theirs`: finishes a rebase paused on
# conflicts that -X theirs cannot auto-resolve — modify/delete on generated
# animation frames whose count drifts between builds (F119 stranded the laptop
# pipeline for a whole day, twice). Every unmerged path resolves in favour of
# the commit being replayed (REBASE_HEAD): keep its version if it has one, else
# delete the path. Anything unexpected aborts, so the repo is never left
# mid-rebase; the commit stays local and the next cycle retries.
git_rebase_rescue() {
  local i f unmerged
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -d "$(git rev-parse --git-path rebase-merge)" ] || \
      [ -d "$(git rev-parse --git-path rebase-apply)" ] || return 0
    unmerged="$(git diff --name-only --diff-filter=U)"
    if [ -n "$unmerged" ]; then
      while IFS= read -r f; do
        if git cat-file -e "REBASE_HEAD:$f" 2>/dev/null; then
          git checkout REBASE_HEAD -- "$f"
        else
          git rm -q -- "$f" 2>/dev/null || true
        fi
      done <<< "$unmerged"
    fi
    GIT_EDITOR=true git rebase --continue >/dev/null 2>&1 || break
  done
  git rebase --abort 2>/dev/null || true
}

# Refuse to run when the checkout is NOT on `main`. These pipelines commit+push, so a
# scheduled render firing while the repo sits on a feature branch would strand (and
# pollute) a whole day of products on the wrong branch — which happened once. Call this
# right after the per-run log header: `require_main || exit 0`. The next fire resumes
# automatically once the checkout is back on main.
require_main() {
  local b
  b="$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  if [ "$b" != "main" ]; then
    echo "$(date '+%F %T'): BRANCH GUARD — repo on '$b', not 'main'; skipping run to avoid stranding updates (run: git checkout main)."
    return 1
  fi
  return 0
}
