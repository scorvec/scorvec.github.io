#!/bin/bash
# Re-orphan a data branch (frames, verify-data) SERVER-SIDE. Sourced, not executed.
#
#     . scripts/lib/frames_reorphan.sh
#     frames_reorphan <commit-we-just-pushed> <branch> [owner/repo]
#
# WHY (2026-09-28). The branch is meant to be ONE parentless commit, so it never
# accumulates history. The publishers used to force-push a parentless commit
# directly - but git's push negotiation is COMMIT-based: a parentless commit
# shares no ancestry with the remote tip, so git sent EVERY object on the branch,
# including the ~2 GB the remote already held. Once the branch passed 2 GiB GitHub
# refused every pack ("pack exceeds maximum allowed size") and the GEPS loops
# stopped publishing for a day. Measured on the same one-file change: 2,238 MB as a
# parentless push, 497 bytes with parent = the remote tip.
#
# So the publishers now push a normal FAST-FORWARD commit (parent = the tip they
# built on; only the new objects travel) and then call this, which asks the GitHub
# API for a parentless commit with THE SAME TREE (the tree is already on the
# server, nothing is uploaded) and moves the branch to it. The old tip and its
# parent become unreachable and GitHub GCs them, exactly as with the old force-push.
#
# LEASE. The API has no --force-with-lease, so: read the ref; move it only if it
# still names our commit; read it again afterwards.
#   * The ref moved on before we looked: if the mover built ON our commit (its
#     parent chain reaches ours) it carries our files and will re-orphan itself -
#     done. Otherwise we cannot tell whether our files survived: return 75 and the
#     caller republishes from a fresh read (a republish that finds nothing to
#     change is a no-op).
#   * The small window between our read and our PATCH: a publisher that pushes a
#     fast-forward of our commit in that window has its tip overwritten by our
#     orphan. IT detects that (its own read finds our orphan, whose chain does not
#     reach its commit -> 75 -> republish), so nothing is lost for longer than one
#     retry. A publisher based on anything older is rejected at push time (not a
#     fast-forward), as before.
# Any API failure leaves the branch on our (parented) commit: every file is
# correct, the branch just carries one extra commit of history until the next
# successful re-orphan. That is a warning, never a reason to fail a publish.
#
# Returns 0 (re-orphaned, or handed over to a later publisher), 75 (republish),
# 1 (API failure; branch left on the parented commit, which is still correct).
#
# Needs `gh` authenticated: GH_TOKEN on a runner (contents: write), the keyring on
# the laptop. GH_BIN overrides the binary (the laptop's conda env shadows the real
# GitHub CLI with an unrelated `gh`; use /opt/homebrew/bin/gh). Never echoes tokens.

frames_reorphan() {
  local new="$1" br="$2" repo="${3:-${GITHUB_REPOSITORY:-scorvec/scorvec.github.io}}"
  local gh="${GH_BIN:-gh}"
  local tree msg cur orphan after sha i
  [ -n "$new" ] && [ -n "$br" ] || { echo "  frames_reorphan: usage <commit> <branch> [repo]"; return 1; }

  cur=$("$gh" api "repos/$repo/git/ref/heads/$br" -q .object.sha 2>/dev/null) \
    || { echo "::warning::re-orphan: could not read $br; branch left on ${new:0:9} (parented, correct)"; return 1; }
  if [ "$cur" != "$new" ]; then
    # Did the mover build on top of us? Follow first parents a few steps.
    sha="$cur"
    for i in 1 2 3 4 5 6 7 8; do
      sha=$("$gh" api "repos/$repo/git/commits/$sha" -q '.parents[0].sha // empty' 2>/dev/null) || break
      [ -n "$sha" ] || break
      if [ "$sha" = "$new" ]; then
        echo "  $br moved on to ${cur:0:9}, built on our ${new:0:9}: that publisher re-orphans"
        return 0
      fi
    done
    echo "  $br is at ${cur:0:9}, which does not descend from our ${new:0:9}: republishing"
    return 75
  fi

  # Same tree, same message; only the parent goes.
  tree=$("$gh" api "repos/$repo/git/commits/$new" -q .tree.sha 2>/dev/null) \
    && msg=$("$gh" api "repos/$repo/git/commits/$new" -q '.message | @json' 2>/dev/null) \
    && [ -n "$tree" ] && [ -n "$msg" ] \
    || { echo "::warning::re-orphan: could not read commit ${new:0:9}; branch left parented"; return 1; }
  orphan=$(printf '{"message":%s,"tree":"%s","parents":[],"author":{"name":"Shawn Corvec","email":"26825570+scorvec@users.noreply.github.com"}}' \
             "$msg" "$tree" \
           | "$gh" api -X POST "repos/$repo/git/commits" --input - -q .sha 2>/dev/null) \
    || { echo "::warning::re-orphan: could not create the parentless commit; branch left parented"; return 1; }

  # Last look before the move (the lease), then the move.
  cur=$("$gh" api "repos/$repo/git/ref/heads/$br" -q .object.sha 2>/dev/null) || cur=""
  if [ "$cur" != "$new" ]; then
    echo "  $br moved to ${cur:0:9} while re-orphaning; leaving it to that publisher"
    return 0
  fi
  printf '{"sha":"%s","force":true}' "$orphan" \
    | "$gh" api -X PATCH "repos/$repo/git/refs/heads/$br" --input - >/dev/null 2>&1 \
    || { echo "::warning::re-orphan: PATCH refused; branch left on ${new:0:9} (parented, correct)"; return 1; }
  after=$("$gh" api "repos/$repo/git/ref/heads/$br" -q .object.sha 2>/dev/null) || after=""
  if [ "$after" = "$orphan" ]; then
    echo "  re-orphaned $br: ${orphan:0:9} (tree ${tree:0:9}, no parent)"
  else
    echo "  re-orphaned $br to ${orphan:0:9}; it has since moved to ${after:0:9}"
  fi
  return 0
}
