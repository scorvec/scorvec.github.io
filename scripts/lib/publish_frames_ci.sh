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
# so whatever is not in the tree we push is deleted. A job must therefore start
# from what is already there and replace only its own subdirectories. Force-pushing
# just your own tree wipes everyone else's frames - which is exactly what
# happened to wave1_maps and vortex_winds earlier that day.
#
# An empty or missing directory is SKIPPED rather than swapped in: a product
# that failed to render must not blank its own frames on the branch.
#
# PRUNE="<path> ..." deletes paths from the branch, for retention.
#
# THE BRANCH NEVER TRAVELS (2026-10-16). Each attempt used to clone the whole branch (--depth 1, 3.2 GB by October) to
# swap a few directories in: 6-7 GB per run with the seed, minutes per clone, and a clone that slow lost the lease race
# to the other publishers ("branch kept moving") or died with a 500. Now, as scripts/models/publish_data.sh has done
# hourly since 2026-09-30, an attempt reads only the branch's TREES (a blobless shallow clone, a few MB), edits them in
# a scratch index (git rm --cached / git add from the workspace), writes the tree with --missing-ok (the other
# products' blobs are referenced, not present) and pushes a parentless commit with the same lease. GIT_NO_LAZY_FETCH
# makes any step that would need a missing blob fail instead of downloading the branch. Before the push, everything
# outside the published and pruned paths must be entry-for-entry the tree we read, and a "+dir" may lose no file.
#
# PINS. Each new commit is pushed together with refs/frames-pins/<sha>: the manifests stamped below name that commit
# (frames_sha), and an orphan commit no ref holds is garbage-collected by GitHub, after which every pinned URL 404s.
# gc.yml deletes the pins no manifest on main names any more.
set -euo pipefail

# No directories is valid when PRUNE is set: that is the GC's prune-only call.
if [ $# -eq 0 ] && [ -z "${PRUNE:-}" ]; then
  echo "usage: $0 <frame-dir> [frame-dir ...]   (or set PRUNE=... for prune-only)"
  exit 2
fi
: "${GH_TOKEN:?GH_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
BRANCH="${FRAMES_BRANCH:-frames}"

# Paths are repository paths: never absolute, never climbing out, never the root.
for d in "$@" ${PRUNE:-}; do
  p="${d#+}"
  case "$p" in "" | . | ./* | /* | ../* | */.. | */../* | *//*) echo "refusing path '$d'"; exit 2 ;; esac
done

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
  find "$1" -type f \( -name '*.webp' -o -name '*.nc' -o -name '*.npz' -o -name '*.json' -o -name '*.bin' \) 2>/dev/null | wc -l | tr -d ' '
}

n=0
for d in "$@"; do
  c=$(count_frames "${d#+}")
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

URL="${FRAMES_URL:-https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git}"
# THE RACE THIS CLOSES (2026-09-02): sst, mjo, aam, aifs-compare, strat, ecape,
# satellite and gdps all publish here and can overlap. Each reads the branch,
# swaps its own dirs in and force-pushes ONE parentless commit - so if two ran
# at once the second push silently threw away the first one's frames. The
# push below is --force-with-lease against the tip we read: if the branch
# moved meanwhile the push is refused, and we re-read (now containing the
# other job's frames), re-swap and try again. Nothing is ever overwritten
# unseen.
ORIG_PWD="$PWD"

# Lines (ls-tree output, path after any TAB) whose path is not $1 or under one of the paths in $1.
not_under() {
  awk -v T="$1" '
    BEGIN { n = split(T, t, " ") }
    { p = $0; sub(/^[^\t]*\t/, "", p); for (i = 1; i <= n; i++) if (p == t[i] || index(p, t[i] "/") == 1) next; print }'
}
# The tree's entries outside every published / pruned path ($TOUCHED), one "mode type oid<TAB>path" line each.
outside() { git -c core.quotePath=false ls-tree -r --full-tree "$1" | not_under "$TOUCHED"; }
listing() { git -c core.quotePath=false ls-tree -r --full-tree --name-only "$1" -- "$2"; }

publish_once() {
cd "$ORIG_PWD" || return 1                 # a retry starts from the workspace again
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' RETURN
# --depth 1 IS LOAD-BEARING (measured 2026-09-28). The commit pushed below has no
# parent, and git negotiates a push by commit ancestry, so from a full repository
# it re-sends EVERY object on the branch (2.24 GB for a one-file change) - past
# GitHub's 2 GiB pack limit that fails outright, which is what stopped the old
# GEPS publisher. From a SHALLOW clone git marks the boundary commit's whole tree as
# already present, so the same push is ~400 bytes. Never deepen or unshallow this.
if git clone -q --depth 1 --filter=blob:none --no-checkout --single-branch --branch "$BRANCH" "$URL" "$TMP/f"; then
  BASE=$(git -C "$TMP/f" rev-parse HEAD) || return 75
  echo "  read $BRANCH at ${BASE:0:10} ($(du -sh "$TMP/f" | cut -f1) of trees)"
else
  # A FAILED CLONE IS NOT A MISSING BRANCH (2026-10-08). mur-sst's clone died with "early EOF" mid-transfer, this branch
  # took it for "no branch yet", started an empty one and force-pushed it: every other product's frames (cansips, climate,
  # gefs, mjo, sfs, snowband, site, the scripts/ state) vanished from the branch. Create only when the remote says the
  # branch does not exist; anything else is a transient failure - retry, never push.
  rm -rf "$TMP/f"
  rc=0; out=$(git ls-remote --exit-code --heads "$URL" "$BRANCH" 2>&1) || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "::warning::clone of $BRANCH failed but the branch exists - retrying, not publishing"
    return 75
  elif [ "$rc" -eq 2 ] && [ -z "$out" ]; then          # 2 = the remote answered and has no such ref
    echo "  no $BRANCH yet - creating it"
    mkdir -p "$TMP/f" && git -C "$TMP/f" init -q && git -C "$TMP/f" remote add origin "$URL" || return 1
    BASE=""
  else
    echo "::warning::cannot reach the remote to check $BRANCH ($out) - retrying, not publishing"
    return 75
  fi
fi

cd "$TMP/f" || return 1
# Everything from here works on trees alone (this function runs inside an && list, where set -e is off, so every step
# is checked by hand).
local -x GIT_INDEX_FILE="$TMP/index" GIT_NO_LAZY_FETCH=1
if [ -n "$BASE" ]; then git read-tree HEAD || return 1; else git read-tree --empty || return 1; fi

# Two modes per argument:
#   dir    SWAP  - the branch copy is replaced by ours (a loop's frame set is
#                  regenerated whole; stale frames must go)
#   +dir   MERGE - our files are added to / overwrite the branch copy, nothing
#                  is deleted. For state that several jobs extend concurrently
#                  (the AIFS verification archive: the twice-daily run adds one
#                  cycle while a backfill adds a hundred). A swap from either
#                  side would silently delete the other's work.
local d c merge TOUCHED="" MERGED=""
for d in "$@"; do
  merge=0
  case "$d" in +*) merge=1; d="${d#+}" ;; esac
  d="${d%/}"
  c=$(count_frames "$ORIG_PWD/$d")
  if [ "$c" -eq 0 ]; then
    echo "  $(basename "$d"): nothing rendered, keeping the branch copy"
    continue
  fi
  TOUCHED="$TOUCHED $d"
  if [ "$merge" -eq 1 ]; then
    # --no-all: `git add <dir>` would otherwise also drop every branch file the workspace does not have
    git --work-tree="$ORIG_PWD" add -f --no-all -- "$d" || return 1
    MERGED="$MERGED $d"
    echo "  $(basename "$d"): merged $c file(s) into the branch copy"
  else
    git rm -r -q --cached --ignore-unmatch -- "$d" >/dev/null || return 1
    git --work-tree="$ORIG_PWD" add -f -- "$d" || return 1
  fi
  # Manifests live on main so the page fetches them same-origin.
  git rm -r -q --cached --ignore-unmatch -- ":(glob)$d/**/*_manifest.json" >/dev/null || return 1
done

# PRUNE removes paths from the branch outright - retention, not publication.
# The swap loop above only ever REPLACES what it is given, so without this an
# archive would grow forever: nothing that stops being rendered is ever removed.
for d in ${PRUNE:-}; do
  d="${d%/}"
  TOUCHED="$TOUCHED $d"
  if git ls-files --error-unmatch -- "$d" >/dev/null 2>&1; then
    git rm -r -q --cached -- "$d" >/dev/null || return 1
    echo "  pruned $d from $BRANCH"
  fi
done

local TREE; TREE=$(git write-tree --missing-ok) || return 1
if [ -n "$BASE" ] && [ "$TREE" = "$(git rev-parse "$BASE^{tree}")" ]; then
  echo "  no frame changes"; PUBLISHED_SHA="$BASE"; PUBLISHED_NEW=0
  pin "$BASE"
  return 0
fi
if [ -n "$BASE" ]; then
  if [ "$(outside "$BASE" | git hash-object --stdin)" != "$(outside "$TREE" | git hash-object --stdin)" ]; then
    echo "::error::the new $BRANCH tree differs outside the published paths - refusing to push"
    return 2
  fi
  for d in $MERGED; do
    if [ -n "$(comm -23 <(listing "$BASE" "$d" | grep -v '_manifest\.json$' | not_under "${PRUNE:-}" | sort) \
                        <(listing "$TREE" "$d" | sort))" ]; then
      echo "::error::merging $d would drop files already on $BRANCH - refusing to push"
      return 2
    fi
  done
fi

MSG="animation frames"; [ "$BRANCH" = frames ] || MSG="$BRANCH update"
local C; C=$(git -c user.name="Shawn Corvec" -c user.email="26825570+scorvec@users.noreply.github.com" \
             commit-tree "$TREE" -m "$MSG $(date -u +%Y-%m-%dT%H:%MZ)") || return 1
git config http.postBuffer 524288000
# pack.window=0 (tested on throwaway repos, 2026-10-16). The push must stay thin: in a shallow repository only a thin
# push marks the read tip's whole tree as already on the remote (without it git sends the entire branch, and here it
# would first have to download it). But a thin push also looks up the old blob at each replaced path as a delta base,
# and those blobs are not in this clone ("could not fetch ... from promisor remote"). A zero delta window keeps the
# thin push and asks for no bases; frames are compressed images that barely delta anyway.
if [ -z "$BASE" ]; then
  git -c pack.window=0 push -q --force origin "$C:refs/heads/$BRANCH" || return 75
elif [ "$BRANCH" != frames ]; then
  git -c pack.window=0 push -q --force-with-lease="refs/heads/$BRANCH:$BASE" origin "$C:refs/heads/$BRANCH" || return 75
elif ! git -c pack.window=0 push -q --atomic --force-with-lease="refs/heads/$BRANCH:$BASE" origin \
       "$C:refs/heads/$BRANCH" "$C:refs/frames-pins/$C"; then
  # Refused. If the branch is still where we read it, the lease was not the reason: publish without the pin rather
  # than not at all (gc.yml creates missing pins for the shas the manifests name).
  [ "$(git ls-remote "$URL" "refs/heads/$BRANCH" | cut -f1)" = "$BASE" ] || return 75
  echo "::warning::the push with refs/frames-pins/${C:0:10} was refused; publishing without the pin"
  git -c pack.window=0 push -q --force-with-lease="refs/heads/$BRANCH:$BASE" origin "$C:refs/heads/$BRANCH" || return 75
fi
echo "  pushed $(listing "$C" assets | grep -c '\.webp$') frames total to $BRANCH (${C:0:10})"
PUBLISHED_SHA="$C"; PUBLISHED_NEW=1
return 0
}

# Nothing new to push, but the manifests are about to be stamped with the tip we read: make sure a pin holds it (a tip
# written by another publisher may have none). Sends nothing when the remote already has the commit at a ref it
# advertises; if the tip has moved on and the push would need the commit's objects, GIT_NO_LAZY_FETCH makes it fail
# rather than fetch them, and gc.yml creates the pin instead.
pin() {
  [ "$BRANCH" = frames ] || return 0
  git ls-remote --exit-code "$URL" "refs/frames-pins/$1" >/dev/null 2>&1 && return 0
  timeout 120 git -c pack.window=0 push -q origin "$1:refs/frames-pins/$1" >/dev/null 2>&1 \
    || echo "  (no pin for ${1:0:10} yet; gc.yml will add it)"
  return 0
}

# Pre-warm the pinned jsDelivr mirror (2026-10-05). On a network that blocks raw.githubusercontent.com the viewers read
# cdn.jsdelivr.net/gh/...@<sha>/, and the first request for each file there is a cold fetch from GitHub (measured 1-14 s a
# frame against ~0.08 s once cached) - a 41-frame loop took minutes for whoever opened it first after a run. So the
# publisher makes that first request: every .webp it just swapped in (SWAP dirs only; "+dir" archives are not viewed as
# loops), 4 at a time, best-effort and capped at FRAMES_WARM_MAX_S (default 420 s); FRAMES_WARM=0 turns it off. One GET
# per new file per run, which is what the first viewer would have sent anyway.
warm_mirror() {
  local sha="$1"; shift
  [ "${FRAMES_WARM:-1}" = "1" ] && [ -n "$sha" ] && [ "${PUBLISHED_NEW:-0}" = "1" ] || return 0
  local d lst; lst=$(mktemp)
  for d in "$@"; do
    case "$d" in +*) continue ;; esac
    [ -d "$d" ] && find "$d" -type f -name '*.webp' >> "$lst"
  done
  local n; n=$(wc -l < "$lst" | tr -d ' ')
  [ "$n" -gt 0 ] || { rm -f "$lst"; return 0; }
  local t0=$SECONDS
  sed "s|^|https://cdn.jsdelivr.net/gh/${GITHUB_REPOSITORY}@${sha}/|" "$lst" |
    timeout "${FRAMES_WARM_MAX_S:-420}" xargs -P 4 -n 1 curl -s -o /dev/null --max-time 60 --retry 1 || true
  echo "  warmed the jsDelivr mirror: $n frame(s) at @${sha:0:10} in $((SECONDS - t0)) s"
  rm -f "$lst"
}

# Pin the jsDelivr mirror (2026-10-05). jsDelivr holds a BRANCH path (@frames) for up to 12 h and frame names are reused
# every run (F00.webp ...), while the manifests on main are fresh - so a network that can only reach the mirror showed
# the previous run's maps under the new run's time labels. A COMMIT path (@<sha>) is immutable and cached correctly, so
# every manifest next to / inside a directory just published gets "frames_sha": the commit holding exactly these frames
# (or a later one). Viewers build the mirror URL from it (assets/frames_root.js) and fall back to @frames without it.
# A manifest belongs to a published dir if it sits inside it, or next to it and names it as a JSON string. Workflows
# publish before their commit step, which snapshots the manifests, so the stamp rides along.
stamp_manifests() {
  local sha="$1"; shift
  [ -n "$sha" ] || return 0
  local d base parent m
  local list=()
  for d in "$@"; do
    d="${d#+}"; [ -d "$d" ] || continue
    base=$(basename "$d"); parent=$(dirname "$d")
    for m in "$d"/*_manifest.json "$d"/manifest.json "$parent"/*_manifest.json "$parent"/manifest.json; do
      [ -f "$m" ] || continue
      case "$m" in "$d"/*) list+=("$m") ;; *) grep -q "\"$base\"" "$m" && list+=("$m") ;; esac
    done
  done
  [ ${#list[@]} -gt 0 ] || return 0
  python3 - "$sha" "${list[@]}" <<'PY' || echo "::warning::could not stamp frames_sha into the manifests"
import json, sys
sha, files = sys.argv[1], sorted(set(sys.argv[2:]))
for f in files:
    try:
        m = json.load(open(f))
    except Exception as e:
        print(f"  {f}: not stamped ({e})"); continue
    if not isinstance(m, dict):
        continue
    m["frames_sha"] = sha
    json.dump(m, open(f, "w"), separators=(",", ":"))
    print(f"  {f}: frames_sha {sha[:10]}")
PY
}
PUBLISHED_SHA=""; PUBLISHED_NEW=0
for attempt in 1 2 3 4 5; do
  publish_once "$@" && { cd "$ORIG_PWD" && stamp_manifests "$PUBLISHED_SHA" "$@"; warm_mirror "$PUBLISHED_SHA" "$@" || true; exit 0; }
  rc=$?
  [ "$rc" -eq 75 ] || exit "$rc"
  echo "  $BRANCH moved under us (another job published) - re-reading, attempt $((attempt + 1))/5"
  sleep $((attempt * 7))
done
echo "::error::could not publish to $BRANCH after 5 attempts (branch kept moving)"
exit 1
