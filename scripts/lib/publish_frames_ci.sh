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

URL="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
# THE RACE THIS CLOSES (2026-09-02): sst, mjo, aam, aifs-compare, strat, ecape,
# satellite and gdps all publish here and can overlap. Each clones the branch,
# swaps its own dirs in and force-pushes ONE parentless commit - so if two ran
# at once the second push silently threw away the first one's frames. The
# push below is --force-with-lease against the tip we cloned: if the branch
# moved meanwhile the push is refused, and we re-clone (now containing the
# other job's frames), re-swap and try again. Nothing is ever overwritten
# unseen.
ORIG_PWD="$PWD"
publish_once() {
cd "$ORIG_PWD" || return 1                 # a retry starts from the workspace again
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' RETURN
# --depth 1 IS LOAD-BEARING (measured 2026-09-28). The commit pushed below has no
# parent, and git negotiates a push by commit ancestry, so from a full repository
# it re-sends EVERY object on the branch (2.24 GB for a one-file change) - past
# GitHub's 2 GiB pack limit that fails outright, which is what stopped the laptop's
# GEPS publisher. From a SHALLOW clone git marks the boundary commit's whole tree as
# already present, so the same push is ~400 bytes. Never deepen or unshallow this.
if git clone --depth 1 --branch "$BRANCH" --single-branch -q "$URL" "$TMP/f"; then
  echo "  cloned $BRANCH ($(du -sh "$TMP/f" | cut -f1))"
  BASE=$(git -C "$TMP/f" rev-parse HEAD)
else
  echo "  no $BRANCH yet - creating it"
  mkdir -p "$TMP/f" && git -C "$TMP/f" init -q && git -C "$TMP/f" remote add origin "$URL"
  BASE=""
fi

# Two modes per argument:
#   dir    SWAP  - the branch copy is replaced by ours (a loop's frame set is
#                  regenerated whole; stale frames must go)
#   +dir   MERGE - our files are added to / overwrite the branch copy, nothing
#                  is deleted. For state that several jobs extend concurrently
#                  (the AIFS verification archive: the twice-daily run adds one
#                  cycle while a backfill adds a hundred). A swap from either
#                  side would silently delete the other's work.
for d in "$@"; do
  merge=0
  case "$d" in +*) merge=1; d="${d#+}" ;; esac
  c=$(count_frames "$d")
  if [ "$c" -eq 0 ]; then
    echo "  $(basename "$d"): nothing rendered, keeping the branch copy"
    continue
  fi
  if [ "$merge" -eq 1 ]; then
    mkdir -p "$TMP/f/$d"
    cp -r "$d/." "$TMP/f/$d/"
    echo "  $(basename "$d"): merged $c file(s) into the branch copy"
  else
    rm -rf "${TMP:?}/f/$d"
    mkdir -p "$TMP/f/$(dirname "$d")"
    cp -r "$d" "$TMP/f/$(dirname "$d")/"
  fi
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
git config user.email "26825570+scorvec@users.noreply.github.com"
git checkout -q --orphan fresh
git add -A
if git diff --cached --quiet; then echo "  no frame changes"; PUBLISHED_SHA="$BASE"; PUBLISHED_NEW=0; return 0; fi
MSG="animation frames"; [ "$BRANCH" = frames ] || MSG="$BRANCH update"
git commit -q -m "$MSG $(date -u +%Y-%m-%dT%H:%MZ)"
git config http.postBuffer 524288000
if [ -n "$BASE" ]; then
  git push -q --force-with-lease="$BRANCH:$BASE" origin "fresh:$BRANCH" || return 75
else
  git push -q --force origin "fresh:$BRANCH" || return 75
fi
echo "  pushed $(find assets -name '*.webp' | wc -l | tr -d ' ') frames total to $BRANCH"
PUBLISHED_SHA=$(git rev-parse HEAD); PUBLISHED_NEW=1
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
  echo "  $BRANCH moved under us (another job published) - re-cloning, attempt $((attempt + 1))/5"
  sleep $((attempt * 7))
done
echo "::error::could not publish to $BRANCH after 5 attempts (branch kept moving)"
exit 1
