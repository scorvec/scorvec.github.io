#!/bin/bash
# Local synoptic-maps build (primary path; the GitHub Action synoptic-maps.yml is
# the fallback for when the laptop is off). Renders the HRRR F00–F48 maps (5 vars ×
# 5 regions) for the latest extended cycle, then commits & pushes only if there's a
# new cycle whose wind/solar plant overlays are already present.
#
# Why local-primary: on CI the render is capped at 2 workers (7 GB RAM) behind a
# 90-min timeout and a queue, so maps land ~5–8 h after cycle init. This laptop has
# 18 cores, so render_maps runs all-cores and finishes in minutes.
#
# Idempotent: gated on (new cycle) AND (wind+solar capacity CSVs committed), and it
# compares against the already-rendered cycle in the national wind manifest — so it
# no-ops cheaply when there's nothing to do and is safe to poll every ~30 min and to
# run alongside the Action (whoever lands the cycle first wins; the other no-ops).
# Invoked by ~/Library/LaunchAgents/com.scorvec.synoptic.plist.
set -uo pipefail

PY="${SYNOPTIC_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/wx/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/scripts/lib/gitlock.sh"; trap git_unlock EXIT
export MPLBACKEND=Agg \
       PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
cd "$REPO" || exit 1

LOG="$REPO/scripts/synoptic/run_local_synoptic.log"
exec >> "$LOG" 2>&1
echo "===================== $(date) ====================="
require_main || exit 0

# Single-instance lock: two concurrent renders share ONE working tree, so a
# `git pull --rebase --autostash` in one run can stash/mix the other's half-written
# frames — leaving manifests ahead of images (viewer timestamps not matching the maps).
# mkdir is atomic; reclaim an orphaned lock (owner PID dead, or <1 min old before its PID
# is written).
LOCK="$REPO/scripts/synoptic/.run.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  owner="$(cat "$LOCK/pid" 2>/dev/null)"
  if { [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; } \
     || [ -n "$(find "$LOCK" -maxdepth 0 -mmin -1 2>/dev/null)" ]; then
    echo "another synoptic run in progress (pid ${owner:-?}) — skipping this fire"; exit 0
  fi
  echo "stale lock (owner pid ${owner:-none} not running) — taking over"
  rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || { echo "could not acquire lock"; exit 0; }
fi
echo $$ > "$LOCK/pid"
trap 'git_unlock; rm -rf "$LOCK" 2>/dev/null' EXIT   # release the git lock AND this run lock

# Pull first so we see the latest wind/solar overlay CSVs + rendered manifest that
# the upstream Actions may have just committed (avoids re-rendering what's done).
# Under the shared git lock so this pull can't run while another pipeline is committing.
git_lock || exit 0
if ! git pull --rebase --autostash -X theirs >/dev/null 2>&1; then
  git_unlock; echo "git pull failed (dirty tree or conflict?); skipping this poll."; exit 0
fi
git_unlock

# Latest 00/06/12/18Z extended cycle whose F48 is actually published on AWS (live
# Herbie probe). Renders as soon as the full f00..f48 set lands rather than on a
# fixed latency guess — earlier when AWS is fast, and never on a partial cycle.
# Exits non-zero when nothing is ready yet → no-op this poll.
ISO=$("$PY" scripts/synoptic/latest_available_cycle.py --format iso) \
  || { echo "no extended cycle with F48 on AWS yet — waiting for next poll."; exit 0; }
DATE="${ISO%T*}"; HH="${ISO#*T}"; HOUR="${HH%:*}"     # 2026-06-06 / 12
COMPACT="${DATE//-/}T${HOUR}Z"                        # 20260606T12Z

# Already rendered? (national wind manifest carries the rendered cycle_compact)
RENDERED=$("$PY" -c "import json,sys;\
print(json.load(open('assets/synoptic/wind/national/manifest.json')).get('cycle_compact',''))" 2>/dev/null || echo "")
ONLY_VARS=""
if [ "$RENDERED" = "$COMPACT" ]; then
  # HRRR is current — but RRFS publishes ~1-3 h behind HRRR, so the first render
  # of a cycle usually catches only its early hours. Backfill RRFS + diffs when
  # the RRFS frame set is still short of HRRR's; otherwise nothing to do.
  HN=$("$PY" -c "import json;print(len(json.load(open('assets/synoptic/wind/national/manifest.json'))['frames']))" 2>/dev/null || echo 0)
  RN=$("$PY" -c "import json;print(len(json.load(open('assets/synoptic/rrfs_wind/national/manifest.json'))['frames']))" 2>/dev/null || echo 0)
  if [ "${RN:-0}" -ge "${HN:-0}" ]; then
    echo "maps already current for $COMPACT (RRFS $RN/$HN) — nothing to do."; exit 0
  fi
  echo "HRRR current for $COMPACT but RRFS partial ($RN/$HN frames) — backfilling RRFS + diffs."
  ONLY_VARS="rrfs_wind,rrfs_t2m,rrfs_smoke,rrfs_solar,rrfs_reflectivity,rrfs_visibility,rrfs_ceiling,t2m_diff,precip_diff"
fi

# Plant overlays are STATIC (locations + capacity); render_maps sizes the rings off
# these canonical files and never off a per-cycle forecast. Gate only on the static
# files — both always exist and only change when updated by hand — so the synoptic
# render is NEVER blocked waiting on a per-cycle plant commit; it renders as soon as
# HRRR F48 is on AWS. (A missing file here would mean the static fleet CSV itself was
# removed, not a per-cycle race.)
WIND_CSV="assets/wind_forecast_data/capacity_plant.csv"
SOLAR_CSV="assets/solar_forecast_data/capacity_plant.csv"
if [ ! -f "$WIND_CSV" ] || [ ! -f "$SOLAR_CSV" ]; then
  echo "static plant overlay file(s) missing (wind:$([ -f "$WIND_CSV" ] && echo y || echo n) solar:$([ -f "$SOLAR_CSV" ] && echo y || echo n)) — skipping."
  exit 0
fi

echo "rendering synoptic maps for $COMPACT  ($DATE ${HOUR}Z) on $(sysctl -n hw.ncpu) cores …"

# Drop stale outputs (PNG→WebP migration, old 9-region layout) for parity with CI.
find assets/synoptic -name "*.png" -delete 2>/dev/null || true
for var in wind solar reflectivity visibility ceiling; do
  for old in caiso spp ercot miso pjm newengland bpa nyiso isone se; do
    rm -rf "assets/synoptic/$var/$old" 2>/dev/null || true
  done
done

( cd "$REPO/scripts/synoptic" && PYTHONPATH="$REPO/scripts/synoptic" \
    "$PY" render_maps.py "$DATE" "$HOUR" --workers 0 ${ONLY_VARS:+--variables "$ONLY_VARS"} ) \
  || { echo "render_maps failed; leaving maps untouched."; exit 1; }

# The heavy frame webp live on the force-pushed `synoptic-frames` ORPHAN branch (served via
# jsDelivr), NOT in main's history — this keeps the repo from ballooning (~350 MB/cycle before).
# Publish frames FIRST, from a throwaway index built as a fresh root commit (no history kept),
# then force-push. main's manifests are only advanced AFTER the frames they reference are on the
# branch, so the viewer never points at frames that don't exist yet.
git_lock || { echo "git lock busy; leaving for the next run"; exit 0; }
# Stage frames under a CYCLE-STAMPED dir at the branch root (same layout as the
# Actions fallback): new URLs each cycle, so jsDelivr edges never serve stale
# frames — the ?v= query-string busting never worked (jsDelivr ignores it).
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/$COMPACT"
for d in assets/synoptic/*/; do
  v="$(basename "$d")"
  [ -d "$d" ] || continue
  find "$d" -name 'F*.webp' | grep -q . || continue
  mkdir -p "$STAGE/$COMPACT/$v"
  cp -R "$d". "$STAGE/$COMPACT/$v/"
done
IDX="$(mktemp)"
( cd "$STAGE"
  GIT_INDEX_FILE="$IDX" git --git-dir="$REPO/.git" --work-tree="$STAGE" read-tree --empty
  GIT_INDEX_FILE="$IDX" git --git-dir="$REPO/.git" --work-tree="$STAGE" add -Af . )
FTREE="$(GIT_INDEX_FILE="$IDX" git write-tree)"; rm -f "$IDX"
FCOMMIT="$(git commit-tree "$FTREE" -m "synoptic frames ${COMPACT}")"  # parent-less → orphan, no history
git update-ref refs/heads/synoptic-frames "$FCOMMIT"
rm -rf "$STAGE"
pushed_frames=0
for i in 1 2 3; do
  if git push -f origin synoptic-frames; then pushed_frames=1; break; fi
  echo "frame push attempt $i failed; retrying…"; sleep 5
done
if [ "$pushed_frames" != 1 ]; then
  echo "ERROR: could not publish frames to synoptic-frames; leaving main at the prior cycle (consistent)."
  git_unlock; exit 1
fi

# Now advance main with the small bits only: manifests/metadata + viewer + charts stamp.
cp "$REPO/scripts/synoptic/viewer.html" "$REPO/assets/synoptic/viewer.html"
if [ -f "$REPO/charts.html" ]; then
  perl -0pi -e "s|(id=\"last-updated-synoptic\">Cycle )[^<]*|\${1}${COMPACT}|g" "$REPO/charts.html"
fi
git add -A assets/synoptic/        # frame webp gitignored → only manifests/json/viewer staged
git add charts.html 2>/dev/null || true
if git diff --staged --quiet; then echo "frames published; no manifest change to commit."; git_unlock; exit 0; fi
git -c user.name="Shawn Corvec" -c user.email="scorvec@outlook.com" \
    commit -m "synoptic maps: cycle ${COMPACT} (manifests; frames on synoptic-frames)"
for i in 1 2 3 4 5; do
  if git pull --rebase --autostash -X theirs && git push; then echo "pushed main (attempt $i)"; git_unlock; exit 0; fi
  echo "main push attempt $i failed; retrying…"; sleep 5
done
echo "ERROR: could not push main after 5 attempts."; git_unlock; exit 1
