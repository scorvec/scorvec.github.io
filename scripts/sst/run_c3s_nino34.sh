#!/bin/bash
# Local monthly build of the C3S multi-model ENSO forecast (ONI + RONI) →
# assets/sst/c3s_nino34.webp + scripts/sst/c3s_nino34_clim.csv, then commit & push
# only if something changed.
#
# Runs LOCALLY only (no GitHub Action): the CDS pulls are small but the lagged
# ensembles (NCEP/UKMO/BoM) peak at ~20 GB RAM to decode, which a CI runner can't
# give. Needs ~/.cdsapirc (Copernicus CDS personal access token).
#
# Cadence: the C3S centres publish each month's seasonal forecast across roughly
# the 6th–14th, so the launchd agent polls a few times a day and this script only
# acts on days 6–16. It runs at most once per day, and once all 7 models are in
# for the month it writes a .done stamp and no-ops the rest of the month. Forecast
# GRIBs are cached on disk and the hindcast climatology is cached in the committed
# CSV, so re-runs within a month are cheap (no re-download).
#
# Invoked by ~/Library/LaunchAgents/com.scorvec.c3s.plist (StartInterval poll).
set -uo pipefail

PY="${C3S_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export MPLBACKEND=Agg SST_SITE_ROOT="$REPO" \
       PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
cd "$REPO" || exit 1

LOG="$REPO/scripts/sst/run_c3s_nino34.log"
exec >> "$LOG" 2>&1
echo "===================== $(date) ====================="

STAMPDIR="$REPO/scripts/sst/data/c3s"; mkdir -p "$STAMPDIR"
ISSUE_GUESS=$(date +%Y%m)
DONE_STAMP="$STAMPDIR/.done_${ISSUE_GUESS}"
DAY_STAMP="$STAMPDIR/.ran_$(date +%Y%m%d)"

if [ "${1:-}" = "--poll" ]; then
  DOM=$((10#$(date +%d)))
  if [ "$DOM" -lt 6 ] || [ "$DOM" -gt 16 ]; then exit 0; fi   # outside the monthly release window
  [ -f "$DONE_STAMP" ] && exit 0                               # all 7 models already committed this month
  [ -f "$DAY_STAMP" ]  && exit 0                               # already ran today
fi
touch "$DAY_STAMP"
find "$STAMPDIR" -maxdepth 1 -name '.ran_*' -mtime +40 -delete 2>/dev/null  # tidy old day-stamps

source "$REPO/scripts/lib/gitlock.sh"
require_main || exit 0

# Single-instance lock (a run can take ~30 min; don't overlap the next poll).
LOCK="$STAMPDIR/.run.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  owner="$(cat "$LOCK/pid" 2>/dev/null)"
  if { [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; } \
     || [ -n "$(find "$LOCK" -maxdepth 0 -mmin -1 2>/dev/null)" ]; then
    echo "another c3s run in progress (pid ${owner:-?}) — skipping this fire"; exit 0
  fi
  echo "stale lock (owner pid ${owner:-none} not running) — taking over"
  rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || { echo "could not acquire lock"; exit 0; }
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK" 2>/dev/null' EXIT

OUT=$("$PY" scripts/sst/c3s_nino34.py 2>&1); echo "$OUT"
NMODELS=$(printf '%s\n' "$OUT" | sed -nE 's/.*\(([0-9]+) models\)/\1/p' | tail -1)

# append/refresh this issue in the forecast-evolution store (cached GRIBs; cheap)
"$PY" scripts/sst/c3s_evolution.py || echo "evolution store update failed; continuing"

# NOAA SFS beta Niño-3.4 feed + global anomaly maps + derived indices
# (NODD zarr; lands by the ~6th; indices need the maps' cached SST clim)
"$PY" scripts/sfs/sfs_nino.py || echo "SFS beta feed failed; continuing"
"$PY" scripts/sfs/sfs_maps.py || echo "SFS beta maps failed; continuing"
"$PY" scripts/sfs/sfs_indices.py || echo "SFS beta indices failed; continuing"
"$PY" scripts/sfs/sfs_daily.py || echo "SFS beta daily failed; continuing"
# PRIVATE strat gate product — writes only to gitignored paths
"$PY" scripts/strat/sfs_gate100.py || echo "SFS gate failed; continuing"

git add assets/sst/c3s_nino34.webp scripts/sst/c3s_nino34_clim.csv assets/sst/data/enso_forecast.json \
        assets/sst/data/c3s_evolution.json assets/sfs scripts/sfs/data
if git diff --staged --quiet; then
  echo "no changes to commit"
  [ "${NMODELS:-0}" -ge 7 ] 2>/dev/null && touch "$DONE_STAMP"
  exit 0
fi
source "$REPO/scripts/lib/gitlock.sh"
trap 'git_unlock; rm -rf "$LOCK" 2>/dev/null' EXIT
git_lock || { echo "git lock busy; leaving as a local commit for the next run"; exit 0; }
git -c user.name="Shawn Corvec" -c user.email="shawncorvec@hotmail.com" \
    commit -m "C3S multi-model Niño-3.4 (ONI+RONI): ${ISSUE_GUESS} refresh (${NMODELS:-?} models, local)"
for i in 1 2 3 4 5; do
  if git pull --rebase --autostash -X theirs && git push; then
    echo "pushed (attempt $i)"
    [ "${NMODELS:-0}" -ge 7 ] 2>/dev/null && touch "$DONE_STAMP"
    git_unlock; exit 0
  fi
  echo "push attempt $i failed; retrying…"; sleep 5
done
echo "ERROR: could not push after 5 attempts."; git_unlock; exit 1
