#!/bin/bash
# Daily out-of-band refresh of the RMM 120-day low-frequency (ENSO) filter map.
#
# run_rmm.py is deliberately CDS-free and only READS data/reference/wind_map120.nc, so the
# Wheeler & Hendon low-frequency filter must be kept current by something else — this wrapper.
# It rebuilds the map from ARCO-ERA5 (anonymous GCS, no CDS, ~7-day latency) via
# src/refresh_map120.py and COMMITS the result, so the deployed RMM pipeline (and its Action
# fallback) always filter against a fresh interannual background instead of a stale one.
#
# Invoked by ~/Library/LaunchAgents/com.scorvec.mjo.map120.plist (daily 04:30, off-peak so the
# ARCO read doesn't starve the production AIFS S3 pulls — see the ARCO-bandwidth note).
set -uo pipefail

PY="${MJO_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
MJO="$(cd "$(dirname "$0")" && pwd)"          # scorvec.github.io/scripts/mjo
REPO="$(cd "$MJO/../.." && pwd)"              # scorvec.github.io
export PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
cd "$MJO" || exit 1

LOG="$MJO/map120.refresh.log"
exec >> "$LOG" 2>&1
echo "===================== $(date) refresh_map120 ====================="

# Fail-soft: any ARCO/fetch error or incomplete window leaves the existing map untouched.
"$PY" src/refresh_map120.py "$@" || { echo "refresh failed (fail-soft: map untouched)"; exit 0; }

cd "$REPO" || exit 1
source "$REPO/scripts/lib/gitlock.sh"
require_main || exit 0
git add scripts/mjo/data/reference/wind_map120.nc scripts/mjo/data/reference/arco_uwind_cache.nc
if git diff --staged --quiet; then echo "map unchanged; nothing to commit"; exit 0; fi

git_lock || { echo "git lock busy; leaving as a local commit for the next run"; exit 0; }
trap 'git_unlock' EXIT
git -c user.name="Shawn Corvec" -c user.email="scorvec@outlook.com" \
    commit -m "RMM: refresh 120-day low-frequency (ENSO) filter map (ARCO-ERA5)"
for i in 1 2 3 4 5; do
  if git pull --rebase --autostash -X theirs origin main && git push; then
    echo "pushed (attempt $i)"; git_unlock; exit 0
  fi
  git_rebase_rescue   # finish the rebase past frame-count conflicts, else abort clean
  echo "push attempt $i failed; retrying…"; sleep 5
done
echo "ERROR: could not push after 5 attempts."; git_unlock; exit 1
