#!/bin/bash
# Local SST/RONI build (primary path; the GitHub Action sst.yml is the fallback
# for when the laptop is off). Rebuilds the OISST anomaly maps + RONI + TAO
# subsurface + ASCAT winds, then commits & pushes only if something changed.
#
# Idempotent: commits only when `git diff` is non-empty, so running it repeatedly
# / alongside the Action is safe (whoever lands the new OISST day first wins; the
# other run no-ops). Invoked daily by ~/Library/LaunchAgents/com.scorvec.sst.plist.
#
# OISST/PSL files are CACHED and only re-downloaded when PSL publishes newer data
# (sst-roni HEAD-checks Last-Modified), so the 4-hourly polls don't re-pull the full
# ~240 MB annual file each time — just when a new day actually lands. The GitHub
# Action runs from a clean checkout, so it has no cache to reuse.
set -uo pipefail

PY="${SST_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export MPLBACKEND=Agg SST_SITE_ROOT="$REPO" \
       PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
cd "$REPO" || exit 1

LOG="$REPO/scripts/sst/run_local_sst.log"
exec >> "$LOG" 2>&1
echo "===================== $(date) ====================="

"$PY" scripts/sst/sst-roni.py || { echo "sst-roni failed"; exit 1; }
"$PY" scripts/sst/tao_subsurface.py --days 120 \
  --out scripts/sst/data/tao_eq_recent.nc \
  --ascii scripts/sst/data/tao_eq_recent.ascii || echo "TAO failed; continuing"
"$PY" scripts/sst/sst_subsurface.py || echo "subsurface failed; continuing"
( cd scripts/sst && SST_SITE_ROOT="$REPO" "$PY" sst_ascat_winds.py ) || echo "ASCAT failed; continuing"

git add sst.html assets/sst/
if git diff --staged --quiet; then echo "no changes to commit"; exit 0; fi
DAY=$("$PY" -c "import json; print(json.load(open('assets/sst/manifest.json'))['sst_valid_day'])" 2>/dev/null)
git -c user.name="Shawn Corvec" -c user.email="shawncorvec@hotmail.com" \
    commit -m "SST/RONI update: OISST ${DAY} (local)"
for i in 1 2 3 4 5; do
  if git pull --rebase --autostash -X theirs && git push; then echo "pushed (attempt $i)"; exit 0; fi
  echo "push attempt $i failed; retrying…"; sleep 5
done
echo "ERROR: could not push after 5 attempts."; exit 1
