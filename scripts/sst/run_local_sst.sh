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

DAY_Q='import json,os;p="assets/sst/manifest.json";print(json.load(open(p)).get("sst_valid_day","") if os.path.exists(p) else "")'
PREV_DAY=$("$PY" -c "$DAY_Q" 2>/dev/null || echo "")
"$PY" scripts/sst/sst-roni.py || { echo "sst-roni failed"; exit 1; }
NEW_DAY=$("$PY" -c "$DAY_Q" 2>/dev/null || echo "")
"$PY" scripts/sst/tao_subsurface.py --days 120 \
  --out scripts/sst/data/tao_eq_recent.nc \
  --ascii scripts/sst/data/tao_eq_recent.ascii || echo "TAO failed; continuing"
"$PY" scripts/sst/sst_subsurface.py || echo "subsurface failed; continuing"
( cd scripts/sst && SST_SITE_ROOT="$REPO" "$PY" sst_ascat_winds.py ) || echo "ASCAT failed; continuing"

# Analog comparison charts (current vs 1997/2015/2023 El Niño) — rebuild ONCE PER
# CALENDAR DAY (or whenever OISST advances). The subsurface analog tracks TAO, which
# can advance independently of OISST (OISST has ~1–2 day PSL latency), so gating purely
# on OISST left the subsurface cross-section a day stale. A per-day stamp keeps the
# 4-hourly polls cheap (only the first poll of a new day rebuilds) while ensuring the
# analogs refresh every day. sst_events loads the ~3.8 GB cached analog-year files.
ANALOG_STAMP="$REPO/scripts/sst/data/.analog_built_day"
TODAY_UTC=$(date -u +%Y-%m-%d)
LAST_ANALOG=$(cat "$ANALOG_STAMP" 2>/dev/null || echo "")
if [ "$TODAY_UTC" != "$LAST_ANALOG" ] || { [ -n "$NEW_DAY" ] && [ "$NEW_DAY" != "$PREV_DAY" ]; }; then
  echo "rebuilding analog charts (day ${LAST_ANALOG:-none}→$TODAY_UTC; OISST ${PREV_DAY:-none}→${NEW_DAY:-none})"
  ok=1
  SST_SITE_ROOT="$REPO" "$PY" scripts/sst/sst_events.py || { echo "sst_events failed; continuing"; ok=0; }
  ( cd scripts/sst && SST_SITE_ROOT="$REPO" "$PY" sst_subsurface_events.py ) || { echo "subsurface analog failed; continuing"; ok=0; }
  [ "$ok" = 1 ] && echo "$TODAY_UTC" > "$ANALOG_STAMP"     # stamp only on full success → retry next poll otherwise
fi

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
