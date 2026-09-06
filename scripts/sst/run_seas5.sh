#!/bin/bash
# SEAS5 seasonal outlook: fetch from the CDS, build, publish — on the laptop.
#
# The CDS credentials (~/.cdsapirc) stay on this machine by decision
# (2026-09-06), so unlike the rest of the site this product is rendered here and
# pushed here; the pre-push hook allows exactly assets/sst/seas5_*.webp and
# assets/sst/data/seas5_outlook.json. launchd (com.scorvec.seas5) fires this
# daily from the 5th to the 12th at 07:30 local:
#   - before the issue is on the CDS, seas5_outlook.py exits after one small
#     probe and nothing is committed;
#   - once built, the issue is stamped in the JSON and later firings are no-ops;
#   - hindcasts (per start month) are cached under scripts/sst/data/seas5/.
#
#     scripts/sst/run_seas5.sh              # this month's issue
#     scripts/sst/run_seas5.sh 202609       # a specific issue
#     FORCE=1 scripts/sst/run_seas5.sh      # rebuild even if already published
set -uo pipefail
PY=/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python
SITE="$HOME/scorvec.github.io"
cd "$SITE/scripts/sst" || exit 1
export MPLBACKEND=Agg
ISSUE="${1:-$(date -u +%Y%m)}"
JSON="$SITE/assets/sst/data/seas5_outlook.json"
LOG="data/seas5/run_${ISSUE}.log"
mkdir -p data/seas5
echo "===================== $(date) issue $ISSUE =====================" >> "$LOG"

if [ "${FORCE:-0}" != "1" ] && [ -f "$JSON" ] && grep -q "\"issue\":\"$ISSUE\"" "$JSON"; then
  echo "  $ISSUE already published; nothing to do" >> "$LOG"; exit 0
fi

"$PY" seas5_outlook.py fetch --issue "$ISSUE" --previous 3 >> "$LOG" 2>&1
if grep -q "is not on the CDS yet" "$LOG"; then
  echo "  $ISSUE not on the CDS yet; will try again tomorrow" >> "$LOG"; exit 0
fi
"$PY" seas5_outlook.py build --issue "$ISSUE" --previous 3 >> "$LOG" 2>&1 || { echo "  BUILD FAILED" >> "$LOG"; exit 1; }
"$PY" seas5_era5.py >> "$LOG" 2>&1 || echo "  ERA5 reference pull incomplete (normals/skill degrade gracefully)" >> "$LOG"   # cached; a no-op after the first run
"$PY" seas5_tele.py --issue "$ISSUE" >> "$LOG" 2>&1 || echo "  teleconnections FAILED" >> "$LOG"
"$PY" seas5_normals.py --issue "$ISSUE" >> "$LOG" 2>&1 || echo "  normals FAILED" >> "$LOG"
# population-weighted temperature distributions from the 6-hourly members (US, Brazil); best-effort
"$PY" seas5_popT.py all --issue "$ISSUE" >> "$LOG" 2>&1 || echo "  popT FAILED (main products still publish)" >> "$LOG"

# Publish: only the SEAS5 outputs. A concurrent CI data commit just shifts our base.
( cd "$SITE" && git add assets/sst/seas5_*.webp assets/sst/data/seas5_*.json \
  && { git diff --staged --quiet && echo "  no change to publish" && exit 0; \
       git -c user.name="Shawn Corvec" -c user.email="scorvec@outlook.com" \
           commit -q -m "data update: SEAS5 ${ISSUE:0:4}-${ISSUE:4:2} issue" \
       && git pull -q --rebase --autostash origin main && git push -q origin HEAD:main \
       && echo "  published $ISSUE"; } ) >> "$LOG" 2>&1 \
  || echo "  (publish FAILED — commit assets/sst/seas5_* by hand)" >> "$LOG"
tail -3 "$LOG"
