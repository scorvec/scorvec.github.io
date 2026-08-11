#!/bin/bash
# Daily 1 PM CT (14:00 ET) render+publish of the 12Z AIFS single vs ENS-control
# animator loops. The 00Z cycle is covered by the morning MJO chain; this gets
# the 12Z loops out ~5 h earlier than the evening chain would. The animator's
# flock makes overlap with any other invocation impossible.
set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
PY=/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python
REPO=/Users/shawn/scorvec.github.io
DATE=$(date -u +%Y%m%d)
cd "$REPO" || exit 1
"$PY" -u scripts/verify/aifs_compare_anim.py --date "$DATE" --time 12 || exit 1
git add assets/sst/anim/aifs_compare assets/sst/anim/aifs_z500 \
        assets/sst/anim/aifs_compare_manifest.json \
        assets/sst/anim/aifs_z500_manifest.json
if [ "$(git diff --cached --name-only | grep -icE 'strat|telecon')" != "0" ]; then
  echo "PRIVATE FILES STAGED - aborting push"; git reset -q; exit 1
fi
git commit -q -m "AIFS animator loops: ${DATE} 12Z" && \
  git pull --rebase --autostash -q && git push -q
