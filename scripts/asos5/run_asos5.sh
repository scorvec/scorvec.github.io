#!/bin/bash
# Refresh the ASOS 5-min precision overlay and force-push it to the asos5-data
# branch. Moved off GitHub Actions 2026-08-28: the `*/15` cron was only honoured
# ~25% of the time (169 of 672 expected runs in a week) because GitHub sheds
# sub-hourly schedules under load, and no amount of cron tuning fixes that.
#
# Cadence barely matters. asos5.html fetches its data LIVE, client-side, from
# api.weather.gov, so the page is current with no server job in the loop; this
# feed is only the MADIS HFMETAR 0.1 degC precision bump on ~9 stations. Running
# it every 30 min from launchd is already far more than the page needs, and is
# more reliable than the 15-min Action ever was.
#
# Pushes a SINGLE PARENTLESS commit (same technique the Action used), so the
# branch never accumulates history and main is untouched — no Pages rebuild.
set -uo pipefail

PY="${ASOS5_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
cd "$REPO" || exit 1

"$PY" scripts/asos5/asos5_update.py || { echo "asos5_update failed"; exit 1; }

JS="data/asos5/asos5_latest.js"
JSON="data/asos5/asos5_latest.json"
for f in "$JS" "$JSON"; do
  [ -s "$REPO/$f" ] || { echo "missing/empty $f — not pushing"; exit 1; }
done

# Build the commit out of loose objects and push it straight to the branch ref.
# This never touches the index, HEAD or the working tree, so it cannot race the
# site pipelines' commit/push sections and needs no git lock.
BLOB_JS=$(git hash-object -w "$JS")
BLOB_JSON=$(git hash-object -w "$JSON")
TREE=$(printf '100644 blob %s\tasos5_latest.js\n100644 blob %s\tasos5_latest.json\n' \
       "$BLOB_JS" "$BLOB_JSON" | git mktree)
COMMIT=$(git commit-tree "$TREE" -m "asos5 data $(date -u +%Y-%m-%dT%H:%M:%SZ)")
if git push --force -q origin "$COMMIT:refs/heads/asos5-data"; then
  echo "asos5 overlay pushed $(date -u +%FT%TZ)"
else
  echo "asos5 push failed"; exit 1
fi
