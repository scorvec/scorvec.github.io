#!/bin/bash
# Force-push the recon mirror tree as ONE parentless commit to the recon-data branch (recon.yml).
#   push_data.sh <tree> <previous-tip checkout>
# Seeding the push with the previous tip's objects means only new/changed blobs are uploaded.
set -euo pipefail
TREE="$1"; PREV="${2:-}"
[ -s "$TREE/manifest.json" ] || { echo "::warning::no manifest.json in $TREE - not pushing"; exit 0; }
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$TREE/manifest.json"   # never publish a broken manifest
cd "$TREE"
rm -rf .git
git init -q -b recon-data
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -q -m "Recon mirror: $(date -u +'%Y-%m-%d %H:%M UTC')"
URL="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
if [ -n "$PREV" ] && [ -d "$PREV/.git" ]; then
  git fetch -q --depth=1 "$PREV" HEAD 2>/dev/null || echo "::warning::could not seed the push from the previous tip - full push"
fi
for i in 1 2 3; do
  if git push -q -f "$URL" recon-data; then echo "recon-data pushed $(date -u +%FT%TZ)"; exit 0; fi
  echo "push attempt $i failed"; sleep $((i * 5))
done
echo "::error::recon-data push failed"; exit 1
