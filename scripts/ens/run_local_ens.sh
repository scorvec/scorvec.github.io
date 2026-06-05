#!/usr/bin/env bash
# Build the ensemble 2 m-temperature products (AIFS-ENS + ECMWF IFS-ENS means):
# run-to-run change (2-panel) + 48-h run-to-run forecast trend (combined). Both are
# run-to-run, so they populate once 2–3 same-hour cycles have been archived.
#
#   scripts/ens/run_local_ens.sh [YYYYMMDD HH]    # default: latest stored cycle
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ENS_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
cd "$REPO"

if [ -n "${1:-}" ] && [ -n "${2:-}" ]; then
  DATE="$1"; RUN="$2"
else                                            # latest cycle present in the ECMWF store
  LATEST=$(ls -d scripts/ecmwf/cache/*z 2>/dev/null | sort | tail -1 | xargs -n1 basename)
  [ -z "$LATEST" ] && { echo "no stored cycle found; pass YYYYMMDD HH"; exit 1; }
  DATE="${LATEST%??z}"; RUN="${LATEST: -3:2}"
fi
echo "== Ensembles t2m build for ${DATE} ${RUN}Z =="

( cd scripts/ens && "$PY" src/run_temps.py --date "$DATE" --run "$RUN" \
      --out-root "$REPO/assets/ens" ) || { echo "run_temps failed"; exit 1; }

git add assets/ens ensembles.html
if git diff --staged --quiet; then echo "no changes to commit"; exit 0; fi
git -c user.name="Shawn Corvec" -c user.email="shawncorvec@hotmail.com" \
    commit -q -m "Ensembles t2m update: ${DATE} ${RUN}Z"
for i in 1 2 3 4 5; do
  if git push; then echo "pushed on attempt $i"; exit 0; fi
  git fetch origin main && { git rebase -X ours origin/main || git rebase --abort; }
  sleep 5
done
