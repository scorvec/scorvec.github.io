#!/bin/bash
# GEML vs GDPS loop suite — one invocation per cycle.
# LaunchAgent fires 02:30 and 14:30 local; geml_compare.py picks the newest
# expected cycle by UTC hour (00Z before ~17 UTC, else 12Z) and falls back to
# today's earlier cycle if Datamart is incomplete.
set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
PY=/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python
REPO=/Users/shawn/scorvec.github.io
cd "$REPO" || exit 1
source "$REPO/scripts/lib/gitlock.sh"
require_main || exit 0

"$PY" -u scripts/geml/geml_compare.py || { echo "geml suite failed"; exit 1; }

git_lock || { echo "git lock busy; skipping publish this run"; exit 0; }
trap git_unlock EXIT
git add assets/sst/anim/geml_z500 assets/sst/anim/geml_t2m \
        assets/sst/anim/geml_syn_na assets/sst/anim/geml_syn_eu \
        assets/sst/anim/geml_syn_eas \
        assets/sst/anim/geml_z500_manifest.json assets/sst/anim/geml_t2m_manifest.json \
        assets/sst/anim/geml_syn_na_manifest.json assets/sst/anim/geml_syn_eu_manifest.json \
        assets/sst/anim/geml_syn_eas_manifest.json
if [ "$(git diff --cached --name-only | grep -icE 'strat|telecon')" != "0" ]; then
  echo "PRIVATE FILES STAGED - aborting push"; git reset -q; exit 1
fi
git diff --cached --quiet && { echo "no changes"; exit 0; }
git commit -q -m "GEML vs GDPS loop suite: $(date -u +%Y%m%d) refresh (local)" && \
  git pull --rebase --autostash -q && git push -q && echo "pushed"
