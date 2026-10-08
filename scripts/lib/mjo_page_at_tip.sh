#!/usr/bin/env bash
# Rebuild mjo.html on main's CURRENT tip inside a workflow commit loop (2026-10-08).
#
# Call after `git reset --hard origin/main` + restoring the run's snapshot (and after the frame pruning), before
# `git add`. mjo.html is generated (scripts/mjo/generate_page.py, which also stamps the site chrome), and the
# snapshot carries the copy this run generated from ITS checkout. If the generator, the chrome or the page itself
# changed on main while the run was going, restoring that copy would put the old page back (the geps.yml / sst.yml
# page-revert race). So: unchanged since checkout -> keep the snapshot's page (identical to a rebuild); changed ->
# rebuild here from the tip's code and the run's restored assets/mjo; rebuild failed -> keep main's page.
#
#   scripts/lib/mjo_page_at_tip.sh <checkout sha>
set -u
BASE="${1:?checkout sha}"
SRC="scripts/mjo/generate_page.py scripts/mjo/clean_section.py scripts/mjo/src/enso_rmm.py scripts/site/apply_chrome.py mjo.html"
# shellcheck disable=SC2086
if git diff --quiet "$BASE" origin/main -- $SRC; then
  exit 0
fi
if python scripts/mjo/generate_page.py; then
  echo "::notice::mjo.html rebuilt on main's tip (its generator, chrome or page changed during this run)"
else
  git checkout origin/main -- mjo.html
  echo "::warning::mjo.html rebuild on main's tip failed; kept main's page (the next run regenerates it)"
fi
exit 0
