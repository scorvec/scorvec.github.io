#!/usr/bin/env bash
# The verification database: every 00Z run's matched pairs, every model, kept for good in the monthly releases
# verify-db-YYYY-MM of this repository: against the surface stations (sfc_<init>.parquet), the radiosondes (raob_)
# and Stage IV (rain_), and the pairs' sums that windows of runs are scored from (stats_, stn_), all written by
# scripts/verify/mverify_scores.py.
#
#   verifydb.sh list                 every file the database holds, one name per line
#   verifydb.sh fetch KIND N DIR     the newest N runs' KIND_<init>.parquet files (N 'all': every run) into DIR
#   verifydb.sh upload DIR           DIR's files into their months' releases (made when missing)
#
# Needs gh with GH_TOKEN and GITHUB_REPOSITORY (as in Actions).
set -euo pipefail
R=${GITHUB_REPOSITORY:?}

tags() { gh release list -R "$R" --limit 1000 --json tagName -q '.[].tagName' | grep '^verify-db-' | sort || true; }

case "${1:-}" in
  list)
    for t in $(tags); do gh release view "$t" -R "$R" --json assets -q '.assets[].name'; done | sort
    ;;
  fetch)
    kind=${2:?KIND}; n=${3:?N}; dir=${4:?DIR}; mkdir -p "$dir"
    names=$("$0" list | grep -E "^${kind}_[0-9]{10}\.parquet$" || true)
    [ "$n" = all ] || names=$(tail -n "$n" <<< "$names")
    # one download per month's release, its files named
    sed -nE "s/^${kind}_([0-9]{4})([0-9]{2}).*/\1-\2 &/p" <<< "$names" | awk '
      { p[$1] = p[$1] " -p " $2 } END { for (m in p) print m p[m] }' | while read -r month pats; do
      # shellcheck disable=SC2086  # the -p arguments are meant to split
      gh release download "verify-db-$month" -R "$R" -D "$dir" --clobber $pats
    done
    echo "database: $(find "$dir" -name "${kind}_*.parquet" | wc -l) ${kind} files"
    ;;
  upload)
    dir=${2:?DIR}
    for f in "$dir"/*_*.parquet; do
      [ -e "$f" ] || continue
      init=$(basename "$f" .parquet); init=${init#*_}
      tag="verify-db-${init:0:4}-${init:4:2}"
      if ! gh release view "$tag" -R "$R" >/dev/null 2>&1; then
        # another job may make the same month's release at the same moment: a failed create is fine if it now exists
        gh release create "$tag" -R "$R" --latest=false --title "Verification database ${init:0:4}-${init:4:2}" \
          --notes "Every 00Z run of the month matched with the observations, every model (scripts/verify/mverify_scores.py, mverify), one row per pair (mverify.match.PAIR_COLUMNS, and kept: whether mverify's default cuts keep it): sfc_<init>.parquet against surface stations (GHCNh), raob_<init>.parquet against radiosondes (IGRA2), rain_<init>.parquet 24 h totals against Stage IV on the 0.5 degree grid. stats_ and stn_ hold the pairs' sums that windows of runs are scored from." \
          || gh release view "$tag" -R "$R" >/dev/null
      fi
      gh release upload "$tag" "$f" -R "$R" --clobber
      echo "database: $(basename "$f") in $tag"
    done
    ;;
  *)
    sed -n '2,11p' "$0"; exit 2
    ;;
esac
