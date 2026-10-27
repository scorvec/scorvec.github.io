#!/usr/bin/env bash
# The verification database: every run's matched pairs (surface stations sfc_<init>.parquet, radiosondes
# raob_<init>.parquet, every model), kept for good in the monthly releases verify-db-YYYY-MM of this repository.
#
#   verifydb.sh list                 every file the database holds, one name per line
#   verifydb.sh fetch N DIR          the newest N runs' files into DIR
#   verifydb.sh upload DIR           DIR's sfc_/raob_ files into their months' releases (made when missing)
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
    n=${2:?N}; dir=${3:?DIR}; mkdir -p "$dir"
    # the newest N runs that have both files
    "$0" list | sed -nE 's/^(sfc|raob)_([0-9]{10})\.parquet$/\2/p' | sort | uniq -d | tail -n "$n" | while read -r init; do
      gh release download "verify-db-${init:0:4}-${init:4:2}" -R "$R" -D "$dir" --clobber \
        -p "sfc_${init}.parquet" -p "raob_${init}.parquet"
    done
    echo "database: $(find "$dir" -name 'sfc_*.parquet' | wc -l) runs fetched"
    ;;
  upload)
    dir=${2:?DIR}
    for f in "$dir"/sfc_*.parquet "$dir"/raob_*.parquet; do
      [ -e "$f" ] || continue
      init=$(basename "$f" .parquet); init=${init#*_}
      tag="verify-db-${init:0:4}-${init:4:2}"
      if ! gh release view "$tag" -R "$R" >/dev/null 2>&1; then
        # another job may make the same month's release at the same moment: a failed create is fine if it now exists
        gh release create "$tag" -R "$R" --latest=false --title "Verification database ${init:0:4}-${init:4:2}" \
          --notes "Matched forecast-observation pairs of every 00Z run of the month (scripts/verify/mverify_scores.py, mverify): sfc_<init>.parquet against surface stations (GHCNh), raob_<init>.parquet against radiosondes, every model, one row per pair (mverify.match.PAIR_COLUMNS)." \
          || gh release view "$tag" -R "$R" >/dev/null
      fi
      gh release upload "$tag" "$f" -R "$R" --clobber
      echo "database: $(basename "$f") in $tag"
    done
    ;;
  *)
    sed -n '2,9p' "$0"; exit 2
    ;;
esac
