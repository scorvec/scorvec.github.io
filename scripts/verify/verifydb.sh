#!/usr/bin/env bash
# The verification database: every 00Z run's matched pairs, every model, kept for good in the monthly releases
# verify-db-YYYY-MM of this repository: against the surface stations (sfc_<init>.parquet), the radiosondes (raob_)
# and Stage IV (rain_), and the pairs' sums that windows of runs are scored from (stats_, stn_), all written by
# scripts/verify/mverify_scores.py; the ensembles' cases and sums (ens-sfc-<m>_, ens-raob-<m>_, ens-rain-<m>_,
# ens-stats_, ens-rel_) by scripts/verify/ens_verify.py. A second copy is kept in the site's object store, under
# verify-db/YYYY-MM/. A file is never replaced or deleted in either place.
#
#   verifydb.sh list                 every file the database holds, one name per line
#   verifydb.sh fetch KIND N DIR     the newest N runs' KIND_<init>.parquet files (N 'all': every run) into DIR
#   verifydb.sh get DIR NAME...      the named files into DIR
#   verifydb.sh upload DIR           DIR's files into their months' releases (made when missing), but none it holds
#   verifydb.sh mirror               every file the object store lacks into it, checked against GitHub's SHA-256
#
# Needs gh with GH_TOKEN and GITHUB_REPOSITORY (as in Actions); mirror, the FRAMES_S3_* store settings as well.
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
  get)
    dir=${2:?DIR}; shift 2; mkdir -p "$dir"
    # one download per month's release, its files named
    printf '%s\n' "$@" | sed -nE 's/^[^_]+_([0-9]{4})([0-9]{2}).*/\1-\2 &/p' | awk '
      { p[$1] = p[$1] " -p " $2 } END { for (m in p) print m p[m] }' | while read -r month pats; do
      # shellcheck disable=SC2086  # the -p arguments are meant to split
      gh release download "verify-db-$month" -R "$R" -D "$dir" --clobber $pats
    done
    ;;
  upload)
    dir=${2:?DIR}
    for f in "$dir"/*_*.parquet; do
      [ -e "$f" ] || continue
      name=$(basename "$f"); init=${name%.parquet}; init=${init#*_}
      tag="verify-db-${init:0:4}-${init:4:2}"
      if ! held=$(gh release view "$tag" -R "$R" --json assets -q '.assets[].name' 2>/dev/null); then
        # another job may make the same month's release at the same moment: a failed create is fine if it now exists
        gh release create "$tag" -R "$R" --latest=false --title "Verification database ${init:0:4}-${init:4:2}" \
          --notes "Every 00Z run of the month matched with the observations, every model (scripts/verify/mverify_scores.py, mverify), one row per pair (mverify.match.PAIR_COLUMNS, and kept: whether mverify's default cuts keep it): sfc_<init>.parquet against surface stations (GHCNh), raob_<init>.parquet against radiosondes (IGRA2), rain_<init>.parquet 24 h totals against Stage IV on the 0.5 degree grid. stats_ and stn_ hold the pairs' sums that windows of runs are scored from. The ensembles (scripts/verify/ens_verify.py: IFS-ENS, AIFS-ENS, GEFS, GEPS to day 15) keep one row per case with its members summarised (mverify.ensemble.summarise): ens-sfc-<m>_, ens-raob-<m>_ and ens-rain-<m>_<init>.parquet, and their sums ens-stats_ and ens-rel_." \
          || gh release view "$tag" -R "$R" >/dev/null
        held=""
      fi
      # a file the database holds stays as it is: a run matched again later (a half-finished upload) may have lost a
      # model its source no longer keeps
      if grep -qxF "$name" <<< "$held"; then echo "database: $name already in $tag, kept"; continue; fi
      gh release upload "$tag" "$f" -R "$R"   # no --clobber: a file that appeared meanwhile fails, never replaced
      echo "database: $name in $tag"
    done
    ;;
  mirror)
    LIB="$(cd "$(dirname "$0")/../lib" && pwd)"   # here only: the other commands run from a checkout of scripts/verify
    # shellcheck source=../lib/frames_env.sh
    . "$LIB/frames_env.sh"
    frames_store_ready || { echo "::error::the object store is not usable: the database's second copy is not made"; exit 1; }
    work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
    "$FRAMES_PY" "$LIB/frames_store.py" files verify-db > "$work/store.txt"
    for t in $(tags); do
      gh release view "$t" -R "$R" --json assets -q '.assets[] | "\(.name) \(.size) \(.digest)"' | sed "s|^|verify-db/${t#verify-db-}/|"
    done > "$work/github.txt"
    # the files the store lacks; one it holds at another size, or that GitHub no longer holds, is reported, not changed
    awk 'FILENAME == ARGV[1] { if (NF == 2) s[$1] = $2; next }
      { g[$1] = 1 }
      !($1 in s) { print; next }
      s[$1] != $2 { printf "::warning::%s: %s bytes in the object store, %s on GitHub\n", $1, s[$1], $2 > "/dev/stderr" }
      END { for (k in s) if (!(k in g)) printf "::warning::%s: in the object store only, not on GitHub\n", k > "/dev/stderr" }' \
      "$work/store.txt" "$work/github.txt" > "$work/want.txt"
    for m in $(cut -d/ -f2 "$work/want.txt" | sort -u); do
      grep "^verify-db/$m/" "$work/want.txt" > "$work/month.txt"
      mkdir -p "$work/verify-db/$m"
      # shellcheck disable=SC2046  # the -p arguments are meant to split
      gh release download "verify-db-$m" -R "$R" -D "$work/verify-db/$m" $(awk -F'[/ ]' '{ print "-p", $3 }' "$work/month.txt")
      awk -F'[/ ]' '{ sub(/^sha256:/, "", $5); print $5 "  " $3 }' "$work/month.txt" > "$work/sums.txt"
      (cd "$work/verify-db/$m" && sha256sum --quiet -c "$work/sums.txt")
      "$FRAMES_PY" "$LIB/frames_store.py" sync --no-replace --root "$work" "+verify-db/$m"
      rm -rf "${work:?}/verify-db/$m"
    done
    echo "database: $(wc -l < "$work/github.txt") files on GitHub, $(wc -l < "$work/want.txt") of them copied to the object store now"
    ;;
  *)
    sed -n '2,16p' "$0"; exit 2
    ;;
esac
