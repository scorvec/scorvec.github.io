#!/bin/bash
# The runs' statistics (ensmaps.py save_stats, ensmaps_super.py save_super), kept as assets of the release
# ensmaps-stats so the change maps of the runs up to 48 h later can compare with them. An asset is
# <model>-<YYYYMMDDHH>.npz, model = ifs | aifs | gefs | geps | super; a run made again replaces its own.
#
#   stats_store.sh get DIR MODEL CYCLE [CYCLE ...]   DIR/<cycle>.npz for every cycle the release holds
#   stats_store.sh put FILE MODEL CYCLE              upload FILE as the model's statistics of the cycle
#   stats_store.sh prune MODEL FLOOR                 delete the model's assets of cycles before FLOOR
#
#   env: GITHUB_REPOSITORY; GH_TOKEN for put and prune (get reads the public download address, no API request)
set -euo pipefail
TAG=ensmaps-stats
R="${GITHUB_REPOSITORY:?}"

case "${1:-}" in
  get)
    dir=$2 model=$3
    shift 3
    mkdir -p "$dir"
    for cyc in "$@"; do
      if curl -fsSL --retry 3 -o "$dir/$cyc.npz.part" "https://github.com/$R/releases/download/$TAG/$model-$cyc.npz"; then
        mv "$dir/$cyc.npz.part" "$dir/$cyc.npz"
        echo "$model $cyc: statistics in hand"
      else
        rm -f "$dir/$cyc.npz.part"
        echo "$model $cyc: no statistics kept"
      fi
    done
    ;;
  put)
    file=$2 name="$3-$4.npz"
    tmp=$(mktemp -d)
    cp "$file" "$tmp/$name"
    gh release view "$TAG" -R "$R" > /dev/null 2>&1 || gh release create "$TAG" -R "$R" --latest=false \
      --title "Global ensemble maps: the runs' statistics" \
      --notes "Per-run statistics of the last 48 h of ensemble runs (scripts/ensmaps/stats_store.sh); the change maps compare with them."
    gh release upload "$TAG" "$tmp/$name" -R "$R" --clobber
    rm -rf "$tmp"
    echo "kept $name"
    ;;
  prune)
    model=$2 floor=$3
    gh release view "$TAG" -R "$R" --json assets -q '.assets[].name' \
      | awk -v m="$model" -v floor="$floor" 'index($0, m "-") == 1 && substr($0, length(m) + 2, 10) < floor' \
      | while read -r name; do
        gh release delete-asset "$TAG" "$name" -R "$R" -y && echo "deleted $name" || echo "::warning::could not delete $name"
      done
    ;;
  *)
    echo "usage: stats_store.sh get DIR MODEL CYCLE... | put FILE MODEL CYCLE | prune MODEL FLOOR" >&2
    exit 2
    ;;
esac
