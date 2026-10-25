#!/bin/bash
# Run a workflow step's independent products at the same time: their downloads overlap each other and the others'
# drawing, and the runner's cores are all used. Each product's output goes to its own log, printed as one group when it
# ends, and a failed product is recorded by degraded.sh as it would be in a step of its own.
#
#     source scripts/lib/parallel.sh
#     par "wave_maps" "keeping the previous loops" python scripts/strat/wave_maps.py ...
#     par "polarcap" "keeping the previous figures" python scripts/strat/polarcap.py ...
#     par_wait                                   # every product started above has finished; their logs are printed
#
# PAR_MAX (default: the number of cores) bounds how many run at once; a further par waits for a free slot.
PAR_DIR="$(mktemp -d)"
PAR_MAX="${PAR_MAX:-$(nproc 2>/dev/null || echo 4)}"
PAR_N=0

par() {
  local name="$1" why="$2"; shift 2
  while [ "$(jobs -rp | wc -l)" -ge "$PAR_MAX" ]; do wait -n 2>/dev/null || sleep 0.5; done
  PAR_N=$((PAR_N + 1))
  local tag
  tag="$(printf '%02d' "$PAR_N")"
  printf '%s\n%s\n' "$name" "$why" > "$PAR_DIR/$tag.name"
  ( start=$(date +%s); "$@" > "$PAR_DIR/$tag.log" 2>&1; rc=$?
    echo "$rc $(( $(date +%s) - start ))" > "$PAR_DIR/$tag.rc" ) &
  echo "  started $name"
}

par_wait() {
  wait
  local f tag name why rc secs
  for f in "$PAR_DIR"/*.name; do
    [ -e "$f" ] || continue
    tag="$(basename "$f" .name)"; name="$(sed -n 1p "$f")"; why="$(sed -n 2p "$f")"
    read -r rc secs < "$PAR_DIR/$tag.rc"
    echo "::group::$name ($secs s, exit $rc)"
    cat "$PAR_DIR/$tag.log"
    echo "::endgroup::"
    [ "$rc" = 0 ] || "$(dirname "${BASH_SOURCE[0]}")/degraded.sh" "$name" "$why"
    rm -f "$PAR_DIR/$tag".*
  done
}
