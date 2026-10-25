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
#
# Each product's numerical libraries run single-threaded (OMP/OpenBLAS/MKL threads 1, unless the caller set them): the
# products already fill the cores, and every library pool sized to all of them on top of that spun the machine on
# thread contention (a frame pool of 18 workers with 18 BLAS threads each took 14 min on system time, 95 s at 1).
#
# Memory is what bounds PAR_MAX on a runner (products that together outgrow it are killed, or take the runner with
# them): each product's group header gives its peak resident memory (GNU time, where there is one), and par_wait ends
# with the least memory the machine had available while they ran (sampled every 5 s).
PAR_DIR="$(mktemp -d)"
PAR_MAX="${PAR_MAX:-$(nproc 2>/dev/null || echo 4)}"
PAR_N=0
PAR_MEM_PID=""
PAR_TIME=()
if /usr/bin/time -f %M -o /dev/null true 2>/dev/null; then PAR_TIME=(/usr/bin/time -f %M -o); fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

par_mem() { while sleep 5; do awk '/^MemAvailable:/ {print $2}' /proc/meminfo; done; }

par() {
  local name="$1" why="$2"; shift 2
  if [ -z "$PAR_MEM_PID" ] && [ -r /proc/meminfo ]; then
    par_mem > "$PAR_DIR/mem" &
    PAR_MEM_PID=$!
    disown "$PAR_MEM_PID"
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$PAR_MAX" ]; do wait -n 2>/dev/null || sleep 0.5; done
  PAR_N=$((PAR_N + 1))
  local tag
  tag="$(printf '%02d' "$PAR_N")"
  printf '%s\n%s\n' "$name" "$why" > "$PAR_DIR/$tag.name"
  local run=("$@")
  if [ ${#PAR_TIME[@]} -gt 0 ] && [ "$(type -t "$1")" = file ]; then run=("${PAR_TIME[@]}" "$PAR_DIR/$tag.rss" "$@"); fi
  ( start=$(date +%s) rc=0; "${run[@]}" > "$PAR_DIR/$tag.log" 2>&1 || rc=$?
    echo "$rc $(( $(date +%s) - start ))" > "$PAR_DIR/$tag.rc" ) &
  echo "  started $name"
}

par_wait() {
  wait
  if [ -n "$PAR_MEM_PID" ]; then kill "$PAR_MEM_PID" 2>/dev/null || true; PAR_MEM_PID=""; fi
  local f tag name why rc secs rss
  for f in "$PAR_DIR"/*.name; do
    [ -e "$f" ] || continue
    tag="$(basename "$f" .name)"; name="$(sed -n 1p "$f")"; why="$(sed -n 2p "$f")"
    read -r rc secs < "$PAR_DIR/$tag.rc"
    rss="$(tail -n 1 "$PAR_DIR/$tag.rss" 2>/dev/null || true)"
    case "$rss" in ''|*[!0-9]*) rss="" ;; *) rss=", peak $((rss / 1024)) MB" ;; esac
    echo "::group::$name ($secs s, exit $rc$rss)"
    cat "$PAR_DIR/$tag.log"
    echo "::endgroup::"
    [ "$rc" = 0 ] || "$(dirname "${BASH_SOURCE[0]}")/degraded.sh" "$name" "$why"
    rm -f "$PAR_DIR/$tag".*
  done
  if [ -s "$PAR_DIR/mem" ]; then
    echo "memory: at least $(( $(sort -n "$PAR_DIR/mem" | head -n 1) / 1024 )) MB of $(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1024 )) MB available throughout"
  fi
  rm -f "$PAR_DIR/mem"
}
