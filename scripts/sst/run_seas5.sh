#!/bin/bash
# ECMWF SEAS5 seasonal outlook driver — the steps of .github/workflows/seas5.yml.
#
# Moved off the laptop 2026-10-04 (user rule: no downloads on the laptop; never download anything twice).
# The laptop launchd job com.scorvec.seas5 is retired. Three stages:
#
#   run_seas5.sh fetch  YYYYMM   this issue's forecasts from the CDS — sequential, one key. Refused
#                                outside GitHub Actions (seas5_ref.guard_fetch). Exit 3 = issue not out yet.
#   run_seas5.sh tables YYYYMM   derived hindcast tables for the issue's start month (seas5_ref.py). With
#                                TABLES_FETCH=1 (Actions only) a missing table's raw hindcast kind is fetched,
#                                reduced and deleted, one kind at a time, within BUDGET_MIN minutes; without it
#                                only what is already on disk is reduced (the laptop's local GRIBs).
#   run_seas5.sh build  YYYYMM   every product from this issue's forecasts + the derived tables, no network.
#                                Also a local build helper: SEAS5_DATA=<dir with forecast/ and ref/>
#                                SST_SITE_ROOT=<scratch> SEAS5_OFFLINE=1 scripts/sst/run_seas5.sh build 202609
#
# Outputs: assets/sst/seas5_*.webp, assets/sst/data/seas5_*.json, and seas5_status.json (complete = every
# hindcast table of the start month exists, so later firings of the workflow are no-ops). Derived tables:
# $SEAS5_REF_DIR (default scripts/sst/data/seas5/ref), mirrored to the release seas5-ref-v1 by the workflow.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY="${PYTHON:-python}"
export MPLBACKEND=Agg
CMD="${1:?usage: run_seas5.sh fetch|tables|build [YYYYMM]}"
ISSUE="${2:-$(date -u +%Y%m)}"
MM="${ISSUE:4:2}"
export SEAS5_ISSUE="$ISSUE"     # raw hindcast fetches take their lead-hour calendar from the issue year (leap Februaries)
SITE="${SST_SITE_ROOT:-$(cd ../.. && pwd)}"
FAILED=()
step() {   # step <label> <command...> — best-effort; a failed product never sinks the others
  local label="$1"; shift
  echo "::group::$label"
  if "$@"; then echo "::endgroup::"; else echo "::endgroup::"; echo "::warning::$label FAILED"; FAILED+=("$label"); fi
}

case "$CMD" in
fetch)
  LOG=$(mktemp)
  "$PY" seas5_outlook.py fetch --issue "$ISSUE" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  if grep -q -e "is not on the CDS yet" -e "the CDS is busy" "$LOG"; then exit 3; fi   # 3 = try again later
  [ "$rc" -eq 0 ] || exit 1
  # the other products' own forecast pulls (6-hourly / daily, regional) — each sequential
  step "fetch popT (6-hourly t2m, US + Brazil)"      "$PY" seas5_popT.py fetch --issue "$ISSUE"
  step "fetch extremes (daily Tmax/Tmin)"            "$PY" seas5_extremes.py fetch --issue "$ISSUE"
  step "fetch wind (6-hourly 10 m u/v, US)"          "$PY" seas5_wind.py fetch --issue "$ISSUE"
  step "fetch snowfall (daily, North America)"       "$PY" seas5_snow.py fetch --issue "$ISSUE"
  step "fetch regimes (daily z500, 51 members)"      "$PY" seas5_regimes_fetch.py forecast --issue "$ISSUE"
  [ ${#FAILED[@]} -eq 0 ] || echo "::warning::forecast fetches incomplete: ${FAILED[*]}"
  exit 0
  ;;
tables)
  ARGS=(hindcast --month "$MM")
  if [ "${TABLES_FETCH:-0}" = "1" ]; then ARGS+=(--fetch --discard-raw --budget-min "${BUDGET_MIN:-200}"); fi
  "$PY" seas5_ref.py "${ARGS[@]}"
  "$PY" seas5_ref.py missing --month "$MM" >/dev/null
  exit $?
  ;;
build)
  step "outlook (indices, terciles, maps, polar caps)" "$PY" seas5_outlook.py build --issue "$ISSUE" --previous 3
  step "normals"                    "$PY" seas5_normals.py --issue "$ISSUE"
  step "teleconnections"            "$PY" seas5_tele.py --issue "$ISSUE"
  step "pme regions"                "$PY" seas5_pme_regions.py --issue "$ISSUE"
  step "regimes"                    "$PY" seas5_regimes.py --issue "$ISSUE"      # after the outlook: quotes its DJF rel. Niño-3.4
  step "popT"                       "$PY" seas5_popT.py build --issue "$ISSUE"
  step "threshold days"             "$PY" seas5_extremes_build.py "$ISSUE"
  step "cold-snap runs"             "$PY" seas5_runs.py --issue "$ISSUE"
  step "wind events"                "$PY" seas5_wind.py build --issue "$ISSUE"
  step "snowfall"                   "$PY" seas5_snow.py build --issue "$ISSUE"
  MISSING=$("$PY" seas5_ref.py missing --month "$MM")
  "$PY" - "$ISSUE" "$MISSING" "$(IFS='|'; echo "${FAILED[*]:-}")" "$SITE/assets/sst/data/seas5_status.json" <<'EOF'
import json, sys, time
issue, missing, failed, out = sys.argv[1:5]
failed = [f for f in failed.split("|") if f]
# complete = the main outlook built and every hindcast table of the start month exists; a failed minor
# product does not hold the issue open (re-running would not fix it), it is listed for a human instead
doc = {"issue": issue, "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
       "missing_tables": missing.split(), "failed": failed,
       "complete": not missing.split() and not any(f.startswith("outlook") for f in failed)}
open(out, "w").write(json.dumps(doc, separators=(",", ":")))
print("status:", doc)
EOF
  [ ${#FAILED[@]} -eq 0 ] || { echo "failed products: ${FAILED[*]}"; }
  exit 0
  ;;
*) echo "unknown command $CMD" >&2; exit 2 ;;
esac
