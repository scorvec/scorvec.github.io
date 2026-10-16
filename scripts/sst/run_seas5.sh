#!/bin/bash
# ECMWF SEAS5 seasonal outlook driver — the steps of .github/workflows/seas5.yml.
#
# Rule (2026-10-04): downloads run in GitHub Actions only; never download anything twice. Three stages:
#
#   run_seas5.sh fetch  YYYYMM   this issue's forecasts from the CDS — sequential, one key. Refused
#                                outside GitHub Actions (seas5_ref.guard_fetch). Exit 3 = issue not out yet.
#   run_seas5.sh tables YYYYMM   derived hindcast tables for the issue's start month (seas5_ref.py). With
#                                TABLES_FETCH=1 (Actions only) a missing table's raw hindcast kind is fetched,
#                                reduced and deleted, one kind at a time, within BUDGET_MIN minutes; without it
#                                only what is already on disk is reduced.
#   run_seas5.sh build  YYYYMM   every product from this issue's forecasts + the derived tables, no network.
#                                Also an offline build helper: SEAS5_DATA=<dir with forecast/ and ref/>
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
  if grep -q "CHECK THE REQUEST" "$LOG"; then echo "::error::SEAS5 $ISSUE: MARS keeps rejecting the request - check it"; exit 1; fi
  if grep -q -e "is not on the CDS yet" -e "the CDS is busy" "$LOG"; then exit 3; fi   # 3 = try again later
  [ "$rc" -eq 0 ] || exit 1
  # the other products' own forecast pulls (6-hourly / daily, regional) — each sequential
  step "fetch popT (6-hourly t2m, US + Brazil)"      "$PY" seas5_popT.py fetch --issue "$ISSUE"
  step "fetch extremes (daily Tmax/Tmin)"            "$PY" seas5_extremes.py fetch --issue "$ISSUE"
  step "fetch wind (6-hourly 10 m u/v, US)"          "$PY" seas5_wind.py fetch --issue "$ISSUE"
  step "fetch snowfall (daily, North America)"       "$PY" seas5_snow.py fetch --issue "$ISSUE"
  step "fetch regimes (daily z500, 51 members)"      "$PY" seas5_regimes_fetch.py forecast --issue "$ISSUE"
  # the polar-vortex product next (one small daily u10 request + a monthly u100): it publishes on its own (seas5.yml
  # "Publish the polar-vortex product"), so it does not wait for the lead-7 fields
  step "fetch polar vortex (daily u10)"              "$PY" seas5_ssw.py fetch --issue "$ISSUE" --only u10
  [ ${#FAILED[@]} -eq 0 ] || echo "::warning::forecast fetches incomplete: ${FAILED[*]}"
  exit 0
  ;;
fetch-l7)
  # lead month 7 (from the October 2026 issue): the sub-daily product reduced to the monthly definition, in
  # seas5_lead7.COMBINED order (single-level requests first). Budgeted: the CDS queued pressure-level requests
  # 1.5–4.5 h each on 2026-10-07/08; no new request after the job has less than 120 min left (timeout 340 min);
  # the cache keeps what came and the next firing fetches the rest.
  L7B=200; if [ -n "${JOB_T0:-}" ]; then L7B=$(( 340 - ( $(date +%s) - JOB_T0 ) / 60 - 120 )); fi
  "$PY" seas5_ssw.py fetch --issue "$ISSUE" --only u100 || echo "::warning::monthly u100 forecast not fetched (the vortex card shows 10 hPa only)"
  if [ "$L7B" -gt 0 ]; then
    "$PY" seas5_lead7.py forecast --issue "$ISSUE" --budget-min "$L7B" || echo "::warning::lead-7 fields incomplete — the next run continues"
  else
    echo "::warning::no time left for the lead-7 fetch this run"
  fi
  exit 0
  ;;
ssw)
  "$PY" seas5_ssw.py build --issue "$ISSUE"
  exit $?
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
  step "polar vortex / SSW risk"    "$PY" seas5_ssw.py build --issue "$ISSUE"
  MISSING=$("$PY" seas5_ref.py missing --month "$MM")
  MISS7=$("$PY" seas5_lead7.py missing --issue "$ISSUE")
  "$PY" - "$ISSUE" "$MISSING" "$(IFS='|'; echo "${FAILED[*]:-}")" "$SITE/assets/sst/data/seas5_status.json" "$MISS7" <<'EOF'
import json, sys, time
issue, missing, failed, out, miss7 = sys.argv[1:6]
failed = [f for f in failed.split("|") if f]
# complete = the main outlook built, every hindcast table of the start month exists (the lead-7 tables included)
# and every lead-7 forecast supplement was fetched; a failed minor product does not hold the issue open
# (re-running would not fix it), it is listed for a human instead
doc = {"issue": issue, "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
       "missing_tables": missing.split(), "missing_lead7_forecast": miss7.split(), "failed": failed,
       "complete": not missing.split() and not miss7.split() and not any(f.startswith("outlook") for f in failed)}
open(out, "w").write(json.dumps(doc, separators=(",", ":")))
print("status:", doc)
EOF
  [ ${#FAILED[@]} -eq 0 ] || { echo "failed products: ${FAILED[*]}"; }
  exit 0
  ;;
*) echo "unknown command $CMD" >&2; exit 2 ;;
esac
