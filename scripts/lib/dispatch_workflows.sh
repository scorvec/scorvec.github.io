#!/bin/bash
# Fire GitHub Actions workflows from the laptop instead of trusting GitHub's cron.
#
# WHY: measured 2026-08-29, GitHub honours roughly one SCHEDULED firing every
# seven hours for this repo and delivers it hours late - a 06:20 cron routinely
# lands after 12:00. workflow_dispatch has no such problem: all seven dispatches
# that day started with 0 s of queue latency. So the scheduler is the unreliable
# part, not Actions itself.
#
# This gives both halves of what we want: the laptop keeps the timing (it is
# awake and reliable) while the runner keeps the compute (the whole point of
# moving these off the laptop). A dispatch is one API call - no cores, no
# downloads, nothing that competes with WRF.
#
# The crons stay in place as a backstop for when the laptop is off or asleep.
# Double-firing is prevented by the "already ran since" check below, not by
# removing them.
#
#     scripts/lib/dispatch_workflows.sh            # fire whatever is due
#     scripts/lib/dispatch_workflows.sh --dry-run  # show what would fire
#     scripts/lib/dispatch_workflows.sh --force sst.yml
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GH="${GH_BIN:-/opt/homebrew/bin/gh}"
cd "$REPO" || exit 1

DRY=0; FORCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --force)   FORCE="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# workflow | UTC HH:MM | extra `gh workflow run` args
#
# Times match the crons in each workflow so behaviour is identical whichever
# path fires. Deliberately NOT listing the hourly satellite/SOI jobs: they are
# backfill-designed (each run repairs the whole window), so sparse firing costs
# them nothing and dispatching 24x/day each would be load for no product gain.
SCHEDULE=$(cat <<'EOF'
sst.yml|19:23|
ecape.yml|00:50|
ecape.yml|06:50|
ecape.yml|12:50|
ecape.yml|18:50|
strat.yml|07:35|
strat.yml|19:35|
gdps-charts.yml|06:20|-f cycle=00
gdps-charts.yml|18:20|-f cycle=12
EOF
)

now_epoch=$(date -u +%s)
today=$(date -u +%Y-%m-%d)
fired=0; skipped=0
# Workflows already dispatched in THIS pass. A run that has only just been
# requested does not yet show up in `gh run list` as satisfying a slot, so
# without this a workflow with two slots both overdue (a fresh one that has
# never run, say) would be fired twice in a single pass.
declare -a DONE=()

# Newest run start for a workflow, as an epoch. Uses createdAt because that is
# when the run was requested; startedAt can lag on a busy queue.
last_run_epoch() {
  local wf="$1" iso
  iso=$("$GH" run list --workflow "$wf" --limit 1 \
        --json createdAt -q '.[0].createdAt' 2>/dev/null) || return 1
  [ -z "$iso" ] && { echo 0; return 0; }
  date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$iso" "+%s" 2>/dev/null || echo 0
}

dispatch() {
  local wf="$1" extra="$2"
  if [ "$DRY" = "1" ]; then
    echo "    would dispatch: $wf $extra"
    return 0
  fi
  # shellcheck disable=SC2086
  if "$GH" workflow run "$wf" $extra >/dev/null 2>&1; then
    echo "    dispatched $wf $extra"
    return 0
  fi
  echo "    FAILED to dispatch $wf $extra" >&2
  return 1
}

if [ -n "$FORCE" ]; then
  dispatch "$FORCE" ""
  exit $?
fi

# How far back an unsatisfied slot is still worth firing. Covers an overnight
# sleep without resurrecting a slot from days ago, which would publish a stale
# cycle for no reason.
MAX_AGE_H=20
yesterday=$(date -u -v-1d +%Y-%m-%d 2>/dev/null || date -u -d "yesterday" +%Y-%m-%d)

while IFS='|' read -r wf hhmm extra; do
  [ -z "$wf" ] && continue
  # Consider today's slot AND yesterday's. Without the latter, a 19:23 slot that
  # the laptop slept through is simply lost once UTC midnight passes - the exact
  # failure this script exists to prevent.
  due_epoch=0
  for d in "$today" "$yesterday"; do
    e=$(date -j -u -f "%Y-%m-%d %H:%M:%S" "$d ${hhmm}:00" "+%s" 2>/dev/null) || continue
    if [ "$e" -le "$now_epoch" ] && [ "$e" -gt "$due_epoch" ]; then due_epoch=$e; fi
  done
  # Nothing due, or the due slot is older than we are willing to chase.
  [ "$due_epoch" -eq 0 ] && continue
  [ $(( (now_epoch - due_epoch) / 3600 )) -ge "$MAX_AGE_H" ] && continue
  # Already ran at or after this slot? Then this slot is satisfied - by us, by
  # GitHub's own cron, or by a manual run. Self-healing: if the laptop was
  # asleep at 06:20 and wakes at 09:00, the slot is still unsatisfied and fires
  # then, which is the behaviour the cron could not give us.
  case " ${DONE[*]:-} " in *" $wf "*) skipped=$((skipped + 1)); continue ;; esac
  lr=$(last_run_epoch "$wf")
  if [ "${lr:-0}" -ge "$due_epoch" ]; then
    skipped=$((skipped + 1))
    continue
  fi
  age_min=$(( (now_epoch - due_epoch) / 60 ))
  if [ "${lr:-0}" -eq 0 ]; then lrs="never"; else lrs=$(date -u -r "$lr" "+%d %H:%MZ" 2>/dev/null); fi
  echo "  $wf due ${hhmm}Z (${age_min} min ago), last run: $lrs"
  if dispatch "$wf" "$extra"; then fired=$((fired + 1)); DONE+=("$wf"); fi
  # A tiny gap so several dispatches in one pass do not race the API.
  sleep 2
done <<< "$SCHEDULE"

echo "  fired $fired, already-satisfied $skipped"
