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

# workflow | slot | extra `gh workflow run` args
#
# Slot syntax:
#   HH:MM     once daily at that UTC time
#
# NOTE the `-f publish=true` on ecape and strat. Those workflows gate their
# commit step on `event_name == 'schedule' || inputs.publish == 'true'`, so a
# dispatch WITHOUT it renders everything and then publishes nothing - the first
# RunAtLoad pass fired all three that way before this was spotted. Manual dry
# runs are still available by dispatching them by hand without the flag.
#   *:MM      every hour at MM past
#   */N:MM    every N hours at MM past (0, N, 2N ... UTC)
#
# Times match the crons in each workflow so behaviour is identical whichever
# path fires.
#
# ECAPE fires at cycle+1h20, not cycle+50min. At 06:55Z on 2026-08-30 the HRRR
# 06Z analysis was not posted yet, so the resolver fell back to 00Z and the run
# spent 72 min re-rendering a loop that was already on the site. Each cycle was
# therefore reaching the page roughly six hours late, when the NEXT slot finally
# saw it. The workflow now also skips outright when the resolved cycle is the
# one already published.
#
# The hourly jobs ARE listed. They are backfill-designed so sparse firing loses
# no data, but GitHub was honouring roughly one firing in seven hours, which
# left the satellite loops up to 7 h stale on the site - freshness is a real
# product difference even when completeness is not. Each is ~1 min of runner
# time and unlimited on a public repo.
SCHEDULE=$(cat <<'EOF'
sst.yml|19:23|
ecape.yml|01:20|-f publish=true
ecape.yml|07:20|-f publish=true
ecape.yml|13:20|-f publish=true
ecape.yml|19:20|-f publish=true
strat.yml|07:35|-f publish=true
strat.yml|19:35|-f publish=true
gdps-charts.yml|06:20|-f cycle=00
gdps-charts.yml|18:20|-f cycle=12
soi-hourly.yml|*:12|
pacific-satellite.yml|*:40|
samerica-satellite.yml|*:25|
olr-hovmoller.yml|01:25|
olr-hovmoller.yml|07:25|
olr-hovmoller.yml|13:25|
olr-hovmoller.yml|19:25|
kiribati-wind.yml|*/6:18|
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
# A run that started shortly BEFORE a slot satisfies it. GitHub's own cron
# straggles in at unpredictable times, and without this a cron firing at 00:39
# for the 00:40 slot would be followed by our dispatch a minute later - two runs
# of the same cycle. Well under the 60 min spacing of the tightest slot.
GRACE_MIN=10
yesterday=$(date -u -v-1d +%Y-%m-%d 2>/dev/null || date -u -d "yesterday" +%Y-%m-%d)

while IFS='|' read -r wf hhmm extra; do
  [ -z "$wf" ] && continue
  # Consider today's slot AND yesterday's. Without the latter, a 19:23 slot that
  # the laptop slept through is simply lost once UTC midnight passes - the exact
  # failure this script exists to prevent.
  due_epoch=0
  case "$hhmm" in
    \*:*|\*/*:*)
      # Recurring slot. Walk back hour by hour from now to the most recent
      # occurrence; 26 covers a full day plus the yesterday allowance without
      # needing separate date arithmetic for the boundary.
      mm="${hhmm##*:}"
      every=1
      case "$hhmm" in */*) every="${hhmm#*/}"; every="${every%%:*}" ;; esac
      for back in $(seq 0 26); do
        cand=$(( now_epoch - back * 3600 ))
        ch=$(date -u -r "$cand" +%H 2>/dev/null) || continue
        # 10# forces base 10: 08 and 09 are invalid octal and would abort here.
        [ $(( 10#$ch % every )) -ne 0 ] && continue
        cd_=$(date -u -r "$cand" +%Y-%m-%d 2>/dev/null) || continue
        e=$(date -j -u -f "%Y-%m-%d %H:%M:%S" "$cd_ ${ch}:${mm}:00" "+%s" 2>/dev/null) || continue
        if [ "$e" -le "$now_epoch" ]; then due_epoch=$e; break; fi
      done
      ;;
    *)
      for d in "$today" "$yesterday"; do
        e=$(date -j -u -f "%Y-%m-%d %H:%M:%S" "$d ${hhmm}:00" "+%s" 2>/dev/null) || continue
        if [ "$e" -le "$now_epoch" ] && [ "$e" -gt "$due_epoch" ]; then due_epoch=$e; fi
      done
      ;;
  esac
  # Nothing due, or the due slot is older than we are willing to chase.
  [ "$due_epoch" -eq 0 ] && continue
  [ $(( (now_epoch - due_epoch) / 3600 )) -ge "$MAX_AGE_H" ] && continue
  # Already ran at or after this slot? Then this slot is satisfied - by us, by
  # GitHub's own cron, or by a manual run. Self-healing: if the laptop was
  # asleep at 06:20 and wakes at 09:00, the slot is still unsatisfied and fires
  # then, which is the behaviour the cron could not give us.
  case " ${DONE[*]:-} " in *" $wf "*) skipped=$((skipped + 1)); continue ;; esac
  lr=$(last_run_epoch "$wf")
  if [ "${lr:-0}" -ge $(( due_epoch - GRACE_MIN * 60 )) ]; then
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
