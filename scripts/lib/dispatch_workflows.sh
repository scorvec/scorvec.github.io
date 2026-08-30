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
#   DOW@HH:MM once a week on that day ("mon@09:40", "tue,fri@08:40")
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
# Added 2026-08-30. These eleven had NO trigger but cron, which was delivering
# 36% of firings over the preceding 24 h and running 4-6 h late when it did.
# The crons stay as the laptop-off backstop; these slots mirror them exactly.
mjo.yml|08:37|
mjo.yml|11:37|
mjo.yml|20:37|
mjo.yml|23:37|
skewt-data.yml|01:35|
skewt-data.yml|02:35|
skewt-data.yml|03:35|
skewt-data.yml|07:35|
skewt-data.yml|13:35|
skewt-data.yml|14:35|
skewt-data.yml|15:35|
skewt-data.yml|19:35|
olr-waves.yml|02:40|
site-stats.yml|05:17|
mur-sst.yml|13:47|
data-freshness.yml|14:25|
kiribati-history.yml|14:40|
qbo.yml|mon@07:15|
colombia-radar.yml|mon@09:40|
sst-events.yml|mon@18:41|
skewt-gaps.yml|tue,fri@08:40|
# Daily frames-branch GC. Waits for a quiet moment itself and skips rather
# than racing a publisher, so a late firing costs nothing.
gc.yml|04:52|
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
# A FAILED run does not satisfy a slot forever. It used to: this read only
# createdAt, so when mur-sst died on an upstream HTTP 403 at 16:48Z the slot
# counted as done and the job would not have run again until the next day - a
# job silently not running, which is the whole failure mode this script exists
# to remove. After RETRY_AFTER_MIN a failed slot goes back to unsatisfied and
# the next pass fires it again; MAX_AGE_H still bounds how long that continues,
# and each retry moves the timestamp, so a permanently broken job retries on
# that interval rather than every pass.
RETRY_AFTER_MIN=120
last_run_epoch() {
  local wf="$1" line iso concl
  line=$("$GH" run list --workflow "$wf" --limit 1 \
        --json createdAt,conclusion,status -q '.[0] | "\(.createdAt)|\(.conclusion)|\(.status)"' 2>/dev/null) || return 1
  [ -z "$line" ] || [ "$line" = "null" ] && { echo 0; return 0; }
  iso="${line%%|*}"; concl="$(echo "$line" | cut -d'|' -f2)"
  local st; st="$(echo "$line" | cut -d'|' -f3)"
  local e; e=$(date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$iso" "+%s" 2>/dev/null || echo 0)
  # Still running counts as satisfied - do not stack a second copy on top.
  if [ "$st" != "completed" ]; then echo "$e"; return 0; fi
  case "$concl" in
    failure|timed_out|cancelled|startup_failure)
      if [ $(( now_epoch - e )) -ge $(( RETRY_AFTER_MIN * 60 )) ]; then
        echo 0                                   # eligible to retry
        return 0
      fi
      ;;
  esac
  echo "$e"
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
    *@*)
      # Weekly slot: "mon@09:40", or several days "tue,fri@08:40". Needed
      # because four of the products run weekly and cron alone was honouring
      # ~36% of firings - a missed weekly slot is a missed WEEK, not an hour.
      dows="${hhmm%@*}"; t="${hhmm#*@}"
      for d in "$today" "$yesterday"; do
        wd=$(date -j -u -f "%Y-%m-%d" "$d" "+%a" 2>/dev/null | tr '[:upper:]' '[:lower:]') || continue
        case ",$dows," in *",$wd,"*) ;; *) continue ;; esac
        e=$(date -j -u -f "%Y-%m-%d %H:%M:%S" "$d ${t}:00" "+%s" 2>/dev/null) || continue
        if [ "$e" -le "$now_epoch" ] && [ "$e" -gt "$due_epoch" ]; then due_epoch=$e; fi
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

# ---------------------------------------------------------------- pages ----
# GitHub deliberately does NOT trigger workflows from pushes made with
# GITHUB_TOKEN, so every commit CI makes to main - manifests, generated HTML,
# static figures - sits undeployed until a human pushes. Measured 2026-08-30:
# five CI commits stacked up behind a deploy that had last run 35 min earlier,
# and assets/ecape/anim/index.json was on main while the site served a 404 for
# it. Animation FRAMES hid the problem, because they come from
# raw.githubusercontent on the frames branch and never touch Pages at all.
#
# So: deploy when main is actually ahead of the last deploy, rather than on a
# timer. Two API calls, and it fires only when there is something to publish.
pages_if_stale() {
  local newest last
  newest=$("$GH" api "repos/${REPO_SLUG}/commits/main" -q '.commit.committer.date' 2>/dev/null) || return 0
  last=$("$GH" run list --workflow pages.yml --limit 1 --json createdAt -q '.[0].createdAt' 2>/dev/null) || return 0
  [ -z "$newest" ] || [ -z "$last" ] && return 0
  local ne le
  ne=$(date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$newest" "+%s" 2>/dev/null) || return 0
  le=$(date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$last" "+%s" 2>/dev/null) || return 0
  # 60 s of slack: a deploy started moments before the commit landed has not
  # published it, but one started after is fine.
  if [ "$ne" -gt $(( le + 60 )) ]; then
    echo "  main is $(( (ne - le) / 60 )) min ahead of the last Pages deploy"
    dispatch "pages.yml" "" && fired=$((fired + 1))
  fi
}
REPO_SLUG="$("$GH" repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo scorvec/scorvec.github.io)"
[ "$DRY" = "1" ] || pages_if_stale

echo "  fired $fired, already-satisfied $skipped"
