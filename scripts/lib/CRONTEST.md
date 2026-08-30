# Cron isolation test — started 2026-08-30

HYPOTHESIS: GitHub's ~36% cron honour rate on this repo is a per-repo
throttle across many scheduled workflows, not a per-workflow property.
If so, a repo with ONE schedule should fire close to 100%.

SETUP
- soi-hourly.yml keeps its cron: '12 * * * *'  (24 expected firings/day)
- the other 19 have their schedule: blocks commented out, marked
  "CRON-TEST 2026-08-30 (restore: uncomment this block)"
- ALL 20 remain covered by scripts/lib/dispatch_workflows.sh, so no
  product stops updating for the duration of the test
- soi-hourly stays in the dispatcher too; the measurement counts only
  runs whose event == "schedule", so dispatcher runs do not mask it

BASELINE (previous 24 h, all 20 workflows on cron)
- overall: 40 of 111 expected firings honoured = 36%
- soi-hourly specifically: 5 of 24 = 21%
- queue latency was 0 s for every event type, so this is delivery,
  not runner availability

READ THE RESULT
  gh run list --workflow soi-hourly.yml --limit 60 \
    --json event,createdAt -q '.[]|select(.event=="schedule")|.createdAt'
Count firings since the start time against one per hour.

RESTORE
  Uncomment every "# CRON-TEST" block (19 files), or:
  git revert <this commit>
