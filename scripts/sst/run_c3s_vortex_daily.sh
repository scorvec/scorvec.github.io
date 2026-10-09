#!/bin/bash
# Laptop driver for the daily C3S u 10 hPa hindcast fetch (scripts/sst/c3s_vortex_daily.py; user 2026-10-09 "you can
# rerun anything on the laptop"). Sequential and resumable: held tables are skipped; every CDS request goes through
# cds_seq.py, which waits while the key has another job in flight and steps aside (deletes its job) whenever a run of the
# `seas5` Actions concurrency group is queued or in progress. Each table is uploaded to the release c3s-daily-u10-v1.
# The code is copied next to the data first, so the run does not depend on the checkout it was started from.
#     nohup bash scripts/sst/run_c3s_vortex_daily.sh [systems...] &
set -u
SRC=$(cd "$(dirname "$0")/../.." && pwd)
D=${C3SVD_DIR:-/Users/shawn/scorvec.github.io/scripts/sst/data/c3s_vortex_daily}
PY=${PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}
mkdir -p "$D/code/scripts/sst" "$D/code/.github/workflows" "$D/logs"
cp "$SRC/scripts/sst/c3s_vortex_daily.py" "$SRC/scripts/sst/cds_seq.py" "$D/code/scripts/sst/"
cp "$SRC"/.github/workflows/*.yml "$D/code/.github/workflows/"
export C3SVD_DIR=$D C3SVD_TAG=${C3SVD_TAG:-c3s-daily-u10-v1} CDS_SEQ_LOG=$D/logs/cds_seq.jsonl CDS_SEQ_REPO=$D/code
# one fetcher at a time: wait for any other c3s_vortex_daily fetch to finish first
while pgrep -f "c3s_vortex_daily.py fetch" >/dev/null; do sleep 60; done
SYSTEMS=${*:-mf_9 ukmo_610 dwd_22 cmcc_4 eccc_4 eccc_5 jma_4 bom_2}
for k in $SYSTEMS; do
  echo "=== $k $(date -u +%FT%TZ)" >> "$D/logs/fetch_$k.log"
  "$PY" -u "$D/code/scripts/sst/c3s_vortex_daily.py" fetch --only "$k" >> "$D/logs/fetch_$k.log" 2>&1
done
echo "pass done $(date -u +%FT%TZ)" >> "$D/logs/run_all.log"
