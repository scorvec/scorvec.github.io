#!/bin/bash
# Local MJO + SST-page atmospheric build (primary path; the GitHub Action mjo.yml is
# the fallback for when the laptop is off). Dependency-tracked, not staged (2026-07-24):
#   Stage 1  RMM core — fetch u@200/850, build the RMM plot + page → COMMIT+PUSH.
#   Stage 2  parallel product groups (SFC / V200 / ANALYSIS / NODATA / HEAVY), each
#            gated only on ITS OWN files: one priority-ordered prefetch publishes
#            smallest-critical-first and renders self-fetch under per-file flocks,
#            so e.g. WAF (u+v@200) never waits behind the ~7 GB AAM pull.
# REVIVED 2026-07-18: the AAM / torque / MMSF / AAM-zonal block is back on (after a
# full math audit), joined by the new Walker-circulation, subtropical-jet and
# SOI-history products; ens_cycle runs with MJO_HEAVY_ATMOS=1 below.
# Downloads are watchdog-robust (download_aifs aborts+retries any stream that stalls),
# so a hung connection can no longer wedge the run. Idempotent: a `.cycle_done_<c>`
# marker skips a finished cycle; an interrupted run resumes at the unfinished stage
# (Stage-2 builders dedupe by init / re-render, so re-running is safe).
# Invoked twice daily by ~/Library/LaunchAgents/com.scorvec.mjo.plist.
set -uo pipefail

PY="${MJO_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/scripts/lib/gitlock.sh"; trap git_unlock EXIT
export MPLBACKEND=Agg PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
export MJO_GRIB_ARCHIVE="${MJO_GRIB_ARCHIVE:-$HOME/mjo/grib_archive}"
cd "$REPO/scripts/mjo" || exit 1

LOG="$REPO/scripts/mjo/run_local.log"

# Reaper: a prior instance stuck >2h (dead download, no timeout) blocks every
# later hourly fire. Kill anything that old before taking the lock.
for pid in $(pgrep -f "scripts/mjo/run_local.sh" 2>/dev/null); do
  [ "$pid" = "$$" ] && continue
  age=$(( $(date +%s) - $(ps -o lstart= -p "$pid" 2>/dev/null | xargs -I{} date -j -f "%a %b %d %T %Y" "{}" +%s 2>/dev/null || echo $(date +%s)) ))
  if [ "$age" -gt 7200 ]; then
    echo "reaper: killing stuck pipeline pid $pid (age ${age}s)"
    pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
  fi
done
# Fresh log each run (keep one previous as .prev) so stale tracebacks from earlier
# runs can't pile up and look current; and drop the ecmwf-opendata tqdm download
# progress spam (carriage-return bars / MB-s ticks) — keep only meaningful lines.
[ -f "$LOG" ] && mv -f "$LOG" "$LOG.prev"
exec > >(grep --line-buffered -avE 'MB/s|kB/s|[0-9]+%\|[█▏▎▍▌▋▊▉ ]|enfo-pf\.grib2: +[0-9]|[[:cntrl:]]\[A|^[[:space:]]*$' > "$LOG") 2>&1
echo "===================== $(date) ====================="
require_main || exit 0

# Cooldown guard: while data/.cooldown_until holds a FUTURE epoch, skip the run — lets
# us stop hammering a rate-limited ECMWF S3 for a few hours; launchd keeps firing but
# no-ops here, then auto-resumes once the window passes. `rm` the file to resume early.
COOLDOWN="$REPO/scripts/mjo/data/.cooldown_until"
if [ -f "$COOLDOWN" ] && [ "$(date +%s)" -lt "$(cat "$COOLDOWN" 2>/dev/null || echo 0)" ]; then
  echo "cooldown active until $(date -r "$(cat "$COOLDOWN" 2>/dev/null || echo 0)" 2>/dev/null) — skipping S3 download"
  exit 0
fi

# Self-watchdog: launchd runs at most ONE instance per label, so a wedged run
# also blocks the REAPER above — it lives in future instances that can never
# spawn while this one is alive. External rescue is therefore impossible; the
# run must cap itself. TERM the whole tree after 6 h (a healthy cycle is < 2 h).
( sleep 21600; echo "self-watchdog: run exceeded 6 h — terminating own tree"
  pkill -TERM -P $$ 2>/dev/null; kill -TERM $$ 2>/dev/null
  # bash defers TERM until its foreground child exits — a wedged child never
  # does (observed: 13 h zombie). Escalate to SIGKILL on the process group.
  sleep 60
  PGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d " ")
  [ -n "$PGID" ] && kill -KILL -- -"$PGID" 2>/dev/null ) &
WATCHDOG=$!
trap 'kill "$WATCHDOG" 2>/dev/null; git_unlock' EXIT

# Auto-rotate the ECMWF mirrors for this run (round-robin + demote whichever threw 503s
# last run), unless ECMWF_SOURCES is set explicitly. Self-steers away from a throttling
# mirror without manual intervention.
if [ -z "${ECMWF_SOURCES:-}" ]; then
  export ECMWF_SOURCES="$("$PY" -c "import sys; sys.path.insert(0,'$REPO/scripts/ecmwf'); import store; print(store.next_mirror_order())" 2>/dev/null)"
  echo "mirrors this run: ${ECMWF_SOURCES:-<default>}"
fi

read -r DATE TIME < <("$PY" -c \
  "import sys; sys.path.insert(0,'src'); from download_aifs import latest_run; d,t=latest_run(); print(d,t)" \
  2>/dev/null)
if ! [[ "$DATE" =~ ^[0-9]{8}$ && "$TIME" =~ ^(00|12)$ ]]; then
  echo "could not determine cycle (DATE='$DATE' TIME='$TIME'); exiting"; exit 0
fi
# Publication sentinel (fetch v2): only start bulk work once the cycle's LAST
# step is on a mirror — half-published cycles caused hours of retry crawl.
if ! "$PY" -c "import sys; sys.path.insert(0,'$REPO/scripts/ecmwf'); import rangefetch as rf; sys.exit(0 if rf.cycle_complete('$DATE','$TIME') else 1)" 2>/dev/null; then
  echo "cycle ${DATE} ${TIME}z not fully published yet — skipping this run"
  exit 0
fi
COMPACT="${DATE}_${TIME}z"
DONE_MARKER="data/.cycle_done_${COMPACT}"
RMM_PNG="$REPO/assets/mjo/rmm_${COMPACT}.png"
if [ -f "$DONE_MARKER" ]; then echo "$COMPACT fully built — nothing to do."; exit 0; fi
echo "building cycle $COMPACT …"

commit_push () {            # $1 = message; commits staged changes (if any) + pushes with retry
  if git diff --staged --quiet; then echo "  ($1: nothing to commit)"; return 0; fi
  git_lock || return 1     # serialise vs the other site pipelines (sst/synoptic/ens)
  git -c user.name="Shawn Corvec" -c user.email="shawncorvec@hotmail.com" commit -m "$1"
  for i in 1 2 3 4 5; do
    if git pull --rebase --autostash -X theirs && git push; then echo "  pushed: $1 (attempt $i)"; git_unlock; return 0; fi
    echo "  push attempt $i failed; retrying…"; sleep 5
  done
  echo "  ERROR: could not push: $1 (left as a local commit; the next run will carry it)"; git_unlock; return 1
}

publish () {                # $1 = message; $2.. = paths — ATOMIC stage+commit+push.
  local msg="$1"; shift     # Product groups commit from parallel subshells, so the
  git_lock || return 1      # `git add` must happen INSIDE the same lock as the
  local p                   # commit or groups cross-stage each other's files.
  for p in "$@"; do [ -e "$p" ] && git add "$p"; done
  if git diff --staged --quiet; then echo "  ($msg: nothing to commit)"; git_unlock; return 0; fi
  git -c user.name="Shawn Corvec" -c user.email="shawncorvec@hotmail.com" commit -m "$msg"
  local i
  for i in 1 2 3 4 5; do
    if git pull --rebase --autostash -X theirs && git push; then echo "  pushed: $msg (attempt $i)"; git_unlock; return 0; fi
    echo "  push attempt $i failed; retrying…"; sleep 5
  done
  echo "  ERROR: could not push: $msg (left as a local commit; the next run will carry it)"; git_unlock; return 1
}

# Priority prefetch starts FIRST (2026-07-25; was after Stage 1): its registry
# leads with the RMM u-file, so Stage 1's download_aifs shares that download via
# the per-file flock, and while the RMM computes the prefetch is already into
# the surface batches — no download dead-time during Stage-1 compute.
"$PY" ../ecmwf/store.py --date "$DATE" --time "$TIME" --prune-days 2 \
  > "$REPO/scripts/mjo/data/heavy_prefetch.log" 2>&1 &
PREFETCH_PID=$!
echo "priority prefetch started in background (pid $PREFETCH_PID)"

# ── Stage 1: RMM core — publish the MJO forecast first ──
if [ ! -f "$RMM_PNG" ]; then
  # Claim the cycle BEFORE the long u-fetch: mjo.yml's check job defers when it
  # sees a claim <3 h old for its target cycle, so the Actions fallback only
  # runs when this laptop is actually down — not merely slower than the cron
  # (2026-07-25: both raced the same slow ECMWF morning; the Action timed out).
  mkdir -p "$REPO/assets/mjo"
  printf '{"cycle":"%s","started":"%s"}\n' "$COMPACT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$REPO/assets/mjo/claim.json"
  ( cd "$REPO" && publish "MJO claim: ${COMPACT} (local)" assets/mjo/claim.json ) || true
  "$PY" src/download_aifs.py --date "$DATE" --time "$TIME" --out-dir data/aifs || { echo "RMM fetch failed"; exit 1; }
  "$PY" run_rmm.py --skip-download --date "$DATE" --time "$TIME" || { echo "RMM build failed"; exit 1; }
  mkdir -p "$REPO/assets/mjo"
  cp "plots/rmm_${COMPACT}.png" "$RMM_PNG"
  ls -t "$REPO"/assets/mjo/rmm_*z.png 2>/dev/null | tail -n +61 | xargs -r rm
else
  echo "RMM plot already present for ${COMPACT}; (re)publishing + resuming with the rest."
fi
# Build the page + publish RMM. Runs even when the plot already existed (e.g. an
# earlier run was interrupted before its commit); commit_push no-ops when clean.
( cd "$REPO" && "$PY" scripts/mjo/generate_page.py )
( cd "$REPO" && git add assets/mjo/ mjo.html scripts/mjo/data/reference/obs_history.nc \
    scripts/mjo/data/reference/ensmean_history.json \
    && commit_push "MJO RMM: ${COMPACT} (local)" )

# ── Stage 2: dependency-tracked PARALLEL product groups (2026-07-24) ──────────
# The old 2a/2b staging made every 200-hPa product wait for the full ~7 GB AAM
# pull even when it needed one small file (WAF: just v@200). Now ONE priority-
# ordered prefetch (store.registry(): rmm-u → sfc → step-0 winds → v@200 →
# heavy 14-level) publishes files smallest-critical-first, and product groups
# run as concurrent subshells. Every render self-fetches via store.ensure()
# under per-file flocks, so each group starts the moment ITS files land and
# never waits on data it doesn't use; the cross-process request budget
# (ecmwf/budget.py) keeps the shared connection polite. Groups commit through
# publish() (atomic stage+commit inside the git lock). The prefetch itself is
# launched before Stage 1 (see above) so it spans the RMM compute.

# group SFC — surface batches: TC + invests + MSLP/wind publish first, then
# Hovmöller + SOI (their 10u/msl series are extracted by ens_cycle light).
( "$PY" src/ens_cycle.py --date "$DATE" --time "$TIME" || echo "ens_cycle (light) had issues; continuing"
  "$PY" ../tc/tc_tracker.py --date "$DATE" --time "$TIME" \
    --out-dir "$REPO/assets/tc" || echo "TC tracker failed; continuing"
  "$PY" ../tc/invest_models.py --out-dir "$REPO/assets/tc" || echo "invest plotter failed; continuing"
  "$PY" src/mslp_wind_anim.py --date "$DATE" --time "$TIME" \
    --anim-dir "$REPO/assets/sst/anim/mslp_wind" \
    --manifest "$REPO/assets/sst/anim/mslp_wind_manifest.json" || echo "MSLP/wind anim failed; continuing"
  ( cd "$REPO" && publish "TC + MSLP/wind: ${COMPACT} (priority, local)" \
      assets/tc tc.html assets/sst/anim/mslp_wind assets/sst/anim/mslp_wind_manifest.json )
  "$PY" src/eq_hovmoller.py --date "$DATE" --time "$TIME" --data-dir data/u10 \
    --out "$REPO/assets/sst/eq_wind_hovmoller.webp" || echo "Hovmöller failed; continuing"
  "$PY" src/soi_forecast.py --date "$DATE" --time "$TIME" --data-dir data/msl \
    --out "$REPO/assets/sst/soi_forecast.webp" || echo "SOI failed; continuing"
  "$PY" src/soi_history.py --out "$REPO/assets/sst/soi_history.webp" \
    || echo "SOI history failed; continuing"
  echo "group SFC done" ) &
GRP_SFC=$!

# group V200 — 200 hPa velocity potential + WAF: u@200 is already cached from
# Stage 1; pf v@200 (~0.6 GB) arrives early in the prefetch order (or is
# fetched here on demand — same file, same flock).
( "$PY" src/wind200_vpot.py --date "$DATE" --time "$TIME" \
    --anim-dir "$REPO/assets/sst/anim/wind200" \
    --manifest "$REPO/assets/sst/anim/wind200_manifest.json" \
    --out "$REPO/assets/sst/wind200.webp" || echo "200hPa velocity potential failed; continuing"
  "$PY" src/waf.py --date "$DATE" --time "$TIME" \
    --anim-dir "$REPO/assets/sst/anim/waf" \
    --manifest "$REPO/assets/sst/anim/waf_manifest.json" \
    --out "$REPO/assets/sst/waf.webp" || echo "WAF failed; continuing"
  echo "group V200 done" ) &
GRP_V200=$!

# group ANALYSIS — Walker + MMSF: step-0 multi-level winds only (tiny files).
# MJO_RENDERER=julia (opt-in) rasterizes the frames via mjo_render.jl with
# per-frame matplotlib fallback — benchmarked 2026-07-24: matplotlib WINS here
# (64 MMSF frames: 20 s mpl vs 50 s julia; small cross-sections don't amortize
# the ~20 s JIT + staging overhead), so the default stays matplotlib.
( "$PY" src/walker.py --date "$DATE" --time "$TIME" \
    --anim-dir "$REPO/assets/sst/anim/walker" \
    --manifest "$REPO/assets/sst/anim/walker_manifest.json" \
    --out "$REPO/assets/sst/walker_anom.webp" || echo "Walker failed; continuing"
  "$PY" src/mmsf.py --date "$DATE" --time "$TIME" --data-dir data/mmsf \
    --anim-dir "$REPO/assets/sst/anim/mmsf" \
    --manifest "$REPO/assets/sst/anim/mmsf_manifest.json" \
    --out "$REPO/assets/sst/mmsf_anom.webp" || echo "MMSF failed; continuing"
  "$PY" "$REPO/scripts/strat/build_strat_products.py" \
    || echo "Stratosphere products failed; continuing"
  "$PY" "$REPO/scripts/strat/qbo_duct.py" --strip-only \
    || echo "QBO strip failed; continuing"
  "$PY" "$REPO/scripts/strat/wave1_monitor.py" \
    || echo "wave-1 monitor failed; continuing"
  echo "group ANALYSIS done" ) &
GRP_ANALYSIS=$!

# group NODATA — no ECMWF dependency at all: run immediately
( "$PY" ../spectra/ke_spectra.py --date "$DATE" --cycle "$TIME" --fxx 0 6 24 48 \
    --out "$REPO/assets/spectra/ke_spectra.webp" || echo "KE spectra failed; continuing"
  echo "group NODATA done" ) &
GRP_NODATA=$!

# group HEAVY (foreground) — AAM suite + torque + jets need the 14-level pull.
# MJO_HEAVY_ATMOS=1: this block was REVIVED 2026-07-18 after a full math audit
# (equator-row split, per-hour torque clim, NaN-band fix, MMSF clim-midpoint).
wait "$PREFETCH_PID" 2>/dev/null || echo "prefetch exited nonzero (see data/heavy_prefetch.log); ens_cycle will retry"
wait "$GRP_SFC" 2>/dev/null   # torque reads data/u10 + data/msl (extracted by group SFC)
MJO_HEAVY_ATMOS=1 "$PY" src/ens_cycle.py --date "$DATE" --time "$TIME" || echo "ens_cycle (heavy) had issues; continuing"
"$PY" src/aam.py --date "$DATE" --time "$TIME" --data-dir data/aam \
  --out "$REPO/assets/sst/aam.webp" || echo "AAM failed; continuing"
"$PY" "$REPO/scripts/strat/qbo_duct.py" --epflux-only \
  || echo "E-P flux ensemble loop failed; continuing"
"$PY" src/aam_zonal.py --date "$DATE" --time "$TIME" \
  --anim-dir "$REPO/assets/sst/anim/aam_zonal" \
  --manifest "$REPO/assets/sst/anim/aam_zonal_manifest.json" || echo "AAM zonal failed; continuing"
"$PY" src/aam_phase.py --date "$DATE" --time "$TIME" \
  --out "$REPO/assets/sst/aam_phase.webp" || echo "AAM phase failed; continuing"
"$PY" src/torque_map_anim.py --date "$DATE" --time "$TIME" --data-dir data/torque \
  --sp-dir data/aam --u10-dir data/u10 --msl-dir data/msl \
  --anim-dir "$REPO/assets/sst/anim/torque" \
  --manifest "$REPO/assets/sst/anim/torque_manifest.json" \
  --ts-out "$REPO/assets/sst/torque_timeseries.webp" \
  --ranges-out "$REPO/assets/sst/torque_ranges.webp" || echo "torque budget failed; continuing"
"$PY" src/jets.py --date "$DATE" --time "$TIME" \
  --out "$REPO/assets/sst/jets.webp" \
  --anim-dir "$REPO/assets/sst/anim/jets" \
  --manifest "$REPO/assets/sst/anim/jets_manifest.json" || echo "jets failed; continuing"
# product_stocks.py paused 2026-07-18 (card removed from fuelburn.html; script kept)

# barrier: every parallel group must land before the page-wide cache-bust and
# the consolidated commit below (SFC already waited above, harmless to repeat)
wait "$GRP_V200" "$GRP_ANALYSIS" "$GRP_NODATA" 2>/dev/null
echo "all product groups finished"

# 850 hPa wind analog Hovmöllers (current developing year vs 1982/97/2015). Refresh the
# current-year ARCO tail (1×/day, ~3 min) + re-render, once per calendar day. The WB2
# 1959-2023 band series is laptop-only (large, gitignored), so this whole block no-ops on
# the Action's clean checkout — the committed webps just carry over from the last local run.
U850_SERIES="data/reference/eq_u850_bandseries.nc"
U850_STAMP="data/.u850_analogs_day"
if [ -f "$U850_SERIES" ] && [ "$(cat "$U850_STAMP" 2>/dev/null)" != "$(date -u +%Y-%m-%d)" ]; then
  YR=$(date -u +%Y)
  if ARCO_HOURS=12 "$PY" src/build_u850_bandseries.py --source arco --start "$YR" --end "$YR" \
        --out "data/reference/eq_u850_${YR}_arco.nc" \
     && "$PY" src/eq_u850_analogs.py \
        --out-anom "$REPO/assets/sst/u850_analogs_anom.webp" \
        --out-abs  "$REPO/assets/sst/u850_analogs_abs.webp"; then
    date -u +%Y-%m-%d > "$U850_STAMP"
  else echo "u850 analogs failed; continuing"; fi
fi

# MEI.v2 daily nowcast — regression of the published bimonthly MEI.v2 onto daily Niño3.4
# (OISST) + SOI (DailySOI) + eq-u850 (the ARCO tail refreshed just above). Cheap (~seconds);
# laptop-only like the u850 block (needs the gitignored band series + the SST repo's OISST),
# so it no-ops on the Action's clean checkout and the committed webps carry over.
mkdir -p data/mei
for f in "nino.ascii|ersst5.nino.mth.91-20.ascii|NINO3.4|https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii" \
         "meiv2.data|meiv2.data|MEI|https://psl.noaa.gov/enso/mei/data/meiv2.data"; do
  IFS='|' read -r dst _ needle url <<< "$f"; tmp=$(mktemp)
  if curl -s --max-time 60 "$url" -o "$tmp" && grep -q "$needle" "$tmp"; then mv "$tmp" "data/mei/$dst"; else rm -f "$tmp"; fi
done
if [ -f data/reference/eq_u850_bandseries.nc ]; then
  VAL=""; [ -f "$REPO/assets/sst/mei/mei_validation.webp" ] || VAL="--validation"
  "$PY" src/build_mei_nowcast.py --apply "$(date -u +%Y)" $VAL || echo "MEI nowcast failed; continuing"
fi

# prune GRIBs older than a week. Archive hard-links share inodes with the working
# data dirs, so disk frees only when both links go — delete from both. Only *.grib2
# is matched, so committed reference *.nc files are never touched.
find "$MJO_GRIB_ARCHIVE" "$REPO/scripts/mjo/data" -name '*.grib2' -type f -mtime +7 -delete 2>/dev/null
find "$MJO_GRIB_ARCHIVE" -type d -empty -delete 2>/dev/null

# Refresh the cache-buster on the static SST-page <img>s this run rebuilt, so
# browsers re-fetch them. (sst.html's ?v= is otherwise only re-stamped by the SST
# builder, so MJO-updated images would serve stale from cache between SST runs;
# the torque/MMSF/Walker animator iframes self-bust via their manifest "ver".)
CB=$(date -u +%Y%m%d%H%M)
perl -0pi -e "s/((?:eq_wind_hovmoller|soi_forecast|soi_history|u850_analogs_anom|u850_analogs_abs|mei\/mei_nowcast|mei\/mei_validation)\.webp)\?v=\w+/\${1}?v=$CB/g" "$REPO/sst.html" 2>/dev/null || true
perl -0pi -e "s/((?:aam|aam_trend|aam_phase|mmsf_anom|walker_anom|jets|torque_timeseries|torque_ranges)\.webp)\?v=\w+/\${1}?v=$CB/g" "$REPO/enso-atmosphere.html" 2>/dev/null || true

# Stage whatever exists (a builder that failed this cycle simply hasn't written its
# outputs — that must not block committing everything else).
( cd "$REPO" && for p in \
    sst.html enso-atmosphere.html \
    assets/sst/eq_wind_hovmoller.webp assets/sst/soi_forecast.webp assets/sst/soi_history.webp \
    assets/sst/anim/mslp_wind assets/sst/anim/mslp_wind_manifest.json \
    assets/sst/anim/wind200 assets/sst/anim/wind200_manifest.json assets/sst/wind200.webp \
    assets/sst/aam.webp assets/sst/aam_trend.webp assets/sst/aam_phase.webp \
    assets/sst/anim/aam_zonal assets/sst/anim/aam_zonal_manifest.json \
    assets/sst/anim/torque assets/sst/anim/torque_manifest.json \
    assets/sst/torque_timeseries.webp assets/sst/torque_ranges.webp \
    assets/sst/anim/mmsf assets/sst/anim/mmsf_manifest.json assets/sst/mmsf_anom.webp \
    assets/sst/anim/walker assets/sst/anim/walker_manifest.json assets/sst/walker_anom.webp \
    assets/sst/jets.webp assets/sst/anim/jets assets/sst/anim/jets_manifest.json \
    assets/sst/anim/waf assets/sst/anim/waf_manifest.json assets/sst/waf.webp \
    assets/tc/anim assets/tc/storms assets/tc/tc_meta.json assets/tc/tracks.json tc.html \
    assets/tc/invests assets/tc/invests_meta.json assets/spectra/ke_spectra.webp \
    scripts/mjo/data/reference/mmsf_vbar_history.nc scripts/mjo/data/reference/walker_ud_history.nc \
    scripts/mjo/data/reference/aam_history.nc scripts/mjo/data/reference/aam_forecast_archive.nc \
    scripts/mjo/data/reference/mei_fit.json \
    assets/sst/u850_analogs_anom.webp assets/sst/u850_analogs_abs.webp \
    assets/sst/mei/mei_nowcast.webp assets/sst/mei/mei_validation.webp \
    assets/sst/mei/mei_analogs.webp assets/sst/mei/mei_history.webp \
    assets/sst/anim/strat10 assets/sst/anim/strat10_manifest.json \
    assets/sst/anim/strat100 assets/sst/anim/strat100_manifest.json \
  ; do [ -e "$p" ] && git add "$p"; done \
  ; commit_push "MJO atmospheric products (eq-wind/SOI + AAM/torque/MMSF/Walker/jets + MEI.v2): ${COMPACT} (local)" )

# Self-heal for IFS-ENS latency: the physics model (IFS-ENS) is disseminated ~1-2 h LATER
# than the AI model (AIFS-ENS), but this run triggers on AIFS availability — so the IFS leg
# of the Hovmöller/SOI can miss on the first attempt and render AIFS-only. Key the decision
# off whether the PRODUCTS actually included IFS (their "<out>.missing" sidecar), NOT mere
# cache presence: IFS can land mid-run AFTER the products already rendered AIFS-only, which
# would fool a cache check into marking done with stale products. If IFS is still missing
# from either product AND the cycle is young enough for IFS to land, DON'T mark done → the
# next hourly poll re-runs Stage 2 (mostly cached) and backfills IFS once it publishes. Give
# up (mark done, AIFS-only) once the cycle is old enough that IFS clearly isn't coming.
HOV_MISS="$REPO/assets/sst/eq_wind_hovmoller.webp.missing"
SOI_MISS="$REPO/assets/sst/soi_forecast.webp.missing"
INIT_EPOCH=$(date -j -u -f "%Y%m%d%H" "${DATE}${TIME}" +%s 2>/dev/null || echo 0)
AGE_H=$(( ( $(date -u +%s) - ${INIT_EPOCH:-0} ) / 3600 ))
if [ "${INIT_EPOCH:-0}" -gt 0 ] && [ "$AGE_H" -lt 10 ] \
   && { grep -qxF ifs "$HOV_MISS" 2>/dev/null || grep -qxF ifs "$SOI_MISS" 2>/dev/null; }; then
  echo "IFS-ENS ${COMPACT} not in the Hovmöller/SOI yet (cycle ${AGE_H}h old; rendered AIFS-only) — not marking done; a later poll will backfill IFS."
  exit 0
fi

touch "$DONE_MARKER"
ls -t data/.cycle_done_* 2>/dev/null | tail -n +9 | xargs -r rm   # keep last 8 markers
echo "cycle ${COMPACT} complete."
