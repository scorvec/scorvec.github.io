#!/bin/bash
# Local MJO + SST-page atmospheric build (primary path; the GitHub Action mjo.yml is
# the fallback for when the laptop is off). Two stages so the MJO/RMM forecast goes
# live within minutes instead of behind the heavy AAM/torque/MMSF downloads:
#   Stage 1  RMM core  — fetch u@200/850, build the RMM plot + page → COMMIT+PUSH.
#   Stage 2  the rest  — one consolidated ENS download (ens_cycle: 10u, msl, sp,
#                        u@13, 10v, v@13), then build Hovmöller/SOI/AAM/torque/MMSF
#                        from cache → COMMIT+PUSH.
# Downloads are watchdog-robust (download_aifs aborts+retries any stream that stalls),
# so a hung connection can no longer wedge the run. Idempotent: a `.cycle_done_<c>`
# marker skips a finished cycle; an interrupted run resumes at the unfinished stage
# (Stage-2 builders dedupe by init / re-render, so re-running is safe).
# Invoked twice daily by ~/Library/LaunchAgents/com.scorvec.mjo.plist.
set -uo pipefail

PY="${MJO_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export MPLBACKEND=Agg PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
export MJO_GRIB_ARCHIVE="${MJO_GRIB_ARCHIVE:-$HOME/mjo/grib_archive}"
cd "$REPO/scripts/mjo" || exit 1

LOG="$REPO/scripts/mjo/run_local.log"
exec >> "$LOG" 2>&1
echo "===================== $(date) ====================="

read -r DATE TIME < <("$PY" -c \
  "import sys; sys.path.insert(0,'src'); from download_aifs import latest_run; d,t=latest_run(); print(d,t)" \
  2>/dev/null)
if ! [[ "$DATE" =~ ^[0-9]{8}$ && "$TIME" =~ ^(00|12)$ ]]; then
  echo "could not determine cycle (DATE='$DATE' TIME='$TIME'); exiting"; exit 0
fi
COMPACT="${DATE}_${TIME}z"
DONE_MARKER="data/.cycle_done_${COMPACT}"
RMM_PNG="$REPO/assets/mjo/rmm_${COMPACT}.png"
if [ -f "$DONE_MARKER" ]; then echo "$COMPACT fully built — nothing to do."; exit 0; fi
echo "building cycle $COMPACT …"

commit_push () {            # $1 = message; commits staged changes (if any) + pushes with retry
  if git diff --staged --quiet; then echo "  ($1: nothing to commit)"; return 0; fi
  git -c user.name="Shawn Corvec" -c user.email="shawncorvec@hotmail.com" commit -m "$1"
  for i in 1 2 3 4 5; do
    if git pull --rebase --autostash -X theirs && git push; then echo "  pushed: $1 (attempt $i)"; return 0; fi
    echo "  push attempt $i failed; retrying…"; sleep 5
  done
  echo "  ERROR: could not push: $1 (left as a local commit; the next run will carry it)"; return 1
}

# ── Stage 1: RMM core — publish the MJO forecast first ──
if [ ! -f "$RMM_PNG" ]; then
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
    && commit_push "MJO RMM: ${COMPACT} (local)" )

# ── Stage 2: one consolidated download of the remaining ENS fields, then build + publish ──
"$PY" src/ens_cycle.py --date "$DATE" --time "$TIME" || echo "ens_cycle (rest) had issues; continuing"
"$PY" src/eq_hovmoller.py --date "$DATE" --time "$TIME" --data-dir data/u10 \
  --out "$REPO/assets/sst/eq_wind_hovmoller.webp" || echo "Hovmöller failed; continuing"
"$PY" src/soi_forecast.py --date "$DATE" --time "$TIME" --data-dir data/msl \
  --out "$REPO/assets/sst/soi_forecast.webp" || echo "SOI failed; continuing"
"$PY" src/aam.py --date "$DATE" --time "$TIME" --data-dir data/aam \
  --out "$REPO/assets/sst/aam.webp" || echo "AAM failed; continuing"
"$PY" src/aam_zonal.py --date "$DATE" --time "$TIME" --data-dir data/aam \
  --anim-dir "$REPO/assets/sst/anim/aam_zonal" \
  --manifest "$REPO/assets/sst/anim/aam_zonal_manifest.json" || echo "AAM zonal failed; continuing"
"$PY" src/torque_map_anim.py --date "$DATE" --time "$TIME" --data-dir data/torque \
  --sp-dir data/aam --u10-dir data/u10 --msl-dir data/msl \
  --anim-dir "$REPO/assets/sst/anim/torque" \
  --manifest "$REPO/assets/sst/anim/torque_manifest.json" \
  --ts-out "$REPO/assets/sst/torque_timeseries.webp" \
  --ranges-out "$REPO/assets/sst/torque_ranges.webp" || echo "torque budget failed; continuing"
"$PY" src/mmsf.py --date "$DATE" --time "$TIME" --data-dir data/mmsf \
  --anim-dir "$REPO/assets/sst/anim/mmsf" \
  --manifest "$REPO/assets/sst/anim/mmsf_manifest.json" \
  --out "$REPO/assets/sst/mmsf_anom.webp" || echo "MMSF failed; continuing"

# prune GRIBs older than a week. Archive hard-links share inodes with the working
# data dirs, so disk frees only when both links go — delete from both. Only *.grib2
# is matched, so committed reference *.nc files are never touched.
find "$MJO_GRIB_ARCHIVE" "$REPO/scripts/mjo/data" -name '*.grib2' -type f -mtime +7 -delete 2>/dev/null
find "$MJO_GRIB_ARCHIVE" -type d -empty -delete 2>/dev/null

# Refresh the cache-buster on the static SST-page <img>s this run rebuilt, so
# browsers re-fetch them. (sst.html's ?v= is otherwise only re-stamped by the SST
# builder, so MJO-updated images would serve stale from cache between SST runs;
# the torque/MMSF animator iframes self-bust via their manifest "ver".)
CB=$(date -u +%Y%m%d%H%M)
perl -0pi -e "s/((?:aam|aam_trend|torque_timeseries|torque_ranges|eq_wind_hovmoller|soi_forecast)\.webp)\?v=\d+/\${1}?v=$CB/g" "$REPO/sst.html" 2>/dev/null || true

( cd "$REPO" && git add \
    scripts/mjo/data/reference/aam_history.nc scripts/mjo/data/reference/aam_forecast_archive.nc \
    scripts/mjo/data/reference/mmsf_vbar_history.nc sst.html \
    assets/sst/eq_wind_hovmoller.webp assets/sst/soi_forecast.webp assets/sst/aam.webp assets/sst/aam_trend.webp \
    assets/sst/anim/torque/ assets/sst/anim/torque_manifest.json assets/sst/torque_timeseries.webp assets/sst/torque_ranges.webp \
    assets/sst/anim/mmsf/ assets/sst/anim/mmsf_manifest.json assets/sst/mmsf_anom.webp \
    assets/sst/anim/aam_zonal/ assets/sst/anim/aam_zonal_manifest.json \
    && commit_push "MJO atmospheric products (wind/SOI + AAM/torque/MMSF/zonal): ${COMPACT} (local)" )

touch "$DONE_MARKER"
ls -t data/.cycle_done_* 2>/dev/null | tail -n +9 | xargs -r rm   # keep last 8 markers
echo "cycle ${COMPACT} complete."
