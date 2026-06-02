#!/bin/bash
# Local MJO build (primary path; the GitHub Action is the fallback for when the
# laptop is off). Fetches the latest AIFS/IFS-ENS cycle, builds the RMM plot +
# page + equatorial wind Hovmöller + SOI forecast, then commits & pushes.
#
# Idempotent: no-ops if the latest cycle's plot is already committed (same
# sentinel as the Action), so running it repeatedly / alongside the Action is
# safe. Invoked twice daily by ~/Library/LaunchAgents/com.scorvec.mjo.plist.
set -uo pipefail

PY="${MJO_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export MPLBACKEND=Agg PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
# keep a copy of every GRIB we download, organised by cycle, for easy local access
export MJO_GRIB_ARCHIVE="${MJO_GRIB_ARCHIVE:-$HOME/mjo/grib_archive}"
cd "$REPO/scripts/mjo" || exit 1

LOG="$REPO/scripts/mjo/run_local.log"
exec >> "$LOG" 2>&1
echo "===================== $(date) ====================="

# latest available cycle (probe is tiny; stderr/banner suppressed)
read -r DATE TIME < <("$PY" -c \
  "import sys; sys.path.insert(0,'src'); from download_aifs import latest_run; d,t=latest_run(); print(d,t)" \
  2>/dev/null)
if ! [[ "$DATE" =~ ^[0-9]{8}$ && "$TIME" =~ ^(00|12)$ ]]; then
  echo "could not determine cycle (DATE='$DATE' TIME='$TIME'); exiting"; exit 0
fi
COMPACT="${DATE}_${TIME}z"
if [ -f "$REPO/assets/mjo/rmm_${COMPACT}.png" ]; then
  echo "$COMPACT already built — nothing to do."; exit 0
fi
echo "building cycle $COMPACT …"

"$PY" src/ens_cycle.py --date "$DATE" --time "$TIME" || { echo "fetch failed"; exit 1; }
"$PY" run_rmm.py --skip-download --date "$DATE" --time "$TIME" || { echo "RMM failed"; exit 1; }

mkdir -p "$REPO/assets/mjo"
cp "plots/rmm_${COMPACT}.png" "$REPO/assets/mjo/rmm_${COMPACT}.png"
ls -t "$REPO"/assets/mjo/rmm_*z.png 2>/dev/null | tail -n +61 | xargs -r rm
( cd "$REPO" && "$PY" scripts/mjo/generate_page.py )
"$PY" src/eq_hovmoller.py --date "$DATE" --time "$TIME" --data-dir data/u10 \
  --out "$REPO/assets/sst/eq_wind_hovmoller.webp" || echo "Hovmöller failed; continuing"
"$PY" src/soi_forecast.py --date "$DATE" --time "$TIME" --data-dir data/msl \
  --out "$REPO/assets/sst/soi_forecast.webp" || echo "SOI failed; continuing"

# AAM is a heavy ~3 GB (13-level) download — once daily, on the 00Z cycle only.
if [ "$TIME" = "00" ]; then
  "$PY" src/aam.py --date "$DATE" --time "$TIME" --data-dir data/aam \
    --out "$REPO/assets/sst/aam.webp" || echo "AAM failed; continuing"
  # ensemble mean — reuses sp (data/aam), 10u (data/u10), msl (data/msl) already
  # downloaded above; fetches only 10v into data/torque. Must run after aam.py
  # (sp + AAM archive for the dM/dt overlay), eq_hovmoller (10u) and soi (msl).
  "$PY" src/torque_map_anim.py --date "$DATE" --time "$TIME" --data-dir data/torque \
    --sp-dir data/aam --u10-dir data/u10 --msl-dir data/msl \
    --anim-dir "$REPO/assets/sst/anim/torque" \
    --manifest "$REPO/assets/sst/anim/torque_manifest.json" \
    --ts-out "$REPO/assets/sst/torque_timeseries.webp" || echo "torque budget failed; continuing"
fi

# prune GRIBs older than a week. The archive hard-links share inodes with the
# working data dirs, so disk is freed only when both links go — delete from both.
# Only *.grib2 is matched, so committed reference *.nc files are never touched.
find "$MJO_GRIB_ARCHIVE" "$REPO/scripts/mjo/data" -name '*.grib2' -type f -mtime +7 -delete 2>/dev/null
find "$MJO_GRIB_ARCHIVE" -type d -empty -delete 2>/dev/null

cd "$REPO"
git add assets/mjo/ mjo.html scripts/mjo/data/reference/obs_history.nc \
        scripts/mjo/data/reference/aam_history.nc scripts/mjo/data/reference/aam_forecast_archive.nc \
        assets/sst/eq_wind_hovmoller.webp assets/sst/soi_forecast.webp assets/sst/aam.webp assets/sst/aam_trend.webp \
        assets/sst/anim/torque/ assets/sst/anim/torque_manifest.json assets/sst/torque_timeseries.webp
if git diff --staged --quiet; then echo "no changes to commit"; exit 0; fi
git -c user.name="Shawn Corvec" -c user.email="shawncorvec@hotmail.com" \
    commit -m "MJO RMM: ${COMPACT} (local)"
for i in 1 2 3 4 5; do
  if git pull --rebase -X theirs && git push; then echo "pushed (attempt $i)"; exit 0; fi
  echo "push attempt $i failed; retrying…"; sleep 5
done
echo "ERROR: could not push after 5 attempts."; exit 1
