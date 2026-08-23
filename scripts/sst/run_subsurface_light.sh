#!/bin/bash
# Light subsurface-only refresh — runs from the 15-min SST poll OUTSIDE the
# 09:00-13:59 ET OISST window, at most once every 3 h. TAO and the Copernicus
# current sections advance daily and independently of OISST (~4-day PSL lag),
# so gating them purely on the OISST window left the subsurface page days
# stale (user report 2026-08-16). Shares the main runner's single-instance
# lock; commits only the subsurface/current products it owns.
set -uo pipefail
PY="${SST_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export MPLBACKEND=Agg SST_SITE_ROOT="$REPO" \
       PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
cd "$REPO" || exit 1

STAMP="$REPO/scripts/sst/data/.subsurface_light_run"
if [ -n "$(find "$STAMP" -mmin -175 2>/dev/null)" ]; then exit 0; fi

LOG="$REPO/scripts/sst/run_local_sst.log"
exec >> "$LOG" 2>&1
echo "----- subsurface light pass $(date) -----"

source "$REPO/scripts/lib/gitlock.sh"
require_main || exit 0

LOCK="$REPO/scripts/sst/.run.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  owner="$(cat "$LOCK/pid" 2>/dev/null)"
  if { [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; } \
     || [ -n "$(find "$LOCK" -maxdepth 0 -mmin -1 2>/dev/null)" ]; then
    echo "SST run in progress — skipping light pass"; exit 0
  fi
  rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || exit 0
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK" 2>/dev/null' EXIT

"$PY" scripts/sst/tao_subsurface.py --days 120 \
  --out scripts/sst/data/tao_eq_recent.nc \
  --ascii scripts/sst/data/tao_eq_recent.ascii || echo "TAO failed; continuing"
"$PY" scripts/sst/sst_subsurface.py || echo "subsurface failed; continuing"
( cd scripts/sst && "$PY" eq_current_section.py ) || echo "eq current section failed; continuing"
( cd scripts/sst && "$PY" eq_current_hovmoller.py ) || echo "eq current hovmoller failed; continuing"
( cd scripts/sst && "$PY" eq_uwind_hovmoller.py ) || echo "eq uwind hovmoller failed; continuing"
( cd scripts/sst && "$PY" eq_current_map.py ) || echo "eq current map failed; continuing"

touch "$STAMP"

# ── hydro tick: forecasts follow the GRIB cycles, not the OISST window ──────
# engines + fan charts whenever a newer tp GRIB exists than last tick; the
# daily PDF fires on the first tick of a new UTC day (~00-03Z).
HTICK="$REPO/scripts/sst/data/.hydro_tick"
NEWEST=$(ls -t "$REPO"/scripts/mjo/data/aifs/*tp.grib2 2>/dev/null | head -1)
if [ -n "$NEWEST" ] && [ "$NEWEST" -nt "$HTICK" ]; then
  echo "hydro tick: new NWP cycle $(basename "$NEWEST")"
  perl -e 'alarm shift; exec @ARGV' 1800 "$PY" scripts/sst/colombia_forecast.py || echo "colombia engine failed; continuing"
  perl -e 'alarm shift; exec @ARGV' 1800 "$PY" scripts/sst/brazil_forecast.py || echo "brazil engine failed; continuing"
  "$PY" scripts/sst/xm_inflow_history.py || echo "inflow chart failed; continuing"
  "$PY" scripts/sst/brazil_model.py || echo "brazil models failed; continuing"
  "$PY" scripts/sst/brazil_rain_chart.py || echo "brazil rain charts failed; continuing"
  touch "$HTICK"
fi
RPT_STAMP="$HOME/.colombia_report_day"
if [ "$(date -u +%F)" != "$(cat "$RPT_STAMP" 2>/dev/null)" ]; then
  perl -e 'alarm shift; exec @ARGV' 1800 "$PY" scripts/sst/daily_report.py \
    && date -u +%F > "$RPT_STAMP" || echo "daily report failed; continuing"
fi

for p in assets/sst/data/tao_section.json \
         assets/sst/equatorial_xsection.webp \
         assets/sst/anim/equatorial assets/sst/anim/manifest.json \
         assets/sst/eq_current_hov.webp assets/sst/eq_uwind_hov.webp \
         assets/sst/anim/eq_cur_section assets/sst/anim/eq_cur_section_manifest.json \
         assets/sst/anim/eq_cur_map assets/sst/anim/eq_cur_map_manifest.json \
  ; do [ -e "$p" ] && git add "$p"; done
# hydro pages live in the private repos now (symlinked); snapshot them there
for d in "$HOME/colombia_hydro" "$HOME/brazil_hydro"; do
  ( cd "$d" && git add -A site raw out 2>/dev/null \
    && git commit -q -m "site/data tick $(date -u +%FT%H:%MZ)" 2>/dev/null \
    && git push -q ) || true
done
if git diff --staged --quiet; then echo "light pass: no changes"; exit 0; fi
trap 'git_unlock; rm -rf "$LOCK" 2>/dev/null' EXIT
git_lock || { echo "git lock busy; leaving as a local commit"; exit 0; }
git -c user.name="Shawn Corvec" -c user.email="scorvec@outlook.com" \
    commit -m "Subsurface refresh (TAO + eq currents): $(date -u +%Y-%m-%dT%H:%MZ) (light)"
for i in 1 2 3; do
  if git pull --rebase --autostash -X theirs origin main && git push; then echo "light pushed (attempt $i)"; git_unlock; exit 0; fi
  git_rebase_rescue   # finish the rebase past frame-count conflicts, else abort clean
  sleep 5
done
echo "light pass: push failed"; git_unlock; exit 1
