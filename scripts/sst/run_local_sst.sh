#!/bin/bash
# Local SST/RONI build (primary path; the GitHub Action sst.yml is the fallback
# for when the laptop is off). Rebuilds the OISST anomaly maps + RONI + TAO
# subsurface + ASCAT winds, then commits & pushes only if something changed.
#
# Idempotent: commits only when `git diff` is non-empty, so running it repeatedly
# / alongside the Action is safe (whoever lands the new OISST day first wins; the
# other run no-ops). Invoked daily by ~/Library/LaunchAgents/com.scorvec.sst.plist.
#
# OISST/PSL files are CACHED and only re-downloaded when PSL publishes newer data
# (sst-roni HEAD-checks Last-Modified), so the 4-hourly polls don't re-pull the full
# ~240 MB annual file each time — just when a new day actually lands. The GitHub
# Action runs from a clean checkout, so it has no cache to reuse.
set -uo pipefail

# Scheduled poll window: the launchd agent fires every 15 min (StartInterval) and passes
# --poll; only actually check during 09:00–13:59 ET (OISST typically posts ~midday ET), so
# the other fires exit instantly. Manual runs (no --poll) always proceed.
if [ "${1:-}" = "--poll" ]; then
  H=$(TZ=America/New_York date +%H); H=$((10#$H))
  if [ "$H" -lt 9 ] || [ "$H" -ge 14 ]; then
    # outside the OISST window: TAO + the Copernicus current sections advance
    # daily regardless of OISST — hand off to the light subsurface refresh
    # (self-gated to once per ~3 h) instead of going fully idle
    exec /bin/bash "$(dirname "$0")/run_subsurface_light.sh"
  fi
fi

PY="${SST_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export MPLBACKEND=Agg SST_SITE_ROOT="$REPO" \
       PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
cd "$REPO" || exit 1

LOG="$REPO/scripts/sst/run_local_sst.log"
exec >> "$LOG" 2>&1
echo "===================== $(date) ====================="
# Only ever render+commit from main (a stray feature-branch checkout once captured a
# whole day of SST/synoptic/MJO commits). Source the lock lib early just for this guard;
# it's re-sourced below where the git critical section actually runs.
source "$REPO/scripts/lib/gitlock.sh"
require_main || exit 0

# Single-instance lock: a slow run (TAO/OISST fetch) must not overlap the next scheduled fire —
# concurrent renders + the push-retry `git reset --hard` race and wipe each other's in-flight
# frames. mkdir is atomic; the lock records its owner PID, so an orphaned lock (a run killed or
# slept without firing its EXIT trap) is reclaimed at once instead of blocking for hours.
LOCK="$REPO/scripts/sst/.run.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  owner="$(cat "$LOCK/pid" 2>/dev/null)"
  if { [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; } \
     || [ -n "$(find "$LOCK" -maxdepth 0 -mmin -1 2>/dev/null)" ]; then   # owner alive, or <1 min (PID not yet written)
    echo "another SST run in progress (pid ${owner:-?}) — skipping this fire"; exit 0
  fi
  echo "stale lock (owner pid ${owner:-none} not running) — taking over"
  rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || { echo "could not acquire lock"; exit 0; }
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK" 2>/dev/null' EXIT

DAY_Q='import json,os;p="assets/sst/manifest.json";print(json.load(open(p)).get("sst_valid_day","") if os.path.exists(p) else "")'
PREV_DAY=$("$PY" -c "$DAY_Q" 2>/dev/null || echo "")
"$PY" scripts/sst/sst-roni.py || { echo "sst-roni failed (retrying once)"; \
  sleep 30; "$PY" scripts/sst/sst-roni.py || echo "sst-roni failed twice; continuing — downstream uses caches"; }
NEW_DAY=$("$PY" -c "$DAY_Q" 2>/dev/null || echo "")
"$PY" scripts/sst/tao_subsurface.py --days 120 \
  --out scripts/sst/data/tao_eq_recent.nc \
  --ascii scripts/sst/data/tao_eq_recent.ascii || echo "TAO failed; continuing"
"$PY" scripts/sst/sst_subsurface.py || echo "subsurface failed; continuing"
"$PY" scripts/sst/wwv_orbit.py || echo "WWV orbit failed; continuing"
( cd scripts/sst && SST_SITE_ROOT="$REPO" "$PY" sst_ascat_winds.py ) || echo "ASCAT failed; continuing"
# perl-alarm wall-clock caps: a hung Earthdata connection once pinned imerg_gatun
# for 18 h, and the run lock then starved every later poll (2026-08-14/15). The
# alarm SIGALRMs the whole step; `|| echo` keeps the chain moving as usual.
perl -e 'alarm shift; exec @ARGV' 5400 "$PY" scripts/sst/imerg_precip.py || echo "IMERG precip failed; continuing"   # NASA GPM IMERG, ~/.netrc Earthdata auth
perl -e 'alarm shift; exec @ARGV' 1800 "$PY" scripts/sst/imerg_precip_anom.py || echo "IMERG precip anomaly failed; continuing"   # vs the committed 20-yr clim
perl -e 'alarm shift; exec @ARGV' 3600 "$PY" scripts/sst/imerg_gatun.py || echo "IMERG Gatun tracker failed; continuing"   # Lake Gatun zoom + rain-vs-level chart
"$PY" scripts/gatun/fetch_data.py || echo "Gatun dashboard data failed; continuing"   # gatun/data.js: ACP levels/projection + ONI
"$PY" scripts/sst/hydro_region_rain.py || echo "XM region rainfall failed; continuing"   # basin rain over XM hydro regions
"$PY" scripts/sst/xm_storage.py || echo "XM storage failed; continuing"   # reservoir storage norms + outflow model (state for the fan)
"$PY" scripts/sst/xm_generation.py || echo "XM generation failed; continuing"   # actual hydro gen history + gen model fit (draws gen fan)
"$PY" scripts/sst/xm_load.py || echo "XM load failed; continuing"   # national demand history + temp link
"$PY" scripts/sst/ons_data.py || echo "ONS data failed; continuing"   # Brazil ENA/EAR daily + charts
"$PY" scripts/sst/brazil_gauges.py || echo "Brazil gauges failed; continuing"   # INMET day cache (monthly zip refresh)
"$PY" scripts/sst/brazil_correction.py || echo "Brazil correction failed; continuing"   # gauge-corrected IMERG field + bias figs
"$PY" scripts/sst/brazil_model.py || echo "Brazil models failed; continuing"   # rain->ENA kernels (draws fans)
perl -e 'alarm shift; exec @ARGV' 1800 "$PY" scripts/sst/brazil_forecast.py || echo "Brazil forecast failed; continuing"   # ENA fans
"$PY" scripts/sst/brazil_rain_chart.py || echo "Brazil rain charts failed; continuing"   # rain fans + skill-corrected map
"$PY" scripts/sst/nwp_bias_leads.py || echo "bias-by-lead failed; continuing"   # AIFS/IFS bias curves, both countries
"$PY" scripts/sst/dam_models.py || echo "Dam models failed; continuing"   # per-dam catchment kernels + states (draws dam fans)
perl -e 'alarm shift; exec @ARGV' 1800 "$PY" scripts/sst/colombia_forecast.py || echo "Colombia forecast failed; continuing"   # AIFS+IFS rain -> inflow + storage fans (rides MJO tp GRIBs)
"$PY" scripts/sst/xm_inflow_history.py || echo "XM inflow norms failed; continuing"   # inflow history + seasonal norms (draws the fan)
"$PY" scripts/sst/colombia_rain_map.py || echo "Colombia rain map failed; continuing"   # IMERG vs IDEAM gauges
# daily PDF briefing: once per UTC day, first runner pass after 00Z
RPT_STAMP="$HOME/.colombia_report_day"
if [ "$(date -u +%F)" != "$(cat "$RPT_STAMP" 2>/dev/null)" ]; then
  perl -e 'alarm shift; exec @ARGV' 1800 "$PY" scripts/sst/daily_report.py \
    && date -u +%F > "$RPT_STAMP" || echo "daily report failed; continuing"
  # daily data snapshot into the private research repo (handoff freshness)
  ( cd "$HOME/colombia_hydro" && git add -A raw reports out 2>/dev/null \
    && git commit -q -m "data: daily snapshot $(date -u +%F)" 2>/dev/null \
    && git push -q ) || true
fi
( cd scripts/sst && "$PY" eq_current_section.py ) || echo "eq current section failed; continuing"
( cd scripts/sst && "$PY" eq_current_hovmoller.py ) || echo "eq current hovmoller failed; continuing"
( cd scripts/sst && "$PY" eq_uwind_hovmoller.py ) || echo "eq uwind hovmoller failed; continuing"
( cd scripts/sst && "$PY" eq_current_map.py ) || echo "eq current map failed; continuing"

# Analog comparison charts (current vs 1997/2015/2023 El Niño) — rebuild ONCE PER
# CALENDAR DAY (or whenever OISST advances). The subsurface analog tracks TAO, which
# can advance independently of OISST (OISST has ~1–2 day PSL latency), so gating purely
# on OISST left the subsurface cross-section a day stale. A per-day stamp keeps the
# 4-hourly polls cheap (only the first poll of a new day rebuilds) while ensuring the
# analogs refresh every day. sst_events loads the ~3.8 GB cached analog-year files.
ANALOG_STAMP="$REPO/scripts/sst/data/.analog_built_day"
TODAY_UTC=$(date -u +%Y-%m-%d)
LAST_ANALOG=$(cat "$ANALOG_STAMP" 2>/dev/null || echo "")
if [ "$TODAY_UTC" != "$LAST_ANALOG" ] || { [ -n "$NEW_DAY" ] && [ "$NEW_DAY" != "$PREV_DAY" ]; }; then
  echo "rebuilding analog charts (day ${LAST_ANALOG:-none}→$TODAY_UTC; OISST ${PREV_DAY:-none}→${NEW_DAY:-none})"
  ok=1
  SST_SITE_ROOT="$REPO" "$PY" scripts/sst/sst_events.py || { echo "sst_events failed; continuing"; ok=0; }
  ( cd scripts/sst && SST_SITE_ROOT="$REPO" "$PY" sst_subsurface_events.py ) || { echo "subsurface analog failed; continuing"; ok=0; }
  # Monthly Niño-region history JSON for the interactive analog explorer (CPC ERSSTv5;
  # changes ~monthly, so once-a-day is ample). Non-fatal — a CPC hiccup keeps the old JSON.
  SST_SITE_ROOT="$REPO" "$PY" scripts/sst/build_nino_history.py || echo "nino history failed; keeping previous JSON"
  # Atmospheric fingerprint maps (ERA5 monthly, CDS): re-request only when a
  # newer month should exist, so this is a no-op most days.
  SST_SITE_ROOT="$REPO" "$PY" scripts/sst/analog_atmos.py || echo "analog atmos failed; keeping previous maps"
  [ "$ok" = 1 ] && echo "$TODAY_UTC" > "$ANALOG_STAMP"     # stamp only on full success → retry next poll otherwise
fi

git add sst.html enso-*.html assets/sst/ gatun/data.js colombia_hydro
if git diff --staged --quiet; then echo "no changes to commit"; exit 0; fi
DAY=$("$PY" -c "import json; print(json.load(open('assets/sst/manifest.json'))['sst_valid_day'])" 2>/dev/null)
source "$REPO/scripts/lib/gitlock.sh"
trap 'git_unlock; rm -rf "$LOCK" 2>/dev/null' EXIT   # both cleanups (this trap replaces the lock-only one above)
git_lock || { echo "git lock busy; leaving as a local commit for the next run"; exit 0; }
git -c user.name="Shawn Corvec" -c user.email="scorvec@outlook.com" \
    commit -m "SST/RONI update: OISST ${DAY} (local)"
for i in 1 2 3 4 5; do
  if git pull --rebase --autostash -X theirs && git push; then echo "pushed (attempt $i)"; git_unlock; exit 0; fi
  echo "push attempt $i failed; retrying…"; sleep 5
done
echo "ERROR: could not push after 5 attempts."; git_unlock; exit 1
