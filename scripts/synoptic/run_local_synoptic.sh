#!/bin/bash
# Local synoptic-maps build (primary path; the GitHub Action synoptic-maps.yml is
# the fallback for when the laptop is off). Renders the HRRR F00–F48 maps (5 vars ×
# 5 regions) for the latest extended cycle, then commits & pushes only if there's a
# new cycle whose wind/solar plant overlays are already present.
#
# Why local-primary: on CI the render is capped at 2 workers (7 GB RAM) behind a
# 90-min timeout and a queue, so maps land ~5–8 h after cycle init. This laptop has
# 18 cores, so render_maps runs all-cores and finishes in minutes.
#
# Idempotent: gated on (new cycle) AND (wind+solar capacity CSVs committed), and it
# compares against the already-rendered cycle in the national wind manifest — so it
# no-ops cheaply when there's nothing to do and is safe to poll every ~30 min and to
# run alongside the Action (whoever lands the cycle first wins; the other no-ops).
# Invoked by ~/Library/LaunchAgents/com.scorvec.synoptic.plist.
set -uo pipefail

PY="${SYNOPTIC_PY:-/opt/homebrew/Caskroom/miniconda/base/envs/wx/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export MPLBACKEND=Agg \
       PATH="$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
cd "$REPO" || exit 1

LOG="$REPO/scripts/synoptic/run_local_synoptic.log"
exec >> "$LOG" 2>&1
echo "===================== $(date) ====================="

# Pull first so we see the latest wind/solar overlay CSVs + rendered manifest that
# the upstream Actions may have just committed (avoids re-rendering what's done).
if ! git pull --rebase -X theirs >/dev/null 2>&1; then
  echo "git pull failed (dirty tree or conflict?); skipping this poll."; exit 0
fi

# Latest 00/06/12/18Z extended cycle (latency-based, no network).
ISO=$("$PY" scripts/find_latest_extended_cycle.py --format iso) || { echo "cycle probe failed"; exit 0; }
COMPACT=$("$PY" scripts/find_latest_extended_cycle.py --format compact)
DATE="${ISO%T*}"; HH="${ISO#*T}"; HOUR="${HH%:*}"     # 2026-06-02 / 18

# Already rendered? (national wind manifest carries the rendered cycle_compact)
RENDERED=$("$PY" -c "import json,sys;\
print(json.load(open('assets/synoptic/wind/national/manifest.json')).get('cycle_compact',''))" 2>/dev/null || echo "")
if [ "$RENDERED" = "$COMPACT" ]; then
  echo "maps already current for $COMPACT — nothing to do."; exit 0
fi

# Require the plant overlays before rendering, so we never render a cycle's maps
# WITHOUT the rings (render_maps deliberately skips the raw inventory). Wind capacity
# is a canonical static-fleet file; solar's per-cycle signal is its forecast_plant.
# A missing file means the wind/solar runs haven't committed yet — skip; the next
# poll renders once they land.
WIND_CSV="assets/wind_forecast_data/capacity_plant.csv"
SOLAR_CSV="assets/solar_forecast_data/forecast_plant_${COMPACT}.csv"
if [ ! -f "$WIND_CSV" ] || [ ! -f "$SOLAR_CSV" ]; then
  echo "new cycle $COMPACT but plant overlays not ready (wind:$([ -f "$WIND_CSV" ] && echo y || echo n) solar:$([ -f "$SOLAR_CSV" ] && echo y || echo n)) — skipping."
  exit 0
fi

echo "rendering synoptic maps for $COMPACT  ($DATE ${HOUR}Z) on $(sysctl -n hw.ncpu) cores …"

# Drop stale outputs (PNG→WebP migration, old 9-region layout) for parity with CI.
find assets/synoptic -name "*.png" -delete 2>/dev/null || true
for var in wind solar reflectivity visibility ceiling; do
  for old in caiso spp ercot miso pjm newengland bpa nyiso isone se; do
    rm -rf "assets/synoptic/$var/$old" 2>/dev/null || true
  done
done

( cd "$REPO/scripts/synoptic" && PYTHONPATH="$REPO/scripts/synoptic" \
    "$PY" render_maps.py "$DATE" "$HOUR" --workers 0 ) \
  || { echo "render_maps failed; leaving maps untouched."; exit 1; }

# Unified viewer page + stamp the cycle into charts.html (matches the Action).
cp "$REPO/scripts/synoptic/viewer.html" "$REPO/assets/synoptic/viewer.html"
if [ -f "$REPO/charts.html" ]; then
  perl -0pi -e "s|(id=\"last-updated-synoptic\">Cycle )[^<]*|\${1}${COMPACT}|g" "$REPO/charts.html"
fi

git add -A assets/synoptic/
git add charts.html 2>/dev/null || true
if git diff --staged --quiet; then echo "render produced no changes; nothing to commit."; exit 0; fi
git -c user.name="Shawn Corvec" -c user.email="shawncorvec@hotmail.com" \
    commit -m "synoptic maps: cycle ${COMPACT} (local)"
for i in 1 2 3 4 5; do
  if git pull --rebase -X theirs && git push; then echo "pushed (attempt $i)"; exit 0; fi
  echo "push attempt $i failed; retrying…"; sleep 5
done
echo "ERROR: could not push after 5 attempts."; exit 1
