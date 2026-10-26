#!/bin/bash
# Build the GEPS days 1-35 page (subseasonal.html) from the newest ECCC GEPS extended cycle. Runs in Actions
# (.github/workflows/geps.yml).
#
#     scripts/geps/run_geps.sh                 # today's Mon/Thu cycle (waits for it), else the newest one
#     WANT=20261001 scripts/geps/run_geps.sh   # a specific cycle
#     FORCE=1 ...                              # rebuild a cycle already published
#     GEPS_OFFLINE=1 WANT=... ...              # no network at all: rebuild from files already held
#
# Expects the reference set in $GEPS_REF (release geps-ref-v3, scripts/geps/make_reference.py) and the state carried
# between runs in $GEPS_STATE (frames branch). Writes the page assets into assets/geps and the frames into
# assets/geps/anim; publishing (frames branch, main) is the workflow's job. Sets new=1 in $GITHUB_OUTPUT when it built.
#
# Order matters: the anomalies must exist before anything is drawn, and the OLR channel must be refreshed before the
# MJO projection reads it. The OLR channel is refreshed while the forecast is fetched, and the products after the maps
# run at the same time (scripts/lib/parallel.sh: each from the anomalies and members on disk, none from another's
# output); every product after forecast.py warns (degraded.sh) rather than failing the run.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SITE="$(cd "$HERE/../.." && pwd)"
OUTROOT="${SITE_ROOT:-$SITE}"          # where assets/geps is written (a scratch copy for checks)
export SITE_ROOT="$OUTROOT"
PY="${PY:-python}"
cd "$HERE" || exit 1
export GEPS_REF="${GEPS_REF:-$HERE/ref}" GEPS_STATE="${GEPS_STATE:-$HERE/state}" GEPS_WORK="${GEPS_WORK:-$HERE/work}"
export STRAT_DL="${STRAT_DL:-$GEPS_WORK/vortex_dl}" STRAT_SERIES_DIR="${STRAT_SERIES_DIR:-$GEPS_STATE/strat_series}"
WORKERS="${RENDER_WORKERS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}"
out() { [ -n "${GITHUB_OUTPUT:-}" ] && echo "$1" >> "$GITHUB_OUTPUT"; return 0; }
step() { echo "::group::$1"; local t0=$SECONDS; shift; "$@"; local rc=$?; echo "  ($((SECONDS - t0)) s, exit $rc)"; echo "::endgroup::"; return $rc; }
mkdir -p "$GEPS_WORK" "$GEPS_STATE" "$OUTROOT/assets/geps/anim"
[ -f "$GEPS_REF/geps8_clim_t2m.npz" ] || { echo "::error::no reference set in $GEPS_REF"; exit 1; }

# Which cycle. Extended (day-35) cycles run Monday and Thursday; the day-35 step lands ~06:45-06:52 UTC. On a
# Mon/Thu wait for TODAY's cycle rather than taking whatever is newest (that would silently rebuild the previous run).
latest() { "$PY" -c 'import forecast; print(forecast.latest_extended() or "")'; }
if [ -n "${WANT:-}" ]; then
  CYCLE="$WANT"
else
  DOW=$(date -u +%u)
  if [ "$DOW" = 1 ] || [ "$DOW" = 4 ]; then
    WANT=$(date -u +%Y%m%d)
    DEADLINE=$(( $(date +%s) + ${WAIT_MAX:-7200} ))
    while :; do
      CYCLE=$(latest)
      [ "$CYCLE" = "$WANT" ] && break
      if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "$WANT extended cycle not published (newest ${CYCLE:-none})"; break
      fi
      echo "  waiting for the $WANT extended cycle (newest ${CYCLE:-none})"; sleep 600
    done
  else
    CYCLE=$(latest)
  fi
fi
[ -n "$CYCLE" ] || { echo "no extended cycle found"; exit 0; }
STAMP="$GEPS_STATE/last_cycle.json"
if [ "${FORCE:-0}" != "1" ] && grep -q "\"$CYCLE\"" "$STAMP" 2>/dev/null; then
  echo "cycle $CYCLE already built - nothing to do (FORCE=1 to override)"; exit 0
fi
export GEPS_CYCLE="$CYCLE"
echo "== GEPS extended cycle $CYCLE =="
DOY=$(date -u -d "${CYCLE:0:4}-${CYCLE:4:2}-${CYCLE:6:2}" +%j 2>/dev/null || date -u -j -f %Y%m%d "$CYCLE" +%j)
DOY=$((10#$DOY))

source "$SITE/scripts/lib/parallel.sh"
# observed OLR / wind channels for the MJO: ERA5 (ARCO, incremental) to ~6 days back, GMGSI proxy for the tail; no
# Datamart, so while the forecast is fetched
if [ "${GEPS_OFFLINE:-0}" != "1" ]; then
  par "MJO observed channels (ARCO ERA5, new days only)" "MJO uses the committed channel" bash -c "cd '$SITE/scripts/mjo' && '$PY' src/fetch_arco_olr_band.py --workers 4 && '$PY' src/fetch_arco_eq_uv.py --days 55 --workers 4"
fi
step "forecast: fetch + de-drifted anomalies" "$PY" forecast.py --date "$CYCLE" || exit 1
par_wait

step "maps (daily frames, weekly, change)" "$PY" render_maps.py --cycle "$CYCLE" --workers "$WORKERS" --base era5 || exit 1
# polar vortex: ERA5-referenced, de-drifted with the S2S GEPS reforecast tables in the reference set (fixed; their
# rebuild needs the ECMWF S2S archive and is not part of this job); then the dripping paint, which reads the 10/100 hPa
# files the frames cached
vortex() {
  (cd "$SITE/scripts/strat" && "$PY" strat_frames.py --date "$CYCLE" && for h in nh sh; do "$PY" strat_series.py --hemi "$h" --date "$CYCLE" || exit 1; done) \
    || "$SITE/scripts/lib/degraded.sh" "vortex frames + series" "the previous vortex products stay"
  (cd "$SITE/scripts/strat" && "$PY" forecast_drip.py --model geps --date "$CYCLE" --out-dir "$OUTROOT/assets/geps") \
    || "$SITE/scripts/lib/degraded.sh" "dripping paint" "the previous figure stays"
}
par "vortex frames + series, dripping paint" "the previous vortex products and dripping paint stay" vortex
par "teleconnections" "the previous figures stay" "$PY" telecon_geps.py --cycle "$CYCLE"
par "tercile probability maps" "the previous maps stay" "$PY" prob_maps.py --cycle "$CYCLE"
par "weather regimes" "the previous regimes stay" "$PY" regimes_geps.py --cycle "$CYCLE"
par "ensemble clusters" "the previous clusters stay" "$PY" cluster_geps.py --cycle "$CYCLE"
par "MJO hindcast skill (replot)" "the previous skill figure stays" "$PY" mjo_hindcast.py --replot --doy-now "$DOY"
par "MJO phase + Hovmoller" "the previous MJO figures stay" bash -c "cd '$SITE/scripts/mjo' && '$PY' '$HERE/mjo_geps.py' --cycle '$CYCLE'"
par "100 hPa heat flux" "the previous figure stays" bash -c "cd '$SITE' && '$PY' scripts/strat/heatflux100_ext.py --model geps --date '$CYCLE' --out-dir '$OUTROOT/assets/geps'"
par_wait

# into the site: static figures + manifests on main, frames (gitignored on main) to the frames branch
FIGS="$GEPS_WORK/figs"
cp "$FIGS"/*.webp "$OUTROOT/assets/geps/"
for d in "$FIGS"/anim/*/; do
  r="$(basename "$d")"
  rm -rf "$OUTROOT/assets/geps/anim/$r"; mkdir -p "$OUTROOT/assets/geps/anim/$r"
  cp "$d"*.webp "$OUTROOT/assets/geps/anim/$r/"
done
cp "$FIGS"/anim/*.json "$OUTROOT/assets/geps/anim/"
# The cycle stamp is DATA, not a page edit: subseasonal.html reads assets/geps/cycle.json and this job never writes or
# commits the page (2026-10-08 - a run that started before a hand edit committed its stale copy of the whole page and
# reverted the edit). Lead day L is valid on the UTC day init + (L - 1), so days 1-35 run init .. init + 34.
"$PY" - "$CYCLE" "$OUTROOT/assets/geps/cycle.json" <<'PYEOF'
import json, sys, datetime as dt
from pathlib import Path
c = sys.argv[1]; d = dt.date(int(c[:4]), int(c[4:6]), int(c[6:8]))
Path(sys.argv[2]).write_text(json.dumps({"cycle": c, "init_date": f"{d:%Y-%m-%d}", "init_hour": "00",
                                         "valid_from": f"{d:%Y-%m-%d}", "valid_to": f"{d + dt.timedelta(days=34):%Y-%m-%d}",
                                         "lead_days": 35}) + "\n")
print(f"  cycle stamp {d} -> {sys.argv[2]}")
PYEOF

step "archive this cycle for the next run's comparisons" "$PY" archive.py --cycle "$CYCLE" || echo "::warning::archive failed"
"$PY" - "$CYCLE" <<'PYEOF'
import json, os, sys, datetime as dt
from pathlib import Path
st = Path(os.environ["GEPS_STATE"]); ana = st / "telecon" / "analysis"; mj = st / "mjo_analysis"
today = dt.date.today()
cut = (today - dt.timedelta(days=70)).strftime("%Y%m%d")     # observed-tail caches: the products show 45 days at most
cut1 = (today - dt.timedelta(days=150)).strftime("%Y%m%d")   # GEPS day-1 OLR (olr1_*): mjo_geps's 120-day filter + a month
n = 0
for d in (ana, mj):
    for q in d.glob("*.npz") if d.exists() else []:
        tag = q.stem.rsplit("_", 1)[-1]
        if tag.isdigit() and tag < (cut1 if q.name.startswith("olr1_") else cut):
            q.unlink(); n += 1
(st / "last_cycle.json").write_text(json.dumps({"cycle": sys.argv[1], "built": dt.datetime.utcnow().isoformat(timespec="minutes") + "Z"}))
print(f"  state: last cycle {sys.argv[1]}, pruned {n} cache files older than {cut} (day-1 OLR: {cut1})")
PYEOF
"$PY" datamart.py
out "new=1"; out "cycle=$CYCLE"
echo "== done: cycle $CYCLE =="
