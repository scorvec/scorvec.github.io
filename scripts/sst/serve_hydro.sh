#!/usr/bin/env bash
# Serve the Colombia hydro site locally.
#
# The site is a static bundle under ~/colombia_hydro/site (symlinked into the
# public repo as colombia_hydro/, gitignored there). Opening index.html with
# file:// breaks the JSON fetches the charts rely on, so it needs a real HTTP
# origin even locally.
#
#     scripts/sst/serve_hydro.sh [port]     # default 8899
set -euo pipefail
PORT="${1:-8899}"
SITE="$HOME/colombia_hydro/site"
[ -d "$SITE" ] || { echo "no site at $SITE"; exit 1; }
echo "serving $SITE  ->  http://localhost:$PORT/"
echo "  (ctrl-C to stop)"
command -v open >/dev/null && (sleep 1; open "http://localhost:$PORT/") &
exec python3 -m http.server "$PORT" --directory "$SITE"
