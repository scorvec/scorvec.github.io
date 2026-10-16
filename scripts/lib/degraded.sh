#!/bin/bash
# Record a product that did not update in this run, for the guards that keep one failure from sinking the others.
#
#     python x.py || "$GITHUB_WORKSPACE/scripts/lib/degraded.sh" "ASCAT wind map"
#     python y.py || "$GITHUB_WORKSPACE/scripts/lib/degraded.sh" "MEI nowcast" "the previous figures stay"
#     "$GITHUB_WORKSPACE/scripts/lib/degraded.sh" --summary          # a last step with `if: always()`
#
# A bare `|| echo "...; continuing"` turned a product that stopped updating into a green run with nothing to see: the
# message sat in the step log. This emits a ::warning:: naming the product (shown on the run page) and appends it to the
# job's degraded list, which --summary writes to the run summary. Always exits 0, so the guard keeps its meaning.
set -u
LIST="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/degraded.txt"
if [ "${1:-}" = "--summary" ]; then
  [ -s "$LIST" ] || exit 0
  {
    echo "### Degraded: $(wc -l < "$LIST" | tr -d ' ') product(s) not updated this run"
    sed 's/^/- /' "$LIST"
  } >> "${GITHUB_STEP_SUMMARY:-/dev/stdout}"
  exit 0
fi
what="${1:-unnamed product}"
why="${2:-the previous version stays}"
echo "::warning title=Not updated::$what failed; $why"
echo "$what: $why" >> "$LIST"
exit 0
