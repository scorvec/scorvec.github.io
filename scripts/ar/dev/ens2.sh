#!/bin/bash
# dev: GEPS and IFS-ENS systems after the Datamart-retry and screen fixes
set -uo pipefail
mkdir -p scratch_out
( python scripts/ar/ar_ens.py system --sys geps --work scratch_out/ens > scratch_out/ens_geps.log 2>&1; echo "geps exit $?" >> scratch_out/ens_geps.log ) &
python scripts/ar/ar_ens.py system --sys ifs --work scratch_out/ens > scratch_out/ens_ifs.log 2>&1; echo "ifs exit $?" >> scratch_out/ens_ifs.log
wait
for s in ifs geps; do echo "===== $s"; grep -v "UserWarning\|_set_context\|fetching\|✓" scratch_out/ens_$s.log | tail -15 | cut -c1-1200; done
