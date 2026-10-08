#!/bin/bash
# dev: IVT validation against ERA5 (Nov 2024 West Coast AR; a recent analysis with all four models)
set -uo pipefail
mkdir -p scratch_out
python scripts/ar/ar_validate.py --models ifs gfs aifs gdps --out scratch_out/val_nov2024.json \
  --times 2024-11-19T12 2024-11-20T00 2024-11-20T12 2024-11-21T00 2024-11-21T12 2024-11-22T00 2>&1 | grep -v "UserWarning\|_set_context"
python scripts/ar/ar_validate.py --models ifs gfs aifs gdps --out scratch_out/val_recent.json \
  --times 2026-09-30T00 2026-09-30T12 2026-10-01T00 2>&1 | grep -v "UserWarning\|_set_context"
