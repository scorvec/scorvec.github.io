#!/bin/bash
set -euo pipefail
mkdir -p scratch_out
python scripts/ar/probe_sources.py 2>&1 | tee scratch_out/probe.txt
