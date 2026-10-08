#!/bin/bash
# dev: ensemble AR probabilities, the four systems + the combination
set -uo pipefail
mkdir -p scratch_out/frames
df -h / | tail -1
for s in gefs geps; do
  ( python scripts/ar/ar_ens.py system --sys $s --work scratch_out/ens > scratch_out/ens_$s.log 2>&1; echo "$s exit $?" >> scratch_out/ens_$s.log ) &
done
python scripts/ar/ar_ens.py system --sys aifs --work scratch_out/ens > scratch_out/ens_aifs.log 2>&1; echo "aifs exit $?" >> scratch_out/ens_aifs.log
df -h / | tail -1; rm -rf scripts/ecmwf/cache/*
python scripts/ar/ar_ens.py system --sys ifs --work scratch_out/ens > scratch_out/ens_ifs.log 2>&1; echo "ifs exit $?" >> scratch_out/ens_ifs.log
wait
for s in aifs ifs gefs geps; do echo "===== $s"; grep -v "UserWarning\|_set_context\|fetching\|✓" scratch_out/ens_$s.log | tail -20; done
ls -la scratch_out/ens
python scripts/ar/ar_ens.py combine --work scratch_out/ens 2>&1 | grep -v "UserWarning\|_set_context" | tail -20
du -sh assets/ar/anim/ens_* assets/ar/anim/pwat* 2>/dev/null
for d in assets/ar/anim/ens_* assets/ar/anim/pwat*; do [ -d $d ] || continue; b=$(basename $d); cp $d/F04.webp scratch_out/frames/${b}_F04.webp; cp $d/F30.webp scratch_out/frames/${b}_F30.webp 2>/dev/null; done
cp assets/ar/ens_coast_*.webp assets/ar/ar_ens.json scratch_out/frames/ 2>/dev/null; cp assets/ar/anim/ens_plume.json scratch_out/ 2>/dev/null
