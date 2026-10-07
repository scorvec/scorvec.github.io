#!/bin/bash
# dev: deterministic IVT, all four models, one cycle
set -uo pipefail
mkdir -p scratch_out/frames
D=$(date -u -d '-9 hours' +%Y%m%d); H=00
[ "$(date -u -d '-9 hours' +%H)" -ge 12 ] && H=12
echo "cycle $D ${H}Z"
for m in gfs aifs ifs gdps; do
  ( python scripts/ar/ar_det.py model --model $m --date $D --time $H --work scratch_out/work --wait-min 0 --procs 2 > scratch_out/det_$m.log 2>&1; echo "$m exit $?" >> scratch_out/det_$m.log ) &
done
wait
for m in gfs aifs ifs gdps; do echo "===== $m"; grep -v "UserWarning\|_set_context" scratch_out/det_$m.log | tail -25; done
python scripts/ar/ar_det.py combine --work scratch_out/work
du -sh assets/ar/anim/det_*
for d in assets/ar/anim/det_*; do b=$(basename $d); cp $d/F00.webp scratch_out/frames/${b}_F00.webp; cp $d/F12.webp scratch_out/frames/${b}_F12.webp; [ -f $d/coast.json ] && cp $d/coast.json scratch_out/frames/${b}_coast.json; done
cp assets/ar/anim/ar_det_manifest.json scratch_out/; cp scratch_out/work/*.json scratch_out/
