#!/bin/bash
# dev run on the runner: deterministic IVT for every model, one cycle, timing + sample frames
set -uo pipefail
mkdir -p scratch_out
D=$(date -u -d '-9 hours' +%Y%m%d); H=00
[ "$(date -u +%H)" -ge 21 ] && H=12
echo "cycle $D ${H}Z"
python - <<'PY'
import numpy as np, sys
sys.path.insert(0, "scripts/ar"); import arlib as A
# unit check of the integral: constant q = 0.01, u = 10 over 1000-300 hPa -> 0.01*10*70000/g = 713.8
lv = A.ECMWF_LEVELS; q = np.full((8, 2, 2), 0.01); u = np.full((8, 2, 2), 10.0); v = np.zeros((8, 2, 2))
ps = np.array([[101000.0, 100000.0], [85000.0, 70000.0]])
e, n = A.integrate(lv, q, u, v, ps)
print("unit integral (expect 713.8, 713.8, 0.01*10*(85000-30000)/g = 560.8, 0.01*10*40000/g = 407.9):", e.ravel())
PY
for m in gfs aifs ifs gdps; do
  python scripts/ar/ar_det.py model --model $m --date $D --time $H --work scratch_out/work --wait-min 0 2>&1 | tail -40
done
python scripts/ar/ar_det.py combine --work scratch_out/work
du -sh assets/ar/anim/det_* || true
mkdir -p scratch_out/frames; for d in assets/ar/anim/det_*; do b=$(basename $d); cp $d/F00.webp scratch_out/frames/${b}_F00.webp 2>/dev/null; cp $d/F12.webp scratch_out/frames/${b}_F12.webp 2>/dev/null; cp $d/coast.json scratch_out/frames/${b}_coast.json 2>/dev/null; done
cp assets/ar/anim/ar_det_manifest.json scratch_out/ || true
