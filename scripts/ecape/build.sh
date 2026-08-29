#!/bin/bash
# Build the gridded ECAPE kernel against SHARPlib.
#
# Clones SHARPlib + fmt (header-only) the same way skewt/build_wasm.sh does, so
# the map and the browser Skew-T track the same upstream. The whole compile is
# ~5 s, which is why this runs in CI rather than shipping a binary.
#
#   ./build.sh [outdir]        default outdir: this directory
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone --depth 1 -q https://github.com/keltonhalbert/SHARPlib.git "$WORK/SHARPlib"
git clone --depth 1 -q https://github.com/fmtlib/fmt.git "$WORK/fmt"

# The wrapper is shared with the WASM build: one definition of the output layout.
cp "$HERE/../skewt/skewt_wasm.cpp" "$WORK/SHARPlib/"
cp "$HERE/ecape_grid.cpp" "$WORK/SHARPlib/"
cd "$WORK/SHARPlib"

CXX="${CXX:-c++}"
OMP=""
# Apple clang needs libomp explicitly; gcc/clang on Linux just take -fopenmp.
if [ "$(uname)" = "Darwin" ] && [ -d /opt/homebrew/opt/libomp ]; then
  OMP="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
else
  OMP="-fopenmp"
fi

# shellcheck disable=SC2086
$CXX -O2 -std=c++17 -DFMT_HEADER_ONLY -Iinclude -I"$WORK/fmt/include" $OMP \
  ecape_grid.cpp skewt_wasm.cpp src/SHARPlib/*.cpp src/SHARPlib/params/*.cpp \
  -o "$OUT/ecape_grid"

echo "built $OUT/ecape_grid"
