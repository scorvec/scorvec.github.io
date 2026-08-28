#!/bin/bash
# Rebuild skewt/sharplib.{js,wasm} from SHARPlib + the wrapper in this directory.
# Needs emsdk (https://github.com/emscripten-core/emsdk) and clones SHARPlib+fmt.
set -e
WORK=$(mktemp -d)
git clone --depth 1 https://github.com/keltonhalbert/SHARPlib.git "$WORK/SHARPlib"
git clone --depth 1 https://github.com/fmtlib/fmt.git "$WORK/fmt"
cp "$(dirname "$0")/skewt_wasm.cpp" "$WORK/SHARPlib/"
cd "$WORK/SHARPlib"
emcc -O2 -std=c++17 -DFMT_HEADER_ONLY -Iinclude -I../fmt/include \
  skewt_wasm.cpp src/SHARPlib/*.cpp src/SHARPlib/params/*.cpp \
  -sMODULARIZE=1 -sEXPORT_NAME=createSharp \
  -sEXPORTED_FUNCTIONS=_compute_sounding,_trace_adiabat,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPF32 -sALLOW_MEMORY_GROWTH=1 \
  -sENVIRONMENT=web,worker \
  -o "$(cd "$(dirname "$0")" && cd ../../skewt && pwd)/sharplib.js"
echo "rebuilt skewt/sharplib.{js,wasm}"
