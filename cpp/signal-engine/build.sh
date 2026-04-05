#!/usr/bin/env bash
# Quick build script for SRFM Signal Engine
# Usage: ./build.sh [Debug|Release] [clean]

set -e

BUILD_TYPE="${1:-Release}"
BUILD_DIR="build/${BUILD_TYPE}"

if [ "$2" = "clean" ]; then
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake ../.. \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_COMPILER=g++ \
    "$@"

cmake --build . --parallel "$(nproc 2>/dev/null || echo 4)"

echo ""
echo "Build complete: ${BUILD_DIR}/"
echo "Executables:"
echo "  signal_engine_cli"
echo "  signal_engine_bench"
echo "  test_bh_physics test_indicators test_garch test_ring_buffer test_performance"
echo ""
echo "Quick test:"
echo "  ./test_bh_physics && ./test_indicators && ./test_garch && ./test_ring_buffer"
echo ""
echo "Benchmark:"
echo "  ./signal_engine_bench"
echo "  ./signal_engine_cli --bench --bench-n 20"
