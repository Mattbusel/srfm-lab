#!/bin/bash
# Build the Rust extension and install it
cd crates/larsa-core
maturin develop --release
echo "Rust extension built successfully"
