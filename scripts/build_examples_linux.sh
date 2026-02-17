#!/bin/bash
# Linux build script for gpulite examples
# Requires GCC 7+ or Clang 5+ with C++17 support

echo "Building gpulite examples for Linux..."

# Change to root directory (assuming script is in scripts/ folder)
cd "$(dirname "$0")/.."

# Create build directories if they don't exist
mkdir -p examples/basic_vector_add/build
mkdir -p examples/matrix_multiply/build
mkdir -p examples/templated_kernels/build

# Build basic_vector_add
echo "Building basic_vector_add..."
cd examples/basic_vector_add/build
cmake .. && make
cd ../../..

# Build matrix_multiply
echo "Building matrix_multiply..."
cd examples/matrix_multiply/build
cmake .. && make
cd ../../..

# Build templated_kernels
echo "Building templated_kernels..."
cd examples/templated_kernels/build
cmake .. && make
cd ../../..

echo "Build complete!"
echo ""
echo "Executables are located in:"
echo "- examples/basic_vector_add/build/vector_add"
echo "- examples/matrix_multiply/build/matrix_multiply"
echo "- examples/templated_kernels/build/templated_kernels"
