cmake_minimum_required(VERSION 3.16)
project(hft_matrix CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AVX2 + FMA + O3
add_compile_options(-O3 -mavx2 -mfma -std=c++17 -Wall -Wextra -march=native)

# SIMD ops library
add_library(linalg_simd STATIC
    simd_ops.cpp
    decomposition.cpp
    portfolio_math.cpp
)
target_include_directories(linalg_simd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Benchmark executable
add_executable(matrix_benchmark benchmark.cpp)
target_link_libraries(matrix_benchmark PRIVATE linalg_simd)

install(TARGETS linalg_simd DESTINATION lib)
install(FILES matrix.hpp DESTINATION include/linalg)
