cmake_minimum_required(VERSION 3.16)
project(hft_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Global optimization flags
add_compile_options(-O3 -mavx2 -mfma -std=c++17 -march=native
                    -Wall -Wextra -Wno-unused-parameter)

add_subdirectory(orderbook)
add_subdirectory(matrix)
add_subdirectory(ringbuffer)
