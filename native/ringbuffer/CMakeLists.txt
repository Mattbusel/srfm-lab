cmake_minimum_required(VERSION 3.16)
project(hft_ringbuffer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -mavx2 -std=c++17 -Wall -Wextra -march=native)

add_library(hft_ringbuffer STATIC
    tick_store.cpp
    compressor.cpp
)
target_include_directories(hft_ringbuffer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Threads REQUIRED)
target_link_libraries(hft_ringbuffer PUBLIC Threads::Threads)

add_executable(ringbuffer_test test_main.cpp)
target_link_libraries(ringbuffer_test PRIVATE hft_ringbuffer)

install(TARGETS hft_ringbuffer DESTINATION lib)
install(FILES ringbuffer.hpp DESTINATION include/hft)
