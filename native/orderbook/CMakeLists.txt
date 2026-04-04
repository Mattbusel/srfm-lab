cmake_minimum_required(VERSION 3.16)
project(hft_orderbook CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
add_compile_options(-O3 -mavx2 -std=c++17 -Wall -Wextra -march=native)

# Order book library
add_library(hft_orderbook STATIC
    orderbook.cpp
    market_impact.cpp
    feed_handler.cpp
    matching_engine.cpp
)

target_include_directories(hft_orderbook PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link with pthreads on Linux
find_package(Threads REQUIRED)
target_link_libraries(hft_orderbook PUBLIC Threads::Threads)

# Test executable
add_executable(orderbook_test test_main.cpp)
target_link_libraries(orderbook_test PRIVATE hft_orderbook)

install(TARGETS hft_orderbook DESTINATION lib)
install(FILES
    order.hpp
    orderbook.hpp
    lockfree_queue.hpp
    market_impact.hpp
    feed_handler.hpp
    matching_engine.hpp
    DESTINATION include/hft)
