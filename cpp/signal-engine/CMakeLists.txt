cmake_minimum_required(VERSION 3.20)
project(srfm_signal_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -O3
        -march=native
        -ffast-math
        -funroll-loops
        -fno-omit-frame-pointer
        -Wall
        -Wextra
        -Wpedantic
    )
elseif(MSVC)
    add_compile_options(
        /O2
        /arch:AVX2
        /fp:fast
        /W4
        /EHsc
    )
endif()

# Include directories
include_directories(include)

# Collect all source files for the library
set(LIB_SOURCES
    src/indicators/ema.cpp
    src/indicators/atr.cpp
    src/indicators/rsi.cpp
    src/indicators/bollinger.cpp
    src/indicators/macd.cpp
    src/indicators/vwap.cpp
    src/indicators/realized_vol.cpp
    src/bh_physics/bh_state.cpp
    src/bh_physics/garch.cpp
    src/bh_physics/ou_detector.cpp
    src/portfolio/risk_parity.cpp
    src/portfolio/pid_controller.cpp
    src/streaming/bar_aggregator.cpp
    src/streaming/feed_processor.cpp
    src/io/csv_reader.cpp
    src/io/json_writer.cpp
    src/io/binary_protocol.cpp
)

# Static library target
add_library(signal_engine_lib STATIC ${LIB_SOURCES})
target_include_directories(signal_engine_lib PUBLIC include)

# CLI executable
add_executable(signal_engine_cli src/main.cpp)
target_link_libraries(signal_engine_cli PRIVATE signal_engine_lib)

# Individual test executables (each has its own main())
add_executable(test_bh_physics    tests/test_bh_physics.cpp)
add_executable(test_indicators    tests/test_indicators.cpp)
add_executable(test_garch         tests/test_garch.cpp)
add_executable(test_ring_buffer   tests/test_ring_buffer.cpp)
add_executable(test_performance   tests/test_performance.cpp)

foreach(t test_bh_physics test_indicators test_garch test_ring_buffer test_performance)
    target_link_libraries(${t} PRIVATE signal_engine_lib)
endforeach()

# Convenience alias that runs all tests
add_custom_target(signal_engine_tests
    COMMAND test_bh_physics
    COMMAND test_indicators
    COMMAND test_garch
    COMMAND test_ring_buffer
    COMMAND test_performance
    DEPENDS test_bh_physics test_indicators test_garch test_ring_buffer test_performance
    COMMENT "Running all SRFM signal engine tests"
)

# Benchmark executable
add_executable(signal_engine_bench benchmarks/bench_signal_throughput.cpp)
target_link_libraries(signal_engine_bench PRIVATE signal_engine_lib)

# Enable threading
find_package(Threads REQUIRED)
target_link_libraries(signal_engine_lib PUBLIC Threads::Threads)

# Platform-specific: mmap support
if(UNIX)
    target_compile_definitions(signal_engine_lib PRIVATE SRFM_HAVE_MMAP=1)
endif()

# Install targets
install(TARGETS signal_engine_lib signal_engine_cli
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

message(STATUS "SRFM Signal Engine v${PROJECT_VERSION}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
