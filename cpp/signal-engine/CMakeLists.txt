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
include_directories(src)           # for composite/ and ipc/ relative includes

# ------------------------------------------------------------------
# Google Test -- required for test_volume_profile and future gtest targets
# ------------------------------------------------------------------
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Try FetchContent fallback (requires CMake 3.14+)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    # For Windows: prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    set(GTEST_AVAILABLE TRUE)
else()
    set(GTEST_AVAILABLE TRUE)
endif()

# ------------------------------------------------------------------
# AVX2/FMA compile flags for new SIMD-heavy TUs
# ------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(SRFM_AVX2_FLAGS -mavx2 -mfma)
elseif(MSVC)
    # MSVC enables AVX2 globally via /arch:AVX2 above; no per-file flag needed
    set(SRFM_AVX2_FLAGS "")
endif()

# ------------------------------------------------------------------
# Core library sources (original)
# ------------------------------------------------------------------
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
    src/quaternion/quat_nav.cpp
    src/portfolio/risk_parity.cpp
    src/portfolio/pid_controller.cpp
    src/streaming/bar_aggregator.cpp
    src/streaming/feed_processor.cpp
    src/io/csv_reader.cpp
    src/io/json_writer.cpp
    src/io/binary_protocol.cpp
)

# ------------------------------------------------------------------
# New indicator / composite / IPC sources
# ------------------------------------------------------------------
set(EXT_SOURCES
    src/indicators/volume_profile.cpp
    src/indicators/order_flow.cpp
    src/indicators/market_breadth.cpp
    src/composite/regime_signal.cpp
    src/ipc/metrics_publisher.cpp
)

# Static library target
add_library(signal_engine_lib STATIC ${LIB_SOURCES} ${EXT_SOURCES})
target_include_directories(signal_engine_lib PUBLIC include src)

# Apply AVX2/FMA to the new extension sources specifically.
# On GCC/Clang this adds -mavx2 -mfma per-file.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND SRFM_AVX2_FLAGS)
    foreach(src ${EXT_SOURCES})
        set_source_files_properties(${src} PROPERTIES
            COMPILE_OPTIONS "${SRFM_AVX2_FLAGS}")
    endforeach()
endif()

# CLI executable
add_executable(signal_engine_cli src/main.cpp)
target_link_libraries(signal_engine_cli PRIVATE signal_engine_lib)

# ------------------------------------------------------------------
# Original test executables (standalone, no GTest)
# ------------------------------------------------------------------
add_executable(test_bh_physics    tests/test_bh_physics.cpp)
add_executable(test_indicators    tests/test_indicators.cpp)
add_executable(test_garch         tests/test_garch.cpp)
add_executable(test_ring_buffer   tests/test_ring_buffer.cpp)
add_executable(test_performance   tests/test_performance.cpp)
add_executable(test_quat_nav      tests/test_quat_nav.cpp)

foreach(t test_bh_physics test_indicators test_garch test_ring_buffer test_performance test_quat_nav)
    target_link_libraries(${t} PRIVATE signal_engine_lib)
endforeach()

# ------------------------------------------------------------------
# Google Test targets (new indicator / composite tests)
# ------------------------------------------------------------------
if(GTEST_AVAILABLE)
    enable_testing()

    # Volume Profile tests
    add_executable(test_volume_profile tests/test_volume_profile.cpp)
    target_link_libraries(test_volume_profile
        PRIVATE signal_engine_lib GTest::gtest GTest::gtest_main)
    target_include_directories(test_volume_profile PRIVATE include src)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND SRFM_AVX2_FLAGS)
        target_compile_options(test_volume_profile PRIVATE ${SRFM_AVX2_FLAGS})
    endif()

    # Register with CTest
    include(GoogleTest)
    gtest_discover_tests(test_volume_profile)

    message(STATUS "Google Test found -- test_volume_profile target enabled")
else()
    message(WARNING "Google Test not found -- test_volume_profile will not be built")
endif()

# ------------------------------------------------------------------
# Convenience alias: runs all tests (legacy + gtest)
# ------------------------------------------------------------------
set(ALL_TEST_TARGETS
    test_bh_physics
    test_indicators
    test_garch
    test_ring_buffer
    test_performance
    test_quat_nav
)
if(GTEST_AVAILABLE)
    list(APPEND ALL_TEST_TARGETS test_volume_profile)
endif()

add_custom_target(signal_engine_tests
    COMMAND test_bh_physics
    COMMAND test_indicators
    COMMAND test_garch
    COMMAND test_ring_buffer
    COMMAND test_performance
    COMMAND test_quat_nav
    DEPENDS ${ALL_TEST_TARGETS}
    COMMENT "Running all SRFM signal engine tests"
)

# ------------------------------------------------------------------
# Benchmark executable
# ------------------------------------------------------------------
add_executable(signal_engine_bench benchmarks/bench_signal_throughput.cpp)
target_link_libraries(signal_engine_bench PRIVATE signal_engine_lib)

# ------------------------------------------------------------------
# Threading
# ------------------------------------------------------------------
find_package(Threads REQUIRED)
target_link_libraries(signal_engine_lib PUBLIC Threads::Threads)

# ------------------------------------------------------------------
# Platform-specific: mmap support
# ------------------------------------------------------------------
if(UNIX)
    target_compile_definitions(signal_engine_lib PUBLIC SRFM_HAVE_MMAP=1)
    target_compile_definitions(signal_engine_cli PRIVATE SRFM_HAVE_MMAP=1)
    if(GTEST_AVAILABLE)
        target_compile_definitions(test_volume_profile PRIVATE SRFM_HAVE_MMAP=1)
    endif()
endif()

# ------------------------------------------------------------------
# Install
# ------------------------------------------------------------------
install(TARGETS signal_engine_lib signal_engine_cli
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY src/composite/ DESTINATION include/composite
    FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY src/ipc/ DESTINATION include/ipc
    FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY src/indicators/ DESTINATION include/indicators
    FILES_MATCHING PATTERN "*.hpp")

message(STATUS "SRFM Signal Engine v${PROJECT_VERSION}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "New modules: volume_profile, order_flow, market_breadth, regime_signal, metrics_publisher")
