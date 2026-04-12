cmake_minimum_required(VERSION 3.20)
project(chronos_lob
    VERSION 1.0.0
    DESCRIPTION "Chronos LOB Simulator — AETERNUS Module 1"
    LANGUAGES CXX
)

# ── Compiler configuration ────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type defaults to Release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# ── Compiler flags ────────────────────────────────────────────────────────────

# Common flags.
set(CHRONOS_COMMON_FLAGS
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -fno-exceptions    # Disable exceptions for performance (use asserts instead).
)

# Release flags.
set(CHRONOS_RELEASE_FLAGS
    -O3
    -march=native       # Enable AVX2, SSE4.2, etc. on host CPU.
    -mtune=native
    -funroll-loops
    -ffast-math
    -DNDEBUG
)

# Debug flags.
set(CHRONOS_DEBUG_FLAGS
    -O0 -g3
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
)

# ── SIMD detection ────────────────────────────────────────────────────────────

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)

if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2 support detected")
    add_compile_options(-mavx2 -mfma)
    add_compile_definitions(CHRONOS_HAS_AVX2=1)
elseif(COMPILER_SUPPORTS_SSE42)
    message(STATUS "SSE4.2 support detected")
    add_compile_options(-msse4.2)
    add_compile_definitions(CHRONOS_HAS_SSE42=1)
else()
    message(STATUS "No SIMD extensions detected; using scalar fallback")
endif()

# ── Optional: Apache Arrow for IPC output ─────────────────────────────────────

option(WITH_ARROW "Link against Apache Arrow for IPC output" OFF)

if(WITH_ARROW)
    find_package(Arrow QUIET)
    if(Arrow_FOUND)
        message(STATUS "Apache Arrow found: ${Arrow_VERSION}")
        add_compile_definitions(CHRONOS_HAS_ARROW=1)
    else()
        message(WARNING "Apache Arrow not found; CSV output only")
        set(WITH_ARROW OFF)
    endif()
endif()

# ── Source files ──────────────────────────────────────────────────────────────

set(CHRONOS_HEADERS
    include/lob_types.hpp
    include/price_level.hpp
    include/matching_engine.hpp
    include/simd_utils.hpp
)

set(CHRONOS_SOURCES
    src/matching_engine.cpp
    src/heston_engine.cpp
)

# ── Library target ────────────────────────────────────────────────────────────

add_library(chronos_core STATIC
    ${CHRONOS_HEADERS}
    ${CHRONOS_SOURCES}
)

target_include_directories(chronos_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(chronos_core
    PRIVATE
        ${CHRONOS_COMMON_FLAGS}
        $<$<CONFIG:Release>:${CHRONOS_RELEASE_FLAGS}>
        $<$<CONFIG:Debug>:${CHRONOS_DEBUG_FLAGS}>
)

if(WITH_ARROW AND Arrow_FOUND)
    target_link_libraries(chronos_core PUBLIC Arrow::arrow_shared)
endif()

# ── Simulator executable ──────────────────────────────────────────────────────

add_executable(lob_simulator src/lob_simulator.cpp)

target_link_libraries(lob_simulator PRIVATE chronos_core)

target_compile_options(lob_simulator
    PRIVATE
        ${CHRONOS_COMMON_FLAGS}
        $<$<CONFIG:Release>:${CHRONOS_RELEASE_FLAGS}>
        $<$<CONFIG:Debug>:${CHRONOS_DEBUG_FLAGS}>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options(lob_simulator PRIVATE -fsanitize=address,undefined)
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    # Simple test: just check the simulator compiles and runs.
    add_executable(test_matching
        tests/test_matching.cpp
    )
    target_link_libraries(test_matching PRIVATE chronos_core)
    target_include_directories(test_matching PRIVATE include)
    target_compile_options(test_matching
        PRIVATE
            ${CHRONOS_COMMON_FLAGS}
            $<$<CONFIG:Release>:${CHRONOS_RELEASE_FLAGS}>
    )
    add_test(NAME matching_engine_test COMMAND test_matching)
endif()

# ── Install ───────────────────────────────────────────────────────────────────

install(TARGETS chronos_core lob_simulator
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES ${CHRONOS_HEADERS} DESTINATION include/chronos)

# ── Summary ───────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "Chronos LOB Simulator configuration:")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:  C++${CMAKE_CXX_STANDARD}")
message(STATUS "  AVX2:          ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "  SSE4.2:        ${COMPILER_SUPPORTS_SSE42}")
message(STATUS "  Arrow IPC:     ${WITH_ARROW}")
message(STATUS "  Tests:         ${BUILD_TESTS}")
message(STATUS "")
