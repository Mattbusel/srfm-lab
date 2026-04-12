cmake_minimum_required(VERSION 3.18)
project(aeternus_rtel
    VERSION 0.1.0
    DESCRIPTION "AETERNUS Real-Time Execution Layer"
    LANGUAGES CXX
)

# ===========================================================================
# Global settings
# ===========================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(
        /W4 /WX-
        /O2
        /arch:AVX2
        /EHsc
        /DNOMINMAX
    )
    # Atomic operations on MSVC
    add_compile_definitions(_WIN32_WINNT=0x0A00)
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -O3
        -march=native
        -fno-omit-frame-pointer
        -fPIC
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-Wno-c99-extensions)
    endif()
endif()

# Address sanitizer (optional)
option(RTEL_ASAN "Enable AddressSanitizer" OFF)
if(RTEL_ASAN)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

# Thread sanitizer (optional)
option(RTEL_TSAN "Enable ThreadSanitizer" OFF)
if(RTEL_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

# ===========================================================================
# Platform detection
# ===========================================================================
if(WIN32)
    add_compile_definitions(RTEL_PLATFORM_WINDOWS=1)
    set(RTEL_PLATFORM_LIBS "")
else()
    add_compile_definitions(RTEL_PLATFORM_POSIX=1)
    set(RTEL_PLATFORM_LIBS rt)
endif()

# ===========================================================================
# Find dependencies
# ===========================================================================
find_package(Threads REQUIRED)

# pybind11 (optional — for Python bridge)
option(RTEL_PYBIND11 "Build pybind11 Python extension" OFF)
if(RTEL_PYBIND11)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "pybind11 found: ${pybind11_VERSION}")
    else()
        message(WARNING "pybind11 not found — Python extension disabled")
        set(RTEL_PYBIND11 OFF)
    endif()
endif()

# ===========================================================================
# Source files
# ===========================================================================
set(RTEL_SOURCES
    src/shm_bus.cpp
    src/ring_buffer.cpp
    src/global_state_registry.cpp
    src/module_wrapper.cpp
    src/scheduler.cpp
    src/latency_monitor.cpp
    src/serialization.cpp
)

set(RTEL_HEADERS
    include/rtel/shm_bus.hpp
    include/rtel/ring_buffer.hpp
    include/rtel/global_state_registry.hpp
    include/rtel/module_wrapper.hpp
    include/rtel/scheduler.hpp
    include/rtel/latency_monitor.hpp
    include/rtel/serialization.hpp
)

# ===========================================================================
# Main RTEL library (static)
# ===========================================================================
add_library(rtel_static STATIC ${RTEL_SOURCES} ${RTEL_HEADERS})

target_include_directories(rtel_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(rtel_static
    PUBLIC
        Threads::Threads
        ${RTEL_PLATFORM_LIBS}
)

set_target_properties(rtel_static PROPERTIES
    OUTPUT_NAME "aeternus_rtel"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# ===========================================================================
# Shared library
# ===========================================================================
add_library(rtel_shared SHARED ${RTEL_SOURCES})

target_include_directories(rtel_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(rtel_shared
    PUBLIC
        Threads::Threads
        ${RTEL_PLATFORM_LIBS}
)

set_target_properties(rtel_shared PROPERTIES
    OUTPUT_NAME "aeternus_rtel"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# ===========================================================================
# pybind11 Python extension (optional)
# ===========================================================================
if(RTEL_PYBIND11)
    pybind11_add_module(rtel_py
        src/python_bindings.cpp
    )
    target_include_directories(rtel_py PRIVATE include)
    target_link_libraries(rtel_py PRIVATE rtel_static)
    set_target_properties(rtel_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    )
endif()

# ===========================================================================
# Tests
# ===========================================================================
option(RTEL_BUILD_TESTS "Build RTEL tests" ON)
if(RTEL_BUILD_TESTS)
    enable_testing()

    add_executable(test_shm_bus tests/test_shm_bus.cpp)
    target_include_directories(test_shm_bus PRIVATE include)
    target_link_libraries(test_shm_bus PRIVATE rtel_static)
    set_target_properties(test_shm_bus PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_test(NAME rtel_unit_tests
        COMMAND test_shm_bus
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    # Test properties
    set_tests_properties(rtel_unit_tests PROPERTIES
        TIMEOUT 60
        PASS_REGULAR_EXPRESSION "Results: [0-9]+ passed, 0 failed"
    )
endif()

# ===========================================================================
# Benchmark executable
# ===========================================================================
option(RTEL_BUILD_BENCHMARKS "Build RTEL benchmarks" OFF)
if(RTEL_BUILD_BENCHMARKS)
    add_executable(rtel_bench
        benchmarks/bench_ring_buffer.cpp
        benchmarks/bench_shm_bus.cpp
        benchmarks/bench_gsr.cpp
    )
    target_include_directories(rtel_bench PRIVATE include)
    target_link_libraries(rtel_bench PRIVATE rtel_static)
endif()

# ===========================================================================
# Installation
# ===========================================================================
include(GNUInstallDirs)

install(TARGETS rtel_static rtel_shared
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/rtel
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# ===========================================================================
# Summary
# ===========================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS " AETERNUS RTEL Build Configuration")
message(STATUS "========================================")
message(STATUS " Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS " C++ standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS " Platform libs:  ${RTEL_PLATFORM_LIBS}")
message(STATUS " Tests:          ${RTEL_BUILD_TESTS}")
message(STATUS " pybind11:       ${RTEL_PYBIND11}")
message(STATUS " ASAN:           ${RTEL_ASAN}")
message(STATUS " TSAN:           ${RTEL_TSAN}")
message(STATUS "========================================")
message(STATUS "")
