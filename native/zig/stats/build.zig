//! build.zig -- Build file for the stats module.
//!
//! Targets:
//!   zig build          -- Builds libstats shared library
//!   zig build test     -- Runs all unit tests
//!   zig build install  -- Installs the shared library to zig-out/lib/

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    // -----------------------------------------------------------------------
    // Shared library: libstats.so / stats.dll
    // -----------------------------------------------------------------------
    // The library exports C-ABI ctypes-compatible symbols from both
    // stats.zig and hurst.zig so Python can load it with ctypes.CDLL.

    const lib = b.addSharedLibrary(.{
        .name             = "stats",
        .root_source_file = b.path("lib_root.zig"),
        .target           = target,
        .optimize         = optimize,
    });

    // Link libc so we can use std.heap.c_allocator in C ABI exports
    lib.linkLibC();

    b.installArtifact(lib);

    // -----------------------------------------------------------------------
    // Static library variant (useful for linking into other Zig executables)
    // -----------------------------------------------------------------------
    const static_lib = b.addStaticLibrary(.{
        .name             = "stats_static",
        .root_source_file = b.path("lib_root.zig"),
        .target           = target,
        .optimize         = optimize,
    });
    static_lib.linkLibC();
    b.installArtifact(static_lib);

    // -----------------------------------------------------------------------
    // Run step (no main, so just verify it compiles)
    // -----------------------------------------------------------------------
    const check_step = b.step("check", "Typecheck stats and hurst modules");
    const check_lib = b.addSharedLibrary(.{
        .name             = "stats_check",
        .root_source_file = b.path("lib_root.zig"),
        .target           = target,
        .optimize         = .Debug,
    });
    check_lib.linkLibC();
    check_step.dependOn(&check_lib.step);

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------
    const test_step = b.step("test", "Run stats + hurst unit tests");

    const stats_tests = b.addTest(.{
        .root_source_file = b.path("tests/test_stats.zig"),
        .target           = target,
        .optimize         = optimize,
    });
    stats_tests.linkLibC();
    // Make stats and hurst importable from the test file
    const stats_module = b.addModule("stats", .{
        .root_source_file = b.path("stats.zig"),
    });
    const hurst_module = b.addModule("hurst", .{
        .root_source_file = b.path("hurst.zig"),
    });
    stats_tests.root_module.addImport("stats", stats_module);
    stats_tests.root_module.addImport("hurst", hurst_module);

    const run_stats_tests = b.addRunArtifact(stats_tests);
    test_step.dependOn(&run_stats_tests.step);

    // -----------------------------------------------------------------------
    // Benchmark executable (ReleaseFast, no test framework)
    // -----------------------------------------------------------------------
    const bench = b.addExecutable(.{
        .name             = "bench_stats",
        .root_source_file = b.path("bench_stats.zig"),
        .target           = target,
        .optimize         = .ReleaseFast,
    });
    bench.root_module.addImport("stats", stats_module);
    bench.root_module.addImport("hurst", hurst_module);
    b.installArtifact(bench);

    const run_bench = b.addRunArtifact(bench);
    if (b.args) |args| run_bench.addArgs(args);
    const bench_step = b.step("bench", "Run stats benchmarks");
    bench_step.dependOn(&run_bench.step);
}
