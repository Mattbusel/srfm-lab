// build.zig -- Build script for libsignal shared library.
// Produces libsignal.so (Linux/macOS) or signal.dll (Windows) as ReleaseFast.
// All C ABI export functions are included.
//
// Usage:
//   zig build                     -- build shared library (ReleaseFast)
//   zig build test                -- run unit tests
//   zig build -Dtarget=x86_64-linux-gnu  -- cross-compile
//
// Output: zig-out/lib/libsignal.so (or .dll on Windows)

const std = @import("std");

pub fn build(b: *std.Build) void {
    // ----------------------------------------------------------
    // Target and optimize options
    // ----------------------------------------------------------
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    // ----------------------------------------------------------
    // Shared library: libsignal
    // ----------------------------------------------------------
    const lib = b.addSharedLibrary(.{
        .name    = "signal",
        .root_source_file = b.path("signal.zig"),
        .target  = target,
        .optimize = optimize,
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
    });

    // Enable AVX2 on x86_64 targets (required for SIMD paths)
    if (target.query.cpu_arch == null or
        target.query.cpu_arch == .x86_64)
    {
        lib.root_module.cpu_features_add.addFeature(
            @intFromEnum(std.Target.x86.Feature.avx2)
        );
        lib.root_module.cpu_features_add.addFeature(
            @intFromEnum(std.Target.x86.Feature.fma)
        );
    }

    // Link libc for heap allocation paths (used in hurst_batch for large windows)
    lib.linkLibC();

    // Install the library to zig-out/lib/
    b.installArtifact(lib);

    // ----------------------------------------------------------
    // Also build a static library for embedding
    // ----------------------------------------------------------
    const static_lib = b.addStaticLibrary(.{
        .name    = "signal_static",
        .root_source_file = b.path("signal.zig"),
        .target  = target,
        .optimize = optimize,
    });
    if (target.query.cpu_arch == null or
        target.query.cpu_arch == .x86_64)
    {
        static_lib.root_module.cpu_features_add.addFeature(
            @intFromEnum(std.Target.x86.Feature.avx2)
        );
        static_lib.root_module.cpu_features_add.addFeature(
            @intFromEnum(std.Target.x86.Feature.fma)
        );
    }
    static_lib.linkLibC();
    b.installArtifact(static_lib);

    // ----------------------------------------------------------
    // Unit tests
    // ----------------------------------------------------------
    const tests = b.addTest(.{
        .root_source_file = b.path("signal.zig"),
        .target  = target,
        .optimize = .Debug,  // tests run in Debug for better error messages
    });
    tests.linkLibC();

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run signal.zig unit tests");
    test_step.dependOn(&run_tests.step);

    // ----------------------------------------------------------
    // Check step -- type-check without emitting code (fast CI step)
    // ----------------------------------------------------------
    const check = b.addSharedLibrary(.{
        .name    = "signal_check",
        .root_source_file = b.path("signal.zig"),
        .target  = target,
        .optimize = .Debug,
    });
    const check_step = b.step("check", "Type-check signal.zig without building");
    check_step.dependOn(&check.step);

    // ----------------------------------------------------------
    // Default step: build shared + static library
    // ----------------------------------------------------------
    const default_step = b.step("all", "Build shared and static libraries");
    default_step.dependOn(b.getInstallStep());
    _ = default_step;
}
