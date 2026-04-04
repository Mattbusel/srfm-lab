const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Main executable ───────────────────────────────────────────────────────
    const exe = b.addExecutable(.{
        .name             = "market_data",
        .root_source_file = b.path("src/main.zig"),
        .target           = target,
        .optimize         = optimize,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the market data processor");
    run_step.dependOn(&run_cmd.step);

    // ── Simulation executable ─────────────────────────────────────────────────
    const sim_exe = b.addExecutable(.{
        .name             = "simulate",
        .root_source_file = b.path("src/simulation.zig"),
        .target           = target,
        .optimize         = optimize,
    });
    b.installArtifact(sim_exe);

    const sim_run = b.addRunArtifact(sim_exe);
    sim_run.step.dependOn(b.getInstallStep());
    const sim_step = b.step("simulate", "Run full end-to-end simulation");
    sim_step.dependOn(&sim_run.step);

    // ── Unit tests ────────────────────────────────────────────────────────────
    const test_files = [_][]const u8{
        "src/decoder.zig",
        "src/orderbook.zig",
        "src/feed.zig",
        "src/stats.zig",
        "src/writer.zig",
        "src/allocator.zig",
        "src/market_maker.zig",
        "src/protocol.zig",
        "src/network.zig",
        "src/simulation.zig",
        "src/risk.zig",
    };

    const test_step = b.step("test", "Run all unit tests");
    for (test_files) |file| {
        const unit_tests = b.addTest(.{
            .root_source_file = b.path(file),
            .target           = target,
            .optimize         = optimize,
        });
        const run_tests = b.addRunArtifact(unit_tests);
        test_step.dependOn(&run_tests.step);
    }

    // ── Benchmark executable ──────────────────────────────────────────────────
    const bench_exe = b.addExecutable(.{
        .name             = "bench",
        .root_source_file = b.path("src/bench.zig"),
        .target           = target,
        .optimize         = .ReleaseFast,
    });
    b.installArtifact(bench_exe);

    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run all benchmarks (ReleaseFast)");
    bench_step.dependOn(&run_bench.step);

    // ── Check step: typecheck all source files without running ─────────────────
    const check_files = [_][]const u8{
        "src/main.zig",
        "src/decoder.zig",
        "src/orderbook.zig",
        "src/feed.zig",
        "src/stats.zig",
        "src/writer.zig",
        "src/allocator.zig",
        "src/market_maker.zig",
        "src/protocol.zig",
        "src/network.zig",
        "src/simulation.zig",
        "src/risk.zig",
        "src/bench.zig",
    };
    const check_step = b.step("check", "Typecheck all modules");
    for (check_files) |file| {
        const check_exe = b.addExecutable(.{
            .name             = "check",
            .root_source_file = b.path(file),
            .target           = target,
            .optimize         = .Debug,
        });
        check_step.dependOn(&check_exe.step);
    }
}
