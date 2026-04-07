//! bench_stats.zig -- Microbenchmarks for stats and hurst modules.

const std = @import("std");
const stats = @import("stats");
const hurst = @import("hurst");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    const N: usize = 1_000_000;

    // EWMA bench
    {
        var e = stats.EWMA(f64).init(0.1);
        const t0 = std.time.nanoTimestamp();
        for (0..N) |i| {
            e.update(@floatFromInt(i));
        }
        const elapsed = std.time.nanoTimestamp() - t0;
        try stdout.print("EWMA({d} iters): {d}ns total, {d:.1}ns/op, final={d:.4}\n",
            .{ N, elapsed, @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(N)), e.get() });
    }

    // RunningVariance bench
    {
        var v = stats.RunningVariance(f64).init();
        const t0 = std.time.nanoTimestamp();
        for (0..N) |i| {
            v.update(@floatFromInt(i));
        }
        const elapsed = std.time.nanoTimestamp() - t0;
        try stdout.print("RunningVariance({d} iters): {d}ns total, {d:.1}ns/op\n",
            .{ N, elapsed, @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(N)) });
    }

    // RunningCorrelation bench
    {
        var c = stats.RunningCorrelation(f64).init();
        const t0 = std.time.nanoTimestamp();
        for (0..N) |i| {
            const x: f64 = @floatFromInt(i);
            c.update(x, x * 1.1 + 0.5);
        }
        const elapsed = std.time.nanoTimestamp() - t0;
        try stdout.print("RunningCorrelation({d} iters): {d}ns total, {d:.1}ns/op, r={d:.6}\n",
            .{ N, elapsed, @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(N)), c.get() });
    }

    // Hurst bench (smaller N due to per-update OLS cost)
    {
        const H = hurst.HurstEstimator(512);
        var h = H.init();
        const t0 = std.time.nanoTimestamp();
        var price: f64 = 100.0;
        for (0..10_000) |i| {
            price += @sin(@as(f64, @floatFromInt(i)) * 0.01) * 0.5;
            h.update(@log(price));
        }
        const elapsed = std.time.nanoTimestamp() - t0;
        try stdout.print("HurstEstimator(10000 iters): {d}ns total, {d:.0}ns/op, H={d:.4}\n",
            .{ elapsed, @as(f64, @floatFromInt(elapsed)) / 10000.0, h.get() });
    }
}
