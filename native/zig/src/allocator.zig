//! Custom allocators optimized for low-latency market data processing.
//! - ArenaAllocator: bulk-free arena for order book state
//! - FixedBufferAllocator: hot-path stack allocator (no heap)
//! - PoolAllocator: fixed-size object pool for Order structs

const std = @import("std");
const mem = std.mem;
const assert = std.debug.assert;

// ============================================================
// Arena Allocator
// All allocations served from a single large backing buffer.
// Reset (free all) is O(1). Individual frees are no-ops.
// ============================================================
pub const ArenaAllocator = struct {
    backing: []u8,
    offset: usize,
    peak: usize,
    total_allocs: u64,

    const Self = @This();

    pub fn init(backing: []u8) Self {
        return Self{
            .backing     = backing,
            .offset      = 0,
            .peak        = 0,
            .total_allocs = 0,
        };
    }

    pub fn allocator(self: *Self) mem.Allocator {
        return mem.Allocator{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    const vtable = mem.Allocator.VTable{
        .alloc   = alloc,
        .resize  = resize,
        .free    = free,
    };

    fn alloc(ctx: *anyopaque, n: usize, log2_align: u8, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *Self = @ptrCast(@alignCast(ctx));
        const alignment = @as(usize, 1) << @intCast(log2_align);

        const aligned_offset = mem.alignForward(usize, self.offset, alignment);
        const new_offset = aligned_offset + n;

        if (new_offset > self.backing.len) return null;

        self.offset = new_offset;
        if (self.offset > self.peak) self.peak = self.offset;
        self.total_allocs += 1;

        return self.backing[aligned_offset..].ptr;
    }

    fn resize(_: *anyopaque, buf: []u8, _: u8, new_len: usize, _: usize) bool {
        // Only support shrinking (no-op) or same size
        return new_len <= buf.len;
    }

    fn free(_: *anyopaque, _: []u8, _: u8, _: usize) void {
        // No-op: arena frees everything at once
    }

    pub fn reset(self: *Self) void {
        self.offset = 0;
    }

    pub fn reset_to_mark(self: *Self, mark: usize) void {
        assert(mark <= self.offset);
        self.offset = mark;
    }

    pub fn mark(self: *const Self) usize {
        return self.offset;
    }

    pub fn used(self: *const Self) usize  { return self.offset; }
    pub fn available(self: *const Self) usize { return self.backing.len - self.offset; }
    pub fn utilization(self: *const Self) f64 {
        return @as(f64, @floatFromInt(self.offset)) /
               @as(f64, @floatFromInt(self.backing.len));
    }
};

// ============================================================
// Fixed Buffer Allocator (stack-based, no heap)
// Used on the hot path where malloc latency is unacceptable.
// Buffer is embedded in the struct (stack-allocated).
// ============================================================
pub fn FixedBufferAllocator(comptime SIZE: usize) type {
    return struct {
        buf: [SIZE]u8 align(64) = undefined,
        offset: usize = 0,

        const Self = @This();

        pub fn allocator(self: *Self) mem.Allocator {
            return mem.Allocator{
                .ptr    = self,
                .vtable = &vtable,
            };
        }

        const vtable = mem.Allocator.VTable{
            .alloc  = alloc,
            .resize = resize,
            .free   = free,
        };

        fn alloc(ctx: *anyopaque, n: usize, log2_align: u8, _: usize) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const alignment = @as(usize, 1) << @intCast(log2_align);
            const aligned  = mem.alignForward(usize, self.offset, alignment);
            const end = aligned + n;
            if (end > SIZE) return null;
            self.offset = end;
            return self.buf[aligned..].ptr;
        }

        fn resize(_: *anyopaque, buf: []u8, _: u8, new_len: usize, _: usize) bool {
            return new_len <= buf.len;
        }

        fn free(_: *anyopaque, _: []u8, _: u8, _: usize) void {}

        pub fn reset(self: *Self) void { self.offset = 0; }
        pub fn used(self: *const Self) usize { return self.offset; }
        pub fn remaining(self: *const Self) usize { return SIZE - self.offset; }
    };
}

// ============================================================
// Pool Allocator for fixed-size objects
// Uses a free-list for O(1) alloc/free with zero fragmentation.
// ============================================================
pub fn PoolAllocator(comptime T: type, comptime CAPACITY: usize) type {
    return struct {
        slots:      [CAPACITY]T = undefined,
        free_list:  [CAPACITY]u32 = undefined,
        free_count: u32 = CAPACITY,
        alloc_count: u64 = 0,
        free_total:  u64 = 0,

        const Self = @This();

        pub fn init(self: *Self) void {
            for (0..CAPACITY) |i| self.free_list[i] = @intCast(CAPACITY - 1 - i);
            self.free_count  = CAPACITY;
            self.alloc_count = 0;
            self.free_total  = 0;
        }

        pub fn alloc(self: *Self) ?*T {
            if (self.free_count == 0) return null;
            self.free_count -= 1;
            const idx = self.free_list[self.free_count];
            self.alloc_count += 1;
            return &self.slots[idx];
        }

        pub fn free(self: *Self, ptr: *T) void {
            const idx: u32 = @intCast((@intFromPtr(ptr) - @intFromPtr(&self.slots[0])) / @sizeOf(T));
            assert(idx < CAPACITY);
            assert(self.free_count < CAPACITY);
            self.free_list[self.free_count] = idx;
            self.free_count += 1;
            self.free_total += 1;
        }

        pub fn available(self: *const Self) u32  { return self.free_count; }
        pub fn in_use(self: *const Self) u32     { return CAPACITY - self.free_count; }
        pub fn capacity(self: *const Self) usize { return CAPACITY; }

        pub fn owns(self: *const Self, ptr: *const T) bool {
            const base = @intFromPtr(&self.slots[0]);
            const p    = @intFromPtr(ptr);
            return p >= base and p < base + CAPACITY * @sizeOf(T)
                   and (p - base) % @sizeOf(T) == 0;
        }
    };
}

// ============================================================
// Slab Allocator: multiple pools of different sizes
// Routes allocations to the smallest fitting pool.
// ============================================================
pub const SlabAllocator = struct {
    // Slab sizes: 32, 64, 128, 256, 512 bytes
    // Each slab holds 1024 objects
    const SLAB_32  = 32;
    const SLAB_64  = 64;
    const SLAB_128 = 128;
    const SLAB_256 = 256;
    const SLAB_CAP = 1024;

    slab32:  [SLAB_CAP][SLAB_32]u8  = undefined,
    slab64:  [SLAB_CAP][SLAB_64]u8  = undefined,
    slab128: [SLAB_CAP][SLAB_128]u8 = undefined,
    slab256: [SLAB_CAP][SLAB_256]u8 = undefined,
    free32:  [SLAB_CAP]u16 = undefined,
    free64:  [SLAB_CAP]u16 = undefined,
    free128: [SLAB_CAP]u16 = undefined,
    free256: [SLAB_CAP]u16 = undefined,
    n32:     u32 = SLAB_CAP,
    n64:     u32 = SLAB_CAP,
    n128:    u32 = SLAB_CAP,
    n256:    u32 = SLAB_CAP,
    fallback: ?mem.Allocator = null,

    const Self = @This();

    pub fn init(self: *Self, fallback: ?mem.Allocator) void {
        for (0..SLAB_CAP) |i| {
            self.free32[i]  = @intCast(SLAB_CAP - 1 - i);
            self.free64[i]  = @intCast(SLAB_CAP - 1 - i);
            self.free128[i] = @intCast(SLAB_CAP - 1 - i);
            self.free256[i] = @intCast(SLAB_CAP - 1 - i);
        }
        self.fallback = fallback;
    }

    pub fn alloc_raw(self: *Self, size: usize) ?[]u8 {
        if (size <= SLAB_32 and self.n32 > 0) {
            self.n32 -= 1;
            const idx = self.free32[self.n32];
            return &self.slab32[idx];
        }
        if (size <= SLAB_64 and self.n64 > 0) {
            self.n64 -= 1;
            const idx = self.free64[self.n64];
            return &self.slab64[idx];
        }
        if (size <= SLAB_128 and self.n128 > 0) {
            self.n128 -= 1;
            const idx = self.free128[self.n128];
            return &self.slab128[idx];
        }
        if (size <= SLAB_256 and self.n256 > 0) {
            self.n256 -= 1;
            const idx = self.free256[self.n256];
            return &self.slab256[idx];
        }
        if (self.fallback) |fb| {
            return fb.alloc(u8, size) catch null;
        }
        return null;
    }

    pub fn stats(self: *const Self) SlabStats {
        return SlabStats{
            .available_32  = self.n32,
            .available_64  = self.n64,
            .available_128 = self.n128,
            .available_256 = self.n256,
        };
    }
};

pub const SlabStats = struct {
    available_32:  u32,
    available_64:  u32,
    available_128: u32,
    available_256: u32,
};

// ============================================================
// Tests
// ============================================================
test "ArenaAllocator basic" {
    var backing: [65536]u8 = undefined;
    var arena = ArenaAllocator.init(&backing);
    const ally = arena.allocator();

    const a = try ally.alloc(u32, 10);
    const b = try ally.alloc(u64, 5);

    for (a, 0..) |*v, i| v.* = @intCast(i);
    for (b, 0..) |*v, i| v.* = @intCast(i * 100);

    try std.testing.expectEqual(@as(u32, 9), a[9]);
    try std.testing.expectEqual(@as(u64, 400), b[4]);
    try std.testing.expect(arena.used() > 0);

    arena.reset();
    try std.testing.expectEqual(@as(usize, 0), arena.used());
}

test "FixedBufferAllocator" {
    var fba = FixedBufferAllocator(4096){};
    const ally = fba.allocator();

    const buf = try ally.alloc(u8, 100);
    try std.testing.expectEqual(@as(usize, 100), buf.len);
    try std.testing.expect(fba.used() >= 100);
    fba.reset();
    try std.testing.expectEqual(@as(usize, 0), fba.used());
}

test "PoolAllocator" {
    const MyStruct = struct { x: u64, y: f64 };
    var pool: PoolAllocator(MyStruct, 128) = undefined;
    pool.init();

    try std.testing.expectEqual(@as(u32, 128), pool.available());

    const p1 = pool.alloc() orelse return error.PoolEmpty;
    const p2 = pool.alloc() orelse return error.PoolEmpty;
    p1.x = 42; p1.y = 3.14;
    p2.x = 99; p2.y = 2.71;

    try std.testing.expectEqual(@as(u32, 126), pool.available());
    try std.testing.expectEqual(@as(u64, 42), p1.x);

    pool.free(p1);
    try std.testing.expectEqual(@as(u32, 127), pool.available());
}
