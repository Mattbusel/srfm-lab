//! Binary writer: writes processed ticks to shared memory ring buffer.
//! Uses a lock-free ring buffer for IPC with C++ consumers.

const std = @import("std");
const mem = std.mem;
const builtin = @import("builtin");
const os = std.os;

// ============================================================
// Shared memory tick format (must match C++ side)
// ============================================================
pub const SharedTick = extern struct {
    timestamp:  i64  align(8),  // ns since epoch
    symbol:     [16]u8,
    price:      i64,            // fixed-point * 100000
    qty:        u64,
    bid_price:  i64,
    ask_price:  i64,
    bid_qty:    u64,
    ask_qty:    u64,
    trade_id:   u64,
    side:       u8,
    tick_type:  u8,             // 0=trade, 1=quote
    flags:      u8,
    _pad:       [5]u8 = [_]u8{0}**5,
    seq_num:    u32,
    _pad2:      [4]u8 = [_]u8{0}**4,

    comptime {
        std.debug.assert(@sizeOf(SharedTick) == 128);
        std.debug.assert(@alignOf(SharedTick) == 8);
    }
};

// ============================================================
// Shared memory ring buffer header (must match C++ RingBufferHeader)
// ============================================================
const CACHE_LINE: usize = 64;
const HEADER_SIZE: usize = 4096; // one OS page

pub const RingHeader = extern struct {
    magic:        u64 align(64),
    version:      u64,
    capacity:     u64,
    element_size: u64,
    _pad0:        [CACHE_LINE - 4*8]u8 = [_]u8{0} ** (CACHE_LINE - 4*8),

    write_seq:    u64 align(64),  // atomic in C++; Zig uses std.atomic
    _pad1:        [CACHE_LINE - 8]u8 = [_]u8{0} ** (CACHE_LINE - 8),

    read_seq:     u64 align(64),
    _pad2:        [CACHE_LINE - 8]u8 = [_]u8{0} ** (CACHE_LINE - 8),
};

const RING_MAGIC:   u64 = 0xDEADBEEF12345678;
const RING_VERSION: u64 = 1;

// ============================================================
// Shared Memory Ring Buffer Writer
// ============================================================
pub const SharedMemWriter = struct {
    path:       []const u8,
    capacity:   usize,    // number of ticks (power of 2)
    mask:       usize,
    mmap_size:  usize,
    header:     *volatile RingHeader,
    data:       [*]volatile SharedTick,
    write_seq:  std.atomic.Value(u64),
    seq_gen:    u32 = 0,
    stats:      WriterStats = .{},

    // Platform-specific file handle
    fd:         if (builtin.os.tag == .windows) std.os.windows.HANDLE else i32,

    const Self = @This();

    pub const WriterStats = struct {
        ticks_written:  u64 = 0,
        bytes_written:  u64 = 0,
        syncs:          u64 = 0,
    };

    pub fn open(path: []const u8, capacity: usize) !Self {
        const cap = next_pow2(capacity);
        const mmap_size = HEADER_SIZE + cap * @sizeOf(SharedTick);

        const mapping = try create_mapping(path, mmap_size);

        const header_ptr: *volatile RingHeader = @ptrCast(@alignCast(mapping.ptr));
        const data_ptr: [*]volatile SharedTick = @ptrCast(@alignCast(mapping.ptr + HEADER_SIZE));

        // Initialize header if new
        if (header_ptr.magic != RING_MAGIC) {
            header_ptr.magic        = RING_MAGIC;
            header_ptr.version      = RING_VERSION;
            header_ptr.capacity     = cap;
            header_ptr.element_size = @sizeOf(SharedTick);
            @atomicStore(u64, &header_ptr.write_seq, 0, .release);
            @atomicStore(u64, &header_ptr.read_seq, 0, .release);
        }

        const write_seq_val = @atomicLoad(u64, &header_ptr.write_seq, .acquire);

        return Self{
            .path       = path,
            .capacity   = cap,
            .mask       = cap - 1,
            .mmap_size  = mmap_size,
            .header     = header_ptr,
            .data       = data_ptr,
            .write_seq  = std.atomic.Value(u64).init(write_seq_val),
            .fd         = mapping.fd,
        };
    }

    // Write a tick to the ring buffer (O(1), lock-free)
    pub fn write_tick(self: *Self, tick: SharedTick) void {
        var t = tick;
        t.seq_num = self.seq_gen;
        self.seq_gen +%= 1;

        const seq = self.write_seq.fetchAdd(1, .acq_rel);
        self.data[seq & self.mask] = t;
        // Ensure data write is visible before updating header
        @atomicStore(u64, &self.header.write_seq, seq + 1, .release);

        self.stats.ticks_written += 1;
        self.stats.bytes_written += @sizeOf(SharedTick);
    }

    pub fn write_trade(self: *Self, symbol: []const u8, price: i64,
                       qty: u64, side: u8, trade_id: u64,
                       bid: i64, ask: i64, ts_ns: i64) void
    {
        var t: SharedTick = std.mem.zeroes(SharedTick);
        t.timestamp  = ts_ns;
        const len = @min(symbol.len, 15);
        @memcpy(t.symbol[0..len], symbol[0..len]);
        t.price      = price;
        t.qty        = qty;
        t.side       = side;
        t.tick_type  = 0;
        t.trade_id   = trade_id;
        t.bid_price  = bid;
        t.ask_price  = ask;
        self.write_tick(t);
    }

    pub fn write_quote(self: *Self, symbol: []const u8,
                       bid: i64, bid_qty: u64,
                       ask: i64, ask_qty: u64,
                       ts_ns: i64) void
    {
        var t: SharedTick = std.mem.zeroes(SharedTick);
        t.timestamp = ts_ns;
        const len = @min(symbol.len, 15);
        @memcpy(t.symbol[0..len], symbol[0..len]);
        t.price     = (bid + ask) / 2;
        t.bid_price = bid;
        t.ask_price = ask;
        t.bid_qty   = bid_qty;
        t.ask_qty   = ask_qty;
        t.tick_type = 1;
        self.write_tick(t);
    }

    pub fn current_write_seq(self: *const Self) u64 {
        return self.write_seq.load(.acquire);
    }

    pub fn sync(self: *Self) void {
        self.stats.syncs += 1;
        sync_mapping(self.fd, self.mmap_size);
    }

    pub fn close(self: *Self) void {
        self.sync();
        close_mapping(self.fd, self.header, self.mmap_size);
    }
};

// ============================================================
// Platform-specific mmap/CreateFileMapping helpers
// ============================================================
const Mapping = struct {
    ptr: [*]u8,
    fd:  if (builtin.os.tag == .windows) std.os.windows.HANDLE else i32,
};

fn next_pow2(n: usize) usize {
    if (n == 0) return 1;
    var v = n - 1;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16; v |= v >> 32;
    return v + 1;
}

fn create_mapping(path: []const u8, size: usize) !Mapping {
    if (builtin.os.tag == .windows) {
        return create_mapping_windows(path, size);
    } else {
        return create_mapping_posix(path, size);
    }
}

fn create_mapping_posix(path: []const u8, size: usize) !Mapping {
    const fd = try std.posix.open(path,
        .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = false },
        0o666);
    try std.posix.ftruncate(fd, @intCast(size));
    const ptr = try std.posix.mmap(null, size,
        std.posix.PROT.READ | std.posix.PROT.WRITE,
        .{ .TYPE = .SHARED },
        fd, 0);
    return Mapping{ .ptr = ptr.ptr, .fd = fd };
}

fn create_mapping_windows(path: []const u8, size: usize) !Mapping {
    _ = path;
    _ = size;
    // Stub for Windows; actual implementation would use CreateFileMapping
    return error.NotImplemented;
}

fn sync_mapping(fd: anytype, size: usize) void {
    if (builtin.os.tag != .windows) {
        std.posix.msync(@ptrFromInt(0), size, std.posix.MS.ASYNC) catch {};
        _ = fd;
    }
}

fn close_mapping(fd: anytype, ptr: anytype, size: usize) void {
    if (builtin.os.tag != .windows) {
        const p: []align(std.mem.page_size) u8 = @alignCast(@as([*]u8, @ptrCast(ptr))[0..size]);
        std.posix.munmap(p);
        std.posix.close(fd);
    }
}

// ============================================================
// In-memory writer for testing (no mmap)
// ============================================================
pub const InMemWriter = struct {
    buf:       []SharedTick,
    capacity:  usize,
    mask:      usize,
    write_seq: u64 = 0,
    allocator: std.mem.Allocator,
    stats:     SharedMemWriter.WriterStats = .{},

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
        const cap = next_pow2(capacity);
        const buf = try allocator.alloc(SharedTick, cap);
        return Self{
            .buf       = buf,
            .capacity  = cap,
            .mask      = cap - 1,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.buf);
    }

    pub fn write_tick(self: *Self, tick: SharedTick) void {
        const seq = self.write_seq;
        self.buf[seq & self.mask] = tick;
        self.write_seq += 1;
        self.stats.ticks_written += 1;
        self.stats.bytes_written += @sizeOf(SharedTick);
    }

    pub fn read_tick(self: *const Self, seq: u64) ?SharedTick {
        if (seq >= self.write_seq) return null;
        if (self.write_seq > self.capacity and seq < self.write_seq - self.capacity)
            return null; // overwritten
        return self.buf[seq & self.mask];
    }

    pub fn latest(self: *const Self) ?SharedTick {
        if (self.write_seq == 0) return null;
        return self.read_tick(self.write_seq - 1);
    }
};

// ============================================================
// Tests
// ============================================================
test "in-mem writer round-trip" {
    const alloc = std.testing.allocator;
    var w = try InMemWriter.init(alloc, 1024);
    defer w.deinit();

    for (0..100) |i| {
        var t: SharedTick = std.mem.zeroes(SharedTick);
        t.price = @intCast(i * 1000);
        t.qty   = @intCast(i + 1);
        w.write_tick(t);
    }

    try std.testing.expectEqual(@as(u64, 100), w.write_seq);

    for (0..100) |i| {
        const t = w.read_tick(@intCast(i)) orelse return error.MissingTick;
        try std.testing.expectEqual(@as(i64, @intCast(i * 1000)), t.price);
    }

    const latest = w.latest() orelse return error.NoLatest;
    try std.testing.expectEqual(@as(i64, 99_000), latest.price);
}

test "ring buffer overflow" {
    const alloc = std.testing.allocator;
    var w = try InMemWriter.init(alloc, 8); // capacity 8
    defer w.deinit();

    for (0..20) |i| {
        var t: SharedTick = std.mem.zeroes(SharedTick);
        t.price = @intCast(i);
        w.write_tick(t);
    }

    // First 12 should be overwritten
    try std.testing.expectEqual(@as(?SharedTick, null), w.read_tick(0));
    // Last 8 should be readable
    const t12 = w.read_tick(12) orelse return error.Missing;
    try std.testing.expectEqual(@as(i64, 12), t12.price);
}

test "SharedTick size" {
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(SharedTick));
}
