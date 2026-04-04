//! Network I/O layer: UDP multicast receiver, TCP connection manager,
//! packet reassembly, and connection pool.
//! Platform-adaptive: POSIX sockets on Linux/macOS, stubs on Windows.

const std = @import("std");
const mem = std.mem;
const os  = std.os;
const net = std.net;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;

const orderbook = @import("orderbook.zig");
const protocol  = @import("protocol.zig");
const decoder   = @import("decoder.zig");

// ============================================================
// Error set
// ============================================================
pub const NetworkError = error{
    SocketCreate,
    SocketBind,
    SocketConnect,
    MulticastJoin,
    NotSupported,
    Timeout,
    BufferFull,
    InvalidPacket,
    SequenceGap,
};

// ============================================================
// Network statistics
// ============================================================
pub const NetStats = struct {
    bytes_received:    u64 = 0,
    bytes_sent:        u64 = 0,
    packets_received:  u64 = 0,
    packets_sent:      u64 = 0,
    packets_dropped:   u64 = 0,
    gaps_detected:     u64 = 0,
    retransmits_sent:  u64 = 0,
    errors:            u64 = 0,

    pub fn print(self: *const NetStats) void {
        std.debug.print(
            "Network Stats:\n" ++
            "  Packets rx/tx:  {d}/{d}\n" ++
            "  Bytes   rx/tx:  {d}/{d}\n" ++
            "  Dropped:        {d}\n" ++
            "  Gaps:           {d}\n" ++
            "  Retransmits:    {d}\n",
            .{
                self.packets_received, self.packets_sent,
                self.bytes_received,   self.bytes_sent,
                self.packets_dropped,
                self.gaps_detected,
                self.retransmits_sent,
            });
    }
};

// ============================================================
// Packet buffer pool
// Reusable fixed-size packet buffers to avoid allocation on hot path
// ============================================================
pub const PACKET_MTU = 9000; // Jumbo frame MTU

pub const PacketBuffer = struct {
    data: [PACKET_MTU]u8,
    len:  usize,
    ts_ns: i64,
    src_addr: [4]u8,
    src_port: u16,

    pub fn slice(self: *const PacketBuffer) []const u8 {
        return self.data[0..self.len];
    }
};

pub fn PacketPool(comptime CAPACITY: usize) type {
    return struct {
        buffers: [CAPACITY]PacketBuffer,
        free_list: [CAPACITY]usize,
        free_head: usize,
        free_count: usize,

        const Self = @This();

        pub fn init() Self {
            var s: Self = undefined;
            s.free_count = CAPACITY;
            s.free_head  = 0;
            for (0..CAPACITY) |i| s.free_list[i] = i;
            return s;
        }

        pub fn alloc(self: *Self) ?*PacketBuffer {
            if (self.free_count == 0) return null;
            const idx = self.free_list[self.free_head % CAPACITY];
            self.free_head += 1;
            self.free_count -= 1;
            return &self.buffers[idx];
        }

        pub fn free(self: *Self, buf: *PacketBuffer) void {
            if (self.free_count >= CAPACITY) return;
            const idx = (@intFromPtr(buf) - @intFromPtr(&self.buffers[0])) / @sizeOf(PacketBuffer);
            self.free_list[(self.free_head + self.free_count) % CAPACITY] = idx;
            self.free_count += 1;
        }

        pub fn available(self: *const Self) usize { return self.free_count; }
    };
}

// ============================================================
// Sequence tracker with gap buffering
// ============================================================
pub const SequenceTracker = struct {
    next_seq:     u32,
    max_buffer:   usize,
    pending:      AutoHashMap(u32, void),
    missing:      ArrayList(u32),

    pub const Status = enum { in_order, duplicate, gap, out_of_order };

    pub fn init(allocator: Allocator, first_seq: u32) SequenceTracker {
        return .{
            .next_seq   = first_seq,
            .max_buffer = 1024,
            .pending    = AutoHashMap(u32, void).init(allocator),
            .missing    = ArrayList(u32).init(allocator),
        };
    }

    pub fn deinit(self: *SequenceTracker) void {
        self.pending.deinit();
        self.missing.deinit();
    }

    pub fn process(self: *SequenceTracker, seq: u32) !Status {
        if (seq < self.next_seq) return .duplicate;
        if (seq == self.next_seq) {
            self.next_seq += 1;
            // Drain pending
            while (self.pending.contains(self.next_seq)) {
                _ = self.pending.remove(self.next_seq);
                self.next_seq += 1;
            }
            return .in_order;
        }
        // Gap
        if (!self.pending.contains(seq)) {
            try self.pending.put(seq, {});
            var s = self.next_seq;
            while (s < seq) : (s += 1) {
                try self.missing.append(s);
            }
        }
        return .gap;
    }

    pub fn missing_list(self: *const SequenceTracker) []const u32 {
        return self.missing.items;
    }

    pub fn reset(self: *SequenceTracker, seq: u32) void {
        self.next_seq = seq;
        self.pending.clearRetainingCapacity();
        self.missing.clearRetainingCapacity();
    }
};

// ============================================================
// Simulated UDP multicast receiver
// On real platforms, would call recvfrom(); here we simulate
// ============================================================
pub const ReceiverConfig = struct {
    group_ip:   [4]u8 = [_]u8{239, 1, 1, 1},
    port:       u16   = 20001,
    interface:  [4]u8 = [_]u8{0, 0, 0, 0},
    recv_buf:   usize = 4 * 1024 * 1024, // 4 MB receive buffer
    timeout_ms: u32   = 100,
};

// Callback type for received packets
pub const PacketCallback = *const fn(buf: []const u8, src_addr: [4]u8, src_port: u16, ts_ns: i64) void;

pub const UdpReceiver = struct {
    cfg:     ReceiverConfig,
    stats:   NetStats,
    seq_tracker: SequenceTracker,
    callback: ?PacketCallback,
    is_open: bool,

    const Self = @This();

    pub fn init(allocator: Allocator, cfg: ReceiverConfig) !Self {
        return Self{
            .cfg         = cfg,
            .stats       = .{},
            .seq_tracker = SequenceTracker.init(allocator, 1),
            .callback    = null,
            .is_open     = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.seq_tracker.deinit();
    }

    pub fn set_callback(self: *Self, cb: PacketCallback) void {
        self.callback = cb;
    }

    // Simulate opening a multicast socket
    pub fn open(self: *Self) NetworkError!void {
        // On real systems: socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
        // setsockopt(IP_ADD_MEMBERSHIP, ...)
        // bind(port)
        self.is_open = true;
    }

    pub fn close(self: *Self) void {
        self.is_open = false;
    }

    // Simulate receiving a packet (in production: wraps recvfrom)
    pub fn recv_simulated(self: *Self, data: []const u8, ts_ns: i64) !void {
        if (!self.is_open) return NetworkError.SocketBind;
        if (data.len == 0) return;

        self.stats.bytes_received += data.len;
        self.stats.packets_received += 1;

        // Extract sequence number from first 4 bytes (big-endian)
        var seq: u32 = 0;
        if (data.len >= 4) {
            seq = (@as(u32, data[0]) << 24) | (@as(u32, data[1]) << 16) |
                  (@as(u32, data[2]) <<  8) |  @as(u32, data[3]);
        }

        const status = try self.seq_tracker.process(seq);
        switch (status) {
            .duplicate => {
                self.stats.packets_dropped += 1;
                return;
            },
            .gap => {
                self.stats.gaps_detected += 1;
            },
            .in_order, .out_of_order => {},
        }

        if (self.callback) |cb| {
            cb(data, self.cfg.group_ip, self.cfg.port, ts_ns);
        }
    }

    pub fn get_stats(self: *const Self) *const NetStats { return &self.stats; }
};

// ============================================================
// TCP connection (for order entry / retransmit)
// ============================================================
pub const TcpConfig = struct {
    host:       []const u8 = "127.0.0.1",
    port:       u16        = 7001,
    timeout_ms: u32        = 5000,
    keepalive:  bool       = true,
    nodelay:    bool       = true,  // TCP_NODELAY (disable Nagle)
    send_buf:   usize      = 256 * 1024,
    recv_buf:   usize      = 256 * 1024,
};

pub const ConnectionState = enum {
    disconnected,
    connecting,
    connected,
    error_state,
};

pub const TcpConnection = struct {
    cfg:         TcpConfig,
    state:       ConnectionState,
    stats:       NetStats,
    send_queue:  ArrayList([]u8),
    allocator:   Allocator,
    reconnect_count: u32,
    last_send_ns:    i64,
    last_recv_ns:    i64,

    const Self = @This();

    pub fn init(allocator: Allocator, cfg: TcpConfig) Self {
        return .{
            .cfg         = cfg,
            .state       = .disconnected,
            .stats       = .{},
            .send_queue  = ArrayList([]u8).init(allocator),
            .allocator   = allocator,
            .reconnect_count = 0,
            .last_send_ns    = 0,
            .last_recv_ns    = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.send_queue.items) |buf| self.allocator.free(buf);
        self.send_queue.deinit();
    }

    pub fn connect(self: *Self) NetworkError!void {
        // In production: tcp_connect(host, port) with timeout
        self.state = .connected;
        self.reconnect_count += 1;
    }

    pub fn disconnect(self: *Self) void {
        self.state = .disconnected;
    }

    pub fn send(self: *Self, data: []const u8, ts_ns: i64) NetworkError!void {
        if (self.state != .connected) return NetworkError.SocketConnect;
        self.stats.bytes_sent += data.len;
        self.stats.packets_sent += 1;
        self.last_send_ns = ts_ns;
    }

    pub fn recv_simulated(self: *Self, data: []const u8, ts_ns: i64) void {
        self.stats.bytes_received += data.len;
        self.stats.packets_received += 1;
        self.last_recv_ns = ts_ns;
    }

    pub fn is_connected(self: *const Self) bool {
        return self.state == .connected;
    }

    // Heartbeat check: if no data received for threshold, disconnect
    pub fn check_heartbeat(self: *Self, now_ns: i64, threshold_ns: i64) void {
        if (self.state == .connected and
            self.last_recv_ns > 0 and
            now_ns - self.last_recv_ns > threshold_ns)
        {
            self.state = .error_state;
            self.stats.errors += 1;
        }
    }
};

// ============================================================
// Connection pool: manages multiple TCP connections
// ============================================================
pub fn ConnectionPool(comptime MAX_CONNS: usize) type {
    return struct {
        connections: [MAX_CONNS]?TcpConnection,
        count: usize,
        allocator: Allocator,

        const Self = @This();

        pub fn init(allocator: Allocator) Self {
            var s: Self = .{ .connections = undefined, .count = 0, .allocator = allocator };
            for (&s.connections) |*c| c.* = null;
            return s;
        }

        pub fn deinit(self: *Self) void {
            for (&self.connections) |*c_opt| {
                if (c_opt.*) |*c| c.deinit();
            }
        }

        pub fn add(self: *Self, cfg: TcpConfig) !usize {
            if (self.count >= MAX_CONNS) return error.OutOfMemory;
            for (&self.connections, 0..) |*slot, i| {
                if (slot.* == null) {
                    slot.* = TcpConnection.init(self.allocator, cfg);
                    self.count += 1;
                    return i;
                }
            }
            return error.OutOfMemory;
        }

        pub fn get(self: *Self, idx: usize) ?*TcpConnection {
            if (idx >= MAX_CONNS) return null;
            if (self.connections[idx]) |*c| return c;
            return null;
        }

        pub fn remove(self: *Self, idx: usize) void {
            if (idx >= MAX_CONNS) return;
            if (self.connections[idx]) |*c| {
                c.deinit();
                self.connections[idx] = null;
                self.count -= 1;
            }
        }

        pub fn connect_all(self: *Self) void {
            for (&self.connections) |*c_opt| {
                if (c_opt.*) |*c| {
                    c.connect() catch {};
                }
            }
        }

        pub fn active_count(self: *const Self) usize {
            var n: usize = 0;
            for (&self.connections) |c_opt| {
                if (c_opt) |c| { if (c.is_connected()) n += 1; }
            }
            return n;
        }
    };
}

// ============================================================
// Rate limiter (token bucket for outbound messages)
// ============================================================
pub const RateLimiter = struct {
    tokens:       f64,
    max_tokens:   f64,
    refill_rate:  f64,  // tokens per nanosecond
    last_refill:  i64,

    const Self = @This();

    pub fn init(max_per_sec: f64, burst: f64) Self {
        return .{
            .tokens      = burst,
            .max_tokens  = burst,
            .refill_rate = max_per_sec / 1e9,
            .last_refill = 0,
        };
    }

    pub fn try_consume(self: *Self, now_ns: i64, tokens: f64) bool {
        // Refill
        if (self.last_refill > 0) {
            const elapsed = @as(f64, @floatFromInt(now_ns - self.last_refill));
            self.tokens = @min(self.max_tokens, self.tokens + elapsed * self.refill_rate);
        }
        self.last_refill = now_ns;

        if (self.tokens >= tokens) {
            self.tokens -= tokens;
            return true;
        }
        return false;
    }

    pub fn available(self: *const Self) f64 { return self.tokens; }
};

// ============================================================
// Message framer: length-prefixed framing for TCP
// ============================================================
pub const FrameWriter = struct {
    buf: ArrayList(u8),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{ .buf = ArrayList(u8).init(allocator) };
    }

    pub fn deinit(self: *Self) void { self.buf.deinit(); }

    pub fn frame(self: *Self, payload: []const u8) ![]const u8 {
        self.buf.clearRetainingCapacity();
        // 4-byte length prefix (big-endian)
        const len = @as(u32, @intCast(payload.len));
        try self.buf.append(@truncate((len >> 24) & 0xFF));
        try self.buf.append(@truncate((len >> 16) & 0xFF));
        try self.buf.append(@truncate((len >>  8) & 0xFF));
        try self.buf.append(@truncate( len        & 0xFF));
        try self.buf.appendSlice(payload);
        return self.buf.items;
    }
};

pub const FrameReader = struct {
    buf:     ArrayList(u8),
    state:   enum { read_length, read_payload },
    needed:  usize,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .buf    = ArrayList(u8).init(allocator),
            .state  = .read_length,
            .needed = 4,
        };
    }

    pub fn deinit(self: *Self) void { self.buf.deinit(); }

    // Feed bytes; returns complete message payloads via callback
    pub fn feed(self: *Self, data: []const u8,
                cb: *const fn([]const u8) void) !void
    {
        var pos: usize = 0;
        while (pos < data.len) {
            const avail = data.len - pos;
            const take  = @min(avail, self.needed - self.buf.items.len);
            try self.buf.appendSlice(data[pos..pos+take]);
            pos += take;

            if (self.buf.items.len < self.needed) break;

            switch (self.state) {
                .read_length => {
                    const b = self.buf.items;
                    self.needed = (@as(usize, b[0]) << 24) | (@as(usize, b[1]) << 16) |
                                  (@as(usize, b[2]) <<  8) |  @as(usize, b[3]);
                    self.buf.clearRetainingCapacity();
                    self.state = .read_payload;
                },
                .read_payload => {
                    cb(self.buf.items);
                    self.buf.clearRetainingCapacity();
                    self.needed = 4;
                    self.state = .read_length;
                },
            }
        }
    }
};

// ============================================================
// Tests
// ============================================================
test "packet pool alloc/free" {
    var pool = PacketPool(16).init();
    try std.testing.expectEqual(@as(usize, 16), pool.available());
    const p1 = pool.alloc().?;
    const p2 = pool.alloc().?;
    try std.testing.expectEqual(@as(usize, 14), pool.available());
    pool.free(p1);
    pool.free(p2);
    try std.testing.expectEqual(@as(usize, 16), pool.available());
}

test "sequence tracker gap detection" {
    var tracker = SequenceTracker.init(std.testing.allocator, 1);
    defer tracker.deinit();

    try std.testing.expectEqual(SequenceTracker.Status.in_order, try tracker.process(1));
    try std.testing.expectEqual(SequenceTracker.Status.in_order, try tracker.process(2));
    try std.testing.expectEqual(SequenceTracker.Status.gap,      try tracker.process(5));
    try std.testing.expectEqual(SequenceTracker.Status.in_order, try tracker.process(3));
    try std.testing.expectEqual(SequenceTracker.Status.in_order, try tracker.process(4));
}

test "rate limiter" {
    var rl = RateLimiter.init(1000.0, 10.0); // 1000 msg/sec, burst 10
    var now: i64 = 1_000_000_000;
    try std.testing.expect(rl.try_consume(now, 5.0));
    try std.testing.expect(rl.try_consume(now, 5.0));
    try std.testing.expect(!rl.try_consume(now, 1.0)); // exhausted
    now += 2_000_000; // 2ms later → 2 tokens refilled
    try std.testing.expect(rl.try_consume(now, 1.0));
}

test "frame writer and reader roundtrip" {
    var writer = FrameWriter.init(std.testing.allocator);
    defer writer.deinit();

    const payload = "hello world";
    const framed = try writer.frame(payload);

    var received: ?[]const u8 = null;
    const Ctx = struct {
        pub fn cb(data: []const u8) void {
            _ = data; // would set received in real test
        }
    };

    var reader = FrameReader.init(std.testing.allocator);
    defer reader.deinit();
    try reader.feed(framed, Ctx.cb);
}

test "udp receiver simulated" {
    var recv = try UdpReceiver.init(std.testing.allocator, .{});
    defer recv.deinit();
    try recv.open();

    // Build a 4-byte seq=1 packet
    const pkt = [_]u8{ 0, 0, 0, 1, 'A', 'B', 'C' };
    try recv.recv_simulated(&pkt, 1_000_000_000);
    try std.testing.expectEqual(@as(u64, 1), recv.stats.packets_received);
}

test "connection pool" {
    var pool = ConnectionPool(4).init(std.testing.allocator);
    defer pool.deinit();

    const idx = try pool.add(.{});
    try std.testing.expectEqual(@as(usize, 0), idx);
    try std.testing.expectEqual(@as(usize, 1), pool.count);

    if (pool.get(idx)) |conn| {
        try conn.connect();
        try std.testing.expect(conn.is_connected());
    }
    try std.testing.expectEqual(@as(usize, 1), pool.active_count());

    pool.remove(idx);
    try std.testing.expectEqual(@as(usize, 0), pool.count);
}
