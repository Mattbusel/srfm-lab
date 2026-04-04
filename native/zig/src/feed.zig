//! Feed Handler: sequence number tracking, gap detection, retransmit requests.
//! Handles UDP multicast market data feeds with gap recovery.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const decoder = @import("decoder.zig");

// ============================================================
// Sequence tracking per upstream source
// ============================================================
pub const FeedSource = struct {
    source_id:          u16,
    expected_seq:       u64,
    received:           u64,
    gaps_detected:      u64,
    retransmit_requests: u64,
    last_heartbeat_ts:  i64,
    is_active:          bool,
};

pub const GapInfo = struct {
    source_id:    u16,
    start_seq:    u64,
    end_seq:      u64,   // exclusive
    detected_at:  i64,   // ns timestamp
    filled:       bool,
};

// Retransmit request packet format
pub const RetransmitRequest = struct {
    source_id: u16,
    start_seq: u64,
    count:     u32,
};

// ============================================================
// Feed Handler
// ============================================================
pub const FeedConfig = struct {
    max_gap_size:         u64 = 100,       // max missing messages before giving up
    heartbeat_interval_ns: i64 = 1_000_000_000, // 1 second
    retransmit_timeout_ns: i64 = 100_000_000,   // 100ms
    max_pending_requests:  usize = 256,
};

pub const FeedHandler = struct {
    sources:            std.AutoHashMap(u16, FeedSource),
    gaps:               ArrayList(GapInfo),
    pending_requests:   ArrayList(RetransmitRequest),
    config:             FeedConfig,
    allocator:          Allocator,
    total_messages:     u64 = 0,
    total_gaps:         u64 = 0,
    total_retransmits:  u64 = 0,
    out_of_order:       u64 = 0,
    duplicates:         u64 = 0,

    const Self = @This();

    pub fn init(allocator: Allocator, config: FeedConfig) Self {
        return .{
            .sources          = std.AutoHashMap(u16, FeedSource).init(allocator),
            .gaps             = ArrayList(GapInfo).init(allocator),
            .pending_requests = ArrayList(RetransmitRequest).init(allocator),
            .config           = config,
            .allocator        = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.sources.deinit();
        self.gaps.deinit();
        self.pending_requests.deinit();
    }

    // Process incoming message with sequence number
    // Returns: .ok, .gap, .duplicate, .out_of_order
    pub const ProcessResult = enum { ok, gap, duplicate, out_of_order };

    pub fn process_message(self: *Self, source_id: u16, seq: u64,
                            timestamp: i64) !ProcessResult
    {
        self.total_messages += 1;

        // Get or create source state
        const entry = try self.sources.getOrPut(source_id);
        if (!entry.found_existing) {
            entry.value_ptr.* = FeedSource{
                .source_id           = source_id,
                .expected_seq        = seq,   // first message sets baseline
                .received            = 0,
                .gaps_detected       = 0,
                .retransmit_requests = 0,
                .last_heartbeat_ts   = timestamp,
                .is_active           = true,
            };
        }

        const src = entry.value_ptr;
        src.last_heartbeat_ts = timestamp;
        src.received += 1;

        if (seq < src.expected_seq) {
            // Duplicate or retransmit (might fill a gap)
            self.duplicates += 1;
            self.mark_gap_filled(source_id, seq);
            return .duplicate;
        }

        if (seq > src.expected_seq) {
            // Gap detected
            const gap_size = seq - src.expected_seq;
            self.total_gaps += 1;
            src.gaps_detected += 1;

            if (gap_size <= self.config.max_gap_size) {
                // Request retransmit
                try self.request_retransmit(source_id, src.expected_seq,
                    @intCast(gap_size), timestamp);
                const gap = GapInfo{
                    .source_id   = source_id,
                    .start_seq   = src.expected_seq,
                    .end_seq     = seq,
                    .detected_at = timestamp,
                    .filled      = false,
                };
                try self.gaps.append(gap);
            } else {
                // Gap too large: reset sequence tracking
                src.expected_seq = seq + 1;
                return .gap;
            }

            src.expected_seq = seq + 1;
            return .gap;
        }

        // In-order message
        src.expected_seq = seq + 1;
        return .ok;
    }

    fn request_retransmit(self: *Self, source_id: u16, start: u64, count: u32,
                           timestamp: i64) !void
    {
        _ = timestamp;
        if (self.pending_requests.items.len >= self.config.max_pending_requests) {
            // Evict oldest
            _ = self.pending_requests.orderedRemove(0);
        }
        const req = RetransmitRequest{
            .source_id = source_id,
            .start_seq = start,
            .count     = count,
        };
        try self.pending_requests.append(req);
        self.total_retransmits += 1;

        // Retrieve source for stats
        if (self.sources.getPtr(source_id)) |src| {
            src.retransmit_requests += 1;
        }
    }

    fn mark_gap_filled(self: *Self, source_id: u16, seq: u64) void {
        for (self.gaps.items) |*g| {
            if (g.source_id == source_id and
                seq >= g.start_seq and seq < g.end_seq)
            {
                // Check if entire gap is now filled (simplified)
                g.filled = true;
            }
        }
    }

    // Expire stale pending requests older than retransmit_timeout_ns
    pub fn expire_requests(self: *Self, now: i64) void {
        var i: usize = 0;
        while (i < self.gaps.items.len) {
            const gap = &self.gaps.items[i];
            if (!gap.filled and
                now - gap.detected_at > self.config.retransmit_timeout_ns)
            {
                _ = self.gaps.orderedRemove(i);
            } else {
                i += 1;
            }
        }
    }

    // Check for stale sources (heartbeat timeout)
    pub fn check_heartbeats(self: *Self, now: i64) []u16 {
        var stale = ArrayList(u16).initCapacity(self.allocator, 8) catch return &.{};
        var it = self.sources.iterator();
        while (it.next()) |entry| {
            const src = entry.value_ptr;
            if (src.is_active and
                now - src.last_heartbeat_ts > self.config.heartbeat_interval_ns * 3)
            {
                src.is_active = false;
                stale.append(src.source_id) catch {};
            }
        }
        return stale.toOwnedSlice() catch &.{};
    }

    pub fn get_source(self: *const Self, id: u16) ?*const FeedSource {
        return self.sources.getPtr(id);
    }

    pub fn pending_gaps(self: *const Self) []const GapInfo {
        return self.gaps.items;
    }

    pub fn pending_retransmit_count(self: *const Self) usize {
        return self.pending_requests.items.len;
    }
};

// ============================================================
// UDP Multicast listener simulation
// In a real system this would use posix sockets.
// We simulate packet arrival with synthetic data.
// ============================================================
pub const PacketHeader = struct {
    source_id:   u16,
    seq_num:     u64,
    timestamp:   i64,   // ns
    payload_len: u16,
};

pub const PacketStats = struct {
    packets_received:  u64 = 0,
    bytes_received:    u64 = 0,
    packets_dropped:   u64 = 0,
    messages_processed: u64 = 0,
};

pub const UdpReceiver = struct {
    feed:       *FeedHandler,
    dec:        decoder.Decoder,
    stats:      PacketStats,
    allocator:  Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, feed: *FeedHandler) Self {
        return .{
            .feed      = feed,
            .dec       = .{},
            .stats     = .{},
            .allocator = allocator,
        };
    }

    // Simulate receiving a UDP packet
    // In real code: recvmsg() or recvfrom() on multicast socket
    pub fn receive_packet(self: *Self, header: PacketHeader, payload: []const u8) !void {
        self.stats.packets_received += 1;
        self.stats.bytes_received += @sizeOf(PacketHeader) + payload.len;

        // Feed handler sequence tracking
        const result = try self.feed.process_message(
            header.source_id, header.seq_num, header.timestamp);
        _ = result;

        // Decode ITCH messages in payload
        var pos: usize = 0;
        while (pos < payload.len) {
            const decode_result = self.dec.decode_one(payload[pos..]) catch break;
            pos += decode_result.len;
            self.stats.messages_processed += 1;
        }
    }
};

// ============================================================
// Tests
// ============================================================
test "sequence tracking - in order" {
    const alloc = std.testing.allocator;
    var feed = FeedHandler.init(alloc, .{});
    defer feed.deinit();

    for (0..100) |i| {
        const r = try feed.process_message(1, @intCast(i), std.time.nanoTimestamp());
        if (i == 0) try std.testing.expectEqual(ProcessResult.ok, r)
        else        try std.testing.expectEqual(ProcessResult.ok, r);
    }
    try std.testing.expectEqual(@as(u64, 0), feed.total_gaps);
}

test "gap detection" {
    const alloc = std.testing.allocator;
    var feed = FeedHandler.init(alloc, .{});
    defer feed.deinit();

    _ = try feed.process_message(1, 0, 0);
    _ = try feed.process_message(1, 1, 0);
    // Skip sequences 2 and 3
    const r = try feed.process_message(1, 4, 0);
    try std.testing.expectEqual(FeedHandler.ProcessResult.gap, r);
    try std.testing.expectEqual(@as(u64, 1), feed.total_gaps);
    try std.testing.expectEqual(@as(usize, 1), feed.pending_gaps().len);
}

test "duplicate detection" {
    const alloc = std.testing.allocator;
    var feed = FeedHandler.init(alloc, .{});
    defer feed.deinit();

    _ = try feed.process_message(1, 0, 0);
    _ = try feed.process_message(1, 1, 0);
    _ = try feed.process_message(1, 2, 0);
    const r = try feed.process_message(1, 1, 0); // duplicate
    try std.testing.expectEqual(FeedHandler.ProcessResult.duplicate, r);
    try std.testing.expectEqual(@as(u64, 1), feed.duplicates);
}

const ProcessResult = FeedHandler.ProcessResult;
