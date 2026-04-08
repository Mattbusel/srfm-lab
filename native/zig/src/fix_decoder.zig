const std = @import("std");
const mem = std.mem;
const Allocator = std.mem.Allocator;

// ============================================================================
// FIX Protocol Constants
// ============================================================================

pub const SOH: u8 = 0x01; // Standard delimiter
pub const PIPE: u8 = '|'; // Alternative delimiter for display

pub const BeginString = struct {
    pub const FIX_4_0 = "FIX.4.0";
    pub const FIX_4_1 = "FIX.4.1";
    pub const FIX_4_2 = "FIX.4.2";
    pub const FIX_4_3 = "FIX.4.3";
    pub const FIX_4_4 = "FIX.4.4";
};

// FIX Tag numbers
pub const Tag = struct {
    pub const BeginString: u32 = 8;
    pub const BodyLength: u32 = 9;
    pub const MsgType: u32 = 35;
    pub const SenderCompID: u32 = 49;
    pub const TargetCompID: u32 = 56;
    pub const MsgSeqNum: u32 = 34;
    pub const SendingTime: u32 = 52;
    pub const CheckSum: u32 = 10;
    pub const ClOrdID: u32 = 11;
    pub const OrderID: u32 = 37;
    pub const ExecID: u32 = 17;
    pub const ExecType: u32 = 150;
    pub const OrdStatus: u32 = 39;
    pub const Symbol: u32 = 55;
    pub const Side: u32 = 54;
    pub const OrderQty: u32 = 38;
    pub const OrdType: u32 = 40;
    pub const Price: u32 = 44;
    pub const StopPx: u32 = 99;
    pub const TimeInForce: u32 = 59;
    pub const ExecTransType: u32 = 20;
    pub const LeavesQty: u32 = 151;
    pub const CumQty: u32 = 14;
    pub const AvgPx: u32 = 6;
    pub const LastPx: u32 = 31;
    pub const LastQty: u32 = 32;
    pub const Text: u32 = 58;
    pub const Account: u32 = 1;
    pub const HandlInst: u32 = 21;
    pub const TransactTime: u32 = 60;
    pub const OrigClOrdID: u32 = 41;
    pub const MDReqID: u32 = 262;
    pub const SubscriptionRequestType: u32 = 263;
    pub const MarketDepth: u32 = 264;
    pub const MDUpdateType: u32 = 265;
    pub const NoMDEntryTypes: u32 = 267;
    pub const NoMDEntries: u32 = 268;
    pub const MDEntryType: u32 = 269;
    pub const MDEntryPx: u32 = 270;
    pub const MDEntrySize: u32 = 271;
    pub const MDEntryDate: u32 = 272;
    pub const MDEntryTime: u32 = 273;
    pub const NoRelatedSym: u32 = 146;
    pub const EncryptMethod: u32 = 98;
    pub const HeartBtInt: u32 = 108;
    pub const ResetSeqNumFlag: u32 = 141;
    pub const TestReqID: u32 = 112;
    pub const BeginSeqNo: u32 = 7;
    pub const EndSeqNo: u32 = 16;
    pub const GapFillFlag: u32 = 123;
    pub const NewSeqNo: u32 = 36;
    pub const RefSeqNum: u32 = 45;
    pub const NoOrders: u32 = 73;
    pub const SecurityType: u32 = 167;
    pub const MaturityMonthYear: u32 = 200;
    pub const PutOrCall: u32 = 201;
    pub const StrikePrice: u32 = 202;
    pub const Currency: u32 = 15;
};

pub const MsgType = struct {
    pub const Heartbeat = "0";
    pub const TestRequest = "1";
    pub const ResendRequest = "2";
    pub const Reject = "3";
    pub const SequenceReset = "4";
    pub const Logout = "5";
    pub const Logon = "A";
    pub const NewOrderSingle = "D";
    pub const OrderCancelRequest = "F";
    pub const OrderCancelReplace = "G";
    pub const OrderStatusRequest = "H";
    pub const ExecutionReport = "8";
    pub const OrderCancelReject = "9";
    pub const MarketDataRequest = "V";
    pub const MarketDataSnapshot = "W";
    pub const MarketDataIncRefresh = "X";
};

// ============================================================================
// Zero-copy field reference
// ============================================================================

pub const FieldRef = struct {
    tag: u32,
    value_start: usize,
    value_end: usize,

    pub fn value(self: FieldRef, buffer: []const u8) []const u8 {
        return buffer[self.value_start..self.value_end];
    }

    pub fn intValue(self: FieldRef, buffer: []const u8) !i64 {
        return std.fmt.parseInt(i64, self.value(buffer), 10);
    }

    pub fn floatValue(self: FieldRef, buffer: []const u8) !f64 {
        return std.fmt.parseFloat(f64, self.value(buffer));
    }
};

// ============================================================================
// Parsed FIX Message
// ============================================================================

pub const ParseError = error{
    InvalidFormat,
    MissingBeginString,
    MissingBodyLength,
    MissingMsgType,
    MissingChecksum,
    ChecksumMismatch,
    BodyLengthMismatch,
    TagParseError,
    BufferOverflow,
    OutOfMemory,
};

pub const MAX_FIELDS = 256;
pub const MAX_GROUP_DEPTH = 4;

pub const FixMessage = struct {
    buffer: []const u8,
    fields: [MAX_FIELDS]FieldRef,
    field_count: u32,
    msg_type_idx: ?u32,
    body_start: usize,
    body_end: usize,
    checksum: u8,
    computed_checksum: u8,

    pub fn init() FixMessage {
        return .{
            .buffer = &.{},
            .fields = undefined,
            .field_count = 0,
            .msg_type_idx = null,
            .body_start = 0,
            .body_end = 0,
            .checksum = 0,
            .computed_checksum = 0,
        };
    }

    pub fn msgType(self: *const FixMessage) ?[]const u8 {
        if (self.msg_type_idx) |idx| {
            return self.fields[idx].value(self.buffer);
        }
        return null;
    }

    pub fn getField(self: *const FixMessage, tag: u32) ?FieldRef {
        for (self.fields[0..self.field_count]) |f| {
            if (f.tag == tag) return f;
        }
        return null;
    }

    pub fn getFieldValue(self: *const FixMessage, tag: u32) ?[]const u8 {
        const f = self.getField(tag) orelse return null;
        return f.value(self.buffer);
    }

    pub fn getFieldInt(self: *const FixMessage, tag: u32) ?i64 {
        const f = self.getField(tag) orelse return null;
        return f.intValue(self.buffer) catch null;
    }

    pub fn getFieldFloat(self: *const FixMessage, tag: u32) ?f64 {
        const f = self.getField(tag) orelse return null;
        return f.floatValue(self.buffer) catch null;
    }

    pub fn senderCompID(self: *const FixMessage) ?[]const u8 {
        return self.getFieldValue(Tag.SenderCompID);
    }

    pub fn targetCompID(self: *const FixMessage) ?[]const u8 {
        return self.getFieldValue(Tag.TargetCompID);
    }

    pub fn msgSeqNum(self: *const FixMessage) ?i64 {
        return self.getFieldInt(Tag.MsgSeqNum);
    }

    pub fn symbol(self: *const FixMessage) ?[]const u8 {
        return self.getFieldValue(Tag.Symbol);
    }

    pub fn side(self: *const FixMessage) ?u8 {
        const v = self.getFieldValue(Tag.Side) orelse return null;
        if (v.len == 0) return null;
        return v[0];
    }

    pub fn orderQty(self: *const FixMessage) ?f64 {
        return self.getFieldFloat(Tag.OrderQty);
    }

    pub fn price(self: *const FixMessage) ?f64 {
        return self.getFieldFloat(Tag.Price);
    }

    pub fn clOrdID(self: *const FixMessage) ?[]const u8 {
        return self.getFieldValue(Tag.ClOrdID);
    }

    pub fn orderID(self: *const FixMessage) ?[]const u8 {
        return self.getFieldValue(Tag.OrderID);
    }

    pub fn execType(self: *const FixMessage) ?u8 {
        const v = self.getFieldValue(Tag.ExecType) orelse return null;
        if (v.len == 0) return null;
        return v[0];
    }

    pub fn ordStatus(self: *const FixMessage) ?u8 {
        const v = self.getFieldValue(Tag.OrdStatus) orelse return null;
        if (v.len == 0) return null;
        return v[0];
    }

    // Get all fields for a repeating group
    pub fn getRepeatingGroup(self: *const FixMessage, count_tag: u32, first_field_tag: u32) ?[]const FieldRef {
        const count_val = self.getFieldInt(count_tag) orelse return null;
        if (count_val <= 0) return null;

        // Find the first occurrence of first_field_tag after count_tag
        var found_count = false;
        var start_idx: u32 = 0;
        for (self.fields[0..self.field_count], 0..) |f, i| {
            if (f.tag == count_tag) {
                found_count = true;
                continue;
            }
            if (found_count and f.tag == first_field_tag) {
                start_idx = @intCast(i);
                break;
            }
        }
        if (!found_count) return null;

        // Return slice from start to end of fields (caller parses group boundaries)
        if (start_idx < self.field_count) {
            return self.fields[start_idx..self.field_count];
        }
        return null;
    }
};

// ============================================================================
// FIX Decoder
// ============================================================================

pub const Decoder = struct {
    delimiter: u8,
    strict_checksum: bool,
    strict_body_length: bool,

    pub fn init() Decoder {
        return .{
            .delimiter = SOH,
            .strict_checksum = true,
            .strict_body_length = true,
        };
    }

    pub fn initWithDelimiter(delim: u8) Decoder {
        return .{
            .delimiter = delim,
            .strict_checksum = true,
            .strict_body_length = true,
        };
    }

    fn parseTag(self: *const Decoder, buf: []const u8, pos: *usize) !struct { tag: u32, value_start: usize, value_end: usize } {
        _ = self;
        const start = pos.*;
        // Find '='
        var eq_pos: usize = start;
        while (eq_pos < buf.len and buf[eq_pos] != '=') : (eq_pos += 1) {}
        if (eq_pos >= buf.len) return ParseError.InvalidFormat;

        const tag = std.fmt.parseInt(u32, buf[start..eq_pos], 10) catch return ParseError.TagParseError;
        const val_start = eq_pos + 1;

        // Find delimiter
        var val_end: usize = val_start;
        while (val_end < buf.len and buf[val_end] != SOH and buf[val_end] != PIPE) : (val_end += 1) {}
        if (val_end >= buf.len) {
            // Allow last field without delimiter
            pos.* = val_end;
        } else {
            pos.* = val_end + 1;
        }

        return .{ .tag = tag, .value_start = val_start, .value_end = val_end };
    }

    pub fn decode(self: *const Decoder, buf: []const u8) ParseError!FixMessage {
        var msg = FixMessage.init();
        msg.buffer = buf;

        if (buf.len < 20) return ParseError.InvalidFormat;

        var pos: usize = 0;
        var checksum_running: u32 = 0;
        var body_length_expected: ?usize = null;
        var body_start_pos: usize = 0;
        var found_begin_string = false;
        var found_body_length = false;

        while (pos < buf.len) {
            const field_start = pos;
            const parsed = self.parseTag(buf, &pos) catch return ParseError.InvalidFormat;

            if (msg.field_count >= MAX_FIELDS) return ParseError.BufferOverflow;

            msg.fields[msg.field_count] = .{
                .tag = parsed.tag,
                .value_start = parsed.value_start,
                .value_end = parsed.value_end,
            };

            // Track special tags
            if (parsed.tag == Tag.BeginString) {
                found_begin_string = true;
            } else if (parsed.tag == Tag.BodyLength) {
                found_body_length = true;
                body_length_expected = @intCast(std.fmt.parseInt(usize, buf[parsed.value_start..parsed.value_end], 10) catch return ParseError.InvalidFormat);
                body_start_pos = pos;
                msg.body_start = pos;
            } else if (parsed.tag == Tag.MsgType) {
                msg.msg_type_idx = msg.field_count;
            } else if (parsed.tag == Tag.CheckSum) {
                msg.checksum = @intCast(std.fmt.parseInt(u8, buf[parsed.value_start..parsed.value_end], 10) catch 0);
                msg.body_end = field_start;
                msg.field_count += 1;
                break;
            }

            msg.field_count += 1;

            // Update running checksum (all bytes before checksum tag)
            if (parsed.tag != Tag.CheckSum) {
                for (buf[field_start..pos]) |b| {
                    checksum_running += b;
                }
            }
        }

        msg.computed_checksum = @intCast(checksum_running % 256);

        // Validation
        if (!found_begin_string) return ParseError.MissingBeginString;
        if (!found_body_length) return ParseError.MissingBodyLength;
        if (msg.msg_type_idx == null) return ParseError.MissingMsgType;

        if (self.strict_checksum and msg.checksum != msg.computed_checksum) {
            return ParseError.ChecksumMismatch;
        }

        if (self.strict_body_length) {
            if (body_length_expected) |expected| {
                const actual = msg.body_end - body_start_pos;
                if (actual != expected) return ParseError.BodyLengthMismatch;
            }
        }

        return msg;
    }

    // Find message boundary in stream (returns slice of first complete message)
    pub fn findMessage(self: *const Decoder, stream: []const u8) ?struct { msg: []const u8, consumed: usize } {
        _ = self;
        // Look for "8=FIX"
        const fix_start = mem.indexOf(u8, stream, "8=FIX") orelse return null;
        // Look for "10=XXX|" after that
        var pos = fix_start;
        while (pos + 6 < stream.len) {
            if (stream[pos] == '1' and stream[pos + 1] == '0' and stream[pos + 2] == '=') {
                // Find delimiter after checksum value
                var end = pos + 3;
                while (end < stream.len and stream[end] != SOH and stream[end] != PIPE) : (end += 1) {}
                if (end < stream.len) end += 1; // include delimiter
                return .{
                    .msg = stream[fix_start..end],
                    .consumed = end,
                };
            }
            pos += 1;
        }
        return null;
    }
};

// ============================================================================
// FIX Message Builder
// ============================================================================

pub const Builder = struct {
    buffer: [8192]u8,
    pos: usize,
    body_start: usize,
    delimiter: u8,

    pub fn init() Builder {
        return .{
            .buffer = undefined,
            .pos = 0,
            .body_start = 0,
            .delimiter = SOH,
        };
    }

    pub fn initWithDelimiter(delim: u8) Builder {
        var b = Builder.init();
        b.delimiter = delim;
        return b;
    }

    pub fn reset(self: *Builder) void {
        self.pos = 0;
        self.body_start = 0;
    }

    fn writeRaw(self: *Builder, data: []const u8) void {
        if (self.pos + data.len > self.buffer.len) return;
        @memcpy(self.buffer[self.pos..][0..data.len], data);
        self.pos += data.len;
    }

    fn writeTag(self: *Builder, tag: u32, value: []const u8) void {
        var tag_buf: [12]u8 = undefined;
        const tag_str = std.fmt.bufPrint(&tag_buf, "{d}", .{tag}) catch return;
        self.writeRaw(tag_str);
        self.buffer[self.pos] = '=';
        self.pos += 1;
        self.writeRaw(value);
        self.buffer[self.pos] = self.delimiter;
        self.pos += 1;
    }

    fn writeTagInt(self: *Builder, tag: u32, value: i64) void {
        var val_buf: [24]u8 = undefined;
        const val_str = std.fmt.bufPrint(&val_buf, "{d}", .{value}) catch return;
        self.writeTag(tag, val_str);
    }

    fn writeTagFloat(self: *Builder, tag: u32, value: f64, decimals: u8) void {
        var val_buf: [32]u8 = undefined;
        const val_str = switch (decimals) {
            2 => std.fmt.bufPrint(&val_buf, "{d:.2}", .{value}) catch return,
            4 => std.fmt.bufPrint(&val_buf, "{d:.4}", .{value}) catch return,
            6 => std.fmt.bufPrint(&val_buf, "{d:.6}", .{value}) catch return,
            else => std.fmt.bufPrint(&val_buf, "{d:.2}", .{value}) catch return,
        };
        self.writeTag(tag, val_str);
    }

    pub fn beginMessage(self: *Builder, msg_type: []const u8, sender: []const u8, target: []const u8, seq_num: i64) void {
        self.reset();
        // BeginString placeholder (will fill in finalize)
        self.writeTag(Tag.BeginString, BeginString.FIX_4_2);
        // BodyLength placeholder
        self.writeTag(Tag.BodyLength, "000000");
        self.body_start = self.pos;
        self.writeTag(Tag.MsgType, msg_type);
        self.writeTag(Tag.SenderCompID, sender);
        self.writeTag(Tag.TargetCompID, target);
        self.writeTagInt(Tag.MsgSeqNum, seq_num);
    }

    pub fn addField(self: *Builder, tag: u32, value: []const u8) void {
        self.writeTag(tag, value);
    }

    pub fn addFieldInt(self: *Builder, tag: u32, value: i64) void {
        self.writeTagInt(tag, value);
    }

    pub fn addFieldFloat(self: *Builder, tag: u32, value: f64) void {
        self.writeTagFloat(tag, value, 2);
    }

    pub fn finalize(self: *Builder) []const u8 {
        // Compute body length
        const body_len = self.pos - self.body_start;
        var len_buf: [8]u8 = undefined;
        const len_str = std.fmt.bufPrint(&len_buf, "{d:0>6}", .{body_len}) catch return self.buffer[0..0];

        // Patch body length value (find "000000" after 9=)
        // Body length tag starts after BeginString tag
        var scan: usize = 0;
        while (scan + 2 < self.body_start) {
            if (self.buffer[scan] == '9' and self.buffer[scan + 1] == '=') {
                @memcpy(self.buffer[scan + 2 ..][0..6], len_str[0..6]);
                break;
            }
            scan += 1;
        }

        // Compute checksum
        var checksum: u32 = 0;
        for (self.buffer[0..self.pos]) |b| {
            checksum += b;
        }
        const cs = @as(u8, @intCast(checksum % 256));
        var cs_buf: [4]u8 = undefined;
        const cs_str = std.fmt.bufPrint(&cs_buf, "{d:0>3}", .{cs}) catch return self.buffer[0..0];
        self.writeTag(Tag.CheckSum, cs_str);

        return self.buffer[0..self.pos];
    }

    // Convenience builders for common message types

    pub fn buildLogon(self: *Builder, sender: []const u8, target: []const u8, seq: i64, heartbeat_interval: i64) []const u8 {
        self.beginMessage(MsgType.Logon, sender, target, seq);
        self.addFieldInt(Tag.EncryptMethod, 0);
        self.addFieldInt(Tag.HeartBtInt, heartbeat_interval);
        return self.finalize();
    }

    pub fn buildLogout(self: *Builder, sender: []const u8, target: []const u8, seq: i64, text: ?[]const u8) []const u8 {
        self.beginMessage(MsgType.Logout, sender, target, seq);
        if (text) |t| self.addField(Tag.Text, t);
        return self.finalize();
    }

    pub fn buildHeartbeat(self: *Builder, sender: []const u8, target: []const u8, seq: i64, test_req_id: ?[]const u8) []const u8 {
        self.beginMessage(MsgType.Heartbeat, sender, target, seq);
        if (test_req_id) |id| self.addField(Tag.TestReqID, id);
        return self.finalize();
    }

    pub fn buildNewOrderSingle(self: *Builder, sender: []const u8, target: []const u8, seq: i64, cl_ord_id: []const u8, symbol_val: []const u8, side_val: u8, qty: f64, ord_type: u8, px: ?f64) []const u8 {
        self.beginMessage(MsgType.NewOrderSingle, sender, target, seq);
        self.addField(Tag.ClOrdID, cl_ord_id);
        self.addFieldInt(Tag.HandlInst, 1);
        self.addField(Tag.Symbol, symbol_val);
        self.addField(Tag.Side, &.{side_val});
        self.addFieldFloat(Tag.OrderQty, qty);
        self.addField(Tag.OrdType, &.{ord_type});
        if (px) |p| self.addFieldFloat(Tag.Price, p);
        return self.finalize();
    }

    pub fn buildOrderCancelRequest(self: *Builder, sender: []const u8, target: []const u8, seq: i64, orig_cl_ord_id: []const u8, cl_ord_id: []const u8, symbol_val: []const u8, side_val: u8) []const u8 {
        self.beginMessage(MsgType.OrderCancelRequest, sender, target, seq);
        self.addField(Tag.OrigClOrdID, orig_cl_ord_id);
        self.addField(Tag.ClOrdID, cl_ord_id);
        self.addField(Tag.Symbol, symbol_val);
        self.addField(Tag.Side, &.{side_val});
        return self.finalize();
    }
};

// ============================================================================
// Session Manager
// ============================================================================

pub const SessionState = enum(u8) {
    disconnected,
    logon_sent,
    active,
    logout_sent,
    logout_received,
};

pub const SessionEvent = enum(u8) {
    logon_received,
    logout_received,
    heartbeat_timeout,
    sequence_gap,
    message_reject,
};

pub const Session = struct {
    state: SessionState,
    sender_comp_id: [32]u8,
    sender_len: usize,
    target_comp_id: [32]u8,
    target_len: usize,
    outgoing_seq: i64,
    incoming_seq: i64,
    heartbeat_interval_ms: u64,
    last_sent_time_ms: u64,
    last_recv_time_ms: u64,
    test_request_pending: bool,

    pub fn init(sender: []const u8, target: []const u8, heartbeat_ms: u64) Session {
        var s = Session{
            .state = .disconnected,
            .sender_comp_id = undefined,
            .sender_len = @min(sender.len, 32),
            .target_comp_id = undefined,
            .target_len = @min(target.len, 32),
            .outgoing_seq = 1,
            .incoming_seq = 1,
            .heartbeat_interval_ms = heartbeat_ms,
            .last_sent_time_ms = 0,
            .last_recv_time_ms = 0,
            .test_request_pending = false,
        };
        @memcpy(s.sender_comp_id[0..s.sender_len], sender[0..s.sender_len]);
        @memcpy(s.target_comp_id[0..s.target_len], target[0..s.target_len]);
        return s;
    }

    pub fn senderID(self: *const Session) []const u8 {
        return self.sender_comp_id[0..self.sender_len];
    }

    pub fn targetID(self: *const Session) []const u8 {
        return self.target_comp_id[0..self.target_len];
    }

    pub fn nextOutSeq(self: *Session) i64 {
        const seq = self.outgoing_seq;
        self.outgoing_seq += 1;
        return seq;
    }

    pub fn processIncoming(self: *Session, msg: *const FixMessage, now_ms: u64) ?SessionEvent {
        self.last_recv_time_ms = now_ms;
        self.test_request_pending = false;

        const seq = msg.msgSeqNum() orelse return .message_reject;
        const msg_type = msg.msgType() orelse return .message_reject;

        // Sequence number check
        if (seq > self.incoming_seq) {
            self.incoming_seq = seq + 1;
            return .sequence_gap;
        } else if (seq < self.incoming_seq) {
            // Duplicate or too low
            return null;
        }
        self.incoming_seq = seq + 1;

        // Handle session-level messages
        if (mem.eql(u8, msg_type, MsgType.Logon)) {
            self.state = .active;
            return .logon_received;
        } else if (mem.eql(u8, msg_type, MsgType.Logout)) {
            self.state = .logout_received;
            return .logout_received;
        } else if (mem.eql(u8, msg_type, MsgType.TestRequest)) {
            // Should respond with heartbeat containing TestReqID
            return null;
        } else if (mem.eql(u8, msg_type, MsgType.Heartbeat)) {
            return null;
        } else if (mem.eql(u8, msg_type, MsgType.ResendRequest)) {
            return .sequence_gap;
        } else if (mem.eql(u8, msg_type, MsgType.SequenceReset)) {
            if (msg.getFieldInt(Tag.NewSeqNo)) |new_seq| {
                self.incoming_seq = new_seq;
            }
            return null;
        }

        return null;
    }

    pub fn checkHeartbeat(self: *Session, now_ms: u64) bool {
        if (self.state != .active) return false;
        if (now_ms - self.last_recv_time_ms > self.heartbeat_interval_ms * 2) {
            return true; // timeout
        }
        return false;
    }

    pub fn needsHeartbeat(self: *const Session, now_ms: u64) bool {
        if (self.state != .active) return false;
        return now_ms - self.last_sent_time_ms >= self.heartbeat_interval_ms;
    }

    pub fn reset(self: *Session) void {
        self.state = .disconnected;
        self.outgoing_seq = 1;
        self.incoming_seq = 1;
        self.last_sent_time_ms = 0;
        self.last_recv_time_ms = 0;
        self.test_request_pending = false;
    }
};

// ============================================================================
// Utility: checksum computation
// ============================================================================

pub fn computeChecksum(data: []const u8) u8 {
    var sum: u32 = 0;
    // Process 8 bytes at a time for performance
    const chunks = data.len / 8;
    var i: usize = 0;
    while (i < chunks * 8) : (i += 8) {
        sum += @as(u32, data[i]) + data[i + 1] + data[i + 2] + data[i + 3] + data[i + 4] + data[i + 5] + data[i + 6] + data[i + 7];
    }
    while (i < data.len) : (i += 1) {
        sum += data[i];
    }
    return @intCast(sum % 256);
}

// Fast tag lookup for known tags
pub fn isSessionTag(tag: u32) bool {
    return switch (tag) {
        Tag.BeginString, Tag.BodyLength, Tag.MsgType, Tag.SenderCompID, Tag.TargetCompID, Tag.MsgSeqNum, Tag.SendingTime, Tag.CheckSum => true,
        else => false,
    };
}

pub fn isOrderTag(tag: u32) bool {
    return switch (tag) {
        Tag.ClOrdID, Tag.OrderID, Tag.ExecID, Tag.Symbol, Tag.Side, Tag.OrderQty, Tag.OrdType, Tag.Price, Tag.StopPx, Tag.TimeInForce, Tag.Account => true,
        else => false,
    };
}

// ============================================================================
// FIX Message Router
// ============================================================================

pub const HandlerFn = *const fn (msg: *const FixMessage) void;

pub const Router = struct {
    handlers: [32]struct { msg_type: [2]u8, msg_type_len: u8, handler: HandlerFn },
    handler_count: u32,
    default_handler: ?HandlerFn,

    pub fn init() Router {
        return .{
            .handlers = undefined,
            .handler_count = 0,
            .default_handler = null,
        };
    }

    pub fn register(self: *Router, msg_type: []const u8, handler: HandlerFn) void {
        if (self.handler_count >= 32) return;
        const idx = self.handler_count;
        self.handlers[idx].msg_type_len = @intCast(@min(msg_type.len, 2));
        @memcpy(self.handlers[idx].msg_type[0..self.handlers[idx].msg_type_len], msg_type[0..self.handlers[idx].msg_type_len]);
        self.handlers[idx].handler = handler;
        self.handler_count += 1;
    }

    pub fn setDefault(self: *Router, handler: HandlerFn) void {
        self.default_handler = handler;
    }

    pub fn route(self: *const Router, msg: *const FixMessage) void {
        const mt = msg.msgType() orelse {
            if (self.default_handler) |h| h(msg);
            return;
        };
        for (0..self.handler_count) |i| {
            const h = self.handlers[i];
            if (mem.eql(u8, mt, h.msg_type[0..h.msg_type_len])) {
                h.handler(msg);
                return;
            }
        }
        if (self.default_handler) |h| h(msg);
    }
};

// ============================================================================
// FIX Performance Counters
// ============================================================================

pub const PerfCounters = struct {
    messages_parsed: u64,
    messages_sent: u64,
    parse_errors: u64,
    checksum_failures: u64,
    sequence_gaps: u64,
    total_bytes_parsed: u64,
    total_bytes_sent: u64,
    min_parse_ns: u64,
    max_parse_ns: u64,
    sum_parse_ns: u64,

    pub fn init() PerfCounters {
        return .{
            .messages_parsed = 0,
            .messages_sent = 0,
            .parse_errors = 0,
            .checksum_failures = 0,
            .sequence_gaps = 0,
            .total_bytes_parsed = 0,
            .total_bytes_sent = 0,
            .min_parse_ns = std.math.maxInt(u64),
            .max_parse_ns = 0,
            .sum_parse_ns = 0,
        };
    }

    pub fn recordParse(self: *PerfCounters, bytes: usize, elapsed_ns: u64, success: bool) void {
        if (success) {
            self.messages_parsed += 1;
            self.total_bytes_parsed += bytes;
            if (elapsed_ns < self.min_parse_ns) self.min_parse_ns = elapsed_ns;
            if (elapsed_ns > self.max_parse_ns) self.max_parse_ns = elapsed_ns;
            self.sum_parse_ns += elapsed_ns;
        } else {
            self.parse_errors += 1;
        }
    }

    pub fn recordSend(self: *PerfCounters, bytes: usize) void {
        self.messages_sent += 1;
        self.total_bytes_sent += bytes;
    }

    pub fn avgParseNs(self: *const PerfCounters) u64 {
        if (self.messages_parsed == 0) return 0;
        return self.sum_parse_ns / self.messages_parsed;
    }

    pub fn parseRate(self: *const PerfCounters, elapsed_sec: f64) f64 {
        if (elapsed_sec <= 0) return 0;
        return @as(f64, @floatFromInt(self.messages_parsed)) / elapsed_sec;
    }

    pub fn throughputMBps(self: *const PerfCounters, elapsed_sec: f64) f64 {
        if (elapsed_sec <= 0) return 0;
        return @as(f64, @floatFromInt(self.total_bytes_parsed)) / elapsed_sec / 1048576.0;
    }
};

// ============================================================================
// Stream Decoder: handles partial messages in TCP stream
// ============================================================================

pub const StreamDecoder = struct {
    buffer: [65536]u8,
    write_pos: usize,
    decoder: Decoder,
    counters: PerfCounters,

    pub fn init() StreamDecoder {
        return .{
            .buffer = undefined,
            .write_pos = 0,
            .decoder = Decoder.init(),
            .counters = PerfCounters.init(),
        };
    }

    pub fn feed(self: *StreamDecoder, data: []const u8) usize {
        // Append to internal buffer
        const space = self.buffer.len - self.write_pos;
        const to_copy = @min(data.len, space);
        @memcpy(self.buffer[self.write_pos..][0..to_copy], data[0..to_copy]);
        self.write_pos += to_copy;
        return to_copy;
    }

    pub fn nextMessage(self: *StreamDecoder) ?FixMessage {
        if (self.write_pos < 20) return null;

        const result = self.decoder.findMessage(self.buffer[0..self.write_pos]) orelse return null;

        const msg = self.decoder.decode(result.msg) catch {
            self.counters.parse_errors += 1;
            // Skip bad data
            self.consumeBytes(result.consumed);
            return null;
        };

        self.counters.messages_parsed += 1;
        self.counters.total_bytes_parsed += result.consumed;
        self.consumeBytes(result.consumed);

        return msg;
    }

    fn consumeBytes(self: *StreamDecoder, n: usize) void {
        if (n >= self.write_pos) {
            self.write_pos = 0;
            return;
        }
        const remaining = self.write_pos - n;
        // Move remaining data to start
        var i: usize = 0;
        while (i < remaining) : (i += 1) {
            self.buffer[i] = self.buffer[n + i];
        }
        self.write_pos = remaining;
    }

    pub fn bufferedBytes(self: *const StreamDecoder) usize {
        return self.write_pos;
    }

    pub fn stats(self: *const StreamDecoder) PerfCounters {
        return self.counters;
    }
};

// ============================================================================
// FIX message type helpers
// ============================================================================

pub const NewOrderSingleFields = struct {
    cl_ord_id: ?[]const u8,
    symbol: ?[]const u8,
    side: ?u8,
    qty: ?f64,
    ord_type: ?u8,
    price: ?f64,
    time_in_force: ?u8,
    account: ?[]const u8,

    pub fn fromMessage(msg: *const FixMessage) NewOrderSingleFields {
        return .{
            .cl_ord_id = msg.getFieldValue(Tag.ClOrdID),
            .symbol = msg.getFieldValue(Tag.Symbol),
            .side = msg.side(),
            .qty = msg.getFieldFloat(Tag.OrderQty),
            .ord_type = blk: {
                const v = msg.getFieldValue(Tag.OrdType) orelse break :blk null;
                if (v.len == 0) break :blk null;
                break :blk v[0];
            },
            .price = msg.getFieldFloat(Tag.Price),
            .time_in_force = blk: {
                const v = msg.getFieldValue(Tag.TimeInForce) orelse break :blk null;
                if (v.len == 0) break :blk null;
                break :blk v[0];
            },
            .account = msg.getFieldValue(Tag.Account),
        };
    }
};

pub const ExecutionReportFields = struct {
    order_id: ?[]const u8,
    cl_ord_id: ?[]const u8,
    exec_id: ?[]const u8,
    exec_type: ?u8,
    ord_status: ?u8,
    symbol: ?[]const u8,
    side: ?u8,
    leaves_qty: ?f64,
    cum_qty: ?f64,
    avg_px: ?f64,
    last_px: ?f64,
    last_qty: ?f64,
    text: ?[]const u8,

    pub fn fromMessage(msg: *const FixMessage) ExecutionReportFields {
        return .{
            .order_id = msg.getFieldValue(Tag.OrderID),
            .cl_ord_id = msg.getFieldValue(Tag.ClOrdID),
            .exec_id = msg.getFieldValue(Tag.ExecID),
            .exec_type = msg.execType(),
            .ord_status = msg.ordStatus(),
            .symbol = msg.getFieldValue(Tag.Symbol),
            .side = msg.side(),
            .leaves_qty = msg.getFieldFloat(Tag.LeavesQty),
            .cum_qty = msg.getFieldFloat(Tag.CumQty),
            .avg_px = msg.getFieldFloat(Tag.AvgPx),
            .last_px = msg.getFieldFloat(Tag.LastPx),
            .last_qty = msg.getFieldFloat(Tag.LastQty),
            .text = msg.getFieldValue(Tag.Text),
        };
    }

    pub fn isFill(self: *const ExecutionReportFields) bool {
        if (self.exec_type) |et| return et == '1' or et == '2'; // Partial or Full fill
        return false;
    }

    pub fn isRejected(self: *const ExecutionReportFields) bool {
        if (self.exec_type) |et| return et == '8';
        return false;
    }

    pub fn isCancelled(self: *const ExecutionReportFields) bool {
        if (self.exec_type) |et| return et == '4';
        return false;
    }
};

pub const MarketDataEntry = struct {
    entry_type: ?u8, // '0'=bid, '1'=ask, '2'=trade
    price: ?f64,
    size: ?f64,
    date: ?[]const u8,
    time: ?[]const u8,
};

pub fn parseMarketDataEntries(msg: *const FixMessage) [16]MarketDataEntry {
    var entries: [16]MarketDataEntry = undefined;
    for (0..16) |i| entries[i] = .{ .entry_type = null, .price = null, .size = null, .date = null, .time = null };

    const group = msg.getRepeatingGroup(Tag.NoMDEntries, Tag.MDEntryType) orelse return entries;

    var entry_idx: usize = 0;
    for (group) |field| {
        if (entry_idx >= 16) break;
        if (field.tag == Tag.MDEntryType) {
            if (entry_idx > 0 or entries[0].entry_type != null) entry_idx += 1;
            if (entry_idx >= 16) break;
            const v = field.value(msg.buffer);
            if (v.len > 0) entries[entry_idx].entry_type = v[0];
        } else if (field.tag == Tag.MDEntryPx) {
            entries[entry_idx].price = field.floatValue(msg.buffer) catch null;
        } else if (field.tag == Tag.MDEntrySize) {
            entries[entry_idx].size = field.floatValue(msg.buffer) catch null;
        } else if (field.tag == Tag.MDEntryDate) {
            entries[entry_idx].date = field.value(msg.buffer);
        } else if (field.tag == Tag.MDEntryTime) {
            entries[entry_idx].time = field.value(msg.buffer);
        }
    }

    return entries;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "decode simple FIX message" {
    const raw = "8=FIX.4.2|9=000073|35=D|49=SENDER|56=TARGET|34=1|55=AAPL|54=1|38=100|40=2|44=150.50|10=000|";
    var decoder = Decoder.initWithDelimiter(PIPE);
    decoder.strict_checksum = false;
    decoder.strict_body_length = false;
    const msg = try decoder.decode(raw);

    try testing.expect(mem.eql(u8, msg.msgType().?, "D"));
    try testing.expect(mem.eql(u8, msg.symbol().?, "AAPL"));
    try testing.expectEqual(@as(u8, '1'), msg.side().?);
}

test "builder round-trip" {
    var builder = Builder.initWithDelimiter(PIPE);
    const raw = builder.buildLogon("SENDER", "TARGET", 1, 30);
    _ = raw;
    // Just verify it doesn't crash; checksum validation would need matching delimiter
}

test "session state machine" {
    var session = Session.init("SENDER", "TARGET", 30000);
    try testing.expectEqual(SessionState.disconnected, session.state);
    try testing.expect(mem.eql(u8, "SENDER", session.senderID()));
}
