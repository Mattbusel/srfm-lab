// fix_engine.cpp — FIX 4.2 session layer implementation.
// Chronos / AETERNUS — production C++ FIX engine.

#include "fix_engine.hpp"
#include <cstring>
#include <cassert>
#include <cstdio>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <ctime>

namespace chronos {
namespace fix {

// ── FixMessage implementation ────────────────────────────────────────────────

std::string FixMessage::to_wire(char sep) const {
    // Build ordered field list (tag 8,9 first, then others, then 10 last)
    std::vector<std::pair<uint32_t, std::string>> ordered;
    ordered.reserve(fields_.size());

    for (const auto& [tag, val] : fields_) {
        if (tag == tags::BeginString || tag == tags::BodyLength || tag == tags::CheckSum)
            continue;
        ordered.emplace_back(tag, val);
    }
    // Sort by tag except keep MsgType, SenderCompID, etc. in proper order
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Build body
    std::string body;
    body.reserve(256);
    for (const auto& [tag, val] : ordered) {
        body += std::to_string(tag);
        body += '=';
        body += val;
        body += sep;
    }

    // Prepend header and append checksum
    std::string begin_string = fields_.count(tags::BeginString) ? fields_.at(tags::BeginString) : "FIX.4.2";
    std::string header = std::to_string(tags::BeginString) + "=" + begin_string + sep;
    std::string body_len_str = std::to_string(body.size());
    std::string body_len_field = std::to_string(tags::BodyLength) + "=" + body_len_str + sep;

    std::string wire = header + body_len_field + body;
    uint8_t chk = compute_checksum(wire);
    char buf[32];
    snprintf(buf, sizeof(buf), "%03u", chk);
    wire += std::to_string(tags::CheckSum) + "=" + buf + sep;
    return wire;
}

uint8_t FixMessage::compute_checksum(const std::string& wire) {
    uint32_t sum = 0;
    // Sum all bytes up to but not including tag 10=
    const char* p = wire.c_str();
    const char* end = wire.c_str() + wire.size();
    // Find last occurrence of "10="
    const char* chk_start = nullptr;
    for (const char* q = end - 4; q >= p; --q) {
        if (q[0] == '\x01' && q[1] == '1' && q[2] == '0' && q[3] == '=') {
            chk_start = q + 1;
            break;
        }
    }
    size_t body_len = chk_start ? (size_t)(chk_start - p) : wire.size();
    for (size_t i = 0; i < body_len; ++i) sum += static_cast<uint8_t>(p[i]);
    return static_cast<uint8_t>(sum % 256);
}

bool FixMessage::verify_checksum() const {
    auto it = fields_.find(tags::CheckSum);
    if (it == fields_.end()) return false;
    try {
        uint8_t expected = static_cast<uint8_t>(std::stoul(it->second));
        std::string wire = to_wire('\x01');
        // Find the checksum field position in wire
        auto pos = wire.rfind("\x0110=");
        if (pos == std::string::npos) return false;
        std::string prefix = wire.substr(0, pos + 1);
        uint8_t computed = compute_checksum(prefix);
        return computed == expected;
    } catch (...) {
        return false;
    }
}

// ── FixParser implementation ──────────────────────────────────────────────────

size_t FixParser::find_message_end(const uint8_t* data, size_t len) const {
    // Scan for "10=NNN\x01" pattern
    const char sep = separator_;
    for (size_t i = 0; i < len; ++i) {
        if (data[i] == '1' && i + 2 < len && data[i+1] == '0' && data[i+2] == '=') {
            size_t j = i + 3;
            while (j < len && data[j] != static_cast<uint8_t>(sep)) ++j;
            if (j < len) return j + 1;
        }
    }
    return len; // not found
}

std::pair<FixMessage, size_t> FixParser::parse(const uint8_t* data, size_t len) {
    size_t end = find_message_end(data, len);
    if (end == 0 || end > len) {
        throw FixParseError("Cannot find message end");
    }

    FixMessage msg;
    size_t i = 0;
    while (i < end) {
        // Find '='
        size_t eq = i;
        while (eq < end && data[eq] != '=') ++eq;
        if (eq >= end) break;

        // Parse tag
        std::string tag_str(reinterpret_cast<const char*>(data + i), eq - i);
        uint32_t tag = 0;
        try { tag = static_cast<uint32_t>(std::stoul(tag_str)); }
        catch (...) { ++i; continue; }

        // Find separator
        size_t val_start = eq + 1;
        size_t val_end = val_start;
        while (val_end < end && data[val_end] != static_cast<uint8_t>(separator_)) ++val_end;

        std::string value(reinterpret_cast<const char*>(data + val_start), val_end - val_start);
        msg.set(tag, value);
        i = val_end + 1;
    }

    if (strict_checksum_ && !msg.verify_checksum()) {
        ++stats_.checksum_errors;
        ++stats_.parse_errors;
        throw FixParseError("Checksum mismatch");
    }

    ++stats_.messages_parsed;
    return {std::move(msg), end};
}

std::vector<FixMessage> FixParser::parse_all(const uint8_t* data, size_t len) {
    std::vector<FixMessage> messages;
    size_t offset = 0;
    while (offset < len) {
        try {
            auto [msg, consumed] = parse(data + offset, len - offset);
            messages.push_back(std::move(msg));
            offset += consumed;
        } catch (const FixParseError&) {
            ++stats_.parse_errors;
            // Try to skip to next message
            ++offset;
            while (offset < len && data[offset] != '8') ++offset; // find next BeginString
        }
    }
    return messages;
}

FixMessage FixParser::parse_string(const std::string& raw) {
    auto [msg, consumed] = parse(
        reinterpret_cast<const uint8_t*>(raw.c_str()), raw.size());
    return msg;
}

std::string FixParser::tag_name(uint32_t tag) {
    static const std::unordered_map<uint32_t, std::string> names = {
        {8, "BeginString"}, {9, "BodyLength"}, {10, "CheckSum"},
        {11, "ClOrdID"}, {14, "CumQty"}, {17, "ExecID"}, {20, "ExecTransType"},
        {22, "IDSource"}, {31, "LastPx"}, {32, "LastShares"}, {34, "MsgSeqNum"},
        {35, "MsgType"}, {36, "NewSeqNo"}, {37, "OrderID"}, {38, "OrderQty"},
        {39, "OrdStatus"}, {40, "OrdType"}, {41, "OrigClOrdID"}, {43, "PossDupFlag"},
        {44, "Price"}, {45, "RefSeqNum"}, {48, "SecurityID"}, {49, "SenderCompID"},
        {50, "SenderSubID"}, {52, "SendingTime"}, {54, "Side"}, {55, "Symbol"},
        {56, "TargetCompID"}, {57, "TargetSubID"}, {58, "Text"}, {59, "TimeInForce"},
        {60, "TransactTime"}, {98, "EncryptMethod"}, {99, "StopPx"},
        {102, "CxlRejReason"}, {103, "OrdRejReason"}, {108, "HeartBtInt"},
        {110, "MinQty"}, {111, "MaxFloor"}, {112, "TestReqID"}, {115, "OnBehalfOfCompID"},
        {122, "OrigSendingTime"}, {123, "GapFillFlag"}, {128, "DeliverToCompID"},
        {141, "ResetSeqNumFlag"}, {148, "Headline"}, {150, "ExecType"},
        {151, "LeavesQty"}, {167, "SecurityType"}, {200, "MaturityMonthYear"},
        {207, "SecurityExchange"}, {262, "MDReqID"}, {263, "SubscriptionRequestType"},
        {264, "MarketDepth"}, {268, "NoMDEntries"}, {269, "MDEntryType"},
        {270, "MDEntryPx"}, {271, "MDEntrySize"}, {279, "MDUpdateAction"},
        {372, "RefMsgType"}, {373, "SessionRejectReason"},
    };
    auto it = names.find(tag);
    return (it != names.end()) ? it->second : "Tag" + std::to_string(tag);
}

// ── FixMessageBuilder implementation ─────────────────────────────────────────

static std::string utc_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    struct tm* gmt = gmtime(&tt);
    char buf[32];
    snprintf(buf, sizeof(buf), "%04d%02d%02d-%02d:%02d:%02d.%03d",
        gmt->tm_year + 1900, gmt->tm_mon + 1, gmt->tm_mday,
        gmt->tm_hour, gmt->tm_min, gmt->tm_sec, (int)ms.count());
    return buf;
}

FixMessage FixMessageBuilder::new_order_single(
    const std::string& cl_ord_id,
    const std::string& symbol,
    char side,
    double qty,
    char ord_type,
    double price)
{
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::NewOrderSingle);
    msg.set(tags::ClOrdID, cl_ord_id);
    msg.set(tags::Symbol, symbol);
    msg.set(tags::Side, std::string(1, side));
    msg.set(tags::OrderQty, qty, 0);
    msg.set(tags::OrdType, std::string(1, ord_type));
    if (price > 0.0) msg.set(tags::Price, price, 4);
    msg.set(tags::TransactTime, utc_timestamp());
    msg.set(tags::TimeInForce, "0"); // Day
    return msg;
}

FixMessage FixMessageBuilder::execution_report(
    const std::string& cl_ord_id,
    const std::string& order_id,
    const std::string& exec_id,
    char exec_type,
    char ord_status,
    const std::string& symbol,
    char side,
    double cum_qty,
    double leaves_qty,
    double avg_px,
    double last_shares,
    double last_px)
{
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::ExecutionReport);
    msg.set(tags::ClOrdID, cl_ord_id);
    msg.set(tags::OrderID, order_id);
    msg.set(tags::ExecID, exec_id);
    msg.set(tags::ExecType, std::string(1, exec_type));
    msg.set(tags::OrdStatus, std::string(1, ord_status));
    msg.set(tags::Symbol, symbol);
    msg.set(tags::Side, std::string(1, side));
    msg.set(tags::CumQty, cum_qty, 0);
    msg.set(tags::LeavesQty, leaves_qty, 0);
    msg.set(tags::AvgPx, avg_px, 4);
    if (last_shares > 0.0) msg.set(tags::LastShares, last_shares, 0);
    if (last_px > 0.0) msg.set(tags::LastPx, last_px, 4);
    msg.set(tags::TransactTime, utc_timestamp());
    return msg;
}

FixMessage FixMessageBuilder::cancel_request(
    const std::string& cl_ord_id,
    const std::string& orig_cl_ord_id,
    const std::string& symbol,
    char side,
    double order_qty)
{
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::OrderCancelRequest);
    msg.set(tags::ClOrdID, cl_ord_id);
    msg.set(tags::OrigClOrdID, orig_cl_ord_id);
    msg.set(tags::Symbol, symbol);
    msg.set(tags::Side, std::string(1, side));
    msg.set(tags::OrderQty, order_qty, 0);
    msg.set(tags::TransactTime, utc_timestamp());
    return msg;
}

FixMessage FixMessageBuilder::cancel_replace_request(
    const std::string& cl_ord_id,
    const std::string& orig_cl_ord_id,
    const std::string& symbol,
    char side,
    double qty,
    double price)
{
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::OrderCancelReplaceRequest);
    msg.set(tags::ClOrdID, cl_ord_id);
    msg.set(tags::OrigClOrdID, orig_cl_ord_id);
    msg.set(tags::Symbol, symbol);
    msg.set(tags::Side, std::string(1, side));
    msg.set(tags::OrderQty, qty, 0);
    msg.set(tags::Price, price, 4);
    msg.set(tags::OrdType, "2"); // Limit
    msg.set(tags::TransactTime, utc_timestamp());
    return msg;
}

FixMessage FixMessageBuilder::logon(int heartbeat_interval, const std::string& reset_flag) {
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::Logon);
    msg.set(tags::EncryptMethod, "0");
    msg.set(tags::HeartBtInt, static_cast<int64_t>(heartbeat_interval));
    if (!reset_flag.empty()) msg.set(tags::ResetSeqNumFlag, reset_flag);
    return msg;
}

FixMessage FixMessageBuilder::logout(const std::string& text) {
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::Logout);
    if (!text.empty()) msg.set(tags::Text, text);
    return msg;
}

FixMessage FixMessageBuilder::heartbeat(const std::string& test_req_id) {
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::Heartbeat);
    if (!test_req_id.empty()) msg.set(tags::TestReqID, test_req_id);
    return msg;
}

FixMessage FixMessageBuilder::test_request(const std::string& test_req_id) {
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::TestRequest);
    msg.set(tags::TestReqID, test_req_id);
    return msg;
}

FixMessage FixMessageBuilder::resend_request(int64_t begin_seq_no, int64_t end_seq_no) {
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::ResendRequest);
    msg.set(tags::RefSeqNum, begin_seq_no);
    msg.set(268u, std::to_string(end_seq_no)); // EndSeqNo = 16
    return msg;
}

// ── FixSession implementation ─────────────────────────────────────────────────

FixSession::FixSession(SessionConfig config)
    : config_(std::move(config)), parser_(config_.validate_checksum) {}

void FixSession::on_data(const uint8_t* data, size_t len) {
    recv_buffer_.append(reinterpret_cast<const char*>(data), len);
    while (!recv_buffer_.empty()) {
        try {
            auto [msg, consumed] = parser_.parse(
                reinterpret_cast<const uint8_t*>(recv_buffer_.data()),
                recv_buffer_.size());
            recv_buffer_.erase(0, consumed);
            dispatch(msg);
        } catch (const FixParseError&) {
            // Not enough data yet or parse error - try to resync
            if (recv_buffer_.size() > static_cast<size_t>(config_.max_message_size)) {
                recv_buffer_.clear(); // drop to resync
            }
            break;
        }
    }
}

void FixSession::on_data(const std::string& raw) {
    on_data(reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
}

void FixSession::dispatch(const FixMessage& msg) {
    if (!validate_session_fields(msg)) return;

    const std::string& type = msg.msg_type();
    if (type == msg_types::Logon) handle_logon(msg);
    else if (type == msg_types::Logout) handle_logout(msg);
    else if (type == msg_types::Heartbeat) handle_heartbeat(msg);
    else if (type == msg_types::TestRequest) handle_test_request(msg);
    else if (type == msg_types::ResendRequest) handle_resend_request(msg);
    else if (type == msg_types::SequenceReset) handle_sequence_reset(msg);
    else if (type == msg_types::Reject) handle_reject(msg);

    if (msg_handler_) msg_handler_(msg);
}

bool FixSession::validate_session_fields(const FixMessage& msg) {
    // Basic validation: check seq num
    int64_t recv_seq = msg.seq_num();
    if (recv_seq > 0) {
        if (recv_seq < next_recv_seq_) return false; // duplicate
        // TODO: handle gap detection
        next_recv_seq_ = recv_seq + 1;
    }
    return true;
}

void FixSession::handle_logon(const FixMessage& msg) {
    transition(SessionState::Active);
}

void FixSession::handle_logout(const FixMessage& msg) {
    transition(SessionState::Disconnected);
}

void FixSession::handle_heartbeat(const FixMessage&) {}
void FixSession::handle_test_request(const FixMessage& msg) {}
void FixSession::handle_resend_request(const FixMessage& msg) {}
void FixSession::handle_sequence_reset(const FixMessage& msg) {
    auto new_seq = msg.get_int(tags::NewSeqNo);
    if (new_seq) next_recv_seq_ = *new_seq;
}
void FixSession::handle_reject(const FixMessage&) {}

void FixSession::transition(SessionState new_state) {
    if (new_state == state_) return;
    SessionState old = state_;
    state_ = new_state;
    if (state_handler_) state_handler_(old, new_state);
}

std::string FixSession::stamp_and_serialise(FixMessage msg) {
    msg.set(tags::BeginString, config_.fix_version);
    msg.set(tags::SenderCompID, config_.sender_comp_id);
    msg.set(tags::TargetCompID, config_.target_comp_id);
    msg.set(tags::MsgSeqNum, next_send_seq_++);
    msg.set(tags::SendingTime, current_utc_timestamp());
    return msg.to_wire('\x01');
}

std::string FixSession::generate_logon() const {
    FixMessage msg = FixMessageBuilder::logon(config_.heartbeat_interval,
        config_.reset_on_logon ? "Y" : "N");
    const_cast<FixSession*>(this)->stamp_and_serialise(msg);
    return msg.to_wire();
}

std::string FixSession::generate_logout(const std::string& text) const {
    return FixMessageBuilder::logout(text).to_wire();
}

std::string FixSession::generate_heartbeat(const std::string& test_req_id) const {
    return FixMessageBuilder::heartbeat(test_req_id).to_wire();
}

std::string FixSession::generate_test_request(const std::string& id) const {
    return FixMessageBuilder::test_request(id).to_wire();
}

std::string FixSession::generate_resend_request(int64_t begin, int64_t end) const {
    return FixMessageBuilder::resend_request(begin, end).to_wire();
}

std::string FixSession::generate_sequence_reset(int64_t new_seq, bool gap_fill) const {
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::SequenceReset);
    msg.set(tags::GapFillFlag, gap_fill ? "Y" : "N");
    msg.set(tags::NewSeqNo, new_seq);
    return msg.to_wire();
}

std::string FixSession::generate_reject(int64_t ref_seq, int reason, const std::string& text) const {
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::Reject);
    msg.set(tags::RefSeqNum, ref_seq);
    msg.set(tags::SessionRejectReason, static_cast<int64_t>(reason));
    if (!text.empty()) msg.set(tags::Text, text);
    return msg.to_wire();
}

std::string FixSession::current_utc_timestamp() { return utc_timestamp(); }

// ── Market data parser ────────────────────────────────────────────────────────

MarketDataSnapshot parse_market_data_snapshot(const FixMessage& msg) {
    MarketDataSnapshot snap;
    snap.symbol = msg.get(tags::Symbol);
    snap.request_id = msg.get(tags::MDReqID);

    auto n_entries = msg.get_int(tags::NoMDEntries).value_or(0);
    // Note: repeating groups would require more complex parsing;
    // this is a simplified single-group extraction
    if (n_entries > 0) {
        MarketDataEntry entry;
        const auto& et = msg.get(tags::MDEntryType);
        entry.entry_type = et.empty() ? '?' : et[0];
        entry.price = msg.get_double(tags::MDEntryPx).value_or(0.0);
        entry.size = msg.get_double(tags::MDEntrySize).value_or(0.0);
        entry.exchange = msg.get(tags::SecurityExchange);
        snap.entries.push_back(entry);
    }
    return snap;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#ifndef FIX_NO_TESTS

namespace tests {

void test_fix_message_build() {
    FixMessage msg = FixMessageBuilder::new_order_single("CLO001", "AAPL", '1', 100.0, '2', 150.0);
    assert(msg.msg_type() == msg_types::NewOrderSingle);
    assert(msg.is_buy());
    assert(msg.order_qty().has_value());
    assert(std::abs(*msg.order_qty() - 100.0) < 0.01);
    assert(msg.price().has_value());
    assert(std::abs(*msg.price() - 150.0) < 0.001);
    printf("test_fix_message_build: PASSED\n");
}

void test_fix_wire_serialise() {
    FixMessage msg;
    msg.set(tags::MsgType, msg_types::Heartbeat);
    msg.set(tags::SenderCompID, "SENDER");
    msg.set(tags::TargetCompID, "TARGET");
    msg.set(tags::MsgSeqNum, (int64_t)1);
    msg.set(tags::SendingTime, "20240101-12:00:00");

    std::string wire = msg.to_wire('\x01');
    assert(!wire.empty());
    assert(wire.find("35=0") != std::string::npos);
    assert(wire.find("49=SENDER") != std::string::npos);
    printf("test_fix_wire_serialise: PASSED\n");
}

void test_fix_parser_roundtrip() {
    // Build a message
    FixMessage orig = FixMessageBuilder::new_order_single("C001", "MSFT", '2', 500.0, '2', 200.0);
    orig.set(tags::SenderCompID, "SENDER");
    orig.set(tags::TargetCompID, "TARGET");
    orig.set(tags::MsgSeqNum, (int64_t)5);
    orig.set(tags::SendingTime, "20240101-09:30:00");
    orig.set(tags::BeginString, "FIX.4.2");
    std::string wire = orig.to_wire('\x01');

    // Parse it back
    FixParser parser;
    FixMessage parsed = parser.parse_string(wire);
    assert(parsed.msg_type() == msg_types::NewOrderSingle);
    assert(parsed.get(tags::SenderCompID) == "SENDER");
    assert(parsed.get(tags::Symbol) == "MSFT");
    printf("test_fix_parser_roundtrip: PASSED\n");
}

void test_fix_session_logon() {
    SessionConfig config;
    config.sender_comp_id = "TESTER";
    config.target_comp_id = "EXCHANGE";
    config.heartbeat_interval = 30;

    FixSession session(config);
    assert(session.state() == SessionState::Disconnected);

    // Build a logon message
    FixMessage logon_msg = FixMessageBuilder::logon(30, "Y");
    logon_msg.set(tags::BeginString, "FIX.4.2");
    logon_msg.set(tags::SenderCompID, "EXCHANGE");
    logon_msg.set(tags::TargetCompID, "TESTER");
    logon_msg.set(tags::MsgSeqNum, (int64_t)1);
    logon_msg.set(tags::SendingTime, "20240101-09:30:00");
    std::string wire = logon_msg.to_wire('\x01');

    session.on_data(wire);
    assert(session.state() == SessionState::Active);
    printf("test_fix_session_logon: PASSED\n");
}

void test_execution_report_build() {
    FixMessage er = FixMessageBuilder::execution_report(
        "CLO001", "ORD001", "EXEC001", '2', '2',
        "NVDA", '1', 100.0, 0.0, 500.0, 100.0, 500.0);
    assert(er.msg_type() == msg_types::ExecutionReport);
    assert(er.get(tags::ExecType) == "2");
    assert(er.get(tags::OrdStatus) == "2");
    printf("test_execution_report_build: PASSED\n");
}

void test_cancel_request_build() {
    FixMessage cr = FixMessageBuilder::cancel_request("NEW001", "CLO001", "TSLA", '1', 50.0);
    assert(cr.msg_type() == msg_types::OrderCancelRequest);
    assert(cr.get(tags::OrigClOrdID) == "CLO001");
    printf("test_cancel_request_build: PASSED\n");
}

void test_cancel_replace_request_build() {
    FixMessage crr = FixMessageBuilder::cancel_replace_request("NEW001", "CLO001", "GOOGL", '2', 200.0, 180.0);
    assert(crr.msg_type() == msg_types::OrderCancelReplaceRequest);
    assert(std::abs(crr.get_double(tags::Price).value_or(0.0) - 180.0) < 0.001);
    printf("test_cancel_replace_request_build: PASSED\n");
}

void test_message_store() {
    FixMessageStore store(5);
    store.store(1, "msg1");
    store.store(2, "msg2");
    store.store(3, "msg3");

    assert(store.retrieve(1).has_value());
    assert(*store.retrieve(1) == "msg1");
    assert(!store.retrieve(99).has_value());

    auto range = store.range(1, 2);
    assert(range.size() == 2);
    printf("test_message_store: PASSED\n");
}

void run_all() {
    test_fix_message_build();
    test_fix_wire_serialise();
    test_fix_parser_roundtrip();
    test_fix_session_logon();
    test_execution_report_build();
    test_cancel_request_build();
    test_cancel_replace_request_build();
    test_message_store();
    printf("All FIX engine tests PASSED\n");
}

} // namespace tests

#endif // FIX_NO_TESTS

} // namespace fix
} // namespace chronos

#ifndef FIX_NO_MAIN
int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--test") {
        chronos::fix::tests::run_all();
        return 0;
    }
    printf("FIX Engine v1.0 — Chronos/AETERNUS\n");
    printf("Usage: fix_engine --test\n");
    return 0;
}
#endif
