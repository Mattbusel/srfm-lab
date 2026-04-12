#pragma once
// fix_engine.hpp — FIX 4.2 session layer, message parsing, FIX tag dictionary.
// Chronos / AETERNUS — production FIX protocol engine.

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <functional>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <chrono>

namespace chronos {
namespace fix {

// ── FIX tag dictionary ────────────────────────────────────────────────────────

namespace tags {
    static constexpr uint32_t BeginString        = 8;
    static constexpr uint32_t BodyLength         = 9;
    static constexpr uint32_t MsgType            = 35;
    static constexpr uint32_t SenderCompID       = 49;
    static constexpr uint32_t TargetCompID       = 56;
    static constexpr uint32_t MsgSeqNum          = 34;
    static constexpr uint32_t SendingTime        = 52;
    static constexpr uint32_t CheckSum           = 10;
    static constexpr uint32_t OnBehalfOfCompID   = 115;
    static constexpr uint32_t DeliverToCompID    = 128;
    static constexpr uint32_t PossDupFlag        = 43;
    static constexpr uint32_t PossResend         = 97;
    static constexpr uint32_t OrigSendingTime    = 122;
    static constexpr uint32_t EncryptMethod      = 98;
    static constexpr uint32_t HeartBtInt         = 108;
    static constexpr uint32_t TestReqID          = 112;
    static constexpr uint32_t RefSeqNum          = 45;
    static constexpr uint32_t RefMsgType         = 372;
    static constexpr uint32_t SessionRejectReason = 373;
    static constexpr uint32_t GapFillFlag        = 123;
    static constexpr uint32_t NewSeqNo           = 36;
    static constexpr uint32_t Symbol             = 55;
    static constexpr uint32_t SecurityID         = 48;
    static constexpr uint32_t IDSource           = 22;
    static constexpr uint32_t SecurityType       = 167;
    static constexpr uint32_t MaturityMonthYear  = 200;
    static constexpr uint32_t MaturityDay        = 205;
    static constexpr uint32_t Side               = 54;
    static constexpr uint32_t OrderQty           = 38;
    static constexpr uint32_t Price              = 44;
    static constexpr uint32_t StopPx             = 99;
    static constexpr uint32_t OrdType            = 40;
    static constexpr uint32_t TimeInForce        = 59;
    static constexpr uint32_t ClOrdID            = 11;
    static constexpr uint32_t OrderID            = 37;
    static constexpr uint32_t ExecID             = 17;
    static constexpr uint32_t ExecTransType      = 20;
    static constexpr uint32_t ExecType           = 150;
    static constexpr uint32_t OrdStatus          = 39;
    static constexpr uint32_t CumQty             = 14;
    static constexpr uint32_t LeavesQty          = 151;
    static constexpr uint32_t AvgPx              = 6;
    static constexpr uint32_t LastShares         = 32;
    static constexpr uint32_t LastPx             = 31;
    static constexpr uint32_t CxlRejReason       = 102;
    static constexpr uint32_t OrdRejReason       = 103;
    static constexpr uint32_t Account            = 1;
    static constexpr uint32_t Currency           = 15;
    static constexpr uint32_t TransactTime       = 60;
    static constexpr uint32_t OrigClOrdID        = 41;
    static constexpr uint32_t MDReqID            = 262;
    static constexpr uint32_t SubscriptionRequestType = 263;
    static constexpr uint32_t MarketDepth        = 264;
    static constexpr uint32_t MDUpdateType       = 265;
    static constexpr uint32_t NoMDEntries        = 268;
    static constexpr uint32_t MDEntryType        = 269;
    static constexpr uint32_t MDEntryPx          = 270;
    static constexpr uint32_t MDEntrySize        = 271;
    static constexpr uint32_t MDEntryDate        = 272;
    static constexpr uint32_t MDEntryTime        = 273;
    static constexpr uint32_t MDUpdateAction     = 279;
    static constexpr uint32_t SecurityExchange   = 207;
    static constexpr uint32_t Text               = 58;
    static constexpr uint32_t ExecBroker         = 76;
    static constexpr uint32_t MinQty             = 110;
    static constexpr uint32_t MaxFloor           = 111;
    static constexpr uint32_t LocateReqd         = 114;
    static constexpr uint32_t PegOffsetValue     = 211;
    static constexpr uint32_t DiscretionInst     = 388;
    static constexpr uint32_t DiscretionOffsetValue = 389;
    static constexpr uint32_t NoAllocs           = 78;
    static constexpr uint32_t AllocAccount       = 79;
    static constexpr uint32_t AllocShares        = 80;
    static constexpr uint32_t ProcessCode        = 81;
    static constexpr uint32_t Headline           = 148;
    static constexpr uint32_t LinesOfText        = 33;
    static constexpr uint32_t RawDataLength      = 95;
    static constexpr uint32_t RawData            = 96;
    static constexpr uint32_t ResetSeqNumFlag    = 141;
    static constexpr uint32_t SenderSubID        = 50;
    static constexpr uint32_t TargetSubID        = 57;
    static constexpr uint32_t SenderLocationID   = 142;
    static constexpr uint32_t TargetLocationID   = 143;
    static constexpr uint32_t SessionStatus      = 1409;
    static constexpr uint32_t DefaultApplVerID   = 1137;
}

// ── MsgType values ────────────────────────────────────────────────────────────

namespace msg_types {
    static const std::string Heartbeat                   = "0";
    static const std::string TestRequest                 = "1";
    static const std::string ResendRequest               = "2";
    static const std::string Reject                      = "3";
    static const std::string SequenceReset               = "4";
    static const std::string Logout                      = "5";
    static const std::string IOI                         = "6";
    static const std::string Logon                       = "A";
    static const std::string News                        = "B";
    static const std::string Email                       = "C";
    static const std::string NewOrderSingle             = "D";
    static const std::string NewOrderList               = "E";
    static const std::string OrderCancelRequest         = "F";
    static const std::string OrderCancelReplaceRequest  = "G";
    static const std::string OrderStatusRequest         = "H";
    static const std::string ExecutionReport            = "8";
    static const std::string OrderCancelReject          = "9";
    static const std::string MarketDataRequest          = "V";
    static const std::string MarketDataSnapshot         = "W";
    static const std::string MarketDataIncrementalRefresh = "X";
    static const std::string SecurityDefinitionRequest  = "c";
    static const std::string SecurityDefinition         = "d";
    static const std::string TradingSessionStatusRequest = "g";
    static const std::string TradingSessionStatus       = "h";
    static const std::string MassQuote                  = "i";
    static const std::string BusinessMessageReject      = "j";
}

// ── Error types ───────────────────────────────────────────────────────────────

class FixParseError : public std::runtime_error {
public:
    explicit FixParseError(const std::string& msg) : std::runtime_error("FIX: " + msg) {}
};

class FixSessionError : public std::runtime_error {
public:
    explicit FixSessionError(const std::string& msg) : std::runtime_error("FIXSession: " + msg) {}
};

// ── FIX message ───────────────────────────────────────────────────────────────

class FixMessage {
public:
    FixMessage() = default;

    void set(uint32_t tag, const std::string& value) {
        fields_[tag] = value;
    }

    void set(uint32_t tag, int64_t value) {
        fields_[tag] = std::to_string(value);
    }

    void set(uint32_t tag, double value, int precision = 6) {
        std::ostringstream oss;
        oss << std::fixed;
        oss.precision(precision);
        oss << value;
        fields_[tag] = oss.str();
    }

    bool has(uint32_t tag) const { return fields_.count(tag) > 0; }

    const std::string& get(uint32_t tag) const {
        static const std::string empty;
        auto it = fields_.find(tag);
        return (it != fields_.end()) ? it->second : empty;
    }

    std::optional<double> get_double(uint32_t tag) const {
        auto it = fields_.find(tag);
        if (it == fields_.end()) return {};
        try { return std::stod(it->second); } catch (...) { return {}; }
    }

    std::optional<int64_t> get_int(uint32_t tag) const {
        auto it = fields_.find(tag);
        if (it == fields_.end()) return {};
        try { return std::stoll(it->second); } catch (...) { return {}; }
    }

    std::string msg_type() const { return get(tags::MsgType); }
    std::string sender() const { return get(tags::SenderCompID); }
    std::string target() const { return get(tags::TargetCompID); }
    int64_t seq_num() const { return get_int(tags::MsgSeqNum).value_or(0); }
    std::string sending_time() const { return get(tags::SendingTime); }

    // Access all fields (ordered by tag)
    const std::unordered_map<uint32_t, std::string>& fields() const { return fields_; }

    // Serialise to wire format (SOH-delimited)
    std::string to_wire(char sep = '\x01') const;

    // Helper accessors
    std::optional<double> price() const { return get_double(tags::Price); }
    std::optional<double> order_qty() const { return get_double(tags::OrderQty); }
    char side() const { auto s = get(tags::Side); return s.empty() ? 0 : s[0]; }
    char ord_type() const { auto s = get(tags::OrdType); return s.empty() ? 0 : s[0]; }

    bool is_buy() const { return side() == '1'; }
    bool is_sell() const { return side() == '2'; }

    // Verify FIX checksum
    bool verify_checksum() const;

    // Compute checksum over wire bytes up to (not including) tag 10
    static uint8_t compute_checksum(const std::string& wire);

private:
    std::unordered_map<uint32_t, std::string> fields_;
};

// ── FIX parser ────────────────────────────────────────────────────────────────

class FixParser {
public:
    struct Stats {
        uint64_t messages_parsed = 0;
        uint64_t parse_errors = 0;
        uint64_t checksum_errors = 0;
    };

    explicit FixParser(bool strict_checksum = false)
        : strict_checksum_(strict_checksum) {}

    // Parse a single FIX message from raw bytes. Returns bytes consumed.
    // Throws FixParseError on fatal parse errors.
    std::pair<FixMessage, size_t> parse(const uint8_t* data, size_t len);

    // Parse all messages from a buffer
    std::vector<FixMessage> parse_all(const uint8_t* data, size_t len);

    // Parse from a string
    FixMessage parse_string(const std::string& raw);

    const Stats& stats() const { return stats_; }

    // Tag name lookup
    static std::string tag_name(uint32_t tag);

private:
    bool strict_checksum_;
    Stats stats_;
    char separator_ = '\x01';

    size_t find_message_end(const uint8_t* data, size_t len) const;
};

// ── FIX message builder ───────────────────────────────────────────────────────

class FixMessageBuilder {
public:
    explicit FixMessageBuilder(const std::string& msg_type) {
        msg_.set(tags::MsgType, msg_type);
    }

    FixMessageBuilder& sender(const std::string& s) { msg_.set(tags::SenderCompID, s); return *this; }
    FixMessageBuilder& target(const std::string& t) { msg_.set(tags::TargetCompID, t); return *this; }
    FixMessageBuilder& seq_num(int64_t n) { msg_.set(tags::MsgSeqNum, n); return *this; }
    FixMessageBuilder& sending_time(const std::string& t) { msg_.set(tags::SendingTime, t); return *this; }
    FixMessageBuilder& field(uint32_t tag, const std::string& val) { msg_.set(tag, val); return *this; }
    FixMessageBuilder& field(uint32_t tag, double val, int prec = 6) { msg_.set(tag, val, prec); return *this; }
    FixMessageBuilder& field(uint32_t tag, int64_t val) { msg_.set(tag, val); return *this; }

    FixMessage build() const { return msg_; }

    static FixMessage new_order_single(
        const std::string& cl_ord_id,
        const std::string& symbol,
        char side,
        double qty,
        char ord_type,
        double price = 0.0
    );

    static FixMessage execution_report(
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
        double last_shares = 0.0,
        double last_px = 0.0
    );

    static FixMessage cancel_request(
        const std::string& cl_ord_id,
        const std::string& orig_cl_ord_id,
        const std::string& symbol,
        char side,
        double order_qty
    );

    static FixMessage cancel_replace_request(
        const std::string& cl_ord_id,
        const std::string& orig_cl_ord_id,
        const std::string& symbol,
        char side,
        double qty,
        double price
    );

    static FixMessage logon(int heartbeat_interval = 30, const std::string& reset_flag = "Y");
    static FixMessage logout(const std::string& text = "");
    static FixMessage heartbeat(const std::string& test_req_id = "");
    static FixMessage test_request(const std::string& test_req_id);
    static FixMessage resend_request(int64_t begin_seq_no, int64_t end_seq_no);

private:
    FixMessage msg_;
};

// ── FIX session state ─────────────────────────────────────────────────────────

enum class SessionState {
    Disconnected,
    Connecting,
    LogonPending,
    Active,
    LogoutPending,
    Reconnecting,
};

struct SessionConfig {
    std::string sender_comp_id;
    std::string target_comp_id;
    int heartbeat_interval = 30;
    bool reset_on_logon = true;
    bool validate_checksum = false;
    int max_message_size = 4096;
    std::string fix_version = "FIX.4.2";
};

class FixSession {
public:
    using MessageHandler = std::function<void(const FixMessage&)>;
    using StateChangeHandler = std::function<void(SessionState, SessionState)>;

    explicit FixSession(SessionConfig config);

    void set_message_handler(MessageHandler h) { msg_handler_ = std::move(h); }
    void set_state_handler(StateChangeHandler h) { state_handler_ = std::move(h); }

    // Process incoming raw bytes
    void on_data(const uint8_t* data, size_t len);
    void on_data(const std::string& raw);

    // Generate outgoing messages
    std::string generate_logon() const;
    std::string generate_logout(const std::string& text = "") const;
    std::string generate_heartbeat(const std::string& test_req_id = "") const;
    std::string generate_test_request(const std::string& test_req_id) const;
    std::string generate_resend_request(int64_t begin, int64_t end) const;
    std::string generate_sequence_reset(int64_t new_seq, bool gap_fill) const;
    std::string generate_reject(int64_t ref_seq, int reason, const std::string& text) const;

    // Send a pre-built message (stamps headers)
    std::string stamp_and_serialise(FixMessage msg);

    // State queries
    SessionState state() const { return state_; }
    bool is_active() const { return state_ == SessionState::Active; }
    int64_t next_send_seq() const { return next_send_seq_; }
    int64_t next_recv_seq() const { return next_recv_seq_; }

    // Force state (for testing)
    void set_state(SessionState s) { state_ = s; }

    const SessionConfig& config() const { return config_; }

private:
    SessionConfig config_;
    SessionState state_ = SessionState::Disconnected;
    int64_t next_send_seq_ = 1;
    int64_t next_recv_seq_ = 1;
    MessageHandler msg_handler_;
    StateChangeHandler state_handler_;
    FixParser parser_;
    std::string recv_buffer_;

    void transition(SessionState new_state);
    void dispatch(const FixMessage& msg);
    void handle_logon(const FixMessage& msg);
    void handle_logout(const FixMessage& msg);
    void handle_heartbeat(const FixMessage& msg);
    void handle_test_request(const FixMessage& msg);
    void handle_resend_request(const FixMessage& msg);
    void handle_sequence_reset(const FixMessage& msg);
    void handle_reject(const FixMessage& msg);
    bool validate_session_fields(const FixMessage& msg);

    static std::string current_utc_timestamp();
    std::string stamp_message(FixMessage& msg) const;
};

// ── Market data subscription ──────────────────────────────────────────────────

struct MarketDataEntry {
    char entry_type; // '0'=Bid, '1'=Ask, '2'=Trade
    double price;
    double size;
    std::string exchange;
};

struct MarketDataSnapshot {
    std::string symbol;
    std::vector<MarketDataEntry> entries;
    std::string request_id;
};

MarketDataSnapshot parse_market_data_snapshot(const FixMessage& msg);

// ── FIX message store (for resend) ───────────────────────────────────────────

class FixMessageStore {
public:
    explicit FixMessageStore(size_t max_size = 10000) : max_size_(max_size) {}

    void store(int64_t seq_num, const std::string& raw) {
        messages_[seq_num] = raw;
        if (messages_.size() > max_size_) {
            messages_.erase(messages_.begin());
        }
    }

    std::optional<std::string> retrieve(int64_t seq_num) const {
        auto it = messages_.find(seq_num);
        if (it == messages_.end()) return {};
        return it->second;
    }

    std::vector<std::pair<int64_t, std::string>> range(int64_t begin, int64_t end) const {
        std::vector<std::pair<int64_t, std::string>> result;
        auto it = messages_.lower_bound(begin);
        while (it != messages_.end() && (end == 0 || it->first <= end)) {
            result.emplace_back(it->first, it->second);
            ++it;
        }
        return result;
    }

    size_t size() const { return messages_.size(); }

private:
    std::map<int64_t, std::string> messages_;
    size_t max_size_;
};

} // namespace fix
} // namespace chronos
