#pragma once
// itch_parser.hpp — NASDAQ ITCH 5.0 binary protocol parser.
// All message types, order book builder.
// Chronos / AETERNUS — production-grade C++ implementation.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <optional>
#include <stdexcept>
#include <functional>
#include <cassert>
#include <algorithm>

namespace chronos {
namespace itch {

// ── Constants ────────────────────────────────────────────────────────────────

static constexpr uint32_t ITCH_PRICE_SCALE = 10000; // ITCH prices are in units of 1/10000

// ── Error types ───────────────────────────────────────────────────────────────

class ItchParseError : public std::runtime_error {
public:
    explicit ItchParseError(const std::string& msg) : std::runtime_error("ITCH: " + msg) {}
};

// ── Nanosecond timestamp ──────────────────────────────────────────────────────

using Nanos = uint64_t;
using OrderRef = uint64_t;
using Price32 = uint32_t;  // ITCH fixed-point price
using Qty32 = uint32_t;

// ── Stock symbol helper ───────────────────────────────────────────────────────

struct Stock {
    char data[8];
    Stock() { std::fill(data, data + 8, ' '); }
    explicit Stock(const char* src) { std::memcpy(data, src, 8); }
    std::string str() const {
        std::string s(data, 8);
        auto pos = s.find_last_not_of(' ');
        return (pos == std::string::npos) ? "" : s.substr(0, pos + 1);
    }
    bool operator==(const Stock& other) const { return std::memcmp(data, other.data, 8) == 0; }
};

// ── Message types ─────────────────────────────────────────────────────────────

enum class MsgType : char {
    SystemEvent          = 'S',
    StockDirectory       = 'R',
    StockTradingAction   = 'H',
    RegShoRestriction    = 'Y',
    MarketParticipantPos = 'L',
    MwcbDeclineLevel     = 'V',
    MwcbStatus           = 'W',
    AddOrder             = 'A',
    AddOrderMpid         = 'F',
    OrderExecuted        = 'E',
    OrderExecutedPrice   = 'C',
    OrderCancel          = 'X',
    OrderDelete          = 'D',
    OrderReplace         = 'U',
    Trade                = 'P',
    CrossTrade           = 'Q',
    BrokenTrade          = 'B',
    NetOrderImbalance    = 'I',
    RetailInterestMsg    = 'N',
    DirectListingPrice   = 'O',
    Unknown              = '\0',
};

// ── Base message ──────────────────────────────────────────────────────────────

struct ItchMsg {
    MsgType type;
    uint16_t stock_locate;
    uint16_t tracking_number;
    Nanos timestamp;

    virtual ~ItchMsg() = default;
    virtual std::string to_string() const = 0;
};

// ── Concrete message types ────────────────────────────────────────────────────

struct MsgSystemEvent : public ItchMsg {
    char event_code;
    std::string to_string() const override {
        return "SystemEvent[event=" + std::string(1, event_code) + "]";
    }
};

struct MsgStockDirectory : public ItchMsg {
    Stock stock;
    char market_category;
    char financial_status;
    uint32_t round_lot_size;
    char round_lots_only;
    char issue_classification;
    char issue_sub_type[2];
    char authenticity;
    char short_sale_threshold;
    char ipo_flag;
    char luld_ref_price_tier;
    char etp_flag;
    uint32_t etp_leverage_factor;
    char inverse_indicator;
    std::string to_string() const override {
        return "StockDirectory[stock=" + stock.str() + " mkt=" + market_category + "]";
    }
};

struct MsgStockTradingAction : public ItchMsg {
    Stock stock;
    char trading_state;
    char reserved;
    char reason[4];
    std::string to_string() const override {
        return "StockTradingAction[stock=" + stock.str() + " state=" + trading_state + "]";
    }
};

struct MsgRegShoRestriction : public ItchMsg {
    Stock stock;
    char reg_sho_action;
    std::string to_string() const override { return "RegShoRestriction"; }
};

struct MsgMarketParticipantPos : public ItchMsg {
    char mpid[4];
    Stock stock;
    char primary_mm;
    char mm_mode;
    char mp_state;
    std::string to_string() const override { return "MarketParticipantPos"; }
};

struct MsgMwcbDeclineLevel : public ItchMsg {
    int64_t level1;
    int64_t level2;
    int64_t level3;
    std::string to_string() const override { return "MwcbDeclineLevel"; }
};

struct MsgMwcbStatus : public ItchMsg {
    char breached_level;
    std::string to_string() const override { return "MwcbStatus"; }
};

struct MsgAddOrder : public ItchMsg {
    OrderRef ref_num;
    char buy_sell;
    Qty32 shares;
    Stock stock;
    Price32 price;
    bool is_mpid = false;
    char attribution[4]{};
    std::string to_string() const override {
        return "AddOrder[ref=" + std::to_string(ref_num)
            + " side=" + buy_sell
            + " shares=" + std::to_string(shares)
            + " price=" + std::to_string(price)
            + " stock=" + stock.str() + "]";
    }
};

struct MsgOrderExecuted : public ItchMsg {
    OrderRef ref_num;
    Qty32 executed_shares;
    uint64_t match_number;
    std::string to_string() const override {
        return "OrderExecuted[ref=" + std::to_string(ref_num) + " shares=" + std::to_string(executed_shares) + "]";
    }
};

struct MsgOrderExecutedPrice : public ItchMsg {
    OrderRef ref_num;
    Qty32 executed_shares;
    uint64_t match_number;
    char printable;
    Price32 execution_price;
    std::string to_string() const override {
        return "OrderExecutedPrice[ref=" + std::to_string(ref_num) + " price=" + std::to_string(execution_price) + "]";
    }
};

struct MsgOrderCancel : public ItchMsg {
    OrderRef ref_num;
    Qty32 cancelled_shares;
    std::string to_string() const override {
        return "OrderCancel[ref=" + std::to_string(ref_num) + " cancelled=" + std::to_string(cancelled_shares) + "]";
    }
};

struct MsgOrderDelete : public ItchMsg {
    OrderRef ref_num;
    std::string to_string() const override {
        return "OrderDelete[ref=" + std::to_string(ref_num) + "]";
    }
};

struct MsgOrderReplace : public ItchMsg {
    OrderRef orig_ref_num;
    OrderRef new_ref_num;
    Qty32 shares;
    Price32 price;
    std::string to_string() const override {
        return "OrderReplace[orig=" + std::to_string(orig_ref_num) + " new=" + std::to_string(new_ref_num) + "]";
    }
};

struct MsgTrade : public ItchMsg {
    OrderRef ref_num;
    char buy_sell;
    Qty32 shares;
    Stock stock;
    Price32 price;
    uint64_t match_number;
    std::string to_string() const override {
        return "Trade[price=" + std::to_string(price) + " shares=" + std::to_string(shares) + "]";
    }
};

struct MsgCrossTrade : public ItchMsg {
    uint64_t shares;
    Stock stock;
    Price32 cross_price;
    uint64_t match_number;
    char cross_type;
    std::string to_string() const override { return "CrossTrade"; }
};

struct MsgBrokenTrade : public ItchMsg {
    uint64_t match_number;
    std::string to_string() const override { return "BrokenTrade"; }
};

struct MsgNetOrderImbalance : public ItchMsg {
    uint64_t paired_shares;
    uint64_t imbalance_shares;
    char imbalance_direction;
    Stock stock;
    Price32 far_price;
    Price32 near_price;
    Price32 ref_price;
    char cross_type;
    char price_variation_indicator;
    std::string to_string() const override { return "NetOrderImbalance"; }
};

struct MsgRetailInterest : public ItchMsg {
    Stock stock;
    char interest_flag;
    std::string to_string() const override { return "RetailInterest"; }
};

struct MsgDirectListingPrice : public ItchMsg {
    Stock stock;
    char open_eligibility_status;
    Price32 min_allowed_price;
    Price32 max_allowed_price;
    Price32 near_execution_price;
    uint64_t near_execution_time;
    std::string to_string() const override { return "DirectListingPrice"; }
};

// ── Binary reader helpers ────────────────────────────────────────────────────

namespace detail {

inline uint8_t read_u8(const uint8_t* buf, size_t off) { return buf[off]; }

inline uint16_t read_u16_be(const uint8_t* buf, size_t off) {
    return (static_cast<uint16_t>(buf[off]) << 8) | buf[off + 1];
}

inline uint32_t read_u32_be(const uint8_t* buf, size_t off) {
    return (static_cast<uint32_t>(buf[off]) << 24)
         | (static_cast<uint32_t>(buf[off+1]) << 16)
         | (static_cast<uint32_t>(buf[off+2]) << 8)
         | static_cast<uint32_t>(buf[off+3]);
}

inline uint64_t read_u64_be(const uint8_t* buf, size_t off) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v = (v << 8) | buf[off + i];
    return v;
}

inline int64_t read_i64_be(const uint8_t* buf, size_t off) {
    return static_cast<int64_t>(read_u64_be(buf, off));
}

inline Nanos read_ts_6(const uint8_t* buf, size_t off) {
    // 6-byte big-endian nanoseconds
    uint64_t hi = read_u16_be(buf, off);
    uint64_t lo = read_u32_be(buf, off + 2);
    return (hi << 32) | lo;
}

inline Stock read_stock(const uint8_t* buf, size_t off) {
    Stock s;
    std::memcpy(s.data, buf + off, 8);
    return s;
}

} // namespace detail

// ── ITCH 5.0 Parser ──────────────────────────────────────────────────────────

class ItchParser {
public:
    struct Stats {
        uint64_t messages_parsed = 0;
        uint64_t parse_errors = 0;
        uint64_t bytes_consumed = 0;
        uint64_t unknown_types = 0;
    };

    ItchParser() = default;

    // Parse a single framed message (2-byte big-endian length prefix + payload).
    // Returns parsed message and advances offset.
    std::shared_ptr<ItchMsg> parse_framed(const uint8_t* buf, size_t buf_len, size_t& offset) {
        if (offset + 2 > buf_len) return nullptr;
        uint16_t msg_len = detail::read_u16_be(buf, offset);
        if (offset + 2 + msg_len > buf_len) return nullptr;

        const uint8_t* payload = buf + offset + 2;
        auto msg = parse_payload(payload, msg_len);
        offset += 2 + msg_len;
        stats_.bytes_consumed += 2 + msg_len;
        if (msg) {
            ++stats_.messages_parsed;
        } else {
            ++stats_.parse_errors;
        }
        return msg;
    }

    std::vector<std::shared_ptr<ItchMsg>> parse_buffer(const uint8_t* buf, size_t buf_len) {
        std::vector<std::shared_ptr<ItchMsg>> messages;
        size_t offset = 0;
        while (offset < buf_len) {
            if (buf_len - offset < 2) break;
            auto msg = parse_framed(buf, buf_len, offset);
            if (msg) messages.push_back(std::move(msg));
        }
        return messages;
    }

    const Stats& stats() const { return stats_; }

private:
    Stats stats_;

    std::shared_ptr<ItchMsg> parse_payload(const uint8_t* p, size_t len) {
        if (len == 0) return nullptr;
        char type_char = static_cast<char>(p[0]);
        MsgType type = static_cast<MsgType>(type_char);

        auto fill_base = [&](auto& msg) {
            msg.type = type;
            msg.stock_locate = detail::read_u16_be(p, 1);
            msg.tracking_number = detail::read_u16_be(p, 3);
            msg.timestamp = detail::read_ts_6(p, 5);
        };

        switch (type) {
            case MsgType::SystemEvent: {
                auto msg = std::make_shared<MsgSystemEvent>();
                fill_base(*msg);
                if (len < 12) return nullptr;
                msg->event_code = static_cast<char>(p[11]);
                return msg;
            }
            case MsgType::StockDirectory: {
                auto msg = std::make_shared<MsgStockDirectory>();
                fill_base(*msg);
                if (len < 39) return nullptr;
                msg->stock = detail::read_stock(p, 11);
                msg->market_category = static_cast<char>(p[19]);
                msg->financial_status = static_cast<char>(p[20]);
                msg->round_lot_size = detail::read_u32_be(p, 21);
                msg->round_lots_only = static_cast<char>(p[25]);
                msg->issue_classification = static_cast<char>(p[26]);
                return msg;
            }
            case MsgType::StockTradingAction: {
                auto msg = std::make_shared<MsgStockTradingAction>();
                fill_base(*msg);
                if (len < 25) return nullptr;
                msg->stock = detail::read_stock(p, 11);
                msg->trading_state = static_cast<char>(p[19]);
                msg->reserved = static_cast<char>(p[20]);
                std::memcpy(msg->reason, p + 21, 4);
                return msg;
            }
            case MsgType::RegShoRestriction: {
                auto msg = std::make_shared<MsgRegShoRestriction>();
                fill_base(*msg);
                if (len < 20) return nullptr;
                msg->stock = detail::read_stock(p, 11);
                msg->reg_sho_action = static_cast<char>(p[19]);
                return msg;
            }
            case MsgType::MarketParticipantPos: {
                auto msg = std::make_shared<MsgMarketParticipantPos>();
                fill_base(*msg);
                if (len < 26) return nullptr;
                std::memcpy(msg->mpid, p + 11, 4);
                msg->stock = detail::read_stock(p, 15);
                msg->primary_mm = static_cast<char>(p[23]);
                msg->mm_mode = static_cast<char>(p[24]);
                msg->mp_state = static_cast<char>(p[25]);
                return msg;
            }
            case MsgType::MwcbDeclineLevel: {
                auto msg = std::make_shared<MsgMwcbDeclineLevel>();
                fill_base(*msg);
                if (len < 35) return nullptr;
                msg->level1 = detail::read_i64_be(p, 11);
                msg->level2 = detail::read_i64_be(p, 19);
                msg->level3 = detail::read_i64_be(p, 27);
                return msg;
            }
            case MsgType::MwcbStatus: {
                auto msg = std::make_shared<MsgMwcbStatus>();
                fill_base(*msg);
                if (len < 12) return nullptr;
                msg->breached_level = static_cast<char>(p[11]);
                return msg;
            }
            case MsgType::AddOrder: {
                auto msg = std::make_shared<MsgAddOrder>();
                fill_base(*msg);
                if (len < 36) return nullptr;
                msg->ref_num = detail::read_u64_be(p, 11);
                msg->buy_sell = static_cast<char>(p[19]);
                msg->shares = detail::read_u32_be(p, 20);
                msg->stock = detail::read_stock(p, 24);
                msg->price = detail::read_u32_be(p, 32);
                return msg;
            }
            case MsgType::AddOrderMpid: {
                auto msg = std::make_shared<MsgAddOrder>();
                fill_base(*msg);
                if (len < 40) return nullptr;
                msg->ref_num = detail::read_u64_be(p, 11);
                msg->buy_sell = static_cast<char>(p[19]);
                msg->shares = detail::read_u32_be(p, 20);
                msg->stock = detail::read_stock(p, 24);
                msg->price = detail::read_u32_be(p, 32);
                msg->is_mpid = true;
                std::memcpy(msg->attribution, p + 36, 4);
                return msg;
            }
            case MsgType::OrderExecuted: {
                auto msg = std::make_shared<MsgOrderExecuted>();
                fill_base(*msg);
                if (len < 31) return nullptr;
                msg->ref_num = detail::read_u64_be(p, 11);
                msg->executed_shares = detail::read_u32_be(p, 19);
                msg->match_number = detail::read_u64_be(p, 23);
                return msg;
            }
            case MsgType::OrderExecutedPrice: {
                auto msg = std::make_shared<MsgOrderExecutedPrice>();
                fill_base(*msg);
                if (len < 36) return nullptr;
                msg->ref_num = detail::read_u64_be(p, 11);
                msg->executed_shares = detail::read_u32_be(p, 19);
                msg->match_number = detail::read_u64_be(p, 23);
                msg->printable = static_cast<char>(p[31]);
                msg->execution_price = detail::read_u32_be(p, 32);
                return msg;
            }
            case MsgType::OrderCancel: {
                auto msg = std::make_shared<MsgOrderCancel>();
                fill_base(*msg);
                if (len < 23) return nullptr;
                msg->ref_num = detail::read_u64_be(p, 11);
                msg->cancelled_shares = detail::read_u32_be(p, 19);
                return msg;
            }
            case MsgType::OrderDelete: {
                auto msg = std::make_shared<MsgOrderDelete>();
                fill_base(*msg);
                if (len < 19) return nullptr;
                msg->ref_num = detail::read_u64_be(p, 11);
                return msg;
            }
            case MsgType::OrderReplace: {
                auto msg = std::make_shared<MsgOrderReplace>();
                fill_base(*msg);
                if (len < 35) return nullptr;
                msg->orig_ref_num = detail::read_u64_be(p, 11);
                msg->new_ref_num = detail::read_u64_be(p, 19);
                msg->shares = detail::read_u32_be(p, 27);
                msg->price = detail::read_u32_be(p, 31);
                return msg;
            }
            case MsgType::Trade: {
                auto msg = std::make_shared<MsgTrade>();
                fill_base(*msg);
                if (len < 44) return nullptr;
                msg->ref_num = detail::read_u64_be(p, 11);
                msg->buy_sell = static_cast<char>(p[19]);
                msg->shares = detail::read_u32_be(p, 20);
                msg->stock = detail::read_stock(p, 24);
                msg->price = detail::read_u32_be(p, 32);
                msg->match_number = detail::read_u64_be(p, 36);
                return msg;
            }
            case MsgType::CrossTrade: {
                auto msg = std::make_shared<MsgCrossTrade>();
                fill_base(*msg);
                if (len < 40) return nullptr;
                msg->shares = detail::read_u64_be(p, 11);
                msg->stock = detail::read_stock(p, 19);
                msg->cross_price = detail::read_u32_be(p, 27);
                msg->match_number = detail::read_u64_be(p, 31);
                msg->cross_type = static_cast<char>(p[39]);
                return msg;
            }
            case MsgType::BrokenTrade: {
                auto msg = std::make_shared<MsgBrokenTrade>();
                fill_base(*msg);
                if (len < 19) return nullptr;
                msg->match_number = detail::read_u64_be(p, 11);
                return msg;
            }
            case MsgType::NetOrderImbalance: {
                auto msg = std::make_shared<MsgNetOrderImbalance>();
                fill_base(*msg);
                if (len < 50) return nullptr;
                msg->paired_shares = detail::read_u64_be(p, 11);
                msg->imbalance_shares = detail::read_u64_be(p, 19);
                msg->imbalance_direction = static_cast<char>(p[27]);
                msg->stock = detail::read_stock(p, 28);
                msg->far_price = detail::read_u32_be(p, 36);
                msg->near_price = detail::read_u32_be(p, 40);
                msg->ref_price = detail::read_u32_be(p, 44);
                msg->cross_type = static_cast<char>(p[48]);
                msg->price_variation_indicator = static_cast<char>(p[49]);
                return msg;
            }
            case MsgType::RetailInterestMsg: {
                auto msg = std::make_shared<MsgRetailInterest>();
                fill_base(*msg);
                if (len < 20) return nullptr;
                msg->stock = detail::read_stock(p, 11);
                msg->interest_flag = static_cast<char>(p[19]);
                return msg;
            }
            case MsgType::DirectListingPrice: {
                auto msg = std::make_shared<MsgDirectListingPrice>();
                fill_base(*msg);
                if (len < 40) return nullptr;
                msg->stock = detail::read_stock(p, 11);
                msg->open_eligibility_status = static_cast<char>(p[19]);
                msg->min_allowed_price = detail::read_u32_be(p, 20);
                msg->max_allowed_price = detail::read_u32_be(p, 24);
                msg->near_execution_price = detail::read_u32_be(p, 28);
                msg->near_execution_time = detail::read_u64_be(p, 32);
                return msg;
            }
            default:
                ++stats_.unknown_types;
                return nullptr;
        }
    }
};

// ── Order book entry ──────────────────────────────────────────────────────────

struct BookEntry {
    OrderRef ref_num;
    char buy_sell;
    Qty32 shares;
    Price32 price;
    Stock stock;
};

// ── Order book reconstruction ────────────────────────────────────────────────

class OrderBook {
public:
    const Stock& symbol() const { return symbol_; }
    void set_symbol(const Stock& s) { symbol_ = s; }

    Nanos last_timestamp() const { return last_ts_; }
    uint64_t sequence() const { return sequence_; }

    void apply(const std::shared_ptr<ItchMsg>& msg) {
        if (!msg) return;
        ++sequence_;
        last_ts_ = msg->timestamp;

        switch (msg->type) {
            case MsgType::AddOrder:
            case MsgType::AddOrderMpid: {
                const auto& m = *std::static_pointer_cast<MsgAddOrder>(msg);
                if (!(m.stock == symbol_)) return;
                BookEntry e{m.ref_num, m.buy_sell, m.shares, m.price, m.stock};
                orders_[m.ref_num] = e;
                add_to_book(m.buy_sell == 'B', m.price, m.shares);
                break;
            }
            case MsgType::OrderExecuted: {
                const auto& m = *std::static_pointer_cast<MsgOrderExecuted>(msg);
                auto it = orders_.find(m.ref_num);
                if (it == orders_.end()) break;
                auto& e = it->second;
                remove_from_book(e.buy_sell == 'B', e.price, m.executed_shares);
                last_trade_price_ = e.price;
                last_trade_qty_ = m.executed_shares;
                total_trade_volume_ += m.executed_shares;
                if (e.shares <= m.executed_shares) orders_.erase(it);
                else e.shares -= m.executed_shares;
                break;
            }
            case MsgType::OrderExecutedPrice: {
                const auto& m = *std::static_pointer_cast<MsgOrderExecutedPrice>(msg);
                auto it = orders_.find(m.ref_num);
                if (it == orders_.end()) break;
                auto& e = it->second;
                remove_from_book(e.buy_sell == 'B', e.price, m.executed_shares);
                last_trade_price_ = m.execution_price;
                last_trade_qty_ = m.executed_shares;
                total_trade_volume_ += m.executed_shares;
                if (e.shares <= m.executed_shares) orders_.erase(it);
                else e.shares -= m.executed_shares;
                break;
            }
            case MsgType::OrderCancel: {
                const auto& m = *std::static_pointer_cast<MsgOrderCancel>(msg);
                auto it = orders_.find(m.ref_num);
                if (it == orders_.end()) break;
                auto& e = it->second;
                remove_from_book(e.buy_sell == 'B', e.price, m.cancelled_shares);
                if (e.shares <= m.cancelled_shares) orders_.erase(it);
                else e.shares -= m.cancelled_shares;
                break;
            }
            case MsgType::OrderDelete: {
                const auto& m = *std::static_pointer_cast<MsgOrderDelete>(msg);
                auto it = orders_.find(m.ref_num);
                if (it == orders_.end()) break;
                auto& e = it->second;
                remove_from_book(e.buy_sell == 'B', e.price, e.shares);
                orders_.erase(it);
                break;
            }
            case MsgType::OrderReplace: {
                const auto& m = *std::static_pointer_cast<MsgOrderReplace>(msg);
                auto it = orders_.find(m.orig_ref_num);
                if (it == orders_.end()) break;
                BookEntry old_e = it->second;
                orders_.erase(it);
                remove_from_book(old_e.buy_sell == 'B', old_e.price, old_e.shares);
                BookEntry new_e{m.new_ref_num, old_e.buy_sell, m.shares, m.price, old_e.stock};
                orders_[m.new_ref_num] = new_e;
                add_to_book(new_e.buy_sell == 'B', new_e.price, new_e.shares);
                break;
            }
            default: break;
        }
    }

    // Best bid (price, qty)
    std::optional<std::pair<Price32, uint64_t>> best_bid() const {
        if (bids_.empty()) return {};
        auto it = bids_.rbegin();
        return std::make_pair(it->first, it->second);
    }

    // Best ask (price, qty)
    std::optional<std::pair<Price32, uint64_t>> best_ask() const {
        if (asks_.empty()) return {};
        auto it = asks_.begin();
        return std::make_pair(it->first, it->second);
    }

    double mid_price() const {
        auto b = best_bid();
        auto a = best_ask();
        if (!b || !a) return 0.0;
        return (static_cast<double>(b->first) + a->first) / 2.0 / ITCH_PRICE_SCALE;
    }

    Price32 spread() const {
        auto b = best_bid();
        auto a = best_ask();
        if (!b || !a) return 0;
        return (a->first >= b->first) ? a->first - b->first : 0;
    }

    double spread_bps() const {
        double mid = mid_price();
        if (mid < 1e-9) return 0.0;
        return static_cast<double>(spread()) / ITCH_PRICE_SCALE / mid * 10000.0;
    }

    size_t order_count() const { return orders_.size(); }
    size_t bid_levels() const { return bids_.size(); }
    size_t ask_levels() const { return asks_.size(); }
    Price32 last_trade_price() const { return last_trade_price_; }
    Qty32 last_trade_qty() const { return last_trade_qty_; }
    uint64_t total_volume() const { return total_trade_volume_; }

    // Get N levels of book depth
    struct DepthLevel { Price32 price; uint64_t qty; };
    std::pair<std::vector<DepthLevel>, std::vector<DepthLevel>> depth(size_t n) const {
        std::vector<DepthLevel> bid_levels_v, ask_levels_v;
        size_t i = 0;
        for (auto it = bids_.rbegin(); it != bids_.rend() && i < n; ++it, ++i)
            bid_levels_v.push_back({it->first, it->second});
        i = 0;
        for (auto it = asks_.begin(); it != asks_.end() && i < n; ++it, ++i)
            ask_levels_v.push_back({it->first, it->second});
        return {bid_levels_v, ask_levels_v};
    }

private:
    Stock symbol_;
    std::unordered_map<OrderRef, BookEntry> orders_;
    std::map<Price32, uint64_t> bids_; // ascending by price
    std::map<Price32, uint64_t> asks_; // ascending by price
    Nanos last_ts_ = 0;
    uint64_t sequence_ = 0;
    Price32 last_trade_price_ = 0;
    Qty32 last_trade_qty_ = 0;
    uint64_t total_trade_volume_ = 0;

    void add_to_book(bool is_bid, Price32 price, Qty32 qty) {
        auto& side = is_bid ? bids_ : asks_;
        side[price] += qty;
    }

    void remove_from_book(bool is_bid, Price32 price, Qty32 qty) {
        auto& side = is_bid ? bids_ : asks_;
        auto it = side.find(price);
        if (it == side.end()) return;
        if (it->second <= qty) side.erase(it);
        else it->second -= qty;
    }
};

// ── Multi-stock order book manager ──────────────────────────────────────────

class OrderBookManager {
public:
    OrderBookManager() = default;

    void apply(const std::shared_ptr<ItchMsg>& msg) {
        if (!msg) return;
        const Stock* stock = get_stock(msg);
        if (!stock) return;

        auto key = stock_key(*stock);
        auto& book = books_[key];
        book.set_symbol(*stock);
        book.apply(msg);
    }

    OrderBook* get_book(const Stock& s) {
        auto key = stock_key(s);
        auto it = books_.find(key);
        if (it == books_.end()) return nullptr;
        return &it->second;
    }

    const OrderBook* get_book(const Stock& s) const {
        auto key = stock_key(s);
        auto it = books_.find(key);
        if (it == books_.end()) return nullptr;
        return &it->second;
    }

    size_t num_instruments() const { return books_.size(); }

    void for_each_book(std::function<void(const Stock&, const OrderBook&)> fn) const {
        for (const auto& [key, book] : books_) {
            fn(book.symbol(), book);
        }
    }

private:
    std::unordered_map<uint64_t, OrderBook> books_;

    static uint64_t stock_key(const Stock& s) {
        uint64_t k = 0;
        std::memcpy(&k, s.data, 8);
        return k;
    }

    static const Stock* get_stock(const std::shared_ptr<ItchMsg>& msg) {
        switch (msg->type) {
            case MsgType::AddOrder:
            case MsgType::AddOrderMpid:
                return &std::static_pointer_cast<MsgAddOrder>(msg)->stock;
            case MsgType::Trade:
                return &std::static_pointer_cast<MsgTrade>(msg)->stock;
            case MsgType::StockDirectory:
                return &std::static_pointer_cast<MsgStockDirectory>(msg)->stock;
            default:
                return nullptr;
        }
    }
};

} // namespace itch
} // namespace chronos
