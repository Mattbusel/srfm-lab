// itch_parser.cpp — NASDAQ ITCH 5.0 parser implementation and utilities.
// Chronos / AETERNUS — production C++ implementation.

#include "itch_parser.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <cmath>

namespace chronos {
namespace itch {

// ── File reader ──────────────────────────────────────────────────────────────

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw ItchParseError("Cannot open file: " + path);
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

// ── Full file parse ───────────────────────────────────────────────────────────

std::vector<std::shared_ptr<ItchMsg>> parse_itch_file(const std::string& path) {
    auto buf = read_file(path);
    ItchParser parser;
    return parser.parse_buffer(buf.data(), buf.size());
}

// ── Build order books from message stream ───────────────────────────────────

OrderBookManager build_books(const std::vector<std::shared_ptr<ItchMsg>>& messages) {
    OrderBookManager mgr;
    for (const auto& msg : messages) {
        mgr.apply(msg);
    }
    return mgr;
}

// ── Price formatting utilities ────────────────────────────────────────────────

std::string format_price(Price32 price_ticks) {
    double p = static_cast<double>(price_ticks) / ITCH_PRICE_SCALE;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << p;
    return oss.str();
}

std::string format_spread(Price32 spread_ticks) {
    double s = static_cast<double>(spread_ticks) / ITCH_PRICE_SCALE;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << s;
    return oss.str();
}

// ── Book snapshot printer ─────────────────────────────────────────────────────

std::string book_snapshot(const OrderBook& book, size_t depth) {
    auto [bids, asks] = book.depth(depth);
    std::ostringstream oss;
    oss << "=== OrderBook: " << book.symbol().str() << " ===\n";
    oss << std::setw(12) << "ASK QTY" << " | " << std::setw(10) << "ASK PRICE" << "\n";
    for (auto it = asks.rbegin(); it != asks.rend(); ++it) {
        oss << std::setw(12) << it->qty << " | " << std::setw(10) << format_price(it->price) << "\n";
    }
    oss << "--- MID: " << std::fixed << std::setprecision(4) << book.mid_price()
        << "  SPREAD: " << format_spread(book.spread()) << " ---\n";
    for (const auto& b : bids) {
        oss << std::setw(12) << format_price(b.price) << " | " << std::setw(10) << b.qty << "\n";
    }
    oss << "Orders: " << book.order_count() << " | Volume: " << book.total_volume() << "\n";
    return oss.str();
}

// ── Statistics computation ────────────────────────────────────────────────────

struct BookStats {
    double time_weighted_spread_bps = 0.0;
    double vwap = 0.0;
    uint64_t total_volume = 0;
    size_t total_messages = 0;
    double min_spread_bps = 1e18;
    double max_spread_bps = 0.0;
    uint64_t add_count = 0;
    uint64_t cancel_count = 0;
    uint64_t delete_count = 0;
    uint64_t execute_count = 0;
    uint64_t replace_count = 0;
};

BookStats compute_stats(const std::vector<std::shared_ptr<ItchMsg>>& messages, const Stock& target_stock) {
    BookStats stats;
    OrderBook book;
    book.set_symbol(target_stock);
    Nanos last_ts = 0;
    double cumulative_spread_ns = 0.0;
    double vwap_num = 0.0;
    double vwap_den = 0.0;

    for (const auto& msg : messages) {
        ++stats.total_messages;
        Nanos cur_ts = msg->timestamp;

        if (last_ts > 0 && cur_ts > last_ts) {
            double spread = book.spread_bps();
            cumulative_spread_ns += spread * static_cast<double>(cur_ts - last_ts);
        }
        last_ts = cur_ts;

        // Count message types
        switch (msg->type) {
            case MsgType::AddOrder:
            case MsgType::AddOrderMpid: ++stats.add_count; break;
            case MsgType::OrderCancel: ++stats.cancel_count; break;
            case MsgType::OrderDelete: ++stats.delete_count; break;
            case MsgType::OrderExecuted:
            case MsgType::OrderExecutedPrice: ++stats.execute_count; break;
            case MsgType::OrderReplace: ++stats.replace_count; break;
            default: break;
        }

        book.apply(msg);

        if (book.last_trade_qty() > 0 && book.last_trade_price() > 0) {
            double p = static_cast<double>(book.last_trade_price()) / ITCH_PRICE_SCALE;
            double q = static_cast<double>(book.last_trade_qty());
            vwap_num += p * q;
            vwap_den += q;
        }
    }

    stats.total_volume = book.total_volume();
    stats.vwap = (vwap_den > 0.0) ? vwap_num / vwap_den : 0.0;

    if (cur_ts > 0 && last_ts > 0) {
        // handled above
    }

    return stats;
}

// ── PCAP UDP packet extraction helper ────────────────────────────────────────

const uint8_t* extract_udp_payload(const uint8_t* frame, size_t frame_len, size_t& payload_len) {
    if (frame_len < 14) { payload_len = 0; return nullptr; }
    uint16_t ether_type = (static_cast<uint16_t>(frame[12]) << 8) | frame[13];
    if (ether_type != 0x0800) { payload_len = 0; return nullptr; }

    if (frame_len < 34) { payload_len = 0; return nullptr; }
    size_t ip_start = 14;
    uint8_t ihl = (frame[ip_start] & 0x0F) * 4;
    uint8_t protocol = frame[ip_start + 9];
    if (protocol != 17) { payload_len = 0; return nullptr; }

    size_t udp_start = ip_start + ihl;
    if (frame_len < udp_start + 8) { payload_len = 0; return nullptr; }

    uint16_t udp_len = (static_cast<uint16_t>(frame[udp_start + 4]) << 8) | frame[udp_start + 5];
    size_t payload_start = udp_start + 8;
    payload_len = udp_len > 8 ? udp_len - 8 : 0;

    if (frame_len < payload_start + payload_len) { payload_len = 0; return nullptr; }
    return frame + payload_start;
}

// ── Tick-by-tick reconstructor ────────────────────────────────────────────────

class TickByTickReconstructor {
public:
    explicit TickByTickReconstructor(const Stock& symbol) {
        book_.set_symbol(symbol);
    }

    struct Tick {
        Nanos timestamp_ns;
        double bid;
        double ask;
        double mid;
        double spread_bps;
        uint64_t bid_size;
        uint64_t ask_size;
        double last_trade;
        uint32_t trade_qty;
    };

    const std::vector<Tick>& ticks() const { return ticks_; }

    void on_message(const std::shared_ptr<ItchMsg>& msg) {
        book_.apply(msg);
        emit_tick(msg->timestamp);
    }

    void on_messages(const std::vector<std::shared_ptr<ItchMsg>>& msgs) {
        for (const auto& m : msgs) on_message(m);
    }

private:
    OrderBook book_;
    std::vector<Tick> ticks_;
    double last_mid_ = 0.0;

    void emit_tick(Nanos ts) {
        auto bid = book_.best_bid();
        auto ask = book_.best_ask();
        if (!bid || !ask) return;

        double b = static_cast<double>(bid->first) / ITCH_PRICE_SCALE;
        double a = static_cast<double>(ask->first) / ITCH_PRICE_SCALE;
        double mid = (b + a) / 2.0;

        if (std::abs(mid - last_mid_) < 1e-10 && !ticks_.empty()) return;
        last_mid_ = mid;

        Tick t;
        t.timestamp_ns = ts;
        t.bid = b;
        t.ask = a;
        t.mid = mid;
        t.spread_bps = book_.spread_bps();
        t.bid_size = bid->second;
        t.ask_size = ask->second;
        t.last_trade = static_cast<double>(book_.last_trade_price()) / ITCH_PRICE_SCALE;
        t.trade_qty = book_.last_trade_qty();
        ticks_.push_back(t);
    }
};

// ── Export utilities ──────────────────────────────────────────────────────────

std::string ticks_to_csv(const std::vector<TickByTickReconstructor::Tick>& ticks) {
    std::ostringstream oss;
    oss << "timestamp_ns,bid,ask,mid,spread_bps,bid_size,ask_size,last_trade,trade_qty\n";
    for (const auto& t : ticks) {
        oss << t.timestamp_ns << ","
            << std::fixed << std::setprecision(4)
            << t.bid << "," << t.ask << "," << t.mid << "," << t.spread_bps << ","
            << t.bid_size << "," << t.ask_size << ","
            << t.last_trade << "," << t.trade_qty << "\n";
    }
    return oss.str();
}

// ── Simple unit tests ─────────────────────────────────────────────────────────

#ifndef ITCH_NO_TESTS

namespace tests {

// Helper to build a framed ITCH AddOrder message
std::vector<uint8_t> make_add_order_frame(
    uint64_t ref_num, char side, uint32_t shares, const char* stock8, uint32_t price)
{
    const int PAYLOAD_LEN = 36;
    std::vector<uint8_t> buf(2 + PAYLOAD_LEN, 0);
    // 2-byte big-endian length
    buf[0] = 0; buf[1] = PAYLOAD_LEN;
    // type = 'A'
    buf[2] = static_cast<uint8_t>('A');
    // stock_locate (2), tracking_number (2): zeros
    // timestamp (6): zeros
    // ref_num (8) at offset 2+11=13
    for (int i = 7; i >= 0; --i) { buf[2 + 11 + (7 - i)] = (ref_num >> (i * 8)) & 0xFF; }
    // side at offset 2+19=21
    buf[2 + 19] = static_cast<uint8_t>(side);
    // shares (4) at offset 2+20=22
    buf[2 + 20] = (shares >> 24) & 0xFF;
    buf[2 + 21] = (shares >> 16) & 0xFF;
    buf[2 + 22] = (shares >> 8) & 0xFF;
    buf[2 + 23] = shares & 0xFF;
    // stock (8) at offset 2+24=26
    std::memcpy(buf.data() + 2 + 24, stock8, 8);
    // price (4) at offset 2+32=34
    buf[2 + 32] = (price >> 24) & 0xFF;
    buf[2 + 33] = (price >> 16) & 0xFF;
    buf[2 + 34] = (price >> 8) & 0xFF;
    buf[2 + 35] = price & 0xFF;
    return buf;
}

void test_add_order_parse() {
    const char stock[8] = "AAPL    ";
    auto frame = make_add_order_frame(42, 'B', 500, stock, 1000000);

    ItchParser parser;
    size_t offset = 0;
    auto msg = parser.parse_framed(frame.data(), frame.size(), offset);

    assert(msg != nullptr);
    assert(msg->type == MsgType::AddOrder);
    auto& add = *std::static_pointer_cast<MsgAddOrder>(msg);
    assert(add.ref_num == 42);
    assert(add.buy_sell == 'B');
    assert(add.shares == 500);
    assert(add.price == 1000000);
    assert(offset == frame.size());
}

void test_order_book_add_execute() {
    const char symbol8[8] = "MSFT    ";
    Stock sym(symbol8);

    OrderBook book;
    book.set_symbol(sym);

    // Build AddOrder message
    {
        auto add = std::make_shared<MsgAddOrder>();
        add->type = MsgType::AddOrder;
        add->timestamp = 1000;
        add->ref_num = 1;
        add->buy_sell = 'B';
        add->shares = 300;
        add->stock = sym;
        add->price = 500000; // $50.00
        book.apply(add);
    }
    {
        auto add = std::make_shared<MsgAddOrder>();
        add->type = MsgType::AddOrder;
        add->timestamp = 1001;
        add->ref_num = 2;
        add->buy_sell = 'S';
        add->shares = 200;
        add->stock = sym;
        add->price = 500100; // $50.01
        book.apply(add);
    }

    auto bid = book.best_bid();
    auto ask = book.best_ask();
    assert(bid.has_value());
    assert(ask.has_value());
    assert(bid->first == 500000);
    assert(ask->first == 500100);
    assert(book.spread() == 100);

    // Execute part of the bid
    {
        auto exec = std::make_shared<MsgOrderExecuted>();
        exec->type = MsgType::OrderExecuted;
        exec->timestamp = 1002;
        exec->ref_num = 1;
        exec->executed_shares = 100;
        exec->match_number = 99;
        book.apply(exec);
    }

    auto bid2 = book.best_bid();
    assert(bid2.has_value());
    assert(bid2->second == 200); // 300 - 100 = 200

    printf("test_order_book_add_execute: PASSED\n");
}

void test_order_book_delete() {
    const char sym8[8] = "GOOG    ";
    Stock sym(sym8);
    OrderBook book;
    book.set_symbol(sym);

    auto add = std::make_shared<MsgAddOrder>();
    add->type = MsgType::AddOrder; add->timestamp = 1; add->ref_num = 7;
    add->buy_sell = 'S'; add->shares = 100; add->stock = sym; add->price = 2000000;
    book.apply(add);
    assert(book.best_ask().has_value());

    auto del = std::make_shared<MsgOrderDelete>();
    del->type = MsgType::OrderDelete; del->timestamp = 2; del->ref_num = 7;
    book.apply(del);
    assert(!book.best_ask().has_value());

    printf("test_order_book_delete: PASSED\n");
}

void test_order_replace() {
    const char sym8[8] = "AMZN    ";
    Stock sym(sym8);
    OrderBook book; book.set_symbol(sym);

    auto add = std::make_shared<MsgAddOrder>();
    add->type = MsgType::AddOrder; add->timestamp = 1; add->ref_num = 10;
    add->buy_sell = 'B'; add->shares = 100; add->stock = sym; add->price = 300000;
    book.apply(add);

    auto rep = std::make_shared<MsgOrderReplace>();
    rep->type = MsgType::OrderReplace; rep->timestamp = 2;
    rep->orig_ref_num = 10; rep->new_ref_num = 11;
    rep->shares = 150; rep->price = 300100;
    book.apply(rep);

    auto bid = book.best_bid();
    assert(bid.has_value());
    assert(bid->first == 300100);
    assert(bid->second == 150);

    printf("test_order_replace: PASSED\n");
}

void run_all() {
    test_add_order_parse();
    test_order_book_add_execute();
    test_order_book_delete();
    test_order_replace();
    printf("All ITCH parser tests PASSED\n");
}

} // namespace tests
#endif // ITCH_NO_TESTS

} // namespace itch
} // namespace chronos

// ── Standalone main (for testing) ────────────────────────────────────────────

#ifndef ITCH_NO_MAIN
int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--test") {
        chronos::itch::tests::run_all();
        return 0;
    }
    if (argc < 2) {
        fprintf(stderr, "Usage: itch_parser <file.itch> [--test]\n");
        return 1;
    }
    try {
        auto msgs = chronos::itch::parse_itch_file(argv[1]);
        chronos::itch::OrderBookManager mgr;
        for (const auto& m : msgs) mgr.apply(m);
        printf("Parsed %zu messages, %zu instruments\n", msgs.size(), mgr.num_instruments());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
#endif
