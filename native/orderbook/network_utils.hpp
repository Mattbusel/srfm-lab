#pragma once
// Network utilities for HFT: UDP multicast simulation, packet framing,
// sequence number tracking, and network latency modeling.

#include "order.hpp"
#include <cstdint>
#include <cstring>
#include <vector>
#include <deque>
#include <queue>
#include <unordered_map>
#include <string>
#include <functional>
#include <random>
#include <chrono>
#include <atomic>
#include <cassert>
#include <algorithm>
#include <numeric>

namespace hft {

// ============================================================
// Packet framing protocol (simplified UDP multicast format)
// ============================================================
static constexpr uint32_t PACKET_MAGIC   = 0x48465431; // "HFT1"
static constexpr uint8_t  MSG_TYPE_ORDER = 0x01;
static constexpr uint8_t  MSG_TYPE_TRADE = 0x02;
static constexpr uint8_t  MSG_TYPE_BOOK  = 0x03;
static constexpr uint8_t  MSG_TYPE_HBEAT = 0xFF;

#pragma pack(push, 1)
struct PacketHeader {
    uint32_t magic;
    uint16_t version;
    uint8_t  msg_type;
    uint8_t  flags;
    uint32_t seq_num;
    uint32_t session_id;
    int64_t  send_time_ns;
    uint16_t payload_len;
    uint16_t checksum;
};
static_assert(sizeof(PacketHeader) == 24, "PacketHeader must be 24 bytes");

struct OrderPacket {
    PacketHeader hdr;
    char         symbol[16];
    int64_t      price;       // fixed-point
    uint64_t     qty;
    uint64_t     order_id;
    uint8_t      side;        // 0=Buy, 1=Sell
    uint8_t      order_type;  // 0=Limit, 1=Market
    uint8_t      tif;
    uint8_t      _pad;
    int64_t      timestamp_ns;
};

struct TradePacket {
    PacketHeader hdr;
    char         symbol[16];
    int64_t      price;
    uint64_t     qty;
    uint64_t     buyer_order_id;
    uint64_t     seller_order_id;
    uint64_t     trade_id;
    uint8_t      aggressor_side;
    uint8_t      _pad[7];
    int64_t      timestamp_ns;
};

struct BookLevelPacket {
    PacketHeader hdr;
    char         symbol[16];
    uint8_t      side;
    uint8_t      level;
    uint8_t      action;      // 0=Add, 1=Update, 2=Delete
    uint8_t      _pad;
    int64_t      price;
    uint64_t     qty;
    uint32_t     order_count;
    uint32_t     _pad2;
};
#pragma pack(pop)

// ============================================================
// Checksum utility (Fletcher-16)
// ============================================================
inline uint16_t fletcher16(const uint8_t* data, size_t len) {
    uint16_t sum1 = 0, sum2 = 0;
    for (size_t i = 0; i < len; ++i) {
        sum1 = (sum1 + data[i]) % 255;
        sum2 = (sum2 + sum1) % 255;
    }
    return (sum2 << 8) | sum1;
}

// ============================================================
// Sequence number manager
// Tracks gaps, reorders, and duplicate packets
// ============================================================
class SeqManager {
public:
    explicit SeqManager(uint32_t expected_first = 1)
        : next_expected_(expected_first) {}

    enum class Status {
        InOrder,
        Duplicate,
        OutOfOrder,
        Gap,
    };

    struct Stats {
        uint64_t in_order     = 0;
        uint64_t duplicates   = 0;
        uint64_t out_of_order = 0;
        uint64_t gaps         = 0;
        uint64_t gap_seq_lost = 0;
    };

    Status process(uint32_t seq_num, std::vector<uint32_t>& recovered_gap) {
        recovered_gap.clear();

        if (seq_num < next_expected_) {
            stats_.duplicates++;
            return Status::Duplicate;
        }

        if (seq_num == next_expected_) {
            next_expected_++;
            stats_.in_order++;
            // Try to drain pending buffer
            while (true) {
                auto it = pending_.find(next_expected_);
                if (it == pending_.end()) break;
                recovered_gap.push_back(next_expected_);
                pending_.erase(it);
                next_expected_++;
                stats_.in_order++;
            }
            return Status::InOrder;
        }

        // seq_num > next_expected_ → gap
        if (!pending_.count(seq_num)) {
            pending_.insert(seq_num);
            // Record gap
            uint64_t gap_size = seq_num - next_expected_;
            if (gap_size > 0) {
                stats_.gaps++;
                stats_.gap_seq_lost += gap_size - 1;
                for (uint32_t s = next_expected_; s < seq_num; ++s)
                    missing_.push_back(s);
            }
        }
        return Status::Gap;
    }

    std::vector<uint32_t> get_missing() const { return missing_; }
    uint32_t next_expected() const noexcept { return next_expected_; }
    const Stats& stats() const noexcept { return stats_; }
    void reset(uint32_t seq = 1) { next_expected_ = seq; pending_.clear(); missing_.clear(); }

private:
    uint32_t                          next_expected_;
    std::unordered_set<uint32_t>      pending_;
    std::vector<uint32_t>             missing_;
    Stats                             stats_;
};

// ============================================================
// Network Latency Model
// Models realistic intraday latency patterns
// ============================================================
class LatencyModel {
public:
    struct Config {
        double base_ns          = 5000.0;   // 5 µs base
        double jitter_ns        = 1000.0;   // 1 µs jitter
        double spike_prob       = 0.001;    // 0.1% spike probability
        double spike_magnitude  = 1e6;      // 1 ms spike
        double diurnal_factor   = 0.3;      // intraday variation
        uint64_t seed           = 42;
    };

    explicit LatencyModel(const Config& cfg = {}) : cfg_(cfg), rng_(cfg.seed) {
        norm_  = std::normal_distribution<double>(0, 1);
        unif_  = std::uniform_real_distribution<double>(0, 1);
    }

    // Returns latency in nanoseconds, given time-of-day in [0,1] (0=open, 1=close)
    double sample(double tod = 0.5) {
        // Log-normal base latency
        double lat = cfg_.base_ns * std::exp(norm_(rng_) * 0.2);

        // Jitter
        lat += cfg_.jitter_ns * std::fabs(norm_(rng_));

        // Diurnal pattern: higher at open/close
        double diurnal = 1.0 + cfg_.diurnal_factor *
                         (std::exp(-50 * (tod - 0.0) * (tod - 0.0)) +
                          std::exp(-50 * (tod - 1.0) * (tod - 1.0)));
        lat *= diurnal;

        // Random spikes
        if (unif_(rng_) < cfg_.spike_prob)
            lat += cfg_.spike_magnitude * (1.0 + unif_(rng_));

        return lat;
    }

    // Batch sample
    std::vector<double> sample_batch(size_t n, double tod = 0.5) {
        std::vector<double> v(n);
        for (auto& x : v) x = sample(tod);
        return v;
    }

    // Model switching latency (co-location vs. off-site)
    double sample_colocation()  { return std::max(200.0, sample(0.5) * 0.01); }
    double sample_cross_venue() { return sample(0.5) * 5.0; }

    struct Percentiles {
        double p50, p95, p99, p999, mean;
    };

    Percentiles benchmark(size_t n = 100000) {
        auto samples = sample_batch(n);
        std::sort(samples.begin(), samples.end());
        Percentiles p{};
        p.p50  = samples[n * 50 / 100];
        p.p95  = samples[n * 95 / 100];
        p.p99  = samples[n * 99 / 100];
        p.p999 = samples[std::min(n - 1, n * 999 / 1000)];
        p.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / n;
        return p;
    }

private:
    Config                                  cfg_;
    std::mt19937                            rng_;
    std::normal_distribution<double>        norm_;
    std::uniform_real_distribution<double>  unif_;
};

// ============================================================
// Simulated UDP multicast feed
// ============================================================
class SimulatedFeed {
public:
    struct Config {
        std::string symbol         = "AAPL";
        double      initial_price  = 150.0;
        double      daily_vol      = 0.20;
        int         n_messages     = 100000;
        double      order_rate     = 1000.0;   // orders/sec
        double      trade_rate     = 200.0;
        double      packet_loss_p  = 0.0001;   // 0.01% loss
        uint64_t    seed           = 42;
    };

    using MessageCallback = std::function<void(const PacketHeader&, const void*)>;

    explicit SimulatedFeed(const Config& cfg = {}) : cfg_(cfg), rng_(cfg.seed), lat_model_() {
        norm_ = std::normal_distribution<double>(0, 1);
        unif_ = std::uniform_real_distribution<double>(0, 1);
        qty_dist_ = std::uniform_int_distribution<int>(50, 500);
        seq_ = 1;
    }

    void set_callback(MessageCallback cb) { callback_ = std::move(cb); }

    // Generate and deliver N messages
    void run() {
        double price = cfg_.initial_price;
        double sigma_dt = cfg_.daily_vol / std::sqrt(252.0 * 6.5 * 3600.0);
        int64_t ts_ns = 34200LL * 1000000000LL; // 9:30 AM
        uint64_t order_id = 1;
        uint64_t trade_id = 1;

        for (int i = 0; i < cfg_.n_messages; ++i) {
            // GBM step
            double z = norm_(rng_);
            price *= std::exp(-0.5 * sigma_dt * sigma_dt + sigma_dt * z);
            price = std::max(price, 0.01);
            ts_ns += 1000000; // 1ms

            // Packet loss simulation
            if (unif_(rng_) < cfg_.packet_loss_p) {
                seq_++;
                dropped_packets_++;
                continue;
            }

            double r = unif_(rng_);
            double threshold_order = cfg_.order_rate / (cfg_.order_rate + cfg_.trade_rate);

            if (r < threshold_order) {
                // Order packet
                OrderPacket pkt{};
                pkt.hdr.magic        = PACKET_MAGIC;
                pkt.hdr.version      = 1;
                pkt.hdr.msg_type     = MSG_TYPE_ORDER;
                pkt.hdr.seq_num      = seq_++;
                pkt.hdr.send_time_ns = ts_ns;
                pkt.hdr.payload_len  = sizeof(OrderPacket) - sizeof(PacketHeader);

                std::memcpy(pkt.symbol, cfg_.symbol.c_str(),
                            std::min(cfg_.symbol.size(), (size_t)15));
                bool is_buy = unif_(rng_) < 0.5;
                double half_sp = price * 0.001;
                double px_d = is_buy ? price - half_sp - unif_(rng_) * half_sp
                                     : price + half_sp + unif_(rng_) * half_sp;

                pkt.price      = double_to_price(px_d);
                pkt.qty        = qty_dist_(rng_);
                pkt.order_id   = order_id++;
                pkt.side       = is_buy ? 0 : 1;
                pkt.order_type = 0; // Limit
                pkt.tif        = 0;
                pkt.timestamp_ns = ts_ns;

                uint8_t* payload = reinterpret_cast<uint8_t*>(&pkt) + sizeof(PacketHeader);
                pkt.hdr.checksum = fletcher16(payload, pkt.hdr.payload_len);

                sent_packets_++;
                if (callback_) callback_(pkt.hdr, &pkt);

            } else {
                // Trade packet
                TradePacket pkt{};
                pkt.hdr.magic        = PACKET_MAGIC;
                pkt.hdr.version      = 1;
                pkt.hdr.msg_type     = MSG_TYPE_TRADE;
                pkt.hdr.seq_num      = seq_++;
                pkt.hdr.send_time_ns = ts_ns;
                pkt.hdr.payload_len  = sizeof(TradePacket) - sizeof(PacketHeader);

                std::memcpy(pkt.symbol, cfg_.symbol.c_str(),
                            std::min(cfg_.symbol.size(), (size_t)15));
                bool buy_agg = unif_(rng_) < 0.5;
                pkt.price           = double_to_price(price);
                pkt.qty             = qty_dist_(rng_) / 5;
                pkt.buyer_order_id  = order_id++;
                pkt.seller_order_id = order_id++;
                pkt.trade_id        = trade_id++;
                pkt.aggressor_side  = buy_agg ? 0 : 1;
                pkt.timestamp_ns    = ts_ns;

                uint8_t* payload = reinterpret_cast<uint8_t*>(&pkt) + sizeof(PacketHeader);
                pkt.hdr.checksum = fletcher16(payload, pkt.hdr.payload_len);

                sent_packets_++;
                if (callback_) callback_(pkt.hdr, &pkt);
            }
        }
    }

    uint64_t sent()    const noexcept { return sent_packets_; }
    uint64_t dropped() const noexcept { return dropped_packets_; }
    double   loss_rate() const noexcept {
        uint64_t total = sent_packets_ + dropped_packets_;
        return total > 0 ? (double)dropped_packets_ / total : 0.0;
    }

private:
    Config              cfg_;
    std::mt19937        rng_;
    LatencyModel        lat_model_;
    std::normal_distribution<double>    norm_;
    std::uniform_real_distribution<double> unif_;
    std::uniform_int_distribution<int>  qty_dist_;
    MessageCallback     callback_;
    uint32_t            seq_ = 1;
    uint64_t            sent_packets_ = 0;
    uint64_t            dropped_packets_ = 0;
};

// ============================================================
// Retransmit Request Manager
// Tracks outstanding retransmit requests and timeouts
// ============================================================
struct RetransmitReq {
    uint32_t begin_seq;
    uint32_t end_seq;
    int64_t  sent_at_ns;
    int      attempts;
};

class RetransmitManager {
public:
    static constexpr int64_t TIMEOUT_NS  = 100'000'000LL; // 100ms
    static constexpr int     MAX_ATTEMPTS = 3;

    void request(uint32_t begin, uint32_t end, int64_t now_ns) {
        pending_.push_back({begin, end, now_ns, 1});
        total_requests_++;
    }

    // Check for timed-out requests and retry
    void tick(int64_t now_ns, std::function<void(uint32_t, uint32_t)> on_retry,
              std::function<void(uint32_t, uint32_t)> on_give_up) {
        auto it = pending_.begin();
        while (it != pending_.end()) {
            if (now_ns - it->sent_at_ns >= TIMEOUT_NS) {
                if (it->attempts >= MAX_ATTEMPTS) {
                    on_give_up(it->begin_seq, it->end_seq);
                    total_give_ups_++;
                    it = pending_.erase(it);
                } else {
                    it->attempts++;
                    it->sent_at_ns = now_ns;
                    on_retry(it->begin_seq, it->end_seq);
                    total_retries_++;
                    ++it;
                }
            } else {
                ++it;
            }
        }
    }

    void on_received(uint32_t seq) {
        pending_.erase(
            std::remove_if(pending_.begin(), pending_.end(),
                [seq](const RetransmitReq& r){ return seq >= r.begin_seq && seq <= r.end_seq; }),
            pending_.end());
    }

    size_t pending_count() const noexcept { return pending_.size(); }
    uint64_t total_requests() const noexcept { return total_requests_; }
    uint64_t total_retries()  const noexcept { return total_retries_; }
    uint64_t total_give_ups() const noexcept { return total_give_ups_; }

private:
    std::vector<RetransmitReq> pending_;
    uint64_t total_requests_ = 0;
    uint64_t total_retries_  = 0;
    uint64_t total_give_ups_ = 0;
};

// ============================================================
// Session manager: heartbeat, login, sequence reset
// ============================================================
class SessionManager {
public:
    enum class State { Disconnected, Connecting, Active, Error };

    struct SessionStats {
        uint64_t messages_received = 0;
        uint64_t heartbeats_sent   = 0;
        uint64_t heartbeats_missed = 0;
        uint64_t reconnects        = 0;
        double   uptime_pct        = 0.0;
        int64_t  last_hb_time_ns   = 0;
    };

    static constexpr int64_t HEARTBEAT_INTERVAL_NS = 1'000'000'000LL; // 1s
    static constexpr int     MAX_MISSED_HB         = 3;

    SessionManager() : state_(State::Disconnected) {}

    void connect(int64_t now_ns) {
        state_ = State::Active;
        connect_time_ns_ = now_ns;
        last_hb_ns_      = now_ns;
        missed_hb_       = 0;
        stats_.reconnects++;
    }

    void on_message(int64_t now_ns) {
        last_msg_ns_ = now_ns;
        stats_.messages_received++;
    }

    // Call periodically to check heartbeat
    void tick(int64_t now_ns) {
        if (state_ != State::Active) return;

        if (now_ns - last_hb_ns_ >= HEARTBEAT_INTERVAL_NS) {
            // Check if we should have received a heartbeat from the peer
            if (now_ns - last_msg_ns_ > HEARTBEAT_INTERVAL_NS * 1.5) {
                missed_hb_++;
                stats_.heartbeats_missed++;
                if (missed_hb_ >= MAX_MISSED_HB) {
                    state_ = State::Error;
                    return;
                }
            }
            // Send our heartbeat
            last_hb_ns_ = now_ns;
            stats_.heartbeats_sent++;
            missed_hb_ = 0;
        }

        int64_t total_ns = now_ns - connect_time_ns_;
        int64_t active_ns = last_msg_ns_ - connect_time_ns_;
        stats_.uptime_pct = total_ns > 0 ? (double)active_ns / total_ns * 100.0 : 0.0;
        stats_.last_hb_time_ns = last_hb_ns_;
    }

    void disconnect() { state_ = State::Disconnected; }
    void reset_error() {
        if (state_ == State::Error) {
            state_ = State::Disconnected;
            missed_hb_ = 0;
        }
    }

    State state() const noexcept { return state_; }
    const SessionStats& stats() const noexcept { return stats_; }
    bool is_active() const noexcept { return state_ == State::Active; }

private:
    State        state_;
    SessionStats stats_;
    int64_t      connect_time_ns_ = 0;
    int64_t      last_hb_ns_      = 0;
    int64_t      last_msg_ns_     = 0;
    int          missed_hb_       = 0;
};

// ============================================================
// Arbitrage opportunity detector (cross-venue)
// Watches prices on two simulated venues, fires on spread
// ============================================================
class CrossVenueArb {
public:
    struct ArbOpportunity {
        std::string symbol;
        double      venue1_ask;
        double      venue2_bid;
        double      spread_bps;
        int64_t     detected_at_ns;
        bool        is_profitable;
    };

    explicit CrossVenueArb(double min_spread_bps = 2.0, double fee_bps = 1.0)
        : min_spread_bps_(min_spread_bps), fee_bps_(fee_bps) {}

    // Update venue prices
    void update(const std::string& sym, int venue, Price bid, Price ask, int64_t ts_ns) {
        auto& state = states_[sym];
        if (venue == 1) { state.v1_bid = bid; state.v1_ask = ask; state.v1_ts = ts_ns; }
        else            { state.v2_bid = bid; state.v2_ask = ask; state.v2_ts = ts_ns; }
        state.last_check = ts_ns;
    }

    // Check for arbitrage opportunities
    std::vector<ArbOpportunity> check_all(int64_t now_ns) {
        std::vector<ArbOpportunity> opps;
        static constexpr int64_t STALE_NS = 100'000'000LL; // 100ms

        for (auto& [sym, s] : states_) {
            if (s.v1_bid == 0 || s.v2_bid == 0) continue;
            if (now_ns - s.v1_ts > STALE_NS || now_ns - s.v2_ts > STALE_NS) continue;

            // Buy on v2 (v2_ask), sell on v1 (v1_bid)
            double v2_ask = price_to_double(s.v2_ask);
            double v1_bid = price_to_double(s.v1_bid);
            if (v2_ask > 0 && v1_bid > v2_ask) {
                double spread_bps = (v1_bid - v2_ask) / v2_ask * 10000.0;
                if (spread_bps > fee_bps_) {
                    total_opportunities_++;
                    opps.push_back({sym, v2_ask, v1_bid, spread_bps, now_ns,
                                   spread_bps >= min_spread_bps_});
                }
            }

            // Buy on v1 (v1_ask), sell on v2 (v2_bid)
            double v1_ask = price_to_double(s.v1_ask);
            double v2_bid = price_to_double(s.v2_bid);
            if (v1_ask > 0 && v2_bid > v1_ask) {
                double spread_bps = (v2_bid - v1_ask) / v1_ask * 10000.0;
                if (spread_bps > fee_bps_) {
                    total_opportunities_++;
                    opps.push_back({sym, v1_ask, v2_bid, spread_bps, now_ns,
                                   spread_bps >= min_spread_bps_});
                }
            }
        }
        return opps;
    }

    uint64_t total_opportunities() const noexcept { return total_opportunities_; }

private:
    struct VenueState {
        Price   v1_bid = 0, v1_ask = 0;
        Price   v2_bid = 0, v2_ask = 0;
        int64_t v1_ts = 0, v2_ts = 0;
        int64_t last_check = 0;
    };

    double                              min_spread_bps_;
    double                              fee_bps_;
    std::unordered_map<std::string, VenueState> states_;
    uint64_t                            total_opportunities_ = 0;
};

// ============================================================
// Throughput meter: measures messages per second
// ============================================================
class ThroughputMeter {
public:
    explicit ThroughputMeter(size_t window_ms = 1000) : window_ms_(window_ms) {}

    void record(int64_t now_ns, uint64_t count = 1) {
        entries_.push_back({now_ns, count});
        total_ += count;
        int64_t cutoff = now_ns - static_cast<int64_t>(window_ms_) * 1000000LL;
        while (!entries_.empty() && entries_.front().ts_ns < cutoff) {
            window_total_ -= entries_.front().count;
            entries_.pop_front();
        }
        window_total_ += count;
    }

    double messages_per_sec(int64_t now_ns) const {
        if (entries_.empty()) return 0.0;
        int64_t window_ns = now_ns - entries_.front().ts_ns;
        return window_ns > 0 ? (double)window_total_ / window_ns * 1e9 : 0.0;
    }

    uint64_t total() const noexcept { return total_; }
    uint64_t window_count() const noexcept { return window_total_; }

private:
    struct Entry { int64_t ts_ns; uint64_t count; };
    std::deque<Entry> entries_;
    size_t   window_ms_;
    uint64_t total_        = 0;
    uint64_t window_total_ = 0;
};

// ============================================================
// Network topology: simulates co-location, cross-connect, WAN
// ============================================================
struct NetworkTopology {
    // Latency tiers (nanoseconds)
    static constexpr double COLOCATION_NS    = 500.0;    // 0.5 µs
    static constexpr double CROSS_CONNECT_NS = 5000.0;   // 5 µs
    static constexpr double METRO_WAN_NS     = 500000.0; // 500 µs
    static constexpr double COAST_WAN_NS     = 40e6;     // 40 ms

    // Bandwidth limits (bytes/sec)
    static constexpr double COLOCATION_BPS    = 10e9;  // 10 Gbps
    static constexpr double CROSS_CONNECT_BPS = 1e9;   // 1 Gbps
    static constexpr double METRO_BPS         = 100e6; // 100 Mbps

    enum class Tier { Colocation, CrossConnect, MetroWAN, CoastWAN };

    static double base_latency_ns(Tier t) {
        switch (t) {
            case Tier::Colocation:    return COLOCATION_NS;
            case Tier::CrossConnect:  return CROSS_CONNECT_NS;
            case Tier::MetroWAN:      return METRO_WAN_NS;
            case Tier::CoastWAN:      return COAST_WAN_NS;
        }
        return 0;
    }

    static double bandwidth_bps(Tier t) {
        switch (t) {
            case Tier::Colocation:    return COLOCATION_BPS;
            case Tier::CrossConnect:  return CROSS_CONNECT_BPS;
            case Tier::MetroWAN:      return METRO_BPS;
            default:                  return 50e6;
        }
    }

    // Serialization latency for a packet
    static double serialization_ns(size_t packet_bytes, Tier t) {
        return packet_bytes * 8.0 / bandwidth_bps(t) * 1e9;
    }

    static double total_latency_ns(size_t packet_bytes, Tier t) {
        return base_latency_ns(t) + serialization_ns(packet_bytes, t);
    }
};

// ============================================================
// ITCH 5.0 partial decoder (simplified, for C++ side)
// Complements the full Zig decoder
// ============================================================
struct ItchHeader {
    uint16_t length;
    uint8_t  msg_type;
};

struct ItchAddOrder {
    static constexpr uint8_t TYPE = 'A';
    uint32_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;        // 6 bytes in real ITCH, stored as u64 here
    uint64_t order_reference;
    char     buy_sell;
    uint32_t shares;
    char     stock[8];
    uint32_t price;            // 4 decimal places
};

struct ItchOrderExecuted {
    static constexpr uint8_t TYPE = 'E';
    uint32_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
    uint64_t order_reference;
    uint32_t executed_shares;
    uint64_t match_number;
};

inline uint32_t itch_read_u32_be(const uint8_t* buf) {
    return (uint32_t(buf[0]) << 24) | (uint32_t(buf[1]) << 16) |
           (uint32_t(buf[2]) <<  8) |  uint32_t(buf[3]);
}

inline uint64_t itch_read_u48_be(const uint8_t* buf) {
    return (uint64_t(buf[0]) << 40) | (uint64_t(buf[1]) << 32) |
           (uint64_t(buf[2]) << 24) | (uint64_t(buf[3]) << 16) |
           (uint64_t(buf[4]) <<  8) |  uint64_t(buf[5]);
}

// Decode add-order from raw ITCH bytes (after the 2-byte length field)
inline bool decode_itch_add_order(const uint8_t* msg, size_t len, ItchAddOrder& out) {
    if (len < 36 || msg[0] != ItchAddOrder::TYPE) return false;
    out.stock_locate    = (uint32_t(msg[1]) << 8) | msg[2];
    out.tracking_number = (uint16_t(msg[3]) << 8) | msg[4];
    out.timestamp       = itch_read_u48_be(msg + 5);
    out.order_reference = (uint64_t(msg[11]) << 56) | (uint64_t(msg[12]) << 48) |
                          (uint64_t(msg[13]) << 40) | (uint64_t(msg[14]) << 32) |
                          (uint64_t(msg[15]) << 24) | (uint64_t(msg[16]) << 16) |
                          (uint64_t(msg[17]) <<  8) |  uint64_t(msg[18]);
    out.buy_sell        = (char)msg[19];
    out.shares          = itch_read_u32_be(msg + 20);
    std::memcpy(out.stock, msg + 24, 8);
    out.price           = itch_read_u32_be(msg + 32);
    return true;
}

} // namespace hft
