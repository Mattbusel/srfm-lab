#include "bar_aggregator.hpp"
#include <cstring>
#include <algorithm>

namespace srfm {

// ============================================================
// AggState methods
// ============================================================

void BarAggregator::AggState::reset_to_bar(const OHLCVBar& src, int64_t start_ns,
                                             int target, int tf_sec) noexcept {
    bar              = src;
    bar.timestamp_ns = start_ns;
    bar.timeframe_sec= tf_sec;
    period_start_ns  = start_ns;
    tick_count       = 1;
    target_ticks     = target;
    active           = true;
}

void BarAggregator::AggState::update(const OHLCVBar& src) noexcept {
    if (src.high > bar.high) bar.high = src.high;
    if (src.low  < bar.low)  bar.low  = src.low;
    bar.close   = src.close;
    bar.volume += src.volume;
    ++tick_count;
}

bool BarAggregator::AggState::should_complete(int64_t ts_ns) const noexcept {
    if (!active) return false;
    int64_t tf_ns = static_cast<int64_t>(bar.timeframe_sec) * constants::NS_PER_SEC;
    return ts_ns >= period_start_ns + tf_ns;
}

// ============================================================
// BarAggregator
// ============================================================

constexpr TimeFrame BarAggregator::TIMEFRAMES[N_TF];

BarAggregator::BarAggregator(BarCallback callback, int symbol_id) noexcept
    : callback_(std::move(callback))
    , symbol_id_(symbol_id)
{
    for (int i = 0; i < N_TF; ++i) {
        std::memset(&states_[i], 0, sizeof(AggState));
        states_[i].target_ticks = static_cast<int>(TIMEFRAMES[i]) / 60;
        states_[i].bar.timeframe_sec = static_cast<int>(TIMEFRAMES[i]);
    }
}

int BarAggregator::tf_index(TimeFrame tf) noexcept {
    for (int i = 0; i < N_TF; ++i)
        if (TIMEFRAMES[i] == tf) return i;
    return 0;
}

int64_t BarAggregator::align_to_tf(int64_t ts_ns, int64_t tf_ns) noexcept {
    return (ts_ns / tf_ns) * tf_ns;
}

void BarAggregator::on_1m_bar(const OHLCVBar& bar) noexcept {
    for (int i = 0; i < N_TF; ++i) {
        AggState& s   = states_[i];
        int       tf_sec = static_cast<int>(TIMEFRAMES[i]);
        int64_t   tf_ns  = static_cast<int64_t>(tf_sec) * constants::NS_PER_SEC;
        int64_t   aligned = align_to_tf(bar.timestamp_ns, tf_ns);

        if (!s.active) {
            s.reset_to_bar(bar, aligned, tf_sec / 60, tf_sec);
        } else if (bar.timestamp_ns >= s.period_start_ns + tf_ns) {
            // Complete the current bar and emit it
            if (callback_) callback_(s.bar);
            ++s.completed;
            // Start new bar
            s.reset_to_bar(bar, aligned, tf_sec / 60, tf_sec);
        } else {
            // Accumulate into current bar
            s.update(bar);
        }
    }
}

const OHLCVBar& BarAggregator::current_bar(TimeFrame tf) const noexcept {
    return states_[tf_index(tf)].bar;
}

int BarAggregator::completed_count(TimeFrame tf) const noexcept {
    return states_[tf_index(tf)].completed;
}

void BarAggregator::reset() noexcept {
    for (int i = 0; i < N_TF; ++i) {
        std::memset(&states_[i], 0, sizeof(AggState));
        states_[i].target_ticks      = static_cast<int>(TIMEFRAMES[i]) / 60;
        states_[i].bar.timeframe_sec = static_cast<int>(TIMEFRAMES[i]);
    }
}

// ============================================================
// MultiSymbolAggregator
// ============================================================

MultiSymbolAggregator::MultiSymbolAggregator(int n_symbols,
                                               BarCallback callback) noexcept
    : n_symbols_(std::min(n_symbols, MAX_INSTRUMENTS))
    , callback_(std::move(callback))
    , pool_{ }  // default-construct all
{
    // Can't pass callback to pool default constructors, so reinitialize
    for (int i = 0; i < n_symbols_; ++i) {
        // Reconstruct in place with proper callback and symbol_id
        pool_[i].~BarAggregator();
        new (&pool_[i]) BarAggregator(callback_, i);
        aggs_[i] = &pool_[i];
    }
}

void MultiSymbolAggregator::on_1m_bar(const OHLCVBar& bar) noexcept {
    int sym = bar.symbol_id;
    if (sym >= 0 && sym < n_symbols_) {
        aggs_[sym]->on_1m_bar(bar);
    }
}

void MultiSymbolAggregator::reset() noexcept {
    for (int i = 0; i < n_symbols_; ++i)
        aggs_[i]->reset();
}

// ============================================================
// Tick to 1m bar converter
// ============================================================

class TickAggregator {
public:
    explicit TickAggregator(int symbol_id = 0) noexcept
        : symbol_id_(symbol_id), has_bar_(false)
    {
        std::memset(&current_, 0, sizeof(current_));
    }

    /// Feed a tick. Returns true and populates completed_bar if a 1m bar closed.
    bool on_tick(const TickData& tick, OHLCVBar& completed_bar) noexcept {
        int64_t bar_start = align_to_minute(tick.timestamp_ns);

        if (!has_bar_) {
            open_bar(tick, bar_start);
            return false;
        }

        if (bar_start > current_.timestamp_ns) {
            // New minute: emit current bar, open new one
            completed_bar = current_;
            open_bar(tick, bar_start);
            return true;
        }

        // Update current bar
        if (tick.price > current_.high) current_.high = tick.price;
        if (tick.price < current_.low)  current_.low  = tick.price;
        current_.close   = tick.price;
        current_.volume += tick.qty;
        return false;
    }

    const OHLCVBar& current() const noexcept { return current_; }

private:
    static int64_t align_to_minute(int64_t ts_ns) noexcept {
        return (ts_ns / constants::NS_PER_MIN) * constants::NS_PER_MIN;
    }

    void open_bar(const TickData& tick, int64_t bar_start_ns) noexcept {
        current_.open         = tick.price;
        current_.high         = tick.price;
        current_.low          = tick.price;
        current_.close        = tick.price;
        current_.volume       = tick.qty;
        current_.timestamp_ns = bar_start_ns;
        current_.symbol_id    = symbol_id_;
        current_.timeframe_sec= 60;
        has_bar_ = true;
    }

    int      symbol_id_;
    bool     has_bar_;
    OHLCVBar current_;
};

} // namespace srfm
