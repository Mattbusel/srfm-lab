#pragma once
#include "srfm/types.hpp"
#include <array>
#include <functional>

namespace srfm {

/// Callback type for completed bars.
using BarCallback = std::function<void(const OHLCVBar&)>;

/// Aggregates 1-minute bars into multi-timeframe bars.
/// Supports: 5m, 15m, 1h, 4h, 1d aggregation.
/// Proper timestamp alignment (floor to timeframe boundary).
class BarAggregator {
public:
    /// Construct with a callback invoked when a bar completes.
    explicit BarAggregator(BarCallback callback,
                           int symbol_id = 0) noexcept;

    /// Feed a 1-minute bar. May trigger callbacks for completed multi-TF bars.
    void on_1m_bar(const OHLCVBar& bar) noexcept;

    /// Returns the current in-progress bar for a given timeframe.
    const OHLCVBar& current_bar(TimeFrame tf) const noexcept;

    /// Returns number of completed bars for each timeframe.
    int completed_count(TimeFrame tf) const noexcept;

    void reset() noexcept;

private:
    struct AggState {
        OHLCVBar bar;
        int64_t  period_start_ns;  // start of current aggregation period
        int      tick_count;       // 1m bars in current period
        int      target_ticks;     // 1m bars needed to complete
        bool     active;           // has received first bar
        int      completed;        // total completed bars

        void reset_to_bar(const OHLCVBar& src, int64_t start_ns,
                          int target, int tf_sec) noexcept;
        void update(const OHLCVBar& src) noexcept;
        bool should_complete(int64_t ts_ns) const noexcept;
    };

    BarCallback callback_;
    int         symbol_id_;

    static constexpr int N_TF = 5;
    static constexpr TimeFrame TIMEFRAMES[N_TF] = {
        TimeFrame::M5, TimeFrame::M15, TimeFrame::H1,
        TimeFrame::H4, TimeFrame::D1
    };

    AggState states_[N_TF];

    /// Returns index into states_ for a given TimeFrame.
    static int tf_index(TimeFrame tf) noexcept;

    /// Align a timestamp to the given timeframe boundary.
    static int64_t align_to_tf(int64_t ts_ns, int64_t tf_ns) noexcept;
};

// ============================================================
// Multi-symbol bar aggregator
// ============================================================

class MultiSymbolAggregator {
public:
    MultiSymbolAggregator(int n_symbols, BarCallback callback) noexcept;

    void on_1m_bar(const OHLCVBar& bar) noexcept;

    const BarAggregator& symbol(int id) const noexcept { return *aggs_[id]; }

    void reset() noexcept;

private:
    int n_symbols_;
    BarCallback callback_;
    std::array<BarAggregator*, MAX_INSTRUMENTS> aggs_{};

    // Pool of BarAggregator objects (avoid dynamic allocation)
    std::array<BarAggregator, MAX_INSTRUMENTS> pool_;
};

} // namespace srfm
