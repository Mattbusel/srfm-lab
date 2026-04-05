#include "feed_processor.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <new>

namespace srfm {

// ============================================================
// FeedProcessor
// ============================================================

FeedProcessor::FeedProcessor(int n_instruments, SignalCallback callback) noexcept
    : n_instruments_(std::min(n_instruments, MAX_INSTRUMENTS))
    , callback_(std::move(callback))
    , risk_parity_(n_instruments_)
    , instruments_(nullptr)
    , last_signals_(nullptr)
{
    // Allocate instrument state array
    instruments_ = new(std::nothrow) InstrumentState*[n_instruments_];
    last_signals_= new(std::nothrow) SignalOutput[n_instruments_];

    if (!instruments_ || !last_signals_) {
        // Allocation failure — set to zero count
        n_instruments_ = 0;
        return;
    }

    for (int i = 0; i < n_instruments_; ++i) {
        instruments_[i] = new(std::nothrow) InstrumentState(i);
        std::memset(&last_signals_[i], 0, sizeof(SignalOutput));
        last_signals_[i].symbol_id = i;
    }
}

FeedProcessor::~FeedProcessor() noexcept {
    for (int i = 0; i < n_instruments_; ++i)
        delete instruments_[i];
    delete[] instruments_;
    delete[] last_signals_;
}

void FeedProcessor::fill_signal_output(const OHLCVBar& bar,
                                        InstrumentState& state,
                                        SignalOutput& out) noexcept {
    out.timestamp_ns = bar.timestamp_ns;
    out.symbol_id    = bar.symbol_id;

    // EMAs
    out.ema_fast = state.ema_fast.update(bar.close);
    out.ema_slow = state.ema_slow.update(bar.close);

    // ATR
    out.atr = state.atr.update(bar);

    // RSI
    out.rsi = state.rsi.update(bar.close);

    // Bollinger
    {
        auto bb_out  = state.bb.update(bar.close);
        out.bb_upper    = bb_out.upper;
        out.bb_mid      = bb_out.mid;
        out.bb_lower    = bb_out.lower;
        out.bb_pct_b    = bb_out.pct_b;
        out.bb_bandwidth= bb_out.bandwidth;
    }

    // MACD
    {
        auto macd_out   = state.macd.update(bar.close);
        out.macd_line   = macd_out.macd_line;
        out.macd_signal = macd_out.signal_line;
        out.macd_hist   = macd_out.histogram;
    }

    // VWAP
    out.vwap = state.vwap.update(bar);

    // Realized vol
    {
        auto rv_out         = state.rv.update(bar);
        out.rv_parkinson    = rv_out.parkinson;
        out.rv_garman_klass = rv_out.garman_klass;
        out.rv_rogers_satchell = rv_out.rogers_satchell;
        out.rv_yang_zhang   = rv_out.yang_zhang;
    }

    // BH physics
    {
        auto bh_out  = state.bh.update(bar);
        out.bh_mass  = bh_out.mass;
        out.bh_dir   = bh_out.bh_dir;
        out.cf_scale = bh_out.cf_scale;
        out.bh_active= bh_out.bh_active ? 1 : 0;
    }

    // GARCH
    {
        auto g_out        = state.garch.update(bar.close);
        out.garch_variance= g_out.variance;
        out.garch_vol_scale= g_out.vol_scale;
    }

    // OU
    {
        auto ou_out       = state.ou.update(bar.close);
        out.ou_zscore     = ou_out.zscore;
        out.ou_half_life  = ou_out.half_life;
        out.ou_long_signal = ou_out.long_signal  ? 1 : 0;
        out.ou_short_signal= ou_out.short_signal ? 1 : 0;
    }

    // Update risk parity with current vol estimate
    double vol = out.rv_yang_zhang > constants::EPSILON
                 ? out.rv_yang_zhang
                 : out.garch_vol_scale > constants::EPSILON
                   ? constants::TARGET_VOL / out.garch_vol_scale
                   : 0.15;

    risk_parity_.update_vol(bar.symbol_id, vol);

    // Compute log return for correlation tracking
    // (using GARCH return as proxy — it's already computed)
    // For simplicity, compute from close and previous close stored in BH state
    // We skip the exact return here and use EMA-based approximation
    if (out.ema_fast > constants::EPSILON && out.ema_slow > constants::EPSILON) {
        double log_ret = std::log(out.ema_fast / out.ema_slow) * 0.1; // proxy
        risk_parity_.update_return(bar.symbol_id, log_ret);
    }

    // Risk sizing
    PositionSize ps;
    risk_parity_.compute_positions(&ps, 1);  // just one instrument
    // Note: compute_positions uses index 0, so we pass index = symbol_id
    {
        PositionSize all[MAX_INSTRUMENTS];
        risk_parity_.compute_positions(all, n_instruments_);
        int sid = std::clamp(bar.symbol_id, 0, n_instruments_ - 1);
        out.position_size = all[sid].final_size;
        out.vol_budget    = all[sid].vol_budget;
        out.corr_factor   = all[sid].corr_factor;
    }

    out.bar_count = state.bh.bar_count();
}

SignalOutput FeedProcessor::process_bar(const OHLCVBar& bar) noexcept {
    int sym = bar.symbol_id;
    if (sym < 0 || sym >= n_instruments_) {
        SignalOutput empty{};
        return empty;
    }

    InstrumentState& state = *instruments_[sym];
    SignalOutput&    out   = last_signals_[sym];

    fill_signal_output(bar, state, out);

    if (callback_) callback_(out);

    return out;
}

void FeedProcessor::process_batch(const OHLCVBar* bars, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        process_bar(bars[i]);
    }
}

const SignalOutput& FeedProcessor::last_signal(int symbol_id) const noexcept {
    static SignalOutput empty{};
    if (symbol_id < 0 || symbol_id >= n_instruments_) return empty;
    return last_signals_[symbol_id];
}

InstrumentState& FeedProcessor::instrument(int symbol_id) noexcept {
    return *instruments_[std::clamp(symbol_id, 0, n_instruments_ - 1)];
}

const InstrumentState& FeedProcessor::instrument(int symbol_id) const noexcept {
    return *instruments_[std::clamp(symbol_id, 0, n_instruments_ - 1)];
}

void FeedProcessor::reset() noexcept {
    for (int i = 0; i < n_instruments_; ++i) {
        instruments_[i]->reset();
        std::memset(&last_signals_[i], 0, sizeof(SignalOutput));
        last_signals_[i].symbol_id = i;
    }
    risk_parity_.reset();
}

} // namespace srfm
