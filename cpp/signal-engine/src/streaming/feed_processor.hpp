#pragma once
#include "srfm/types.hpp"
#include "srfm/ring_buffer.hpp"
#include "../indicators/ema.hpp"
#include "../indicators/atr.hpp"
#include "../indicators/rsi.hpp"
#include "../indicators/bollinger.hpp"
#include "../indicators/macd.hpp"
#include "../indicators/vwap.hpp"
#include "../indicators/realized_vol.hpp"
#include "../bh_physics/bh_state.hpp"
#include "../bh_physics/garch.hpp"
#include "../bh_physics/ou_detector.hpp"
#include "../quaternion/quat_nav.hpp"
#include "../portfolio/risk_parity.hpp"
#include "../portfolio/pid_controller.hpp"
#include <array>
#include <functional>

namespace srfm {

/// All indicator state for one instrument.
struct InstrumentState {
    int      symbol_id;

    // Standard indicators
    EMA      ema_fast;   // 9-period
    EMA      ema_slow;   // 21-period
    ATR      atr;        // 14-period
    RSI      rsi;        // 14-period
    BollingerBands bb;   // 20-period
    MACD     macd;       // 12/26/9
    VWAP     vwap;       // session VWAP

    // Realized vol
    RealizedVolEstimator rv;

    // BH physics
    BHState  bh;

    // Quaternion navigation (one per instrument; persists across bars)
    QuatNav  quat_nav;

    // GARCH
    GARCHTracker garch;

    // OU
    OUDetector ou;

    // Constructor must initialize all members
    explicit InstrumentState(int id) noexcept
        : symbol_id(id)
        , ema_fast(9)
        , ema_slow(21)
        , atr(14)
        , rsi(14)
        , bb(20, 2.0)
        , macd(12, 26, 9)
        , vwap(constants::NS_PER_DAY)
        , rv(20, 252.0)
        , bh()
        , quat_nav()
        , garch()
        , ou(constants::OU_WINDOW)
    {}

    void reset() noexcept {
        ema_fast.reset();
        ema_slow.reset();
        atr.reset();
        rsi.reset();
        bb.reset();
        macd.reset();
        vwap.reset();
        rv.reset();
        bh.reset();
        quat_nav.reset();
        garch.reset();
        ou.reset();
    }
};

using SignalCallback = std::function<void(const SignalOutput&)>;

/// Processes incoming bar stream and updates all indicators.
/// Latency target: <10 microseconds per instrument per bar.
class FeedProcessor {
public:
    FeedProcessor(int n_instruments,
                  SignalCallback callback = nullptr) noexcept;

    ~FeedProcessor() noexcept;

    /// Process a single bar for one instrument. Returns computed SignalOutput.
    SignalOutput process_bar(const OHLCVBar& bar) noexcept;

    /// Process a batch of bars (potentially multi-instrument).
    void process_batch(const OHLCVBar* bars, std::size_t n) noexcept;

    /// Get the last signal output for a given instrument.
    const SignalOutput& last_signal(int symbol_id) const noexcept;

    /// Returns reference to instrument state (for inspection / testing).
    InstrumentState& instrument(int symbol_id) noexcept;
    const InstrumentState& instrument(int symbol_id) const noexcept;

    int n_instruments() const noexcept { return n_instruments_; }

    void reset() noexcept;
    void set_callback(SignalCallback cb) noexcept { callback_ = std::move(cb); }

private:
    int             n_instruments_;
    SignalCallback  callback_;
    RiskParity      risk_parity_;

    // Instrument states (heap-allocated to avoid stack pressure)
    InstrumentState** instruments_;
    SignalOutput*     last_signals_;

    // Preallocated output buffers
    void fill_signal_output(const OHLCVBar& bar,
                             InstrumentState& state,
                             SignalOutput& out) noexcept;
};

} // namespace srfm
