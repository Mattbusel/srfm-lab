#pragma once
#include <cstdint>
#include <cstring>
#include <array>
#include <string_view>

// Cache line size for alignment
#define SRFM_CACHE_LINE 64

namespace srfm {

// ============================================================
// Core bar / tick data types
// ============================================================

/// OHLCV bar with nanosecond timestamp. Aligned to cache line.
struct alignas(SRFM_CACHE_LINE) OHLCVBar {
    double   open;
    double   high;
    double   low;
    double   close;
    double   volume;
    int64_t  timestamp_ns;   // Unix epoch nanoseconds
    int32_t  symbol_id;      // instrument index (0-based)
    int32_t  timeframe_sec;  // bar duration in seconds (60 = 1m, 300 = 5m, ...)
    // padding to 64 bytes: 5*8 + 8 + 4 + 4 = 56 bytes → pad 8
    uint8_t  _pad[8];

    OHLCVBar() = default;
    OHLCVBar(double o, double h, double l, double c, double v,
             int64_t ts, int32_t sym = 0, int32_t tf = 60)
        : open(o), high(h), low(l), close(c), volume(v),
          timestamp_ns(ts), symbol_id(sym), timeframe_sec(tf) {
        std::memset(_pad, 0, sizeof(_pad));
    }
};
static_assert(sizeof(OHLCVBar) == 64, "OHLCVBar must be 64 bytes");

/// Raw tick / trade data
struct alignas(32) TickData {
    int64_t  timestamp_ns;
    double   price;
    double   qty;
    int32_t  symbol_id;
    int8_t   side;   // 1=buy, -1=sell, 0=unknown
    uint8_t  _pad[3];
};
static_assert(sizeof(TickData) == 32, "TickData must be 32 bytes");

// ============================================================
// Signal output
// ============================================================

/// All signals for a single instrument at a single bar timestamp.
/// Packed to avoid unnecessary padding waste; aligned to 64 bytes.
struct alignas(SRFM_CACHE_LINE) SignalOutput {
    // Timestamp & identity
    int64_t  timestamp_ns;
    int32_t  symbol_id;
    int32_t  bar_count;

    // BH physics signals
    double   bh_mass;
    double   bh_dir;         // directional component
    double   cf_scale;       // BullScale ratio
    int8_t   bh_active;      // 1 = BH formed
    uint8_t  bh_pad[7];

    // Standard indicators
    double   ema_fast;
    double   ema_slow;
    double   atr;
    double   rsi;

    // Bollinger Bands
    double   bb_upper;
    double   bb_mid;
    double   bb_lower;
    double   bb_pct_b;
    double   bb_bandwidth;

    // MACD
    double   macd_line;
    double   macd_signal;
    double   macd_hist;

    // VWAP
    double   vwap;

    // Realized vol estimators
    double   rv_parkinson;
    double   rv_garman_klass;
    double   rv_rogers_satchell;
    double   rv_yang_zhang;

    // GARCH
    double   garch_variance;
    double   garch_vol_scale;

    // OU detector
    double   ou_zscore;
    double   ou_half_life;
    int8_t   ou_long_signal;
    int8_t   ou_short_signal;
    uint8_t  ou_pad[6];

    // Risk / sizing
    double   position_size;
    double   vol_budget;
    double   corr_factor;

    // Quaternion navigation signals (read-only observability; not wired into
    // entry/exit logic yet).
    // Q_current: running orientation quaternion in SRFM 4-space.
    double   nav_qw;
    double   nav_qx;
    double   nav_qy;
    double   nav_qz;
    // angular_vel: d(Q_current)/dt in radians/bar.
    //   High => regime rotation in progress; low => stable heading.
    double   nav_angular_vel;
    // geodesic_dev: SLERP-extrapolation deviation angle (radians),
    //   curvature-corrected by BH mass.  Measures how far the market
    //   deviated from its inertial path this bar.
    double   nav_geodesic_dev;

    // Padding: previous _fill[8] absorbed by nav fields; add 24 bytes to
    // keep the struct on a clean 64-byte multiple (248 + 48 + 24 = 320 = 5*64).
    uint8_t  _fill[24];

    SignalOutput() { std::memset(this, 0, sizeof(*this)); }
};

// We don't enforce exact size here; just ensure alignment
static_assert(alignof(SignalOutput) == SRFM_CACHE_LINE, "SignalOutput alignment");

// ============================================================
// Enumerations
// ============================================================

enum class TimeFrame : int32_t {
    M1   = 60,
    M5   = 300,
    M15  = 900,
    H1   = 3600,
    H4   = 14400,
    D1   = 86400,
};

enum class SignalType : uint8_t {
    BH_MASS      = 0x01,
    BH_DIR       = 0x02,
    EMA_CROSS    = 0x03,
    RSI          = 0x04,
    MACD         = 0x05,
    BB_PCT_B     = 0x06,
    GARCH_VOL    = 0x07,
    OU_ZSCORE    = 0x08,
    ATR          = 0x09,
    VWAP_DEV     = 0x0A,
    POSITION_SZ  = 0x0B,
};

// ============================================================
// Constants
// ============================================================

namespace constants {
    inline constexpr double SPEED_OF_LIGHT   = 299792458.0;  // m/s (used as c in BH metric)
    inline constexpr double BH_MASS_THRESH   = 1.92;
    inline constexpr double BH_EMA_DECAY     = 0.924;
    inline constexpr double BH_COLLAPSE_RATE = 0.992;
    inline constexpr double GARCH_OMEGA      = 0.000001;
    inline constexpr double GARCH_ALPHA      = 0.1;
    inline constexpr double GARCH_BETA       = 0.85;
    inline constexpr double TARGET_VOL       = 0.15;   // 15% annualized
    inline constexpr double PER_INST_RISK    = 0.02;   // 2% per instrument
    inline constexpr double OU_ZSCORE_LONG   = -1.5;
    inline constexpr double OU_ZSCORE_SHORT  =  1.5;
    inline constexpr int    OU_WINDOW        = 50;
    inline constexpr int    CORR_WINDOW      = 30 * 1440;  // 30 days of 1m bars
    inline constexpr double EPSILON          = 1e-10;
    inline constexpr int64_t NS_PER_SEC      = 1'000'000'000LL;
    inline constexpr int64_t NS_PER_MIN      = 60LL * NS_PER_SEC;
    inline constexpr int64_t NS_PER_HOUR     = 3600LL * NS_PER_SEC;
    inline constexpr int64_t NS_PER_DAY      = 86400LL * NS_PER_SEC;
}

// ============================================================
// Utility: symbol name table (simple static array)
// ============================================================

inline constexpr int MAX_INSTRUMENTS = 25;

struct SymbolTable {
    std::array<char[16], MAX_INSTRUMENTS> names{};
    int count = 0;

    void add(std::string_view name) {
        if (count >= MAX_INSTRUMENTS) return;
        auto len = std::min(name.size(), size_t(15));
        std::memcpy(names[count], name.data(), len);
        names[count][len] = '\0';
        ++count;
    }
    std::string_view get(int idx) const {
        if (idx < 0 || idx >= count) return "UNKNOWN";
        return names[idx];
    }
};

} // namespace srfm
