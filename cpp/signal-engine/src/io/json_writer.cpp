#include "json_writer.hpp"
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cassert>

namespace srfm {

// ============================================================
// Fast number formatting utilities
// ============================================================

static const char* DIGIT_TABLE =
    "00010203040506070809"
    "10111213141516171819"
    "20212223242526272829"
    "30313233343536373839"
    "40414243444546474849"
    "50515253545556575859"
    "60616263646566676869"
    "70717273747576777879"
    "80818283848586878889"
    "90919293949596979899";

char* fmt_int64(char* buf, int64_t val) noexcept {
    if (val == 0) { *buf++ = '0'; return buf; }

    char tmp[24];
    int  len = 0;
    bool neg = (val < 0);
    if (neg) { *buf++ = '-'; val = -val; }

    uint64_t u = static_cast<uint64_t>(val);
    while (u >= 100) {
        uint64_t q = u / 100;
        uint32_t r = static_cast<uint32_t>(u % 100) * 2;
        tmp[len++] = DIGIT_TABLE[r + 1];
        tmp[len++] = DIGIT_TABLE[r];
        u = q;
    }
    if (u >= 10) {
        uint32_t r = static_cast<uint32_t>(u) * 2;
        tmp[len++] = DIGIT_TABLE[r + 1];
        tmp[len++] = DIGIT_TABLE[r];
    } else {
        tmp[len++] = '0' + static_cast<char>(u);
    }

    for (int i = len - 1; i >= 0; --i) *buf++ = tmp[i];
    return buf;
}

char* fmt_double(char* buf, double val) noexcept {
    if (std::isnan(val))  { std::memcpy(buf, "null", 4); return buf + 4; }
    if (std::isinf(val))  {
        if (val > 0) { std::memcpy(buf, "1e308",  5); return buf + 5; }
        else         { std::memcpy(buf, "-1e308", 6); return buf + 6; }
    }
    if (val == 0.0)       { *buf++ = '0'; *buf++ = '.'; *buf++ = '0'; return buf; }

    bool neg = (val < 0.0);
    if (neg) { *buf++ = '-'; val = -val; }

    // Scale to 6 significant digits
    // Use snprintf-style but hand-rolled for speed
    // For financial data (prices ~0.001 to 1,000,000), use fixed-point notation
    if (val >= 0.0001 && val < 10000000.0) {
        // Fixed notation: determine decimal places needed
        int64_t int_part  = static_cast<int64_t>(val);
        double  frac_part = val - static_cast<double>(int_part);

        buf = fmt_int64(buf, int_part);
        *buf++ = '.';

        // Write 6 decimal places
        for (int d = 0; d < 6; ++d) {
            frac_part *= 10.0;
            int digit  = static_cast<int>(frac_part);
            *buf++     = '0' + digit;
            frac_part -= digit;
        }
        // Trim trailing zeros
        while (buf[-1] == '0' && buf[-2] != '.') --buf;
        return buf;
    } else {
        // Exponential notation via sprintf fallback
        int n = std::snprintf(buf, 32, "%.6g", val);
        return buf + n;
    }
}

// ============================================================
// JSONWriter helpers
// ============================================================

char* JSONWriter::write_double(char* p, double val) noexcept {
    return fmt_double(p, val);
}

char* JSONWriter::write_int64(char* p, int64_t val) noexcept {
    return fmt_int64(p, val);
}

char* JSONWriter::write_key_double(char* p, const char* key, double val) noexcept {
    *p++ = '"';
    while (*key) *p++ = *key++;
    *p++ = '"'; *p++ = ':';
    p = fmt_double(p, val);
    *p++ = ',';
    return p;
}

char* JSONWriter::write_key_int64(char* p, const char* key, int64_t val) noexcept {
    *p++ = '"';
    while (*key) *p++ = *key++;
    *p++ = '"'; *p++ = ':';
    p = fmt_int64(p, val);
    *p++ = ',';
    return p;
}

char* JSONWriter::write_key_int(char* p, const char* key, int val) noexcept {
    return write_key_int64(p, key, static_cast<int64_t>(val));
}

char* JSONWriter::write_key_bool(char* p, const char* key, bool val) noexcept {
    *p++ = '"';
    while (*key) *p++ = *key++;
    *p++ = '"'; *p++ = ':';
    if (val) { std::memcpy(p, "true",  4); p += 4; }
    else     { std::memcpy(p, "false", 5); p += 5; }
    *p++ = ',';
    return p;
}

// ============================================================
// Serialization
// ============================================================

int JSONWriter::serialize(const SignalOutput& sig, char* buf, int buf_size) noexcept {
    char*       p   = buf;
    const char* end = buf + buf_size - 4;  // leave room for trailing "}\n\0"

    *p++ = '{';

    // Identity
    p = write_key_int64(p, "ts",     sig.timestamp_ns);
    p = write_key_int(p,   "sym",    sig.symbol_id);
    p = write_key_int(p,   "bars",   sig.bar_count);

    // BH physics
    p = write_key_double(p, "bh_mass",   sig.bh_mass);
    p = write_key_double(p, "bh_dir",    sig.bh_dir);
    p = write_key_double(p, "cf_scale",  sig.cf_scale);
    p = write_key_bool(p,   "bh_active", sig.bh_active != 0);

    // Standard indicators
    p = write_key_double(p, "ema_fast",  sig.ema_fast);
    p = write_key_double(p, "ema_slow",  sig.ema_slow);
    p = write_key_double(p, "atr",       sig.atr);
    p = write_key_double(p, "rsi",       sig.rsi);

    // Bollinger
    p = write_key_double(p, "bb_upper",  sig.bb_upper);
    p = write_key_double(p, "bb_mid",    sig.bb_mid);
    p = write_key_double(p, "bb_lower",  sig.bb_lower);
    p = write_key_double(p, "bb_pct_b",  sig.bb_pct_b);
    p = write_key_double(p, "bb_bw",     sig.bb_bandwidth);

    // MACD
    p = write_key_double(p, "macd",      sig.macd_line);
    p = write_key_double(p, "macd_sig",  sig.macd_signal);
    p = write_key_double(p, "macd_hist", sig.macd_hist);

    // VWAP
    p = write_key_double(p, "vwap",      sig.vwap);

    // Realized vol
    p = write_key_double(p, "rv_park",   sig.rv_parkinson);
    p = write_key_double(p, "rv_gk",     sig.rv_garman_klass);
    p = write_key_double(p, "rv_rs",     sig.rv_rogers_satchell);
    p = write_key_double(p, "rv_yz",     sig.rv_yang_zhang);

    // GARCH
    p = write_key_double(p, "garch_var", sig.garch_variance);
    p = write_key_double(p, "garch_vsc", sig.garch_vol_scale);

    // OU
    p = write_key_double(p, "ou_z",      sig.ou_zscore);
    p = write_key_double(p, "ou_hl",     sig.ou_half_life);
    p = write_key_bool(p,   "ou_long",   sig.ou_long_signal  != 0);
    p = write_key_bool(p,   "ou_short",  sig.ou_short_signal != 0);

    // Risk
    p = write_key_double(p, "pos_sz",    sig.position_size);
    p = write_key_double(p, "vol_bgt",   sig.vol_budget);
    p = write_key_double(p, "corr_f",    sig.corr_factor);

    // Remove trailing comma, close object
    if (p > buf && p[-1] == ',') --p;
    *p++ = '}';
    *p++ = '\n';
    *p   = '\0';

    (void)end;  // end is used as safety guide; we assume buf_size >= 2048
    return static_cast<int>(p - buf);
}

std::string JSONWriter::to_string(const SignalOutput& sig) noexcept {
    char buf[2048];
    int n = serialize(sig, buf, sizeof(buf));
    return std::string(buf, n);
}

// ============================================================
// File I/O
// ============================================================

JSONWriter::JSONWriter(std::string filepath) noexcept
    : file_(nullptr)
    , records_(0)
{
    if (!filepath.empty()) open(filepath);
}

JSONWriter::~JSONWriter() noexcept {
    close();
}

bool JSONWriter::open(const std::string& filepath) noexcept {
    close();
    file_ = std::fopen(filepath.c_str(), "wb");
    return file_ != nullptr;
}

void JSONWriter::close() noexcept {
    if (file_) { std::fflush(file_); std::fclose(file_); file_ = nullptr; }
}

bool JSONWriter::write(const SignalOutput& sig) noexcept {
    if (!file_) return false;
    char buf[2048];
    int n = serialize(sig, buf, sizeof(buf));
    bool ok = std::fwrite(buf, 1, static_cast<std::size_t>(n), file_) ==
              static_cast<std::size_t>(n);
    if (ok) ++records_;
    return ok;
}

bool JSONWriter::write_array(const SignalOutput* sigs, std::size_t n) noexcept {
    if (!file_) return false;
    std::fputc('[', file_);
    std::fputc('\n', file_);
    for (std::size_t i = 0; i < n; ++i) {
        char buf[2048];
        int len = serialize(sigs[i], buf, sizeof(buf));
        // Replace trailing '\n' with ',' or nothing
        if (len > 1 && buf[len - 2] == '}') {
            buf[len - 1] = (i + 1 < n) ? ',' : ' ';
            buf[len]     = '\n';
            buf[len + 1] = '\0';
            len += 1;
        }
        std::fwrite(buf, 1, static_cast<std::size_t>(len), file_);
    }
    std::fputc(']', file_);
    std::fputc('\n', file_);
    records_ += n;
    return true;
}

} // namespace srfm
