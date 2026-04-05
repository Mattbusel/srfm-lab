#include "binary_protocol.hpp"
#include <cstring>
#include <algorithm>
#include <cstdio>

#if defined(SRFM_HAVE_MMAP)
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#endif

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#endif

namespace srfm {

// ============================================================
// BinaryEncoder
// ============================================================

int BinaryEncoder::encode(const SignalOutput& sig,
                           BinaryMessage* out, int max_messages) noexcept {
    int n = 0;
    auto emit = [&](SignalType type, double value) {
        if (n >= max_messages) return;
        out[n].timestamp_ns = sig.timestamp_ns;
        out[n].symbol_id    = static_cast<uint16_t>(sig.symbol_id);
        out[n].signal_type  = static_cast<uint8_t>(type);
        out[n].value        = value;
        ++n;
    };

    emit(SignalType::BH_MASS,   sig.bh_mass);
    emit(SignalType::BH_DIR,    sig.bh_dir);
    emit(SignalType::EMA_CROSS, sig.ema_fast - sig.ema_slow);
    emit(SignalType::RSI,       sig.rsi);
    emit(SignalType::MACD,      sig.macd_hist);
    emit(SignalType::BB_PCT_B,  sig.bb_pct_b);
    emit(SignalType::GARCH_VOL, sig.garch_vol_scale);
    emit(SignalType::OU_ZSCORE, sig.ou_zscore);
    emit(SignalType::ATR,       sig.atr);
    emit(SignalType::VWAP_DEV,  sig.vwap > 0.0
                                ? (sig.ema_fast - sig.vwap) / sig.vwap
                                : 0.0);
    emit(SignalType::POSITION_SZ, sig.position_size);
    return n;
}

uint32_t BinaryEncoder::checksum(const BinaryMessage* msgs, int n) noexcept {
    uint32_t xor_sum = 0;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(msgs);
    std::size_t total = static_cast<std::size_t>(n) * sizeof(BinaryMessage);
    for (std::size_t i = 0; i < total; ++i)
        xor_sum ^= static_cast<uint32_t>(bytes[i]) << ((i % 4) * 8);
    return xor_sum;
}

int BinaryEncoder::encode_packet(const SignalOutput* sigs, int n_sigs,
                                  uint8_t* out_buf, int buf_size) noexcept {
    static constexpr int MAX_MSGS_PER_SIG = 16;
    BinaryMessage msgs[MAX_INSTRUMENTS * MAX_MSGS_PER_SIG];
    int total_msgs = 0;

    for (int i = 0; i < n_sigs && total_msgs + MAX_MSGS_PER_SIG <= static_cast<int>(
             sizeof(msgs) / sizeof(BinaryMessage)); ++i) {
        total_msgs += encode(sigs[i], msgs + total_msgs,
                             MAX_MSGS_PER_SIG);
    }

    int required = static_cast<int>(sizeof(BinaryPacketHeader)) +
                   total_msgs * static_cast<int>(sizeof(BinaryMessage));
    if (required > buf_size) return -1;

    BinaryPacketHeader hdr{};
    hdr.magic       = BINARY_MAGIC;
    hdr.version     = BINARY_VERSION;
    hdr.n_messages  = static_cast<uint16_t>(total_msgs);
    hdr.batch_ts_ns = n_sigs > 0 ? sigs[0].timestamp_ns : 0;
    hdr.checksum    = checksum(msgs, total_msgs);

    std::memcpy(out_buf, &hdr, sizeof(hdr));
    std::memcpy(out_buf + sizeof(hdr), msgs,
                static_cast<std::size_t>(total_msgs) * sizeof(BinaryMessage));
    return required;
}

// ============================================================
// BinaryDecoder
// ============================================================

void BinaryDecoder::apply(const BinaryMessage& msg, SignalOutput& out) noexcept {
    out.timestamp_ns = msg.timestamp_ns;
    out.symbol_id    = static_cast<int32_t>(msg.symbol_id);

    switch (static_cast<SignalType>(msg.signal_type)) {
    case SignalType::BH_MASS:    out.bh_mass      = msg.value; break;
    case SignalType::BH_DIR:     out.bh_dir       = msg.value; break;
    case SignalType::EMA_CROSS:  break;  // derived, not stored directly
    case SignalType::RSI:        out.rsi          = msg.value; break;
    case SignalType::MACD:       out.macd_hist    = msg.value; break;
    case SignalType::BB_PCT_B:   out.bb_pct_b     = msg.value; break;
    case SignalType::GARCH_VOL:  out.garch_vol_scale = msg.value; break;
    case SignalType::OU_ZSCORE:  out.ou_zscore    = msg.value; break;
    case SignalType::ATR:        out.atr          = msg.value; break;
    case SignalType::VWAP_DEV:   break;  // not stored in SignalOutput directly
    case SignalType::POSITION_SZ:out.position_size= msg.value; break;
    }
}

bool BinaryDecoder::validate_header(const BinaryPacketHeader& hdr) noexcept {
    return hdr.magic   == BINARY_MAGIC &&
           hdr.version == BINARY_VERSION &&
           hdr.n_messages <= 10000u;
}

int BinaryDecoder::decode_packet(const uint8_t* buf, int buf_size,
                                  SignalOutput* out, int max_out) noexcept {
    if (buf_size < static_cast<int>(sizeof(BinaryPacketHeader))) return -1;

    BinaryPacketHeader hdr;
    std::memcpy(&hdr, buf, sizeof(hdr));
    if (!validate_header(hdr)) return -1;

    int msg_bytes = static_cast<int>(hdr.n_messages) *
                    static_cast<int>(sizeof(BinaryMessage));
    if (buf_size < static_cast<int>(sizeof(hdr)) + msg_bytes) return -1;

    const BinaryMessage* msgs = reinterpret_cast<const BinaryMessage*>(
        buf + sizeof(hdr));

    // Verify checksum
    if (checksum(msgs, hdr.n_messages) != hdr.checksum) return -1;

    // Apply messages to output structs, keyed by symbol_id
    int max_sym = 0;
    for (int i = 0; i < static_cast<int>(hdr.n_messages); ++i)
        max_sym = std::max(max_sym, static_cast<int>(msgs[i].symbol_id));

    int n_out = std::min(max_sym + 1, max_out);
    for (int i = 0; i < n_out; ++i) std::memset(&out[i], 0, sizeof(SignalOutput));

    for (int i = 0; i < static_cast<int>(hdr.n_messages); ++i) {
        int sym = static_cast<int>(msgs[i].symbol_id);
        if (sym < max_out) apply(msgs[i], out[sym]);
    }
    return n_out;
}

// ============================================================
// Named pipe I/O
// ============================================================

BinaryPipeWriter::BinaryPipeWriter(const std::string& pipe_name) noexcept
    : pipe_name_(pipe_name)
    , fd_(-1)
{}

BinaryPipeWriter::~BinaryPipeWriter() noexcept { close(); }

#if defined(SRFM_HAVE_MMAP)
bool BinaryPipeWriter::open() noexcept {
    // Create FIFO if it doesn't exist
    ::mkfifo(pipe_name_.c_str(), 0644);
    // Open non-blocking to avoid hanging if no reader
    fd_ = ::open(pipe_name_.c_str(), O_WRONLY | O_NONBLOCK);
    return fd_ >= 0;
}

void BinaryPipeWriter::close() noexcept {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

bool BinaryPipeWriter::write(const BinaryMessage& msg) noexcept {
    if (fd_ < 0) return false;
    return ::write(fd_, &msg, sizeof(msg)) == static_cast<ssize_t>(sizeof(msg));
}

bool BinaryPipeWriter::write_batch(const BinaryMessage* msgs, int n) noexcept {
    if (fd_ < 0) return false;
    std::size_t sz = static_cast<std::size_t>(n) * sizeof(BinaryMessage);
    return ::write(fd_, msgs, sz) == static_cast<ssize_t>(sz);
}
#else
// Windows / fallback: use file-based simulation
bool BinaryPipeWriter::open() noexcept {
    fd_ = 1;  // stdout sentinel
    FILE* f = std::fopen(pipe_name_.c_str(), "wb");
    if (!f) return false;
    std::fclose(f);
    return true;
}
void BinaryPipeWriter::close() noexcept { fd_ = -1; }
bool BinaryPipeWriter::write(const BinaryMessage& msg) noexcept {
    if (fd_ < 0) return false;
    FILE* f = std::fopen(pipe_name_.c_str(), "ab");
    if (!f) return false;
    bool ok = std::fwrite(&msg, sizeof(msg), 1, f) == 1;
    std::fclose(f);
    return ok;
}
bool BinaryPipeWriter::write_batch(const BinaryMessage* msgs, int n) noexcept {
    if (fd_ < 0) return false;
    FILE* f = std::fopen(pipe_name_.c_str(), "ab");
    if (!f) return false;
    bool ok = std::fwrite(msgs, sizeof(BinaryMessage), static_cast<std::size_t>(n), f)
              == static_cast<std::size_t>(n);
    std::fclose(f);
    return ok;
}
#endif

// BinaryPipeReader

BinaryPipeReader::BinaryPipeReader(const std::string& pipe_name) noexcept
    : pipe_name_(pipe_name)
    , fd_(-1)
{}

BinaryPipeReader::~BinaryPipeReader() noexcept { close(); }

#if defined(SRFM_HAVE_MMAP)
bool BinaryPipeReader::open() noexcept {
    fd_ = ::open(pipe_name_.c_str(), O_RDONLY | O_NONBLOCK);
    return fd_ >= 0;
}
void BinaryPipeReader::close() noexcept {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}
bool BinaryPipeReader::read(BinaryMessage& out) noexcept {
    if (fd_ < 0) return false;
    return ::read(fd_, &out, sizeof(out)) == static_cast<ssize_t>(sizeof(out));
}
#else
bool BinaryPipeReader::open() noexcept { fd_ = 0; return true; }
void BinaryPipeReader::close() noexcept { fd_ = -1; }
bool BinaryPipeReader::read(BinaryMessage& out) noexcept {
    if (fd_ < 0) return false;
    FILE* f = std::fopen(pipe_name_.c_str(), "rb");
    if (!f) return false;
    bool ok = std::fread(&out, sizeof(out), 1, f) == 1;
    std::fclose(f);
    return ok;
}
#endif

} // namespace srfm
