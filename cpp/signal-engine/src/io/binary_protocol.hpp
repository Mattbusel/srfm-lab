#pragma once
#include "srfm/types.hpp"
#include <cstdint>
#include <cstring>
#include <string>

namespace srfm {

// ============================================================
// Wire format: 19 bytes per message
//   [timestamp_ns : 8 bytes] [symbol_id : 2 bytes]
//   [signal_type  : 1 byte]  [value     : 8 bytes]
// Total: 19 bytes
// ============================================================

#pragma pack(push, 1)
struct BinaryMessage {
    int64_t  timestamp_ns;   // 8 bytes
    uint16_t symbol_id;      // 2 bytes
    uint8_t  signal_type;    // 1 byte  (SignalType enum)
    double   value;          // 8 bytes
    // Total: 19 bytes
};
#pragma pack(pop)

static_assert(sizeof(BinaryMessage) == 19, "BinaryMessage must be exactly 19 bytes");

/// Multi-signal packet header (for batching multiple signals per bar).
#pragma pack(push, 1)
struct BinaryPacketHeader {
    uint32_t magic;          // 0x53524D46 = "SRMF"
    uint16_t version;        // protocol version (1)
    uint16_t n_messages;     // number of BinaryMessage records following
    int64_t  batch_ts_ns;    // timestamp of batch
    uint32_t checksum;       // XOR checksum of all message bytes
};
#pragma pack(pop)

static_assert(sizeof(BinaryPacketHeader) == 20, "BinaryPacketHeader must be 20 bytes");

inline constexpr uint32_t BINARY_MAGIC   = 0x53524D46u; // "SRMF" little-endian
inline constexpr uint16_t BINARY_VERSION = 1;

// ============================================================
// Encoder: SignalOutput → BinaryMessage array
// ============================================================

class BinaryEncoder {
public:
    /// Encode a SignalOutput into a pre-allocated array of BinaryMessages.
    /// Returns the number of messages written.
    static int encode(const SignalOutput& sig,
                      BinaryMessage* out, int max_messages) noexcept;

    /// Encode a full packet (header + messages) into a byte buffer.
    /// Returns total bytes written, or -1 on overflow.
    static int encode_packet(const SignalOutput* sigs, int n_sigs,
                               uint8_t* out_buf, int buf_size) noexcept;

    /// Compute XOR checksum of messages.
    static uint32_t checksum(const BinaryMessage* msgs, int n) noexcept;
};

// ============================================================
// Decoder: BinaryMessage → SignalOutput reconstruction
// ============================================================

class BinaryDecoder {
public:
    /// Apply a single BinaryMessage to a SignalOutput struct.
    static void apply(const BinaryMessage& msg, SignalOutput& out) noexcept;

    /// Decode a full packet. Returns number of SignalOutput objects populated.
    static int decode_packet(const uint8_t* buf, int buf_size,
                              SignalOutput* out, int max_out) noexcept;

    /// Validate packet header. Returns true if valid.
    static bool validate_header(const BinaryPacketHeader& hdr) noexcept;
};

// ============================================================
// Named pipe / shared memory writer (platform-specific)
// ============================================================

class BinaryPipeWriter {
public:
    explicit BinaryPipeWriter(const std::string& pipe_name) noexcept;
    ~BinaryPipeWriter() noexcept;

    bool open() noexcept;
    void close() noexcept;
    bool write(const BinaryMessage& msg) noexcept;
    bool write_batch(const BinaryMessage* msgs, int n) noexcept;
    bool is_open() const noexcept { return fd_ >= 0; }

private:
    std::string pipe_name_;
    int         fd_;
};

class BinaryPipeReader {
public:
    explicit BinaryPipeReader(const std::string& pipe_name) noexcept;
    ~BinaryPipeReader() noexcept;

    bool open() noexcept;
    void close() noexcept;

    /// Read one message. Returns false if no data available (non-blocking).
    bool read(BinaryMessage& out) noexcept;

    bool is_open() const noexcept { return fd_ >= 0; }

private:
    std::string pipe_name_;
    int         fd_;
};

} // namespace srfm
