#include "tick_store.cpp"
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <limits>

// LZ4-like lightweight compressor (self-contained, no external dep)
// Uses a simplified hash chain match finder with 4-byte literals.
// Not compatible with standard LZ4 format; optimized for tick data patterns.

namespace tickstore {
namespace compress {

// ============================================================
// Delta encoding for prices
// Stores: first absolute price, then differences
// Prices are int64_t fixed-point. Deltas typically fit in int16_t.
// ============================================================
std::vector<uint8_t> delta_encode_prices(const int64_t* prices, size_t n) {
    if (n == 0) return {};
    std::vector<uint8_t> out;
    out.reserve(n * 3); // estimate: ~3 bytes per delta on avg

    // Store n as 4 bytes
    uint32_t nn = static_cast<uint32_t>(n);
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&nn),
                          reinterpret_cast<uint8_t*>(&nn) + 4);

    // Store first price as full int64
    out.insert(out.end(), reinterpret_cast<const uint8_t*>(&prices[0]),
                          reinterpret_cast<const uint8_t*>(&prices[0]) + 8);

    // Store deltas with variable-length encoding
    // Delta fits in:
    //   [-63, 63]      → 1 byte  (sign bit | 6-bit magnitude, flag=0)
    //   [-16383, 16383]→ 2 bytes (flag=01 | 14-bit)
    //   [-2097151,...]  → 3 bytes (flag=10 | 22-bit)
    //   else           → 1-byte flag=11 + 8 bytes raw

    for (size_t i = 1; i < n; ++i) {
        int64_t delta = prices[i] - prices[i-1];
        if (delta >= -63 && delta <= 63) {
            // 1 byte: [0][sign][6-bit abs]
            uint8_t v = (delta < 0) ? (0x40 | (uint8_t)(-delta)) : (uint8_t)(delta);
            out.push_back(v);
        } else if (delta >= -16383 && delta <= 16383) {
            // 2 bytes: [01][sign][6-bit hi] [8-bit lo]
            int64_t ad = std::abs(delta);
            uint8_t hi = 0x40 | ((delta < 0) ? 0x20 : 0) | (uint8_t)((ad >> 8) & 0x3F);
            uint8_t lo = (uint8_t)(ad & 0xFF);
            out.push_back(hi); out.push_back(lo);
        } else if (delta >= -(1<<22) && delta < (1<<22)) {
            // 3 bytes: [10][sign][6-bit hi] [8-bit mid] [8-bit lo]
            int64_t ad = std::abs(delta);
            uint8_t b0 = 0x80 | ((delta < 0) ? 0x20 : 0) | (uint8_t)((ad >> 16) & 0x3F);
            uint8_t b1 = (uint8_t)((ad >> 8) & 0xFF);
            uint8_t b2 = (uint8_t)(ad & 0xFF);
            out.push_back(b0); out.push_back(b1); out.push_back(b2);
        } else {
            // 9 bytes: [11000000] + 8-byte raw int64
            out.push_back(0xC0);
            out.insert(out.end(), reinterpret_cast<uint8_t*>(&delta),
                                  reinterpret_cast<uint8_t*>(&delta) + 8);
        }
    }
    return out;
}

std::vector<int64_t> delta_decode_prices(const uint8_t* data, size_t data_len) {
    if (data_len < 12) return {};
    const uint8_t* p = data;
    const uint8_t* end = data + data_len;

    uint32_t n;
    std::memcpy(&n, p, 4); p += 4;

    int64_t first;
    std::memcpy(&first, p, 8); p += 8;

    std::vector<int64_t> prices(n);
    prices[0] = first;

    for (size_t i = 1; i < n && p < end; ++i) {
        uint8_t flag = (*p >> 6) & 0x3;
        if (flag == 0) {
            // 1 byte
            bool neg = (*p & 0x40) != 0;
            int64_t mag = *p & 0x3F; p++;
            prices[i] = prices[i-1] + (neg ? -mag : mag);
        } else if (flag == 1) {
            // 2 bytes
            bool neg = (*p & 0x20) != 0;
            int64_t hi = *p & 0x1F; p++;
            int64_t lo = *p; p++;
            int64_t mag = (hi << 8) | lo;
            prices[i] = prices[i-1] + (neg ? -mag : mag);
        } else if (flag == 2) {
            // 3 bytes
            bool neg = (*p & 0x20) != 0;
            int64_t b0 = *p & 0x1F; p++;
            int64_t b1 = *p; p++;
            int64_t b2 = *p; p++;
            int64_t mag = (b0 << 16) | (b1 << 8) | b2;
            prices[i] = prices[i-1] + (neg ? -mag : mag);
        } else {
            // 9 bytes
            p++; // skip flag byte
            int64_t delta;
            std::memcpy(&delta, p, 8); p += 8;
            prices[i] = prices[i-1] + delta;
        }
    }
    return prices;
}

// ============================================================
// Run-length encoding for timestamps
// Timestamps increase monotonically; gaps (intervals) tend to be similar
// Store: base timestamp + run-length encoded intervals
// ============================================================
struct RLETimestamps {
    uint64_t base_ts;
    std::vector<uint8_t> encoded;
};

RLETimestamps rle_encode_timestamps(const uint64_t* ts, size_t n) {
    if (n == 0) return {0, {}};

    RLETimestamps rle;
    rle.base_ts = ts[0];
    rle.encoded.reserve(n * 2);

    // Compute deltas (nanoseconds between ticks)
    std::vector<uint64_t> deltas(n);
    deltas[0] = 0;
    for (size_t i = 1; i < n; ++i)
        deltas[i] = ts[i] - ts[i-1];

    // Store n
    uint32_t nn = static_cast<uint32_t>(n);
    rle.encoded.insert(rle.encoded.end(),
                       reinterpret_cast<uint8_t*>(&nn),
                       reinterpret_cast<uint8_t*>(&nn) + 4);

    // RLE: for each delta, check if same as previous
    // Format: [count_byte][8-byte delta] or [0][count_byte][8-byte_delta] for count>127
    size_t i = 0;
    while (i < n) {
        uint64_t cur_delta = deltas[i];
        size_t run = 1;
        while (i + run < n && deltas[i + run] == cur_delta && run < 32767) ++run;

        if (run < 128) {
            rle.encoded.push_back(static_cast<uint8_t>(run));
        } else {
            rle.encoded.push_back(0);
            uint16_t r16 = static_cast<uint16_t>(run);
            rle.encoded.insert(rle.encoded.end(),
                               reinterpret_cast<uint8_t*>(&r16),
                               reinterpret_cast<uint8_t*>(&r16) + 2);
        }
        rle.encoded.insert(rle.encoded.end(),
                           reinterpret_cast<const uint8_t*>(&cur_delta),
                           reinterpret_cast<const uint8_t*>(&cur_delta) + 8);
        i += run;
    }
    return rle;
}

std::vector<uint64_t> rle_decode_timestamps(const RLETimestamps& rle) {
    const uint8_t* p   = rle.encoded.data();
    const uint8_t* end = p + rle.encoded.size();
    if (p + 4 > end) return {};

    uint32_t n;
    std::memcpy(&n, p, 4); p += 4;

    std::vector<uint64_t> ts(n);
    uint64_t t = rle.base_ts;
    size_t i = 0;

    while (i < n && p < end) {
        uint8_t first = *p++;
        size_t run;
        if (first == 0 && p + 2 <= end) {
            uint16_t r16;
            std::memcpy(&r16, p, 2); p += 2;
            run = r16;
        } else {
            run = first;
        }
        if (p + 8 > end) break;
        uint64_t delta;
        std::memcpy(&delta, p, 8); p += 8;

        for (size_t j = 0; j < run && i < n; ++j, ++i) {
            ts[i] = t;
            t += delta;
        }
    }
    return ts;
}

// ============================================================
// Minimal LZ4-inspired block compressor
// Operates on arbitrary byte buffers
// Format: sequence of tokens
//   token = [literal_len:4 | match_len:4] [extra_literal_len...] [literals...]
//            [offset:2] [extra_match_len...]
// ============================================================
static constexpr size_t kHashBits    = 16;
static constexpr size_t kHashSize    = 1 << kHashBits;
static constexpr size_t kMinMatch    = 4;
static constexpr size_t kMaxLitLen   = 255 + 15;
static constexpr size_t kMaxMatchLen = 255 + 4;

static uint32_t hash4(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return (v * 2654435761U) >> (32 - kHashBits);
}

std::vector<uint8_t> lz4_compress(const uint8_t* src, size_t src_len) {
    if (src_len == 0) return {};

    std::vector<uint8_t> dst;
    dst.reserve(src_len + src_len / 255 + 16);

    // Store original length
    uint32_t olen = static_cast<uint32_t>(src_len);
    dst.insert(dst.end(), reinterpret_cast<uint8_t*>(&olen),
                          reinterpret_cast<uint8_t*>(&olen) + 4);

    // Hash table: position of last occurrence
    std::vector<int32_t> htab(kHashSize, -1);

    const uint8_t* anchor = src;
    const uint8_t* ip     = src;
    const uint8_t* end    = src + src_len;
    const uint8_t* ilimit = end - kMinMatch;

    auto encode_varint = [&](size_t v) {
        while (v >= 255) { dst.push_back(255); v -= 255; }
        dst.push_back(static_cast<uint8_t>(v));
    };

    while (ip < ilimit) {
        uint32_t h = hash4(ip);
        int32_t  ref_pos = htab[h];
        htab[h] = static_cast<int32_t>(ip - src);

        // Find match
        int match_len = 0;
        if (ref_pos >= 0) {
            const uint8_t* ref = src + ref_pos;
            int32_t offset = static_cast<int32_t>(ip - ref);
            if (offset > 0 && offset <= 65535) {
                // Verify and extend match
                const uint8_t* mp = ip;
                const uint8_t* rp = ref;
                while (mp + match_len < end - 4 &&
                       std::memcmp(mp + match_len, rp + match_len, 4) == 0)
                    match_len += 4;
                while (mp + match_len < end &&
                       mp[match_len] == rp[match_len])
                    ++match_len;

                if (match_len >= static_cast<int>(kMinMatch)) {
                    // Emit: literals + match
                    size_t lit_len = static_cast<size_t>(ip - anchor);
                    size_t token_pos = dst.size();
                    uint8_t tok = 0;
                    if (lit_len >= 15) tok |= 0xF0;
                    else               tok |= static_cast<uint8_t>(lit_len << 4);
                    size_t ml = static_cast<size_t>(match_len) - kMinMatch;
                    if (ml >= 15) tok |= 0x0F;
                    else          tok |= static_cast<uint8_t>(ml);
                    dst.push_back(tok);

                    if (lit_len >= 15) encode_varint(lit_len - 15);
                    dst.insert(dst.end(), anchor, ip);

                    uint16_t off16 = static_cast<uint16_t>(offset);
                    dst.insert(dst.end(), reinterpret_cast<uint8_t*>(&off16),
                                          reinterpret_cast<uint8_t*>(&off16) + 2);

                    if (ml >= 15) encode_varint(ml - 15);

                    ip     += match_len;
                    anchor  = ip;
                    continue;
                }
            }
        }
        ++ip;
    }

    // Final literals
    size_t lit_len = static_cast<size_t>(end - anchor);
    if (lit_len > 0) {
        uint8_t tok = (lit_len >= 15) ? 0xF0 : static_cast<uint8_t>(lit_len << 4);
        dst.push_back(tok);
        if (lit_len >= 15) encode_varint(lit_len - 15);
        dst.insert(dst.end(), anchor, end);
    }
    return dst;
}

std::vector<uint8_t> lz4_decompress(const uint8_t* src, size_t src_len) {
    if (src_len < 4) return {};
    const uint8_t* ip  = src;
    const uint8_t* end = src + src_len;

    uint32_t orig_len;
    std::memcpy(&orig_len, ip, 4); ip += 4;

    std::vector<uint8_t> dst;
    dst.reserve(orig_len);

    while (ip < end) {
        uint8_t tok = *ip++;
        size_t lit_len = (tok >> 4) & 0xF;
        if (lit_len == 15) {
            uint8_t extra;
            do { extra = *ip++; lit_len += extra; } while (extra == 255);
        }
        if (ip + lit_len > end) break;
        dst.insert(dst.end(), ip, ip + lit_len);
        ip += lit_len;
        if (ip >= end) break;

        uint16_t offset;
        std::memcpy(&offset, ip, 2); ip += 2;
        if (offset == 0) break;

        size_t match_len = (tok & 0xF) + kMinMatch;
        if ((tok & 0xF) == 15) {
            uint8_t extra;
            do { extra = *ip++; match_len += extra; } while (extra == 255);
        }

        size_t match_start = dst.size() - offset;
        for (size_t j = 0; j < match_len; ++j)
            dst.push_back(dst[match_start + j]);
    }
    return dst;
}

// ============================================================
// Full tick compression pipeline:
//   1. Delta-encode prices
//   2. RLE-encode timestamps
//   3. Store quantities as-is
//   4. LZ4-compress the combined stream
// ============================================================
struct CompressedBlock {
    std::vector<uint8_t> data;
    size_t               tick_count;
    uint64_t             first_ts;
    uint64_t             last_ts;
    double               compression_ratio;
};

CompressedBlock compress_ticks(const Tick* ticks, size_t n) {
    if (n == 0) return {{}, 0, 0, 0, 0.0};

    // Extract fields
    std::vector<int64_t>  prices(n), bid_prices(n), ask_prices(n);
    std::vector<uint64_t> timestamps(n), qtys(n);
    std::vector<uint8_t>  sides(n), types(n);

    for (size_t i = 0; i < n; ++i) {
        timestamps[i] = ticks[i].timestamp;
        prices[i]     = ticks[i].price;
        bid_prices[i] = ticks[i].bid_price;
        ask_prices[i] = ticks[i].ask_price;
        qtys[i]       = ticks[i].qty;
        sides[i]      = ticks[i].side;
        types[i]      = ticks[i].tick_type;
    }

    // Encode each stream
    auto enc_prices    = delta_encode_prices(prices.data(), n);
    auto enc_bid       = delta_encode_prices(bid_prices.data(), n);
    auto enc_ask       = delta_encode_prices(ask_prices.data(), n);
    auto rle_ts        = rle_encode_timestamps(timestamps.data(), n);

    // Pack into single buffer: [header][prices][bid][ask][ts_base(8)][ts_encoded][qty+flags]
    std::vector<uint8_t> raw;
    raw.reserve(enc_prices.size() + enc_bid.size() + enc_ask.size() +
                rle_ts.encoded.size() + 8 + n * 9);

    // Header: 4-byte count
    uint32_t cnt = static_cast<uint32_t>(n);
    raw.insert(raw.end(), reinterpret_cast<uint8_t*>(&cnt),
                          reinterpret_cast<uint8_t*>(&cnt) + 4);

    // Section lengths
    uint32_t len_p   = static_cast<uint32_t>(enc_prices.size());
    uint32_t len_bid = static_cast<uint32_t>(enc_bid.size());
    uint32_t len_ask = static_cast<uint32_t>(enc_ask.size());
    uint32_t len_ts  = static_cast<uint32_t>(rle_ts.encoded.size());
    raw.insert(raw.end(), reinterpret_cast<uint8_t*>(&len_p),   reinterpret_cast<uint8_t*>(&len_p)+4);
    raw.insert(raw.end(), reinterpret_cast<uint8_t*>(&len_bid), reinterpret_cast<uint8_t*>(&len_bid)+4);
    raw.insert(raw.end(), reinterpret_cast<uint8_t*>(&len_ask), reinterpret_cast<uint8_t*>(&len_ask)+4);
    raw.insert(raw.end(), reinterpret_cast<uint8_t*>(&len_ts),  reinterpret_cast<uint8_t*>(&len_ts)+4);

    // Data
    raw.insert(raw.end(), enc_prices.begin(), enc_prices.end());
    raw.insert(raw.end(), enc_bid.begin(),    enc_bid.end());
    raw.insert(raw.end(), enc_ask.begin(),    enc_ask.end());
    raw.insert(raw.end(), reinterpret_cast<const uint8_t*>(&rle_ts.base_ts),
                          reinterpret_cast<const uint8_t*>(&rle_ts.base_ts) + 8);
    raw.insert(raw.end(), rle_ts.encoded.begin(), rle_ts.encoded.end());

    // Quantities, sides, types (raw)
    for (size_t i = 0; i < n; ++i) {
        raw.insert(raw.end(), reinterpret_cast<uint8_t*>(&qtys[i]),
                              reinterpret_cast<uint8_t*>(&qtys[i]) + 8);
        raw.push_back(sides[i]);
    }

    // LZ4 compress
    auto compressed = lz4_compress(raw.data(), raw.size());

    double ratio = (compressed.size() > 0)
                   ? static_cast<double>(n * sizeof(Tick)) / compressed.size()
                   : 0.0;

    return {std::move(compressed), n,
            timestamps.front(), timestamps.back(), ratio};
}

std::vector<Tick> decompress_ticks(const CompressedBlock& block) {
    if (block.data.empty()) return {};

    // LZ4 decompress
    auto raw = lz4_decompress(block.data.data(), block.data.size());
    if (raw.size() < 20) return {};

    const uint8_t* p = raw.data();
    uint32_t n;
    std::memcpy(&n, p, 4); p += 4;

    uint32_t len_p, len_bid, len_ask, len_ts;
    std::memcpy(&len_p,   p, 4); p += 4;
    std::memcpy(&len_bid, p, 4); p += 4;
    std::memcpy(&len_ask, p, 4); p += 4;
    std::memcpy(&len_ts,  p, 4); p += 4;

    auto prices    = delta_decode_prices(p, len_p);    p += len_p;
    auto bid_ps    = delta_decode_prices(p, len_bid);   p += len_bid;
    auto ask_ps    = delta_decode_prices(p, len_ask);   p += len_ask;

    uint64_t base_ts;
    std::memcpy(&base_ts, p, 8); p += 8;
    RLETimestamps rle_ts;
    rle_ts.base_ts = base_ts;
    rle_ts.encoded.assign(p, p + len_ts); p += len_ts;
    auto timestamps = rle_decode_timestamps(rle_ts);

    std::vector<Tick> ticks(n);
    for (size_t i = 0; i < n; ++i) {
        if (i < prices.size())     ticks[i].price     = prices[i];
        if (i < bid_ps.size())     ticks[i].bid_price = bid_ps[i];
        if (i < ask_ps.size())     ticks[i].ask_price = ask_ps[i];
        if (i < timestamps.size()) ticks[i].timestamp = timestamps[i];

        uint64_t qty;
        std::memcpy(&qty, p, 8); p += 8;
        ticks[i].qty  = qty;
        ticks[i].side = *p++;
    }
    return ticks;
}

} // namespace compress
} // namespace tickstore
