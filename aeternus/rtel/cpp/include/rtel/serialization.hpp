// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// serialization.hpp — Zero-Copy Serialization
// =============================================================================
// FlatBuffers-style manual layout for tensor data.
// Supports:
//   - Numpy array header compatible format (.npy magic + header)
//   - Python ctypes-compatible memory layout
//   - Endianness detection and conversion
//   - Schema versioning with forward compatibility
//
// Layout for a serialized tensor blob:
//   [0..7]    Magic: 0x93 "NUMPY" (8 bytes)  — for .npy compat
//   [8..9]    Major version, minor version (uint8 each)
//   [10..11]  Header length (uint16_t LE)
//   [12..H]   JSON-like header dict: {'descr': '<f4', 'fortran_order': False, 'shape': (N,)}
//   [H..]     Raw tensor data (C-order, dtype as specified)
//
// For RTEL internal use, an extended header is prepended:
//   [0..7]    RTEL magic: 0xAE7E4E5552544C01 (little-endian)
//   [8..11]   Schema version (uint32_t)
//   [12..15]  Flags
//   [16..]    TensorDescriptor (fixed 256 bytes)
//   [272..]   Tensor data
// =============================================================================

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "shm_bus.hpp"

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// Endianness utilities
// ---------------------------------------------------------------------------
inline bool host_is_little_endian() noexcept {
    uint32_t x = 1;
    return *reinterpret_cast<const uint8_t*>(&x) == 1;
}

template<typename T>
T swap_bytes(T val) noexcept {
    T result{};
    const uint8_t* src = reinterpret_cast<const uint8_t*>(&val);
    uint8_t* dst = reinterpret_cast<uint8_t*>(&result);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        dst[i] = src[sizeof(T) - 1 - i];
    }
    return result;
}

template<typename T>
T to_little_endian(T val) noexcept {
    return host_is_little_endian() ? val : swap_bytes(val);
}

template<typename T>
T from_little_endian(T val) noexcept {
    return host_is_little_endian() ? val : swap_bytes(val);
}

template<typename T>
T to_big_endian(T val) noexcept {
    return host_is_little_endian() ? swap_bytes(val) : val;
}

// ---------------------------------------------------------------------------
// RTEL binary blob header
// ---------------------------------------------------------------------------
static constexpr uint64_t kRTELBlobMagic  = 0xAE7E4E5552544C01ULL;
static constexpr uint32_t kRTELSchemaV1   = 1;

struct alignas(8) RTELBlobHeader {
    uint64_t         magic      = kRTELBlobMagic;
    uint32_t         schema_ver = kRTELSchemaV1;
    uint32_t         flags      = 0;
    uint64_t         timestamp_ns= 0;
    TensorDescriptor tensor;
    uint8_t          reserved[64] = {};

    static constexpr uint32_t FLAG_COMPRESSED = 1u << 0;
    static constexpr uint32_t FLAG_NUMPY_COMPAT = 1u << 1;
    static constexpr uint32_t FLAG_FORTRAN_ORDER = 1u << 2;

    bool valid() const noexcept { return magic == kRTELBlobMagic; }
};
static_assert(sizeof(RTELBlobHeader) <= 512, "RTELBlobHeader too large");

// ---------------------------------------------------------------------------
// NumpyArrayHeader — .npy format header (simplified)
// ---------------------------------------------------------------------------
struct NumpyArrayHeader {
    uint8_t  magic[8]     = {0x93,'N','U','M','P','Y',0x01,0x00};
    uint16_t header_len   = 0;
    // Followed by ASCII dict string padded to multiple of 64 bytes

    static constexpr std::size_t kMagicLen = 8;
    static constexpr std::size_t kHeaderLenOff = 8;
};

// ---------------------------------------------------------------------------
// Serializer — converts tensors to/from binary blobs
// ---------------------------------------------------------------------------
class Serializer {
public:
    Serializer() = default;

    // Serialize a tensor buffer to a byte vector (RTEL format)
    std::vector<uint8_t> serialize(const TensorDescriptor& td,
                                    const void* data,
                                    std::size_t data_bytes) const;

    // Deserialize from RTEL format blob
    // Returns (TensorDescriptor, pointer to data start, data size)
    // Pointer is into the provided buffer — zero-copy
    struct DeserResult {
        TensorDescriptor  td;
        const uint8_t*    data_ptr  = nullptr;
        std::size_t       data_bytes= 0;
        bool              ok        = false;
    };
    DeserResult deserialize(const uint8_t* blob, std::size_t blob_size) const noexcept;

    // Serialize as numpy .npy format (for Python interop)
    std::vector<uint8_t> serialize_numpy(const TensorDescriptor& td,
                                          const void* data,
                                          std::size_t data_bytes) const;

    // Deserialize numpy .npy format
    DeserResult deserialize_numpy(const uint8_t* blob, std::size_t blob_size) const noexcept;

    // In-place endianness conversion (for float32/float16 arrays)
    static void convert_endianness_f32(float* data, std::size_t n) noexcept;
    static void convert_endianness_f16(uint16_t* data, std::size_t n) noexcept;

    // Build numpy header dict string
    static std::string make_numpy_header_dict(const TensorDescriptor& td);

    // Validate RTEL blob
    static bool validate_blob(const uint8_t* blob, std::size_t size) noexcept;

    // Compute simple checksum (FNV-1a over data)
    static uint32_t checksum_fnv1a(const uint8_t* data, std::size_t n) noexcept;
};

// ---------------------------------------------------------------------------
// StreamSerializer — streaming serialization for large tensors
// ---------------------------------------------------------------------------
class StreamSerializer {
public:
    explicit StreamSerializer(std::size_t chunk_size = 64 * 1024)
        : chunk_size_(chunk_size) {}

    // Serialize tensor in chunks; calls callback for each chunk
    using ChunkCallback = std::function<void(const uint8_t*, std::size_t, bool last)>;
    void serialize_streaming(const TensorDescriptor& td,
                             const void* data,
                             std::size_t data_bytes,
                             ChunkCallback cb) const;

    // Reassemble chunks into complete blob
    std::vector<uint8_t> reassemble(const std::vector<std::vector<uint8_t>>& chunks) const;

private:
    std::size_t chunk_size_;

    struct ChunkHeader {
        uint64_t total_size   = 0;
        uint32_t chunk_idx    = 0;
        uint32_t total_chunks = 0;
        uint32_t chunk_size   = 0;
        uint32_t flags        = 0;  // bit0: last chunk
    };
};

// ---------------------------------------------------------------------------
// SchemaRegistry — versioned schema management
// ---------------------------------------------------------------------------
struct SchemaEntry {
    uint32_t    version;
    std::string name;
    std::string description;
    DType       dtype;
    std::vector<uint64_t> shape;  // 0 = dynamic dimension
};

class SchemaRegistry {
public:
    static SchemaRegistry& instance();

    void register_schema(const SchemaEntry& e);
    const SchemaEntry* lookup(uint32_t version, const std::string& name) const;

    // Register all AETERNUS standard schemas
    void register_aeternus_schemas();

private:
    SchemaRegistry() = default;
    std::unordered_map<std::string, SchemaEntry> schemas_;
};

// ---------------------------------------------------------------------------
// CTypesLayout — generates Python ctypes struct definitions
// ---------------------------------------------------------------------------
class CTypesLayout {
public:
    // Generate Python ctypes.Structure source code for a TensorDescriptor
    static std::string generate_ctypes_code(const TensorDescriptor& td,
                                             const std::string& class_name);

    // Generate numpy dtype string (e.g. "float32", "int64")
    static std::string numpy_dtype_str(DType d) noexcept;

    // Generate a memoryview-compatible buffer protocol descriptor
    static std::string buffer_format_str(DType d) noexcept;
};

} // namespace aeternus::rtel
