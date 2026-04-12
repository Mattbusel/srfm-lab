// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// serialization.cpp — Zero-Copy Serialization Implementation
// =============================================================================

#include "rtel/serialization.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// Serializer::checksum_fnv1a
// ---------------------------------------------------------------------------
uint32_t Serializer::checksum_fnv1a(const uint8_t* data, std::size_t n) noexcept {
    uint32_t hash = 2166136261u;
    for (std::size_t i = 0; i < n; ++i) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash;
}

// ---------------------------------------------------------------------------
// Serializer::validate_blob
// ---------------------------------------------------------------------------
bool Serializer::validate_blob(const uint8_t* blob, std::size_t size) noexcept {
    if (!blob || size < sizeof(RTELBlobHeader)) return false;
    const auto* hdr = reinterpret_cast<const RTELBlobHeader*>(blob);
    return hdr->valid();
}

// ---------------------------------------------------------------------------
// Serializer::serialize
// ---------------------------------------------------------------------------
std::vector<uint8_t> Serializer::serialize(const TensorDescriptor& td,
                                             const void* data,
                                             std::size_t data_bytes) const {
    std::size_t total = align_up(sizeof(RTELBlobHeader), 64) + data_bytes;
    std::vector<uint8_t> blob(total, 0);

    auto* hdr = reinterpret_cast<RTELBlobHeader*>(blob.data());
    hdr->magic       = kRTELBlobMagic;
    hdr->schema_ver  = kRTELSchemaV1;
    hdr->flags       = 0;
    hdr->timestamp_ns= now_ns();
    hdr->tensor      = td;
    hdr->tensor.payload_bytes = data_bytes;

    if (data && data_bytes > 0) {
        uint8_t* dst = blob.data() + align_up(sizeof(RTELBlobHeader), 64);
        std::memcpy(dst, data, data_bytes);
    }
    return blob;
}

// ---------------------------------------------------------------------------
// Serializer::deserialize
// ---------------------------------------------------------------------------
Serializer::DeserResult
Serializer::deserialize(const uint8_t* blob, std::size_t blob_size) const noexcept {
    DeserResult r{};
    if (!validate_blob(blob, blob_size)) return r;
    const auto* hdr = reinterpret_cast<const RTELBlobHeader*>(blob);
    r.td         = hdr->tensor;
    std::size_t off = align_up(sizeof(RTELBlobHeader), 64);
    if (off + hdr->tensor.payload_bytes > blob_size) return r;
    r.data_ptr   = blob + off;
    r.data_bytes = hdr->tensor.payload_bytes;
    r.ok         = true;
    return r;
}

// ---------------------------------------------------------------------------
// Serializer::make_numpy_header_dict
// ---------------------------------------------------------------------------
std::string Serializer::make_numpy_header_dict(const TensorDescriptor& td) {
    std::ostringstream oss;
    oss << "{'descr': '" << dtype_numpy_str(td.dtype) << "', ";
    oss << "'fortran_order': False, ";
    oss << "'shape': (";
    for (uint8_t i = 0; i < td.ndim; ++i) {
        if (i > 0) oss << ", ";
        oss << td.shape[i];
    }
    if (td.ndim == 1) oss << ",";
    oss << ")}";
    return oss.str();
}

// ---------------------------------------------------------------------------
// Serializer::serialize_numpy
// ---------------------------------------------------------------------------
std::vector<uint8_t> Serializer::serialize_numpy(const TensorDescriptor& td,
                                                   const void* data,
                                                   std::size_t data_bytes) const {
    std::string dict = make_numpy_header_dict(td);
    // Pad dict to multiple of 64 bytes
    std::size_t hdr_len = dict.size() + 1;  // +1 for newline
    hdr_len = align_up(hdr_len, 64);
    dict.resize(hdr_len - 1, ' ');
    dict.back() = '\n';

    // Magic (6) + major/minor (2) + header_len (2) + dict
    std::size_t prefix = 6 + 2 + 2;
    std::size_t total  = prefix + hdr_len + data_bytes;
    std::vector<uint8_t> blob(total, 0);

    uint8_t* p = blob.data();
    // Magic: \x93NUMPY\x01\x00
    p[0] = 0x93; p[1]='N'; p[2]='U'; p[3]='M'; p[4]='P'; p[5]='Y';
    p[6] = 0x01; p[7] = 0x00;
    // Header length (little-endian uint16_t)
    uint16_t hl = static_cast<uint16_t>(hdr_len);
    std::memcpy(p + 8, &hl, 2);
    // Dict
    std::memcpy(p + 10, dict.c_str(), dict.size());
    // Data
    if (data && data_bytes > 0) {
        std::memcpy(p + prefix + hdr_len, data, data_bytes);
    }
    return blob;
}

// ---------------------------------------------------------------------------
// Serializer::deserialize_numpy
// ---------------------------------------------------------------------------
Serializer::DeserResult
Serializer::deserialize_numpy(const uint8_t* blob, std::size_t blob_size) const noexcept {
    DeserResult r{};
    if (!blob || blob_size < 10) return r;
    // Check magic
    if (blob[0] != 0x93 || blob[1] != 'N' || blob[2] != 'U' ||
        blob[3] != 'M' || blob[4] != 'P' || blob[5] != 'Y') return r;

    uint16_t hdr_len = 0;
    std::memcpy(&hdr_len, blob + 8, 2);
    hdr_len = from_little_endian(hdr_len);

    std::size_t data_off = 10 + hdr_len;
    if (data_off >= blob_size) return r;

    r.data_ptr   = blob + data_off;
    r.data_bytes = blob_size - data_off;
    r.ok         = true;
    // td left as default — caller parses dict string if needed
    r.td.payload_bytes = r.data_bytes;
    return r;
}

// ---------------------------------------------------------------------------
// Serializer::convert_endianness_f32
// ---------------------------------------------------------------------------
void Serializer::convert_endianness_f32(float* data, std::size_t n) noexcept {
    if (host_is_little_endian()) return;  // assume LE target
    for (std::size_t i = 0; i < n; ++i) {
        uint32_t tmp;
        std::memcpy(&tmp, data + i, 4);
        tmp = swap_bytes(tmp);
        std::memcpy(data + i, &tmp, 4);
    }
}

void Serializer::convert_endianness_f16(uint16_t* data, std::size_t n) noexcept {
    if (host_is_little_endian()) return;
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = swap_bytes(data[i]);
    }
}

// ---------------------------------------------------------------------------
// StreamSerializer
// ---------------------------------------------------------------------------
void StreamSerializer::serialize_streaming(const TensorDescriptor& td,
                                            const void* data,
                                            std::size_t data_bytes,
                                            ChunkCallback cb) const {
    if (!cb || !data || data_bytes == 0) return;

    uint32_t total_chunks = static_cast<uint32_t>(
        (data_bytes + chunk_size_ - 1) / chunk_size_);

    const uint8_t* src = reinterpret_cast<const uint8_t*>(data);
    std::vector<uint8_t> chunk_buf(sizeof(ChunkHeader) + chunk_size_);

    for (uint32_t idx = 0; idx < total_chunks; ++idx) {
        std::size_t offset = idx * chunk_size_;
        std::size_t size   = std::min(chunk_size_, data_bytes - offset);
        bool last = (idx + 1 == total_chunks);

        ChunkHeader* ch = reinterpret_cast<ChunkHeader*>(chunk_buf.data());
        ch->total_size   = data_bytes;
        ch->chunk_idx    = idx;
        ch->total_chunks = total_chunks;
        ch->chunk_size   = static_cast<uint32_t>(size);
        ch->flags        = last ? 1u : 0u;
        std::memcpy(chunk_buf.data() + sizeof(ChunkHeader), src + offset, size);

        cb(chunk_buf.data(), sizeof(ChunkHeader) + size, last);
    }
    (void)td;
}

std::vector<uint8_t> StreamSerializer::reassemble(
    const std::vector<std::vector<uint8_t>>& chunks) const {
    if (chunks.empty()) return {};
    const auto* ch0 = reinterpret_cast<const ChunkHeader*>(chunks[0].data());
    std::vector<uint8_t> result(ch0->total_size);
    for (const auto& c : chunks) {
        const auto* ch = reinterpret_cast<const ChunkHeader*>(c.data());
        std::size_t offset = ch->chunk_idx * chunk_size_;
        std::size_t size   = ch->chunk_size;
        std::memcpy(result.data() + offset, c.data() + sizeof(ChunkHeader), size);
    }
    return result;
}

// ---------------------------------------------------------------------------
// SchemaRegistry
// ---------------------------------------------------------------------------
SchemaRegistry& SchemaRegistry::instance() {
    static SchemaRegistry r;
    return r;
}

void SchemaRegistry::register_schema(const SchemaEntry& e) {
    std::string key = e.name + "_v" + std::to_string(e.version);
    schemas_[key] = e;
}

const SchemaEntry* SchemaRegistry::lookup(uint32_t version,
                                           const std::string& name) const {
    std::string key = name + "_v" + std::to_string(version);
    auto it = schemas_.find(key);
    return (it != schemas_.end()) ? &it->second : nullptr;
}

void SchemaRegistry::register_aeternus_schemas() {
    // LOB snapshot schema
    register_schema({1, "lob_snapshot", "Order book snapshot",
                     DType::FLOAT64, {kMaxAssets, sizeof(LOBSnapshot)/sizeof(double)}});
    // Vol surface schema
    register_schema({1, "vol_surface", "Implied volatility surface",
                     DType::FLOAT64, {kMaxAssets, kMaxStrikes, kMaxExpiries}});
    // Lumina predictions schema
    register_schema({1, "lumina_predictions", "Return/risk/confidence forecasts",
                     DType::FLOAT32, {3, kMaxAssets}});
    // HyperAgent actions schema
    register_schema({1, "agent_actions", "Position delta actions",
                     DType::FLOAT32, {kMaxAssets}});
    // Graph adjacency schema
    register_schema({1, "graph_adjacency", "CSR sparse graph",
                     DType::FLOAT32, {kCSRMaxEdges}});
}

// ---------------------------------------------------------------------------
// CTypesLayout
// ---------------------------------------------------------------------------
std::string CTypesLayout::numpy_dtype_str(DType d) noexcept {
    switch (d) {
        case DType::FLOAT32:   return "float32";
        case DType::FLOAT16:   return "float16";
        case DType::FLOAT64:   return "float64";
        case DType::INT32:     return "int32";
        case DType::INT64:     return "int64";
        case DType::UINT8:     return "uint8";
        case DType::COMPLEX64: return "complex64";
        default:               return "uint8";
    }
}

std::string CTypesLayout::buffer_format_str(DType d) noexcept {
    switch (d) {
        case DType::FLOAT32:   return "f";
        case DType::FLOAT16:   return "e";
        case DType::FLOAT64:   return "d";
        case DType::INT32:     return "i";
        case DType::INT64:     return "q";
        case DType::UINT8:     return "B";
        case DType::COMPLEX64: return "Zf";
        default:               return "B";
    }
}

std::string CTypesLayout::generate_ctypes_code(const TensorDescriptor& td,
                                                 const std::string& class_name) {
    std::ostringstream oss;
    oss << "import ctypes\nimport numpy as np\n\n";
    oss << "class " << class_name << "(ctypes.Structure):\n";
    oss << "    _pack_ = 1\n";
    oss << "    _fields_ = [\n";

    std::string ctype;
    switch (td.dtype) {
        case DType::FLOAT32:   ctype = "ctypes.c_float";  break;
        case DType::FLOAT64:   ctype = "ctypes.c_double"; break;
        case DType::INT32:     ctype = "ctypes.c_int32";  break;
        case DType::INT64:     ctype = "ctypes.c_int64";  break;
        case DType::UINT8:     ctype = "ctypes.c_uint8";  break;
        default:               ctype = "ctypes.c_uint8";  break;
    }

    uint64_t total = 1;
    for (uint8_t i = 0; i < td.ndim; ++i) total *= td.shape[i];

    oss << "        ('data', " << ctype << " * " << total << "),\n";
    oss << "    ]\n\n";

    oss << "    def to_numpy(self):\n";
    oss << "        return np.frombuffer(self.data, dtype=np.'"
        << numpy_dtype_str(td.dtype) << "')";
    if (td.ndim > 1) {
        oss << ".reshape(";
        for (uint8_t i = 0; i < td.ndim; ++i) {
            if (i) oss << ", ";
            oss << td.shape[i];
        }
        oss << ")";
    }
    oss << "\n";
    return oss.str();
}

} // namespace aeternus::rtel
