#pragma once
#include "srfm/types.hpp"
#include <string>
#include <cstdio>

namespace srfm {

/// Hand-rolled JSON writer for SignalOutput structs.
/// No external JSON library. Knows the exact SignalOutput schema.
class JSONWriter {
public:
    explicit JSONWriter(std::string filepath = "") noexcept;
    ~JSONWriter() noexcept;

    bool open(const std::string& filepath) noexcept;
    void close() noexcept;

    /// Write a single SignalOutput as a JSON object on one line (NDJSON format).
    bool write(const SignalOutput& sig) noexcept;

    /// Write multiple signals as a JSON array.
    bool write_array(const SignalOutput* sigs, std::size_t n) noexcept;

    /// Serialize a SignalOutput to a pre-allocated buffer.
    /// Returns number of bytes written.
    static int serialize(const SignalOutput& sig, char* buf, int buf_size) noexcept;

    /// Serialize a SignalOutput to a std::string.
    static std::string to_string(const SignalOutput& sig) noexcept;

    bool is_open() const noexcept { return file_ != nullptr; }
    std::size_t records_written() const noexcept { return records_; }

private:
    FILE*       file_;
    std::size_t records_;

    static char* write_key_double(char* p, const char* key, double val) noexcept;
    static char* write_key_int64(char* p, const char* key, int64_t val) noexcept;
    static char* write_key_int(char* p, const char* key, int val) noexcept;
    static char* write_key_bool(char* p, const char* key, bool val) noexcept;
    static char* write_double(char* p, double val) noexcept;
    static char* write_int64(char* p, int64_t val) noexcept;
};

/// Formats a double to a buffer with limited precision (6 sig figs).
/// Returns pointer past the last written character.
char* fmt_double(char* buf, double val) noexcept;

/// Formats an int64 to buffer. Returns pointer past last char.
char* fmt_int64(char* buf, int64_t val) noexcept;

} // namespace srfm
