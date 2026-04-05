#pragma once
#include "srfm/types.hpp"
#include <string>
#include <vector>
#include <functional>

namespace srfm {

/// Column index mapping for CSV files.
struct CSVColumnMap {
    int timestamp = 0;
    int open      = 1;
    int high      = 2;
    int low       = 3;
    int close     = 4;
    int volume    = 5;
    int symbol    = -1;  // -1 = not present (single symbol file)

    /// Auto-detect columns from header row.
    static CSVColumnMap from_header(const std::string& header_line) noexcept;
};

using BarConsumer = std::function<void(const OHLCVBar&)>;

/// Fast CSV reader for OHLCV data using memory-mapped I/O (POSIX) or
/// streaming read (Windows fallback).
///
/// Supports formats:
///   - Unix timestamp (seconds or milliseconds, auto-detected)
///   - ISO 8601 datetime
///   - Header row auto-detection
class CSVReader {
public:
    explicit CSVReader(std::string filepath,
                       int symbol_id     = 0,
                       int timeframe_sec = 60) noexcept;
    ~CSVReader() noexcept;

    CSVReader(const CSVReader&) = delete;
    CSVReader& operator=(const CSVReader&) = delete;

    /// Open the file. Returns false on failure.
    bool open() noexcept;

    /// Read all bars, calling consumer for each. Returns bar count.
    std::size_t read_all(BarConsumer consumer) noexcept;

    /// Read up to max_bars bars. Returns bars read.
    std::size_t read_bars(OHLCVBar* out_bars, std::size_t max_bars) noexcept;

    /// Total bytes in file.
    std::size_t file_size() const noexcept { return file_size_; }

    /// True if file is open and mapped.
    bool is_open() const noexcept { return data_ != nullptr; }

    void close() noexcept;

    void set_column_map(const CSVColumnMap& map) noexcept { col_map_ = map; }
    void set_delimiter(char delim) noexcept { delimiter_ = delim; }
    void set_skip_header(bool skip) noexcept { skip_header_ = skip; }

    /// Returns the number of parse errors encountered.
    int error_count() const noexcept { return error_count_; }

private:
    bool        setup_mmap() noexcept;
    bool        setup_stream() noexcept;

    const char* parse_line(const char* pos, const char* end,
                            OHLCVBar& out) noexcept;
    const char* skip_line(const char* pos, const char* end) noexcept;

    /// Parse a timestamp field. Handles: Unix seconds, Unix ms, Unix ns.
    int64_t     parse_timestamp(const char* start, int len) noexcept;

    /// Parse a double from a char buffer.
    static double parse_double(const char* start, int len) noexcept;

    std::string  filepath_;
    int          symbol_id_;
    int          timeframe_sec_;
    char         delimiter_;
    bool         skip_header_;
    int          error_count_;

    CSVColumnMap col_map_;

    // Memory-mapped file state
    const char*  data_;
    std::size_t  file_size_;

#if defined(SRFM_HAVE_MMAP)
    int          fd_;
    void*        mmap_ptr_;
#else
    // Fallback: heap buffer
    char*        heap_buf_;
#endif
};

// ============================================================
// Convenience free function
// ============================================================

/// Load all bars from a CSV file into a vector.
std::vector<OHLCVBar> load_csv(const std::string& path,
                                 int symbol_id = 0,
                                 int timeframe_sec = 60);

} // namespace srfm
