#include "csv_reader.hpp"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <cstdio>

#if defined(SRFM_HAVE_MMAP)
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace srfm {

// ============================================================
// CSVColumnMap
// ============================================================

static std::string to_lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

CSVColumnMap CSVColumnMap::from_header(const std::string& header_line) noexcept {
    CSVColumnMap m;
    // Split by comma
    int col = 0;
    std::size_t start = 0;
    auto process = [&](std::string tok) {
        // Strip whitespace and quotes
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '"' || tok.front() == '\r'))
            tok.erase(tok.begin());
        while (!tok.empty() && (tok.back()  == ' ' || tok.back()  == '"' || tok.back()  == '\r'))
            tok.pop_back();
        std::string lc = to_lower(tok);
        if (lc == "timestamp" || lc == "time" || lc == "date" || lc == "datetime")
            m.timestamp = col;
        else if (lc == "open" || lc == "o")    m.open      = col;
        else if (lc == "high" || lc == "h")    m.high      = col;
        else if (lc == "low"  || lc == "l")    m.low       = col;
        else if (lc == "close"|| lc == "c")    m.close     = col;
        else if (lc == "volume"|| lc == "vol" || lc == "v") m.volume = col;
        else if (lc == "symbol"|| lc == "sym" || lc == "ticker") m.symbol = col;
        ++col;
    };
    for (std::size_t i = 0; i <= header_line.size(); ++i) {
        if (i == header_line.size() || header_line[i] == ',') {
            process(header_line.substr(start, i - start));
            start = i + 1;
        }
    }
    return m;
}

// ============================================================
// CSVReader
// ============================================================

CSVReader::CSVReader(std::string filepath, int symbol_id, int timeframe_sec) noexcept
    : filepath_(std::move(filepath))
    , symbol_id_(symbol_id)
    , timeframe_sec_(timeframe_sec)
    , delimiter_(',')
    , skip_header_(true)
    , error_count_(0)
    , data_(nullptr)
    , file_size_(0)
#if defined(SRFM_HAVE_MMAP)
    , fd_(-1)
    , mmap_ptr_(MAP_FAILED)
#else
    , heap_buf_(nullptr)
#endif
{}

CSVReader::~CSVReader() noexcept {
    close();
}

bool CSVReader::open() noexcept {
#if defined(SRFM_HAVE_MMAP)
    return setup_mmap();
#else
    return setup_stream();
#endif
}

#if defined(SRFM_HAVE_MMAP)
bool CSVReader::setup_mmap() noexcept {
    fd_ = ::open(filepath_.c_str(), O_RDONLY);
    if (fd_ < 0) return false;

    struct stat st;
    if (::fstat(fd_, &st) < 0) { ::close(fd_); fd_ = -1; return false; }

    file_size_ = static_cast<std::size_t>(st.st_size);
    if (file_size_ == 0) { ::close(fd_); fd_ = -1; return false; }

    mmap_ptr_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) { ::close(fd_); fd_ = -1; return false; }

    data_ = static_cast<const char*>(mmap_ptr_);
    return true;
}
#endif

bool CSVReader::setup_stream() noexcept {
    FILE* f = std::fopen(filepath_.c_str(), "rb");
    if (!f) return false;

    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) { std::fclose(f); return false; }

    file_size_  = static_cast<std::size_t>(sz);
    heap_buf_   = new(std::nothrow) char[file_size_ + 1];
    if (!heap_buf_) { std::fclose(f); return false; }

    std::size_t nread = std::fread(heap_buf_, 1, file_size_, f);
    std::fclose(f);
    heap_buf_[nread] = '\0';
    file_size_       = nread;
    data_            = heap_buf_;
    return true;
}

void CSVReader::close() noexcept {
#if defined(SRFM_HAVE_MMAP)
    if (mmap_ptr_ != MAP_FAILED) {
        ::munmap(mmap_ptr_, file_size_);
        mmap_ptr_ = MAP_FAILED;
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#else
    delete[] heap_buf_;
    heap_buf_ = nullptr;
#endif
    data_      = nullptr;
    file_size_ = 0;
}

const char* CSVReader::skip_line(const char* pos, const char* end) noexcept {
    while (pos < end && *pos != '\n') ++pos;
    if (pos < end) ++pos;  // skip '\n'
    return pos;
}

int64_t CSVReader::parse_timestamp(const char* start, int len) noexcept {
    if (len <= 0) return 0;

    // Check if it looks like a number
    bool is_digit = std::isdigit(static_cast<unsigned char>(start[0]));

    if (is_digit) {
        // Integer Unix timestamp
        int64_t v = 0;
        for (int i = 0; i < len && std::isdigit(static_cast<unsigned char>(start[i])); ++i)
            v = v * 10 + (start[i] - '0');

        // Heuristic: seconds → < 2e10, ms → < 2e13, ns → >= 2e13
        if (v < 20000000000LL)           // seconds
            return v * constants::NS_PER_SEC;
        else if (v < 20000000000000LL)   // milliseconds
            return v * 1000000LL;
        else                             // nanoseconds
            return v;
    }

    // Try to parse ISO 8601: "2024-01-15 09:30:00" or "2024-01-15T09:30:00"
    // Simple parser: extract year, month, day, hour, min, sec
    if (len >= 19) {
        auto rd = [&](int off, int n) -> int {
            int v = 0;
            for (int i = 0; i < n; ++i) v = v * 10 + (start[off + i] - '0');
            return v;
        };
        int year = rd(0,4), mon = rd(5,2), day = rd(8,2);
        int hour = rd(11,2), min = rd(14,2), sec = rd(17,2);
        // Approximate: days since epoch
        // Use a simple Gregorian calculation
        auto is_leap = [](int y) { return (y%4==0 && y%100!=0) || y%400==0; };
        int days_per_month[] = {31,28,31,30,31,30,31,31,30,31,30,31};
        if (is_leap(year)) days_per_month[1] = 29;

        int64_t days = 0;
        for (int y = 1970; y < year; ++y)
            days += is_leap(y) ? 366 : 365;
        for (int m = 0; m < mon - 1; ++m)
            days += days_per_month[m];
        days += day - 1;
        int64_t ts_sec = days * 86400LL + hour * 3600LL + min * 60LL + sec;
        return ts_sec * constants::NS_PER_SEC;
    }
    return 0;
}

double CSVReader::parse_double(const char* start, int len) noexcept {
    // Fast path: use strtod with a null-terminated copy is slow; use manual parse
    if (len <= 0) return 0.0;

    bool   neg   = (start[0] == '-');
    int    i     = neg ? 1 : 0;
    double val   = 0.0;
    double frac  = 1.0;
    bool   has_dot = false;

    for (; i < len; ++i) {
        char c = start[i];
        if (c == '.') { has_dot = true; continue; }
        if (c < '0' || c > '9') break;
        int digit = c - '0';
        if (has_dot) {
            frac /= 10.0;
            val  += digit * frac;
        } else {
            val   = val * 10.0 + digit;
        }
    }
    return neg ? -val : val;
}

const char* CSVReader::parse_line(const char* pos, const char* end,
                                   OHLCVBar& out) noexcept {
    // Find all field starts and lengths
    static constexpr int MAX_COLS = 16;
    const char* field_start[MAX_COLS];
    int         field_len[MAX_COLS];
    int         n_fields = 0;

    const char* p = pos;
    field_start[0] = p;
    while (p < end && *p != '\n') {
        if (*p == delimiter_ && n_fields < MAX_COLS - 1) {
            field_len[n_fields] = static_cast<int>(p - field_start[n_fields]);
            ++n_fields;
            field_start[n_fields] = p + 1;
        }
        ++p;
    }
    // Last field
    if (n_fields < MAX_COLS) {
        field_len[n_fields] = static_cast<int>(p - field_start[n_fields]);
        // Strip \r
        if (field_len[n_fields] > 0 &&
            field_start[n_fields][field_len[n_fields]-1] == '\r')
            --field_len[n_fields];
        ++n_fields;
    }
    // Advance past '\n'
    if (p < end) ++p;

    auto get_f = [&](int col) -> double {
        if (col < 0 || col >= n_fields) return 0.0;
        return parse_double(field_start[col], field_len[col]);
    };

    out.timestamp_ns  = parse_timestamp(field_start[col_map_.timestamp],
                                         field_len[col_map_.timestamp]);
    out.open          = get_f(col_map_.open);
    out.high          = get_f(col_map_.high);
    out.low           = get_f(col_map_.low);
    out.close         = get_f(col_map_.close);
    out.volume        = get_f(col_map_.volume);
    out.symbol_id     = symbol_id_;
    out.timeframe_sec = timeframe_sec_;

    // Basic sanity check
    if (out.close <= 0.0 || out.high < out.low) {
        ++error_count_;
    }

    return p;
}

std::size_t CSVReader::read_all(BarConsumer consumer) noexcept {
    if (!data_) return 0;

    const char* pos = data_;
    const char* end = data_ + file_size_;
    std::size_t count = 0;

    // Skip header
    if (skip_header_ && pos < end) {
        // Detect header by checking if first field looks like a number
        const char* line_end = pos;
        while (line_end < end && *line_end != '\n') ++line_end;

        bool first_is_digit = pos < end && std::isdigit(static_cast<unsigned char>(*pos));
        if (!first_is_digit) {
            // Parse header to build column map
            std::string header(pos, line_end);
            col_map_ = CSVColumnMap::from_header(header);
            pos = (line_end < end) ? line_end + 1 : end;
        }
    }

    OHLCVBar bar;
    while (pos < end) {
        // Skip blank lines
        while (pos < end && (*pos == '\n' || *pos == '\r')) ++pos;
        if (pos >= end) break;

        const char* next = parse_line(pos, end, bar);
        if (bar.close > 0.0 && bar.high >= bar.low) {
            consumer(bar);
            ++count;
        }
        pos = next;
    }
    return count;
}

std::size_t CSVReader::read_bars(OHLCVBar* out_bars, std::size_t max_bars) noexcept {
    std::size_t count = 0;
    read_all([&](const OHLCVBar& bar) {
        if (count < max_bars) out_bars[count++] = bar;
    });
    return count;
}

// ============================================================
// Convenience loader
// ============================================================

std::vector<OHLCVBar> load_csv(const std::string& path,
                                 int symbol_id, int timeframe_sec) {
    std::vector<OHLCVBar> bars;
    bars.reserve(100000);
    CSVReader reader(path, symbol_id, timeframe_sec);
    if (!reader.open()) return bars;
    reader.read_all([&](const OHLCVBar& bar) { bars.push_back(bar); });
    return bars;
}

} // namespace srfm
