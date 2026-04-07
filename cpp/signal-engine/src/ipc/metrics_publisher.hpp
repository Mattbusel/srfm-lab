#pragma once
// metrics_publisher.hpp -- Publishes signal engine metrics to the
// observability layer via a memory-mapped JSON status file.
//
// Writes to /tmp/srfm_signal_status.json every N bars (default: 10).
// Uses mmap + atomic pointer swap on POSIX; falls back to fwrite on Windows.
// Thread-safe via std::atomic and a lightweight spin lock.

#include "srfm/types.hpp"
#include "composite/regime_signal.hpp"
#include <atomic>
#include <string>
#include <cstdint>
#include <cstring>
#include <array>
#include <chrono>
#include <mutex>

// Platform detection for mmap
#if defined(__linux__) || defined(__APPLE__) || defined(SRFM_HAVE_MMAP)
#  define SRFM_MMAP_AVAILABLE 1
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace srfm {
namespace ipc {

// ------------------------------------------------------------------
// PerSymbolMetrics -- per-instrument metrics snapshot
// ------------------------------------------------------------------

struct PerSymbolMetrics {
    int32_t symbol_id   = -1;
    double  bh_mass     = 0.0;  // latest BH mass signal
    double  last_signal = 0.0;  // latest composite signal [-1, +1]
    double  last_rsi    = 50.0;
    double  last_atr    = 0.0;
    int32_t last_regime = 0;    // 0=RANGING .. 3=VOLATILE
    int64_t last_bar_ts = 0;
};

// ------------------------------------------------------------------
// EngineMetrics -- top-level engine health snapshot
// ------------------------------------------------------------------

struct EngineMetrics {
    // Throughput
    int64_t bars_processed      = 0;
    double  signals_per_sec     = 0.0;
    double  ring_buffer_fill_pct= 0.0;  // 0..1

    // Latency
    int64_t last_bar_latency_ns = 0;   // time to process most recent bar
    int64_t p99_latency_ns      = 0;
    int64_t p50_latency_ns      = 0;

    // Wall clock at snapshot time
    int64_t snapshot_ts_ns      = 0;

    // Per-symbol array (up to MAX_INSTRUMENTS entries)
    std::array<PerSymbolMetrics, MAX_INSTRUMENTS> symbols{};
    int     symbol_count        = 0;

    // Engine state
    bool    is_running          = false;
    char    version[16]         = "1.0.0";
};

// ------------------------------------------------------------------
// MetricsPublisher
// ------------------------------------------------------------------

class MetricsPublisher {
public:
    static constexpr std::size_t MMAP_SIZE = 65536;  // 64 KiB -- sufficient for JSON

    explicit MetricsPublisher(const char* filepath = "/tmp/srfm_signal_status.json",
                               int         publish_every_n_bars = 10) noexcept
        : filepath_(filepath)
        , publish_interval_(publish_every_n_bars)
        , bar_counter_(0)
    {
        open_file();
    }

    ~MetricsPublisher() noexcept { close_file(); }

    // Non-copyable, non-movable (owns OS resources)
    MetricsPublisher(const MetricsPublisher&)             = delete;
    MetricsPublisher& operator=(const MetricsPublisher&)  = delete;

    // ------------------------------------------------------------------
    // Update interface
    // ------------------------------------------------------------------

    /// Record that one bar was processed and update latency stats.
    /// Call this immediately after processing a bar.
    void record_bar(int64_t latency_ns) noexcept {
        latency_ring_[static_cast<std::size_t>(ring_head_)] = latency_ns;
        ring_head_ = (ring_head_ + 1) % LATENCY_RING;
        if (ring_fill_ < LATENCY_RING) ++ring_fill_;

        int64_t cnt = bars_processed_.fetch_add(1, std::memory_order_relaxed) + 1;

        // Track throughput: accumulate over publish window
        ++window_bars_;
        auto now = now_ns();
        if (window_start_ns_ == 0) window_start_ns_ = now;

        if (cnt % publish_interval_ == 0) {
            flush(now);
        }
    }

    /// Update per-symbol metrics.
    void update_symbol(int32_t symbol_id,
                        double  bh_mass,
                        double  last_signal,
                        double  last_rsi,
                        double  last_atr,
                        int32_t last_regime,
                        int64_t bar_ts) noexcept
    {
        if (symbol_id < 0 || symbol_id >= MAX_INSTRUMENTS) return;
        std::lock_guard<std::mutex> lock(mu_);
        auto& s         = metrics_.symbols[static_cast<std::size_t>(symbol_id)];
        s.symbol_id     = symbol_id;
        s.bh_mass       = bh_mass;
        s.last_signal   = last_signal;
        s.last_rsi      = last_rsi;
        s.last_atr      = last_atr;
        s.last_regime   = last_regime;
        s.last_bar_ts   = bar_ts;
        if (symbol_id >= metrics_.symbol_count)
            metrics_.symbol_count = symbol_id + 1;
    }

    /// Update symbol metrics from a RegimeSignal.
    void update_from_regime(const composite::RegimeSignal& sig,
                             double bh_mass = 0.0) noexcept
    {
        update_symbol(sig.symbol_id,
                      bh_mass,
                      sig.composite,
                      sig.rsi,
                      sig.atr,
                      sig.regime_idx,
                      sig.timestamp_ns);
    }

    /// Set ring buffer fill percentage (0..1).
    void set_ring_buffer_fill(double pct) noexcept {
        ring_buf_fill_.store(pct, std::memory_order_relaxed);
    }

    /// Returns current wall-clock nanoseconds (static helper, defined in .cpp).
    static int64_t clock_ns() noexcept;

    void set_running(bool running) noexcept {
        is_running_.store(running, std::memory_order_relaxed);
        if (running) open_file();
    }

    /// Force an immediate publish regardless of interval.
    void flush_now() noexcept { flush(now_ns()); }

    int64_t bars_processed() const noexcept {
        return bars_processed_.load(std::memory_order_relaxed);
    }

    const std::string& filepath() const noexcept { return filepath_; }

private:
    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    static int64_t now_ns() noexcept {
        using namespace std::chrono;
        return static_cast<int64_t>(
            duration_cast<nanoseconds>(
                steady_clock::now().time_since_epoch()).count());
    }

    void flush(int64_t now) noexcept {
        std::lock_guard<std::mutex> lock(mu_);

        // Compute throughput
        double elapsed_s = (window_start_ns_ > 0)
                           ? (now - window_start_ns_) * 1e-9
                           : 1.0;
        double sps = (elapsed_s > 1e-9)
                     ? static_cast<double>(window_bars_) / elapsed_s
                     : 0.0;
        window_bars_    = 0;
        window_start_ns_= now;

        metrics_.bars_processed       = bars_processed_.load(std::memory_order_relaxed);
        metrics_.signals_per_sec      = sps;
        metrics_.ring_buffer_fill_pct = ring_buf_fill_.load(std::memory_order_relaxed);
        metrics_.snapshot_ts_ns       = now;
        metrics_.is_running           = is_running_.load(std::memory_order_relaxed);

        // Latency percentiles from ring
        if (ring_fill_ > 0) {
            metrics_.last_bar_latency_ns = latency_ring_[
                static_cast<std::size_t>((ring_head_ - 1 + LATENCY_RING) % LATENCY_RING)];

            // Sort a copy for percentiles
            std::array<int64_t, LATENCY_RING> sorted;
            int n = ring_fill_;
            std::copy(latency_ring_.begin(),
                      latency_ring_.begin() + n,
                      sorted.begin());
            std::sort(sorted.begin(), sorted.begin() + n);
            metrics_.p50_latency_ns = sorted[static_cast<std::size_t>(n / 2)];
            metrics_.p99_latency_ns = sorted[static_cast<std::size_t>(n * 99 / 100)];
        }

        // Render JSON
        char buf[MMAP_SIZE];
        int  written = render_json(buf, sizeof(buf), metrics_);
        if (written <= 0) return;

        write_to_file(buf, static_cast<std::size_t>(written));
    }

    static int render_json(char* buf, std::size_t cap,
                            const EngineMetrics& m) noexcept
    {
        // Manual JSON rendering to avoid heap allocations on hot path.
        int pos = 0;
        auto w = [&](const char* s) {
            std::size_t l = std::strlen(s);
            if (static_cast<std::size_t>(pos) + l < cap) {
                std::memcpy(buf + pos, s, l);
                pos += static_cast<int>(l);
            }
        };
        auto wi64 = [&](int64_t v) {
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "%lld", static_cast<long long>(v));
            w(tmp);
        };
        auto wdbl = [&](double v) {
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "%.6f", v);
            w(tmp);
        };

        w("{");
        w("\"bars_processed\":"); wi64(m.bars_processed); w(",");
        w("\"signals_per_sec\":"); wdbl(m.signals_per_sec); w(",");
        w("\"ring_buffer_fill_pct\":"); wdbl(m.ring_buffer_fill_pct); w(",");
        w("\"last_bar_latency_ns\":"); wi64(m.last_bar_latency_ns); w(",");
        w("\"p50_latency_ns\":"); wi64(m.p50_latency_ns); w(",");
        w("\"p99_latency_ns\":"); wi64(m.p99_latency_ns); w(",");
        w("\"snapshot_ts_ns\":"); wi64(m.snapshot_ts_ns); w(",");
        w("\"is_running\":"); w(m.is_running ? "true" : "false"); w(",");
        w("\"version\":\""); w(m.version); w("\",");
        w("\"symbols\":[");
        for (int i = 0; i < m.symbol_count; ++i) {
            if (i > 0) w(",");
            const auto& s = m.symbols[static_cast<std::size_t>(i)];
            w("{");
            w("\"id\":"); { char t[16]; snprintf(t, sizeof(t), "%d", s.symbol_id); w(t); } w(",");
            w("\"bh_mass\":"); wdbl(s.bh_mass); w(",");
            w("\"signal\":"); wdbl(s.last_signal); w(",");
            w("\"rsi\":"); wdbl(s.last_rsi); w(",");
            w("\"atr\":"); wdbl(s.last_atr); w(",");
            w("\"regime\":\""); w(composite::regime_label(s.last_regime)); w("\"");
            w("}");
        }
        w("]}");
        if (static_cast<std::size_t>(pos) < cap) buf[pos] = '\0';
        return pos;
    }

    // ------------------------------------------------------------------
    // File I/O (mmap where available, fwrite fallback elsewhere)
    // ------------------------------------------------------------------

    void open_file() noexcept {
#if defined(SRFM_MMAP_AVAILABLE)
        if (fd_ >= 0) return;  // already open
        fd_ = ::open(filepath_.c_str(),
                     O_RDWR | O_CREAT | O_TRUNC,
                     static_cast<mode_t>(0644));
        if (fd_ < 0) { mmap_ptr_ = nullptr; return; }
        // Pre-size the file
        if (::ftruncate(fd_, static_cast<off_t>(MMAP_SIZE)) < 0) {
            ::close(fd_); fd_ = -1; return;
        }
        mmap_ptr_ = static_cast<char*>(
            ::mmap(nullptr, MMAP_SIZE,
                   PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
        if (mmap_ptr_ == MAP_FAILED) {
            mmap_ptr_ = nullptr;
            ::close(fd_); fd_ = -1;
        }
#endif
    }

    void close_file() noexcept {
#if defined(SRFM_MMAP_AVAILABLE)
        if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
            ::munmap(mmap_ptr_, MMAP_SIZE);
            mmap_ptr_ = nullptr;
        }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
    }

    void write_to_file(const char* data, std::size_t len) noexcept {
#if defined(SRFM_MMAP_AVAILABLE)
        if (mmap_ptr_ && len < MMAP_SIZE) {
            std::memcpy(mmap_ptr_, data, len);
            // Zero-pad the rest to avoid stale JSON
            std::memset(mmap_ptr_ + len, 0, MMAP_SIZE - len);
            ::msync(mmap_ptr_, len, MS_ASYNC);
            return;
        }
#endif
        // Fallback: open, write, close
        FILE* f = std::fopen(filepath_.c_str(), "w");
        if (!f) return;
        std::fwrite(data, 1, len, f);
        std::fclose(f);
    }

    // ------------------------------------------------------------------
    // Data
    // ------------------------------------------------------------------
    static constexpr int LATENCY_RING = 128;

    std::string     filepath_;
    int             publish_interval_;
    std::mutex      mu_;

    std::atomic<int64_t> bars_processed_{0};
    std::atomic<double>  ring_buf_fill_{0.0};
    std::atomic<bool>    is_running_{false};

    int64_t                              window_start_ns_= 0;
    int64_t                              window_bars_    = 0;
    int                                  bar_counter_    = 0;

    std::array<int64_t, LATENCY_RING>    latency_ring_{};
    int                                  ring_head_      = 0;
    int                                  ring_fill_      = 0;

    EngineMetrics                        metrics_{};

#if defined(SRFM_MMAP_AVAILABLE)
    int   fd_       = -1;
    char* mmap_ptr_ = nullptr;
#endif
};

} // namespace ipc
} // namespace srfm
