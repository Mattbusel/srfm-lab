#pragma once
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace tickstore {

static constexpr size_t CACHE_LINE = 64;

// Header stored at start of mmap region
struct alignas(CACHE_LINE) RingBufferHeader {
    uint64_t magic;          // 0xDEADBEEF12345678
    uint64_t version;        // format version
    uint64_t capacity;       // total data capacity in bytes
    uint64_t element_size;   // size of each element
    char     padding0[CACHE_LINE - 4*sizeof(uint64_t)];

    alignas(CACHE_LINE)
    std::atomic<uint64_t> write_seq;   // monotonically increasing write sequence
    char     padding1[CACHE_LINE - sizeof(std::atomic<uint64_t>)];

    alignas(CACHE_LINE)
    std::atomic<uint64_t> read_seq;    // last committed read sequence
    char     padding2[CACHE_LINE - sizeof(std::atomic<uint64_t>)];
};

static constexpr uint64_t RING_MAGIC   = 0xDEADBEEF12345678ULL;
static constexpr uint64_t RING_VERSION = 1;

// Lock-free ring buffer backed by memory-mapped file
// Supports:
//   - O(1) append
//   - Multiple concurrent readers (each maintains their own read pointer)
//   - Persistence across process restarts via mmap
//   - Power-of-2 capacity for fast modulo
//
// Layout of mmap region:
//   [RingBufferHeader] [element_0] [element_1] ... [element_{cap-1}]
template <typename T>
class RingBuffer {
    static_assert(std::is_trivially_copyable_v<T>,
                  "RingBuffer element must be trivially copyable");

public:
    static constexpr size_t kHeaderSize = 4096; // page-aligned header

    // Create or open a ring buffer backed by file `path`
    // `capacity` must be power-of-2; if file exists and matches, reopen
    RingBuffer(const std::string& path, size_t capacity);
    ~RingBuffer();

    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    // Producer: append element, returns sequence number
    // Returns UINT64_MAX on overflow (oldest entry overwritten in circular mode)
    uint64_t push(const T& item) noexcept;

    // Producer: batch push; returns number pushed
    size_t push_batch(const T* items, size_t count) noexcept;

    // Consumer: read element at sequence `seq`
    // Returns false if seq is not available (either not yet written or overwritten)
    bool read(uint64_t seq, T& out) const noexcept;

    // Consumer: get current write sequence (next slot to be written)
    uint64_t write_seq() const noexcept {
        return header_->write_seq.load(std::memory_order_acquire);
    }

    // Consumer: peek at latest written element
    bool peek_latest(T& out) const noexcept {
        uint64_t w = write_seq();
        if (w == 0) return false;
        return read(w - 1, out);
    }

    // Snapshot: copy up to `count` elements starting from `start_seq`
    // Returns number actually read
    size_t snapshot(uint64_t start_seq, T* buf, size_t count) const noexcept;

    size_t   capacity()     const noexcept { return capacity_; }
    uint64_t size()         const noexcept { return write_seq(); }
    bool     empty()        const noexcept { return write_seq() == 0; }
    bool     is_open()      const noexcept { return data_ != nullptr; }

    // Flush mmap to disk
    void sync() noexcept;

private:
    std::string        path_;
    size_t             capacity_;  // number of elements (power of 2)
    size_t             mask_;      // capacity - 1
    size_t             mmap_size_; // total mmap size in bytes

    RingBufferHeader*  header_;
    T*                 data_;      // points into mmap after header

#ifdef _WIN32
    HANDLE file_handle_  {INVALID_HANDLE_VALUE};
    HANDLE mmap_handle_  {NULL};
#else
    int    fd_           {-1};
#endif
    void*  mmap_base_    {nullptr};

    void open_or_create(const std::string& path, size_t capacity);
    void close();
    bool is_readable(uint64_t seq) const noexcept;
};

// ---- Implementation ----

static size_t next_pow2(size_t n) {
    if (n == 0) return 1;
    --n;
    n |= n>>1; n |= n>>2; n |= n>>4; n |= n>>8; n |= n>>16; n |= n>>32;
    return ++n;
}

template <typename T>
RingBuffer<T>::RingBuffer(const std::string& path, size_t capacity)
    : path_(path),
      capacity_(next_pow2(capacity)),
      mask_(capacity_ - 1),
      mmap_size_(kHeaderSize + sizeof(T) * capacity_),
      header_(nullptr), data_(nullptr)
{
    open_or_create(path, capacity_);
}

template <typename T>
RingBuffer<T>::~RingBuffer() {
    sync();
    close();
}

#ifdef _WIN32
template <typename T>
void RingBuffer<T>::open_or_create(const std::string& path, size_t /*cap*/) {
    DWORD access = GENERIC_READ | GENERIC_WRITE;
    DWORD share  = FILE_SHARE_READ | FILE_SHARE_WRITE;
    file_handle_ = CreateFileA(path.c_str(), access, share, nullptr,
                                OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE)
        throw std::runtime_error("RingBuffer: cannot open file: " + path);

    LARGE_INTEGER sz;
    sz.QuadPart = static_cast<LONGLONG>(mmap_size_);
    mmap_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READWRITE,
                                       sz.HighPart, sz.LowPart, nullptr);
    if (!mmap_handle_) {
        CloseHandle(file_handle_);
        throw std::runtime_error("RingBuffer: CreateFileMapping failed");
    }

    mmap_base_ = MapViewOfFile(mmap_handle_, FILE_MAP_ALL_ACCESS, 0, 0, mmap_size_);
    if (!mmap_base_) {
        CloseHandle(mmap_handle_);
        CloseHandle(file_handle_);
        throw std::runtime_error("RingBuffer: MapViewOfFile failed");
    }

    header_ = reinterpret_cast<RingBufferHeader*>(mmap_base_);
    data_   = reinterpret_cast<T*>(static_cast<char*>(mmap_base_) + kHeaderSize);

    // Initialize header if new file
    if (header_->magic != RING_MAGIC) {
        std::memset(mmap_base_, 0, mmap_size_);
        header_->magic        = RING_MAGIC;
        header_->version      = RING_VERSION;
        header_->capacity     = capacity_;
        header_->element_size = sizeof(T);
        header_->write_seq.store(0, std::memory_order_release);
        header_->read_seq.store(0, std::memory_order_release);
    }
}

template <typename T>
void RingBuffer<T>::close() {
    if (mmap_base_) { UnmapViewOfFile(mmap_base_); mmap_base_ = nullptr; }
    if (mmap_handle_) { CloseHandle(mmap_handle_); mmap_handle_ = nullptr; }
    if (file_handle_ != INVALID_HANDLE_VALUE) { CloseHandle(file_handle_); file_handle_ = INVALID_HANDLE_VALUE; }
}

template <typename T>
void RingBuffer<T>::sync() noexcept {
    if (mmap_base_) FlushViewOfFile(mmap_base_, 0);
}

#else // POSIX

template <typename T>
void RingBuffer<T>::open_or_create(const std::string& path, size_t /*cap*/) {
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd_ < 0) throw std::runtime_error("RingBuffer: cannot open: " + path);

    // Extend file to needed size
    if (::ftruncate(fd_, static_cast<off_t>(mmap_size_)) != 0)
        throw std::runtime_error("RingBuffer: ftruncate failed");

    mmap_base_ = ::mmap(nullptr, mmap_size_, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd_, 0);
    if (mmap_base_ == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error("RingBuffer: mmap failed");
    }

    // Advise kernel about access pattern
    ::madvise(mmap_base_, mmap_size_, MADV_SEQUENTIAL);

    header_ = reinterpret_cast<RingBufferHeader*>(mmap_base_);
    data_   = reinterpret_cast<T*>(static_cast<char*>(mmap_base_) + kHeaderSize);

    if (header_->magic != RING_MAGIC) {
        std::memset(mmap_base_, 0, mmap_size_);
        header_->magic        = RING_MAGIC;
        header_->version      = RING_VERSION;
        header_->capacity     = capacity_;
        header_->element_size = sizeof(T);
        header_->write_seq.store(0, std::memory_order_release);
        header_->read_seq.store(0, std::memory_order_release);
    }
}

template <typename T>
void RingBuffer<T>::close() {
    if (mmap_base_ && mmap_base_ != MAP_FAILED) {
        ::munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

template <typename T>
void RingBuffer<T>::sync() noexcept {
    if (mmap_base_ && mmap_base_ != MAP_FAILED)
        ::msync(mmap_base_, mmap_size_, MS_ASYNC);
}
#endif

template <typename T>
uint64_t RingBuffer<T>::push(const T& item) noexcept {
    uint64_t seq = header_->write_seq.fetch_add(1, std::memory_order_acq_rel);
    data_[seq & mask_] = item;
    // Ensure write is visible
    std::atomic_thread_fence(std::memory_order_release);
    return seq;
}

template <typename T>
size_t RingBuffer<T>::push_batch(const T* items, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) push(items[i]);
    return count;
}

template <typename T>
bool RingBuffer<T>::is_readable(uint64_t seq) const noexcept {
    uint64_t w = header_->write_seq.load(std::memory_order_acquire);
    if (seq >= w) return false;
    // Check not overwritten (circular: w - capacity > seq)
    if (w > capacity_ && seq < w - capacity_) return false;
    return true;
}

template <typename T>
bool RingBuffer<T>::read(uint64_t seq, T& out) const noexcept {
    if (!is_readable(seq)) return false;
    std::atomic_thread_fence(std::memory_order_acquire);
    out = data_[seq & mask_];
    // Verify not overwritten during read
    return is_readable(seq);
}

template <typename T>
size_t RingBuffer<T>::snapshot(uint64_t start_seq, T* buf, size_t count) const noexcept {
    size_t n = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!read(start_seq + i, buf[i])) break;
        ++n;
    }
    return n;
}

} // namespace tickstore
