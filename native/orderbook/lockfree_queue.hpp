#pragma once
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <new>

namespace hft {

// Cache line size for x86-64
static constexpr size_t CACHE_LINE = 64;

// Round up to next power of 2
constexpr size_t next_pow2(size_t n) {
    if (n == 0) return 1;
    --n;
    n |= n >> 1; n |= n >> 2; n |= n >> 4;
    n |= n >> 8; n |= n >> 16; n |= n >> 32;
    return ++n;
}

// Single-Producer Single-Consumer lock-free queue
// Uses sequentially consistent loads/stores for head/tail indices
// to avoid ABA issues. Elements are stored in a ring buffer.
template <typename T, size_t Capacity>
class SPSCQueue {
    static_assert(std::is_trivially_copyable_v<T>,
                  "SPSCQueue requires trivially copyable type");
    static constexpr size_t kCapacity = next_pow2(Capacity);
    static constexpr size_t kMask     = kCapacity - 1;

    struct alignas(CACHE_LINE) PaddedAtomic {
        std::atomic<size_t> val{0};
        char _pad[CACHE_LINE - sizeof(std::atomic<size_t>)];
    };

    alignas(CACHE_LINE) T       slots_[kCapacity];
    alignas(CACHE_LINE) PaddedAtomic head_;   // written by consumer
    alignas(CACHE_LINE) PaddedAtomic tail_;   // written by producer
    // cached copies to reduce cross-core traffic
    alignas(CACHE_LINE) size_t  head_cache_{0};
    alignas(CACHE_LINE) size_t  tail_cache_{0};

public:
    SPSCQueue() = default;
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    // Producer side: returns true if enqueued
    bool push(const T& item) noexcept {
        const size_t tail = tail_.val.load(std::memory_order_relaxed);
        const size_t next_tail = (tail + 1) & kMask;
        if (next_tail == head_cache_) {
            head_cache_ = head_.val.load(std::memory_order_acquire);
            if (next_tail == head_cache_) return false; // full
        }
        slots_[tail] = item;
        tail_.val.store(next_tail, std::memory_order_release);
        return true;
    }

    // Producer side: move overload
    bool push(T&& item) noexcept {
        const size_t tail = tail_.val.load(std::memory_order_relaxed);
        const size_t next_tail = (tail + 1) & kMask;
        if (next_tail == head_cache_) {
            head_cache_ = head_.val.load(std::memory_order_acquire);
            if (next_tail == head_cache_) return false;
        }
        slots_[tail] = std::move(item);
        tail_.val.store(next_tail, std::memory_order_release);
        return true;
    }

    // Consumer side: returns true if dequeued
    bool pop(T& item) noexcept {
        const size_t head = head_.val.load(std::memory_order_relaxed);
        if (head == tail_cache_) {
            tail_cache_ = tail_.val.load(std::memory_order_acquire);
            if (head == tail_cache_) return false; // empty
        }
        item = slots_[head];
        head_.val.store((head + 1) & kMask, std::memory_order_release);
        return true;
    }

    // Non-destructive peek
    bool peek(T& item) const noexcept {
        const size_t head = head_.val.load(std::memory_order_relaxed);
        const size_t tail = tail_.val.load(std::memory_order_acquire);
        if (head == tail) return false;
        item = slots_[head];
        return true;
    }

    size_t size() const noexcept {
        const size_t tail = tail_.val.load(std::memory_order_acquire);
        const size_t head = head_.val.load(std::memory_order_acquire);
        return (tail - head + kCapacity) & kMask;
    }

    bool   empty()    const noexcept { return size() == 0; }
    size_t capacity() const noexcept { return kCapacity - 1; }
};

// Multi-Producer Multi-Consumer queue using sequence numbers
// Uses a ring buffer with per-slot sequence counters
template <typename T, size_t Capacity>
class MPMCQueue {
    static constexpr size_t kCapacity = next_pow2(Capacity);
    static constexpr size_t kMask     = kCapacity - 1;

    struct Slot {
        std::atomic<size_t> sequence;
        T                   data;
    };

    alignas(CACHE_LINE) std::atomic<size_t>    head_{0};
    char _pad0[CACHE_LINE - sizeof(std::atomic<size_t>)];
    alignas(CACHE_LINE) std::atomic<size_t>    tail_{0};
    char _pad1[CACHE_LINE - sizeof(std::atomic<size_t>)];
    alignas(CACHE_LINE) Slot                   slots_[kCapacity];

public:
    MPMCQueue() {
        for (size_t i = 0; i < kCapacity; ++i)
            slots_[i].sequence.store(i, std::memory_order_relaxed);
    }

    MPMCQueue(const MPMCQueue&) = delete;
    MPMCQueue& operator=(const MPMCQueue&) = delete;

    bool push(const T& item) noexcept {
        size_t tail = tail_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = slots_[tail & kMask];
            size_t seq = slot.sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(tail);
            if (diff == 0) {
                if (tail_.compare_exchange_weak(tail, tail + 1, std::memory_order_relaxed))
                {
                    slot.data = item;
                    slot.sequence.store(tail + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false; // full
            } else {
                tail = tail_.load(std::memory_order_relaxed);
            }
        }
    }

    bool pop(T& item) noexcept {
        size_t head = head_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = slots_[head & kMask];
            size_t seq = slot.sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(head + 1);
            if (diff == 0) {
                if (head_.compare_exchange_weak(head, head + 1, std::memory_order_relaxed))
                {
                    item = slot.data;
                    slot.sequence.store(head + kCapacity, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false; // empty
            } else {
                head = head_.load(std::memory_order_relaxed);
            }
        }
    }

    size_t size() const noexcept {
        const size_t tail = tail_.load(std::memory_order_acquire);
        const size_t head = head_.load(std::memory_order_acquire);
        return tail > head ? tail - head : 0;
    }
    bool empty() const noexcept { return size() == 0; }
};

// Batch operations for improved throughput
template <typename Queue, typename T>
size_t batch_push(Queue& q, const T* items, size_t count) noexcept {
    size_t pushed = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!q.push(items[i])) break;
        ++pushed;
    }
    return pushed;
}

template <typename Queue, typename T>
size_t batch_pop(Queue& q, T* items, size_t max_count) noexcept {
    size_t popped = 0;
    while (popped < max_count && q.pop(items[popped])) ++popped;
    return popped;
}

} // namespace hft
