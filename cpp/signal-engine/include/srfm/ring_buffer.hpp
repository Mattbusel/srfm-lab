#pragma once
#include <atomic>
#include <array>
#include <optional>
#include <cstdint>
#include <cassert>

namespace srfm {

/// Lock-free single-producer single-consumer ring buffer.
/// Capacity must be a power of 2.
/// Uses acquire/release memory ordering for correctness.
/// Cache-line padding separates head and tail to prevent false sharing.
template<typename T, std::size_t Capacity>
class RingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "RingBuffer Capacity must be a power of 2");
    static_assert(Capacity >= 2, "RingBuffer Capacity must be >= 2");

    static constexpr std::size_t MASK = Capacity - 1;

public:
    RingBuffer() noexcept
        : head_(0), tail_(0) {}

    // Non-copyable, non-movable
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    /// Producer: try to push an item. Returns false if full.
    /// Only one thread may call push at a time (single producer).
    [[nodiscard]] bool push(const T& item) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) & MASK;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false; // full
        }
        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    /// Producer: push by move.
    [[nodiscard]] bool push(T&& item) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) & MASK;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        buffer_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    /// Consumer: try to pop an item. Returns nullopt if empty.
    /// Only one thread may call pop at a time (single consumer).
    [[nodiscard]] std::optional<T> pop() noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt; // empty
        }
        T item = buffer_[tail];
        tail_.store((tail + 1) & MASK, std::memory_order_release);
        return item;
    }

    /// Consumer: pop into pre-allocated storage. Returns false if empty.
    [[nodiscard]] bool pop(T& out) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        out = buffer_[tail];
        tail_.store((tail + 1) & MASK, std::memory_order_release);
        return true;
    }

    /// Peek at the front element without consuming. Returns nullptr if empty.
    [[nodiscard]] const T* peek() const noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return nullptr;
        }
        return &buffer_[tail];
    }

    /// Returns true if empty (approximate — check is not atomic).
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    /// Returns approximate number of items in the buffer.
    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        return (h - t + Capacity) & MASK;
    }

    /// Returns maximum capacity (usable slots = Capacity - 1).
    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return Capacity - 1;
    }

    /// Drain all items, calling fn on each. Returns count drained.
    template<typename Fn>
    std::size_t drain(Fn&& fn) noexcept {
        std::size_t count = 0;
        T item;
        while (pop(item)) {
            fn(item);
            ++count;
        }
        return count;
    }

private:
    // Separate head and tail onto different cache lines to eliminate false sharing.
    alignas(64) std::atomic<std::size_t> head_;
    alignas(64) std::atomic<std::size_t> tail_;
    alignas(64) std::array<T, Capacity>  buffer_;
};

// ============================================================
// Convenience aliases for common sizes
// ============================================================

template<typename T> using RingBuffer1K  = RingBuffer<T, 1024>;
template<typename T> using RingBuffer4K  = RingBuffer<T, 4096>;
template<typename T> using RingBuffer16K = RingBuffer<T, 16384>;

// ============================================================
// Multi-producer / single-consumer variant using CAS
// ============================================================

template<typename T, std::size_t Capacity>
class MPSCRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static constexpr std::size_t MASK = Capacity - 1;

    struct Slot {
        alignas(64) std::atomic<std::size_t> seq;
        T data;
    };

public:
    MPSCRingBuffer() noexcept {
        for (std::size_t i = 0; i < Capacity; ++i)
            slots_[i].seq.store(i, std::memory_order_relaxed);
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
    }

    MPSCRingBuffer(const MPSCRingBuffer&) = delete;
    MPSCRingBuffer& operator=(const MPSCRingBuffer&) = delete;

    [[nodiscard]] bool push(const T& item) noexcept {
        std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = slots_[pos & MASK];
            std::size_t seq = slot.seq.load(std::memory_order_acquire);
            std::intptr_t diff = static_cast<std::intptr_t>(seq) -
                                 static_cast<std::intptr_t>(pos);
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed)) {
                    slot.data = item;
                    slot.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false; // full
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
    }

    [[nodiscard]] bool pop(T& out) noexcept {
        std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        Slot& slot = slots_[pos & MASK];
        std::size_t seq = slot.seq.load(std::memory_order_acquire);
        std::intptr_t diff = static_cast<std::intptr_t>(seq) -
                             static_cast<std::intptr_t>(pos + 1);
        if (diff == 0) {
            dequeue_pos_.store(pos + 1, std::memory_order_relaxed);
            out = slot.data;
            slot.seq.store(pos + Capacity, std::memory_order_release);
            return true;
        }
        return false;
    }

    [[nodiscard]] bool empty() const noexcept {
        std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        return slots_[pos & MASK].seq.load(std::memory_order_acquire) != pos + 1;
    }

private:
    alignas(64) std::atomic<std::size_t> enqueue_pos_;
    alignas(64) std::atomic<std::size_t> dequeue_pos_;
    alignas(64) std::array<Slot, Capacity> slots_;
};

} // namespace srfm
