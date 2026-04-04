#pragma once
// Memory pool for Order objects.
// Eliminates per-order heap allocation on the hot path.
// Uses a slab of pre-allocated Order objects with a free-list.

#include "order.hpp"
#include <vector>
#include <cstdint>
#include <cassert>
#include <atomic>
#include <new>
#include <stdexcept>

namespace hft {

// ---- Single-threaded order pool ----
template <size_t Capacity = 131072>
class OrderPool {
public:
    OrderPool() {
        // Initialize free list in reverse so first alloc returns slot 0
        free_list_.reserve(Capacity);
        for (int i = static_cast<int>(Capacity) - 1; i >= 0; --i)
            free_list_.push_back(static_cast<uint32_t>(i));
    }

    Order* alloc() noexcept {
        if (free_list_.empty()) return nullptr;
        uint32_t idx = free_list_.back();
        free_list_.pop_back();
        ++alloc_count_;
        return new (&pool_[idx]) Order();
    }

    Order* alloc(OrderId id, const char* sym, Side side, OrderType type,
                  TimeInForce tif, Price price, Quantity qty, Timestamp ts) noexcept
    {
        Order* o = alloc();
        if (!o) return nullptr;
        o->id         = id;
        std::strncpy(o->symbol, sym, 15);
        o->symbol[15] = '\0';
        o->side       = side;
        o->order_type = type;
        o->tif        = tif;
        o->price      = price;
        o->qty        = qty;
        o->display_qty = qty;
        o->timestamp  = ts;
        return o;
    }

    void free(Order* o) noexcept {
        if (!o) return;
        assert(owns(o));
        uint32_t idx = static_cast<uint32_t>(o - &pool_[0]);
        o->~Order();
        free_list_.push_back(idx);
        ++free_count_;
    }

    bool  owns(const Order* o) const noexcept {
        return o >= &pool_[0] && o < &pool_[Capacity];
    }
    size_t in_use()    const noexcept { return Capacity - free_list_.size(); }
    size_t available() const noexcept { return free_list_.size(); }
    size_t capacity()  const noexcept { return Capacity; }
    uint64_t total_allocs() const noexcept { return alloc_count_; }
    uint64_t total_frees()  const noexcept { return free_count_; }

private:
    alignas(CACHE_LINE) Order pool_[Capacity];
    std::vector<uint32_t> free_list_;
    uint64_t alloc_count_ = 0;
    uint64_t free_count_  = 0;
};

// ---- Thread-safe version using CAS free-list ----
struct AtomicFreeNode {
    uint32_t            next;
    alignas(CACHE_LINE) char pad[CACHE_LINE - 4];
};

template <size_t Capacity = 65536>
class ConcurrentOrderPool {
    static_assert(Capacity > 0 && (Capacity & (Capacity-1)) == 0,
                  "Capacity must be power of 2");
public:
    ConcurrentOrderPool() {
        // Build lock-free free-list stack (Treiber stack)
        for (size_t i = 0; i < Capacity; ++i) {
            nodes_[i].next = (i + 1 < Capacity) ? static_cast<uint32_t>(i+1) : UINT32_MAX;
        }
        top_.store(0, std::memory_order_relaxed);
    }

    Order* alloc() noexcept {
        uint64_t top = top_.load(std::memory_order_acquire);
        for (;;) {
            uint32_t idx = static_cast<uint32_t>(top & 0xFFFFFFFF);
            if (idx == UINT32_MAX) return nullptr; // pool exhausted
            uint32_t next = nodes_[idx].next;
            // ABA prevention: pack version in upper 32 bits
            uint64_t new_top = (static_cast<uint64_t>((top >> 32) + 1) << 32) | next;
            if (top_.compare_exchange_weak(top, new_top,
                    std::memory_order_acq_rel, std::memory_order_acquire))
            {
                ++alloc_count_;
                return new (&pool_[idx]) Order();
            }
        }
    }

    void free(Order* o) noexcept {
        if (!o) return;
        uint32_t idx = static_cast<uint32_t>(o - &pool_[0]);
        o->~Order();
        uint64_t top = top_.load(std::memory_order_acquire);
        for (;;) {
            uint32_t old_top = static_cast<uint32_t>(top & 0xFFFFFFFF);
            nodes_[idx].next = old_top;
            uint64_t new_top = (static_cast<uint64_t>((top >> 32) + 1) << 32) | idx;
            if (top_.compare_exchange_weak(top, new_top,
                    std::memory_order_acq_rel, std::memory_order_acquire))
            {
                ++free_count_;
                return;
            }
        }
    }

    bool owns(const Order* o) const noexcept {
        return o >= &pool_[0] && o < &pool_[Capacity];
    }

private:
    alignas(CACHE_LINE) Order              pool_[Capacity];
    alignas(CACHE_LINE) AtomicFreeNode     nodes_[Capacity];
    alignas(CACHE_LINE) std::atomic<uint64_t> top_{0};
    std::atomic<uint64_t> alloc_count_{0};
    std::atomic<uint64_t> free_count_{0};
};

} // namespace hft
