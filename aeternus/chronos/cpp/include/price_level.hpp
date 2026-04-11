#pragma once
/// price_level.hpp — Cache-aligned price level with FIFO queue of orders.
///
/// Maintains:
///   - O(1) total qty (maintained incrementally)
///   - FIFO time priority queue (intrusive doubly-linked list)
///   - O(1) front access for matching
///   - O(n) removal by order ID (rare; cancel operations)

#include "lob_types.hpp"
#include <cassert>
#include <cstring>

namespace chronos {

// ── Intrusive List Node ───────────────────────────────────────────────────────

/// Each order in the level queue uses intrusive linking.
/// We embed prev/next order IDs to avoid pointer chasing across allocations.
struct OrderNode {
    Order   order;
    OrderId prev_id;   ///< Previous in FIFO queue (0 = front).
    OrderId next_id;   ///< Next in FIFO queue (0 = back).
};

// ── Price Level ───────────────────────────────────────────────────────────────

/// A single price level in the LOB: FIFO queue.
///
/// Storage: fixed-size array of OrderNode. For production use, replace
/// with a pool allocator. Max orders per level: MAX_ORDERS_PER_LEVEL.
constexpr size_t MAX_ORDERS_PER_LEVEL = 256;

class CACHE_ALIGN PriceLevel {
public:
    TickPrice   price;
    Qty         total_qty;       ///< Sum of all leaves_qty in this level.
    uint32_t    order_count;     ///< Number of resting orders.
    uint32_t    pad;
    Side        side;
    uint8_t     pad2[7];

    PriceLevel() noexcept : price(0), total_qty(0.0), order_count(0), pad(0),
                             side(Side::Bid) {
        std::memset(nodes_, 0, sizeof(nodes_));
        head_id_ = 0;
        tail_id_ = 0;
        next_slot_ = 1;
    }

    explicit PriceLevel(TickPrice p, Side s) noexcept : PriceLevel() {
        price = p;
        side = s;
    }

    bool empty() const noexcept { return order_count == 0; }

    double price_f64() const noexcept { return from_tick(price); }

    // ── FIFO Push ───────────────────────────────────────────────────────────

    /// Add order to the back of the FIFO queue.
    bool push_back(const Order& order) noexcept {
        if (order_count >= MAX_ORDERS_PER_LEVEL) return false;

        uint32_t slot = alloc_slot();
        if (slot == 0) return false;

        OrderNode& node = nodes_[slot];
        node.order = order;
        node.prev_id = tail_id_;
        node.next_id = 0;

        if (tail_id_ != 0) {
            find_node(tail_id_)->next_id = slot;
        } else {
            head_id_ = slot;
        }
        tail_id_ = slot;

        total_qty += order.leaves_qty;
        ++order_count;
        return true;
    }

    // ── Front Access ────────────────────────────────────────────────────────

    Order* front() noexcept {
        if (head_id_ == 0) return nullptr;
        return &nodes_[head_id_].order;
    }

    const Order* front() const noexcept {
        if (head_id_ == 0) return nullptr;
        return &nodes_[head_id_].order;
    }

    /// Apply fill to front order. Pops it if fully filled.
    /// Returns actual fill qty.
    Qty fill_front(Qty fill_qty) noexcept {
        Order* front_order = front();
        if (!front_order) return 0.0;

        Qty actual = front_order->apply_fill(fill_qty);
        total_qty -= actual;

        if (front_order->status == OrderStatus::Filled) {
            pop_front();
        }
        return actual;
    }

    // ── Pop Front ───────────────────────────────────────────────────────────

    void pop_front() noexcept {
        if (head_id_ == 0) return;

        uint32_t old_head = head_id_;
        OrderNode& node = nodes_[old_head];

        head_id_ = node.next_id;
        if (head_id_ != 0) {
            nodes_[head_id_].prev_id = 0;
        } else {
            tail_id_ = 0;
        }

        free_slot(old_head);
        if (order_count > 0) --order_count;
    }

    // ── Remove by ID ────────────────────────────────────────────────────────

    /// Remove order by ID. Returns the removed order (or empty if not found).
    bool remove_by_id(OrderId id, Order& out_order) noexcept {
        uint32_t slot = find_slot_by_order_id(id);
        if (slot == 0) return false;

        OrderNode& node = nodes_[slot];
        out_order = node.order;
        total_qty -= node.order.leaves_qty;

        // Unlink.
        if (node.prev_id != 0) {
            nodes_[node.prev_id].next_id = node.next_id;
        } else {
            head_id_ = node.next_id;
        }
        if (node.next_id != 0) {
            nodes_[node.next_id].prev_id = node.prev_id;
        } else {
            tail_id_ = node.prev_id;
        }

        free_slot(slot);
        if (order_count > 0) --order_count;
        return true;
    }

    // ── Find ────────────────────────────────────────────────────────────────

    Order* find_by_id(OrderId id) noexcept {
        uint32_t slot = find_slot_by_order_id(id);
        return slot != 0 ? &nodes_[slot].order : nullptr;
    }

    /// Queue position (0-based, FIFO order). Returns -1 if not found.
    int queue_position(OrderId id) const noexcept {
        int pos = 0;
        uint32_t cur = head_id_;
        while (cur != 0) {
            if (nodes_[cur].order.id == id) return pos;
            cur = nodes_[cur].next_id;
            ++pos;
        }
        return -1;
    }

    /// Qty ahead of a given order (sum of leaves_qty before it in queue).
    Qty qty_ahead(OrderId id) const noexcept {
        Qty ahead = 0.0;
        uint32_t cur = head_id_;
        while (cur != 0) {
            if (nodes_[cur].order.id == id) return ahead;
            ahead += nodes_[cur].order.leaves_qty;
            cur = nodes_[cur].next_id;
        }
        return -1.0; // Not found.
    }

    // ── Iteration ───────────────────────────────────────────────────────────

    /// Iterate over all orders in FIFO order, calling `fn(order)`.
    template <typename Fn>
    void for_each(Fn&& fn) const noexcept {
        uint32_t cur = head_id_;
        while (cur != 0) {
            fn(nodes_[cur].order);
            cur = nodes_[cur].next_id;
        }
    }

    template <typename Fn>
    void for_each_mut(Fn&& fn) noexcept {
        uint32_t cur = head_id_;
        while (cur != 0) {
            uint32_t next = nodes_[cur].next_id;
            fn(nodes_[cur].order);
            cur = next;
        }
    }

    // ── Level summary ────────────────────────────────────────────────────────

    Level to_level() const noexcept {
        return Level(price, total_qty, order_count);
    }

private:
    // Fixed-size node pool. Index 0 is reserved as null.
    OrderNode nodes_[MAX_ORDERS_PER_LEVEL + 1];
    uint32_t head_id_;
    uint32_t tail_id_;
    uint32_t next_slot_;  ///< Next free slot (simple bump allocator + freelist).
    // Small free list for released slots.
    uint32_t free_list_[MAX_ORDERS_PER_LEVEL];
    uint32_t free_list_top_;

    uint32_t alloc_slot() noexcept {
        if (free_list_top_ > 0) {
            return free_list_[--free_list_top_];
        }
        if (next_slot_ <= MAX_ORDERS_PER_LEVEL) {
            return next_slot_++;
        }
        return 0; // Out of slots.
    }

    void free_slot(uint32_t slot) noexcept {
        if (slot == 0) return;
        std::memset(&nodes_[slot], 0, sizeof(OrderNode));
        if (free_list_top_ < MAX_ORDERS_PER_LEVEL) {
            free_list_[free_list_top_++] = slot;
        }
    }

    OrderNode* find_node(uint32_t slot) noexcept {
        return &nodes_[slot];
    }

    uint32_t find_slot_by_order_id(OrderId id) const noexcept {
        uint32_t cur = head_id_;
        while (cur != 0) {
            if (nodes_[cur].order.id == id) return cur;
            cur = nodes_[cur].next_id;
        }
        return 0;
    }
};

} // namespace chronos
