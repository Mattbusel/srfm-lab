#pragma once
// Price ladder: array-based O(1) price level lookup for a contiguous price range.
// Much faster than std::map for instruments with bounded tick ranges.
// Falls back to hash-map for prices outside the pre-allocated range.

#include "order.hpp"
#include <vector>
#include <unordered_map>
#include <optional>
#include <cassert>
#include <cstring>

namespace hft {

// Array-based price ladder for price levels.
// Priced in ticks from a base price. O(1) lookup/insert/remove.
class PriceLadder {
public:
    // num_ticks: number of tick slots on each side of base
    // tick_size: price increment (fixed-point)
    PriceLadder(Price base_price, Price tick_size, size_t num_ticks)
        : base_(base_price),
          tick_(tick_size),
          num_ticks_(num_ticks),
          slots_(2 * num_ticks + 1),
          slot_active_(2 * num_ticks + 1, false)
    {
        assert(tick_size > 0);
        for (size_t i = 0; i < slots_.size(); ++i)
            slots_[i].price = base_ + (static_cast<Price>(i) - static_cast<Price>(num_ticks_)) * tick_;
    }

    // Convert price to array index; returns -1 if out of range
    int price_to_idx(Price p) const noexcept {
        Price delta = p - base_;
        if (delta % tick_ != 0) return -1; // not on-tick
        int64_t idx = delta / tick_ + static_cast<int64_t>(num_ticks_);
        if (idx < 0 || static_cast<size_t>(idx) >= slots_.size()) return -1;
        return static_cast<int>(idx);
    }

    PriceLevel* get(Price p) noexcept {
        int idx = price_to_idx(p);
        if (idx < 0) {
            auto it = overflow_.find(p);
            return it != overflow_.end() ? &it->second : nullptr;
        }
        return slot_active_[idx] ? &slots_[idx] : nullptr;
    }

    const PriceLevel* get(Price p) const noexcept {
        int idx = price_to_idx(p);
        if (idx < 0) {
            auto it = overflow_.find(p);
            return it != overflow_.end() ? &it->second : nullptr;
        }
        return slot_active_[idx] ? &slots_[idx] : nullptr;
    }

    PriceLevel& get_or_create(Price p) {
        int idx = price_to_idx(p);
        if (idx >= 0) {
            if (!slot_active_[idx]) {
                slots_[idx] = PriceLevel(p);
                slot_active_[idx] = true;
                active_prices_.push_back(idx);
            }
            return slots_[idx];
        }
        // Overflow
        auto it = overflow_.find(p);
        if (it == overflow_.end()) {
            overflow_.emplace(p, PriceLevel(p));
            return overflow_[p];
        }
        return it->second;
    }

    void remove(Price p) {
        int idx = price_to_idx(p);
        if (idx >= 0 && slot_active_[idx]) {
            slots_[idx] = PriceLevel(p); // reset
            slot_active_[idx] = false;
            active_prices_.erase(
                std::find(active_prices_.begin(), active_prices_.end(), idx));
        } else {
            overflow_.erase(p);
        }
    }

    // Best bid (highest active price below or at base+n)
    std::optional<Price> best_bid(Price max_price) const noexcept {
        int max_idx = price_to_idx(max_price);
        if (max_idx < 0) max_idx = static_cast<int>(slots_.size()) - 1;

        for (int i = max_idx; i >= 0; --i) {
            if (slot_active_[i] && !slots_[i].empty())
                return slots_[i].price;
        }
        // Check overflow
        Price best = 0;
        bool found = false;
        for (auto& [p, lvl] : overflow_) {
            if (!lvl.empty() && p <= price_to_idx(max_price) + base_ &&
                (!found || p > best))
            { best = p; found = true; }
        }
        return found ? std::optional<Price>(best) : std::nullopt;
    }

    // Best ask (lowest active price above min_price)
    std::optional<Price> best_ask(Price min_price) const noexcept {
        int min_idx = price_to_idx(min_price);
        if (min_idx < 0) min_idx = 0;

        for (int i = min_idx; i < static_cast<int>(slots_.size()); ++i) {
            if (slot_active_[i] && !slots_[i].empty())
                return slots_[i].price;
        }
        Price best = 0;
        bool found = false;
        for (auto& [p, lvl] : overflow_) {
            if (!lvl.empty() && (!found || p < best))
            { best = p; found = true; }
        }
        return found ? std::optional<Price>(best) : std::nullopt;
    }

    size_t active_level_count() const noexcept {
        size_t cnt = active_prices_.size();
        for (auto& [p, lvl] : overflow_) if (!lvl.empty()) ++cnt;
        return cnt;
    }

    void recenter(Price new_base) {
        // Shift the window. Clears overflowed prices that are now in range.
        // Simple approach: rebuild — expensive but rare.
        PriceLadder new_ladder(new_base, tick_, num_ticks_);
        for (int i : active_prices_) {
            if (!slots_[i].empty()) {
                auto& nl = new_ladder.get_or_create(slots_[i].price);
                nl = slots_[i];
            }
        }
        for (auto& [p, lvl] : overflow_) {
            if (!lvl.empty()) {
                auto& nl = new_ladder.get_or_create(p);
                nl = lvl;
            }
        }
        *this = std::move(new_ladder);
    }

private:
    Price                                       base_;
    Price                                       tick_;
    size_t                                      num_ticks_;
    std::vector<PriceLevel>                     slots_;
    std::vector<bool>                           slot_active_;
    std::vector<int>                            active_prices_; // active slot indices
    std::unordered_map<Price, PriceLevel>       overflow_;
};

} // namespace hft
