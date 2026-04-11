#pragma once
/// matching_engine.hpp — Matching engine interface.
///
/// Pure interface (abstract base) for the LOB matching engine.
/// Concrete implementations in matching_engine.cpp.

#include "lob_types.hpp"
#include "price_level.hpp"
#include <functional>
#include <vector>
#include <cstdint>

namespace chronos {

// ── Callback Types ────────────────────────────────────────────────────────────

/// Called for every fill generated during matching.
using FillCallback = std::function<void(const Fill&)>;

/// Called when a resting order is cancelled (by request or IOC expiry).
using CancelCallback = std::function<void(OrderId, InstId)>;

// ── Matching Engine Interface ─────────────────────────────────────────────────

class IMatchingEngine {
public:
    virtual ~IMatchingEngine() = default;

    /// Submit a new order. May generate fills.
    virtual void submit(Order& order) = 0;

    /// Cancel a resting order. Returns true if found and cancelled.
    virtual bool cancel(OrderId order_id) = 0;

    /// Modify price and/or quantity of a resting order (cancel-replace).
    /// Returns true if found and replaced.
    virtual bool modify(OrderId order_id, TickPrice new_price, Qty new_qty) = 0;

    /// Get current best bid/ask.
    virtual TickPrice best_bid() const noexcept = 0;
    virtual TickPrice best_ask() const noexcept = 0;

    /// Get market snapshot (top N levels).
    virtual MarketSnapshot snapshot(size_t depth = DEPTH_LEVELS) const noexcept = 0;

    /// VWAP of sweeping qty on given side.
    virtual double vwap_sweep(Side side, Qty qty) const noexcept = 0;

    /// Register fill callback.
    virtual void set_fill_callback(FillCallback cb) = 0;

    /// Register cancel callback.
    virtual void set_cancel_callback(CancelCallback cb) = 0;

    /// Total fill count.
    virtual uint64_t fill_count() const noexcept = 0;

    /// Order count (resting).
    virtual size_t order_count() const noexcept = 0;
};

// ── Factory ───────────────────────────────────────────────────────────────────

/// Create a price-time priority matching engine for given instrument.
std::unique_ptr<IMatchingEngine> create_matching_engine(InstId instrument_id);

} // namespace chronos
