#pragma once
#include <cstdint>
#include <string>
#include <chrono>

namespace hft {

enum class Side : uint8_t {
    Buy  = 0,
    Sell = 1
};

enum class OrderType : uint8_t {
    Limit   = 0,
    Market  = 1,
    Stop    = 2,
    Iceberg = 3
};

enum class TimeInForce : uint8_t {
    GTC = 0,  // Good Till Cancel
    IOC = 1,  // Immediate Or Cancel
    FOK = 2,  // Fill Or Kill
    GTD = 3,  // Good Till Date
    DAY = 4   // Day order
};

enum class OrderStatus : uint8_t {
    New         = 0,
    PartialFill = 1,
    Filled      = 2,
    Cancelled   = 3,
    Rejected    = 4,
    Expired     = 5
};

using OrderId    = uint64_t;
using Price      = int64_t;   // fixed-point: actual_price * PRICE_SCALE
using Quantity   = uint64_t;
using Timestamp  = int64_t;   // nanoseconds since epoch

static constexpr int64_t PRICE_SCALE = 100000; // 5 decimal places

inline double price_to_double(Price p) { return static_cast<double>(p) / PRICE_SCALE; }
inline Price  double_to_price(double d) { return static_cast<Price>(d * PRICE_SCALE + 0.5); }

struct Order {
    OrderId      id;
    char         symbol[16];
    Side         side;
    OrderType    order_type;
    TimeInForce  tif;
    OrderStatus  status;
    Price        price;
    Price        stop_price;      // for stop orders
    Quantity     qty;
    Quantity     filled_qty;
    Quantity     display_qty;     // for iceberg: visible quantity
    Quantity     hidden_qty;      // for iceberg: hidden reserve
    Timestamp    timestamp;       // order entry time (ns)
    Timestamp    expiry_time;     // for GTD
    uint32_t     trader_id;
    uint32_t     venue_id;
    uint64_t     sequence_num;

    // doubly-linked list pointers within price level
    Order*       prev;
    Order*       next;

    Order() noexcept
        : id(0), side(Side::Buy), order_type(OrderType::Limit),
          tif(TimeInForce::GTC), status(OrderStatus::New),
          price(0), stop_price(0), qty(0), filled_qty(0),
          display_qty(0), hidden_qty(0), timestamp(0), expiry_time(0),
          trader_id(0), venue_id(0), sequence_num(0),
          prev(nullptr), next(nullptr)
    {
        symbol[0] = '\0';
    }

    Order(OrderId id_, const char* sym, Side side_, OrderType type_,
          TimeInForce tif_, Price price_, Quantity qty_, Timestamp ts_) noexcept
        : id(id_), side(side_), order_type(type_), tif(tif_),
          status(OrderStatus::New), price(price_), stop_price(0),
          qty(qty_), filled_qty(0), display_qty(qty_), hidden_qty(0),
          timestamp(ts_), expiry_time(0), trader_id(0), venue_id(0),
          sequence_num(0), prev(nullptr), next(nullptr)
    {
        std::strncpy(symbol, sym, 15);
        symbol[15] = '\0';
    }

    Quantity leaves_qty() const noexcept { return qty - filled_qty; }
    bool     is_active()  const noexcept {
        return status == OrderStatus::New || status == OrderStatus::PartialFill;
    }
    bool     is_buy()  const noexcept { return side == Side::Buy; }
    bool     is_sell() const noexcept { return side == Side::Sell; }

    void set_symbol(const std::string& s) noexcept {
        std::strncpy(symbol, s.c_str(), 15);
        symbol[15] = '\0';
    }
};

struct Trade {
    uint64_t  trade_id;
    OrderId   aggressor_id;
    OrderId   passive_id;
    char      symbol[16];
    Side      aggressor_side;
    Price     price;
    Quantity  qty;
    Timestamp timestamp;
    uint64_t  sequence_num;

    Trade() noexcept
        : trade_id(0), aggressor_id(0), passive_id(0),
          aggressor_side(Side::Buy), price(0), qty(0),
          timestamp(0), sequence_num(0)
    { symbol[0] = '\0'; }
};

struct PriceLevel {
    Price    price;
    Quantity total_qty;
    uint32_t order_count;
    Order*   head;   // front of queue (oldest = first to match)
    Order*   tail;   // back of queue  (newest = last to match)

    PriceLevel() noexcept
        : price(0), total_qty(0), order_count(0),
          head(nullptr), tail(nullptr) {}

    explicit PriceLevel(Price p) noexcept
        : price(p), total_qty(0), order_count(0),
          head(nullptr), tail(nullptr) {}

    void push_back(Order* o) noexcept {
        o->prev = tail;
        o->next = nullptr;
        if (tail) tail->next = o;
        else      head = o;
        tail = o;
        total_qty += o->leaves_qty();
        ++order_count;
    }

    void remove(Order* o) noexcept {
        if (o->prev) o->prev->next = o->next;
        else         head = o->next;
        if (o->next) o->next->prev = o->prev;
        else         tail = o->prev;
        total_qty -= o->leaves_qty();
        --order_count;
        o->prev = o->next = nullptr;
    }

    bool empty() const noexcept { return head == nullptr; }
};

} // namespace hft
