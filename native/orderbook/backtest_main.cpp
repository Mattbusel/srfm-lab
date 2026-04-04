#include "backtester.hpp"
#include "strategy_engine.hpp"
#include <iostream>
#include <ctime>

using namespace hft;

// Simple mean-reversion strategy for backtesting
class MeanReversionStrategy : public Strategy {
public:
    struct Params {
        size_t   lookback   = 20;
        double   entry_z    = 1.5;
        double   exit_z     = 0.2;
        Quantity trade_qty  = 100;
        double   tick_size  = 0.01;
    };

    explicit MeanReversionStrategy(const Params& p = {}) : p_(p) {}
    std::string name() const override { return "MeanReversion"; }

    std::vector<Signal> on_book_update(const Context& ctx) override {
        const auto& f = ctx.features;
        if (f.mid_price == 0) return {};

        price_history_.push_back(f.mid_price);
        if (price_history_.size() > p_.lookback) price_history_.pop_front();
        if (price_history_.size() < 5) return {};

        // Compute z-score of current price
        double mean = 0.0, var = 0.0;
        for (auto p : price_history_) mean += p;
        mean /= price_history_.size();
        for (auto p : price_history_) var += (p - mean) * (p - mean);
        var /= price_history_.size();
        double std_dev = std::sqrt(var);
        if (std_dev < 1e-8) return {};

        double z = (f.mid_price - mean) / std_dev;

        std::vector<Signal> signals;

        if (!in_position_) {
            if (z < -p_.entry_z) {
                // Price below mean: buy
                Signal sig{};
                sig.type  = SignalType::BuyLimit;
                sig.price = double_to_price(f.mid_price - p_.tick_size);
                sig.qty   = p_.trade_qty;
                sig.tif   = TimeInForce::GTC;
                sig.confidence = std::min(std::fabs(z) / p_.entry_z, 1.0);
                sig.signal_ts  = ctx.timestamp_ns;
                position_dir_  = 1;
                in_position_   = true;
                signals.push_back(sig);
            } else if (z > p_.entry_z) {
                // Price above mean: sell
                Signal sig{};
                sig.type  = SignalType::SellLimit;
                sig.price = double_to_price(f.mid_price + p_.tick_size);
                sig.qty   = p_.trade_qty;
                sig.tif   = TimeInForce::GTC;
                sig.confidence = std::min(std::fabs(z) / p_.entry_z, 1.0);
                sig.signal_ts  = ctx.timestamp_ns;
                position_dir_  = -1;
                in_position_   = true;
                signals.push_back(sig);
            }
        } else {
            bool exit = (position_dir_ == 1 && z > -p_.exit_z) ||
                        (position_dir_ == -1 && z < p_.exit_z);
            if (exit) {
                Signal sig{};
                sig.type  = (position_dir_ == 1) ? SignalType::SellMkt : SignalType::BuyMkt;
                sig.qty   = p_.trade_qty;
                sig.confidence = 1.0;
                sig.signal_ts  = ctx.timestamp_ns;
                in_position_   = false;
                position_dir_  = 0;
                signals.push_back(sig);
            }
        }
        return signals;
    }

    void reset() override {
        price_history_.clear();
        in_position_ = false;
        position_dir_ = 0;
    }

private:
    Params p_;
    std::deque<double> price_history_;
    bool in_position_ = false;
    int  position_dir_ = 0;
};

int main() {
    std::cout << "HFT Backtester\n" << std::string(50,'=') << std::endl;

    // Generate synthetic events
    SyntheticEventGenerator::Config gen_cfg{};
    gen_cfg.symbol       = "SYNTH";
    gen_cfg.initial_price = 150.0;
    gen_cfg.daily_vol    = 0.25;
    gen_cfg.lambda_new   = 1000.0;
    gen_cfg.lambda_cancel = 800.0;
    gen_cfg.lambda_trade = 200.0;
    gen_cfg.num_events   = 200000;
    gen_cfg.seed         = 42;

    std::cout << "Generating " << gen_cfg.num_events << " synthetic market events..." << std::endl;
    SyntheticEventGenerator gen(gen_cfg);
    auto events = gen.generate();
    std::cout << "Generated " << events.size() << " events." << std::endl;

    // Run backtest with Market Maker strategy
    {
        std::cout << "\n--- Market Maker Backtest ---" << std::endl;
        Backtester::Config bt_cfg{};
        bt_cfg.fee_per_share = 0.001;
        bt_cfg.slippage_bps  = 0.5;
        bt_cfg.initial_capital = 1000000.0;

        MarketMakerStrategy::Params mm_params{};
        mm_params.base_half_spread = 2.0;
        mm_params.quote_qty        = 100;

        MarketMakerStrategy mm(mm_params);
        Backtester bt("SYNTH", bt_cfg);
        auto result = bt.run(events, mm);
        result.print();
    }

    // Run backtest with Mean Reversion strategy
    {
        std::cout << "\n--- Mean Reversion Backtest ---" << std::endl;
        Backtester::Config bt_cfg{};
        bt_cfg.fee_per_share = 0.001;
        bt_cfg.slippage_bps  = 1.0;
        bt_cfg.initial_capital = 1000000.0;

        MeanReversionStrategy::Params mr_params{};
        mr_params.lookback   = 30;
        mr_params.entry_z    = 1.5;
        mr_params.exit_z     = 0.3;
        mr_params.trade_qty  = 200;

        MeanReversionStrategy mr(mr_params);
        Backtester bt("SYNTH", bt_cfg);
        auto result = bt.run(events, mr);
        result.print();
    }

    return 0;
}
