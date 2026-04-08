#pragma once
#include <cstdint>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <deque>
#include <limits>
#include <cassert>

namespace srfm::regime {

enum class MarketRegime : uint8_t {
    Bull = 0, Bear = 1, Sideways = 2, Unknown = 3
};

enum class VolRegimeState : uint8_t {
    Low = 0, Normal = 1, High = 2, Crisis = 3
};

enum class TrendState : uint8_t {
    StrongUp = 0, WeakUp = 1, Neutral = 2, WeakDown = 3, StrongDown = 4
};

enum class CorrelationState : uint8_t {
    Dispersed = 0, Normal = 1, Herding = 2
};

enum class MomentumState : uint8_t {
    StrongPositive = 0, Positive = 1, Neutral = 2, Negative = 3, StrongNegative = 4
};

struct RegimeProbs {
    double bull;
    double bear;
    double sideways;
    MarketRegime most_likely() const {
        if (bull >= bear && bull >= sideways) return MarketRegime::Bull;
        if (bear >= bull && bear >= sideways) return MarketRegime::Bear;
        return MarketRegime::Sideways;
    }
};

// ----------- HMM Regime Classifier -----------
class HMMRegime {
public:
    static constexpr int N_STATES = 3;
    static constexpr int N_OBS_BINS = 16;

    struct Params {
        double transition[N_STATES][N_STATES];
        double emission_mean[N_STATES];
        double emission_var[N_STATES];
        double initial[N_STATES];
    };

    HMMRegime();
    explicit HMMRegime(const Params& p);

    void update(double observation);
    MarketRegime current_state() const;
    RegimeProbs state_probabilities() const;
    std::vector<MarketRegime> viterbi(const std::vector<double>& observations) const;
    void reset();
    const Params& params() const { return params_; }
    void set_params(const Params& p) { params_ = p; }
    uint64_t update_count() const { return count_; }

private:
    double gaussian_pdf(double x, double mu, double sigma2) const;
    void forward_step(double obs);

    Params params_;
    double alpha_[N_STATES];
    double scaling_;
    uint64_t count_;
    MarketRegime current_;
};

// ----------- Vol Regime Classifier -----------
class VolRegime {
public:
    struct Config {
        double ewma_lambda = 0.94;
        double low_threshold = 0.08;
        double normal_threshold = 0.16;
        double high_threshold = 0.30;
        int history_window = 252;
        double annualization = 252.0;
    };

    VolRegime();
    explicit VolRegime(const Config& cfg);

    void update(double log_return);
    VolRegimeState current_state() const { return state_; }
    double current_vol() const { return std::sqrt(variance_ * config_.annualization); }
    double ewma_variance() const { return variance_; }
    double percentile_rank() const;
    void reset();
    uint64_t update_count() const { return count_; }

private:
    Config config_;
    double variance_;
    VolRegimeState state_;
    std::deque<double> vol_history_;
    uint64_t count_;
};

// ----------- Trend Regime Classifier -----------
class TrendRegime {
public:
    struct Config {
        int fast_ma_period = 50;
        int slow_ma_period = 200;
        int adx_period = 14;
        int regression_window = 60;
        double strong_threshold = 25.0;
        double weak_threshold = 15.0;
    };

    TrendRegime();
    explicit TrendRegime(const Config& cfg);

    void update(double price, double high, double low, double close);
    TrendState current_state() const { return state_; }
    double fast_ma() const { return fast_ma_; }
    double slow_ma() const { return slow_ma_; }
    double adx_value() const { return adx_; }
    double r_squared() const { return r2_; }
    bool is_trending() const { return adx_ > config_.weak_threshold; }
    void reset();
    uint64_t update_count() const { return count_; }

private:
    void update_ma(double price);
    void update_adx(double high, double low, double close);
    void update_regression(double price);

    Config config_;
    TrendState state_;
    double fast_ma_, slow_ma_;
    double fast_sum_, slow_sum_;
    std::deque<double> fast_window_, slow_window_;
    double adx_, plus_di_, minus_di_;
    double atr_;
    double prev_high_, prev_low_, prev_close_;
    double smoothed_plus_dm_, smoothed_minus_dm_, smoothed_tr_;
    double dx_sum_;
    std::deque<double> dx_history_;
    double r2_;
    std::deque<double> regression_window_;
    uint64_t count_;
};

// ----------- Correlation Regime Classifier -----------
class CorrelationRegime {
public:
    struct Config {
        int window = 60;
        int n_assets = 10;
        double herding_threshold = 0.7;
        double dispersed_threshold = 0.3;
    };

    CorrelationRegime();
    explicit CorrelationRegime(const Config& cfg);

    void update(const std::vector<double>& returns);
    CorrelationState current_state() const { return state_; }
    double average_correlation() const { return avg_corr_; }
    double dispersion() const { return dispersion_; }
    bool is_herding() const { return avg_corr_ > config_.herding_threshold; }
    const std::vector<std::vector<double>>& correlation_matrix() const { return corr_matrix_; }
    void reset();

private:
    void recompute_correlation();

    Config config_;
    CorrelationState state_;
    double avg_corr_;
    double dispersion_;
    std::vector<std::deque<double>> return_history_;
    std::vector<std::vector<double>> corr_matrix_;
    uint64_t count_;
};

// ----------- Momentum Regime Classifier -----------
class MomentumRegime {
public:
    struct Config {
        int lookback_period = 252;
        int skip_period = 21;
        int n_assets = 10;
        double strong_threshold = 1.5;
        double threshold = 0.5;
    };

    MomentumRegime();
    explicit MomentumRegime(const Config& cfg);

    void update(const std::vector<double>& prices);
    MomentumState current_state() const { return state_; }
    double average_momentum() const { return avg_momentum_; }
    double cross_sectional_rank() const { return cs_rank_; }
    void reset();

private:
    Config config_;
    MomentumState state_;
    double avg_momentum_;
    double cs_rank_;
    std::vector<std::deque<double>> price_history_;
    uint64_t count_;
};

// ----------- Regime Ensemble -----------
class RegimeEnsemble {
public:
    struct Config {
        double hmm_weight = 0.30;
        double vol_weight = 0.20;
        double trend_weight = 0.20;
        double corr_weight = 0.15;
        double momentum_weight = 0.15;
        int hysteresis_count = 3;
    };

    RegimeEnsemble();
    explicit RegimeEnsemble(const Config& cfg);

    void update(MarketRegime hmm, VolRegimeState vol, TrendState trend,
                CorrelationState corr, MomentumState mom);
    MarketRegime current_regime() const { return confirmed_; }
    RegimeProbs probabilities() const { return probs_; }
    int confirmations() const { return confirm_count_; }
    void reset();

private:
    RegimeProbs compute_probs(MarketRegime hmm, VolRegimeState vol,
                              TrendState trend, CorrelationState corr,
                              MomentumState mom) const;

    Config config_;
    MarketRegime confirmed_;
    MarketRegime candidate_;
    int confirm_count_;
    RegimeProbs probs_;
    uint64_t count_;
};

// ----------- Regime Transition Matrix -----------
class RegimeTransition {
public:
    static constexpr int N_STATES = 3;

    RegimeTransition();

    void update(MarketRegime from, MarketRegime to);
    double transition_prob(MarketRegime from, MarketRegime to) const;
    double expected_duration(MarketRegime state) const;
    void get_matrix(double out[N_STATES][N_STATES]) const;
    MarketRegime most_likely_next(MarketRegime current) const;
    void reset();
    uint64_t total_transitions() const { return total_; }

private:
    uint64_t counts_[N_STATES][N_STATES];
    uint64_t row_totals_[N_STATES];
    uint64_t total_;
};

// ----------- Online Baum-Welch Parameter Estimator -----------
class OnlineBaumWelch {
public:
    static constexpr int N_STATES = 3;

    struct Config {
        double learning_rate = 0.01;
        int min_samples = 100;
        double min_variance = 1e-6;
    };

    OnlineBaumWelch();
    explicit OnlineBaumWelch(const Config& cfg);

    void update(double observation, HMMRegime& hmm);
    uint64_t iteration_count() const { return iter_; }
    void reset();

private:
    Config config_;
    double gamma_sum_[N_STATES];
    double gamma_obs_sum_[N_STATES];
    double gamma_obs2_sum_[N_STATES];
    double xi_sum_[N_STATES][N_STATES];
    uint64_t iter_;
};

// ----------- Regime Duration Tracker -----------
class RegimeDurationTracker {
public:
    RegimeDurationTracker();

    void update(MarketRegime regime);
    int current_duration() const { return current_dur_; }
    double avg_duration(MarketRegime regime) const;
    int max_duration(MarketRegime regime) const;
    int min_duration(MarketRegime regime) const;
    double median_duration(MarketRegime regime) const;
    int total_transitions() const { return n_transitions_; }
    void reset();

private:
    MarketRegime current_;
    int current_dur_;
    int n_transitions_;
    std::vector<int> durations_[4]; // per regime
};

// ----------- Regime Signal Generator -----------
class RegimeSignalGenerator {
public:
    struct Config {
        double bull_signal = 1.0;
        double bear_signal = -1.0;
        double sideways_signal = 0.0;
        double transition_alpha = 0.3;
        bool smooth_transitions = true;
    };

    RegimeSignalGenerator();
    explicit RegimeSignalGenerator(const Config& cfg);

    void update(MarketRegime regime, const RegimeProbs& probs);
    double signal() const { return signal_; }
    double raw_signal() const { return raw_signal_; }
    double regime_confidence() const;
    void reset();

private:
    Config config_;
    double signal_;
    double raw_signal_;
    RegimeProbs last_probs_;
};

// ----------- Composite Regime Manager -----------
class CompositeRegimeManager {
public:
    struct Config {
        HMMRegime::Params hmm_params;
        VolRegime::Config vol_config;
        TrendRegime::Config trend_config;
        CorrelationRegime::Config corr_config;
        MomentumRegime::Config momentum_config;
        RegimeEnsemble::Config ensemble_config;
        OnlineBaumWelch::Config bw_config;
        RegimeSignalGenerator::Config signal_config;
        bool enable_online_learning = true;
    };

    CompositeRegimeManager();
    explicit CompositeRegimeManager(const Config& cfg);

    struct UpdateInput {
        double log_return;
        double price;
        double high;
        double low;
        double close;
        std::vector<double> asset_returns;
        std::vector<double> asset_prices;
    };

    struct UpdateOutput {
        MarketRegime regime;
        RegimeProbs probs;
        VolRegimeState vol_state;
        TrendState trend_state;
        CorrelationState corr_state;
        MomentumState momentum_state;
        double signal;
        double confidence;
        int regime_duration;
        double transition_prob_to_bull;
        double transition_prob_to_bear;
        double expected_regime_duration;
    };

    UpdateOutput update(const UpdateInput& input);
    MarketRegime current_regime() const;
    void reset();
    uint64_t update_count() const { return count_; }

private:
    HMMRegime hmm_;
    VolRegime vol_;
    TrendRegime trend_;
    CorrelationRegime corr_;
    MomentumRegime momentum_;
    RegimeEnsemble ensemble_;
    RegimeTransition transition_;
    OnlineBaumWelch bw_;
    RegimeDurationTracker duration_;
    RegimeSignalGenerator signal_gen_;
    MarketRegime prev_regime_;
    uint64_t count_;
    bool enable_learning_;
};

// ----------- Regime Performance Tracker -----------
class RegimePerformanceTracker {
public:
    RegimePerformanceTracker();

    void update(MarketRegime regime, double return_value);
    double avg_return(MarketRegime regime) const;
    double total_return(MarketRegime regime) const;
    double sharpe(MarketRegime regime) const;
    double max_drawdown(MarketRegime regime) const;
    int count(MarketRegime regime) const;
    void reset();

private:
    struct Stats {
        double sum_returns = 0;
        double sum_sq_returns = 0;
        double peak = 0;
        double drawdown = 0;
        double max_dd = 0;
        double cumulative = 0;
        int n = 0;
    };
    Stats stats_[4];
};

// ----------- Regime Backtest Engine -----------
class RegimeBacktester {
public:
    struct TradeResult {
        MarketRegime entry_regime;
        MarketRegime exit_regime;
        double entry_price;
        double exit_price;
        double pnl;
        int holding_period;
    };

    RegimeBacktester();

    void add_observation(double price, double log_return, MarketRegime regime);
    double regime_edge(MarketRegime regime) const;
    double hit_rate(MarketRegime regime) const;
    const std::vector<TradeResult>& trades() const { return trades_; }
    double total_pnl() const;
    void reset();

private:
    struct Observation {
        double price;
        double log_return;
        MarketRegime regime;
    };
    std::vector<Observation> history_;
    std::vector<TradeResult> trades_;
    bool in_trade_;
    double entry_price_;
    MarketRegime entry_regime_;
    int hold_count_;
};

} // namespace srfm::regime
