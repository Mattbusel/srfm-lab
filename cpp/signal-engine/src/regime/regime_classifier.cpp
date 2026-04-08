#include "regime_classifier.hpp"
#include <cstring>
#include <cfloat>
#include <immintrin.h>

namespace srfm::regime {

// ============================================================================
// HMMRegime Implementation
// ============================================================================

static constexpr double PI = 3.14159265358979323846;
static constexpr double TWO_PI = 2.0 * PI;
static constexpr double LOG_2PI = 1.8378770664093453;
static constexpr double MIN_PROB = 1e-300;

HMMRegime::HMMRegime() : scaling_(1.0), count_(0), current_(MarketRegime::Unknown) {
    // Default params: bull=positive mean/low var, bear=negative/high var, sideways=zero/med var
    params_.transition[0][0] = 0.95; params_.transition[0][1] = 0.025; params_.transition[0][2] = 0.025;
    params_.transition[1][0] = 0.025; params_.transition[1][1] = 0.95; params_.transition[1][2] = 0.025;
    params_.transition[2][0] = 0.03; params_.transition[2][1] = 0.03; params_.transition[2][2] = 0.94;
    params_.emission_mean[0] = 0.0005;   // bull daily return
    params_.emission_mean[1] = -0.0005;  // bear daily return
    params_.emission_mean[2] = 0.0;      // sideways
    params_.emission_var[0] = 0.0001;    // low vol
    params_.emission_var[1] = 0.0004;    // high vol
    params_.emission_var[2] = 0.00015;   // medium vol
    params_.initial[0] = 1.0 / 3.0;
    params_.initial[1] = 1.0 / 3.0;
    params_.initial[2] = 1.0 / 3.0;
    for (int i = 0; i < N_STATES; ++i) alpha_[i] = params_.initial[i];
}

HMMRegime::HMMRegime(const Params& p) : params_(p), scaling_(1.0), count_(0), current_(MarketRegime::Unknown) {
    for (int i = 0; i < N_STATES; ++i) alpha_[i] = params_.initial[i];
}

double HMMRegime::gaussian_pdf(double x, double mu, double sigma2) const {
    double diff = x - mu;
    double exponent = -0.5 * diff * diff / sigma2;
    return std::exp(exponent) / std::sqrt(TWO_PI * sigma2);
}

void HMMRegime::forward_step(double obs) {
    double new_alpha[N_STATES];
    double sum = 0.0;
    for (int j = 0; j < N_STATES; ++j) {
        double trans_sum = 0.0;
        for (int i = 0; i < N_STATES; ++i) {
            trans_sum += alpha_[i] * params_.transition[i][j];
        }
        double emit = gaussian_pdf(obs, params_.emission_mean[j], params_.emission_var[j]);
        new_alpha[j] = trans_sum * emit;
        sum += new_alpha[j];
    }
    scaling_ = (sum > MIN_PROB) ? sum : MIN_PROB;
    for (int j = 0; j < N_STATES; ++j) {
        alpha_[j] = new_alpha[j] / scaling_;
    }
}

void HMMRegime::update(double observation) {
    if (count_ == 0) {
        for (int i = 0; i < N_STATES; ++i) {
            alpha_[i] = params_.initial[i] * gaussian_pdf(observation, params_.emission_mean[i], params_.emission_var[i]);
        }
        double sum = 0.0;
        for (int i = 0; i < N_STATES; ++i) sum += alpha_[i];
        if (sum > MIN_PROB) {
            for (int i = 0; i < N_STATES; ++i) alpha_[i] /= sum;
        }
    } else {
        forward_step(observation);
    }
    ++count_;
    int best = 0;
    for (int i = 1; i < N_STATES; ++i) {
        if (alpha_[i] > alpha_[best]) best = i;
    }
    current_ = static_cast<MarketRegime>(best);
}

MarketRegime HMMRegime::current_state() const { return current_; }

RegimeProbs HMMRegime::state_probabilities() const {
    return { alpha_[0], alpha_[1], alpha_[2] };
}

std::vector<MarketRegime> HMMRegime::viterbi(const std::vector<double>& observations) const {
    const int T = static_cast<int>(observations.size());
    if (T == 0) return {};

    // delta[t][j] = max probability of path ending in state j at time t (log domain)
    std::vector<std::array<double, N_STATES>> delta(T);
    std::vector<std::array<int, N_STATES>> psi(T);

    // Initialize
    for (int j = 0; j < N_STATES; ++j) {
        double emit = gaussian_pdf(observations[0], params_.emission_mean[j], params_.emission_var[j]);
        delta[0][j] = std::log(params_.initial[j] + MIN_PROB) + std::log(emit + MIN_PROB);
        psi[0][j] = 0;
    }

    // Recurse
    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < N_STATES; ++j) {
            double best_val = -std::numeric_limits<double>::infinity();
            int best_i = 0;
            for (int i = 0; i < N_STATES; ++i) {
                double val = delta[t - 1][i] + std::log(params_.transition[i][j] + MIN_PROB);
                if (val > best_val) { best_val = val; best_i = i; }
            }
            double emit = gaussian_pdf(observations[t], params_.emission_mean[j], params_.emission_var[j]);
            delta[t][j] = best_val + std::log(emit + MIN_PROB);
            psi[t][j] = best_i;
        }
    }

    // Backtrack
    std::vector<MarketRegime> path(T);
    int best = 0;
    for (int j = 1; j < N_STATES; ++j) {
        if (delta[T - 1][j] > delta[T - 1][best]) best = j;
    }
    path[T - 1] = static_cast<MarketRegime>(best);
    for (int t = T - 2; t >= 0; --t) {
        best = psi[t + 1][best];
        path[t] = static_cast<MarketRegime>(best);
    }
    return path;
}

void HMMRegime::reset() {
    for (int i = 0; i < N_STATES; ++i) alpha_[i] = params_.initial[i];
    scaling_ = 1.0;
    count_ = 0;
    current_ = MarketRegime::Unknown;
}

// ============================================================================
// VolRegime Implementation
// ============================================================================

VolRegime::VolRegime() : config_(), variance_(0.0002), state_(VolRegimeState::Normal), count_(0) {}

VolRegime::VolRegime(const Config& cfg) : config_(cfg), variance_(0.0002), state_(VolRegimeState::Normal), count_(0) {}

void VolRegime::update(double log_return) {
    if (count_ == 0) {
        variance_ = log_return * log_return;
    } else {
        variance_ = config_.ewma_lambda * variance_ + (1.0 - config_.ewma_lambda) * log_return * log_return;
    }
    ++count_;

    double ann_vol = std::sqrt(variance_ * config_.annualization);
    vol_history_.push_back(ann_vol);
    if (static_cast<int>(vol_history_.size()) > config_.history_window) {
        vol_history_.pop_front();
    }

    if (ann_vol < config_.low_threshold) state_ = VolRegimeState::Low;
    else if (ann_vol < config_.normal_threshold) state_ = VolRegimeState::Normal;
    else if (ann_vol < config_.high_threshold) state_ = VolRegimeState::High;
    else state_ = VolRegimeState::Crisis;
}

double VolRegime::percentile_rank() const {
    if (vol_history_.size() < 2) return 0.5;
    double current = vol_history_.back();
    int below = 0;
    int total = static_cast<int>(vol_history_.size());
    for (const auto& v : vol_history_) {
        if (v < current) ++below;
    }
    return static_cast<double>(below) / static_cast<double>(total);
}

void VolRegime::reset() {
    variance_ = 0.0002;
    state_ = VolRegimeState::Normal;
    vol_history_.clear();
    count_ = 0;
}

// ============================================================================
// TrendRegime Implementation
// ============================================================================

TrendRegime::TrendRegime()
    : config_(), state_(TrendState::Neutral), fast_ma_(0), slow_ma_(0),
      fast_sum_(0), slow_sum_(0), adx_(0), plus_di_(0), minus_di_(0), atr_(0),
      prev_high_(0), prev_low_(0), prev_close_(0),
      smoothed_plus_dm_(0), smoothed_minus_dm_(0), smoothed_tr_(0),
      dx_sum_(0), r2_(0), count_(0) {}

TrendRegime::TrendRegime(const Config& cfg)
    : config_(cfg), state_(TrendState::Neutral), fast_ma_(0), slow_ma_(0),
      fast_sum_(0), slow_sum_(0), adx_(0), plus_di_(0), minus_di_(0), atr_(0),
      prev_high_(0), prev_low_(0), prev_close_(0),
      smoothed_plus_dm_(0), smoothed_minus_dm_(0), smoothed_tr_(0),
      dx_sum_(0), r2_(0), count_(0) {}

void TrendRegime::update_ma(double price) {
    fast_window_.push_back(price);
    fast_sum_ += price;
    if (static_cast<int>(fast_window_.size()) > config_.fast_ma_period) {
        fast_sum_ -= fast_window_.front();
        fast_window_.pop_front();
    }
    fast_ma_ = fast_sum_ / static_cast<double>(fast_window_.size());

    slow_window_.push_back(price);
    slow_sum_ += price;
    if (static_cast<int>(slow_window_.size()) > config_.slow_ma_period) {
        slow_sum_ -= slow_window_.front();
        slow_window_.pop_front();
    }
    slow_ma_ = slow_sum_ / static_cast<double>(slow_window_.size());
}

void TrendRegime::update_adx(double high, double low, double close) {
    if (count_ == 0) {
        prev_high_ = high; prev_low_ = low; prev_close_ = close;
        return;
    }

    double up_move = high - prev_high_;
    double down_move = prev_low_ - low;
    double plus_dm = (up_move > down_move && up_move > 0) ? up_move : 0.0;
    double minus_dm = (down_move > up_move && down_move > 0) ? down_move : 0.0;

    double tr1 = high - low;
    double tr2 = std::abs(high - prev_close_);
    double tr3 = std::abs(low - prev_close_);
    double tr = std::max({tr1, tr2, tr3});

    int period = config_.adx_period;
    if (count_ <= static_cast<uint64_t>(period)) {
        smoothed_plus_dm_ += plus_dm;
        smoothed_minus_dm_ += minus_dm;
        smoothed_tr_ += tr;
        if (count_ == static_cast<uint64_t>(period)) {
            // First smoothed values are just sums
        }
    } else {
        smoothed_plus_dm_ = smoothed_plus_dm_ - smoothed_plus_dm_ / period + plus_dm;
        smoothed_minus_dm_ = smoothed_minus_dm_ - smoothed_minus_dm_ / period + minus_dm;
        smoothed_tr_ = smoothed_tr_ - smoothed_tr_ / period + tr;
    }

    if (count_ >= static_cast<uint64_t>(period) && smoothed_tr_ > 0) {
        plus_di_ = 100.0 * smoothed_plus_dm_ / smoothed_tr_;
        minus_di_ = 100.0 * smoothed_minus_dm_ / smoothed_tr_;
        double di_sum = plus_di_ + minus_di_;
        double dx = (di_sum > 0) ? 100.0 * std::abs(plus_di_ - minus_di_) / di_sum : 0.0;

        dx_history_.push_back(dx);
        dx_sum_ += dx;
        if (static_cast<int>(dx_history_.size()) > period) {
            dx_sum_ -= dx_history_.front();
            dx_history_.pop_front();
        }
        adx_ = dx_sum_ / static_cast<double>(dx_history_.size());
    }

    atr_ = (smoothed_tr_ > 0 && count_ >= static_cast<uint64_t>(period))
           ? smoothed_tr_ / period : tr;

    prev_high_ = high; prev_low_ = low; prev_close_ = close;
}

void TrendRegime::update_regression(double price) {
    regression_window_.push_back(price);
    if (static_cast<int>(regression_window_.size()) > config_.regression_window) {
        regression_window_.pop_front();
    }
    int n = static_cast<int>(regression_window_.size());
    if (n < 10) { r2_ = 0; return; }

    // y = a + b*x, compute R^2
    double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    int idx = 0;
    for (const auto& y : regression_window_) {
        double x = static_cast<double>(idx);
        sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y;
        ++idx;
    }
    double dn = static_cast<double>(n);
    double denom = dn * sxx - sx * sx;
    if (std::abs(denom) < 1e-15) { r2_ = 0; return; }
    double b = (dn * sxy - sx * sy) / denom;
    double a = (sy - b * sx) / dn;

    double ss_res = 0, ss_tot = 0;
    double y_mean = sy / dn;
    idx = 0;
    for (const auto& y : regression_window_) {
        double y_hat = a + b * static_cast<double>(idx);
        ss_res += (y - y_hat) * (y - y_hat);
        ss_tot += (y - y_mean) * (y - y_mean);
        ++idx;
    }
    r2_ = (ss_tot > 1e-15) ? 1.0 - ss_res / ss_tot : 0.0;
}

void TrendRegime::update(double price, double high, double low, double close) {
    update_ma(price);
    update_adx(high, low, close);
    update_regression(price);
    ++count_;

    // Determine trend state
    bool bullish_cross = fast_ma_ > slow_ma_;
    if (adx_ > config_.strong_threshold) {
        state_ = bullish_cross ? TrendState::StrongUp : TrendState::StrongDown;
    } else if (adx_ > config_.weak_threshold) {
        state_ = bullish_cross ? TrendState::WeakUp : TrendState::WeakDown;
    } else {
        state_ = TrendState::Neutral;
    }
}

void TrendRegime::reset() {
    state_ = TrendState::Neutral;
    fast_ma_ = slow_ma_ = 0;
    fast_sum_ = slow_sum_ = 0;
    fast_window_.clear(); slow_window_.clear();
    adx_ = plus_di_ = minus_di_ = 0;
    atr_ = 0;
    prev_high_ = prev_low_ = prev_close_ = 0;
    smoothed_plus_dm_ = smoothed_minus_dm_ = smoothed_tr_ = 0;
    dx_sum_ = 0;
    dx_history_.clear();
    r2_ = 0;
    regression_window_.clear();
    count_ = 0;
}

// ============================================================================
// CorrelationRegime Implementation
// ============================================================================

CorrelationRegime::CorrelationRegime()
    : config_(), state_(CorrelationState::Normal), avg_corr_(0), dispersion_(0), count_(0) {
    return_history_.resize(config_.n_assets);
    corr_matrix_.assign(config_.n_assets, std::vector<double>(config_.n_assets, 0.0));
}

CorrelationRegime::CorrelationRegime(const Config& cfg)
    : config_(cfg), state_(CorrelationState::Normal), avg_corr_(0), dispersion_(0), count_(0) {
    return_history_.resize(cfg.n_assets);
    corr_matrix_.assign(cfg.n_assets, std::vector<double>(cfg.n_assets, 0.0));
}

void CorrelationRegime::update(const std::vector<double>& returns) {
    int n = std::min(static_cast<int>(returns.size()), config_.n_assets);
    for (int i = 0; i < n; ++i) {
        return_history_[i].push_back(returns[i]);
        if (static_cast<int>(return_history_[i].size()) > config_.window) {
            return_history_[i].pop_front();
        }
    }
    ++count_;
    if (static_cast<int>(return_history_[0].size()) >= 10) {
        recompute_correlation();
    }
}

void CorrelationRegime::recompute_correlation() {
    int n = config_.n_assets;
    int w = static_cast<int>(return_history_[0].size());

    // Compute means
    std::vector<double> means(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (const auto& r : return_history_[i]) means[i] += r;
        means[i] /= w;
    }

    // Compute variances and covariances
    std::vector<double> vars(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (const auto& r : return_history_[i]) {
            double d = r - means[i];
            vars[i] += d * d;
        }
    }

    double sum_corr = 0.0;
    int pair_count = 0;
    for (int i = 0; i < n; ++i) {
        corr_matrix_[i][i] = 1.0;
        for (int j = i + 1; j < n; ++j) {
            double cov = 0.0;
            auto it_i = return_history_[i].begin();
            auto it_j = return_history_[j].begin();
            for (int k = 0; k < w; ++k, ++it_i, ++it_j) {
                cov += (*it_i - means[i]) * (*it_j - means[j]);
            }
            double denom = std::sqrt(vars[i] * vars[j]);
            double corr = (denom > 1e-15) ? cov / denom : 0.0;
            corr = std::clamp(corr, -1.0, 1.0);
            corr_matrix_[i][j] = corr;
            corr_matrix_[j][i] = corr;
            sum_corr += corr;
            ++pair_count;
        }
    }

    avg_corr_ = (pair_count > 0) ? sum_corr / pair_count : 0.0;

    // Dispersion = std dev of pairwise correlations
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double d = corr_matrix_[i][j] - avg_corr_;
            sum_sq += d * d;
        }
    }
    dispersion_ = (pair_count > 1) ? std::sqrt(sum_sq / pair_count) : 0.0;

    if (avg_corr_ > config_.herding_threshold) state_ = CorrelationState::Herding;
    else if (avg_corr_ < config_.dispersed_threshold) state_ = CorrelationState::Dispersed;
    else state_ = CorrelationState::Normal;
}

void CorrelationRegime::reset() {
    state_ = CorrelationState::Normal;
    avg_corr_ = dispersion_ = 0;
    for (auto& h : return_history_) h.clear();
    for (auto& row : corr_matrix_) std::fill(row.begin(), row.end(), 0.0);
    count_ = 0;
}

// ============================================================================
// MomentumRegime Implementation
// ============================================================================

MomentumRegime::MomentumRegime()
    : config_(), state_(MomentumState::Neutral), avg_momentum_(0), cs_rank_(0.5), count_(0) {
    price_history_.resize(config_.n_assets);
}

MomentumRegime::MomentumRegime(const Config& cfg)
    : config_(cfg), state_(MomentumState::Neutral), avg_momentum_(0), cs_rank_(0.5), count_(0) {
    price_history_.resize(cfg.n_assets);
}

void MomentumRegime::update(const std::vector<double>& prices) {
    int n = std::min(static_cast<int>(prices.size()), config_.n_assets);
    for (int i = 0; i < n; ++i) {
        price_history_[i].push_back(prices[i]);
        if (static_cast<int>(price_history_[i].size()) > config_.lookback_period + 1) {
            price_history_[i].pop_front();
        }
    }
    ++count_;

    if (static_cast<int>(price_history_[0].size()) < config_.lookback_period + 1) return;

    // 12-1 month momentum: return from t-lookback to t-skip
    std::vector<double> momentums(n);
    for (int i = 0; i < n; ++i) {
        int sz = static_cast<int>(price_history_[i].size());
        double p_old = price_history_[i][0]; // oldest
        int skip_idx = sz - 1 - config_.skip_period;
        if (skip_idx < 0) skip_idx = 0;
        double p_recent = price_history_[i][skip_idx];
        momentums[i] = (p_old > 0) ? (p_recent / p_old - 1.0) : 0.0;
    }

    double sum = 0;
    for (auto m : momentums) sum += m;
    avg_momentum_ = sum / n;

    // Cross-sectional rank of asset 0
    if (n > 1) {
        std::vector<double> sorted = momentums;
        std::sort(sorted.begin(), sorted.end());
        auto it = std::lower_bound(sorted.begin(), sorted.end(), momentums[0]);
        int rank = static_cast<int>(it - sorted.begin());
        cs_rank_ = static_cast<double>(rank) / static_cast<double>(n - 1);
    } else {
        cs_rank_ = 0.5;
    }

    if (avg_momentum_ > config_.strong_threshold * 0.01) state_ = MomentumState::StrongPositive;
    else if (avg_momentum_ > config_.threshold * 0.01) state_ = MomentumState::Positive;
    else if (avg_momentum_ < -config_.strong_threshold * 0.01) state_ = MomentumState::StrongNegative;
    else if (avg_momentum_ < -config_.threshold * 0.01) state_ = MomentumState::Negative;
    else state_ = MomentumState::Neutral;
}

void MomentumRegime::reset() {
    state_ = MomentumState::Neutral;
    avg_momentum_ = 0;
    cs_rank_ = 0.5;
    for (auto& h : price_history_) h.clear();
    count_ = 0;
}

// ============================================================================
// RegimeEnsemble Implementation
// ============================================================================

RegimeEnsemble::RegimeEnsemble()
    : config_(), confirmed_(MarketRegime::Unknown), candidate_(MarketRegime::Unknown),
      confirm_count_(0), probs_{1.0/3, 1.0/3, 1.0/3}, count_(0) {}

RegimeEnsemble::RegimeEnsemble(const Config& cfg)
    : config_(cfg), confirmed_(MarketRegime::Unknown), candidate_(MarketRegime::Unknown),
      confirm_count_(0), probs_{1.0/3, 1.0/3, 1.0/3}, count_(0) {}

RegimeProbs RegimeEnsemble::compute_probs(MarketRegime hmm, VolRegimeState vol,
                                           TrendState trend, CorrelationState corr,
                                           MomentumState mom) const {
    double bull = 0, bear = 0, side = 0;

    // HMM contribution
    switch (hmm) {
        case MarketRegime::Bull: bull += config_.hmm_weight; break;
        case MarketRegime::Bear: bear += config_.hmm_weight; break;
        default: side += config_.hmm_weight; break;
    }

    // Vol contribution
    switch (vol) {
        case VolRegimeState::Low: bull += config_.vol_weight * 0.6; side += config_.vol_weight * 0.4; break;
        case VolRegimeState::Normal: side += config_.vol_weight * 0.5; bull += config_.vol_weight * 0.3; bear += config_.vol_weight * 0.2; break;
        case VolRegimeState::High: bear += config_.vol_weight * 0.6; side += config_.vol_weight * 0.2; bull += config_.vol_weight * 0.2; break;
        case VolRegimeState::Crisis: bear += config_.vol_weight * 0.9; side += config_.vol_weight * 0.1; break;
    }

    // Trend contribution
    switch (trend) {
        case TrendState::StrongUp: bull += config_.trend_weight; break;
        case TrendState::WeakUp: bull += config_.trend_weight * 0.6; side += config_.trend_weight * 0.4; break;
        case TrendState::Neutral: side += config_.trend_weight; break;
        case TrendState::WeakDown: bear += config_.trend_weight * 0.6; side += config_.trend_weight * 0.4; break;
        case TrendState::StrongDown: bear += config_.trend_weight; break;
    }

    // Correlation contribution
    switch (corr) {
        case CorrelationState::Herding: bear += config_.corr_weight * 0.7; side += config_.corr_weight * 0.3; break;
        case CorrelationState::Normal: side += config_.corr_weight * 0.4; bull += config_.corr_weight * 0.3; bear += config_.corr_weight * 0.3; break;
        case CorrelationState::Dispersed: bull += config_.corr_weight * 0.6; side += config_.corr_weight * 0.4; break;
    }

    // Momentum contribution
    switch (mom) {
        case MomentumState::StrongPositive: bull += config_.momentum_weight; break;
        case MomentumState::Positive: bull += config_.momentum_weight * 0.7; side += config_.momentum_weight * 0.3; break;
        case MomentumState::Neutral: side += config_.momentum_weight; break;
        case MomentumState::Negative: bear += config_.momentum_weight * 0.7; side += config_.momentum_weight * 0.3; break;
        case MomentumState::StrongNegative: bear += config_.momentum_weight; break;
    }

    double total = bull + bear + side;
    if (total > 0) { bull /= total; bear /= total; side /= total; }
    return { bull, bear, side };
}

void RegimeEnsemble::update(MarketRegime hmm, VolRegimeState vol, TrendState trend,
                             CorrelationState corr, MomentumState mom) {
    probs_ = compute_probs(hmm, vol, trend, corr, mom);
    MarketRegime raw = probs_.most_likely();
    ++count_;

    if (confirmed_ == MarketRegime::Unknown) {
        confirmed_ = raw;
        candidate_ = raw;
        confirm_count_ = 1;
        return;
    }

    if (raw == confirmed_) {
        candidate_ = confirmed_;
        confirm_count_ = config_.hysteresis_count;
    } else if (raw == candidate_) {
        ++confirm_count_;
        if (confirm_count_ >= config_.hysteresis_count) {
            confirmed_ = candidate_;
        }
    } else {
        candidate_ = raw;
        confirm_count_ = 1;
    }
}

void RegimeEnsemble::reset() {
    confirmed_ = MarketRegime::Unknown;
    candidate_ = MarketRegime::Unknown;
    confirm_count_ = 0;
    probs_ = {1.0/3, 1.0/3, 1.0/3};
    count_ = 0;
}

// ============================================================================
// RegimeTransition Implementation
// ============================================================================

RegimeTransition::RegimeTransition() : total_(0) {
    std::memset(counts_, 0, sizeof(counts_));
    std::memset(row_totals_, 0, sizeof(row_totals_));
}

void RegimeTransition::update(MarketRegime from, MarketRegime to) {
    int f = static_cast<int>(from);
    int t = static_cast<int>(to);
    if (f >= N_STATES || t >= N_STATES) return;
    ++counts_[f][t];
    ++row_totals_[f];
    ++total_;
}

double RegimeTransition::transition_prob(MarketRegime from, MarketRegime to) const {
    int f = static_cast<int>(from);
    int t = static_cast<int>(to);
    if (f >= N_STATES || t >= N_STATES) return 0.0;
    if (row_totals_[f] == 0) return 1.0 / N_STATES;
    return static_cast<double>(counts_[f][t]) / static_cast<double>(row_totals_[f]);
}

double RegimeTransition::expected_duration(MarketRegime state) const {
    int s = static_cast<int>(state);
    if (s >= N_STATES || row_totals_[s] == 0) return 0.0;
    double p_stay = static_cast<double>(counts_[s][s]) / static_cast<double>(row_totals_[s]);
    return (1.0 - p_stay > 1e-10) ? 1.0 / (1.0 - p_stay) : 1e6;
}

void RegimeTransition::get_matrix(double out[N_STATES][N_STATES]) const {
    for (int i = 0; i < N_STATES; ++i) {
        for (int j = 0; j < N_STATES; ++j) {
            out[i][j] = transition_prob(static_cast<MarketRegime>(i), static_cast<MarketRegime>(j));
        }
    }
}

MarketRegime RegimeTransition::most_likely_next(MarketRegime current) const {
    int c = static_cast<int>(current);
    if (c >= N_STATES || row_totals_[c] == 0) return current;
    int best = 0;
    for (int j = 1; j < N_STATES; ++j) {
        if (counts_[c][j] > counts_[c][best]) best = j;
    }
    return static_cast<MarketRegime>(best);
}

void RegimeTransition::reset() {
    std::memset(counts_, 0, sizeof(counts_));
    std::memset(row_totals_, 0, sizeof(row_totals_));
    total_ = 0;
}

// ============================================================================
// OnlineBaumWelch Implementation
// ============================================================================

OnlineBaumWelch::OnlineBaumWelch() : config_(), iter_(0) {
    std::memset(gamma_sum_, 0, sizeof(gamma_sum_));
    std::memset(gamma_obs_sum_, 0, sizeof(gamma_obs_sum_));
    std::memset(gamma_obs2_sum_, 0, sizeof(gamma_obs2_sum_));
    std::memset(xi_sum_, 0, sizeof(xi_sum_));
}

OnlineBaumWelch::OnlineBaumWelch(const Config& cfg) : config_(cfg), iter_(0) {
    std::memset(gamma_sum_, 0, sizeof(gamma_sum_));
    std::memset(gamma_obs_sum_, 0, sizeof(gamma_obs_sum_));
    std::memset(gamma_obs2_sum_, 0, sizeof(gamma_obs2_sum_));
    std::memset(xi_sum_, 0, sizeof(xi_sum_));
}

void OnlineBaumWelch::update(double observation, HMMRegime& hmm) {
    ++iter_;
    if (iter_ < static_cast<uint64_t>(config_.min_samples)) return;

    auto probs = hmm.state_probabilities();
    double gamma[N_STATES] = { probs.bull, probs.bear, probs.sideways };

    double lr = config_.learning_rate;

    // Accumulate sufficient statistics with exponential forgetting
    for (int i = 0; i < N_STATES; ++i) {
        gamma_sum_[i] = (1.0 - lr) * gamma_sum_[i] + lr * gamma[i];
        gamma_obs_sum_[i] = (1.0 - lr) * gamma_obs_sum_[i] + lr * gamma[i] * observation;
        gamma_obs2_sum_[i] = (1.0 - lr) * gamma_obs2_sum_[i] + lr * gamma[i] * observation * observation;
    }

    // Update emission parameters
    auto params = hmm.params();
    for (int i = 0; i < N_STATES; ++i) {
        if (gamma_sum_[i] > 1e-10) {
            double new_mean = gamma_obs_sum_[i] / gamma_sum_[i];
            double new_var = gamma_obs2_sum_[i] / gamma_sum_[i] - new_mean * new_mean;
            new_var = std::max(new_var, config_.min_variance);

            params.emission_mean[i] = (1.0 - lr) * params.emission_mean[i] + lr * new_mean;
            params.emission_var[i] = (1.0 - lr) * params.emission_var[i] + lr * new_var;
        }
    }

    // Update transition matrix using xi statistics
    // Approximate: use product of consecutive gammas
    double sum_trans[N_STATES] = {};
    for (int i = 0; i < N_STATES; ++i) {
        for (int j = 0; j < N_STATES; ++j) {
            double xi_approx = gamma[i] * params.transition[i][j] * gamma[j];
            xi_sum_[i][j] = (1.0 - lr) * xi_sum_[i][j] + lr * xi_approx;
            sum_trans[i] += xi_sum_[i][j];
        }
    }
    for (int i = 0; i < N_STATES; ++i) {
        if (sum_trans[i] > 1e-10) {
            for (int j = 0; j < N_STATES; ++j) {
                params.transition[i][j] = xi_sum_[i][j] / sum_trans[i];
            }
        }
    }

    hmm.set_params(params);
}

void OnlineBaumWelch::reset() {
    std::memset(gamma_sum_, 0, sizeof(gamma_sum_));
    std::memset(gamma_obs_sum_, 0, sizeof(gamma_obs_sum_));
    std::memset(gamma_obs2_sum_, 0, sizeof(gamma_obs2_sum_));
    std::memset(xi_sum_, 0, sizeof(xi_sum_));
    iter_ = 0;
}

// ============================================================================
// RegimeDurationTracker Implementation
// ============================================================================

RegimeDurationTracker::RegimeDurationTracker()
    : current_(MarketRegime::Unknown), current_dur_(0), n_transitions_(0) {}

void RegimeDurationTracker::update(MarketRegime regime) {
    if (regime == current_) {
        ++current_dur_;
    } else {
        if (current_ != MarketRegime::Unknown && current_dur_ > 0) {
            durations_[static_cast<int>(current_)].push_back(current_dur_);
            ++n_transitions_;
        }
        current_ = regime;
        current_dur_ = 1;
    }
}

double RegimeDurationTracker::avg_duration(MarketRegime regime) const {
    int r = static_cast<int>(regime);
    if (r >= 4 || durations_[r].empty()) return 0;
    double sum = 0;
    for (auto d : durations_[r]) sum += d;
    return sum / durations_[r].size();
}

int RegimeDurationTracker::max_duration(MarketRegime regime) const {
    int r = static_cast<int>(regime);
    if (r >= 4 || durations_[r].empty()) return 0;
    return *std::max_element(durations_[r].begin(), durations_[r].end());
}

int RegimeDurationTracker::min_duration(MarketRegime regime) const {
    int r = static_cast<int>(regime);
    if (r >= 4 || durations_[r].empty()) return 0;
    return *std::min_element(durations_[r].begin(), durations_[r].end());
}

double RegimeDurationTracker::median_duration(MarketRegime regime) const {
    int r = static_cast<int>(regime);
    if (r >= 4 || durations_[r].empty()) return 0;
    auto sorted = durations_[r];
    std::sort(sorted.begin(), sorted.end());
    size_t mid = sorted.size() / 2;
    if (sorted.size() % 2 == 0) return (sorted[mid - 1] + sorted[mid]) / 2.0;
    return sorted[mid];
}

void RegimeDurationTracker::reset() {
    current_ = MarketRegime::Unknown;
    current_dur_ = 0;
    n_transitions_ = 0;
    for (auto& d : durations_) d.clear();
}

// ============================================================================
// RegimeSignalGenerator Implementation
// ============================================================================

RegimeSignalGenerator::RegimeSignalGenerator()
    : config_(), signal_(0), raw_signal_(0), last_probs_{1.0/3, 1.0/3, 1.0/3} {}

RegimeSignalGenerator::RegimeSignalGenerator(const Config& cfg)
    : config_(cfg), signal_(0), raw_signal_(0), last_probs_{1.0/3, 1.0/3, 1.0/3} {}

void RegimeSignalGenerator::update(MarketRegime regime, const RegimeProbs& probs) {
    last_probs_ = probs;

    // Probability-weighted signal
    raw_signal_ = probs.bull * config_.bull_signal
                + probs.bear * config_.bear_signal
                + probs.sideways * config_.sideways_signal;

    if (config_.smooth_transitions) {
        signal_ = config_.transition_alpha * raw_signal_ + (1.0 - config_.transition_alpha) * signal_;
    } else {
        switch (regime) {
            case MarketRegime::Bull: signal_ = config_.bull_signal; break;
            case MarketRegime::Bear: signal_ = config_.bear_signal; break;
            case MarketRegime::Sideways: signal_ = config_.sideways_signal; break;
            default: break;
        }
    }
}

double RegimeSignalGenerator::regime_confidence() const {
    double max_p = std::max({last_probs_.bull, last_probs_.bear, last_probs_.sideways});
    return max_p; // confidence = probability of most likely state
}

void RegimeSignalGenerator::reset() {
    signal_ = raw_signal_ = 0;
    last_probs_ = {1.0/3, 1.0/3, 1.0/3};
}

// ============================================================================
// CompositeRegimeManager Implementation
// ============================================================================

CompositeRegimeManager::CompositeRegimeManager()
    : prev_regime_(MarketRegime::Unknown), count_(0), enable_learning_(true) {}

CompositeRegimeManager::CompositeRegimeManager(const Config& cfg)
    : hmm_(cfg.hmm_params), vol_(cfg.vol_config), trend_(cfg.trend_config),
      corr_(cfg.corr_config), momentum_(cfg.momentum_config),
      ensemble_(cfg.ensemble_config), bw_(cfg.bw_config),
      signal_gen_(cfg.signal_config), prev_regime_(MarketRegime::Unknown),
      count_(0), enable_learning_(cfg.enable_online_learning) {}

CompositeRegimeManager::UpdateOutput CompositeRegimeManager::update(const UpdateInput& input) {
    // Update all sub-classifiers
    hmm_.update(input.log_return);
    vol_.update(input.log_return);
    trend_.update(input.price, input.high, input.low, input.close);

    if (!input.asset_returns.empty()) {
        corr_.update(input.asset_returns);
    }
    if (!input.asset_prices.empty()) {
        momentum_.update(input.asset_prices);
    }

    // Online parameter learning
    if (enable_learning_) {
        bw_.update(input.log_return, hmm_);
    }

    // Ensemble vote
    ensemble_.update(hmm_.current_state(), vol_.current_state(),
                     trend_.current_state(), corr_.current_state(),
                     momentum_.current_state());

    MarketRegime regime = ensemble_.current_regime();

    // Track transitions
    if (prev_regime_ != MarketRegime::Unknown) {
        transition_.update(prev_regime_, regime);
    }

    // Track duration
    duration_.update(regime);

    // Generate signal
    signal_gen_.update(regime, ensemble_.probabilities());

    prev_regime_ = regime;
    ++count_;

    UpdateOutput out{};
    out.regime = regime;
    out.probs = ensemble_.probabilities();
    out.vol_state = vol_.current_state();
    out.trend_state = trend_.current_state();
    out.corr_state = corr_.current_state();
    out.momentum_state = momentum_.current_state();
    out.signal = signal_gen_.signal();
    out.confidence = signal_gen_.regime_confidence();
    out.regime_duration = duration_.current_duration();
    out.transition_prob_to_bull = transition_.transition_prob(regime, MarketRegime::Bull);
    out.transition_prob_to_bear = transition_.transition_prob(regime, MarketRegime::Bear);
    out.expected_regime_duration = transition_.expected_duration(regime);

    return out;
}

MarketRegime CompositeRegimeManager::current_regime() const {
    return ensemble_.current_regime();
}

void CompositeRegimeManager::reset() {
    hmm_.reset(); vol_.reset(); trend_.reset(); corr_.reset(); momentum_.reset();
    ensemble_.reset(); transition_.reset(); bw_.reset(); duration_.reset(); signal_gen_.reset();
    prev_regime_ = MarketRegime::Unknown; count_ = 0;
}

// ============================================================================
// RegimePerformanceTracker Implementation
// ============================================================================

RegimePerformanceTracker::RegimePerformanceTracker() {}

void RegimePerformanceTracker::update(MarketRegime regime, double return_value) {
    int r = static_cast<int>(regime);
    if (r >= 4) return;
    auto& s = stats_[r];
    s.sum_returns += return_value;
    s.sum_sq_returns += return_value * return_value;
    s.cumulative += return_value;
    if (s.cumulative > s.peak) s.peak = s.cumulative;
    s.drawdown = s.peak - s.cumulative;
    if (s.drawdown > s.max_dd) s.max_dd = s.drawdown;
    ++s.n;
}

double RegimePerformanceTracker::avg_return(MarketRegime regime) const {
    int r = static_cast<int>(regime);
    if (r >= 4 || stats_[r].n == 0) return 0;
    return stats_[r].sum_returns / stats_[r].n;
}

double RegimePerformanceTracker::total_return(MarketRegime regime) const {
    int r = static_cast<int>(regime);
    if (r >= 4) return 0;
    return stats_[r].sum_returns;
}

double RegimePerformanceTracker::sharpe(MarketRegime regime) const {
    int r = static_cast<int>(regime);
    if (r >= 4 || stats_[r].n < 2) return 0;
    double mean = stats_[r].sum_returns / stats_[r].n;
    double var = stats_[r].sum_sq_returns / stats_[r].n - mean * mean;
    if (var <= 0) return 0;
    return mean / std::sqrt(var) * std::sqrt(252.0);
}

double RegimePerformanceTracker::max_drawdown(MarketRegime regime) const {
    int r = static_cast<int>(regime);
    if (r >= 4) return 0;
    return stats_[r].max_dd;
}

int RegimePerformanceTracker::count(MarketRegime regime) const {
    int r = static_cast<int>(regime);
    if (r >= 4) return 0;
    return stats_[r].n;
}

void RegimePerformanceTracker::reset() {
    for (auto& s : stats_) s = {};
}

// ============================================================================
// RegimeBacktester Implementation
// ============================================================================

RegimeBacktester::RegimeBacktester()
    : in_trade_(false), entry_price_(0), entry_regime_(MarketRegime::Unknown), hold_count_(0) {}

void RegimeBacktester::add_observation(double price, double log_return, MarketRegime regime) {
    history_.push_back({price, log_return, regime});

    // Simple regime-change trading: enter on bull, exit on non-bull
    if (!in_trade_ && regime == MarketRegime::Bull) {
        in_trade_ = true;
        entry_price_ = price;
        entry_regime_ = regime;
        hold_count_ = 0;
    } else if (in_trade_) {
        ++hold_count_;
        if (regime != entry_regime_ || hold_count_ > 60) {
            TradeResult tr;
            tr.entry_regime = entry_regime_;
            tr.exit_regime = regime;
            tr.entry_price = entry_price_;
            tr.exit_price = price;
            tr.pnl = (price - entry_price_) / entry_price_;
            tr.holding_period = hold_count_;
            trades_.push_back(tr);
            in_trade_ = false;
        }
    }
}

double RegimeBacktester::regime_edge(MarketRegime regime) const {
    double sum = 0;
    int cnt = 0;
    for (const auto& obs : history_) {
        if (obs.regime == regime) {
            sum += obs.log_return;
            ++cnt;
        }
    }
    return (cnt > 0) ? sum / cnt : 0;
}

double RegimeBacktester::hit_rate(MarketRegime regime) const {
    int wins = 0, total = 0;
    for (const auto& t : trades_) {
        if (t.entry_regime == regime) {
            ++total;
            if (t.pnl > 0) ++wins;
        }
    }
    return (total > 0) ? static_cast<double>(wins) / total : 0;
}

double RegimeBacktester::total_pnl() const {
    double sum = 0;
    for (const auto& t : trades_) sum += t.pnl;
    return sum;
}

void RegimeBacktester::reset() {
    history_.clear();
    trades_.clear();
    in_trade_ = false;
    entry_price_ = 0;
    entry_regime_ = MarketRegime::Unknown;
    hold_count_ = 0;
}

} // namespace srfm::regime
