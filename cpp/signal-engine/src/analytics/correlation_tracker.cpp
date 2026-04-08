#include "correlation_tracker.hpp"
#include <cstring>
#include <numeric>
#include <random>

namespace srfm::analytics {

// ============================================================================
// IncrementalPearson
// ============================================================================

IncrementalPearson::IncrementalPearson(int window)
    : window_(window), n_(0), mean_x_(0), mean_y_(0), m2_x_(0), m2_y_(0), co_moment_(0) {}

void IncrementalPearson::update(double x, double y) {
    x_buf_.push_back(x);
    y_buf_.push_back(y);

    if (static_cast<int>(x_buf_.size()) <= window_) {
        // Growing phase: Welford online update
        ++n_;
        double dx = x - mean_x_;
        mean_x_ += dx / n_;
        double dx2 = x - mean_x_;
        m2_x_ += dx * dx2;

        double dy = y - mean_y_;
        mean_y_ += dy / n_;
        double dy2 = y - mean_y_;
        m2_y_ += dy * dy2;

        co_moment_ += dx * (y - mean_y_); // note: uses old mean_y_ for x, new for y cross
    } else {
        // Sliding window: remove oldest, add newest
        double old_x = x_buf_.front(); x_buf_.pop_front();
        double old_y = y_buf_.front(); y_buf_.pop_front();

        // Recompute from scratch every window for numerical stability
        // This is O(window) but happens every step; for O(1) we'd need a more complex scheme
        n_ = window_;
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        auto ix = x_buf_.begin();
        auto iy = y_buf_.begin();
        for (int i = 0; i < n_; ++i, ++ix, ++iy) {
            sx += *ix; sy += *iy;
        }
        mean_x_ = sx / n_;
        mean_y_ = sy / n_;
        ix = x_buf_.begin();
        iy = y_buf_.begin();
        m2_x_ = m2_y_ = co_moment_ = 0;
        for (int i = 0; i < n_; ++i, ++ix, ++iy) {
            double dx = *ix - mean_x_;
            double dy = *iy - mean_y_;
            m2_x_ += dx * dx;
            m2_y_ += dy * dy;
            co_moment_ += dx * dy;
        }
    }
}

double IncrementalPearson::correlation() const {
    if (n_ < 2) return 0.0;
    double denom = std::sqrt(m2_x_ * m2_y_);
    return (denom > 1e-15) ? co_moment_ / denom : 0.0;
}

double IncrementalPearson::covariance() const {
    if (n_ < 2) return 0.0;
    return co_moment_ / (n_ - 1);
}

void IncrementalPearson::reset() {
    n_ = 0; mean_x_ = mean_y_ = m2_x_ = m2_y_ = co_moment_ = 0;
    x_buf_.clear(); y_buf_.clear();
}

// ============================================================================
// RollingSpearman
// ============================================================================

RollingSpearman::RollingSpearman(int window) : window_(window), rho_(0) {}

void RollingSpearman::update(double x, double y) {
    x_buf_.push_back(x);
    y_buf_.push_back(y);
    if (static_cast<int>(x_buf_.size()) > window_) {
        x_buf_.pop_front();
        y_buf_.pop_front();
    }
    if (static_cast<int>(x_buf_.size()) >= 3) {
        recompute();
    }
}

void RollingSpearman::recompute() {
    int n = static_cast<int>(x_buf_.size());
    // Compute ranks
    std::vector<int> x_idx(n), y_idx(n);
    std::iota(x_idx.begin(), x_idx.end(), 0);
    std::iota(y_idx.begin(), y_idx.end(), 0);

    std::vector<double> xv(x_buf_.begin(), x_buf_.end());
    std::vector<double> yv(y_buf_.begin(), y_buf_.end());

    std::sort(x_idx.begin(), x_idx.end(), [&](int a, int b) { return xv[a] < xv[b]; });
    std::sort(y_idx.begin(), y_idx.end(), [&](int a, int b) { return yv[a] < yv[b]; });

    std::vector<double> x_rank(n), y_rank(n);
    for (int i = 0; i < n; ++i) { x_rank[x_idx[i]] = static_cast<double>(i + 1); }
    for (int i = 0; i < n; ++i) { y_rank[y_idx[i]] = static_cast<double>(i + 1); }

    // Handle ties by averaging ranks
    for (int i = 0; i < n; ) {
        int j = i;
        while (j < n && xv[x_idx[j]] == xv[x_idx[i]]) ++j;
        if (j > i + 1) {
            double avg = 0;
            for (int k = i; k < j; ++k) avg += x_rank[x_idx[k]];
            avg /= (j - i);
            for (int k = i; k < j; ++k) x_rank[x_idx[k]] = avg;
        }
        i = j;
    }
    for (int i = 0; i < n; ) {
        int j = i;
        while (j < n && yv[y_idx[j]] == yv[y_idx[i]]) ++j;
        if (j > i + 1) {
            double avg = 0;
            for (int k = i; k < j; ++k) avg += y_rank[y_idx[k]];
            avg /= (j - i);
            for (int k = i; k < j; ++k) y_rank[y_idx[k]] = avg;
        }
        i = j;
    }

    // Spearman = Pearson of ranks
    double mean_xr = 0, mean_yr = 0;
    for (int i = 0; i < n; ++i) { mean_xr += x_rank[i]; mean_yr += y_rank[i]; }
    mean_xr /= n; mean_yr /= n;

    double cov = 0, vx = 0, vy = 0;
    for (int i = 0; i < n; ++i) {
        double dx = x_rank[i] - mean_xr;
        double dy = y_rank[i] - mean_yr;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    double denom = std::sqrt(vx * vy);
    rho_ = (denom > 1e-15) ? cov / denom : 0.0;
}

void RollingSpearman::reset() {
    rho_ = 0;
    x_buf_.clear(); y_buf_.clear();
}

// ============================================================================
// EWMACorrelation
// ============================================================================

EWMACorrelation::EWMACorrelation() : config_(), mean_x_(0), mean_y_(0), var_x_(0), var_y_(0), cov_(0), count_(0) {}
EWMACorrelation::EWMACorrelation(const Config& cfg) : config_(cfg), mean_x_(0), mean_y_(0), var_x_(0), var_y_(0), cov_(0), count_(0) {}

void EWMACorrelation::update(double x, double y) {
    if (count_ == 0) {
        mean_x_ = x; mean_y_ = y;
        var_x_ = var_y_ = cov_ = 0;
    } else {
        double lam = config_.lambda;
        double dx = x - mean_x_;
        double dy = y - mean_y_;
        mean_x_ = lam * mean_x_ + (1.0 - lam) * x;
        mean_y_ = lam * mean_y_ + (1.0 - lam) * y;
        var_x_ = lam * (var_x_ + (1.0 - lam) * dx * dx);
        var_y_ = lam * (var_y_ + (1.0 - lam) * dy * dy);
        cov_ = lam * (cov_ + (1.0 - lam) * dx * dy);
    }
    ++count_;
}

double EWMACorrelation::correlation() const {
    double denom = std::sqrt(var_x_ * var_y_);
    return (denom > 1e-15) ? cov_ / denom : 0.0;
}

void EWMACorrelation::reset() {
    mean_x_ = mean_y_ = var_x_ = var_y_ = cov_ = 0;
    count_ = 0;
}

// ============================================================================
// CorrelationMatrix
// ============================================================================

CorrelationMatrix::CorrelationMatrix() : config_(), count_(0) {
    int n = config_.n_assets;
    return_history_.resize(n);
    means_.resize(n, 0); vars_.resize(n, 0);
    corr_.assign(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i) corr_[i][i] = 1.0;
    ewma_means_.resize(n, 0); ewma_vars_.resize(n, 0);
    ewma_covs_.assign(n, std::vector<double>(n, 0));
}

CorrelationMatrix::CorrelationMatrix(const Config& cfg) : config_(cfg), count_(0) {
    int n = cfg.n_assets;
    return_history_.resize(n);
    means_.resize(n, 0); vars_.resize(n, 0);
    corr_.assign(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i) corr_[i][i] = 1.0;
    ewma_means_.resize(n, 0); ewma_vars_.resize(n, 0);
    ewma_covs_.assign(n, std::vector<double>(n, 0));
}

void CorrelationMatrix::update(const std::vector<double>& returns) {
    ++count_;
    if (config_.use_ewma) update_ewma(returns);
    else update_rolling(returns);
}

void CorrelationMatrix::update_rolling(const std::vector<double>& returns) {
    int n = config_.n_assets;
    int rn = std::min(n, static_cast<int>(returns.size()));
    for (int i = 0; i < rn; ++i) {
        return_history_[i].push_back(returns[i]);
        if (static_cast<int>(return_history_[i].size()) > config_.window) {
            return_history_[i].pop_front();
        }
    }

    int w = static_cast<int>(return_history_[0].size());
    if (w < 3) return;

    // Recompute means and variances
    for (int i = 0; i < rn; ++i) {
        double s = 0;
        for (auto v : return_history_[i]) s += v;
        means_[i] = s / w;
        double v = 0;
        for (auto r : return_history_[i]) { double d = r - means_[i]; v += d * d; }
        vars_[i] = v;
    }

    // Recompute correlations
    for (int i = 0; i < rn; ++i) {
        corr_[i][i] = 1.0;
        for (int j = i + 1; j < rn; ++j) {
            double cov = 0;
            auto it_i = return_history_[i].begin();
            auto it_j = return_history_[j].begin();
            for (int k = 0; k < w; ++k, ++it_i, ++it_j) {
                cov += (*it_i - means_[i]) * (*it_j - means_[j]);
            }
            double denom = std::sqrt(vars_[i] * vars_[j]);
            double c = (denom > 1e-15) ? cov / denom : 0.0;
            c = std::clamp(c, -1.0, 1.0);
            corr_[i][j] = c;
            corr_[j][i] = c;
        }
    }
}

void CorrelationMatrix::update_ewma(const std::vector<double>& returns) {
    int n = std::min(config_.n_assets, static_cast<int>(returns.size()));
    double lam = config_.ewma_lambda;

    if (count_ == 1) {
        for (int i = 0; i < n; ++i) ewma_means_[i] = returns[i];
        return;
    }

    std::vector<double> demeaned(n);
    for (int i = 0; i < n; ++i) {
        demeaned[i] = returns[i] - ewma_means_[i];
        ewma_means_[i] = lam * ewma_means_[i] + (1.0 - lam) * returns[i];
    }

    for (int i = 0; i < n; ++i) {
        ewma_vars_[i] = lam * (ewma_vars_[i] + (1.0 - lam) * demeaned[i] * demeaned[i]);
        for (int j = i; j < n; ++j) {
            ewma_covs_[i][j] = lam * (ewma_covs_[i][j] + (1.0 - lam) * demeaned[i] * demeaned[j]);
            ewma_covs_[j][i] = ewma_covs_[i][j];
        }
    }

    // Convert to correlations
    for (int i = 0; i < n; ++i) {
        corr_[i][i] = 1.0;
        for (int j = i + 1; j < n; ++j) {
            double denom = std::sqrt(ewma_vars_[i] * ewma_vars_[j]);
            double c = (denom > 1e-15) ? ewma_covs_[i][j] / denom : 0.0;
            c = std::clamp(c, -1.0, 1.0);
            corr_[i][j] = c;
            corr_[j][i] = c;
        }
    }
}

double CorrelationMatrix::correlation(int i, int j) const {
    if (i < 0 || i >= config_.n_assets || j < 0 || j >= config_.n_assets) return 0.0;
    return corr_[i][j];
}

void CorrelationMatrix::get_matrix(std::vector<std::vector<double>>& out) const { out = corr_; }

double CorrelationMatrix::average_correlation() const {
    int n = config_.n_assets;
    double sum = 0; int cnt = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) { sum += corr_[i][j]; ++cnt; }
    return (cnt > 0) ? sum / cnt : 0.0;
}

double CorrelationMatrix::dispersion() const {
    double avg = average_correlation();
    int n = config_.n_assets;
    double sum = 0; int cnt = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) { double d = corr_[i][j] - avg; sum += d * d; ++cnt; }
    return (cnt > 0) ? std::sqrt(sum / cnt) : 0.0;
}

void CorrelationMatrix::reset() {
    int n = config_.n_assets;
    for (auto& h : return_history_) h.clear();
    std::fill(means_.begin(), means_.end(), 0);
    std::fill(vars_.begin(), vars_.end(), 0);
    for (auto& row : corr_) std::fill(row.begin(), row.end(), 0);
    for (int i = 0; i < n; ++i) corr_[i][i] = 1.0;
    std::fill(ewma_means_.begin(), ewma_means_.end(), 0);
    std::fill(ewma_vars_.begin(), ewma_vars_.end(), 0);
    for (auto& row : ewma_covs_) std::fill(row.begin(), row.end(), 0);
    count_ = 0;
}

// ============================================================================
// EigenDecomposition
// ============================================================================

EigenDecomposition::EigenDecomposition(int max_iter, double tol) : max_iter_(max_iter), tol_(tol) {}

void EigenDecomposition::power_iteration(const std::vector<std::vector<double>>& A, int n,
                                          std::vector<double>& eigvec, double& eigval) {
    eigvec.resize(n);
    // Random initial vector
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) eigvec[i] = dist(gen);

    // Normalize
    double norm = 0;
    for (int i = 0; i < n; ++i) norm += eigvec[i] * eigvec[i];
    norm = std::sqrt(norm);
    for (int i = 0; i < n; ++i) eigvec[i] /= norm;

    eigval = 0;
    for (int iter = 0; iter < max_iter_; ++iter) {
        // Multiply: w = A * v
        std::vector<double> w(n, 0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                w[i] += A[i][j] * eigvec[j];

        // Compute eigenvalue = v^T * w
        double new_eigval = 0;
        for (int i = 0; i < n; ++i) new_eigval += eigvec[i] * w[i];

        // Normalize w
        norm = 0;
        for (int i = 0; i < n; ++i) norm += w[i] * w[i];
        norm = std::sqrt(norm);
        if (norm < 1e-15) break;
        for (int i = 0; i < n; ++i) w[i] /= norm;

        // Check convergence
        if (std::abs(new_eigval - eigval) < tol_) {
            eigval = new_eigval;
            eigvec = w;
            return;
        }
        eigval = new_eigval;
        eigvec = w;
    }
}

void EigenDecomposition::deflate(std::vector<std::vector<double>>& A, int n,
                                  const std::vector<double>& eigvec, double eigval) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] -= eigval * eigvec[i] * eigvec[j];
}

void EigenDecomposition::compute_top_k(const std::vector<std::vector<double>>& matrix, int k,
                                        std::vector<double>& eigenvalues,
                                        std::vector<std::vector<double>>& eigenvectors) {
    int n = static_cast<int>(matrix.size());
    k = std::min(k, n);
    eigenvalues.resize(k);
    eigenvectors.resize(k);

    auto A = matrix; // copy for deflation
    for (int i = 0; i < k; ++i) {
        std::vector<double> vec;
        double val;
        power_iteration(A, n, vec, val);
        eigenvalues[i] = val;
        eigenvectors[i] = vec;
        deflate(A, n, vec, val);
    }
}

double EigenDecomposition::absorption_ratio(const std::vector<std::vector<double>>& matrix, int k) {
    int n = static_cast<int>(matrix.size());
    if (n == 0) return 0.0;

    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> eigenvectors;
    compute_top_k(matrix, k, eigenvalues, eigenvectors);

    double top_sum = 0;
    for (auto v : eigenvalues) top_sum += std::max(v, 0.0);

    double total = 0;
    for (int i = 0; i < n; ++i) total += matrix[i][i]; // trace = sum of all eigenvalues

    return (total > 1e-15) ? top_sum / total : 0.0;
}

// ============================================================================
// CorrelationBreakdown
// ============================================================================

CorrelationBreakdown::CorrelationBreakdown()
    : config_(), breakdown_(false), frob_change_(0), z_score_(0), baseline_set_(false), count_(0) {}

CorrelationBreakdown::CorrelationBreakdown(const Config& cfg)
    : config_(cfg), breakdown_(false), frob_change_(0), z_score_(0), baseline_set_(false), count_(0) {}

void CorrelationBreakdown::update(const std::vector<std::vector<double>>& current_corr) {
    ++count_;
    int n = static_cast<int>(current_corr.size());

    if (!baseline_set_) {
        baseline_ = current_corr;
        baseline_set_ = true;
        return;
    }

    // Frobenius norm of difference
    double frob = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double d = current_corr[i][j] - baseline_[i][j];
            frob += d * d;
        }
    frob_change_ = std::sqrt(frob);

    frob_history_.push_back(frob_change_);
    if (static_cast<int>(frob_history_.size()) > config_.baseline_window) {
        frob_history_.pop_front();
    }

    // Z-score of current change vs history
    if (frob_history_.size() >= 10) {
        double sum = 0, sum2 = 0;
        for (auto v : frob_history_) { sum += v; sum2 += v * v; }
        int hn = static_cast<int>(frob_history_.size());
        double mean = sum / hn;
        double var = sum2 / hn - mean * mean;
        double std_dev = (var > 0) ? std::sqrt(var) : 1e-10;
        z_score_ = (frob_change_ - mean) / std_dev;
        breakdown_ = z_score_ > config_.threshold_sigma;
    }

    // Update baseline with exponential decay
    double alpha = 0.01;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            baseline_[i][j] = (1.0 - alpha) * baseline_[i][j] + alpha * current_corr[i][j];
}

void CorrelationBreakdown::reset() {
    breakdown_ = false; frob_change_ = z_score_ = 0;
    baseline_.clear(); frob_history_.clear();
    baseline_set_ = false; count_ = 0;
}

// ============================================================================
// LeadLagCorrelation
// ============================================================================

LeadLagCorrelation::LeadLagCorrelation(int window) : window_(window) {
    std::memset(lags_, 0, sizeof(lags_));
}

void LeadLagCorrelation::update(double x, double y) {
    x_buf_.push_back(x);
    y_buf_.push_back(y);
    if (static_cast<int>(x_buf_.size()) > window_) {
        x_buf_.pop_front();
        y_buf_.pop_front();
    }

    int n = static_cast<int>(x_buf_.size());
    if (n < MAX_LAG + 3) return;

    // Compute cross-correlation at each lag
    std::vector<double> xv(x_buf_.begin(), x_buf_.end());
    std::vector<double> yv(y_buf_.begin(), y_buf_.end());

    // Means
    double mx = 0, my = 0;
    for (int i = 0; i < n; ++i) { mx += xv[i]; my += yv[i]; }
    mx /= n; my /= n;

    // Variances
    double vx = 0, vy = 0;
    for (int i = 0; i < n; ++i) {
        vx += (xv[i] - mx) * (xv[i] - mx);
        vy += (yv[i] - my) * (yv[i] - my);
    }
    double denom = std::sqrt(vx * vy);
    if (denom < 1e-15) { std::memset(lags_, 0, sizeof(lags_)); return; }

    for (int lag = -MAX_LAG; lag <= MAX_LAG; ++lag) {
        double cov = 0;
        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            int j = i + lag;
            if (j >= 0 && j < n) {
                cov += (xv[i] - mx) * (yv[j] - my);
                ++cnt;
            }
        }
        lags_[lag + MAX_LAG] = (cnt > 0) ? cov / denom : 0.0;
    }
}

double LeadLagCorrelation::correlation_at_lag(int lag) const {
    if (lag < -MAX_LAG || lag > MAX_LAG) return 0.0;
    return lags_[lag + MAX_LAG];
}

int LeadLagCorrelation::optimal_lag() const {
    int best = 0;
    double best_abs = 0;
    for (int i = 0; i < 2 * MAX_LAG + 1; ++i) {
        if (std::abs(lags_[i]) > best_abs) {
            best_abs = std::abs(lags_[i]);
            best = i - MAX_LAG;
        }
    }
    return best;
}

double LeadLagCorrelation::max_correlation() const {
    double best = 0;
    for (auto v : lags_) if (std::abs(v) > std::abs(best)) best = v;
    return best;
}

void LeadLagCorrelation::get_all_lags(double out[2 * MAX_LAG + 1]) const {
    std::memcpy(out, lags_, sizeof(lags_));
}

void LeadLagCorrelation::reset() {
    x_buf_.clear(); y_buf_.clear();
    std::memset(lags_, 0, sizeof(lags_));
}

// ============================================================================
// CorrelationTracker (combined)
// ============================================================================

CorrelationTracker::CorrelationTracker() : config_(), absorption_ratio_(0), count_(0) {
    CorrelationMatrix::Config rcfg;
    rcfg.n_assets = config_.n_assets; rcfg.window = config_.window; rcfg.use_ewma = false;
    rolling_matrix_ = CorrelationMatrix(rcfg);
    rcfg.use_ewma = true; rcfg.ewma_lambda = config_.ewma_lambda;
    ewma_matrix_ = CorrelationMatrix(rcfg);
    CorrelationBreakdown::Config bcfg;
    bcfg.threshold_sigma = config_.breakdown_threshold;
    breakdown_ = CorrelationBreakdown(bcfg);
}

CorrelationTracker::CorrelationTracker(const Config& cfg) : config_(cfg), absorption_ratio_(0), count_(0) {
    CorrelationMatrix::Config rcfg;
    rcfg.n_assets = cfg.n_assets; rcfg.window = cfg.window; rcfg.use_ewma = false;
    rolling_matrix_ = CorrelationMatrix(rcfg);
    rcfg.use_ewma = true; rcfg.ewma_lambda = cfg.ewma_lambda;
    ewma_matrix_ = CorrelationMatrix(rcfg);
    CorrelationBreakdown::Config bcfg;
    bcfg.threshold_sigma = cfg.breakdown_threshold;
    breakdown_ = CorrelationBreakdown(bcfg);
}

void CorrelationTracker::update(const std::vector<double>& returns) {
    rolling_matrix_.update(returns);
    ewma_matrix_.update(returns);
    ++count_;

    // Periodically compute eigenvalues (expensive)
    if (count_ % 10 == 0 && count_ > 30) {
        std::vector<std::vector<double>> mat;
        rolling_matrix_.get_matrix(mat);
        absorption_ratio_ = eigen_.absorption_ratio(mat, config_.top_k_eigen);
        breakdown_.update(mat);
    }
}

double CorrelationTracker::pearson(int i, int j) const { return rolling_matrix_.correlation(i, j); }
double CorrelationTracker::ewma_corr(int i, int j) const { return ewma_matrix_.correlation(i, j); }
double CorrelationTracker::average_correlation() const { return rolling_matrix_.average_correlation(); }
double CorrelationTracker::absorption_ratio() const { return absorption_ratio_; }
bool CorrelationTracker::is_breakdown() const { return breakdown_.is_breakdown(); }

double CorrelationTracker::herding_score() const {
    // Combine average correlation + absorption ratio
    return 0.5 * rolling_matrix_.average_correlation() + 0.5 * absorption_ratio_;
}

void CorrelationTracker::reset() {
    rolling_matrix_.reset(); ewma_matrix_.reset(); breakdown_.reset();
    absorption_ratio_ = 0; count_ = 0;
}

} // namespace srfm::analytics
