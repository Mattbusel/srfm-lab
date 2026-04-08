#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <deque>
#include <array>
#include <algorithm>

namespace srfm::analytics {

// ----------- Incremental Pearson Correlation -----------
class IncrementalPearson {
public:
    explicit IncrementalPearson(int window = 60);

    void update(double x, double y);
    double correlation() const;
    double covariance() const;
    double mean_x() const { return mean_x_; }
    double mean_y() const { return mean_y_; }
    double var_x() const { return m2_x_ / static_cast<double>(n_); }
    double var_y() const { return m2_y_ / static_cast<double>(n_); }
    void reset();
    int count() const { return n_; }

private:
    int window_;
    int n_;
    double mean_x_, mean_y_;
    double m2_x_, m2_y_, co_moment_;
    std::deque<double> x_buf_, y_buf_;
};

// ----------- Rolling Spearman Rank Correlation -----------
class RollingSpearman {
public:
    explicit RollingSpearman(int window = 60);

    void update(double x, double y);
    double correlation() const { return rho_; }
    void reset();
    int count() const { return static_cast<int>(x_buf_.size()); }

private:
    void recompute();
    int window_;
    double rho_;
    std::deque<double> x_buf_, y_buf_;
};

// ----------- Exponentially Weighted Correlation -----------
class EWMACorrelation {
public:
    struct Config {
        double lambda = 0.97;
    };

    EWMACorrelation();
    explicit EWMACorrelation(const Config& cfg);

    void update(double x, double y);
    double correlation() const;
    double covariance() const { return cov_; }
    double var_x() const { return var_x_; }
    double var_y() const { return var_y_; }
    void reset();
    uint64_t count() const { return count_; }

private:
    Config config_;
    double mean_x_, mean_y_;
    double var_x_, var_y_, cov_;
    uint64_t count_;
};

// ----------- N x N Correlation Matrix -----------
class CorrelationMatrix {
public:
    struct Config {
        int n_assets = 10;
        int window = 60;
        double ewma_lambda = 0.97;
        bool use_ewma = false;
    };

    CorrelationMatrix();
    explicit CorrelationMatrix(const Config& cfg);

    void update(const std::vector<double>& returns);
    double correlation(int i, int j) const;
    void get_matrix(std::vector<std::vector<double>>& out) const;
    double average_correlation() const;
    double dispersion() const;
    void reset();
    int n_assets() const { return config_.n_assets; }

private:
    Config config_;
    // For rolling Pearson
    std::vector<std::deque<double>> return_history_;
    std::vector<double> means_;
    std::vector<double> vars_;
    std::vector<std::vector<double>> corr_;
    // For EWMA
    std::vector<double> ewma_means_;
    std::vector<double> ewma_vars_;
    std::vector<std::vector<double>> ewma_covs_;
    uint64_t count_;

    void update_rolling(const std::vector<double>& returns);
    void update_ewma(const std::vector<double>& returns);
};

// ----------- Eigenvalue Decomposition (Power Iteration) -----------
class EigenDecomposition {
public:
    explicit EigenDecomposition(int max_iter = 200, double tol = 1e-10);

    // Compute top K eigenvalues/vectors from symmetric matrix
    void compute_top_k(const std::vector<std::vector<double>>& matrix, int k,
                       std::vector<double>& eigenvalues,
                       std::vector<std::vector<double>>& eigenvectors);

    // Absorption ratio: fraction of variance explained by top K
    double absorption_ratio(const std::vector<std::vector<double>>& matrix, int k);

private:
    void power_iteration(const std::vector<std::vector<double>>& A, int n,
                         std::vector<double>& eigvec, double& eigval);
    void deflate(std::vector<std::vector<double>>& A, int n,
                 const std::vector<double>& eigvec, double eigval);
    int max_iter_;
    double tol_;
};

// ----------- Correlation Breakdown Detection -----------
class CorrelationBreakdown {
public:
    struct Config {
        int baseline_window = 252;
        double threshold_sigma = 2.5;
    };

    CorrelationBreakdown();
    explicit CorrelationBreakdown(const Config& cfg);

    void update(const std::vector<std::vector<double>>& current_corr);
    bool is_breakdown() const { return breakdown_; }
    double frobenius_change() const { return frob_change_; }
    double z_score() const { return z_score_; }
    void reset();

private:
    Config config_;
    bool breakdown_;
    double frob_change_;
    double z_score_;
    std::vector<std::vector<double>> baseline_;
    std::deque<double> frob_history_;
    bool baseline_set_;
    uint64_t count_;
};

// ----------- Lead-Lag Correlation -----------
class LeadLagCorrelation {
public:
    static constexpr int MAX_LAG = 10;

    explicit LeadLagCorrelation(int window = 120);

    void update(double x, double y);
    double correlation_at_lag(int lag) const; // positive lag: x leads y
    int optimal_lag() const;
    double max_correlation() const;
    void get_all_lags(double out[2 * MAX_LAG + 1]) const;
    void reset();

private:
    int window_;
    std::deque<double> x_buf_, y_buf_;
    double lags_[2 * MAX_LAG + 1]; // lags from -MAX_LAG to +MAX_LAG
};

// ----------- Full Correlation Tracker (combines everything) -----------
class CorrelationTracker {
public:
    struct Config {
        int n_assets = 10;
        int window = 60;
        double ewma_lambda = 0.97;
        int top_k_eigen = 3;
        double breakdown_threshold = 2.5;
    };

    CorrelationTracker();
    explicit CorrelationTracker(const Config& cfg);

    void update(const std::vector<double>& returns);
    double pearson(int i, int j) const;
    double ewma_corr(int i, int j) const;
    double average_correlation() const;
    double absorption_ratio() const;
    bool is_breakdown() const;
    double herding_score() const;
    void reset();

private:
    Config config_;
    CorrelationMatrix rolling_matrix_;
    CorrelationMatrix ewma_matrix_;
    EigenDecomposition eigen_;
    CorrelationBreakdown breakdown_;
    double absorption_ratio_;
    uint64_t count_;
};

} // namespace srfm::analytics
