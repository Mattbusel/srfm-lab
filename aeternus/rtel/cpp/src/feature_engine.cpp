// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// feature_engine.cpp — Real-time feature engineering for ML pipeline
// =============================================================================
// Computes LOB-derived features, vol surface features, graph features,
// and cross-asset features at microsecond latency.
// All computations are designed for online (streaming) updates.

#include "rtel/global_state_registry.hpp"
#include "rtel/ring_buffer.hpp"
#include "rtel/shm_bus.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtel {
namespace features {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int kMaxAssets    = 512;
static constexpr int kMaxLevels    = 10;
static constexpr int kMaxLags      = 20;
static constexpr int kEwmaWindow   = 256;
static constexpr double kEps       = 1e-10;

// ---------------------------------------------------------------------------
// Welford online mean/variance
// ---------------------------------------------------------------------------
struct WelfordStats {
    double mean   = 0.0;
    double M2     = 0.0;
    double count  = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();

    void update(double x) {
        ++count;
        double delta  = x - mean;
        mean         += delta / count;
        double delta2 = x - mean;
        M2           += delta * delta2;
        if (x < min_val) min_val = x;
        if (x > max_val) max_val = x;
    }

    double variance() const {
        return (count > 1) ? M2 / (count - 1.0) : 0.0;
    }

    double std_dev() const {
        return std::sqrt(variance());
    }

    double z_score(double x) const {
        double s = std_dev();
        return (s > kEps) ? (x - mean) / s : 0.0;
    }

    void reset() {
        mean = 0.0; M2 = 0.0; count = 0.0;
        min_val = std::numeric_limits<double>::max();
        max_val = std::numeric_limits<double>::lowest();
    }
};

// ---------------------------------------------------------------------------
// EWMA (exponential weighted moving average)
// ---------------------------------------------------------------------------
struct EWMA {
    double alpha;
    double value   = 0.0;
    double var     = 0.0;      // EWMA variance
    bool   init    = false;

    explicit EWMA(double alpha = 0.1) : alpha(alpha) {}

    void update(double x) {
        if (!init) { value = x; var = 0.0; init = true; return; }
        double diff = x - value;
        value += alpha * diff;
        var    = (1.0 - alpha) * (var + alpha * diff * diff);
    }

    double std_dev() const { return std::sqrt(var); }

    double z_score(double x) const {
        double s = std_dev();
        return (s > kEps) ? (x - value) / s : 0.0;
    }

    void reset() { value = 0.0; var = 0.0; init = false; }
};

// ---------------------------------------------------------------------------
// CircularBuffer — fixed-size rolling window
// ---------------------------------------------------------------------------
template<typename T, int N>
struct CircularBuffer {
    std::array<T, N> data{};
    int head = 0;
    int size = 0;

    void push(const T& v) {
        data[head] = v;
        head = (head + 1) % N;
        if (size < N) ++size;
    }

    // Access from newest (0) to oldest (size-1)
    const T& at(int i) const {
        int idx = (head - 1 - i + N) % N;
        return data[idx];
    }

    bool full() const { return size == N; }
    int  len()  const { return size; }
    void clear() { head = 0; size = 0; data.fill(T{}); }
};

// ---------------------------------------------------------------------------
// LOB feature vector (per asset)
// ---------------------------------------------------------------------------
struct LOBFeatures {
    // Imbalance features
    double bid_ask_imbalance;       // (bid_qty - ask_qty) / (bid_qty + ask_qty)
    double bid_ask_imbalance_l2;    // same for top 2 levels
    double bid_ask_imbalance_l5;
    double bid_ask_imbalance_l10;

    // Price features
    double mid_price;
    double micro_price;             // qty-weighted mid
    double spread;
    double spread_bps;              // spread in basis points
    double log_spread;

    // Depth features
    double bid_depth_l1;
    double bid_depth_l3;
    double bid_depth_l5;
    double ask_depth_l1;
    double ask_depth_l3;
    double ask_depth_l5;
    double total_depth;

    // VWAP features
    double bid_vwap_l5;
    double ask_vwap_l5;
    double vwap_spread;

    // Shape features
    double bid_slope;               // linear fit slope of bid side
    double ask_slope;
    double bid_convexity;
    double ask_convexity;

    // Derived
    double ob_pressure;             // (weighted_bid - weighted_ask) / total
    double kyle_lambda;             // price impact estimate

    static constexpr int kDim = 26;

    void to_array(double* out) const {
        out[0]  = bid_ask_imbalance;
        out[1]  = bid_ask_imbalance_l2;
        out[2]  = bid_ask_imbalance_l5;
        out[3]  = bid_ask_imbalance_l10;
        out[4]  = mid_price;
        out[5]  = micro_price;
        out[6]  = spread;
        out[7]  = spread_bps;
        out[8]  = log_spread;
        out[9]  = bid_depth_l1;
        out[10] = bid_depth_l3;
        out[11] = bid_depth_l5;
        out[12] = ask_depth_l1;
        out[13] = ask_depth_l3;
        out[14] = ask_depth_l5;
        out[15] = total_depth;
        out[16] = bid_vwap_l5;
        out[17] = ask_vwap_l5;
        out[18] = vwap_spread;
        out[19] = bid_slope;
        out[20] = ask_slope;
        out[21] = bid_convexity;
        out[22] = ask_convexity;
        out[23] = ob_pressure;
        out[24] = kyle_lambda;
        out[25] = 0.0;  // padding
    }
};

// ---------------------------------------------------------------------------
// LOB feature extractor
// ---------------------------------------------------------------------------
class LOBFeatureExtractor {
public:
    struct Level { double price; double size; };

    LOBFeatures extract(const std::vector<Level>& bids,
                        const std::vector<Level>& asks) const
    {
        LOBFeatures f{};
        if (bids.empty() || asks.empty()) return f;

        // Mid price
        double best_bid = bids[0].price;
        double best_ask = asks[0].price;
        f.mid_price     = 0.5 * (best_bid + best_ask);
        f.spread        = best_ask - best_bid;
        f.spread_bps    = (f.mid_price > kEps) ? f.spread / f.mid_price * 1e4 : 0.0;
        f.log_spread    = (f.spread > kEps) ? std::log(f.spread) : -30.0;

        // Depth sums
        auto depth_sum = [](const std::vector<Level>& side, int n) {
            double d = 0.0;
            for (int i = 0; i < std::min((int)side.size(), n); ++i)
                d += side[i].size;
            return d;
        };

        f.bid_depth_l1  = depth_sum(bids, 1);
        f.bid_depth_l3  = depth_sum(bids, 3);
        f.bid_depth_l5  = depth_sum(bids, 5);
        f.ask_depth_l1  = depth_sum(asks, 1);
        f.ask_depth_l3  = depth_sum(asks, 3);
        f.ask_depth_l5  = depth_sum(asks, 5);
        f.total_depth   = depth_sum(bids, kMaxLevels) + depth_sum(asks, kMaxLevels);

        // Imbalance
        auto imbalance = [&](int n) {
            double b = depth_sum(bids, n), a = depth_sum(asks, n);
            double t = b + a;
            return (t > kEps) ? (b - a) / t : 0.0;
        };
        f.bid_ask_imbalance     = imbalance(1);
        f.bid_ask_imbalance_l2  = imbalance(2);
        f.bid_ask_imbalance_l5  = imbalance(5);
        f.bid_ask_imbalance_l10 = imbalance(10);

        // Micro price (size-weighted mid)
        {
            double bq = f.bid_depth_l1, aq = f.ask_depth_l1;
            double tot = bq + aq;
            f.micro_price = (tot > kEps) ?
                (best_bid * aq + best_ask * bq) / tot : f.mid_price;
        }

        // VWAP
        auto vwap = [](const std::vector<Level>& side, int n) {
            double vol = 0.0, notional = 0.0;
            for (int i = 0; i < std::min((int)side.size(), n); ++i) {
                vol      += side[i].size;
                notional += side[i].price * side[i].size;
            }
            return (vol > kEps) ? notional / vol : 0.0;
        };
        f.bid_vwap_l5 = vwap(bids, 5);
        f.ask_vwap_l5 = vwap(asks, 5);
        f.vwap_spread = f.ask_vwap_l5 - f.bid_vwap_l5;

        // Book slope (linear regression of size vs level index)
        auto book_slope = [](const std::vector<Level>& side, int n) -> std::pair<double,double> {
            int k = std::min((int)side.size(), n);
            if (k < 2) return {0.0, 0.0};
            double sx=0, sy=0, sxy=0, sxx=0;
            for (int i = 0; i < k; ++i) {
                sx  += i; sy  += side[i].size;
                sxy += i * side[i].size; sxx += i * i;
            }
            double det = k * sxx - sx * sx;
            if (std::abs(det) < kEps) return {0.0, 0.0};
            double slope = (k * sxy - sx * sy) / det;
            // convexity: mean second diff
            double conv = 0.0;
            for (int i = 1; i < k-1; ++i)
                conv += side[i+1].size - 2*side[i].size + side[i-1].size;
            return {slope, (k > 2) ? conv / (k-2) : 0.0};
        };
        auto [bs, bc] = book_slope(bids, kMaxLevels);
        auto [as_, ac] = book_slope(asks, kMaxLevels);
        f.bid_slope = bs; f.bid_convexity = bc;
        f.ask_slope = as_; f.ask_convexity = ac;

        // OB pressure (exponentially weighted)
        {
            double bp = 0.0, ap = 0.0;
            for (int i = 0; i < std::min((int)bids.size(), kMaxLevels); ++i) {
                double w = std::exp(-0.5 * i);
                bp += w * bids[i].size;
                ap += w * asks[i].size;
            }
            double tot = bp + ap;
            f.ob_pressure = (tot > kEps) ? (bp - ap) / tot : 0.0;
        }

        // Kyle lambda (rough estimate: spread / depth)
        f.kyle_lambda = (f.total_depth > kEps) ? f.spread / f.total_depth : 0.0;

        return f;
    }
};

// ---------------------------------------------------------------------------
// Temporal LOB features (rolling windows)
// ---------------------------------------------------------------------------
struct TemporalLOBFeatures {
    double mid_price_ret_1;   // 1-step log return
    double mid_price_ret_5;
    double mid_price_ret_10;
    double mid_price_ret_20;

    double vol_1;             // realized vol over 1-step
    double vol_5;
    double vol_20;

    double imbalance_ma5;     // moving average of imbalance
    double imbalance_ma20;

    double spread_ma5;
    double spread_ma20;

    double trade_flow_5;      // net trade flow (buy-sell) over 5 steps
    double trade_flow_20;

    double auto_corr_1;       // 1-lag autocorrelation of returns
    double auto_corr_5;

    static constexpr int kDim = 15;

    void to_array(double* out) const {
        out[0]  = mid_price_ret_1;
        out[1]  = mid_price_ret_5;
        out[2]  = mid_price_ret_10;
        out[3]  = mid_price_ret_20;
        out[4]  = vol_1;
        out[5]  = vol_5;
        out[6]  = vol_20;
        out[7]  = imbalance_ma5;
        out[8]  = imbalance_ma20;
        out[9]  = spread_ma5;
        out[10] = spread_ma20;
        out[11] = trade_flow_5;
        out[12] = trade_flow_20;
        out[13] = auto_corr_1;
        out[14] = auto_corr_5;
    }
};

class TemporalFeatureExtractor {
    static constexpr int kBufLen = kMaxLags + 2;
    CircularBuffer<double, 32> mid_prices_;
    CircularBuffer<double, 32> imbalances_;
    CircularBuffer<double, 32> spreads_;
    CircularBuffer<double, 32> trade_flows_;

public:
    void update(double mid, double imbal, double spread, double trade_flow) {
        mid_prices_.push(mid);
        imbalances_.push(imbal);
        spreads_.push(spread);
        trade_flows_.push(trade_flow);
    }

    TemporalLOBFeatures extract() const {
        TemporalLOBFeatures f{};
        int n = mid_prices_.len();
        if (n < 2) return f;

        // Log returns
        auto log_ret = [&](int lag) -> double {
            if (n <= lag) return 0.0;
            double p0 = mid_prices_.at(0), p1 = mid_prices_.at(lag);
            return (p0 > kEps && p1 > kEps) ? std::log(p0 / p1) : 0.0;
        };
        f.mid_price_ret_1  = log_ret(1);
        f.mid_price_ret_5  = log_ret(5);
        f.mid_price_ret_10 = log_ret(10);
        f.mid_price_ret_20 = log_ret(20);

        // Realized vol
        auto realized_vol = [&](int window) -> double {
            int k = std::min(n-1, window);
            if (k < 1) return 0.0;
            double sum2 = 0.0;
            for (int i = 0; i < k; ++i) {
                double p0 = mid_prices_.at(i), p1 = mid_prices_.at(i+1);
                double r = (p0 > kEps && p1 > kEps) ? std::log(p0/p1) : 0.0;
                sum2 += r*r;
            }
            return std::sqrt(sum2 / k);
        };
        f.vol_1  = realized_vol(1);
        f.vol_5  = realized_vol(5);
        f.vol_20 = realized_vol(20);

        // MA of imbalance and spread
        auto ma = [&](const auto& buf, int window) -> double {
            int k = std::min(buf.len(), window);
            if (k == 0) return 0.0;
            double sum = 0.0;
            for (int i = 0; i < k; ++i) sum += buf.at(i);
            return sum / k;
        };
        f.imbalance_ma5  = ma(imbalances_, 5);
        f.imbalance_ma20 = ma(imbalances_, 20);
        f.spread_ma5     = ma(spreads_, 5);
        f.spread_ma20    = ma(spreads_, 20);

        // Trade flow
        auto flow_sum = [&](int window) -> double {
            int k = std::min(trade_flows_.len(), window);
            double s = 0.0;
            for (int i = 0; i < k; ++i) s += trade_flows_.at(i);
            return s;
        };
        f.trade_flow_5  = flow_sum(5);
        f.trade_flow_20 = flow_sum(20);

        // Autocorrelation of returns
        auto autocorr = [&](int lag) -> double {
            int k = std::min(n-1, 20);
            if (k <= lag) return 0.0;
            std::vector<double> rets;
            rets.reserve(k);
            for (int i = 0; i < k; ++i) {
                double p0 = mid_prices_.at(i), p1 = mid_prices_.at(i+1);
                rets.push_back((p0 > kEps && p1 > kEps) ? std::log(p0/p1) : 0.0);
            }
            int m = (int)rets.size() - lag;
            if (m <= 0) return 0.0;
            double mu = 0.0;
            for (double r : rets) mu += r;
            mu /= rets.size();
            double cov = 0.0, var = 0.0;
            for (int i = 0; i < m; ++i) {
                cov += (rets[i] - mu) * (rets[i+lag] - mu);
            }
            for (double r : rets) var += (r-mu)*(r-mu);
            return (var > kEps) ? cov / var : 0.0;
        };
        f.auto_corr_1 = autocorr(1);
        f.auto_corr_5 = autocorr(5);

        return f;
    }
};

// ---------------------------------------------------------------------------
// Cross-asset features
// ---------------------------------------------------------------------------
struct CrossAssetFeatures {
    double correlation_mean;        // mean pairwise correlation
    double correlation_dispersion;  // std of pairwise correlations
    double lead_lag_score;          // mean absolute lead-lag
    double sector_dispersion;       // cross-sectional vol
    double factor_loading_1;        // projection onto 1st PC
    double factor_loading_2;        // projection onto 2nd PC
    double beta_to_index;           // beta to equal-weighted index
    double residual_vol;            // idiosyncratic vol

    static constexpr int kDim = 8;

    void to_array(double* out) const {
        out[0] = correlation_mean;
        out[1] = correlation_dispersion;
        out[2] = lead_lag_score;
        out[3] = sector_dispersion;
        out[4] = factor_loading_1;
        out[5] = factor_loading_2;
        out[6] = beta_to_index;
        out[7] = residual_vol;
    }
};

// Incremental correlation matrix using EWMA
class EWMACorrelationMatrix {
    int n_;
    double alpha_;
    std::vector<double> means_;
    std::vector<double> vars_;
    std::vector<double> covs_;  // upper triangular, row-major

    int idx(int i, int j) const {
        if (i > j) std::swap(i, j);
        return i * n_ - i*(i+1)/2 + j;
    }

public:
    EWMACorrelationMatrix(int n, double alpha = 0.05)
        : n_(n), alpha_(alpha),
          means_(n, 0.0), vars_(n, 0.0),
          covs_(n*(n+1)/2, 0.0) {}

    void update(const std::vector<double>& x) {
        assert((int)x.size() == n_);
        std::vector<double> centered(n_);
        for (int i = 0; i < n_; ++i) {
            double dx = x[i] - means_[i];
            means_[i] += alpha_ * dx;
            vars_[i]   = (1.0 - alpha_) * (vars_[i] + alpha_ * dx * dx);
            centered[i] = dx;
        }
        // Update covariances
        for (int i = 0; i < n_; ++i) {
            for (int j = i; j < n_; ++j) {
                covs_[idx(i,j)] = (1.0 - alpha_) *
                    (covs_[idx(i,j)] + alpha_ * centered[i] * centered[j]);
            }
        }
    }

    double correlation(int i, int j) const {
        double vi = vars_[i], vj = vars_[j];
        if (vi < kEps || vj < kEps) return 0.0;
        return covs_[idx(i,j)] / std::sqrt(vi * vj);
    }

    // Returns (mean_corr, dispersion_corr)
    std::pair<double,double> correlation_stats() const {
        if (n_ < 2) return {0.0, 0.0};
        double sum = 0.0, sum2 = 0.0;
        int cnt = 0;
        for (int i = 0; i < n_; ++i) {
            for (int j = i+1; j < n_; ++j) {
                double c = correlation(i, j);
                sum  += c; sum2 += c*c; ++cnt;
            }
        }
        if (cnt == 0) return {0.0, 0.0};
        double mean = sum / cnt;
        double var  = sum2 / cnt - mean*mean;
        return {mean, std::sqrt(std::max(0.0, var))};
    }
};

// ---------------------------------------------------------------------------
// Vol surface features
// ---------------------------------------------------------------------------
struct VolSurfaceFeatures {
    double atm_vol;           // at-the-money vol
    double term_slope;        // slope across expiries
    double smile_skew;        // skew of the smile
    double smile_curvature;   // second moment of smile
    double vol_of_vol;        // EWMA vol of atm_vol changes
    double iv_spread_25d;     // 25-delta call/put spread
    double risk_reversal_10d;
    double butterfly_25d;
    double calendar_spread;   // front vs back month vol diff
    double surface_norm;      // Frobenius norm of surface

    static constexpr int kDim = 10;

    void to_array(double* out) const {
        out[0] = atm_vol;
        out[1] = term_slope;
        out[2] = smile_skew;
        out[3] = smile_curvature;
        out[4] = vol_of_vol;
        out[5] = iv_spread_25d;
        out[6] = risk_reversal_10d;
        out[7] = butterfly_25d;
        out[8] = calendar_spread;
        out[9] = surface_norm;
    }
};

class VolSurfaceFeatureExtractor {
    EWMA atm_vol_ewma_{0.1};

public:
    // surface: [n_strikes × n_expiries]
    VolSurfaceFeatures extract(const std::vector<std::vector<double>>& surface,
                               int atm_idx) const
    {
        VolSurfaceFeatures f{};
        int ns = (int)surface.size();
        int ne = (ns > 0) ? (int)surface[0].size() : 0;
        if (ns == 0 || ne == 0) return f;

        // ATM vol (front expiry)
        if (atm_idx >= 0 && atm_idx < ns)
            f.atm_vol = surface[atm_idx][0];

        // Term slope: regression of ATM vol vs expiry
        if (ne >= 2) {
            double sx=0, sy=0, sxy=0, sxx=0;
            int k = std::min(ne, 12);
            for (int j = 0; j < k; ++j) {
                double v = (atm_idx >= 0 && atm_idx < ns) ? surface[atm_idx][j] : 0.0;
                sx += j; sy += v; sxy += j*v; sxx += j*j;
            }
            double det = k*sxx - sx*sx;
            f.term_slope = (std::abs(det) > kEps) ? (k*sxy - sx*sy)/det : 0.0;
        }

        // Smile skew and curvature (front expiry, across strikes)
        if (ns >= 3) {
            double mean_v = 0.0, mean2 = 0.0, mean3 = 0.0;
            for (int i = 0; i < ns; ++i) mean_v += surface[i][0];
            mean_v /= ns;
            for (int i = 0; i < ns; ++i) {
                double d = surface[i][0] - mean_v;
                mean2 += d*d; mean3 += d*d*d;
            }
            mean2 /= ns; mean3 /= ns;
            double std3 = std::pow(std::max(kEps, mean2), 1.5);
            f.smile_skew      = mean3 / std3;
            f.smile_curvature = mean2 / (mean_v * mean_v + kEps);
        }

        // Risk reversals and butterflies at 25-delta (approximate: ±25% moneyness)
        {
            int lo = std::max(0, atm_idx - ns/4);
            int hi = std::min(ns-1, atm_idx + ns/4);
            double v_lo = surface[lo][0], v_hi = surface[hi][0];
            double v_atm = f.atm_vol;
            f.iv_spread_25d     = v_hi - v_lo;
            f.risk_reversal_10d = v_hi - v_lo;  // simplified
            f.butterfly_25d     = 0.5*(v_hi + v_lo) - v_atm;
        }

        // Calendar spread
        if (ne >= 2 && atm_idx >= 0 && atm_idx < ns)
            f.calendar_spread = surface[atm_idx][ne-1] - surface[atm_idx][0];

        // Surface norm
        double norm2 = 0.0;
        for (int i = 0; i < ns; ++i)
            for (int j = 0; j < ne; ++j)
                norm2 += surface[i][j] * surface[i][j];
        f.surface_norm = std::sqrt(norm2);

        return f;
    }
};

// ---------------------------------------------------------------------------
// Graph features
// ---------------------------------------------------------------------------
struct GraphFeatures {
    double density;
    double mean_degree;
    double std_degree;
    double mean_clustering;
    double spectral_gap;
    double modularity;
    double mean_pagerank;
    double max_pagerank;
    double mean_betweenness;
    double n_communities;

    static constexpr int kDim = 10;

    void to_array(double* out) const {
        out[0] = density;
        out[1] = mean_degree;
        out[2] = std_degree;
        out[3] = mean_clustering;
        out[4] = spectral_gap;
        out[5] = modularity;
        out[6] = mean_pagerank;
        out[7] = max_pagerank;
        out[8] = mean_betweenness;
        out[9] = n_communities;
    }
};

GraphFeatures extract_graph_features(const std::vector<std::vector<double>>& adj,
                                      int n_nodes)
{
    GraphFeatures f{};
    if (n_nodes == 0) return f;

    // Degree (weighted)
    std::vector<double> degree(n_nodes, 0.0);
    double total_edges = 0.0;
    for (int i = 0; i < n_nodes; ++i) {
        for (int j = 0; j < n_nodes; ++j) {
            if (i != j) {
                degree[i]   += adj[i][j];
                total_edges += adj[i][j];
            }
        }
    }
    total_edges *= 0.5;

    // Density
    double max_edges = (double)n_nodes * (n_nodes-1) * 0.5;
    f.density = (max_edges > 0) ? total_edges / max_edges : 0.0;

    // Mean/std degree
    double sum_d = 0.0, sum_d2 = 0.0;
    for (int i = 0; i < n_nodes; ++i) { sum_d += degree[i]; sum_d2 += degree[i]*degree[i]; }
    f.mean_degree = sum_d / n_nodes;
    double var_d  = sum_d2/n_nodes - f.mean_degree*f.mean_degree;
    f.std_degree  = std::sqrt(std::max(0.0, var_d));

    // Clustering coefficient
    double total_clustering = 0.0;
    for (int i = 0; i < n_nodes; ++i) {
        double ki = degree[i];
        if (ki < 2) continue;
        double triangles = 0.0;
        for (int j = 0; j < n_nodes; ++j)
            for (int k = 0; k < n_nodes; ++k)
                if (j != i && k != i && j != k)
                    triangles += adj[i][j] * adj[j][k] * adj[k][i];
        total_clustering += triangles / (ki * (ki - 1));
    }
    f.mean_clustering = total_clustering / n_nodes;

    // PageRank (power iteration, 20 steps)
    {
        std::vector<double> pr(n_nodes, 1.0/n_nodes);
        std::vector<double> pr_next(n_nodes);
        double damping = 0.85;
        for (int iter = 0; iter < 20; ++iter) {
            double sum_pr = 0.0;
            for (int i = 0; i < n_nodes; ++i) {
                pr_next[i] = (1.0 - damping) / n_nodes;
                for (int j = 0; j < n_nodes; ++j) {
                    if (degree[j] > kEps)
                        pr_next[i] += damping * adj[j][i] / degree[j] * pr[j];
                }
                sum_pr += pr_next[i];
            }
            for (int i = 0; i < n_nodes; ++i) pr[i] = pr_next[i] / sum_pr;
        }
        double sum_pr = 0.0, max_pr = 0.0;
        for (double v : pr) { sum_pr += v; max_pr = std::max(max_pr, v); }
        f.mean_pagerank = sum_pr / n_nodes;
        f.max_pagerank  = max_pr;
    }

    // Modularity estimate (simple community detection by thresholding)
    {
        double threshold = f.mean_degree;
        std::vector<int> community(n_nodes, -1);
        int n_comm = 0;
        for (int i = 0; i < n_nodes; ++i) {
            if (community[i] >= 0) continue;
            community[i] = n_comm++;
            for (int j = i+1; j < n_nodes; ++j) {
                if (adj[i][j] > threshold) community[j] = community[i];
            }
        }
        f.n_communities = (double)n_comm;
        // Modularity Q
        double Q = 0.0;
        double two_m = 2.0 * total_edges;
        if (two_m > kEps) {
            for (int i = 0; i < n_nodes; ++i)
                for (int j = 0; j < n_nodes; ++j)
                    if (community[i] == community[j])
                        Q += adj[i][j] - degree[i]*degree[j]/two_m;
            Q /= two_m;
        }
        f.modularity = Q;
    }

    // Spectral gap (approx from degree sequence)
    {
        double max_d = *std::max_element(degree.begin(), degree.end());
        double min_d = *std::min_element(degree.begin(), degree.end());
        f.spectral_gap = (max_d > kEps) ? (max_d - min_d) / max_d : 0.0;
    }

    return f;
}

// ---------------------------------------------------------------------------
// Combined feature vector assembler
// ---------------------------------------------------------------------------
class FeatureAssembler {
public:
    static constexpr int kLOBDim      = LOBFeatures::kDim;
    static constexpr int kTempDim     = TemporalLOBFeatures::kDim;
    static constexpr int kCrossDim    = CrossAssetFeatures::kDim;
    static constexpr int kVolDim      = VolSurfaceFeatures::kDim;
    static constexpr int kGraphDim    = GraphFeatures::kDim;
    static constexpr int kTotalDim    = kLOBDim + kTempDim + kCrossDim + kVolDim + kGraphDim;

    // Assemble into pre-allocated buffer
    static void assemble(double* out,
                          const LOBFeatures& lob,
                          const TemporalLOBFeatures& temp,
                          const CrossAssetFeatures& cross,
                          const VolSurfaceFeatures& vol,
                          const GraphFeatures& graph)
    {
        double* p = out;
        lob.to_array(p);   p += kLOBDim;
        temp.to_array(p);  p += kTempDim;
        cross.to_array(p); p += kCrossDim;
        vol.to_array(p);   p += kVolDim;
        graph.to_array(p);
    }

    static std::vector<std::string> feature_names() {
        std::vector<std::string> names;
        names.insert(names.end(), {
            "lob_bid_ask_imbalance", "lob_bid_ask_imbalance_l2",
            "lob_bid_ask_imbalance_l5", "lob_bid_ask_imbalance_l10",
            "lob_mid_price", "lob_micro_price", "lob_spread",
            "lob_spread_bps", "lob_log_spread",
            "lob_bid_depth_l1", "lob_bid_depth_l3", "lob_bid_depth_l5",
            "lob_ask_depth_l1", "lob_ask_depth_l3", "lob_ask_depth_l5",
            "lob_total_depth", "lob_bid_vwap_l5", "lob_ask_vwap_l5",
            "lob_vwap_spread", "lob_bid_slope", "lob_ask_slope",
            "lob_bid_convexity", "lob_ask_convexity",
            "lob_ob_pressure", "lob_kyle_lambda", "lob_pad0"
        });
        names.insert(names.end(), {
            "temp_ret_1", "temp_ret_5", "temp_ret_10", "temp_ret_20",
            "temp_vol_1", "temp_vol_5", "temp_vol_20",
            "temp_imbal_ma5", "temp_imbal_ma20",
            "temp_spread_ma5", "temp_spread_ma20",
            "temp_flow_5", "temp_flow_20",
            "temp_autocorr_1", "temp_autocorr_5"
        });
        names.insert(names.end(), {
            "cross_corr_mean", "cross_corr_disp", "cross_lead_lag",
            "cross_sector_disp", "cross_factor1", "cross_factor2",
            "cross_beta_idx", "cross_resid_vol"
        });
        names.insert(names.end(), {
            "vol_atm", "vol_term_slope", "vol_smile_skew",
            "vol_smile_curv", "vol_of_vol", "vol_iv_spread_25d",
            "vol_rr_10d", "vol_bf_25d", "vol_calendar_spread",
            "vol_surface_norm"
        });
        names.insert(names.end(), {
            "graph_density", "graph_mean_degree", "graph_std_degree",
            "graph_clustering", "graph_spectral_gap", "graph_modularity",
            "graph_mean_pr", "graph_max_pr", "graph_betweenness",
            "graph_n_communities"
        });
        return names;
    }
};

// ---------------------------------------------------------------------------
// FeatureNormalizer — online z-score normalization
// ---------------------------------------------------------------------------
class FeatureNormalizer {
    int dim_;
    std::vector<WelfordStats> stats_;

public:
    explicit FeatureNormalizer(int dim) : dim_(dim), stats_(dim) {}

    void update(const double* x) {
        for (int i = 0; i < dim_; ++i) stats_[i].update(x[i]);
    }

    void normalize(const double* in, double* out) const {
        for (int i = 0; i < dim_; ++i)
            out[i] = stats_[i].z_score(in[i]);
    }

    void normalize_inplace(double* x) const {
        for (int i = 0; i < dim_; ++i)
            x[i] = stats_[i].z_score(x[i]);
    }

    void reset() { for (auto& s : stats_) s.reset(); }
};

// ---------------------------------------------------------------------------
// RollingFeatureMatrix — stores history for ML training
// ---------------------------------------------------------------------------
class RollingFeatureMatrix {
    int dim_;
    int capacity_;
    std::vector<double> buf_;   // [capacity × dim], row-major
    int head_ = 0;
    int size_ = 0;

public:
    RollingFeatureMatrix(int dim, int capacity)
        : dim_(dim), capacity_(capacity), buf_(dim * capacity, 0.0) {}

    void push(const double* row) {
        std::copy(row, row + dim_, buf_.data() + head_ * dim_);
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) ++size_;
    }

    // Get row i from newest (0) to oldest (size_-1)
    const double* row(int i) const {
        int idx = (head_ - 1 - i + capacity_) % capacity_;
        return buf_.data() + idx * dim_;
    }

    int size()     const { return size_; }
    int dim()      const { return dim_; }
    bool is_full() const { return size_ == capacity_; }

    // Pack into output tensor [seq_len × dim]
    void to_tensor(double* out, int seq_len) const {
        int actual = std::min(seq_len, size_);
        // Write newest-first into out[0..actual-1]
        for (int i = 0; i < actual; ++i) {
            std::copy(row(i), row(i) + dim_, out + i * dim_);
        }
        // Zero-pad if needed
        for (int i = actual; i < seq_len; ++i) {
            std::fill(out + i * dim_, out + (i+1) * dim_, 0.0);
        }
    }
};

// ---------------------------------------------------------------------------
// FeatureEngine — top-level per-asset feature computation engine
// ---------------------------------------------------------------------------
class FeatureEngine {
    static constexpr int kSeqLen = 32;
    static constexpr int kFeatDim = FeatureAssembler::kTotalDim;

    LOBFeatureExtractor       lob_extractor_;
    TemporalFeatureExtractor  temporal_extractor_;
    VolSurfaceFeatureExtractor vol_extractor_;
    FeatureNormalizer         normalizer_{kFeatDim};
    RollingFeatureMatrix      history_{kFeatDim, kSeqLen * 4};

    double last_mid_  = 0.0;
    double last_flow_ = 0.0;

public:
    // Update with new LOB state
    void update_lob(const std::vector<LOBFeatureExtractor::Level>& bids,
                    const std::vector<LOBFeatureExtractor::Level>& asks,
                    double trade_flow = 0.0)
    {
        auto lob  = lob_extractor_.extract(bids, asks);
        temporal_extractor_.update(lob.mid_price, lob.bid_ask_imbalance,
                                   lob.spread, trade_flow);
        last_mid_  = lob.mid_price;
        last_flow_ = trade_flow;

        // Assemble cross-asset and vol placeholders (zeros if not available)
        CrossAssetFeatures  cross{};
        VolSurfaceFeatures  vol{};
        GraphFeatures       graph{};

        auto temp = temporal_extractor_.extract();

        double feat[kFeatDim];
        FeatureAssembler::assemble(feat, lob, temp, cross, vol, graph);
        normalizer_.update(feat);
        history_.push(feat);
    }

    // Get latest normalized feature vector
    std::vector<double> get_features() const {
        if (history_.size() == 0) return std::vector<double>(kFeatDim, 0.0);
        std::vector<double> out(kFeatDim);
        std::copy(history_.row(0), history_.row(0) + kFeatDim, out.data());
        return out;
    }

    // Get sequence tensor [seq_len × feat_dim]
    std::vector<double> get_sequence(int seq_len = kSeqLen) const {
        std::vector<double> out(seq_len * kFeatDim, 0.0);
        history_.to_tensor(out.data(), seq_len);
        return out;
    }

    int dim()  const { return kFeatDim; }
    int seq()  const { return kSeqLen; }
    int n_obs() const { return history_.size(); }
};

// ---------------------------------------------------------------------------
// MultiAssetFeatureEngine
// ---------------------------------------------------------------------------
class MultiAssetFeatureEngine {
    int n_assets_;
    std::vector<FeatureEngine> engines_;
    EWMACorrelationMatrix corr_matrix_;

    // Per-asset mid prices for cross-asset features
    std::vector<double> current_mids_;

public:
    explicit MultiAssetFeatureEngine(int n_assets)
        : n_assets_(n_assets),
          engines_(n_assets),
          corr_matrix_(n_assets, 0.03),
          current_mids_(n_assets, 0.0)
    {}

    void update_asset(int asset_id,
                      const std::vector<LOBFeatureExtractor::Level>& bids,
                      const std::vector<LOBFeatureExtractor::Level>& asks,
                      double trade_flow = 0.0)
    {
        if (asset_id < 0 || asset_id >= n_assets_) return;
        engines_[asset_id].update_lob(bids, asks, trade_flow);
        if (!bids.empty() && !asks.empty())
            current_mids_[asset_id] = 0.5 * (bids[0].price + asks[0].price);
        corr_matrix_.update(current_mids_);
    }

    std::vector<double> get_features(int asset_id) const {
        if (asset_id < 0 || asset_id >= n_assets_) return {};
        return engines_[asset_id].get_features();
    }

    auto correlation_stats() const { return corr_matrix_.correlation_stats(); }

    int n_assets() const { return n_assets_; }
};

// ---------------------------------------------------------------------------
// Feature importance tracker (online, using gradient sign correlation)
// ---------------------------------------------------------------------------
class FeatureImportanceTracker {
    int dim_;
    std::vector<double> importance_;
    std::vector<double> grad_sign_corr_;
    int n_updates_ = 0;

public:
    explicit FeatureImportanceTracker(int dim)
        : dim_(dim), importance_(dim, 1.0/dim),
          grad_sign_corr_(dim, 0.0) {}

    // Update using realized PnL and feature vector
    void update(const double* features, double pnl) {
        double sign_pnl = (pnl > 0) ? 1.0 : (pnl < 0 ? -1.0 : 0.0);
        for (int i = 0; i < dim_; ++i) {
            double sign_f = (features[i] > 0) ? 1.0 : (features[i] < 0 ? -1.0 : 0.0);
            grad_sign_corr_[i] = 0.95 * grad_sign_corr_[i] + 0.05 * sign_f * sign_pnl;
        }
        // Normalize
        double sum = 0.0;
        for (int i = 0; i < dim_; ++i) sum += std::abs(grad_sign_corr_[i]);
        if (sum > kEps)
            for (int i = 0; i < dim_; ++i)
                importance_[i] = std::abs(grad_sign_corr_[i]) / sum;
        ++n_updates_;
    }

    const std::vector<double>& importance() const { return importance_; }

    // Top-k feature indices by importance
    std::vector<int> top_k(int k) const {
        std::vector<int> idx(dim_);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + std::min(k, dim_),
                          idx.end(),
                          [this](int a, int b) {
                              return importance_[a] > importance_[b];
                          });
        idx.resize(std::min(k, dim_));
        return idx;
    }
};

// ---------------------------------------------------------------------------
// LabelGenerator — generates ML labels from future returns
// ---------------------------------------------------------------------------
class LabelGenerator {
    int horizon_;
    CircularBuffer<double, 64> price_buf_;

public:
    explicit LabelGenerator(int horizon = 5) : horizon_(horizon) {}

    void push_price(double price) { price_buf_.push(price); }

    // Returns label if horizon steps of history available
    std::optional<double> get_label(double threshold_bps = 5.0) const {
        if (price_buf_.len() <= horizon_) return std::nullopt;
        double p_now  = price_buf_.at(0);
        double p_then = price_buf_.at(horizon_);
        if (p_then < kEps) return std::nullopt;
        double ret_bps = (p_now - p_then) / p_then * 1e4;
        if      (ret_bps >  threshold_bps) return  1.0;
        else if (ret_bps < -threshold_bps) return -1.0;
        else                               return  0.0;
    }

    double get_continuous_label() const {
        if (price_buf_.len() <= horizon_) return 0.0;
        double p_now  = price_buf_.at(0);
        double p_then = price_buf_.at(horizon_);
        return (p_then > kEps) ? (p_now - p_then) / p_then : 0.0;
    }
};

}  // namespace features
}  // namespace rtel
